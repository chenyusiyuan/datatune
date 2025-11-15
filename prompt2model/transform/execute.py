"""TransformExecutor: plan/execute-style conversion with self-consistency.

This module consumes:
- an expanded task spec (from :mod:`prompt2model.task_expansion.expander`)
- generation config (temperature, self-consistency, filters)
- a raw dataset with ``input_src`` / ``output_src`` / label / source

For each row it:
1. Builds a transform prompt and samples k candidate (input, output) pairs.
2. Runs an optional judge model to pick the best candidate.
3. Applies language/length/sensitive-word/novelty filters.
4. Emits a final HF Dataset with standardized columns: input/output/label/source.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import datasets
from tqdm import tqdm

from prompt2model.utils import APIAgent, get_formatted_logger
import os
from prompt2model.utils.validation import (
    contains_sensitive,
    extract_judge_index,
    is_chinese_text,
    length_ok,
    parse_strict_io_json,
    passes_ngram_novelty,
)

logger = get_formatted_logger("TransformExecutor")


def _load_prompt_template(name: str) -> str:
    base = Path(__file__).resolve().parent.parent / "prompts"
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


@dataclass
class GenerationConfig:
    """Lightweight wrapper around generation-related configuration."""

    gen_model: str
    judge_model: str
    init_temp: float = 0.4
    max_temp: float = 1.0
    k: int = 5
    vote: str = "majority_llm_judge"
    max_retries: int = 1
    lang: str = "zh"
    min_len: int = 12
    max_len: int = 240
    rougeL_threshold: float = 0.9
    sensitive_keywords: List[str] | None = None
    requests_per_minute: int = 60

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "GenerationConfig":
        llm_backend = cfg.get("llm_backend", {})
        sc = cfg.get("self_consistency", {})
        filters = cfg.get("filters", {})
        temp_cfg = llm_backend.get("temperature", {})
        output_len = filters.get("output_len", {})
        novelty = filters.get("ngram_novelty", {})

        return cls(
            gen_model=llm_backend.get("gen_model", "llama3.1:8b"),
            judge_model=llm_backend.get("judge_model", llm_backend.get("gen_model", "llama3.1:8b")),
            init_temp=float(temp_cfg.get("init", 0.4)),
            max_temp=float(temp_cfg.get("max", 1.0)),
            k=int(sc.get("k", 5)),
            vote=str(sc.get("vote", "majority_llm_judge")),
            max_retries=int(sc.get("max_retries", 1)),
            lang=str(filters.get("lang", "zh")),
            min_len=int(output_len.get("min", 12)),
            max_len=int(output_len.get("max", 240)),
            rougeL_threshold=float(novelty.get("rougeL_threshold", 0.9)),
            sensitive_keywords=list(filters.get("sensitive_keywords", [])),
            requests_per_minute=int(cfg.get("requests_per_minute", 60)),
        )


class TransformExecutor:
    """Execute row-wise transformation with self-consistency voting."""

    def __init__(
        self,
        expanded_spec: Dict[str, Any],
        gen_cfg: Dict[str, Any],
        llm_gen: Optional[APIAgent] = None,
        llm_judge: Optional[APIAgent] = None,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.expanded_spec = expanded_spec
        self.gen_cfg = GenerationConfig.from_dict(gen_cfg)
        self.run_dir = run_dir
        self.logger = logger

        # --- Multi-GPU / multi-Ollama 调度 ---
        # 若设置 P2M_OLLAMA_BASES=http://host1:11434,http://host2:11435
        # 则针对生成模型创建多路 APIAgent，在每次候选生成时并行调用。
        bases_env = os.getenv("P2M_OLLAMA_BASES", "").strip()
        self._gen_agents: List[APIAgent] = []
        if bases_env:
            bases = [b.strip() for b in bases_env.split(",") if b.strip()]
            for base in bases:
                self._gen_agents.append(
                    APIAgent(model_name=self.gen_cfg.gen_model, ollama_base=base)
                )
            if not self._gen_agents:
                self._gen_agents.append(APIAgent(model_name=self.gen_cfg.gen_model))
        else:
            self._gen_agents.append(llm_gen or APIAgent(model_name=self.gen_cfg.gen_model))

        self.llm_gen = self._gen_agents[0]
        # 评审模型通常较轻，保持单路即可；如有需要，可按同样方式扩展。
        self.llm_judge = llm_judge or APIAgent(model_name=self.gen_cfg.judge_model)

        self._generate_template = _load_prompt_template("transform_generate.txt")
        self._judge_template = _load_prompt_template("transform_judge.txt")

        self._accepted_outputs: List[str] = []
        self._failures_path: Optional[Path] = None
        self._candidates_dir: Optional[Path] = None

        if self.run_dir is not None:
            self._failures_path = self.run_dir / "failures.jsonl"
            self._candidates_dir = self.run_dir / "candidates"
            self._candidates_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- Prompt construction ---------------------------
    def _build_generate_prompt(self, row: Dict[str, Any]) -> str:
        expanded_text = self.expanded_spec.get("raw") or str(self.expanded_spec)
        row_json = json.dumps(row, ensure_ascii=False)

        prompt = self._generate_template
        prompt = prompt.replace("{expanded_spec}", expanded_text)
        prompt = prompt.replace("{row_json}", row_json)
        return prompt

    def _build_judge_prompt(
        self,
        the_input: str,
        candidates: Sequence[Dict[str, str]],
    ) -> str:
        expanded_text = self.expanded_spec.get("raw") or str(self.expanded_spec)
        checklist = self.expanded_spec.get("checklist", expanded_text)
        lines = []
        for idx, cand in enumerate(candidates):
            lines.append(f"{idx}) {cand.get('output', '').strip()}")
        candidates_block = "\n".join(lines)

        prompt = self._judge_template
        prompt = prompt.replace("{expanded_spec}", expanded_text)
        prompt = prompt.replace("{checklist}", str(checklist))
        prompt = prompt.replace("{the_input}", the_input)
        prompt = prompt.replace("{candidates_block}", candidates_block)
        return prompt

    # ----------------------------- LLM helpers -----------------------------
    async def _generate_batch_async(self, prompts: List[str], temperature: float) -> List[Any]:
        """Call generator in batch using one或多路 APIAgent."""
        if len(self._gen_agents) == 1:
            return await self._gen_agents[0].generate_batch_completion(
                prompts,
                temperature=temperature,
                responses_per_request=1,
                requests_per_minute=self.gen_cfg.requests_per_minute,
                show_progress=False,
            )

        # 多 GPU：将 prompts 轮询分配到不同 agent，并行发送请求。
        num_agents = len(self._gen_agents)
        chunks: List[List[str]] = [[] for _ in range(num_agents)]
        # 记录每个全局索引对应的 (agent_idx, local_idx)，以便恢复原始顺序。
        index_map: List[tuple[int, int]] = []
        for i, p in enumerate(prompts):
            agent_idx = i % num_agents
            local_idx = len(chunks[agent_idx])
            chunks[agent_idx].append(p)
            index_map.append((agent_idx, local_idx))

        tasks = []
        for agent_idx, agent in enumerate(self._gen_agents):
            chunk = chunks[agent_idx]
            if not chunk:
                tasks.append(None)
                continue
            tasks.append(
                agent.generate_batch_completion(
                    chunk,
                    temperature=temperature,
                    responses_per_request=1,
                    requests_per_minute=self.gen_cfg.requests_per_minute,
                    show_progress=False,
                )
            )

        # 为了简单起见，对 None 过滤后 gather，再按 index_map 还原。
        coro_list = [t for t in tasks if t is not None]
        results_per_agent: List[List[Any]] = []
        if coro_list:
            gathered = await asyncio.gather(*coro_list)
            # 将 gathered 重新映射回 agent_idx 顺序（有些 agent 可能 chunk 为空）
            j = 0
            for agent_idx, chunk in enumerate(chunks):
                if not chunk:
                    results_per_agent.append([])
                else:
                    results_per_agent.append(gathered[j])
                    j += 1

        # 依据 index_map 拼回原始 prompts 顺序
        merged: List[Any] = [None] * len(prompts)
        for global_idx, (agent_idx, local_idx) in enumerate(index_map):
            merged[global_idx] = results_per_agent[agent_idx][local_idx]
        return merged

    def _generate_candidates_for_row(
        self,
        row: Dict[str, Any],
        attempt_idx: int,
    ) -> List[Dict[str, str]]:
        """Generate k candidate (input, output) pairs for a single row."""
        k = max(1, self.gen_cfg.k)
        base_prompt = self._build_generate_prompt(row)
        prompts = [base_prompt] * k

        # Simple temperature schedule between init and max.
        t0 = self.gen_cfg.init_temp
        t1 = self.gen_cfg.max_temp
        frac = min(1.0, max(0.0, attempt_idx / max(1, self.gen_cfg.max_retries)))
        temperature = t0 + (t1 - t0) * frac

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:  # pragma: no cover - when no loop exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            responses = loop.run_until_complete(self._generate_batch_async(prompts, temperature))
        except Exception as exc:  # pragma: no cover - transport errors
            self.logger.error("Batch generation failed: %s", exc)
            return []

        candidates: List[Dict[str, str]] = []
        for resp in responses:
            parsed = parse_strict_io_json(resp)
            if parsed is not None:
                candidates.append(parsed)
        return candidates

    def _judge_candidates(
        self,
        the_input: str,
        candidates: Sequence[Dict[str, str]],
    ) -> int:
        """Select the best candidate index, or -1 if all are rejected."""
        if not candidates:
            return -1

        if self.gen_cfg.vote == "rule_score":
            # Simple heuristic: prefer Chinese, valid length, then shortest output.
            scored: List[tuple[float, int]] = []
            for idx, cand in enumerate(candidates):
                out = cand.get("output", "")
                score = 0.0
                if is_chinese_text(out):
                    score += 1.0
                if length_ok(out, self.gen_cfg.min_len, self.gen_cfg.max_len):
                    score += 0.5
                # Slight preference for shorter answers.
                score -= len(out) / 1000.0
                scored.append((score, idx))
            scored.sort(reverse=True)
            return scored[0][1]

        # Default: majority_llm_judge (single-call judge that returns an index).
        prompt = self._build_judge_prompt(the_input, candidates)
        resp = self.llm_judge.generate_one_completion(
            prompt=prompt,
            temperature=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        idx = extract_judge_index(resp)
        if 0 <= idx < len(candidates):
            return idx
        return -1

    # ----------------------------- Filtering -----------------------------
    def _passes_filters(self, cand: Dict[str, str]) -> bool:
        out = cand.get("output", "")
        if not is_chinese_text(out) and self.gen_cfg.lang == "zh":
            return False
        if not length_ok(out, self.gen_cfg.min_len, self.gen_cfg.max_len):
            return False
        if self.gen_cfg.sensitive_keywords:
            if contains_sensitive(out, self.gen_cfg.sensitive_keywords):
                return False
        if not passes_ngram_novelty(out, self._accepted_outputs, self.gen_cfg.rougeL_threshold):
            return False
        return True

    def _record_failure(
        self,
        row: Dict[str, Any],
        reason: str,
        last_candidates: Optional[Sequence[Dict[str, str]]] = None,
    ) -> None:
        if self._failures_path is None:
            return
        payload = {
            "row": row,
            "reason": reason,
        }
        if last_candidates is not None:
            payload["candidates"] = last_candidates
        try:
            with self._failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - IO robustness
            self.logger.warning("Failed to write failure record: %s", exc)

    def _record_candidates(
        self,
        idx: int,
        row: Dict[str, Any],
        candidates: Sequence[Dict[str, str]],
        chosen_index: int,
    ) -> None:
        if self._candidates_dir is None:
            return
        payload = {
            "row_index": idx,
            "row": row,
            "candidates": candidates,
            "chosen_index": chosen_index,
        }
        path = self._candidates_dir / f"{idx:06d}.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to write candidates file %s: %s", path, exc)

    # ------------------------------ Public API ------------------------------
    def transform_dataset(
        self,
        ds: datasets.Dataset,
        mapping: Dict[str, Any] | None = None,
        postprocess_cfg: Dict[str, Any] | None = None,
    ) -> datasets.Dataset:
        """Transform a dataset into input/output pairs.

        Args:
            ds: Source HF dataset, expected to contain at least input_src/output_src.
            mapping: Optional mapping/config dict (kept for extensibility).
            postprocess_cfg: Postprocess config (currently unused here but kept
                for compatibility with the YAML spec).

        Returns:
            A HF Dataset with columns: input, output, label, source.
        """
        del mapping, postprocess_cfg  # currently unused

        records: List[Dict[str, Any]] = []
        num_failures = 0

        iterator = tqdm(
            enumerate(ds),
            total=len(ds),
            desc="Transforming dataset",
        )
        for idx, row in iterator:
            src_row = dict(row)
            attempts = 0
            chosen: Optional[Dict[str, str]] = None
            last_cands: List[Dict[str, str]] = []

            while attempts <= self.gen_cfg.max_retries:
                cands = self._generate_candidates_for_row(src_row, attempts)
                last_cands = cands
                if not cands:
                    attempts += 1
                    continue

                the_input = cands[0]["input"]
                best_idx = self._judge_candidates(the_input, cands)
                if best_idx < 0 or best_idx >= len(cands):
                    attempts += 1
                    continue

                cand = cands[best_idx]
                if not self._passes_filters(cand):
                    attempts += 1
                    continue

                chosen = cand
                self._record_candidates(idx, src_row, cands, best_idx)
                break

            if chosen is None:
                num_failures += 1
                self._record_failure(src_row, "no_valid_candidate", last_cands)
                continue

            out_text = chosen["output"]
            self._accepted_outputs.append(out_text)
            records.append(
                {
                    "input": chosen["input"],
                    "output": out_text,
                    "label": src_row.get("label", "retrieved"),
                    "source": src_row.get("source", "unknown"),
                }
            )

        self.logger.info(
            "TransformExecutor finished: %d success, %d failures",
            len(records),
            num_failures,
        )

        if not records:
            return datasets.Dataset.from_dict(
                {"input": [], "output": [], "label": [], "source": []}
            )

        return datasets.Dataset.from_dict(
            {
                "input": [r["input"] for r in records],
                "output": [r["output"] for r in records],
                "label": [r["label"] for r in records],
                "source": [r["source"] for r in records],
            }
        )


__all__ = ["TransformExecutor", "GenerationConfig"]
