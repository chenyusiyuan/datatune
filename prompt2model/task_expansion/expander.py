"""Task expansion engine.

This module turns a high-level task rules YAML into an expanded execution spec
using a single LLM call. The expanded spec is cached to ``expanded_spec.json``
under the current run directory for reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from prompt2model.utils import APIAgent, get_formatted_logger

logger = get_formatted_logger("TaskExpansionEngine")


def _load_prompt_template(name: str) -> str:
    """Load a prompt template from ``prompt2model/prompts``."""
    base = Path(__file__).resolve().parent.parent / "prompts"
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


class TaskExpansionEngine:
    """Generate an expanded execution spec from task rules."""

    def __init__(
        self,
        spec_yaml: Dict[str, Any],
        llm: Optional[APIAgent] = None,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.spec_yaml = spec_yaml
        self.llm = llm or APIAgent()
        self.run_dir = run_dir
        self.logger = logger
        self._template = _load_prompt_template("task_expansion.txt")

    @property
    def cache_path(self) -> Optional[Path]:
        if self.run_dir is None:
            return None
        return self.run_dir / "expanded_spec.json"

    def _build_prompt(self) -> str:
        task = self.spec_yaml.get("task", {})
        base_instruction = task.get("base_instruction", "")
        io_format = task.get("io_format", {})
        output_constraints = io_format.get("output_constraints", {})
        answer_policies = task.get("answer_policies", [])
        style = task.get("style", {})
        expansion_hints = task.get("expansion_hints", [])

        prompt = self._template
        prompt = prompt.replace("{task.base_instruction}", str(base_instruction))
        prompt = prompt.replace("{task.io_format.input_template}", str(io_format.get("input_template", "")))
        prompt = prompt.replace("{task.io_format.context_template}", str(io_format.get("context_template", "")))
        prompt = prompt.replace(
            "{task.io_format.output_constraints}", json.dumps(output_constraints, ensure_ascii=False, indent=2)
        )
        prompt = prompt.replace(
            "{task.answer_policies}", "\n".join(f"- {p}" for p in answer_policies)
        )
        prompt = prompt.replace("{task.style}", json.dumps(style, ensure_ascii=False, indent=2))
        prompt = prompt.replace(
            "{task.expansion_hints}", "\n".join(f"- {h}" for h in expansion_hints)
        )
        return prompt

    def _call_llm(self, prompt: str) -> str:
        resp = self.llm.generate_one_completion(
            prompt=prompt,
            temperature=0.3,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        # Avoid importing parse_responses just for this; APIAgent already returns
        # OpenAI-like dict, so normalize here.
        from prompt2model.utils.parse_responses import extract_content

        return extract_content(resp)

    def run(self, force: bool = False) -> Dict[str, Any]:
        """Run task expansion, optionally using a cached expanded_spec.json."""
        cache_path = self.cache_path
        if cache_path is not None and cache_path.exists() and not force:
            self.logger.info("Loading cached expanded_spec from %s", cache_path)
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        prompt = self._build_prompt()
        self.logger.info("Calling LLM for task expansion...")
        expanded_text = self._call_llm(prompt).strip()

        expanded_spec: Dict[str, Any] = {
            "raw": expanded_text,
            # For now we expose the same text as checklist; callers can
            # optionally parse it into structured rules if desired.
            "checklist": expanded_text,
        }

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(expanded_spec, f, ensure_ascii=False, indent=2)
                self.logger.info("Saved expanded_spec to %s", cache_path)
            except Exception as exc:  # pragma: no cover - IO robustness
                self.logger.warning("Failed to write expanded_spec cache: %s", exc)

        return expanded_spec


__all__ = ["TaskExpansionEngine"]

