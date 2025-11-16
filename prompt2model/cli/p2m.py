"""CLI wrapper around the prompt2model pipeline.

This module adds a non-interactive entrypoint that:

1. 读取外部 YAML：白名单数据集 / 任务规则 / 生成配置（通过环境变量指定路径）。
2. 运行一次 Task Expansion 并缓存 expanded_spec。
3. 使用 FixedDatasetLoader 直接加载白名单数据集（跳过远程检索 + 列选择）。
4. 调用 TransformExecutor 做计划-执行式转换 + 自一致性评审。
5. 统一去重、落盘 HF Dataset + JSONL/CSV，并做聚类可视化。

如果未设置 ``P2M_FIXED_CONFIG``，则回退到根目录交互式 ``p2m.py`` 流水线。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import datasets
import yaml

from prompt2model.dataset_retriever.fixed_loader import FixedDatasetLoader
from prompt2model.task_expansion import TaskExpansionEngine
from prompt2model.transform import TransformExecutor
from prompt2model.utils.logging_utils import get_formatted_logger

logger = get_formatted_logger("p2m_cli")

# ===== 默认后端与 YAML 配置（环境变量控制）=====
# - P2M_FIXED_CONFIG   : 白名单数据集配置（必填，否则回退到交互式 p2m.py）
# - P2M_TASK_RULES     : 任务规则 YAML（缺省为 config.task_rules.yaml）
# - P2M_GEN_CONFIG     : 生成 & 自一致性 YAML（缺省为 config.generation.yaml）
#
# 示例：
#   export P2M_FIXED_CONFIG=examples/config.fixed_datasets.yaml
#   export P2M_TASK_RULES=examples/config.task_rules.yaml
#   export P2M_GEN_CONFIG=examples/config.generation.yaml
#   python -m prompt2model.cli.p2m

os.environ.setdefault("P2M_BACKEND", "ollama")

HF_ENDPOINT="https://hf-mirror.com"
HF_DATASETS_TRUST_REMOTE_CODE=True
FIXED_CONFIG_ENV = os.getenv("P2M_FIXED_CONFIG") or "config.fixed_datasets.yaml"
TASK_RULES_ENV = os.getenv("P2M_TASK_RULES") or "config.task_rules.yaml"
GEN_CONFIG_ENV = os.getenv("P2M_GEN_CONFIG") or "config.generation.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping.")
    return data


def _load_prompt_examples(task_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load few-shot examples for pure synthetic generation.

    Priority:
    1) task.examples in task_rules YAML (config.task_rules.yaml);
    2) examples field from prompt YAML (P2M_PROMPT_FILE or prompt.yaml);
    3) otherwise, return an empty list (no few-shot guidance).
    """
    # 1) Prefer explicit examples under task config, if present.
    task_block = task_cfg.get("task", {}) or {}
    examples = task_block.get("examples")
    if isinstance(examples, list) and examples:
        return examples

    # 2) Fallback: load examples from prompt.yaml-style file.
    prompt_file = os.getenv("P2M_PROMPT_FILE") or "prompt.yaml"
    path = Path(prompt_file)
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - IO robustness
        logger.warning("Failed to load prompt file %s: %s", path, exc)
        return []

    exs = meta.get("examples", [])
    if not isinstance(exs, list):
        return []

    # Keep as-is; downstream generator is robust to dict/str formats.
    return exs


def _ensure_run_dir(output_dir: Path, run_id: str | None = None) -> Path:
    if run_id is None:
        from time import strftime

        run_id = strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _dedup_dataset(ds: datasets.Dataset) -> datasets.Dataset:
    """Reuse the existing post-dedup logic from the interactive script."""
    from p2m import _post_dedup_dataset  # type: ignore[import]

    return _post_dedup_dataset(ds)


def _make_spec_cache_id(spec: Dict[str, Any], idx: int) -> str:
    """Build a stable, filesystem-safe id for a fixed dataset spec."""
    dataset_name = str(spec.get("dataset_name", f"ds{idx}"))
    config_name = str(spec.get("config_name", "default"))
    split = str(spec.get("split", "train"))
    base = f"{idx}_{dataset_name}_{config_name}_{split}"
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in base)
    return safe


def _export_sidecars(ds: datasets.Dataset, out_stem: Path) -> None:
    from p2m import _export_sidecars as _export  # type: ignore[import]

    _export(ds, out_stem)


def _visualize_dataset(run_dir: Path, ds: datasets.Dataset) -> None:
    """Generate embeddings + cluster figure for a single dataset."""
    if len(ds) == 0:
        logger.info("No data to visualize; skipping visualization.")
        return

    # Use the same helpers as the original script for consistency.
    from p2m import _embed_with_ollama, _reduce_and_plot  # type: ignore[import]

    import pandas as pd

    df = ds.to_pandas().copy()
    if "input" not in df.columns or "output" not in df.columns:
        logger.warning("Dataset missing input/output columns; skipping visualization.")
        return
    if "label" not in df.columns:
        df["label"] = "retrieved"
    if "source" not in df.columns:
        df["source"] = "whitelist"

    df["__text__"] = df["input"].astype(str) + " || " + df["output"].astype(str)

    model = os.getenv("P2M_OLLAMA_EMBED_MODEL", "nomic-embed-text")
    emb = _embed_with_ollama(df["__text__"].values, model=model)

    import numpy as np

    emb_path = run_dir / "embeddings.npy"
    np.save(emb_path, emb)
    df.to_csv(run_dir / "mix_meta.csv", index=False, encoding="utf-8")

    dr_method = os.getenv("P2M_DR_METHOD", "umap")
    out_path = run_dir / os.getenv("P2M_OUT", "synth_clusters.png")
    _reduce_and_plot(emb, df, color_col="label", method=dr_method, out_path=str(out_path))
    logger.info("Cluster figure saved to: %s", out_path)


def run_fixed_pipeline(
    fixed_config: Path,
    task_rules: Path,
    gen_config: Path,
    output_dir: Path,
    run_id: str | None = None,
) -> None:
    """Run the fixed-dataset + transform pipeline."""
    logger.info("=== [p2m CLI] Fixed dataset pipeline START ===")
    fixed_cfg = _load_yaml(fixed_config)
    task_cfg = _load_yaml(task_rules)
    gen_cfg = _load_yaml(gen_config)

    # Optional few-shot guidance for pure synthetic generation.
    prompt_examples: List[Dict[str, Any]] = _load_prompt_examples(task_cfg)

    # Optional limits on how many examples to use.
    limits_cfg: Dict[str, Any] = dict(gen_cfg.get("limits", {}) or {})
    try:
        max_ret_per_ds = int(limits_cfg.get("max_retrieved_per_dataset", -1))
    except (TypeError, ValueError):
        max_ret_per_ds = -1
    try:
        synthetic_num_examples = int(limits_cfg.get("synthetic_num_examples", 0))
    except (TypeError, ValueError):
        synthetic_num_examples = 0

    datasets_specs: List[Dict[str, Any]] = list(fixed_cfg.get("datasets", []) or [])
    if not datasets_specs:
        raise ValueError(f"No datasets specified in {fixed_config}")

    # Configure models from gen_config.
    llm_backend = gen_cfg.get("llm_backend", {})
    gen_model = llm_backend.get("gen_model")
    judge_model = llm_backend.get("judge_model", gen_model)
    if gen_model:
        os.environ["P2M_GEN_MODEL"] = gen_model

    run_dir = _ensure_run_dir(output_dir, run_id)
    logger.info(
        "Run directory: %s | fixed=%s | task_rules=%s | gen_config=%s",
        run_dir,
        fixed_config,
        task_rules,
        gen_config,
    )
    logger.info(
        "Total fixed dataset specs: %d | max_ret_per_ds=%d | synthetic_num_examples=%d",
        len(datasets_specs),
        max_ret_per_ds,
        synthetic_num_examples,
    )

    # 1) Task expansion (cached).
    expander = TaskExpansionEngine(task_cfg, run_dir=run_dir)
    expanded_spec = expander.run()

    # 2) Load whitelist datasets and transform per spec.
    loader = FixedDatasetLoader()
    all_transformed: List[datasets.Dataset] = []

    for idx, spec in enumerate(datasets_specs):
        cache_id = _make_spec_cache_id(spec, idx)
        cache_dir = run_dir / f"transformed_{cache_id}"

        # Load DatasetDict with a single spec to keep postprocess mapping simple.
        if cache_dir.exists():
            logger.info(
                "------ Found cached transformed dataset for %s at %s, loading.",
                spec.get("dataset_name"),
                cache_dir,
            )
            transformed = datasets.load_from_disk(str(cache_dir))
        else:
            logger.info(
                "------ Processing fixed dataset: %s (config=%s, split=%s)",
                spec.get("dataset_name"),
                spec.get("config_name"),
                spec.get("split", "train"),
            )
            ddict = loader.load_all([spec])
            train = ddict["train"]
            if len(train) == 0:
                logger.warning(
                    "Fixed dataset %s produced no rows after filtering; skipping.",
                    spec.get("dataset_name"),
                )
                continue

            # Optional cap per dataset after filtering.
            if max_ret_per_ds == 0:
                logger.info(
                    "Skipping dataset %s because limits.max_retrieved_per_dataset=0",
                    spec.get("dataset_name"),
                )
                continue
            if max_ret_per_ds > 0 and len(train) > max_ret_per_ds:
                logger.info(
                    "Limiting dataset %s from %d to %d rows (limits.max_retrieved_per_dataset).",
                    spec.get("dataset_name"),
                    len(train),
                    max_ret_per_ds,
                )
                # Stable prefix selection; 如需随机子集可后续改为 shuffle 再 select.
                train = train.select(range(max_ret_per_ds))

            logger.info(
                "Loaded fixed dataset %s with %d rows.",
                spec.get("dataset_name"),
                len(train),
            )
            logger.info("Starting TransformExecutor for dataset %s", spec.get("dataset_name"))
            executor = TransformExecutor(
                expanded_spec=expanded_spec,
                gen_cfg=gen_cfg,
                run_dir=run_dir,
            )
            transformed = executor.transform_dataset(
                train,
                mapping=spec,
                postprocess_cfg=spec.get("postprocess"),
            )
            try:
                transformed.save_to_disk(str(cache_dir))
                logger.info("Saved transformed dataset cache to %s", cache_dir)
            except Exception as exc:  # pragma: no cover - IO robustness
                logger.warning("Failed to save transformed dataset cache to %s: %s", cache_dir, exc)

        all_transformed.append(transformed)

    if not all_transformed:
        logger.warning("No transformed data produced; exiting.")
        return

    if len(all_transformed) == 1:
        transformed_ds = all_transformed[0]
    else:
        transformed_ds = datasets.concatenate_datasets(all_transformed)

    # 3) 纯合成数据（可选）
    synthetic_ds = None
    if synthetic_num_examples > 0:
        synth_cache_dir = run_dir / "synthetic_dataset"
        if synth_cache_dir.exists():
            logger.info("Found cached synthetic dataset at %s, loading.", synth_cache_dir)
            try:
                synthetic_ds = datasets.load_from_disk(str(synth_cache_dir))
            except Exception as exc:  # pragma: no cover - IO robustness
                logger.warning("Failed to load cached synthetic dataset %s: %s", synth_cache_dir, exc)
                synthetic_ds = None

        if synthetic_ds is None:
            logger.info("Starting pure synthetic generation: %d examples", synthetic_num_examples)
            try:
                from prompt2model.dataset_generator.base import DatasetSplit
                from prompt2model.dataset_generator.prompt_based import PromptBasedDatasetGenerator
                from prompt2model.prompt_parser import MockPromptSpec, TaskType

                task_block = task_cfg.get("task", {}) or {}
                base_instruction = str(task_block.get("base_instruction", "")).strip()
                if not base_instruction:
                    base_instruction = (
                        "根据用户问题生成一个合规、安全、简明的回答，"
                        "输入为 Question: ...，输出仅包含最终答复。"
                    )
                # Few-shot examples:
                # - Prefer task.examples in task_rules YAML;
                # - Then fallback to prompt.yaml examples (if present);
                # - If both missing, run zero-shot as before.
                examples = prompt_examples if isinstance(prompt_examples, list) else None
                prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION, base_instruction, examples)

                llm_backend_cfg = gen_cfg.get("llm_backend", {}) or {}
                temp_cfg = llm_backend_cfg.get("temperature", {}) or {}
                initial_temperature = float(temp_cfg.get("init", 0.4))
                max_temperature = float(temp_cfg.get("max", 1.0))
                presence_penalty = float(llm_backend_cfg.get("presence_penalty", 0.0))
                frequency_penalty = float(llm_backend_cfg.get("frequency_penalty", 0.0))
                responses_per_request = int(llm_backend_cfg.get("responses_per_request", 1))
                max_batch_size = int(llm_backend_cfg.get("max_batch_size", 3))
                requests_per_minute = int(llm_backend_cfg.get("requests_per_minute", 80))

                gen = PromptBasedDatasetGenerator(
                    initial_temperature=initial_temperature,
                    max_temperature=max_temperature,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    max_batch_size=max_batch_size,
                    responses_per_request=responses_per_request,
                    requests_per_minute=requests_per_minute,
                    filter_duplicated_examples=True,
                )
                synth_raw = gen.generate_dataset_split(
                    prompt_spec, synthetic_num_examples, split=DatasetSplit.TRAIN
                )
                # 统一列名以便后续去重/可视化
                synthetic_ds = datasets.Dataset.from_dict(
                    {
                        "input": list(synth_raw["input_col"]),
                        "output": list(synth_raw["output_col"]),
                        "label": ["synthetic"] * len(synth_raw),
                        "source": ["synthetic"] * len(synth_raw),
                    }
                )
                try:
                    synthetic_ds.save_to_disk(str(synth_cache_dir))
                    logger.info("Saved synthetic dataset cache to %s", synth_cache_dir)
                except Exception as exc:  # pragma: no cover - IO robustness
                    logger.warning("Failed to save synthetic dataset cache to %s: %s", synth_cache_dir, exc)
                logger.info("Pure synthetic generation finished: %d examples", len(synthetic_ds))
            except Exception as exc:  # pragma: no cover - generation is best-effort
                logger.warning("Synthetic generation failed, skipping synthetic step: %s", exc)
                synthetic_ds = None

    # 4) 合并转换数据 + 合成数据，再做去重与导出。
    if synthetic_ds is not None:
        final_ds = datasets.concatenate_datasets([transformed_ds, synthetic_ds])
    else:
        final_ds = transformed_ds

    final_ds = _dedup_dataset(final_ds)

    gen_root = run_dir / "generated_dataset"
    final_ds.save_to_disk(str(gen_root))
    _export_sidecars(final_ds, gen_root / "generated_dataset")

    # Save manifest for reproducibility.
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "backend": os.getenv("P2M_BACKEND"),
        "gen_model": os.getenv("P2M_GEN_MODEL"),
        "judge_model": judge_model,
        "configs": {
            "fixed_config": str(fixed_config),
            "task_rules": str(task_rules),
            "gen_config": str(gen_config),
        },
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 4) Visualization (single dataset).
    _visualize_dataset(run_dir, final_ds)
    logger.info(
        "=== [p2m CLI] Pipeline DONE | examples=%d | run_dir=%s ===",
        len(final_ds),
        run_dir,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prompt2Model CLI with fixed datasets + transform pipeline.\n"
            "YAML 配置路径通过环境变量控制：P2M_FIXED_CONFIG / P2M_TASK_RULES / P2M_GEN_CONFIG。"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Output directory for run artifacts (default: runs).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID suffix (otherwise timestamp-based).",
    )

    args = parser.parse_args(argv)

    # 若未设置 P2M_FIXED_CONFIG，则回退到原始交互式脚本。
    if not FIXED_CONFIG_ENV:
        from p2m import main as interactive_main  # type: ignore[import]

        interactive_main()
        return

    fixed_config = Path(FIXED_CONFIG_ENV)
    task_rules = Path(TASK_RULES_ENV)
    gen_config = Path(GEN_CONFIG_ENV)
    output_dir = Path(args.output_dir)

    if not fixed_config.exists():
        raise FileNotFoundError(f"P2M_FIXED_CONFIG not found: {fixed_config}")
    if not task_rules.exists():
        raise FileNotFoundError(f"P2M_TASK_RULES not found: {task_rules}")
    if not gen_config.exists():
        raise FileNotFoundError(f"P2M_GEN_CONFIG not found: {gen_config}")

    run_fixed_pipeline(
        fixed_config=fixed_config,
        task_rules=task_rules,
        gen_config=gen_config,
        output_dir=output_dir,
        run_id=args.run_id,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
