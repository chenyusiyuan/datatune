"""Fixed whitelist-based dataset loader.

This bypasses the dual-encoder retriever and column-selection prompts by
directly loading a set of HuggingFace datasets specified in a YAML config,
then adding canonical columns:

- ``input_src``: raw input text column from the source dataset
- ``output_src``: raw output/label column (may be None when generation is needed)
- ``label``: tag for downstream visualization (defaults to ``\"retrieved\"``)
- ``source``: dataset identifier (usually the HF dataset name)

Each dataset entry in the config should look like:

.. code-block:: yaml

    datasets:
      - dataset_name: Hello-SimpleAI/HC3-Chinese
        config_name: default
        split: train
        input_col: question
        output_col: null
        filters:
          regex_any:
            - "银行|信用卡|理财"

The postprocess rules are consumed later by the TransformExecutor and are not
used in this loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import datasets

from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("FixedDatasetLoader")


@dataclass
class FixedDatasetSpec:
    """Specification for one fixed dataset entry."""

    dataset_name: str
    config_name: Optional[str] = None
    split: str = "train"
    input_col: str = "input"
    output_col: Optional[str] = None
    filters: Dict[str, Any] | None = None
    postprocess: Dict[str, Any] | None = None


class FixedDatasetLoader:
    """Load a list of fixed datasets and standardize their columns."""

    def __init__(self) -> None:
        self.logger = logger

    def _apply_filters(
        self,
        ds: datasets.Dataset,
        spec: FixedDatasetSpec,
    ) -> datasets.Dataset:
        """Apply simple row filters defined in the spec."""
        filters = spec.filters or {}
        if not filters:
            return ds

        regex_any: Iterable[str] = filters.get("regex_any", []) or []
        intent_in: Iterable[str] = filters.get("intent_in", []) or []
        language_in: Iterable[str] = filters.get("language_in", []) or []

        # Lazily import re only when needed.
        import re

        regex_patterns = [re.compile(p) for p in regex_any]
        intent_set = set(intent_in)
        lang_set = set(language_in)

        def _keep(example: Dict[str, Any]) -> bool:
            keep = True
            if regex_patterns:
                # Apply regex on the declared input column when available.
                text_val = str(example.get(spec.input_col, ""))
                keep = any(p.search(text_val) for p in regex_patterns)
                if not keep:
                    return False

            if intent_set:
                intent_val = example.get("intent")
                if intent_val not in intent_set:
                    return False

            if lang_set:
                lang_val = example.get("language") or example.get("lang") or example.get(
                    "locale"
                )
                if lang_val not in lang_set:
                    return False

            return keep

        filtered = ds.filter(_keep)
        self.logger.info(
            "Applied filters to %s: %d -> %d rows",
            spec.dataset_name,
            len(ds),
            len(filtered),
        )
        return filtered

    def _standardize_columns(
        self,
        ds: datasets.Dataset,
        spec: FixedDatasetSpec,
    ) -> datasets.Dataset:
        """Add canonical columns input_src/output_src/source.

        We intentionally do NOT keep or introduce a ``label`` column here to avoid
        schema mismatches when concatenating datasets with differing label types.
        """

        def _map_row(example: Dict[str, Any]) -> Dict[str, Any]:
            example["input_src"] = example.get(spec.input_col)
            if spec.output_col is None:
                example["output_src"] = None
            else:
                example["output_src"] = example.get(spec.output_col)
            # For downstream bookkeeping.
            example.setdefault("source", spec.dataset_name)
            # Drop any existing label to keep schemas consistent downstream.
            if "label" in example:
                example.pop("label", None)
            return example

        return ds.map(_map_row)

    def load_all(self, fixed_spec_list: List[Dict[str, Any]]) -> datasets.DatasetDict:
        """Load and combine all datasets defined by a list of specs.

        Args:
            fixed_spec_list: List of raw YAML spec dicts under ``datasets:``.

        Returns:
            DatasetDict with a single ``\"train\"`` split containing all rows.
        """
        if not fixed_spec_list:
            empty = datasets.Dataset.from_dict(
                {
                    "input_src": [],
                    "output_src": [],
                    "label": [],
                    "source": [],
                }
            )
            return datasets.DatasetDict({"train": empty})

        combined_splits: List[datasets.Dataset] = []

        for raw in fixed_spec_list:
            spec = FixedDatasetSpec(
                dataset_name=raw.get("dataset_name", ""),
                config_name=raw.get("config_name"),
                split=raw.get("split", "train"),
                input_col=raw.get("input_col", "input"),
                output_col=raw.get("output_col"),
                filters=raw.get("filters"),
                postprocess=raw.get("postprocess"),
            )
            if not spec.dataset_name:
                self.logger.warning("Skipping spec without dataset_name: %s", raw)
                continue

            self.logger.info(
                "Loading fixed dataset %s (config=%s, split=%s)",
                spec.dataset_name,
                spec.config_name,
                spec.split,
            )
            try:
                hf_ds = datasets.load_dataset(
                    spec.dataset_name,
                    spec.config_name,
                    split=spec.split,
                    trust_remote_code=True,
                )
            except Exception as e:  # pragma: no cover - network / HF errors
                self.logger.error(
                    "Failed to load dataset %s (%s): %s",
                    spec.dataset_name,
                    spec.config_name,
                    e,
                )
                continue

            hf_ds = self._apply_filters(hf_ds, spec)
            if len(hf_ds) == 0:
                self.logger.warning("Dataset %s became empty after filtering.", spec.dataset_name)
                continue

            standardized = self._standardize_columns(hf_ds, spec)
            combined_splits.append(standardized)

        if not combined_splits:
            empty = datasets.Dataset.from_dict(
                {
                    "input_src": [],
                    "output_src": [],
                    "label": [],
                    "source": [],
                }
            )
            return datasets.DatasetDict({"train": empty})

        if len(combined_splits) == 1:
            train = combined_splits[0]
        else:
            train = datasets.concatenate_datasets(combined_splits)

        return datasets.DatasetDict({"train": train})


__all__ = ["FixedDatasetLoader", "FixedDatasetSpec"]
