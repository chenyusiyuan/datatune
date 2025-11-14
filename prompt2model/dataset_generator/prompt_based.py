"""A simple dataset generator that uses APIs."""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import nest_asyncio
import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.dataset_generator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import (
    API_ERRORS,
    APIAgent,
    api_tools,
    count_tokens_from_string,
    get_formatted_logger,
    handle_api_error,
)

nest_asyncio.apply()
logger = get_formatted_logger("DatasetGenerator")

# ===== 使用本地 Ollama 的指令模型生成合成数据 =====
# 可按需改为 "llama3.2:3b-instruct" / "qwen2.5:7b-instruct" 等已在 Ollama 中可用的模型名称
DEFAULT_GEN_MODEL = "llama3.1:8b"


@dataclass(frozen=True)
class Example:
    """An example from a dataset, containing input and output columns."""

    input_col: str
    output_col: str

    def __eq__(self, other) -> bool:
        """Example equality."""
        return self.input_col == other.input_col and self.output_col == other.output_col

    def __lt__(self, other) -> bool:
        """Example less than."""
        return self.input_col < other.input_col or self.output_col < other.output_col


class PromptBasedDatasetGenerator(DatasetGenerator):
    """A abstract class for NLP dataset generation using a prompted API."""

    def __init__(
        self,
        max_api_calls: int = None,
        initial_temperature: float = 0.5,
        max_temperature: float = 1.7,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        max_batch_size: int = 5,
        responses_per_request: int = 1,#5
        requests_per_minute: int = 80,
        filter_duplicated_examples: bool = True,
        cache_root: str = "cached_generated_dataset",
    ):
        """Initializes an instance of the PromptBasedDatasetGenerator."""
        if max_api_calls and max_api_calls <= 0:
            raise ValueError("max_api_calls must be > 0")
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0
        self.initial_temperature = initial_temperature
        self.max_temperature = max_temperature
        if self.initial_temperature < 0:
            raise ValueError(
                f"initial_temperature must be >= 0, but initial_temperature={self.initial_temperature}"
            )
        if self.max_temperature > 2.0:
            raise ValueError(
                f"max_temperature must be <= 2.0 but max_temperature={self.max_temperature}"
            )

        if self.initial_temperature > self.max_temperature:
            raise ValueError(
                f"initial_temperature={self.initial_temperature} must be <= max_temperature={self.max_temperature}"
            )
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_batch_size = max_batch_size
        self.responses_per_request = responses_per_request
        self.requests_per_minute = requests_per_minute
        self.filter_duplicated_examples = filter_duplicated_examples

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        generated_examples: list[Example],
        context_cutoff: int = 3500,
    ) -> str:
        """Generates a prompt string with a strict 'JSON-only' constraint."""
        while True:
            # Low-quality few-shot（动态从已生成样本中抽取）
            if len(generated_examples) == 0:
                low_quality_example_string = "N/A\n"
                random_selected_generated_example_num = 0
            else:
                low_quality_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(generated_examples), 10)
                )
                random_examples = random.sample(
                    generated_examples, random_selected_generated_example_num
                )
                for example in random_examples:
                    low_quality_example_string += (
                        f'input="{example.input_col}"\noutput="{example.output_col}"\n'
                    )

            # 三种模板：COMPLEX/MIDDLE/SIMPLE
            template_type_dict = {1: "COMPLEX", 2: "MIDDLE", 0: "SIMPLE"}
            template_type = template_type_dict[
                random_selected_generated_example_num % 3
            ]

            prompt = construct_meta_prompt(
                instruction=instruction,
                low_quality_example_string=low_quality_example_string,
                high_quality_example_string=few_shot_example_string,
                template_type=template_type,
            )

            # 严格要求“仅返回 JSON”，禁止解释、Markdown、代码块
            strict_suffix = (
                "\n\n"
                "You must return ONLY a single compact JSON object on one line, with keys exactly:\n"
                '{"input": "...", "output": "..."}\n'
                "No markdown, no code fences, no commentary, no extra keys.\n"
                "如果无法生成，请输出一个空 JSON：{\"input\": \"\", \"output\": \"\"}\n"
            )
            prompt = prompt + strict_suffix

            if count_tokens_from_string(prompt) < context_cutoff:
                return prompt
            else:
                base_str = instruction + (few_shot_example_string or "")
                if count_tokens_from_string(base_str) > context_cutoff:
                    logger.warning(
                        "The original input prompt is too long. Consider writing a shorter prompt."
                    )
                # 继续循环尝试更短组合
                continue

    def apply_multi_vote_filtering(
        self,
        generated_examples: list[Example],
    ) -> list[Example]:
        """Multi-vote to construct generated_dataset from input_output_map."""
        if not self.filter_duplicated_examples:
            raise ValueError("Multi-vote filtering is not enabled.")
        filtered_examples = []

        input_output_map: dict[str, Counter] = defaultdict(Counter)
        for ex in generated_examples:
            input_output_map[ex.input_col][ex.output_col] += 1

        for input_str, output_counter in input_output_map.items():
            most_common_count = output_counter.most_common(1)[0][1]

            most_frequent_outputs = [
                output
                for output, count in output_counter.items()
                if count == most_common_count
            ]

            # 同频率取更短的；再按字典序
            most_frequent_outputs.sort(key=len)
            final_output = most_frequent_outputs[0]
            filtered_examples.append(Example(input_str, final_output))
        return filtered_examples

    def compute_batch_size(self, num_examples: int, generated_dataset_size: int) -> int:
        """Computes the batch size for API calls in a batch."""
        max_api_calls = (
            self.max_batch_size
            if self.max_api_calls is None
            else self.max_api_calls - self.api_call_counter
        )
        batch_size = min(
            self.max_batch_size,
            math.ceil(
                ((num_examples - generated_dataset_size) / self.responses_per_request)
            ),
            max_api_calls,
        )
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        return batch_size

    def extract_and_append_responses(
        self, completions: list[openai.Completion], generated_examples: list[Example]
    ) -> None:
        """Robustly extracts {"input": "...", "output": "..."} from API responses."""

        def _first_json_obj(text: str):
            import re

            m = re.search(r"\{.*?\}", text, flags=re.S)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

        def _normalize_io(obj: dict):
            # 字段别名：question/answer → input/output
            if "input" not in obj and "question" in obj:
                obj["input"] = obj.get("question")
            if "output" not in obj and "answer" in obj:
                obj["output"] = obj.get("answer")
            return obj

        for completion in completions:
            try:
                for choice in completion.get("choices", []):
                    content = ""
                    msg = choice.get("message") or {}
                    if isinstance(msg, dict):
                        content = msg.get("content", "") or ""
                    else:
                        content = str(choice)

                    obj = None
                    # 1) 直接 JSON
                    try:
                        obj = json.loads(content)
                    except Exception:
                        # 2) 从文本中抽第一段 JSON
                        obj = _first_json_obj(content)

                    if not isinstance(obj, dict):
                        logger.warning(
                            f"Error happened parsing API choice: {choice}"
                        )
                        continue

                    obj = _normalize_io(obj)
                    missing = [k for k in ("input", "output") if k not in obj]
                    if missing:
                        logger.warning(f"API response missing keys: {missing}")
                        continue

                    input_str = str(obj.get("input", "")).strip()
                    output_str = str(obj.get("output", "")).strip()
                    if input_str and output_str:
                        generated_examples.append(Example(input_str, output_str))
                        logger.info(f"input: \n\n{input_str}\n\n")
                        logger.info(f"output: \n\n{output_str}\n\n")
                    else:
                        logger.info("Empty input/output. Discard.")
            except Exception:
                logger.warning(
                    f"Error happened when parsing API completion: {completion}"
                )
                continue

    async def generate_responses(
        self,
        chat_api: APIAgent,
        generated_dataset_size: int,
        expected_num_examples: int,
        prompts: list[str],
    ) -> list[openai.Completion]:
        """Asynchronously generates responses using APIAgent."""
        dynamic_temperature = (
            (self.max_temperature - self.initial_temperature)
            * generated_dataset_size
            / max(1, expected_num_examples)
            + self.initial_temperature
        )
        clipped_temperature = max(0.0, min(2.0, dynamic_temperature))

        responses = await chat_api.generate_batch_completion(
            prompts,
            temperature=clipped_temperature,
            responses_per_request=self.responses_per_request,
            requests_per_minute=self.requests_per_minute,
        )
        return responses

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit = DatasetSplit.TRAIN,
    ) -> Dataset:
        """Generates a dataset split using API-based LMs."""
        all_generated_examples: list[Example] = []
        generated_examples: list[Example] = []

        # ---- 统一 few-shot 示例为字符串（支持 list/dict/str 混合）----
        def _examples_to_string(examples) -> str:
            if not examples:
                return ""
            if isinstance(examples, str):
                return examples
            parts = []
            for ex in examples if isinstance(examples, (list, tuple)) else [examples]:
                if isinstance(ex, dict):
                    inp = str(ex.get("input", ex.get("question", ""))).strip()
                    out = str(ex.get("output", ex.get("answer", ""))).strip()
                    if inp or out:
                        parts.append(f'input="{inp}"\noutput="{out}"')
                else:
                    s = str(ex).strip()
                    if s:
                        parts.append(s)
            return "\n".join(parts)

        few_shot_example_string = _examples_to_string(prompt_spec.examples)
        # -------------------------------------------------------------

        pbar = tqdm(total=num_examples, desc="Generating examples")

        # 使用本地 Ollama 模型：在这里显式创建 agent（不要用默认的 default_api_agent）
        chat_api = APIAgent(model_name=DEFAULT_GEN_MODEL, max_tokens=4000)

        while len(generated_examples) < num_examples:
            if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                logger.warning("Maximum number of API calls reached.")
                break

            batch_size = self.compute_batch_size(num_examples, len(generated_examples))
            self.api_call_counter += batch_size

            # Generate prompts for the batch call.
            prompts = [
                self.construct_prompt(
                    instruction=prompt_spec.instruction,
                    few_shot_example_string=few_shot_example_string,
                    generated_examples=generated_examples,
                )
                for _ in range(batch_size)
            ]

            try:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(
                    self.generate_responses(
                        chat_api=chat_api,
                        generated_dataset_size=len(generated_examples),
                        expected_num_examples=num_examples,
                        prompts=prompts,
                    )
                )
            except API_ERRORS as e:
                handle_api_error(e)
                continue  # 发生可重试错误时跳过本批次

            # Extract the responses and add new examples to the dataset.
            prev_length = len(generated_examples)
            self.extract_and_append_responses(responses, all_generated_examples)
            generated_examples = (
                self.apply_multi_vote_filtering(all_generated_examples)
                if self.filter_duplicated_examples
                else all_generated_examples
            )

            pbar.update(len(generated_examples) - prev_length)

        if len(generated_examples) >= num_examples:
            generated_examples = generated_examples[:num_examples]

        return Dataset.from_dict(
            {
                "input_col": [ex.input_col for ex in generated_examples],
                "output_col": [ex.output_col for ex in generated_examples],
            }
        )
