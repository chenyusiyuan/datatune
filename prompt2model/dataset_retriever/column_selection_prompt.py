"""Utilities to construct an LLM "metaprompt" for our column selection."""

from __future__ import annotations  # noqa FI58

METAPROMPT_BASE = """Your objective is to carefully analyze the task and the dataset and decide, for EACH CANDIDATE COLUMN, whether it is relevant input, relevant output (at most ONE), irrelevant, or ambiguous.

You MUST answer with a single valid JSON object with EXACTLY these four keys: input, output, irrelevant, ambiguous.

Hard constraints:
- You MUST choose column names EXCLUSIVELY from the provided Columns list (case-sensitive). Do NOT invent or rename columns (e.g., do NOT output "input", "output", "text", "label" unless they appear in Columns).
- At most ONE column may be in "output".
- If no column fits, return empty arrays for both "input" and "output".
- JSON only, no comments, no trailing commas, and no extra text before/after the JSON object.
- The value of each key must be a JSON array of strings (the chosen column names)."""  # noqa: E501

# 所有 few-shot 的“Columns”与“输出”必须严格一致
METAPROMPT_EXAMPLES = [
    (
        # Summarization
        """You are tasked with the following process. In this task, you will generate summaries for given texts. For this task, you will use the Scientific Papers dataset from HuggingFace.
Dataset description: Scientific papers datasets contains two sets of long and structured documents. The datasets are obtained from ArXiv and PubMed OpenAccess repositories.
A sample data instance:
{
  "abstract": "We have studied ...",
  "article": "The leptonic decays ...",
  "section_names": "[sec:introduction] introduction ..."
}
Columns: [abstract, article, section_names]""",
        # 期望选择
        """{
  "input": ["article"],
  "output": ["abstract"],
  "irrelevant": ["section_names"],
  "ambiguous": []
}""",
    ),
    (
        # Hate speech classification
        """You are tasked with the following process. Detect whether a given tweet uses hateful speech. Use the hate_speech_offensive dataset.
Dataset description: An annotated dataset for hate speech and offensive language detection on tweets.
A sample:
{
  "count": 3,
  "hate_speech_count": 0,
  "offensive_language_count": 0,
  "neither_count": 3,
  "label": 2,
  "tweet": "As a woman you shouldn't complain ..."
}
Columns: [tweet, label, count, hate_speech_count, offensive_language_count, neither_count]""",
        """{
  "input": ["tweet"],
  "output": ["label"],
  "irrelevant": [],
  "ambiguous": ["hate_speech_count", "offensive_language_count", "neither_count", "count"]
}""",
    ),
    (
        # 翻译场景（与列名严格匹配）
        """You are tasked with the following process. Translate from English to Spanish.
Dataset: opus_books (config: en-es)
A sample:
{
  "id": "000123",
  "translation_en": "This is a book.",
  "translation_es": "Este es un libro."
}
Columns: [id, translation_en, translation_es]""",
        """{
  "input": ["translation_en"],
  "output": ["translation_es"],
  "irrelevant": ["id"],
  "ambiguous": []
}""",
    ),
    (
        # 仅有一个聚合列 → 模糊
        """You are tasked with the following process. Your job is to translate between languages.
A sample:
{
  "translation": {"ca": "Hola", "en": "Hello"}
}
Columns: [translation]""",
        """{
  "input": [],
  "output": [],
  "irrelevant": [],
  "ambiguous": ["translation"]
}""",
    ),
    (
        # 与任务不符 → 全部无关
        """You are tasked with the following process. Summarize a text.
Dataset: math_qa
A sample:
{
  "Problem": "a multiple choice test ...",
  "Rationale": "5 choices ...",
  "options": "a) 24, b) 120 ...",
  "correct": "c",
  "annotated_formula": "power(5,4)"
}
Columns: [Problem, Rationale, options, correct, annotated_formula]""",
        """{
  "input": [],
  "output": [],
  "irrelevant": ["Problem", "Rationale", "options", "correct", "annotated_formula"],
  "ambiguous": []
}""",
    ),
]

INPUT_PROMPT_TEMPLATE = """You are tasked with the following process. {instruction}
Dataset: {dataset_name}
Dataset Description: {dataset_description}
Columns: [{dataset_columns}]
A sample data instance from this dataset is as follows:
{sample_row}
"""

SINGLE_DEMONSTRATION_TEMPLATE = (
    'Task and Data:\n"""\n{prompt}\n"""\n\nRequired Columns:\n{columns}'
)

ENDING_LINE = (
    "After seeing the examples, strictly choose column names ONLY from the Columns list above (case-sensitive). "
    "Return ONLY a single JSON object with keys: input, output, irrelevant, ambiguous. At most ONE column in output."
)


def build_input(
    instruction: str,
    dataset_name: str,
    dataset_description: str,
    dataset_columns: str,
    sample_row: dict,
) -> str:
    """Template function to build input based on arguments."""
    prompt = INPUT_PROMPT_TEMPLATE.format(
        instruction=instruction,
        dataset_name=dataset_name,
        dataset_description=dataset_description,
        dataset_columns=dataset_columns,
        sample_row=sample_row,
    )
    return prompt


def construct_prompt_for_column_selection(
    instruction: str,
    dataset_name: str,
    dataset_description: str,
    dataset_columns: str,
    sample_row: dict,
    retry_hint: str | None = None,
) -> str:
    """Generate prompt for column selection."""
    sections = [METAPROMPT_BASE]
    for prompt, columns in METAPROMPT_EXAMPLES:
        sections.append(
            SINGLE_DEMONSTRATION_TEMPLATE.format(prompt=prompt, columns=columns)
        )
    # 真实输入
    input_prompt = build_input(
        instruction, dataset_name, dataset_description, dataset_columns, sample_row
    )
    if retry_hint:
        input_prompt += (
            f"\nIMPORTANT: Your previous answer used invalid column names: {retry_hint}.\n"
            f"Choose ONLY from Columns: [{dataset_columns}] (case-sensitive)."
        )
    sections.append(
        SINGLE_DEMONSTRATION_TEMPLATE.format(prompt=input_prompt, columns="")
    )  # 预测列，留空
    sections.append(ENDING_LINE)
    return "\n\n------\n\n".join(sections)
