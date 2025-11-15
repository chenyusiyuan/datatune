from prompt2model.prompt_parser.parse_responses import extract_content, sanitize_name
import re

# 内容规整
cases = [
    {"choices": [{"message": {"content": "Dataset: clinc_oos\nConfig: plus"}}]},
    {"choices": [{"text": "Dataset: clinc_oos\nConfig: small"}]},
    "Dataset: clinc_oos\nConfig: imbalanced",
]
pat = r"Dataset:\s*(?P<ds>[^\n]+)\s*Config:\s*(?P<cfg>[^\n]+)"
for c in cases:
    s = extract_content(c)
    assert isinstance(s, str) and re.search(pat, s)

# 名称清洗
assert sanitize_name("[a] small") == "small"
assert sanitize_name("  [c]   plus ") == "plus"

# AttrDict 行为
from prompt2model.utils.api_tools import _to_attrdict
d = {"choices": [{"message": {"content": "ok"}}]}
o = _to_attrdict(d)
assert o["choices"][0]["message"]["content"] == "ok"
assert o.choices[0].message.content == "ok"
