#!/usr/bin/env python3
"""简化的 Ollama 集成验证"""
import os
import sys

os.environ["P2M_BACKEND"] = "ollama"
os.environ["P2M_GEN_MODEL"] = "llama3.1:8b"

print("=" * 70)
print("Ollama 集成验证")
print("=" * 70)
print()

# 测试 1: APIAgent
print("✓ 测试 1: APIAgent 配置")
from prompt2model.utils.api_tools import APIAgent

agent1 = APIAgent()
print(f"  默认: model_name={agent1.model_name}, backend={agent1.backend}")
assert agent1.model_name == "llama3.1:8b"
assert agent1.backend == "ollama"

agent2 = APIAgent(model_name="qwen2.5:7b")
print(f"  自定义: model_name={agent2.model_name}")
assert agent2.model_name == "qwen2.5:7b"
print("  ✓ 通过")
print()

# 测试 2: 检查代码文件
print("✓ 测试 2: 检查源代码配置")
import subprocess

files_checks = [
    ("prompt2model/dataset_generator/prompt_based.py", 'os.getenv("P2M_GEN_MODEL"'),
    ("prompt2model/utils/api_tools.py", 'llama3.1:8b'),
    ("prompt2model/dataset_transformer/prompt_based.py", 'os.getenv("P2M_GEN_MODEL"'),
    ("p2m.py", 'llama3.1:8b'),
]

for filepath, pattern in files_checks:
    result = subprocess.run(
        ["grep", "-q", pattern, filepath],
        capture_output=True
    )
    status = "✓" if result.returncode == 0 else "✗"
    print(f"  {status} {filepath}: {pattern}")

print()

print("=" * 70)
print("✓ 验证完成！")
print("=" * 70)
print()
print("修改列表:")
print("  1. prompt2model/dataset_generator/prompt_based.py")
print("     - 添加 import os")
print("     - DEFAULT_GEN_MODEL 从环境变量读取")
print()
print("  2. prompt2model/dataset_transformer/prompt_based.py")  
print("     - 添加 import os")
print("     - generate_responses() 默认使用环境变量模型")
print()
print("  3. prompt2model/utils/api_tools.py")
print("     - APIAgent 默认模型改为 llama3.1:8b")
print()
print("  4. p2m.py")
print("     - P2M_GEN_MODEL 默认值改为 llama3.1:8b")
