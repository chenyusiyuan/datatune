#!/usr/bin/env python3
"""测试 Ollama 集成是否正常工作"""
import os
import sys

# 设置环境变量
os.environ["P2M_BACKEND"] = "ollama"
os.environ["P2M_GEN_MODEL"] = "llama3.1:8b"

# 测试 1: 检查环境变量
print("=" * 60)
print("测试 1: 环境变量配置")
print("=" * 60)
print(f"P2M_BACKEND: {os.getenv('P2M_BACKEND')}")
print(f"P2M_GEN_MODEL: {os.getenv('P2M_GEN_MODEL')}")
print(f"P2M_OLLAMA_BASE: {os.getenv('P2M_OLLAMA_BASE', 'http://localhost:11434')}")
print()

# 测试 2: 检查 DEFAULT_GEN_MODEL 是否正确读取
print("=" * 60)
print("测试 2: 模块导入和默认模型")
print("=" * 60)
try:
    from prompt2model.dataset_generator.prompt_based import DEFAULT_GEN_MODEL
    print(f"✓ DEFAULT_GEN_MODEL: {DEFAULT_GEN_MODEL}")
    assert DEFAULT_GEN_MODEL == "llama3.1:8b", f"Expected 'llama3.1:8b', got '{DEFAULT_GEN_MODEL}'"
    print("✓ 默认模型设置正确")
except Exception as e:
    print(f"✗ 错误: {e}")
    sys.exit(1)
print()

# 测试 3: 测试 APIAgent 创建
print("=" * 60)
print("测试 3: APIAgent 初始化")
print("=" * 60)
try:
    from prompt2model.utils.api_tools import APIAgent
    agent = APIAgent(model_name="llama3.1:8b", max_tokens=100)
    print(f"✓ APIAgent 创建成功")
    print(f"  - Backend: {agent.backend}")
    print(f"  - Model: {agent.model_name}")
    print(f"  - Ollama base: {agent.ollama_base}")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 测试 4: 测试简单 API 调用
print("=" * 60)
print("测试 4: Ollama API 调用")
print("=" * 60)
try:
    response = agent.generate_one_completion(
        prompt="Say 'Hello, world!' and nothing else.",
        temperature=0.1,
        responses_per_request=1
    )
    print(f"✓ API 调用成功")
    print(f"  响应: {str(response)[:100]}...")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("✓ 所有测试通过！Ollama 集成正常工作")
print("=" * 60)
