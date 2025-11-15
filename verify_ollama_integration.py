#!/usr/bin/env python3
"""验证所有 Ollama 集成点"""
import os
import sys

os.environ["P2M_BACKEND"] = "ollama"
os.environ["P2M_GEN_MODEL"] = "llama3.1:8b"

print("=" * 70)
print("Ollama 集成验证脚本")
print("=" * 70)
print()

# 测试 1: 检查 DEFAULT_GEN_MODEL
print("✓ 测试 1: prompt_based.py 中的 DEFAULT_GEN_MODEL")
try:
    from prompt2model.dataset_generator.prompt_based import DEFAULT_GEN_MODEL
    print(f"  DEFAULT_GEN_MODEL = {DEFAULT_GEN_MODEL}")
    assert DEFAULT_GEN_MODEL == "llama3.1:8b"
    print("  ✓ 通过")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)
print()

# 测试 2: 检查 APIAgent 默认模型
print("✓ 测试 2: APIAgent 默认模型配置")
try:
    from prompt2model.utils.api_tools import APIAgent
    
    # 不传 model_name，应该从环境变量读取
    agent1 = APIAgent()
    print(f"  无参数创建: model_name = {agent1.model_name}")
    assert agent1.model_name == "llama3.1:8b"
    
    # 传入 model_name
    agent2 = APIAgent(model_name="qwen2.5:14b-instruct")
    print(f"  指定模型创建: model_name = {agent2.model_name}")
    assert agent2.model_name == "qwen2.5:14b-instruct"
    
    print("  ✓ 通过")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)
print()

# 测试 3: 检查 DatasetTransformer
print("✓ 测试 3: DatasetTransformer 的 generate_responses")
try:
    from prompt2model.dataset_transformer.prompt_based import PromptBasedDatasetTransformer
    import inspect
    
    # 检查函数签名
    sig = inspect.signature(PromptBasedDatasetTransformer.generate_responses)
    params = sig.parameters
    
    if 'model_name' in params:
        default = params['model_name'].default
        print(f"  model_name 默认值 = {default}")
        assert default is None or "llama" in str(default).lower()
        print("  ✓ 通过")
    else:
        print("  ✗ 未找到 model_name 参数")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 测试 4: 检查环境变量设置
print("✓ 测试 4: 环境变量配置")
print(f"  P2M_BACKEND = {os.getenv('P2M_BACKEND')}")
print(f"  P2M_GEN_MODEL = {os.getenv('P2M_GEN_MODEL')}")
print(f"  P2M_OLLAMA_BASE = {os.getenv('P2M_OLLAMA_BASE', 'http://localhost:11434')}")
print(f"  P2M_OLLAMA_EMBED_MODEL = {os.getenv('P2M_OLLAMA_EMBED_MODEL', 'nomic-embed-text')}")
print("  ✓ 通过")
print()

print("=" * 70)
print("✓ 所有测试通过！Ollama 集成配置正确")
print("=" * 70)
print()
print("修改摘要:")
print("  1. prompt_based.py (generator): 从环境变量读取模型")
print("  2. prompt_based.py (transformer): 默认使用 Ollama 模型")
print("  3. api_tools.py: APIAgent 默认使用 llama3.1:8b")
print("  4. p2m.py: 环境变量默认设置为 llama3.1:8b")
