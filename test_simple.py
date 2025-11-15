import os
os.environ["P2M_BACKEND"] = "ollama"
os.environ["P2M_GEN_MODEL"] = "llama3.1:8b"

from prompt2model.dataset_generator.prompt_based import DEFAULT_GEN_MODEL
print(f"DEFAULT_GEN_MODEL: {DEFAULT_GEN_MODEL}")

from prompt2model.utils.api_tools import APIAgent
agent = APIAgent(model_name="llama3.1:8b", max_tokens=50)
print(f"Agent backend: {agent.backend}")
print(f"Agent model: {agent.model_name}")
print("✓ 所有配置正确！")
