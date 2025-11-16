# Repository Guidelines

## Project Structure & Modules
- Core library code lives in `prompt2model/` (retrievers, transformers, trainers, demo utilities).
- CLI and main entrypoints: `p2m.py`, `prompt2model/cli/p2m.py`, and `prompt2model_demo.py` / `.ipynb`.
- Tests and helpers are under `tests/` and `test_helpers/`.
- Example configs and prompts: `config.fixed_datasets.yaml`, `config.generation.yaml`, `config.task_rules.yaml`, `prompt.yaml`, `prompt_examples.md`, and `examples/`.
- One-off scripts live in `scripts/`.

## Build, Test, and Development
- Install in editable mode with dev/test extras:  
  `pip install -e .[dev,test]`
- Run the full unit test suite:  
  `pytest`
- Optional type checking for core code:  
  `mypy prompt2model`

## Coding Style & Naming Conventions
- Target Python 3.9+ with 4-space indentation and type hints.
- Follow existing patterns in nearby modules; prefer explicit, descriptive names over abbreviations.
- Keep functions small and focused; prefer adding helpers inside existing modules over new top-level scripts.
- Do not introduce new formatters or linters; respect `pyproject.toml` and `mypy.ini`.

## Testing Guidelines
- Place new tests in `tests/` next to the feature or bugfix (e.g., `tests/dataset_retriever_test.py`).
- Use `pytest`-style tests with clear `test_<feature>` function names.
- When changing behavior, add at least one test that fails before the change and passes after.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., `Add fixed dataset loader`, `Drop label column from transform output`).
- For pull requests, include: a short summary, motivation, any breaking changes, and how you tested (commands and configs).
- Keep changes focused and reasonably small; separate unrelated refactors into different PRs when possible.
