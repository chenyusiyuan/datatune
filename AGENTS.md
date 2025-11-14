# Repository Guidelines

## Project Structure & Modules
- Core library code lives in `prompt2model/` (dataset retriever, generator, trainer, demo utilities).
- CLI / demos: `p2m.py`, `prompt2model_demo.py`, and `prompt2model_demo.ipynb`.
- Scripts and one-off utilities: `scripts/`.
- Tests and helpers: `tests/` and `test_helpers/`.
- Example configs and prompts: `prompt.yaml`, `prompt_examples.md`, and `examples/`.

## Build, Test, and Development
- Install in editable mode:
  - `pip install -e .[dev,test]`
- Run unit tests:
  - `pytest` (uses settings in `pyproject.toml`).
- Type checking (if needed):
  - `mypy prompt2model`

## Coding Style & Naming
- Python 3.9+ codebase; use 4-space indentation and type hints.
- Follow existing patterns in nearby files; prefer explicit, descriptive names over abbreviations.
- Keep functions small and focused; avoid adding new top-level scripts when a module function fits.
- Do not introduce new formatting or linting tools; respect existing `mypy.ini` and config in `pyproject.toml`.

## Testing Guidelines
- Prefer adding or updating tests under `tests/` alongside the feature or bugfix.
- Name tests `test_<feature>.py` and use `pytest` style assertions.
- When changing behavior, add at least one test that would fail before your change and pass after.

## Commit & Pull Request Practices
- Write concise, imperative commit messages (e.g., `Add dataset size helper`, `Fix HF retriever timeout`).
- For pull requests, include:
  - A short summary of the change and motivation.
  - Any breaking changes or migration notes.
  - How you tested (commands run, datasets or configs used).
- Keep PRs focused and reasonably small; separate unrelated refactors into different PRs when possible.

