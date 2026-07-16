# Repository Guidelines

## Project Structure & Module Organization

SEER is a Python 3.12 package under `src/seer`. Core modules are split by concern:
`model.py` wraps the language model with concept and energy heads, `energy.py`
defines certainty scoring, `optim.py` handles optimizer grouping, `train.py` and
`eval.py` provide training/evaluation utilities, and `config.py` contains typed
configuration. Tests live in `tests/` and mirror source concerns with files such
as `test_model.py`, `test_energy.py`, and `test_train.py`. Project design notes
belong in `docs/`, especially `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, and
`docs/ROADMAP.md`.

## Build, Test, and Development Commands

- `uv sync` installs runtime and development dependencies from `pyproject.toml`
  and `uv.lock`.
- `uv run pytest` runs the full test suite configured for `tests/`.
- `uv run pytest tests/test_model.py` runs a focused test file during iteration.
- `uv run ruff check .` checks formatting-adjacent lint rules, imports, upgrades,
  and bugbear issues.
- `uv build` builds the package using Hatchling.

## Coding Style & Naming Conventions

Use 4-space indentation, Python type hints, and small modules with explicit
responsibilities. Ruff is configured for Python 3.12 with a 100-character line
length and lint groups `E`, `F`, `I`, `UP`, and `B`. Keep public class names in
`PascalCase`, functions and variables in `snake_case`, and tests named
`test_<behavior>`. Preserve semantically important attribute names such as
`concept_proj`, `energy_head`, and `self_certainty`, which are referenced by
optimizer grouping logic.

## Testing Guidelines

Tests use `pytest` with lightweight fakes in `tests/fakes.py` to avoid network
downloads or external model checkpoints. Prefer deterministic tests with
`torch.manual_seed(...)` where tensor values or shapes matter. Add tests next to
the affected behavior and cover tensor shapes, gradient behavior, calibration
ranges, and optimizer classification when changing model internals.

## Commit & Pull Request Guidelines

Git history uses concise Conventional Commit-style prefixes such as `feat:` and
`docs:`. Keep commits focused and imperative, for example
`feat: add energy calibration metrics`. Pull requests should include the intent,
key design tradeoffs, tests run, and any documentation updates. Link related
issues or roadmap items, and include screenshots only for generated reports or
visual artifacts.

## Agent-Specific Instructions

When the user types `/graphify`, invoke the `graphify` skill before any other
work. Do not download model weights in tests; use fakes or cached checkpoints.
