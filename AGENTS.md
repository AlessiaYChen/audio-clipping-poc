# Repository Guidelines

## Project Structure & Module Organization
Source code belongs in `src/audio_clipping/`, grouped by responsibility (e.g., `processing`, `io`, `cli`). Shared utilities live in `src/audio_clipping/common.py`. Tests mirror that layout inside `tests/`, keeping fixtures under `tests/fixtures`. Reusable WAV snippets and reference plots go in `samples/` so unit and manual checks depend on the same assets. Experiments and evaluation scripts belong in `notebooks/` or `scripts/`; keep generated artifacts out of version control except for lightweight SVGs placed under `docs/assets/`.

## Build, Test, and Development Commands
Use Python 3.11+: `python -m venv .venv; .venv\Scripts\activate`. Install tooling with `pip install -r requirements.txt -r requirements-dev.txt`. Run the proof-of-concept via `python -m audio_clipping.cli --input samples/snare.wav --threshold -6`, which emits a clipped file and console diagnostics. Execute the full test suite with `pytest -q`, and run style checks using `ruff check src tests` and `black --check src tests`. `mypy src` keeps the signal-processing math honest by flagging type drift.

## Coding Style & Naming Conventions
Adopt Black’s 88-character width and 4-space indentation. Prefer dataclasses for immutable DSP configuration, and snake_case for functions, PascalCase for public classes, SCREAMING_SNAKE_CASE for constants (thresholds, sample rates). Every module should expose a minimal `__all__` to clarify the intended surface. Run `ruff format` before pushing to normalize import order and remove unused symbols.

## Testing Guidelines
Write pytest files as `test_<module>.py`, and name cases `test_<behavior>_<condition>`. Use hypothesis strategies or parameterized tests to cover edge amplitudes (e.g., 0.0, 0.99, 1.01). Store gold-master WAVs under `samples/fixtures/` and load them through helper factories instead of hard-coded paths. Confirm coverage stays above 85% using `pytest --cov=audio_clipping --cov-report=term-missing`.

## Commit & Pull Request Guidelines
Follow Conventional Commits (`feat(dsp): add soft clipper`). Limit to one concern per commit, referencing issues with `Refs #42` when applicable. PRs need: summary of changes, verification evidence (`pytest`, demo CLI output), and screenshots of waveform diffs if UI tools were used. Request review before merging, and wait for CI green checks.

## Security & Configuration Tips
Store secrets (API keys for cloud storage or telemetry) in `.env` loaded via `python-dotenv`. Provide sanitized defaults in `.env.example`. Large assets belong in Git LFS; never commit raw recordings over ~5 MB.
