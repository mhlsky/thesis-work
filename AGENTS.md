# Thesis Work Agent Instructions

## Communication

- Default to Chinese for explanations, plans, summaries, and review comments unless the user explicitly asks for English.
- Keep updates concise and practical. State what you are going to inspect or change before substantial work.
- When assumptions affect experiment behavior or results, state them explicitly.

## Project Scope

- This repository is a Python 3.11 project managed with `uv`.
- Preserve the standard `src/` layout. Package code belongs under `src/ship_motion/`.
- Prefer adding reusable logic to `src/ship_motion/` and keeping entry scripts thin.
- Keep configuration in `configs/` rather than hardcoding experiment parameters in scripts.
- Treat files under `docs/` as project documentation. Update docs when behavior, workflow, or structure changes materially.

## Safety Rules

- Do not modify files outside this project directory unless the user explicitly asks for it.
- Do not modify files under `data/` unless the user explicitly asks for dataset changes.
- Do not commit or fabricate large generated artifacts, model weights, caches, or experiment outputs.
- Prefer writing generated outputs to `outputs/` or another user-designated directory, and avoid polluting the repo root.
- Avoid destructive cleanup commands against data, outputs, or virtual environment directories unless the user explicitly requests them.

## Environment And Commands

- Use `uv` for environment and command execution when possible.
- Prefer PowerShell-compatible commands and scripts for examples and automation.
- Before introducing a new dependency, confirm it is necessary and add it to `pyproject.toml`.
- Lightweight fallback logic is acceptable for smoke tests or early validation, but when work moves into formal implementation, training, evaluation, or integration, switch back to the proper runtime environment and official libraries when feasible.
- Prefer repository-documented commands first:
  - `uv sync`
  - `uv run python -m ship_motion.data.dataset --config configs/base.yaml --smoke`
  - `pwsh -File .\\scripts\\smoke_test.ps1`

## Code Changes

- Follow the existing code style and keep changes minimal unless a broader refactor is necessary.
- Avoid moving files or restructuring modules without a clear need.
- When adding or substantially rewriting code, prefer beginner-friendly comments and docstrings that explain the purpose of the code, the role of key variables, and non-obvious control flow.
- Add brief comments only where the code would otherwise be hard to follow.
- Keep scripts, configs, and package code consistent with each other when changing experiment workflow.

## Configuration

- Add field-level explanations for important configuration files when creating or updating them, so a beginner can understand what each key controls.
- Prefer keeping configuration self-explanatory with short inline comments or nearby documentation when the file format allows it.

## Verification

- For code changes, run the smallest relevant verification you can.
- Prefer smoke tests or targeted module execution before proposing broader test runs.
- If verification cannot be run, say so clearly and explain why.

## Reviews

- In review mode, prioritize bugs, experiment-risk regressions, incorrect assumptions, and missing validation.
- Call out result reproducibility risks explicitly when config, preprocessing, or evaluation logic changes.
