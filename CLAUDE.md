# Claude Code — Wildfire Dissertation

The full project guide is in `AGENTS.md` (shared with Codex). Read it:

@AGENTS.md

For the current dissertation scope, supervisor Q&A, terminology corrections,
and supporting-document locations, also read:

@../PROJECT_HANDOFF.md

## Claude-specific notes
- Use the **PowerShell** tool for Windows commands; the Bash tool only for POSIX scripts.
- For anything importing torch/ML, use the `pytorch_env` interpreter
  (`C:\Users\Afnan\anaconda3\envs\pytorch_env\python.exe`), never the system `python`.
- CPU-only locally — keep local runs tiny; real training is on Kaggle GPU.
