# Contributing

Thanks for your interest in VoiceQA. The project is pre-alpha, so expect things to change quickly.

## Setup

```
python -m venv .venv
pip install -e ".[dev]"
```

## Before opening a PR

```
ruff check .
ruff format .
pytest
```

CI runs on Windows, macOS and Linux with Python 3.11 to 3.13. Please keep it green on all of them.

## Guidelines

- Add a test for every behavior change or bug fix.
- Keep pull requests small and focused on one thing.
- Use short conventional commit messages, for example `feat: add g711 codec` or `fix: handle sip 486 busy`.
- Core dependencies must be Apache-2.0 compatible. GPL code (for example PJSIP or Piper) can only live in optional, separately installed plugins.

## Reporting bugs

Open an issue with your OS, Python version, the output of `voiceqa doctor`, and the steps to reproduce. Remove any phone numbers, SIP credentials or recordings you don't want to share.
