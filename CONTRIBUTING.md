# Contributing

Thanks for your interest in contributing to Project Robin.

## Development

1. Fork or clone the repository.
2. Copy `.env.example` to `.env`.
3. Install dependencies with `uv sync`.
4. Run checks before opening a pull request:

```bash
uv run ruff check .
uv run pytest
```

## Pull Requests

* Keep changes focused.
* Include tests for behavior changes.
* Update documentation when changing public API behavior.
* Do not commit secrets, face images, embeddings, logs, or production data.

## Security

For security-sensitive issues, do not open a public issue with exploit details. Follow `SECURITY.md`.
