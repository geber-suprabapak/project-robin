<!-- repo-map:start -->
## Development and verification

This is a Python 3.12+ FastAPI service managed with uv. Use `uv sync --group dev`
to provision development dependencies, `uv run ruff check src tests` for linting,
and `uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=70` for
the documented coverage gate. Read `README.md` before starting a local API server;
the project configuration remains authoritative.

## Codebase semantic map

Start repository-wide orientation at `docs/codebase-map/README.md`.

- Domain vocabulary and aliases: `docs/codebase-map/glossary.md`
- Semantic module ownership: `docs/codebase-map/modules.md`
- Architecture boundaries: `docs/codebase-map/architecture/`
- Cross-module flows: `docs/codebase-map/flows/`
- Rules that changes must preserve: `docs/codebase-map/invariants.md`
- Recorded decision sources: `docs/codebase-map/decisions/README.md`

For structural questions such as symbol callers, imports, dependencies, or code relationships, prefer an available code graph / codebase-memory tool before broad repository scans. The graph is an accelerator; source code, tests, schemas, and project docs remain authoritative.

Read only the map pages relevant to the task, then inspect the implementation they point to.
<!-- repo-map:end -->
