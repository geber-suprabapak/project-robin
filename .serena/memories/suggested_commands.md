# Suggested Commands

## Run Server
```powershell
# Quick start (recommended)
.\run-local.ps1

# Manual with uv
uv sync --python 3.12
uv run python main.py
```

## Development
```powershell
# Install dependencies
uv sync --python 3.12

# Run with hot reload
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Docker
```powershell
docker-compose up -d
```

## API Testing
```powershell
# Health check
curl http://localhost:8000/health

# Docs
# Open http://localhost:8000/docs
```

## System Commands (Windows)
- `ls` / `dir` - list files
- `git status` - git status
- `uv run python -c "..."` - run python code
