# Code Style & Conventions

## Python
- Type hints: Always use (Pydantic models, function signatures)
- Docstrings: Google style for public APIs
- Logging: Use `logging` module with structured format
- Async: Use async functions for I/O operations

## Naming
- Functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE

## Architecture
- Singleton pattern for inference engine
- Service layer for business logic
- Pydantic models for request/response
- FastAPI dependencies for auth

## Error Handling
- Custom exceptions (e.g., `ImageDecodeError`, `FaceDetectionError`)
- HTTPException for API errors with proper status codes
