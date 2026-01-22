"""Exception handlers for the API."""

import logging
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from src.config import settings

logger = logging.getLogger(__name__)


async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.__class__.__name__, "message": exc.detail}
    )


async def general_exception_handler(request, exc: Exception):
    """General exception handler for uncaught exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.environment == "development" else None
        }
    )
