"""FastAPI dependencies for Robin's private Astra-to-technical-service boundary."""

from typing import Optional

from fastapi import Header, HTTPException, status
from src.config import settings


async def verify_jwt_bearer(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    astra_user_id: Optional[str] = Header(None, alias="X-Astra-User-Id"),
) -> str:
    """Require the dedicated Astra service credential and explicit user context."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'",
        )
    if authorization[7:] != settings.robin_service_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Robin service credential",
        )
    if not astra_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Astra user context",
        )
    return astra_user_id
