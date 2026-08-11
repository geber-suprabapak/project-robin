"""
FastAPI dependencies for the Face Recognition API.

Provides authentication and authorization utilities.
"""

from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Header, HTTPException, status
from src.config import settings


@lru_cache(maxsize=4)
def _get_jwks_client(url: str) -> jwt.PyJWKClient:
    """Reuse PyJWT's bounded JWKS/key caches between requests."""
    return jwt.PyJWKClient(
        url,
        cache_keys=True,
        max_cached_keys=16,
        lifespan=300,
        headers={"User-Agent": "project-robin/1.0"},
    )


async def verify_jwt_bearer(authorization: Optional[str] = Header(None, alias="Authorization")) -> str:
    """
    Dependency to verify Supabase JWT Bearer token.
    
    Extracts the Bearer token from Authorization header, verifies its signature
    using Supabase JWKS (preferred) or the legacy HS256 secret, and returns the
    user_id from the ``sub`` claim.
    
    Args:
        authorization: Authorization header value (Bearer <token>)
        
    Returns:
        The user_id (sub claim) from the verified JWT
        
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
        HTTPException: 500 if JWT authentication is not configured
        
    Example:
        ```python
        @app.post("/protected")
        async def protected_route(user_id: str = Depends(verify_jwt_bearer)):
            return {"user_id": user_id}
        ```
    """
    use_jwks = bool(settings.supabase_jwks_url and settings.supabase_jwt_issuer)
    if not use_jwks and not settings.supabase_jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT authentication not configured"
        )
    
    # Extract Bearer token
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'"
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    try:
        if use_jwks:
            signing_key = _get_jwks_client(settings.supabase_jwks_url).get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience=settings.supabase_jwt_audience,
                issuer=settings.supabase_jwt_issuer,
                options={"require": ["sub", "exp", "iss", "aud"]},
            )
        else:
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience=settings.supabase_jwt_audience,
                options={"require": ["sub", "exp", "aud"]},
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user identifier"
            )
        
        return user_id
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
