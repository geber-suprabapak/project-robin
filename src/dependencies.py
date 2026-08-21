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
    Dependency to verify an OIDC/Logto JWT Bearer token.
 
    The token subject is used only as Robin's technical artifact key. Robin does
    not query Astra or any business database.
 
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
    use_jwks = bool(settings.jwt_jwks_url and settings.jwt_issuer)
    if not use_jwks and not settings.jwt_secret:
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
            signing_key = _get_jwks_client(settings.jwt_jwks_url).get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=settings.jwt_audience,
                issuer=settings.jwt_issuer,
                options={"require": ["sub", "exp", "iss", "aud"]},
            )
        else:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=["HS256"],
                audience=settings.jwt_audience,
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
