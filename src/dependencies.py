"""
FastAPI dependencies for the Face Recognition API.

Provides authentication and authorization utilities.
"""

from typing import Optional

from fastapi import Header, HTTPException, status
from src.config import settings


async def verify_jwt_bearer(authorization: Optional[str] = Header(None, alias="Authorization")) -> str:
    """
    Dependency to verify Supabase JWT Bearer token.
    
    Extracts the Bearer token from Authorization header, verifies signature
    using SUPABASE_JWT_SECRET, and returns the user_id from 'sub' claim.
    
    Args:
        authorization: Authorization header value (Bearer <token>)
        
    Returns:
        The user_id (sub claim) from the verified JWT
        
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
        HTTPException: 500 if JWT secret is not configured
        
    Example:
        ```python
        @app.post("/protected")
        async def protected_route(user_id: str = Depends(verify_jwt_bearer)):
            return {"user_id": user_id}
        ```
    """
    import jwt
    
    # Check JWT secret is configured
    if not settings.supabase_jwt_secret:
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
        # Decode and verify JWT
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={"require": ["sub", "exp"]}
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
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
