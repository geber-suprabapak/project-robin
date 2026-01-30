"""
FastAPI dependencies for the Face Recognition API.
"""

from fastapi import Header, HTTPException, status
from src.config import settings


async def verify_admin_key(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> str:
    """
    Dependency to validate admin API key.
    
    Args:
        x_admin_key: Admin secret key from X-Admin-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If key is missing or invalid
    """
    if not settings.admin_secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin authentication not configured"
        )
    
    if x_admin_key != settings.admin_secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key"
        )
    
    return x_admin_key


async def verify_jwt_bearer(authorization: str = Header(..., alias="Authorization")) -> str:
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
    """
    import jwt
    
    # Check JWT secret is configured
    if not settings.supabase_jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT authentication not configured"
        )
    
    # Extract Bearer token
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

