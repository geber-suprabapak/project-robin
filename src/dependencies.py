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


async def verify_client_key(
    x_client_key: str = Header(None, alias="X-Client-Key"),
    x_admin_key: str = Header(None, alias="X-Admin-Key")
) -> str:
    """
    Dependency to validate client API key.
    
    Accepts either:
    - X-Client-Key header (for kiosk/frontend devices)
    - X-Admin-Key header (admin can access all endpoints)
    
    Args:
        x_client_key: Client key from X-Client-Key header
        x_admin_key: Admin key from X-Admin-Key header (fallback)
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: 401 if no valid key provided
    """
    # Check admin key first (admin can do everything)
    if x_admin_key and settings.admin_secret_key:
        if x_admin_key == settings.admin_secret_key:
            return x_admin_key
    
    # Check client key configuration
    if not settings.client_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Client authentication not configured"
        )
    
    # Validate client key
    if not x_client_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Client Key"
        )
    
    if x_client_key != settings.client_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Client Key"
        )
    
    return x_client_key
