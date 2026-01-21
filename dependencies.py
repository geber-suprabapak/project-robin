"""
FastAPI dependencies for the Face Recognition API.
"""

from fastapi import Header, HTTPException, status
from config import settings


async def get_admin_api_key(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> str:
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
