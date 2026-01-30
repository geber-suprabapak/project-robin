"""
Supabase client service for user profile management.

Note: Face embedding operations have been migrated to Qdrant.
This service now focuses solely on user profile lookup and management.
"""

from typing import Optional, Dict, Any
from supabase import create_client, Client
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class SupabaseService:
    """
    Service for interacting with Supabase database.
    
    Handles user profile operations. Face embeddings are now
    managed by the Qdrant service.
    """
    
    def __init__(self):
        """Initialize Supabase client."""
        self.client: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Supabase client connection."""
        try:
            if not settings.supabase_url or not settings.supabase_key:
                logger.warning(
                    "Supabase credentials not configured. "
                    "User profile operations will be unavailable."
                )
                return
            
            self.client = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=settings.supabase_service_role_key or settings.supabase_key
            )
            logger.info("✓ Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize Supabase client: {str(e)}")
            self.client = None
    
    def is_connected(self) -> bool:
        """
        Check if Supabase client is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self.client is not None
    
    async def get_user_profile_by_nis(self, nis: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile by NIS from user_profiles table.
        
        Args:
            nis: Student ID (NIS/NISN)
            
        Returns:
            User profile dict or None if not found
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot fetch user profile.")
            return None
        
        try:
            response = self.client.table("user_profiles").select("*").eq("nis", nis).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"✗ Error fetching user profile by NIS={nis}: {str(e)}")
            return None
    
    async def get_user_profile_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile by user_id from user_profiles table.
        
        Args:
            user_id: User ID (UUID from Supabase auth)
            
        Returns:
            User profile dict or None if not found
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot fetch user profile.")
            return None
        
        try:
            response = self.client.table("user_profiles").select("*").eq("user_id", user_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"✗ Error fetching user profile by user_id={user_id}: {str(e)}")
            return None
    
    async def get_student_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve student info by user_id using RPC function.
        
        Args:
            user_id: User ID to look up
            
        Returns:
            Student data dict or None if not found
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot fetch student data.")
            return None
        
        try:
            response = self.client.rpc(
                "get_student_by_user_id",
                {"p_user_id": user_id}
            ).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"✗ Error fetching student by user_id={user_id}: {str(e)}")
            return None


# Global Supabase service instance
supabase_service = SupabaseService()
