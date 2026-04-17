"""
Supabase client service for user profile management.

Note: Face embedding operations have been migrated to Qdrant.
This service now focuses solely on user profile lookup and management.
"""

from typing import Optional, Dict, Any
from supabase import create_client, Client
import asyncio
import logging

from starlette.concurrency import run_in_threadpool

from src.config import settings

logger = logging.getLogger(__name__)


class SupabaseServiceError(RuntimeError):
    """Base exception for Supabase dependency failures."""


class SupabaseUnavailableError(SupabaseServiceError):
    """Raised when Supabase is not configured or cannot be reached."""


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

    async def _execute_with_retry(self, operation: str, func):
        """Run sync Supabase calls off the event loop with timeout and retry."""
        if not self.is_connected():
            raise SupabaseUnavailableError("Supabase client is not configured")

        last_error: Exception | None = None
        for attempt in range(settings.supabase_max_retries + 1):
            try:
                return await asyncio.wait_for(
                    run_in_threadpool(func),
                    timeout=settings.supabase_timeout_seconds,
                )
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning("Supabase %s timed out on attempt %s", operation, attempt + 1)
            except Exception as e:
                last_error = e
                logger.warning("Supabase %s failed on attempt %s: %s", operation, attempt + 1, str(e))

            if attempt < settings.supabase_max_retries:
                await asyncio.sleep(0.1 * (attempt + 1))

        raise SupabaseUnavailableError(f"Supabase {operation} failed: {last_error}") from last_error

    async def is_ready(self) -> bool:
        """Check whether Supabase can answer a lightweight query."""
        try:
            await self._execute_with_retry(
                "readiness check",
                lambda: self.client.table("user_profiles").select("user_id").limit(1).execute()
            )
            return True
        except Exception as e:
            logger.warning("Supabase readiness check failed: %s", str(e))
            return False
    
    async def get_user_profile_by_nis(self, nis: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile by NIS from user_profiles table.
        
        Args:
            nis: Student ID (NIS/NISN)
            
        Returns:
            User profile dict or None if not found
        """
        try:
            response = await self._execute_with_retry(
                "fetch user profile by NIS",
                lambda: self.client.table("user_profiles").select("*").eq("nis", nis).execute()
            )
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except SupabaseServiceError:
            raise
        except Exception as e:
            logger.error(f"✗ Error fetching user profile by NIS={nis}: {str(e)}")
            raise SupabaseUnavailableError(f"Failed to fetch user profile by NIS={nis}: {str(e)}") from e
    
    async def get_user_profile_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile by user_id from user_profiles table.
        
        Args:
            user_id: User ID (UUID from Supabase auth)
            
        Returns:
            User profile dict or None if not found
        """
        try:
            response = await self._execute_with_retry(
                "fetch user profile by user_id",
                lambda: self.client.table("user_profiles").select("*").eq("user_id", user_id).execute()
            )
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except SupabaseServiceError:
            raise
        except Exception as e:
            logger.error(f"✗ Error fetching user profile by user_id={user_id}: {str(e)}")
            raise SupabaseUnavailableError(f"Failed to fetch user profile by user_id={user_id}: {str(e)}") from e
    
    async def get_student_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve student info by user_id using RPC function.
        
        Args:
            user_id: User ID to look up
            
        Returns:
            Student data dict or None if not found
        """
        try:
            response = await self._execute_with_retry(
                "fetch student by user_id",
                lambda: self.client.rpc(
                    "get_student_by_user_id",
                    {"p_user_id": user_id}
                ).execute()
            )
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except SupabaseServiceError:
            raise
        except Exception as e:
            logger.error(f"✗ Error fetching student by user_id={user_id}: {str(e)}")
            raise SupabaseUnavailableError(f"Failed to fetch student by user_id={user_id}: {str(e)}") from e


# Global Supabase service instance
supabase_service = SupabaseService()
