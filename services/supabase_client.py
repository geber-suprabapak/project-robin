"""
Supabase client service for database operations.
"""

from typing import Optional, List, Dict, Any
from supabase import create_client, Client
import numpy as np
from config import settings
import logging

logger = logging.getLogger(__name__)


class SupabaseService:
    """Service for interacting with Supabase database."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.client: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Supabase client connection."""
        try:
            if not settings.supabase_url or not settings.supabase_key:
                logger.warning("Supabase credentials not configured. Database operations will be simulated.")
                return
            
            self.client = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=settings.supabase_service_role_key or settings.supabase_key
            )
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if Supabase client is connected."""
        return self.client is not None
    
    async def find_face_match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """
        Find matching face in database by comparing embeddings using pgvector.
        
        Args:
            embedding: Face embedding vector to match
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with student info if match found, None otherwise
        """
        if not self.is_connected():
            # Simulation mode - return mock data
            logger.warning("Supabase not connected. Returning simulated match.")
            return {
                "nis": "12345678",
                "nama": "Ahmad Rizki (SIMULATED)",
                "kelas": "XII IPA 1",
                "confidence": 0.95
            }
        
        try:
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Call the RPC function for vector similarity search
            response = self.client.rpc(
                "find_face_match",
                {
                    "query_embedding": embedding_list,
                    "match_threshold": threshold,
                    "match_count": 1  # Return only the best match
                }
            ).execute()
            
            if response.data and len(response.data) > 0:
                match = response.data[0]
                logger.info(
                    f"Match found - NIS: {match['nis']}, "
                    f"Confidence: {match['confidence']:.3f}, "
                    f"Distance: {match['distance']:.3f}"
                )
                return match
            
            logger.info("No matching face found in database")
            return None
            
        except Exception as e:
            logger.error(f"Error during face match search: {str(e)}")
            # Return None instead of re-raising to allow graceful handling
            return None

    
    async def get_student_by_nis(self, nis: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve student information by NIS.
        
        Args:
            nis: Student NIS (Nomor Induk Siswa)
            
        Returns:
            Student data dictionary or None if not found
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot fetch student data.")
            return None
        
        try:
            response = self.client.table("biodata_siswa").select("*").eq("nis", nis).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching student data: {str(e)}")
            return None
    
    async def store_face_embedding(
        self,
        nis: str,
        embedding: np.ndarray,
        user_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        quality_score: Optional[float] = None
    ) -> bool:
        """
        Store or update face embedding for a student.
        
        Args:
            nis: Student NIS
            embedding: Face embedding vector
            user_id: UUID of the user (optional)
            camera_id: Camera/kiosk identifier (optional)
            quality_score: Image quality score (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot store embedding.")
            return False
        
        try:
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Call the upsert RPC function
            response = self.client.rpc(
                "upsert_face_embedding",
                {
                    "p_nis": nis,
                    "p_user_id": user_id,
                    "p_embedding": embedding_list,
                    "p_camera_id": camera_id,
                    "p_quality_score": quality_score
                }
            ).execute()
            
            if response.data:
                logger.info(f"Successfully stored embedding for NIS {nis} (ID: {response.data})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error storing face embedding: {str(e)}")
            return False

    async def get_user_profile_by_nis(self, nis: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile by NIS from user_profiles table.
        
        Args:
            nis: Student NIS (Nomor Induk Siswa)
            
        Returns:
            User profile data dictionary or None if not found
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
            logger.error(f"Error fetching user profile: {str(e)}")
            return None
    
    async def enroll_student(
        self,
        nis: str,
        name: str,
        embedding: np.ndarray,
        class_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Enroll a student by storing their face embedding.
        
        Looks up user_id from user_profiles table by NIS, then stores
        the embedding in face_embeddings table.
        
        Args:
            nis: Student NIS
            name: Student full name (for logging)
            embedding: Face embedding vector (512-dim)
            class_name: Optional class name
            
        Returns:
            UUID of inserted record if successful, None otherwise
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot enroll student.")
            return None
        
        try:
            # Lookup user_id from user_profiles
            user_profile = await self.get_user_profile_by_nis(nis)
            user_id = user_profile.get("user_id") if user_profile else None
            
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Call the upsert RPC function with camera_id as "enrollment"
            response = self.client.rpc(
                "upsert_face_embedding",
                {
                    "p_nis": nis,
                    "p_user_id": user_id,
                    "p_embedding": embedding_list,
                    "p_camera_id": "enrollment",
                    "p_quality_score": None
                }
            ).execute()
            
            if response.data:
                record_id = str(response.data)
                logger.info(f"Successfully enrolled student {name} (NIS: {nis}, ID: {record_id})")
                return record_id
            
            logger.warning(f"Enrollment returned no data for NIS {nis}")
            return None
            
        except Exception as e:
            logger.error(f"Error enrolling student: {str(e)}")
            return None



# Global Supabase service instance
supabase_service = SupabaseService()
