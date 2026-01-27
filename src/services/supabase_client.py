"""
Supabase client service for database operations.
"""

from typing import Optional, List, Dict, Any
from supabase import create_client, Client
import numpy as np
from src.config import settings
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
    
    async def get_user_profile_by_nis(self, nis: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile by NIS from user_profiles table."""
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
    
    async def get_user_profile_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile by user_id from user_profiles table."""
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot fetch user profile.")
            return None
        
        try:
            response = self.client.table("user_profiles").select("*").eq("user_id", user_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching user profile by id: {str(e)}")
            return None
    
    async def get_student_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve student info by user_id using RPC function."""
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
            logger.error(f"Error fetching student by user_id: {str(e)}")
            return None
    
    async def find_face_match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """
        Find matching face in database and return complete student info.
        Uses find_face_match RPC to get user_id, then looks up student data server-side.
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Returning simulated match.")
            return {
                "nis": "12345678",
                "nama": "Ahmad Rizki (SIMULATED)",
                "kelas": "XII IPA 1",
                "confidence": 0.95
            }
        
        try:
            embedding_list = embedding.tolist()
            
            # Step 1: Find face match - returns user_id only
            response = self.client.rpc(
                "find_face_match",
                {
                    "query_embedding": embedding_list,
                    "match_threshold": threshold,
                    "match_count": 1
                }
            ).execute()
            
            if not response.data or len(response.data) == 0:
                logger.info("No matching face found in database")
                return None
            
            match = response.data[0]
            user_id = match["user_id"]
            confidence = match["confidence"]
            distance = match["distance"]
            
            # Step 2: Server-side lookup of student info
            student = await self.get_student_by_user_id(user_id)
            
            if not student:
                logger.warning(f"Face matched user_id {user_id} but no student profile found")
                return None
            
            result = {
                "user_id": user_id,
                "nis": student["nis"],
                "nama": student["nama"],
                "kelas": student.get("kelas"),
                "confidence": confidence,
                "distance": distance
            }
            
            logger.info(
                f"Match found - NIS: {result['nis']}, "
                f"Confidence: {confidence:.3f}, Distance: {distance:.3f}"
            )
            return result
            
        except Exception as e:
            logger.error(f"Error during face match search: {str(e)}")
            return None

    async def verify_face_1to1(
        self,
        user_id: str,
        embedding: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """
        1:1 Face verification - compare embedding against user's enrolled embeddings only.
        
        More secure and faster than 1:N matching since it only checks against
        the authenticated user's registered face embeddings.
        
        Args:
            user_id: The user_id from JWT 'sub' claim
            embedding: The query embedding to verify
            threshold: Minimum confidence threshold (default 0.6)
            
        Returns:
            Dict with verification result, or None if error/no embeddings found
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot verify face.")
            return None
        
        try:
            embedding_list = embedding.tolist()
            
            response = self.client.rpc(
                "verify_face_1to1",
                {
                    "p_user_id": user_id,
                    "query_embedding": embedding_list,
                    "match_threshold": threshold
                }
            ).execute()
            
            if not response.data or len(response.data) == 0:
                logger.info(f"No embeddings found for user_id {user_id}")
                return None
            
            result = response.data[0]
            verified = result.get("verified", False)
            confidence = result.get("confidence", 0.0)
            distance = result.get("distance", 1.0)
            best_match_index = result.get("best_match_index", 0)
            
            logger.info(
                f"1:1 Verification - user_id: {user_id}, "
                f"verified: {verified}, confidence: {confidence:.3f}, "
                f"distance: {distance:.3f}, best_index: {best_match_index}"
            )
            
            return {
                "verified": verified,
                "confidence": confidence,
                "distance": distance,
                "best_match_index": best_match_index
            }
            
        except Exception as e:
            logger.error(f"Error during 1:1 face verification: {str(e)}")
            return None

    async def delete_user_embeddings(self, user_id: str) -> int:
        """Delete all face embeddings for a user (for re-enrollment)."""
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot delete embeddings.")
            return 0
        
        try:
            response = self.client.rpc(
                "delete_user_embeddings",
                {"p_user_id": user_id}
            ).execute()
            
            deleted_count = response.data if response.data else 0
            logger.info(f"Deleted {deleted_count} embeddings for user_id {user_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            return 0
    
    async def insert_face_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
        embedding_index: int,
        quality_score: Optional[float] = None
    ) -> Optional[str]:
        """Insert a single face embedding for multi-image enrollment."""
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot insert embedding.")
            return None
        
        try:
            embedding_list = embedding.tolist()
            
            response = self.client.rpc(
                "insert_face_embedding",
                {
                    "p_user_id": user_id,
                    "p_embedding": embedding_list,
                    "p_image_index": embedding_index,
                    "p_quality_score": quality_score
                }
            ).execute()
            
            if response.data:
                return str(response.data)
            return None
        except Exception as e:
            logger.error(f"Error inserting embedding: {str(e)}")
            return None
    
    async def get_user_embedding_count(self, user_id: str) -> int:
        """Get the count of enrolled face embeddings for a user."""
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot get embedding count.")
            return 0
        
        try:
            response = self.client.rpc(
                "count_user_embeddings",
                {"p_user_id": user_id}
            ).execute()
            return response.data if response.data else 0
        except Exception as e:
            logger.error(f"Error getting embedding count: {str(e)}")
            return 0
    
    async def enroll_student_multi(
        self,
        nis: str,
        name: str,
        embeddings: List[np.ndarray],
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enroll a student with multiple face embeddings.
        Looks up user_id by NIS, deletes existing embeddings, then inserts all new ones.
        """
        if not self.is_connected():
            logger.warning("Supabase not connected. Cannot enroll student.")
            return {
                "success": False, 
                "user_id": None,
                "inserted_count": 0, 
                "total_embeddings": 0,
                "message": "Database not connected"
            }
        
        try:
            # Step 1: Lookup user_id from user_profiles by NIS
            user_profile = await self.get_user_profile_by_nis(nis)
            
            if not user_profile:
                logger.warning(f"No user profile found for NIS {nis}")
                return {
                    "success": False,
                    "user_id": None,
                    "inserted_count": 0,
                    "total_embeddings": 0,
                    "message": f"No user profile found for NIS {nis}. Student must register first."
                }
            
            user_id = user_profile.get("user_id")
            if not user_id:
                logger.warning(f"User profile for NIS {nis} has no user_id")
                return {
                    "success": False,
                    "user_id": None,
                    "inserted_count": 0,
                    "total_embeddings": 0,
                    "message": f"User profile for NIS {nis} is incomplete (no user_id)"
                }
            
            # Step 2: Delete existing embeddings first
            deleted = await self.delete_user_embeddings(user_id)
            logger.info(f"Deleted {deleted} existing embeddings for user_id {user_id}")
            
            # Step 3: Insert all new embeddings
            inserted_count = 0
            for idx, embedding in enumerate(embeddings):
                result = await self.insert_face_embedding(
                    user_id=user_id,
                    embedding=embedding,
                    embedding_index=idx + 1  # 1-based index
                )
                if result:
                    inserted_count += 1
            
            # Step 4: Get final count
            total = await self.get_user_embedding_count(user_id)
            
            logger.info(
                f"Enrolled student {name} (NIS: {nis}) with {inserted_count}/{len(embeddings)} embeddings"
            )
            
            return {
                "success": inserted_count > 0,
                "user_id": user_id,
                "inserted_count": inserted_count,
                "total_embeddings": total,
                "message": f"Enrolled {inserted_count} face images successfully"
            }
            
        except Exception as e:
            logger.error(f"Error in multi-enrollment: {str(e)}")
            return {
                "success": False,
                "user_id": None,
                "inserted_count": 0,
                "total_embeddings": 0,
                "message": f"Enrollment failed: {str(e)}"
            }


# Global Supabase service instance
supabase_service = SupabaseService()
