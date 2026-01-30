"""
Qdrant client service for face embedding storage and verification.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
import logging

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from src.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database."""
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection_name
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Qdrant client connection."""
        try:
            # Build connection parameters
            # Use url if protocol is present in host
            if "://" in settings.qdrant_host:
                self.client = QdrantClient(
                    url=settings.qdrant_host,
                    api_key=settings.qdrant_api_key or None
                )
            else:
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key or None,
                    https=settings.qdrant_https
                )
            
            # Ensure collection exists
            self._ensure_collection()
            logger.info(f"Qdrant client initialized - Collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            self.client = None
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if not self.client:
            return
        
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,  # 512 dimensions
                    distance=Distance.COSINE
                )
            )
            
            # Create payload index for user_id (faster filtering)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def is_connected(self) -> bool:
        """Check if Qdrant client is connected."""
        if not self.client:
            return False
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    async def enroll_user_embeddings(
        self,
        user_id: str,
        embeddings: List[np.ndarray],
        model_name: str = "arcface"
    ) -> Dict[str, Any]:
        """
        Enroll multiple face embeddings for a user.
        Deletes existing embeddings first, then inserts new ones.
        
        Args:
            user_id: User ID from JWT sub claim
            embeddings: List of face embedding vectors
            model_name: Name of the face recognition model used
            
        Returns:
            Dict with success status and counts
        """
        if not self.is_connected():
            logger.warning("Qdrant not connected. Cannot enroll embeddings.")
            return {
                "success": False,
                "inserted_count": 0,
                "message": "Qdrant not connected"
            }
        
        try:
            # Step 1: Delete existing embeddings for this user
            deleted = await self.delete_user_embeddings(user_id)
            logger.info(f"Deleted {deleted} existing embeddings for user_id {user_id}")
            
            # Step 2: Prepare points with payload
            enrolled_at = datetime.now(timezone.utc).isoformat()
            points = []
            
            for idx, embedding in enumerate(embeddings):
                point_id = str(uuid.uuid4())  # Unique ID per embedding
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "user_id": user_id,
                            "embedding_index": idx + 1,
                            "model_name": model_name,
                            "confidence_floor": settings.face_match_threshold,
                            "enrolled_at": enrolled_at
                        }
                    )
                )
            
            # Step 3: Upsert all points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"Enrolled {len(points)} embeddings for user_id {user_id}")
            
            return {
                "success": True,
                "inserted_count": len(points),
                "total_embeddings": len(points),
                "message": f"Enrolled {len(points)} face embeddings successfully"
            }
            
        except Exception as e:
            logger.error(f"Error during enrollment: {str(e)}")
            return {
                "success": False,
                "inserted_count": 0,
                "message": f"Enrollment failed: {str(e)}"
            }
    
    async def retrieve_user_embeddings(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all embeddings for a user by scrolling with filter.
        
        Args:
            user_id: User ID to retrieve embeddings for
            
        Returns:
            List of dicts with vector and payload
        """
        if not self.is_connected():
            logger.warning("Qdrant not connected. Cannot retrieve embeddings.")
            return []
        
        try:
            # Scroll through all points matching user_id
            result, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                with_vectors=True,
                with_payload=True,
                limit=100  # Max embeddings per user (10 expected, buffer for safety)
            )
            
            embeddings = []
            for point in result:
                embeddings.append({
                    "id": point.id,
                    "vector": np.array(point.vector),
                    "payload": point.payload
                })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {str(e)}")
            return []
    
    async def verify_face_1to1(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """
        1:1 Face verification - compare embedding against user's enrolled embeddings.
        
        Uses manual cosine similarity calculation (no ANN search).
        
        Args:
            user_id: User ID from JWT sub claim
            query_embedding: The face embedding to verify
            threshold: Minimum similarity threshold (default 0.6)
            
        Returns:
            Dict with verification result, or None if no embeddings found
        """
        if not self.is_connected():
            logger.warning("Qdrant not connected. Cannot verify face.")
            return None
        
        try:
            # Step 1: Retrieve all embeddings for this user
            user_embeddings = await self.retrieve_user_embeddings(user_id)
            
            if not user_embeddings:
                logger.info(f"No embeddings found for user_id {user_id}")
                return None
            
            # Step 2: Calculate cosine similarity for each
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            best_similarity = -1.0
            best_match_index = 0
            
            for emb_data in user_embeddings:
                stored_vector = emb_data["vector"]
                stored_norm = stored_vector / np.linalg.norm(stored_vector)
                
                # Cosine similarity
                similarity = float(np.dot(query_norm, stored_norm))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = emb_data["payload"].get("embedding_index", 0)
            
            # Step 3: Compare against threshold
            verified = best_similarity >= threshold
            
            # Convert similarity to confidence (cosine similarity is already 0-1 for normalized vectors)
            confidence = max(0.0, min(1.0, best_similarity))
            
            # Distance = 1 - similarity (for compatibility with old API)
            distance = 1.0 - confidence
            
            logger.info(
                f"1:1 Verification - user_id: {user_id}, "
                f"verified: {verified}, confidence: {confidence:.3f}, "
                f"best_index: {best_match_index}"
            )
            
            return {
                "verified": verified,
                "confidence": confidence,
                "distance": distance,
                "best_match_index": best_match_index
            }
            
        except Exception as e:
            logger.error(f"Error during 1:1 verification: {str(e)}")
            return None
    
    async def delete_user_embeddings(self, user_id: str) -> int:
        """
        Delete all embeddings for a user.
        
        Args:
            user_id: User ID to delete embeddings for
            
        Returns:
            Number of points deleted
        """
        if not self.is_connected():
            logger.warning("Qdrant not connected. Cannot delete embeddings.")
            return 0
        
        try:
            # First count existing points
            count_before = await self.get_user_embedding_count(user_id)
            
            # Delete all points matching user_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                wait=True
            )
            
            logger.info(f"Deleted {count_before} embeddings for user_id {user_id}")
            return count_before
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            return 0
    
    async def get_user_embedding_count(self, user_id: str) -> int:
        """
        Count embeddings for a user.
        
        Args:
            user_id: User ID to count embeddings for
            
        Returns:
            Number of embeddings
        """
        if not self.is_connected():
            return 0
        
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                exact=True
            )
            return result.count
            
        except Exception as e:
            logger.error(f"Error counting embeddings: {str(e)}")
            return 0


# Global Qdrant service instance
qdrant_service = QdrantService()
