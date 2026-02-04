"""
Qdrant client service for face embedding storage and verification.

Uses AsyncQdrantClient for non-blocking I/O operations with FastAPI.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
import logging

import numpy as np
from qdrant_client import AsyncQdrantClient
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
    """
    Async service for interacting with Qdrant vector database.
    
    Handles face embedding storage, retrieval, and 1:1 verification.
    Uses lazy initialization to avoid blocking on startup.
    """
    
    def __init__(self):
        """Initialize service (client initialization is lazy)."""
        self._client: Optional[AsyncQdrantClient] = None
        self.collection_name = settings.qdrant_collection_name
    
    async def get_client(self) -> AsyncQdrantClient:
        """
        Get or create AsyncQdrantClient instance (lazy initialization).
        
        Returns:
            AsyncQdrantClient instance
            
        Raises:
            RuntimeError: If client initialization fails
        """
        if self._client is None:
            try:
                # Build connection based on host format
                if "://" in settings.qdrant_host:
                    # Full URL provided (e.g., https://qdrant.example.com)
                    self._client = AsyncQdrantClient(
                        url=settings.qdrant_host,
                        api_key=settings.qdrant_api_key or None
                    )
                else:
                    # Host + port provided
                    self._client = AsyncQdrantClient(
                        host=settings.qdrant_host,
                        port=settings.qdrant_port,
                        api_key=settings.qdrant_api_key or None,
                        https=settings.qdrant_https
                    )
                
                # Ensure collection exists
                await self._ensure_collection()
                logger.info(f"✓ Qdrant client initialized - Collection: {self.collection_name}")
                
            except Exception as e:
                logger.error(f"✗ Failed to initialize Qdrant client: {str(e)}")
                raise RuntimeError(f"Qdrant initialization failed: {str(e)}")
        
        return self._client
    
    async def _ensure_collection(self) -> None:
        """
        Create collection if it doesn't exist.
        
        Creates collection with:
        - 512-dimensional vectors (ArcFace)
        - Cosine distance metric
        - Indexed user_id field for fast filtering
        """
        if not self._client:
            return
        
        try:
            collections = await self._client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dim,  # 512 dimensions
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload index for user_id (faster filtering)
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="user_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                logger.info(f"✓ Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"✗ Failed to ensure collection: {str(e)}")
            raise
    
    async def is_connected(self) -> bool:
        """
        Check if Qdrant client is connected and responsive.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            client = await self.get_client()
            await client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant connection check failed: {str(e)}")
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
        This ensures users always have exactly the embeddings from their
        most recent enrollment.
        
        Args:
            user_id: User ID from JWT sub claim
            embeddings: List of face embedding vectors (numpy arrays)
            model_name: Name of the face recognition model used
            
        Returns:
            Dict with success status and counts:
            {
                "success": bool,
                "inserted_count": int,
                "total_embeddings": int,
                "message": str
            }
        """
        try:
            client = await self.get_client()
            
            # Step 1: Delete existing embeddings for this user
            deleted = await self.delete_user_embeddings(user_id)
            logger.info(f"Deleted {deleted} existing embeddings for user_id={user_id}")
            
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
            await client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"✓ Enrolled {len(points)} embeddings for user_id={user_id}")
            
            return {
                "success": True,
                "inserted_count": len(points),
                "total_embeddings": len(points),
                "message": f"Enrolled {len(points)} face embeddings successfully"
            }
            
        except Exception as e:
            logger.error(f"✗ Enrollment error for user_id={user_id}: {str(e)}")
            return {
                "success": False,
                "inserted_count": 0,
                "message": f"Enrollment failed: {str(e)}"
            }
    
    async def retrieve_user_embeddings(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all embeddings for a user.
        
        Args:
            user_id: User ID to retrieve embeddings for
            
        Returns:
            List of dicts with vector and payload:
            [
                {
                    "id": str,
                    "vector": np.ndarray,
                    "payload": dict
                },
                ...
            ]
        """
        try:
            client = await self.get_client()
            
            # Scroll through all points matching user_id
            result, _ = await client.scroll(
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
            logger.error(f"✗ Error retrieving embeddings for user_id={user_id}: {str(e)}")
            return []
    
    async def verify_face_1to1(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """
        Perform 1:1 face verification against user's enrolled embeddings.
        
        Uses manual cosine similarity calculation (no ANN search).
        This ensures we're only comparing against the authenticated
        user's embeddings, not the entire database.
        
        Args:
            user_id: User ID from JWT sub claim
            query_embedding: The face embedding to verify
            threshold: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            Verification result dict or None if no embeddings found:
            {
                "verified": bool,
                "confidence": float,
                "distance": float,
                "best_match_index": int
            }
        """
        try:
            # Step 1: Retrieve all embeddings for this user
            user_embeddings = await self.retrieve_user_embeddings(user_id)
            
            if not user_embeddings:
                logger.info(f"No embeddings found for user_id={user_id}")
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
            
            # Convert similarity to confidence (cosine similarity is already 0-1)
            confidence = max(0.0, min(1.0, best_similarity))
            
            # Distance = 1 - similarity (for compatibility)
            distance = 1.0 - confidence
            
            logger.info(
                f"1:1 Verification - user_id={user_id}, "
                f"verified={verified}, confidence={confidence:.3f}, "
                f"best_index={best_match_index}"
            )
            
            return {
                "verified": verified,
                "confidence": confidence,
                "distance": distance,
                "best_match_index": best_match_index
            }
            
        except Exception as e:
            logger.error(f"✗ Verification error for user_id={user_id}: {str(e)}")
            return None
    
    async def delete_user_embeddings(self, user_id: str) -> int:
        """
        Delete all embeddings for a user.
        
        Used during re-enrollment to clear old embeddings.
        
        Args:
            user_id: User ID to delete embeddings for
            
        Returns:
            Number of embeddings deleted
        """
        try:
            client = await self.get_client()
            
            # First count existing points
            count_before = await self.get_user_embedding_count(user_id)
            
            # Delete all points matching user_id
            await client.delete(
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
            
            logger.info(f"Deleted {count_before} embeddings for user_id={user_id}")
            return count_before
            
        except Exception as e:
            logger.error(f"✗ Error deleting embeddings for user_id={user_id}: {str(e)}")
            return 0
    
    async def get_user_embedding_count(self, user_id: str) -> int:
        """
        Count embeddings for a user.
        
        Args:
            user_id: User ID to count embeddings for
            
        Returns:
            Number of embeddings stored
        """
        try:
            client = await self.get_client()
            
            result = await client.count(
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
            logger.error(f"✗ Error counting embeddings for user_id={user_id}: {str(e)}")
            return 0


# Global Qdrant service instance
qdrant_service = QdrantService()
