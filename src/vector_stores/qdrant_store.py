"""
Qdrant Vector Store Implementation
Supports both IVF and HNSW index types
"""

from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    HnswConfigDiff, OptimizersConfigDiff
)
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.vector_stores.base import VectorStore
from config.settings import QDRANT_CONFIG


class QdrantVectorStore(VectorStore):
    """
    Qdrant vector store implementation
    
    Qdrant uses HNSW natively. For IVF-like behavior, we adjust HNSW 
    parameters to simulate different performance characteristics.
    """
    
    def __init__(
        self, 
        collection_name: str = "rag_benchmark",
        index_type: str = "HNSW",
        persist_directory: str = None
    ):
        """
        Initialize Qdrant Vector Store
        
        Args:
            collection_name: Name of the collection
            index_type: "IVF" or "HNSW" (affects HNSW parameters)
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.index_type = index_type.upper()
        self.persist_directory = persist_directory or QDRANT_CONFIG.get("persist_directory", "./data/qdrant")
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize Qdrant client with local storage
        self.client = QdrantClient(path=self.persist_directory)
        
        # Delete existing collection if it exists (for clean benchmarking)
        try:
            self.client.delete_collection(collection_name=collection_name)
        except:
            pass
        
        # Configure HNSW parameters based on index type
        if self.index_type == "HNSW":
            # Standard HNSW settings - higher quality, more memory
            hnsw_config = HnswConfigDiff(
                m=QDRANT_CONFIG.get("hnsw_m", 16),
                ef_construct=QDRANT_CONFIG.get("hnsw_ef_construction", 200),
            )
            self.ef_search = QDRANT_CONFIG.get("hnsw_ef_search", 128)
        else:  # IVF-like settings
            # Lower M and ef values to simulate IVF-like behavior
            hnsw_config = HnswConfigDiff(
                m=QDRANT_CONFIG.get("ivf_m", 8),
                ef_construct=QDRANT_CONFIG.get("ivf_ef_construction", 50),
            )
            self.ef_search = QDRANT_CONFIG.get("ivf_ef_search", 16)
        
        # Create collection with vector configuration
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=QDRANT_CONFIG.get("embedding_dimension", 1536),
                distance=Distance.COSINE
            ),
            hnsw_config=hnsw_config
        )
        
        logger.info(f"Created Qdrant collection: {collection_name} with {index_type} settings")
    
    def store(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict] = None,
        chunk_ids: List[str] = None
    ) -> Dict:
        """
        Store vectors in Qdrant
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata for each chunk
            chunk_ids: Optional custom chunk IDs
            
        Returns:
            Storage statistics
        """
        try:
            start_time = time.time()
            
            # Generate or use provided chunk IDs
            if chunk_ids is None:
                chunk_ids = [f"chunk_{i}" for i in range(len(texts))]
            
            # Prepare points for Qdrant
            points = []
            for i, (text, embedding, chunk_id) in enumerate(zip(texts, embeddings, chunk_ids)):
                payload = {
                    "text": text,
                    "chunk_id": chunk_id,
                    "text_preview": text[:200]
                }
                if metadata and i < len(metadata):
                    payload.update(metadata[i])
                
                points.append(PointStruct(
                    id=i,
                    vector=embedding,
                    payload=payload
                ))
            
            # Store in Qdrant (batch upsert)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            storage_time = time.time() - start_time
            
            stats = {
                "items_stored": len(texts),
                "storage_time_seconds": storage_time,
                "throughput_items_per_second": len(texts) / storage_time if storage_time > 0 else 0,
                "index_type": self.index_type,
            }
            
            logger.info(f"Stored {len(texts)} vectors in {storage_time:.2f}s")
            return stats
            
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            raise
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, similarity_score, text) tuples
        """
        try:
            start_time = time.time()
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                search_params={"ef": self.ef_search}
            )
            
            search_time = time.time() - start_time
            
            return [
                (hit.payload.get("chunk_id", f"chunk_{hit.id}"), hit.score, hit.payload.get("text", ""))
                for hit in results
            ]
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def search_with_ids(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> Tuple[List[str], List[float], float]:
        """
        Search and return chunk IDs (for evaluation)
        
        Returns:
            Tuple of (chunk_ids, scores, search_time)
        """
        start_time = time.time()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            search_params={"ef": self.ef_search}
        )
        
        search_time = time.time() - start_time
        
        chunk_ids = [hit.payload.get("chunk_id", f"chunk_{hit.id}") for hit in results]
        scores = [hit.score for hit in results]
        
        return chunk_ids, scores, search_time
    
    def delete(self) -> None:
        """Delete the collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "collection_name": self.collection_name,
                "index_type": self.index_type,
                "total_vectors": info.points_count,
                "path": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
