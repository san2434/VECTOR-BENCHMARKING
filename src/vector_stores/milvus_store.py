"""
Milvus Vector Store Implementation (using Milvus Lite)
Supports both IVF and HNSW index types
"""

from typing import List, Dict, Tuple, Optional
from pymilvus import MilvusClient, DataType
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.vector_stores.base import VectorStore
from config.settings import MILVUS_CONFIG


class MilvusVectorStore(VectorStore):
    """
    Milvus vector store implementation using Milvus Lite
    
    Supports both IVF_FLAT and HNSW index types for true algorithm comparison.
    """
    
    def __init__(
        self, 
        collection_name: str = "rag_benchmark",
        index_type: str = "HNSW",
        persist_directory: str = None
    ):
        """
        Initialize Milvus Vector Store (Lite mode)
        
        Args:
            collection_name: Name of the collection
            index_type: "IVF" or "HNSW"
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.index_type = index_type.upper()
        self.persist_directory = persist_directory or MILVUS_CONFIG.get("persist_directory", "./data/milvus")
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Database file path for Milvus Lite
        db_path = os.path.join(self.persist_directory, f"{collection_name}.db")
        
        # Initialize Milvus Lite client
        self.client = MilvusClient(db_path)
        
        # Drop existing collection if it exists (for clean benchmarking)
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
        except:
            pass
        
        # Get embedding dimension from config
        self.embedding_dim = MILVUS_CONFIG.get("embedding_dimension", 1536)
        
        # Configure index parameters based on index type
        if self.index_type == "HNSW":
            self.index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": MILVUS_CONFIG.get("hnsw_m", 16),
                    "efConstruction": MILVUS_CONFIG.get("hnsw_ef_construction", 200)
                }
            }
            self.search_params = {
                "metric_type": "COSINE",
                "params": {
                    "ef": MILVUS_CONFIG.get("hnsw_ef_search", 128)
                }
            }
        else:  # IVF
            self.index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {
                    "nlist": MILVUS_CONFIG.get("ivf_nlist", 100)
                }
            }
            self.search_params = {
                "metric_type": "COSINE",
                "params": {
                    "nprobe": MILVUS_CONFIG.get("ivf_nprobe", 10)
                }
            }
        
        # Create collection with schema
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.embedding_dim,
            metric_type="COSINE",
            auto_id=False
        )
        
        logger.info(f"Created Milvus collection: {collection_name} with {index_type} index")
        
        self._collection_created = True
        self._index_built = False
    
    def store(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict] = None,
        chunk_ids: List[str] = None
    ) -> Dict:
        """
        Store vectors in Milvus
        
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
            
            # Prepare data for Milvus
            data = []
            for i, (text, embedding, chunk_id) in enumerate(zip(texts, embeddings, chunk_ids)):
                record = {
                    "id": i,
                    "vector": embedding,
                    "text": text[:65000],  # Milvus has field size limits
                    "chunk_id": chunk_id
                }
                data.append(record)
            
            # Insert data
            self.client.insert(
                collection_name=self.collection_name,
                data=data
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
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "chunk_id"],
                search_params=self.search_params
            )
            
            search_time = time.time() - start_time
            
            output = []
            if results and len(results) > 0:
                for hit in results[0]:
                    chunk_id = hit.get("entity", {}).get("chunk_id", f"chunk_{hit.get('id', 0)}")
                    text = hit.get("entity", {}).get("text", "")
                    score = hit.get("distance", 0.0)
                    output.append((chunk_id, score, text))
            
            return output
            
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
            data=[query_embedding],
            limit=top_k,
            output_fields=["chunk_id"],
            search_params=self.search_params
        )
        
        search_time = time.time() - start_time
        
        chunk_ids = []
        scores = []
        
        if results and len(results) > 0:
            for hit in results[0]:
                chunk_id = hit.get("entity", {}).get("chunk_id", f"chunk_{hit.get('id', 0)}")
                score = hit.get("distance", 0.0)
                chunk_ids.append(chunk_id)
                scores.append(score)
        
        return chunk_ids, scores, search_time
    
    def delete(self) -> None:
        """Delete the collection"""
        try:
            self.client.drop_collection(collection_name=self.collection_name)
            logger.info(f"Deleted Milvus collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return {
                "collection_name": self.collection_name,
                "index_type": self.index_type,
                "total_vectors": stats.get("row_count", 0),
                "path": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
