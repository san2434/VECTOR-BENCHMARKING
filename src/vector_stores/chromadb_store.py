"""
ChromaDB Vector Store Implementation
Supports both IVF and HNSW index types
"""

from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.vector_stores.base import VectorStore
from config.settings import CHROMADB_CONFIG


class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector store implementation
    
    Note: ChromaDB primarily uses HNSW internally. For IVF-like behavior,
    we adjust HNSW parameters to simulate different performance characteristics.
    """
    
    def __init__(
        self, 
        collection_name: str = "rag_benchmark",
        index_type: str = "HNSW"
    ):
        """
        Initialize ChromaDB Vector Store
        
        Args:
            collection_name: Name of the collection
            index_type: "IVF" or "HNSW" (affects HNSW parameters)
        """
        self.collection_name = collection_name
        self.index_type = index_type.upper()
        self.chunk_id_map = {}  # Map internal IDs to chunk IDs
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=CHROMADB_CONFIG["persist_directory"]
        )
        
        # Configure collection metadata based on index type
        if self.index_type == "HNSW":
            # Standard HNSW settings - higher quality, more memory
            collection_metadata = {
                "hnsw:space": CHROMADB_CONFIG["hnsw_space"],
                "hnsw:M": CHROMADB_CONFIG["hnsw_m"],
                "hnsw:construction_ef": CHROMADB_CONFIG["hnsw_ef_construction"],
                "hnsw:search_ef": CHROMADB_CONFIG["hnsw_ef_search"],
            }
        else:  # IVF-like settings
            # Lower M and ef values to simulate IVF-like behavior (faster, less accurate)
            collection_metadata = {
                "hnsw:space": CHROMADB_CONFIG["hnsw_space"],
                "hnsw:M": 8,  # Lower connectivity
                "hnsw:construction_ef": 50,  # Faster construction
                "hnsw:search_ef": CHROMADB_CONFIG["ivf_nprobe"],  # Fewer candidates
            }
        
        # Delete existing collection if it exists (for clean benchmarking)
        try:
            self.client.delete_collection(name=collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata=collection_metadata
        )
        
        logger.info(f"Created ChromaDB collection: {collection_name} with {index_type} settings")
    
    def store(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict] = None,
        chunk_ids: List[str] = None
    ) -> Dict:
        """
        Store vectors in ChromaDB
        
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
            
            # Store mapping
            for i, cid in enumerate(chunk_ids):
                self.chunk_id_map[cid] = i
            
            # Prepare metadata
            if metadata is None:
                metadata = []
            
            metadatas = []
            for i, text in enumerate(texts):
                meta = metadata[i] if i < len(metadata) else {}
                meta["chunk_id"] = chunk_ids[i]
                meta["text_preview"] = text[:200]
                metadatas.append(meta)
            
            # Store in ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
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
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
            
            search_time = time.time() - start_time
            
            # Extract results
            ids = results['ids'][0]
            documents = results['documents'][0]
            distances = results['distances'][0]
            
            # Convert distances to similarity scores (cosine distance to similarity)
            # ChromaDB returns squared L2 distance for cosine, so we convert
            similarities = [1 - d for d in distances]
            
            return [
                (chunk_id, sim, doc) 
                for chunk_id, sim, doc in zip(ids, similarities, documents)
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
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances"]
        )
        
        search_time = time.time() - start_time
        
        ids = results['ids'][0]
        distances = results['distances'][0]
        similarities = [1 - d for d in distances]
        
        return ids, similarities, search_time
    
    def delete(self) -> None:
        """Delete the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "index_type": self.index_type,
                "total_vectors": count,
                "path": CHROMADB_CONFIG["persist_directory"]
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
