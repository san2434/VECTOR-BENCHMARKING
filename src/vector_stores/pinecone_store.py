"""
Pinecone Vector Store Implementation

Supports both IVF and HNSW-like indexing strategies.
Note: Pinecone uses proprietary indexing - we simulate IVF/HNSW via pod types and parameters.
"""

from typing import List, Dict, Tuple, Optional
from pinecone import Pinecone, ServerlessSpec, PodSpec
from config.settings import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_CONFIG
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.vector_stores.base import VectorStore


class PineconeVectorStore(VectorStore):
    """
    Pinecone vector store implementation with IVF/HNSW-like support.
    
    Pinecone's indexing is managed internally, but we can influence behavior:
    - IVF-like: Use p1 pod type (optimized for throughput)
    - HNSW-like: Use s1 pod type (optimized for accuracy)
    """
    
    def __init__(
        self, 
        index_name: str, 
        index_type: str = "HNSW",
        metric: str = "cosine", 
        dimension: int = 1536,
        use_serverless: bool = True
    ):
        """
        Initialize Pinecone Vector Store
        
        Args:
            index_name: Name of the Pinecone index
            index_type: Type of index strategy ("IVF" or "HNSW")
            metric: Distance metric (cosine, euclidean, dotproduct)
            dimension: Dimension of vectors
            use_serverless: Use serverless spec (recommended for testing)
        """
        self.index_name = index_name
        self.index_type = index_type
        self.metric = metric
        self.dimension = dimension
        self.use_serverless = use_serverless
        self.chunk_id_mapping = {}  # Maps internal IDs to chunk IDs
        
        # Initialize Pinecone client (new API)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if doesn't exist
        self._create_index_if_needed()
        
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name} (index_type={index_type})")
    
    def _create_index_if_needed(self):
        """Create index if it doesn't exist with appropriate configuration"""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name} ({self.index_type})")
                
                if self.use_serverless:
                    # Serverless spec - good for testing
                    spec = ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                else:
                    # Pod-based spec with IVF/HNSW-like configuration
                    if self.index_type == "IVF":
                        # P1 pods are optimized for throughput (IVF-like)
                        spec = PodSpec(
                            environment=PINECONE_ENVIRONMENT,
                            pod_type="p1.x1",
                            pods=1,
                            replicas=1,
                            shards=1
                        )
                    else:  # HNSW
                        # S1 pods are optimized for accuracy (HNSW-like)
                        spec = PodSpec(
                            environment=PINECONE_ENVIRONMENT,
                            pod_type="s1.x1",
                            pods=1,
                            replicas=1,
                            shards=1
                        )
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=spec
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def store(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict] = None,
        chunk_ids: List[str] = None
    ) -> Dict:
        """
        Store vectors in Pinecone
        
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
            
            vectors = []
            for i, (chunk_id, text, embedding) in enumerate(zip(chunk_ids, texts, embeddings)):
                # Store mapping
                self.chunk_id_mapping[chunk_id] = i
                
                # Prepare metadata
                meta = metadata[i] if metadata else {}
                meta["text"] = text[:1000]  # Pinecone metadata size limit
                meta["chunk_id"] = chunk_id
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": meta
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            storage_time = time.time() - start_time
            
            stats = {
                "vectors_stored": len(vectors),
                "storage_time_seconds": round(storage_time, 3),
                "index_type": self.index_type,
                "index_name": self.index_name
            }
            
            logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone ({storage_time:.2f}s)")
            return stats
            
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        try:
            results = self.index.query(
                vector=query_embedding, 
                top_k=top_k, 
                include_metadata=True
            )
            return [
                (match['metadata'].get('text', ''), match['score']) 
                for match in results['matches']
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
        Search and return chunk IDs with timing information.
        
        Used for evaluation to compare retrieved chunks against ground truth.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (chunk_ids, similarity_scores, search_time_seconds)
        """
        try:
            start_time = time.time()
            
            results = self.index.query(
                vector=query_embedding, 
                top_k=top_k, 
                include_metadata=True
            )
            
            search_time = time.time() - start_time
            
            chunk_ids = [match['metadata'].get('chunk_id', match['id']) for match in results['matches']]
            scores = [match['score'] for match in results['matches']]
            
            return chunk_ids, scores, search_time
            
        except Exception as e:
            logger.error(f"Error in search_with_ids: {str(e)}")
            return [], [], 0.0
    
    def delete(self) -> None:
        """Delete the index"""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
    
    def clear(self) -> None:
        """Clear all vectors from the index"""
        try:
            self.index.delete(delete_all=True)
            self.chunk_id_mapping = {}
            logger.info(f"Cleared all vectors from Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "index_name": self.index_name,
                "index_type": self.index_type,
                "dimension": self.dimension,
                "total_vectors": stats.get('total_vector_count', 0),
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def get_chunk_texts(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        Retrieve text content for given chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            Dictionary mapping chunk_id to text content
        """
        try:
            results = self.index.fetch(ids=chunk_ids)
            return {
                id: vec.get('metadata', {}).get('text', '')
                for id, vec in results.get('vectors', {}).items()
            }
        except Exception as e:
            logger.error(f"Error fetching chunks: {str(e)}")
            return {}
