"""
PostgreSQL with pgvector Vector Store Implementation

Supports both IVF (ivfflat) and HNSW indexing strategies.
"""

from typing import List, Dict, Tuple, Optional
import psycopg2
from psycopg2.extras import execute_values
from config.settings import POSTGRESQL_CONFIG
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.vector_stores.base import VectorStore


class PostgresVectorStore(VectorStore):
    """
    PostgreSQL with pgvector vector store implementation.
    
    Supports:
    - IVF (ivfflat): Inverted file index for approximate nearest neighbor
    - HNSW: Hierarchical Navigable Small World graphs
    """
    
    def __init__(
        self, 
        table_name: str = "documents", 
        index_type: str = "HNSW",
        dimension: int = 1536
    ):
        """
        Initialize PostgreSQL Vector Store
        
        Args:
            table_name: Name of the table
            index_type: Type of index ("IVF" or "HNSW")
            dimension: Vector dimension (default 1536 for OpenAI embeddings)
        """
        self.table_name = table_name
        self.index_type = index_type.upper()
        self.dimension = dimension
        self.conn = None
        self.cursor = None
        
        # Get index parameters from config
        self.ivf_config = POSTGRESQL_CONFIG.get("ivf_params", {"lists": 100, "probes": 10})
        self.hnsw_config = POSTGRESQL_CONFIG.get("hnsw_params", {"m": 16, "ef_construction": 64})
        
        self._connect()
        self._create_table()
    
    def _connect(self):
        """Connect to PostgreSQL database"""
        try:
            # Extract connection parameters (exclude index params)
            conn_params = {
                k: v for k, v in POSTGRESQL_CONFIG.items() 
                if k not in ["ivf_params", "hnsw_params"]
            }
            
            self.conn = psycopg2.connect(**conn_params)
            self.cursor = self.conn.cursor()
            
            # Enable pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
            
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            raise
    
    def _create_table(self):
        """Create documents table with vector column"""
        try:
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(100) UNIQUE,
                    text TEXT,
                    embedding vector({self.dimension}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()
            logger.info(f"Created table: {self.table_name}")
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            raise
    
    def _create_index(self, vector_count: int):
        """
        Create vector index based on index_type.
        
        Args:
            vector_count: Number of vectors (used to adjust IVF lists parameter)
        """
        try:
            index_name = f"{self.table_name}_{self.index_type.lower()}_idx"
            
            # Drop existing index if any
            self.cursor.execute(f"DROP INDEX IF EXISTS {index_name};")
            
            if self.index_type == "IVF":
                # Adjust lists based on data size (rule of thumb: sqrt(n) to n/1000)
                lists = min(self.ivf_config["lists"], max(1, int(vector_count ** 0.5)))
                
                self.cursor.execute(f"""
                    CREATE INDEX {index_name} 
                    ON {self.table_name} 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {lists});
                """)
                logger.info(f"Created IVF index with lists={lists}")
                
            elif self.index_type == "HNSW":
                m = self.hnsw_config["m"]
                ef_construction = self.hnsw_config["ef_construction"]
                
                self.cursor.execute(f"""
                    CREATE INDEX {index_name} 
                    ON {self.table_name} 
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction});
                """)
                logger.info(f"Created HNSW index with m={m}, ef_construction={ef_construction}")
            
            self.conn.commit()
            logger.info(f"Created {self.index_type} index on {self.table_name}")
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            self.conn.rollback()
    
    def store(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict] = None,
        chunk_ids: List[str] = None
    ) -> Dict:
        """
        Store vectors in PostgreSQL
        
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
            
            data = []
            for i, (chunk_id, text, embedding) in enumerate(zip(chunk_ids, texts, embeddings)):
                meta = json.dumps(metadata[i]) if metadata else json.dumps({"index": i})
                data.append((chunk_id, text, embedding, meta))
            
            query = f"""
                INSERT INTO {self.table_name} (chunk_id, text, embedding, metadata)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """
            
            execute_values(self.cursor, query, data)
            self.conn.commit()
            
            insert_time = time.time() - start_time
            
            # Create index after data load
            index_start = time.time()
            self._create_index(len(texts))
            index_time = time.time() - index_start
            
            total_time = time.time() - start_time
            
            stats = {
                "vectors_stored": len(texts),
                "insert_time_seconds": round(insert_time, 3),
                "index_build_time_seconds": round(index_time, 3),
                "total_time_seconds": round(total_time, 3),
                "index_type": self.index_type,
                "table_name": self.table_name
            }
            
            logger.info(f"Successfully stored {len(texts)} vectors in PostgreSQL ({total_time:.2f}s)")
            return stats
            
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            self.conn.rollback()
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        try:
            # Set search parameters based on index type
            if self.index_type == "IVF":
                probes = self.ivf_config.get("probes", 10)
                self.cursor.execute(f"SET ivfflat.probes = {probes};")
            elif self.index_type == "HNSW":
                ef_search = self.hnsw_config.get("ef_search", 40)
                self.cursor.execute(f"SET hnsw.ef_search = {ef_search};")
            
            query = f"""
                SELECT text, 1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            
            embedding_str = str(query_embedding)
            self.cursor.execute(query, (embedding_str, embedding_str, top_k))
            results = self.cursor.fetchall()
            
            return results
            
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
            
            # Set search parameters based on index type
            if self.index_type == "IVF":
                probes = self.ivf_config.get("probes", 10)
                self.cursor.execute(f"SET ivfflat.probes = {probes};")
            elif self.index_type == "HNSW":
                ef_search = self.hnsw_config.get("ef_search", 40)
                self.cursor.execute(f"SET hnsw.ef_search = {ef_search};")
            
            query = f"""
                SELECT chunk_id, 1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            
            embedding_str = str(query_embedding)
            self.cursor.execute(query, (embedding_str, embedding_str, top_k))
            results = self.cursor.fetchall()
            
            search_time = time.time() - start_time
            
            chunk_ids = [row[0] for row in results]
            scores = [float(row[1]) for row in results]
            
            return chunk_ids, scores, search_time
            
        except Exception as e:
            logger.error(f"Error in search_with_ids: {str(e)}")
            return [], [], 0.0
    
    def delete(self) -> None:
        """Delete all data from table"""
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name};")
            self.conn.commit()
            logger.info(f"Deleted table: {self.table_name}")
        except Exception as e:
            logger.error(f"Error deleting table: {str(e)}")
    
    def clear(self) -> None:
        """Clear all vectors from table without dropping it"""
        try:
            self.cursor.execute(f"TRUNCATE TABLE {self.table_name};")
            self.conn.commit()
            logger.info(f"Cleared all vectors from table: {self.table_name}")
        except Exception as e:
            logger.error(f"Error clearing table: {str(e)}")
            self.conn.rollback()
    
    def get_stats(self) -> Dict:
        """Get table statistics"""
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            count = self.cursor.fetchone()[0]
            
            # Check if index exists
            index_name = f"{self.table_name}_{self.index_type.lower()}_idx"
            self.cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = %s AND indexname = %s;
            """, (self.table_name, index_name))
            has_index = self.cursor.fetchone() is not None
            
            return {
                "table_name": self.table_name,
                "index_type": self.index_type,
                "total_vectors": count,
                "has_index": has_index,
                "dimension": self.dimension
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
            placeholders = ','.join(['%s'] * len(chunk_ids))
            query = f"""
                SELECT chunk_id, text 
                FROM {self.table_name} 
                WHERE chunk_id IN ({placeholders});
            """
            self.cursor.execute(query, tuple(chunk_ids))
            results = self.cursor.fetchall()
            return {row[0]: row[1] for row in results}
        except Exception as e:
            logger.error(f"Error fetching chunks: {str(e)}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("PostgreSQL connection closed")
