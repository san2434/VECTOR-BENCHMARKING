"""
Vector Store Factory

Creates vector store instances for all benchmark configurations.
"""

from typing import Dict, Optional
from config.settings import BENCHMARK_CONFIGURATIONS, CHROMADB_CONFIG, POSTGRESQL_CONFIG, QDRANT_CONFIG, MILVUS_CONFIG
import logging

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """
    Factory class to create vector store instances for benchmarking.
    
    Supports creating stores for all 6 configurations:
    - Pinecone (IVF, HNSW)
    - ChromaDB (IVF, HNSW)
    - PostgreSQL (IVF, HNSW)
    """
    
    @staticmethod
    def create(
        db_type: str, 
        index_type: str, 
        collection_name: str = "benchmark",
        **kwargs
    ):
        """
        Create a vector store instance.
        
        Args:
            db_type: Database type ("pinecone", "chromadb", "postgresql")
            index_type: Index type ("IVF" or "HNSW")
            collection_name: Name for the collection/index/table
            **kwargs: Additional arguments passed to the store constructor
            
        Returns:
            Vector store instance
        """
        db_type = db_type.lower()
        index_type = index_type.upper()
        
        if db_type == "pinecone":
            from src.vector_stores.pinecone_store import PineconeVectorStore
            index_name = f"{collection_name}_{index_type.lower()}"
            return PineconeVectorStore(
                index_name=index_name,
                index_type=index_type,
                **kwargs
            )
            
        elif db_type == "chromadb":
            from src.vector_stores.chromadb_store import ChromaDBVectorStore
            collection = f"{collection_name}_{index_type.lower()}"
            return ChromaDBVectorStore(
                collection_name=collection,
                index_type=index_type,
                persist_directory=CHROMADB_CONFIG.get("persist_directory", "./data/chromadb"),
                **kwargs
            )
            
        elif db_type == "postgresql":
            from src.vector_stores.postgres_store import PostgresVectorStore
            table_name = f"{collection_name}_{index_type.lower()}"
            return PostgresVectorStore(
                table_name=table_name,
                index_type=index_type,
                **kwargs
            )
        
        elif db_type == "qdrant":
            from src.vector_stores.qdrant_store import QdrantVectorStore
            coll_name = f"{collection_name}_{index_type.lower()}"
            return QdrantVectorStore(
                collection_name=coll_name,
                index_type=index_type,
                persist_directory=QDRANT_CONFIG.get("persist_directory", "./data/qdrant"),
                **kwargs
            )
        
        elif db_type == "milvus":
            from src.vector_stores.milvus_store import MilvusVectorStore
            coll_name = f"{collection_name}_{index_type.lower()}"
            return MilvusVectorStore(
                collection_name=coll_name,
                index_type=index_type,
                persist_directory=MILVUS_CONFIG.get("persist_directory", "./data/milvus"),
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown database type: {db_type}")
    
    @staticmethod
    def create_from_config(config: Dict, collection_name: str = "benchmark", **kwargs):
        """
        Create a vector store from a benchmark configuration dict.
        
        Args:
            config: Configuration dict with 'db' and 'index_type' keys
            collection_name: Base name for the collection
            **kwargs: Additional arguments
            
        Returns:
            Vector store instance
        """
        return VectorStoreFactory.create(
            db_type=config["db"],
            index_type=config["index_type"],
            collection_name=collection_name,
            **kwargs
        )
    
    @staticmethod
    def create_all_stores(collection_name: str = "benchmark", **kwargs) -> Dict:
        """
        Create all 10 benchmark vector stores.
        
        Args:
            collection_name: Base name for collections
            **kwargs: Additional arguments passed to all stores
            
        Returns:
            Dict mapping config name to store instance
        """
        stores = {}
        
        for config in BENCHMARK_CONFIGURATIONS:
            name = config["name"]
            try:
                stores[name] = VectorStoreFactory.create_from_config(
                    config, 
                    collection_name=collection_name,
                    **kwargs
                )
                logger.info(f"Created store: {name}")
            except Exception as e:
                logger.error(f"Failed to create store {name}: {str(e)}")
                stores[name] = None
        
        return stores
    
    @staticmethod
    def get_available_configurations():
        """Get list of available benchmark configurations."""
        return [
            {
                "name": config["name"],
                "db": config["db"],
                "index_type": config["index_type"]
            }
            for config in BENCHMARK_CONFIGURATIONS
        ]


def create_vector_store(config_name: str, collection_name: str = "benchmark", **kwargs):
    """
    Convenience function to create a vector store by configuration name.
    
    Args:
        config_name: Name of the configuration (e.g., "pinecone_ivf", "chromadb_hnsw")
        collection_name: Base name for the collection
        **kwargs: Additional arguments
        
    Returns:
        Vector store instance
    """
    # Find matching configuration
    config = None
    for c in BENCHMARK_CONFIGURATIONS:
        if c["name"] == config_name:
            config = c
            break
    
    if config is None:
        available = [c["name"] for c in BENCHMARK_CONFIGURATIONS]
        raise ValueError(f"Unknown configuration: {config_name}. Available: {available}")
    
    return VectorStoreFactory.create_from_config(config, collection_name, **kwargs)
