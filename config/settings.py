"""
Configuration Settings for RAG Vector Benchmarking System
Supports 6 configurations: 3 Vector DBs × 2 Index Types

Evaluation Metrics:
- Retrieval: Recall@k, MRR, Context Precision, NDCG@k
- Response: Exact Match, F1 Score, Semantic Similarity
"""

import os
from typing import Literal, List, Dict
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.0  # Deterministic for benchmarking

# =============================================================================
# DOCUMENT CONFIGURATION
# =============================================================================
# Place your document in the project root or specify full path
DOCUMENT_PATH = "document.pdf"  # <-- CHANGE THIS TO YOUR DOCUMENT
SUPPORTED_FORMATS = [".pdf", ".txt", ".md"]

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks (10%)
MIN_CHUNK_SIZE = 100      # Minimum chunk size to keep

# =============================================================================
# VECTOR DATABASE CONFIGURATIONS
# =============================================================================

# All 10 configurations to benchmark (5 DBs × 2 index types)
BENCHMARK_CONFIGURATIONS: List[Dict] = [
    # Cloud-based
    {"db": "pinecone", "index_type": "IVF", "name": "pinecone_ivf"},
    {"db": "pinecone", "index_type": "HNSW", "name": "pinecone_hnsw"},
    # Local databases
    {"db": "chromadb", "index_type": "IVF", "name": "chromadb_ivf"},
    {"db": "chromadb", "index_type": "HNSW", "name": "chromadb_hnsw"},
    {"db": "postgres", "index_type": "IVF", "name": "postgres_ivf"},
    {"db": "postgres", "index_type": "HNSW", "name": "postgres_hnsw"},
    {"db": "qdrant", "index_type": "IVF", "name": "qdrant_ivf"},
    {"db": "qdrant", "index_type": "HNSW", "name": "qdrant_hnsw"},
    {"db": "milvus", "index_type": "IVF", "name": "milvus_ivf"},
    {"db": "milvus", "index_type": "HNSW", "name": "milvus_hnsw"},
]

# Pinecone Configuration
PINECONE_CONFIG = {
    "api_key": PINECONE_API_KEY,
    "environment": PINECONE_ENVIRONMENT,
    "index_prefix": "rag-benchmark",
    "metric": "cosine",
    # IVF-specific (Pinecone uses pod-based indexes for IVF-like behavior)
    "ivf_pod_type": "p1.x1",
    # HNSW-specific (Pinecone serverless uses HNSW)
    "hnsw_cloud": "aws",
    "hnsw_region": "us-east-1",
}

# ChromaDB Configuration
CHROMADB_CONFIG = {
    "persist_directory": "./data/chromadb",
    "collection_prefix": "rag_benchmark",
    # IVF parameters
    "ivf_nlist": 100,        # Number of clusters
    "ivf_nprobe": 10,        # Clusters to search
    # HNSW parameters (ChromaDB default)
    "hnsw_space": "cosine",
    "hnsw_m": 16,            # Max connections per node
    "hnsw_ef_construction": 200,
    "hnsw_ef_search": 50,
}

# PostgreSQL Configuration
POSTGRESQL_CONFIG = {
    "host": os.getenv("POSTGRESQL_HOST", "localhost"),
    "port": int(os.getenv("POSTGRESQL_PORT", 5432)),
    "user": os.getenv("POSTGRESQL_USER", "postgres"),
    "password": os.getenv("POSTGRESQL_PASSWORD", "password"),
    "database": os.getenv("POSTGRESQL_DATABASE", "vector_rag"),
    "table_prefix": "rag_benchmark",
    # IVF parameters (pgvector ivfflat)
    "ivf_params": {
        "lists": 100,        # Number of clusters
        "probes": 10,        # Clusters to search
    },
    # HNSW parameters (pgvector hnsw)
    "hnsw_params": {
        "m": 16,
        "ef_construction": 64,
        "ef_search": 40,
    },
}

# Qdrant Configuration (Local)
QDRANT_CONFIG = {
    "persist_directory": "./data/qdrant",
    "embedding_dimension": EMBEDDING_DIMENSION,
    # HNSW parameters
    "hnsw_m": 16,                    # Max connections per node
    "hnsw_ef_construction": 200,     # Construction time accuracy
    "hnsw_ef_search": 128,           # Search time accuracy
    # IVF-like parameters (simulated via HNSW tuning)
    "ivf_m": 8,                      # Lower connectivity for IVF-like behavior
    "ivf_ef_construction": 50,       # Faster construction
    "ivf_ef_search": 16,             # Fewer candidates
}

# Milvus Configuration (Local - Milvus Lite)
MILVUS_CONFIG = {
    "persist_directory": "./data/milvus",
    "embedding_dimension": EMBEDDING_DIMENSION,
    # HNSW parameters
    "hnsw_m": 16,
    "hnsw_ef_construction": 200,
    "hnsw_ef_search": 128,
    # IVF parameters (true IVF_FLAT)
    "ivf_nlist": 100,                # Number of clusters
    "ivf_nprobe": 10,                # Clusters to search
}

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
TOP_K_VALUES = [1, 3, 5, 10]  # k values for Recall@k
DEFAULT_TOP_K = 5             # Default retrieval count
RETRIEVAL_RUNS = 3            # Number of runs for averaging

# =============================================================================
# GROUND TRUTH & EVALUATION CONFIGURATION
# =============================================================================

# Ground Truth Files (you'll create these)
GROUND_TRUTH_DIR = "./data/ground_truth"
QUESTIONS_FILE = "questions.json"
HUMAN_ANSWERS_FILE = "human_answers.json"
RELEVANT_CHUNKS_FILE = "relevant_chunks.json"

# Evaluation Metrics to Calculate
RETRIEVAL_METRICS = [
    "recall_at_k",      # % of relevant docs in top-k
    "mrr",              # Mean Reciprocal Rank
    "context_precision", # Precision of retrieved context
    "ndcg_at_k",        # Normalized Discounted Cumulative Gain
]

RESPONSE_METRICS = [
    "exact_match",       # Exact string match
    "f1_score",          # Token-level F1
    "semantic_similarity", # Embedding similarity
]

# Evaluation Configuration (for benchmark suite)
EVALUATION_CONFIG = {
    "top_k_values": [1, 3, 5, 10],
    "default_top_k": 5,
    "retrieval_runs": 3,
    "warmup_queries": 3,
    "metrics": {
        "retrieval": RETRIEVAL_METRICS,
        "response": RESPONSE_METRICS,
    }
}

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================
BENCHMARK_RESULTS_DIR = "./results"
BENCHMARK_RUNS = 3            # Number of runs for statistical significance
WARMUP_QUERIES = 3            # Warmup queries before timing

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = "./logs"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_name(db: str, index_type: str) -> str:
    """Get standardized configuration name"""
    return f"{db}_{index_type.lower()}"

def get_all_config_names() -> List[str]:
    """Get all configuration names"""
    return [c["name"] for c in BENCHMARK_CONFIGURATIONS]

def get_config_by_name(name: str) -> Dict:
    """Get configuration by name"""
    for config in BENCHMARK_CONFIGURATIONS:
        if config["name"] == name:
            return config
    raise ValueError(f"Configuration not found: {name}")
