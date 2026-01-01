# Architecture & Design

## System Overview

The RAG Vector Benchmarking System is designed to comprehensively evaluate different vector databases and indexing algorithms for retrieval-augmented generation tasks.

```
Data Extraction → Embedding → Storage → Indexing → Retrieval → Generation
     ↓              ↓           ↓         ↓           ↓           ↓
Web Scraper   Chunking       VectorDB  IVF/HNSW   VectorStore  GPT-4o Mini
             OpenAI API      Store                              
```

## Component Architecture

### 1. Data Extraction Layer (`src/data_extraction/`)

**Purpose**: Extract unstructured text from web sources

- `WebScraper`: Fetches and cleans HTML content
  - Uses BeautifulSoup4 for parsing
  - Handles errors gracefully
  - Supports multiple URLs

### 2. Embedding Layer (`src/embeddings/`)

**Purpose**: Convert text to vector representations

- `TextChunker`: Splits documents into manageable chunks
  - Character-based chunking with overlap
  - Sentence-based chunking option
  - Configurable chunk size

- `EmbeddingService`: Creates embeddings using OpenAI
  - Uses text-embedding-3-small model
  - Batch processing for efficiency
  - Returns 1536-dimensional vectors

### 3. Vector Storage Layer (`src/vector_stores/`)

**Purpose**: Persist vectors in different database systems

- `VectorStore` (Abstract Base Class)
  - `store()`: Insert vectors
  - `search()`: Query similar vectors
  - `delete()`: Remove data
  - `get_stats()`: Retrieve metrics

**Implementations**:

1. **ChromaDB** (`chromadb_store.py`)
   - Local, file-based storage
   - Built-in embedding support
   - Cosine distance metric
   - No external dependencies

2. **Pinecone** (`pinecone_store.py`)
   - Cloud-based vector database
   - Scalable to billions of vectors
   - Multiple index types supported
   - Managed infrastructure

3. **PostgreSQL** (`postgres_store.py`)
   - Traditional SQL database
   - pgvector extension for vectors
   - IVF and HNSW index support
   - Self-hosted or managed

### 4. Indexing Strategies (`src/indexing/`)

**Purpose**: Optimize vector search performance

- **IVF (Inverted File)**
  - Divides vectors into clusters
  - Probes nearest clusters first
  - Good memory efficiency
  - Trade-off: accuracy for speed

- **HNSW (Hierarchical Navigable Small World)**
  - Graph-based approach
  - Multi-layer hierarchical structure
  - Excellent search quality
  - Higher memory requirements

### 5. RAG Pipeline (`src/rag/`)

**Purpose**: Complete retrieval and generation workflow

- `RAGPipeline`
  - `retrieve()`: Find relevant documents
  - `generate()`: Create responses using GPT
  - `query()`: End-to-end processing
  - Timing instrumentation

### 6. Benchmarking Suite (`src/benchmark/`)

**Purpose**: Measure and compare performance

- `BenchmarkSuite`
  - `benchmark_storage()`: Measure insert performance
  - `benchmark_search()`: Measure query performance
  - `benchmark_end_to_end()`: Complete workflow timing
  - JSON result export

## Data Flow

```
1. EXTRACTION
   URLs → Web Scraper → Raw Text

2. PREPROCESSING
   Raw Text → Text Chunker → Chunks (500 chars, 50 overlap)

3. EMBEDDING
   Chunks → OpenAI API → Vectors (1536-dim)

4. STORAGE
   Vectors → ChromaDB / Pinecone / PostgreSQL

5. INDEXING
   Stored Vectors → IVF or HNSW Index

6. BENCHMARKING
   ├─ Storage: throughput, latency
   ├─ Search: QPS, latency, accuracy
   └─ E2E: complete pipeline timing

7. RESULTS
   → JSON Report with comparisons
```

## Configuration Management

`config/settings.py` centralizes all configuration:

```python
# Data Sources
URLS_TO_SCRAPE = [...]
CHUNK_SIZE = 500

# APIs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

# Databases
PINECONE_CONFIG = {...}
CHROMADB_PATH = "./data/chromadb"
POSTGRESQL_CONFIG = {...}

# Benchmarking
BENCHMARK_QUERIES_COUNT = 10
```

## Key Design Decisions

### 1. Modular Architecture
- Each component is independent
- Easy to swap implementations
- Facilitates testing and benchmarking

### 2. Abstract Base Classes
- `VectorStore` defines interface
- Different implementations behind same API
- Easy to add new vector databases

### 3. Error Handling
- Graceful degradation
- Logging at each level
- Continues if one DB fails

### 4. Batch Processing
- Embeddings processed in batches
- Vectors stored in batches
- Improves efficiency and throughput

### 5. Instrumentation
- Timing at every operation
- Memory usage tracking
- Detailed logging

## Performance Considerations

### Storage Optimization
- Batch inserts reduce overhead
- Efficient JSON serialization
- Connection pooling where applicable

### Search Optimization
- Query embedding cached where possible
- Multiple runs for statistical significance
- Configurable parameters (nprobe, ef)

### Memory Usage
- IVF: ~ (vector_size * n + centroid_size)
- HNSW: ~ (vector_size * n * max_m)
- ChromaDB: Full vectors in memory
- Pinecone: Vectors on remote servers

## Scalability

### Horizontal Scaling
- Pinecone: Automatic sharding
- PostgreSQL: Replication and sharding
- ChromaDB: Limited (single node)

### Vertical Scaling
- Batch size adjustable
- Memory-efficient index structures
- Approximate search methods

## Security Considerations

1. **API Keys**: Stored in `.env`, not in code
2. **Database Credentials**: Environment variables
3. **Data Privacy**: Local storage options available
4. **Rate Limiting**: Batch processing respects API limits

## Testing Strategy

Current implementation includes:
- Integration tests via main.py
- Example scenarios in examples.py
- Benchmark validation built-in

Recommended additions:
- Unit tests for each module
- Mock external APIs
- Load testing scenarios

## Future Enhancements

1. **Additional Vector DBs**
   - Weaviate
   - Milvus
   - Elasticsearch

2. **More Indexing Algorithms**
   - LSH (Locality Sensitive Hashing)
   - ScaNN
   - Product Quantization

3. **Advanced Features**
   - Approximate Nearest Neighbor optimization
   - Distributed training
   - Incremental indexing
   - Real-time monitoring

4. **Production Deployment**
   - Docker containerization
   - Kubernetes orchestration
   - REST API wrapper
   - Web dashboard
