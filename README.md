# RAG Vector Benchmarking System

A comprehensive system for evaluating the efficiency and performance of different vector databases and indexing algorithms for Retrieval Augmented Generation (RAG).

## Features

- **Data Extraction**: Web scraping to extract unstructured text from internet sources
- **Embeddings**: Text chunking and vector embeddings using OpenAI's API
- **Vector Databases**: Integration with 3 different vector stores:
  - Pinecone (cloud-based)
  - ChromaDB (local)
  - PostgreSQL with pgvector
- **Indexing Algorithms**: Comparison of different indexing strategies:
  - IVF (Inverted File)
  - HNSW (Hierarchical Navigable Small World)
- **RAG Pipeline**: Complete retrieval and generation pipeline using GPT-4o mini
- **Benchmarking**: Comprehensive performance metrics including:
  - Storage time and throughput
  - Search latency and QPS
  - End-to-end query performance

## Project Structure

```
├── config/
│   └── settings.py          # Configuration and environment variables
├── src/
│   ├── data_extraction/
│   │   └── web_scraper.py  # Web scraping utilities
│   ├── embeddings/
│   │   ├── text_chunker.py # Text chunking logic
│   │   └── embedding_service.py # OpenAI embedding service
│   ├── vector_stores/
│   │   ├── base.py         # Abstract base class
│   │   ├── pinecone_store.py
│   │   ├── chromadb_store.py
│   │   └── postgres_store.py
│   ├── indexing/
│   │   └── index_strategies.py # IVF and HNSW implementations
│   ├── rag/
│   │   └── pipeline.py     # RAG pipeline
│   └── benchmark/
│       └── benchmark_suite.py # Benchmarking utilities
├── data/                    # Data directory for ChromaDB
├── results/                 # Benchmark results
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Installation

### 1. Clone and Setup Environment

```bash
cd "VECTOR BENCHMARKING"
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys and database credentials:
# - OPENAI_API_KEY: Your OpenAI API key
# - PINECONE_API_KEY: Your Pinecone API key
# - PostgreSQL credentials (if using)
```

### 4. Set Up PostgreSQL (Optional)

For PostgreSQL benchmarking:

```bash
# Install PostgreSQL and pgvector extension
brew install postgresql
# Create database
createdb vector_rag
# Connect and enable pgvector
psql vector_rag -c "CREATE EXTENSION vector;"
```

## Usage

### Basic Execution

```bash
python main.py
```

This will:
1. Extract text from configured URLs
2. Create embeddings for text chunks
3. Store in all three vector databases
4. Run benchmarks
5. Generate a comprehensive report in `results/`

### Using Individual Components

```python
from src.data_extraction.web_scraper import extract_text_from_urls
from src.embeddings.text_chunker import chunk_documents
from src.embeddings.embedding_service import create_embeddings
from src.vector_stores.chromadb_store import ChromaVectorStore
from src.rag.pipeline import RAGPipeline

# Extract data
urls = ["https://example.com"]
text = extract_text_from_urls(urls)

# Chunk and embed
chunks = chunk_documents([text])
embeddings = create_embeddings(chunks)

# Store in vector database
store = ChromaVectorStore("my_collection")
store.store(chunks, embeddings)

# Use RAG pipeline
from src.embeddings.embedding_service import EmbeddingService
embedding_service = EmbeddingService()
rag = RAGPipeline(store, embedding_service)

result = rag.query("What is AI?")
print(result['answer'])
```

## Configuration

Edit `config/settings.py` to customize:

- **URLs to scrape**: `URLS_TO_SCRAPE`
- **Embedding model**: `EMBEDDING_MODEL` (default: text-embedding-3-small)
- **Chunk size**: `CHUNK_SIZE` (default: 500 characters)
- **Chunk overlap**: `CHUNK_OVERLAP` (default: 50 characters)
- **Vector databases**: `DATABASES`
- **Indexing algorithms**: `INDEXING_ALGORITHMS`

## Benchmarking Metrics

The system measures:

### Storage Metrics
- Storage time (seconds)
- Throughput (items/second)
- Memory usage

### Search Metrics
- Average search latency (ms)
- Search throughput (QPS)
- Min/max/std deviation

### End-to-End Metrics
- Query completion time
- Retrieval time
- Generation time

## Results

Benchmark results are saved to `results/` as JSON files with timestamps:
- `benchmark_results_YYYYMMDD_HHMMSS.json`
- `complete_benchmark_report.json`

## Vector Database Comparison

| Database | Index Type | Local | Cloud | Scalability | Cost |
|----------|-----------|-------|-------|-------------|------|
| ChromaDB | Exact/IVF | ✓ | ✗ | Limited | Free |
| Pinecone | IVF/HNSW | ✗ | ✓ | High | Paid |
| PostgreSQL | IVF/HNSW | ✓ | ✓ | Medium | Free |

## Indexing Algorithm Comparison

| Algorithm | Speed | Memory | Accuracy | Use Case |
|-----------|-------|--------|----------|----------|
| IVF | Fast | Low | Good | Large-scale approximate search |
| HNSW | Very Fast | Higher | Very Good | Real-time applications |

## Troubleshooting

### OpenAI API Key Issues
- Ensure `OPENAI_API_KEY` is set in `.env`
- Check API quota and billing

### Pinecone Connection Issues
- Verify `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`
- Check network connectivity

### PostgreSQL Issues
- Ensure PostgreSQL is running
- Verify pgvector extension is installed
- Check connection credentials

### Embedding Errors
- Verify OpenAI API key
- Check rate limits
- Reduce batch size in `embedding_service.py`

## Dependencies

- **openai**: GPT-4o mini API
- **pinecone-client**: Pinecone integration
- **chromadb**: ChromaDB integration
- **psycopg2**: PostgreSQL driver
- **pgvector**: PostgreSQL vector support
- **numpy, scikit-learn**: Numerical operations
- **beautifulsoup4**: Web scraping
- **python-dotenv**: Environment configuration

## Future Enhancements

- [ ] Support for more embedding models
- [ ] Additional indexing algorithms (LSH, ScaNN)
- [ ] Support for other vector databases (Weaviate, Milvus)
- [ ] Distributed benchmarking
- [ ] Real-time performance monitoring
- [ ] Cost analysis and optimization

## License

MIT

## Support

For issues or questions, please create an issue in the repository.
