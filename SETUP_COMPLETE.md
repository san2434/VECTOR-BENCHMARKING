# Project Setup Complete ✅

## RAG Vector Benchmarking System

A comprehensive system for evaluating and comparing different vector databases and indexing algorithms for Retrieval Augmented Generation (RAG) applications.

---

## 📦 What's Been Created

### Core Components

1. **Data Extraction** (`src/data_extraction/`)
   - Web scraper for extracting text from URLs
   - Support for multiple sources
   - HTML parsing and cleaning

2. **Embeddings** (`src/embeddings/`)
   - Text chunking with configurable overlap
   - OpenAI embedding service integration
   - Batch processing for efficiency

3. **Vector Stores** (`src/vector_stores/`)
   - **ChromaDB**: Local, file-based vector store
   - **Pinecone**: Cloud-based scalable solution
   - **PostgreSQL**: SQL database with pgvector extension
   - Unified interface for all implementations

4. **Indexing Algorithms** (`src/indexing/`)
   - **IVF** (Inverted File): Fast approximate search
   - **HNSW** (Hierarchical Navigable Small World): High-quality search

5. **RAG Pipeline** (`src/rag/`)
   - Retrieval: Find relevant documents
   - Generation: Create responses using GPT-4o mini
   - End-to-end query processing

6. **Benchmarking** (`src/benchmark/`)
   - Storage performance metrics
   - Search latency and throughput
   - End-to-end pipeline benchmarking
   - JSON report generation

### Configuration & Documentation

- `config/settings.py` - Centralized configuration
- `.env.example` - Environment variables template
- `README.md` - Comprehensive documentation
- `QUICKSTART.md` - Quick setup guide
- `ARCHITECTURE.md` - System design and architecture

### Executable Files

- `main.py` - Main benchmarking script
- `examples.py` - Example usage scenarios
- `setup.sh` - Linux/Mac setup script
- `setup.bat` - Windows setup script

---

## 🚀 Quick Start

### 1. Setup Environment

**On macOS/Linux:**
```bash
cd "VECTOR BENCHMARKING"
bash setup.sh
```

**On Windows:**
```bash
cd "VECTOR BENCHMARKING"
setup.bat
```

### 2. Configure API Keys

```bash
# Edit .env with your credentials
nano .env  # or use your preferred editor

# Add at minimum:
OPENAI_API_KEY=your_openai_key_here
```

### 3. Run Benchmarking

```bash
python main.py
```

### 4. View Results

Results are saved in `results/complete_benchmark_report.json`

---

## 📊 What Gets Benchmarked

### Vector Databases
- ✅ **ChromaDB** (always works - local)
- 🔧 **Pinecone** (if configured)
- 📦 **PostgreSQL** (if running locally)

### Indexing Algorithms
- **IVF**: Fast approximate search
- **HNSW**: High-quality search results

### Metrics Collected
- Storage time and throughput (items/second)
- Search latency (ms/query)
- Queries per second (QPS)
- Statistical analysis (min, max, std dev)

---

## 📁 Project Structure

```
VECTOR BENCHMARKING/
├── config/
│   └── settings.py              # Configuration
├── src/
│   ├── data_extraction/         # Web scraping
│   ├── embeddings/              # Chunking & embeddings
│   ├── vector_stores/           # DB implementations
│   ├── indexing/                # IVF & HNSW
│   ├── rag/                     # RAG pipeline
│   └── benchmark/               # Benchmarking
├── data/                        # Data storage
├── results/                     # Benchmark results
├── main.py                      # Main entry point
├── examples.py                  # Example usage
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── setup.sh / setup.bat         # Setup scripts
├── README.md                    # Full documentation
├── QUICKSTART.md               # Quick start guide
└── ARCHITECTURE.md             # System design
```

---

## 🔧 Technologies Used

### APIs & Services
- **OpenAI**: GPT-4o mini and text-embedding-3-small
- **Pinecone**: Cloud vector database
- **PostgreSQL**: SQL database with pgvector

### Python Libraries
- **chromadb**: Vector database
- **openai**: LLM and embedding API
- **scikit-learn**: Machine learning algorithms
- **numpy**: Numerical computing
- **requests**: HTTP library
- **beautifulsoup4**: Web scraping

---

## ✨ Key Features

1. **Multiple Vector Databases**
   - Compare performance across 3 different systems
   - Unified interface for easy switching

2. **Indexing Algorithm Comparison**
   - IVF for speed
   - HNSW for quality
   - Analyze trade-offs

3. **Comprehensive Benchmarking**
   - Storage performance
   - Search speed and accuracy
   - End-to-end RAG pipeline timing

4. **Flexible Configuration**
   - Easy to modify URLs, chunk sizes, etc.
   - Support for multiple environments

5. **Detailed Reporting**
   - JSON results for analysis
   - Statistical metrics included

---

## 🎯 Next Steps

### Immediate (Required)
1. ✅ Edit `.env` with OpenAI API key
2. ✅ Run `python main.py`
3. ✅ Review results in `results/`

### Short Term
1. Modify `URLS_TO_SCRAPE` in `config/settings.py`
2. Test with different chunk sizes
3. Adjust embedding parameters

### Medium Term
1. Set up Pinecone (for cloud benchmarking)
2. Configure PostgreSQL (for self-hosted option)
3. Run comparative analysis

### Long Term
1. Deploy best-performing configuration
2. Integrate into production pipeline
3. Monitor performance over time

---

## 🐛 Troubleshooting

### Module Import Errors
```bash
# Ensure venv is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### OpenAI API Errors
- Verify API key in `.env`
- Check account at https://platform.openai.com/
- Ensure billing is set up

### Vector Database Errors
- ChromaDB: Works out of the box (local)
- Pinecone: Optional - will be skipped if not configured
- PostgreSQL: Optional - will be skipped if not running

---

## 📚 Documentation

- **README.md** - Full feature documentation
- **QUICKSTART.md** - Setup and basic usage
- **ARCHITECTURE.md** - System design and internals
- **examples.py** - Code examples and use cases

---

## 🤝 Integration Examples

### Use Individual Components

```python
from src.embeddings.embedding_service import create_embeddings
from src.vector_stores.chromadb_store import ChromaVectorStore

# Create embeddings
texts = ["Hello world", "Another document"]
embeddings = create_embeddings(texts)

# Store in vector database
store = ChromaVectorStore("my_collection")
store.store(texts, embeddings)

# Search
query_embedding = create_embeddings(["hello"])[0]
results = store.search(query_embedding, top_k=5)
```

### Build Custom RAG Pipeline

```python
from src.rag.pipeline import RAGPipeline
from src.embeddings.embedding_service import EmbeddingService
from src.vector_stores.chromadb_store import ChromaVectorStore

store = ChromaVectorStore("my_rag")
embedding_service = EmbeddingService()
rag = RAGPipeline(store, embedding_service)

# Process queries
result = rag.query("What is machine learning?")
print(result['answer'])
```

---

## 📊 Benchmark Output Example

```json
{
  "timestamp": "2024-12-31T06:00:00",
  "vector_stores": {
    "chromadb": {
      "storage": {
        "storage_time_seconds": 2.5,
        "throughput_items_per_second": 40
      },
      "search": {
        "avg_search_time_seconds": 0.012,
        "throughput_queries_per_second": 83.33
      }
    },
    "pinecone": {
      "storage": {...},
      "search": {...}
    }
  },
  "indexing_algorithms": {
    "IVF": {
      "build_time": 1.2,
      "avg_search_time": 0.008
    },
    "HNSW": {
      "build_time": 1.5,
      "avg_search_time": 0.006
    }
  }
}
```

---

## 🎓 Learning Resources

### Vector Databases
- ChromaDB Docs: https://docs.trychroma.com
- Pinecone Docs: https://docs.pinecone.io
- pgvector: https://github.com/pgvector/pgvector

### Indexing Algorithms
- IVF: Johnson et al. (2019) - "Billion-scale similarity search"
- HNSW: Malkov & Yashunin (2018) - "Efficient and robust approximate nearest neighbor search"

### RAG Systems
- OpenAI API: https://platform.openai.com/docs
- Vector DB Comparison: https://www.sicara.ai/blog/vector-databases

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🆘 Support

If you encounter issues:

1. Check QUICKSTART.md for setup help
2. Review ARCHITECTURE.md for system design
3. See examples.py for usage patterns
4. Check logs in `results/` for error details

---

**Your RAG benchmarking system is ready! 🎉**

Start with:
```bash
cd "VECTOR BENCHMARKING"
python main.py
```
