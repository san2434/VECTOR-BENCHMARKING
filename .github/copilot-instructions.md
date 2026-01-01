<!-- Copilot Instructions for RAG Vector Benchmarking Project -->
- [x] Verify copilot-instructions.md exists in .github directory
- [x] Clarify Project Requirements
- [x] Scaffold the Project
- [x] Customize the Project
- [ ] Install Required Extensions
- [ ] Compile the Project
- [ ] Create and Run Task
- [ ] Launch the Project
- [ ] Ensure Documentation is Complete

## Project Summary

Comprehensive RAG (Retrieval Augmented Generation) benchmarking system that:
- Extracts unstructured text from the internet
- Creates embeddings using OpenAI's GPT-4o mini
- Stores vectors in 3 different databases (Pinecone, ChromaDB, PostgreSQL)
- Compares indexing algorithms (IVF, HNSW)
- Benchmarks and identifies the most efficient vectorization and retrieval method

## Tech Stack
- Python 3.8+
- OpenAI API (GPT-4o mini)
- Pinecone, ChromaDB, PostgreSQL with pgvector
- scikit-learn, NumPy, requests, BeautifulSoup4

## Key Files
- `main.py` - Main execution script
- `config/settings.py` - Configuration
- `src/data_extraction/` - Web scraping
- `src/embeddings/` - Chunking and embeddings
- `src/vector_stores/` - Database implementations
- `src/indexing/` - IVF and HNSW algorithms
- `src/rag/` - RAG pipeline
- `src/benchmark/` - Benchmarking suite
