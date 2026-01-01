# Quick Start Guide

## Prerequisites
- Python 3.8+
- OpenAI API key
- (Optional) Pinecone API key
- (Optional) PostgreSQL database

## Step 1: Environment Setup

```bash
# Navigate to project
cd "VECTOR BENCHMARKING"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required:
#   - OPENAI_API_KEY=sk-...
# 
# Optional:
#   - PINECONE_API_KEY=...
#   - PostgreSQL connection details
```

## Step 3: Run Benchmarking

```bash
python main.py
```

This will:
1. Extract text from Wikipedia articles
2. Create embeddings (this uses your OpenAI API key)
3. Store in ChromaDB (always works locally)
4. Attempt to store in Pinecone (if configured)
5. Attempt to store in PostgreSQL (if available)
6. Generate benchmark results in `results/` folder

## Expected Output

```
INFO:__main__:Starting RAG Benchmarking System
INFO:__main__:============================================================
INFO:__main__:Extracting text from 3 URLs...
INFO:__main__:Extracted XXXX characters
INFO:__main__:Chunking text...
INFO:__main__:Created XX chunks
INFO:__main__:Creating embeddings...
INFO:__main__:Created XX embeddings

... (benchmarking runs) ...

INFO:__main__:RAG Benchmarking Complete!
```

## Results

Check `results/complete_benchmark_report.json` for detailed metrics including:
- Storage performance (throughput, time)
- Search performance (latency, QPS)
- Indexing algorithm comparison

## Minimal Setup (ChromaDB Only)

To run with just ChromaDB (no external services):

```bash
# Set minimal .env
echo "OPENAI_API_KEY=your-key-here" > .env

# Run
python main.py
```

## Next Steps

1. Review results in `results/`
2. Modify URLs in `config/settings.py` to test with different data
3. Adjust chunk size and embedding parameters in `config/settings.py`
4. Deploy the most efficient configuration to production

## Troubleshooting

### Module not found errors
```bash
# Ensure you're in the project directory and venv is activated
source venv/bin/activate
pip install -r requirements.txt
```

### OpenAI API errors
- Verify API key in `.env`
- Check API quota at https://platform.openai.com/

### Pinecone/PostgreSQL errors
- These are optional - the system works without them
- See README.md for setup instructions
