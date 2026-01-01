"""
Example script showing how to use different components
"""

from config.settings import URLS_TO_SCRAPE
from src.data_extraction.web_scraper import extract_text_from_urls
from src.embeddings.text_chunker import chunk_documents
from src.embeddings.embedding_service import create_embeddings
from src.vector_stores.chromadb_store import ChromaVectorStore
from src.indexing.index_strategies import IVFIndex, HNSWIndex


def example_basic_rag():
    """Basic RAG example"""
    print("Example 1: Basic RAG Pipeline")
    print("-" * 50)
    
    # 1. Extract text
    print("Step 1: Extracting text...")
    text = extract_text_from_urls(["https://en.wikipedia.org/wiki/Artificial_intelligence"])
    print(f"Extracted {len(text)} characters")
    
    # 2. Chunk text
    print("\nStep 2: Chunking text...")
    chunks = chunk_documents([text], chunk_size=500, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks")
    
    # 3. Create embeddings
    print("\nStep 3: Creating embeddings...")
    embeddings = create_embeddings(chunks[:5])  # Use first 5 for demo
    print(f"Created {len(embeddings)} embeddings")
    
    # 4. Store in vector database
    print("\nStep 4: Storing in ChromaDB...")
    store = ChromaVectorStore("example_collection")
    store.store(chunks[:5], embeddings)
    print("Stored successfully")
    
    # 5. Query
    print("\nStep 5: Querying...")
    query_embedding = create_embeddings(["What is artificial intelligence?"])[0]
    results = store.search(query_embedding, top_k=3)
    print(f"Found {len(results)} results")
    for text, score in results:
        print(f"  - Score: {score:.2f}")
    
    store.delete()
    print("\nExample 1 complete!\n")


def example_indexing_algorithms():
    """Compare indexing algorithms"""
    print("Example 2: Indexing Algorithms")
    print("-" * 50)
    
    # Create sample embeddings
    import numpy as np
    embeddings = np.random.rand(100, 1536).tolist()
    
    # Create IVF index
    print("Building IVF index...")
    ivf = IVFIndex(embeddings, n_clusters=10)
    
    # Create HNSW index
    print("Building HNSW index...")
    hnsw = HNSWIndex(embeddings, max_m=8)
    
    # Test query
    query = np.random.rand(1536).tolist()
    
    print("\nSearching with IVF...")
    ivf_results = ivf.search(query, top_k=5)
    print(f"IVF found indices: {ivf_results}")
    
    print("\nSearching with HNSW...")
    hnsw_results = hnsw.search(query, top_k=5)
    print(f"HNSW found indices: {hnsw_results}")
    
    print("\nExample 2 complete!\n")


def example_multiple_stores():
    """Use multiple vector stores"""
    print("Example 3: Multiple Vector Stores")
    print("-" * 50)
    
    # Create sample data
    texts = [
        "Machine learning is a type of artificial intelligence",
        "Deep learning uses neural networks",
        "NLP processes text data"
    ]
    embeddings = create_embeddings(texts)
    
    # Store in ChromaDB
    print("Storing in ChromaDB...")
    chroma = ChromaVectorStore("multi_store_example")
    chroma.store(texts, embeddings)
    
    # Query
    query_embedding = create_embeddings(["artificial intelligence"])[0]
    results = chroma.search(query_embedding, top_k=2)
    
    print("\nResults from ChromaDB:")
    for text, score in results:
        print(f"  - {text[:50]}... (score: {score:.2f})")
    
    chroma.delete()
    print("\nExample 3 complete!\n")


if __name__ == "__main__":
    try:
        example_basic_rag()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_indexing_algorithms()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_multiple_stores()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    print("All examples complete!")
