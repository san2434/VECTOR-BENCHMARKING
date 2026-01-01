#!/usr/bin/env python3
"""
Minimal memory-efficient benchmark test for 8GB RAM systems.
Tests ChromaDB only with the quantum computing paper.
"""

import os
import sys
import json
import time
import gc
from typing import List, Dict

# Force garbage collection
gc.collect()

# Load environment
from dotenv import load_dotenv
load_dotenv()

def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Simple chunking without heavy libraries."""
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunk_text = text[i:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"chunk_{chunk_id:04d}",
                "text": chunk_text,
                "metadata": {"start": i, "end": end}
            })
            chunk_id += 1
        i += chunk_size - overlap
    return chunks

def embed_texts_batch(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """Embed texts in small batches to save memory."""
    from openai import OpenAI
    client = OpenAI()
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)
        gc.collect()  # Free memory after each batch
    
    return all_embeddings

def test_chromadb(chunks: List[Dict], embeddings: List[List[float]]):
    """Test ChromaDB storage and retrieval."""
    import chromadb
    
    persist_dir = "./data/chromadb_test"
    
    # Clean up any existing test DB
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
    
    print("\n--- ChromaDB Test ---")
    
    # Create client
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents in batches
    print("Adding documents to ChromaDB...")
    batch_size = 50
    start_time = time.time()
    
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        
        collection.add(
            ids=[c["id"] for c in batch_chunks],
            documents=[c["text"] for c in batch_chunks],
            embeddings=batch_embeddings,
            metadatas=[c["metadata"] for c in batch_chunks]
        )
        gc.collect()
    
    index_time = time.time() - start_time
    print(f"  Indexed {len(chunks)} chunks in {index_time:.2f}s")
    
    # Test queries
    test_queries = [
        "What is quantum computing?",
        "How does a qubit differ from a classical bit?",
        "What is quantum entanglement?",
    ]
    
    print("\nRunning test queries...")
    from openai import OpenAI
    client_openai = OpenAI()
    
    for query in test_queries:
        # Embed query
        query_response = client_openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = query_response.data[0].embedding
        
        # Search
        start_time = time.time()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        search_time = (time.time() - start_time) * 1000
        
        print(f"\nQuery: {query}")
        print(f"  Search time: {search_time:.2f}ms")
        print(f"  Top results:")
        for j, (doc_id, doc, dist) in enumerate(zip(
            results["ids"][0], 
            results["documents"][0],
            results["distances"][0]
        )):
            print(f"    {j+1}. [{doc_id}] (dist={dist:.4f}): {doc[:80]}...")
    
    # Cleanup
    del collection
    del client
    gc.collect()
    
    return index_time

def test_rag_answer(question: str, context_chunks: List[str]) -> str:
    """Generate an answer using RAG."""
    from openai import OpenAI
    client = OpenAI()
    
    context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the question based only on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=200
    )
    
    return response.choices[0].message.content

def main():
    print("=" * 60)
    print("Minimal RAG Benchmark Test")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    # Load document
    doc_path = "./data/quantum_vision_full_text.txt"
    if not os.path.exists(doc_path):
        print(f"ERROR: Document not found at {doc_path}")
        sys.exit(1)
    
    print(f"\nLoading document: {doc_path}")
    with open(doc_path, "r") as f:
        text = f.read()
    print(f"  Document length: {len(text):,} characters")
    
    # Chunk the text
    print("\nChunking text...")
    chunks = simple_chunk(text, chunk_size=500, overlap=50)
    print(f"  Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\nCreating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts_batch(texts, batch_size=20)
    print(f"  Created {len(embeddings)} embeddings")
    gc.collect()
    
    # Test ChromaDB
    index_time = test_chromadb(chunks, embeddings)
    
    # Test a RAG question
    print("\n" + "=" * 60)
    print("Testing RAG Pipeline")
    print("=" * 60)
    
    # Load ground truth
    gt_path = "./data/ground_truth/quantum_vision_qa.json"
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        
        # Test first 2 questions
        for qa in gt_data["questions"][:2]:
            question = qa["text"]  # JSON uses "text" not "question"
            expected = qa["answer"]
            
            print(f"\nQuestion: {question}")
            print(f"Expected: {expected[:100]}...")
            
            # Get relevant chunks (simple search)
            import chromadb
            client = chromadb.PersistentClient(path="./data/chromadb_test")
            collection = client.get_collection("test_collection")
            
            from openai import OpenAI
            openai_client = OpenAI()
            query_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[question]
            )
            query_embedding = query_response.data[0].embedding
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            # Generate answer
            answer = test_rag_answer(question, results["documents"][0])
            print(f"RAG Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
