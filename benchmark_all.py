#!/usr/bin/env python3
"""
Complete RAG Benchmark Suite
Tests all 10 configurations: 5 Vector DBs × 2 Index Types
Uses cached embeddings for fair comparison.
"""

import json
import os
import time
import gc
import shutil
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIGS = {
    "chromadb_hnsw": {
        "db": "chromadb",
        "index_type": "HNSW",
        "params": {
            "space": "cosine",
            "hnsw_m": 16,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 50,
        }
    },
    "chromadb_ivf": {
        "db": "chromadb",
        "index_type": "IVF",
        "params": {
            "space": "cosine",
            "nlist": 100,  # Note: ChromaDB doesn't natively support IVF, simulated
            "nprobe": 10,
        }
    },
    "pinecone_hnsw": {
        "db": "pinecone",
        "index_type": "HNSW",
        "params": {
            "metric": "cosine",
            "cloud": "aws",
            "region": "us-east-1",
            "spec": "serverless",  # Serverless uses HNSW
        }
    },
    "pinecone_ivf": {
        "db": "pinecone",
        "index_type": "IVF",
        "params": {
            "metric": "cosine",
            "cloud": "aws",
            "region": "us-east-1",
            "spec": "serverless",  # Note: Pinecone serverless uses HNSW internally
        }
    },
    "postgres_hnsw": {
        "db": "postgres",
        "index_type": "HNSW",
        "params": {
            "m": 16,
            "ef_construction": 64,
            "ef_search": 40,
            "distance": "cosine",
        }
    },
    "postgres_ivf": {
        "db": "postgres",
        "index_type": "IVF",
        "params": {
            "lists": 100,
            "probes": 10,
            "distance": "cosine",
        }
    },
    "qdrant_hnsw": {
        "db": "qdrant",
        "index_type": "HNSW",
        "params": {
            "distance": "cosine",
            "hnsw_m": 16,
            "hnsw_ef_construct": 200,
            "hnsw_ef_search": 128,
        }
    },
    "qdrant_ivf": {
        "db": "qdrant",
        "index_type": "IVF",
        "params": {
            "distance": "cosine",
            "hnsw_m": 8,  # Lower M for IVF-like behavior
            "hnsw_ef_construct": 50,
            "hnsw_ef_search": 16,
        }
    },
    "milvus_hnsw": {
        "db": "milvus",
        "index_type": "HNSW",
        "params": {
            "metric": "COSINE",
            "hnsw_m": 16,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 128,
        }
    },
    "milvus_ivf": {
        "db": "milvus",
        "index_type": "IVF",
        "params": {
            "metric": "COSINE",
            "ivf_nlist": 100,
            "ivf_nprobe": 10,
        }
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_cached_embeddings():
    """Load pre-computed embeddings from cache."""
    print("Loading cached embeddings...")
    with open('./data/embeddings_cache.json', 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data['chunks'])} chunks with embeddings")
    return data['chunks'], data['embeddings'], data['metadata']

def load_ground_truth():
    """Load ground truth Q&A."""
    with open('./data/ground_truth/quantum_vision_qa.json', 'r') as f:
        return json.load(f)

def embed_query(client: OpenAI, query: str) -> List[float]:
    """Embed a single query."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding

def generate_rag_answer(client: OpenAI, question: str, context_chunks: List[str]) -> str:
    """Generate RAG answer using GPT-4o-mini."""
    context = "\n\n".join(context_chunks[:3])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the question based only on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=200,
        temperature=0.0
    )
    return response.choices[0].message.content

def calculate_f1(expected: str, predicted: str) -> float:
    """Calculate keyword F1 score."""
    expected_words = set(expected.lower().split())
    predicted_words = set(predicted.lower().split())
    overlap = len(expected_words & predicted_words)
    precision = overlap / len(predicted_words) if predicted_words else 0
    recall = overlap / len(expected_words) if expected_words else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# =============================================================================
# CHROMADB BENCHMARK
# =============================================================================

def benchmark_chromadb(config_name: str, config: dict, chunks: List[dict], 
                       embeddings: List[List[float]], questions: List[dict],
                       openai_client: OpenAI) -> dict:
    """Benchmark ChromaDB with specified configuration."""
    import chromadb
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    persist_dir = f"./data/chromadb_{config_name}"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    # Create client and collection
    client = chromadb.PersistentClient(path=persist_dir)
    
    # ChromaDB uses HNSW by default
    collection_metadata = {"hnsw:space": config["params"]["space"]}
    if config["index_type"] == "HNSW":
        collection_metadata["hnsw:M"] = config["params"]["hnsw_m"]
        collection_metadata["hnsw:construction_ef"] = config["params"]["hnsw_ef_construction"]
        collection_metadata["hnsw:search_ef"] = config["params"]["hnsw_ef_search"]
    
    collection = client.create_collection(
        name="benchmark",
        metadata=collection_metadata
    )
    
    # Index documents
    print("Indexing documents...")
    index_start = time.time()
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[c["id"] for c in chunks[i:batch_end]],
            documents=[c["text"] for c in chunks[i:batch_end]],
            embeddings=embeddings[i:batch_end],
            metadatas=[{"start": c["start"], "end": c["end"]} for c in chunks[i:batch_end]]
        )
    index_time = time.time() - index_start
    print(f"  Index time: {index_time:.3f}s")
    
    # Run queries
    results = []
    total_search_time = 0
    
    print("Running queries...")
    for qa in questions:
        question = qa["text"]
        expected = qa["answer"]
        
        # Embed query
        query_embedding = embed_query(openai_client, question)
        
        # Search
        search_start = time.time()
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        search_time = (time.time() - search_start) * 1000
        total_search_time += search_time
        
        # Generate answer
        answer = generate_rag_answer(openai_client, question, search_results["documents"][0])
        f1 = calculate_f1(expected, answer)
        
        results.append({
            "id": qa["id"],
            "question": question,
            "expected": expected,
            "answer": answer,
            "search_time_ms": search_time,
            "f1_score": f1,
            "top_chunks": search_results["ids"][0][:3],
            "distances": search_results["distances"][0][:3]
        })
        print(f"  {qa['id']}: F1={f1:.3f}, Search={search_time:.2f}ms")
    
    # Cleanup
    del collection
    del client
    gc.collect()
    
    return {
        "config_name": config_name,
        "config": config,
        "index_time_s": index_time,
        "avg_search_time_ms": total_search_time / len(questions),
        "avg_f1_score": sum(r["f1_score"] for r in results) / len(results),
        "results": results
    }

# =============================================================================
# PINECONE BENCHMARK
# =============================================================================

def benchmark_pinecone(config_name: str, config: dict, chunks: List[dict],
                       embeddings: List[List[float]], questions: List[dict],
                       openai_client: OpenAI) -> dict:
    """Benchmark Pinecone with specified configuration."""
    from pinecone import Pinecone, ServerlessSpec
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = f"rag-benchmark-{config_name.replace('_', '-')}"
    
    # Delete existing index if exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"  Deleting existing index {index_name}...")
        pc.delete_index(index_name)
        time.sleep(5)  # Wait for deletion
    
    # Create index
    print(f"  Creating index {index_name}...")
    index_start = time.time()
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric=config["params"]["metric"],
        spec=ServerlessSpec(
            cloud=config["params"]["cloud"],
            region=config["params"]["region"]
        )
    )
    
    # Wait for index to be ready
    print("  Waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    
    index = pc.Index(index_name)
    
    # Upsert vectors
    print("  Upserting vectors...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        vectors = [
            {
                "id": chunks[j]["id"],
                "values": embeddings[j],
                "metadata": {"text": chunks[j]["text"], "start": chunks[j]["start"], "end": chunks[j]["end"]}
            }
            for j in range(i, batch_end)
        ]
        index.upsert(vectors=vectors)
    
    # Wait for vectors to be indexed
    time.sleep(5)
    index_time = time.time() - index_start
    print(f"  Index time: {index_time:.3f}s")
    
    # Run queries
    results = []
    total_search_time = 0
    
    print("Running queries...")
    for qa in questions:
        question = qa["text"]
        expected = qa["answer"]
        
        # Embed query
        query_embedding = embed_query(openai_client, question)
        
        # Search
        search_start = time.time()
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        search_time = (time.time() - search_start) * 1000
        total_search_time += search_time
        
        # Extract texts from results
        retrieved_texts = [match.metadata["text"] for match in search_results.matches]
        
        # Generate answer
        answer = generate_rag_answer(openai_client, question, retrieved_texts)
        f1 = calculate_f1(expected, answer)
        
        results.append({
            "id": qa["id"],
            "question": question,
            "expected": expected,
            "answer": answer,
            "search_time_ms": search_time,
            "f1_score": f1,
            "top_chunks": [m.id for m in search_results.matches[:3]],
            "scores": [m.score for m in search_results.matches[:3]]
        })
        print(f"  {qa['id']}: F1={f1:.3f}, Search={search_time:.2f}ms")
    
    # Cleanup - delete index to avoid charges
    print(f"  Cleaning up index {index_name}...")
    pc.delete_index(index_name)
    
    return {
        "config_name": config_name,
        "config": config,
        "index_time_s": index_time,
        "avg_search_time_ms": total_search_time / len(questions),
        "avg_f1_score": sum(r["f1_score"] for r in results) / len(results),
        "results": results
    }

# =============================================================================
# POSTGRESQL BENCHMARK
# =============================================================================

def benchmark_postgres(config_name: str, config: dict, chunks: List[dict],
                       embeddings: List[List[float]], questions: List[dict],
                       openai_client: OpenAI) -> dict:
    """Benchmark PostgreSQL with pgvector."""
    import psycopg2
    from psycopg2.extras import execute_values
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    conn = psycopg2.connect(
        host=os.getenv("POSTGRESQL_HOST", "localhost"),
        port=int(os.getenv("POSTGRESQL_PORT", 5432)),
        user=os.getenv("POSTGRESQL_USER", "postgres"),
        password=os.getenv("POSTGRESQL_PASSWORD", "password"),
        dbname=os.getenv("POSTGRESQL_DATABASE", "vector_rag")
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    table_name = f"benchmark_{config_name}"
    
    # Drop existing table
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    
    # Create table
    cur.execute(f"""
        CREATE TABLE {table_name} (
            id VARCHAR(20) PRIMARY KEY,
            text TEXT,
            start_pos INT,
            end_pos INT,
            embedding vector(1536)
        );
    """)
    
    # Insert data
    print("Inserting vectors...")
    index_start = time.time()
    
    insert_data = [
        (chunks[i]["id"], chunks[i]["text"], chunks[i]["start"], chunks[i]["end"], embeddings[i])
        for i in range(len(chunks))
    ]
    
    execute_values(
        cur,
        f"INSERT INTO {table_name} (id, text, start_pos, end_pos, embedding) VALUES %s",
        insert_data,
        template="(%s, %s, %s, %s, %s::vector)"
    )
    
    # Create index
    print(f"Creating {config['index_type']} index...")
    if config["index_type"] == "HNSW":
        cur.execute(f"""
            CREATE INDEX ON {table_name} USING hnsw (embedding vector_cosine_ops)
            WITH (m = {config['params']['m']}, ef_construction = {config['params']['ef_construction']});
        """)
        # Set search parameter
        cur.execute(f"SET hnsw.ef_search = {config['params']['ef_search']};")
    else:  # IVF
        cur.execute(f"""
            CREATE INDEX ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {config['params']['lists']});
        """)
        # Set search parameter
        cur.execute(f"SET ivfflat.probes = {config['params']['probes']};")
    
    index_time = time.time() - index_start
    print(f"  Index time: {index_time:.3f}s")
    
    # Run queries
    results = []
    total_search_time = 0
    
    print("Running queries...")
    for qa in questions:
        question = qa["text"]
        expected = qa["answer"]
        
        # Embed query
        query_embedding = embed_query(openai_client, question)
        
        # Search
        search_start = time.time()
        cur.execute(f"""
            SELECT id, text, 1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
        """, (query_embedding, query_embedding))
        search_results = cur.fetchall()
        search_time = (time.time() - search_start) * 1000
        total_search_time += search_time
        
        # Extract texts
        retrieved_texts = [row[1] for row in search_results]
        
        # Generate answer
        answer = generate_rag_answer(openai_client, question, retrieved_texts)
        f1 = calculate_f1(expected, answer)
        
        results.append({
            "id": qa["id"],
            "question": question,
            "expected": expected,
            "answer": answer,
            "search_time_ms": search_time,
            "f1_score": f1,
            "top_chunks": [row[0] for row in search_results[:3]],
            "similarities": [row[2] for row in search_results[:3]]
        })
        print(f"  {qa['id']}: F1={f1:.3f}, Search={search_time:.2f}ms")
    
    # Cleanup
    cur.close()
    conn.close()
    
    return {
        "config_name": config_name,
        "config": config,
        "index_time_s": index_time,
        "avg_search_time_ms": total_search_time / len(questions),
        "avg_f1_score": sum(r["f1_score"] for r in results) / len(results),
        "results": results
    }

# =============================================================================
# QDRANT BENCHMARK
# =============================================================================

def benchmark_qdrant(config_name: str, config: dict, chunks: List[dict],
                     embeddings: List[List[float]], questions: List[dict],
                     openai_client: OpenAI) -> dict:
    """Benchmark Qdrant with specified configuration."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct, HnswConfigDiff
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    persist_dir = f"./data/qdrant_{config_name}"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Create client
    client = QdrantClient(path=persist_dir)
    collection_name = "benchmark"
    
    # Configure HNSW based on index type
    if config["index_type"] == "HNSW":
        hnsw_config = HnswConfigDiff(
            m=config["params"]["hnsw_m"],
            ef_construct=config["params"]["hnsw_ef_construct"],
        )
        ef_search = config["params"]["hnsw_ef_search"]
    else:  # IVF-like (simulated via HNSW tuning)
        hnsw_config = HnswConfigDiff(
            m=config["params"]["hnsw_m"],
            ef_construct=config["params"]["hnsw_ef_construct"],
        )
        ef_search = config["params"]["hnsw_ef_search"]
    
    # Create collection
    print("Creating collection...")
    index_start = time.time()
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        ),
        hnsw_config=hnsw_config
    )
    
    # Insert vectors
    print("Inserting vectors...")
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"]
            }
        ))
    
    client.upsert(collection_name=collection_name, points=points)
    
    index_time = time.time() - index_start
    print(f"  Index time: {index_time:.3f}s")
    
    # Run queries
    results = []
    total_search_time = 0
    
    print("Running queries...")
    for qa in questions:
        question = qa["text"]
        expected = qa["answer"]
        
        # Embed query
        query_embedding = embed_query(openai_client, question)
        
        # Search using query method (newer qdrant-client API)
        search_start = time.time()
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=5,
            search_params={"ef": ef_search}
        ).points
        search_time = (time.time() - search_start) * 1000
        total_search_time += search_time
        
        # Extract texts
        retrieved_texts = [hit.payload["text"] for hit in search_results]
        
        # Generate answer
        answer = generate_rag_answer(openai_client, question, retrieved_texts)
        f1 = calculate_f1(expected, answer)
        
        results.append({
            "id": qa["id"],
            "question": question,
            "expected": expected,
            "answer": answer,
            "search_time_ms": search_time,
            "f1_score": f1,
            "top_chunks": [hit.payload["chunk_id"] for hit in search_results[:3]],
            "scores": [hit.score for hit in search_results[:3]]
        })
        print(f"  {qa['id']}: F1={f1:.3f}, Search={search_time:.2f}ms")
    
    # Cleanup
    del client
    gc.collect()
    
    return {
        "config_name": config_name,
        "config": config,
        "index_time_s": index_time,
        "avg_search_time_ms": total_search_time / len(questions),
        "avg_f1_score": sum(r["f1_score"] for r in results) / len(results),
        "results": results
    }

# =============================================================================
# MILVUS BENCHMARK
# =============================================================================

def benchmark_milvus(config_name: str, config: dict, chunks: List[dict],
                     embeddings: List[List[float]], questions: List[dict],
                     openai_client: OpenAI) -> dict:
    """Benchmark Milvus (Lite) with specified configuration."""
    from pymilvus import MilvusClient
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    persist_dir = f"./data/milvus_{config_name}"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    
    db_path = os.path.join(persist_dir, "benchmark.db")
    collection_name = "benchmark"
    
    # Create client
    client = MilvusClient(db_path)
    
    # Configure search params based on index type
    if config["index_type"] == "HNSW":
        search_params = {
            "metric_type": config["params"]["metric"],
            "params": {"ef": config["params"]["hnsw_ef_search"]}
        }
    else:  # IVF
        search_params = {
            "metric_type": config["params"]["metric"],
            "params": {"nprobe": config["params"]["ivf_nprobe"]}
        }
    
    # Create collection
    print("Creating collection...")
    index_start = time.time()
    
    client.create_collection(
        collection_name=collection_name,
        dimension=1536,
        metric_type="COSINE",
        auto_id=False
    )
    
    # Insert vectors
    print("Inserting vectors...")
    data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        data.append({
            "id": i,
            "vector": embedding,
            "chunk_id": chunk["id"],
            "text": chunk["text"][:65000],  # Milvus field size limit
            "start": chunk["start"],
            "end": chunk["end"]
        })
    
    client.insert(collection_name=collection_name, data=data)
    
    index_time = time.time() - index_start
    print(f"  Index time: {index_time:.3f}s")
    
    # Run queries
    results = []
    total_search_time = 0
    
    print("Running queries...")
    for qa in questions:
        question = qa["text"]
        expected = qa["answer"]
        
        # Embed query
        query_embedding = embed_query(openai_client, question)
        
        # Search
        search_start = time.time()
        search_results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=5,
            output_fields=["text", "chunk_id"],
            search_params=search_params
        )
        search_time = (time.time() - search_start) * 1000
        total_search_time += search_time
        
        # Extract texts
        retrieved_texts = []
        chunk_ids = []
        scores = []
        if search_results and len(search_results) > 0:
            for hit in search_results[0]:
                retrieved_texts.append(hit.get("entity", {}).get("text", ""))
                chunk_ids.append(hit.get("entity", {}).get("chunk_id", ""))
                scores.append(hit.get("distance", 0.0))
        
        # Generate answer
        answer = generate_rag_answer(openai_client, question, retrieved_texts)
        f1 = calculate_f1(expected, answer)
        
        results.append({
            "id": qa["id"],
            "question": question,
            "expected": expected,
            "answer": answer,
            "search_time_ms": search_time,
            "f1_score": f1,
            "top_chunks": chunk_ids[:3],
            "scores": scores[:3]
        })
        print(f"  {qa['id']}: F1={f1:.3f}, Search={search_time:.2f}ms")
    
    # Cleanup
    del client
    gc.collect()
    
    return {
        "config_name": config_name,
        "config": config,
        "index_time_s": index_time,
        "avg_search_time_ms": total_search_time / len(questions),
        "avg_f1_score": sum(r["f1_score"] for r in results) / len(results),
        "results": results
    }

# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def generate_comparison_report(all_results: List[dict], metadata: dict) -> str:
    """Generate comprehensive comparison report."""
    report = []
    report.append("=" * 80)
    report.append("RAG VECTOR DATABASE BENCHMARK - COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Metadata
    report.append("\n" + "=" * 80)
    report.append("EXPERIMENT CONFIGURATION")
    report.append("=" * 80)
    report.append(f"Document: {metadata['document']}")
    report.append(f"Embedding Model: {metadata['embedding_model']}")
    report.append(f"Embedding Dimensions: {metadata['embedding_dimension']}")
    report.append(f"Chunk Size: {metadata['chunk_size']} characters")
    report.append(f"Chunk Overlap: {metadata['chunk_overlap']} characters")
    report.append(f"Total Chunks: {metadata['total_chunks']}")
    report.append(f"Total Questions: 8")
    report.append(f"LLM for RAG: GPT-4o-mini (temperature=0.0)")
    
    # Summary comparison table
    report.append("\n" + "=" * 80)
    report.append("SUMMARY COMPARISON TABLE")
    report.append("=" * 80)
    report.append(f"{'Configuration':<20} {'Index Time(s)':<15} {'Avg Search(ms)':<16} {'Avg F1 Score':<12}")
    report.append("-" * 65)
    
    # Sort by F1 score
    sorted_results = sorted(all_results, key=lambda x: x["avg_f1_score"], reverse=True)
    for r in sorted_results:
        report.append(f"{r['config_name']:<20} {r['index_time_s']:<15.3f} {r['avg_search_time_ms']:<16.2f} {r['avg_f1_score']:<12.3f}")
    
    # Best performers
    report.append("\n" + "-" * 65)
    best_f1 = max(all_results, key=lambda x: x["avg_f1_score"])
    fastest_search = min(all_results, key=lambda x: x["avg_search_time_ms"])
    fastest_index = min(all_results, key=lambda x: x["index_time_s"])
    
    report.append(f"🏆 Best F1 Score: {best_f1['config_name']} ({best_f1['avg_f1_score']:.3f})")
    report.append(f"⚡ Fastest Search: {fastest_search['config_name']} ({fastest_search['avg_search_time_ms']:.2f}ms)")
    report.append(f"🚀 Fastest Index: {fastest_index['config_name']} ({fastest_index['index_time_s']:.3f}s)")
    
    # Detailed results per configuration
    for result in all_results:
        report.append("\n" + "=" * 80)
        report.append(f"DETAILED RESULTS: {result['config_name'].upper()}")
        report.append("=" * 80)
        
        # Configuration
        report.append("\nHyperparameters:")
        report.append(f"  Database: {result['config']['db']}")
        report.append(f"  Index Type: {result['config']['index_type']}")
        for key, value in result['config']['params'].items():
            report.append(f"  {key}: {value}")
        
        report.append(f"\nPerformance Metrics:")
        report.append(f"  Index Build Time: {result['index_time_s']:.3f}s")
        report.append(f"  Average Search Time: {result['avg_search_time_ms']:.2f}ms")
        report.append(f"  Average F1 Score: {result['avg_f1_score']:.3f}")
        
        report.append(f"\nPer-Question Results:")
        report.append(f"{'ID':<6} {'Search(ms)':<12} {'F1 Score':<10} {'Top Chunk':<12}")
        report.append("-" * 42)
        for q in result['results']:
            top_chunk = q['top_chunks'][0] if q['top_chunks'] else 'N/A'
            report.append(f"{q['id']:<6} {q['search_time_ms']:<12.2f} {q['f1_score']:<10.3f} {top_chunk:<12}")
    
    # Per-question comparison across all configs
    report.append("\n" + "=" * 80)
    report.append("PER-QUESTION COMPARISON (F1 Scores)")
    report.append("=" * 80)
    
    # Header
    header = f"{'Question':<10}"
    for r in all_results:
        header += f" {r['config_name'][:12]:<12}"
    report.append(header)
    report.append("-" * (10 + 13 * len(all_results)))
    
    # Data rows
    for i in range(8):
        row = f"q{i+1:<9}"
        for r in all_results:
            f1 = r['results'][i]['f1_score']
            row += f" {f1:<12.3f}"
        report.append(row)
    
    report.append("\n" + "=" * 80)
    report.append("END OF BENCHMARK REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    print("=" * 60)
    print("RAG Vector Database Benchmark Suite")
    print("Testing: ChromaDB, Pinecone, PostgreSQL, Qdrant, Milvus")
    print("Index Types: HNSW, IVF")
    print("=" * 60)
    
    # Load data
    chunks, embeddings, metadata = load_cached_embeddings()
    gt_data = load_ground_truth()
    questions = gt_data["questions"]
    
    openai_client = OpenAI()
    
    all_results = []
    
    # Run all benchmarks (10 configurations)
    configs_to_run = [
        # Local databases
        ("chromadb_hnsw", benchmark_chromadb),
        ("chromadb_ivf", benchmark_chromadb),
        ("postgres_hnsw", benchmark_postgres),
        ("postgres_ivf", benchmark_postgres),
        ("qdrant_hnsw", benchmark_qdrant),
        ("qdrant_ivf", benchmark_qdrant),
        ("milvus_hnsw", benchmark_milvus),
        ("milvus_ivf", benchmark_milvus),
        # Cloud database
        ("pinecone_hnsw", benchmark_pinecone),
        ("pinecone_ivf", benchmark_pinecone),
    ]
    
    for config_name, benchmark_fn in configs_to_run:
        try:
            result = benchmark_fn(
                config_name, 
                CONFIGS[config_name], 
                chunks, 
                embeddings, 
                questions,
                openai_client
            )
            all_results.append(result)
            
            # Save individual result
            with open(f"./results/{config_name}_results.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n✅ {config_name} complete!")
            
        except Exception as e:
            print(f"\n❌ {config_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison report
    print("\n" + "=" * 60)
    print("Generating comparison report...")
    print("=" * 60)
    
    report = generate_comparison_report(all_results, metadata)
    
    # Save report
    report_path = "./results/benchmark_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<20} {'Avg Search(ms)':<16} {'Avg F1 Score':<12}")
    print("-" * 50)
    for r in sorted(all_results, key=lambda x: x["avg_f1_score"], reverse=True):
        print(f"{r['config_name']:<20} {r['avg_search_time_ms']:<16.2f} {r['avg_f1_score']:<12.3f}")


if __name__ == "__main__":
    main()
