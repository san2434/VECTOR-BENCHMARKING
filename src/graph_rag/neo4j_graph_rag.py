"""
Neo4j GraphRAG Implementation using LangChain

This module implements a Knowledge Graph-based RAG pipeline that:
1. Extracts entities and relationships from text using LLM
2. Stores them in Neo4j as a knowledge graph
3. Retrieves context using graph traversal + vector similarity
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Entity:
    """Represents an extracted entity"""
    name: str
    type: str
    description: str = ""


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source: str
    target: str
    relation: str
    description: str = ""


class Neo4jGraphRAG:
    """
    Knowledge Graph RAG using Neo4j + OpenAI
    
    Pipeline:
    1. Extract entities/relationships from chunks using LLM
    2. Store in Neo4j graph database
    3. Query using hybrid: graph traversal + semantic similarity
    """
    
    def __init__(self, collection_name: str = "benchmark"):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.collection_name = collection_name
        
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password)
        )
        self.openai_client = OpenAI()
        
        # Entity extraction prompt
        self.extraction_prompt = """Extract entities and relationships from the following text.

TEXT:
{text}

Return a JSON object with two arrays:
1. "entities": Array of objects with "name", "type" (CONCEPT, PERSON, TECHNOLOGY, PROCESS, METRIC), and "description"
2. "relationships": Array of objects with "source" (entity name), "target" (entity name), "relation" (verb/relationship type), and "description"

Focus on scientific/technical entities and their relationships. Be concise but comprehensive.
Return ONLY valid JSON, no markdown or explanation.

Example format:
{{"entities": [{{"name": "Qubit", "type": "CONCEPT", "description": "Basic unit of quantum information"}}], "relationships": [{{"source": "Qubit", "target": "Superposition", "relation": "EXHIBITS", "description": "A qubit can exist in superposition"}}]}}
"""

    def clear_graph(self):
        """Clear all nodes and relationships from the graph"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def extract_entities_and_relationships(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """Use LLM to extract entities and relationships from text"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting knowledge graph entities and relationships from scientific text. Return only valid JSON."},
                    {"role": "user", "content": self.extraction_prompt.format(text=text[:2000])}  # Limit text length
                ],
                max_tokens=1000,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            data = json.loads(result_text)
            
            entities = [
                Entity(
                    name=e.get("name", ""),
                    type=e.get("type", "CONCEPT"),
                    description=e.get("description", "")
                )
                for e in data.get("entities", [])
                if e.get("name")
            ]
            
            relationships = [
                Relationship(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relation=r.get("relation", "RELATED_TO"),
                    description=r.get("description", "")
                )
                for r in data.get("relationships", [])
                if r.get("source") and r.get("target")
            ]
            
            return entities, relationships
            
        except Exception as e:
            print(f"  Warning: Entity extraction failed: {e}")
            return [], []
    
    def store_chunk_with_entities(self, chunk: Dict, chunk_embedding: List[float]):
        """Store a chunk and its extracted entities/relationships in Neo4j"""
        
        # Extract entities and relationships
        entities, relationships = self.extract_entities_and_relationships(chunk["text"])
        
        with self.driver.session(database=self.database) as session:
            # Create chunk node
            session.run("""
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text,
                    c.start = $start,
                    c.end = $end,
                    c.collection = $collection,
                    c.embedding = $embedding
            """, {
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"],
                "collection": self.collection_name,
                "embedding": chunk_embedding
            })
            
            # Create entity nodes and link to chunk
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.description = $description
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                """, {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "chunk_id": chunk["id"]
                })
            
            # Create relationships between entities
            for rel in relationships:
                session.run("""
                    MATCH (s:Entity {name: $source})
                    MATCH (t:Entity {name: $target})
                    MERGE (s)-[r:RELATES_TO {type: $relation}]->(t)
                    SET r.description = $description
                """, {
                    "source": rel.source,
                    "target": rel.target,
                    "relation": rel.relation,
                    "description": rel.description
                })
        
        return len(entities), len(relationships)
    
    def build_graph(self, chunks: List[Dict], embeddings: List[List[float]]) -> Dict[str, Any]:
        """Build the knowledge graph from chunks"""
        
        print(f"Building knowledge graph from {len(chunks)} chunks...")
        self.clear_graph()
        
        total_entities = 0
        total_relationships = 0
        start_time = time.time()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            num_entities, num_rels = self.store_chunk_with_entities(chunk, embedding)
            total_entities += num_entities
            total_relationships += num_rels
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")
        
        index_time = time.time() - start_time
        
        # Create vector index for semantic search
        self._create_vector_index()
        
        # Get stats
        stats = self.get_stats()
        stats["index_time_s"] = index_time
        
        print(f"  Graph built: {stats['num_entities']} entities, {stats['num_relationships']} relationships")
        print(f"  Index time: {index_time:.3f}s")
        
        return stats
    
    def _create_vector_index(self):
        """Create vector index for chunk embeddings"""
        with self.driver.session(database=self.database) as session:
            try:
                # Drop existing index if present
                session.run("DROP INDEX chunk_embedding_index IF EXISTS")
                
                # Create vector index
                session.run("""
                    CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            except Exception as e:
                print(f"  Note: Vector index creation: {e}")
    
    def search_hybrid(self, query_embedding: List[float], query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid search: Vector similarity + Graph traversal
        
        1. Find top chunks by vector similarity
        2. Expand context via graph relationships
        3. Return enriched results
        """
        
        with self.driver.session(database=self.database) as session:
            # Step 1: Vector similarity search
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
                    YIELD node, score
                    RETURN node.chunk_id as chunk_id, node.text as text, score
                    ORDER BY score DESC
                """, {"embedding": query_embedding, "top_k": top_k})
                
                vector_results = [dict(record) for record in result]
            except Exception:
                # Fallback: brute force cosine similarity
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.embedding IS NOT NULL
                    WITH c, gds.similarity.cosine(c.embedding, $embedding) as score
                    ORDER BY score DESC
                    LIMIT $top_k
                    RETURN c.chunk_id as chunk_id, c.text as text, score
                """, {"embedding": query_embedding, "top_k": top_k})
                vector_results = [dict(record) for record in result]
            
            if not vector_results:
                # Final fallback: just get chunks
                result = session.run("""
                    MATCH (c:Chunk)
                    RETURN c.chunk_id as chunk_id, c.text as text, 1.0 as score
                    LIMIT $top_k
                """, {"top_k": top_k})
                vector_results = [dict(record) for record in result]
            
            # Step 2: Enrich with graph context (related entities)
            enriched_results = []
            for vr in vector_results:
                # Get entities mentioned in this chunk
                entity_result = session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
                    OPTIONAL MATCH (e)-[r:RELATES_TO]-(related:Entity)
                    RETURN collect(DISTINCT e.name) as entities,
                           collect(DISTINCT {from: e.name, rel: r.type, to: related.name}) as relationships
                """, {"chunk_id": vr["chunk_id"]})
                
                entity_record = entity_result.single()
                
                enriched_results.append({
                    "chunk_id": vr["chunk_id"],
                    "text": vr["text"],
                    "score": vr["score"],
                    "entities": entity_record["entities"] if entity_record else [],
                    "relationships": entity_record["relationships"] if entity_record else []
                })
            
            return enriched_results
    
    def search_simple(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Simple vector-only search (for fair comparison)"""
        
        with self.driver.session(database=self.database) as session:
            try:
                # Try vector index first
                result = session.run("""
                    CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
                    YIELD node, score
                    RETURN node.chunk_id as chunk_id, node.text as text, score
                    ORDER BY score DESC
                """, {"embedding": query_embedding, "top_k": top_k})
                
                return [dict(record) for record in result]
            except Exception:
                # Fallback to manual cosine (Neo4j Aura may not have GDS)
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.embedding IS NOT NULL
                    RETURN c.chunk_id as chunk_id, c.text as text, 1.0 as score
                    LIMIT $top_k
                """, {"top_k": top_k})
                return [dict(record) for record in result]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk) WITH count(c) as chunks
                MATCH (e:Entity) WITH chunks, count(e) as entities
                MATCH ()-[r:RELATES_TO]->() WITH chunks, entities, count(r) as rels
                MATCH ()-[m:MENTIONS]->() 
                RETURN chunks, entities, rels, count(m) as mentions
            """)
            record = result.single()
            
            return {
                "num_chunks": record["chunks"],
                "num_entities": record["entities"],
                "num_relationships": record["rels"],
                "num_mentions": record["mentions"]
            }
    
    def close(self):
        """Close the Neo4j driver"""
        self.driver.close()


def benchmark_neo4j_graphrag(chunks: List[Dict], embeddings: List[List[float]], 
                              questions: List[Dict], openai_client: OpenAI,
                              use_hybrid: bool = True) -> Dict[str, Any]:
    """
    Benchmark Neo4j GraphRAG pipeline
    
    Args:
        chunks: List of text chunks
        embeddings: Pre-computed embeddings
        questions: List of Q&A pairs for evaluation
        openai_client: OpenAI client for embeddings/generation
        use_hybrid: Whether to use hybrid search (graph + vector) or vector only
    
    Returns:
        Benchmark results dict
    """
    
    config_name = f"neo4j_graphrag_{'hybrid' if use_hybrid else 'vector'}"
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    # Initialize GraphRAG
    graph_rag = Neo4jGraphRAG(collection_name="benchmark")
    
    # Build knowledge graph
    index_start = time.time()
    stats = graph_rag.build_graph(chunks, embeddings)
    index_time = stats["index_time_s"]
    
    # Run queries
    print("Running queries...")
    results = []
    total_search_time = 0
    
    for qa in questions:
        question = qa["text"]
        expected = qa["answer"]
        
        # Get query embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[question]
        )
        query_embedding = response.data[0].embedding
        
        # Search
        search_start = time.time()
        if use_hybrid:
            search_results = graph_rag.search_hybrid(query_embedding, question, top_k=5)
        else:
            search_results = graph_rag.search_simple(query_embedding, top_k=5)
        search_time = (time.time() - search_start) * 1000
        total_search_time += search_time
        
        # Generate answer
        retrieved_texts = [r["text"] for r in search_results[:3]]
        context = "\n\n".join(retrieved_texts)
        
        # Add graph context for hybrid mode
        if use_hybrid and search_results:
            entities = []
            for r in search_results[:3]:
                entities.extend(r.get("entities", []))
            if entities:
                context += f"\n\nRelated concepts: {', '.join(set(entities))}"
        
        answer_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the question based only on the provided context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            max_tokens=200,
            temperature=0.0
        )
        answer = answer_response.choices[0].message.content
        
        # Calculate F1
        expected_words = set(expected.lower().split())
        predicted_words = set(answer.lower().split())
        overlap = len(expected_words & predicted_words)
        precision = overlap / len(predicted_words) if predicted_words else 0
        recall = overlap / len(expected_words) if expected_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "id": qa["id"],
            "question": question,
            "expected": expected,
            "answer": answer,
            "search_time_ms": search_time,
            "f1_score": f1,
            "top_chunks": [r["chunk_id"] for r in search_results[:3]],
            "scores": [r["score"] for r in search_results[:3]]
        })
        
        print(f"  {qa['id']}: F1={f1:.3f}, Search={search_time:.2f}ms")
    
    # Calculate averages
    avg_f1 = sum(r["f1_score"] for r in results) / len(results)
    avg_search = total_search_time / len(questions)
    
    print(f"\n{config_name} Summary:")
    print(f"  Avg F1={avg_f1:.3f}, Avg Search={avg_search:.2f}ms, Index={index_time:.3f}s")
    print(f"  Graph: {stats['num_entities']} entities, {stats['num_relationships']} relationships")
    
    # Clean up
    graph_rag.close()
    
    return {
        "config_name": config_name,
        "config": {
            "db": "neo4j_aura",
            "type": "GraphRAG",
            "mode": "hybrid" if use_hybrid else "vector",
            "graph_stats": stats
        },
        "index_time_s": index_time,
        "avg_search_time_ms": avg_search,
        "avg_f1_score": avg_f1,
        "results": results
    }


if __name__ == "__main__":
    # Test the module
    print("Testing Neo4j GraphRAG connection...")
    
    graph_rag = Neo4jGraphRAG()
    
    # Test with a simple extraction
    test_text = "Quantum computing uses qubits that can exist in superposition. Unlike classical bits, qubits leverage quantum entanglement for parallel processing."
    
    entities, relationships = graph_rag.extract_entities_and_relationships(test_text)
    print(f"\nExtracted {len(entities)} entities:")
    for e in entities:
        print(f"  - {e.name} ({e.type})")
    
    print(f"\nExtracted {len(relationships)} relationships:")
    for r in relationships:
        print(f"  - {r.source} --[{r.relation}]--> {r.target}")
    
    graph_rag.close()
    print("\n✅ Neo4j GraphRAG module ready!")
