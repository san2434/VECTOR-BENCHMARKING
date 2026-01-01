"""
RAG (Retrieval Augmented Generation) Pipeline Module
"""

from typing import List, Dict, Tuple
import openai
from config.settings import MODEL, OPENAI_API_KEY
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


class RAGPipeline:
    """RAG Pipeline for retrieval and generation"""
    
    def __init__(self, vector_store, embedding_service):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_store: Vector store instance
            embedding_service: Embedding service instance
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (document, relevance_score) tuples
        """
        try:
            # Embed the query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Search in vector store
            start_time = time.time()
            results = self.vector_store.search(query_embedding, top_k=top_k)
            search_time = time.time() - start_time
            
            logger.info(f"Retrieved {len(results)} documents in {search_time:.4f}s")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate(self, query: str, context: List[str], max_tokens: int = 500) -> str:
        """
        Generate answer based on retrieved context
        
        Args:
            query: Original query
            context: List of context documents
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response
        """
        try:
            # Prepare prompt
            context_text = "\n".join([f"- {doc}" for doc in context])
            prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""
            
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            generation_time = time.time() - start_time
            
            answer = response['choices'][0]['message']['content']
            logger.info(f"Generated response in {generation_time:.4f}s")
            
            return answer
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""
    
    def query(self, query: str, top_k: int = 5) -> Dict:
        """
        Complete RAG query: retrieve and generate
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieved docs and generated answer
        """
        try:
            start_time = time.time()
            
            # Retrieve
            retrieved = self.retrieve(query, top_k=top_k)
            context = [doc for doc, _ in retrieved]
            
            # Generate
            answer = self.generate(query, context)
            
            total_time = time.time() - start_time
            
            return {
                "query": query,
                "retrieved_documents": retrieved,
                "answer": answer,
                "total_time": total_time,
                "num_documents": len(retrieved)
            }
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "retrieved_documents": [],
                "answer": ""
            }
