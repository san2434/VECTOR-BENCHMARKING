"""
Embedding Service Module
Creates embeddings using OpenAI API
"""

from typing import List
import openai
import logging
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


class EmbeddingService:
    """Creates embeddings for text using OpenAI"""
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        """
        Initialize EmbeddingService
        
        Args:
            model: Embedding model to use
        """
        self.model = model
        self.dimension = EMBEDDING_DIMENSION
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model
            )
            
            # Sort by index to maintain order
            embeddings = sorted(response['data'], key=lambda x: x['index'])
            vectors = [item['embedding'] for item in embeddings]
            
            logger.info(f"Created {len(vectors)} embeddings")
            return vectors
            
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[str], batch_size: int = 20) -> List[List[float]]:
        """
        Create embeddings for text chunks in batches
        
        Args:
            chunks: List of text chunks
            batch_size: Number of chunks to process at once
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = self.embed_texts(batch)
            all_embeddings.extend(embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}")
        
        return all_embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """
        Alias for embed_chunks - create embeddings for texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        return self.embed_chunks(texts, batch_size)


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to create embeddings
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    service = EmbeddingService()
    return service.embed_texts(texts)
