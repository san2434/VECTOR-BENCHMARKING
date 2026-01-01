"""
Text Chunking Module
Splits large text documents into smaller chunks
"""

from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Chunks text into smaller, overlapping segments"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize TextChunker
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping segments
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            
            # Skip empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position
            start = end - self.chunk_overlap
            
            # Prevent infinite loop on very small overlap
            if start >= end:
                break
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks (groups of sentences)
        """
        # Simple sentence splitting
        sentences = text.replace('! ', '.\n').replace('? ', '.\n').split('.')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks


def chunk_documents(documents: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Convenience function to chunk multiple documents
    
    Args:
        documents: List of documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all chunks from all documents
    """
    chunker = TextChunker(chunk_size, chunk_overlap)
    all_chunks = []
    
    for doc in documents:
        chunks = chunker.chunk_text(doc)
        all_chunks.extend(chunks)
    
    return all_chunks
