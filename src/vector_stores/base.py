"""
Base Vector Store Interface
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def store(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict] = None) -> None:
        """Store vectors and associated metadata"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def delete(self) -> None:
        """Delete all data from store"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """Get statistics about the store"""
        pass
