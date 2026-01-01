"""
Indexing Strategies Module
Implements different indexing algorithms for vector search
"""

from typing import List, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexingStrategy:
    """Base class for indexing strategies"""
    
    def __init__(self, embeddings: List[List[float]], index_type: str):
        """
        Initialize indexing strategy
        
        Args:
            embeddings: List of embedding vectors
            index_type: Type of index (IVF, HNSW, etc.)
        """
        self.embeddings = np.array(embeddings)
        self.index_type = index_type
        self.index = None
    
    def build_index(self):
        """Build the index"""
        raise NotImplementedError


class IVFIndex:
    """IVF (Inverted File) Index Implementation"""
    
    def __init__(self, embeddings: List[List[float]], n_clusters: int = 100):
        """
        Initialize IVF Index
        
        Args:
            embeddings: List of embedding vectors
            n_clusters: Number of clusters for IVF
        """
        self.embeddings = np.array(embeddings)
        self.n_clusters = n_clusters
        self.index = None
        self.cluster_centers = None
        
        self._build_ivf_index()
    
    def _build_ivf_index(self):
        """Build IVF index using KMeans clustering"""
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            kmeans.fit(self.embeddings)
            
            self.cluster_centers = kmeans.cluster_centers_
            self.cluster_assignments = kmeans.labels_
            
            # Build inverted file
            self.inverted_file = {}
            for idx, cluster_id in enumerate(self.cluster_assignments):
                if cluster_id not in self.inverted_file:
                    self.inverted_file[cluster_id] = []
                self.inverted_file[cluster_id].append(idx)
            
            logger.info(f"Built IVF index with {self.n_clusters} clusters")
            return self
        except Exception as e:
            logger.error(f"Error building IVF index: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5, nprobe: int = 10) -> List[int]:
        """
        Search using IVF index
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            nprobe: Number of clusters to probe
            
        Returns:
            Indices of top-k nearest neighbors
        """
        query = np.array(query_embedding).reshape(1, -1)
        
        # Find nearest cluster centers
        distances = np.linalg.norm(self.cluster_centers - query, axis=1)
        nearest_clusters = np.argsort(distances)[:nprobe]
        
        # Collect candidates from nearest clusters
        candidates = []
        for cluster_id in nearest_clusters:
            if cluster_id in self.inverted_file:
                candidates.extend(self.inverted_file[cluster_id])
        
        # Compute distances to candidates
        candidate_embeddings = self.embeddings[candidates]
        candidate_distances = np.linalg.norm(candidate_embeddings - query, axis=1)
        
        # Return top-k
        top_indices = np.argsort(candidate_distances)[:top_k]
        return [candidates[i] for i in top_indices]


class HNSWIndex:
    """HNSW (Hierarchical Navigable Small World) Index Implementation"""
    
    def __init__(self, embeddings: List[List[float]], max_m: int = 16, ef_construction: int = 200):
        """
        Initialize HNSW Index
        
        Args:
            embeddings: List of embedding vectors
            max_m: Maximum number of connections per node
            ef_construction: Size of dynamic list
        """
        self.embeddings = np.array(embeddings)
        self.max_m = max_m
        self.ef_construction = ef_construction
        self.graph = {}
        
        self._build_hnsw_index()
    
    def _build_hnsw_index(self):
        """Build HNSW index using hierarchical structure"""
        try:
            # Simplified HNSW implementation
            # For production, use hnswlib library
            
            n = len(self.embeddings)
            
            # Initialize graph
            for i in range(n):
                self.graph[i] = set()
            
            # Add edges based on similarity (simplified)
            for i in range(n):
                # Find approximate nearest neighbors
                distances = np.linalg.norm(self.embeddings - self.embeddings[i], axis=1)
                nearest = np.argsort(distances)[1:min(self.max_m + 1, n)]
                
                for j in nearest:
                    self.graph[i].add(j)
                    self.graph[j].add(i)
            
            logger.info(f"Built HNSW index with max_m={self.max_m}")
            return self
        except Exception as e:
            logger.error(f"Error building HNSW index: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5, ef: int = None) -> List[int]:
        """
        Search using HNSW index
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            ef: Size of dynamic list for search
            
        Returns:
            Indices of top-k nearest neighbors
        """
        if ef is None:
            ef = max(top_k * 2, 50)
        
        query = np.array(query_embedding)
        
        # Start from a random node
        current = 0
        visited = set([current])
        candidates = [(np.linalg.norm(self.embeddings[current] - query), current)]
        
        # Search layer by layer
        for _ in range(ef):
            lowerbound = candidates[0][0]
            
            if lowerbound > candidates[-1][0]:
                break
            
            nearest = min(visited, key=lambda x: np.linalg.norm(self.embeddings[x] - query))
            
            for neighbor in self.graph.get(nearest, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distance = np.linalg.norm(self.embeddings[neighbor] - query)
                    
                    if distance < candidates[-1][0] or len(candidates) < ef:
                        candidates.append((distance, neighbor))
                        candidates.sort()
                        if len(candidates) > ef:
                            candidates.pop()
        
        # Return top-k
        return [idx for _, idx in sorted(candidates)[:top_k]]


def get_index(embeddings: List[List[float]], index_type: str = "IVF") -> IndexingStrategy:
    """
    Factory function to get the right indexing strategy
    
    Args:
        embeddings: List of embedding vectors
        index_type: Type of index (IVF or HNSW)
        
    Returns:
        Index instance
    """
    if index_type.upper() == "IVF":
        return IVFIndex(embeddings)
    elif index_type.upper() == "HNSW":
        return HNSWIndex(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
