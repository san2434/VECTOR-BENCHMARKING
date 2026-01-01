"""
Evaluation Metrics Module
Implements Recall@k, MRR, Context Precision, NDCG and Response Accuracy metrics
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Retrieval evaluation metrics for RAG systems
    
    Metrics:
    - Recall@k: Proportion of relevant documents retrieved in top-k
    - MRR: Mean Reciprocal Rank - average of reciprocal ranks of first relevant result
    - Context Precision: Precision of retrieved context (relevant / retrieved)
    - NDCG@k: Normalized Discounted Cumulative Gain
    """
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calculate Recall@k
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by relevance)
            relevant_ids: Set of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_retrieved = retrieved_at_k.intersection(relevant_ids)
        
        return len(relevant_retrieved) / len(relevant_ids)
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calculate Precision@k (Context Precision)
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_retrieved = retrieved_at_k.intersection(relevant_ids)
        
        return len(relevant_retrieved) / k
    
    @staticmethod
    def mrr(retrieved_ids_list: List[List[str]], relevant_ids_list: List[Set[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank across multiple queries
        
        Args:
            retrieved_ids_list: List of retrieved ID lists for each query
            relevant_ids_list: List of relevant ID sets for each query
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        if not retrieved_ids_list:
            return 0.0
        
        reciprocal_ranks = []
        
        for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
            rr = RetrievalMetrics._reciprocal_rank(retrieved_ids, relevant_ids)
            reciprocal_ranks.append(rr)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def _reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculate reciprocal rank for a single query"""
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int, 
                  relevance_scores: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            relevance_scores: Optional dict of doc_id -> relevance score (default: binary)
            
        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0
        
        # Default to binary relevance
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate ideal DCG
        ideal_rels = sorted([relevance_scores.get(doc_id, 0.0) for doc_id in relevant_ids], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_rels[:k]):
            idcg += rel / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def context_precision(retrieved_chunks: List[str], relevant_chunks: Set[str]) -> float:
        """
        Calculate context precision - how much of retrieved context is relevant
        
        Args:
            retrieved_chunks: List of retrieved chunk IDs
            relevant_chunks: Set of actually relevant chunk IDs
            
        Returns:
            Context precision score (0.0 to 1.0)
        """
        if not retrieved_chunks:
            return 0.0
        
        relevant_retrieved = set(retrieved_chunks).intersection(relevant_chunks)
        return len(relevant_retrieved) / len(retrieved_chunks)
    
    @staticmethod
    def calculate_all(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Calculate all retrieval metrics
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary of metric name -> score
        """
        results = {}
        
        for k in k_values:
            results[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
            results[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
            results[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k)
        
        # Single query MRR (reciprocal rank)
        results["reciprocal_rank"] = RetrievalMetrics._reciprocal_rank(retrieved_ids, relevant_ids)
        
        # Context precision (using all retrieved)
        results["context_precision"] = RetrievalMetrics.context_precision(retrieved_ids, relevant_ids)
        
        return results


class ResponseMetrics:
    """
    Response evaluation metrics for comparing LLM answers to human answers
    
    Metrics:
    - Exact Match: Binary - does the answer exactly match?
    - F1 Score: Token-level overlap
    - Semantic Similarity: Embedding-based similarity
    """
    
    @staticmethod
    def exact_match(predicted: str, ground_truth: str, normalize: bool = True) -> float:
        """
        Calculate exact match score
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            normalize: Whether to normalize text before comparison
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if normalize:
            predicted = ResponseMetrics._normalize_text(predicted)
            ground_truth = ResponseMetrics._normalize_text(ground_truth)
        
        return 1.0 if predicted == ground_truth else 0.0
    
    @staticmethod
    def f1_score(predicted: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        pred_tokens = ResponseMetrics._tokenize(predicted)
        truth_tokens = ResponseMetrics._tokenize(ground_truth)
        
        if not pred_tokens or not truth_tokens:
            return 0.0 if not pred_tokens and not truth_tokens else 0.0
        
        # Count common tokens
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def semantic_similarity(
        predicted: str, 
        ground_truth: str, 
        embedding_func=None
    ) -> float:
        """
        Calculate semantic similarity using embeddings
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            embedding_func: Function that takes text and returns embedding vector
            
        Returns:
            Cosine similarity score (-1.0 to 1.0, typically 0.0 to 1.0)
        """
        if embedding_func is None:
            # Default: use simple word overlap as proxy
            return ResponseMetrics._word_overlap_similarity(predicted, ground_truth)
        
        pred_embedding = np.array(embedding_func(predicted))
        truth_embedding = np.array(embedding_func(ground_truth))
        
        # Cosine similarity
        dot_product = np.dot(pred_embedding, truth_embedding)
        norm_pred = np.linalg.norm(pred_embedding)
        norm_truth = np.linalg.norm(truth_embedding)
        
        if norm_pred == 0 or norm_truth == 0:
            return 0.0
        
        return dot_product / (norm_pred * norm_truth)
    
    @staticmethod
    def _word_overlap_similarity(text1: str, text2: str) -> float:
        """Simple word overlap similarity as fallback"""
        words1 = set(ResponseMetrics._tokenize(text1))
        words2 = set(ResponseMetrics._tokenize(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower().strip()
        # Remove punctuation
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        text = ResponseMetrics._normalize_text(text)
        return text.split()
    
    @staticmethod
    def calculate_all(
        predicted: str, 
        ground_truth: str,
        embedding_func=None
    ) -> Dict[str, float]:
        """
        Calculate all response metrics
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            embedding_func: Optional embedding function for semantic similarity
            
        Returns:
            Dictionary of metric name -> score
        """
        return {
            "exact_match": ResponseMetrics.exact_match(predicted, ground_truth),
            "f1_score": ResponseMetrics.f1_score(predicted, ground_truth),
            "semantic_similarity": ResponseMetrics.semantic_similarity(
                predicted, ground_truth, embedding_func
            ),
        }


class BenchmarkEvaluator:
    """
    Comprehensive evaluator for RAG benchmarking
    Aggregates results across multiple queries and configurations
    """
    
    def __init__(self, embedding_func=None):
        """
        Initialize evaluator
        
        Args:
            embedding_func: Function to compute embeddings for semantic similarity
        """
        self.embedding_func = embedding_func
        self.results = {}
    
    def evaluate_retrieval(
        self,
        config_name: str,
        query_id: str,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for a single query
        
        Args:
            config_name: Configuration name (e.g., "pinecone_ivf")
            query_id: Query identifier
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: Set of relevant chunk IDs
            k_values: k values for metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = RetrievalMetrics.calculate_all(retrieved_ids, relevant_ids, k_values)
        
        # Store results
        if config_name not in self.results:
            self.results[config_name] = {"retrieval": {}, "response": {}}
        
        self.results[config_name]["retrieval"][query_id] = metrics
        
        return metrics
    
    def evaluate_response(
        self,
        config_name: str,
        query_id: str,
        predicted_answer: str,
        ground_truth_answer: str
    ) -> Dict[str, float]:
        """
        Evaluate response for a single query
        
        Args:
            config_name: Configuration name
            query_id: Query identifier
            predicted_answer: LLM-generated answer
            ground_truth_answer: Human ground truth answer
            
        Returns:
            Dictionary of metrics
        """
        metrics = ResponseMetrics.calculate_all(
            predicted_answer, 
            ground_truth_answer,
            self.embedding_func
        )
        
        # Store results
        if config_name not in self.results:
            self.results[config_name] = {"retrieval": {}, "response": {}}
        
        self.results[config_name]["response"][query_id] = metrics
        
        return metrics
    
    def get_aggregate_results(self, config_name: str = None) -> Dict:
        """
        Get aggregated results across all queries
        
        Args:
            config_name: Optional specific configuration to aggregate
            
        Returns:
            Aggregated metrics with mean, std, min, max
        """
        configs_to_process = [config_name] if config_name else list(self.results.keys())
        
        aggregated = {}
        
        for config in configs_to_process:
            if config not in self.results:
                continue
            
            config_results = self.results[config]
            aggregated[config] = {
                "retrieval": self._aggregate_metrics(config_results.get("retrieval", {})),
                "response": self._aggregate_metrics(config_results.get("response", {})),
            }
        
        return aggregated
    
    def _aggregate_metrics(self, query_metrics: Dict[str, Dict[str, float]]) -> Dict:
        """Aggregate metrics across queries"""
        if not query_metrics:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for metrics in query_metrics.values():
            all_metrics.update(metrics.keys())
        
        aggregated = {}
        for metric_name in all_metrics:
            values = [
                metrics.get(metric_name, 0.0) 
                for metrics in query_metrics.values()
            ]
            
            aggregated[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "n": len(values),
            }
        
        return aggregated
    
    def get_comparison_table(self) -> Dict[str, Dict[str, float]]:
        """
        Get comparison table of mean metrics across all configurations
        
        Returns:
            Dict[config_name -> Dict[metric_name -> mean_value]]
        """
        aggregated = self.get_aggregate_results()
        
        comparison = {}
        for config_name, config_results in aggregated.items():
            comparison[config_name] = {}
            
            # Retrieval metrics
            for metric_name, stats in config_results.get("retrieval", {}).items():
                comparison[config_name][f"retrieval_{metric_name}"] = stats["mean"]
            
            # Response metrics
            for metric_name, stats in config_results.get("response", {}).items():
                comparison[config_name][f"response_{metric_name}"] = stats["mean"]
        
        return comparison
