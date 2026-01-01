"""
Comprehensive Benchmarking Suite

Runs benchmarks across all 6 vector store configurations:
- Pinecone (IVF, HNSW)
- ChromaDB (IVF, HNSW)  
- PostgreSQL (IVF, HNSW)

Collects:
- Storage performance metrics
- Search performance metrics
- Retrieval accuracy (Recall@k, MRR, Context Precision, NDCG)
- Response quality metrics (F1, Exact Match, Semantic Similarity)
"""

from typing import Dict, List, Tuple, Optional, Any
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict

from config.settings import BENCHMARK_CONFIGURATIONS, EVALUATION_CONFIG
from src.vector_stores.factory import VectorStoreFactory
from src.evaluation.metrics import RetrievalMetrics, ResponseMetrics, BenchmarkEvaluator
from src.evaluation.ground_truth import GroundTruth
from src.embeddings.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StorageMetrics:
    """Storage performance metrics"""
    vectors_stored: int = 0
    insert_time_seconds: float = 0.0
    index_build_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    throughput_vectors_per_second: float = 0.0


@dataclass
class SearchMetrics:
    """Search performance metrics"""
    num_queries: int = 0
    avg_search_time_ms: float = 0.0
    min_search_time_ms: float = 0.0
    max_search_time_ms: float = 0.0
    p50_search_time_ms: float = 0.0
    p95_search_time_ms: float = 0.0
    p99_search_time_ms: float = 0.0
    throughput_qps: float = 0.0


@dataclass
class RetrievalQualityMetrics:
    """Retrieval quality metrics"""
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    context_precision: float = 0.0


@dataclass
class ResponseQualityMetrics:
    """Response quality metrics"""
    exact_match: float = 0.0
    f1_score: float = 0.0
    semantic_similarity: float = 0.0


@dataclass
class ConfigurationBenchmarkResult:
    """Complete benchmark results for a single configuration"""
    config_name: str
    db_type: str
    index_type: str
    storage: StorageMetrics
    search: SearchMetrics
    retrieval_quality: RetrievalQualityMetrics
    response_quality: Optional[ResponseQualityMetrics] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for comparing vector store configurations.
    """
    
    def __init__(
        self, 
        output_dir: str = "./results",
        ground_truth: Optional[GroundTruth] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize Benchmark Suite
        
        Args:
            output_dir: Directory to save benchmark results
            ground_truth: Ground truth data for evaluation
            embedding_service: Service for creating embeddings
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ground_truth = ground_truth
        self.embedding_service = embedding_service or EmbeddingService()
        
        self.retrieval_metrics = RetrievalMetrics()
        self.response_metrics = ResponseMetrics()
        self.evaluator = BenchmarkEvaluator()
        
        self.results: Dict[str, ConfigurationBenchmarkResult] = {}
    
    def run_storage_benchmark(
        self, 
        vector_store, 
        texts: List[str], 
        embeddings: List[List[float]],
        chunk_ids: List[str] = None
    ) -> StorageMetrics:
        """
        Benchmark storage performance.
        
        Args:
            vector_store: Vector store instance
            texts: List of text chunks
            embeddings: List of embeddings
            chunk_ids: Optional chunk IDs
            
        Returns:
            Storage metrics
        """
        try:
            start_time = time.time()
            
            # Store returns stats dict
            store_result = vector_store.store(
                texts=texts,
                embeddings=embeddings,
                chunk_ids=chunk_ids
            )
            
            total_time = time.time() - start_time
            
            metrics = StorageMetrics(
                vectors_stored=store_result.get("vectors_stored", len(texts)),
                insert_time_seconds=store_result.get("insert_time_seconds", total_time),
                index_build_time_seconds=store_result.get("index_build_time_seconds", 0.0),
                total_time_seconds=store_result.get("total_time_seconds", total_time),
                throughput_vectors_per_second=len(texts) / total_time if total_time > 0 else 0.0
            )
            
            logger.info(f"Storage: {metrics.vectors_stored} vectors in {metrics.total_time_seconds:.2f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"Storage benchmark error: {str(e)}")
            return StorageMetrics()
    
    def run_search_benchmark(
        self,
        vector_store,
        query_embeddings: List[List[float]],
        top_k: int = 10,
        num_warmup: int = 3
    ) -> Tuple[SearchMetrics, List[Tuple[List[str], List[float]]]]:
        """
        Benchmark search performance.
        
        Args:
            vector_store: Vector store instance
            query_embeddings: List of query embeddings
            top_k: Number of results per query
            num_warmup: Warmup queries before timing
            
        Returns:
            Tuple of (search metrics, list of (chunk_ids, scores) for each query)
        """
        try:
            # Warmup queries
            for i in range(min(num_warmup, len(query_embeddings))):
                vector_store.search_with_ids(query_embeddings[i], top_k=top_k)
            
            # Timed queries
            search_times = []
            all_results = []
            
            for query_embedding in query_embeddings:
                chunk_ids, scores, search_time = vector_store.search_with_ids(
                    query_embedding, top_k=top_k
                )
                search_times.append(search_time * 1000)  # Convert to ms
                all_results.append((chunk_ids, scores))
            
            search_times = np.array(search_times)
            total_time = search_times.sum() / 1000  # Back to seconds
            
            metrics = SearchMetrics(
                num_queries=len(query_embeddings),
                avg_search_time_ms=float(np.mean(search_times)),
                min_search_time_ms=float(np.min(search_times)),
                max_search_time_ms=float(np.max(search_times)),
                p50_search_time_ms=float(np.percentile(search_times, 50)),
                p95_search_time_ms=float(np.percentile(search_times, 95)),
                p99_search_time_ms=float(np.percentile(search_times, 99)),
                throughput_qps=len(query_embeddings) / total_time if total_time > 0 else 0.0
            )
            
            logger.info(f"Search: {metrics.avg_search_time_ms:.2f}ms avg, {metrics.throughput_qps:.1f} QPS")
            return metrics, all_results
            
        except Exception as e:
            logger.error(f"Search benchmark error: {str(e)}")
            return SearchMetrics(), []
    
    def evaluate_retrieval_quality(
        self,
        retrieved_results: List[Tuple[List[str], List[float]]],
        relevant_chunk_ids: List[List[str]]
    ) -> RetrievalQualityMetrics:
        """
        Evaluate retrieval quality against ground truth.
        
        Args:
            retrieved_results: List of (chunk_ids, scores) for each query
            relevant_chunk_ids: List of relevant chunk IDs for each query
            
        Returns:
            Retrieval quality metrics
        """
        try:
            recalls_at_1 = []
            recalls_at_5 = []
            recalls_at_10 = []
            mrrs = []
            ndcgs_at_5 = []
            ndcgs_at_10 = []
            context_precisions = []
            
            for (retrieved_ids, _), relevant_ids in zip(retrieved_results, relevant_chunk_ids):
                # Recall@k
                recalls_at_1.append(self.retrieval_metrics.recall_at_k(retrieved_ids[:1], relevant_ids))
                recalls_at_5.append(self.retrieval_metrics.recall_at_k(retrieved_ids[:5], relevant_ids))
                recalls_at_10.append(self.retrieval_metrics.recall_at_k(retrieved_ids[:10], relevant_ids))
                
                # MRR
                mrrs.append(self.retrieval_metrics.mrr(retrieved_ids, relevant_ids))
                
                # NDCG
                ndcgs_at_5.append(self.retrieval_metrics.ndcg_at_k(retrieved_ids[:5], relevant_ids))
                ndcgs_at_10.append(self.retrieval_metrics.ndcg_at_k(retrieved_ids[:10], relevant_ids))
                
                # Context Precision
                context_precisions.append(self.retrieval_metrics.context_precision(retrieved_ids, relevant_ids))
            
            metrics = RetrievalQualityMetrics(
                recall_at_1=float(np.mean(recalls_at_1)),
                recall_at_5=float(np.mean(recalls_at_5)),
                recall_at_10=float(np.mean(recalls_at_10)),
                mrr=float(np.mean(mrrs)),
                ndcg_at_5=float(np.mean(ndcgs_at_5)),
                ndcg_at_10=float(np.mean(ndcgs_at_10)),
                context_precision=float(np.mean(context_precisions))
            )
            
            logger.info(f"Retrieval Quality: Recall@5={metrics.recall_at_5:.3f}, MRR={metrics.mrr:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Retrieval quality evaluation error: {str(e)}")
            return RetrievalQualityMetrics()
    
    def evaluate_response_quality(
        self,
        generated_responses: List[str],
        ground_truth_responses: List[str]
    ) -> ResponseQualityMetrics:
        """
        Evaluate response quality against ground truth.
        
        Args:
            generated_responses: Generated RAG responses
            ground_truth_responses: Human ground truth responses
            
        Returns:
            Response quality metrics
        """
        try:
            exact_matches = []
            f1_scores = []
            semantic_sims = []
            
            for generated, truth in zip(generated_responses, ground_truth_responses):
                exact_matches.append(float(self.response_metrics.exact_match(generated, truth)))
                f1_scores.append(self.response_metrics.f1_score(generated, truth))
                semantic_sims.append(self.response_metrics.semantic_similarity(
                    generated, truth, self.embedding_service
                ))
            
            metrics = ResponseQualityMetrics(
                exact_match=float(np.mean(exact_matches)),
                f1_score=float(np.mean(f1_scores)),
                semantic_similarity=float(np.mean(semantic_sims))
            )
            
            logger.info(f"Response Quality: F1={metrics.f1_score:.3f}, Semantic={metrics.semantic_similarity:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Response quality evaluation error: {str(e)}")
            return ResponseQualityMetrics()
    
    def run_single_configuration(
        self,
        config: Dict,
        texts: List[str],
        embeddings: List[List[float]],
        chunk_ids: List[str],
        query_embeddings: List[List[float]],
        relevant_chunk_ids: List[List[str]],
        collection_name: str = "benchmark"
    ) -> ConfigurationBenchmarkResult:
        """
        Run complete benchmark for a single configuration.
        
        Args:
            config: Configuration dict with 'db', 'index_type', 'name'
            texts: Document chunks
            embeddings: Chunk embeddings
            chunk_ids: Chunk identifiers
            query_embeddings: Query embeddings
            relevant_chunk_ids: Ground truth relevant chunks per query
            collection_name: Base name for collection
            
        Returns:
            Complete benchmark results
        """
        config_name = config["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark: {config_name}")
        logger.info(f"{'='*60}")
        
        errors = []
        
        try:
            # Create vector store
            vector_store = VectorStoreFactory.create_from_config(
                config, collection_name=collection_name
            )
            
            # Storage benchmark
            storage_metrics = self.run_storage_benchmark(
                vector_store, texts, embeddings, chunk_ids
            )
            
            # Search benchmark
            search_metrics, search_results = self.run_search_benchmark(
                vector_store, query_embeddings, top_k=10
            )
            
            # Retrieval quality
            retrieval_quality = self.evaluate_retrieval_quality(
                search_results, relevant_chunk_ids
            )
            
            result = ConfigurationBenchmarkResult(
                config_name=config_name,
                db_type=config["db"],
                index_type=config["index_type"],
                storage=storage_metrics,
                search=search_metrics,
                retrieval_quality=retrieval_quality,
                errors=errors
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Configuration {config_name} failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return ConfigurationBenchmarkResult(
                config_name=config_name,
                db_type=config["db"],
                index_type=config["index_type"],
                storage=StorageMetrics(),
                search=SearchMetrics(),
                retrieval_quality=RetrievalQualityMetrics(),
                errors=errors
            )
    
    def run_all_configurations(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        chunk_ids: List[str],
        query_texts: List[str],
        relevant_chunk_ids: List[List[str]],
        configurations: List[Dict] = None,
        collection_name: str = "benchmark"
    ) -> Dict[str, ConfigurationBenchmarkResult]:
        """
        Run benchmarks on all configurations.
        
        Args:
            texts: Document chunks
            embeddings: Chunk embeddings
            chunk_ids: Chunk identifiers
            query_texts: Query texts (will be embedded)
            relevant_chunk_ids: Ground truth relevant chunks per query
            configurations: List of configs to run (default: all 6)
            collection_name: Base name for collections
            
        Returns:
            Dict mapping config name to results
        """
        if configurations is None:
            configurations = BENCHMARK_CONFIGURATIONS
        
        # Generate query embeddings
        logger.info(f"Generating embeddings for {len(query_texts)} queries...")
        query_embeddings = self.embedding_service.embed_batch(query_texts)
        
        results = {}
        
        for config in configurations:
            result = self.run_single_configuration(
                config=config,
                texts=texts,
                embeddings=embeddings,
                chunk_ids=chunk_ids,
                query_embeddings=query_embeddings,
                relevant_chunk_ids=relevant_chunk_ids,
                collection_name=collection_name
            )
            results[config["name"]] = result
            self.results[config["name"]] = result
        
        return results
    
    def generate_comparison_report(self) -> Dict:
        """
        Generate a comparison report across all configurations.
        
        Returns:
            Comparison report with rankings
        """
        if not self.results:
            return {"error": "No results to compare"}
        
        report = {
            "summary": {},
            "rankings": {},
            "details": {}
        }
        
        # Extract metrics for comparison
        metrics_data = {
            "storage_time": {},
            "search_latency": {},
            "throughput": {},
            "recall_at_5": {},
            "mrr": {},
            "context_precision": {}
        }
        
        for name, result in self.results.items():
            metrics_data["storage_time"][name] = result.storage.total_time_seconds
            metrics_data["search_latency"][name] = result.search.avg_search_time_ms
            metrics_data["throughput"][name] = result.search.throughput_qps
            metrics_data["recall_at_5"][name] = result.retrieval_quality.recall_at_5
            metrics_data["mrr"][name] = result.retrieval_quality.mrr
            metrics_data["context_precision"][name] = result.retrieval_quality.context_precision
        
        # Generate rankings (lower is better for time/latency, higher for others)
        for metric, data in metrics_data.items():
            reverse = metric not in ["storage_time", "search_latency"]
            sorted_configs = sorted(data.items(), key=lambda x: x[1], reverse=reverse)
            report["rankings"][metric] = [
                {"rank": i+1, "config": name, "value": round(value, 4)}
                for i, (name, value) in enumerate(sorted_configs)
            ]
        
        # Summary: best config for each metric
        report["summary"] = {
            "fastest_storage": report["rankings"]["storage_time"][0]["config"],
            "lowest_latency": report["rankings"]["search_latency"][0]["config"],
            "highest_throughput": report["rankings"]["throughput"][0]["config"],
            "best_recall": report["rankings"]["recall_at_5"][0]["config"],
            "best_mrr": report["rankings"]["mrr"][0]["config"],
            "best_context_precision": report["rankings"]["context_precision"][0]["config"]
        }
        
        # Detailed results
        for name, result in self.results.items():
            report["details"][name] = {
                "storage": asdict(result.storage),
                "search": asdict(result.search),
                "retrieval_quality": asdict(result.retrieval_quality),
                "errors": result.errors
            }
        
        return report
    
    def save_results(self, filename: str = None) -> str:
        """
        Save benchmark results to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare serializable data
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_configurations": len(self.results),
            "configurations": [config["name"] for config in BENCHMARK_CONFIGURATIONS],
            "comparison_report": self.generate_comparison_report(),
            "raw_results": {
                name: {
                    "config_name": result.config_name,
                    "db_type": result.db_type,
                    "index_type": result.index_type,
                    "storage": asdict(result.storage),
                    "search": asdict(result.search),
                    "retrieval_quality": asdict(result.retrieval_quality),
                    "response_quality": asdict(result.response_quality) if result.response_quality else None,
                    "errors": result.errors
                }
                for name, result in self.results.items()
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved results to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return ""
    
    def print_summary(self):
        """Print a formatted summary of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        report = self.generate_comparison_report()
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        print("\n📊 BEST PERFORMERS:")
        print("-"*40)
        for metric, config in report["summary"].items():
            metric_display = metric.replace("_", " ").title()
            print(f"  {metric_display}: {config}")
        
        print("\n📈 RANKINGS BY METRIC:")
        print("-"*40)
        
        for metric, rankings in report["rankings"].items():
            metric_display = metric.replace("_", " ").title()
            print(f"\n  {metric_display}:")
            for item in rankings[:3]:  # Top 3
                print(f"    #{item['rank']} {item['config']}: {item['value']}")
        
        print("\n" + "="*80)
