"""
RAG Vector Store Benchmarking System

Complete benchmarking pipeline that:
1. Loads document from file (PDF, TXT, MD)
2. Chunks and embeds the document
3. Runs benchmarks on 6 configurations:
   - Pinecone (IVF, HNSW)
   - ChromaDB (IVF, HNSW)
   - PostgreSQL (IVF, HNSW)
4. Evaluates retrieval quality against ground truth
5. Generates comparison report

Usage:
    python main.py --document path/to/document.pdf --ground-truth path/to/qa.json
    python main.py --demo  # Run with sample data
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    BENCHMARK_CONFIGURATIONS, EVALUATION_CONFIG
)
from src.data_extraction.pdf_extractor import extract_document, DocumentExtractor
from src.embeddings.text_chunker import TextChunker
from src.embeddings.embedding_service import EmbeddingService
from src.evaluation.ground_truth import GroundTruth
from src.evaluation.metrics import BenchmarkEvaluator
from src.benchmark.benchmark_suite import BenchmarkSuite
from src.vector_stores.factory import VectorStoreFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGBenchmarkRunner:
    """
    Main orchestrator for the RAG benchmarking pipeline.
    """
    
    def __init__(
        self,
        output_dir: str = "./results",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory for results output
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize services
        self.embedding_service = EmbeddingService()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.benchmark_suite = BenchmarkSuite(
            output_dir=str(self.output_dir),
            embedding_service=self.embedding_service
        )
        
        # Data storage
        self.document_text: str = ""
        self.chunks: List[str] = []
        self.chunk_ids: List[str] = []
        self.embeddings: List[List[float]] = []
        self.ground_truth: Optional[GroundTruth] = None
    
    def load_document(self, document_path: str) -> Dict:
        """
        Load and extract text from a document.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Document statistics
        """
        logger.info(f"Loading document: {document_path}")
        
        extractor = DocumentExtractor(document_path)
        self.document_text = extractor.extract()
        
        stats = {
            "path": document_path,
            "characters": len(self.document_text),
            "words": len(self.document_text.split()),
            "lines": self.document_text.count('\n') + 1
        }
        
        logger.info(f"Loaded document: {stats['characters']} chars, {stats['words']} words")
        return stats
    
    def chunk_document(self) -> Dict:
        """
        Chunk the loaded document into smaller pieces.
        
        Returns:
            Chunking statistics
        """
        if not self.document_text:
            raise ValueError("No document loaded. Call load_document() first.")
        
        logger.info(f"Chunking document (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        self.chunks = self.text_chunker.chunk_text(self.document_text)
        self.chunk_ids = [f"chunk_{i:04d}" for i in range(len(self.chunks))]
        
        stats = {
            "num_chunks": len(self.chunks),
            "avg_chunk_length": sum(len(c) for c in self.chunks) / len(self.chunks) if self.chunks else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        
        logger.info(f"Created {stats['num_chunks']} chunks (avg {stats['avg_chunk_length']:.0f} chars)")
        return stats
    
    def embed_chunks(self) -> Dict:
        """
        Create embeddings for all chunks.
        
        Returns:
            Embedding statistics
        """
        if not self.chunks:
            raise ValueError("No chunks created. Call chunk_document() first.")
        
        logger.info(f"Embedding {len(self.chunks)} chunks...")
        
        self.embeddings = self.embedding_service.embed_batch(self.chunks)
        
        stats = {
            "num_embeddings": len(self.embeddings),
            "embedding_dimension": len(self.embeddings[0]) if self.embeddings else 0,
            "model": EMBEDDING_MODEL
        }
        
        logger.info(f"Created {stats['num_embeddings']} embeddings (dim={stats['embedding_dimension']})")
        return stats
    
    def load_ground_truth(self, ground_truth_path: str) -> Dict:
        """
        Load ground truth Q&A data from JSON file.
        
        Expected format:
        {
            "questions": [
                {
                    "id": "q1",
                    "text": "What is...?",
                    "answer": "The answer is...",
                    "relevant_chunks": ["chunk_0001", "chunk_0023"]
                }
            ]
        }
        
        Args:
            ground_truth_path: Path to ground truth JSON
            
        Returns:
            Ground truth statistics
        """
        logger.info(f"Loading ground truth: {ground_truth_path}")
        
        self.ground_truth = GroundTruth()
        self.ground_truth.load_from_json(ground_truth_path)
        
        stats = {
            "num_questions": len(self.ground_truth.questions),
            "path": ground_truth_path
        }
        
        logger.info(f"Loaded {stats['num_questions']} ground truth questions")
        return stats
    
    def run_benchmarks(
        self,
        configurations: List[Dict] = None,
        collection_name: str = "benchmark"
    ) -> Dict:
        """
        Run benchmarks on all (or specified) configurations.
        
        Args:
            configurations: List of configurations to test (default: all 6)
            collection_name: Base name for collections
            
        Returns:
            Benchmark results
        """
        if not self.embeddings:
            raise ValueError("No embeddings created. Call embed_chunks() first.")
        
        if configurations is None:
            configurations = BENCHMARK_CONFIGURATIONS
        
        # Prepare ground truth data
        if self.ground_truth and self.ground_truth.questions:
            query_texts = [q.text for q in self.ground_truth.questions]
            relevant_chunk_ids = [
                self.ground_truth.get_relevant_chunks(q.id)
                for q in self.ground_truth.questions
            ]
        else:
            # Use sample queries if no ground truth
            query_texts = self._get_sample_queries()
            relevant_chunk_ids = [[] for _ in query_texts]  # No ground truth
            logger.warning("No ground truth loaded - retrieval quality metrics will be zeros")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING BENCHMARKS ON {len(configurations)} CONFIGURATIONS")
        logger.info(f"{'='*60}")
        
        results = self.benchmark_suite.run_all_configurations(
            texts=self.chunks,
            embeddings=self.embeddings,
            chunk_ids=self.chunk_ids,
            query_texts=query_texts,
            relevant_chunk_ids=relevant_chunk_ids,
            configurations=configurations,
            collection_name=collection_name
        )
        
        return results
    
    def _get_sample_queries(self) -> List[str]:
        """Get sample queries for testing when no ground truth is provided."""
        return [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the important concepts discussed?",
            "What conclusions are drawn?",
            "What are the recommendations?"
        ]
    
    def generate_report(self) -> Dict:
        """
        Generate final comparison report.
        
        Returns:
            Comparison report
        """
        report = self.benchmark_suite.generate_comparison_report()
        return report
    
    def save_results(self, filename: str = None) -> str:
        """
        Save benchmark results to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        return self.benchmark_suite.save_results(filename)
    
    def print_summary(self):
        """Print formatted benchmark summary."""
        self.benchmark_suite.print_summary()


def create_sample_ground_truth(output_path: str, num_questions: int = 5):
    """
    Create a sample ground truth template file.
    
    Args:
        output_path: Path to save the template
        num_questions: Number of sample questions
    """
    template = {
        "description": "Ground truth Q&A for benchmarking. Fill in with actual questions and answers.",
        "instructions": [
            "1. Replace sample questions with actual questions about your document",
            "2. Provide the expected/correct answer for each question",
            "3. List the chunk IDs that contain relevant information",
            "4. Chunk IDs will be in format 'chunk_XXXX' after document processing"
        ],
        "questions": [
            {
                "id": f"q{i+1}",
                "text": f"Sample question {i+1}: What is...?",
                "answer": f"Sample answer {i+1}: The answer is...",
                "relevant_chunks": ["chunk_0001", "chunk_0002"]
            }
            for i in range(num_questions)
        ]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Created sample ground truth template: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Vector Store Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --document data/document.pdf --ground-truth data/qa.json
  python main.py --demo
  python main.py --create-template data/ground_truth/template.json
  python main.py --list-configs
        """
    )
    
    parser.add_argument(
        "--document", "-d",
        type=str,
        help="Path to document file (PDF, TXT, or MD)"
    )
    
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        help="Path to ground truth JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    parser.add_argument(
        "--configs", "-c",
        type=str,
        nargs="+",
        help="Specific configurations to test (e.g., chromadb_hnsw postgresql_ivf)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with sample data"
    )
    
    parser.add_argument(
        "--create-template",
        type=str,
        metavar="PATH",
        help="Create sample ground truth template at specified path"
    )
    
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available benchmark configurations"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size (default: {CHUNK_SIZE})"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Chunk overlap (default: {CHUNK_OVERLAP})"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_template:
        create_sample_ground_truth(args.create_template)
        return
    
    if args.list_configs:
        print("\nAvailable Benchmark Configurations:")
        print("-" * 40)
        for config in BENCHMARK_CONFIGURATIONS:
            print(f"  {config['name']}: {config['db']} with {config['index_type']}")
        return
    
    # Run benchmark
    if args.demo:
        run_demo(args.output)
    elif args.document:
        run_benchmark(
            document_path=args.document,
            ground_truth_path=args.ground_truth,
            output_dir=args.output,
            config_names=args.configs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    else:
        parser.print_help()
        print("\n⚠️  Please provide --document or --demo flag")


def run_benchmark(
    document_path: str,
    ground_truth_path: str = None,
    output_dir: str = "./results",
    config_names: List[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
):
    """
    Run the complete benchmark pipeline.
    """
    print("\n" + "="*60)
    print("🚀 RAG VECTOR STORE BENCHMARKING SYSTEM")
    print("="*60)
    
    # Initialize runner
    runner = RAGBenchmarkRunner(
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Step 1: Load document
    print("\n📄 Step 1: Loading document...")
    doc_stats = runner.load_document(document_path)
    print(f"   ✓ Loaded: {doc_stats['words']:,} words, {doc_stats['characters']:,} characters")
    
    # Step 2: Chunk document
    print("\n📦 Step 2: Chunking document...")
    chunk_stats = runner.chunk_document()
    print(f"   ✓ Created {chunk_stats['num_chunks']} chunks")
    
    # Step 3: Create embeddings
    print("\n🔢 Step 3: Creating embeddings...")
    embed_stats = runner.embed_chunks()
    print(f"   ✓ Created {embed_stats['num_embeddings']} embeddings (dim={embed_stats['embedding_dimension']})")
    
    # Step 4: Load ground truth (optional)
    if ground_truth_path:
        print("\n📋 Step 4: Loading ground truth...")
        gt_stats = runner.load_ground_truth(ground_truth_path)
        print(f"   ✓ Loaded {gt_stats['num_questions']} questions")
    else:
        print("\n📋 Step 4: No ground truth provided (using sample queries)")
    
    # Step 5: Filter configurations if specified
    configurations = BENCHMARK_CONFIGURATIONS
    if config_names:
        configurations = [c for c in configurations if c["name"] in config_names]
        print(f"\n🎯 Running {len(configurations)} selected configurations")
    
    # Step 6: Run benchmarks
    print("\n⚡ Step 5: Running benchmarks...")
    runner.run_benchmarks(configurations=configurations)
    
    # Step 7: Generate and save results
    print("\n💾 Step 6: Saving results...")
    results_path = runner.save_results()
    print(f"   ✓ Results saved to: {results_path}")
    
    # Print summary
    runner.print_summary()
    
    print("\n✅ Benchmarking complete!")


def run_demo(output_dir: str = "./results"):
    """
    Run a demo benchmark with sample data.
    """
    print("\n" + "="*60)
    print("🎮 DEMO MODE - Running with sample data")
    print("="*60)
    
    # Create sample document
    sample_text = """
    Machine Learning and Artificial Intelligence
    
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on developing
    computer programs that can access data and use it to learn for themselves.
    
    Types of Machine Learning:
    
    1. Supervised Learning: The algorithm learns from labeled training data and makes predictions.
    Common applications include spam detection, image classification, and price prediction.
    
    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
    Common applications include customer segmentation, anomaly detection, and recommendation systems.
    
    3. Reinforcement Learning: The algorithm learns through trial and error, receiving rewards
    or penalties for actions. Common applications include game playing, robotics, and autonomous vehicles.
    
    Deep Learning:
    
    Deep learning is a subset of machine learning that uses neural networks with many layers.
    These deep neural networks can learn complex patterns from large amounts of data.
    Key architectures include Convolutional Neural Networks (CNNs) for image processing
    and Recurrent Neural Networks (RNNs) for sequential data.
    
    Natural Language Processing:
    
    NLP is a field of AI that focuses on the interaction between computers and human language.
    Applications include sentiment analysis, machine translation, chatbots, and text summarization.
    Modern NLP heavily relies on transformer architectures and large language models.
    """
    
    # Create sample ground truth
    sample_ground_truth = {
        "questions": [
            {
                "id": "q1",
                "text": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                "relevant_chunks": ["chunk_0000", "chunk_0001"]
            },
            {
                "id": "q2",
                "text": "What are the types of machine learning?",
                "answer": "The three main types are supervised learning, unsupervised learning, and reinforcement learning.",
                "relevant_chunks": ["chunk_0001", "chunk_0002"]
            },
            {
                "id": "q3",
                "text": "What is deep learning?",
                "answer": "Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns from large amounts of data.",
                "relevant_chunks": ["chunk_0002", "chunk_0003"]
            }
        ]
    }
    
    # Save sample data to temp files
    data_dir = Path("./data/demo")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    doc_path = data_dir / "sample_document.txt"
    with open(doc_path, 'w') as f:
        f.write(sample_text)
    
    gt_path = data_dir / "sample_ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(sample_ground_truth, f, indent=2)
    
    # Run benchmark with only ChromaDB (quick demo)
    print("\n⚠️  Demo mode: Testing only ChromaDB configurations for speed")
    
    run_benchmark(
        document_path=str(doc_path),
        ground_truth_path=str(gt_path),
        output_dir=output_dir,
        config_names=["chromadb_ivf", "chromadb_hnsw"],
        chunk_size=200,
        chunk_overlap=50
    )


if __name__ == "__main__":
    main()
