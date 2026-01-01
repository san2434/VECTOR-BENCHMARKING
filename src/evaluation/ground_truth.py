"""
Ground Truth Management Module
Handles questions, human answers, and relevant chunk mappings for evaluation
"""

import json
import os
import logging
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Represents a benchmark question"""
    id: str
    text: str
    category: Optional[str] = None
    difficulty: Optional[str] = None  # easy, medium, hard
    expected_answer_type: Optional[str] = None  # factual, analytical, summary


@dataclass
class HumanAnswer:
    """Represents a human-provided ground truth answer"""
    question_id: str
    answer: str
    confidence: Optional[float] = None  # 0.0 to 1.0
    annotator: Optional[str] = None
    notes: Optional[str] = None


@dataclass 
class RelevantChunks:
    """Represents the relevant chunks for a question"""
    question_id: str
    chunk_ids: List[str]
    primary_chunk_id: Optional[str] = None  # Most important chunk
    relevance_scores: Optional[Dict[str, float]] = None  # chunk_id -> relevance (0-1)


class GroundTruth:
    """
    Manages ground truth data for RAG evaluation
    
    Structure:
    - questions.json: List of questions
    - human_answers.json: Human-provided answers
    - relevant_chunks.json: Mapping of questions to relevant chunks
    """
    
    def __init__(self, ground_truth_dir: str = "./data/ground_truth"):
        """
        Initialize Ground Truth manager
        
        Args:
            ground_truth_dir: Directory to store ground truth files
        """
        self.ground_truth_dir = ground_truth_dir
        self.questions: Dict[str, Question] = {}
        self.human_answers: Dict[str, HumanAnswer] = {}
        self.relevant_chunks: Dict[str, RelevantChunks] = {}
        
        # Create directory if it doesn't exist
        os.makedirs(ground_truth_dir, exist_ok=True)
        
        # Load existing data if available
        self._load_all()
    
    def _load_all(self):
        """Load all ground truth data from files"""
        self._load_questions()
        self._load_human_answers()
        self._load_relevant_chunks()
    
    def _load_questions(self):
        """Load questions from file"""
        filepath = os.path.join(self.ground_truth_dir, "questions.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                for q in data.get("questions", []):
                    question = Question(**q)
                    self.questions[question.id] = question
                logger.info(f"Loaded {len(self.questions)} questions")
            except Exception as e:
                logger.warning(f"Error loading questions: {e}")
    
    def _load_human_answers(self):
        """Load human answers from file"""
        filepath = os.path.join(self.ground_truth_dir, "human_answers.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                for a in data.get("answers", []):
                    answer = HumanAnswer(**a)
                    self.human_answers[answer.question_id] = answer
                logger.info(f"Loaded {len(self.human_answers)} human answers")
            except Exception as e:
                logger.warning(f"Error loading human answers: {e}")
    
    def _load_relevant_chunks(self):
        """Load relevant chunks mapping from file"""
        filepath = os.path.join(self.ground_truth_dir, "relevant_chunks.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                for rc in data.get("mappings", []):
                    chunks = RelevantChunks(**rc)
                    self.relevant_chunks[chunks.question_id] = chunks
                logger.info(f"Loaded {len(self.relevant_chunks)} relevant chunk mappings")
            except Exception as e:
                logger.warning(f"Error loading relevant chunks: {e}")
    
    def save_all(self):
        """Save all ground truth data to files"""
        self._save_questions()
        self._save_human_answers()
        self._save_relevant_chunks()
    
    def _save_questions(self):
        """Save questions to file"""
        filepath = os.path.join(self.ground_truth_dir, "questions.json")
        data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "count": len(self.questions)
            },
            "questions": [asdict(q) for q in self.questions.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.questions)} questions")
    
    def _save_human_answers(self):
        """Save human answers to file"""
        filepath = os.path.join(self.ground_truth_dir, "human_answers.json")
        data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "count": len(self.human_answers)
            },
            "answers": [asdict(a) for a in self.human_answers.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.human_answers)} human answers")
    
    def _save_relevant_chunks(self):
        """Save relevant chunks to file"""
        filepath = os.path.join(self.ground_truth_dir, "relevant_chunks.json")
        data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "count": len(self.relevant_chunks)
            },
            "mappings": [asdict(rc) for rc in self.relevant_chunks.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.relevant_chunks)} relevant chunk mappings")
    
    # =========================================================================
    # Question Management
    # =========================================================================
    
    def add_question(
        self,
        question_id: str,
        text: str,
        category: str = None,
        difficulty: str = None,
        expected_answer_type: str = None
    ) -> Question:
        """Add a new question"""
        question = Question(
            id=question_id,
            text=text,
            category=category,
            difficulty=difficulty,
            expected_answer_type=expected_answer_type
        )
        self.questions[question_id] = question
        return question
    
    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a question by ID"""
        return self.questions.get(question_id)
    
    def get_all_questions(self) -> List[Question]:
        """Get all questions"""
        return list(self.questions.values())
    
    # =========================================================================
    # Human Answer Management
    # =========================================================================
    
    def add_human_answer(
        self,
        question_id: str,
        answer: str,
        confidence: float = None,
        annotator: str = None,
        notes: str = None
    ) -> HumanAnswer:
        """Add a human answer for a question"""
        human_answer = HumanAnswer(
            question_id=question_id,
            answer=answer,
            confidence=confidence,
            annotator=annotator,
            notes=notes
        )
        self.human_answers[question_id] = human_answer
        return human_answer
    
    def get_human_answer(self, question_id: str) -> Optional[str]:
        """Get human answer text for a question"""
        answer = self.human_answers.get(question_id)
        return answer.answer if answer else None
    
    # =========================================================================
    # Relevant Chunks Management
    # =========================================================================
    
    def add_relevant_chunks(
        self,
        question_id: str,
        chunk_ids: List[str],
        primary_chunk_id: str = None,
        relevance_scores: Dict[str, float] = None
    ) -> RelevantChunks:
        """Add relevant chunks for a question"""
        relevant = RelevantChunks(
            question_id=question_id,
            chunk_ids=chunk_ids,
            primary_chunk_id=primary_chunk_id or (chunk_ids[0] if chunk_ids else None),
            relevance_scores=relevance_scores
        )
        self.relevant_chunks[question_id] = relevant
        return relevant
    
    def get_relevant_chunk_ids(self, question_id: str) -> Set[str]:
        """Get set of relevant chunk IDs for a question"""
        relevant = self.relevant_chunks.get(question_id)
        return set(relevant.chunk_ids) if relevant else set()
    
    def get_relevance_scores(self, question_id: str) -> Optional[Dict[str, float]]:
        """Get relevance scores for chunks"""
        relevant = self.relevant_chunks.get(question_id)
        return relevant.relevance_scores if relevant else None
    
    # =========================================================================
    # Evaluation Helpers
    # =========================================================================
    
    def get_evaluation_data(self, question_id: str) -> Dict:
        """
        Get all data needed for evaluation of a single question
        
        Returns:
            Dict with question, human_answer, relevant_chunk_ids
        """
        return {
            "question": self.get_question(question_id),
            "human_answer": self.get_human_answer(question_id),
            "relevant_chunk_ids": self.get_relevant_chunk_ids(question_id),
            "relevance_scores": self.get_relevance_scores(question_id)
        }
    
    def get_all_evaluation_data(self) -> List[Dict]:
        """Get evaluation data for all questions"""
        return [self.get_evaluation_data(q.id) for q in self.questions.values()]
    
    def get_relevant_chunks(self, question_id: str) -> List[str]:
        """
        Get list of relevant chunk IDs for a question.
        
        Used by benchmark suite for evaluation.
        
        Args:
            question_id: Question ID
            
        Returns:
            List of relevant chunk IDs
        """
        relevant = self.relevant_chunks.get(question_id)
        return relevant.chunk_ids if relevant else []
    
    def load_from_json(self, filepath: str):
        """
        Load ground truth data from a single JSON file.
        
        Expected format:
        {
            "questions": [
                {
                    "id": "q1",
                    "text": "What is...?",
                    "answer": "The answer is...",
                    "relevant_chunks": ["chunk_0001", "chunk_0002"],
                    "difficulty": "easy",  # optional
                    "category": "overview"  # optional
                }
            ]
        }
        
        Args:
            filepath: Path to the JSON file
        """
        logger.info(f"Loading ground truth from: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        questions_data = data.get("questions", [])
        
        for q_data in questions_data:
            q_id = q_data.get("id")
            if not q_id:
                continue
            
            # Add question
            question = Question(
                id=q_id,
                text=q_data.get("text", ""),
                category=q_data.get("category"),
                difficulty=q_data.get("difficulty"),
                expected_answer_type=q_data.get("expected_answer_type")
            )
            self.questions[q_id] = question
            
            # Add human answer
            if "answer" in q_data:
                human_answer = HumanAnswer(
                    question_id=q_id,
                    answer=q_data["answer"],
                    confidence=q_data.get("confidence"),
                    annotator=q_data.get("annotator"),
                    notes=q_data.get("notes")
                )
                self.human_answers[q_id] = human_answer
            
            # Add relevant chunks
            if "relevant_chunks" in q_data:
                relevant = RelevantChunks(
                    question_id=q_id,
                    chunk_ids=q_data["relevant_chunks"],
                    primary_chunk_id=q_data.get("primary_chunk"),
                    relevance_scores=q_data.get("relevance_scores")
                )
                self.relevant_chunks[q_id] = relevant
        
        logger.info(f"Loaded {len(self.questions)} questions from {filepath}")
    
    def validate(self) -> Dict:
        """
        Validate ground truth data completeness
        
        Returns:
            Validation report
        """
        issues = []
        
        # Check all questions have answers
        for q_id in self.questions:
            if q_id not in self.human_answers:
                issues.append(f"Question {q_id} missing human answer")
            if q_id not in self.relevant_chunks:
                issues.append(f"Question {q_id} missing relevant chunks mapping")
        
        return {
            "valid": len(issues) == 0,
            "questions_count": len(self.questions),
            "answers_count": len(self.human_answers),
            "chunk_mappings_count": len(self.relevant_chunks),
            "issues": issues
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the ground truth data"""
        return {
            "total_questions": len(self.questions),
            "total_answers": len(self.human_answers),
            "total_chunk_mappings": len(self.relevant_chunks),
            "categories": list(set(q.category for q in self.questions.values() if q.category)),
            "difficulties": list(set(q.difficulty for q in self.questions.values() if q.difficulty)),
            "avg_relevant_chunks": (
                sum(len(rc.chunk_ids) for rc in self.relevant_chunks.values()) / 
                len(self.relevant_chunks) if self.relevant_chunks else 0
            ),
        }


def create_sample_ground_truth(ground_truth_dir: str = "./data/ground_truth"):
    """
    Create a sample ground truth structure with templates
    
    This creates template files that users can fill in with their own data
    """
    gt = GroundTruth(ground_truth_dir)
    
    # Add sample questions (users should replace with their own)
    gt.add_question(
        "q1",
        "What is the main topic discussed in the document?",
        category="general",
        difficulty="easy",
        expected_answer_type="summary"
    )
    gt.add_question(
        "q2", 
        "What specific data or figures are mentioned?",
        category="factual",
        difficulty="medium",
        expected_answer_type="factual"
    )
    gt.add_question(
        "q3",
        "What conclusions or recommendations are made?",
        category="analysis",
        difficulty="hard",
        expected_answer_type="analytical"
    )
    
    # Add sample human answers (users should replace)
    gt.add_human_answer(
        "q1",
        "[Replace with accurate human-provided answer]",
        confidence=1.0,
        annotator="human_expert"
    )
    gt.add_human_answer(
        "q2",
        "[Replace with accurate human-provided answer with specific figures]",
        confidence=1.0,
        annotator="human_expert"
    )
    gt.add_human_answer(
        "q3",
        "[Replace with accurate human-provided answer]",
        confidence=1.0,
        annotator="human_expert"
    )
    
    # Add sample relevant chunks (users should replace with actual chunk IDs)
    gt.add_relevant_chunks(
        "q1",
        ["chunk_0", "chunk_1", "chunk_2"],
        primary_chunk_id="chunk_0",
        relevance_scores={"chunk_0": 1.0, "chunk_1": 0.8, "chunk_2": 0.6}
    )
    gt.add_relevant_chunks(
        "q2",
        ["chunk_5", "chunk_6"],
        primary_chunk_id="chunk_5"
    )
    gt.add_relevant_chunks(
        "q3",
        ["chunk_10", "chunk_11", "chunk_12", "chunk_13"],
        primary_chunk_id="chunk_10"
    )
    
    gt.save_all()
    
    logger.info(f"Created sample ground truth in {ground_truth_dir}")
    logger.info("Please update the files with your actual questions, answers, and relevant chunk IDs")
    
    return gt
