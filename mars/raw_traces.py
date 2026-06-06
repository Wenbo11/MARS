"""
Data models for raw trace data with per-token confidence scores.

Defines dataclasses for the compact NPZ format that stores full per-token
confidence arrays.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class RawTrace:
    """A single reasoning trace with per-token confidence scores.

    Attributes:
        confs: Per-token confidence scores (unpadded, actual length)
        answer: The extracted answer string (or None if not found)
        is_correct: Whether the answer matches ground truth
    """
    confs: np.ndarray  # Shape: (num_tokens,)
    answer: Optional[str]
    is_correct: bool

    @property
    def num_tokens(self) -> int:
        return len(self.confs)

    @property
    def mean_confidence(self) -> float:
        if len(self.confs) == 0:
            return 0.0
        return float(np.mean(self.confs))


@dataclass
class RawQuestion:
    """A question with all its raw traces.

    Attributes:
        question_id: Unique identifier
        question_text: The full question text
        ground_truth: The correct answer
        traces: List of all raw traces for this question
    """
    question_id: int
    question_text: str
    ground_truth: str
    traces: List[RawTrace] = field(default_factory=list)

    @property
    def num_traces(self) -> int:
        return len(self.traces)

    @property
    def pass_at_1(self) -> float:
        if not self.traces:
            return 0.0
        return sum(1 for t in self.traces if t.is_correct) / len(self.traces)


@dataclass
class RawDataset:
    """Dataset containing all questions with raw traces.

    Attributes:
        name: Name of the dataset (e.g., "AIME2025")
        model: Name of the model (e.g., "DeepSeek-8B")
        questions: List of questions with traces
    """
    name: str
    model: str
    questions: List[RawQuestion] = field(default_factory=list)

    @property
    def num_questions(self) -> int:
        return len(self.questions)

    @property
    def total_traces(self) -> int:
        return sum(q.num_traces for q in self.questions)

    def get_question(self, qid: int) -> Optional[RawQuestion]:
        for q in self.questions:
            if q.question_id == qid:
                return q
        return None
