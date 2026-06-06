"""
Loader for compact NPZ trace format.

Loads preprocessed trace data from NPZ files into RawTrace/RawQuestion/RawDataset.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import re

from .raw_traces import RawTrace, RawQuestion, RawDataset


def _extract_qid_from_filename(filename: str) -> int:
    """Extract question ID from NPZ filename (e.g., 'q00.npz' -> 0)."""
    match = re.search(r'q(\d+)\.npz', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract question ID from {filename}")


def load_question_npz(npz_path: Path) -> RawQuestion:
    """Load a single question from an NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    question_id = int(data['question_id'])
    ground_truth = str(data['ground_truth'])
    question_text = str(data['question_text']) if 'question_text' in data else ''

    confs_padded = data['confs']      # (n_traces, max_tokens)
    lengths = data['lengths']          # (n_traces,)
    is_correct = data['is_correct']    # (n_traces,)
    answers = data['answers']          # (n_traces,)

    traces = []
    for i in range(len(lengths)):
        length = int(lengths[i])
        trace = RawTrace(
            confs=confs_padded[i, :length].copy(),
            answer=str(answers[i]) if answers[i] is not None else None,
            is_correct=bool(is_correct[i])
        )
        traces.append(trace)

    return RawQuestion(
        question_id=question_id,
        question_text=question_text,
        ground_truth=ground_truth,
        traces=traces
    )


def load_npz_dataset(
    data_dir: Path,
    name: str = "AIME2025",
    model: str = "DeepSeek-8B",
    question_ids: Optional[List[int]] = None
) -> RawDataset:
    """Load all questions from a directory of NPZ files."""
    data_dir = Path(data_dir)

    npz_files = sorted([
        f for f in data_dir.glob("q*.npz")
        if f.name != "metadata.npz"
    ])

    if not npz_files:
        raise FileNotFoundError(f"No question NPZ files found in {data_dir}")

    if question_ids is not None:
        target_ids = set(question_ids)
        npz_files = [
            f for f in npz_files
            if _extract_qid_from_filename(f.name) in target_ids
        ]

    questions = []
    for npz_path in npz_files:
        try:
            question = load_question_npz(npz_path)
            questions.append(question)
        except Exception as e:
            print(f"Warning: Failed to load {npz_path}: {e}")
            continue

    questions.sort(key=lambda q: q.question_id)

    return RawDataset(
        name=name,
        model=model,
        questions=questions
    )
