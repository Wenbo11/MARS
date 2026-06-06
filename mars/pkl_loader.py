"""
Loader for the unified pickle trace format.

Loads data from {model}/dataset.pkl files into the same RawQuestion + ProbedQuestion
objects used by the simulation pipeline.
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .probed_traces import ProbedQuestion, ProbedTrace
from .raw_traces import RawDataset, RawQuestion, RawTrace


def load_pkl_dataset(
    pkl_path: Path,
    name: str = "",
    model: str = "",
    question_ids: Optional[List[int]] = None,
) -> Tuple[RawDataset, Dict[int, ProbedQuestion]]:
    """Load a full dataset from a single pickle file.

    Returns:
        (RawDataset, dict mapping qid -> ProbedQuestion)
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    by_question: Dict[int, list] = defaultdict(list)
    for (qid, _trace_idx), entry in data.items():
        if question_ids is not None and qid not in question_ids:
            continue
        by_question[qid].append(entry)

    questions: List[RawQuestion] = []
    probed_questions: Dict[int, ProbedQuestion] = {}

    for qid in sorted(by_question):
        entries = by_question[qid]
        raw_q, probed_q = _build_question(qid, entries)
        questions.append(raw_q)
        probed_questions[qid] = probed_q

    dataset = RawDataset(name=name, model=model, questions=questions)
    return dataset, probed_questions


def split_pkl_to_per_question(
    pkl_path: Path,
    cache_dir: Path,
) -> Tuple[List[int], Dict[int, str], Dict[int, List[int]]]:
    """Split a dataset pkl into per-question pkl files for worker loading.

    Extracts lightweight metadata (ground truth, per-trace token lengths)
    without building full RawTrace objects, keeping memory usage low.

    Returns:
        (question_ids, ground_truth_map, trace_lengths_map)
        trace_lengths_map: {qid -> sorted list of token lengths per trace}
    """
    pkl_path = Path(pkl_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "_metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return meta["question_ids"], meta["ground_truth"], meta["trace_lengths"]

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    by_question: Dict[int, list] = defaultdict(list)
    for (qid, _trace_idx), entry in data.items():
        by_question[qid].append(entry)

    ground_truth_map: Dict[int, str] = {}
    trace_lengths: Dict[int, List[int]] = {}

    for qid in sorted(by_question):
        entries = by_question[qid]
        entries_sorted = sorted(entries, key=lambda e: e["trace_idx"])

        q_path = cache_dir / f"q{qid:02d}.pkl"
        if not q_path.exists():
            with open(q_path, "wb") as f:
                pickle.dump(entries_sorted, f, protocol=pickle.HIGHEST_PROTOCOL)

        ground_truth_map[qid] = entries[0]["ground_truth"]
        trace_lengths[qid] = [len(e["confs"]) for e in entries_sorted]  # sorted by trace_idx

    del data

    question_ids = sorted(by_question.keys())
    meta = {
        "question_ids": question_ids,
        "ground_truth": ground_truth_map,
        "trace_lengths": trace_lengths,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    return question_ids, ground_truth_map, trace_lengths


def load_pkl_question(q_pkl_path: Path) -> Tuple[RawQuestion, ProbedQuestion]:
    """Load a single question from a per-question pkl file."""
    with open(q_pkl_path, "rb") as f:
        entries = pickle.load(f)
    qid = entries[0]["qid"]
    return _build_question(qid, entries)


def _build_question(
    qid: int, entries: list
) -> Tuple[RawQuestion, ProbedQuestion]:
    """Convert pickle entries for one question into RawQuestion + ProbedQuestion.

    Probe positions for the question grid are the 2048-multiple positions that
    appear in the data. Per-trace termination positions are kept in each trace's
    probe_results but excluded from the shared grid, since they're not available
    across all alive traces.
    """
    ground_truth = entries[0]["ground_truth"]

    traces: List[RawTrace] = []
    probed_traces: List[ProbedTrace] = []
    grid_positions: set = set()

    entries_sorted = sorted(entries, key=lambda e: e["trace_idx"])

    for entry in entries_sorted:
        trace_idx = entry["trace_idx"]
        confs = np.asarray(entry["confs"], dtype=np.float32)
        answer = entry.get("extracted_answer")
        probes = entry["probes"]

        is_correct = _determine_correctness(answer, ground_truth, probes)

        traces.append(RawTrace(
            confs=confs,
            answer=answer,
            is_correct=is_correct,
        ))

        probe_results: Dict[int, str] = {}
        for pos, probe_data in probes.items():
            pos_int = int(pos)
            ans = probe_data.get("answer")
            if ans is not None:
                probe_results[pos_int] = ans
            if pos_int % 2048 == 0:
                grid_positions.add(pos_int)

        probed_traces.append(ProbedTrace(
            trace_id=trace_idx,
            final_answer=answer,
            num_tokens=len(confs),
            probe_results=probe_results,
        ))

    raw_q = RawQuestion(
        question_id=qid,
        question_text="",
        ground_truth=ground_truth,
        traces=traces,
    )

    probed_q = ProbedQuestion(
        question_id=qid,
        traces=probed_traces,
        probe_positions=sorted(grid_positions),
    )

    return raw_q, probed_q


def _determine_correctness(
    answer: Optional[str], ground_truth: str, probes: dict
) -> bool:
    """Determine is_correct from the last probe or extracted_answer."""
    if not probes:
        return False
    last_pos = max(probes.keys())
    last_probe = probes[last_pos]
    if "is_correct" in last_probe:
        return bool(last_probe["is_correct"])
    from .answer_equiv import answers_equivalent
    if answer is not None:
        try:
            return answers_equivalent(answer, ground_truth)
        except Exception:
            return answer == ground_truth
    return False
