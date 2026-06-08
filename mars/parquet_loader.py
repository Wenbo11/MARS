"""
Loader for the parquet trace format (the format published on Hugging Face).

A parquet file holds one row per trace, with columns
``qid, trace_idx, ground_truth, extracted_answer, confs (list<float16>),
probes (list<struct{position, answer, is_correct, raw_text, avg_conf}>)``.

To avoid duplicating reconstruction logic, this module reads the parquet, rebuilds
the same per-trace entry dicts the pickle format uses, and writes the **identical**
per-question ``.pkl`` cache that :func:`mars.pkl_loader.split_pkl_to_per_question`
produces. Everything downstream (the worker, ``load_pkl_question``,
``_build_question``) is then shared and unchanged.
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _rows_to_entries(parquet_path: Path) -> Dict[int, list]:
    """Read a parquet file into {qid: [entry_dict, ...]} (pickle-format entries)."""
    import pyarrow.parquet as pq

    cols = pq.read_table(parquet_path).to_pydict()
    n = len(cols["qid"])
    by_question: Dict[int, list] = defaultdict(list)
    for i in range(n):
        probes: Dict[int, dict] = {}
        for p in cols["probes"][i]:
            probes[int(p["position"])] = {
                "answer": p["answer"],
                "is_correct": p["is_correct"],
                "raw_text": p["raw_text"],
                "avg_conf": p["avg_conf"],
            }
        entry = {
            "qid": int(cols["qid"][i]),
            "trace_idx": int(cols["trace_idx"][i]),
            "ground_truth": cols["ground_truth"][i],
            "extracted_answer": cols["extracted_answer"][i],
            # keep float16 exactly; _build_question casts to float32 (bit-identical)
            "confs": np.asarray(cols["confs"][i], dtype=np.float16),
            "probes": probes,
        }
        by_question[entry["qid"]].append(entry)
    return by_question


def split_parquet_to_per_question(
    parquet_path: Path,
    cache_dir: Path,
) -> Tuple[List[int], Dict[int, str], Dict[int, List[int]]]:
    """Split a parquet trace file into per-question ``q{qid}.pkl`` cache files.

    Produces the same cache layout as
    :func:`mars.pkl_loader.split_pkl_to_per_question`, so workers load it via the
    shared ``load_pkl_question`` path.

    Returns:
        (question_ids, ground_truth_map, trace_lengths_map)
    """
    parquet_path = Path(parquet_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "_metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return meta["question_ids"], meta["ground_truth"], meta["trace_lengths"]

    by_question = _rows_to_entries(parquet_path)

    ground_truth_map: Dict[int, str] = {}
    trace_lengths: Dict[int, List[int]] = {}

    for qid in sorted(by_question):
        entries_sorted = sorted(by_question[qid], key=lambda e: e["trace_idx"])

        q_path = cache_dir / f"q{qid:02d}.pkl"
        if not q_path.exists():
            with open(q_path, "wb") as f:
                pickle.dump(entries_sorted, f, protocol=pickle.HIGHEST_PROTOCOL)

        ground_truth_map[qid] = entries_sorted[0]["ground_truth"]
        trace_lengths[qid] = [len(e["confs"]) for e in entries_sorted]

    question_ids = sorted(by_question.keys())
    meta = {
        "question_ids": question_ids,
        "ground_truth": ground_truth_map,
        "trace_lengths": trace_lengths,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    return question_ids, ground_truth_map, trace_lengths


def load_parquet_dataset(
    parquet_path: Path,
    name: str = "",
    model: str = "",
    question_ids: Optional[List[int]] = None,
):
    """Load a full dataset from a parquet file (mirrors ``load_pkl_dataset``)."""
    from .pkl_loader import _build_question
    from .raw_traces import RawDataset

    by_question = _rows_to_entries(Path(parquet_path))

    questions = []
    probed_questions = {}
    for qid in sorted(by_question):
        if question_ids is not None and qid not in question_ids:
            continue
        raw_q, probed_q = _build_question(qid, by_question[qid])
        questions.append(raw_q)
        probed_questions[qid] = probed_q

    dataset = RawDataset(name=name, model=model, questions=questions)
    return dataset, probed_questions
