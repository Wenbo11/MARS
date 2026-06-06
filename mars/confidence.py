"""
Stateless confidence metric computation.

Extracted from RawTrace methods and integrated_online.precompute_question_data().
All functions operate on numpy arrays — no class methods.
"""

from typing import Dict, List

import numpy as np


def group_confidences(confs: np.ndarray, length: int, window_size: int = 2048) -> np.ndarray:
    """Sliding-window mean confidences for a single trace.

    Uses cumsum for O(n) computation.

    Args:
        confs: [num_tokens] per-token confidence scores (unpadded)
        length: actual number of tokens
        window_size: sliding window width

    Returns:
        [n_windows] array of mean confidences per window
    """
    if length < window_size:
        return np.array([np.mean(confs[:length])]) if length > 0 else np.array([0.0])

    cumsum = np.cumsum(confs[:length])
    cumsum = np.insert(cumsum, 0, 0)
    window_sums = cumsum[window_size:] - cumsum[:-window_size]
    return window_sums / window_size


def group_confidences_batch(
    confs_list: List[np.ndarray],
    lengths: np.ndarray,
    window_size: int = 2048,
) -> List[np.ndarray]:
    """Compute group_confidences for multiple traces.

    Args:
        confs_list: list of n_traces unpadded confidence arrays
        lengths: [n_traces] actual token counts

    Returns:
        list of n_traces arrays, each [n_windows_i]
    """
    return [
        group_confidences(confs_list[i], int(lengths[i]), window_size)
        for i in range(len(confs_list))
    ]


def min_group_confidence(confs: np.ndarray, length: int, window_size: int = 2048) -> float:
    """Minimum group confidence (lowest window)."""
    gc = group_confidences(confs, length, window_size)
    return float(np.min(gc))


def bottom_10_confidence(confs: np.ndarray, length: int, window_size: int = 2048) -> float:
    """Mean of bottom 10% group confidences. Uses np.partition for O(n)."""
    gc = group_confidences(confs, length, window_size)
    if len(gc) == 0:
        return 0.0

    num_bottom = max(1, int(len(gc) * 0.1))
    if num_bottom >= len(gc):
        return float(np.mean(gc))

    bottom_vals = np.partition(gc, num_bottom)[:num_bottom]
    return float(np.mean(bottom_vals))


def compute_confidence_metrics(
    traces_confs: List[np.ndarray],
    lengths: np.ndarray,
    window_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """Batch-compute all confidence metrics for a question's traces.

    Args:
        traces_confs: list of n_traces unpadded confidence arrays
        lengths: [n_traces] int array

    Returns:
        dict with keys:
            'min_group':  [n_traces] float64
            'bottom_10':  [n_traces] float64
            'mean':       [n_traces] float64
            'tail_conf':  [n_traces] float64
            'group_confs': list of per-trace group confidence arrays
    """
    n_traces = len(traces_confs)
    min_group = np.zeros(n_traces, dtype=np.float64)
    bottom_10 = np.zeros(n_traces, dtype=np.float64)
    mean_conf = np.zeros(n_traces, dtype=np.float64)
    tail_conf = np.zeros(n_traces, dtype=np.float64)
    all_group_confs = []

    for i in range(n_traces):
        gc = group_confidences(traces_confs[i], int(lengths[i]), window_size)
        all_group_confs.append(gc)
        min_group[i] = float(np.min(gc))

        num_bottom = max(1, int(len(gc) * 0.1))
        if num_bottom >= len(gc):
            bottom_10[i] = float(np.mean(gc))
        else:
            bottom_10[i] = float(np.mean(np.partition(gc, num_bottom)[:num_bottom]))

        length = int(lengths[i])
        mean_conf[i] = float(np.mean(traces_confs[i][:length])) if length > 0 else 0.0

        # tail_conf: mean of last window_size tokens
        if length > 0:
            tail_start = max(0, length - window_size)
            tail_conf[i] = float(np.mean(traces_confs[i][tail_start:length]))
        else:
            tail_conf[i] = 0.0

    return {
        'min_group': min_group,
        'bottom_10': bottom_10,
        'mean': mean_conf,
        'tail_conf': tail_conf,
        'group_confs': all_group_confs,
    }


def confidence_at_positions(
    confs: np.ndarray,
    length: int,
    positions: np.ndarray,
    window_size: int = 2048,
) -> np.ndarray:
    """Compute group confidence at specific probe positions for one trace.

    For each position p, returns the mean confidence of the window ending at
    token p. If p < window_size, returns the mean of confs[0:p].

    Args:
        confs: [max_tokens] per-token confidence scores
        length: actual number of tokens
        positions: [n_positions] int — probe positions to evaluate
        window_size: sliding window width

    Returns:
        [n_positions] float32 — confidence at each position
    """
    n_pos = len(positions)
    result = np.zeros(n_pos, dtype=np.float32)

    if length == 0:
        return result

    # Build cumsum once
    cumsum = np.cumsum(confs[:length].astype(np.float64))
    cumsum = np.insert(cumsum, 0, 0.0)

    for j, pos in enumerate(positions):
        if pos >= length:
            # Past end of trace — use full trace mean
            result[j] = cumsum[length] / length
        elif pos < window_size:
            # Partial window
            end = min(pos + 1, length)
            result[j] = cumsum[end] / end if end > 0 else 0.0
        else:
            # Full window ending at pos
            start = pos - window_size + 1
            end = pos + 1
            if end > length:
                end = length
            result[j] = (cumsum[end] - cumsum[start]) / (end - start)

    return result


def confidence_at_positions_batch(
    traces_confs: List[np.ndarray],
    lengths: np.ndarray,
    positions: np.ndarray,
    window_size: int = 2048,
) -> np.ndarray:
    """Vectorized confidence_at_positions for all traces in a question.

    Args:
        traces_confs: list of n_traces unpadded confidence arrays
        lengths: [n_traces]
        positions: [n_positions]
        window_size: sliding window width

    Returns:
        [n_traces, n_positions] float32
    """
    n_traces = len(traces_confs)
    n_pos = len(positions)
    result = np.zeros((n_traces, n_pos), dtype=np.float32)

    for i in range(n_traces):
        result[i] = confidence_at_positions(
            traces_confs[i], int(lengths[i]), positions, window_size
        )

    return result


def find_truncation_position(
    group_confs: np.ndarray,
    num_tokens: int,
    threshold: float,
    window_size: int,
) -> int:
    """Find position where group confidence first drops below threshold.

    Args:
        group_confs: pre-computed group confidence array for a trace
        num_tokens: total number of tokens in the trace
        threshold: confidence threshold
        window_size: sliding window size

    Returns:
        Token position where truncated, or num_tokens if not truncated.
    """
    if len(group_confs) == 0:
        return num_tokens

    below_threshold = group_confs < threshold
    if not np.any(below_threshold):
        return num_tokens

    first_idx = int(np.argmax(below_threshold))
    return first_idx + window_size
