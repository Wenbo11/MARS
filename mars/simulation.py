"""
Unified simulation engine.

Every method (offline, dco, sc-mars / dco-mars and their -cal / -oracle
variants, sc-pp) is a call to simulate_with_stopping() with different inputs
and a different StoppingStrategy.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .answer_equiv import answers_equivalent, group_equivalent_answers
from .confidence import (
    compute_confidence_metrics,
    confidence_at_positions_batch,
    find_truncation_position,
)
from .probed_traces import ProbedQuestion
from .q_model import compute_flips, compute_streaks
from .raw_traces import RawQuestion
from .voting import (
    PositionState,
    StoppingStrategy,
    compute_votes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrecomputedQuestion:
    """All precomputed data for one question, ready for simulation."""
    question_id: int
    ground_truth: str
    n_traces: int

    # From answer_equiv
    canonical_map: Dict[int, str]   # {group_id: str}
    answer_ids: np.ndarray          # [n_traces] int16 (final answer group)
    n_answers: int

    # From confidence
    num_tokens: np.ndarray          # [n_traces] int64
    min_group: np.ndarray           # [n_traces] float64
    bottom_10: np.ndarray           # [n_traces] float64
    mean_conf: np.ndarray           # [n_traces] float64
    tail_conf: np.ndarray           # [n_traces] float64
    group_confs: List[np.ndarray]   # per-trace group confidence arrays

    # From probed_traces — answers at each probe position
    ans_ids_at_pos: np.ndarray      # [n_traces, n_positions] int16
    probe_positions: np.ndarray     # [n_positions] int

    # Per-position confidence
    conf_at_pos: np.ndarray         # [n_traces, n_positions] float32

    # Per-trace q-model features (always precomputed)
    flips: np.ndarray               # [n_traces, n_positions] int16
    streaks: np.ndarray             # [n_traces, n_positions] int16


def precompute_question(
    raw_q: RawQuestion,
    probed_q: ProbedQuestion,
    window_size: int = 2048,
) -> PrecomputedQuestion:
    """Precompute all derived data for a question.

    Called once per question per worker process. All downstream simulation
    uses only numpy arrays from this object.

    Important: collects ALL answers (final + probed intermediate) before
    grouping, matching the original code's behavior. This ensures probed
    answers that only appear at intermediate positions are properly grouped.
    """
    traces = raw_q.traces
    n_traces = len(traces)
    probe_positions = np.array(probed_q.probe_positions, dtype=np.int64)
    n_positions = len(probe_positions)

    # 1. Collect ALL answers (final + probed) for grouping
    all_answer_strings: List[str] = []
    for t in traces:
        if t.answer:
            all_answer_strings.append(t.answer)
    probed_index = {pt.trace_id: pt for pt in probed_q.traces}
    for pt in probed_q.traces:
        for ans in pt.probe_results.values():
            if ans:
                all_answer_strings.append(ans)
        if pt.final_answer:
            all_answer_strings.append(pt.final_answer)

    # Group all answers (including probed intermediate)
    canonical_map, all_ids = group_equivalent_answers(all_answer_strings)
    n_answers = len(canonical_map)

    # Build answer_to_id lookup from grouping results
    answer_to_id: Dict[str, int] = {}
    for idx, ans in enumerate(all_answer_strings):
        if ans is not None and all_ids[idx] >= 0:
            answer_to_id[ans] = int(all_ids[idx])

    # Map final answers to IDs
    answer_ids = np.full(n_traces, -1, dtype=np.int16)
    for i, t in enumerate(traces):
        if t.answer is not None and t.answer in answer_to_id:
            answer_ids[i] = answer_to_id[t.answer]

    # 2. Confidence metrics
    traces_confs = [t.confs for t in traces]
    lengths = np.array([t.num_tokens for t in traces], dtype=np.int64)
    metrics = compute_confidence_metrics(traces_confs, lengths, window_size)

    # 3. Per-position confidence
    conf_at_pos = confidence_at_positions_batch(
        traces_confs, lengths, probe_positions, window_size
    )

    # 4. Answer IDs at each probe position
    ans_ids_at_pos = np.full((n_traces, n_positions), -1, dtype=np.int16)

    for i, trace in enumerate(traces):
        probed = probed_index.get(i)
        if probed is None:
            # Unprobed trace — leave ans_ids_at_pos[i, :] as -1 (no data).
            # Bootstrap sampling must restrict to probed_trace_ids so these
            # rows are never accessed.  Do NOT fall back to trace.answer.
            continue
        for j, pos in enumerate(probe_positions):
            if pos >= trace.num_tokens:
                # Trace finished before this position — use final answer.
                # This takes priority over probed results to ensure consistency
                # with answer_ids (which always uses trace.answer). Without this,
                # max-token traces (trace.answer=None but valid probed answer)
                # would participate in intermediate votes but not final votes.
                ans = trace.answer
            elif int(pos) in probed.probe_results:
                # Probed trace with data at this position — use it.
                ans = probed.probe_results[int(pos)]
            else:
                # Probed trace, still running, but no probe data here.
                raise ValueError(
                    f"Trace {i} (num_tokens={trace.num_tokens}) is missing "
                    f"probe result at position {pos} (trace still running). "
                    f"Available probes: {sorted(probed.probe_results.keys())}"
                )
            if ans is not None:
                if ans in answer_to_id:
                    ans_ids_at_pos[i, j] = answer_to_id[ans]
                else:
                    # Fallback: try equivalence with existing representatives
                    for gid, rep in canonical_map.items():
                        try:
                            if answers_equivalent(ans, rep):
                                answer_to_id[ans] = gid
                                ans_ids_at_pos[i, j] = gid
                                break
                        except:
                            continue

    # 5. Flips and streaks
    flips = compute_flips(ans_ids_at_pos)
    streaks_arr = compute_streaks(ans_ids_at_pos)

    return PrecomputedQuestion(
        question_id=raw_q.question_id,
        ground_truth=raw_q.ground_truth,
        n_traces=n_traces,
        canonical_map=canonical_map,
        answer_ids=answer_ids,
        n_answers=n_answers,
        num_tokens=lengths,
        min_group=metrics['min_group'],
        bottom_10=metrics['bottom_10'],
        mean_conf=metrics['mean'],
        tail_conf=metrics['tail_conf'],
        group_confs=metrics['group_confs'],
        ans_ids_at_pos=ans_ids_at_pos,
        probe_positions=probe_positions,
        conf_at_pos=conf_at_pos,
        flips=flips,
        streaks=streaks_arr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed Cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(data_path: Path, qid: int, window_size: int) -> Path:
    """Return cache file path for a precomputed question."""
    return data_path / "precomputed" / f"w{window_size}" / f"q{qid:02d}.npz"


def save_precomputed(pc: PrecomputedQuestion, path: Path) -> None:
    """Save PrecomputedQuestion to an NPZ file.

    group_confs (list of variable-length arrays) is stored as a concatenated
    array plus an offsets array for reconstruction.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate variable-length group_confs
    gc_concat = np.concatenate(pc.group_confs) if pc.group_confs else np.array([], dtype=np.float64)
    gc_lengths = np.array([len(gc) for gc in pc.group_confs], dtype=np.int64)

    # Canonical map as JSON bytes
    canonical_json = json.dumps(
        {str(k): v for k, v in pc.canonical_map.items()}
    ).encode()

    np.savez_compressed(
        path,
        question_id=np.array(pc.question_id),
        ground_truth=np.array(pc.ground_truth, dtype=object),
        n_traces=np.array(pc.n_traces),
        n_answers=np.array(pc.n_answers),
        answer_ids=pc.answer_ids,
        num_tokens=pc.num_tokens,
        min_group=pc.min_group,
        bottom_10=pc.bottom_10,
        mean_conf=pc.mean_conf,
        tail_conf=pc.tail_conf,
        gc_concat=gc_concat,
        gc_lengths=gc_lengths,
        ans_ids_at_pos=pc.ans_ids_at_pos,
        probe_positions=pc.probe_positions,
        conf_at_pos=pc.conf_at_pos,
        flips=pc.flips,
        streaks=pc.streaks,
        canonical_json=np.void(canonical_json),
    )


def load_precomputed(path: Path) -> PrecomputedQuestion:
    """Load PrecomputedQuestion from an NPZ cache file."""
    import json

    data = np.load(path, allow_pickle=True)

    canonical_json = bytes(data['canonical_json'])
    canonical_map = {int(k): v for k, v in json.loads(canonical_json).items()}

    # Reconstruct group_confs from concatenated array + lengths
    gc_concat = data['gc_concat']
    gc_lengths = data['gc_lengths']
    group_confs = []
    offset = 0
    for length in gc_lengths:
        group_confs.append(gc_concat[offset:offset + length])
        offset += length

    # Load mean_conf and tail_conf (may be absent in old caches)
    mean_conf = data['mean_conf'] if 'mean_conf' in data else None
    tail_conf = data['tail_conf'] if 'tail_conf' in data else None

    return PrecomputedQuestion(
        question_id=int(data['question_id']),
        ground_truth=str(data['ground_truth']),
        n_traces=int(data['n_traces']),
        canonical_map=canonical_map,
        answer_ids=data['answer_ids'],
        n_answers=int(data['n_answers']),
        num_tokens=data['num_tokens'],
        min_group=data['min_group'],
        bottom_10=data['bottom_10'],
        mean_conf=mean_conf,
        tail_conf=tail_conf,
        group_confs=group_confs,
        ans_ids_at_pos=data['ans_ids_at_pos'],
        probe_positions=data['probe_positions'],
        conf_at_pos=data['conf_at_pos'],
        flips=data['flips'],
        streaks=data['streaks'],
    )


def precompute_question_cached(
    raw_q: RawQuestion,
    probed_q: ProbedQuestion,
    data_path: Path,
    window_size: int = 2048,
) -> PrecomputedQuestion:
    """Load from cache if available, otherwise compute and save.

    Handles backward compatibility: if loaded cache lacks mean_conf/tail_conf,
    recomputes from scratch and re-saves.
    """
    cache = _cache_path(data_path, raw_q.question_id, window_size)

    if cache.exists():
        pc = load_precomputed(cache)
        # Check for missing fields from older cache versions
        if pc.mean_conf is None or pc.tail_conf is None:
            pc = precompute_question(raw_q, probed_q, window_size)
            save_precomputed(pc, cache)
        return pc

    pc = precompute_question(raw_q, probed_q, window_size)
    save_precomputed(pc, cache)
    return pc


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_bootstrap(
    precomputed: PrecomputedQuestion,
    n_iterations: int,
    budget: int,
    seed: int,
    probed_trace_ids: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Generate bootstrap samples as numpy arrays.

    Args:
        precomputed: precomputed question data
        n_iterations: number of bootstrap iterations
        budget: number of traces per iteration
        seed: random seed
        probed_trace_ids: if provided, sample only from these trace IDs
            (to avoid information leakage from unprobed traces)

    Returns dict with:
        'indices':          [n_iter, budget] int64    (trace indices)
        'sampled_ans_ids':  [n_iter, budget, n_pos] int16
        'sampled_weights':  [n_iter, budget] float64  (bottom_10 confidence)
        'sampled_tokens':   [n_iter, budget] int64
        'sampled_min_confs':[n_iter, budget] float64
        'sampled_conf_at_pos': [n_iter, budget, n_pos] float32
        'sampled_flips':    [n_iter, budget, n_pos] int16
        'sampled_streaks':  [n_iter, budget, n_pos] int16
        'sampled_final_ans':[n_iter, budget] int16
    """
    n_pool = len(probed_trace_ids) if probed_trace_ids is not None else precomputed.n_traces

    # Generate index matrix
    all_indices = np.empty((n_iterations, budget), dtype=np.int64)
    for i in range(n_iterations):
        rng = np.random.default_rng(seed + i)
        sampled = rng.choice(n_pool, size=budget, replace=True)
        if probed_trace_ids is not None:
            all_indices[i] = probed_trace_ids[sampled]
        else:
            all_indices[i] = sampled

    # Fancy-index into precomputed arrays
    result = {
        'indices': all_indices,
        'sampled_ans_ids': precomputed.ans_ids_at_pos[all_indices],         # [n_iter, budget, n_pos]
        'sampled_weights': precomputed.bottom_10[all_indices],              # [n_iter, budget]
        'sampled_tokens': precomputed.num_tokens[all_indices],              # [n_iter, budget]
        'sampled_min_confs': precomputed.min_group[all_indices],            # [n_iter, budget]
        'sampled_conf_at_pos': precomputed.conf_at_pos[all_indices],        # [n_iter, budget, n_pos]
        'sampled_flips': precomputed.flips[all_indices],                    # [n_iter, budget, n_pos]
        'sampled_streaks': precomputed.streaks[all_indices],                # [n_iter, budget, n_pos]
        'sampled_final_ans': precomputed.answer_ids[all_indices],           # [n_iter, budget]
        'sampled_bottom_10': precomputed.bottom_10[all_indices],            # [n_iter, budget]
    }

    # Optional fields (may be None for old caches)
    if precomputed.mean_conf is not None:
        result['sampled_mean_confs'] = precomputed.mean_conf[all_indices]   # [n_iter, budget]
    if precomputed.tail_conf is not None:
        result['sampled_tail_confs'] = precomputed.tail_conf[all_indices]   # [n_iter, budget]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Deepconf Multi-Voting
# ─────────────────────────────────────────────────────────────────────────────

def compute_deepconf_votes(
    final_ans: np.ndarray,
    eligibility: np.ndarray,
    mean_conf: np.ndarray,
    tail_conf: np.ndarray,
    bottom_10: np.ndarray,
    min_group: np.ndarray,
    n_answers: int,
) -> Dict[str, np.ndarray]:
    """Compute all 7 deepconf voting methods.

    Args:
        final_ans:   [n_iter, budget] int16 — final answer IDs
        eligibility: [n_iter, budget] bool — base eligibility mask
        mean_conf:   [n_iter, budget] float64 — mean token confidence per trace
        tail_conf:   [n_iter, budget] float64 — mean of last 2048 token confs
        bottom_10:   [n_iter, budget] float64 — mean of bottom 10% windows
        min_group:   [n_iter, budget] float64 — min window confidence
        n_answers:   int — number of unique answer groups

    Returns:
        Dict of {method_name: [n_iter] int answer_ids}.
    """
    results = {}

    # 1. majority: uniform weights
    votes = compute_votes(final_ans, None, eligibility, n_answers)
    results['majority'] = np.argmax(votes, axis=1)

    # 2. mean_conf_weighted
    votes = compute_votes(final_ans, mean_conf, eligibility, n_answers)
    results['mean_conf_weighted'] = np.argmax(votes, axis=1)

    # 3. tail_conf_weighted
    votes = compute_votes(final_ans, tail_conf, eligibility, n_answers)
    results['tail_conf_weighted'] = np.argmax(votes, axis=1)

    # 4. bottom_window_weighted
    votes = compute_votes(final_ans, bottom_10, eligibility, n_answers)
    results['bottom_window_weighted'] = np.argmax(votes, axis=1)

    # 5. min_window_weighted
    votes = compute_votes(final_ans, min_group, eligibility, n_answers)
    results['min_window_weighted'] = np.argmax(votes, axis=1)

    # 6. top10_tail_filtered: keep top 10% by tail_conf, weighted by tail_conf
    tail_for_pct = np.where(eligibility, tail_conf, -np.inf)
    pct90_tail = np.nanpercentile(
        np.where(eligibility, tail_conf, np.nan), 90, axis=1,
    )  # [n_iter] — may contain nan if no eligible traces
    pct90_tail = np.nan_to_num(pct90_tail, nan=-np.inf)
    top10_tail_elig = eligibility & (tail_for_pct >= pct90_tail[:, np.newaxis])
    votes = compute_votes(final_ans, tail_conf, top10_tail_elig, n_answers)
    results['top10_tail_filtered'] = np.argmax(votes, axis=1)

    # 7. top10_bottom_window_filtered: keep top 10% by bottom_10, weighted by bottom_10
    b10_for_pct = np.where(eligibility, bottom_10, -np.inf)
    pct90_b10 = np.nanpercentile(
        np.where(eligibility, bottom_10, np.nan), 90, axis=1,
    )
    pct90_b10 = np.nan_to_num(pct90_b10, nan=-np.inf)
    top10_b10_elig = eligibility & (b10_for_pct >= pct90_b10[:, np.newaxis])
    votes = compute_votes(final_ans, bottom_10, top10_b10_elig, n_answers)
    results['top10_bottom_window_filtered'] = np.argmax(votes, axis=1)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Result for one question across all iterations."""
    answers: np.ndarray          # [n_iter] int — winning answer ID per iteration
    is_correct: np.ndarray       # [n_iter] bool
    total_tokens: np.ndarray     # [n_iter] int64
    baseline_tokens: np.ndarray  # [n_iter] int64 — full-budget cost for the same sample
    stop_positions: np.ndarray   # [n_iter] int — probe position where stopped
    stopped_by: np.ndarray       # [n_iter] object — 'margin', 'consensus', 'budget'
    canonical_map: Dict[int, str]


# ─────────────────────────────────────────────────────────────────────────────
# Unified Simulation Loop
# ─────────────────────────────────────────────────────────────────────────────

def simulate_with_stopping(
    sampled_ans_ids: np.ndarray,
    sampled_tokens: np.ndarray,
    probe_positions: np.ndarray,
    canonical_map: Dict[int, str],
    n_answers: int,
    stopping_strategies: List[Tuple[str, StoppingStrategy]],
    weights: Optional[np.ndarray] = None,
    filter_mask: Optional[np.ndarray] = None,
    truncation_positions: Optional[np.ndarray] = None,
    sampled_final_ans: Optional[np.ndarray] = None,
    min_voters: int = 0,
) -> SimulationResult:
    """Unified simulation loop: vote at each position, stop when strategy fires.

    At each probe position, votes are computed as a SNAPSHOT (not accumulated).
    Each trace contributes its answer at the current position, weighted by its
    confidence. Strategies are checked in order — first match wins for the label.

    Args:
        sampled_ans_ids:       [n_iter, budget, n_pos] int16 — intermediate answers
        sampled_tokens:        [n_iter, budget] int64
        probe_positions:       [n_pos] int
        canonical_map:         {int -> str}
        n_answers:             number of unique answer groups
        stopping_strategies:   list of (label, strategy) — checked in order
        weights:               [n_iter, budget] float | None — per-trace vote weight
        filter_mask:           [n_iter, budget] or [n_iter, budget, n_pos] bool | None
                               2D = same filter at every position; 3D = position-specific
        truncation_positions:  [n_iter, budget] int64 | None — truncation positions
        sampled_final_ans:     [n_iter, budget] int16 | None — final answer IDs.
                               Used for budget-stopped iterations instead of the
                               intermediate answer at the last probe position, since
                               traces with num_tokens > last_probe_position may have
                               a different intermediate answer than their final answer.

    Returns:
        SimulationResult
    """
    n_iter, budget, n_pos = sampled_ans_ids.shape

    # Per-iteration full-budget cost (the counterfactual: no early stopping)
    if truncation_positions is not None:
        baseline_per_trace = np.minimum(sampled_tokens, truncation_positions)
    else:
        baseline_per_trace = sampled_tokens
    result_baseline = baseline_per_trace.sum(axis=1)

    # Result arrays
    result_answers = np.full(n_iter, -1, dtype=np.int32)
    result_correct = np.zeros(n_iter, dtype=bool)
    result_tokens = np.zeros(n_iter, dtype=np.int64)
    result_positions = np.zeros(n_iter, dtype=np.int64)
    result_stopped_by = np.empty(n_iter, dtype=object)
    result_stopped_by[:] = "budget"

    # Track which iterations are still running
    still_running = np.ones(n_iter, dtype=bool)

    for pos_idx in range(n_pos):
        n_active = still_running.sum()
        if n_active == 0:
            break

        pos = int(probe_positions[pos_idx])

        # Get answer IDs at this position for active iterations
        active_mask = still_running
        ans = sampled_ans_ids[active_mask, :, pos_idx]  # [n_active, budget]

        # Eligibility: has answer AND not filtered
        has_answer = ans >= 0

        if filter_mask is not None:
            if filter_mask.ndim == 2:
                elig = has_answer & filter_mask[active_mask]
            else:
                elig = has_answer & filter_mask[active_mask, :, pos_idx]
        else:
            elig = has_answer.copy()

        # Truncation: exclude traces past truncation position
        if truncation_positions is not None:
            active_trunc = truncation_positions[active_mask]
            active_num_tokens = sampled_tokens[active_mask]
            was_truncated = active_trunc < active_num_tokens
            past_truncation = was_truncated & (pos >= active_trunc)
            elig &= ~past_truncation

        # Weights for active iterations
        if weights is not None:
            if weights.ndim == 3:
                w = weights[active_mask, :, pos_idx]
            else:
                w = weights[active_mask]
        else:
            w = None

        # Compute votes at this position (snapshot, NOT accumulated)
        vote_weights = compute_votes(ans, w, elig, n_answers)  # [n_active, n_answers]

        # Compute active weight (traces not yet finished at this position)
        active_num_tokens = sampled_tokens[active_mask]  # [n_active, budget]
        trace_not_finished = active_num_tokens > pos     # [n_active, budget]
        active_and_eligible = elig & trace_not_finished

        if w is not None:
            active_weight_vals = np.where(active_and_eligible, w, 0.0)
        else:
            active_weight_vals = active_and_eligible.astype(np.float64)

        w_active = active_weight_vals.sum(axis=1)  # [n_active]

        # Active weight by answer
        w_active_by_ans = np.zeros((n_active, n_answers), dtype=np.float64)
        iter_idx_bc = np.broadcast_to(
            np.arange(n_active)[:, np.newaxis], (n_active, budget)
        )
        active_elig_with_ans = active_and_eligible & (ans >= 0)
        if active_elig_with_ans.any():
            np.add.at(
                w_active_by_ans,
                (iter_idx_bc[active_elig_with_ans], ans[active_elig_with_ans]),
                active_weight_vals[active_elig_with_ans],
            )

        n_active_traces = active_and_eligible.sum(axis=1)  # [n_active]

        # Build position state
        active_iter_indices = np.where(active_mask)[0]  # [n_active] original iteration IDs
        state = PositionState(
            vote_weights=vote_weights,
            answer_ids=ans,
            weights=w,
            eligible=elig,
            w_active=w_active,
            w_active_by_ans=w_active_by_ans,
            pos_idx=pos_idx,
            n_active_traces=n_active_traces,
            active_iter_indices=active_iter_indices,
        )

        # Check each stopping strategy in order (first match wins for label)
        can_stop = np.zeros(n_active, dtype=bool)
        stop_labels = np.empty(n_active, dtype=object)

        for label, strategy in stopping_strategies:
            can_stop_this = strategy.check(state)
            newly_stopped = can_stop_this & ~can_stop
            stop_labels[newly_stopped] = label
            can_stop |= can_stop_this

        # Gate on minimum voters
        if min_voters > 0:
            n_voters = elig.sum(axis=1)  # [n_active]
            can_stop &= (n_voters >= min_voters)

        # Record results for newly stopped iterations
        if can_stop.any():
            active_indices = np.where(active_mask)[0]
            stopped_indices = active_indices[can_stop]

            for idx in stopped_indices:
                active_pos = int(np.where(active_indices == idx)[0][0])
                leader_id = np.argmax(vote_weights[active_pos])
                if vote_weights[active_pos].sum() > 0:
                    result_answers[idx] = leader_id

                result_positions[idx] = pos
                result_stopped_by[idx] = stop_labels[active_pos]

                if truncation_positions is not None:
                    tokens_per_trace = np.minimum(
                        np.minimum(sampled_tokens[idx], truncation_positions[idx]),
                        pos
                    )
                else:
                    tokens_per_trace = np.minimum(sampled_tokens[idx], pos)
                result_tokens[idx] = tokens_per_trace.sum()

            still_running[stopped_indices] = False

    # Handle iterations that never stopped (budget) — vote on final answers
    budget_mask = still_running
    if budget_mask.any():
        final_pos = int(probe_positions[-1]) if n_pos > 0 else 0

        # Use final answers for budget-stopped iterations.  Intermediate
        # answers at the last probe position can differ from final answers
        # for traces with num_tokens > last_probe_position.
        if sampled_final_ans is not None:
            ans_final = sampled_final_ans[budget_mask]
        else:
            ans_final = sampled_ans_ids[budget_mask, :, n_pos - 1]
        has_ans_final = ans_final >= 0

        if filter_mask is not None:
            if filter_mask.ndim == 2:
                elig_final = has_ans_final & filter_mask[budget_mask]
            else:
                elig_final = has_ans_final & filter_mask[budget_mask, :, n_pos - 1]
        else:
            elig_final = has_ans_final.copy()

        if truncation_positions is not None:
            trunc_final = truncation_positions[budget_mask]
            tokens_final = sampled_tokens[budget_mask]
            was_trunc = trunc_final < tokens_final
            past_trunc = was_trunc & (final_pos >= trunc_final)
            elig_final &= ~past_trunc

        if weights is not None:
            if weights.ndim == 3:
                w_final = weights[budget_mask, :, n_pos - 1]
            else:
                w_final = weights[budget_mask]
        else:
            w_final = None
        votes_final = compute_votes(ans_final, w_final, elig_final, n_answers)

        budget_indices = np.where(budget_mask)[0]
        for i, idx in enumerate(budget_indices):
            leader_id = np.argmax(votes_final[i])
            if votes_final[i].sum() > 0:
                result_answers[idx] = leader_id

            result_positions[idx] = final_pos

            if truncation_positions is not None:
                tokens_per_trace = np.minimum(sampled_tokens[idx], truncation_positions[idx])
            else:
                tokens_per_trace = sampled_tokens[idx]
            result_tokens[idx] = tokens_per_trace.sum()

            result_stopped_by[idx] = "budget"

    return SimulationResult(
        answers=result_answers,
        is_correct=result_correct,  # filled by caller via check_correctness()
        total_tokens=result_tokens,
        baseline_tokens=result_baseline,
        stop_positions=result_positions,
        stopped_by=result_stopped_by,
        canonical_map=canonical_map,
    )


def simulate_oracle(
    sampled_ans_ids: np.ndarray,
    sampled_tokens: np.ndarray,
    probe_positions: np.ndarray,
    n_answers: int,
    sampled_final_ans: np.ndarray,
    weights: Optional[np.ndarray] = None,
    filter_mask: Optional[np.ndarray] = None,
    truncation_positions: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute oracle stopping positions (optimistic and absorbing).

    The oracle uses perfect foresight to find the earliest stopping position
    that preserves the final-answer vote.

    Target = vote(final_answers) using the same weights/eligibility.

    Optimistic oracle: first position where voted answer matches target.
    Absorbing oracle: earliest position from which voted answer matches
                      target at ALL subsequent positions.

    Returns dict with:
        'target':              [n_iter] int32 - target answer ID
        'leaders':             [n_iter, n_pos] int32 - voted answer at each position
        'optimistic_pos_idx':  [n_iter] int64 - position index for optimistic oracle
        'absorbing_pos_idx':   [n_iter] int64 - position index for absorbing oracle
        'optimistic_tokens':   [n_iter] int64 - total tokens for optimistic oracle
        'absorbing_tokens':    [n_iter] int64 - total tokens for absorbing oracle
        'baseline_tokens':     [n_iter] int64 - total tokens with no stopping
    """
    n_iter, budget, n_pos = sampled_ans_ids.shape

    # Step 1: Compute target = vote(final_answers) with same weights/eligibility
    # Must match the budget handler in simulate_with_stopping() exactly:
    # apply filter_mask AND truncation_positions to determine voter eligibility.
    final_elig = (sampled_final_ans >= 0)
    if filter_mask is not None:
        if filter_mask.ndim == 2:
            final_elig = final_elig & filter_mask
        else:
            final_elig = final_elig & filter_mask[:, :, -1]

    if truncation_positions is not None:
        final_pos = int(probe_positions[-1]) if n_pos > 0 else 0
        was_trunc = truncation_positions < sampled_tokens
        past_trunc = was_trunc & (final_pos >= truncation_positions)
        final_elig = final_elig & ~past_trunc

    if weights is not None and weights.ndim == 3:
        w_target = weights[:, :, -1]
    else:
        w_target = weights
    final_votes = compute_votes(sampled_final_ans, w_target, final_elig, n_answers)
    target = np.argmax(final_votes, axis=1)  # [n_iter]

    # Step 2: Compute voted answer at each probe position
    leaders = np.full((n_iter, n_pos), -1, dtype=np.int32)

    for pos_idx in range(n_pos):
        pos = int(probe_positions[pos_idx])
        ans = sampled_ans_ids[:, :, pos_idx]  # [n_iter, budget]
        elig = (ans >= 0)

        if filter_mask is not None:
            if filter_mask.ndim == 2:
                elig = elig & filter_mask
            else:
                elig = elig & filter_mask[:, :, pos_idx]

        if truncation_positions is not None:
            was_truncated = truncation_positions < sampled_tokens
            past_truncation = was_truncated & (pos >= truncation_positions)
            elig = elig & ~past_truncation

        if weights is not None and weights.ndim == 3:
            w_pos = weights[:, :, pos_idx]
        else:
            w_pos = weights
        votes = compute_votes(ans, w_pos, elig, n_answers)
        leaders[:, pos_idx] = np.argmax(votes, axis=1)

    # Step 3: Find oracle positions
    match = (leaders == target[:, None])  # [n_iter, n_pos]

    # Optimistic: first position where vote matches target
    has_match = match.any(axis=1)
    raw_first = match.argmax(axis=1)  # first True (or 0 if none)
    optimistic_pos_idx = np.where(has_match, raw_first, n_pos - 1).astype(np.int64)

    # Absorbing: one past last deviation
    no_match = ~match
    has_deviation = no_match.any(axis=1)
    raw_last_from_end = no_match[:, ::-1].argmax(axis=1)
    last_dev_idx = n_pos - 1 - raw_last_from_end
    # absorbing can be n_pos (past all probe positions) if last position deviates
    absorbing_raw = np.where(has_deviation, last_dev_idx + 1, 0).astype(np.int64)
    absorbing_needs_baseline = absorbing_raw >= n_pos
    absorbing_pos_idx = np.minimum(absorbing_raw, n_pos - 1)

    # Step 4: Compute tokens at every position (vectorized)
    tokens_at_pos = np.zeros((n_iter, n_pos), dtype=np.int64)
    for pos_idx in range(n_pos):
        pos = int(probe_positions[pos_idx])
        if truncation_positions is not None:
            capped = np.minimum(np.minimum(sampled_tokens, truncation_positions), pos)
        else:
            capped = np.minimum(sampled_tokens, pos)
        tokens_at_pos[:, pos_idx] = capped.sum(axis=1)

    # Baseline tokens (no stopping)
    if truncation_positions is not None:
        baseline_tokens = np.minimum(sampled_tokens, truncation_positions).sum(axis=1)
    else:
        baseline_tokens = sampled_tokens.sum(axis=1)

    iter_range = np.arange(n_iter)

    # Optimistic: use baseline if no match found at any position
    optimistic_tokens = np.where(
        has_match,
        tokens_at_pos[iter_range, optimistic_pos_idx],
        baseline_tokens,
    )

    # Absorbing: use baseline if last position still deviates
    absorbing_tokens = np.where(
        absorbing_needs_baseline,
        baseline_tokens,
        tokens_at_pos[iter_range, absorbing_pos_idx],
    )

    return {
        'target': target,
        'leaders': leaders,
        'optimistic_pos_idx': optimistic_pos_idx,
        'absorbing_pos_idx': absorbing_pos_idx,
        'optimistic_tokens': optimistic_tokens,
        'absorbing_tokens': absorbing_tokens,
        'baseline_tokens': baseline_tokens,
    }


def check_correctness(
    result: SimulationResult,
    ground_truth: str,
) -> np.ndarray:
    """Check if each iteration's answer is correct.

    Args:
        result: SimulationResult
        ground_truth: correct answer string

    Returns:
        [n_iter] bool
    """
    n_iter = len(result.answers)
    is_correct = np.zeros(n_iter, dtype=bool)

    for i in range(n_iter):
        ans_id = result.answers[i]
        if ans_id >= 0 and ans_id in result.canonical_map:
            ans_str = result.canonical_map[ans_id]
            try:
                is_correct[i] = answers_equivalent(ans_str, ground_truth)
            except:
                is_correct[i] = ans_str == ground_truth

    return is_correct


def _calibrate_gamma(
    precomputed: PrecomputedQuestion,
    warmup_ids: np.ndarray,
    q_values: np.ndarray,
    method: str,
    method_config: dict,
    qid: int,
) -> float:
    """Calibrate gamma from warmup traces (already filtered to valid final answers)."""
    from .voting import calibrate_gamma_warmup

    if len(warmup_ids) < 4:
        print(f"    Q{qid:02d}: warmup_gamma=1.00 (only {len(warmup_ids)} warmup traces)")
        return 1.0

    warmup_ans = precomputed.ans_ids_at_pos[warmup_ids]
    warmup_q_vals = q_values[warmup_ids]
    warmup_elig = (warmup_ans >= 0)
    if method.startswith('dco'):
        warmup_conf = precomputed.conf_at_pos[warmup_ids]
        threshold = np.percentile(precomputed.min_group[warmup_ids], 90)
        cum_min = np.minimum.accumulate(warmup_conf, axis=1)
        warmup_elig = warmup_elig & (cum_min >= threshold)
        warmup_weights = warmup_conf
    else:
        warmup_weights = None
    warmup_final_ans = precomputed.answer_ids[warmup_ids]
    gamma = calibrate_gamma_warmup(
        warmup_ans, warmup_q_vals, warmup_weights, warmup_elig,
        precomputed.n_answers,
        ucb_z=method_config.get('ucb_z', 0.0),
        final_answer_ids=warmup_final_ans,
        gamma_min=method_config.get('gamma_min', 0.5),
    )
    print(f"    Q{qid:02d}: warmup_gamma={gamma:.2f} ({len(warmup_ids)} warmup traces)")
    return gamma


# ─────────────────────────────────────────────────────────────────────────────
# Per-question Worker
# ─────────────────────────────────────────────────────────────────────────────

def worker_process_question(args: tuple) -> tuple[int, Optional[Dict[str, Any]]]:
    """Worker function for multiprocessing.

    Loads data from disk, precomputes, samples, simulates.
    Returns lightweight results (no large arrays).

    Args:
        args: (qid, data_path, method_config)
            method_config is a dict with keys:
                budget, n_iterations, seed, window_size,
                method, delta, warmup, gamma, warmup_gamma, ...

    Returns:
        (question_id, results_dict or None)
    """
    qid, data_path_str, method_config = args

    from pathlib import Path
    from .npz_loader import load_question_npz
    from .pkl_loader import load_pkl_question
    from .probed_traces import load_probed_question
    from .voting import NeverStop

    data_path = Path(data_path_str)
    budget = method_config['budget']
    n_iterations = method_config['n_iterations']
    seed = method_config['seed']
    window_size = method_config['window_size']
    method = method_config['method']
    data_format = method_config.get('data_format', 'npz')

    # Load question data
    if data_format == 'pkl':
        q_pkl_path = Path(method_config['pkl_cache_dir']) / f"q{qid:02d}.pkl"
        try:
            raw_q, probed_q = load_pkl_question(q_pkl_path)
        except Exception as e:
            print(f"Worker: Failed to load question {qid} from pkl: {e}")
            return qid, None
    else:
        question_path = data_path / "raw_traces_compact" / f"q{qid:02d}.npz"
        try:
            raw_q = load_question_npz(question_path)
        except Exception as e:
            print(f"Worker: Failed to load question {qid}: {e}")
            return qid, None

        probed_q = load_probed_question(data_path, qid)
        if probed_q is None:
            print(f"Worker: No probed data for question {qid}")
            return qid, None

    # Precompute (with caching)
    cache_base = Path(method_config['pkl_cache_dir']) if data_format == 'pkl' else data_path
    precomputed = precompute_question_cached(raw_q, probed_q, cache_base, window_size)

    # Bootstrap and warmup sample only from traces with a valid final answer.
    # Traces with answer_ids=-1 never completed reasoning: they contribute
    # meaningless "no answer" votes at the final position, and their oracle q
    # is mechanically 1.0 (intermediate answer != -1) which inflates the NC
    # threshold for oracle methods but not for learned q-models, creating an
    # inconsistency that causes gamma calibration failures.
    all_probed_ids = np.array(
        [t.trace_id for t in probed_q.traces], dtype=np.int64
    )
    has_final = precomputed.answer_ids[all_probed_ids] >= 0
    probed_trace_ids = all_probed_ids[has_final]

    samples = sample_bootstrap(
        precomputed, n_iterations, budget,
        seed=seed + qid * 10000,
        probed_trace_ids=probed_trace_ids,
    )

    warmup_n = method_config.get('warmup', 16)
    warmup_ids = probed_trace_ids[:warmup_n]

    # Warmup reference answer: majority final answer among warmup traces.
    # Used to guard probe-0 stops against placeholder-answer false positives.
    _wf = precomputed.answer_ids[warmup_ids]
    _wf_valid = _wf[_wf >= 0]
    warmup_ref_answer = int(np.bincount(_wf_valid).argmax()) if len(_wf_valid) > 0 else -1

    # ── Configure method ──
    #
    # Methods are parameterizations of one simulation loop, differing in vote
    # weighting, trace filtering, and stopping rule:
    #   offline            : uniform weights, no stopping, all 7 DeepConf voting methods
    #   dco                : conf weights + threshold filter (DeepConf Online), NeverStop
    #   sc-mars            : uniform weights, MARS stopping (learned 5-feature q-model)
    #   dco-mars           : DeepConf filter + MARS stopping
    #   *-mars-cal         : MARS with per-question gamma calibration (oracle-q for calib)
    #   sc/dco-mars-oracle : MARS stopping with oracle-q (diagnostic upper bound)
    #   sc-pp              : Parallel-Probe baseline (consensus stability + pruning)
    #   oracle / oracle-dco: oracle stopping bounds (optimistic/absorbing)
    #
    weights = None
    filter_mask = None
    truncation_positions = None

    if method in ('dco', 'oracle-dco', 'dco-mars-oracle', 'dco-mars-cal'):
        weights = samples['sampled_conf_at_pos']  # [n_iter, budget, n_pos] — position-dependent

    # Threshold filter (the DeepConf-Online "discard low-confidence traces" step).
    # Note: dco-mars gets the filter+truncation but NOT confidence weighting
    # (it is intentionally absent from the tuple above), so it votes uniformly
    # over the filtered set.
    if method in ('dco', 'oracle-dco', 'dco-mars-oracle', 'dco-mars', 'dco-mars-cal'):
        warmup_n = method_config.get('warmup', 16)
        warmup_confs = samples['sampled_min_confs'][:, :warmup_n]  # OK: warmup fully observed
        thresholds = np.percentile(warmup_confs, 90, axis=1)  # [n_iter]

        # Dynamic eligibility: cumulative-min of conf_at_pos ensures permanent truncation.
        # Once a trace's confidence drops below threshold at any probe position,
        # it stays excluded at all subsequent positions (F_t-measurable).
        cum_min_conf = np.minimum.accumulate(
            samples['sampled_conf_at_pos'], axis=2  # along position axis
        )  # [n_iter, budget, n_pos]
        filter_mask = cum_min_conf >= thresholds[:, np.newaxis, np.newaxis]  # 3D

        # Truncation positions for token counting: traces physically stop generating
        # when a window drops below threshold (simulates the logits processor).
        trunc_pos = np.zeros((n_iterations, budget), dtype=np.int64)
        for iter_idx in range(n_iterations):
            threshold = thresholds[iter_idx]
            for j in range(budget):
                trace_idx = samples['indices'][iter_idx, j]
                trunc_pos[iter_idx, j] = find_truncation_position(
                    precomputed.group_confs[trace_idx],
                    int(precomputed.num_tokens[trace_idx]),
                    threshold,
                    window_size,
                )
        truncation_positions = trunc_pos

    # Oracle: compute both oracle bounds and return early (no simulate_with_stopping)
    if method.startswith('oracle'):
        oracle_result = simulate_oracle(
            sampled_ans_ids=samples['sampled_ans_ids'],
            sampled_tokens=samples['sampled_tokens'],
            probe_positions=precomputed.probe_positions,
            n_answers=precomputed.n_answers,
            sampled_final_ans=samples['sampled_final_ans'],
            weights=weights,
            filter_mask=filter_mask,
            truncation_positions=truncation_positions,
        )

        # Correctness: oracle always outputs the target (= vote of final answers)
        target_ids = oracle_result['target']
        oracle_correct = np.zeros(n_iterations, dtype=bool)
        oracle_answer_strings = []
        for i in range(n_iterations):
            aid = int(target_ids[i])
            if aid >= 0 and aid in precomputed.canonical_map:
                ans_str = precomputed.canonical_map[aid]
                oracle_answer_strings.append(ans_str)
                try:
                    oracle_correct[i] = answers_equivalent(ans_str, raw_q.ground_truth)
                except Exception:
                    oracle_correct[i] = ans_str == raw_q.ground_truth
            else:
                oracle_answer_strings.append(None)

        output = {
            'is_correct': oracle_correct.tolist(),
            'tokens': oracle_result['baseline_tokens'].tolist(),
            'baseline_tokens': oracle_result['baseline_tokens'].tolist(),
            'positions': [int(precomputed.probe_positions[-1])] * n_iterations,
            'stopped_by': ['budget'] * n_iterations,
            'answers': oracle_answer_strings,
            'oracle': {
                'optimistic_tokens': oracle_result['optimistic_tokens'].tolist(),
                'absorbing_tokens': oracle_result['absorbing_tokens'].tolist(),
                'optimistic_pos_idx': oracle_result['optimistic_pos_idx'].tolist(),
                'absorbing_pos_idx': oracle_result['absorbing_pos_idx'].tolist(),
                'baseline_tokens': oracle_result['baseline_tokens'].tolist(),
            },
        }
        return qid, output

    # Stopping strategies (label, strategy) — checked in order, first match wins
    if method in ('sc-mars', 'dco-mars', 'sc-mars-cal', 'dco-mars-cal'):
        # MARS: learned 5-feature switch-probability model + Platt calibration.
        # *-cal variants: learned q at inference, oracle q for gamma calibration.
        from .q_model import (
            build_switch_features,
            build_training_labels,
            fit_q_model,
            fit_platt_calibration,
            precompute_switch_probs,
        )
        from .voting import MarsStopping

        delta = method_config.get('delta', 0.05)
        use_oracle_gamma = method.endswith('-cal')

        warmup_X = build_switch_features(
            precomputed.probe_positions,
            precomputed.conf_at_pos[warmup_ids],
            precomputed.flips[warmup_ids],
            precomputed.streaks[warmup_ids],
        )
        warmup_y = build_training_labels(
            precomputed.ans_ids_at_pos[warmup_ids],
            precomputed.answer_ids[warmup_ids],
        )
        model = fit_q_model(warmup_X, warmup_y)

        # Platt calibration on warmup in-sample predictions
        warmup_q_pred = model.predict(warmup_X)
        calibrator = fit_platt_calibration(warmup_q_pred, warmup_y)

        # Precompute q values for all traces
        q_values = precompute_switch_probs(
            model, calibrator,
            precomputed.conf_at_pos, precomputed.flips,
            precomputed.streaks, precomputed.probe_positions,
        )
        sampled_q = q_values[samples['indices']]

        w_max = 1.0

        gamma = method_config.get('gamma', 1.0)
        if method_config.get('warmup_gamma', False):
            if use_oracle_gamma:
                from .q_model import compute_oracle_q_values
                calib_q = compute_oracle_q_values(
                    precomputed.ans_ids_at_pos, precomputed.answer_ids,
                )
            else:
                calib_q = q_values
            gamma = _calibrate_gamma(
                precomputed, warmup_ids, calib_q,
                method, method_config, qid,
            )

        stopping_strategies = [
            ("margin", MarsStopping(
                q_t=sampled_q,
                w_max=w_max,
                delta=delta,
                n_answers=precomputed.n_answers,
                use_correction=False,
                gamma=gamma,
                warmup_ref_answer=warmup_ref_answer,
            )),
        ]
    elif method in ('dco-mars-oracle', 'sc-mars-oracle'):
        # Oracle q_t + MarsStopping (no correction) — diagnostic
        from .q_model import compute_oracle_q_values
        from .voting import MarsStopping

        delta = method_config.get('delta', 0.05)

        oracle_q = compute_oracle_q_values(
            precomputed.ans_ids_at_pos,  # [n_traces, n_positions]
            precomputed.answer_ids,       # [n_traces]
        )
        sampled_q = oracle_q[samples['indices']]  # [n_iter, budget, n_positions]

        w_max = 1.0  # conf_at_pos ∈ [0, 1], safe upper bound

        gamma = method_config.get('gamma', 1.0)
        if method_config.get('warmup_gamma', False):
            gamma = _calibrate_gamma(
                precomputed, warmup_ids, oracle_q,
                method, method_config, qid,
            )

        stopping_strategies = [
            ("margin", MarsStopping(
                q_t=sampled_q,
                w_max=w_max,
                delta=delta,
                n_answers=precomputed.n_answers,
                use_correction=False,
                gamma=gamma,
                warmup_ref_answer=warmup_ref_answer,
            )),
        ]
    elif method == 'sc-pp':
        from .voting import ParallelProbeStopping

        pp_conv = method_config.get('pp_conv', 3)
        pp_warmup = method_config.get('pp_warmup', 4)
        pp_prune = method_config.get('pp_prune_patience', 2)

        stopping_strategies = [
            ("pp_consensus", ParallelProbeStopping(
                n_iter=n_iterations,
                budget=budget,
                n_answers=precomputed.n_answers,
                conv=pp_conv,
                warmup=pp_warmup,
                prune_patience=pp_prune,
            )),
        ]

    else:
        # offline / dco — no stopping
        stopping_strategies = [
            ("none", NeverStop()),
        ]

    # Run simulation
    min_voters = method_config.get('min_voters', 0)
    result = simulate_with_stopping(
        sampled_ans_ids=samples['sampled_ans_ids'],
        sampled_tokens=samples['sampled_tokens'],
        probe_positions=precomputed.probe_positions,
        canonical_map=precomputed.canonical_map,
        n_answers=precomputed.n_answers,
        stopping_strategies=stopping_strategies,
        weights=weights,
        filter_mask=filter_mask,
        truncation_positions=truncation_positions,
        sampled_final_ans=samples['sampled_final_ans'],
        min_voters=min_voters,
    )

    # Check correctness
    is_correct = check_correctness(result, raw_q.ground_truth)

    # Convert to serializable output
    answer_strings = []
    for i in range(n_iterations):
        aid = result.answers[i]
        if aid >= 0 and aid in precomputed.canonical_map:
            answer_strings.append(precomputed.canonical_map[aid])
        else:
            answer_strings.append(None)

    output = {
        'is_correct': is_correct.tolist(),
        'tokens': result.total_tokens.tolist(),
        'baseline_tokens': result.baseline_tokens.tolist(),
        'positions': result.stop_positions.tolist(),
        'stopped_by': result.stopped_by.tolist(),
        'answers': answer_strings,
    }

    # For offline: compute all 7 deepconf voting methods
    if method == 'offline':
        from .answer_equiv import answers_equivalent as _ans_equiv

        final_ans = samples['sampled_final_ans']  # [n_iter, budget]
        base_elig = final_ans >= 0

        voting_results = compute_deepconf_votes(
            final_ans=final_ans,
            eligibility=base_elig,
            mean_conf=samples.get('sampled_mean_confs', np.zeros_like(final_ans, dtype=np.float64)),
            tail_conf=samples.get('sampled_tail_confs', np.zeros_like(final_ans, dtype=np.float64)),
            bottom_10=samples['sampled_bottom_10'],
            min_group=samples['sampled_min_confs'],
            n_answers=precomputed.n_answers,
        )

        # Check correctness for each voting method and serialize
        voting_out = {}
        for vm_name, vm_answer_ids in voting_results.items():
            vm_correct = np.zeros(n_iterations, dtype=bool)
            vm_answers = []
            for i in range(n_iterations):
                aid = int(vm_answer_ids[i])
                if aid >= 0 and aid in precomputed.canonical_map:
                    ans_str = precomputed.canonical_map[aid]
                    vm_answers.append(ans_str)
                    try:
                        vm_correct[i] = _ans_equiv(ans_str, raw_q.ground_truth)
                    except Exception:
                        vm_correct[i] = ans_str == raw_q.ground_truth
                else:
                    vm_answers.append(None)
            voting_out[vm_name] = {
                'answers': vm_answers,
                'is_correct': vm_correct.tolist(),
            }

        output['voting_results'] = voting_out

    return qid, output


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Dataset Processing
# ─────────────────────────────────────────────────────────────────────────────

def simulate_dataset_parallel(
    data_path: str,
    question_ids: List[int],
    method_config: Dict[str, Any],
    n_workers: int = 4,
) -> Dict[int, Dict[str, Any]]:
    """Run simulation across all questions in parallel.

    Args:
        data_path: path to data directory
        question_ids: list of question IDs to process
        method_config: configuration dict for worker_process_question
        n_workers: number of parallel workers

    Returns:
        Dict mapping question_id -> results dict
    """
    worker_args = [
        (qid, data_path, method_config)
        for qid in question_ids
    ]

    per_question_results = {}

    if n_workers > 1 and len(question_ids) > 1:
        print(f"  Processing {len(question_ids)} questions with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(worker_process_question, args): args[0]
                for args in worker_args
            }
            for future in as_completed(futures):
                qid, results = future.result()
                if results is not None:
                    per_question_results[qid] = results
                    print(f"    Completed question {qid}")
                else:
                    print(f"    Warning: question {qid} returned no results")
    else:
        print(f"  Processing {len(question_ids)} questions sequentially...")
        for args in worker_args:
            qid, results = worker_process_question(args)
            if results is not None:
                per_question_results[qid] = results
                print(f"    Completed question {qid}")

    return per_question_results
