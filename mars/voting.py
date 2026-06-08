"""
Consolidated voting logic and pluggable stopping strategies.

All voting methods reduce to: eligible traces contribute weighted votes, pick
the leader. All stopping strategies implement the StoppingStrategy protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Voting
# ─────────────────────────────────────────────────────────────────────────────

def compute_votes(
    answer_ids: np.ndarray,
    weights: Optional[np.ndarray],
    eligible: np.ndarray,
    n_answers: int,
) -> np.ndarray:
    """Accumulate weighted votes for a batch of iterations at one position.

    Args:
        answer_ids: [n_iter, budget] int — answer group ID per trace (-1 = no answer)
        weights:    [n_iter, budget] float — per-trace vote weight (None = uniform 1.0)
        eligible:   [n_iter, budget] bool — which traces are eligible to vote
        n_answers:  number of unique answer groups

    Returns:
        vote_weights: [n_iter, n_answers] float64 — accumulated vote weight per answer
    """
    n_iter, budget = answer_ids.shape
    vote_weights = np.zeros((n_iter, n_answers), dtype=np.float64)

    # Valid = has answer AND is eligible
    valid = (answer_ids >= 0) & eligible

    if not valid.any():
        return vote_weights

    iter_idx = np.broadcast_to(
        np.arange(n_iter)[:, np.newaxis], answer_ids.shape
    )

    if weights is not None:
        np.add.at(
            vote_weights,
            (iter_idx[valid], answer_ids[valid]),
            weights[valid],
        )
    else:
        np.add.at(
            vote_weights,
            (iter_idx[valid], answer_ids[valid]),
            1,
        )

    return vote_weights


def get_leader(vote_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract leader info from vote weight matrix.

    Args:
        vote_weights: [n_iter, n_answers]

    Returns:
        leader_ids:    [n_iter] int   — answer ID of leader
        leader_weight: [n_iter] float — leader's total weight
    """
    leader_ids = np.argmax(vote_weights, axis=1)
    leader_weight = np.take_along_axis(
        vote_weights, leader_ids[:, np.newaxis], axis=1
    ).squeeze(1)
    return leader_ids, leader_weight


# ─────────────────────────────────────────────────────────────────────────────
# Stopping Strategy Protocol
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionState:
    """Everything known at a probe position, passed to stopping strategies."""
    vote_weights: np.ndarray       # [n_active, n_answers]
    answer_ids: np.ndarray         # [n_active, budget] answer ID at this position
    weights: Optional[np.ndarray]  # [n_active, budget] per-trace vote weight
    eligible: np.ndarray           # [n_active, budget] which traces are eligible
    w_active: np.ndarray           # [n_active] total weight of not-yet-finished traces
    w_active_by_ans: np.ndarray    # [n_active, n_answers] active weight by answer
    pos_idx: int                   # index into probe_positions
    n_active_traces: np.ndarray    # [n_active] count of active traces per iteration
    active_iter_indices: Optional[np.ndarray] = None  # [n_active] int — original iteration indices


class StoppingStrategy(Protocol):
    def check(self, state: PositionState) -> np.ndarray:
        """Return [n_active] bool: can this iteration stop?"""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Strategies
# ─────────────────────────────────────────────────────────────────────────────

class MarsStopping:
    """Per-trace q model: stop when margin > sum(q_t * c_t) + correction.

    Uses precomputed q_t values indexed by (bootstrap_sample, position).
    """

    def __init__(
        self,
        q_t: np.ndarray,
        w_max: float,
        delta: float,
        n_answers: int,
        use_correction: bool = True,
        gamma: float = 1.0,
        warmup_ref_answer: int = -1,
    ):
        """
        Args:
            q_t:       [n_iter, budget, n_positions] float — precomputed P(switch)
            w_max:     float — maximum vote weight (for Hoeffding correction)
            delta:     float — failure probability bound
            n_answers: int — total number of answer groups (for union bound |A|)
            use_correction: bool — if False, skip the Hoeffding correction term
            gamma:     float — cost shrinkage factor for L-voter and other-voter
                       adversarial costs (1.0 = full worst-case, <1 = shrunk)
            warmup_ref_answer: int — if >= 0, block probe-0 stops when leader
                       differs from this answer (guards against placeholder-answer
                       early stops where q cannot distinguish switchers)
        """
        self.q_t = q_t
        self.w_max = w_max
        self.delta = delta
        self.n_answers = n_answers
        self.use_correction = use_correction
        self.gamma = gamma
        self.warmup_ref_answer = warmup_ref_answer

    def check(self, state: PositionState) -> np.ndarray:
        vote_weights = state.vote_weights        # [n_active, n_answers]
        answer_ids = state.answer_ids            # [n_active, budget]
        weights = state.weights                  # [n_active, budget] or None
        eligible = state.eligible                # [n_active, budget]
        w_active_by_ans = state.w_active_by_ans  # [n_active, n_answers]
        pos_idx = state.pos_idx
        n_active_traces = state.n_active_traces  # [n_active]

        n_active, n_answers_state = vote_weights.shape
        n_budget = answer_ids.shape[1]

        # Get q values at this position for active iterations
        if state.active_iter_indices is not None:
            q_at_pos = self.q_t[state.active_iter_indices, :, pos_idx]  # [n_active, budget]
        else:
            q_at_pos = self.q_t[:n_active, :, pos_idx]  # [n_active, budget]

        # Trace weights (default 1.0 for uniform)
        if weights is not None:
            w_t = weights
        else:
            w_t = np.ones((n_active, n_budget), dtype=np.float64)

        # Leader
        leader_ids = np.argmax(vote_weights, axis=1)  # [n_active]
        leader_weights = np.take_along_axis(
            vote_weights, leader_ids[:, np.newaxis], axis=1
        ).squeeze(1)

        has_votes = vote_weights.sum(axis=1) > 0

        # Hoeffding correction: w_max * sqrt(2 * N * log(|A| / delta))
        if self.use_correction:
            log_term = np.log(max(self.n_answers, 1) / self.delta)
            correction = self.w_max * np.sqrt(
                2.0 * n_active_traces.astype(np.float64) * log_term
            )  # [n_active]
        else:
            correction = np.zeros_like(n_active_traces, dtype=np.float64)

        # Precompute leader mask (shared by worst-case and per-challenger checks)
        trace_ans = answer_ids  # [n_active, budget]
        votes_for_leader = trace_ans == leader_ids[:, np.newaxis]  # [n_active, budget]
        active_mask = eligible  # [n_active, budget]

        # Worst-case zero-vote challenger: no trace votes for it, so every
        # switching trace contributes maximum cost (no c_j=0 reductions).
        # This handles challengers with 0 current votes that the per-answer
        # loop would skip — without it, early unanimous consensus causes
        # NC to fire immediately even when most traces will switch later.
        # gamma shrinks L-voter (2w) and other-voter (w) costs; no k-voter here.
        c_worst = self.gamma * np.where(votes_for_leader, 2.0 * w_t, w_t) * active_mask
        worst_threshold = (q_at_pos * c_worst).sum(axis=1) + correction
        can_stop = (leader_weights > worst_threshold) & has_votes

        # Per-challenger refinement: challengers with actual votes have
        # LOWER thresholds (k-voter departure benefit: c_j=-w_j),
        # but also smaller margins (M_a = V_L - V_a < V_L).
        # The AND logic means this can only make can_stop stricter.
        for a in range(n_answers_state):
            a_votes = vote_weights[:, a]
            has_a_votes = a_votes > 0
            is_leader_a = leader_ids == a
            relevant = has_a_votes & ~is_leader_a & has_votes
            if not relevant.any():
                continue

            M_a = leader_weights - a_votes  # [n_active]

            votes_for_a = trace_ans == a  # [n_active, budget]
            # gamma shrinks L-voter (2w) and other-voter (w) costs;
            # k-voter departure benefit (-w) is already tight, unchanged.
            c_t = np.where(
                votes_for_leader, self.gamma * 2.0 * w_t,
                np.where(votes_for_a, -w_t, self.gamma * w_t)
            ) * active_mask  # [n_active, budget]

            adv_expect = (q_at_pos * c_t).sum(axis=1)  # [n_active]
            threshold = adv_expect + correction  # [n_active]
            safe_a = M_a > threshold
            can_stop &= safe_a | ~relevant

        if self.warmup_ref_answer >= 0 and pos_idx == 0:
            can_stop &= (leader_ids == self.warmup_ref_answer)

        return can_stop


class ConsensusStopping:
    """Consensus: stop when leader has >= tau fraction of total weight."""

    def __init__(self, tau: float = 0.95):
        self.tau = tau

    def check(self, state: PositionState) -> np.ndarray:
        vote_weights = state.vote_weights  # [n_active, n_answers]
        total_weight = vote_weights.sum(axis=1)  # [n_active]
        has_votes = total_weight > 0

        _, leader_weight = get_leader(vote_weights)

        ratio = np.zeros_like(total_weight)
        np.divide(leader_weight, total_weight, out=ratio, where=has_votes)

        return has_votes & (ratio >= self.tau)


class CompositeStopping:
    """Combine multiple strategies with OR logic: stop if ANY strategy says stop."""

    def __init__(self, strategies: List[StoppingStrategy]):
        self.strategies = strategies

    def check(self, state: PositionState) -> np.ndarray:
        result = np.zeros(state.vote_weights.shape[0], dtype=bool)
        for strategy in self.strategies:
            result |= strategy.check(state)
        return result


class ParallelProbeStopping:
    """Parallel-Probe (Zheng et al. 2026): consensus stability + deviation pruning.

    Faithful reimplementation of the 2D probing controller from
    "Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing".

    Two mechanisms:
      1. Consensus early stopping: halt when majority vote is stable for
         `conv` consecutive probe positions.
      2. Deviation-based branch pruning: kill a trace that disagrees with
         consensus for `prune_patience` consecutive positions (after warmup).

    Maintains internal state across check() calls (stability counts, prune masks).
    Must be instantiated fresh per question.
    """

    def __init__(
        self,
        n_iter: int,
        budget: int,
        n_answers: int,
        conv: int = 3,
        warmup: int = 4,
        prune_patience: int = 2,
    ):
        self.conv = conv
        self.warmup = warmup
        self.prune_patience = prune_patience
        self.n_answers = n_answers

        self.stable_count = np.zeros(n_iter, dtype=np.int32)
        self.prev_winner = np.full(n_iter, -1, dtype=np.int32)
        self.pruned = np.zeros((n_iter, budget), dtype=bool)
        self.off_track = np.zeros((n_iter, budget), dtype=np.int32)

    def check(self, state: PositionState) -> np.ndarray:
        pos_idx = state.pos_idx
        answer_ids = state.answer_ids      # [n_active, budget]
        eligible = state.eligible          # [n_active, budget]
        n_active = answer_ids.shape[0]

        # Map active iterations to their original indices
        if state.active_iter_indices is not None:
            idx = state.active_iter_indices
        else:
            idx = np.arange(n_active)

        # Apply prune mask to eligibility (pruned traces excluded from votes)
        elig = eligible & ~self.pruned[idx]

        # Recompute votes with pruning applied (uniform weights — PP uses no weighting)
        vote_weights = np.zeros((n_active, self.n_answers), dtype=np.float64)
        valid = (answer_ids >= 0) & elig
        if valid.any():
            iter_bc = np.broadcast_to(
                np.arange(n_active)[:, np.newaxis], answer_ids.shape
            )
            np.add.at(vote_weights, (iter_bc[valid], answer_ids[valid]), 1)

        # Current majority winner per iteration
        has_votes = vote_weights.sum(axis=1) > 0
        winner = np.where(has_votes, np.argmax(vote_weights, axis=1), -1)

        # Update stability count
        same_as_prev = (winner == self.prev_winner[idx]) & (winner >= 0)
        self.stable_count[idx] = np.where(same_as_prev, self.stable_count[idx] + 1, 1)
        self.prev_winner[idx] = winner

        # Pruning (only after warmup)
        if pos_idx >= self.warmup and self.prune_patience > 0:
            # For each trace: if answer != winner, increment off_track; else reset
            trace_ans = answer_ids  # [n_active, budget]
            agrees = (trace_ans == winner[:, np.newaxis]) | ~elig
            self.off_track[idx] = np.where(
                agrees,
                0,
                self.off_track[idx] + 1,
            )
            # Prune traces exceeding patience
            newly_pruned = self.off_track[idx] >= self.prune_patience
            self.pruned[idx] |= newly_pruned

        # Can stop: stability >= conv AND past warmup
        can_stop = (self.stable_count[idx] >= self.conv) & has_votes
        if pos_idx < self.warmup:
            can_stop[:] = False

        return can_stop


class NeverStop:
    """Run all positions (for offline methods)."""

    def check(self, state: PositionState) -> np.ndarray:
        return np.zeros(state.vote_weights.shape[0], dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup Gamma Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_gamma_warmup(
    ans_ids_at_pos: np.ndarray,
    q_at_pos: np.ndarray,
    weights_at_pos: Optional[np.ndarray],
    eligible_at_pos: np.ndarray,
    n_answers: int,
    gamma_grid: Optional[np.ndarray] = None,
    ucb_z: float = 0.0,
    final_answer_ids: Optional[np.ndarray] = None,
    gamma_min: float = 0.5,
) -> float:
    """Calibrate gamma per-question by sweeping on warmup traces with UCB correction.

    Runs NC stopping on warmup traces for each gamma in the grid.
    Walks down from $\\gamma=1.0$ and returns the smallest gamma in the
    contiguous band that preserves the reference answer. Then applies a
    UCB correction to compensate for the structural bias: warmup has fewer
    challengers than bootstrap, making NC systematically easier and gamma
    systematically too low.

    The reference answer is determined from final_answer_ids (the true
    full-generation answers) rather than the last grid position, which can
    differ for late-converging traces.

    UCB correction: $\\gamma_{\\text{out}} = \\min(1, \\gamma_{\\text{band}} + z / \\sqrt{n_{\\text{elig}}})$
    where $n_{\\text{elig}}$ is the number of eligible traces at the $\\gamma=1$
    stopping position.

    Args:
        ans_ids_at_pos:    [n_warmup, n_positions] int — answer IDs at each position
        q_at_pos:          [n_warmup, n_positions] float — P(switch) per trace/position
        weights_at_pos:    [n_warmup, n_positions] float or None — vote weights (None=uniform)
        eligible_at_pos:   [n_warmup, n_positions] bool — position-dependent eligibility
        n_answers:         int — number of unique answer groups
        gamma_grid:        optional gamma values to sweep (default: 0.0 to 1.0 step 0.1)
        ucb_z:             UCB correction strength (default 0.0 = no correction)
        final_answer_ids:  [n_warmup] int — true final answer IDs from full generation.
                           Used as the budget reference answer. Falls back to last grid
                           position if None.
        gamma_min:         floor for calibrated gamma (default 0.5)

    Returns:
        float — calibrated gamma (band + UCB + floor, clipped to [gamma_min, 1])
    """
    if gamma_grid is None:
        gamma_grid = np.arange(0.0, 1.05, 0.1)

    n_warmup, n_positions = ans_ids_at_pos.shape

    # Determine the true budget reference answer from final answer IDs
    if final_answer_ids is not None:
        w_budget = (weights_at_pos[:, -1] if weights_at_pos is not None
                    else np.ones(n_warmup, dtype=np.float64))
        vote_final = np.zeros(n_answers, dtype=np.float64)
        valid_final = final_answer_ids >= 0
        if valid_final.any():
            np.add.at(vote_final, final_answer_ids[valid_final], w_budget[valid_final])
        budget_ref_answer = int(np.argmax(vote_final))
    else:
        budget_ref_answer = None

    def _nc_stopping_result(gamma: float) -> tuple[int, int]:
        """Find (answer_id, stop_pos_idx) for NC stopping with given gamma."""
        for pos_idx in range(n_positions):
            ans = ans_ids_at_pos[:, pos_idx]
            elig = eligible_at_pos[:, pos_idx]
            w = (weights_at_pos[:, pos_idx] if weights_at_pos is not None
                 else np.ones(n_warmup, dtype=np.float64))

            # Compute votes
            vote_w = np.zeros(n_answers, dtype=np.float64)
            valid = (ans >= 0) & elig
            if not valid.any():
                continue
            np.add.at(vote_w, ans[valid], w[valid])

            leader_id = int(np.argmax(vote_w))
            leader_weight = vote_w[leader_id]
            if leader_weight <= 0:
                continue

            q = q_at_pos[:, pos_idx]
            votes_for_leader = (ans == leader_id) & elig

            # Worst-case zero-vote challenger guard
            c_worst = gamma * np.where(votes_for_leader, 2.0 * w, w) * elig
            worst_threshold = (q * c_worst).sum()
            if leader_weight <= worst_threshold:
                continue

            # Per-challenger refinement
            all_safe = True
            for a in range(n_answers):
                if a == leader_id:
                    continue
                if vote_w[a] <= 0:
                    continue
                M_a = leader_weight - vote_w[a]
                votes_for_a = (ans == a) & elig
                c_t = np.where(
                    votes_for_leader, gamma * 2.0 * w,
                    np.where(votes_for_a, -w, gamma * w),
                ) * elig
                if M_a <= (q * c_t).sum():
                    all_safe = False
                    break

            if all_safe:
                return leader_id, pos_idx

        # Budget: use final answer IDs if available, else last grid position
        if budget_ref_answer is not None:
            return budget_ref_answer, n_positions - 1

        ans_last = ans_ids_at_pos[:, -1]
        elig_last = eligible_at_pos[:, -1]
        w_last = (weights_at_pos[:, -1] if weights_at_pos is not None
                  else np.ones(n_warmup, dtype=np.float64))
        vote_last = np.zeros(n_answers, dtype=np.float64)
        valid_last = (ans_last >= 0) & elig_last
        if valid_last.any():
            np.add.at(vote_last, ans_last[valid_last], w_last[valid_last])
        return int(np.argmax(vote_last)), n_positions - 1

    # Compute answer for each gamma
    results = {}
    for gamma in gamma_grid:
        results[round(float(gamma), 2)] = _nc_stopping_result(float(gamma))

    ref_answer = results[1.0][0]

    # Walk down from γ=1.0: find smallest γ in the contiguous band
    # that preserves the reference answer.
    smallest_stable = 1.0
    for gamma in sorted(results.keys(), reverse=True):
        if results[gamma][0] == ref_answer:
            smallest_stable = gamma
        else:
            break

    # UCB correction: compensate for structural bias from fewer challengers
    # in warmup vs bootstrap. n_elig at γ=1.0 stop position determines the
    # effective sample size.
    if ucb_z > 0:
        ref_stop_pos = results[1.0][1]
        n_elig = int(eligible_at_pos[:, ref_stop_pos].sum())
        n_elig = max(n_elig, 1)  # avoid division by zero
        correction = ucb_z / np.sqrt(n_elig)
        smallest_stable = min(1.0, smallest_stable + correction)

    smallest_stable = max(smallest_stable, gamma_min)
    return round(smallest_stable, 2)
