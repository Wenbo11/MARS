# System Design: MARS Simulator

> Architecture of the confidence-based voting and early-stopping simulator
> (`mars/` package + `examples/run_experiment.py`).

## 1. Problem Statement

The simulator evaluates **early-stopping strategies for parallel LLM reasoning
traces** on math benchmarks. Given a pool of pre-generated traces per question,
it bootstrap-samples a budget of K traces, votes at each checkpoint, and stops
when the leader is provably safe — saving tokens.

The system supports multiple models and datasets:
- **Models**: DeepSeek-R1-Distill-Qwen-8B, Qwen3-32B, Qwen3-next (unified pickle format; a legacy NPZ+JSON format is also supported)
- **Datasets**: AIME-2025, HMMT, BRUMO-2025 (30 questions each), AIME-2024 (28 questions)
- **Data scale**: ~4096 traces per question, probed every 2048 tokens

The package is a layered library with a single CLI entry point. All methods are
parameterizations of one simulation loop, differing in vote weighting, trace
filtering, and stopping strategy.

---

## 2. Methods

All methods are parameterizations of a single simulation loop, differing in how
votes are weighted, which traces are eligible, and how the stopping decision is
made. The released set is exactly the methods reported in the paper.

| Method | Weights | Filter/Trunc | Stopping Strategy |
|--------|---------|--------------|-------------------|
| offline | uniform | None | NeverStop + all 7 DeepConf voting methods |
| dco | conf_at_pos | threshold + trunc | NeverStop (DeepConf Online baseline) |
| sc-qm3-nc | uniform | None | MARS (learned 5-feature q-model) |
| dco-qm3-nc | uniform | threshold + trunc | MARS on DeepConf-filtered traces |
| sc-qm3-nc-oqg | uniform | None | MARS + per-question $\gamma$ calibration |
| dco-qm3-nc-oqg | conf_at_pos | threshold + trunc | MARS + $\gamma$ calibration on DeepConf |
| sc-oq-nc, dco-oq-nc | uniform / conf | None / filter | MARS with oracle-$q$ (diagnostic) |
| oracle, oracle-dco | — | — | Oracle stopping bounds (optimistic/absorbing) |
| sc-pp | uniform | None | Parallel-Probe baseline |

Where:
- **Weights**: per-trace vote weight (`1.0` for uniform, or `conf_at_pos` — position-dependent sliding-window mean confidence). `dco`-prefixed methods use confidence weighting; `sc`-prefixed use uniform. (Note: `dco-qm3-nc` applies the filter but votes uniformly over the filtered set; `dco-qm3-nc-oqg` adds confidence weighting.)
- **Filter/Trunc**: DCO methods use a 90th-percentile threshold on `min_group_confidence` computed from warmup traces. Eligibility is position-dependent: once a trace's cumulative-min confidence drops below threshold at any position, it stays excluded at all subsequent positions ($\mathcal{F}_t$-measurable).
- **Stopping strategy**: a pluggable object that decides when to stop (see Section 4.2).
- **MARS / NC (Necessary Condition)**: `PerTraceQStopping` — stop when, for every challenger, the leader's margin exceeds the expected adversarial switch cost $\sum_j q_j c_j^k(\gamma)$. Includes a synthetic zero-vote challenger guard. The Hoeffding correction $\epsilon(N,\delta)$ from the safety theorem is available (`use_correction=True`) but **off** in all reported runs.
- **Gamma calibration**: the `-oqg` methods (with `--warmup-gamma`) calibrate the cost shrinkage factor $\gamma$ per-question from warmup traces, with a UCB correction for structural bias.

The canonical method list lives in `examples/run_experiment.py` (`--method` choices).

#### Switch-Probability (q) Model

| Model | Features | Notes |
|-------|----------|-------|
| Learned q (`-qm3-`) | position, confidence, flips, streak, conf_trend | 5-feature logistic regression + Platt calibration |
| Oracle q (`-oq-`) | 1.0 if intermediate != final, else 0.0 | Ground-truth switch labels — diagnostic, not $\mathcal{F}_t$-measurable |

#### Deepconf Voting Methods (offline)

| # | Method | Weight | Filter |
|---|--------|--------|--------|
| 1 | majority | uniform (1.0) | none |
| 2 | mean_conf_weighted | mean of all token confs | none |
| 3 | tail_conf_weighted | mean of last 2048 token confs | none |
| 4 | bottom_window_weighted | bottom 10% of sliding windows | none |
| 5 | min_window_weighted | min sliding window | none |
| 6 | top10_tail_filtered | tail_conf weight | keep only top 10% by tail_conf |
| 7 | top10_bottom_window_filtered | bottom_10 weight | keep only top 10% by bottom_10 |

Methods 1-5 differ only in weight scheme (all traces vote). Methods 6-7 additionally filter out the bottom 90% of traces by a confidence metric before voting.

---

## 3. Architecture Layers

```
  L0   answer_equiv    raw_traces    npz_loader    pkl_loader    probed_traces    (data)
        │                │
  L1   confidence       voting        q_model       results_io      (primitives)
        │                │                │              │
  L2   simulation ───────┴────────────────┘              │           (engine)
        │                                                │
  L3   scripts ──────────────────────────────────────────┘           (CLI wrappers)
```

**Rule**: each layer imports only from layers below. No lateral imports within a layer. No upward imports.

---

## 4. Module Specifications

### 4.0 Layer 0 — Data Models and Loaders

#### `raw_traces.py`

Defines the core data model for pre-generated traces.

```
RawTrace
  confs: np.ndarray          # per-token confidence scores
  answer: Optional[str]      # final extracted answer
  is_correct: bool           # matches ground truth
  # Properties:
  num_tokens: int            # len(confs)
  mean_confidence: float

RawQuestion
  question_id: int
  question_text: str
  ground_truth: str
  traces: List[RawTrace]
  # Properties:
  num_traces: int
  pass_at_1: float           # fraction of correct traces

RawDataset
  name: str
  model: str
  questions: List[RawQuestion]
```

Confidence computation methods (`min_group_confidence()`, `bottom_10_confidence()`, etc.) live in `confidence.py` as stateless functions operating on numpy arrays, not as instance methods on these data classes.

#### `npz_loader.py`

Loads compact NPZ files into `RawDataset`. Used for the DeepSeek-8B legacy format (per-question NPZ files with padded confidence arrays + lengths).

```
load_question_npz(npz_path) -> RawQuestion
load_npz_dataset(data_dir, question_ids=None) -> RawDataset
```

#### `pkl_loader.py`

Loads the unified pickle trace format used by Qwen3-32B. A single `.pkl` file per dataset contains all traces with per-token confidences, probed answers at 2048-multiple positions, and metadata. Produces the same `RawQuestion` + `ProbedQuestion` objects as the NPZ+JSON path.

```
load_pkl_dataset(pkl_path, name, model, question_ids=None)
    -> (RawDataset, Dict[int, ProbedQuestion])

split_pkl_to_per_question(pkl_path, cache_dir)
    -> (question_ids, ground_truth_map, trace_lengths_map)
    # Splits ~4GB pkl into per-question cache files (~120MB each)
    # for efficient worker loading. Caches metadata to avoid re-parsing.

load_pkl_question(q_pkl_path) -> (RawQuestion, ProbedQuestion)
    # Workers call this to load a single question from cache.
```

Probe positions for the pkl format use the 2048-multiple grid (shared across all alive traces at each checkpoint), excluding per-trace termination positions which are trace-specific.

#### `probed_traces.py`

Probed trace data structures and JSON loader. The data structures (`ProbedTrace`, `ProbedQuestion`) are shared by both NPZ+JSON and pkl loading paths. The JSON file loader (`load_probed_question`) is used only for the DeepSeek-8B format; `pkl_loader.py` constructs `ProbedTrace`/`ProbedQuestion` objects directly from pickle entries.

```
ProbedTrace
  trace_id: int
  final_answer: Optional[str]
  num_tokens: int
  probe_results: dict[int, str]   # position -> answer
  # Methods:
  get_answer_at_position(position: int) -> Optional[str]

ProbedQuestion
  question_id: int
  traces: List[ProbedTrace]
  probe_positions: List[int]
  # Properties:
  num_traces: int

load_probed_question(data_dir, qid) -> Optional[ProbedQuestion]
```

#### `answer_equiv.py`

Mathematical equivalence checking with consolidated grouping.

```
answers_equivalent(a, b) -> bool
    # Fast paths: exact match, case-insensitive, normalized, numeric comparison
    # Plausibility filter: _is_plausible_math() skips SymPy on garbage strings
    # Fallback: Dynasor math_equal (cached via _math_equal_cached)

canonical_numeric_key(answer) -> tuple[str, bool]

group_equivalent_answers(answers: list[str], max_symbolic_pairwise=50)
    -> tuple[dict, np.ndarray]
    """
    Group equivalent answers into integer IDs.

    Returns:
        canonical_map: {group_id: representative_string}
        answer_ids:    np.ndarray of shape [len(answers)], dtype int16

    Algorithm:
        1. Numeric fast path: canonicalize via canonical_numeric_key, hash-group   O(n)
        2. Symbolic grouping: pairwise answers_equivalent on remaining             O(m^2)
        3. Top-K cross-check (K=20): symbolic groups checked against the K
           largest existing groups to catch numeric-symbolic merges without
           O(m*n) cost on the long tail of rare/garbage answers.
    """
```

Key design choices:
- **Top-K cross-check** (K=20 largest groups) replaces checking all groups. The correct answer is almost always among the most popular groups, so this catches real merges without O(m*n) cost. Critical for Qwen3-32B data which can have 1000+ unique answers per question.
- **Plausibility pre-filter** (`_is_plausible_math`): rejects long strings with no mathematical content (digits, LaTeX operators) before expensive SymPy parsing. Checks first 50 chars so answers with correct values followed by trailing garbage still pass.

---

### 4.1 Layer 1 — Primitives

#### `confidence.py`

Stateless functions for confidence metric computation.

```python
def group_confidences(confs: np.ndarray, length: int, window_size: int) -> np.ndarray:
    """Sliding-window mean confidences for a single trace.

    Uses cumsum for O(n) computation.

    Args:
        confs:       [max_tokens] per-token confidence scores
        length:      actual number of tokens (confs beyond this are padding)
        window_size: sliding window width (default 2048)

    Returns:
        [n_windows] array of mean confidences per window
    """

def group_confidences_batch(
    confs_list: List[np.ndarray], lengths: np.ndarray, window_size: int
) -> List[np.ndarray]:
    """Batch group_confidences over multiple traces.

    Args:
        confs_list: list of per-trace confidence arrays (variable length)
        lengths:    [n_traces]

    Returns:
        list of n_traces arrays, each [n_windows_i]
    """

def min_group_confidence(confs: np.ndarray, length: int, window_size: int) -> float:
    """min(group_confidences) — single worst window."""

def bottom_10_confidence(confs: np.ndarray, length: int, window_size: int) -> float:
    """mean(bottom 10% of group_confidences). Uses np.partition for O(n)."""

def compute_confidence_metrics(
    traces_confs: List[np.ndarray], lengths: np.ndarray, window_size: int
) -> dict[str, np.ndarray]:
    """Batch-compute all metrics for a question's traces.

    Args:
        traces_confs: list of per-trace confidence arrays (variable length)
        lengths:      [n_traces]

    Returns dict with keys:
        'min_group':   [n_traces] float64
        'bottom_10':   [n_traces] float64
        'mean':        [n_traces] float64
        'tail_conf':   [n_traces] float64  — mean of last window_size tokens
        'group_confs': List[np.ndarray] — per-trace group confidence arrays
    """

def confidence_at_positions(
    confs: np.ndarray, length: int, positions: np.ndarray, window_size: int
) -> np.ndarray:
    """Compute group confidence at specific probe positions for one trace.

    For each position p, returns the mean confidence of the window
    ending at token p. If p < window_size, returns the mean of
    confs[0:p].

    Args:
        confs:      [max_tokens] per-token confidence scores
        length:     actual number of tokens
        positions:  [n_positions] int — probe positions to evaluate
        window_size: sliding window width

    Returns:
        [n_positions] float32 — confidence at each position
    """

def confidence_at_positions_batch(
    traces_confs: List[np.ndarray], lengths: np.ndarray,
    positions: np.ndarray, window_size: int
) -> np.ndarray:
    """Vectorized confidence_at_positions for all traces in a question.

    Args:
        traces_confs: list of per-trace confidence arrays (variable length)
        lengths:      [n_traces]
        positions:    [n_positions]
        window_size:  sliding window width

    Returns:
        [n_traces, n_positions] float32
    """

def find_truncation_position(
    group_confs: np.ndarray,
    num_tokens: int,
    threshold: float,
    window_size: int
) -> int:
    """Find the token position where a single trace first drops below threshold.

    Truncation threshold is computed externally (e.g. 90th percentile of
    warmup traces' min_group_confidence via np.percentile).

    Returns:
        int — truncation position for this trace (num_tokens if never drops)
    """
```

#### `voting.py`

Consolidated voting logic, stopping strategy interface, and gamma calibration.

All voting methods reduce to: eligible traces contribute weighted votes, pick the leader. All stopping strategies reduce to: given the current vote state and trace state, return a boolean per iteration.

```python
# ── Voting ──

def compute_votes(
    answer_ids: np.ndarray,
    weights: np.ndarray | None,
    eligible: np.ndarray,
    n_answers: int
) -> np.ndarray:
    """Accumulate weighted votes for a batch of iterations at one position.

    Args:
        answer_ids: [n_iter, budget] int — answer group ID per trace (-1 = no answer yet)
        weights:    [n_iter, budget] float — per-trace vote weight (None = uniform 1.0)
        eligible:   [n_iter, budget] bool — which traces are eligible to vote
        n_answers:  number of unique answer groups

    Returns:
        vote_weights: [n_iter, n_answers] float64 — accumulated vote weight per answer

    Implementation:
        Uses np.add.at for scatter-add across (iteration, answer_id) pairs.
        Fully vectorized across iterations and traces.
    """

def get_leader(vote_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract leader info from vote weight matrix.

    Args:
        vote_weights: [n_iter, n_answers]

    Returns:
        leader_ids:    [n_iter] int   — answer ID of leader
        leader_weight: [n_iter] float — leader's total weight
    """

# ── Stopping Strategy Protocol ──

@dataclass
class PositionState:
    """Everything known at a probe position, passed to stopping strategies."""
    vote_weights: np.ndarray        # [n_active, n_answers] snapshot vote weights at this position
    answer_ids: np.ndarray          # [n_active, budget] answer ID at this position
    weights: Optional[np.ndarray]   # [n_active, budget] per-trace vote weight
    eligible: np.ndarray            # [n_active, budget] which traces are eligible
    w_active: np.ndarray            # [n_active] total weight of not-yet-finished eligible traces
    w_active_by_ans: np.ndarray     # [n_active, n_answers] active weight by answer group
    pos_idx: int                    # index into probe_positions
    n_active_traces: np.ndarray     # [n_active] count of active traces per iteration

class StoppingStrategy(Protocol):
    def check(self, state: PositionState) -> np.ndarray:
        """Return [n_active] bool: can this iteration stop?"""
        ...

# ── Built-in Strategies ──

class PerTraceQStopping:
    """MARS stopping: stop when margin > sum(q_t * c_t(gamma)) [+ correction].

    The core MARS rule. Uses per-trace switch-probability estimates q_t and the
    adversarial switch cost c_t(gamma). Includes a synthetic zero-vote challenger
    guard: at early positions where all traces give the same preliminary answer,
    computes worst-case adversarial cost (no c_j=0 reductions) to prevent spurious
    early stops. The Hoeffding correction is off (use_correction=False) in all
    reported runs.
    """
    def __init__(self, q_t: np.ndarray, w_max: float, delta: float,
                 n_answers: int, use_correction: bool = True, gamma: float = 1.0,
                 warmup_ref_answer: int = -1):
        """
        Args:
            q_t:            [n_iter, budget, n_positions] float — precomputed P(switch)
            w_max:          float — maximum vote weight (for Hoeffding correction)
            delta:          float — failure probability bound (correction only)
            n_answers:      int — total number of answer groups (for union bound |A|)
            use_correction: bool — if False, skip the Hoeffding term (MARS default)
            gamma:          float — cost shrinkage factor (1.0 = full worst-case)
            warmup_ref_answer: int — guard probe-0 stops against placeholder answers
        """

    def check(self, state: PositionState) -> np.ndarray:
        """Vectorized across active iterations.

        1. Worst-case zero-vote challenger guard:
            c_worst = gamma * where(votes_for_leader, 2w, w) * eligible
            can_stop = leader_weight > sum(q * c_worst) [+ correction]
        2. Per-challenger refinement (tighter for challengers with actual votes):
            c_t includes k-voter departure benefit (c_j = -w_j, unchanged by gamma)
            can_stop &= M_a > sum(q * c_t) [+ correction]  for each challenger a
        """

class ConsensusStopping:
    """Consensus: stop when leader has >= tau fraction of total weight."""
    def __init__(self, tau: float = 0.95):
        self.tau = tau

    def check(self, state: PositionState) -> np.ndarray:
        """leader_weight / vote_weights.sum() >= tau"""

class ParallelProbeStopping:
    """Parallel-Probe baseline (Zheng et al. 2026): consensus stability + pruning.

    Halts when the majority answer is unchanged for `conv` consecutive probes,
    and prunes traces that disagree with consensus for `prune_patience` probes
    (after `warmup` probes). Stateful — instantiate fresh per question.
    """
    def __init__(self, n_iter, budget, n_answers, conv=3, warmup=4, prune_patience=2):
        ...
    def check(self, state: PositionState) -> np.ndarray:
        ...

class CompositeStopping:
    """Combine multiple strategies with OR logic: stop if ANY strategy says stop."""
    def __init__(self, strategies: list[StoppingStrategy]):
        self.strategies = strategies

    def check(self, state: PositionState) -> np.ndarray:
        """can_stop = strategy_1.check() | strategy_2.check() | ..."""

class NeverStop:
    """Run all positions (for offline methods)."""
    def check(self, state: PositionState) -> np.ndarray:
        """return np.zeros(n_active, dtype=bool)"""

# ── Warmup Gamma Calibration ──

def calibrate_gamma_warmup(
    ans_ids_at_pos, q_at_pos, weights_at_pos, eligible_at_pos,
    n_answers, gamma_grid=None, ucb_z=0.0,
) -> float:
    """Calibrate gamma per-question by sweeping on warmup traces.

    Walks down from gamma=1.0, finds the smallest gamma in the contiguous
    band that preserves the reference answer, then applies UCB correction
    to compensate for structural bias (warmup has fewer challengers than
    bootstrap).

    UCB correction: gamma_out = min(1, gamma_band + z / sqrt(n_elig))
    """
```

#### `q_model.py`

Per-trace switch probability model: a 5-feature logistic regression with Platt calibration, plus oracle q values for diagnostics.

```python
# ── Feature Computation (all precomputable, all trace-intrinsic) ──

def compute_flips(ans_ids_at_pos) -> np.ndarray:
    """Cumulative answer-change count. [n_traces, n_positions] int16"""

def compute_streaks(ans_ids_at_pos) -> np.ndarray:
    """Consecutive same-answer count. [n_traces, n_positions] int16"""

# ── Feature Matrix ──

def build_feature_matrix(positions, conf_at_pos, flips, streaks) -> np.ndarray:
    """Base 4 features [position, confidence, flips, streak]"""

def build_feature_matrix_v3(positions, conf_at_pos, flips, streaks) -> np.ndarray:
    """5 features [position, confidence, flips, streak, conf_trend]
    (the MARS switch-probability feature set)."""

# ── Training ──

def build_training_labels(ans_ids_at_pos, final_answer_ids) -> np.ndarray:
    """Binary switch labels: 1 if intermediate != final. [n_warmup * n_positions] int8"""

@dataclass
class FittedQModel:
    """Fitted logistic regression for q_j(t). Dimension-agnostic.
    Stores coefficients + standardization params for vectorized prediction.
    """
    coefficients: np.ndarray    # [D+1] float64 — [intercept, beta_1..D]
    feature_means: np.ndarray   # [D] float64
    feature_stds: np.ndarray    # [D] float64

    def predict(self, X) -> np.ndarray:
        """z = intercept + standardize(X) @ betas; q = sigmoid(z)"""

def fit_q_model(X, y) -> FittedQModel:
    """L-BFGS-B logistic regression with mild L2 regularization.
    Dimension-agnostic (D inferred from X.shape[1]).
    Training is NOT in the hot path (~20ms per question)."""

# ── Platt Calibration ──

@dataclass
class PlattCalibrator:
    """Platt scaling: calibrated_q = sigmoid(a * logit(q_raw) + b).
    Near-identity when model is well-specified (a~1, b~0)."""
    a: float
    b: float
    def calibrate(self, q) -> np.ndarray: ...

def fit_platt_calibration(q_pred, y_true) -> PlattCalibrator:
    """Fit on warmup predictions vs true labels."""

# ── Oracle q (diagnostic) ──

def compute_oracle_q_values(ans_ids_at_pos, final_answer_ids) -> np.ndarray:
    """1.0 where trace will switch, 0.0 otherwise.
    NOT F_t-measurable — uses ground-truth future information."""

# ── Precomputing q_t for simulation ──

def precompute_q_values_v3(model, calibrator, conf_at_pos, flips, streaks,
                            positions) -> np.ndarray:
    """5 features + optional Platt calibration → [n_traces, n_positions] float32."""
```

#### `results_io.py`

Standardized experiment I/O.

```python
@dataclass
class ExperimentConfig:
    method: str
    dataset: str
    budget: int
    n_iterations: int
    warmup: int
    window_size: int
    seed: int
    weighting: str           # "uniform" or "confidence"
    truncation: bool
    stopping: str            # "alpha_margin", "per_trace_q", "none"
    extra: dict              # method-specific params (alpha, delta, etc.)

def save_experiment(
    config: ExperimentConfig,
    per_iteration_rows: list[dict],
    per_question_summary: dict,
    overall_summary: dict,
    output_dir: Path
) -> Path:
    """Save experiment results to standardized directory structure.

    Creates:
        {output_dir}/
            config.json
            results.csv               # per-iteration rows
            summary_per_question.csv  # aggregated per question
            summary_overall.csv       # single-row overall summary

    Returns:
        Path to created directory
    """

def load_experiment(exp_dir: Path) -> tuple[ExperimentConfig, pd.DataFrame]:
    """Load a saved experiment."""

def aggregate_per_question(
    results_df: pd.DataFrame,
    baseline_tokens_per_question: Optional[Dict[int, int]] = None
) -> pd.DataFrame:
    """Aggregate per-iteration results to per-question summary.

    Computes: accuracy, accuracy_std, n_correct, n_iterations,
              total_tokens_mean, total_tokens_sum, token_savings_pct,
              mean_position, stopped_by_{margin,consensus,budget}
    """

def aggregate_overall(per_question_df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate per-question summary to overall summary dict."""

def generate_output_dir(
    base_dir: Path, dataset: str, method: str, **params
) -> Path:
    """Generate timestamped output directory name.

    Pattern: {base_dir}/{dataset}/{method}_{param_str}_{timestamp}/
    """
```

---

### 4.2 Layer 2 — Simulation Engine

#### `simulation.py`

The unified simulation loop. Every method is a call to `simulate_with_stopping()` with different inputs and a different stopping strategy.

```python
@dataclass
class SimulationResult:
    """Result for one question across all iterations."""
    answers: np.ndarray          # [n_iter] int — winning answer ID per iteration
    is_correct: np.ndarray       # [n_iter] bool
    total_tokens: np.ndarray     # [n_iter] int64
    stop_positions: np.ndarray   # [n_iter] int — probe position where stopped
    stopped_by: np.ndarray       # [n_iter] object — 'margin', 'consensus', 'budget'
    canonical_map: dict          # answer_id -> string

def simulate_with_stopping(
    sampled_ans_ids: np.ndarray,
    sampled_tokens: np.ndarray,
    probe_positions: np.ndarray,
    canonical_map: dict,
    n_answers: int,
    stopping_strategies: List[Tuple[str, StoppingStrategy]],
    weights: Optional[np.ndarray] = None,
    filter_mask: Optional[np.ndarray] = None,
    truncation_positions: Optional[np.ndarray] = None,
) -> SimulationResult:
    """Unified simulation loop: vote at each position, stop when strategy fires.

    At each probe position, votes are computed as a SNAPSHOT (not accumulated).
    Each trace contributes its answer at the current position, weighted by its
    confidence. Strategies are checked in order — first match wins for the label.

    Args:
        sampled_ans_ids:       [n_iter, budget, n_pos] int16
        sampled_tokens:        [n_iter, budget] int64
        probe_positions:       [n_pos] int
        canonical_map:         {int -> str}
        n_answers:             int
        stopping_strategies:   List of (label, strategy) — checked in order
        weights:               [n_iter, budget] float | None
        filter_mask:           [n_iter, budget] or [n_iter, budget, n_pos] bool | None
                               2D = same filter at every position; 3D = position-specific
        truncation_positions:  [n_iter, budget] int64 | None — per-trace truncation pos

    Returns:
        SimulationResult

    The loop:
        still_running = np.ones(n_iter, dtype=bool)

        for pos_idx, pos in enumerate(probe_positions):
            active = still_running
            n_active = active.sum()
            if n_active == 0: break

            # ── VECTORIZED across active iterations ──
            ans  = sampled_ans_ids[active, :, pos_idx]
            elig = (ans >= 0)
            if filter_mask is not None:
                elig &= filter_mask[active] or filter_mask[active, :, pos_idx]
            if truncation_positions is not None:
                elig &= ~past_truncation_mask

            # Compute SNAPSHOT votes at this position (NOT accumulated)
            votes = compute_votes(ans, weights[active] if weights else None, elig, n_answers)

            # Build position state
            state = PositionState(
                vote_weights    = votes,
                answer_ids      = ans,
                weights         = weights[active] if weights else None,
                eligible        = elig,
                w_active        = ...,
                w_active_by_ans = ...,
                pos_idx         = pos_idx,
                n_active_traces = ...,
            )

            # Check each stopping strategy in order (first match wins for label)
            for label, strategy in stopping_strategies:
                can_stop_this = strategy.check(state)
                # newly_stopped get this label
            # ── end vectorized block ──

            record results for stopped iterations
            still_running[stopped] = False

        record budget-stop for any still_running
    """
```

#### Precomputation helper

```python
@dataclass
class PrecomputedQuestion:
    """All precomputed data for one question, ready for simulation."""
    question_id: int
    ground_truth: str
    n_traces: int

    # From answer_equiv
    canonical_map: Dict[int, str]   # {group_id: str}
    answer_ids: np.ndarray          # [n_traces] int16  (final answer group)
    n_answers: int

    # From confidence
    num_tokens: np.ndarray          # [n_traces] int64
    min_group: np.ndarray           # [n_traces] float64
    bottom_10: np.ndarray           # [n_traces] float64
    mean_conf: np.ndarray           # [n_traces] float64
    tail_conf: np.ndarray           # [n_traces] float64
    group_confs: List[np.ndarray]   # per-trace group confidence arrays
    conf_at_pos: np.ndarray         # [n_traces, n_positions] float32

    # From probed_traces — answers at each probe position
    ans_ids_at_pos: np.ndarray      # [n_traces, n_positions] int16
    probe_positions: np.ndarray     # [n_positions] int

    # Per-trace q-model features (always precomputed)
    flips: np.ndarray               # [n_traces, n_positions] int16
    streaks: np.ndarray             # [n_traces, n_positions] int16

def precompute_question(
    raw_q: RawQuestion,
    probed_q: ProbedQuestion,
    window_size: int = 2048
) -> PrecomputedQuestion:
    """Precompute all derived data for a question.

    Called once per question per worker process.
    All downstream simulation uses only numpy arrays from this object.

    Important: collects ALL answers (final + probed intermediate) before
    grouping, ensuring probed answers that only appear at intermediate
    positions are properly grouped.

    Computes:
        1. Answer grouping (answer_equiv.group_equivalent_answers) — uses
           all final + probed intermediate answers
        2. Confidence metrics (confidence.compute_confidence_metrics)
        3. Per-position confidence (confidence.confidence_at_positions_batch)
        4. Answer IDs at each probe position (from probed_traces)
        5. Per-trace features: flips and streaks (q_model.compute_flips/streaks)

    Steps 1-5 are all precomputed once. The q model features (step 5) are
    always computed regardless of stopping strategy — the cost is negligible
    and it avoids conditional logic.
    """

def sample_bootstrap(
    precomputed: PrecomputedQuestion,
    n_iterations: int,
    budget: int,
    seed: int,
    probed_trace_ids: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Generate bootstrap samples as numpy arrays.

    Args:
        probed_trace_ids: if provided, sample only from these trace IDs
            (avoids information leakage from unprobed traces, which would
            fall back to their final answer at every position)

    Returns dict with:
        'indices':           [n_iter, budget] int64    (trace indices)
        'sampled_ans_ids':   [n_iter, budget, n_pos] int16
        'sampled_weights':   [n_iter, budget] float64  (bottom_10 confidence)
        'sampled_tokens':    [n_iter, budget] int64
        'sampled_min_confs': [n_iter, budget] float64  (min_group_confidence)
        'sampled_conf_at_pos': [n_iter, budget, n_pos] float32
        'sampled_flips':     [n_iter, budget, n_pos] int16
        'sampled_streaks':   [n_iter, budget, n_pos] int16
        'sampled_final_ans': [n_iter, budget] int16    (final answer IDs)
        'sampled_bottom_10': [n_iter, budget] float64  (bottom_10 confidence)
        'sampled_mean_confs': [n_iter, budget] float64 (mean token confidence)
        'sampled_tail_confs': [n_iter, budget] float64 (tail window confidence)

    Implementation:
        Per iteration: rng = default_rng(seed + i); indices = rng.choice(...)
        Then fancy-index into precomputed arrays. Single numpy op per array.
    """

# ── Deepconf Multi-Voting ──

def compute_deepconf_votes(
    final_ans: np.ndarray,
    eligibility: np.ndarray,
    mean_conf: np.ndarray,
    tail_conf: np.ndarray,
    bottom_10: np.ndarray,
    min_group: np.ndarray,
    n_answers: int,
) -> Dict[str, np.ndarray]:
    """Compute all 7 deepconf voting methods on final answers.

    Returns dict of {method_name: [n_iter] int answer_ids}.
    Methods 1-5 use different weight schemes on all eligible traces.
    Methods 6-7 additionally filter to top 10% by confidence before voting.
    Called by worker_process_question() for the 'offline' method only.
    """

# ── Precomputation Cache ──

def precompute_question_cached(
    raw_q, probed_q, data_path: Path, window_size: int = 2048
) -> PrecomputedQuestion:
    """Load from cache if available, otherwise compute and save.

    Cache path: {data_path}/precomputed/w{window_size}/q{qid:02d}.npz
    Stores all PrecomputedQuestion arrays in compressed NPZ format.
    Variable-length group_confs are stored as concatenated array + lengths.

    Handles backward compatibility: if loaded cache lacks mean_conf/tail_conf,
    recomputes from scratch and re-saves.
    """
```

---

### 4.3 Layer 3 — Single Unified Script

A single `examples/run_experiment.py` handles all methods via `--method` and `--model` flags. Method-specific logic (weights, filtering, stopping strategy) is configured inside `worker_process_question()`.

#### `examples/run_experiment.py`

```python
MODELS = {
    "deepseek-8b": "DeepSeek-8B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-next": "Qwen3-next",
}

PKL_DATASETS = {
    "deepseek-8b": {
        "aime-2025": "./data/DeepSeek-8B/aime25_deepseek.pkl",
        "hmmt":      "./data/DeepSeek-8B/hmmt25_deepseek.pkl",
        "brumo-2025":"./data/DeepSeek-8B/brumo25_deepseek.pkl",
        "aime-2024": "./data/DeepSeek-8B/aime24_deepseek.pkl",
    },
    "qwen3-32b": { "...": "./data/Qwen3-32B/<dataset>.pkl" },
    "qwen3-next": { "...": "./data/Qwen3-next/<dataset>_thinking.pkl" },
}

def parse_args():
    # --model {deepseek-8b, qwen3-32b, qwen3-next}
    # --dataset {aime-2025, hmmt, brumo-2025, aime-2024}
    # --method {offline, dco, sc-qm3-nc, dco-qm3-nc,
    #           sc-qm3-nc-oqg, dco-qm3-nc-oqg, sc-oq-nc, dco-oq-nc,
    #           oracle, oracle-dco, sc-pp}
    # --delta, --budget, --iterations
    # --warmup, --window, --seed, --workers, --output-dir
    # --gamma, --warmup-gamma, --ucb-z, --gamma-min
    # --pp-conv, --pp-warmup, --pp-prune-patience

def main():
    args = parse_args()

    # Dual-format data resolution:
    #   1. If --model is in PKL_DATASETS and dataset matches -> pkl format
    #   2. Else if --model/--dataset is in DATASET_PATHS -> NPZ+JSON format
    #   3. Else error
    # For pkl: split_pkl_to_per_question() creates per-question cache files
    # For NPZ: discover questions from raw_traces_compact/*.npz

    method_config = {
        'method': args.method, 'budget': args.budget,
        'n_iterations': args.iterations, 'warmup': args.warmup,
        'window_size': args.window, 'seed': args.seed,
        'tau': args.tau, 'alpha': args.alpha,
        'delta': args.delta, 'k_conservative': args.k_conservative,
        'gamma': args.gamma, 'warmup_gamma': args.warmup_gamma,
        'ucb_z': args.ucb_z,
        'data_format': 'pkl' if use_pkl else 'npz',
        'pkl_cache_dir': str(q_cache_dir) if use_pkl else None,
    }

    per_question_results = simulate_dataset_parallel(
        data_path, question_ids, method_config, n_workers=args.workers,
    )

    # Build DataFrame, aggregate, print summary, save results
    save_experiment(config, per_iteration_rows, per_q_summary, overall, output_dir)
```

Method dispatch happens inside `worker_process_question()` in `simulation.py`. The worker first loads data (format-aware: pkl or NPZ+JSON), then configures weights, filtering, and stopping:

```python
# ── Data loading inside worker ──
if data_format == 'pkl':
    raw_q, probed_q = load_pkl_question(q_cache_dir / f"q{qid:02d}.pkl")
else:
    raw_q = load_question_npz(data_path / "raw_traces_compact" / f"q{qid:02d}.npz")
    probed_q = load_probed_question(data_path, qid)

# ── Method configuration ──
#
# Weighting (dc*/dco* prefix):
if method matches 'dc*' or 'dco*':
    weights = samples['sampled_conf_at_pos']  # [n_iter, budget, n_pos] position-dependent

# Filtering (dco* prefix):
if method matches 'dco*':
    # 90th pctl threshold from warmup min_group_confidence
    # Dynamic eligibility: cumulative-min ensures permanent truncation
    filter_mask = cum_min_conf >= thresholds  # 3D [n_iter, budget, n_pos]

# Stopping strategy (by method suffix):
#   *-qm3-nc / *-oq-nc -> PerTraceQStopping (MARS, use_correction=False)
#   sc-pp              -> ParallelProbeStopping
#   offline/dco        -> NeverStop (+ 7 deepconf voting methods for offline)

# Q model selection (by method infix):
#   *-qm3-* -> learned 5-feature logistic q + Platt calibration
#   *-oq-*  -> oracle q (ground-truth switch labels; diagnostic)

# Gamma calibration (if --warmup-gamma):
#   calibrate_gamma_warmup() on warmup traces, passed to PerTraceQStopping
```

---

## 5. Computation Model

### 5.1 Axes of Computation

| Axis | Size | Strategy | Rationale |
|------|------|----------|-----------|
| Questions | 30 | **multiprocess** | Fully independent; avoids GIL from sympy |
| Iterations | 64 | **numpy vectorized** | Independent given same question; batch ops |
| Traces (budget) | 512 | **numpy vectorized** | Independent within iteration; contiguous memory |
| Positions | ~128 | **sequential** | Stopping at pos[i] decides if pos[i+1] runs |
| Answer grouping | ~50 unique | **precomputed once** | O(m^2) symbolic equiv amortized |
| q_t values | 512 x ~128 | **precomputed once** | Logistic prediction is pure matrix multiply |

### 5.2 Phase-by-Phase Execution

```
Phase 0: PRECOMPUTATION (per question, done once, cached to disk as NPZ)
    group_confidences      — cumsum-based, vectorized over 512 traces
    bottom_10_confidence   — np.partition + np.mean, vectorized
    conf_at_pos            — cumsum-based, vectorized over 512 traces x ~128 positions
    group_answers          — numeric O(n) + symbolic O(m^2), done once
    ans_ids_at_pos         — precompute answer IDs at every probe position
    flips, streaks         — single pass over ans_ids_at_pos, vectorized over 512 traces
    Cost: ~54s per question (cold), ~4s per question (from NPZ cache).
    Amortized across all 64 iterations and all subsequent runs.

Phase 1: Q MODEL TRAINING (per question, only for NC/CLT/WD stopping methods)
    build_feature_matrix   — assemble [n_warmup * n_pos, D] matrix (D=4/5/6 by variant)
    build_training_labels  — compare ans_ids_at_pos to final answer, vectorized
    fit_q_model            — logistic regression on ~160 observations, D features
    [optional] fit_platt_calibration — on warmup predictions vs labels (QM2/QM3 only)
    precompute_q_values_v3 — predict on [512, ~128] = ~65K samples, pure matrix multiply
    [optional] calibrate_gamma_warmup — sweep gamma grid on warmup (if --warmup-gamma)
    For oracle-q methods: compute_oracle_q_values (no training needed)
    Cost: ~20ms per question (dominated by logistic fit, not in hot path).
    Skipped entirely for alpha-margin and offline methods.

Phase 2: BOOTSTRAP SAMPLING (per question, single numpy call)
    rng.choice(n_traces, size=(n_iter, budget)) -> [64, 512] index matrix
    Fancy-index into precomputed arrays -> sampled_ans_ids, sampled_weights, etc.
    For q model: sampled_q = q_values[indices]  (one additional fancy-index)
    Cost: ~1ms per question.

Phase 3: SIMULATION LOOP (per question, sequential over positions)
    for each position (sequential):
        compute SNAPSHOT votes — np.add.at, vectorized over [n_active, budget]
            (votes are fresh at each position, NOT accumulated)
        build PositionState
        check each labeled strategy in order — first match wins for label
            alpha-margin:  boolean comparison, ~0.1ms
            NC (per-trace q): zero-vote guard + per-challenger sum(q*c), ~0.3ms
            CLT:           E[swing] + z*sqrt(Var[swing]), ~0.4ms
        mask off stopped   — index update
    Early exit when all iterations stopped.
    Cost per position step: ~0.5ms (alpha) or ~0.8ms (NC/CLT).
    Worst case: ~128 steps = 60-100ms per question.

Phase 4: AGGREGATION (main process, after all workers return)
    aggregate_per_question  — groupby + mean/std, pure numpy
    aggregate_overall       — mean across questions
    save to CSV + JSON
    Cost: ~1ms total.
```

### 5.3 Parallelization Strategy

```
ProcessPoolExecutor(max_workers=N)

Question 0  -->  Worker 0: load from disk -> precompute -> [fit q] -> sample -> simulate
Question 1  -->  Worker 1: load from disk -> precompute -> [fit q] -> sample -> simulate
...
Question 29 -->  Worker K: load from disk -> precompute -> [fit q] -> sample -> simulate
```

**Why multiprocess** (not multithread):
- numpy releases GIL for compute, but `answer_equiv` uses sympy (GIL-bound)
- Each worker loads data from disk independently

**Why load from disk per worker** (not serialize):
- `confs[512, 65536]` float32 = 128MB per question
- NPZ load: ~50ms. Pickle serialization: ~500ms. 10x faster to re-read.

**Dual-format data loading**:
- **NPZ format** (DeepSeek-8B): worker loads `q{qid}.npz` from `raw_traces_compact/` + per-trace JSON from `probed_traces/`. Each question is already a separate file.
- **Pickle format** (Qwen3-32B): the full dataset pkl (~4GB) is split into per-question cache files (`~120MB each`) by `split_pkl_to_per_question()` at startup (main process). Workers then load individual `q{qid}.pkl` files from cache. This avoids each worker loading the full 4GB file.

**What gets returned** (lightweight):
- 5 arrays of shape `[64]` per question (~2KB). Negligible serialization cost.

### 5.4 Memory Layout

```
Per question, per worker:
    sampled_ans_ids   [64, 512, ~128]  int16    ~  8 MB
    filter_mask       [64, 512]        bool     ~32 KB    (2D, or None)
    sampled_weights   [64, 512]        float64  ~256 KB
    sampled_tokens    [64, 512]        int64    ~256 KB
    vote_weights      [64, ~50]        float64  ~ 25 KB   (snapshot per position)
    ──────────────────────────────────────────────────────
    Shared across methods:                       ~ 9 MB

    Per-trace q adds (only when using PerTraceQStopping):
    sampled_q         [64, 512, ~128]  float32  ~  8 MB
    sampled_flips     [64, 512, ~128]  int16    ~  8 MB  (only during precompute)
    sampled_streaks   [64, 512, ~128]  int16    ~  8 MB  (only during precompute)
    ──────────────────────────────────────────────────────
    At simulation time: sampled_q is the only addition.
    flips/streaks are consumed during precompute_q_values_v3 and not retained.
    Total with q model: ~20 MB per worker. Still fits in L3 cache.
```

### 5.5 The Position Loop — Why Sequential is OK

The loop over probe positions is the only sequential part. However:

1. **Each step is fully vectorized** across all 64 iterations simultaneously
2. **Early exit** when all iterations have stopped (easy questions: 1-2 steps)
3. **Shrinking active set**: iterations that stop are masked out, reducing work
4. **Worst case** is ~128 steps x ~0.8ms = ~100ms per question, which is fast

The sequential constraint is fundamental: we cannot know whether to stop at position `p` without computing the votes at position `p`, and earlier positions may have already stopped some iterations (shrinking the active set).

### 5.6 Per-Trace Q: Why It Stays Vectorized

A potential concern is that the per-trace q stopping check is more complex than alpha-margin. Here's why it remains efficient:

**All four q model features are precomputed.** Position, confidence, flips, and streak are all trace-intrinsic (see `per_trace_q_estimation.md` Section 2.4). They don't depend on the ensemble state. So `q_t[trace, position]` is computed once during Phase 1 and indexed during Phase 3.

**The stopping check at each position:**
```
For each challenger a:
    c_t[i,j] = adversarial_cost(ans[i,j], leader[i], a)    # [n_active, budget]
    threshold[i] = (sampled_q[i,:,pos] * c_t[i,:]).sum(1)  # [n_active]
    threshold[i] += w_max * sqrt(2 * N[i] * log(|A|/delta))
    safe[i] = margin[i] > threshold[i]                      # [n_active]
can_stop = all challengers safe
```

This is a dot product (`q * c`) summed over traces, then a scalar comparison — fully vectorized across iterations. The additional cost vs alpha-margin is one `[n_active, budget]` elementwise multiply and sum per challenger, roughly 2x the work per position step. With ~50 unique answers (challengers), this is ~50 dot products of size 512, which numpy handles in microseconds.

---

## 6. Module Summary

| File | Purpose |
|------|---------|
| `mars/raw_traces.py` | Data models: RawTrace, RawQuestion, RawDataset |
| `mars/npz_loader.py` | Legacy NPZ loading |
| `mars/pkl_loader.py` | Unified pickle loading + per-question splitting |
| `mars/probed_traces.py` | Probed trace data structures + loader |
| `mars/answer_equiv.py` | Equivalence checking + consolidated grouping (top-K cross-check) |
| `mars/confidence.py` | Confidence metrics + per-position confidence |
| `mars/voting.py` | Voting + stopping strategies (NC/MARS, consensus, Parallel-Probe) + gamma calibration |
| `mars/q_model.py` | Per-trace switch-probability model (5 features) + Platt calibration + oracle q |
| `mars/simulation.py` | Unified simulation loop + precomputation + cache + parallel dispatch |
| `mars/results_io.py` | Standardized experiment I/O |
| `examples/run_experiment.py` | Unified CLI entry point with multi-model/dataset support |

---

## 7. Extending: Adding a New Method

The stopping strategy is orthogonal to weights and filtering. A new method is a
new branch in `worker_process_question()` that sets `weights`/`filter_mask` and
chooses a `stopping_strategies` list.

### Example: a new filtering method (top-10% filtering, no truncation)

```python
# In worker_process_question(), add a new method branch:
if method == 'dc-filtered':
    weights = samples['sampled_min_confs']
    thresh = np.percentile(samples['sampled_min_confs'][:, :warmup_n], 90, axis=1)
    filter_mask = samples['sampled_min_confs'] >= thresh[:, np.newaxis]  # 2D
    # No truncation_positions — filter only, no truncation
    stopping_strategies = [("none", NeverStop())]
```

### Example: a custom stopping rule

Implement the `StoppingStrategy` protocol — a `check(state) -> bool[n_active]`
method — and plug it into `stopping_strategies`. To use MARS stopping with a
custom q estimate:

```python
model = fit_q_model(warmup_X, warmup_y)
q_values = precompute_q_values_v3(model, calibrator, precomp.conf_at_pos,
                                  precomp.flips, precomp.streaks,
                                  precomp.probe_positions)
sampled_q = q_values[samples['indices']]

stopping_strategies = [
    ("margin", PerTraceQStopping(q_t=sampled_q, w_max=1.0,
                                 delta=0.05, n_answers=precomp.n_answers,
                                 use_correction=False, gamma=gamma)),
]

# Everything else (weights, filter_mask, simulate_with_stopping call) stays identical.
```

No changes to the simulation loop, voting logic, or result I/O. The stopping
strategy is the only thing that changes.

# Appendix
## Design Flow Chart
```
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                                DATA LAYER (L0)                                  │
  │                                                                                 │
  │   On Disk (two formats)                                                         │
  │   ─────────────────────                                                         │
  │   NPZ+JSON (DeepSeek-8B):                                                      │
  │     data/DeepSeek-8B/{dataset}-offline/raw_traces_compact/q*.npz                │
  │         confs[4096,max_tok]  lengths[4096]  answers[4096]                       │
  │     data/DeepSeek-8B/{dataset}-offline/probed_traces/{run_dir}/                 │
  │         {trace_id}.json → position → answer  (1024 probed traces)              │
  │                                                                                 │
  │   Unified Pickle (Qwen3-32B):                                                  │
  │     data/Qwen3-32B/{dataset}.pkl                                                │
  │         (qid, trace_idx) → {confs, probes, extracted_answer, ground_truth}     │
  │     Split at startup → .cache_{stem}/q{qid}.pkl (~120MB each)                  │
  │                                                                                 │
  │   Loaders                                                                       │
  │   ───────                                                                       │
  │   npz_loader.py ──────▶ RawDataset ─▶ RawQuestion ─▶ RawTrace  (DeepSeek-8B)   │
  │   probed_traces.py ───▶ ProbedQuestion ─▶ ProbedTrace           (DeepSeek-8B)   │
  │   pkl_loader.py ──────▶ (RawQuestion, ProbedQuestion)           (Qwen3-32B)     │
  │                                                                                 │
  │   answer_equiv.py                                                               │
  │   ───────────────                                                               │
  │   answers_equivalent(a, b) → bool                                               │
  │     └─ plausibility pre-filter + cached math_equal                              │
  │   group_equivalent_answers(answers) → (canonical_map, answer_ids[n_traces])     │
  │     └─ numeric fast path + symbolic O(m^2) + top-K cross-check (K=20)           │
  └───────────────────────────────────┬─────────────────────────────────────────────┘
                                      │
                      ┌───────────────┼───────────────────────────┐
                      ▼               ▼                           ▼
  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────────┐
  │  confidence.py (L1)  │ │   q_model.py (L1)    │ │     results_io.py (L1)       │
  │                      │ │                      │ │                              │
  │ group_confidences()  │ │ compute_flips()      │ │ save_experiment()            │
  │ group_confs_batch()  │ │ compute_streaks()    │ │ load_experiment()            │
  │ min_group_conf()     │ │ build_feature_matrix │ │ aggregate_per_question()     │
  │ bottom_10_conf()     │ │ build_feature_mtx_v3 │ │ aggregate_overall()          │
  │ compute_conf_metrics │ │ fit_q_model()        │ │ generate_output_dir()        │
  │ conf_at_positions()  │ │ precompute_q_vals_v3 │ │                              │
  │ conf_at_pos_batch()  │ │ fit_platt_calib()    │ │                              │
  │ find_trunc_position  │ │ compute_oracle_q()   │ │ ExperimentConfig:            │
  │                      │ │ PlattCalibrator      │ │   method, dataset, budget,   │
  │                      │ │ FittedQModel         │ │   stopping, weighting, ...   │
  └──────────┬───────────┘ └──────────┬───────────┘ └──────────────┬───────────────┘
             │                        │                            │
             ▼                        ▼                            │
  ┌────────────────────────────────────────────────────────┐       │
  │                    voting.py (L1)                      │       │
  │                                                        │       │
  │  Voting                                                │       │
  │  ──────                                                │       │
  │  compute_votes(ans_ids, weights, eligible, n_ans)      │       │
  │    └─ np.add.at scatter-add, vectorized [n_iter, K]    │       │
  │  get_leader(vote_weights)                              │       │
  │    └─ returns (leader_id, leader_weight)               │       │
  │                                                        │       │
  │  Stopping Strategy Protocol                            │       │
  │  ──────────────────────────                            │       │
  │  StoppingStrategy.check(PositionState) → bool[n_active] │       │
  │                                                        │       │
  │  ┌──────────────────────────────┐ ┌────────────────────┐│       │
  │  │ PerTraceQStopping (MARS)     │ │ ParallelProbeStopping ││     │
  │  │                              │ │                    │ │       │
  │  │ 1. Zero-vote guard:          │ │ consensus stable   │ │       │
  │  │    leader > sum(q*c_worst)   │ │ for `conv` probes  │ │       │
  │  │ 2. Per-challenger:           │ │ + deviation        │ │       │
  │  │    M_a > sum(q*c_t(gamma))   │ │   pruning          │ │       │
  │  │ gamma shrinks L/other costs  │ │ (baseline)         │ │       │
  │  └──────────────────────────────┘ └────────────────────┘│       │
  │                                                        │       │
  │  ┌──────────────────┐ ┌──────────────┐ ┌────────────┐  │       │
  │  │ ConsensusStopping│ │CompositeStop │ │ NeverStop  │  │       │
  │  │ leader/total>=tau│ │ s1|s2|...(OR)│ │ False[:]   │  │       │
  │  └──────────────────┘ └──────────────┘ └────────────┘  │       │
  │                                                        │       │
  │  calibrate_gamma_warmup() — per-question gamma + UCB   │       │
  └──────────────────────────┬─────────────────────────────┘       │
                             │                                     │
                             ▼                                     │
  ┌────────────────────────────────────────────────────────────────┼─────────────┐
  │                      simulation.py (L2)                        │             │
  │                                                                │             │
  │  ┌──────────────────────────────────────────────────────────┐  │             │
  │  │  PrecomputedQuestion                                     │  │             │
  │  │                                                          │  │             │
  │  │  # From answer_equiv                                     │  │             │
  │  │  canonical_map     {group_id: str}                       │  │             │
  │  │  answer_ids        [n_traces] int16                      │  │             │
  │  │  n_answers         int                                   │  │             │
  │  │                                                          │  │             │
  │  │  # From confidence                                       │  │             │
  │  │  num_tokens        [n_traces] int64                      │  │             │
  │  │  min_group         [n_traces] float64                    │  │             │
  │  │  bottom_10         [n_traces] float64                    │  │             │
  │  │  mean_conf         [n_traces] float64                    │  │             │
  │  │  tail_conf         [n_traces] float64                    │  │             │
  │  │  group_confs       list of [n_windows_i] per trace       │  │             │
  │  │  conf_at_pos       [n_traces, n_pos] float32             │  │             │
  │  │                                                          │  │             │
  │  │  # From probed_traces                                    │  │             │
  │  │  ans_ids_at_pos    [n_traces, n_pos] int16               │  │             │
  │  │  probe_positions   [n_pos] int                           │  │             │
  │  │                                                          │  │             │
  │  │  # From q_model                                          │  │             │
  │  │  flips             [n_traces, n_pos] int16               │  │             │
  │  │  streaks           [n_traces, n_pos] int16               │  │             │
  │  └──────────────────────────────────────────────────────────┘  │             │
  │                           │                                    │             │
  │                           ▼                                    │             │
  │  ┌──────────────────────────────────────────────────────────┐  │             │
  │  │  sample_bootstrap(precomputed, n_iter, budget, seed,     │  │             │
  │  │                   probed_trace_ids)                      │  │             │
  │  │                                                          │  │             │
  │  │  Per iteration: rng(seed+i).choice(probed_ids, 512)     │  │             │
  │  │    └─ samples from probed traces only (avoid leakage)    │  │             │
  │  │                                                          │  │             │
  │  │  Fancy-index into precomputed:                           │  │             │
  │  │    indices          [64, 512]       int64                │  │             │
  │  │    sampled_ans_ids  [64, 512, ~128] int16                │  │             │
  │  │    sampled_weights  [64, 512]       float64 (bottom_10)  │  │             │
  │  │    sampled_tokens   [64, 512]       int64                │  │             │
  │  │    sampled_min_confs[64, 512]       float64              │  │             │
  │  │    sampled_conf_at_pos [64, 512, ~128] float32           │  │             │
  │  │    sampled_flips    [64, 512, ~128] int16                │  │             │
  │  │    sampled_streaks  [64, 512, ~128] int16                │  │             │
  │  │    sampled_final_ans[64, 512]       int16                │  │             │
  │  │    sampled_bottom_10[64, 512]       float64              │  │             │
  │  │    sampled_mean_confs[64, 512]      float64              │  │             │
  │  │    sampled_tail_confs[64, 512]      float64              │  │             │
  │  └──────────────────────────────────────────────────────────┘  │             │
  │                           │                                    │             │
  │                           ▼                                    │             │
  │  ┌──────────────────────────────────────────────────────────┐  │             │
  │  │  simulate_with_stopping(                                 │  │             │
  │  │      sampled_ans_ids, sampled_tokens, probe_positions,   │  │             │
  │  │      canonical_map, n_answers,                           │  │             │
  │  │      stopping_strategies,          ◄── pluggable         │  │             │
  │  │      weights, filter_mask, truncation_positions          │  │             │
  │  │  )                                                       │  │             │
  │  │                                                          │  │             │
  │  │  still_running = [True] * 64                             │  │             │
  │  │                                                          │  │             │
  │  │  for pos_idx, pos in enumerate(probe_positions):         │  │             │
  │  │  │                                               SEQUENTIAL             │
  │  │  │  active = still_running          # [64] bool          │  │             │
  │  │  │  if active.sum() == 0: break     # early exit         │  │             │
  │  │  │                                                       │  │             │
  │  │  │  ┌─── VECTORIZED across active iterations ─────────┐  │  │             │
  │  │  │  │                                                 │  │  │             │
  │  │  │  │  ans  = sampled_ans_ids[active, :, pos_idx]     │  │  │             │
  │  │  │  │  elig = (ans >= 0)                              │  │  │             │
  │  │  │  │  if filter_mask: elig &= filter_mask[active]    │  │  │             │
  │  │  │  │  if truncation:  elig &= ~past_truncation       │  │  │             │
  │  │  │  │                                                 │  │  │             │
  │  │  │  │  votes = compute_votes(ans, w, elig, n_ans)     │  │  │             │
  │  │  │  │  # SNAPSHOT: fresh votes, NOT accumulated       │  │  │             │
  │  │  │  │                                                 │  │  │             │
  │  │  │  │  state = PositionState(                         │  │  │             │
  │  │  │  │      vote_weights, answer_ids, weights,         │  │  │             │
  │  │  │  │      eligible, w_active, w_active_by_ans, ...   │  │  │             │
  │  │  │  │  )                                              │  │  │             │
  │  │  │  │                                                 │  │  │             │
  │  │  │  │  for label, strat in stopping_strategies:       │  │  │             │
  │  │  │  │    can_stop |= strat.check(state) ◄── strategy  │  │  │             │
  │  │  │  │                                                 │  │  │             │
  │  │  │  └─────────────────────────────────────────────────┘  │  │             │
  │  │  │                                                       │  │             │
  │  │  │  newly_stopped = can_stop                             │  │             │
  │  │  │  result_*[stopped] = ...    # record answer, tokens   │  │             │
  │  │  │  still_running[active] &= ~newly_stopped              │  │             │
  │  │  │                                                       │  │             │
  │  │  endfor                                                  │  │             │
  │  │                                                          │  │             │
  │  │  result_*[still_running] = budget_stop                   │  │             │
  │  │                                                          │  │             │
  │  │  return SimulationResult(                                │  │             │
  │  │      answers, is_correct, total_tokens,                  │  │             │
  │  │      stop_positions, stopped_by, canonical_map           │  │             │
  │  │  )                                                       │  │             │
  │  └──────────────────────────────────────────────────────────┘  │             │
  └────────────────────────────────────────────────────────────────┼─────────────┘
                                      │                            │
                                      ▼                            │
  ┌────────────────────────────────────────────────────────────────┴────────────┐
  │                           SCRIPTS (L3)                                      │
  │                                                                             │
  │  Single unified script: parse CLI → resolve data format →                  │
  │                         dispatch by --method → simulate → save             │
  │                                                                             │
  │  ┌────────────────────────────────────────────────────────────────────────┐ │
  │  │  examples/run_experiment.py                                            │ │
  │  │                                                                        │ │
  │  │  --model {deepseek-8b, qwen3-32b, qwen3-next}                         │ │
  │  │  --dataset {aime-2025, hmmt, brumo-2025, aime-2024}                   │ │
  │  │  --method {offline, dco, sc-qm3-nc, dco-qm3-nc,                       │ │
  │  │            sc-qm3-nc-oqg, dco-qm3-nc-oqg, sc-oq-nc, dco-oq-nc,        │ │
  │  │            oracle, oracle-dco, sc-pp}                                 │ │
  │  │  --warmup-gamma, --ucb-z, --gamma, --gamma-min                       │ │
  │  │                                                                        │ │
  │  │  Data format resolution:                                               │ │
  │  │    PKL_DATASETS[model][dataset] → pkl format (split + cache)          │ │
  │  │    --data-dir → legacy NPZ+JSON format                               │ │
  │  │                                                                        │ │
  │  │  Method dispatch in worker_process_question():                         │ │
  │  │    Weighting: sc-* uniform | dco-* conf_at_pos                        │ │
  │  │    Filtering: dco-* threshold+trunc (cumulative-min)                  │ │
  │  │    Q model:   *-qm3-* learned 5-feature | *-oq-* oracle              │ │
  │  │    Stopping:  *-nc MARS | sc-pp Parallel-Probe | else NeverStop      │ │
  │  │    Gamma:     --warmup-gamma calibrates per-question                  │ │
  │  └────────────────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                           PARALLELIZATION                                    │
  │                                                                              │
  │  ProcessPoolExecutor(max_workers=N)                                          │
  │                                                                              │
  │  ┌────────────────────────────────────────────────────────────┐              │
  │  │  Worker per question (fully independent):                  │              │
  │  │                                                            │              │
  │  │  1. Load data from disk (format-aware):                    │              │
  │  │       NPZ: load_question_npz() + load_probed_question()   │              │
  │  │       PKL: load_pkl_question(cache/q{qid}.pkl)             │              │
  │  │  2. precompute_question_cached()                           │              │
  │  │       answer grouping, confidence metrics,                 │              │
  │  │       conf_at_pos, ans_ids_at_pos, flips, streaks          │              │
  │  │       (~54s cold / ~4s from NPZ cache per question)        │              │
  │  │  3. [if QM/NC/CLT] fit_q_model + precompute_q             │              │
  │  │     [if gamma] calibrate_gamma_warmup()                    │              │
  │  │  4. sample_bootstrap() — from probed traces only           │              │
  │  │  5. configure method (weights, filter, stopping)           │              │
  │  │  6. simulate_with_stopping()                               │              │
  │  │  7. Return lightweight results           (~2KB)            │              │
  │  └────────────────────────────────────────────────────────────┘              │
  │                                                                              │
  │  Q0 ──▶ Worker 0 ──┐                                                         │
  │  Q1 ──▶ Worker 1 ──┤                                                         │
  │  Q2 ──▶ Worker 2 ──┤                                                         │
  │  ...               ├──▶ main process: aggregate + save via results_io        │
  │  Q28 ──▶ Worker N ─┤                                                         │
  │  Q29 ──▶ Worker N ─┘                                                         │
  │                                                                              │
  │  No data serialization across processes.                                     │
  │  Each worker loads from disk (50ms) vs pickle serialize (500ms).             │
  │  Returns 5 arrays of shape [64] per question ≈ 2KB.                          │
  └──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                              OUTPUT                                          │
  │                                                                              │
  │  results/{dataset}/{method}_{params}_{timestamp}/                            │
  │      ├── config.json               # ExperimentConfig serialized             │
  │      ├── results.csv               # per-(question, iteration) rows          │
  │      ├── summary_per_question.csv  # mean/std per question                   │
  │      ├── summary_overall.csv       # single-row cross-question summary       │
  │      └── results_{voting_method}.csv  # (offline only) one per voting method │
  └──────────────────────────────────────────────────────────────────────────────┘
```