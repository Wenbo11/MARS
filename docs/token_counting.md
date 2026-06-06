# Token Counting in the Simulation Engine

## Overview

Token counting simulates the computational cost of running LLM traces under
different stopping and truncation regimes. The total tokens for a bootstrap
iteration is the sum of per-trace token contributions, where each trace is
capped by whichever constraint binds first.

## Two Code Paths

Token counting lives in `simulate_with_stopping()` in `mars/simulation.py`.
There are two paths depending on whether early stopping fires.

### Path A: Early stopping at probe position $p$

When a stopping strategy triggers at probe position $p$
(`mars/simulation.py:637-644`):

```
tokens_per_trace = min(sampled_tokens, truncation_positions, p)
result_tokens    = sum over all traces in the bootstrap sample
```

Each trace contributes `min(full_length, trunc_point, stop_position)` — the
tightest of three caps:

| Cap | Meaning |
|-----|---------|
| `sampled_tokens` | The trace's full generation length |
| `truncation_positions` | Where the logits processor would have forced EOS (see below) |
| `p` (probe position) | We stopped here, so no trace needs to run past this point |

### Path B: Budget exhaustion (no early stopping)

When all probe positions are exhausted without stopping
(`mars/simulation.py:694-698`):

```
tokens_per_trace = min(sampled_tokens, truncation_positions)
result_tokens    = sum over all traces in the bootstrap sample
```

There is no $p$ cap because we never stopped early. Each trace contributes
`min(full_length, trunc_point)`.

This is the path the bare `dco` method takes (it uses `NeverStop`). DCO-family
methods with stopping strategies (e.g., `dco-qm3-nc`) can stop early and use
Path A instead.

## Truncation Positions

For DCO-family methods, `truncation_positions` simulates the original DeepConf
Online logits processor, which monitors sliding-window confidence during
generation and forces EOS when confidence drops below a threshold.

Computation (`mars/simulation.py`, DCO filter block):

```python
truncation_positions[iter_idx, j] = find_truncation_position(
    precomputed.group_confs[trace_idx],
    num_tokens[trace_idx],
    threshold,
    window_size,
)
```

`find_truncation_position()` (in `mars/confidence.py`) scans the
precomputed sliding-window confidence array and returns
`first_bad_window + window_size` — the token position where the logits
processor would have injected EOS.

If no window drops below threshold, it returns `num_tokens` (no truncation).

## Important: Truncation Affects All Traces

Token counting does NOT distinguish between traces that are eligible to vote
and those filtered out. ALL traces contribute tokens up to their truncation
point. This matches the original system where the logits processor truncates
during generation regardless of whether the trace's answer ends up being used
in voting. Truncation is a generation-time mechanism, not a post-hoc filtering
decision.

## Relationship to Filter Mask and Voting Eligibility

The simulation uses two separate mechanisms that serve different purposes:

| Mechanism | Purpose | Shape | Code location |
|-----------|---------|-------|---------------|
| `filter_mask` (3D) | Controls which traces can **vote** at each position | `[n_iter, budget, n_pos]` | Built in the DCO filter block of `worker_process_question()`; consumed in `simulate_with_stopping()` |
| `truncation_positions` | Controls **token accounting** (simulates generation-time truncation) | `[n_iter, budget]` | Same DCO filter block; consumed in `simulate_with_stopping()` token accounting |

The 3D `filter_mask` is built from the cumulative minimum of `conf_at_pos` and
is $\mathcal{F}_t$-measurable (no future information leakage). The
`truncation_positions` are derived from a dense scan of `group_confs` and are
also causal — truncation only takes effect at positions where the triggering
evidence is already observable.

## Methods Without Truncation

For methods that don't use DCO-style truncation (e.g., `sc-qm3-nc`, `offline`),
`truncation_positions` is `None` and the token cap simplifies to:

- Early stopping: `min(sampled_tokens, p)`
- Budget: `sampled_tokens` (full cost, no savings)
