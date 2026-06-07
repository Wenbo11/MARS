# Trace Data Format

This document specifies the data that bridges the two halves of the repo: the
`generation/` pipeline *produces* it, and the `mars/` simulator *consumes* it.

## Question files (input to generation)

A dataset is a JSON-lines file, one math problem per line:

```json
{"question": "Find the number of ...", "answer": "204"}
```

`data/brumo_2025.jsonl` is the bundled sample. `answer` is the ground-truth
string; equivalence at scoring time is checked with
`dynasor.core.evaluator.math_equal` (so `204`, `204.0`, `\boxed{204}` all match).

## Aggregated trace pickle (generation → simulator)

Stage 3 of generation (`generation/aggregate.py`) writes `aggregated_traces.pkl`:
a single Python pickle holding a dict keyed by `(qid, trace_idx)`:

```python
{
    (qid, trace_idx): {
        "qid":              int,            # required — question id
        "trace_idx":        int,            # required — trace index within the question
        "ground_truth":     str,            # required — question-level answer (same for all traces of a qid)
        "confs":            np.ndarray,     # required — [num_tokens] per-token confidence (float; stored float16)
        "probes": {                         # required — intermediate answers at checkpoints
            token_position: {
                "answer":     str | None,   # required — answer if the trace were stopped here
                "is_correct": bool,         # optional — matches ground_truth
                "raw_text":   str,          # optional — the probe completion
                "avg_conf":   float,        # optional
            },
            ...
        },
        "extracted_answer": str | None,     # optional — final answer parsed from the full trace
    },
    ...
}
```

Required vs. optional refers to what `mars/pkl_loader.py` accesses: `confs`,
`probes`, `qid`, `trace_idx`, and `ground_truth` are read unconditionally — a
record missing any of them fails to load. `extracted_answer` is read with a
default of `None`. Within each probe, only `answer` is required (`is_correct` is
used as a fast path when present, else recomputed via `math_equal`).

> **Generation note:** run `generation/aggregate.py` with `--conf-data-dir`
> pointing at the Stage 1 output — that flag is what merges `confs` (and
> `extracted_answer`) into each record. Without it the pickle lacks `confs` and
> will not load.

Key points the simulator relies on (`mars/pkl_loader.py`):

- **`confs`** is the full per-token confidence trace. The simulator derives
  sliding-window confidence metrics from it (window size 2048 by default) and
  uses it for DeepConf weighting/filtering and token accounting.
- **`probes`** keys are absolute token positions. The simulator uses only the
  **2048-multiple** positions as a shared checkpoint grid across all traces
  (`pos % 2048 == 0`); per-trace termination positions are kept per-trace but
  excluded from the shared grid (they are not available across all live traces).
- **`extracted_answer`** is the full-budget answer used as each trace's final
  vote and as the label for the switch-probability model
  (`y = 1{answer(t) != answer(T)}`).
- **`ground_truth`** is read from the first trace of each question.
- Traces whose `extracted_answer` is `None` (never finished reasoning) are
  excluded from bootstrap sampling and warmup.

## Per-question cache (simulator internal)

On first use the simulator splits the monolithic pickle into per-question files
and a method-agnostic precompute cache. Both live next to the data and are
**regenerated on demand** — they are not part of the release artifact:

```
<data>/.cache_<stem>/
├── _metadata.pkl                 # question ids, ground truth, trace lengths
├── q00.pkl, q01.pkl, ...         # per-question lists of trace entries
└── precomputed/w2048/q00.npz ... # frozen answer grouping + features per question
```

The `precomputed/w{window}/q{qid}.npz` files cache the expensive per-question
work: answer grouping (`canonical_map`, `answer_ids`), confidence metrics, and
the q-model features (`flips`, `streaks`, `conf_at_pos`, `ans_ids_at_pos`). The
cache is keyed only by `(qid, window_size)` — **not** by method — so every method
reuses the same frozen grouping. This is what makes results reproducible: given
the same trace pool and window, the answer grouping is identical across runs and
methods.

> Because answer grouping is frozen in the cache, reproducing published numbers
> requires the same trace pool **and** a consistent `math_equal` (the Dynasor
> dependency). Deleting the cache and recomputing with a different Dynasor
> version could in principle regroup borderline symbolic answers.

## NPZ format (legacy, optional)

An older compact layout is still supported via `--data-dir` for the
DeepSeek-8B NPZ format: `raw_traces_compact/q{qid}.npz` plus a probed-traces
sidecar. New data should use the pickle format above; see `mars/npz_loader.py`
and `mars/probed_traces.py` for the schema.
