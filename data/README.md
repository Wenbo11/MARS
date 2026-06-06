# Data

## What's committed

- `*.jsonl` — question files (problem + ground-truth answer), one per dataset.
  `brumo_2025.jsonl` is included as the bundled sample. These are small and
  version-controlled.

## What's NOT committed

Trace pools (the pre-generated reasoning traces with per-token confidence and
probed intermediate answers) are large — **5–7 GB per dataset**, ~55–120 MB per
question — and are excluded by `.gitignore`. The simulator expects them at:

```
data/<Model>/<dataset>.pkl        # e.g. data/DeepSeek-8B/brumo25_deepseek.pkl
```

keyed by `(qid, trace_idx)` — see [`../docs/data_format.md`](../docs/data_format.md).
The `examples/run_experiment.py` registry (`PKL_DATASETS`) maps
`(model, dataset)` to these paths.

## How to obtain trace pools

1. **Generate them** with the `generation/` pipeline (needs a CUDA GPU):
   generate → probe → aggregate. See [`../generation/README.md`](../generation/README.md).
   This reproduces the exact pools used in the paper.

2. **Download** a released pool (if/when hosted externally) and place the `.pkl`
   under `data/<Model>/`.

On first use the simulator splits the monolithic pool into a per-question cache
(`.cache_<stem>/`) and a method-agnostic precompute cache
(`.cache_<stem>/precomputed/`). Both are regenerated on demand and gitignored.

## Quick demo on a subset

If you have a full pool but want a fast smoke test, slice out a couple of
questions:

```bash
python examples/make_subset.py \
    --input  data/DeepSeek-8B/brumo25_deepseek.pkl \
    --output data/sample/brumo25_demo.pkl \
    --qids 0 1 --max-traces 512
```

> A trace-capped subset is for exercising the pipeline only. Reproducing the
> published numbers requires the full 4096-trace pool.
