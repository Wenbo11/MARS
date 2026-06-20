# Trace Generation & Probing

Data-generation pipeline for the MARS simulator. It produces reasoning traces,
probes intermediate answers at fixed token intervals, and aggregates everything
into the pickle format the simulator consumes (see
[`../docs/data_format.md`](../docs/data_format.md)).

All stages use the [SGLang](https://github.com/sgl-project/sglang) Engine
(offline, no server). Multi-GPU is handled via multiprocessing (TP=1 per GPU).

> Requires the `generation` extra: `uv sync --extra generation`
> (installs `sglang`, `transformers`, `ninja`). A CUDA GPU is required —
> this stage cannot run CPU-only.

## Pipeline

```
dataset.jsonl  (math competition questions: {"question", "answer"})
    │  generate_traces.py   ← Stage 1: GENERATE
    ▼
outputs/deepconf_simple_qid*_rid*.pkl   (traces with per-token confidence)
    │  probe.py             ← Stage 2: PROBE
    ▼
probe_results/traces/*.pkl   (intermediate answers at each probe point)
    │  aggregate.py         ← Stage 3: AGGREGATE
    ▼
probe_results/aggregated/{aggregated_traces.pkl, aggregated_summary.csv}
```

The aggregated pickle is keyed by `(qid, trace_idx)` and is exactly the input
to `examples/run_experiment.py` / the `mars` simulator.

## Stage 1 — Generate traces (`generate_traces.py`)

Generates reasoning traces with a thinking model, collecting per-token top-k
logprobs for confidence scoring.

```bash
# 2 GPUs, all 30 questions
python generation/generate_traces.py --rid deepseek --num-gpus 2

# Single question, 1 GPU (quick test)
python generation/generate_traces.py --rid test --qids 0 --num-gpus 1

# Custom model and dataset
python generation/generate_traces.py --rid myrun \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --dataset data/brumo_2025.jsonl --num-gpus 4
```

Configuration constants at the top of the file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_BUDGET` | 4096 | Total traces per question |
| `BATCH_SIZE` | 16 | Traces per GPU batch (reduce if OOM) |
| `MAX_TOKENS` | 64000 | Max generation length |
| `WINDOW_SIZE` | 2048 | Sliding window for confidence |
| `DATASET_FILE` | `data/brumo_2025.jsonl` | Bundled sample dataset |

Completed batches are skipped on restart — safe to kill and relaunch.

## Stage 2 — Probe (`probe.py`)

Runs an evaluator model at fixed token intervals along each trace. At each probe
point the trace is truncated and a suffix is injected to force a `\boxed{...}`
answer.

```bash
python generation/probe.py \
  --input-dir outputs_brumo25_deepseek \
  --output-dir probe_results/brumo25 \
  --model-path Qwen/Qwen3-32B \
  --num-gpus 2 --probe-interval 2048 \
  --batch-size 64 --max-model-len 32768 --mem-fraction 0.85
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--probe-interval` | 2048 | Probe every N tokens |
| `--batch-size` | 64 | Inference batch size per GPU |
| `--max-model-len` | 32768 | Max sequence length for probe model |
| `--mem-fraction` | 0.85 | GPU memory fraction for SGLang |

Supports resume — existing per-trace pkls are skipped. The probe model can
differ from the generation model; a smaller model (e.g. 32B) is typical.

## Stage 3 — Aggregate (`aggregate.py`)

Merges all per-trace probe pkls into a single summary.

```bash
python generation/aggregate.py \
  probe_results/brumo25/traces \
  --conf-data-dir outputs_brumo25_deepseek
```

Output (in `probe_results/brumo25/traces/aggregated/`):
- `aggregated_traces.pkl` — dict keyed by `(qid, trace_idx)`
- `aggregated_summary.csv` — flat CSV, one row per `(qid, trace_idx, token_position)`

> **`--conf-data-dir` is required for simulator-compatible output.** It merges
> the per-token `confs` (and `extracted_answer`) from Stage 1 into each record.
> The `mars` simulator requires `confs`, so without this flag the resulting
> `aggregated_traces.pkl` will fail to load. Point it at the Stage 1 output dir.

## Notes

- `helper.py` holds shared utilities (prompt prep, answer extraction,
  confidence). It is imported by the stage scripts, not run directly.
- Answer evaluation uses `dynasor.core.evaluator.math_equal` (installed from the
  [Dynasor](https://github.com/hao-ai-lab/Dynasor) GitHub repo via the project
  dependencies).
- If you hit `FileNotFoundError: ninja`, ensure `ninja` is installed and on PATH.
- Reduce `BATCH_SIZE` / `--batch-size` if you encounter OOM.
