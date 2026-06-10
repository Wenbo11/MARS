<div align="center">
  <h1>MARS</h1>
  <em>Margin-Adversarial Risk-controlled Stopping</em>
  <br/>
  <em>Token-efficient early stopping for parallel LLM reasoning</em>
</div>

---

MARS decides *when to stop* a pool of parallel reasoning traces. Self-consistency
and confidence voting run every trace to its full length and then vote; MARS
instead probes intermediate answers at checkpoints and stops as soon as the
leading answer is provably safe — when its margin over every challenger exceeds
the worst-case damage that future answer switches could inflict. It matches
full-budget accuracy across all 18 settings (within 0.6 pp where it changes, and
improving it by up to 0.8 pp in several) while saving **25–47%** of tokens under
self-consistency and a further **14–29%** on top of DeepConf.

This repository has two halves:

- **`mars/`** — the offline **simulator**: a vectorized engine that bootstraps
  budget-K parallel runs from a pool of pre-generated traces, votes at each
  checkpoint, and applies a pluggable stopping rule. This is what reproduces the
  paper tables.
- **`generation/`** — the **data pipeline** that produces the traces: generate →
  probe → aggregate (SGLang, multi-GPU). Needs a CUDA GPU.

The two are linked only by the trace pickle format
([`docs/data_format.md`](docs/data_format.md)).

## How it works

At checkpoint $t$, for each challenger answer $k$ the leader $L$ is certified
safe when

$$M_k(t) \ge \sum_{j\in\mathcal{A}_t} q_j\, c_j^k(\gamma),$$

where $M_k(t)=V_L(t)-V_k(t)$ is the current margin, $q_j$ is a per-trace switch
probability (how likely trace $j$ changes its answer before the budget ends), and
$c_j^k(\gamma)$ is the adversarial switch cost. The leader stops once **every**
challenger — including a synthetic unseen one — is certified.

- **$q_j$** is a 5-feature logistic model (checkpoint position, probe confidence,
  answer-flip count, stability streak, confidence trend) fit on 16 warmup traces
  per question, with Platt calibration.
- **$\gamma \in [\tfrac12, 1]$** relaxes the fully-adversarial cost toward observed
  destination behavior, calibrated per question from warmup traces.

See the paper for the safety theorem and proofs; the `mars/` source is small and
documented inline (`mars/voting.py` for the stopping rule, `mars/q_model.py` for
the switch-probability model).

## Install

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Answer
equivalence uses `dynasor.core.evaluator.math_equal` from the
[Dynasor](https://github.com/hao-ai-lab/Dynasor) repo (installed automatically;
**not** the unrelated PyPI package named `dynasor`).

```bash
uv sync                      # simulator only
uv sync --extra generation   # + trace/probe generation (needs CUDA + SGLang)
```

## Quick start (simulator)

Run a method over a model–dataset pool. Each run writes a timestamped folder
under `results/{model}/{dataset}/` containing `config.json` (the exact
parameters), `results.csv` (per-iteration: answer, correctness, tokens,
stop position), and `summary_per_question.csv` / `summary_overall.csv`.

```bash
# Self-consistency baseline (full budget, no stopping)
uv run python examples/run_experiment.py --model deepseek-8b --dataset brumo-2025 --method offline

# MARS on self-consistency (calibrated gamma)
uv run python examples/run_experiment.py --model deepseek-8b --dataset brumo-2025 \
    --method sc-mars-cal --warmup-gamma --ucb-z 1.0 --gamma-min 0.5

# MARS on DeepConf Online
uv run python examples/run_experiment.py --model deepseek-8b --dataset brumo-2025 \
    --method dco-mars-cal --warmup-gamma --ucb-z 1.0 --gamma-min 0.5
```

All reported runs use `--budget 512 --iterations 64 --warmup 16 --window 2048 --seed 42`.

## Methods

MARS layers on two voting pipelines: **SC** (self-consistency, uniform weights)
and **DCO** (DeepConf Online, confidence weighting + threshold filtering). Each
`--method` is a code consumed by `examples/run_experiment.py`:

| Paper row | SC pipeline | DCO pipeline |
|---|---|---|
| Baseline (full budget) | `offline` | `dco` |
| MARS, fully conservative ($\gamma=1$) | `sc-mars` | `dco-mars` |
| MARS, calibrated $\gamma$ | `sc-mars-cal` † | `dco-mars-cal` † |
| MARS, oracle-$q$ diagnostic | `sc-mars-oracle` | `dco-mars-oracle` |
| Parallel-Probe baseline | `sc-pp` | — |

† add `--warmup-gamma --ucb-z 1.0 --gamma-min 0.5` for per-question $\gamma$
calibration. (The `-oracle` rows replace the learned switch model with the
retrospective switch indicator — a diagnostic upper bound, not deployable.)

> The Hoeffding concentration term $\epsilon(N,\delta)$ from the safety theorem
> is **off** in all reported runs (the calibrated rule above is used). It can be
> enabled in code (`MarsStopping(use_correction=True)`); `--delta` only
> affects that path.

## Generating traces

To build your own trace pool (or reproduce on a new model/dataset), see
[`generation/README.md`](generation/README.md). The 3-stage pipeline writes the
`(qid, trace_idx) → {confs, extracted_answer, ground_truth, probes}` pickle the
simulator reads.

## Repository layout

```
mars/             # simulator package (loaders, voting, q-model, simulation engine)
generation/       # trace + probe generation (SGLang, multi-GPU)
examples/         # run_experiment.py — unified CLI
docs/             # data_format.md — the trace pickle schema
data/             # question files (JSONL) + trace pools (see docs/data_format.md)
```

## Acknowledgements

Math-answer equivalence is adapted from
[Dynasor](https://github.com/hao-ai-lab/Dynasor) (and originally Qwen2.5-Math).
Licensed under MIT.
