# Experiment Output Convention

This document defines the standard output format for experiments in this repository.
See `mars/results_io.py` for the implementation.

## Directory Structure

Each experiment creates a timestamped subfolder under `./results/{dataset}/`:

```
results/
└── {dataset}/
    └── {method}_{param_str}_{YYYYMMDD}_{HHMMSS}/
        ├── config.json              # Experiment parameters
        ├── results.csv              # Full per-iteration data
        ├── summary_per_question.csv # Aggregated stats per question
        └── summary_overall.csv      # Overall performance summary
```

Example: `results/aime-2025/sc-mars-cal_ggwarmup_20260503_204401/`

For the `offline` method, additional per-voting-method CSVs are created:
`results_{voting_method}.csv`.

## File Schema

### 1. results.csv

One row per (question, iteration) pair.

#### Columns

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | int | Question/problem index |
| `iteration` | int | Bootstrap iteration index |
| `voted_answer` | str | Model's predicted answer (None if no eligible traces) |
| `ground_truth` | str | Correct answer |
| `is_correct` | bool | Whether prediction matches ground truth |
| `total_tokens` | int | Total tokens consumed in this iteration |
| `stopped_by` | str | Stopping reason: `"margin"`, `"consensus"`, `"budget"`, or `"oracle_optimistic"` / `"oracle_absorbing"` |
| `position` | int | Probe position (token index) where stopping occurred |

### 2. summary_per_question.csv

Aggregated statistics per question.

#### Always-present columns

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | int | Question index |
| `accuracy` | float | Mean correctness (0.0 to 1.0) |
| `accuracy_pct` | float | Accuracy as percentage |
| `accuracy_std` | float | Standard deviation across iterations |
| `n_correct` | int | Count of correct iterations |
| `n_iterations` | int | Number of bootstrap iterations |
| `total_tokens_mean` | float | Average tokens per iteration |
| `total_tokens_sum` | int | Total tokens across all iterations |

#### Conditional columns (present when relevant data exists)

| Column | Condition | Description |
|--------|-----------|-------------|
| `token_savings_pct` | baseline available | Percentage of tokens saved vs offline |
| `mean_position` | `position` in results | Average stopping position |
| `stopped_by_margin` | stopping data present | Count of iterations stopped by margin |
| `stopped_by_consensus` | stopping data present | Count stopped by consensus |
| `stopped_by_budget` | stopping data present | Count stopped by budget exhaustion |
| `mean_n_truncated` | truncation methods | Average traces truncated |

### 3. summary_overall.csv

Single-row overall performance summary.

| Column | Type | Description |
|--------|------|-------------|
| `accuracy` | float | Overall accuracy (0.0 to 1.0) |
| `accuracy_pct` | float | Accuracy as percentage |
| `accuracy_std` | float | Standard deviation |
| `n_questions` | int | Number of questions |
| `total_tokens_mean` | float | Average tokens per iteration |
| `mean_token_savings_pct` | float | Mean token savings across questions (if available) |
| `stopped_by_margin` | int | Total iterations stopped by margin |
| `stopped_by_consensus` | int | Total iterations stopped by consensus |
| `stopped_by_budget` | int | Total iterations stopped by budget |

### 4. config.json

Serialized `ExperimentConfig` dataclass with all parameters needed to reproduce the run.

```json
{
  "method": "sc-mars-cal",
  "dataset": "aime-2025",
  "budget": 512,
  "n_iterations": 64,
  "warmup": 16,
  "window_size": 2048,
  "seed": 42,
  "weighting": "uniform",
  "truncation": false,
  "stopping": "cal",
  "extra": {
    "delta": 0.05,
    "gamma": 1.0,
    "warmup_gamma": true,
    "ucb_z": 1.0,
    "gamma_min": 0.5
  }
}
```

## Naming Convention

**Folder pattern**: `{method}_{param_str}_{YYYYMMDD}_{HHMMSS}/`

| Component | Description |
|-----------|-------------|
| `method` | Method name (e.g., `sc-mars`, `dco-mars-cal`) |
| `param_str` | Encoded parameters (e.g., `ggwarmup`, `g1`) |
| `YYYYMMDD_HHMMSS` | Timestamp |

Folders are nested under `results/{dataset}/` (e.g., `results/aime-2025/`).
