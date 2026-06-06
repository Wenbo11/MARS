"""
Standardized experiment I/O.

A single set of functions for saving, loading, and aggregating experiment
results into the convention documented in docs/experiment_output_convention.md.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ExperimentConfig:
    """Experiment configuration — saved alongside results."""
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
    extra: Dict[str, Any] = field(default_factory=dict)  # method-specific params

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def save_experiment(
    config: ExperimentConfig,
    per_iteration_rows: list[Dict[str, Any]],
    per_question_summary: Dict[int, Dict[str, Any]],
    overall_summary: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save experiment results to standardized directory structure.

    Creates:
        {output_dir}/
            config.json
            results.csv               # per-iteration rows
            summary_per_question.csv
            summary_overall.csv

    Returns:
        Path to created directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Per-iteration results
    results_df = pd.DataFrame(per_iteration_rows)
    required_cols = [
        "question_id", "iteration", "voted_answer", "ground_truth",
        "is_correct", "total_tokens",
    ]
    present_required = [c for c in required_cols if c in results_df.columns]
    optional_cols = [c for c in results_df.columns if c not in required_cols]
    results_df = results_df[present_required + optional_cols]
    results_df.to_csv(output_dir / "results.csv", index=False)

    # Per-question summary
    per_q_rows = []
    for qid in sorted(per_question_summary.keys()):
        row = {"question_id": qid, **per_question_summary[qid]}
        per_q_rows.append(row)
    pd.DataFrame(per_q_rows).to_csv(output_dir / "summary_per_question.csv", index=False)

    # Overall summary
    pd.DataFrame([overall_summary]).to_csv(output_dir / "summary_overall.csv", index=False)

    return output_dir


def load_experiment(exp_dir: Path) -> tuple[ExperimentConfig, pd.DataFrame]:
    """Load a saved experiment."""
    exp_dir = Path(exp_dir)

    with open(exp_dir / "config.json") as f:
        config_dict = json.load(f)

    config = ExperimentConfig(**config_dict)
    results_df = pd.read_csv(exp_dir / "results.csv")

    return config, results_df


def aggregate_per_question(
    results_df: pd.DataFrame,
    baseline_tokens_per_question: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Aggregate per-iteration results to per-question summary.

    Token savings are computed per-iteration then averaged:
        savings_i = 1 - method_tokens_i / baseline_tokens_i
    This uses the per-iteration baseline (full-budget cost for the same
    bootstrap sample), so savings are exactly 0 for offline and always
    non-negative for stopping methods.

    Falls back to fixed per-question baseline if baseline_tokens column
    is absent (backward compatibility).
    """
    rows = []
    has_per_iter_baseline = "baseline_tokens" in results_df.columns

    for qid, group in results_df.groupby("question_id"):
        n_iter = len(group)
        n_correct = int(group["is_correct"].sum())
        acc = n_correct / n_iter if n_iter > 0 else 0.0
        tokens_mean = float(group["total_tokens"].mean())
        tokens_sum = int(group["total_tokens"].sum())

        savings = 0.0
        if has_per_iter_baseline:
            per_iter_savings = 1.0 - group["total_tokens"].values / group["baseline_tokens"].values
            savings = float(per_iter_savings.mean()) * 100
        elif baseline_tokens_per_question and qid in baseline_tokens_per_question:
            baseline = baseline_tokens_per_question[qid]
            if baseline > 0:
                savings = (1 - tokens_mean / baseline) * 100

        row = {
            "question_id": qid,
            "accuracy": acc,
            "accuracy_pct": acc * 100,
            "accuracy_std": float(group["is_correct"].std()),
            "n_correct": n_correct,
            "n_iterations": n_iter,
            "total_tokens_mean": tokens_mean,
            "total_tokens_sum": tokens_sum,
            "token_savings_pct": savings,
        }

        if has_per_iter_baseline:
            row["baseline_tokens_mean"] = float(group["baseline_tokens"].mean())

        if "position" in group.columns:
            row["mean_position"] = float(group["position"].mean())

        if "stopped_by" in group.columns:
            for reason in ["margin", "consensus", "budget"]:
                row[f"stopped_by_{reason}"] = int((group["stopped_by"] == reason).sum())

        if "alpha" in group.columns:
            row["mean_alpha"] = float(group["alpha"].mean())

        if "n_truncated" in group.columns:
            row["mean_n_truncated"] = float(group["n_truncated"].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_overall(per_question_df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate per-question summary to single-row overall summary."""
    result = {
        "accuracy": float(per_question_df["accuracy"].mean()),
        "accuracy_pct": float(per_question_df["accuracy_pct"].mean()),
        "accuracy_std": float(per_question_df["accuracy"].std()),
        "n_questions": len(per_question_df),
        "total_tokens_mean": float(per_question_df["total_tokens_mean"].mean()),
    }

    if "token_savings_pct" in per_question_df.columns:
        result["mean_token_savings_pct"] = float(per_question_df["token_savings_pct"].mean())

    for col in ["stopped_by_margin", "stopped_by_consensus", "stopped_by_budget"]:
        if col in per_question_df.columns:
            result[col] = int(per_question_df[col].sum())

    return result


def generate_output_dir(
    base_dir: Path,
    dataset: str,
    method: str,
    **params,
) -> Path:
    """Generate timestamped output directory name.

    Pattern: {base_dir}/{dataset}/{method}_{param_str}_{timestamp}/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            param_parts.append(f"{k}{v}".replace(".", ""))
        else:
            param_parts.append(f"{k}{v}")

    name_parts = [method] + param_parts + [timestamp]
    dirname = "_".join(name_parts)
    return Path(base_dir) / dataset / dirname
