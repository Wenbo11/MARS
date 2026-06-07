"""
Per-trace switch probability model.

Estimates q_j(t) = P(trace j switches answer between checkpoint t and the
full-budget endpoint) using logistic regression on five F_t-measurable,
trace-intrinsic features: [position, confidence, flips, streak, conf_trend],
with optional Platt calibration.

See the paper's switch-probability appendix for the model details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Feature Computation (all precomputable, all trace-intrinsic)
# ─────────────────────────────────────────────────────────────────────────────

def compute_flips(ans_ids_at_pos: np.ndarray) -> np.ndarray:
    """Cumulative answer-change count at each probe position.

    Args:
        ans_ids_at_pos: [n_traces, n_positions] int16

    Returns:
        flips: [n_traces, n_positions] int16
            flips[t, p] = number of times answer changed across probes [0..p]
    """
    if ans_ids_at_pos.shape[1] <= 1:
        return np.zeros_like(ans_ids_at_pos, dtype=np.int16)

    # Detect changes: diff != 0 between consecutive positions
    changes = (np.diff(ans_ids_at_pos, axis=1) != 0).astype(np.int16)
    # Cumulative sum, prepend 0 for position 0
    flips = np.zeros_like(ans_ids_at_pos, dtype=np.int16)
    flips[:, 1:] = np.cumsum(changes, axis=1)
    return flips


def compute_streaks(ans_ids_at_pos: np.ndarray) -> np.ndarray:
    """Consecutive same-answer count at each probe position.

    Args:
        ans_ids_at_pos: [n_traces, n_positions] int16

    Returns:
        streaks: [n_traces, n_positions] int16
            streaks[t, p] = number of consecutive probes up to and including p
                            with the same answer as answer(t, p).
                            Minimum value is 1 (the current probe itself).
    """
    n_traces, n_positions = ans_ids_at_pos.shape
    streaks = np.ones_like(ans_ids_at_pos, dtype=np.int16)

    for p in range(1, n_positions):
        same = ans_ids_at_pos[:, p] == ans_ids_at_pos[:, p - 1]
        streaks[:, p] = np.where(same, streaks[:, p - 1] + 1, 1)

    return streaks


# ─────────────────────────────────────────────────────────────────────────────
# Feature Matrix Assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_base_features(
    positions: np.ndarray,
    conf_at_pos: np.ndarray,
    flips: np.ndarray,
    streaks: np.ndarray,
) -> np.ndarray:
    """Assemble the feature matrix for the logistic model.

    Args:
        positions:   [n_positions] int — probe positions (broadcast across traces)
        conf_at_pos: [n_traces, n_positions] float32
        flips:       [n_traces, n_positions] int16
        streaks:     [n_traces, n_positions] int16

    Returns:
        X: [n_traces * n_positions, 4] float64
           columns: [position, confidence, flips, streak]
    """
    n_traces, n_positions = conf_at_pos.shape

    # Broadcast positions to [n_traces, n_positions]
    pos_broadcast = np.broadcast_to(positions[np.newaxis, :], (n_traces, n_positions))

    X = np.column_stack([
        pos_broadcast.ravel().astype(np.float64),
        conf_at_pos.ravel().astype(np.float64),
        flips.ravel().astype(np.float64),
        streaks.ravel().astype(np.float64),
    ])

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Training Labels
# ─────────────────────────────────────────────────────────────────────────────

def build_training_labels(
    ans_ids_at_pos: np.ndarray,
    final_answer_ids: np.ndarray,
) -> np.ndarray:
    """Build binary switch labels from warmup traces.

    Args:
        ans_ids_at_pos:   [n_warmup, n_positions] int16 — answer at each probe
        final_answer_ids: [n_warmup] int16 — final answer

    Returns:
        y: [n_warmup * n_positions] int8
           y[t*P + p] = 1 if ans_ids_at_pos[t, p] != final_answer_ids[t]
    """
    # [n_warmup, n_positions] bool
    switched = ans_ids_at_pos != final_answer_ids[:, np.newaxis]
    return switched.ravel().astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Fitted Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FittedQModel:
    """Fitted logistic regression model for per-trace switch probability.

    Stores coefficients and standardization parameters for vectorized prediction.
    """
    coefficients: np.ndarray    # [5] float64 — [intercept, beta_1..4]
    feature_means: np.ndarray   # [4] float64
    feature_stds: np.ndarray    # [4] float64

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict P(switch) for each row.

        Args:
            X: [n_samples, 4] float64 — raw features

        Returns:
            q: [n_samples] float64 — predicted switch probabilities
        """
        # Standardize
        X_std = (X - self.feature_means) / np.maximum(self.feature_stds, 1e-10)
        # Linear combination
        z = self.coefficients[0] + X_std @ self.coefficients[1:]
        # Sigmoid
        return 1.0 / (1.0 + np.exp(-z))


# ─────────────────────────────────────────────────────────────────────────────
# Model Fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_q_model(X: np.ndarray, y: np.ndarray) -> FittedQModel:
    """Fit logistic regression to warmup training data.

    Args:
        X: [n_samples, D] float64 — features (D can be 4 or 6)
        y: [n_samples] int8 — binary labels

    Returns:
        FittedQModel with fitted coefficients
    """
    from scipy.optimize import minimize

    n_samples, D = X.shape

    # Standardize features
    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)
    feature_stds = np.maximum(feature_stds, 1e-10)  # avoid div by zero
    X_std = (X - feature_means) / feature_stds

    y = y.astype(np.float64)

    def neg_log_likelihood(params):
        intercept = params[0]
        betas = params[1:]
        z = intercept + X_std @ betas
        # Numerically stable log-likelihood
        ll = np.sum(y * z - np.logaddexp(0, z))
        # L2 regularization (mild)
        ll -= 0.01 * np.sum(betas ** 2)
        return -ll

    def gradient(params):
        intercept = params[0]
        betas = params[1:]
        z = intercept + X_std @ betas
        p = 1.0 / (1.0 + np.exp(-z))
        residual = y - p  # [n_samples]

        grad = np.zeros_like(params)
        grad[0] = -residual.sum()
        grad[1:] = -(X_std.T @ residual) + 0.02 * betas
        return grad

    # Initialize at zeros — dimension-agnostic
    params0 = np.zeros(D + 1)

    result = minimize(
        neg_log_likelihood,
        params0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 200},
    )

    return FittedQModel(
        coefficients=result.x,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Oracle q_t (ground-truth switch labels)
# ─────────────────────────────────────────────────────────────────────────────

def compute_oracle_q_values(
    ans_ids_at_pos: np.ndarray,   # [n_traces, n_positions]
    final_answer_ids: np.ndarray, # [n_traces]
) -> np.ndarray:
    """Oracle q_t: 1.0 if intermediate answer != final answer, 0.0 otherwise.

    Args:
        ans_ids_at_pos:   [n_traces, n_positions] int16 — answer at each probe
        final_answer_ids: [n_traces] int16 — final answer

    Returns:
        q_values: [n_traces, n_positions] float32
            1.0 where the trace will switch, 0.0 where it won't.
    """
    switched = (ans_ids_at_pos != final_answer_ids[:, np.newaxis]).astype(np.float32)
    return switched


# ─────────────────────────────────────────────────────────────────────────────
# Platt Calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlattCalibrator:
    """Platt scaling: calibrated_q = sigmoid(a * logit(q_raw) + b).

    When a=1, b=0 this is the identity. Trained on warmup predictions vs labels
    to correct systematic over/under-estimation of switch probability.
    """
    a: float
    b: float

    def calibrate(self, q: np.ndarray) -> np.ndarray:
        """Map raw predictions to calibrated probabilities.

        Args:
            q: [n] float — raw predicted switch probabilities

        Returns:
            calibrated: [n] float — calibrated probabilities in [0, 1]
        """
        q_clipped = np.clip(q, 1e-7, 1.0 - 1e-7)
        logit_q = np.log(q_clipped / (1.0 - q_clipped))
        z = self.a * logit_q + self.b
        return 1.0 / (1.0 + np.exp(-z))


def fit_platt_calibration(
    q_pred: np.ndarray,
    y_true: np.ndarray,
) -> PlattCalibrator:
    """Fit Platt scaling on warmup predictions vs true labels.

    Optimizes a, b to minimize cross-entropy of sigmoid(a * logit(q) + b) vs y.
    Trained on in-sample warmup predictions. With well-specified logistic regression,
    this is approximately the identity (a≈1, b≈0). With model misspecification or
    feature gaps, it corrects systematic bias.

    Args:
        q_pred: [n] float — raw model predictions on training data
        y_true: [n] int8/float — binary switch labels

    Returns:
        PlattCalibrator with fitted a, b
    """
    from scipy.optimize import minimize

    q_clipped = np.clip(q_pred, 1e-7, 1.0 - 1e-7)
    logit_q = np.log(q_clipped / (1.0 - q_clipped))
    y = y_true.astype(np.float64)

    def neg_ll(params):
        a, b = params
        z = a * logit_q + b
        return -np.sum(y * z - np.logaddexp(0, z))

    def grad(params):
        a, b = params
        z = a * logit_q + b
        p = 1.0 / (1.0 + np.exp(-z))
        residual = y - p
        return np.array([-np.sum(residual * logit_q), -np.sum(residual)])

    result = minimize(neg_ll, [1.0, 0.0], jac=grad, method='L-BFGS-B',
                      options={'maxiter': 100})
    return PlattCalibrator(a=result.x[0], b=result.x[1])


# ─────────────────────────────────────────────────────────────────────────────
# Feature Matrix (5 features) + q_t precompute
# ─────────────────────────────────────────────────────────────────────────────

def build_switch_features(
    positions: np.ndarray,          # [n_positions]
    conf_at_pos: np.ndarray,        # [n_traces, n_positions] float32
    flips: np.ndarray,              # [n_traces, n_positions] int16
    streaks: np.ndarray,            # [n_traces, n_positions] int16
) -> np.ndarray:
    """Assemble the 5-feature matrix for the switch-probability model.

    Columns: [position, confidence, flips, streak, conf_trend]

    Returns:
        X: [n_traces * n_positions, 5] float64
    """
    n_traces, n_positions = conf_at_pos.shape

    # Base 4 features: [position, confidence, flips, streak]
    X_base = _build_base_features(positions, conf_at_pos, flips, streaks)

    # conf_trend: conf(p) - conf(p-1), 0 at first position
    conf_trend = np.zeros_like(conf_at_pos, dtype=np.float64)
    if n_positions > 1:
        conf_trend[:, 1:] = np.diff(conf_at_pos.astype(np.float64), axis=1)

    return np.column_stack([X_base, conf_trend.ravel()])


def precompute_switch_probs(
    model: FittedQModel,
    calibrator: Optional[PlattCalibrator],
    conf_at_pos: np.ndarray,
    flips: np.ndarray,
    streaks: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Precompute q_t for all traces, all positions (+ optional calibration).

    Uses 5 features: [position, confidence, flips, streak, conf_trend].

    Args:
        model:       fitted logistic model (5 features)
        calibrator:  Platt calibrator (or None to skip)
        conf_at_pos: [n_traces, n_positions] float32
        flips:       [n_traces, n_positions] int16
        streaks:     [n_traces, n_positions] int16
        positions:   [n_positions] int

    Returns:
        q_values: [n_traces, n_positions] float32
    """
    X = build_switch_features(
        positions, conf_at_pos, flips, streaks,
    )
    q_flat = model.predict(X)

    if calibrator is not None:
        q_flat = calibrator.calibrate(q_flat)

    n_traces, n_positions = conf_at_pos.shape
    return q_flat.reshape(n_traces, n_positions).astype(np.float32)
