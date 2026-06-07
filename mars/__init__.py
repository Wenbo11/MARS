"""MARS — Margin-Adversarial Risk-controlled Stopping.

Offline simulator for confidence-based early stopping of parallel LLM
reasoning traces. Given a pool of pre-generated traces with intermediate
answers probed at fixed token intervals, MARS simulates budget-K parallel
runs (via bootstrap), votes at each checkpoint, and decides when the leading
answer is safe to commit — saving tokens without sacrificing accuracy.

See the paper (Margin-Adversarial Risk-controlled Stopping) for the method, and
`docs/data_format.md` for the trace pickle schema produced by the `generation/`
pipeline.
"""

from .raw_traces import RawDataset, RawQuestion, RawTrace
from .probed_traces import ProbedQuestion, ProbedTrace, load_probed_question
from .npz_loader import load_npz_dataset, load_question_npz
from .pkl_loader import (
    load_pkl_dataset,
    load_pkl_question,
    split_pkl_to_per_question,
)
from .answer_equiv import answers_equivalent, group_equivalent_answers
from .voting import (
    MarsStopping,
    ConsensusStopping,
    CompositeStopping,
    ParallelProbeStopping,
    NeverStop,
    calibrate_gamma_warmup,
)
from .simulation import (
    PrecomputedQuestion,
    SimulationResult,
    precompute_question,
    precompute_question_cached,
    sample_bootstrap,
    simulate_with_stopping,
    simulate_dataset_parallel,
)
from .results_io import (
    ExperimentConfig,
    save_experiment,
    load_experiment,
    aggregate_per_question,
    aggregate_overall,
)

__version__ = "0.1.0"

__all__ = [
    # data models / loaders
    "RawDataset", "RawQuestion", "RawTrace",
    "ProbedQuestion", "ProbedTrace", "load_probed_question",
    "load_npz_dataset", "load_question_npz",
    "load_pkl_dataset", "load_pkl_question", "split_pkl_to_per_question",
    "answers_equivalent", "group_equivalent_answers",
    # stopping strategies
    "MarsStopping", "ConsensusStopping", "CompositeStopping",
    "ParallelProbeStopping", "NeverStop", "calibrate_gamma_warmup",
    # simulation engine
    "PrecomputedQuestion", "SimulationResult",
    "precompute_question", "precompute_question_cached", "sample_bootstrap",
    "simulate_with_stopping", "simulate_dataset_parallel",
    # results io
    "ExperimentConfig", "save_experiment", "load_experiment",
    "aggregate_per_question", "aggregate_overall",
]
