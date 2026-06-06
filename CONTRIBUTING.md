# Contributing to MARS

Contributions are welcome — bug fixes, new datasets/models, documentation, and
new stopping strategies.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. Set up the environment with `uv sync` (add `--extra generation` if you touch
   the generation pipeline).
3. If you change behavior, verify that the reported benchmark numbers still
   reproduce (see the README results table) and note any deltas in the PR.
4. If you add a method or strategy, document it in `docs/system_design.md` and
   add it to the `--method` choices in `examples/run_experiment.py`.
5. Keep the layered import rule: each module imports only from layers below it
   (see `docs/system_design.md` §3).

## Adding a stopping strategy

Implement the `StoppingStrategy` protocol — a single
`check(state: PositionState) -> np.ndarray` method returning a per-iteration
boolean — and wire it into a new `--method` branch in
`worker_process_question()`. No change to the simulation loop is needed
(see `docs/system_design.md` §7).

## Reproducibility

Every result should be traceable to the exact code and configuration. When
reporting numbers, include the method, model, dataset, and seed. The simulator
is deterministic given `(trace pool, window, seed)`.

## Issues

Please use GitHub issues for bugs and feature requests, with enough detail to
reproduce.
