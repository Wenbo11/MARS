#!/usr/bin/env python3
"""
Extract a small subset of an aggregated trace pickle for demos / smoke tests.

The full per-dataset trace pools are large (5-7 GB) and are not committed. This
tool slices out a few questions (and optionally caps traces per question) so the
simulator can be exercised without the full pool.

NOTE: capping traces per question changes results — a subset is for trying the
pipeline, NOT for reproducing the paper numbers (which use the full 4096-trace
pool). Reproduction requires the full pickle.

Usage:
    python examples/make_subset.py \
        --input  data/DeepSeek-8B/brumo25_deepseek.pkl \
        --output data/sample/brumo25_demo.pkl \
        --qids 0 1 --max-traces 512
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Full aggregated pkl")
    ap.add_argument("--output", required=True, type=Path, help="Subset pkl to write")
    ap.add_argument("--qids", type=int, nargs="+", default=[0, 1],
                    help="Question ids to keep")
    ap.add_argument("--max-traces", type=int, default=None,
                    help="Cap traces per question (default: keep all)")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    keep_qids = set(args.qids)
    by_q = defaultdict(list)
    for (qid, trace_idx), entry in data.items():
        if qid in keep_qids:
            by_q[qid].append((trace_idx, entry))

    subset = {}
    for qid, items in by_q.items():
        items.sort(key=lambda x: x[0])
        if args.max_traces is not None:
            items = items[: args.max_traces]
        for trace_idx, entry in items:
            subset[(qid, trace_idx)] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_traces = len(subset)
    print(f"Wrote {args.output} — {len(by_q)} questions, {n_traces} traces "
          f"(qids={sorted(by_q)}, max_traces={args.max_traces})")


if __name__ == "__main__":
    main()
