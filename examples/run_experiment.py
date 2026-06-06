#!/usr/bin/env python3
"""
Unified experiment runner for MARS and its baselines.

Usage examples:
    # Self-consistency baseline (full budget, no stopping)
    python examples/run_experiment.py --dataset aime-2025 --method offline

    # DeepConf Online baseline
    python examples/run_experiment.py --dataset aime-2025 --method dco

    # MARS on self-consistency, fully conservative (gamma=1)
    python examples/run_experiment.py --dataset aime-2025 --method sc-qm3-nc

    # MARS on self-consistency, calibrated gamma
    python examples/run_experiment.py --dataset aime-2025 --method sc-qm3-nc-oqg \
        --warmup-gamma --ucb-z 1.0 --gamma-min 0.5

    # MARS on DeepConf Online, calibrated gamma
    python examples/run_experiment.py --dataset aime-2025 --method dco-qm3-nc-oqg \
        --warmup-gamma --ucb-z 1.0 --gamma-min 0.5

    # Parallel-Probe baseline
    python examples/run_experiment.py --dataset aime-2025 --method sc-pp --pp-conv 3

See the README method table for the full mapping to paper rows.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


from mars.npz_loader import load_npz_dataset
from mars.pkl_loader import split_pkl_to_per_question
from mars.probed_traces import load_probed_question
from mars.results_io import (
    ExperimentConfig,
    aggregate_overall,
    aggregate_per_question,
    generate_output_dir,
    save_experiment,
)
from mars.simulation import simulate_dataset_parallel


MODELS = {
    "deepseek-8b": "DeepSeek-8B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-next": "Qwen3-next",
}

DATASET_PATHS = {
    "aime-2025": {},
    "hmmt": {},
    "aime-2024": {},
    "brumo-2025": {},
}

PKL_DATASETS = {
    "qwen3-32b": {
        "aime-2025": "./data/Qwen3-32B/aime25.pkl",
        "hmmt": "./data/Qwen3-32B/hmmt.pkl",
        "brumo-2025": "./data/Qwen3-32B/brumo25.pkl",
        "aime-2024": "./data/Qwen3-32B/aime24.pkl",
    },
    "deepseek-8b": {
        "aime-2025": "./data/DeepSeek-8B/aime25_deepseek.pkl",
        "aime-2024": "./data/DeepSeek-8B/aime24_deepseek.pkl",
        "hmmt": "./data/DeepSeek-8B/hmmt25_deepseek.pkl",
        "brumo-2025": "./data/DeepSeek-8B/brumo25_deepseek.pkl",
    },
    "qwen3-next": {
        "aime-2025": "./data/Qwen3-next/aime25_thinking.pkl",
        "aime-2024": "./data/Qwen3-next/aime24_thinking.pkl",
        "hmmt": "./data/Qwen3-next/hmmt_thinking.pkl",
        "brumo-2025": "./data/Qwen3-next/brumo25_thinking.pkl",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM decoding trace evaluation experiment"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()),
        default="aime-2025",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="deepseek-8b",
        help="Model that generated the traces",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (legacy NPZ format only)",
    )
    parser.add_argument(
        "--method",
        choices=[
            # baselines (full budget, no stopping)
            "offline", "dco",
            # MARS — learned 5-feature q-model (paper "MARS")
            "sc-qm3-nc", "dco-qm3-nc",
            "sc-qm3-nc-oqg", "dco-qm3-nc-oqg",
            # MARS — oracle-q diagnostic (retrospective switch labels)
            "sc-oq-nc", "dco-oq-nc",
            # oracle stopping bounds (diagnostic)
            "oracle", "oracle-dco",
            # Parallel-Probe baseline
            "sc-pp",
        ],
        default="sc-qm3-nc-oqg",
        help="See README method table. *-oqg adds --warmup-gamma calibration.",
    )
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Failure probability for the Hoeffding correction "
                             "(theory only; correction is OFF in all reported runs)")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-voters", type=int, default=0,
                        help="Minimum eligible traces before allowing early stop")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Cost shrinkage factor for the stopping rule (1.0 = full worst-case)")
    parser.add_argument("--warmup-gamma", action="store_true",
                        help="Calibrate gamma per-question from warmup traces (overrides --gamma)")
    parser.add_argument("--ucb-z", type=float, default=0.0,
                        help="UCB correction strength for warmup gamma (0 = no correction)")
    parser.add_argument("--gamma-min", type=float, default=0.5,
                        help="Floor for calibrated gamma (default 0.5)")
    parser.add_argument("--pp-conv", type=int, default=3,
                        help="Parallel-Probe: consecutive stable votes to stop (token-equiv default=3)")
    parser.add_argument("--pp-warmup", type=int, default=4,
                        help="Parallel-Probe: warmup probe steps before stopping/pruning (default=4)")
    parser.add_argument("--pp-prune-patience", type=int, default=2,
                        help="Parallel-Probe: consecutive disagreements before pruning (default=2)")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data source
    use_pkl = False
    pkl_path = None

    if args.data_dir:
        data_path = Path(args.data_dir)
    elif args.model in PKL_DATASETS and args.dataset in PKL_DATASETS[args.model]:
        pkl_path = Path(PKL_DATASETS[args.model][args.dataset])
        data_path = pkl_path.parent
        use_pkl = True
    elif args.dataset in DATASET_PATHS and args.model in DATASET_PATHS[args.dataset]:
        data_path = Path(DATASET_PATHS[args.dataset][args.model])
    else:
        print(f"Error: no data for --model {args.model} --dataset {args.dataset}")
        sys.exit(1)

    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model}")
    print(f"Data:       {pkl_path or data_path}")
    print(f"Method:     {args.method}")
    print(f"Budget:     {args.budget}")
    print(f"Iterations: {args.iterations}")
    print(f"Seed:       {args.seed}")

    if args.min_voters > 0:
        print(f"Min voters: {args.min_voters}")
    if args.gamma != 1.0:
        print(f"Gamma:      {args.gamma}")
    if args.method == 'sc-pp':
        print(f"PP conv:      {args.pp_conv}")
        print(f"PP warmup:    {args.pp_warmup}")
        print(f"PP prune:     {args.pp_prune_patience}")
    if args.warmup_gamma:
        print(f"Warmup gamma: enabled")
        if args.ucb_z > 0:
            print(f"UCB z:        {args.ucb_z}")

    # Discover available questions and load ground truth
    if use_pkl:
        # Split the large pkl into per-question files for efficient worker loading
        q_cache_dir = pkl_path.parent / f".cache_{pkl_path.stem}"
        print(f"Splitting pkl into per-question cache: {q_cache_dir}")
        question_ids, ground_truth_map, _ = split_pkl_to_per_question(
            pkl_path, q_cache_dir,
        )
    else:
        raw_traces_dir = data_path / "raw_traces_compact"
        if not raw_traces_dir.exists():
            print(f"Error: {raw_traces_dir} does not exist")
            sys.exit(1)

        question_ids = []
        for f in sorted(raw_traces_dir.glob("q*.npz")):
            import re
            match = re.search(r'q(\d+)\.npz', f.name)
            if match:
                qid = int(match.group(1))
                probed = load_probed_question(data_path, qid)
                if probed is not None:
                    question_ids.append(qid)
                else:
                    print(f"  Warning: No probed data for question {qid}, skipping")

        dataset = load_npz_dataset(raw_traces_dir, name=args.dataset)
        ground_truth_map = {q.question_id: q.ground_truth for q in dataset.questions}

    print(f"Questions:  {len(question_ids)}")
    print()

    # Build method config
    method_config = {
        'method': args.method,
        'budget': args.budget,
        'n_iterations': args.iterations,
        'warmup': args.warmup,
        'window_size': args.window,
        'seed': args.seed,
        'delta': args.delta,
        'min_voters': args.min_voters,
        'gamma': args.gamma,
        'warmup_gamma': args.warmup_gamma,
        'ucb_z': args.ucb_z,
        'gamma_min': args.gamma_min,
        'pp_conv': args.pp_conv,
        'pp_warmup': args.pp_warmup,
        'pp_prune_patience': args.pp_prune_patience,
        'data_format': 'pkl' if use_pkl else 'npz',
        'pkl_cache_dir': str(q_cache_dir) if use_pkl else None,
    }

    # Run simulation
    start_time = time.time()
    per_question_results = simulate_dataset_parallel(
        data_path=str(data_path),
        question_ids=question_ids,
        method_config=method_config,
        n_workers=args.workers,
    )
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f}s")

    # Build per-iteration DataFrame
    per_iteration_rows = []
    for qid in sorted(per_question_results.keys()):
        results = per_question_results[qid]
        gt = ground_truth_map.get(qid, "")
        for i in range(len(results['is_correct'])):
            row = {
                "question_id": qid,
                "iteration": i,
                "voted_answer": results['answers'][i] or "",
                "ground_truth": gt,
                "is_correct": results['is_correct'][i],
                "total_tokens": results['tokens'][i],
                "baseline_tokens": results['baseline_tokens'][i],
                "stopped_by": results['stopped_by'][i],
                "position": results['positions'][i],
            }
            per_iteration_rows.append(row)

    results_df = pd.DataFrame(per_iteration_rows)

    # Aggregate (per-iteration baseline is in the DataFrame)
    per_q_df = aggregate_per_question(results_df)
    overall = aggregate_overall(per_q_df)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Overall accuracy:    {overall['accuracy_pct']:.1f}%")
    print(f"Mean tokens:         {overall['total_tokens_mean']:.0f}")
    if 'mean_token_savings_pct' in overall:
        print(f"Token savings:       {overall['mean_token_savings_pct']:.1f}%")
    for reason in ['stopped_by_margin', 'stopped_by_consensus', 'stopped_by_budget']:
        if reason in overall:
            print(f"{reason.replace('_', ' ').title()}: {overall[reason]}")
    print(f"{'='*60}")

    # Per-question detail
    print(f"\nPer-question accuracy:")
    for _, row in per_q_df.iterrows():
        qid = int(row['question_id'])
        acc = row['accuracy_pct']
        savings = row.get('token_savings_pct', 0)
        print(f"  Q{qid:02d}: {acc:5.1f}%  savings={savings:5.1f}%")

    # Oracle summary
    if args.method.startswith('oracle'):
        # Build per-iteration rows for each oracle type
        for oracle_type in ('optimistic', 'absorbing'):
            token_key = f'{oracle_type}_tokens'
            oracle_rows = []
            for qid in sorted(per_question_results.keys()):
                results = per_question_results[qid]
                oracle_data = results['oracle']
                gt = ground_truth_map.get(qid, "")
                for i in range(len(results['is_correct'])):
                    oracle_rows.append({
                        "question_id": qid,
                        "iteration": i,
                        "voted_answer": results['answers'][i] or "",
                        "ground_truth": gt,
                        "is_correct": results['is_correct'][i],
                        "total_tokens": oracle_data[token_key][i],
                        "baseline_tokens": oracle_data['baseline_tokens'][i],
                        "stopped_by": f"oracle_{oracle_type}",
                        "position": 0,
                    })
            oracle_df = pd.DataFrame(oracle_rows)

            # Savings vs method baseline (per-iteration)
            oracle_per_q_method = aggregate_per_question(oracle_df)
            oracle_overall_method = aggregate_overall(oracle_per_q_method)

            print(f"\n{'='*60}")
            print(f"Oracle ({oracle_type}):")
            print(f"  Accuracy:      {oracle_overall_method['accuracy_pct']:.1f}%")
            print(f"  Mean tokens:   {oracle_overall_method['total_tokens_mean']:.0f}")
            if 'mean_token_savings_pct' in oracle_overall_method:
                print(f"  Token savings: {oracle_overall_method['mean_token_savings_pct']:.1f}%")
            print(f"{'='*60}")

            print(f"\n  Per-question ({oracle_type}):")
            for _, row in oracle_per_q_method.iterrows():
                qid = int(row['question_id'])
                acc = row['accuracy_pct']
                savings = row.get('token_savings_pct', 0)
                print(f"    Q{qid:02d}: {acc:5.1f}%  savings={savings:5.1f}%")

    # Multi-voting summary for offline method
    voting_per_method_rows = {}  # {vm_name: [rows]}
    if args.method == 'offline':
        # Collect voting method names from first available question
        voting_method_names = []
        for qid in sorted(per_question_results.keys()):
            vr = per_question_results[qid].get('voting_results')
            if vr:
                voting_method_names = list(vr.keys())
                break

        if voting_method_names:
            print(f"\n{'='*60}")
            print(f"Deepconf voting results (offline):")
            print(f"{'='*60}")

            for vm_name in voting_method_names:
                vm_rows = []
                per_q_accs = []
                for qid in sorted(per_question_results.keys()):
                    vr = per_question_results[qid].get('voting_results', {})
                    gt = ground_truth_map.get(qid, "")
                    if vm_name in vr:
                        correct = vr[vm_name]['is_correct']
                        answers = vr[vm_name]['answers']
                        acc = np.mean(correct) * 100
                        per_q_accs.append(acc)
                        for i in range(len(correct)):
                            vm_rows.append({
                                "question_id": qid,
                                "iteration": i,
                                "voted_answer": answers[i] or "",
                                "ground_truth": gt,
                                "is_correct": correct[i],
                                "total_tokens": per_question_results[qid]['tokens'][i],
                            })

                voting_per_method_rows[vm_name] = vm_rows
                if per_q_accs:
                    overall_acc = np.mean(per_q_accs)
                    print(f"  {vm_name:35s} {overall_acc:5.1f}%")

    # Save results
    if not args.no_save:
        # Determine weighting and truncation for config metadata.
        # DCO-family methods (prefix "dco") apply confidence weighting and the
        # threshold filter + truncation; SC-family use uniform weights, no filter.
        weighting = "confidence" if args.method.startswith("dco") else "uniform"
        truncation = args.method.startswith("dco")

        if "-" in args.method:
            stopping = args.method.split("-")[-1]
        else:
            stopping = "none"

        config = ExperimentConfig(
            method=args.method,
            dataset=args.dataset,
            budget=args.budget,
            n_iterations=args.iterations,
            warmup=args.warmup,
            window_size=args.window,
            seed=args.seed,
            weighting=weighting,
            truncation=truncation,
            stopping=stopping,
            extra={
                "delta": args.delta,
                "gamma": args.gamma,
                "warmup_gamma": args.warmup_gamma,
                "ucb_z": args.ucb_z,
                "gamma_min": args.gamma_min,
                "pp_conv": args.pp_conv,
                "pp_warmup": args.pp_warmup,
                "pp_prune_patience": args.pp_prune_patience,
            },
        )

        per_q_summary = {}
        for _, row in per_q_df.iterrows():
            per_q_summary[int(row['question_id'])] = row.drop('question_id').to_dict()

        gamma_tag = "gwarmup" if args.warmup_gamma else "g1"
        is_baseline = args.method in ('offline', 'dco', 'oracle', 'oracle-dco')
        if args.method == 'sc-pp':
            tag_kwargs = {"c": str(args.pp_conv), "w": str(args.pp_warmup), "p": str(args.pp_prune_patience)}
        elif not is_baseline:
            tag_kwargs = {"g": gamma_tag}
        else:
            tag_kwargs = {}
        output_dir = generate_output_dir(
            Path(args.output_dir) / MODELS[args.model], args.dataset, args.method,
            **tag_kwargs,
        )

        save_experiment(config, per_iteration_rows, per_q_summary, overall, output_dir)

        # Save one CSV per voting method for offline
        for vm_name, vm_rows in voting_per_method_rows.items():
            vm_df = pd.DataFrame(vm_rows)
            vm_df.to_csv(output_dir / f"results_{vm_name}.csv", index=False)

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
