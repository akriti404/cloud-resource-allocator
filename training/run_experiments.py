"""
Multi-seed experiment runner for the Cloud Resource Allocator project.

Usage:
    python training/run_experiments.py                        # seeds 42 7 123
    python training/run_experiments.py --seeds 42 7 123       # explicit seeds
    python training/run_experiments.py --eval-only            # re-evaluate existing checkpoint

Outputs:
    training/results/multi_seed_summary.csv
    training/plots/multi_seed_results.png
    checkpoints/dqn_seed_{seed}.pt   (one per seed)
"""

import sys
import os
import argparse
import json
import shutil
import csv

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path regardless of CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import training.train as train_module

# Scheduler keys as they appear in experiment_metadata.json eval_results
_SCHEDULERS = [
    "dqn_hybrid",
    "dqn_pure",
    "FirstFit",
    "BestFit",
    "GreedyPriority",
    "RoundRobin",
    "Random",
]

# Keys in experiment_metadata.json that hold full scheduler result dicts
_BASELINE_NAMES = ["FirstFit", "BestFit", "GreedyPriority", "RoundRobin", "Random"]
_DQN_NAMES = ["dqn_hybrid", "dqn_pure"]

_CHECKPOINT_PATH = "checkpoints/dqn_resource.pt"
_METADATA_PATH = "checkpoints/experiment_metadata.json"
_RESULTS_DIR = "training/results"
_PLOTS_DIR = "training/plots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_metadata_eval_results() -> dict:
    """Load eval_results dict from experiment_metadata.json."""
    with open(_METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return metadata["eval_results"]


def _run_eval_only() -> dict:
    """
    Load existing checkpoint and evaluate all schedulers directly.
    Returns eval_results dict with same shape as experiment_metadata.json.
    """
    from agents.dqn_agent import DQNAgent
    from baselines.heuristics import (
        FirstFitBaseline, BestFitBaseline, GreedyPriorityBaseline,
        RoundRobinBaseline, RandomBaseline,
    )
    from utils.data_loader import BorgDatasetLoader

    cfg = train_module.CONFIG
    eval_csv = "data/borg_eval_subset.csv" if os.path.exists("data/borg_eval_subset.csv") else "data/borg_trace_subset.csv"

    borg_eval_loader = BorgDatasetLoader(
        file_path=eval_csv,
        scale_cpu=cfg["cpu_capacity"],
        scale_mem=cfg["mem_capacity"],
    )

    # Build agent and load weights
    from env.resource_env import ResourceAllocationEnv
    _env = ResourceAllocationEnv(
        n_machines=cfg["n_machines"],
        cpu_capacity=cfg["cpu_capacity"],
        mem_capacity=cfg["mem_capacity"],
        max_jobs_per_ep=cfg["max_jobs_per_ep"],
        seed=42,
        dataset_loader=borg_eval_loader,
    )
    obs_dim = _env.observation_space.shape[0]
    n_actions = _env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_machines=cfg["n_machines"],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay_steps=cfg["epsilon_decay_steps"],
        batch_size=cfg["batch_size"],
        buffer_capacity=cfg["buffer_capacity"],
        hidden=cfg["hidden"],
    )
    agent.load(_CHECKPOINT_PATH)

    eval_episodes = cfg["eval_episodes"]
    print(f"  Running hybrid DQN evaluation ({eval_episodes} eps)...")
    dqn_hybrid = train_module.evaluate_agent(
        agent,
        borg_eval_loader,
        eval_episodes,
        priority_guard_threshold=cfg["eval_priority_guard_threshold"],
    )

    print(f"  Running pure DQN evaluation ({eval_episodes} eps)...")
    dqn_pure = train_module.evaluate_agent_pure(agent, borg_eval_loader, eval_episodes)

    baselines = [
        FirstFitBaseline(),
        BestFitBaseline(),
        GreedyPriorityBaseline(),
        RoundRobinBaseline(),
        RandomBaseline(),
    ]
    baseline_results = {}
    for bl in baselines:
        print(f"  Running {bl.name} evaluation ({eval_episodes} eps)...")
        baseline_results[bl.name] = train_module.evaluate_baseline(bl, borg_eval_loader, eval_episodes)

    eval_results = {"dqn_hybrid": dqn_hybrid, "dqn_pure": dqn_pure}
    eval_results.update(baseline_results)
    return eval_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(all_results: list[dict]) -> dict:
    """
    all_results: list of eval_results dicts (one per seed).
    Returns: {scheduler: {metric: (mean, std), ...}, ...}
    """
    aggregated = {}
    for sched in _SCHEDULERS:
        reward_means = [r[sched]["reward_mean"] for r in all_results if sched in r]
        placement_means = [r[sched]["placement_rate_mean"] for r in all_results if sched in r]
        breach_means = [r[sched]["breach_rate_mean"] for r in all_results if sched in r]

        if not reward_means:
            continue

        aggregated[sched] = {
            "reward_mean": float(np.mean(reward_means)),
            "reward_std": float(np.std(reward_means)),
            "placement_rate_mean": float(np.mean(placement_means)),
            "placement_rate_std": float(np.std(placement_means)),
            "breach_rate_mean": float(np.mean(breach_means)),
            "breach_rate_std": float(np.std(breach_means)),
        }
    return aggregated


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def _save_csv(aggregated: dict) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(_RESULTS_DIR, "multi_seed_summary.csv")
    fieldnames = [
        "scheduler",
        "reward_mean", "reward_std",
        "placement_rate_mean", "placement_rate_std",
        "breach_rate_mean", "breach_rate_std",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sched in _SCHEDULERS:
            if sched not in aggregated:
                continue
            row = {"scheduler": sched}
            row.update(aggregated[sched])
            writer.writerow(row)
    print(f"  Saved → {csv_path}")


# ---------------------------------------------------------------------------
# Output: console table
# ---------------------------------------------------------------------------

def _print_table(aggregated: dict) -> None:
    col_w = [20, 22, 22, 22]
    header = (
        "Scheduler".ljust(col_w[0])
        + "Reward (mean±std)".ljust(col_w[1])
        + "Placement (mean±std)".ljust(col_w[2])
        + "Breach (mean±std)".ljust(col_w[3])
    )
    sep = "-" * sum(col_w)
    print("\n" + sep)
    print(header)
    print(sep)
    for sched in _SCHEDULERS:
        if sched not in aggregated:
            continue
        d = aggregated[sched]
        reward_s = f"{d['reward_mean']:.2f} ± {d['reward_std']:.2f}"
        place_s = f"{d['placement_rate_mean']:.3f} ± {d['placement_rate_std']:.3f}"
        breach_s = f"{d['breach_rate_mean']:.3f} ± {d['breach_rate_std']:.3f}"
        print(
            sched.ljust(col_w[0])
            + reward_s.ljust(col_w[1])
            + place_s.ljust(col_w[2])
            + breach_s.ljust(col_w[3])
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Output: bar chart
# ---------------------------------------------------------------------------

def _save_plot(aggregated: dict) -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(_PLOTS_DIR, "multi_seed_results.png")

    scheds = [s for s in _SCHEDULERS if s in aggregated]
    means = [aggregated[s]["reward_mean"] for s in scheds]
    stds = [aggregated[s]["reward_std"] for s in scheds]
    colors = ["steelblue" if s in _DQN_NAMES else "grey" for s in scheds]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(scheds, means, yerr=stds, capsize=5, color=colors, edgecolor="white")

    # Non-zero-anchored y-axis
    if means:
        min_mean = min(means)
        max_mean = max(means)
        max_std = max(stds) if stds else 0.0
        y_lo = min_mean - 2 * max_std
        y_hi = max_mean + 2 * max_std
        if all(m >= 0 for m in means):
            y_lo = max(0.0, y_lo)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Scheduler", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Multi-Seed Experiment Results: DQN vs Baselines", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Label each bar with its mean value
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(stds) * 0.1 if stds else 1),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue", label="DQN"),
        Patch(facecolor="grey", label="Baseline"),
    ]
    ax.legend(handles=legend_handles, fontsize=10)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed experiment runner for the Cloud Resource Allocator."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 7, 123],
        help="Space-separated list of RNG seeds (default: 42 7 123)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; re-evaluate existing checkpoint for each seed.",
    )
    args = parser.parse_args()

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    os.makedirs(_PLOTS_DIR, exist_ok=True)

    all_eval_results = []

    if args.eval_only:
        # ------------------------------------------------------------------ #
        # Eval-only mode: evaluate existing checkpoint once and replicate     #
        # across seeds for aggregation (checkpoint is seed-independent here). #
        # ------------------------------------------------------------------ #
        print("\n[eval-only] Evaluating existing checkpoint...")
        eval_results = _run_eval_only()
        # Use the same result for each seed (checkpoint is fixed)
        all_eval_results = [eval_results] * len(args.seeds)
    else:
        # ------------------------------------------------------------------ #
        # Training mode: run train() for each seed                            #
        # ------------------------------------------------------------------ #
        for s in args.seeds:
            print(f"\n{'='*60}")
            print(f"  Seed {s}: starting training run")
            print(f"{'='*60}")

            # Patch CONFIG for a shorter run (edit here for full production runs)
            original_n_episodes = train_module.CONFIG["n_episodes"]
            original_eval_every = train_module.CONFIG["eval_every"]
            # For development/testing runs. Change to 800/50 for full production runs.
            train_module.CONFIG["n_episodes"] = 800
            train_module.CONFIG["eval_every"] = 50
            try:
                train_module.train(seed=s, eval_csv="data/borg_eval_subset.csv")
            finally:
                train_module.CONFIG["n_episodes"] = original_n_episodes
                train_module.CONFIG["eval_every"] = original_eval_every

            # Copy checkpoint for this seed
            seed_ckpt = f"checkpoints/dqn_seed_{s}.pt"
            shutil.copy2(_CHECKPOINT_PATH, seed_ckpt)
            print(f"  Copied checkpoint → {seed_ckpt}")

            # Read eval results from metadata written by train()
            eval_results = _read_metadata_eval_results()
            all_eval_results.append(eval_results)

    # ---------------------------------------------------------------------- #
    # Aggregate across seeds                                                  #
    # ---------------------------------------------------------------------- #
    print("\nAggregating results across seeds...")
    aggregated = _aggregate(all_eval_results)

    _print_table(aggregated)
    _save_csv(aggregated)
    _save_plot(aggregated)

    print("\nDone. Outputs:")
    print(f"  training/results/multi_seed_summary.csv")
    print(f"  training/plots/multi_seed_results.png")
    for s in args.seeds:
        ckpt = f"checkpoints/dqn_seed_{s}.pt"
        if os.path.exists(ckpt):
            print(f"  {ckpt}")


if __name__ == "__main__":
    main()
