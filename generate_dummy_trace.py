
import argparse
import os

import numpy as np
import pandas as pd


# Priority tiers: (value, weight) — low pri most common, high pri rare.
_PRIORITY_TIERS = [1, 2, 3]
_PRIORITY_PROBS  = [0.60, 0.30, 0.10]   # matches env._generate_job()


def create_synthetic_trace(
    filename: str = "data/borg_trace_subset.csv",
    num_rows: int = 5_000,
    seed: int = 42,
) -> None:
    """
    Generate a synthetic cluster workload trace and save to CSV.

    Args:
        filename:  Output CSV path.
        num_rows:  Number of job records to generate.
        seed:      RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # CPU requests: log-normal, clipped to (0.01, 0.50) fraction of node  #
    # Mean ~0.10, most jobs < 0.25, occasional large jobs up to 0.50      #
    # ------------------------------------------------------------------ #
    cpu_req = np.exp(rng.normal(loc=-2.5, scale=0.8, size=num_rows))
    cpu_req = np.clip(cpu_req, 0.01, 0.50)

    # ------------------------------------------------------------------ #
    # Memory requests: correlated with CPU (r ≈ 0.7) + independent noise  #
    # ------------------------------------------------------------------ #
    mem_noise = np.exp(rng.normal(loc=-2.5, scale=0.8, size=num_rows))
    mem_req = 0.70 * cpu_req + 0.30 * np.clip(mem_noise, 0.01, 0.50)
    mem_req = np.clip(mem_req, 0.01, 0.50)

    # ------------------------------------------------------------------ #
    # Priority: 3 tiers matching ResourceAllocationEnv reward function    #
    # ------------------------------------------------------------------ #
    priority = rng.choice(_PRIORITY_TIERS, size=num_rows, p=_PRIORITY_PROBS)

    # ------------------------------------------------------------------ #
    # Duration: log-normal, most jobs 1–5 steps, few outliers up to 30   #
    # ------------------------------------------------------------------ #
    duration = np.exp(rng.normal(loc=1.2, scale=0.7, size=num_rows))
    duration = np.clip(np.round(duration).astype(int), 1, 30)

    df = pd.DataFrame({
        "cpu_req":  cpu_req.astype(np.float32),
        "mem_req":  mem_req.astype(np.float32),
        "priority": priority.astype(np.int32),
        "duration": duration,
    })

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

    # ------------------------------------------------------------------ #
    # Print summary statistics so you can sanity-check the distribution   #
    # ------------------------------------------------------------------ #
    print(f"Generated {num_rows} jobs → {filename}")
    print(f"\n  CPU req  — mean: {cpu_req.mean():.3f}  std: {cpu_req.std():.3f}"
          f"  p50: {np.median(cpu_req):.3f}  p95: {np.percentile(cpu_req, 95):.3f}")
    print(f"  Mem req  — mean: {mem_req.mean():.3f}  std: {mem_req.std():.3f}"
          f"  p50: {np.median(mem_req):.3f}  p95: {np.percentile(mem_req, 95):.3f}")
    print(f"  Duration — mean: {duration.mean():.1f}  std: {duration.std():.1f}"
          f"  p50: {np.median(duration):.0f}  p95: {np.percentile(duration, 95):.0f}")
    pri_counts = {p: int((priority == p).sum()) for p in _PRIORITY_TIERS}
    print(f"  Priority — {pri_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cluster workload trace.")
    parser.add_argument("--output",   default="data/borg_trace_subset.csv", help="Output CSV path")
    parser.add_argument("--rows",     type=int, default=5_000,  help="Number of job records")
    parser.add_argument("--seed",     type=int, default=42,     help="RNG seed")
    args = parser.parse_args()

    create_synthetic_trace(filename=args.output, num_rows=args.rows, seed=args.seed)
