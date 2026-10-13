"""
preprocess_borg_trace.py
------------------------
Preprocesses the raw Kaggle Google Borg 2019 cluster-sample CSV into a
clean 4-column workload file consumed by KaggleBorgLoader.

Usage:
    python scripts/preprocess_borg_trace.py [options]

Output CSV schema:
    cpu_req   float   Absolute CPU units (cpu_raw * cpu_capacity)
    mem_req   float   Absolute memory units (mem_raw * mem_capacity)
    priority  int     Tier {1, 2, 3} derived from rank-thirds via pd.qcut
    duration  int     SYNTHETIC — log-normal seeded, NOT from trace
"""

import argparse
import ast
import os
import sys
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess Kaggle Borg 2019 cluster-sample trace CSV."
    )
    p.add_argument("--input",  default="data/raw/borg_traces_data.csv",
                   help="Path to raw CSV (default: data/raw/borg_traces_data.csv)")
    p.add_argument("--output", default="data/processed/borg_jobs.csv",
                   help="Path for processed output CSV (default: data/processed/borg_jobs.csv)")
    p.add_argument("--seed",   type=int, default=42,
                   help="RNG seed for synthetic duration (default: 42)")
    p.add_argument("--n-jobs", type=int, default=5000,
                   help="Max rows in output after all filters (default: 5000)")
    p.add_argument("--cpu-capacity", type=float, default=16.0,
                   help="CPU capacity per machine, matches env (default: 16.0)")
    p.add_argument("--mem-capacity", type=float, default=64.0,
                   help="Memory capacity per machine, matches env (default: 64.0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Resource request parser
# ---------------------------------------------------------------------------

def _parse_resource_request(raw_val):
    """
    Parse a resource_request string of the form
    "{'cpus': 0.01, 'memory': 0.003906}" into (cpu, mem) floats.
    Returns (None, None) if unparseable.
    """
    try:
        d = ast.literal_eval(str(raw_val))
        return float(d.get("cpus", float("nan"))), float(d.get("memory", float("nan")))
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load raw CSV
    # ------------------------------------------------------------------
    print(f"[preprocess] Reading {args.input} …")
    df_raw = pd.read_csv(args.input)
    n_total = len(df_raw)
    print(f"  Input rows: {n_total}")

    # ------------------------------------------------------------------
    # Step 1: Filter to SCHEDULE events only
    # SCHEDULE rows are used as a provisional arrival proxy.
    # This is a simplification — the trace mixes lifecycle events;
    # no complete task-lifetime reconstruction has been validated.
    # ------------------------------------------------------------------
    df = df_raw[df_raw["event"] == "SCHEDULE"].copy()
    n_schedule = len(df)
    print(f"  SCHEDULE rows: {n_schedule}")

    # ------------------------------------------------------------------
    # Step 2: Parse resource_request using ast.literal_eval
    # ------------------------------------------------------------------
    parsed = df["resource_request"].apply(_parse_resource_request)
    df["cpu_raw"] = parsed.apply(lambda x: x[0])
    df["mem_raw"] = parsed.apply(lambda x: x[1])

    n_unparseable = df["cpu_raw"].isna().sum()
    print(f"  Unparseable resource_request rows: {n_unparseable}")
    if n_unparseable > 0:
        df = df.dropna(subset=["cpu_raw", "mem_raw"])

    # ------------------------------------------------------------------
    # Step 3: Drop rows where cpu_raw or mem_raw is 0, NaN, or negative
    # ------------------------------------------------------------------
    mask_valid = (
        df["cpu_raw"].notna() & (df["cpu_raw"] > 0) &
        df["mem_raw"].notna() & (df["mem_raw"] > 0)
    )
    n_zero_or_neg = (~mask_valid).sum()
    print(f"  Rows dropped (zero/NaN/negative resource): {n_zero_or_neg}")
    df = df[mask_valid].copy()
    n_after_filter = len(df)
    print(f"  Rows after zero/missing filter: {n_after_filter}")

    # ------------------------------------------------------------------
    # Step 4: Scale to absolute environment units
    #   cpu_req = cpu_raw * cpu_capacity   (e.g. fraction × 16.0)
    #   mem_req = mem_raw * mem_capacity   (e.g. fraction × 64.0)
    # Then drop jobs that would exceed single-machine capacity.
    # ------------------------------------------------------------------
    df["cpu_req"] = df["cpu_raw"] * args.cpu_capacity
    df["mem_req"] = df["mem_raw"] * args.mem_capacity

    over_cap_mask = (df["cpu_req"] > args.cpu_capacity) | (df["mem_req"] > args.mem_capacity)
    n_over = over_cap_mask.sum()
    frac_over = n_over / len(df) if len(df) > 0 else 0.0
    print(f"  Rows exceeding capacity: {n_over}  ({frac_over:.2%} of post-filter rows)")
    df = df[~over_cap_mask].copy()
    n_after_cap = len(df)
    print(f"  Rows after capacity filter: {n_after_cap}")

    # ------------------------------------------------------------------
    # Step 5: Map priority to 3 tiers via rank-thirds (pd.qcut)
    # This uses equal-frequency binning on the raw priority column.
    # The resulting tiers (1=low, 2=medium, 3=high) are a rank-based
    # prototype and do NOT represent Google's original priority classes.
    # ------------------------------------------------------------------
    pri_min = df["priority"].min()
    pri_max = df["priority"].max()
    print(f"  Raw priority range: [{pri_min}, {pri_max}]")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        priority_tiers, bins = pd.qcut(
            df["priority"],
            q=3,
            labels=[1, 2, 3],
            retbins=True,
            duplicates="drop",
        )

    print(f"  Priority tier cutoffs (qcut bins): {bins.tolist()}")
    df["priority"] = priority_tiers.astype(int)
    tier_dist = df["priority"].value_counts().sort_index()
    print(f"  Priority tier distribution:\n{tier_dist.to_string()}")

    # ------------------------------------------------------------------
    # Step 6: Sort by 'time' column ascending (temporal order)
    # Falls back to original row order if time is degenerate.
    # ------------------------------------------------------------------
    if "time" in df.columns:
        if df["time"].notna().any() and df["time"].nunique() > 1:
            df = df.sort_values("time").reset_index(drop=True)
        else:
            print(
                "  WARNING: 'time' column is all zeros or all NaN — "
                "falling back to original row order."
            )
    else:
        print("  WARNING: No 'time' column found — using original row order.")

    # ------------------------------------------------------------------
    # Step 7: Deduplicate by (collection_id, instance_index)
    # Keep the first occurrence after temporal sort to approximate
    # unique job arrivals. Lifecycle reconstruction is NOT validated.
    # ------------------------------------------------------------------
    n_before_dedup = len(df)
    if "collection_id" in df.columns and "instance_index" in df.columns:
        df = df.drop_duplicates(subset=["collection_id", "instance_index"], keep="first")
    else:
        print("  WARNING: collection_id/instance_index columns not found — skipping dedup.")
    n_removed_dedup = n_before_dedup - len(df)
    print(f"  Rows removed by deduplication: {n_removed_dedup}")
    n_after_dedup = len(df)
    print(f"  Rows after deduplication: {n_after_dedup}")

    # ------------------------------------------------------------------
    # Step 8: Sample up to --n-jobs rows (preserve temporal order)
    # ------------------------------------------------------------------
    df = df.head(args.n_jobs).copy()
    n_output = len(df)
    print(f"  Rows after sampling (n_jobs={args.n_jobs}): {n_output}")

    # ------------------------------------------------------------------
    # Step 9: Generate SYNTHETIC duration
    # SYNTHETIC/APPROXIMATED — not derived from trace.
    # Log-normal with mean=1.5, sigma=0.5 in log-space, clipped to [1,30].
    # Seeded for reproducibility.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    # SYNTHETIC/APPROXIMATED — not derived from trace.
    raw_dur = rng.lognormal(mean=1.5, sigma=0.5, size=n_output)
    duration = np.clip(raw_dur, 1, 30).astype(int)
    df["duration"] = duration

    # ------------------------------------------------------------------
    # Step 10: Select final 4 columns and write output CSV
    # cpu_req and mem_req are already in absolute environment units (NOT fractions).
    # ------------------------------------------------------------------
    out_cols = ["cpu_req", "mem_req", "priority", "duration"]
    df_out = df[out_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\n  Saved {n_output} rows → {args.output}")

    # ------------------------------------------------------------------
    # Full summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Input rows             : {n_total}")
    print(f"  SCHEDULE rows          : {n_schedule}")
    print(f"  Dropped (unparseable)  : {n_unparseable}")
    print(f"  Dropped (zero/neg res) : {n_zero_or_neg}")
    print(f"  Dropped (over capacity): {n_over}  ({frac_over:.2%})")
    print(f"  Dropped (dedup)        : {n_removed_dedup}")
    print(f"  Output rows            : {n_output}")

    cpu_s = df_out["cpu_req"]
    mem_s = df_out["mem_req"]
    dur_s = df_out["duration"]

    print(f"\n  cpu_req stats (absolute, 0–{args.cpu_capacity}):")
    print(f"    mean={cpu_s.mean():.4f}  std={cpu_s.std():.4f}  "
          f"p50={cpu_s.quantile(0.50):.4f}  p95={cpu_s.quantile(0.95):.4f}  "
          f"max={cpu_s.max():.4f}")

    print(f"\n  mem_req stats (absolute, 0–{args.mem_capacity}):")
    print(f"    mean={mem_s.mean():.4f}  std={mem_s.std():.4f}  "
          f"p50={mem_s.quantile(0.50):.4f}  p95={mem_s.quantile(0.95):.4f}  "
          f"max={mem_s.max():.4f}")

    print(f"\n  priority tier distribution:")
    for tier, cnt in df_out["priority"].value_counts().sort_index().items():
        print(f"    tier {tier}: {cnt} rows ({cnt/n_output:.1%})")

    print(f"\n  duration stats [SYNTHETIC — seeded log-normal, NOT from trace]:")
    print(f"    mean={dur_s.mean():.2f}  std={dur_s.std():.2f}  "
          f"p50={dur_s.quantile(0.50):.1f}  p95={dur_s.quantile(0.95):.1f}")

    print(f"\n  WARNING: Duration is approximated using a seeded log-normal distribution. "
          f"It is NOT derived from the trace.")
    print("=" * 60)


if __name__ == "__main__":
    main()
