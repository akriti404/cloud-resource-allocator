# data/ — Workload Data Directory

## Source

**Kaggle Google Borg 2019 cluster-sample CSV**  
Original dataset: [Google Cluster Data 2019 on Kaggle](https://www.kaggle.com/)  
Raw filename: `borg_traces_data.csv`  
Total rows in raw file: 405,894 rows × 34 columns

## File Locations

| File | Location | In git? |
|------|----------|---------|
| Raw trace | `data/raw/borg_traces_data.csv` | **No** — 313 MB, ignored by `.gitignore`. Download from Kaggle manually. |
| Processed workload | `data/processed/borg_jobs.csv` | **Yes** — 145 KB, committed |
| Raw profiling stats | `data/borg_raw_profile.txt` | **Yes** — documents the dataset analysis |
| Synthetic train trace | `data/borg_trace_subset.csv` | No — regenerate with `python generate_dummy_trace.py` |
| Synthetic eval trace | `data/borg_eval_subset.csv` | No — regenerate with `python generate_dummy_trace.py --seed 99 --output data/borg_eval_subset.csv` |

## How to Regenerate

```bash
python scripts/preprocess_borg_trace.py
```

Optional arguments:

```
--input  data/raw/borg_traces_data.csv   # path to raw CSV
--output data/processed/borg_jobs.csv   # output path
--seed   42                              # RNG seed for synthetic duration
--n-jobs 5000                            # max output rows
--cpu-capacity 16.0                      # must match env cpu_capacity
--mem-capacity 64.0                      # must match env mem_capacity
```

## Processed File Schema

`data/processed/borg_jobs.csv` — 4 columns, 5,000 rows (default):

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `cpu_req` | float32 | (0, 16.0] | Absolute CPU units per job (NOT a fraction) |
| `mem_req` | float32 | (0, 64.0] | Absolute memory units per job (NOT a fraction) |
| `priority` | int | {1, 2, 3} | Priority tier (1=low, 2=medium, 3=high) |
| `duration` | int | [1, 30] | Steps — **SYNTHETIC, NOT from trace** |

## Filtering Decisions

1. **SCHEDULE events only** — rows where `event == "SCHEDULE"` are used as a
   provisional job-arrival proxy. This is a simplification: the trace mixes
   lifecycle events and no complete task-lifetime reconstruction has been
   validated.  
   _69,104 SCHEDULE rows_ from 405,894 total.

2. **Zero / missing / negative resource_request removed** — rows where the
   parsed `cpus` or `memory` field is zero, NaN, or negative are dropped.  
   _104 rows dropped._

3. **Over-capacity jobs removed** — after scaling, jobs where
   `cpu_req > cpu_capacity` OR `mem_req > mem_capacity` are dropped.  
   _0 rows dropped (0.00%)_ — all scaled requests fit within the environment.

4. **Deduplication** — the first occurrence of each `(collection_id, instance_index)`
   pair is kept after temporal sort. Lifecycle reconstruction across events is
   not validated.  
   _13,197 duplicate rows removed._

## Scaling

Resource fractions from the trace are converted to absolute environment units:

```
cpu_req = cpu_raw * 16.0    # cpu_capacity = 16.0
mem_req = mem_raw * 64.0    # mem_capacity = 64.0
```

These multipliers **must match** the `cpu_capacity` and `mem_capacity` settings
in `env/resource_env.py` and `training/train.py`. Do not change them independently.

## Priority Mapping

Source `priority` field contains raw values in range `[0, 360]` across 13 distinct
levels (e.g., 0, 25, 101, 103, 105, 107, 114, 115, 116, 118, 119, 200, 360).

Mapping rule: **rank-thirds via `pd.qcut(priority, q=3, labels=[1,2,3])`** —
equal-frequency binning into 3 tiers.

**Actual cutoffs from preprocessing run:**

| Bin | Raw priority range | Tier label |
|-----|--------------------|------------|
| Low | [0, 103) | 1 |
| Medium | [103, 116) | 2 |
| High | [116, 360] | 3 |

qcut bin edges: `[0.0, 103.0, 116.0, 360.0]`

**Caveat:** These tiers are a rank-based prototype and do NOT represent
Google's original priority classes. The source priority values observed in
this Kaggle export may reflect a transformation from the official schema.

## Duration — ⚠️ SYNTHETIC

**Duration is NOT derived from the trace.**

`start_time` / `end_time` in this CSV describe measurement windows, not
validated task runtimes. Grouping by `(collection_id, instance_index)` did
not yield complete SCHEDULE-to-terminal-event pairs.

Duration is **approximated** using a seeded log-normal distribution:

```python
rng = np.random.default_rng(seed=42)
duration = np.clip(rng.lognormal(mean=1.5, sigma=0.5, size=n), 1, 30).astype(int)
```

Stats from the default 5,000-row run:
- mean=4.53, std=2.70, p50=4.0, p95=10.0
- Range: [1, 30] steps

## Known Caveats

The following caveats apply (from `docs/AI_CONTEXT_KAGGLE_UPDATED.md`, section 2):

1. **Mixed lifecycle/usage fields** — this CSV mixes lifecycle event fields
   with usage-window fields in the same table.

2. **start_time / end_time are measurement windows** — their difference is
   not a validated task runtime.

3. **No complete lifecycle reconstruction** — grouping by
   `(collection_id, instance_index)` or `(collection_id, instance_index, machine_id)`
   did not yield groups with both `SCHEDULE` and a terminal event.

4. **Numeric event codes may be transformed** — `event` labels and
   `instance_events_type` codes are internally consistent but the numeric
   codes appeared reversed relative to the official schema. This may reflect
   a Kaggle-export transformation; do not assume an untouched original table.

5. **Real-trace-informed prototype only** — be transparent that task durations
   are approximated. Do not claim a fully real-trace-driven scheduler.

6. **Temporal order approximates arrivals** — the environment advances one job
   per step; chronological ordering approximates arrivals rather than simulating
   exact inter-arrival gaps.

7. **Priority tier mapping is provisional** — rank-thirds binning is a
   documented prototype choice, not validated against Google's original
   priority semantics.

8. **Raw file must not be committed** — keep `data/raw/` out of git. The
   processed file (`data/processed/borg_jobs.csv`) and profiling stats
   (`data/borg_raw_profile.txt`) are committed and tracked.
