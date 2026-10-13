# Cloud Resource Allocator

A deep reinforcement learning framework for intelligent job placement in a simulated cloud environment. The project models resource allocation as a sequential decision-making problem: for each incoming synthetic Borg-style job, an RL agent observes current machine states and selects where to place the job. The primary agent is a Dueling DQN with Double DQN, Prioritized Experience Replay, and heuristic-guided exploration. Performance is compared against five classical baseline schedulers across multiple seeds to produce statistically grounded results.

The workload is fully synthetic (Borg-style log-normal distributions). Due to deliberate cluster over-provisioning relative to job arrival rate, utilization is intentionally low (~11–12% CPU/memory) — this is mathematically expected and not a bug. See the architecture summary and known limitations below.

---

## Architecture Summary

| Module | Role |
|---|---|
| `env/resource_env.py` | Gymnasium-based cluster scheduling environment: 4 machines, 16 CPU cores / 64 GB RAM each, 16-dim observation, discrete action per machine |
| `env/server.py`, `env/task.py` | **Legacy unused files** — not part of the active system |
| `agents/dqn_agent.py` | Dueling DQN + Double DQN + Prioritized Experience Replay + target network |
| `baselines/heuristics.py` | FirstFit, BestFit, GreedyPriority, RoundRobin, Random schedulers |
| `training/train.py` | Main training loop (800 ep, seed=42 default), evaluation functions, plots, metadata JSON |
| `training/run_experiments.py` | Multi-seed experiment runner: summary CSV, bar chart, per-seed checkpoints |
| `utils/data_loader.py` | `BorgDatasetLoader` — loads CSV, scales resources to absolute cluster units, wraps around |
| `utils/replay_buffer.py` | Prioritized replay buffer |
| `utils/logger.py` | Episode logger |
| `generate_dummy_trace.py` | Synthetic workload generator (log-normal CPU/mem, discrete priority, log-normal duration) |

---

## Setup

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Generate training dataset
python generate_dummy_trace.py

# Generate held-out evaluation dataset
python generate_dummy_trace.py --seed 99 --output data/borg_eval_subset.csv
```

---

## Running Training

```bash
python training/train.py
```

Trains for 800 episodes (seed=42 by default). Uses `data/borg_eval_subset.csv` for evaluation if it exists, otherwise falls back to `data/borg_trace_subset.csv`.

Writes:
- `checkpoints/dqn_resource.pt`
- `checkpoints/experiment_metadata.json`
- `training/plots/results.png`

---

## Running Multi-Seed Experiments

```bash
python training/run_experiments.py

# or with custom seeds:
python training/run_experiments.py --seeds 42 7 123

# or skip training and re-evaluate an existing checkpoint:
python training/run_experiments.py --eval-only
```

Produces:
- `training/results/multi_seed_summary.csv`
- `training/plots/multi_seed_results.png`
- `checkpoints/dqn_seed_{seed}.pt` (one per seed)

> **Note:** By default `run_experiments.py` uses **100 episodes per seed** for speed. To run full production experiments, edit the `n_episodes` override in the script from `100` to `800` (and `eval_every` from `25` to `50`).

---

## Running Tests

```bash
.\venv\Scripts\python.exe -m unittest discover tests -v
```

All 18 tests should pass.

---

## Results

### Current Results (single-run, hybrid DQN eval, pre-Task-3 baseline)

*Test configuration: 4 machines, 5,000 generated jobs, 100 jobs/episode, 800 training episodes.*

*Note: these are from a single training run using hybrid (teacher-assisted) evaluation. Fair multi-seed results will replace this table after running `run_experiments.py`.*

| Scheduler | Mean Reward | Notes |
|---|---|---|
| Best Fit | 96.22 | Heuristic |
| Greedy Priority | 95.98 | Heuristic |
| First Fit | 95.97 | Heuristic |
| DQN (hybrid eval) | 95.76 | RL agent, teacher-assisted eval |
| Random | 83.41 | Heuristic |
| Round Robin | 81.98 | Heuristic |

### Multi-Seed Results (to be filled after running run_experiments.py)

| Scheduler | Reward mean ± std | Placement Rate | SLA Breach Rate |
|---|---|---|---|
| DQN (hybrid) | — | — | — |
| DQN (pure) | — | — | — |
| Best Fit | — | — | — |
| Greedy Priority | — | — | — |
| First Fit | — | — | — |
| Round Robin | — | — | — |
| Random | — | — | — |

---

## Known Limitations

- **Synthetic workload only.** Low utilization (~11–12% CPU/memory) is mathematically expected given cluster over-provisioning relative to job arrival rate — not a scheduler deficiency. See `docs/TASK1_AUDIT_RUOTH.md` for the full Little's Law derivation.
- **Single-run results do not have confidence intervals.** Use `run_experiments.py` for multi-seed evaluation with mean ± std.
- **Hybrid vs pure DQN distinction.** Hybrid eval uses a GreedyPriority teacher to override low-confidence DQN decisions; pure eval uses the DQN alone. The hybrid result is optimistic relative to real deployment.
- **DQN does NOT consistently outperform the best heuristic baselines** on this workload. BestFit and GreedyPriority are strong competitors.

---

## Project Structure

```text
cloud-resource-allocator/
│
├── agents/
│   └── dqn_agent.py
│
├── baselines/
│   └── heuristics.py
│
├── checkpoints/            ← model checkpoints (not in git)
│
├── data/                   ← generated CSVs (not in git)
│   ├── borg_trace_subset.csv
│   └── borg_eval_subset.csv
│
├── docs/
│   └── TASK1_AUDIT_RUOTH.md
│
├── env/
│   ├── resource_env.py     ← active environment
│   ├── server.py           ← legacy, unused
│   └── task.py             ← legacy, unused
│
├── tests/
│   ├── test_workload.py
│   ├── test_environment.py
│   ├── test_evaluation.py
│   └── test_dqn_agent.py
│
├── training/
│   ├── train.py
│   ├── run_experiments.py
│   ├── plots/              ← generated plots (not in git)
│   └── results/            ← multi-seed CSV (not in git)
│
├── utils/
│   ├── data_loader.py
│   ├── replay_buffer.py
│   └── logger.py
│
├── generate_dummy_trace.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python, PyTorch, Gymnasium, NumPy, Pandas, Matplotlib

---

## Team / Task Ownership

| Task | Description | Owner |
|---|---|---|
| Task 1 | Workload, Environment, Reward | Ruoth |
| Task 2 | DQN Agent, Training Pipeline | Anamika |
| Task 3 | Evaluation, Reproducibility, Integration, Documentation | Akriti |
