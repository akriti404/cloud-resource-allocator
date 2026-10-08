# Cloud Resource Allocator

A Deep Reinforcement Learning-based framework for intelligent job placement in a simulated cloud computing environment.

The project models cloud resource allocation as a sequential decision-making problem. For each incoming job, an RL agent observes the current state of the available machines and selects a suitable machine for execution.

The current implementation uses a **Dueling Deep Q-Network (Dueling DQN)** combined with Double DQN, Prioritized Experience Replay, feasibility-aware action selection, reward normalization, and heuristic-guided exploration.

---

## Overview

Efficient resource allocation is an important problem in cloud computing. Incoming workloads need to be assigned to available machines while considering multiple factors such as:

- CPU availability
- Memory availability
- Job priority
- Job duration
- Resource utilization
- Load distribution
- Allocation feasibility

Traditional scheduling algorithms such as First Fit and Best Fit rely on predefined rules. This project explores whether a reinforcement learning agent can learn an effective allocation policy through repeated interaction with a simulated cloud environment.

### Workflow

```text
Workload
   │
   ▼
Incoming Job
   │
   ▼
Current Cluster State
   │
   ▼
Dueling DQN Agent
   │
   ▼
Select Machine
   │
   ▼
Allocate Resources
   │
   ▼
Calculate Reward
   │
   ▼
Store Experience
   │
   ▼
Train DQN
```

---

## Problem Formulation

The scheduling problem is formulated as a reinforcement learning problem. At every time step:

1. A job arrives.
2. The agent observes the current cluster state.
3. The agent selects a machine.
4. The job is allocated if the placement is feasible.
5. The environment updates resource availability.
6. A reward is calculated.
7. The transition is stored for learning.

The objective is to learn a policy that makes effective resource allocation decisions over a sequence of jobs.

---

## Reinforcement Learning Setup

### Agent
The learning agent is a **Dueling Deep Q-Network (DQN)**.

### Environment
The environment represents a cluster containing multiple machines with limited CPU and memory resources.

### State
The state represents the current condition of the cluster along with information about the incoming job. For the default four-machine configuration, the observation has 16 dimensions and contains information related to:
- Available CPU on each machine
- Available memory on each machine
- Active workload information
- Current job characteristics
- Episode progress

### Actions
Each action corresponds to selecting one of the available machines. For four machines:
- `Action 0` $\rightarrow$ Machine 0
- `Action 1` $\rightarrow$ Machine 1
- `Action 2` $\rightarrow$ Machine 2
- `Action 3` $\rightarrow$ Machine 3

### Reward
The reward function encourages desirable scheduling decisions while penalizing poor or infeasible allocations, considering:
- Successful resource allocation
- Job priority
- Resource efficiency
- SLA/resource violations
- Resource packing
- Load distribution

---

## Deep Learning Approach

The project uses a **Dueling DQN architecture** implemented with PyTorch. Instead of storing Q-values in a table, a neural network learns to approximate the Q-function:

```text
                  State
                    │
                    ▼
          Fully Connected Layer
                    │
                    ▼
                  ReLU
                    │
                    ▼
          Fully Connected Layer
                    │
           ┌────────┴────────┐
           ▼                 ▼
     Value Stream    Advantage Stream
           │                 │
           ▼                 ▼
          V(s)             A(s,a)
           │                 │
           └────────┬────────┘
                    │
                    ▼
                  Q(s,a)
```

The Dueling architecture separates the estimation of:
- **State value ($V(s)$):** How valuable the current state is.
- **Action advantage ($A(s,a)$):** How much better one action is compared with other actions.

---

## DQN Enhancements

The implementation combines several established Deep RL techniques:

- **Dueling DQN:** Separates state-value and action-advantage estimation to provide a more structured Q-value representation.
- **Double DQN:** Uses the online network to select the next action and the target network to evaluate it, reducing Q-value overestimation.
- **Prioritized Experience Replay:** Experiences with larger temporal-difference errors receive higher priority during replay, allowing the agent to learn more frequently from informative transitions.
- **Target Network:** A separate target network is maintained to provide more stable learning targets, updated via soft/Polyak updates.
- **Feasibility-Aware Action Selection:** Masks actions that cannot accommodate the current job's resource requirements before selection.
- **Reward Normalization:** Uses running reward statistics to normalize rewards and stabilize training.
- **Heuristic-Guided Exploration:** Applies a Greedy Priority heuristic early in training to guide initial exploration, gradually decaying as training progresses.

---

## Workload Generation

The current repository includes a synthetic workload generator:
```bash
python generate_dummy_trace.py
```

It creates jobs containing CPU requirements, memory requirements, priorities, and durations, saving the output to:
```text
data/borg_trace_subset.csv
```
> **Note:** The default experiment uses a synthetic Borg-style workload and should not be interpreted as the original Google production Borg trace.

---

## Baseline Scheduling Algorithms

The learned DQN policy is compared with several conventional scheduling strategies:

- **First Fit:** Assigns a job to the first machine that can accommodate it.
- **Best Fit:** Places the job on a machine where remaining resources are used efficiently.
- **Greedy Priority:** Uses job priority and resource availability for placement decisions.
- **Round Robin:** Cycles through machines sequentially.
- **Random:** Selects a feasible machine randomly.

---

## Getting Started

### 1. Create a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the Workload
```bash
python generate_dummy_trace.py
```

### 4. Train the DQN
```bash
python training/train.py
```

---

## Evaluation

The trained DQN is evaluated against the baseline scheduling strategies, tracking mean episode reward, CPU/memory utilization, and per-machine metrics.

- **Training & Evaluation Plots:** `training/plots/`
- **Model Checkpoints:** `checkpoints/`

### Current Baseline Results
*Test Configuration: 4 machines, 5000 generated jobs, 100 jobs/episode, 800 training episodes (CPU).*

| Scheduler | Reward |
| :--- | ---: |
| **Best Fit** | 96.22 |
| **Greedy Priority** | 95.98 |
| **First Fit** | 95.97 |
| **DQN** | **95.76** |
| **Random** | 83.41 |
| **Round Robin** | 81.98 |

> **Takeaway:** The initial experiment shows that the DQN is competitive with heuristic methods but does not consistently outperform the strongest heuristic baselines on the current synthetic workload. This serves as our foundational baseline.

---

## Research Direction

Planned experiments to improve the resource allocation policy include:
- More challenging workload generation and higher resource utilization scenarios
- Improved reward design and state representation
- DQN hyperparameter optimization and variant comparisons
- Multi-seed evaluations and ablation studies
- Generalization testing on unseen workloads

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
├── data/
│   └── borg_trace_subset.csv
│
├── env/
│   ├── resource_env.py
│   ├── server.py
│   └── task.py
│
├── training/
│   ├── train.py
│   └── plots/
│
├── utils/
│   ├── config.py
│   └── data_loader.py
│
├── checkpoints/
├── generate_dummy_trace.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies Used

- Python
- PyTorch
- Gymnasium
- NumPy
- Pandas
- Matplotlib

---

## Development Status

### Implemented
- [x] Cloud resource allocation environment
- [x] Synthetic workload generation
- [x] Dueling DQN agent & Double DQN
- [x] Prioritized Experience Replay
- [x] Target network with soft updates
- [x] Feasibility-aware action selection
- [x] Reward normalization & Heuristic-guided exploration
- [x] Classical scheduling baselines
- [x] Training pipeline & Evaluation visualization

### Planned
- [ ] Improved workload scenarios & higher-load experiments
- [ ] Reward-function & state representation refinement
- [ ] DQN variant comparison & multi-seed evaluation
- [ ] Ablation experiments & unseen-workload evaluation