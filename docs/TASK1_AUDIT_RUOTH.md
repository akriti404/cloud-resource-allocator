# Task 1 Audit Report: Workload, Environment, and Reward System

**Owner:** Ruoth (Task 1)  
**Target Repository:** `cloud-resource-allocator`  
**Date:** October 2026  

---

## 1. Executive Summary

This audit covers:
1. The synthetic Borg-style workload schema and distribution statistics.
2. The Gymnasium-based cluster scheduling environment (`env/resource_env.py`).
3. Mathematical analysis of the historical low resource utilization (~11.4% CPU, ~11.9% memory).
4. Full specification of the reward formulation.
5. Unit test coverage across workload generation, data loading, resource constraints, job lifecycle, and rewards.

---

## 2. Workload Schema & Distribution Analysis

### Workload Generator (`generate_dummy_trace.py`)
The generator generates 4 key metrics for each job:
- **`cpu_req`**: $\exp(\mathcal{N}(-2.5, 0.8))$, clipped to $[0.01, 0.50]$ of a node capacity. Mean $\approx 0.109$, median $\approx 0.082$.
- **`mem_req`**: Strongly correlated with CPU ($0.70 \cdot \text{cpu} + 0.30 \cdot \text{noise}$), clipped to $[0.01, 0.50]$. Mean $\approx 0.110$, median $\approx 0.092$.
- **`priority`**: Discrete tiers $\{1, 2, 3\}$ sampled with probabilities $[0.60, 0.30, 0.10]$.
- **`duration`**: $\text{round}(\exp(\mathcal{N}(1.2, 0.7)))$, clipped to $[1, 30]$ time steps. Mean $\approx 4.3$, median $\approx 3$.

### Scaling via `BorgDatasetLoader` (`utils/data_loader.py`)
In `BorgDatasetLoader`, fractional values are converted to absolute cluster resource units:
$$\text{cpu\_req}_{\text{abs}} = \text{cpu\_req} \times 16.0 \quad (\text{range: } [0.16, 8.0] \text{ cores})$$
$$\text{mem\_req}_{\text{abs}} = \text{mem\_req} \times 64.0 \quad (\text{range: } [0.64, 32.0] \text{ GB})$$

---

## 3. Environment State, Action, and Transition Dynamics

### Cluster Architecture
- $N = 4$ machines.
- Capacity per machine: $16.0$ CPU cores, $64.0$ GB RAM (Total: 64 cores, 256 GB).
- Legacy files `env/server.py` and `env/task.py` are unused legacy queue implementations and decoupled from `ResourceAllocationEnv`.

### Observation Space
Vector dimension: $3N + 4 = 16$ normalized features in $[0.0, 1.0]$:
1. `cpu_free[0..3] / cpu_capacity` (4 floats)
2. `mem_free[0..3] / mem_capacity` (4 floats)
3. `clip(len(slots[m]) / 10.0, 0.0, 1.0)` (4 floats)
4. `current_job[0] / cpu_capacity` (1 float)
5. `current_job[1] / mem_capacity` (1 float)
6. `current_job[2] / 3.0` (1 float)
7. `jobs_processed / max_jobs` (1 float, progress)

### Action Space & Transitions
- Discrete action: $m \in \{0, 1, \dots, N-1\}$.
- In step $t$:
  1. `_tick_jobs()` decrements TTL for all active jobs across machines. Any job with $\text{TTL} \le 0$ completes and frees its allocated resources.
  2. If machine $m$ has enough resources (`cpu_free[m] >= job_cpu` and `mem_free[m] >= job_mem`), the job is assigned for its full `duration` (TTL).
  3. If resources are insufficient, an **SLA breach** occurs: the job is dropped, no resources change, and a penalty is charged.

---

## 4. Implemented Reward Formulation

When job with priority $P \in \{1, 2, 3\}$ is routed to machine $m$:

### Feasible Placement:
1. **Efficiency Base Reward**:
   $$\text{cpu\_util}_m = 1.0 - \frac{\text{cpu\_free}[m]}{\text{cpu\_capacity}}, \quad \text{mem\_util}_m = 1.0 - \frac{\text{mem\_free}[m]}{\text{mem\_capacity}}$$
   $$\text{efficiency}_m = \frac{\text{cpu\_util}_m + \text{mem\_util}_m}{2}$$
   $$R_{\text{base}} = P \times (0.5 + 0.5 \times \text{efficiency}_m)$$
2. **Packing Bonus**:
   $$\text{If } \frac{\text{cpu\_free}[m]}{\text{cpu\_capacity}} < 0.15 \text{ and } \frac{\text{mem\_free}[m]}{\text{mem\_capacity}} < 0.15 \implies R_{\text{bonus}} = +0.5 \times P$$
3. **Balance Penalty**:
   $$R_{\text{penalty}} = 0.3 \times \left(\sigma(\mathbf{cpu\_util}) + \sigma(\mathbf{mem\_util})\right)$$

$$R = R_{\text{base}} + R_{\text{bonus}} - R_{\text{penalty}}$$

### SLA Breach:
$$R = -\text{sla\_breach\_penalty} \quad (\text{default } -2.0)$$

---

## 5. Root Cause of Observed Low Resource Utilization (~11.4%)

By **Little's Law**:
$$L = \lambda \times W$$
- Arrival rate: $\lambda = 1.0 \text{ job/step}$.
- Mean service time / duration: $W \approx 4.3 \text{ steps}$.
- Average concurrent active jobs in cluster:
  $$L \approx 4.3 \text{ jobs}$$
- Resource demand:
  - Average CPU demand: $4.3 \times 1.74 \text{ cores} \approx 7.48 \text{ cores}$.
  - Cluster CPU capacity: $4 \times 16.0 = 64 \text{ cores}$.
  - **Theoretical max CPU utilization**: $\frac{7.48}{64} \approx \mathbf{11.7\%}$.
  - Average Memory demand: $4.3 \times 7.04 \text{ GB} \approx 30.27 \text{ GB}$.
  - Cluster Memory capacity: $4 \times 64 = 256 \text{ GB}$.
  - **Theoretical max Memory utilization**: $\frac{30.27}{256} \approx \mathbf{11.8\%}$.

**Conclusion:** The low utilization is a direct mathematical consequence of cluster over-provisioning relative to job arrival rate, not an environment bug or poor scheduler placement.

---

## 6. Test Suite Coverage

Located in `tests/`:
- `tests/test_workload.py`: Validates schema, distribution ranges, reproducibility, and loader wraparound.
- `tests/test_environment.py`: Validates allocation, SLA breach handling, TTL transitions, resource deallocation, and exact reward calculations.

All 10 unit tests pass.
