import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Protocol, Tuple
from dataclasses import dataclass, field


@dataclass
class JobSlot:
    """Tracks a running job occupying a machine slot."""
    cpu_used: float
    mem_used: float
    priority: float
    ttl: int


class DatasetLoader(Protocol):
    """Interface that any dataset loader passed to ResourceAllocationEnv must satisfy."""
    def next_job(self) -> Tuple[np.ndarray, int]:
        """Return (job_features [cpu, mem, priority], duration_steps)."""
        ...

    def reset(self) -> None:
        """Reset the loader back to the start of the dataset."""
        ...


class ResourceAllocationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_machines: int = 4,
        cpu_capacity: float = 16.0,
        mem_capacity: float = 64.0,
        max_jobs_per_ep: int = 200,
        job_min_duration: int = 3,
        job_max_duration: int = 15,
        sla_breach_penalty: float = 2.0,
        seed: Optional[int] = None,
        dataset_loader: Optional[DatasetLoader] = None,
    ):
        super().__init__()

        assert n_machines >= 1, "n_machines must be >= 1."
        assert cpu_capacity > 0, "cpu_capacity must be positive."
        assert mem_capacity > 0, "mem_capacity must be positive."
        assert job_min_duration >= 1, "job_min_duration must be >= 1."
        assert job_max_duration >= job_min_duration, "job_max_duration must be >= job_min_duration."
        assert sla_breach_penalty >= 0, "Pass a positive magnitude; the sign is applied internally."

        self.n_machines = n_machines
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.max_jobs = max_jobs_per_ep
        self.job_min_duration = job_min_duration
        self.job_max_duration = job_max_duration
        self.sla_breach_penalty = sla_breach_penalty 

        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(n_machines)
        obs_dim = 3 * n_machines + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.cpu_free: np.ndarray = np.empty(0)
        self.mem_free: np.ndarray = np.empty(0)
        self.slots: list[list[JobSlot]] = []
        self.current_job: np.ndarray = np.empty(0)
        self.jobs_processed: int = 0
        self.total_reward: float = 0.0

        self._n_allocated: int = 0
        self._n_sla_breach: int = 0
        self._n_completed: int = 0

        self.dataset_loader = dataset_loader
        self.current_job_duration = 0

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.cpu_free = np.full(self.n_machines, self.cpu_capacity, dtype=np.float32)
        self.mem_free = np.full(self.n_machines, self.mem_capacity, dtype=np.float32)
        self.slots: list[list[JobSlot]] = [[] for _ in range(self.n_machines)]

        self.jobs_processed = 0
        self.total_reward = 0.0

        self._n_allocated = 0
        self._n_sla_breach = 0
        self._n_completed = 0

        self.current_job = self._generate_job()
        return self._obs(), {}

    def step(self, action: int):
        self._tick_jobs()

        job_cpu, job_mem, job_priority = self.current_job
        reward = 0.0
        info: dict = {}
        m = int(action)

        if m < 0 or m >= self.n_machines:
            raise ValueError(f"Invalid action {action}. Expected 0..{self.n_machines - 1}")

        if self.cpu_free[m] >= job_cpu and self.mem_free[m] >= job_mem:
            self.cpu_free[m] -= job_cpu
            self.mem_free[m] -= job_mem

            duration = self.current_job_duration

            self.slots[m].append(
                JobSlot(
                    cpu_used=job_cpu,
                    mem_used=job_mem,
                    priority=job_priority,
                    ttl=duration,
                )
            )

            cpu_util = 1.0 - (self.cpu_free[m] / self.cpu_capacity)
            mem_util = 1.0 - (self.mem_free[m] / self.mem_capacity)
            efficiency = (cpu_util + mem_util) / 2.0
            reward = job_priority * (0.5 + 0.5 * efficiency)

            # Packing bonus: reward tight fits that leave < 15% headroom on BOTH dimensions
            cpu_headroom = self.cpu_free[m] / self.cpu_capacity
            mem_headroom = self.mem_free[m] / self.mem_capacity
            if cpu_headroom < 0.15 and mem_headroom < 0.15:
                reward += 0.5 * job_priority

            # Balance penalty: discourage load imbalance across machines
            cpu_utils_all = 1.0 - (self.cpu_free / self.cpu_capacity)
            mem_utils_all = 1.0 - (self.mem_free / self.mem_capacity)
            balance_penalty = float(np.std(cpu_utils_all) + np.std(mem_utils_all))
            reward -= 0.3 * balance_penalty

            info["event"] = "allocated"
            info["machine"] = m
            info["duration"] = duration
            self._n_allocated += 1

        else:
            reward = -self.sla_breach_penalty
            info["event"] = "sla_breach"
            self._n_sla_breach += 1

        self.jobs_processed += 1
        self.total_reward   += reward
        done = self.jobs_processed >= self.max_jobs

        if not done:
            self.current_job = self._generate_job()

        return self._obs(), reward, done, False, info

    def _tick_jobs(self):
        """Decrement TTL on every running job; free resources on completion."""
        for m in range(self.n_machines):
            active_jobs = []
            for slot in self.slots[m]:
                slot.ttl -= 1
                if slot.ttl <= 0:
                    self.cpu_free[m] += slot.cpu_used
                    self.mem_free[m] += slot.mem_used
                    self.cpu_free[m] = min(self.cpu_free[m], self.cpu_capacity)
                    self.mem_free[m] = min(self.mem_free[m], self.mem_capacity)
                    self._n_completed += 1
                else:
                    active_jobs.append(slot)
            self.slots[m] = active_jobs

    def _generate_job(self) -> np.ndarray:
        if self.dataset_loader is not None:
            job_features, duration = self.dataset_loader.next_job()
            self.current_job_duration = duration
            return job_features
        else:
            cpu_req = float(self.rng.uniform(0.5, self.cpu_capacity * 0.4))
            mem_req = float(self.rng.uniform(1.0, self.mem_capacity * 0.4))
            priority = float(self.rng.choice([1.0, 2.0, 3.0], p=[0.50, 0.35, 0.15]))

            self.current_job_duration = int(
                self.rng.integers(self.job_min_duration, self.job_max_duration + 1)
            )
            return np.array([cpu_req, mem_req, priority], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        """Build the normalised observation vector.
        Layout: [cpu_free(n), mem_free(n), slot_count(n), job_cpu, job_mem, job_pri, progress]
        Total dim = 3*n_machines + 4
        """
        cpu_norm = self.cpu_free / self.cpu_capacity
        mem_norm = self.mem_free / self.mem_capacity

        # Slot count per machine (normalised by a reasonable max of 10)
        slot_counts = np.array(
            [len(self.slots[m]) / 10.0 for m in range(self.n_machines)],
            dtype=np.float32,
        )
        slot_counts = np.clip(slot_counts, 0.0, 1.0)

        job_cpu_norm = np.array(
            [self.current_job[0] / self.cpu_capacity], dtype=np.float32
        )
        job_mem_norm = np.array(
            [self.current_job[1] / self.mem_capacity], dtype=np.float32
        )
        # Priorities are sampled from {1.0, 2.0, 3.0}; normalise to [0.09, 1.0] range.
        job_pri_norm = np.array([self.current_job[2] / 3.0], dtype=np.float32)

        # Episode progress (0 → 1)
        progress = np.array(
            [self.jobs_processed / max(self.max_jobs, 1)], dtype=np.float32
        )

        return np.concatenate(
            [cpu_norm, mem_norm, slot_counts, job_cpu_norm, job_mem_norm, job_pri_norm, progress]
        ).astype(np.float32)

    def utilization(self) -> dict:
        """Current utilization — aggregate and per-machine."""
        cpu_util = 1.0 - self.cpu_free.mean() / self.cpu_capacity
        mem_util = 1.0 - self.mem_free.mean() / self.mem_capacity
        return {
            "cpu": float(cpu_util),
            "mem": float(mem_util),
            "cpu_per_machine": (1.0 - self.cpu_free / self.cpu_capacity).tolist(),
            "mem_per_machine": (1.0 - self.mem_free / self.mem_capacity).tolist(),
        }

    def episode_stats(self) -> dict:
        """Aggregate counters for the current episode so far."""
        total = max(self.jobs_processed, 1)
        return {
            "jobs_processed": self.jobs_processed,
            "allocated": self._n_allocated,
            "sla_breaches": self._n_sla_breach,
            "jobs_completed": self._n_completed,
            "placement_rate": self._n_allocated / total,
            "breach_rate": self._n_sla_breach / total,
            "total_reward": self.total_reward,
        }

    def render(self):
        u = self.utilization()
        job = self.current_job
        busy = sum(1 for machine_slots in self.slots if len(machine_slots) > 0)
        print(
            f"[Jobs {self.jobs_processed:4d}/{self.max_jobs}] "
            f"CPU={u['cpu']:.1%}  MEM={u['mem']:.1%}  "
            f"Busy machines={busy}/{self.n_machines} | "
            f"Next job: CPU={job[0]:.1f} MEM={job[1]:.1f} Pri={int(job[2])}"
        )
