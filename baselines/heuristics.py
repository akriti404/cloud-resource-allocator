import numpy as np


def _least_deficit_machine(env, job_cpu: float, job_mem: float) -> int:
    """Pick machine with the smallest combined CPU/MEM shortfall."""
    return min(
        range(env.n_machines),
        key=lambda m: max(0.0, job_cpu - env.cpu_free[m]) + max(0.0, job_mem - env.mem_free[m]),
    )

class FirstFitBaseline:
    name = "FirstFit"
    def select_action(self, obs: np.ndarray, env) -> int:
        job_cpu = env.current_job[0]
        job_mem = env.current_job[1]
        for m in range(env.n_machines):
            if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem:
                return m
        return _least_deficit_machine(env, job_cpu, job_mem)


class BestFitBaseline:
    name = "BestFit"
    def select_action(self, obs: np.ndarray, env) -> int:
        job_cpu = env.current_job[0]
        job_mem = env.current_job[1]
        best_m = None
        best_score = float("inf")
        for m in range(env.n_machines):
            if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem:
                leftover = (env.cpu_free[m] - job_cpu) + (env.mem_free[m] - job_mem)
                if leftover < best_score:
                    best_score = leftover
                    best_m = m
        return best_m if best_m is not None else _least_deficit_machine(env, job_cpu, job_mem)


class RandomBaseline:
    name = "Random"
    def select_action(self, obs: np.ndarray, env) -> int:
        return np.random.randint(env.n_machines)


class GreedyPriorityBaseline:
    name = "GreedyPriority"
    def select_action(self, obs: np.ndarray, env) -> int:
        job_cpu = env.current_job[0]
        job_mem = env.current_job[1]
        job_priority = env.current_job[2]

        candidates = [
            m for m in range(env.n_machines)
            if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem
        ]
        if not candidates:
            return _least_deficit_machine(env, job_cpu, job_mem)

        if job_priority >= 2.0:
            return min(candidates, key=lambda m: (env.cpu_free[m] - job_cpu) + (env.mem_free[m] - job_mem))
        else:
            return candidates[0]

class RoundRobinBaseline:
    name = "RoundRobin"
    def __init__(self):
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0

    def select_action(self, obs: np.ndarray, env) -> int:
        job_cpu = env.current_job[0]
        job_mem = env.current_job[1]
        
        for i in range(env.n_machines):
            m = (self.current_idx + i) % env.n_machines
            if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem:
                self.current_idx = (m + 1) % env.n_machines
                return m

        m = self.current_idx
        self.current_idx = (self.current_idx + 1) % env.n_machines
        return m