import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

_VALID_PRIORITIES = {1, 2, 3}


class BorgDatasetLoader:
    """
    Loads a synthetic cluster workload trace CSV produced by generate_dummy_trace.py.

    The CSV must have columns: cpu_req, mem_req, priority, duration.
    priority must be in {1, 2, 3} to match ResourceAllocationEnv's reward function.
    """

    def __init__(
        self, file_path: str, scale_cpu: float = 16.0, scale_mem: float = 64.0
    ):
        _log.info("Loading dataset from %s", file_path)
        self.data = pd.read_csv(file_path)

        # Validate priority values match what the env reward function expects
        unexpected = set(self.data["priority"].unique()) - _VALID_PRIORITIES
        if unexpected:
            _log.warning(
                "Dataset contains unexpected priority values %s. "
                "Expected only {1, 2, 3}. Regenerate with generate_dummy_trace.py.",
                unexpected,
            )

        # Scale fractional requests to absolute resource units
        self.data["cpu_req"] = self.data["cpu_req"] * scale_cpu
        self.data["mem_req"] = self.data["mem_req"] * scale_mem

        self.curr_indx = 0
        self.max_indx = len(self.data)
        _log.info(
            "Loaded %d jobs (cpu scaled x%.0f, mem scaled x%.0f)",
            self.max_indx, scale_cpu, scale_mem,
        )

    def next_job(self) -> tuple[np.ndarray, int]:
        if self.curr_indx >= self.max_indx:
            self.curr_indx = 0  # wrap around

        row = self.data.iloc[self.curr_indx]
        self.curr_indx += 1

        job_features = np.array(
            [row["cpu_req"], row["mem_req"], row["priority"]], dtype=np.float32
        )
        duration = int(row["duration"])
        return job_features, duration

    def reset(self) -> None:
        self.curr_indx = 0
