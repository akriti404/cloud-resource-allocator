"""
KaggleBorgLoader — loads the preprocessed Borg jobs CSV produced by
scripts/preprocess_borg_trace.py.

Public interface is identical to BorgDatasetLoader in utils/data_loader.py.
The processed CSV already stores cpu_req and mem_req in absolute environment
units (cpu * 16.0, mem * 64.0), so no additional scaling is applied.
"""
import logging
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)
_VALID_PRIORITIES = {1, 2, 3}


class KaggleBorgLoader:
    """
    Loads the preprocessed Kaggle Borg 2019 cluster-sample workload.

    The CSV must have columns: cpu_req, mem_req, priority, duration
    produced by scripts/preprocess_borg_trace.py.
    cpu_req and mem_req are already in absolute environment units
    (NOT fractions — no further scaling is applied).
    """

    def __init__(
        self,
        csv_path: str,
        cpu_capacity: float = 16.0,
        mem_capacity: float = 64.0,
    ):
        _log.info("Loading Kaggle Borg dataset from %s", csv_path)
        self.data = pd.read_csv(csv_path)
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity

        unexpected = set(self.data["priority"].unique()) - _VALID_PRIORITIES
        if unexpected:
            _log.warning(
                "Dataset contains unexpected priority values %s. Expected only {1,2,3}.",
                unexpected,
            )

        self.curr_indx = 0
        self.max_indx = len(self.data)
        print(f"[KaggleBorgLoader] Loaded {self.max_indx} preprocessed Borg jobs from {csv_path}")

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
