import numpy as np
from typing import Tuple


class SumTree:
    """Binary sum-tree for O(log n) prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with proportional prioritization."""

    def __init__(self, capacity: int = 50_000, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self._max_priority = 1.0
        self._beta = 0.4
        self._beta_increment = 1e-4

    def push(self, state, action, reward, next_state, done):
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple:
        indices = []
        priorities = []
        batch = []
        segment = self.tree.total() / batch_size

        self._beta = min(1.0, self._beta + self._beta_increment)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback: re-sample from beginning
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        states, actions, rewards, next_states, dones = zip(*batch)

        sampling_probs = np.array(priorities, dtype=np.float64) / self.tree.total()
        sampling_probs = np.clip(sampling_probs, 1e-8, None)
        is_weights = (self.tree.n_entries * sampling_probs) ** (-self._beta)
        is_weights /= is_weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            np.array(is_weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on absolute TD errors."""
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + 1e-5) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(int(idx), priority)

    def __len__(self):
        return self.tree.n_entries