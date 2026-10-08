import numpy as np
from collections import deque


class RunningStats:
    def __init__(self, window: int = 100):
        self.window = window
        self.values = deque(maxlen=window)

    def push(self, val):
        self.values.append(val)

    @property
    def mean(self):
        return np.mean(self.values) if self.values else 0.0

    @property
    def std(self):
        return np.std(self.values) if self.values else 0.0

    def __repr__(self):
        return f"mean={self.mean:.3f} std={self.std:.3f}"


class EpisodeLogger:
    def __init__(self, print_every: int = 50):
        self.print_every = print_every
        self.ep = 0
        self.rewards = RunningStats(100)
        self.losses = RunningStats(100)
        self.epsilons = RunningStats(100)

    def log_episode(self, reward, loss, epsilon, extra: dict = None):
        self.ep += 1
        self.rewards.push(reward)
        self.losses.push(loss)
        self.epsilons.push(epsilon)
        if self.ep % self.print_every == 0:
            extra_str = ""
            if extra:
                extra_str = "  " + "  ".join(f"{k}={v:.3f}" for k, v in extra.items())
            print(
                f"Ep {self.ep:5d} | "
                f"Reward {self.rewards.mean:7.2f} ± {self.rewards.std:.2f} | "
                f"Loss {self.losses.mean:.4f} | "
                f"ε={self.epsilons.mean:.3f}"
                + extra_str
            )