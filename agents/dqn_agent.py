import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Sequence

from utils.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class RunningNormalizer:
    """
    Welford online algorithm for computing a running mean and variance.

    Tracks statistics across all rewards seen so far, giving a stable,
    consistent normalization signal regardless of batch composition.
    Unlike per-batch normalization, the same raw reward always maps to
    the same normalized value (given the global history), which keeps
    Bellman targets consistent across updates.
    """

    def __init__(self):
        self.count = 0
        self.mean  = 0.0
        self.M2    = 0.0   # sum of squared deviations from the mean

    def update(self, x: float):
        """Update running stats with a single new reward value."""
        self.count += 1
        delta      = x - self.mean
        self.mean += delta / self.count
        self.M2   += delta * (x - self.mean)

    @property
    def std(self) -> float:
        """Population std dev; returns 1.0 until at least 2 samples seen."""
        if self.count < 2:
            return 1.0
        return max((self.M2 / (self.count - 1)) ** 0.5, 1e-8)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "M2": self.M2}

    def load_state_dict(self, d: dict):
        self.count = d.get("count", 0)
        self.mean  = d.get("mean",  0.0)
        self.M2    = d.get("M2",    0.0)


class DuelingDQN(nn.Module):
    """
    Dueling DQN: separates state-value V(s) and advantage A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat      = self.shared(x)
        value     = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_machines: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        hidden: int = 256,
        device: Optional[str] = None,
    ):
        self.n_actions         = n_actions
        self.n_machines        = n_machines
        self.gamma             = gamma
        self.batch_size        = batch_size
        self.tau               = 0.005

        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.online_net = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss(reduction='none')

        self.memory            = PrioritizedReplayBuffer(buffer_capacity, alpha=0.6)
        self.reward_normalizer = RunningNormalizer()
        self.steps             = 0
        self.last_loss         = 0.0

    def _valid_actions_from_state(self, state: np.ndarray) -> np.ndarray:
        cpu_free = state[: self.n_machines]
        mem_free = state[self.n_machines : 2 * self.n_machines]
        job_cpu  = state[3 * self.n_machines]
        job_mem  = state[3 * self.n_machines + 1]

        feasible = [
            m for m in range(min(self.n_machines, self.n_actions))
            if cpu_free[m] >= job_cpu and mem_free[m] >= job_mem
        ]

        if self.n_actions > self.n_machines:
            feasible.extend(range(self.n_machines, self.n_actions))

        return np.array(feasible, dtype=np.int64) if feasible else np.arange(self.n_actions, dtype=np.int64)

    def _valid_action_mask_from_obs_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        cpu_free = obs_batch[:, : self.n_machines]
        mem_free = obs_batch[:, self.n_machines : 2 * self.n_machines]
        job_cpu  = obs_batch[:, 3 * self.n_machines].unsqueeze(1)
        job_mem  = obs_batch[:, 3 * self.n_machines + 1].unsqueeze(1)

        mask = (cpu_free >= job_cpu) & (mem_free >= job_mem)

        if self.n_actions > self.n_machines:
            extra_mask = torch.ones(
                (obs_batch.shape[0], self.n_actions - self.n_machines),
                dtype=torch.bool,
                device=obs_batch.device,
            )
            mask = torch.cat([mask, extra_mask], dim=1)

        no_valid = ~mask.any(dim=1)
        mask[no_valid] = True
        return mask

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[Sequence[int]] = None,
        explore: bool = True,
    ) -> int:
        if valid_actions is None:
            valid_actions_np = self._valid_actions_from_state(state)
        else:
            valid_actions_np = np.array(valid_actions, dtype=np.int64)
            if valid_actions_np.size == 0:
                valid_actions_np = np.arange(self.n_actions, dtype=np.int64)

        if explore and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions_np))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t).squeeze(0).cpu().numpy()

        masked_q = np.full(self.n_actions, -np.inf, dtype=np.float32)
        masked_q[valid_actions_np] = q_values[valid_actions_np]
        return int(np.argmax(masked_q))

    def store(self, state, action, reward, next_state, done):
        # Update running stats before the transition enters the buffer so
        # the normalizer always has at least as much history as the buffer.
        self.reward_normalizer.update(reward)
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)
        is_weights_t  = torch.FloatTensor(is_weights).to(self.device)

        rewards_t = torch.FloatTensor(
            [self.reward_normalizer.normalize(r) for r in rewards]
        ).to(self.device)

        current_q = (
            self.online_net(states_t)
            .gather(1, actions_t.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            next_online_q = self.online_net(next_states_t)
            next_valid_mask = self._valid_action_mask_from_obs_batch(next_states_t)
            next_online_q = next_online_q.masked_fill(~next_valid_mask, torch.finfo(next_online_q.dtype).min)
            next_actions = next_online_q.argmax(dim=1)
            next_q       = (
                self.target_net(next_states_t)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Element-wise loss weighted by importance-sampling weights
        elementwise_loss = self.loss_fn(current_q, target_q)
        loss = (is_weights_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update PER priorities with TD errors
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.steps  += 1

        # Polyak soft update of target network
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        self.last_loss = loss.item()
        return self.last_loss

    def save(self, path: str):
        torch.save(
            {
                "online":     self.online_net.state_dict(),
                "target":     self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "steps":      self.steps,
                "normalizer": self.reward_normalizer.state_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        if "normalizer" in ckpt:
            self.reward_normalizer.load_state_dict(ckpt["normalizer"])
        logger.info("Loaded checkpoint ← %s", path)