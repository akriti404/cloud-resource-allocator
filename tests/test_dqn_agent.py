"""
Regression tests for the epsilon warmup freeze fix (Task 2 - Anamika).

Verifies that DQNAgent._freeze_epsilon correctly prevents epsilon decay
during teacher-guided warmup, and that _warmup_with_teacher resets the
flag even when an exception is raised mid-warmup.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent
from env.resource_env import ResourceAllocationEnv
from baselines.heuristics import GreedyPriorityBaseline
from training.train import _warmup_with_teacher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(epsilon_start=1.0, epsilon_end=0.05, decay_steps=100):
    """
    Minimal DQNAgent for isolated epsilon tests.

    n_machines=2 => obs_dim = 3*2 + 4 = 10, matching ResourceAllocationEnv
    with n_machines=2 and the index offsets in _valid_action_mask_from_obs_batch.
    batch_size=4 and buffer_capacity=32 keep tests fast.
    """
    return DQNAgent(
        obs_dim=10,
        n_actions=2,
        n_machines=2,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=decay_steps,
        batch_size=4,
        buffer_capacity=32,
    )


def _push_transitions(agent, n=8):
    """Push n trivial transitions so the buffer exceeds batch_size."""
    rng = np.random.default_rng(0)
    for _ in range(n):
        s  = rng.random(10).astype(np.float32)
        ns = rng.random(10).astype(np.float32)
        agent.store(s, 0, 1.0, ns, False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEpsilonWarmupFreeze(unittest.TestCase):

    def test_freeze_flag_is_false_by_default(self):
        """Newly constructed agent must have _freeze_epsilon == False."""
        agent = _make_agent()
        self.assertFalse(agent._freeze_epsilon)

    def test_epsilon_unchanged_while_frozen(self):
        """update() must not decrement epsilon when _freeze_epsilon is True."""
        agent = _make_agent()
        _push_transitions(agent)
        initial_epsilon = agent.epsilon

        agent._freeze_epsilon = True
        for _ in range(20):
            agent.update()

        self.assertEqual(agent.epsilon, initial_epsilon)

    def test_epsilon_decays_normally_when_unfrozen(self):
        """
        update() must decrement epsilon at the correct per-step rate when
        _freeze_epsilon is False (the default), confirming the guard does not
        break the existing decay path.
        """
        agent = _make_agent(decay_steps=100)
        _push_transitions(agent)

        n_updates = 10
        for _ in range(n_updates):
            agent.update()

        expected = max(agent.epsilon_end, 1.0 - n_updates * agent.epsilon_decay)
        self.assertAlmostEqual(agent.epsilon, expected, places=6)

    def test_epsilon_preserved_across_warmup_boundary(self):
        """
        Simulate the full warmup -> main-training boundary.

        After warmup updates with _freeze_epsilon=True, epsilon must still
        equal epsilon_start (1.0). After main-training updates with the flag
        cleared, epsilon must have decayed by exactly that many steps.
        """
        decay_steps = 200
        agent = _make_agent(decay_steps=decay_steps)
        _push_transitions(agent, n=8)

        # --- warmup phase ---
        agent._freeze_epsilon = True
        for _ in range(50):
            agent.update()

        self.assertEqual(
            agent.epsilon, 1.0,
            msg="Epsilon must be 1.0 after warmup, got {}".format(agent.epsilon),
        )

        # --- main training phase ---
        agent._freeze_epsilon = False
        main_updates = 40
        for _ in range(main_updates):
            agent.update()

        expected = max(0.05, 1.0 - main_updates * agent.epsilon_decay)
        self.assertAlmostEqual(agent.epsilon, expected, places=6)

    def test_freeze_flag_reset_on_warmup_exception(self):
        """
        _warmup_with_teacher must reset _freeze_epsilon to False even when an
        exception is raised mid-warmup.

        Uses a real ResourceAllocationEnv (no dataset loader, built-in random
        job generator) and a real GreedyPriorityBaseline. agent.update() is
        monkeypatched on the instance to raise RuntimeError after 5 calls,
        forcing an exception inside the warmup loop. Confirms:
          1. RuntimeError propagates out of _warmup_with_teacher.
          2. _freeze_epsilon is False after the exception.
        """
        env = ResourceAllocationEnv(
            n_machines=2,
            cpu_capacity=16.0,
            mem_capacity=64.0,
            max_jobs_per_ep=20,
            seed=0,
        )
        teacher = GreedyPriorityBaseline()

        # obs_dim derived from env so it stays consistent with n_machines=2
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],  # 3*2+4 = 10
            n_actions=env.action_space.n,             # 2
            n_machines=2,
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=1000,
            batch_size=4,
            buffer_capacity=32,
        )

        # Monkeypatch update() on the instance only; raises after 5 calls
        call_count = {"n": 0}
        original_update = agent.update

        def update_that_raises():
            call_count["n"] += 1
            if call_count["n"] > 5:
                raise RuntimeError("simulated mid-warmup failure")
            return original_update()

        agent.update = update_that_raises

        with self.assertRaises(RuntimeError):
            _warmup_with_teacher(agent, env, teacher, warmup_episodes=5)

        self.assertFalse(
            agent._freeze_epsilon,
            msg="_freeze_epsilon must be False after warmup raises an exception",
        )


if __name__ == "__main__":
    unittest.main()
