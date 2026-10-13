"""
Tests for evaluation functions introduced in Task 3.

Covers:
  - evaluate_agent_pure runs and returns expected keys
  - evaluate_agent and evaluate_baseline surface placement_rate_mean / breach_rate_mean
  - evaluate_agent_pure never instantiates GreedyPriorityBaseline (no teacher)
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent
from baselines.heuristics import FirstFitBaseline
from training.train import evaluate_agent, evaluate_agent_pure, evaluate_baseline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'borg_trace_subset.csv'
)

_EXPECTED_KEYS = {
    'reward_mean', 'reward_std', 'cpu_util', 'mem_util',
    'cpu_per_machine', 'mem_per_machine',
    'placement_rate_mean', 'breach_rate_mean',
}


def _make_agent():
    """Minimal DQNAgent with n_machines=4 (obs_dim=16, n_actions=4)."""
    return DQNAgent(
        obs_dim=16,
        n_actions=4,
        n_machines=4,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100,
        batch_size=4,
        buffer_capacity=32,
    )


def _make_loader():
    from utils.data_loader import BorgDatasetLoader
    return BorgDatasetLoader(
        file_path=_DATA_PATH,
        scale_cpu=16.0,
        scale_mem=64.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvaluationFunctions(unittest.TestCase):

    def test_evaluate_agent_pure_runs_and_returns_expected_keys(self):
        """evaluate_agent_pure must complete and return all 8 expected keys."""
        agent = _make_agent()
        loader = _make_loader()
        result = evaluate_agent_pure(agent, loader, n_episodes=2, seed=42)
        self.assertIsInstance(result, dict)
        for key in _EXPECTED_KEYS:
            self.assertIn(key, result, msg=f"Missing key: {key}")

    def test_eval_functions_return_placement_breach_keys(self):
        """Both evaluate_agent and evaluate_baseline must return placement/breach keys in [0, 1]."""
        agent = _make_agent()
        loader = _make_loader()
        baseline = FirstFitBaseline()

        agent_result = evaluate_agent(agent, loader, n_episodes=2)
        baseline_result = evaluate_baseline(baseline, loader, n_episodes=2)

        for result, label in [(agent_result, "evaluate_agent"), (baseline_result, "evaluate_baseline")]:
            self.assertIn('placement_rate_mean', result, msg=f"{label} missing placement_rate_mean")
            self.assertIn('breach_rate_mean', result, msg=f"{label} missing breach_rate_mean")
            self.assertGreaterEqual(result['placement_rate_mean'], 0.0,
                                    msg=f"{label} placement_rate_mean < 0")
            self.assertLessEqual(result['placement_rate_mean'], 1.0,
                                  msg=f"{label} placement_rate_mean > 1")
            self.assertGreaterEqual(result['breach_rate_mean'], 0.0,
                                    msg=f"{label} breach_rate_mean < 0")
            self.assertLessEqual(result['breach_rate_mean'], 1.0,
                                  msg=f"{label} breach_rate_mean > 1")

    def test_evaluate_agent_pure_uses_no_teacher(self):
        """evaluate_agent_pure must never instantiate GreedyPriorityBaseline."""
        agent = _make_agent()
        loader = _make_loader()

        with patch('training.train.GreedyPriorityBaseline') as mock_greedy:
            evaluate_agent_pure(agent, loader, n_episodes=2)
            self.assertEqual(
                mock_greedy.call_count, 0,
                msg="GreedyPriorityBaseline should not be instantiated in evaluate_agent_pure"
            )


if __name__ == "__main__":
    unittest.main()
