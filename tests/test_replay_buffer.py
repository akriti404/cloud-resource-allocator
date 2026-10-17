
import os
import sys
import unittest
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils.replay_buffer import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer(unittest.TestCase):

    def test_push_stores_transition(self):
        buffer = PrioritizedReplayBuffer(capacity=8)
        state = np.array([1.0, 2.0], dtype=np.float32)
        next_state = np.array([2.0, 3.0], dtype=np.float32)

        buffer.push(state, 0, 1.0, next_state, False)

        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.tree.data[0][1], 0)
        self.assertAlmostEqual(buffer.tree.data[0][2], 1.0)

    def test_sample_returns_expected_shapes(self):
        buffer = PrioritizedReplayBuffer(capacity=8)

        for i in range(4):
            buffer.push(
                np.array([i, i + 1], dtype=np.float32),
                i % 2,
                float(i),
                np.array([i + 1, i + 2], dtype=np.float32),
                False,
            )

        states, actions, rewards, next_states, dones, indices, weights = (
            buffer.sample(4)
        )

        self.assertEqual(states.shape, (4, 2))
        self.assertEqual(actions.shape, (4,))
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(next_states.shape, (4, 2))
        self.assertEqual(dones.shape, (4,))
        self.assertEqual(indices.shape, (4,))
        self.assertEqual(weights.shape, (4,))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights > 0))
        self.assertLessEqual(float(weights.max()), 1.0 + 1e-6)

    def test_terminal_flag_is_preserved(self):
        buffer = PrioritizedReplayBuffer(capacity=8)
        buffer.push(
            np.array([1.0]),
            0,
            1.0,
            np.array([2.0]),
            True,
        )

        transition = buffer.tree.data[0]
        self.assertEqual(transition[4], 1.0)

    def test_priority_update_uses_absolute_td_error(self):
        buffer = PrioritizedReplayBuffer(capacity=8)
        buffer.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False)

        leaf_index = buffer.capacity - 1
        td_error = -2.0
        buffer.update_priorities(
            np.array([leaf_index]),
            np.array([td_error]),
        )

        expected = (abs(td_error) + 1e-5) ** buffer.alpha
        self.assertAlmostEqual(
            buffer.tree.tree[leaf_index], expected, places=6
        )

    def test_new_transition_uses_maximum_raw_td_error_priority(self):
        buffer = PrioritizedReplayBuffer(capacity=8)
        buffer.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False)

        leaf_index = buffer.capacity - 1
        buffer.update_priorities(
            np.array([leaf_index]),
            np.array([4.0]),
        )

        # A new transition should receive the maximum raw TD-error
        # priority transformed by alpha exactly once.
        buffer.push(np.array([3.0]), 0, 1.0, np.array([4.0]), False)

        new_leaf_index = buffer.capacity
        expected = (4.0 + 1e-5) ** buffer.alpha

        self.assertAlmostEqual(
            buffer.tree.tree[new_leaf_index], expected, places=6
        )
if __name__ == "__main__":
    unittest.main()
