import unittest
import numpy as np

from env.resource_env import ResourceAllocationEnv, JobSlot


class DummyLoader:
    def __init__(self, jobs):
        self.jobs = jobs
        self.idx = 0

    def next_job(self):
        job = self.jobs[self.idx % len(self.jobs)]
        self.idx += 1
        return np.array(job[:3], dtype=np.float32), int(job[3])

    def reset(self):
        self.idx = 0


class TestResourceAllocationEnv(unittest.TestCase):
    def setUp(self):
        self.env = ResourceAllocationEnv(
            n_machines=4,
            cpu_capacity=16.0,
            mem_capacity=64.0,
            max_jobs_per_ep=10,
            sla_breach_penalty=2.5,
            seed=42,
        )

    def test_initialization_and_reset(self):
        obs, _ = self.env.reset(seed=123)
        self.assertEqual(len(self.env.cpu_free), 4)
        self.assertEqual(len(self.env.mem_free), 4)
        np.testing.assert_allclose(self.env.cpu_free, [16.0, 16.0, 16.0, 16.0])
        np.testing.assert_allclose(self.env.mem_free, [64.0, 64.0, 64.0, 64.0])
        self.assertEqual(len(self.env.slots), 4)
        for s in self.env.slots:
            self.assertEqual(len(s), 0)

        # Expected obs dim: 3 * 4 + 4 = 16
        self.assertEqual(obs.shape, (16,))
        self.assertTrue((obs >= 0.0).all() and (obs <= 1.0).all())

    def test_successful_allocation(self):
        # Provide deterministic job: [cpu, mem, priority, duration]
        loader = DummyLoader([[4.0, 16.0, 2.0, 5]])
        env = ResourceAllocationEnv(
            n_machines=4,
            cpu_capacity=16.0,
            mem_capacity=64.0,
            dataset_loader=loader,
            seed=42,
        )
        obs, _ = env.reset()
        obs, reward, done, _, info = env.step(1)

        self.assertEqual(info["event"], "allocated")
        self.assertEqual(info["machine"], 1)
        self.assertAlmostEqual(env.cpu_free[1], 12.0)
        self.assertAlmostEqual(env.mem_free[1], 48.0)
        self.assertEqual(len(env.slots[1]), 1)
        self.assertEqual(env.slots[1][0].ttl, 5)

        # Other machines unaffected
        for m in [0, 2, 3]:
            self.assertEqual(env.cpu_free[m], 16.0)
            self.assertEqual(env.mem_free[m], 64.0)

    def test_infeasible_allocation_sla_breach(self):
        # Job requiring more CPU than machine capacity
        loader = DummyLoader([[20.0, 16.0, 1.0, 3]])
        env = ResourceAllocationEnv(
            n_machines=2,
            cpu_capacity=16.0,
            mem_capacity=64.0,
            sla_breach_penalty=3.0,
            dataset_loader=loader,
        )
        env.reset()
        obs, reward, done, _, info = env.step(0)

        self.assertEqual(info["event"], "sla_breach")
        self.assertEqual(reward, -3.0)
        # Resources remain unallocated
        self.assertEqual(env.cpu_free[0], 16.0)
        self.assertEqual(env.mem_free[0], 64.0)
        self.assertEqual(len(env.slots[0]), 0)

    def test_job_ttl_countdown_and_resource_freeing(self):
        # 1 job with duration 2
        loader = DummyLoader([
            [4.0, 16.0, 1.0, 2],
            [1.0, 1.0, 1.0, 10],  # second job
            [1.0, 1.0, 1.0, 10],  # third job
        ])
        env = ResourceAllocationEnv(
            n_machines=2,
            cpu_capacity=16.0,
            mem_capacity=64.0,
            dataset_loader=loader,
        )
        loader.reset()
        env.reset()

        # Step 1: allocate job 1 to machine 0 with TTL=2
        env.step(0)
        self.assertAlmostEqual(env.cpu_free[0], 12.0)
        self.assertEqual(len(env.slots[0]), 1)
        self.assertEqual(env.slots[0][0].ttl, 2)

        # Step 2: _tick_jobs decrements TTL to 1; allocate job 2 to machine 1
        env.step(1)
        self.assertEqual(env.slots[0][0].ttl, 1)
        self.assertAlmostEqual(env.cpu_free[0], 12.0)

        # Step 3: _tick_jobs decrements TTL to 0 -> job 1 completes and frees resources!
        env.step(1)
        self.assertEqual(len(env.slots[0]), 0)
        self.assertAlmostEqual(env.cpu_free[0], 16.0)
        self.assertAlmostEqual(env.mem_free[0], 64.0)

    def test_packing_bonus_and_balance_penalty(self):
        # Test packing bonus condition: cpu_headroom < 0.15 AND mem_headroom < 0.15
        # For cpu_capacity=10.0, mem_capacity=10.0, job = 9.0, 9.0 (leaving 10% headroom)
        loader = DummyLoader([[9.0, 9.0, 2.0, 5]])
        env = ResourceAllocationEnv(
            n_machines=2,
            cpu_capacity=10.0,
            mem_capacity=10.0,
            dataset_loader=loader,
        )
        env.reset()
        _, reward, _, _, _ = env.step(0)

        # Machine 0 utilization = 0.9, 0.9; efficiency = 0.9
        # Machine 1 utilization = 0.0, 0.0
        # Base reward = 2.0 * (0.5 + 0.5 * 0.9) = 2.0 * 0.95 = 1.90
        # Packing bonus = 0.5 * 2.0 = 1.00
        # Balance penalty: cpu_utils = [0.9, 0.0], mem_utils = [0.9, 0.0]
        # std([0.9, 0.0]) = 0.45. Balance penalty = 0.3 * (0.45 + 0.45) = 0.27
        # Expected reward = 1.90 + 1.00 - 0.27 = 2.63
        expected_base = 2.0 * (0.5 + 0.5 * 0.9)
        expected_bonus = 1.0
        expected_penalty = 0.3 * (np.std([0.9, 0.0]) + np.std([0.9, 0.0]))
        expected_reward = expected_base + expected_bonus - expected_penalty

        self.assertAlmostEqual(reward, expected_reward, places=5)

    def test_invalid_action_raises(self):
        env = ResourceAllocationEnv(n_machines=2)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(-1)
        with self.assertRaises(ValueError):
            env.step(2)

    def test_episode_done_flag(self):
        max_jobs = 3
        env = ResourceAllocationEnv(n_machines=2, max_jobs_per_ep=max_jobs)
        env.reset()

        _, _, done1, _, _ = env.step(0)
        self.assertFalse(done1)
        _, _, done2, _, _ = env.step(0)
        self.assertFalse(done2)
        _, _, done3, _, _ = env.step(0)
        self.assertTrue(done3)


if __name__ == "__main__":
    unittest.main()
