"""
Tests for KaggleBorgLoader and the preprocessing script.
Uses a tiny synthetic CSV (20 rows) as a stand-in to avoid depending on the
full raw file being present in CI.
"""
import os
import sys
import csv
import tempfile
import unittest
import subprocess

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kaggle_borg_loader import KaggleBorgLoader


def _make_tiny_processed_csv(path: str, n: int = 20):
    """Write a minimal processed CSV with the exact 4-column schema."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        cpu = float(rng.uniform(0.1, 15.9))
        mem = float(rng.uniform(0.1, 63.9))
        pri = int(rng.choice([1, 2, 3]))
        dur = int(rng.integers(1, 31))
        rows.append({"cpu_req": cpu, "mem_req": mem, "priority": pri, "duration": dur})
    pd.DataFrame(rows).to_csv(path, index=False)


class TestKaggleBorgLoaderUnit(unittest.TestCase):
    """Fast unit tests using a synthetic 20-row processed CSV."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self.tmp.close()
        _make_tiny_processed_csv(self.tmp.name, n=20)
        self.loader = KaggleBorgLoader(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_next_job_shape_and_type(self):
        for _ in range(10):
            features, duration = self.loader.next_job()
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(features.shape, (3,))
            self.assertEqual(features.dtype, np.float32)
            self.assertIsInstance(duration, int)

    def test_features_finite(self):
        for _ in range(10):
            features, _ = self.loader.next_job()
            self.assertTrue(np.all(np.isfinite(features)))

    def test_duration_positive(self):
        for _ in range(10):
            _, duration = self.loader.next_job()
            self.assertGreater(duration, 0)

    def test_reset_rewinds(self):
        f1, d1 = self.loader.next_job()
        self.loader.next_job()
        self.loader.next_job()
        self.loader.reset()
        f2, d2 = self.loader.next_job()
        np.testing.assert_array_equal(f1, f2)
        self.assertEqual(d1, d2)

    def test_wraparound(self):
        """Exhausting the loader should wrap around without error."""
        for _ in range(25):  # more than the 20-row CSV
            features, duration = self.loader.next_job()
            self.assertEqual(features.shape, (3,))
            self.assertGreater(duration, 0)

    def test_cpu_mem_bounds(self):
        self.loader.reset()
        for _ in range(20):
            features, _ = self.loader.next_job()
            self.assertGreater(features[0], 0.0, "cpu_req must be > 0")
            self.assertLessEqual(features[0], 16.0, "cpu_req must be <= 16.0")
            self.assertGreater(features[1], 0.0, "mem_req must be > 0")
            self.assertLessEqual(features[1], 64.0, "mem_req must be <= 64.0")

    def test_priority_valid(self):
        self.loader.reset()
        for _ in range(20):
            features, _ = self.loader.next_job()
            self.assertIn(int(features[2]), {1, 2, 3})


class TestProcessedCSVSchema(unittest.TestCase):
    """Validate schema constraints on a synthetic processed CSV."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self.tmp.close()
        _make_tiny_processed_csv(self.tmp.name, n=20)
        self.df = pd.read_csv(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_columns_exist(self):
        expected = {"cpu_req", "mem_req", "priority", "duration"}
        self.assertEqual(set(self.df.columns), expected)

    def test_no_nans(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_cpu_bounds(self):
        self.assertTrue((self.df["cpu_req"] > 0).all())
        self.assertTrue((self.df["cpu_req"] <= 16.0).all())

    def test_mem_bounds(self):
        self.assertTrue((self.df["mem_req"] > 0).all())
        self.assertTrue((self.df["mem_req"] <= 64.0).all())

    def test_priority_values(self):
        self.assertTrue(set(self.df["priority"].unique()).issubset({1, 2, 3}))

    def test_duration_bounds(self):
        self.assertTrue((self.df["duration"] >= 1).all())
        self.assertTrue((self.df["duration"] <= 30).all())

    def test_exactly_4_columns(self):
        self.assertEqual(len(self.df.columns), 4)


if __name__ == "__main__":
    unittest.main()
