import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from generate_dummy_trace import create_synthetic_trace, _PRIORITY_TIERS, _PRIORITY_PROBS
from utils.data_loader import BorgDatasetLoader


class TestWorkloadGeneration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = os.path.join(self.temp_dir.name, "test_trace.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_trace_schema_and_ranges(self):
        num_rows = 500
        create_synthetic_trace(filename=self.trace_file, num_rows=num_rows, seed=123)

        self.assertTrue(os.path.exists(self.trace_file))
        df = pd.read_csv(self.trace_file)

        # Expected columns
        expected_cols = {"cpu_req", "mem_req", "priority", "duration"}
        self.assertEqual(set(df.columns), expected_cols)
        self.assertEqual(len(df), num_rows)

        # Value bounds
        self.assertTrue((df["cpu_req"] >= 0.01).all() and (df["cpu_req"] <= 0.50).all())
        self.assertTrue((df["mem_req"] >= 0.01).all() and (df["mem_req"] <= 0.50).all())
        self.assertTrue((df["priority"].isin([1, 2, 3])).all())
        self.assertTrue((df["duration"] >= 1).all() and (df["duration"] <= 30).all())

    def test_trace_reproducibility(self):
        file1 = os.path.join(self.temp_dir.name, "trace1.csv")
        file2 = os.path.join(self.temp_dir.name, "trace2.csv")

        create_synthetic_trace(filename=file1, num_rows=200, seed=42)
        create_synthetic_trace(filename=file2, num_rows=200, seed=42)

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_data_loader_scaling_and_iteration(self):
        create_synthetic_trace(filename=self.trace_file, num_rows=20, seed=99)
        scale_cpu = 16.0
        scale_mem = 64.0
        loader = BorgDatasetLoader(self.trace_file, scale_cpu=scale_cpu, scale_mem=scale_mem)

        self.assertEqual(loader.max_indx, 20)
        self.assertEqual(loader.curr_indx, 0)

        # First job
        job_features, duration = loader.next_job()
        self.assertEqual(len(job_features), 3)
        self.assertIsInstance(duration, int)
        self.assertEqual(loader.curr_indx, 1)

        # Scale check: raw values in trace were <= 0.50, so scaled should be <= 8.0 and <= 32.0
        self.assertLessEqual(job_features[0], scale_cpu * 0.50 + 1e-5)
        self.assertLessEqual(job_features[1], scale_mem * 0.50 + 1e-5)

        # Iterate past max_indx to test wraparound
        for _ in range(19):
            loader.next_job()
        self.assertEqual(loader.curr_indx, 20)

        # 21st read should wrap around to index 1
        _, _ = loader.next_job()
        self.assertEqual(loader.curr_indx, 1)

        # Reset
        loader.reset()
        self.assertEqual(loader.curr_indx, 0)


if __name__ == "__main__":
    unittest.main()
