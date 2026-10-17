"""
Focused tests for training/run_experiments.py (Task 2 - Anamika).

Two tests:
  1. test_eval_only_warning_is_printed  -- verifies the [WARNING] line is
     actually emitted to stdout when --eval-only is used. Mocks _run_eval_only,
     _save_csv, and _save_plot so no real checkpoint, dataset, or filesystem
     I/O is required.
  2. test_eval_only_aggregation_produces_zero_std -- characterisation test that
     documents the known limitation: replicating one result across N seed slots
     yields reward_std == 0.0. Calls _aggregate directly; no mocking needed.
"""

import sys
import os
import io
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import training.run_experiments as run_exp
from training.run_experiments import _aggregate, _SCHEDULERS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_eval_result(reward_mean=95.0):
    """
    Return a minimal eval_results dict shaped like the real one.
    Every key in _SCHEDULERS maps to a sub-dict with the four fields
    that _aggregate and _print_table read.
    """
    entry = {
        "reward_mean":         reward_mean,
        "reward_std":          2.0,
        "placement_rate_mean": 0.9,
        "breach_rate_mean":    0.1,
    }
    return {sched: dict(entry) for sched in _SCHEDULERS}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvalOnlyBehaviour(unittest.TestCase):

    def test_eval_only_warning_is_printed(self):
        """
        main() must print a [WARNING] line to stdout when --eval-only is used.

        The warning must mention:
          - "[WARNING]"        so it stands out visually
          - "variance metrics" so the user knows which numbers are affected
          - "0.0"              so the user knows the exact symptom

        _run_eval_only is replaced with a stub that returns fake data so no
        checkpoint file, trained model, or dataset is required.
        _save_csv and _save_plot are stubbed to prevent filesystem writes.
        """
        fake_result = _make_eval_result()

        with patch("sys.argv", ["run_experiments.py", "--eval-only",
                                 "--seeds", "42", "7"]), \
             patch.object(run_exp, "_run_eval_only", return_value=fake_result), \
             patch.object(run_exp, "_save_csv"), \
             patch.object(run_exp, "_save_plot"):

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                run_exp.main()

        output = buf.getvalue()

        self.assertIn(
            "[WARNING]", output,
            msg="--eval-only must print a [WARNING] line to stdout",
        )
        self.assertIn(
            "variance metrics", output,
            msg="Warning must mention 'variance metrics'",
        )
        self.assertIn(
            "0.0", output,
            msg="Warning must state that variance metrics will be 0.0",
        )

    def test_eval_only_aggregation_produces_zero_std(self):
        """
        Characterisation test: documents the known limitation that replicating
        one eval_results dict across N seed slots yields reward_std == 0.0.

        Calls _aggregate directly -- no mocking, no I/O.
        """
        fake_result = _make_eval_result(reward_mean=95.0)
        replicated = [fake_result] * 3   # mirrors [eval_results] * len(args.seeds)

        aggregated = _aggregate(replicated)

        for sched in _SCHEDULERS:
            if sched not in aggregated:
                continue
            self.assertEqual(
                aggregated[sched]["reward_std"], 0.0,
                msg=(
                    f"Scheduler '{sched}': reward_std must be 0.0 when the same "
                    "result is replicated across seeds (known --eval-only limitation)"
                ),
            )


if __name__ == "__main__":
    unittest.main()
