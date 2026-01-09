import unittest

from src.timing_utils import DEFAULT_TIMING_KEYS, TimingTracker


class TestTimingTracker(unittest.TestCase):
    def test_epoch_wall_sanity(self) -> None:
        tracker = TimingTracker()
        tracker.add("epoch_wall", 10.0)
        tracker.add("env_interaction", 3.0)
        tracker.add("imagination_rollout", 2.0)
        tracker.add("diffusion_sampling_teacher", 4.0)
        tracker.add("policy_value_update", 1.0)

        log = tracker.to_log(keys=DEFAULT_TIMING_KEYS)
        accounted = sum(
            log[f"timing/{key}_sec"] for key in DEFAULT_TIMING_KEYS if key != "epoch_wall"
        )
        residual = log["timing/epoch_wall_sec"] - accounted
        self.assertAlmostEqual(residual, 0.0, places=6)

    def test_counts(self) -> None:
        tracker = TimingTracker()
        tracker.add("env_interaction", 0.5)
        tracker.add("env_interaction", 0.25)
        log = tracker.to_log(keys=["env_interaction"])
        self.assertEqual(log["timing/env_interaction_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
