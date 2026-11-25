import unittest

from src.analysis.temporal_features import quantile, extract


class TemporalTests(unittest.TestCase):
    def test_quantiles_and_missing(self):
        self.assertEqual(quantile([0, 10], .9), 9)
        self.assertIsNone(quantile([], .9))
        self.assertEqual(quantile([3], .99), 3)

    def test_intervals_phases_and_parameters(self):
        def event(t, message, template):
            return {"event_time": f"2017-05-14 00:00:{t:02d}.000",
                    "message": message, "template_id": template, "source_file": "file",
                    "line_number": str(t), "component": "nova-compute", "severity": "INFO"}
        rows = [event(0, "Attempting claim: memory 2048 MB, disk 20 GB, vcpus 1 CPU", "a"),
                event(2, "Claim successful", "b"), event(8, "VM Started (Lifecycle Event)", "c")]
        values, evidence = extract(rows, ["a", "b", "c", "d"])
        self.assertEqual(values["duration_seconds"], 8)
        self.assertEqual(values["gap_mean_seconds"], 4)
        self.assertEqual(values["gap_median_seconds"], 4)
        self.assertEqual(values["gap_std_seconds"], 2)
        self.assertAlmostEqual(values["gap_p90_seconds"], 5.6)
        self.assertEqual(values["claim_seconds"], 2)
        self.assertEqual(evidence["claim_seconds"]["status"], "identified")
        self.assertIsNone(values["delete_seconds"])
        self.assertIsNone(values["retry_count_max"])
        self.assertEqual(values["requested_memory_MB_max"], 2048)
        self.assertEqual(values["count_d"], 0)
        self.assertIsNone(values["http_5xx_count"])
        one, _ = extract(rows[:1], ["a"])
        self.assertIsNone(one["gap_mean_seconds"])
        self.assertIsNone(one["claim_seconds"])
        duplicate, _ = extract(rows + [rows[1].copy()], ["a", "b", "c"])
        self.assertIsNone(duplicate["claim_seconds"])
