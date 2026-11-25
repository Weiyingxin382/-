import unittest

from src.analysis.instance_feasibility import compare, summarize, MARKERS


class FeasibilityTests(unittest.TestCase):
    def test_cross_file_records_form_one_sequence(self):
        rows = [{"event_time": f"2017-05-14 00:00:0{i}.000", "template_id": str(i),
                 "message": text, "component": "nova-compute",
                 "source_file": "normal1" if i < 3 else "normal2", "line_number": str(i)}
                for i, text in enumerate(MARKERS.values())]
        data = summarize(list(reversed(rows)))
        self.assertEqual(data["log_count"], 7)
        self.assertEqual(data["duration_seconds"], 6)
        self.assertTrue(data["observed_lifecycle_complete"])
        self.assertEqual(len(data["source_counts"]), 2)
        self.assertTrue(compare(data, data)["same_order"])
        self.assertEqual(compare(data, data)["aligned_gap_mae_seconds"], 0)

    def test_frequency_match_does_not_prove_order_or_completeness(self):
        base = {"template_counts": {"a": 1, "b": 1}, "component_counts": {"x": 2},
                "duration_seconds": 2, "event_gaps_seconds": [2]}
        result = compare(dict(base, template_order=["a", "b"]),
                         dict(base, template_order=["b", "a"]))
        self.assertTrue(result["same_frequencies"])
        self.assertFalse(result["same_order"])
        self.assertIsNone(result["aligned_gap_mae_seconds"])
