import unittest

from src.analysis.template_audit import compare_sequences, edit_distance, sequence_summary


class AuditTests(unittest.TestCase):
    def test_stable_time_order_and_intervals(self):
        rows = sequence_summary([
            {"event_time": "2017-05-14 00:00:01.250", "template_id": "b"},
            {"event_time": "2017-05-14 00:00:00.000", "template_id": "a"},
            {"event_time": "2017-05-14 00:00:01.250", "template_id": "c"},
        ])
        self.assertEqual([r["template_id"] for r in rows], ["a", "b", "c"])
        self.assertEqual([r["delta_seconds"] for r in rows], ["", 1.25, 0])

    def test_frequency_equality_does_not_imply_same_order(self):
        def events(order):
            return [{"template_id": token, "delta_seconds": 1} for token in order]
        result = compare_sequences({"x": events("ab"), "y": events("ba")})[0]
        self.assertTrue(result["identical_frequencies"])
        self.assertFalse(result["identical_order"])
        self.assertEqual(result["edit_distance"], 2)
        self.assertIsNone(result["aligned_gap_mae_seconds"])
        self.assertEqual(edit_distance([], ["a"]), 1)
