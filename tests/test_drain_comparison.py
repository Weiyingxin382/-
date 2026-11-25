import unittest

from src.analysis.drain_comparison import parameter_values, sample_indices


class DrainComparisonTests(unittest.TestCase):
    def test_raw_parameter_values_keep_error_and_resource_units_context(self):
        values = parameter_values("Bad response code while validating token: 503")
        self.assertEqual(values["error_code"], ["503"])
        values = parameter_values("memory 2048 MB, took 1.5 seconds status: 200")
        self.assertEqual(values["resource_value"], ["2048"])
        self.assertEqual(values["time_value"], ["1.5"])
        self.assertEqual(values["http_status"], ["200"])

    def test_sampling_is_repeatable_without_truth_labels(self):
        rows = [{"message": "VM stopped", "component": "nova-compute", "severity": "INFO"}] * 20
        first = sample_indices(rows, [1] * 20)
        self.assertEqual(first, sample_indices(rows, [1] * 20))
        self.assertTrue(all(index < 20 for index in first[0]))
