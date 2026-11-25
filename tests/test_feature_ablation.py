import unittest
import numpy as np

from src.models.feature_ablation import column_profile, evaluate, score_spearman


class AblationTests(unittest.TestCase):
    def test_constant_score_correlation_is_missing(self):
        self.assertIsNone(score_spearman([1., 1.], [2., 2.]))
        self.assertAlmostEqual(score_spearman([1., 2.], [3., 4.]), 1.)

    def test_constant_and_near_constant_separated(self):
        matrix = np.array([[0., 0.]] * 99 + [[0., 1.]])
        profiles = column_profile(matrix, ["constant", "rare"])
        self.assertTrue(profiles["constant"]["constant"])
        self.assertFalse(profiles["constant"]["near_constant_nonconstant"])
        self.assertTrue(profiles["rare"]["near_constant_nonconstant"])

    def test_evaluation_excludes_training_and_handles_ties(self):
        result = evaluate({"train": 100., "test": 1., "known": 1.}, 1.,
                          {"test"}, {"known"})
        self.assertEqual(result["known_anomaly_rank_intervals"]["known"], (1, 2))
        self.assertEqual(result["candidate_alerts"], 0)
