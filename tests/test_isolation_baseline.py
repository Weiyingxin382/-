import unittest

from src.models.isolation_baseline import quality_reasons, rank_bounds
from src.analysis.temporal_features import TIME_FEATURES


class IsolationBaselineTests(unittest.TestCase):
    def test_installed_pipeline_components_fit_on_training_only(self):
        import numpy as np
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("variance", VarianceThreshold()),
                             ("scale", StandardScaler())])
        pipeline.fit(np.array([[1., 0.], [3., 0.], [np.nan, 0.]]))
        self.assertEqual(pipeline["impute"].statistics_.tolist(), [2., 0.])
        self.assertEqual(pipeline["scale"].mean_.tolist(), [2.])
        pipeline.transform(np.array([[999., 100.]]))
        self.assertEqual(pipeline["scale"].mean_.tolist(), [2.])

    def test_quality_does_not_use_candidate_similarity(self):
        row = dict.fromkeys(TIME_FEATURES, "1")
        row.update(instance_label_metadata="unknown", batch_metadata="openstack_normal1.log",
                   candidate_metadata="0", reference_similarity_descriptive="0")
        self.assertEqual(quality_reasons(row, {"observed_lifecycle_complete": True}), [])
        row["batch_metadata"] += "|openstack_normal2.log"
        self.assertIn("not_single_normal_collection_batch",
                      quality_reasons(row, {"observed_lifecycle_complete": True}))

    def test_tied_ranks_are_intervals(self):
        self.assertEqual(rank_bounds([3, 2, 2, 1], 2), (2, 3))

    def test_threshold_counts_are_json_serializable(self):
        import json
        import numpy as np
        from src.analysis.temporal_features import quantile
        threshold = float(quantile(list(np.array([1., 2., 3.])), .99))
        count = sum(score > threshold for score in (1., 4.))
        self.assertEqual(json.loads(json.dumps({"count": count})), {"count": 1})
