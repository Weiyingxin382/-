import json
import unittest
from unittest.mock import patch

import test_web_backend
from src.web.database import connect
from src.web.model_service import ModelService
from src.web.errors import AppError


class AnalysisApiTests(unittest.TestCase):
    write = test_web_backend.WebBackendTests.write
    get = test_web_backend.WebBackendTests.get

    def setUp(self):
        test_web_backend.WebBackendTests.setUp(self)
        with connect(self.settings.database) as db, db:
            instance = db.execute("SELECT id FROM instances").fetchone()[0]
            db.execute("INSERT INTO feature_sets VALUES(1,?,'test','{}','[]')", (self.dataset,))
            values = {"duration_seconds": 7199.999, "gap_mean_seconds": 7199.999,
                      "gap_median_seconds": 7199.999, "gap_max_seconds": 7199.999,
                      "gap_std_seconds": 0, "gap_p90_seconds": 7199.999}
            db.execute("INSERT INTO instance_features VALUES(?,1,?,?,?,?)",
                       (self.dataset, instance, json.dumps(values), "{}", '{"lifecycle_complete":true,"sources":{}}'))
            db.execute("""INSERT INTO model_registry VALUES(1,'test_p99','duration_p99','normal1',NULL,NULL,
                        '["duration_seconds"]','{}',45,'{}','{}')""")
            db.execute("INSERT INTO detection_runs VALUES(1,?,1,1,'experiment','succeeded','hash','{}',45)", (self.dataset,))
            db.execute("INSERT INTO instance_scores VALUES(1,?,?,7199.999,1,'test',1,1,1)", (self.dataset, instance))

    def tearDown(self):
        test_web_backend.WebBackendTests.tearDown(self)

    def test_templates_intersection_and_log_association(self):
        result = self.get("templates").json()["data"]
        self.assertEqual(result["pagination"]["total"], 1)
        self.assertEqual(result["items"][0]["frequency"], 3)
        filtered = self.get("templates", component="nova-api", severity="INFO").json()["data"]["items"][0]
        self.assertEqual(filtered["frequency"], 2)
        self.assertEqual(filtered["global_frequency"], 3)
        self.assertEqual(self.get("templates/summary").json()["data"]["log_count"], 3)
        self.assertEqual(self.get("templates/T_test/logs").json()["data"]["pagination"]["total"], 3)
        self.assertEqual(self.get("templates/missing").status_code, 404)
        self.assertEqual(self.get("templates", frequency_min=5, frequency_max=1).status_code, 422)

    def test_timeline_cross_page_gap_and_unlinked_logs_excluded(self):
        first = self.get("instances/" + test_web_backend.UUID + "/timeline", page_size=1).json()["data"]
        second = self.get("instances/" + test_web_backend.UUID + "/timeline", page_size=1, page=2).json()["data"]
        self.assertEqual(first["pagination"]["total"], 2)
        self.assertIsNone(first["items"][0]["delta_seconds"])
        self.assertEqual(second["items"][0]["sequence_index"], 2)
        self.assertAlmostEqual(second["items"][0]["delta_seconds"], 7199.999)

    def test_history_and_inference_scope(self):
        history = self.client.get("/api/v1/detection-runs/1/scores").json()["data"]
        self.assertEqual(history["items"][0]["rank_best"], 1)
        response = self.client.post("/api/v1/inferences", json={
            "dataset_id": self.dataset, "feature_set_id": 1,
            "instance_uuid": test_web_backend.UUID, "model_key": "test_p99"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["data"]["is_alert"])
        self.assertIsNone(response.json()["data"]["rank_best"])
        self.assertEqual(response.json()["data"]["instance_label_metadata"], "unknown")
        with connect(self.settings.database) as db, db:
            db.execute("UPDATE instance_features SET metadata_json=?", ('{"lifecycle_complete":false}',))
        self.assertEqual(self.client.post("/api/v1/inferences", json={
            "dataset_id": self.dataset, "feature_set_id": 1, "instance_uuid": test_web_backend.UUID,
            "model_key": "test_p99"}).status_code, 422)

    def test_unregistered_artifact_rejected(self):
        with self.assertRaises(AppError) as caught:
            ModelService().load({"method":"if_time", "artifact_path": str(self.root/"untrusted.pkl")})
        self.assertEqual(caught.exception.status, 503)

    def test_registered_missing_model_reports_unavailable(self):
        path = self.root / "models" / "isolation_forest" / "run_02" / "openstack_normal1_if_time.pkl"
        record = {"method": "if_time", "model_key": "missing", "training_batch": "normal1",
                  "artifact_path": str(path), "dependency_versions_json": "{}",
                  "feature_order_json": "[]", "threshold": .6, "source_json": "{}"}
        with patch("src.web.model_service.ROOT", self.root):
            result = ModelService().describe(record)
        self.assertEqual(result["state"], "unavailable")
        self.assertIsNotNone(result["unavailable_reason"])

    def test_inference_rejects_feature_source_mismatch(self):
        with connect(self.settings.database) as db, db:
            db.execute("UPDATE model_registry SET config_json=?", ('{"input_sha256":{"expected":"hash"}}',))
        response = self.client.post("/api/v1/inferences", json={
            "dataset_id": self.dataset, "feature_set_id": 1, "instance_uuid": test_web_backend.UUID,
            "model_key": "test_p99"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "FEATURE_VERSION_MISMATCH")

    def test_model_dependency_mismatch_is_diagnostic(self):
        record = {"method": "if_time", "model_key": "wrong_version", "training_batch": "normal1",
                  "artifact_path": str(self.root / "models/isolation_forest/run_02/openstack_normal1_if_time.pkl"),
                  "dependency_versions_json": '{"numpy":"0.0.0"}', "feature_order_json": "[]",
                  "threshold": .6, "source_json": "{}"}
        with patch("src.web.model_service.ROOT", self.root):
            result = ModelService().describe(record)
        self.assertEqual(result["state"], "unavailable")
        self.assertIn("numpy", result["unavailable_reason"]["reason"])
