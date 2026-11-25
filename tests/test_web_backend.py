import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient
from src.web.app import create_app
from src.web.config import Settings
from src.web.database import connect, initialize
from src.web.importer import create_job, perform_import, LOG_FIELDS


UUID = "544fd51c-4edc-4780-baae-ba1d80a0acfc"


class WebBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)
        self.settings = Settings(database=self.root / "app.db", input_dir=self.root,
                                 expected_logs=3, expected_instances=1, expected_templates=1)
        self.write("openstack_instance_labels.csv", ["instance_uuid", "anomaly_label"],
                   [{"instance_uuid": UUID, "anomaly_label": "unknown"}])
        self.write("openstack_template_statistics.csv",
                   ["template_id", "normalized_message", "frequency", "component_counts", "severity_counts"],
                   [{"template_id": "T_test", "normalized_message": "test", "frequency": 3,
                     "component_counts": '{"nova-api":2,"nova-compute":1}',
                     "severity_counts": '{"INFO":2,"ERROR":1}'}])
        self.rows = []
        for i in range(3):
            self.rows.append({
                "source_file": "first.log" if i < 2 else "second.log", "line_number": str(i+1),
                "raw_text": f"原文 {i} GET /servers/%literal_ {UUID}",
                "component": "nova-api" if i < 2 else "nova-compute",
                "event_time": "2017-05-16 12:00:00.001" if i < 2 else "2017-05-16 14:00:00.000",
                "process_id": "20", "severity": "INFO" if i < 2 else "ERROR",
                "module": "nova.test", "message": 'GET "quoted" %literal_ 中文',
                "instance_uuid": UUID if i != 1 else "",
            })
        self.write("openstack_structured.csv", sorted(LOG_FIELDS), self.rows)
        self.write("openstack_template_analysis.csv", [*sorted(LOG_FIELDS), "template_id", "normalized_message"],
                   [dict(row, template_id="T_test", normalized_message="test") for row in self.rows])
        initialize(self.settings.database)
        job, _ = create_job(self.settings, "first", self.settings.manifest_key, "test")
        perform_import(self.settings, job)
        self.client_context = TestClient(create_app(self.settings))
        self.client = self.client_context.__enter__()
        self.dataset = self.client.get("/api/v1/datasets").json()["data"]["items"][0]["id"]

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def write(self, filename, fields, rows):
        with (self.root / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)

    def get(self, route, **query):
        return self.client.get("/api/v1/" + route, params={"dataset_id": self.dataset, **query})

    def test_health_docs_initialization_and_duplicate_import(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)
        initialize(self.settings.database)
        job, _ = create_job(self.settings, "second", self.settings.manifest_key, "test")
        perform_import(self.settings, job)
        with connect(self.settings.database) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM logs").fetchone()[0], 3)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0], 1)
            self.assertFalse(db.execute("PRAGMA foreign_key_check").fetchall())

    def test_pagination_cursor_detail_and_filters(self):
        first = self.get("logs", page_size=1).json()["data"]
        second = self.get("logs", page_size=1, cursor=first["pagination"]["next_cursor"]).json()["data"]
        self.assertNotEqual(first["items"][0]["id"], second["items"][0]["id"])
        self.assertEqual(self.get("logs", severity="INFO", component="nova-api", start_time="2017-05-16T12:00:00", end_time="2017-05-16T13:00:00").json()["data"]["pagination"]["total"], 2)
        details = self.get("logs/" + str(first["items"][0]["id"])).json()["data"]
        self.assertEqual(details["raw_text"], self.rows[2]["raw_text"])
        self.assertEqual(details["instance_label_metadata"], "unknown")
        self.assertEqual(self.get("logs", cursor="broken").status_code, 422)
        self.assertEqual(self.get("logs", page_size=1, cursor=first["pagination"]["next_cursor"], severity="INFO").status_code, 422)

    def test_literal_keywords_errors_and_empty_results(self):
        for term in ('"quoted"', "%literal_", "中文", "GET", "G"):
            self.assertEqual(self.get("logs", q=term).json()["data"]["pagination"]["total"], 3, term)
        self.assertEqual(self.get("logs", q="' OR 1=1 --").json()["data"]["pagination"]["total"], 0)
        for params in ({"page_size": 101}, {"severity": "INVALID"}, {"start_time": "bad"},
                       {"start_time": "2017-05-16T00:00:00Z"}, {"cursor": "abc", "page": 1}):
            response = self.get("logs", **params)
            self.assertEqual(response.status_code, 422)
            self.assertIn("error", response.json())
        self.assertEqual(self.get("logs/99999").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/logs", params={"dataset_id":999}).status_code, 404)

    def test_overview_trend_and_options(self):
        overview = self.get("overview").json()["data"]
        self.assertEqual(overview["log_count"], 3)
        self.assertEqual(overview["unlinked_log_count"], 1)
        self.assertEqual(overview["instance_count"], 1)
        trend = self.get("overview/trend", bucket="hour").json()["data"]
        self.assertEqual([bucket["count"] for bucket in trend], [2, 0, 1])
        self.assertEqual(sum(x["count"] for x in self.get("overview/trend", severity="INFO").json()["data"]), 2)
        self.assertEqual(self.get("log-options").json()["data"]["severity"], {"ERROR":1,"INFO":2})

    def test_failed_import_rollback_and_idempotency_conflict(self):
        path = self.root / "openstack_structured.csv"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("invalid row\n")
        job, _ = create_job(self.settings, "bad", self.settings.manifest_key, "test")
        perform_import(self.settings, job)
        with connect(self.settings.database) as db:
            self.assertEqual(db.execute("SELECT state FROM import_jobs WHERE id=?", (job,)).fetchone()[0], "failed")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM logs").fetchone()[0], 3)
        response = self.client.post("/api/v1/imports", headers={"Idempotency-Key":"first"},
                                    json={"input_manifest_key":self.settings.manifest_key,"dataset_name":"different"})
        self.assertEqual(response.status_code, 409)

    def test_registered_import_route_and_repeated_key(self):
        payload = {"input_manifest_key": self.settings.manifest_key, "dataset_name": "test"}
        response = self.client.post("/api/v1/imports", headers={"Idempotency-Key": "http"}, json=payload)
        self.assertEqual(response.status_code, 202)
        self.assertIn("state", response.json()["data"])
        job = response.json()["data"]["id"]
        # TestClient 会等待任务在线程池中运行；通过 shutdown 验证任务完成再读取。
        self.client.app.state.executor.shutdown(wait=True)
        status = self.client.get(f"/api/v1/imports/{job}").json()["data"]
        self.assertEqual(status["state"], "succeeded")
        self.assertIsNone(status["error"])
        repeated = self.client.post("/api/v1/imports", headers={"Idempotency-Key": "http"}, json=payload)
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json()["data"]["id"], job)
