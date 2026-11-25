import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.analysis.template_analysis import ANALYSIS, STATISTICS, analyze
from src.analysis.template_rules import normalize_message


UUID = "544fd51c-4edc-4780-baae-ba1d80a0acfc"
UNKNOWN = "ae651dff-c7ad-43d6-ac96-bbcd820ccca8"


class TemplateAnalysisTests(unittest.TestCase):
    def test_api_dynamic_fields_and_static_semantics(self):
        message = (
            f'[req-{UUID}] 10.11.10.1 "GET /openstack/2013-10-17/meta_data.json '
            'HTTP/1.1" status: 200 len: 967 time: 0.001 s'
        )
        normalized, kinds = normalize_message(message)
        self.assertEqual(normalized, '[<REQ_ID>] <IPV4> "GET /openstack/2013-10-17/meta_data.json '
                         'HTTP/1.1" status: 200 len: <NUM> time: <NUM> s')
        self.assertEqual(kinds, {"request_id", "ipv4", "api_path", "number"})
        self.assertNotEqual(normalized, normalize_message(message.replace("200", "404"))[0])

    def test_path_uuid_timestamp_and_invalid_ipv4(self):
        message = f"2026-11-17T12:34:56.123Z /var/lib/nova/instances/{UUID}/disk 999.1.2.3 8 MB"
        normalized, kinds = normalize_message(message)
        self.assertEqual(normalized, "<TIMESTAMP> /var/lib/nova/instances/<UUID>/disk 999.1.2.3 <NUM> MB")
        self.assertEqual(kinds, {"timestamp", "file_path", "uuid", "number"})

    def test_deleted_instance_paths_share_template(self):
        for uuid in (UUID, UNKNOWN):
            path = f"Deletion of /var/lib/nova/instances/{uuid}_del complete"
            normalized, kinds = normalize_message(path)
            self.assertEqual(normalized, "Deletion of /var/lib/nova/instances/<UUID>_del complete")
            self.assertEqual(kinds, {"file_path", "uuid"})

    def test_percent_encoded_timestamp_preserves_query_structure(self):
        message = ('GET /v2/servers/detail?all_tenants=True'
                   '&changes-since=2017-05-16T11%3A34%3A58.666937%2B00%3A00 HTTP/1.1')
        normalized, kinds = normalize_message(message)
        self.assertEqual(normalized, 'GET /v2/servers/detail?all_tenants=True'
                         '&changes-since=<TIMESTAMP> HTTP/1.1')
        self.assertEqual(kinds, {"api_path", "timestamp"})

    def test_template_scope_and_statistics(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            logs = root / "logs.csv"
            labels = root / "labels.csv"
            fields = ("source_file", "line_number", "component", "severity", "module",
                      "event_time", "message", "instance_uuid", "raw_text")
            rows = [
                ("a.log", "1", "nova-api", "INFO", "nova.api", "2017-05-14 19:39:01.445",
                 "Took 12 seconds", UUID, "raw one"),
                ("b.log", "2", "nova-compute", "WARNING", "nova.compute", "2017-05-15 19:39:01.445",
                 "Took 9 seconds", UNKNOWN, "raw two"),
            ]
            with logs.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)
            labels.write_text(f"instance_uuid,anomaly_label\n{UUID},anomalous\n{UNKNOWN},unknown\n",
                              encoding="utf-8")
            report = analyze(logs, labels, root)
            with (root / ANALYSIS).open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, [*fields, "normalized_message", "template_id"])
                output = list(reader)
            self.assertEqual([tuple(row[field] for field in fields) for row in output], rows)
            template_id = output[0]["template_id"]
            self.assertEqual(output[1]["template_id"], template_id)
            self.assertEqual(report["templates"]["frequency_histogram"], {2: 1})
            self.assertEqual(report["templates"]["distribution_by_group"]["severity"]["INFO"], {template_id: 1})
            self.assertEqual(set(report["anomalous_instances"]), {UUID})
            self.assertEqual(report["anomalous_instances"][UUID]["template_counts"], {template_id: 1})
            with (root / STATISTICS).open(encoding="utf-8", newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["frequency"], "2")
