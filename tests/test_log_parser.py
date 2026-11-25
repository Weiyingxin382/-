import csv
from io import BytesIO
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from src.data.data_processor import FAILURES, LABELS, STRUCTURED, process_archive
from src.data.log_parser import parse_log_line
from src.data.openstack_config import LOG_FILES


VM = "3edec1e4-9678-4a3a-a21b-a145a4ee5e61"
ANOMALOUS_VM = "544fd51c-4edc-4780-baae-ba1d80a0acfc"
REQUEST = "[req-e285b551-587f-4c1d-8eba-dceb2673637f - - - - -]"
HEADER = "nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:02.007 2931 INFO nova.virt.libvirt.driver "


class LogParserTests(unittest.TestCase):
    def test_extract_fields(self):
        message = f"{REQUEST} [instance: {VM}] Creating image"
        self.assertEqual(parse_log_line(HEADER + message), {
            "component": "nova-compute",
            "event_time": "2017-05-14 19:39:02.007",
            "process_id": 2931,
            "severity": "INFO",
            "module": "nova.virt.libvirt.driver",
            "message": message,
            "instance_uuid": VM,
        })

    def test_rotated_file_and_api_instance(self):
        header = HEADER.replace(".log.", ".log.1.")
        self.assertEqual(parse_log_line(header + f'{REQUEST} "GET /v2/project/servers/{VM}"')["instance_uuid"], VM)

    def test_ignore_request_id_and_ambiguous_instances(self):
        self.assertEqual(parse_log_line(HEADER + REQUEST)["instance_uuid"], "")
        self.assertEqual(parse_log_line(HEADER + f"[instance: {VM}] for instance {ANOMALOUS_VM}")["instance_uuid"], "")
        self.assertIsNone(parse_log_line("malformed line"))

    def test_archive_preserves_lines_and_label_scope(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            archive_path = root / "sample.tar.gz"
            known = f"{HEADER}[instance: {ANOMALOUS_VM}] VM Stopped"
            unknown = f"{HEADER}[instance: {VM}] Creating image"
            members = {
                "anomaly_labels.txt": f"Header\n\n{ANOMALOUS_VM}\n".encode(),
                LOG_FILES[0]: f"{known}\r\nunparsed\n{unknown}".encode(),
                LOG_FILES[1]: b"\xff\n",
                LOG_FILES[2]: b"",
            }
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, content in members.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    archive.addfile(member, BytesIO(content))

            report = process_archive(archive_path, root)
            self.assertEqual(report["total"]["total_lines"], 4)
            self.assertEqual(report["total"]["parsed_lines"], 2)
            self.assertEqual(report["anomaly_uuid_log_counts"], {ANOMALOUS_VM: 1})
            with (root / STRUCTURED).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
                self.assertEqual([(row["line_number"], row["raw_text"]) for row in rows],
                                 [("1", known), ("3", unknown)])
                self.assertNotIn("anomaly_label", rows[0])
            with (root / LABELS).open(encoding="utf-8", newline="") as stream:
                labels = {row["instance_uuid"]: row["anomaly_label"] for row in csv.DictReader(stream)}
                self.assertEqual(labels, {ANOMALOUS_VM: "anomalous", VM: "unknown"})
            with (root / FAILURES).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
                self.assertEqual(rows[0]["line_number"], "2")
                self.assertEqual(rows[0]["raw_text"], "unparsed")
                self.assertEqual(rows[1]["original_bytes_hex"], "ff")
            with self.assertRaises(FileExistsError):
                process_archive(archive_path, root)
