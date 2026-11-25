"""运行方式：python -m src.data.data_processor"""

import argparse
import csv
import json
import logging
import os
from collections import Counter
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from .label_loader import load_anomaly_uuids
from .log_parser import PARSED_FIELDS, parse_log_line
from .openstack_config import ARCHIVE_PATH, LOG_FILES, OUTPUT_DIR


LOGGER = logging.getLogger(__name__)
STRUCTURED, FAILURES, LABELS, REPORT = (
    "openstack_structured.csv",
    "openstack_unparsed.csv",
    "openstack_instance_labels.csv",
    "openstack_parse_report.json",
)
TRACE_FIELDS = ("source_file", "line_number", "raw_text")


def process_log(stream, source_file, structured, failures, instances):
    counts = Counter()
    parsed = 0
    total = 0
    for total, raw_line in enumerate(stream, start=1):
        raw_line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
        fields = None
        original_bytes_hex = ""
        try:
            raw_text = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            reason = "invalid_utf8"
            raw_text = raw_line.decode("utf-8", errors="backslashreplace")
            original_bytes_hex = raw_line.hex()
        else:
            fields = parse_log_line(raw_text)
            reason = "invalid_log_header"
        trace = dict(source_file=source_file, line_number=total, raw_text=raw_text)
        if fields is None:
            failures.writerow(dict(trace, reason=reason, original_bytes_hex=original_bytes_hex))
            counts[reason] += 1
            continue
        structured.writerow(dict(trace, **fields))
        parsed += 1
        if fields["instance_uuid"]:
            instances[fields["instance_uuid"]] += 1
    return {
        "total_lines": total,
        "parsed_lines": parsed,
        "unparsed_lines": total - parsed,
        "success_rate": parsed / total if total else None,
        "failure_reasons": dict(counts),
    }


def process_archive(archive_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (STRUCTURED, FAILURES, LABELS, REPORT)
    for filename in outputs:
        if (output_dir / filename).exists():
            raise FileExistsError(f"输出文件已存在，请使用新的输出目录：{output_dir / filename}")

    # 先完成全部文件，再发布结果。硬链接创建目标时会拒绝覆盖同名文件。
    with TemporaryDirectory(dir=output_dir) as temporary:
        staging = Path(temporary)
        instances = Counter()
        report = {"files": {}, "label_scope": "instance"}
        with (
            tarfile.open(archive_path, "r:gz") as archive,
            (staging / STRUCTURED).open("w", encoding="utf-8", newline="") as structured_file,
            (staging / FAILURES).open("w", encoding="utf-8", newline="") as failures_file,
        ):
            anomaly_uuids = load_anomaly_uuids(archive)
            structured = csv.DictWriter(structured_file, TRACE_FIELDS + PARSED_FIELDS)
            failures = csv.DictWriter(failures_file, TRACE_FIELDS + ("reason", "original_bytes_hex"))
            structured.writeheader()
            failures.writeheader()
            for source_file in LOG_FILES:
                member = archive.getmember(source_file)
                if not member.isfile():
                    raise ValueError(f"{source_file} 必须是普通文件")
                with archive.extractfile(member) as stream:
                    stats = process_log(stream, source_file, structured, failures, instances)
                report["files"][source_file] = stats
                LOGGER.info("%s: parsed %d/%d", source_file, stats["parsed_lines"], stats["total_lines"])

        with (staging / LABELS).open("w", encoding="utf-8", newline="") as label_file:
            writer = csv.writer(label_file)
            writer.writerow(("instance_uuid", "anomaly_label"))
            for uuid in sorted(instances.keys() | anomaly_uuids):
                writer.writerow((uuid, "anomalous" if uuid in anomaly_uuids else "unknown"))
        report["anomaly_uuid_log_counts"] = {uuid: instances[uuid] for uuid in sorted(anomaly_uuids)}
        report["total"] = {
            field: sum(stats[field] for stats in report["files"].values())
            for field in ("total_lines", "parsed_lines", "unparsed_lines")
        }
        total = report["total"]
        total["success_rate"] = total["parsed_lines"] / total["total_lines"] if total["total_lines"] else None
        (staging / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        published = []
        try:
            for filename in outputs:
                destination = output_dir / filename
                os.link(staging / filename, destination)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink()
            raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="将 OpenStack 压缩日志转换为结构化 CSV。")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        process_archive(args.archive, args.output_dir)
    except (OSError, tarfile.TarError, EOFError, KeyError, ValueError) as error:
        LOGGER.error("OpenStack 日志处理失败：%s", error)
        return 1
    LOGGER.info("处理结果已保存至 %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
