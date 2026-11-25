"""运行方式：python -m src.analysis.template_analysis"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.data_processor import LABELS, STRUCTURED
from src.data.openstack_config import OUTPUT_DIR

from .template_rules import LIMITATIONS, RULE_NOTES, normalize_message


LOGGER = logging.getLogger(__name__)
ANALYSIS = "openstack_template_analysis.csv"
STATISTICS = "openstack_template_statistics.csv"
REPORT = "openstack_template_report.json"
GROUPS = ("component", "severity", "module")
ADDED_FIELDS = ("normalized_message", "template_id")


def top_messages(messages, limit=10):
    return [{"message": message, "count": count} for message, count in messages.most_common(limit)]


def length_summary(lengths):
    count = sum(lengths.values())
    if not count:
        return {"count": 0, "histogram": {}}
    cumulative = 0
    percentiles = {}
    for length, frequency in sorted(lengths.items()):
        cumulative += frequency
        for percentile in (50, 90, 95, 99):
            if cumulative >= math.ceil(count * percentile / 100):
                percentiles.setdefault(f"p{percentile}", length)
    return {
        "count": count, "min": min(lengths), "max": max(lengths),
        "mean": round(sum(n * f for n, f in lengths.items()) / count, 3),
        **percentiles, "histogram": dict(sorted(lengths.items())),
    }


def instance_summary(events):
    events = sorted(events, key=lambda row: row["event_time"])
    if not events:
        return {"log_count": 0, "template_counts": {}, "minute_counts": {}}
    times = [datetime.fromisoformat(row["event_time"]) for row in events]
    return {
        "log_count": len(events),
        "first_event_time": events[0]["event_time"],
        "last_event_time": events[-1]["event_time"],
        "duration_seconds": (times[-1] - times[0]).total_seconds(),
        "minute_counts": dict(Counter(t.isoformat(sep=" ", timespec="minutes") for t in times)),
        "severity_counts": dict(Counter(r["severity"] for r in events)),
        "component_counts": dict(Counter(r["component"] for r in events)),
        "module_counts": dict(Counter(r["module"] for r in events)),
        "template_counts": dict(Counter(r["template_id"] for r in events)),
        "events": events,
    }


def collect(log_path, analysis_path, anomalous_uuids):
    raw_messages = defaultdict(Counter)
    lengths = Counter()
    component_lengths = defaultdict(Counter)
    parameters = Counter()
    component_parameters = defaultdict(Counter)
    group_counts = {field: Counter() for field in GROUPS}
    templates = {}
    template_counts = Counter()
    template_groups = defaultdict(lambda: {"component": Counter(), "severity": Counter()})
    instance_events = {uuid: [] for uuid in sorted(anomalous_uuids)}
    with (
        log_path.open(encoding="utf-8", newline="") as source,
        analysis_path.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        required = {*GROUPS, "message", "event_time", "instance_uuid", "source_file", "line_number"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("结构化日志 CSV 缺少必需字段")
        if set(ADDED_FIELDS).intersection(reader.fieldnames):
            raise ValueError("输入已包含模板结果字段")
        writer = csv.DictWriter(destination, [*reader.fieldnames, *ADDED_FIELDS])
        writer.writeheader()
        for row in reader:
            message = row["message"]
            component = row["component"]
            raw_messages[component][message] += 1
            lengths[len(message)] += 1
            component_lengths[component][len(message)] += 1
            for field in GROUPS:
                group_counts[field][row[field]] += 1

            normalized, kinds = normalize_message(message)
            parameters.update(kinds)
            component_parameters[component].update(kinds)
            # 使用正文模板的哈希，输入顺序和样本频次变化时 ID 保持稳定。
            template_id = "T_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            if templates.setdefault(template_id, normalized) != normalized:
                raise ValueError(f"模板 ID 冲突：{template_id}")
            template_counts[template_id] += 1
            for field, counter in template_groups[template_id].items():
                counter[row[field]] += 1
            writer.writerow(dict(row, normalized_message=normalized, template_id=template_id))
            if row["instance_uuid"] in instance_events:
                fields = ("source_file", "line_number", "event_time", *GROUPS, "message")
                instance_events[row["instance_uuid"]].append(
                    dict({field: row[field] for field in fields}, template_id=template_id)
                )

    all_messages = Counter()
    exploration = {
        "message_length_chars": length_summary(lengths),
        "group_counts": group_counts,
        "dynamic_parameter_log_counts": parameters,
        "components": {},
    }
    for component, messages in sorted(raw_messages.items()):
        all_messages.update(messages)
        exploration["components"][component] = {
            "unique_raw_messages": len(messages),
            "message_length_chars": length_summary(component_lengths[component]),
            "dynamic_parameter_log_counts": component_parameters[component],
            "top_raw_messages": top_messages(messages, 5),
        }
    exploration["unique_raw_messages"] = len(all_messages)
    exploration["top_raw_messages"] = top_messages(all_messages)
    instances = {uuid: instance_summary(events) for uuid, events in instance_events.items()}
    return exploration, templates, template_counts, template_groups, instances


def write_statistics(path, templates, counts, groups):
    total = counts.total()
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("template_id", "normalized_message", "frequency", "share",
                         "component_counts", "severity_counts"))
        for key in ordered:
            writer.writerow((key, templates[key], counts[key], counts[key] / total,
                             json.dumps(groups[key]["component"], sort_keys=True),
                             json.dumps(groups[key]["severity"], sort_keys=True)))

    def summaries(keys):
        return [{"template_id": key, "normalized_message": templates[key],
                 "frequency": counts[key]} for key in keys]

    group_distribution = {}
    for field in ("component", "severity"):
        distribution = defaultdict(dict)
        for key in ordered:
            for value, count in groups[key][field].items():
                distribution[value][key] = count
        group_distribution[field] = dict(sorted(distribution.items()))
    return {
        "total_logs": total, "total_templates": len(templates),
        "frequency_histogram": dict(sorted(Counter(counts.values()).items())),
        "singleton_templates": sum(count == 1 for count in counts.values()),
        "top_templates": summaries(ordered[:10]),
        "bottom_templates": summaries(sorted(counts, key=lambda key: (counts[key], key))[:10]),
        "distribution_by_group": group_distribution,
    }


def analyze(log_path: Path, labels_path: Path, output_dir: Path) -> dict:
    with labels_path.open(encoding="utf-8", newline="") as stream:
        labels = csv.DictReader(stream)
        if not labels.fieldnames or not {"instance_uuid", "anomaly_label"}.issubset(labels.fieldnames):
            raise ValueError("实例标签 CSV 缺少必需字段")
        anomalous_uuids = {row["instance_uuid"] for row in labels if row["anomaly_label"] == "anomalous"}
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (ANALYSIS, STATISTICS, REPORT)
    for name in outputs:
        if (output_dir / name).exists():
            raise FileExistsError(f"输出文件已存在，请使用新的输出目录：{output_dir / name}")
    with TemporaryDirectory(dir=output_dir) as temporary:
        staging = Path(temporary)
        exploration, templates, counts, groups, instances = collect(
            log_path, staging / ANALYSIS, anomalous_uuids
        )
        LOGGER.info("已归一化 %d 条日志", counts.total())
        report = {
            "exploration": exploration,
            "templates": write_statistics(staging / STATISTICS, templates, counts, groups),
            "anomalous_instances": instances,
            "label_scope": "instance",
            "normalization_rules": RULE_NOTES,
            "limitations": LIMITATIONS,
        }
        (staging / REPORT).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # 和解析模块相同，先生成完整结果，再以拒绝覆盖的方式发布。
        published = []
        try:
            for name in outputs:
                destination = output_dir / name
                os.link(staging / name, destination)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink()
            raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="分析 OpenStack 日志并提取规则模板")
    parser.add_argument("--logs", type=Path, default=OUTPUT_DIR / STRUCTURED)
    parser.add_argument("--labels", type=Path, default=OUTPUT_DIR / LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        report = analyze(args.logs, args.labels, args.output_dir)
    except (OSError, ValueError, csv.Error) as error:
        LOGGER.error("OpenStack 日志模板分析失败：%s", error)
        return 1
    LOGGER.info("已提取 %d 个模板，结果保存在 %s", report["templates"]["total_templates"], args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
