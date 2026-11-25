"""标准库实例时序特征及固定 P99 时长基线。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from src.analysis.template_audit import sequence_summary
from src.data.openstack_config import OUTPUT_DIR


EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
OUTPUT_NAMES = ("instance_features.csv", "temporal_feature_report.json",
                "temporal_baseline_results.csv", "temporal_analysis.md")
QUANTILE = 0.99
EVENTS = {
    "claim_attempt": "Attempting claim:",
    "claim_success": "Claim successful",
    "create": "Creating image",
    "start": "VM Started (Lifecycle Event)",
    "pause": "VM Paused (Lifecycle Event)",
    "resume": "VM Resumed (Lifecycle Event)",
    "stop": "VM Stopped (Lifecycle Event)",
    "delete_start": "Deleting instance files ",
    "delete_end": "_del complete",
}
PHASES = {
    "claim_seconds": ("claim_attempt", "claim_success"),
    "create_to_start_seconds": ("create", "start"),
    "pause_to_resume_seconds": ("pause", "resume"),
    "start_to_stop_seconds": ("start", "stop"),
    "delete_seconds": ("delete_start", "delete_end"),
}
TIME_FEATURES = ("duration_seconds", "gap_mean_seconds", "gap_median_seconds",
                 "gap_max_seconds", "gap_std_seconds", "gap_p90_seconds")
PARAMETERS = {
    "requested_memory_MB": r"Attempting claim: memory ([0-9.]+) MB",
    "requested_disk_GB": r"Attempting claim:.*disk ([0-9.]+) GB",
    "requested_vcpus_CPU": r"Attempting claim:.*vcpus ([0-9.]+) CPU",
    "total_memory_MB": r"Total memory: ([0-9.]+) MB",
    "used_memory_MB": r"Total memory:.*used: ([0-9.]+) MB",
    "memory_limit_MB": r"memory limit: ([0-9.]+) MB",
    "free_memory_MB": r"memory limit:.*free: ([0-9.]+) MB",
    "total_disk_GB": r"Total disk: ([0-9.]+) GB",
    "used_disk_GB": r"Total disk:.*used: ([0-9.]+) GB",
    "total_vcpu_VCPU": r"Total vcpu: ([0-9.]+) VCPU",
    "used_vcpu_VCPU": r"Total vcpu:.*used: ([0-9.]+) VCPU",
    "http_response_time_seconds": r"\btime:\s*([0-9.]+)",
    "retry_count": r"\bretr(?:y|ies)\s*(?:count\s*)?[:=]\s*(\d+)",
}


def quantile(values, probability):
    """线性插值分位数，位置为 (n-1)*p。"""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": min(values), "max": max(values),
        "mean": statistics.mean(values), "median": statistics.median(values),
        "std_population": statistics.pstdev(values),
        "p10": quantile(values, .1), "p90": quantile(values, .9),
        "p95": quantile(values, .95), "p99": quantile(values, .99),
        "iqr": quantile(values, .75) - quantile(values, .25),
    }


def extract(rows, template_ids):
    events = sequence_summary(rows)
    gaps = [row["delta_seconds"] for row in events[1:]]
    order = [r["template_id"] for r in events]
    counts = Counter(order)
    features = {
        "log_count": len(events), "template_count": len(counts),
        "sequence_length": len(order),
        "template_transition_count": max(0, len(order) - 1),
        "template_change_count": sum(a != b for a, b in zip(order, order[1:])),
        "duration_seconds": events[-1]["elapsed_seconds"],
        "gap_mean_seconds": statistics.mean(gaps) if gaps else None,
        "gap_median_seconds": statistics.median(gaps) if gaps else None,
        "gap_max_seconds": max(gaps) if gaps else None,
        "gap_std_seconds": statistics.pstdev(gaps) if gaps else None,
        "gap_p90_seconds": quantile(gaps, .9),
    }
    for component in ("nova-api", "nova-compute", "nova-scheduler"):
        features["component_" + component] = sum(r["component"] == component for r in events)
    for severity in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE", "WARN", "FATAL"):
        features["severity_" + severity] = sum(r["severity"] == severity for r in events)
    for template in template_ids:
        features["count_" + template] = counts[template]
    phase_evidence = {}
    occurrences = {name: [r for r in events if text in r["message"]] for name, text in EVENTS.items()}
    for phase, (start, end) in PHASES.items():
        features[phase] = None
        evidence = {"start_match": EVENTS[start], "end_match": EVENTS[end],
                    "start_count": len(occurrences[start]), "end_count": len(occurrences[end])}
        if len(occurrences[start]) == len(occurrences[end]) == 1:
            a, b = occurrences[start][0], occurrences[end][0]
            elapsed = b["elapsed_seconds"] - a["elapsed_seconds"]
            evidence["endpoints"] = [{k: r[k] for k in ("template_id", "event_time", "source_file", "line_number")} for r in (a, b)]
            if elapsed >= 0:
                features[phase] = elapsed
                evidence["status"] = "identified"
            else:
                evidence["status"] = "reversed_time"
        else:
            evidence["status"] = "missing_or_multiple_endpoints"
        phase_evidence[phase] = evidence
    for name, pattern in PARAMETERS.items():
        values = [float(value) for row in events for value in re.findall(pattern, row["message"], flags=re.I)]
        for suffix, value in (("min", min(values) if values else None),
                              ("max", max(values) if values else None),
                              ("mean", statistics.mean(values) if values else None)):
            features[name + "_" + suffix] = value
    codes = Counter(code for row in events for code in re.findall(r"\bstatus:\s*([1-5]\d{2})\b", row["message"]))
    for group in range(1, 6):
        # 无响应记录时保留缺失，有响应时未出现某类可明确计零。
        features[f"http_{group}xx_count"] = sum(n for code, n in codes.items() if code.startswith(str(group))) if codes else None
    return features, phase_evidence


def run(inputs, feasibility, output):
    paths = [inputs / name for name in ("openstack_structured.csv", "openstack_template_analysis.csv", "openstack_instance_labels.csv")]
    paths += [feasibility / "instance_distribution_report.json", feasibility / "candidate_reference_instances.csv"]
    hashes = {}
    for path in paths:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    with paths[2].open(encoding="utf-8", newline="") as stream:
        labels = {r["instance_uuid"]: r["anomaly_label"] for r in csv.DictReader(stream)}
    with paths[4].open(encoding="utf-8", newline="") as stream:
        candidates = {r["instance_uuid"]: r for r in csv.DictReader(stream)}
    prior = json.loads(paths[3].read_text(encoding="utf-8"))
    groups = defaultdict(list)
    request_only = 0
    unassigned = 0
    template_ids = set()
    with paths[0].open(encoding="utf-8", newline="") as source, paths[1].open(encoding="utf-8", newline="") as analysis:
        for a, b in itertools.zip_longest(csv.DictReader(source), csv.DictReader(analysis)):
            if a is None or b is None or any(b[k] != v for k, v in a.items()):
                raise ValueError("原始字段或行数不一致")
            template_ids.add(b["template_id"])
            if b["instance_uuid"]:
                groups[b["instance_uuid"]].append(b)
            else:
                unassigned += 1
                request_only += bool(re.search(r"\breq-[0-9a-f-]{36}\b", b["message"], re.I))
    records = []
    evidence = {}
    features_list = []
    for uuid, rows in sorted(groups.items()):
        values, phases = extract(rows, sorted(template_ids))
        features_list = list(values)
        if uuid not in prior["instances"]:
            raise ValueError("实例不在既有调查中")
        old = prior["instances"][uuid]
        if len(rows) != old["log_count"] or abs(values["duration_seconds"] - old["duration_seconds"]) > 1e-9:
            raise ValueError("日志数或时间跨度与历史调查不一致")
        sources = sorted({r["source_file"] for r in rows})
        record = {
            "instance_uuid": uuid, "instance_label_metadata": labels.get(uuid, "unknown"),
            "candidate_metadata": int(uuid in candidates),
            "exact_candidate_metadata": int(uuid in candidates and candidates[uuid]["selection_basis"] == "exact_order_components_lifecycle"),
            "batch_metadata": "|".join(sources),
            "dates_metadata": "|".join(sorted({r["event_time"][:10] for r in rows})),
            "reference_similarity_descriptive": max(
                (c["edit_similarity"] for c in old.get("comparisons_to_anomalies", {}).values()), default=1.0
            ),
            **values,
        }
        records.append(record)
        evidence[uuid] = phases
    pools = {
        "all_instances": records,
        "all_candidates": [r for r in records if r["candidate_metadata"]],
        "exact_candidates": [r for r in records if r["exact_candidate_metadata"]],
        "known_anomalous": [r for r in records if r["instance_label_metadata"] == "anomalous"],
    }
    for batch in ("openstack_normal1.log", "openstack_normal2.log"):
        pools[batch + "_candidates"] = [r for r in pools["all_candidates"] if r["batch_metadata"] == batch]
    pools["cross_file_candidates"] = [r for r in pools["all_candidates"] if "|" in r["batch_metadata"]]
    summaries = {name: {feature: distribution([r[feature] for r in rows]) for feature in TIME_FEATURES} for name, rows in pools.items()}
    baseline = []
    thresholds = {}
    for name in ("all_candidates", "exact_candidates"):
        pool = pools[name]
        threshold = quantile([r["duration_seconds"] for r in pool], QUANTILE)
        if threshold is None or threshold <= 0:
            raise ValueError("候选池缺少有效正时长")
        thresholds[name] = threshold
        # 排名统一在全部唯一实例中计算，并对同分使用竞争排名。
        scores = {r["instance_uuid"]: r["duration_seconds"] / threshold for r in records}
        for r in records:
            score = scores[r["instance_uuid"]]
            baseline.append({
                "instance_uuid": r["instance_uuid"], "baseline_pool": name,
                "duration_seconds": r["duration_seconds"], "p99_threshold_seconds": threshold,
                "duration_score": score, "exceeds_threshold": int(r["duration_seconds"] > threshold),
                "rank_all_instances": 1 + sum(v > score for v in scores.values()),
                "candidate_metadata": r["candidate_metadata"],
                "exact_candidate_metadata": r["exact_candidate_metadata"],
                "instance_label_metadata": r["instance_label_metadata"],
            })
    report = {
        "input_sha256": hashes, "sample_count": len(records),
        "pool_counts": {name: len(rows) for name, rows in pools.items()},
        "unassigned_logs": unassigned, "request_id_without_vm_logs": request_only,
        "model_feature_allowlist": features_list,
        "excluded_columns": [k for k in records[0] if k not in features_list],
        "feature_missing_counts": {k: sum(r[k] is None for r in records) for k in features_list},
        "phase_rules": PHASES, "phase_evidence_by_instance": evidence,
        "phase_policy": "起止标志各唯一且时间非负才计算。同时间戳稳定排序不代表因果。",
        "parameter_rules_and_units": PARAMETERS,
        "quantile_method": "linear interpolation at (n-1)*p",
        "std_method": "population standard deviation; one gap std=0; no gap missing",
        "descriptive_distributions": summaries, "fixed_quantile": QUANTILE,
        "thresholds_seconds": thresholds,
        "baseline_exceedances": {
            name: {group: sum(r["duration_seconds"] > threshold for r in pools[group])
                   for group in ("all_candidates", "exact_candidates", "known_anomalous")}
            for name, threshold in thresholds.items()
        },
        "known_anomaly_results": [r for r in baseline if r["instance_label_metadata"] == "anomalous"],
        "limitations": [
            "候选没有独立正常证据，超过阈值比例不能解释为可靠误报率。",
            "P99 为本次预先固定的描述基线，候选池同时用于分位数拟合与超阈值统计，未做泛化性能评估。",
            "候选由异常参考的相似性筛选，存在选择偏差。训练基线需要另行预定义不依赖异常标签的训练池。",
            "日期、UUID、文件名、标签、候选成员及参考相似度仅为审计或事后描述，不进入模型白名单。",
            "全部模板频次均保留，未依据四个异常的结果筛选模板。后续降维和预处理仅在训练数据拟合。",
            "资源量和阶段耗时的采集不完整，缺失保持空值，不能直接交给要求完整数值输入的模型。",
            "采集日期与异常标签混杂，时长差异可能来自负载和采集批次。",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        if (output / name).exists():
            raise FileExistsError(output / name)
    with TemporaryDirectory(dir=output) as directory:
        stage = Path(directory)
        for name, rows in ((OUTPUT_NAMES[0], records), (OUTPUT_NAMES[2], baseline)):
            with (stage / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (stage / OUTPUT_NAMES[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / OUTPUT_NAMES[3]).write_text(markdown(report), encoding="utf-8")
        for path in paths:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[str(path)]:
                    raise ValueError("输入数据在运行中变化")
        for name in OUTPUT_NAMES:
            os.link(stage / name, output / name)
    return report


def markdown(report):
    lines = ["# OpenStack VM 时序特征与固定分位数基线", "",
             f"有效实例 {report['sample_count']} 个。分组数量为 {report['pool_counts']}。",
             f"排除未关联日志 {report['unassigned_logs']} 条，其中仅有请求 ID 的 {report['request_id_without_vm_logs']} 条。一个 UUID 仅对应一个全局实例。",
             "阶段起止模板、原行号、识别依据及缺失原因见 JSON。参数单位写入列名，未观测参数保留空值；模板、组件等可明确计数的缺席事件记为零。",
             "", "## 时间分布", "",
             "|分组|数量|跨度中位数|跨度 P99|跨度标准差|最大间隔中位数|",
             "|---|---:|---:|---:|---:|---:|"]
    for name, data in report["descriptive_distributions"].items():
        d = data["duration_seconds"]
        lines.append(f"|{name}|{d['count']}|{d.get('median')}|{d.get('p99')}|{d.get('std_population')}|{data['gap_max_seconds'].get('median')}|")
    lines += ["", "## 固定 P99 评分", "",
              "score=实例跨度/候选池 P99，严格大于 1 为超过阈值；并列竞争排名在全部实例内计算。分别以全部候选和完全匹配子集拟合阈值，未根据异常评分调整分位数。",
              f"阈值秒数为 {report['thresholds_seconds']}，超阈值计数为 {report['baseline_exceedances']}。",
              "", "|基线池|异常 UUID|跨度秒|评分|全体排名|", "|---|---|---:|---:|---:|"]
    for r in report["known_anomaly_results"]:
        lines.append(f"|{r['baseline_pool']}|{r['instance_uuid']}|{r['duration_seconds']}|{r['duration_score']:.6f}|{r['rank_all_instances']}|")
    lines += ["", "## 研究解释与下一阶段", "",
              "总跨度、最大相邻间隔和生命周期阶段耗时值得进一步研究。分组分位数只提供当前样本中的差异证据，不能证明因果或泛化能力。normal1、normal2 的完整分布分别报告，跨文件实例单列。",
              "已经具备复现实例数值特征和探索模型的工程基础，正式引入 Isolation Forest 前仍需固定独立于四个异常的训练池，按批次或时间留出，制定仅在训练集拟合的缺失处理、尺度处理及常量列筛选方案。候选标签未经确认，不能报告可靠分类准确率或误报率。", "",
              *report["limitations"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--feasibility-dir", type=Path, default=EXPERIMENTS / "instance_feasibility")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "temporal_features")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        report = run(args.input_dir, args.feasibility_dir, args.output_dir)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        logging.error("特征分析失败：%s", error)
        return 1
    logging.info("已生成 %d 个实例特征", report["sample_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
