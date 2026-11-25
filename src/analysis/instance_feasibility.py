"""调查实例对照可行性，运行 python -m src.analysis.instance_feasibility。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from src.analysis.template_audit import edit_distance, sequence_summary
from src.data.openstack_config import OUTPUT_DIR


DESTINATION = Path(__file__).resolve().parents[3] / "experiments" / "instance_feasibility"
OUTPUTS = ("instance_distribution_report.json", "candidate_reference_instances.csv",
           "anomaly_detection_feasibility.md")
NORMAL_SCENARIO_FILES = {"openstack_normal1.log", "openstack_normal2.log"}
MARKERS = {
    "claim": "Claim successful",
    "image_creation": "Creating image",
    "started": "VM Started (Lifecycle Event)",
    "spawned": "Instance spawned successfully.",
    "build_completed": "seconds to build instance.",
    "deletion_completed": "_del complete",
    "stopped": "VM Stopped (Lifecycle Event)",
}


def summarize(rows):
    events = sequence_summary(rows)
    flags = {name: any(text in row["message"] for row in rows) for name, text in MARKERS.items()}
    return {
        "log_count": len(events),
        "template_counts": Counter(r["template_id"] for r in events),
        "template_order": [r["template_id"] for r in events],
        "component_counts": Counter(r["component"] for r in events),
        "source_counts": Counter(r["source_file"] for r in events),
        "first_event": events[0]["event_time"], "last_event": events[-1]["event_time"],
        "duration_seconds": events[-1]["elapsed_seconds"],
        "event_gaps_seconds": [r["delta_seconds"] for r in events[1:]],
        "lifecycle_markers": flags,
        "observed_lifecycle_complete": all(flags.values()),
        "trace": [{"source_file": r["source_file"], "line_number": r["line_number"],
                   "event_time": r["event_time"], "template_id": r["template_id"]} for r in events],
    }


def compare(candidate, reference):
    x, y = candidate["template_order"], reference["template_order"]
    distance = edit_distance(x, y)
    common = candidate["template_counts"].keys() & reference["template_counts"].keys()
    return {
        "same_order": x == y,
        "same_frequencies": candidate["template_counts"] == reference["template_counts"],
        "same_components": candidate["component_counts"] == reference["component_counts"],
        "edit_distance": distance,
        "edit_similarity": 1 - distance / max(len(x), len(y)),
        "reference_template_coverage": len(common) / len(set(y)),
        "duration_ratio": candidate["duration_seconds"] / reference["duration_seconds"] if reference["duration_seconds"] else None,
        "aligned_gap_mae_seconds": (
            sum(abs(a - b) for a, b in zip(candidate["event_gaps_seconds"], reference["event_gaps_seconds"]))
            / len(candidate["event_gaps_seconds"]) if x == y and len(x) > 1 else None
        ),
    }


def investigate(inputs, output):
    paths = [inputs / name for name in ("openstack_structured.csv",
             "openstack_template_analysis.csv", "openstack_instance_labels.csv")]
    hashes = {}
    for path in paths:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    with paths[2].open(encoding="utf-8", newline="") as stream:
        labels = {row["instance_uuid"]: row["anomaly_label"] for row in csv.DictReader(stream)}
    grouped = defaultdict(list)
    files = defaultdict(lambda: {
        "total_logs": 0, "linked_logs": 0, "uuids": set(),
        "request_id_without_vm": 0, "no_request_id_without_vm": 0,
        "unassigned_examples": [], "dates": Counter(),
    })
    environments = Counter()
    with paths[0].open(encoding="utf-8", newline="") as source, paths[1].open(encoding="utf-8", newline="") as analysis:
        for original, row in itertools.zip_longest(csv.DictReader(source), csv.DictReader(analysis)):
            if original is None or row is None or any(row[k] != value for k, value in original.items()):
                raise ValueError("结构化 CSV 和模板 CSV 的原始字段不一致")
            info = files[row["source_file"]]
            info["total_logs"] += 1
            info["dates"][row["event_time"][:10]] += 1
            environments.update(re.findall(r"\bcp-[\w.-]+", row["message"]))
            uuid = row["instance_uuid"]
            if uuid:
                grouped[uuid].append(row)
                info["linked_logs"] += 1
                info["uuids"].add(uuid)
            else:
                has_request = bool(re.search(r"\breq-[0-9a-f-]{36}\b", row["message"], re.I))
                info["request_id_without_vm" if has_request else "no_request_id_without_vm"] += 1
                if has_request and len(info["unassigned_examples"]) < 3:
                    info["unassigned_examples"].append({k: row[k] for k in ("line_number", "message", "event_time")})
    instances = {uuid: summarize(rows) for uuid, rows in sorted(grouped.items())}
    anomalies = {uuid: data for uuid, data in instances.items() if labels.get(uuid) == "anomalous"}
    if not anomalies:
        raise ValueError("没有可关联的已标注异常实例")
    reference_uuid = sorted(anomalies)[0]
    reference = anomalies[reference_uuid]
    candidates = []
    matches = Counter()
    for uuid, instance in instances.items():
        if uuid in anomalies:
            continue
        comparisons = {a: compare(instance, data) for a, data in anomalies.items()}
        instance["comparisons_to_anomalies"] = comparisons
        result = comparisons[reference_uuid]
        normal_scenario = set(instance["source_counts"]).issubset(NORMAL_SCENARIO_FILES)
        exact = result["same_order"] and result["same_components"] and instance["observed_lifecycle_complete"]
        similar = (instance["observed_lifecycle_complete"]
                   and result["reference_template_coverage"] == 1
                   and result["edit_similarity"] >= 0.9)
        if result["same_frequencies"]:
            matches["same_template_frequencies"] += 1
        if result["same_order"]:
            matches["same_template_order"] += 1
        if similar:
            matches["similar_lifecycle_all_sources"] += 1
        if normal_scenario and similar:
            matches["normal_scenario_candidates"] += 1
            if exact:
                matches["normal_scenario_exact_candidates"] += 1
            candidates.append({
                "instance_uuid": uuid, "existing_label": labels.get(uuid, "unknown"),
                "selection_basis": "exact_order_components_lifecycle" if exact else "coverage_1_edit_similarity_ge_0.9_lifecycle",
                "independent_normal_evidence": "unavailable",
                "log_count": instance["log_count"],
                "unique_templates": len(instance["template_counts"]),
                "source_counts": json.dumps(instance["source_counts"], sort_keys=True),
                "component_counts": json.dumps(instance["component_counts"], sort_keys=True),
                "first_event": instance["first_event"], "last_event": instance["last_event"],
                "duration_seconds": instance["duration_seconds"],
                **result,
            })
    cross = {uuid: list(data["source_counts"]) for uuid, data in instances.items() if len(data["source_counts"]) > 1}
    for info in files.values():
        info["unique_vm_count"] = len(info["uuids"])
        info["uuids"] = sorted(info["uuids"])
    reference_templates = set(reference["template_order"])
    template_presence = {
        key: {"other_instance_count": sum(key in d["template_counts"] for u, d in instances.items() if u not in anomalies),
              "other_log_count": sum(d["template_counts"][key] for u, d in instances.items() if u not in anomalies)}
        for key in sorted(reference_templates)
    }
    report = {
        "input_sha256": hashes, "files": files, "unique_instances": len(instances),
        "label_counts": Counter(labels.values()), "cross_file_instances": cross,
        "instances": instances, "reference_uuid": reference_uuid,
        "anomalies_share_order": all(d["template_order"] == reference["template_order"] for d in anomalies.values()),
        "reference_template_presence": template_presence, "candidate_counts": matches,
        "observed_hostnames": environments,
        "selection_policy": "非异常标签；仅 normal1/normal2 来源；七个生命周期标志均出现；覆盖参考26模板且编辑相似度>=0.9。exact 再要求顺序及组件计数一致。不依据耗时筛选，避免掩盖异常差异。",
        "limitations": [
            "只按既有 instance_uuid 关联，请求 ID 无法单独证明 VM 归属。",
            "生命周期完整仅指七个消息标志均被观测到，不证明操作全部成功或实例健康。",
            "日期和来源仅用于调查混杂，UUID 仅用于分组，均不应作为异常检测特征。",
            "相同时间戳按输入顺序稳定排序，跨进程因果与时钟同步未知。",
            "同一 UUID 全局合并，只统计一次，不按文件切成多个实例。",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (output / name).exists():
            raise FileExistsError(output / name)
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        (stage / OUTPUTS[0]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (stage / OUTPUTS[1]).open("w", encoding="utf-8", newline="") as stream:
            fields = list(candidates[0]) if candidates else ["instance_uuid", "existing_label", "selection_basis", "independent_normal_evidence"]
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(candidates)
        (stage / OUTPUTS[2]).write_text(make_report(report, candidates), encoding="utf-8")
        for path in paths:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[path.name]:
                    raise ValueError("分析过程中输入变化")
        for name in OUTPUTS:
            os.link(stage / name, output / name)
    return report


def make_report(report, candidates):
    lines = ["# OpenStack 实例对照可行性调查", "",
             "本报告中的候选表示事件记录可比较，正常状态仍缺少独立证据。", "",
             "|文件|唯一 VM|关联日志|仅请求 ID 无 VM|日期分布|",
             "|---|---:|---:|---:|---|"]
    for file, info in report["files"].items():
        lines.append(f"|{file}|{info['unique_vm_count']}|{info['linked_logs']}|{info['request_id_without_vm']}|{dict(info['dates'])}|")
    lines += ["", f"全局唯一实例 {report['unique_instances']} 个，跨文件实例 {len(report['cross_file_instances'])} 个。全局按 UUID 合并后统计，文件唯一数不可直接相加。",
              f"候选 {len(candidates)} 个，其中完全相同顺序、组件分布且具备生命周期标志的有 {report['candidate_counts'].get('normal_scenario_exact_candidates', 0)} 个。",
              report["selection_policy"], "",
              "## 可比较性和证据边界", "",
              "异常参考为四个实例中 UUID 排序最小者，同时为每个其他实例计算与四个异常实例各自的相似度、时长比例和可对齐间隔差。逐实例顺序、间隔、模板频次及原行号在 JSON 中。候选 CSV 给出组件计数、日期、文件来源及参考比较值。",
              "日期范围与环境字符串见 JSON。消息内观察到同名主机不能证明配置、负载或注入场景一致。不同采集日、来源和运行批次均可能形成混杂。相似生命周期和成功响应没有提供独立正常标签；应补查采集说明、故障注入记录、运维监控和实例健康验证。", ""]
    if candidates:
        durations = sorted(row["duration_seconds"] for row in candidates)
        lines.append(f"候选执行跨度最小 {durations[0]} 秒，最大 {durations[-1]} 秒。耗时未参与筛选，需按实例生命周期阶段和负载分层后比较，避免用异常参考拟合正常时长。")
        exact_durations = [row["duration_seconds"] for row in candidates if row["selection_basis"] == "exact_order_components_lifecycle"]
        anomaly_durations = [data["duration_seconds"] for data in report["instances"].values()
                             if "comparisons_to_anomalies" not in data]
        if exact_durations:
            lines.append(f"完全匹配候选跨度 {min(exact_durations)}–{max(exact_durations)} 秒；四个异常实例跨度 {min(anomaly_durations)}–{max(anomaly_durations)} 秒。异常采集为 2017-05-14，正常场景为 2017-05-16/17，时长差异无法排除批次、负载及环境变化。")
        lines.append(f"其他实例中模板频次完全相同的有 {report['candidate_counts'].get('same_template_frequencies', 0)} 个，顺序完全相同的有 {report['candidate_counts'].get('same_template_order', 0)} 个，统计包含 abnormal 文件中的 unknown 实例。26 个参考模板广泛存在于其他实例，单凭出现这些模板无法区分异常。")
    lines += ["", "## 三种研究方案", "",
              "|方案|可行性与标签风险|泄漏风险及控制|论文可支持的结论|",
              "|---|---|---|---|",
              "|A 可靠正常证据对照|现有输入缺少独立正常证据，须补证后再使用。unknown 保持未确认。|按全局 UUID、采集批次分组划分，核查跨文件、同源重复及异常注入批次。|补证前只能描述候选差异，不能报告可靠正常/异常分类性能。|",
              "|B 正常采集场景无监督基线|适合当前数据，可建立场景行为参考，但训练池可能含未标注异常。|按时间/批次留出，拟合统计量、模板选择和阈值仅用开发数据；禁止文件名、绝对日期、UUID 进入特征。候选按四个异常模式筛选会产生选择偏差，实际基线应使用预先定义的场景全体适格实例并保留不完整实例的覆盖统计。|可报告行为偏离、已知异常排名及个案分析，不能解释为确认正常实例上的分类性能。|",
              "|C 独立公开数据验证|若目标是定量验证算法，需要补充带可靠标签及足够独立样本的数据，或取得当前数据的正常证据。需核验标签粒度和授权。|按数据集规定的会话/实例分组，不把日志行随机拆分；独立数据的调参和测试分离。|能验证算法在该数据集上的性能，不能直接外推到 OpenStack。|",
              "", "## 下一阶段", "",
              "优先执行 B，先建立实例模板频次、转移、阶段顺序、相邻时间间隔及阶段耗时的描述基线。提取原文资源量与耗时参数并保留单位。四个已知异常只用于独立观察和排名，避免反复据其调阈值。建立采集来源、批次与异常注入的证据表。若论文需要可复现的二分类指标，必须补充可靠标签及独立样本，来源可以是本数据补证或其他公开数据集。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        report = investigate(args.input_dir, args.output_dir)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        logging.error("实例调查失败：%s", error)
        return 1
    logging.info("调查完成：%s", dict(report["candidate_counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
