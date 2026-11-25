"""模板审计及实例时序分析，运行 python -m src.analysis.template_audit。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from src.analysis.template_rules import TOKENS, FILE_PATH, normalize_message
from src.data.openstack_config import OUTPUT_DIR


OUTPUTS = ("template_quality_report.json", "abnormal_instance_sequence.csv",
           "template_comparison_report.md")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "experiments" / "template_audit"


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def sequence_summary(rows):
    # 同一毫秒内用输入顺序稳定排序，无法据此推断跨进程因果顺序。
    rows = sorted(rows, key=lambda row: row["event_time"])
    previous = None
    for index, row in enumerate(rows, 1):
        event = datetime.fromisoformat(row["event_time"])
        row["sequence_index"] = index
        row["delta_seconds"] = "" if previous is None else (event - previous).total_seconds()
        row["elapsed_seconds"] = (event - datetime.fromisoformat(rows[0]["event_time"])).total_seconds()
        previous = event
    return rows


def compare_sequences(sequences):
    comparisons = []
    for left, right in itertools.combinations(sorted(sequences), 2):
        a, b = sequences[left], sequences[right]
        x, y = [r["template_id"] for r in a], [r["template_id"] for r in b]
        ca, cb = Counter(x), Counter(y)
        union = ca.keys() | cb.keys()
        denominator = sum(max(ca[k], cb[k]) for k in union)
        distance = edit_distance(x, y)
        comparisons.append({
            "left_uuid": left, "right_uuid": right,
            "identical_order": x == y, "identical_frequencies": ca == cb,
            "edit_distance": distance,
            "normalized_edit_similarity": 1 - distance / max(len(x), len(y)) if x or y else None,
            "frequency_weighted_jaccard": sum(min(ca[k], cb[k]) for k in union) / denominator if denominator else None,
            "aligned_gap_mae_seconds": (
                sum(abs(a[i]["delta_seconds"] - b[i]["delta_seconds"]) for i in range(1, len(a)))
                / (len(a) - 1) if x == y and len(a) > 1 else None
            ),
        })
    return comparisons


def audit(inputs, destination):
    paths = {name: inputs / name for name in (
        "openstack_structured.csv", "openstack_template_analysis.csv",
        "openstack_template_statistics.csv", "openstack_instance_labels.csv")}
    hashes = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    with paths["openstack_instance_labels.csv"].open(encoding="utf-8", newline="") as stream:
        labels = {r["instance_uuid"]: r["anomaly_label"] for r in csv.DictReader(stream)}
    sequences = {uuid: [] for uuid, label in labels.items() if label == "anomalous"}
    templates = {}
    parameter_examples = defaultdict(lambda: defaultdict(dict))
    with (
        paths["openstack_structured.csv"].open(encoding="utf-8", newline="") as source,
        paths["openstack_template_analysis.csv"].open(encoding="utf-8", newline="") as analysis,
    ):
        source_reader = csv.DictReader(source)
        for original, row in itertools.zip_longest(source_reader, csv.DictReader(analysis)):
            if original is None or row is None or any(row[k] != v for k, v in original.items()):
                raise ValueError("结构化日志与模板 CSV 原始字段或行数不一致")
            key = row["template_id"]
            normalized, _ = normalize_message(row["message"])
            if normalized != row["normalized_message"]:
                raise ValueError("模板结果与现有规则不一致")
            evidence = {k: row[k] for k in ("source_file", "line_number", "event_time", "module", "severity", "message", "raw_text")}
            record = templates.setdefault(key, {
                "normalized_message": normalized, "frequency": 0, "representatives": [],
                "module_counts": Counter(), "severity_counts": Counter(), "component_counts": Counter(),
                "http_status_counts": Counter(), "file_path_examples": [],
                "semantic_numeric_evidence": {},
            })
            if record["normalized_message"] != normalized:
                raise ValueError("同一 ID 对应不同模板")
            record["frequency"] += 1
            for field in ("module", "severity", "component"):
                record[field + "_counts"][row[field]] += 1
            if len(record["representatives"]) < 3 and all(e["message"] != row["message"] for e in record["representatives"]):
                record["representatives"].append(evidence)
            record["http_status_counts"].update(re.findall(r"\bstatus:\s*([1-5]\d{2})\b", row["message"]))
            for category, pattern in {
                "http_or_exit_code": r"status:|response code|Exit code:",
                "resource": r"memory|disk|vcpu|VCPU|MB|GB",
                "time": r"seconds|time:",
                "retry": r"retr(?:y|ies|ied)|attempt",
            }.items():
                if re.search(pattern, row["message"], re.IGNORECASE):
                    examples = record["semantic_numeric_evidence"].setdefault(category, [])
                    if len(examples) < 2 and all(e["message"] != row["message"] for e in examples):
                        examples.append(evidence)
            for path in FILE_PATH.findall(row["message"]):
                if path not in record["file_path_examples"] and len(record["file_path_examples"]) < 3:
                    record["file_path_examples"].append(path)
            # 每种参数保存最多三个不同值的原始行证据，避免报告随数据量无限增长。
            for match in TOKENS.finditer(row["message"]):
                kind = match.lastgroup
                examples = parameter_examples[key][kind]
                if len(examples) < 3:
                    examples.setdefault(match.group(), evidence)
            if row["instance_uuid"] in sequences:
                sequences[row["instance_uuid"]].append(dict(row))
    with paths["openstack_template_statistics.csv"].open(encoding="utf-8", newline="") as stream:
        statistics = list(csv.DictReader(stream))
    if len(statistics) != len(templates) or {r["template_id"]: int(r["frequency"]) for r in statistics} != {k: v["frequency"] for k, v in templates.items()}:
        raise ValueError("模板统计文件与逐行结果不一致")

    split_groups = defaultdict(list)
    merge_candidates = []
    numeric = []
    for key, record in templates.items():
        record["parameter_examples"] = parameter_examples[key]
        text = record["normalized_message"]
        body = re.sub(r"^\[[^\]]*\]\s*", "", text)
        split_groups[body].append(key)
        if len(record["module_counts"]) > 1 or len(record["severity_counts"]) > 1:
            merge_candidates.append(key)
        if "<NUM>" in text:
            numeric.append(key)
    splits = [{"body": body, "template_ids": ids} for body, ids in split_groups.items() if len(ids) > 1]
    sequences = {uuid: sequence_summary(rows) for uuid, rows in sequences.items()}
    summaries = {}
    for uuid, rows in sequences.items():
        gaps = [r["delta_seconds"] for r in rows[1:]]
        summaries[uuid] = {
            "event_count": len(rows), "template_order": [r["template_id"] for r in rows],
            "template_counts": Counter(r["template_id"] for r in rows),
            "first_event": rows[0]["event_time"] if rows else None,
            "last_event": rows[-1]["event_time"] if rows else None,
            "duration_seconds": rows[-1]["elapsed_seconds"] if rows else None,
            "max_gap_seconds": max(gaps) if gaps else None,
            "zero_gap_count": gaps.count(0), "severity_counts": Counter(r["severity"] for r in rows),
        }
    comparisons = compare_sequences(sequences)
    report = {
        "input_sha256": hashes, "total_logs": sum(r["frequency"] for r in templates.values()),
        "template_count": len(templates),
        "frequency_histogram": Counter(r["frequency"] for r in templates.values()),
        "templates": templates,
        "audit_scope": "全部模板统计及每个模板最多三条不同正文样本，候选检测不能证明语义等价或差异",
        "context_only_split_candidates": splits,
        "cross_module_or_severity_merge_candidates": merge_candidates,
        "numeric_information_loss_templates": numeric,
        "components": {
            c: {"template_count": sum(c in r["component_counts"] for r in templates.values()),
                "log_count": sum(r["component_counts"][c] for r in templates.values())}
            for c in ("nova-api", "nova-compute", "nova-scheduler")
        },
        "instance_sequences": summaries, "pairwise_comparisons": comparisons,
        "normal_control": {"label_counts": Counter(labels.values()),
                           "explicit_normal_uuids": [u for u, label in labels.items() if label == "normal"],
                           "conclusion": "unknown 缺乏正常证据；文件名和成功响应不足以证明整个实例正常"},
        "confirmed_findings": [
            {"issue": "非 status 写法的响应码及进程退出码被数值掩码处理",
             "template_ids": [k for k, r in templates.items() if "Bad response code" in r["normalized_message"] or "Exit code:" in r["normalized_message"]],
             "effect": "丢失错误码值；当前样本无法证明模板内存在多个不同错误码，属于参数信息丢失风险"},
            {"issue": "空正文成为独立模板",
             "template_ids": [k for k, r in templates.items() if not r["normalized_message"]],
             "effect": "不宜作为独立业务动作；堆栈续行也需要结合上下文解释"},
        ],
        "limitations": [
            "资源值、耗时、重试次数被 <NUM> 合并，模板不能保留大小和阈值差异，须从原文独立提取。",
            "HTTP 状态码按显式 status 字段检查；错误码和其他数值的业务意义仍需人工审查。",
            "文件路径抽样检查静态目录及动态值，无法证明任意路径的等价性。",
            "同毫秒日志按输入稳定排序，跨进程时钟同步情况未知，顺序相同不证明因果相同。",
            "四个已标注实例不能构成可靠二分类评估，需额外正常证据和独立异常样本。",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (destination / name).exists():
            raise FileExistsError(destination / name)
    with TemporaryDirectory(dir=destination) as temporary:
        stage = Path(temporary)
        (stage / OUTPUTS[0]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fields = ["instance_uuid", "sequence_index", "delta_seconds", "elapsed_seconds",
                  "template_id", "normalized_message", *source_reader.fieldnames]
        fields = list(dict.fromkeys(fields))
        with (stage / OUTPUTS[1]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for rows in sequences.values():
                writer.writerows(rows)
        (stage / OUTPUTS[2]).write_text(comparison_markdown(report), encoding="utf-8")
        for name, path in paths.items():
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[name]:
                    raise ValueError("输入在审计过程中变化")
        for name in OUTPUTS:
            os.link(stage / name, destination / name)
    return report


def comparison_markdown(report):
    comparisons = report["pairwise_comparisons"]
    lines = [
        "# OpenStack 模板质量及实例时序分析", "",
        f"共 {report['total_logs']} 条日志、{report['template_count']} 个模板。全部逐行核对原始字段和规则输出，并核对模板频次。每个模板最多三个不同正文样本及参数证据见 JSON。", "",
        "## 模板质量判断", "",
        "当前基线适用于事件计数研究。缺少人工事件真值，无法给出解析准确率或认定全部模板语义合格。",
        f"发现 {len(report['context_only_split_candidates'])} 组去掉首个请求上下文后正文完全相同的拆分候选；{len(report['cross_module_or_severity_merge_candidates'])} 个跨模块或级别模板需核查；{len(report['numeric_information_loss_templates'])} 个模板含数值占位符。",
        "上下文拆分可作为过度拆分证据；跨模块或级别只是过度合并线索。相同源码消息中的不同资源量与耗时可共享模板，但后续必须保留数值参数。",
        "已确认 Bad response code while validating token 和 Exit code 的数值被替换，显式 status 字段保留无法覆盖所有错误码写法；JSON confirmed_findings 列出模板 ID 及原文证据。空正文和堆栈续行被计为事件，业务序列应结合原始上下文解释。",
        "HTTP 状态码分布、路径样本、资源/时间/重试等匹配参数的原文证据按模板保存在 JSON。未出现重试日志时不能据此评价重试规则质量。", "",
        "## 异常实例序列", "",
        "|实例|事件数|模板数|跨度秒|最大间隔秒|",
        "|---|---:|---:|---:|---:|",
    ]
    for uuid, data in report["instance_sequences"].items():
        lines.append(f"|{uuid}|{data['event_count']}|{len(data['template_counts'])}|{data['duration_seconds']}|{data['max_gap_seconds']}|")
    lines += ["", f"六对比较中，顺序完全相同的有 {sum(r['identical_order'] for r in comparisons)} 对，频次完全相同的有 {sum(r['identical_frequencies'] for r in comparisons)} 对。逐对编辑距离、频次加权 Jaccard 及可对齐的间隔差见 JSON。",
              "序列 CSV 保留全部原始字段，首事件间隔为空，时间间隔使用原始时间。实例标签没有扩展至事件。",
              "", "## 正常对照与数据充分性", "",
              "现有标签仅为 anomalous 或 unknown，无法建立有可靠证据的正常实例对照。正常命名文件和 HTTP 200 无法证明实例完整生命周期正常。数据可用于描述事件模式与提出特征假设，无法支持可靠的正常/异常二分类性能结论。", "",
              "## Drain 可行性与公平实验", "",
              "建议在下一阶段把 Drain 作为对比基线。当前规则按参数类型替换并对正文精确分组；Drain 使用固定深度树搜索候选，再按 token 相似度更新模板。它仍可使用领域正则预处理，可能减轻未知动态标识符造成的拆分，同时可能合并生命周期动作、资源指标或状态码；token 长度、阈值和输入顺序也可能影响结果。",
              "公平实验应使用相同 message、相同样本、固定输入顺序及共同参数掩码，并另设无掩码消融。按组件、频次和候选问题分层采样，人工标注事件真值，保留独立测试集；只在开发集选择深度和相似度阈值。比较分组准确率、成对 precision/recall/F1、过度合并/拆分证据、状态码及参数保真、运行时间和峰值内存。原文数值另存参数，避免用模板数量替代质量。分别报告联合解析与按组件解析，记录版本、硬件、配置及随机种子。当前未运行 Drain，尚无效果或速度结论。",
              "Drain 的机制见[原始论文](https://pinjiahe.github.io/files/pdf/research/ICWS17.pdf)及[官方实现说明](https://github.com/logpai/logparser/blob/main/docs/tools/Drain.md)。", "",
              "## 下一阶段特征", "",
              "构建实例级模板频次、模板转移和短 n-gram、生命周期阶段顺序、相邻间隔及总时长、各阶段等待时间、severity 计数与组件/module 组合。另提取 CPU/内存/磁盘请求量及已用量、资源比率、HTTP 状态码、请求耗时、重试计数和路径类型。UUID 仅用于分组，source_file 和绝对日期仅用于追溯。时间特征使用相对时间，数值单位统一后保留。证实正常标签前不把 unknown 当正常训练。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="OpenStack 模板质量与实例时序审计")
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        report = audit(args.input_dir, args.output_dir)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        logging.error("审计失败：%s", error)
        return 1
    logging.info("完成 %d 个模板审计，结果保存至 %s", report["template_count"], args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
