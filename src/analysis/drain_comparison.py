"""固定 Drain 参数、相同输入和分层待人工审核样本的模板对比。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
import platform
import random
import re
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from src.analysis.drain_core import Drain
from src.analysis.template_rules import TOKENS, FILE_PATH, normalize_message
from src.analysis.temporal_features import EXPERIMENTS
from src.data.openstack_config import OUTPUT_DIR


CONFIG = {"depth": 4, "similarity": .4, "max_children": 100}
SEED = 42
OUTPUTS = ("drain_templates.csv", "drain_log_assignments.csv",
           "template_comparison.json", "template_comparison.md", "manual_review_samples.csv")
SEMANTICS = {
    "http_status": r"\bstatus:\s*([1-5]\d{2})\b",
    "error_code": r"(?:Bad response code while validating token:|Exit code:)\s*(\d+)",
    "resource_value": r"\b([0-9]+(?:\.[0-9]+)?)\s+(?:MB|GB|VCPU|CPU)\b",
    "time_value": r"\b([0-9]+(?:\.[0-9]+)?)\s+seconds\b",
}


def measure(messages, method):
    """内存为输入已加载后的 Python 新分配峰值，不是进程总 RSS。"""
    tracemalloc.start()
    start = time.perf_counter()
    parser = Drain(**CONFIG) if method != "rule" else None
    ids = []
    exact = {}
    for message in messages:
        text = message if method == "drain_raw" else normalize_message(message)[0]
        if parser is None:
            ids.append(exact.setdefault(text, len(exact) + 1))
        else:
            ids.append(parser.add(text))
    templates = parser.templates() if parser else {value: key for key, value in exact.items()}
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ids, templates, {
        "instrumented_seconds": seconds, "logs_per_second": len(messages) / seconds,
        "python_allocations_peak_bytes_excluding_preloaded_input": peak,
        "template_count": len(templates),
    }


def sample_indices(rows, rule_ids):
    frequencies = Counter(rule_ids)
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        message = row["message"]
        frequency = frequencies[rule_ids[index]]
        strata["frequency:" + ("high_ge_1000" if frequency >= 1000 else "low_le_10" if frequency <= 10 else "middle")].append(index)
        strata["component:" + row["component"]].append(index)
        strata["severity:" + row["severity"]].append(index)
        for name, pattern in SEMANTICS.items():
            if re.search(pattern, message):
                strata["case:" + name].append(index)
        if FILE_PATH.search(message):
            strata["case:file_path"].append(index)
        if "_del" in message:
            strata["case:deletion_path"].append(index)
    generator = random.Random(SEED)
    selected = defaultdict(list)
    for name, indices in sorted(strata.items()):
        for index in generator.sample(indices, min(12, len(indices))):
            selected[index].append(name)
    # 每个已有模板抽两条，便于核查稀有事件及同模板参数变化。
    by_template = defaultdict(list)
    for index, key in enumerate(rule_ids):
        by_template[key].append(index)
    for key, indices in sorted(by_template.items()):
        for index in generator.sample(indices, min(2, len(indices))):
            selected[index].append("rule_template_coverage")
    return selected, {key: len(value) for key, value in strata.items()}


def parameter_values(message):
    values = defaultdict(list)
    for match in TOKENS.finditer(message):
        if match.lastgroup != "static_number":
            values[match.lastgroup].append(match.group())
    values["file_path"] = FILE_PATH.findall(message)
    for name, pattern in SEMANTICS.items():
        values[name] = re.findall(pattern, message)
    return dict(values)


def compare(rows, output):
    messages = [r["message"] for r in rows]
    measurements = {}
    assignments = {}
    templates = {}
    for method in ("rule", "drain_shared", "drain_raw"):
        ids, mined, metrics = measure(messages, method)
        assignments[method], templates[method], measurements[method] = ids, mined, metrics
        logging.info("%s: %d templates, %.3f seconds", method, len(mined), metrics["instrumented_seconds"])
    legacy_map = {}
    for row, key in zip(rows, assignments["rule"]):
        if key in legacy_map and legacy_map[key] != row["template_id"]:
            raise ValueError("历史规则模板映射不一致")
        legacy_map[key] = row["template_id"]
        if templates["rule"][key] != row["normalized_message"]:
            raise ValueError("当前规则输出与历史模板不一致")
    frequencies = {method: Counter(ids) for method, ids in assignments.items()}
    template_details = {}
    diagnostics = {}
    for method, ids in assignments.items():
        cross_rule = defaultdict(Counter)
        cross_method = defaultdict(Counter)
        component = defaultdict(Counter)
        severity = defaultdict(Counter)
        semantic_values = defaultdict(lambda: defaultdict(Counter))
        semantic_retention = defaultdict(Counter)
        examples = defaultdict(list)
        for index, (row, key) in enumerate(zip(rows, ids)):
            cross_rule[key][row["template_id"]] += 1
            cross_method[row["template_id"]][key] += 1
            component[key][row["component"]] += 1
            severity[key][row["severity"]] += 1
            text = messages[index] if method == "drain_raw" else normalize_message(messages[index])[0]
            tokens = text.split()
            mined_tokens = templates[method][key].split()
            if len(examples[key]) < 3 and all(e["message"] != row["message"] for e in examples[key]):
                examples[key].append({k: row[k] for k in ("source_file", "line_number", "message", "severity", "component")})
            for name, pattern in SEMANTICS.items():
                matches = list(re.finditer(pattern, messages[index]))
                semantic_values[key][name].update(match.group(1) for match in matches)
                # 同一正则在算法输入上仍能识别且最终模板同位置保留原值才计保留。
                processed = list(re.finditer(pattern, text))
                for match in matches:
                    semantic_retention[name]["raw_occurrences"] += 1
                for match in processed:
                    position = len(text[:match.start(1)].split())
                    # 若值紧接冒号，同 token 内也可保留，使用完整 token 相等检查。
                    if position < len(tokens) and len(tokens) == len(mined_tokens):
                        if tokens[position] == mined_tokens[position]:
                            semantic_retention[name]["literal_retained_in_final_template"] += 1
        details = []
        for key in sorted(templates[method]):
            details.append({
                "template_id": f"{method}_{key:05d}", "template": templates[method][key],
                "frequency": frequencies[method][key],
                "legacy_rule_ids": dict(cross_rule[key]),
                "component_counts": dict(component[key]), "severity_counts": dict(severity[key]),
                "semantic_value_distributions": {k: dict(v) for k, v in semantic_values[key].items()},
                "examples": examples[key],
            })
        template_details[method] = details
        diagnostics[method] = {
            "frequency_histogram": dict(Counter(frequencies[method].values())),
            "templates_by_component": {c: sum(c in count for count in component.values())
                                       for c in sorted({r["component"] for r in rows})},
            "templates_by_severity": {s: sum(s in count for count in severity.values())
                                      for s in sorted({r["severity"] for r in rows})},
            "final_template_semantic_literal_retention": {name: dict(count) for name, count in semantic_retention.items()},
            "multiple_rule_groups_in_one_cluster_candidates": sum(len(v) > 1 for v in cross_rule.values()),
            "one_rule_group_in_multiple_clusters_candidates": sum(len(v) > 1 for v in cross_method.values()),
            "mixed_http_status_clusters": [
                f"{method}_{key:05d}" for key, values in semantic_values.items()
                if len(values["http_status"]) > 1
            ],
        }
    selected, strata = sample_indices(rows, assignments["rule"])
    report = {
        "implementation": "project standard-library Drain reproduction v1, not official package",
        "sources": ["https://pinjiahe.github.io/files/pdf/research/ICWS17.pdf",
                    "https://github.com/logpai/logparser/blob/main/logparser/Drain/Drain.py"],
        "drain_parameters": CONFIG, "sample_seed": SEED,
        "python": sys.version, "platform": platform.platform(), "input_count": len(rows),
        "input_order": "existing structured CSV row order, no shuffle; global component mixing",
        "preprocessing": "rule and drain_shared use unchanged normalize_message; drain_raw is auxiliary unmasked ablation",
        "id_policy": "cluster IDs assigned on first occurrence; assignments retain online cluster ID and final generalized template",
        "performance_measurement": "one run per method, same preloaded input, tracemalloc active; includes preprocessing, clustering and assignment storage; excludes CSV IO, diagnostics and manual sampling. Python allocation peak is not RSS. Timing is instrumented and order fixed.",
        "measurements": measurements, "diagnostics": diagnostics,
        "templates": template_details, "manual_sample_count": len(selected),
        "manual_strata_population": strata, "manual_verified_count": 0,
        "manual_grouping_metrics": None,
        "manual_policy": "所有人工标签留空；规则ID仅用于抽样覆盖及算法交叉表，不是真值。抽样不具有总体无偏估计性质。",
        "limitations": [
            "算法分组交叉关系只代表差异，不能用来判定谁过度合并/拆分。",
            "HTTP 状态码的多值混合是语义保真警示，是否按不同事件分组需人工定义。",
            "资源量、耗时作为动态参数被掩码属于预期泛化；独立保留原文与参数字段才能恢复数值。",
            "参数字段由统一扫描器从原文提取，不是 Drain 自动参数恢复能力的验证。",
            "保真统计仅覆盖四种显式模式，不代表所有业务语义和上下文被保留。",
            "模板变化会影响历史行的最终模板，使用最终模板而非到达时模板评估。",
            "空白按 split() 处理，空正文作为零 token 单独聚类；深度按 LogPAI 定义。",
            "采用项目复现版本，尚未与官方实现逐条做一致性验证。",
        ],
    }
    with (output / "drain_templates.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("method", "template_id", "template", "frequency", "component_counts", "severity_counts", "legacy_rule_ids")
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for method in ("drain_shared", "drain_raw"):
            for detail in template_details[method]:
                writer.writerow({"method": method, **{
                    key: json.dumps(detail[key], ensure_ascii=False, sort_keys=True) if isinstance(detail[key], dict) else detail[key]
                    for key in fields if key != "method"}})
    with (output / "drain_log_assignments.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(rows[0]) + ["drain_shared_id", "drain_shared_final_template", "drain_raw_id",
                                 "drain_raw_final_template", "raw_parameter_values_json"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            values = dict(row, raw_parameter_values_json=json.dumps(parameter_values(row["message"]), ensure_ascii=False))
            for method in ("drain_shared", "drain_raw"):
                key = assignments[method][i]
                values[method + "_id"] = f"{method}_{key:05d}"
                values[method + "_final_template"] = templates[method][key]
            writer.writerow(values)
    with (output / "manual_review_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["sample_id", "strata", *rows[0], "drain_shared_id", "drain_shared_final_template",
                  "drain_raw_id", "drain_raw_final_template", "human_event_group", "reviewer",
                  "review_status", "evidence_notes", "rule_issue", "drain_issue"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for number, (index, tags) in enumerate(sorted(selected.items()), 1):
            entry = dict(rows[index], sample_id=f"S{number:04d}", strata="|".join(tags),
                         review_status="pending", human_event_group="", reviewer="", evidence_notes="",
                         rule_issue="", drain_issue="")
            for method in ("drain_shared", "drain_raw"):
                key = assignments[method][index]
                entry[method + "_id"] = f"{method}_{key:05d}"
                entry[method + "_final_template"] = templates[method][key]
            writer.writerow(entry)
    return report


def markdown(report):
    lines = ["# OpenStack Drain 与规则归一化对比", "",
             "采用标准库复现 Drain v1，按固定参数处理相同 message。主要比较共享预处理版本，原始正文 Drain 为预处理消融。已有规则模板和 ID 保持原样。",
             "", "|方法|模板数|仪器化耗时秒|日志/秒|Python新增分配峰值MiB|",
             "|---|---:|---:|---:|---:|"]
    for method, metrics in report["measurements"].items():
        lines.append(f"|{method}|{metrics['template_count']}|{metrics['instrumented_seconds']:.4f}|{metrics['logs_per_second']:.1f}|{metrics['python_allocations_peak_bytes_excluding_preloaded_input']/1024**2:.3f}|")
    for method, diagnostics in report["diagnostics"].items():
        lines += ["", f"{method} 按组件模板数 {diagnostics['templates_by_component']}。",
                  f"混合 HTTP 状态码模板 {diagnostics['mixed_http_status_clusters']}。",
                  f"相对规则的一簇多组候选 {diagnostics['multiple_rule_groups_in_one_cluster_candidates']}，一组多簇候选 {diagnostics['one_rule_group_in_multiple_clusters_candidates']}。"]
        lines.append(f"最终模板语义数值保留统计 {diagnostics['final_template_semantic_literal_retention']}。")
    lines += ["", "## 高频模板", ""]
    for method, details in report["templates"].items():
        for detail in sorted(details, key=lambda d: (-d["frequency"], d["template_id"]))[:5]:
            lines.append(f"- {method} {detail['frequency']} 条，{detail['template_id']}，{detail['template']}")
    lines += ["", "## 人工评价状态", "",
              f"固定种子 {SEED} 抽取 {report['manual_sample_count']} 条待审核样本。覆盖频次、组件、级别、错误码、资源量、路径和每个规则模板。人工已核验数量为 0，没有人工分组准确率。",
              "审核者应以原文、源码语义和业务上下文定义事件组，填 human_event_group、reviewer、review_status 和证据，不以规则 ID 或 Drain ID 作为真值。建议遮蔽算法列进行首轮标注，再检查同组与异组的两两关系。当前无法判断哪种方法总体过度合并或拆分更严重。",
              "", "## 结论边界", "",
              "可以保留 Drain 作为系统可切换的研究候选。正式采用前，需人工核验混合状态码及生命周期动作、路径泛化案例，并与官方实现交叉验证。共享预处理可控制正则差异，但也继承现有规则丢失错误码的局限。",
              "最终模板中的动态值掩码与参数不可恢复是不同问题。两种方法均保留 message 和原文扫描参数，能够追溯，不将其宣称为解析器的参数自动恢复性能。",
              "耗时在 tracemalloc 开启时测量，仅一次固定顺序运行，不能外推生产吞吐或进程 RSS。", "",
              *report["limitations"], "",
              "[Drain 原始论文](https://pinjiahe.github.io/files/pdf/research/ICWS17.pdf)",
              "[LogPAI 实现](https://github.com/logpai/logparser/blob/main/logparser/Drain/Drain.py)", ""]
    return "\n".join(lines)


def run(output):
    if any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("输出已存在，请指定新目录")
    inputs = [OUTPUT_DIR / name for name in ("openstack_structured.csv", "openstack_template_analysis.csv",
                                             "openstack_template_statistics.csv")]
    fingerprints = {}
    for path in inputs:
        with path.open("rb") as stream:
            fingerprints[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    rows = []
    with inputs[0].open(encoding="utf-8", newline="") as original, inputs[1].open(encoding="utf-8", newline="") as analysis:
        for source, row in itertools.zip_longest(csv.DictReader(original), csv.DictReader(analysis)):
            if source is None or row is None or any(row[key] != value for key, value in source.items()):
                raise ValueError("输入字段或行数不一致")
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        report = compare(rows, stage)
        report["input_sha256"] = fingerprints
        report["implementation_sha256"] = hashlib.sha256(Path(__file__).with_name("drain_core.py").read_bytes()).hexdigest()
        (stage / OUTPUTS[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (stage / OUTPUTS[3]).write_text(markdown(report), encoding="utf-8")
        for path, expected in fingerprints.items():
            with Path(path).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("输入在实验中变化")
        for name in OUTPUTS:
            os.link(stage / name, output / name)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "drain_comparison")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.output_dir)


if __name__ == "__main__":
    main()
