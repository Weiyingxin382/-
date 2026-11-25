"""固定跨批次实例级基线。依赖安装须事先获得授权。"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import pickle
import sys
import time
from pathlib import Path

from src.analysis.temporal_features import TIME_FEATURES, EXPERIMENTS, quantile
from src.data.openstack_config import OUTPUT_DIR


PARAMS = dict(n_estimators=300, max_samples=256, contamination="auto",
              max_features=1.0, bootstrap=False, random_state=42, n_jobs=1)
BATCHES = ("openstack_normal1.log", "openstack_normal2.log")
DEPENDENCIES = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def quality_reasons(row, prior):
    reasons = []
    if row["instance_label_metadata"] == "anomalous":
        reasons.append("known_anomalous_excluded")
    if row["batch_metadata"] not in BATCHES:
        reasons.append("not_single_normal_collection_batch")
    if not prior["observed_lifecycle_complete"]:
        reasons.append("missing_lifecycle_markers")
    if any(row[k] == "" for k in TIME_FEATURES):
        reasons.append("missing_time_feature")
    return reasons


def rank_bounds(scores, value):
    return 1 + sum(s > value for s in scores), sum(s >= value for s in scores)


def rank1_case(output, rows, prior):
    top = max(rows, key=lambda row: float(row["duration_seconds"]))
    uuid = top["instance_uuid"]
    with (OUTPUT_DIR / "openstack_template_analysis.csv").open(encoding="utf-8", newline="") as stream:
        events = [r for r in csv.DictReader(stream) if r["instance_uuid"] == uuid]
    events.sort(key=lambda row: row["event_time"])
    from datetime import datetime
    for row in events:
        if row["event_time"] not in row["raw_text"] or uuid not in row["raw_text"]:
            raise ValueError("排名第一实例原始文本核验失败")
    gaps = [(datetime.fromisoformat(b["event_time"]) - datetime.fromisoformat(a["event_time"])).total_seconds()
            for a, b in zip(events, events[1:])]
    span = (datetime.fromisoformat(events[-1]["event_time"]) - datetime.fromisoformat(events[0]["event_time"])).total_seconds()
    if span != float(top["duration_seconds"]):
        raise ValueError("排名第一实例跨度不一致")
    text = ["# 时长排名第一实例案例", "", f"UUID {uuid}，标签 {top['instance_label_metadata']}。",
            f"关联 {len(events)} 条、{top['template_count']} 个模板。跨度 {span} 秒，最大间隔 {max(gaps)} 秒。",
            f"来源 {top['batch_metadata']}，组件计数 {prior['instances'][uuid]['component_counts']}。",
            "时间字段逐条与 raw_text 核对，UUID 在原文明确出现，跨度和间隔由原事件时间重新计算。未发现本次核查可确认的时间字段提取或 UUID 聚合错误。",
            "主要长间隔跨 normal1 尾部与 normal2 首部。无法区分实例长时间存续、采集缺口或未关联事件，缺少启动、构建完成和删除完成标志，完整周期不可比较。",
            "四个已知异常均为 28 条、26 个模板，跨度 52.909–74.561 秒。该实例仅有 12 条日志且跨文件，按事先固定完整性规则排除出训练/合格测试池，另在全部实例评分中保留。unknown 继续表示未确认。",
            "", "|来源|行号|时间|模板|正文|", "|---|---:|---|---|---|"]
    for row in events:
        text.append("|" + "|".join(row[k].replace("|", "\\|") for k in ("source_file", "line_number", "event_time", "template_id", "message")) + "|")
    (output / "rank1_instance_case.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run(output, model_dir):
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    names = ("experiment_config.json", "anomaly_scores.csv", "experiment_results.json",
             "model_comparison.md")
    model_names = [f"{batch.split('.')[0]}_{method}.pkl" for batch in BATCHES for method in ("if_templates", "if_time", "if_fused")]
    for path in [output / name for name in names] + [model_dir / name for name in model_names]:
        if path.exists():
            raise FileExistsError(path)
    input_paths = [EXPERIMENTS / "temporal_features" / name for name in ("instance_features.csv", "temporal_feature_report.json")]
    input_paths += [EXPERIMENTS / "instance_feasibility" / "instance_distribution_report.json"]
    fingerprints = {}
    for path in input_paths:
        with path.open("rb") as f:
            fingerprints[str(path)] = hashlib.file_digest(f, "sha256").hexdigest()
    rows = read_csv(input_paths[0])
    schema = json.loads(input_paths[1].read_text(encoding="utf-8"))
    prior = json.loads(input_paths[2].read_text(encoding="utf-8"))
    template_features = ["log_count", "template_count"] + sorted(k for k in rows[0] if k.startswith("count_T_"))
    sets = {"if_templates": template_features, "if_time": list(TIME_FEATURES),
            "if_fused": template_features + list(TIME_FEATURES)}
    assert all(set(fields).issubset(schema["model_feature_allowlist"]) for fields in sets.values())
    excluded = {}
    eligible = []
    for row in rows:
        reasons = quality_reasons(row, prior["instances"][row["instance_uuid"]])
        if reasons:
            excluded[row["instance_uuid"]] = reasons
        else:
            eligible.append(row)
    config = {
        "python": sys.version, "dependencies": versions, "input_sha256": fingerprints,
        "isolation_forest_params": PARAMS, "feature_sets": sets,
        "quality_policy": "single normal1/normal2 source; all seven prior lifecycle markers observed; six time features present; known anomalies excluded. No reference similarity or candidate membership filter.",
        "threshold_policy": "training score P99, strict greater-than",
        "preprocessing": "median imputation, zero-variance removal and standard scaling fit only on training; full feature pipeline saved",
        "template_baseline": "mean absolute training-standardized template feature deviation; constant columns use scale=1",
        "subset_policy": "legacy exact-candidate membership used only as evaluation subgroup, not training filter",
        "tie_policy": "rank interval [1+strictly_higher, number_greater_equal]; TopK definite if worst_rank<=K, possible if best_rank<=K",
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if not (output / "rank1_instance_case.md").exists():
        rank1_case(output, rows, prior)
    results = {"eligible_count": len(eligible), "excluded_instances": excluded, "runs": [],
               "limitations": ["Unknown labels are unconfirmed. Alert proportion is not a false positive rate.",
                               "Four known anomalies share collection date and cannot establish generalization.",
                               "Templates were extracted globally before split, a transductive vocabulary limitation.",
                               "Exact-match subgroup was historically selected against anomalies and is descriptive only.",
                               "P99 from training scores is an in-sample threshold, not calibrated held-out risk.",
                               "Follow-up ablations should drop duration, redundant gap statistics, log count and correlated template counts using development data only."]}
    score_rows = []
    for train_batch, test_batch in (BATCHES, BATCHES[::-1]):
        train = [r for r in eligible if r["batch_metadata"] == train_batch]
        test = [r for r in eligible if r["batch_metadata"] == test_batch]
        train_ids = {r["instance_uuid"] for r in train}
        test_ids = {r["instance_uuid"] for r in test}
        known_ids = {r["instance_uuid"] for r in rows if r["instance_label_metadata"] == "anomalous"}
        assert not train_ids & test_ids and not train_ids & known_ids
        def matrix(items, fields):
            return np.array([[float(r[k]) if r[k] != "" else np.nan for k in fields] for r in items])
        for method in ("duration_p99", "template_deviation", "if_templates", "if_time", "if_fused"):
            started = time.perf_counter()
            correlations = []
            preprocessing_audit = {}
            fields = template_features if method == "template_deviation" else sets.get(method, ["duration_seconds"])
            x, all_x = matrix(train, fields), matrix(rows, fields)
            if method == "duration_p99":
                training_scores, scores = x[:, 0], all_x[:, 0]
            elif method == "template_deviation":
                median = np.nanmedian(x, axis=0)
                x = np.where(np.isnan(x), median, x)
                all_x = np.where(np.isnan(all_x), median, all_x)
                mean, scale = x.mean(axis=0), x.std(axis=0)
                scale[scale == 0] = 1
                training_scores = np.abs((x - mean) / scale).mean(axis=1)
                scores = np.abs((all_x - mean) / scale).mean(axis=1)
            else:
                model = Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("variance", VarianceThreshold()),
                                  ("scale", StandardScaler()),
                                  ("forest", IsolationForest(**PARAMS))])
                model.fit(x)
                training_scores = -model.score_samples(x)
                scores = -model.score_samples(all_x)
                preprocessing_audit = {
                    "imputer_statistics": model["impute"].statistics_.tolist(),
                    "retained_features": [field for field, retained in zip(fields, model["variance"].get_support()) if retained],
                    "scaler_mean": model["scale"].mean_.tolist(),
                    "forest_training_samples": int(model["forest"].max_samples_),
                }
                active = np.nanstd(x, axis=0) > 0
                active_fields = [f for f, keep in zip(fields, active) if keep]
                if active.sum() > 1:
                    corr = np.corrcoef(x[:, active], rowvar=False)
                    correlations = [{"a": active_fields[i], "b": active_fields[j], "pearson": float(corr[i, j])}
                                    for i in range(len(active_fields)) for j in range(i + 1, len(active_fields))
                                    if abs(corr[i, j]) >= .95]
                with (model_dir / f"{train_batch.split('.')[0]}_{method}.pkl").open("xb") as stream:
                    pickle.dump({"pipeline": model, "features": fields, "config": config}, stream)
            threshold = float(quantile(list(training_scores), .99))
            runtime = time.perf_counter() - started
            score_map = {r["instance_uuid"]: float(s) for r, s in zip(rows, scores)}
            evaluation_ids = test_ids | known_ids
            eval_scores = [score_map[u] for u in evaluation_ids]
            anomaly_results = []
            for row in rows:
                uuid = row["instance_uuid"]
                value = score_map[uuid]
                low, high = rank_bounds(eval_scores, value) if uuid in evaluation_ids else (None, None)
                entry = {"train_batch": train_batch, "test_batch": test_batch, "method": method,
                         "instance_uuid": uuid, "score": value, "threshold_train_p99": threshold,
                         "alert": int(value > threshold), "test_rank_best": low, "test_rank_worst": high,
                         "sample_role": "train" if uuid in train_ids else "test" if uuid in test_ids else "known_anomalous" if uuid in known_ids else "excluded",
                         "exact_subset_metadata": row["exact_candidate_metadata"],
                         "label_metadata": row["instance_label_metadata"]}
                score_rows.append(entry)
                if uuid in known_ids:
                    anomaly_results.append(entry)
            subgroups = {}
            for name, subset in (("all_eligible_test", test), ("legacy_exact_test", [r for r in test if r["exact_candidate_metadata"] == "1"])):
                ids = {r["instance_uuid"] for r in subset}
                alerts = sum(score_map[u] > threshold for u in ids)
                sub_scores = [score_map[u] for u in ids | known_ids]
                ranks = {u: rank_bounds(sub_scores, score_map[u]) for u in sorted(known_ids)}
                subgroups[name] = {"count": len(ids), "alerts": alerts, "alert_fraction": alerts / len(ids) if ids else None,
                                  "known_anomaly_rank_intervals": ranks,
                                  "topk": {k: {"definite_hits": sum(b <= k for a, b in ranks.values()),
                                               "possible_hits": sum(a <= k for a, b in ranks.values())} for k in (5, 10, 20)}}
            results["runs"].append({"train_batch": train_batch, "test_batch": test_batch, "method": method,
                                    "train_count": len(train), "test_count": len(test), "runtime_seconds": runtime,
                                    "train_uuids": sorted(train_ids), "test_uuids": sorted(test_ids),
                                    "preprocessing_audit": preprocessing_audit,
                                    "threshold": threshold, "known_anomalies": anomaly_results,
                                    "subgroups": subgroups, "high_training_correlations": correlations})
    with (output / "anomaly_scores.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    for path, expected in fingerprints.items():
        with Path(path).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    (output / "experiment_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = ["# OpenStack 跨批次无监督排序", "", f"合格实例 {len(eligible)}，排除 {len(excluded)}。排除原因逐 UUID 记录。", "",
             "所有训练池来自正常采集场景，逐实例正常状态未经确认。模型与参数固定，四个已知异常只参加事后评分。评分越高越异常，阈值为训练评分 P99。",
             "模板基线为训练标准化绝对偏差均值，同时额外报告模板 Isolation Forest，以区分特征效应与算法效应。",
             "", "|训练批次|方法|训练/测试数|异常排名区间|测试告警数|Top5确定/可能命中|耗时秒|",
             "|---|---|---|---|---:|---|---:|"]
    for run in results["runs"]:
        subgroup = run["subgroups"]["all_eligible_test"]
        ranks = {x["instance_uuid"][:8]: [x["test_rank_best"], x["test_rank_worst"]] for x in run["known_anomalies"]}
        lines.append(f"|{run['train_batch']}|{run['method']}|{run['train_count']}/{run['test_count']}|{ranks}|{subgroup['alerts']}|{subgroup['topk'][5]}|{run['runtime_seconds']:.4f}|")
    lines += ["", "排名池为目标批次合格实例与四个已知异常，训练实例不参与排名。并列给出区间，TopK 确定/可能命中避免 UUID 排序打破并列。",
              "全部合格与历史完全匹配子集的 Top5/10/20、告警比例及每个异常的排名在 JSON 中。",
              "", "## 限制与消融计划", "", *results["limitations"],
              "相关系数绝对值>=0.95 的训练特征对在 JSON 中。后续预定义去掉总跨度、去掉均值/标准差等冗余间隔统计、去掉日志数以及只保留模板比例的消融；本次不据四个异常表现调参。", ""]
    (output / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "isolation_forest")
    parser.add_argument("--model-dir", type=Path, default=EXPERIMENTS.parent / "models" / "isolation_forest")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run(args.output_dir, args.model_dir)
    except (OSError, ValueError, ImportError) as error:
        logging.error("实验失败：%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
