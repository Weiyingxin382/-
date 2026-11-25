"""复用 run_02 的固定划分、阈值政策与 D/E 模型，研究特征消融。"""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import logging
import pickle
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.temporal_features import EXPERIMENTS, TIME_FEATURES, distribution, quantile
from src.models.isolation_baseline import read_csv, rank_bounds, quality_reasons


OUTPUTS = ("ablation_scores.csv", "feature_correlation.csv",
           "ablation_results.json", "ablation_report.md")


def score_spearman(left, right):
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return float(spearmanr(left, right).statistic)


def feature_sets(config):
    return {
        "A_duration": ["duration_seconds"],
        "B_max_gap": ["gap_max_seconds"],
        "C_duration_max_gap": ["duration_seconds", "gap_max_seconds"],
        "D_all_time": config["feature_sets"]["if_time"],
        "E_time_templates": config["feature_sets"]["if_fused"],
    }


def column_profile(matrix, fields):
    profiles = {}
    for index, field in enumerate(fields):
        values = matrix[:, index]
        observed = values[~np.isnan(values)]
        counts = Counter(observed.tolist())
        mode_fraction = max(counts.values()) / len(observed) if len(observed) else None
        profiles[field] = {
            "missing_count": int(np.isnan(values).sum()),
            "distinct_count": len(counts),
            "constant": len(counts) <= 1,
            "mode_fraction": mode_fraction,
            "near_constant_nonconstant": len(counts) > 1 and mode_fraction >= .99,
        }
    return profiles


def split_usage(model, fields):
    retained = [field for field, keep in zip(fields, model["variance"].get_support()) if keep]
    counts = Counter()
    for tree, mapping in zip(model["forest"].estimators_, model["forest"].estimators_features_):
        for feature in tree.tree_.feature:
            if feature >= 0:
                counts[retained[int(mapping[feature])]] += 1
    total = sum(counts.values())
    temporal = sum(value for field, value in counts.items() if field in TIME_FEATURES)
    return {
        "retained_features": retained, "split_node_counts": dict(counts),
        "total_split_nodes": total, "temporal_split_fraction": temporal / total if total else None,
        "interpretation": "分裂节点频次描述树结构，不是预测特征重要性或因果归因。",
    }


def evaluate(score_map, threshold, test_ids, known_ids):
    ranks = {u: rank_bounds([score_map[v] for v in test_ids | known_ids], score_map[u])
             for u in sorted(known_ids)}
    count = sum(score_map[u] > threshold for u in test_ids)
    return {
        "candidate_count": len(test_ids), "candidate_alerts": count,
        "candidate_alert_fraction": count / len(test_ids) if test_ids else None,
        "known_anomaly_rank_intervals": ranks,
        "known_anomaly_scores": {u: score_map[u] for u in sorted(known_ids)},
        "known_anomaly_alerts": sum(score_map[u] > threshold for u in known_ids),
        "topk": {str(k): {"definite_hits": sum(high <= k for low, high in ranks.values()),
                         "possible_hits": sum(low <= k for low, high in ranks.values())}
                 for k in (5, 10, 20)},
    }


def run(output, baseline, models):
    if any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("消融输出已存在，请指定新的输出目录")
    config = json.loads((baseline / "experiment_config.json").read_text(encoding="utf-8"))
    previous = json.loads((baseline / "experiment_results.json").read_text(encoding="utf-8"))
    for name, version in config["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"依赖版本与 run_02 不一致：{name}")
    for path, expected in config["input_sha256"].items():
        with Path(path).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"输入与 run_02 不一致：{path}")
    paths = [EXPERIMENTS / "temporal_features" / "instance_features.csv",
             EXPERIMENTS / "instance_feasibility" / "instance_distribution_report.json"]
    rows = read_csv(paths[0])
    rows_by_uuid = {r["instance_uuid"]: r for r in rows}
    if len(rows_by_uuid) != len(rows):
        raise ValueError("实例 UUID 重复")
    prior = json.loads(paths[1].read_text(encoding="utf-8"))
    known_ids = {u for u, r in rows_by_uuid.items() if r["instance_label_metadata"] == "anomalous"}
    eligible = {u for u, row in rows_by_uuid.items() if not quality_reasons(row, prior["instances"][u])}
    old_scores = {(r["train_batch"], r["method"], r["instance_uuid"]): float(r["score"])
                  for r in read_csv(baseline / "anomaly_scores.csv")}
    sets = feature_sets(config)
    tracked = [baseline / "experiment_config.json", baseline / "experiment_results.json",
               baseline / "anomaly_scores.csv", baseline / "rank1_instance_case.md",
               baseline / "model_comparison.md", *models.glob("*.pkl"),
               *[Path(p) for p in config["input_sha256"]]]
    hashes = {}
    for path in tracked:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "fixed_config": config, "feature_sets": sets, "input_sha256": hashes,
        "near_constant_policy": "nonconstant column with modal fraction >= 0.99 among observed training values",
        "evaluation_policy": "run_02 test UUIDs plus four known anomalies; strict train-score P99; tied rank intervals",
        "runs": [], "batch_analysis": {}, "within_direction_score_stability": [],
        "cross_direction_known_score_stability": [],
        "limitations": [
            "分裂次数不是特征重要性，不提供具体预测的可靠因果归因。",
            "候选实例未逐实例确认正常，告警比例无法解释为误报率。",
            "异常样本只有四个且采集日期混杂，探索性排名不代表泛化性能。",
            "全量模板词表在划分前建立，历史完全匹配子集依赖异常参考，只作事后描述。",
            "常量列只在训练集筛除，近似常量仅审计，不据异常结果额外筛除。",
            "D/E 复用 run_02，原记录为合并训练/评分/存储耗时；本次另计推理耗时，不伪造独立训练耗时。",
        ],
    }
    correlations = []
    outputs = []
    scores_by_run = {}

    def matrix(items, fields):
        return np.array([[float(row[field]) if row[field] != "" else np.nan for field in fields] for row in items])

    directions = [r for r in previous["runs"] if r["method"] == "if_time"]
    for direction in directions:
        batch = direction["train_batch"]
        train_ids, test_ids = set(direction["train_uuids"]), set(direction["test_uuids"])
        assert not train_ids & test_ids and not train_ids & known_ids
        assert train_ids | test_ids == eligible
        train = [row for row in rows if row["instance_uuid"] in train_ids]
        assert all(row["batch_metadata"] == batch for row in train)
        temporal = matrix(train, list(TIME_FEATURES))
        corr = np.corrcoef(temporal, rowvar=False)
        for i, a in enumerate(TIME_FEATURES):
            for j, b in enumerate(TIME_FEATURES):
                correlations.append({"training_batch": batch, "feature_a": a, "feature_b": b,
                                     "pearson": float(corr[i, j]),
                                     "spearman": float(spearmanr(temporal[:, i], temporal[:, j]).statistic)})
        template_fields = config["feature_sets"]["if_templates"]
        profiles = column_profile(matrix(train, template_fields), template_fields)
        report["batch_analysis"][batch] = {
            "train_count": len(train_ids), "test_count": len(test_ids),
            "train_uuids": sorted(train_ids), "test_uuids": sorted(test_ids),
            "time_distributions": {field: distribution(temporal[:, i].tolist()) for i, field in enumerate(TIME_FEATURES)},
            "template_column_profiles": profiles,
            "template_constant_count": sum(p["constant"] for p in profiles.values()),
            "template_near_constant_nonconstant_count": sum(p["near_constant_nonconstant"] for p in profiles.values()),
        }
        for name, fields in sets.items():
            x, all_x = matrix(train, fields), matrix(rows, fields)
            reuse_method = {"D_all_time": "if_time", "E_time_templates": "if_fused"}.get(name)
            load_seconds = 0
            train_seconds = None
            if reuse_method:
                start = time.perf_counter()
                with (models / f"{batch.split('.')[0]}_{reuse_method}.pkl").open("rb") as stream:
                    artifact = pickle.load(stream)
                model = artifact["pipeline"]
                load_seconds = time.perf_counter() - start
                if artifact["features"] != fields:
                    raise ValueError("复用模型特征顺序不一致")
            else:
                model = Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("variance", VarianceThreshold()), ("scale", StandardScaler()),
                                  ("forest", IsolationForest(**config["isolation_forest_params"]))])
                start = time.perf_counter()
                model.fit(x)
                train_seconds = time.perf_counter() - start
            # 核验训练集填补统计，验证加载模型与固定划分相符。
            np.testing.assert_allclose(model["impute"].statistics_, np.nanmedian(x, axis=0))
            start = time.perf_counter()
            train_scores = -model.score_samples(x)
            train_inference_seconds = time.perf_counter() - start
            start = time.perf_counter()
            scores = -model.score_samples(all_x)
            inference_seconds = time.perf_counter() - start
            threshold = float(quantile(list(train_scores), .99))
            score_map = {row["instance_uuid"]: float(score) for row, score in zip(rows, scores)}
            scores_by_run[(batch, name)] = score_map
            old_runtime = None
            if reuse_method:
                np.testing.assert_allclose(scores, [old_scores[(batch, reuse_method, row["instance_uuid"])] for row in rows], rtol=0, atol=1e-12)
                old_run = next(r for r in previous["runs"] if r["train_batch"] == batch and r["method"] == reuse_method)
                assert abs(old_run["threshold"] - threshold) < 1e-12
                old_runtime = old_run["runtime_seconds"]
            evaluation = evaluate(score_map, threshold, test_ids, known_ids)
            exact_ids = {u for u in test_ids if rows_by_uuid[u]["exact_candidate_metadata"] == "1"}
            sub = evaluate(score_map, threshold, exact_ids, known_ids)
            report["runs"].append({
                "train_batch": batch, "test_batch": direction["test_batch"], "group": name,
                "train_count": len(train), "test_count": len(test_ids), "features": fields,
                "model_reused": bool(reuse_method), "threshold": threshold,
                "training_seconds": train_seconds, "model_load_seconds": load_seconds,
                "training_score_inference_seconds": train_inference_seconds,
                "all_2069_score_inference_seconds": inference_seconds,
                "prior_combined_runtime_seconds": old_runtime,
                "all_eligible_test": evaluation, "legacy_exact_test": sub,
                "tree_split_usage": split_usage(model, fields),
            })
            for row in rows:
                uuid = row["instance_uuid"]
                rank = rank_bounds([score_map[u] for u in test_ids | known_ids], score_map[uuid]) if uuid in test_ids | known_ids else ("", "")
                outputs.append({
                    "train_batch": batch, "group": name, "instance_uuid": uuid,
                    "score": score_map[uuid], "train_p99_threshold": threshold,
                    "alert": int(score_map[uuid] > threshold),
                    "rank_best": rank[0], "rank_worst": rank[1],
                    "role_metadata": "train" if uuid in train_ids else "test" if uuid in test_ids else "known_anomalous" if uuid in known_ids else "excluded",
                })
        for a, b in itertools.combinations(sets, 2):
            ids = sorted(test_ids | known_ids)
            va, vb = ([scores_by_run[(batch, name)][u] for u in ids] for name in (a, b))
            report["within_direction_score_stability"].append({
                "train_batch": batch, "group_a": a, "group_b": b,
                "spearman_same_evaluation_set": score_spearman(va, vb)})
    for name in sets:
        batches = [d["train_batch"] for d in directions]
        a, b = ([scores_by_run[(batch, name)][u] for u in sorted(known_ids)] for batch in batches)
        report["cross_direction_known_score_stability"].append({
            "group": name, "spearman_four_known": score_spearman(a, b),
            "mean_absolute_score_difference": float(np.mean(np.abs(np.array(a) - np.array(b)))),
            "limitation": "仅四个共同已知实例，测试候选集合不同，不构成稳定性统计推断。",
        })
    report["duration_max_gap_correlations"] = [r for r in correlations if r["feature_a"] == "duration_seconds" and r["feature_b"] == "gap_max_seconds"]
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as directory:
        stage = Path(directory)
        for filename, rows_out in ((OUTPUTS[0], outputs), (OUTPUTS[1], correlations)):
            with (stage / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, list(rows_out[0]))
                writer.writeheader()
                writer.writerows(rows_out)
        (stage / OUTPUTS[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (stage / OUTPUTS[3]).write_text(markdown(report), encoding="utf-8")
        for path, expected in hashes.items():
            with Path(path).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("历史输入或模型在运行中变化")
        for filename in OUTPUTS:
            os.link(stage / filename, output / filename)
    return report


def markdown(report):
    lines = ["# Isolation Forest 时序消融与解释性分析", "",
             "沿用 run_02 实例划分、300 棵树、max_samples=256、random_state=42、训练分数 P99。A/B/C 新训练，D/E 复用已保存模型，逐实例评分和阈值与 run_02 核对一致。",
             "异常 UUID 顺序为 1643649d、544fd51c、a445709b、ae651dff。排名池为目标批次合格候选加四个已知异常，并列采用区间。",
             "", "|训练批次|组|异常排名区间|Top5/10/20 确定命中|候选告警|新训练秒|推理2069实例秒|",
             "|---|---|---|---|---|---|---|"]
    for run in report["runs"]:
        e = run["all_eligible_test"]
        ranks = list(e["known_anomaly_rank_intervals"].values())
        hits = [e["topk"][str(k)]["definite_hits"] for k in (5, 10, 20)]
        lines.append(f"|{run['train_batch']}|{run['group']}|{ranks}|{hits}|{e['candidate_alerts']}/{e['candidate_count']}|{run['training_seconds']}|{run['all_2069_score_inference_seconds']:.6f}|")
    lines += ["", "## 相关性与模板分布", "",
              f"跨度与最大间隔的分批次相关性为 {report['duration_max_gap_correlations']}。",
              "全部时序 Pearson/Spearman 见 feature_correlation.csv，分批次分布和模板常量审计见 JSON。"]
    for batch, data in report["batch_analysis"].items():
        lines.append(f"{batch} 模板统计列中，常量 {data['template_constant_count']} 个，非恒定但众数占比>=99% 的 {data['template_near_constant_nonconstant_count']} 个。")
    for run in report["runs"]:
        if run["group"] in ("D_all_time", "E_time_templates"):
            usage = run["tree_split_usage"]
            lines.append(f"{run['train_batch']} {run['group']} 保留 {len(usage['retained_features'])} 列，时间特征分裂节点占比 {usage['temporal_split_fraction']:.6f}。")
    lines += ["", "## 解释范围", "",
              "A/B/C 中四个异常存在完全同分，跨方向四点 Spearman 因常量而未定义，JSON 以 null 表示。不得利用 UUID 排序人为区分同分异常。",
              "大量模板常量经训练集方差筛选已删除，不能将其直接认定为融合效果下降的原因。仍保留的模板变化可能使树分裂分配到事件数量或模板计数上，改变对时间偏离的敏感性。分裂频次与消融结果只能提供机制线索，无法证明单条评分由某特征主导。",
              "不同组合原始评分不具备统一尺度；报告同时给出固定评价池排名、告警和 Spearman 排序相关性。跨方向只对共同四个异常计算分数变化，不能将不同测试集合的排名差直接解释为模型不稳定。",
              "本次不依据结果调参、改阈值或筛选实例。是否继续优化应优先考虑独立验证数据与采集批次混杂，保留单特征和时长阈值作为后续参照。",
              "", "## 局限", "", *report["limitations"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "feature_ablation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.output_dir, EXPERIMENTS / "isolation_forest" / "run_02",
        EXPERIMENTS.parent / "models" / "isolation_forest" / "run_02")
    logging.info("特征消融完成：%s", args.output_dir)


if __name__ == "__main__":
    main()
