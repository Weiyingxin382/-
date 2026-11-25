"""第三阶段迁移及已有研究产物的显式离线导入。"""

import csv
import hashlib
import json
from pathlib import Path

from src.analysis.temporal_features import TIME_FEATURES
from src.web.config import ROOT, Settings
from src.web.database import connect, initialize


SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_sets(
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
 version TEXT NOT NULL, manifest_json TEXT NOT NULL, model_allowlist_json TEXT NOT NULL,
 UNIQUE(dataset_id,version), UNIQUE(dataset_id,id));
CREATE TABLE IF NOT EXISTS instance_features(
 dataset_id INTEGER NOT NULL, feature_set_id INTEGER NOT NULL, instance_id INTEGER NOT NULL,
 features_json TEXT NOT NULL, phase_evidence_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
 PRIMARY KEY(feature_set_id,instance_id),
 FOREIGN KEY(dataset_id,instance_id) REFERENCES instances(dataset_id,id),
 FOREIGN KEY(dataset_id,feature_set_id) REFERENCES feature_sets(dataset_id,id));
CREATE TABLE IF NOT EXISTS model_registry(
 id INTEGER PRIMARY KEY, model_key TEXT UNIQUE NOT NULL, method TEXT NOT NULL,
 training_batch TEXT NOT NULL, artifact_path TEXT, artifact_sha256 TEXT,
 feature_order_json TEXT NOT NULL, dependency_versions_json TEXT NOT NULL,
 threshold REAL NOT NULL, source_json TEXT NOT NULL, config_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS detection_runs(
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
 feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id),
 model_id INTEGER NOT NULL REFERENCES model_registry(id),
 origin TEXT NOT NULL, state TEXT NOT NULL, evaluation_set_sha256 TEXT NOT NULL,
 scope_json TEXT NOT NULL, threshold_snapshot REAL NOT NULL,
 UNIQUE(dataset_id,feature_set_id,model_id,origin), UNIQUE(dataset_id,id));
CREATE TABLE IF NOT EXISTS instance_scores(
 run_id INTEGER NOT NULL, dataset_id INTEGER NOT NULL, instance_id INTEGER NOT NULL,
 score REAL NOT NULL, is_alert INTEGER NOT NULL, sample_role TEXT NOT NULL,
 included_in_ranking INTEGER NOT NULL, rank_best INTEGER, rank_worst INTEGER,
 PRIMARY KEY(run_id,instance_id),
 FOREIGN KEY(dataset_id,run_id) REFERENCES detection_runs(dataset_id,id),
 FOREIGN KEY(dataset_id,instance_id) REFERENCES instances(dataset_id,id));
CREATE INDEX IF NOT EXISTS idx_scores_rank ON instance_scores(run_id,included_in_ranking,score DESC,instance_id);
INSERT OR IGNORE INTO schema_migrations(version) VALUES(2);
"""


def migrate(path):
    with connect(path) as db:
        db.executescript(SCHEMA)
        db.commit()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_analysis(settings, dataset_id=1):
    initialize(settings.database)
    migrate(settings.database)
    experiments = ROOT / "experiments"
    research = experiments / "isolation_forest" / "run_02"
    feature_path = experiments / "temporal_features" / "instance_features.csv"
    feature_report_path = experiments / "temporal_features" / "temporal_feature_report.json"
    feasibility_path = experiments / "instance_feasibility" / "instance_distribution_report.json"
    config = json.loads((research / "experiment_config.json").read_text(encoding="utf-8"))
    results = json.loads((research / "experiment_results.json").read_text(encoding="utf-8"))
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    for path, expected in config["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("研究输入哈希与 run_02 不符")
    tracked = [feature_path, feature_report_path, feasibility_path, research / "experiment_config.json",
               research / "experiment_results.json", research / "anomaly_scores.csv"]
    manifest = {str(path): digest(path) for path in tracked}
    version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with connect(settings.database) as db, db:
        db.execute("BEGIN IMMEDIATE")
        data = db.execute("SELECT * FROM datasets WHERE id=? AND state='ready'", (dataset_id,)).fetchone()
        if not data:
            raise ValueError("请先导入 ready 日志数据集")
        for name, item in json.loads(data["manifest_json"]).items():
            expected = feature_report["input_sha256"].get(str(settings.input_dir / name))
            if expected and expected != item["sha256"]:
                raise ValueError("日志快照与特征来源不一致")
        previous = db.execute("SELECT id FROM feature_sets WHERE dataset_id=? AND version=?", (dataset_id, version)).fetchone()
        if previous:
            return previous["id"]
        feature_set = db.execute("INSERT INTO feature_sets(dataset_id,version,manifest_json,model_allowlist_json) VALUES(?,?,?,?)",
                                 (dataset_id, version, json.dumps(manifest), json.dumps(feature_report["model_feature_allowlist"]))).lastrowid
        instance_ids = {r["instance_uuid"]: r for r in db.execute("SELECT * FROM instances WHERE dataset_id=?", (dataset_id,))}
        seen = set()
        with feature_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                uuid = row["instance_uuid"]
                instance = instance_ids[uuid]
                if uuid in seen or int(row["log_count"]) != instance["log_count"]:
                    raise ValueError("实例重复或关联行数不一致")
                seen.add(uuid)
                values = {k: float(row[k]) if row[k] != "" else None for k in feature_report["model_feature_allowlist"]}
                metadata = {"lifecycle_complete": feasibility["instances"][uuid]["observed_lifecycle_complete"],
                            "candidate_legacy": int(row["candidate_metadata"]),
                            "exact_candidate_legacy": int(row["exact_candidate_metadata"]),
                            "sources": feasibility["instances"][uuid]["source_counts"]}
                db.execute("INSERT INTO instance_features VALUES(?,?,?,?,?,?)",
                           (dataset_id, feature_set, instance["id"], json.dumps(values, allow_nan=False),
                            json.dumps(feature_report["phase_evidence_by_instance"][uuid]), json.dumps(metadata)))
        if seen != instance_ids.keys():
            raise ValueError("特征实例集合与数据库不一致")
        with (research / "anomaly_scores.csv").open(encoding="utf-8", newline="") as stream:
            score_rows = list(csv.DictReader(stream))
        for run in results["runs"]:
            if run["method"] not in ("duration_p99", "if_time"):
                continue
            batch = run["train_batch"]
            method = run["method"]
            model_key = batch.removesuffix(".log") + "_" + method + "_run02"
            artifact = ROOT / "models" / "isolation_forest" / "run_02" / (batch.removesuffix(".log") + "_if_time.pkl") if method == "if_time" else None
            source = {"experiment_key": "if_run_02", "test_batch": run["test_batch"],
                      "result_sha256": manifest[str(research / "experiment_results.json")]}
            db.execute("""INSERT OR IGNORE INTO model_registry(model_key,method,training_batch,artifact_path,artifact_sha256,
                       feature_order_json,dependency_versions_json,threshold,source_json,config_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (model_key, method, batch, str(artifact) if artifact else None,
                        digest(artifact) if artifact and artifact.exists() else None,
                        json.dumps(list(TIME_FEATURES) if method == "if_time" else ["duration_seconds"]),
                        json.dumps(config["dependencies"]), run["threshold"], json.dumps(source), json.dumps(config)))
            model = db.execute("SELECT * FROM model_registry WHERE model_key=?", (model_key,)).fetchone()
            if model["threshold"] != run["threshold"]:
                raise ValueError("登记模型阈值发生变化")
            evaluation_ids = sorted(set(run["test_uuids"]) | {r["instance_uuid"] for r in run["known_anomalies"]})
            evaluation_hash = hashlib.sha256(json.dumps(evaluation_ids).encode()).hexdigest()
            run_id = db.execute("""INSERT INTO detection_runs(dataset_id,feature_set_id,model_id,origin,state,evaluation_set_sha256,
                               scope_json,threshold_snapshot) VALUES(?,?,?,'experiment','succeeded',?,?,?)""",
                                (dataset_id, feature_set, model["id"], evaluation_hash,
                                 json.dumps({"train_batch": batch, "test_batch": run["test_batch"],
                                             "train_count": run["train_count"], "test_count": run["test_count"],
                                             "source_experiment_key": "if_run_02"}), run["threshold"])).lastrowid
            matched = [r for r in score_rows if r["train_batch"] == batch and r["method"] == method]
            if len(matched) != len(instance_ids):
                raise ValueError("历史评分数量不一致")
            for row in matched:
                db.execute("INSERT INTO instance_scores VALUES(?,?,?,?,?,?,?,?,?)",
                           (run_id, dataset_id, instance_ids[row["instance_uuid"]]["id"], float(row["score"]),
                            int(row["alert"]), row["sample_role"], int(row["instance_uuid"] in evaluation_ids),
                            int(row["test_rank_best"]) if row["test_rank_best"] else None,
                            int(row["test_rank_worst"]) if row["test_rank_worst"] else None))
        for path, expected in manifest.items():
            if digest(path) != expected:
                raise ValueError("源文件在导入期间变化")
        return feature_set


if __name__ == "__main__":
    print("feature_set_id =", import_analysis(Settings()))
