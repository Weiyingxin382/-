"""登记模型按需缓存，仅执行已有 Pipeline 推理。"""

import importlib.metadata
import json
import math
from pathlib import Path
import pickle
from threading import Lock
import time

from src.analysis.temporal_features import TIME_FEATURES
from src.web.analysis_store import digest
from src.web.config import ROOT
from src.web.errors import AppError


class ModelService:
    def __init__(self):
        self.cache = {}
        self.lock = Lock()

    def load(self, record):
        if record["method"] == "duration_p99":
            return None
        try:
            path = Path(record["artifact_path"]).resolve()
            trusted = (ROOT / "models" / "isolation_forest" / "run_02").resolve()
            if path.parent != trusted or path.name not in ("openstack_normal1_if_time.pkl", "openstack_normal2_if_time.pkl"):
                raise ValueError("工件不在可信登记列表")
            for name, expected in json.loads(record["dependency_versions_json"]).items():
                if importlib.metadata.version(name) != expected:
                    raise ValueError("运行依赖不兼容：" + name)
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size, record["artifact_sha256"])
            with self.lock:
                cached = self.cache.get(record["model_key"])
                if cached and cached[0] == stamp:
                    return cached[1]
                if digest(path) != record["artifact_sha256"]:
                    raise ValueError("模型哈希不一致")
                with path.open("rb") as stream:
                    artifact = pickle.load(stream)
                if artifact["features"] != json.loads(record["feature_order_json"]) or artifact["features"] != list(TIME_FEATURES):
                    raise ValueError("模型特征顺序不兼容")
                if artifact["config"]["isolation_forest_params"] != json.loads(record["config_json"])["isolation_forest_params"]:
                    raise ValueError("模型预处理配置来源不一致")
                self.cache[record["model_key"]] = (stamp, artifact["pipeline"])
                return artifact["pipeline"]
        except (OSError, ValueError, KeyError, ImportError, pickle.UnpicklingError, EOFError) as error:
            raise AppError(503, "MODEL_UNAVAILABLE", "登记模型不可用", {"reason": str(error)}) from error

    def describe(self, row):
        record = dict(row)
        state, reason = "available", None
        try:
            self.load(record)
        except AppError as error:
            state, reason = "unavailable", error.details
        return {
            "model_key": record["model_key"], "method": record["method"], "version": "run_02",
            "training_batch": record["training_batch"], "state": state, "unavailable_reason": reason,
            "threshold": record["threshold"], "feature_order": json.loads(record["feature_order_json"]),
            "score_definition": "duration_seconds" if record["method"] == "duration_p99" else "negative_score_samples",
            "threshold_source_experiment": json.loads(record["source_json"]),
        }

    def infer(self, record, feature):
        expected = json.loads(record["config_json"]).get("input_sha256", {})
        actual = json.loads(feature["feature_manifest_json"])
        for path, checksum in expected.items():
            if path not in actual or actual[path] != checksum:
                raise AppError(422, "FEATURE_VERSION_MISMATCH", "特征快照与登记实验的来源不兼容")
        values = json.loads(feature["features_json"])
        metadata = json.loads(feature["metadata_json"])
        fields = json.loads(record["feature_order_json"])
        missing = [field for field in fields if values.get(field) is None or not math.isfinite(values[field])]
        if missing or not metadata["lifecycle_complete"]:
            raise AppError(422, "INSTANCE_INELIGIBLE", "实例特征或生命周期不满足推理要求",
                           {"missing_features": missing, "lifecycle_complete": metadata["lifecycle_complete"]})
        start = time.perf_counter()
        model = self.load(record)
        score = float(values["duration_seconds"]) if model is None else float(-model.score_samples([[values[f] for f in fields]])[0])
        return {"model_key": record["model_key"], "model_version": "run_02", "origin": "online_inference",
                "raw_score": score, "threshold": record["threshold"], "is_alert": score > record["threshold"],
                "score_definition": "duration_seconds" if model is None else "negative_score_samples",
                "threshold_source": json.loads(record["source_json"]),
                "duration_ratio": score / record["threshold"] if model is None else None,
                "rank_best": None, "rank_worst": None,
                "inference_ms": (time.perf_counter() - start) * 1000}
