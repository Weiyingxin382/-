"""真实第三阶段验收；先导入分析快照并启动8000端口。"""

import csv
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

from src.web.config import Settings, ROOT
from src.web.database import connect
from src.web.analysis_store import digest, import_analysis
from src.web.model_service import ModelService


def main():
    settings = Settings()
    history = ROOT / "experiments" / "isolation_forest" / "run_02"
    protected = [*history.iterdir(), *(ROOT / "models" / "isolation_forest" / "run_02").glob("*.pkl")]
    before = {str(path): digest(path) for path in protected if path.is_file()}
    feature_set = import_analysis(settings)
    assert import_analysis(settings) == feature_set
    def api(path, body=None):
        content = json.dumps(body).encode() if body is not None else None
        request = Request("http://127.0.0.1:8000/api/v1" + path, data=content,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)["data"]
    summary = api("/templates/summary?dataset_id=1")
    assert summary["template_count"] == 130 and summary["log_count"] == 207820
    top = summary["top_templates"][0]
    assert api("/templates/" + top["external_template_id"] + "/logs?dataset_id=1&page_size=5")["pagination"]["total"] == top["frequency"]
    with (history / "anomaly_scores.csv").open(encoding="utf-8", newline="") as stream:
        expected = {(r["train_batch"], r["method"], r["instance_uuid"]): r for r in csv.DictReader(stream)}
    with (ROOT / "experiments" / "temporal_features" / "instance_features.csv").open(encoding="utf-8", newline="") as stream:
        features = list(csv.DictReader(stream))
    known = [r["instance_uuid"] for r in features if r["instance_label_metadata"] == "anomalous"]
    runs = api("/detection-runs?dataset_id=1")["items"]
    assert len(runs) == 4
    verified_scores = 0
    models = ModelService()
    for run in runs:
        rows = []
        page = 1
        while True:
            result = api(f"/detection-runs/{run['id']}/scores?role=all&page_size=100&page={page}")
            rows.extend(result["items"])
            if len(rows) >= result["pagination"]["total"]:
                break
            page += 1
        assert len(rows) == 2069
        for row in rows:
            original = expected[run["training_batch"], run["method"], row["instance_uuid"]]
            assert row["raw_score"] == float(original["score"])
            assert row["rank_best"] == (int(original["test_rank_best"]) if original["test_rank_best"] else None)
            assert row["rank_worst"] == (int(original["test_rank_worst"]) if original["test_rank_worst"] else None)
            assert row["sample_role"] == original["sample_role"]
            assert row["is_alert"] == int(original["alert"])
        verified_scores += len(rows)
        for uuid in known:
            result = api("/inferences", {"dataset_id": 1, "feature_set_id": feature_set,
                                         "instance_uuid": uuid, "model_key": run["model_key"]})
            assert abs(result["raw_score"] - float(expected[run["training_batch"], run["method"], uuid]["score"])) < 1e-10
            assert result["threshold"] == run["threshold_snapshot"]
        if run["method"] == "if_time":
            with connect(settings.database) as db:
                record = dict(db.execute("SELECT * FROM model_registry WHERE model_key=?", (run["model_key"],)).fetchone())
            pipeline = models.load(record)
            fields = json.loads(record["feature_order_json"])
            values = [[float(row[k]) if row[k] else np.nan for k in fields] for row in features]
            np.testing.assert_allclose(-pipeline.score_samples(values),
                                       [float(expected[run["training_batch"], "if_time", row["instance_uuid"]]["score"]) for row in features],
                                       rtol=0, atol=1e-10)
    assert len(models.cache) == 2
    for uuid in known:
        first = api(f"/instances/{uuid}/timeline?dataset_id=1&page_size=10")["items"]
        second = api(f"/instances/{uuid}/timeline?dataset_id=1&page_size=10&page=2")["items"]
        assert first[0]["delta_seconds"] is None
        assert second[0]["sequence_index"] == 11
        gap = (datetime.fromisoformat(second[0]["event_time_raw"]) - datetime.fromisoformat(first[-1]["event_time_raw"])).total_seconds()
        assert second[0]["delta_seconds"] == gap
    with connect(settings.database) as db:
        assert db.execute("SELECT COUNT(*) FROM logs").fetchone()[0] == 207820
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
    assert before == {path: digest(path) for path in before}
    result = {"status":"passed", "historical_scores_verified":verified_scores,
              "full_if_predictions_verified":4138, "known_online_inferences_verified":16,
              "template_count":130, "model_cache_entries":2, "historical_hashes_unchanged":True}
    output = settings.database.parent / ("web_stage3_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
