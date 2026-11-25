"""首阶段真实数据验收，须先运行导入与 Uvicorn。结果写新文件。"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import time
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from src.web.config import Settings
from src.web.database import connect
from src.web.importer import normalize_time


def main():
    settings = Settings()
    measurements = []

    def get(path, query=None, status=200):
        url = "http://127.0.0.1:8000" + path
        if query:
            url += "?" + urlencode(query)
        started = time.perf_counter()
        try:
            with urlopen(url, timeout=60) as response:
                code, body = response.status, response.read()
        except HTTPError as error:
            code, body = error.code, error.read()
        measurements.append({"route": path, "query": query, "status": code,
                             "elapsed_ms": (time.perf_counter()-started)*1000})
        assert code == status, (url, code, body[:300])
        return json.loads(body) if path != "/docs" else body

    get("/docs")
    get("/api/v1/health")
    data = get("/api/v1/datasets")["data"]["items"][0]
    dataset_id = data["id"]
    severity, component, sources = Counter(), Counter(), Counter()
    total = linked = selected = keyword = 0
    with settings.manifest()["openstack_structured.csv"].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            linked += bool(row["instance_uuid"])
            severity[row["severity"]] += 1
            component[row["component"]] += 1
            sources[row["source_file"]] += 1
            keyword += "status: 503" in row["message"] or "status: 503" in row["raw_text"]
            selected += row["severity"] == "INFO" and row["component"] == "nova-api" and "2017-05-16" <= row["event_time"] < "2017-05-17"
    query = {"dataset_id": dataset_id}
    overview = get("/api/v1/overview", query)["data"]
    assert overview["log_count"] == total == 207820
    assert overview["linked_log_count"] == linked
    assert overview["severity_counts"] == severity
    assert overview["component_counts"] == component
    assert overview["source_counts"] == sources
    assert overview["instance_count"] == 2069 and overview["template_count"] == 130
    first = get("/api/v1/logs", {**query, "page_size": 10})["data"]
    second = get("/api/v1/logs", {**query, "page_size": 10, "cursor": first["pagination"]["next_cursor"]})["data"]
    assert not {r["id"] for r in first["items"]} & {r["id"] for r in second["items"]}
    page2 = get("/api/v1/logs", {**query, "page_size": 10, "page": 2})["data"]
    assert second["items"] == page2["items"]
    filtered = get("/api/v1/logs", {**query, "severity":"INFO", "component":"nova-api",
                                   "start_time":"2017-05-16T00:00:00", "end_time":"2017-05-17T00:00:00"})["data"]
    assert filtered["pagination"]["total"] == selected
    assert get("/api/v1/logs", {**query, "q":"status: 503"})["data"]["pagination"]["total"] == keyword
    details = get("/api/v1/logs/" + str(first["items"][0]["id"]), query)["data"]
    assert normalize_time(details["event_time_raw"]) == details["event_time"]
    trend = get("/api/v1/overview/trend", {**query, "bucket":"hour"})["data"]
    assert sum(r["count"] for r in trend) == total
    for invalid in ({"page_size":101}, {"page":0}, {"start_time":"invalid"}, {"severity":"bad"}, {"cursor":"broken"}):
        get("/api/v1/logs", {**query, **invalid}, 422)
    get("/api/v1/logs/999999999", query, 404)
    get("/api/v1/log-options", query)
    # 重复导入通过 CLI 另行执行后，数据库仍仅保留一版实际日志。
    with connect(settings.database) as db:
        assert db.execute("SELECT COUNT(*) FROM logs").fetchone()[0] == total
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        manifest = json.loads(db.execute("SELECT manifest_json FROM datasets WHERE id=?", (dataset_id,)).fetchone()[0])
        indexes = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index'")]
        plans = [tuple(r) for r in db.execute("EXPLAIN QUERY PLAN SELECT id FROM logs WHERE dataset_id=? AND severity=? ORDER BY event_time DESC,id DESC LIMIT 50", (dataset_id,"INFO"))]
    for filename, item in manifest.items():
        with settings.manifest()[filename].open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest() == item["sha256"]
    result = {"status":"passed", "log_count":total, "linked_logs":linked,
              "severity_counts":severity,"component_counts":component,"source_counts":sources,
              "measurements":measurements,"indexes":indexes,"query_plan":plans,
              "input_hashes_unchanged":True}
    output = settings.database.parent / ("backend_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
