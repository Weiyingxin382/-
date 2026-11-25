"""清单 CSV 导入，可通过 python -m src.web.importer 执行。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from src.data.log_parser import PARSED_FIELDS
from src.web.config import Settings
from src.web.database import connect, initialize, has_fts
from src.web.errors import AppError


LOGGER = logging.getLogger(__name__)
LOG_FIELDS = {"source_file", "line_number", "raw_text", *PARSED_FIELDS}


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize_time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        raise ValueError("日志时间不接受时区转换")
    return parsed.isoformat(timespec="microseconds")


def reader(stream, required):
    result = csv.DictReader(stream)
    if not result.fieldnames or not required.issubset(result.fieldnames) or len(result.fieldnames) != len(set(result.fieldnames)):
        raise ValueError("CSV 表头缺失、重复或不符合清单")
    return result


def create_job(settings, key, manifest_key, dataset_name):
    if manifest_key != settings.manifest_key:
        raise AppError(422, "UNKNOWN_MANIFEST", "未登记的输入清单")
    payload = json.dumps({"input_manifest_key": manifest_key, "dataset_name": dataset_name}, sort_keys=True)
    with connect(settings.database) as db, db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT * FROM import_jobs WHERE idempotency_key=?", (key,)).fetchone()
        if existing:
            if existing["request_json"] != payload:
                raise AppError(409, "IDEMPOTENCY_CONFLICT", "幂等键对应不同请求")
            return existing["id"], False
        cursor = db.execute("INSERT INTO import_jobs(state,idempotency_key,request_json,expected_rows,created_at) VALUES('queued',?,?,?,?)",
                            (key, payload, settings.expected_logs, now()))
        return cursor.lastrowid, True


def perform_import(settings, job_id):
    paths = settings.manifest()
    manifest = {}
    count = 0
    try:
        with connect(settings.database) as db, db:
            db.execute("BEGIN IMMEDIATE")
            claimed = db.execute("UPDATE import_jobs SET state='running',started_at=? WHERE id=? AND state='queued'", (now(), job_id))
            if claimed.rowcount != 1:
                return
        for name, path in paths.items():
            with path.open("rb") as stream:
                manifest[name] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(), "path": str(path)}
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        with connect(settings.database) as db, db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT * FROM import_jobs WHERE id=?", (job_id,)).fetchone()
            if not job or job["state"] != "running":
                return
            previous = db.execute("SELECT id FROM datasets WHERE manifest_sha256=? AND state='ready'", (fingerprint,)).fetchone()
            if previous:
                db.execute("UPDATE import_jobs SET state='succeeded',dataset_id=?,progress_rows=?,finished_at=? WHERE id=?",
                           (previous["id"], settings.expected_logs or 0, now(), job_id))
                return
            name = json.loads(job["request_json"])["dataset_name"]
            dataset = db.execute("INSERT INTO datasets(name,version,manifest_sha256,manifest_json,state,created_at) VALUES(?,?,?,?,'importing',?)",
                                 (name, fingerprint[:16], fingerprint, json.dumps(manifest), now())).lastrowid
            labels = {}
            with paths["openstack_instance_labels.csv"].open(encoding="utf-8-sig", newline="") as stream:
                for row in reader(stream, {"instance_uuid", "anomaly_label"}):
                    uuid = str(UUID(row["instance_uuid"]))
                    if uuid in labels or row["anomaly_label"] not in ("unknown", "anomalous", "normal"):
                        raise ValueError("实例标签重复或无效")
                    labels[uuid] = row["anomaly_label"]
                    db.execute("INSERT INTO instances(dataset_id,instance_uuid,label,label_evidence_json) VALUES(?,?,?,?)",
                               (dataset, uuid, labels[uuid], json.dumps({"source": "openstack_instance_labels.csv", "scope": "instance"})))
            instance_ids = {r["instance_uuid"]: r["id"] for r in db.execute("SELECT id,instance_uuid FROM instances WHERE dataset_id=?", (dataset,))}
            template_ids = {}
            with paths["openstack_template_statistics.csv"].open(encoding="utf-8-sig", newline="") as stream:
                for row in reader(stream, {"template_id", "normalized_message", "frequency", "component_counts", "severity_counts"}):
                    frequency = int(row["frequency"])
                    if frequency < 1 or row["template_id"] in template_ids:
                        raise ValueError("模板频次或 ID 无效")
                    for field in ("component_counts", "severity_counts"):
                        values = json.loads(row[field])
                        if not isinstance(values, dict) or any(type(v) is not int or v < 0 for v in values.values()) or sum(values.values()) != frequency:
                            raise ValueError("模板分布数量不一致")
                    key = db.execute("INSERT INTO templates(dataset_id,parser_version,external_template_id,normalized_message,frequency,component_counts_json,severity_counts_json) VALUES(?,'rule_v1',?,?,?,?,?)",
                                     (dataset, row["template_id"], row["normalized_message"], frequency, row["component_counts"], row["severity_counts"])).lastrowid
                    template_ids[row["template_id"]] = (key, row["normalized_message"])
            source_ids = {}
            batch = []
            sql = "INSERT INTO logs(dataset_id,source_id,line_number,input_order,component,severity,module,event_time_raw,event_time,process_id,message,raw_text,instance_id,template_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            with paths["openstack_structured.csv"].open(encoding="utf-8-sig", newline="") as original, paths["openstack_template_analysis.csv"].open(encoding="utf-8-sig", newline="") as analysis:
                for count, (row, enriched) in enumerate(itertools.zip_longest(reader(original, LOG_FIELDS), reader(analysis, LOG_FIELDS | {"template_id", "normalized_message"})), 1):
                    if row is None or enriched is None or any(enriched.get(k) != v for k, v in row.items()) or None in row:
                        raise ValueError(f"结构化与模板 CSV 原字段不一致，数据行 {count}")
                    source = row["source_file"]
                    if source not in source_ids:
                        source_ids[source] = db.execute("INSERT INTO log_sources(dataset_id,name) VALUES(?,?)", (dataset, source)).lastrowid
                    uuid = str(UUID(row["instance_uuid"])) if row["instance_uuid"] else None
                    if uuid and uuid not in instance_ids:
                        raise ValueError(f"实例标签表缺失 UUID，数据行 {count}")
                    key, template = template_ids[enriched["template_id"]]
                    if template != enriched["normalized_message"]:
                        raise ValueError(f"模板正文不一致，数据行 {count}")
                    line, pid = int(row["line_number"]), int(row["process_id"])
                    if line <= 0 or pid < 0:
                        raise ValueError(f"行号或进程号非法，数据行 {count}")
                    batch.append((dataset, source_ids[source], line, count, row["component"], row["severity"], row["module"],
                                  row["event_time"], normalize_time(row["event_time"]), pid, row["message"], row["raw_text"], instance_ids.get(uuid), key))
                    if len(batch) == 2000:
                        db.executemany(sql, batch)
                        batch.clear()
                db.executemany(sql, batch)
            actual = db.execute("SELECT COUNT(*) FROM logs WHERE dataset_id=?", (dataset,)).fetchone()[0]
            for value, expected, label in ((actual, settings.expected_logs, "日志"), (len(instance_ids), settings.expected_instances, "实例"), (len(template_ids), settings.expected_templates, "模板")):
                if expected is not None and value != expected:
                    raise ValueError(f"{label}数量不一致 {value}/{expected}")
            mismatch = db.execute("SELECT t.id FROM templates t LEFT JOIN logs l ON l.dataset_id=t.dataset_id AND l.template_id=t.id WHERE t.dataset_id=? GROUP BY t.id HAVING COUNT(l.id)!=t.frequency", (dataset,)).fetchone()
            if mismatch:
                raise ValueError("实际模板频次不一致")
            db.execute("UPDATE instances SET log_count=(SELECT COUNT(*) FROM logs WHERE dataset_id=instances.dataset_id AND instance_id=instances.id),first_event=(SELECT MIN(event_time) FROM logs WHERE dataset_id=instances.dataset_id AND instance_id=instances.id),last_event=(SELECT MAX(event_time) FROM logs WHERE dataset_id=instances.dataset_id AND instance_id=instances.id) WHERE dataset_id=?", (dataset,))
            db.execute("UPDATE log_sources SET row_count=(SELECT COUNT(*) FROM logs WHERE dataset_id=log_sources.dataset_id AND source_id=log_sources.id),linked_instance_count=(SELECT COUNT(DISTINCT instance_id) FROM logs WHERE dataset_id=log_sources.dataset_id AND source_id=log_sources.id),first_event=(SELECT MIN(event_time) FROM logs WHERE dataset_id=log_sources.dataset_id AND source_id=log_sources.id),last_event=(SELECT MAX(event_time) FROM logs WHERE dataset_id=log_sources.dataset_id AND source_id=log_sources.id) WHERE dataset_id=?", (dataset,))
            if has_fts(db):
                db.execute("INSERT INTO logs_fts(rowid,message,raw_text) SELECT id,message,raw_text FROM logs WHERE dataset_id=?", (dataset,))
            for filename, path in paths.items():
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != manifest[filename]["sha256"]:
                        raise ValueError("导入过程中输入文件变化")
            summary = {"log_count": actual, "instance_count": len(instance_ids), "template_count": len(template_ids)}
            db.execute("UPDATE datasets SET state='ready',ready_at=?,overview_json=? WHERE id=?", (now(), json.dumps(summary), dataset))
            db.execute("UPDATE import_jobs SET state='succeeded',dataset_id=?,progress_rows=?,finished_at=? WHERE id=?", (dataset, actual, now(), job_id))
            LOGGER.info("导入完成 dataset=%s logs=%s", dataset, actual)
    except Exception as error:
        LOGGER.exception("导入失败 job=%s row=%s", job_id, count)
        with connect(settings.database) as db, db:
            db.execute("UPDATE import_jobs SET state='failed',error_json=?,finished_at=? WHERE id=?",
                       (json.dumps({"reason": str(error), "data_row": count}, ensure_ascii=False), now(), job_id))


def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    initialize(settings.database)
    job, _ = create_job(settings, str(uuid4()), settings.manifest_key, "OpenStack")
    perform_import(settings, job)
    with connect(settings.database) as db:
        result = dict(db.execute("SELECT * FROM import_jobs WHERE id=?", (job,)).fetchone())
        print(json.dumps(result, ensure_ascii=False))
    return 0 if result["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
