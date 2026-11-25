"""模板、历史实例评分与按需推理路由。"""

import json
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.web.database import connect
from src.web.errors import AppError
from src.web import repositories
from src.web.schemas import LogQuery


class TemplateQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: int = Field(gt=0)
    parser_version: str = "rule_v1"
    q: str | None = Field(default=None, max_length=200)
    component: str | None = None
    severity: str | None = None
    frequency_min: int = Field(default=0, ge=0)
    frequency_max: int | None = Field(default=None, ge=0)
    sort: Literal["frequency_desc", "frequency_asc"] = "frequency_desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def bounds(self):
        if self.frequency_max is not None and self.frequency_max < self.frequency_min:
            raise ValueError("频次上界小于下界")
        return self


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: int = Field(gt=0)
    feature_set_id: int = Field(gt=0)
    instance_uuid: UUID
    model_key: str


def decode_template(row):
    item = dict(row)
    for field in ("component_counts", "severity_counts"):
        item[field] = json.loads(item.pop(field + "_json"))
    item["frequency"] = item.get("matched_frequency", item["global_frequency"])
    return item


def template_rows(db, query):
    clauses = ["t.dataset_id=?", "t.parser_version=?"]
    values = [query.dataset_id, query.parser_version]
    if query.q:
        clauses.append("instr(t.normalized_message,?)>0")
        values.append(query.q)
    filtered = bool(query.component or query.severity)
    for name in ("component", "severity"):
        if getattr(query, name):
            clauses.append("l." + name + "=?")
            values.append(getattr(query, name))
    join = " JOIN logs l ON l.dataset_id=t.dataset_id AND l.template_id=t.id " if filtered else ""
    frequency = "COUNT(l.id)" if filtered else "t.frequency"
    sql = "SELECT t.external_template_id,t.parser_version,t.normalized_message,t.frequency global_frequency,t.component_counts_json,t.severity_counts_json," + frequency + " matched_frequency FROM templates t" + join
    sql += " WHERE " + " AND ".join(clauses)
    if filtered:
        sql += " GROUP BY t.id"
    result = [decode_template(row) for row in db.execute(sql, values)]
    result = [row for row in result if row["matched_frequency"] >= query.frequency_min and
              (query.frequency_max is None or row["matched_frequency"] <= query.frequency_max)]
    return sorted(result, key=lambda row: ((-1 if query.sort == "frequency_desc" else 1) * row["matched_frequency"], row["external_template_id"]))


def instance_row(db, dataset_id, uuid):
    row = db.execute("SELECT * FROM instances WHERE dataset_id=? AND instance_uuid=?", (dataset_id, str(uuid))).fetchone()
    if not row:
        raise AppError(404, "INSTANCE_NOT_FOUND", "实例不存在")
    return dict(row)


def feature_row(db, dataset_id, instance_id, feature_set_id=None):
    sql = """SELECT f.*,v.manifest_json feature_manifest_json FROM instance_features f
             JOIN feature_sets v ON v.id=f.feature_set_id WHERE f.dataset_id=? AND f.instance_id=?"""
    values = [dataset_id, instance_id]
    if feature_set_id:
        sql += " AND f.feature_set_id=?"
        values.append(feature_set_id)
    row = db.execute(sql + " ORDER BY f.feature_set_id DESC LIMIT 1", values).fetchone()
    if not row:
        raise AppError(422, "FEATURES_UNAVAILABLE", "实例特征快照尚未导入或版本不匹配")
    return dict(row)


def build_router(settings, envelope, models):
    router = APIRouter(prefix="/api/v1")

    @router.get("/templates")
    def templates(request: Request, query: Annotated[TemplateQuery, Query()]):
        with connect(settings.database) as db:
            version = repositories.dataset(db, query.dataset_id)
            rows = template_rows(db, query)
        start = (query.page - 1) * query.page_size
        return envelope(request, {"items": rows[start:start+query.page_size],
                                  "pagination": {"total": len(rows), "page": query.page, "page_size": query.page_size}}, version)

    @router.get("/templates/summary")
    def template_summary(request: Request, query: Annotated[TemplateQuery, Query()]):
        from collections import Counter
        with connect(settings.database) as db:
            version = repositories.dataset(db, query.dataset_id)
            rows = template_rows(db, query)
        ordered = sorted(rows, key=lambda row: (-row["matched_frequency"], row["external_template_id"]))
        return envelope(request, {"template_count": len(rows), "log_count": sum(r["matched_frequency"] for r in rows),
                                  "frequency_histogram": dict(Counter(r["matched_frequency"] for r in rows)),
                                  "top_templates": ordered[:10], "bottom_templates": ordered[::-1][:10]}, version)

    @router.get("/templates/{external_id}")
    def template_detail(request: Request, external_id: str, dataset_id: int = Query(gt=0), parser_version: str = "rule_v1"):
        with connect(settings.database) as db:
            version = repositories.dataset(db, dataset_id)
            row = db.execute("""SELECT external_template_id,parser_version,normalized_message,frequency global_frequency,
                             component_counts_json,severity_counts_json FROM templates WHERE dataset_id=? AND parser_version=? AND external_template_id=?""",
                             (dataset_id, parser_version, external_id)).fetchone()
            if not row:
                raise AppError(404, "TEMPLATE_NOT_FOUND", "模板不存在")
        return envelope(request, decode_template(row), version)

    @router.get("/templates/{external_id}/logs")
    def template_logs(request: Request, external_id: str, query: Annotated[LogQuery, Query()]):
        with connect(settings.database) as db:
            version = repositories.dataset(db, query.dataset_id)
            if not db.execute("SELECT 1 FROM templates WHERE dataset_id=? AND external_template_id=?", (query.dataset_id, external_id)).fetchone():
                raise AppError(404, "TEMPLATE_NOT_FOUND", "模板不存在")
            query.template_id = external_id
            result = repositories.logs(db, query)
        return envelope(request, result, version)

    @router.get("/feature-sets")
    def feature_sets(request: Request, dataset_id: int = Query(gt=0)):
        with connect(settings.database) as db:
            version = repositories.dataset(db, dataset_id)
            rows = [dict(r) for r in db.execute("SELECT id,version,model_allowlist_json FROM feature_sets WHERE dataset_id=? ORDER BY id", (dataset_id,))]
        for row in rows:
            row["model_allowlist"] = json.loads(row.pop("model_allowlist_json"))
        return envelope(request, rows, version)

    @router.get("/models")
    def model_list(request: Request):
        with connect(settings.database) as db:
            rows = [models.describe(dict(row)) for row in db.execute("SELECT * FROM model_registry ORDER BY id")]
        return envelope(request, rows)

    @router.get("/models/{model_key}")
    def model_detail(request: Request, model_key: str):
        with connect(settings.database) as db:
            row = db.execute("SELECT * FROM model_registry WHERE model_key=?", (model_key,)).fetchone()
            if not row:
                raise AppError(404, "MODEL_NOT_FOUND", "模型未登记")
            result = models.describe(dict(row))
        return envelope(request, result)

    @router.get("/detection-runs")
    def runs(request: Request, dataset_id: int = Query(gt=0), page: int = Query(1, ge=1),
             page_size: int = Query(50, ge=1, le=100), model_key: str | None = None):
        with connect(settings.database) as db:
            version = repositories.dataset(db, dataset_id)
            where = " WHERE r.dataset_id=?"
            args = [dataset_id]
            if model_key:
                where += " AND m.model_key=?"
                args.append(model_key)
            rows = [dict(row) for row in db.execute("""SELECT r.*,m.model_key,m.method,m.training_batch FROM detection_runs r
                        JOIN model_registry m ON m.id=r.model_id""" + where + " ORDER BY r.id LIMIT ? OFFSET ?",
                        [*args, page_size, (page-1)*page_size])]
            total = db.execute("SELECT COUNT(*) FROM detection_runs r JOIN model_registry m ON m.id=r.model_id" + where, args).fetchone()[0]
        for row in rows:
            row["scope"] = json.loads(row.pop("scope_json"))
        return envelope(request, {"items": rows, "pagination": {"total": total, "page": page, "page_size": page_size}}, version)

    @router.get("/detection-runs/{run_id}")
    def run_detail(request: Request, run_id: int):
        with connect(settings.database) as db:
            row = db.execute("SELECT * FROM detection_runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                raise AppError(404, "RUN_NOT_FOUND", "检测运行不存在")
            version = repositories.dataset(db, row["dataset_id"])
            result = dict(row)
            result["scope"] = json.loads(result.pop("scope_json"))
        return envelope(request, result, version)

    @router.get("/detection-runs/{run_id}/scores")
    def scores(request: Request, run_id: int, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100),
               instance_uuid: UUID | None = None, alert: bool | None = None,
               role: Literal["evaluation", "all", "train", "test", "known_anomalous", "excluded"] = "evaluation"):
        with connect(settings.database) as db:
            run = db.execute("SELECT * FROM detection_runs WHERE id=?", (run_id,)).fetchone()
            if not run:
                raise AppError(404, "RUN_NOT_FOUND", "检测运行不存在")
            version = repositories.dataset(db, run["dataset_id"])
            clauses = ["s.run_id=?"]
            args = [run_id]
            if role == "evaluation":
                clauses.append("s.included_in_ranking=1")
            elif role != "all":
                clauses.append("s.sample_role=?")
                args.append(role)
            if instance_uuid:
                clauses.append("i.instance_uuid=?")
                args.append(str(instance_uuid))
            if alert is not None:
                clauses.append("s.is_alert=?")
                args.append(int(alert))
            sql = " FROM instance_scores s JOIN instances i ON i.id=s.instance_id WHERE " + " AND ".join(clauses)
            total = db.execute("SELECT COUNT(*)" + sql, args).fetchone()[0]
            rows = [dict(r) for r in db.execute("""SELECT s.run_id,i.instance_uuid,s.score raw_score,s.is_alert,
                    s.sample_role,s.included_in_ranking,s.rank_best,s.rank_worst,i.label instance_label_metadata""" +
                    sql + " ORDER BY s.score DESC,i.instance_uuid LIMIT ? OFFSET ?", [*args, page_size, (page-1)*page_size])]
        return envelope(request, {"items": rows, "threshold": run["threshold_snapshot"], "origin": "experiment",
                                  "evaluation_scope": json.loads(run["scope_json"]),
                                  "pagination": {"total": total, "page": page, "page_size": page_size}}, version)

    @router.get("/instances/{uuid}")
    def instance_detail(request: Request, uuid: UUID, dataset_id: int = Query(gt=0)):
        with connect(settings.database) as db:
            version = repositories.dataset(db, dataset_id)
            instance = instance_row(db, dataset_id, uuid)
            feature = feature_row(db, dataset_id, instance["id"])
            value = dict(instance_uuid=str(uuid), instance_label_metadata=instance["label"], log_count=instance["log_count"],
                         first_event=instance["first_event"], last_event=instance["last_event"],
                         **json.loads(feature["metadata_json"]))
            value["features"] = json.loads(feature["features_json"])
            value["feature_set_id"] = feature["feature_set_id"]
        return envelope(request, value, version)

    @router.get("/instances/{uuid}/features")
    def instance_features(request: Request, uuid: UUID, dataset_id: int = Query(gt=0), feature_set_id: int | None = Query(None, gt=0)):
        with connect(settings.database) as db:
            version = repositories.dataset(db, dataset_id)
            instance = instance_row(db, dataset_id, uuid)
            feature = feature_row(db, dataset_id, instance["id"], feature_set_id)
        return envelope(request, {"feature_set_id": feature["feature_set_id"], "values": json.loads(feature["features_json"]),
                                  "phase_evidence": json.loads(feature["phase_evidence_json"])}, version)

    @router.get("/instances/{uuid}/timeline")
    def timeline(request: Request, uuid: UUID, dataset_id: int = Query(gt=0),
                 page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100)):
        with connect(settings.database) as db:
            version = repositories.dataset(db, dataset_id)
            instance = instance_row(db, dataset_id, uuid)
            feature = feature_row(db, dataset_id, instance["id"])
            offset = (page-1)*page_size
            # 带回上一行以计算跨页间隔，最多读取101条日志。
            rows = db.execute("""SELECT l.id log_id,l.event_time,l.event_time_raw,l.message,l.component,l.severity,
                         l.line_number,s.name source_file,t.external_template_id FROM logs l
                         JOIN log_sources s ON s.id=l.source_id JOIN templates t ON t.id=l.template_id
                         WHERE l.dataset_id=? AND l.instance_id=? ORDER BY l.event_time,l.input_order LIMIT ? OFFSET ?""",
                         (dataset_id, instance["id"], page_size + int(offset > 0), max(0, offset-1))).fetchall()
            previous = datetime.fromisoformat(rows[0]["event_time"]) if offset and rows else None
            events = []
            for index, row in enumerate(rows[int(offset > 0):]):
                event = dict(row)
                current = datetime.fromisoformat(row["event_time"])
                event["delta_seconds"] = (current-previous).total_seconds() if previous else None
                event["elapsed_seconds"] = (current-datetime.fromisoformat(instance["first_event"])).total_seconds()
                event["sequence_index"] = offset+index+1
                previous = current
                events.append(event)
        return envelope(request, {"items": events, "phases": json.loads(feature["phase_evidence_json"]),
                                  "pagination": {"total": instance["log_count"], "page": page, "page_size": page_size}}, version)

    @router.post("/inferences")
    def inference(request: Request, body: InferenceRequest):
        with connect(settings.database) as db:
            version = repositories.dataset(db, body.dataset_id)
            instance = instance_row(db, body.dataset_id, body.instance_uuid)
            feature = feature_row(db, body.dataset_id, instance["id"], body.feature_set_id)
            record = db.execute("SELECT * FROM model_registry WHERE model_key=?", (body.model_key,)).fetchone()
            if not record:
                raise AppError(404, "MODEL_NOT_FOUND", "模型未登记")
            result = models.infer(dict(record), feature)
        result.update(instance_uuid=str(body.instance_uuid), feature_set_id=body.feature_set_id,
                      instance_label_metadata=instance["label"])
        return envelope(request, result, version)

    return router
