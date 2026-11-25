"""所有查询均通过数据库，查询值绑定参数。"""

import base64
import hashlib
import json
from datetime import datetime, timedelta

from src.web.database import has_fts
from src.web.errors import AppError


JOINS = """ FROM logs l JOIN log_sources s ON s.id=l.source_id
 LEFT JOIN instances i ON i.id=l.instance_id JOIN templates t ON t.id=l.template_id """
FIELDS = """l.id,s.name source_file,l.line_number,l.event_time,l.component,l.severity,
 l.module,substr(l.message,1,240) message_preview,i.instance_uuid,
 t.external_template_id"""


def filter_source(query):
    return JOINS if query.instance_uuid or query.template_id else " FROM logs l "


def dataset(db, dataset_id):
    row = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if not row:
        raise AppError(404, "DATASET_NOT_FOUND", "数据集不存在")
    if row["state"] != "ready":
        raise AppError(409, "DATASET_NOT_READY", "数据集尚未完成导入")
    return dict(row)


def conditions(db, query):
    clauses = ["l.dataset_id=?"]
    values = [query.dataset_id]
    for field, column in (("severity", "l.severity"), ("component", "l.component"),
                          ("module", "l.module"), ("source_id", "l.source_id"),
                          ("instance_uuid", "i.instance_uuid"), ("template_id", "t.external_template_id")):
        value = getattr(query, field)
        if value is not None:
            clauses.append(column + "=?")
            values.append(str(value) if field == "instance_uuid" else value)
    for field, operator in (("start_time", ">="), ("end_time", "<")):
        value = getattr(query, field)
        if value:
            clauses.append("l.event_time" + operator + "?")
            values.append(value)
    if query.q:
        if has_fts(db) and len(query.q) >= 3:
            clauses.append("l.id IN (SELECT rowid FROM logs_fts WHERE logs_fts MATCH ?)")
            values.append('"' + query.q.replace('"', '""') + '"')
        clauses.append("(instr(l.message,?)>0 OR instr(l.raw_text,?)>0)")
        values.extend([query.q, query.q])
    return " WHERE " + " AND ".join(clauses), values


def query_signature(query):
    fields = query.model_dump(mode="json", exclude={"page", "cursor", "include_total", "bucket"})
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def logs(db, query):
    where, args = conditions(db, query)
    total = db.execute("SELECT COUNT(*)" + filter_source(query) + where, args).fetchone()[0] if query.include_total else None
    page = query.page or 1
    offset = (page - 1) * query.page_size
    if query.cursor:
        try:
            cursor = json.loads(base64.urlsafe_b64decode(query.cursor).decode())
            if cursor["signature"] != query_signature(query) or type(cursor["id"]) is not int:
                raise ValueError()
            where += " AND (l.event_time,l.id)<(?,?)"
            args += [cursor["time"], cursor["id"]]
            offset = 0
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            raise AppError(422, "INVALID_CURSOR", "游标无效或查询条件已变化") from error
    result = [dict(r) for r in db.execute("SELECT " + FIELDS + JOINS + where + " ORDER BY l.event_time DESC,l.id DESC LIMIT ? OFFSET ?",
                                        [*args, query.page_size + 1, offset])]
    more = len(result) > query.page_size
    result = result[:query.page_size]
    next_cursor = None
    if more and result:
        last = result[-1]
        next_cursor = base64.urlsafe_b64encode(json.dumps({"signature": query_signature(query), "time": last["event_time"], "id": last["id"]}).encode()).decode()
    return {"items": result, "pagination": {"total": total, "page": None if query.cursor else page,
            "page_size": query.page_size, "next_cursor": next_cursor}}


def detail(db, dataset_id, log_id):
    row = db.execute("SELECT " + FIELDS + """,l.message,l.raw_text,l.event_time_raw,l.process_id,
                     t.normalized_message,t.parser_version,i.label instance_label_metadata""" +
                     JOINS + " WHERE l.dataset_id=? AND l.id=?", (dataset_id, log_id)).fetchone()
    if not row:
        raise AppError(404, "LOG_NOT_FOUND", "日志不存在")
    return dict(row)


def overview(db, query):
    where, args = conditions(db, query)
    source = filter_source(query)
    result = {}
    for field, expression in (("log_count", "COUNT(*)"), ("linked_log_count", "COUNT(l.instance_id)"),
                              ("instance_count", "COUNT(DISTINCT l.instance_id)"), ("template_count", "COUNT(DISTINCT l.template_id)")):
        result[field] = db.execute("SELECT " + expression + source + where, args).fetchone()[0]
    result["unlinked_log_count"] = result["log_count"] - result["linked_log_count"]
    source_names = dict(db.execute("SELECT id,name FROM log_sources WHERE dataset_id=?", (query.dataset_id,)))
    for field, column in (("severity", "l.severity"), ("component", "l.component"), ("source", "l.source_id")):
        result[field + "_counts"] = {
            source_names[row[0]] if field == "source" else row[0]: row[1]
            for row in db.execute("SELECT " + column + ",COUNT(*)" + source + where + " GROUP BY " + column, args)}
    result["global_instance_label_counts"] = {r[0]: r[1] for r in db.execute("SELECT label,COUNT(*) FROM instances WHERE dataset_id=? GROUP BY label", (query.dataset_id,))}
    result["applied_filters"] = query.model_dump(mode="json", exclude={"page", "page_size", "cursor", "include_total", "bucket"})
    return result


def trend(db, query):
    where, args = conditions(db, query)
    bounds = db.execute("SELECT MIN(l.event_time),MAX(l.event_time)" + filter_source(query) + where, args).fetchone()
    if not bounds[0] and not (query.start_time and query.end_time):
        return []
    start = datetime.fromisoformat(query.start_time or bounds[0])
    end = datetime.fromisoformat(query.end_time or bounds[1])
    step = timedelta(hours=1) if query.bucket == "hour" else timedelta(days=1)
    current = start.replace(minute=0, second=0, microsecond=0)
    if query.bucket == "day":
        current = current.replace(hour=0)
    count = int((end - current) // step) + 1
    if count > 501:
        raise AppError(422, "TOO_MANY_BUCKETS", "时间桶超过500，请缩小时间范围或使用按天聚合")
    prefix = 13 if query.bucket == "hour" else 10
    grouped = dict(db.execute(f"SELECT substr(l.event_time,1,{prefix}),COUNT(*)" + filter_source(query) + where + f" GROUP BY substr(l.event_time,1,{prefix})", args))
    items = []
    while current <= end:
        if query.end_time and current >= end:
            break
        following = current + step
        items.append({"bucket_start": current.isoformat(), "bucket_end": following.isoformat(),
                      "count": grouped.get(current.isoformat()[:prefix], 0)})
        current = following
    if len(items) > 500:
        raise AppError(422, "TOO_MANY_BUCKETS", "时间桶超过500")
    return items
