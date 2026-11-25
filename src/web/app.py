"""启动 .venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import logging
import sqlite3
import time
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.web.config import Settings
from src.web.database import connect, initialize, has_fts
from src.web.errors import AppError
from src.web.importer import create_job, perform_import, now
from src.web.schemas import ImportRequest, LogQuery
from src.web.services import QueryService
from src.web.analysis_store import migrate
from src.web.analysis_api import build_router
from src.web.model_service import ModelService


LOGGER = logging.getLogger(__name__)


def create_app(settings=None):
    settings = settings or Settings()
    service = QueryService(settings)
    models = ModelService()

    @asynccontextmanager
    async def lifespan(application):
        initialize(settings.database)
        migrate(settings.database)
        with connect(settings.database) as db, db:
            db.execute("UPDATE import_jobs SET state='interrupted',finished_at=? WHERE state IN ('queued','running')", (now(),))
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="csv-import")
        application.state.executor = executor
        yield
        executor.shutdown(wait=True)

    application = FastAPI(title="云平台运维日志智能分析系统", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def request_metadata(request, call_next):
        request.state.request_id = str(uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter()-start)*1000:.3f}"
        LOGGER.info("%s %s %s %.3fms", request.method, request.url.path, response.status_code, (time.perf_counter()-start)*1000)
        return response

    def envelope(request, value, version=None):
        meta = {"request_id": request.state.request_id, "event_time_basis": "source_naive", "source_timezone": None}
        if version:
            meta.update(dataset_id=version["id"], data_version=version["version"])
        return {"data": value, "meta": meta}

    @application.exception_handler(AppError)
    async def app_error(request, error):
        return JSONResponse(status_code=error.status, content={"error": {"code": error.code, "message": error.message, "details": error.details},
                                                             "meta": {"request_id": request.state.request_id}})

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        details = [{"field": list(e["loc"]), "message": e["msg"]} for e in error.errors()]
        return await app_error(request, AppError(422, "VALIDATION_ERROR", "请求参数无效", details))

    @application.exception_handler(HTTPException)
    async def http_error(request, error):
        return await app_error(request, AppError(error.status_code, "HTTP_ERROR", str(error.detail)))

    @application.exception_handler(sqlite3.Error)
    async def database_error(request, error):
        LOGGER.exception("数据库错误 request=%s", request.state.request_id)
        return await app_error(request, AppError(503, "DATABASE_UNAVAILABLE", "数据库暂不可用"))

    @application.exception_handler(Exception)
    async def unexpected_error(request, error):
        LOGGER.exception("请求处理失败 request=%s", request.state.request_id)
        return await app_error(request, AppError(500, "INTERNAL_ERROR", "请求处理失败，请通过 request_id 查询运行日志"))

    @application.get("/api/v1/health")
    def health(request: Request):
        with connect(settings.database) as db:
            db.execute("SELECT 1").fetchone()
            fts = has_fts(db)
            model_states = [models.describe(dict(row)) for row in db.execute("SELECT * FROM model_registry ORDER BY id")]
        return envelope(request, {"status": "ok", "database": {"state": "available", "schema_version": 2, "fts5": fts},
                                  "models": model_states, "default_detection_method": "duration_p99"})

    @application.get("/api/v1/datasets")
    def datasets(request: Request, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100)):
        with connect(settings.database) as db:
            rows = [dict(r) for r in db.execute("SELECT id,name,version,state,source_timezone,created_at,ready_at,overview_json FROM datasets WHERE state='ready' ORDER BY id LIMIT ? OFFSET ?", (page_size, (page-1)*page_size))]
            total = db.execute("SELECT COUNT(*) FROM datasets WHERE state='ready'").fetchone()[0]
        for row in rows:
            row.update(json.loads(row.pop("overview_json") or "{}"))
        return envelope(request, {"items": rows, "pagination": {"total": total, "page": page, "page_size": page_size}})

    @application.get("/api/v1/datasets/{dataset_id}")
    def dataset_detail(request: Request, dataset_id: int):
        from src.web.repositories import dataset
        with connect(settings.database) as db:
            row = dataset(db, dataset_id)
        row.update(json.loads(row.pop("overview_json") or "{}"))
        row.pop("manifest_json")
        return envelope(request, row)

    @application.get("/api/v1/input-manifests")
    def manifests(request: Request):
        return envelope(request, [{"key": settings.manifest_key, "available": all(p.is_file() for p in settings.manifest().values())}])

    @application.post("/api/v1/imports")
    def submit_import(request: Request, body: ImportRequest, idempotency_key: Annotated[str, Header(min_length=1, max_length=200)]):
        job_id, created = create_job(settings, idempotency_key, body.input_manifest_key, body.dataset_name)
        with connect(settings.database) as db:
            job = dict(db.execute("SELECT id,state,dataset_id,progress_rows,expected_rows,created_at,started_at,finished_at FROM import_jobs WHERE id=?", (job_id,)).fetchone())
        job["status_url"] = f"/api/v1/imports/{job_id}"
        if created:
            application.state.executor.submit(perform_import, settings, job_id)
        return JSONResponse(status_code=202 if created else 200, content=envelope(request, job))

    @application.get("/api/v1/imports/{import_id}")
    def import_status(request: Request, import_id: int):
        with connect(settings.database) as db:
            row = db.execute("SELECT * FROM import_jobs WHERE id=?", (import_id,)).fetchone()
        if not row:
            raise AppError(404, "IMPORT_NOT_FOUND", "导入任务不存在")
        result = dict(row)
        error_json = result.pop("error_json")
        result["error"] = json.loads(error_json) if error_json else None
        result.pop("request_json")
        return envelope(request, result)

    @application.get("/api/v1/logs")
    def logs(request: Request, query: Annotated[LogQuery, Query()]):
        value, version = service.query("logs", query.dataset_id, query)
        return envelope(request, value, version)

    @application.get("/api/v1/logs/{log_id}")
    def log_detail(request: Request, log_id: int, dataset_id: int = Query(gt=0)):
        value, version = service.query("detail", dataset_id, dataset_id, log_id)
        return envelope(request, value, version)

    @application.get("/api/v1/overview")
    def overview(request: Request, query: Annotated[LogQuery, Query()]):
        value, version = service.query("overview", query.dataset_id, query)
        return envelope(request, value, version)

    @application.get("/api/v1/overview/trend")
    def trend(request: Request, query: Annotated[LogQuery, Query()]):
        value, version = service.query("trend", query.dataset_id, query)
        return envelope(request, value, version)

    @application.get("/api/v1/log-options")
    def options(request: Request, dataset_id: int = Query(gt=0)):
        from src.web.repositories import dataset
        with connect(settings.database) as db:
            version = dataset(db, dataset_id)
            result = {}
            for field in ("component", "severity", "module"):
                result[field] = {r[0]: r[1] for r in db.execute("SELECT " + field + ",COUNT(*) FROM logs WHERE dataset_id=? GROUP BY " + field, (dataset_id,))}
            result["sources"] = [dict(r) for r in db.execute("SELECT id,name,row_count FROM log_sources WHERE dataset_id=?", (dataset_id,))]
        return envelope(request, result, version)

    application.include_router(build_router(settings, envelope, models))
    return application


app = create_app()
