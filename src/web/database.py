"""连接按调用创建，事务使用 WAL，迁移可重复执行。"""

from contextlib import contextmanager
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS datasets(
 id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
 manifest_sha256 TEXT NOT NULL, manifest_json TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('importing','ready','failed')),
 source_timezone TEXT, overview_json TEXT,
 created_at TEXT NOT NULL, ready_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ready_manifest ON datasets(manifest_sha256) WHERE state='ready';
CREATE TABLE IF NOT EXISTS import_jobs(
 id INTEGER PRIMARY KEY, dataset_id INTEGER REFERENCES datasets(id),
 state TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL,
 request_json TEXT NOT NULL, progress_rows INTEGER NOT NULL DEFAULT 0,
 expected_rows INTEGER, error_json TEXT, created_at TEXT NOT NULL,
 started_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS log_sources(
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
 name TEXT NOT NULL, row_count INTEGER DEFAULT 0, linked_instance_count INTEGER DEFAULT 0,
 unparsed_count INTEGER DEFAULT 0, first_event TEXT, last_event TEXT,
 UNIQUE(dataset_id,name), UNIQUE(dataset_id,id));
CREATE TABLE IF NOT EXISTS instances(
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
 instance_uuid TEXT NOT NULL, label TEXT NOT NULL CHECK(label IN ('unknown','anomalous','normal')),
 label_evidence_json TEXT NOT NULL, log_count INTEGER DEFAULT 0,
 first_event TEXT, last_event TEXT, UNIQUE(dataset_id,instance_uuid), UNIQUE(dataset_id,id));
CREATE TABLE IF NOT EXISTS templates(
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
 parser_version TEXT NOT NULL, external_template_id TEXT NOT NULL,
 normalized_message TEXT NOT NULL, frequency INTEGER NOT NULL,
 component_counts_json TEXT NOT NULL, severity_counts_json TEXT NOT NULL,
 UNIQUE(dataset_id,parser_version,external_template_id), UNIQUE(dataset_id,id));
CREATE TABLE IF NOT EXISTS logs(
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
 source_id INTEGER NOT NULL, line_number INTEGER NOT NULL CHECK(line_number>0),
 input_order INTEGER NOT NULL, component TEXT NOT NULL, severity TEXT NOT NULL,
 module TEXT NOT NULL, event_time_raw TEXT NOT NULL, event_time TEXT NOT NULL,
 process_id INTEGER NOT NULL, message TEXT NOT NULL, raw_text TEXT NOT NULL,
 instance_id INTEGER, template_id INTEGER NOT NULL,
 UNIQUE(dataset_id,source_id,line_number), UNIQUE(dataset_id,input_order),
 FOREIGN KEY(dataset_id,source_id) REFERENCES log_sources(dataset_id,id),
 FOREIGN KEY(dataset_id,instance_id) REFERENCES instances(dataset_id,id),
 FOREIGN KEY(dataset_id,template_id) REFERENCES templates(dataset_id,id));
CREATE INDEX IF NOT EXISTS idx_logs_time ON logs(dataset_id,event_time DESC,id DESC);
CREATE INDEX IF NOT EXISTS idx_logs_severity_time ON logs(dataset_id,severity,event_time DESC,id DESC);
CREATE INDEX IF NOT EXISTS idx_logs_component_time ON logs(dataset_id,component,event_time DESC,id DESC);
CREATE INDEX IF NOT EXISTS idx_logs_source_time ON logs(dataset_id,source_id,event_time DESC,id DESC);
CREATE INDEX IF NOT EXISTS idx_logs_instance_time ON logs(dataset_id,instance_id,event_time,input_order);
CREATE INDEX IF NOT EXISTS idx_logs_template_time ON logs(dataset_id,template_id,event_time DESC,id DESC);
INSERT OR IGNORE INTO schema_migrations(version) VALUES(1);
"""


@contextmanager
def connect(path):
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        yield connection
    finally:
        connection.close()


def initialize(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        try:
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(message,raw_text,content='logs',content_rowid='id',tokenize='trigram case_sensitive 1')")
        except sqlite3.OperationalError as error:
            if "no such" not in str(error).lower() and "tokenizer" not in str(error).lower():
                raise
        db.commit()


def has_fts(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name='logs_fts'").fetchone() is not None
