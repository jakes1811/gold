"""Audit tables written by every job (never raises: a job must not fail because auditing failed).
Default backend = JSON lines in data/audit/<table>.jsonl (no database needed).
Set FINTEL_AUDIT_BACKEND=postgres (+ FINTEL_PG_* settings, psycopg2 installed) to write to Postgres instead.

Tables: bronze_ingestion_audit, silver_processing_audit, gold_aggregation_audit, pipeline_execution_audit.
"""
import json
import threading
import time
import uuid
from datetime import datetime, timezone

from config import settings

AUDIT_COLUMNS = {
    "bronze_ingestion_audit": ["run_id", "job_name", "topic", "batch_id", "record_count", "status",
                               "latency_ms", "error_count", "error_message"],
    "silver_processing_audit": ["run_id", "job_name", "batch_id", "input_count", "processed_count",
                                "rejected_count", "ticker_count", "duration_ms", "status", "error_message"],
    "gold_aggregation_audit": ["run_id", "job_name", "batch_id", "window_start_min", "window_end_max",
                               "input_news_count", "input_tick_count", "output_row_count", "late_records",
                               "duration_ms", "status", "error_message"],
    "pipeline_execution_audit": ["run_id", "job_name", "started_at", "ended_at", "status", "records_processed",
                                 "throughput_per_sec", "error_count", "last_error", "heartbeat_at"],
}
_lock = threading.Lock()
_runs = {}
_pg_down_until = 0.0


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pg():
    global _pg_down_until
    if settings.AUDIT_BACKEND != "postgres" or time.time() < _pg_down_until:
        return None
    try:
        import psycopg2
        return psycopg2.connect(host=settings.PG_HOST, port=settings.PG_PORT, dbname=settings.PG_DB,
                                user=settings.PG_USER, password=settings.PG_PASSWORD, connect_timeout=2)
    except Exception:
        _pg_down_until = time.time() + 60
        return None


def _jsonl(table):
    settings.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return settings.AUDIT_DIR / f"{table}.jsonl"


def write_audit(table: str, row: dict) -> None:
    try:
        unknown = set(row) - set(AUDIT_COLUMNS[table])
        if unknown:
            raise ValueError(f"{table}: unknown column(s) {sorted(unknown)}")
        row = dict(row)
        if table != "pipeline_execution_audit":
            row["recorded_at"] = _now()
        con = _pg()
        if con is not None:
            try:
                cols = [c for c in list(AUDIT_COLUMNS[table]) + ["recorded_at"] if c in row]
                with con, con.cursor() as cur:
                    if table == "pipeline_execution_audit":
                        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "run_id")
                        cur.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join(['%s']*len(cols))}) "
                                    f"ON CONFLICT (run_id) DO UPDATE SET {sets}", [row[c] for c in cols])
                    else:
                        cur.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join(['%s']*len(cols))})",
                                    [row[c] for c in cols])
                con.close()
                return
            except Exception:
                pass
        with _lock, open(_jsonl(table), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as error:                      # auditing must never break a job
        print(f"[AUDIT WARNING] {table}: {error}")


def job_start(run_id: str, job_name: str) -> None:
    _runs[run_id] = time.time()
    write_audit("pipeline_execution_audit", {"run_id": run_id, "job_name": job_name, "started_at": _now(),
                                              "status": "RUNNING", "records_processed": 0,
                                              "throughput_per_sec": 0.0, "error_count": 0, "heartbeat_at": _now()})


def job_heartbeat(run_id: str, job_name: str, records_processed: int, error_count: int = 0, last_error=None) -> None:
    dur = max(time.time() - _runs.setdefault(run_id, time.time()), 1e-6)
    write_audit("pipeline_execution_audit", {"run_id": run_id, "job_name": job_name, "status": "RUNNING",
                                              "records_processed": int(records_processed),
                                              "throughput_per_sec": round(records_processed / dur, 3),
                                              "error_count": int(error_count), "last_error": last_error,
                                              "heartbeat_at": _now()})


def job_end(run_id: str, job_name: str, status: str = "STOPPED", last_error=None) -> None:
    write_audit("pipeline_execution_audit", {"run_id": run_id, "job_name": job_name, "ended_at": _now(),
                                              "status": status, "last_error": last_error, "heartbeat_at": _now()})


class Heartbeat:
    """Background heartbeat for plain-Python jobs. `counter()` returns the records processed so far."""

    def __init__(self, job_name, counter, every=30):
        self.job_name, self.counter, self.every = job_name, counter, every
        self.run_id = new_run_id()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        job_start(self.run_id, self.job_name)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.every):
            job_heartbeat(self.run_id, self.job_name, self.counter())

    def stop(self, status="STOPPED", error=None):
        self._stop.set()
        job_heartbeat(self.run_id, self.job_name, self.counter())
        job_end(self.run_id, self.job_name, status=status, last_error=error)


def read_table(table: str) -> list:
    """All rows of an audit table. pipeline_execution_audit: latest merged state per run_id."""
    rows = []
    con = _pg()
    if con is not None:
        try:
            import psycopg2.extras
            with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"SELECT * FROM {table}")
                rows = [dict(r) for r in cur.fetchall()]
            con.close()
            return rows
        except Exception:
            pass
    p = _jsonl(table)
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    if table == "pipeline_execution_audit":
        merged = {}
        for r in rows:
            merged.setdefault(r["run_id"], {}).update({k: v for k, v in r.items() if v is not None})
        rows = list(merged.values())
    return rows
