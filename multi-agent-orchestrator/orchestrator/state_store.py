"""SQLite persistence. Every state transition is written immediately (WAL mode) so that
another process (``orchestrator status`` / a future web dashboard) can read it and a crashed
run can be resumed."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import TaskSpec, TaskStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    request TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    base_ref TEXT,
    base_commit TEXT,
    integration_branch TEXT,
    integration_worktree TEXT,
    status TEXT NOT NULL,
    plan_json TEXT,
    config_json TEXT,
    repair_round INTEGER NOT NULL DEFAULT 0,
    stop_requested INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    error TEXT,
    summary_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
    run_id TEXT NOT NULL,
    id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    provider TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'work',
    repair_round INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    branch TEXT,
    worktree TEXT,
    base_commit TEXT,
    commit_sha TEXT,
    integrated_sha TEXT,
    changed_files TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT,
    purpose TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    command_json TEXT,
    cwd TEXT,
    pid INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    exit_code INTEGER,
    timed_out INTEGER NOT NULL DEFAULT 0,
    cancelled INTEGER NOT NULL DEFAULT 0,
    log_path TEXT,
    cost_usd REAL,
    usage_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    reviewer TEXT NOT NULL,
    target_commit TEXT,
    verdict TEXT NOT NULL,
    summary TEXT,
    findings_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    round INTEGER NOT NULL DEFAULT 0,
    command_json TEXT,
    exit_code INTEGER,
    passed INTEGER NOT NULL,
    timed_out INTEGER NOT NULL DEFAULT 0,
    output_path TEXT,
    output_tail TEXT,
    duration_seconds REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_run ON agent_runs(run_id, id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, db_path: str | Path, redactor=None):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._redact = redactor or (lambda s: s)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- helpers
    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    # ------------------------------------------------------------------- runs
    def create_run(self, run_id: str, request: str, repo_path: str, config: dict, status: str) -> None:
        ts = now_iso()
        self._exec(
            "INSERT INTO runs (id, request, repo_path, status, config_json, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (run_id, request, repo_path, status, json.dumps(config), ts, ts),
        )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        for k in ("plan_json", "summary_json"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE runs SET {cols} WHERE id=?", (*fields.values(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        rows = self._query("SELECT * FROM runs WHERE id=?", (run_id,))
        return dict(rows[0]) if rows else None

    def latest_run_id(self) -> str | None:
        rows = self._query("SELECT id FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1")
        return rows[0]["id"] if rows else None

    def list_runs(self, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------ tasks
    def add_task(self, run_id: str, spec: TaskSpec, status: TaskStatus = TaskStatus.PENDING,
                 repair_round: int = 0) -> None:
        seq_rows = self._query("SELECT COALESCE(MAX(seq), 0) AS m FROM tasks WHERE run_id=?", (run_id,))
        seq = (seq_rows[0]["m"] or 0) + 1
        self._exec(
            "INSERT INTO tasks (run_id, id, seq, spec_json, provider, kind, repair_round, status, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, spec.id, seq, json.dumps(spec.to_dict(), ensure_ascii=False), spec.provider,
             spec.kind, repair_round, status.value, now_iso()),
        )

    def update_task(self, run_id: str, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        if "status" in fields and isinstance(fields["status"], TaskStatus):
            fields["status"] = fields["status"].value
        if "changed_files" in fields and not isinstance(fields["changed_files"], (str, type(None))):
            fields["changed_files"] = json.dumps(fields["changed_files"])
        if "spec" in fields:
            spec = fields.pop("spec")
            fields["spec_json"] = json.dumps(spec.to_dict(), ensure_ascii=False)
            fields["provider"] = spec.provider
        if "error" in fields and fields["error"]:
            fields["error"] = self._redact(str(fields["error"]))[:8000]
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE tasks SET {cols} WHERE run_id=? AND id=?", (*fields.values(), run_id, task_id))

    def get_task(self, run_id: str, task_id: str) -> dict | None:
        rows = self._query("SELECT * FROM tasks WHERE run_id=? AND id=?", (run_id, task_id))
        return self._task_row(rows[0]) if rows else None

    def list_tasks(self, run_id: str) -> list[dict]:
        return [self._task_row(r) for r in self._query(
            "SELECT * FROM tasks WHERE run_id=? ORDER BY seq", (run_id,))]

    @staticmethod
    def _task_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["spec"] = TaskSpec.from_dict(json.loads(d.pop("spec_json")))
        d["changed_files"] = json.loads(d["changed_files"]) if d.get("changed_files") else []
        return d

    # ------------------------------------------------------------- agent runs
    def start_agent_run(self, run_id: str, task_id: str | None, purpose: str, provider: str,
                        model: str | None, command: list[str], cwd: str, log_path: str) -> int:
        cmd = [self._redact(c) if len(c) < 2000 else self._redact(c[:200]) + "...<truncated>" for c in command]
        cur = self._exec(
            "INSERT INTO agent_runs (run_id, task_id, purpose, provider, model, command_json, cwd, "
            "started_at, log_path) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, task_id, purpose, provider, model, json.dumps(cmd), cwd, now_iso(), log_path),
        )
        return int(cur.lastrowid)

    def finish_agent_run(self, agent_run_id: int, **fields: Any) -> None:
        fields["finished_at"] = now_iso()
        if "usage_json" in fields and not isinstance(fields["usage_json"], (str, type(None))):
            fields["usage_json"] = json.dumps(fields["usage_json"])
        if fields.get("error"):
            fields["error"] = self._redact(str(fields["error"]))[:8000]
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE agent_runs SET {cols} WHERE id=?", (*fields.values(), agent_run_id))

    def set_agent_pid(self, agent_run_id: int, pid: int) -> None:
        self._exec("UPDATE agent_runs SET pid=? WHERE id=?", (pid, agent_run_id))

    def list_agent_runs(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._query("SELECT * FROM agent_runs WHERE run_id=? ORDER BY id", (run_id,))]

    def total_cost(self, run_id: str) -> float:
        rows = self._query("SELECT COALESCE(SUM(cost_usd), 0) AS c FROM agent_runs WHERE run_id=?", (run_id,))
        return float(rows[0]["c"] or 0.0)

    # ---------------------------------------------------------------- reviews
    def add_review(self, run_id: str, round_no: int, reviewer: str, target_commit: str, verdict: str,
                   summary: str, findings: list[dict], error: str | None = None) -> None:
        self._exec(
            "INSERT INTO reviews (run_id, round, reviewer, target_commit, verdict, summary, findings_json, "
            "error, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, round_no, reviewer, target_commit, verdict, self._redact(summary or ""),
             json.dumps(findings, ensure_ascii=False), self._redact(error) if error else None, now_iso()),
        )

    def list_reviews(self, run_id: str) -> list[dict]:
        out = []
        for r in self._query("SELECT * FROM reviews WHERE run_id=? ORDER BY id", (run_id,)):
            d = dict(r)
            d["findings"] = json.loads(d.pop("findings_json") or "[]")
            out.append(d)
        return out

    # ------------------------------------------------------------------ tests
    def add_test_result(self, run_id: str, scope: str, round_no: int, command: list[str] | None,
                        exit_code: int | None, passed: bool, timed_out: bool, output_path: str | None,
                        output_tail: str, duration: float) -> None:
        self._exec(
            "INSERT INTO test_results (run_id, scope, round, command_json, exit_code, passed, timed_out, "
            "output_path, output_tail, duration_seconds, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, scope, round_no, json.dumps(command), exit_code, int(passed), int(timed_out),
             output_path, self._redact(output_tail or "")[-6000:], duration, now_iso()),
        )

    def list_test_results(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._query("SELECT * FROM test_results WHERE run_id=? ORDER BY id", (run_id,))]

    # ----------------------------------------------------------------- events
    def log_event(self, run_id: str, message: str, level: str = "info", task_id: str | None = None) -> None:
        self._exec("INSERT INTO events (run_id, task_id, ts, level, message) VALUES (?,?,?,?,?)",
                   (run_id, task_id, now_iso(), level, self._redact(message)))

    def list_events(self, run_id: str, after_id: int = 0, task_id: str | None = None) -> list[dict]:
        if task_id:
            rows = self._query("SELECT * FROM events WHERE run_id=? AND id>? AND task_id=? ORDER BY id",
                               (run_id, after_id, task_id))
        else:
            rows = self._query("SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, after_id))
        return [dict(r) for r in rows]
