"""End-to-end scenarios (spec section 10) driven through the real engine, real git worktrees and real
subprocesses. The agent CLIs are replaced by ``fake_agent.py`` which emits the exact output formats of
``claude -p --output-format stream-json`` and ``codex exec --json``."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import CALC, ROOT, git, plan
from orchestrator.engine import Orchestrator
from orchestrator.models import TaskStatus
from orchestrator.procutil import pid_alive

BACKEND = "def mul(a, b):\n    return a * b\n"
BACKEND_TEST = "from backend import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n"
FRONTEND = "<html><body>calc</body></html>\n"


def by_task(events, purpose="work"):
    out = {}
    for e in events:
        if e["purpose"] == purpose:
            out.setdefault(e["task"], {}).setdefault(e["event"], []).append(e)
    return out


def task_status(orch, run_id):
    return {t["id"]: t["status"] for t in orch.store.list_tasks(run_id)}


# --- 1, 2, 4, 9: parallel claude + codex in isolated worktrees, integration, main untouched -------------
def test_parallel_isolated_worktrees_integration_and_main_untouched(h, repo):
    h.scenario({
        "plan": plan(
            {"id": "t1-backend", "provider": "codex", "files": ["backend.py", "tests/test_backend.py"]},
            {"id": "t2-frontend", "provider": "claude", "files": ["web/**"]},
        ),
        "tasks": {
            "t1-backend": {"sleep": 1.5, "write": {"backend.py": BACKEND, "tests/test_backend.py": BACKEND_TEST}},
            "t2-frontend": {"sleep": 1.5, "write": {"web/index.html": FRONTEND}},
        },
    })
    main_before = git(repo, "rev-parse", "main")
    status_before = git(repo, "status", "--porcelain")
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED", orch.store.list_events(run_id)

    ev = by_task(h.events())
    b, f = ev["t1-backend"], ev["t2-frontend"]
    # (1) both providers ran at the same time
    assert b["start"][0]["provider"] == "codex" and f["start"][0]["provider"] == "claude"
    assert b["start"][0]["t"] < f["end"][0]["t"] and f["start"][0]["t"] < b["end"][0]["t"], "not parallel"
    # ... in different worktrees
    assert b["start"][0]["cwd"] != f["start"][0]["cwd"]
    assert repo.resolve() not in Path(b["start"][0]["cwd"]).parents
    # (2) neither saw the other's files although they wrote concurrently
    assert "web/index.html" not in b["end"][0]["files"]
    assert "backend.py" not in f["end"][0]["files"]
    # (4) both commits are on the integration branch
    run = orch.store.get_run(run_id)
    integ = run["integration_branch"]
    files = git(repo, "ls-tree", "-r", "--name-only", integ).splitlines()
    assert {"backend.py", "web/index.html", "tests/test_backend.py"} <= set(files)
    log = git(repo, "log", "--format=%B", f"main..{integ}")
    assert f"Orchestrator-Task: {run_id}/t1-backend" in log and f"Orchestrator-Task: {run_id}/t2-frontend" in log
    # (9) main and the user's checkout are untouched
    assert git(repo, "rev-parse", "main") == main_before
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(repo, "status", "--porcelain") == status_before
    assert not (repo / "backend.py").exists()
    # worktrees cleaned up, agent branches kept
    assert git(repo, "worktree", "list").count("\n") == 0
    assert git(repo, "branch", "--list", f"agent/{run_id}/*")
    # reviews: both reviewers reviewed the same integration commit
    reviews = orch.store.list_reviews(run_id)
    assert {r["reviewer"] for r in reviews} == {"claude", "codex"}
    assert len({r["target_commit"] for r in reviews}) == 1
    assert reviews[0]["target_commit"] == git(repo, "rev-parse", integ)
    # (10) report exists and is explicit
    report = (Path(orch.data_dir) / "runs" / run_id / "report.md").read_text()
    assert "SUCCEEDED" in report and "t1-backend" in report and "Nothing was pushed" in report


# --- 3: dependencies -------------------------------------------------------------------------------
def test_dependent_task_waits_and_builds_on_dependency(h, repo):
    h.scenario({
        "plan": plan(
            {"id": "t1-models", "provider": "codex", "files": ["backend.py"]},
            {"id": "t2-other", "provider": "claude", "files": ["web/**"]},
            {"id": "t3-api", "provider": "claude", "files": ["api.py"], "depends_on": ["t1-models"]},
        ),
        "tasks": {
            "t1-models": {"sleep": 1.0, "write": {"backend.py": BACKEND}},
            "t2-other": {"sleep": 0.2, "write": {"web/a.html": FRONTEND}},
            "t3-api": {"write": {"api.py": "from backend import mul\n"}},
        },
    })
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED"
    ev = by_task(h.events())
    assert ev["t3-api"]["start"][0]["t"] >= ev["t1-models"]["end"][0]["t"]
    # t3's worktree was created from the integration head that already contained t1's work
    assert "backend.py" in ev["t3-api"]["end"][0]["files"]


def test_failed_dependency_blocks_dependants(h, repo):
    h.scenario({
        "plan": plan(
            {"id": "t1", "provider": "codex", "files": ["a.py"]},
            {"id": "t2", "provider": "claude", "files": ["b.py"], "depends_on": ["t1"]},
        ),
        "tasks": {"t1": {"exit_code": 1}, "t2": {"write": {"b.py": "x = 1\n"}}},
    })
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    st = task_status(orch, run_id)
    assert st == {"t1": "FAILED", "t2": "BLOCKED"}
    assert status == "NEEDS_ATTENTION"
    assert orch.store.get_task(run_id, "t1")["attempts"] == 2      # 1 retry
    assert "t2" not in by_task(h.events())                           # never started


# --- 5: conflicts ------------------------------------------------------------------------------------
def test_conflict_aborts_safely_and_is_reported(h, repo):
    h.scenario({
        "plan": plan(
            {"id": "t1", "provider": "codex", "files": ["calc.py"]},
            {"id": "t2", "provider": "claude", "files": ["calc.py"]},
        ),
        "tasks": {
            "t1": {"sleep": 0.2, "write": {"calc.py": "def add(a, b):\n    return b + a  # codex\n"}},
            "t2": {"sleep": 1.0, "write": {"calc.py": "def add(a, b):\n    return a + b  # claude\n"}},
        },
    })
    orch = h.orchestrator(limits={"max_repair_rounds": 0})
    run_id, status = h.run(orch, repo)
    st = task_status(orch, run_id)
    assert sorted(st.values()) == ["COMPLETED", "CONFLICT"]
    assert status == "NEEDS_ATTENTION"
    conflicted = next(t for t in orch.store.list_tasks(run_id) if t["status"] == "CONFLICT")
    assert "calc.py" in conflicted["error"]
    run = orch.store.get_run(run_id)
    integ_files = git(repo, "show", f"{run['integration_branch']}:calc.py")
    assert "# codex" in integ_files and "<<<<<<<" not in integ_files     # nothing overwritten / no markers
    assert conflicted["branch"] and git(repo, "branch", "--list", conflicted["branch"])  # kept for inspection
    report = orch.store.get_run(run_id)
    assert "conflicting" in report["error"]


def test_conflict_is_reassigned_as_repair_and_resolved(h, repo):
    h.scenario({
        "plan": plan(
            {"id": "t1", "provider": "codex", "files": ["calc.py"]},
            {"id": "t2", "provider": "claude", "files": ["calc.py"]},
        ),
        "tasks": {
            "t1": {"sleep": 0.2, "write": {"calc.py": "def add(a, b):\n    return b + a  # codex\n"}},
            # 1st attempt conflicts; the re-run (on top of the integration head) keeps codex's change
            "t2": [{"sleep": 1.0, "write": {"calc.py": "def add(a, b):\n    return a + b  # claude\n"}},
                   {"write": {"calc.py": "def add(a, b):\n    return b + a  # codex\n\n\ndef sub(a, b):\n"
                                         "    return a - b\n"}}],
        },
    })
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED", orch.store.list_events(run_id)
    assert task_status(orch, run_id) == {"t1": "COMPLETED", "t2": "COMPLETED"}
    assert orch.store.get_run(run_id)["repair_round"] == 1


# --- 6 + 7: test failure -> repair task; bounded rounds ----------------------------------------------------
def test_failing_tests_create_repair_task_that_fixes_them(h, repo):
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["calc.py"]}),
        "tasks": {"t1": {"write": {"calc.py": "def add(a, b):\n    return a - b\n"}}},   # bug
        "repair": {"write": {"calc.py": CALC}},
    })
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED"
    tasks = {t["id"]: t for t in orch.store.list_tasks(run_id)}
    assert tasks["repair-r1"]["kind"] == "repair" and tasks["repair-r1"]["status"] == "COMPLETED"
    # the repair went to the author of the broken file (codex)
    assert tasks["repair-r1"]["provider"] == "codex"
    tests = orch.store.list_test_results(run_id)
    assert [t["passed"] for t in tests if t["scope"] == "integration"] == [0, 1]
    # the repair prompt contained the failing test output
    rep = [e for e in h.events() if e["purpose"] == "repair"]
    assert rep


def test_repair_rounds_are_bounded(h, repo):
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["calc.py"]}),
        "tasks": {"t1": {"write": {"calc.py": "def add(a, b):\n    return a - b\n"}}},
        "repair": {"write": {"calc.py": "def add(a, b):\n    return a * b  # still wrong\n"}},
    })
    orch = h.orchestrator(limits={"max_repair_rounds": 2})
    t0 = time.time()
    run_id, status = h.run(orch, repo)
    assert time.time() - t0 < 60
    assert status == "NEEDS_ATTENTION"
    ids = [t["id"] for t in orch.store.list_tasks(run_id)]
    assert ids == ["t1", "repair-r1", "repair-r2"]
    run = orch.store.get_run(run_id)
    assert run["repair_round"] == 2 and "max repair rounds (2) reached" in run["error"]
    assert [t["passed"] for t in orch.store.list_test_results(run_id)] == [0, 0, 0]


def test_blocking_review_finding_triggers_repair(h, repo):
    finding = {"severity": "high", "file": "backend.py", "line": 1, "category": "bug",
               "title": "mul ignores negative numbers", "recommendation": "handle negatives"}
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["backend.py"]}),
        "tasks": {"t1": {"write": {"backend.py": BACKEND}}},
        "reviews": {"claude": [{"verdict": "request_changes", "summary": "bug", "findings": [finding]},
                               {"verdict": "approve", "summary": "ok", "findings": []}]},
        "repair": {"write": {"backend.py": BACKEND + "# negatives handled\n"}},
    })
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED"
    reviews = orch.store.list_reviews(run_id)
    assert [(r["round"], r["reviewer"], r["verdict"]) for r in reviews] == [
        (0, "claude", "request_changes"), (0, "codex", "approve"),
        (1, "claude", "approve"), (1, "codex", "approve")] or \
        sorted((r["round"], r["reviewer"], r["verdict"]) for r in reviews) == [
        (0, "claude", "request_changes"), (0, "codex", "approve"),
        (1, "claude", "approve"), (1, "codex", "approve")]
    assert reviews[0]["findings"][0]["severity"] == "high"
    assert task_status(orch, run_id)["repair-r1"] == "COMPLETED"


# --- validation rules ---------------------------------------------------------------------------------
def test_out_of_scope_change_is_rejected_then_retried(h, repo):
    h.scenario({
        "plan": plan({"id": "t1", "provider": "claude", "files": ["web/**"]}),
        "tasks": {"t1": [{"write": {"web/a.html": FRONTEND, "calc.py": "hacked = True\n"}},
                         {"write": {"web/a.html": FRONTEND}}]},
    })
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED"
    t = orch.store.get_task(run_id, "t1")
    assert t["attempts"] == 2 and t["changed_files"] == ["web/a.html"]
    assert any("outside the allowed scope" in e["message"] for e in orch.store.list_events(run_id))
    run = orch.store.get_run(run_id)
    assert git(repo, "show", f"{run['integration_branch']}:calc.py") == CALC.strip()


def test_deleting_existing_files_and_forbidden_paths_are_rejected(h, repo):
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["**"]}),
        "tasks": {"t1": {"delete": ["README.md"], "write": {".env": "SECRET=1\n"}}},
    })
    orch = h.orchestrator(limits={"max_task_retries": 0, "max_repair_rounds": 0})
    run_id, status = h.run(orch, repo)
    t = orch.store.get_task(run_id, "t1")
    assert t["status"] == "FAILED"
    assert "deletion of existing files is not allowed" in t["error"] and "forbidden path" in t["error"]
    assert status == "NEEDS_ATTENTION"


def test_no_changes_is_a_failure(h, repo):
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["x.py"]}), "tasks": {"t1": {}}})
    orch = h.orchestrator(limits={"max_task_retries": 0, "max_repair_rounds": 0})
    run_id, _ = h.run(orch, repo)
    assert "no file changes" in orch.store.get_task(run_id, "t1")["error"]


def test_tests_not_configured_is_not_reported_as_success(h, repo):
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["x.py"]}),
                "tasks": {"t1": {"write": {"x.py": "x = 1\n"}}}})
    orch = h.orchestrator(validation={"test_command": None})
    _, status = h.run(orch, repo)
    assert status == "UNVERIFIED"


# --- timeouts / process trees -----------------------------------------------------------------------------
def test_timeout_kills_agent_process_tree(h, repo, tmp_path, monkeypatch):
    pidfile = tmp_path / "child.pid"
    monkeypatch.setenv("ORCH_FAKE_CHILD_PIDFILE", str(pidfile))
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["x.py"]}),
                "tasks": {"t1": {"spawn_child_sleep": 60, "sleep": 60}}})
    orch = h.orchestrator(limits={"task_timeout_seconds": 2, "max_task_retries": 0, "max_repair_rounds": 0})
    t0 = time.time()
    run_id, status = h.run(orch, repo)
    assert time.time() - t0 < 30
    t = orch.store.get_task(run_id, "t1")
    assert t["status"] == "FAILED" and "timed out" in t["error"]
    child = int(pidfile.read_text())
    deadline = time.time() + 5
    while pid_alive(child) and time.time() < deadline:
        time.sleep(0.1)
    assert not pid_alive(child), "grandchild process survived"
    ar = orch.store.list_agent_runs(run_id)
    assert any(a["timed_out"] for a in ar)


# --- stop / resume / crash ----------------------------------------------------------------------------------
def test_stop_terminates_agents_and_resume_completes(h, repo):
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]},
                     {"id": "t2", "provider": "claude", "files": ["b.py"]}),
        "tasks": {"t1": [{"sleep": 60}, {"write": {"a.py": "a = 1\n"}}],
                  "t2": [{"sleep": 60}, {"write": {"b.py": "b = 1\n"}}]},
    })
    orch = h.orchestrator()
    run_id = orch.create_run("x", repo)

    async def scenario():
        job = asyncio.create_task(orch.execute(run_id))
        for _ in range(100):
            await asyncio.sleep(0.1)
            if len([e for e in h.events() if e["event"] == "start" and e["purpose"] == "work"]) == 2:
                break
        # stop from "another terminal": a separate Orchestrator instance writes the request to SQLite
        other = Orchestrator(orch.config)
        other.request_stop(run_id)
        t0 = time.time()
        status = await asyncio.wait_for(job, timeout=30)
        return status, time.time() - t0

    status, took = asyncio.run(scenario())
    assert status == "STOPPED" and took < 15
    assert set(task_status(orch, run_id).values()) == {"CANCELLED"}
    for e in h.events():
        if e["event"] == "start":
            assert not pid_alive(e["pid"])
    status = asyncio.run(orch.resume(run_id))
    assert status == "SUCCEEDED"
    assert set(task_status(orch, run_id).values()) == {"COMPLETED"}


def test_cancel_single_task(h, repo):
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]},
                     {"id": "t2", "provider": "claude", "files": ["b.py"]}),
        "tasks": {"t1": {"sleep": 60}, "t2": {"sleep": 1, "write": {"b.py": "b = 1\n"}}},
    })
    orch = h.orchestrator(limits={"max_repair_rounds": 0})
    run_id = orch.create_run("x", repo)

    async def scenario():
        job = asyncio.create_task(orch.execute(run_id))
        for _ in range(100):
            await asyncio.sleep(0.1)
            if any(e["task"] == "t1" and e["event"] == "start" for e in h.events()):
                break
        orch.request_cancel(run_id, "t1")
        return await asyncio.wait_for(job, timeout=30)

    status = asyncio.run(scenario())
    assert task_status(orch, run_id) == {"t1": "CANCELLED", "t2": "COMPLETED"}
    assert status == "NEEDS_ATTENTION"


def test_hard_crash_keeps_records_and_resume_recovers(h, repo):
    """Kill -9 the orchestrator process mid-run, inspect state from a new process, then resume."""
    if os.name == "nt":
        pytest.skip("uses POSIX SIGKILL")
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]},
                     {"id": "t2", "provider": "claude", "files": ["b.py"], "depends_on": ["t1"]}),
        "tasks": {"t1": {"write": {"a.py": "a = 1\n"}},
                  "t2": [{"sleep": 120}, {"write": {"b.py": "b = 1\n"}}]},
    })
    cfg_path = h.tmp / "cfg.yaml"
    import yaml
    cfg = h.config()
    cfg_path.write_text(yaml.safe_dump(cfg))
    proc = subprocess.Popen([sys.executable, str(ROOT / "main.py"), "--config", str(cfg_path), "run", "crash test",
                             "--repo", str(repo), "--quiet"], cwd=str(ROOT), env=os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.time() + 60
    while time.time() < deadline:
        if any(e["task"] == "t2" and e["event"] == "start" for e in h.events()):
            break
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("t2 never started: " + proc.stdout.read().decode())
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    agent_pid = next(e["pid"] for e in h.events() if e["task"] == "t2" and e["event"] == "start")

    orch = Orchestrator(cfg)
    run_id = orch.store.latest_run_id()
    run = orch.store.get_run(run_id)
    assert run["status"] == "RUNNING" and not pid_alive(run["pid"])          # detectable as interrupted
    st = task_status(orch, run_id)
    assert st["t1"] == "COMPLETED" and st["t2"] == "RUNNING"
    from orchestrator.cli import main as cli_main
    assert cli_main(["--config", str(cfg_path), "status", run_id]) == 0

    status = asyncio.run(orch.resume(run_id))
    assert status == "SUCCEEDED"
    assert not pid_alive(agent_pid), "orphaned agent was not terminated on resume"
    assert task_status(orch, run_id) == {"t1": "COMPLETED", "t2": "COMPLETED"}
    starts = [e for e in h.events() if e["task"] == "t1" and e["event"] == "start"]
    assert len(starts) == 1, "completed task must not be re-run"


def test_plan_only_then_resume(h, repo):
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]}),
                "tasks": {"t1": {"write": {"a.py": "a = 1\n"}}}})
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo, plan_only=True)
    assert status == "PLANNED" and task_status(orch, run_id) == {"t1": "PENDING"}
    assert asyncio.run(orch.resume(run_id)) == "SUCCEEDED"


# --- providers -------------------------------------------------------------------------------------
def test_unavailable_codex_falls_back_and_review_is_flagged(h, repo):
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]}),
                "tasks": {"t1": {"write": {"a.py": "a = 1\n"}}}})
    cfg = h.config()
    cfg["providers"]["codex"]["command"] = ["definitely-not-installed-codex"]
    orch = Orchestrator(cfg)
    run_id, status = h.run(orch, repo)
    t = orch.store.get_task(run_id, "t1")
    assert t["provider"] == "claude" and t["status"] == "COMPLETED"
    reviews = orch.store.list_reviews(run_id)
    assert any(r["reviewer"] == "codex" and r["verdict"] == "error" for r in reviews)
    assert status == "NEEDS_ATTENTION"     # review.required=true and the codex review could not run


def test_invalid_plan_fails_cleanly(h, repo):
    h.scenario({"plan": plan({"id": "a", "provider": "codex", "files": ["a.py"], "depends_on": ["b"]},
                             {"id": "b", "provider": "codex", "files": ["b.py"], "depends_on": ["a"]})})
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "FAILED" and "cycle" in orch.store.get_run(run_id)["error"]


def test_secrets_are_redacted_from_logs(h, repo, monkeypatch):
    monkeypatch.setenv("MY_SERVICE_TOKEN", "supersecretvalue123456")
    h.scenario({"plan": plan({"id": "t1", "provider": "claude", "files": ["a.py"]}),
                "tasks": {"t1": {"write": {"a.py": "a = 1\n"},
                                 "print": "leak: supersecretvalue123456 and sk-ant-abcdefghijklmnop123"}}})
    orch = h.orchestrator()
    run_id, _ = h.run(orch, repo)
    logs = [Path(a["log_path"]).read_text() for a in orch.store.list_agent_runs(run_id) if a["log_path"]]
    joined = "\n".join(logs)
    assert "leak:" in joined
    assert "supersecretvalue123456" not in joined and "sk-ant-abcdefghijklmnop123" not in joined


def test_max_parallel_workers_is_respected(h, repo):
    tasks = [{"id": f"t{i}", "provider": "codex" if i % 2 else "claude", "files": [f"f{i}.py"]} for i in range(5)]
    h.scenario({"plan": plan(*tasks),
                "tasks": {t["id"]: {"sleep": 0.8, "write": {f"f{i}.py": "x = 1\n"}} for i, t in enumerate(tasks)}})
    orch = h.orchestrator(limits={"max_parallel_workers": 2})
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED"
    ev = sorted(((e["t"], 1 if e["event"] == "start" else -1) for e in h.events() if e["purpose"] == "work"))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    assert peak == 2


def test_resume_after_stop_on_last_attempt_does_not_loop(h, repo):
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]}),
                "tasks": {"t1": [{"sleep": 60}, {"write": {"a.py": "a = 1\n"}}]}})
    orch = h.orchestrator(limits={"max_task_retries": 0})
    run_id = orch.create_run("x", repo)

    async def scenario():
        job = asyncio.create_task(orch.execute(run_id))
        for _ in range(100):
            await asyncio.sleep(0.1)
            if any(e["event"] == "start" and e["purpose"] == "work" for e in h.events()):
                break
        orch.request_stop(run_id)
        return await asyncio.wait_for(job, timeout=30)

    assert asyncio.run(scenario()) == "STOPPED"
    assert asyncio.run(asyncio.wait_for(orch.resume(run_id), timeout=60)) == "SUCCEEDED"
    assert orch.store.get_task(run_id, "t1")["attempts"] == 1


def test_retry_failed_after_exhausted_repairs_creates_new_repair_tasks(h, repo):
    """Regression: resume --retry-failed restarts round counting; repair task ids must not collide."""
    h.scenario({
        "plan": plan({"id": "t1", "provider": "codex", "files": ["calc.py"]}),
        "tasks": {"t1": {"write": {"calc.py": "def add(a, b):\n    return a - b\n"}},
                  "repair-r1": {"write": {"calc.py": "def add(a, b):\n    return a * b\n"}},
                  "repair-r2": {"write": {"calc.py": "def add(a, b):\n    return a * 1 * b\n"}},
                  "repair-r1-2": {"write": {"calc.py": CALC}}},
    })
    orch = h.orchestrator(limits={"max_repair_rounds": 2})
    run_id, status = h.run(orch, repo)
    assert status == "NEEDS_ATTENTION"
    status = asyncio.run(orch.resume(run_id, retry_failed=True))
    assert status == "SUCCEEDED", orch.store.list_events(run_id)[-10:]
    ids = [t["id"] for t in orch.store.list_tasks(run_id)]
    assert ids == ["t1", "repair-r1", "repair-r2", "repair-r1-2"]
    assert len(set(ids)) == len(ids)
