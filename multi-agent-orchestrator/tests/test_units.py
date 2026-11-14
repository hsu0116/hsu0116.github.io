from __future__ import annotations

import asyncio
import json
import sys

import pytest

from conftest import git, plan
from orchestrator.agents import make_adapter
from orchestrator.agents.base import Invocation, extract_json
from orchestrator.agents.claude_adapter import ClaudeAdapter
from orchestrator.agents.codex_adapter import CodexAdapter
from orchestrator.cli import main as cli_main
from orchestrator.config import ConfigError, load_config
from orchestrator.models import TaskStatus
from orchestrator.planner import PlanError, validate_plan
from orchestrator.procutil import resolve_command
from orchestrator.review_manager import REVIEW_SCHEMA, parse_review
from orchestrator.scheduler import DagScheduler
from orchestrator.security import Redactor, is_safe_relative, path_matches
from orchestrator.worktree_manager import Git, GitSafetyError, WorktreeManager


# ------------------------------------------------------------------ security
@pytest.mark.parametrize("path,patterns,expected", [
    ("src/api/x.py", ["src/api/**"], True),
    ("src/api", ["src/api/**"], True),
    ("src/apix/x.py", ["src/api/**"], False),
    ("tests/test_a.py", ["tests/test_a.py"], True),
    ("deep/dir/x.pyc", ["*.pyc"], True),
    ("a/__pycache__/m.pyc", ["**/__pycache__/**"], True),
    ("__pycache__/m.pyc", ["**/__pycache__/**"], True),
    ("web/index.html", ["web"], True),
    ("anything/at/all", ["**"], True),
    ("calc.py", ["web/**", "tests/**"], False),
    (".env", ["**/.env", ".env"], True),
    ("sub\\win\\path.py", ["sub/**"], True),
])
def test_path_matches(path, patterns, expected):
    assert path_matches(path, patterns) is expected


def test_is_safe_relative():
    assert is_safe_relative("a/b.py")
    assert not is_safe_relative("../x")
    assert not is_safe_relative("/etc/passwd")
    assert not is_safe_relative("C:/Windows/x")


def test_redactor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnopqrstuvwxyz")
    r = Redactor()
    out = r("key sk-proj-abcdefghijklmnopqrstuvwxyz token=abcd1234efgh ghp_abcdefghijklmnopqrstuvwxyz12 "
            "Authorization: Bearer abcdefghijklmnopqrstuvwx")
    assert "abcdefghijklmnopqrstuvwxyz" not in out
    assert "abcd1234efgh" not in out
    assert "ghp_" not in out
    assert out.count("***REDACTED***") >= 4


# ------------------------------------------------------------------ config
def test_config_defaults_and_validation(tmp_path):
    cfg = load_config(None, {"storage": {"data_dir": str(tmp_path)}})
    assert cfg["limits"]["max_parallel_workers"] == 3
    assert cfg["integration"]["auto_push"] is False
    with pytest.raises(ConfigError):
        load_config(None, {"validation": {"test_command": "pytest -q"}})      # shell strings refused
    with pytest.raises(ConfigError):
        load_config(None, {"integration": {"require_manual_main_merge": False}})
    with pytest.raises(ConfigError):
        load_config(None, {"limits": {"max_parallel_workers": 0}})


def test_bundled_config_yaml_loads():
    from conftest import ROOT
    cfg = load_config(ROOT / "config.yaml")
    assert cfg["agents"]["reviewers"] == ["claude", "codex"]
    assert cfg["limits"]["max_repair_rounds"] == 2


# ------------------------------------------------------------------ planner
def _cfg(tmp_path):
    return load_config(None, {"storage": {"data_dir": str(tmp_path)}})


def test_plan_validation_ok_and_warnings(tmp_path):
    p = validate_plan(plan({"id": "T1 Models!", "provider": "codex", "files": ["a/**"]},
                           {"id": "t2", "provider": "claude", "files": ["a/**"]},
                           {"id": "t3", "provider": "claude", "files": [], "depends_on": ["T1 Models!"]}),
                      _cfg(tmp_path), ["claude", "codex"])
    ids = [t.id for t in p.tasks]
    assert ids == ["t1-models", "t2", "t3"]
    assert p.tasks[2].depends_on == ["t1-models"] and p.tasks[2].files == ["**"]
    assert any("overlapping scope" in w for w in p.warnings)


@pytest.mark.parametrize("bad,msg", [
    (plan({"id": "a", "provider": "codex", "depends_on": ["zzz"]}), "unknown task"),
    (plan({"id": "a", "provider": "codex"}, {"id": "a", "provider": "codex"}), "duplicate"),
    (plan({"id": "a", "provider": "codex", "depends_on": ["a"]}), "itself"),
    (plan({"id": "a", "provider": "codex", "files": ["../outside/**"]}), "unsafe"),
    ({"tasks": []}, "non-empty"),
])
def test_plan_validation_errors(tmp_path, bad, msg):
    with pytest.raises(PlanError, match=msg):
        validate_plan(bad, _cfg(tmp_path), ["claude", "codex"])


def test_plan_task_limit(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["limits"]["max_tasks"] = 2
    with pytest.raises(PlanError, match="limit"):
        validate_plan(plan(*({"id": f"t{i}", "provider": "codex"} for i in range(3))), cfg, ["codex"])


# ------------------------------------------------------------------ scheduler
def test_scheduler_dag_order_parallelism_and_blocking():
    deps = {"a": [], "b": [], "c": ["a"], "d": ["c", "b"], "e": ["x"], "x": []}
    status = {k: TaskStatus.PENDING for k in deps}
    order: list[str] = []
    outcome = {"x": TaskStatus.FAILED}

    async def run_task(tid):
        order.append(tid)
        status[tid] = TaskStatus.RUNNING
        await asyncio.sleep(0.05)
        status[tid] = outcome.get(tid, TaskStatus.COMPLETED)

    def set_status(tid, st, _err):
        status[tid] = st

    async def go():
        s = DagScheduler(deps, lambda t: status[t], set_status, run_task, 2, asyncio.Event())
        await s.run()
        return s

    s = asyncio.run(go())
    assert order.index("c") > order.index("a") and order.index("d") > order.index("c")
    assert status["e"] == TaskStatus.BLOCKED and "e" not in order
    assert status["d"] == TaskStatus.COMPLETED
    assert s.max_observed_parallel == 2


# ------------------------------------------------------------------ adapters
def test_claude_adapter_builds_restricted_command(tmp_path):
    cfg = _cfg(tmp_path)
    a = ClaudeAdapter(cfg["providers"]["claude"], base_argv=["claude"])
    inv = a.build(purpose="work", prompt="PROMPT", cwd=str(tmp_path), artifacts_dir=str(tmp_path), model="m1")
    argv = inv.argv
    assert argv[:2] == ["claude", "-p"]
    assert "--restricted" in argv and argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert "--dangerously-skip-permissions" not in argv and "bypassPermissions" not in argv
    assert "PROMPT" not in argv and inv.stdin_text == "PROMPT"      # prompt goes through stdin
    assert "Bash(git push:*)" in argv[argv.index("--disallowedTools"):]
    assert argv[argv.index("--model") + 1] == "m1"
    review = a.build(purpose="review", prompt="p", cwd=str(tmp_path), artifacts_dir=str(tmp_path),
                     schema=REVIEW_SCHEMA)
    tools = review.argv[review.argv.index("--tools") + 1]
    assert "Write" not in tools and "Edit" not in tools
    assert json.loads(review.argv[review.argv.index("--json-schema") + 1]) == REVIEW_SCHEMA


def test_claude_adapter_parses_real_stream_json_shape(tmp_path):
    a = ClaudeAdapter({}, base_argv=["claude"])
    inv = Invocation("claude", "review", [], str(tmp_path), "")
    out = "\n".join([
        json.dumps({"type": "system", "subtype": "init", "model": "x"}),
        json.dumps({"type": "stream_event", "event": {}}),
        json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "{}",
                    "structured_output": {"verdict": "approve", "summary": "", "findings": []},
                    "total_cost_usd": 0.12, "usage": {"input_tokens": 3}, "permission_denials": [
                        {"tool_name": "Bash"}]}),
    ])
    res = a.parse(inv, 0, out, "")
    assert res.ok and res.cost_usd == 0.12 and res.structured["verdict"] == "approve"
    assert res.permission_denials == [{"tool_name": "Bash"}]
    bad = a.parse(inv, 1, "", "boom")
    assert not bad.ok and "no result event" in bad.error
    assert ClaudeAdapter.log_line(json.dumps({"type": "stream_event"})) is None


def test_codex_adapter_builds_sandboxed_command_and_parses(tmp_path):
    cfg = _cfg(tmp_path)
    a = CodexAdapter(cfg["providers"]["codex"], base_argv=["codex"])
    inv = a.build(purpose="work", prompt="P", cwd=str(tmp_path / "wt"), artifacts_dir=str(tmp_path))
    argv = inv.argv
    assert argv[:2] == ["codex", "exec"] and argv[-1] == "-"
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert argv[argv.index("--cd") + 1] == str(tmp_path / "wt")
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert 'approval_policy="never"' in argv
    rv = a.build(purpose="review", prompt="P", cwd=str(tmp_path), artifacts_dir=str(tmp_path), schema=REVIEW_SCHEMA)
    assert rv.argv[rv.argv.index("--sandbox") + 1] == "read-only"
    schema_file = rv.argv[rv.argv.index("--output-schema") + 1]
    assert json.loads(open(schema_file).read()) == REVIEW_SCHEMA
    open(rv.extra["last_message"], "w").write('{"verdict":"approve","summary":"s","findings":[]}')
    out = "\n".join([json.dumps({"type": "thread.started"}),
                     json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}),
                     json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})])
    res = a.parse(rv, 0, out, "")
    assert res.ok and res.usage == {"input_tokens": 6, "output_tokens": 3}
    assert res.structured["verdict"] == "approve"
    failed = a.parse(rv, 1, json.dumps({"type": "turn.failed", "error": {"message": "quota"}}), "")
    assert not failed.ok and "quota" in failed.error


def test_review_parsing_normalises():
    r = parse_review("codex", "abc", {"verdict": "weird", "summary": "s", "findings": [
        {"severity": "BLOCKER", "file": "a.py", "line": "x", "title": "t", "recommendation": "r"}]})
    assert r.verdict == "request_changes" and r.findings[0].severity == "medium" and r.findings[0].line is None
    assert parse_review("codex", "abc", None).verdict == "error"


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('text before {"a": 2} after') == {"a": 2}
    assert extract_json("nothing") is None


def test_resolve_command_errors():
    from orchestrator.procutil import CommandNotFound
    with pytest.raises(CommandNotFound):
        resolve_command(None, "surely-not-a-real-binary-xyz")
    assert resolve_command([sys.executable, "x"], "ignored")[1] == "x"


def test_windows_cmd_shim_resolution(tmp_path, monkeypatch):
    import orchestrator.procutil as pu
    shim_dir = tmp_path / "npm"
    js = shim_dir / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    js.parent.mkdir(parents=True)
    js.write_text("// js")
    (shim_dir / "node.exe").write_text("")
    shim = shim_dir / "codex.cmd"
    shim.write_text('@ECHO off\r\nSET "_prog=node"\r\n"%_prog%"  "%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js" %*\r\n'
                    .replace("\\", "/") if False else
                    '@ECHO off\r\n"%_prog%"  "%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js" %*\r\n')
    monkeypatch.setattr(pu, "IS_WINDOWS", True)
    argv = pu.resolve_command([str(shim)], "codex")
    assert argv[0].endswith("node.exe") and argv[1].replace("\\", "/").endswith("@openai/codex/bin/codex.js")


# ------------------------------------------------------------------ git guard
def test_git_guard_refuses_dangerous_operations(repo, tmp_path):
    cfg = _cfg(tmp_path)
    wt = WorktreeManager(repo, tmp_path / "wts", cfg)
    g = wt.git
    for args in (["push", "origin", "main"], ["reset", "--hard", "HEAD~1"], ["clean", "-fdx"],
                 ["checkout", "main"], ["branch", "-D", "main"], ["branch", "-f", "main", "HEAD"],
                 ["commit", "--force"]):
        with pytest.raises(GitSafetyError):
            g.run(*args)
    with pytest.raises(GitSafetyError):
        wt.create_worktree(tmp_path / "wts" / "x", "main", "HEAD")
    with pytest.raises(GitSafetyError):
        wt.create_worktree(tmp_path / "wts" / "y", "feature/user-branch", "HEAD")
    with pytest.raises(GitSafetyError):
        wt.remove_worktree(repo)                       # never outside the orchestrator worktree root
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_worktree_changes_detection(repo, tmp_path):
    cfg = _cfg(tmp_path)
    wt = WorktreeManager(repo, tmp_path / "wts", cfg)
    p = wt.create_worktree(tmp_path / "wts" / "t", "agent/r1/t-x", "HEAD")
    (p / "calc.py").write_text("changed\n")
    (p / "new dir").mkdir()
    (p / "new dir" / "ü.py").write_text("x\n")
    (p / "README.md").unlink()
    (p / "__pycache__").mkdir()
    (p / "__pycache__" / "c.pyc").write_text("")
    ch = {c.path: c.status for c in wt.changes(p, cfg["validation"]["ignore_patterns"])}
    assert ch == {"calc.py": "M", "new dir/ü.py": "A", "README.md": "D"}
    sha = wt.commit_paths(p, sorted(ch), "msg\n\nOrchestrator-Task: r1/t")
    assert set(wt.commit_files(sha)) == set(ch)
    assert (p / "__pycache__" / "c.pyc").exists() and wt.changes(p, cfg["validation"]["ignore_patterns"]) == []
    wt.remove_worktree(p, delete_branch="agent/r1/t-x")
    assert not p.exists() and not wt.branch_exists("agent/r1/t-x")


# ------------------------------------------------------------------ CLI
def test_cli_status_plan_logs_report_on_completed_run(h, repo, capsys, tmp_path):
    import yaml
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]}),
                "tasks": {"t1": {"write": {"a.py": "a = 1\n"}}}})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(h.config()))
    rc = cli_main(["--config", str(cfg_path), "run", "make a", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0 and "finished: SUCCEEDED" in out and "task -> COMPLETED" in out
    for cmd in (["status"], ["status", "--all"], ["plan"], ["logs"], ["logs", "--agent", "--task", "t1"],
                ["report"], ["status", "--json"]):
        assert cli_main(["--config", str(cfg_path), *cmd]) == 0, cmd
    out = capsys.readouterr().out
    assert "1/1 tasks completed" in out and "# Orchestrator report" in out and "[codex/general]" in out
    assert cli_main(["--config", str(cfg_path), "status", "nope"]) == 1
