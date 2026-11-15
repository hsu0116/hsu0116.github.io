"""Terminal CLI. A thin layer over ``engine.Orchestrator`` and the SQLite state."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from .config import ConfigError, load_config
from .engine import Orchestrator, OrchestratorError
from .models import RunStatus
from .procutil import pid_alive
from .worktree_manager import GitError, GitSafetyError

ROOT = Path(__file__).resolve().parent.parent
LEVEL_MARK = {"info": " ", "warning": "!", "error": "X"}


def _find_config(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("ORCHESTRATOR_CONFIG")
    if env:
        return env
    for cand in (Path.cwd() / "config.yaml", ROOT / "config.yaml"):
        if cand.exists():
            return str(cand)
    return None


def _print_event(e: dict) -> None:
    task = f"[{e['task_id']}] " if e["task_id"] else ""
    print(f"{e['ts'][11:19]} {LEVEL_MARK.get(e['level'], ' ')} {task}{e['message']}", flush=True)


async def _tail_events(orch: Orchestrator, run_id: str, stop: asyncio.Event, after: int = 0) -> int:
    last = after
    while True:
        for e in orch.store.list_events(run_id, after_id=last):
            _print_event(e)
            last = e["id"]
        if stop.is_set():
            return last
        await asyncio.sleep(0.3)


async def _run_with_tail(orch: Orchestrator, run_id: str, coro, quiet: bool):
    stop = asyncio.Event()
    after = 0
    events = orch.store.list_events(run_id)
    if events:
        for e in events[-5:]:
            _print_event(e)
        after = events[-1]["id"]
    tail = None if quiet else asyncio.create_task(_tail_events(orch, run_id, stop, after))
    try:
        return await coro
    finally:
        stop.set()
        if tail:
            await tail


def _resolve_run(orch: Orchestrator, run_id: str | None) -> str:
    rid = run_id or orch.store.latest_run_id()
    if not rid or not orch.store.get_run(rid):
        raise OrchestratorError("no run found" if not rid else f"unknown run {rid}")
    return rid


def _effective_status(run: dict) -> str:
    st = run["status"]
    if RunStatus(st).active and not pid_alive(run["pid"]):
        return f"{st} (INTERRUPTED - process not running; use 'resume')"
    return st


# ----------------------------------------------------------------------------- commands
def cmd_doctor(orch: Orchestrator, args) -> int:
    from .doctor import run_doctor
    ok, lines = run_doctor(orch.config)
    print("\n".join(lines))
    return 0 if ok else 1


def cmd_run(orch: Orchestrator, args) -> int:
    repo = args.repo or os.getcwd()
    run_id = orch.create_run(args.request, repo, args.base)
    print(f"run id: {run_id}")
    status = asyncio.run(_run_with_tail(
        orch, run_id, orch.execute(run_id, plan_only=args.plan_only, plan_file=args.plan_file), args.quiet))
    print(f"\nrun {run_id} finished: {status}")
    print(f"report: {orch.data_dir / 'runs' / run_id / 'report.md'}")
    return 0 if status in (RunStatus.SUCCEEDED.value, RunStatus.PLANNED.value) else 2


def cmd_resume(orch: Orchestrator, args) -> int:
    run_id = _resolve_run(orch, args.run_id)
    status = asyncio.run(_run_with_tail(orch, run_id, orch.resume(run_id, retry_failed=args.retry_failed),
                                        args.quiet))
    print(f"\nrun {run_id} finished: {status}")
    return 0 if status == RunStatus.SUCCEEDED.value else 2


def cmd_status(orch: Orchestrator, args) -> int:
    if args.all:
        for r in orch.store.list_runs(50):
            print(f"{r['id']}  {_effective_status(r):<20}  {r['created_at']}  {r['request'][:60]}")
        return 0
    run_id = _resolve_run(orch, args.run_id)
    run = orch.store.get_run(run_id)
    tasks = orch.store.list_tasks(run_id)
    if args.json:
        print(json.dumps({"run": {k: v for k, v in run.items() if k != "config_json"},
                          "tasks": [{**{k: v for k, v in t.items() if k != "spec"}, "spec": t["spec"].to_dict()}
                                    for t in tasks]}, indent=2, ensure_ascii=False))
        return 0
    done = sum(1 for t in tasks if t["status"] == "COMPLETED")
    print(f"run:          {run_id}")
    print(f"status:       {_effective_status(run)}")
    print(f"request:      {run['request']}")
    print(f"repo:         {run['repo_path']}")
    print(f"integration:  {run['integration_branch']}  (base {run['base_ref']} @ {(run['base_commit'] or '')[:10]})")
    print(f"progress:     {done}/{len(tasks)} tasks completed, repair round {run['repair_round']}")
    print(f"usage:        ${orch.store.total_cost(run_id):.4f} API-equivalent estimate "
          f"(subscription usage, not billed)")
    if run["error"]:
        print(f"problem:      {run['error']}")
    if tasks:
        print()
        print(f"{'TASK':<22} {'PROVIDER':<8} {'STATUS':<11} {'TRY':<4} {'DEPENDS ON':<20} ERROR")
        for t in tasks:
            deps = ",".join(t["spec"].depends_on) or "-"
            err = (t["error"] or "").replace("\n", " ")[:70]
            print(f"{t['id']:<22} {t['provider']:<8} {t['status']:<11} {t['attempts']:<4} {deps:<20} {err}")
    active = [a for a in orch.store.list_agent_runs(run_id) if not a["finished_at"]]
    if active and RunStatus(run["status"]).active and pid_alive(run["pid"]):
        print("\nactive agents:")
        for a in active:
            print(f"  {a['provider']:<7} {a['purpose']:<7} task={a['task_id'] or '-'} pid={a['pid']} since {a['started_at']}")
    return 0


def cmd_plan(orch: Orchestrator, args) -> int:
    run_id = _resolve_run(orch, args.run_id)
    run = orch.store.get_run(run_id)
    if not run["plan_json"]:
        print("no plan yet")
        return 1
    plan = json.loads(run["plan_json"])
    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    print(f"plan for {run_id}: {plan.get('summary', '')}\n")
    if plan.get("shared_contracts"):
        print("shared contracts:\n" + plan["shared_contracts"] + "\n")
    for t in plan["tasks"]:
        print(f"- {t['id']} [{t['provider']}/{t['role']}] {t['title']}")
        print(f"    depends on: {', '.join(t['depends_on']) or '-'}")
        print(f"    files:      {', '.join(t['files'])}")
        for c in t.get("acceptance_criteria", []):
            print(f"    done when:  {c}")
        if t.get("verification"):
            print(f"    verify:     {t['verification']}")
    for w in plan.get("warnings", []):
        print(f"! {w}")
    return 0


def cmd_logs(orch: Orchestrator, args) -> int:
    run_id = _resolve_run(orch, args.run_id)
    if args.agent:
        runs = [a for a in orch.store.list_agent_runs(run_id) if not args.task or a["task_id"] == args.task]
        if not runs:
            print("no agent logs")
            return 1
        path = runs[-1]["log_path"]
        print(f"== {path}")
        if path and Path(path).exists():
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-args.tail:]))
        return 0
    events = orch.store.list_events(run_id, task_id=args.task)
    for e in events[-args.tail:]:
        _print_event(e)
    if args.follow:
        last = events[-1]["id"] if events else 0
        try:
            while True:
                for e in orch.store.list_events(run_id, after_id=last, task_id=args.task):
                    _print_event(e)
                    last = e["id"]
                run = orch.store.get_run(run_id)
                if not RunStatus(run["status"]).active or not pid_alive(run["pid"]):
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return 0


def cmd_stop(orch: Orchestrator, args) -> int:
    run_id = _resolve_run(orch, args.run_id)
    print(orch.request_stop(run_id))
    return 0


def cmd_cancel(orch: Orchestrator, args) -> int:
    orch.request_cancel(args.run_id, args.task_id)
    print(f"cancel requested for task {args.task_id}")
    return 0


def cmd_report(orch: Orchestrator, args) -> int:
    from .report import build_report, write_report
    run_id = _resolve_run(orch, args.run_id)
    text = build_report(orch.store, run_id)
    print(text)
    if args.write:
        print(f"written to {write_report(orch.store, run_id, orch.data_dir / 'runs' / run_id)}")
    return 0


def _usage_rows(orch: Orchestrator, run_id: str | None) -> list[dict]:
    if run_id:
        return orch.store.list_agent_runs(run_id)
    return [dict(r) for r in orch.store._query("SELECT * FROM agent_runs ORDER BY id")]


def cmd_usage(orch: Orchestrator, args) -> int:
    """Token usage per provider and a check that only subscription logins are used."""
    from .agents import PROVIDERS
    rows = _usage_rows(orch, args.run_id)
    scope = f"run {args.run_id}" if args.run_id else f"all runs ({len(orch.store.list_runs(100000))})"
    print(f"== Agent usage: {scope}\n")
    stats: dict[str, dict] = {}
    violations = 0
    for r in rows:
        st = stats.setdefault(r["provider"], {"calls": 0, "failed": 0, "cost": 0.0, "tokens": {}})
        st["calls"] += 1
        if r["exit_code"] not in (0,) or r["error"]:
            st["failed"] += 1
        st["cost"] += r["cost_usd"] or 0.0
        if r["error"] and "SUBSCRIPTION POLICY VIOLATION" in r["error"]:
            violations += 1
        try:
            usage = json.loads(r["usage_json"]) if r["usage_json"] else {}
        except (json.JSONDecodeError, TypeError):
            usage = {}
        for k, v in usage.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and "token" in k:
                st["tokens"][k] = st["tokens"].get(k, 0) + v
    if not stats:
        print("no agent calls recorded")
    for prov, st in stats.items():
        t = st["tokens"]
        total = sum(t.values())
        print(f"[{prov}] calls={st['calls']} (failed/cancelled {st['failed']})")
        for k in sorted(t):
            print(f"    {k:<32} {int(t[k]):>15,}")
        print(f"    {'total tokens':<32} {int(total):>15,}")
        if prov == "claude":
            print(f"    API-equivalent estimate          ${st['cost']:,.2f}  (NOT billed: subscription login)")
        else:
            print("    API-equivalent estimate          n/a (codex reports tokens only; NOT billed: ChatGPT login)")
        print()
    print("== Billing safety check")
    for p in PROVIDERS:
        ok, msg = orch.available(p)
        print(f"  {p:<7} {'OK  ' if ok else 'WARN'} {msg}")
    print(f"  subscription policy violations recorded: {violations}")
    print("\nThe orchestrator strips API keys and refuses API-key logins, so agent calls use only your")
    print("Claude / ChatGPT subscription usage. Plan limits and any optional paid extras are shown on the")
    print("providers' own settings pages (see README).")
    return 0 if violations == 0 else 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchestrator",
                                description="Claude Code x Codex multi-agent orchestrator (git worktree based)")
    p.add_argument("--config", help="config.yaml path (default: ./config.yaml, $ORCHESTRATOR_CONFIG, or the bundled one)")
    p.add_argument("--data-dir", help="override storage.data_dir")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("doctor", help="check Python, git, claude and codex installation/authentication")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("run", help="plan and execute a development request")
    s.add_argument("request")
    s.add_argument("--repo", help="target git repository (default: current directory)")
    s.add_argument("--base", help="base branch/commit (default: current branch)")
    s.add_argument("--plan-file", help="use this JSON plan instead of the AI planner")
    s.add_argument("--plan-only", action="store_true", help="only create the plan; execute later with 'resume'")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="show run status")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--all", action="store_true", help="list all runs")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("list", help="list runs")
    s.set_defaults(func=lambda o, a: cmd_status(o, argparse.Namespace(all=True, run_id=None, json=False)))

    s = sub.add_parser("plan", help="show the task plan of a run")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("logs", help="show orchestrator events or agent output")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--task")
    s.add_argument("--agent", action="store_true", help="show the latest agent output log (of --task)")
    s.add_argument("--tail", type=int, default=200)
    s.add_argument("-f", "--follow", action="store_true")
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("stop", help="stop a running run (agents are terminated, resumable)")
    s.add_argument("run_id", nargs="?")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("cancel", help="cancel one task of a running run")
    s.add_argument("run_id")
    s.add_argument("task_id")
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("resume", help="resume a stopped/interrupted run")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--retry-failed", action="store_true", help="also retry FAILED/CONFLICT/CANCELLED tasks")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_resume)

    s = sub.add_parser("report", help="print the final report")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--write", action="store_true", help="also write report.md")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("usage", help="token usage per provider and subscription-only billing check")
    s.add_argument("run_id", nargs="?", help="limit to one run (default: all runs)")
    s.set_defaults(func=cmd_usage)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        overrides = {"storage": {"data_dir": str(Path(args.data_dir).resolve())}} if args.data_dir else None
        config = load_config(_find_config(args.config), overrides)
        orch = Orchestrator(config)
        return args.func(orch, args)
    except (ConfigError, OrchestratorError, GitError, GitSafetyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted - agents terminated; resume with: orchestrator resume", file=sys.stderr)
        return 130
