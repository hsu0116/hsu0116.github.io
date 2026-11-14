"""Final run report (Markdown) built only from persisted state, so it also works after a crash."""
from __future__ import annotations

import json
from pathlib import Path

from .models import TaskStatus
from .state_store import StateStore


def _fmt_cost(v) -> str:
    return f"${v:.4f}" if v else "-"


def build_report(store: StateStore, run_id: str) -> str:
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"unknown run {run_id}")
    tasks = store.list_tasks(run_id)
    agent_runs = store.list_agent_runs(run_id)
    reviews = store.list_reviews(run_id)
    tests = store.list_test_results(run_id)
    plan = json.loads(run["plan_json"]) if run["plan_json"] else None
    lines: list[str] = []
    a = lines.append
    a(f"# Orchestrator report - {run_id}")
    a("")
    a(f"- **Status:** {run['status']}")
    a(f"- **Request:** {run['request']}")
    a(f"- **Repository:** `{run['repo_path']}`")
    a(f"- **Base:** `{run['base_ref']}` @ `{(run['base_commit'] or '')[:12]}`")
    a(f"- **Integration branch:** `{run['integration_branch']}` (main is never modified automatically)")
    a(f"- **Repair rounds used:** {run['repair_round']}")
    a(f"- **Created / finished:** {run['created_at']} / {run['finished_at'] or '-'}")
    total = sum((r["cost_usd"] or 0) for r in agent_runs)
    a(f"- **Agent invocations:** {len(agent_runs)}, usage {_fmt_cost(total)} as an API-equivalent estimate "
      f"(subscription logins only - nothing is billed to an API key; codex reports tokens only)")
    if run["error"]:
        a(f"- **Problem:** {run['error']}")
    a("")
    if plan:
        a("## Plan")
        a(plan.get("summary", ""))
        if plan.get("warnings"):
            a("")
            for w in plan["warnings"]:
                a(f"- warning: {w}")
        a("")
    a("## Tasks")
    a("")
    a("| Task | Provider | Kind | Status | Attempts | Depends on | Changed files | Commit |")
    a("|---|---|---|---|---|---|---|---|")
    for t in tasks:
        spec = t["spec"]
        files = ", ".join(t["changed_files"][:6]) + (" ..." if len(t["changed_files"]) > 6 else "")
        a(f"| {t['id']} | {t['provider']} | {t['kind']} | {t['status']} | {t['attempts']} | "
          f"{', '.join(spec.depends_on) or '-'} | {files or '-'} | {(t['integrated_sha'] or '')[:10] or '-'} |")
    failed = [t for t in tasks if t["status"] != TaskStatus.COMPLETED.value]
    if failed:
        a("")
        a("### Failures / open items")
        for t in failed:
            a(f"- **{t['id']}** ({t['status']}): {t['error'] or 'no details'}"
              + (f" - branch `{t['branch']}` kept for inspection" if t["branch"] else ""))
    a("")
    a("## Tests")
    if not tests:
        a("- no tests were run")
    for tr in tests:
        cmd = " ".join(json.loads(tr["command_json"]) or []) if tr["command_json"] and tr["command_json"] != "null" else "-"
        verdict = "PASSED" if tr["passed"] else ("SKIPPED" if cmd == "-" else "FAILED")
        a(f"- [{tr['scope']} / round {tr['round']}] **{verdict}** `{cmd}` exit={tr['exit_code']}"
          f"{' TIMEOUT' if tr['timed_out'] else ''} ({(tr['duration_seconds'] or 0):.1f}s)"
          + (f" - output: `{tr['output_path']}`" if tr["output_path"] else ""))
    last = next((tr for tr in reversed(tests) if tr["scope"] == "integration"), None)
    if last and not last["passed"] and last["output_tail"]:
        a("")
        a("Last failing integration test output (tail):")
        a("```")
        a(last["output_tail"][-2500:])
        a("```")
    a("")
    a("## Cross reviews")
    if not reviews:
        a("- no reviews")
    for r in reviews:
        a(f"- round {r['round']} **{r['reviewer']}** -> `{r['verdict']}` on `{(r['target_commit'] or '')[:10]}`"
          + (f" - error: {r['error']}" if r["error"] else ""))
        if r["summary"]:
            a(f"  - {r['summary'][:500]}")
        for f in r["findings"]:
            loc = f"{f.get('file')}{(':' + str(f['line'])) if f.get('line') else ''}"
            a(f"  - [{f.get('severity')}] `{loc}` {f.get('title')} - {f.get('recommendation', '')[:300]}")
    a("")
    a("## Agent invocations")
    a("")
    a("| # | Task | Purpose | Provider | Exit | Timeout | Cancelled | Usage (API-eq. est.) | Log |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in agent_runs:
        a(f"| {r['id']} | {r['task_id'] or '-'} | {r['purpose']} | {r['provider']} | {r['exit_code']} | "
          f"{'yes' if r['timed_out'] else ''} | {'yes' if r['cancelled'] else ''} | {_fmt_cost(r['cost_usd'])} | "
          f"`{Path(r['log_path']).name if r['log_path'] else '-'}` |")
    a("")
    a("## Next steps (manual)")
    if run["status"] in ("SUCCEEDED", "UNVERIFIED"):
        a("Review the integration branch, then merge it yourself if you are satisfied:")
    else:
        a("The run did not fully succeed. Inspect the failures above; you can fix the integration branch "
          "manually or run `orchestrator resume --retry-failed " + run_id + "`.")
    a("")
    a("```bash")
    a(f"git -C \"{run['repo_path']}\" log --oneline {run['base_commit'][:12] if run['base_commit'] else ''}..{run['integration_branch']}")
    a(f"git -C \"{run['repo_path']}\" diff {run['base_ref']}...{run['integration_branch']}")
    a(f"git -C \"{run['repo_path']}\" switch {run['base_ref']}")
    a(f"git -C \"{run['repo_path']}\" merge --no-ff {run['integration_branch']}")
    a("```")
    a("")
    a("Nothing was pushed. Pushing is always a manual step.")
    return "\n".join(lines) + "\n"


def write_report(store: StateStore, run_id: str, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "report.md"
    path.write_text(build_report(store, run_id), encoding="utf-8")
    return path
