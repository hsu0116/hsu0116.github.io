"""Execution engine. UI-independent: the CLI (and a future web dashboard) only call the public
methods below and read the SQLite state."""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path

from .agent_runner import AgentRunner
from .agents import PLAN, PROVIDERS, REPAIR, WORK, AgentAdapter, make_adapter
from .integrator import Integrator
from .models import (TERMINAL_TASK_STATES, Plan, ReviewResult, RunStatus, TaskSpec, TaskStatus)
from .planner import (PLAN_SCHEMA, PlanError, apply_provider_availability, build_planner_prompt,
                      load_plan_file, render, validate_plan)
from .procutil import kill_process_tree, pid_alive
from .result_validator import validate_task_result
from .review_manager import blocking_findings, run_reviews
from .scheduler import DagScheduler
from .security import Redactor
from .state_store import StateStore, now_iso
from .test_runner import TestOutcome, resolve_test_command, run_tests
from .worktree_manager import GitError, GitSafetyError, WorktreeManager

STOPPED_MARK = "stopped by user request"


class OrchestratorError(RuntimeError):
    pass


def new_run_id() -> str:
    return datetime.now().strftime("r%Y%m%d-%H%M%S-") + secrets.token_hex(2)


class Orchestrator:
    def __init__(self, config: dict, adapters: dict[str, AgentAdapter] | None = None):
        self.config = config
        self.data_dir = Path(config["storage"]["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.redact = Redactor()
        self.store = StateStore(self.data_dir / "orchestrator.db", self.redact)
        self.adapters: dict[str, AgentAdapter] = adapters or {p: make_adapter(p, config) for p in PROVIDERS}
        self._availability: dict[str, tuple[bool, str]] = {}
        # per-execution state
        self.stop_event = asyncio.Event()
        self.task_cancel: dict[str, asyncio.Event] = {}
        self.integ_lock = asyncio.Lock()
        self.budget_exceeded = False

    # ================================================================ public API
    def create_run(self, request: str, repo: str | Path, base_ref: str | None = None) -> str:
        if self.config["integration"].get("auto_push"):
            raise OrchestratorError("integration.auto_push=true is not supported; pushing is always manual")
        wt = WorktreeManager(repo, self.data_dir / "worktrees", self.config)
        wt.ensure_repo()
        run_id = new_run_id()
        base_ref = base_ref or wt.current_branch() or "HEAD"
        base_commit = wt.resolve_commit(base_ref)
        self.store.create_run(run_id, request, str(wt.repo), self.config, RunStatus.PLANNING.value)
        self.store.update_run(run_id, base_ref=base_ref, base_commit=base_commit,
                              integration_branch=wt.integration_branch_name(run_id),
                              integration_worktree=str(self.data_dir / "worktrees" / run_id / "_integration"))
        self.store.log_event(run_id, f"run created for {wt.repo} (base {base_ref} @ {base_commit[:10]})")
        return run_id

    def request_stop(self, run_id: str) -> str:
        run = self._run(run_id)
        self.store.update_run(run_id, stop_requested=1)
        self.store.log_event(run_id, "stop requested", level="warning")
        if RunStatus(run["status"]).active and not pid_alive(run["pid"]):
            self._kill_orphaned_agents(run_id)
            self.store.update_run(run_id, status=RunStatus.STOPPED.value)
            return "orchestrator process is not running; run marked STOPPED"
        return "stop requested; running agents will be terminated"

    def request_cancel(self, run_id: str, task_id: str) -> None:
        if not self.store.get_task(run_id, task_id):
            raise OrchestratorError(f"unknown task {task_id}")
        self.store.update_task(run_id, task_id, cancel_requested=1)
        self.store.log_event(run_id, "cancel requested", level="warning", task_id=task_id)

    async def execute(self, run_id: str, *, plan_only: bool = False, plan_file: str | None = None) -> str:
        """Run (or continue) the full pipeline. Returns the final RunStatus value."""
        run = self._run(run_id)
        self.run_id = run_id
        self.store.update_run(run_id, pid=os.getpid(), stop_requested=0, error=None)
        self.stop_event = asyncio.Event()
        self.task_cancel = {}
        self.budget_exceeded = False
        self.wt = WorktreeManager(run["repo_path"], self.data_dir / "worktrees", self.config)
        self.run_dir = self.data_dir / "runs" / run_id
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self.runner = AgentRunner(self.store, self.redact, self.run_dir / "logs",
                                  self.config["logging"].get("save_agent_output", True))
        watcher = asyncio.create_task(self._watch_control(run_id))
        try:
            self._setup_integration(run)
            run = self._run(run_id)
            if not run["plan_json"]:
                self._set_status(RunStatus.PLANNING)
                plan = await self._make_plan(run, plan_file)
                if plan is None:
                    return self._finish(RunStatus.FAILED)
                if plan_only:
                    self._set_status(RunStatus.PLANNED)
                    self._event("plan created (--plan-only); use 'resume' to execute it")
                    return RunStatus.PLANNED.value
            self._set_status(RunStatus.RUNNING)
            await self._run_dag()
            if self.stop_event.is_set():
                return self._finish(RunStatus.NEEDS_ATTENTION if self.budget_exceeded else RunStatus.STOPPED)
            final = await self._verify_and_repair()
            return self._finish(final)
        except (GitError, GitSafetyError, PlanError, OrchestratorError) as e:
            self.store.update_run(run_id, error=self.redact(str(e)))
            self._event(f"run failed: {e}", "error")
            return self._finish(RunStatus.FAILED)
        except asyncio.CancelledError:
            self._event("orchestrator interrupted", "warning")
            self.store.update_run(run_id, status=RunStatus.STOPPED.value)
            raise
        except Exception as e:  # unexpected bug: never leave the run looking "active"
            self.store.update_run(run_id, error=self.redact(f"internal error: {e!r}"))
            self._event(f"internal error: {e!r}", "error")
            self._finish(RunStatus.FAILED)
            raise
        finally:
            watcher.cancel()
            self.store.update_run(run_id, pid=None)

    async def resume(self, run_id: str, retry_failed: bool = False) -> str:
        run = self._run(run_id)
        status = RunStatus(run["status"])
        if status.active and pid_alive(run["pid"]) and run["pid"] != os.getpid():
            raise OrchestratorError(f"run {run_id} is still being executed by process {run['pid']}")
        if status == RunStatus.SUCCEEDED and not retry_failed:
            return status.value
        self.store.log_event(run_id, f"resuming run (previous status {status.value})")
        self._prepare_resume(run, retry_failed)
        return await self.execute(run_id)

    # ================================================================ internals
    def _run(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if not run:
            raise OrchestratorError(f"unknown run id {run_id}")
        return run

    def _event(self, msg: str, level: str = "info", task_id: str | None = None) -> None:
        self.store.log_event(self.run_id, msg, level=level, task_id=task_id)

    def _set_status(self, st: RunStatus) -> None:
        self.store.update_run(self.run_id, status=st.value)

    def _set_task(self, task_id: str, status: TaskStatus, error: str | None = None, **extra) -> None:
        fields = {"status": status, **extra}
        if error is not None or status in (TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.COMPLETED):
            fields["error"] = error
        if status.terminal:
            fields.setdefault("finished_at", now_iso())
        self.store.update_task(self.run_id, task_id, **fields)
        self._event(f"task -> {status.value}{(': ' + error) if error else ''}",
                    "error" if status in (TaskStatus.FAILED, TaskStatus.CONFLICT) else "info", task_id)

    def _task_status(self, task_id: str) -> TaskStatus:
        return TaskStatus(self.store.get_task(self.run_id, task_id)["status"])

    def available(self, provider: str) -> tuple[bool, str]:
        if provider not in self._availability:
            self._availability[provider] = self.adapters[provider].availability()
        return self._availability[provider]

    def available_providers(self) -> list[str]:
        return [p for p in PROVIDERS if self.available(p)[0]]

    async def _watch_control(self, run_id: str) -> None:
        while True:
            try:
                run = self.store.get_run(run_id)
                if run and run["stop_requested"] and not self.stop_event.is_set():
                    self._event("stop request received - terminating agents", "warning")
                    self.stop_event.set()
                for t in self.store.list_tasks(run_id):
                    ev = self.task_cancel.get(t["id"])
                    if ev and t["cancel_requested"] and not ev.is_set():
                        ev.set()
                if self.stop_event.is_set():
                    for ev in self.task_cancel.values():
                        ev.set()
            except Exception:  # pragma: no cover - keep watching
                pass
            await asyncio.sleep(0.5)

    def _budget_ok(self) -> bool:
        limit = self.config["limits"].get("max_total_cost_usd")
        if limit is None:
            return True
        spent = self.store.total_cost(self.run_id)
        if spent >= float(limit):
            if not self.budget_exceeded:
                self.budget_exceeded = True
                self.store.update_run(self.run_id, error=f"cost limit reached: ${spent:.2f} >= ${limit}")
                self._event(f"cost limit reached (${spent:.2f} >= ${limit}); no new agents will start", "error")
                self.stop_event.set()
            return False
        return True

    # ---------------------------------------------------------------- setup
    def _setup_integration(self, run: dict) -> None:
        branch, path = run["integration_branch"], Path(run["integration_worktree"])
        self.wt.ensure_repo()
        if self.wt.branch_exists(branch):
            self.wt.attach_worktree(path, branch)
        else:
            self.wt.create_worktree(path, branch, run["base_commit"])
            self._event(f"integration branch {branch} created at {run['base_commit'][:10]}")
        self.integrator = Integrator(self.wt, path)
        if self.integrator.recover():
            self._event("aborted an unfinished cherry-pick left by a previous crash", "warning")
        # crash recovery: commits already integrated (trailers) count as completed
        for task_id, sha in self.wt.task_trailers(path, run["base_commit"]).items():
            t = self.store.get_task(run["id"], task_id)
            if t and TaskStatus(t["status"]) != TaskStatus.COMPLETED:
                self.store.update_task(run["id"], task_id, status=TaskStatus.COMPLETED, integrated_sha=sha,
                                       error=None, finished_at=now_iso())
                self._event("recovered as COMPLETED from integration branch history", task_id=task_id)

    def _kill_orphaned_agents(self, run_id: str) -> None:
        """Agents started by a crashed orchestrator may still run: terminate their process trees."""
        for a in self.store.list_agent_runs(run_id):
            if a["finished_at"]:
                continue
            if a["pid"] and pid_alive(a["pid"]):
                kill_process_tree(a["pid"])
                self.store.log_event(run_id, f"terminated orphaned {a['provider']} process {a['pid']}",
                                     "warning", a["task_id"])
            self.store.finish_agent_run(a["id"], cancelled=1, error="orchestrator stopped before the agent finished")

    def _prepare_resume(self, run: dict, retry_failed: bool) -> None:
        self.run_id = run["id"]
        self._kill_orphaned_agents(run["id"])
        wt = WorktreeManager(run["repo_path"], self.data_dir / "worktrees", self.config)
        resettable = {TaskStatus.RUNNING, TaskStatus.VALIDATING, TaskStatus.READY, TaskStatus.BLOCKED}
        if retry_failed:
            resettable |= {TaskStatus.FAILED, TaskStatus.CONFLICT, TaskStatus.CANCELLED}
        for t in self.store.list_tasks(run["id"]):
            st = TaskStatus(t["status"])
            stopped = st == TaskStatus.CANCELLED and (t["error"] or "").startswith(STOPPED_MARK)
            if st in resettable or stopped:
                if t["worktree"]:
                    try:
                        wt.remove_worktree(t["worktree"], delete_branch=t["branch"])
                    except (GitError, GitSafetyError) as e:
                        self.store.log_event(run["id"], f"cleanup failed: {e}", "warning", t["id"])
                fields = dict(status=TaskStatus.PENDING, error=None, worktree=None, branch=None, commit_sha=None,
                              cancel_requested=0)
                if retry_failed and st in (TaskStatus.FAILED, TaskStatus.CONFLICT, TaskStatus.CANCELLED):
                    fields["attempts"] = 0
                elif st in (TaskStatus.RUNNING, TaskStatus.VALIDATING) or stopped:
                    fields["attempts"] = max(0, int(t["attempts"]) - 1)   # interrupted attempt does not count
                self.store.update_task(run["id"], t["id"], **fields)
                self.store.log_event(run["id"], f"reset {st.value} -> PENDING for resume", task_id=t["id"])
        if run["status"] in (RunStatus.NEEDS_ATTENTION.value, RunStatus.FAILED.value) and retry_failed:
            self.store.update_run(run["id"], repair_round=0)

    # ---------------------------------------------------------------- planning
    async def _make_plan(self, run: dict, plan_file: str | None) -> Plan | None:
        available = self.available_providers()
        for p in PROVIDERS:
            ok, msg = self.available(p)
            self._event(f"provider {p}: {'available' if ok else 'UNAVAILABLE'} ({msg})", "info" if ok else "warning")
        if not available:
            raise OrchestratorError("no agent provider is available (install/authenticate claude or codex)")
        if plan_file:
            raw = load_plan_file(plan_file)
            self._event(f"using plan file {plan_file}")
        else:
            planner = self.config["agents"]["planner"]
            if planner not in available:
                if not self.config["agents"].get("fallback_on_unavailable"):
                    raise OrchestratorError(f"planner provider {planner} unavailable")
                self._event(f"planner {planner} unavailable, using {available[0]}", "warning")
                planner = available[0]
            prompt = build_planner_prompt(run["id"], run["request"], self.config, available)
            adapter = self.adapters[planner]
            inv = adapter.build(purpose=PLAN, prompt=prompt, cwd=run["integration_worktree"],
                                artifacts_dir=str(self.run_dir / "artifacts"), schema=PLAN_SCHEMA)
            res = await self.runner.run(adapter, inv, run_id=run["id"], task_id=None,
                                        timeout=self.config["limits"]["planner_timeout_seconds"],
                                        cancel_event=self.stop_event, log_name="planner")
            if not res.ok or res.structured is None:
                self.store.update_run(run["id"], error=f"planning failed: {res.error or 'no structured plan returned'}")
                self._event(f"planning failed: {res.error or 'no structured plan'}", "error")
                return None
            raw = res.structured
        plan = validate_plan(raw, self.config, available)
        apply_provider_availability(plan, available, self.config["agents"].get("fallback_on_unavailable", True))
        self.store.update_run(run["id"], plan_json=plan.to_dict())
        for spec in plan.tasks:
            self.store.add_task(run["id"], spec)
        for w in plan.warnings:
            self._event(f"plan warning: {w}", "warning")
        self._event(f"plan accepted: {len(plan.tasks)} tasks")
        return plan

    def _plan(self) -> Plan:
        return Plan.from_dict(json.loads(self._run(self.run_id)["plan_json"] or "{}"))

    # ---------------------------------------------------------------- DAG phase
    async def _run_dag(self, task_ids: list[str] | None = None) -> None:
        tasks = self.store.list_tasks(self.run_id)
        if task_ids is not None:
            tasks = [t for t in tasks if t["id"] in task_ids]
        deps = {t["id"]: t["spec"].depends_on for t in tasks}
        for tid in deps:
            self.task_cancel.setdefault(tid, asyncio.Event())

        def set_status(tid: str, st: TaskStatus, err: str | None) -> None:
            self._set_task(tid, st, err)

        def cancel_requested(tid: str) -> bool:
            return bool(self.store.get_task(self.run_id, tid)["cancel_requested"])

        sched = DagScheduler(deps, self._task_status, set_status, self._run_task,
                             self.config["limits"]["max_parallel_workers"], self.stop_event, cancel_requested,
                             on_log=lambda m: self._event(m))
        await sched.run()
        self._event(f"scheduler finished (max parallel agents observed: {sched.max_observed_parallel})")

    def _worker_prompt(self, spec: TaskSpec, row: dict, feedback: str) -> str:
        plan = self._plan()
        run = self._run(self.run_id)
        extra = ""
        if feedback:
            extra += f"\n## Previous attempt was rejected\n{feedback}\nFix these problems in this attempt.\n"
        return render("worker.md", purpose=REPAIR if spec.kind == "repair" else WORK, task_id=spec.id,
                      run_id=self.run_id, round=row["repair_round"], provider=spec.provider,
                      request=run["request"], shared_contracts=plan.shared_contracts or "(none)",
                      title=spec.title, description=spec.description,
                      acceptance="\n".join(f"- {a}" for a in spec.acceptance_criteria) or "- (see description)",
                      verification=spec.verification or "(not specified)",
                      files="\n".join(f"- {f}" for f in spec.files), extra=extra)

    async def _run_task(self, task_id: str) -> None:
        row = self.store.get_task(self.run_id, task_id)
        spec: TaskSpec = row["spec"]
        run = self._run(self.run_id)
        max_attempts = 1 + int(self.config["limits"]["max_task_retries"])
        cancel_ev = self.task_cancel.setdefault(task_id, asyncio.Event())
        feedback = ""
        attempt = row["attempts"]
        while attempt < max_attempts:
            attempt += 1
            if self.stop_event.is_set():
                self._set_task(task_id, TaskStatus.CANCELLED, STOPPED_MARK)
                return
            if not self._budget_ok():
                self._set_task(task_id, TaskStatus.CANCELLED, STOPPED_MARK + " (cost limit)")
                return
            provider = spec.provider
            if not self.available(provider)[0]:
                alt = self.available_providers()
                if not alt or not self.config["agents"].get("fallback_on_unavailable", True):
                    self._set_task(task_id, TaskStatus.FAILED, f"provider {provider} unavailable")
                    return
                self._event(f"provider {provider} unavailable, using {alt[0]}", "warning", task_id)
                provider = spec.provider = alt[0]
                self.store.update_task(self.run_id, task_id, spec=spec)
            # base commit: same base for independent tasks, integration head for dependants / repairs
            async with self.integ_lock:
                independent = not spec.depends_on and spec.kind == "work" and not row["repair_round"]
                base = run["base_commit"] if independent else self.integrator.head()
            branch = self.wt.task_branch_name(self.run_id, task_id, spec.title)
            path = self.data_dir / "worktrees" / self.run_id / task_id
            # clean leftovers from a previous attempt of this task
            if path.exists() or self.wt.branch_exists(branch):
                self.wt.remove_worktree(path, delete_branch=branch)
            self.wt.create_worktree(path, branch, base)
            self._set_task(task_id, TaskStatus.RUNNING, None, attempts=attempt, branch=branch, worktree=str(path),
                           base_commit=base, started_at=now_iso(), commit_sha=None)
            adapter = self.adapters[provider]
            prompt = self._worker_prompt(spec, self.store.get_task(self.run_id, task_id), feedback)
            inv = adapter.build(purpose=REPAIR if spec.kind == "repair" else WORK, prompt=prompt, cwd=str(path),
                                artifacts_dir=str(self.run_dir / "artifacts" / task_id), model=spec.model)
            res = await self.runner.run(adapter, inv, run_id=self.run_id, task_id=task_id,
                                        timeout=self.config["limits"]["task_timeout_seconds"],
                                        cancel_event=cancel_ev, log_name=f"{task_id}-a{attempt}-{provider}")
            if res.cancelled:
                msg = STOPPED_MARK if self.stop_event.is_set() else "cancelled by user"
                self._set_task(task_id, TaskStatus.CANCELLED, msg)
                self._cleanup_task_worktree(task_id, keep_branch=False)
                return
            self._set_task(task_id, TaskStatus.VALIDATING, None)
            v = validate_task_result(self.wt, str(path), base, spec, res, self.config)
            for w in v.warnings:
                self._event(f"validation warning: {w}", "warning", task_id)
            if v.ok and self.config["validation"].get("run_tests_per_task"):
                cmd = resolve_test_command(self.config, path)
                if cmd:
                    t = await run_tests(cmd, path, self.config["limits"]["test_timeout_seconds"],
                                        self.run_dir / "tests" / f"{task_id}-a{attempt}.txt")
                    self._record_test(task_id, t, 0)
                    if not t.passed:
                        v.ok = False
                        v.errors.append("task tests failed:\n" + t.output[-3000:])
            if not v.ok:
                err = "; ".join(v.errors)
                self.store.update_task(self.run_id, task_id, changed_files=v.changed_files)
                if attempt < max_attempts and not self.stop_event.is_set():
                    self._event(f"attempt {attempt} rejected: {err[:500]} - retrying", "warning", task_id)
                    feedback = err
                    self._cleanup_task_worktree(task_id, keep_branch=False)
                    continue
                self._save_worktree_diff(task_id, attempt, path, None)
                self._set_task(task_id, TaskStatus.FAILED, err, changed_files=v.changed_files)
                self._cleanup_task_worktree(task_id, keep_branch=True)
                return
            if not v.changed_files and spec.allow_no_changes:
                self._set_task(task_id, TaskStatus.COMPLETED, None, changed_files=[])
                self._cleanup_task_worktree(task_id, keep_branch=False)
                return
            message = (f"[{provider}] {task_id}: {spec.title}\n\n{(res.final_text or '').strip()[:1500]}\n\n"
                       f"Orchestrator-Task: {self.run_id}/{task_id}\nOrchestrator-Provider: {provider}\n")
            sha = self.wt.commit_paths(path, v.changed_files, self.redact(message))
            self._save_worktree_diff(task_id, attempt, path, sha)
            self.store.update_task(self.run_id, task_id, commit_sha=sha, changed_files=v.changed_files)
            async with self.integ_lock:
                integrated = self._integrated_files()
                ir = self.integrator.cherry_pick(sha, integrated, task_id)
            for w in ir.overlap_warnings:
                self._event(f"overlap: {w}", "warning", task_id)
            if not ir.ok:
                detail = f"cherry-pick conflict in {ir.conflict_files}" if ir.conflict_files else "integration failed"
                self._set_task(task_id, TaskStatus.CONFLICT, f"{detail}: {ir.error}")
                self._cleanup_task_worktree(task_id, keep_branch=True)
                return
            self._set_task(task_id, TaskStatus.COMPLETED, None, integrated_sha=ir.integrated_sha)
            self._cleanup_task_worktree(task_id, keep_branch=True)
            return
        # never leave a task in a non-terminal state (the scheduler would start it again)
        if not self._task_status(task_id).terminal:
            self._set_task(task_id, TaskStatus.FAILED, f"retry limit reached ({max_attempts} attempts)")

    def _integrated_files(self) -> dict[str, list[str]]:
        return {t["id"]: t["changed_files"] for t in self.store.list_tasks(self.run_id)
                if t["status"] == TaskStatus.COMPLETED.value}

    def _cleanup_task_worktree(self, task_id: str, keep_branch: bool) -> None:
        t = self.store.get_task(self.run_id, task_id)
        if not t or not t["worktree"]:
            return
        try:
            self.wt.remove_worktree(t["worktree"], delete_branch=None if keep_branch else t["branch"])
            self.store.update_task(self.run_id, task_id, worktree=None,
                                   **({} if keep_branch else {"branch": None}))
        except (GitError, GitSafetyError) as e:
            self._event(f"worktree cleanup failed: {e}", "warning", task_id)

    def _save_worktree_diff(self, task_id: str, attempt: int, path: Path, sha: str | None) -> None:
        if not self.config["logging"].get("save_git_diff", True):
            return
        try:
            if sha:
                diff = self.wt.git.run("show", "--no-color", "--format=%H%n%B", sha, cwd=path, check=False).stdout
            else:  # rejected attempt: tracked diff + untracked file list (nothing is staged)
                diff = self.wt.git.run("diff", "--no-color", cwd=path, check=False).stdout
                diff += "\n# status:\n" + self.wt.git.run("status", "--short", "--untracked-files=all",
                                                         cwd=path, check=False).stdout
            out = self.run_dir / "diffs" / f"{task_id}-a{attempt}.diff"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(self.redact(diff), encoding="utf-8")
        except (GitError, OSError):
            pass

    def _record_test(self, scope: str, t: TestOutcome, round_no: int) -> None:
        out_path = None
        if self.config["logging"].get("save_test_results", True) and t.configured:
            p = self.run_dir / "tests" / f"{scope}-r{round_no}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.redact(t.output), encoding="utf-8")
            out_path = str(p)
        self.store.add_test_result(self.run_id, scope, round_no, t.command, t.exit_code, t.passed, t.timed_out,
                                   out_path, t.output[-6000:] if t.configured else (t.skipped_reason or ""),
                                   t.duration)
        if t.configured:
            self._event(f"tests [{scope}] {'PASSED' if t.passed else 'FAILED'} (exit={t.exit_code}, "
                        f"{t.duration:.1f}s)", "info" if t.passed else "error")
        else:
            self._event(f"tests [{scope}] skipped: {t.skipped_reason}", "warning")

    # ---------------------------------------------------------------- verify / repair
    async def _verify_and_repair(self) -> RunStatus:
        run = self._run(self.run_id)
        max_rounds = int(self.config["limits"]["max_repair_rounds"])
        round_no = int(run["repair_round"])
        blocking_sev = self.config["validation"]["blocking_severities"]
        while True:
            self._set_status(RunStatus.VERIFYING)
            integ_path = run["integration_worktree"]
            cmd = resolve_test_command(self.config, integ_path)
            tests = await run_tests(cmd, integ_path, self.config["limits"]["test_timeout_seconds"])
            self._record_test("integration", tests, round_no)
            reviews = await self._review(round_no, tests)
            if self.stop_event.is_set():
                return RunStatus.NEEDS_ATTENTION if self.budget_exceeded else RunStatus.STOPPED
            blocking = blocking_findings(reviews, blocking_sev)
            tasks = self.store.list_tasks(self.run_id)
            conflicts = [t for t in tasks if t["status"] == TaskStatus.CONFLICT.value]
            tests_failed = tests.configured and not tests.passed
            if not tests_failed and not blocking and not conflicts:
                return self._final_status(tests, reviews)
            problems = []
            if tests_failed:
                problems.append("integration tests failed")
            if blocking:
                problems.append(f"{len(blocking)} blocking review finding(s)")
            if conflicts:
                problems.append(f"{len(conflicts)} conflicting task(s)")
            if round_no >= max_rounds:
                msg = f"max repair rounds ({max_rounds}) reached; unresolved: {', '.join(problems)}"
                self._event(msg, "error")
                self.store.update_run(self.run_id, error=msg)
                return RunStatus.NEEDS_ATTENTION
            round_no += 1
            self.store.update_run(self.run_id, repair_round=round_no)
            self._set_status(RunStatus.REPAIRING)
            self._event(f"repair round {round_no}/{max_rounds}: {', '.join(problems)}", "warning")
            if conflicts:
                await self._requeue_conflicts(conflicts, round_no)
                if self.stop_event.is_set():
                    return RunStatus.STOPPED
            if tests_failed or blocking:
                repair_id = self._create_repair_task(round_no, tests if tests_failed else None, blocking)
                await self._run_dag([repair_id])
                if self.stop_event.is_set():
                    return RunStatus.NEEDS_ATTENTION if self.budget_exceeded else RunStatus.STOPPED

    async def _requeue_conflicts(self, conflicts: list[dict], round_no: int) -> None:
        """Re-run conflicting tasks on top of the current integration head (with the conflict as context)."""
        ids = []
        for t in conflicts:
            spec: TaskSpec = t["spec"]
            spec.description += (f"\n\nNOTE (repair round {round_no}): a previous implementation of this task "
                                 f"conflicted with work integrated by other agents ({t['error'][:600]}). "
                                 f"Re-implement the task on top of the current code; keep the other agents' "
                                 f"changes intact.")
            spec.depends_on = []      # it now starts from the integration head that contains everything
            spec.kind = "work"
            self.store.update_task(self.run_id, t["id"], spec=spec, status=TaskStatus.PENDING, attempts=0,
                                   error=None, commit_sha=None, branch=None, repair_round=round_no)
            ids.append(t["id"])
        # dependants that were blocked by the conflict get another chance as well
        for t in self.store.list_tasks(self.run_id):
            if t["status"] == TaskStatus.BLOCKED.value:
                self.store.update_task(self.run_id, t["id"], status=TaskStatus.PENDING, error=None)
                ids.append(t["id"])
        await self._run_dag(ids)

    def _choose_repair_provider(self, files: set[str]) -> str:
        votes: Counter[str] = Counter()
        for t in self.store.list_tasks(self.run_id):
            if set(t["changed_files"]) & files:
                votes[t["provider"]] += 1
        avail = self.available_providers()
        for p, _ in votes.most_common():
            if p in avail:
                return p
        default = self.config["agents"].get("repair_default", "claude")
        return default if default in avail else (avail[0] if avail else default)

    def _create_repair_task(self, round_no: int, tests: TestOutcome | None,
                            blocking: list[tuple[str, object]]) -> str:
        files: set[str] = set()
        known = {f for t in self.store.list_tasks(self.run_id) for f in t["changed_files"]}
        failure_text = ""
        if tests is not None:
            failure_text = f"Command: {' '.join(tests.command or [])}\nExit code: {tests.exit_code}\n" \
                           f"Output (tail):\n```\n{self.redact(tests.output[-6000:])}\n```"
            for f in known:
                if f and re.search(re.escape(f), tests.output):
                    files.add(f)
        findings_text = ""
        for reviewer, f in blocking:
            files.add(f.file)
            findings_text += (f"- [{f.severity}] ({reviewer}) {f.file}"
                              f"{(':' + str(f.line)) if f.line else ''} - {f.title}. Fix: {f.recommendation}\n")
        provider = self._choose_repair_provider(files)
        tid = f"repair-r{round_no}"
        desc = ("Fix the problems found while verifying the integrated result of all tasks."
                + render("repair.md", round=round_no, test_failures=failure_text or "(tests passed)",
                         findings=findings_text or "(none)"))
        spec = TaskSpec(id=tid, title=f"Repair round {round_no}", description=desc, provider=provider,
                        role="repair", files=["**"], kind="repair",
                        acceptance_criteria=["All automated tests pass", "Blocking review findings are resolved"],
                        verification="Integration test command passes")
        self.store.add_task(self.run_id, spec, repair_round=round_no)
        self._event(f"repair task {tid} created and assigned to {provider} (files: {sorted(files) or 'unknown'})",
                    "warning")
        return tid

    async def _review(self, round_no: int, tests: TestOutcome) -> list[ReviewResult]:
        if not self.config["review"].get("enabled", True):
            return []
        run = self._run(self.run_id)
        reviewers = [r for r in self.config["agents"]["reviewers"] if r in self.adapters]
        usable: list[str] = []
        unavailable: list[ReviewResult] = []
        for r in reviewers:
            ok, msg = self.available(r)
            if ok:
                usable.append(r)
            else:
                unavailable.append(ReviewResult(r, "", "error", error=f"reviewer unavailable: {msg}"))
                self.store.add_review(self.run_id, round_no, r, "", "error", "", [], error=f"unavailable: {msg}")
                self._event(f"reviewer {r} unavailable: {msg}", "warning")
        if not usable or not self._budget_ok():
            return unavailable
        head = self.integrator.head()
        if head == run["base_commit"]:
            self._event("nothing integrated - skipping review", "warning")
            return []
        authors: dict[str, list[str]] = {}
        for t in self.store.list_tasks(self.run_id):
            if t["status"] == TaskStatus.COMPLETED.value and t["changed_files"]:
                authors.setdefault(t["provider"], []).append(t["id"])
        diff = self.integrator.diff_since(run["base_commit"])
        limit = int(self.config["limits"]["max_diff_chars_for_review"])
        if len(diff) > limit:
            diff = diff[:limit] + f"\n... [diff truncated at {limit} chars - read the files for details]"
        if self.config["logging"].get("save_git_diff", True):
            p = self.run_dir / "diffs" / f"integration-r{round_no}.diff"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.redact(self.integrator.diff_since(run["base_commit"])), encoding="utf-8")
        plan = self._plan()
        tasks_text = "\n".join(f"- {t.id} ({t.provider}): {t.title}" for t in plan.tasks)
        if tests.configured:
            test_summary = f"{'PASSED' if tests.passed else 'FAILED'} (exit {tests.exit_code})\n" \
                           f"```\n{tests.output[-2500:]}\n```"
        else:
            test_summary = "no test command configured"
        self._event(f"cross review round {round_no} by {', '.join(usable)} at {head[:10]}")
        results = await run_reviews(
            runner=self.runner, adapters=self.adapters, reviewers=usable, run_id=self.run_id, round_no=round_no,
            request=run["request"], tasks_text=tasks_text, authors=authors, cwd=run["integration_worktree"],
            base_commit=run["base_commit"], target_commit=head, diff_stat=self.integrator.diff_since(
                run["base_commit"], stat=True), diff=self.redact(diff), test_summary=self.redact(test_summary),
            max_parallel=self.config["limits"]["max_parallel_reviewers"],
            timeout=self.config["limits"]["review_timeout_seconds"],
            artifacts_dir=str(self.run_dir / "artifacts" / f"review-r{round_no}"), cancel_event=self.stop_event)
        for r in results:
            self.store.add_review(self.run_id, round_no, r.reviewer, r.target_commit, r.verdict, r.summary,
                                  [f.to_dict() for f in r.findings], r.error)
            sev = Counter(f.severity for f in r.findings)
            self._event(f"review by {r.reviewer}: {r.verdict} "
                        f"{dict(sev) if sev else ''}{(' error=' + r.error) if r.error else ''}",
                        "warning" if r.verdict != "approve" else "info")
        return results + unavailable

    def _final_status(self, tests: TestOutcome, reviews: list[ReviewResult]) -> RunStatus:
        tasks = self.store.list_tasks(self.run_id)
        unfinished = [t["id"] for t in tasks if t["status"] != TaskStatus.COMPLETED.value]
        if unfinished:
            self._event(f"tasks not completed: {', '.join(unfinished)}", "error")
            return RunStatus.NEEDS_ATTENTION
        review_errors = [r for r in reviews if r.verdict == "error"]
        if self.config["review"].get("enabled") and self.config["review"].get("required"):
            if not reviews or review_errors:
                self._event("required cross review incomplete", "error")
                return RunStatus.NEEDS_ATTENTION
        if not tests.configured:
            return RunStatus.UNVERIFIED
        return RunStatus.SUCCEEDED

    # ---------------------------------------------------------------- finish
    def _finish(self, status: RunStatus) -> str:
        run = self._run(self.run_id)
        # main / base branch must never move because of the orchestrator
        if run["base_ref"] and run["base_ref"] != "HEAD":
            try:
                now = self.wt.branch_sha(run["base_ref"])
                if now and now != run["base_commit"]:
                    self._event(f"NOTE: base branch {run['base_ref']} moved from {run['base_commit'][:10]} to "
                                f"{now[:10]} during the run (not by the orchestrator)", "warning")
            except (GitError, AttributeError):
                pass
        if status not in (RunStatus.PLANNED,):
            # task worktrees are removed; branches are kept for inspection
            for t in self.store.list_tasks(self.run_id):
                if t["worktree"]:
                    self._cleanup_task_worktree(t["id"], keep_branch=True)
        if status not in (RunStatus.PLANNED, RunStatus.STOPPED) and run["integration_worktree"]:
            # the integration branch stays; its worktree is re-created on resume if needed
            try:
                self.wt.remove_worktree(run["integration_worktree"])
            except (GitError, GitSafetyError, AttributeError) as e:
                self._event(f"integration worktree cleanup failed: {e}", "warning")
        self.store.update_run(self.run_id, status=status.value,
                              finished_at=now_iso() if status not in (RunStatus.PLANNED, RunStatus.STOPPED) else None)
        self._event(f"run finished with status {status.value}", "info" if status == RunStatus.SUCCEEDED else "warning")
        try:
            from .report import write_report
            path = write_report(self.store, self.run_id, self.run_dir)
            self._event(f"report written to {path}")
        except Exception as e:  # pragma: no cover
            self._event(f"report generation failed: {e}", "error")
        return status.value
