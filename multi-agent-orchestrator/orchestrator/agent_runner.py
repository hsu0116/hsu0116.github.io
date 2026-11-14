"""Runs one agent CLI invocation as an isolated subprocess.

Handles: argument-array execution (no shell), prompt on stdin, real-time redacted logging,
per-invocation timeout, cancellation (user stop / per-task cancel), and process-tree cleanup.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from .agents.base import AgentAdapter, Invocation
from .models import AgentResult
from .procutil import child_env, kill_process_tree_async, popen_kwargs
from .state_store import StateStore

STREAM_LIMIT = 32 * 1024 * 1024


class AgentRunner:
    def __init__(self, store: StateStore, redactor, logs_dir: str | Path, save_output: bool = True,
                 on_policy_violation=None):
        self.on_policy_violation = on_policy_violation
        self.store = store
        self.redact = redactor
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.save_output = save_output

    async def run(self, adapter: AgentAdapter, inv: Invocation, *, run_id: str, task_id: str | None,
                  timeout: float, cancel_event: asyncio.Event | None = None,
                  log_name: str | None = None) -> AgentResult:
        name = log_name or f"{task_id or inv.purpose}-{inv.provider}"
        agent_run_id = self.store.start_agent_run(run_id, task_id, inv.purpose, inv.provider, inv.model,
                                                  inv.argv, inv.cwd, "")
        log_path = self.logs_dir / f"{name}-{agent_run_id}.log"
        raw_path = self.logs_dir / f"{name}-{agent_run_id}.raw.jsonl"
        self.store._exec("UPDATE agent_runs SET log_path=? WHERE id=?", (str(log_path), agent_run_id))
        self.store.log_event(run_id, f"{inv.provider} {inv.purpose} started (log: {log_path.name})",
                             task_id=task_id)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        timed_out = cancelled = False
        started = time.monotonic()
        log_f = log_path.open("w", encoding="utf-8") if self.save_output else None
        raw_f = raw_path.open("w", encoding="utf-8") if self.save_output else None
        log_fn = getattr(adapter, "log_line", None)

        def write_log(text: str) -> None:
            if log_f is not None and text:
                log_f.write(self.redact(text) + "\n")
                log_f.flush()

        write_log(f"# {inv.provider} {inv.purpose} cwd={inv.cwd}")
        write_log("# argv: " + " ".join(self.redact(a)[:200] for a in inv.argv))

        try:
            proc = await asyncio.create_subprocess_exec(
                *inv.argv, cwd=inv.cwd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=child_env(remove=inv.env_remove), limit=STREAM_LIMIT,
                **popen_kwargs())
        except OSError as e:
            res = AgentResult(provider=inv.provider, exit_code=None, error=f"failed to start: {e}",
                              log_path=str(log_path))
            write_log(f"[orchestrator] failed to start process: {e}")
            if log_f:
                log_f.close()
            if raw_f:
                raw_f.close()
            self.store.finish_agent_run(agent_run_id, exit_code=None, error=res.error)
            return res

        self.store.set_agent_pid(agent_run_id, proc.pid)

        async def feed_stdin() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(inv.stdin_text.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        violation: list[str] = []
        abort_event = asyncio.Event()

        async def read_stream(stream, sink: list[str], is_err: bool) -> None:
            while True:
                line_b = await stream.readline()
                if not line_b:
                    break
                line = line_b.decode("utf-8", errors="replace").rstrip("\r\n")
                sink.append(line)
                if not is_err and not violation:
                    v = adapter.guard_line(line)
                    if v:
                        violation.append(v)
                        abort_event.set()
                if raw_f is not None and not is_err:
                    raw_f.write(self.redact(line) + "\n")
                if is_err:
                    write_log(f"[stderr] {line}")
                else:
                    write_log(log_fn(line) if log_fn else line)

        readers = [asyncio.create_task(feed_stdin()),
                   asyncio.create_task(read_stream(proc.stdout, stdout_lines, False)),
                   asyncio.create_task(read_stream(proc.stderr, stderr_lines, True))]
        waiter = asyncio.create_task(proc.wait())
        cancel_waiter = asyncio.create_task(cancel_event.wait()) if cancel_event else None
        abort_waiter = asyncio.create_task(abort_event.wait())
        try:
            pending = {waiter, abort_waiter} | ({cancel_waiter} if cancel_waiter else set())
            done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if abort_waiter in done:
                write_log(f"[orchestrator] {violation[0]}")
                await kill_process_tree_async(proc)
            elif waiter not in done:
                if cancel_waiter is not None and cancel_waiter in done:
                    cancelled = True
                    write_log("[orchestrator] cancellation requested - terminating process tree")
                else:
                    timed_out = True
                    write_log(f"[orchestrator] timeout after {timeout}s - terminating process tree")
                await kill_process_tree_async(proc)
        except asyncio.CancelledError:
            cancelled = True
            write_log("[orchestrator] orchestrator cancelled - terminating process tree")
            await kill_process_tree_async(proc)
            raise
        finally:
            abort_waiter.cancel()
            if cancel_waiter is not None:
                cancel_waiter.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(*readers, return_exceptions=True), timeout=10)
            except asyncio.TimeoutError:
                for r in readers:
                    r.cancel()
            if not waiter.done():
                waiter.cancel()
            if cancelled and proc.returncode is None:
                await kill_process_tree_async(proc)
            duration = time.monotonic() - started
            if log_f:
                log_f.write(f"# exit={proc.returncode} duration={duration:.1f}s\n")
                log_f.close()
            if raw_f:
                raw_f.close()
            if cancelled:
                self.store.finish_agent_run(agent_run_id, exit_code=proc.returncode, cancelled=1,
                                            error="cancelled")

        exit_code = proc.returncode
        stdout = "\n".join(stdout_lines)
        stderr = "\n".join(stderr_lines)
        res = adapter.parse(inv, exit_code, stdout, stderr)
        res.timed_out = timed_out
        res.cancelled = cancelled
        res.log_path = str(log_path)
        if violation:
            res.error = violation[0]
            res.exit_code = res.exit_code if res.exit_code not in (0, None) else -1
            self.store.log_event(run_id, violation[0], level="error", task_id=task_id)
            if self.on_policy_violation:
                self.on_policy_violation(violation[0])
        elif timed_out:
            res.error = f"timed out after {timeout:.0f}s"
        elif cancelled:
            res.error = "cancelled"
        if res.error:
            res.error = self.redact(res.error)
        if not cancelled:
            self.store.finish_agent_run(
                agent_run_id, exit_code=exit_code, timed_out=int(timed_out), cancelled=int(cancelled),
                cost_usd=res.cost_usd, usage_json=res.usage or None, error=res.error)
        level = "info" if res.ok else "error"
        cost = f" cost=${res.cost_usd:.4f}" if res.cost_usd is not None else ""
        self.store.log_event(run_id, f"{inv.provider} {inv.purpose} finished exit={exit_code}"
                             f"{' TIMEOUT' if timed_out else ''}{' CANCELLED' if cancelled else ''}{cost}"
                             f"{(' error=' + res.error[:300]) if res.error else ''}", level=level, task_id=task_id)
        if res.permission_denials:
            self.store.log_event(run_id, f"{len(res.permission_denials)} tool call(s) denied by permission policy: "
                                 + ", ".join(str(d.get('tool_name')) for d in res.permission_denials[:10]),
                                 level="warning", task_id=task_id)
        return res
