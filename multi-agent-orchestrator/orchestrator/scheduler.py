"""DAG scheduler: runs independent tasks in parallel (bounded), starts a task only once all of its
dependencies are COMPLETED, and marks dependants of failed tasks as BLOCKED."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from .models import TERMINAL_TASK_STATES, TaskStatus

GetStatus = Callable[[str], TaskStatus]
SetStatus = Callable[[str, TaskStatus, str | None], None]
RunTask = Callable[[str], Awaitable[None]]
IsCancelRequested = Callable[[str], bool]


class DagScheduler:
    def __init__(self, deps: dict[str, list[str]], get_status: GetStatus, set_status: SetStatus,
                 run_task: RunTask, max_parallel: int, stop_event: asyncio.Event,
                 cancel_requested: IsCancelRequested | None = None, on_log: Callable[[str], None] | None = None):
        self.deps = deps
        self.get_status = get_status
        self.set_status = set_status
        self.run_task = run_task
        self.max_parallel = max_parallel
        self.stop_event = stop_event
        self.cancel_requested = cancel_requested or (lambda _tid: False)
        self.log = on_log or (lambda _m: None)
        self.max_observed_parallel = 0

    def _refresh(self) -> list[str]:
        """Update PENDING tasks to READY / BLOCKED. Returns READY task ids in plan order."""
        ready = []
        for tid, deps in self.deps.items():
            st = self.get_status(tid)
            if st in (TaskStatus.PENDING, TaskStatus.READY) and self.cancel_requested(tid):
                self.set_status(tid, TaskStatus.CANCELLED, "cancelled by user before start")
                continue
            if st == TaskStatus.PENDING:
                dep_states = [self.get_status(d) for d in deps]
                bad = [d for d, s in zip(deps, dep_states) if s in TERMINAL_TASK_STATES and s != TaskStatus.COMPLETED]
                if bad:
                    self.set_status(tid, TaskStatus.BLOCKED, f"dependency not completed: {', '.join(bad)}")
                    continue
                if all(s == TaskStatus.COMPLETED for s in dep_states):
                    self.set_status(tid, TaskStatus.READY, None)
                    st = TaskStatus.READY
            if st == TaskStatus.READY:
                ready.append(tid)
        return ready

    async def run(self) -> None:
        running: dict[str, asyncio.Task] = {}
        while True:
            # a task can be blocked only after its dependency reached a terminal state; loop until stable
            ready = [] if self.stop_event.is_set() else self._refresh()
            while ready and len(running) < self.max_parallel and not self.stop_event.is_set():
                tid = ready.pop(0)
                if tid in running:
                    continue
                self.log(f"starting task {tid}")
                running[tid] = asyncio.create_task(self.run_task(tid), name=f"task-{tid}")
                self.max_observed_parallel = max(self.max_observed_parallel, len(running))
            if not running:
                if self.stop_event.is_set() or not self._refresh():
                    return
                continue
            stop_wait = asyncio.create_task(self.stop_event.wait())
            done, _ = await asyncio.wait(set(running.values()) | {stop_wait}, return_when=asyncio.FIRST_COMPLETED)
            if stop_wait not in done:
                stop_wait.cancel()
            for tid, t in list(running.items()):
                if t.done():
                    running.pop(tid)
                    exc = t.exception() if not t.cancelled() else None
                    if exc is not None:
                        self.log(f"task {tid} crashed: {exc!r}")
                        if not self.get_status(tid).terminal:
                            self.set_status(tid, TaskStatus.FAILED, f"internal error: {exc!r}")
            if self.stop_event.is_set() and running:
                # tasks observe the stop event themselves and terminate their agent process trees
                await asyncio.gather(*running.values(), return_exceptions=True)
                running.clear()
