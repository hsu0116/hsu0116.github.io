"""Runs the project's test command (argument array, never a shell string) in a worktree."""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .procutil import child_env, kill_process_tree_async, popen_kwargs, resolve_command


@dataclass
class TestOutcome:
    command: list[str] | None
    passed: bool
    exit_code: int | None
    timed_out: bool
    output: str
    duration: float
    skipped_reason: str | None = None

    @property
    def configured(self) -> bool:
        return self.command is not None


def detect_test_command(worktree: str | Path) -> list[str] | None:
    wt = Path(worktree)
    pkg = wt / "package.json"
    if pkg.exists():
        try:
            scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {})
        except (json.JSONDecodeError, OSError):
            scripts = {}
        test = scripts.get("test", "")
        if test and "no test specified" not in test:
            return ["npm", "test", "--silent"]
    py_markers = ["pytest.ini", "conftest.py", "tests", "test"]
    has_py_tests = any((wt / m).exists() for m in py_markers) or any(wt.glob("test_*.py"))
    pyproject = wt / "pyproject.toml"
    if pyproject.exists() and "pytest" in pyproject.read_text(encoding="utf-8", errors="ignore"):
        has_py_tests = True
    if has_py_tests:
        return [sys.executable, "-m", "pytest", "-q"]
    return None


def resolve_test_command(config: dict, worktree: str | Path) -> list[str] | None:
    tc = config["validation"].get("test_command", "auto")
    if tc is None:
        return None
    cmd = detect_test_command(worktree) if tc == "auto" else list(tc)
    if not cmd:
        return None
    if cmd[0] in ("python", "python3", "py"):
        cmd[0] = sys.executable
    elif shutil.which(cmd[0]):
        cmd = resolve_command(cmd, cmd[0])
    return cmd


async def run_tests(cmd: list[str] | None, cwd: str | Path, timeout: float, output_path: Path | None = None) -> TestOutcome:
    if not cmd:
        return TestOutcome(None, False, None, False, "", 0.0, skipped_reason="no test command configured/detected")
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL,
                                                    env=child_env({"PYTHONDONTWRITEBYTECODE": "1"}),
                                                    **popen_kwargs())
    except OSError as e:
        return TestOutcome(cmd, False, None, False, f"failed to start test command: {e}", 0.0)
    timed_out = False
    try:
        out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        await kill_process_tree_async(proc)
        out_b = b""
    except asyncio.CancelledError:
        await kill_process_tree_async(proc)
        raise
    output = out_b.decode("utf-8", errors="replace")
    if timed_out:
        output += f"\n[orchestrator] test command timed out after {timeout}s"
    duration = time.monotonic() - started
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    passed = (proc.returncode == 0) and not timed_out
    return TestOutcome(cmd, passed, proc.returncode, timed_out, output, duration)
