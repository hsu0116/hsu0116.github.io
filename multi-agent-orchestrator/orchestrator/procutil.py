"""Cross-platform subprocess helpers (Windows + POSIX).

* Commands are always argument lists (no shell string concatenation).
* On Windows, npm ``.cmd`` shims are resolved to ``node <script.js>`` so that no
  argument is ever re-parsed by ``cmd.exe``.
* Whole process trees are terminated (``taskkill /T`` on Windows, process groups on POSIX).
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"


class CommandNotFound(RuntimeError):
    pass


def _resolve_cmd_shim(shim: Path) -> list[str] | None:
    """Turn an npm-generated .cmd shim into [node, script.js]."""
    try:
        text = shim.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r'%(?:~?dp0%?|dp0%)\\?([^"%]*?node_modules[\\/][^"]+?\.(?:js|cjs|mjs))', text, re.I)
    if not m:
        m = re.search(r'(node_modules[\\/][^"]+?\.(?:js|cjs|mjs))', text, re.I)
    if not m:
        return None
    script = (shim.parent / m.group(1).replace("\\", "/").lstrip("/")).resolve()
    if not script.exists():
        return None
    local_node = shim.parent / "node.exe"
    node = str(local_node) if local_node.exists() else shutil.which("node")
    if not node:
        return None
    return [node, str(script)]


def resolve_command(spec: list[str] | str | None, default_name: str) -> list[str]:
    """Return the argv prefix used to start a CLI."""
    if spec:
        argv = [spec] if isinstance(spec, str) else list(spec)
        first = shutil.which(argv[0]) or argv[0]
        argv[0] = first
    else:
        found = shutil.which(default_name)
        if not found:
            raise CommandNotFound(f"'{default_name}' was not found on PATH")
        argv = [found]
    exe = Path(argv[0])
    if IS_WINDOWS and exe.suffix.lower() in (".cmd", ".bat"):
        resolved = _resolve_cmd_shim(exe)
        if resolved is None:
            raise CommandNotFound(
                f"{exe} is a batch shim that could not be resolved to a node script. "
                f"Set providers.<name>.command to an .exe or to [node, path/to/cli.js] in config.yaml")
        argv = resolved + argv[1:]
    return argv


def popen_kwargs() -> dict:
    """Arguments that place the child in its own process group."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_process_tree(pid: int) -> None:
    """Terminate a process and all of its descendants. Never raises."""
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                return
    except Exception:  # pragma: no cover - best effort
        pass


async def kill_process_tree_async(proc: asyncio.subprocess.Process, grace: float = 3.0) -> None:
    if proc.returncode is not None:
        return
    kill_process_tree(proc.pid)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
        return
    except asyncio.TimeoutError:
        pass
    if not IS_WINDOWS:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except asyncio.TimeoutError:  # pragma: no cover
        pass


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if IS_WINDOWS:
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                                 text=True, check=False).stdout
            return str(pid) in out
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a terminated but not yet reaped process (zombie) is not alive
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            state = f.read().rsplit(b")", 1)[1].split()[0]
        return state not in (b"Z", b"X")
    except (OSError, IndexError):
        return True


def child_env(extra: dict[str, str] | None = None, remove: tuple[str, ...] | list[str] = ()) -> dict[str, str]:
    env = dict(os.environ)
    blocked = {r.upper() for r in remove}
    for k in list(env):
        if k.upper() in blocked:   # case-insensitive: Windows env names are case-insensitive
            del env[k]
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["NO_COLOR"] = "1"
    # Never let a child inherit an interactive git credential prompt.
    env["GIT_TERMINAL_PROMPT"] = "0"
    if extra:
        env.update(extra)
    return env


def python_executable() -> str:
    return sys.executable
