"""Stage 1: environment check (`orchestrator doctor`)."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys

from .agents import PROVIDERS, make_adapter
from .procutil import child_env

INSTALL_HINTS = {
    "git": "Windows: winget install --id Git.Git -e   (then: git config --global core.longpaths true)",
    "claude": "Windows (PowerShell): irm https://claude.ai/install.ps1 | iex   or: npm install -g @anthropic-ai/claude-code\n"
              "      then run 'claude' once and log in (authentication must be done by you)",
    "codex": "npm install -g @openai/codex   then: codex login   (authentication must be done by you)",
    "node": "Windows: winget install OpenJS.NodeJS.LTS   (needed for npm-installed CLIs)",
}


def _version(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=child_env(), check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout or r.stderr).strip().splitlines()[0] if r.returncode == 0 else None


def run_doctor(config: dict) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    py_ok = sys.version_info >= (3, 11)
    ok &= py_ok
    lines.append(f"[{'OK' if py_ok else 'FAIL'}] Python {sys.version.split()[0]} (>= 3.11 required)")
    git_v = _version(["git", "--version"]) if shutil.which("git") else None
    ok &= git_v is not None
    lines.append(f"[{'OK' if git_v else 'FAIL'}] {git_v or 'git not found'}")
    if not git_v:
        lines.append(f"      install: {INSTALL_HINTS['git']}")
    else:
        r = subprocess.run(["git", "help", "-a"], capture_output=True, text=True, check=False)
        has_wt = "worktree" in r.stdout
        lines.append(f"[{'OK' if has_wt else 'FAIL'}] git worktree {'available' if has_wt else 'missing'}")
        ok &= has_wt
    node_v = _version(["node", "--version"]) if shutil.which("node") else None
    lines.append(f"[{'OK' if node_v else 'WARN'}] node {node_v or 'not found'}")
    any_provider = False
    for p in PROVIDERS:
        adapter = make_adapter(p, config)
        avail, msg = adapter.availability()
        any_provider |= avail
        lines.append(f"[{'OK' if avail else 'WARN'}] {p}: {msg}")
        if not avail:
            lines.append(f"      install/auth: {INSTALL_HINTS[p]}")
    if not any_provider:
        ok = False
        lines.append("[FAIL] no agent CLI is usable")
    if platform.system() == "Windows":
        lines.append("Tip: enable long paths: git config --global core.longpaths true")
    return ok, lines
