"""OpenAI Codex CLI adapter (``codex exec``), built against codex-cli 0.169.

* workers run with ``--sandbox workspace-write`` (writes limited to the worktree, no network by default)
* planner / reviewers run with ``--sandbox read-only``
* approvals are never requested (``-c approval_policy="never"``): anything outside the sandbox fails
* the prompt is sent on stdin (``-``); the final message is written with ``-o`` and, for structured
  purposes, constrained with ``--output-schema``
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..models import AgentResult
from ..procutil import CommandNotFound, child_env, resolve_command
from ..subscription import ALL_BLOCKED_ENV, CODEX_FORCED_LOGIN_ARGS, check_codex_login_status
from .base import PLAN, REVIEW, AgentAdapter, Invocation, extract_json


class CodexAdapter(AgentAdapter):
    name = "codex"
    blocked_env = ALL_BLOCKED_ENV

    def base_argv(self) -> list[str]:
        if self._base_argv is None:
            self._base_argv = resolve_command(self.cfg.get("command"), "codex")
        return list(self._base_argv)

    def availability(self) -> tuple[bool, str]:
        try:
            argv = self.base_argv()
        except CommandNotFound as e:
            return False, str(e)
        try:
            env = child_env(remove=self.blocked_env)
            ver = subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=60,
                                 env=env, check=False)
            if ver.returncode != 0:
                return False, (ver.stderr or ver.stdout).strip()[:300]
            st = subprocess.run(argv + CODEX_FORCED_LOGIN_ARGS + ["login", "status"], capture_output=True,
                                text=True, timeout=60, env=env, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"cannot execute codex: {e}"
        ok, msg = check_codex_login_status(st.stdout + st.stderr)
        if st.returncode != 0 and ok:
            ok, msg = False, f"'codex login status' failed: {(st.stdout + st.stderr).strip()[:200]}"
        return ok, f"{ver.stdout.strip()} - {msg}"

    def build(self, *, purpose: str, prompt: str, cwd: str, artifacts_dir: str,
              model: str | None = None, schema: dict | None = None) -> Invocation:
        cfg = self.cfg
        art = Path(artifacts_dir)
        art.mkdir(parents=True, exist_ok=True)
        last_msg = art / f"codex-{purpose}-last-message.txt"
        sandbox = cfg.get("reviewer_sandbox", "read-only") if purpose in (PLAN, REVIEW) \
            else cfg.get("worker_sandbox", "workspace-write")
        argv = self.base_argv() + [
            "exec",
            "--json",
            "--color", "never",
            "--sandbox", sandbox,
            "--cd", cwd,
            "--ephemeral",
            "-c", 'approval_policy="never"',
            *CODEX_FORCED_LOGIN_ARGS,          # subscription (ChatGPT) login only, never an API key
            "-o", str(last_msg),
        ]
        m = self.model_for(purpose, model)
        if m:
            argv += ["--model", m]
        if schema is not None:
            argv += ["--output-schema", self.write_schema(artifacts_dir, schema, f"codex-{purpose}-schema.json")]
        argv += list(cfg.get("extra_args") or [])
        argv.append("-")  # read prompt from stdin
        return Invocation(provider=self.name, purpose=purpose, argv=argv, cwd=cwd, stdin_text=prompt,
                          model=m, artifacts_dir=artifacts_dir, extra={"last_message": str(last_msg)},
                          env_remove=self.blocked_env)

    @staticmethod
    def log_line(line: str) -> str | None:
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line
        t = ev.get("type", "")
        if t in ("item.completed", "item.started"):
            item = ev.get("item") or {}
            it = item.get("type") or item.get("item_type")
            if it in ("agent_message", "reasoning"):
                return f"[{it}] {str(item.get('text', ''))[:2000]}" if t == "item.completed" else None
            if it == "command_execution":
                return f"[cmd:{t.split('.')[1]}] {item.get('command')} exit={item.get('exit_code')}"
            if it == "file_change":
                return f"[file_change] {json.dumps(item.get('changes'), ensure_ascii=False)[:500]}"
            return f"[{it}]" if t == "item.completed" else None
        if t in ("turn.completed", "turn.failed", "error", "thread.started"):
            return f"[{t}] {json.dumps({k: v for k, v in ev.items() if k != 'type'}, ensure_ascii=False)[:500]}"
        return None

    def parse(self, inv: Invocation, exit_code: int | None, stdout: str, stderr: str) -> AgentResult:
        res = AgentResult(provider=self.name, exit_code=exit_code)
        usage: dict = {}
        errors: list[str] = []
        last_agent_message = ""
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            t = ev.get("type")
            if t == "turn.completed" and isinstance(ev.get("usage"), dict):
                for k, v in ev["usage"].items():
                    if isinstance(v, (int, float)):
                        usage[k] = usage.get(k, 0) + v
            elif t == "turn.failed":
                errors.append(str((ev.get("error") or {}).get("message") or ev))
            elif t == "error":
                errors.append(str(ev.get("message") or ev))
            elif t == "item.completed":
                item = ev.get("item") or {}
                if (item.get("type") or item.get("item_type")) == "agent_message":
                    last_agent_message = item.get("text") or last_agent_message
        lm_path = Path(inv.extra.get("last_message", ""))
        if lm_path.is_file():
            res.final_text = lm_path.read_text(encoding="utf-8", errors="replace")
        else:
            res.final_text = last_agent_message
        res.usage = usage
        if inv.purpose in (PLAN, REVIEW):
            res.structured = extract_json(res.final_text)
        if exit_code != 0 or errors:
            res.error = "; ".join(errors)[:1000] or f"codex exited with {exit_code}: {stderr.strip()[-500:]}"
        return res
