"""Claude Code CLI adapter (``claude -p``), verified against Claude Code 2.1.x.

Permission model (verified by experiment, see README "Security model"):
  --restricted            ignores user/project settings, removes code-running tools unless named in
                          --tools, and confines file tools to the working directory
  --tools ...             the only built-in tools that exist in the session
  --permission-mode dontAsk + --allowedTools ...   anything not pre-approved is denied (no prompt)
The prompt is sent on stdin, never on the command line.
"""
from __future__ import annotations

import json
import subprocess

from ..models import AgentResult
from ..procutil import CommandNotFound, child_env, resolve_command
from .base import PLAN, REVIEW, AgentAdapter, Invocation, extract_json


class ClaudeAdapter(AgentAdapter):
    name = "claude"

    def base_argv(self) -> list[str]:
        if self._base_argv is None:
            self._base_argv = resolve_command(self.cfg.get("command"), "claude")
        return list(self._base_argv)

    def availability(self) -> tuple[bool, str]:
        try:
            argv = self.base_argv()
        except CommandNotFound as e:
            return False, str(e)
        try:
            r = subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=60,
                               env=child_env(), check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"cannot execute claude: {e}"
        if r.returncode != 0:
            return False, (r.stderr or r.stdout).strip()[:300]
        return True, r.stdout.strip()

    def build(self, *, purpose: str, prompt: str, cwd: str, artifacts_dir: str,
              model: str | None = None, schema: dict | None = None) -> Invocation:
        cfg = self.cfg
        argv = self.base_argv() + [
            "-p",
            "--output-format", "stream-json", "--verbose",
            "--restricted",
            "--permission-mode", "dontAsk",
            "--no-session-persistence",
        ]
        if purpose in (PLAN, REVIEW):
            tools = cfg.get("reviewer_tools") or ["Read", "Glob", "Grep"]
            allowed = cfg.get("reviewer_allowed_tools") or ["Read", "Glob", "Grep"]
        else:
            tools = cfg.get("worker_tools") or ["Read", "Write", "Edit", "Glob", "Grep"]
            allowed = cfg.get("worker_allowed_tools") or ["Read", "Write", "Edit", "Glob", "Grep"]
        argv += ["--tools", ",".join(tools)]
        argv += ["--allowedTools", *allowed]
        disallowed = cfg.get("disallowed_tools") or []
        if disallowed:
            argv += ["--disallowedTools", *disallowed]
        m = self.model_for(purpose, model)
        if m:
            argv += ["--model", m]
        budget = cfg.get("max_budget_usd_per_call")
        if budget:
            argv += ["--max-budget-usd", str(budget)]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        argv += list(cfg.get("extra_args") or [])
        return Invocation(provider=self.name, purpose=purpose, argv=argv, cwd=cwd, stdin_text=prompt,
                          model=m, artifacts_dir=artifacts_dir)

    @staticmethod
    def log_line(line: str) -> str | None:
        """Condense stream-json into a readable log line (None = skip)."""
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line
        t = ev.get("type")
        if t in ("stream_event", "rate_limit_event", "active_goal", "autocompact_state"):
            return None
        if t == "assistant":
            parts = []
            for c in (ev.get("message") or {}).get("content") or []:
                if c.get("type") == "text" and c.get("text"):
                    parts.append(c["text"].strip())
                elif c.get("type") == "tool_use":
                    inp = json.dumps(c.get("input"), ensure_ascii=False)
                    parts.append(f"[tool] {c.get('name')} {inp[:300]}")
            return "\n".join(parts) if parts else None
        if t == "user":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                    return f"[tool error] {str(c.get('content'))[:300]}"
            return None
        if t == "result":
            return (f"[result] subtype={ev.get('subtype')} is_error={ev.get('is_error')} "
                    f"cost_usd={ev.get('total_cost_usd')} turns={ev.get('num_turns')}")
        if t == "system" and ev.get("subtype") == "init":
            return f"[init] model={ev.get('model')} cwd={ev.get('cwd')}"
        return None

    def parse(self, inv: Invocation, exit_code: int | None, stdout: str, stderr: str) -> AgentResult:
        result_ev = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if ev.get("type") == "result":
                result_ev = ev
        res = AgentResult(provider=self.name, exit_code=exit_code)
        if result_ev is None:
            res.error = f"no result event from claude (exit={exit_code}): {stderr.strip()[-500:]}"
            return res
        res.final_text = result_ev.get("result") or ""
        res.cost_usd = result_ev.get("total_cost_usd")
        res.usage = result_ev.get("usage") or {}
        res.permission_denials = result_ev.get("permission_denials") or []
        res.structured = result_ev.get("structured_output")
        if res.structured is None and inv.purpose in (PLAN, REVIEW):
            res.structured = extract_json(res.final_text)
        if result_ev.get("is_error") or result_ev.get("subtype") not in (None, "success"):
            res.error = f"claude reported {result_ev.get('subtype')}: {res.final_text[:500]}"
        return res
