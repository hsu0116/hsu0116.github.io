"""Scripted stand-in for the ``claude`` and ``codex`` CLIs, used by the test-suite and for dry runs.

Usage (configured as ``providers.<name>.command``)::

    [python, -m, orchestrator.agents.fake_agent, claude]   # then the real claude arguments follow
    [python, -m, orchestrator.agents.fake_agent, codex]    # then the real codex arguments follow

It receives exactly the arguments the real adapters build, reads the prompt from stdin and
prints output in the same format as the real CLI (stream-json for claude, JSONL + ``-o`` file for
codex), so the adapters' parsing code is exercised. Behaviour is driven by a JSON scenario file
named in ``ORCH_FAKE_SCENARIO``; activity is appended to ``ORCH_FAKE_LOG`` (JSONL).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

MARKER = re.compile(r"<!-- orchestrator: (.*?) -->")


def _meta(prompt: str) -> dict:
    m = MARKER.search(prompt)
    meta = {}
    if m:
        for kv in m.group(1).split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                meta[k] = v
    return meta


def _arg_value(argv: list[str], *names: str) -> str | None:
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _log(event: dict) -> None:
    path = os.environ.get("ORCH_FAKE_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _attempt(key: str) -> int:
    d = Path(os.environ.get("ORCH_FAKE_STATE", ".")) / "fake-attempts"
    d.mkdir(parents=True, exist_ok=True)
    f = d / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".count")
    n = int(f.read_text()) if f.exists() else 0
    f.write_text(str(n + 1))
    return n


def _pick(value, index: int):
    if isinstance(value, list):
        if not value:
            return {}
        return value[min(index, len(value) - 1)]
    return value or {}


def _apply_actions(actions: dict, cwd: Path) -> str:
    for rel, content in (actions.get("write") or {}).items():
        p = cwd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    for rel in actions.get("delete") or []:
        p = cwd / rel
        if p.exists():
            p.unlink()
    if actions.get("spawn_child_sleep"):
        # leaves a grandchild process behind to verify process-tree termination
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({actions['spawn_child_sleep']})"])
        pidfile = os.environ.get("ORCH_FAKE_CHILD_PIDFILE")
        if pidfile:
            Path(pidfile).write_text(str(child.pid))
    if actions.get("sleep"):
        time.sleep(float(actions["sleep"]))
    if actions.get("print"):
        print(actions["print"], file=sys.stderr, flush=True)
    return actions.get("message", "done")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("claude", "codex"):
        print("usage: fake_agent (claude|codex) <cli args...>", file=sys.stderr)
        return 2
    flavor = sys.argv[1]
    argv = sys.argv[2:]
    if "--version" in argv:
        print(f"fake-{flavor} 0.0.0")
        return 0
    if flavor == "codex" and argv[:2] == ["login", "status"]:
        print("Logged in using fake credentials")
        return 0
    prompt = sys.stdin.read()
    meta = _meta(prompt)
    purpose = meta.get("purpose", "work")
    task = meta.get("task", "")
    cwd = Path(_arg_value(argv, "--cd", "-C") or os.getcwd())
    scenario_path = os.environ.get("ORCH_FAKE_SCENARIO")
    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8")) if scenario_path else {}

    start = time.time()
    _log({"event": "start", "provider": flavor, "purpose": purpose, "task": task, "t": start, "cwd": str(cwd),
          "pid": os.getpid()})
    exit_code = 0
    structured = None
    text = "done"
    if purpose == "plan":
        structured = scenario.get("plan")
        text = json.dumps(structured)
    elif purpose == "review":
        reviews = scenario.get("reviews", {}).get(flavor)
        rnd = int(meta.get("round", "0"))
        structured = _pick(reviews, rnd) if reviews is not None else scenario.get(
            "default_review", {"verdict": "approve", "summary": "looks good", "findings": []})
        if structured.get("_exit_code"):
            exit_code = int(structured["_exit_code"])
        text = json.dumps(structured)
    else:
        key = task
        spec = scenario.get("tasks", {}).get(key)
        if spec is None and purpose == "repair":
            spec = scenario.get("repair", {})
        idx = _attempt(f"{purpose}-{key}")
        actions = _pick(spec, idx) if spec is not None else {}
        text = _apply_actions(actions, cwd)
        exit_code = int(actions.get("exit_code", 0))
    snapshot = sorted(str(p.relative_to(cwd)).replace("\\", "/") for p in cwd.rglob("*")
                      if p.is_file() and ".git" not in p.relative_to(cwd).parts)
    _log({"event": "end", "provider": flavor, "purpose": purpose, "task": task, "t": time.time(), "cwd": str(cwd),
          "exit_code": exit_code, "files": snapshot})

    if flavor == "claude":
        print(json.dumps({"type": "system", "subtype": "init", "model": "fake", "cwd": str(cwd)}))
        print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": text[:200]}]}}))
        res = {"type": "result", "subtype": "success" if exit_code == 0 else "error_during_execution",
               "is_error": exit_code != 0, "result": text, "total_cost_usd": 0.01, "num_turns": 1,
               "usage": {"input_tokens": 10, "output_tokens": 5}, "permission_denials": []}
        if "--json-schema" in argv:
            res["structured_output"] = structured
        print(json.dumps(res), flush=True)
    else:
        print(json.dumps({"type": "thread.started", "thread_id": "fake"}))
        print(json.dumps({"type": "item.completed", "item": {"id": "1", "type": "agent_message", "text": text}}))
        if exit_code == 0:
            print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 7}}))
        else:
            print(json.dumps({"type": "turn.failed", "error": {"message": "scripted failure"}}))
        out = _arg_value(argv, "-o", "--output-last-message")
        if out:
            Path(out).write_text(text, encoding="utf-8")
        sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
