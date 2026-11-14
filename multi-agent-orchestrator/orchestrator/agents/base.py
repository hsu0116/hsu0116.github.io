"""Provider adapter interface. An adapter only *builds* a command and *parses* its output;
process execution lives in ``agent_runner`` so all providers share timeouts, logging,
cancellation and process-tree cleanup."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import AgentResult

# purposes
PLAN = "plan"
WORK = "work"
REVIEW = "review"
REPAIR = "repair"


@dataclass
class Invocation:
    provider: str
    purpose: str
    argv: list[str]
    cwd: str
    stdin_text: str
    model: str | None = None
    artifacts_dir: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class AgentAdapter:
    name = "base"

    def __init__(self, provider_config: dict, base_argv: list[str] | None = None):
        self.cfg = provider_config
        self._base_argv = base_argv

    # -- to implement -------------------------------------------------------
    def build(self, *, purpose: str, prompt: str, cwd: str, artifacts_dir: str,
              model: str | None = None, schema: dict | None = None) -> Invocation:
        raise NotImplementedError

    def parse(self, inv: Invocation, exit_code: int | None, stdout: str, stderr: str) -> AgentResult:
        raise NotImplementedError

    def availability(self) -> tuple[bool, str]:
        return True, "ok"

    # -- helpers ------------------------------------------------------------
    def model_for(self, purpose: str, override: str | None) -> str | None:
        if override:
            return override
        if purpose == PLAN and self.cfg.get("planner_model"):
            return self.cfg["planner_model"]
        if purpose == REVIEW and self.cfg.get("reviewer_model"):
            return self.cfg["reviewer_model"]
        return self.cfg.get("model")

    @staticmethod
    def write_schema(artifacts_dir: str, schema: dict, name: str = "schema.json") -> str:
        p = Path(artifacts_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(schema), encoding="utf-8")
        return str(p)


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object from model output (fenced or bare)."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    import re
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None
