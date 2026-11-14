"""Core data models shared by the engine, the CLI and (later) a web dashboard."""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any


class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"          # waiting for dependencies
    READY = "READY"              # dependencies satisfied, waiting for a worker slot
    RUNNING = "RUNNING"          # agent process is running
    VALIDATING = "VALIDATING"    # agent finished, checking/committing/integrating
    COMPLETED = "COMPLETED"      # commit integrated into the integration branch
    FAILED = "FAILED"            # agent or validation failed (retries exhausted)
    CONFLICT = "CONFLICT"        # cherry-pick conflict, integration aborted safely
    BLOCKED = "BLOCKED"          # a dependency did not complete
    CANCELLED = "CANCELLED"      # cancelled by the user / stop request

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_TASK_STATES


TERMINAL_TASK_STATES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CONFLICT,
    TaskStatus.BLOCKED,
    TaskStatus.CANCELLED,
}


class RunStatus(str, enum.Enum):
    PLANNING = "PLANNING"
    PLANNED = "PLANNED"                  # plan only (--plan-only), waiting for resume
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"              # integration tests + cross review
    REPAIRING = "REPAIRING"
    SUCCEEDED = "SUCCEEDED"              # all tasks integrated, tests passed, no blocking findings
    UNVERIFIED = "UNVERIFIED"            # integrated but no test command configured
    NEEDS_ATTENTION = "NEEDS_ATTENTION"  # failures/conflicts/test failures remain
    FAILED = "FAILED"                    # orchestration error (e.g. planning failed)
    STOPPED = "STOPPED"                  # stopped by user, resumable

    @property
    def active(self) -> bool:
        return self in {RunStatus.PLANNING, RunStatus.RUNNING, RunStatus.VERIFYING, RunStatus.REPAIRING}


class Severity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def normalize_task_id(raw: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", str(raw).strip().lower()).strip("-")
    return s[:40] or "task"


@dataclass
class TaskSpec:
    id: str
    title: str
    description: str
    provider: str = "claude"
    role: str = "general"
    model: str | None = None
    depends_on: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)          # allowed change scope (glob patterns)
    acceptance_criteria: list[str] = field(default_factory=list)
    verification: str = ""
    kind: str = "work"                                      # work | repair
    allow_no_changes: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "provider": self.provider,
            "role": self.role,
            "model": self.model,
            "depends_on": list(self.depends_on),
            "files": list(self.files),
            "acceptance_criteria": list(self.acceptance_criteria),
            "verification": self.verification,
            "kind": self.kind,
            "allow_no_changes": self.allow_no_changes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskSpec":
        return cls(
            id=d["id"],
            title=d.get("title") or d["id"],
            description=d.get("description", ""),
            provider=d.get("provider") or "claude",
            role=d.get("role") or "general",
            model=d.get("model"),
            depends_on=list(d.get("depends_on") or []),
            files=list(d.get("files") or []),
            acceptance_criteria=list(d.get("acceptance_criteria") or []),
            verification=d.get("verification") or "",
            kind=d.get("kind") or "work",
            allow_no_changes=bool(d.get("allow_no_changes", False)),
        )


@dataclass
class Plan:
    summary: str
    tasks: list[TaskSpec]
    shared_contracts: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "shared_contracts": self.shared_contracts,
            "tasks": [t.to_dict() for t in self.tasks],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        return cls(
            summary=d.get("summary", ""),
            shared_contracts=d.get("shared_contracts", "") or "",
            tasks=[TaskSpec.from_dict(t) for t in d.get("tasks", [])],
            warnings=list(d.get("warnings") or []),
        )


@dataclass
class AgentResult:
    provider: str
    exit_code: int | None
    timed_out: bool = False
    cancelled: bool = False
    final_text: str = ""
    structured: Any = None
    cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    log_path: str | None = None
    permission_denials: list[Any] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled and not self.error


@dataclass
class ValidationResult:
    ok: bool
    changed_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReviewFinding:
    severity: str
    file: str
    title: str
    recommendation: str = ""
    line: int | None = None
    category: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "category": self.category,
            "title": self.title,
            "recommendation": self.recommendation,
        }


@dataclass
class ReviewResult:
    reviewer: str
    target_commit: str
    verdict: str                      # approve | request_changes | error
    summary: str = ""
    findings: list[ReviewFinding] = field(default_factory=list)
    error: str | None = None
