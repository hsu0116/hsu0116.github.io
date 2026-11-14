"""Claude <-> Codex cross review of the same integration commit."""
from __future__ import annotations

import asyncio
from typing import Any

from .agents import REVIEW, AgentAdapter
from .agent_runner import AgentRunner
from .models import ReviewFinding, ReviewResult, Severity
from .planner import render

SEVERITIES = [s.value for s in Severity]

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "summary", "findings"],
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "request_changes"]},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "file", "line", "category", "title", "recommendation"],
                "properties": {
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "file": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                    "category": {"type": "string"},
                    "title": {"type": "string"},
                    "recommendation": {"type": "string"},
                },
            },
        },
    },
}


def parse_review(reviewer: str, target: str, data: Any) -> ReviewResult:
    if not isinstance(data, dict) or "findings" not in data:
        return ReviewResult(reviewer, target, "error", error="reviewer returned no valid JSON review")
    findings = []
    for f in data.get("findings") or []:
        if not isinstance(f, dict):
            continue
        sev = str(f.get("severity", "info")).lower()
        if sev not in SEVERITIES:
            sev = "medium"
        line = f.get("line")
        findings.append(ReviewFinding(severity=sev, file=str(f.get("file", "")), title=str(f.get("title", "")),
                                      recommendation=str(f.get("recommendation", "")),
                                      line=line if isinstance(line, int) else None,
                                      category=str(f.get("category", "general"))))
    verdict = data.get("verdict") if data.get("verdict") in ("approve", "request_changes") else "request_changes"
    return ReviewResult(reviewer, target, verdict, summary=str(data.get("summary", "")), findings=findings)


def reviewer_focus(reviewer: str, authors: dict[str, list[str]]) -> str:
    others = {p: ts for p, ts in authors.items() if p != reviewer}
    if not others:
        return f"all changes (no code from another provider; note: same-provider review)"
    return "; ".join(f"{p} (tasks {', '.join(ts)})" for p, ts in others.items())


async def run_reviews(*, runner: AgentRunner, adapters: dict[str, AgentAdapter], reviewers: list[str],
                      run_id: str, round_no: int, request: str, tasks_text: str, authors: dict[str, list[str]],
                      cwd: str, base_commit: str, target_commit: str, diff_stat: str, diff: str,
                      test_summary: str, max_parallel: int, timeout: float, artifacts_dir: str,
                      cancel_event: asyncio.Event | None = None) -> list[ReviewResult]:
    sem = asyncio.Semaphore(max_parallel)
    authors_text = ", ".join(f"{p}: {', '.join(ts)}" for p, ts in authors.items()) or "unknown"

    async def one(reviewer: str) -> ReviewResult:
        async with sem:
            adapter = adapters[reviewer]
            prompt = render("reviewer.md", run_id=run_id, round=round_no, reviewer=reviewer, authors=authors_text,
                            review_focus=reviewer_focus(reviewer, authors), request=request, tasks=tasks_text,
                            target_commit=target_commit, base_commit=base_commit, diff_stat=diff_stat or "(none)",
                            diff=diff or "(empty)", test_summary=test_summary)
            inv = adapter.build(purpose=REVIEW, prompt=prompt, cwd=cwd, artifacts_dir=artifacts_dir,
                                schema=REVIEW_SCHEMA)
            res = await runner.run(adapter, inv, run_id=run_id, task_id=None, timeout=timeout,
                                   cancel_event=cancel_event, log_name=f"review-r{round_no}-{reviewer}")
            if not res.ok:
                return ReviewResult(reviewer, target_commit, "error", error=res.error or "review failed")
            return parse_review(reviewer, target_commit, res.structured)

    return list(await asyncio.gather(*(one(r) for r in reviewers)))


def blocking_findings(reviews: list[ReviewResult], blocking: list[str]) -> list[tuple[str, ReviewFinding]]:
    out = []
    for r in reviews:
        for f in r.findings:
            if f.severity in blocking:
                out.append((r.reviewer, f))
    return out
