"""Validates an agent's work in its worktree before it is committed and integrated."""
from __future__ import annotations

from .models import AgentResult, TaskSpec, ValidationResult
from .security import is_safe_relative, path_matches
from .worktree_manager import WorktreeManager


def validate_task_result(wt: WorktreeManager, worktree: str, base_commit: str, spec: TaskSpec,
                         agent: AgentResult, config: dict) -> ValidationResult:
    v = ValidationResult(ok=True)
    vcfg = config["validation"]

    # 1. agent success
    if not agent.ok:
        v.ok = False
        v.errors.append(f"agent run failed: {agent.error or f'exit code {agent.exit_code}'}")

    # 2. changes exist
    changes = wt.changes(worktree, vcfg.get("ignore_patterns") or [])
    v.changed_files = sorted({c.path for c in changes})
    for c in changes:
        if c.status == "R" and c.old_path:
            v.changed_files.append(c.old_path)
    if not changes and not spec.allow_no_changes:
        v.ok = False
        v.errors.append("agent produced no file changes")

    # 3. scope / forbidden paths / deletions
    forbidden = vcfg.get("forbidden_paths") or []
    out_of_scope: list[str] = []
    for c in changes:
        paths = [c.path] + ([c.old_path] if c.old_path else [])
        for p in paths:
            if not is_safe_relative(p):
                v.ok = False
                v.errors.append(f"unsafe path: {p}")
            elif path_matches(p, forbidden):
                v.ok = False
                v.errors.append(f"forbidden path modified: {p}")
            elif not path_matches(p, spec.files or ["**"]):
                out_of_scope.append(p)
        deleted = c.status == "D" or (c.status == "R" and c.old_path)
        if deleted:
            gone = c.path if c.status == "D" else c.old_path
            if gone and wt.file_exists_at(base_commit, gone):
                v.deleted_files.append(gone)
    if out_of_scope:
        msg = f"changes outside the allowed scope {spec.files}: {sorted(set(out_of_scope))}"
        if vcfg.get("scope_violation", "fail") == "fail":
            v.ok = False
            v.errors.append(msg)
        else:
            v.warnings.append(msg)
    if v.deleted_files and not vcfg.get("allow_deletions", False):
        v.ok = False
        v.errors.append(f"deletion of existing files is not allowed: {v.deleted_files}")
    if agent.permission_denials:
        v.warnings.append(f"{len(agent.permission_denials)} tool call(s) were denied by the permission policy")
    v.changed_files = sorted(set(v.changed_files))
    return v
