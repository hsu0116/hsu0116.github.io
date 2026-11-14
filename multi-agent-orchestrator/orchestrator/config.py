"""Configuration loading (YAML) with safe defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "agents": {
        "planner": "claude",
        "workers": [
            {"provider": "codex", "role": "backend"},
            {"provider": "claude", "role": "frontend"},
            {"provider": "codex", "role": "testing"},
        ],
        "reviewers": ["claude", "codex"],
        # Provider used for repair tasks when no author can be inferred.
        "repair_default": "claude",
        # If a provider is not installed / not logged in, re-assign its tasks.
        "fallback_on_unavailable": True,
    },
    "providers": {
        "claude": {
            "command": None,            # None -> resolve "claude" on PATH; or a list, e.g. ["C:/.../claude.exe"]
            "model": None,
            "reviewer_model": None,
            "planner_model": None,
            # Usage cap per call, measured in the CLI's API-equivalent estimate (subscription logins
            # are not billed per call; this only limits how much of your plan's usage one call may take).
            "max_budget_usd_per_call": 5.0,
            # Built-in tools exposed to workers (--tools); everything else is removed.
            "worker_tools": ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            # Pre-approved rules (--allowedTools). Anything not listed is denied (dontAsk mode).
            "worker_allowed_tools": [
                "Read", "Write", "Edit", "Glob", "Grep",
                "Bash(python -m pytest:*)", "Bash(python3 -m pytest:*)", "Bash(py -m pytest:*)",
                "Bash(pytest:*)", "Bash(npm test:*)",
                "Bash(git status:*)", "Bash(git diff:*)",
            ],
            "disallowed_tools": [
                "Bash(git push:*)", "Bash(git reset:*)", "Bash(git checkout:*)",
                "Bash(git clean:*)", "Bash(git commit:*)", "Bash(rm:*)", "WebFetch", "WebSearch",
            ],
            "reviewer_tools": ["Read", "Glob", "Grep", "Bash"],
            "reviewer_allowed_tools": [
                "Read", "Glob", "Grep",
                "Bash(git diff:*)", "Bash(git log:*)", "Bash(git show:*)",
            ],
            "extra_args": [],
        },
        "codex": {
            "command": None,            # None -> resolve "codex" on PATH (npm shim resolved to node + js)
            "model": None,
            "reviewer_model": None,
            "planner_model": None,
            "worker_sandbox": "workspace-write",
            "reviewer_sandbox": "read-only",
            "extra_args": [],
        },
    },
    "limits": {
        "max_parallel_workers": 3,
        "max_parallel_reviewers": 2,
        "max_repair_rounds": 2,
        "max_task_retries": 1,           # extra attempts after a failed agent run
        "task_timeout_seconds": 1800,
        "planner_timeout_seconds": 600,
        "review_timeout_seconds": 900,
        "test_timeout_seconds": 900,
        "max_tasks": 8,
        "max_total_cost_usd": None,      # usage cap (API-equivalent estimate) for the whole run
        "max_diff_chars_for_review": 60000,
    },
    "integration": {
        "method": "cherry-pick",
        "require_manual_main_merge": True,
        "auto_push": False,
        "branch_prefix": "orchestrator",
        "agent_branch_prefix": "agent",
        "protected_branches": ["main", "master", "develop", "release", "production"],
    },
    "validation": {
        # Command (argument list) run in the integration worktree. "auto" detects pytest / npm test.
        "test_command": "auto",
        "run_tests_per_task": False,
        "scope_violation": "fail",       # fail | warn
        "allow_deletions": False,        # deleting files that existed at the base commit
        "forbidden_paths": [
            ".git/**", ".env", "**/.env", "**/*.pem", "**/*.key", "**/id_rsa*",
            ".github/workflows/**",
        ],
        "blocking_severities": ["critical", "high"],
        # generated files that are never committed nor counted as changes
        "ignore_patterns": [
            "**/__pycache__/**", "**/*.pyc", ".pytest_cache/**", "**/.pytest_cache/**",
            "node_modules/**", "**/node_modules/**", ".DS_Store", "**/.DS_Store", "*.egg-info/**",
            ".coverage", "htmlcov/**",
        ],
    },
    "review": {
        "enabled": True,
        "required": True,                # a failed reviewer makes the run NEEDS_ATTENTION
    },
    "storage": {
        "type": "sqlite",
        "data_dir": "data",
    },
    "logging": {
        "save_agent_output": True,
        "save_git_diff": True,
        "save_test_results": True,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigError(ValueError):
    pass


def validate_config(cfg: dict[str, Any]) -> None:
    lim = cfg["limits"]
    for key in ("max_parallel_workers", "max_parallel_reviewers", "task_timeout_seconds"):
        if not isinstance(lim[key], int) or lim[key] < 1:
            raise ConfigError(f"limits.{key} must be a positive integer")
    if not isinstance(lim["max_repair_rounds"], int) or lim["max_repair_rounds"] < 0:
        raise ConfigError("limits.max_repair_rounds must be >= 0")
    if cfg["integration"]["method"] != "cherry-pick":
        raise ConfigError("integration.method: only 'cherry-pick' is supported")
    if not cfg["integration"]["require_manual_main_merge"]:
        raise ConfigError("integration.require_manual_main_merge=false is not supported: "
                          "the orchestrator never merges into main automatically")
    tc = cfg["validation"]["test_command"]
    if not (tc in (None, "auto") or (isinstance(tc, list) and all(isinstance(x, str) for x in tc))):
        raise ConfigError("validation.test_command must be 'auto', null, or a list of strings "
                          "(argument array, no shell string)")
    if cfg["validation"]["scope_violation"] not in ("fail", "warn"):
        raise ConfigError("validation.scope_violation must be 'fail' or 'warn'")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    base_dir = Path.cwd()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError("config root must be a mapping")
        cfg = deep_merge(cfg, data)
        base_dir = p.resolve().parent
    if overrides:
        cfg = deep_merge(cfg, overrides)
    data_dir = Path(cfg["storage"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    cfg["storage"]["data_dir"] = str(data_dir.resolve())
    validate_config(cfg)
    return cfg
