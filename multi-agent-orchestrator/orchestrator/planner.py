"""AI planner (Claude by default) + strict plan validation (ids, scopes, DAG)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TASK_ID_RE, Plan, TaskSpec, normalize_task_id
from .security import is_safe_relative

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Strict JSON schema usable by both `claude --json-schema` and `codex --output-schema`
# (all properties required, no additional properties).
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "shared_contracts", "tasks"],
    "properties": {
        "summary": {"type": "string"},
        "shared_contracts": {"type": "string"},
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "title", "description", "provider", "role", "depends_on", "files",
                             "acceptance_criteria", "verification"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "provider": {"type": "string", "enum": ["claude", "codex"]},
                    "role": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                    "verification": {"type": "string"},
                },
            },
        },
    },
}


class PlanError(ValueError):
    pass


def render(template_name: str, **values: Any) -> str:
    text = (PROMPTS_DIR / template_name).read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def build_planner_prompt(run_id: str, request: str, config: dict, providers: list[str]) -> str:
    workers = config["agents"]["workers"]
    agents = "\n".join(f"- {w['provider']} (preferred role: {w.get('role', 'general')})"
                       for w in workers if w["provider"] in providers) or \
        "\n".join(f"- {p}" for p in providers)
    return render("planner.md", run_id=run_id, request=request, agents=agents,
                  max_tasks=config["limits"]["max_tasks"], providers=", ".join(providers))


def topological_order(tasks: list[TaskSpec]) -> list[str]:
    ids = {t.id for t in tasks}
    indeg = {t.id: 0 for t in tasks}
    children: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for d in t.depends_on:
            if d in ids:
                indeg[t.id] += 1
                children[d].append(t.id)
    order = []
    queue = [t.id for t in tasks if indeg[t.id] == 0]
    while queue:
        n = queue.pop(0)
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != len(tasks):
        cyc = sorted(i for i, d in indeg.items() if d > 0)
        raise PlanError(f"dependency cycle detected among tasks: {', '.join(cyc)}")
    return order


def ancestors(tasks: dict[str, TaskSpec], tid: str) -> set[str]:
    seen: set[str] = set()
    stack = list(tasks[tid].depends_on)
    while stack:
        d = stack.pop()
        if d in seen or d not in tasks:
            continue
        seen.add(d)
        stack.extend(tasks[d].depends_on)
    return seen


def validate_plan(raw: Any, config: dict, available_providers: list[str]) -> Plan:
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list) or not raw["tasks"]:
        raise PlanError("plan must be an object with a non-empty 'tasks' array")
    warnings: list[str] = []
    max_tasks = config["limits"]["max_tasks"]
    if len(raw["tasks"]) > max_tasks:
        raise PlanError(f"plan has {len(raw['tasks'])} tasks, limit is {max_tasks}")
    role_provider = {w.get("role"): w["provider"] for w in config["agents"]["workers"]}
    tasks: list[TaskSpec] = []
    id_map: dict[str, str] = {}
    for i, t in enumerate(raw["tasks"], 1):
        if not isinstance(t, dict):
            raise PlanError(f"task #{i} is not an object")
        raw_id = str(t.get("id") or f"t{i}")
        tid = raw_id if TASK_ID_RE.match(raw_id) else normalize_task_id(raw_id)
        if tid in id_map.values():
            raise PlanError(f"duplicate task id '{tid}'")
        id_map[raw_id] = tid
        spec = TaskSpec.from_dict({**t, "id": tid})
        if not spec.description.strip():
            raise PlanError(f"task {tid} has no description")
        if spec.provider not in ("claude", "codex"):
            spec.provider = role_provider.get(spec.role, available_providers[0])
            warnings.append(f"task {tid}: unknown provider, assigned {spec.provider}")
        files = [f.strip() for f in spec.files if isinstance(f, str) and f.strip()]
        for f in files:
            if not is_safe_relative(f.replace("**", "x").replace("*", "x")):
                raise PlanError(f"task {tid}: unsafe file scope '{f}'")
        if not files:
            warnings.append(f"task {tid}: no file scope given - task may change any file")
            files = ["**"]
        spec.files = files
        tasks.append(spec)
    known = set(id_map.values())
    for spec in tasks:
        deps = []
        for d in spec.depends_on:
            nd = id_map.get(d, d if TASK_ID_RE.match(str(d)) else normalize_task_id(str(d)))
            if nd not in known:
                raise PlanError(f"task {spec.id} depends on unknown task '{d}'")
            if nd == spec.id:
                raise PlanError(f"task {spec.id} depends on itself")
            if nd not in deps:
                deps.append(nd)
        spec.depends_on = deps
    topological_order(tasks)
    # parallel tasks with overlapping literal scopes -> warn (cherry-pick will catch real conflicts)
    by_id = {t.id: t for t in tasks}
    for a in tasks:
        for b in tasks:
            if a.id >= b.id or a.id in ancestors(by_id, b.id) or b.id in ancestors(by_id, a.id):
                continue
            overlap = set(a.files) & set(b.files)
            if overlap or "**" in a.files or "**" in b.files:
                warnings.append(f"tasks {a.id} and {b.id} may run in parallel with overlapping scope "
                                f"{sorted(overlap) or ['**']}")
    return Plan(summary=str(raw.get("summary", "")), shared_contracts=str(raw.get("shared_contracts", "") or ""),
                tasks=tasks, warnings=warnings)


def apply_provider_availability(plan: Plan, available: list[str], fallback: bool) -> None:
    for t in plan.tasks:
        if t.provider not in available:
            if not fallback or not available:
                raise PlanError(f"task {t.id} needs provider '{t.provider}' which is unavailable")
            new = available[0]
            plan.warnings.append(f"task {t.id}: provider {t.provider} unavailable, reassigned to {new}")
            t.provider = new


def load_plan_file(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
