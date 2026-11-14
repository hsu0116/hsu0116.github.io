from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.config import load_config  # noqa: E402
from orchestrator.engine import Orchestrator  # noqa: E402

FAKE = str(ROOT / "orchestrator" / "agents" / "fake_agent.py")

CALC = "def add(a, b):\n    return a + b\n"
TEST_CALC = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "target"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "tester")
    (r / "calc.py").write_text(CALC)
    (r / "tests").mkdir()
    (r / "tests" / "test_calc.py").write_text(TEST_CALC)
    (r / "conftest.py").write_text("")
    (r / "README.md").write_text("# target\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "initial")
    return r


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp = tmp_path
        self.scenario_path = tmp_path / "scenario.json"
        self.log_path = tmp_path / "fake-log.jsonl"
        self.state = tmp_path / "fake-state"
        self.state.mkdir()
        monkeypatch.setenv("ORCH_FAKE_SCENARIO", str(self.scenario_path))
        monkeypatch.setenv("ORCH_FAKE_LOG", str(self.log_path))
        monkeypatch.setenv("ORCH_FAKE_STATE", str(self.state))

    def config(self, **limits_and_more) -> dict:
        overrides = {
            "providers": {
                "claude": {"command": [sys.executable, FAKE, "claude"]},
                "codex": {"command": [sys.executable, FAKE, "codex"]},
            },
            "limits": {"task_timeout_seconds": 60, "max_task_retries": 1, "max_repair_rounds": 2},
            "validation": {"test_command": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]},
            "storage": {"data_dir": str(self.tmp / "data")},
        }
        for section, values in limits_and_more.items():
            overrides.setdefault(section, {}).update(values)
        return load_config(None, overrides)

    def scenario(self, data: dict) -> None:
        self.scenario_path.write_text(json.dumps(data))

    def events(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [json.loads(x) for x in self.log_path.read_text().splitlines() if x.strip()]

    def orchestrator(self, **cfg) -> Orchestrator:
        return Orchestrator(self.config(**cfg))

    def run(self, orch: Orchestrator, repo: Path, request: str = "build it", **kw) -> tuple[str, str]:
        run_id = orch.create_run(request, repo)
        status = asyncio.run(orch.execute(run_id, **kw))
        return run_id, status


@pytest.fixture
def h(tmp_path, monkeypatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def plan(*tasks: dict, summary: str = "test plan", contracts: str = "") -> dict:
    full = []
    for t in tasks:
        full.append({"title": t["id"], "description": f"do {t['id']}", "role": "general", "depends_on": [],
                     "acceptance_criteria": ["works"], "verification": "pytest", **t})
    return {"summary": summary, "shared_contracts": contracts, "tasks": full}
