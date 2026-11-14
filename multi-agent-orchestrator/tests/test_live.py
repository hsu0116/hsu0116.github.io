"""Live test against the real CLIs. Costs money; run explicitly:

    ORCH_LIVE_MODEL=claude-sonnet-5-5 python -m pytest -m live -s tests/test_live.py

Codex is used when it is installed and logged in; otherwise its tasks fall back to Claude and the
codex review is skipped (reviewers are limited to the available providers).
"""
from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from conftest import git
from orchestrator.config import load_config
from orchestrator.engine import Orchestrator

pytestmark = pytest.mark.live

REQUEST = ("Create a small Python text utility package. 1) `textkit/slug.py` with `slugify(text: str) -> str` "
           "(lowercase, ASCII letters/digits separated by single dashes, strip leading/trailing dashes) and "
           "tests in `tests/test_slug.py`. 2) `textkit/stats.py` with `word_count(text: str) -> int` and "
           "`top_words(text: str, n: int) -> list[tuple[str, int]]` (case-insensitive, ties sorted "
           "alphabetically) with tests in `tests/test_stats.py`. 3) After both exist, a command line module "
           "`textkit/cli.py` (argparse: `slug TEXT` and `stats FILE`) with tests in `tests/test_cli.py`. "
           "Keep it small, standard library only.")


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_live_end_to_end(tmp_path):
    repo = tmp_path / "textkit-repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "tester")
    (repo / "README.md").write_text("# textkit\n")
    (repo / "textkit").mkdir()
    (repo / "textkit" / "__init__.py").write_text("")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "conftest.py").write_text("")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "skeleton")
    main_sha = git(repo, "rev-parse", "main")

    model = os.environ.get("ORCH_LIVE_MODEL")
    probe = Orchestrator(load_config(None, {"storage": {"data_dir": str(tmp_path / "probe")}}))
    reviewers = [p for p in ("claude", "codex") if probe.available(p)[0]]
    data_dir = os.environ.get("ORCH_LIVE_DATA_DIR") or str(tmp_path / "data")
    cfg = load_config(None, {
        "providers": {"claude": {"model": model, "max_budget_usd_per_call": 1.5}},
        "agents": {"reviewers": reviewers},
        "limits": {"max_tasks": 4, "task_timeout_seconds": 900, "max_total_cost_usd": 12.0},
        "validation": {"test_command": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]},
        "storage": {"data_dir": data_dir},
    })
    orch = Orchestrator(cfg)
    run_id = orch.create_run(REQUEST, repo)

    async def go():
        from orchestrator.cli import _run_with_tail
        return await _run_with_tail(orch, run_id, orch.execute(run_id), quiet=False)

    status = asyncio.run(go())
    print(open(os.path.join(data_dir, "runs", run_id, "report.md")).read())
    tasks = orch.store.list_tasks(run_id)
    assert len(tasks) >= 2
    assert git(repo, "rev-parse", "main") == main_sha                     # main untouched
    assert status in ("SUCCEEDED", "NEEDS_ATTENTION")
    if status == "SUCCEEDED":
        integ = orch.store.get_run(run_id)["integration_branch"]
        files = git(repo, "ls-tree", "-r", "--name-only", integ)
        assert "textkit/slug.py" in files and "textkit/stats.py" in files
