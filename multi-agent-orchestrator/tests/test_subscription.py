"""Subscription-only policy: no agent may ever run on API-key billing."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from conftest import plan
from orchestrator.agents.claude_adapter import ClaudeAdapter
from orchestrator.agents.codex_adapter import CodexAdapter
from orchestrator.subscription import (check_claude_auth_status, check_codex_login_status,
                                       claude_init_violation)


@pytest.mark.parametrize("status,ok", [
    ({"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"}, True),
    ({"loggedIn": True, "authMethod": "oauth_token"}, True),
    ({"loggedIn": True, "authMethod": "api_key", "apiKeySource": "ANTHROPIC_API_KEY"}, False),
    ({"loggedIn": True, "authMethod": "api_key", "apiKeySource": "/login managed key"}, False),
    ({"loggedIn": True, "authMethod": "api_key_helper", "apiKeySource": "apiKeyHelper"}, False),
    ({"loggedIn": True, "authMethod": "third_party"}, False),
    ({"loggedIn": True, "authMethod": "claude.ai", "apiKeySource": "ANTHROPIC_API_KEY"}, False),
    ({"loggedIn": False, "authMethod": "none"}, False),
])
def test_claude_auth_status_policy(status, ok):
    assert check_claude_auth_status(json.dumps(status))[0] is ok
    assert check_claude_auth_status("garbage")[0] is False


def test_claude_init_guard():
    good = json.dumps({"type": "system", "subtype": "init", "apiKeySource": "none"})
    bad = json.dumps({"type": "system", "subtype": "init", "apiKeySource": "ANTHROPIC_API_KEY"})
    assert claude_init_violation(good) is None
    assert "POLICY VIOLATION" in claude_init_violation(bad)
    assert claude_init_violation(json.dumps({"type": "assistant"})) is None


@pytest.mark.parametrize("text,ok", [
    ("Logged in using ChatGPT", True),
    ("Logged in using an API key - sk-proj-***", False),
    ("Logged in using access token", False),
    ("Not logged in", False),
])
def test_codex_login_policy(text, ok):
    assert check_codex_login_status(text)[0] is ok


def test_adapters_strip_api_keys_and_force_chatgpt_login(tmp_path):
    c = ClaudeAdapter({}, base_argv=["claude"]).build(purpose="work", prompt="p", cwd=str(tmp_path),
                                                      artifacts_dir=str(tmp_path))
    assert "ANTHROPIC_API_KEY" in c.env_remove and "ANTHROPIC_AUTH_TOKEN" in c.env_remove
    x = CodexAdapter({}, base_argv=["codex"]).build(purpose="work", prompt="p", cwd=str(tmp_path),
                                                    artifacts_dir=str(tmp_path))
    assert "OPENAI_API_KEY" in x.env_remove and 'forced_login_method="chatgpt"' in x.argv
    from orchestrator.procutil import child_env
    import os
    os.environ["OPENAI_API_KEY"] = "sk-test-should-not-pass"
    try:
        assert "OPENAI_API_KEY" not in child_env(remove=x.env_remove)
    finally:
        del os.environ["OPENAI_API_KEY"]


def test_api_keys_never_reach_agent_processes(h, repo, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-thisMustNeverBeUsed000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-thisMustNeverBeUsed0000000000")
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]},
                             {"id": "t2", "provider": "claude", "files": ["b.py"]}),
                "tasks": {"t1": {"write": {"a.py": "a = 1\n"}}, "t2": {"write": {"b.py": "b = 1\n"}}}})
    orch = h.orchestrator()
    _, status = h.run(orch, repo)
    assert status == "SUCCEEDED"
    events = h.events()
    assert events and all(e.get("leaked_env", []) == [] for e in events if "leaked_env" in e)
    codex_calls = [e for e in events if e.get("provider") == "codex" and e.get("event") == "start"]
    assert codex_calls and all('forced_login_method="chatgpt"' in e["argv"] for e in codex_calls)


def test_claude_api_key_login_is_refused_and_work_goes_to_subscription_codex(h, repo, monkeypatch):
    monkeypatch.setenv("ORCH_FAKE_CLAUDE_AUTH", json.dumps({"loggedIn": True, "authMethod": "api_key",
                                                            "apiKeySource": "ANTHROPIC_API_KEY"}))
    h.scenario({"plan": plan({"id": "t1", "provider": "claude", "files": ["a.py"]}),
                "tasks": {"t1": {"write": {"a.py": "a = 1\n"}}}})
    orch = h.orchestrator(agents={"reviewers": ["codex"]})
    run_id, status = h.run(orch, repo)
    assert status == "SUCCEEDED"
    assert not [e for e in h.events() if e.get("provider") == "claude" and e["event"] == "start"]
    assert orch.store.get_task(run_id, "t1")["provider"] == "codex"
    assert any("API billing is not allowed" in e["message"] for e in orch.store.list_events(run_id))


def test_no_subscription_login_at_all_fails_without_running_agents(h, repo, monkeypatch):
    monkeypatch.setenv("ORCH_FAKE_CLAUDE_AUTH", json.dumps({"loggedIn": True, "authMethod": "api_key"}))
    monkeypatch.setenv("ORCH_FAKE_CODEX_LOGIN", "Logged in using an API key - sk-***")
    h.scenario({"plan": plan({"id": "t1", "provider": "claude", "files": ["a.py"]})})
    orch = h.orchestrator()
    run_id, status = h.run(orch, repo)
    assert status == "FAILED"
    assert "no agent provider is available" in orch.store.get_run(run_id)["error"]
    assert not [e for e in h.events() if e["event"] == "start"]


def test_runtime_api_key_detection_kills_agent_and_fails_run(h, repo, monkeypatch):
    """If the CLI still starts on an API key (e.g. managed settings), the init event gives it away."""
    monkeypatch.setenv("ORCH_FAKE_API_KEY_SOURCE", "ANTHROPIC_API_KEY")
    h.scenario({"plan": plan({"id": "t1", "provider": "codex", "files": ["a.py"]},
                             {"id": "t2", "provider": "claude", "files": ["b.py"]}),
                "tasks": {"t1": {"sleep": 20, "write": {"a.py": "a = 1\n"}},
                          "t2": {"write": {"b.py": "b = 1\n"}}}})
    orch = h.orchestrator()
    run_id = orch.create_run("x", repo)
    t0 = time.time()
    status = asyncio.run(orch.execute(run_id, plan_file=None))
    assert time.time() - t0 < 15                    # killed, not waited for
    assert status == "FAILED"
    run = orch.store.get_run(run_id)
    assert "SUBSCRIPTION POLICY VIOLATION" in run["error"]
    tasks = {t["id"]: t["status"] for t in orch.store.list_tasks(run_id)}
    assert "COMPLETED" not in tasks.values()        # nothing produced under API billing was integrated
