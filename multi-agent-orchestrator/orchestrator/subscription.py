"""Subscription-only policy: agents must never bill an API key.

This policy is always on and cannot be disabled from config.

Claude Code (values verified against Claude Code 2.1.x):
  * API-key environment variables are removed from every agent process
    (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, Bedrock/Vertex/Foundry switches).
  * Before use, ``claude auth status --json`` must report ``authMethod`` "claude.ai" (Pro/Max login)
    or "oauth_token" (subscription token from ``claude setup-token``). "api_key", "api_key_helper",
    "third_party" and "none" are refused.
  * At run time the ``system/init`` event of every ``claude -p`` call must report
    ``apiKeySource: "none"``; otherwise the process tree is killed at once and the run is stopped.
  * the keys of the *other* provider are removed as well (ALL_BLOCKED_ENV), also for test commands.
  * ``--restricted`` makes the CLI ignore user/project settings files, so an ``apiKeyHelper``
    there is not used either.

Codex CLI (verified against codex-cli 0.169):
  * OPENAI_API_KEY / CODEX_API_KEY are removed from the agent environment.
  * every call passes ``-c forced_login_method="chatgpt"`` (Codex rejects API-key auth then).
  * ``codex login status`` must say "Logged in using ChatGPT".
"""
from __future__ import annotations

import json

CLAUDE_BLOCKED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
)
CODEX_BLOCKED_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "AZURE_OPENAI_API_KEY")
# every agent process and test run gets neither provider's keys
ALL_BLOCKED_ENV = CLAUDE_BLOCKED_ENV + CODEX_BLOCKED_ENV

CLAUDE_ALLOWED_AUTH_METHODS = ("claude.ai", "oauth_token")
CODEX_FORCED_LOGIN_ARGS = ["-c", 'forced_login_method="chatgpt"']

POLICY_VIOLATION = "SUBSCRIPTION POLICY VIOLATION"


def check_claude_auth_status(raw: str) -> tuple[bool, str]:
    """Validate the output of ``claude auth status --json``."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False, "could not read 'claude auth status --json' output"
    if not data.get("loggedIn"):
        return False, "not logged in - run 'claude' and sign in with your Claude Pro account"
    method = data.get("authMethod")
    if data.get("apiKeySource") and data.get("apiKeySource") != "none":
        return False, (f"an API key is configured ({data.get('apiKeySource')}); API billing is not allowed. "
                       f"Remove it and log in with your Claude subscription")
    if method not in CLAUDE_ALLOWED_AUTH_METHODS:
        return False, (f"auth method '{method}' is not a Claude subscription login; API billing is not allowed. "
                       f"Run 'claude auth login' with your Pro account")
    plan = data.get("subscriptionType") or data.get("planTier")
    return True, f"subscription login ({method}{', plan ' + str(plan) if plan else ''})"


def claude_init_violation(line: str) -> str | None:
    """Runtime guard for one stream-json line; returns an error if the call would use an API key."""
    if '"init"' not in line or '"system"' not in line:
        return None
    try:
        ev = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if ev.get("type") != "system" or ev.get("subtype") != "init":
        return None
    src = ev.get("apiKeySource")
    if src not in (None, "none"):
        return f"{POLICY_VIOLATION}: claude started with API key source '{src}' - process terminated"
    return None


def check_codex_login_status(text: str) -> tuple[bool, str]:
    t = text.strip()
    if "logged in using chatgpt" in t.lower():
        return True, "ChatGPT subscription login"
    if "not logged in" in t.lower() or not t:
        return False, "not logged in - run 'codex login' and choose 'Sign in with ChatGPT'"
    return False, f"'{t[:80]}' is not a ChatGPT subscription login; API billing is not allowed"
