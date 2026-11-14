from .base import AgentAdapter, Invocation, PLAN, REPAIR, REVIEW, WORK
from .claude_adapter import ClaudeAdapter
from .codex_adapter import CodexAdapter

ADAPTERS = {"claude": ClaudeAdapter, "codex": CodexAdapter}
PROVIDERS = tuple(ADAPTERS)


def make_adapter(provider: str, config: dict) -> AgentAdapter:
    if provider not in ADAPTERS:
        raise ValueError(f"unknown provider '{provider}' (supported: {', '.join(ADAPTERS)})")
    return ADAPTERS[provider](config["providers"].get(provider, {}))


__all__ = ["AgentAdapter", "Invocation", "PLAN", "REPAIR", "REVIEW", "WORK", "ClaudeAdapter",
           "CodexAdapter", "make_adapter", "PROVIDERS"]
