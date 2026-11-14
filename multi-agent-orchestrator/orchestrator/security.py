"""Secret redaction and path-scope helpers."""
from __future__ import annotations

import fnmatch
import os
import re
from typing import Iterable

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)['\"]?[^\s'\"]{8,}"),
]
_SENSITIVE_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.I)


class Redactor:
    """Removes secrets from text before it is written to logs or the database."""

    def __init__(self, extra_values: Iterable[str] = ()):
        values = {v for v in extra_values if v and len(v) >= 8}
        for name, value in os.environ.items():
            if _SENSITIVE_ENV.search(name) and value and len(value) >= 8:
                values.add(value)
        # longest first so overlapping values are fully masked
        self._values = sorted(values, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        if not text:
            return text
        for v in self._values:
            if v in text:
                text = text.replace(v, "***REDACTED***")
        for pat in _SECRET_PATTERNS:
            if pat.groups:
                text = pat.sub(lambda m: m.group(1) + "***REDACTED***", text)
            else:
                text = pat.sub("***REDACTED***", text)
        return text


def normalize_rel_path(path: str) -> str:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def path_matches(path: str, patterns: Iterable[str]) -> bool:
    """gitignore-like matching: 'src/**' matches everything under src, '*.py' matches basenames too,
    '**/x/**' matches x at any depth (including the top level)."""
    p = normalize_rel_path(path)
    for raw in patterns:
        pat = normalize_rel_path(raw).rstrip("/")
        if not pat:
            continue
        if pat in ("**", "**/*"):
            return True
        if pat.startswith("**/"):
            parts = p.split("/")
            if any(path_matches("/".join(parts[i:]), [pat[3:]]) for i in range(len(parts))):
                return True
            continue
        if pat.endswith("/**"):
            prefix = pat[:-3]
            if p == prefix or p.startswith(prefix + "/") or fnmatch.fnmatchcase(p, pat[:-3] + "/*"):
                return True
            if any(ch in prefix for ch in "*?[") and fnmatch.fnmatchcase(p, pat.replace("/**", "/*")):
                return True
            continue
        if fnmatch.fnmatchcase(p, pat):
            return True
        if pat.startswith("**/") and (fnmatch.fnmatchcase(p, pat[3:]) or fnmatch.fnmatchcase(p.rsplit("/", 1)[-1], pat[3:])):
            return True
        if "/" not in pat and fnmatch.fnmatchcase(p.rsplit("/", 1)[-1], pat):
            return True
        # a plain directory name: "src" or "src/" covers its content
        if not any(ch in pat for ch in "*?[") and p.startswith(pat + "/"):
            return True
    return False


def is_safe_relative(path: str) -> bool:
    p = normalize_rel_path(path)
    if not p or p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        return False
    return ".." not in p.split("/")
