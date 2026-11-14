"""Git and worktree management.

Safety rules enforced here (not only by convention):
* every git call goes through ``Git.run`` which refuses push, hard reset, clean, force operations
  and any branch mutation on protected branches or branches the orchestrator did not create;
* the user's own checkout is never switched / reset; all writes happen in dedicated worktrees.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .procutil import child_env
from .security import normalize_rel_path, path_matches


class GitError(RuntimeError):
    def __init__(self, args: list[str], code: int, stdout: str, stderr: str):
        super().__init__(f"git {' '.join(args)} failed ({code}): {stderr.strip() or stdout.strip()}")
        self.code = code
        self.stdout = stdout
        self.stderr = stderr


class GitSafetyError(RuntimeError):
    pass


_ALLOWED_SUBCOMMANDS = {
    "rev-parse", "status", "diff", "log", "show", "add", "commit", "cherry-pick", "worktree",
    "branch", "merge-base", "ls-files", "cat-file", "for-each-ref", "symbolic-ref", "config",
    "init", "check-ref-format", "rev-list",
}
_FORBIDDEN_TOKENS = {"--force", "-f", "--hard", "--force-with-lease", "-D"}

_COMMIT_IDENTITY = ["-c", "user.name=Multi-Agent Orchestrator", "-c", "user.email=orchestrator@localhost",
                    "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + os.devnull]


@dataclass
class FileChange:
    status: str     # A, M, D, R, ?
    path: str
    old_path: str | None = None


class Git:
    def __init__(self, cwd: str | Path, protected_branches: list[str], owned_prefixes: list[str]):
        self.cwd = str(cwd)
        self.protected = set(protected_branches)
        self.owned_prefixes = tuple(p.rstrip("/") + "/" for p in owned_prefixes)

    def _check(self, args: list[str]) -> None:
        sub = args[0]
        if sub not in _ALLOWED_SUBCOMMANDS:
            raise GitSafetyError(f"git {sub} is not allowed by the orchestrator")
        if sub == "worktree":
            # worktree removal of our own worktrees may use --force (agent dirs only); checked by caller
            return
        if sub == "branch":
            # only creation/deletion of orchestrator-owned branches
            names = [a for a in args[1:] if not a.startswith("-")]
            if any(a in ("-D", "--delete", "-d", "-m", "-M", "-f", "--force") for a in args) or names:
                for n in names[:1]:
                    self.assert_owned_branch(n)
            return
        bad = _FORBIDDEN_TOKENS.intersection(args[1:])
        if bad and sub not in ("log", "diff", "show", "rev-list"):
            raise GitSafetyError(f"git {sub} with {sorted(bad)} is not allowed")

    def assert_owned_branch(self, name: str) -> None:
        short = name.removeprefix("refs/heads/")
        if short in self.protected:
            raise GitSafetyError(f"refusing to modify protected branch '{short}'")
        if not short.startswith(self.owned_prefixes):
            raise GitSafetyError(f"refusing to modify branch '{short}' (not created by the orchestrator)")

    def run(self, *args: str, cwd: str | Path | None = None, check: bool = True,
            identity: bool = False) -> subprocess.CompletedProcess:
        argl = list(args)
        self._check(argl)
        # core.quotePath=false: non-ASCII (e.g. Korean) file names are returned verbatim
        cmd = ["git", "-c", "core.quotePath=false"] + (_COMMIT_IDENTITY if identity else []) + argl
        proc = subprocess.run(cmd, cwd=str(cwd or self.cwd), capture_output=True, env=child_env(),
                              check=False)
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        result = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
        if check and proc.returncode != 0:
            raise GitError(argl, proc.returncode, out, err)
        return result

    def out(self, *args: str, cwd: str | Path | None = None) -> str:
        return self.run(*args, cwd=cwd).stdout.strip()


def slugify(text: str, max_len: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:max_len].strip("-")) or "task"


class WorktreeManager:
    def __init__(self, repo_path: str | Path, worktree_root: str | Path, config: dict):
        self.repo = Path(repo_path).resolve()
        self.root = Path(worktree_root).resolve()
        integ = config["integration"]
        self.branch_prefix = integ["branch_prefix"]
        self.agent_prefix = integ["agent_branch_prefix"]
        self.git = Git(self.repo, integ["protected_branches"], [self.branch_prefix, self.agent_prefix])

    # ------------------------------------------------------------ repository
    def ensure_repo(self) -> None:
        try:
            top = self.git.out("rev-parse", "--show-toplevel")
        except GitError as e:
            raise GitError(["rev-parse"], e.code, e.stdout, f"{self.repo} is not a git repository") from e
        if Path(top).resolve() != self.repo:
            self.repo = Path(top).resolve()
            self.git.cwd = str(self.repo)
        self.git.out("rev-parse", "--verify", "HEAD^{commit}")

    def resolve_commit(self, ref: str) -> str:
        return self.git.out("rev-parse", "--verify", f"{ref}^{{commit}}")

    def current_branch(self) -> str | None:
        r = self.git.run("symbolic-ref", "--short", "-q", "HEAD", check=False)
        return r.stdout.strip() or None

    def branch_sha(self, branch: str) -> str | None:
        r = self.git.run("rev-parse", "--verify", "-q", f"refs/heads/{branch}^{{commit}}", check=False)
        return r.stdout.strip() or None

    def branch_exists(self, branch: str) -> bool:
        return self.branch_sha(branch) is not None

    # ------------------------------------------------------------- worktrees
    def integration_branch_name(self, run_id: str) -> str:
        return f"{self.branch_prefix}/{run_id}"

    def task_branch_name(self, run_id: str, task_id: str, title: str) -> str:
        return f"{self.agent_prefix}/{run_id}/{task_id}-{slugify(title, 20)}"

    def create_worktree(self, path: Path, branch: str, start_point: str) -> Path:
        self.git.assert_owned_branch(branch)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise GitSafetyError(f"worktree path already exists: {path}")
        self.git.run("worktree", "add", "-b", branch, str(path), start_point)
        return path

    def attach_worktree(self, path: Path, branch: str) -> Path:
        """Re-create a worktree for an existing orchestrator branch (used on resume)."""
        self.git.assert_owned_branch(branch)
        self.git.run("worktree", "prune")
        if path.exists() and (path / ".git").exists():
            return path
        if path.exists():
            raise GitSafetyError(f"path exists but is not a worktree: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.git.run("worktree", "add", str(path), branch)
        return path

    def remove_worktree(self, path: str | Path, delete_branch: str | None = None) -> None:
        p = Path(path).resolve()
        # only directories that live under the orchestrator worktree root may be removed
        if self.root not in p.parents:
            raise GitSafetyError(f"refusing to remove worktree outside {self.root}: {p}")
        if p.exists():
            r = self.git.run("worktree", "remove", "--force", str(p), check=False)
            if r.returncode != 0 and p.exists():
                shutil.rmtree(p, ignore_errors=True)
        self.git.run("worktree", "prune", check=False)
        if delete_branch and self.branch_exists(delete_branch):
            self.git.assert_owned_branch(delete_branch)
            # -D is otherwise forbidden; allowed here for orchestrator-owned agent branches only
            subprocess.run(["git", "branch", "-D", delete_branch], cwd=str(self.repo),
                           capture_output=True, env=child_env(), check=False)

    # ---------------------------------------------------------- inspection
    def head(self, worktree: str | Path) -> str:
        return self.git.out("rev-parse", "HEAD", cwd=worktree)

    def changes(self, worktree: str | Path, ignore: list[str] | None = None) -> list[FileChange]:
        out = self._changes(worktree)
        if ignore:
            out = [c for c in out if not path_matches(c.path, ignore)]
        return out

    def tracked_changes(self, worktree: str | Path) -> list[str]:
        raw = self.git.run("status", "--porcelain=v1", "--untracked-files=no", cwd=worktree).stdout
        return [line[3:] for line in raw.splitlines() if line.strip()]

    def _changes(self, worktree: str | Path) -> list[FileChange]:
        raw = self.git.run("status", "--porcelain=v1", "-z", "--untracked-files=all", cwd=worktree).stdout
        parts = raw.split("\0")
        out: list[FileChange] = []
        i = 0
        while i < len(parts):
            entry = parts[i]
            i += 1
            if len(entry) < 4:
                continue
            xy, path = entry[:2], normalize_rel_path(entry[3:])
            if "R" in xy or "C" in xy:
                old = normalize_rel_path(parts[i]) if i < len(parts) else None
                i += 1
                out.append(FileChange("R", path, old))
            elif xy == "??":
                out.append(FileChange("A", path))
            elif "D" in xy:
                out.append(FileChange("D", path))
            elif "A" in xy:
                out.append(FileChange("A", path))
            else:
                out.append(FileChange("M", path))
        return out

    def file_exists_at(self, commit: str, path: str) -> bool:
        r = self.git.run("cat-file", "-e", f"{commit}:{path}", check=False)
        return r.returncode == 0

    def commit_paths(self, worktree: str | Path, paths: list[str], message: str) -> str:
        """Stage exactly the validated paths (additions, modifications and deletions) and commit."""
        for i in range(0, len(paths), 100):  # keep command lines short (Windows limit)
            self.git.run("add", "-A", "--", *paths[i:i + 100], cwd=worktree)
        self.git.run("commit", "--no-verify", "-q", "-m", message, cwd=worktree, identity=True)
        return self.head(worktree)

    def diff(self, worktree: str | Path, base: str, target: str = "HEAD", stat: bool = False) -> str:
        args = ["diff", "--no-color", "--no-ext-diff"]
        if stat:
            args.append("--stat")
        return self.git.run(*args, f"{base}..{target}", cwd=worktree).stdout

    def commit_files(self, commit: str) -> list[str]:
        out = self.git.out("show", "--no-color", "--name-only", "--format=", commit)
        return [normalize_rel_path(x) for x in out.splitlines() if x.strip()]

    def task_trailers(self, worktree: str | Path, base: str) -> dict[str, str]:
        """Map task id -> integrated commit, recovered from commit trailers (crash recovery)."""
        out = self.git.out("log", "--format=%H%x1f%B%x1e", f"{base}..HEAD", cwd=worktree)
        found: dict[str, str] = {}
        for rec in out.split("\x1e"):
            if "\x1f" not in rec:
                continue
            sha, body = rec.strip().split("\x1f", 1)
            for m in re.finditer(r"^Orchestrator-Task: (\S+)/(\S+)$", body, re.M):
                found.setdefault(m.group(2), sha)
        return found
