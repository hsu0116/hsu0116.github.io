"""Integrates validated task commits into the integration branch with ``git cherry-pick``.

The integration branch lives in its own worktree; the user's checkout and main are never touched.
On conflict the cherry-pick is aborted (the integration branch stays exactly as before) and the
conflicting files are reported. Conflicts are never auto-resolved by overwriting code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .worktree_manager import GitError, WorktreeManager


@dataclass
class IntegrationResult:
    ok: bool
    integrated_sha: str | None = None
    conflict_files: list[str] = field(default_factory=list)
    error: str | None = None
    overlap_warnings: list[str] = field(default_factory=list)


class Integrator:
    def __init__(self, wt: WorktreeManager, integration_worktree: str | Path):
        self.wt = wt
        self.path = str(integration_worktree)

    def recover(self) -> bool:
        """Abort a cherry-pick left half-done by a crash. Returns True if something was aborted."""
        git_dir = self.wt.git.out("rev-parse", "--git-dir", cwd=self.path)
        gd = Path(git_dir if Path(git_dir).is_absolute() else Path(self.path) / git_dir)
        if (gd / "CHERRY_PICK_HEAD").exists():
            self.wt.git.run("cherry-pick", "--abort", cwd=self.path, check=False)
            return True
        return False

    def head(self) -> str:
        return self.wt.head(self.path)

    def is_clean(self) -> bool:
        return not self.wt.tracked_changes(self.path)

    def cherry_pick(self, commit: str, integrated_files: dict[str, list[str]] | None = None,
                    task_id: str = "") -> IntegrationResult:
        res = IntegrationResult(ok=False)
        # pre-check: overlap with files already integrated by other tasks of this run
        if integrated_files:
            mine = set(self.wt.commit_files(commit))
            for other, files in integrated_files.items():
                common = mine & set(files)
                if common and other != task_id:
                    res.overlap_warnings.append(f"also modified by {other}: {sorted(common)}")
        if not self.is_clean():
            res.error = "integration worktree is not clean; refusing to cherry-pick"
            return res
        r = self.wt.git.run("cherry-pick", "-x", "--allow-empty", commit, cwd=self.path, check=False,
                            identity=True)
        if r.returncode == 0:
            res.ok = True
            res.integrated_sha = self.head()
            return res
        unmerged = self.wt.git.run("diff", "--name-only", "--diff-filter=U", cwd=self.path, check=False)
        res.conflict_files = [x for x in unmerged.stdout.splitlines() if x.strip()]
        res.error = (r.stderr.strip() or r.stdout.strip())[-1500:]
        abort = self.wt.git.run("cherry-pick", "--abort", cwd=self.path, check=False)
        if abort.returncode != 0:
            res.error += f"\n[cherry-pick --abort failed: {abort.stderr.strip()}]"
        return res

    def diff_since(self, base: str, stat: bool = False) -> str:
        try:
            return self.wt.diff(self.path, base, "HEAD", stat=stat)
        except GitError as e:
            return f"<diff unavailable: {e}>"
