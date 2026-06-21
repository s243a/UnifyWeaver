#!/usr/bin/env python3
"""Small patch helper for Codex sessions.

Usage:
    python3 scripts/codex_patch.py check path/to/change.patch
    python3 scripts/codex_patch.py apply path/to/change.patch

The helper intentionally accepts a patch file instead of inline patch text so the
command prefix can be approved once while the diff remains inspectable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_patch(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise SystemExit(f"patch file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"patch path is not a file: {path}")
    return path


def run_git_apply(patch: Path, check_only: bool) -> int:
    cmd = ["git", "apply", "--recount", "--check" if check_only else "--index", str(patch)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Apply an inspectable unified diff through git apply.")
    parser.add_argument("mode", choices=["check", "apply"], help="Validate or stage-apply the patch.")
    parser.add_argument("patch", help="Path to a unified diff patch file.")
    args = parser.parse_args(argv)

    patch = resolve_patch(args.patch)
    return run_git_apply(patch, args.mode == "check")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
