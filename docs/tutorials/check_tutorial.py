#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# check_tutorial.py - verify that a tutorial's shown outputs are real.
#
# Extracts every ```console fenced block from a Markdown file, treats
# the first line (which must start with "$ ") as a shell command and
# the remaining lines as the expected output, executes the command
# with bash from the current working directory (run this from the
# repository root), and fails if the actual stdout differs or the
# command exits nonzero.
#
# A tutorial that drifts from the code therefore fails CI instead of
# silently rotting.  Comparison is exact per line after stripping the
# trailing newline run of each side.
#
# Usage:
#   python3 docs/tutorials/check_tutorial.py docs/tutorials/TUTORIAL_pattern_stache.md

import re
import subprocess
import sys

FENCE = re.compile(r"^```console\s*$")
END = re.compile(r"^```\s*$")


def extract_blocks(lines):
    """Yield (lineno, command, expected_output) per ```console block."""
    i = 0
    while i < len(lines):
        if FENCE.match(lines[i]):
            start = i + 1
            j = start
            while j < len(lines) and not END.match(lines[j]):
                j += 1
            block = lines[start:j]
            if not block or not block[0].startswith("$ "):
                raise SystemExit(
                    f"line {start + 1}: console block must start with '$ command'")
            command = block[0][2:]
            expected = "\n".join(block[1:])
            yield (start + 1, command, expected)
            i = j + 1
        else:
            i += 1


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} TUTORIAL.md")
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    blocks = list(extract_blocks(lines))
    if not blocks:
        raise SystemExit(f"{path}: no console blocks found - nothing verified")

    failures = 0
    for lineno, command, expected in blocks:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=300)
        actual = proc.stdout.rstrip("\n")
        want = expected.rstrip("\n")
        if proc.returncode != 0:
            failures += 1
            print(f"FAIL (exit {proc.returncode}) line {lineno}: {command}")
            sys.stdout.write(proc.stderr)
        elif actual != want:
            failures += 1
            print(f"FAIL (output) line {lineno}: {command}")
            print("  expected:")
            for l in want.split("\n"):
                print(f"    {l}")
            print("  actual:")
            for l in actual.split("\n"):
                print(f"    {l}")
        else:
            print(f"PASS line {lineno}: {command[:70]}{'...' if len(command) > 70 else ''}")

    print(f"{len(blocks) - failures}/{len(blocks)} tutorial blocks verified")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
