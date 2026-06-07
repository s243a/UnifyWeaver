#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""Equivalence test for examples/benchmark/build_max_distance.py.

max_dist_to_root is the (max,+) longest-walk-to-root metric. On a DAG it must
equal the longest simple path (in parent hops) to the root. Builds a tiny
diamond DAG where the longest and shortest paths differ, runs the tool, and
compares the materialised metric_max_dist_to_root against an independent
longest-path oracle.

Skips gracefully if python-lmdb is not installed.
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import lmdb
    HAVE_LMDB = True
except ImportError:
    HAVE_LMDB = False

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "examples" / "benchmark" / "build_max_distance.py"
I32 = struct.Struct("<i")


def enc(i):
    return I32.pack(i)


# child -> [parents]; root = 1. Diamond: 4 reaches root both directly (len 1)
# and via 3->2->1 (len 3). Longest walk to root: 2->1, 3->2, 4->3.
ADJ = {2: [1], 3: [2], 4: [1, 3]}
ROOT = 1
MAX_DEPTH = 10


def brute_max(seed):
    """Longest simple path seed->root in parent hops (per-path visited set)."""
    best = -1

    def dfs(node, depth, visited):
        nonlocal best
        if node == ROOT:
            best = max(best, depth)
            return
        if len(visited) >= MAX_DEPTH:
            return
        for p in ADJ.get(node, []):
            if p in visited:
                continue
            dfs(p, depth + 1, visited | {p})

    dfs(seed, 0, {seed})
    return best


@unittest.skipUnless(HAVE_LMDB, "python-lmdb not installed")
class TestMaxDistanceNodeDP(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lmdb = Path(self.tmp.name) / "scoped"
        env = lmdb.open(str(self.lmdb), max_dbs=16, map_size=10 * 1024 * 1024, subdir=True)
        cp = env.open_db(b"category_parent", dupsort=True, create=True)
        meta = env.open_db(b"meta", create=True)
        with env.begin(write=True) as txn:
            for child, ps in ADJ.items():
                for p in ps:
                    txn.put(enc(child), enc(p), db=cp, dupdata=True)
            txn.put(b"scoped_root", str(ROOT).encode(), db=meta)
            txn.put(b"scoped_max_depth", str(MAX_DEPTH).encode(), db=meta)
        env.sync()
        env.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_tool(self):
        env = dict(os.environ)
        env.setdefault("LANG", "C.UTF-8")
        proc = subprocess.run(
            [sys.executable, str(TOOL), "--lmdb", str(self.lmdb),
             "--root", str(ROOT), "--max-depth", str(MAX_DEPTH)],
            capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def _read_metric(self):
        env = lmdb.open(str(self.lmdb), max_dbs=16, readonly=True, subdir=True, lock=False)
        m = env.open_db(b"metric_max_dist_to_root", create=False)
        out = {}
        with env.begin() as txn:
            for k, v in txn.cursor(db=m):
                out[I32.unpack(k)[0]] = I32.unpack(v)[0]
        env.close()
        return out

    def test_nodedp_equals_longest_path(self):
        self._run_tool()
        got = self._read_metric()
        for node in (2, 3, 4):
            self.assertEqual(got.get(node), brute_max(node),
                             msg=f"max_dist({node}) node-DP {got.get(node)} != oracle {brute_max(node)}")
        self.assertEqual(got.get(ROOT), 0)

    def test_known_values(self):
        # 2->1 (len 1); 3->2->1 (len 2); 4->3->2->1 (len 3, the long arm).
        self._run_tool()
        got = self._read_metric()
        self.assertEqual(got[2], 1)
        self.assertEqual(got[3], 2)
        self.assertEqual(got[4], 3)


if __name__ == "__main__":
    unittest.main()
