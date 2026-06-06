#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""Equivalence test for examples/benchmark/build_effective_distance.py.

On a DAG the length-bucketed node-DP must equal the brute-force sum over simple
paths that the per-path category_ancestor kernel computes (ROOT_ANCHORED_METRICS
spec §5). Builds a tiny DAG LMDB, runs the tool, and compares the materialised
metric_effective_distance against an independent path-enumeration oracle.

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
TOOL = REPO_ROOT / "examples" / "benchmark" / "build_effective_distance.py"
I32 = struct.Struct("<i")
F64 = struct.Struct("<d")


def enc(i):
    return I32.pack(i)


# child -> [parents]; root = 1. Acyclic.
ADJ = {2: [1], 3: [1, 2], 4: [2, 3]}
ROOT = 1
MAX_DEPTH = 10
EXP = 2.0


def brute_S(seed):
    """Sum over simple paths seed->root of (L+1)^(-EXP), matching the kernel
    (hop = depth+1, distance = hop+1 = depth+2; per-path visited set)."""
    total = 0.0

    def dfs(node, depth, visited):
        nonlocal total
        parents = ADJ.get(node, [])
        if ROOT not in visited and ROOT in parents:
            total += (depth + 2) ** (-EXP)      # L = depth+1
        if len(visited) >= MAX_DEPTH:
            return
        for p in parents:
            if p in visited:
                continue
            dfs(p, depth + 1, visited | {p})

    dfs(seed, 0, {seed})
    return total


@unittest.skipUnless(HAVE_LMDB, "python-lmdb not installed")
class TestEffectiveDistanceNodeDP(unittest.TestCase):
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
             "--root", str(ROOT), "--max-depth", str(MAX_DEPTH), "--exponent", str(EXP)],
            capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def _read_metric(self):
        env = lmdb.open(str(self.lmdb), max_dbs=16, readonly=True, subdir=True, lock=False)
        m = env.open_db(b"metric_effective_distance", create=False)
        out = {}
        with env.begin() as txn:
            for k, v in txn.cursor(db=m):
                out[I32.unpack(k)[0]] = F64.unpack(v)[0]
        env.close()
        return out

    def test_nodedp_equals_path_enumeration(self):
        self._run_tool()
        got = self._read_metric()
        for node in (2, 3, 4):
            self.assertAlmostEqual(got.get(node, 0.0), brute_S(node), places=12,
                                   msg=f"S({node}) node-DP {got.get(node)} != oracle {brute_S(node)}")
        # root has no path to itself in a DAG -> absent / zero
        self.assertAlmostEqual(got.get(ROOT, 0.0), 0.0, places=12)

    def test_known_values(self):
        # S(2)=2^-2=0.25; S(3)=2^-2+3^-2; S(4)=2*3^-2+4^-2
        self._run_tool()
        got = self._read_metric()
        self.assertAlmostEqual(got[2], 0.25, places=12)
        self.assertAlmostEqual(got[3], 0.25 + 1/9, places=12)
        self.assertAlmostEqual(got[4], 2/9 + 1/16, places=12)


if __name__ == "__main__":
    unittest.main()
