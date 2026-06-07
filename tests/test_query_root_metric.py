#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""Unit test for examples/benchmark/query_root_metric.py.

Builds a tiny scoped LMDB (with the metric_min_dist_to_root sub-db) via
build_scoped_subtree_lmdb.py, then checks that query_root_metric.py answers
min_dist_to_root by lookup and that its --verify (fresh BFS oracle) passes.

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
    import lmdb  # noqa: F401
    HAVE_LMDB = True
except ImportError:
    HAVE_LMDB = False

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER = REPO_ROOT / "examples" / "benchmark" / "build_scoped_subtree_lmdb.py"
EFFDIST = REPO_ROOT / "examples" / "benchmark" / "build_effective_distance.py"
QUERY = REPO_ROOT / "examples" / "benchmark" / "query_root_metric.py"
I32 = struct.Struct("<i")


def enc(i):
    return I32.pack(i)


@unittest.skipUnless(HAVE_LMDB, "python-lmdb not installed")
class TestQueryRootMetric(unittest.TestCase):
    def setUp(self):
        import lmdb
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "full"
        self.out = Path(self.tmp.name) / "scoped"
        self.src.mkdir()
        # Same graph as the builder test: 2->{3,4}, 3->{5}; min dist to root 2
        # is 2:0, 3:1, 4:1, 5:2. Out-of-scope 9->8 dropped.
        cp_edges = [(3, 2), (4, 2), (5, 3), (9, 8)]
        cc_edges = [(2, 3), (2, 4), (3, 5), (8, 9)]
        env = lmdb.open(str(self.src), max_dbs=16, map_size=10 * 1024 * 1024, subdir=True)
        cp = env.open_db(b"category_parent", dupsort=True, create=True)
        cc = env.open_db(b"category_child", dupsort=True, create=True)
        with env.begin(write=True) as txn:
            for child, parent in cp_edges:
                txn.put(enc(child), enc(parent), db=cp, dupdata=True)
            for parent, child in cc_edges:
                txn.put(enc(parent), enc(child), db=cc, dupdata=True)
        env.sync()
        env.close()
        # Build the scoped DB (default --min-dist on).
        env_os = dict(os.environ)
        env_os.setdefault("LANG", "C.UTF-8")
        proc = subprocess.run(
            [sys.executable, str(BUILDER), "--src", str(self.src), "--root", "2",
             "--out", str(self.out), "--max-depth", "10"],
            capture_output=True, text=True, env=env_os)
        self.assertEqual(proc.returncode, 0, f"builder failed:\n{proc.stderr}")

    def tearDown(self):
        self.tmp.cleanup()

    def _query(self, *extra):
        env_os = dict(os.environ)
        env_os.setdefault("LANG", "C.UTF-8")
        return subprocess.run(
            [sys.executable, str(QUERY), "--lmdb", str(self.out), *extra],
            capture_output=True, text=True, env=env_os)

    def test_lookup_distances(self):
        proc = self._query("2", "3", "4", "5", "9")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = dict(line.split("\t") for line in proc.stdout.splitlines() if "\t" in line)
        self.assertEqual(rows["2"], "0")
        self.assertEqual(rows["3"], "1")
        self.assertEqual(rows["4"], "1")
        self.assertEqual(rows["5"], "2")
        self.assertEqual(rows["9"], "unreachable")  # out of scope

    def test_verify_passes(self):
        proc = self._query("--verify")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("verify: OK", proc.stderr)

    def test_histogram(self):
        proc = self._query("--histogram")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        hist = dict(line.split("\t") for line in proc.stdout.splitlines() if "\t" in line)
        # distances 0,1,2 with counts 1,2,1
        self.assertEqual(hist.get("0"), "1")
        self.assertEqual(hist.get("1"), "2")
        self.assertEqual(hist.get("2"), "1")

    def _build_effective_distance(self, exponent="5"):
        env_os = dict(os.environ)
        env_os.setdefault("LANG", "C.UTF-8")
        proc = subprocess.run(
            [sys.executable, str(EFFDIST), "--lmdb", str(self.out),
             "--exponent", exponent],
            capture_output=True, text=True, env=env_os)
        self.assertEqual(proc.returncode, 0, f"build_effective_distance failed:\n{proc.stderr}")

    def test_effective_distance_lookup(self):
        # Scoped category_parent: 3->2, 4->2, 5->3. With root 2 and exponent 5:
        #   S(3) = S(4) = 2^-5 = 0.03125 ; S(5) = 3^-5 = 1/243.
        self._build_effective_distance("5")
        proc = self._query("--metric", "effective_distance", "3", "4", "5", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = dict(line.split("\t") for line in proc.stdout.splitlines() if "\t" in line)
        self.assertAlmostEqual(float(rows["3"]), 2 ** -5, places=12)
        self.assertAlmostEqual(float(rows["4"]), 2 ** -5, places=12)
        self.assertAlmostEqual(float(rows["5"]), 3 ** -5, places=12)
        # root has no path to itself in this DAG -> absent
        self.assertEqual(rows["2"], "unreachable")

    def test_effective_distance_verify(self):
        self._build_effective_distance("5")
        proc = self._query("--metric", "effective_distance", "--verify")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("verify: OK", proc.stderr)

    def test_effective_distance_histogram_is_summary(self):
        # f64 metric -> summary stats, not a value/count histogram.
        self._build_effective_distance("5")
        proc = self._query("--metric", "effective_distance", "--histogram")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = dict(line.split("\t") for line in proc.stdout.splitlines() if "\t" in line)
        self.assertEqual(rows.get("count"), "3")  # nodes 3,4,5


if __name__ == "__main__":
    unittest.main()
