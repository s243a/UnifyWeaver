#!/usr/bin/env python3
"""Unit test for examples/benchmark/build_scoped_subtree_lmdb.py.

Builds a tiny full-graph LMDB with a known root subtree plus out-of-scope
nodes, runs the scoped-subtree builder, and asserts the scoped LMDB contains
exactly the reachable subtree (F# `reachableToRoot` + both-endpoints-in-demand
edge filter semantics).

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
BUILDER = REPO_ROOT / "examples" / "benchmark" / "build_scoped_subtree_lmdb.py"
I32 = struct.Struct("<i")


def enc(i):
    return I32.pack(i)


def dec(b):
    return I32.unpack(b)[0]


@unittest.skipUnless(HAVE_LMDB, "python-lmdb not installed")
class TestBuildScopedSubtreeLmdb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "full"
        self.out = Path(self.tmp.name) / "scoped"
        self.src.mkdir()
        # Graph: root 2. category_child (parent -> children): 2->{3,4}, 3->{5}.
        # category_parent (child -> parent): 3->2, 4->2, 5->3, and an
        # out-of-scope edge 9->8. Reachable-from-2 set = {2,3,4,5}; node 8/9
        # are unreachable and must be dropped.
        cp_edges = [(3, 2), (4, 2), (5, 3), (9, 8)]   # child -> parent
        cc_edges = [(2, 3), (2, 4), (3, 5), (8, 9)]   # parent -> child
        env = lmdb.open(str(self.src), max_dbs=16, map_size=10 * 1024 * 1024,
                        subdir=True)
        cp = env.open_db(b"category_parent", dupsort=True, create=True)
        cc = env.open_db(b"category_child", dupsort=True, create=True)
        i2s = env.open_db(b"i2s", create=True)
        s2i = env.open_db(b"s2i", create=True)
        with env.begin(write=True) as txn:
            for child, parent in cp_edges:
                txn.put(enc(child), enc(parent), db=cp, dupdata=True)
            for parent, child in cc_edges:
                txn.put(enc(parent), enc(child), db=cc, dupdata=True)
            for nid in (2, 3, 4, 5, 8, 9):
                txn.put(enc(nid), str(nid).encode(), db=i2s)
                txn.put(str(nid).encode(), enc(nid), db=s2i)
        env.sync()
        env.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_builder(self, root=2, max_depth=10):
        env = dict(os.environ)
        env.setdefault("LANG", "C.UTF-8")
        proc = subprocess.run(
            [sys.executable, str(BUILDER),
             "--src", str(self.src), "--root", str(root),
             "--out", str(self.out), "--max-depth", str(max_depth)],
            capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0,
                         f"builder failed:\n{proc.stderr}")
        return proc

    def _read_scoped(self):
        env = lmdb.open(str(self.out), max_dbs=16, readonly=True, subdir=True,
                        lock=False)
        cp = env.open_db(b"category_parent", dupsort=True, create=False)
        cc = env.open_db(b"category_child", dupsort=True, create=False)
        parent_edges, child_edges, nodes = set(), set(), set()
        with env.begin() as txn:
            for k, v in txn.cursor(db=cp):
                parent_edges.add((dec(k), dec(v)))
                nodes.add(dec(k))
                nodes.add(dec(v))
            for k, v in txn.cursor(db=cc):
                child_edges.add((dec(k), dec(v)))
        env.close()
        return parent_edges, child_edges, nodes

    def test_scoped_contains_only_subtree(self):
        self._run_builder()
        parent_edges, child_edges, nodes = self._read_scoped()
        # child->parent edges with both endpoints in {2,3,4,5}
        self.assertEqual(parent_edges, {(3, 2), (4, 2), (5, 3)})
        # reverse edges parent->child
        self.assertEqual(child_edges, {(2, 3), (2, 4), (3, 5)})
        # node set is exactly the reachable subtree; 8 and 9 excluded
        self.assertEqual(nodes, {2, 3, 4, 5})
        self.assertNotIn(8, nodes)
        self.assertNotIn(9, nodes)

    def test_depth_bound_prunes(self):
        # max_depth=1 keeps only root's direct children: {2,3,4}; 5 (depth 2)
        # drops, so edge 5->3 is excluded.
        self._run_builder(max_depth=1)
        parent_edges, _child_edges, nodes = self._read_scoped()
        self.assertEqual(nodes, {2, 3, 4})
        self.assertEqual(parent_edges, {(3, 2), (4, 2)})

    def test_scoped_i2s_restricted_to_demand(self):
        self._run_builder()
        env = lmdb.open(str(self.out), max_dbs=16, readonly=True, subdir=True,
                        lock=False)
        i2s = env.open_db(b"i2s", create=False)
        ids = set()
        with env.begin() as txn:
            for k, _v in txn.cursor(db=i2s):
                ids.add(dec(k))
        env.close()
        self.assertEqual(ids, {2, 3, 4, 5})

    def test_scoped_meta_marker(self):
        # The `meta` marker lets the runtime auto-detect a pre-scoped DB
        # (WAM_DEMAND=auto) and skip the redundant reachable_to_root BFS.
        self._run_builder(root=2, max_depth=7)
        env = lmdb.open(str(self.out), max_dbs=16, readonly=True, subdir=True,
                        lock=False)
        meta = env.open_db(b"meta", create=False)
        with env.begin() as txn:
            scoped = txn.get(b"scoped", db=meta)
            root = txn.get(b"scoped_root", db=meta)
            depth = txn.get(b"scoped_max_depth", db=meta)
        env.close()
        self.assertEqual(scoped, b"1")
        self.assertEqual(root, b"2")
        self.assertEqual(depth, b"7")


if __name__ == "__main__":
    unittest.main()
