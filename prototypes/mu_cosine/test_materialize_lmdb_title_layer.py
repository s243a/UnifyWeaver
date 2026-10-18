#!/usr/bin/env python3
"""Synthetic checks for LMDB title-layer materialization."""

import gzip
import json
import os
import tempfile
from unittest import mock

from lmdb_id import dec_id, enc_id
from materialize_lmdb_title_layer import main, parse_values


def _run_cli(argv):
    with mock.patch("sys.argv", argv):
        return main()


def test_parse_values_handles_quoted_commas_and_parens():
    rows = list(parse_values("(1,14,'A,B'),(2,14,'C)D'),(3,0,'Article');"))
    assert rows == [["1", "14", "A,B"], ["2", "14", "C)D"], ["3", "0", "Article"]]


def test_materializer_writes_real_titles_and_refreshes_stale_keys():
    try:
        import lmdb
    except ImportError:
        print("  skip test_materializer_writes_real_titles_and_refreshes_stale_keys (python-lmdb unavailable)")
        return

    with tempfile.TemporaryDirectory() as td:
        lmdb_dir = os.path.join(td, "graph.lmdb")
        page_dump = os.path.join(td, "page.sql.gz")
        manifest = os.path.join(td, "manifest.json")
        env = lmdb.open(lmdb_dir, max_dbs=16, map_size=64 * 1024 * 1024, subdir=True)
        cp = env.open_db(b"category_parent", dupsort=True, create=True)
        title_i2s = env.open_db(b"title_i2s", create=True)
        title_s2i = env.open_db(b"title_s2i", create=True)
        with env.begin(write=True) as txn:
            txn.put(enc_id(2), enc_id(1), db=cp, dupdata=True)
            txn.put(enc_id(2), enc_id(4), db=cp, dupdata=True)
            txn.put(enc_id(3), enc_id(1), db=cp, dupdata=True)
            txn.put(enc_id(99), b"Stale", db=title_i2s)
            txn.put(b"Stale", enc_id(99), db=title_s2i)
        env.sync()
        env.close()

        with gzip.open(page_dump, "wt", encoding="utf-8") as f:
            f.write(
                "INSERT INTO `page` VALUES "
                "(1,14,'Root'),(2,14,'Child'),(3,0,'Article'),(4,14,'Second_dup_parent'),"
                "(999,14,'Out_of_graph');\n"
            )

        _run_cli([
            "materialize_lmdb_title_layer.py",
            "--lmdb-dir", lmdb_dir,
            "--page-dump", page_dump,
            "--manifest", manifest,
            "--map-size-gib", "0.1",
        ])

        env = lmdb.open(lmdb_dir, max_dbs=16, readonly=True, lock=False, subdir=True)
        title_i2s = env.open_db(b"title_i2s", create=False)
        title_s2i = env.open_db(b"title_s2i", create=False)
        meta = env.open_db(b"meta", create=False)
        with env.begin() as txn:
            assert txn.get(enc_id(1), db=title_i2s) == b"Root"
            assert txn.get(enc_id(2), db=title_i2s) == b"Child"
            assert txn.get(enc_id(3), db=title_i2s) is None
            assert txn.get(enc_id(4), db=title_i2s) == b"Second_dup_parent"
            assert txn.get(enc_id(99), db=title_i2s) is None
            assert dec_id(txn.get(b"Root", db=title_s2i)) == 1
            assert dec_id(txn.get(b"Second_dup_parent", db=title_s2i)) == 4
            assert txn.get(b"Stale", db=title_s2i) is None
            assert txn.get(b"title_layer_kind", db=meta) == b"mediawiki_page_titles"
            assert txn.get(b"title_layer_count", db=meta) == b"3"
        env.close()

        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        assert m["graph_node_count"] == 4
        assert m["category_title_rows_scanned"] == 4
        assert m["title_layer_count"] == 3
        assert m["refreshed"] is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} LMDB title-layer tests passed")
