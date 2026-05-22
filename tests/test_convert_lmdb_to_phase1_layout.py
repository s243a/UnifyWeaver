#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lmdb


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "examples" / "benchmark" / "convert_lmdb_to_phase1_layout.py"


def i32(value: int) -> bytes:
    return struct.pack("<i", value)


class ConvertLmdbToPhase1LayoutTest(unittest.TestCase):
    def test_converter_writes_reverse_index_manifest_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src.lmdb"
            dst = tmp_path / "dst.lmdb"

            src.mkdir()
            env = lmdb.open(str(src), map_size=1 << 20, max_dbs=4, subdir=True)
            main_db = env.open_db(b"main", dupsort=True)
            with env.begin(write=True) as txn:
                txn.put(i32(10), i32(20), db=main_db)
                txn.put(i32(10), i32(30), db=main_db)
                txn.put(i32(11), i32(20), db=main_db)
            env.close()

            result = subprocess.run(
                [sys.executable, str(CONVERTER), str(src), str(dst)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            manifest = json.loads((dst / "phase1_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "unifyweaver.phase1_lmdb_manifest.v1")
            self.assertEqual(manifest["id_encoding"], "int32_le")
            self.assertEqual(manifest["relations"]["category_parent/2"]["edge_count"], 3)
            self.assertEqual(manifest["relations"]["category_child/2"]["storage_kind"], "phase1_lmdb_subdb")
            self.assertEqual(manifest["reverse_indexes"][0]["relation"], "category_child/2")
            self.assertEqual(manifest["reverse_indexes"][0]["edge_count"], 3)

            dst_env = lmdb.open(str(dst), readonly=True, max_dbs=8, lock=False)
            try:
                with dst_env.begin() as txn:
                    meta_db = dst_env.open_db(b"meta", txn=txn, create=False)
                    cc_db = dst_env.open_db(b"category_child", txn=txn, dupsort=True, create=False)
                    self.assertEqual(txn.get(b"id_encoding", db=meta_db), b"int32_le")
                    self.assertEqual(txn.get(b"category_child_edge_count", db=meta_db), b"3")
                    self.assertEqual(txn.get(b"reverse_index_relation", db=meta_db), b"category_child/2")
                    cursor = txn.cursor(db=cc_db)
                    self.assertEqual(
                        sorted((struct.unpack("<i", k)[0], struct.unpack("<i", v)[0]) for k, v in cursor),
                        [(20, 10), (20, 11), (30, 10)],
                    )
            finally:
                dst_env.close()


if __name__ == "__main__":
    unittest.main()
