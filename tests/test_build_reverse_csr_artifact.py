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
BUILDER = ROOT / "examples" / "benchmark" / "build_reverse_csr_artifact.py"
I32 = struct.Struct("<i")
IDX_RECORD = struct.Struct("<iQI")


def i32(value: int) -> bytes:
    return I32.pack(value)


def read_idx(path: Path) -> list[tuple[int, int, int]]:
    data = path.read_bytes()
    return [
        IDX_RECORD.unpack_from(data, offset)
        for offset in range(0, len(data), IDX_RECORD.size)
    ]


def read_i32_values(path: Path) -> list[int]:
    data = path.read_bytes()
    return [
        I32.unpack_from(data, offset)[0]
        for offset in range(0, len(data), I32.size)
    ]


class BuildReverseCsrArtifactTest(unittest.TestCase):
    def test_builder_writes_parent_sorted_csr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"

            phase1.mkdir()
            env = lmdb.open(str(phase1), map_size=1 << 20, max_dbs=8, subdir=True)
            cp_db = env.open_db(b"category_parent", dupsort=True)
            with env.begin(write=True) as txn:
                txn.put(i32(10), i32(20), db=cp_db)
                txn.put(i32(12), i32(20), db=cp_db)
                txn.put(i32(11), i32(30), db=cp_db)
                txn.put(i32(13), i32(20), db=cp_db)
            env.close()

            result = subprocess.run(
                [sys.executable, str(BUILDER), str(phase1), str(out_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            meta = json.loads((out_dir / "category_child.csr.meta").read_text(encoding="utf-8"))
            self.assertEqual(meta["format"], "unifyweaver.reverse_csr.v1")
            self.assertEqual(meta["relation"], "category_child/2")
            self.assertEqual(meta["source_relation"], "category_parent/2")
            self.assertEqual(meta["storage_kind"], "csr_pread_artifact")
            self.assertEqual(meta["id_encoding"], "int32_le")
            self.assertEqual(meta["ordering"], "parent_sort")
            self.assertEqual(meta["index_backend"], "sorted_array")
            self.assertEqual(meta["io_policy"], "buffered_pread")
            self.assertEqual(meta["parent_count"], 2)
            self.assertEqual(meta["edge_count"], 4)
            self.assertEqual(meta["index_record_bytes"], IDX_RECORD.size)
            self.assertEqual(meta["value_record_bytes"], I32.size)

            self.assertEqual(
                read_idx(out_dir / "category_child.csr.idx"),
                [
                    (20, 0, 3),
                    (30, 3, 1),
                ],
            )
            self.assertEqual(
                read_i32_values(out_dir / "category_child.csr.val"),
                [10, 12, 13, 11],
            )

    def test_builder_writes_lmdb_offset_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"

            phase1.mkdir()
            env = lmdb.open(str(phase1), map_size=1 << 20, max_dbs=8, subdir=True)
            cp_db = env.open_db(b"category_parent", dupsort=True)
            with env.begin(write=True) as txn:
                txn.put(i32(10), i32(20), db=cp_db)
                txn.put(i32(12), i32(20), db=cp_db)
                txn.put(i32(11), i32(30), db=cp_db)
            env.close()

            result = subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    str(phase1),
                    str(out_dir),
                    "--index-backend",
                    "lmdb_offset",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            meta = json.loads((out_dir / "category_child.csr.meta").read_text(encoding="utf-8"))
            self.assertEqual(meta["index_backend"], "lmdb_offset")
            self.assertEqual(meta["offset_index_path"], "category_child.csr.offsets.lmdb")
            self.assertGreater(meta["offset_index_bytes"], 0)

            offset_env = lmdb.open(str(out_dir / "category_child.csr.offsets.lmdb"), readonly=True, max_dbs=2, lock=False, subdir=True)
            try:
                with offset_env.begin() as txn:
                    offsets_db = offset_env.open_db(b"offsets", txn=txn, create=False)
                    self.assertEqual(struct.unpack("<QI", txn.get(i32(20), db=offsets_db)), (0, 2))
                    self.assertEqual(struct.unpack("<QI", txn.get(i32(30), db=offsets_db)), (2, 1))
            finally:
                offset_env.close()

    def test_builder_refuses_existing_output_without_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"

            phase1.mkdir()
            out_dir.mkdir()
            env = lmdb.open(str(phase1), map_size=1 << 20, max_dbs=8, subdir=True)
            env.open_db(b"category_parent", dupsort=True)
            env.close()

            result = subprocess.run(
                [sys.executable, str(BUILDER), str(phase1), str(out_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 3)
            self.assertIn("pass --refresh", result.stderr)


if __name__ == "__main__":
    unittest.main()
