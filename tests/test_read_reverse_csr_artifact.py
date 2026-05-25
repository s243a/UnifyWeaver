#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lmdb

from test_build_reverse_csr_artifact import BUILDER, ROOT, i32


READER = ROOT / "examples" / "benchmark" / "read_reverse_csr_artifact.py"
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))
from read_reverse_csr_artifact import ReverseCsrArtifact  # noqa: E402


def make_phase1_lmdb(path: Path) -> None:
    path.mkdir()
    env = lmdb.open(str(path), map_size=1 << 20, max_dbs=8, subdir=True)
    cp_db = env.open_db(b"category_parent", dupsort=True)
    cc_db = env.open_db(b"category_child", dupsort=True)
    with env.begin(write=True) as txn:
        for child, parent in [(10, 20), (12, 20), (11, 30), (13, 20)]:
            txn.put(i32(child), i32(parent), db=cp_db)
            txn.put(i32(parent), i32(child), db=cc_db)
    env.close()


class ReadReverseCsrArtifactTest(unittest.TestCase):
    def test_lookup_reads_children_for_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"
            make_phase1_lmdb(phase1)

            build = subprocess.run(
                [sys.executable, str(BUILDER), str(phase1), str(out_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)

            lookup = subprocess.run(
                [sys.executable, str(READER), "lookup", str(out_dir), "20"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(lookup.returncode, 0, lookup.stderr)
            self.assertEqual(lookup.stdout.splitlines(), ["10", "12", "13"])

            missing = subprocess.run(
                [sys.executable, str(READER), "lookup", str(out_dir), "99"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(missing.returncode, 0, missing.stderr)
            self.assertEqual(missing.stdout, "")

    def test_validate_compares_against_phase1_category_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"
            make_phase1_lmdb(phase1)

            build = subprocess.run(
                [sys.executable, str(BUILDER), str(phase1), str(out_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)

            validate = subprocess.run(
                [sys.executable, str(READER), "validate", str(out_dir), str(phase1)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr)
            self.assertEqual(validate.stdout.strip(), "validated parents=2 edges=4")

    def test_lookup_reads_lmdb_offset_index_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"
            make_phase1_lmdb(phase1)

            build = subprocess.run(
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
            self.assertEqual(build.returncode, 0, build.stderr)

            lookup = subprocess.run(
                [sys.executable, str(READER), "lookup", str(out_dir), "20"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(lookup.returncode, 0, lookup.stderr)
            self.assertEqual(lookup.stdout.splitlines(), ["10", "12", "13"])

    def test_artifact_keeps_values_file_open_until_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"
            make_phase1_lmdb(phase1)

            build = subprocess.run(
                [sys.executable, str(BUILDER), str(phase1), str(out_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)

            with ReverseCsrArtifact(out_dir) as artifact:
                self.assertFalse(artifact._values.closed)
                self.assertEqual(artifact.lookup(20), [10, 12, 13])

            self.assertTrue(artifact._values.closed)
            with self.assertRaisesRegex(ValueError, "CSR values file is closed"):
                artifact.lookup(20)

    def test_reader_rejects_unsupported_id_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            out_dir = tmp_path / "category_child_csr"
            make_phase1_lmdb(phase1)

            build = subprocess.run(
                [sys.executable, str(BUILDER), str(phase1), str(out_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)

            meta_path = out_dir / "category_child.csr.meta"
            meta_text = meta_path.read_text(encoding="utf-8")
            meta_path.write_text(meta_text.replace('"id_encoding": "int32_le"', '"id_encoding": "decimal_utf8"'), encoding="utf-8")

            lookup = subprocess.run(
                [sys.executable, str(READER), "lookup", str(out_dir), "20"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(lookup.returncode, 0)
            self.assertIn("unsupported CSR id_encoding", lookup.stderr)


if __name__ == "__main__":
    unittest.main()
