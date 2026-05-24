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
GENERATOR = ROOT / "examples" / "benchmark" / "generate_synthetic_phase1_lmdb.py"
BENCHMARK = ROOT / "examples" / "benchmark" / "benchmark_reverse_csr_lookup.py"
I32 = struct.Struct("<i")


class GenerateSyntheticPhase1LmdbTest(unittest.TestCase):
    def test_generator_writes_phase1_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "phase1.lmdb"
            result = subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    str(out_dir),
                    "--parents",
                    "3",
                    "--children-per-parent",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            manifest = json.loads((out_dir / "phase1_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "unifyweaver.synthetic_phase1_lmdb.v1")
            self.assertEqual(manifest["id_encoding"], "int32_le")
            self.assertEqual(manifest["parents"], 3)
            self.assertEqual(manifest["children_per_parent"], 2)
            self.assertEqual(manifest["edge_count"], 6)

            env = lmdb.open(str(out_dir), readonly=True, max_dbs=8, lock=False, subdir=True)
            try:
                with env.begin() as txn:
                    cc_db = env.open_db(b"category_child", txn=txn, dupsort=True, create=False)
                    cursor = txn.cursor(db=cc_db)
                    rows = [
                        (I32.unpack(k)[0], I32.unpack(v)[0])
                        for k, v in cursor
                    ]
                    self.assertEqual(rows, [(1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (3, 9)])
            finally:
                env.close()

    def test_generated_fixture_runs_reverse_csr_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            csr_dir = tmp_path / "csr"
            subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    str(phase1),
                    "--parents",
                    "8",
                    "--children-per-parent",
                    "4",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARK),
                    str(phase1),
                    "--csr-dir",
                    str(csr_dir),
                    "--sample-parents",
                    "5",
                    "--iterations",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("backend\tsample_parents\titerations", result.stdout)
            self.assertIn("csr_artifact_bytes\tparent_lmdb_env_bytes\tphase1_lmdb_env_bytes", result.stdout)
            self.assertIn("csr\t5\t2\t20\t", result.stdout)
            self.assertIn("lmdb\t5\t2\t20\t", result.stdout)


if __name__ == "__main__":
    unittest.main()
