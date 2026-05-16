#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark" / "benchmark_csharp_query_rust_lmdb_sink.py"

SPEC = importlib.util.spec_from_file_location("benchmark_csharp_query_rust_lmdb_sink", SCRIPT)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CSharpQueryRustLmdbSinkBenchmarkTests(unittest.TestCase):
    def test_prepare_command_uses_rust_lmdb_sink_target(self) -> None:
        command = MODULE.prepare_command(
            scale="rust_lmdb_10k",
            max_edges=10_000,
            dump_path=Path("dump.sql.gz"),
            output_root=Path("/tmp/bench"),
            map_size=1234,
            refresh_lmdb=True,
        )
        self.assertIn("--sink-lmdb", command)
        self.assertIn("--lmdb-sink-target", command)
        self.assertIn("rust", command)
        self.assertIn("--refresh-lmdb", command)
        self.assertIn("--lmdb-map-size", command)
        self.assertIn("1234", command)

    def test_benchmark_command_uses_lmdb_only_without_tsv_requirement(self) -> None:
        command = MODULE.benchmark_command(
            scale="rust_lmdb_10k",
            output_root=Path("/tmp/bench"),
            lookup_keys=8,
            lookup_repetitions=2,
        )
        self.assertIn("--benchmark-root", command)
        self.assertIn("/tmp/bench", command)
        self.assertIn("--use-scale-lmdb-artifact", command)
        self.assertIn("--lmdb-only", command)
        self.assertIn("--lookup-keys", command)
        self.assertIn("8", command)

    def test_lmdb_row_from_tsv_requires_one_lmdb_row(self) -> None:
        row = MODULE.lmdb_row_from_tsv(
            "scale\trun\tmode\trows\tdistinct_categories\tlookup_keys\tartifact_bytes\topen_ms\tlookup_ms\tbucket_ms\tscan_ms\tretained_bytes\tscan_hash\tlookup_hash\tbucket_hash\n"
            "s\t1\tlmdb\t2\t4\t2\t100\t0.1\t0.2\t0.3\t0.4\t10\ta\tb\tc\n"
        )
        self.assertEqual(row["mode"], "lmdb")
        self.assertEqual(row["rows"], "2")
        with self.assertRaisesRegex(RuntimeError, "expected one lmdb row"):
            MODULE.lmdb_row_from_tsv("scale\trun\tmode\ns\t1\tpreload\n")


if __name__ == "__main__":
    unittest.main()
