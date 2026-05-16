#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark" / "prepare_csharp_query_enwiki_category_fixture.py"

SPEC = importlib.util.spec_from_file_location("prepare_csharp_query_enwiki_category_fixture", SCRIPT)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareCSharpQueryEnwikiCategoryFixtureTests(unittest.TestCase):
    def test_edge_rows_from_mysql_stream_filters_subcat_and_caps(self) -> None:
        stream = StringIO(
            "10\t0\tA\t0\tsubcat\t0\t20\n"
            "11\t0\tB\t0\tpage\t0\t21\n"
            "12\t0\tC\t0\tsubcat\t0\t22\n"
            "13\t0\tD\t0\tsubcat\t0\t23\n"
        )
        rows, scanned = MODULE.edge_rows_from_mysql_stream(stream, 2)
        self.assertEqual(rows, [("10", "20"), ("12", "22")])
        self.assertEqual(scanned, 3)

    def test_prepare_from_stream_writes_fixture(self) -> None:
        stream = StringIO("10\t0\tA\t0\tsubcat\t0\t20\n12\t0\tC\t0\tsubcat\t0\t22\n")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = MODULE.prepare_from_stream(
                scale="500k_cats",
                max_edges=10,
                output_root=Path(tmp),
                stream=stream,
            )
            self.assertEqual((output_dir / "category_parent.tsv").read_text(encoding="utf-8").splitlines(), [
                "child\tparent",
                "10\t20",
                "12\t22",
            ])
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["scale"], "500k_cats")
            self.assertEqual(metadata["n_hierarchy_edges"], 2)
            self.assertEqual(metadata["mysql_rows_scanned"], 2)


if __name__ == "__main__":
    unittest.main()
