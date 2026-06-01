#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import lmdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_wam_c_child_csr_scale_sweep import (  # noqa: E402
    CSR_TARGETS,
    DEFAULT_SCALES,
    I32,
    IDX_RECORD,
    build_artifact_only_rows,
    category_id_map,
    generated_category_parent_tsv_bytes,
    matrix_command,
    read_tsv_column,
    read_tsv_pairs,
    write_reverse_csr_artifact,
)


class WamCChildCsrScaleSweepTests(unittest.TestCase):
    def test_matrix_command_uses_compile_only_csr_layout_targets(self) -> None:
        command = matrix_command("10x,1k")

        self.assertEqual(command[0], sys.executable)
        self.assertIn("benchmark_effective_distance_matrix.py", command[1])
        self.assertIn("--target-sets", command)
        self.assertEqual(command[command.index("--target-sets") + 1], "c-wam-child-csr-layouts")
        self.assertIn("--compile-only-targets", command)
        self.assertEqual(command[command.index("--compile-only-targets") + 1], ",".join(CSR_TARGETS))
        self.assertEqual(command[command.index("--baseline-target") + 1], "c-wam-accumulated-child-csr")

    def test_matrix_command_accepts_extra_matrix_args(self) -> None:
        command = matrix_command(DEFAULT_SCALES, ["--keep-temp"])

        self.assertEqual(command[-1], "--keep-temp")

    def test_category_id_map_matches_generator_sorting_surface(self) -> None:
        ids = category_id_map(
            [("ChildB", "Parent"), ("ChildA", "Parent")],
            [("Article", "ArticleOnly")],
            ["RootOnly"],
        )

        self.assertEqual(
            ids,
            {
                "ArticleOnly": 1,
                "ChildA": 2,
                "ChildB": 3,
                "Parent": 4,
                "RootOnly": 5,
            },
        )

    def test_artifact_only_csr_records_are_parent_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            ids = {"child_b": 1, "child_a": 2, "parent": 3, "other_parent": 4}

            parent_count, edge_count, category_count = write_reverse_csr_artifact(
                [("child_b", "parent"), ("child_a", "parent"), ("child_b", "other_parent")],
                ids,
                out_dir,
                "sorted_array",
            )

            self.assertEqual(parent_count, 2)
            self.assertEqual(edge_count, 3)
            self.assertEqual(category_count, 4)
            self.assertEqual(
                read_idx(out_dir / "category_child.csr.idx"),
                [
                    (3, 0, 2),
                    (4, 2, 1),
                ],
            )
            self.assertEqual(read_i32_values(out_dir / "category_child.csr.val"), [1, 2, 1])

    def test_artifact_only_lmdb_offsets_match_idx_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            ids = {"child_b": 1, "child_a": 2, "parent": 3}

            write_reverse_csr_artifact(
                [("child_b", "parent"), ("child_a", "parent")],
                ids,
                out_dir,
                "lmdb_offset",
            )

            offset_env = lmdb.open(
                str(out_dir / "category_child.csr.offsets.lmdb"),
                readonly=True,
                max_dbs=2,
                lock=False,
                subdir=True,
            )
            try:
                with offset_env.begin() as txn:
                    offsets_db = offset_env.open_db(b"offsets", txn=txn, create=False)
                    self.assertEqual(struct.unpack("<QI", txn.get(I32.pack(3), db=offsets_db)), (0, 2))
            finally:
                offset_env.close()

    def test_build_artifact_only_rows_reads_benchmark_tsvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            scale = "dev"

            rows = build_artifact_only_rows([scale], artifact_root)

            self.assertEqual([row.target for row in rows], CSR_TARGETS)
            self.assertTrue(all(row.scale == scale for row in rows))
            self.assertTrue(all(row.edge_count > 0 for row in rows))
            self.assertTrue(all(row.reverse_csr_index_bytes > 0 for row in rows))
            self.assertTrue(all(row.reverse_csr_values_bytes > 0 for row in rows))
            self.assertEqual(rows[0].reverse_csr_offsets_lmdb_bytes, 0)
            self.assertGreater(rows[-1].reverse_csr_offsets_lmdb_bytes, 0)

    def test_tsv_readers_skip_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pair_path = tmp_path / "pairs.tsv"
            column_path = tmp_path / "column.tsv"
            pair_path.write_text("left\tright\nA\tB\nC\tD\n", encoding="utf-8")
            column_path.write_text("value\nRoot\n", encoding="utf-8")

            self.assertEqual(read_tsv_pairs(pair_path), [("A", "B"), ("C", "D")])
            self.assertEqual(read_tsv_column(column_path), ["Root"])

    def test_generated_parent_tsv_size_excludes_source_header(self) -> None:
        self.assertEqual(generated_category_parent_tsv_bytes([("A", "B"), ("C", "D")]), 8)


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


if __name__ == "__main__":
    unittest.main()
