#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_effective_distance_matrix import parse_args, resolve_requested_targets  # noqa: E402
from benchmark_target_matrix import (  # noqa: E402
    KERNEL_TARGET_PAIRS,
    TARGETS,
    list_kernel_pairs_text,
    list_targets_text,
    resolve_targets,
)


class BenchmarkTargetMatrixTests(unittest.TestCase):
    def test_clojure_targets_are_registered(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["clojure-wam"],
        )

        self.assertEqual(
            targets,
            [
                "clojure-wam-accumulated",
                "clojure-wam-seeded",
                "clojure-wam-seeded-no-kernels",
                "clojure-wam-accumulated-no-kernels",
            ],
        )
        self.assertEqual(TARGETS["clojure-wam-accumulated"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-accumulated-no-kernels"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-seeded"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-seeded-no-kernels"].category, "hybrid-wam")

    def test_default_hybrid_wam_excludes_clojure_scaffolds(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["hybrid-wam"],
        )

        self.assertIn("go-wam-accumulated", targets)
        self.assertIn("haskell-interp-ffi", targets)
        self.assertIn("clojure-wam-accumulated", targets)
        self.assertIn("clojure-wam-accumulated-no-kernels", targets)
        self.assertIn("clojure-wam-seeded", targets)
        self.assertIn("clojure-wam-seeded-no-kernels", targets)

    def test_list_targets_includes_clojure_scaffold_set(self) -> None:
        text = list_targets_text()

        self.assertIn("clojure-wam-accumulated\thybrid-wam", text)
        self.assertIn("clojure-wam-accumulated-no-kernels\thybrid-wam", text)
        self.assertIn("clojure-wam-seeded\thybrid-wam", text)
        self.assertIn("clojure-wam-seeded-no-kernels\thybrid-wam", text)
        self.assertIn(
            "clojure-wam\tclojure-wam-accumulated,clojure-wam-seeded,"
            "clojure-wam-seeded-no-kernels,clojure-wam-accumulated-no-kernels",
            text,
        )
        self.assertIn("clojure-wam-scaffold\t", text)

    def test_effective_distance_runner_resolves_seeded_clojure_targets(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "benchmark_effective_distance_matrix.py",
                "--targets",
                "clojure-wam-seeded,prolog-accumulated",
            ]
            args = parse_args()
            self.assertEqual(resolve_requested_targets(args), ["clojure-wam-seeded", "prolog-accumulated"])
        finally:
            sys.argv = original_argv

    def test_kernel_pair_registry_covers_registered_wam_pairs(self) -> None:
        pairs = {(pair.family, pair.mode): pair for pair in KERNEL_TARGET_PAIRS}

        expected = {
            ("rust", "seeded"),
            ("rust", "accumulated"),
            ("go", "accumulated"),
            ("clojure", "seeded"),
            ("clojure", "accumulated"),
            ("haskell", "interpreter"),
            ("haskell", "lowered"),
        }
        self.assertEqual(set(pairs), expected)

        for pair in KERNEL_TARGET_PAIRS:
            self.assertIn(pair.kernels_target, TARGETS)
            self.assertIn(pair.no_kernels_target, TARGETS)

    def test_list_kernel_pairs_text_is_tsv(self) -> None:
        text = list_kernel_pairs_text()

        self.assertTrue(text.startswith("family\tmode\tkernels_target\tno_kernels_target\n"))
        self.assertIn(
            "rust\taccumulated\twam-rust-accumulated\twam-rust-accumulated-no-kernels",
            text,
        )
        self.assertIn(
            "haskell\tlowered\thaskell-lowered-ffi\thaskell-lowered-only",
            text,
        )

    def test_effective_distance_runner_accepts_kernel_pair_listing(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "benchmark_effective_distance_matrix.py",
                "--list-kernel-pairs",
            ]
            args = parse_args()
            self.assertTrue(args.list_kernel_pairs)
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
