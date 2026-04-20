#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_effective_distance_matrix import parse_args, resolve_requested_targets  # noqa: E402
from benchmark_target_matrix import TARGETS, list_targets_text, resolve_targets  # noqa: E402


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
        self.assertTrue(
            all(
                TARGETS[target].category == "hybrid-wam-scaffold"
                for target in targets
                if target not in {"clojure-wam-accumulated", "clojure-wam-accumulated-no-kernels"}
            )
        )

    def test_default_hybrid_wam_excludes_clojure_scaffolds(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["hybrid-wam"],
        )

        self.assertIn("go-wam-accumulated", targets)
        self.assertIn("haskell-interp-ffi", targets)
        self.assertIn("clojure-wam-accumulated", targets)
        self.assertIn("clojure-wam-accumulated-no-kernels", targets)
        self.assertNotIn("clojure-wam-seeded", targets)

    def test_list_targets_includes_clojure_scaffold_set(self) -> None:
        text = list_targets_text()

        self.assertIn("clojure-wam-accumulated\thybrid-wam", text)
        self.assertIn("clojure-wam-accumulated-no-kernels\thybrid-wam", text)
        self.assertIn(
            "clojure-wam\tclojure-wam-accumulated,clojure-wam-seeded,"
            "clojure-wam-seeded-no-kernels,clojure-wam-accumulated-no-kernels",
            text,
        )
        self.assertIn(
            "clojure-wam-scaffold\tclojure-wam-seeded,clojure-wam-seeded-no-kernels",
            text,
        )

    def test_effective_distance_runner_skips_scaffold_targets(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "benchmark_effective_distance_matrix.py",
                "--targets",
                "clojure-wam-seeded,prolog-accumulated",
            ]
            args = parse_args()
            self.assertEqual(resolve_requested_targets(args), ["prolog-accumulated"])
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
