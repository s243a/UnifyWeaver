#!/usr/bin/env python3
"""Codegen guard for the WAM matrix-bench generator (Rust target).

The Rust matrix bench is a large Prolog-atom template; the only end-to-end
exercise of it (the LMDB cross-target conformance test) builds with cargo and is
opt-in, so template corruption otherwise reaches no CI guard. This test runs the
generator (swipl only, no cargo) and asserts the emitted main.rs contains the
WAM_DEMAND skip-demand-BFS wiring and that the lazy/cached transform still
splices in (a regression there would silently drop demand setup).

Skips gracefully if swipl is not on PATH.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "examples" / "benchmark" / "generate_wam_rust_matrix_benchmark.pl"
HAVE_SWIPL = shutil.which("swipl") is not None


@unittest.skipUnless(HAVE_SWIPL, "swipl not on PATH")
class TestMatrixDemandFlag(unittest.TestCase):
    def _generate(self, materialisation):
        out = Path(self.tmp.name) / materialisation
        env = dict(os.environ)
        env.setdefault("LANG", "C.UTF-8")
        proc = subprocess.run(
            ["swipl", "-q", "-s", str(GEN), "--",
             "dummy", str(out), "accumulated", "functions", "kernels_on",
             "cursor", "auto", materialisation, "297283"],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
        self.assertEqual(proc.returncode, 0,
                         f"generate ({materialisation}) failed:\n{proc.stderr}")
        main_rs = (out / "src" / "main.rs").read_text()
        return main_rs

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_eager_has_demand_flag_and_branch(self):
        src = self._generate("eager")
        self.assertIn('std::env::var("WAM_DEMAND")', src,
                      "WAM_DEMAND env read missing from eager bench")
        self.assertIn("demand_enabled", src)
        # both branches of the eager edge-load present
        self.assertIn("if demand_enabled {", src)
        self.assertIn("reachable_to_root(root_id, max_depth_limit)", src)

    def test_lazy_has_demand_flag_and_lazy_setup(self):
        src = self._generate("lazy")
        self.assertIn('std::env::var("WAM_DEMAND")', src)
        self.assertIn("demand_enabled", src)
        # the lazy materialisation transform still splices in: demand-set wiring
        # + the LazyCategoryParents struct must survive the rewrite.
        self.assertIn("set_demand_set", src)
        self.assertIn("LazyCategoryParents", src)
        self.assertIn("register_lazy_lookup", src)

    def test_cached_has_demand_flag_and_cache(self):
        src = self._generate("cached")
        self.assertIn('std::env::var("WAM_DEMAND")', src)
        self.assertIn("CachedLookup", src)
        self.assertIn("set_demand_set", src)

    def test_auto_branch_and_is_scoped(self):
        # WAM_DEMAND=auto consults the DB scoped marker via is_scoped().
        src = self._generate("lazy")
        self.assertIn('Ok("auto")', src)
        self.assertIn(".is_scoped()", src)
        # The is_scoped() definition lives in the emitted LmdbFactSource module.
        src_dir = Path(self.tmp.name) / "lazy" / "src"
        defs = [p.read_text() for p in src_dir.glob("*.rs")]
        self.assertTrue(any("pub fn is_scoped" in d for d in defs),
                        "is_scoped() definition missing from generated Rust")


if __name__ == "__main__":
    unittest.main()
