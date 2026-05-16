#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
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

    def test_lmdb_sink_input_uses_capped_page_id_edge_rows_without_header(self) -> None:
        self.assertEqual(
            MODULE.lmdb_sink_input([("10", "20"), ("12", "20"), ("10", "22")]),
            "10\t20\n12\t20\n10\t22\n",
        )

    def test_lmdb_consumer_env_uses_csharp_query_provider_compatible_defaults(self) -> None:
        env = MODULE.lmdb_consumer_env(lmdb_path=Path("/tmp/category_parent.lmdb"), map_size=1234)
        self.assertEqual(env["UW_LMDB_PATH"], "/tmp/category_parent.lmdb")
        self.assertEqual(env["UW_LMDB_MAP_SIZE"], "1234")
        self.assertEqual(env["UW_LMDB_DBNAME"], "main")
        self.assertEqual(env["UW_LMDB_DUPSORT"], "1")
        self.assertEqual(env["UW_KEY_COL"], "0")
        self.assertEqual(env["UW_VAL_COL"], "1")
        self.assertEqual(env["UW_KEY_ENCODING"], "utf8")
        self.assertEqual(env["UW_VAL_ENCODING"], "utf8")

    def test_write_lmdb_artifact_invokes_csharp_consumer_and_writes_manifest(self) -> None:
        calls = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="ingest_to_lmdb (csharp): in=2 written=2 skipped=0")

        original_which = MODULE.shutil.which
        MODULE.shutil.which = lambda name: "/usr/bin/dotnet" if name == "dotnet" else original_which(name)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fixture_dir = Path(tmp)
                (fixture_dir / "category_parent.tsv").write_text("child\tparent\n10\t20\n12\t22\n", encoding="utf-8")

                manifest_path = MODULE.write_lmdb_artifact(
                    fixture_dir=fixture_dir,
                    edges=[("10", "20"), ("12", "22")],
                    refresh=False,
                    map_size=4096,
                    run=fake_run,
                )

                self.assertEqual(manifest_path, fixture_dir / "category_parent.lmdb.manifest.json")
                self.assertEqual(len(calls), 1)
                command = calls[0][0][0]
                kwargs = calls[0][1]
                self.assertEqual(command[:2], ["dotnet", "run"])
                self.assertEqual(kwargs["input"], "10\t20\n12\t22\n")
                self.assertEqual(kwargs["env"]["UW_LMDB_PATH"], str(fixture_dir / "category_parent.lmdb"))
                self.assertEqual(kwargs["env"]["UW_LMDB_MAP_SIZE"], "4096")

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["Format"], "unifyweaver.lmdb_relation.v1")
                self.assertEqual(manifest["PredicateName"], "category_parent")
                self.assertEqual(manifest["EnvironmentPath"], "category_parent.lmdb")
                self.assertEqual(manifest["DatabaseName"], "main")
                self.assertTrue(manifest["DupSort"])
                self.assertEqual(manifest["KeyEncoding"], "utf8")
                self.assertEqual(manifest["ValueEncoding"], "utf8")
                self.assertEqual(manifest["RowCount"], 2)
                self.assertEqual(manifest["SourceLength"], (fixture_dir / "category_parent.tsv").stat().st_size)
        finally:
            MODULE.shutil.which = original_which

    def test_write_lmdb_artifact_reuses_existing_manifest_without_consumer(self) -> None:
        def fail_run(*args, **kwargs):
            raise AssertionError("consumer should not run when manifest and LMDB directory already exist")

        original_which = MODULE.shutil.which
        MODULE.shutil.which = lambda name: "/usr/bin/dotnet" if name == "dotnet" else original_which(name)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fixture_dir = Path(tmp)
                (fixture_dir / "category_parent.lmdb").mkdir()
                manifest_path = fixture_dir / "category_parent.lmdb.manifest.json"
                manifest_path.write_text("{}\n", encoding="utf-8")

                self.assertEqual(
                    MODULE.write_lmdb_artifact(
                        fixture_dir=fixture_dir,
                        edges=[("10", "20")],
                        refresh=False,
                        map_size=4096,
                        run=fail_run,
                    ),
                    manifest_path,
                )
        finally:
            MODULE.shutil.which = original_which


if __name__ == "__main__":
    unittest.main()
