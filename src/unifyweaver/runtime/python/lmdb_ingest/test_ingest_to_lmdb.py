#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Unit tests for ingest_to_lmdb.

Covers:
- tsv_unescape round-trip with the parser's escape conventions
  (\\\\ \\t \\n \\r \\xNN, NULL as \\N).
- Integer-keyed (existing) mode is byte-equivalent to its prior behaviour.
- Text-keyed (new) intern mode writes s2i / i2s / meta sub-dbs and
  produces int32_le edge entries.
- Compile-time atoms sidecar reserves low IDs and survives collisions
  (input string literally equal to a reserved atom returns the reserved ID).
- Idempotence guard: rerunning without UW_FORCE_REINGEST exits non-zero.
"""

import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lmdb

# Direct import from the script's directory (sibling file).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import ingest_to_lmdb as ingest  # noqa: E402


SCRIPT = _HERE / "ingest_to_lmdb.py"


def run_ingest(stdin_text: str, env: dict) -> subprocess.CompletedProcess:
    """Spawn the script as a subprocess so env-var handling is exercised."""
    full_env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=stdin_text,
        capture_output=True,
        text=True,
        env=full_env,
    )


def le32(i: int) -> bytes:
    return struct.pack("<i", i)


class TestTsvUnescape(unittest.TestCase):
    def test_passthrough(self):
        self.assertEqual(ingest.tsv_unescape("hello"), "hello")
        self.assertEqual(ingest.tsv_unescape(""), "")

    def test_two_byte_escapes(self):
        self.assertEqual(ingest.tsv_unescape("a\\\\b"), "a\\b")
        self.assertEqual(ingest.tsv_unescape("a\\tb"), "a\tb")
        self.assertEqual(ingest.tsv_unescape("a\\nb"), "a\nb")
        self.assertEqual(ingest.tsv_unescape("a\\rb"), "a\rb")

    def test_null_marker_preserved(self):
        # \N is left as-is so callers can detect it.
        self.assertEqual(ingest.tsv_unescape("\\N"), "\\N")

    def test_hex_escape(self):
        # \x41 == 'A'
        self.assertEqual(ingest.tsv_unescape("a\\x41b"), "aAb")
        # \xff is non-ASCII
        self.assertEqual(ingest.tsv_unescape("\\xff"), "\xff")


class TestCompileTimeAtomsLoader(unittest.TestCase):
    def test_loads_in_order_skips_blanks(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("[]\n.\ntrue\n\nfail\n")
            path = f.name
        try:
            atoms = ingest.load_compile_time_atoms(path)
            self.assertEqual(atoms, ["[]", ".", "true", "fail"])
        finally:
            os.unlink(path)

    def test_duplicate_is_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("[]\ntrue\n[]\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                ingest.load_compile_time_atoms(path)
        finally:
            os.unlink(path)


class TestIntegerKeyedBackwardCompat(unittest.TestCase):
    """The existing UW_KEY_ENCODING=int32_le path should be unchanged."""

    def test_simple_int_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            lmdb_dir = Path(tmp) / "db"
            stdin_text = "1\t100\n2\t200\n3\t300\n"
            r = run_ingest(stdin_text, {
                "UW_LMDB_PATH": str(lmdb_dir),
                "UW_KEY_ENCODING": "int32_le",
                "UW_VAL_ENCODING": "int32_le",
                "UW_KEY_COL": "0",
                "UW_VAL_COL": "1",
            })
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            env = lmdb.open(str(lmdb_dir), readonly=True, max_dbs=4)
            with env.begin() as txn:
                self.assertEqual(txn.get(le32(1)), le32(100))
                self.assertEqual(txn.get(le32(2)), le32(200))
                self.assertEqual(txn.get(le32(3)), le32(300))
            env.close()


class TestTextKeyedInternMode(unittest.TestCase):
    """New text-keyed intern mode: s2i / i2s / meta sub-dbs."""

    def _ingest_three_pairs(self, lmdb_dir: Path, extra_env=None):
        stdin_text = "Cats\tFelines\nDogs\tCanines\nFelines\tMammals\n"
        env = {
            "UW_LMDB_PATH": str(lmdb_dir),
            "UW_LMDB_DBNAME": "category_parent",
            "UW_LMDB_DUPSORT": "1",
            "UW_KEY_COL": "0",
            "UW_VAL_COL": "1",
            "UW_INTERN_KEY": "1",
            "UW_INTERN_VAL": "1",
            "UW_LMDB_S2I_DB": "s2i",
            "UW_LMDB_I2S_DB": "i2s",
            "UW_LMDB_META_DB": "meta",
            "UW_LMDB_APPEND": "0",  # input not sorted by interned id
            "UW_SCHEMA_VERSION": "1",
        }
        if extra_env:
            env.update(extra_env)
        return run_ingest(stdin_text, env)

    def test_writes_intern_subdbs_and_int_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            lmdb_dir = Path(tmp) / "db"
            r = self._ingest_three_pairs(lmdb_dir)
            self.assertEqual(r.returncode, 0, msg=r.stderr)

            env = lmdb.open(str(lmdb_dir), readonly=True, max_dbs=8)
            s2i_db = env.open_db(b"s2i")
            i2s_db = env.open_db(b"i2s")
            meta_db = env.open_db(b"meta")
            edges_db = env.open_db(b"category_parent", dupsort=True)

            with env.begin() as txn:
                # s2i should contain all four unique strings.
                cats_id = txn.get(b"Cats", db=s2i_db)
                dogs_id = txn.get(b"Dogs", db=s2i_db)
                felines_id = txn.get(b"Felines", db=s2i_db)
                mammals_id = txn.get(b"Mammals", db=s2i_db)
                self.assertIsNotNone(cats_id)
                self.assertIsNotNone(dogs_id)
                self.assertIsNotNone(felines_id)
                self.assertIsNotNone(mammals_id)

                # i2s round-trip
                self.assertEqual(txn.get(cats_id, db=i2s_db), b"Cats")
                self.assertEqual(txn.get(felines_id, db=i2s_db), b"Felines")

                # Edges sub-db: keys are int32_le (id of "Cats"), values
                # are int32_le (id of "Felines"), etc.
                self.assertIsNotNone(txn.get(cats_id, db=edges_db))
                # dupsort means we can have multiple values per key.
                cur = txn.cursor(db=edges_db)
                cur.set_key(cats_id)
                vals = list(cur.iternext_dup())
                self.assertIn(felines_id, vals)

                # Meta keys
                self.assertEqual(txn.get(b"schema_version", db=meta_db), b"1")
                next_id_bytes = txn.get(b"next_id", db=meta_db)
                self.assertEqual(len(next_id_bytes), 4)
                next_id = struct.unpack("<i", next_id_bytes)[0]
                # Five unique strings: Cats, Felines, Dogs, Canines, Mammals.
                self.assertEqual(next_id, 5)

                # cli_args present
                self.assertIn(
                    b"ingest_to_lmdb",
                    txn.get(b"cli_args", db=meta_db),
                )
            env.close()

    def test_idempotence_guard_blocks_reingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            lmdb_dir = Path(tmp) / "db"
            r1 = self._ingest_three_pairs(lmdb_dir)
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)
            # Second run must refuse without UW_FORCE_REINGEST.
            r2 = self._ingest_three_pairs(lmdb_dir)
            self.assertEqual(r2.returncode, 3, msg=r2.stderr)
            self.assertIn("UW_FORCE_REINGEST", r2.stderr)

    def test_force_reingest_overrides_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            lmdb_dir = Path(tmp) / "db"
            r1 = self._ingest_three_pairs(lmdb_dir)
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)
            r2 = self._ingest_three_pairs(
                lmdb_dir, extra_env={"UW_FORCE_REINGEST": "1"}
            )
            self.assertEqual(r2.returncode, 0, msg=r2.stderr)


class TestCompileTimeAtomsCollision(unittest.TestCase):
    """Input row containing a string equal to a reserved compile-time
    atom must return the reserved low ID, not allocate a fresh one."""

    def test_reserved_id_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            lmdb_dir = Path(tmp) / "db"
            atoms_path = Path(tmp) / "atoms.txt"
            atoms_path.write_text("[]\n.\ntrue\nfail\n")

            # First column literally "true" — should map to reserved ID 2.
            stdin_text = "true\tCanines\nDogs\tCanines\n"
            env = {
                "UW_LMDB_PATH": str(lmdb_dir),
                "UW_LMDB_DBNAME": "category_parent",
                "UW_LMDB_DUPSORT": "1",
                "UW_KEY_COL": "0",
                "UW_VAL_COL": "1",
                "UW_INTERN_KEY": "1",
                "UW_INTERN_VAL": "1",
                "UW_LMDB_S2I_DB": "s2i",
                "UW_LMDB_I2S_DB": "i2s",
                "UW_LMDB_META_DB": "meta",
                "UW_COMPILE_TIME_ATOMS": str(atoms_path),
                "UW_SCHEMA_VERSION": "1",
            }
            r = run_ingest(stdin_text, env)
            self.assertEqual(r.returncode, 0, msg=r.stderr)

            env_h = lmdb.open(str(lmdb_dir), readonly=True, max_dbs=8)
            s2i_db = env_h.open_db(b"s2i")
            meta_db = env_h.open_db(b"meta")
            with env_h.begin() as txn:
                # "true" is at reserved ID 2 ([] = 0, . = 1, true = 2, fail = 3)
                self.assertEqual(txn.get(b"true", db=s2i_db), le32(2))
                # And compile_time_atoms_count should be 4
                cnt_bytes = txn.get(b"compile_time_atoms_count", db=meta_db)
                self.assertEqual(struct.unpack("<i", cnt_bytes)[0], 4)
                # "Dogs" got the next free ID (4 or above)
                dogs_id = struct.unpack("<i", txn.get(b"Dogs", db=s2i_db))[0]
                self.assertGreaterEqual(dogs_id, 4)
            env_h.close()


class TestInternModeRequiresAllSubDbs(unittest.TestCase):
    def test_missing_meta_db_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lmdb_dir = Path(tmp) / "db"
            r = run_ingest("a\tb\n", {
                "UW_LMDB_PATH": str(lmdb_dir),
                "UW_INTERN_KEY": "1",
                "UW_INTERN_VAL": "1",
                "UW_LMDB_S2I_DB": "s2i",
                "UW_LMDB_I2S_DB": "i2s",
                # UW_LMDB_META_DB intentionally omitted
            })
            self.assertEqual(r.returncode, 2)
            self.assertIn("UW_LMDB_META_DB", r.stderr)


if __name__ == "__main__":
    unittest.main()
