#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""
read_reverse_csr_artifact.py — inspect and validate a reverse CSR artifact.

This is the first reader/probe for category_child CSR artifacts. It keeps
the access path deliberately simple: load the small parent index into
memory, binary-search it, then seek/read the corresponding child slice
from the values file.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import lmdb
except ImportError:
    sys.stderr.write("read_reverse_csr_artifact: 'lmdb' Python package required\n")
    sys.exit(1)


IDX_RECORD = struct.Struct("<iQI")
I32 = struct.Struct("<i")
EXPECTED_FORMAT = "unifyweaver.reverse_csr.v1"


class ReverseCsrArtifact:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.meta_path = artifact_dir / "category_child.csr.meta"
        self.meta = self._load_meta(self.meta_path)
        self.idx_path = artifact_dir / self.meta["index_path"]
        self.val_path = artifact_dir / self.meta["values_path"]
        self.index = self._load_index(self.idx_path)

    @staticmethod
    def _load_meta(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"missing CSR manifest: {path}")
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("format") != EXPECTED_FORMAT:
            raise ValueError(f"unsupported CSR format: {meta.get('format')!r}")
        if meta.get("relation") != "category_child/2":
            raise ValueError(f"unsupported CSR relation: {meta.get('relation')!r}")
        if meta.get("storage_kind") != "csr_pread_artifact":
            raise ValueError(f"unsupported CSR storage_kind: {meta.get('storage_kind')!r}")
        if meta.get("id_encoding") != "int32_le":
            raise ValueError(f"unsupported CSR id_encoding: {meta.get('id_encoding')!r}")
        if meta.get("ordering") != "parent_sort":
            raise ValueError(f"unsupported CSR ordering: {meta.get('ordering')!r}")
        if meta.get("index_record_bytes") != IDX_RECORD.size:
            raise ValueError("CSR index_record_bytes does not match reader format")
        if meta.get("value_record_bytes") != I32.size:
            raise ValueError("CSR value_record_bytes does not match reader format")
        return meta

    def _load_index(self, path: Path) -> list[tuple[int, int, int]]:
        data = path.read_bytes()
        if len(data) % IDX_RECORD.size != 0:
            raise ValueError("CSR index file size is not a whole number of records")
        records = [
            IDX_RECORD.unpack_from(data, offset)
            for offset in range(0, len(data), IDX_RECORD.size)
        ]
        if len(records) != self.meta["parent_count"]:
            raise ValueError("CSR parent_count does not match index record count")
        parents = [parent for parent, _offset_edges, _count in records]
        if parents != sorted(parents):
            raise ValueError("CSR index parents are not sorted")
        return records

    def lookup(self, parent: int) -> list[int]:
        lo = 0
        hi = len(self.index)
        while lo < hi:
            mid = (lo + hi) // 2
            mid_parent = self.index[mid][0]
            if mid_parent < parent:
                lo = mid + 1
            else:
                hi = mid
        if lo >= len(self.index) or self.index[lo][0] != parent:
            return []

        _parent, offset_edges, count = self.index[lo]
        byte_offset = offset_edges * I32.size
        byte_count = count * I32.size
        with self.val_path.open("rb") as values:
            values.seek(byte_offset)
            data = values.read(byte_count)
        if len(data) != byte_count:
            raise ValueError("CSR values file ended before requested child slice")
        return [I32.unpack_from(data, offset)[0] for offset in range(0, len(data), I32.size)]

    def parents(self) -> list[int]:
        return [parent for parent, _offset_edges, _count in self.index]


def lmdb_children_by_parent(phase1_lmdb_dir: Path) -> dict[int, list[int]]:
    env = lmdb.open(str(phase1_lmdb_dir), readonly=True, max_dbs=8, lock=False, subdir=True)
    result: dict[int, list[int]] = {}
    try:
        with env.begin() as txn:
            cc_db = env.open_db(b"category_child", txn=txn, dupsort=True, create=False)
            cursor = txn.cursor(db=cc_db)
            for key, value in cursor:
                if len(key) != 4 or len(value) != 4:
                    continue
                parent = I32.unpack(key)[0]
                child = I32.unpack(value)[0]
                result.setdefault(parent, []).append(child)
    finally:
        env.close()
    return {parent: sorted(children) for parent, children in result.items()}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read or validate a reverse CSR artifact.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lookup = subparsers.add_parser("lookup", help="print children for one parent id")
    lookup.add_argument("artifact_dir", type=Path)
    lookup.add_argument("parent_id", type=int)

    validate = subparsers.add_parser("validate", help="compare CSR against Phase 1 LMDB category_child")
    validate.add_argument("artifact_dir", type=Path)
    validate.add_argument("phase1_lmdb_dir", type=Path)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    artifact = ReverseCsrArtifact(args.artifact_dir)

    if args.command == "lookup":
        for child in artifact.lookup(args.parent_id):
            print(child)
        return 0

    if args.command == "validate":
        expected = lmdb_children_by_parent(args.phase1_lmdb_dir)
        csr_parents = set(artifact.parents())
        expected_parents = set(expected)
        if csr_parents != expected_parents:
            missing = sorted(expected_parents - csr_parents)[:10]
            extra = sorted(csr_parents - expected_parents)[:10]
            sys.stderr.write(f"parent key mismatch: missing={missing} extra={extra}\n")
            return 4
        for parent in sorted(expected):
            actual_children = artifact.lookup(parent)
            expected_children = expected[parent]
            if actual_children != expected_children:
                sys.stderr.write(
                    f"children mismatch for parent={parent}: "
                    f"actual={actual_children[:10]} expected={expected_children[:10]}\n"
                )
                return 5
        print(f"validated parents={len(expected)} edges={sum(len(v) for v in expected.values())}")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
