#!/usr/bin/env python3
"""Materialize real category-title lookup sub-dbs inside a Phase-1 category LMDB.

The graph sub-dbs (`category_parent`, `category_child`) stay unsigned int32 page/category IDs for fast traversal. This
adds separate title lookup tables for human-readable consumers:

  title_i2s  uint32_le page_id -> UTF-8 category title
  title_s2i  UTF-8 category title -> uint32_le page_id

It reads a gzip-compressed MediaWiki `page` SQL dump, keeps namespace 14 rows whose page_id already appears in the
LMDB graph, and writes only the title layer. Existing graph data is not rewritten. `--map-size-gib` is LMDB's maximum
map size; on sparse-file filesystems it does not immediately allocate that much disk, but Windows filesystems may be
less forgiving, so increase it only as needed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

from lmdb_id import dec_id, enc_id

CATEGORY_NS = 14


def parse_values(sql_values):
    i, n = 0, len(sql_values)
    while i < n:
        if sql_values[i] != "(":
            i += 1
            continue
        i += 1
        fields, cur, in_str = [], [], False
        while i < n:
            ch = sql_values[i]
            if in_str:
                if ch == "\\":
                    cur.append(sql_values[i + 1] if i + 1 < n else "")
                    i += 2
                    continue
                if ch == "'":
                    if i + 1 < n and sql_values[i + 1] == "'":
                        cur.append("'")
                        i += 2
                        continue
                    in_str = False
                    i += 1
                    continue
                cur.append(ch)
                i += 1
            else:
                if ch == "'":
                    in_str = True
                    i += 1
                elif ch == ",":
                    fields.append("".join(cur))
                    cur = []
                    i += 1
                elif ch == ")":
                    fields.append("".join(cur))
                    i += 1
                    break
                else:
                    cur.append(ch)
                    i += 1
        yield fields
        while i < n and sql_values[i] != "(":
            i += 1


def iter_page_titles(page_dump):
    try:
        with gzip.open(page_dump, "rt", encoding="utf-8", errors="strict") as f:
            for line in f:
                if not line.startswith("INSERT INTO"):
                    continue
                pos = line.find("VALUES")
                if pos < 0:
                    continue
                for row in parse_values(line[pos + 6:]):
                    if len(row) < 3:
                        continue
                    try:
                        page_id = int(row[0])
                        ns = int(row[1])
                    except ValueError:
                        continue
                    if ns == CATEGORY_NS and row[2]:
                        yield page_id, row[2]
    except gzip.BadGzipFile as exc:
        raise ValueError(f"--page-dump must be a gzip-compressed MediaWiki page SQL dump: {page_dump}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"--page-dump is not valid UTF-8 at byte {exc.start}: {page_dump}") from exc


def graph_node_ids(env, cp_db):
    """Return every node appearing as either child key or any duplicate parent value."""
    nodes = set()
    with env.begin(buffers=True) as txn:
        cur = txn.cursor(db=cp_db)
        try:
            if not cur.first():
                return nodes
            while True:
                nodes.add(dec_id(cur.key()))
                nodes.add(dec_id(cur.value()))
                for value in cur.iternext_dup(keys=False, values=True):
                    nodes.add(dec_id(value))
                if not cur.next_nodup():
                    break
        finally:
            close = getattr(cur, "close", None)
            if close is not None:
                close()
    return nodes


def write_manifest(path, manifest):
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(out_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def abort_quietly(txn):
    if txn is None:
        return
    try:
        txn.abort()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb-dir", required=True, type=Path)
    ap.add_argument("--page-dump", required=True, type=Path)
    ap.add_argument("--title-i2s-db", default="title_i2s")
    ap.add_argument("--title-s2i-db", default="title_s2i")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--map-size-gib", type=float, default=16.0)
    args = ap.parse_args()

    try:
        import lmdb
    except ImportError as exc:
        raise SystemExit("python-lmdb is required") from exc

    env = lmdb.open(str(args.lmdb_dir), max_dbs=32, map_size=int(args.map_size_gib * 1024**3), subdir=True)
    cp = env.open_db(b"category_parent", create=False)
    title_i2s = env.open_db(args.title_i2s_db.encode("utf-8"), create=True)
    title_s2i = env.open_db(args.title_s2i_db.encode("utf-8"), create=True)
    meta = env.open_db(b"meta", create=True)

    with env.begin(write=True) as txn:
        txn.drop(title_i2s, delete=False)
        txn.drop(title_s2i, delete=False)

    nodes = graph_node_ids(env, cp)
    matched = 0
    category_title_rows_scanned = 0
    collisions = []
    txn = None
    try:
        txn = env.begin(write=True)
        for page_id, title in iter_page_titles(args.page_dump):
            category_title_rows_scanned += 1
            if page_id not in nodes:
                continue
            title_bytes = title.encode("utf-8")
            page_bytes = enc_id(page_id)
            existing = txn.get(title_bytes, db=title_s2i)
            if existing is not None and bytes(existing) != page_bytes:
                collisions.append((title, dec_id(existing), page_id))
                if len(collisions) >= 10:
                    break
                continue
            txn.put(page_bytes, title_bytes, db=title_i2s)
            txn.put(title_bytes, page_bytes, db=title_s2i)
            matched += 1
            if matched % 500_000 == 0:
                txn.commit()
                txn = None
                print(f"materialized {matched} category titles")
                txn = env.begin(write=True)
        if collisions:
            examples = "; ".join(f"{title}: {old_id} vs {new_id}" for title, old_id, new_id in collisions[:5])
            raise ValueError(f"duplicate category titles with different page IDs in page dump: {examples}")
        txn.put(b"title_layer_kind", b"mediawiki_page_titles", db=meta)
        txn.put(b"title_i2s_db", args.title_i2s_db.encode("utf-8"), db=meta)
        txn.put(b"title_s2i_db", args.title_s2i_db.encode("utf-8"), db=meta)
        txn.put(b"title_layer_count", str(matched).encode("ascii"), db=meta)
        txn.put(b"title_layer_refreshed", b"1", db=meta)
        txn.commit()
        txn = None
    except lmdb.MapFullError as exc:
        abort_quietly(txn)
        txn = None
        raise SystemExit("LMDB map is full while writing title tables; rerun with a larger --map-size-gib") from exc
    except BaseException:
        abort_quietly(txn)
        txn = None
        raise
    finally:
        env.sync()
        env.close()

    manifest = {
        "lmdb_dir": str(args.lmdb_dir),
        "page_dump": str(args.page_dump),
        "graph_node_count": len(nodes),
        "category_title_rows_scanned": category_title_rows_scanned,
        "title_layer_count": matched,
        "title_i2s_db": args.title_i2s_db,
        "title_s2i_db": args.title_s2i_db,
        "refreshed": True,
    }
    manifest_path = args.manifest or (args.lmdb_dir.parent / "title_layer_manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"materialized {matched} real category titles into {args.lmdb_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
