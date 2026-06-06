#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
enwiki_category_ingest_correct.py — proper 3-dump (Correct-mode) ingest of the
full English Wikipedia category graph into the Phase-1 LMDB layout.

WHY A DEDICATED ENWIKI SCRIPT (separate from simplewiki_category_ingest_text.pl):
simplewiki used the text-keyed Python-consumer path, which interns the raw
categorylinks columns. On the newer MediaWiki schema that leaves the graph in a
MIXED id space — `cl_from` is a page_id but `cl_target_id` is a *linktarget_id* —
so parent ids never match child ids and multi-hop traversal is broken. (Observed
on the prior enwiki fixture: Category:Main_topic_classifications had 0 children;
a high-out-degree admin hub looked like the "root".)

The fix is the 3-dump Correct mode in the mysql_stream_lmdb Rust binary, which
resolves  cl_target_id -> lt_title -> page_id  via the linktarget + page dumps,
producing a coherent, walkable page_id-keyed graph. This script orchestrates:

  1. mysql_stream_lmdb --mode correct  (categorylinks + linktarget + page)
       -> a fixture TSV of  child_page_id \t parent_page_id  (decimal).
  2. Build the Phase-1 LMDB from that TSV:
       category_parent  int32_le child  -> int32_le parent   (DUPSORT)
       category_child   int32_le parent -> int32_le child    (DUPSORT, reverse)
       s2i / i2s        real category titles <-> page_id      (--titles, default)
       meta             schema/version markers
  3. Emit a drop-in fixture: root_ids.txt (resolved from --root-title),
       article_category.tsv (node->node), metadata.json.

Output is consumed by build_scoped_subtree_lmdb.py and the WAM matrix benches.

Usage (defaults point at the gemini context dumps; override as needed):
  python3 examples/streaming/enwiki_category_ingest_correct.py \
      [--categorylinks <path>] [--linktarget <path>] [--page <path>] \
      [--out data/benchmark/enwiki_cats_correct] \
      [--root-title Main_topic_classifications] \
      [--titles | --no-titles] [--map-size-gib 16] [--keep-tmp]
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import lmdb

REPO_ROOT = Path(__file__).resolve().parents[2]
MYSQL_STREAM_DIR = REPO_ROOT / "src/unifyweaver/runtime/rust/mysql_stream"
LMDB_SINK = MYSQL_STREAM_DIR / "target/release/mysql_stream_lmdb"
MYSQL_STREAM = MYSQL_STREAM_DIR / "target/release/mysql_stream"
DUMP_DIR = REPO_ROOT / "context/gemini/UnifyWeaver/data/enwiki"
NS_CATEGORY = 14
I32 = struct.Struct("<i")


def enc(i):
    return I32.pack(i)


def page_title_to_id(page_dump: Path, want_title=None):
    """Stream the page dump via mysql_stream; yield (page_id, title) for ns=14.

    If want_title is given, returns just that page_id (int) or None, early-exit.
    Otherwise returns a dict {page_id: title}.
    """
    proc = subprocess.Popen([str(MYSQL_STREAM), str(page_dump)],
                            stdout=subprocess.PIPE)
    out = {}
    try:
        for raw in proc.stdout:
            cols = raw.split(b"\t")
            if len(cols) < 3 or cols[1] != b"14":
                continue
            try:
                pid = int(cols[0])
            except ValueError:
                continue
            title = cols[2].rstrip(b"\n")
            if want_title is not None:
                if title == want_title:
                    proc.kill()
                    return pid
            else:
                out[pid] = title.decode("utf-8", "replace")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return None if want_title is not None else out


def run_correct_ingest(args, tsv_path, tmp_lmdb):
    map_size = int(args.map_size_gib * 1024**3)
    cmd = [
        str(LMDB_SINK), str(args.categorylinks), str(tmp_lmdb),
        "--mode", "correct",
        "--linktarget-dump", str(args.linktarget),
        "--page-dump", str(args.page),
        "--map-size", str(map_size),
        "--batch-size", "200000",
        "--fixture-tsv", str(tsv_path),
        "--refresh",
    ]
    sys.stderr.write("[enwiki-correct] " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def build_phase1(tsv_path, out_dir, map_size, titles_map):
    """Build the Phase-1 LMDB from a child\tparent decimal TSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_dir), max_dbs=16, map_size=map_size, subdir=True)
    cp = env.open_db(b"category_parent", dupsort=True, create=True)
    cc = env.open_db(b"category_child", dupsort=True, create=True)
    s2i = env.open_db(b"s2i", create=True)
    i2s = env.open_db(b"i2s", create=True)
    meta = env.open_db(b"meta", create=True)
    nodes = set()
    edges = 0
    txn = env.begin(write=True)
    with open(tsv_path, "rb") as f:
        first = f.readline()  # header child\tparent
        for line in f:
            parts = line.split(b"\t")
            if len(parts) < 2:
                continue
            try:
                child = int(parts[0]); parent = int(parts[1])
            except ValueError:
                continue
            txn.put(enc(child), enc(parent), db=cp, dupdata=True)
            txn.put(enc(parent), enc(child), db=cc, dupdata=True)
            nodes.add(child); nodes.add(parent)
            edges += 1
            if edges % 1_000_000 == 0:
                txn.commit(); txn = env.begin(write=True)
                sys.stderr.write(f"[enwiki-correct] {edges} edges written\n")
    # intern tables: real titles when available, else identity.
    interned = 0
    for nid in nodes:
        title = titles_map.get(nid) if titles_map else None
        s = title if title is not None else str(nid)
        b = s.encode("utf-8")
        txn.put(b, enc(nid), db=s2i)
        txn.put(enc(nid), b, db=i2s)
        interned += 1
    txn.put(b"schema_version", b"1", db=meta)
    txn.put(b"ingest", b"enwiki_correct_3dump", db=meta)
    txn.commit()
    env.sync(); env.close()
    return len(nodes), edges, interned


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categorylinks", type=Path,
                    default=DUMP_DIR / "enwiki-latest-categorylinks.sql.gz")
    ap.add_argument("--linktarget", type=Path,
                    default=DUMP_DIR / "enwiki-latest-linktarget.sql.gz")
    ap.add_argument("--page", type=Path,
                    default=DUMP_DIR / "enwiki-latest-page.sql.gz")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data/benchmark/enwiki_cats_correct")
    ap.add_argument("--root-title", default="Main_topic_classifications",
                    help="category title (underscores) recorded in metadata/root_categories")
    ap.add_argument("--root-id", type=int, default=7345184,
                    help="page_id of the root category. Default 7345184 = "
                         "Category:Main_topic_classifications (already resolved); "
                         "pass -1 to re-resolve from the page dump via --root-title.")
    # Titles are an OUTPUT-readability layer: scoping + traversal only need the
    # coherent page_id space the Correct mode already produces. Default OFF so
    # the fast path parses each dump exactly once (inside the Rust binary) — no
    # second Python pass over the 2.4GB page dump. --titles re-reads the page
    # dump to populate s2i/i2s with real category names.
    ap.add_argument("--titles", dest="titles", action="store_true", default=False,
                    help="populate s2i/i2s with real category titles (extra page-dump pass)")
    ap.add_argument("--no-titles", dest="titles", action="store_false",
                    help="identity intern str(page_id) (default; fast path)")
    ap.add_argument("--map-size-gib", type=float, default=16.0)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    for p, name in [(LMDB_SINK, "mysql_stream_lmdb"), (MYSQL_STREAM, "mysql_stream")]:
        if not p.exists():
            sys.stderr.write(f"error: {name} not built ({p}); run `cargo build --release` "
                             f"in {MYSQL_STREAM_DIR}\n")
            return 1
    for p in (args.categorylinks, args.linktarget, args.page):
        if not p.exists():
            sys.stderr.write(f"error: dump not found: {p}\n")
            return 1

    t0 = time.time()
    out_dir = args.out
    lmdb_dir = out_dir / "lmdb_resident"
    tmp_lmdb = out_dir / "_tmp_string_lmdb"
    tsv_path = out_dir / "edges_child_parent.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_size = int(args.map_size_gib * 1024**3)

    # 1. Correct-mode ingest -> fixture TSV (decimal page_id edges).
    run_correct_ingest(args, tsv_path, tmp_lmdb)
    sys.stderr.write(f"[enwiki-correct] fixture TSV at {tsv_path} (t={time.time()-t0:.1f}s)\n")

    # 2. (optional) real category titles for ns=14 page ids.
    titles_map = {}
    if args.titles:
        sys.stderr.write("[enwiki-correct] building page_id->title map (ns=14)...\n")
        titles_map = page_title_to_id(args.page)
        sys.stderr.write(f"[enwiki-correct] {len(titles_map)} category titles "
                         f"(t={time.time()-t0:.1f}s)\n")

    # 3. build Phase-1 LMDB from the TSV.
    n_nodes, n_edges, interned = build_phase1(tsv_path, lmdb_dir, map_size, titles_map)
    sys.stderr.write(f"[enwiki-correct] Phase-1 LMDB: {n_nodes} nodes, {n_edges} edges, "
                     f"{interned} interned (t={time.time()-t0:.1f}s)\n")

    # 4. resolve root + sidecar fixture files.
    if args.root_id is not None and args.root_id >= 0:
        root_id = args.root_id  # known (default 7345184); no page re-parse
    elif titles_map:
        root_id = next((pid for pid, t in titles_map.items()
                        if t == args.root_title), None)
    else:
        root_id = page_title_to_id(args.page, want_title=args.root_title.encode("utf-8"))
    with open(out_dir / "root_ids.txt", "w") as f:
        f.write(f"{root_id}\n" if root_id is not None else "")
    with open(out_dir / "root_categories.tsv", "w") as f:
        f.write("root_category\n")
        f.write(f"{args.root_title}\n")

    # article_category.tsv: node -> node over all graph nodes (int-native seeds).
    env = lmdb.open(str(lmdb_dir), max_dbs=16, readonly=True, subdir=True, lock=False)
    i2s = env.open_db(b"i2s", create=False)
    with open(out_dir / "article_category.tsv", "w") as f:
        f.write("article\tcategory\n")
        with env.begin() as t:
            for k, _ in t.cursor(db=i2s):
                nid = I32.unpack(k)[0]
                f.write(f"{nid}\t{nid}\n")
    # confirm multi-hop coherence: root's child count
    root_children = 0
    if root_id is not None:
        cc = env.open_db(b"category_child", dupsort=True, create=False)
        with env.begin() as t:
            c = t.cursor(db=cc)
            if c.set_key(enc(root_id)):
                root_children = c.count()
    env.close()

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "name": "enwiki_cats_correct",
            "ingest": "3-dump Correct mode (categorylinks+linktarget+page)",
            "node_count": n_nodes,
            "edge_count": n_edges,
            "titles_in_intern": bool(titles_map),
            "root_title": args.root_title,
            "root_id": root_id,
            "root_child_count": root_children,
            "build_seconds": round(time.time() - t0, 1),
        }, f, indent=2)

    if not args.keep_tmp:
        # the string-keyed throwaway LMDB is not needed downstream
        import shutil
        shutil.rmtree(tmp_lmdb, ignore_errors=True)

    sys.stderr.write(
        f"[enwiki-correct] DONE in {time.time()-t0:.1f}s. root '{args.root_title}' "
        f"= page_id {root_id}, child_count={root_children} "
        f"({'COHERENT' if root_children > 0 else 'STILL BROKEN — investigate'})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
