#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_abi_verify.sh -- build the symbol-level ABI store from THIS machine's
# real Ubuntu/Debian data plus the review fixtures, then verify.
#
#   1. Real data: libc6 + libselinux1 `.symbols` -> symprov (since bounds);
#      /bin/ls -> symreq (attributed via the ELF version index) + NEEDED;
#      the release axis from `apt-cache madison` + dpkg (real candidates).
#   2. Cross-check: readelf(libc.so.6) vs `.symbols` on EXACT sym@node
#      identity (must be 100%), plus the corrected per-name earliest-row
#      comparison the old script got wrong (must also be 100%).
#   3. ELF fixtures (gcc): foo@LIB_1 vs foo@LIB_2 under one soname (with the
#      loader as ground truth), COMMON_1 cross-attribution, PUBLIC +
#      unversioned obligations, a missing ELF, template `.symbols` rejects.
#   4. test_abi.pl: the Prolog assertions over all of the above.
#
# Everything lands in ./.out (gitignored). Requires: node, readelf, swipl;
# gcc for the ELF fixtures (ABI_ALLOW_SKIP=1 tolerates a missing gcc).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/.out"
STORE="$OUT/store"
FX="$OUT/fx"
SRC="$HERE/fixtures"
INGEST="$HERE/ingest_symbols.mjs"
BINARY="${ABI_BINARY:-/bin/ls}"
LIBSO="${ABI_LIBSO:-/lib/x86_64-linux-gnu/libc.so.6}"
ARCH="$(dpkg --print-architecture 2>/dev/null || echo amd64)"

find_symbols() {   # <package> -> path of its installed .symbols control member
  local p
  for p in "/var/lib/dpkg/info/$1:$ARCH.symbols" "/var/lib/dpkg/info/$1.symbols"; do
    [ -f "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}
LIBC_SYM="$(find_symbols libc6)" || { echo "no libc6 .symbols file found" >&2; exit 1; }
SELINUX_SYM="$(find_symbols libselinux1 || true)"

rm -rf "$OUT"
mkdir -p "$STORE" "$OUT/cmp" "$FX"

echo "== 1. Real data: .symbols (since bounds) + $BINARY requires + release axis =="
node "$INGEST" symbols-file "$LIBC_SYM" --out "$STORE"
if [ -n "$SELINUX_SYM" ]; then
  node "$INGEST" symbols-file "$SELINUX_SYM" --out "$STORE" --append
fi
node "$INGEST" requires "$BINARY" --out "$STORE" --append

# Release axis: every libc6 candidate the package index knows (madison) plus
# the installed version. These are actual releases, not symbol-intro points.
mapfile -t LIBC_RELS < <( { apt-cache madison libc6 2>/dev/null | awk -F'|' '{gsub(/ /,"",$2); print $2}'; dpkg-query -W -f '${Version}\n' libc6; } | sort -u )
node "$INGEST" releases libc.so.6 "${LIBC_RELS[@]}" --out "$STORE" --append
if [ -n "$SELINUX_SYM" ]; then
  mapfile -t SEL_RELS < <( { apt-cache madison libselinux1 2>/dev/null | awk -F'|' '{gsub(/ /,"",$2); print $2}'; dpkg-query -W -f '${Version}\n' libselinux1; } | sort -u )
  node "$INGEST" releases libselinux.so.1 "${SEL_RELS[@]}" --out "$STORE" --append
fi
# The declared dependency dpkg-shlibdeps generated for the binary's package:
# ground truth for abi_floor (test_abi.pl compares against this file).
OWNER="$(dpkg -S "$BINARY" 2>/dev/null | head -1 | cut -d: -f1 || true)"
if [ -n "$OWNER" ]; then
  dpkg-query -W -f '${Pre-Depends}, ${Depends}\n' "$OWNER" > "$OUT/declared_depends.txt"
  echo "declared deps of $OWNER: $(cat "$OUT/declared_depends.txt")"
fi

echo
echo "== 2. Cross-check: readelf($LIBSO) vs $LIBC_SYM =="
node "$INGEST" elf "$LIBSO" --out "$OUT/cmp/elf" 2>/dev/null
node "$INGEST" symbols-file "$LIBC_SYM" --out "$OUT/cmp/sym" 2>/dev/null
node "$HERE/crosscheck.mjs" "$OUT/cmp/sym/symprov.jsonl" "$OUT/cmp/elf/symprov.jsonl" libc.so.6 | tee "$OUT/crosscheck.txt"

echo
echo "== 3. Fixtures =="
echo "-- .symbols templates: simple cases accepted; unsupported template rows rejected loudly"
node "$INGEST" symbols-file "$SRC/simple.symbols" --release 1.2-1 --out "$FX/simple"
for t in tmpl_symver tmpl_cxx tmpl_arch; do
  if node "$INGEST" symbols-file "$SRC/$t.symbols" --release 1.0 --out "$FX/$t" 2>"$FX/$t.err"; then
    echo "FAIL: $t.symbols was accepted without --arch / with template rows" >&2; exit 1
  else
    echo "  rejected $t.symbols (exit $?): $(head -2 "$FX/$t.err" | tail -1)"
  fi
done
node "$INGEST" symbols-file "$SRC/tmpl_arch.symbols" --release 1.0 --arch amd64 --out "$FX/tmpl_arch_amd64"
echo "  tmpl_arch with --arch amd64 -> $(tr -d '\n' < "$FX/tmpl_arch_amd64/symprov.jsonl" | sed 's/\]\[/] [/g')"

echo "-- missing ELF: loud failure + INCOMPLETE evidence row (never an empty success)"
if node "$INGEST" requires "$FX/does-not-exist" --out "$FX/store_missing"; then
  echo "FAIL: missing ELF ingested successfully" >&2; exit 1
fi
echo "  evidence: $(cat "$FX/store_missing/evidence.jsonl")"

if command -v gcc >/dev/null 2>&1; then
  echo "-- building ELF fixtures with gcc"
  mkdir -p "$FX/v1" "$FX/v2" "$FX/lib"
  gcc -shared -fPIC -Wl,--version-script="$SRC/foo_v1.map" -Wl,-soname,libfoo.so.1 -o "$FX/v1/libfoo.so.1" "$SRC/foo.c"
  gcc -shared -fPIC -Wl,--version-script="$SRC/foo_v2.map" -Wl,-soname,libfoo.so.1 -o "$FX/v2/libfoo.so.1" "$SRC/foo.c"
  gcc -Wl,-z,now -o "$FX/usefoo" "$SRC/usefoo.c" -L"$FX/v1" -l:libfoo.so.1
  gcc -shared -fPIC -Wl,--version-script="$SRC/common.map" -Wl,-soname,libalpha.so.1 -o "$FX/lib/libalpha.so.1" "$SRC/alpha.c"
  gcc -shared -fPIC -Wl,--version-script="$SRC/common.map" -Wl,-soname,libbeta.so.1  -o "$FX/lib/libbeta.so.1"  "$SRC/beta.c"
  gcc -Wl,-z,now -o "$FX/usecommon" "$SRC/usecommon.c" -L"$FX/lib" -l:libalpha.so.1 -l:libbeta.so.1
  gcc -shared -fPIC -Wl,--version-script="$SRC/pub.map" -Wl,-soname,libpub.so.1 -o "$FX/lib/libpub.so.1" "$SRC/pub.c"
  gcc -shared -fPIC -Wl,-soname,libplain.so.1 -o "$FX/lib/libplain.so.1" "$SRC/plain.c"
  gcc -Wl,-z,now -o "$FX/usepub" "$SRC/usepub.c" -L"$FX/lib" -l:libpub.so.1 -l:libplain.so.1

  echo "-- loader ground truth for foo@LIB_1 vs foo@LIB_2 (same soname libfoo.so.1)"
  LD_LIBRARY_PATH="$FX/v1" "$FX/usefoo" && echo "  v1: runs (foo@LIB_1 present)"
  if LD_LIBRARY_PATH="$FX/v2" "$FX/usefoo" 2>"$FX/usefoo_v2.err"; then
    echo "FAIL: loader accepted libfoo v2 (expected: undefined symbol foo, version LIB_1)" >&2; exit 1
  fi
  echo "  v2: loader rejects -> $(sed 's/^[^:]*: //' "$FX/usefoo_v2.err")"
  grep -q "undefined symbol: foo, version LIB_1" "$FX/usefoo_v2.err" || { echo "FAIL: unexpected loader error" >&2; exit 1; }

  echo "-- ingesting fixture stores"
  node "$INGEST" requires "$FX/usefoo" --out "$FX/store_foo_v1"
  node "$INGEST" elf "$FX/v1/libfoo.so.1" --release 1.0-1 --out "$FX/store_foo_v1" --append
  node "$INGEST" releases libfoo.so.1 1.0-1 --out "$FX/store_foo_v1" --append
  node "$INGEST" requires "$FX/usefoo" --out "$FX/store_foo_v2"
  node "$INGEST" elf "$FX/v2/libfoo.so.1" --release 2.0-1 --out "$FX/store_foo_v2" --append
  node "$INGEST" releases libfoo.so.1 2.0-1 --out "$FX/store_foo_v2" --append
  node "$INGEST" requires "$FX/usecommon" --out "$FX/store_common"
  node "$INGEST" elf "$FX/lib/libalpha.so.1" --release 1.0-1 --out "$FX/store_common" --append
  node "$INGEST" elf "$FX/lib/libbeta.so.1"  --release 1.0-1 --out "$FX/store_common" --append
  echo "  usecommon symreq: $(cat "$FX/store_common/symreq.jsonl" | tr '\n' ' ')"
  node "$INGEST" requires "$FX/usepub" --out "$FX/store_pub"
  node "$INGEST" elf "$FX/lib/libpub.so.1"   --release 1.0-1 --out "$FX/store_pub" --append
  node "$INGEST" elf "$FX/lib/libplain.so.1" --release 1.0-1 --out "$FX/store_pub" --append
  node "$INGEST" requires "$FX/usepub" --out "$FX/store_pub_nolibpub"
  node "$INGEST" elf "$FX/lib/libplain.so.1" --release 1.0-1 --out "$FX/store_pub_nolibpub" --append
  node "$INGEST" requires "$FX/usepub" --out "$FX/store_pub_noplain"
  node "$INGEST" elf "$FX/lib/libpub.so.1" --release 1.0-1 --out "$FX/store_pub_noplain" --append
  echo "  usepub symreq: $(grep -E 'pub_fn|plain_fn' "$FX/store_pub/symreq.jsonl" | tr '\n' ' ')"
  touch "$FX/BUILT"
else
  echo "WARNING: gcc not found; ELF fixtures not built (test_abi.pl will SKIP them only with ABI_ALLOW_SKIP=1)" >&2
fi

echo
echo "== 4. Prolog resolver verification =="
swipl -q -g run -t halt "$HERE/test_abi.pl" -- "$OUT"
