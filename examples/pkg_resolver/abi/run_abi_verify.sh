#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_abi_verify.sh -- build the symbol-level ABI store from THIS machine's
# real Ubuntu/Debian data plus the review fixtures, then verify.
#
#   1. Real data: libc6 + libselinux1 `.symbols` -> symprov (since bounds;
#      libc6 cross-checked against libc.so.6 with --elf so default-version
#      binding is proven); /bin/ls -> symreq (attributed via the ELF version
#      index) + NEEDED; the release axis from `apt-cache madison` + dpkg (real
#      candidates); a declared soname succession (libc.so.7 replaces libc.so.6).
#   2. Cross-check: readelf(libc.so.6) vs `.symbols` on EXACT sym@node
#      identity (must be 100%) and per-name node sets (must be 100%); the
#      empty-input and dropped-node negatives must FAIL (Sol P2c/P3).
#   3. ELF fixtures (gcc): foo@LIB_1 vs foo@LIB_2 under one soname (with the
#      loader as ground truth), COMMON_1 cross-attribution, PUBLIC +
#      unversioned obligations, a missing ELF, non-default-only exports (Sol
#      P1b, loader as ground truth), `.symbols` template accept/reject cases
#      including (optional) with and without an ELF cross-check (Sol P1c),
#      unknown tags, bad --release, and batch-mode atomicity (Sol P2).
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

fail() { echo "FAIL: $*" >&2; exit 1; }

# expect_reject <label> <cmd...>: the command must exit 3 (evidence failure).
expect_reject() {
  local label="$1"; shift
  local rc=0
  "$@" 2>"$FX/$label.err" || rc=$?
  [ "$rc" -eq 3 ] || fail "$label: expected exit 3, got $rc: $(head -3 "$FX/$label.err")"
  echo "  rejected ($label, exit 3): $(sed 's/^ingest_symbols: //' "$FX/$label.err" | head -2 | tail -1 | sed 's/^ *//')"
}

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
# libc6's .symbols is cross-checked against the real libc.so.6: every row must
# be exported (and vice versa), (optional) rows are dropped unless exported,
# and the default-version binding of each row is taken from the ELF.
node "$INGEST" symbols-file "$LIBC_SYM" --elf "$LIBSO" --out "$STORE"
if [ -n "$SELINUX_SYM" ]; then
  node "$INGEST" symbols-file "$SELINUX_SYM" --out "$STORE" --append
fi
node "$INGEST" requires "$BINARY" --out "$STORE" --append
# Declared soname succession: the only way an offered soname can be a
# soname_mismatch (Sol P2d); anything else that is not NEEDED is not_needed.
node "$INGEST" replaces libc.so.7 libc.so.6 --out "$STORE" --append

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
echo "-- (sol-P3) fixed regression pair: two-node symbol, non-numeric node, Base, other soname ignored"
node "$HERE/crosscheck.mjs" "$SRC/crosscheck/sym.symprov.jsonl" "$SRC/crosscheck/elf.symprov.jsonl" libx.so.1 | tee "$OUT/crosscheck_fixture.txt"
grep -q 'identity agreement: 5/5 (100.0%)' "$OUT/crosscheck_fixture.txt" || fail "crosscheck fixture: expected 5/5 identity"
grep -q 'per-name node-set agreement: 4/4 (100.0%)' "$OUT/crosscheck_fixture.txt" || fail "crosscheck fixture: expected 4/4 per-name"
echo "-- (sol-P3) dropped node must FAIL (exact identity and per-name both catch it)"
if node "$HERE/crosscheck.mjs" "$SRC/crosscheck/sym.symprov.jsonl" "$SRC/crosscheck/elf_dropnode.symprov.jsonl" libx.so.1 >"$OUT/crosscheck_drop.txt" 2>&1; then
  fail "crosscheck accepted a dropped node"
fi
grep -q 'FAIL: exact identity sets differ' "$OUT/crosscheck_drop.txt" || fail "crosscheck: unexpected failure text: $(cat "$OUT/crosscheck_drop.txt")"
echo "  expected failure, got: $(grep FAIL "$OUT/crosscheck_drop.txt" | sed 's/^ *//')"
echo "-- (sol-P2c) empty inputs must FAIL, never pass on NaN"
if node "$HERE/crosscheck.mjs" /dev/null /dev/null libc.so.6 >"$OUT/crosscheck_empty.txt" 2>&1; then
  fail "crosscheck passed on /dev/null vs /dev/null"
fi
grep -q 'FAIL: empty symbol set' "$OUT/crosscheck_empty.txt" || fail "crosscheck: unexpected empty-input text: $(cat "$OUT/crosscheck_empty.txt")"
echo "  expected failure, got: $(grep FAIL "$OUT/crosscheck_empty.txt" | sed 's/^ *//')"
if node "$HERE/crosscheck.mjs" "$SRC/crosscheck/sym.symprov.jsonl" /dev/null libx.so.1 >/dev/null 2>&1; then
  fail "crosscheck passed with an empty readelf side"
fi

echo
echo "== 3. Fixtures =="
echo "-- .symbols templates: simple cases accepted; unsupported template rows rejected loudly"
node "$INGEST" symbols-file "$SRC/simple.symbols" --release 1.2-1 --out "$FX/simple"
for t in tmpl_symver tmpl_cxx tmpl_arch; do
  expect_reject "$t" node "$INGEST" symbols-file "$SRC/$t.symbols" --release 1.0 --out "$FX/$t"
done
node "$INGEST" symbols-file "$SRC/tmpl_arch.symbols" --release 1.0 --arch amd64 --out "$FX/tmpl_arch_amd64"
echo "  tmpl_arch with --arch amd64 -> $(tr -d '\n' < "$FX/tmpl_arch_amd64/symprov.jsonl" | sed 's/\]\[/] [/g')"

echo "-- (sol-optional) an (optional) row without an ELF cross-check is NOT an export: file rejected"
expect_reject tmpl_optional node "$INGEST" symbols-file "$SRC/tmpl_optional.symbols" --release 1.0 --out "$FX/tmpl_optional"
[ ! -e "$FX/tmpl_optional/symprov.jsonl" ] || fail "tmpl_optional: store written despite rejection"
echo "-- (sol-P2b) an ARBITRARY unknown tag is rejected (whitelist), nothing partial written"
expect_reject tmpl_unknown_tag node "$INGEST" symbols-file "$SRC/tmpl_unknown_tag.symbols" --release 1.0 --out "$FX/tmpl_unknown_tag"
[ ! -e "$FX/tmpl_unknown_tag/symprov.jsonl" ] || fail "tmpl_unknown_tag: store written despite rejection"
echo "-- (sol-P2b) --release is validated with the Debian-version gate"
expect_reject bad_release node "$INGEST" symbols-file "$SRC/simple.symbols" --release definitely-not-a-debian-version --out "$FX/bad_release"
[ ! -e "$FX/bad_release" ] || fail "bad_release: store written despite rejection"
expect_reject bad_release_elf node "$INGEST" elf "$LIBSO" --release definitely-not-a-debian-version --out "$FX/bad_release_elf"
echo "-- (sol-P2a) batch mode: a file with one unresolvable block is rejected ATOMICALLY (exit 3, no rows)"
rc=0; node "$INGEST" symbols-dir "$SRC/batch" --out "$FX/batch" 2>"$FX/batch.err" || rc=$?
[ "$rc" -eq 3 ] || fail "symbols-dir: expected exit 3, got $rc: $(cat "$FX/batch.err")"
grep -q 'files=0 rejected=1 symprov=0' "$FX/batch.err" || fail "symbols-dir: expected files=0 rejected=1 symprov=0, got: $(tail -1 "$FX/batch.err")"
[ ! -s "$FX/batch/symprov.jsonl" ] || fail "symbols-dir: rows written from a rejected file: $(cat "$FX/batch/symprov.jsonl")"
[ ! -s "$FX/batch/evidence.jsonl" ] || fail "symbols-dir: evidence written from a rejected file"
echo "  $(tail -1 "$FX/batch.err"); symprov.jsonl and evidence.jsonl empty"

echo "-- missing ELF: loud failure + INCOMPLETE evidence row (never an empty success)"
if node "$INGEST" requires "$FX/does-not-exist" --out "$FX/store_missing"; then
  fail "missing ELF ingested successfully"
fi
echo "  evidence: $(cat "$FX/store_missing/evidence.jsonl")"

if command -v gcc >/dev/null 2>&1; then
  echo "-- building ELF fixtures with gcc"
  mkdir -p "$FX/v1" "$FX/v2" "$FX/lib" "$FX/hid_base" "$FX/hid_idx3" "$FX/hid_idx2"
  gcc -shared -fPIC -Wl,--version-script="$SRC/foo_v1.map" -Wl,-soname,libfoo.so.1 -o "$FX/v1/libfoo.so.1" "$SRC/foo.c"
  gcc -shared -fPIC -Wl,--version-script="$SRC/foo_v2.map" -Wl,-soname,libfoo.so.1 -o "$FX/v2/libfoo.so.1" "$SRC/foo.c"
  gcc -Wl,-z,now -o "$FX/usefoo" "$SRC/usefoo.c" -L"$FX/v1" -l:libfoo.so.1
  gcc -shared -fPIC -Wl,--version-script="$SRC/common.map" -Wl,-soname,libalpha.so.1 -o "$FX/lib/libalpha.so.1" "$SRC/alpha.c"
  gcc -shared -fPIC -Wl,--version-script="$SRC/common.map" -Wl,-soname,libbeta.so.1  -o "$FX/lib/libbeta.so.1"  "$SRC/beta.c"
  gcc -Wl,-z,now -o "$FX/usecommon" "$SRC/usecommon.c" -L"$FX/lib" -l:libalpha.so.1 -l:libbeta.so.1
  gcc -shared -fPIC -Wl,--version-script="$SRC/pub.map" -Wl,-soname,libpub.so.1 -o "$FX/lib/libpub.so.1" "$SRC/pub.c"
  gcc -shared -fPIC -Wl,-soname,libplain.so.1 -o "$FX/lib/libplain.so.1" "$SRC/plain.c"
  gcc -Wl,-z,now -o "$FX/usepub" "$SRC/usepub.c" -L"$FX/lib" -l:libpub.so.1 -l:libplain.so.1
  # libtmpl.so.1: what tmpl_optional.symbols describes (plain_fn@Base only).
  gcc -shared -fPIC -Wl,-soname,libtmpl.so.1 -o "$FX/lib/libtmpl.so.1" "$SRC/plain.c"
  # libhid.so.1: unversioned build (usehid links here), hidden-only at index 3, hidden-only at index 2.
  gcc -shared -fPIC -Wl,-soname,libhid.so.1 -o "$FX/hid_base/libhid.so.1" "$SRC/hid_plain.c"
  gcc -shared -fPIC -Wl,--version-script="$SRC/hid_idx3.map" -Wl,-soname,libhid.so.1 -o "$FX/hid_idx3/libhid.so.1" "$SRC/hid.c"
  gcc -shared -fPIC -Wl,--version-script="$SRC/hid_idx2.map" -Wl,-soname,libhid.so.1 -o "$FX/hid_idx2/libhid.so.1" "$SRC/hid.c"
  gcc -Wl,-z,now -o "$FX/usehid" "$SRC/usehid.c" -L"$FX/hid_base" -l:libhid.so.1

  echo "-- loader ground truth for foo@LIB_1 vs foo@LIB_2 (same soname libfoo.so.1)"
  LD_LIBRARY_PATH="$FX/v1" "$FX/usefoo" && echo "  v1: runs (foo@LIB_1 present)"
  if LD_LIBRARY_PATH="$FX/v2" "$FX/usefoo" 2>"$FX/usefoo_v2.err"; then
    fail "loader accepted libfoo v2 (expected: undefined symbol foo, version LIB_1)"
  fi
  echo "  v2: loader rejects -> $(sed 's/^[^:]*: //' "$FX/usefoo_v2.err")"
  grep -q "undefined symbol: foo, version LIB_1" "$FX/usefoo_v2.err" || fail "unexpected loader error"

  echo "-- (sol-P1b) loader ground truth for an UNVERSIONED reference vs a NON-DEFAULT-only export"
  LD_LIBRARY_PATH="$FX/hid_base" "$FX/usehid" && echo "  base (hid_fn@Base): runs"
  if LD_LIBRARY_PATH="$FX/hid_idx3" "$FX/usehid" 2>"$FX/usehid_idx3.err"; then
    fail "loader bound an unversioned reference to hidden hid_fn@HID_1 at verdef index 3"
  fi
  grep -q "undefined symbol: hid_fn" "$FX/usehid_idx3.err" || fail "unexpected loader error: $(cat "$FX/usehid_idx3.err")"
  echo "  hidden @HID_1 at verdef index 3: loader rejects -> $(sed 's/^[^:]*: //' "$FX/usehid_idx3.err")"
  LD_LIBRARY_PATH="$FX/hid_idx2" "$FX/usehid" && echo "  hidden @HID_1 at verdef index 2 (oldest node): loader binds (glibc legacy rule) -> runs"
  readelf -W --dyn-syms "$FX/hid_idx3/libhid.so.1" | grep -q ' hid_fn@HID_1$' || fail "hid_fn is not a single-@ (hidden) export"

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
  # Sol P1b: usehid (unversioned hid_fn) vs the three libhid builds. The real
  # libc.so.6 is ingested too: an unversioned reference may bind to ANY NEEDED
  # object, so a hard veto needs complete evidence for all of them.
  for v in hid_base hid_idx3 hid_idx2; do
    node "$INGEST" requires "$FX/usehid" --out "$FX/store_$v"
    node "$INGEST" elf "$FX/$v/libhid.so.1" --release 1.0-1 --out "$FX/store_$v" --append
    node "$INGEST" elf "$LIBSO" --out "$FX/store_$v" --append 2>/dev/null
  done
  echo "  hid_idx3 symprov: $(grep hid_fn "$FX/store_hid_idx3/symprov.jsonl")"
  grep -q '|hid_fn@HID_1",\["at","1.0-1","nondefault"\]' "$FX/store_hid_idx3/symprov.jsonl" || fail "hid_idx3: hid_fn@HID_1 not recorded as nondefault"
  grep -q '|hid_fn@HID_1",\["at","1.0-1","default"\]' "$FX/store_hid_idx2/symprov.jsonl" || fail "hid_idx2: hid_fn@HID_1 (verdef index 2) not recorded as default"
  # Sol P1c: (optional) row cross-checked against the ELF -> dropped, file accepted, evidence complete.
  echo "-- (sol-optional) (optional) row WITH --elf: maybe_fn is not exported -> dropped; plain_fn kept; evidence complete"
  node "$INGEST" symbols-file "$SRC/tmpl_optional.symbols" --release 1.0 --elf "$FX/lib/libtmpl.so.1" --out "$FX/tmpl_optional_elf" 2>"$FX/tmpl_optional_elf.err"
  grep -q 'maybe_fn@Base not exported' "$FX/tmpl_optional_elf.err" || fail "tmpl_optional --elf: expected a dropped-row notice"
  echo "  $(tr -d '\n' < "$FX/tmpl_optional_elf/symprov.jsonl") $(cat "$FX/tmpl_optional_elf/evidence.jsonl")"
  # A NON-optional row missing from the ELF is a hard inconsistency -> reject.
  expect_reject tmpl_arch_elf node "$INGEST" symbols-file "$SRC/tmpl_arch.symbols" --release 1.0 --arch amd64 --elf "$FX/lib/libtmpl.so.1" --out "$FX/tmpl_arch_elf"
  touch "$FX/BUILT"
else
  echo "WARNING: gcc not found; ELF fixtures not built (test_abi.pl will SKIP them only with ABI_ALLOW_SKIP=1)" >&2
fi

echo
echo "== 4. Prolog resolver verification =="
swipl -q -g run -t halt "$HERE/test_abi.pl" -- "$OUT"
