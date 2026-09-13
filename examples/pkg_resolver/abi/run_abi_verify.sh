#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_abi_verify.sh -- build the symbol-level ABI store from THIS machine's
# real Ubuntu/Debian data and verify it against the PoC's numbers.
#
# Tiers exercised:
#   1. .symbols control member  -> symprov intervals (zero binary download)
#   2. readelf on libc.so.6     -> symprov (fallback), cross-checked vs tier 1
#   3. readelf on /bin/ls       -> symreq (verneed floor) + NEEDED sonames
#
# Everything lands in ./.out (gitignored). Requires: node, readelf, swipl.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/.out"
STORE="$OUT/store"
BINARY="${ABI_BINARY:-/bin/ls}"
SONAME="${ABI_SONAME:-libc.so.6}"
LIBSO="${ABI_LIBSO:-/lib/x86_64-linux-gnu/libc.so.6}"

# Locate libc6's .symbols control member (arch-qualified on multiarch).
SYMFILE=""
for c in /var/lib/dpkg/info/libc6:amd64.symbols /var/lib/dpkg/info/libc6.symbols; do
  [ -f "$c" ] && SYMFILE="$c" && break
done
[ -z "$SYMFILE" ] && SYMFILE="$(ls /var/lib/dpkg/info/libc6*.symbols 2>/dev/null | grep -v i386 | head -1)"
[ -z "$SYMFILE" ] && { echo "no libc6 .symbols file found" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$STORE" "$OUT/cmp"

echo "== Tier 1: ingest $SYMFILE (.symbols control member) -> symprov =="
node "$HERE/ingest_symbols.mjs" symbols-file "$SYMFILE" --out "$STORE"

echo
echo "== Tier 3: ingest $BINARY requires -> symreq + NEEDED =="
node "$HERE/ingest_symbols.mjs" requires "$BINARY" --out "$STORE" --append

echo
echo "== Tier 2 cross-check: readelf($LIBSO) vs .symbols intros =="
node "$HERE/ingest_symbols.mjs" symbols-file "$SYMFILE" --out "$OUT/cmp/sym" >/dev/null 2>&1
node "$HERE/ingest_symbols.mjs" elf "$LIBSO" --out "$OUT/cmp/elf" >/dev/null 2>&1
node -e '
const fs=require("fs");
const load=(f,so)=>{const m=new Map();for(const l of fs.readFileSync(f,"utf8").split("\n")){if(!l)continue;const [k,v]=JSON.parse(l);if(!k.startsWith(so+"|"))continue;m.set(k.slice(so.length+1),v.split("#")[0]);}return m;};
const so=process.argv[3];
const s=load(process.argv[1]+"/symprov.jsonl",so), e=load(process.argv[2]+"/symprov.jsonl",so);
let shared=0,agree=0; const dis=[];
for(const [sym,iv] of e){ if(s.has(sym)){shared++; if(s.get(sym)===iv)agree++; else if(dis.length<5)dis.push(sym+" (.symbols="+s.get(sym)+" elf="+iv+")");}}
const pct=100*agree/shared;
console.log("  .symbols libc.so.6 syms:",s.size,"| readelf syms:",e.size,"| shared:",shared);
console.log("  intro agreement:",agree+"/"+shared,"("+pct.toFixed(1)+"%)");
console.log("  sample disagreements (glibc 2.34 pthread/rt merge):",JSON.stringify(dis));
if(pct<85){console.error("  FAIL: intro agreement below 85%");process.exit(1);}
if(s.get("getenv")!==e.get("getenv")){console.error("  FAIL: getenv intro mismatch");process.exit(1);}
console.log("  PASS: curated .symbols and readelf agree on the intro axis");
' "$OUT/cmp/sym" "$OUT/cmp/elf" "$SONAME"

echo
echo "== Prolog resolver verification =="
swipl -q -g run -t halt "$HERE/test_abi.pl" -- "$STORE"
