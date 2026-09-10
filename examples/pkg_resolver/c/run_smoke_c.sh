#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_smoke_c.sh -- named C WAM smoke slice vs SWI. Not full corpus parity.
# Cases from examples/pkg_resolver/test_resolver.pl corpus_case/4:
#   empty_requests, single_package, backtrack_conflict_deeper,
#   unsatisfiable_missing
#
# Query inputs do not constrain the output to the expected answer.
# A deliberately wrong expected-answer control must be detected.
#
#   bash examples/pkg_resolver/c/run_smoke_c.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
GEN="$HERE/generated"
BIN="$GEN/smoke_runner"
OUT="$HERE/.smoke_out"
ORACLE="$HERE/swi_oracle.pl"

mkdir -p "$OUT"

if [ ! -x "$BIN" ]; then
  echo "run_smoke_c.sh: $BIN missing -- run build.sh first" >&2
  exit 2
fi

CASES="empty_requests single_package backtrack_conflict_deeper unsatisfiable_missing"

extract_answer() {
  local file="$1"
  awk '
    /^STATUS / { status=$2 }
    /^TERM / { sub(/^TERM /, ""); term=$0 }
    END {
      if (status == "") status="missing_status"
      if (status == "ok") print "ok " term
      else print status
    }
  ' "$file"
}

classify_c_rc() {
  local rc="$1"
  case "$rc" in
    0) echo logical ;;
    2) echo usage ;;
    3) echo generator_omission ;;
    4) echo runtime_error ;;
    124) echo timeout ;;
    137) echo killed ;;
    139) echo crash_segfault ;;
    136) echo crash_fpe ;;
    *)
      if [ "$rc" -ge 128 ]; then
        echo "crash_signal_$((rc-128))"
      else
        echo "unexpected_exit_$rc"
      fi
      ;;
  esac
}

run_captured() {
  local label="$1"
  local logfile="$2"
  shift 2
  set +e
  timeout 15 "$@" >"$logfile" 2>&1
  local rc=$?
  set -e
  echo "orig_exit=$rc" >>"$logfile"
  printf '%s' "$rc"
}

echo "== C builtin inventory (lib.c) =="
if [ -f "$GEN/lib.c" ]; then
  grep -o 'INSTR_BUILTIN_CALL, .as.pred = { .pred = "[^"]*"' "$GEN/lib.c" \
    | sed 's/.*pred = "//;s/"$//' \
    | sort -u >"$OUT/builtins_used.txt" || true
  if [ -s "$OUT/builtins_used.txt" ]; then
    echo "used:"
    cat "$OUT/builtins_used.txt"
  else
    echo "(no INSTR_BUILTIN_CALL sites)"
    : >"$OUT/builtins_used.txt"
  fi
  cat >"$OUT/builtins_c_runtime.txt" <<'EOF'
true/0
fail/0
false/0
!/0
=/2
==/2
\==/2
atom/1
integer/1
number/1
float/1
var/1
nonvar/1
compound/1
is_list/1
is/2
functor/3
arg/3
atom_concat/3
>/2
</2
>=/2
=</2
=:=/2
=\=/2
EOF
  : >"$OUT/builtins_unimplemented.txt"
  while IFS= read -r b; do
    [ -z "$b" ] && continue
    if ! grep -Fxq "$b" "$OUT/builtins_c_runtime.txt"; then
      echo "$b" >>"$OUT/builtins_unimplemented.txt"
    fi
  done <"$OUT/builtins_used.txt"
  if [ -s "$OUT/builtins_unimplemented.txt" ]; then
    echo "UNIMPLEMENTED in C runtime (static inventory hint only; escaping can mislead):"
    cat "$OUT/builtins_unimplemented.txt"
  else
    echo "all inventoried builtins have C runtime arms (inventory is a hint, not reach proof)"
  fi
fi

FAILS=0
PASS=0

echo "== smoke cases =="
for id in $CASES; do
  echo "-- $id --"
  swi_log="$OUT/${id}.swi.txt"
  c_log="$OUT/${id}.c.txt"
  SWI_RC="$(run_captured swi "$swi_log" swipl -q -g main -t halt "$ORACLE" -- "$id")"
  echo "SWI orig_exit=$SWI_RC"
  if [ "$SWI_RC" -ne 0 ]; then
    echo "CLASS oracle_error"
    echo "SWI output:"
    cat "$swi_log"
    FAILS=$((FAILS+1))
    continue
  fi
  C_RC="$(run_captured c "$c_log" "$BIN" "$id")"
  C_CLASS="$(classify_c_rc "$C_RC")"
  echo "C orig_exit=$C_RC class=$C_CLASS"
  SWI_ANS="$(extract_answer "$swi_log")"
  C_ANS="$(extract_answer "$c_log")"
  echo "SWI $SWI_ANS"
  echo "C   $C_ANS"
  if [ "$C_CLASS" != "logical" ]; then
    echo "RESULT $id BLOCKED ($C_CLASS) -- not a logical mismatch"
    echo "C output:"
    cat "$c_log"
    if grep -q '^BUILTIN ' "$c_log"; then
      echo "REACHED unsupported builtin:"
      grep '^BUILTIN \|^KIND \|^ARITY ' "$c_log" || true
    fi
    FAILS=$((FAILS+1))
    continue
  fi
  if [ "$SWI_ANS" = "$C_ANS" ]; then
    echo "RESULT $id MATCH"
    PASS=$((PASS+1))
  else
    echo "RESULT $id MISMATCH"
    if [ -s "$OUT/builtins_unimplemented.txt" ] && [ "$C_ANS" = "fail" ] && [ "$SWI_ANS" != "fail" ]; then
      echo "NOTE: C reported fail while SWI succeeded; inventory lists unimplemented ops, but fail is not proof they were reached"
    fi
    FAILS=$((FAILS+1))
  fi
done

echo "== wrong expected-answer control =="
# Use the C output of single_package against a term that is not the SWI
# answer. Comparator must report mismatch. If it reports match, the
# harness is broken.
ctrl_got="$OUT/single_package.c.txt"
if [ ! -f "$ctrl_got" ]; then
  echo "mismatch control: missing C output for single_package" >&2
  exit 1
fi
WRONG="$OUT/wrong_expected.txt"
{
  echo "CASE single_package"
  echo "STATUS ok"
  echo "TERM [-(bar,v(9,9,9))]"
} >"$WRONG"
GOT_ANS="$(extract_answer "$ctrl_got")"
WRONG_ANS="$(extract_answer "$WRONG")"
echo "C got:     $GOT_ANS"
echo "wrong exp: $WRONG_ANS"
if [ "$GOT_ANS" = "$WRONG_ANS" ]; then
  echo "mismatch control: FAILED (wrong expected answer compared equal)"
  echo "orig_exit=1"
  exit 1
fi
echo "mismatch control: detected (ok) -- comparator rejects a wrong expected term"

echo "== summary =="
echo "matched $PASS / $(echo $CASES | wc -w)  mismatches_or_blocks $FAILS"
if [ "$FAILS" -ne 0 ]; then
  echo "orig_exit=1"
  exit 1
fi
echo "orig_exit=0"
exit 0
