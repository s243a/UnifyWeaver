#!/usr/bin/env bash
set -uo pipefail

# Explicit scratch roots for legacy test helpers; this runs inside the gate.
export UW_SMOKE_TMPDIR=/tmp
export TMPDIR=/tmp
unset PREFIX

swipl -g run_tests -t halt tests/test_wam_c_sort.pl
sort_exit=$?
printf 'sort_exit=%s\n' "$sort_exit"

swipl -g run_tests -t halt tests/test_wam_c_builtin_diagnostics.pl
diagnostics=$?
printf 'diagnostics_exit=%s\n' "$diagnostics"

swipl -g run_tests -t halt tests/test_wam_c_indexed_dispatch.pl
indexed=$?
printf 'indexed_exit=%s\n' "$indexed"

bash examples/pkg_resolver/c/build_and_smoke.sh
smoke=$?
printf 'smoke_exit=%s\n' "$smoke"

if (( sort_exit != 0 || diagnostics != 0 || indexed != 0 || smoke != 0 )); then
    exit 1
fi
