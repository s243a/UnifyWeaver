#!/usr/bin/env bash
set -uo pipefail

# Run through the reviewed command gate; preserve every suite's exit status.
export UW_SMOKE_TMPDIR=/tmp
export TMPDIR=/tmp
unset PREFIX

swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_maplist.pl
maplist_exit=$?
printf 'maplist_exit=%s\n' "$maplist_exit"

swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_member.pl
member_exit=$?
printf 'member_exit=%s\n' "$member_exit"

swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_builtin_diagnostics.pl
diagnostics_exit=$?
printf 'diagnostics_exit=%s\n' "$diagnostics_exit"

swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_sort.pl
sort_exit=$?
printf 'sort_exit=%s\n' "$sort_exit"

swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_indexed_dispatch.pl
indexed_exit=$?
printf 'indexed_exit=%s\n' "$indexed_exit"

bash examples/pkg_resolver/c/build_and_smoke.sh
smoke_exit=$?
printf 'smoke_exit=%s\n' "$smoke_exit"

if (( maplist_exit != 0 || member_exit != 0 || diagnostics_exit != 0 || sort_exit != 0 || indexed_exit != 0 || smoke_exit != 0 )); then
    exit 1
fi
