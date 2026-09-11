#!/usr/bin/env bash
set -uo pipefail

# Run through the reviewed command gate; preserve every suite's exit status.
export UW_SMOKE_TMPDIR=/tmp
export TMPDIR=/tmp
unset PREFIX

swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_reverse.pl
reverse_exit=$?
printf 'reverse_exit=%s\n' "$reverse_exit"

bash examples/pkg_resolver/c/verify_maplist.sh
maplist_chain_exit=$?
printf 'verify_maplist_exit=%s\n' "$maplist_chain_exit"

if (( reverse_exit != 0 || maplist_chain_exit != 0 )); then
    exit 1
fi
