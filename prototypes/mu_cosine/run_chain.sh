#!/bin/bash
# Sequential job chain — NEVER use pgrep/pkill patterns to sequence jobs (self-match
# deadlock/suicide: 3 incidents). Usage: ./run_chain.sh "cmd1" "cmd2" ...
for cmd in "$@"; do
  echo "=== chain: $cmd ==="
  bash -c "$cmd" || echo "=== chain: FAILED ($?): $cmd ==="
done
