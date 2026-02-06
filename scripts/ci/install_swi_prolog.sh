#!/usr/bin/env bash
set -euo pipefail

PPA="ppa:swi-prolog/stable"

retry() {
  local attempts="$1"
  local delay_seconds="$2"
  shift 2

  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi

    local exit_code=$?
    if [ "$attempt" -ge "$attempts" ]; then
      return "$exit_code"
    fi

    echo "WARN: command failed (attempt ${attempt}/${attempts}, exit=${exit_code}): $*" >&2
    sleep "$delay_seconds"
    attempt=$((attempt + 1))
    delay_seconds=$((delay_seconds * 2))
  done
}

retry 3 5 sudo apt-get update -qq
retry 3 5 sudo apt-get install -y --no-install-recommends software-properties-common

added=0
if retry 2 5 timeout 90 sudo apt-add-repository "$PPA" -y; then
  added=1
else
  echo "WARN: Could not add ${PPA}; installing SWI-Prolog from Ubuntu repo." >&2
fi

if ! retry 3 5 sudo apt-get update -qq; then
  if [ "$added" -eq 1 ]; then
    echo "WARN: apt-get update failed after adding ${PPA}; removing PPA and retrying..." >&2
    sudo apt-add-repository --remove "$PPA" -y || true
    retry 3 5 sudo apt-get update -qq
  else
    echo "ERROR: apt-get update failed." >&2
    exit 1
  fi
fi

retry 3 5 sudo apt-get install -y --no-install-recommends swi-prolog

