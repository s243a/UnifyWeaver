#!/usr/bin/env bash
set -euo pipefail

PPA="ppa:swi-prolog/stable"

log() {
  echo "INFO: $*" >&2
}

warn() {
  echo "WARN: $*" >&2
}

error() {
  echo "ERROR: $*" >&2
}

retry() {
  local attempts="$1"
  local delay_seconds="$2"
  shift 2

  local attempt=1
  while true; do
    local exit_code
    if "$@"; then
      return 0
    else
      exit_code=$?
    fi

    if [ "$attempt" -ge "$attempts" ]; then
      return "$exit_code"
    fi

    warn "command failed (attempt ${attempt}/${attempts}, exit=${exit_code}): $*"
    sleep "$delay_seconds"
    attempt=$((attempt + 1))
    delay_seconds=$((delay_seconds * 2))
  done
}

print_runner_diagnostics() {
  if [ -f /etc/os-release ]; then
    log "Runner OS metadata:"
    sed 's/^/  /' /etc/os-release || true
  fi
}

print_swi_diagnostics() {
  if command -v swipl >/dev/null 2>&1; then
    log "SWI-Prolog version:"
    swipl --version || true
  else
    warn "swipl is not on PATH after installation."
  fi

  if command -v apt-cache >/dev/null 2>&1; then
    log "apt-cache policy swi-prolog:"
    apt-cache policy swi-prolog | sed 's/^/  /' || true
  fi

  if command -v dpkg-query >/dev/null 2>&1; then
    log "Installed swi-prolog package version:"
    dpkg-query -W -f='  ${Version}\n' swi-prolog 2>/dev/null || true
  fi
}

remove_ppa_and_refresh() {
  warn "Removing ${PPA} and refreshing apt indexes..."
  timeout 90 sudo apt-add-repository --remove "$PPA" -y || true
  retry 3 5 sudo apt-get update -qq
}

ppa_source_present() {
  find /etc/apt/sources.list /etc/apt/sources.list.d \
    -type f \( -name '*.list' -o -name '*.sources' \) \
    -exec grep -q 'ppa.launchpadcontent.net/swi-prolog/stable' {} + 2>/dev/null
}

print_runner_diagnostics
retry 3 5 sudo apt-get update -qq
retry 3 5 sudo apt-get install -y --no-install-recommends software-properties-common

added=0
if retry 3 5 timeout 120 sudo apt-add-repository "$PPA" -y; then
  added=1
  log "Added ${PPA}."
else
  warn "Could not add ${PPA}; installing SWI-Prolog from default Ubuntu repo."
  if ppa_source_present; then
    warn "${PPA} appears to be partially configured; removing it before fallback."
    remove_ppa_and_refresh
  fi
fi

if ! retry 3 5 sudo apt-get update -qq; then
  if [ "$added" -eq 1 ]; then
    warn "apt-get update failed after adding ${PPA}; falling back to default Ubuntu repo."
    remove_ppa_and_refresh
    added=0
  else
    error "apt-get update failed before SWI-Prolog installation."
    exit 1
  fi
fi

if ! retry 3 5 sudo apt-get install -y --no-install-recommends swi-prolog; then
  if [ "$added" -eq 1 ]; then
    warn "SWI-Prolog install failed with ${PPA}; retrying from default Ubuntu repo."
    remove_ppa_and_refresh
    added=0
    retry 3 5 sudo apt-get install -y --no-install-recommends swi-prolog
  else
    error "Failed to install swi-prolog from default Ubuntu repo."
    exit 1
  fi
fi

if ! command -v swipl >/dev/null 2>&1; then
  error "swi-prolog install completed but swipl is not available on PATH."
  exit 1
fi

print_swi_diagnostics
