#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
set -euo pipefail
# Resolve our own directory; works when executed or sourced
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"   # robust path [1]
: "${SCRIPT_DIR:="$SELF_DIR"}"                             # default to our folder [1]
: "${TARGET_ROOT:="$SCRIPT_DIR"}"                          # pin to test_env by default [1]

# Repo-local config in the same directory; source if present
CFG="$SCRIPT_DIR/.unifyweaver.conf"
[[ -r "$CFG" ]] && { # shellcheck disable=SC1090
                      source "$CFG"; }                     # may define SWIPL_CMD [1]

# Atomic upsert key=value into CFG
upsert_cfg() {
  local key="$1" val="$2" tmp qv
  tmp="$(mktemp)"                                  # atomic write target [1]
  # Escape single quotes in value for single-quoted assignment: ' -> '\'' 
  qv=$(printf "%s" "$val" | sed "s/'/'\"'\"'/g")   # standard escape pattern [2]
  if [[ -f "$CFG" ]]; then
    # q is a literal single-quote for AWK to assemble k='v'
    local q="'"
    awk -v k="$key" -v v="$qv" -v q="$q" '
      BEGIN{done=0}
      /^[[:space:]]*#/ { print; next }             # keep comments
      /^[[:space:]]*$/ { print; next }             # keep blanks
      $0 ~ "^[[:space:]]*"k"=" {
        print k"="q v q; done=1; next              # replace existing
      }
      { print }
      END{ if(!done) print k"="q v q }             # append if missing
    ' "$CFG" > "$tmp"
  else
    printf "%s='%s'\n" "$key" "$qv" > "$tmp"       # create new file
  fi
  mv "$tmp" "$CFG"                                  # atomic replace
}
# atomic write [2]

# Ask-once yes/no; defaults to "no"; persists to CFG
ask_yes_no() {
  local var="$1" prompt="$2" default="${3:-no}" ans
  if [[ -n "${!var:-}" && "${!var}" != "ask" ]]; then return 0; fi
  read -p "$prompt (Y/n): " -n 1 -r; echo
  if [[ -z "${REPLY:-}" ]]; then ans="$default"
  elif [[ "$REPLY" =~ ^[Yy]$ ]]; then ans="yes"
  else ans="no"
  fi
  printf -v "$var" '%s' "$ans"; export "$var"
  upsert_cfg "$var" "$ans"
}                                                          # cache answers [1]

# Platform detection and WSL interop
detect_platform() { case "$(uname -s)" in CYGWIN*) echo cygwin;; Darwin) echo macos;;
  Linux) if [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then echo wsl; else echo linux; fi;;
  *) echo unknown;; esac; }
PLATFORM="$(detect_platform)"                               # detect once [23]
interop_enabled() { [[ -r /proc/sys/fs/binfmt_misc/WSLInterop ]] && grep -q '^enabled' /proc/sys/fs/binfmt_misc/WSLInterop; }  # WSL binfmt [23]

# PATH helper and wrapper marker
WRAPPER_PATH="$TARGET_ROOT/bin/swipl"
path_prepend_once() { case ":$PATH:" in *":$1:"*) :;; *) PATH="$1:$PATH";; esac; export PATH; }  # idempotent PATH [1]
WRAPPER_MARK='# unifyweaver-swipl-wrapper v1'
is_our_wrapper() { [[ -x "$WRAPPER_PATH" ]] && grep -qF "$WRAPPER_MARK" "$WRAPPER_PATH"; }  # wrapper detection [1]

# Persist chosen swipl
set_swipl_cmd() { SWIPL_CMD="$1"; export SWIPL_CMD; upsert_cfg SWIPL_CMD "$SWIPL_CMD"; }    # single source of truth [1]

# Candidate searchers
get_common_swipl_paths() {
  case "$PLATFORM" in
    cygwin) printf '%s\n' "/cygdrive/c/Program Files/swipl/bin" "/cygdrive/c/Program Files (x86)/swipl/bin" "/cygdrive/c/swipl/bin" "/usr/local/bin" "/usr/bin" ;;
    wsl)    printf '%s\n' "/mnt/c/Program Files/swipl/bin" "/mnt/c/Program Files (x86)/swipl/bin" "/mnt/c/swipl/bin" "/usr/local/bin" "/usr/bin" "/opt/swipl/bin" ;;
    linux)  printf '%s\n' "/usr/bin" "/usr/local/bin" "/opt/swipl/bin" "/snap/swi-prolog/current/bin" "$HOME/.local/bin" ;;
    macos)  printf '%s\n' "/usr/local/bin" "/opt/homebrew/bin" "/opt/local/bin" "/usr/bin" "/Applications/SWI-Prolog.app/Contents/MacOS" ;;
    *)      printf '%s\n' "/usr/bin" "/usr/local/bin" ;;
  esac
}                                                          # candidates [23]
#find_swipl() {
#  if command -v swipl >/dev/null 2>&1; then command -v swipl; return 0; fi
#  while IFS= read -r base; do [[ -x "$base/swipl" ]] && { echo "$base/swipl"; return 0; }; done < <(get_common_swipl_paths)
#  return 1
#}

find_swipl() {
  if command -v swipl >/dev/null 2>&1; then command -v swipl; return 0; fi
  while IFS= read -r base; do 
    if [[ -x "$base/swipl" && -f "$base/swipl" ]]; then
      echo "$base/swipl"
      return 0
    fi
  done < <(get_common_swipl_paths)
  return 1
}
                                                          # native swipl [1]
#find_windows_swipl() {
#  local c; for c in "/mnt/c/Program Files/swipl/bin/swipl.exe" "/mnt/c/Program Files (x86)/swipl/bin/swipl.exe" "/mnt/c/swipl/bin/swipl.exe"; do
#    [[ -x "$c" ]] && { echo "$c"; return 0; }
#  done; return 1
#}                                                          # swipl.exe on WSL [23][24]

find_windows_swipl() {
  while IFS= read -r base; do
    if [[ -x "$base/swipl.exe" ]]; then
      echo "$base/swipl.exe"
      return 0
    fi
  done < <(get_common_swipl_paths)
  return 1
}

# Main entry: ensure_SWIPL ready (no module loads here)
ensure_swipl_ready() {
  local swipl_native

  # 1) Check if native Linux swipl exists (for potential upgrade offer)
  swipl_native="$(find_swipl 2>/dev/null)" || swipl_native=""

  # Special case: --force-windows flag for testing
  if [[ "${FORCE_WINDOWS_SWIPL:-0}" == "1" ]]; then
    echo "============================================"
    echo "FORCE_WINDOWS_SWIPL mode enabled"
    echo "Forcing Windows SWI-Prolog wrapper for testing"
    echo "============================================"
    swipl_native=""  # Pretend native doesn't exist
  fi

  # 2) Honor persisted SWIPL_CMD if it exists and is valid
  if [[ -n "${SWIPL_CMD:-}" && -x "$SWIPL_CMD" ]]; then

    # Check if user has wrapper configured but native is available
    if [[ "$SWIPL_CMD" == "$WRAPPER_PATH" && -n "$swipl_native" && "$PLATFORM" == "wsl" ]]; then
      echo "============================================"
      echo "Native Linux SWI-Prolog detected: $swipl_native"
      echo "Current config uses Windows swipl wrapper."
      echo ""
      echo "RECOMMENDATION: Switch to native version."
      echo "  - Arrow keys will work (readline support)"
      echo "  - Better terminal integration"
      echo ""
      echo "Windows version is fine for Cygwin, but in WSL"
      echo "you'll likely have no arrow key support."
      echo "============================================"

      ask_yes_no SWITCH_TO_NATIVE "Switch to native Linux SWI-Prolog now?" "yes"

      if [[ "$SWITCH_TO_NATIVE" == "yes" ]]; then
        set_swipl_cmd "$swipl_native"
        echo "Switched to: $swipl_native"
        return 0
      fi
    fi

    # Use the persisted command (user's choice)
    if [[ "$SWIPL_CMD" == "$WRAPPER_PATH" ]]; then
      path_prepend_once "$(dirname "$SWIPL_CMD")"
    fi
    return 0
  fi

  # 3) No persisted command - prefer native swipl if available
  if [[ -n "$swipl_native" ]]; then
    set_swipl_cmd "$swipl_native"
    echo "Using native SWI-Prolog: $swipl_native"
    return 0
  fi

  # 4) WSL fallback via wrapper (with warning)
  if [[ "$PLATFORM" == "wsl" ]] && interop_enabled; then
    if is_our_wrapper; then
      path_prepend_once "$TARGET_ROOT/bin"
      set_swipl_cmd "$WRAPPER_PATH"
      echo "WARNING: Using Windows SWI-Prolog - arrow keys may not work in WSL"
      return 0
    fi

    echo "============================================"
    echo "No native Linux SWI-Prolog found."
    echo ""
    echo "RECOMMENDATION: Install native version:"
    echo "  sudo apt-add-repository ppa:swi-prolog/stable"
    echo "  sudo apt update && sudo apt install swi-prolog"
    echo ""
    echo "Otherwise, can create wrapper for Windows swipl.exe"
    echo "  (Warning: arrow keys won't work in WSL)"
    echo "============================================"

    WRAP_SWIPL="${WRAP_SWIPL:-ask}"
    ask_yes_no WRAP_SWIPL "Create wrapper for Windows swipl.exe anyway?" "no"

    if [[ "$WRAP_SWIPL" == "yes" ]]; then
      local win_swipl
      if [[ -n "${WIN_SWIPL_PATH:-}" && -x "$WIN_SWIPL_PATH" ]]; then
        win_swipl="$WIN_SWIPL_PATH"
      else
        win_swipl="$(find_windows_swipl)" || win_swipl=""
        [[ -n "$win_swipl" ]] && upsert_cfg WIN_SWIPL_PATH "$win_swipl"
      fi
      if [[ -n "$win_swipl" && -x "$win_swipl" ]]; then
        mkdir -p "$TARGET_ROOT/bin"
        cat > "$WRAPPER_PATH" <<EOF
#!/usr/bin/env bash
$WRAPPER_MARK
exec "$win_swipl" "\$@"
EOF
        chmod +x "$WRAPPER_PATH"
        path_prepend_once "$TARGET_ROOT/bin"
        set_swipl_cmd "$WRAPPER_PATH"
        echo "Created wrapper (arrow keys won't work in WSL)"
        return 0
      fi
    fi
  fi

  return 1
}

ensure_swipl_ready
