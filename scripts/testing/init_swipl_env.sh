#!/usr/bin/env bash
# init_swipl_env.sh
# UnifyWeaver SWI-Prolog Environment Setup
# Source this script to add SWI-Prolog to PATH
#
# Usage:
#   source ./scripts/testing/init_swipl_env.sh
#   source ./scripts/init_swipl_env.sh  # From test environment root
#   . ./scripts/init_swipl_env.sh

# Prevent execution (must be sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[ERROR] This script must be sourced, not executed" >&2
    echo "Usage: source ${0}" >&2
    exit 1
fi

# Detect if running with --quiet flag
QUIET=false
if [[ "${1:-}" == "--quiet" ]]; then
    QUIET=true
fi

print_info() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "\033[32m$1\033[0m"  # Green
    fi
}

print_warn() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "\033[33m$1\033[0m"  # Yellow
    fi
}

print_error() {
    echo -e "\033[31m$1\033[0m" >&2  # Red
}

# --- Platform detection ---
detect_platform() {
    case "$(uname -s)" in
        CYGWIN*) echo "cygwin" ;;
        Darwin)  echo "macos" ;;
        Linux)
            if [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        *) echo "unknown" ;;
    esac
}

PLATFORM="$(detect_platform)"

# --- Find SWI-Prolog ---
find_swipl() {
    # Check if already on PATH
    if command -v swipl >/dev/null 2>&1; then
        command -v swipl
        return 0
    fi

    # Platform-specific search paths
    local search_paths=()
    case "$PLATFORM" in
        cygwin)
            search_paths=(
                "/cygdrive/c/Program Files/swipl/bin"
                "/cygdrive/c/Program Files (x86)/swipl/bin"
                "/cygdrive/c/swipl/bin"
                "/usr/local/bin"
                "/usr/bin"
            )
            ;;
        wsl)
            search_paths=(
                "/mnt/c/Program Files/swipl/bin"
                "/mnt/c/Program Files (x86)/swipl/bin"
                "/mnt/c/swipl/bin"
                "/usr/local/bin"
                "/usr/bin"
                "/opt/swipl/bin"
                "$HOME/.local/bin"
            )
            ;;
        linux)
            search_paths=(
                "/usr/bin"
                "/usr/local/bin"
                "/opt/swipl/bin"
                "/snap/swi-prolog/current/bin"
                "$HOME/.local/bin"
            )
            ;;
        macos)
            search_paths=(
                "/usr/local/bin"
                "/opt/homebrew/bin"
                "/opt/local/bin"
                "/usr/bin"
                "/Applications/SWI-Prolog.app/Contents/MacOS"
            )
            ;;
    esac

    # Search for swipl executable
    for dir in "${search_paths[@]}"; do
        if [[ -x "$dir/swipl" && -f "$dir/swipl" ]]; then
            echo "$dir/swipl"
            return 0
        fi
        # Also check for .exe on WSL/Cygwin
        if [[ "$PLATFORM" =~ ^(wsl|cygwin)$ ]] && [[ -x "$dir/swipl.exe" ]]; then
            echo "$dir/swipl.exe"
            return 0
        fi
    done

    return 1
}

# --- Main setup ---
SWIPL_CMD="$(find_swipl)"

if [[ -z "$SWIPL_CMD" ]]; then
    print_error "[ERROR] SWI-Prolog not found in common locations"
    print_error ""
    print_error "Install instructions:"
    case "$PLATFORM" in
        wsl|linux)
            print_error "  sudo apt-add-repository ppa:swi-prolog/stable"
            print_error "  sudo apt update && sudo apt install swi-prolog"
            ;;
        macos)
            print_error "  brew install swi-prolog"
            ;;
        cygwin)
            print_error "  Install from: https://www.swi-prolog.org/download/stable"
            ;;
    esac
    return 1
fi

# --- Add to PATH if needed ---
SWIPL_BIN_DIR="$(dirname "$SWIPL_CMD")"

if [[ ":$PATH:" != *":$SWIPL_BIN_DIR:"* ]]; then
    export PATH="$SWIPL_BIN_DIR:$PATH"
    print_info "[✓] Added SWI-Prolog to PATH: $SWIPL_BIN_DIR"
else
    print_info "[✓] SWI-Prolog already in PATH: $SWIPL_BIN_DIR"
fi

# --- Set locale for SWI-Prolog ---
export LANG='en_US.UTF-8'
export LC_ALL='en_US.UTF-8'

# --- Verify installation ---
if command -v swipl >/dev/null 2>&1; then
    SWIPL_VERSION="$(swipl --version 2>&1 | head -n1)"
    print_info "[✓] $SWIPL_VERSION"
else
    print_error "[ERROR] SWI-Prolog found but cannot execute: $SWIPL_CMD"
    return 1
fi

return 0
