#!/usr/bin/env bash
# Reorder $PATH so that directories defined in /etc/environment's PATH
# take precedence, without duplicates. Run with: source ./reorder-path.sh

set -euo pipefail

# Extract PATH value from /etc/environment (supports PATH="..." or PATH=...)
etc_env_line="$(grep -E '^[[:space:]]*PATH=' /etc/environment | head -n1 || true)"
if [[ -z "${etc_env_line}" ]]; then
  echo "No PATH found in /etc/environment" >&2
  return 1 2>/dev/null || exit 1
fi

# Strip leading PATH= and surrounding quotes if present
etc_env_path="${etc_env_line#*=}"
etc_env_path="${etc_env_path%"}"
etc_env_path="${etc_env_path#"}"

# Helper: remove exact directory matches from PATH
remove_from_path() {
  local target="$1"
  local IFS=':'
  read -r -a parts <<< "${PATH:-}"
  local out=""
  for p in "${parts[@]}"; do
    [[ -z "$p" ]] && continue
    if [[ "$p" != "$target" ]]; then
      out="${out:+$out:}$p"
    fi
done
  PATH="$out"
}

# Process baseline dirs from right-to-left so final order matches /etc/environment
IFS=':' read -r -a base_parts <<< "${etc_env_path}"
for (( i=${#base_parts[@]}-1; i>=0; i-- )); do
  dir="${base_parts[i]}"
  [[ -z "$dir" ]] && continue
  # Normalize trivial trailing slashes
  dir="${dir%/}"
  remove_from_path "$dir"
  PATH="${dir}${PATH:+:${PATH}}"
done

# Optional final de-dup pass (defensive), preserving first occurrence
dedup() {
  local IFS=':'
  read -r -a parts <<< "${PATH:-}"
  local -A seen=()
  local out=""
  for p in "${parts[@]}"; do
    [[ -z "$p" ]] && continue
    p="${p%/}"
    if [[ -z "${seen[$p]:-}" ]]; then
      seen[$p]=1
      out="${out:+$out:}$p"
    fi
done
  PATH="$out"
}
dedup

# Export updated PATH for the current shell
export PATH