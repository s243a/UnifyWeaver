#!/usr/bin/env bash
# Extract pearltrees XML fragments (Tree/RefPearl) optionally filtering title by regex (case-insensitive) and privacy <=1.
# Usage: ./extract_pearltrees_fragments.sh [-f title_regex] <input_rdf>
# Outputs NUL-delimited XML fragments to stdout.

set -euo pipefail

FILTER=""
INPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--filter)
      FILTER="$2"; shift 2 ;;
    *) INPUT="$1"; shift ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Usage: $0 [-f title_regex] <input_rdf>" >&2
  exit 1
fi

AWK=./scripts/utils/select_xml_elements.awk
TAG="pt:Tree|pt:RefPearl"

# Extract fragments
if [[ -n "$FILTER" ]]; then
  awk -f "$AWK" -v tag="$TAG" "$INPUT" | \
    awk 'BEGIN{RS="\0"; ORS="\0"; IGNORECASE=1} /<dcterms:title>/{if(",$FILTER,"~tolower($0)) print}'
else
  awk -f "$AWK" -v tag="$TAG" "$INPUT"
fi
