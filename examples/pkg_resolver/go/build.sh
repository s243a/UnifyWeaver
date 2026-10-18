#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- compile examples/pkg_resolver/resolver.pl through wam_go
# (prefer_wam(true)) into this directory, then `go build` the JSON shim.
#
#   bash examples/pkg_resolver/go/build.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../resolver.pl"
OUT="$HERE"

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT"

cd "$HERE"
go build -o uwresolve ./cmd/uwresolve
echo "build.sh: go build clean -> $HERE/uwresolve"
