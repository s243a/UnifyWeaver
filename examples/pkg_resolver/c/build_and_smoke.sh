#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep fresh compilation and execution in one reviewed action. A failed build
# must never fall through to executing an older artifact.
bash "$HERE/build.sh"
bash "$HERE/run_smoke_c.sh"
