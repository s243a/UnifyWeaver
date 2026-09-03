// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// Thin CLI around the compiled WAM package. Resolver logic lives in
// the generated WAM project (from resolver.pl), not here.

package main

import (
	"os"

	wam "uw-pkg-resolver"
)

func main() {
	os.Exit(wam.CLI(os.Args[1:]))
}
