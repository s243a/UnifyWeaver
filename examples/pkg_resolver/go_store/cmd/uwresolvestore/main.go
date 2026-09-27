// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// Thin CLI around the compiled WAM store-backed package. Resolver logic
// lives in the generated WAM project (from resolver.pl + resolver_store.pl);
// catalog facts come from the D43 indexed seek stores compiled in.

package main

import (
	"os"

	wam "uw-pkg-resolver-store"
)

func main() {
	os.Exit(wam.CLI(os.Args[1:]))
}
