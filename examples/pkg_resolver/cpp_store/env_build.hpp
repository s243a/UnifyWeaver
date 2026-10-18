// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// env_build.hpp -- build the store adapter's machine-local environment term
//   env(CatId, Base, Installed, Requested, Layers, Excluded, Aliases)
// from a JSONL differential row. The catalog-as-term (packages/depends/
// conflicts/provides) is NOT built here -- those live in the D43 store and
// are read via seek fact sources at run time. The env layout, the env-vs-
// catalog-id fallback, and every field shape mirror store_diff_runner.pl's
// json_to_env/2 (the SWI oracle) and rust_store/shim/main.rs::env_term.
#pragma once
#include "json.hpp"
#include "wam_runtime.h"

namespace env_build {

// env_to_term(row) : row is the WHOLE JSONL object, so it can see a
// top-level catalog_id and either a nested "env" object or the fields
// laid out flat on the row (both accepted, per json_to_env/2).
wam_cpp::Value env_to_term(const json::Json& row);

}  // namespace env_build
