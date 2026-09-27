// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// env_build.cpp -- see env_build.hpp. A ~40-line trim of
// term_build::catalog_to_term: it builds ONLY the machine-local env term
// (no packages/depends/conflicts/provides), reusing term_build's
// hold_term / pair_term / layer_term / alias_term for the field shapes so
// they stay in lockstep with the term-catalog lane and the SWI oracle.

#include "env_build.hpp"
#include "term_build.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace env_build {

using wam_cpp::CellPtr;
using wam_cpp::Value;

namespace {

Value atom(const std::string& name) { return Value::Atom(name); }

Value st(const std::string& name, std::vector<Value> args) {
    std::string functor = name + "/" + std::to_string(args.size());
    std::vector<CellPtr> cargs;
    cargs.reserve(args.size());
    for (auto& a : args) cargs.push_back(std::make_shared<Value>(std::move(a)));
    return Value::Compound(std::move(functor), std::move(cargs));
}

Value list_of(std::vector<Value> items) {
    Value cur = Value::Atom("[]");
    for (auto it = items.rbegin(); it != items.rend(); ++it) {
        std::vector<CellPtr> cargs;
        cargs.reserve(2);
        cargs.push_back(std::make_shared<Value>(std::move(*it)));
        cargs.push_back(std::make_shared<Value>(std::move(cur)));
        cur = Value::Compound("[|]/2", std::move(cargs));
    }
    return cur;
}

// Build a list from an array field, applying `f` to each element; empty (or
// absent / non-array) yields []. Mirrors `maplist(F, Field, Out) ; Out=[]`.
template <typename F>
Value list_field(const json::Json* d, const std::string& key, F f) {
    std::vector<Value> items;
    if (d) {
        if (const json::Json* fld = d->find(key)) {
            if (fld->isArray()) {
                for (const auto& item : fld->asArray()) items.push_back(f(item));
            }
        }
    }
    return list_of(std::move(items));
}

}  // namespace

Value env_to_term(const json::Json& row) {
    // CatId: top-level catalog_id, else env.catalog_id, else `default`
    //        (json_to_env/2 clauses 40-43).
    std::string cat_id = "default";
    if (const json::Json* top = row.find("catalog_id")) {
        if (top->isString()) cat_id = top->asString();
    } else if (const json::Json* env = row.find("env")) {
        if (const json::Json* nested = env->find("catalog_id")) {
            if (nested->isString()) cat_id = nested->asString();
        }
    }

    // Field source D: the nested "env" object if present, else the row itself
    // (json_to_env/2 clause 44).
    const json::Json* d = row.find("env");
    if (!d) d = &row;

    Value base      = list_field(d, "base",      [](const json::Json& r) { return term_build::hold_term(r); });
    Value installed = list_field(d, "installed",  [](const json::Json& r) { return term_build::pair_term(r[0].asString(), r[1]); });
    Value requested = list_field(d, "requested",  [](const json::Json& r) { return Value::Atom(r.asString()); });
    Value layers    = list_field(d, "layers",     [](const json::Json& r) { return term_build::layer_term(r); });
    Value excluded  = list_field(d, "excluded",   [](const json::Json& r) { return Value::Atom(r.asString()); });
    Value aliases   = list_field(d, "aliases",    [](const json::Json& r) { return term_build::alias_term(r); });

    return st("env", {
        atom(cat_id),
        std::move(base),
        std::move(installed),
        std::move(requested),
        std::move(layers),
        std::move(excluded),
        std::move(aliases),
    });
}

}  // namespace env_build
