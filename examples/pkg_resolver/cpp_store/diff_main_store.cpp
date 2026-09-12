// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// diff_main_store.cpp -- JSONL differential runner for the C++ WAM
// pkg_resolver STORE lane. Same edge as cpp/diff_main.cpp, but the catalog
// is NOT read from the row: the big three (packages/depends/conflicts) plus
// revdeps/provides live in the D43 seek store and are read via CallForeign
// fact sources at run time. Each row carries only the machine-local env
// (env(CatId, Base, Installed, Requested, Layers, Excluded, Aliases)) built
// by env_build::env_to_term, and the *_store/N predicates are called.
//
// Reads JSON lines from stdin (id, env-or-flat-fields, query, args), executes
// the query against the compiled WAM store adapter, and emits one compact
// JSON line to stdout per input line.

#include "wam_runtime.h"
#include "json.hpp"
#include "term_build.hpp"
#include "term_to_json.hpp"
#include "env_build.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using wam_cpp::CellPtr;
using wam_cpp::Value;
using wam_cpp::WamState;

namespace {

// See cpp/diff_main.cpp for the full rationale: WamState::query() reuses the
// A-registers as body scratch, so the query's real output must be captured
// from the ORIGINAL argument cell (bound via shared_ptr identity) taken
// BEFORE run(), not read back from "A<arity>" afterward. Copied verbatim.
std::vector<CellPtr> query_capture(WamState& vm, const std::string& pred_key, const std::vector<Value>& args) {
    std::vector<CellPtr> cells;
    auto it = vm.labels.find(pred_key);
    if (it == vm.labels.end()) return cells;
    for (auto& c : vm.regs) c.reset();
    for (std::size_t k = 0; k < args.size(); ++k) {
        vm.set_cell("A" + std::to_string(k + 1), std::make_shared<Value>(args[k]));
    }
    for (std::size_t k = 0; k < args.size(); ++k) {
        cells.push_back(vm.get_cell("A" + std::to_string(k + 1)));
    }
    vm.trail.clear();
    vm.choice_points.clear();
    vm.aggregate_frames.clear();
    vm.mode_stack.clear();
    vm.env_stack.clear();
    vm.retract_iters.clear();
    vm.current_pred_iters.clear();
    // Reset the store-seek + dynamic iterator side stacks too: a cut can leave
    // a leaked ForeignIterator behind, and this driver reuses ONE vm across all
    // JSONL cases, so a stale iterator would contaminate the next query.
    vm.foreign_iters.clear();
    vm.dynamic_iters.clear();
    vm.pc = it->second;
    vm.cp = 0;
    vm.cut_barrier = 0;
    vm.indexed_entry = false;
    vm.pending_level_set = false;
    vm.halt = false;
    if (!vm.run()) cells.clear();
    return cells;
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

void run_case(WamState& vm, const json::Json& row, json::Json& out) {
    const json::Json* q_ptr = row.find("query");
    if (!q_ptr || !q_ptr->isString()) throw std::runtime_error("missing or invalid query");
    const std::string& q = q_ptr->asString();
    const json::Json* args_ptr = row.find("args");

    // The env is built from the WHOLE row (top-level catalog_id + nested
    // "env" object OR flat fields), never from a "catalog" field.
    Value env_term = env_build::env_to_term(row);

    if (q == "resolve" || q == "resolve_layered") {
        std::vector<Value> req_vals;
        if (args_ptr && args_ptr->isArray()) {
            for (const auto& item : args_ptr->asArray()) {
                req_vals.push_back(term_build::request_term(item));
            }
        }
        Value reqs_list = list_of(std::move(req_vals));
        std::vector<Value> query_args = {
            std::move(env_term),
            std::move(reqs_list),
            Value::Unbound("Out")
        };
        const std::string pred = (q == "resolve") ? "resolve_store/3" : "resolve_layered_store/3";
        auto cells = query_capture(vm, pred, query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::sel_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "explain_blocked") {
        if (!args_ptr) throw std::runtime_error("explain_blocked missing args");
        Value req = term_build::request_term(*args_ptr);
        std::vector<Value> query_args = {
            std::move(env_term),
            std::move(req),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "explain_blocked_list_store/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::blocked_list_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "layer_closure") {
        if (!args_ptr) throw std::runtime_error("layer_closure missing args");
        Value req = term_build::request_term(*args_ptr);
        std::vector<Value> query_args = {
            std::move(env_term),
            std::move(req),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "layer_closure_store/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::sel_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "removal_orphans") {
        if (!args_ptr || !args_ptr->isString()) throw std::runtime_error("removal_orphans missing string args");
        std::vector<Value> query_args = {
            std::move(env_term),
            Value::Atom(args_ptr->asString()),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "removal_orphans_store/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::sel_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "safe_upgrade") {
        if (!args_ptr || !args_ptr->isArray() || args_ptr->size() < 2) {
            throw std::runtime_error("safe_upgrade expects 2-element args array");
        }
        const auto& a = args_ptr->asArray();
        std::vector<Value> query_args = {
            std::move(env_term),
            Value::Atom(a[0].asString()),
            term_build::ver_term(a[1]),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "safe_upgrade_store/4", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::normalize_verdict(*cells[3]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "upgrade_set") {
        if (!args_ptr || !args_ptr->isArray() || args_ptr->size() < 2) {
            throw std::runtime_error("upgrade_set expects 2-element args array");
        }
        const auto& a = args_ptr->asArray();
        std::vector<Value> query_args = {
            std::move(env_term),
            Value::Atom(a[0].asString()),
            term_build::ver_term(a[1]),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "upgrade_set_result_store/4", query_args);
        if (!cells.empty()) {
            json::Json r = term_json::normalize_upgrade(*cells[3]);
            if (r.find("ok")) {
                out.set("ok", *r.find("ok"));
            } else if (r.find("fail")) {
                out.set("fail", *r.find("fail"));
            }
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "freeze_audit") {
        std::vector<Value> query_args = {
            std::move(env_term),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "freeze_audit_store/2", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::audit_list_to_json(*cells[1]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "dependents") {
        if (!args_ptr || !args_ptr->isString()) throw std::runtime_error("dependents missing string args");
        std::vector<Value> query_args = {
            std::move(env_term),
            Value::Atom(args_ptr->asString()),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "dependents_store/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::sel_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "dependents_installed") {
        if (!args_ptr || !args_ptr->isString()) throw std::runtime_error("dependents_installed missing string args");
        std::vector<Value> query_args = {
            std::move(env_term),
            Value::Atom(args_ptr->asString()),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "dependents_installed_store/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::sel_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else {
        throw std::runtime_error("unknown query: " + q);
    }
}

} // namespace

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    WamState vm;
    wam_cpp::Program::apply_setup(vm);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json::Json out = json::Json::makeObject();
        try {
            json::Json row = json::parse(line);
            if (const json::Json* id_val = row.find("id")) {
                out.set("id", *id_val);
            } else {
                out.set("id", json::Json::makeNull());
            }
            run_case(vm, row, out);
        } catch (const std::exception& e) {
            out.set("crash", json::Json::makeString(e.what()));
        } catch (...) {
            out.set("crash", json::Json::makeString("unknown exception"));
        }
        std::cout << json::dump(out) << "\n";
    }

    // Opt-in cache/IO attribution for the memory×scale benchmark (stderr, so it
    // never pollutes the JSONL on stdout). D43 byte counters cover BOTH backends
    // (indexed ifstream reads and lmdb keyed scans); the L1/L2 hit/miss triple
    // is only nonzero for the lmdb lazy+cached backend.
    if (std::getenv("UW_WAM_CACHE_ATTRIBUTION")) {
        unsigned long long l1 = 0, l2 = 0, miss = 0;
        for (const auto& kv : vm.seek_fact_sources) {
            l1   += kv.second->l1_hits();
            l2   += kv.second->l2_hits();
            miss += kv.second->cache_misses();
        }
        std::fprintf(stderr,
            "[cache-attribution] fact_io_bytes=%llu fact_io_reads=%llu "
            "l1_hits=%llu l2_hits=%llu misses=%llu\n",
            (unsigned long long) wam_cpp::fact_io_bytes(),
            (unsigned long long) wam_cpp::fact_io_reads(),
            l1, l2, miss);
    }
    return 0;
}
