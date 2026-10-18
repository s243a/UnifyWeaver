// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// diff_main.cpp -- JSONL differential runner for C++ WAM pkg_resolver.
// Reads JSON lines from stdin (each containing id, catalog, query, args),
// executes the query against the compiled WAM resolver state, and emits
// one compact JSON line to stdout per input line.

#include "wam_runtime.h"
#include "json.hpp"
#include "term_build.hpp"
#include "term_to_json.hpp"

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

// FIX: WamState::query() seeds "A1..Aarity" registers, then RUNS the
// compiled body. Multi-goal bodies reuse those SAME global A-registers as
// scratch space for their own nested calls (this is correct WAM behavior --
// Y-registers, not A-registers, are what threads a value across calls
// within a clause body). So by the time query() returns, "A<arity>" may no
// longer hold the query's real output -- it holds whatever the last nested
// call happened to leave there. The true output is still correctly bound
// (compiled clauses thread it via Y-registers back into the ORIGINAL
// argument cell by shared_ptr identity/mutation), so capturing that cell
// BEFORE run() starts, and reading IT afterward, is safe.
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
    const json::Json* cat_ptr = row.find("catalog");
    if (!cat_ptr) throw std::runtime_error("missing catalog");
    const json::Json* q_ptr = row.find("query");
    if (!q_ptr || !q_ptr->isString()) throw std::runtime_error("missing or invalid query");
    const std::string& q = q_ptr->asString();
    const json::Json* args_ptr = row.find("args");

    Value cat_term = term_build::catalog_to_term(*cat_ptr);

    if (q == "resolve") {
        std::vector<Value> req_vals;
        if (args_ptr && args_ptr->isArray()) {
            for (const auto& item : args_ptr->asArray()) {
                req_vals.push_back(term_build::request_term(item));
            }
        }
        Value reqs_list = list_of(std::move(req_vals));
        std::vector<Value> query_args = {
            std::move(cat_term),
            std::move(reqs_list),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "resolve/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::sel_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "resolve_layered") {
        std::vector<Value> req_vals;
        if (args_ptr && args_ptr->isArray()) {
            for (const auto& item : args_ptr->asArray()) {
                req_vals.push_back(term_build::request_term(item));
            }
        }
        Value reqs_list = list_of(std::move(req_vals));
        std::vector<Value> query_args = {
            std::move(cat_term),
            std::move(reqs_list),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "resolve_layered/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::sel_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "explain_blocked") {
        if (!args_ptr) throw std::runtime_error("explain_blocked missing args");
        Value req = term_build::request_term(*args_ptr);
        std::vector<Value> query_args = {
            std::move(cat_term),
            std::move(req),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "explain_blocked_list/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::blocked_list_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "layer_closure") {
        if (!args_ptr) throw std::runtime_error("layer_closure missing args");
        Value req = term_build::request_term(*args_ptr);
        std::vector<Value> query_args = {
            std::move(cat_term),
            std::move(req),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "layer_closure/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::term_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "removal_orphans") {
        if (!args_ptr || !args_ptr->isString()) throw std::runtime_error("removal_orphans missing string args");
        std::vector<Value> query_args = {
            std::move(cat_term),
            Value::Atom(args_ptr->asString()),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "removal_orphans/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::term_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "safe_upgrade") {
        if (!args_ptr || !args_ptr->isArray() || args_ptr->size() < 2) {
            throw std::runtime_error("safe_upgrade expects 2-element args array");
        }
        const auto& a = args_ptr->asArray();
        std::vector<Value> query_args = {
            std::move(cat_term),
            Value::Atom(a[0].asString()),
            term_build::ver_term(a[1]),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "safe_upgrade/4", query_args);
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
            std::move(cat_term),
            Value::Atom(a[0].asString()),
            term_build::ver_term(a[1]),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "upgrade_set_result/4", query_args);
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
            std::move(cat_term),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "freeze_audit/2", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::audit_list_to_json(*cells[1]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "dependents") {
        if (!args_ptr || !args_ptr->isString()) throw std::runtime_error("dependents missing string args");
        std::vector<Value> query_args = {
            std::move(cat_term),
            Value::Atom(args_ptr->asString()),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "dependents/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::term_to_json(*cells[2]));
        } else {
            out.set("fail", json::Json::makeBool(true));
        }
    } else if (q == "dependents_installed") {
        if (!args_ptr || !args_ptr->isString()) throw std::runtime_error("dependents_installed missing string args");
        std::vector<Value> query_args = {
            std::move(cat_term),
            Value::Atom(args_ptr->asString()),
            Value::Unbound("Out")
        };
        auto cells = query_capture(vm, "dependents_installed/3", query_args);
        if (!cells.empty()) {
            out.set("ok", term_json::term_to_json(*cells[2]));
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
    return 0;
}
