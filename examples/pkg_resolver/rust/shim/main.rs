// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// main.rs -- EDGE of the Rust-WAM compiled uw-resolve P0.5.
//
// WHAT IS IN HERE, exhaustively: conversion between JSON catalogs/requests
// and WAM `Value` terms, plus driving WamState::run for one entry predicate
// per query. There is NO resolver logic -- no candidate order, no constraint
// arithmetic, no layer walk, no topological sort. Those live in the generated
// crate (compiler output from examples/pkg_resolver/resolver.pl).
//
// Usage:
//   uw_resolve            < cases.jsonl > results.jsonl
//   uw_resolve --bench    < one_case.json          (load/resolve timings)

mod json;

use json::{parse, to_string, J};
use std::io::{self, Read, Write};
use std::time::Instant;
use uw_resolve_wam::state::WamState;
use uw_resolve_wam::value::Value;
use uw_resolve_wam::{setup_foreign_predicates, shared_wam_program};

// ---------------------------------------------------------------------------
// JSON -> WAM term
// ---------------------------------------------------------------------------

fn atom(name: &str) -> Value {
    Value::Atom(name.to_string())
}

fn s(functor: &str, args: Vec<Value>) -> Value {
    let arity = args.len();
    Value::strv(format!("{}/{}", functor, arity), args)
}

fn ver_term(v: &J) -> Value {
    let a = v.as_arr();
    let get = |i: usize| a.get(i).map(|x| x.as_i64()).unwrap_or(0);
    s(
        "v",
        vec![
            Value::Integer(get(0)),
            Value::Integer(get(1)),
            Value::Integer(get(2)),
        ],
    )
}

fn constraint_term(c: &J) -> Value {
    match c {
        J::Str(text) if text == "any" => atom("any"),
        J::Null => atom("any"),
        J::Obj(_) => {
            let op = c.get("op").map(|o| o.as_str().to_string()).unwrap_or_default();
            match op.as_str() {
                "eq" | "gte" | "lt" => s(&op, vec![ver_term(c.get("v").unwrap_or(&J::Null))]),
                "range" => s(
                    "range",
                    vec![
                        ver_term(c.get("lo").unwrap_or(&J::Null)),
                        ver_term(c.get("hi").unwrap_or(&J::Null)),
                    ],
                ),
                other => panic!("uw_resolve shim: unknown constraint op {}", other),
            }
        }
        other => panic!("uw_resolve shim: unknown constraint {:?}", other),
    }
}

fn pair_term(name: &str, ver: &J) -> Value {
    s("-", vec![atom(name), ver_term(ver)])
}

fn hold_term(row: &J) -> Value {
    let a = row.as_arr();
    let name = a.first().map(|x| x.as_str()).unwrap_or("");
    let nil = J::Null;
    let ver = a.get(1).unwrap_or(&nil);
    if a.len() >= 3 {
        s(
            "base",
            vec![pair_term(name, ver), atom(a[2].as_str())],
        )
    } else {
        pair_term(name, ver)
    }
}

fn layer_term(row: &J) -> Value {
    let name = row.get("name").map(|n| n.as_str()).unwrap_or("");
    let empty = J::Arr(vec![]);
    let pkgs = row.get("packages").unwrap_or(&empty);
    s(
        "layer",
        vec![
            atom(name),
            Value::list(pkgs.as_arr().iter().map(hold_term).collect()),
        ],
    )
}

fn alias_term(row: &J) -> Value {
    let a = row.as_arr();
    s(
        "alias",
        vec![
            atom(a.first().map(|x| x.as_str()).unwrap_or("")),
            atom(a.get(1).map(|x| x.as_str()).unwrap_or("")),
        ],
    )
}

fn pkg_term(row: &J) -> Value {
    let a = row.as_arr();
    let nil = J::Null;
    s(
        "package",
        vec![
            atom(a.first().map(|x| x.as_str()).unwrap_or("")),
            ver_term(a.get(1).unwrap_or(&nil)),
        ],
    )
}

fn dep_term(row: &J) -> Value {
    let a = row.as_arr();
    let nil = J::Null;
    s(
        "depends",
        vec![
            atom(a.first().map(|x| x.as_str()).unwrap_or("")),
            ver_term(a.get(1).unwrap_or(&nil)),
            atom(a.get(2).map(|x| x.as_str()).unwrap_or("")),
            constraint_term(a.get(3).unwrap_or(&nil)),
        ],
    )
}

fn conf_term(row: &J) -> Value {
    let a = row.as_arr();
    let nil = J::Null;
    s(
        "conflicts",
        vec![
            atom(a.first().map(|x| x.as_str()).unwrap_or("")),
            ver_term(a.get(1).unwrap_or(&nil)),
            atom(a.get(2).map(|x| x.as_str()).unwrap_or("")),
        ],
    )
}

fn installed_term(row: &J) -> Value {
    let a = row.as_arr();
    let nil = J::Null;
    pair_term(
        a.first().map(|x| x.as_str()).unwrap_or(""),
        a.get(1).unwrap_or(&nil),
    )
}

fn request_term(req: &J) -> Value {
    match req {
        J::Obj(_) if req.get("req").is_some() => s(
            "req",
            vec![
                atom(req.get("req").map(|r| r.as_str()).unwrap_or("")),
                constraint_term(req.get("constraint").unwrap_or(&J::Null)),
            ],
        ),
        other => atom(other.as_str()),
    }
}

/// catalog/6 when the P0.5 extras are absent or all empty (exactly the rule the
/// JS shim uses, so the two builds hand the compiled program the same term).
fn catalog_term(cat: &J) -> Value {
    let empty = J::Arr(vec![]);
    let list = |key: &str| cat.get(key).unwrap_or(&empty).as_arr().to_vec();
    let core = vec![
        Value::list(list("packages").iter().map(pkg_term).collect()),
        Value::list(list("depends").iter().map(dep_term).collect()),
        Value::list(list("conflicts").iter().map(conf_term).collect()),
        Value::list(list("base").iter().map(hold_term).collect()),
        Value::list(list("installed").iter().map(installed_term).collect()),
        Value::list(
            list("requested")
                .iter()
                .map(|r| atom(r.as_str()))
                .collect(),
        ),
    ];
    let layers = list("layers");
    let excluded = list("excluded");
    let aliases = list("aliases");
    if layers.is_empty() && excluded.is_empty() && aliases.is_empty() {
        return Value::strv("catalog/6".to_string(), core);
    }
    let mut args = core;
    args.push(Value::list(layers.iter().map(layer_term).collect()));
    args.push(Value::list(
        excluded.iter().map(|e| atom(e.as_str())).collect(),
    ));
    args.push(Value::list(aliases.iter().map(alias_term).collect()));
    Value::strv("catalog/9".to_string(), args)
}

// ---------------------------------------------------------------------------
// WAM term -> JSON
// ---------------------------------------------------------------------------

/// Functor name of a compound, with any `/arity` suffix stripped.
fn functor_of(v: &Value) -> Option<(String, &[Value])> {
    match v {
        Value::Str(f, args) => {
            let inner = f
                .strip_prefix("str(")
                .and_then(|x| x.strip_suffix(')'))
                .unwrap_or(f);
            let name = match inner.rsplit_once('/') {
                Some((n, ar)) if ar.chars().all(|c| c.is_ascii_digit()) && !ar.is_empty() => n,
                _ => inner,
            };
            Some((name.to_string(), args))
        }
        _ => None,
    }
}

fn ver_json(v: &Value) -> J {
    match functor_of(v) {
        Some((name, args)) if name == "v" && args.len() == 3 => J::Arr(
            args.iter()
                .map(|a| match a {
                    Value::Integer(n) => J::Int(*n),
                    Value::Float(f) => J::Int(*f as i64),
                    other => J::Str(format!("{}", other)),
                })
                .collect(),
        ),
        _ => J::Str(format!("{}", v)),
    }
}

fn constraint_json(v: &Value) -> J {
    match v {
        Value::Atom(a) if a == "any" => J::s("any"),
        _ => match functor_of(v) {
            Some((name, args)) if (name == "eq" || name == "gte" || name == "lt") && args.len() == 1 => {
                J::obj(vec![("op", J::Str(name)), ("v", ver_json(&args[0]))])
            }
            Some((name, args)) if name == "range" && args.len() == 2 => J::obj(vec![
                ("op", J::s("range")),
                ("lo", ver_json(&args[0])),
                ("hi", ver_json(&args[1])),
            ]),
            _ => J::Str(format!("{}", v)),
        },
    }
}

fn list_items(v: &Value) -> Vec<Value> {
    match v {
        Value::List(items) => items.to_vec(),
        Value::Atom(a) if a == "[]" => vec![],
        _ => match functor_of(v) {
            Some((name, args)) if (name == "[|]" || name == ".") && args.len() == 2 => {
                let mut out = vec![args[0].clone()];
                out.extend(list_items(&args[1]));
                out
            }
            _ => vec![],
        },
    }
}

/// One `Name-Ver` pair as `[name, [a,b,c]]`.
fn pair_json(v: &Value) -> J {
    match functor_of(v) {
        Some((name, args)) if name == "-" && args.len() == 2 => J::Arr(vec![
            J::Str(format!("{}", args[0])),
            ver_json(&args[1]),
        ]),
        _ => J::Str(format!("{}", v)),
    }
}

fn sel_json(v: &Value) -> J {
    J::Arr(list_items(v).iter().map(pair_json).collect())
}

fn blocked_json(v: &Value) -> J {
    match functor_of(v) {
        Some((name, args)) if name == "blocked" && args.len() == 3 => {
            let needs = match functor_of(&args[1]) {
                Some((n, a)) if n == "needs" && a.len() == 1 => constraint_json(&a[0]),
                _ => constraint_json(&args[1]),
            };
            let base_has = match functor_of(&args[2]) {
                Some((n, a)) if n == "base_has" && a.len() == 1 => ver_json(&a[0]),
                _ => ver_json(&args[2]),
            };
            J::obj(vec![
                ("base_has", base_has),
                ("name", J::Str(format!("{}", args[0]))),
                ("needs", needs),
            ])
        }
        _ => J::Str(format!("{}", v)),
    }
}

fn verdict_json(v: &Value) -> J {
    if let Value::Atom(a) = v {
        if a == "no_candidate" {
            return J::obj(vec![("verdict", J::s("no_candidate"))]);
        }
    }
    match functor_of(v) {
        Some((name, args)) if name == "safe" && args.len() == 1 => {
            let cost = match functor_of(&args[0]) {
                Some((n, a)) if n == "cost" && a.len() == 1 => J::Str(format!("{}", a[0])),
                _ => J::Str(format!("{}", args[0])),
            };
            J::obj(vec![("cost", cost), ("verdict", J::s("safe"))])
        }
        Some((name, args)) if name == "coordinated" && args.len() == 1 => J::obj(vec![
            ("set", sel_json(&args[0])),
            ("verdict", J::s("coordinated")),
        ]),
        Some((name, args)) if name == "unsafe" && args.len() == 1 => J::obj(vec![
            ("reason", J::Str(format!("{}", args[0]))),
            ("verdict", J::s("unsafe")),
        ]),
        _ => J::Str(format!("{}", v)),
    }
}

fn upgrade_json(v: &Value) -> J {
    if let Value::Atom(a) = v {
        if a == "no_candidate" {
            return J::obj(vec![("fail", J::Bool(true))]);
        }
    }
    match functor_of(v) {
        Some((name, args)) if name == "ok" && args.len() == 1 => {
            J::obj(vec![("ok", sel_json(&args[0]))])
        }
        Some((name, _)) if name == "blocked" => J::obj(vec![(
            "ok",
            J::obj(vec![("blocked", blocked_json(v))]),
        )]),
        _ => J::obj(vec![("ok", J::Str(format!("{}", v)))]),
    }
}

fn audit_json(v: &Value) -> J {
    match functor_of(v) {
        Some((name, args)) if name == "audit" && args.len() == 2 => {
            let who = J::Str(format!("{}", args[0]));
            if let Value::Atom(a) = &args[1] {
                if a == "over_frozen" {
                    return J::obj(vec![("kind", J::s("over_frozen")), ("name", who)]);
                }
            }
            match functor_of(&args[1]) {
                Some((n, a)) if (n == "suggest" || n == "held") && a.len() == 1 => J::obj(vec![
                    ("kind", J::Str(n)),
                    ("name", who),
                    ("reason", J::Str(format!("{}", a[0]))),
                ]),
                _ => J::obj(vec![
                    ("kind", J::s("held")),
                    ("name", who),
                    ("reason", J::Str(format!("{}", args[1]))),
                ]),
            }
        }
        _ => J::Str(format!("{}", v)),
    }
}

// ---------------------------------------------------------------------------
// Driving the compiled program
// ---------------------------------------------------------------------------

const OUT_VAR: &str = "_uw_shim_out";

/// Run `pred` with `args` in A1.. and one fresh output variable appended.
/// Returns the dereferenced output term, or None when the goal failed.
fn call_pred(vm: &mut WamState, pred: &str, args: Vec<Value>) -> Option<Value> {
    let target = *vm.labels.get(pred)?;
    vm.reset_query();
    for (i, a) in args.iter().enumerate() {
        vm.set_reg(&format!("A{}", i + 1), a.clone());
    }
    vm.set_reg(
        &format!("A{}", args.len() + 1),
        Value::Unbound(OUT_VAR.to_string()),
    );
    vm.cp = 0;
    vm.pc = target;
    if vm.run() {
        let out = vm.deref_heap(&Value::Unbound(OUT_VAR.to_string()));
        // A "success" that left the output variable unbound means the machine
        // returned without ever running the goal that binds it (e.g. a
        // premature jump to PC 0). Reporting that as an empty selection would
        // launder a runtime defect into a plausible-looking answer, so panic
        // loudly instead -- the runner turns it into a crash row.
        if out.is_unbound() {
            panic!("uw_resolve shim: {} succeeded with {} unbound", pred, OUT_VAR);
        }
        Some(out)
    } else {
        None
    }
}

fn run_case(vm: &mut WamState, row: &J) -> J {
    let empty = J::Null;
    let cat = catalog_term(row.get("catalog").unwrap_or(&empty));
    let query = row.get("query").map(|q| q.as_str()).unwrap_or("");
    let args = row.get("args").unwrap_or(&empty);

    let ok_or_fail = |r: Option<Value>, f: &dyn Fn(&Value) -> J| match r {
        Some(v) => J::obj(vec![("ok", f(&v))]),
        None => J::obj(vec![("fail", J::Bool(true))]),
    };

    match query {
        "resolve" | "resolve_layered" => {
            let pred = if query == "resolve" {
                "resolve/3"
            } else {
                "resolve_layered/3"
            };
            let reqs = Value::list(args.as_arr().iter().map(request_term).collect());
            ok_or_fail(call_pred(vm, pred, vec![cat, reqs]), &sel_json)
        }
        "explain_blocked" => {
            let req = request_term(args);
            match call_pred(vm, "explain_blocked_list/3", vec![cat, req]) {
                Some(v) => J::obj(vec![(
                    "ok",
                    J::Arr(list_items(&v).iter().map(blocked_json).collect()),
                )]),
                None => J::obj(vec![("fail", J::Bool(true))]),
            }
        }
        "layer_closure" => {
            let req = request_term(args);
            ok_or_fail(call_pred(vm, "layer_closure/3", vec![cat, req]), &sel_json)
        }
        "removal_orphans" => ok_or_fail(
            call_pred(vm, "removal_orphans/3", vec![cat, atom(args.as_str())]),
            &sel_json,
        ),
        "dependents" => ok_or_fail(
            call_pred(vm, "dependents/3", vec![cat, atom(args.as_str())]),
            &sel_json,
        ),
        "dependents_installed" => ok_or_fail(
            call_pred(vm, "dependents_installed/3", vec![cat, atom(args.as_str())]),
            &sel_json,
        ),
        "safe_upgrade" => {
            let a = args.as_arr();
            let nil = J::Null;
            let pkg = atom(a.first().map(|x| x.as_str()).unwrap_or(""));
            let ver = ver_term(a.get(1).unwrap_or(&nil));
            ok_or_fail(
                call_pred(vm, "safe_upgrade/4", vec![cat, pkg, ver]),
                &verdict_json,
            )
        }
        "upgrade_set" => {
            let a = args.as_arr();
            let nil = J::Null;
            let pkg = atom(a.first().map(|x| x.as_str()).unwrap_or(""));
            let ver = ver_term(a.get(1).unwrap_or(&nil));
            match call_pred(vm, "upgrade_set_result/4", vec![cat, pkg, ver]) {
                Some(v) => upgrade_json(&v),
                None => J::obj(vec![("fail", J::Bool(true))]),
            }
        }
        "freeze_audit" => match call_pred(vm, "freeze_audit/2", vec![cat]) {
            Some(v) => J::obj(vec![(
                "ok",
                J::Arr(list_items(&v).iter().map(audit_json).collect()),
            )]),
            None => J::obj(vec![("fail", J::Bool(true))]),
        },
        other => J::obj(vec![(
            "crash",
            J::Str(format!("unknown query {}", other)),
        )]),
    }
}

fn new_vm() -> WamState {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);
    vm
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let bench = argv.iter().any(|a| a == "--bench");

    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("read stdin");

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    if bench {
        // B3: one query on a large catalog, load (term build) and resolve timed
        // separately.
        let row = parse(input.trim()).expect("bench input must be one JSON object");
        let mut vm = new_vm();
        let t_load = Instant::now();
        let empty = J::Null;
        let cat = catalog_term(row.get("catalog").unwrap_or(&empty));
        let reqs = Value::list(
            row.get("args")
                .unwrap_or(&empty)
                .as_arr()
                .iter()
                .map(request_term)
                .collect(),
        );
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        let query = row.get("query").map(|q| q.as_str()).unwrap_or("resolve_layered");
        let pred = match query {
            "resolve" => "resolve/3",
            _ => "resolve_layered/3",
        };
        let t_run = Instant::now();
        let result = call_pred(&mut vm, pred, vec![cat, reqs]);
        let resolve_ms = t_run.elapsed().as_secs_f64() * 1000.0;
        let n = result.as_ref().map(|v| list_items(v).len()).unwrap_or(0);
        writeln!(out, "{}", to_string(&match result {
            Some(ref v) => J::obj(vec![("ok", sel_json(v))]),
            None => J::obj(vec![("fail", J::Bool(true))]),
        }))
        .unwrap();
        out.flush().unwrap();
        eprintln!("load_ms={:.3}", load_ms);
        eprintln!("resolve_ms={:.3}", resolve_ms);
        eprintln!("selection_size={}", n);
        return;
    }

    let mut vm = new_vm();
    for line in input.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = match parse(line) {
            Ok(r) => r,
            Err(e) => {
                writeln!(
                    out,
                    "{}",
                    to_string(&J::obj(vec![("crash", J::Str(format!("bad json: {}", e)))]))
                )
                .unwrap();
                continue;
            }
        };
        let id = row.get("id").cloned().unwrap_or(J::Null);
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_case(&mut vm, &row)
        })) {
            Ok(r) => r,
            Err(_) => J::obj(vec![("crash", J::s("shim panic (see stderr)"))]),
        };
        let mut fields: Vec<(String, J)> = vec![("id".to_string(), id)];
        if let J::Obj(inner) = result {
            fields.extend(inner);
        }
        writeln!(out, "{}", to_string(&J::Obj(fields))).unwrap();
    }
    out.flush().unwrap();
}
