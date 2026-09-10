// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// main.rs -- EDGE of the Rust-WAM compiled uw-resolve P2 STORE adapter.
//
// WHAT IS IN HERE, exhaustively: conversion between JSON env/requests and WAM
// `Value` terms, plus driving WamState::run for one entry predicate per query.
// There is NO resolver logic and NO catalog: package/depends/conflicts/revdep/
// provides facts come from the D43 indexed seek stores compiled into the crate
// (register_indexed_seek_fact2). The machine-local environment stays a term.
// Mirrors examples/pkg_resolver/go_store/shim_store.go for the Rust lane, and
// reuses the term<->JSON readers of examples/pkg_resolver/rust/shim/main.rs so
// the store answers are byte-identical to the term-catalog answers.
//
// Usage:
//   uw_resolve_store              < cases.jsonl > results.jsonl
//   uw_resolve_store --corpus     < cases.jsonl        (also compares expected)
//   uw_resolve_store --scale-probe DIR                 (D43 bytes-read proof)

mod json;

use json::{parse, to_string, J};
use std::io::{self, Read, Write};
use std::time::Instant;
use uw_resolve_wam_store::seek_fact_source::{fact_io_bytes, fact_io_reads, reset_fact_io};
use uw_resolve_wam_store::state::WamState;
use uw_resolve_wam_store::value::Value;
use uw_resolve_wam_store::{setup_foreign_predicates, shared_wam_program};

// ---------------------------------------------------------------------------
// JSON -> WAM term
// ---------------------------------------------------------------------------

fn atom(name: &str) -> Value {
    Value::Atom(name.to_string().into())
}

fn s(functor: &str, args: Vec<Value>) -> Value {
    let arity = args.len();
    Value::strv(format!("{}/{}", functor, arity), args)
}

fn j_text(v: &J) -> String {
    match v {
        J::Str(s) => s.clone(),
        J::Int(n) => n.to_string(),
        J::Float(f) => format!("{}", *f as i64),
        _ => String::new(),
    }
}

/// P0 v/3 triples `[M,I,P]` and P3 `{"deb":[Epoch,[[order,num],…],[…]]}`.
fn ver_term(v: &J) -> Value {
    if let Some(d) = v.get("deb") {
        let a = d.as_arr();
        let epoch = a.first().map(|x| x.as_i64()).unwrap_or(0);
        let empty = J::Arr(vec![]);
        return s(
            "deb",
            vec![
                Value::Integer(epoch),
                segs_term(a.get(1).unwrap_or(&empty)),
                segs_term(a.get(2).unwrap_or(&empty)),
            ],
        );
    }
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

fn segs_term(v: &J) -> Value {
    let items: Vec<Value> = v
        .as_arr()
        .iter()
        .map(|seg| {
            let pair = seg.as_arr();
            let empty = J::s("");
            let order = j_text(pair.first().unwrap_or(&empty));
            let num = pair.get(1).map(|x| x.as_i64()).unwrap_or(0);
            let codes: Vec<Value> = order.chars().map(|c| Value::Integer(c as i64)).collect();
            s("s", vec![Value::list(codes), Value::Integer(num)])
        })
        .collect();
    Value::list(items)
}

fn constraint_term(c: &J) -> Value {
    match c {
        J::Str(text) if text == "any" => atom("any"),
        J::Null => atom("any"),
        J::Obj(_) => {
            let op = c.get("op").map(|o| o.as_str().to_string()).unwrap_or_default();
            match op.as_str() {
                "eq" | "gte" | "lt" | "lte" | "gt" => {
                    s(&op, vec![ver_term(c.get("v").unwrap_or(&J::Null))])
                }
                "range" => s(
                    "range",
                    vec![
                        ver_term(c.get("lo").unwrap_or(&J::Null)),
                        ver_term(c.get("hi").unwrap_or(&J::Null)),
                    ],
                ),
                other => panic!("uw_resolve_store shim: unknown constraint op {}", other),
            }
        }
        other => panic!("uw_resolve_store shim: unknown constraint {:?}", other),
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
        s("base", vec![pair_term(name, ver), atom(a[2].as_str())])
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

/// env(CatId, Base, Installed, Requested, Layers, Excluded, Aliases). No
/// catalog rows: those live in the compiled-in seek stores. Mirrors the Go
/// shim's envToTerm/envOf: read `env` when present, else fall back to the
/// `catalog` env fields with `catalog_id`.
fn env_term(row: &J) -> Value {
    let empty = J::Null;
    let (env, cat_fallback) = match row.get("env") {
        Some(e @ J::Obj(_)) => (e, false),
        _ => (row.get("catalog").unwrap_or(&empty), true),
    };
    let cat_id = row
        .get("catalog_id")
        .map(|c| c.as_str().to_string())
        .or_else(|| {
            env.get("catalog_id").map(|c| c.as_str().to_string())
        })
        .unwrap_or_else(|| "default".to_string());
    let _ = cat_fallback;
    let empty_arr = J::Arr(vec![]);
    let list = |key: &str| env.get(key).unwrap_or(&empty_arr).as_arr().to_vec();
    s(
        "env",
        vec![
            atom(&cat_id),
            Value::list(list("base").iter().map(hold_term).collect()),
            Value::list(list("installed").iter().map(installed_term).collect()),
            Value::list(list("requested").iter().map(|r| atom(r.as_str())).collect()),
            Value::list(list("layers").iter().map(layer_term).collect()),
            Value::list(list("excluded").iter().map(|e| atom(e.as_str())).collect()),
            Value::list(list("aliases").iter().map(alias_term).collect()),
        ],
    )
}

// ---------------------------------------------------------------------------
// WAM term -> JSON  (identical to examples/pkg_resolver/rust/shim/main.rs)
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

fn int_json(v: &Value) -> J {
    match v {
        Value::Integer(n) => J::Int(*n),
        Value::Float(f) => J::Int(*f as i64),
        other => J::Str(format!("{}", other)),
    }
}

fn codes_to_string(v: &Value) -> String {
    list_items(v)
        .iter()
        .filter_map(|c| match c {
            Value::Integer(n) => char::from_u32(*n as u32),
            _ => None,
        })
        .collect()
}

fn ver_json(v: &Value) -> J {
    match functor_of(v) {
        Some((name, args)) if name == "v" && args.len() == 3 => {
            J::Arr(args.iter().map(int_json).collect())
        }
        Some((name, args)) if name == "s" && args.len() == 2 => {
            J::Arr(vec![J::s(&codes_to_string(&args[0])), int_json(&args[1])])
        }
        Some((name, args)) if name == "deb" && args.len() == 3 => J::obj(vec![(
            "deb",
            J::Arr(vec![
                int_json(&args[0]),
                J::Arr(list_items(&args[1]).iter().map(ver_json).collect()),
                J::Arr(list_items(&args[2]).iter().map(ver_json).collect()),
            ]),
        )]),
        _ => J::Str(format!("{}", v)),
    }
}

fn constraint_json(v: &Value) -> J {
    match v {
        Value::Atom(a) if a == "any" => J::s("any"),
        _ => match functor_of(v) {
            Some((name, args))
                if (name == "eq"
                    || name == "gte"
                    || name == "lt"
                    || name == "lte"
                    || name == "gt")
                    && args.len() == 1 =>
            {
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
        Some((name, args)) if name == "-" && args.len() == 2 => {
            J::Arr(vec![J::Str(format!("{}", args[0])), ver_json(&args[1])])
        }
        _ => J::Str(format!("{}", v)),
    }
}

fn sel_json(v: &Value) -> J {
    J::Arr(list_items(v).iter().map(pair_json).collect())
}

fn atom_json(v: &Value) -> J {
    match v {
        Value::Atom(a) => J::Str(a.as_str().to_string()),
        other => J::Str(format!("{}", other)),
    }
}

fn unwrap_functor<'a>(v: &'a Value, name: &str) -> &'a Value {
    match functor_of(v) {
        Some((n, args)) if n == name && args.len() == 1 => &args[0],
        _ => v,
    }
}

fn alt_reason_json(v: &Value) -> J {
    match functor_of(v) {
        Some((name, args)) if name == "alt" && args.len() == 2 => {
            let reason = match &args[1] {
                Value::Atom(a) if a == "unsatisfiable" => J::s("unsatisfiable"),
                other => match functor_of(other) {
                    Some((n, _)) if n == "blocked" => blocked_json(other),
                    _ => atom_json(other),
                },
            };
            J::obj(vec![("dep", atom_json(&args[0])), ("reason", reason)])
        }
        _ => J::Str(format!("{}", v)),
    }
}

fn blocked_json(v: &Value) -> J {
    match functor_of(v) {
        Some((name, args)) if name == "blocked" && args.len() == 1 => {
            match functor_of(&args[0]) {
                Some((n, a)) if n == "alternatives" && a.len() == 1 => J::obj(vec![(
                    "alternatives",
                    J::Arr(list_items(&a[0]).iter().map(alt_reason_json).collect()),
                )]),
                _ => J::Str(format!("{}", v)),
            }
        }
        Some((name, args)) if name == "blocked" && args.len() == 3 => {
            let needs = constraint_json(unwrap_functor(&args[1], "needs"));
            match functor_of(&args[2]) {
                Some((n, a)) if n == "providers" && a.len() == 1 => J::obj(vec![
                    ("name", atom_json(&args[0])),
                    ("needs", needs),
                    (
                        "providers",
                        J::Arr(list_items(&a[0]).iter().map(blocked_json).collect()),
                    ),
                ]),
                Some((n, a)) if n == "base_has" && a.len() == 1 => J::obj(vec![
                    ("base_has", ver_json(&a[0])),
                    ("name", atom_json(&args[0])),
                    ("needs", needs),
                ]),
                _ => J::obj(vec![
                    ("base_has", ver_json(&args[2])),
                    ("name", atom_json(&args[0])),
                    ("needs", needs),
                ]),
            }
        }
        Some((name, args)) if name == "alt" && args.len() == 2 => alt_reason_json(v),
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
        Some((name, _)) if name == "blocked" => {
            J::obj(vec![("ok", J::obj(vec![("blocked", blocked_json(v))]))])
        }
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
fn call_pred(vm: &mut WamState, pred: &str, args: Vec<Value>) -> Option<Value> {
    let target = *vm.labels.get(pred)?;
    vm.reset_query();
    for (i, a) in args.iter().enumerate() {
        vm.set_reg(&format!("A{}", i + 1), a.clone());
    }
    vm.set_reg(
        &format!("A{}", args.len() + 1),
        Value::Unbound(OUT_VAR.to_string().into()),
    );
    vm.cp = 0;
    vm.pc = target;
    if vm.run() {
        let out = vm.deref_heap(&Value::Unbound(OUT_VAR.to_string().into()));
        if out.is_unbound() {
            panic!("uw_resolve_store shim: {} succeeded with {} unbound", pred, OUT_VAR);
        }
        Some(out)
    } else {
        None
    }
}

fn run_case(vm: &mut WamState, row: &J) -> J {
    let empty = J::Null;
    let env = env_term(row);
    let query = row.get("query").map(|q| q.as_str()).unwrap_or("");
    let args = row.get("args").unwrap_or(&empty);

    let ok_or_fail = |r: Option<Value>, f: &dyn Fn(&Value) -> J| match r {
        Some(v) => J::obj(vec![("ok", f(&v))]),
        None => J::obj(vec![("fail", J::Bool(true))]),
    };

    match query {
        "resolve" | "resolve_layered" => {
            let pred = if query == "resolve" {
                "resolve_store/3"
            } else {
                "resolve_layered_store/3"
            };
            let reqs = Value::list(args.as_arr().iter().map(request_term).collect());
            ok_or_fail(call_pred(vm, pred, vec![env, reqs]), &sel_json)
        }
        "explain_blocked" => {
            let req = request_term(args);
            match call_pred(vm, "explain_blocked_list_store/3", vec![env, req]) {
                Some(v) => J::obj(vec![(
                    "ok",
                    J::Arr(list_items(&v).iter().map(blocked_json).collect()),
                )]),
                None => J::obj(vec![("fail", J::Bool(true))]),
            }
        }
        "layer_closure" => {
            let req = request_term(args);
            ok_or_fail(
                call_pred(vm, "layer_closure_store/3", vec![env, req]),
                &sel_json,
            )
        }
        "removal_orphans" => ok_or_fail(
            call_pred(vm, "removal_orphans_store/3", vec![env, atom(args.as_str())]),
            &sel_json,
        ),
        "dependents" => ok_or_fail(
            call_pred(vm, "dependents_store/3", vec![env, atom(args.as_str())]),
            &sel_json,
        ),
        "dependents_installed" => ok_or_fail(
            call_pred(vm, "dependents_installed_store/3", vec![env, atom(args.as_str())]),
            &sel_json,
        ),
        "safe_upgrade" => {
            let a = args.as_arr();
            let nil = J::Null;
            let pkg = atom(a.first().map(|x| x.as_str()).unwrap_or(""));
            let ver = ver_term(a.get(1).unwrap_or(&nil));
            ok_or_fail(
                call_pred(vm, "safe_upgrade_store/4", vec![env, pkg, ver]),
                &verdict_json,
            )
        }
        "upgrade_set" => {
            let a = args.as_arr();
            let nil = J::Null;
            let pkg = atom(a.first().map(|x| x.as_str()).unwrap_or(""));
            let ver = ver_term(a.get(1).unwrap_or(&nil));
            match call_pred(vm, "upgrade_set_result_store/4", vec![env, pkg, ver]) {
                Some(v) => upgrade_json(&v),
                None => J::obj(vec![("fail", J::Bool(true))]),
            }
        }
        "freeze_audit" => match call_pred(vm, "freeze_audit_store/2", vec![env]) {
            Some(v) => J::obj(vec![(
                "ok",
                J::Arr(list_items(&v).iter().map(audit_json).collect()),
            )]),
            None => J::obj(vec![("fail", J::Bool(true))]),
        },
        other => J::obj(vec![("crash", J::Str(format!("unknown query {}", other)))]),
    }
}

fn new_vm() -> WamState {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);
    vm
}

/// Order-independent JSON serialisation (object keys sorted, recursively), so
/// corpus comparison against `expected` does not depend on field order.
fn stable_stringify(v: &J) -> String {
    match v {
        J::Null => "null".to_string(),
        J::Bool(b) => b.to_string(),
        J::Int(n) => n.to_string(),
        J::Float(f) => {
            if f.fract() == 0.0 && f.is_finite() {
                (*f as i64).to_string()
            } else {
                format!("{}", f)
            }
        }
        J::Str(_) => to_string(v),
        J::Arr(items) => {
            let parts: Vec<String> = items.iter().map(stable_stringify).collect();
            format!("[{}]", parts.join(","))
        }
        J::Obj(fields) => {
            let mut kv: Vec<(&String, &J)> = fields.iter().map(|(k, val)| (k, val)).collect();
            kv.sort_by(|a, b| a.0.cmp(b.0));
            let parts: Vec<String> = kv
                .iter()
                .map(|(k, val)| format!("{}:{}", to_string(&J::Str((*k).clone())), stable_stringify(val)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn run_scale_probe(dir: &str) -> i32 {
    let probe_path = format!("{}/probe.json", dir);
    let mut text = String::new();
    match std::fs::File::open(&probe_path).and_then(|mut f| f.read_to_string(&mut text).map(|_| ())) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{}: {}", probe_path, e);
            return 2;
        }
    }
    let probe = parse(text.trim()).expect("probe.json must be one JSON object");

    // Total store size = sum of the five .data blobs (what a term-catalog load
    // would have to read in full).
    let mut total_store: u64 = 0;
    for name in ["pkg", "dep", "conflict", "revdep", "provide"] {
        if let Ok(md) = std::fs::metadata(format!("{}/{}.data", dir, name)) {
            total_store += md.len();
        }
    }

    let mut vm = new_vm();
    reset_fact_io();
    let t0 = Instant::now();
    let got = run_case(&mut vm, &probe);
    let resolve = t0.elapsed().as_secs_f64();
    let bytes_read = fact_io_bytes();
    let reads = fact_io_reads();

    let n = got
        .get("ok")
        .map(|v| v.as_arr().len())
        .unwrap_or(0);

    println!("rust_store_resolve_s {:.3}", resolve);
    println!("rust_store_total_s {:.3}", resolve);
    println!("rust_store_bytes_read {}", bytes_read);
    println!("rust_store_n_reads {}", reads);
    println!("rust_store_total_store_bytes {}", total_store);
    if total_store > 0 {
        println!(
            "rust_store_read_fraction {:.6}",
            bytes_read as f64 / total_store as f64
        );
    }
    println!("rust_store_selection_size {}", n);
    println!("rust_store_result {}", to_string(&got));
    0
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();

    if argv.iter().any(|a| a == "--scale-probe") {
        let dir = argv
            .iter()
            .position(|a| a == "--scale-probe")
            .and_then(|i| argv.get(i + 1))
            .cloned()
            .unwrap_or_else(|| ".".to_string());
        std::process::exit(run_scale_probe(&dir));
    }

    let corpus = argv.iter().any(|a| a == "--corpus");

    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("read stdin");

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let mut vm = new_vm();
    let mut n = 0;
    let mut divergences = 0;
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
        let mut fields: Vec<(String, J)> = vec![("id".to_string(), id.clone())];
        if let J::Obj(ref inner) = result {
            fields.extend(inner.clone());
        }
        writeln!(out, "{}", to_string(&J::Obj(fields))).unwrap();

        if corpus {
            n += 1;
            let empty = J::Null;
            let exp = row.get("expected").unwrap_or(&empty);
            if stable_stringify(&result) != stable_stringify(exp) {
                divergences += 1;
                eprintln!(
                    "DIVERGE {}\n  expected {}\n  got      {}",
                    to_string(&id),
                    stable_stringify(exp),
                    stable_stringify(&result)
                );
            } else {
                eprintln!("ok {}", to_string(&id));
            }
        }
    }
    out.flush().unwrap();

    if corpus {
        if divergences != 0 {
            eprintln!("corpus-under-rust-store: {} divergences / {}", divergences, n);
            std::process::exit(1);
        }
        eprintln!("corpus-under-rust-store: {}/{} matched SWI", n, n);
    }
}
