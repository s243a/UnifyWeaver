// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// Hot-path opt #2 (interning) stress tests.
//
// These pin the properties interning must preserve for byte-identical output,
// and in particular the SORT-ORDER TRAP: interned ids are assigned in first-seen
// order, NOT name order, so standard order of terms must still compare by NAME.
// The comparators de-intern to &str for the ordering decision (approach a); these
// tests prove that holds even when the interned ids are deliberately NOT in name
// order. They pass under BOTH the `intern` feature (Sym = u32 id) and the default
// (Sym = String), so they hold for the ON and OFF builds alike.

use std::cmp::Ordering;
use std::collections::HashMap;
use uw_resolve_wam::state::WamState;
use uw_resolve_wam::value::Value;

fn state() -> WamState {
    WamState::new(Vec::new(), HashMap::new())
}
fn atom(s: &str) -> Value {
    Value::Atom(s.to_string().into())
}
fn cmpd(f: &str, args: Vec<Value>) -> Value {
    Value::strv(f.to_string(), args)
}

// ---------------------------------------------------------------------------
// The sort-order trap, end to end through the comparator.
// ---------------------------------------------------------------------------

/// Force the interner to assign ids in the OPPOSITE order to the names, then
/// prove that sorting orders by NAME (not by id). Under the OFF build this is a
/// plain name sort; under ON it is the real trap: if the comparator compared
/// ids, the result would come out reversed.
#[test]
fn sort_orders_by_name_not_intern_id() {
    // Intern in descending-name order first so the earliest (smallest) ids map
    // to the largest names. Unique prefix avoids collision with names interned
    // by other tests in the same process.
    let names_desc = [
        "zzz_trap_h", "zzz_trap_g", "zzz_trap_f", "zzz_trap_e",
        "zzz_trap_d", "zzz_trap_c", "zzz_trap_b", "zzz_trap_a",
    ];
    // Touch them in descending order to bias id assignment.
    let _bias: Vec<Value> = names_desc.iter().map(|n| atom(n)).collect();

    let vm = state();
    // Now present them shuffled and sort with the decorate-sort comparator.
    let mut terms: Vec<Value> = [
        "zzz_trap_c", "zzz_trap_a", "zzz_trap_h", "zzz_trap_e",
        "zzz_trap_b", "zzz_trap_g", "zzz_trap_d", "zzz_trap_f",
    ]
    .iter()
    .map(|n| atom(n))
    .collect();
    terms.sort_by(|a, b| vm.term_compare_derefed(a, b));

    let got: Vec<String> = terms.iter().map(|t| format!("{}", t)).collect();
    let want = vec![
        "zzz_trap_a", "zzz_trap_b", "zzz_trap_c", "zzz_trap_d",
        "zzz_trap_e", "zzz_trap_f", "zzz_trap_g", "zzz_trap_h",
    ];
    assert_eq!(got, want, "atoms must sort by NAME, not interned id");

    // Same via the raw (re-deref) comparator -- both must agree.
    let mut terms2 = terms.clone();
    terms2.sort_by(|a, b| vm.term_compare(a, b));
    let got2: Vec<String> = terms2.iter().map(|t| format!("{}", t)).collect();
    assert_eq!(got2, want, "raw comparator must also sort by NAME");
}

/// Compound functors are also ordered by NAME (arity first, then functor name,
/// then args). Force reverse-id assignment on the functor names.
#[test]
fn compound_functor_orders_by_name_not_id() {
    let _bias = [
        cmpd("zzz_fn_d", vec![Value::Integer(0)]),
        cmpd("zzz_fn_c", vec![Value::Integer(0)]),
        cmpd("zzz_fn_b", vec![Value::Integer(0)]),
        cmpd("zzz_fn_a", vec![Value::Integer(0)]),
    ];
    let vm = state();
    let mut terms = vec![
        cmpd("zzz_fn_c", vec![Value::Integer(1)]),
        cmpd("zzz_fn_a", vec![Value::Integer(1)]),
        cmpd("zzz_fn_d", vec![Value::Integer(1)]),
        cmpd("zzz_fn_b", vec![Value::Integer(1)]),
    ];
    terms.sort_by(|a, b| vm.term_compare_derefed(a, b));
    let got: Vec<String> = terms.iter().map(|t| format!("{}", t)).collect();
    assert_eq!(
        got,
        vec!["zzz_fn_a(1)", "zzz_fn_b(1)", "zzz_fn_c(1)", "zzz_fn_d(1)"],
        "compound functors must sort by NAME, not interned id"
    );
}

// ---------------------------------------------------------------------------
// De-intern / round-trip through Display (output boundary).
// ---------------------------------------------------------------------------

#[test]
fn display_round_trips_names() {
    // Names first seen at runtime (never at codegen) must round-trip through the
    // Display de-intern.
    let names = ["\u{1f600}emoji", "with space", "sym-with-dash", "", "café", "!@#$%"];
    for n in names {
        assert_eq!(format!("{}", atom(n)), n, "atom name must de-intern verbatim");
    }
    // Compound display uses the de-interned functor + args.
    let t = cmpd("dyn_functor", vec![atom("arg_one"), Value::Integer(7)]);
    assert_eq!(format!("{}", t), "dyn_functor(arg_one, 7)");
}

// ---------------------------------------------------------------------------
// Equality is by NAME (canonical interning): id-equality iff name-equality.
// ---------------------------------------------------------------------------

#[test]
fn atom_equality_is_by_name() {
    // Two independently-constructed atoms with the same name are equal, even
    // though they were built at different sites.
    assert_eq!(atom("shared_name"), atom("shared_name"));
    assert_ne!(atom("name_x"), atom("name_y"));
    // A name first seen now still equals a later construction of the same name.
    let a = atom("late_bound_name_42");
    let b = atom("late_bound_name_42");
    assert_eq!(a, b);
    // Compound equality: same functor + args.
    assert_eq!(
        cmpd("eqf", vec![atom("p"), Value::Integer(1)]),
        cmpd("eqf", vec![atom("p"), Value::Integer(1)])
    );
    assert_ne!(
        cmpd("eqf", vec![atom("p")]),
        cmpd("eqf", vec![atom("q")])
    );
}

// ---------------------------------------------------------------------------
// Antisymmetry / reflexivity of the comparator over an intern-scrambled corpus.
// ---------------------------------------------------------------------------

#[test]
fn comparator_is_a_total_order_over_interned_names() {
    // Names interned in an order unrelated to their sort order.
    let names = [
        "q_9", "q_1", "q_44", "q_2", "q_10", "q_3", "q_100", "q_0",
    ];
    let vm = state();
    let terms: Vec<Value> = names.iter().map(|n| atom(n)).collect();
    for a in &terms {
        assert_eq!(vm.term_compare_derefed(a, a), Ordering::Equal, "reflexive");
        for b in &terms {
            let ab = vm.term_compare_derefed(a, b);
            let ba = vm.term_compare_derefed(b, a);
            assert_eq!(ab, ba.reverse(), "antisymmetric");
            // Must match the NAME ordering (string compare of the atom names).
            let na = format!("{}", a);
            let nb = format!("{}", b);
            assert_eq!(ab, na.cmp(&nb), "ordering must equal name ordering");
        }
    }
}
