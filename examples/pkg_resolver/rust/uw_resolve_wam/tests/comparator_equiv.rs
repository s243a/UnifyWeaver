// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// Round #1 (decorate-sort) comparator stress test.
//
// `term_compare_derefed` is the decorate-sort comparator: it trusts that both
// operands are ALREADY fully dereferenced (the invariant the sort/msort/
// keysort/setof builtins establish by pre-dereffing each element once) and so
// does no `deref_heap`/`deref_var` at any level and never allocates a functor
// String per comparison. This test pins the correctness contract that makes
// that safe: on already-dereferenced terms it must produce the EXACT same
// ordering as the original `term_compare` (which re-derefs both operands), for
// every shape in the standard order of terms and every edge case, so that the
// decorate_sort=ON build is byte-identical to the OFF build.
//
// This is the direct `term_compare_derefed == term_compare` equivalence check;
// the end-to-end byte-identity is covered by the corpus / differential / 5k
// selection gates run against the ON and OFF binaries.

use std::cmp::Ordering;
use std::collections::HashMap;
use uw_resolve_wam::state::WamState;
use uw_resolve_wam::value::Value;

fn state() -> WamState {
    // An empty program is enough: the comparator reads no code, and with no
    // bindings `deref_heap(deref_var(v))` is the identity on a term already in
    // dereferenced form (bare functors, unbound vars, no heap Refs) -- exactly
    // the form the pre-dereffing sort builtins hand to the comparator.
    WamState::new(Vec::new(), HashMap::new())
}

fn atom(s: &str) -> Value {
    Value::Atom(s.to_string().into())
}
fn var(s: &str) -> Value {
    Value::Unbound(s.to_string().into())
}
fn cmpd(f: &str, args: Vec<Value>) -> Value {
    // Bare functor name, as `deref_heap` normalises it via `functor_of`.
    Value::strv(f.to_string(), args)
}

/// A representative corpus of FULLY-DEREFERENCED terms covering every branch
/// of the standard order of terms: Var < Number < Atom/[] < Compound, with
/// the intra-class edge cases (Float before an equal Integer; atoms textually;
/// lists cons-wise with a strict prefix before its extension; compounds by
/// arity then functor name then args).
fn corpus() -> Vec<Value> {
    vec![
        // class 0 -- variables, ordered by internal name
        var("_G0"),
        var("_G1"),
        var("_G10"),
        var("Z"),
        // class 1 -- numbers, by value, Float before an equal Integer
        Value::Integer(-100),
        Value::Integer(-1),
        Value::Float(-1.0),
        Value::Integer(0),
        Value::Float(0.0),
        Value::Float(0.5),
        Value::Integer(1),
        Value::Float(1.0),
        Value::Integer(2),
        Value::Float(2.5),
        Value::Integer(1000000),
        // class 2 -- atoms (and the empty list, which orders as the atom "[]")
        atom(""),
        atom("[]"),
        Value::list(Vec::new()),
        atom("a"),
        atom("aa"),
        atom("ab"),
        atom("b"),
        atom("foo"),
        atom("zzz"),
        Value::Bool(false),
        Value::Bool(true),
        // class 3 -- lists (cons-wise; a strict prefix precedes its extension)
        Value::list(vec![Value::Integer(1)]),
        Value::list(vec![Value::Integer(1), Value::Integer(2)]),
        Value::list(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]),
        Value::list(vec![Value::Integer(2)]),
        Value::list(vec![atom("a"), atom("b")]),
        Value::list(vec![atom("a"), Value::Integer(1)]),
        // class 3 -- compounds, by arity then functor name then args
        cmpd("f", vec![atom("x")]),
        cmpd("f", vec![atom("y")]),
        cmpd("g", vec![atom("x")]),
        cmpd("f", vec![atom("x"), atom("y")]),
        cmpd("f", vec![atom("x"), atom("z")]),
        cmpd("g", vec![atom("x"), atom("y")]),
        cmpd("h", vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]),
        // functor names with embedded slashes (functor_of / display_functor_name
        // normalisation must agree): "a/b" (non-numeric tail, kept)
        cmpd("a/b", vec![atom("x"), atom("y")]),
        // nested / deep terms mixing every class
        cmpd(
            "node",
            vec![
                Value::list(vec![Value::Integer(1), cmpd("g", vec![atom("a")])]),
                cmpd("pair", vec![var("_K"), Value::Float(3.5)]),
            ],
        ),
        cmpd(
            "node",
            vec![
                Value::list(vec![Value::Integer(1), cmpd("g", vec![atom("a")])]),
                cmpd("pair", vec![var("_K"), Value::Float(3.6)]),
            ],
        ),
        deep(6),
        deep(7),
        // duplicates (for the dedup-equivalence check)
        atom("a"),
        Value::Integer(1),
        cmpd("f", vec![atom("x"), atom("y")]),
    ]
}

// A left-nested compound of the given depth: f(f(f(...(a)...))).
fn deep(n: usize) -> Value {
    let mut v = atom("a");
    for _ in 0..n {
        v = cmpd("f", vec![v]);
    }
    v
}

#[test]
fn derefed_matches_term_compare_on_all_pairs() {
    let s = state();
    let terms = corpus();
    for (i, a) in terms.iter().enumerate() {
        for (j, b) in terms.iter().enumerate() {
            let base = s.term_compare(a, b);
            let deref = s.term_compare_derefed(a, b);
            assert_eq!(
                base, deref,
                "ordering mismatch at ({i},{j}): term_compare={base:?} term_compare_derefed={deref:?}"
            );
        }
    }
}

#[test]
fn derefed_is_self_consistent() {
    // Antisymmetry and reflexivity of the decorate-sort comparator, so it is a
    // valid total order for sort_by/dedup_by.
    let s = state();
    let terms = corpus();
    for a in &terms {
        assert_eq!(s.term_compare_derefed(a, a), Ordering::Equal);
    }
    for a in &terms {
        for b in &terms {
            let ab = s.term_compare_derefed(a, b);
            let ba = s.term_compare_derefed(b, a);
            assert_eq!(ab, ba.reverse(), "not antisymmetric");
        }
    }
}

#[test]
fn sort_and_dedup_are_identical_under_both_comparators() {
    let s = state();
    let terms = corpus();

    let mut by_base = terms.clone();
    by_base.sort_by(|a, b| s.term_compare(a, b));
    let mut by_deref = terms.clone();
    by_deref.sort_by(|a, b| s.term_compare_derefed(a, b));

    assert_eq!(by_base.len(), by_deref.len());
    for (k, (a, b)) in by_base.iter().zip(by_deref.iter()).enumerate() {
        assert_eq!(
            s.term_compare(a, b),
            Ordering::Equal,
            "sorted sequences diverge at position {k}"
        );
    }

    // sort/2 dedup path: adjacent-equal removal must land on the same set.
    let mut ded_base = by_base.clone();
    ded_base.dedup_by(|a, b| s.term_compare(a, b) == Ordering::Equal);
    let mut ded_deref = by_deref.clone();
    ded_deref.dedup_by(|a, b| s.term_compare_derefed(a, b) == Ordering::Equal);
    assert_eq!(
        ded_base.len(),
        ded_deref.len(),
        "dedup produced different lengths"
    );
    for (k, (a, b)) in ded_base.iter().zip(ded_deref.iter()).enumerate() {
        assert_eq!(
            s.term_compare(a, b),
            Ordering::Equal,
            "deduped sequences diverge at position {k}"
        );
    }
}

#[test]
fn float_orders_before_equal_integer() {
    // A named edge case of the standard order: numeric value ties break with
    // Float preceding Integer.
    let s = state();
    assert_eq!(
        s.term_compare_derefed(&Value::Float(1.0), &Value::Integer(1)),
        Ordering::Less
    );
    assert_eq!(
        s.term_compare_derefed(&Value::Integer(1), &Value::Float(1.0)),
        Ordering::Greater
    );
}

#[test]
fn class_boundaries_hold() {
    // Var < Number < Atom < Compound.
    let s = state();
    let v = var("_X");
    let n = Value::Integer(0);
    let a = atom("a");
    let c = cmpd("f", vec![atom("x")]);
    assert_eq!(s.term_compare_derefed(&v, &n), Ordering::Less);
    assert_eq!(s.term_compare_derefed(&n, &a), Ordering::Less);
    assert_eq!(s.term_compare_derefed(&a, &c), Ordering::Less);
    // empty list orders as an atom, below any non-empty compound/list
    let empty = Value::list(Vec::new());
    let one = Value::list(vec![Value::Integer(1)]);
    assert_eq!(s.term_compare_derefed(&empty, &one), Ordering::Less);
}
