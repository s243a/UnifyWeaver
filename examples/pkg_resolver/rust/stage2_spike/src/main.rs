// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// ============================================================================
// THROWAWAY Stage-2 measurement spike -- NOT part of any default build.
// ============================================================================
//
// Question (WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md §5): does fusing a chain of
// deterministic-in-practice predicates into ONE native region, entered with a
// cheap minimal-locals snapshot (P2, clears O2) and connected by direct native
// calls (P1, clears O1), beat the interpreter -- whose per-call machinery is the
// recorded 21% backtrack + 12% restore_regs?
//
// Family measured: matching_deps/4 -> dep_to_req/3 (B2 15.2% of dispatches).
//
//   matching_deps([], _Name, _Ver, []).
//   matching_deps([depends(N,V,D,C)|Rest], Name, Ver, Out) :-
//       ( N==Name, V==Ver -> dep_to_req(D,C,Req), Out=[Req|Rs] ; Out=Rs ),
//       matching_deps(Rest, Name, Ver, Rs).
//   dep_to_req(alternatives(Alts), _C, req(alternatives(Alts), any)) :- !.
//   dep_to_req(D, C, req(D, C)).
//
// The value model (value.rs) is VENDORED VERBATIM from the real runtime, so
// Value::clone cost (String allocs for Atom/Str/Unbound; O(1) Arc bumps for
// List/compound spines) is exactly the runtime's. save_regs/restore_regs are
// COPIED VERBATIM from state.rs (the two functions whose cost this spike is
// about). So the machinery numbers are the runtime's own, not a re-model.
//
// What is held constant between the two paths: the *body work* (head unify,
// N==Name/V==Ver guard, dep_to_req rewrite, cons of the output list). Fusion
// does not remove that logical work -- it removes/replaces the per-call
// CHOICE-POINT MACHINERY (snapshots, restores, stack Arc-clones, re-dispatch).
// So the honest comparison is machinery-per-element, body held equal, plus a
// full per-call breakdown (snapshot / dispatch / body).

#![allow(dead_code)] // value.rs is vendored verbatim; not all helpers are used here.
mod value;
use value::Value;

use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Register file + snapshot machinery -- copied VERBATIM from state.rs.
// ---------------------------------------------------------------------------

const MAX_REGS: usize = 600; // state.rs: pub const MAX_REGS: usize = 600;

/// state.rs::save_regs -- the full-register-file snapshot the interpreter takes
/// on EVERY TryMeElse (multi-clause entry) and lowered_dispatch guard. Verbatim.
#[inline(never)]
fn save_regs(regs: &[Value]) -> Vec<(usize, Value)> {
    regs.iter()
        .enumerate()
        .take(200) // A registers (0..99) + X registers (100..199)
        .filter(|(_, v)| !matches!(v, Value::Uninit))
        .map(|(i, v)| (i, v.clone()))
        .collect()
}

/// state.rs::restore_regs -- verbatim (clears A/X, writes back saved).
#[inline(never)]
fn restore_regs(regs: &mut [Value], saved: &[(usize, Value)]) {
    for i in 0..200 {
        if i < regs.len() {
            regs[i] = Value::Uninit;
        }
    }
    for (i, v) in saved {
        if *i < regs.len() {
            regs[*i] = v.clone();
        }
    }
}

/// The minimal-locals snapshot (P2). For the matching_deps region the live set
/// that needs rollback is exactly the region's own working registers: the input
/// list cursor, Name, Ver, and the output-tail accumulator. Fixed size,
/// independent of how many registers the *caller* left live.
#[derive(Clone)]
struct MiniSnap {
    list: Value, // A1: remaining depends() list  (List -> O(1) Arc bump)
    name: Value, // A2: Name  (Atom -> String clone)
    ver: Value,  // A3: Ver   (compound -> functor String + O(1) Arc args)
    out: Value,  // A4: output tail accumulator
    trail_len: usize,
    heap_len: usize,
}

#[inline(never)]
fn mini_snapshot(regs: &[Value], trail_len: usize, heap_len: usize) -> MiniSnap {
    MiniSnap {
        list: regs[0].clone(),
        name: regs[1].clone(),
        ver: regs[2].clone(),
        out: regs[3].clone(),
        trail_len,
        heap_len,
    }
}

// ---------------------------------------------------------------------------
// Representative register file, built to mirror the resolver's live set at a
// matching_deps TryMeElse. Parameterised by the number of *extra* live X
// registers the surrounding computation has left behind (measured in Phase A).
// ---------------------------------------------------------------------------

fn rep_depends_list(n: usize, name: &str, ver: &Value, match_frac: f64) -> Value {
    // A catalog slice of depends(N,V,D,C) rows. `match_frac` of them match
    // (N==name, V==ver); the rest are other packages (the common case: most
    // rows in the depends_list are for other packages).
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let matches = (i as f64) < (n as f64) * match_frac;
        let (rn, rv) = if matches {
            (name.to_string(), ver.clone())
        } else {
            (format!("pkg_{}", i), Value::make_str("v", vec![
                Value::Integer((i % 9) as i64), Value::Integer(0), Value::Integer(0)]))
        };
        // D = a dependency name atom; C = a constraint (mostly gte(...)).
        let d = Value::Atom(format!("dep_{}", i % 32));
        let c = Value::make_str("gte", vec![Value::make_str("v", vec![
            Value::Integer(1), Value::Integer(0), Value::Integer(0)])]);
        rows.push(Value::make_str("depends", vec![Value::Atom(rn), rv, d, c]));
    }
    Value::list(rows)
}

fn build_regs(extra_live: usize, list: Value, name: Value, ver: Value) -> Vec<Value> {
    let mut regs = vec![Value::Uninit; MAX_REGS];
    regs[0] = list;                       // A1 = List
    regs[1] = name;                       // A2 = Name
    regs[2] = ver;                        // A3 = Ver
    regs[3] = Value::Unbound("_Out".into()); // A4 = Out
    // Extra live registers the deep resolver call chain leaves in X slots.
    // Mix tuned to the MEASURED save_regs profile over the real 2600-case B2
    // corpus (UW_SNAP_STATS): per call ~5 compound (Str: functor String + Arc
    // args), ~4.6 atom/unbound (String clone), ~8.2 list (Arc O(1) bump). So
    // per 18-slot cycle: 5 Str, 5 Atom, 8 List. This keeps the full-save_regs
    // cost faithful to reality rather than overstating it.
    for k in 0..extra_live {
        let slot = 100 + k; // X registers start at 100
        if slot >= 200 { break; }
        regs[slot] = match k % 18 {
            0..=4 => Value::make_str("v", vec![Value::Integer(k as i64), // Str
                    Value::Integer(0), Value::Integer(0)]),
            5..=9 => Value::Atom(format!("name_{}", k)),                 // String
            _ => Value::list((0..3).map(|j| Value::Integer(j)).collect()), // List: Arc
        };
    }
    regs
}

// ---------------------------------------------------------------------------
// The body work (held constant across both paths): dep_to_req + the guard.
// ---------------------------------------------------------------------------

/// dep_to_req/3, native. clause1: alternatives(_) -> req(alternatives(Alts),any)
/// (cut); clause2: req(D,C). Deterministic. This is the DIRECT NATIVE CALL (P1)
/// -- the fused caller invokes it as a Rust fn, never via vm.pc/vm.run().
#[inline]
fn dep_to_req(d: &Value, c: &Value) -> Value {
    match d {
        Value::Str(f, args) if f == "alternatives" && args.len() == 1 => {
            Value::make_str("req", vec![
                Value::make_str("alternatives", vec![args[0].clone()]),
                Value::Atom("any".into()),
            ])
        }
        _ => Value::make_str("req", vec![d.clone(), c.clone()]),
    }
}

/// The per-row guard + body, shared by both paths.
#[inline]
fn row_to_req(row: &Value, name: &Value, ver: &Value) -> Option<Value> {
    // row = depends(N, V, D, C)
    if let Value::Str(f, a) = row {
        if f == "depends" && a.len() == 4 {
            let n = &a[0]; let v = &a[1]; let d = &a[2]; let c = &a[3];
            if n == name && v == ver {
                return Some(dep_to_req(d, c));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// PATH 1 -- interpreter machinery model (faithful to state.rs step/backtrack).
//
// Per non-terminal matching_deps element the interpreter (2-clause pred) does:
//   * TryMeElse: push ChoicePoint { save_regs (FULL), stack Arc-clone, ... }
//   * try clause 1 (base []): get_nil A1 fails on a non-empty list
//   * backtrack: restore_regs (FULL) + trail unwind + stack Arc restore + pop
//   * clause 2 body: head unify + guard + [dep_to_req is ITSELF a Call: another
//     TryMeElse save_regs + clause-1(alternatives) fail + backtrack restore] +
//     cons Out + Execute self (re-dispatch through run()).
// The body work is done once either way; what PATH 1 adds over PATH 2 is the
// choice-point machinery (2 full save_regs + 2 full restore_regs + 2 stack
// Arc-clones per element) and the re-dispatch.
// ---------------------------------------------------------------------------

fn interp_machinery_per_list(
    regs: &[Value],
    scratch: &mut Vec<Value>, // the runtime's OWN register file (self.regs), reused
    stack: &Arc<Vec<u8>>,
    list: &Value,
) -> usize {
    // Returns a black-box accumulator so the optimiser cannot elide the work.
    // restore_regs mutates the machine's existing register file IN PLACE (no
    // per-call 600-slot allocation) -- so `scratch` is allocated once by the
    // caller and reused, exactly as self.regs is.
    let mut sink = 0usize;
    let mut cur = list.clone();
    let mut trail: Vec<usize> = Vec::new();
    loop {
        let (head, tail) = match &cur {
            Value::List(a) if !a.is_empty() => (a[0].clone(), Value::List(a.tail())),
            _ => break,
        };
        // --- matching_deps clause choice point (TryMeElse L_matching_deps_4_2) ---
        //     lib.rs line 3110. switch_on_term is a NoOp here (no first-arg
        //     indexing in this runtime), so this CP is pushed EVERY element.
        let cp_regs = save_regs(regs);          // FULL snapshot #1  (O2)
        let cp_stack = Arc::clone(stack);        // stack Arc clone
        let cp_trail = trail.len();
        // clause 1 (base []) head-match GetConstant [] A1 fails on non-empty:
        restore_regs(scratch, &cp_regs);        // FULL restore #1 IN PLACE (restore_regs 12%)
        let _ = Arc::clone(&cp_stack);
        trail.truncate(cp_trail);
        sink = sink.wrapping_add(cp_regs.len());

        // clause 2 body: Allocate + head unify + the ITE ( N==Name,V==Ver -> ...).
        // --- ITE guard choice point (TryMeElse L_ite_else_41, lib.rs 3130) ---
        //     pushed EVERY element (matching or not) for the (->;) soft cut.
        let cp_ite = save_regs(regs);           // FULL snapshot #2
        let _ = Arc::clone(stack);
        sink = sink.wrapping_add(cp_ite.len());
        if let Value::Str(_, a) = &head {
            let n = &a[0]; let v = &a[1]; let d = &a[2]; let c = &a[3];
            if n == &regs[1] && v == &regs[2] {
                // guard true -> CutTo Y11 prunes the ITE CP (pop, no restore).
                // --- dep_to_req clause choice point (Call, TryMeElse line 1765) ---
                let cp2 = save_regs(regs);       // FULL snapshot #3
                let cp2_stack = Arc::clone(stack);
                // clause 1 (alternatives(_)) fails for the common dep -> backtrack
                restore_regs(scratch, &cp2);     // FULL restore #3 IN PLACE
                let _ = Arc::clone(&cp2_stack);
                sink = sink.wrapping_add(cp2.len());
                let req = dep_to_req(d, c);        // body work (shared)
                sink = sink.wrapping_add(match &req { Value::Str(_, ra) => ra.len(), _ => 0 });
            } else {
                // guard fails -> backtrack into ITE else branch: FULL restore #2.
                restore_regs(scratch, &cp_ite);
            }
        }
        cur = tail;
    }
    sink
}

// ---------------------------------------------------------------------------
// PATH 2 -- fused native region (P1 + P2).
//   * ONE minimal-locals snapshot at region ENTRY (not per element, not full).
//   * loop over the list: guard + DIRECT native dep_to_req + cons. No choice
//     point, no re-dispatch, no per-element snapshot.
// ---------------------------------------------------------------------------

fn fused_region(regs: &[Value], list: &Value, name: &Value, ver: &Value) -> (Value, usize) {
    // P2: one cheap minimal snapshot for the whole region (rollback point).
    let _snap = mini_snapshot(regs, 0, 0);
    let mut out: Vec<Value> = Vec::new();
    let mut sink = 0usize;
    let mut cur = list.clone();
    loop {
        let (head, tail) = match &cur {
            Value::List(a) if !a.is_empty() => (a[0].clone(), Value::List(a.tail())),
            _ => break,
        };
        if let Some(req) = row_to_req(&head, name, ver) { // guard + DIRECT dep_to_req (P1)
            sink = sink.wrapping_add(match &req { Value::Str(_, ra) => ra.len(), _ => 0 });
            out.push(req);
        }
        cur = tail;
    }
    (Value::list(out), sink)
}

/// Reference walk (pure recursion, no machinery) -- the correctness oracle.
fn reference_matching_deps(list: &Value, name: &Value, ver: &Value) -> Value {
    let mut out = Vec::new();
    let mut cur = list.clone();
    while let Value::List(a) = &cur {
        if a.is_empty() { break; }
        if let Some(req) = row_to_req(&a[0], name, ver) { out.push(req); }
        cur = Value::List(a.tail());
    }
    Value::list(out)
}

// ---------------------------------------------------------------------------
// timing helpers
// ---------------------------------------------------------------------------

fn time_ns<F: FnMut() -> usize>(iters: u64, mut f: F) -> (f64, usize) {
    // warmup
    let mut sink = 0usize;
    for _ in 0..(iters / 10).max(1) { sink = sink.wrapping_add(f()); }
    let t = Instant::now();
    for _ in 0..iters { sink = sink.wrapping_add(f()); }
    let ns = t.elapsed().as_nanos() as f64 / iters as f64;
    (ns, sink)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // extra_live = live registers beyond matching_deps' own 4, measured in Phase A.
    let extra_live: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let list_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let match_frac: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.15);

    let name = Value::Atom("libfoo".into());
    let ver = Value::make_str("v", vec![Value::Integer(2), Value::Integer(1), Value::Integer(0)]);
    let list = rep_depends_list(list_len, "libfoo", &ver, match_frac);
    let regs = build_regs(extra_live, list.clone(), name.clone(), ver.clone());
    let stack: Arc<Vec<u8>> = Arc::new(vec![0u8; 64]); // stand-in for the Arc<Vec<StackEntry>>

    // ---- correctness: fused == reference, and dep_to_req both clauses ----
    let refv = reference_matching_deps(&list, &name, &ver);
    let (fusedv, _) = fused_region(&regs, &list, &name, &ver);
    assert!(refv == fusedv, "CORRECTNESS FAIL: fused != reference");
    // dep_to_req clause coverage
    let alts = Value::make_str("alternatives", vec![Value::list(vec![Value::Atom("a".into())])]);
    let r1 = dep_to_req(&alts, &Value::Atom("any".into()));
    assert!(matches!(&r1, Value::Str(f,a) if f=="req" && matches!(&a[1], Value::Atom(x) if x=="any")),
        "dep_to_req alternatives clause wrong");
    let r2 = dep_to_req(&Value::Atom("libbar".into()), &Value::make_str("gte", vec![Value::Integer(1)]));
    assert!(matches!(&r2, Value::Str(f,_) if f=="req"), "dep_to_req default clause wrong");
    let n_match = if let Value::List(a) = &refv { a.len() } else { 0 };
    eprintln!("correctness: OK (fused==reference; dep_to_req both clauses); matches={}/{}", n_match, list_len);

    let iters: u64 = 200_000;

    // ---- Bench A: snapshot cost (the O2 crux) ----
    let (full_ns, s1) = time_ns(iters, || {
        let v = save_regs(&regs); v.len().wrapping_add(v.iter().map(|(i,_)| *i).sum::<usize>())
    });
    let (mini_ns, s2) = time_ns(iters, || {
        let m = mini_snapshot(&regs, 0, 0);
        (matches!(m.list, Value::List(_)) as usize)
            .wrapping_add(matches!(m.name, Value::Atom(_)) as usize)
    });
    std::hint::black_box((s1, s2));

    // ---- Bench B: per-call machinery, interpreter vs fused, whole list ----
    let mut scratch = regs.clone(); // the machine's own register file, reused (self.regs)
    let (interp_ns, s3) = time_ns(iters, || interp_machinery_per_list(&regs, &mut scratch, &stack, &list));
    let (fused_ns, s4) = time_ns(iters, || { let (_, s) = fused_region(&regs, &list, &name, &ver); s });
    std::hint::black_box((s3, s4));

    // ---- Bench C: nondet resumable CP cycle, full vs minimal (extensibility) ----
    // One re-entry = push a choice point + later resume/restore. Interpreter:
    // full save_regs + full restore_regs. Minimal-resume: mini snapshot + mini
    // restore. This is the per-solution cost the plan says must beat 21%+12%.
    let mut cp_scratch = regs.clone(); // machine register file, reused
    let (cp_full_ns, s5) = time_ns(iters, || {
        let cp = save_regs(&regs);              // push
        restore_regs(&mut cp_scratch, &cp);     // resume/backtrack IN PLACE
        cp.len()
    });
    let (cp_mini_ns, s6) = time_ns(iters, || {
        let m = mini_snapshot(&regs, 0, 0);     // push (resume-arm + minimal locals)
        // resume: rebind only the region's own locals
        let mut r = [Value::Uninit, Value::Uninit, Value::Uninit, Value::Uninit];
        r[0] = m.list.clone(); r[1] = m.name.clone(); r[2] = m.ver.clone(); r[3] = m.out.clone();
        (matches!(r[0], Value::List(_)) as usize).wrapping_add(m.trail_len)
    });
    std::hint::black_box((s5, s6));

    // live-register count + clone-type mix actually captured by full save_regs:
    let full_snap = save_regs(&regs);
    let full_live = full_snap.len();
    let (mut sc, mut ac) = (0u32, 0u32);
    for (_, v) in &full_snap {
        match v {
            Value::Atom(_) | Value::Unbound(_) => sc += 1,
            Value::Str(_, _) => { sc += 1; ac += 1; }
            Value::List(_) => ac += 1,
            _ => {}
        }
    }
    eprintln!("microbench mix: live={} str_clones={} arc_clones={} (measured real: live~17.9 str~9.6 arc~13.2)", full_live, sc, ac);

    println!("=== Stage-2 spike: matching_deps/4 -> dep_to_req/3 ===");
    println!("params: extra_live={} list_len={} match_frac={} full_save_regs_live={}",
        extra_live, list_len, match_frac, full_live);
    println!();
    println!("[A] SNAPSHOT COST (the O2 crux), per snapshot:");
    println!("    full save_regs      : {:8.1} ns   ({} live regs cloned)", full_ns, full_live);
    println!("    minimal-locals snap : {:8.1} ns   (4 region locals)", mini_ns);
    println!("    ratio full/mini     : {:8.2}x", full_ns / mini_ns);
    println!();
    println!("[B] PER-CALL MACHINERY, whole depends list (len {}):", list_len);
    println!("    interpreter path    : {:8.1} ns   ({:6.1} ns/element)", interp_ns, interp_ns / list_len as f64);
    println!("    fused native region : {:8.1} ns   ({:6.1} ns/element)", fused_ns, fused_ns / list_len as f64);
    println!("    speedup             : {:8.2}x", interp_ns / fused_ns);
    println!();
    println!("[C] NONDET RESUMABLE CP CYCLE (push+resume), extensibility:");
    println!("    full  save/restore  : {:8.1} ns   (interpreter backtrack+restore_regs)", cp_full_ns);
    println!("    minimal save/resume : {:8.1} ns   (resume-arm + minimal locals)", cp_mini_ns);
    println!("    ratio full/mini     : {:8.2}x", cp_full_ns / cp_mini_ns);
}
