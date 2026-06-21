// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// boundary_splice_complexity_bench.rs — P3 headroom measurement for the boundary
// distribution cache (docs/design/WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md).
//
// The point it makes: the boundary distribution is PRIMARILY a complexity
// reduction, not just a cache. The effective-distance kernel sums over ALL paths
// seed->root (exponential in a graph with diamonds, capped at budget). The edge
// cache removes LMDB-seek cost but CANNOT avoid that enumeration. A boundary
// distribution caches the path-LENGTH histogram of the shared upper cone, which
// represents the exponentially-many paths compactly, so a query that reaches the
// boundary splices (O(budget)) instead of re-enumerating. The precompute-once /
// reuse-across-seeds is the secondary caching/amortization layer.
//
// Method A: full path enumeration seed->root (the kernel's behaviour; parent
//           lookups are HashMap hits = the warm edge cache).
// Method B: walk seed->boundary, then splice the cached suffix histogram.
//
// Graph: a DENSE core near the root (diamonds -> exponentially many paths to
// root) behind a thin BOUNDARY cut, with a SPARSE periphery below where seeds
// attach only to boundary nodes — so every seed->root path crosses the cut.
//
// Run (no deps, std only):  rustc -O boundary_splice_complexity_bench.rs && ./boundary_splice_complexity_bench
use std::collections::HashMap;
use std::time::Instant;

type Hist = Vec<u64>;

fn add_shifted(out: &mut Hist, src: &Hist, max: usize) {
    for (l, &c) in src.iter().enumerate() {
        let l2 = l + 1;
        if l2 <= max {
            if out.len() <= l2 { out.resize(l2 + 1, 0); }
            out[l2] += c;
        }
    }
}

// Suffix histogram H_node->root via the recurrence (memoised, cycle-safe).
fn suffix_hist(node: u32, root: u32, par: &HashMap<u32, Vec<u32>>, max: usize,
               memo: &mut HashMap<u32, Hist>, st: &mut Vec<u32>) -> Hist {
    if node == root { return vec![1]; }
    if let Some(h) = memo.get(&node) { return h.clone(); }
    if st.contains(&node) { return vec![]; }
    st.push(node);
    let mut out = vec![];
    if let Some(ps) = par.get(&node) {
        for &p in ps { let ph = suffix_hist(p, root, par, max, memo, st); add_shifted(&mut out, &ph, max); }
    }
    st.pop();
    memo.insert(node, out.clone());
    out
}

fn wpow(h: &Hist, n: f64) -> f64 {
    h.iter().enumerate().filter(|(l, _)| *l > 0).map(|(l, &c)| c as f64 * (l as f64).powf(-n)).sum()
}

// Method A: full path enumeration (what the kernel does; the edge cache cannot avoid this).
fn full_enum_wpow(seed: u32, root: u32, par: &HashMap<u32, Vec<u32>>, budget: usize, n: f64) -> f64 {
    let mut h: Hist = vec![];
    let mut stack = vec![(seed, 0usize, vec![seed])];
    while let Some((nd, len, vis)) = stack.pop() {
        if nd == root { if h.len() <= len { h.resize(len + 1, 0); } h[len] += 1; continue; }
        if len >= budget { continue; }
        if let Some(ps) = par.get(&nd) {
            for &p in ps { if vis.contains(&p) { continue; } let mut v = vis.clone(); v.push(p); stack.push((p, len + 1, v)); }
        }
    }
    wpow(&h, n)
}

// Method B: walk seed->boundary, splice the cached suffix histogram, stop.
fn splice_wpow(seed: u32, root: u32, par: &HashMap<u32, Vec<u32>>, bset: &HashMap<u32, Hist>, budget: usize, n: f64) -> f64 {
    let mut h: Hist = vec![];
    let mut stack = vec![(seed, 0usize, vec![seed])];
    while let Some((nd, len, vis)) = stack.pop() {
        if nd == root { if h.len() <= len { h.resize(len + 1, 0); } h[len] += 1; continue; }
        if let Some(sh) = bset.get(&nd) {
            for (b, &c) in sh.iter().enumerate() {
                let l = len + b;
                if l <= budget && c > 0 { if h.len() <= l { h.resize(l + 1, 0); } h[l] += c; }
            }
            continue; // stop at the boundary
        }
        if len >= budget { continue; }
        if let Some(ps) = par.get(&nd) {
            for &p in ps { if vis.contains(&p) { continue; } let mut v = vis.clone(); v.push(p); stack.push((p, len + 1, v)); }
        }
    }
    wpow(&h, n)
}

fn build(core: u32, periph: u32, core_parents: usize, seed: u64) -> (HashMap<u32, Vec<u32>>, Vec<u32>, Vec<u32>) {
    let mut rng = seed;
    let mut rnd = |m: u32| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as u32) % m };
    let mut par: HashMap<u32, Vec<u32>> = HashMap::new();
    // DENSE core 1..core toward root (low index = near root): many paths to root.
    for i in 1..core {
        let mut ps = vec![]; let k = (core_parents as u32).min(i);
        for _ in 0..k.max(1) { ps.push(rnd(i)); }
        ps.sort(); ps.dedup(); par.insert(i, ps);
    }
    // BOUNDARY band = top W core nodes (entry points from the periphery).
    let w = 40u32.min(core - 1);
    let boundary: Vec<u32> = ((core - w)..core).collect();
    // SPARSE periphery attaches ONLY to boundary nodes, so the boundary is a true cut.
    for i in core..core + periph {
        let mut ps = vec![];
        for _ in 0..2 { ps.push(core - w + rnd(w)); }
        ps.sort(); ps.dedup(); par.insert(i, ps);
    }
    let seeds: Vec<u32> = (core..core + periph).collect();
    (par, boundary, seeds)
}

fn main() {
    let budget = 10usize;
    let n = 2.0f64;
    println!("{:<26} {:>11} {:>12} {:>9} {:>8}", "config", "A_full_ms", "B_splice_ms", "speedup", "equal?");
    for (core, periph, cp) in [(120u32, 500u32, 3usize), (160, 500, 3), (200, 500, 3), (240, 500, 3)] {
        let (par, bvec, seeds) = build(core, periph, cp, 42);
        // precompute the boundary suffix histograms ONCE (the caching/amortization layer)
        let mut memo = HashMap::new();
        let bset: HashMap<u32, Hist> = bvec.iter().map(|&b| {
            let mut st = vec![]; (b, suffix_hist(b, 0, &par, budget, &mut memo, &mut st))
        }).collect();
        let t = Instant::now(); let mut sa = 0.0; for &s in &seeds { sa += full_enum_wpow(s, 0, &par, budget, n); }
        let ta = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now(); let mut sb = 0.0; for &s in &seeds { sb += splice_wpow(s, 0, &par, &bset, budget, n); }
        let tb = t.elapsed().as_secs_f64() * 1e3;
        let eq = (sa - sb).abs() < 1e-6 * sa.abs().max(1.0);
        println!("core={:<4} cp={:<2} seeds={:<5} {:>11.1} {:>12.2} {:>8.1}x {:>8}",
                 core, cp, seeds.len(), ta, tb, ta / tb, if eq { "yes" } else { "NO!" });
    }
}
