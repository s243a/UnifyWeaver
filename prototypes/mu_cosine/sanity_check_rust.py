#!/usr/bin/env python3
"""Sanity-check that a dense μ map feeds `gated_ic` / `lin_from_ic` in the **real** Rust core.

Not a reimplementation: this renders `templates/targets/rust_wam/boundary_cache.rs.mustache` (the only
mustache tag is `{{date}}`) into a throwaway crate and calls the actual `condense_scc`,
`lift_mu_to_components`, `gated_ic`, and `lin_from_ic` on the dense μ — mirroring the loader in the
`wikipedia_gated_similarity_tracks_physics_relatedness` Rust test (name interning → ids, μ resolved by
name, unresolved → silently absent ⇒ μ=0).

It checks the integration *contract*, not the physics ordering (a raw untrained MiniLM cosine has no
domain supervision, so EM≫Thermo need not hold — that is the trained encoder's job). Specifically:
  * COVERAGE: how many emitted μ names resolve against the graph (the guard the dense map exists to fix).
  * gated_ic produces finite IC for in-domain nodes (μ ≥ threshold), +∞ for out-of-domain.
  * lin_from_ic returns a finite similarity in [0,1] for a representative in-domain pair.

    python3 sanity_check_rust.py --fuzzy dense_mu_physics.tsv

Requires `cargo`. Exits non-zero on contract violation.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
TEMPLATE = os.path.join(REPO, "templates", "targets", "rust_wam", "boundary_cache.rs.mustache")
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")

CARGO_TOML = """[package]
name = "mu-sanity"
version = "0.0.0"
edition = "2021"

[[bin]]
name = "mu_sanity"
path = "src/main.rs"

[profile.release]
opt-level = 2
"""

# Mirrors the loader in `wikipedia_gated_similarity_tracks_physics_relatedness`, then reports the
# integration contract instead of asserting the (training-dependent) physics ordering.
MAIN_RS = r'''
mod boundary_cache;
use boundary_cache::*;
use std::collections::HashMap;

fn main() {
    let tsv = std::env::var("UW_CATEGORY_TSV").expect("UW_CATEGORY_TSV");
    let fz = std::env::var("UW_FUZZY_NODES").expect("UW_FUZZY_NODES");

    // --- graph: intern names -> ids, build child-component parents (verbatim from the Rust test) ---
    let text = std::fs::read_to_string(&tsv).expect("read UW_CATEGORY_TSV");
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut parents: HashMap<u32, Vec<u32>> = HashMap::new();
    for (ln, line) in text.lines().enumerate() {
        if ln == 0 && line.starts_with("child") { continue; }
        let mut it = line.split('\t');
        let (c, p) = match (it.next(), it.next()) { (Some(c), Some(p)) => (c, p), _ => continue };
        let mut intern = |s: &str| *ids.entry(s.to_string()).or_insert_with(|| { names.push(s.to_string()); (names.len() - 1) as u32 });
        let ci = intern(c); let pi = intern(p);
        parents.entry(ci).or_default().push(pi);
    }

    // --- dense mu: resolve by name; count coverage (a name absent from the graph silently => mu=0) ---
    let mut mu: HashMap<u32, f64> = HashMap::new();
    let (mut emitted, mut resolved) = (0usize, 0usize);
    for l in std::fs::read_to_string(&fz).expect("read fuzzy").lines() {
        if l.trim_start().starts_with('#') || l.trim().is_empty() { continue; }
        let mut it = l.split('\t');
        if let (Some(nm), Some(m)) = (it.next(), it.next().and_then(|s| s.trim().parse::<f64>().ok())) {
            emitted += 1;
            if let Some(&i) = ids.get(nm.trim()) { mu.insert(i, m); resolved += 1; }
        }
    }
    let cov = if emitted > 0 { 100.0 * resolved as f64 / emitted as f64 } else { 0.0 };
    println!("coverage: {}/{} emitted names resolved against the graph ({:.2}%)", resolved, emitted, cov);
    println!("graph nodes: {}; nodes with mu: {}", names.len(), mu.len());

    // --- feed gated_ic / lin_from_ic exactly as the core does ---
    let cond = condense_scc(&parents);
    let cmu = lift_mu_to_components(&mu, &cond);
    let total_mu: f64 = cmu.values().sum();
    assert!(total_mu > 0.0, "total mu must be > 0 (dense map produced no mass)");
    let threshold = 0.3_f64;
    let ic = gated_ic(&cond.parents, &cmu, threshold, total_mu);

    let finite = ic.values().filter(|v| v.is_finite()).count();
    let infinite = ic.values().filter(|v| v.is_infinite()).count();
    println!("gated_ic (threshold {threshold}): {} components, {finite} finite IC, {infinite} +inf (out-of-domain)", ic.len());
    assert!(finite > 0, "gated_ic produced no finite IC -- dense mu did not feed the cone machinery");

    // lin_from_ic over a representative in-domain pair (must resolve and be finite in [0,1]).
    let comp = |nm: &str| -> Option<u32> { ids.get(nm).map(|i| cond.comp[i]) };
    let lin = |a: &str, b: &str| -> Option<f64> { lin_from_ic(comp(a)?, comp(b)?, &cond.parents, &ic) };
    let mut shown = 0;
    for (a, b) in [("Electromagnetism", "Optics"), ("Thermodynamics", "Optics"), ("Physics", "Energy")] {
        match lin(a, b) {
            Some(v) => {
                println!("lin_from_ic({a}, {b}) = {v:.4}");
                assert!(v.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&v), "lin out of [0,1]: {v}");
                shown += 1;
            }
            None => println!("lin_from_ic({a}, {b}) = None (no shared finite-IC ancestor or name absent)"),
        }
    }
    assert!(shown > 0, "no representative pair produced a lin_from_ic value");
    println!("SANITY OK: dense mu feeds gated_ic / lin_from_ic in the Rust core.");
}
'''


def render_module(dst):
    with open(TEMPLATE, encoding="utf-8") as f:
        src = f.read()
    # The only real mustache tag is {{date}}; the other `{{...}}` are Rust format-string escapes
    # inside string literals (e.g. `{{0, A}}` -> `{0, A}`), which must be left untouched.
    src = src.replace("{{date}}", "rendered by sanity_check_rust.py")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=GRAPH)
    ap.add_argument("--fuzzy", default=os.path.join(ROOT, "dense_mu_physics.tsv"),
                    help="dense μ map (name<TAB>μ) to sanity-check")
    ap.add_argument("--keep", action="store_true", help="keep the temp crate (debug)")
    args = ap.parse_args()

    if shutil.which("cargo") is None:
        sys.exit("error: cargo not found; needed to compile the real Rust core")
    if not os.path.exists(args.fuzzy):
        sys.exit(f"error: dense μ map not found: {args.fuzzy} (run gen_dense_mu.py first)")

    crate = tempfile.mkdtemp(prefix="mu_sanity_")
    try:
        src = os.path.join(crate, "src")
        os.makedirs(src)
        with open(os.path.join(crate, "Cargo.toml"), "w") as f:
            f.write(CARGO_TOML)
        with open(os.path.join(src, "main.rs"), "w") as f:
            f.write(MAIN_RS.strip() + "\n")
        render_module(os.path.join(src, "boundary_cache.rs"))

        env = dict(os.environ, UW_CATEGORY_TSV=os.path.abspath(args.graph),
                   UW_FUZZY_NODES=os.path.abspath(args.fuzzy))
        print(f"building + running sanity crate in {crate} ...", file=sys.stderr)
        r = subprocess.run(["cargo", "run", "--release", "--quiet"], cwd=crate, env=env)
        sys.exit(r.returncode)
    finally:
        if not args.keep:
            shutil.rmtree(crate, ignore_errors=True)


if __name__ == "__main__":
    main()
