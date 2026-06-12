//! T7 (parallel / Tier-2) runtime substrate for the WAM Rust target.
//!
//! Parallelises the independent solution-generating branches of a *forkable
//! aggregate* (`findall/3`, `aggregate_all/4`) — the only context where fan-out
//! is sound, because branches are order-independent and their solution lists
//! merge into one accumulator.
//!
//! A WAM machine is mutable shared state, so it cannot be shared across threads;
//! each parallel worker runs on its own **clone** of the machine. That fork cost
//! is what sequential backtracking avoids, and it dominates for cheap branches —
//! so this collector is **gated**: it probes a few branches, estimates the
//! sequential vs parallel cost (including the measured thread-pool overhead), and
//! only fans out on a clear win. For cheap workloads it stays sequential and adds
//! essentially nothing; for expensive ones it captures ~core-count speedup.
//! Benchmark evidence + design: docs/reports/wam_rust_t7_parallel_perf.md.
//!
//! Generic over the machine `M` and the per-branch runner, so the interpreter's
//! `BeginAggregate` path supplies the real closures (clone the machine at the
//! aggregate's outer choice point; run branch `k` to its `EndAggregate`s,
//! returning that branch's solutions). Ordering matches generator order: branch
//! 0's solutions, then branch 1's, … (findall/aggregate order), preserved by
//! running contiguous branch ranges and concatenating chunks in order.

use std::time::Instant;
use std::thread;

/// Tuning for the adaptive gate. `pool_overhead_us` should be measured once at
/// startup via [`measure_pool_overhead`]; the rest have sane defaults.
#[derive(Clone, Copy, Debug)]
pub struct ParConfig {
    /// Worker threads to fan out across (typically `available_parallelism`).
    pub cores: usize,
    /// Branches run sequentially as a cost probe before deciding.
    pub probe: usize,
    /// Safety margin: only parallelise when `est_par * margin < est_seq`.
    pub margin: f64,
    /// Measured cost of spawning the worker pool, microseconds.
    pub pool_overhead_us: f64,
    /// Force a decision (testing / override). `None` = use the model.
    pub force: Option<bool>,
}

impl ParConfig {
    pub fn new(cores: usize, pool_overhead_us: f64) -> Self {
        ParConfig { cores: cores.max(1), probe: 8, margin: 1.25, pool_overhead_us, force: None }
    }
}

/// Outcome of a gated collection: the solutions in generator order, and whether
/// the parallel path was taken (useful for tests / telemetry).
pub struct Collected<V> {
    pub solutions: Vec<V>,
    pub went_parallel: bool,
}

/// Measure the worker-pool spawn overhead once, in microseconds — spawn `cores`
/// threads that do nothing and take the best of several runs.
pub fn measure_pool_overhead(cores: usize) -> f64 {
    let cores = cores.max(1);
    let mut best = f64::MAX;
    for _ in 0..16 {
        let t0 = Instant::now();
        thread::scope(|s| {
            for _ in 0..cores {
                s.spawn(|| std::hint::black_box(0u8));
            }
        });
        best = best.min(t0.elapsed().as_secs_f64() * 1e6);
    }
    best
}

/// Gated parallel collection over `n_branches` independent branches.
///
/// `run_branch(&mut machine, k)` must reset the (cloned) machine to the aggregate
/// fork point and run branch `k`, returning that branch's solutions in order.
/// The collector reuses one clone per worker (chunked), so `run_branch` is
/// responsible for resetting state between successive branches in the chunk —
/// exactly what restoring to the outer choice point does in the interpreter.
pub fn gated_collect<M, R, V>(
    base: &M,
    n_branches: usize,
    run_branch: R,
    cfg: &ParConfig,
) -> Collected<V>
where
    M: Clone + Send + Sync,
    R: Fn(&mut M, usize) -> Vec<V> + Sync,
    V: Send,
{
    if n_branches == 0 {
        return Collected { solutions: Vec::new(), went_parallel: false };
    }

    // --- probe: run the first `probe` branches sequentially, timing them ---
    let probe = cfg.probe.min(n_branches);
    let mut seq_machine = base.clone();
    let mut out: Vec<V> = Vec::new();
    let t0 = Instant::now();
    for k in 0..probe {
        out.extend(run_branch(&mut seq_machine, k));
    }
    let probe_us = t0.elapsed().as_secs_f64() * 1e6;
    let per_us = probe_us / (probe.max(1) as f64);
    let remaining = n_branches - probe;

    // --- decide ---
    let go_parallel = match cfg.force {
        Some(f) => f && remaining >= cfg.cores,
        None => {
            let est_seq = per_us * n_branches as f64;
            let est_par = est_seq / cfg.cores as f64 + cfg.pool_overhead_us;
            est_par * cfg.margin < est_seq && remaining >= cfg.cores
        }
    };

    if !go_parallel {
        // finish sequentially on the same machine (zero extra risk)
        for k in probe..n_branches {
            out.extend(run_branch(&mut seq_machine, k));
        }
        return Collected { solutions: out, went_parallel: false };
    }

    // --- fan out the remaining branches, chunked (one clone per worker) ---
    let cores = cfg.cores;
    let chunk = (remaining + cores - 1) / cores;
    let run_ref = &run_branch;
    let mut chunk_results: Vec<(usize, Vec<V>)> = thread::scope(|s| {
        let mut handles = Vec::new();
        for c in 0..cores {
            let lo = probe + c * chunk;
            let hi = (probe + (c + 1) * chunk).min(n_branches);
            if lo >= hi {
                continue;
            }
            let base_ref = base;
            handles.push(s.spawn(move || {
                let mut m = base_ref.clone(); // one clone per worker, reused
                let mut local = Vec::new();
                for k in lo..hi {
                    local.extend(run_ref(&mut m, k));
                }
                (lo, local)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // concatenate chunks in branch order to preserve generator order
    chunk_results.sort_by_key(|(lo, _)| *lo);
    for (_, mut v) in chunk_results {
        out.append(&mut v);
    }
    Collected { solutions: out, went_parallel: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A stand-in WAM machine: a small heap that branches read, plus a scratch reg.
    #[derive(Clone)]
    struct Machine {
        heap: Vec<i64>,
        reg: i64,
    }
    impl Machine {
        fn new() -> Self {
            Machine { heap: vec![1; 256], reg: 0 }
        }
    }

    // Branch k does `work` ops then yields a single solution. Resets reg first
    // (the "backtrack to the fork point" the interpreter would do).
    fn make_runner(work: u64) -> impl Fn(&mut Machine, usize) -> Vec<i64> + Sync {
        move |m: &mut Machine, k: usize| {
            m.reg = 0; // reset for this branch
            let mut a = (k as i64).wrapping_mul(2654435761);
            for i in 0..work {
                a = a
                    .wrapping_add(m.heap[(i as usize) % m.heap.len()])
                    .rotate_left(5)
                    ^ (i as i64);
                a = a.wrapping_mul(0x9E3779B1u32 as i64);
            }
            m.reg = a;
            vec![a]
        }
    }

    fn sequential(base: &Machine, n: usize, work: u64) -> Vec<i64> {
        let mut m = base.clone();
        let run = make_runner(work);
        let mut o = Vec::new();
        for k in 0..n {
            o.extend(run(&mut m, k));
        }
        o
    }

    fn cfg(force: Option<bool>) -> ParConfig {
        let cores = thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        let mut c = ParConfig::new(cores, measure_pool_overhead(cores));
        c.force = force;
        c
    }

    // Decisive correctness check: forced-parallel result equals the sequential
    // result *exactly* (same values, same generator order), for several sizes.
    #[test]
    fn parallel_equals_sequential_in_order() {
        let base = Machine::new();
        for &n in &[1usize, 7, 64, 257, 1000] {
            let want = sequential(&base, n, 500);
            let got = gated_collect(&base, n, make_runner(500), &cfg(Some(true)));
            assert_eq!(got.solutions, want, "n={} parallel != sequential", n);
        }
    }

    // Empty / single-branch edge cases.
    #[test]
    fn edge_cases() {
        let base = Machine::new();
        let got0 = gated_collect(&base, 0, make_runner(10), &cfg(Some(true)));
        assert!(got0.solutions.is_empty() && !got0.went_parallel);
        let got1 = gated_collect(&base, 1, make_runner(10), &cfg(Some(true)));
        assert_eq!(got1.solutions, sequential(&base, 1, 10)); // too few to fan out
        assert!(!got1.went_parallel);
    }

    // The gate stays sequential for cheap branches (no regression risk).
    #[test]
    fn gate_stays_sequential_when_cheap() {
        let base = Machine::new();
        let got = gated_collect(&base, 1000, make_runner(5), &cfg(None));
        assert!(!got.went_parallel, "cheap workload should not fan out");
        assert_eq!(got.solutions, sequential(&base, 1000, 5));
    }

    // The gate fans out for expensive branches, and is still correct.
    #[test]
    fn gate_goes_parallel_when_expensive() {
        let base = Machine::new();
        let n = 512;
        let got = gated_collect(&base, n, make_runner(200_000), &cfg(None));
        assert!(got.went_parallel, "expensive workload should fan out");
        assert_eq!(got.solutions, sequential(&base, n, 200_000));
    }

    // Sanity: on an expensive workload the parallel path is actually faster than
    // sequential (guards against the substrate being accidentally serial).
    #[test]
    fn expensive_parallel_is_faster() {
        let cores = thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        if cores < 2 {
            return; // nothing to prove on a uniprocessor
        }
        let base = Machine::new();
        let n = 256;
        let work = 200_000;
        let t0 = Instant::now();
        let s = sequential(&base, n, work);
        let seq = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let p = gated_collect(&base, n, make_runner(work), &cfg(Some(true)));
        let par = t1.elapsed().as_secs_f64();
        assert_eq!(p.solutions, s);
        assert!(par < seq, "parallel ({:.4}s) not faster than sequential ({:.4}s)", par, seq);
    }
}
