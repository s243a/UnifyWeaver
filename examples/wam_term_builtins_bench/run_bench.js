#!/usr/bin/env node
// WAM-WAT benchmark runner. Loads a .wasm module and runs each
// exported benchmark function N times, reporting ns/call for each.
//
// Usage: node run_bench.js <wasm-file> [iterations]
//
// Exports matching /^bench_/ are auto-discovered and benchmarked.
// Each export takes no args and returns i32 (1=success, 0=fail).
// The harness does 100 warmup calls, then measures N iterations.

const fs = require('fs');
const [,, wasmPath, iterStr] = process.argv;
const iters = parseInt(iterStr || '10000', 10);

if (!wasmPath) {
  console.error('Usage: node run_bench.js <wasm-file> [iterations]');
  process.exit(1);
}

const bytes = fs.readFileSync(wasmPath);
const imports = {
  env: {
    print_i64: () => {},
    print_char: () => {},
    print_newline: () => {},
  }
};

(async () => {
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  const exports = instance.exports;

  // Auto-discover bench_ exports
  const benchmarks = Object.keys(exports)
    .filter(k => k.startsWith('bench_') && typeof exports[k] === 'function')
    .sort();

  if (benchmarks.length === 0) {
    console.error('No bench_* exports found');
    process.exit(1);
  }

  console.log(`WAM-WAT Benchmark Suite (${iters} iterations per workload)`);
  console.log('='.repeat(70));
  console.log(`${'Workload'.padEnd(30)} ${'ns/call'.padStart(10)} ${'calls/s'.padStart(12)} ${'ok'.padStart(6)}`);
  console.log('-'.repeat(70));

  const results = [];

  for (const name of benchmarks) {
    const fn = exports[name];

    // Warmup
    for (let i = 0; i < 100; i++) fn();

    // Measure
    const t0 = process.hrtime.bigint();
    let ok = 0;
    for (let i = 0; i < iters; i++) {
      if (fn() === 1) ok++;
    }
    const t1 = process.hrtime.bigint();

    const totalNs = Number(t1 - t0);
    const perCallNs = totalNs / iters;
    const callsPerSec = 1e9 / perCallNs;
    const allOk = ok === iters;

    // Export name uses _ where Prolog used _ (bench_sum_big_0 → bench_sum_big)
    const label = name.replace(/_0$/, '');

    console.log(
      `${label.padEnd(30)} ${perCallNs.toFixed(0).padStart(10)} ${callsPerSec.toFixed(0).padStart(12)} ${(allOk ? 'OK' : `${ok}/${iters}`).padStart(6)}`
    );

    results.push({ name: label, ns: perCallNs, ok: allOk });
  }

  console.log('-'.repeat(70));

  // Summary
  const failed = results.filter(r => !r.ok);
  if (failed.length > 0) {
    console.log(`\nWARNING: ${failed.length} workload(s) returned fail:`);
    for (const f of failed) console.log(`  - ${f.name}`);
  }

  // Output JSON for programmatic consumption
  const jsonPath = wasmPath.replace(/\.wasm$/, '_results.json');
  fs.writeFileSync(jsonPath, JSON.stringify({
    engine: 'wam-wat-v8',
    iterations: iters,
    results: results.map(r => ({ workload: r.name, ns_per_call: r.ns, ok: r.ok }))
  }, null, 2));
  console.log(`\nResults written to ${jsonPath}`);
})();
