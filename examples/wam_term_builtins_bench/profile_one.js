#!/usr/bin/env node
// Profile-focused runner: hammer a single bench export for a long
// time so CPU sampling has enough resolution.
//
// Usage: node profile_one.js <wasm-file> <bench_name> [iterations]

const fs = require('fs');
const [,, wasmPath, benchName, iterStr] = process.argv;
const iters = parseInt(iterStr || '500000', 10);

if (!wasmPath || !benchName) {
  console.error('Usage: node profile_one.js <wasm-file> <bench_name> [iterations]');
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
  const fn = instance.exports[benchName];
  if (typeof fn !== 'function') {
    console.error(`Export ${benchName} not found`);
    process.exit(1);
  }

  // Warmup.
  for (let i = 0; i < 10000; i++) fn();

  // Measure.
  const t0 = process.hrtime.bigint();
  let ok = 0;
  for (let i = 0; i < iters; i++) ok += fn();
  const t1 = process.hrtime.bigint();

  const elapsedNs = Number(t1 - t0);
  console.log(`${benchName}: ${iters} iters in ${(elapsedNs/1e6).toFixed(1)}ms, ${(elapsedNs/iters).toFixed(0)}ns/call, ok=${ok}`);
})();
