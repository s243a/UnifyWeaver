<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->
# PAR-1 Integration Patch — JS pattern cross-target parity harness

PAR-1 adds a self-contained harness. It needs **no** changes to run
(`swipl -q -g run_tests -t halt tests/test_js_pattern_cross_target_conformance.pl`
is green as-is). Everything below is *optional wiring* kept out of the tree so
it does not collide with the parallel `annotated_js` / `vanilla_js` work.

## Files added by PAR-1 (no patch needed)

- `tests/js_pattern_conformance_fixtures.pl`
- `tests/test_js_pattern_cross_target_conformance.pl`
- `docs/JS_PATTERN_CONFORMANCE.md`

None of the constrained files were touched: `core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, `glue/js_glue.pl`.

## 1. Optional CI job (`.github/workflows/test.yml`)

Mirror the existing `wam_conformance_smoke` job. Not applied here to avoid a
merge conflict with the parallel target work; add when ready:

```yaml
  js_pattern_conformance:
    name: JS Pattern Conformance
    if: github.event_name == 'pull_request' || github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install SWI-Prolog
        run: bash scripts/ci/install_swi_prolog.sh

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install TypeScript + nbb
        run: npm i -g typescript nbb

      - name: Run JS pattern conformance
        run: |
          swipl -q -g run_tests -t halt tests/test_js_pattern_cross_target_conformance.pl
```

The harness skips any arm whose runtime is absent, so the job is safe even if
`nbb` install is dropped; with `typescript` + `node` present the TypeScript
arm is exercised.

## 2. Contract for `annotated_js` / `vanilla_js` (the parallel targets)

The harness auto-discovers these arms once their targets exist. To be picked
up, a target must:

1. Be registered in `target_registry.pl` with a `target_module/2` clause
   (e.g. `target_module(annotated_js, annotated_js_target).`) — this is the
   PAR-* card that owns those targets, not PAR-1.
2. Expose `compile_predicate/3` from its module. The harness calls:

   ```prolog
   Module:compile_predicate(Name/Arity,
                            [module_name(jsct), function_name(FnName)],
                            Code).
   ```

   - `Name/Arity` is the fixture's oracle predicate (e.g. `cfib/2`).
   - `function_name(FnName)` is the JS function the harness's appended driver
     will call (`FnName` per `js_conformance_program/6`, e.g. `fib`). If the
     target names the function differently, either honour `function_name/1`
     or update the fixture's `FnName`.
   - The generated `Code` must define that function at module top level; the
     harness appends `console.log(FnName(<parsed argv[2]>));` and runs it with
     `node prog.js <arg>`.

3. Return, for the **numeric** family, a function whose value matches the
   Prolog oracle; for the **structural** family, print `true` / `false` for a
   ground query (the harness compares against the boolean oracle).

Until then both arms skip cleanly via `ja_target_present/1`.

## 3. When the ClojureScript recursion path lands

`ja_xfail(clojurescript, <numeric program>)` entries in the harness mark the
numeric programs as tolerated-divergent because the CLJS pattern target does
not yet emit clean recursive numeric code (recursion currently reaches the
shared TypeScript `compile_linear_pattern` clause, not a ClojureScript one).
Once a ClojureScript recursion path exists and `nbb` runs the output, remove
those `ja_xfail/2` clauses so the arm becomes a real check.
