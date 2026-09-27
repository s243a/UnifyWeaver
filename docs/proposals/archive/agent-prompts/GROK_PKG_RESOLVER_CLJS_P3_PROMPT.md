<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — Clojure WAM P3 port: the CLJS lane catches up to the Debian resolver

> Branch from the coordinator tip as `grok/pkg-resolver-cljs-p3`. The CLJS
> lane is pinned to the PRE-P3 resolver: 11 of 51 corpus rows diverge (D57).
> This round follows the wam_go P3 playbook (D61) on the Clojure runtime:
> inventory builtins first, port runtime + shim, rebuild, measure the D59
> index honestly on the fleet's slowest leg.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`).
Branch from **`claude/peerhailer-exploratory-docs-aodas5`** as
`grok/pkg-resolver-cljs-p3`. SWI-Prolog (`swipl`, run under
`LC_ALL=C.UTF-8`); nbb + Node v18+ (`/opt/node22/bin/nbb` in the
coordinator env; use whatever nbb you have).

### Context to read first
- Ledger rows D57, D59, D61 in `docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`:
  what P3 added, the 11-row lane lag, the G1 catalog index
  (`index_threshold(64)`), and the D61 Go-port playbook you are mirroring —
  including its two find classes (unclassified builtins = silent fails;
  meta-call machinery clobbering registers / leaving stale CPs, which the
  corpus missed and the differential caught).
- `examples/pkg_resolver/README.md` — P3 semantics + the JSONL encodings
  (`{"deb":[Epoch,[[order,num],…],[…]]}`, alternatives groups, provides
  3/4-ary, `catalog/10`) the shim must speak.
- `examples/pkg_resolver/cljs/README.md` + `docs/WAM_CLOJURE_STATUS.md` —
  the D49 backtracking model (immutable-map choice points, `:cut-bar`) and
  the D54 indexing/dispatch machinery your changes must not regress.
- The JS builtin precedent: `builtin_maplist` in
  `templates/targets/javascript_wam/runtime.js.mustache` (meta-calls USER
  predicates); Go added `maplist/2,3,4` + `predsort/3` (D61 — predsort's
  comparator invocation MUST restore registers and truncate leftover CPs;
  Go got this wrong first and only the differential saw it).

### The work
1. **Inventory the gap.** Compile `examples/pkg_resolver/resolver.pl`
   through the Clojure WAM lane (`examples/pkg_resolver/cljs/build.sh`)
   and enumerate every builtin the generated code references that the
   Clojure runtime lacks or mishandles (`maplist/2,3,4`, `predsort/3`,
   `string_codes/2` empty edge are known from the JS/Go precedents; check
   `functor/3`/`arg/3`/`=../2`-class too). List them in the handoff.
2. **Port the runtime** (`templates/targets/clojure_wam/runtime.clj.mustache`
   + `wam_clojure_target.pl` where dispatch/classification lives).
   `maplist`/`predsort` meta-call user predicates; mind the immutable-state
   model — comparator/goal invocation must not leak register writes or
   choice points into the caller (the D61 predsort lesson, restated for a
   persistent-map state: thread the sub-call's state explicitly, take only
   the bindings you need). STRONGLY encouraged: a loud unknown-builtin
   diagnostic behind an env var (the `UW_WAM_WARN_UNKNOWN` convention from
   Rust/Go) — additive, off by default.
3. **Port the shim/edge** (`examples/pkg_resolver/cljs/resolver.cljs`,
   `pkg.cljs`'s catalog edge, `run_corpus.cljs`, `diff_runner_cljs.cljs`,
   `scale_to_catalog.mjs` if needed): deb versions both directions,
   alternatives, provides, `catalog/10`, and the extended blocked shapes
   (`providers([...])`, `alternatives([...])`).
4. **Rebuild and gate.**
   - `run_corpus_cljs.sh` → **51/51** vs SWI.
   - `run_differential_cljs.sh` → **2,600 cases, 0 divergences**.
   - CLI parity: `node --test examples/pkg_resolver/cljs/test_pkg_cli_cljs.mjs`
     → **153/153** (pkg.cljs rides the same resolver build).
   - `tests/core/test_clojurescript_wam_backtracking.pl` (D49 probes) and
     `tests/core/test_clojurescript_wam_indexing.pl` (D54 probes) — green.
   - All `tests/core/test_clojurescript_*.pl` + `tests/test_wam_clojure_*.pl`
     (the two lmdb.h generator tests and the `runtime_smoke` classpath
     non-termination are KNOWN pre-existing environment failures — verify
     pristine-identical if they fail for you, per the D54 note).
   - Shared-lane sanity: 4 `tests/test_wam_javascript_*.pl` suites,
     `CONFORMANCE_TARGETS=javascript`, cli_args cljs 17/17 + 25/25,
     pkg_resolver wamjs corpus 51/51.
5. **Benchmarks (B1–B3).** B1 corpus wall; B2 differential leg wall + SWI
   reference; B3 one `resolve_layered` on the 5k catalog (seed
   `0xc0ffee01`), load/resolve split, `bench_scale.sh` machinery. The
   interesting question: CLJS pays ~5µs/interpreted instruction, so the
   D59 index CUTS instructions ~9× but its BUILD is interpreted too — Go
   got only 1.04× because its fast baseline made the build cost dominant;
   CLJS's slow baseline (pre-P3: ~39–42s) may go either way. Report the
   honest number and, if you can, the step-census split (index build vs
   query) so the reading is mechanical, not vibes.

### Guardrails
- You may edit ONLY: `examples/pkg_resolver/cljs/` (all of it),
  `examples/pkg_resolver/run_differential_cljs.sh`,
  `src/unifyweaver/targets/wam_clojure_target.pl` +
  `wam_clojure_lowered_emitter.pl`, `templates/targets/clojure_wam/`,
  clojure test files (additive; existing probes stay green),
  `docs/WAM_CLOJURE_STATUS.md`.
- FROZEN: `examples/pkg_resolver/resolver.pl` and everything else under
  `examples/pkg_resolver/` (`cli/` — its generated corpus included —,
  `go/`, `rust/`, `debian/`, `store*`, `wamjs*`, probe files),
  `examples/cli_args/` (all — the cljs argparser is import-only), every
  other target, `wam_target.pl`, `wam_text_parser.pl`, shared harness.
  A bug you cannot fix inside owned files → minimal repro in the handoff.
- Wrong output is worse than refusal; the differential is the gate.

### Acceptance (must pass before handoff)
1. Corpus **51/51**; differential **2,600/0** (summaries + wall times).
2. CLI **153/153**; D49 + D54 probe suites green; clojure suites green
   (pre-existing env failures verified pristine-identical).
3. Shared-lane list green.
4. B1–B3 reported with the index reading.
5. Builtin inventory + whether the warn knob landed.

### Handoff format
Return: the builtin inventory (found → implemented → classified), the
runtime changes with the register/CP-hygiene story for meta-calls, shim
changes, corpus/differential/CLI numbers + timings, the B3 index verdict
with whatever census split you got, any bugs beyond builtins (symptom →
cause → fix → probe), status-doc updates, and residuals.

## ↑↑↑ Copy to here
