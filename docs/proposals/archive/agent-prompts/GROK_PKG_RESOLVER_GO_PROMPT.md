<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — uw-resolve on Go: compile the resolver through wam_go (+ cut-semantics audit)

> Branch from the coordinator tip as `grok/pkg-resolver-go`. Two concurrent
> rounds own `examples/pkg_resolver/cljs/` and `examples/pkg_resolver/rust/`
> plus the clojurescript and wam_rust target files — do not create or touch
> any of those. This round puts the backtracking resolver on **Go** (the
> portability target) and brings wam_go's cut semantics up to the §9 bar.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`).
Branch from **`claude/peerhailer-exploratory-docs-aodas5`** (the coordinator
tip — it already has everything you need) as `grok/pkg-resolver-go`.
SWI-Prolog (`swipl`); Go 1.22+; Node v18+ (for the differential generator).

### The mission
`examples/pkg_resolver/resolver.pl` is a frozen, SWI-oracled package resolver
(P0.5 semantics; 10 queries; genuine backtracking with cuts at API edges),
already gated on wam_javascript (corpus 38/38; seeded differential 2400/0).
Compile it through **wam_go** with the same bar. Expect the program to expose
real wam_go bugs — that is the value. Known hazards you will likely hit:

1. **Cut/choice-point barrier semantics — never audited on Go.** Read
   `docs/WAM_BACKEND_CONVENTIONS.md` §9 ("Cut is a barrier, never a stack
   wipe"): cut_barrier = WAM B0, rebased by BOTH Call AND Execute; isolation
   floors; opaque scopes (ITE condition, `\+`, `call/1`, aggregate inners).
   The JavaScript audit found 12 divergences of this class; this resolver is
   the program that exposed it (`!` wiping ALL choice points). The 35-probe
   corpus is `tests/test_wam_javascript_cut_semantics.pl`.
2. **X↔Y register aliasing (A2, verified):** `wam_go_target.pl:2103-2105`
   maps X_n→n+99 / Y_n→n+199, so **X101 ≡ Y1** on the flat `vm.Regs`; Y is
   saved only across the callee's own `Allocate` (`:2558-2598`). Wide facts /
   deep bodies can silently clobber. See `docs/WAM_FLEET_GAPS.md` (go row).
3. **A3 residual:** Go's `BuiltinExecute` handles Execute-of-builtin for
   *known* builtins (the fleet's reference fix), but a builtin missed by
   translation-time classification silently fails (`wam_go_target.pl:2661-2669`).

### Deliverables
1. **`examples/pkg_resolver/go/`** (new): `build.sh` (swipl → wam_go project,
   regenerable), the generated Go project (committed), a thin shim (term↔JSON
   IO ONLY — catalogs/requests/env in as data, selections/explanations out as
   JSON; no resolver logic in Go), `run_corpus_go.sh` driving all 38 corpus
   scenarios through the compiled binary vs SWI (explanation terms included),
   and `run_differential_go.sh`: the SAME seeded generator
   (`gen_catalogs.mjs`, same seed), **≥2400 cases, 0 divergences** on all 10
   queries (SWI leg vs Go leg).
2. **wam_go fixes** the program forces, each with a minimal probe test:
   cut semantics to the §9 bar; anything the aliasing or A3 residual breaks.
3. **`tests/test_wam_go_cut_semantics.pl`** (new): port the 35-probe
   cut-semantics corpus from the JS suite to drive the wam_go build —
   SWI-oracled, all passing. Probes whose shapes wam_go cannot compile yet:
   refuse-to-compile-loudly, documented — never wrong output.
4. `examples/pkg_resolver/go/README.md`: build/run, fixes, benchmarks,
   residuals. Update `docs/WAM_GO_STATUS.md` honestly (A2/A3 rows, cut audit).

### Benchmark protocol (the coordinator assembles a cross-target table)
- B1: wall time of one full 38-scenario corpus run (Go binary, single
  process; note `go build` time separately).
- B2: wall time of the Go differential leg + case count (SWI leg on the same
  machine for reference).
- B3: one `resolve_layered` on the 5k-package scale catalog (generate via
  `examples/pkg_resolver/store/gen_scale_catalog.mjs`, seed `0xc0ffee01`;
  convert to your input form): load and resolve times separately; SWI
  reference numbers come from `run_scale_demo.sh`.

### Guardrails
- You may edit ONLY: `examples/pkg_resolver/go/` (new),
  `src/unifyweaver/targets/wam_go_target.pl`, `wam_go_lowered_emitter.pl`,
  the Go WAM templates (under `templates/`), `tests/test_wam_go_*.pl`
  (additive; every existing wam_go test stays green), `docs/WAM_GO_STATUS.md`.
- FROZEN: everything else in `examples/pkg_resolver/` (`resolver.pl`,
  `test_resolver.pl`, `cli/`, `wamjs*`, `store*`, `cljs/`, `rust/` — the last
  two belong to concurrent rounds even if absent on your branch: do not
  create them), `examples/cli_args/` (all), ALL javascript/wam_javascript and
  wam_rust and clojurescript files, `wam_target.pl`, `wam_text_parser.pl`,
  shared harness/registry/matrix/glue.
- Re-run and keep green before handoff: the full pre-existing wam_go suite
  (all `tests/test_wam_go_*.pl`), the three `tests/test_wam_javascript_*.pl`
  suites + cut-semantics + `CONFORMANCE_TARGETS=javascript` (proving you
  broke nothing shared), cli_args 17/17 + differential 5067/0/0, and the
  pkg_resolver SWI corpus 38/38 + wamjs 39/39 + term differential 2400/0.
- Wrong output is worse than refusal; the differential is the semantics gate.

### Acceptance (must pass before handoff)
1. `run_corpus_go.sh` → 38/38 matching SWI.
2. `run_differential_go.sh` → ≥2400 cases, 0 divergences (paste summary +
   both legs' wall times).
3. `tests/test_wam_go_cut_semantics.pl` → all probes pass (state count and
   any refused-loudly shapes).
4. Full pre-existing gate list re-run green.
5. B1–B3 numbers reported.

### Handoff format
Return: every wam_go fix (symptom → cause → fix → probe), cut-probe count,
corpus + differential numbers with timings, B1–B3, the honest status-doc
updates you made, and residuals.

## ↑↑↑ Copy to here
