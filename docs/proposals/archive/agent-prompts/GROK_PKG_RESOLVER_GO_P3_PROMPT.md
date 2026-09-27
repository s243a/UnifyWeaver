<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — wam_go P3 port: the Go lane catches up to the Debian resolver

> Branch from the coordinator tip as `grok/pkg-resolver-go-p3`. The Go lane
> is pinned to the PRE-P3 resolver: 11 of 51 corpus rows diverge (D57), and
> rebuilding against the current resolver WITHOUT porting is worse (30/51 —
> missing builtins break helper paths shared with old scenarios). This round
> ports the runtime + shim, rebuilds, and measures the D59 index on Go.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`).
Branch from **`claude/peerhailer-exploratory-docs-aodas5`** as
`grok/pkg-resolver-go-p3`. SWI-Prolog (`swipl`, run under
`LC_ALL=C.UTF-8`); Go 1.22+; Node v18+ (differential generator).

### Context to read first
- Ledger rows D57, D59, D60 in `docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`:
  what P3 added (deb versions / Provides / alternatives), the exact 11
  diverging rows and why a naive rebuild is worse, and the new G1 catalog
  index (`index_threshold(64)`) your rebuild will pick up for free.
- `examples/pkg_resolver/README.md` — P3 semantics + the JSONL encodings
  (`{"deb":[Epoch,[[order,num],…],[…]]}`, `{"alternatives":[…]}`, provides
  rows) your shim must speak.
- The D57 JS precedent: the compiler emits `BuiltinCall maplist/N` for the
  P3 helper paths; the JS runtime gained `maplist/2,3,4` (meta-calling USER
  predicates — see `builtin_maplist` in
  `templates/targets/javascript_wam/runtime.js.mustache`) and
  `string_codes/2` with the empty-string ↔ `[]` edge, pinned by
  `maplist_string_codes_runtime` in `tests/test_wam_javascript_builtins.pl`.
- D60's cautionary tale: a guard using `functor/3` + `memberchk/2` silently
  failed EVERY wamjs query — the A3 silent-unknown-builtin class. Expect Go
  to have its own gaps; find them loudly, not by divergence-hunting.

### The work
1. **Inventory the gap.** Compile `examples/pkg_resolver/resolver.pl`
   through wam_go and enumerate every builtin the generated WAM references
   that the Go runtime lacks or mishandles (`maplist/2,3,4` and
   `string_codes/2` are known from the JS precedent; there may be more —
   `functor/3`, `memberchk/2` class included if the resolver now uses them).
   List them in the handoff.
2. **Port the runtime.** Implement the missing builtins in the Go WAM
   runtime (templates + `wam_go_target.pl` classification so they are
   KNOWN builtins — remember Go's A3 residual: an unclassified builtin
   silently fails). `maplist` must meta-call user predicates like the JS
   one. STRONGLY encouraged while you are in there: a loud
   unknown-builtin diagnostic gated behind an env var (mirror Rust's
   `UW_WAM_WARN_UNKNOWN=1`), so the next gap announces itself — additive,
   off by default.
3. **Port the shim.** `examples/pkg_resolver/go/shim.go` (+ helpers) must
   round-trip the P3 JSON: deb versions (both directions — catalogs in,
   selections/explanations out, including `blocked(..., providers([...]))`
   and `blocked(alternatives([...]))` shapes), alternatives groups in
   depends rows, provides rows (3- and 4-ary), `catalog/10`.
4. **Rebuild and gate.** `bash examples/pkg_resolver/go/build.sh` against
   the CURRENT resolver (it now carries the D59 index — you get it for
   free), then:
   - `run_corpus_go.sh` → **51/51** vs SWI (the 11 P3 rows included).
   - `run_differential_go.sh` → **≥2,600 cases, 0 divergences** (the
     generator now emits deb/provides/alternatives; same seeds as the
     wamjs leg).
   - Cut semantics 35/35 (`tests/test_wam_go_cut_semantics.pl`), the five
     D51 probe suites, `test_wam_go_frameless_ite_level.pl`, and the FULL
     pre-existing `tests/test_wam_go_*.pl` sweep — the D53 barrier and
     D51 fixes must survive your builtin additions.
   - Shared-lane sanity: the four `tests/test_wam_javascript_*.pl` suites,
     `CONFORMANCE_TARGETS=javascript`, cli_args 17/17 + 5067/0/0, and the
     pkg_resolver wamjs corpus 51/51 — proving you broke nothing shared.
5. **Benchmarks (B1–B3, the D51 protocol).** B1 corpus wall; B2
   differential leg wall + SWI reference; B3 one `resolve_layered` on the
   5k catalog (seed `0xc0ffee01`), load/resolve split. Go's pre-index B3
   was 11.0 s of linear scans; the index cut wamjs 9.2× — report what Go
   gets, and the with-index vs `index_threshold` behavior if you can
   toggle it. Note `scale_to_case`-style tooling may need the P3 rows it
   previously stripped.

### Guardrails
- You may edit ONLY: `examples/pkg_resolver/go/` (all of it),
  `src/unifyweaver/targets/wam_go_target.pl` + `wam_go_lowered_emitter.pl`,
  `templates/targets/go_wam/`, `tests/test_wam_go_*.pl` (additive; every
  existing test stays green), `docs/WAM_GO_STATUS.md`.
- FROZEN: `examples/pkg_resolver/resolver.pl` and everything else under
  `examples/pkg_resolver/` (`cli/`, `cljs/`, `rust/`, `debian/`, `store*`,
  `wamjs*`, probe/test files), `examples/cli_args/` (all), every other
  target's files, `wam_target.pl`, `wam_text_parser.pl`, shared harness.
  If the resolver's shapes expose a bug you cannot fix inside your owned
  files, document it with a minimal repro instead.
- Wrong output is worse than refusal; the differential is the gate.

### Acceptance (must pass before handoff)
1. Corpus **51/51**; differential **≥2,600 / 0** (paste summaries + wall
   times both legs).
2. Full wam_go suite + cut/frameless/D51 probes green.
3. Shared-lane list green.
4. B1–B3 reported, index effect discussed.
5. The builtin-gap inventory, and whether the warn-knob landed.

### Handoff format
Return: the builtin inventory (found → implemented → classified), shim
changes, corpus/differential numbers + timings, benchmark table with the
index effect, any wam_go bugs found beyond builtins (symptom → cause →
fix → probe), status-doc updates, and residuals.

## ↑↑↑ Copy to here
