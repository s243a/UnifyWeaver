<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — uw-resolve P2: the seek-indexed catalog backend (GP-LMDB meets the resolver)

> Branch from `grok/pkg-resolver-p05` as `grok/pkg-resolver-p2`. The runtime
> freeze from P0.5 is LIFTED — the cut-semantics hardening has merged — but a
> concurrent CLI round owns `examples/pkg_resolver/cli/` (do not create or
> touch that directory) and the frozen dirs stay frozen. This round connects
> your two artifacts: the D43 indexed fact stores become the resolver's
> catalog backend, so resolution can run against a catalog too big to load.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`).
Branch from **`grok/pkg-resolver-p05`** as `grok/pkg-resolver-p2`. SWI-Prolog
(`swipl`); Node v18+.

IMPORTANT: first update your branch's runtime files to the coordinator tip —
the cut-semantics hardening (barrier model, 12 fixes, new
`tests/test_wam_javascript_cut_semantics.pl`) merged after your parent:
`git checkout origin/claude/peerhailer-exploratory-docs-aodas5 -- templates/targets/javascript_wam src/unifyweaver/targets/wam_javascript_target.pl src/unifyweaver/targets/wam_javascript_lowered_emitter.pl src/unifyweaver/bindings/javascript_wam_bindings.pl tests/test_wam_javascript_cut_semantics.pl tests/test_wam_javascript_lowered.pl docs/WAM_JAVASCRIPT_STATUS.md docs/WAM_BACKEND_CONVENTIONS.md`
Then rebuild `examples/pkg_resolver/wamjs/` via its `build.sh` and confirm the
P0.5 gates still hold (38/38 SWI, 39/39 wamjs, differential 0 divergences)
BEFORE starting P2 work — that is your baseline.

### The gap (P2)
The resolver takes the whole catalog as an in-memory term. Real catalogs
(a distro's package index: tens of thousands of packages) cannot be loaded
per query. You built the answer already: the D43 indexed fact stores
(`indexed(Prefix)` dependency-free seek store; `lmdb(Dir)` opt-in). P2 wires
them together: **store-backed resolution** with bytes-read-proportional-to-
the-query, not to the catalog.

### Design
1. **Catalog dump format + store builder.** Define a JSONL catalog dump (one
   record per fact: package/depends/conflicts/base/layer/excluded/alias rows —
   document the schema in the README) and a builder step that compiles a dump
   into D43 indexed stores keyed for the resolver's access patterns:
   packages-by-name, depends-by-name, conflicts-by-name, reverse-deps-by-
   depname (for `dependents`/`upgrade_set` — precompute at build time so the
   reverse query is also a seek, not a scan). Reuse
   `scripts/js_wam/uw_fact_index.js` (extend if needed); `lmdb(...)` support
   comes free through the same declaration form — smoke-test it gated exactly
   as D43 did (missing-package error path ungated).
2. **Store-backed resolver entry.** Keep `resolver.pl`'s catalog-as-term API
   UNTOUCHED (every existing gate must pass unchanged). Add a thin adapter
   layer — e.g. `resolver_store.pl` — exposing the same 10 queries where the
   per-name catalog lookups are served by fact-source predicates instead of
   list members. On the wamjs side those predicates are D27/D43
   `javascript_wam_fact_sources` declarations (`indexed(Prefix)` stores). On
   the SWI side (the oracle), read the SAME stores or the same JSONL dump —
   your choice, but SWI and node must consume identical data; say which you
   chose and why.
3. **Environment stays small.** `base`/`layers`/`installed`/`requested`/
   `excluded`/`aliases` are machine-local and tiny — they may stay term-side
   (passed per query) while the big three (packages/depends/conflicts +
   reverse-deps) come from stores. This split is the realistic deployment
   shape (huge shared catalog + small local state) — document it.
4. **Scale demonstration.** Generate a synthetic catalog of ≥5,000 packages /
   ≥15,000 dependency edges (seeded), build the stores, and show:
   (a) a bound resolution (`resolve_layered` on one request) touching a tiny
   fraction of the store — paste the bytes-read counter vs total store size
   (the D43 proof, now under real resolution);
   (b) wall time for single-request resolution on the 5k catalog vs the same
   query on a term catalog of the same content (loading cost included);
   (c) the SWI-vs-wamjs differential ON STORE-BACKED resolution: ≥500 seeded
   request/environment cases against the 5k catalog, 0 divergences.

### Gates
- All P0.5 gates unchanged: SWI corpus 38/38, wamjs corpus 39/39, term-catalog
  differential ≥2400 / 0 divergences.
- New: store-backed corpus (the same 38 scenarios driven through the store
  adapter — expected results identical to term-catalog results, asserted) +
  the ≥500-case store-backed differential + the bytes-read proof.
- The three `tests/test_wam_javascript_*.pl` suites + the cut-semantics suite +
  `CONFORMANCE_TARGETS=javascript` + cli_args 17/17 + 5067/0/0 — all green.

### Guardrails
- You may edit: `examples/pkg_resolver/` EXCEPT `cli/` (concurrent round owns
  it — do not create it either), plus `scripts/js_wam/uw_fact_index.js` /
  `uw_fact_lmdb.js` (extensions additive; D43's store tests must stay green),
  plus runtime/emitter ONLY if store-backed resolution exposes a real bug
  (probe-pinned fix, listed in the handoff — the D44 precedent).
- Frozen: `examples/cli_args/` (all), `wam_target.pl`, `wam_text_parser.pl`,
  conformance harness, registry/matrix/glue, `resolver.pl`'s existing API and
  `test_resolver.pl`'s existing scenarios (append-only).
- No repo `package.json` dependency; lmdb stays the opt-in tier per D43's
  policy sentence.

### Acceptance (must pass before handoff)
1. Baseline confirmation (post-runtime-update rebuild): P0.5 gates green.
2. Store-backed corpus: 38/38 identical to term-catalog results.
3. Store-backed differential: ≥500 cases on the 5k catalog, 0 divergences.
4. Bytes-read proof under resolution + the term-vs-store timing comparison.
5. Full pre-existing gate list re-run green.

### Handoff format
Return: the JSONL dump schema + store key layout (incl. the reverse-deps
store), the SWI-side data choice and why, the environment split rationale,
scale numbers (bytes-read, timings, differential), any runtime fixes forced
(symptom → cause → fix → probe), whether the lmdb-gated arm ran in your env,
and residuals (Provides/virtual, epoch/tilde, write paths, incremental store
updates remain deferred).

## ↑↑↑ Copy to here
