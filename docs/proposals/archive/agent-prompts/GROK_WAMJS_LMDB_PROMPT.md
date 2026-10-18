<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — indexed persistent fact stores: LMDB (opt-in) + dependency-free (default) (item 2, GP-LMDB)

> Branch from `grok/wamjs-perf2` (latest JS-WAM) and EXTEND. One fact-source
> interface, TWO storage backends: a dependency-free indexed file store (the
> default, always available) and real LMDB via the `lmdb` npm package (opt-in —
> the FIRST allowed optional dependency, and it stays optional: default builds
> remain no-npm). Missing optional dep = clear error, NEVER a silent fallback.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-perf2`** as `grok/wamjs-lmdb`. Prolog dialect is SWI-Prolog
(`swipl`); the JS runtime targets Node (`node`, v18+).

### The gap (GP-LMDB, item 2 of the post-demonstration roadmap)
D27 gave `wam_javascript` external fact sources (`javascript_wam_fact_sources/1`,
`CallFactStream`, TSV/CSV/JSONL via `fs`, first-arg indexed **in memory** — the whole
file is read and parsed at startup). That's fine for small files and useless for big
ones. Add **persistent indexed stores**: bound-argument lookup that seeks to the
answer WITHOUT loading or scanning the whole file.

### Design — one interface, two backends
Extend the D27 fact-source declaration with a store form (exact option syntax is
yours; keep it consistent with D27's `source(P/N, file(Path))` style), e.g.:
- `source(P/N, indexed(Path))` — **backend B, the default capability, zero deps.**
- `source(P/N, lmdb(Path))` — **backend A, opt-in, uses the `lmdb` npm package.**

Both present IDENTICAL semantics through the existing `CallFactStream` machinery:
unbound call = enumerate all facts (deterministic order — document it); first arg
bound = indexed lookup returning only matching facts; other-args-bound = filter on
the streamed candidates (as D27 does). Cells go through `parse_term` exactly like
D27 (so numbers/atoms/strings/compounds round-trip; note the D34 string-tag and
D37 literal rules apply).

**Backend B — dependency-free indexed store.** Design a two-part on-disk format you
generate at BUILD time: a data file (the facts, e.g. length-prefixed lines or a
compact binary) + an index (first-arg key → file offset(s); a sorted key table
binary-searched via `fs.read` at offsets, or a hash bucket table — your call,
justify it). The runtime must do bound-A1 lookup with O(log n) seeks and WITHOUT
reading the whole data file (prove it: the acceptance includes a lookup on a store
whose data file is large, with a counter showing bytes read « file size). Ship the
builder as a swipl or node tool: `uw_fact_index build <tsv|jsonl> <store>` (name
yours) so a store is reproducible from the same flat files D27 reads. Be honest in
the docs: this is LMDB-STYLE (persistent + indexed + seek-based), not LMDB — our
own format, read-oriented, single-writer-at-build-time.

**Backend A — real LMDB.** Use the `lmdb` npm package (native bindings). Load it
lazily (`await import('lmdb')` / `createRequire` in a try/catch) ONLY when an
`lmdb(...)` source is declared; if the package is missing, throw ONE clear,
actionable error naming the store, the missing package, and the install command —
never fall back to backend B (different formats must never be silently swapped).
Ship a loader tool (node script) that builds an LMDB database from the same flat
files, keyed so first-arg lookup is a range/exact get. Key encoding: preserve the
D34 term distinctions (atom vs string vs number) in your key scheme — document it.
Default builds MUST remain no-npm: no `package.json` dependency entry for the repo;
the LMDB path documents `npm install lmdb` as a user action. This is the fleet's
FIRST optional-dependency tier — write the policy sentence in the status doc
(opt-in per source declaration, absence is a loud error, default builds unaffected).

### Reference
- `templates/targets/javascript_wam/runtime.js.mustache` — D27's `CallFactStream` +
  fact-source machinery (extend, don't fork), `parse_term`, the string tag.
- `src/unifyweaver/targets/wam_javascript_target.pl` + `javascript_wam_bindings.pl`
  — where `javascript_wam_fact_sources/1` is parsed and emitted.
- `tests/test_wam_javascript_fact_sources.pl` — the D27 test patterns to mirror.
- `docs/WAM_BACKEND_CONVENTIONS.md` §7/§8 — still binding.
- Note the fleet doc `docs/WAM_FLEET_GAPS.md` lists LMDB as out-of-scope-matching-
  Lua; you are changing that for THIS target — the status doc should say so.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Runtime + target support for both store forms; the backend-B builder tool; the
  backend-A loader tool.
- Extend `tests/test_wam_javascript_fact_sources.pl`:
  - Backend B fully tested unconditionally: build a store from a fixture (a few
    thousand rows so indexing is meaningful), bound-A1 lookup matches SWI on the
    same facts, unbound enumeration matches, other-arg filtering matches, and the
    bytes-read « file-size proof for a bound lookup.
  - Backend A tests `condition`-gated on the `lmdb` package actually loading in
    the test environment; ALWAYS test (ungated) the missing-package path: an
    `lmdb(...)` source without the package produces the exact actionable error.
  - A shared-semantics probe: the SAME fixture through B and (when available) A
    yields identical results.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: the two store forms, the B format spec
  (so it's reimplementable), the A key encoding, the optional-dependency policy
  sentence, and honest B-vs-LMDB framing.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, `wam_target.pl`, `wam_text_parser.pl`, the conformance
  harness, `docs/WAM_FLEET_GAPS.md`, or anything frozen under `examples/cli_args/`.
  Do NOT add any dependency to a repo `package.json`.

### Constraints — CRITICAL
- D27's existing `file(Path)` sources are byte-for-byte unchanged.
- No silent cross-backend fallback, ever. Missing `lmdb` = the one loud error.
- Do NOT break: 48/48 conformance, builtins/lowered/fact-sources/parser/term-meta/
  string/profiling suites, corpus 17/17, differential 0/0 (re-run all; the
  argparser build doesn't use these stores, so any change there means a regression).

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl` →
   green incl. all new probes (state whether the lmdb-gated arm ran in your env).
2. Bound-lookup proof: the bytes-read counter output for backend B on the large
   fixture, pasted.
3. All other suites + `CONFORMANCE_TARGETS=javascript` green; corpus 17/17;
   differential 0/0.
4. Node-vs-SWI outputs for the shared-semantics probe, pasted.

### Handoff format
Return: the option syntax you chose, the B format spec + why, the A key encoding,
whether the lmdb arm ran in your environment, the bytes-read proof, suite results,
and any residual (e.g. multi-arg secondary indexes, write paths — both are OUT of
scope this round; say so rather than starting them).

## ↑↑↑ Copy to here
