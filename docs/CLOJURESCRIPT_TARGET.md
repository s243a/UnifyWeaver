<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# UnifyWeaver ClojureScript Target

The ClojureScript target compiles Prolog predicates to ClojureScript / Clojure
that runs under a **sci** (Small Clojure Interpreter) runtime — in the browser
(Scittle), on Node (nbb), or as a standalone script binary (Babashka / bb).

It is a *variant* of the JVM Clojure target
(`src/unifyweaver/targets/clojure_target.pl`): it `use_module`s the base target
and overrides only the JVM→JS differences, following the same inheritance
pattern the Python family uses
(`python_cython_target : python_target`). The bulk of the codegen — clause
lowering, recursion patterns, expression translation — is reused unchanged.

- Target module: `src/unifyweaver/targets/clojurescript_target.pl`
- Bindings: `src/unifyweaver/bindings/clojurescript_bindings.pl` (67 core/seq/
  collection/string/threading bindings)
- Tests: `tests/core/test_clojurescript_target.pl`,
  `tests/core/test_clojurescript_runtime_smoke.pl`

## Capabilities

- `streaming`, `functional`, `lisp`, `interpreted`
- `browser` / `scittle` — in-browser Scittle/SCI page (no build step)
- `nbb` — Node ClojureScript sci runtime (standalone executable script)
- `babashka` — Babashka (bb) sci runtime, run as an external binary (no npm dep)

## Runtime variants

The runtime is chosen with the `runtime(Kind)` compile option. It selects the
**entrypoint banner / shebang** and, crucially, **whether the JVM→JS host
interop rewrite is applied**.

| `runtime(...)` | Host  | Interop            | Shebang                 | Where it runs |
|----------------|-------|--------------------|-------------------------|---------------|
| *(unset)* / `default` | JS | JVM→JS rewrite | none | historical output; embed in a page |
| `scittle`      | JS    | JVM→JS rewrite     | none (browser page)     | browser (Scittle/SCI) |
| `nbb`          | JS    | JVM→JS rewrite     | `#!/usr/bin/env nbb`    | Node (nbb) |
| `bb`           | JVM   | **none** (kept)    | `#!/usr/bin/env bb`     | Babashka binary |

### Why bb is different

Babashka (`bb`) is **Clojure**, not ClojureScript. Although it too is powered by
sci, its host is the JVM, so it uses JVM-style host interop:
`Integer/parseInt`, `Math/abs`, `(catch Exception e ...)`, `(.getMessage e)`.
For `runtime(bb)` the target therefore **does not** apply the JVM→JS rewrite —
the base Clojure code is emitted verbatim (plus a `bb` shebang). This matches
the peerhailer `shell:bb` idea: bb is an external binary with no npm dependency.

nbb and Scittle are ClojureScript on a JavaScript host, so they get the rewrite
(`js/parseInt`, `js/Math.abs`, `(catch :default e ...)`, `(.-message e)`).

The single JVM→JS rewrite (`clojurescript_interop_rewrite/2`) stays centralized;
the runtime variant only chooses whether to invoke it and which banner/shebang
to prepend (`clojurescript_from_clojure/3`).

## Usage

Compile a predicate to each variant:

```prolog
?- compile_predicate_to_clojurescript(double/2, [], Code).             % default
?- compile_predicate_to_clojurescript(double/2, [runtime(nbb)], Code). % nbb script
?- compile_predicate_to_clojurescript(double/2, [runtime(bb)], Code).  % bb script
```

The generated scripts read their argument from `*command-line-args*`, so they
run as standalone one-arg CLI programs:

```sh
# nbb (Node ClojureScript) — install once: npm install nbb
nbb double.cljs 21          # -> 42

# bb (Babashka) — a single native binary, no npm
bb double.clj 21            # -> 42
```

Both `nbb` and `bb` honour the leading shebang line, so the files can also be
made executable directly (`chmod +x double.clj && ./double.clj 21`).

### Browser (Scittle) page

For the browser, wrap the generated ClojureScript in a Scittle page:

```prolog
?- compile_predicate_to_clojurescript(double/2, [runtime(scittle)], Code),
   generate_scittle_html("Demo", [main_ns('generated.demo'), cljs(Code)], HTML).
```

`generate_scittle_html/3` embeds the code in a `<script type="application/x-scittle">`
tag — the same page the SciREPL `clojurescript` kernel executes. Bundle Scittle
locally for offline use (see the kernel proposal under
`docs/handoff/scirepl-clojurescript-kernel/`).

### shadow-cljs build config

`generate_shadow_cljs_edn/2` emits a `shadow-cljs.edn` (the JS-world analogue of
`deps.edn`) for a full browser build, if you want a compiled bundle instead of
the no-build Scittle path.

## Testing

Unit tests (no runtime required):

```sh
swipl -q -g test_clojurescript_target -t halt tests/core/test_clojurescript_target.pl
```

End-to-end runtime smoke tests actually execute the generated code. Each runtime
variant is **skipped (not failed)** when its binary is absent, so CI passes
without a sci runtime installed:

```sh
# nbb arms (install: npm install nbb)
NBB=./node_modules/.bin/nbb \
  swipl -q -g test_clojurescript_runtime_smoke -t halt \
  tests/core/test_clojurescript_runtime_smoke.pl

# bb arms (bb on PATH, or BB=/path/to/bb)
BB=/usr/local/bin/bb \
  swipl -q -g test_clojurescript_runtime_smoke -t halt \
  tests/core/test_clojurescript_runtime_smoke.pl
```

nbb is discovered via the `NBB` env var then PATH; bb via the `BB` env var then
PATH.

## Future work: squint

[squint](https://github.com/squint-cljs/squint) is a **build-based** cljs→js
compiler (it produces plain JavaScript ahead of time, rather than interpreting
with sci). It is **out of scope** for the current card: unlike bb/nbb/Scittle it
needs a compile step (`npx squint compile`) and a small runtime, so it does not
fit the "run the generated file directly" model the other variants share.

Because squint consumes ClojureScript with JS host interop, it could in
principle reuse the existing JS-interop output (the same rewrite as `nbb`) behind
a `runtime(squint)` option plus a `squint compile` invocation and a generated
`squint.edn`. If added, it should be gated exactly like the bb/nbb runtime smoke
arms (skip when the `squint` toolchain is absent). Not implemented here.
