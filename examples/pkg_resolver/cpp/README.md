<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# uw-resolve on C++ (`wam_cpp`)

Compile the frozen package resolver (`../resolver.pl`, P0.5 + P3 semantics)
through the **C++ WAM target** (`src/unifyweaver/targets/wam_cpp_target.pl`,
interpreter emit mode) and validate the 51-scenario contract corpus against
SWI-Prolog as the oracle.

This is the foundation of the C++ lane (milestone 1 of the cross-target
ladder). It mirrors the native Go lane (`../go/`): a build script that
transpiles the resolver through the WAM backend and compiles the emitted
C++, plus a corpus runner that diffs every answer against SWI.

## Status (measured on this WSL box, SWI 9.x, g++ -std=c++17)

| Gate | Result |
| --- | --- |
| resolver.pl compiles through wam_cpp end-to-end | **yes** (interpreter mode) |
| C++ WAM binary runs the corpus | **yes** |
| Milestone 1 — corpus vs SWI | **51 / 51 match** |
| Milestone 2 — differential 2600/0 | not started |
| Milestone 3 — store backend | not started |

Compile cost (interpreter emit mode, `-O0`): `wam_runtime.cpp` ~5 s / ~0.4 GB
peak, `generated_program.cpp` (150 predicates) ~50 s / ~1.4 GB peak. Well
inside this box's RAM. **Do not** switch to `emit_mode(functions)`/`mixed(...)`
on a low-RAM box without staging — lowered per-predicate C++ is the path that
has OOM'd `cpp_e2e` historically.

## Build / run

From the repo root (SWI 9.x + a C++17 g++/clang on PATH):

```bash
bash examples/pkg_resolver/cpp/build.sh          # transpile + g++ -> cpp/uwresolve
bash examples/pkg_resolver/cpp/run_corpus_cpp.sh # 51-case corpus vs SWI
```

## How the lane works

Unlike the Go lane's JSON shim, this lane never serialises the catalog to and
from an external format — the C++ WAM's hand-written term parser accepts only
alphanumeric functors (it rejects `-`, so `a-v(1,0,0)` and even `-(a,...)` will
not parse). Instead the corpus data lives **inside** the compiled binary:

- `build.pl` asserts the clauses of `resolver.pl` + the corpus data of
  `test_resolver.pl` (`scenario_catalog/2`, `corpus_case/4`, up to the plunit
  block) + `driver.pl` into `user`, then calls `write_wam_cpp_project/3`
  (`emit_mode(interpreter)`, `emit_main(true)`).
- `driver.pl` is a **shared** driver run on both engines. `emit_all/0` walks
  all 51 `corpus_case/4`, runs each query against the resolver, wraps the
  answer in `ok(_)`/`fail`, and prints `<CaseId> <canonical-term>` per line.
  `ser/2` is a whitespace-free, always-functional canonical term writer using
  only builtins the C++ WAM implements (`functor/3`, `arg/3`, `atom_number/2`,
  `atomic_list_concat/2`), so the two legs are byte-identical whenever the C++
  WAM executes the resolver like SWI. Any per-line diff is a genuine
  C++-vs-SWI divergence, not a formatting artefact.
- `run_corpus_cpp.sh` runs `emit_all/0` on SWI directly (oracle) and on the
  `uwresolve` binary (the generated `main.cpp` shim prints a trailing
  `true` line, which the runner drops), then `diff`s.

The generated main shim (`emit_main(true)`) runs `argv[1]` as a predicate key;
`./cpp/uwresolve 'emit_all/0'` is the whole corpus.

## Milestone 1 resolution

The five original mismatches had two independent causes:

- A sole `\+` clause compiled with the Y-level soft-cut form could emit
  `get_level YN` without first allocating its own environment. The barrier then
  aliased a caller Y-register. The shared WAM compiler now allocates a frame for
  exactly that inline-negation shape, while legacy and runtime-builtin negation
  modes retain their previous allocation behaviour.
- The deb-version cases call `predsort/3` through `sort_versions_desc/2`, but the
  example build did not include the C++ WAM predsort helpers. The build now uses
  `include_stdlib([predsort])`, which pulls in `predsort/3` and its registered
  transitive helpers without changing the frozen resolver.

After both corrections, the generated C++ executable matches the SWI oracle on
all 51 corpus scenarios. The focused Y-register/negation tests and the complete
C++ WAM e2e suite are the regression gates for the shared compiler behaviour.

## Next steps

1. **Milestone 2 (differential 2600/0):** the seeded differential feeds
   generated catalogs as JSON; because the C++ parser can't take catalog text,
   this lane needs either (a) a C++ JSON→WAM-term shim like `../go/shim.go`, or
   (b) the generator emitting Prolog catalog *facts* compiled into a per-run
   binary.
2. **Milestone 3 (store backend):** mirror `../go` D43 indexed store; the C++
   runtime already has an LMDB `FactSource` (`WAM_CPP_ENABLE_LMDB`), so the
   indexed/lmdb split has a starting point.

## Scope

Only C++ codegen/runtime + this `cpp/` directory are in scope. `resolver.pl` /
`resolver_store.pl` are the frozen spec; every fix for a resolver behaviour
goes in the C++ target/runtime, never in the spec.
