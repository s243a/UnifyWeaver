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
| Milestone 1 — corpus vs SWI | **46 / 51 match** |
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

## The one blocker to 51/51

All 5 non-matching scenarios trace to a **single C++ WAM permanent-variable
(Y-register) codegen bug** in interpreter emit mode, triggered by a `\+`
(negation) inside a clause that carries several permanent variables:

- 4 cases are `explain_blocked` (`layered_blocked_explanation`,
  `two_blocked_ceilings`, `deb_epoch_ceiling`, `deb_tilde_ceiling`): they all
  route through `resolver.pl`'s `blocked_acc/5`, whose first if-then-else is
  `( base_ver(Cat,Name,BV), \+ satisfies(BV,C) -> [blocked(...)|Acc0] ; Acc0 )`.
  On C++ that condition is taken to the ELSE branch, so the `blocked(...)` fact
  is dropped and the query returns `ok([])`.
- 1 case is `deb_tilde_picks_release` (`resolve` over deb versions): `resolve`
  fails on C++ where SWI succeeds — same class of bug on `resolve`'s deep
  recursion (its version guards also use `\+`).

It is NOT a resolver.pl issue, a deb-version-comparison issue, or a driver
issue — each primitive works in isolation on the C++ WAM:

- `satisfies/2`, `\+ satisfies/2`, `base_ver/3`, `walk_pkg_for_blocked/5`,
  `collect_deps/4`, and the deb `version_lt/2` (tilde ordering) all return the
  correct result standalone;
- the same first if-then-else, reproduced with fewer permanent variables,
  returns the correct non-empty accumulator (`bug_repro_ok/0`);
- adding the surrounding permanent variables of the real clause flips it to
  the empty branch (`bug_repro_bug/0`).

Minimal reproducer, both shipped in `driver.pl`:

```bash
./cpp/uwresolve 'bug_repro_ok/0'    # then=v(1,0,0)   (condition succeeds)
./cpp/uwresolve 'bug_repro_bug/0'   # after_ite1=[]    (same condition, ELSE)
```

SWI runs both identically (non-empty). Fixing it means the C++ WAM target's
Y-register allocation / negation lowering (`wam_cpp_target.pl` +
`\+`/`NegationReturn` handling in the emitted `wam_runtime.cpp`), cross-checked
against the existing `cpp_e2e_yreg_isolation` / `cpp_e2e_not_*` suite. That is
a codegen fix outside this lane; it is deliberately left for a focused C++ WAM
change rather than patched around here (the resolver is frozen).

## Next steps

1. **Milestone 1 → 51/51:** fix the Y-register/negation codegen bug above, or
   (to unblock sooner and localise risk) try `emit_mode(mixed([blocked_acc/5,
   resolve/3, resolve_pending/... ]))` so only the affected predicates are
   lowered to direct C++ — smaller than a full `functions` build, but confirm
   RAM headroom before running it.
2. **Milestone 2 (differential 2600/0):** the seeded differential feeds
   generated catalogs as JSON; because the C++ parser can't take catalog text,
   this lane needs either (a) a C++ JSON→WAM-term shim like `../go/shim.go`, or
   (b) the generator emitting Prolog catalog *facts* compiled into a per-run
   binary. Blocked behind milestone 1 anyway.
3. **Milestone 3 (store backend):** mirror `../go` D43 indexed store; the C++
   runtime already has an LMDB `FactSource` (`WAM_CPP_ENABLE_LMDB`), so the
   indexed/lmdb split has a starting point.

## Scope

Only C++ codegen/runtime + this `cpp/` directory are in scope. `resolver.pl` /
`resolver_store.pl` are the frozen spec; every fix for a resolver behaviour
goes in the C++ target/runtime, never in the spec.
