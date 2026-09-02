<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# `pkg` — the CLI, as two Prolog specs composed at the edges

This is the front door for uw-resolve. It is not a third program: it is the
**oracle-verified argparser** (`examples/cli_args/cli_args.pl`, compiled through
`wam_javascript`) driving the **frozen-base resolver** (`../resolver.pl`, P0.5,
compiled the same way), with a small amount of JavaScript holding the two
together.

```
argv ──▶ compiled cli_args parse_args/3 ──▶ dispatch ──▶ compiled resolver ──▶ output
          (registry from pkg_schema.pl)                   (../wamjs/resolver.mjs)
```

## The "two specs, zero logic in JS" claim, stated precisely

`pkg.mjs` contains exactly four kinds of code, and nothing else:

| in `pkg.mjs` | not in `pkg.mjs` |
| --- | --- |
| conversion (`"2.35.0"` ⇄ `[2,35,0]`, catalog file → JSON data, resolver answers → the JSON document below) | any flag rule, `--`-handling, arity check, or usage-error wording — every one of those strings is produced by compiled `cli_args` |
| dispatch (which command calls which resolver query) | any candidate ordering, constraint comparison, layer walk, freeze rule, orphan rule or upgrade closure — all compiled `resolver.pl` |
| formatting (the tables below, and `JSON.stringify`) | |
| exit codes (a table keyed by the document's `status`) | |

**The command grammar is not in `pkg.mjs` either.** It lives in
[`pkg_schema.pl`](pkg_schema.pl) as a `cli_args` registry term;
[`derive.pl`](derive.pl) renders it to `generated/pkg_registry.json` in the
object shape `examples/cli_args/wamjs/cliArgs.mjs` already knows how to convert
back into the Prolog term. `pkg.mjs` hands that object to `parseArgs(argv,
registry)` — the transpiled `parse_args/3` — and reads the answer.

**One deliberate exception, called out rather than hidden:** `pkg deps` is a
*projection* over the catalog's own `depends` rows (filter by name, sort by
version then dependency), not a resolver query — `resolver.pl` exports no
"direct dependencies" relation. It filters rows; it does not interpret them.
Because it never enters the resolver, it is also the one command that does
**not** apply alias rewriting (aliases are a resolver request-edge rule).

## Vocabulary

Names follow **Pkg** (Puppy Linux, `usr/sbin/pkg`) wherever Pkg already had a
command for the same question — see `docs/proposals/PKG_MINING_NOTES.md` §2 —
and are spelled as *questions* where uw-resolve answers something Pkg could not
ask at all.

| `pkg` command | Pkg heritage | resolver query | what it answers |
| --- | --- | --- | --- |
| `resolve <name>…` | `add\|a`, `install\|i` | `resolve/3` | the naive closure — the one stock apt would run, base or no base |
| `install-plan <name>…` | `get-only\|go` (dry-run demand, §7) | `resolve_layered/3` + `layer_closure/3` | the plan that leaves every loaded layer alone, plus its install order |
| `layer <name>…` | `sfs-combine\|sc`, `pkg-combine\|pc` (§5) | *(alias of `install-plan`)* | the same manifest, under the SFS-builder's word for it |
| `why-blocked <name>` | *(no Pkg equivalent)* | `explain_blocked_list/3` | which frozen-base holds are the version ceiling, and by how much |
| `deps <name>` | `deps\|e`, `list-deps\|le` | *(catalog projection)* | the raw `depends` rows for a package |
| `what-needs <name>` | `what-needs\|wn` (§3.6, top-10 #2) | `dependents/3`, `dependents_installed/3` | reverse dependencies; `--installed` narrows to what is actually here |
| `orphans <name>` | the orphan trim in `remove\|rm` (§3.4) | `removal_orphans/3` | what removing it would strand — Pkg's own dead code path, made to work |
| `why-frozen <name>` | *(no Pkg equivalent)* | `freeze_audit/2` | one hold's reason, including the derived `over_frozen` / `suggest` verdicts |
| `audit` | *(no Pkg equivalent)* | `freeze_audit/2` | every hold's reason — the "TrixiePup freezes too much" diagnosis as a query |
| `safe-upgrade <name> <ver>` | *(no Pkg equivalent)* | `safe_upgrade/4` | may this hold move, and what has to move with it |

Global options (declared per command, see below): `--catalog <file>`, `--json`.
`what-needs` additionally takes `--installed`.

### Where the options may go

`cli_args`' `GLOBAL_OPTIONS` are fixed at `--state` / `--name`: they belong to
the parser being mirrored, not to `pkg`. So `pkg`'s own cross-cutting options
are declared **per command** in the registry, and must come *after* the command
word. An option before the command would drop the whole line onto `cli_args`'
legacy lenient parser, where nothing is checked at all; `pkg` refuses it
instead:

```
$ pkg --catalog c.json resolve editor
pkg: the command must come first (got --catalog)
```

`PKG_CATALOG` in the environment is the fallback when `--catalog` is absent.

## Exit codes

| code | meaning | examples |
| --- | --- | --- |
| `0` | the query answered | a selection, a plan, `why-blocked` finding nothing, a `safe`/`coordinated` verdict |
| `1` | the query is false, or the answer is "blocked" | no solution; `why-blocked` naming a ceiling; `why-frozen` on a package that is not held; an `unsafe` / `no_candidate` verdict |
| `2` | usage error | every message compiled `cli_args` produces, plus `unknown command` / `no catalog` / `bad version`, which the CLI owns |

Concretely, `status` in the JSON document maps `ok`/`clear` → 0,
`blocked`/`fail` → 1; usage errors never produce a document.

## Output

Human-readable by default, one fact per line, columns aligned:

```
$ pkg why-blocked firefox-esr --catalog examples-frozen-base.json
firefox-esr is blocked by the frozen base — 2 ceilings
  glibc  needs >=2.35.0  base has 2.31.0
  gtk    needs >=3.24.0  base has 2.24.0
```

`--json` prints the machine form. **Every** document has `command` and
`status`; the rest is per command.

### JSON schema

Shared shapes:

```
Version    := "MAJOR.MINOR.PATCH"                      e.g. "2.35.0"
Pair       := { "name": string, "version": Version }
Constraint := { "op": "any" }
            | { "op": "eq"|"gte"|"lt", "version": Version }
            | { "op": "range", "lo": Version, "hi": Version }
Status     := "ok" | "fail" | "blocked" | "clear" | "not-frozen"
```

| command | document |
| --- | --- |
| `resolve` | `{command, status:"ok", requests:[string], selection:[Pair]}` · on failure `{command, status:"fail", requests, reason:"no_solution"}` |
| `install-plan` | `{command:"install-plan", status:"ok", requests, selection:[Pair], manifests:[{request:string, order:[Pair]}]}` · on failure `{…, status:"fail", reason:"no_solution"\|"no_manifest_for:<name>", manifests:[]}` |
| `why-blocked` | `{command, status:"blocked"\|"clear", request:string, blocked:[{name, needs:Constraint, base_has:Version}]}` |
| `deps` | `{command, status:"ok", package:string, depends:[{version:Version, dep:string, constraint:Constraint}]}` |
| `what-needs` | `{command, status:"ok", package:string, installed_only:bool, dependents:[Pair]}` |
| `orphans` | `{command, status:"ok", package:string, orphans:[Pair]}` |
| `why-frozen` | `{command, status:"ok", package, kind:"held"\|"suggest"\|"over_frozen", reason:string\|null}` · `{…, status:"not-frozen", kind:null, reason:null}` |
| `audit` | `{command, status:"ok", audit:[{name, kind, reason:string\|null}]}` |
| `safe-upgrade` | `{command, status, package, version:Version, result:R}` where `R` is one of `{verdict:"safe", cost}` · `{verdict:"coordinated", set:[Pair]}` · `{verdict:"unsafe", reason}` · `{verdict:"no_candidate"}` |

`manifests` is one entry per requested name; the CLI never merges two
`layer_closure` answers, because merging them would be resolution.

`install-plan`'s `selection` is unordered-but-sorted (the layered selection);
`order` is `layer_closure`'s topological manifest, dependencies first.

## Files

| path | what |
| --- | --- |
| `pkg_schema.pl` | **the command grammar**, as a `cli_args` registry term. The only place the surface is defined. |
| `pkg.mjs` | the CLI: conversion + dispatch + formatting + exit codes. |
| `examples/teaching.pl` | the small catalog (`catalog/6`, P0 shape) — classic resolve and the layered plan disagree, and the disagreement has a name. |
| `examples/frozen_base.pl` | the P0.5 catalog (`catalog/9`) — freeze reasons, a loaded `devx` layer, an exclusion, aliases, an over-frozen hold, a coordinated upgrade. |
| `derive.pl` | **the regenerator.** Renders the registry, both catalogs, and the whole expected corpus (SWI running `resolver.pl`) into `generated/`. |
| `generated/pkg_registry.json` | derived from `pkg_schema.pl`. |
| `generated/catalogs/*.json` | derived from `examples/*.pl` — the form the resolver shim reads. |
| `generated/expected.json` | derived: every case's `--json` document and exit code, straight from SWI. |
| `test_pkg_cli.mjs` | the contract corpus: `pkg.mjs` as a subprocess, human **and** `--json`, against `generated/expected.json`. |

Nothing under `generated/` is hand-written or hand-edited.

## Build / run

```bash
# regenerate everything derived (registry, catalogs, expectations)
swipl -q -g derive -t halt examples/pkg_resolver/cli/derive.pl

# the CLI contract corpus
node --test examples/pkg_resolver/cli/test_pkg_cli.mjs

# drive it by hand
export PKG_CATALOG=examples/pkg_resolver/cli/generated/catalogs/frozen_base.json
node examples/pkg_resolver/cli/pkg.mjs audit
node examples/pkg_resolver/cli/pkg.mjs safe-upgrade glibc 2.35.0
node examples/pkg_resolver/cli/pkg.mjs why-blocked firefox-esr --json
```

The two compiled artifacts this composes are built by their own scripts and are
not rebuilt here:

```bash
bash examples/cli_args/wamjs/build.sh        # the argparser
bash examples/pkg_resolver/wamjs/build.sh    # the resolver
```

## How the expectations are derived, not typed

`derive.pl` runs each corpus case against `resolver.pl` **in SWI-Prolog** and
writes the CLI's documented JSON document plus the exit code. `test_pkg_cli.mjs`
then runs `pkg.mjs` as a subprocess and compares:

* `--json` output, deep-equal against the SWI-derived document;
* human output, against that same document put through `renderHuman` — the
  renderer is a pure function of the document, so the human expectation is the
  SWI answer too, and the test fails if either the document or its printing
  drifts;
* a handful of **golden literals** for the demo-critical lines (the blocked
  explanation, the coordinated set, the over-frozen audit row, the alias, the
  empty layer-satisfied plan, and the classic-vs-layered contrast) so that a
  formatting regression cannot pass by moving both sides at once;
* the usage errors, byte for byte — including `unknown option --nope`,
  `missing argument: name`, `unexpected extra argument: extra` and
  `--catalog needs a value`, all of which are strings compiled `cli_args`
  produces, never `pkg.mjs`.

To change an expectation you change the spec and re-run `derive.pl`. There is no
other way to change it.

## Deferred

* **`--ask` / interactivity** (mining notes §7, top-10 #9). A universal
  confirm-everything prefix belongs to a *plan executor*; `pkg` today is
  read-only, so there is nothing to confirm. It also wants a `cli_args`
  global option, and `GLOBAL_OPTIONS` is part of the mirrored oracle.
* **stdin composition** (`pkg li vim | pkg status -`, mining notes §7). The
  `-`-means-stdin convention needs a parser rule, and parser rules live in
  `cli_args.pl` — which is frozen against its own oracle. Deferred until
  there is a reason to fork the grammar.
* **write paths** — install / remove / commit a layer. `pkg` prints plans; it
  never applies one.
* **bb host / GP-LMDB catalog backend.** `--catalog` reads one JSON file. The
  P2 indexed fact-source path is where a real catalog comes from.
* **Debian epoch/tilde versions, `Provides:`/virtuals** — resolver-side P0.5
  limits, inherited as-is.
* **`search` / `list-installed` / repo management** — Pkg's catalog-ingestion
  and lookup half. Those are `package/2` lookups and repo plumbing, not
  resolution queries; they need the P2 catalog backend first.
* **A `help` command.** Usage is printed on any usage error; there is no
  `pkg help` because it would be a command with no query behind it.
