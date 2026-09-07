# `cli_args` — a Prolog reference implementation of peerhailer's CLI parser

**Step A1 of the UnifyWeaver transpilation maturity demonstration.**

This directory holds a pure-Prolog reimplementation of
[peerhailer](https://github.com/s243a/peerhailer)'s `src/cliArgs.js` — the typed
`hail` argument parser — together with the JavaScript oracle it was written
against, a plunit port of the oracle's 17-test contract matrix, and a
differential harness that runs both implementations over a seeded sample and
compares the results.

The Prolog here is the *source of truth* for the later steps: A2 pushes it
through the `wam_javascript` target and A3 through the pattern-based JS targets,
and both are judged against this same oracle. So the code mirrors the oracle
exactly — including its quirks — rather than cleaning anything up.

## Files

| file | what it is |
| --- | --- |
| `cli_args.pl` | the reference implementation — module `cli_args`, exports `parse_args/2`, `parse_args/3`, `default_registry/1`, `global_options/1`, `is_long_flag/1`, `looks_like_legacy_flag/1` |
| `test_cli_args.pl` | plunit port of all 17 corpus tests; entry point `test_cli_args/0` |
| `diff_runner.pl` | Prolog side of the differential harness (stdin argv-lines → JSONL) |
| `diff_runner.mjs` | oracle side of the differential harness (same protocol) |
| `gen_cases.mjs` | seeded, reproducible case generator |
| `compare_jsonl.mjs` | semantic JSONL comparer (flags compared as objects, not text) |
| `run_differential.sh` | the whole harness in one command |
| `oracle/cliArgs.js` | **read-only** vendored oracle — peerhailer `src/cliArgs.js` |
| `oracle/cliArgs.test.mjs` | **read-only** vendored corpus — peerhailer `test/cliArgs.test.mjs` |
| `oracle/package.json` | scoping file only (`"type": "module"`), so the vendored `.js` loads as ESM under UnifyWeaver's CommonJS-default root `package.json` |

Oracle provenance: **peerhailer @ `08ad35e`**
(`feat(cli): hail route status/approve/send — drive confidential routing from
the CLI (#59)`). The two `oracle/` files are byte-identical copies and keep
peerhailer's own header; they are never edited here. `oracle/cliArgs.test.mjs`
is kept for reference only — it imports `../src/cliArgs.js`, a path that exists
in peerhailer's tree, not this one, so run it from the peerhailer checkout
(acceptance step 3 below). Its content is what `test_cli_args.pl` ports.

## The `Result` convention

```prolog
parse_args(+Argv, -Result)
parse_args(+Argv, +Registry, -Result)
```

* `Argv` — a list of SWI **strings** (the raw tokens, i.e. `process.argv.slice(2)`).
* `Result` — `ok(Positional, Flags)` or `error(Message)`.
  * `Positional` — list of strings, in the oracle's `positional` order.
  * `Flags` — list of `Key-Value` pairs. `Key` is a string; `Value` is a string
    or one of the atoms `true` / `false`.
  * `Message` — the **exact** string the oracle's `CliError` carries
    (`"unknown option --force"`, `"--profile needs a value"`,
    `"missing argument: name"`, `"unexpected extra argument: yes"`).

`parse_args/2` **never throws for a usage error and never fails** — every usage
error comes back as `error(Message)`. The JS transpile maps that back at the
module edge:

```js
const r = parse_args(argv);          // the compiled predicate
if (r.tag === "error") throw new CliError(r.message);
return { positional: r.positional, flags: r.flags };
```

This is deliberate: `throw` is the single construct in the oracle that has no
clean first-order Prolog counterpart, so it is pushed out of the compiled core
and reintroduced only in the hand-written shim.

### Flags ordering

`Flags` is an *ordered association list* that reproduces JS object key-insertion
order:

1. leading globals first (in the order they appear before the command),
2. then the strict/lenient parse order,
3. a later assignment to an existing key **overwrites the value but keeps the
   original position** — matching `obj[k] = v` and `{ ...leadingGlobals, ...flags }`.

Verified against the oracle:

```
--name N --state S block bob --state T   →  ["name"-"N", "state"-"T"]
--state A block bob --state B --include-key →  ["state"-"B", "include-key"-true]
daemon --chat=false --route --ui=0       →  ["chat"-false, "route"-true, "ui"-true]
```

*(Not modelled: JS orders integer-like keys first, so `{"a":1,"2":3}` enumerates
as `2, a`. A flag key can only be integer-like via a token like `--2`, which
neither `isLongFlag` nor `looksLikeLegacyFlag` accepts and which the strict
parser rejects as an unknown option; it is reachable only in the lenient
fallback. The differential comparer compares `flags` as an object, so ordering
is not part of the exactness claim anyway.)*

## The schema, as data

`parse_args/3` takes a registry of the same shape `default_registry/1` returns,
mirroring the JS `registry?` parameter:

```
Registry    ::= list of Name-Entry
Entry       ::= schema(Options, Positionals)   % a leaf command
              | group(Actions)                 % grouped: keyed by first positional
Actions     ::= list of ActionName-schema(Options, Positionals)
Options     ::= list of OptionName-Kind
Kind        ::= boolean | string | optional
Positionals ::= list of names; "[name]" is optional, "...name" is variadic
```

A grouped command — `route`, verbatim from `cli_args.pl`:

```prolog
"route"-group([
    "discover"-schema([ "dest"-string, "dest-file"-string,
                        "control"-string, "port"-string ], []),
    "status"-schema([ "dest"-string, "dest-file"-string,
                      "control"-string, "port"-string ], []),
    "approve"-schema([ "dest"-string, "dest-file"-string,
                       "seal-key"-string, "seal-key-file"-string,
                       "control"-string, "port"-string ], []),
    "send"-schema([ "dest"-string, "dest-file"-string, "public"-boolean,
                    "ttl"-string, "budget"-string,
                    "control"-string, "port"-string ],
                  ["...message"])
])
```

and a leaf command:

```prolog
"block"-schema(["include-key"-boolean], ["name"]).
```

The four oracle mechanisms map one-to-one onto four predicate groups:

| oracle | Prolog |
| --- | --- |
| `parseArgs` leading-globals scan | `scan_leading_globals/4`, `is_global_key/1` |
| `schemaFor` | `schema_for/5`, `registry_entry/3`, `action_entry/3` |
| `parseStrict` | `parse_strict/4`, `strict_loop/8`, `strict_option/11`, `check_arity/3` |
| `parseLenient` | `parse_lenient/3`, `lenient_loop/5` |

## Running it

### 1. The contract matrix (plunit)

```
$ swipl -q -g test_cli_args -t halt examples/cli_args/test_cli_args.pl
.................
cli_args contract matrix: 17/17 tests passed
```

### 2. The differential harness

```
$ bash examples/cli_args/run_differential.sh
== generating seeded cases ==
cases: 5067 lines -> .../examples/cli_args/.diff_out/cases.txt
== running the JavaScript oracle (peerhailer parseArgs) ==
== running the Prolog reference implementation ==
== comparing ==
sample size:          5067 argv-lines
  oracle ok results:  4150
  oracle errors:      917
divergences:          0
message mismatches:   0
```

The sample is the 17-test corpus (25 lines), a 42-line hand-written quirk
sweep, 3000 uniform pseudorandom lines of length 2–7 over the task's token
alphabet, and 2000 more of the same rooted at a command token so the strict path
gets deep coverage. The PRNG is mulberry32 with a fixed seed, so the sample is
byte-reproducible.

Artifacts land in `.diff_out/` (git-ignored) for inspection: `cases.txt`,
`oracle.jsonl`, `prolog.jsonl`.

The two runners share a stdin/stdout protocol, so either can be swapped for a
transpiled build in A2/A3 without touching the comparer:

```
stdin  : one argv-line per line, tokens space-separated; lines with no tokens skipped
stdout : one JSON object per input line —
         {"ok":{"positional":[...],"flags":{...}}}  |  {"error":"<message>"}
```

### 3. The oracle's own suite (unchanged)

```
$ node --test /home/user/s243a/peerhailer/test/cliArgs.test.mjs
# pass 17
# fail 0
```

## Oracle subtleties chased down

Each of these is a place the oracle's *actual* behaviour differs from what a
"reasonable" reimplementation would do. All are modelled and all are covered by
the quirk sweep in `gen_cases.mjs`.

1. **Two different flag regexes.** The strict parser uses
   `isLongFlag = /^--[a-z][a-z0-9-]*(=|$)/i`; the lenient parser uses
   `looksLikeLegacyFlag = /^--[a-z][a-z0-9-]*$/i` — bare `--word` only. That is
   the whole reason `tunnels --a --b=c` reads `--b=c` as the *value* of `--a`.
   Both are re-expressed as character logic (`is_long_flag/1`,
   `looks_like_legacy_flag/1`) — no `library(pcre)`, so the compilers only see
   first-order Prolog. The `(=|$)` alternation needs no backtracking: `=` is not
   in `[a-z0-9-]`, so the maximal run always stops at it, and the character
   after any *shorter* run is by construction a run character (neither `=` nor
   end-of-string). Hence "maximal run, then require `=` or end" is exact.
2. **A PEM is not a flag.** `-----BEGIN-PUBLIC-KEY-----` has `-` as its third
   character, so `isLongFlag` rejects it and `--key -----BEGIN-...` consumes it
   as a value. Getting the "first char after `--` must be a *letter*" rule wrong
   silently breaks `--key`, `--dest` and `--seal-key`.
3. **`--` is asymmetric between the two parsers.** `parseStrict` treats it as a
   terminator (its tail becomes pure positionals, so `--state child.json` after
   it stays payload); `parseLenient` has no `--` handling at all, so `--` there
   parses as a flag with the **empty-string key** (`tunnels -- bob` →
   `{"": "bob"}`). The leading-globals scan stops at `--` without consuming it,
   and `--` is never accepted as a global's value (`--state --` leaves
   `state: true` in the scan — and then, because the command position holds
   `--`, the whole line falls to the lenient parse, where `--state` *does* take
   `--` as its value: `{"state":"--"}`).
4. **Boolean truthiness is `!== "false"`, not "is a boolean-ish word".**
   `--ui=0`, `--ui=no`, `--ui=off` are all **on**; only `--ui=false` is off.
5. **The `optional` kind and `unknown option` share one `if (!kind)` test.** An
   option whose kind is falsy is refused; the `optional` kind then falls through
   the `string` branch, so `--debug` with nothing usable after it lands as bare
   `true` while `--debug 2` takes the value.
6. **Bracket stripping in the missing-argument message.** `names[values.length]`
   is reported after `.replace(/[[\]]/g, "")`, so an `[optional]` name in that
   slot appears without its brackets. (Reachable only via a schema where an
   optional precedes a required positional; modelled anyway.)
7. **JS property lookup walks `Object.prototype`.** This is the big one. The
   oracle looks things up with plain property access on object literals —
   `key in GLOBAL_OPTIONS`, `options[key]`, `registry[command]`,
   `entry.actions[action]` — so an inherited `Object.prototype` member answers as
   a **present, truthy** entry:
   * `--toString x block bob` → `toString` passes `key in GLOBAL_OPTIONS`, so it
     is consumed as a *leading global* → `{"toString":"x"}`.
   * `block bob --toString v` → `options["toString"]` is a function: truthy, so
     not "unknown", and neither `"boolean"` nor `"string"`, so it takes the
     `optional` path → `{"toString":"v"}`.
   * `constructor bob` → `registry["constructor"]` is the `Object` function,
     which has no `.actions`/`.options`/`.positionals`, so it parses as
     `schema([], [])` → `unexpected extra argument: bob`.
   * `route toString --state P` → `entry.actions["toString"]` is truthy, so the
     action is *consumed* → `positional: ["route","toString"]`.
   * `--__proto__ x block bob` and `block bob --__proto__ v` → the key resolves,
     the value is consumed, but `obj["__proto__"] = <primitive>` invokes the
     inherited accessor, which **ignores** it — no property is created, so the
     flags come back empty.

   `cli_args.pl` models this with `js_object_prototype_keys/1` plus the
   `__proto__` no-op in `flags_set/4`. It is isolated to four places
   (`is_global_key/1`, `option_kind/3`, `registry_entry/3`, `action_entry/3`)
   and is the only part of the file that exists purely to mirror a JS artifact.
8. **`maybeAction` falsiness.** `maybeAction && entry.actions[maybeAction]` —
   an empty-string action token is falsy in JS, so `parseArgs(["route",""])`
   falls to the lenient parse rather than looking `""` up. Modelled by the
   `Action \== ""` guard in `schema_for/5`.
9. **The lenient fallback is passed the *whole* argv.** When the leading-globals
   scan finds no command (or a non-global `--token` in command position), the
   accumulated globals are **discarded** and `parseLenient(argv)` re-parses from
   token 0. So `--state P --force x` never sees a strict parse and `--force`
   greedily eats `x`.
10. **Uppercase matters twice.** Both regexes are case-insensitive, so `--STATE`
    *is* a long flag — but the option table lookup is case-**sensitive**, so
    `block --STATE x` fails with `unknown option --STATE` rather than falling
    through to a positional.

## Notes for A2 (`wam_javascript`) and A3 (pattern targets)

The file is written to be transpilable, but a few things are worth flagging:

* **No cuts, no exceptions, no `assert`/`retract`, no `library(pcre)`, no
  `library(apply)`.** `cli_args.pl` uses only `sub_string/5`, `string_length/2`,
  `string_concat/3`, `string_chars/2`, `char_code/2`, `length/2`, `reverse/2`,
  `append/3`, arithmetic comparison and `if-then-else`. `format/2` appears only
  in `diff_runner.pl` and `test_cli_args.pl`, never in the module under test.
  (`test_cli_args.pl` does use `exclude/3`, `forall/2` and `split_string/4` — it
  is harness code, not compilation input.)
* **SWI strings vs atoms.** Everything — tokens, keys, values, schema names — is
  an SWI *string*, compared with `==`/`\==`, never with `=`. A target that
  collapses strings to atoms (or to char lists) must keep `""` distinct from
  `''` and must not intern, because `flags_set/4` compares keys by `==`. The
  empty-string flag key from `--` (subtlety 3) is a live case.
* **Mixed-type values.** `Flags` values are strings *or* the atoms `true`/`false`.
  A target with a tagged value representation must keep `true` (atom) distinct
  from `"true"` (string) — the corpus asserts `flags["include-key"] === true`,
  not `"true"`. `emit_json_value/1` in `diff_runner.pl` is the one place that
  discrimination is observable.
* **Determinism.** Every predicate in `cli_args.pl` is det or semidet, and
  `parse_args/2` is verified det (`deterministic/1` is `true` after the call).
  Two predicates were reshaped to *stay* det under first-argument indexing:
  `merge_flags/3` delegates to `merge_flags_/3` so the driving list is argument
  one, and `nth0_default/4` is a single clause with an explicit conditional
  because its first argument is an integer. A target that emits choice points
  for these will not be wrong, but it will be slow and it will diverge from the
  det-ness assumption the JS shim relies on (it calls the predicate once and
  reads a single solution).
* **Structural output arguments.** `parse_strict/4` and `check_arity/3` return a
  tagged term (`ok(...)`/`err(Message)`) instead of throwing. Targets that
  flatten compound terms need to keep the tag; this is the pattern the whole
  "no exceptions in the compiled core" design rests on.
* **`string_chars/2` in both directions.** `is_long_flag/1` decomposes a string
  to chars; `strip_brackets/2` rebuilds one from chars. A target that only
  implements the decompose direction will fail `missing argument:` messages for
  `[optional]` names.
* **Deep recursion.** `lenient_loop/5` and `strict_loop/8` recurse once per argv
  token with accumulators, so they are last-call-optimisable. Argv is short in
  practice, but a target without LCO will hold one frame per token.
* **Integer arithmetic only.** `is`, `=:=`, `<`, `>`, `>=`, `=<` on small
  non-negative integers (string indices, counts, character codes). No floats,
  no bignums.
* **`char_code/2` and code-point ranges.** `js_alpha/1` and `js_flag_char/1`
  compare against `0'a`, `0'z`, `0'A`, `0'Z`, `0'0`, `0'9`, `0'-`. A target
  compiling `0'x` literals incorrectly breaks both regexes silently — the
  differential harness is the guard against exactly that.
* **Reusing the harness.** A2/A3 should re-point `run_differential.sh`'s Prolog
  leg at the transpiled build (it only needs the stdin/stdout protocol above)
  and re-run with the same seed. Same cases, same expected JSONL, zero
  divergences is the bar.
