# WAM C++: ISO Error Terms — Philosophy

How the C++ WAM target should report runtime errors to user Prolog
code. Discusses the alternatives we considered, why we picked
compile-time per-predicate dispatch over runtime flags, and how the
config format choice falls out of "this is a Prolog codebase
configuring Prolog predicates."

Companion docs:
- `WAM_CPP_ISO_ERRORS_SPECIFICATION.md` — the concrete shape we ship.

## 1. What ISO errors are

Standard Prolog (ISO/IEC 13211-1) specifies that certain builtins
**throw** rather than fail when given malformed arguments:

```prolog
?- X is foo.        % type_error(evaluable, foo/0)
?- X is _.          % instantiation_error
?- succ(-1, _).     % type_error(not_less_than_zero, -1)
?- atom_length(X, _). % instantiation_error
```

The thrown term has the structure `error(ErrorType, Context)` where
`ErrorType` is one of a fixed set (`type_error/2`,
`instantiation_error`, `domain_error/2`, `existence_error/2`,
`permission_error/3`, `evaluation_error/1`, ...) and `Context` is
implementation-defined.

The catch/3 + throw/1 infrastructure landed earlier already makes the
matching side work — user code can write
`catch(Goal, error(type_error(_, _), _), Handler)` today. **What's
missing is the builtins themselves throwing structured errors instead
of just failing.**

## 2. Why this is more configurable than it looks

The naive framing — "add ISO errors everywhere" — assumes a single
mode for the whole program. Two reasons that's wrong.

**Compatibility with existing code.** Pre-existing Prolog code may
rely on failure semantics for control flow:

```prolog
% Common pre-ISO idiom: try to compute, fall back if fails.
safe_div(_, 0, fallback) :- !.
safe_div(X, Y, R) :- R is X / Y.   % might throw under ISO
```

The first clause's cut handles the divide-by-zero case in pre-ISO
mode but is bypassed if `is/2` throws. Code like this exists in real
codebases; we don't want to break it.

**Performance.** The user raised this concern in review. The short
answer: for the builtins we ship today (`is/2`, arith compares,
`succ/2`) ISO errors are essentially free on the happy path because
the validating check already exists — only the *action* on the
failed check changes:

```cpp
// Today
if (!ok) return false;

// ISO
if (!ok) { throw_iso_error("type_error", ...); return false; }
```

The branch existed already; the body of the false-branch grows. Happy
path: byte-identical.

The longer answer: this **isn't true in general**. A future
`atom_length/2` or `functor/3` implemented lazily (returns false on
non-atom args without an explicit check) would need a new validating
check to throw the ISO-mandated `type_error`. There the ISO version
*does* have a hot-path cost the lax version avoids. So we want a
mechanism that lets each predicate's call sites decide.

## 3. Three knob layouts we considered

### 3.1 Compile-time global flag (rejected)

```prolog
write_wam_cpp_project(Preds, [iso_errors(true)], Dir).
```

Generator emits `#define WAM_CPP_ISO_ERRORS 1`; runtime uses `#if` to
choose throw vs fail. Zero per-op cost when off.

Rejected because it can't express "ISO mode on for new code, lax for
legacy module X." That's the realistic migration story.

### 3.2 Runtime VM flag (rejected)

```cpp
struct WamState { bool iso_errors_enabled = false; ... };
// every error site:
if (vm.iso_errors_enabled) throw_iso_error(...);
else return false;
```

Flexible, dynamic, but adds one branch per error site (even if
predictable). The bigger issue: it's still a *global* knob — there's
no way to scope it to a single predicate without a save/restore stack
on every call, which gets expensive.

### 3.3 Compile-time per-predicate via distinct builtin keys (chosen)

The generator already knows which predicate each instruction belongs
to. We register **three** flavors of each ISO-relevant builtin:

| Form | Key | Behavior | Who writes it |
|---|---|---|---|
| Default | `is/2` | Resolves to lax or ISO at generator time based on the surrounding predicate's mode | The user writes `X is Expr`; the WAM compiler emits `builtin_call is/2` |
| Explicit ISO | `is_iso/2` | Always throws ISO errors | The user writes `is_iso(X, Expr)` directly |
| Explicit lax | `is_lax/2` | Always fails silently | The user writes `is_lax(X, Expr)` directly |

Per-predicate dispatch only rewrites the **default** form. Explicit
`is_iso/2` and `is_lax/2` call sites are never rewritten — they say
what they mean and survive a mode flip on the enclosing predicate.
This is what the user feedback asked for: "declaring a predicate ISO
should mean the default versions in its clauses use ISO, unless we
specify them explicitly as lax."

**Concrete example.** A predicate annotated ISO with a mix of forms:

```prolog
:- iso_errors_override(my_calc/2, true).

my_calc(X, R) :-
    R1 is X + 1,           % default → rewrites to is_iso/2
    R2 is_lax R1 * 2,      % explicit lax — stays lax
    R is_iso R2 - 0.       % explicit ISO — stays ISO (redundant but legal).
```

After the generator pass, the WAM call sites emit:

```
builtin_call is_iso/2, 2    % rewritten from is/2
builtin_call is_lax/2, 2    % unchanged
builtin_call is_iso/2, 2    % unchanged
```

**Properties:**

- **Zero cost on the happy path**, all three flavors. They are
  separate functions; dispatch is the same single `if` chain it was
  before. Mode choice literally just sends you to a different
  function.
- **Per-predicate granularity**: each default call site picks its
  flavor independently. Module A can be ISO-strict, module B lax,
  they coexist in one binary with no global state.
- **No runtime overhead**: the choice was made at compile time;
  nothing to check.
- **Explicit overrides survive**: a developer who wants
  belt-and-suspenders behavior writes `is_iso/2` directly and the
  generator doesn't second-guess them.
- **No coupling between predicates**: changing `pred_a/2`'s ISO mode
  doesn't affect `pred_b/2` even when both call `is/2`.

The cost is some generator complexity (the per-call-site rewrite) and
some runtime code duplication (three flavors per ISO-relevant
builtin, though lax and default can share a function body). Both are
contained and easy to reason about.

## 4. Why a config file

Two-level configuration — global default + per-predicate override —
needs to live somewhere. Options:

### Inline as compile options

```prolog
write_wam_cpp_project(Preds, [
    iso_errors(true),
    iso_errors(legacy_lookup/3, false),
    iso_errors(unsafe_div/3, false)
], Dir).
```

Works for small projects, gets unwieldy past a few overrides, and
mixes "what predicates" with "how to compile them" in the call site.

### A config file

```prolog
% iso_errors_config.pl
:- iso_errors_default(true).
:- iso_errors_override(legacy_lookup/3, false).
:- iso_errors_override(unsafe_div/3, false).
```

```prolog
write_wam_cpp_project(Preds, [
    iso_errors_config('iso_errors.pl')
], Dir).
```

Scales to dozens of overrides. Lives alongside the code, version-
controlled, reviewable. The compile-options form stays available as a
shorthand for one-off cases (and is the underlying mechanism — the
config file expands into the same option terms).

## 5. Config format: Prolog vs JSON vs YAML

We picked Prolog. Three reasons.

**Native syntax for predicate indicators.** `foo/2` is a first-class
Prolog term. In JSON it becomes a string `"foo/2"` that needs
parsing. In YAML, same. Prolog's compound-term syntax handles this
natively.

**Trivial to load.** A Prolog config is just `consult/1` — no parser
to write or maintain. JSON would pull in a JSON library; YAML's
worse.

**Idiomatic for the codebase.** Everything else in UnifyWeaver is
Prolog source. A Prolog config means contributors don't need to
context-switch.

**Counter-arguments considered:**

- JSON is more familiar across teams. True, but the contributors to
  this codebase already write Prolog daily.
- JSON enables tooling integration. True for CI/CD, but the use
  case here is dev-time config read by the Prolog generator — not
  shipped to other systems.
- A config could need richer structure (lists, conditional rules)
  someday. Prolog handles all of those natively; JSON would force
  ad-hoc schema design.

## 6. Default value

**Default off (lax).** Reasoning:

- Existing tests rely on failure semantics in places. Flipping the
  default to ISO would break them en masse. Easier to opt-in
  per-predicate or per-project than to opt-out everywhere.
- ISO-strict is the *long-term* desired state for new code, but new
  code can explicitly request it via the config. Legacy code stays
  working without modification.
- The default can flip later without further design work — that's
  just a one-line change in the config defaults once enough of the
  ecosystem is ISO-clean.

## 7. What's intentionally out of scope

- **Runtime mode switching.** The mode is fixed at compile time. No
  `set_prolog_flag(iso, true)` equivalent.
- **Wildcard / module-level rules.** Overrides are per
  predicate-indicator. If a module wants all its predicates ISO, it
  lists them. We can add wildcards (`module:*` or `*/*`) later
  without breaking the per-predicate form.
- **Builtins not currently shipped.** `atom_length/2`, `functor/3`,
  `arg/3`, etc. would need their own audit when added — see the
  perf discussion in §2.
- **Auto-fixing legacy code.** The audit tool (§9) reports; it
  doesn't rewrite. Migration is opt-in per predicate by editing the
  config.

## 8. Divide-by-zero: throw, NaN, or both?

Worth its own section because the user flagged that "for numbers,
divide-by-zero should give NaN, which has rules in various
programming languages." This is a real tension between ISO Prolog
and floating-point conventions.

**ISO Prolog says:**

- `1 // 0` (integer division) → throw
  `evaluation_error(zero_divisor)`.
- `1 / 0`  (general division) → throw
  `evaluation_error(zero_divisor)`.
- IEEE 754 says floats divide to ±∞ or NaN; ISO predates wide IEEE
  adoption and chose throw uniformly.

**Languages we might compare to:**

- C / C++: integer `1/0` is undefined behavior; float `1.0/0.0` is
  `inf` (no exception).
- JavaScript / Lua: `1/0` is `Infinity`, `0/0` is `NaN`. No
  exceptions.
- Python: integer `1//0` throws `ZeroDivisionError`; float `1.0/0.0`
  *also* throws (Python doesn't follow IEEE here).
- Haskell: integer throws; `Float`/`Double` divide produces `Infinity`
  or `NaN`.

**Our position:** ISO-mode follows ISO (throws for both). Lax-mode
follows IEEE 754: integer divide-by-zero fails silently (preserves
current behavior); float divide produces `inf` or `nan` and succeeds.
This makes the choice consistent with the three-form design — if you
want NaN/inf semantics, write `is_lax/2` (or have your predicate in
lax mode); if you want exception semantics, write `is_iso/2` (or have
it in ISO mode).

Concrete behavior table:

| Expression | Lax / Default(lax) | ISO / Default(iso) |
|---|---|---|
| `X is 1 // 0` | `false` (fail) | throw `evaluation_error(zero_divisor)` |
| `X is 1 / 0` | `false` (fail) | throw `evaluation_error(zero_divisor)` |
| `X is 1.0 / 0.0` | `X = inf` | throw `evaluation_error(zero_divisor)` |
| `X is 0.0 / 0.0` | `X = nan` | throw `evaluation_error(zero_divisor)` |
| `X is -1.0 / 0.0` | `X = -inf` | throw `evaluation_error(zero_divisor)` |

The lax behavior change (currently divide-by-zero fails uniformly;
proposed lax behavior: only integer fails, float produces
inf/nan) is a small breaking change worth flagging. It comes online
with the implementation, not the docs.

## 9. Audit tooling

User feedback called this out as a good idea; moving it from
"probably not" into the plan.

Ship a `wam_cpp_iso_audit/2` predicate alongside the rewrite. Given
a predicate list and a config, it walks each predicate's compiled
WAM and reports:

- Which `builtin_call` sites would resolve to ISO vs lax under the
  current config.
- Which sites would change behavior if the predicate's mode flipped.
- Which sites use explicit `is_iso/2` or `is_lax/2` overrides (so
  reviewers know mode-flips won't touch them).

Output is a Prolog list of records, e.g.:

```prolog
[ audit(my_calc/2, [
      site(pc=12, lax_key=is/2, iso_key=is_iso/2,
           resolved=is_iso/2, source=default,
           would_change_under_lax=true),
      site(pc=18, lax_key=is_lax/2, iso_key=is_lax/2,
           resolved=is_lax/2, source=explicit_lax,
           would_change_under_lax=false)
  ])
]
```

A pretty-printer renders this as a human-readable table for command-
line use. The audit is read-only; it doesn't modify the generator
state. Useful as both a migration aid and a sanity check on the
config file.

## 10. Other open questions

Flagged for review during the implementation PR, not now.

- **Should the `Context` part of `error(_, Context)` carry the call
  site?** ISO lets it be implementation-defined. A minimal version
  is just `Context = _` (unbound). A more useful version stores
  e.g. the predicate indicator that threw. Open.
- **Module-qualified config rules.** Bare `Name/Arity` matches in
  any module today; multi-module projects may want explicit module
  scoping. Revisit when those land.
