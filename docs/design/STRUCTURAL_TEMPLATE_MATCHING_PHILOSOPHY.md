# Structural Template Matching — Philosophy

*Why the template system's `{{match}}`/`{{case}}` dispatch might grow from exact-string
comparison to term structure, what that would actually buy, and what it would cost. The current
mechanism lives in `src/unifyweaver/core/template_system.pl`; the motivating consumer is sketched
in `prototypes/mu_cosine/DESIGN_desugaring_to_prolog_goals.md` §4.1. A SPECIFICATION and an
IMPLEMENTATION_PLAN are deliberately deferred — see "Why no specification yet". This doc is the
**why**.*

**Status: philosophy note. Nothing is proposed for implementation, and no existing behaviour is
changed.** It is written from the `mu_cosine` lane about a **core** component that 67 targets
depend on, so it argues for prototyping outside core before touching it.

## The question

`template_system.pl` dispatches on exact string equality:

```prolog
case_matches(Value, Pattern) :- Value = Pattern.
```

and the file already names the intended extensions — glob, PCRE2 regex, or **structured
patterns** — noting the predicate "is factored out to make this extension straightforward."

Should it grow structured patterns, so a template can dispatch on the *shape* of a term rather
than on a spelling?

## What exists today, stated exactly

`=` is unification, but both operands are strings by the time it runs. `resolve_match/5`
converts them:

```prolog
member(Key=DictValue, Dict),
atom_string(DictValue, DictValueStr),        % stringified
member(case(CaseValue, CaseBody), Cases),
atom_string(CaseValue, CaseValueStr),        % stringified
case_matches(DictValueStr, CaseValueStr)     % two strings
```

So the system does exact string match, exactly as documented. The clause is well *positioned*
for the extension; it does not already perform it. That distinction matters because it is easy
to read "`=` is unification" and conclude the capability is present.

## The insight: matching is not the point — binding is

The obvious framing of this change is "let cases match structure." That framing undersells it
and, taken alone, produces something barely worth building.

Consider what happens after a successful match today. `resolve_match/5` returns the body raw,
and the caller re-expands it against the **original** dictionary:

```prolog
resolve_match(Key, Dict, Cases, Default, Rendered),
expand_match_blocks(Rendered, Dict, RenderedExpanded),
```

Under structural matching, `{{case substrate(C)}}` would bind `C` — and then **discard the
binding**, because no channel exists from the match into the body's scope. What remains is a
more permissive string compare.

The value is entirely in the other half:

```text
{{match arg}}{{case substrate(C)}}corpus is {{C}}{{/match}}
```

**Structural matching without binding propagation is not worth building.** That is the central
claim of this note, and it is why "extend `case_matches/2`" is the wrong way to scope the work:
the predicate is the small part.

## Why anyone wants this at all

Two consumers, neither of which exists yet.

**Constraints as dispatch keys.** If the process-expression surface desugars to Prolog goals,
constraints are discharged before ordinary body goals, which makes them the subset guaranteed
ground at template-selection time — the only safe things to dispatch on. Template selection then
becomes instance resolution: a template file is a predicate, `{{case}}` blocks are its clause
heads, the dict is the goal, and rendering is resolution.

**The typed AST as a transpiler target.** The project has 67 targets sharing
`common_generator.pl` and per-target mustache libraries under `templates/targets/<target>/`.
Once a language desugars to goals, its AST is one more target — the same introspection that
produces every other backend produces the structure an encoder embeds. Dispatching that on
strings works; dispatching on `substrate(C)` with `C` bound is what makes it pleasant.

Both are speculative today. That is the honest reason this is philosophy and not a plan.

## Expressive, not more powerful

Worth stating because the opposite is a tempting inference.

Structural dispatch replaces a fixed enumeration of spellings with open patterns. That is a real
gain in **expressiveness**. It does not let the system **prove** anything it could not prove
before:

- a closed-fact constraint such as `non_amplifying(Op)` remains a lookup, whatever the matcher;
- a numeric constraint such as `μ(A→C) ≤ min(links)` is not checkable by unification at all — it
  needs refinement or dependent types, which nothing here proposes.

Unification supplies the *dispatch*. Proof obligations still discharge once per operator, by
property test. A matcher that appears to promise more than that will be believed.

## Three costs, in descending order of how easily they are missed

**1. Case order becomes semantically significant.** Exact strings cannot overlap; structured
patterns can. Prolog commits to the first matching clause, and `resolve_match/5` uses `member/2`
under `->`, so first-match-wins is already the behaviour. Today that is harmless. With patterns
it means **reordering a template file can change output**, silently. Haskell and Mercury forbid
overlapping instances precisely to avoid this. The options are to require non-overlapping cases
and check it at load, or to accept ordering as part of the contract and version it — but not to
inherit it by accident.

**2. Binding scope and shadowing are undesigned.** If a match binds `C` and the surrounding dict
already has `C`, which wins? A scoped child dict is the obvious answer, and "obvious" is how
scope bugs get shipped. This needs deciding before anything depends on it.

**3. Patterns must be *read*, not split.** `{{case ...}}` bodies are currently produced by
splitting template text. A structured pattern has to be parsed as a term, which reopens
operator-table and variable-naming questions the string form never had to answer. This is the
substantial piece of work, and it is invisible from the one-line predicate.

## The constraint that shapes everything: 67 targets

`template_system.pl` is core. Every target depends on `case_matches/2` behaving as it does.

So any extension must be **opt-in and default-preserving**: exact-string semantics remain the
default, structural matching is requested explicitly per template or per call, and no existing
target changes behaviour. That is not a nicety — it is the difference between an additive change
and a migration across the whole backend surface.

## Why no specification yet

Writing a pattern language before knowing which patterns are needed is how a project acquires an
abstraction it cannot remove. Both motivating consumers are hypothetical; neither has produced a
single real dispatch pattern.

The cheaper order:

1. **Prototype outside core.** Build a structural dispatcher in `prototypes/mu_cosine/` against
   the AST target only, where nothing is sealed and the blast radius is one directory.
2. **Collect real patterns.** Let the consumer say what shapes it actually dispatches on.
3. **Then specify**, with evidence rather than speculation, and only then touch core.

This also preserves the most useful outcome: discovering that structural dispatch does *not*
earn its keep for the AST target, learned in a prototype instead of across 67 backends.

## Non-goals

- Changing the behaviour of any existing target.
- Regex or glob matching — separately listed in `template_system.pl`'s future extensions, and
  orthogonal to structure.
- Making templates identity-bearing. Note that if templates ever generate an AST rather than
  render text, they move from a *cache* key into an *identity* preimage; that escalation is
  discussed in `DESIGN_desugaring_to_prolog_goals.md` §4.1 and is not proposed here.
- Any claim that structural matching strengthens what the constraint system can prove.
