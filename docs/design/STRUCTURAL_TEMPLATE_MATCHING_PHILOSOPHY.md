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

## What it would look like

The block syntax does not change. `{{match}}`, `{{case}}`, `{{default}}`, `{{/match}}` and the
delimiters stay exactly as they are. Three things change: the case value becomes a **term**
rather than a tag, the dict value becomes a term, and the body may reference **variables bound by
the match**.

Today, from `templates/targets/rust_wam/time_builtin.rs.mustache` — the key is a plain tag
selecting which section of the file to emit:

```text
{{match part}}{{case helpers}}
    fn time_value_seconds(value: &Value) -> Option<i64> { ... }
```

Structural, minimally:

```text
{{match node}}
{{case substrate(C)}}  substrate over {{C}}
{{case judge(J)}}      judged by {{J}}
{{/match}}
```

The motivating case — goals to AST nodes:

```text
{{match goal}}
{{case has_type(X, substrate(C))}}
  TypeNode { term: {{X}}, kind: "substrate", corpus: {{C}} }
{{case non_amplifying(Op)}}
  ConstraintNode { kind: "non_amplifying", op: {{Op}} }
{{default}}
  /* unhandled goal: {{goal}} */
{{/match}}
```

That last block is unreachable with string matching: it would need one case per concrete
spelling — `has_type_x_substrate_pearltrees`, and so on for every corpus — which is the
fixed-enumeration problem the change exists to remove.

### The syntax similarity is itself the hazard

Note `{{C}}` above. It is **syntactically indistinguishable** from an ordinary interpolation.
Under the string parser `C` is merely a missing dict key, and the mustache rule for a missing key
is to render the **empty string**, not to raise.

So a structural template fed to the string parser does not fail. It emits a document with holes
where the bindings belong. Same file, same extension, same `{{...}}`, two parsers, and nothing
visible to say which applies. *Something* therefore has to mark the dialect.

### Three ways to mark it

1. **Distinct extension.** Strongest signal, visible in directory listings and editor
   configuration.
2. **First-line pragma.** `{{! dialect: pattern_stache }}` — mustache comments are already valid
   mustache and are ignored by stock renderers, so this is backward compatible and the file
   self-describes when opened. Invisible in a listing.
3. **Both.** Extension for tooling, pragma for self-description.

Either of the first two gives the same guarantee *by construction*: no existing file carries the
marker, so no existing file can reach the structural parser, whatever the configuration says.
That is strictly stronger than a `structural(true)` config option, which can be set at a call
site, inherited through `Config`, or defaulted globally, and then silently change how 67 targets
parse.

### Naming

The dialect name and the file extension are different things, and languages routinely separate
them (TypeScript/`.ts`, Markdown/`.md`). Recommendation:

| | |
|---|---|
| dialect | `pattern_stache` — pattern matching with mustache syntax |
| extension | `.stache` |
| pragma | `{{! dialect: pattern_stache }}` |

`pattern_stache` uses an underscore rather than a hyphen for a concrete reason: the pragma value
would be read by a Prolog loader, where `pattern_stache` is a valid unquoted atom while
`pattern-stache` parses as the compound `-(pattern, stache)` and would need quoting at every read
site.

`.stache` truncates from the front, so it reads as deliberate rather than as a typo — unlike a
`p`- or `s`- prefix on `mustache`. It keeps the lineage the dialect genuinely retains without
claiming to *be* mustache.

### The pragma is a versioned header, not just a dialect marker

The extension and the pragma answer different questions, so neither is redundant:

| marker | job |
|---|---|
| `.stache` | selects the **parser family** — this file is not string-match mustache |
| pragma | pins the **version** within that family |

Versioning matters here for the same reason it matters everywhere else in this project, which
already versions everything that can change output — `REGISTRY_VERSION`, `CONTRACT_VERSION`
(`pec-v2`), `VOCAB_VERSION` (`tok-v1`), `RENDERER_VERSION` (`"r2"`). A dialect version is the
same category of thing, and it becomes load-bearing under the escalation noted above: if
templates ever generate an AST rather than render text, extending the pattern language later
(guards, arithmetic, nested patterns) would otherwise silently change how existing files parse.

There is precedent in the outputs already. Every generated file opens with a provenance header:

```text
// Generated by UnifyWeaver WAM-to-Rust transpilation
// Date: {{date}}
```

The outputs self-describe; the templates should too.

**Prefer a term to a `key: value` string.** The loader is Prolog, so

```text
{{! dialect(pattern_stache, 1) }}
```

is read directly by `read_term_from_atom/3` with no bespoke parser, and extends without
re-parsing — `dialect(pattern_stache, 2, [guards])`. This is the same reasoning that chose the
underscore: the pragma is consumed by Prolog, so it should be a Prolog term. The cost is that it
reads a little more like code and less like prose than `{{! dialect: pattern_stache }}`.

Two rules follow, both cheap now and awkward later:

- **Fixed position.** Mustache comments are legal anywhere. If the pragma is load-bearing the
  loader must require it at a fixed position — the first non-empty line — rather than scanning
  for it. Otherwise a `{{! dialect(...) }}` occurring inside a case body could be taken for a
  header, or two occurrences could disagree.
- **An unknown version is an error, not a fallback.** A loader meeting a dialect version it does
  not implement refuses the file. It does not parse it with the newest available parser and hope.
  This is the same fail-closed posture as unknown `impl` values returning `no_candidates` rather
  than a default (`DESIGN_process_expression_patterns.md` §16.15).

Rejected, with reasons worth not re-deriving:

- **`.plt`** — the standard SWI-Prolog `plunit` test-file extension. No `.plt` files exist here
  yet only because 30 files declare tests inline with `begin_tests`; the collision becomes real
  the moment anyone splits tests out, in exactly the lane that would read these templates.
- **`.hbs`** — claims Handlebars compatibility the dialect would not have.
- **`.mst`, `.mus`** — concise, but they buy brevity with ambiguity in a name that is read in
  directory listings.
- **`.tache`** — one letter from `.cache`, in a system whose default `cache_dir` is
  `templates/cache` (created on demand by `save_template_to_cache_file/3`), so cached and
  templated paths already sit side by side.
- **`.pattern_stache`** as an extension — the name is good; paying fifteen characters for it in
  every filename is not.

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

## Open question: is `.stache` backward compatible with `.mustache`?

Deliberately unsettled, but the evidence is cheap to gather and is recorded here so the decision
is not made from intuition.

**Measurement.** 24 distinct `{{case}}` values exist across `templates/` and `src/`, after
discarding two scanning artifacts — the literal `...` and an empty value, both from comments in
`template_system.pl` that describe the syntax rather than use it:

| shape | count | fate when read as a term |
|---|---:|---|
| lowercase-atom (`helpers`, `lazy`, `v1`) | 22 | reads as the same atom — compatible |
| hyphenated (`wam-fsharp`, `wam-haskell`) | 2 | reads as the compound `-(wam, fsharp)` — **silent mismatch** |
| uppercase | 0 | — |

The two hyphenated values live in `template_system.pl`'s own self-test rather than in a
production template, so nothing ships broken today. But a hyphenated tag was a natural thing for
someone to write once, which means it is a natural thing to write again.

**The zero is the important number.** A case value beginning with an uppercase letter —
`{{case Helpers}}` — matches that literal string under the current parser. Read as a term it is a
**fresh variable, which unifies with anything**, so the first case swallows all input and every
later case becomes unreachable. Silent, total, and absent today only by chance.

**Quoting resolves both — the incompatibility is with *unquoted* reading, not with the values.**
Measured against SWI's reader:

```text
wam-fsharp         -> compound -/2
'wam-fsharp'       -> atom
Helpers            -> VARIABLE (unifies with anything)
'Helpers'          -> atom
helpers            -> atom
```

So `{{case 'wam-fsharp'}}` and `{{case 'Helpers'}}` would both be correct. Existing files write
these unquoted, which is why they would change meaning.

**The failure is a successful parse, not an error.** This is the part that constrains the design.
Neither `wam-fsharp` nor `Helpers` fails to read — each reads *into the wrong thing*. So the
obvious rescue, *attempt a term read and fall back to a string on failure*, cannot work: there is
no failure to catch. A silent successful misparse is undetectable by fallback, which rules out
the cheapest compatibility strategy before it is proposed.

**Put the decision on the extension, not on the header's presence.** If a `.stache` file
*without* a header meant mustache semantics, `.stache` files would carry two possible semantics
and the extension would no longer say which parser applies — reinstating the invisible-difference
hazard the extension exists to remove. The rule that avoids this:

| file | rule |
|---|---|
| `.mustache` | every case value is a literal; string-compared, exactly as today |
| `.stache` | header required; case values are read as terms. A headerless `.stache` is an error |

No file is then ambiguous, and neither extension needs to inspect the other's conventions.

### The migration checklist

Converting a `.mustache` file to `.stache` means adding the header and quoting any case value
that is not already a lowercase atom:

| shape | example | action | how it fails unquoted |
|---|---|---|---|
| lowercase atom | `{{case helpers}}` | none | — reads as the same atom |
| hyphenated | `{{case wam-fsharp}}` | quote | **silently** — reads as compound `-/2` |
| other operator chars | `{{case 3-way}}` | quote | **silently** — reads as compound `-/2` |
| uppercase-initial | `{{case Helpers}}` | quote | **silently and totally** — becomes a variable matching every input |
| contains a space | `{{case a b}}` | quote | loudly — the read raises |

The middle three are the dangerous rows: they parse, so no error surfaces. Only the space case
fails loudly, which means review — or the pre-processor below — has to catch the others rather
than relying on the reader to complain.

### A quote-inserting pre-processor, and its limit

That checklist is mechanical, so it can be automated — but only for files in which *every* case
value is a literal, which is exactly what a legacy `.mustache` file is. The rule "quote anything
that is not already a lowercase atom" is unambiguous there.

It cannot be applied inside a `.stache` file. Faced with `{{case substrate(C)}}` a pre-processor
has no way to tell a pattern it must leave alone from a literal tag it must quote; both are
merely "not a lowercase atom." So the tool is a **migration aid, not a compatibility layer**, in
one of two shapes:

- a **one-shot converter**, `.mustache` → `.stache`, quoting as it goes; or
- a **load-time shim for `.mustache` only**, auto-quoting every case so legacy files can be read
  through the pattern parser and one code path serves both.

Neither is urgent. Until such a tool exists, `.mustache` keeps its current string parser and the
two paths stay separate — which is the conservative default anyway, and makes "assume legacy" a
deferral with a stated precondition rather than a standing assumption.

**What remains open** is the framing, not really the mechanism:

- **(A) Require the header; a `.stache` file without one is an error.** Every case value is read
  as a term. Renaming a `.mustache` file costs one line plus quoting any hyphenated or
  uppercase tag; 22 of 24 work unchanged.
- **(C) Clean break — claim no compatibility at all.** Identical in practice to (A); differs only
  in what is promised.
- **(D) Mark the intent per case rather than per file.** A literal case stays a literal; a
  pattern is flagged where it is written:

  ```text
  {{case helpers}}          literal tag — string compare, unchanged
  {{case ?substrate(C)}}    pattern — read as a term
  ```

  (or a distinct keyword, `{{pcase substrate(C)}}`). This gives **backward compatibility with
  zero migration** — every existing unquoted value stays literal, hyphens included — and makes
  overlap detection finer-grained, since only marked cases can overlap. The cost is one more
  piece of syntax in a dialect whose appeal is that it stays mustache-shaped.

(A) and (C) put the decision at the file boundary; (D) puts it at the point of use. The
file-level options need a migration and buy uniformity; (D) needs none and buys precision. The
trade is genuine and is not resolved here.

Whichever is chosen, **do not promise superset semantics**. A superset promise is what would
invite someone to write `{{case Helpers}}` unquoted and lose an afternoon to a case that silently
swallows every input.

## Is this document enough to build from?

No — and that is deliberate. A philosophy note is enough to decide *whether* to proceed and to
scope a prototype. It is not enough to hand someone a build, because six things would have to be
invented rather than read:

1. the binding-propagation mechanism — scoped child dict, and how shadowing resolves;
2. the term-reading context for `{{case}}` bodies — operator table, variable naming, read options;
3. whether case overlap is detected, and if so at load or never;
4. determinism when a body fails after a case has committed;
5. loader dispatch — a new `try_source/4` strategy, or a branch inside `render_named_template/4`;
6. what the Dict contract becomes once values may be terms, and whether that breaks callers that
   currently pass atoms.

Those six are the contents of the SPECIFICATION, and per "Why no specification yet" they should
be *answered by a prototype* rather than guessed at in advance. The handoff-able unit today is
therefore a prototype in `prototypes/mu_cosine/` against a single consumer, reporting which
patterns it actually needed — not a change to core.

## Non-goals

- Changing the behaviour of any existing target.
- Regex or glob matching — separately listed in `template_system.pl`'s future extensions, and
  orthogonal to structure.
- Making templates identity-bearing. Note that if templates ever generate an AST rather than
  render text, they move from a *cache* key into an *identity* preimage; that escalation is
  discussed in `DESIGN_desugaring_to_prolog_goals.md` §4.1 and is not proposed here.
- Any claim that structural matching strengthens what the constraint system can prove.
