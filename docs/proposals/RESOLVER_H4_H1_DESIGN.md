<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve: H4 + H1 design — active same-state cycle closure and version-only backtracking in `resolver.pl`

> Design round on the D64 exploration (`RESOLVER_H4_H1_EXPLORATION.md`).
> Design only: no repository code changes in this round. The Prolog below
> was prototyped in a scratch copy of `examples/pkg_resolver/resolver.pl`
> and every claim marked **[measured]** was run on that prototype, on SWI
> and on a scratch wamjs build of it; claims marked **[on paper]** were not
> executed. The prototype is not committed; the implementer re-derives it
> from §1, §2 and §4, which are complete listings.
>
> Inputs, read in full: the exploration; `resolver.pl` at `8d0ed83`;
> `RESOLVER_PRUNING_DESIGN.md` §1.3, §2 G4, §3.2, §4, §7;
> `test_pruning_probes.pl` (CE1–CE6, A1–A11); ledger rows D44–D64.

## 0. Decisions at a glance

| Question the brief leaves to this round | Decision |
|---|---|
| Reference semantics for H4 | **Approach 3** (active same-state cycle closure), exactly as §1. Approach 4 documented as fallback in §6.5, not needed on any evidence gathered here. |
| How the generation and marks are carried | One threaded term `st(Gen, Active)` as a sixth argument of the search loop; `resolve_pending/5` stays as the public-internal wrapper. Expansion completion is a `done(Pkg, Ver, Gen)` sentinel in the pending list. No assert/retract. |
| Where the generation advances | Exactly one site: the `from_catalog` arm of `resolve_pending`, on every `[Pkg-Ver|Acc]` cons — real picks and provider picks alike, so the H2 second-version shape advances it for free. |
| Validate-then-suppress ordering | Closure test sits *after* `pick_need/8`, which is where the incoming request is validated (`satisfies(BV, C)` / `provides_sat`). A closing edge that fails validation fails the branch before the closure test is reached. |
| H1 repair | **Version-only** restoration (§2). Held-name and held-provider arms untouched and still committed; providers still only when no real candidate. |
| Order of landing | **H4, then H1**, two commits, one approved re-baseline (§5.3). The H1-only cell of the matrix turns finite failures into divergence **[measured]** — the concrete reason H4 must go first. |
| explain_blocked | Explanations describe the **preferred candidate's blockage** (§4). The only code change is ordering: report a repeated name's ceiling before suppressing its walk. |
| Store adapter | `resolver_store.pl` carries a textual mirror of the same loop and must take both edits in the same commits (§1.8). |

**Evidence summary [measured].** On the prototype: the 50-scenario contract
corpus is green unchanged; probes CE2/CE5/CE6 go red exactly as this
design says they must and the other 23 probes stay green; the 2,600-case
seeded SWI oracle is **byte-identical** with H4 alone and differs on
**exactly 11 lines** with H4+H1, all eleven `{"fail":true}` → `{"ok": …}`
on `resolve_layered`/`layer_closure`; the wamjs build of the prototype
passes the corpus 51/51 and the 2,600-case differential at 0 divergences
against the prototype oracle, plus 11 hand-written cyclic/H1 cases at 0
divergences. SWI inference overhead on the 2,600 cases: +0.03 % (H4),
+0.06 % (H4+H1).

---

## 1. Chosen approach, concretely: approach 3

### 1.1 State: one threaded term, one wrapper

The search loop gains a sixth argument `St = st(Gen, Active)`:

- `Gen` — the **branch-local selection generation**, a non-negative
  integer, `0` at entry, incremented on every catalog insertion.
- `Active` — the **stack of open held expansions**, each `a(Pkg, Ver, Gen)`,
  where `Gen` is the generation *at the time the expansion was opened*.

Because `St` is an ordinary term passed down the recursion and never
asserted, backtracking restores marks and generation together: the
binding of the `St` argument in a failed branch is simply undone. This is
the whole of "restored on backtracking" — there is no code for it.

`resolve_pending/5` keeps its name and arity as a one-clause wrapper so
the probe file's internal calls (`a3_resolve_agrees_and_icat_never_escapes`
calls `resolver:resolve_pending/5`) stay valid:

```prolog
resolve_pending(Mode, Cat, Pending, Acc, Sel) :-
    resolve_pending(Mode, Cat, Pending, Acc, st(0, []), Sel).
```

`resolve/3` and `resolve_layered/3` are unchanged (they call the wrapper).
`resolve_alternatives/6` becomes `/7`, threading `St` unchanged.

### 1.2 The loop (complete listing; replaces `resolve_pending/5` and `resolve_alternatives/6`)

```prolog
resolve_pending(_Mode, _Cat, [], Acc, _St, Acc).
resolve_pending(Mode, Cat, [Item|Rest], Acc, St, Sel) :-
    (   Item = done(Pkg, Ver, Gen)
    ->  % a held expansion completed: pop its mark. Strict LIFO (§1.4), so
        % the mark is the head; this unification is an internal assertion.
        St = st(GenNow, [a(Pkg, Ver, Gen)|Active1]),
        resolve_pending(Mode, Cat, Rest, Acc, st(GenNow, Active1), Sel)
    ;   Item = req(Name, C),
        (   Name = alternatives(Alts)
        ->  resolve_alternatives(Mode, Cat, Alts, Rest, Acc, St, Sel)
        ;   selected_ver(Acc, Name, Ver)
        ->  % commit: a second version of the same name is never added (diamond)
            satisfies(Ver, C),
            resolve_pending(Mode, Cat, Rest, Acc, St, Sel)
        ;   already_provided(Cat, Acc, Name, C)
        ->  resolve_pending(Mode, Cat, Rest, Acc, St, Sel)
        ;   pick_need(Mode, Cat, Name, C, Acc, Pkg, Ver, Origin),
            % G2 comment from D59 stays here verbatim.
            (   Origin = from_base
            ->  St = st(Gen, Active),
                (   active_member(Active, Pkg, Ver, Gen)
                ->  % H4, approach 3: same-state cycle closure. The incoming
                    % request was already validated by pick_need/8 (held
                    % version satisfies C, or the held provider provides
                    % Name at C). Omit only the repeated expansion; every
                    % remaining obligation in Rest is kept.
                    resolve_pending(Mode, Cat, Rest, Acc, St, Sel)
                ;   collect_deps(Cat, Pkg, Ver, DepReqs),
                    append(DepReqs, [done(Pkg, Ver, Gen)|Rest], More),
                    resolve_pending(Mode, Cat, More, Acc,
                                    st(Gen, [a(Pkg, Ver, Gen)|Active]), Sel)
                )
            ;   no_acc_conflicts(Cat, Pkg, Ver, Acc),
                collect_deps(Cat, Pkg, Ver, DepReqs),
                append(DepReqs, Rest, More),
                St = st(Gen, Active),
                Gen1 is Gen + 1,          % the ONLY site that advances Gen
                resolve_pending(Mode, Cat, More, [Pkg-Ver|Acc],
                                st(Gen1, Active), Sel)
            )
        )
    ).

% Same-generation marks are a prefix of the stack (§1.4), so the scan may
% stop at the first older mark. Integer `<`/`=:=`-class comparison and
% `==` on atoms/versions are both already exercised by resolver.pl
% (version_lt/2, same_key/4, seen_name/2).
active_member([a(P, V, G)|Rest], Pkg, Ver, Gen) :-
    (   G < Gen
    ->  fail
    ;   P == Pkg, V == Ver
    ->  true
    ;   active_member(Rest, Pkg, Ver, Gen)
    ).

resolve_alternatives(Mode, Cat, Alts, Rest, Acc, St, Sel) :-
    (   first_alt_already(Mode, Cat, Acc, Alts)
    ->  resolve_pending(Mode, Cat, Rest, Acc, St, Sel)
    ;   member(dep(N, C), Alts),
        resolve_pending(Mode, Cat, [req(N, C)|Rest], Acc, St, Sel)
    ).
```

Everything else in the file (`pick_need/8` apart from §2, `collect_deps/4`,
`selected_ver/3`, `already_provided/4`, `first_alt_already/4`,
`no_acc_conflicts/4`, the index) is untouched.

Design notes on the listing:

- **Classic mode is structurally unchanged.** No `from_base` pick exists
  in classic mode, so no `done/3` item is ever pushed and `Active` stays
  `[]`; `Gen` advances but is never read. The only added work per pending
  item is the failed `Item = done(_,_,_)` unification. The mode argument
  is not consulted anywhere new.
- **`done/3` is a pending-list item, not a request.** It is dispatched
  first in the if-then-else so the `req/2` body is not nested inside a
  then-branch. G4 (unshipped) would have to skip `done/3` items in its
  `no_doomed_req/3` scan; it already skips non-atoms via `atom(D)`.
- **No new cuts** (pruning-design invariant 6): if-then-else only.
- **`Gen1 is Gen + 1`** is the arithmetic the file already uses
  (`key_dep_rows/3`, `long_enough/2`).

### 1.3 Where the generation advances

Exactly one site: the `from_catalog` arm, immediately before the recursive
call that conses `[Pkg-Ver|Acc]`. This covers, without special-casing:

- a real pick (`Pkg = Name`);
- a provider pick (`Pkg \== Name`), including the **H2 shape** — a second
  version of a name already in `Acc` arriving through `provider_candidate/5`
  (CE3's `mawk-2.0` next to `mawk-1.0`). The exploration's warning that
  "counting only newly-selected *names* is insufficient" is met because the
  counter counts *conses onto `Acc`*, not names.

`from_base` never advances `Gen` (nothing is inserted), and neither do the
`selected_ver`, `already_provided`, satisfied-alternative and closure paths.

### 1.4 Marks are a strict LIFO stack; same-generation marks are a prefix

**Claim (LIFO).** At any point of a forward branch, `Active` is exactly
the chain of `from_base` expansions whose `done/3` sentinel has not yet
been consumed, newest first; and every `done(Pkg, Ver, Gen)` item that is
consumed finds `a(Pkg, Ver, Gen)` at the head of `Active`.

*Why.* An expansion of `Pkg-Ver` pushes `DepReqs ++ [done(Pkg,Ver,Gen)|Rest]`.
Everything any of those `DepReqs` expands is pushed in front of that
`done`, together with its own `done`, so nested sentinels are consumed
before the enclosing one; siblings are consumed in order. Backtracking
undoes pushes and pops together because both the pending list and `St`
are bindings of the same clause body. Hence the head-unification in the
`done` branch is an assertion that cannot fail on a correct
implementation — and a reviewer can make it loud (§7, R2).

**Corollary (prefix).** `Gen` is monotone along a branch and each mark
records the generation at its push, so marks with `G =:= Gen` are the
newest ones: a prefix of the stack. `active_member/4` stops at the first
older mark. The scan is therefore O(number of held expansions opened
since the last insertion), never O(|Acc|), and never O(total held
expansions on the path).

### 1.5 Restoration on backtracking — the single-term argument

The exploration requires that "backtracking must restore marks AND
generation together". With one term `st(Gen, Active)` bound per recursion
step this is automatic. The design deliberately rejects the two shapes a
"small" implementation might reach for:

- a global/asserted counter with threaded marks: after a failed insertion
  the counter has moved on while marks have not, so a legitimately
  same-state re-request is not closed and is expanded once more (a
  replay, not a hang — but an unbounded chain of failed insertions inside
  a cycle makes the replays unbounded);
- asserted marks with a threaded counter: a mark from a failed sibling
  branch survives and closes an expansion in the next branch, which is
  *unsound* — probe `stale_mark_would_be_unsound` (§6.1) succeeds on such
  an implementation and must fail on the correct one **[measured on the
  correct one: fails; the construction is such that a stale `b` mark from
  the failed `x-2` branch would hide `z`'s ceiling in the `x-1` branch]**.

### 1.6 Ordering: validate first, then suppress — the exact site

The closure test is the *first* goal of the `from_base` arm, and the
`from_base` arm is only entered after `pick_need/8` has succeeded. For a
held name `pick_need(layered, …)` succeeds only through
`base_ver(Cat, Name, BV), satisfies(BV, C)`; for a held provider only
through `layer_provider/5`, i.e. `provides_sat(Cat, Pkg, Ver, Name, C)`.
So the sequence for an incoming `req(Name, C)` is, in this order:

1. `selected_ver(Acc, Name, _)` — the diamond commit, unchanged;
2. `already_provided(Cat, Acc, Name, C)` — unchanged;
3. `pick_need/8` — **request validation against the held ceiling**;
4. only then `active_member/4` — closure.

Trace for the exploration's unsoundness witness, held `a-1`, `b-1`,
`a → b any`, `b → a = 2`, request `[a]`:

```
pending [req(a,any)]                  St = st(0, [])
  a: not selected, not provided; pick_need → base_ver a-1, satisfies(1, any) ✓ → from_base
     active_member([], a, 1, 0) fails → expand:  pending [req(b,any), done(a,1,0)]   St = st(0,[a(a,1,0)])
  b: pick_need → base_ver b-1 ✓ → from_base; not active → expand:
     pending [req(a,eq(2)), done(b,1,0), done(a,1,0)]   St = st(0,[a(b,1,0),a(a,1,0)])
  a=2: not selected, not provided; pick_need → base_ver a-1, satisfies(1, eq(2)) FAILS
     → the branch fails at step 3; step 4 is never reached.
resolve_layered fails.                                   [measured: no, on all four cells]
```

Putting the closure before `pick_need/8` (or keying it on the *name*
rather than on the validated `Pkg-Ver`) would close this request and
answer `[]` — wrong, and the reviewer's R4 in §7 pins it.

### 1.7 Runtime portability

Per D60, the constructs known safe on all five WAM runtimes are structural
unification, `==`, `sort/2`, if-then-else; `functor/3` and `memberchk/2`
are not. The listing uses, beyond the file's existing vocabulary:

| Construct | Already exercised in `resolver.pl`? | Used here for |
|---|---|---|
| structural unification against `done/3`, `st/2`, `a/3` | yes (every accessor) | dispatch and state |
| `is/2` with `+ 1` | yes (`key_dep_rows/3`) | `Gen1` |
| integer `<` | yes (`version_lt/2`, `long_enough/2`) | prefix stop |
| `==` on atoms and version terms | yes (`seen_name/2`, `same_key/4`) | mark match |
| a `member/2` choice point inside a **then**-branch (§2) | then-branch nondeterminism exists (`Name = alternatives(_) -> resolve_alternatives(...)`); else-branch `member/2` exists (`resolve_alternatives/6`) | H1 |

**[measured]** The scratch wamjs build of the prototype (the leg that
silently broke on `functor/3` in D60): corpus 51/51 vs SWI; the 2,600-case
differential at **0 divergences** against the prototype's SWI oracle,
which already contains the 11 H1 answer changes; the 11 hand-written
cyclic/H1 cases (§6.1, JSON form) at **0 divergences**. Go, Rust and
ClojureScript were not built here; §6.3 says what they must show.

No hole was found on paper or in the wamjs run; approach 4 therefore
stays a fallback (§6.5), not the primary.

### 1.8 The store adapter carries a mirror of this loop

`resolver_store.pl` has `resolve_pending_store/5`, `resolve_alternatives_store`,
`pick_need_store/7` "kept textually parallel" to the term loop (D59
mirrored G2 there). It carries H4 and H1 too. Both §1 and §2 edits must be
applied to the mirror in the same commits, with the same names
(`st/2`, `done/3`, `a/3`, `active_member/4`, `candidate_versions_store/4`),
otherwise the 503-case store differential and the term differential
disagree on exactly the changed cases. The store-vs-term identity check
(same catalog through both adapters) is part of §5.3 step 4.

### 1.9 Cost

Per pending item: one failing unification. Per held expansion: one
`append/3` cell, one `Active` cons, one `done` item, one `active_member`
scan bounded by the same-generation prefix. Per insertion: one `is/2`.
**[measured]** SWI, 2,600 cases: 29,492,021 → 29,499,877 inferences (H4),
→ 29,509,922 (H4+H1). The wamjs 5k-package scale run (B3) has 2.3× headroom
under its 2,000,000-step cap (854,887 steps in D59); the per-item cost is
a handful of instructions, so no cap regression is expected — to be
confirmed by the B3 rerun in §5.3 step 5, since that is a measurement,
not a proof.

---

## 2. H1 repair, concretely: version-only restoration

### 2.1 The rewrite

Today (`pick_need/8`, third clause):

```prolog
pick_need(layered, Cat, Name, C, _Acc, Pkg, Ver, Origin) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C), Pkg = Name, Ver = BV, Origin = from_base
    ;   layer_provider(Cat, Name, C, Pkg, Ver)
    ->  Origin = from_base
    ;   candidates_high_first(Cat, Name, C, Ver)      % committed by ->
    ->  Pkg = Name, Origin = from_catalog
    ;   provider_candidate(Cat, Name, C, Pkg, Ver), Origin = from_catalog
    ).
```

After (only the third arm changes):

```prolog
pick_need(layered, Cat, Name, C, _Acc, Pkg, Ver, Origin) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C),
        Pkg = Name,
        Ver = BV,
        Origin = from_base
    ;   layer_provider(Cat, Name, C, Pkg, Ver)
    ->  Origin = from_base
    ;   candidate_versions(Cat, Name, C, Desc),
        Desc = [_|_]
    ->  member(Ver, Desc),          % H1: descending versions on backtracking
        Pkg = Name,
        Origin = from_catalog
    ;   provider_candidate(Cat, Name, C, Pkg, Ver),
        Origin = from_catalog
    ).
```

with the candidate list factored out of `candidates_high_first/4` so there
is one definition of the order:

```prolog
% Highest version first. Excluded names produce no candidates.
candidates_high_first(Cat, Name, C, Ver) :-
    candidate_versions(Cat, Name, C, Desc),
    member(Ver, Desc).

% The descending candidate list itself (det; [] when excluded or none).
candidate_versions(Cat, Name, C, Desc) :-
    (   excluded_name(Cat, Name)
    ->  Desc = []
    ;   matching_versions_in(Cat, Name, C, Vs),
        sort_versions_desc(Vs, Desc)
    ).
```

`candidates_high_first/4` (classic `pick_need`, `pick/7`, `layered_walk_ver/4`,
`pick_repair/4`) yields the same solution sequence as before: `\+ excluded`
then `matching_versions_in` then `sort_versions_desc` then `member`, in
that order, with the excluded case now failing through `member(_, [])`
instead of through `\+`. **[measured]** the 2,600-case classic-and-layered
oracle is byte-identical under H4 and changes only the 11 H1 lines under
H4+H1, so no classic answer moved through this refactor.

### 2.2 What the rewrite does and does not uncommit

- **Held ceilings stay committed.** The `base_ver` arm is the first
  condition and is unchanged: a held name never reaches the version
  enumeration, and if its held version fails `C` the whole `pick_need`
  fails — the `->` still commits. Probe `a9_pkg_held_virtual_provider_not_selected_fails`
  (held `foo-1.0`, request `foo >= 2.0`, `foo-2.0` in the catalog) still
  fails **[measured]** and becomes this round's *commitment witness*.
- **Loaded-provider priority stays committed.** The `layer_provider` arm
  is unchanged and still committed by its `->`.
- **Providers only when no real candidate.** The condition is the
  *non-emptiness of the real candidate list*, not the success of a pick;
  once real candidates exist the else-arm (`provider_candidate`) is never
  reached, even after all real versions have failed downstream. This is
  the version-only choice. Witness **[measured]**: real `x-1 → missing`,
  `p provides x`, request `app → x`: `resolve` (classic, clause 2 fallback)
  answers `[app-1, p-1]`; `resolve_layered` fails, before and after this
  round. The classic/layered difference on this shape is *documented*, not
  new, and the brief chose not to widen it.
- **What is uncommitted**: exactly the descending real versions of a
  non-held name. The first solution is still the highest version; lower
  ones are reached only after the branch under a higher one fails.

### 2.3 A choice point in a then-branch

`member(Ver, Desc)` lives in the *then* part of an if-then-else. Only the
condition is committed; then-branch choice points survive in ISO Prolog
and in every WAM leg's barrier model (D45 §9: the ITE cuts the
*condition's* choice points; D49/D50/D53 fixed lowered-ITE variants of
exactly this). The corpus already relies on then-branch nondeterminism
(`Name = alternatives(_) -> resolve_alternatives(...)`, whose body
enumerates). §7 R7 asks each leg to show all solutions of a then-branch
`member/2` in order, as a stand-alone probe modeled on D63's enumerator
probe, so a regression here is caught before the resolver corpus is.

---

## 3. Interaction proof obligation: H4 + H1 together

Notation: `M` = number of distinct `package(Name, Ver)` rows in the
catalog; `H` = number of distinct held `Pkg-Ver` pairs reachable through
`base_ver/3` or `layer_provider/5`; `D` = the longest dependency list;
`A` = the longest alternatives group; `P` = the longest provides list.
All finite for a ground, well-formed catalog (the precondition §7.4 of the
pruning design asks to state explicitly; this design inherits it).

**Lemma 1 (insertions are bounded).** Along any forward branch (no
backtracking), at most `M` `from_catalog` insertions occur, and they
insert pairwise-distinct `Pkg-Ver`.
*Proof.* A real pick of `Name` happens only if `selected_ver(Acc, Name, _)`
failed, i.e. no version of `Name` is in `Acc`; so each real pick adds a
name not yet present. A provider pick of `Pkg-Ver` for `req(Name, C)`
happens only if `already_provided/4` failed, i.e. no pair in `Acc`
provides `Name` at `C` through *any* provides row; since `provider_candidate`
selects `Pkg-Ver` through a provides row that satisfies `C`, if `Pkg-Ver`
were already in `Acc` that same row would have made `already_provided`
succeed. Hence no pair is inserted twice. ∎

**Lemma 2 (same-generation held depth is bounded by H).** Between two
consecutive insertions, the `Active` marks pushed all carry the current
`Gen`, form the prefix of the stack (§1.4), and are pairwise distinct
`Pkg-Ver`: a `from_base` pick of a pair already in the prefix is closed,
not expanded. So the nesting depth of open held expansions at one
generation is at most `H`. ∎

**Lemma 3 (every forward branch is finite).** The pending list is consumed
left to right; each consumed item either pushes nothing, or pushes at most
`D + 1` items (a `from_base` expansion plus its sentinel), or inserts and
pushes at most `D` items, or (alternatives) pushes one item. Between
insertions, the pushed items form a tree of held expansions whose depth
is at most `H` (Lemma 2) and whose branching is at most `D`, hence at most
`D^H` expansions, each consuming finitely many items. By Lemma 1 there are
at most `M` insertion phases. So a forward branch consumes finitely many
items. ∎

**Theorem (the combined search terminates).** `resolve_pending/6` under
§1 + §2 terminates on every ground, well-formed catalog and request list,
in both modes.
*Proof.* The search is a depth-first traversal of a tree whose nodes are
choice points and whose edges are forward steps. Every choice point has
finitely many alternatives: `member(Ver, Desc)` (≤ `M`), `provider_candidate`
(≤ `P`), `member(dep(N,C), Alts)` (≤ `A`), classic `pick_need`'s two
clauses, `no_acc_conflicts` (det), everything else det or a test. Every
root-to-leaf path is a forward branch and is finite by Lemma 3 —
crucially, Lemma 3 did not depend on *which* alternative was taken at any
choice point, only on the state carried into the branch, and each branch
starts from the exact `st(Gen, Active)` bound at its choice point (§1.5).
A finitely-branching tree with no infinite path is finite (König), so the
traversal visits finitely many nodes, each with finite work (`collect_deps`,
`append`, list scans, `sort`). ∎

**Where a naive combination loops, and why this one does not.** H1 alone
opens branches under lower versions; a lower version can lead into a held
cycle that the highest version did not. Without H4 that branch diverges
(the H1-only matrix cell, §6.2, turns two *finite failures* of the
baseline into a timeout and a stack overflow **[measured]**). With H4 the
cycle is closed at its first same-state repetition inside that branch
(Lemma 2 applies per branch because marks and generation are branch-local).
The remaining naive shapes — global counter, asserted marks — are exactly
the two rejected in §1.5.

**Preservation theorem (H4 changes only branches on which the baseline
diverges).** Let the baseline be `resolver.pl` at `8d0ed83` (H1 still
committed). Along any baseline DFS path, if a `from_base` expansion of
`Pkg-Ver` is re-requested while `Pkg-Ver`'s earlier expansion is still
open and `Acc` is unchanged, the baseline diverges on that path.
*Proof sketch.* The earlier expansion pushed `DepReqs = collect_deps(Pkg, Ver)`
(a function of `Cat` alone); processing those items from `Acc` reached the
re-request without any insertion and without failing, taking first
choices everywhere (the path is a DFS prefix). The baseline re-expands
the same `DepReqs` from the same `Acc`; the steps taken depend only on
`Acc`, `Cat`, and the item sequence (the tail `Rest` is not consulted
until `DepReqs` is exhausted, and if it were, the earlier expansion's
sentinel would have been consumed, i.e. it would not be *open*). So the
same first choices are taken and the same re-request is reached with
`Acc` unchanged, again; inductively the leftmost branch is infinite, and a
DFS never returns from it. ∎
Consequently approach 3, which takes identical steps until such a
repetition, produces the baseline's first answer whenever the baseline
has one, and turns baseline divergences into finite outcomes — nothing
else. **[measured]** H4-only oracle byte-identical on 2,600 cases;
`ce2_ds1` unchanged; every changed scenario in §6.2's H4 column was a
timeout or resource exhaustion in the baseline column.

**Ordering rationale (H4 first).**
1. H1-only introduces failure→divergence on real shapes **[measured]**
   (`lower_version_enters_cycle`, `rollback_marks_with_generation`), so an
   H1-first stage cannot even be baselined with the current harness (no
   timeout notion, §7.4 of the pruning design).
2. H4-only has a clean inventory (divergence→finite only) and is checked
   by a byte-identity diff plus a list of formerly-divergent cases.
3. H1's inventory (§5.2) is then measured against a *terminating* oracle.
4. The exploration's own point: H1 cannot backtrack past a nonterminating
   highest candidate (`needs_lower_retry_cyclic`: neither/H1-only diverge,
   H4-only fails, both succeed **[measured]**).

---

## 4. `explain_blocked` decision

**Decision: explanations describe the preferred candidate's blockage**,
i.e. the ceilings met on the walk that follows the resolver's *first*
choices (highest version, real before provider, held used in place) —
not the absence of any solution. Justification:

- That is what the traversal has always computed. `layered_walk_ver/4`
  commits to the highest version; `blocked_from/4` never establishes
  unsatisfiability (the exploration says so; H3/CE4 shows a non-empty
  explanation next to a successful resolve today). "Absence of a solution"
  would need a full search, would change every enumeration's order and
  multiplicity, and would re-open G4's proof territory; out of scope.
- The user-facing consumer (`pkg why-blocked`, D47) asks "why is the
  thing I would get blocked?" — a diagnostic of the frozen base, which is
  exactly the preferred path.
- After H1, `resolve_layered` may succeed via `x-1` while the explanation
  reports `x-2`'s ceiling. That is the H3 class of disagreement, already
  accepted and probe-pinned; this round widens neither its definition nor
  its probe set beyond E3 below, and documents it in the README's hazard
  list as the *intended* reading rather than a hazard.

**The one code change: validate before suppressing, in both explain
walkers.** Today a repeated name on the path is dropped *before* its new
constraint is checked, which is why held `a-1 → b-1 → a = 2` explains as
`[]`. The path-local `Seen` keeps its job (stop the *walk* on a repeated
name, which is what bounds the traversal); it stops gating the *ceiling
reports*.

`blocked_from/4`: drop `\+ seen_name(Seen, Name)` from the `base_has`
clause and the `providers` clause; keep it on the walk clause:

```prolog
blocked_from(Cat, req(Name, C), _Seen, Blocked) :-
    base_ver(Cat, Name, BV),
    \+ satisfies(BV, C),
    Blocked = blocked(Name, needs(C), base_has(BV)).
blocked_from(Cat, req(Name, C), _Seen, Blocked) :-
    virtual_provider_ceilings(Cat, Name, C, Reasons),
    Reasons \== [],
    Blocked = blocked(Name, needs(C), providers(Reasons)).
blocked_from(Cat, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),               % the walk is what Seen bounds
    walk_pkg_for_blocked(Cat, Name, C, Pkg, Ver),
    collect_deps(Cat, Pkg, Ver, DepReqs),
    member(Dep, DepReqs),
    blocked_from(Cat, Dep, [Name|Seen], Blocked).
```

`blocked_acc/5`: delete the first clause (`atom(Name), seen_name(Seen, Name), !`)
and move the seen test to the walk half of the general clause; the
alternatives clause (with its existing cut) stays first:

```prolog
blocked_acc(Cat, req(alternatives(Alts), _), Seen, Acc0, Acc) :-
    !,
    alt_reasons(Cat, Alts, Seen, Rs),
    Acc = [blocked(alternatives(Rs))|Acc0].
blocked_acc(Cat, req(Name, C), Seen, Acc0, Acc) :-
    (   base_ver(Cat, Name, BV),
        \+ satisfies(BV, C)
    ->  Acc1 = [blocked(Name, needs(C), base_has(BV))|Acc0]
    ;   virtual_provider_ceilings(Cat, Name, C, Reasons),
        Reasons \== []
    ->  Acc1 = [blocked(Name, needs(C), providers(Reasons))|Acc0]
    ;   Acc1 = Acc0
    ),
    (   seen_name(Seen, Name)
    ->  Acc = Acc1                          % repeated name: no second walk
    ;   walk_pkg_for_blocked(Cat, Name, C, Pkg, Ver)
    ->  collect_deps(Cat, Pkg, Ver, DepReqs),
        blocked_acc_list(Cat, DepReqs, [Name|Seen], Acc1, Acc)
    ;   Acc = Acc1
    ).
```

Termination of the walkers is unchanged: every recursive call still
extends `Seen` with a name not in it, so a path has at most one walk per
distinct name. On any catalog where no name repeats on any walk path
(every existing corpus and differential explain case), `seen_name` never
succeeds and both walkers take exactly the old steps in the old order:
enumeration order and multiplicity preserved **[measured: the 2,600-case
oracle's `explain_blocked` lines are unchanged; the two-path duplicate
probe `explain_two_paths_multiplicity` still yields the reason twice]**.
Where a name does repeat, the change adds reasons (a repeated name's
failing ceiling), never removes one, and a satisfied cycle stays
unreported.

**New explain probes** (all **[measured]** on the prototype):

| Probe | Catalog | Asserts |
|---|---|---|
| E1 `explain_closing_edge` | held `a-1 → b-1 → a = 2` | `explain_blocked_list(Cat, a, L)`, `L == [blocked(a, needs(eq(2)), base_has(1))]`; `findall` over `explain_blocked/3` gives the same single reason; `resolve_layered(Cat,[a],_)` fails |
| E2 `explain_satisfied_cycle` | CE5's `a ↔ b` held, all `any` | `explain_blocked/3` has no solution; list is `[]`; `resolve_layered` answers `[]` — a satisfied cycle is never "blocked" |
| E3 `explain_preferred_candidate` | `app → x any`; `x-2 → h ≥ 2`, `h` held `1`; `x-1` no deps | `resolve_layered` answers `[app-1, x-1]` (post-H1) **and** `explain_blocked_list` answers `[blocked(h, needs(gte(2)), base_has(1))]` — success and explanation are separate observables, by design |
| E4 `explain_two_paths_multiplicity` | `app → [c, b]`, both `→ h ≥ 2`, `h` held `1` | `findall` yields the reason exactly twice, as before |

---

## 5. Invariants and the re-baseline plan

### 5.1 Invariants (every implementation of this design must satisfy)

- **I1 Classic mode byte-identical.** `resolve/3` and every classic query
  on every corpus scenario and every seeded differential case: identical
  oracle bytes before and after both stages.
- **I2 Layered non-affected cases byte-identical.** Every layered case
  whose baseline result is finite and reached without backtracking into
  a `from_catalog` version choice is byte-identical after both stages.
  Operationally: the old-vs-new diff of the oracle must consist *only* of
  lines classified by §5.2, and the H4-stage diff must be empty except on
  cases whose baseline run timed out.
- **I3 No success→failure, ever.** Neither stage may turn an `{"ok": …}`
  into `{"fail": true}` or a timeout.
- **I4 Held commitment.** `a9_pkg_held_virtual_provider_not_selected_fails`
  and `held_ceiling_committed` fail before and after (the H1 arm never
  reaches a held name).
- **I5 Explanations preserved off-cycle.** Every existing explain case
  (corpus + differential) identical in content, order, multiplicity.
- **I6 Purity.** No assert/retract, no global counters, no new cuts; the
  `icat/3` wrapper still never escapes and is still rejected as input.
- **I7 Leg parity.** Each leg built from the new `resolver.pl` matches the
  *new* SWI oracle at 0 divergences on corpus, term differential and (for
  wamjs) store differential; the §9 cut probes stay green.
- **I8 Term/store agreement.** `resolver.pl` and `resolver_store.pl`
  answer identically on every shared catalog, including the §6.1 cyclic
  set encoded for the store adapter.
- **I9 Resource exhaustion is not an answer.** The SWI runner reports
  `{"timeout": true}` distinctly from `{"fail": true}`; a leg that hits a
  step cap reports it distinctly too (H6 for wamjs must at least mark it).
  The four-way matrix is meaningless without this.

### 5.2 Answer-change inventory (intentional changes)

| Kind | Stage | Where it comes from | Witnesses **[measured]** |
|---|---|---|---|
| divergence → success | H4 | a held cycle whose closure and remaining obligations are all satisfiable | CE5 (`[]`), `cycle_order_y_before_x`, `cycle_in_alt_satisfiable`, `held_provider_cycle`, `distinct_held_versions`, `satisfiable_cyclic_highest`, `named_layer_cycle`, `cycle_then_doomed_alt` (the §7.1 shape: divergence → success *through the next alternative*) |
| divergence → failure | H4 | a held cycle closed, then an unsatisfiable obligation with no alternative | `needs_lower_retry_cyclic` under H4-only (`no`); becomes success once H1 lands |
| failure → success | H1 | a downstream dead end under the highest version, satisfiable under a lower one | CE6 layered, CE2 modified catalog, `lower_version_enters_cycle` (needs both), `h2_gen_advance_order`, **11 seeded differential cases**: `g413`, `g701`, `g921`, `g1193`, `g1251`, `g1593`, `g1653`, `g1683`, `g1771`, `p3g31`, `p3g183` (5 × `resolve_layered`, 6 × `layer_closure`; two in the `deb/3` world) |
| success → different success | H1 | a lower version rescues an *earlier* alternative | `alt_rescued_by_lower`: `[app, d-1, q, y]` → `[app, d-1, p, x-1, y]`. Not present in the 2,600 seeds. |
| `[]` explanation → reason | explain edit | a repeated name whose new constraint fails its held ceiling | E1 |

Nothing else changes. In particular: no classic answer, no explain answer
on a non-repeating walk, no `removal_orphans`/`safe_upgrade`/`upgrade_set`/
`freeze_audit`/`dependents` answer (none of them call the search loop).

### 5.3 Re-baseline procedure (deliberate, human-approved, in this order)

0. **Harness first (no resolver change).** Give `diff_runner.pl` a
   per-case `call_with_time_limit/2` (suggest 5 s) emitting
   `{"timeout": true}`; teach `compare_jsonl.mjs` to treat `timeout` as its
   own class; have the wamjs shim emit `{"exhausted": true}` on the step
   cap instead of `fail` (the H6 fix, or a minimal marker). Snapshot the
   baseline oracle: `swi_base.jsonl` (2,600), `swi_store_base.jsonl` (503),
   `corpus_base.jsonl` (`dump_corpus.pl`). Commit the snapshots.
1. **Land H4** (§1, term + store mirror). Regenerate all three oracle
   files. `diff` old vs new: **every** differing line must have
   `{"timeout": true}` (or the exhausted marker) on the old side. Expected
   on the current seeds: zero differing lines **[measured]**. Extend
   `gen_catalogs.mjs` with a *new* seeded family (held cycles, multi-version
   names, held providers; new ids, existing seeds untouched) and record
   its H4 changed-set. Human approves the changed set by id; the approval
   is recorded in the ledger row.
2. **Land H1** (§2, term + store mirror) and the explain edit (§4).
   Regenerate. `diff` H4-oracle vs new: every differing line must be
   `fail → ok` or `ok → ok'`; zero `ok → fail`, zero `→ timeout`. Expected
   on the current seeds: exactly the 11 lines of §5.2 **[measured]**.
   Human approves by id.
3. **Probe and corpus edits** (§5.4), in the same commit as the stage that
   changes them, so each commit is green on its own.
4. **Legs.** Rebuild wamjs, wamjs_store and Go from the new `resolver.pl`;
   corpus and differentials at 0 divergences against the *new* oracle;
   term/store identity on the §6.1 set. Rust and ClojureScript builds are
   still pre-P3 (D57/D63); they join the gate when their P3 port rounds
   land, and this design's §6.1 set is added to their port briefs.
5. **Perf.** Rerun the B2/B3 numbers in `BENCHMARKS.md` on SWI and wamjs;
   the wamjs B3 step count must stay under the cap.
6. **Commit the new baseline snapshots** and replace the old ones; update
   the pruning design's §1.3, §3.2 (H1, H4 → fixed, with the ledger row),
   §4 invariant 4 ("layered's commit (H1) included" is deleted), and the
   README hazard list (H1, H4 fixed; H3's reading per §4).

### 5.4 Existing assertions that must be edited

| Assertion | Today | After H4 | After H4+H1 |
|---|---|---|---|
| `ce2` | `S1 == [a-1,b-2,d-1,x-2]` and modified catalog **fails** | unchanged **[measured]** | split: `S1` unchanged; modified catalog **answers** `[a-1,b-1,d-2,x-2]` **[measured]** |
| `ce5` | succeeds **only** on timeout | inverted: `resolve_layered(Cat,[a],S)` returns within the limit with `S == []`; rename `ce5_cyclic_held_deps_terminate` **[measured]** | same |
| `ce6` | classic `[b-2,d-1,x-1]`, layered **fails** | unchanged **[measured]** | layered `== [b-2,d-1,x-1]` (classic parity restored) **[measured]** |
| `ce1`, `ce3`, `ce4`, `ce_h5`, A1–A11 incl. both A9 pairings | — | unchanged **[measured: 23/23 green]** | unchanged **[measured]** |
| `a3_resolve_agrees_and_icat_never_escapes` | calls `resolve_pending/5` | still valid through the wrapper | same |
| README "Known semantic hazards" | H1, H4 listed | H4 → fixed | H1 → fixed; H3 text gains the §4 reading |
| `RESOLVER_PRUNING_DESIGN.md` §4 inv. 4 | "layered's commit (H1) included" | — | clause removed; §1.3 marked historical |
| `BENCHMARKS.md` | D59 numbers | — | rerun (step 5) |

The probe file's header sentence "they succeed on the pre-pruning resolver
and must still succeed after the guards land" must be reworded: CE2/CE5/CE6
become *pins of the new semantics*, and the D59 wording is kept as history.

---

## 6. Corpus and validation

### 6.1 Scenario list (all mandated shapes; every row was run on the prototype)

Catalogs are `v(M,I,P)` unless noted; `held` = base list. Expected answers
are the H4+H1 answers. The JSON form of eleven of these (differential
schema) ran through the wamjs build at 0 divergences.

| Id | Shape | Expected (both fixes) |
|---|---|---|
| `ce5_cycle_ok` | held `a ↔ b`, request `[a]` | `[]` |
| `cycle_closing_edge_fails` | held `a-1 → b-1 → a = 2` | fails |
| `cycle_order_y_before_x` | held `a → [b, x]`, `b → [a, y]`; `y-2 → d = 1`, `x-2 → d any` | `[d-1, x-2, y-2]` (y before x: `d-1` chosen by y, x accepts it) |
| `cycle_in_alt_rescued` | `app → alt(p, z)`, `p → a`, held `a → b → a = 2` | `[app, z]` (finite in the baseline too: the closing edge fails) |
| `cycle_in_alt_satisfiable` | same, closing edge `any` | `[app, p]` |
| `cycle_then_doomed_alt` | `p → [a, h ≥ 2]`, held `a ↔ b`, `h` held `1` | `[app, z]` — the §7.1 shape, now legitimately |
| `h2_second_version_progress` | held `mawk-1`, `h`; `h → awk`; `mawk-2 provides awk`, `mawk-2 → h` | `[app, mawk-2]` (insertion of a second version advances `Gen`; the re-expansion of `h` is not closed) |
| `h2_gen_advance_order` | held `mawk-1`, `h`; `h → [awk, w]`; `mawk-2 provides awk`, `mawk-2 → [h, q]`; `w-2 → d = 1`, `q-2 → d = 2` | `[app, d-1, mawk-2, q-1, w-2]` — `h` is **replayed** under `mawk-2` (w before q). With the increment removed the answer is `[app, d-2, mawk-2, q-2, w-1]`; the baseline and H4-only fail (H1 shape as well) |
| `held_provider_cycle` | `postfix provides mta`, `postfix → app2`, `app2 → mta`, both held | `[app]` (mark keyed on the held *provider* pair) |
| `distinct_held_versions` | base holds `a-1` and `a-2`; `a-2 provides v`, `a-2 → a` | `[app]` (two marks, two keys) |
| `named_layer_cycle` | `a ↔ b` held in `layer(dev, …)` | `[]` (`base_ver` sees named layers) |
| `sibling_scope` | held `a → [b, c]`, `b → x`, `c → x`, `x → y` | `[y]` (x is re-expanded under `c`: its mark was popped; approach 4 would skip it; same answer) |
| `rollback_marks_with_generation` | held `a → [x, b, d = 1]`; `x-2 → [b, d = 2]`; `x-1 → b`; held `b → a` | `[d-1, x-1]` (marks+gen restored after `x-2` fails) |
| `stale_mark_would_be_unsound` | held `a → x`; `x-2 → [b, q]` (q missing); `x-1 → b`; held `b → z = 2`, `z` held `1` | **fails** (a stale `b` mark would hide `z`'s ceiling and succeed) |
| `stale_mark_control_succeeds` | same with `z = 1` | `[x-1]` |
| `needs_lower_retry_cyclic` | `b → x`; `x-2 → [h, k = 2]`; held `h ↔ k`, `k` held `1`; `x-1` bare | `[b, x-1]` |
| `satisfiable_cyclic_highest` | `b → x`; `x-2 → h`; held `h ↔ k` | `[b, x-2]` |
| `lower_version_enters_cycle` | `x-2 → k = 2` (fails), `x-1 → h`, held `h ↔ k` | `[b, x-1]` |
| `alt_rescued_by_lower` | `app → [alt(p, q), y]`; `p → x`; `x-2 → d = 2`; `y → d = 1` | `[app, d-1, p, x-1, y]` |
| `ce6_layered` | CE6 | `[b-2, d-1, x-1]` |
| `ce2_ds1` / `ce2_ds2` | CE2 both catalogs | `[a-1,b-2,d-1,x-2]` / `[a-1,b-1,d-2,x-2]` |
| `version_only_no_provider_fallback` | `x-1 → missing`, `p provides x` | layered fails; classic `[app, p]` (documented) |
| `held_ceiling_committed` | A9(i) | fails |
| `classic_cycle_unchanged` | classic on `a ↔ b`, nothing held | `[a, b]` |
| E1–E4 | §4 | as §4 |
| `cyc_layer_closure` | `layer_closure` through a held cycle | `[app]` |

### 6.2 The four-way matrix on SWI **[measured]**

`neither` = `8d0ed83`; `H4` = §1 only; `H1` = §2 only; `both` = §1+§2.
`t/o` = 2 s time limit; `stack` = 1 GB stack overflow (resource
exhaustion, the other observable).

| Scenario | neither | H4 | H1 | both |
|---|---|---|---|---|
| `ce2_ds1` | S1 | S1 | S1 | S1 |
| `ce2_ds2` | fail | fail | ok | ok |
| `ce5_cycle_ok` | t/o | `[]` | t/o | `[]` |
| `ce6_layered` | fail | fail | ok | ok |
| `cycle_closing_edge_fails` | fail | fail | fail | fail |
| `cycle_order_y_before_x` | stack | ok | stack | ok |
| `cycle_in_alt_satisfiable` | stack | ok | t/o | ok |
| `cycle_then_doomed_alt` | t/o | `[app,z]` | t/o | `[app,z]` |
| `h2_gen_advance_order` | fail | fail | ok | ok (`…q-1, w-2`; `…q-2, w-1` with the increment removed) |
| `held_provider_cycle` | t/o | ok | t/o | ok |
| `distinct_held_versions` | t/o | ok | t/o | ok |
| `named_layer_cycle` | t/o | `[]` | t/o | `[]` |
| `sibling_scope` | `[y]` | `[y]` | `[y]` | `[y]` |
| `rollback_marks_with_generation` | stack | fail | stack | ok |
| `stale_mark_would_be_unsound` | fail | fail | fail | fail |
| `stale_mark_control_succeeds` | fail | fail | ok | ok |
| `needs_lower_retry_cyclic` | t/o | fail | t/o | ok |
| `satisfiable_cyclic_highest` | stack | ok | stack | ok |
| `lower_version_enters_cycle` | fail | fail | **stack** | ok |
| `alt_rescued_by_lower` | `…q…` | `…q…` | `…p,x-1…` | `…p,x-1…` |
| `held_ceiling_committed` | fail | fail | fail | fail |
| `version_only_no_provider_fallback` | fail | fail | fail | fail |
| `explain_preferred_candidate` resolve / explain | fail / reason | fail / reason | ok / reason | ok / reason |

Reading: the H4 column never turns a finite baseline outcome into anything
else (I2/I3); the H1 column turns finite failures into divergence twice
(`lower_version_enters_cycle`, `rollback_marks_with_generation`) — the
measured form of "H4 is required for H1"; the `both` column is finite
everywhere.

How to produce the four cells without flags: `neither` and `both` are the
pre-round and post-round commits; `H4` is the intermediate commit; `H1`
is a throw-away branch applying §2 on the pre-round commit, built and
run for the matrix only, never merged. No mode flag enters `resolver.pl`.

### 6.3 Runtimes

| Leg | Status for this round | What it must show |
|---|---|---|
| SWI | oracle **[measured]** | §5.3 diffs; matrix §6.2 |
| wamjs (term + store) | prototype build **[measured]**: corpus 51/51, differential 2,600/0 vs new oracle, cyclic set 11/0 | the same after landing; B3 under the step cap; H6 marker |
| Go | P3-current (D61); not built here | corpus, differential, cyclic set at 0 vs new oracle; `switch_default_chain`/`maplist_predsort` probes stay green |
| Rust | pre-P3 build (D57/D63) | joins when its P3 port lands; §6.1 set added to that brief; the D63 unify fix makes the extra list traffic linear |
| ClojureScript | pre-P3 build (D57) | joins when its P3 port lands; same |

Each leg also runs R7 (§7) as a stand-alone probe before the resolver
corpus, so an ITE/then-branch regression is attributed to the runtime and
not to this design.

### 6.4 Resource exhaustion as a separate observable

Every runner distinguishes three outcomes per case: answer, finite
failure, exhausted (timeout or step cap). The matrix table above records
`t/o` and `stack` separately because they *were* different on the same
catalogs (the stack overflow is the 1 GB SWI limit being hit before 2 s).
The differential comparator must never equate `exhausted` with `fail`.

### 6.5 Approach 4, kept as the documented fallback

If, on one runtime, the `done/3` sentinel or the two-field state proves
unworkable (no such evidence exists after the wamjs run), approach 4 is
the substitute with the same validate-then-suppress ordering and the same
generation site:

```prolog
% St = seen(Visited): held pairs expanded since the latest insertion.
;   Origin = from_base
->  St = seen(Visited),
    (   visited_member(Visited, Pkg, Ver)
    ->  resolve_pending(Mode, Cat, Rest, Acc, St, Sel)
    ;   collect_deps(Cat, Pkg, Ver, DepReqs),
        append(DepReqs, Rest, More),                 % no sentinel
        resolve_pending(Mode, Cat, More, Acc, seen([Pkg-Ver|Visited]), Sel)
    )
;   ...,                                             % from_catalog
    resolve_pending(Mode, Cat, More, [Pkg-Ver|Acc], seen([]), Sel)   % reset
```

It needs no generation counter and no arithmetic, at the cost of the extra
proof obligation the exploration names: a *completed* held expansion
repeated at unchanged `Acc` is skipped rather than replayed, so one must
show the replay was a no-op (it added no selection; its ground tests were
already passed on the same `Acc`; but alternatives inside it could, on
the replay, be resolved *by a later selection in `Acc`* — which is
impossible at unchanged `Acc` — and pending obligations pushed by the
replay are the same ground tests). The exploration's 5,000-case agreement
between 3 and 4 is the empirical half of that argument; `sibling_scope`
is the shape where the two differ in steps but not in answer. Approach 4
must not be chosen silently: the ledger row names which approach shipped.

---

## 7. Review protocol — falsifiable assertions

The same external reviewer is asked to try to break each of these, on the
implementation, using the §6.1 catalogs plus their own sweeps.

- **R1 (H4 changes only divergent branches).** For every catalog and
  request on which the pre-round resolver returns a finite answer or
  finite failure within the budget, the H4-stage resolver returns the
  identical result. Sweep: seeded families with and without held cycles,
  both modes, with a timeout; any finite→different is a falsification.
- **R2 (LIFO is exact).** Instrument the `done` branch so a head mismatch
  throws instead of failing; no §6.1 case and no sweep case throws. A
  throw falsifies §1.4.
- **R3 (single-site generation).** The only `Gen` increment is in the
  `from_catalog` arm. Removing it must change `h2_gen_advance_order`
  (**[measured]**: `…q-1, w-2` → `…q-2, w-1`, because `h`'s re-expansion
  under the provider insertion is then closed instead of replayed) and
  `ce2_ds1` (closing the second `base0` expansion makes `x` pick `d-2`
  first and the answer becomes `[a-1,b-1,d-2,x-2]`). Moving the
  increment to `from_base` must make `ce5_cycle_ok` diverge. The
  bare H2 probe `h2_second_version_progress` is *not* sensitive to the
  site (same answer either way) and must not be cited as evidence for it.
- **R4 (closure after validation).** `cycle_closing_edge_fails` fails and
  E1 explains it. Swapping the order (closure before `pick_need`) must
  make it answer `[]` — confirm the swap is detected by the probe.
- **R5 (marks are keyed on the validated `Pkg-Ver`, not the name).**
  `distinct_held_versions` and `held_provider_cycle` answer as listed; a
  name-keyed mark changes at least one of them or diverges.
- **R6 (restoration).** `stale_mark_would_be_unsound` fails and
  `stale_mark_control_succeeds` succeeds; an asserted-mark implementation
  passes the control and *succeeds* the unsound case.
- **R7 (then-branch enumeration on every leg).** A stand-alone predicate
  `( true -> member(X, [1,2,3]) ; fail )` yields `[1,2,3]` under `findall`
  on each leg; then the resolver corpus.
- **R8 (version-only, not classic fallback).** `version_only_no_provider_fallback`
  fails in layered mode after the round; if it succeeds, a provider was
  reached after real candidates existed.
- **R9 (held commitment).** I4's two witnesses fail before and after.
- **R10 (combined termination).** Every §6.1 case and the new seeded
  cyclic family terminate under `both` with the timeout never firing;
  further, the reviewer is invited to construct a catalog on which `both`
  diverges — the theorem in §3 says none exists for ground well-formed
  input; a counterexample falsifies Lemma 1, 2 or the preservation of
  branch-local state.
- **R11 (inventory completeness).** The old-vs-new oracle diff contains
  only lines of the five §5.2 kinds, and the H4-stage diff only lines
  whose old side is exhausted. Any other line falsifies I2/I3.
- **R12 (explain).** E1–E4 as stated; on every explain case of the
  existing corpus and differential, identical output. A repeated name
  with a *satisfied* constraint never produces a reason.
- **R13 (term/store).** `resolver_store.pl` agrees with `resolver.pl` on
  the §6.1 set encoded as an env + store.

---

## Appendix A — worked traces on the listing

**CE2, original catalog** (held `base0-1`; `base0 → [a, b]`, `a → [base0, x]`,
`b-2 → d = 1`, `x-2 → d ≥ 1`), request `[base0]`:

```
[base0]                         st(0,[])
 base0 held → expand           [a, b, done(base0,1,0)]            st(0,[a(base0,1,0)])
 a → catalog a-1 (Gen 0→1)     [base0, x, b, done(base0,1,0)]     st(1,[a(base0,1,0)])   Acc [a-1]
 base0: validated (any), active_member? head a(base0,1,0): 0 < 1 → stop → NOT closed → expand again
                               [a, b, done(base0,1,1), x, b, done(base0,1,0)]   st(1,[a(base0,1,1),a(base0,1,0)])
 a: selected ✓                  b → b-2 (Gen 1→2) → d = 1 → d-1 (2→3)
 done(base0,1,1) → pop          x → x-2 (3→4) → d ≥ 1: selected d-1 ✓
 b: selected ✓                  done(base0,1,0) → pop
answer [a-1, b-2, d-1, x-2]     (byte-identical to today)
```

**CE5** (`a ↔ b` held), request `[a]`:

```
[a] st(0,[]) → expand a: [b, done(a)] st(0,[a(a,1,0)])
 b held → expand: [a, done(b), done(a)] st(0,[a(b,1,0),a(a,1,0)])
 a: validated (any); active_member: head b ≠ a, same gen → next: a(a,1,0) match → CLOSED
 done(b) pop; done(a) pop; Acc [] → answer []
```

**`needs_lower_retry_cyclic`** (`b → x`; `x-2 → [h, k = 2]`; held `h ↔ k`,
`k` held `1`; `x-1` bare), request `[b]`:

```
b → b-1 (Gen 1); x → choice point Desc = [2,1]; try x-2 (Gen 2): [h, k=2]
 h expand (mark h@2): [k, done(h), k=2]; k expand (mark k@2): [h, done(k), done(h), k=2]
 h: validated, active at gen 2 → closed; done(k); done(h)
 k = 2: base_ver k-1, satisfies(1, eq(2)) fails → branch fails
backtrack to member: x-1 (Gen 2), st restored to st(1,[]) → no deps → answer [b-1, x-1]
```

## Appendix B — what was run, and where

Scratch only (not committed): a copy of `resolver.pl` with §1, §2 and §4
applied verbatim; variants with §1 only and §2 only; `test_resolver.pl`
and `test_pruning_probes.pl` run against each; a scenario driver with a
2 s `call_with_time_limit` and resource-error catch; `gen_catalogs.mjs`'s
2,600 cases through `diff_runner.pl` for all three variants with
`statistics(inferences)`; the wamjs build (`wamjs/build.pl` on the
prototype source, `emit_mode(mixed)`), `run_corpus.mjs` and
`diff_runner_wamjs.mjs` against the prototype oracle; eleven hand-written
cyclic/H1 cases in the differential JSON schema through both. All
numbers quoted as **[measured]** come from those runs on the same box on
the same day; nothing in the repository was modified other than this file.
