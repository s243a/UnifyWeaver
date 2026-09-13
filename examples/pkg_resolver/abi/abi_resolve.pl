:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% abi_resolve.pl -- symbol-level ABI-compatibility resolver (redesigned after
% the PR #4262 review, revised after Sol's re-review; see REVIEW_NOTES.md for
% the point-by-point map).
%
% MODEL
%   Two independent axes:
%     * ELF version-node axis: a node (GLIBC_2.34, LIBSELINUX_1.0, COMMON_1,
%       PUBLIC, Base) is an opaque label. A requirement `Sym@Node` on soname So
%       is satisfied only by a provider row with the SAME (So, Sym, Node) --
%       string equality, never numeric ordering, never a bare-name fallback.
%       `Base` = dpkg's spelling of "unversioned". An unversioned REQUIREMENT
%       (no node) binds, as the loader does, only to a `Base` export or to a
%       DEFAULT export (`@@`, or the oldest version node); a provider row whose
%       default binding is unproven (a `.symbols` row not cross-checked against
%       the ELF) yields unknown, never compatible.
%     * Debian package-version axis: the `.symbols` minimum-version and the
%       release candidates are deb/3 terms produced by debian/deb_parse.pl and
%       ordered by the frozen resolver:version_lt/2 (epoch, ~, revision all
%       handled there). A non-deb release id is kept as label(Atom) and only
%       ever matches itself.
%   Evidence is explicit and every provider bound is tied to the evidence row
%   it rests on (Sol P1a):
%     prov_evidence(So, Src, R0, complete)  -- So's export set is known
%       completely at evidence release R0 (Src = symbols | elf).
%     symprov(So, Sym, Node, since(Min, MinAtom, R0, Bind)) -- from the
%       `.symbols` evidence at R0: exported at R0 and, by the curated lower
%       bound, at every release >= Min. Min is NOT a ground-truth introduction
%       date (Debian policy lets it be raised); R < Min is `below_floor` (the
%       conservative floor dpkg-shlibdeps emits) unless direct evidence says
%       otherwise.
%     symprov(So, Sym, Node, at(R0, Bind)) -- from readelf at R0.
%     Bind = default | nondefault | unproven  (default-version binding)
%     req_evidence(Bin, Src, Status, Detail) -- Bin's requirement set is
%       complete, or why not (missing_file / readelf_failed / inconsistent).
%   Per-identity status at release Rel aggregates EVERY complete evidence row
%   of the soname (ident_status/5): evidence AT Rel decides directly; else the
%   nearest evidence BELOW Rel (presence extrapolates upward) and the nearest
%   evidence ABOVE Rel (absence propagates downward, a curated floor covers
%   Rel >= Min) are combined. A release satisfied by ANY evidence row is
%   never vetoed by another row's bound.
%   Extrapolation (defeasible, documented): within a soname, exports do not
%   disappear (removing one is an ABI break that requires a soname bump), so
%   presence at R0 extends to R > R0 with basis `extrapolated`, and absence
%   from a complete export set at R1 is a veto for R =< R1. Absence at R1 says
%   nothing about R > R1 (later releases add symbols): unknown. Presence never
%   becomes a guarantee: compatible(_) is defeasible. A hypothetical
%   drop(Sym, Node, At) models an in-soname removal to exercise the upper bound.
%
% VERDICTS  abi_verdict(Bin, So, Rel, Verdict):
%   compatible(exact | curated | extrapolated)
%       exact        -- every requirement observed by readelf at exactly Rel
%       curated      -- at least one rests on `.symbols` metadata only (Sol residual)
%       extrapolated -- at least one rests on the monotone-export assumption
%   incompatible([missing(Sym@Node) | missing(Sym@Node, Why) | below_floor(Sym@Node, MinAtom)
%                 | soname_mismatch(offered(So), needed(N)) ...])   -- HARD veto,
%       only reachable when requirement AND provider evidence are complete
%   unknown([no_requires_evidence(Bin) | requires_evidence(Status, Detail)
%            | no_provider_evidence(So) | unknown(Sym@Node, Why)
%            | unknown(Sym, Why) ...])
%   not_needed(So)   -- Bin has no DT_NEEDED entry for So and So is not a
%                       declared replacement (replaces/2) of a NEEDED soname
%
% RANGE  abi_range(Bin, So, Releases, Result): every release in the ACTUAL
%   candidate axis is evaluated; range(Min, Max, Pairs) has compatible verdicts
%   at BOTH ends by construction (Min/Max are drawn from the compatible set).
%
% Frozen resolver.pl / resolver_store.pl are NOT edited.

:- module(abi_resolve, [
    op(200, xfx, @),
    load_abi_store/1,
    abi_store_clear/0,
    symprov/4,
    symreq/5,
    needed/2,
    replaces/2,
    prov_evidence/4,
    req_evidence/4,
    release/3,
    rel_term/2,
    rel_le/2,
    rel_lt/2,
    ident_status/5,
    provides_at/5,
    req_status/5,
    abi_verdict/4,
    abi_verdict/5,
    abi_floor/3,
    soname_offer/3,
    release_axis/2,
    abi_range/3,
    abi_range/4,
    abi_range/5,
    range_min_max/3
]).

:- op(200, xfx, @).            % Sym@Node terms in statuses/verdicts

:- use_module('../resolver', [version_lt/2]).
:- use_module('../debian/deb_parse', [parse_deb_version/2]).
:- use_module(library(http/json)).
:- use_module(library(lists)).
:- use_module(library(apply)).

:- dynamic symprov/4.          % symprov(SoName, Sym, Node, Bound)   Bound = since(Deb, Atom, R0, Bind) | at(R0, Bind)
:- dynamic symreq/5.           % symreq(Binary, Sym, Node, SoName, Bind)   Node/SoName = none if unversioned
:- dynamic needed/2.           % needed(Binary, SoName)
:- dynamic replaces/2.         % replaces(NewSoName, OldSoName)   declared soname succession
:- dynamic prov_evidence/4.    % prov_evidence(SoName, Src, Rel, Status)
:- dynamic req_evidence/4.     % req_evidence(Binary, Src, Status, Detail)
:- dynamic release/3.          % release(SoName, Rel, Atom)

% ---------------------------------------------------------------------------
% Store loading (P/2 JSONL: [Key, Value] per line; values may be JSON arrays)
% ---------------------------------------------------------------------------

abi_store_clear :-
    retractall(symprov(_, _, _, _)),
    retractall(symreq(_, _, _, _, _)),
    retractall(needed(_, _)),
    retractall(replaces(_, _)),
    retractall(prov_evidence(_, _, _, _)),
    retractall(req_evidence(_, _, _, _)),
    retractall(release(_, _, _)).

load_abi_store(Dir) :-
    abi_store_clear,
    load_rows(Dir, 'symprov.jsonl',  assert_symprov),
    load_rows(Dir, 'symreq.jsonl',   assert_symreq),
    load_rows(Dir, 'needed.jsonl',   assert_needed),
    load_rows(Dir, 'replaces.jsonl', assert_replaces),
    load_rows(Dir, 'evidence.jsonl', assert_evidence),
    load_rows(Dir, 'releases.jsonl', assert_release).

load_rows(Dir, File, Handler) :-
    atomic_list_concat([Dir, '/', File], Path),
    (   exists_file(Path)
    ->  setup_call_cleanup(open(Path, read, S),
                           load_row_lines(S, Path, 1, Handler),
                           close(S))
    ;   true
    ).

load_row_lines(S, Path, N, Handler) :-
    read_line_to_string(S, Line),
    (   Line == end_of_file
    ->  true
    ;   (   Line == ""
        ->  true
        ;   atom_string(Atom, Line),
            (   catch(atom_json_term(Atom, [K, V], [value_string_as(atom)]), _, fail),
                catch(call(Handler, K, V), _, fail)
            ->  true
            ;   throw(error(abi_store_row(Path, N, Line), load_abi_store/1))
            )
        ),
        N1 is N + 1,
        load_row_lines(S, Path, N1, Handler)
    ).

% "<soname>|<sym>@<node>" ->
%   ["since", MinVer, EvidenceRelease, Bind] -> since(Deb, MinVer, R0, Bind)
%   ["at", EvidenceRelease, Bind]            -> at(R0, Bind)
% The evidence release is part of the row so a bound is tied to the evidence
% it came from even when several evidence rows exist for one soname.
assert_symprov(K, [Kind | V]) :-
    split_first(K, '|', So, Ident),
    split_last(Ident, '@', Sym, Node),
    Node \== '',
    (   Kind == since
    ->  V = [Min, EvRel, Bind0],
        parse_deb_version(Min, Deb), rel_term(EvRel, R0),
        binding(Node, Bind0, Bind),
        Bound = since(Deb, Min, R0, Bind)
    ;   Kind == at
    ->  V = [EvRel, Bind0],
        rel_term(EvRel, R0),
        binding(Node, Bind0, Bind),
        Bound = at(R0, Bind)
    ),
    assertz(symprov(So, Sym, Node, Bound)).

% A `Base` (unversioned) export always binds an unversioned reference.
binding('Base', _, default) :- !.
binding(_, Bind, Bind) :- memberchk(Bind, [default, nondefault, unproven]).

% "<binary>|<sym>[@<node>]" -> symreq(Bin, Sym, Node, SoName, Bind)
assert_symreq(K, [So0, Bind]) :-
    split_first(K, '|', Bin, Ident),
    (   split_last(Ident, '@', Sym, Node)
    ->  Node \== '', So0 \== '', So = So0
    ;   Sym = Ident, Node = none, So = none
    ),
    memberchk(Bind, ['GLOBAL', 'WEAK']),
    assertz(symreq(Bin, Sym, Node, So, Bind)).

assert_needed(Bin, So) :-
    atom(So),
    assertz(needed(Bin, So)).

assert_replaces(New, Old) :-
    atom(Old),
    assertz(replaces(New, Old)).

assert_evidence(K, V) :-
    split_first(K, '|', Kind, Subject),
    (   Kind == provides
    ->  V = [Src, RelAtom, Status, _Source],
        memberchk(Src, [symbols, elf]),
        rel_term(RelAtom, Rel),
        assertz(prov_evidence(Subject, Src, Rel, Status))
    ;   Kind == requires
    ->  V = [Src, Status, Detail],
        assertz(req_evidence(Subject, Src, Status, Detail))
    ).

assert_release(So, V) :-
    rel_term(V, Rel),
    (   release(So, Rel, _) -> true ; assertz(release(So, Rel, V)) ).

split_first(Atom, Sep, Before, After) :-
    sub_atom(Atom, B, _, A, Sep), !,
    sub_atom(Atom, 0, B, _, Before),
    sub_atom(Atom, _, A, 0, After).

split_last(Atom, Sep, Before, After) :-
    sub_atom(Atom, B, _, A, Sep),
    \+ ( sub_atom(Atom, B2, _, _, Sep), B2 > B ), !,
    sub_atom(Atom, 0, B, _, Before),
    sub_atom(Atom, _, A, 0, After).

% ---------------------------------------------------------------------------
% Release axis (Debian package versions; labels only match themselves)
% ---------------------------------------------------------------------------

% rel_term(+Atom, -Rel): deb/3 via the frozen parser, else label(Atom).
rel_term(Atom, Rel) :-
    (   catch(parse_deb_version(Atom, Deb), _, fail),
        Deb = deb(_, [s([], _)|_], _)          % upstream starts with a digit (Policy 5.6.12)
    ->  Rel = Deb
    ;   Rel = label(Atom)
    ).

rel_lt(deb(E1, U1, R1), deb(E2, U2, R2)) :-
    version_lt(deb(E1, U1, R1), deb(E2, U2, R2)).

rel_le(A, B) :-
    (   A = deb(_, _, _), B = deb(_, _, _)
    ->  \+ version_lt(B, A)
    ;   A == B
    ).

rel_cmp(Order, A-_, B-_) :-
    (   rel_lt(A, B) -> Order = (<)
    ;   rel_lt(B, A) -> Order = (>)
    ;   A = deb(_, _, _), B = label(_) -> Order = (<)
    ;   A = label(_), B = deb(_, _, _) -> Order = (>)
    ;   A == B -> Order = (=)
    ;   compare(Order, A, B)
    ).

% release_axis(SoName, AscendingAtoms): the actual release candidates known for
% the soname (ingested `releases` rows), ascending, deduplicated.
release_axis(So, Atoms) :-
    findall(R-A, release(So, R, A), Pairs0),
    predsort(rel_cmp, Pairs0, Pairs),
    pairs_values(Pairs, Atoms).

% ---------------------------------------------------------------------------
% Evidence rows and what each says about one identity (Sol P1a)
% ---------------------------------------------------------------------------

% bound_evidence(So, Bound, Src, R0): the complete evidence row a bound rests
% on. A bound whose evidence row is missing/incomplete is not usable.
bound_evidence(So, since(_, _, R0, _), symbols, R0) :- prov_evidence(So, symbols, R0, complete).
bound_evidence(So, at(R0, _), elf, R0)              :- prov_evidence(So, elf, R0, complete).

% observed(So, Sym, Node, Src, R1, Bound): the identity is in the Src
% evidence taken at R1.
observed(So, Sym, Node, symbols, R1, B) :- B = since(_, _, R1, _), symprov(So, Sym, Node, B).
observed(So, Sym, Node, elf, R1, B)     :- B = at(R1, _),          symprov(So, Sym, Node, B).

% ev_says(So, Sym, Node, R1, Says): for every complete evidence row (Src, R1)
% of So, whether Sym@Node is present in it (and under which bound) or absent.
ev_says(So, Sym, Node, R1, Says) :-
    prov_evidence(So, Src, R1, complete),
    (   observed(So, Sym, Node, Src, R1, Bound)
    ->  Says = present(Src, Bound)
    ;   Says = absent(Src)
    ).

% ident_status(+So, +Sym, +Node, +Rel, -Status): the status of the exact
% identity Sym@Node on So at Rel, aggregated over ALL complete evidence rows.
%   provided(exact | curated | extrapolated)
%   missing(Why)                 absent from a complete export set at/above Rel
%   below_floor(MinAtom)         only a curated floor above Rel covers it, and Rel < Min
%   unknown(Why)                 no evidence row speaks about Rel
% Rule: evidence AT Rel decides (readelf before .symbols). Otherwise the
% nearest evidence BELOW Rel (last known state) and the nearest evidence ABOVE
% Rel are combined: presence below extrapolates upward unless the row above
% observed absence (then the identity was dropped somewhere in between:
% unknown, not a false compat and not a false veto); absence above propagates
% downward (monotone exports); a curated floor above covers Rel >= Min.
ident_status(So, Sym, Node, Rel, Status) :-
    findall(R1-Says, ev_says(So, Sym, Node, R1, Says), Rows),
    Rows \== [],
    (   member(Rel1-_, Rows), Rel1 == Rel
    ->  says_at(Rows, Rel, Says),
        says_status(Says, Rel, Status)
    ;   nearest_below(Rows, Rel, Below),
        nearest_above(Rows, Rel, Above),
        combine(Below, Above, Rel, Status)
    ).

% says_at(Rows, R, Says): what the evidence taken at exactly R says; when both
% tiers were taken at R, readelf (direct observation) decides.
says_at(Rows, R, Says) :-
    (   member(R1-present(elf, B), Rows), R1 == R    -> Says = present(elf, B)
    ;   member(R1-absent(elf), Rows), R1 == R        -> Says = absent(elf)
    ;   member(R1-present(symbols, B), Rows), R1 == R -> Says = present(symbols, B)
    ;   Says = absent(symbols)
    ).

says_status(present(elf, _), _, provided(exact)).
says_status(present(symbols, since(Min, MinAtom, _, _)), Rel, Status) :-
    ( rel_le(Min, Rel) -> Status = provided(curated) ; Status = below_floor(MinAtom) ).
says_status(absent(Src), Rel, missing(observed_absent(Src, Rel))).

% Distinct evidence releases strictly below / above Rel; the nearest one wins.
nearest_below(Rows, Rel, Below) :-
    findall(R-R, ( member(R-_, Rows), rel_lt(R, Rel) ), Bs0),
    predsort(rel_cmp, Bs0, Bs),
    (   Bs == [] -> Below = none
    ;   last(Bs, R0-_), says_at(Rows, R0, Says), Below = ev(R0, Says)
    ).

nearest_above(Rows, Rel, Above) :-
    findall(R-R, ( member(R-_, Rows), rel_lt(Rel, R) ), As0),
    predsort(rel_cmp, As0, As),
    (   As == [] -> Above = none
    ;   As = [R1-_|_], says_at(Rows, R1, Says), Above = ev(R1, Says)
    ).

% combine(Below, Above, Rel, Status)
combine(ev(R0, present(_, _)), ev(R1, absent(Src)), _, unknown(dropped_between(R0, Src, R1))) :- !.
combine(ev(_, present(_, _)), ev(_, present(symbols, since(Min, _, _, _))), Rel, provided(curated)) :-
    rel_le(Min, Rel), !.
combine(ev(_, present(_, _)), _, _, provided(extrapolated)) :- !.
combine(_, ev(R1, absent(Src)), _, missing(observed_absent(Src, R1))) :- !.
combine(_, ev(_, present(symbols, since(Min, MinAtom, _, _))), Rel, Status) :- !,
    ( rel_le(Min, Rel) -> Status = provided(curated) ; Status = below_floor(MinAtom) ).
combine(_, ev(R1, present(elf, _)), _, unknown(evidence_release(R1))) :- !.
combine(ev(R0, absent(Src)), none, _, unknown(absent_at(Src, R0))) :- !.
combine(none, none, _, unknown(no_evidence)).

% provides_at(So, Sym, Node, Rel, Basis): So exports exactly Sym@Node at Rel.
provides_at(So, Sym, Node, Rel, Basis) :-
    ident_status(So, Sym, Node, Rel, provided(Basis)).

% node_binding(So, Sym, Node, Bind): the default-version binding recorded for
% an export (default: an unversioned reference binds to it; nondefault: it
% does not; unproven: `.symbols` row not cross-checked against the ELF).
node_binding(So, Sym, Node, Bind) :-
    symprov(So, Sym, Node, Bound),
    (   Bound = since(_, _, _, B) -> Bind = B ; Bound = at(_, B) -> Bind = B ).

hyp_dropped(drop(Sym, Node, At), Sym, Node, Rel) :-
    rel_term(At, AtRel),
    rel_le(AtRel, Rel).

% ---------------------------------------------------------------------------
% Per-requirement status
% ---------------------------------------------------------------------------

% req_status(Bin, So, Rel, Hyp, Status) enumerates one Status per requirement
% of Bin that concerns So (versioned requirements attributed to So via the
% version index, plus Bin's unversioned requirements, which the loader
% resolves against any NEEDED object).
req_status(Bin, So, Rel, Hyp, Status) :-
    symreq(Bin, Sym, Node, So, Bind),
    Node \== none,
    versioned_status(So, Sym, Node, Bind, Rel, Hyp, Status).
req_status(Bin, So, Rel, Hyp, Status) :-
    symreq(Bin, Sym, none, none, Bind),
    unversioned_status(Bin, So, Sym, Bind, Rel, Hyp, Status).

% missing(Sym@Node) = absent from the complete export set observed AT Rel;
% missing(Sym@Node, observed_absent(Src, R1)) = absent at a LATER release R1,
% hence absent at Rel under monotone exports (the inference is visible).
versioned_status(So, Sym, Node, Bind, Rel, Hyp, Status) :-
    (   hyp_dropped(Hyp, Sym, Node, Rel)
    ->  Status = missing(Sym@Node, hypothetical_drop)
    ;   ident_status(So, Sym, Node, Rel, S)
    ->  (   S = provided(Basis)       -> Status = provided(Sym@Node, Basis)
        ;   S = below_floor(MinAtom)  -> Status = below_floor(Sym@Node, MinAtom)
        ;   S = missing(_), Bind == 'WEAK' -> Status = weak_unresolved(Sym@Node)
        ;   S = missing(observed_absent(_, R1)), R1 == Rel -> Status = missing(Sym@Node)
        ;   S = missing(Why)          -> Status = missing(Sym@Node, Why)
        ;   S = unknown(Why)          -> Status = unknown(Sym@Node, Why)
        )
    ;   Status = unknown(Sym@Node, no_provider_evidence(So))
    ).

% An unversioned reference binds (Sol P1b) to a `Base` export or to a DEFAULT
% export of Sym in any NEEDED object -- never to a non-default (`@`) one. A
% provider row whose binding is unproven (a `.symbols` row not cross-checked
% against the ELF) can only make the answer unknown. Against the queried So
% we evaluate at Rel; against the other NEEDED objects at their own evidence
% release. Complete absence is a veto only when every NEEDED object has
% complete provider evidence.
unversioned_status(Bin, So, Sym, Bind, Rel, Hyp, Status) :-
    (   \+ hyp_dropped_any(Hyp, Sym),
        (   unversioned_in(So, Sym, Rel, Node, Basis)
        ->  Status = provided(Sym, default_node(So, Node, Basis))
        ;   needed(Bin, S), S \== So, prov_evidence(S, _, R0, complete),
            unversioned_in(S, Sym, R0, Node, Basis)
        ->  Status = provided(Sym, default_node(S, Node, Basis))
        ;   fail
        )
    ->  true
    ;   Bind == 'WEAK'
    ->  Status = weak_unresolved(Sym)          % a weak ref never vetoes, so missing evidence is moot
    ;   unversioned_unproven(Bin, So, Sym, Rel, S1, N1)
    ->  Status = unknown(Sym, default_binding_unproven(S1, N1))
    ;   needed(Bin, S), \+ prov_evidence(S, _, _, complete)
    ->  Status = unknown(Sym, no_provider_evidence(S))
    ;   unversioned_unknown(Bin, So, Sym, Rel, _, Why)
    ->  Status = unknown(Sym, Why)
    ;   unversioned_nondefault(Bin, So, Sym, Rel, S3, N3)
    ->  Status = missing(Sym, no_default_export(S3, N3))
    ;   Status = missing(Sym)
    ).

% A default-bound export of Sym in S provided at Rel (any node).
unversioned_in(S, Sym, Rel, Node, Basis) :-
    node_binding(S, Sym, Node, default),
    ident_status(S, Sym, Node, Rel, provided(Basis)).

% Some export of Sym that is provided at the relevant release but whose
% default binding is unproven.
unversioned_unproven(Bin, So, Sym, Rel, S, Node) :-
    needed_at(Bin, So, Rel, S, R),
    node_binding(S, Sym, Node, unproven),
    ident_status(S, Sym, Node, R, provided(_)).

unversioned_unknown(Bin, So, Sym, Rel, S, Why) :-
    needed_at(Bin, So, Rel, S, R),
    node_binding(S, Sym, Node, B), B \== nondefault,
    ident_status(S, Sym, Node, R, unknown(Why)).

unversioned_nondefault(Bin, So, Sym, Rel, S, Node) :-
    needed_at(Bin, So, Rel, S, R),
    node_binding(S, Sym, Node, nondefault),
    ident_status(S, Sym, Node, R, provided(_)).

% needed_at(Bin, So, Rel, S, R): the queried So at Rel, other NEEDED objects
% at their own evidence release(s).
needed_at(_, So, Rel, So, Rel).
needed_at(Bin, So, _, S, R) :- needed(Bin, S), S \== So, prov_evidence(S, _, R, complete).

hyp_dropped_any(drop(Sym, _, _), Sym).

% ---------------------------------------------------------------------------
% Verdict
% ---------------------------------------------------------------------------

abi_verdict(Bin, So, RelAtom, Verdict) :-
    abi_verdict(Bin, So, RelAtom, none, Verdict).

abi_verdict(Bin, So, RelAtom, Hyp, Verdict) :-
    rel_term(RelAtom, Rel),
    (   \+ req_evidence(Bin, _, _, _)
    ->  Verdict = unknown([no_requires_evidence(Bin)])
    ;   req_evidence(Bin, _, Status, Detail), Status \== complete
    ->  Verdict = unknown([requires_evidence(Status, Detail)])
    ;   soname_offer(Bin, So, mismatch(N))
    ->  Verdict = incompatible([soname_mismatch(offered(So), needed(N))])
    ;   \+ needed(Bin, So)
    ->  Verdict = not_needed(So)
    ;   \+ prov_evidence(So, _, _, complete)
    ->  Verdict = unknown([no_provider_evidence(So)])
    ;   findall(S, req_status(Bin, So, Rel, Hyp, S), Ss),
        aggregate_statuses(Ss, Verdict)
    ).

% The verdict basis is the WEAKEST basis among the provided requirements:
% exact < curated < extrapolated.
aggregate_statuses(Ss, Verdict) :-
    include(hard_veto, Ss, Hard),
    include(is_unknown, Ss, Unk),
    (   Hard \== []
    ->  Verdict = incompatible(Hard)
    ;   Unk \== []
    ->  Verdict = unknown(Unk)
    ;   has_basis(Ss, extrapolated)
    ->  Verdict = compatible(extrapolated)
    ;   has_basis(Ss, curated)
    ->  Verdict = compatible(curated)
    ;   Verdict = compatible(exact)
    ).

has_basis(Ss, Basis) :-
    (   memberchk(provided(_, Basis), Ss) -> true
    ;   memberchk(provided(_, default_node(_, _, Basis)), Ss)
    ).

hard_veto(missing(_)).
hard_veto(missing(_, _)).
hard_veto(below_floor(_, _)).
is_unknown(unknown(_, _)).

% soname_offer(Bin, So, Offer): needed | mismatch(NeededSoName) | not_needed.
% Offering libfoo.so.2 to a binary whose DT_NEEDED says libfoo.so.1 is a hard
% veto ONLY under a declared succession relation replaces(libfoo.so.2,
% libfoo.so.1) (the loader matches DT_NEEDED by exact soname string). Without
% that declaration a name that is not NEEDED is simply not_needed -- no stem
% heuristic (Sol P2d: libfoo.so.2 vs an unrelated NEEDED libfoo.so.1-extra).
soname_offer(Bin, So, Offer) :-
    (   needed(Bin, So)
    ->  Offer = needed
    ;   replaces(So, N), needed(Bin, N)
    ->  Offer = mismatch(N)
    ;   Offer = not_needed
    ).

% ---------------------------------------------------------------------------
% Floor: the curated lower bound implied by `.symbols` (= dpkg-shlibdeps' dep)
% ---------------------------------------------------------------------------

% abi_floor(Bin, So, FloorAtom): the highest `.symbols` minimum-version among
% the provider rows matched (exactly, by node) by Bin's requirements on So.
% Fails if any versioned requirement on So has no since() provider row
% (missing symbol, or readelf-only evidence).
abi_floor(Bin, So, Floor) :-
    findall(Sym-Node, ( symreq(Bin, Sym, Node, So, _), Node \== none ), Reqs),
    Reqs \== [],
    maplist(req_floor(So), Reqs, Mins),
    max_deb(Mins, _-Floor).

req_floor(So, Sym-Node, Max) :-
    findall(Deb-Atom, symprov(So, Sym, Node, since(Deb, Atom, _, _)), Ms),
    Ms \== [],
    max_deb(Ms, Max).

max_deb([M|Ms], Max) :- foldl(max_deb_1, Ms, M, Max).
max_deb_1(D-A, D0-A0, Out) :- ( rel_lt(D0, D) -> Out = D-A ; Out = D0-A0 ).

% ---------------------------------------------------------------------------
% Range over the actual release axis
% ---------------------------------------------------------------------------

abi_range(Bin, So, Result) :-
    release_axis(So, Rels),
    abi_range(Bin, So, Rels, none, Result).

% abi_range(Bin, So, Hyp, Result): store axis, hypothetical drop(Sym, Node, At).
abi_range(Bin, So, Hyp, Result) :-
    release_axis(So, Rels),
    abi_range(Bin, So, Rels, Hyp, Result).

% abi_range(Bin, So, RelAtoms, Hyp, Result):
%   range(Min, Max, Pairs)  -- Min/Max are releases with compatible(_) verdicts
%   no_candidate(Pairs)     -- every release incompatible / not needed
%   unknown(Pairs)          -- no compatible release, some unknown
%   no_releases             -- empty axis
% Pairs = [RelAtom-Verdict ...] ascending.
abi_range(_Bin, _So, [], _Hyp, no_releases) :- !.
abi_range(Bin, So, RelAtoms, Hyp, Result) :-
    maplist(rel_pair, RelAtoms, P0),
    predsort(rel_cmp, P0, Sorted),
    pairs_values(Sorted, Asc),
    findall(A-V, ( member(A, Asc), abi_verdict(Bin, So, A, Hyp, V) ), Pairs),
    range_min_max(Pairs, Pairs, Result).

rel_pair(A, R-A) :- rel_term(A, R).

range_min_max(Pairs, Detail, Result) :-
    findall(A, member(A-compatible(_), Pairs), Compat),
    (   Compat = [Min|_]
    ->  last(Compat, Max),
        Result = range(Min, Max, Detail)
    ;   memberchk(_-unknown(_), Pairs)
    ->  Result = unknown(Detail)
    ;   Result = no_candidate(Detail)
    ).
