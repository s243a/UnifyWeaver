:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% abi_resolve.pl -- symbol-level ABI-compatibility resolver (redesigned after
% the PR #4262 review; see REVIEW_NOTES.md for the point-by-point map).
%
% MODEL
%   Two independent axes:
%     * ELF version-node axis: a node (GLIBC_2.34, LIBSELINUX_1.0, COMMON_1,
%       PUBLIC, Base) is an opaque label. A requirement `Sym@Node` on soname So
%       is satisfied only by a provider row with the SAME (So, Sym, Node) --
%       string equality, never numeric ordering, never a bare-name fallback.
%       `Base` = dpkg's spelling of "unversioned"; an unversioned REQUIREMENT
%       (no node) is satisfied by any exported node of that symbol.
%     * Debian package-version axis: the `.symbols` minimum-version and the
%       release candidates are deb/3 terms produced by debian/deb_parse.pl and
%       ordered by the frozen resolver:version_lt/2 (epoch, ~, revision all
%       handled there). A non-deb release id is kept as label(Atom) and only
%       ever matches itself.
%   Evidence is explicit:
%     prov_evidence(So, Src, R0, complete)  -- So's export set is known
%       completely at evidence release R0 (Src = symbols | elf).
%     req_evidence(Bin, Src, Status, Detail) -- Bin's requirement set is
%       complete, or why not (missing_file / readelf_failed / inconsistent).
%   Provider bounds:
%     since(Min, MinAtom) -- from `.symbols`: exported at R0 and, by the
%       curated lower bound, at every release >= Min. Min is NOT a ground-truth
%       introduction date (Debian policy lets it be raised); R < Min is treated
%       as `below_floor` (the same conservative floor dpkg-shlibdeps emits).
%     at(R0)              -- from readelf: exported at exactly R0.
%   Extrapolation (defeasible, documented): within a soname, exports do not
%   disappear (removing one is an ABI break that requires a soname bump), so
%   presence at R0 is extrapolated to R > R0 with basis `extrapolated`, and
%   absence from a complete export set is a veto for every release of the
%   soname. Presence never becomes a guarantee: compatible(_) is defeasible.
%   For at(R0) evidence, R < R0 is UNKNOWN (readelf says nothing about older
%   releases). A hypothetical drop(Sym, Node, At) models an in-soname removal
%   (violating the assumption) to exercise the upper bound.
%
% VERDICTS  abi_verdict(Bin, So, Rel, Verdict):
%   compatible(exact | extrapolated)
%   incompatible([missing(Sym@Node) | below_floor(Sym@Node, MinAtom)
%                 | soname_mismatch(offered(So), needed(N)) ...])   -- HARD veto,
%       only reachable when requirement AND provider evidence are complete
%   unknown([no_requires_evidence(Bin) | requires_evidence(Status, Detail)
%            | no_provider_evidence(So) | unknown(Sym@Node, evidence_release(R0))
%            | unknown(Sym, no_provider_evidence(S)) ...])
%   not_needed(So)   -- Bin has no DT_NEEDED entry for So (and no stem clash)
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
    prov_evidence/4,
    req_evidence/4,
    release/3,
    rel_term/2,
    rel_le/2,
    rel_lt/2,
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

:- dynamic symprov/4.          % symprov(SoName, Sym, Node, Bound)   Bound = since(Deb, Atom) | at(Rel)
:- dynamic symreq/5.           % symreq(Binary, Sym, Node, SoName, Bind)   Node/SoName = none if unversioned
:- dynamic needed/2.           % needed(Binary, SoName)
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
    retractall(prov_evidence(_, _, _, _)),
    retractall(req_evidence(_, _, _, _)),
    retractall(release(_, _, _)).

load_abi_store(Dir) :-
    abi_store_clear,
    load_rows(Dir, 'symprov.jsonl',  assert_symprov),
    load_rows(Dir, 'symreq.jsonl',   assert_symreq),
    load_rows(Dir, 'needed.jsonl',   assert_needed),
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

% "<soname>|<sym>@<node>" -> since(Deb, Atom) | at(Rel)
assert_symprov(K, [Kind, V]) :-
    split_first(K, '|', So, Ident),
    split_last(Ident, '@', Sym, Node),
    Node \== '',
    (   Kind == since
    ->  parse_deb_version(V, Deb), Bound = since(Deb, V)
    ;   Kind == at
    ->  rel_term(V, Rel), Bound = at(Rel)
    ),
    assertz(symprov(So, Sym, Node, Bound)).

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

assert_evidence(K, V) :-
    split_first(K, '|', Kind, Subject),
    (   Kind == provides
    ->  V = [Src, RelAtom, Status, _Source],
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
% Provision at a release
% ---------------------------------------------------------------------------

% provides_at(So, Sym, Node, Rel, Basis): So exports exactly Sym@Node at Rel;
% Basis = exact (inside the evidence) | extrapolated (beyond the evidence
% release, under the in-soname monotone-export assumption).
provides_at(So, Sym, Node, Rel, Basis) :-
    symprov(So, Sym, Node, Bound),
    prov_evidence(So, _, R0, complete),
    bound_holds(Bound, R0, Rel, Basis).

bound_holds(since(Min, _), R0, Rel, Basis) :-
    rel_le(Min, Rel),
    (   rel_le(Rel, R0) -> Basis = exact ; Basis = extrapolated ).
bound_holds(at(R0), _, Rel, Basis) :-
    (   Rel == R0 -> Basis = exact
    ;   rel_lt(R0, Rel) -> Basis = extrapolated
    ).

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

versioned_status(So, Sym, Node, Bind, Rel, Hyp, Status) :-
    (   hyp_dropped(Hyp, Sym, Node, Rel)
    ->  Status = missing(Sym@Node, hypothetical_drop)
    ;   symprov(So, Sym, Node, Bound)
    ->  prov_evidence(So, _, R0, complete),
        (   bound_holds(Bound, R0, Rel, Basis)
        ->  Status = provided(Sym@Node, Basis)
        ;   Bound = since(_, MinAtom)
        ->  Status = below_floor(Sym@Node, MinAtom)
        ;   Status = unknown(Sym@Node, evidence_release(R0))
        )
    ;   Bind == 'WEAK'
    ->  Status = weak_unresolved(Sym@Node)
    ;   Status = missing(Sym@Node)
    ).

% An unversioned reference binds to any exported node of Sym in any NEEDED
% object. Against the queried So we evaluate at Rel; against the other NEEDED
% objects at their own evidence release. Complete absence is a veto only when
% every NEEDED object has complete provider evidence.
unversioned_status(Bin, So, Sym, Bind, Rel, Hyp, Status) :-
    (   \+ hyp_dropped_any(Hyp, Sym),
        (   provides_at(So, Sym, _, Rel, Basis)
        ->  Status = provided(Sym, any_node(So, Basis))
        ;   needed(Bin, S), S \== So, prov_evidence(S, _, R0, complete),
            provides_at(S, Sym, _, R0, _)
        ->  Status = provided(Sym, any_node(S, exact))
        ;   fail
        )
    ->  true
    ;   Bind == 'WEAK'
    ->  Status = weak_unresolved(Sym)          % a weak ref never vetoes, so missing evidence is moot
    ;   needed(Bin, S), \+ prov_evidence(S, _, _, complete)
    ->  Status = unknown(Sym, no_provider_evidence(S))
    ;   Status = missing(Sym)
    ).

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

aggregate_statuses(Ss, Verdict) :-
    include(hard_veto, Ss, Hard),
    include(is_unknown, Ss, Unk),
    (   Hard \== []
    ->  Verdict = incompatible(Hard)
    ;   Unk \== []
    ->  Verdict = unknown(Unk)
    ;   memberchk(provided(_, extrapolated), Ss)
    ->  Verdict = compatible(extrapolated)
    ;   memberchk(provided(_, any_node(_, extrapolated)), Ss)
    ->  Verdict = compatible(extrapolated)
    ;   Verdict = compatible(exact)
    ).

hard_veto(missing(_)).
hard_veto(missing(_, _)).
hard_veto(below_floor(_, _)).
is_unknown(unknown(_, _)).

% soname_offer(Bin, So, Offer): needed | mismatch(NeededSoName) | not_needed.
% Offering libfoo.so.2 to a binary whose DT_NEEDED says libfoo.so.1 is a hard
% veto: the loader matches DT_NEEDED by exact soname string.
soname_offer(Bin, So, Offer) :-
    (   needed(Bin, So)
    ->  Offer = needed
    ;   so_stem(So, Stem),
        needed(Bin, N), so_stem(N, Stem)
    ->  Offer = mismatch(N)
    ;   Offer = not_needed
    ).

so_stem(So, Stem) :-
    (   sub_atom(So, B, _, _, '.so')
    ->  sub_atom(So, 0, B, _, Stem)
    ;   Stem = So
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

req_floor(So, Sym-Node, Deb-Atom) :-
    symprov(So, Sym, Node, since(Deb, Atom)).

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
