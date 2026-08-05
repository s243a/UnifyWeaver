:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pe_resolved_ast.pl - the goal-term <-> resolved_ast correspondence:
% the STRUCTURAL oracle over the sealed golden bundle.
%
% Every row of PROCESS_EXPRESSION_GOLDEN_v3.json carries `resolved_ast`
% — the elaborated node structure as sealed JSON.  Consumers 3-4
% verified emitted STRINGS against the bundle; this module verifies the
% elaborated TERM: it projects a ground goal term into the
% resolved_ast shape so tests can compare structure before rendering.
% With both oracles on the same rows, a renderer bug and a builder bug
% can no longer mask each other.  No Python involved anywhere: the
% bundle is read as data (the standing rule — fixtures as data, never
% the vNext machinery).
%
% ============================================================
% THE CORRESPONDENCE (goal convention -> resolved_ast JSON)
% ============================================================
% Goal terms follow pe_emit's documented convention.  The projection:
%
%   bare registered atom        A
%     -> {kind:"atom", name:"A", output:<registry output of A>,
%         args:[], kwargs:[], mods:[], pins:[]}
%
%   modified atom               mod(Base, M)
%     -> the node of Base with "M" APPENDED to its mods list
%        (mods accumulate innermost-first; the bundle's rows carry
%        single mods, e.g. luna.D -> mods:["D"])
%
%   pinned expression           pin(Expr, P)
%     -> the node of Expr with "P" appended to its pins list
%        (resolved_ast is the FULL structure: pins are present in the
%        sealed JSON; a semantic-side comparison strips them — see
%        strip_pins/2)
%
%   numeric positional literal  N  (e.g. max's first argument)
%     -> {lexical:"<N via ~w>", literal:true, value_type:"number"}
%
%   operator application        Name(Pos..., Kw...)
%     -> {kind:"apply", name:"Name", output:<registry output>,
%         args:[<node per positional, in order>],
%         kwargs:[<entry per kwarg, defaults resolved, sorted by key>],
%         mods:[], pins:[]}
%
%   kwarg (unary wrapper)       K(V), functor K = key
%     value kind (number|int|string|number_list|int_list|estimand|impl):
%       -> {key:"K", lexical:<canonical value spelling>, value_type:"<kind>"}
%          lexical spellings: numbers via ~w; lists "[v1,v2]"; string,
%          estimand and impl as JSON strings (double-quoted, \" and \\
%          escaped, non-ASCII preserved) — the SAME spelling pe_emit
%          renders, shared via pe_emit's exported json_escape/2 so the
%          two consumers cannot drift from each other (each is still
%          independently pinned to its own sealed expected values)
%     node kind (a declared output type, e.g. mu -> judge):
%       -> {key:"K", node:<node of V>, value_type:"<kind>"}
%
%   defaults: every registry default not explicitly given is added
%   (v0.4 has exactly one: lineage's decay=0.85), exactly as the
%   canonical string resolves them.
%
% Comparison discipline: dicts built here and dicts read by
% json_read_dict have anonymous tags and are never ==; normalize both
% sides with normalize_json_term/2 (dicts -> obj(SortedPairs)) before
% comparing.

:- module(pe_resolved_ast, [
    resolved_ast_projection/2,   % +GroundGoal, -Dict
    normalize_json_term/2,       % +DictOrValue, -PlainTerm
    strip_pins/2,                % +Dict, -DictNoPins (every node)
    golden_resolved_ast/2,       % ?RowName, -Dict (bundle read as data)
    projection_matches_row/2     % +RowName, +GroundGoal (semidet)
]).

:- use_module(pe_registry_mirror,
              [pe_atom/1, pe_operator/1, pe_kwspec/4, pe_output/2]).
:- use_module(pe_emit, [json_escape/2]).
:- use_module(library(http/json)).
:- use_module(library(apply)).
:- use_module(library(lists)).

%% resolved_ast_projection(+GroundGoal, -Dict)
resolved_ast_projection(pin(E, P), D) :-
    !,
    resolved_ast_projection(E, D0),
    atom_string(P, PS),
    Pins0 = D0.pins,
    append(Pins0, [PS], Pins),
    D = D0.put(pins, Pins).
resolved_ast_projection(mod(B, M), D) :-
    !,
    resolved_ast_projection(B, D0),
    atom_string(M, MS),
    Mods0 = D0.mods,
    append(Mods0, [MS], Mods),
    D = D0.put(mods, Mods).
resolved_ast_projection(N, D) :-
    number(N),
    !,
    lex_plain(N, Lex),
    D = _{lexical: Lex, literal: true, value_type: "number"}.
resolved_ast_projection(A, D) :-
    atom(A),
    pe_atom(A),
    !,
    atom_string(A, NameS),
    pe_output(A, Out),
    atom_string(Out, OutS),
    D = _{args: [], kind: "atom", kwargs: [], mods: [], name: NameS,
          output: OutS, pins: []}.
resolved_ast_projection(Goal, D) :-
    compound(Goal),
    compound_name_arguments(Goal, Name, RawArgs),
    pe_operator(Name),
    !,
    split_kwargs(RawArgs, Name, Pos, Kw0),
    findall(K-V, ( pe_kwspec(Name, K, _, default(V)),
                   \+ memberchk(K-_, Kw0) ),
            Defaults),
    append(Kw0, Defaults, Kw1),
    msort(Kw1, Kw),
    maplist(resolved_ast_projection, Pos, Args),
    maplist(kw_entry(Name), Kw, Kwargs),
    atom_string(Name, NameS),
    pe_output(Name, Out),
    atom_string(Out, OutS),
    D = _{args: Args, kind: "apply", kwargs: Kwargs, mods: [],
          name: NameS, output: OutS, pins: []}.
resolved_ast_projection(Goal, _) :-
    throw(error(pe_resolved_ast(unprojectable(Goal)), _)).

split_kwargs([], _, [], []).
split_kwargs([A|Rest], Op, Pos, [K-V|Kw]) :-
    compound(A),
    compound_name_arguments(A, K, [V]),
    pe_kwspec(Op, K, _, _),
    !,
    split_kwargs(Rest, Op, Pos, Kw).
split_kwargs([A|Rest], Op, [A|Pos], Kw) :-
    split_kwargs(Rest, Op, Pos, Kw).

kw_entry(Op, K-V, Entry) :-
    once(pe_kwspec(Op, K, Kind, _)),
    atom_string(K, KS),
    atom_string(Kind, KindS),
    (   value_kind_lex(Kind, V, Lex)
    ->  Entry = _{key: KS, lexical: Lex, value_type: KindS}
    ;   % node-valued kwarg: the declared kind is an output type
        resolved_ast_projection(V, Node),
        Entry = _{key: KS, node: Node, value_type: KindS}
    ).

value_kind_lex(number, V, Lex)      :- lex_plain(V, Lex).
value_kind_lex(int, V, Lex)         :- lex_plain(V, Lex).
value_kind_lex(string, V, Lex)      :- lex_quoted(V, Lex).
value_kind_lex(estimand, V, Lex)    :- lex_quoted(V, Lex).
value_kind_lex(impl, V, Lex)        :- lex_quoted(V, Lex).
value_kind_lex(number_list, V, Lex) :- lex_list(V, Lex).
value_kind_lex(int_list, V, Lex)    :- lex_list(V, Lex).

lex_plain(V, Lex) :- format(string(Lex), "~w", [V]).
lex_list(Vs, Lex) :-
    maplist([V, S]>>format(string(S), "~w", [V]), Vs, Ss),
    atomic_list_concat(Ss, ',', A),
    format(string(Lex), "[~w]", [A]).
lex_quoted(V, Lex) :-
    json_escape(V, Esc),
    format(string(Lex), "\"~w\"", [Esc]).

%% normalize_json_term(+Value, -Term)
%  dicts -> obj(SortedPairs), lists mapped, scalars unchanged; makes
%  built dicts and json_read_dict dicts comparable with ==.
normalize_json_term(V, obj(Norm)) :-
    is_dict(V),
    !,
    dict_pairs(V, _, Pairs),
    msort(Pairs, Sorted),
    maplist([K-V0, K-N]>>normalize_json_term(V0, N), Sorted, Norm).
normalize_json_term(V, list(Norm)) :-
    is_list(V),
    !,
    maplist(normalize_json_term, V, Norm).
normalize_json_term(V, V).

%% strip_pins(+Dict, -DictNoPins)
%  Empty the pins list at EVERY node (the semantic form strips pins
%  per node, mirroring process_cards._canonical keep_pins=False), so
%  an unpinned spelling can be compared against a pinned row's sealed
%  structure.
strip_pins(D, Out) :-
    is_dict(D),
    !,
    dict_pairs(D, Tag, Pairs),
    maplist(strip_pins_pair, Pairs, Pairs1),
    dict_pairs(Out, Tag, Pairs1).
strip_pins(L, Out) :-
    is_list(L),
    !,
    maplist(strip_pins, L, Out).
strip_pins(V, V).

strip_pins_pair(pins-_, pins-[]) :- !.
strip_pins_pair(K-V, K-V1) :- strip_pins(V, V1).

%% golden_resolved_ast(?RowName, -Dict)
%  The sealed bundle's resolved_ast for a row, read as data and cached.
:- dynamic golden_ast_cache/2.

golden_resolved_ast(Name, Dict) :-
    (   golden_ast_cache(_, _)
    ->  true
    ;   load_bundle
    ),
    golden_ast_cache(Name, Dict).

load_bundle :-
    module_property(pe_resolved_ast, file(Here)),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/../PROCESS_EXPRESSION_GOLDEN_v3.json'], Path),
    setup_call_cleanup(
        open(Path, read, S, [encoding(utf8)]),
        json_read_dict(S, Bundle),
        close(S)),
    Bundle.registry_version == "v0.4",
    forall(member(Row, Bundle.rows),
           ( atom_string(N, Row.name),
             assertz(golden_ast_cache(N, Row.resolved_ast)) )).

%% projection_matches_row(+RowName, +GroundGoal)
%  Semidet: the goal's projection is structurally identical to the
%  named sealed row's resolved_ast.
projection_matches_row(Name, Goal) :-
    golden_resolved_ast(Name, Sealed),
    resolved_ast_projection(Goal, Built),
    normalize_json_term(Sealed, N1),
    normalize_json_term(Built, N2),
    N1 == N2.
