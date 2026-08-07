:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pe_emit.pl - byte-exact verification of the third pattern_stache
% consumer against the sealed golden bundle.
%
% Run from this directory:
%   swipl -g run_tests -t halt test_pe_emit.pl
%
% For every row of PROCESS_EXPRESSION_GOLDEN_v4.json the emitter is
% driven from the corresponding goal term (convention documented in
% pe_emit.pl) and the output must equal the row's
% canonical_identity_string byte-for-byte; the full form must equal
% canonical_full_string (identical for all rows except the pinned
% one).  Verification is against sealed bytes, not opinion: the JSON
% strings are the fixture, no Python involved.

:- use_module(pe_emit).
:- use_module(library(plunit)).
:- use_module(library(http/json)).

%% ============================================
%% Golden bundle access
%% ============================================

:- dynamic golden_row/3.   % golden_row(Name, IdentityString, FullString)

load_golden :-
    retractall(golden_row(_, _, _)),
    module_property(pe_emit, file(Here)),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/../PROCESS_EXPRESSION_GOLDEN_v4.json'], Path),
    setup_call_cleanup(
        open(Path, read, S, [encoding(utf8)]),
        json_read_dict(S, Bundle),
        close(S)),
    Bundle.registry_version == "v0.5",
    forall(member(Row, Bundle.rows),
           ( atom_string(Name, Row.name),
             assertz(golden_row(Name,
                                Row.canonical_identity_string,
                                Row.canonical_full_string)) )).

:- initialization(load_golden).

%% ============================================
%% The 30 goal terms, one per golden row
%% ============================================
%
% Hand-written under the documented goal convention; the golden
% strings are the only expected values.

golden_goal('atom-bare',        mod(graph, discrim)).
golden_goal('atom-dual-bare',   e5).
golden_goal(blend,              blend(mod(luna, 'D'), mod(luna, 'S'))).
golden_goal('blend-variadic',   blend(mod(luna, 'D'), mod(luna, 'S'), graph)).
golden_goal('dir-blend',        blend(mod(graph, discrim), mod(llm, element), mod(llm, subcat))).
golden_goal('distill-3tier',
    distill(e5(routing(e5, mod(sonnet, lineage), t([0.02, 0.03]), menus([10, 20]))))).
golden_goal('e5-auto',          e5(margin(t(0.03)))).
golden_goal('escaped-string',
    routing(e5, haiku, t([0.02]), menus([10]), manifest("a\"b\\c"))).
golden_goal('estimand-impl',
    lca_frac(simplemind, estimand(path), impl(structural))).
golden_goal('graph-judge',
    max(0.02,
        product(hop_decay(simplemind, gamma(0.6)), lca_frac(simplemind)),
        estimand(path))).
golden_goal('haiku-n10',        e5(routing(e5, haiku, t([0.02]), menus([10])))).
golden_goal('int-spelled-number',      margin(t(1))).
golden_goal('int-spelled-number-list', routing(e5, haiku, t([1]), menus([10]))).
golden_goal('kalman-fused',     kalman(mod(luna, 'D'), mod(luna, 'S'))).
golden_goal('lineage-haiku',
    lineage(pearltrees, mu(haiku), estimand(ancestry))).
golden_goal('margin-number',    margin(t(0.03))).
golden_goal('menu-required-int', menu(graph, n(10))).
golden_goal('mu-judge-kwarg',   lineage(simplewiki, mu(mod(sonnet, lineage)))).
golden_goal('neg-number',       lineage(fs, decay(-0.5))).
golden_goal('numeric-positional-literal', max(0.02, e5(margin(t(0.03))))).
golden_goal(pinned,
    pin(lineage(pearltrees, decay(0.85)), 'run/2026-07-25')).
golden_goal('sonnet-lin-n10',
    e5(routing(e5, mod(sonnet, lineage), t([0.02]), menus([10])))).
golden_goal('sonnet-lin-n20',
    e5(routing(e5, mod(sonnet, lineage), t([0.02, 0.03]), menus([10, 20])))).
golden_goal('substrate-atom',   fs).
golden_goal('utf8-string',
    routing(e5, haiku, t([0.02]), menus([10]), manifest("héllo·wörld"))).

%% ---- v0.5 additions (golden v4) ----
% enwiki joins the registered substrates; nothing else about the row
% is new, which is the point of including it.
golden_goal('enwiki-substrate',
    lineage(enwiki, mu(graph), estimand(ancestry))).
% cowalk: the first operator whose kwargs use the ENUMERATED kinds
% `walk` and `weight`.  Note the sealed canonical form injects
% weight="uniform" — cowalk's registry default — even though the
% authored expression omits it.
golden_goal('cowalk-sibling',
    cowalk(enwiki, walk(sibling), estimand(path))).
golden_goal('cowalk-weighted-cousin',
    cowalk(simplewiki, walk(cousin), weight(idf_node_size),
           mu(haiku), estimand(path))).
% pick over a menu: an operator applied to an operator, no kwargs.
golden_goal('pick-root',
    pick(menu(graph, n(10)))).
% Scientific notation.  GOAL CONVENTION NOTE: Prolog's reader
% normalizes the float literal to its own spelling (1e-05 and 1.0e-5
% both read as the same float), so the goal term carries a FLOAT, not
% a spelling — the v0.5 surface spelling `1e-05` is produced by
% pe_number.pl's CPython-repr rendering, not by how the literal was
% typed here.
golden_goal('scientific-notation-number',
    margin(t(1.0e-5))).

%% ============================================
%% Byte-exact verification, all 30 rows
%% ============================================

:- begin_tests(golden_bytes).

% Every golden row has a goal term and vice versa — no row silently
% unaccounted for.
test(bijection_rows_to_goals) :-
    aggregate_all(count, golden_row(_, _, _), 30),
    aggregate_all(count, golden_goal(_, _), 30),
    forall(golden_row(Name, _, _), golden_goal(Name, _)),
    forall(golden_goal(Name, _), golden_row(Name, _, _)).

% Semantic surface: byte equality with canonical_identity_string,
% for every row.
test(semantic_bytes, [forall(golden_goal(Name, Goal))]) :-
    golden_row(Name, Expected, _),
    pe_semantic(Goal, Got),
    (   Got == Expected
    ->  true
    ;   format(user_error, "~w:~n  expected ~q~n  got      ~q~n", [Name, Expected, Got]),
        fail
    ).

% Full surface: byte equality with canonical_full_string, for every
% row (identical to identity except the pinned row).
test(full_bytes, [forall(golden_goal(Name, Goal))]) :-
    golden_row(Name, _, Expected),
    pe_full(Goal, Got),
    (   Got == Expected
    ->  true
    ;   format(user_error, "~w:~n  expected ~q~n  got      ~q~n", [Name, Expected, Got]),
        fail
    ).

% The pinned row is the one place semantic and full diverge, exactly
% by the pin suffix.
test(pin_semantics) :-
    golden_goal(pinned, Goal),
    pe_semantic(Goal, Sem),
    pe_full(Goal, Full),
    Sem == "lineage(pearltrees,decay=0.85)",
    Full == "lineage(pearltrees,decay=0.85)@run/2026-07-25".

:- end_tests(golden_bytes).

%% ============================================
%% Unit tests for the driver's canonicalization pieces
%% ============================================

:- begin_tests(pe_emit_units).

% Default resolution is observable: mu given, decay injected.
test(default_injected) :-
    pe_semantic(lineage(pearltrees, mu(haiku)), S),
    S == "lineage(pearltrees,decay=0.85,mu=haiku)".

% An explicit value equal to the default renders identically.
test(explicit_default_same_bytes) :-
    pe_semantic(lineage(pearltrees, decay(0.85)), A),
    pe_semantic(lineage(pearltrees), B),
    A == B.

% Kwarg order in the GOAL is irrelevant: canonical order is sorted.
test(kwarg_order_canonicalized) :-
    pe_semantic(routing(e5, haiku, menus([10]), t([0.02])), A),
    pe_semantic(routing(e5, haiku, t([0.02]), menus([10])), B),
    A == B,
    A == "routing(e5,haiku,menus=[10],t=[0.02])".

% JSON escaping: quote and backslash escaped, non-ASCII passes
% through, control characters use JSON escapes.
test(json_escape_quote_backslash) :-
    pe_semantic(routing(e5, haiku, manifest("a\"b\\c")), S),
    S == "routing(e5,haiku,manifest=\"a\\\"b\\\\c\")".

test(json_escape_newline) :-
    pe_semantic(routing(e5, haiku, manifest("a\nb")), S),
    S == "routing(e5,haiku,manifest=\"a\\nb\")".

% Unknown goal forms fail closed with a named error.
test(unknown_form_is_error, error(pe_emit(unknown_form(_)))) :-
    pe_semantic(frobnicate(x), _).

% Nested pins strip everywhere in the semantic form and render
% everywhere in the full form (process_cards strips/keeps per node).
test(nested_pin_strips_in_semantic) :-
    pe_semantic(blend(pin(mod(luna, 'D'), 'run/1'), mod(luna, 'S')), S),
    S == "blend(luna.D,luna.S)".

test(nested_pin_kept_in_full) :-
    pe_full(blend(pin(mod(luna, 'D'), 'run/1'), mod(luna, 'S')), S),
    S == "blend(luna.D@run/1,luna.S)".

:- end_tests(pe_emit_units).
