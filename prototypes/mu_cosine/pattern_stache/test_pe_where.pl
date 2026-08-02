:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pe_where.pl - verification of the where-clause form (fourth
% pattern_stache consumer) against the sealed golden bundle.
%
% Run from this directory:
%   swipl -g run_tests -t halt test_pe_where.pl
%
% Both directions of the DESIGN_desugaring_to_prolog_goals.md §12
% claim are tested against sealed bytes:
%   - a where-spelling with the repetition factored into a binding
%     produces output BYTE-IDENTICAL to the direct repeated-literal
%     spelling, and therefore to the golden row;
%   - the binding provably vanishes: the emitted surface carries no
%     variable name and no binding syntax (grep-level assertion, on
%     top of the byte equality that already implies it).

:- use_module(pe_where).
:- use_module(pe_emit, [pe_semantic/2, pe_full/2]).
:- use_module(library(plunit)).
:- use_module(library(http/json)).
:- use_module('../../../src/unifyweaver/core/pattern_stache', []).

%% ============================================
%% Golden bundle access
%% ============================================

:- dynamic golden_row/3.   % golden_row(Name, IdentityString, FullString)

load_golden :-
    retractall(golden_row(_, _, _)),
    module_property(pe_where, file(Here)),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/../PROCESS_EXPRESSION_GOLDEN_v3.json'], Path),
    setup_call_cleanup(
        open(Path, read, S, [encoding(utf8)]),
        json_read_dict(S, Bundle),
        close(S)),
    Bundle.registry_version == "v0.4",
    forall(member(Row, Bundle.rows),
           ( atom_string(Name, Row.name),
             assertz(golden_row(Name,
                                Row.canonical_identity_string,
                                Row.canonical_full_string)) )).

:- initialization(load_golden).

%% ============================================
%% Where-spellings vs direct spellings
%% ============================================
%
% where_spelling(RowName, WhereTerm, DirectGoal)
%
% The five rows whose golden expression contains a repeated atom in
% bindable positions (the §12 case: the repetition factored into ONE
% binding that reaches both occurrences), plus a representative set of
% single-occurrence bindings covering every value kind: number, whole
% list, string, judge expression, enum atom, corpus atom, and a
% binding inside the semantic part of a pinned expression.

% -- repeated atoms: one binding, two (or more) occurrences --
where_spelling(blend,
    where(blend(mod(L, 'D'), mod(L, 'S')), [L = luna]),
    blend(mod(luna, 'D'), mod(luna, 'S'))).
where_spelling('blend-variadic',
    where(blend(mod(L, 'D'), mod(L, 'S'), graph), [L = luna]),
    blend(mod(luna, 'D'), mod(luna, 'S'), graph)).
where_spelling('dir-blend',
    where(blend(mod(graph, discrim), mod(L, element), mod(L, subcat)), [L = llm]),
    blend(mod(graph, discrim), mod(llm, element), mod(llm, subcat))).
where_spelling('graph-judge',
    where(max(0.02,
              product(hop_decay(C, gamma(0.6)), lca_frac(C)),
              estimand(path)),
          [C = simplemind]),
    max(0.02,
        product(hop_decay(simplemind, gamma(0.6)), lca_frac(simplemind)),
        estimand(path))).
where_spelling('kalman-fused',
    where(kalman(mod(L, 'D'), mod(L, 'S')), [L = luna]),
    kalman(mod(luna, 'D'), mod(luna, 'S'))).

% -- representative single-occurrence bindings --
where_spelling('e5-auto',
    where(e5(margin(t(T))), [T = 0.03]),
    e5(margin(t(0.03)))).
where_spelling('lineage-haiku',
    where(lineage(S, mu(M), estimand(E)),
          [S = pearltrees, M = haiku, E = ancestry]),
    lineage(pearltrees, mu(haiku), estimand(ancestry))).
where_spelling('int-spelled-number-list',
    where(routing(e5, haiku, t(T), menus([10])), [T = [1]]),
    routing(e5, haiku, t([1]), menus([10]))).
where_spelling('utf8-string',
    where(routing(e5, haiku, t([0.02]), menus([10]), manifest(M)),
          [M = "héllo·wörld"]),
    routing(e5, haiku, t([0.02]), menus([10]), manifest("héllo·wörld"))).
where_spelling(pinned,
    where(pin(lineage(S, decay(0.85)), 'run/2026-07-25'), [S = pearltrees]),
    pin(lineage(pearltrees, decay(0.85)), 'run/2026-07-25')).
where_spelling('substrate-atom',
    where(C, [C = fs]),
    fs).

:- begin_tests(where_golden_bytes).

% The §12 measurement, re-proven in Prolog against the bundle: the
% where-spelling, the direct repeated-literal spelling, and the sealed
% golden bytes agree — semantic surface.
test(where_equals_direct_equals_golden_semantic,
     [forall(where_spelling(Name, WhereTerm, Direct))]) :-
    golden_row(Name, Golden, _),
    where_semantic(WhereTerm, FromWhere),
    pe_semantic(Direct, FromDirect),
    (   FromWhere == FromDirect, FromDirect == Golden
    ->  true
    ;   format(user_error,
               "~w:~n  golden ~q~n  direct ~q~n  where  ~q~n",
               [Name, Golden, FromDirect, FromWhere]),
        fail
    ).

% Same, full surface (pins rendered; identical except the pinned row).
test(where_equals_direct_equals_golden_full,
     [forall(where_spelling(Name, WhereTerm, Direct))]) :-
    golden_row(Name, _, Golden),
    where_full(WhereTerm, FromWhere),
    pe_full(Direct, FromDirect),
    FromWhere == FromDirect,
    FromDirect == Golden.

% Elaboration output is the ground goal itself, byte-identical in
% structure to the direct spelling (==, not just same rendering).
test(elaboration_yields_direct_goal,
     [forall(where_spelling(_Name, WhereTerm, Direct))]) :-
    elaborate_where(WhereTerm, Ground),
    Ground == Direct.

:- end_tests(where_golden_bytes).

%% ============================================
%% The discharge property: the binding vanishes
%% ============================================

:- begin_tests(where_discharge).

% The where-term is built FROM TEXT with a named variable, so "no
% variable name in the output" is meaningful; the golden row used has
% no uppercase characters, so any surviving variable name would be
% caught by the case scan, and any surviving binding syntax by the
% substring scans.
test(binding_vanishes_from_surface) :-
    read_term_from_atom(
        'where(max(0.02, product(hop_decay(C, gamma(0.6)), lca_frac(C)), estimand(path)), [C = simplemind])',
        W, [variable_names(['C' = _])]),
    where_semantic(W, S),
    golden_row('graph-judge', S, _),          % byte-equal to sealed row
    \+ sub_string(S, _, _, _, "where"),
    \+ sub_string(S, _, _, _, "C"),
    forall(( string_code(_, S, Code), code_type(Code, alpha) ),
           \+ code_type(Code, upper)).

% The caller's term is never mutated: after elaboration the caller
% still holds an open where-term with its variable unbound.
test(caller_term_not_mutated) :-
    W = where(lca_frac(C), [C = simplemind]),
    where_semantic(W, S),
    S == "lca_frac(simplemind)",
    var(C).

:- end_tests(where_discharge).

%% ============================================
%% Fail-closed cases
%% ============================================

:- begin_tests(where_fail_closed).

% A variable with no binding is an error at emit time: this prototype
% targets ground emission; residuation is the elaborator's future.
test(unbound_variable_is_error,
     error(pe_where(unbound_after_elaboration(_)))) :-
    where_semantic(
        where(product(hop_decay(C, gamma(0.6)), lca_frac(_D)), [C = simplemind]),
        _).

% A binding for a variable that does not occur hides a typo.
test(dead_binding_is_error,
     error(pe_where(dead_binding(_)))) :-
    where_semantic(where(lca_frac(simplemind), [_C = fs]), _).

% Two bindings for one variable are an error even when the values
% agree — a duplicate is a spelling accident.
test(duplicate_binding_is_error,
     error(pe_where(duplicate_binding_for_one_variable))) :-
    where_semantic(where(lca_frac(C), [C = simplemind, C = simplemind]), _).

test(duplicate_conflicting_binding_is_error,
     error(pe_where(duplicate_binding_for_one_variable))) :-
    where_semantic(where(lca_frac(C), [C = simplemind, C = fs]), _).

% Illegal value for the position reached, position named: a number
% kwarg fed an atom.
test(illegal_kwarg_value_names_position,
     error(pe_where(illegal_binding_value(foo, at(kwarg(margin, t)))))) :-
    where_semantic(where(e5(margin(t(T))), [T = foo]), _).

% ...a number-list element fed an atom.
test(illegal_list_element_names_position,
     error(pe_where(illegal_binding_value(fs, at(list_elem(routing, t)))))) :-
    where_semantic(
        where(routing(e5, haiku, t([T]), menus([10])), [T = fs]), _).

% ...an unregistered atom in a positional argument slot.
test(illegal_positional_value_names_position,
     error(pe_where(illegal_binding_value(frobnicate, at(arg(lca_frac, 1)))))) :-
    where_semantic(where(lca_frac(C), [C = frobnicate]), _).

% ...a value illegal at ONE of the several positions the variable
% reaches (here: both mod bases) — checked for every occurrence.
test(illegal_mod_base_names_position,
     error(pe_where(illegal_binding_value(0.5, at(mod_base))))) :-
    where_semantic(where(blend(mod(L, 'D'), mod(L, 'S')), [L = 0.5]), _).

% §12: bindings never reach a pin channel.  A binding variable in a
% pin position is refused regardless of its value.
test(binding_in_pin_position_is_error,
     error(pe_where(binding_reaches_pin_channel(pin_name)))) :-
    where_full(
        where(pin(lineage(pearltrees, decay(0.85)), P), [P = 'run/2026-07-25']),
        _).

% ...and a pin smuggled in AS a binding value is refused too.
test(pin_as_binding_value_is_error,
     error(pe_where(illegal_binding_value(_, at(arg(lca_frac, 1)))))) :-
    where_semantic(
        where(lca_frac(C), [C = pin(simplemind, 'run/1')]), _).

% Malformed binding lists fail closed with the offending element.
test(non_eq_binding_is_error,
     error(pe_where(bad_binding(simplemind)))) :-
    where_semantic(where(lca_frac(_C), [simplemind]), _).

test(bound_lhs_is_error,
     error(pe_where(bad_binding(_)))) :-
    where_semantic(where(lca_frac(simplemind), [fs = simplemind]), _).

test(non_list_bindings_is_error,
     error(pe_where(bad_binding_list(_)))) :-
    where_semantic(where(lca_frac(_C), oops), _).

test(non_where_term_is_error,
     error(pe_where(not_a_where_term(_)))) :-
    where_semantic(lca_frac(simplemind), _).

% Empty binding list over an already-ground goal is legal (a where
% that binds nothing discharges trivially).
test(empty_bindings_ground_goal_ok) :-
    where_semantic(where(lca_frac(simplemind), []), S),
    S == "lca_frac(simplemind)".

:- end_tests(where_fail_closed).
