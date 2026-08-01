:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_constraint_dispatch.pl - tests for the SECOND pattern_stache
% consumer: constraints as dispatch keys.
%
% Run from this directory:
%   swipl -g run_tests -t halt test_constraint_dispatch.pl
%
% The question these tests exist to answer (report:
% REPORT_pattern_stache_second_consumer.md): do the four pattern
% shapes found by the first consumer still suffice, and does
% constraint dispatch need anything term-shaped that AST emission
% did not?

:- use_module(pattern_stache, [render_stache/3, load_stache_file/2]).
:- use_module(constraint_dispatch,
              [constraint_plan/3, discharge_goal/3, discharge_store/3]).
:- use_module(library(plunit)).

:- dynamic stored_template_path/1.
:- prolog_load_context(directory, Dir),
   atomic_list_concat([Dir, '/constraint_check.stache'], Path),
   assertz(stored_template_path(Path)).

%% ============================================
%% End-to-end: goal store -> discharge ledger
%% ============================================

:- begin_tests(constraint_end_to_end).

test(store_to_ledger) :-
    stored_template_path(Path),
    Goals = [
        has_type(principal_tree(pearltrees), substrate(pearltrees)),
        has_type(t7, substrate(wikipedia)),
        has_type(j1, judge(sonnet)),
        non_amplifying(min),
        non_amplifying(sum),
        owns(s243a, pearltrees),
        mu_bounded(path, 0.5),
        in_support(decay, harvest_2026),
        frobnicate(x)
    ],
    discharge_store(Path, Goals, Ledger),
    Ledger == [
        selected(has_type(principal_tree(pearltrees), substrate(pearltrees)),
                 pt_lineage_walk(principal_tree(pearltrees))),
        selected(has_type(t7, substrate(wikipedia)),
                 substrate_table(t7, wikipedia)),
        selected(has_type(j1, judge(sonnet)),
                 judge_registry(j1, sonnet)),
        discharged(non_amplifying(min)),
        failed(non_amplifying(sum)),
        selected(owns(s243a, pearltrees),
                 ownership_table(s243a, pearltrees)),
        obligation(mu_bounded(path, 0.5),
                   property_test(mu_bounded(path, 0.5))),
        runtime(in_support(decay, harvest_2026),
                corpus_data(in_support(decay, harvest_2026))),
        no_checker(frobnicate(x))
    ].

% The template loads silently: the refinement pair
% (substrate(pearltrees) before substrate(C)) passes the load-time
% overlap check as the specific-before-general idiom, and no case is
% unreachable.
test(template_loads_clean) :-
    stored_template_path(Path),
    load_stache_file(Path, stache(1, _)).

:- end_tests(constraint_end_to_end).

%% ============================================
%% Refinement idiom, now needed by a real consumer
%% ============================================
%
% The first consumer exercised specific-before-general only in tests.
% Here it is load-bearing: pearltrees gets its dedicated lineage-walk
% checker, every other corpus falls through to the generic table.

:- begin_tests(constraint_refinement).

test(specific_corpus_takes_specialized_checker) :-
    stored_template_path(Path),
    load_stache_file(Path, T),
    constraint_plan(T, has_type(t1, substrate(pearltrees)), Plan),
    Plan == checker(pt_lineage_walk(t1), phase(elaboration)).

test(other_corpus_takes_general_checker) :-
    stored_template_path(Path),
    load_stache_file(Path, T),
    constraint_plan(T, has_type(t1, substrate(enwiki)), Plan),
    Plan == checker(substrate_table(t1, enwiki), phase(elaboration)).

:- end_tests(constraint_refinement).

%% ============================================
%% Bindings capturing compound subterms
%% ============================================
%
% New VALUE kind relative to consumer 1 (where every bound leaf was an
% atom or number): here a pattern variable captures a whole compound,
% and the plan term must survive the render -> read round trip.

:- begin_tests(constraint_compound_capture).

test(compound_subterm_round_trips) :-
    stored_template_path(Path),
    load_stache_file(Path, T),
    constraint_plan(T, has_type(principal_tree(pearltrees), substrate(pearltrees)), Plan),
    Plan == checker(pt_lineage_walk(principal_tree(pearltrees)), phase(elaboration)).

:- end_tests(constraint_compound_capture).

%% ============================================
%% The mode split lives in the driver, not the patterns
%% ============================================
%
% "Dispatch only on goals guaranteed discharged": a nonground goal is
% routed to residual/1 by the driver BEFORE the template is consulted.
% This is the reason constraint dispatch needed no guards in the
% pattern language — the one guard-shaped question (groundness) is
% answered by discharge ordering, upstream of dispatch.

:- begin_tests(constraint_mode_split).

test(nonground_routed_to_residual_not_dispatched) :-
    stored_template_path(Path),
    discharge_store(Path, [has_type(_X, substrate(_C))], [Entry]),
    Entry = residual(has_type(_, substrate(_))).

% If a nonground goal DID reach the dispatcher, the dict contract
% refuses it — defense in depth from consumer 1's Q6 answer.
test(dispatcher_still_refuses_nonground,
     error(pattern_stache(nonground_dispatch(constraint, _)))) :-
    stored_template_path(Path),
    load_stache_file(Path, T),
    constraint_plan(T, has_type(_X, substrate(pearltrees)), _).

:- end_tests(constraint_mode_split).

%% ============================================
%% Extensibility: a new constraint form is a new case, not a driver edit
%% ============================================

:- begin_tests(constraint_extensibility).

% Simulate a template extended with a constraint form the driver has
% never heard of.  The plan flows through plan_entry/4 on its phase
% alone; no clause of constraint_dispatch changes.
test(new_constraint_form_needs_no_driver_change) :-
    T = "{{match constraint}}{{case disjoint(A, B)}}checker(overlap_scan({{q:A}}, {{q:B}}), phase(obligation)){{/match}}",
    discharge_goal(T, disjoint(substrate(a), substrate(b)), Entry),
    Entry == obligation(disjoint(substrate(a), substrate(b)),
                        overlap_scan(substrate(a), substrate(b))).

% A template emitting a phase outside the driver protocol fails
% closed with a named error — the phase vocabulary is the CLOSED half
% of the contract, constraint forms are the open half.
test(unknown_phase_fails_closed,
     error(constraint_dispatch(unknown_phase(surprise, _, _)))) :-
    T = "{{match constraint}}{{case f(A)}}checker(x({{q:A}}), phase(surprise)){{/match}}",
    discharge_goal(T, f(a), _).

:- end_tests(constraint_extensibility).

%% ============================================
%% Quoted interpolation: the demand consumer 1 deferred
%% ============================================
%
% Consumer 2 reads its rendered output back as a term, so
% interpolation must be RE-READABLE.  These two tests are the evidence
% pair: plain {{K}} breaks on an atom needing quotes; {{q:K}} round-
% trips it.  This is the one thing constraint dispatch needed that AST
% emission did not — and it is an interpolation form, not a pattern
% shape.

:- begin_tests(constraint_quoting).

test(plain_interpolation_breaks_plan_reading,
     error(constraint_dispatch(unreadable_plan(_, _, syntax_error(_))))) :-
    T = "{{match constraint}}{{case owns(S, C)}}checker(ownership_table({{S}}, {{C}}), phase(elaboration)){{/match}}",
    constraint_plan(T, owns('my account', pearltrees), _).

test(quoted_interpolation_round_trips) :-
    T = "{{match constraint}}{{case owns(S, C)}}checker(ownership_table({{q:S}}, {{q:C}}), phase(elaboration)){{/match}}",
    constraint_plan(T, owns('my account', pearltrees), Plan),
    Plan == checker(ownership_table('my account', pearltrees), phase(elaboration)).

% Quoted interpolation leaves plain atoms and numbers exactly as the
% unquoted form renders them — ~q only adds quotes where re-reading
% needs them.
test(quoted_form_is_invisible_on_plain_atoms) :-
    render_stache("{{match g}}{{case f(A, B)}}{{q:A}}/{{A}} {{q:B}}/{{B}}{{/match}}",
                  [g=f(min, 0.5)], R),
    R == "min/min 0.5/0.5".

:- end_tests(constraint_quoting).
