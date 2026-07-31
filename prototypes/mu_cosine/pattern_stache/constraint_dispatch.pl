:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% constraint_dispatch.pl - PROTOTYPE second consumer for pattern_stache
%
% "Constraints as dispatch keys" — the second consumer named in
% docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md.  A store of
% discharged constraint goals selects each goal's CHECKER by shape via
% constraint_check.stache, instead of an enumerated if/elif chain in
% an elaborator (compare process_expression_vnext/elaborator.py, which
% hard-codes the one goal type it discharges).
%
% Division of labour, and where extensibility lives:
%   - the TEMPLATE enumerates constraint FORMS (open: adding a new
%     constraint form is adding a {{case}}, no driver edit);
%   - this DRIVER enumerates discharge PHASES (closed protocol between
%     template and elaborator: elaboration | obligation | runtime |
%     no_checker), plus the mode split;
%   - the MODE SPLIT (ground vs residual) happens in the driver BEFORE
%     dispatch, per the desugaring doc's rule "dispatch only on goals
%     guaranteed discharged": a nonground goal never reaches the
%     template.  This is why the pattern language needs no guards —
%     the one guard-shaped question (groundness) is owned by the
%     discharge ordering, not by patterns.
%
% What this does NOT claim: dispatch selects the checker; it proves
% nothing.  mu_bounded(path, 0.5) is dispatched to a property-test
% obligation precisely because a numeric constraint is not checkable
% by unification.  Only closed-fact lookup is actually executed here,
% to show the loop closes; everything else lands in the ledger as the
% named checker it selected.

:- module(constraint_dispatch, [
    constraint_plan/3,        % +Template, +Goal, -Plan
    discharge_goal/3,         % +Template, +Goal, -LedgerEntry
    discharge_store/3,        % +TemplatePath, +Goals, -Ledger
    demo/0
]).

:- use_module(pattern_stache, [render_stache/3, load_stache_file/2]).
:- use_module(library(lists)).
:- use_module(library(apply)).

%% The one checker kind executed here: a closed fact table, the
%% "closed-fact constraint remains a lookup, whatever the matcher"
%% case from the philosophy doc.
closed_fact(non_amplifying(min)).
closed_fact(non_amplifying(product)).

%% constraint_plan(+Template, +Goal, -Plan)
%  Dispatch one ground goal through the template and read the rendered
%  line BACK AS A TERM: checker(How, phase(Phase)).  This read-back is
%  what forced {{q:Key}} quoted interpolation into the dispatcher —
%  with plain {{Key}}, a value like 'my account' renders unquoted and
%  the plan line no longer parses (tested).
constraint_plan(Template, Goal, Plan) :-
    render_stache(Template, [constraint=Goal], Rendered),
    normalize_space(string(Line), Rendered),
    catch(
        term_string(Plan, Line),
        error(syntax_error(Why), _),
        throw(error(constraint_dispatch(unreadable_plan(Goal, Line, syntax_error(Why))), _))
    ).

%% discharge_goal(+Template, +Goal, -LedgerEntry)
%  Ledger entries:
%    residual(G)        - nonground: travels with the term, never dispatched
%    discharged(G)      - closed-fact checker ran and succeeded
%    failed(G)          - closed-fact checker ran and the fact is absent
%    selected(G, How)   - elaboration-phase checker chosen; execution
%                         is the elaborator's job, not the dispatcher's
%    obligation(G, How) - discharges once per operator, by property test
%    runtime(G, How)    - needs corpus data; checked at runtime
%    no_checker(G)      - no case matched: fail closed, named error entry
discharge_goal(_Template, Goal, residual(Goal)) :-
    \+ ground(Goal),
    !.
discharge_goal(Template, Goal, Entry) :-
    constraint_plan(Template, Goal, checker(How, phase(Phase))),
    plan_entry(Phase, How, Goal, Entry).

plan_entry(no_checker, _, Goal, no_checker(Goal)) :- !.
plan_entry(elaboration, closed_fact(Fact), Goal, Entry) :-
    !,
    (   closed_fact(Fact)
    ->  Entry = discharged(Goal)
    ;   Entry = failed(Goal)
    ).
plan_entry(elaboration, How, Goal, selected(Goal, How)) :- !.
plan_entry(obligation, How, Goal, obligation(Goal, How)) :- !.
plan_entry(runtime, How, Goal, runtime(Goal, How)) :- !.
plan_entry(Phase, How, Goal, _) :-
    % a template emitting a phase outside the protocol is a template
    % bug; fail closed rather than invent a ledger entry
    throw(error(constraint_dispatch(unknown_phase(Phase, How, Goal)), _)).

%% discharge_store(+TemplatePath, +Goals, -Ledger)
discharge_store(TemplatePath, Goals, Ledger) :-
    load_stache_file(TemplatePath, Template),
    maplist(discharge_goal(Template), Goals, Ledger).

%% demo
%  The store below is the desugaring doc's §3 mode table plus the
%  goals named in the philosophy doc, end to end.  Run:
%    swipl -g constraint_dispatch:demo -t halt constraint_dispatch.pl
demo :-
    module_property(constraint_dispatch, file(Here)),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/constraint_check.stache'], TemplatePath),
    Goals = [
        has_type(principal_tree(pearltrees), substrate(pearltrees)),
        has_type(t7, substrate(wikipedia)),
        has_type(j1, judge(sonnet)),
        non_amplifying(min),
        non_amplifying(sum),
        owns(s243a, pearltrees),
        mu_bounded(path, 0.5),
        in_support(decay, harvest_2026),
        has_type(_X, substrate(_C)),
        frobnicate(x)
    ],
    discharge_store(TemplatePath, Goals, Ledger),
    format("=== constraint_dispatch demo: goal store -> discharge ledger ===~n"),
    forall(
        ( nth1(I, Goals, G), nth1(I, Ledger, E) ),
        format("~q~n    => ~q~n", [G, E])
    ).
