:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pattern_stache.pl - tests for the production pattern_stache engine
% (src/unifyweaver/core/pattern_stache.pl, dialect version 1).
%
% Run: swipl -g run_tests -t halt tests/core/test_pattern_stache.pl
%
% Contents, in order:
%   - the 38 dispatcher tests ported from the prototype
%     (prototypes/mu_cosine/pattern_stache/test_pattern_stache.pl),
%     organized by the six questions they answered;
%   - the 12 consumer tests ported from the second consumer
%     (test_constraint_dispatch.pl), with the consumer's small driver
%     inlined here as test-local helpers — the phase protocol is
%     consumer-level, so the production engine deliberately ships no
%     driver;
%   - the migration checklist from
%     docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md as tests:
%     the silent misparse rows are why the checklist exists;
%   - the linearity boundary new to the production engine (SPEC,
%     deliberate exclusions).
%
% Nested-{{match}} tests are ported intact and pass, but nesting is
% outside the v1 contract — they characterize implementation
% behaviour, they do not promise it (SPEC, deliberate exclusions).

:- use_module('../../src/unifyweaver/core/pattern_stache').
:- use_module(library(plunit)).

% Write Text to a fresh temp file with the given extension.
make_file(Text, Ext, Path) :-
    tmp_file(pattern_stache_test, Base),
    atom_concat(Base, Ext, Path),
    setup_call_cleanup(
        open(Path, write, S),
        write(S, Text),
        close(S)).

make_stache_file(Text, Path) :- make_file(Text, '.stache', Path).

header("{{! dialect(pattern_stache, 1) }}\n").

% The goals->AST worked example (first witnessed consumer), as a
% fixture written at test time so the prototype directory stays the
% untouched historical witness.
goal_to_ast_template(T) :-
    header(H),
    string_concat(H,
"{{match goal}}
{{case has_type(X, substrate(C))}}TypeNode { term: {{X}}, kind: \"substrate\", corpus: {{C}} }
{{case has_type(X, judge(J))}}TypeNode { term: {{X}}, kind: \"judge\", judge: {{J}} }
{{case non_amplifying(Op)}}ConstraintNode { kind: \"non_amplifying\", op: {{Op}} }
{{case mu_bounded(Op, B)}}ConstraintNode { kind: \"mu_bounded\", op: {{Op}}, bound: {{B}} }
{{default}}/* unhandled goal: {{goal}} */
{{/match}}
", T).

goals_to_ast(Template, Goals, Lines) :-
    maplist(goal_to_ast_line(Template), Goals, Lines).

goal_to_ast_line(Template, Goal, Line) :-
    render_stache(Template, [goal=Goal], Rendered),
    normalize_space(string(Line), Rendered).

%% ============================================
%% End-to-end worked example (consumer 1)
%% ============================================

:- begin_tests(worked_example).

test(goals_to_ast_end_to_end) :-
    goal_to_ast_template(Text),
    make_stache_file(Text, Path),
    load_stache_file(Path, Template),
    Goals = [
        has_type(x, substrate(pearltrees)),
        has_type(y, substrate(wikipedia)),
        has_type(j1, judge(sonnet)),
        non_amplifying(min),
        mu_bounded(path, 0.5),
        in_support(decay, d)
    ],
    goals_to_ast(Template, Goals, Lines),
    Lines == [
        "TypeNode { term: x, kind: \"substrate\", corpus: pearltrees }",
        "TypeNode { term: y, kind: \"substrate\", corpus: wikipedia }",
        "TypeNode { term: j1, kind: \"judge\", judge: sonnet }",
        "ConstraintNode { kind: \"non_amplifying\", op: min }",
        "ConstraintNode { kind: \"mu_bounded\", op: path, bound: 0.5 }",
        "/* unhandled goal: in_support(decay,d) */"
    ].

test(default_interpolates_term) :-
    goal_to_ast_template(Text),
    make_stache_file(Text, Path),
    load_stache_file(Path, Template),
    goals_to_ast(Template, [weird(nested(term), [1,2])], [L]),
    L == "/* unhandled goal: weird(nested(term),[1,2]) */".

:- end_tests(worked_example).

%% ============================================
%% Q1: binding propagation and shadowing
%% ============================================

:- begin_tests(q1_bindings).

test(bindings_reach_body) :-
    render_stache("{{match g}}{{case f(A, B)}}a={{A}} b={{B}}{{/match}}",
                  [g=f(1, two)], R),
    R == "a=1 b=two".

test(binding_shadows_outer_key) :-
    render_stache("{{match g}}{{case f(C)}}inner={{C}}{{/match}}",
                  [g=f(bound_by_match), 'C'=from_outer_dict], R),
    R == "inner=bound_by_match".

test(shadow_does_not_leak_past_block) :-
    render_stache("{{match g}}{{case f(C)}}in:{{C}}{{/match}} out:{{C}}",
                  [g=f(inner_val), 'C'=outer_val], R),
    R == "in:inner_val out:outer_val".

test(outer_keys_visible_in_body) :-
    render_stache("{{match g}}{{case f(A)}}{{A}}/{{other}}{{/match}}",
                  [g=f(x), other=kept], R),
    R == "x/kept".

% Characterizes nested-match behaviour; nesting itself is outside the
% v1 contract (SPEC, deliberate exclusions).
test(nested_match_sees_outer_bindings) :-
    render_stache(
        "{{match g}}{{case f(Inner)}}{{match Inner}}{{case sub(K)}}{{K}}{{/match}}{{/match}}",
        [g=f(sub(deep))], R),
    R == "deep".

:- end_tests(q1_bindings).

%% ============================================
%% Q2: term reading of case patterns
%% ============================================

:- begin_tests(q2_term_reading).

test(pattern_with_variable_names) :-
    read_case_pattern('has_type(X, substrate(C))', P, Vs),
    P = has_type(V1, substrate(V2)),
    Vs == ['X'=V1, 'C'=V2].

test(anonymous_var_binds_nothing) :-
    read_case_pattern('f(_, K)', _P, Vs),
    Vs = ['K'=_],
    length(Vs, 1).

test(anonymous_var_matches) :-
    render_stache("{{match g}}{{case f(_, K)}}k={{K}}{{/match}}",
                  [g=f(ignored, kept)], R),
    R == "k=kept".

test(standard_operators_available) :-
    read_case_pattern('A-B', P, ['A'=X, 'B'=Y]),
    P == X-Y.

test(operator_pattern_dispatch) :-
    render_stache("{{match g}}{{case A-B}}{{A}} minus {{B}}{{/match}}",
                  [g = 3-4], R),
    R == "3 minus 4".

test(quoted_hyphenated_literal) :-
    render_stache("{{match t}}{{case 'wam-fsharp'}}FS{{/match}}",
                  [t='wam-fsharp'], R),
    R == "FS".

test(bad_pattern_is_error, error(pattern_stache(bad_case_pattern(_, syntax_error(_))))) :-
    render_stache("{{match g}}{{case f(}}body{{/match}}", [g=x], _).

:- end_tests(q2_term_reading).

%% ============================================
%% Q3: overlap detection at load
%% ============================================

:- begin_tests(q3_overlap).

test(unreachable_case_is_load_error,
     error(pattern_stache(unreachable_case(case_index(2), subsumed_by(case_index(1)), _)))) :-
    header(H),
    string_concat(H, "{{match g}}{{case f(X)}}general {{X}}{{case f(a)}}specific{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, _).

test(duplicate_case_is_load_error,
     error(pattern_stache(unreachable_case(_, _, _)))) :-
    header(H),
    string_concat(H, "{{match g}}{{case f(X)}}one{{case f(Y)}}two{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, _).

test(refinement_idiom_allowed) :-
    header(H),
    string_concat(H, "{{match g}}{{case f(a)}}SPECIFIC{{case f(X)}}GENERAL {{X}}{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, Template),
    render_stache(Template, [g=f(a)], R1),
    render_stache(Template, [g=f(zzz)], R2),
    R1 == "SPECIFIC",
    R2 == "GENERAL zzz".

test(partial_overlap_first_match_wins) :-
    header(H),
    string_concat(H, "{{match g}}{{case p(a, Y)}}FIRST {{Y}}{{case p(X, b)}}SECOND {{X}}{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, Template),
    render_stache(Template, [g=p(a, b)], R),
    R == "FIRST b".

test(nested_unreachable_detected_at_load,
     error(pattern_stache(unreachable_case(_, _, _)))) :-
    header(H),
    string_concat(H,
        "{{match g}}{{case f(I)}}{{match I}}{{case s(X)}}g{{case s(a)}}dead{{/match}}{{/match}}",
        T),
    make_stache_file(T, Path),
    load_stache_file(Path, _).

:- end_tests(q3_overlap).

%% ============================================
%% Q4: determinism — commit at case selection
%% ============================================

:- begin_tests(q4_determinism).

test(no_backtracking_into_case_selection) :-
    render_stache(
        "{{match g}}{{case p(a, Y)}}[{{match Y}}{{case zzz}}never{{/match}}]{{case p(X, b)}}SECOND{{/match}}",
        [g=p(a, b)], R),
    R == "[]".

test(body_error_propagates_no_fallthrough,
     error(pattern_stache(bad_case_pattern(_, _)))) :-
    render_stache(
        "{{match g}}{{case p(a, Y)}}{{match Y}}{{case f(}}x{{/match}}{{case p(X, b)}}SECOND{{/match}}",
        [g=p(a, b)], _).

test(no_match_no_default_is_empty) :-
    render_stache("{{match g}}{{case f(X)}}body{{/match}}", [g=other], R),
    R == "".

test(missing_key_uses_default) :-
    render_stache("{{match nope}}{{case f(X)}}body{{default}}DEF{{/match}}", [], R),
    R == "DEF".

:- end_tests(q4_determinism).

%% ============================================
%% Q5: loader dispatch — extension + header, fail closed
%% ============================================

:- begin_tests(q5_loader).

test(wrong_extension_rejected, error(pattern_stache(not_a_stache_file(_)))) :-
    make_file("x", '.mustache', Path),
    load_stache_file(Path, _).

test(headerless_stache_rejected, error(pattern_stache(missing_dialect_header(_)))) :-
    make_stache_file("{{match g}}{{case a}}b{{/match}}", Path),
    load_stache_file(Path, _).

test(unknown_version_rejected, error(pattern_stache(unsupported_dialect_version(_, 2)))) :-
    make_stache_file("{{! dialect(pattern_stache, 2) }}\nbody", Path),
    load_stache_file(Path, _).

test(late_pragma_not_a_header, error(pattern_stache(missing_dialect_header(_)))) :-
    make_stache_file("some text\n{{! dialect(pattern_stache, 1) }}\n", Path),
    load_stache_file(Path, _).

test(blank_lines_before_header_ok) :-
    make_stache_file("\n  \n{{! dialect(pattern_stache, 1) }}\nhello {{w}}", Path),
    load_stache_file(Path, Template),
    render_stache(Template, [w=world], R),
    R == "hello world".

test(load_and_render_roundtrip) :-
    goal_to_ast_template(Text),
    make_stache_file(Text, Path),
    load_stache_file(Path, Template),
    Template = stache(1, _).

:- end_tests(q5_loader).

%% ============================================
%% Q6: dict contract with term values
%% ============================================

:- begin_tests(q6_dict_contract).

test(atom_values_still_work) :-
    render_stache("{{match mode}}{{case cached}}C{{case eager}}E{{/match}}",
                  [mode=cached], R),
    R == "C".

test(term_values_dispatch) :-
    render_stache("{{match g}}{{case s(K)}}{{K}}{{/match}}", [g=s(hit)], R),
    R == "hit".

test(numeric_leaf_values) :-
    render_stache("{{match g}}{{case bound(B)}}b={{B}}{{/match}}",
                  [g=bound(0.5)], R),
    R == "b=0.5".

test(nonground_dispatch_is_error,
     error(pattern_stache(nonground_dispatch(g, _)))) :-
    render_stache("{{match g}}{{case f(X)}}{{X}}{{/match}}", [g=f(_Free)], _).

test(unknown_placeholder_left_verbatim) :-
    render_stache("a {{missing}} b", [], R),
    R == "a {{missing}} b".

:- end_tests(q6_dict_contract).

%% ============================================
%% Characterization: the string dialect on term values (read-only)
%% ============================================
%
% Read-only use of template_system.pl to keep the compatibility claims
% of the SPEC's conflation table on record.  Nothing here modifies the
% string dialect.

:- use_module('../../src/unifyweaver/core/template_system', [render_template/3]).

:- begin_tests(core_characterization).

test(core_match_throws_on_compound_dict_value, error(type_error(atom, _))) :-
    template_system:render_template(
        "{{match goal}}{{case x}}b{{/match}}",
        [goal=has_type(x, substrate(pearltrees))], _).

test(core_substitution_throws_on_compound_value, error(type_error(atom, _))) :-
    template_system:render_template("v={{v}}", [v=f(a)], _).

test(core_leaves_unknown_placeholder_verbatim) :-
    template_system:render_template("corpus is {{C}}", [], R),
    R == "corpus is {{C}}".

test(core_string_path_on_structural_template) :-
    template_system:render_template(
        "{{match goal}}{{case substrate(C)}}corpus is {{C}}{{default}}fallback {{C}}{{/match}}",
        [goal=other], R),
    R == "fallback {{C}}".

:- end_tests(core_characterization).

%% ============================================
%% Consumer 2: constraints as dispatch keys
%% ============================================
%
% Ported from the prototype's test_constraint_dispatch.pl.  The
% consumer's driver (mode split, plan read-back, phase protocol,
% closed-fact execution) is inlined below as TEST-LOCAL helpers: the
% phase vocabulary is a consumer contract, not engine behaviour, so
% the production module ships no driver.

closed_fact(non_amplifying(min)).
closed_fact(non_amplifying(product)).

constraint_template(T) :-
    header(H),
    string_concat(H,
"{{match constraint}}
{{case has_type(X, substrate(pearltrees))}}checker(pt_lineage_walk({{q:X}}), phase(elaboration))
{{case has_type(X, substrate(C))}}checker(substrate_table({{q:X}}, {{q:C}}), phase(elaboration))
{{case has_type(X, judge(J))}}checker(judge_registry({{q:X}}, {{q:J}}), phase(elaboration))
{{case non_amplifying(Op)}}checker(closed_fact(non_amplifying({{q:Op}})), phase(elaboration))
{{case owns(S, C)}}checker(ownership_table({{q:S}}, {{q:C}}), phase(elaboration))
{{case mu_bounded(Op, B)}}checker(property_test(mu_bounded({{q:Op}}, {{q:B}})), phase(obligation))
{{case in_support(K, V)}}checker(corpus_data(in_support({{q:K}}, {{q:V}})), phase(runtime))
{{default}}checker(none, phase(no_checker))
{{/match}}
", T).

load_constraint_template(Template) :-
    constraint_template(Text),
    make_stache_file(Text, Path),
    load_stache_file(Path, Template).

constraint_plan(Template, Goal, Plan) :-
    render_stache(Template, [constraint=Goal], Rendered),
    normalize_space(string(Line), Rendered),
    catch(
        term_string(Plan, Line),
        error(syntax_error(Why), _),
        throw(error(consumer_test(unreadable_plan(Goal, Line, syntax_error(Why))), _))
    ).

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
    throw(error(consumer_test(unknown_phase(Phase, How, Goal)), _)).

discharge_store(Template, Goals, Ledger) :-
    maplist(discharge_goal(Template), Goals, Ledger).

:- begin_tests(constraint_end_to_end).

test(store_to_ledger) :-
    load_constraint_template(Template),
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
    discharge_store(Template, Goals, Ledger),
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

test(template_loads_clean) :-
    load_constraint_template(stache(1, _)).

:- end_tests(constraint_end_to_end).

:- begin_tests(constraint_refinement).

test(specific_corpus_takes_specialized_checker) :-
    load_constraint_template(T),
    constraint_plan(T, has_type(t1, substrate(pearltrees)), Plan),
    Plan == checker(pt_lineage_walk(t1), phase(elaboration)).

test(other_corpus_takes_general_checker) :-
    load_constraint_template(T),
    constraint_plan(T, has_type(t1, substrate(enwiki)), Plan),
    Plan == checker(substrate_table(t1, enwiki), phase(elaboration)).

:- end_tests(constraint_refinement).

:- begin_tests(constraint_compound_capture).

test(compound_subterm_round_trips) :-
    load_constraint_template(T),
    constraint_plan(T, has_type(principal_tree(pearltrees), substrate(pearltrees)), Plan),
    Plan == checker(pt_lineage_walk(principal_tree(pearltrees)), phase(elaboration)).

:- end_tests(constraint_compound_capture).

:- begin_tests(constraint_mode_split).

test(nonground_routed_to_residual_not_dispatched) :-
    load_constraint_template(T),
    discharge_store(T, [has_type(_X, substrate(_C))], [Entry]),
    Entry = residual(has_type(_, substrate(_))).

test(dispatcher_still_refuses_nonground,
     error(pattern_stache(nonground_dispatch(constraint, _)))) :-
    load_constraint_template(T),
    constraint_plan(T, has_type(_X, substrate(pearltrees)), _).

:- end_tests(constraint_mode_split).

:- begin_tests(constraint_extensibility).

test(new_constraint_form_needs_no_driver_change) :-
    T = "{{match constraint}}{{case disjoint(A, B)}}checker(overlap_scan({{q:A}}, {{q:B}}), phase(obligation)){{/match}}",
    discharge_goal(T, disjoint(substrate(a), substrate(b)), Entry),
    Entry == obligation(disjoint(substrate(a), substrate(b)),
                        overlap_scan(substrate(a), substrate(b))).

test(unknown_phase_fails_closed,
     error(consumer_test(unknown_phase(surprise, _, _)))) :-
    T = "{{match constraint}}{{case f(A)}}checker(x({{q:A}}), phase(surprise)){{/match}}",
    discharge_goal(T, f(a), _).

:- end_tests(constraint_extensibility).

:- begin_tests(constraint_quoting).

test(plain_interpolation_breaks_plan_reading,
     error(consumer_test(unreadable_plan(_, _, syntax_error(_))))) :-
    T = "{{match constraint}}{{case owns(S, C)}}checker(ownership_table({{S}}, {{C}}), phase(elaboration)){{/match}}",
    constraint_plan(T, owns('my account', pearltrees), _).

test(quoted_interpolation_round_trips) :-
    T = "{{match constraint}}{{case owns(S, C)}}checker(ownership_table({{q:S}}, {{q:C}}), phase(elaboration)){{/match}}",
    constraint_plan(T, owns('my account', pearltrees), Plan),
    Plan == checker(ownership_table('my account', pearltrees), phase(elaboration)).

test(quoted_form_is_invisible_on_plain_atoms) :-
    render_stache("{{match g}}{{case f(A, B)}}{{q:A}}/{{A}} {{q:B}}/{{B}}{{/match}}",
                  [g=f(min, 0.5)], R),
    R == "min/min 0.5/0.5".

:- end_tests(constraint_quoting).

%% ============================================
%% Migration checklist (STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md)
%% ============================================
%
% The checklist's dangerous rows are the SILENT ones: they parse, so
% no error surfaces, and only review or tooling can catch them.  These
% tests keep each row's behaviour on record against the production
% reader.

:- begin_tests(migration_checklist).

% Row: lowercase atom — reads as the same atom, no action needed.
test(lowercase_atom_unchanged) :-
    read_case_pattern(helpers, P, []),
    P == helpers.

% Row: hyphenated, quoted — parses as the atom, matches the atom.
test(quoted_hyphenated_parses_as_atom) :-
    read_case_pattern('\'wam-fsharp\'', P, []),
    P == 'wam-fsharp'.

% Row: hyphenated, bare — SILENTLY reads as compound -(wam, fsharp).
% The read succeeds; nothing complains.
test(bare_hyphenated_becomes_compound_silently) :-
    read_case_pattern('wam-fsharp', P, []),
    P == wam-fsharp,
    compound(P),
    P = -(wam, fsharp).

% ...and the observable failure: the case no longer matches the atom
% it visually spells.  Dispatch falls through to the default.
test(bare_hyphenated_case_misses_the_atom) :-
    render_stache("{{match t}}{{case wam-fsharp}}HIT{{default}}MISS{{/match}}",
                  [t='wam-fsharp'], R),
    R == "MISS".

% Row: other operator chars — same silent compound misparse.
test(three_way_becomes_compound_silently) :-
    read_case_pattern('3-way', P, []),
    P == 3-way.

% Row: uppercase-initial, bare — SILENTLY reads as a variable, which
% unifies with anything...
test(bare_uppercase_reads_as_variable_silently) :-
    read_case_pattern('Helpers', P, Vs),
    var(P),
    Vs = ['Helpers'=P].

% ...so as a single case it swallows every input, silently.
test(bare_uppercase_swallows_everything) :-
    render_stache("{{match t}}{{case Helpers}}SWALLOWED {{Helpers}}{{default}}D{{/match}}",
                  [t=anything_at_all], R),
    R == "SWALLOWED anything_at_all".

% ...but the overlap trichotomy catches it AT LOAD the moment any case
% follows it: a variable pattern subsumes everything, so the next case
% is unreachable.  The checklist hazard is fully silent only in a
% single-case block.
test(bare_uppercase_with_following_case_is_load_error,
     error(pattern_stache(unreachable_case(case_index(2), subsumed_by(case_index(1)), _)))) :-
    header(H),
    string_concat(H, "{{match t}}{{case Helpers}}A{{case helpers}}B{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, _).

% Row: uppercase-initial, quoted — parses as the atom, matches it.
test(quoted_uppercase_parses_as_atom) :-
    read_case_pattern('\'Helpers\'', P, []),
    P == 'Helpers',
    render_stache("{{match t}}{{case 'Helpers'}}HIT{{/match}}", [t='Helpers'], R),
    R == "HIT".

% Row: contains a space — fails LOUDLY; the read raises.
test(space_fails_loudly,
     error(pattern_stache(bad_case_pattern(_, syntax_error(_))))) :-
    read_case_pattern('a b', _, _).

:- end_tests(migration_checklist).

%% ============================================
%% Linearity: the v1 contract boundary is enforced
%% ============================================

:- begin_tests(linearity).

test(nonlinear_pattern_is_load_error,
     error(pattern_stache(nonlinear_pattern(_)))) :-
    read_case_pattern('f(X, X)', _, _).

test(nonlinear_deep_is_load_error,
     error(pattern_stache(nonlinear_pattern(_)))) :-
    read_case_pattern('g(X, h(X))', _, _).

test(nonlinear_rejected_in_template,
     error(pattern_stache(nonlinear_pattern(_)))) :-
    header(H),
    string_concat(H, "{{match g}}{{case f(X, X)}}same{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, _).

% Repeated `_` is NOT non-linear: each occurrence is a fresh variable.
test(repeated_wildcard_is_fine) :-
    read_case_pattern('f(_, _)', P, Vs),
    P = f(A, B),
    A \== B,
    Vs == [].

% Distinct variables everywhere remain fine, of course.
test(linear_pattern_unaffected) :-
    read_case_pattern('f(X, g(Y), Z)', _, Vs),
    length(Vs, 3).

:- end_tests(linearity).
