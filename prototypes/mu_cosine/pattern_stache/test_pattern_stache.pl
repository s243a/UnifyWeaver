:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pattern_stache.pl - tests for the pattern_stache prototype
%
% Run from this directory:
%   swipl -g run_tests -t halt test_pattern_stache.pl
%
% The tests are organised by the six open questions from
% docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md ("Is this
% document enough to build from?"), plus the end-to-end worked example
% and a characterization suite that runs the CORE renderer (read-only)
% to establish what actually happens when term-valued dicts hit the
% string path.

:- use_module(pattern_stache).
:- use_module(library(plunit)).

% Locate the worked-example template next to this file (recorded at
% load time, since prolog_load_context/2 is unavailable at run time).
:- dynamic stored_template_path/1.
:- prolog_load_context(directory, Dir),
   atomic_list_concat([Dir, '/goal_to_ast.stache'], Path),
   assertz(stored_template_path(Path)).

% Write Text to a fresh temp file with a .stache extension.
make_stache_file(Text, Path) :-
    tmp_file(pattern_stache_test, Base),
    atom_concat(Base, '.stache', Path),
    setup_call_cleanup(
        open(Path, write, S),
        write(S, Text),
        close(S)).

header("{{! dialect(pattern_stache, 1) }}\n").

%% ============================================
%% End-to-end worked example
%% ============================================

:- begin_tests(worked_example).

test(goals_to_ast_end_to_end) :-
    stored_template_path(Path),
    Goals = [
        has_type(x, substrate(pearltrees)),
        has_type(y, substrate(wikipedia)),
        has_type(j1, judge(sonnet)),
        non_amplifying(min),
        mu_bounded(path, 0.5),
        in_support(decay, d)
    ],
    goals_to_ast(Path, Goals, Lines),
    Lines == [
        "TypeNode { term: x, kind: \"substrate\", corpus: pearltrees }",
        "TypeNode { term: y, kind: \"substrate\", corpus: wikipedia }",
        "TypeNode { term: j1, kind: \"judge\", judge: sonnet }",
        "ConstraintNode { kind: \"non_amplifying\", op: min }",
        "ConstraintNode { kind: \"mu_bounded\", op: path, bound: 0.5 }",
        "/* unhandled goal: in_support(decay,d) */"
    ].

% The default body interpolates the WHOLE dict value ({{goal}}) as a
% term — the "unreachable with string matching" block from the doc.
test(default_interpolates_term) :-
    stored_template_path(Path),
    goals_to_ast(Path, [weird(nested(term), [1,2])], [L]),
    L == "/* unhandled goal: weird(nested(term),[1,2]) */".

:- end_tests(worked_example).

%% ============================================
%% Q1: binding propagation and shadowing
%% ============================================

:- begin_tests(q1_bindings).

% Pattern variables become dict keys visible to the body.
test(bindings_reach_body) :-
    render_stache("{{match g}}{{case f(A, B)}}a={{A}} b={{B}}{{/match}}",
                  [g=f(1, two)], R),
    R == "a=1 b=two".

% A binding SHADOWS an outer dict key of the same name inside the body...
test(binding_shadows_outer_key) :-
    render_stache("{{match g}}{{case f(C)}}inner={{C}}{{/match}}",
                  [g=f(bound_by_match), 'C'=from_outer_dict], R),
    R == "inner=bound_by_match".

% ...and the shadow is scoped to the body: after {{/match}} the outer
% key is visible again, untouched.
test(shadow_does_not_leak_past_block) :-
    render_stache("{{match g}}{{case f(C)}}in:{{C}}{{/match}} out:{{C}}",
                  [g=f(inner_val), 'C'=outer_val], R),
    R == "in:inner_val out:outer_val".

% Outer dict keys remain visible inside the body (child scope EXTENDS,
% it does not replace).
test(outer_keys_visible_in_body) :-
    render_stache("{{match g}}{{case f(A)}}{{A}}/{{other}}{{/match}}",
                  [g=f(x), other=kept], R),
    R == "x/kept".

% Nested match: inner block sees the outer block's bindings.
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

% `_` is a true wildcard: it matches but produces NO binding.
test(anonymous_var_binds_nothing) :-
    read_case_pattern('f(_, K)', _P, Vs),
    Vs = ['K'=_],
    length(Vs, 1).

test(anonymous_var_matches) :-
    render_stache("{{match g}}{{case f(_, K)}}k={{K}}{{/match}}",
                  [g=f(ignored, kept)], R),
    R == "k=kept".

% Standard operator table applies: infix patterns read normally.
test(standard_operators_available) :-
    read_case_pattern('A-B', P, ['A'=X, 'B'=Y]),
    P == X-Y.

test(operator_pattern_dispatch) :-
    render_stache("{{match g}}{{case A-B}}{{A}} minus {{B}}{{/match}}",
                  [g = 3-4], R),
    R == "3 minus 4".

% Quoted atoms restore literal semantics for values that would
% otherwise misparse (the migration checklist's hyphen row).
test(quoted_hyphenated_literal) :-
    render_stache("{{match t}}{{case 'wam-fsharp'}}FS{{/match}}",
                  [t='wam-fsharp'], R),
    R == "FS".

% A syntax error in a case pattern is a NAMED error, not a silent skip.
test(bad_pattern_is_error, error(pattern_stache(bad_case_pattern(_, syntax_error(_))))) :-
    render_stache("{{match g}}{{case f(}}body{{/match}}", [g=x], _).

:- end_tests(q2_term_reading).

%% ============================================
%% Q3: overlap detection at load
%% ============================================

:- begin_tests(q3_overlap).

% General before specific: the specific case can never fire -> load error.
test(unreachable_case_is_load_error,
     error(pattern_stache(unreachable_case(case_index(2), subsumed_by(case_index(1)), _)))) :-
    header(H),
    string_concat(H, "{{match g}}{{case f(X)}}general {{X}}{{case f(a)}}specific{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, _).

% Identical patterns (variants of each other) are also unreachable.
test(duplicate_case_is_load_error,
     error(pattern_stache(unreachable_case(_, _, _)))) :-
    header(H),
    string_concat(H, "{{match g}}{{case f(X)}}one{{case f(Y)}}two{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, _).

% Specific before general is the refinement idiom: loads silently,
% and both orderings of input dispatch correctly.
test(refinement_idiom_allowed) :-
    header(H),
    string_concat(H, "{{match g}}{{case f(a)}}SPECIFIC{{case f(X)}}GENERAL {{X}}{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, Template),
    render_stache(Template, [g=f(a)], R1),
    render_stache(Template, [g=f(zzz)], R2),
    R1 == "SPECIFIC",
    R2 == "GENERAL zzz".

% Partial overlap (neither subsumes): allowed, warned on stderr,
% first match wins.
test(partial_overlap_first_match_wins) :-
    header(H),
    string_concat(H, "{{match g}}{{case p(a, Y)}}FIRST {{Y}}{{case p(X, b)}}SECOND {{X}}{{/match}}", T),
    make_stache_file(T, Path),
    load_stache_file(Path, Template),
    render_stache(Template, [g=p(a, b)], R),    % matches both; first committed
    R == "FIRST b".

% Overlap checking reaches NESTED match blocks at load too.
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

% Once a case commits, a body whose nested match produces nothing does
% NOT cause reconsideration of a later case that would also match.
test(no_backtracking_into_case_selection) :-
    render_stache(
        "{{match g}}{{case p(a, Y)}}[{{match Y}}{{case zzz}}never{{/match}}]{{case p(X, b)}}SECOND{{/match}}",
        [g=p(a, b)], R),
    % first case commits; its inner match has no matching case and no
    % default, so it renders empty — output is the first body, not SECOND
    R == "[]".

% Once a case commits, an error raised while rendering its body
% propagates; the dispatcher does not fall through to the next case.
% (Rendered from a string so the load-time check cannot pre-empt the
% render-time error.)
test(body_error_propagates_no_fallthrough,
     error(pattern_stache(bad_case_pattern(_, _)))) :-
    render_stache(
        "{{match g}}{{case p(a, Y)}}{{match Y}}{{case f(}}x{{/match}}{{case p(X, b)}}SECOND{{/match}}",
        [g=p(a, b)], _).

% No match and no default renders empty (same convention as core).
test(no_match_no_default_is_empty) :-
    render_stache("{{match g}}{{case f(X)}}body{{/match}}", [g=other], R),
    R == "".

% Missing key falls to default with no bindings.
test(missing_key_uses_default) :-
    render_stache("{{match nope}}{{case f(X)}}body{{default}}DEF{{/match}}", [], R),
    R == "DEF".

:- end_tests(q4_determinism).

%% ============================================
%% Q5: loader dispatch — extension + header, fail closed
%% ============================================

:- begin_tests(q5_loader).

test(wrong_extension_rejected, error(pattern_stache(not_a_stache_file(_)))) :-
    tmp_file(pattern_stache_test, Base),
    atom_concat(Base, '.mustache', Path),
    setup_call_cleanup(open(Path, write, S), write(S, "x"), close(S)),
    load_stache_file(Path, _).

test(headerless_stache_rejected, error(pattern_stache(missing_dialect_header(_)))) :-
    make_stache_file("{{match g}}{{case a}}b{{/match}}", Path),
    load_stache_file(Path, _).

test(unknown_version_rejected, error(pattern_stache(unsupported_dialect_version(_, 2)))) :-
    make_stache_file("{{! dialect(pattern_stache, 2) }}\nbody", Path),
    load_stache_file(Path, _).

% Header must be the FIRST non-empty line: a pragma later in the file
% does not count.
test(late_pragma_not_a_header, error(pattern_stache(missing_dialect_header(_)))) :-
    make_stache_file("some text\n{{! dialect(pattern_stache, 1) }}\n", Path),
    load_stache_file(Path, _).

% Blank lines before the header are tolerated.
test(blank_lines_before_header_ok) :-
    make_stache_file("\n  \n{{! dialect(pattern_stache, 1) }}\nhello {{w}}", Path),
    load_stache_file(Path, Template),
    render_stache(Template, [w=world], R),
    R == "hello world".

test(load_and_render_roundtrip) :-
    stored_template_path(Path),
    load_stache_file(Path, Template),
    Template = stache(1, _).

:- end_tests(q5_loader).

%% ============================================
%% Q6: dict contract with term values
%% ============================================

:- begin_tests(q6_dict_contract).

% Legacy-style atom dicts against literal atom cases work unchanged.
test(atom_values_still_work) :-
    render_stache("{{match mode}}{{case cached}}C{{case eager}}E{{/match}}",
                  [mode=cached], R),
    R == "C".

% Term values dispatch structurally.
test(term_values_dispatch) :-
    render_stache("{{match g}}{{case s(K)}}{{K}}{{/match}}", [g=s(hit)], R),
    R == "hit".

% Numbers survive matching and interpolation.
test(numeric_leaf_values) :-
    render_stache("{{match g}}{{case bound(B)}}b={{B}}{{/match}}",
                  [g=bound(0.5)], R),
    R == "b=0.5".

% A NONGROUND dict value is refused: dispatch may not bind the data.
test(nonground_dispatch_is_error,
     error(pattern_stache(nonground_dispatch(g, _)))) :-
    render_stache("{{match g}}{{case f(X)}}{{X}}{{/match}}", [g=f(_Free)], _).

% Unknown placeholders are left VERBATIM, not erased (same as core).
test(unknown_placeholder_left_verbatim) :-
    render_stache("a {{missing}} b", [], R),
    R == "a {{missing}} b".

:- end_tests(q6_dict_contract).

%% ============================================
%% Characterization: what CORE actually does with term values
%% ============================================
%
% Read-only use of src/unifyweaver/core/template_system.pl to put the
% report's compatibility claims on record.  Nothing here modifies core.

:- use_module('../../../src/unifyweaver/core/template_system', [render_template/3]).

:- begin_tests(core_characterization).

% Core's resolve_match/5 stringifies dict values with atom_string/2,
% which THROWS type_error(atom, _) on a compound.  So feeding a
% term-valued dict to the legacy string path fails LOUDLY at the first
% match block — it does not silently mis-render.
test(core_match_throws_on_compound_dict_value, error(type_error(atom, _))) :-
    template_system:render_template(
        "{{match goal}}{{case x}}b{{/match}}",
        [goal=has_type(x, substrate(pearltrees))], _).

% Core substitution ALSO throws on compound values (render_template_string
% uses atom_string/2 on every value) — even without a match block.
test(core_substitution_throws_on_compound_value, error(type_error(atom, _))) :-
    template_system:render_template("v={{v}}", [v=f(a)], _).

% Core leaves unknown placeholders verbatim — NOT the empty string the
% mustache spec (and the philosophy doc's hazard section) would predict.
test(core_leaves_unknown_placeholder_verbatim) :-
    template_system:render_template("corpus is {{C}}", [], R),
    R == "corpus is {{C}}".

% Consequence of the two facts above, on record: a structural template
% fed to the string parser with an ATOM-valued dict does not throw; the
% pattern case simply never matches (its text is not string-equal), the
% default is taken, and pattern variables in it survive as literal
% {{C}} text rather than becoming empty holes.
test(core_string_path_on_structural_template) :-
    template_system:render_template(
        "{{match goal}}{{case substrate(C)}}corpus is {{C}}{{default}}fallback {{C}}{{/match}}",
        [goal=other], R),
    R == "fallback {{C}}".

:- end_tests(core_characterization).
