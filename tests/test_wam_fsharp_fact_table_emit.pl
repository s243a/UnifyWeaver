% test_wam_fsharp_fact_table_emit.pl
%
% T9 fact-table inline for F# — codegen-level checks (no dotnet needed).
% Verifies the value-literal mapping, the inline window gate (default in-range),
% the fact_table_inline(false) opt-out, below-min, and above-cap behaviour.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_fsharp_lowered_emitter').

:- dynamic user:ek/2.
user:ek(a, 1). user:ek(a, 2). user:ek(b, 3).
user:ek(a, 4). user:ek(c, 5). user:ek(b, 6).

lower(Opts, Code) :-
    wam_target:compile_predicate_to_wam(ek/2, [], W),
    lower_predicate_to_fsharp(user:ek/2, W, Opts, lowered(_, _, Code)).

:- begin_tests(wam_fsharp_fact_table_emit).

% value-literal emitter maps each ground term kind to its F# Value variant.
test(value_literals) :-
    emit_fact_table_fsharp(lowered_p_2, 2, [[foo, 7], [bar, -3]], Code),
    assertion(sub_string(Code, _, _, _, "Atom \"foo\"")),
    assertion(sub_string(Code, _, _, _, "Integer (7)")),
    assertion(sub_string(Code, _, _, _, "Integer (-3)")),
    assertion(sub_string(Code, _, _, _, "factTableAttempt args cands")).

% nested term / list / float literals.
test(value_literals_compound) :-
    emit_fact_table_fsharp(lowered_q_1, 1, [[f(a, [1, 2])], [3.5]], Code),
    assertion(sub_string(Code, _, _, _, "Str (\"f\", [Atom \"a\"; VList [Integer (1); Integer (2)]])")),
    assertion(sub_string(Code, _, _, _, "Float (3.5)")).

% default in-range (no inline option): 6 rows in [4,256] -> fact table.
test(default_in_range_classifies) :-
    lower([t9_min_rows(4)], Code),
    assertion(sub_string(Code, _, _, _, "factTableAttempt")),
    assertion(sub_string(Code, _, _, _, "lowered_ek_2_rows")).

% explicit opt-out -> ordinary lowering, no fact table.
test(explicit_disable_off) :-
    lower([t9_min_rows(4), fact_table_inline(false)], Code),
    assertion(\+ sub_string(Code, _, _, _, "factTableAttempt")).

% below t9_min_rows -> not a fact table.
test(below_min_off) :-
    lower([t9_min_rows(100)], Code),
    assertion(\+ sub_string(Code, _, _, _, "factTableAttempt")).

% above t9_max_rows -> not a fact table (steered to an external source).
test(above_cap_off) :-
    lower([t9_min_rows(2), t9_max_rows(5)], Code),
    assertion(\+ sub_string(Code, _, _, _, "factTableAttempt")).

:- end_tests(wam_fsharp_fact_table_emit).
