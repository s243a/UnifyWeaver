% Behavioral pin for wam_cpp_lowerable/3's `multi_clause_1` classification — the
% fallback Reason for a multi-clause predicate that is neither a distinct-first-arg
% clause chain (T5) nor all-clauses-lowerable (T4). Previously unasserted.
% Hand-built instruction list (same style as test_wam_cpp_generator.pl's
% lowerability_* tests) — no WAM compilation, no C++ compiler.

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').

:- begin_tests(wam_cpp_lowered_multi_clause_1).

test(gate_picks_multi_clause_1) :-
    Instrs = [try_me_else("L2"),
              get_variable("X1", "A1"),
              proceed,
              trust_me,
              get_variable("X1", "A1"),
              get_integer(5, "A2")],   % non-terminal supported instr -> T4 declines
    wam_cpp_lowerable(wam_cpp_choice/1, Instrs, Reason),
    assertion(Reason == multi_clause_1).

:- end_tests(wam_cpp_lowered_multi_clause_1).
