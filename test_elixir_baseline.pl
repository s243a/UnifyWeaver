:- use_module(src/unifyweaver/targets/wam_elixir_target).
:- use_module(src/unifyweaver/targets/wam_target).
:- use_module(library(lists)).
:- use_module(library(readutil)).

:- dynamic test_sum/2.
test_sum(0, 0).
test_sum(N, S) :- N > 0, N1 is N - 1, test_sum(N1, S1), S is S1 + N.

:- dynamic test_structure/2.
test_structure(foo(X, Y), bar(Y, X)).

main :-
    % Generate WAM code for test_sum/2 and test_structure/2
    wam_target:compile_predicate_to_wam(test_sum/2, [], WamCodeSum),
    wam_target:compile_predicate_to_wam(test_structure/2, [], WamCodeStruct),
    Predicates = [test_sum/2-WamCodeSum, test_structure/2-WamCodeStruct],
    Options = [module_name(sum_bench), emit_mode(lowered)],
    write_wam_elixir_project(Predicates, Options, 'benchmarks/elixir_baseline'),
    write('Generated Elixir baseline project in benchmarks/elixir_baseline'), nl,
    
    % Verify that the structure operations were lowered correctly (not stubs)
    read_file_to_string('benchmarks/elixir_baseline/lib/test_structure.ex', StructCode, []),
    (   sub_string(StructCode, _, _, _, 'WamRuntime.step_get_structure_ref')
    ->  write('get_structure lowered successfully.'), nl
    ;   write('FAILED: get_structure not lowered!'), nl, halt(1)
    ),
    (   sub_string(StructCode, _, _, _, 'WamRuntime.step_unify_variable')
    ->  write('unify_variable lowered successfully.'), nl
    ;   write('FAILED: unify_variable not lowered!'), nl, halt(1)
    ),
    
    % Verify sum code
    read_file_to_string('benchmarks/elixir_baseline/lib/test_sum.ex', SumCode, []),
    (   sub_string(SumCode, _, _, _, 'choice_points:')
    ->  write('try_me_else lowered successfully.'), nl
    ;   write('FAILED: try_me_else not lowered!'), nl, halt(1)
    ).
