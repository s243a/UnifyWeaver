:- encoding(utf8).
% Test suite for LLVM target
% Usage: swipl -g run_tests -t halt tests/test_llvm_target.pl

:- use_module('../src/unifyweaver/targets/llvm_target').

%% Test predicates (for facts test)
test_person(john, 25).
test_person(jane, 30).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% Tests
test_llvm_tail_recursion :-
    Test = 'LLVM: compile_tail_recursion',
    (   llvm_target:compile_tail_recursion_llvm(sum/2, [], Code),
        sub_atom(Code, _, _, _, 'musttail call'),
        sub_atom(Code, _, _, _, 'define i64 @sum')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing musttail or function definition')
    ).

test_llvm_tail_recursion_export :-
    Test = 'LLVM: tail_recursion with export',
    (   llvm_target:compile_tail_recursion_llvm(sum/2, [export(true)], Code),
        sub_atom(Code, _, _, _, 'dllexport'),
        sub_atom(Code, _, _, _, '_ext')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing export wrapper')
    ).

test_llvm_linear_recursion :-
    Test = 'LLVM: compile_linear_recursion',
    (   llvm_target:compile_linear_recursion_llvm(fib/2, [], Code),
        sub_atom(Code, _, _, _, '@memo'),
        sub_atom(Code, _, _, _, 'getelementptr')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing memo table')
    ).

test_llvm_mutual_recursion :-
    Test = 'LLVM: compile_mutual_recursion',
    (   llvm_target:compile_mutual_recursion_llvm([is_even/1, is_odd/1], [], Code),
        sub_atom(Code, _, _, _, '@is_even'),
        sub_atom(Code, _, _, _, '@is_odd'),
        sub_atom(Code, _, _, _, 'musttail call i1 @is_odd'),
        sub_atom(Code, _, _, _, 'musttail call i1 @is_even')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing mutual recursion with musttail')
    ).

test_llvm_facts :-
    Test = 'LLVM: compile_facts',
    (   llvm_target:compile_predicate_to_llvm(test_person/2, [], Code),
        sub_atom(Code, _, _, _, '@str.'),
        sub_atom(Code, _, _, _, '_count')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing string constants or count')
    ).

test_llvm_transitive_closure :-
    Test = 'LLVM: compile_transitive_closure',
    (   llvm_target:compile_transitive_closure_llvm(reachable/2, [], Code),
        sub_atom(Code, _, _, _, '@add_edge'),
        sub_atom(Code, _, _, _, 'bfs_loop'),
        sub_atom(Code, _, _, _, '@visited')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing BFS worklist constructs')
    ).

%% Phase 2 Tests
test_llvm_shared_library :-
    Test = 'LLVM Phase 2: compile_shared_library',
    (   llvm_target:compile_shared_library_llvm(
            [func(sum, 2, tail_recursion), func(fib, 2, linear_recursion)],
            [library_name(test_lib)],
            Code),
        sub_atom(Code, _, _, _, 'dllexport'),
        sub_atom(Code, _, _, _, '@sum'),
        sub_atom(Code, _, _, _, '@fib')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing exports or functions')
    ).

test_llvm_c_header :-
    Test = 'LLVM Phase 2: generate_c_header',
    (   llvm_target:generate_c_header(
            [func(sum, 2, tail_recursion), func(factorial, 1, factorial)],
            Header),
        sub_atom(Header, _, _, _, 'int64_t sum'),
        sub_atom(Header, _, _, _, 'PROLOG_MATH_H')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing header declarations')
    ).

test_llvm_cgo_bindings :-
    Test = 'LLVM Phase 2: generate_cgo_bindings',
    (   llvm_target:generate_cgo_bindings(
            [func(sum, 2, tail_recursion)],
            GoCode),
        sub_atom(GoCode, _, _, _, 'package prologmath'),
        sub_atom(GoCode, _, _, _, 'func sum')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Go bindings')
    ).

test_llvm_rust_ffi :-
    Test = 'LLVM Phase 2: generate_rust_ffi',
    (   llvm_target:generate_rust_ffi(
            [func(sum, 2, tail_recursion)],
            RustCode),
        sub_atom(RustCode, _, _, _, 'extern "C"'),
        sub_atom(RustCode, _, _, _, 'pub fn sum')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Rust FFI')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('LLVM Target Test Suite~n'),
    format('========================================~n~n'),
    
    format('--- Phase 1 Tests ---~n'),
    test_llvm_tail_recursion,
    test_llvm_tail_recursion_export,
    test_llvm_linear_recursion,
    test_llvm_mutual_recursion,
    test_llvm_transitive_closure,
    test_llvm_facts,
    
    format('~n--- Phase 2 Tests ---~n'),
    test_llvm_shared_library,
    test_llvm_c_header,
    test_llvm_cgo_bindings,
    test_llvm_rust_ffi,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
