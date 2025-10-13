% test_linear_constraints.pl - Test linear recursion with various constraints
:- use_module('../src/unifyweaver/core/advanced/linear_recursion').
:- use_module('../src/unifyweaver/core/constraint_analyzer').

test_linear_with_constraints :-
    writeln('=== LINEAR RECURSION CONSTRAINT TESTS ==='),

    % Setup output directory
    (   exists_directory('output/constraints') -> true
    ;   make_directory('output/constraints')
    ),

    % Clear previous test predicates
    catch(abolish(test_fib/2), _, true),
    catch(abolish(test_sum/2), _, true),
    catch(abolish(test_count/2), _, true),

    % Test 1: Linear recursion with unique(false) - no memoization
    writeln('\nTest 1: Linear recursion with unique(false)'),
    assertz(user:(test_sum(0, 0))),
    assertz(user:(test_sum(N, S) :- N > 0, N1 is N - 1, test_sum(N1, S1), S is N + S1)),

    % Declare constraint
    declare_constraint(test_sum/2, [unique(false)]),

    compile_linear_recursion(test_sum/2, [], Code1),
    write_bash_file('output/constraints/test_sum_no_memo.sh', Code1),
    writeln('  ✓ Compiled with unique(false) - memoization disabled'),

    % Verify memo is disabled
    (   sub_string(Code1, _, _, _, 'Memoization disabled') ->
        writeln('  ✓ Confirmed: memoization disabled in generated code')
    ;   writeln('  ✗ FAIL: memoization should be disabled')
    ),

    % Test 2: Linear recursion with ordered constraint
    writeln('\nTest 2: Linear recursion with ordered=true (unordered=false)'),
    assertz(user:(test_count(0, 0))),
    assertz(user:(test_count(N, C) :- N > 0, N1 is N - 1, test_count(N1, C1), C is C1 + 1)),

    % Declare constraint
    declare_constraint(test_count/2, [ordered]),

    compile_linear_recursion(test_count/2, [], Code2),
    write_bash_file('output/constraints/test_count_hash.sh', Code2),
    writeln('  ✓ Compiled with unordered(false) - hash-based memoization'),

    % Verify hash strategy
    (   sub_string(Code2, _, _, _, 'hash strategy') ->
        writeln('  ✓ Confirmed: hash-based memoization in generated code')
    ;   writeln('  ✗ FAIL: should use hash-based memoization')
    ),

    % Test 3: Default behavior (unique=true, ordered=false)
    writeln('\nTest 3: Linear recursion with default constraints'),
    assertz(user:(test_fib(0, 0))),
    assertz(user:(test_fib(1, 1))),
    assertz(user:(test_fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, test_fib(N1, F1), test_fib(N2, F2), F is F1 + F2)),

    % No constraints - should use default
    compile_linear_recursion(test_fib/2, [], Code3),
    write_bash_file('output/constraints/test_fib_default.sh', Code3),
    writeln('  ✓ Compiled with default constraints'),

    % Verify standard memo
    (   sub_string(Code3, _, _, _, 'standard strategy') ->
        writeln('  ✓ Confirmed: standard memoization in generated code')
    ;   writeln('  ✗ FAIL: should use standard memoization')
    ),

    % Cleanup
    clear_constraints(test_sum/2),
    clear_constraints(test_count/2),

    writeln('\n=== LINEAR RECURSION CONSTRAINT TESTS COMPLETE ===').

% Helper to write bash files
write_bash_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream),
    % Make executable
    atom_concat('chmod +x ', Path, ChmodCmd),
    shell(ChmodCmd).

% Run tests
:- initialization((test_linear_with_constraints, halt)).
