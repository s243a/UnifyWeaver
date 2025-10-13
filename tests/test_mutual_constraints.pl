% test_mutual_constraints.pl - Test mutual recursion with constraints
:- use_module('../src/unifyweaver/core/advanced/mutual_recursion').
:- use_module('../src/unifyweaver/core/constraint_analyzer').

test_mutual_with_constraints :-
    writeln('=== MUTUAL RECURSION CONSTRAINT TESTS ==='),

    % Setup output directory
    (   exists_directory('output/constraints') -> true
    ;   make_directory('output/constraints')
    ),

    % Clear previous test predicates
    catch(abolish(test_even1/1), _, true),
    catch(abolish(test_odd1/1), _, true),
    catch(abolish(test_even2/1), _, true),
    catch(abolish(test_odd2/1), _, true),
    catch(abolish(test_even3/1), _, true),
    catch(abolish(test_odd3/1), _, true),

    % Test 1: Default constraints (unique=true, unordered=true)
    writeln('\nTest 1: Mutual recursion with default constraints'),
    assertz(user:test_even1(0)),
    assertz(user:(test_even1(N) :- N > 0, N1 is N - 1, test_odd1(N1))),
    assertz(user:test_odd1(1)),
    assertz(user:(test_odd1(N) :- N > 1, N1 is N - 1, test_even1(N1))),

    compile_mutual_recursion([test_even1/1, test_odd1/1], [], Code1),
    write_bash_file('output/constraints/even_odd_default.sh', Code1),
    writeln('  ✓ Compiled with default constraints'),

    % Verify standard memoization
    (   sub_string(Code1, _, _, _, 'standard strategy') ->
        writeln('  ✓ Confirmed: standard shared memoization')
    ;   writeln('  ✗ FAIL: should use standard memoization')
    ),

    % Test 2: Ordered constraint (unordered=false)
    writeln('\nTest 2: Mutual recursion with ordered constraint'),
    assertz(user:test_even2(0)),
    assertz(user:(test_even2(N) :- N > 0, N1 is N - 1, test_odd2(N1))),
    assertz(user:test_odd2(1)),
    assertz(user:(test_odd2(N) :- N > 1, N1 is N - 1, test_even2(N1))),

    % Declare ordered constraint on one predicate (should propagate to whole SCC)
    declare_constraint(test_even2/1, [ordered]),

    compile_mutual_recursion([test_even2/1, test_odd2/1], [], Code2),
    write_bash_file('output/constraints/even_odd_ordered.sh', Code2),
    writeln('  ✓ Compiled with ordered constraint on test_even2/1'),

    % Verify hash memoization
    (   sub_string(Code2, _, _, _, 'hash strategy') ->
        writeln('  ✓ Confirmed: hash-based shared memoization')
    ;   writeln('  ✗ FAIL: should use hash-based memoization')
    ),

    % Test 3: unique(false) constraint - disables memoization
    writeln('\nTest 3: Mutual recursion with unique(false) constraint'),
    assertz(user:test_even3(0)),
    assertz(user:(test_even3(N) :- N > 0, N1 is N - 1, test_odd3(N1))),
    assertz(user:test_odd3(1)),
    assertz(user:(test_odd3(N) :- N > 1, N1 is N - 1, test_even3(N1))),

    % Declare unique(false) on one predicate (should disable memo for whole SCC)
    declare_constraint(test_odd3/1, [unique(false)]),

    compile_mutual_recursion([test_even3/1, test_odd3/1], [], Code3),
    write_bash_file('output/constraints/even_odd_no_memo.sh', Code3),
    writeln('  ✓ Compiled with unique(false) on test_odd3/1'),

    % Verify memoization disabled
    (   sub_string(Code3, _, _, _, 'Shared memoization disabled') ->
        writeln('  ✓ Confirmed: shared memoization disabled')
    ;   writeln('  ✗ FAIL: should disable shared memoization')
    ),

    % Cleanup
    clear_constraints(test_even2/1),
    clear_constraints(test_odd3/1),

    writeln('\n=== MUTUAL RECURSION CONSTRAINT TESTS COMPLETE ===').

% Helper to write bash files
write_bash_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream),
    % Make executable
    atom_concat('chmod +x ', Path, ChmodCmd),
    shell(ChmodCmd).

% Run tests
:- initialization((test_mutual_with_constraints, halt)).
