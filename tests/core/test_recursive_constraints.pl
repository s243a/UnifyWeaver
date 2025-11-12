:- module(test_recursive_constraints, [
    test_recursive_constraints/0
]).

:- use_module(library(constraint_analyzer)).
:- use_module(library(recursive_compiler)).
:- use_module(library(stream_compiler)).

test_recursive_constraints :- 
    format('~n=== Testing Recursive Constraint System Integration ===~n'),
    setup_test_data,

    % Test 1: Default behavior (unique + unordered = sort -u)
    format('~n--- Test 1: Recursive Default constraints (sort -u) ---~n'),
    test_recursive_default_constraints,

    % Test 2: Ordered deduplication (unique + ordered = hash)
    format('~n--- Test 2: Recursive Ordered constraints (hash dedup) ---~n'),
    test_recursive_ordered_constraints,

    % Test 3: No deduplication (unique=false)
    format('~n--- Test 3: No deduplication ---~n'),
    test_no_deduplication,

    cleanup_test_data,
    format('~n=== All recursive constraint integration tests passed! ===~n').

setup_test_data :-
    catch(abolish(user:test_rec_fact/2), _, true),
    catch(abolish(user:test_rec_default/2), _, true),
    catch(abolish(user:test_rec_ordered/2), _, true),
    catch(abolish(user:test_rec_no_dedup/2), _, true),

    assertz(user:(test_rec_fact(a, b))),
    assertz(user:(test_rec_fact(b, c))),

    assertz(user:(test_rec_default(X, Y) :- test_rec_fact(X, Y))),
    assertz(user:(test_rec_default(X, Z) :- test_rec_default(X, Y), test_rec_fact(Y, Z))),

    assertz(user:(test_rec_ordered(X, Y) :- test_rec_fact(X, Y))),
    assertz(user:(test_rec_ordered(X, Z) :- test_rec_ordered(X, Y), test_rec_fact(Y, Z))),

    % Add duplicate rule for no_dedup test
    assertz(user:(test_rec_no_dedup(a,b))),
    assertz(user:(test_rec_no_dedup(a,b))).

cleanup_test_data :-
    catch(abolish(user:test_rec_fact/2), _, true),
    catch(abolish(user:test_rec_default/2), _, true),
    catch(abolish(user:test_rec_ordered/2), _, true),
    catch(abolish(user:test_rec_no_dedup/2), _, true),
    clear_constraints(test_rec_default/2),
    clear_constraints(test_rec_ordered/2),
    clear_constraints(test_rec_no_dedup/2).

test_recursive_default_constraints :-
    compile_recursive(test_rec_default/2, [], Code),
    (   sub_string(Code, _, _, _, "sort -u") ->
        format('  ✓ Uses sort -u (as expected)~n')
    ;   format('  ✗ FAILED: Expected sort -u~n'),
        fail
    ).

test_recursive_ordered_constraints :-
    declare_constraint(test_rec_ordered/2, [unique, ordered]),
    compile_recursive(test_rec_ordered/2, [], Code),
    (   sub_string(Code, _, _, _, "declare -A seen") ->
        format('  ✓ Uses hash-based dedup (as expected)~n')
    ;   format('  ✗ FAILED: Expected hash-based dedup~n'),
        fail
    ).

test_no_deduplication :-

    declare_constraint(test_rec_no_dedup/2, [unique(false)]),

    stream_compiler:compile_predicate(test_rec_no_dedup/2, [], Code),

    

    open('output/core_tests/test_no_dedup.sh', write, Stream, [newline(posix)]),

    write(Stream, Code),

    close(Stream),

    (   catch(
            process_create(path(bash), ['output/core_tests/test_no_dedup.sh'], [stdout(pipe(OutStream))]),
            Error,
            (format('  ✗ FAILED: process_create error: ~w~n', [Error]), fail)
        ),
        read_string(OutStream, _, Output),
        close(OutStream),
        string_length(Output, L),
        L > 0 ->
        split_string(Output, "\n", "\s\t\n", Lines),
        list_to_set(Lines, Set),
        length(Lines, LenLines),
        length(Set, LenSet),
        (   LenLines > LenSet ->
            format('  ✓ Duplicates found (as expected)~n')
        ;   format('  ✗ FAILED: Expected duplicates, but output is unique~n'),
            fail
        )
    ;   format('  ✗ FAILED: test_no_dedup.sh failed to run or produced no output~n'),
        fail
    ).
