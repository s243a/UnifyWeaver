:- use_module('../src/unifyweaver/core/wam_dict').

:- dynamic test_failed/0.

pass(Test) :- format('[PASS] ~w~n', [Test]).
fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (test_failed -> true ; assert(test_failed)).

test_dict_new :-
    Test = 'wam_dict: new creates empty dict',
    (   wam_dict_new(D),
        \+ wam_dict_has_key(foo, D)
    ->  pass(Test)
    ;   fail_test(Test, 'new dict should be empty')
    ).

test_dict_insert_lookup :-
    Test = 'wam_dict: insert and lookup',
    (   wam_dict_new(D0),
        wam_dict_insert(hello, world, D0, D1),
        wam_dict_lookup(hello, D1, V),
        V == world
    ->  pass(Test)
    ;   fail_test(Test, 'insert/lookup failed')
    ).

test_dict_lookup_missing :-
    Test = 'wam_dict: lookup fails on missing key',
    (   wam_dict_new(D),
        \+ wam_dict_lookup(missing, D, _)
    ->  pass(Test)
    ;   fail_test(Test, 'should fail on missing key')
    ).

test_dict_lookup_default :-
    Test = 'wam_dict: lookup with default',
    (   wam_dict_new(D),
        wam_dict_lookup(missing, D, fallback, V),
        V == fallback
    ->  pass(Test)
    ;   fail_test(Test, 'default value not returned')
    ).

test_dict_overwrite :-
    Test = 'wam_dict: insert overwrites existing key',
    (   wam_dict_new(D0),
        wam_dict_insert(k, old, D0, D1),
        wam_dict_insert(k, new, D1, D2),
        wam_dict_lookup(k, D2, V),
        V == new
    ->  pass(Test)
    ;   fail_test(Test, 'overwrite failed')
    ).

test_dict_remove :-
    Test = 'wam_dict: remove key',
    (   wam_dict_new(D0),
        wam_dict_insert(k, v, D0, D1),
        wam_dict_remove(k, D1, D2),
        \+ wam_dict_has_key(k, D2)
    ->  pass(Test)
    ;   fail_test(Test, 'remove failed')
    ).

test_dict_remove_missing :-
    Test = 'wam_dict: remove missing key is no-op',
    (   wam_dict_new(D),
        wam_dict_remove(missing, D, D2),
        wam_dict_size(D2, 0)
    ->  pass(Test)
    ;   fail_test(Test, 'remove missing should be no-op')
    ).

test_dict_from_list :-
    Test = 'wam_dict: from_list builds dict',
    (   wam_dict_from_list([a-1, b-2, c-3], D),
        wam_dict_lookup(b, D, V),
        V == 2,
        wam_dict_size(D, 3)
    ->  pass(Test)
    ;   fail_test(Test, 'from_list failed')
    ).

test_dict_to_list :-
    Test = 'wam_dict: to_list round-trips',
    (   wam_dict_from_list([a-1, b-2], D),
        wam_dict_to_list(D, Pairs),
        length(Pairs, 2)
    ->  pass(Test)
    ;   fail_test(Test, 'to_list failed')
    ).

test_dict_keys :-
    Test = 'wam_dict: keys returns all keys',
    (   wam_dict_from_list([x-1, y-2, z-3], D),
        wam_dict_keys(D, Keys),
        msort(Keys, Sorted),
        Sorted == [x, y, z]
    ->  pass(Test)
    ;   fail_test(Test, 'keys failed')
    ).

test_dict_fold :-
    Test = 'wam_dict: fold sums values',
    (   wam_dict_from_list([a-10, b-20, c-30], D),
        wam_dict_fold([_K, V, Acc, Out]>>(Out is Acc + V), D, 0, Sum),
        Sum == 60
    ->  pass(Test)
    ;   fail_test(Test, 'fold failed')
    ).

test_dict_grouped :-
    Test = 'wam_dict: grouped dict collects multiple values per key',
    (   wam_dict_from_grouped([a-1, b-2, a-3, b-4, a-5], D),
        wam_dict_lookup_all(a, D, As),
        wam_dict_lookup_all(b, D, Bs),
        wam_dict_lookup_all(c, D, Cs),
        As == [1, 3, 5],
        Bs == [2, 4],
        Cs == []
    ->  pass(Test)
    ;   fail_test(Test, 'grouped dict failed')
    ).

test_dict_append :-
    Test = 'wam_dict: append adds to list',
    (   wam_dict_new(D0),
        wam_dict_append(k, a, D0, D1),
        wam_dict_append(k, b, D1, D2),
        wam_dict_append(k, c, D2, D3),
        wam_dict_lookup_all(k, D3, Vs),
        Vs == [a, b, c]
    ->  pass(Test)
    ;   fail_test(Test, 'append failed')
    ).

run_tests :-
    format('~n========================================~n'),
    format('WAM Dictionary Tests~n'),
    format('========================================~n~n'),
    test_dict_new,
    test_dict_insert_lookup,
    test_dict_lookup_missing,
    test_dict_lookup_default,
    test_dict_overwrite,
    test_dict_remove,
    test_dict_remove_missing,
    test_dict_from_list,
    test_dict_to_list,
    test_dict_keys,
    test_dict_fold,
    test_dict_grouped,
    test_dict_append,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).

:- initialization(run_tests, main).
