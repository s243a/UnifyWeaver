:- encoding(utf8).
% Test Go aggregation patterns

:- use_module('src/unifyweaver/targets/go_target').

% Test aggregations
test_sum :-
    write('=== Test: SUM aggregation ==='), nl, nl,
    go_target:compile_predicate_to_go(total/1,
        [aggregation(sum)], GoCode),
    write(GoCode), nl, nl.

test_count :-
    write('=== Test: COUNT aggregation ==='), nl, nl,
    go_target:compile_predicate_to_go(num_records/0,
        [aggregation(count)], GoCode),
    write(GoCode), nl, nl.

test_max :-
    write('=== Test: MAX aggregation ==='), nl, nl,
    go_target:compile_predicate_to_go(maximum/1,
        [aggregation(max)], GoCode),
    write(GoCode), nl, nl.

test_min :-
    write('=== Test: MIN aggregation ==='), nl, nl,
    go_target:compile_predicate_to_go(minimum/1,
        [aggregation(min)], GoCode),
    write(GoCode), nl, nl.

test_avg :-
    write('=== Test: AVG aggregation ==='), nl, nl,
    go_target:compile_predicate_to_go(average/1,
        [aggregation(avg)], GoCode),
    write(GoCode), nl, nl.

run_all :-
    test_sum,
    test_count,
    test_max,
    test_min,
    test_avg,
    write('All aggregation tests completed!'), nl.

% Usage:
% ?- consult('test_go_aggregation.pl').
% ?- run_all.
