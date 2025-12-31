:- encoding(utf8).
% Test AWK aggregation patterns

:- use_module('src/unifyweaver/targets/awk_target').

% Test aggregations
test_sum :-
    write('=== Test: SUM aggregation ==='), nl, nl,
    awk_target:compile_predicate_to_awk(total/1,
        [aggregation(sum)], AwkCode),
    write(AwkCode), nl, nl.

test_count :-
    write('=== Test: COUNT aggregation ==='), nl, nl,
    awk_target:compile_predicate_to_awk(num_records/0,
        [aggregation(count)], AwkCode),
    write(AwkCode), nl, nl.

test_max :-
    write('=== Test: MAX aggregation ==='), nl, nl,
    awk_target:compile_predicate_to_awk(maximum/1,
        [aggregation(max)], AwkCode),
    write(AwkCode), nl, nl.

test_min :-
    write('=== Test: MIN aggregation ==='), nl, nl,
    awk_target:compile_predicate_to_awk(minimum/1,
        [aggregation(min)], AwkCode),
    write(AwkCode), nl, nl.

test_avg :-
    write('=== Test: AVG aggregation ==='), nl, nl,
    awk_target:compile_predicate_to_awk(average/1,
        [aggregation(avg)], AwkCode),
    write(AwkCode), nl, nl.

run_all :-
    test_sum,
    test_count,
    test_max,
    test_min,
    test_avg,
    write('All aggregation tests completed!'), nl.

% Usage:
% ?- consult('test_awk_aggregation.pl').
% ?- run_all.
