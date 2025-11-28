:- encoding(utf8).
:- use_module('src/unifyweaver/targets/go_target').

test_write_sum :-
    go_target:compile_predicate_to_go(total/1, [aggregation(sum)], Code),
    go_target:write_go_program(Code, 'sum.go').

test_write_count :-
    go_target:compile_predicate_to_go(num_records/0, [aggregation(count)], Code),
    go_target:write_go_program(Code, 'count.go').

test_write_all :-
    test_write_sum,
    test_write_count.
