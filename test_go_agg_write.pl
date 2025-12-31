:- encoding(utf8).
:- use_module(library(filesex)).
:- use_module('src/unifyweaver/targets/go_target').

ensure_output_dir :- make_directory_path('output_test').

test_write_sum :-
    ensure_output_dir,
    go_target:compile_predicate_to_go(total/1, [aggregation(sum)], Code),
    go_target:write_go_program(Code, 'output_test/sum.go').

test_write_count :-
    ensure_output_dir,
    go_target:compile_predicate_to_go(num_records/0, [aggregation(count)], Code),
    go_target:write_go_program(Code, 'output_test/count.go').

test_write_all :-
    test_write_sum,
    test_write_count.
