:- encoding(utf8).
:- use_module('src/unifyweaver/targets/go_target').

test_all :-
    go_target:compile_predicate_to_go(maximum/1, [aggregation(max)], MaxCode),
    go_target:write_go_program(MaxCode, 'max.go'),
    go_target:compile_predicate_to_go(minimum/1, [aggregation(min)], MinCode),
    go_target:write_go_program(MinCode, 'min.go'),
    go_target:compile_predicate_to_go(average/1, [aggregation(avg)], AvgCode),
    go_target:write_go_program(AvgCode, 'avg.go').
