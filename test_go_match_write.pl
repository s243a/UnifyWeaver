:- encoding(utf8).
:- use_module('src/unifyweaver/targets/go_target').

parse_log(Line, Time, Level) :-
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', auto, [Time, Level]).

test :-
    go_target:compile_predicate_to_go(parse_log/3, [], Code),
    make_directory_path('output_test'),
    go_target:write_go_program(Code, 'output_test/parse_log.go').
