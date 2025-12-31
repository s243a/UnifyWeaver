:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/go_target').

% Define predicates with aggregation
count_users(Count) :-
    aggregate(count, json_record([name-_]), Count).

sum_age(Sum) :-
    aggregate(sum(Age), json_record([age-Age]), Sum).

avg_age(Avg) :-
    aggregate(avg(Age), json_record([age-Age]), Avg).

max_age(Max) :-
    aggregate(max(Age), json_record([age-Age]), Max).

min_age(Min) :-
    aggregate(min(Age), json_record([age-Age]), Min).

% Test runner
run_test :-
    shell('mkdir -p output'),
    
    % Create input data
    open('output/users.jsonl', write, Stream),
    format(Stream, '{"name": "Alice", "age": 25}~n', []),
    format(Stream, '{"name": "Bob", "age": 30}~n', []),
    format(Stream, '{"name": "Charlie", "age": 35}~n', []),
    format(Stream, '{"name": "Dave", "age": 40}~n', []),
    close(Stream),
    
    test_aggregation(count_users/1, 'count_prog.go', 4),
    test_aggregation(sum_age/1, 'sum_prog.go', 130),
    test_aggregation(avg_age/1, 'avg_prog.go', 32.5),
    test_aggregation(max_age/1, 'max_prog.go', 40),
    test_aggregation(min_age/1, 'min_prog.go', 25).

test_aggregation(Pred/Arity, ProgName, Expected) :-
    format('Testing ~w...~n', [Pred]),
    
    % Compile
    compile_predicate_to_go(Pred/Arity, [json_input(true)], GoCode),
    atom_concat('output/', ProgName, ProgPath),
    write_go_program(GoCode, ProgPath),
    
    % Run
    format(atom(Cmd), 'go run ~w < output/users.jsonl > output/result.txt', [ProgPath]),
    shell(Cmd, ExitCode),
    
    (   ExitCode =:= 0
    ->  true
    ;   format('FAILURE: Go program failed~n'), fail
    ),
    
    % Verify
    read_file_to_string('output/result.txt', OutputStr, []),
    split_string(OutputStr, "\n", "\s\t\n", [Line|_]),
    number_string(Actual, Line),
    
    (   Actual =:= Expected
    ->  format('SUCCESS: Got ~w (expected ~w)~n', [Actual, Expected])
    ;   format('FAILURE: Got ~w (expected ~w)~n', [Actual, Expected]),
        fail
    ).

:- run_test, halt.
