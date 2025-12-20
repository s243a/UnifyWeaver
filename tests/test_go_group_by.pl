:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/go_target').

% Define predicates with grouped aggregation

% 1. Count users by role (using simple key)
% Data: {"name": "A", "role": "admin"}, {"name": "B", "role": "user"}, ...
count_by_role(Key, Count) :-
    group_by(Key, json_record([role-Key]), count, Count).

% 2. Sum salary by department
% Data: {"dept": "eng", "salary": 100}, {"dept": "sales", "salary": 200}, ...
sum_salary_by_dept(Dept, TotalSalary) :-
    group_by(Dept, json_record([dept-Dept, salary-Salary]), sum(Salary), TotalSalary).

% Test runner
run_test :-
    shell('mkdir -p output'),
    
    % Create input data for roles
    open('output/users_roles.jsonl', write, Stream1),
    format(Stream1, '{"name": "Alice", "role": "admin"}~n', []),
    format(Stream1, '{"name": "Bob", "role": "user"}~n', []),
    format(Stream1, '{"name": "Charlie", "role": "user"}~n', []),
    format(Stream1, '{"name": "Dave", "role": "admin"}~n', []),
    format(Stream1, '{"name": "Eve", "role": "user"}~n', []),
    close(Stream1),
    
    % Create input data for salaries
    open('output/salaries.jsonl', write, Stream2),
    format(Stream2, '{"dept": "eng", "salary": 1000}~n', []),
    format(Stream2, '{"dept": "sales", "salary": 2000}~n', []),
    format(Stream2, '{"dept": "eng", "salary": 1500}~n', []),
    format(Stream2, '{"dept": "hr", "salary": 1000}~n', []),
    format(Stream2, '{"dept": "eng", "salary": 2000}~n', []),
    close(Stream2),
    
    test_group_by(count_by_role/2, 'users_roles.jsonl', 'count_role.go', 
                  ['admin: 2', 'user: 3']),
                  
    test_group_by(sum_salary_by_dept/2, 'salaries.jsonl', 'sum_dept.go', 
                  ['eng: 4500', 'sales: 2000', 'hr: 1000']).

test_group_by(Pred/Arity, InputFile, ProgName, ExpectedLines) :-
    format('Testing ~w...~n', [Pred]),
    
    % Compile (using json_input(true) for streams)
    compile_predicate_to_go(Pred/Arity, [json_input(true)], GoCode),
    atom_concat('output/', ProgName, ProgPath),
    write_go_program(GoCode, ProgPath),
    
    % Run
    format(atom(Cmd), 'go run ~w < output/~w > output/result_~w.txt', [ProgPath, InputFile, ProgName]),
    shell(Cmd, ExitCode),
    
    (   ExitCode =:= 0
    ->  true
    ;   format('FAILURE: Go program failed~n'), fail
    ),
    
    % Verify output contains expected lines
    atom_concat('output/result_', ProgName, ResultPath),
    atom_concat(ResultPath, '.txt', FullResultPath),
    read_file_to_string(FullResultPath, OutputStr, []),
    
    check_contains_lines(OutputStr, ExpectedLines).

check_contains_lines(_, []).
check_contains_lines(Output, [Line|Rest]) :-
    (   sub_string(Output, _, _, _, Line)
    ->  format('  Found expected: ~w~n', [Line])
    ;   format('FAILURE: Missing expected output: ~w~n', [Line]),
        fail
    ),
    check_contains_lines(Output, Rest).

:- run_test, halt.
