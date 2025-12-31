:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/go_target').

% 1. Multiple Aggregations: Count and Sum by Dept
% Data: {"dept": "eng", "salary": 1000}, ...
stats_by_dept(Dept, Count, TotalSalary) :-
    group_by(Dept, json_record([dept-Dept, salary-Salary]), 
             [count(Count), sum(Salary, TotalSalary)]).

% 2. HAVING Clause: Filter groups with sum > 2000
high_salary_dept(Dept, TotalSalary) :-
    group_by(Dept, json_record([dept-Dept, salary-Salary]), sum(Salary, TotalSalary)),
    TotalSalary > 2000.

% Test runner
run_test :-
    shell('mkdir -p output'),
    
    % Create input data
    open('output/salaries_multi.jsonl', write, Stream),
    format(Stream, '{"dept": "eng", "salary": 1000}~n', []),
    format(Stream, '{"dept": "sales", "salary": 3000}~n', []),
    format(Stream, '{"dept": "eng", "salary": 1500}~n', []),
    format(Stream, '{"dept": "hr", "salary": 1000}~n', []),
    format(Stream, '{"dept": "eng", "salary": 2000}~n', []),
    % eng: count=3, sum=4500
    % sales: count=1, sum=3000
    % hr: count=1, sum=1000
    close(Stream),
    
    % Test 1: Multi-aggregation
    test_group_by(stats_by_dept/3, 'salaries_multi.jsonl', 'multi_stats.go', 
                  ['eng: 3: 4500', 
                   'sales: 1: 3000', 
                   'hr: 1: 1000']),
                   
    % Test 2: HAVING clause
    test_group_by(high_salary_dept/2, 'salaries_multi.jsonl', 'having_stats.go',
                  ['eng: 4500', 'sales: 3000']), % hr is filtered out (1000 <= 2000)
                  
    halt.

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

:- run_test.
