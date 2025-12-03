:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/go_target').

% Define a predicate to process logs
process_log(Timestamp, Level, Message) :-
    json_record([timestamp-Timestamp, level-Level, message-Message]).

% Test runner
run_test :-
    % 1. Compile the predicate with parallel execution (4 workers)
    compile_predicate_to_go(process_log/3, 
        [json_input(true), workers(4), unique(false)], 
        GoCode),
    
    % 2. Create output directory and write Go code
    shell('mkdir -p output'),
    write_go_program(GoCode, 'output/parallel_log.go'),
    
    % 3. Create a large input file (1000 lines)
    % 3. Create a large input file (1000 lines) using Prolog
    open('output/large_input.jsonl', write, Stream),
    forall(between(1, 1000, I),
        format(Stream, '{"timestamp":"2023-10-27T10:00:~w", "level":"INFO", "message":"Log entry ~w"}~n', [I, I])
    ),
    close(Stream),
    
    % 4. Run the Go program
    format('Running Go program with 4 workers...~n'),
    shell('go run output/parallel_log.go < output/large_input.jsonl > output/parallel_output.txt', ExitCode),
    (   ExitCode =:= 0
    ->  format('Go program executed successfully.~n')
    ;   format('Go program failed with exit code ~w~n', [ExitCode]),
        fail
    ),
    
    % 5. Verify output count
    setup_call_cleanup(
        open('output/parallel_output.txt', read, OutputStream),
        (   read_string(OutputStream, _, OutputStr),
            split_string(OutputStr, "\n", "\n", Lines),
            length(Lines, OutputCount)
        ),
        close(OutputStream)
    ),
    
    format('Output count: ~w~n', [OutputCount]),

    (   OutputCount =:= 1000
    ->  format('SUCCESS: Processed ~w lines (expected 1000)~n', [OutputCount])
    ;   format('FAILURE: Processed ~w lines (expected 1000)~n', [OutputCount]),
        fail
    ).
    
    % Cleanup (commented out for debugging)
    % shell('rm output/large_input.jsonl output/parallel_output.txt').

:- run_test, halt.
