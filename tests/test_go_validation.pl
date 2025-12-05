:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/go_target').

% Define schema with validation constraints
:- json_schema(user, [
    field(age, integer, [min(18), max(120)]),
    field(email, string, [format(email)]),
    field(nickname, string, [optional])
]).

% Define predicate using the schema
process_user(Age, Email, Nickname) :-
    json_record([age-Age, email-Email, nickname-Nickname]).

% Test runner
run_test :-
    % 1. Compile the predicate
    compile_predicate_to_go(process_user/3, 
        [json_input(true), json_schema(user)], 
        GoCode),
    
    % 2. Create output directory and write Go code
    shell('mkdir -p output'),
    write_go_program(GoCode, 'output/validation_prog.go'),
    
    % 3. Create input file with valid and invalid records
    open('output/validation_input.jsonl', write, Stream),
    % Valid
    format(Stream, '{"age": 25, "email": "alice@example.com", "nickname": "Ali"}~n', []),
    % Invalid: Age too low
    format(Stream, '{"age": 10, "email": "bob@example.com"}~n', []),
    % Invalid: Age too high
    format(Stream, '{"age": 150, "email": "charlie@example.com"}~n', []),
    % Invalid: Bad email format
    format(Stream, '{"age": 30, "email": "not-an-email"}~n', []),
    % Valid: Missing optional nickname
    format(Stream, '{"age": 40, "email": "dave@example.com"}~n', []),
    close(Stream),
    
    % 4. Run the Go program
    format('Running Go validation test...~n'),
    shell('go run output/validation_prog.go < output/validation_input.jsonl > output/validation_output.txt', ExitCode),
    
    (   ExitCode =:= 0
    ->  format('Go program executed successfully.~n')
    ;   format('Go program failed with exit code ~w~n', [ExitCode]),
        fail
    ),
    
    % 5. Verify output count (Expect 2 valid records)
    setup_call_cleanup(
        open('output/validation_output.txt', read, OutputStream),
        (   read_string(OutputStream, _, OutputStr),
            split_string(OutputStr, "\n", "\n", Lines),
            length(Lines, OutputCount)
        ),
        close(OutputStream)
    ),
    
    format('Output count: ~w~n', [OutputCount]),

    (   OutputCount =:= 2
    ->  format('SUCCESS: Processed ~w valid records (expected 2)~n', [OutputCount])
    ;   format('FAILURE: Processed ~w valid records (expected 2)~n', [OutputCount]),
        fail
    ).

:- run_test, halt.
