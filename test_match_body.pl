:- use_module('src/unifyweaver/targets/go_target').

% Test combining body predicates with match captures
%
% Example: Parse log entries where we have name:logline format
% and want to extract level and message from the logline

% Body predicate - defines the input structure
log_entry(alice, '2025-01-15 ERROR timeout occurred').
log_entry(bob, '2025-01-15 INFO operation successful').
log_entry(charlie, '2025-01-15 WARNING slow response').

% Rule that combines body predicate with match captures
% This should:
% 1. Read stdin with name:logline format
% 2. Split into field1 (Name) and field2 (Line)
% 3. Apply regex to field2 to extract Level and Message
% 4. Output Name:Level:Message
parsed(Name, Level, Message) :-
    log_entry(Name, Line),
    match(Line, '([A-Z]+): (.+)', auto, [Level, Message]).

test_match_body :-
    % Compile the predicate
    compile_predicate_to_go(parsed/3, [], Code),

    % Display the generated code
    format('~n=== Generated Go Code ===~n~s~n', [Code]).

test_write_match_body :-
    % Compile and write
    compile_predicate_to_go(parsed/3, [], Code),
    write_go_program(Code, 'parsed.go'),
    format('Generated parsed.go~n').
