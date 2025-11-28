:- use_module('src/unifyweaver/targets/go_target').

% Test with tab delimiter to avoid colon conflicts

% Body predicate
log_entry(alice, '2025-01-15 ERROR: timeout occurred').
log_entry(bob, '2025-01-15 INFO: operation successful').
log_entry(charlie, '2025-01-15 WARNING: slow response').

% Rule with match+body predicates
parsed(Name, Level, Message) :-
    log_entry(Name, Line),
    match(Line, '([A-Z]+): (.+)', auto, [Level, Message]).

test :-
    % Compile with tab delimiter
    compile_predicate_to_go(parsed/3, [field_delimiter(tab)], Code),
    write_go_program(Code, 'parsed_tab.go'),
    format('Generated parsed_tab.go with tab delimiter~n').
