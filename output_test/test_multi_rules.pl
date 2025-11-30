:- encoding(utf8).
% Test multiple rules compilation

:- use_module('../src/unifyweaver/targets/go_target').

% Test data
log('ERROR: connection timeout').
log('WARNING: slow response').
log('ERROR: auth failed').
log('INFO: request completed').
log('CRITICAL: database down').

% Filter either ERROR or WARNING logs
alert(Line) :-
    log(Line),
    match(Line, 'ERROR').

alert(Line) :-
    log(Line),
    match(Line, 'WARNING').

alert(Line) :-
    log(Line),
    match(Line, 'CRITICAL').

test :-
    write('=== Test: Multiple Rules - alert ==='), nl,
    go_target:compile_predicate_to_go(alert/1, [], Code),
    write(Code), nl.

% Usage: swipl -q -t "consult('test_multi_rules.pl'), test, halt"
