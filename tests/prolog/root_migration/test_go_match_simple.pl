:- encoding(utf8).
% Test Go match predicate with capture groups (simple version - no body predicates)

:- use_module('src/unifyweaver/targets/go_target').

% Test: Extract timestamp and level from stdin (two capture groups)
% No body predicates - just read from stdin and match
parse_log(Line, Time, Level) :-
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', auto, [Time, Level]).

% Compile test
test_captures :-
    write('=== Test: Two Capture Groups (Time + Level) from stdin ==='), nl, nl,
    go_target:compile_predicate_to_go(parse_log/3, [], Code),
    write(Code), nl.

% Usage:
% ?- consult('test_go_match_simple.pl').
% ?- test_captures.
