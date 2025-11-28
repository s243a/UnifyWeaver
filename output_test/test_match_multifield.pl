:- encoding(utf8).
% Test multi-field with match constraints

:- use_module('../src/unifyweaver/targets/go_target').

% Test data
user_log(alice, 'ERROR: login failed').
user_log(bob, 'INFO: successful login').
user_log(charlie, 'ERROR: timeout').

% Filter error logs and extract user
error_user(User, Msg) :-
    user_log(User, Msg),
    match(Msg, 'ERROR').

test :-
    write('=== Test: Multi-field with match ==='), nl,
    go_target:compile_predicate_to_go(error_user/2, [], Code),
    write(Code), nl.

% Usage: swipl -q -t "consult('test_match_multifield.pl'), test, halt"
