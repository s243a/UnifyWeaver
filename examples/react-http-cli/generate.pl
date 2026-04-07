#!/usr/bin/env swipl

:- initialization(main, main).

main(_Argv) :-
    consult('react_http_cli_module.pl'),
    react_http_cli_module:generate_all,
    halt(0).
