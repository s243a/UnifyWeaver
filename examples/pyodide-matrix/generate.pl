#!/usr/bin/env swipl

:- initialization(main, main).

main(_Argv) :-
    consult('matrix_module.pl'),
    matrix_module:generate_all,
    halt(0).
