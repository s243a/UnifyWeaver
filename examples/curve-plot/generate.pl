#!/usr/bin/env swipl

:- initialization(main, main).

main(_Argv) :-
    consult('curve_module.pl'),
    curve_module:generate_all,
    halt(0).
