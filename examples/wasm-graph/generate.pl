#!/usr/bin/env swipl

:- initialization(main, main).

main(_Argv) :-
    consult('graph_module.pl'),
    graph_module:generate_all,
    halt(0).
