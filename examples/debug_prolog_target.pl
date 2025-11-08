:- use_module('../src/unifyweaver/targets/prolog_target').

% Test predicate
double(X, Y) :- Y is X * 2.

main :-
    format('Testing code generation...~n', []),
    generate_prolog_script([double/2], [], Code),
    format('~n==== GENERATED CODE ====~n~w~n====================~n', [Code]),
    halt(0).

:- initialization(main, main).
