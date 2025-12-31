:- initialization(main).

file_dependency('main.o', 'main.c').
file_dependency('main.o', 'utils.h').
file_dependency('main.c', 'utils.c').
file_dependency('utils.o', 'utils.c').
file_dependency('utils.o', 'utils.h').

transitive_dependency(F, D) :- file_dependency(F, D).
transitive_dependency(F, D) :-
    file_dependency(F, I),
    transitive_dependency(I, D).

main :-
    write('Query 1: Direct dependencies of main.o'), nl,
    findall(X, file_dependency('main.o', X), Results1),
    write('Results: '), write(Results1), nl, nl,
    
    write('Query 2: Transitive dependency check'), nl,
    (transitive_dependency('main.o', 'utils.c') ->
        write('true') ; write('false')), nl, nl,
    
    write('Query 3: All transitive dependencies of main.o'), nl,
    findall(X, transitive_dependency('main.o', X), Results3),
    write('Results: '), write(Results3), nl,
    
    halt.
