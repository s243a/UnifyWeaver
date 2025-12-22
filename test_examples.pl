% examples.pl
file_dependency('main.o', 'main.c').
file_dependency('main.o', 'utils.h').
file_dependency('main.c', 'utils.c').
file_dependency('utils.o', 'utils.c').
file_dependency('utils.o', 'utils.h').

transitive_dependency(F, D) :- file_dependency(F, D).
transitive_dependency(F, D) :-
    file_dependency(F, I),
    transitive_dependency(I, D).
