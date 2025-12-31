:- initialization(main).

% Test example from Chapter 2
file_dependency('main.o', 'main.c').
file_dependency('main.o', 'utils.h').
file_dependency('main.c', 'utils.c').
file_dependency('utils.o', 'utils.c').
file_dependency('utils.o', 'utils.h').

transitive_dependency(F, D) :- file_dependency(F, D).
transitive_dependency(F, D) :-
    file_dependency(F, I),
    transitive_dependency(I, D).

% Test family tree from education examples
parent(abraham, ishmael).
parent(abraham, isaac).
parent(sarah, isaac).
parent(isaac, esau).
parent(isaac, jacob).

grandparent(GP, GC) :- parent(GP, P), parent(P, GC).

ancestor(A, D) :- parent(A, D).
ancestor(A, D) :- parent(A, P), ancestor(P, D).

main :-
    nl, write('=== CHAPTER 2: PROLOG FUNDAMENTALS ==='), nl,
    
    write('Test 1: Direct dependencies'), nl,
    findall(X, file_dependency('main.o', X), R1),
    write('  Result: '), write(R1), nl, nl,
    
    write('Test 2: Transitive dependency check'), nl,
    (transitive_dependency('main.o', 'utils.c') -> 
        write('  Result: true') ; write('  Result: false')), nl, nl,
    
    write('Test 3: All transitive dependencies'), nl,
    findall(X, transitive_dependency('main.o', X), R3),
    write('  Result: '), write(R3), nl, nl,
    
    write('Test 4: Unification and query'), nl,
    write('  Query: parent(abraham, X)'), nl,
    findall(X, parent(abraham, X), R4),
    write('  Result: '), write(R4), nl, nl,
    
    write('Test 5: Grandparent rule'), nl,
    findall(GP-GC, grandparent(GP, GC), R5),
    write('  Result: '), write(R5), nl, nl,
    
    write('Test 6: Ancestor transitive closure'), nl,
    findall(A-D, ancestor(abraham, D), R6),
    write('  Result: '), write(R6), nl, nl,
    
    halt.
