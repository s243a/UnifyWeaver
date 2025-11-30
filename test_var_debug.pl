:- encoding(utf8).
% Debug variable sharing

% Simple test case
adult(X, Age) :- Age > 18.

test :-
    Head = adult(_,_),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses = [H-B],
    write('Head: '), write(H), nl,
    write('Body: '), write(B), nl,
    H =.. [_|Args],
    write('Head args: '), write(Args), nl,
    % Extract the constraint
    B = (Age > 18),
    write('Constraint var Age: '), write(Age), nl,
    % Check if they're the same
    Args = [X, Age2],
    write('Age2 from head: '), write(Age2), nl,
    (Age == Age2 -> write('SAME VAR!') ; write('DIFFERENT VARS!')), nl.

:- initialization(test).
