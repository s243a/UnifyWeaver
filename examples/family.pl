% Facts - direct parent relationships
parent(john, mary).
parent(mary, sue).
parent(sue, alice).
parent(john, bob).

% Rules - transitive closure
ancestor(X, Y) :- parent(X, Y).                    % Base case
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).    % Recursive case
