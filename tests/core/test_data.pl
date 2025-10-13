:- module(test_data, [
    parent/2,
    ancestor/2
]).

parent(a, b).
parent(b, c).
parent(c, d).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
