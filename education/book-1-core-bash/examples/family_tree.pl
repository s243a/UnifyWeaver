% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)

% This file defines a simple family tree.
% It is used as an example for the UnifyWeaver educational series.

% --- Facts ---
% parent(Parent, Child).

parent(abraham, ishmael).
parent(abraham, isaac).
parent(sarah, isaac).

parent(isaac, esau).
parent(isaac, jacob).
parent(rebekah, esau).
parent(rebekah, jacob).

parent(jacob, reuben).
parent(jacob, simeon).
parent(jacob, levi).
parent(jacob, judah).
% ... and so on for all 12 sons and 1 daughter.


% --- Rules ---

% grandparent(Grandparent, Grandchild)
grandparent(GP, GC) :-
    parent(GP, P),
    parent(P, GC).

% ancestor(Ancestor, Descendant)
% This is a classic transitive closure.

% Base case: A is an ancestor of D if A is a parent of D.
ancestor(A, D) :- parent(A, D).

% Recursive step: A is an ancestor of D if A is a parent of P,
% and P is an ancestor of D.
ancestor(A, D) :- 
    parent(A, P),
    ancestor(P, D).


% --- Advanced Recursion Example ---

% Declare a constraint for the predicate below.
:- constraint(count_items/3, [unique(true)]).

% count_items(List, Accumulator, FinalCount)
% A tail-recursive predicate to count items in a list.

% Base Case: When the list is empty, the final count is the accumulator.
count_items([], Acc, Acc).

% Recursive Step: For a non-empty list, increment the accumulator
% and recurse on the tail of the list.
count_items([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count_items(T, Acc1, N).

