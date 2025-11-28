% is_even(N) :- True if N is an even number.
is_even(0).
is_even(N) :-
    N > 0,
    N1 is N - 1,
    is_odd(N1).

% is_odd(N) :- True if N is an odd number.
is_odd(N) :-
    N > 0,
    N1 is N - 1,
    is_even(N1).
