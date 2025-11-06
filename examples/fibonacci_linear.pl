:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% fibonacci_linear.pl - Fibonacci using multi-call linear recursion
%
% With the multi-call linear recursion enhancement, fibonacci should now
% compile to bash using linear recursion with memoization, NOT the fold pattern.
%
% This is simpler and more efficient than the fold approach.

% Fibonacci - now compiles with linear recursion + memoization
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

% Example usage:
% ?- fib(10, F).
% F = 55.
