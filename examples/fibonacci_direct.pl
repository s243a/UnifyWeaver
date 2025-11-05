:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% fibonacci_direct.pl - Fibonacci using direct recursive strategy
%
% This file explicitly requests the direct recursive code generation strategy
% instead of the default fold-based approach.

% Load pattern_matchers for strategy directive
:- use_module('../src/unifyweaver/core/advanced/pattern_matchers').

% Request direct recursive strategy (not fold)
:- recursion_strategy(fib/2, direct).

% Fibonacci - should compile to direct recursive bash with memoization
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
