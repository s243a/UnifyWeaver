<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 5: Tail Recursion

This chapter covers tail-recursive predicates and how they compile to efficient AWK while loops.

## What is Tail Recursion?

Tail recursion is a special form of recursion where the recursive call is the last operation in the function. This allows the compiler to optimize it into a loop, avoiding stack overflow.

```prolog
% Tail-recursive: recursive call is last
factorial(0, Acc, Acc).
factorial(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, Result).  % Last operation

% NOT tail-recursive: operation after recursive call
factorial_bad(0, 1).
factorial_bad(N, Result) :-
    N > 0,
    N1 is N - 1,
    factorial_bad(N1, R),
    Result is R * N.  % Operation AFTER recursion
```

## Why Tail Recursion for AWK?

AWK doesn't have built-in recursion support (no function call stack for user functions in traditional AWK). The AWK target transforms tail-recursive Prolog predicates into while loops:

```
Tail-Recursive Prolog → AWK While Loop
```

This transformation:
- Avoids stack limitations
- Produces efficient iterative code
- Works within AWK's execution model

## Tail Recursion Pattern

### Required Structure

```prolog
% Base case
predicate(BaseCondition, Acc, Result) :-
    BaseCondition,
    Result = Acc.

% Recursive case
predicate(Input, Acc, Result) :-
    RecursiveCondition,
    NewInput is ...,
    NewAcc is ...,
    predicate(NewInput, NewAcc, Result).  % MUST be last
```

### Compilation Result

```awk
function predicate(Input, Acc) {
    while (1) {
        if (BaseCondition) {
            return Acc
        }

        # Update variables
        NewInput = ...
        NewAcc = ...

        # Loop instead of recurse
        Input = NewInput
        Acc = NewAcc
    }
}
```

## Example: Factorial

### Prolog Definition

```prolog
factorial(0, Acc, Acc).
factorial(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, Result).

% Entry point (wrapper)
fact(N, Result) :-
    factorial(N, 1, Result).
```

### Generated AWK

```awk
function factorial(N, Acc) {
    while (1) {
        if (N == 0) {
            return Acc
        }
        if (N > 0) {
            N1 = N - 1
            Acc1 = Acc * N
            N = N1
            Acc = Acc1
        }
    }
}

{
    N = $1
    Result = factorial(N, 1)
    print Result
}
```

## Example: Fibonacci

### Prolog Definition

```prolog
% Tail-recursive Fibonacci with accumulators
fib(0, A, _, A).
fib(N, A, B, Result) :-
    N > 0,
    N1 is N - 1,
    Sum is A + B,
    fib(N1, B, Sum, Result).

% Entry point
fibonacci(N, Result) :-
    fib(N, 0, 1, Result).
```

### Generated AWK

```awk
function fib(N, A, B) {
    while (1) {
        if (N == 0) {
            return A
        }
        if (N > 0) {
            N1 = N - 1
            Sum = A + B
            N = N1
            temp = B
            B = Sum
            A = temp
        }
    }
}

{
    N = $1
    Result = fib(N, 0, 1)
    print Result
}
```

## Example: Sum of List

### Prolog Definition

```prolog
% Sum numbers from N down to 1
sum_to(0, Acc, Acc).
sum_to(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    sum_to(N1, Acc1, Result).

sum(N, Result) :-
    sum_to(N, 0, Result).
```

### Generated AWK

```awk
function sum_to(N, Acc) {
    while (1) {
        if (N == 0) {
            return Acc
        }
        N1 = N - 1
        Acc1 = Acc + N
        N = N1
        Acc = Acc1
    }
}

{
    N = $1
    Result = sum_to(N, 0)
    print Result
}
```

## Example: GCD (Greatest Common Divisor)

### Prolog Definition

```prolog
% Euclidean algorithm
gcd(A, 0, A).
gcd(A, B, Result) :-
    B > 0,
    R is A mod B,
    gcd(B, R, Result).
```

### Generated AWK

```awk
function gcd(A, B) {
    while (1) {
        if (B == 0) {
            return A
        }
        R = A % B
        A = B
        B = R
    }
}

{
    A = $1
    B = $2
    Result = gcd(A, B)
    print Result
}
```

## Example: Power (Exponentiation)

### Prolog Definition

```prolog
% Power with accumulator
power(_, 0, Acc, Acc).
power(Base, Exp, Acc, Result) :-
    Exp > 0,
    Exp1 is Exp - 1,
    Acc1 is Acc * Base,
    power(Base, Exp1, Acc1, Result).

pow(Base, Exp, Result) :-
    power(Base, Exp, 1, Result).
```

### Generated AWK

```awk
function power(Base, Exp, Acc) {
    while (1) {
        if (Exp == 0) {
            return Acc
        }
        Exp1 = Exp - 1
        Acc1 = Acc * Base
        Exp = Exp1
        Acc = Acc1
    }
}

{
    Base = $1
    Exp = $2
    Result = power(Base, Exp, 1)
    print Result
}
```

## Accumulator Pattern

The key to tail recursion is the **accumulator** - a parameter that builds up the result:

```prolog
% Without accumulator (NOT tail-recursive)
sum_bad(0, 0).
sum_bad(N, Result) :-
    N > 0,
    N1 is N - 1,
    sum_bad(N1, R),
    Result is R + N.  % Operation after recursion!

% With accumulator (tail-recursive)
sum_good(0, Acc, Acc).
sum_good(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,  % Build result in accumulator
    sum_good(N1, Acc1, Result).  % Recursive call is last
```

## Limitations

### Only Tail Recursion Supported

The AWK target only compiles tail-recursive predicates. These patterns will NOT work:

```prolog
% NOT supported: Non-tail recursion
bad_factorial(0, 1).
bad_factorial(N, Result) :-
    N > 0,
    N1 is N - 1,
    bad_factorial(N1, R),
    Result is R * N.  % Operation after recursion

% NOT supported: Mutual recursion
even(0).
even(N) :- N > 0, N1 is N - 1, odd(N1).

odd(N) :- N > 0, N1 is N - 1, even(N1).
```

### Detecting Tail Position

The recursive call must be the **last goal** in the clause body:

```prolog
% Tail position - OK
foo(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    foo(N1, Acc1, Result).  % ✓ Last goal

% NOT tail position - Will not compile
foo(N, Acc, Result) :-
    N > 0,
    foo(N1, Acc1, Result),  % ✗ Not last
    N1 is N - 1,
    Acc1 is Acc + N.
```

## Complete Example

```prolog
% file: tail_recursion.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

% Factorial
factorial(0, Acc, Acc).
factorial(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, Result).

fact(N, Result) :-
    factorial(N, 1, Result).

% Fibonacci
fib(0, A, _, A).
fib(N, A, B, Result) :-
    N > 0,
    N1 is N - 1,
    Sum is A + B,
    fib(N1, B, Sum, Result).

fibonacci(N, Result) :-
    fib(N, 0, 1, Result).

% GCD
gcd(A, 0, A).
gcd(A, B, Result) :-
    B > 0,
    R is A mod B,
    gcd(B, R, Result).

% Sum 1 to N
sum_to(0, Acc, Acc).
sum_to(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    sum_to(N1, Acc1, Result).

sum_n(N, Result) :-
    sum_to(N, 0, Result).

% Generate scripts
generate :-
    compile_predicate_to_awk(fact/2, [], AWK1),
    write_awk_script('factorial.awk', AWK1),

    compile_predicate_to_awk(fibonacci/2, [], AWK2),
    write_awk_script('fibonacci.awk', AWK2),

    compile_predicate_to_awk(gcd/3, [], AWK3),
    write_awk_script('gcd.awk', AWK3),

    compile_predicate_to_awk(sum_n/2, [], AWK4),
    write_awk_script('sum_n.awk', AWK4),

    format('Generated tail-recursive AWK scripts~n').

:- initialization(generate, main).
```

Test:
```bash
swipl tail_recursion.pl

echo "5" | awk -f factorial.awk     # 120
echo "10" | awk -f fibonacci.awk    # 55
echo "48 18" | awk -f gcd.awk       # 6
echo "10" | awk -f sum_n.awk        # 55
```

## Summary

In this chapter, you learned:

- What tail recursion is and why it matters
- How tail recursion compiles to AWK while loops
- The accumulator pattern for building results
- Classic examples: factorial, fibonacci, GCD, sum
- Limitations: only tail recursion, no mutual recursion
- How to identify tail position in clause bodies

## Next Chapter

In Chapter 6, we'll explore regex pattern matching with the `match/2`, `match/3`, and `match/4` predicates.
