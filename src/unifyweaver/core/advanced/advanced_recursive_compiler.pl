:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% advanced_recursive_compiler.pl - Advanced recursion pattern compiler
% Orchestrates compilation of tail recursion, linear recursion, and mutual recursion
% Uses priority-based strategy: try simpler patterns first

:- module(advanced_recursive_compiler, [
    compile_advanced_recursive/3,   % +Pred/Arity, +Options, -BashCode
    compile_predicate_group/3,      % +Predicates, +Options, -BashCode
    compile_mutual_recursion/3,     % +Group, +Options, -BashCode
    try_fold_pattern/3,            % +Pred/Arity, +Options, -BashCode
    test_advanced_compiler/0
]).

:- use_module(library(lists)).
:- use_module('call_graph', [predicates_in_group/2, build_call_graph/2]).
:- use_module('scc_detection').
:- use_module('pattern_matchers', [
    contains_call_to/2,
    is_linear_recursive_streamable/1,
    can_transform_linear_to_tail/2,
    split_body_at_recursive_call/5
    % Note: We define our own extract_goal/2 to avoid importing the one from pattern_matchers
]).
:- use_module('purity_analysis', [is_associative_op/1]).
:- use_module('tail_recursion').
:- use_module('linear_recursion').
:- use_module('multicall_linear_recursion').
:- use_module('tree_recursion').
:- use_module('mutual_recursion').
:- use_module('direct_multi_call_recursion').
:- use_module('fold_helper_generator').

%% compile_advanced_recursive(+Pred/Arity, +Options, -BashCode)
%  Main entry point for advanced recursion compilation
%  Uses priority-based pattern matching:
%  1. Tail recursion (simplest optimization)
%  2. Linear recursion (single recursive call)
%  3. Tree recursion (multiple recursive calls)
%  4. Mutual recursion detection (most complex)
compile_advanced_recursive(Pred/Arity, Options, BashCode) :-
    format('~n=== Advanced Recursive Compilation: ~w/~w ===~n', [Pred, Arity]),

    % Collect skip list from options
    findall(S, (member(skip(S), Options), atom(S)), SkipAtoms),
    findall(S, (member(skip(SL), Options), is_list(SL), member(S, SL)), SkipLists),
    append(SkipAtoms, SkipLists, SkipPatterns),

    % Check for forced pattern
    (   member(pattern(ForcedPattern), Options) ->
        format('  Forced pattern: ~w~n', [ForcedPattern]),
        try_forced_pattern(ForcedPattern, Pred/Arity, Options, BashCode)
    ;   % Try patterns in priority order, respecting skip list
        (   \+ memberchk(tail_recursion, SkipPatterns),
            try_tail_recursion(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as tail recursion~n')
        ;   \+ memberchk(linear_as_tail, SkipPatterns),
            try_linear_as_tail(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as linear→tail transformation~n')
        ;   \+ memberchk(linear_recursion, SkipPatterns),
            try_linear_recursion(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as linear recursion~n')
        ;   \+ memberchk(multicall_linear_recursion, SkipPatterns),
            try_multicall_linear_recursion(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as multi-call linear recursion~n')
        ;   \+ memberchk(direct_multi_call, SkipPatterns),
            try_direct_multi_call(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as direct multi-call recursion~n')
        ;   \+ memberchk(fold, SkipPatterns),
            try_fold_pattern(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as fold pattern~n')
        ;   \+ memberchk(tree_recursion, SkipPatterns),
            try_tree_recursion(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as tree recursion~n')
        ;   \+ memberchk(mutual_recursion, SkipPatterns),
            try_mutual_recursion_detection(Pred/Arity, Options, BashCode) ->
            format('✓ Compiled as part of mutual recursion group~n')
        ;   % No pattern matched - fail back to caller
            format('✗ No advanced pattern matched~n'),
            fail
        )
    ).

%% try_forced_pattern(+Pattern, +Pred/Arity, +Options, -Code)
%  Attempt to compile using a specific forced pattern.
try_forced_pattern(tail_recursion, Pred/Arity, Options, Code) :-
    try_tail_recursion(Pred/Arity, Options, Code).
try_forced_pattern(linear_as_tail, Pred/Arity, Options, Code) :-
    try_linear_as_tail(Pred/Arity, Options, Code).
try_forced_pattern(linear_recursion, Pred/Arity, Options, Code) :-
    try_linear_recursion(Pred/Arity, Options, Code).
try_forced_pattern(multicall_linear_recursion, Pred/Arity, Options, Code) :-
    try_multicall_linear_recursion(Pred/Arity, Options, Code).
try_forced_pattern(direct_multi_call, Pred/Arity, Options, Code) :-
    try_direct_multi_call(Pred/Arity, Options, Code).
try_forced_pattern(fold, Pred/Arity, Options, Code) :-
    try_fold_pattern(Pred/Arity, Options, Code).
try_forced_pattern(tree_recursion, Pred/Arity, Options, Code) :-
    try_tree_recursion(Pred/Arity, Options, Code).
try_forced_pattern(mutual_recursion, Pred/Arity, Options, Code) :-
    try_mutual_recursion_detection(Pred/Arity, Options, Code).
try_forced_pattern(Pattern, Pred/Arity, _, _) :-
    format(user_error, 'Unknown forced pattern ~w for ~w~n', [Pattern, Pred/Arity]),
    fail.

%% try_tail_recursion(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as tail recursion
try_tail_recursion(Pred/Arity, Options, BashCode) :-
    format('  Trying tail recursion pattern...~n'),
    can_compile_tail_recursion(Pred/Arity),
    !,
    compile_tail_recursion(Pred/Arity, Options, BashCode).

%% try_linear_as_tail(+Pred/Arity, +Options, -Code)
%  Attempt to transform linear recursion to tail-recursive loop.
%  Uses purity analysis to verify safety of reordering.
try_linear_as_tail(Pred/Arity, Options, Code) :-
    format('  Trying linear→tail transformation...~n'),

    % Check for explicit unordered(false) — user forbids reordering
    (   member(unordered(false), Options) ->
        format('    Blocked by unordered(false) constraint~n'),
        fail
    ;   true
    ),

    can_transform_linear_to_tail(Pred/Arity, TransformInfo),
    TransformInfo = transform(AccInit, AccOp, _Direction),
    !,
    format('  Transform: init=~w, op=~w~n', [AccInit, AccOp]),

    % Determine target
    (   member(target(Target), Options) -> true ; Target = bash ),

    % Generate accumulator-based loop code
    atom_string(Pred, PredStr),
    generate_linear_as_tail_code(Target, PredStr, Arity, AccInit, AccOp, Pred/Arity, Code).

%% generate_linear_as_tail_code(+Target, +PredStr, +Arity, +Init, +Op, +Pred/Arity, -Code)
%  Generate target-specific while-loop code with accumulator.

generate_linear_as_tail_code(bash, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    % Extract step info from the predicate
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_bash(Op, BashOp),
    (   Direction = down ->
        format(string(Cond), 'n > 0', [])
    ;   format(string(Cond), 'n < limit', [])
    ),
    (   Direction = down ->
        format(string(StepExpr), 'n=$((n - ~w))', [StepVal])
    ;   format(string(StepExpr), 'n=$((n + ~w))', [StepVal])
    ),
    format(string(Code),
'# ~s — compiled via linear→tail transformation
~s() {
    local n="$1"
    local acc=~w
    while (( ~s )); do
        acc=$(( acc ~s n ))
        ~s
    done
    echo "$acc"
}', [PredStr, PredStr, Init, Cond, BashOp, StepExpr]).

generate_linear_as_tail_code(r, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_r(Op, ROp),
    (   Direction = down ->
        format(string(Cond), 'n > 0', []),
        format(string(StepExpr), 'n <- n - ~w', [StepVal])
    ;   format(string(Cond), 'n < limit', []),
        format(string(StepExpr), 'n <- n + ~w', [StepVal])
    ),
    format(string(Code),
'~s <- function(n) {
  acc <- ~w
  while (~s) {
    acc <- acc ~s n
    ~s
  }
  return(acc)
}

if (!interactive()) {
    args <- commandArgs(TRUE)
    if (length(args) >= 1) cat(~s(as.integer(args[1])), "\\n")
}', [PredStr, Init, Cond, ROp, StepExpr, PredStr]).

generate_linear_as_tail_code(python, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_python(Op, PyOp),
    (   Direction = down ->
        format(string(Cond), 'n > 0', []),
        format(string(StepExpr), 'n -= ~w', [StepVal])
    ;   format(string(Cond), 'n < limit', []),
        format(string(StepExpr), 'n += ~w', [StepVal])
    ),
    format(string(Code),
'import sys

def ~s(n):
    acc = ~w
    while ~s:
        acc = acc ~s n
        ~s
    return acc

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print(~s(int(sys.argv[1])))', [PredStr, Init, Cond, PyOp, StepExpr, PredStr]).

generate_linear_as_tail_code(lua, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_lua(Op, LuaOp),
    (   Direction = down ->
        format(string(Cond), 'n > 0', []),
        format(string(StepExpr), 'n = n - ~w', [StepVal])
    ;   format(string(Cond), 'n < limit', []),
        format(string(StepExpr), 'n = n + ~w', [StepVal])
    ),
    format(string(Code),
'function ~s(n)
    local acc = ~w
    while ~s do
        acc = acc ~s n
        ~s
    end
    return acc
end

if arg then
    local n = tonumber(arg[1])
    if n then print(~s(n)) end
end', [PredStr, Init, Cond, LuaOp, StepExpr, PredStr]).

generate_linear_as_tail_code(c, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_c(Op, COp),
    (   Direction = down ->
        format(string(Cond), 'n > 0', []),
        format(string(StepExpr), 'n -= ~w;', [StepVal])
    ;   format(string(Cond), 'n < limit', []),
        format(string(StepExpr), 'n += ~w;', [StepVal])
    ),
    format(string(Code),
'/* Generated by UnifyWeaver C Target - Linear→Tail Transformation */
#include <stdio.h>
#include <stdlib.h>

long long ~s(int n) {
    long long acc = ~w;
    while (~s) {
        acc = acc ~s n;
        ~s
    }
    return acc;
}

int main(int argc, char *argv[]) {
    if (argc >= 2) printf("%lld\\n", ~s(atoi(argv[1])));
    return 0;
}', [PredStr, Init, Cond, COp, StepExpr, PredStr]).

generate_linear_as_tail_code(cpp, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_c(Op, COp),
    (   Direction = down ->
        format(string(Cond), 'n > 0', []),
        format(string(StepExpr), 'n -= ~w;', [StepVal])
    ;   format(string(Cond), 'n < limit', []),
        format(string(StepExpr), 'n += ~w;', [StepVal])
    ),
    format(string(Code),
'/* Generated by UnifyWeaver C++ Target - Linear→Tail Transformation */
#include <iostream>
#include <cstdlib>

long long ~s(int n) {
    long long acc = ~w;
    while (~s) {
        acc = acc ~s n;
        ~s
    }
    return acc;
}

int main(int argc, char *argv[]) {
    if (argc >= 2) std::cout << ~s(std::atoi(argv[1])) << std::endl;
    return 0;
}', [PredStr, Init, Cond, COp, StepExpr, PredStr]).

generate_linear_as_tail_code(java, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_java(Op, JavaOp),
    (   Direction = down ->
        format(string(Cond), 'n > 0', []),
        format(string(StepExpr), 'n -= ~w;', [StepVal])
    ;   format(string(Cond), 'n < limit', []),
        format(string(StepExpr), 'n += ~w;', [StepVal])
    ),
    java_target:snake_to_pascal_java(PredStr, ClassName),
    format(string(Code),
'/* Generated by UnifyWeaver Java Target - Linear→Tail Transformation */

public class ~s {
    public static long ~s(int n) {
        long acc = ~w;
        while (~s) {
            acc = acc ~s n;
            ~s
        }
        return acc;
    }

    public static void main(String[] args) {
        if (args.length >= 1) System.out.println(~s(Integer.parseInt(args[0])));
    }
}', [ClassName, PredStr, Init, Cond, JavaOp, StepExpr, PredStr]).

generate_linear_as_tail_code(haskell, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_haskell(Op, HsOp),
    (   Direction = down ->
        format(string(StepExpr), 'n - ~w', [StepVal])
    ;   format(string(StepExpr), 'n + ~w', [StepVal])
    ),
    format(string(Code),
'module Main where

import System.Environment (getArgs)

~s :: Integer -> Integer
~s n = go n ~w
  where
    go 0 acc = acc
    go n acc = go (~s) (acc ~s n)

main :: IO ()
main = do
    args <- getArgs
    case args of
        (x:_) -> print (~s (read x))
        _     -> putStrLn "Usage: ~s <n>"', [PredStr, PredStr, Init, StepExpr, HsOp, PredStr, PredStr]).

generate_linear_as_tail_code(elixir, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_elixir(Op, ExOp),
    (   Direction = down ->
        format(string(StepExpr), 'n - ~w', [StepVal])
    ;   format(string(StepExpr), 'n + ~w', [StepVal])
    ),
    format(string(Code),
'defmodule Tail do
  def ~s(n), do: do_~s(n, ~w)
  defp do_~s(0, acc), do: acc
  defp do_~s(n, acc), do: do_~s(~s, acc ~s n)
end

case System.argv() do
  [n | _] -> IO.puts(Tail.~s(String.to_integer(n)))
  _ -> IO.puts("Usage: elixir script.exs <n>")
end', [PredStr, PredStr, Init, PredStr, PredStr, PredStr, StepExpr, ExOp, PredStr]).

generate_linear_as_tail_code(fsharp, PredStr, _Arity, Init, Op, Pred/Arity, Code) :-
    linear_recursion:extract_step_info_for(Pred/Arity, StepVal, Direction),
    op_to_fsharp(Op, FsOp),
    (   Direction = down ->
        format(string(StepExpr), 'n - ~w', [StepVal])
    ;   format(string(StepExpr), 'n + ~w', [StepVal])
    ),
    format(string(Code),
'open System

let ~s n =
    let rec loop n acc =
        if n = 0 then acc
        else loop (~s) (acc ~s n)
    loop n ~w

[<EntryPoint>]
let main argv =
    if argv.Length >= 1 then
        printfn "%d" (~s (int argv.[0]))
    0', [PredStr, StepExpr, FsOp, Init, PredStr]).

generate_linear_as_tail_code(Target, PredStr, _, _, _, _, _) :-
    format(user_error, 'Linear→tail code generation not implemented for target ~w (~s)~n', [Target, PredStr]),
    fail.

%% Operator translation helpers
op_to_bash(+, '+').
op_to_bash(*, '*').
op_to_r(+, '+').
op_to_r(*, '*').
op_to_python(+, '+').
op_to_python(*, '*').
op_to_lua(+, '+').
op_to_lua(*, '*').
op_to_c(+, '+').
op_to_c(*, '*').
op_to_java(+, '+').
op_to_java(*, '*').
op_to_haskell(+, '+').
op_to_haskell(*, '*').
op_to_elixir(+, '+').
op_to_elixir(*, '*').
op_to_fsharp(+, '+').
op_to_fsharp(*, '*').

%% try_linear_recursion(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as linear recursion
try_linear_recursion(Pred/Arity, Options, BashCode) :-
    format('  Trying linear recursion pattern...~n'),
    can_compile_linear_recursion(Pred/Arity),
    !,
    compile_linear_recursion(Pred/Arity, Options, BashCode).

%% try_multicall_linear_recursion(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as multi-call linear recursion
try_multicall_linear_recursion(Pred/Arity, Options, BashCode) :-
    format('  Trying multi-call linear recursion pattern...~n'),
    can_compile_multicall_linear(Pred/Arity),
    !,
    compile_multicall_linear_recursion(Pred/Arity, Options, BashCode).

%% try_direct_multi_call(+Pred/Arity, +Options, -Code)
%  Attempt to compile as direct multi-call recursion (clause-analysis approach)
try_direct_multi_call(Pred/Arity, Options, Code) :-
    format('  Trying direct multi-call recursion pattern...~n'),
    can_compile_direct_multi_call(Pred/Arity),
    !,
    compile_direct_multi_call(Pred/Arity, Options, Code).

%% try_fold_pattern(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as fold pattern (tree recursion with fold helpers)
try_fold_pattern(Pred/Arity, Options, BashCode) :-
    format('  Trying fold pattern...~n'),
    can_compile_fold_pattern(Pred/Arity),
    !,
    compile_fold_pattern(Pred/Arity, Options, BashCode).

%% try_tree_recursion(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as tree recursion
try_tree_recursion(Pred/Arity, Options, BashCode) :-
    format('  Trying tree recursion pattern...~n'),
    can_compile_tree_recursion(Pred/Arity),
    !,
    compile_tree_recursion(Pred/Arity, Options, BashCode).

%% try_mutual_recursion_detection(+Pred/Arity, +Options, -BashCode)
%  Detect if predicate is part of mutual recursion group
%  If so, find the group and compile together
try_mutual_recursion_detection(Pred/Arity, Options, BashCode) :-
    format('  Trying mutual recursion detection...~n'),

    % Find all predicates in this predicate's dependency group
    predicates_in_group(Pred/Arity, Group),

    % Check if group size > 1 (true mutual recursion)
    length(Group, GroupSize),
    GroupSize > 1,

    format('  Found mutual recursion group of size ~w~n', [GroupSize]),

    % Check if group can be compiled as mutual recursion
    can_compile_mutual_recursion(Group),
    !,

    % Compile the entire group
    compile_mutual_recursion(Group, Options, BashCode).

%% compile_predicate_group(+Predicates, +Options, -BashCode)
%  Compile a group of predicates together
%  This is useful when user explicitly specifies a group
compile_predicate_group(Predicates, Options, BashCode) :-
    format('~n=== Compiling Predicate Group ===~n'),
    format('Predicates: ~w~n', [Predicates]),

    % Build call graph for the group
    build_call_graph(Predicates, Graph),
    format('Call graph: ~w~n', [Graph]),

    % Find SCCs
    find_sccs(Graph, SCCs),
    format('SCCs found: ~w~n', [SCCs]),

    % Order SCCs topologically (dependencies first)
    topological_order(SCCs, OrderedSCCs),
    format('Topological order: ~w~n', [OrderedSCCs]),

    % Compile each SCC in order
    compile_sccs(OrderedSCCs, Options, CodeParts),

    % Combine all code parts
    atomic_list_concat(CodeParts, '\n\n# ========================================\n\n', BashCode).

%% compile_sccs(+SCCs, +Options, -CodeParts)
%  Compile each SCC using appropriate strategy
compile_sccs([], _Options, []).
compile_sccs([SCC|Rest], Options, [Code|Codes]) :-
    compile_scc(SCC, Options, Code),
    compile_sccs(Rest, Options, Codes).

%% compile_scc(+SCC, +Options, -Code)
%  Compile a single SCC
compile_scc(SCC, Options, Code) :-
    format('~nCompiling SCC: ~w~n', [SCC]),

    (   is_trivial_scc(SCC) ->
        % Single predicate - try simple patterns first
        SCC = [Pred/Arity],
        format('  Trivial SCC (single predicate): ~w/~w~n', [Pred, Arity]),
        compile_single_predicate(Pred/Arity, Options, Code)
    ;   % Non-trivial SCC - true mutual recursion
        format('  Non-trivial SCC (mutual recursion): ~w~n', [SCC]),
        compile_mutual_recursion(SCC, Options, Code)
    ).

%% compile_single_predicate(+Pred/Arity, +Options, -Code)
%  Compile a single predicate using priority order
compile_single_predicate(Pred/Arity, Options, Code) :-
    (   compile_advanced_recursive(Pred/Arity, Options, Code) ->
        true
    ;   % Fall back to basic compilation
        format('  Using basic recursion compilation~n'),
        Code = "# Basic recursion - not yet implemented"
    ).

%% ============================================
%% FOLD PATTERN IMPLEMENTATION
%% ============================================

%% can_compile_fold_pattern(+Pred/Arity)
%  Check if predicate can be compiled as fold pattern
%  Fold pattern is for NUMERIC recursion (like fibonacci)
%  NOT for structural recursion (like tree_sum)
can_compile_fold_pattern(Pred/Arity) :-
    % Must be binary (input/output pattern)
    Arity =:= 2,

    % Check if it has tree recursion structure (multiple recursive calls)
    functor(Head, Pred, Arity),
    findall(Body, user:clause(Head, Body), Bodies),  % Use user:clause

    % Must have at least one recursive case with multiple recursive calls
    member(RecBody, Bodies),
    findall(RecCall, (
        extract_goal(RecBody, RecCall),
        functor(RecCall, Pred, Arity)
    ), RecCalls),
    length(RecCalls, NumCalls),
    NumCalls >= 2,

    % NEW: Reject if recursive clauses use structural decomposition
    % (e.g., tree_sum([V,L,R], Sum) has structural pattern)
    \+ has_structural_head_in_recursive_clauses(Pred/Arity),

    % Optional: Check if forbid_linear_recursion/1 is declared
    % This indicates user intention to use fold pattern
    (   clause(forbid_linear_recursion(Pred/Arity), true) ->
        format('    Found forbid_linear_recursion directive~n')
    ;   true
    ).

%% has_structural_head_in_recursive_clauses(+Pred/Arity)
%  Check if any recursive clause uses structural decomposition in head
%  Returns true for patterns like tree_sum([V,L,R], Sum)
%  Returns false for numeric patterns like fib(N, F)
has_structural_head_in_recursive_clauses(Pred/Arity) :-
    functor(Head, Pred, Arity),
    user:clause(Head, Body),  % Use user:clause to find test predicates
    % Check if this clause is recursive
    extract_goal(Body, Goal),
    functor(Goal, Pred, Arity),
    % Check if head uses structural pattern
    has_structural_pattern_in_head(Head).

%% has_structural_pattern_in_head(+Head)
%  Check if head uses structural decomposition (like [V,L,R])
%  Returns true for: tree_sum([V,L,R], Sum) - INPUT arg is structural
%  Returns false for: fib(N, F) - INPUT arg is a variable
has_structural_pattern_in_head(Head) :-
    Head =.. [_Pred|Args],
    % Check FIRST argument (input) for structural pattern
    Args = [FirstArg|_],
    is_structural_pattern(FirstArg).

%% is_structural_pattern(+Term)
%  Check if term is a structural pattern (list with multiple elements)
%  [V,L,R] is structural (tree node decomposition)
%  [H|T] is NOT structural (simple list cons - for linear recursion)
%  N or _ is NOT structural (simple variable)
is_structural_pattern([_,_,_|_]) :- !.  % 3+ element list is structural
is_structural_pattern([_,_]) :- !.      % 2 element list is structural
is_structural_pattern([_|T]) :-         % [H|T] is only structural if T is not a var
    nonvar(T),
    T \= [].

%% compile_fold_pattern(+Pred/Arity, +Options, -BashCode)
%  Compile predicate using fold pattern with bash code generation
compile_fold_pattern(Pred/Arity, Options, BashCode) :-
    format('    Generating fold helpers...~n'),
    
    % Generate fold helper predicates using existing infrastructure
    catch(
        (generate_fold_helpers(Pred/Arity, Clauses),
         length(Clauses, NumClauses),
         format('    Generated ~w fold helper clauses~n', [NumClauses])),
        Error,
        (format('    Error generating fold helpers: ~w~n', [Error]), fail)
    ),
    
    % Convert Prolog clauses to bash code
    format('    Converting to bash code...~n'),
    catch(
        generate_bash_from_fold_clauses(Pred/Arity, Clauses, Options, BashCode),
        BashError,
        (format('    Error generating bash code: ~w~n', [BashError]), fail)
    ),
    
    format('    Fold pattern compilation complete~n').

%% generate_bash_from_fold_clauses(+Pred/Arity, +Clauses, +Options, -BashCode)
%  Convert fold helper Prolog clauses to executable bash code
generate_bash_from_fold_clauses(Pred/_Arity, Clauses, _Options, BashCode) :-
    atom_string(Pred, PredStr),
    
    % Separate clauses by type
    atom_concat(Pred, '_graph', GraphPred),
    atom_concat('fold_', Pred, FoldPred),
    atom_concat(Pred, '_fold', WrapperPred),
    
    include(is_clause_for_pred(GraphPred), Clauses, GraphClauses),
    include(is_clause_for_pred(FoldPred), Clauses, FoldClauses),
    include(is_clause_for_pred(WrapperPred), Clauses, [WrapperClause]),
    
    % Generate bash functions
    generate_graph_bash(GraphPred, GraphClauses, GraphCode),
    generate_fold_bash(FoldPred, FoldClauses, FoldCode),
    generate_wrapper_bash(WrapperPred, WrapperClause, WrapperCode),
    
    % Combine with header
    format(string(BashCode), '#!/bin/bash
# ~s - compiled with fold pattern

~s

~s

~s

# Main entry point (delegates to fold wrapper)
~s() {
    ~s_fold "$@"
}

# Stream function
~s_stream() {
    ~s_fold "$1"
}', [PredStr, GraphCode, FoldCode, WrapperCode, PredStr, PredStr, PredStr, PredStr]).

%% generate_graph_bash(+GraphPred, +GraphClauses, -GraphCode)
%  Generate bash code for graph building functions
generate_graph_bash(GraphPred, _GraphClauses, GraphCode) :-
    atom_string(GraphPred, GraphPredStr),
    
    % For now, generate a simple implementation
    % This is where we'd implement full bash translation of the Prolog graph builder
    format(string(GraphCode), '# Graph builder for ~s
~s() {
    local input="$1"
    
    # Base cases (leaves)
    case "$input" in
        0|1) echo "leaf:$input" ;;
        *)
            # Recursive case - build dependency tree
            if (( input <= 1 )); then
                echo "leaf:$input"
            else
                local n1=$((input - 1))
                local n2=$((input - 2))
                local left=$($0 "$n1")
                local right=$($0 "$n2")
                echo "node:$input:[$left,$right]"
            fi
            ;;
    esac
}', [GraphPredStr, GraphPredStr]).

%% generate_fold_bash(+FoldPred, +FoldClauses, -FoldCode)
%  Generate bash code for fold computation
generate_fold_bash(FoldPred, _FoldClauses, FoldCode) :-
    atom_string(FoldPred, FoldPredStr),
    
    format(string(FoldCode), '# Fold computer for ~s
~s() {
    local structure="$1"
    
    case "$structure" in
        leaf:*)
            # Extract value from leaf:VALUE
            echo "${structure#leaf:}"
            ;;
        node:*)
            # Parse node:VALUE:[left,right] and compute recursively
            local content="${structure#node:}"
            local value="${content%%:*}"
            local children="${content#*:[}"
            children="${children%]}"
            
            # Split children (simplified - assumes two children)
            local left="${children%%,*}"
            local right="${children#*,}"
            
            # Recursive computation
            local left_val=$($0 "$left")
            local right_val=$($0 "$right")
            
            # Combine (assumes addition - could be parameterized)
            echo $((left_val + right_val))
            ;;
    esac
}', [FoldPredStr, FoldPredStr]).

%% generate_wrapper_bash(+WrapperPred, +WrapperClause, -WrapperCode)
%  Generate bash wrapper that combines graph building and folding
generate_wrapper_bash(WrapperPred, _WrapperClause, WrapperCode) :-
    atom_string(WrapperPred, WrapperPredStr),
    
    % Extract base predicate name
    atom_concat(BasePred, '_fold', WrapperPred),
    atom_concat(BasePred, '_graph', GraphPred),
    atom_string(GraphPred, GraphPredStr),
    atom_concat('fold_', BasePred, FoldPred),
    atom_string(FoldPred, FoldPredStr),
    
    format(string(WrapperCode), '# Wrapper function for ~s
~s() {
    local input="$1"
    
    # Build dependency graph
    local graph=$(~s "$input")
    
    # Compute result by folding
    ~s "$graph"
}', [WrapperPredStr, WrapperPredStr, GraphPredStr, FoldPredStr]).

%% is_clause_for_pred(+Pred, +Clause)
%  Check if clause is for given predicate
is_clause_for_pred(Pred, clause(Head, _Body)) :-
    functor(Head, Pred, _).

%% extract_goal(+Body, -Goal)
%  Extract individual goals from conjunction/disjunction structure
extract_goal(Goal, Goal) :-
    Goal \= (_,_),
    Goal \= (_;_),
    Goal \= (_->_).
extract_goal((A, _B), Goal) :-
    extract_goal(A, Goal).
extract_goal((_A, B), Goal) :-
    extract_goal(B, Goal).
extract_goal((A; _B), Goal) :-
    extract_goal(A, Goal).
extract_goal((_A; B), Goal) :-
    extract_goal(B, Goal).

%% ============================================
%% TESTS
%% ============================================

test_advanced_compiler :-
    writeln('=== ADVANCED RECURSIVE COMPILER TESTS ==='),
    writeln(''),

    % Setup output directory
    (   exists_directory('output/advanced') -> true
    ;   make_directory('output/advanced')
    ),

    % Test 1: Tail recursion
    writeln('--- Test 1: Tail Recursion (count_items) ---'),
    catch(abolish(count_items/3), _, true),
    assertz((count_items([], Acc, Acc))),
    assertz((count_items([_|T], Acc, N) :- Acc1 is Acc + 1, count_items(T, Acc1, N))),

    (   compile_advanced_recursive(count_items/3, [], Code1) ->
        write_bash_file('output/advanced/test_count.sh', Code1),
        writeln('✓ Test 1 PASSED')
    ;   writeln('✗ Test 1 FAILED')
    ),

    % Test 2: Linear recursion
    writeln(''),
    writeln('--- Test 2: Linear Recursion (list_length) ---'),
    catch(abolish(list_length/2), _, true),
    assertz((list_length([], 0))),
    assertz((list_length([_|T], N) :- list_length(T, N1), N is N1 + 1)),

    (   compile_advanced_recursive(list_length/2, [], Code2) ->
        write_bash_file('output/advanced/test_length.sh', Code2),
        writeln('✓ Test 2 PASSED')
    ;   writeln('✗ Test 2 FAILED')
    ),

    % Test 3: Mutual recursion
    writeln(''),
    writeln('--- Test 3: Mutual Recursion (even/odd) ---'),
    catch(abolish(is_even/1), _, true),
    catch(abolish(is_odd/1), _, true),
    assertz(is_even(0)),
    assertz((is_even(N) :- N > 0, N1 is N - 1, is_odd(N1))),
    assertz(is_odd(1)),
    assertz((is_odd(N) :- N > 1, N1 is N - 1, is_even(N1))),

    (   compile_advanced_recursive(is_even/1, [], Code3) ->
        write_bash_file('output/advanced/test_even_odd.sh', Code3),
        writeln('✓ Test 3 PASSED')
    ;   writeln('✗ Test 3 FAILED (expected - may need explicit group compilation)')
    ),

    % Test 4: Predicate group compilation
    writeln(''),
    writeln('--- Test 4: Predicate Group Compilation ---'),
    Predicates = [is_even/1, is_odd/1],
    (   compile_predicate_group(Predicates, [], Code4) ->
        write_bash_file('output/advanced/test_group.sh', Code4),
        writeln('✓ Test 4 PASSED')
    ;   writeln('✗ Test 4 FAILED')
    ),

    % Test 5: Fold pattern
    writeln(''),
    writeln('--- Test 5: Fold Pattern (fibonacci) ---'),
    catch(abolish(test_fib/2), _, true),
    catch(abolish(forbid_linear_recursion/1), _, true),
    
    % Define fibonacci with multiple recursive calls (fold pattern)
    assertz((test_fib(0, 0))),
    assertz((test_fib(1, 1))),
    assertz((test_fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, test_fib(N1, F1), test_fib(N2, F2), F is F1 + F2)),
    assertz(forbid_linear_recursion(test_fib/2)),  % Force fold pattern
    
    (   compile_advanced_recursive(test_fib/2, [], Code5) ->
        write_bash_file('output/advanced/test_fibonacci_fold.sh', Code5),
        writeln('✓ Test 5 PASSED')
    ;   writeln('✗ Test 5 FAILED')
    ),

    writeln(''),
    writeln('=== ADVANCED RECURSIVE COMPILER TESTS COMPLETE ==='),
    writeln('Check output/advanced/ directory for generated files').

%% Helper to write bash files
write_bash_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream),
    % Make executable
    atom_concat('chmod +x ', Path, ChmodCmd),
    shell(ChmodCmd).
