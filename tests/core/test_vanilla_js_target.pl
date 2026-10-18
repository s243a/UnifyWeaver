:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_vanilla_js_target.pl - plunit tests for the Vanilla JS pattern target.
%
% Covers: a fact, each recursion pattern (tail / linear / list_fold /
% transitive_closure), a module, and assertions that the emitted code contains
% NO TypeScript-only syntax (`: number`, `interface `, `<T>`, non-null `!`) and
% is structurally valid JavaScript (balanced delimiters).

:- module(test_vanilla_js_target, [test_vanilla_js_target/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../../src/unifyweaver/targets/vanilla_js_target').
:- use_module('../../src/unifyweaver/core/recursive_compiler').
% Loading the advanced recursion compiler makes the multifile hook modules
% (tail_recursion, tree_recursion, ...) available so the G-P10 dispatch tests
% below can call and inspect the compile_*_pattern(vanilla_js, ...) clauses.
:- use_module('../../src/unifyweaver/core/advanced/advanced_recursive_compiler', []).

test_vanilla_js_target :-
    (   run_tests([vanilla_js_target])
    ->  format(user_output, '~n[test_vanilla_js_target] ALL TESTS PASSED~n', [])
    ;   format(user_output, '~n[test_vanilla_js_target] SOME TESTS FAILED~n', []),
        fail
    ).

:- begin_tests(vanilla_js_target).

% --- helpers ---------------------------------------------------------------

has(Code, Substr)   :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

%% no_ts_syntax(+Code) — assert none of the TypeScript-only surface remains.
no_ts_syntax(Code) :-
    hasnt(Code, ": number"),
    hasnt(Code, ": string"),
    hasnt(Code, ": boolean"),
    hasnt(Code, ": void"),
    hasnt(Code, "interface "),
    hasnt(Code, "<T>"),
    hasnt(Code, "<T, R>"),
    hasnt(Code, "<number"),
    hasnt(Code, "<string"),
    hasnt(Code, ")!"),
    hasnt(Code, " as any").

%% valid_js(+Code) — lightweight structural validity: balanced () {} [].
valid_js(Code) :-
    string_chars(Code, Chars),
    balanced(Chars, 0, 0, 0).

balanced([], P, B, S) :- P =:= 0, B =:= 0, S =:= 0.
balanced([C|Cs], P, B, S) :-
    ( C == '('  -> P1 is P+1, B1 = B, S1 = S
    ; C == ')'  -> P1 is P-1, B1 = B, S1 = S, P1 >= 0
    ; C == '{'  -> B1 is B+1, P1 = P, S1 = S
    ; C == '}'  -> B1 is B-1, P1 = P, S1 = S, B1 >= 0
    ; C == '['  -> S1 is S+1, P1 = P, B1 = B
    ; C == ']'  -> S1 is S-1, P1 = P, B1 = B, S1 >= 0
    ; P1 = P, B1 = B, S1 = S
    ),
    balanced(Cs, P1, B1, S1).

% --- target metadata -------------------------------------------------------

test(target_info) :-
    vanilla_js_target:target_info(Info),
    Info.name == "VanillaJS",
    Info.family == javascript,
    Info.file_extension == ".js",
    memberchk(plain, Info.features),
    memberchk(transitive_closure, Info.recursion_patterns).

% --- the centralized type-strip rewrite ------------------------------------

test(type_strip_synthetic) :-
    TS = "export interface FooFact {\n  arg1: string;\n}\n\nexport const f = (n: number, xs: string[]): number => {\n  const m = new Map<number, number>();\n  return m.get(n)! + (n as any);\n};\nconst g = <T, R>(x: T): R => x;\n",
    vanilla_js_target:vanilla_js_type_strip(TS, Js),
    no_ts_syntax(Js),
    has(Js, "const f = (n, xs) =>"),
    has(Js, "new Map()"),
    has(Js, "m.get(n)"),
    valid_js(Js).

% --- G-P14: union types / namespace import / inline arrow param ------------
%
% Rewrite-robustness cases found during G-P8 streaming. Union return/param
% types (`: number | null`) must be stripped whole (no leftover `| null` in
% type position); namespace imports and inline arrow-param annotations must
% come out as valid JS.

test(type_strip_union_return_and_param) :-
    TS = "export const f = (n: number, m: string | null): number | null => {\n  return n > 0 ? n : null;\n};\n",
    vanilla_js_target:vanilla_js_type_strip(TS, Js),
    has(Js, "const f = (n, m) =>"),
    hasnt(Js, "| null"),
    hasnt(Js, ": string"),
    hasnt(Js, ": number"),
    % the ternary `: null` (not a type annotation) must be preserved
    has(Js, "n > 0 ? n : null"),
    no_ts_syntax(Js),
    valid_js(Js).

test(type_strip_namespace_import) :-
    TS = "import * as readline from \"readline\";\nconst rl = readline.createInterface({ input: process.stdin });\n",
    vanilla_js_target:vanilla_js_type_strip(TS, Js),
    has(Js, "import * as readline from \"readline\";"),
    no_ts_syntax(Js),
    valid_js(Js).

test(type_strip_inline_arrow_param) :-
    TS = "rl.on(\"line\", (line: string) => {\n  addFact(line);\n});\n",
    vanilla_js_target:vanilla_js_type_strip(TS, Js),
    has(Js, "(line) =>"),
    hasnt(Js, ": string"),
    no_ts_syntax(Js),
    valid_js(Js).

% the stripped union-return shape parses and runs under stock node
test(union_return_runs_on_node) :-
    TSu = "export const f = (n: number): number | null => (n > 0 ? n : null);\nconsole.log(f(5));\n",
    vanilla_js_target:vanilla_js_type_strip(TSu, Js),
    hasnt(Js, "| null"),
    hasnt(Js, ": number"),
    run_node(Js, 'x', Output),
    atom_number(Output, 5).

% --- a fact ----------------------------------------------------------------

test(fact) :-
    assertz(user:colour(red)),
    assertz(user:colour(blue)),
    vanilla_js_target:compile_facts(colour, 1, Code),
    has(Code, "colourFacts"),
    no_ts_syntax(Code),
    valid_js(Code),
    retractall(user:colour(_)).

% --- recursion patterns ----------------------------------------------------

test(tail_recursion) :-
    vanilla_js_target:compile_recursion(sum/2, [pattern(tail_recursion)], Code),
    has(Code, "const sum = (n, acc = 0) =>"),
    no_ts_syntax(Code),
    valid_js(Code).

test(list_fold) :-
    vanilla_js_target:compile_recursion(total/2, [pattern(list_fold)], Code),
    has(Code, ".reduce("),
    no_ts_syntax(Code),
    valid_js(Code).

test(linear_recursion) :-
    vanilla_js_target:compile_recursion(fib/2, [pattern(linear_recursion)], Code),
    has(Code, "new Map()"),
    no_ts_syntax(Code),
    valid_js(Code).

% --- a module (multiple predicates) ----------------------------------------

test(module) :-
    vanilla_js_target:compile_module(
        [pred(sum, 2, tail_recursion), pred(factorial, 1, factorial)],
        [module_name('PrologMath')],
        Code),
    has(Code, "factorial"),
    has(Code, "sum"),
    no_ts_syntax(Code),
    valid_js(Code).

% --- transitive closure (real recursive_compiler path + strip) -------------

test(transitive_closure) :-
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(user:edge(c, d)),
    recursive_compiler:compile_transitive_closure(
        typescript, path, 2, edge, [input(embedded)], TsCode),
    vanilla_js_target:vanilla_js_type_strip(TsCode, Code),
    has(Code, "const findAll = (start) =>"),
    has(Code, "new Map()"),
    has(Code, "new Set([start])"),
    has(Code, "addFact(\"a\", \"b\")"),
    no_ts_syntax(Code),
    valid_js(Code),
    retractall(user:edge(_, _)).

% ===========================================================================
% G-P10: advanced-pattern dispatch for target(vanilla_js)
%
% Regression coverage for the bug where vanilla_js_target registered ZERO
% compile_*_pattern multifile clauses. The advanced recursion compiler
% dispatches on the target atom, so target(vanilla_js) previously found NO
% clause and FAILED (while the sibling annotated_js worked). vanilla_js now
% registers a clause per pattern that delegates to the `typescript` clause and
% then applies vanilla_js_type_strip/2, exactly mirroring annotated_js_target.
% ===========================================================================

%% run_node(+Code, +Arg, -Output) — write vanilla JS to a temp file and run it
%  under stock node with a single CLI argument, returning trimmed stdout.
run_node(Code, Arg, Output) :-
    tmp_file(vanilla_js_node, Base),
    atom_concat(Base, '.mjs', File),
    setup_call_cleanup(
        ( open(File, write, S), write(S, Code), close(S) ),
        run_node_file(File, Arg, Output),
        catch(delete_file(File), _, true)
    ).

run_node_file(File, Arg, Output) :-
    ( number(Arg) -> term_to_atom(Arg, ArgA) ; ArgA = Arg ),
    process_create(path(node), [File, ArgA],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out), close(Err),
    process_wait(PID, _Status),
    ( ErrStr == "" -> true
    ; format(user_error, "node stderr: ~w~n", [ErrStr]) ),
    normalize_space(atom(Output), OutStr).

%% fib_oracle(+N, -F) — the Prolog oracle for the tree/fibonacci case.
fib_oracle(0, 0) :- !.
fib_oracle(1, 1) :- !.
fib_oracle(N, F) :- N > 1, N1 is N-1, N2 is N-2,
    fib_oracle(N1, F1), fib_oracle(N2, F2), F is F1+F2.

% --- proof the dispatch clauses are now REGISTERED (previously ALL absent) --

test(advanced_hooks_registered) :-
    clause(tail_recursion:compile_tail_pattern(vanilla_js,_,_,_,_,_,_,_,_), _),
    clause(linear_recursion:compile_linear_pattern(vanilla_js,_,_,_,_,_,_,_), _),
    clause(tree_recursion:compile_tree_pattern(vanilla_js,_,_,_,_,_), _),
    clause(multicall_linear_recursion:compile_multicall_pattern(vanilla_js,_,_,_,_,_), _),
    clause(direct_multi_call_recursion:compile_direct_multicall_pattern(vanilla_js,_,_,_,_), _),
    clause(mutual_recursion:compile_mutual_pattern(vanilla_js,_,_,_,_), _),
    clause(advanced_recursive_compiler:compile_general_recursive_pattern(vanilla_js,_,_,_,_,_), _).

% --- each pattern: target(vanilla_js) now DISPATCHES and yields valid JS -----

% tail / accumulator: sum-of-list loop with `acc += item`.
test(advanced_tail_dispatch) :-
    tail_recursion:compile_tail_pattern(vanilla_js, "sumTail", 3,
        [], [], 2, arithmetic(_ + _), false, Code),
    has(Code, "const sumTail = (items) =>"),
    has(Code, "acc += item"),
    no_ts_syntax(Code),
    valid_js(Code).

% linear (numeric, memoized): reads user:clause for the recursive body.
test(advanced_linear_dispatch) :-
    assertz(user:tri(0, 0)),
    assertz((user:tri(N, R) :- N > 0, N1 is N-1, tri(N1, R1), R is R1+N)),
    linear_recursion:compile_linear_pattern(vanilla_js, "tri", 2,
        [clause(tri(0,0), true)], [], false, none, Code),
    has(Code, "const tri = (n) =>"),
    has(Code, "new Map()"),
    no_ts_syntax(Code),
    valid_js(Code),
    retractall(user:tri(_, _)).

% tree (fibonacci, memoized).
test(advanced_tree_dispatch) :-
    tree_recursion:compile_tree_pattern(vanilla_js, fib, fib, 2, false, Code),
    has(Code, "const fib = (n) =>"),
    has(Code, "new Map()"),
    no_ts_syntax(Code),
    valid_js(Code).

% multi-call linear (two base cases, two recursive calls).
test(advanced_multicall_dispatch) :-
    multicall_linear_recursion:compile_multicall_pattern(vanilla_js, "mfib",
        [clause(mfib(0,0), true), clause(mfib(1,1), true)], [], false, Code),
    has(Code, "const mfib = (n) =>"),
    has(Code, "new Map()"),
    no_ts_syntax(Code),
    valid_js(Code).

% direct multi-call.
test(advanced_direct_multicall_dispatch) :-
    direct_multi_call_recursion:compile_direct_multicall_pattern(vanilla_js, "dfib",
        [clause(dfib(0,0), true), clause(dfib(1,1), true)], [], Code),
    has(Code, "const dfib = (n) =>"),
    has(Code, "new Map()"),
    no_ts_syntax(Code),
    valid_js(Code).

% mutual recursion (is_even / is_odd): reads user:clause.
test(advanced_mutual_dispatch) :-
    assertz(user:is_even(0)),
    assertz((user:is_even(N) :- N > 0, N1 is N-1, is_odd(N1))),
    assertz((user:is_odd(N) :- N > 0, N1 is N-1, is_even(N1))),
    mutual_recursion:compile_mutual_pattern(vanilla_js,
        [is_even/1, is_odd/1], true, none, Code),
    has(Code, "const is_even = (n) =>"),
    has(Code, "const is_odd = (n) =>"),
    has(Code, "new Map()"),
    no_ts_syntax(Code),
    valid_js(Code),
    retractall(user:is_even(_)),
    retractall(user:is_odd(_)).

% general / visited-set traversal (transitive closure shape).
test(advanced_general_dispatch) :-
    advanced_recursive_compiler:compile_general_recursive_pattern(vanilla_js,
        "reaches", 2,
        [(reaches(a, b), true)],
        [(reaches(X, Y), (edge(X, Z), reaches(Z, Y)))],
        Code),
    has(Code, "function reaches(arg1)"),
    no_ts_syntax(Code),
    valid_js(Code).

% --- run two dispatched patterns under node and match the Prolog oracle ------

% tail/accumulator sum: sumTail([2,4,6,8]) === 20 (oracle: sum_list/2).
test(advanced_tail_runs_on_node) :-
    tail_recursion:compile_tail_pattern(vanilla_js, "sumTail", 3,
        [], [], 2, arithmetic(_ + _), false, Code),
    no_ts_syntax(Code),
    run_node(Code, '2,4,6,8', Output),
    sum_list([2,4,6,8], Expected),
    atom_number(Output, Got),
    Got =:= Expected.

% tree/fibonacci: fib(10) === 55 (oracle: fib_oracle/2).
test(advanced_tree_runs_on_node) :-
    tree_recursion:compile_tree_pattern(vanilla_js, fib, fib, 2, false, Code),
    no_ts_syntax(Code),
    run_node(Code, 10, Output),
    fib_oracle(10, Expected),
    atom_number(Output, Got),
    Got =:= Expected.

:- end_tests(vanilla_js_target).
