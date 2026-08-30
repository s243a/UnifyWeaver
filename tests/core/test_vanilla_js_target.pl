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
:- use_module('../../src/unifyweaver/targets/vanilla_js_target').
:- use_module('../../src/unifyweaver/core/recursive_compiler').

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

:- end_tests(vanilla_js_target).
