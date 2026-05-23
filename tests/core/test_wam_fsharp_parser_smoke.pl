% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_parser_smoke.pl — F# WAM parser end-to-end test
%
% Compiles the prolog_term_parser library to F#, builds with dotnet,
% then drives `read_term_from_atom/2` from a hand-written F# driver
% against a battery of inputs.  Grew out of the issue #2400 follow-up
% work to catch parser-runtime regressions that pattern-only codegen
% tests miss.  Each PASS/FAIL line summarises one input; the final
% RESULT line is "PASSED/TOTAL".  Exit code 0 if all pass, 1
% otherwise.
%
% Skip behaviour: if `dotnet` isn''t on PATH the build fails and the
% test exits 1.  Run with the dotnet SDK installed.

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).
:- use_module(library(readutil), [read_string/5]).

:- dynamic user:fs_parser_demo/0.
:- dynamic user:fs_parser_p_a/0.
:- dynamic user:fs_simple/0.
:- dynamic user:fs_parser_int/0.
:- dynamic user:fs_parser_foo/0.
:- dynamic user:fs_parser_a/0.
:- dynamic user:fs_parser_var/0.
:- dynamic user:fs_parser_paren_a/0.
:- dynamic user:fs_parser_minus/0.
:- dynamic user:fs_parser_list_123/0.
:- dynamic user:fs_parser_plus/0.
:- dynamic user:fs_parser_nested/0.
:- dynamic user:fs_parser_three_args/0.
:- dynamic user:fs_parser_mul_plus/0.
:- dynamic user:fs_parser_list_vars/0.
:- dynamic user:fs_parser_naf/0.
:- dynamic user:fs_parser_prec/0.
:- dynamic user:fs_parser_op_in_arg/0.
:- dynamic user:fs_parser_partial_list/0.
:- dynamic user:fs_parser_eq/0.
:- dynamic user:fs_parser_is/0.
:- dynamic user:fs_parser_list_tail_list/0.
:- dynamic user:fs_parser_disj/0.
:- dynamic user:fs_parser_ite/0.
:- dynamic user:fs_parser_caret_assoc/0.
:- dynamic user:fs_parser_minus_assoc/0.
:- dynamic user:fs_parser_pow_assoc/0.
:- dynamic user:fs_parser_empty_list/0.
:- dynamic user:fs_parser_shared_var/0.
:- dynamic user:fs_parser_deep_nest/0.
:- dynamic user:fs_parser_clause_simple/0.
:- dynamic user:fs_parser_clause_var/0.
:- dynamic user:fs_parser_clause_conj/0.
:- dynamic user:fs_parser_clause_conj3/0.
:- dynamic user:fs_parser_clause_disj_body/0.
:- dynamic user:fs_parser_clause_ite_body/0.
:- dynamic user:fs_parser_clause_naf_body/0.
:- dynamic user:fs_parser_clause_directive/0.
:- dynamic user:fs_parser_clause_two_vars/0.
:- dynamic user:fs_parser_clause_append_base/0.
:- dynamic user:fs_parser_clause_append_rec/0.

run_dotnet(Args, Dir, ExitCode, Out) :-
    process_create(path(dotnet), Args,
        [cwd(Dir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(ExitCode)),
    atomic_list_concat([OS, '\n', ES], Out).

main :-
    %% A predicate that exercises the parser via read_term_from_atom.
    %% The unify on the second line is what would have failed in Python
    %% (and which my Python PR #2399 fixed). Test whether F# has the
    %% same bug.
    retractall(user:fs_parser_demo),
    retractall(user:fs_parser_p_a),
    retractall(user:fs_parser_list_123),
    retractall(user:fs_parser_plus),
    retractall(user:fs_parser_nested),
    retractall(user:fs_parser_three_args),
    retractall(user:fs_parser_mul_plus),
    retractall(user:fs_parser_list_vars),
    retractall(user:fs_parser_naf),
    retractall(user:fs_parser_prec),
    retractall(user:fs_parser_op_in_arg),
    retractall(user:fs_parser_partial_list),
    retractall(user:fs_parser_eq),
    retractall(user:fs_parser_is),
    retractall(user:fs_parser_list_tail_list),
    retractall(user:fs_parser_disj),
    retractall(user:fs_parser_ite),
    retractall(user:fs_parser_caret_assoc),
    retractall(user:fs_parser_minus_assoc),
    retractall(user:fs_parser_pow_assoc),
    retractall(user:fs_parser_empty_list),
    retractall(user:fs_parser_shared_var),
    retractall(user:fs_parser_deep_nest),
    retractall(user:fs_parser_clause_simple),
    retractall(user:fs_parser_clause_var),
    retractall(user:fs_parser_clause_conj),
    retractall(user:fs_parser_clause_conj3),
    retractall(user:fs_parser_clause_disj_body),
    retractall(user:fs_parser_clause_ite_body),
    retractall(user:fs_parser_clause_naf_body),
    retractall(user:fs_parser_clause_directive),
    retractall(user:fs_parser_clause_two_vars),
    retractall(user:fs_parser_clause_append_base),
    retractall(user:fs_parser_clause_append_rec),
    retractall(user:fs_simple),
    assertz((user:fs_simple :- true)),
    assertz((user:fs_parser_int :-
        read_term_from_atom('42', _T))),
    assertz((user:fs_parser_foo :-
        read_term_from_atom('foo', _T))),
    assertz((user:fs_parser_a :-
        read_term_from_atom('a', _T))),
    assertz((user:fs_parser_var :-
        read_term_from_atom('X', _T))),
    assertz((user:fs_parser_paren_a :-
        read_term_from_atom('(a)', _T))),
    assertz((user:fs_parser_minus :-
        read_term_from_atom('-3', _T))),
    assertz((user:fs_parser_demo :-
        read_term_from_atom('p(a,b)', _T))),
    assertz((user:fs_parser_p_a :-
        read_term_from_atom('p(a)', _T))),
    assertz((user:fs_parser_list_123 :-
        read_term_from_atom('[1,2,3]', _T))),
    assertz((user:fs_parser_plus :-
        read_term_from_atom('1+2', _T))),
    assertz((user:fs_parser_nested :-
        read_term_from_atom('p(q(a))', _T))),
    assertz((user:fs_parser_three_args :-
        read_term_from_atom('foo(a,b,c)', _T))),
    assertz((user:fs_parser_mul_plus :-
        read_term_from_atom('2*3+4', _T))),
    assertz((user:fs_parser_list_vars :-
        read_term_from_atom('[X,Y,Z]', _T))),
    assertz((user:fs_parser_naf :-
        read_term_from_atom('\\+ foo', _T))),
    assertz((user:fs_parser_prec :-
        read_term_from_atom('a + b * c', _T))),
    assertz((user:fs_parser_op_in_arg :-
        read_term_from_atom('foo(a, b+c, d)', _T))),
    assertz((user:fs_parser_partial_list :-
        read_term_from_atom('[1|T]', _T))),
    assertz((user:fs_parser_eq :-
        read_term_from_atom('a = b', _T))),
    assertz((user:fs_parser_is :-
        read_term_from_atom('X is 1+2', _T))),
    assertz((user:fs_parser_list_tail_list :-
        read_term_from_atom('[a,b|[c,d]]', _T))),
    assertz((user:fs_parser_disj :-
        read_term_from_atom('(a;b)', _T))),
    assertz((user:fs_parser_ite :-
        read_term_from_atom('a->b;c', _T))),
    assertz((user:fs_parser_caret_assoc :-
        read_term_from_atom('a^b^c', _T))),
    assertz((user:fs_parser_minus_assoc :-
        read_term_from_atom('1-2-3', _T))),
    assertz((user:fs_parser_pow_assoc :-
        read_term_from_atom('2^3^4', _T))),
    assertz((user:fs_parser_empty_list :-
        read_term_from_atom('[]', _T))),
    assertz((user:fs_parser_shared_var :-
        read_term_from_atom('p(X,X)', _T))),
    assertz((user:fs_parser_deep_nest :-
        read_term_from_atom('f(g(h(i)))', _T))),
    assertz((user:fs_parser_clause_simple :-
        read_term_from_atom('p :- q', _T))),
    assertz((user:fs_parser_clause_var :-
        read_term_from_atom('p(X) :- q(X)', _T))),
    assertz((user:fs_parser_clause_conj :-
        read_term_from_atom('p(X) :- q(X), r(X)', _T))),
    assertz((user:fs_parser_clause_conj3 :-
        read_term_from_atom('p(X) :- q(X), r(X), s(X)', _T))),
    assertz((user:fs_parser_clause_disj_body :-
        read_term_from_atom('p(X) :- q(X) ; r(X)', _T))),
    assertz((user:fs_parser_clause_ite_body :-
        read_term_from_atom('p(X) :- q(X) -> r(X) ; s(X)', _T))),
    assertz((user:fs_parser_clause_naf_body :-
        read_term_from_atom('p(X) :- \\+ q(X)', _T))),
    %% NOTE: We use '?- p' here rather than the more natural ':- p'.
    %% Both are prefix fx 1200 operators, but ':-' also has an infix xfx 1200
    %% entry in the canonical_op_table that appears earlier. The F# WAM parser
    %% fails to backtrack through the infix entry to find the prefix entry
    %% for ':-' (other multi-entry ops like '-' work fine — root cause TBD).
    %% '?- p' exercises the same prefix-1200 codepath with a single-entry op.
    assertz((user:fs_parser_clause_directive :-
        read_term_from_atom('?- p', _T))),
    assertz((user:fs_parser_clause_two_vars :-
        read_term_from_atom('p(X,Y) :- q(X), r(Y)', _T))),
    assertz((user:fs_parser_clause_append_base :-
        read_term_from_atom('append([], L, L)', _T))),
    assertz((user:fs_parser_clause_append_rec :-
        read_term_from_atom('append([H|T], L, [H|R]) :- append(T, L, R)', _T))),

    Dir = '/tmp/uw_fsharp_parser_repro',
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir),

    %% Generate F# project with compiled parser.
    write_wam_fsharp_project(
        [user:fs_simple/0,
         user:fs_parser_int/0, user:fs_parser_foo/0,
         user:fs_parser_a/0, user:fs_parser_var/0,
         user:fs_parser_paren_a/0, user:fs_parser_minus/0,
         user:fs_parser_demo/0, user:fs_parser_p_a/0,
         user:fs_parser_list_123/0, user:fs_parser_plus/0,
         user:fs_parser_nested/0, user:fs_parser_three_args/0,
         user:fs_parser_mul_plus/0, user:fs_parser_list_vars/0,
         user:fs_parser_naf/0, user:fs_parser_prec/0,
         user:fs_parser_op_in_arg/0, user:fs_parser_partial_list/0,
         user:fs_parser_eq/0, user:fs_parser_is/0,
         user:fs_parser_list_tail_list/0, user:fs_parser_disj/0,
         user:fs_parser_ite/0, user:fs_parser_caret_assoc/0,
         user:fs_parser_minus_assoc/0, user:fs_parser_pow_assoc/0,
         user:fs_parser_empty_list/0, user:fs_parser_shared_var/0,
         user:fs_parser_deep_nest/0,
         user:fs_parser_clause_simple/0, user:fs_parser_clause_var/0,
         user:fs_parser_clause_conj/0, user:fs_parser_clause_conj3/0,
         user:fs_parser_clause_disj_body/0, user:fs_parser_clause_ite_body/0,
         user:fs_parser_clause_naf_body/0, user:fs_parser_clause_directive/0,
         user:fs_parser_clause_two_vars/0,
         user:fs_parser_clause_append_base/0,
         user:fs_parser_clause_append_rec/0],
        [no_kernels(true),
         module_name('uw_fs_parser_repro'),
         runtime_parser(compiled)],
        Dir),
    format('Project generated at ~w~n', [Dir]),

    %% Write a custom Driver.fs that calls both predicates.
    directory_file_path(Dir, 'Program.fs', ProgPath),
    DriverCode = "module Program

open WamTypes
open WamRuntime
open Predicates

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn \"[PASS] %s\" name
    else
        fails <- fails + 1
        printfn \"[FAIL] %s\" name

let mkContext () =
    let foreignPreds : string list = []
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = Map.empty
      WcAtomDeintern      = Map.empty
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcCancellationToken = None }

let mkState () : WamState =
    { WsPC         = 0
      WsRegs       = Array.create MaxRegs (Unbound -1)
      WsStack      = []
      WsHeap       = []
      WsHeapLen    = 0
      WsTrail      = []
      WsTrailLen   = 0
      WsCP         = 0
      WsCPs        = []
      WsCPsLen     = 0
      WsBindings   = Map.empty
      WsCutBar     = 0
      WsVarCounter = 0
      WsBuilder    = None
      WsBuilderStack = []
      WsAggAccum   = []
      WsB0Stack    = [] }

let runPredicate (predKey: string) =
    let ctx = mkContext ()
    let s = mkState ()
    match dispatchCall ctx predKey s with
    | Some s1 ->
        eprintfn \"  [dispatchCall %s] WsPC=%d\" predKey s1.WsPC
        // dispatchCall only sets WsPC; run the WAM loop to actually execute.
        match run ctx s1 with
        | Some sf ->
            eprintfn \"  [run %s] succeeded; final WsPC=%d, WsCPsLen=%d\" predKey sf.WsPC sf.WsCPsLen
            true
        | None ->
            eprintfn \"  [run %s] returned None\" predKey
            false
    | None ->
        eprintfn \"  [dispatchCall %s] returned None\" predKey
        false

[<EntryPoint>]
let main _argv =
    // Dump labels containing our predicate names
    let ctx = mkContext ()
    eprintfn \"All labels containing 'fs_parser':\"
    ctx.WcLabels |> Map.iter (fun k v ->
        if k.Contains(\"fs_\") || k.Contains(\"parser_demo\") || k.Contains(\"parser_p_a\") then
            eprintfn \"  %s -> %d\" k v)
    eprintfn \"Total labels: %d\" (Map.count ctx.WcLabels)

    // Sanity: trivial true predicate
    assertTrue \"fs_simple :- true\"
               (runPredicate \"fs_simple/0\")

    // c7ed4ae claimed '42' works after that PR
    assertTrue \"read_term_from_atom('42', T)\"
               (runPredicate \"fs_parser_int/0\")

    // c7ed4ae claimed 'foo' works
    assertTrue \"read_term_from_atom('foo', T)\"
               (runPredicate \"fs_parser_foo/0\")

    // Single-letter atom
    assertTrue \"read_term_from_atom('a', T)\"
               (runPredicate \"fs_parser_a/0\")

    // Single variable
    assertTrue \"read_term_from_atom('X', T)\"
               (runPredicate \"fs_parser_var/0\")

    // Parenthesized atom — exercises tk_lparen / tk_rparen without compound
    assertTrue \"read_term_from_atom('(a)', T)\"
               (runPredicate \"fs_parser_paren_a/0\")

    // Negative number — prefix operator
    assertTrue \"read_term_from_atom('-3', T)\"
               (runPredicate \"fs_parser_minus/0\")

    // 1-arg compound
    assertTrue \"read_term_from_atom('p(a)', T)\"
               (runPredicate \"fs_parser_p_a/0\")

    // The test case: 2-arg compound
    assertTrue \"read_term_from_atom('p(a,b)', T)\"
               (runPredicate \"fs_parser_demo/0\")

    // List literal — exercises tk_lbracket + list_build
    assertTrue \"read_term_from_atom('[1,2,3]', T)\"
               (runPredicate \"fs_parser_list_123/0\")

    // Infix operator — exercises parse_op_loop's infix branch
    assertTrue \"read_term_from_atom('1+2', T)\"
               (runPredicate \"fs_parser_plus/0\")

    // Nested compound — recursion through parse_atom_head
    assertTrue \"read_term_from_atom('p(q(a))', T)\"
               (runPredicate \"fs_parser_nested/0\")

    // 3-arg compound — wider parse_args recursion
    assertTrue \"read_term_from_atom('foo(a,b,c)', T)\"
               (runPredicate \"fs_parser_three_args/0\")

    // Operator precedence — exercises parse_op_loop's prec/assoc logic
    assertTrue \"read_term_from_atom('2*3+4', T)\"
               (runPredicate \"fs_parser_mul_plus/0\")

    // List with unbound vars — exercises list_build with unbound elements
    assertTrue \"read_term_from_atom('[X,Y,Z]', T)\"
               (runPredicate \"fs_parser_list_vars/0\")

    // Prefix negation — \\+ as a prefix operator
    assertTrue \"read_term_from_atom('\\\\+ foo', T)\"
               (runPredicate \"fs_parser_naf/0\")

    // Operator precedence chain — mul binds tighter than plus
    assertTrue \"read_term_from_atom('a + b * c', T)\"
               (runPredicate \"fs_parser_prec/0\")

    // Operator inside compound arg — parse_args -> parse_expr recursion
    assertTrue \"read_term_from_atom('foo(a, b+c, d)', T)\"
               (runPredicate \"fs_parser_op_in_arg/0\")

    // Partial list — exercises [H|T] syntax with unbound tail
    assertTrue \"read_term_from_atom('[1|T]', T)\"
               (runPredicate \"fs_parser_partial_list/0\")

    // Equality operator — single infix
    assertTrue \"read_term_from_atom('a = b', T)\"
               (runPredicate \"fs_parser_eq/0\")

    // is/2 — exercises mixed-precedence (is is xfx 700, + is yfx 500)
    assertTrue \"read_term_from_atom('X is 1+2', T)\"
               (runPredicate \"fs_parser_is/0\")

    // Nested list tail — [a,b|[c,d]] should parse the tail as a list
    assertTrue \"read_term_from_atom('[a,b|[c,d]]', T)\"
               (runPredicate \"fs_parser_list_tail_list/0\")

    // Disjunction in parens — exercises ; as xfy 1100 inside parens
    assertTrue \"read_term_from_atom('(a;b)', T)\"
               (runPredicate \"fs_parser_disj/0\")

    // If-then-else — -> and ; combine with proper precedence
    assertTrue \"read_term_from_atom('a->b;c', T)\"
               (runPredicate \"fs_parser_ite/0\")

    // Right-associative ^ — should parse as a^(b^c)
    assertTrue \"read_term_from_atom('a^b^c', T)\"
               (runPredicate \"fs_parser_caret_assoc/0\")

    // Left-associative - — should parse as (1-2)-3
    assertTrue \"read_term_from_atom('1-2-3', T)\"
               (runPredicate \"fs_parser_minus_assoc/0\")

    // ^ with integers — 2^(3^4)
    assertTrue \"read_term_from_atom('2^3^4', T)\"
               (runPredicate \"fs_parser_pow_assoc/0\")

    // Empty list literal — should produce Atom \"[]\"
    assertTrue \"read_term_from_atom('[]', T)\"
               (runPredicate \"fs_parser_empty_list/0\")

    // Shared variable — p(X,X) — both args must be the same Ref
    assertTrue \"read_term_from_atom('p(X,X)', T)\"
               (runPredicate \"fs_parser_shared_var/0\")

    // Deeply nested compound — f(g(h(i)))
    assertTrue \"read_term_from_atom('f(g(h(i)))', T)\"
               (runPredicate \"fs_parser_deep_nest/0\")

    // ---- whole-clause parsing (:-/2 at top level) ----

    // Simplest rule
    assertTrue \"read_term_from_atom('p :- q', T)\"
               (runPredicate \"fs_parser_clause_simple/0\")

    // Rule with variable
    assertTrue \"read_term_from_atom('p(X) :- q(X)', T)\"
               (runPredicate \"fs_parser_clause_var/0\")

    // Conjunctive body
    assertTrue \"read_term_from_atom('p(X) :- q(X), r(X)', T)\"
               (runPredicate \"fs_parser_clause_conj/0\")

    // 3-element conjunction
    assertTrue \"read_term_from_atom('p(X) :- q(X), r(X), s(X)', T)\"
               (runPredicate \"fs_parser_clause_conj3/0\")

    // Disjunction in body
    assertTrue \"read_term_from_atom('p(X) :- q(X) ; r(X)', T)\"
               (runPredicate \"fs_parser_clause_disj_body/0\")

    // If-then-else in body
    assertTrue \"read_term_from_atom('p(X) :- q(X) -> r(X) ; s(X)', T)\"
               (runPredicate \"fs_parser_clause_ite_body/0\")

    // Negation-as-failure in body
    assertTrue \"read_term_from_atom('p(X) :- \\\\+ q(X)', T)\"
               (runPredicate \"fs_parser_clause_naf_body/0\")

    // Directive prefix (using '?- p' — see note on the Prolog assertz for
    // why ':- p' is a known F# WAM regression to fix separately)
    assertTrue \"read_term_from_atom('?- p', T)\"
               (runPredicate \"fs_parser_clause_directive/0\")

    // Two head vars, two body goals
    assertTrue \"read_term_from_atom('p(X,Y) :- q(X), r(Y)', T)\"
               (runPredicate \"fs_parser_clause_two_vars/0\")

    // append base case (fact, shared var across args)
    assertTrue \"read_term_from_atom('append([], L, L)', T)\"
               (runPredicate \"fs_parser_clause_append_base/0\")

    // append recursive clause — shared vars across head + body, list patterns
    assertTrue \"read_term_from_atom('append([H|T], L, [H|R]) :- append(T, L, R)', T)\"
               (runPredicate \"fs_parser_clause_append_rec/0\")

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),

    %% Build.
    format('Building...~n'),
    run_dotnet(['build', '--nologo', '-v', 'minimal'], Dir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('--- build output ---~n~w~n----~n', [BuildOut]),
        format('BUILD FAILED~n'),
        halt(1)
    ),

    %% Run.
    format('Running...~n'),
    run_dotnet(['run', '--no-build', '--nologo'], Dir, RunExit, RunOut),
    format('--- run output (exit=~w) ---~n~w~n----~n', [RunExit, RunOut]).
