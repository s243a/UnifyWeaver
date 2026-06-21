% test_wam_fsharp_lowered_t6.pl
%
% End-to-end test for the F# T6 lowering — first-argument indexing (native
% string match), lowering type T6 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md. F# is the third atom-keyed
% target to get the gated T6 back-end (after Rust and C++); it represents atoms
% as plain strings (`Atom of string`), so a many-branch F# `match` on the atom's
% string compiles to an efficient hash/jump dispatch — O(1) where the T5
% if/elif cascade is O(n).
%
% Gated like Rust/C++: T6 fires only when every clause discriminates on a
% distinct ATOM and there are at least t6_min_clauses of them (default 8). Below
% the threshold the F# compiler would just flatten the match back to a cascade,
% so the few-clause predicate must STAY T5.
%
% Predicates:
%   * shade/1 (10 atom facts) — must lower as T6; dispatch correctly including a
%     non-first clause reached through the native match.
%   * grade/2 (10 atom RULE clauses, each remainder runs an is/2 builtin) — must
%     lower as T6 and run the clause body under the matched arm.
%   * few/1 (3 atom facts) — below the gate, must STAY T5 (the cascade).
%
% Skipped automatically when `dotnet` is not on PATH.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').
:- use_module('../src/unifyweaver/targets/wam_fsharp_lowered_emitter').

:- dynamic user:shade/1, user:grade/2, user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10).

user:grade(g01, R) :- R is 1 + 0.
user:grade(g02, R) :- R is 1 + 1.
user:grade(g03, R) :- R is 1 + 2.
user:grade(g04, R) :- R is 1 + 3.
user:grade(g05, R) :- R is 1 + 4.
user:grade(g06, R) :- R is 1 + 5.
user:grade(g07, R) :- R is 1 + 6.
user:grade(g08, R) :- R is 1 + 7.
user:grade(g09, R) :- R is 1 + 8.
user:grade(g10, R) :- R is 1 + 9.

user:few(a). user:few(b). user:few(c).

dotnet_available :-
    catch(( process_create(path(dotnet), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_fsharp_lowered_t6, [condition(dotnet_available)]).

% Codegen gate: the many-clause atom predicates emit the native string match
% (T6); the few-clause one stays the if/elif cascade (T5). The threshold is
% configurable.
test(gate_picks_t6_for_many_t5_for_few) :-
    wam_target:compile_predicate_to_wam(shade/1, [], Ws),
    lower_predicate_to_fsharp(shade/1, Ws, [], lowered(_, _, ShadeCode)),
    assertion(sub_string(ShadeCode, _, _, _, "T6 first-argument indexing")),
    assertion(sub_string(ShadeCode, _, _, _, "match t6s with")),
    wam_target:compile_predicate_to_wam(grade/2, [], Wg),
    lower_predicate_to_fsharp(grade/2, Wg, [], lowered(_, _, GradeCode)),
    assertion(sub_string(GradeCode, _, _, _, "T6 first-argument indexing")),
    wam_target:compile_predicate_to_wam(few/1, [], Wf),
    lower_predicate_to_fsharp(few/1, Wf, [], lowered(_, _, FewCode)),
    assertion(sub_string(FewCode, _, _, _, "T5 first-argument dispatch")),
    assertion(\+ sub_string(FewCode, _, _, _, "T6 first-argument indexing")),
    % threshold override: few/1 lowers as T6 when the gate is lowered to 3.
    lower_predicate_to_fsharp(few/1, Wf, [t6_min_clauses(3)], lowered(_, _, FewT6)),
    assertion(sub_string(FewT6, _, _, _, "T6 first-argument indexing")).

% Build + run: the T6 native-match dispatch returns the Prolog-correct result
% for clause hits (including non-first clauses), no-match, and a remainder
% (guard-body) mismatch.
test(t6_exec) :-
    Dir = 'output/test_wam_fsharp_t6_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_fsharp_project(
        [user:shade/1, user:grade/2, user:few/1],
        [module_name('t6proj'), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/Lowered.fs'], LoweredPath),
    read_file_to_string(LoweredPath, LSrc, []),
    assertion(sub_string(LSrc, _, _, _, "T6 first-argument indexing")),
    atomic_list_concat([Dir, '/Program.fs'], ProgPath),
    fsharp_t6_source(Src),
    setup_call_cleanup(open(ProgPath, write, S), write(S, Src), close(S)),
    format(atom(Cmd),
        'cd ~w && DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 dotnet run --project t6proj.fsproj -c Release 2>&1',
        [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 11 PASS")
    ->  true
    ;   format(user_error, "~n[fsharp t6 test output]~n~w~n", [OutStr]),
        throw(fsharp_t6_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_fsharp_lowered_t6).

% Harness: bound first arg, call each lowered function, check Some/None. Cases
% cover clause-1, several non-first clauses (the native-match payoff), no-match,
% and a grade remainder mismatch. few/1 (T5) is exercised too as a control.
fsharp_t6_source(
"module Program

open WamTypes
open WamRuntime
open Lowered

let testEmptyState =
    { WsPC = 0; WsRegs = Array.create MaxRegs (Unbound -1); WsStack = []
      WsHeap = []; WsHeapLen = 0; WsTrail = []; WsTrailLen = 0
      WsCP = 0; WsCPs = []; WsCPsLen = 0; WsBindings = Map.empty
      WsCutBar = 0; WsVarCounter = 0; WsBuilder = None; WsBuilderStack = []
      WsAggAccum = []; WsB0Stack = []; WsCatchers = [] }

let testCtx : WamContext =
    { WcCode = [||]; WcLabels = Map.empty; WcForeignFacts = Map.empty
      WcFfiFacts = Map.empty; WcFfiWeightedFacts = Map.empty
      WcAtomIntern = Map.empty; WcAtomDeintern = Map.empty
      WcForeignConfig = Map.empty; WcLoweredPredicates = loweredPredicates
      WcLookupSources = Map.empty; WcCancellationToken = None }

let runP (f: WamContext -> WamState -> WamState option) (regs: (int * Value) list) : bool =
    let arr = Array.create MaxRegs (Unbound -1)
    regs |> List.iter (fun (idx, v) -> arr.[idx] <- v)
    let s0 = { testEmptyState with WsRegs = arr }
    match f testCtx s0 with Some _ -> true | None -> false

let i (n: int) : Value = Integer n
let a (s: string) : Value = Atom s

[<EntryPoint>]
let main _argv =
    let cases : (string * bool * bool) list =
        [ (\"shade(s01)\",   runP lowered_shade_1 [(1, a \"s01\")], true)
          (\"shade(s05)\",   runP lowered_shade_1 [(1, a \"s05\")], true)
          (\"shade(s10)\",   runP lowered_shade_1 [(1, a \"s10\")], true)
          (\"shade(zz)\",    runP lowered_shade_1 [(1, a \"zz\")],  false)
          (\"grade(g01,1)\", runP lowered_grade_2 [(1, a \"g01\"); (2, i 1)],  true)
          (\"grade(g05,5)\", runP lowered_grade_2 [(1, a \"g05\"); (2, i 5)],  true)
          (\"grade(g10,10)\",runP lowered_grade_2 [(1, a \"g10\"); (2, i 10)], true)
          (\"grade(g05,9)\", runP lowered_grade_2 [(1, a \"g05\"); (2, i 9)],  false)
          (\"grade(zz,1)\",  runP lowered_grade_2 [(1, a \"zz\"); (2, i 1)],   false)
          (\"few(b)\",       runP lowered_few_1 [(1, a \"b\")], true)
          (\"few(z)\",       runP lowered_few_1 [(1, a \"z\")], false) ]
    let mutable fails = 0
    for (name, got, want) in cases do
        if got <> want then
            fails <- fails + 1
            printfn \"FAIL %s got %b want %b\" name got want
    if fails = 0 then printfn \"ALL %d PASS\" (List.length cases)
    else printfn \"%d FAILURES\" fails
    0
").
