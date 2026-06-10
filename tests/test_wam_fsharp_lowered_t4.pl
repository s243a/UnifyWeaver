% test_wam_fsharp_lowered_t4.pl
%
% End-to-end execution test for the F# T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from
% Scala/Rust/Go/C++/Haskell.
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers to ALL clauses inline: each clause becomes a
% `WamState -> WamState option` and the function tries them in order on the
% SAME input state (`t4clause_1 s_init |> Option.orElseWith (fun () ->
% t4clause_2 s_init) |> ...`), taking the first Some. F#'s immutability gives
% a free per-clause restore, so — unlike the imperative targets — no
% snapshot/restore and no choice point are needed; the interpreter is never
% entered for the predicate.
%
% Pins (BOUND first arg; the payoff is the non-first clauses running natively):
%   * grade/2 — fact chain with a REPEATED first arg (alice in clauses 1 & 3);
%   * rel/2   — RULE chain with a VARIABLE first arg (=/2 body).
%
% Skipped automatically when `dotnet` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').

:- dynamic user:grade/2.
:- dynamic user:rel/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

dotnet_available :-
    catch(( process_create(path(dotnet), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_fsharp_lowered_t4, [condition(dotnet_available)]).

test(t4_exec_parity) :-
    Dir = 'output/test_wam_fsharp_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_fsharp_project(
        [user:grade/2, user:rel/2],
        [module_name('t4proj'), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/Lowered.fs'], LoweredPath),
    ( exists_file(LoweredPath) -> read_file_to_string(LoweredPath, LSrc, []) ; LSrc = "" ),
    assertion(sub_string(LSrc, _, _, _, "T4 all-clauses inline")),
    atomic_list_concat([Dir, '/Program.fs'], ProgPath),
    fsharp_t4_source(Src),
    setup_call_cleanup(open(ProgPath, write, S), write(S, Src), close(S)),
    format(atom(Cmd),
        'cd ~w && DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 dotnet run --project t4proj.fsproj -c Release 2>&1',
        [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 10 PASS")
    ->  true
    ;   format(user_error, "~n[fsharp t4 test output]~n~w~n", [OutStr]),
        throw(fsharp_t4_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_fsharp_lowered_t4).

% Builds a minimal WamContext/WamState, sets the A-registers with a BOUND
% first arg, calls each lowered function, and checks Some/None. The cases
% exercise the non-first clauses (grade clauses 2 & 3, rel clause 2) — the T4
% payoff — plus the no-match cases.
fsharp_t4_source(
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

let a (s: string) : Value = Atom s

[<EntryPoint>]
let main _argv =
    let cases : (string * bool * bool) list =
        [ (\"grade(alice,a)\", runP lowered_grade_2 [(1, a \"alice\"); (2, a \"a\")], true)
          (\"grade(bob,b)\",   runP lowered_grade_2 [(1, a \"bob\"); (2, a \"b\")], true)
          (\"grade(alice,c)\", runP lowered_grade_2 [(1, a \"alice\"); (2, a \"c\")], true)
          (\"grade(alice,b)\", runP lowered_grade_2 [(1, a \"alice\"); (2, a \"b\")], false)
          (\"grade(carol,a)\", runP lowered_grade_2 [(1, a \"carol\"); (2, a \"a\")], false)
          (\"grade(bob,c)\",   runP lowered_grade_2 [(1, a \"bob\"); (2, a \"c\")], false)
          (\"rel(p,one)\", runP lowered_rel_2 [(1, a \"p\"); (2, a \"one\")], true)
          (\"rel(q,two)\", runP lowered_rel_2 [(1, a \"q\"); (2, a \"two\")], true)
          (\"rel(p,two)\", runP lowered_rel_2 [(1, a \"p\"); (2, a \"two\")], false)
          (\"rel(q,one)\", runP lowered_rel_2 [(1, a \"q\"); (2, a \"one\")], false) ]
    let fails = cases |> List.filter (fun (_, got, want) -> got <> want)
    fails |> List.iter (fun (n, got, want) -> printfn \"FAIL %s: got %b want %b\" n got want)
    if List.isEmpty fails then printfn \"ALL %d PASS\" (List.length cases); 0
    else printfn \"%d FAILURES\" (List.length fails); 1
").
