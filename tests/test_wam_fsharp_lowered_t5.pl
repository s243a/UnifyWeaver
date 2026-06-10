% test_wam_fsharp_lowered_t5.pl
%
% End-to-end execution test for the F# T5 lowering — "multi-clause as a
% first-argument dispatch" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from the
% Scala/Rust/Go/Haskell emitters via the shared wam_clause_chain front-end.
%
% A predicate whose clauses discriminate on a DISTINCT first-argument
% constant now lowers ALL of its clauses to native F#, selected by a
% deref-and-match cascade, instead of lowering only clause 1 and reaching
% clauses 2+ through the interpreter on backtrack. When the first argument is
% bound this is deterministic dispatch with no interpreter hop; when it is
% unbound it defers to the interpreter via the same choice-point / backtrack /
% run fallback the ordinary multi-clause path uses.
%
% Pins (the cases preload a BOUND first arg, exercising every clause incl. the
% non-first ones — the T5 payoff):
%   * color/1 — fact chain, atom discriminators;
%   * sz/2    — fact chain with a second head match in each remainder;
%   * op/2    — RULE chain (each remainder runs an is/2 builtin).
%
% Skipped automatically when `dotnet` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').

:- dynamic user:color/1.
:- dynamic user:sz/2.
:- dynamic user:op/2.

user:color(red).
user:color(green).
user:color(blue).

user:sz(small, 1).
user:sz(medium, 2).
user:sz(large, 3).

user:op(add, R) :- R is 1 + 1.
user:op(mul, R) :- R is 2 * 3.
user:op(neg, R) :- R is 0 - 1.

dotnet_available :-
    catch(( process_create(path(dotnet), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_fsharp_lowered_t5, [condition(dotnet_available)]).

test(t5_exec_parity) :-
    Dir = 'output/test_wam_fsharp_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM F# project with the lowered emitter enabled.
    write_wam_fsharp_project(
        [user:color/1, user:sz/2, user:op/2],
        [module_name('t5proj'), emit_mode(functions)], Dir),
    % Sanity: the generated lowered code must be the T5 dispatch.
    atomic_list_concat([Dir, '/Lowered.fs'], LoweredPath),
    ( exists_file(LoweredPath) -> read_file_to_string(LoweredPath, LSrc, []) ; LSrc = "" ),
    assertion(sub_string(LSrc, _, _, _, "t5fallback")),
    % 2. Replace the generated bench Program.fs with a harness calling the
    %    lowered functions directly (bound first arg).
    atomic_list_concat([Dir, '/Program.fs'], ProgPath),
    fsharp_t5_source(Src),
    setup_call_cleanup(open(ProgPath, write, S), write(S, Src), close(S)),
    % 3. Compile + run.
    format(atom(Cmd),
        'cd ~w && DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 dotnet run --project t5proj.fsproj -c Release 2>&1',
        [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 14 PASS")
    ->  true
    ;   format(user_error, "~n[fsharp t5 test output]~n~w~n", [OutStr]),
        throw(fsharp_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_fsharp_lowered_t5).

% Builds a minimal WamContext/WamState, sets the A-registers with a BOUND
% first arg, calls each lowered function, and checks Some/None. F# atoms are
% plain strings (Atom of string), so no atom-id coordination is needed. The
% cases exercise every clause including the non-first ones (green/blue,
% medium/large, mul/neg) — the T5 payoff — plus the no-match cases.
fsharp_t5_source(
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
        [ (\"color(red)\", runP lowered_color_1 [(1, a \"red\")], true)
          (\"color(green)\", runP lowered_color_1 [(1, a \"green\")], true)
          (\"color(blue)\", runP lowered_color_1 [(1, a \"blue\")], true)
          (\"color(yellow)\", runP lowered_color_1 [(1, a \"yellow\")], false)
          (\"sz(small,1)\", runP lowered_sz_2 [(1, a \"small\"); (2, i 1)], true)
          (\"sz(medium,2)\", runP lowered_sz_2 [(1, a \"medium\"); (2, i 2)], true)
          (\"sz(large,3)\", runP lowered_sz_2 [(1, a \"large\"); (2, i 3)], true)
          (\"sz(small,2)\", runP lowered_sz_2 [(1, a \"small\"); (2, i 2)], false)
          (\"sz(big,1)\", runP lowered_sz_2 [(1, a \"big\"); (2, i 1)], false)
          (\"op(add,2)\", runP lowered_op_2 [(1, a \"add\"); (2, i 2)], true)
          (\"op(mul,6)\", runP lowered_op_2 [(1, a \"mul\"); (2, i 6)], true)
          (\"op(neg,-1)\", runP lowered_op_2 [(1, a \"neg\"); (2, i -1)], true)
          (\"op(add,3)\", runP lowered_op_2 [(1, a \"add\"); (2, i 3)], false)
          (\"op(div,1)\", runP lowered_op_2 [(1, a \"div\"); (2, i 1)], false) ]
    let fails = cases |> List.filter (fun (_, got, want) -> got <> want)
    fails |> List.iter (fun (n, got, want) -> printfn \"FAIL %s: got %b want %b\" n got want)
    if List.isEmpty fails then printfn \"ALL %d PASS\" (List.length cases); 0
    else printfn \"%d FAILURES\" (List.length fails); 1
").
