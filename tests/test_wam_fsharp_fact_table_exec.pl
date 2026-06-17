% test_wam_fsharp_fact_table_exec.pl
%
% T9 fact-table inline for the F# target. An all-ground-facts predicate whose row
% count is in [t9_min_rows, t9_max_rows] lowers to a static row table + first-arg
% index + a backtracking enumerator (factTableAttempt + FactTableRetry choice
% point), registered as a lowered predicate so call/execute reach it via the same
% loweredPredicates map T4/T5/T6 use. Verifies the query-mode matrix end-to-end; a
% rule calling the fact predicate (firstedge/1) is a compile check on the
% call-site wiring.
%
% Skipped when dotnet is not on PATH.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').
:- use_module('../src/unifyweaver/targets/wam_fsharp_lowered_emitter').

:- dynamic user:edge/2, user:firstedge/1.
user:edge(a, 1). user:edge(a, 2). user:edge(b, 3).
user:edge(a, 4). user:edge(c, 5). user:edge(b, 6).
% call-site: a rule that calls the fact-table predicate
user:firstedge(X) :- user:edge(a, X).

dotnet_available :-
    catch(( process_create(path(dotnet), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_fsharp_fact_table_exec, [condition(dotnet_available)]).

% Codegen: edge/2 lowers to the T9 fact-table enumerator (default in-range).
test(codegen_emits_fact_table) :-
    wam_target:compile_predicate_to_wam(edge/2, [], W),
    lower_predicate_to_fsharp(user:edge/2, W, [t9_min_rows(4)], lowered(_, _, Code)),
    assertion(sub_string(Code, _, _, _, "factTableAttempt")),
    assertion(sub_string(Code, _, _, _, "lowered_edge_2_rows")),
    % opt-out forces the ordinary lowering (no fact table)
    lower_predicate_to_fsharp(user:edge/2, W,
        [t9_min_rows(4), fact_table_inline(false)], lowered(_, _, Code2)),
    assertion(\+ sub_string(Code2, _, _, _, "factTableAttempt")).

% Build + run: query-mode matrix + call-site dispatch.
test(query_mode_matrix_exec) :-
    Dir = 'output/test_wam_fsharp_t9_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_fsharp_project(
        [user:edge/2, user:firstedge/1],
        [module_name('t9proj'), emit_mode(functions), t9_min_rows(4)], Dir),
    atomic_list_concat([Dir, '/Lowered.fs'], LoweredPath),
    read_file_to_string(LoweredPath, LSrc, []),
    assertion(sub_string(LSrc, _, _, _, "factTableAttempt")),
    atomic_list_concat([Dir, '/Program.fs'], ProgPath),
    fsharp_t9_source(Src),
    setup_call_cleanup(open(ProgPath, write, S), write(S, Src), close(S)),
    format(atom(Cmd),
        'cd ~w && DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 dotnet run --project t9proj.fsproj -c Release 2>&1',
        [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 5 PASS")
    ->  true
    ;   format(user_error, "~n[fsharp t9 test output]~n~w~n", [OutStr]),
        throw(fsharp_t9_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_fsharp_fact_table_exec).

% Harness: seed registers, call the lowered predicate, then drive backtracking
% via WamRuntime.backtrack collecting each solution's bound vars. Exercises the
% FactTableRetry choice point.
fsharp_t9_source(
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

let i (n: int) : Value = Integer n
let a (s: string) : Value = Atom s

// Enumerate all solutions of `f` seeded with `regs`, reading `vars` per solution.
let sols (f: WamContext -> WamState -> WamState option) (regs: (int * Value) list) (vars: int list) : Value list list =
    let arr = Array.create MaxRegs (Unbound -1)
    regs |> List.iter (fun (idx, v) -> arr.[idx] <- v)
    let s0 = { testEmptyState with WsRegs = arr }
    let rec loop sOpt acc =
        match sOpt with
        | None -> List.rev acc
        | Some s ->
            let row = vars |> List.map (fun r -> match getReg r s with Some v -> derefVar s.WsBindings v | None -> Unbound -1)
            loop (backtrack s) (row :: acc)
    loop (f testCtx s0) []

[<EntryPoint>]
let main _argv =
    // (+,-) edge(a, X) -> [1;2;4] in source order
    let m1 = sols lowered_edge_2 [(1, a \"a\"); (2, Unbound 100)] [2]
    // (-,+) edge(K, 3) -> [b]
    let m2 = sols lowered_edge_2 [(1, Unbound 101); (2, i 3)] [1]
    // (-,-) edge(K, X) -> all 6 rows
    let m3 = sols lowered_edge_2 [(1, Unbound 102); (2, Unbound 103)] [1; 2]
    // (+,+) membership
    let m4yes = sols lowered_edge_2 [(1, a \"a\"); (2, i 2)] []
    let m4no  = sols lowered_edge_2 [(1, a \"a\"); (2, i 3)] []
    let cases : (string * bool) list =
        [ (\"(+,-)\",   m1 = [[i 1];[i 2];[i 4]])
          (\"(-,+)\",   m2 = [[a \"b\"]])
          (\"(-,-)\",   m3 = [[a \"a\"; i 1];[a \"a\"; i 2];[a \"b\"; i 3];[a \"a\"; i 4];[a \"c\"; i 5];[a \"b\"; i 6]])
          (\"(+,+)yes\", List.length m4yes = 1)
          (\"(+,+)no\",  List.length m4no = 0) ]
    let mutable fails = 0
    for (name, ok) in cases do
        if not ok then
            fails <- fails + 1
            printfn \"FAIL %s\" name
    if fails = 0 then printfn \"ALL %d PASS\" (List.length cases)
    else printfn \"%d FAILURES\" fails
    0
").
