% test_wam_fsharp_lowered_ite_exec.pl
%
% End-to-end execution test for F# if-then-else / negation / once lowering
% (emit_mode(functions)).
%
% Generates a WAM F# project with the lowered emitter enabled, compiles it
% with the dotnet/F# toolchain, and runs a harness that calls each lowered
% function and asserts the (Some/None) outcome. Counterpart to the Go, Rust,
% C++ and Haskell exec tests. Pins:
%
%   * sequential ITEs   — fseqite(10,pos,small) must fail;
%   * nested ITEs       — fnestite/2 (inner block in the then-arm);
%   * negation (\+)      — fneg/1 (commit is the !/0 builtin);
%   * simple ITEs        — fite/2.
%
% F# previously did NOT lower any predicate with an if-then-else in clause 1
% (the gate rejected the internal try_me_else); the shared-structurer
% conversion enabled it. Skipped when `dotnet` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').

:- dynamic user:fite/2.
:- dynamic user:fneg/1.
:- dynamic user:fseqite/3.
:- dynamic user:fnestite/2.

user:fite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:fneg(X)          :- \+ X > 0.
user:fseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:fnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

dotnet_available :-
    catch(( process_create(path(dotnet), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_fsharp_lowered_ite_exec, [condition(dotnet_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_fsharp_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM F# project with the lowered emitter enabled.
    write_wam_fsharp_project(
        [user:fite/2, user:fneg/1, user:fseqite/3, user:fnestite/2],
        [module_name('iteproj'), emit_mode(functions)], Dir),
    % 2. Replace the generated bench Program.fs (the last-compiled, entry
    %    module) with a harness that calls the lowered functions directly.
    atomic_list_concat([Dir, '/Program.fs'], ProgPath),
    fsharp_test_source(Src),
    setup_call_cleanup(open(ProgPath, write, S), write(S, Src), close(S)),
    % 3. Compile + run.
    format(atom(Cmd),
        'cd ~w && DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 dotnet run --project iteproj.fsproj -c Release 2>&1',
        [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 15 PASS")
    ->  true
    ;   format(user_error, "~n[fsharp ite test output]~n~w~n", [OutStr]),
        throw(fsharp_test_failed(Status))
    ).

:- end_tests(wam_fsharp_lowered_ite_exec).

% Builds a minimal WamContext/WamState, sets the A-registers, calls each
% lowered function, and checks Some (success) / None (failure). F# atoms are
% plain strings (Atom of string), so no atom-id coordination is needed.
fsharp_test_source(
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
        [ (\"fite(5,pos)\", runP lowered_fite_2 [(1, i 5); (2, a \"pos\")], true)
          (\"fite(5,nonpos)\", runP lowered_fite_2 [(1, i 5); (2, a \"nonpos\")], false)
          (\"fite(-1,nonpos)\", runP lowered_fite_2 [(1, i -1); (2, a \"nonpos\")], true)
          (\"fite(-1,pos)\", runP lowered_fite_2 [(1, i -1); (2, a \"pos\")], false)
          (\"fneg(5)\", runP lowered_fneg_1 [(1, i 5)], false)
          (\"fneg(-1)\", runP lowered_fneg_1 [(1, i -1)], true)
          (\"fneg(0)\", runP lowered_fneg_1 [(1, i 0)], true)
          (\"fseqite(10,pos,big)\", runP lowered_fseqite_3 [(1, i 10); (2, a \"pos\"); (3, a \"big\")], true)
          (\"fseqite(10,pos,small)\", runP lowered_fseqite_3 [(1, i 10); (2, a \"pos\"); (3, a \"small\")], false)
          (\"fseqite(3,pos,small)\", runP lowered_fseqite_3 [(1, i 3); (2, a \"pos\"); (3, a \"small\")], true)
          (\"fseqite(-1,nonpos,small)\", runP lowered_fseqite_3 [(1, i -1); (2, a \"nonpos\"); (3, a \"small\")], true)
          (\"fnestite(20,big)\", runP lowered_fnestite_2 [(1, i 20); (2, a \"big\")], true)
          (\"fnestite(5,small)\", runP lowered_fnestite_2 [(1, i 5); (2, a \"small\")], true)
          (\"fnestite(-1,neg)\", runP lowered_fnestite_2 [(1, i -1); (2, a \"neg\")], true)
          (\"fnestite(20,small)\", runP lowered_fnestite_2 [(1, i 20); (2, a \"small\")], false) ]
    let fails = cases |> List.filter (fun (_, got, want) -> got <> want)
    fails |> List.iter (fun (n, got, want) -> printfn \"FAIL %s: got %b want %b\" n got want)
    if List.isEmpty fails then printfn \"ALL %d PASS\" (List.length cases); 0
    else printfn \"%d FAILURES\" (List.length fails); 1
").
