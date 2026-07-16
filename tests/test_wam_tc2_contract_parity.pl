:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_tc2_contract_parity.pl — fleet-wide transitive_closure2
% contract (strict R+) parity suite.
%
% Contract: docs/design/WAM_TRANSITIVE_CLOSURE2_CONTRACT.md
% Oracle:   tests/fixtures/tc2_contract_oracle.pl
%
% Layers:
%   1. Oracle self-tests + hardcoded fixture expectations
%   2. Structural R+ pattern checks (handlers / templates)
%   3. Native dispatch proof (emit contains foreign/native TC2)
%   4. Executable smoke where toolchains exist (dotnet / gcc / cargo)
%   5. Acyclic native vs oracle set-normalized compare (F#)
%   6. Pin: cyclic generic WAM recursion is not the oracle
%
% Usage (from repo root):
%   swipl -q -g run_tests -t halt tests/test_wam_tc2_contract_parity.pl
%
% F# e2e also covered by tests/core/test_wam_fsharp_kernel_gate_tc.pl.

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(process)).
:- use_module(library(debug), [assertion/1]).

:- use_module('fixtures/tc2_contract_oracle', [
    tc2_oracle_reachable/3,
    tc2_oracle_reaches/3,
    tc2_fixture/3,
    tc2_fixture_expected/3
]).

:- use_module('../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module('../src/unifyweaver/targets/wam_c_target',
              [write_wam_c_project/3]).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3,
               compile_wam_runtime_to_rust/2]).
:- use_module('../src/unifyweaver/core/recursive_kernel_detection',
              [detect_recursive_kernel/4]).

:- dynamic user:tc_edge/2.
:- dynamic user:tc/2.
:- dynamic user:tc_parent/2.
:- dynamic user:tc_ancestor/2.

dotnet_available :-
    catch(
        ( process_create(path(dotnet), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, _) ),
        _, fail).

gcc_available :-
    catch(
        ( process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, _) ),
        _, fail).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, _) ),
        _, fail).

tmp_dir(Tag, Dir) :-
    get_time(T),
    Stamp is round(T * 1000000),
    format(atom(Dir), '/tmp/uw_tc2_~w_~w', [Tag, Stamp]),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir).

read_file_string(Path, String) :-
    read_file_to_string(Path, String, []).

assert_tc_cycle_program :-
    retractall(user:tc_edge(_, _)),
    retractall(user:tc(_, _)),
    assertz(user:tc_edge(a, b)),
    assertz(user:tc_edge(b, c)),
    assertz(user:tc_edge(c, d)),
    assertz(user:tc_edge(c, a)),
    assertz((user:tc(X, Y) :- tc_edge(X, Y))),
    assertz((user:tc(X, Y) :- tc_edge(X, Z), tc(Z, Y))).

assert_tc_chain_program :-
    retractall(user:tc_edge(_, _)),
    retractall(user:tc(_, _)),
    assertz(user:tc_edge(a, b)),
    assertz(user:tc_edge(b, c)),
    assertz(user:tc_edge(c, d)),
    assertz((user:tc(X, Y) :- tc_edge(X, Y))),
    assertz((user:tc(X, Y) :- tc_edge(X, Z), tc(Z, Y))).

assert_c_tc_program :-
    retractall(user:tc_parent(_, _)),
    retractall(user:tc_ancestor(_, _)),
    assertz(user:tc_parent(a, b)),
    assertz(user:tc_parent(b, c)),
    assertz(user:tc_parent(c, d)),
    assertz(user:tc_parent(c, a)),
    assertz((user:tc_ancestor(X, Y) :- tc_parent(X, Y))),
    assertz((user:tc_ancestor(X, Y) :- tc_parent(X, Z), tc_ancestor(Z, Y))).

% ============================================================
% 1. Oracle
% ============================================================

:- begin_tests(tc2_oracle).

test(hardcoded_fixture_expectations) :-
    tc2_fixture_expected(chain, a, [b, c, d]),
    tc2_fixture_expected(chain, c, [d]),
    tc2_fixture_expected(chain, d, []),
    tc2_fixture_expected(chain, z, []),
    tc2_fixture_expected(self_loop, a, [a, b]),
    tc2_fixture_expected(self_loop, b, []),
    tc2_fixture_expected(two_cycle, a, [a, b]),
    tc2_fixture_expected(two_cycle, b, [a, b]),
    tc2_fixture_expected(long_cycle_exit, a, [a, b, c, d]),
    tc2_fixture_expected(long_cycle_exit, c, [a, b, c, d]),
    tc2_fixture_expected(long_cycle_exit, d, []),
    tc2_fixture_expected(dup_edges, a, [b, c]),
    tc2_fixture_expected(sink_disconnected, a, [b]),
    tc2_fixture_expected(sink_disconnected, z, []).

test(oracle_matches_all_fixture_sources) :-
    forall(
        ( tc2_fixture(Name, Edges, Sources),
          member(Src, Sources)
        ),
        ( tc2_oracle_reachable(Edges, Src, Sorted),
          tc2_fixture_expected(Name, Src, Sorted)
        )
    ).

test(bound_mode_follows_set_membership) :-
    tc2_oracle_reaches([a-b, b-a], a, a),
    tc2_oracle_reaches([a-b, b-a], a, b),
    \+ tc2_oracle_reaches([a-b, b-c], a, a),
    \+ tc2_oracle_reaches([a-b], a, z).

test(acyclic_source_not_reflexive) :-
    tc2_oracle_reachable([a-b, b-c], a, Sorted),
    \+ memberchk(a, Sorted).

:- end_tests(tc2_oracle).

% ============================================================
% 2. Structural R+ pattern checks
% ============================================================

:- begin_tests(tc2_structural_rplus).

test(rust_handler_does_not_seed_seen_with_source) :-
    compile_wam_runtime_to_rust([], Code),
    atom_string(Code, S),
    assertion(sub_string(S, _, _, _, "collect_native_transitive_closure_nodes")),
    assertion(sub_string(S, _, _, _, "let mut seen: HashSet<String> = HashSet::new();")),
    assertion(sub_string(S, _, _, _, "let mut stack: Vec<String> = vec![start.to_string()];")),
    assertion(sub_string(S, _, _, _, "if seen.insert(next.clone())")),
    assertion(\+ sub_string(S, _, _, _, "seen.insert(start")).

test(fsharp_mustache_rplus_visited) :-
    read_file_string(
        'templates/targets/fsharp_wam/kernel_transitive_closure.fs.mustache', S),
    assertion(sub_string(S, _, _, _, "strict R+")),
    assertion(sub_string(S, _, _, _, "let visited = System.Collections.Generic.HashSet<int>()")),
    assertion(sub_string(S, _, _, _, "let mutable frontier = [source]")),
    assertion(sub_string(S, _, _, _, "if visited.Add(n) then")),
    assertion(\+ sub_string(S, _, _, _, "visited.Add(source)")).

test(haskell_mustache_rplus_visited) :-
    read_file_string(
        'templates/targets/haskell_wam/kernel_transitive_closure.hs.mustache', S),
    assertion(sub_string(S, _, _, _, "strict R+")),
    assertion(sub_string(S, _, _, _, "go [source] IS.empty []")),
    assertion(sub_string(S, _, _, _, "newTargets = filter")),
    assertion(\+ sub_string(S, _, _, _, "IS.insert source")).

test(c_handler_does_not_seed_visited_with_start) :-
    read_file_string('src/unifyweaver/targets/wam_c_target.pl', S),
    assertion(sub_string(S, _, _, _, "wam_transitive_closure_handler")),
    assertion(sub_string(S, _, _, _, "Strict R+")),
    assertion(sub_string(S, _, _, _, "do NOT seed visited with start")),
    assertion(sub_string(S, _, _, _, "int visited_len = 0;")),
    assertion(\+ sub_string(S, _, _, _, "visited[0] = start")).

test(r_runtime_does_not_seed_visited_with_source) :-
    read_file_string('templates/targets/r_wam/runtime.R.mustache', S),
    assertion(sub_string(S, _, _, _, "WamRuntime$transitive_closure2")),
    assertion(sub_string(S, _, _, _, "Strict R+")),
    assertion(sub_string(S, _, _, _, "do not seed with Source")),
    assertion(sub_string(S, _, _, _, "visited <- new.env(parent = emptyenv())")),
    assertion(sub_string(S, _, _, _, "queue   <- list(src_v)")).

test(scala_handler_rplus_visited) :-
    read_file_string('src/unifyweaver/targets/wam_scala_target.pl', S),
    assertion(sub_string(S, _, _, _, "recursive_kernel(transitive_closure2")),
    assertion(sub_string(S, _, _, _,
        "val visited = scala.collection.mutable.LinkedHashSet[WamTerm]()")),
    assertion(sub_string(S, _, _, _,
        "val queue = scala.collection.mutable.Queue[WamTerm](source)")),
    assertion(sub_string(S, _, _, _, "visited += nb")),
    assertion(\+ sub_string(S, _, _, _,
        "LinkedHashSet[WamTerm](source)")).

test(llvm_stream_and_bound_self_are_rplus) :-
    read_file_string('templates/targets/llvm_wam/state.ll.mustache', S),
    assertion(sub_string(S, _, _, _, "wam_tc2_stream_run")),
    assertion(sub_string(S, _, _, _, "do NOT mark start visited")),
    assertion(sub_string(S, _, _, _, "wam_tc2_rplus_reaches")),
    assertion(sub_string(S, _, _, _, "tc_rplus_self")),
    assertion(sub_string(S, _, _, _,
        "Strict R+: Source==Target needs a self-loop")).

test(go_handler_seeds_queue_from_neighbors) :-
    read_file_string('src/unifyweaver/targets/wam_go_target.pl', S),
    assertion(sub_string(S, _, _, _, "collectNativeTransitiveClosureResults")),
    assertion(sub_string(S, _, _, _,
        "queue := append([]string(nil), adjacency[source]...)")).

:- end_tests(tc2_structural_rplus).

% ============================================================
% 3. Native dispatch proof
% ============================================================

:- begin_tests(tc2_native_dispatch).

test(detector_fires_transitive_closure2) :-
    assert_tc_cycle_program,
    findall(tc(X, Y)-Body, clause(user:tc(X, Y), Body), Clauses),
    detect_recursive_kernel(tc, 2, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_closure2, _, _)).

test(fsharp_native_dispatch_emitted) :-
    assert_tc_cycle_program,
    tmp_dir(fs_dispatch, Dir),
    write_wam_fsharp_project(
        [user:tc/2, user:tc_edge/2],
        [module_name('uw_tc2_dispatch')],
        Dir),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_string(RT, RTS),
    assertion(sub_string(RTS, _, _, _, "nativeKernel_transitive_closure")),
    assertion(sub_string(RTS, _, _, _, "| \"tc/2\" ->")),
    assertion(sub_string(RTS, _, _, _, "FFIStreamRetry")),
    !.

test(rust_foreign_kind_registered_for_tc2) :-
    assert_tc_cycle_program,
    tmp_dir(rs_dispatch, Dir),
    write_wam_rust_project(
        [user:tc/2, user:tc_edge/2],
        [module_name('uw_tc2_rs_dispatch'), foreign_lowering(true)],
        Dir),
    directory_file_path(Dir, 'src/lib.rs', Lib),
    read_file_string(Lib, S),
    assertion(sub_string(S, _, _, _, "transitive_closure2")),
    assertion(sub_string(S, _, _, _, "register_foreign_native_kind(\"tc/2\"")),
    directory_file_path(Dir, 'src/state.rs', State),
    read_file_string(State, SS),
    assertion(sub_string(SS, _, _, _, "collect_native_transitive_closure_nodes")),
    !.

test(c_registers_transitive_closure_handler) :-
    assert_c_tc_program,
    tmp_dir(c_dispatch, Dir),
    write_wam_c_project([user:tc_ancestor/2], [], Dir),
    directory_file_path(Dir, 'lib.c', Lib),
    read_file_string(Lib, LibS),
    assertion(sub_string(LibS, _, _, _,
        "wam_register_transitive_closure_kernel(state, \"tc_ancestor/2\")")),
    directory_file_path(Dir, 'wam_runtime.c', RT),
    read_file_string(RT, S),
    assertion(sub_string(S, _, _, _, "wam_transitive_closure_handler")),
    assertion(sub_string(S, _, _, _, "do NOT seed visited with start")),
    !.

:- end_tests(tc2_native_dispatch).

% ============================================================
% 4. Executable smokes
% ============================================================

:- begin_tests(tc2_executable).

test(fsharp_cycle_rplus_e2e, [condition(dotnet_available)]) :-
    assert_tc_cycle_program,
    tmp_dir(fs_e2e, Dir),
    write_wam_fsharp_project(
        [user:tc/2, user:tc_edge/2],
        [module_name('uw_tc2_e2e')],
        Dir),
    directory_file_path(Dir, 'Program.fs', Prog),
    tc2_write_fsharp_rplus_driver(Prog),
    run_dotnet_build(Dir, BuildExit, BuildOut),
    assertion(BuildExit =:= 0),
    ( BuildExit =:= 0 -> true
    ; format(user_error, 'fsharp e2e build:~n~w~n', [BuildOut]), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    assertion(RunExit =:= 0),
    assertion(sub_string(RunOut, _, _, _, "OK cycle_from_a")),
    assertion(sub_string(RunOut, _, _, _, "OK bound_source")),
    assertion(sub_string(RunOut, _, _, _, "OK sink_d")),
    !.

test(c_bound_source_via_cycle, [condition(gcc_available)]) :-
    assert_c_tc_program,
    tmp_dir(c_e2e, Dir),
    write_wam_c_project([user:tc_ancestor/2], [], Dir),
    directory_file_path(Dir, 'wam_runtime.c', RuntimePath),
    directory_file_path(Dir, 'lib.c', LibPath),
    directory_file_path(Dir, 'main.c', MainPath),
    directory_file_path(Dir, 'tc2_c_smoke', ExePath),
    tc2_c_cycle_main(MainCode),
    setup_call_cleanup(
        open(MainPath, write, Out, [encoding(utf8)]),
        write(Out, MainCode),
        close(Out)),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    format(atom(Cmd),
        'gcc -O0 -std=c11 -I~w ~w ~w ~w -o ~w 2>~w/gcc.err',
        [IncludeDir, RuntimePath, LibPath, MainPath, ExePath, Dir]),
    shell(Cmd, GccExit),
    ( GccExit =:= 0 -> true
    ; directory_file_path(Dir, 'gcc.err', Err),
      ( exists_file(Err) -> read_file_string(Err, E),
        format(user_error, 'gcc failed:~n~w~n', [E]) ; true ),
      fail
    ),
    shell(ExePath, RunExit),
    assertion(RunExit =:= 0),
    !.

test(rust_collect_rplus_unit, [condition(cargo_available)]) :-
    assert_tc_cycle_program,
    tmp_dir(rs_e2e, Dir),
    write_wam_rust_project(
        [user:tc/2, user:tc_edge/2],
        [module_name('uw_tc2_rs'), foreign_lowering(true)],
        Dir),
    tc2_append_rust_unit(Dir),
    format(atom(Cmd),
        'cd ~w && cargo test --quiet tc2_rplus_contract -- --nocapture >~w/cargo.out 2>~w/cargo.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'cargo.err', ErrPath),
      directory_file_path(Dir, 'cargo.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'rust tc2 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

:- end_tests(tc2_executable).

% ============================================================
% 5. Acyclic native vs oracle + cyclic WAM note
% ============================================================

:- begin_tests(tc2_no_kernels_acyclic).

test(fsharp_acyclic_native_matches_oracle_sorted,
     [condition(dotnet_available)]) :-
    assert_tc_chain_program,
    tc2_oracle_reachable([a-b, b-c, c-d], a, Oracle),
    tmp_dir(fs_chain, Dir),
    write_wam_fsharp_project(
        [user:tc/2, user:tc_edge/2],
        [module_name('uw_tc2_chain')],
        Dir),
    directory_file_path(Dir, 'Program.fs', Prog),
    tc2_write_fsharp_chain_driver(Prog, Oracle),
    run_dotnet_build(Dir, BuildExit, BuildOut),
    assertion(BuildExit =:= 0),
    ( BuildExit =:= 0 -> true
    ; format(user_error, 'fsharp chain build:~n~w~n', [BuildOut]), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    assertion(RunExit =:= 0),
    assertion(sub_string(RunOut, _, _, _, "OK native_matches_oracle")),
    !.

test(cyclic_generic_wam_is_not_the_oracle) :-
    % On cyclic graphs, no_kernels recursive WAM explores path proofs
    % (may duplicate or not terminate). Finite oracle + native R+
    % handlers are authoritative; do not cross-compare cyclic
    % no_kernels streams against the oracle. Acyclic set-normalized
    % compare is allowed (see fsharp_acyclic_native_matches_oracle_sorted).
    assertion(true).

:- end_tests(tc2_no_kernels_acyclic).

% ============================================================
% Helpers
% ============================================================

run_dotnet_build(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['build', '--nologo', '-v', 'q', '-c', 'Release'],
            [cwd(Dir), stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, exit(Exit)),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

run_dotnet_run(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['run', '--no-build', '-c', 'Release', '--no-launch-profile', '--'],
            [cwd(Dir),
             environment(['DOTNET_NOLOGO'='1', 'DOTNET_ROLL_FORWARD'='Major']),
             stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, exit(Exit)),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

tc2_write_fsharp_rplus_driver(ProgPath) :-
    Driver =
"module Program

open System
open WamTypes
open WamRuntime
open Predicates

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn \"OK %s\" name
    else
        fails <- fails + 1
        printfn \"FAIL %s\" name

let mkAtoms () =
    let pairs = [(\"a\", 1); (\"b\", 2); (\"c\", 3); (\"d\", 4)]
    Map.ofList pairs, Map.ofList (pairs |> List.map (fun (s, i) -> (i, s)))

let mkEdgeFacts (intern: Map<string, int>) =
    let edges = [(\"a\", \"b\"); (\"b\", \"c\"); (\"c\", \"d\"); (\"c\", \"a\")]
    let grouped =
        edges
        |> List.map (fun (f, t) -> Map.find f intern, Map.find t intern)
        |> List.groupBy fst
        |> List.map (fun (k, vs) -> k, vs |> List.map snd)
    Map.ofList [(\"tc_edge\", Map.ofList grouped)]

let mkContext () =
    let foreignPreds = [\"tc/2\"]
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    let intern, deintern = mkAtoms ()
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = mkEdgeFacts intern
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = intern
      WcAtomDeintern      = deintern
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcLookupSources     = Map.empty
      WcCancellationToken = None }

let mkState (regs: Value array) : WamState =
    { WsPC         = 0
      WsRegs       = regs
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
      WsB0Stack    = []
      WsCatchers   = [] }

let collectTargets (ctx: WamContext) (source: string) (boundTarget: string option) : string list =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom source
    match boundTarget with
    | Some t -> regs.[2] <- Atom t
    | None   -> regs.[2] <- Unbound 100
    match callForeign ctx \"tc/2\" (mkState regs) with
    | None -> []
    | Some s1 ->
        let readTarget (s: WamState) =
            match getReg 2 s with
            | Some (Atom a) -> a
            | _ -> \"?\"
        let rec gather (s: WamState) (acc: string list) =
            let tgt = readTarget s
            match backtrack s with
            | Some s2 -> gather s2 (tgt :: acc)
            | None -> List.rev (tgt :: acc)
        gather s1 []

[<EntryPoint>]
let main _argv =
    let ctx = mkContext ()
    let fromA = collectTargets ctx \"a\" None |> List.sort
    assertTrue \"cycle_from_a\" (fromA = [\"a\"; \"b\"; \"c\"; \"d\"])
    let boundSrc = collectTargets ctx \"a\" (Some \"a\")
    assertTrue \"bound_source\" (boundSrc = [\"a\"])
    let fromD = collectTargets ctx \"d\" None
    assertTrue \"sink_d\" (fromD = [])
    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    setup_call_cleanup(
        open(ProgPath, write, Out, [encoding(utf8)]),
        write(Out, Driver),
        close(Out)).

tc2_write_fsharp_chain_driver(ProgPath, Oracle) :-
    maplist(tc2_atom_lit, Oracle, Lits),
    atomic_list_concat(Lits, '; ', LitBody),
    format(atom(ExpectedFs), '[~w]', [LitBody]),
    format(atom(Driver),
"module Program

open System
open WamTypes
open WamRuntime
open Predicates

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn \"OK %s\" name
    else
        fails <- fails + 1
        printfn \"FAIL %s\" name

let mkAtoms () =
    let pairs = [(\"a\", 1); (\"b\", 2); (\"c\", 3); (\"d\", 4)]
    Map.ofList pairs, Map.ofList (pairs |> List.map (fun (s, i) -> (i, s)))

let mkEdgeFacts (intern: Map<string, int>) =
    let edges = [(\"a\", \"b\"); (\"b\", \"c\"); (\"c\", \"d\")]
    let grouped =
        edges
        |> List.map (fun (f, t) -> Map.find f intern, Map.find t intern)
        |> List.groupBy fst
        |> List.map (fun (k, vs) -> k, vs |> List.map snd)
    Map.ofList [(\"tc_edge\", Map.ofList grouped)]

let mkContext () =
    let foreignPreds = [\"tc/2\"]
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    let intern, deintern = mkAtoms ()
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = mkEdgeFacts intern
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = intern
      WcAtomDeintern      = deintern
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcLookupSources     = Map.empty
      WcCancellationToken = None }

let mkState (regs: Value array) : WamState =
    { WsPC         = 0
      WsRegs       = regs
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
      WsB0Stack    = []
      WsCatchers   = [] }

let collectTargets (ctx: WamContext) (source: string) : string list =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom source
    regs.[2] <- Unbound 100
    match callForeign ctx \"tc/2\" (mkState regs) with
    | None -> []
    | Some s1 ->
        let readTarget (s: WamState) =
            match getReg 2 s with
            | Some (Atom a) -> a
            | _ -> \"?\"
        let rec gather (s: WamState) (acc: string list) =
            let tgt = readTarget s
            match backtrack s with
            | Some s2 -> gather s2 (tgt :: acc)
            | None -> List.rev (tgt :: acc)
        gather s1 []

[<EntryPoint>]
let main _argv =
    let ctx = mkContext ()
    let got = collectTargets ctx \"a\" |> List.sort
    let expected = ~w |> List.sort
    assertTrue \"native_matches_oracle\" (got = expected)
    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
        [ExpectedFs]),
    setup_call_cleanup(
        open(ProgPath, write, Out, [encoding(utf8)]),
        write(Out, Driver),
        close(Out)).

tc2_atom_lit(A, Lit) :-
    format(atom(Lit), '"~w"', [A]).

tc2_c_cycle_main(Code) :-
    Code =
'#include "wam_runtime.h"

void setup_tc_ancestor_2(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_ancestor_2(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_transitive_edge(&state, "a", "b");
    wam_register_transitive_edge(&state, "b", "c");
    wam_register_transitive_edge(&state, "c", "d");
    wam_register_transitive_edge(&state, "c", "a");

    /* Bound Source via nonempty cycle must succeed (strict R+). */
    {
        WamValue args[2] = { val_atom("a"), val_atom("a") };
        int rc = wam_run_predicate(&state, "tc_ancestor/2", args, 2);
        if (rc != 0 || state.P != WAM_HALT) {
            wam_free_state(&state);
            return 11;
        }
    }

    wam_free_state(&state);
    wam_state_init(&state);
    setup_tc_ancestor_2(&state);
    setup_detected_wam_c_kernels(&state);
    wam_register_transitive_edge(&state, "a", "b");
    wam_register_transitive_edge(&state, "b", "c");
    wam_register_transitive_edge(&state, "c", "d");
    wam_register_transitive_edge(&state, "c", "a");

    /* Sink d has no outgoing edges — unbound Target must fail. */
    {
        WamValue args[2] = { val_atom("d"), val_unbound("T") };
        int rc = wam_run_predicate(&state, "tc_ancestor/2", args, 2);
        if (rc == 0 && state.P == WAM_HALT) {
            wam_free_state(&state);
            return 12;
        }
    }

    wam_free_state(&state);
    return 0;
}
'.

tc2_append_rust_unit(Dir) :-
    directory_file_path(Dir, 'src/lib.rs', Path),
    read_file_string(Path, Existing),
    Unit =
"

#[cfg(test)]
mod tc2_rplus_contract {
    use super::*;
    use std::collections::HashMap;
    use state::WamState;

    #[test]
    fn cycle_includes_source_under_rplus() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"tc_edge\",
            &[(\"a\", \"b\"), (\"b\", \"c\"), (\"c\", \"d\"), (\"c\", \"a\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_closure_nodes(\"a\", \"tc_edge\", &mut nodes);
        nodes.sort();
        assert_eq!(
            nodes,
            vec![
                \"a\".to_string(),
                \"b\".to_string(),
                \"c\".to_string(),
                \"d\".to_string()
            ]
        );
    }

    #[test]
    fn acyclic_excludes_source() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"tc_edge\",
            &[(\"a\", \"b\"), (\"b\", \"c\"), (\"c\", \"d\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_closure_nodes(\"a\", \"tc_edge\", &mut nodes);
        nodes.sort();
        assert_eq!(
            nodes,
            vec![\"b\".to_string(), \"c\".to_string(), \"d\".to_string()]
        );
        assert!(!nodes.iter().any(|n| n == \"a\"));
    }
}
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        ( write(Out, Existing), write(Out, Unit) ),
        close(Out)).
