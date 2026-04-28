:- encoding(utf8).
%% Macro benchmark for the mode-analysis arc on a real workload.
%%
%% Generates two effective-distance Haskell WAM projects from the
%% same Prolog source, distinguished only by which mode declarations
%% are in effect when write_wam_haskell_project/3 runs:
%%
%%   variant_lowered:   mode(category_ancestor(-, +, -, +)) AND
%%                      mode(category_parent(?, ?))     — `\+ member`
%%                      lowering fires across the parent call.
%%   variant_unlowered: only mode(category_ancestor(...)) — Parent's
%%                      bound state is destroyed by the opaque
%%                      category_parent call, lowering does not fire.
%%
%% Each project's standard Main.hs prints `query_ms` to stderr after
%% running the effective-distance workload against the 1k benchmark
%% data set. We compare query_ms across the two variants.
%%
%% Skip: same as wam_term_construction_bench.pl.

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).

tool_runs(Path, Args) :-
    catch(
        (   process_create(path(Path), Args,
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail).

ghc_available   :- tool_runs(ghc,   ['--version']).
cabal_available :- tool_runs(cabal, ['--version']).

repo_root(Root) :-
    source_file(repo_root(_), This),
    file_directory_name(This, BenchDir),
    file_directory_name(BenchDir, TestsDir),
    file_directory_name(TestsDir, Root).

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== '' -> Root = R0 ; Root = '/tmp' ).

build_dir_for(Variant, Dir) :-
    tmp_root(Root),
    atom_concat('uw_eff_dist_macro_', Variant, Sub),
    directory_file_path(Root, Sub, Dir).

facts_path(Path) :-
    repo_root(Root),
    directory_file_path(Root, 'data/benchmark/1k/facts.pl', Path).

workload_path(Path) :-
    repo_root(Root),
    directory_file_path(Root,
        'examples/benchmark/effective_distance.pl', Path).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

%% ========================================================================
%% Project generation
%% ========================================================================

generate_project(Variant, Dir) :-
    workload_path(Wp),
    load_files(Wp, [silent(true)]),
    %% Reset relevant modes; declare per variant.
    retractall(user:mode(category_ancestor(_, _, _, _))),
    retractall(user:mode(category_parent(_, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    (   Variant == lowered
    ->  assertz(user:mode(category_parent(?, ?)))
    ;   true
    ),
    ensure_dir(Dir),
    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4
    ],
    Options = [module_name('uw-eff-dist-macro'), use_hashmap(false)],
    write_wam_haskell_project(Predicates, Options, Dir).

%% ========================================================================
%% Cabal build / run
%% ========================================================================

run_cabal(Args, Dir, ExitCode, Output, ErrOutput) :-
    catch(
        setup_call_cleanup(
            process_create(path(cabal), Args,
                [cwd(Dir),
                 stdout(pipe(Out)),
                 stderr(pipe(Err)),
                 process(Pid)]),
            (   read_string(Out, _, OutStr),
                read_string(Err, _, ErrStr),
                process_wait(Pid, exit(ExitCode)),
                Output = OutStr,
                ErrOutput = ErrStr
            ),
            (catch(close(Out), _, true), catch(close(Err), _, true))
        ),
        E,
        (ExitCode = -1, format(string(Output), "~w", [E]), ErrOutput = "")).

%% Parse `query_ms=N` from the binary's stderr.
parse_query_ms(StderrStr, N) :-
    split_string(StderrStr, "\n", "\r", Lines),
    member(Line, Lines),
    sub_string(Line, 0, _, _, "query_ms="),
    sub_string(Line, 9, _, 0, NumStr),
    string_to_atom(NumStr, NumAtom),
    atom_number(NumAtom, N), !.

%% Some metrics reported alongside query_ms — useful for sanity checks.
parse_metric(StderrStr, Key, V) :-
    split_string(StderrStr, "\n", "\r", Lines),
    member(Line, Lines),
    string_concat(Key, "=", Prefix),
    string_concat(Prefix, ValStr, Line),
    string_to_atom(ValStr, V), !.

%% ========================================================================
%% Main
%% ========================================================================

build_and_run(Variant, ResultMs, TupleCount) :-
    build_dir_for(Variant, Dir),
    catch(cleanup_dir(Dir), _, true),
    format("[INFO] generating ~w project at ~w~n", [Variant, Dir]),
    generate_project(Variant, Dir),
    format("[INFO] cabal v2-build ~w~n", [Variant]),
    run_cabal(['v2-build', 'uw-eff-dist-macro'], Dir, BuildEC, _, BuildErr),
    (   BuildEC \== 0
    ->  format("[FAIL] cabal build failed for ~w (exit ~w)~n--- err ---~n~w~n",
              [Variant, BuildEC, BuildErr]),
        ResultMs = -1, TupleCount = -1
    ;   facts_path(FactsPath),
        file_directory_name(FactsPath, FactsDir),
        format("[INFO] cabal v2-run ~w (cwd=~w)~n", [Variant, FactsDir]),
        run_cabal(['v2-run', '-v0', 'uw-eff-dist-macro', '--',
                   FactsDir], Dir, RunEC, _, RunErr),
        (   RunEC \== 0
        ->  format("[FAIL] benchmark run failed for ~w (exit ~w)~n~w~n",
                  [Variant, RunEC, RunErr]),
            ResultMs = -1, TupleCount = -1
        ;   (   parse_query_ms(RunErr, ResultMs)
            ->  true
            ;   format("[FAIL] could not parse query_ms from stderr~n~w~n", [RunErr]),
                ResultMs = -1
            ),
            (   parse_metric(RunErr, "tuple_count", TupleCount)
            ->  true ; TupleCount = -1
            )
        )
    ).

cleanup_dir(Dir) :-
    (   exists_directory(Dir)
    ->  process_create(path(rm), ['-rf', Dir], [process(Pid)]),
        process_wait(Pid, _)
    ;   true
    ).

main :-
    format("~n========================================~n"),
    format("WAM effective-distance macro benchmark~n"),
    format("========================================~n~n"),
    (   \+ ghc_available
    ->  format("[SKIP] ghc not on PATH~n"), halt(0)
    ;   \+ cabal_available
    ->  format("[SKIP] cabal not on PATH~n"), halt(0)
    ;   true
    ),
    %% Run lowered first, unlowered second, lowered again, unlowered again
    %% to spot cache-warming bias.
    build_and_run(lowered,   L1, T1),
    build_and_run(unlowered, U1, T1u),
    build_and_run(unlowered, U2, _),
    build_and_run(lowered,   L2, _),
    format("~n========================================~n"),
    format("Results (1k facts, query_ms)~n"),
    format("========================================~n"),
    format("  trial 1: lowered = ~w ms,   unlowered = ~w ms~n", [L1, U1]),
    format("  trial 2: unlowered = ~w ms, lowered = ~w ms~n",   [U2, L2]),
    format("  tuple_count: lowered=~w  unlowered=~w (must match for correctness)~n",
           [T1, T1u]),
    (   integer(L1), integer(L2), integer(U1), integer(U2),
        L1 > 0, L2 > 0, U1 > 0, U2 > 0
    ->  Lavg is (L1 + L2) / 2,
        Uavg is (U1 + U2) / 2,
        Speedup is Uavg / Lavg,
        format("~n  mean lowered:   ~3f ms~n", [Lavg]),
        format("  mean unlowered: ~3f ms~n", [Uavg]),
        format("  speedup: ~3f x~n", [Speedup])
    ;   format("~n[WARN] could not compute means (some runs failed)~n")
    ),
    (   getenv('WAM_EFF_DIST_BENCH_KEEP', V), V \== ''
    ->  format("[INFO] keeping build dirs (WAM_EFF_DIST_BENCH_KEEP set)~n", [])
    ;   build_dir_for(lowered, DL), build_dir_for(unlowered, DU),
        catch(cleanup_dir(DL), _, true),
        catch(cleanup_dir(DU), _, true)
    ),
    halt(0).

:- initialization(main, main).
