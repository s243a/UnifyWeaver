:- encoding(utf8).
%% Macro benchmark for the mode-analysis + IntSet visited arc on a real
%% workload.
%%
%% Generates THREE effective-distance Haskell WAM projects from the
%% same Prolog source, distinguished by which directives are in effect
%% when write_wam_haskell_project/3 runs:
%%
%%   variant_unlowered: no extra modes — Parent's bound state is
%%                      destroyed by the opaque category_parent call,
%%                      `\+ member` falls back to the slow builtin path.
%%   variant_lowered:   mode(category_ancestor(-, +, -, +)) AND
%%                      mode(category_parent(?, ?))     — Phase G
%%                      `not_member_list` lowering fires (constant-
%%                      factor win over the builtin dispatch).
%%   variant_intset:    same as `lowered` PLUS
%%                      visited_set(category_ancestor/4, 4) — Phase H
%%                      IntSet path: BuildEmptySet/SetInsert/
%%                      NotMemberSet replace the list-walk entirely
%%                      (algorithmic O(N) → O(log N) on visited check).
%%
%% Each project's standard Main.hs prints `query_ms` to stderr after
%% running effective-distance against the 1k benchmark data set. We
%% compare query_ms across all three variants.
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
    (   getenv('WAM_EFF_DIST_BENCH_SCALE', Scale), Scale \== ''
    ->  format(atom(Sub), 'data/benchmark/~w/facts.pl', [Scale])
    ;   Sub = 'data/benchmark/1k/facts.pl'
    ),
    directory_file_path(Root, Sub, Path).

workload_path(Path) :-
    repo_root(Root),
    directory_file_path(Root,
        'examples/benchmark/effective_distance.pl', Path).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

%% ========================================================================
%% Project generation
%% ========================================================================

%% Override max_depth/1 from the WAM_EFF_DIST_BENCH_MAX_DEPTH env var
%% (default 10 — the workload's own clause). Useful for the IntSet
%% crossover sweep: at small depth IntSet's per-call allocation
%% dominates the tree-descent; at deeper depth the algorithmic
%% O(log N) should eventually dominate the constant factor.
override_max_depth :-
    (   getenv('WAM_EFF_DIST_BENCH_MAX_DEPTH', V), V \== ''
    ->  atom_number(V, N), integer(N), N >= 1,
        retractall(user:max_depth(_)),
        assertz(user:max_depth(N))
    ;   true
    ).

generate_project(Variant, Dir) :-
    workload_path(Wp),
    load_files(Wp, [silent(true)]),
    %% Reset everything per variant.
    retractall(user:mode(category_ancestor(_, _, _, _))),
    retractall(user:mode(category_parent(_, _))),
    retractall(user:visited_set(_, _)),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    (   (Variant == lowered ; Variant == intset)
    ->  assertz(user:mode(category_parent(?, ?)))
    ;   true
    ),
    (   Variant == intset
    ->  assertz(user:visited_set(category_ancestor/4, 4))
    ;   true
    ),
    override_max_depth,
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
    (   getenv('WAM_EFF_DIST_BENCH_SCALE', SS), SS \== '' -> Scale = SS ; Scale = '1k' ),
    (   getenv('WAM_EFF_DIST_BENCH_MAX_DEPTH', MD), MD \== '' -> MaxDepth = MD
    ;   MaxDepth = '10 (workload default)'
    ),
    format("[INFO] scale=~w  max_depth=~w~n~n", [Scale, MaxDepth]),
    (   \+ ghc_available
    ->  format("[SKIP] ghc not on PATH~n"), halt(0)
    ;   \+ cabal_available
    ->  format("[SKIP] cabal not on PATH~n"), halt(0)
    ;   true
    ),
    %% Run all three variants twice with rotating order to spot
    %% cache-warming bias.
    build_and_run(intset,    I1, T1i),
    build_and_run(lowered,   L1, T1l),
    build_and_run(unlowered, U1, T1u),
    build_and_run(unlowered, U2, _),
    build_and_run(lowered,   L2, _),
    build_and_run(intset,    I2, _),
    format("~n========================================~n"),
    format("Results (scale=~w max_depth=~w, query_ms)~n", [Scale, MaxDepth]),
    format("========================================~n"),
    format("  trial 1: intset = ~w ms, lowered = ~w ms, unlowered = ~w ms~n",
           [I1, L1, U1]),
    format("  trial 2: unlowered = ~w ms, lowered = ~w ms, intset = ~w ms~n",
           [U2, L2, I2]),
    format("  tuple_count: intset=~w  lowered=~w  unlowered=~w (must match)~n",
           [T1i, T1l, T1u]),
    (   integer(I1), integer(I2), integer(L1), integer(L2),
        integer(U1), integer(U2),
        I1 > 0, I2 > 0, L1 > 0, L2 > 0, U1 > 0, U2 > 0
    ->  Iavg is (I1 + I2) / 2,
        Lavg is (L1 + L2) / 2,
        Uavg is (U1 + U2) / 2,
        SpeedupLowered is Uavg / Lavg,
        SpeedupIntset  is Uavg / Iavg,
        SpeedupCompound is Lavg / Iavg,
        format("~n  mean intset:    ~3f ms~n", [Iavg]),
        format("  mean lowered:   ~3f ms~n", [Lavg]),
        format("  mean unlowered: ~3f ms~n", [Uavg]),
        format("  speedup intset vs unlowered:    ~3f x  (Phase G + H combined)~n",
               [SpeedupIntset]),
        format("  speedup lowered vs unlowered:   ~3f x  (Phase G alone)~n",
               [SpeedupLowered]),
        format("  speedup intset vs lowered:      ~3f x  (Phase H IntSet alone)~n",
               [SpeedupCompound])
    ;   format("~n[WARN] could not compute means (some runs failed)~n")
    ),
    (   getenv('WAM_EFF_DIST_BENCH_KEEP', V), V \== ''
    ->  format("[INFO] keeping build dirs (WAM_EFF_DIST_BENCH_KEEP set)~n", [])
    ;   build_dir_for(intset, DI), build_dir_for(lowered, DL),
        build_dir_for(unlowered, DU),
        catch(cleanup_dir(DI), _, true),
        catch(cleanup_dir(DL), _, true),
        catch(cleanup_dir(DU), _, true)
    ),
    halt(0).

:- initialization(main, main).
