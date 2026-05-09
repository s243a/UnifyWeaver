:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').

%% generate_wam_haskell_matrix_benchmark.pl
%%
%% Generates an effective-distance Haskell benchmark from OPTIMIZED Prolog,
%% with explicit control over the execution mode:
%%
%%   - interpreter + no_kernels(true)  => pure interpreter baseline
%%   - interpreter + kernels enabled   => hybrid WAM + FFI
%%   - functions   + no_kernels(true)  => lowered-only
%%   - functions   + kernels enabled   => lowered with WAM fallback + FFI
%%
%% This makes the benchmark path explicit:
%%   workload Prolog
%%     -> prolog_target optimization
%%     -> optimized Prolog predicates
%%     -> WAM/Haskell project generation
%%
%% Usage:
%%   swipl -q -s generate_wam_haskell_matrix_benchmark.pl -- \
%%       <facts.pl> <output-dir> <seeded|accumulated> <interpreter|functions> <kernels_on|kernels_off>

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom]
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom]
    ->  LmdbModeAtom = none
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> <seeded|accumulated> <interpreter|functions> <kernels_on|kernels_off> [<none|auto|true|false|resident|resident_cursor>]~n',
            []),
        halt(1)
    ),
    generate(FactsPath, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(FactsPath, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, OutputDir) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    parse_variant(VariantAtom, OptimizationOptions),
    parse_emit_mode(EmitModeAtom, EmitMode),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    parse_lmdb_mode(LmdbModeAtom, LmdbOptions),
    count_facts_in_file(FactsPath, FactCount),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),
    collect_wam_predicates(VariantAtom, Predicates),
    query_pred_for_variant(VariantAtom, QueryPredOpts),
    %% Default -A64M nursery for the matrix bench: at -N>=2 it cuts
    %% GC time from ~3.9s to ~1.1s on 100k_cats and roughly halves
    %% total_ms; the -N1 regression is ~24% (1.0 -> 1.3s) which we
    %% accept as the bench's typical run is parallel.  Documented in
    %% WAM_PERF_OPTIMIZATION_LOG.md Phase L appendix #6.  Override by
    %% passing +RTS -A1M -RTS at run time.
    append([[module_name('wam-haskell-matrix-bench'),
             emit_mode(EmitMode),
             fact_count(FactCount),
             with_rtsopts('-A64M')],
            KernelOptions, LmdbOptions, QueryPredOpts], Options),
    write_wam_haskell_project(Predicates, Options, OutputDir),
    format(user_error,
           '[WAM-Haskell-Matrix] variant=~w emit_mode=~w kernels=~w lmdb=~w fact_count=~w output=~w~n',
           [VariantAtom, EmitMode, KernelModeAtom, LmdbModeAtom, FactCount, OutputDir]).

%% count_facts_in_file(+FactsPath, -N)
%
%  Count `category_parent/2` clauses in the given facts.pl. Used to
%  populate `fact_count(N)` in Options so resolve_auto_use_lmdb/2 can
%  decide based on scale. Falls back to 0 if the file isn't readable
%  or has no category_parent/2 clauses — the auto resolver then
%  conservatively picks IntMap.
count_facts_in_file(FactsPath, N) :-
    (   exists_file(FactsPath)
    ->  setup_call_cleanup(
            open(FactsPath, read, Stream, [encoding(utf8)]),
            count_category_parent_clauses(Stream, 0, N),
            close(Stream))
    ;   N = 0
    ).

count_category_parent_clauses(Stream, Acc, N) :-
    catch(read_term(Stream, Term, []), _, (Term = end_of_file)),
    (   Term == end_of_file
    ->  N = Acc
    ;   (   nonvar(Term),
            (   Term = category_parent(_, _)
            ;   Term = (category_parent(_, _) :- _)
            )
        ->  Acc1 is Acc + 1,
            count_category_parent_clauses(Stream, Acc1, N)
        ;   count_category_parent_clauses(Stream, Acc, N)
        )
    ).

parse_lmdb_mode(none, []).
parse_lmdb_mode(auto, [use_lmdb(auto)]).
parse_lmdb_mode(true, [use_lmdb(true)]).
parse_lmdb_mode(false, [use_lmdb(false)]).
%% resident: full LMDB-resident path (Phase 2b.2). Reads intern table,
%% article-category map, and forward-edge index from named sub-dbs
%% written by the streaming-pipeline ingester
%% (src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py).
%% Requires the fixture's data.mdb to expose s2i / i2s / meta /
%% category_parent / article_category sub-dbs.
%% L2 sharded cache is the default for LMDB-using bench modes: hits
%% the kernel's repeated edge lookups (multi-parent DAG paths and
%% shared-ancestor seeds) while avoiding the per-HEC L1 duplication
%% problem (sparks have no region affinity, so per-thread caches
%% accumulate the same hot edges N times). MoE-style spark routing
%% would unlock L1; until then sharded is the right default.
%% Override at the matrix-bench-generator caller via additional Options.
parse_lmdb_mode(resident, [
    use_lmdb(true),
    lmdb_layout(dupsort),
    int_atom_seeds(lmdb),
    lmdb_cache_mode(sharded)
]).
%% resident_cursor: resident path + Phase 2b.3 cursor-based demand BFS.
%% Skips the parentsIndex pre-load step and walks the LMDB
%% category_child sub-db on demand. Requires the fixture to have been
%% ingested with the reverse-edge sub-db (use ingest_resident_lmdb_fixture.py).
%% L2 sharded cache especially matters in this mode: the kernel's
%% category_parent lookups go through cpEdgeLookup (LMDB) instead of
%% an in-memory IntMap, so cache hit rate directly closes the per-call
%% FFI overhead gap vs `resident` mode.
parse_lmdb_mode(resident_cursor, [
    use_lmdb(true),
    lmdb_layout(dupsort),
    int_atom_seeds(lmdb),
    demand_bfs_mode(cursor),
    lmdb_cache_mode(sharded)
]).

parse_variant(seeded, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false)
]).
parse_variant(accumulated, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false),
    seeded_accumulation(auto)
]).

parse_emit_mode(interpreter, interpreter).
parse_emit_mode(functions, functions).

parse_kernel_mode(kernels_on, []).
parse_kernel_mode(kernels_off, [no_kernels(true)]).

query_pred_for_variant(seeded, []).
query_pred_for_variant(accumulated, [
    query_pred('category_ancestor$effective_distance_sum_selected/3')
]).

collect_wam_predicates(seeded, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:power_sum_bound/4
]).
collect_wam_predicates(accumulated, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).
