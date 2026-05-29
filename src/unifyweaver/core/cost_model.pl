:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% cost_model.pl — Cache-regime-aware access-pattern cost model
%
% First-principles cost estimates for choosing between sorted-seek and
% sequential-scan patterns over LMDB-backed (or similar) data sources,
% and for deciding whether cache warming will pay off.
%
% This module is opt-in: targets that want regime-aware decisions call
% these predicates with workload metadata + a few hardware constants
% (or default constants) and get back a recommended access pattern or
% a warming threshold. Targets that don't opt in keep their existing
% behaviour.
%
% See docs/design/CACHE_COST_MODEL_PHILOSOPHY.md for the full theory.
% See docs/design/WAM_PERF_OPTIMIZATION_LOG.md Phase M appendix #10
% for the empirical measurements that motivate the formulas.
%
% The model deliberately stays a closed-form formula with constants
% within an order of magnitude of any commodity hardware. It's
% groundwork for Phase 2c, when article-level (rather than category-
% level) workloads start to push past free RAM.

:- module(cost_model, [
    %% Regime estimation
    cache_regime/3,             % +RFreeBytes, +WBytes, -FHot

    %% Cost estimates (returns milliseconds)
    scan_time_ms/4,             % +WBytes, +FHot, +Constants, -ScanMs
    seek_time_ms/4,             % +KKeys, +FHot, +Constants, -SortMs

    %% Crossover and recommendations
    crossover_k/4,              % +WBytes, +FHot, +Constants, -KCross
    warming_payoff_m/5,         % +WWarmBytes, +KPerQuery, +FHot, +Constants, -MMin
    recommend_access_pattern/5, % +KKeys, +WBytes, +RFreeBytes, +Constants, -Pattern
    resolve_reverse_index/2,    % +Options, -ReverseIndex
    validate_reverse_index_option/2, % +Spec, -Normalized

    %% LMDB materialisation-mode resolver (eager | lazy | cached)
    resolve_auto_lmdb_materialisation/2, % +Options, -Mode
    resolve_lmdb_cache_capacity/2,       % +Options, -CapacityEntries
    lmdb_cache_capacity_free_pct/2,      % +Options, -Fraction
    lmdb_cache_capacity_floor_bytes/2,   % +Options, -Bytes
    lmdb_edge_size_bytes/1,              % -Bytes

    resolve_csr_io_policy/2,    % +Options, -Policy
    resolve_csr_index_backend/2, % +Options, -Backend
    csr_index_backend_options_from_benchmark_tsv/3,
                                % +Path, +WorkloadOptions, -Options
    csr_index_backend_options_from_benchmark_rows/3,
                                % +Rows, +WorkloadOptions, -Options

    %% Hardware constants
    default_constants/1,        % -Constants
    constant/3,                 % +Name, +Constants, -Value

    %% System probes (impure)
    read_mem_available_bytes/1  % -Bytes
]).

:- use_module(library(option)).
:- use_module(library(apply)).
:- use_module(library(pairs)).
:- use_module(library(readutil)).

%% =====================================================================
%% Hardware constants
%% =====================================================================

%! default_constants(-Constants) is det.
%
%  SSD-typical defaults for commodity Linux/WSL hardware. Values are
%  within an order of magnitude of any modern consumer machine.
%
%  Constants is an option list of:
%    - s_mem_seq_bps(BytesPerSec) — sequential bandwidth from page cache
%    - s_disk_seq_bps(BytesPerSec) — sequential bandwidth from cold storage
%    - t_mem_seek_us(Microseconds) — one cached B-tree-walk seek
%    - t_disk_seek_us(Microseconds) — one uncached B-tree-walk seek
%
%  Override per-machine by passing a custom list to the cost predicates.
default_constants([
    s_mem_seq_bps(5_000_000_000),   % 5 GB/s mmap memcpy
    s_disk_seq_bps(500_000_000),    % 500 MB/s sequential SSD
    t_mem_seek_us(1.0),             % 1 µs cached B-tree walk
    t_disk_seek_us(100.0)           % 100 µs SSD random read
]).

%! constant(+Name, +Constants, -Value) is det.
%
%  Look up a named constant from a constants list, falling back to
%  default_constants/1 if not present.
constant(Name, Constants, Value) :-
    Term =.. [Name, V],
    (   member(Term, Constants)
    ->  Value = V
    ;   default_constants(Defaults),
        member(Term, Defaults)
    ->  Value = V
    ;   throw(error(domain_error(cost_model_constant, Name), _))
    ).

%% =====================================================================
%% Regime
%% =====================================================================

%! cache_regime(+RFreeBytes, +WBytes, -FHot) is det.
%
%  FHot is the fraction of working set W that fits in available RAM.
%  Returns 1.0 (all hot) when working set fits entirely; drops toward
%  0.0 as W exceeds R_free. Returns 1.0 for empty/zero working sets.
cache_regime(_RFreeBytes, WBytes, 1.0) :-
    WBytes =< 0, !.
cache_regime(RFreeBytes, WBytes, FHot) :-
    Ratio is RFreeBytes / WBytes,
    FHot is min(1.0, max(0.0, Ratio)).

%% =====================================================================
%% Effective bandwidth and latency
%% =====================================================================

bandwidth_eff(FHot, Constants, BwBps) :-
    constant(s_mem_seq_bps, Constants, SMem),
    constant(s_disk_seq_bps, Constants, SDisk),
    BwBps is FHot * SMem + (1.0 - FHot) * SDisk.

latency_eff(FHot, Constants, LatUs) :-
    constant(t_mem_seek_us, Constants, TMem),
    constant(t_disk_seek_us, Constants, TDisk),
    LatUs is FHot * TMem + (1.0 - FHot) * TDisk.

%% =====================================================================
%% Cost estimates
%% =====================================================================

%! scan_time_ms(+WBytes, +FHot, +Constants, -ScanMs) is det.
%
%  Estimated time to sequentially scan WBytes given cache regime FHot.
scan_time_ms(WBytes, FHot, Constants, ScanMs) :-
    bandwidth_eff(FHot, Constants, BwBps),
    Seconds is WBytes / BwBps,
    ScanMs is Seconds * 1000.0.

%! seek_time_ms(+KKeys, +FHot, +Constants, -SortMs) is det.
%
%  Estimated time to do K well-distributed sorted seeks given cache
%  regime FHot. Uses the *amortised* seek cost — for tightly-clustered
%  keys the actual cost will be lower because consecutive keys often
%  hit the same B-tree leaf. The model is therefore conservative
%  (slightly overestimates seek cost), which biases toward scan when
%  the choice is borderline.
seek_time_ms(KKeys, FHot, Constants, SortMs) :-
    latency_eff(FHot, Constants, LatUs),
    Microseconds is KKeys * LatUs,
    SortMs is Microseconds / 1000.0.

%% =====================================================================
%% Crossover
%% =====================================================================

%! crossover_k(+WBytes, +FHot, +Constants, -KCross) is det.
%
%  KCross is the number of distinct-key seeks at which sorted-seek time
%  equals full-scan time. If K_query < KCross use sorted seeks; if
%  K_query > KCross use a full scan. The crossover shifts left
%  dramatically in the cold regime (FHot small) because random seeks
%  become real disk operations.
crossover_k(WBytes, FHot, Constants, KCross) :-
    bandwidth_eff(FHot, Constants, BwBps),
    latency_eff(FHot, Constants, LatUs),
    %% K * LatUs/1e6 = WBytes/BwBps  →  K = (WBytes * 1e6) / (BwBps * LatUs)
    KCross is (WBytes * 1.0e6) / (BwBps * LatUs).

%% =====================================================================
%% Warming-pays-off threshold
%% =====================================================================

%! warming_payoff_m(+WWarmBytes, +KPerQuery, +FHot, +Constants, -MMin) is det.
%
%  MMin is the minimum number of queries (M) at which pre-warming
%  WWarmBytes pays off, given each query touches KPerQuery keys and
%  the current cache regime is FHot.
%
%  In the all-hot regime (FHot ≈ 1) the cold/hot per-query times are
%  nearly identical, so the denominator is tiny and MMin is huge —
%  warming doesn't help. As FHot drops, the denominator grows and
%  MMin shrinks. The exact zero of the denominator is suppressed via
%  a small epsilon to avoid division by zero; callers that get
%  MMin = positive_infinity should treat it as "warming pointless."
%
%  Returns positive_infinity when FHot ≈ 1 (warming pointless).
warming_payoff_m(WWarmBytes, _KPerQuery, FHot, _Constants, positive_infinity) :-
    FHot >= 1.0 - 1.0e-6, !,
    WWarmBytes >= 0.
warming_payoff_m(WWarmBytes, KPerQuery, FHot, Constants, MMin) :-
    %% Cost to warm: scan WWarmBytes once at the current regime's
    %% bandwidth (we have to actually load the bytes from disk to
    %% warm the cache, so use the cold-path bandwidth fraction).
    scan_time_ms(WWarmBytes, FHot, Constants, TWarmMs),
    %% Per-query cost cold (working set not in cache) vs hot (in cache):
    %% the difference is the cold-cache penalty per query.
    seek_time_ms(KPerQuery, 0.0, Constants, TQueryColdMs),
    seek_time_ms(KPerQuery, 1.0, Constants, TQueryHotMs),
    Delta is TQueryColdMs - TQueryHotMs,
    (   Delta =< 1.0e-9
    ->  MMin = positive_infinity
    ;   MMinReal is TWarmMs / Delta,
        MMin is ceiling(MMinReal)
    ).

%% =====================================================================
%% Recommendation
%% =====================================================================

%! recommend_access_pattern(+KKeys, +WBytes, +RFreeBytes, +Constants, -Pattern) is det.
%
%  Pattern is `sort` if K < K_cross, else `scan`. RFreeBytes is the
%  free RAM available to the process (use read_mem_available_bytes/1
%  on Linux to obtain it).
%
%  Border behaviour: when K = K_cross, prefer sort (the formula
%  slightly overestimates seek cost so a tied result is more likely
%  scan than the math implies).
recommend_access_pattern(KKeys, WBytes, RFreeBytes, Constants, Pattern) :-
    cache_regime(RFreeBytes, WBytes, FHot),
    crossover_k(WBytes, FHot, Constants, KCross),
    (   KKeys =< KCross
    ->  Pattern = sort
    ;   Pattern = scan
    ).

%% =====================================================================
%% Reverse-index artifact policy
%% =====================================================================

%! resolve_reverse_index(+Options, -ReverseIndex) is det.
%
%  Resolve the reverse-index artifact request from workload metadata.
%  The current `auto` policy is deliberately conservative: it returns
%  `none` unless descendant lookup is semantically required. Explicit
%  reverse_index(...) terms are normalized and validated.
resolve_reverse_index(Options, ReverseIndex) :-
    option(reverse_index(Spec0), Options, auto),
    resolve_reverse_index_spec(Spec0, Options, ReverseIndex).

resolve_reverse_index_spec(auto, Options, ReverseIndex) :-
    !,
    option(needs_descendant_lookup(NeedsDesc), Options, false),
    option(expected_query_count_per_artifact(QueryCount), Options, 1),
    (   NeedsDesc == true
    ->  ReverseIndex = artifact([
            relation(category_child/2),
            storage_kind(mmap_array_artifact),
            phase(planning_only),
            id_encoding(int32_le)
        ])
    ;   QueryCount < 100
    ->  ReverseIndex = none
    ;   ReverseIndex = none
    ).
resolve_reverse_index_spec(Spec, _Options, ReverseIndex) :-
    validate_reverse_index_option(Spec, ReverseIndex).

%! validate_reverse_index_option(+Spec, -Normalized) is det.
%
%  Validate the user-facing reverse_index(...) spec and fill in
%  conservative defaults. Throws domain_error/2 for unsupported
%  values.
validate_reverse_index_option(none, none) :- !.
validate_reverse_index_option(auto, auto) :- !.
validate_reverse_index_option(lmdb(Opts0), lmdb(Opts)) :-
    !,
    normalize_reverse_options(lmdb, Opts0, Opts).
validate_reverse_index_option(mmap_array(Opts0), mmap_array(Opts)) :-
    !,
    normalize_reverse_options(mmap_array, Opts0, Opts).
validate_reverse_index_option(csr(Opts0), csr(Opts)) :-
    !,
    normalize_reverse_options(csr, Opts0, Opts).
validate_reverse_index_option(artifact(Opts0), artifact(Opts)) :-
    !,
    normalize_reverse_options(artifact, Opts0, Opts).
validate_reverse_index_option(Spec, _) :-
    throw(error(domain_error(reverse_index_option, Spec), _)).

normalize_reverse_options(Kind, Opts0, Opts) :-
    must_be(list, Opts0),
    validate_known_reverse_options(Opts0),
    option(phase(Phase), Opts0, planning_only),
    validate_reverse_phase(Phase),
    option(id_encoding(IdEncoding), Opts0, int32_le),
    validate_id_encoding(IdEncoding),
    normalize_kind_options(Kind, Opts0, KindOpts),
    append([phase(Phase), id_encoding(IdEncoding)], KindOpts, Opts).

normalize_kind_options(lmdb, _Opts0, []).
normalize_kind_options(mmap_array, _Opts0, []).
normalize_kind_options(csr, Opts0, Opts) :-
    option(ordering(Ordering), Opts0, parent_sort),
    validate_reverse_ordering(Ordering),
    resolve_csr_index_backend(Opts0, IndexBackend),
    option(cache_bytes(CacheBytes), Opts0, 0),
    validate_nonnegative_integer(cache_bytes, CacheBytes),
    option(block_size_edges(BlockSizeEdges), Opts0, 0),
    validate_nonnegative_integer(block_size_edges, BlockSizeEdges),
    resolve_csr_io_policy(Opts0, IoPolicy),
    Opts = [
        ordering(Ordering),
        index_backend(IndexBackend),
        io_policy(IoPolicy),
        cache_bytes(CacheBytes),
        block_size_edges(BlockSizeEdges)
    ].
normalize_kind_options(artifact, Opts0, Opts) :-
    option(relation(Relation), Opts0, category_child/2),
    validate_predicate_indicator(Relation),
    option(storage_kind(StorageKind), Opts0, mmap_array_artifact),
    validate_storage_kind(StorageKind),
    option(ordering(Ordering), Opts0, parent_sort),
    validate_reverse_ordering(Ordering),
    option(cache_bytes(CacheBytes), Opts0, 0),
    validate_nonnegative_integer(cache_bytes, CacheBytes),
    option(block_size_edges(BlockSizeEdges), Opts0, 0),
    validate_nonnegative_integer(block_size_edges, BlockSizeEdges),
    artifact_index_options(StorageKind, Opts0, IndexOpt),
    artifact_io_options(StorageKind, Opts0, IoOpt),
    append([
        relation(Relation),
        storage_kind(StorageKind),
        ordering(Ordering),
        cache_bytes(CacheBytes),
        block_size_edges(BlockSizeEdges)
    ], IndexOpt, PrefixOpts),
    append(PrefixOpts, IoOpt, Opts).

artifact_index_options(csr_pread_artifact, Opts0, [index_backend(IndexBackend)]) :-
    !,
    resolve_csr_index_backend(Opts0, IndexBackend).
artifact_index_options(_StorageKind, Opts0, []) :-
    \+ option(index_backend(_), Opts0),
    !.
artifact_index_options(StorageKind, _Opts0, _) :-
    throw(error(permission_error(use, index_backend, StorageKind), _)).

artifact_io_options(csr_pread_artifact, Opts0, [io_policy(IoPolicy)]) :-
    !,
    resolve_csr_io_policy(Opts0, IoPolicy).
artifact_io_options(_StorageKind, Opts0, []) :-
    \+ option(io_policy(_), Opts0),
    !.
artifact_io_options(StorageKind, _Opts0, _) :-
    throw(error(permission_error(use, io_policy, StorageKind), _)).

%! resolve_csr_io_policy(+Options, -Policy) is det.
%
%  Resolve CSR I/O policy. Explicit non-auto values are validated and
%  preserved. `auto` picks buffered_pread_drop for planning/warmup,
%  direct_io only when all direct-I/O preconditions are declared and
%  measured to win, and buffered_pread otherwise.
resolve_csr_io_policy(Options, Policy) :-
    option(io_policy(Policy0), Options, auto),
    resolve_csr_io_policy_value(Policy0, Options, Policy).

resolve_csr_io_policy_value(auto, Options, buffered_pread_drop) :-
    option(phase(Phase), Options, planning_only),
    memberchk(Phase, [planning_only, cache_warmup]),
    !.
resolve_csr_io_policy_value(auto, Options, direct_io) :-
    option(phase(runtime_available), Options),
    option(block_size_edges(BlockSizeEdges), Options),
    BlockSizeEdges >= 65536,
    option(platform_supports_direct_io(true), Options),
    option(alignment_verified(true), Options),
    option(measured_direct_io_win(true), Options),
    !.
resolve_csr_io_policy_value(auto, _Options, buffered_pread) :- !.
resolve_csr_io_policy_value(Policy, _Options, Policy) :-
    validate_csr_io_policy(Policy),
    !.

%! resolve_csr_index_backend(+Options, -Backend) is det.
%
%  Resolve CSR index backend. Omitted values keep the current sorted
%  array behavior. Explicit `auto` may select lmdb_offset when measured
%  lookup savings amortize the marginal build cost and the offset index
%  fits the configured memory budget.
resolve_csr_index_backend(Options, Backend) :-
    option(index_backend(Backend0), Options, sorted_array),
    resolve_csr_index_backend_value(Backend0, Options, Backend).

resolve_csr_index_backend_value(auto, Options, lmdb_offset) :-
    csr_lmdb_offset_pays_off(Options),
    csr_lmdb_offset_memory_fits(Options),
    !.
resolve_csr_index_backend_value(auto, _Options, sorted_array) :- !.
resolve_csr_index_backend_value(Backend, _Options, Backend) :-
    validate_csr_index_backend(Backend),
    !.

%! csr_index_backend_options_from_benchmark_tsv(+Path, +WorkloadOptions, -Options) is det.
%
%  Read benchmark_reverse_csr_lookup.py TSV output and produce the
%  measured option terms consumed by resolve_csr_index_backend/2.
%  WorkloadOptions should provide expected_child_lookups_per_query/1 and
%  expected_query_count_per_artifact/1. It may also provide
%  available_memory_bytes/1 or lmdb_offset_memory_fits(true).
csr_index_backend_options_from_benchmark_tsv(Path, WorkloadOptions, Options) :-
    read_file_to_string(Path, Text, []),
    tsv_rows(Text, Rows),
    csr_index_backend_options_from_benchmark_rows(Rows, WorkloadOptions, Options).

%! csr_index_backend_options_from_benchmark_rows(+Rows, +WorkloadOptions, -Options) is det.
%
%  Rows is a list of Field-ValueString pairs as produced by tsv_rows/2.
csr_index_backend_options_from_benchmark_rows(Rows, WorkloadOptions, Options) :-
    must_be(list, WorkloadOptions),
    memberchk(expected_child_lookups_per_query(LookupsPerQuery), WorkloadOptions),
    memberchk(expected_query_count_per_artifact(QueryCount), WorkloadOptions),
    benchmark_row_for_index_backend(Rows, sorted_array, SortedRow),
    benchmark_row_for_index_backend(Rows, lmdb_offset, LmdbRow),
    row_number(SortedRow, median_ms, SortedMsPer1000),
    row_number(LmdbRow, median_ms, LmdbMsPer1000),
    row_number(SortedRow, csr_build_seconds, SortedBuildSeconds),
    row_number(LmdbRow, csr_build_seconds, LmdbBuildSeconds),
    row_number(LmdbRow, offset_index_bytes, LmdbOffsetBytes),
    BaseOptions = [
        index_backend(auto),
        expected_child_lookups_per_query(LookupsPerQuery),
        expected_query_count_per_artifact(QueryCount),
        sorted_array_lookup_ms_per_1000(SortedMsPer1000),
        lmdb_offset_lookup_ms_per_1000(LmdbMsPer1000),
        sorted_array_build_seconds(SortedBuildSeconds),
        lmdb_offset_build_seconds(LmdbBuildSeconds),
        lmdb_offset_bytes(LmdbOffsetBytes)
    ],
    include(csr_index_passthrough_workload_option, WorkloadOptions, PassThrough),
    append(BaseOptions, PassThrough, Options).

tsv_rows(Text, Rows) :-
    split_string(Text, "\n", "\n\r", RawLines),
    exclude(=(""), RawLines, Lines),
    Lines = [HeaderLine|DataLines],
    split_string(HeaderLine, "\t", "", HeaderStrings),
    maplist(atom_string, Headers, HeaderStrings),
    maplist(tsv_row(Headers), DataLines, Rows).

tsv_row(Headers, Line, Row) :-
    split_string(Line, "\t", "\r", Values),
    same_length(Headers, Values),
    pairs_keys_values(Row, Headers, Values).

benchmark_row_for_index_backend(Rows, Backend, Row) :-
    atom_string(Backend, BackendString),
    member(Row, Rows),
    memberchk(index_backend-BackendString, Row),
    !.
benchmark_row_for_index_backend(_Rows, Backend, _Row) :-
    throw(error(existence_error(csr_benchmark_row, Backend), _)).

row_number(Row, Field, Number) :-
    memberchk(Field-String, Row),
    number_string(Number, String),
    !.
row_number(Row, Field, _Number) :-
    (   memberchk(Field-String, Row)
    ->  throw(error(type_error(number_string(Field), String), _))
    ;   throw(error(existence_error(csr_benchmark_field, Field), _))
    ).

csr_index_passthrough_workload_option(available_memory_bytes(_)).
csr_index_passthrough_workload_option(csr_index_memory_fraction(_)).
csr_index_passthrough_workload_option(lmdb_offset_memory_fits(true)).

csr_lmdb_offset_pays_off(Options) :-
    option(expected_child_lookups_per_query(LookupsPerQuery), Options),
    validate_nonnegative_number(expected_child_lookups_per_query, LookupsPerQuery),
    LookupsPerQuery > 0,
    option(expected_query_count_per_artifact(QueryCount), Options),
    validate_nonnegative_number(expected_query_count_per_artifact, QueryCount),
    QueryCount > 0,
    option(sorted_array_lookup_ms_per_1000(SortedMsPer1000), Options),
    validate_nonnegative_number(sorted_array_lookup_ms_per_1000, SortedMsPer1000),
    option(lmdb_offset_lookup_ms_per_1000(LmdbMsPer1000), Options),
    validate_nonnegative_number(lmdb_offset_lookup_ms_per_1000, LmdbMsPer1000),
    LookupSavingsMsPer1000 is SortedMsPer1000 - LmdbMsPer1000,
    LookupSavingsMsPer1000 > 0,
    option(sorted_array_build_seconds(SortedBuildSeconds), Options),
    validate_nonnegative_number(sorted_array_build_seconds, SortedBuildSeconds),
    option(lmdb_offset_build_seconds(LmdbBuildSeconds), Options),
    validate_nonnegative_number(lmdb_offset_build_seconds, LmdbBuildSeconds),
    MarginalBuildSeconds is max(0, LmdbBuildSeconds - SortedBuildSeconds),
    TotalLookupSavingsSeconds is
        (LookupsPerQuery * QueryCount / 1000.0) *
        (LookupSavingsMsPer1000 / 1000.0),
    TotalLookupSavingsSeconds >= MarginalBuildSeconds.

csr_lmdb_offset_memory_fits(Options) :-
    option(lmdb_offset_memory_fits(true), Options),
    !.
csr_lmdb_offset_memory_fits(Options) :-
    option(lmdb_offset_bytes(LmdbOffsetBytes), Options),
    validate_nonnegative_number(lmdb_offset_bytes, LmdbOffsetBytes),
    option(available_memory_bytes(AvailableBytes), Options),
    validate_nonnegative_number(available_memory_bytes, AvailableBytes),
    AvailableBytes > 0,
    option(csr_index_memory_fraction(Fraction), Options, 0.05),
    validate_fraction(csr_index_memory_fraction, Fraction),
    LmdbOffsetBytes =< AvailableBytes * Fraction.

validate_known_reverse_options([]).
validate_known_reverse_options([Opt|Rest]) :-
    functor(Opt, Name, 1),
    reverse_option_key(Name),
    !,
    validate_known_reverse_options(Rest).
validate_known_reverse_options([Opt|_]) :-
    throw(error(domain_error(reverse_index_option_key, Opt), _)).

reverse_option_key(phase).
reverse_option_key(id_encoding).
reverse_option_key(ordering).
reverse_option_key(index_backend).
reverse_option_key(cache_bytes).
reverse_option_key(block_size_edges).
reverse_option_key(io_policy).
reverse_option_key(expected_child_lookups_per_query).
reverse_option_key(expected_query_count_per_artifact).
reverse_option_key(sorted_array_lookup_ms_per_1000).
reverse_option_key(lmdb_offset_lookup_ms_per_1000).
reverse_option_key(sorted_array_build_seconds).
reverse_option_key(lmdb_offset_build_seconds).
reverse_option_key(lmdb_offset_bytes).
reverse_option_key(available_memory_bytes).
reverse_option_key(csr_index_memory_fraction).
reverse_option_key(lmdb_offset_memory_fits).
reverse_option_key(platform_supports_direct_io).
reverse_option_key(alignment_verified).
reverse_option_key(measured_direct_io_win).
reverse_option_key(relation).
reverse_option_key(storage_kind).

validate_reverse_phase(Phase) :-
    memberchk(Phase, [planning_only, cache_warmup, runtime_available]),
    !.
validate_reverse_phase(Phase) :-
    throw(error(domain_error(reverse_index_phase, Phase), _)).

validate_id_encoding(Encoding) :-
    memberchk(Encoding, [int32_le, decimal_utf8]),
    !.
validate_id_encoding(Encoding) :-
    throw(error(domain_error(reverse_index_id_encoding, Encoding), _)).

validate_csr_io_policy(Policy) :-
    memberchk(Policy, [auto, buffered_pread, buffered_pread_drop, direct_io]),
    !.
validate_csr_io_policy(Policy) :-
    throw(error(domain_error(csr_io_policy, Policy), _)).

validate_csr_index_backend(Backend) :-
    memberchk(Backend, [
        auto,
        sorted_array,
        lmdb_offset,
        dense_direct
    ]),
    !.
validate_csr_index_backend(Backend) :-
    throw(error(domain_error(csr_index_backend, Backend), _)).

validate_nonnegative_number(_Name, Value) :-
    number(Value),
    Value >= 0,
    !.
validate_nonnegative_number(Name, Value) :-
    throw(error(domain_error(nonnegative_number(Name), Value), _)).

validate_fraction(_Name, Value) :-
    number(Value),
    Value >= 0,
    Value =< 1,
    !.
validate_fraction(Name, Value) :-
    throw(error(domain_error(fraction(Name), Value), _)).

validate_reverse_ordering(Ordering) :-
    memberchk(Ordering, [
        parent_sort,
        root_bfs,
        component_degree,
        multi_root_bfs,
        graph_partition,
        spectral,
        bandwidth_heuristic,
        embedding_cluster
    ]),
    !.
validate_reverse_ordering(Ordering) :-
    throw(error(domain_error(reverse_index_ordering, Ordering), _)).

validate_storage_kind(StorageKind) :-
    memberchk(StorageKind, [
        binary_artifact,
        delimited_artifact,
        lmdb_artifact,
        mmap_array_artifact,
        csr_pread_artifact
    ]),
    !.
validate_storage_kind(StorageKind) :-
    throw(error(domain_error(reverse_index_storage_kind, StorageKind), _)).

validate_predicate_indicator(Name/Arity) :-
    atom(Name),
    integer(Arity),
    Arity >= 0,
    !.
validate_predicate_indicator(PI) :-
    throw(error(domain_error(predicate_indicator, PI), _)).

validate_nonnegative_integer(_Name, Value) :-
    integer(Value),
    Value >= 0,
    !.
validate_nonnegative_integer(Name, Value) :-
    throw(error(domain_error(nonnegative_integer(Name), Value), _)).

%% =====================================================================
%% LMDB materialisation-mode resolver
%% =====================================================================
%%
%% `lmdb_materialisation(auto | eager | lazy | cached)` is a codegen
%% option (WAM-Rust today; target-agnostic so Haskell/C# can reuse it).
%% This resolver implements the `auto` decision per
%% docs/design/WAM_LMDB_LAZY_SPECIFICATION.md §7.2 and the cache-sizing
%% rule in WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md §3.1.
%%
%% IMPORTANT: mode selection is a *codegen-time* decision driven purely
%% by static fixture metadata. It must NOT read `/proc/meminfo` —
%% MemAvailable is a runtime quantity, and baking a codegen-time
%% reading into the emitted binary would be wrong on any machine other
%% than the build host. The MemAvailable-dependent cache-capacity clamp
%% (§3.1) is therefore applied at *runtime* in the generated target
%% code; this resolver only emits the static `unrestricted_working_set`
%% default plus the clamp parameters (cap_pct, headroom floor).

%! lmdb_edge_size_bytes(-Bytes) is det.
%
%  Approximate in-cache cost of one parent edge: an i32 child key plus
%  an i32 parent value plus container/hashing overhead. Used to turn a
%  demand-set *edge count* into a memory footprint for the §3.1 clamp.
lmdb_edge_size_bytes(12).

%! resolve_auto_lmdb_materialisation(+Options, -Mode) is det.
%
%  Pick `eager` | `lazy` | `cached` from workload metadata. Explicit
%  callers pass `lmdb_materialisation(eager|lazy|cached)` and get it
%  back unchanged; `auto` (or an unset option) runs the decision tree.
%
%  Decision tree (spec §7.2):
%    1. If an explicit `memory_budget(Bytes)` is *declared* and the
%       demand set doesn't fit in it, the demand set can't be held
%       resident → use a lazy form (`lazy` if the workload is
%       segregated, else `cached`). No live MemAvailable read here —
%       only a caller-declared budget triggers this branch.
%    2. Otherwise the demand set fits. Choose `eager` iff the one-shot
%       up-front materialisation is cheap (estimated cold build time
%       within the per-process build budget, scaled by the expected
%       query count so a long-lived process tolerates a bigger build).
%       Else fall to the lazy form (`lazy` if segregated, else
%       `cached`).
%
%  Recognised metadata options (all optional, conservative defaults):
%    - fact_count(F)                        total edges in the source
%    - demand_set_estimate(D)               edges the queries will touch
%                                           (defaults to fact_count)
%    - workload_segregated(Bool)            default false
%    - expected_query_count_per_process(NQ) default 1
%    - memory_budget(Bytes)                 declared cap; absent = unbounded
%    - eager_build_budget_ms(Ms)            default 1000
%    - constants(C)                         hardware constants list
resolve_auto_lmdb_materialisation(Options, Mode) :-
    option(lmdb_materialisation(Requested), Options, auto),
    resolve_lmdb_materialisation_value(Requested, Options, Mode).

resolve_lmdb_materialisation_value(eager,  _Options, eager) :- !.
resolve_lmdb_materialisation_value(lazy,   _Options, lazy) :- !.
resolve_lmdb_materialisation_value(cached, _Options, cached) :- !.
resolve_lmdb_materialisation_value(auto, Options, Mode) :-
    !,
    lmdb_demand_set_estimate(Options, D),
    (   lmdb_demand_set_overflows_budget(D, Options)
    ->  lmdb_lazy_form(Options, Mode)
    ;   lmdb_eager_build_is_cheap(D, Options)
    ->  Mode = eager
    ;   lmdb_lazy_form(Options, Mode)
    ).
resolve_lmdb_materialisation_value(Other, _Options, _) :-
    throw(error(domain_error(lmdb_materialisation, Other), _)).

%  When the demand set can't live in RAM we still need on-demand
%  access; segregated workloads prefer bare `lazy` (no cross-query
%  reuse to exploit), everything else prefers `cached`.
lmdb_lazy_form(Options, Mode) :-
    option(workload_segregated(WS), Options, false),
    (   WS == true
    ->  Mode = lazy
    ;   Mode = cached
    ).

%  Demand-set edge estimate, defaulting to fact_count, then to 0.
lmdb_demand_set_estimate(Options, D) :-
    option(fact_count(F), Options, 0),
    option(demand_set_estimate(D0), Options, F),
    (   number(D0), D0 > 0
    ->  D = D0
    ;   D = F
    ).

%  True only when an explicit budget is declared AND the demand set's
%  footprint exceeds it. No `memory_budget` option => never fires
%  (codegen does not read MemAvailable).
lmdb_demand_set_overflows_budget(D, Options) :-
    option(memory_budget(Budget), Options),
    number(Budget),
    lmdb_edge_size_bytes(EdgeBytes),
    Footprint is D * EdgeBytes,
    Footprint > Budget.

%  Eager wins when the up-front demand-set build is cheap. Estimate the
%  build as D cold parent-edge seeks; tolerate up to
%  eager_build_budget_ms × max(1, NQ) (a long-lived process amortises a
%  bigger one-time build). With the default 1 s budget and 100 µs cold
%  seek the eager↔cached crossover lands at ~10k edges — so 1k → eager,
%  simplewiki/enwiki → cached, matching plan §3.
lmdb_eager_build_is_cheap(D, Options) :-
    option(eager_build_budget_ms(Budget0), Options, 1000),
    option(expected_query_count_per_process(NQ), Options, 1),
    resolve_cost_constants(Options, C),
    seek_time_ms(D, 0.0, C, BuildMs),
    Budget is Budget0 * max(1, NQ),
    BuildMs =< Budget.

resolve_cost_constants(Options, C) :-
    (   option(constants(C0), Options), is_list(C0)
    ->  C = C0
    ;   default_constants(C)
    ).

%! resolve_lmdb_cache_capacity(+Options, -CapacityEntries) is det.
%
%  Codegen-time *default* cache capacity in entries (distinct child
%  keys). This is the `unrestricted_working_set` term of §3.1 — the
%  capacity that would serve the workload with no evictions, derived
%  from the demand-set estimate. The MemAvailable clamp of §3.1 is NOT
%  applied here; the generated target applies it at runtime. Floored at
%  1024 so tiny fixtures still get a usable cache.
%
%  An explicit `cache_capacity(N)` option overrides the estimate.
resolve_lmdb_cache_capacity(Options, Capacity) :-
    (   option(cache_capacity(Explicit), Options), integer(Explicit), Explicit > 0
    ->  Capacity = Explicit
    ;   lmdb_demand_set_estimate(Options, D),
        Capacity is max(1024, D)
    ).

%! lmdb_cache_capacity_free_pct(+Options, -Fraction) is det.
%
%  `cap_pct` from §3.1 — the fraction of (MemAvailable − headroom) the
%  runtime cache may claim. Override per-predicate with
%  `cache_capacity_free_pct(F)`.
lmdb_cache_capacity_free_pct(Options, Fraction) :-
    option(cache_capacity_free_pct(Fraction0), Options, 0.5),
    validate_fraction(cache_capacity_free_pct, Fraction0),
    Fraction = Fraction0.

%! lmdb_cache_capacity_floor_bytes(+Options, -Bytes) is det.
%
%  Fixed component of the §3.1 `headroom_floor`
%  (`max(512 MB, 0.10 × MemTotal)`). The runtime takes the max of this
%  and 10% of MemTotal. Override with `cache_capacity_floor_bytes(B)`.
lmdb_cache_capacity_floor_bytes(Options, Bytes) :-
    option(cache_capacity_floor_bytes(Bytes0), Options, 536870912), % 512 MiB
    validate_nonnegative_integer(cache_capacity_floor_bytes, Bytes0),
    Bytes = Bytes0.

%% =====================================================================
%% System probe
%% =====================================================================

%! read_mem_available_bytes(-Bytes) is det.
%
%  Read MemAvailable from /proc/meminfo. Returns the kernel's own
%  estimate of how much RAM can be allocated without swapping —
%  i.e. MemFree + reclaimable page cache + slab. This is the right
%  number for "how much free RAM does the process effectively have",
%  not MemFree, which underestimates by the size of the cache.
%
%  Throws if /proc/meminfo isn't readable (non-Linux platforms).
%  Callers should catch and fall back to a sensible default (or
%  pass a configured value through workload metadata).
read_mem_available_bytes(Bytes) :-
    setup_call_cleanup(
        open('/proc/meminfo', read, Stream),
        read_meminfo_field(Stream, "MemAvailable", KiB),
        close(Stream)
    ),
    Bytes is KiB * 1024.

read_meminfo_field(Stream, Field, Value) :-
    read_line_to_string(Stream, Line),
    (   Line == end_of_file
    ->  throw(error(domain_error(meminfo_field, Field), _))
    ;   split_string(Line, ":", "", [FieldStr, RestStr]),
        FieldStr == Field
    ->  split_string(RestStr, " ", " kKB", Parts),
        once((member(NumStr, Parts), NumStr \= "")),
        number_string(Value, NumStr)
    ;   read_meminfo_field(Stream, Field, Value)
    ).
