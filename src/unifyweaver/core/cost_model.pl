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
    resolve_csr_io_policy/2,    % +Options, -Policy
    resolve_csr_index_backend/2, % +Options, -Backend

    %% Hardware constants
    default_constants/1,        % -Constants
    constant/3,                 % +Name, +Constants, -Value

    %% System probes (impure)
    read_mem_available_bytes/1  % -Bytes
]).

:- use_module(library(option)).

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
    artifact_index_options(StorageKind, Opts0, IndexOpt),
    artifact_io_options(StorageKind, Opts0, IoOpt),
    append([
        relation(Relation),
        storage_kind(StorageKind),
        ordering(Ordering),
        cache_bytes(CacheBytes)
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
