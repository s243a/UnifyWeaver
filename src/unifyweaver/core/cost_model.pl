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

    %% Hardware constants
    default_constants/1,        % -Constants
    constant/3,                 % +Name, +Constants, -Value

    %% System probes (impure)
    read_mem_available_bytes/1  % -Bytes
]).

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
