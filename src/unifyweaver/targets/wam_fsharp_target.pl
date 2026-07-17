:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_fsharp_target.pl - WAM-to-F# Transpilation Target
%
% Compiles WAM instructions to F# code using F#'s native immutable Map<'K,'V>.
%
% Design mirrors the Haskell WAM target closely (use Haskell as primary
% reference, Rust as secondary — see PR #1378 perf log for lessons applied):
%
%   - Value array for registers → O(1) array access with Array.copy snapshots
%     for choice points (same insight as Go's [512]Value)
%   - WamContext separated from WamState (hot/cold split — Haskell Phase-B)
%   - Labels pre-resolved to int PCs at load time (Phase-C win: -47%)
%   - Skip WAM-compilation of FFI-owned fact predicates (Phase-D: -70%)
%   - Atom interning at FFI boundary (Phase-D: -48% query time)
%     Implemented via buildAtomInternTable in generated Program.fs
%
% Key F# idioms vs Haskell:
%   - `match x with | Pat -> ...` instead of `case x of`
%   - `{ s with Field = v }` record update syntax
%   - `option<'T>` / `Some` / `None` instead of `Maybe` / `Just` / `Nothing`
%   - `Map.tryFind k m` instead of `Map.lookup k m`
%   - `Map.add k v m` instead of `Map.insert k v m`
%   - No Bang patterns / UNPACK — F# is strict by default (no laziness overhead)
%   - `Array.get arr i` instead of `arr ! i` for O(1) instruction fetch
%   - `Parallel.map` (TPL) for intra-query parallelism instead of `parMap`
%
% Pipeline:
%   Prolog source → fsharp_target.pl (native lowering, preferred)
%                 → wam_target.pl (WAM compilation, fallback)
%                 → wam_fsharp_target.pl (THIS FILE: WAM → F#)
%
% See: docs/design/WAM_PERF_OPTIMIZATION_LOG.md
%      src/unifyweaver/targets/wam_haskell_target.pl (primary reference)
%      src/unifyweaver/targets/wam_elixir_target.pl  (structural reference)

:- module(wam_fsharp_target, [
    compile_wam_predicate_to_fsharp/4,   % +Pred/Arity, +WamCode, +Options, -FSharpCode
    compile_wam_runtime_to_fsharp/3,     % +Options, +DetectedKernels, -Code
    write_wam_fsharp_project/3,          % +Predicates, +Options, +ProjectDir
    wam_fsharp_native_kernel_supported/1, % +recursive_kernel(Kind,...)
    wam_fsharp_native_kernel_kind/1,     % ?Kind — authoritative native allow-list
    wam_fsharp_resolve_emit_mode/2,      % +Options, -Mode
    wam_fsharp_partition_predicates/5,   % +Mode, +Predicates, +DK, -Interp, -Lowered
    % Fact-shape classification parity with Haskell/Elixir hybrid targets
    classify_fact_predicate_fs/4,        % +PredIndicator, +WamLines, +Options, -Info
    fsharp_fact_only/2,                  % +Segments, -Bool
    fsharp_first_arg_groundness/3,       % +Segments, +Arity, -Status
    fsharp_pick_layout/5,                % +PredIndicator, +NClauses, +FactOnly, +Options, -Layout
    split_wam_into_segments_fs/2,        % +Lines, -Segments
    % ISO error config + rewrite (parity with C++/Elixir/Python).  See
    % docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md for the shared
    % contract.  Initial F# adoption ships is_iso/2 and is_lax/2; the
    % comparison-op and succ/2 variants are follow-ups.
    iso_errors_resolve_options/2,        % +Options, -Config
    iso_errors_load_config/2,            % +File, -Config
    iso_errors_mode_for/3,               % +Config, +PI, -Mode
    iso_errors_warn_multi_module/2,      % +Config, +Predicates
    iso_errors_rewrite/4,                % +Config, +PI, +Items0, -Items
    iso_errors_rewrite_text/4,           % +Config, +PI, +WamText0, -WamText
    wam_fsharp_iso_audit/3,              % +Predicates, +Options, -Audit
    wam_fsharp_iso_audit_report/1,       % +Audit
    % Cost-based auto-resolvers
    resolve_auto_edge_store_fs/2,        % +Options0, -Options
    resolve_auto_lmdb_materialisation_fs/2, % +Options0, -Options
    resolve_auto_lmdb_cache_tier_fs/2,   % +Options0, -Options
    resolve_auto_csr_index_backend_fs/2, % +Options0, -Options
    resolve_fsharp_cost_options/2        % +Options0, -Options
]).

:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/recurrence_evaluation_strategy',
              [select_evaluation_strategy/3]).
:- use_module('../core/recurrence_inputs',
              [build_recurrence_term/3, build_workload_signals/2]).
:- use_module('../core/recursive_kernel_detection',
             [detect_recursive_kernel/4, kernel_metadata/4, kernel_config/2,
              kernel_register_layout/2, kernel_native_call/2, kernel_template_file/2]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../core/cost_model', [
    recommend_access_pattern/5,
    read_mem_available_bytes/1,
    resolve_csr_index_backend/2
]).
:- use_module('../core/purity_certificate', [analyze_predicate_purity/2]).
:- use_module('../bindings/fsharp_wam_bindings').
:- use_module('../core/iso_errors',
              [ iso_errors_resolve_options/2,
                iso_errors_load_config/2,
                iso_errors_mode_for/3,
                iso_errors_warn_multi_module/2,
                iso_errors_rewrite/4,
                iso_errors_rewrite_item/3,
                iso_errors_audit_normalise_pi/2,
                iso_errors_audit_walk/5
              ]).
:- use_module('../targets/wam_text_parser',
              [ wam_tokenize_line/2,
                wam_recognise_instruction/2
              ]).
:- use_module('../core/prolog_term_parser').
:- use_module('../core/cpp_runtime_parser_wrappers').
:- use_module(wam_runtime_parser_capability, [
    parser_dependent_body_goal/2,
    wam_target_runtime_parser/3
]).

% Phase 3 lowered emitter lives in wam_fsharp_lowered_emitter.
:- reexport('wam_fsharp_lowered_emitter',
            [wam_fsharp_lowerable/3, lower_predicate_to_fsharp/4]).

% ============================================================================
% Emit mode selector — identical hierarchy to Haskell target
% ============================================================================
%
% Three modes:
%   interpreter   — every predicate via the instruction-array interpreter
%   functions     — every predicate attempts lowering; falls back to interp
%   mixed(List)   — named predicates attempt lowering; rest use interpreter
%
% Selector hierarchy (checked in order):
%   1. emit_mode(Mode) option
%   2. user:wam_fsharp_emit_mode(Mode) dynamic fact
%   3. default: interpreter

:- multifile user:wam_fsharp_emit_mode/1.

%% wam_fsharp_resolve_emit_mode(+Options, -Mode)
wam_fsharp_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  wam_fsharp_validate_emit_mode(M0, Mode)
    ;   catch(user:wam_fsharp_emit_mode(M1), _, fail)
    ->  wam_fsharp_validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

wam_fsharp_validate_emit_mode(interpreter, interpreter) :- !.
wam_fsharp_validate_emit_mode(functions,   functions)   :- !.
wam_fsharp_validate_emit_mode(mixed(L), mixed(L)) :- is_list(L), !.
wam_fsharp_validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_fsharp_emit_mode, Other),
                wam_fsharp_resolve_emit_mode/2)).

%% wam_fsharp_partition_predicates(+Mode, +Predicates, +DK, -Interp, -Lowered)
wam_fsharp_partition_predicates(interpreter, Predicates, _, Predicates, []) :- !.
wam_fsharp_partition_predicates(functions, Predicates, DK, Interpreted, Lowered) :- !,
    pairs_keys(DK, KernelKeys),
    wam_fsharp_partition_try_lower(Predicates, KernelKeys, Interpreted, Lowered).
wam_fsharp_partition_predicates(mixed(HotPreds), Predicates, DK, Interpreted, Lowered) :- !,
    pairs_keys(DK, KernelKeys),
    wam_fsharp_partition_mixed(Predicates, HotPreds, KernelKeys, Interpreted, Lowered).

wam_fsharp_partition_try_lower([], _, [], []).
wam_fsharp_partition_try_lower([P|Rest], KK, Interpreted, Lowered) :-
    pred_key_fs(P, Key),
    (   member(Key, KK)
    ->  Interpreted = [P|IR],
        wam_fsharp_partition_try_lower(Rest, KK, IR, Lowered)
    ;   wam_fsharp_predicate_wamcode(P, WamCode),
        (   wam_fsharp_lowerable(P, WamCode, _Reason)
        ->  Lowered = [P|LR],
            wam_fsharp_partition_try_lower(Rest, KK, Interpreted, LR)
        ;   Interpreted = [P|IR],
            wam_fsharp_partition_try_lower(Rest, KK, IR, Lowered)
        )
    ).

wam_fsharp_partition_mixed([], _, _, [], []).
wam_fsharp_partition_mixed([P|Rest], HotPreds, KK, Interpreted, Lowered) :-
    pred_key_fs(P, Key),
    (   member(Key, KK)
    ->  Interpreted = [P|IR],
        wam_fsharp_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
    ;   wam_fsharp_indicator_in_list(P, HotPreds)
    ->  wam_fsharp_predicate_wamcode(P, WamCode),
        (   wam_fsharp_lowerable(P, WamCode, _Reason)
        ->  Lowered = [P|LR],
            wam_fsharp_partition_mixed(Rest, HotPreds, KK, Interpreted, LR)
        ;   Interpreted = [P|IR],
            wam_fsharp_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
        )
    ;   Interpreted = [P|IR],
        wam_fsharp_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
    ).

pred_key_fs(P, Key) :-
    (P = _Mod:Pred/Arity -> true ; P = Pred/Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]).

wam_fsharp_indicator_in_list(P, HotPreds) :- member(P, HotPreds), !.
wam_fsharp_indicator_in_list(_Mod:Pred/Arity, HotPreds) :-
    member(Pred/Arity, HotPreds), !.

wam_fsharp_predicate_wamcode(PI, WamCode) :-
    %% Module-qualified indicators (e.g. prolog_term_parser:foo/2 from
    %% the compiled runtime parser library) are passed through to the
    %% WAM compiler verbatim so it looks clauses up in the right
    %% module.  Stripping to the bare Name/Arity, the previous
    %% behaviour, defaulted lookup to `user:` and produced
    %% "WAM target: no clauses for user:foo/2" for library
    %% predicates.
    (   PI = user:Pred/Arity
    ->  wam_target:compile_predicate_to_wam(Pred/Arity, [], WamCode)
    ;   PI = _Module:_Pred/_Arity
    ->  wam_target:compile_predicate_to_wam(PI, [], WamCode)
    ;   wam_target:compile_predicate_to_wam(PI, [], WamCode)
    ).

% ============================================================================
% FFI-owned fact detection (Phase D: skip WAM-compilation for pure FFI facts)
%
% A predicate is FFI-owned if:
%   1. It was detected as a recursive kernel (has FFI bindings)
%   2. All its clauses are pure facts (body = true)
%
% Skipping WAM compilation of these predicates was the -70% total query
% time win in Haskell/Go — the FFI kernel path handles them directly.
% ============================================================================

%% is_ffi_owned_fact_fs(+PredIndicator, +DetectedKernels)
is_ffi_owned_fact_fs(PI, DetectedKernels) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    member(Key-_, DetectedKernels),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    forall(member(_-Body, Clauses), Body == true).

%% ffi_owned_fact_filter_fs(+DetectedKernels, +PI) — exclude/3 callback
ffi_owned_fact_filter_fs(DetectedKernels, PI) :-
    is_ffi_owned_fact_fs(PI, DetectedKernels).

% ============================================================================
% Kernel detection (shared with Haskell target pattern)
% ============================================================================

%% wam_fsharp_native_kernel_kind(?Kind)
%  Authoritative allow-list of recursive-kernel kinds for which F# can
%  actually emit a nativeKernel_* handler. Detection may fire on more
%  kinds (shared detector); unsupported kinds must stay ordinary WAM
%  predicates — never CallForeign / executeForeign to a missing symbol.
%
%  Inventory (do not confuse with this allow-list):
%   - Generic WAM recursive tc/2 already works (no_kernels(true)).
%   - category_ancestor / bidirectional_ancestor have .fs.mustache bodies.
%   - lmdb_fact_source reachableToRoot* is demand-pruning BFS over
%     category_child (includes root); not a drop-in for transitive_closure2.
%   - BuildEmptySet / SetInsert / NotMemberSet + FFIStreamRetry are the
%     WAM/FFI plumbing TC2/TD3 streaming reuses.
%   - transitive_distance3 Mustache handler emits (atom,int) pairs via the
%     existing multi-output binder (dist+ contract).
%   - transitive_parent_distance4 Mustache handler emits (atom,atom,int)
%     triples via the existing three-output binder (shortest-positive
%     parents contract).
%   - transitive_step_parent_distance5 Mustache handler emits
%     (atom,atom,atom,int) quadruples via the existing four-output binder
%     (shortest-positive correlated step/parent contract).
%   - weighted_shortest_path3 Mustache handler emits (atom,float) pairs via
%     the existing two-output binder (finite nonnegative Dijkstra contract).
%     Relation-keyed weighted facts are materialized into WcFfiWeightedFacts.
wam_fsharp_native_kernel_kind(category_ancestor).
wam_fsharp_native_kernel_kind(bidirectional_ancestor).
wam_fsharp_native_kernel_kind(transitive_closure2).
wam_fsharp_native_kernel_kind(transitive_distance3).
wam_fsharp_native_kernel_kind(transitive_parent_distance4).
wam_fsharp_native_kernel_kind(transitive_step_parent_distance5).
wam_fsharp_native_kernel_kind(weighted_shortest_path3).

fsharp_kernel_template_path(Kind, AbsPath) :-
    kernel_template_file(Kind, HsTemplateFile),
    atom_concat(Base, '.hs.mustache', HsTemplateFile),
    atom_concat(Base, '.fs.mustache', TemplateFile),
    atom_concat('templates/targets/fsharp_wam/', TemplateFile, RelPath),
    (   source_file(wam_fsharp_target, SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),
        file_directory_name(SrcDir, TargetsDir),
        file_directory_name(TargetsDir, UnifyWeaverDir),
        file_directory_name(UnifyWeaverDir, ProjectDir),
        atomic_list_concat([ProjectDir, '/', RelPath], AbsPath)
    ;   AbsPath = RelPath
    ).

%% wam_fsharp_native_kernel_supported(+Kernel)
wam_fsharp_native_kernel_supported(recursive_kernel(Kind, _Pred, _ConfigOps)) :-
    wam_fsharp_native_kernel_kind(Kind),
    fsharp_kernel_template_path(Kind, HandlerPath),
    exists_file(HandlerPath).

filter_supported_kernel_pairs_fs([], []).
filter_supported_kernel_pairs_fs([Key-Kernel|Rest], Supported) :-
    (   wam_fsharp_native_kernel_supported(Kernel)
    ->  Supported = [Key-Kernel|SupportedRest]
    ;   Kernel = recursive_kernel(Kind, _, _),
        format(user_error,
               '[WAM-FSharp] detected ~w (~w) but no native F# handler; falling back to WAM~n',
               [Key, Kind]),
        Supported = SupportedRest
    ),
    filter_supported_kernel_pairs_fs(Rest, SupportedRest).

detect_kernels_fs([], []).
detect_kernels_fs([PI|Rest], Kernels) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses \= [],
        detect_recursive_kernel(Pred, Arity, Clauses, Kernel)
    ->  format(atom(Key), '~w/~w', [Pred, Arity]),
        filter_supported_kernel_pairs_fs([Key-Kernel], Supported),
        append(Supported, RestKernels, Kernels)
    ;   Kernels = RestKernels
    ),
    detect_kernels_fs(Rest, RestKernels).

% ============================================================================
% Base PC computation (for lowered function PC offset correctness)
% ============================================================================

compute_base_pcs_fs(Predicates, Map) :-
    compute_base_pcs_fs_(Predicates, 1, Map).

compute_base_pcs_fs_([], _, []).
compute_base_pcs_fs_([PI|Rest], StartPC, [Key-StartPC|RestMap]) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_fsharp_predicate_wamcode(PI, WamCode),
    count_wam_instructions_fs(WamCode, Count),
    NextPC is StartPC + Count,
    compute_base_pcs_fs_(Rest, NextPC, RestMap).

count_wam_instructions_fs(WamCode, Count) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    include(is_wam_instruction_line_fs, Lines, InstrLines),
    length(InstrLines, Count).

is_wam_instruction_line_fs(Line) :-
    split_string(Line, "", " \t", [Trimmed]),
    Trimmed \== "",
    \+ sub_string(Trimmed, _, 1, 0, ":").

predicate_base_pc_fs(P, Map, PC) :-
    (   P = _Mod:Pred/Arity -> true ; P = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    (   member(Key-PC, Map) -> true ; PC = 1 ).

% ============================================================================
% lower_all: run the lowered emitter over LoweredList
% ============================================================================

lower_all_fs(Preds, BasePCMap, DetectedKernels, Entries) :-
    lower_all_fs(Preds, BasePCMap, DetectedKernels, [], Entries).
lower_all_fs([], _, _, _, []).
lower_all_fs([P|Rest], BasePCMap, DetectedKernels, Options, [Entry|RestEntries]) :-
    wam_fsharp_predicate_wamcode(P, WamCode),
    predicate_base_pc_fs(P, BasePCMap, BasePC),
    pairs_keys(DetectedKernels, ForeignKeys),
    % Thread the user Options (t9_min_rows / t9_max_rows / fact_table_inline)
    % so lower_predicate_to_fsharp can apply the T9 fact-table inline.
    lower_predicate_to_fsharp(P, WamCode,
        [base_pc(BasePC), foreign_preds(ForeignKeys)|Options], Entry),
    lower_all_fs(Rest, BasePCMap, DetectedKernels, Options, RestEntries).


% ============================================================================
% FACT SHAPE CLASSIFICATION (parity infra with Haskell/Elixir hybrid targets)
% ============================================================================

classify_fact_predicate_fs(PredIndicator, WamLines, Options, Info) :-
    split_wam_into_segments_fs(WamLines, Segments),
    length(Segments, NClauses),
    fsharp_fact_only(Segments, FactOnly),
    (PredIndicator = _:_/Arity -> true ; PredIndicator = _/Arity),
    fsharp_first_arg_groundness(Segments, Arity, FirstArg),
    fsharp_pick_layout(PredIndicator, NClauses, FactOnly, Options, Layout),
    Info = fact_shape_info(NClauses, FactOnly, FirstArg, Layout).

split_wam_into_segments_fs([], []).
split_wam_into_segments_fs([L0|Rest], Segments) :-
    normalize_space(string(L), L0),
    (   L == ""
    ->  split_wam_into_segments_fs(Rest, Segments)
    ;   sub_string(L, _, 1, 0, ":")
    ->  split_wam_into_segments_body_fs(Rest, [], Body, Rem),
        Segments = [segment(L, Body)|Tail],
        split_wam_into_segments_fs(Rem, Tail)
    ;   split_wam_into_segments_body_fs(Rest, [L], Body, Rem),
        Segments = [segment('<entry>', Body)|Tail],
        split_wam_into_segments_fs(Rem, Tail)
    ).

split_wam_into_segments_body_fs([], Acc, Body, []) :-
    reverse(Acc, Body).
split_wam_into_segments_body_fs([L0|Rest], Acc, Body, Rem) :-
    normalize_space(string(L), L0),
    (   L == ""
    ->  split_wam_into_segments_body_fs(Rest, Acc, Body, Rem)
    ;   sub_string(L, _, 1, 0, ":")
    ->  reverse(Acc, Body),
        Rem = [L|Rest]
    ;   split_wam_into_segments_body_fs(Rest, [L|Acc], Body, Rem)
    ).

fsharp_fact_only([], true).
fsharp_fact_only([segment(_, Instrs)|Rest], Bool) :-
    fsharp_segment_fact_only(Instrs, B0),
    fsharp_fact_only(Rest, BR),
    (B0 == true, BR == true -> Bool = true ; Bool = false).

fsharp_segment_fact_only([], false).
fsharp_segment_fact_only(Instrs, Bool) :-
    exclude(fsharp_nonsemantic_line, Instrs, Sem),
    (   append(Prefix, ["proceed"], Sem),
        forall(member(I, Prefix), fsharp_fact_head_instr(I))
    ->  Bool = true
    ;   Bool = false
    ).

fsharp_nonsemantic_line(L) :- sub_string(L, 0, _, _, "%"), !.
fsharp_nonsemantic_line(_):- fail.

fsharp_fact_head_instr(I) :-
    member(I, ["allocate", "deallocate"]), !.
fsharp_fact_head_instr(I) :- sub_string(I, 0, _, _, "get_"), !.
fsharp_fact_head_instr(I) :- sub_string(I, 0, _, _, "unify_"), !.

fsharp_first_arg_groundness([], _, mixed).
fsharp_first_arg_groundness(Segments, _Arity, Status) :-
    maplist(fsharp_seg_first_arg_mode, Segments, Modes),
    (   Modes \= [], forall(member(M, Modes), M == ground)
    ->  Status = always_ground
    ;   Modes \= [], forall(member(M, Modes), M == non_ground)
    ->  Status = never_ground
    ;   Status = mixed
    ).

fsharp_seg_first_arg_mode(segment(_, Instrs), Mode) :-
    (   member(I, Instrs),
        (sub_string(I, 0, _, _, "get_constant"), sub_string(I, _, _, 0, ", A1")
        ;sub_string(I, 0, _, _, "get_structure"), sub_string(I, _, _, 0, ", A1")
        ;sub_string(I, 0, _, _, "get_list A1"))
    ->  Mode = ground
    ;   member(I2, Instrs),
        (sub_string(I2, 0, _, _, "get_variable"), sub_string(I2, _, _, 0, ", A1")
        ;sub_string(I2, 0, _, _, "get_value"), sub_string(I2, _, _, 0, ", A1"))
    ->  Mode = non_ground
    ;   Mode = mixed
    ).

fsharp_pick_layout(_PredIndicator, NClauses, FactOnly, Options, Layout) :-
    option(fact_layout_threshold(T), Options, 32),
    (   FactOnly == true,
        NClauses >= T
    ->  Layout = inline_data
    ;   Layout = compiled
    ).

% ============================================================================
% PHASE 2: Backtrack Function
% ============================================================================

backtrack_fsharp(Code) :-
    Code = '/// Restore state from the top choice point.
/// Dispatches: aggregate frame → finalizeAggregate, builtin → resumeBuiltin, normal → restore.
and backtrack (s: WamState) : WamState option =
    match s.WsCPs with
    | [] -> None
    | cp :: rest ->
        match cp.CpAggFrame with
        | Some af -> finalizeAggregate af.AggReturnPC s
        | None ->
        match cp.CpBuiltin with
        | Some bs -> resumeBuiltin bs cp rest s
        | None ->
        let trailLen  = cp.CpTrailLen
        let diff      = s.WsTrailLen - trailLen
        let newEntries= s.WsTrail |> List.take diff |> List.rev
        let restoredBindings = List.fold undoBinding cp.CpBindings newEntries
        // Bug B fix from PR #2350: keep the CP on the stack.  Standard
        // WAM convention is that backtrack RESTORES from the top CP
        // without popping it; only TrustMe pops.  Between successive
        // backtracks, RetryMeElse modifies the top CP''s CpNextPC to
        // point to the next clause''s entry — so the CP MUST remain on
        // the stack for the next-clause logic to work.  Previously
        // WsCPs was set to `rest` here, popping the CP; that
        // accidentally worked for chains of length ≤ 2 (the only CP
        // was needed once) but failed for 3+ when RetryMeElse arrived
        // at an empty stack.  See PR #2350''s query smoke
        // `query_X_parent_eve_BUG` for the regression test that
        // surfaced this.
        Some { s with
                 WsPC       = cp.CpNextPC
                 // Array.copy: cp stays on the stack across backtracks (so
                 // RetryMeElse can modify it), and putReg now mutates
                 // WsRegs in place.  Without copying here, in-place writes
                 // after a restore would corrupt the CP''s saved registers.
                 WsRegs     = Array.copy cp.CpRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = trailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = restoredBindings
                 WsCutBar   = cp.CpCutBar
                 WsB0Stack  = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                 // WsCPs intentionally unchanged: keep CP on stack so
                 // RetryMeElse can modify it (or TrustMe can pop it later).
               }

and resumeBuiltin (bs: BuiltinState) (cp: ChoicePoint) (rest: ChoicePoint list) (s: WamState) : WamState option =
    match bs with
    | FactRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | FactRetry (vid, v :: vs, retPC) ->
        let newBindings = Map.add vid (Atom v) cp.CpBindings
        let newRegs     = Array.copy cp.CpRegs
        newRegs.[2]    <- Atom v
        let newCPs      = match vs with
                          | [] -> rest
                          | _  -> { cp with CpBuiltin = Some (FactRetry (vid, vs, retPC)) } :: rest
        let diff = s.WsTrailLen - cp.CpTrailLen
        Some { s with
                 WsPC       = retPC
                 WsRegs     = newRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = cp.CpTrailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = newBindings
                 WsCutBar   = cp.CpCutBar
                 WsB0Stack  = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }
    | HopsRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | HopsRetry (vid, h :: hs, retPC) ->
        let newBindings = Map.add vid (Integer h) cp.CpBindings
        let newRegs     = Array.copy cp.CpRegs
        newRegs.[3]    <- Integer h
        let newCPs      = match hs with
                          | [] -> rest
                          | _  -> { cp with CpBuiltin = Some (HopsRetry (vid, hs, retPC)) } :: rest
        let diff = s.WsTrailLen - cp.CpTrailLen
        Some { s with
                 WsPC       = retPC
                 WsRegs     = newRegs
                 WsStack    = cp.CpStack
                 WsCP       = cp.CpCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = cp.CpTrailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = newBindings
                 WsCutBar   = cp.CpCutBar
                 WsB0Stack  = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }
    | SelectRetry (_, _, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | SelectRetry (elemReg, outReg, candidates, retPC) ->
        // Restore from snapshot ONCE up front, then walk candidates with
        // unifyVal until one succeeds (or all fail).  unifyVal returns
        // None on failure without committing trail entries, so iterating
        // candidates against the restored state is safe.
        let diff = s.WsTrailLen - cp.CpTrailLen
        let restoredS = { s with
                             WsRegs     = Array.copy cp.CpRegs
                             WsStack    = cp.CpStack
                             WsCP       = cp.CpCP
                             WsTrail    = List.skip diff s.WsTrail
                             WsTrailLen = cp.CpTrailLen
                             WsHeap     = List.take cp.CpHeapLen s.WsHeap
                             WsHeapLen  = cp.CpHeapLen
                             WsBindings = cp.CpBindings
                             WsCutBar   = cp.CpCutBar
                             WsB0Stack  = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                             WsPC       = retPC }
        let rec tryNext pairs =
            match pairs with
            | [] -> backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
            | (sel, restList) :: more ->
                match getReg elemReg restoredS with
                | None -> backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
                | Some elem_ ->
                    match unifyVal elem_ sel restoredS with
                    | None    -> tryNext more
                    | Some s2 ->
                        match bindOutput outReg (VList restList) s2 with
                        | None    -> tryNext more
                        | Some s3 ->
                            let newCPs =
                                match more with
                                | [] -> rest
                                | _  -> { cp with CpBuiltin = Some (SelectRetry (elemReg, outReg, more, retPC)) } :: rest
                            Some { s3 with
                                     WsPC       = retPC
                                     WsStack    = cp.CpStack
                                     WsCP       = cp.CpCP
                                     WsCutBar   = cp.CpCutBar
                                     WsB0Stack  = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                                     WsCPs      = newCPs
                                     WsCPsLen   = List.length newCPs }
        tryNext candidates
    | MemberRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | MemberRetry (elemReg, candidates, retPC) ->
        // Restore from snapshot, then walk remaining list elements with
        // unifyTerms until one succeeds (or all fail).  Mirrors the
        // SelectRetry pattern: the snapshot lets unifyTerms attempts be
        // independent of each other''s trail entries.
        let diff = s.WsTrailLen - cp.CpTrailLen
        let restoredS = { s with
                             WsRegs     = Array.copy cp.CpRegs
                             WsStack    = cp.CpStack
                             WsCP       = cp.CpCP
                             WsTrail    = List.skip diff s.WsTrail
                             WsTrailLen = cp.CpTrailLen
                             WsHeap     = List.take cp.CpHeapLen s.WsHeap
                             WsHeapLen  = cp.CpHeapLen
                             WsBindings = cp.CpBindings
                             WsCutBar   = cp.CpCutBar
                             WsB0Stack  = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                             WsPC       = retPC }
        let rec tryNext cands =
            match cands with
            | [] -> backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
            | x :: more ->
                match getReg elemReg restoredS with
                | None -> backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
                | Some elem_ ->
                    let xd = derefVar restoredS.WsBindings x
                    match unifyTerms elem_ xd restoredS with
                    | None    -> tryNext more
                    | Some s2 ->
                        let newCPs =
                            match more with
                            | [] -> rest
                            | _  -> { cp with CpBuiltin = Some (MemberRetry (elemReg, more, retPC)) } :: rest
                        Some { s2 with
                                 WsPC      = retPC
                                 WsStack   = cp.CpStack
                                 WsCP      = cp.CpCP
                                 WsCutBar  = cp.CpCutBar
                                 WsB0Stack = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                                 WsCPs     = newCPs
                                 WsCPsLen  = List.length newCPs }
        tryNext candidates
    | FactTableRetry (_, [], _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | FactTableRetry (args, candidates, retPC) ->
        // Restore from snapshot, then walk remaining candidate rows; unify all
        // columns of the next matching row against the query args (the snapshot
        // makes each row attempt independent).  Mirrors MemberRetry but with a
        // multi-column unify per row.
        let diff = s.WsTrailLen - cp.CpTrailLen
        let restoredS = { s with
                             WsRegs     = Array.copy cp.CpRegs
                             WsStack    = cp.CpStack
                             WsCP       = cp.CpCP
                             WsTrail    = List.skip diff s.WsTrail
                             WsTrailLen = cp.CpTrailLen
                             WsHeap     = List.take cp.CpHeapLen s.WsHeap
                             WsHeapLen  = cp.CpHeapLen
                             WsBindings = cp.CpBindings
                             WsCutBar   = cp.CpCutBar
                             WsB0Stack  = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                             WsPC       = retPC }
        let arity = List.length args
        let rec tryNext cs =
            match cs with
            | [] -> backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
            | row :: more ->
                match row with
                | VList cols when List.length cols = arity ->
                    match unifyColumns (List.zip args cols) restoredS with
                    | Some s2 ->
                        let newCPs =
                            match more with
                            | [] -> rest
                            | _  -> { cp with CpBuiltin = Some (FactTableRetry (args, more, retPC)) } :: rest
                        Some { s2 with
                                 WsPC      = retPC
                                 WsStack   = cp.CpStack
                                 WsCP      = cp.CpCP
                                 WsCutBar  = cp.CpCutBar
                                 WsB0Stack = (let n = List.length s.WsB0Stack - cp.CpB0StackLen in if n > 0 then List.skip n s.WsB0Stack else s.WsB0Stack)
                                 WsCPs     = newCPs
                                 WsCPsLen  = List.length newCPs }
                    | None -> tryNext more
                | _ -> tryNext more
        tryNext candidates
    | FFIStreamRetry (_, _, [], _, _, _, _) ->
        backtrack { s with WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
    | FFIStreamRetry (outRegs, outVars, tuple :: restTuples, retPC,
                      returnCP, returnCutBar, returnB0Stack) ->
        let newRegs     = Array.copy cp.CpRegs
        List.iter2 (fun rN v -> newRegs.[rN] <- v) outRegs tuple
        let newBindings = List.fold2
                            (fun m vid v -> if vid = -1 then m else Map.add vid v m)
                            cp.CpBindings outVars tuple
        let newCPs = match restTuples with
                     | [] -> rest
                     | _  -> { cp with CpBuiltin = Some (FFIStreamRetry (outRegs, outVars, restTuples, retPC,
                                                                         returnCP, returnCutBar, returnB0Stack)) } :: rest
        let diff = s.WsTrailLen - cp.CpTrailLen
        Some { s with
                 WsPC       = retPC
                 WsRegs     = newRegs
                 WsStack    = cp.CpStack
                 WsCP       = returnCP
                 WsTrail    = List.skip diff s.WsTrail
                 WsTrailLen = cp.CpTrailLen
                 WsHeap     = List.take cp.CpHeapLen s.WsHeap
                 WsHeapLen  = cp.CpHeapLen
                 WsBindings = newBindings
                 WsCutBar   = returnCutBar
                 WsB0Stack  = returnB0Stack
                 WsCPs      = newCPs
                 WsCPsLen   = List.length newCPs }

and backtrackInner (_returnPC: int) (s: WamState) : WamState option =
    match s.WsCPs with
    | cp :: _ when cp.CpAggFrame.IsSome -> None   // reached aggregate frame = done
    | _ -> backtrack s

and finalizeAggregate (returnPC: int) (s: WamState) : WamState option =
    let rec go cps =
        match cps with
        | [] -> None
        | cp :: rest ->
            match cp.CpAggFrame with
            | Some af ->
                let accum  = List.rev s.WsAggAccum
                let result = applyAggregation af.AggType accum
                let cpState= { s with WsRegs = cp.CpRegs; WsStack = cp.CpStack; WsBindings = cp.CpBindings }
                let resVal = getReg af.AggResReg cpState
                let diff   = s.WsTrailLen - cp.CpTrailLen
                let restoredTrail = List.skip diff s.WsTrail
                let finalRegs, finalBindings, finalTrail, finalTrailLen =
                    match resVal with
                    | Some (Unbound vid) ->
                        let r = Array.copy cp.CpRegs
                        r.[af.AggResReg] <- result
                        ( r
                        , Map.add vid result cp.CpBindings
                        , { TrailVarId = vid; TrailOldVal = Map.tryFind vid cp.CpBindings } :: restoredTrail
                        , cp.CpTrailLen + 1 )
                    // Array.copy: putReg now mutates WsRegs in place, so the
                    // finalize''d state must not alias cp.CpRegs.
                    | _ -> (Array.copy cp.CpRegs, cp.CpBindings, restoredTrail, cp.CpTrailLen)
                Some { s with
                         WsPC       = returnPC
                         WsRegs     = finalRegs
                         WsStack    = cp.CpStack
                         WsBindings = finalBindings
                         WsTrail    = finalTrail
                         WsTrailLen = finalTrailLen
                         WsHeap     = List.take cp.CpHeapLen s.WsHeap
                         WsHeapLen  = cp.CpHeapLen
                         WsCP       = cp.CpCP
                         WsCPs      = rest
                         WsCPsLen   = s.WsCPsLen - 1
                         WsAggAccum = [] }
            | None -> go rest
    go s.WsCPs

and applyAggregation (typ: string) (vals: Value list) : Value =
    match typ with
    | "sum" ->
        let toNum = function Integer n -> float n | Float f -> f | _ -> 0.0
        let s = List.sumBy toNum vals
        if float (int s) = s then Integer (int s) else Float s
    | "count"   -> Integer (List.length vals)
    | "collect" -> VList vals
    | _         -> VList vals'.

% ============================================================================
% PHASE 3: Step Function
% ============================================================================

step_function_fsharp(Code) :-
    Code = '/// Execute a single WAM instruction.
/// WamContext is read-only — threaded through without per-step allocation.
let rec step (ctx: WamContext) (s: WamState) (instr: Instruction) : WamState option =
    match instr with
    | GetConstant (c, ai) ->
        let valOpt = getReg ai s
        match valOpt with
        | Some v when v = c -> Some { s with WsPC = s.WsPC + 1 }
        // Empty-list equivalence (issue #2400 continuation).
        // `Atom "[]"` and `VList []` both denote Prolog''s empty
        // list.  Two materialization paths produce them — `Atom`
        // when GetNil / a literal `[]` constant was emitted, `VList
        // []` when an addToBuilder run collapsed a `[H|[]]` to a
        // singleton then materialized as the empty list at the
        // tail.  GetConstant Atom "[]" must succeed against either.
        | Some (VList []) when c = Atom "[]" -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Unbound vid) ->
            let r = Array.copy s.WsRegs
            r.[ai] <- c
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = r
                     WsBindings= Map.add vid c s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | _ -> None

    | GetVariable (xn, ai) ->
        match getReg ai s with
        | Some dv -> Some (putReg xn dv { s with WsPC = s.WsPC + 1 })
        | None    -> None

    | GetValue (xn, ai) ->
        // Structural unify (not F# `=`). Ground append/reverse answers
        // materialize the result spine as Str("[|]", [h; t]) via
        // PutList+SetVariable+PutStructure, while A2 / peeled tails are
        // often compact VList — same VList <-> "[|]"/2 class unifyTerms
        // already handles for UnifyValue / =/2. Shallow equality made
        // capp([a],[b],[a,b]) fail in the base-case GetValue while
        // capp([a],[b],X), X=[a,b] (open + =/2) passed (FS-LIST-PARTIAL-TAIL).
        match getReg ai s, getReg xn s with
        | Some a, Some x -> unifyVal a x s
        | _              -> None

    | GetStructure (fn, arity, ai) ->
        // If we''re already inside a build/read context, push it so the
        // parent structure can resume filling once this inner one
        // materializes / is exhausted.  Without this push, nested
        // patterns like `[op(:-,1200,xfx) | Rest]` overwrite the outer
        // BuildList and lose track of where to return.
        match getReg ai s with
        | Some (Str (fn0, args)) when fn0 = fn && List.length args = arity ->
            // arity = 0: no args to read, outer builder (if any) keeps running.
            if arity = 0 then Some { s with WsPC = s.WsPC + 1 }
            else
                let push = pushBuilderIfActive s
                Some { push with WsPC = s.WsPC + 1; WsBuilder = Some (ReadArgs args) }
        // The WAM compiler emits `GetStructure (\"[|]\", 2, ai)` to walk
        // the spine of a partial list (e.g. matching `[H|T]` when `H`
        // is itself the head of a destructured nested list).  When
        // `ai` holds a VList that was built via PutList / GetList, we
        // need to read it as a cons cell.  Without this case, every
        // list-iterating parser predicate (`take_digits`, `take_ident`,
        // `tokenize_loop`, `parse_args`, ...) fails on its first
        // cons-cell match.
        | Some (VList (h :: t)) when fn = \"[|]\" && arity = 2 ->
            let tailVal = if List.isEmpty t then Atom \"[]\" else VList t
            let push = pushBuilderIfActive s
            Some { push with WsPC = s.WsPC + 1; WsBuilder = Some (ReadArgs [h; tailVal]) }
        | Some (Unbound vid) when arity = 0 ->
            // Atom-arity struct write: bind reg to Str(fn, []).  Outer
            // builder''s arg slot, if any, already holds the Unbound var
            // we just bound to the atom -- no append needed, outer
            // continues.
            // Y-aware write via putReg so Yk registers land in the
            // env frame rather than the X-register array.
            let str = Str (fn, [])
            let s0 = putReg ai str s
            Some { s0 with
                     WsPC      = s.WsPC + 1
                     WsBindings= Map.add vid str s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | Some (Unbound _) ->
            let push = pushBuilderIfActive s
            Some { push with WsPC = s.WsPC + 1; WsBuilder = Some (BuildStruct (fn, ai, arity, [])) }
        | _ -> None

    | GetList ai ->
        let push = pushBuilderIfActive s
        match getReg ai s with
        | Some (VList (h :: t)) ->
            let tailVal = if List.isEmpty t then Atom \"[]\" else VList t
            Some { push with WsPC = s.WsPC + 1; WsBuilder = Some (ReadArgs [h; tailVal]) }
        // Cons cell built with a non-ground tail (e.g. `[H | T]` where T
        // is an unbound var) materializes as Str(\"[|]\", [h; t]) so the
        // tail can stay symbolic.  GetList must match that shape too.
        | Some (Str (\"[|]\", [h; t])) ->
            Some { push with WsPC = s.WsPC + 1; WsBuilder = Some (ReadArgs [h; t]) }
        | Some (Unbound _) ->
            Some { push with WsPC = s.WsPC + 1; WsBuilder = Some (BuildList (ai, [])) }
        | _ -> None

    | UnifyVariable xn ->
        match readNextArg s with
        | Some (v, s1) -> Some (putReg xn v { s1 with WsPC = s.WsPC + 1 })
        | None ->
            // Write mode: GetList / GetStructure on an unbound register
            // sets WsBuilder = Some (BuildList _) | Some (BuildStruct _),
            // and the following Unify* instructions must APPEND a fresh
            // var to the structure being constructed.  The R and Python
            // runtimes both do the same (templates/targets/r_wam/runtime.R.mustache
            // around UnifyVariable, src/unifyweaver/targets/wam_python_runtime
            // /WamRuntime.py:1760).  addToBuilder advances PC and
            // materializes the struct/list once arity is reached, so
            // PC is NOT advanced here directly.
            let vid = s.WsVarCounter
            let var = Unbound vid
            let s1 = putReg xn var { s with WsVarCounter = s.WsVarCounter + 1 }
            addToBuilder var s1

    | UnifyValue xn ->
        match readNextArg s with
        | Some (v, s1) ->
            match getReg xn s with
            | Some x -> unifyVal v x s1
            | None   -> None
        | None ->
            // Write mode: append value-of-xn to builder.
            match getReg xn s with
            | Some v -> addToBuilder v s
            | None   -> None

    | UnifyConstant c ->
        match readNextArg s with
        | Some (v, s1) -> unifyVal v c s1
        | None ->
            // Write mode: append constant to builder.
            addToBuilder c s

    | PutConstant (c, ai) ->
        let r = Array.copy s.WsRegs
        r.[ai] <- c
        Some { s with WsPC = s.WsPC + 1; WsRegs = r }

    | PutVariable (xn, ai) ->
        let vid = s.WsVarCounter
        let var = Unbound vid
        let s1  = putReg xn var s
        let r   = Array.copy s1.WsRegs
        r.[ai] <- var
        Some { s1 with
                 WsPC        = s.WsPC + 1
                 WsRegs      = r
                 WsVarCounter= s.WsVarCounter + 1 }

    | PutValue (xn, ai) ->
        match getReg xn s with
        | Some v ->
            let r = Array.copy s.WsRegs
            r.[ai] <- v
            Some { s with WsPC = s.WsPC + 1; WsRegs = r }
        | None   -> None

    | PutStructure (fn, ai, arity) ->
        // PutStructure starts a fresh build context.  If we''re already
        // inside one (nested PutStructure for compound subterms in a
        // body call), push the outer so it can resume.  When
        // addToBuilder materializes the struct, it''ll bind the
        // destination''s previous vid to the new term IF doing so
        // doesn''t create a cycle (i.e. vid doesn''t appear inside the
        // new term).  The cycle check lives in addToBuilder (#2400
        // continuation).
        let push = pushBuilderIfActive s
        Some { push with WsPC      = s.WsPC + 1
                         WsBuilder = Some (BuildStruct (fn, ai, arity, [])) }

    | PutList ai ->
        let push = pushBuilderIfActive s
        Some { push with WsPC      = s.WsPC + 1
                         WsBuilder = Some (BuildList (ai, [])) }

    | SetValue xn ->
        match getReg xn s with
        | Some v -> addToBuilder v s
        | None   -> None

    | SetVariable xn ->
        let vid = s.WsVarCounter
        let var = Unbound vid
        let s1 = putReg xn var { s with WsVarCounter = s.WsVarCounter + 1 }
        addToBuilder var s1

    | SetConstant c -> addToBuilder c s

    //
    // Standard-WAM B0 protocol (issue #2400 follow-up).
    //
    // Call / CallResolved: save the caller''s WsCutBar onto WsB0Stack
    // and set WsCutBar = current WsCPsLen.  This is the count BEFORE
    // the callee''s leading TryMeElse pushes CP_self, so a `!` inside
    // the callee drops above WsCutBar = correctly kills CP_self.
    // Without this, Allocate set WsCutBar = post-TryMeElse count and
    // cut was a no-op against the predicate''s own retry CP.
    //
    // Execute / ExecutePc (tail call): update WsCutBar in place but
    // do NOT push.  The caller''s frame is being replaced — Proceed
    // pops, so a push here would unbalance the stack.  The pairing
    // invariant is: WsB0Stack depth changes only via Call/Proceed
    // pairs.
    //
    // Proceed: pop WsB0Stack into WsCutBar to restore the caller''s
    // barrier.  See the Proceed handler below.
    //
    | CallResolved (pc, _arity) ->
        Some { s with WsPC      = pc
                      WsCP      = s.WsPC + 1
                      WsB0Stack = s.WsCutBar :: s.WsB0Stack
                      WsCutBar  = s.WsCPsLen }

    | CallForeign (pred, _arity) ->
        // The generated binder completes the same return transition as
        // Proceed for both the first result and every FFIStreamRetry result.
        executeForeign ctx pred true { s with WsCP      = s.WsPC + 1
                                              WsB0Stack = s.WsCutBar :: s.WsB0Stack
                                              WsCutBar  = s.WsCPsLen }

    | ExecuteForeign pred ->
        // Foreign tail call: preserve the caller''s continuation and do not
        // push B0. The binder performs the callee''s missing Proceed.
        executeForeign ctx pred true { s with WsCutBar = s.WsCPsLen }

    // catch/3 and throw/1 are emitted by the WAM compiler as
    // Call / Execute (meta-call shape) rather than BuiltinCall.
    // Dispatch them through the BuiltinCall step branch so the ISO
    // substrate runs.  For Call: BuiltinCall returns PC=WsPC+1 which
    // matches the natural advance past the Call site.  For Execute:
    // BuiltinCall returns PC=WsPC+1 (past the Execute itself), then
    // we convert that to PC=WsCP to honor tail-call return semantics.
    // ISO meta-builtins emitted as Call by the WAM compiler.  These
    // are predicate names the shared wam_target.pl does NOT classify
    // as is_builtin_goal, so they normally fall through to label
    // lookup -- which would fail since these aren''t labelled
    // predicates.  Routing through BuiltinCall here picks up the F#
    // step arms for ISO catch/throw/is_iso/is_lax.
    | Call (pred, arity) when isIsoMetaBuiltin pred ->
        let sc = { s with WsCP      = s.WsPC + 1
                          WsB0Stack = s.WsCutBar :: s.WsB0Stack
                          WsCutBar  = s.WsCPsLen }
        step ctx sc (BuiltinCall (pred, arity))

    | Call (pred, _arity) ->
        let sc = { s with WsCP      = s.WsPC + 1
                          WsB0Stack = s.WsCutBar :: s.WsB0Stack
                          WsCutBar  = s.WsCPsLen }
        match Map.tryFind pred ctx.WcLoweredPredicates with
        | Some fn -> fn ctx sc
        | None ->
        match callIndexedFact2 ctx pred sc with
        | Some sr -> Some sr
        | None ->
        match Map.tryFind pred ctx.WcLabels with
        | Some pc -> Some { sc with WsPC = pc }
        | None    -> None

    | Execute pred when isIsoMetaBuiltin pred ->
        // Tail-call meta-builtin: dispatch through BuiltinCall, then
        // convert the PC+1 advance (past the Execute slot) to WsCP so
        // control returns to the outer caller as a normal Proceed
        // would.  Execute itself does not push WsB0Stack, so there''s
        // nothing to pop here.
        let arity = isoMetaBuiltinArity pred
        match step ctx s (BuiltinCall (pred, arity)) with
        | Some sNext -> Some { sNext with WsPC = sNext.WsCP }
        | None -> None

    | Execute pred ->
        // Tail call: no return, so no push.  Update barrier in place.
        let st = { s with WsCutBar = s.WsCPsLen }
        match Map.tryFind pred ctx.WcLoweredPredicates with
        | Some fn -> fn ctx st
        | None ->
        match callIndexedFact2 ctx pred st with
        | Some sr -> Some sr
        | None ->
        match Map.tryFind pred ctx.WcLabels with
        | Some pc -> Some { st with WsPC = pc }
        | None    -> None

    | ExecutePc pc ->
        // Tail call (resolved form): no push, update barrier only.
        Some { s with WsPC     = pc
                      WsCutBar = s.WsCPsLen }

    | Jump label ->
        match Map.tryFind label ctx.WcLabels with
        | Some pc -> Some { s with WsPC = pc }
        | None    -> None

    | JumpPc pc -> Some { s with WsPC = pc }

    | Proceed ->
        // Standard-WAM return: jump to WsCP (caller''s return PC)
        // and restore the caller''s cut barrier from WsB0Stack.
        // Call/CallResolved pushed; Proceed pops.  Tail-call
        // (Execute/ExecutePc) does NOT push, so the pop here matches
        // the most recent non-tail Call up the chain.  When the
        // stack is empty (top-level entry), keep WsCutBar at 0.
        let ret = s.WsCP
        let cutBarR, stackR =
            match s.WsB0Stack with
            | top :: rest -> top, rest
            | []          -> 0,   []
        if ret = 0 then Some { s with WsPC      = 0
                                      WsCutBar  = cutBarR
                                      WsB0Stack = stackR }
        else Some { s with WsPC      = ret
                           WsCP      = 0
                           WsCutBar  = cutBarR
                           WsB0Stack = stackR }

    | Fail -> None

    | Allocate ->
        // Save the previous cut barrier in the frame so Deallocate can
        // restore it for nested intra-clause cuts (e.g. a cut inside
        // a goal that was itself inside another goal''s environment).
        //
        // Issue #2400 follow-up: do NOT overwrite WsCutBar here.  The
        // cross-call cut barrier (B0) is now managed by Call/Proceed
        // via WsB0Stack — Call sets WsCutBar = WsCPsLen BEFORE the
        // callee''s TryMeElse pushes CP_self, which is the value cuts
        // inside the callee body need.  If Allocate then re-set
        // WsCutBar = WsCPsLen (the count AFTER TryMeElse), `:- ..., !.`
        // would never drop the predicate''s own retry CP — exactly the
        // bug that caused tokenize_one''s CPs to leak past the parser''s
        // cut and corrupt later bindings.
        //
        // EfSavedCutBar still snapshots WsCutBar so Deallocate restores
        // the right value when a sub-goal inside this clause had its
        // own Allocate/Deallocate pair (the WsB0Stack handles
        // call-to-call; this frame field handles intra-clause nesting).
        let frame = { EfSavedCP    = s.WsCP
                      EfYRegs      = Map.empty
                      EfSavedCutBar= s.WsCutBar }
        Some { s with
                 WsPC    = s.WsPC + 1
                 WsStack = frame :: s.WsStack }

    | Deallocate ->
        match s.WsStack with
        | ef :: rest ->
            Some { s with
                     WsPC     = s.WsPC + 1
                     WsStack  = rest
                     WsCP     = ef.EfSavedCP
                     WsCutBar = ef.EfSavedCutBar }
        | []         -> None

    | TryMeElse label ->
        let nextPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
        let cp = { CpNextPC   = nextPC
                   CpRegs     = Array.copy s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpB0StackLen = List.length s.WsB0Stack
                   CpAggFrame = None
                   CpBuiltin  = None }
        Some { s with WsPC = s.WsPC + 1; WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    | TryMeElsePc nextPC ->
        let cp = { CpNextPC   = nextPC
                   CpRegs     = Array.copy s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpB0StackLen = List.length s.WsB0Stack
                   CpAggFrame = None
                   CpBuiltin  = None }
        Some { s with WsPC = s.WsPC + 1; WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    // TrustMe / RetryMeElse: when an indexed SwitchOnConstantPc jumps
    // directly to a clause that begins with one of these (the WAM
    // emitter places labels just before Retry/Trust), there''s no
    // choice point to pop or modify — we''re committing deterministically
    // to that clause.  Treat the empty-CP case as a no-op advance
    // instead of failing, so indexed dispatch works (Bug A from PR #2350).
    | TrustMe ->
        match s.WsCPs with
        | _ :: rest -> Some { s with WsPC = s.WsPC + 1; WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
        | []        -> Some { s with WsPC = s.WsPC + 1 }

    | RetryMeElse label ->
        let nextPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
        match s.WsCPs with
        | cp :: rest ->
            Some { s with WsPC = s.WsPC + 1; WsCPs = { cp with CpNextPC = nextPC } :: rest }
        | [] ->
            // Indexing-bypass synthesis: an instruction like
            // SwitchOnTerm jumped directly to this clause without
            // running clause 1''s TryMeElse, so no CP exists yet.
            // Without synthesis the clause becomes deterministic --
            // its failure can''t fall back to the next clause.
            // Symmetric with TrustMe''s "empty CP = no-op advance".
            let cp = { CpNextPC   = nextPC
                       CpRegs     = Array.copy s.WsRegs
                       CpStack    = s.WsStack
                       CpCP       = s.WsCP
                       CpTrailLen = s.WsTrailLen
                       CpHeapLen  = s.WsHeapLen
                       CpBindings = s.WsBindings
                       CpCutBar   = s.WsCutBar
                       CpB0StackLen = List.length s.WsB0Stack
                       CpAggFrame = None
                       CpBuiltin  = None }
            Some { s with WsPC = s.WsPC + 1; WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    | RetryMeElsePc nextPC ->
        match s.WsCPs with
        | cp :: rest ->
            Some { s with WsPC = s.WsPC + 1; WsCPs = { cp with CpNextPC = nextPC } :: rest }
        | [] ->
            let cp = { CpNextPC   = nextPC
                       CpRegs     = Array.copy s.WsRegs
                       CpStack    = s.WsStack
                       CpCP       = s.WsCP
                       CpTrailLen = s.WsTrailLen
                       CpHeapLen  = s.WsHeapLen
                       CpBindings = s.WsBindings
                       CpCutBar   = s.WsCutBar
                       CpB0StackLen = List.length s.WsB0Stack
                       CpAggFrame = None
                       CpBuiltin  = None }
            Some { s with WsPC = s.WsPC + 1; WsCPs = cp :: s.WsCPs; WsCPsLen = s.WsCPsLen + 1 }

    //
    // Indexed-dispatch chain ops (issue #2400).  Used in chains
    // synthesized by wam_target.pl''s build_term_index_with_chains
    // for switch_on_term / switch_on_constant / switch_on_structure
    // targets with >1 matching clauses.  Layout:
    //   L_<P>_<A>_<group>_dispatch:
    //       try   L_<P>_<A>_<I1>_body
    //       retry L_<P>_<A>_<I2>_body
    //       ...
    //       trust L_<P>_<A>_<IN>_body
    // Unlike try_me_else / retry_me_else / trust_me, these CARRY the
    // body label as the jump target and store the in-chain
    // fall-through PC (= next chain instruction) in the CP, so the
    // chain is self-contained and never confuses with outer CPs.
    //
    | TryPc targetPC ->
        let cp = { CpNextPC   = s.WsPC + 1
                   CpRegs     = Array.copy s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpB0StackLen = List.length s.WsB0Stack
                   CpAggFrame = None
                   CpBuiltin  = None }
        Some { s with WsPC = targetPC
                      WsCPs = cp :: s.WsCPs
                      WsCPsLen = s.WsCPsLen + 1 }

    | Try label ->
        let targetPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
        let cp = { CpNextPC   = s.WsPC + 1
                   CpRegs     = Array.copy s.WsRegs
                   CpStack    = s.WsStack
                   CpCP       = s.WsCP
                   CpTrailLen = s.WsTrailLen
                   CpHeapLen  = s.WsHeapLen
                   CpBindings = s.WsBindings
                   CpCutBar   = s.WsCutBar
                   CpB0StackLen = List.length s.WsB0Stack
                   CpAggFrame = None
                   CpBuiltin  = None }
        Some { s with WsPC = targetPC
                      WsCPs = cp :: s.WsCPs
                      WsCPsLen = s.WsCPsLen + 1 }

    | RetryPc targetPC ->
        match s.WsCPs with
        | cp :: rest ->
            Some { s with WsPC = targetPC
                          WsCPs = { cp with CpNextPC = s.WsPC + 1 } :: rest }
        | [] ->
            // Should not happen if the chain is well-formed: a
            // preceding `try` always pushed a CP.  Fall back to
            // synthesizing one with the current state so we don''t
            // silently lose backtrack ability.
            let cp = { CpNextPC   = s.WsPC + 1
                       CpRegs     = Array.copy s.WsRegs
                       CpStack    = s.WsStack
                       CpCP       = s.WsCP
                       CpTrailLen = s.WsTrailLen
                       CpHeapLen  = s.WsHeapLen
                       CpBindings = s.WsBindings
                       CpCutBar   = s.WsCutBar
                       CpB0StackLen = List.length s.WsB0Stack
                       CpAggFrame = None
                       CpBuiltin  = None }
            Some { s with WsPC = targetPC
                          WsCPs = cp :: s.WsCPs
                          WsCPsLen = s.WsCPsLen + 1 }

    | Retry label ->
        let targetPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
        match s.WsCPs with
        | cp :: rest ->
            Some { s with WsPC = targetPC
                          WsCPs = { cp with CpNextPC = s.WsPC + 1 } :: rest }
        | [] ->
            let cp = { CpNextPC   = s.WsPC + 1
                       CpRegs     = Array.copy s.WsRegs
                       CpStack    = s.WsStack
                       CpCP       = s.WsCP
                       CpTrailLen = s.WsTrailLen
                       CpHeapLen  = s.WsHeapLen
                       CpBindings = s.WsBindings
                       CpCutBar   = s.WsCutBar
                       CpB0StackLen = List.length s.WsB0Stack
                       CpAggFrame = None
                       CpBuiltin  = None }
            Some { s with WsPC = targetPC
                          WsCPs = cp :: s.WsCPs
                          WsCPsLen = s.WsCPsLen + 1 }

    | TrustPc targetPC ->
        match s.WsCPs with
        | _ :: rest -> Some { s with WsPC = targetPC
                                     WsCPs = rest
                                     WsCPsLen = s.WsCPsLen - 1 }
        | []        -> Some { s with WsPC = targetPC }

    | Trust label ->
        let targetPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue 0
        match s.WsCPs with
        | _ :: rest -> Some { s with WsPC = targetPC
                                     WsCPs = rest
                                     WsCPsLen = s.WsCPsLen - 1 }
        | []        -> Some { s with WsPC = targetPC }

    // Phase 4.1 / Phase J — parallel WAM. ParTryMeElse dispatches through
    // forkOrSequential, which forks all branches in parallel when inside a
    // forkable aggregate frame (sum/count/bag/set/findall with enough
    // branches), otherwise falls back to the sequential TryMeElse step.
    // The Retry/Trust variants always alias to sequential since they
    // appear inside the chain that the fork enumerates wholesale — only
    // the leading ParTryMeElse triggers the fork decision.
    | ParTryMeElse label    -> forkOrSequential ctx s (Choice1Of2 label)
    | ParTryMeElsePc pc     -> forkOrSequential ctx s (Choice2Of2 pc)
    | ParRetryMeElse label  -> step ctx s (RetryMeElse label)
    | ParTrustMe            -> step ctx s TrustMe
    | ParRetryMeElsePc pc   -> step ctx s (RetryMeElsePc pc)

    // SwitchOnConstantPc / SwitchOnConstant: indexed dispatch on A1.
    // Hits jump to the matching clause; misses fall through to the
    // next instruction (typically TryMeElse) which then runs the linear
    // chain.  Falling through on miss (rather than returning None)
    // mirrors the WAM compiler''s `tom:default` semantics — `default`
    // labels get dropped by resolveCallInstrs, so the table just lacks
    // an entry, and falling through is the right behavior.  Without
    // this, indexed dispatch fails outright on any A1 not explicitly
    // listed in the table.
    | SwitchOnConstantPc table ->
        let valOpt = getReg 1 s
        match valOpt with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Atom key) ->
            match binarySearchStr table key with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Integer n) ->
            match binarySearchStr table (string n) with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> Some { s with WsPC = s.WsPC + 1 }

    | SwitchOnConstant table ->
        let valOpt = getReg 1 s
        match valOpt with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | Some v ->
            match Map.tryFind v table with
            | Some label ->
                match Map.tryFind label ctx.WcLabels with
                | Some pc -> Some { s with WsPC = pc }
                | None    -> Some { s with WsPC = s.WsPC + 1 }
            | None -> Some { s with WsPC = s.WsPC + 1 }
        | None -> Some { s with WsPC = s.WsPC + 1 }

    // SwitchOnTerm: type-based dispatch on A1.  Emitted by the WAM
    // compiler for predicates with mixed first-arg shapes (atoms,
    // compounds, lists, variables).  Falls through on miss so the
    // linear try_me_else chain still runs.  The label form is
    // rewritten to SwitchOnTermPc by resolveCallInstrs at load time.
    | SwitchOnTermPc (constTable, structTable, listPc) ->
        let valOpt = getReg 1 s
        match valOpt with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Atom a) ->
            match binarySearchStr constTable a with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Integer n) ->
            match binarySearchStr constTable (string n) with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Str (fn, args)) ->
            let key = sprintf "%s/%d" fn (List.length args)
            match binarySearchStr structTable key with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> Some { s with WsPC = s.WsPC + 1 }
        // `VList []` IS the empty list — same as Atom "[]" — so it
        // should look up "[]" in constTable, NOT jump to the cons-cell
        // list chain.  Issue #2400 follow-up: without this, parse_op_loop''s
        // base case `parse_op_loop([], ...)` was bypassed for A1=VList []
        // and routed to the [H|T] chain whose head GetList read [] as
        // an empty-list-typed cons (no head) and failed.
        | Some (VList []) ->
            match binarySearchStr constTable "[]" with
            | Some pc -> Some { s with WsPC = pc }
            | None    -> Some { s with WsPC = s.WsPC + 1 }
        | Some (VList _) when listPc > 0 ->
            Some { s with WsPC = listPc }
        | _ -> Some { s with WsPC = s.WsPC + 1 }

    | SwitchOnTerm _ ->
        // Should have been rewritten to SwitchOnTermPc by resolveCallInstrs.
        // If we hit this, fall through deterministically rather than fail.
        Some { s with WsPC = s.WsPC + 1 }

    | BuiltinCall ("!/0", _) ->
        // Cut: discard CPs created since clause entry.  WsCPs is
        // stack-with-head-newest, so we DROP the newest
        // (WsCPsLen - WsCutBar) entries to keep the OLDEST WsCutBar
        // (the CPs that existed when Allocate set WsCutBar).  The
        // prior `List.take WsCutBar` kept the NEWEST WsCutBar -- the
        // exact opposite -- but every previously-tested predicate had
        // either cut_bar = 0 (top level) or kept fewer CPs by
        // happenstance, so the bug stayed hidden until the parser
        // library''s in-clause cuts (every `tokenize_one` arm has
        // `:- !.`) drove it.
        let drop = max 0 (s.WsCPsLen - s.WsCutBar)
        Some { s with
                 WsPC    = s.WsPC + 1
                 WsCPs   = List.skip drop s.WsCPs
                 WsCPsLen= s.WsCutBar }

    | CutIte ->
        match s.WsCPs with
        | _ :: rest -> Some { s with WsPC = s.WsPC + 1; WsCPs = rest; WsCPsLen = s.WsCPsLen - 1 }
        | []        -> Some { s with WsPC = s.WsPC + 1 }

    | BuiltinCall ("nonvar/1", _) ->
        match getReg 1 s with
        | Some (Unbound _) -> None
        | Some _           -> Some { s with WsPC = s.WsPC + 1 }
        | None             -> None

    | BuiltinCall ("var/1", _) ->
        match getReg 1 s with
        | Some (Unbound _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                -> None

    | BuiltinCall ("atom/1", _) ->
        match getReg 1 s with
        | Some (Atom _) -> Some { s with WsPC = s.WsPC + 1 }
        | _             -> None

    | BuiltinCall ("integer/1", _) ->
        match getReg 1 s with
        | Some (Integer _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                -> None

    | BuiltinCall ("number/1", _) ->
        match getReg 1 s with
        | Some (Integer _) | Some (Float _) -> Some { s with WsPC = s.WsPC + 1 }
        | _                                  -> None

    // is/2 and is_lax/2 share the lax body: arithmetic eval that fails
    // silently on bad input.  is_iso/2 below has its own body that throws
    // structured ISO error terms instead.  This split implements the
    // three-form ISO/lax dispatch defined in WAM_CPP_ISO_ERRORS_SPECIFICATION.md
    // and matches the Python/Elixir/C++ shipped reference shape.
    | BuiltinCall ("is/2", _) | BuiltinCall ("is_lax/2", _) ->
        let expr   = s.WsRegs.[2] |> derefVar s.WsBindings
        let result = evalArith s.WsBindings expr
        let lhs    = getReg 1 s
        match lhs, result with
        | Some (Unbound vid), Some r ->
            let v = if float (int r) = r then Integer (int r) else Float r
            let regs = Array.copy s.WsRegs
            regs.[1] <- v
            Some { s with
                     WsPC      = s.WsPC + 1
                     WsRegs    = regs
                     WsBindings= Map.add vid v s.WsBindings
                     WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                     WsTrailLen= s.WsTrailLen + 1 }
        | Some (Integer n), Some r when float n = r -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // is_iso/2 — same eval semantics as is/2, but on bad eval raise a
    // structured ISO error via throwIsoError.  Failure classification:
    //   - unbound var anywhere in RHS -> instantiation_error
    //   - integer or float zero divide  -> evaluation_error(zero_divisor)
    //   - non-evaluable atom / unknown functor -> type_error(evaluable, X/N)
    // The actual unwind is via WamException, which propagates up to the
    // nearest catch/3 try/with (set up by the catch/3 builtin).
    | BuiltinCall ("is_iso/2", _) ->
        let expr = s.WsRegs.[2] |> derefVar s.WsBindings
        // Step 1: instantiation check (helper in fsharp_wam_bindings).
        if hasUnboundDeep s.WsBindings expr then
            throwIsoError s (makeInstantiationError ())
        // Step 2: zero-divide check.  Parens around the inner match
        // are mandatory in F# so subsequent `| Str ...` arms attach to
        // the outer match, not the inner.
        let rec hasZeroDivide v =
            match derefVar s.WsBindings v with
            | Str ((\"/\" | \"//\" | \"mod\" | \"rem\"), [_; rhs]) ->
                (match derefVar s.WsBindings rhs with
                 | Integer 0 -> true
                 | Float f when f = 0.0 -> true
                 | _ -> false)
            | Str (_, args) -> List.exists hasZeroDivide args
            | _ -> false
        if hasZeroDivide expr then
            throwIsoError s (makeEvaluationError \"zero_divisor\")
        // Step 3: evaluate, classify any remaining failure as type_error.
        let result = evalArith s.WsBindings expr
        match result with
        | None ->
            // Build culprit term.  For an Atom ''foo'' use foo/0; for a
            // Str f(args) use f/arity.  Anything else uses unknown/0.
            let culprit =
                match derefVar s.WsBindings expr with
                | Atom name      -> makePredIndicator name 0
                | Str (fn, args) -> makePredIndicator fn (List.length args)
                | _              -> makePredIndicator \"unknown\" 0
            throwIsoError s (makeTypeError \"evaluable\" culprit)
        | Some r ->
            let lhs = getReg 1 s
            match lhs with
            | Some (Unbound vid) ->
                let v = if float (int r) = r then Integer (int r) else Float r
                let regs = Array.copy s.WsRegs
                regs.[1] <- v
                Some { s with
                         WsPC      = s.WsPC + 1
                         WsRegs    = regs
                         WsBindings= Map.add vid v s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
            | Some (Integer n) when float n = r -> Some { s with WsPC = s.WsPC + 1 }
            | _ -> None

    // ISO arithmetic-compare variants.  Each iso_arithCompare branch
    // classifies a failed eval as either instantiation_error (any
    // side has unbound) or type_error(evaluable, X/N).  Lax variants
    // share the existing op/2 body via the alias match below.
    | BuiltinCall (\"<_iso/2\", _) | BuiltinCall (\">_iso/2\", _)
    | BuiltinCall (\">=_iso/2\", _) | BuiltinCall (\"=<_iso/2\", _)
    | BuiltinCall (\"=:=_iso/2\", _) | BuiltinCall (\"=\\\\=_iso/2\", _) ->
        let op = match instr with
                 | BuiltinCall (k, _) -> k
                 | _ -> \"<_iso/2\"
        let a1 = getReg 1 s
        let a2 = getReg 2 s
        // ISO instantiation check fires on either side.
        let unboundA = match a1 with Some v -> hasUnboundDeep s.WsBindings v | None -> true
        let unboundB = match a2 with Some v -> hasUnboundDeep s.WsBindings v | None -> true
        if unboundA || unboundB then
            throwIsoError s (makeInstantiationError ())
        let r1 = a1 |> Option.bind (evalArith s.WsBindings)
        let r2 = a2 |> Option.bind (evalArith s.WsBindings)
        match r1, r2 with
        | None, _ ->
            let culprit = match a1 with Some v -> arithCulprit s.WsBindings v | None -> makePredIndicator \"unknown\" 0
            throwIsoError s (makeTypeError \"evaluable\" culprit)
        | _, None ->
            let culprit = match a2 with Some v -> arithCulprit s.WsBindings v | None -> makePredIndicator \"unknown\" 0
            throwIsoError s (makeTypeError \"evaluable\" culprit)
        | Some a, Some b ->
            let ok =
                match op with
                | \"<_iso/2\"   -> a < b
                | \">_iso/2\"   -> a > b
                | \">=_iso/2\"  -> a >= b
                | \"=<_iso/2\"  -> a <= b
                | \"=:=_iso/2\" -> abs (a - b) < System.Double.Epsilon
                | \"=\\\\=_iso/2\" -> abs (a - b) >= System.Double.Epsilon
                | _ -> false
            if ok then Some { s with WsPC = s.WsPC + 1 } else None

    | BuiltinCall ("length/2", _) ->
        match flattenList s s.WsRegs.[1] with
        | Some items ->
            let len = List.length items
            match getReg 2 s with
            | Some (Unbound vid) ->
                let v = Integer len
                let regs = Array.copy s.WsRegs
                regs.[2] <- v
                Some { s with
                         WsPC      = s.WsPC + 1
                         WsRegs    = regs
                         WsBindings= Map.add vid v s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
            | Some (Integer n) when n = len -> Some { s with WsPC = s.WsPC + 1 }
            | _ -> None
        | _ -> None

    // Lax arithmetic comparisons: silent failure on bad eval.  Each
    // op/2 also matches op_lax/2 so user code can write the explicit
    // form to bypass an ISO-mode rewrite.  See is_iso/is_lax above
    // for the three-form contract.
    | BuiltinCall ("</2", _) | BuiltinCall (\"<_lax/2\", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a < b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall (">/2", _) | BuiltinCall (\">_lax/2\", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a > b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // >=/2, =</2, =:=/2, =\\=/2 — arithmetic comparisons (parity with Rust /
    // C++ comparison builtins). Each evaluates both sides via evalArith and
    // dispatches on the operator. =:=/2 and =\\=/2 use EPSILON tolerance to
    // bridge integer / float comparisons (mirrors the Rust implementation).
    | BuiltinCall (">=/2", _) | BuiltinCall (\">=_lax/2\", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a >= b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("=</2", _) | BuiltinCall (\"=<_lax/2\", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when a <= b -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("=:=/2", _) | BuiltinCall (\"=:=_lax/2\", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when abs (a - b) < System.Double.Epsilon ->
            Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("=\\\\=/2", _) | BuiltinCall (\"=\\\\=_lax/2\", _) ->
        let v1 = getReg 1 s |> Option.bind (evalArith s.WsBindings)
        let v2 = getReg 2 s |> Option.bind (evalArith s.WsBindings)
        match v1, v2 with
        | Some a, Some b when abs (a - b) >= System.Double.Epsilon ->
            Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // succ/2 + succ_lax/2 - bidirectional successor.  Lax semantics:
    // X bound non-negative integer -> Y = X+1; Y bound positive
    // integer -> X = Y-1; both unbound, or X negative, or Y zero/
    // negative -> silent fail.  Mirrors Python _execute_succ_lax.
    | BuiltinCall (\"succ/2\", _) | BuiltinCall (\"succ_lax/2\", _) ->
        let a = getReg 1 s
        let b = getReg 2 s
        let bindReg reg value =
            match getReg reg s with
            | Some (Unbound vid) ->
                let regs = Array.copy s.WsRegs
                regs.[reg] <- value
                Some { s with
                         WsPC      = s.WsPC + 1
                         WsRegs    = regs
                         WsBindings= Map.add vid value s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
            | Some bound when bound = value -> Some { s with WsPC = s.WsPC + 1 }
            | _ -> None
        match a, b with
        | Some (Integer n), _ when n >= 0 -> bindReg 2 (Integer (n + 1))
        | _, Some (Integer m) when m > 0  -> bindReg 1 (Integer (m - 1))
        | _ -> None

    // succ_iso/2 - bidirectional successor with ISO error throws.
    // Mirrors Python _execute_succ_iso and Aït-Kaci spec §6:
    //   both unbound          -> instantiation_error
    //   X not integer         -> type_error(integer, X)
    //   Y not integer         -> type_error(integer, Y)
    //   X negative            -> type_error(not_less_than_zero, X)
    //   Y zero or negative    -> domain_error(not_less_than_zero, Y)
    | BuiltinCall (\"succ_iso/2\", _) ->
        let a = getReg 1 s
        let b = getReg 2 s
        let aUnbound = match a with Some (Unbound _) -> true | None -> true | _ -> false
        let bUnbound = match b with Some (Unbound _) -> true | None -> true | _ -> false
        if aUnbound && bUnbound then
            throwIsoError s (makeInstantiationError ())
        // Type check non-unbound args.
        (match a with
         | Some v when not aUnbound ->
            (match v with
             | Integer _ -> ()
             | _ -> throwIsoError s (makeTypeError \"integer\" v))
         | _ -> ())
        (match b with
         | Some v when not bUnbound ->
            (match v with
             | Integer _ -> ()
             | _ -> throwIsoError s (makeTypeError \"integer\" v))
         | _ -> ())
        // Negativity / domain checks.
        match a, b with
        | Some (Integer n), _ when n < 0 ->
            throwIsoError s (makeTypeError \"not_less_than_zero\" (Integer n))
        | _, Some (Integer m) when m <= 0 ->
            throwIsoError s (makeDomainError \"not_less_than_zero\" (Integer m))
        | Some (Integer n), _ ->
            let v = Integer (n + 1)
            match getReg 2 s with
            | Some (Unbound vid) ->
                let regs = Array.copy s.WsRegs
                regs.[2] <- v
                Some { s with
                         WsPC      = s.WsPC + 1
                         WsRegs    = regs
                         WsBindings= Map.add vid v s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
            | Some bound when bound = v -> Some { s with WsPC = s.WsPC + 1 }
            | _ -> None
        | _, Some (Integer m) ->
            let v = Integer (m - 1)
            match getReg 1 s with
            | Some (Unbound vid) ->
                let regs = Array.copy s.WsRegs
                regs.[1] <- v
                Some { s with
                         WsPC      = s.WsPC + 1
                         WsRegs    = regs
                         WsBindings= Map.add vid v s.WsBindings
                         WsTrail   = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                         WsTrailLen= s.WsTrailLen + 1 }
            | Some bound when bound = v -> Some { s with WsPC = s.WsPC + 1 }
            | _ -> None
        | _ -> None

    // ==/2, \\==/2 — structural term equality (no unification, no binding).
    // Both sides are dereferenced through the binding chain, then compared
    // by F# structural equality on the Value DU. Mirrors the Rust ==/2 +
    // C++ ==/2 / \\==/2 step cases.
    | BuiltinCall ("==/2", _) ->
        // Standard-order term equality.  Uses termEqual which derefs
        // and normalises VList <-> Str ''[|]'' encoding so e.g. an
        // append-built list compares equal to the literal that
        // appears in source.  Raw F# equality would miss this.
        match getReg 1 s, getReg 2 s with
        | Some a, Some b when termEqual a b s -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("\\\\==/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b when not (termEqual a b s) -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // true/0 / fail/0 — trivial control. Mirrors the Rust execute_control
    // dispatch (true succeeds and advances, fail short-circuits).
    | BuiltinCall ("true/0", _) -> Some { s with WsPC = s.WsPC + 1 }
    | BuiltinCall ("fail/0", _) -> None

    // compound/1 — type check matching Str (any) and non-empty VList.
    // F# VList is a flat list (not cons-cells), so a non-empty VList is
    // exactly the compound case.
    | BuiltinCall ("compound/1", _) ->
        match getReg 1 s with
        | Some (Str _)        -> Some { s with WsPC = s.WsPC + 1 }
        | Some (VList (_::_)) -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // float/1 — type check for Float values.
    | BuiltinCall ("float/1", _) ->
        match getReg 1 s with
        | Some (Float _) -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // is_list/1 — type check for proper lists. Since F# VList is a flat
    // Value list (head/tail items, not cons cells), VList _ already means
    // a proper list (empty or non-empty). Atom "[]" also counts as the
    // empty list per Prolog convention.
    | BuiltinCall ("is_list/1", _) ->
        match getReg 1 s with
        | Some (VList _)      -> Some { s with WsPC = s.WsPC + 1 }
        | Some (Atom "[]")    -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // write/1 / display/1 — print the dereferenced value to stdout.
    // Mirrors the Rust write/1 + display/1 fast-path (both use Display
    // for now; standard Prolog differentiates quoting between them).
    | BuiltinCall ("write/1", _) | BuiltinCall ("display/1", _) ->
        match getReg 1 s with
        | Some v ->
            printf "%s" (sprintf "%A" v)
            Some { s with WsPC = s.WsPC + 1 }
        | None -> None

    // nl/0 — newline to stdout.
    | BuiltinCall ("nl/0", _) ->
        printfn ""
        Some { s with WsPC = s.WsPC + 1 }

    // ========================================================================
    // Phase G — atom / string builtins (parity with Go target)
    // ========================================================================
    // F# Atom is string-based, so most of these collapse to direct .NET
    // string operations. The list-of-codes / list-of-chars conversions
    // use the F# Value DU shape: VList of Value list (a flat list).
    //
    // Reference: templates/targets/go_wam/state.go.mustache case arms
    // for atom_concat/3, atom_length/2, char_code/2, atom_codes/2,
    // atom_string/2, upcase_atom/2, downcase_atom/2, atom_number/2,
    // succ/2.

    | BuiltinCall ("atom_concat/3", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Atom a), Some (Atom b) ->
            match bindOutput 3 (Atom (a + b)) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("atom_length/2", _) | BuiltinCall ("string_length/2", _) ->
        match getReg 1 s with
        | Some (Atom a) ->
            match bindOutput 2 (Integer a.Length) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // char_code/2 — single-char atom <-> Unicode code point. Bidirectional.
    | BuiltinCall ("char_code/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Atom a), _ when a.Length = 1 ->
            match bindOutput 2 (Integer (int a.[0])) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | Some (Unbound _), Some (Integer c) when c >= 0 && c <= 65535 ->
            match bindOutput 1 (Atom (string (char c))) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // atom_codes/2 — atom <-> list of Integer code points. Bidirectional.
    | BuiltinCall ("atom_codes/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Atom a), _ ->
            let codes = a |> Seq.map (fun c -> Integer (int c)) |> List.ofSeq
            match bindOutput 2 (VList codes) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | Some (Unbound _), Some listVal ->
            // Accept both VList (proper-list shape) and Str(\"[|]\",_)
            // cons-cell chains; the parser library produces the latter
            // via take_ident / take_digits / take_syms accumulators.
            match valueToProperList s.WsBindings listVal with
            | None       -> None
            | Some items ->
                let folder acc v =
                    match acc, derefVar s.WsBindings v with
                    | Some (sb: System.Text.StringBuilder), Integer c when c >= 0 && c <= 65535 ->
                        Some (sb.Append(char c))
                    | _ -> None
                match List.fold folder (Some (System.Text.StringBuilder())) items with
                | Some sb ->
                    match bindOutput 1 (Atom (sb.ToString())) s with
                    | None    -> None
                    | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
                | None -> None
        | _ -> None

    // number_codes/2 — number <-> list of decimal code points.
    // Bidirectional like atom_codes/2.  Without this, the parser
    // library''s `codes_to_number(Cs, N) :- number_codes(N, Cs).`
    // failed silently, so tokenize/2 on \"42\" never produced a
    // tk_num token and every numeric / compound / list input
    // bottomed out as None.  Accept both VList and Str(\"[|]\",_)
    // shapes for the code list (the parser builds the latter
    // via take_digits accumulators).
    | BuiltinCall (\"number_codes/2\", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Integer n), _ ->
            let codes =
                (string n) |> Seq.map (fun c -> Integer (int c)) |> List.ofSeq
            match bindOutput 2 (VList codes) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | Some (Float f), _ ->
            let codes =
                (sprintf \"%g\" f) |> Seq.map (fun c -> Integer (int c)) |> List.ofSeq
            match bindOutput 2 (VList codes) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | Some (Unbound _), Some listVal ->
            match valueToProperList s.WsBindings listVal with
            | None       -> None
            | Some items ->
                let folder acc v =
                    match acc, derefVar s.WsBindings v with
                    | Some (sb: System.Text.StringBuilder), Integer c when c >= 0 && c <= 65535 ->
                        Some (sb.Append(char c))
                    | _ -> None
                match List.fold folder (Some (System.Text.StringBuilder())) items with
                | Some sb ->
                    let str = sb.ToString()
                    // Prefer Integer (matches Prolog''s number_codes
                    // forward semantics for integral input); fall back
                    // to Float when there''s a decimal point or
                    // scientific notation.  Bail out on garbage so
                    // number_codes(N, \"abc\") fails rather than
                    // binding N to NaN.
                    let parsed =
                        match System.Int64.TryParse(str) with
                        | true, n -> Some (Integer (int n))
                        | false, _ ->
                            match System.Double.TryParse(
                                    str,
                                    System.Globalization.NumberStyles.Float,
                                    System.Globalization.CultureInfo.InvariantCulture) with
                            | true, f -> Some (Float f)
                            | false, _ -> None
                    match parsed with
                    | Some numVal ->
                        match bindOutput 1 numVal s with
                        | None    -> None
                        | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
                    | None -> None
                | None -> None
        | _ -> None

    // atom_chars/2 — atom <-> list of single-char atoms. Bidirectional.
    | BuiltinCall ("atom_chars/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Atom a), _ ->
            let chars = a |> Seq.map (fun c -> Atom (string c)) |> List.ofSeq
            match bindOutput 2 (VList chars) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | Some (Unbound _), Some (VList items) ->
            let folder acc v =
                match acc, derefVar s.WsBindings v with
                | Some (sb: System.Text.StringBuilder), Atom c when c.Length = 1 ->
                    Some (sb.Append(c))
                | _ -> None
            match List.fold folder (Some (System.Text.StringBuilder())) items with
            | Some sb ->
                match bindOutput 1 (Atom (sb.ToString())) s with
                | None    -> None
                | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
            | None -> None
        | _ -> None

    // atom_string/2 — atom <-> "string". F# Atom is already string-based,
    // so this is essentially an identity reinterpretation. Mirrors the
    // Go target''s atom_string/2 + string_to_atom/2 unified case (where
    // both directions succeed when either side is an Atom).
    | BuiltinCall ("atom_string/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Atom a), _ ->
            match bindOutput 2 (Atom a) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _, Some (Atom a) ->
            match bindOutput 1 (Atom a) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // string_to_atom/2 — String in A1, Atom in A2 (reversed arg order
    // relative to atom_string/2 per SWI convention).
    | BuiltinCall ("string_to_atom/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Atom a), _ ->
            match bindOutput 2 (Atom a) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _, Some (Atom a) ->
            match bindOutput 1 (Atom a) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // upcase_atom/2, downcase_atom/2 — Unicode-aware ASCII-safe case
    // conversion using .NET''s invariant-culture transforms.
    | BuiltinCall ("upcase_atom/2", _) ->
        match getReg 1 s with
        | Some (Atom a) ->
            match bindOutput 2 (Atom (a.ToUpperInvariant())) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("downcase_atom/2", _) ->
        match getReg 1 s with
        | Some (Atom a) ->
            match bindOutput 2 (Atom (a.ToLowerInvariant())) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // atom_number/2 — atom <-> Integer / Float. Tries integer parse first,
    // then float parse (invariant culture for the float form). Reverse
    // direction emits the canonical %d (Integer) or %g (Float) format.
    | BuiltinCall ("atom_number/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Atom a), _ ->
            let asInt = System.Int64.TryParse(a)
            match asInt with
            | true, n ->
                match bindOutput 2 (Integer (int n)) s with
                | None    -> None
                | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
            | _ ->
                let asFloat = System.Double.TryParse(a,
                                System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture)
                match asFloat with
                | true, f ->
                    match bindOutput 2 (Float f) s with
                    | None    -> None
                    | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
                | _ -> None
        | _, Some (Integer n) ->
            match bindOutput 1 (Atom (string n)) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _, Some (Float f) ->
            let txt = f.ToString("G", System.Globalization.CultureInfo.InvariantCulture)
            match bindOutput 1 (Atom txt) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // succ/2 — Integer successor. succ(X, Y) iff Y = X + 1 and both are
    // non-negative. Bidirectional. Mirrors the Go target''s succ/2.
    | BuiltinCall ("succ/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Integer x), _ when x >= 0 ->
            match bindOutput 2 (Integer (x + 1)) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _, Some (Integer y) when y > 0 ->
            match bindOutput 1 (Integer (y - 1)) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // ========================================================================
    // Phase H — list / sort / order / unification builtins (Go parity)
    // ========================================================================
    // Reference: templates/targets/go_wam/state.go.mustache.  These all
    // operate on F#''s VList shape (a flat Value list).  Standard-order
    // comparisons use the runtime compareValue helper (defined in
    // WamTypes.fs alongside copyTermWalk and friends) rather than the
    // Value type''s CustomComparison override, which has a recursive
    // CompareTo implementation that is not safe for the hot path.

    // append/3 — concat two ground lists. Reverse / partial modes are
    // intentionally not supported here (parity with Go''s listToSlice +
    // slice concat: both A1 and A2 must be VList).
    | BuiltinCall ("append/3", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b ->
            match flattenList s a, flattenList s b with
            | Some xs, Some ys ->
                match bindOutput 3 (VList (xs @ ys)) s with
                | None    -> None
                | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
            | _ -> None
        | _ -> None

    // reverse/2 — list reverse. Bidirectional: whichever side is bound
    // becomes the source, the other becomes the output.
    | BuiltinCall ("reverse/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, _ when (flattenList s a).IsSome ->
            let items = (flattenList s a).Value
            match bindOutput 2 (VList (List.rev items)) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _, Some b when (flattenList s b).IsSome ->
            let items = (flattenList s b).Value
            match bindOutput 1 (VList (List.rev items)) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // last/2 — extract the last element of a non-empty list.
    | BuiltinCall ("last/2", _) ->
        match flattenList s s.WsRegs.[1] with
        | Some (items) when not (List.isEmpty items) ->
            match bindOutput 2 (List.last items) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // nth0/3, nth1/3 — index access. nth0 is 0-based, nth1 is 1-based.
    | BuiltinCall ("nth0/3", _) | BuiltinCall ("nth1/3", _) ->
        let base_ =
            match instr with
            | BuiltinCall ("nth1/3", _) -> 1
            | _ -> 0
        match getReg 1 s, flattenList s s.WsRegs.[2] with
        | Some (Integer i), Some items ->
            let idx = i - base_
            if idx >= 0 && idx < List.length items then
                match bindOutput 3 (List.item idx items) s with
                | None    -> None
                | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
            else None
        | _ -> None

    // memberchk/2 — deterministic membership check: succeed once on the
    // first match, no choice points. Mirrors the Go memberchk/2 fast
    // path that just walks the list and returns true on first Unify.
    | BuiltinCall ("memberchk/2", _) ->
        match getReg 1 s, flattenList s s.WsRegs.[2] with
        | Some elem_, Some items ->
            if items |> List.exists (fun v -> derefVar s.WsBindings v = elem_) then
                Some { s with WsPC = s.WsPC + 1 }
            else None
        | _ -> None

    // delete/3 — remove ALL occurrences of A2 from list A1, produce A3.
    | BuiltinCall ("delete/3", _) ->
        match flattenList s s.WsRegs.[1], getReg 2 s with
        | Some items, Some target ->
            let kept = items |> List.filter (fun v -> derefVar s.WsBindings v <> target)
            match bindOutput 3 (VList kept) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // select/3 — choose one element to remove with full backtracking
    // semantics (parity with the Go target''s SelectResults ChoicePoint
    // enumeration).  Computes all (selected, rest) splits upfront, then
    // attempts unifyVal on each in turn.  The first successful split
    // commits; if more remain, a SelectRetry CP is pushed so backtrack
    // resumes through resumeBuiltin (SelectRetry arm above) and tries
    // the next candidate.
    | BuiltinCall ("select/3", _) ->
        match getReg 1 s, getReg 2 s with
        | Some elem_, Some (VList items) ->
            let rec splits prefix rest =
                match rest with
                | [] -> []
                | x :: xs ->
                    (derefVar s.WsBindings x, List.rev prefix @ xs)
                    :: splits (x :: prefix) xs
            let allSplits = splits [] items
            let retPC = s.WsPC + 1
            let rec tryNext pairs =
                match pairs with
                | [] -> None
                | (sel, restList) :: more ->
                    match unifyVal elem_ sel s with
                    | None    -> tryNext more
                    | Some s2 ->
                        match bindOutput 3 (VList restList) s2 with
                        | None    -> tryNext more
                        | Some s3 ->
                            let newCPs, newCPsLen =
                                if List.isEmpty more then
                                    s.WsCPs, s.WsCPsLen
                                else
                                    let cp = { CpNextPC   = retPC
                                               CpRegs     = Array.copy s.WsRegs
                                               CpStack    = s.WsStack
                                               CpCP       = s.WsCP
                                               CpTrailLen = s.WsTrailLen
                                               CpHeapLen  = s.WsHeapLen
                                               CpBindings = s.WsBindings
                                               CpCutBar   = s.WsCutBar
                                               CpB0StackLen = List.length s.WsB0Stack
                                               CpAggFrame = None
                                               CpBuiltin  = Some (SelectRetry (1, 3, more, retPC)) }
                                    cp :: s.WsCPs, s.WsCPsLen + 1
                            Some { s3 with
                                     WsPC      = retPC
                                     WsCPs     = newCPs
                                     WsCPsLen  = newCPsLen }
            tryNext allSplits
        | _ -> None

    // numlist/3 — generate the inclusive integer range [Lo..Hi] into A3.
    | BuiltinCall ("numlist/3", _) ->
        match getReg 1 s, getReg 2 s with
        | Some (Integer lo), Some (Integer hi) when lo <= hi ->
            let items = [ for n in lo .. hi -> Integer n ]
            match bindOutput 3 (VList items) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // sort/2 — standard-order sort with dedup. msort/2 — same sort
    // without dedup. Both use compareValue (the runtime helper) for the
    // sort key.
    | BuiltinCall ("sort/2", _) ->
        match flattenList s s.WsRegs.[1] with
        | Some items ->
            let deref_ = items |> List.map (derefVar s.WsBindings)
            let sorted = deref_ |> List.sortWith compareValue
            let rec dedup xs =
                match xs with
                | []                              -> []
                | [x]                             -> [x]
                | x :: y :: rest when compareValue x y = 0 -> dedup (x :: rest)
                | x :: rest                       -> x :: dedup rest
            match bindOutput 2 (VList (dedup sorted)) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("msort/2", _) ->
        match flattenList s s.WsRegs.[1] with
        | Some items ->
            let sorted = items |> List.map (derefVar s.WsBindings) |> List.sortWith compareValue
            match bindOutput 2 (VList sorted) s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // compare/3 — three-way compare: bind A1 to ''<'', ''='', or ''>'' based
    // on the standard-order comparison of A2 vs A3.
    | BuiltinCall ("compare/3", _) ->
        match getReg 2 s, getReg 3 s with
        | Some a, Some b ->
            let c = compareValue a b
            let order = if c < 0 then Atom "<" elif c > 0 then Atom ">" else Atom "="
            match bindOutput 1 order s with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // @</2, @=</2, @>/2, @>=/2 — standard-order comparison predicates.
    | BuiltinCall ("@</2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b when compareValue a b < 0 -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("@=</2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b when compareValue a b <= 0 -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("@>/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b when compareValue a b > 0 -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("@>=/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b when compareValue a b >= 0 -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // =/2 — explicit unification. Delegates to the local unifyVal helper
    // that lives at the end of the step function (binding-trail aware).
    | BuiltinCall ("=/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b -> unifyVal a b s
        | _ -> None

    // \\=/2 — non-unifiable check. Succeeds when A1 and A2 cannot unify
    // under the current binding state.  Implemented by attempting
    // unification on a copy of the state and inverting the result; if
    // the trial unifies, \\=/2 fails (bindings are discarded since we
    // never thread the trial state out).
    | BuiltinCall ("\\\\=/2", _) ->
        match getReg 1 s, getReg 2 s with
        | Some a, Some b ->
            match unifyVal a b s with
            | Some _ -> None
            | None   -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | BuiltinCall ("member/2", _) ->
        // member(Elem, List): non-deterministic — unifies Elem with the
        // first matching element of List and pushes a MemberRetry CP so
        // backtracking can try subsequent matches.  The parser uses
        // `member(op(Name, P, T), OpTable), is_op_type(T), !` and depends
        // on the backtracking when the first match fails the type guard
        // (e.g. resolve_prefix on '':-'' must skip the xfx 1200 entry to
        // find the fx 1200 entry).  Walks both list encodings (VList and
        // Str ''[|]'' cons cells) so it works against either runtime
        // representation.
        let elem_ = s.WsRegs.[1] |> derefVar s.WsBindings
        let retPC = s.WsPC + 1
        let rec flatten lst =
            match derefVar s.WsBindings lst with
            | VList xs            -> xs
            | Atom "[]"           -> []
            | Str ("[|]", [h; t]) -> h :: flatten t
            | _                   -> []
        let items = flatten s.WsRegs.[2]
        let rec tryFrom cands =
            match cands with
            | [] -> None
            | x :: rest ->
                let xd = derefVar s.WsBindings x
                match unifyTerms elem_ xd s with
                | None     -> tryFrom rest
                | Some st1 ->
                    let newCPs, newCPsLen =
                        match rest with
                        | [] -> s.WsCPs, s.WsCPsLen
                        | _  ->
                            let cp = { CpNextPC   = retPC
                                       CpRegs     = Array.copy s.WsRegs
                                       CpStack    = s.WsStack
                                       CpCP       = s.WsCP
                                       CpTrailLen = s.WsTrailLen
                                       CpHeapLen  = s.WsHeapLen
                                       CpBindings = s.WsBindings
                                       CpCutBar   = s.WsCutBar
                                       CpB0StackLen = List.length s.WsB0Stack
                                       CpAggFrame = None
                                       CpBuiltin  = Some (MemberRetry (1, rest, retPC)) }
                            cp :: s.WsCPs, s.WsCPsLen + 1
                    Some { st1 with
                             WsPC     = retPC
                             WsCPs    = newCPs
                             WsCPsLen = newCPsLen }
        tryFrom items

    | BeginAggregate (aggType, valReg, resReg) ->
        // Ensure the result register holds a fresh Unbound var BEFORE
        // snapshotting -- without this, finalizeAggregate''s getReg on
        // an unset Y-reg returns None and the aggregated result is
        // silently dropped.  The compiler doesn''t always emit an
        // explicit PutVariable for the result destination (it''s often
        // a Y-reg used only after EndAggregate via PutValue), so the
        // runtime has to seed it.
        let s0 =
            match getReg resReg s with
            | Some _ -> s
            | None ->
                // Fresh Unbound var that finalizeAggregate can later
                // bind to the aggregated result.  No bindings-map entry
                // here -- a self-binding (vid -> Unbound vid) would loop
                // derefVar; absence in the map is the truly-unbound
                // representation.
                let vid   = s.WsVarCounter
                let fresh = Unbound vid
                putReg resReg fresh { s with WsVarCounter = s.WsVarCounter + 1 }
        let cp = { CpNextPC   = s0.WsPC
                   CpRegs     = Array.copy s0.WsRegs
                   CpStack    = s0.WsStack
                   CpCP       = s0.WsCP
                   CpTrailLen = s0.WsTrailLen
                   CpHeapLen  = s0.WsHeapLen
                   CpBindings = s0.WsBindings
                   CpCutBar   = s0.WsCutBar
                   CpB0StackLen = List.length s0.WsB0Stack
                   CpAggFrame = Some { AggType          = aggType
                                       AggValReg        = valReg
                                       AggResReg        = resReg
                                       AggReturnPC      = 0
                                       AggMergeStrategy = inferMergeStrategy aggType }
                   CpBuiltin  = None }
        Some { s0 with
                 WsPC       = s0.WsPC + 1
                 WsCPs      = cp :: s0.WsCPs
                 WsCPsLen   = s0.WsCPsLen + 1
                 WsAggAccum = [] }

    | EndAggregate valReg ->
        let v = getReg valReg s |> Option.defaultValue (Integer 0) |> derefVar s.WsBindings
        let returnPC  = s.WsPC + 1
        let updatedCPs= updateNearestAggFrame returnPC s.WsCPs
        let s1 = { s with WsAggAccum = v :: s.WsAggAccum; WsCPs = updatedCPs }
        match backtrackInner returnPC s1 with
        | Some s2 -> Some s2
        | None    -> finalizeAggregate returnPC s1

    | BuiltinCall ("functor/3", _) ->
        let t = getReg 1 s
        match t with
        | Some (Unbound vid) ->
            let nArg = getReg 2 s
            let aArg = getReg 3 s
            match nArg, aArg with
            | Some nameVal, Some (Integer arity) when arity >= 0 ->
                let mBuilt =
                    if arity = 0 then Some (nameVal, s.WsVarCounter)
                    else match nameVal with
                         | Atom fname ->
                             let c0    = s.WsVarCounter
                             let args  = [ for i in 0 .. arity - 1 -> Unbound (c0 + i) ]
                             Some (Str (fname, args), c0 + arity)
                         | _ -> None
                match mBuilt with
                | None -> None
                | Some (built, newCtr) ->
                    let regs = Array.copy s.WsRegs
                    regs.[1] <- built
                    Some { s with
                             WsPC        = s.WsPC + 1
                             WsRegs      = regs
                             WsBindings  = Map.add vid built s.WsBindings
                             WsTrail     = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                             WsTrailLen  = s.WsTrailLen + 1
                             WsVarCounter= newCtr }
            | _ -> None
        | Some tVal ->
            let mInfo = match tVal with
                        | Str (fn, args)  -> Some (Atom fn, List.length args)
                        | VList []        -> Some (Atom "[]", 0)
                        | VList _         -> Some (Atom ".", 2)
                        | Atom _          -> Some (tVal, 0)
                        | Integer _       -> Some (tVal, 0)
                        | Float _         -> Some (tVal, 0)
                        | _               -> None
            match mInfo with
            | None -> None
            | Some (name, arity) ->
                match bindOutput 2 name s with
                | None   -> None
                | Some s1 ->
                    match bindOutput 3 (Integer arity) s1 with
                    | None    -> None
                    | Some s2 -> Some { s2 with WsPC = s2.WsPC + 1 }
        | None -> None

    | BuiltinCall ("copy_term/2", _) ->
        match getReg 1 s with
        | Some tVal ->
            let copy, newCtr, _ = copyTermWalk s.WsVarCounter Map.empty tVal
            let s0 = { s with WsVarCounter = newCtr }
            match bindOutput 2 copy s0 with
            | None    -> None
            | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | None -> None

    // arg/3: A1 = N (integer, 1-based), A2 = T (compound/list),
    // A3 = output unified with the selected argument. Mirrors the
    // Haskell/Rust/C++ baseline (see wam_haskell_target.pl arg/3 step case).
    | BuiltinCall ("arg/3", _) ->
        let n = getReg 1 s
        let t = getReg 2 s
        match n, t with
        | Some (Integer idx), Some tVal when idx >= 1 ->
            let mArg =
                match tVal with
                | Str (_, args) when idx <= List.length args -> Some (List.item (idx - 1) args)
                | VList (x :: _) when idx = 1 -> Some x
                | VList (_ :: xs) when idx = 2 -> Some (VList xs)
                | _ -> None
            match mArg with
            | None -> None
            | Some a ->
                match bindOutput 3 a s with
                | None    -> None
                | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | _ -> None

    // =../2 (univ): A1 = T, A2 = L. Decompose (instantiated A1) or
    // compose (unbound A1, list in A2). Mirrors the Haskell/Rust/C++
    // baseline (see wam_haskell_target.pl =../2 step case).
    | BuiltinCall ("=../2", _) ->
        let t = getReg 1 s
        match t with
        | Some (Unbound vid) ->
            // Compose mode: read a proper list from A2.
            // The runtime stores proper lists in two equivalent
            // shapes: `VList [x1; ...; xn]` (compact) and `Str ("[|]",
            // [head; tail])` cons cells (when the tail had to stay
            // symbolic at materialization time — e.g. came back from
            // parse_args bound to a head/tail var pair).  Both are
            // valid Prolog lists; flatten Str-cons into the same
            // sequence VList would have before destructuring.
            let rec flattenCons (v: Value) : Value list option =
                match derefVar s.WsBindings v with
                | VList xs               -> Some xs
                | Atom "[]"              -> Some []
                | Str ("[|]", [h; tail]) ->
                    match flattenCons tail with
                    | Some rest -> Some (derefVar s.WsBindings h :: rest)
                    | None      -> None
                | _ -> None
            let l = getReg 2 s
            let itemsOpt =
                match l with
                | Some lv -> flattenCons lv
                | None    -> None
            match itemsOpt with
            | None -> None
            | Some items ->
                let mBuilt =
                    match items with
                    | []                   -> None
                    | [x]                  -> Some x
                    | (Atom fname) :: rest -> Some (Str (fname, rest))
                    | _                    -> None
                match mBuilt with
                | None       -> None
                | Some built ->
                    let regs = Array.copy s.WsRegs
                    regs.[1] <- built
                    Some { s with
                             WsPC       = s.WsPC + 1
                             WsRegs     = regs
                             WsBindings = Map.add vid built s.WsBindings
                             WsTrail    = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                             WsTrailLen = s.WsTrailLen + 1 }
        | Some tVal ->
            // Decompose mode: build list from T.
            let mList =
                match tVal with
                | Str (fn, args)  -> Some (VList ((Atom fn) :: args))
                | Atom _          -> Some (VList [tVal])
                | Integer _       -> Some (VList [tVal])
                | Float _         -> Some (VList [tVal])
                | VList []        -> Some (VList [Atom "[]"])
                | VList (x :: xs) -> Some (VList [Atom "."; x; VList xs])
                | _               -> None
            match mList with
            | None    -> None
            | Some lv ->
                match bindOutput 2 lv s with
                | None    -> None
                | Some s1 -> Some { s1 with WsPC = s1.WsPC + 1 }
        | None -> None

    // \\+/1 (negation as failure): A1 = Goal. Resolves the goal label and
    // runs the snapshot; success of the inner run => NAF fails, failure of
    // the inner run => NAF succeeds. Mirrors the Haskell baseline
    // (wam_haskell_target.pl \\+/1 step case, sans the interned-atom table —
    // F# Atom is string-based, so no lookupAtom indirection is needed).
    | BuiltinCall ("\\\\+/1", _) ->
        let goalOpt = getReg 1 s
        match goalOpt with
        // Fast path: \\+ member(X, L) — walk the list inline.
        | Some (Str ("member", [needle; haystack])) ->
            let n = derefVar s.WsBindings needle
            let h = derefVar s.WsBindings haystack
            let found =
                match h with
                | VList items -> items |> List.exists (fun item -> derefVar s.WsBindings item = n)
                | _ -> false
            if found then None else Some { s with WsPC = s.WsPC + 1 }
        // Fast path: \\+ true always fails, \\+ fail always succeeds.
        | Some (Atom "true") -> None
        | Some (Atom "fail") -> Some { s with WsPC = s.WsPC + 1 }
        // General path: resolve the goal label, snapshot-and-run.
        // If the goal''s entry instruction is a ParTryMeElse(Pc), dispatch
        // through runNegationParallel for fork-eligible chains; otherwise
        // fall back to the sequential snapshot-and-run.
        | Some (Str (fname, args)) ->
            let goalKey = sprintf "%s/%d" fname (List.length args)
            match Map.tryFind goalKey ctx.WcLabels with
            | Some pc ->
                let dArgs = args |> List.map (derefVar s.WsBindings)
                let regsSnap = Array.create MaxRegs (Unbound -1)
                dArgs |> List.iteri (fun i v -> regsSnap.[i + 1] <- v)
                let snap = { s with WsRegs = regsSnap }
                if pc >= 0 && pc < ctx.WcCode.Length then
                    match ctx.WcCode.[pc] with
                    | ParTryMeElse elseLabel ->
                        let elsePC = Map.tryFind elseLabel ctx.WcLabels |> Option.defaultValue -1
                        if runNegationParallel ctx snap pc elsePC then None
                        else Some { s with WsPC = s.WsPC + 1 }
                    | ParTryMeElsePc elsePC ->
                        if runNegationParallel ctx snap pc elsePC then None
                        else Some { s with WsPC = s.WsPC + 1 }
                    | _ ->
                        let snapshot = { snap with WsPC = pc; WsCP = 0; WsCutBar = 0 }
                        match run ctx snapshot with
                        | Some _ -> None
                        | None   -> Some { s with WsPC = s.WsPC + 1 }
                else Some { s with WsPC = s.WsPC + 1 }
            | None -> Some { s with WsPC = s.WsPC + 1 }  // unknown pred: treat as failing goal
        // Atom as 0-arity goal (e.g. \\+ some_pred)
        | Some (Atom fname) ->
            let goalKey = sprintf "%s/0" fname
            match Map.tryFind goalKey ctx.WcLabels with
            | Some pc ->
                let snapshot = { s with WsPC = pc; WsCP = 0; WsCutBar = 0 }
                match run ctx snapshot with
                | Some _ -> None
                | None   -> Some { s with WsPC = s.WsPC + 1 }
            | None -> Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // ========================================================================
    // ISO catch/3 and throw/1 — exception-style non-local control.
    //
    // catch(Goal, Catcher, Recovery): pushes a CatcherFrame holding a
    // snapshot of pre-call state, runs Goal recursively via `run`, pops
    // the frame on normal return.  If Goal (or any nested goal) executes
    // throw/1, F# raises WamException with the thrown term; the try/with
    // here catches it, restores the snapshot, unifies Catcher with the
    // thrown term, and runs Recovery.  Mirrors Python _execute_catch.
    //
    // throw(Term): deep-derefs Term through the current bindings and
    // raises WamException.  Propagates through the run loop until caught
    // by a catch/3 try/with or by the top-level runPredicate harness
    // (which prints \"Uncaught Prolog throw: ...\" and returns None).
    // ========================================================================
    | BuiltinCall (\"throw/1\", _) ->
        // Plain throw/1: pass the user''s term through unchanged.  ISO
        // wrapping in error(ErrorTerm, Context) is the job of
        // throwIsoError, which is_iso/2 and friends call directly when
        // they detect a structured ISO failure.  User code that wants
        // ISO shape passes error(...) terms explicitly.
        match getReg 1 s with
        | Some v ->
            let thrown = derefDeep s.WsBindings v
            raise (WamException thrown)
        | None ->
            raise (WamException (Atom \"instantiation_error\"))

    | BuiltinCall (\"catch/3\", _) ->
        let goal = getReg 1 s
        let catcherTerm  = getReg 2 s |> Option.defaultValue (Unbound -1)
        let recoveryTerm = getReg 3 s |> Option.defaultValue (Unbound -1)
        let snapshotRegs = Array.copy s.WsRegs
        let snapshot = { s with WsRegs = snapshotRegs }
        let frame =
            { CfCatcherTerm  = catcherTerm
              CfRecoveryTerm = recoveryTerm
              CfSnapshot     = snapshot
              CfSnapshotRegs = snapshotRegs }
        // Resolve the goal and build a child state to run.  Two
        // dispatch paths:
        //   1. Label lookup (user-defined predicate) -> jump to PC and
        //      run.  This is the common case.
        //   2. Builtin call (catch/3, throw/1, is_iso/2, is_lax/2, etc.)
        //      -- the goal name isn''t a label so we fall through to a
        //      BuiltinCall step.  Needed so nested catch/3 inside the
        //      caller''s catch goal works, and so is_iso/2 (which the
        //      WAM compiler emits as Execute, not BuiltinCall) can fire
        //      its throw.
        // true/fail are inlined to avoid round-tripping them through
        // BuiltinCall.
        let runGoalInChild (goalVal: Value) : WamState option =
            let runWithArgs goalKey arity (dArgs: Value list) =
                let regsCh = Array.create MaxRegs (Unbound -1)
                dArgs |> List.iteri (fun i v -> regsCh.[i + 1] <- v)
                let child =
                    { s with
                        WsRegs     = regsCh
                        WsCP       = 0
                        WsCutBar   = 0
                        WsB0Stack  = []
                        WsCatchers = frame :: s.WsCatchers }
                match Map.tryFind goalKey ctx.WcLabels with
                | Some pc -> run ctx { child with WsPC = pc }
                | None ->
                    // Fall through to BuiltinCall dispatch.  The PC is
                    // immaterial here -- BuiltinCall doesn''t read WsPC
                    // for dispatch; we set it to 0 so a recursive run
                    // halts cleanly when the builtin returns.
                    step ctx { child with WsPC = 0 } (BuiltinCall (goalKey, arity))
            match goalVal with
            | Atom \"true\" -> Some s
            | Atom \"fail\" -> None
            | Str (fname, args) ->
                let arity = List.length args
                let dArgs = args |> List.map (derefVar s.WsBindings)
                let goalKey = sprintf \"%s/%d\" fname arity
                runWithArgs goalKey arity dArgs
            | Atom fname ->
                let goalKey = sprintf \"%s/0\" fname
                runWithArgs goalKey 0 []
            | _ -> None
        try
            match goal with
            | Some g ->
                match runGoalInChild g with
                | Some resultState ->
                    // Goal succeeded.  Carry bindings/trail/heap forward
                    // (they belong to the caller now), but revert WsRegs
                    // to the catch/3 caller''s regs since the child used
                    // a fresh register array.  Filter our specific
                    // catcher frame out by reference identity in case
                    // nested catches pushed their own.
                    let withoutFrame =
                        resultState.WsCatchers
                        |> List.filter (fun f -> not (System.Object.ReferenceEquals(f, frame)))
                    Some { resultState with
                             WsRegs     = s.WsRegs
                             WsCatchers = withoutFrame
                             WsPC       = s.WsPC + 1 }
                | None -> None
            | None -> None
        with
        | WamException thrown ->
            // Restore from snapshot, try to unify Catcher with Thrown,
            // and on success run Recovery.  On unify failure rethrow so
            // an outer catcher can try.
            let regsRestored = Array.copy frame.CfSnapshotRegs
            let restored = { frame.CfSnapshot with WsRegs = regsRestored }
            // Unify in-place against the restored state''s bindings.
            let rec unifyTerms (a: Value) (b: Value) (st: WamState) : WamState option =
                let da = derefVar st.WsBindings a
                let db = derefVar st.WsBindings b
                match da, db with
                | Unbound va, Unbound vb when va = vb -> Some st
                | Unbound vid, v | v, Unbound vid ->
                    Some { st with
                             WsBindings = Map.add vid v st.WsBindings
                             WsTrail    = { TrailVarId = vid
                                            TrailOldVal = Map.tryFind vid st.WsBindings } :: st.WsTrail
                             WsTrailLen = st.WsTrailLen + 1 }
                | Atom a1, Atom a2 when a1 = a2 -> Some st
                | Integer i1, Integer i2 when i1 = i2 -> Some st
                | Float f1, Float f2 when f1 = f2 -> Some st
                | Str (f1, a1), Str (f2, a2) when f1 = f2 && List.length a1 = List.length a2 ->
                    let rec walk xs ys st0 =
                        match xs, ys with
                        | [], [] -> Some st0
                        | x :: xr, y :: yr ->
                            match unifyTerms x y st0 with
                            | Some st1 -> walk xr yr st1
                            | None -> None
                        | _ -> None
                    walk a1 a2 st
                | VList l1, VList l2 when List.length l1 = List.length l2 ->
                    let rec walk xs ys st0 =
                        match xs, ys with
                        | [], [] -> Some st0
                        | x :: xr, y :: yr ->
                            match unifyTerms x y st0 with
                            | Some st1 -> walk xr yr st1
                            | None -> None
                        | _ -> None
                    walk l1 l2 st
                | _ -> None
            match unifyTerms frame.CfCatcherTerm thrown restored with
            | Some unified ->
                // Run the recovery goal in a child state, similar to the
                // catch''s goal path.  Recovery success advances PC past
                // catch/3; failure causes catch to fail.
                let recoveryRun =
                    match derefVar unified.WsBindings frame.CfRecoveryTerm with
                    // Common builtin atoms: handle directly without
                    // requiring a label/builtin dispatch round-trip.
                    | Atom \"true\"  -> Some unified
                    | Atom \"fail\"  -> None
                    | Str (fname, args) ->
                        let dArgs = args |> List.map (derefVar unified.WsBindings)
                        let goalKey = sprintf \"%s/%d\" fname (List.length args)
                        match Map.tryFind goalKey ctx.WcLabels with
                        | Some pc ->
                            let regsCh = Array.create MaxRegs (Unbound -1)
                            dArgs |> List.iteri (fun i v -> regsCh.[i + 1] <- v)
                            let child =
                                { unified with
                                    WsPC      = pc
                                    WsRegs    = regsCh
                                    WsCP      = 0
                                    WsCutBar  = 0
                                    WsB0Stack = [] }
                            run ctx child
                        | None -> None
                    | Atom fname ->
                        let goalKey = sprintf \"%s/0\" fname
                        match Map.tryFind goalKey ctx.WcLabels with
                        | Some pc ->
                            let child =
                                { unified with
                                    WsPC      = pc
                                    WsCP      = 0
                                    WsCutBar  = 0
                                    WsB0Stack = [] }
                            run ctx child
                        | None -> None
                    | _ -> None
                match recoveryRun with
                | Some _ -> Some { unified with WsPC = s.WsPC + 1 }
                | None   -> None
            | None ->
                // Unify failed -- rethrow for outer catcher.
                raise (WamException thrown)

    // ========================================================================
    // Phase I — Haskell-only specialized instructions (perf optimizations).
    // ========================================================================
    // Reference: src/unifyweaver/targets/wam_haskell_target.pl. These are
    // emitted by the WAM compiler''s binding-analysis pass when it can
    // prove the operand shapes statically.

    // PutStructureDyn — like PutStructure but the functor name and arity
    // come from registers at runtime. Used after =../2 / functor/3 with
    // variable-shaped output.
    | PutStructureDyn (nameReg, arityReg, targetReg) ->
        let mName  = getReg nameReg s
        let mArity = getReg arityReg s
        match mName, mArity with
        | Some (Atom fname), Some (Integer arity) when arity >= 0 ->
            let push = pushBuilderIfActive s
            Some { push with
                     WsPC      = s.WsPC + 1
                     WsBuilder = Some (BuildStruct (fname, targetReg, arity, [])) }
        | _ -> None  // name must be Atom, arity a non-negative Integer

    // Arg — specialized arg/3 with literal N (positive integer). Reads T
    // from tReg, extracts the Nth subterm, unifies with aReg.
    | Arg (n, tReg, aReg) when n >= 1 ->
        match getReg tReg s with
        | Some tVal ->
            let mElem =
                match tVal with
                | Str (_, args) when n <= List.length args -> Some (List.item (n - 1) args)
                | VList (x :: _)  when n = 1 -> Some x
                | VList (_ :: xs) when n = 2 -> Some (VList xs)
                | _ -> None
            match mElem with
            | None -> None
            | Some elem_ ->
                // Mirror Haskell semantics: if aReg is uninitialized (the
                // -1 sentinel via getReg = None), insert directly; if it
                // dereferences to Unbound, also unify by binding; otherwise
                // require structural equality.
                if aReg < 0 || aReg >= s.WsRegs.Length then None
                else
                    let regSlot = s.WsRegs.[aReg]
                    match regSlot with
                    | Unbound -1 ->
                        // Sentinel = uninitialized — just write the value.
                        let r = Array.copy s.WsRegs
                        r.[aReg] <- elem_
                        Some { s with WsPC = s.WsPC + 1; WsRegs = r }
                    | _ ->
                        let dv = derefVar s.WsBindings regSlot
                        match dv with
                        | Unbound vid ->
                            let r = Array.copy s.WsRegs
                            r.[aReg] <- elem_
                            Some { s with
                                     WsPC       = s.WsPC + 1
                                     WsRegs     = r
                                     WsBindings = Map.add vid elem_ s.WsBindings
                                     WsTrail    = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                                     WsTrailLen = s.WsTrailLen + 1 }
                        | existing when existing = elem_ ->
                            Some { s with WsPC = s.WsPC + 1 }
                        | _ -> None
        | None -> None
    | Arg (_, _, _) -> None

    // NotMemberList — specialized \\+ member(X, L) on a bound VList L.
    // Walks L inline, succeeds when X cannot unify with any item.
    | NotMemberList (xReg, lReg) ->
        match getReg xReg s, getReg lReg s with
        | Some x, Some (VList items) ->
            let found =
                items |> List.exists (fun item -> derefVar s.WsBindings item = x)
            if found then None else Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    // NotMemberConstAtoms — \\+ member(X, [a, b, c, ...]) with the atoms
    // baked into the instruction. Single-dispatch fast path:
    //   - X is an Atom: succeed iff X.name is not in the atom set.
    //   - X is an Unbound or Ref (could-unify): fail (matches Prolog).
    //   - X is any other ground value: succeed (cannot unify with atoms).
    | NotMemberConstAtoms (xReg, atoms) ->
        match getReg xReg s with
        | Some (Atom name) ->
            if List.contains name atoms then None
            else Some { s with WsPC = s.WsPC + 1 }
        | Some (Unbound _) -> None
        | Some (Ref _)     -> None
        | Some _           -> Some { s with WsPC = s.WsPC + 1 }
        | None             -> None

    // BuildEmptySet — write VSet Set.empty into the target register.
    // Bootstraps the visited-set argument for category-ancestor-style
    // recursive predicates.
    | BuildEmptySet reg ->
        if reg < 0 || reg >= s.WsRegs.Length then None
        else
            let r = Array.copy s.WsRegs
            r.[reg] <- VSet Set.empty
            Some { s with WsPC = s.WsPC + 1; WsRegs = r }

    // SetInsert — read Atom from elemReg + VSet from inReg, write the
    // inserted VSet to outReg. Fails if either register has the wrong shape.
    | SetInsert (elemReg, inReg, outReg) ->
        match getReg elemReg s, getReg inReg s with
        | Some (Atom name), Some (VSet s0) ->
            if outReg < 0 || outReg >= s.WsRegs.Length then None
            else
                let r = Array.copy s.WsRegs
                r.[outReg] <- VSet (Set.add name s0)
                Some { s with WsPC = s.WsPC + 1; WsRegs = r }
        | _ -> None

    // NotMemberSet — O(log N) membership check via Set.contains. Succeeds
    // when the Atom is NOT in the VSet. Replaces the O(N) NotMemberList
    // walk on the visited-set hot path.
    | NotMemberSet (elemReg, setReg) ->
        match getReg elemReg s, getReg setReg s with
        | Some (Atom name), Some (VSet s0) ->
            if Set.contains name s0 then None
            else Some { s with WsPC = s.WsPC + 1 }
        | _ -> None

    | _ -> None   // fallback for unhandled instructions

and updateNearestAggFrame (rpc: int) (cps: ChoicePoint list) : ChoicePoint list =
    match cps with
    | [] -> []
    | cp :: rest ->
        match cp.CpAggFrame with
        | Some af -> { cp with CpAggFrame = Some { af with AggReturnPC = rpc } } :: rest
        | None    -> cp :: updateNearestAggFrame rpc rest

// Recursive structural unification — does NOT touch WsPC.  Returns
// the updated state (with new bindings/trail entries) or None on
// mismatch.  Handles the VList / Str cons-cell equivalence:
// `VList [h1; ...; hn]` represents `[h1, ..., hn]` with `[]` tail;
// `Str ("[|]", [h; t])` represents `[h | t]` where t may be a
// variable, another structure, or another VList.  Both encodings
// flow out of GetList / addToBuilder depending on which path
// materialized them, so a unifier that only does structural F#
// equality (==) misses the equivalence — that''s exactly the bug
// that broke parse_args'' `Tokens1 = [tk_rparen|RestOut]` check on
// the F# parser regression for `p(a)` / `p(a,b)`.
and unifyTerms (a: Value) (b: Value) (s: WamState) : WamState option =
    let a = derefVar s.WsBindings a
    let b = derefVar s.WsBindings b
    let bindUnbound vid v st =
        { st with
            WsBindings = Map.add vid v st.WsBindings
            WsTrail    = { TrailVarId = vid; TrailOldVal = Map.tryFind vid st.WsBindings } :: st.WsTrail
            WsTrailLen = st.WsTrailLen + 1 }
    match a, b with
    | Unbound vid, v -> Some (bindUnbound vid v s)
    | v, Unbound vid -> Some (bindUnbound vid v s)
    // [] equivalences.
    | VList [], Atom "[]" -> Some s
    | Atom "[]", VList [] -> Some s
    // List ~ list element-wise.
    | VList (h1 :: t1), VList (h2 :: t2) ->
        match unifyTerms h1 h2 s with
        | None    -> None
        | Some s1 -> unifyTerms (VList t1) (VList t2) s1
    // List ~ cons-cell.  VList [h; ...] ≡ Str ("[|]", [h; VList [...]]).
    | VList (h :: t), Str ("[|]", [hb; tb]) ->
        match unifyTerms h hb s with
        | None    -> None
        | Some s1 -> unifyTerms (VList t) tb s1
    | Str ("[|]", [ha; ta]), VList (h :: t) ->
        match unifyTerms ha h s with
        | None    -> None
        | Some s1 -> unifyTerms ta (VList t) s1
    // Structural recurse on same-functor compounds.
    | Str (f1, args1), Str (f2, args2)
        when f1 = f2 && List.length args1 = List.length args2 ->
        let rec go xs ys st =
            match xs, ys with
            | [], [] -> Some st
            | x :: xt, y :: yt ->
                match unifyTerms x y st with
                | None     -> None
                | Some st1 -> go xt yt st1
            | _ -> None
        go args1 args2 s
    | x, y when x = y -> Some s
    | _               -> None

and flattenList (s: WamState) (v: Value) : Value list option =
    // Walk a list in either runtime encoding (VList xs OR a Str ''[|]''
    // cons-cell chain) into an F# list of elements.  Returns None for
    // an improper list (one whose final tail is not [] / VList []) or a
    // non-list value.  Use this at the entry of every list-consuming
    // builtin (append/3, length/2, reverse/2, ...) so they accept
    // whichever encoding the upstream builder produced -- BuildList
    // chooses Str(''[|]'') when the tail is unbound during build, so
    // even fully-ground source-level list literals can show up as
    // cons-cell chains after a SetVariable + PutStructure pair.
    let rec walk acc v =
        match derefVar s.WsBindings v with
        | VList xs            -> Some (List.rev acc @ xs)
        | Atom "[]"           -> Some (List.rev acc)
        | Str ("[|]", [h; t]) -> walk (h :: acc) t
        | _                   -> None
    walk [] v

and termEqual (a: Value) (b: Value) (s: WamState) : bool =
    // Standard-order structural equality (Prolog ==/2): like unifyTerms
    // but pure — no binding, no trail.  Derefs both sides through the
    // bindings, normalises VList <-> Str ''[|]'' cons-cell list encoding
    // (same equivalence as unifyTerms), treats unbound vars as equal
    // only when they share the same var id.
    let a = derefVar s.WsBindings a
    let b = derefVar s.WsBindings b
    match a, b with
    | Unbound v1, Unbound v2 -> v1 = v2
    | Unbound _, _           -> false
    | _, Unbound _           -> false
    | VList [], Atom "[]"    -> true
    | Atom "[]", VList []    -> true
    | VList (h1 :: t1), VList (h2 :: t2) ->
        termEqual h1 h2 s && termEqual (VList t1) (VList t2) s
    | VList (h :: t), Str ("[|]", [hb; tb]) ->
        termEqual h hb s && termEqual (VList t) tb s
    | Str ("[|]", [ha; ta]), VList (h :: t) ->
        termEqual ha h s && termEqual ta (VList t) s
    | Str (f1, a1), Str (f2, a2)
        when f1 = f2 && List.length a1 = List.length a2 ->
        List.forall2 (fun x y -> termEqual x y s) a1 a2
    | x, y -> x = y

and unifyVal (a: Value) (b: Value) (s: WamState) : WamState option =
    // Public unification entry — performs the recursive unification
    // via unifyTerms (which is PC-agnostic) and advances WsPC by 1
    // on success so the caller (typically the =/2 / unify_value
    // step handler) doesn''t need to.
    match unifyTerms a b s with
    | None    -> None
    | Some sN -> Some { sN with WsPC = sN.WsPC + 1 }

/// Unify a list of (arg, column) pairs left-to-right, threading state. Used by
/// the T9 fact-table enumerator to unify every column of a candidate row.
and unifyColumns (pairs: (Value * Value) list) (st: WamState) : WamState option =
    match pairs with
    | [] -> Some st
    | (a, c) :: ps ->
        match unifyTerms a c st with
        | None     -> None
        | Some st2 -> unifyColumns ps st2

/// T9 fact-table enumerator: try each candidate row (a VList of column values)
/// against the query args, leaving a FactTableRetry choice point for the rest so
/// backtracking yields the next matching row. On success sets WsPC = retPC (the
/// call-site continuation: pc+1 for `call`, the saved cp for tail-call
/// `execute`). Mirrors select/3''s choice-point construction.
and factTableAttempt (args: Value list) (cands: Value list) (retPC: int) (s: WamState) : WamState option =
    let arity = List.length args
    let rec tryNext cs =
        match cs with
        | [] -> None
        | row :: more ->
            match row with
            | VList cols when List.length cols = arity ->
                match unifyColumns (List.zip args cols) s with
                | Some s2 ->
                    let newCPs, newCPsLen =
                        if List.isEmpty more then s.WsCPs, s.WsCPsLen
                        else
                            let cp = { CpNextPC   = retPC
                                       CpRegs     = Array.copy s.WsRegs
                                       CpStack    = s.WsStack
                                       CpCP       = s.WsCP
                                       CpTrailLen = s.WsTrailLen
                                       CpHeapLen  = s.WsHeapLen
                                       CpBindings = s.WsBindings
                                       CpCutBar   = s.WsCutBar
                                       CpB0StackLen = List.length s.WsB0Stack
                                       CpAggFrame = None
                                       CpBuiltin  = Some (FactTableRetry (args, more, retPC)) }
                            cp :: s.WsCPs, s.WsCPsLen + 1
                    Some { s2 with WsPC = retPC; WsCPs = newCPs; WsCPsLen = newCPsLen }
                | None -> tryNext more
            | _ -> tryNext more
    tryNext cands'.

% ============================================================================
% PHASE 4: Run Loop
% ============================================================================

run_loop_fsharp(Code) :-
    Code = '/// Main execution loop. Runs until halt (pc=0), failure, or
/// cancellation.  WamContext is read-only. Tail-recursive via use of
/// trampolining.
///
/// Hard-cancel: when ctx.WcCancellationToken is set and reports
/// IsCancellationRequested, the loop returns None immediately.  Wired
/// by runNegationParallel so a successful branch can halt its siblings''
/// wasted work.  The check is one Map.tryFind + IsCancellationRequested
/// per iteration when a token is set, zero overhead when not.
and run (ctx: WamContext) (s: WamState) : WamState option =
    if s.WsPC = 0 then Some s
    elif (match ctx.WcCancellationToken with
          | Some t when t.IsCancellationRequested -> true
          | _ -> false) then None
    else
        let instr = ctx.WcCode.[s.WsPC]
        match step ctx s instr with
        | Some s2 -> run ctx s2
        | None    ->
            match backtrack s with
            | Some s2 -> run ctx s2
            | None    -> None

/// Run multiple seed states in parallel using TPL (System.Threading.Tasks.Parallel).
/// WamContext is read-only and safely shared across threads.
/// Each seed gets its own WamState copy — no shared mutable state.
and runParallel (ctx: WamContext) (seeds: WamState list) : WamState option list =
    seeds
    |> List.toArray
    |> Array.Parallel.map (fun seed -> run ctx seed)
    |> Array.toList

/// Minimum branch count below which forking is overhead-only.  Matches the
/// Haskell baseline (wam_haskell_target.pl forkMinBranches = 3).
and forkMinBranches : int = 3

/// Walk a Par*-chain starting at parPC, following the else-label of each
/// non-terminal ParRetryMeElse / ParRetryMeElsePc until ParTrustMe.  Returns
/// the entry PC of every branch (one past the corresponding Par* instr) in
/// chain order.  Pre-Par variants (RetryMeElse / RetryMeElsePc / TrustMe)
/// terminate the chain — the fork still covers everything up to that point.
/// Mirrors wam_haskell_target.pl enumerateParBranches.
and enumerateParBranches (ctx: WamContext) (parPC: int) (elsePC: int) : int list =
    let rec collectRest pc acc =
        if pc < 0 || pc >= ctx.WcCode.Length then List.rev acc
        else
            match ctx.WcCode.[pc] with
            | ParRetryMeElse label ->
                let nextPC = Map.tryFind label ctx.WcLabels |> Option.defaultValue -1
                collectRest nextPC ((pc + 1) :: acc)
            | ParRetryMeElsePc nextPC ->
                collectRest nextPC ((pc + 1) :: acc)
            | ParTrustMe ->
                List.rev ((pc + 1) :: acc)
            | RetryMeElse _ | RetryMeElsePc _ | TrustMe ->
                List.rev acc  // mixed sequential/parallel chain — stop here
            | _ -> List.rev acc
    (parPC + 1) :: collectRest elsePC []

/// Fork all branches of a Par*-chain and check whether ANY succeeds.
/// Used by \\+/1 to evaluate negation of a goal with parallel alternative
/// clauses: success of any branch => the goal succeeds => negation fails.
///
/// F#-specific notes vs Haskell baseline:
///   - Haskell uses Control.Concurrent.Async with waitAny + cancel for
///     race-to-cancel semantics.  F# branches run `run ctx snapshot`,
///     which is a tight tail-recursive loop without cancellation checks,
///     so a true "hard cancel" of in-progress branches would need a
///     CancellationToken threaded through every step — out of scope.
///   - This implementation uses Async.Choice for "soft race-to-cancel":
///     each branch is wrapped as Async<unit option> returning Some () on
///     success, None on failure.  Async.Choice returns as soon as the
///     first Some arrives and signals cancellation to the others.  The
///     others'' `run` loops don''t honor cancellation tokens, so they
///     continue running in the background until they naturally finish —
///     but `runNegationParallel` returns immediately on the first
///     success, so wall time is bounded by the FIRST successful branch
///     rather than the SLOWEST branch (the Async.Parallel behavior).
and runNegationParallel (ctx: WamContext) (s: WamState) (entryPC: int) (elsePC: int) : bool =
    // enumerateParBranches returns the BRANCH BODY entry PCs (per its own
    // contract: "Each branch''s body begins at chainOpPC + 1"), so the
    // snapshot here just sets WsPC = pc directly.  An earlier port (and
    // the Haskell baseline) used `pc + 1`, which double-stepped past the
    // first instruction of each branch and could land out of bounds for
    // a chain whose last branch is short — the F# runtime smoke caught
    // this when a 3-branch all-Fail fixture crashed in `run` with
    // IndexOutOfRangeException reading WcCode.[ pc + 1 ] on the tail.
    let branchPCs = enumerateParBranches ctx entryPC elsePC
    if List.length branchPCs >= forkMinBranches then
        let branchAction pc =
            async {
                // Async.CancellationToken pulls the token that Async.Choice
                // wires into each child workflow.  When Async.Choice gets
                // its first Some, it cancels the token, and `run`''s loop
                // check (run_loop_fsharp) returns None on the next iter,
                // halting sibling branches'' work.  Hard-cancel: real CPU
                // / thread savings on the wasted siblings beyond the
                // wall-time win that Async.Choice alone provides.
                let! token = Async.CancellationToken
                let ctxC = { ctx with WcCancellationToken = Some token }
                let snapshot = { s with WsPC = pc; WsCP = 0; WsCutBar = 0 }
                if (run ctxC snapshot).IsSome then return Some ()
                else return None
            }
        // Async.Choice: returns the first Some.  Wall time bounded by
        // the FIRST successful branch rather than the SLOWEST one.
        let result =
            branchPCs
            |> List.map branchAction
            |> Async.Choice
            |> Async.RunSynchronously
        result.IsSome
    else
        // Too few branches for fork overhead to pay off — run sequentially.
        match run ctx { s with WsPC = entryPC; WsCP = 0; WsCutBar = 0 } with
        | Some _ -> true
        | None   -> false

/// Classify an aggregate type string as a MergeStrategy.  Sum / count /
/// bag / set / findall are forkable — their per-branch results combine
/// associatively, so parallel evaluation is safe.  Everything else falls
/// back to sequential.  Mirrors wam_haskell_target.pl inferMergeStrategy.
and inferMergeStrategy (aggType: string) : MergeStrategy =
    match aggType with
    | "sum"     -> MergeSum
    | "count"   -> MergeCount
    | "bag"     -> MergeBag
    | "set"     -> MergeSet
    | "findall" -> MergeFindall
    | "collect" -> MergeFindall   // alias
    | _         -> MergeSequential

/// True iff the strategy can be evaluated by forking branches in parallel.
and isForkableStrategy (ms: MergeStrategy) : bool =
    match ms with
    | MergeSum | MergeCount | MergeBag | MergeSet | MergeFindall -> true
    | MergeSequential -> false

/// Walk WsCPs to find the nearest aggregate frame''s merge strategy, if any.
and currentAggMergeStrategy (s: WamState) : MergeStrategy option =
    let rec go cps =
        match cps with
        | [] -> None
        | cp :: rest ->
            match cp.CpAggFrame with
            | Some af -> Some af.AggMergeStrategy
            | None    -> go rest
    go s.WsCPs

/// Walk WsCPs to find the nearest aggregate frame, returning the whole frame.
and currentAggFrame (s: WamState) : AggFrame option =
    let rec go cps =
        match cps with
        | [] -> None
        | cp :: rest ->
            match cp.CpAggFrame with
            | Some af -> Some af
            | None    -> go rest
    go s.WsCPs

/// Remove the nearest aggregate-frame CP from a CP list.  Used after
/// forkParBranches consumes the aggregate to drop the frame so subsequent
/// backtracking doesn''t try to re-finalize it.
and removeNearestAggFrame (cps: ChoicePoint list) : ChoicePoint list =
    let rec go cps =
        match cps with
        | [] -> []
        | cp :: rest ->
            match cp.CpAggFrame with
            | Some _ -> rest                   // drop it
            | None   -> cp :: go rest
    go cps

/// Scan WcCode forward from a Par* instruction to find the matching
/// EndAggregate''s return PC (one past the EndAggregate).  Returns 0
/// (treated as halt by run) on overrun so a malformed chain fails gracefully
/// rather than indexing out of bounds.  Mirrors wam_haskell_target.pl
/// findOuterEndAggregate.
and findOuterEndAggregate (ctx: WamContext) (startPC: int) : int =
    let lo, hi = 0, ctx.WcCode.Length - 1
    let rec go pc =
        if pc < lo || pc > hi then 0
        else
            match ctx.WcCode.[pc] with
            | EndAggregate _ -> pc + 1
            | _              -> go (pc + 1)
    go (startPC + 1)

/// Combine per-branch aggregate results from forkParBranches into a single
/// final Value, applying the merge strategy at the cross-branch level.
/// Each input is one branch''s already-aggregated value (computed by its
/// own finalizeAggregate from a single-element WsAggAccum), so we are
/// folding aggregates of aggregates here.  Semantics chosen to keep the
/// fork equivalent to the sequential version:
///   sum     : numeric sum of per-branch sums
///   count   : sum of per-branch counts (each branch counts its hits)
///   bag     : concat of per-branch VLists
///   set     : Set.ofList over the flattened bag, back to VList (dedup)
///   findall : same as bag — findall preserves duplicates and order
///   other   : fallback to VList of the raw per-branch results
and combineParBranchResults (ms: MergeStrategy) (results: Value list) : Value =
    match ms with
    | MergeSum ->
        let toNum v =
            match v with
            | Integer n -> float n
            | Float f   -> f
            | _         -> 0.0
        let total = results |> List.sumBy toNum
        if float (int total) = total then Integer (int total) else Float total
    | MergeCount ->
        let toInt v = match v with Integer n -> n | _ -> 0
        Integer (results |> List.sumBy toInt)
    | MergeBag | MergeFindall ->
        let extract v = match v with VList items -> items | _ -> [v]
        VList (results |> List.collect extract)
    | MergeSet ->
        let extract v = match v with VList items -> items | _ -> [v]
        let allItems = results |> List.collect extract
        VList (List.distinct allItems)
    | MergeSequential ->
        VList results

/// Fork all branches of a Par*-chain inside a forkable aggregate frame.
/// Each branch runs to completion (including its own EndAggregate +
/// finalizeAggregate) in parallel via Async.Parallel; we then read the
/// per-branch result from the response register and merge them.
///
/// After fork: returns a state with WsPC = retPC (one past the outer
/// EndAggregate), WsRegs[ResReg] bound to the merged value, and the
/// aggregate frame removed from WsCPs.  Mirrors wam_haskell_target.pl
/// forkParBranches.
and forkParBranches (ctx: WamContext) (s: WamState) (af: AggFrame)
                    (parPC: int) (elsePC: int) : WamState option =
    let branchPCs = enumerateParBranches ctx parPC elsePC
    let runBranch pc =
        async {
            // Each branch inherits the aggregate frame in WsCPs so its own
            // EndAggregate can finalize via the standard path, binding the
            // per-branch aggregate to AggResReg.  WsCP = 0 makes proceed
            // halt the branch.  WsAggAccum starts empty so the branch only
            // accumulates its own contribution.
            let snapshot = { s with
                               WsPC       = pc
                               WsCP       = 0
                               WsCutBar   = 0
                               WsAggAccum = []
                               // Fresh B0 stack for each branch — parent''s
                               // pushes belong to a different execution
                               // context.  Branch runs with WsCP=0 so a
                               // mismatched pop would halt; starting empty
                               // keeps Call/Proceed pairing consistent.
                               WsB0Stack  = []
                               // Catchers don''t span forks: a throw in a
                               // forked branch propagates as uncaught
                               // within the branch, killing it but not
                               // the parent (whose catch frame holds a
                               // snapshot of pre-fork state).  Conservative
                               // choice for v1 -- distributed catch
                               // across parallel branches is out of scope.
                               WsCatchers = [] }
            return run ctx snapshot
        }
    let results =
        branchPCs
        |> List.map runBranch
        |> Async.Parallel
        |> Async.RunSynchronously
    // Extract per-branch aggregate result from AggResReg of each successful branch.
    let perBranchValues =
        results
        |> Array.choose (fun rOpt ->
            rOpt |> Option.bind (fun final ->
                if af.AggResReg >= 0 && af.AggResReg < final.WsRegs.Length then
                    let raw = final.WsRegs.[af.AggResReg]
                    match raw with
                    | Unbound -1 -> None   // sentinel: branch didn''t finalize
                    | v -> Some (derefVar final.WsBindings v)
                else None))
        |> Array.toList
    let combined = combineParBranchResults af.AggMergeStrategy perBranchValues
    let retPC = findOuterEndAggregate ctx parPC
    // Drop the consumed aggregate frame from WsCPs and bind combined to AggResReg.
    let outerCPs = removeNearestAggFrame s.WsCPs
    let regs = Array.copy s.WsRegs
    if af.AggResReg >= 0 && af.AggResReg < regs.Length then
        regs.[af.AggResReg] <- combined
    // Thread through bindings + trail if the outer register slot was Unbound.
    let preBindings, preTrail, preTrailLen =
        if af.AggResReg >= 0 && af.AggResReg < s.WsRegs.Length then
            match derefVar s.WsBindings s.WsRegs.[af.AggResReg] with
            | Unbound vid ->
                (Map.add vid combined s.WsBindings,
                 { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail,
                 s.WsTrailLen + 1)
            | _ -> (s.WsBindings, s.WsTrail, s.WsTrailLen)
        else (s.WsBindings, s.WsTrail, s.WsTrailLen)
    Some { s with
             WsPC       = retPC
             WsRegs     = regs
             WsBindings = preBindings
             WsTrail    = preTrail
             WsTrailLen = preTrailLen
             WsCPs      = outerCPs
             WsCPsLen   = List.length outerCPs
             WsAggAccum = [] }

/// Dispatcher for ParTryMeElse / ParTryMeElsePc.  Checks if we''re inside
/// a forkable aggregate frame with enough branches; if so, fork.  Otherwise
/// fall back to the sequential TryMeElse step semantics (a choice point
/// is pushed, the first branch runs, backtracking will explore the rest).
and forkOrSequential (ctx: WamContext) (s: WamState)
                     (elseTarget: Choice<string, int>) : WamState option =
    let fallback () =
        match elseTarget with
        | Choice1Of2 lbl -> step ctx s (TryMeElse lbl)
        | Choice2Of2 pc  -> step ctx s (TryMeElsePc pc)
    match currentAggFrame s with
    | Some af when isForkableStrategy af.AggMergeStrategy ->
        let elsePC =
            match elseTarget with
            | Choice2Of2 pc  -> pc
            | Choice1Of2 lbl -> Map.tryFind lbl ctx.WcLabels |> Option.defaultValue -1
        if elsePC <= 0 then fallback ()
        else
            let branches = enumerateParBranches ctx s.WsPC elsePC
            if List.length branches >= forkMinBranches then
                forkParBranches ctx s af s.WsPC elsePC
            else fallback ()
    | _ -> fallback ()

/// Indexed fact dispatch for 2-arg facts via BuiltinState CP.
/// O(1) Map lookup; first match returned, FactRetry CP for the rest.
and callIndexedFact2 (ctx: WamContext) (pred: string) (s: WamState) : WamState option =
    let basePred = pred |> Seq.takeWhile ((<>) ''/'') |> System.String.Concat
    let retPC    = s.WsCP
    match Map.tryFind basePred ctx.WcForeignFacts with
    | None -> None
    | Some factIndex ->
        let a1 = s.WsRegs.[1] |> derefVar s.WsBindings
        let a2 = s.WsRegs.[2] |> derefVar s.WsBindings
        match a1 with
        | Atom key ->
            match Map.tryFind key factIndex with
            | Some (v :: rest) ->
                match a2 with
                | Unbound vid ->
                    let newRegs     = Array.copy s.WsRegs
                    newRegs.[2]    <- Atom v
                    let newBindings = Map.add vid (Atom v) s.WsBindings
                    let newTrail    = { TrailVarId = vid; TrailOldVal = Map.tryFind vid s.WsBindings } :: s.WsTrail
                    let newCPs, newCPsLen =
                        match rest with
                        | [] -> s.WsCPs, s.WsCPsLen
                        | _  ->
                            let cp = { CpNextPC   = retPC
                                       CpRegs     = Array.copy s.WsRegs
                                       CpStack    = s.WsStack
                                       CpCP       = s.WsCP
                                       CpTrailLen = s.WsTrailLen
                                       CpHeapLen  = s.WsHeapLen
                                       CpBindings = s.WsBindings
                                       CpCutBar   = s.WsCutBar
                                       CpB0StackLen = List.length s.WsB0Stack
                                       CpAggFrame = None
                                       CpBuiltin  = Some (FactRetry (vid, rest, retPC)) }
                            cp :: s.WsCPs, s.WsCPsLen + 1
                    Some { s with
                             WsPC      = retPC
                             WsRegs    = newRegs
                             WsBindings= newBindings
                             WsTrail   = newTrail
                             WsTrailLen= s.WsTrailLen + 1
                             WsCPs     = newCPs
                             WsCPsLen  = newCPsLen }
                | Atom existing ->
                    if existing = v then Some { s with WsPC = retPC }
                    elif List.contains existing rest then Some { s with WsPC = retPC }
                    else None
                | _ -> None
            | _ -> None
        | _ -> None

/// Dispatch a Call to another predicate for use by lowered functions.
and dispatchCall (ctx: WamContext) (pred: string) (sc: WamState) : WamState option =
    try
        match Map.tryFind pred ctx.WcLoweredPredicates with
        | Some fn -> fn ctx sc
        | None ->
        match callIndexedFact2 ctx pred sc with
        | Some sr -> Some sr
        | None ->
        match Map.tryFind pred ctx.WcLabels with
        | Some pc ->
            // Save the caller''s intended return PC and set WsCP = 0 before
            // entering run.  Without this, the called predicate''s outermost
            // Proceed sets WsPC = WsCP (the caller''s post-call PC), so the
            // interpreter loop keeps executing the CALLER''s WAM continuation
            // -- which is wrong when the caller is a lowered function that
            // intends to handle its own continuation in F#.  Resetting WsCP
            // to 0 makes Proceed stop the run loop (run returns on WsPC = 0);
            // we restore WsCP before handing control back so the lowered
            // caller''s subsequent Allocate sees the right value.
            let savedCP = sc.WsCP
            match run ctx { sc with WsPC = pc; WsCP = 0 } with
            | Some sf -> Some { sf with WsCP = savedCP }
            | None    -> None
        | None    -> None
    with
    | WamException term ->
        // Uncaught throw at top-level dispatch.  Print a diagnostic on
        // stderr (matching the Python runtime''s behavior) and return
        // None so the caller observes a failed predicate rather than a
        // .NET crash.  Tests that want to assert specific errors should
        // use catch/3 from the Prolog side.
        eprintfn \"Uncaught Prolog throw from %s: %A\" pred term
        None

/// Foreign call for lowered functions.
and callForeign (ctx: WamContext) (pred: string) (sc: WamState) : WamState option =
    // Lowered/manual callers own their continuation; keep the raw return
    // shape. Interpreter CallForeign/ExecuteForeign pass completeReturn=true.
    executeForeign ctx pred false sc

{{execute_foreign}}

/// Resolve Call instructions at load time:
///   - Foreign predicates → CallForeign
///   - Known labels → CallResolved (direct PC)
///   - Others → left as Call (runtime dispatch)
let resolveCallInstrs (labels: Map<string, int>) (foreignPreds: string list) (instrs: Instruction list) : Instruction list =
    instrs |> List.map (fun instr ->
        match instr with
        | Call (pred, arity) when List.contains pred foreignPreds ->
            CallForeign (pred, arity)
        | Call (pred, arity) ->
            match Map.tryFind pred labels with
            | Some pc -> CallResolved (pc, arity)
            | None    -> Call (pred, arity)
        | Execute pred when List.contains pred foreignPreds ->
            ExecuteForeign pred
        | Execute pred ->
            match Map.tryFind pred labels with
            | Some pc -> ExecutePc pc
            | None    -> Execute pred
        | Jump label ->
            match Map.tryFind label labels with
            | Some pc -> JumpPc pc
            | None    -> Jump label
        | TryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> TryMeElsePc pc
            | None    -> TryMeElse label
        | RetryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> RetryMeElsePc pc
            | None    -> RetryMeElse label
        | Try label ->
            match Map.tryFind label labels with
            | Some pc -> TryPc pc
            | None    -> Try label
        | Retry label ->
            match Map.tryFind label labels with
            | Some pc -> RetryPc pc
            | None    -> Retry label
        | Trust label ->
            match Map.tryFind label labels with
            | Some pc -> TrustPc pc
            | None    -> Trust label
        | ParTryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> ParTryMeElsePc pc
            | None    -> ParTryMeElse label
        | ParRetryMeElse label ->
            match Map.tryFind label labels with
            | Some pc -> ParRetryMeElsePc pc
            | None    -> ParRetryMeElse label
        | SwitchOnConstant table ->
            let extractKey = function
                | Atom s  -> s
                | Integer n -> string n
                | v -> sprintf "%A" v
            let pcTable =
                table |> Map.toList
                      |> List.choose (fun (v, label) ->
                            Map.tryFind label labels |> Option.map (fun pc -> (extractKey v, pc)))
                      |> List.sortBy fst
                      |> List.toArray
            SwitchOnConstantPc pcTable
        | SwitchOnTerm (constTable, structTable, listLabel) ->
            // Resolve labels to PCs; entries whose label is unknown
            // (e.g. ` default`, which is dropped above) drop out of
            // the table -- the step handler falls through on miss.
            let resolveTable (t: (string * string) array) : (string * int) array =
                t |> Array.choose (fun (k, lbl) ->
                        Map.tryFind lbl labels |> Option.map (fun pc -> (k, pc)))
                  |> Array.sortBy fst
            let listPc =
                if listLabel = "none" then 0
                else Map.tryFind listLabel labels |> Option.defaultValue 0
            SwitchOnTermPc (resolveTable constTable, resolveTable structTable, listPc)
        | i -> i)'.

% ============================================================================
% compile_wam_runtime_to_fsharp/3
% ============================================================================

%% compile_wam_runtime_to_fsharp(+Options, +DetectedKernels, -Code)
compile_wam_runtime_to_fsharp(_Options, DetectedKernels, Code) :-
    step_function_fsharp(StepCode),
    backtrack_fsharp(BacktrackCode),
    run_loop_fsharp(RunLoopTemplate),
    generate_kernel_fsharp(DetectedKernels, KernelFunctionsCode, ExecuteForeignCode),
    render_template(RunLoopTemplate,
                    [execute_foreign=ExecuteForeignCode],
                    RunLoopCode),
    format(string(Code),
'module WamRuntime

open WamTypes

~w

~w

~w

~w
', [KernelFunctionsCode, StepCode, BacktrackCode, RunLoopCode]).

% ============================================================================
% Kernel code generation — F# analogs of the Haskell kernel helpers
% ============================================================================

generate_kernel_fsharp([], KF, EF) :- !,
    KF = "// No kernels detected.",
    EF = "and executeForeign (_ctx: WamContext) (_pred: string) (_completeReturn: bool) (_s: WamState) : WamState option = None".
generate_kernel_fsharp(DetectedKernels, KernelFunctionsCode, ExecuteForeignCode) :-
    % The Mustache bodies are named by kernel kind, not by predicate key, and
    % every runtime fact lookup is supplied as an argument.  Rendering once
    % per detected predicate therefore produces duplicate `let
    % nativeKernel_*` definitions when two predicates share a kind.  Keep all
    % predicate pairs for executeForeign dispatch, but emit one reusable body
    % for each distinct kind.
    distinct_kernel_handler_pairs_fs(DetectedKernels, HandlerKernels),
    maplist(render_kernel_function_fs, HandlerKernels, KernelParts),
    atomic_list_concat(KernelParts, '\n\n', KernelFunctionsCode),
    generate_execute_foreign_fs(DetectedKernels, ExecuteForeignCode).

distinct_kernel_handler_pairs_fs(KernelPairs, DistinctPairs) :-
    distinct_kernel_handler_pairs_fs(KernelPairs, [], DistinctPairs).

distinct_kernel_handler_pairs_fs([], _, []).
distinct_kernel_handler_pairs_fs([Key-Kernel|Rest], SeenKinds, DistinctPairs) :-
    Kernel = recursive_kernel(Kind, _, _),
    (   memberchk(Kind, SeenKinds)
    ->  DistinctPairs = RestDistinct,
        NextSeen = SeenKinds
    ;   DistinctPairs = [Key-Kernel|RestDistinct],
        NextSeen = [Kind|SeenKinds]
    ),
    distinct_kernel_handler_pairs_fs(Rest, NextSeen, RestDistinct).

render_kernel_function_fs(Key-Kernel, Code) :-
    Kernel = recursive_kernel(Kind, _, ConfigOps),
    (   fsharp_kernel_template_path(Kind, AbsPath),
        exists_file(AbsPath)
    ->  read_file_to_string(AbsPath, Template, []),
        config_ops_to_template_vars_fs(ConfigOps, TemplateVars),
        render_template(Template, TemplateVars, Code0),
        atom_string(Code0, Code)
    ;   throw(error(existence_error(fsharp_native_kernel_handler, Kind),
                    context(render_kernel_function_fs/2, Key)))
    ).

config_ops_to_template_vars_fs([], []).
config_ops_to_template_vars_fs([Op|Rest], [Key=Value|RestVars]) :-
    Op =.. [Key, RawValue],
    (   RawValue = Pred/_ -> Value = Pred ; Value = RawValue ),
    config_ops_to_template_vars_fs(Rest, RestVars).

generate_execute_foreign_fs(DetectedKernels, Code) :-
    with_output_to(string(Code), (
        format("and executeForeign (ctx: WamContext) (pred: string) (completeReturn: bool) (s: WamState) : WamState option =~n"),
        format("    match pred with~n"),
        forall(member(KV, DetectedKernels), emit_execute_foreign_entry_fs(KV)),
        format("    | _ -> None~n")
    )).

emit_execute_foreign_entry_fs(Key-Kernel) :-
    Kernel = recursive_kernel(Kind, _, ConfigOps),
    (   kernel_register_layout(Kind, RegSpecs),
        kernel_native_call(Kind, CallSpec)
    ->  resolve_call_spec_fs(CallSpec, ConfigOps, ResolvedCallSpec),
        emit_ef_clause_fs(Key, RegSpecs, ResolvedCallSpec)
    ;   format('    // executeForeign: no metadata for ~w~n', [Key])
    ).

resolve_call_spec_fs(call(Func, Args), ConfigOps, call(Func, ResolvedArgs)) :-
    maplist(resolve_arg_spec_fs(ConfigOps), Args, ResolvedArgs).

resolve_arg_spec_fs(ConfigOps, config_facts_from(ConfigKey), config_facts(FactName)) :- !,
    Op =.. [ConfigKey, RawValue],
    member(Op, ConfigOps),
    (   RawValue = Pred/_ -> FactName = Pred ; FactName = RawValue ).
resolve_arg_spec_fs(ConfigOps, config_weighted_facts_from(ConfigKey), config_weighted_facts(FactName)) :- !,
    Op =.. [ConfigKey, RawValue],
    member(Op, ConfigOps),
    (   RawValue = Pred/_ -> FactName = Pred ; FactName = RawValue ).
resolve_arg_spec_fs(_, Arg, Arg).

emit_ef_clause_fs(Key, RegSpecs, call(FuncName, ArgSpecs)) :-
    format('    | "~w" ->~n', [Key]),
    include(is_input_reg_fs, RegSpecs, InputRegs),
    format('        let '),
    emit_input_let_bindings_fs(InputRegs, first),
    emit_config_let_bindings_fs(ArgSpecs),
    include(is_output_reg_fs, RegSpecs, OutputRegs),
    emit_case_and_call_fs(InputRegs, OutputRegs, FuncName, ArgSpecs),
    format('~n').

is_input_reg_fs(input(_, _)).
is_output_reg_fs(output(_, _)).

emit_input_let_bindings_fs([], _).
emit_input_let_bindings_fs([input(RegN, Type)|Rest], Pos) :-
    reg_var_name_fs(RegN, VarName),
    fsharp_wam_reg_default(Type, Default),
    (   Pos = first -> true ; format('        let ') ),
    format('~w = (let rv = s.WsRegs.[~w] in match rv with Unbound -1 -> ~w | _ -> rv) |> derefVar s.WsBindings~n',
           [VarName, RegN, Default]),
    emit_input_let_bindings_fs(Rest, rest).

reg_var_name_fs(N, Name) :- format(atom(Name), 'r~w', [N]).

emit_config_let_bindings_fs([]).
emit_config_let_bindings_fs([config_facts(FactKey)|Rest]) :-
    format('        let ~w_facts = resolveFactLookup "~w" ctx~n',
           [FactKey, FactKey]),
    emit_config_let_bindings_fs(Rest).
emit_config_let_bindings_fs([config_weighted_facts(FactKey)|Rest]) :-
    format('        let ~w_facts = Map.tryFind "~w" ctx.WcFfiWeightedFacts |> Option.defaultValue Map.empty~n',
           [FactKey, FactKey]),
    emit_config_let_bindings_fs(Rest).
emit_config_let_bindings_fs([config_int(ConfigKey, Default)|Rest]) :-
    format('        let ~w_cfg = Map.tryFind "~w" ctx.WcForeignConfig |> Option.defaultValue ~w~n',
           [ConfigKey, ConfigKey, Default]),
    emit_config_let_bindings_fs(Rest).
emit_config_let_bindings_fs([config_float(ConfigKey, Default)|Rest]) :-
    format('        let ~w_cfg = ~w~n', [ConfigKey, Default]),
    emit_config_let_bindings_fs(Rest).
emit_config_let_bindings_fs([_|Rest]) :-
    emit_config_let_bindings_fs(Rest).

emit_case_and_call_fs(InputRegs, OutputRegs, FuncName, ArgSpecs) :-
    length(InputRegs, NInputs),
    (   NInputs =:= 1
    ->  InputRegs = [input(RegN1, _)],
        reg_var_name_fs(RegN1, ScrutName),
        format('        match ~w with~n', [ScrutName]),
        emit_single_case_branch_fs(InputRegs, OutputRegs, FuncName, ArgSpecs)
    ;   format('        match ('),
        emit_scrutinee_tuple_fs(InputRegs, first),
        format(') with~n'),
        format('        | ('),
        emit_pattern_tuple_fs(InputRegs, first),
        format(') ->~n'),
        emit_native_call_and_binding_fs(OutputRegs, FuncName, ArgSpecs, InputRegs, "            ")
    ),
    format('        | _ -> None~n').

emit_single_case_branch_fs([input(RegN, Type)|_], OutputRegs, FuncName, ArgSpecs) :-
    reg_var_name_fs(RegN, VarName),
    type_pattern_fs(Type, VarName, Pattern),
    format('        | ~w ->~n', [Pattern]),
    emit_native_call_and_binding_fs(OutputRegs, FuncName, ArgSpecs, [input(RegN, Type)], "            ").

emit_scrutinee_tuple_fs([], _).
emit_scrutinee_tuple_fs([input(RegN, _)|Rest], Pos) :-
    reg_var_name_fs(RegN, VarName),
    (   Pos = first -> true ; format(', ') ),
    format('~w', [VarName]),
    emit_scrutinee_tuple_fs(Rest, rest).

emit_pattern_tuple_fs([], _).
emit_pattern_tuple_fs([input(RegN, Type)|Rest], Pos) :-
    reg_var_name_fs(RegN, VarName),
    type_pattern_fs(Type, VarName, Pattern),
    (   Pos = first -> true ; format(', ') ),
    format('~w', [Pattern]),
    emit_pattern_tuple_fs(Rest, rest).

type_pattern_fs(atom,        VarName, Pattern) :- format(atom(Pattern), 'Atom ~wS', [VarName]).
type_pattern_fs(integer,     VarName, Pattern) :- format(atom(Pattern), 'Integer ~wI', [VarName]).
type_pattern_fs(vlist_atoms, VarName, Pattern) :- format(atom(Pattern), 'VList ~wL', [VarName]).

emit_native_call_and_binding_fs(OutputRegs, FuncName, ArgSpecs, InputRegs, Indent) :-
    format('~wlet results = ~w', [Indent, FuncName]),
    emit_call_args_fs(ArgSpecs, InputRegs),
    format('~n'),
    emit_stream_binding_multi_fs(OutputRegs, Indent).

emit_call_args_fs([], _).
emit_call_args_fs([Spec|Rest], InputRegs) :-
    format(' '),
    emit_one_call_arg_fs(Spec, InputRegs),
    emit_call_args_fs(Rest, InputRegs).

emit_one_call_arg_fs(config_facts(FactKey), _) :-
    format('~w_facts', [FactKey]).
emit_one_call_arg_fs(config_weighted_facts(FactKey), _) :-
    format('~w_facts', [FactKey]).
emit_one_call_arg_fs(config_int(ConfigKey, _), _) :-
    format('~w_cfg', [ConfigKey]).
emit_one_call_arg_fs(config_float(ConfigKey, _), _) :-
    format('~w_cfg', [ConfigKey]).
emit_one_call_arg_fs(reg(RegN), InputRegs) :-
    member(input(RegN, Type), InputRegs),
    reg_var_name_fs(RegN, VarName),
    emit_reg_extraction_fs(VarName, Type).
emit_one_call_arg_fs(derived(length, RegN), InputRegs) :-
    member(input(RegN, _Type), InputRegs),
    reg_var_name_fs(RegN, VarName),
    format('(~wL |> List.choose (function Atom v -> Some v | _ -> None) |> List.length)', [VarName]).

emit_reg_extraction_fs(VarName, atom) :-
    format('(Map.tryFind ~wS ctx.WcAtomIntern |> Option.defaultValue -1)', [VarName]).
emit_reg_extraction_fs(VarName, vlist_atoms) :-
    format('(~wL |> List.choose (function Atom v -> Map.tryFind v ctx.WcAtomIntern | _ -> None))', [VarName]).
emit_reg_extraction_fs(VarName, integer) :-
    format('~wI', [VarName]).

emit_stream_binding_multi_fs(OutputRegs, Indent) :-
    length(OutputRegs, NOuts),
    format('~wlet retPC = s.WsCP~n', [Indent]),
    format('~wlet returnCP, returnCutBar, returnB0Stack =~n', [Indent]),
    format('~w    if completeReturn then~n', [Indent]),
    format('~w        match s.WsB0Stack with~n', [Indent]),
    format('~w        | top :: rest -> 0, top, rest~n', [Indent]),
    format('~w        | [] -> 0, 0, []~n', [Indent]),
    format('~w    else~n', [Indent]),
    format('~w        s.WsCP, s.WsCutBar, s.WsB0Stack~n', [Indent]),
    emit_multi_out_derefs_fs(OutputRegs, Indent),
    % Respect already-bound output registers (e.g. tc(+Source, +Target)):
    % filter the native result stream before FFIStreamRetry packing.
    % Mirrors Rust transitive_closure2 target_filter retain.
    emit_bound_output_filter_fs(OutputRegs, NOuts, Indent),
    format('~wlet bindResult ', [Indent]),
    emit_tuple_pattern_fs(NOuts),
    format(' =~n', []),
    format('~w    let ', [Indent]),
    emit_multi_wrap_bindings_fs(OutputRegs, 1),
    format('~w    let trailDelta = [', [Indent]),
    emit_outreg_values_list_fs(OutputRegs, 1),
    format('] |> List.sumBy (function Unbound _ -> 1 | _ -> 0)~n', []),
    format('~w    { s with WsPC = retPC~n', [Indent]),
    format('~w             WsCP = returnCP~n', [Indent]),
    format('~w             WsCutBar = returnCutBar~n', [Indent]),
    format('~w             WsB0Stack = returnB0Stack~n', [Indent]),
    emit_multi_reg_updates_fs(OutputRegs, Indent),
    emit_multi_binding_updates_fs(OutputRegs, Indent),
    emit_multi_trail_updates_fs(OutputRegs, Indent),
    format('~w             WsTrailLen = s.WsTrailLen + trailDelta }~n', [Indent]),
    format('~wmatch results with~n', [Indent]),
    format('~w| [] -> None~n', [Indent]),
    format('~w| [h] -> Some (bindResult h)~n', [Indent]),
    format('~w| h :: restResults ->~n', [Indent]),
    format('~w    let s1 = bindResult h~n', [Indent]),
    emit_multi_outvars_fs(OutputRegs, Indent),
    format('~w    let restWrapped = restResults |> List.map (fun ', [Indent]),
    emit_tuple_pattern_fs(NOuts),
    format(' -> [', []),
    emit_multi_wrap_list_fs(OutputRegs, 1),
    format('])~n', []),
    format('~w    let cp = { CpNextPC   = retPC~n', [Indent]),
    format('~w               CpRegs     = Array.copy s.WsRegs~n', [Indent]),
    format('~w               CpStack    = s.WsStack~n', [Indent]),
    format('~w               CpCP       = s.WsCP~n', [Indent]),
    format('~w               CpTrailLen = s.WsTrailLen~n', [Indent]),
    format('~w               CpHeapLen  = s.WsHeapLen~n', [Indent]),
    format('~w               CpBindings = s.WsBindings~n', [Indent]),
    format('~w               CpCutBar   = s.WsCutBar~n', [Indent]),
    format('~w               CpB0StackLen = List.length s.WsB0Stack~n', [Indent]),
    format('~w               CpAggFrame = None~n', [Indent]),
    format('~w               CpBuiltin  = Some (FFIStreamRetry (', [Indent]),
    emit_outregs_list_fs(OutputRegs),
    format(', outVars, restWrapped, retPC, returnCP, returnCutBar, returnB0Stack)) }~n', []),
    format('~w    Some { s1 with WsCPs = cp :: s1.WsCPs; WsCPsLen = s1.WsCPsLen + 1 }~n', [Indent]).

%% emit_bound_output_filter_fs(+OutputRegs, +NOuts, +Indent)
%  Drop native results that conflict with already-bound output registers.
emit_bound_output_filter_fs(OutputRegs, NOuts, Indent) :-
    format('~wlet results =~n', [Indent]),
    format('~w    results~n', [Indent]),
    format('~w    |> List.filter (fun ', [Indent]),
    emit_tuple_pattern_fs(NOuts),
    format(' ->~n', []),
    emit_bound_output_filter_body_fs(OutputRegs, 1, Indent),
    format('~w       )~n', [Indent]).

%% Single-output: a bare match expression. Multi-output: &&-chain of matches.
emit_bound_output_filter_body_fs([output(RegN, Type)], I, Indent) :- !,
    format('~w        match outReg_~w with~n', [Indent, RegN]),
    format('~w        | Unbound _ -> true~n', [Indent]),
    emit_bound_output_type_match_fs(Type, I, Indent),
    format('~w        | _ -> false~n', [Indent]).
emit_bound_output_filter_body_fs(OutputRegs, I, Indent) :-
    format('~w        (~n', [Indent]),
    emit_bound_output_filter_conds_fs(OutputRegs, I, Indent),
    emit_output_alias_filter_conds_fs(OutputRegs, Indent),
    format('~w        )~n', [Indent]).

emit_bound_output_filter_conds_fs([], _, Indent) :-
    format('~w         true~n', [Indent]).
emit_bound_output_filter_conds_fs([output(RegN, Type)|Rest], I, Indent) :-
    format('~w         (match outReg_~w with~n', [Indent, RegN]),
    format('~w          | Unbound _ -> true~n', [Indent]),
    % Multi-output matches are nested two spaces deeper than the
    % single-output form. Keep every case at the same offside column.
    atom_concat(Indent, '  ', NestedIndent),
    emit_bound_output_type_match_fs(Type, I, NestedIndent),
    format('~w          | _ -> false)~n', [Indent]),
    (   Rest = []
    ->  true
    ;   format('~w         &&~n', [Indent]),
        I1 is I + 1,
        emit_bound_output_filter_conds_fs(Rest, I1, Indent)
    ).

%% Aliased output registers must still obey ordinary Prolog unification.
%  The native tuple components are ground, so pairwise equality of their
%  wrapped WAM Values is sufficient. Without this guard td(+S,X,X) could
%  write an Atom to A2 and an Integer to A3 while adding the same variable id
%  twice to WsBindings. Filtering once here also makes every FFIStreamRetry
%  tuple safe for the existing direct retry binder.
emit_output_alias_filter_conds_fs(OutputRegs, Indent) :-
    index_output_regs_fs(OutputRegs, 1, Indexed),
    emit_output_alias_pairs_fs(Indexed, Indent).

index_output_regs_fs([], _, []).
index_output_regs_fs([output(RegN, Type)|Rest], I,
                     [indexed_output(I, RegN, Type)|IndexedRest]) :-
    I1 is I + 1,
    index_output_regs_fs(Rest, I1, IndexedRest).

emit_output_alias_pairs_fs([], _).
emit_output_alias_pairs_fs([Left|Rest], Indent) :-
    emit_output_alias_pairs_with_fs(Left, Rest, Indent),
    emit_output_alias_pairs_fs(Rest, Indent).

emit_output_alias_pairs_with_fs(_, [], _).
emit_output_alias_pairs_with_fs(indexed_output(I, RegN, Type),
                                [indexed_output(J, OtherRegN, OtherType)|Rest],
                                Indent) :-
    fsharp_wam_result_wrap_rv(Type, I, LeftValue),
    fsharp_wam_result_wrap_rv(OtherType, J, RightValue),
    format('~w         &&~n', [Indent]),
    format('~w         (match outReg_~w, outReg_~w with~n',
           [Indent, RegN, OtherRegN]),
    format('~w          | Unbound leftVar, Unbound rightVar when leftVar = rightVar ->~n',
           [Indent]),
    format('~w              (~w) = (~w)~n', [Indent, LeftValue, RightValue]),
    format('~w          | _ -> true)~n', [Indent]),
    emit_output_alias_pairs_with_fs(indexed_output(I, RegN, Type),
                                    Rest, Indent).

emit_bound_output_type_match_fs(atom, I, Indent) :- !,
    format('~w        | Atom t ->~n', [Indent]),
    format('~w            match Map.tryFind t ctx.WcAtomIntern with~n', [Indent]),
    format('~w            | Some id -> id = rv_~w~n', [Indent, I]),
    format('~w            | None -> false~n', [Indent]).
emit_bound_output_type_match_fs(integer, I, Indent) :- !,
    format('~w        | Integer i -> i = rv_~w~n', [Indent, I]).
emit_bound_output_type_match_fs(float, I, Indent) :- !,
    format('~w        | Float f -> f = rv_~w~n', [Indent, I]).
emit_bound_output_type_match_fs(_, _, _Indent).

emit_multi_out_derefs_fs([], _).
emit_multi_out_derefs_fs([output(RegN, _)|Rest], Indent) :-
    format('~wlet outReg_~w = s.WsRegs.[~w] |> derefVar s.WsBindings~n',
           [Indent, RegN, RegN]),
    emit_multi_out_derefs_fs(Rest, Indent).

emit_outreg_values_list_fs([], _).
emit_outreg_values_list_fs([output(RegN, _)|Rest], I) :-
    (   I =:= 1 -> true ; format('; ', []) ),
    format('outReg_~w', [RegN]),
    I1 is I + 1,
    emit_outreg_values_list_fs(Rest, I1).

emit_tuple_pattern_fs(1) :- format('rv_1', []).
emit_tuple_pattern_fs(N) :-
    N > 1,
    format('(', []),
    emit_tuple_pattern_args_fs(1, N),
    format(')', []).

emit_tuple_pattern_args_fs(I, N) :-
    I < N, !,
    format('rv_~w, ', [I]),
    I1 is I + 1,
    emit_tuple_pattern_args_fs(I1, N).
emit_tuple_pattern_args_fs(N, N) :- format('rv_~w', [N]).

emit_multi_wrap_bindings_fs([], _) :- format('~n', []).
emit_multi_wrap_bindings_fs([output(_, Type)|Rest], I) :-
    fsharp_wam_result_wrap_rv(Type, I, WrapExpr),
    (   I =:= 1 -> format('w_~w = ~w', [I, WrapExpr])
    ;              format('~n                let w_~w = ~w', [I, WrapExpr])
    ),
    I1 is I + 1,
    emit_multi_wrap_bindings_fs(Rest, I1).

emit_multi_reg_updates_fs(OutputRegs, Indent) :-
    format('~w             WsRegs = (let r = Array.copy s.WsRegs in ', [Indent]),
    emit_reg_set_chain_fs(OutputRegs, 1),
    format('r)~n', []).

emit_reg_set_chain_fs([], _).
emit_reg_set_chain_fs([output(RegN, _)|Rest], I) :-
    format('r.[~w] <- w_~w; ', [RegN, I]),
    I1 is I + 1,
    emit_reg_set_chain_fs(Rest, I1).

emit_multi_binding_updates_fs(OutputRegs, Indent) :-
    format('~w             WsBindings = ', [Indent]),
    emit_binding_add_chain_fs(OutputRegs, 1, 's.WsBindings'),
    format('~n', []).

emit_binding_add_chain_fs([], _, Base) :- format('~w', [Base]).
emit_binding_add_chain_fs([output(RegN, _)|Rest], I, Base) :-
    format('(match outReg_~w with | Unbound v -> Map.add v w_~w | _ -> id) ', [RegN, I]),
    I1 is I + 1,
    (   Rest = []
    ->  format('~w', [Base])
    ;   format('(', []),
        emit_binding_add_chain_fs(Rest, I1, Base),
        format(')', [])
    ).

emit_multi_trail_updates_fs(OutputRegs, Indent) :-
    format('~w             WsTrail = ', [Indent]),
    emit_trail_entry_chain_fs(OutputRegs, 1, 's.WsTrail'),
    format('~n', []).

emit_trail_entry_chain_fs([], _, Base) :- format('~w', [Base]).
emit_trail_entry_chain_fs([output(RegN, _)|Rest], I, Base) :-
    format('(match outReg_~w with | Unbound v -> (fun tl -> { TrailVarId = v; TrailOldVal = Map.tryFind v s.WsBindings } :: tl) | _ -> id) ', [RegN]),
    I1 is I + 1,
    (   Rest = []
    ->  format('~w', [Base])
    ;   format('(', []),
        emit_trail_entry_chain_fs(Rest, I1, Base),
        format(')', [])
    ).

emit_multi_outvars_fs(OutputRegs, Indent) :-
    format('~w    let outVars = [', [Indent]),
    emit_outvars_list_fs(OutputRegs, 1),
    format(']~n', []).

emit_outvars_list_fs([], _).
emit_outvars_list_fs([output(RegN, _)|Rest], I) :-
    (   I =:= 1 -> true ; format('; ', []) ),
    % Parenthesize each match: bare `match ... | _ -> -1; match ...`
    % is parsed as one list element (`;` sequences inside the last arm),
    % which breaks List.fold2/iter2 in FFIStreamRetry for multi-output.
    format('(match outReg_~w with | Unbound v -> v | _ -> -1)', [RegN]),
    I1 is I + 1,
    emit_outvars_list_fs(Rest, I1).

emit_outregs_list_fs(OutputRegs) :-
    format('[', []),
    emit_outregs_list_items_fs(OutputRegs, 1),
    format(']', []).

emit_outregs_list_items_fs([], _).
emit_outregs_list_items_fs([output(RegN, _)|Rest], I) :-
    (   I =:= 1 -> true ; format('; ', []) ),
    format('~w', [RegN]),
    I1 is I + 1,
    emit_outregs_list_items_fs(Rest, I1).

emit_multi_wrap_list_fs([], _).
emit_multi_wrap_list_fs([output(_, Type)|Rest], I) :-
    fsharp_wam_result_wrap_rv(Type, I, WrapExpr),
    (   I =:= 1 -> true ; format('; ', []) ),
    format('~w', [WrapExpr]),
    I1 is I + 1,
    emit_multi_wrap_list_fs(Rest, I1).

% ============================================================================
% compile_wam_predicate_to_fsharp/4
% ============================================================================

%% compile_wam_predicate_to_fsharp(+PredIndicator, +WamCode, +Options, -Code)
%  Converts WAM assembly text for a single predicate into F# source defining
%  an Instruction list and a labels map.  Mirrors wam_haskell_target's
%  compile_wam_predicate_to_haskell/4 but emits F# discriminated-union syntax.
compile_wam_predicate_to_fsharp(PredIndicator, WamCode, Options, Code) :-
    (   PredIndicator = _M:Pred/Arity -> true ; PredIndicator = Pred/Arity ),
    (   string(WamCode) -> WamStr = WamCode ; atom_string(WamCode, WamStr) ),
    (   member(base_pc(BasePC), Options) -> true ; BasePC = 1 ),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_fsharp(Lines, BasePC, InstrExprs, LabelExprs),
    atomic_list_concat(InstrExprs, '\n      ; ', InstrCode),
    atomic_list_concat(LabelExprs, '; ', LabelCode),
    atom_string(Pred, PredStr),
    atomic_list_concat(PredParts, '$', PredStr),
    atomic_list_concat(PredParts, '_', PredSafe),
    format(atom(FuncName), '~w_~w', [PredSafe, Arity]),
    format(string(Code),
'// ~w/~w
let ~w_code : Instruction list =
    [ ~w
    ]

let ~w_labels : Map<string, int> =
    Map.ofList [ ~w ]
', [Pred, Arity, FuncName, InstrCode, FuncName, LabelCode]).

%% fs_split_wam_line(+Line, -Tokens)
%
% Split a WAM-text line into tokens, treating single-quoted atoms
% (`' '`, `'foo bar'`, etc.) as one token even when their contents
% include whitespace or commas.  The previous splitter
% (`split_string(Line, " \\t,", ...)`) chopped through quoted
% atoms — `get_constant ' ', A2` came out as
% `["get_constant", "'", "'", "A2"]` and missed every emitter
% clause, falling into the UNKNOWN `Proceed` stub.  The portable
% Prolog parser uses `' '` and similar literally in its whitespace
% / character-class tables, so the parser predicates were all
% silently no-op'd.
fs_split_wam_line(Line, Tokens) :-
    atom_string(LineA, Line),
    atom_chars(LineA, Chars),
    fs_tokenize_chars(Chars, [], [], Toks0),
    reverse(Toks0, Toks1),
    maplist([Cs, S]>>(atom_chars(A, Cs), atom_string(A, S)), Toks1, Tokens).

% fs_tokenize_chars(+Chars, +CurrentTokenAcc, +TokenListAcc, -Result)
% CurrentTokenAcc is a reversed list of chars.  When a separator
% is seen outside a quote, flush the current token.
fs_tokenize_chars([], [], Toks, Toks) :- !.
fs_tokenize_chars([], Cur, Toks, [Tok|Toks]) :-
    reverse(Cur, Tok).
fs_tokenize_chars([C|Rest], Cur, Toks, Out) :-
    (   fs_wam_separator_char(C)
    ->  (   Cur == []
        ->  fs_tokenize_chars(Rest, [], Toks, Out)
        ;   reverse(Cur, Tok),
            fs_tokenize_chars(Rest, [], [Tok|Toks], Out)
        )
    ;   C == ''''
    ->  fs_tokenize_quoted(Rest, [''''|Cur], QCur, QRest),
        fs_tokenize_chars(QRest, QCur, Toks, Out)
    ;   fs_tokenize_chars(Rest, [C|Cur], Toks, Out)
    ).

% Inside a single-quoted atom — keep every char (including the
% closing quote) attached to the current token, treat doubled
% quotes `''` as an escaped quote inside the atom (ISO).
fs_tokenize_quoted([], Cur, Cur, []).
fs_tokenize_quoted([''''|Rest], Cur, OutCur, OutRest) :-
    (   Rest = [''''|Rest1]
    ->  fs_tokenize_quoted(Rest1, [''''|[''''|Cur]], OutCur, OutRest)
    ;   OutCur = [''''|Cur],
        OutRest = Rest
    ).
fs_tokenize_quoted([C|Rest], Cur, OutCur, OutRest) :-
    C \== '''',
    fs_tokenize_quoted(Rest, [C|Cur], OutCur, OutRest).

fs_wam_separator_char(' ').
fs_wam_separator_char('\t').
fs_wam_separator_char(',').

%% wam_lines_to_fsharp(+Lines, +PC, -InstrExprs, -LabelExprs)
wam_lines_to_fsharp([], _, [], []).
wam_lines_to_fsharp([Line|Rest], PC, Instrs, Labels) :-
    fs_split_wam_line(Line, Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_fsharp(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LExpr), '("~w", ~w)', [LabelName, PC]),
            Labels = [LExpr|LR],
            wam_lines_to_fsharp(Rest, PC, Instrs, LR)
        ;   wam_instr_to_fsharp(CleanParts, FsExpr),
            NPC is PC + 1,
            Instrs = [FsExpr|IR],
            wam_lines_to_fsharp(Rest, NPC, IR, Labels)
        )
    ).

%% fs_reg_name_to_int(+RegName, -Int)
%  A1-A99 → 1-99, X1-X99 → 101-199, Y1-Y99 → 201-299.
fs_reg_name_to_int(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, Bank),
    sub_atom(RegA, 1, _, 0, NumA),
    atom_number(NumA, Num),
    (   Bank == 'A' -> Int = Num
    ;   Bank == 'X' -> Int is Num + 100
    ;   Bank == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).

%% fs_clean_comma(+Str, -Clean) — strip trailing comma
fs_clean_comma(Str, Clean) :-
    (   sub_string(Str, _, 1, 0, ",")
    ->  sub_string(Str, 0, _, 1, Clean)
    ;   Clean = Str
    ).

%% fs_wam_value(+WamVal, -FsExpr)
%  WamVal may arrive as a string (typical) or an atom (e.g., from
%  sub_atom/5 inside fs_parse_switch_entries).  Normalize to a string
%  before calling number_string/2, which throws type_error(list, _)
%  when given an atom.
fs_wam_value(Val0, Fs) :-
    (   string(Val0) -> Val1 = Val0
    ;   atom(Val0)   -> atom_string(Val0, Val1)
    ;   number(Val0) -> number_string(Val0, Val1)
    ;   Val1 = Val0
    ),
    %% Atom-vs-number disambiguation: fs_split_wam_line
    %% preserves the outer single quotes attached to the token
    %% (so `'+'` stays distinct from the unquoted `+`, and
    %% `'5'` stays distinct from the integer 5).
    %% fs_strip_quoted_atom/3 returns ForceAtom=true iff the
    %% token had outer quotes; when set, emit an Atom
    %% regardless of whether the inner content reparses as a
    %% number.
    fs_strip_quoted_atom(Val1, Val, ForceAtom),
    (   ForceAtom == true
    ->  fs_escape_string_for_fsharp(Val, Escaped),
        format(string(Fs), 'Atom "~w"', [Escaped])
    ;   number_string(N, Val), integer(N)
    ->  format(string(Fs), 'Integer ~w', [N])
    ;   number_string(F, Val), float(F)
    ->  format(string(Fs), 'Float ~w', [F])
    ;   Val == "[]"
    ->  Fs = "Atom \"[]\""
    ;   fs_escape_string_for_fsharp(Val, Escaped),
        format(string(Fs), 'Atom "~w"', [Escaped])
    ).

%% fs_strip_quoted_atom(+S0, -S, -ForceAtom)
%
%  Strip surrounding single quotes, returning the inner atom
%  name in S. ForceAtom = true iff the token had outer single
%  quotes -- the quote-state-preserving convention from
%  wam_target:quote_wam_constant/2 (and mirrored by
%  fs_split_wam_line / wam_text_parser:wam_tokenize_line/2) is
%  the source of atom-vs-number truth: bare `5` is the integer;
%  quoted `'5'` is the atom.
fs_strip_quoted_atom(S0, S, ForceAtom) :-
    string_chars(S0, Chars0),
    (   Chars0 = [''''|Rest], append(Inner, [''''], Rest)
    ->  fs_unescape_quoted(Inner, InnerU),
        string_chars(S, InnerU),
        ForceAtom = true
    ;   ForceAtom = false,
        S = S0
    ).

fs_unescape_quoted([], []).
fs_unescape_quoted([''''|[''''|Rest]], [''''|Out]) :-
    !,
    fs_unescape_quoted(Rest, Out).
fs_unescape_quoted([C|Rest], [C|Out]) :-
    fs_unescape_quoted(Rest, Out).

%% Escape special F#-string chars inside an atom name so
%% `Atom "foo\"bar"` parses correctly.  Backslash and double-quote
%% are the two that matter for F# `"..."` literals.
fs_escape_string_for_fsharp(S0, S) :-
    string_chars(S0, Chars0),
    fs_escape_fs_chars(Chars0, Chars1),
    string_chars(S, Chars1).

fs_escape_fs_chars([], []).
fs_escape_fs_chars(['\\'|Rest], ['\\','\\'|Out]) :-
    !, fs_escape_fs_chars(Rest, Out).
fs_escape_fs_chars(['"'|Rest], ['\\','"'|Out]) :-
    !, fs_escape_fs_chars(Rest, Out).
fs_escape_fs_chars([C|Rest], [C|Out]) :-
    fs_escape_fs_chars(Rest, Out).

%% fs_parse_functor(+FunctorString, -Name, -Arity)
%  Extract name and arity from "name/N" format.  E.g. "-/1" → ("-", 1).
fs_parse_functor(FN, Name, Arity) :-
    atom_string(FNA, FN),
    (   sub_atom(FNA, Before, 1, _, '/'),
        sub_atom(FNA, 0, Before, _, Name),
        After is Before + 1,
        sub_atom(FNA, After, _, 0, ArityStr),
        atom_number(ArityStr, Arity)
    ->  true
    ;   Name = FNA, Arity = 0
    ).

%% fs_escape_string(+In, -Out) — escape backslashes for F# string literals
fs_escape_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", Out).

%% wam_instr_to_fsharp(+Parts, -FSharpExpr)
wam_instr_to_fsharp(["get_constant", C, Ai], Fs) :-
    fs_clean_comma(C, CC), fs_clean_comma(Ai, CAi),
    fs_wam_value(CC, FsVal), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetConstant (~w, ~w)', [FsVal, AiI]).
wam_instr_to_fsharp(["get_variable", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetVariable (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["get_value", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetValue (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["get_structure", FN, Ai], Fs) :-
    %% Three-token form emitted by wam_target.pl: `get_structure F/N, Ai`.
    %% Functor and arity are joined with `/`; parse them out the same
    %% way `put_structure` does (the WAM compiler emits both in the
    %% same shape).  The previous 4-token clause never matched any
    %% real WAM output and silently fell through to the UNKNOWN
    %% stub, so any predicate doing compound-term matching in its
    %% head (the portable parser's `op/3` tables, list cells, etc.)
    %% executed as a no-op.
    fs_clean_comma(FN, CFN), fs_clean_comma(Ai, CAi),
    fs_parse_functor(CFN, FuncName, FsArity),
    fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetStructure ("~w", ~w, ~w)', [FuncName, FsArity, AiI]).
wam_instr_to_fsharp(["get_list", Ai], Fs) :-
    fs_clean_comma(Ai, CAi), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetList ~w', [AiI]).
wam_instr_to_fsharp(["get_nil", Ai], Fs) :-
    fs_clean_comma(Ai, CAi), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetConstant (Atom "[]", ~w)', [AiI]).
wam_instr_to_fsharp(["get_integer", N, Ai], Fs) :-
    fs_clean_comma(N, CN), fs_clean_comma(Ai, CAi),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'GetConstant (Integer ~w, ~w)', [Num, AiI]).
wam_instr_to_fsharp(["unify_variable", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'UnifyVariable ~w', [XnI]).
wam_instr_to_fsharp(["unify_value", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'UnifyValue ~w', [XnI]).
wam_instr_to_fsharp(["unify_constant", C], Fs) :-
    fs_wam_value(C, FsVal),
    format(string(Fs), 'UnifyConstant (~w)', [FsVal]).
wam_instr_to_fsharp(["put_constant", C, Ai], Fs) :-
    fs_clean_comma(C, CC), fs_clean_comma(Ai, CAi),
    fs_wam_value(CC, FsVal), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutConstant (~w, ~w)', [FsVal, AiI]).
wam_instr_to_fsharp(["put_variable", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutVariable (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["put_value", Xn, Ai], Fs) :-
    fs_clean_comma(Xn, CXn), fs_clean_comma(Ai, CAi),
    fs_reg_name_to_int(CXn, XnI), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutValue (~w, ~w)', [XnI, AiI]).
wam_instr_to_fsharp(["put_structure", FN, Ai], Fs) :-
    fs_clean_comma(FN, CFN), fs_clean_comma(Ai, CAi),
    fs_parse_functor(CFN, FuncName, FsArity),
    fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutStructure ("~w", ~w, ~w)', [FuncName, AiI, FsArity]).
wam_instr_to_fsharp(["put_list", Ai], Fs) :-
    fs_clean_comma(Ai, CAi), fs_reg_name_to_int(CAi, AiI),
    format(string(Fs), 'PutList ~w', [AiI]).
wam_instr_to_fsharp(["set_value", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'SetValue ~w', [XnI]).
wam_instr_to_fsharp(["set_variable", Xn], Fs) :-
    fs_clean_comma(Xn, CXn), fs_reg_name_to_int(CXn, XnI),
    format(string(Fs), 'SetVariable ~w', [XnI]).
wam_instr_to_fsharp(["set_constant", C], Fs) :-
    fs_wam_value(C, FsVal),
    format(string(Fs), 'SetConstant (~w)', [FsVal]).
wam_instr_to_fsharp(["allocate"], "Allocate").
wam_instr_to_fsharp(["deallocate"], "Deallocate").
wam_instr_to_fsharp(["call", P, N], Fs) :-
    fs_clean_comma(P, CP), fs_clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    format(string(Fs), 'Call ("~w", ~w)', [CP, Num]).
wam_instr_to_fsharp(["call_foreign", P, N], Fs) :-
    fs_clean_comma(P, CP), fs_clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    format(string(Fs), 'CallForeign ("~w", ~w)', [CP, Num]).
wam_instr_to_fsharp(["execute", P], Fs) :-
    format(string(Fs), 'Execute "~w"', [P]).
wam_instr_to_fsharp(["proceed"], "Proceed").
wam_instr_to_fsharp(["fail"], "Fail").
wam_instr_to_fsharp(["jump", Label], Fs) :-
    format(string(Fs), 'Jump "~w"', [Label]).
wam_instr_to_fsharp(["cut_ite"], "CutIte").
wam_instr_to_fsharp(["builtin_call", Op, N], Fs) :-
    fs_clean_comma(Op, COp), fs_clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; throw(error(domain_error(wam_integer, CN), wam_instr_to_fsharp/2)) ),
    fs_escape_string(COp, ECOp),
    format(string(Fs), 'BuiltinCall ("~w", ~w)', [ECOp, Num]).
wam_instr_to_fsharp(["try_me_else", Label], Fs) :-
    format(string(Fs), 'TryMeElse "~w"', [Label]).
wam_instr_to_fsharp(["retry_me_else", Label], Fs) :-
    format(string(Fs), 'RetryMeElse "~w"', [Label]).
wam_instr_to_fsharp(["trust_me"], "TrustMe").
%% Indexed-dispatch chain ops (try / retry / trust without _me_else).
%% Emitted by wam_target.pl into try/retry/trust chains synthesized
%% for switch_on_term / switch_on_constant / switch_on_structure
%% targets whose dispatch group has >1 matching clauses.  Semantics:
%%   try   L : push CP with CpNextPC = PC+1 (the next chain
%%             instruction), CpRegs/etc. = current; jump to L.
%%   retry L : modify top CP's CpNextPC = PC+1; jump to L.
%%   trust L : pop top CP; jump to L.
%% See issue #2400 for the motivating bug and the docs at the
%% Try*Pc step cases in the generated F# runtime.
wam_instr_to_fsharp(["try", Label], Fs) :-
    format(string(Fs), 'Try "~w"', [Label]).
wam_instr_to_fsharp(["retry", Label], Fs) :-
    format(string(Fs), 'Retry "~w"', [Label]).
wam_instr_to_fsharp(["trust", Label], Fs) :-
    format(string(Fs), 'Trust "~w"', [Label]).
wam_instr_to_fsharp(["switch_on_constant"|Entries], Fs) :-
    fs_parse_switch_entries(Entries, FsPairs),
    atomic_list_concat(FsPairs, '; ', PairsStr),
    format(string(Fs), 'SwitchOnConstant (Map.ofList [~w])', [PairsStr]).
wam_instr_to_fsharp(["switch_on_constant_a2"|Entries], Fs) :-
    fs_parse_switch_entries(Entries, FsPairs),
    atomic_list_concat(FsPairs, '; ', PairsStr),
    format(string(Fs), 'SwitchOnConstant (Map.ofList [~w])', [PairsStr]).
%% switch_on_constant_fallthrough is shape-compatible with
%% switch_on_constant -- both are "match A1 against constants, jump
%% to label on hit".  The runtime difference (fall through vs fail
%% on miss) is already the SwitchOnConstantPc semantics in the
%% generated step function, so re-using SwitchOnConstant here is
%% the right thing.  Emitted when the WAM compiler has a constant
%% prefix followed by a variable clause; the variable clauses are
%% reached via the try_me_else chain past PC+1.
wam_instr_to_fsharp(["switch_on_constant_fallthrough"|Entries], Fs) :-
    fs_parse_switch_entries(Entries, FsPairs),
    atomic_list_concat(FsPairs, '; ', PairsStr),
    format(string(Fs), 'SwitchOnConstant (Map.ofList [~w])', [PairsStr]).
%% Symmetric A2 variant -- same handling.
wam_instr_to_fsharp(["switch_on_constant_a2_fallthrough"|Entries], Fs) :-
    fs_parse_switch_entries(Entries, FsPairs),
    atomic_list_concat(FsPairs, '; ', PairsStr),
    format(string(Fs), 'SwitchOnConstant (Map.ofList [~w])', [PairsStr]).
%% switch_on_term: type-based dispatch on A1.
%% Format: switch_on_term CLen ConstEntries... SLen StructEntries... ListLabel
%% emitted by format_switch_on_term/4 in wam_target.pl.  Entries are
%% `key:label` strings; "default" entries refer to the first
%% clause and are stripped by resolveCallInstrs at load time.
wam_instr_to_fsharp(["switch_on_term", CLenS | Rest], Fs) :-
    fs_clean_comma(CLenS, CLenT),
    (   number_string(CLen, CLenT)
    ->  true
    ;   throw(error(domain_error(wam_integer, CLenT), wam_instr_to_fsharp/2))
    ),
    length(Const, CLen),
    append(Const, [SLenS|Rest2], Rest),
    fs_clean_comma(SLenS, SLenT),
    (   number_string(SLen, SLenT)
    ->  true
    ;   throw(error(domain_error(wam_integer, SLenT), wam_instr_to_fsharp/2))
    ),
    length(Struct, SLen),
    append(Struct, [ListLabel0], Rest2),
    fs_clean_comma(ListLabel0, ListLabel),
    fs_parse_switch_term_entries(Const, ConstPairs),
    fs_parse_switch_term_entries(Struct, StructPairs),
    fs_term_entries_to_array_literal(ConstPairs, ConstArr),
    fs_term_entries_to_array_literal(StructPairs, StructArr),
    format(string(Fs), 'SwitchOnTerm (~w, ~w, "~w")',
           [ConstArr, StructArr, ListLabel]).
%% A2 variant -- same shape, dispatches on A2 rather than A1.  The
%% runtime currently treats A1 and A2 identically (the step handler
%% inspects register 1); a proper A2 variant is future work.
wam_instr_to_fsharp(["switch_on_term_a2", CLenS | Rest], Fs) :-
    fs_clean_comma(CLenS, CLenT),
    (   number_string(CLen, CLenT)
    ->  true
    ;   throw(error(domain_error(wam_integer, CLenT), wam_instr_to_fsharp/2))
    ),
    length(Const, CLen),
    append(Const, [SLenS|Rest2], Rest),
    fs_clean_comma(SLenS, SLenT),
    (   number_string(SLen, SLenT)
    ->  true
    ;   throw(error(domain_error(wam_integer, SLenT), wam_instr_to_fsharp/2))
    ),
    length(Struct, SLen),
    append(Struct, [ListLabel0], Rest2),
    fs_clean_comma(ListLabel0, ListLabel),
    fs_parse_switch_term_entries(Const, ConstPairs),
    fs_parse_switch_term_entries(Struct, StructPairs),
    fs_term_entries_to_array_literal(ConstPairs, ConstArr),
    fs_term_entries_to_array_literal(StructPairs, StructArr),
    format(string(Fs), 'SwitchOnTerm (~w, ~w, "~w")',
           [ConstArr, StructArr, ListLabel]).

wam_instr_to_fsharp(["begin_aggregate", Type, ValReg, ResReg], Fs) :-
    fs_clean_comma(Type, CT), fs_clean_comma(ValReg, CV), fs_clean_comma(ResReg, CR),
    fs_reg_name_to_int(CV, VI), fs_reg_name_to_int(CR, RI),
    format(string(Fs), 'BeginAggregate ("~w", ~w, ~w)', [CT, VI, RI]).
wam_instr_to_fsharp(["end_aggregate", ValReg], Fs) :-
    fs_clean_comma(ValReg, CV),
    fs_reg_name_to_int(CV, VI),
    format(string(Fs), 'EndAggregate ~w', [VI]).

% ----------------------------------------------------------------------
% Phase I — Haskell-only specialized instructions (text parse rules).
% Match the WAM-text mnemonics emitted by the WAM compiler''s
% optimization pass, mirroring the Haskell parser rules in
% wam_haskell_target.pl wam_instr_to_haskell.
% ----------------------------------------------------------------------

wam_instr_to_fsharp(["put_structure_dyn", NameReg, ArityReg, TargetReg], Fs) :-
    fs_clean_comma(NameReg, CN), fs_clean_comma(ArityReg, CA), fs_clean_comma(TargetReg, CT),
    fs_reg_name_to_int(CN, NI), fs_reg_name_to_int(CA, AI), fs_reg_name_to_int(CT, TI),
    format(string(Fs), 'PutStructureDyn (~w, ~w, ~w)', [NI, AI, TI]).

wam_instr_to_fsharp(["arg", N, TReg, AReg], Fs) :-
    fs_clean_comma(N, CN), fs_clean_comma(TReg, CT), fs_clean_comma(AReg, CA),
    (   number_string(NI, CN)
    ->  true
    ;   throw(error(domain_error(arg_specialization_n, CN), wam_instr_to_fsharp/2))
    ),
    fs_reg_name_to_int(CT, TI), fs_reg_name_to_int(CA, AI),
    format(string(Fs), 'Arg (~w, ~w, ~w)', [NI, TI, AI]).

wam_instr_to_fsharp(["not_member_list", XReg, LReg], Fs) :-
    fs_clean_comma(XReg, CX), fs_clean_comma(LReg, CL),
    fs_reg_name_to_int(CX, XI), fs_reg_name_to_int(CL, LI),
    format(string(Fs), 'NotMemberList (~w, ~w)', [XI, LI]).

%% Variable-arity: not_member_const_atoms XReg Atom1 Atom2 ... AtomN
%% Emits an F# list literal of atom strings.
wam_instr_to_fsharp(["not_member_const_atoms", XReg | AtomTokens], Fs) :-
    AtomTokens \= [],
    fs_clean_comma(XReg, CX), fs_reg_name_to_int(CX, XI),
    maplist([Tok, Quoted]>>(
        fs_clean_comma(Tok, CTok),
        fs_escape_string(CTok, EscTok),
        format(atom(Quoted), '"~w"', [EscTok])
    ), AtomTokens, QuotedAtoms),
    atomic_list_concat(QuotedAtoms, '; ', AtomsList),
    format(string(Fs), 'NotMemberConstAtoms (~w, [~w])', [XI, AtomsList]).

wam_instr_to_fsharp(["build_empty_set", Reg], Fs) :-
    fs_clean_comma(Reg, CR), fs_reg_name_to_int(CR, RI),
    format(string(Fs), 'BuildEmptySet ~w', [RI]).

wam_instr_to_fsharp(["set_insert", EReg, InReg, OutReg], Fs) :-
    fs_clean_comma(EReg, CE), fs_clean_comma(InReg, CI), fs_clean_comma(OutReg, CO),
    fs_reg_name_to_int(CE, EI), fs_reg_name_to_int(CI, II), fs_reg_name_to_int(CO, OI),
    format(string(Fs), 'SetInsert (~w, ~w, ~w)', [EI, II, OI]).

wam_instr_to_fsharp(["not_member_set", EReg, SReg], Fs) :-
    fs_clean_comma(EReg, CE), fs_clean_comma(SReg, CS),
    fs_reg_name_to_int(CE, EI), fs_reg_name_to_int(CS, SI),
    format(string(Fs), 'NotMemberSet (~w, ~w)', [EI, SI]).

% Fallback for unknown instructions.
%
% The F# WAM emitter doesn't recognize every WAM instruction the
% shared compiler can produce -- specifically `get_structure`,
% certain `unify_constant`/`get_constant` forms, and the
% `switch_on_term` / `switch_on_constant_fallthrough` variants
% used by indexed dispatch -- and silently replaced them with
% `Proceed`, which lets the predicate "compile" but breaks runtime
% semantics (predicate returns immediately without doing the
% work).  This was found while wiring the compiled portable Prolog
% parser through F# (`runtime_parser(compiled)`): the parser's WAM
% has ~120 lines emitted as `Proceed` stubs across `get_structure`
% (op/3, [|]/2, tk_atom/1, tk_num/1, tk_sym/1, tk_var/1, -/2),
% `get_constant ' '`, `unify_constant ' '`, and four
% `switch_on_*` variants.
%
% Until the emitter learns these instructions, surface the gap on
% stderr at codegen time rather than letting it pretend to compile
% cleanly.  Deduplicated per (head-token, arity) so a noisy
% predicate doesn't drown the output.
:- dynamic wam_fsharp_unknown_seen/2.

wam_instr_to_fsharp(Parts, Fs) :-
    atomic_list_concat(Parts, ' ', Joined),
    (   Parts = [Head|_], length(Parts, Len)
    ->  (   wam_fsharp_unknown_seen(Head, Len)
        ->  true
        ;   assertz(wam_fsharp_unknown_seen(Head, Len)),
            format(user_error,
                   '[WAM-FSharp] WARNING: instruction not supported by emitter -> emitting Proceed stub: ~w~n',
                   [Joined])
        )
    ;   true
    ),
    format(string(Fs), '(* UNKNOWN: ~w *) Proceed', [Joined]).

%% fs_parse_switch_term_entries(+Entries, -Pairs)
%
% Parse "key:label" string entries into (Key, Label) pairs.  Mirrors
% fs_parse_switch_entries but returns string keys (no value-typing)
% because SwitchOnTerm always keys by atom-name or "F/N" string.
% Placed after the wam_instr_to_fsharp clause group so the latter is
% contiguous; callers are the switch_on_term / switch_on_term_a2
% clauses above.
fs_parse_switch_term_entries([], []).
fs_parse_switch_term_entries([E|Rest], [(K, L)|Pairs]) :-
    fs_clean_comma(E, ECl),
    atom_string(EA, ECl),
    (   sub_atom(EA, Before, 1, _, ':')
    ->  sub_atom(EA, 0, Before, _, KA),
        After is Before + 1,
        sub_atom(EA, After, _, 0, LA),
        atom_string(KA, K),
        atom_string(LA, L)
    ;   K = ECl,
        L = "default"
    ),
    fs_parse_switch_term_entries(Rest, Pairs).

fs_term_entries_to_array_literal([], "[||]") :- !.
fs_term_entries_to_array_literal(Pairs, Literal) :-
    maplist([(K, L), S]>>format(string(S), '("~w", "~w")', [K, L]), Pairs, Strs),
    atomic_list_concat(Strs, '; ', Joined),
    format(string(Literal), '[| ~w |]', [Joined]).

%% fs_parse_switch_entries(+Entries, -FSharpPairs)
fs_parse_switch_entries([], []).
fs_parse_switch_entries([Entry|Rest], [FsPair|FsRest]) :-
    fs_clean_comma(Entry, CEntry),
    (   sub_atom(CEntry, Before, 1, _, ':')
    ->  sub_atom(CEntry, 0, Before, _, Key),
        After is Before + 1,
        sub_atom(CEntry, After, _, 0, Label),
        fs_wam_value(Key, FsKey),
        format(string(FsPair), '(~w, "~w")', [FsKey, Label])
    ;   format(string(FsPair), '(Atom "~w", "default")', [CEntry])
    ),
    fs_parse_switch_entries(Rest, FsRest).

% ============================================================================
% write_wam_fsharp_project/3 — Project Generation
% ============================================================================

%% Runtime parser integration helpers ---------------------------------------
%
% F# adopts the `compiled(prolog_term_parser)` runtime-parser mode
% from `wam_runtime_parser_capability.pl`, mirroring Python's
% approach.  When that mode is selected, the portable parser
% predicates (defined in `src/unifyweaver/core/prolog_term_parser.pl`)
% and the target-agnostic wrappers (`read_term_from_atom/2,3` etc.,
% from `src/unifyweaver/core/cpp_runtime_parser_wrappers.pl`) are
% appended to the user's predicate list so they're compiled to WAM
% and emitted into `Predicates.fs`.  When `none` is selected, any
% predicate whose body has a statically visible parser-dependent
% call is rejected here rather than silently producing a runtime
% that lacks the underlying parsing capability.

%% fsharp_project_predicates(+UserPreds, +Mode, -ProjectPreds)
%
%  In `compiled` mode, append the portable parser + wrapper
%  predicate indicators.  In other modes, leave the list untouched.
fsharp_project_predicates(Predicates, compiled(prolog_term_parser), ProjectPredicates) :-
    !,
    fsharp_runtime_parser_predicates(ParserPreds),
    fsharp_runtime_parser_wrapper_predicates(WrapperPreds),
    append([Predicates, ParserPreds, WrapperPreds], Combined),
    sort(Combined, ProjectPredicates).
fsharp_project_predicates(Predicates, _RuntimeParserMode, Predicates).

%% fsharp_runtime_parser_predicates(-Predicates)
%
%  Enumerate non-imported predicates of `prolog_term_parser` as
%  `prolog_term_parser:Name/Arity` indicators.  Matches Python's
%  approach in `python_runtime_parser_predicates/1`.
fsharp_runtime_parser_predicates(Predicates) :-
    findall(prolog_term_parser:Name/Arity,
            ( current_predicate(prolog_term_parser:Name/Arity),
              functor(Head, Name, Arity),
              once(clause(prolog_term_parser:Head, _)),
              \+ predicate_property(prolog_term_parser:Head, imported_from(_))
            ),
            Raw),
    sort(Raw, Predicates).

%% fsharp_runtime_parser_wrapper_predicates(-Predicates)
%
%  Enumerate non-imported predicates of `cpp_runtime_parser_wrappers`
%  (the module name is C++-historical but the wrappers themselves are
%  target-agnostic per the module-header comment).
fsharp_runtime_parser_wrapper_predicates(Predicates) :-
    findall(cpp_runtime_parser_wrappers:Name/Arity,
            ( current_predicate(cpp_runtime_parser_wrappers:Name/Arity),
              functor(Head, Name, Arity),
              once(clause(cpp_runtime_parser_wrappers:Head, _)),
              \+ predicate_property(cpp_runtime_parser_wrappers:Head,
                                    imported_from(_))
            ),
            Raw),
    sort(Raw, Predicates).

%% fsharp_validate_runtime_parser_mode(+Predicates, +Mode)
%
%  When the mode is `none`, reject any user predicate whose body
%  statically calls a parser-dependent builtin.  Mirrors
%  `validate_python_runtime_parser_mode/2`.
fsharp_validate_runtime_parser_mode(Predicates, none) :-
    !,
    (   fsharp_predicates_parser_dependency(Predicates, Pred, Builtin)
    ->  throw(error(permission_error(use, runtime_parser, Builtin),
                    context(write_wam_fsharp_project/3,
                            parser_disabled_for_predicate(Pred))))
    ;   true
    ).
fsharp_validate_runtime_parser_mode(_Predicates, _Mode).

fsharp_predicates_parser_dependency(Predicates, Pred, Builtin) :-
    member(Pred, Predicates),
    fsharp_predicate_clause(Pred, _Head, Body),
    parser_dependent_body_goal(Body, Builtin),
    !.

fsharp_predicate_clause(Module:Name/Arity, Head, Body) :-
    !,
    functor(Head, Name, Arity),
    clause(Module:Head, Body).
fsharp_predicate_clause(Name/Arity, Head, Body) :-
    functor(Head, Name, Arity),
    clause(user:Head, Body).

% ============================================================================
% PHASE: ISO error configuration, rewrite, and audit
% ============================================================================
%
% Shared contract: docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md +
%                   docs/design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md
%
% The reusable helpers (option resolution, mode lookup, config-file
% loader, multi-module warning, item-level rewrite, audit walker)
% live in src/unifyweaver/core/iso_errors.pl and are re-exported by
% the use_module at the top of this file.  What stays here:
%   - F# key tables (multifile facts asserted into iso_errors).
%   - Text-level rewrite (Python/F# parse style; Elixir uses a
%     slightly different parser that hasn''t been reconciled yet).
%   - wam_fsharp_iso_audit/3 itself plus its parse-lines helpers.
%
% Adding an entry to the key tables without a matching runtime
% branch in step_function_fsharp silently rewrites default calls to
% dead keys.  F# currently ships is/2, the six arithmetic-compare
% ops, and succ/2.

% Multifile dispatch tables -- assert into iso_errors so the shared
% mode/audit helpers see our entries.
iso_errors:iso_errors_default_to_iso("is/2", "is_iso/2").
iso_errors:iso_errors_default_to_iso(">/2", ">_iso/2").
iso_errors:iso_errors_default_to_iso("</2", "<_iso/2").
iso_errors:iso_errors_default_to_iso(">=/2", ">=_iso/2").
iso_errors:iso_errors_default_to_iso("=</2", "=<_iso/2").
iso_errors:iso_errors_default_to_iso("=:=/2", "=:=_iso/2").
iso_errors:iso_errors_default_to_iso("=\\=/2", "=\\=_iso/2").
iso_errors:iso_errors_default_to_iso("succ/2", "succ_iso/2").

iso_errors:iso_errors_default_to_lax("is/2", "is_lax/2").
iso_errors:iso_errors_default_to_lax(">/2", ">_lax/2").
iso_errors:iso_errors_default_to_lax("</2", "<_lax/2").
iso_errors:iso_errors_default_to_lax(">=/2", ">=_lax/2").
iso_errors:iso_errors_default_to_lax("=</2", "=<_lax/2").
iso_errors:iso_errors_default_to_lax("=:=/2", "=:=_lax/2").
iso_errors:iso_errors_default_to_lax("=\\=/2", "=\\=_lax/2").
iso_errors:iso_errors_default_to_lax("succ/2", "succ_lax/2").

%% iso_errors_rewrite_text(+Config, +PI, +WamText, -RewrittenText)
%  Text-level rewrite that walks the WAM text line-by-line, splicing
%  default-form keys into their resolved ISO/lax forms.  Mirrors
%  Python wam_python_target:iso_errors_rewrite_text/4.
iso_errors_rewrite_text(Config, PI, WamText, RewrittenText) :-
    iso_errors_mode_for(Config, PI, Mode),
    (   Mode == false,
        \+ iso_errors_has_lax_entries
    ->  RewrittenText = WamText
    ;   atom_string(WamText, S),
        split_string(S, "\n", "", Lines),
        maplist(iso_errors_rewrite_line(Mode), Lines, RewrittenLines),
        atomic_list_concat(RewrittenLines, '\n', RewrittenText)
    ).

iso_errors_has_lax_entries :- iso_errors:iso_errors_default_to_lax(_, _), !.

% Rewrite one line via the shared items pipeline. Tokenize the line
% with the shared quote-aware tokenizer, recognise it to a structured
% WAM item, apply the shared item-level ISO rewrite, and splice the
% new key back into the original line.
%
% iso_errors_rewrite_item/3 (in core/iso_errors) is the single source
% of truth for the builtin_call / put_structure / call / execute key
% swaps — this target no longer carries its own per-shape text rules.
% All four rewritable shapes carry the key as their first argument, so
% arg(1, ...) extracts old/new keys generically.
%
% Splicing rather than re-printing the recognised item preserves the
% original whitespace byte-for-byte, so existing text output is
% unaffected. Any failure along the chain (unrecognised line, key not
% in the ISO/lax tables, splice miss) falls through to OutLine = Line.
iso_errors_rewrite_line(Mode, Line, OutLine) :-
    (   wam_tokenize_line(Line, Tokens),
        wam_recognise_instruction(Tokens, Item0),
        iso_errors_rewrite_item(Mode, Item0, Item1),
        Item0 \== Item1,
        arg(1, Item0, OldKey),
        arg(1, Item1, NewKey),
        iso_errors_splice_line(Line, OldKey, NewKey, OutLine)
    ->  true
    ;   OutLine = Line
    ).

% Kept: the audit (iso_errors_audit_classify_line/2) still uses this to
% strip the trailing comma from a builtin_call key token.
iso_errors_clean_key_token(Token0, Token) :-
    (   string_concat(Token, ",", Token0)
    ->  true
    ;   Token = Token0
    ).

iso_errors_splice_line(Line, Key, NewKey, OutLine) :-
    sub_string(Line, Before, KLen, _After, Key),
    string_length(Key, KLen),
    sub_string(Line, 0, Before, _, Pre),
    PostStart is Before + KLen,
    sub_string(Line, PostStart, _, 0, Post),
    string_concat(Pre, NewKey, T1),
    string_concat(T1, Post, OutLine), !.

%% wam_fsharp_iso_audit(+Predicates, +Options, -Audit)
%  Read-only audit pass that reports per-call-site resolution of
%  default/explicit_iso/explicit_lax keys for each predicate in
%  Predicates.  Useful for reviewing whether an iso_errors config does
%  what was intended before generation.  Mirrors C++/Elixir/Python.
wam_fsharp_iso_audit(Predicates, Options, Audit) :-
    iso_errors_resolve_options(Options, Config),
    findall(audit(PI, Mode, Sites), (
        member(P, Predicates),
        iso_errors_audit_normalise_pi(P, PI),
        iso_errors_mode_for(Config, PI, Mode),
        iso_errors_audit_predicate(PI, Mode, Sites)
    ), Audit).

iso_errors_audit_predicate(PI, Mode, Sites) :-
    (   catch(
            ( iso_errors_audit_wam_for_pi(PI, WamText),
              iso_errors_audit_parse_lines(WamText, Items)
            ),
            _, fail)
    ->  iso_errors_audit_walk(Items, 0, Mode, [], SitesRev),
        reverse(SitesRev, Sites)
    ;   Sites = []
    ).

iso_errors_audit_wam_for_pi(Module:Pred/Arity, WamText) :- !,
    compile_predicate_to_wam(Module:Pred/Arity, [], WamText).
iso_errors_audit_wam_for_pi(Pred/Arity, WamText) :-
    compile_predicate_to_wam(Pred/Arity, [], WamText).

iso_errors_audit_parse_lines(WamText, Items) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    maplist(iso_errors_audit_parse_one, Lines, MaybeItems),
    exclude(==(skip), MaybeItems, Items).

iso_errors_audit_parse_one(Line, Item) :-
    split_string(Line, " \t", " \t", Parts0),
    exclude(==(""), Parts0, Parts),
    iso_errors_audit_classify_line(Parts, Item).

iso_errors_audit_classify_line([], skip).
iso_errors_audit_classify_line([Tok], label) :-
    string_concat(_, ":", Tok), !.
iso_errors_audit_classify_line(["builtin_call", Key0 | _], builtin_call(Key, 0)) :- !,
    iso_errors_clean_key_token(Key0, Key).
iso_errors_audit_classify_line(_, other).

wam_fsharp_iso_audit_report([]).
wam_fsharp_iso_audit_report([audit(PI, Mode, Sites)|Rest]) :-
    format('~w [~w]~n', [PI, Mode]),
    (   Sites == []
    ->  format('  (no builtin_call sites)~n', [])
    ;   forall(member(site(PC, Orig, Res, Src, Flip), Sites),
               format('  pc=~w  ~w -> ~w  (~w)  flip-changes=~w~n',
                      [PC, Orig, Res, Src, Flip]))
    ),
    wam_fsharp_iso_audit_report(Rest).

%% maybe_upgrade_bidirectional(+KV0, -KV)
%  Upgrade a category_ancestor kernel to bidirectional_ancestor.
%
%  NOTE: as of the bidirectional-not-default fix, no longer called
%  from any compile-time path. Kept for backwards-compatibility of
%  external callers that may use it directly. The decision to swap
%  kernel kinds in the F# WAM target's emission pipeline now flows
%  through wam_fsharp_apply_strategy_choice/4 (below), which gates
%  on the allow_bidirectional_kernel_swap/1 flag.
maybe_upgrade_bidirectional(Key-recursive_kernel(category_ancestor, PI, Config),
                            Key-recursive_kernel(bidirectional_ancestor, PI, Config)) :- !.
maybe_upgrade_bidirectional(KV, KV).

%% wam_fsharp_apply_strategy_selector(+ResWorkload, +Options, +KV0, -KV)
%
%  Phase 5a: per-kernel strategy-selector decision. Builds a
%  Recurrence from the detected kernel, calls
%  recurrence_evaluation_strategy:select_evaluation_strategy/3,
%  then applies the resulting Strategy as a kernel upgrade IF the
%  caller has set allow_bidirectional_kernel_swap(true).
%
%  ResWorkload is the workload signal list built once per compile
%  via recurrence_inputs:build_workload_signals/2.
%
%  Options carries the raw caller-provided options for the
%  bidirectional-not-default flag check.
wam_fsharp_apply_strategy_selector(ResWorkload,
                                   Options,
                                   Key-DetectedKernel,
                                   Key-MaybeUpgraded) :-
    recurrence_inputs:build_recurrence_term(DetectedKernel, [], Recurrence),
    recurrence_evaluation_strategy:select_evaluation_strategy(
        Recurrence, ResWorkload,
        strategy_choice(Strategy, _Trace)),
    wam_fsharp_apply_strategy_choice(Strategy, Options, DetectedKernel, MaybeUpgraded).

%% wam_fsharp_apply_strategy_choice(+Strategy, +Options, +DetectedKernel,
%%                                  -EmittedKernel)
%
%  Apply the selector's chosen strategy as a kernel-kind transformation
%  ONLY when the caller has explicitly opted in via
%  allow_bidirectional_kernel_swap(true).
%
%  Why opt-in by default:
%
%  templates/targets/fsharp_wam/program.fs.mustache has a hardcoded
%  call to nativeKernel_category_ancestor with the unidirectional
%  6-argument signature. The bidirectional kernel
%  (nativeKernel_bidirectional_ancestor) takes 7 arguments with a
%  different signature (parent/child lookup pair + cost/budget
%  floats). When the kernel-kind swap fires without a corresponding
%  Program.fs template update, the generated F# fails to compile
%  with FS0039 ('nativeKernel_category_ancestor' is not defined).
%
%  Until program.fs.mustache is parameterised to emit kernel-specific
%  benchmark loops (the template-system supports conditional
%  rendering — see docs/design/WAM_TEMPLATE_MATCH_CASE_TESTING.md
%  and docs/design/TEMPLATE_ENGINE.md), the swap is opt-in. The
%  cost-model decision is still computed and recorded in the trace;
%  this gate just suppresses the actual kernel-kind transformation.
%
%  Currently the only auto-upgrade wired is category_ancestor →
%  bidirectional_ancestor when the selector returns per_query(bidirectional).
%  All other (Strategy, KernelKind) combinations leave the kernel
%  unchanged regardless of the flag.
wam_fsharp_apply_strategy_choice(strategy(per_query(bidirectional)),
                                 Options,
                                 recursive_kernel(category_ancestor, PI, Config),
                                 EmittedKernel) :-
    !,
    (   option(allow_bidirectional_kernel_swap(true), Options)
    ->  EmittedKernel = recursive_kernel(bidirectional_ancestor, PI, Config),
        format(user_error,
               '[WAM-FSharp] strategy-selector upgraded category_ancestor -> bidirectional_ancestor (~w) [allow_bidirectional_kernel_swap(true)]~n',
               [PI])
    ;   EmittedKernel = recursive_kernel(category_ancestor, PI, Config),
        format(user_error,
               '[WAM-FSharp] strategy-selector would upgrade ~w to bidirectional_ancestor but the swap is suppressed by default (see allow_bidirectional_kernel_swap/1 in wam_fsharp_target.pl docstring)~n',
               [PI])
    ).
wam_fsharp_apply_strategy_choice(_, _Options, Kernel, Kernel).

%% =====================================================================
%% Cost-based auto-resolvers for F# WAM target
%% =====================================================================

%% resolve_auto_edge_store_fs(+Options0, -Options)
%  Resolve edge_store(auto) into a concrete store selection.
%  Decides between lmdb_cached, lmdb_eager, csr, or dual_csr based on
%  workload metadata. When edge_store is not auto (or absent), pass through.
resolve_auto_edge_store_fs(Options0, Options) :-
    (   option(edge_store(auto), Options0)
    ->  compute_edge_store_fs(Options0, Store),
        exclude(=(edge_store(_)), Options0, Rest),
        apply_edge_store_fs(Store, Rest, Options)
    ;   Options = Options0
    ).

compute_edge_store_fs(Options, Store) :-
    option(expected_query_count(Q), Options, 1),
    option(expected_lookups_per_query(L), Options, 50),
    option(edge_count(E), Options, 0),
    option(graph_mutability(Mut), Options, 'static'),
    option(needs_reverse(NeedsRev), Options, false),
    TotalLookups is Q * L,
    %% Cost model (milliseconds):
    %%   LMDB eager: one-time load E*0.005ms, per-lookup ~0.5us (Map.tryFind)
    %%   LMDB cached: no load, per-lookup ~2us warm (cursor + L2 cache)
    %%   CSR: build E*0.01ms, per-lookup ~1.5us (binary search on index)
    EagerLoadMs is E * 0.005,
    EagerPerLookupUs is 0.5,
    CachedPerLookupUs is 2.0,
    CsrBuildMs is E * 0.01,
    CsrPerLookupUs is 1.5,
    %% Total wall time per mode (ms):
    EagerTotalMs is EagerLoadMs + TotalLookups * EagerPerLookupUs / 1000,
    CachedTotalMs is TotalLookups * CachedPerLookupUs / 1000,
    CsrTotalMs is CsrBuildMs + TotalLookups * CsrPerLookupUs / 1000,
    (   Mut == 'dynamic'
    ->  Store = lmdb_cached
    ;   NeedsRev == true, CsrTotalMs =< CachedTotalMs
    ->  Store = dual_csr
    ;   (   CsrTotalMs =< EagerTotalMs, CsrTotalMs =< CachedTotalMs
        ->  Store = csr
        ;   EagerTotalMs =< CachedTotalMs
        ->  Store = lmdb_eager
        ;   Store = lmdb_cached
        )
    ),
    (   option(edge_store_verbose(true), Options)
    ->  format(user_error,
               '[WAM-FSharp] edge_store(auto): Q=~w L=~w E=~w eager=~2fms cached=~2fms csr=~2fms -> ~w~n',
               [Q, L, E, EagerTotalMs, CachedTotalMs, CsrTotalMs, Store])
    ;   true
    ).

apply_edge_store_fs(lmdb_cached, Opts, [lmdb_materialisation(cached) | Opts]).
apply_edge_store_fs(lmdb_eager, Opts, [lmdb_materialisation(eager) | Opts]).
apply_edge_store_fs(csr, Opts, Opts).
apply_edge_store_fs(dual_csr, Opts, Opts).

%% resolve_auto_lmdb_materialisation_fs(+Options0, -Options)
%  Resolve lmdb_materialisation(auto) into eager/lazy/cached using
%  the shared cost model. When not auto, pass through.
resolve_auto_lmdb_materialisation_fs(Options0, Options) :-
    (   option(lmdb_materialisation(auto), Options0)
    ->  compute_lmdb_materialisation_fs(Options0, Mode),
        exclude(=(lmdb_materialisation(_)), Options0, Rest),
        Options = [lmdb_materialisation(Mode) | Rest]
    ;   Options = Options0
    ).

compute_lmdb_materialisation_fs(Options, Mode) :-
    option(fact_count(F), Options, 0),
    option(demand_set_estimate(D), Options, F),
    option(expected_query_count(NQ), Options, 1),
    option(workload_segregated(WS), Options, false),
    option(working_set_fraction(WSF), Options, 0.05),
    EdgeBytes is D * 50,
    (   option(memory_budget(B), Options)
    ->  true
    ;   catch(read_mem_available_bytes(B), _, B = 1_000_000_000)
    ),
    (   EdgeBytes > B
    ->  (WS == true -> Mode = lazy ; Mode = cached)
    ;   KKeys is integer(F * WSF),
        option(cost_model_constants(Constants), Options, []),
        recommend_access_pattern(KKeys, EdgeBytes, B, Constants, Pattern),
        (   Pattern = sort
        ->  (NQ >= 10, F =< 100_000 -> Mode = eager ; Mode = cached)
        ;   (EdgeBytes > B -> Mode = cached ; Mode = eager)
        )
    ),
    (   option(materialisation_verbose(true), Options)
    ->  format(user_error,
               '[WAM-FSharp] lmdb_materialisation(auto): F=~w D=~w B=~w NQ=~w WS=~w -> ~w~n',
               [F, D, B, NQ, WS, Mode])
    ;   true
    ).

%% resolve_auto_lmdb_cache_tier_fs(+Options0, -Options)
%  Resolve lmdb_l2_capacity(auto) into a concrete cache size.
%  For eager/lazy modes the capacity is 0 (no cache tier).
%  For cached mode the capacity scales with demand set size.
resolve_auto_lmdb_cache_tier_fs(Options0, Options) :-
    option(lmdb_l2_capacity(L2Val), Options0, auto),
    (   L2Val == auto
    ->  compute_l2_capacity_fs(Options0, Cap),
        exclude(=(lmdb_l2_capacity(_)), Options0, Rest),
        Options = [lmdb_l2_capacity(Cap) | Rest]
    ;   Options = Options0
    ).

compute_l2_capacity_fs(Options, Cap) :-
    option(lmdb_materialisation(Mode), Options, cached),
    (   Mode \= cached
    ->  Cap = 0
    ;   option(fact_count(F), Options, 0),
        option(demand_set_estimate(D), Options, F),
        Target is max(256, min(65536, integer(D * 0.1))),
        Cap = Target
    ),
    (   option(l2_capacity_verbose(true), Options)
    ->  format(user_error,
               '[WAM-FSharp] lmdb_l2_capacity(auto): mode=~w -> ~w~n',
               [Mode, Cap])
    ;   true
    ).

%% resolve_auto_csr_index_backend_fs(+Options0, -Options)
%  Resolve csr_index_backend(auto) by delegating to the shared
%  cost rule in cost_model.pl.
resolve_auto_csr_index_backend_fs(Options0, Options) :-
    (   option(csr_path(_), Options0),
        option(csr_index_backend(auto), Options0)
    ->  resolve_csr_index_backend(Options0, Backend),
        exclude(=(csr_index_backend(_)), Options0, Rest),
        Options = [csr_index_backend(Backend) | Rest]
    ;   Options = Options0
    ).

%% resolve_fsharp_cost_options(+Options0, -Options)
%  Composed resolver chain: runs all four auto-resolvers in sequence.
resolve_fsharp_cost_options(Options0, Options) :-
    resolve_auto_edge_store_fs(Options0, Options1),
    resolve_auto_lmdb_materialisation_fs(Options1, Options2),
    resolve_auto_lmdb_cache_tier_fs(Options2, Options3),
    resolve_auto_csr_index_backend_fs(Options3, Options).

%% write_wam_fsharp_project(+Predicates, +Options, +ProjectDir)
%  Generates a complete F# project with:
%  - WamTypes.fs:   Value DU, WamState, WamContext, helpers
%  - WamRuntime.fs: step, backtrack, run loop, executeForeign
%  - Predicates.fs: compiled predicates (instruction arrays)
%  - Lowered.fs:    lowered predicate functions (Phase 3+)
%  - Program.fs:    benchmark driver
%  - wam-fsharp-bench.fsproj: project file
write_wam_fsharp_project(Predicates, Options0, ProjectDir) :-
    make_directory_path(ProjectDir),

    % Phase 0: resolve cost-based auto options
    resolve_fsharp_cost_options(Options0, Options),

    % Resolve runtime parser mode (capability hook).  When `compiled`
    % is selected, the portable parser predicates + the target-agnostic
    % `read_term_from_atom`/`parse_*` wrappers are appended to the
    % user's predicate list so the F# WAM compiler emits them too.
    % When `none`, statically visible parser-dependent bodies are
    % rejected at codegen time (Lua-style) rather than silently
    % producing a stubbed runtime.
    wam_target_runtime_parser(wam_fsharp, Options, RuntimeParserMode),
    fsharp_validate_runtime_parser_mode(Predicates, RuntimeParserMode),
    fsharp_project_predicates(Predicates, RuntimeParserMode, ProjectPredicates),
    length(Predicates, NUserPreds),
    length(ProjectPredicates, NAllPreds),
    format(user_error,
           '[WAM-FSharp] runtime_parser=~w (user=~w total=~w)~n',
           [RuntimeParserMode, NUserPreds, NAllPreds]),

    % Kernel detection (skip with no_kernels(true)).  Kernel detection
    % runs on ProjectPredicates so portable-parser predicates can also
    % be considered, but in practice they're not recursive kernels.
    (   option(no_kernels(true), Options)
    ->  DetectedKernels = [],
        format(user_error, '[WAM-FSharp] kernel detection suppressed~n', [])
    ;   detect_kernels_fs(ProjectPredicates, DetectedKernels0),
        % Phase 5a: replace the option-driven upgrade gate with the
        % recurrence-evaluation-strategy selector. Caller options flow
        % into the workload; the selector decides per-kernel whether
        % to upgrade. Backwards-compatible: when caller passes
        % kernel_mode(bidirectional) + csr_path(_) explicitly, the
        % selector resolves to per_query(bidirectional) via either
        % step_third_option (search admissible) or step_caller_wins
        % (fallback). When the caller passes neither but workload
        % signals support bidirectional, the selector auto-selects.
        recurrence_inputs:build_workload_signals(Options, ResWorkload),
        maplist(wam_fsharp_apply_strategy_selector(ResWorkload, Options),
                DetectedKernels0, SelectedKernels),
        % Strategy selection may replace a detected kernel kind. Recheck the
        % emitted kind so an unsupported upgrade can never reintroduce an
        % undefined nativeKernel_* call after the initial capability gate.
        filter_supported_kernel_pairs_fs(SelectedKernels, DetectedKernels),
        (   DetectedKernels \= []
        ->  pairs_keys(DetectedKernels, DetectedKeys),
            format(user_error, '[WAM-FSharp] detected kernels: ~w~n', [DetectedKeys])
        ;   true
        )
    ),

    % Resolve emit mode and partition
    wam_fsharp_resolve_emit_mode(Options, EmitMode),
    wam_fsharp_partition_predicates(EmitMode, ProjectPredicates, DetectedKernels, InterpList, LoweredList),
    length(InterpList, NInterp),
    length(LoweredList, NLower),
    format(user_error, '[WAM-FSharp] emit_mode=~w  interpreted=~w  lowered=~w~n',
           [EmitMode, NInterp, NLower]),

    % Generate WamTypes.fs (separate module for types)
    fsharp_wam_type_header(TypeHeader),
    directory_file_path(ProjectDir, 'WamTypes.fs', TypesPath),
    write_fs_file(TypesPath, TypeHeader),

    % Generate WamRuntime.fs (runtime functions only)
    compile_wam_runtime_to_fsharp(Options, DetectedKernels, RuntimeCode),
    directory_file_path(ProjectDir, 'WamRuntime.fs', RuntimePath),
    write_fs_file(RuntimePath, RuntimeCode),

    % Compute base PCs for all predicates (shared between Predicates.fs and Lowered.fs)
    compute_base_pcs_fs(ProjectPredicates, BasePCMap),

    % Generate Predicates.fs (skip FFI-owned facts — Phase D: -70%)
    compile_predicates_to_fsharp(ProjectPredicates, Options, DetectedKernels, BasePCMap, PredsCode),
    directory_file_path(ProjectDir, 'Predicates.fs', PredsPath),
    write_fs_file(PredsPath, PredsCode),

    % Generate Lowered.fs
    lower_all_fs(LoweredList, BasePCMap, DetectedKernels, Options, LoweredEntries),
    generate_lowered_fs(LoweredEntries, LoweredCode),
    directory_file_path(ProjectDir, 'Lowered.fs', LoweredPath),
    write_fs_file(LoweredPath, LoweredCode),

    % Generate LmdbFactSource.fs when lmdb_path is set.
    (   option(lmdb_path(_), Options)
    ->  fsharp_lmdb_template_source(LmdbTemplateCode),
        directory_file_path(ProjectDir, 'LmdbFactSource.fs', LmdbPath),
        write_fs_file(LmdbPath, LmdbTemplateCode),
        option(lmdb_materialisation(LmdbMode), Options, cached),
        option(lmdb_l2_capacity(L2Cap), Options, auto),
        format(user_error, '[WAM-FSharp] LMDB fact source included (materialisation=~w, l2_capacity=~w)~n', [LmdbMode, L2Cap])
    ;   true
    ),

    % Generate CsrReader.fs when csr_path is set.
    (   option(csr_path(_), Options)
    ->  fsharp_csr_template_source(CsrTemplateCode),
        directory_file_path(ProjectDir, 'CsrReader.fs', CsrFsPath),
        write_fs_file(CsrFsPath, CsrTemplateCode),
        format(user_error, '[WAM-FSharp] CSR reader included~n', [])
    ;   true
    ),

    % Generate Program.fs (benchmark driver).  The driver calls into
    % USER predicates only -- portable-parser predicates are library
    % code, not entry points -- so this stays on the original list.
    option(module_name(ModName), Options, 'wam-fsharp-bench'),
    generate_program_fs(Predicates, DetectedKernels, Options, ProgramCode),
    directory_file_path(ProjectDir, 'Program.fs', ProgramPath),
    write_fs_file(ProgramPath, ProgramCode),

    % Generate .fsproj
    generate_fsproj(ModName, Options, FsprojCode),
    format(atom(FsprojFile), '~w.fsproj', [ModName]),
    directory_file_path(ProjectDir, FsprojFile, FsprojPath),
    write_fs_file(FsprojPath, FsprojCode),

    format(user_error, '[WAM-FSharp] Generated project at: ~w~n', [ProjectDir]).

%% fsharp_lmdb_template_source(-Code)
%  Reads the LmdbFactSource.fs.mustache template and returns it
%  as a string.  The template is a plain F# module with no mustache
%  variables (the LMDB path is passed at runtime, not baked in).
fsharp_lmdb_template_source(Code) :-
    source_file(wam_fsharp_target:_, SrcFile),
    file_directory_name(SrcFile, SrcDir),
    file_directory_name(SrcDir, TargetsDir),
    file_directory_name(TargetsDir, UnifyWeaverDir),
    file_directory_name(UnifyWeaverDir, ProjectRoot),
    atom_concat(ProjectRoot, '/templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache', TemplatePath),
    read_file_to_string(TemplatePath, Code, []).

%% fsharp_csr_template_source(-Code)
%  Reads the CsrReader.fs.mustache template.
fsharp_csr_template_source(Code) :-
    source_file(wam_fsharp_target:_, SrcFile),
    file_directory_name(SrcFile, SrcDir),
    file_directory_name(SrcDir, TargetsDir),
    file_directory_name(TargetsDir, UnifyWeaverDir),
    file_directory_name(UnifyWeaverDir, ProjectRoot),
    atom_concat(ProjectRoot, '/templates/targets/fsharp_wam/csr_reader.fs.mustache', TemplatePath),
    read_file_to_string(TemplatePath, Code, []).

%% fsharp_program_template_source(-Code)
%  Reads the program.fs.mustache template for the main Program.fs file.
fsharp_program_template_source(Code) :-
    source_file(wam_fsharp_target:_, SrcFile),
    file_directory_name(SrcFile, SrcDir),
    file_directory_name(SrcDir, TargetsDir),
    file_directory_name(TargetsDir, UnifyWeaverDir),
    file_directory_name(UnifyWeaverDir, ProjectRoot),
    atom_concat(ProjectRoot, '/templates/targets/fsharp_wam/program.fs.mustache', TemplatePath),
    read_file_to_string(TemplatePath, Code, []).

%% write_fs_file(+Path, +Content)
write_fs_file(Path, Content) :-
    open(Path, write, Stream, [encoding(utf8)]),
    write(Stream, Content),
    close(Stream).

%% compile_predicates_to_fsharp(+Predicates, +Options, +DetectedKernels, +BasePCMap, -Code)
compile_predicates_to_fsharp(Predicates, Options, DetectedKernels, BasePCMap, Code) :-
    % Phase D: skip FFI-owned facts (predicates handled entirely by FFI kernel path)
    exclude(ffi_owned_fact_filter_fs(DetectedKernels), Predicates, WamPredicates),
    (   length(Predicates, NAll), length(WamPredicates, NWam), NSkipped is NAll - NWam,
        NSkipped > 0
    ->  format(user_error, '[WAM-FSharp] skipped ~w FFI-owned fact predicates~n', [NSkipped])
    ;   true
    ),
    % ISO error config: resolve once, warn about ambiguous bare overrides,
    % pass through compile_one_predicate_fs so each predicate gets its
    % WAM text rewritten according to its resolved Mode.  Mirrors Python
    % wam_python_target:compile_all_predicates ISO wiring.
    iso_errors_resolve_options(Options, IsoConfig),
    iso_errors_warn_multi_module(IsoConfig, WamPredicates),
    maplist(compile_one_predicate_fs(Options, BasePCMap, IsoConfig), WamPredicates, PredCodes),
    atomic_list_concat(PredCodes, '\n\n', AllPredCode),
    % Build merged code list and label map
    maplist(pred_func_name_fs, WamPredicates, FuncNames),
    emit_merged_code_build_fs(FuncNames, MergedCodeBuild),
    % WSP3: materialize relation-keyed weighted adjacency from declared
    % inline edge facts so generated projects do not leave WcFfiWeightedFacts
    % empty when a native weighted_shortest_path3 handler is present.
    emit_weighted_ffi_facts_fs(DetectedKernels, Options, WeightedBlock),
    format(string(Code),
'module Predicates

open WamTypes
open WamRuntime

~w

~w
~w
', [AllPredCode, MergedCodeBuild, WeightedBlock]).

%% emit_weighted_ffi_facts_fs(+DetectedKernels, +Options, -Code)
%  Emit declaredWeightedEdgeFacts / buildWeightedFfiFacts helpers into
%  Predicates.fs. External weighted fact-source options (when present) are
%  merged on top of inline triples so existing external wiring is preserved.
emit_weighted_ffi_facts_fs(DetectedKernels, Options, Code) :-
    findall(Rel-Triples,
            ( member(_-recursive_kernel(weighted_shortest_path3, _, ConfigOps),
                     DetectedKernels),
              member(edge_pred(EdgePred/3), ConfigOps),
              atom(EdgePred),
              collect_weighted_edge_triples_fs(EdgePred, Triples0),
              Rel = EdgePred,
              Triples = Triples0
            ),
            InlineRaw),
    sort(InlineRaw, InlineSorted),
    (   option(external_weighted_facts(External), Options)
    ->  true
    ;   External = []
    ),
    append(InlineSorted, External, Combined0),
    keysort(Combined0, Combined1),
    group_pairs_by_key(Combined1, Grouped),
    maplist(merge_weighted_rel_group_fs, Grouped, RelEntries),
    emit_weighted_ffi_facts_block_fs(RelEntries, Code).

merge_weighted_rel_group_fs(Rel-Lists, Rel-Triples) :-
    append(Lists, Flat),
    sort(Flat, Triples).

collect_weighted_edge_triples_fs(EdgePred, Triples) :-
    functor(Head, EdgePred, 3),
    findall(triple(From, To, W),
            (   clause(user:Head, true),
                Head =.. [EdgePred, From, To, W0],
                atom(From),
                atom(To),
                number(W0),
                W is float(W0)
            ),
            Triples).

emit_weighted_ffi_facts_block_fs(RelEntries, Code) :-
    maplist(emit_weighted_rel_entry_fs, RelEntries, EntryAtoms),
    (   EntryAtoms = []
    ->  EntriesStr = ''
    ;   atomic_list_concat(EntryAtoms, ';\n      ', EntriesJoined),
        format(atom(EntriesStr), '~n      ~w~n    ', [EntriesJoined])
    ),
    format(string(Code),
'
/// Relation-keyed weighted edge facts declared inline (and any
/// option(external_weighted_facts/1) triples). Used to populate
/// WcFfiWeightedFacts for weighted_shortest_path3 / A* kernels.
let declaredWeightedEdgeFacts : Map<string, (string * string * float) list> =
    Map.ofList [~w]

let declaredWeightedAtoms : string list =
    declaredWeightedEdgeFacts
    |> Map.toList
    |> List.collect (fun (_, triples) ->
        triples |> List.collect (fun (f, t, _) -> [f; t]))
    |> List.distinct

/// Build interned adjacency Map<rel, Map<from, (to * weight) list>>.
let buildWeightedFfiFacts (intern: Map<string, int>)
    : Map<string, Map<int, (int * float) list>> =
    declaredWeightedEdgeFacts
    |> Map.map (fun _ triples ->
        triples
        |> List.choose (fun (f, t, w) ->
            match Map.tryFind f intern, Map.tryFind t intern with
            | Some fi, Some ti -> Some (fi, (ti, w))
            | _ -> None)
        |> List.groupBy fst
        |> List.map (fun (k, vs) -> k, vs |> List.map snd)
        |> Map.ofList)
', [EntriesStr]).

emit_weighted_rel_entry_fs(Rel-Triples, Atom) :-
    maplist(emit_weighted_triple_literal_fs, Triples, TripleAtoms),
    (   TripleAtoms = []
    ->  TriplesStr = ''
    ;   atomic_list_concat(TripleAtoms, '; ', TriplesStr)
    ),
    format(atom(Atom), '("~w", [~w])', [Rel, TriplesStr]).

emit_weighted_triple_literal_fs(triple(From, To, W), Atom) :-
    format(atom(Atom), '("~w", "~w", ~w)', [From, To, W]).

compile_one_predicate_fs(Options, BasePCMap, IsoConfig, PredIndicator, Code) :-
    %% Shape check: must be Module:Name/Arity or Name/Arity.  The
    %% destructured Pred/Arity aren't used further (we pass
    %% PredIndicator wholesale to the helpers below); underscore-
    %% prefix the names so SWI doesn't flag them as singletons.
    (   PredIndicator = _M:_Pred/_Arity -> true ; PredIndicator = _Pred/_Arity ),
    wam_fsharp_predicate_wamcode(PredIndicator, WamCode),
    %% ISO rewrite happens at the text level (matches Python).  Items
    %% level rewrite is exported for future use but the F# emitter
    %% currently parses WAM text, so a text-level pass is the natural
    %% integration point.
    iso_errors_pi_for_rewrite(PredIndicator, RewritePI),
    iso_errors_rewrite_text(IsoConfig, RewritePI, WamCode, WamCodeIso),
    predicate_base_pc_fs(PredIndicator, BasePCMap, BasePC),
    compile_wam_predicate_to_fsharp(PredIndicator, WamCodeIso, [base_pc(BasePC)|Options], Code).

% Normalise an F# predicate indicator (possibly module-qualified or
% paired) to the bare Name/Arity form iso_errors_mode_for expects.
iso_errors_pi_for_rewrite(_M:Pred/Arity, Pred/Arity) :- !.
iso_errors_pi_for_rewrite(Pred/Arity-_, Pred/Arity) :- !.
iso_errors_pi_for_rewrite(PI, PI).

pred_func_name_fs(PI, FN) :-
    (   PI = _M:P/A -> true ; PI = P/A ),
    atom_string(P, PStr),
    atomic_list_concat(Parts, '$', PStr),
    atomic_list_concat(Parts, '_', PSafe),
    format(atom(FN), '~w_~w', [PSafe, A]).

emit_merged_code_build_fs([], Code) :-
    %% Even the empty case must reserve index 0 as a halt sentinel:
    %% `run`'s loop short-circuits on `s.WsPC = 0` before fetching from
    %% WcCode, so we never actually execute index 0.  Having it present
    %% means PC numbering (1-based per the WAM compiler / labels) aligns
    %% with array indexing.  Without the sentinel, the first real
    %% instruction lands at index 0 (unreachable) and label "pred/N" → 1
    %% points to the SECOND emitted instruction — the cause of Bug A in
    %% PR #2350's query smoke.
    Code = 'let allCode : Instruction array = [| Fail |]\nlet allLabels : Map<string, int> = Map.empty'.
emit_merged_code_build_fs(FuncNames, Code) :-
    FuncNames \= [],
    maplist([FN, Expr]>>(format(atom(Expr), '~w_code', [FN])), FuncNames, CodeExprs),
    atomic_list_concat(CodeExprs, ' @ ', CodeConcat),
    maplist([FN, Expr]>>(format(atom(Expr), '~w_labels', [FN])), FuncNames, LabelExprs),
    atomic_list_concat(LabelExprs, ' |> mapUnion ', LabelUnion),
    %% Prepend a Fail sentinel at index 0 so WAM-PC 1 (the first real
    %% instruction emitted by the WAM compiler) maps to array index 1.
    %% Closes the off-by-one (Bug A from PR #2350): without the
    %% sentinel, SwitchOnConstantPc — emitted as the first instruction
    %% by the WAM compiler — lands at array index 0 where `run`'s halt
    %% sentinel `if s.WsPC = 0 then Some s` short-circuits before
    %% fetching, making indexed dispatch unreachable.
    format(string(Code),
'let allCode : Instruction array = (Fail :: (~w)) |> List.toArray
let allLabels : Map<string, int> = ~w', [CodeConcat, LabelUnion]).

%% generate_lowered_fs(+LoweredEntries, -Code)
generate_lowered_fs([], Code) :- !,
    Code = 'module Lowered

open WamTypes
open WamRuntime

// No predicates lowered.
let loweredPredicates : Map<string, WamContext -> WamState -> WamState option> =
    Map.empty'.
generate_lowered_fs(LoweredEntries, Code) :-
    LoweredEntries \= [],
    with_output_to(string(FuncBodies), (
        forall(member(lowered(_, _, FsCode), LoweredEntries),
               format('~w~n', [FsCode]))
    )),
    with_output_to(string(MapEntries), (
        format('    Map.ofList~n'),
        format('        [ '),
        emit_lowered_entries_fs(LoweredEntries),
        format('        ]~n')
    )),
    format(string(Code),
'module Lowered

open WamTypes
open WamRuntime

~w
let loweredPredicates : Map<string, WamContext -> WamState -> WamState option> =
~w', [FuncBodies, MapEntries]).

emit_lowered_entries_fs([lowered(PredName, FuncName, _)|Rest]) :-
    format('("~w", ~w)~n', [PredName, FuncName]),
    emit_lowered_entries_rest_fs(Rest).
emit_lowered_entries_rest_fs([]).
emit_lowered_entries_rest_fs([lowered(PredName, FuncName, _)|Rest]) :-
    format('          ; ("~w", ~w)~n', [PredName, FuncName]),
    emit_lowered_entries_rest_fs(Rest).

%% generate_program_fs(+Predicates, +DetectedKernels, +Options, -Code)
generate_program_fs(_Predicates, DetectedKernels, Options, Code) :-
    pairs_keys(DetectedKernels, ForeignKeys),
    format_foreign_preds_fs(ForeignKeys, ForeignPredsStr),
    generate_lookup_sources_expr_fs(Options, LookupSourcesExpr),
    (option(csr_path(_), Options) -> HasCsr = true ; HasCsr = false),
    (option(csr_kernel(true), Options) -> HasCsrKernel = true ; HasCsrKernel = false),
    (option(lmdb_path(_), Options) -> HasLmdb = true ; HasLmdb = false),
    option(lmdb_materialisation(Materialisation), Options, cached),
    option(lmdb_l2_capacity(L2Cap), Options, 4096),
    (   integer(L2Cap)
    ->  atom_number(L2CapStr, L2Cap)
    ;   L2CapStr = L2Cap
    ),
    (   option(kernel_mode(bidirectional), Options)
    ->  HasBidir = true,
        option(child_branch_factor(BranchFactor), Options, 15.0),
        option(graph_dimensionality(Dimensionality), Options, 5.0)
    ;   HasBidir = false, BranchFactor = 15.0, Dimensionality = 5.0
    ),
    %% Determine the kernel_kind for the benchmark-loop {{match}}
    %% block in program.fs.mustache. The benchmark loop targets a
    %% single primary kernel; if multiple kernels are detected,
    %% the first one wins (matches the prior hardcoded behaviour
    %% which assumed category_ancestor). Empty list -> unknown,
    %% which triggers the {{default}} stub case.
    (   DetectedKernels = [_-recursive_kernel(FirstKernelKind, _, _) | _]
    ->  KernelKind = FirstKernelKind
    ;   KernelKind = unknown
    ),
    %% Additive conformance driver (CONF-FSHARP): when true, Program.fs
    %% prints true/false for argv[0]=pred/arity via tryRun. Default
    %% benchmark driver (human-facing TSV/LMDB timing) is unchanged.
    option(conformance_main(ConfMain), Options, false),
    Dict = [
        foreign_preds = ForeignPredsStr,
        lookup_sources_expr = LookupSourcesExpr,
        has_csr = HasCsr,
        has_csr_kernel = HasCsrKernel,
        has_lmdb = HasLmdb,
        has_bidirectional = HasBidir,
        kernel_kind = KernelKind,
        branch_factor = BranchFactor,
        dimensionality = Dimensionality,
        materialisation = Materialisation,
        l2_capacity = L2CapStr,
        conformance_main = ConfMain
    ],
    fsharp_program_template_source(Template),
    render_template(Template, Dict, Code).

format_foreign_preds_fs([], '').
format_foreign_preds_fs(Keys, Str) :-
    Keys \= [],
    maplist([Key, Q]>>(format(atom(Q), '"~w"', [Key])), Keys, Quoted),
    atomic_list_concat(Quoted, '; ', Str).

%% generate_lookup_sources_expr_fs(+Options, -Expr)
%  Generate the F# expression for WcLookupSources based on configured
%  data sources (CSR, LMDB, or Map.empty).
generate_lookup_sources_expr_fs(Options, Expr) :-
    findall(Entry, lookup_source_entry_fs(Options, Entry), Entries),
    (   Entries = []
    ->  Expr = 'Map.empty'
    ;   atomic_list_concat(Entries, '; ', EntriesStr),
        format(atom(Expr), 'Map.ofList [ ~w ]', [EntriesStr])
    ).

lookup_source_entry_fs(Options, Entry) :-
    option(csr_path(CsrPath), Options),
    % csr_kernel mode constructs its CSR source directly in the kernel branch
    % (forward category_parent CSR), so skip the auto WcLookupSources entry --
    % its default category_child relation would point at a non-existent file.
    \+ option(csr_kernel(true), Options),
    option(csr_relation(Rel), Options, category_child),
    % Resolve the CSR artifact dir against factsDir at runtime (a relative
    % csr_path like "csr" would otherwise be opened against the process CWD,
    % not the fixture dir). Path.Combine leaves an absolute csr_path unchanged.
    format(atom(Entry),
        '("~w", CsrReader.CsrLookupSource(System.IO.Path.Combine(factsDir, "~w"), "~w") :> ILookupSource)',
        [Rel, CsrPath, Rel]).

lookup_source_entry_fs(Options, Entry) :-
    option(csr_parent_path(CsrParentPath), Options),
    format(atom(Entry),
        '("category_parent", CsrReader.CsrLookupSource(System.IO.Path.Combine(factsDir, "~w"), "category_parent") :> ILookupSource)',
        [CsrParentPath]).

%% generate_fsproj(+ModName, +Options, -Code)
generate_fsproj(ModName, Options, Code) :-
    (   option(lmdb_path(_), Options)
    ->  LmdbCompile  = '\n    <Compile Include="LmdbFactSource.fs" />',
        NeedLightningDB = true
    ;   LmdbCompile  = '',
        NeedLightningDB = false
    ),
    (   option(csr_path(_), Options)
    ->  CsrCompile = '\n    <Compile Include="CsrReader.fs" />',
        NeedLightningDBForCsr = true
    ;   CsrCompile = '',
        NeedLightningDBForCsr = false
    ),
    %% Include LightningDB if either LMDB or CSR needs it
    (   (NeedLightningDB = true ; NeedLightningDBForCsr = true)
    ->  LmdbPackage = '\n  <ItemGroup>\n    <PackageReference Include="LightningDB" Version="0.21.0" />\n  </ItemGroup>\n'
    ;   LmdbPackage = ''
    ),
    format(string(Code),
'<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AssemblyName>~w</AssemblyName>
    <Optimize>true</Optimize>
    <Nullable>enable</Nullable>
    <Deterministic>true</Deterministic>
  </PropertyGroup>

  <ItemGroup>
    <Compile Include="WamTypes.fs" />~w~w
    <Compile Include="WamRuntime.fs" />
    <Compile Include="Predicates.fs" />
    <Compile Include="Lowered.fs" />
    <Compile Include="Program.fs" />
  </ItemGroup>
~w
</Project>
', [ModName, LmdbCompile, CsrCompile, LmdbPackage]).
