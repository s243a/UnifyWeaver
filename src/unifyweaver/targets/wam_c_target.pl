:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_c_target.pl - WAM-to-C Transpilation Target
%
% Transpiles WAM runtime predicates to C code.
%
% Design goals:
% - C99 or C11 compatible.
% - Explicit memory and pointer handling.
% - WAM registers (A, S, H, HB, TR, P, CP, B) mapped to C struct fields.
% - Unification trail and heap modeled as C arrays with explicit bounds.

:- module(wam_c_target, [
    compile_step_wam_to_c/2,          % +Options, -CCode
    compile_wam_helpers_to_c/2,       % +Options, -CCode
    compile_wam_runtime_to_c/2,       % +Options, -CCode
    compile_wam_predicate_to_c/4,     % +Pred/Arity, +WamCode, +Options, -CCode
    wam_instruction_to_c_literal/2,   % +WamInstr, -CCode
    wam_instruction_to_c_literal/3,   % +WamInstr, +LabelMap, -CCode
    detect_kernels/2,                 % +Predicates, -DetectedKernels
    generate_setup_detected_kernels_c/2, % +DetectedKernels, -CCode
    generate_setup_reverse_index_c/2, % +Options, -CCode
    resolve_wam_c_reverse_index_plan/2, % +Options, -Plan
    plan_wam_c_lowered_helpers/4,     % +Predicates, +Options, +DetectedKeys, -Plans
    write_wam_c_project/3             % +Predicates, +Options, +ProjectDir
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(pairs), [pairs_keys/2]).

:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../bindings/c_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/cost_model', [resolve_reverse_index/2]).
:- use_module('../core/recursive_kernel_detection', [
    detect_recursive_kernel/4
]).

%% resolve_wam_c_reverse_index_plan(+Options, -Plan)
%  Normalizes reverse-index options for WAM-C and reports which normalized
%  variants the generated C runtime can currently consume.
resolve_wam_c_reverse_index_plan(Options, Plan) :-
    resolve_reverse_index(Options, ReverseIndex),
    wam_c_reverse_index_capabilities(ReverseIndex, Capabilities),
    Plan = wam_c_reverse_index_plan(ReverseIndex, Capabilities).

wam_c_reverse_index_capabilities(none, [
    planning(unneeded),
    runtime_child_lookup(unavailable)
]) :- !.
wam_c_reverse_index_capabilities(artifact(Opts), [
    planning(accepted),
    runtime_child_lookup(available),
    runtime_api(wam_reverse_csr_lookup_children),
    runtime_index_backend(sorted_array),
    runtime_io(RuntimeIo)
]) :-
    memberchk(storage_kind(csr_pread_artifact), Opts),
    memberchk(index_backend(sorted_array), Opts),
    wam_c_reverse_index_runtime_io_policy_supported(artifact(Opts)),
    wam_c_reverse_index_runtime_io_capability(artifact(Opts), RuntimeIo),
    !.
wam_c_reverse_index_capabilities(csr(Opts), [
    planning(accepted),
    runtime_child_lookup(available_after_artifact_build),
    runtime_api(wam_reverse_csr_lookup_children),
    runtime_index_backend(sorted_array),
    runtime_io(RuntimeIo)
]) :-
    memberchk(index_backend(sorted_array), Opts),
    wam_c_reverse_index_runtime_io_policy_supported(csr(Opts)),
    wam_c_reverse_index_runtime_io_capability(csr(Opts), RuntimeIo),
    !.
wam_c_reverse_index_capabilities(ReverseIndex, [
    planning(accepted),
    runtime_child_lookup(available),
    runtime_api(wam_reverse_csr_lookup_children),
    runtime_index_backend(lmdb_offset),
    runtime_io(RuntimeIo),
    runtime_requires(wam_c_enable_lmdb)
]) :-
    wam_c_reverse_index_uses_lmdb_offset(ReverseIndex),
    wam_c_reverse_index_runtime_io_policy_supported(ReverseIndex),
    wam_c_reverse_index_runtime_io_capability(ReverseIndex, RuntimeIo),
    !.
wam_c_reverse_index_capabilities(ReverseIndex, [
    planning(accepted),
    runtime_child_lookup(unsupported),
    runtime_reason(csr_io_policy_not_implemented(Policy)),
    runtime_io(Policy)
]) :-
    wam_c_reverse_index_runtime_io_policy(ReverseIndex, Policy),
    \+ wam_c_supported_reverse_index_runtime_io_policy(Policy),
    !.
wam_c_reverse_index_capabilities(ReverseIndex, [
    planning(accepted),
    runtime_child_lookup(unsupported),
    runtime_reason(reverse_index_kind_not_implemented)
]) :-
    ReverseIndex \= none.

wam_c_reverse_index_uses_lmdb_offset(csr(Opts)) :-
    memberchk(index_backend(lmdb_offset), Opts).
wam_c_reverse_index_uses_lmdb_offset(artifact(Opts)) :-
    memberchk(storage_kind(csr_pread_artifact), Opts),
    memberchk(index_backend(lmdb_offset), Opts).

wam_c_reverse_index_runtime_io_policy(csr(Opts), Policy) :-
    memberchk(phase(runtime_available), Opts),
    memberchk(io_policy(Policy), Opts).
wam_c_reverse_index_runtime_io_policy(artifact(Opts), Policy) :-
    memberchk(phase(runtime_available), Opts),
    memberchk(io_policy(Policy), Opts).

wam_c_reverse_index_declared_io_policy(csr(Opts), Policy) :-
    memberchk(io_policy(Policy), Opts).
wam_c_reverse_index_declared_io_policy(artifact(Opts), Policy) :-
    memberchk(io_policy(Policy), Opts).

wam_c_supported_reverse_index_runtime_io_policy(buffered_pread).
wam_c_supported_reverse_index_runtime_io_policy(buffered_pread_drop).
wam_c_supported_reverse_index_runtime_io_policy(direct_io).

wam_c_reverse_index_runtime_io_capability(ReverseIndex, pread_drop) :-
    wam_c_reverse_index_declared_io_policy(ReverseIndex, buffered_pread_drop),
    !.
wam_c_reverse_index_runtime_io_capability(ReverseIndex, direct_io) :-
    wam_c_reverse_index_declared_io_policy(ReverseIndex, direct_io),
    !.
wam_c_reverse_index_runtime_io_capability(_ReverseIndex, pread).

wam_c_reverse_index_runtime_io_policy_supported(ReverseIndex) :-
    (   wam_c_reverse_index_runtime_io_policy(ReverseIndex, Policy)
    ->  wam_c_supported_reverse_index_runtime_io_policy(Policy)
    ;   true
    ).

wam_c_require_reverse_index_runtime_io_policy(ReverseIndex) :-
    (   wam_c_reverse_index_runtime_io_policy(ReverseIndex, Policy),
        \+ wam_c_supported_reverse_index_runtime_io_policy(Policy)
    ->  throw(error(permission_error(use, csr_io_policy, Policy), _))
    ;   wam_c_reverse_index_runtime_io_policy(ReverseIndex, direct_io),
        \+ wam_c_reverse_index_direct_io_preconditions(ReverseIndex)
    ->  throw(error(permission_error(use, csr_io_policy, direct_io), _))
    ;   true
    ).

wam_c_reverse_index_direct_io_preconditions(csr(Opts)) :-
    memberchk(block_size_edges(BlockSizeEdges), Opts),
    integer(BlockSizeEdges),
    BlockSizeEdges > 0.
wam_c_reverse_index_direct_io_preconditions(artifact(Opts)) :-
    memberchk(block_size_edges(BlockSizeEdges), Opts),
    integer(BlockSizeEdges),
    BlockSizeEdges > 0.

% ============================================================================
% PHASE 4: Hybrid Module Assembly
% ============================================================================

%% write_wam_c_project(+Predicates, +Options, +ProjectDir)
%  Generates a full C project for the given predicates.
write_wam_c_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    detect_kernels_for_options(Predicates, Options, DetectedKernels),
    % Generate runtime .c and .h files
    compile_wam_runtime_to_c(Options, RuntimeCode),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    write_file(RuntimePath, RuntimeCode),

    % Compile predicates and generate lib.c
    pairs_keys(DetectedKernels, DetectedKeys),
    plan_wam_c_lowered_helpers(Predicates, Options, DetectedKeys, LoweredPlans),
    maybe_report_wam_c_lowered_helper_plan(Options, LoweredPlans),
    compile_lowered_helpers_for_project(LoweredPlans, LoweredKeys, LoweredCode, SetupLoweredCode),
    render_wam_c_lowered_helper_plan(LoweredPlans, LoweredPlanCode),
    generate_setup_detected_kernels_c(DetectedKernels, SetupKernelCode),
    generate_setup_reverse_index_c(Options, SetupReverseIndexCode),
    compile_predicates_for_project(Predicates,
                                   [detected_kernel_keys(DetectedKeys),
                                    lowered_helper_keys(LoweredKeys)|Options],
                                   PredicatesCode),
    format(atom(LibCode), '#include "wam_runtime.h"~n~n~w~n~n~w~n~n~w~n~n~w~n~n~w',
           [LoweredPlanCode, SetupKernelCode, SetupReverseIndexCode,
            LoweredCode, SetupLoweredCode]),
    format(atom(LibCodeWithPredicates), '~w~n~n~w', [LibCode, PredicatesCode]),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    write_file(LibPath, LibCodeWithPredicates),

    format('WAM C project created at: ~w~n', [ProjectDir]).

%% write_file(+Path, +Content)
write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

%% compile_predicates_for_project(+Predicates, +Options, -Code)
compile_predicates_for_project([], _, "").
compile_predicates_for_project([PredIndicator|Rest], Options, Code) :-
    predicate_indicator_parts(PredIndicator, Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    option(detected_kernel_keys(DetectedKeys), Options, []),
    option(lowered_helper_keys(LoweredKeys), Options, []),
    (   memberchk(Key, DetectedKeys)
    ->  format(atom(WamCode), '~w/~w:\n    call_foreign ~w, ~w\n    proceed',
               [Pred, Arity, Key, Arity]),
        compile_wam_predicate_to_c(Module:Pred/Arity, WamCode, Options, PredCode)
    ;   memberchk(Key, LoweredKeys)
    ->  format(atom(WamCode), '~w/~w:\n    call_foreign ~w, ~w\n    proceed',
               [Pred, Arity, Key, Arity]),
        compile_wam_predicate_to_c(Module:Pred/Arity, WamCode, Options, PredCode)
    ;   wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode)
    ->  compile_wam_predicate_to_c(Module:Pred/Arity, WamCode, Options, PredCode)
    ;   format(atom(PredCode), '// ~w/~w: compilation failed', [Pred, Arity])
    ),
    compile_predicates_for_project(Rest, Options, RestCode),
    format(atom(Code), '~w\n\n~w', [PredCode, RestCode]).

predicate_indicator_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
predicate_indicator_parts(Pred/Arity, user, Pred, Arity).

detect_kernels_for_options(Predicates, Options, DetectedKernels) :-
    (   option(no_kernels(true), Options)
    ->  DetectedKernels = [],
        format(user_error, '[WAM-C] kernel detection suppressed~n', [])
    ;   detect_kernels(Predicates, DetectedKernels),
        (   DetectedKernels \= []
        ->  pairs_keys(DetectedKernels, DetectedKeys),
            format(user_error, '[WAM-C] detected kernels: ~w~n', [DetectedKeys])
        ;   true
        )
    ).

%% detect_kernels(+Predicates, -DetectedKernels)
%  Run the shared recursive-kernel detector over project predicates.
detect_kernels([], []).
detect_kernels([PI|Rest], Kernels) :-
    predicate_indicator_parts(PI, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses \= [],
        detect_recursive_kernel(Pred, Arity, Clauses, Kernel),
        wam_c_supported_kernel(Kernel)
    ->  format(atom(Key), '~w/~w', [Pred, Arity]),
        Kernels = [Key-Kernel|RestKernels]
    ;   Kernels = RestKernels
    ),
    detect_kernels(Rest, RestKernels).

wam_c_supported_kernel(recursive_kernel(category_ancestor, _Pred, _ConfigOps)).
wam_c_supported_kernel(recursive_kernel(bidirectional_ancestor, _Pred, _ConfigOps)).
wam_c_supported_kernel(recursive_kernel(transitive_closure2, _Pred, _ConfigOps)).
wam_c_supported_kernel(recursive_kernel(transitive_distance3, _Pred, _ConfigOps)).
wam_c_supported_kernel(recursive_kernel(transitive_parent_distance4, _Pred, _ConfigOps)).
wam_c_supported_kernel(recursive_kernel(transitive_step_parent_distance5, _Pred, _ConfigOps)).
wam_c_supported_kernel(recursive_kernel(weighted_shortest_path3, _Pred, _ConfigOps)).
wam_c_supported_kernel(recursive_kernel(astar_shortest_path4, _Pred, _ConfigOps)).

%% generate_setup_detected_kernels_c(+DetectedKernels, -CCode)
%  Emit C startup wiring for detected kernels. This function registers only
%  native handlers; callers still decide when to load/register fact sources.
generate_setup_detected_kernels_c([], Code) :- !,
    Code = 'void setup_detected_wam_c_kernels(WamState* state) {\n    (void)state;\n}'.
generate_setup_detected_kernels_c(DetectedKernels, Code) :-
    maplist(wam_c_kernel_registration_line, DetectedKernels, Lines),
    atomic_list_concat(Lines, '\n', Body),
    format(atom(Code),
           'void setup_detected_wam_c_kernels(WamState* state) {\n~w\n}',
           [Body]).

%% generate_setup_reverse_index_c(+Options, -CCode)
%  Emit project-level lifecycle hooks for a declaratively configured
%  runtime reverse CSR. The generated setup function only attaches an
%  artifact when the normalized reverse_index phase is runtime_available
%  and the caller provided the concrete artifact paths.
generate_setup_reverse_index_c(Options, Code) :-
    (   wam_c_reverse_index_setup_plan(Options, Plan)
    ->  wam_c_reverse_index_setup_code(Plan, Code)
    ;   wam_c_reverse_index_noop_setup_code(Code)
    ).

wam_c_reverse_index_setup_plan(Options, Plan) :-
    resolve_wam_c_reverse_index_plan(Options,
        wam_c_reverse_index_plan(ReverseIndex, Capabilities)),
    wam_c_require_reverse_index_runtime_io_policy(ReverseIndex),
    wam_c_reverse_index_runtime_available(ReverseIndex, Capabilities),
    wam_c_reverse_index_setup_backend(ReverseIndex, Backend),
    wam_c_reverse_index_setup_policy(ReverseIndex, IoPolicy),
    wam_c_reverse_index_setup_block_size_edges(ReverseIndex, BlockSizeEdges),
    wam_c_reverse_index_setup_paths(Backend, Options, Paths),
    option(category_id_map(CategoryIdMap), Options, []),
    Plan = wam_c_reverse_index_setup(Backend, IoPolicy, BlockSizeEdges, Paths, CategoryIdMap).

wam_c_reverse_index_runtime_available(artifact(Opts), Capabilities) :-
    memberchk(phase(runtime_available), Opts),
    memberchk(runtime_child_lookup(available), Capabilities).
wam_c_reverse_index_runtime_available(csr(Opts), Capabilities) :-
    memberchk(phase(runtime_available), Opts),
    (   memberchk(runtime_child_lookup(available), Capabilities)
    ;   memberchk(runtime_child_lookup(available_after_artifact_build), Capabilities)
    ).

wam_c_reverse_index_setup_backend(ReverseIndex, lmdb_offset) :-
    wam_c_reverse_index_uses_lmdb_offset(ReverseIndex),
    !.
wam_c_reverse_index_setup_backend(_ReverseIndex, sorted_array).

wam_c_reverse_index_setup_policy(ReverseIndex, Policy) :-
    wam_c_reverse_index_runtime_io_policy(ReverseIndex, Policy),
    !.
wam_c_reverse_index_setup_policy(_ReverseIndex, buffered_pread).

wam_c_reverse_index_setup_block_size_edges(csr(Opts), BlockSizeEdges) :-
    memberchk(block_size_edges(BlockSizeEdges), Opts),
    !.
wam_c_reverse_index_setup_block_size_edges(artifact(Opts), BlockSizeEdges) :-
    memberchk(block_size_edges(BlockSizeEdges), Opts),
    !.
wam_c_reverse_index_setup_block_size_edges(_ReverseIndex, 0).

wam_c_reverse_index_setup_paths(sorted_array, Options,
                                sorted_array(IndexPath, ValuesPath)) :-
    option(reverse_csr_index_path(IndexPath), Options),
    option(reverse_csr_values_path(ValuesPath), Options).
wam_c_reverse_index_setup_paths(lmdb_offset, Options,
                                lmdb_offset(OffsetPath, ValuesPath, DbiName)) :-
    option(reverse_csr_offset_lmdb_path(OffsetPath), Options),
    option(reverse_csr_values_path(ValuesPath), Options),
    option(reverse_csr_offset_lmdb_dbi(DbiName), Options, offsets).

wam_c_reverse_index_noop_setup_code(
'bool setup_wam_c_reverse_index_artifacts(WamState* state,
                                         WamReverseCsrArtifact* bidirectional_child_csr) {
    (void)state;
    (void)bidirectional_child_csr;
    return true;
}

void teardown_wam_c_reverse_index_artifacts(WamState* state,
                                           WamReverseCsrArtifact* bidirectional_child_csr) {
    (void)state;
    (void)bidirectional_child_csr;
}
').

wam_c_reverse_index_setup_code(
        wam_c_reverse_index_setup(Backend, IoPolicy, BlockSizeEdges, Paths, CategoryIdMap),
        Code) :-
    wam_c_reverse_index_load_call(Backend, IoPolicy, BlockSizeEdges, Paths, LoadCall),
    wam_c_category_id_setup_lines(CategoryIdMap, IdLines),
    format(atom(Code),
'bool setup_wam_c_reverse_index_artifacts(WamState* state,
                                         WamReverseCsrArtifact* bidirectional_child_csr) {
    wam_reverse_csr_init(bidirectional_child_csr);
    if (!(~w)) {
        wam_reverse_csr_close(bidirectional_child_csr);
        return false;
    }
~w
    wam_attach_bidirectional_child_csr(state, bidirectional_child_csr);
    return true;
}

void teardown_wam_c_reverse_index_artifacts(WamState* state,
                                           WamReverseCsrArtifact* bidirectional_child_csr) {
    wam_attach_bidirectional_child_csr(state, NULL);
    wam_reverse_csr_close(bidirectional_child_csr);
}
', [LoadCall, IdLines]).

wam_c_reverse_index_load_call(sorted_array,
                              buffered_pread,
                              _BlockSizeEdges,
                              sorted_array(IndexPath, ValuesPath),
                              LoadCall) :-
    c_string_literal(IndexPath, IndexLit),
    c_string_literal(ValuesPath, ValuesLit),
    format(atom(LoadCall),
           'wam_reverse_csr_load(bidirectional_child_csr, ~w, ~w)',
           [IndexLit, ValuesLit]).
wam_c_reverse_index_load_call(sorted_array,
                              buffered_pread_drop,
                              _BlockSizeEdges,
                              sorted_array(IndexPath, ValuesPath),
                              LoadCall) :-
    c_string_literal(IndexPath, IndexLit),
    c_string_literal(ValuesPath, ValuesLit),
    format(atom(LoadCall),
           'wam_reverse_csr_load_pread_drop(bidirectional_child_csr, ~w, ~w)',
           [IndexLit, ValuesLit]).
wam_c_reverse_index_load_call(sorted_array,
                              direct_io,
                              BlockSizeEdges,
                              sorted_array(IndexPath, ValuesPath),
                              LoadCall) :-
    c_string_literal(IndexPath, IndexLit),
    c_string_literal(ValuesPath, ValuesLit),
    format(atom(LoadCall),
           'wam_reverse_csr_load_direct_io(bidirectional_child_csr, ~w, ~w, ~w)',
           [IndexLit, ValuesLit, BlockSizeEdges]).
wam_c_reverse_index_load_call(lmdb_offset,
                              buffered_pread,
                              _BlockSizeEdges,
                              lmdb_offset(OffsetPath, ValuesPath, DbiName),
                              LoadCall) :-
    c_string_literal(OffsetPath, OffsetLit),
    c_string_literal(ValuesPath, ValuesLit),
    c_string_literal(DbiName, DbiLit),
    format(atom(LoadCall),
           'wam_reverse_csr_load_lmdb_offset(bidirectional_child_csr, ~w, ~w, ~w)',
           [ValuesLit, OffsetLit, DbiLit]).
wam_c_reverse_index_load_call(lmdb_offset,
                              buffered_pread_drop,
                              _BlockSizeEdges,
                              lmdb_offset(OffsetPath, ValuesPath, DbiName),
                              LoadCall) :-
    c_string_literal(OffsetPath, OffsetLit),
    c_string_literal(ValuesPath, ValuesLit),
    c_string_literal(DbiName, DbiLit),
    format(atom(LoadCall),
           'wam_reverse_csr_load_lmdb_offset_pread_drop(bidirectional_child_csr, ~w, ~w, ~w)',
           [ValuesLit, OffsetLit, DbiLit]).
wam_c_reverse_index_load_call(lmdb_offset,
                              direct_io,
                              BlockSizeEdges,
                              lmdb_offset(OffsetPath, ValuesPath, DbiName),
                              LoadCall) :-
    c_string_literal(OffsetPath, OffsetLit),
    c_string_literal(ValuesPath, ValuesLit),
    c_string_literal(DbiName, DbiLit),
    format(atom(LoadCall),
           'wam_reverse_csr_load_lmdb_offset_direct_io(bidirectional_child_csr, ~w, ~w, ~w, ~w)',
           [ValuesLit, OffsetLit, DbiLit, BlockSizeEdges]).

wam_c_category_id_setup_lines([], '').
wam_c_category_id_setup_lines(CategoryIdMap, Lines) :-
    CategoryIdMap \= [],
    maplist(wam_c_category_id_setup_line, CategoryIdMap, LineList),
    atomic_list_concat(LineList, '\n', Body),
    format(atom(Lines), '~w\n', [Body]).

wam_c_category_id_setup_line(Atom-Id, Line) :-
    !,
    wam_c_category_id_setup_line_(Atom, Id, Line).
wam_c_category_id_setup_line(category_id(Atom, Id), Line) :-
    !,
    wam_c_category_id_setup_line_(Atom, Id, Line).
wam_c_category_id_setup_line(id(Atom, Id), Line) :-
    !,
    wam_c_category_id_setup_line_(Atom, Id, Line).

wam_c_category_id_setup_line_(Atom, Id, Line) :-
    integer(Id),
    c_string_literal(Atom, AtomLit),
    format(atom(Line),
           '    wam_register_category_id(state, ~w, ~w);',
           [AtomLit, Id]).

c_string_literal(Value, Literal) :-
    format(atom(Raw), '~w', [Value]),
    atom_codes(Raw, Codes),
    phrase(c_string_literal_codes(Codes), EscapedCodes),
    atom_codes(Escaped, EscapedCodes),
    format(atom(Literal), '"~w"', [Escaped]).

c_string_literal_codes([]) --> [].
c_string_literal_codes([0'\\|Rest]) --> "\\\\", c_string_literal_codes(Rest).
c_string_literal_codes([0'"|Rest]) --> "\\\"", c_string_literal_codes(Rest).
c_string_literal_codes([10|Rest]) --> "\\n", c_string_literal_codes(Rest).
c_string_literal_codes([13|Rest]) --> "\\r", c_string_literal_codes(Rest).
c_string_literal_codes([9|Rest]) --> "\\t", c_string_literal_codes(Rest).
c_string_literal_codes([Code|Rest]) --> [Code], c_string_literal_codes(Rest).

wam_c_kernel_registration_line(Key-recursive_kernel(category_ancestor, _Pred, ConfigOps), Line) :-
    wam_c_kernel_max_depth(ConfigOps, MaxDepth),
    format(atom(Line),
           '    wam_register_category_ancestor_kernel(state, "~w", ~w);',
           [Key, MaxDepth]).
wam_c_kernel_registration_line(Key-recursive_kernel(bidirectional_ancestor, _Pred, ConfigOps), Line) :-
    wam_c_kernel_max_depth(ConfigOps, MaxDepth),
    wam_c_kernel_float_config(ConfigOps, parent_step_cost, 1.0, ParentCost),
    wam_c_kernel_float_config(ConfigOps, child_step_cost, 3.0, ChildCost),
    wam_c_kernel_float_config(ConfigOps, cost_budget, 10.0, Budget),
    format(atom(Line),
           '    wam_register_bidirectional_ancestor_kernel(state, "~w", ~w, ~w, ~w, ~w);',
           [Key, MaxDepth, ParentCost, ChildCost, Budget]).
wam_c_kernel_registration_line(Key-recursive_kernel(transitive_closure2, _Pred, _ConfigOps), Line) :-
    format(atom(Line),
           '    wam_register_transitive_closure_kernel(state, "~w");',
           [Key]).
wam_c_kernel_registration_line(Key-recursive_kernel(transitive_distance3, _Pred, _ConfigOps), Line) :-
    format(atom(Line),
           '    wam_register_transitive_distance_kernel(state, "~w");',
           [Key]).
wam_c_kernel_registration_line(Key-recursive_kernel(transitive_parent_distance4, _Pred, _ConfigOps), Line) :-
    format(atom(Line),
           '    wam_register_transitive_parent_distance_kernel(state, "~w");',
           [Key]).
wam_c_kernel_registration_line(Key-recursive_kernel(transitive_step_parent_distance5, _Pred, ConfigOps), Line) :-
    member(edge_pred(EdgePred/2), ConfigOps),
    format(atom(Line),
           '    wam_register_transitive_step_parent_distance_kernel(state, "~w", "~w");',
           [Key, EdgePred]).
wam_c_kernel_registration_line(Key-recursive_kernel(weighted_shortest_path3, _Pred, _ConfigOps), Line) :-
    format(atom(Line),
           '    wam_register_weighted_shortest_path_kernel(state, "~w");',
           [Key]).
wam_c_kernel_registration_line(Key-recursive_kernel(astar_shortest_path4, _Pred, _ConfigOps), Line) :-
    format(atom(Line),
           '    wam_register_astar_shortest_path_kernel(state, "~w");',
           [Key]).

wam_c_kernel_max_depth(ConfigOps, MaxDepth) :-
    (   member(max_depth(MaxDepth0), ConfigOps),
        integer(MaxDepth0),
        MaxDepth0 > 0
    ->  MaxDepth = MaxDepth0
    ;   MaxDepth = 10
    ).

wam_c_kernel_float_config(ConfigOps, Key, Default, Value) :-
    Term =.. [Key, Value0],
    (   member(Term, ConfigOps),
        number(Value0),
        Value0 > 0
    ->  Value = Value0
    ;   Value = Default
    ).

% ============================================================================
% Prototype native lowered helpers
% ============================================================================

plan_wam_c_lowered_helpers(Predicates, Options, DetectedKeys, Plans) :-
    (   option(lowered_helpers(true), Options)
    ->  maplist(wam_c_predicate_key, Predicates, AvailableKeys),
        maplist(plan_wam_c_lowered_helper(DetectedKeys, AvailableKeys), Predicates, Plans)
    ;   maplist(plan_wam_c_lowered_helper_disabled, Predicates, Plans)
    ).

wam_c_predicate_key(PredIndicator, Key) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]).

plan_wam_c_lowered_helper(DetectedKeys, AvailableKeys, PredIndicator, Plan) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    (   memberchk(Key, DetectedKeys)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, interpreted, detected_kernel)
    ;   lowered_fact_helper_rows(PredIndicator, Rows)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, fact_only(Rows))
    ;   lowered_body_call_helper(PredIndicator, AvailableKeys, CalleeKey, CalleeArity)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, body_call(CalleeKey, CalleeArity))
    ;   lowered_body_call_projected_helper(PredIndicator, AvailableKeys, CalleeKey, CalleeArity, CalleeArgs)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, body_call_projected(CalleeKey, CalleeArity, CalleeArgs))
    ;   lowered_body_call_projection_row_constrained_helper(PredIndicator, AvailableKeys, CalleeKey, Rows)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, filtered_fact(CalleeKey, Rows))
    ;   lowered_body_call_rejection_reason(PredIndicator, AvailableKeys, Reason)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, rejected, Reason)
    ;   lowered_comparison_filtered_fact_helper(PredIndicator, CalleeKey, Rows)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, comparison_filtered_fact(CalleeKey, Rows))
    ;   lowered_filtered_fact_helper(PredIndicator, CalleeKey, Rows)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, filtered_fact(CalleeKey, Rows))
    ;   lowered_fact_helper_rejection_reason(PredIndicator, Reason),
        Plan = wam_c_lowered_helper_plan(Key, PredIndicator, rejected, Reason)
    ).

plan_wam_c_lowered_helper_disabled(PredIndicator, Plan) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    Plan = wam_c_lowered_helper_plan(Key, PredIndicator, interpreted, lowering_disabled).

lowered_fact_helper_rejection_reason(PredIndicator, Reason) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Args-Body,
            (   user:clause(Head, Body),
                Head =.. [_|Args]
            ),
            [HeadArgs-Body0]),
    Body0 \== true,
    lowered_filter_rejection_reason(HeadArgs, Body0, Reason),
    !.
lowered_fact_helper_rejection_reason(PredIndicator, non_fact_clause) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    user:clause(Head, Body),
    Body \== true,
    !.
lowered_fact_helper_rejection_reason(PredIndicator, unsupported_fact_arguments) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    user:clause(Head, true),
    !.
lowered_fact_helper_rejection_reason(_, no_clauses).

lowered_filter_rejection_reason(HeadArgs, Body0, Reason) :-
    strip_module_qualification(Body0, Body),
    lowered_comparison_filter_rejection_reason(HeadArgs, Body, Reason),
    !.
lowered_filter_rejection_reason(_HeadArgs, Body0, multi_goal_body) :-
    strip_module_qualification(Body0, Body),
    Body = (_, _),
    !.
lowered_filter_rejection_reason(_HeadArgs, Body0, unsupported_comparison_guard) :-
    strip_module_qualification(Body0, Body),
    Body =.. [Pred|Args],
    length(Args, Arity),
    wam_c_lowered_comparison_guard(Pred/Arity),
    !.
lowered_filter_rejection_reason(HeadArgs, Body0, non_constant_filter_argument) :-
    strip_module_qualification(Body0, Body),
    Body =.. [_CalleePred|CalleeArgs],
    member(Arg, CalleeArgs),
    \+ lowered_filter_arg_supported(Arg, HeadArgs),
    !.
lowered_filter_rejection_reason(HeadArgs, Body0, no_matching_filter_rows) :-
    strip_module_qualification(Body0, Body),
    Body =.. [CalleePred|CalleeArgs],
    length(CalleeArgs, CalleeArity),
    \+ wam_c_lowered_comparison_guard(CalleePred/CalleeArity),
    maplist(var, HeadArgs),
    callee_args_supported_for_filter(CalleeArgs, HeadArgs),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, CalleeRows),
    \+ (   member(CalleeRow, CalleeRows),
           callee_row_matches_filter(CalleeArgs, HeadArgs, CalleeRow)
       ),
    !.
lowered_filter_rejection_reason(HeadArgs, Body0, unsupported_filter_callee) :-
    strip_module_qualification(Body0, Body),
    Body =.. [CalleePred|CalleeArgs],
    length(CalleeArgs, CalleeArity),
    maplist(lowered_filter_arg_supported_(HeadArgs), CalleeArgs),
    CalleeIndicator = user:CalleePred/CalleeArity,
    \+ lowered_fact_helper_rows(CalleeIndicator, _Rows),
    !.

lowered_comparison_filter_rejection_reason(HeadArgs, Body, Reason) :-
    Body = (Call0, Guard0),
    Call0 \= (_, _),
    Guard0 \= (_, _),
    strip_module_qualification(Call0, Call),
    strip_module_qualification(Guard0, Guard),
    Guard =.. [GuardPred, Left, Right],
    wam_c_lowered_comparison_guard(GuardPred/2),
    comparison_filter_call_context(HeadArgs, Call, CalleeArgs, CalleeRows),
    comparison_filter_guard_rejection_reason(CalleeArgs, CalleeRows, GuardPred, Left, Right, Reason).

comparison_filter_call_context(HeadArgs, Call, CalleeArgs, CalleeRows) :-
    Call =.. [CalleePred|CalleeArgs],
    length(CalleeArgs, CalleeArity),
    \+ wam_c_lowered_comparison_guard(CalleePred/CalleeArity),
    maplist(var, HeadArgs),
    maplist(callee_filter_arg_supported, CalleeArgs),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, CalleeRows).

comparison_filter_guard_rejection_reason(CalleeArgs, _CalleeRows, _GuardPred, Left, Right, comparison_guard_unbound_variable) :-
    (   comparison_arg_unbound_variable(Left, CalleeArgs)
    ;   comparison_arg_unbound_variable(Right, CalleeArgs)
    ),
    !.
comparison_filter_guard_rejection_reason(CalleeArgs, _CalleeRows, _GuardPred, Left, Right, unsupported_comparison_expression) :-
    (   \+ comparison_arg_basic(Left, CalleeArgs)
    ;   \+ comparison_arg_basic(Right, CalleeArgs)
    ),
    !.
comparison_filter_guard_rejection_reason(CalleeArgs, CalleeRows, GuardPred, Left, Right, non_integer_comparison_rows) :-
    wam_c_lowered_ordering_guard(GuardPred),
    comparison_arg_supported(Left, CalleeArgs),
    comparison_arg_supported(Right, CalleeArgs),
    member(CalleeRow, CalleeRows),
    callee_row_matches_arg_constraints(CalleeArgs, CalleeRow),
    comparison_arg_value(Left, CalleeArgs, CalleeRow, LeftValue),
    comparison_arg_value(Right, CalleeArgs, CalleeRow, RightValue),
    (   \+ integer(LeftValue)
    ;   \+ integer(RightValue)
    ),
    !.
comparison_filter_guard_rejection_reason(CalleeArgs, CalleeRows, GuardPred, Left, Right, no_matching_comparison_rows) :-
    comparison_arg_supported(Left, CalleeArgs),
    comparison_arg_supported(Right, CalleeArgs),
    \+ ( member(CalleeRow, CalleeRows),
         callee_row_matches_comparison_filter(CalleeArgs, GuardPred, Left, Right, CalleeRow)
       ),
    !.

wam_c_lowered_comparison_guard((=)/2).
wam_c_lowered_comparison_guard((==)/2).
wam_c_lowered_comparison_guard((\==)/2).
wam_c_lowered_comparison_guard((>)/2).
wam_c_lowered_comparison_guard((<)/2).
wam_c_lowered_comparison_guard((>=)/2).
wam_c_lowered_comparison_guard((=<)/2).
wam_c_lowered_comparison_guard((=:=)/2).
wam_c_lowered_comparison_guard((=\=)/2).
wam_c_lowered_comparison_guard(is/2).

wam_c_lowered_ordering_guard((>)).
wam_c_lowered_ordering_guard((<)).
wam_c_lowered_ordering_guard((>=)).
wam_c_lowered_ordering_guard((=<)).
wam_c_lowered_ordering_guard((=:=)).
wam_c_lowered_ordering_guard((=\=)).

compile_lowered_helpers_for_project(Plans, LoweredKeys, Code, SetupCode) :-
    findall(Key-CodePart-SetupLine,
            (   member(wam_c_lowered_helper_plan(Key, PredIndicator, lowered, fact_only(Rows)), Plans),
                lowered_fact_helper_for_predicate(PredIndicator, Rows, Key, CodePart, SetupLine)
            ;   member(wam_c_lowered_helper_plan(Key, PredIndicator, lowered, filtered_fact(_CalleeKey, Rows)), Plans),
                lowered_fact_helper_for_predicate(PredIndicator, Rows, Key, CodePart, SetupLine)
            ;   member(wam_c_lowered_helper_plan(Key, PredIndicator, lowered, comparison_filtered_fact(_ComparisonCalleeKey, Rows)), Plans),
                lowered_fact_helper_for_predicate(PredIndicator, Rows, Key, CodePart, SetupLine)
            ),
            FactEntries),
    findall(Key-CodePart-SetupLine,
            (   member(wam_c_lowered_helper_plan(Key, PredIndicator, lowered, body_call(CalleeKey, CalleeArity)), Plans),
                lowered_body_call_helper_for_predicate(PredIndicator, CalleeKey, CalleeArity, Key, CodePart, SetupLine)
            ),
            BodyCallEntries),
    findall(Key-CodePart-SetupLine,
            (   member(wam_c_lowered_helper_plan(Key, PredIndicator, lowered, body_call_projected(CalleeKey, CalleeArity, CalleeArgs)), Plans),
                lowered_body_call_projected_helper_for_predicate(PredIndicator, CalleeKey, CalleeArity, CalleeArgs, Key, CodePart, SetupLine)
            ),
            ProjectedBodyCallEntries),
    append([FactEntries, BodyCallEntries, ProjectedBodyCallEntries], Entries),
    findall(K, member(K-_-_, Entries), LoweredKeys),
    findall(C, member(_-C-_, Entries), Codes),
    findall(S, member(_-_-S, Entries), SetupLines),
    atomic_list_concat(Codes, '\n\n', Code),
    (   SetupLines = []
    ->  SetupCode = 'void setup_lowered_wam_c_helpers(WamState* state) {\n    (void)state;\n}'
    ;   atomic_list_concat(SetupLines, '\n', SetupBody),
        format(atom(SetupCode),
               'void setup_lowered_wam_c_helpers(WamState* state) {\n~w\n}',
               [SetupBody])
    ).

render_wam_c_lowered_helper_plan([], '// WAM-C lowered helper plan: none').
render_wam_c_lowered_helper_plan(Plans, Code) :-
    maplist(render_wam_c_lowered_helper_plan_line, Plans, Lines),
    atomic_list_concat(['// WAM-C lowered helper plan'|Lines], '\n', Code).

render_wam_c_lowered_helper_plan_line(wam_c_lowered_helper_plan(Key, _PredIndicator, Action, Reason), Line) :-
    wam_c_lowered_plan_reason_label(Reason, ReasonLabel),
    format(atom(Line), '// - ~w ~w: ~w', [Action, Key, ReasonLabel]).

maybe_report_wam_c_lowered_helper_plan(Options, Plans) :-
    (   option(report_lowered_helpers(true), Options)
    ->  findall(Key-ReasonLabel, wam_c_lowered_helper_plan_by_action(Plans, lowered, Key, ReasonLabel), Lowered),
        findall(Key-ReasonLabel, wam_c_lowered_helper_plan_by_action(Plans, interpreted, Key, ReasonLabel), Interpreted),
        findall(Key-ReasonLabel, wam_c_lowered_helper_plan_by_action(Plans, rejected, Key, ReasonLabel), Rejected),
        format(user_error,
               '[WAM-C] lowered helper plan: lowered=~w interpreted=~w rejected=~w~n',
               [Lowered, Interpreted, Rejected])
    ;   true
    ).

wam_c_lowered_helper_plan_by_action(Plans, Action, Key, ReasonLabel) :-
    member(wam_c_lowered_helper_plan(Key, _, Action, Reason), Plans),
    wam_c_lowered_plan_reason_label(Reason, ReasonLabel).

wam_c_lowered_plan_reason_label(fact_only(_Rows), fact_only) :- !.
wam_c_lowered_plan_reason_label(body_call(_CalleeKey, _CalleeArity), body_call) :- !.
wam_c_lowered_plan_reason_label(body_call_projected(_CalleeKey, _CalleeArity, _CalleeArgs), body_call_projected) :- !.
wam_c_lowered_plan_reason_label(filtered_fact(_CalleeKey, _Rows), filtered_fact) :- !.
wam_c_lowered_plan_reason_label(comparison_filtered_fact(_CalleeKey, _Rows), comparison_filtered_fact) :- !.
wam_c_lowered_plan_reason_label(Reason, Reason).

lowered_fact_helper_rows(PredIndicator, Rows) :-
    catch(lowered_fact_helper_rows_(PredIndicator, Rows),
          error(permission_error(access, private_procedure, _), _),
          fail).

lowered_fact_helper_rows_(PredIndicator, Rows) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Args,
            (   user:clause(Head, true),
                Head =.. [_|Args],
                maplist(wam_c_lowered_constant, Args)
            ),
            Rows),
    Rows \= [],
    \+ ( user:clause(Head, Body), Body \== true ).

lowered_fact_helper_for_predicate(PredIndicator, Rows, Key, Code, SetupLine) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_c_symbol_name(Pred, Arity, Symbol),
    wam_c_lowered_fact_helper_code(Symbol, Arity, Rows, Code),
    format(atom(SetupLine),
           '    wam_register_foreign_predicate(state, "~w", ~w, ~w);',
           [Key, Arity, Symbol]).

wam_c_lowered_fact_helper_code(Symbol, 0, Rows, Code) :-
    !,
    wam_c_lowered_fact_dispatch(0, Rows, BodyCode),
    format(atom(Code),
'static bool ~w(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 0) return false;
~w
    return false;
}',
           [Symbol, BodyCode]).
wam_c_lowered_fact_helper_code(Symbol, Arity, Rows, Code) :-
    Arity > 0,
    length(Rows, RowCount),
    format(atom(RowTableSymbol), '~w_rows', [Symbol]),
    format(atom(ScanSymbol), '~w_scan_rows', [Symbol]),
    wam_c_lowered_static_row_table(RowTableSymbol, Arity, Rows, RowTableCode),
    lowered_fact_bucket_arrays(Symbol, Rows, BucketArraysCode, DispatchCasesCode, BucketCount),
    Mask is BucketCount - 1,
    format(atom(Code),
'~w

static bool ~w(WamState *state, WamValue **cells, const int *row_indices, int row_count) {
    for (int row_pos = 0; row_pos < row_count; row_pos++) {
        int row_index = row_indices ? row_indices[row_pos] : row_pos;
        bool match = true;
        for (int col = 0; col < ~w; col++) {
            if (!val_is_unbound(*cells[col]) && !val_equal(*cells[col], ~w[row_index][col])) {
                match = false;
                break;
            }
        }
        if (match) {
            for (int col = 0; col < ~w; col++) {
                if (val_is_unbound(*cells[col])) {
                    trail_binding(state, cells[col]);
                    *cells[col] = ~w[row_index][col];
                }
            }
            return true;
        }
    }
    return false;
}

~w

static bool ~w(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != ~w) return false;
    WamValue *cells[~w];
    for (int i = 0; i < ~w; i++) cells[i] = wam_deref_ptr(state, &state->A[i]);
    if (!val_is_unbound(*cells[0])) {
        unsigned int bucket = 0;
        if (cells[0]->tag == VAL_ATOM) {
            bucket = wam_hash_string(cells[0]->data.atom) & ~w;
        } else if (cells[0]->tag == VAL_INT) {
            bucket = ((unsigned int)cells[0]->data.integer * 2654435761u) & ~w;
        } else {
            return false;
        }
        switch (bucket) {
~w
        default:
            return false;
        }
    }
    return ~w(state, cells, NULL, ~w);
}',
           [RowTableCode, ScanSymbol, Arity, RowTableSymbol, Arity, RowTableSymbol,
            BucketArraysCode, Symbol, Arity, Arity, Arity, Mask, Mask, DispatchCasesCode,
            ScanSymbol, RowCount]).

wam_c_lowered_static_row_table(RowTableSymbol, Arity, Rows, Code) :-
    maplist(wam_c_lowered_static_row, Rows, RowCodes),
    atomic_list_concat(RowCodes, ',\n', RowInitCode),
    format(atom(Code),
'static const WamValue ~w[][~w] = {
~w
};',
           [RowTableSymbol, Arity, RowInitCode]).

wam_c_lowered_static_row(Args, Code) :-
    maplist(c_static_value_literal, Args, ValueCodes),
    atomic_list_concat(ValueCodes, ', ', ValuesCode),
    format(atom(Code), '    { ~w }', [ValuesCode]).

c_static_value_literal(Atom, Lit) :-
    atom(Atom),
    !,
    format(atom(Lit), '{ .tag = VAL_ATOM, .data.atom = "~w" }', [Atom]).
c_static_value_literal(Int, Lit) :-
    integer(Int),
    !,
    format(atom(Lit), '{ .tag = VAL_INT, .data.integer = ~w }', [Int]).
c_static_value_literal(Float, Lit) :-
    float(Float),
    format(atom(Lit), '{ .tag = VAL_FLOAT, .data.floating = ~16g }', [Float]).

lowered_fact_bucket_arrays(Symbol, Rows, ArraysCode, CasesCode, BucketCount) :-
    findall(First, member([First|_], Rows), FirstValues0),
    sort(FirstValues0, FirstValues),
    length(FirstValues, FirstValueCount),
    lowered_fact_bucket_count(FirstValueCount, BucketCount),
    findall(Bucket,
            lowered_fact_bucket_for_values(FirstValues, BucketCount, Bucket),
            Buckets),
    findall(ArrayCode-CaseCode,
            (   member(Bucket, Buckets),
                lowered_fact_bucket_row_indices(Rows, BucketCount, Bucket, RowIndices),
                lowered_fact_bucket_symbols(Symbol, Bucket, ArraySymbol),
                lowered_fact_bucket_array(ArraySymbol, RowIndices, ArrayCode),
                length(RowIndices, RowIndexCount),
                format(atom(CaseCode),
'        case ~w:
            return ~w_scan_rows(state, cells, ~w, ~w);',
                       [Bucket, Symbol, ArraySymbol, RowIndexCount])
            ),
            Pairs),
    findall(Array, member(Array-_, Pairs), Arrays),
    findall(Case, member(_-Case, Pairs), Cases),
    atomic_list_concat(Arrays, '\n', ArraysCode),
    atomic_list_concat(Cases, '\n', CasesCode).

lowered_fact_bucket_row_indices(Rows, BucketCount, Bucket, RowIndices) :-
    findall(Index,
            (   nth0(Index, Rows, [First|_]),
                lowered_fact_first_arg_bucket(First, BucketCount, Bucket)
            ),
            RowIndices).

lowered_fact_bucket_symbols(Symbol, Bucket, ArraySymbol) :-
    format(atom(ArraySymbol), '~w_bucket_~w_rows', [Symbol, Bucket]).

lowered_fact_bucket_array(ArraySymbol, RowIndices, Code) :-
    atomic_list_concat(RowIndices, ', ', RowIndexCode),
    format(atom(Code), 'static const int ~w[] = { ~w };', [ArraySymbol, RowIndexCode]).

lowered_body_call_helper(PredIndicator, AvailableKeys, CalleeKey, CalleeArity) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Args-Body,
            (   user:clause(Head, Body),
                Head =.. [_|Args]
            ),
            Clauses),
    Clauses = [Args-Body],
    Body \== true,
    Body =.. [CalleePred|CalleeArgs],
    length(CalleeArgs, CalleeArity),
    Args == CalleeArgs,
    (Pred/Arity) \== (CalleePred/CalleeArity),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, _Rows),
    format(atom(CalleeKey), '~w/~w', [CalleePred, CalleeArity]),
    memberchk(CalleeKey, AvailableKeys).

lowered_body_call_projected_helper(PredIndicator, AvailableKeys, CalleeKey, CalleeArity, CalleeArgs) :-
    body_call_projection_context(PredIndicator, HeadArgs, CalleePred, CalleeArity, CalleeKey, CalleeArgs),
    format(atom(CalleeKey), '~w/~w', [CalleePred, CalleeArity]),
    memberchk(CalleeKey, AvailableKeys),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, _Rows),
    head_args_project_once(HeadArgs, CalleeArgs),
    callee_local_variables_singletons(HeadArgs, CalleeArgs).

lowered_body_call_projection_row_constrained_helper(PredIndicator, AvailableKeys, CalleeKey, Rows) :-
    body_call_projection_context(PredIndicator, HeadArgs, CalleePred, CalleeArity, CalleeKey, CalleeArgs),
    format(atom(CalleeKey), '~w/~w', [CalleePred, CalleeArity]),
    memberchk(CalleeKey, AvailableKeys),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, CalleeRows),
    head_args_project_at_least_once(HeadArgs, CalleeArgs),
    repeated_head_variable_projection(HeadArgs, CalleeArgs),
    callee_local_variables_singletons(HeadArgs, CalleeArgs),
    include(callee_row_matches_arg_constraints(CalleeArgs), CalleeRows, MatchingRows),
    findall(Projected,
            (   member(CalleeRow, MatchingRows),
                project_callee_row_to_head(HeadArgs, CalleeArgs, CalleeRow, Projected)
            ),
            ProjectedRows0),
    sort(ProjectedRows0, Rows),
    Rows \= [].

lowered_body_call_rejection_reason(PredIndicator, AvailableKeys, Reason) :-
    body_call_context(PredIndicator, CalleePred, CalleeArity, CalleeKey),
    callee_has_user_clause(CalleePred, CalleeArity),
    (   \+ memberchk(CalleeKey, AvailableKeys)
    ->  Reason = body_call_callee_not_available
    ;   CalleeIndicator = user:CalleePred/CalleeArity,
        \+ lowered_fact_helper_rows(CalleeIndicator, _Rows)
    ->  Reason = body_call_callee_not_lowerable
    ),
    !.
lowered_body_call_rejection_reason(PredIndicator, AvailableKeys, Reason) :-
    body_call_projection_context(PredIndicator, HeadArgs, CalleePred, CalleeArity, CalleeKey, CalleeArgs),
    callee_has_user_clause(CalleePred, CalleeArity),
    memberchk(CalleeKey, AvailableKeys),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, _Rows),
    body_call_projection_rejection_reason(HeadArgs, CalleeArgs, Reason),
    !.

body_call_context(PredIndicator, CalleePred, CalleeArity, CalleeKey) :-
    projected_body_call_context(PredIndicator, HeadArgs, CalleePred, CalleeArity, CalleeKey, CalleeArgs),
    HeadArgs == CalleeArgs.

projected_body_call_context(PredIndicator, HeadArgs, CalleePred, CalleeArity, CalleeKey, CalleeArgs) :-
    body_call_projection_context(PredIndicator, HeadArgs, CalleePred, CalleeArity, CalleeKey, CalleeArgs),
    maplist(projected_body_call_arg_supported(HeadArgs), CalleeArgs).

body_call_projection_context(PredIndicator, HeadArgs, CalleePred, CalleeArity, CalleeKey, CalleeArgs) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Args-Body,
            (   user:clause(Head, Body),
                Head =.. [_|Args]
            ),
            [Args-Body0]),
    Body0 \== true,
    strip_module_qualification(Body0, Body),
    Body \= (_, _),
    Body =.. [CalleePred|CalleeArgs],
    length(CalleeArgs, CalleeArity),
    (Pred/Arity) \== (CalleePred/CalleeArity),
    HeadArgs = Args,
    maplist(var, HeadArgs),
    maplist(var, CalleeArgs),
    format(atom(CalleeKey), '~w/~w', [CalleePred, CalleeArity]).

projected_body_call_arg_supported(HeadArgs, Arg) :-
    member(HeadArg, HeadArgs),
    Arg == HeadArg.

body_call_projection_rejection_reason(HeadArgs, CalleeArgs, body_call_projection_omits_head_variable) :-
    member(HeadArg, HeadArgs),
    variable_occurrence_count(HeadArg, CalleeArgs, 0),
    !.
body_call_projection_rejection_reason(HeadArgs, CalleeArgs, body_call_projection_repeats_head_variable) :-
    member(HeadArg, HeadArgs),
    variable_occurrence_count(HeadArg, CalleeArgs, Count),
    Count > 1,
    !.
body_call_projection_rejection_reason(HeadArgs, CalleeArgs, body_call_projection_repeats_callee_local_variable) :-
    member(CalleeArg, CalleeArgs),
    \+ projected_body_call_arg_supported(HeadArgs, CalleeArg),
    variable_occurrence_count(CalleeArg, CalleeArgs, Count),
    Count > 1,
    !.

variable_occurrence_count(Needle, Args, Count) :-
    include(same_variable(Needle), Args, Matches),
    length(Matches, Count).

head_args_project_once([], _CalleeArgs).
head_args_project_once([HeadArg|Rest], CalleeArgs) :-
    variable_occurrence_count(HeadArg, CalleeArgs, 1),
    head_args_project_once(Rest, CalleeArgs).

head_args_project_at_least_once([], _CalleeArgs).
head_args_project_at_least_once([HeadArg|Rest], CalleeArgs) :-
    variable_occurrence_count(HeadArg, CalleeArgs, Count),
    Count >= 1,
    head_args_project_at_least_once(Rest, CalleeArgs).

repeated_head_variable_projection(HeadArgs, CalleeArgs) :-
    member(HeadArg, HeadArgs),
    variable_occurrence_count(HeadArg, CalleeArgs, Count),
    Count > 1,
    !.

callee_local_variables_singletons(_HeadArgs, []).
callee_local_variables_singletons(HeadArgs, [CalleeArg|Rest]) :-
    (   projected_body_call_arg_supported(HeadArgs, CalleeArg)
    ->  true
    ;   variable_occurrence_count(CalleeArg, Rest, 0)
    ),
    callee_local_variables_singletons(HeadArgs, Rest).

same_variable(Left, Right) :-
    Left == Right.

callee_has_user_clause(Pred, Arity) :-
    functor(Head, Pred, Arity),
    catch(user:clause(Head, _Body),
          error(permission_error(access, private_procedure, _), _),
          fail),
    !.

lowered_body_call_helper_for_predicate(PredIndicator, CalleeKey, CalleeArity, Key, Code, SetupLine) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_c_symbol_name(Pred, Arity, Symbol),
    wam_c_symbol_for_key(CalleeKey, CalleeSymbol),
    format(atom(Code),
'static bool ~w(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != ~w) return false;
    return ~w(state, "~w", ~w);
}',
           [Symbol, Arity, CalleeSymbol, CalleeKey, CalleeArity]),
    format(atom(SetupLine),
           '    wam_register_foreign_predicate(state, "~w", ~w, ~w);',
           [Key, Arity, Symbol]).

lowered_body_call_projected_helper_for_predicate(PredIndicator, CalleeKey, CalleeArity, _CalleeArgs, Key, Code, SetupLine) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_c_symbol_name(Pred, Arity, Symbol),
    wam_c_symbol_for_key(CalleeKey, CalleeSymbol),
    projected_body_call_arg_assignments(PredIndicator, ArgAssignments),
    projected_body_call_result_copies(PredIndicator, ResultCopies),
    MaxArity is max(Arity, CalleeArity),
    format(atom(Code),
'static bool ~w(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != ~w) return false;
    WamValue saved_args[~w];
    for (int i = 0; i < ~w; i++) saved_args[i] = state->A[i];
~w
    bool ok = ~w(state, "~w", ~w);
    if (ok) {
        WamValue result_args[~w];
        for (int i = 0; i < ~w; i++) result_args[i] = state->A[i];
~w
    } else {
        for (int i = 0; i < ~w; i++) state->A[i] = saved_args[i];
    }
    for (int i = ~w; i < ~w; i++) state->A[i] = saved_args[i];
    return ok;
}',
           [Symbol, Arity, MaxArity, MaxArity, ArgAssignments, CalleeSymbol, CalleeKey, CalleeArity, CalleeArity, CalleeArity, ResultCopies, Arity, Arity, MaxArity]),
    format(atom(SetupLine),
           '    wam_register_foreign_predicate(state, "~w", ~w, ~w);',
           [Key, Arity, Symbol]).

projected_body_call_arg_assignments(PredIndicator, Code) :-
    projected_body_call_clause_args(PredIndicator, HeadArgs, CalleeArgs),
    findall(Line,
            (   nth0(I, CalleeArgs, Arg),
                projected_body_call_arg_assignment(I, Arg, HeadArgs, Line)
            ),
            Lines),
    atomic_list_concat(Lines, '\n', Code).

projected_body_call_clause_args(PredIndicator, HeadArgs, CalleeArgs) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    user:clause(Head, Body),
    Body \== true,
    Head =.. [_|HeadArgs],
    strip_module_qualification(Body, StrippedBody),
    StrippedBody =.. [_CalleePred|CalleeArgs].

projected_body_call_arg_assignment(I, Arg, _HeadArgs, Line) :-
    wam_c_lowered_constant(Arg),
    !,
    c_value_literal(Arg, ValLit),
    format(atom(Line), '    state->A[~w] = ~w;', [I, ValLit]).
projected_body_call_arg_assignment(I, Arg, HeadArgs, Line) :-
    nth0(HeadIndex, HeadArgs, HeadArg),
    Arg == HeadArg,
    !,
    format(atom(Line), '    state->A[~w] = saved_args[~w];', [I, HeadIndex]).
projected_body_call_arg_assignment(I, _Arg, _HeadArgs, Line) :-
    format(atom(Line), '    state->A[~w] = val_unbound("_");', [I]).

projected_body_call_result_copies(PredIndicator, Code) :-
    projected_body_call_clause_args(PredIndicator, HeadArgs, CalleeArgs),
    findall(Line,
            (   nth0(HeadIndex, HeadArgs, HeadArg),
                nth0(CalleeIndex, CalleeArgs, CalleeArg),
                HeadArg == CalleeArg,
                format(atom(Line), '        state->A[~w] = result_args[~w];', [HeadIndex, CalleeIndex])
            ),
            Lines),
    atomic_list_concat(Lines, '\n', Code).

wam_c_symbol_for_key(Key, Symbol) :-
    atomic_list_concat([PredAtom, ArityAtom], '/', Key),
    atom_number(ArityAtom, Arity),
    wam_c_symbol_name(PredAtom, Arity, Symbol).

lowered_filtered_fact_helper(PredIndicator, CalleeKey, Rows) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(HeadArgs-Body,
            (   user:clause(Head, Body),
                Head =.. [_|HeadArgs]
            ),
            Clauses),
    Clauses = [HeadArgs-Body0],
    Body0 \== true,
    strip_module_qualification(Body0, Body),
    Body \= (_, _),
    Body =.. [CalleePred|CalleeArgs],
    length(CalleeArgs, CalleeArity),
    \+ wam_c_lowered_comparison_guard(CalleePred/CalleeArity),
    (Pred/Arity) \== (CalleePred/CalleeArity),
    maplist(var, HeadArgs),
    callee_args_supported_for_filter(CalleeArgs, HeadArgs),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, CalleeRows),
    include(callee_row_matches_filter(CalleeArgs, HeadArgs), CalleeRows, MatchingRows),
    findall(Projected,
            (   member(CalleeRow, MatchingRows),
                project_callee_row_to_head(HeadArgs, CalleeArgs, CalleeRow, Projected)
            ),
            ProjectedRows0),
    sort(ProjectedRows0, Rows),
    Rows \= [],
    format(atom(CalleeKey), '~w/~w', [CalleePred, CalleeArity]).

lowered_comparison_filtered_fact_helper(PredIndicator, CalleeKey, Rows) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(HeadArgs-Body,
            (   user:clause(Head, Body),
                Head =.. [_|HeadArgs]
            ),
            Clauses),
    Clauses = [HeadArgs-Body0],
    Body0 \== true,
    strip_module_qualification(Body0, Body),
    Body = (Call0, Guard0),
    strip_module_qualification(Call0, Call),
    strip_module_qualification(Guard0, Guard),
    lowered_fact_call_for_filter(Pred/Arity, HeadArgs, Call, CalleeKey, CalleeArgs, CalleeRows),
    Guard =.. [GuardPred, Left, Right],
    wam_c_lowered_comparison_guard(GuardPred/2),
    comparison_arg_supported(Left, CalleeArgs),
    comparison_arg_supported(Right, CalleeArgs),
    include(callee_row_matches_comparison_filter(CalleeArgs, GuardPred, Left, Right), CalleeRows, MatchingRows),
    findall(Projected,
            (   member(CalleeRow, MatchingRows),
                project_callee_row_to_head(HeadArgs, CalleeArgs, CalleeRow, Projected)
            ),
            ProjectedRows0),
    sort(ProjectedRows0, Rows),
    Rows \= [].

lowered_fact_call_for_filter(CurrentPred/CurrentArity, HeadArgs, Call, CalleeKey, CalleeArgs, CalleeRows) :-
    Call =.. [CalleePred|CalleeArgs],
    length(CalleeArgs, CalleeArity),
    \+ wam_c_lowered_comparison_guard(CalleePred/CalleeArity),
    (CurrentPred/CurrentArity) \== (CalleePred/CalleeArity),
    maplist(var, HeadArgs),
    maplist(callee_filter_arg_supported, CalleeArgs),
    CalleeIndicator = user:CalleePred/CalleeArity,
    lowered_fact_helper_rows(CalleeIndicator, CalleeRows),
    format(atom(CalleeKey), '~w/~w', [CalleePred, CalleeArity]).

callee_filter_arg_supported(Arg) :-
    var(Arg),
    !.
callee_filter_arg_supported(Arg) :-
    wam_c_lowered_constant(Arg).

comparison_arg_supported(Arg, _CalleeArgs) :-
    wam_c_lowered_constant(Arg),
    !.
comparison_arg_supported(Arg, CalleeArgs) :-
    member(CalleeArg, CalleeArgs),
    Arg == CalleeArg.

comparison_arg_basic(Arg, CalleeArgs) :-
    comparison_arg_supported(Arg, CalleeArgs),
    !.
comparison_arg_basic(Arg, _CalleeArgs) :-
    var(Arg),
    !.

comparison_arg_unbound_variable(Arg, CalleeArgs) :-
    var(Arg),
    \+ ( member(CalleeArg, CalleeArgs),
         Arg == CalleeArg
       ).

callee_row_matches_comparison_filter(CalleeArgs, GuardPred, Left, Right, CalleeRow) :-
    callee_row_matches_arg_constraints(CalleeArgs, CalleeRow),
    comparison_arg_value(Left, CalleeArgs, CalleeRow, LeftValue),
    comparison_arg_value(Right, CalleeArgs, CalleeRow, RightValue),
    wam_c_eval_lowered_comparison_guard(GuardPred, LeftValue, RightValue).

callee_row_matches_arg_constraints(CalleeArgs, CalleeRow) :-
    same_length(CalleeArgs, CalleeRow),
    \+ ( nth0(I, CalleeArgs, Arg),
         nth0(I, CalleeRow, Value),
         wam_c_lowered_constant(Arg),
         Arg \== Value
       ),
    \+ ( nth0(I, CalleeArgs, Arg),
         var(Arg),
         nth0(J, CalleeArgs, OtherArg),
         J > I,
         Arg == OtherArg,
         nth0(I, CalleeRow, Value),
         nth0(J, CalleeRow, OtherValue),
         Value \== OtherValue
       ).

comparison_arg_value(Arg, _CalleeArgs, _CalleeRow, Arg) :-
    wam_c_lowered_constant(Arg),
    !.
comparison_arg_value(Arg, CalleeArgs, CalleeRow, Value) :-
    nth0(Index, CalleeArgs, CalleeArg),
    Arg == CalleeArg,
    !,
    nth0(Index, CalleeRow, Value).

wam_c_eval_lowered_comparison_guard((=), Left, Right) :- Left == Right.
wam_c_eval_lowered_comparison_guard((==), Left, Right) :- Left == Right.
wam_c_eval_lowered_comparison_guard((\==), Left, Right) :- Left \== Right.
wam_c_eval_lowered_comparison_guard((>), Left, Right) :- integer(Left), integer(Right), Left > Right.
wam_c_eval_lowered_comparison_guard((<), Left, Right) :- integer(Left), integer(Right), Left < Right.
wam_c_eval_lowered_comparison_guard((>=), Left, Right) :- integer(Left), integer(Right), Left >= Right.
wam_c_eval_lowered_comparison_guard((=<), Left, Right) :- integer(Left), integer(Right), Left =< Right.
wam_c_eval_lowered_comparison_guard((=:=), Left, Right) :- integer(Left), integer(Right), Left =:= Right.
wam_c_eval_lowered_comparison_guard((=\=), Left, Right) :- integer(Left), integer(Right), Left =\= Right.

strip_module_qualification(Module:Body, Stripped) :-
    atom(Module),
    !,
    strip_module_qualification(Body, Stripped).
strip_module_qualification(Body, Body).

callee_args_supported_for_filter([], _).
callee_args_supported_for_filter([Arg|Rest], HeadArgs) :-
    lowered_filter_arg_supported(Arg, HeadArgs),
    callee_args_supported_for_filter(Rest, HeadArgs).

lowered_filter_arg_supported(Arg, HeadArgs) :-
    lowered_filter_arg_supported_(HeadArgs, Arg).

lowered_filter_arg_supported_(HeadArgs, Arg) :-
    (   member(HeadArg, HeadArgs),
        Arg == HeadArg
    ->  true
    ;   wam_c_lowered_constant(Arg)
    ).

callee_row_matches_filter(CalleeArgs, HeadArgs, CalleeRow) :-
    same_length(CalleeArgs, CalleeRow),
    callee_row_matches_arg_constraints(CalleeArgs, CalleeRow),
    callee_row_matches_filter_(CalleeArgs, HeadArgs, CalleeRow).

callee_row_matches_filter_([], _, []).
callee_row_matches_filter_([Arg|ArgRest], HeadArgs, [Value|ValueRest]) :-
    (   wam_c_lowered_constant(Arg)
    ->  Arg == Value
    ;   member(HeadArg, HeadArgs),
        Arg == HeadArg
    ->  true
    ),
    callee_row_matches_filter_(ArgRest, HeadArgs, ValueRest).

project_callee_row_to_head([], _, _, []).
project_callee_row_to_head([HeadArg|Rest], CalleeArgs, CalleeRow, [Value|Values]) :-
    nth0(Index, CalleeArgs, CalleeArg),
    CalleeArg == HeadArg,
    !,
    nth0(Index, CalleeRow, Value),
    project_callee_row_to_head(Rest, CalleeArgs, CalleeRow, Values).

wam_c_lowered_constant(Arg) :- atom(Arg), !.
wam_c_lowered_constant(Arg) :- integer(Arg).

wam_c_symbol_name(Pred, Arity, Symbol) :-
    atom_chars(Pred, Chars),
    maplist(wam_c_symbol_char, Chars, SafeChars),
    atom_chars(SafePred, SafeChars),
    format(atom(Symbol), 'wam_c_lowered_~w_~w', [SafePred, Arity]).

wam_c_symbol_char(Char, Char) :-
    char_type(Char, alnum), !.
wam_c_symbol_char('_', '_') :- !.
wam_c_symbol_char(_, '_').

wam_c_lowered_fact_dispatch(0, Rows, Code) :-
    !,
    maplist(wam_c_lowered_fact_row(0), Rows, RowCodes),
    atomic_list_concat(RowCodes, '\n', RowCode),
    format(atom(Code), '~w', [RowCode]).
wam_c_lowered_fact_dispatch(Arity, Rows, Code) :-
    Arity > 0,
    maplist(wam_c_lowered_fact_row(Arity), Rows, FallbackRowCodes),
    atomic_list_concat(FallbackRowCodes, '\n', FallbackCode),
    lowered_fact_first_arg_dispatch(Rows, DispatchCode),
    format(atom(Code),
'    WamValue *cells[~w];
    for (int i = 0; i < ~w; i++) cells[i] = wam_deref_ptr(state, &state->A[i]);
~w
~w',
           [Arity, Arity, DispatchCode, FallbackCode]).

lowered_fact_first_arg_dispatch(Rows, Code) :-
    findall(First, member([First|_], Rows), FirstValues0),
    sort(FirstValues0, FirstValues),
    length(FirstValues, FirstValueCount),
    lowered_fact_bucket_count(FirstValueCount, BucketCount),
    findall(CaseCode,
            (   lowered_fact_bucket_for_values(FirstValues, BucketCount, Bucket),
                include(first_arg_in_bucket(BucketCount, Bucket), FirstValues, BucketValues),
                lowered_fact_bucket_case(BucketCount, Bucket, BucketValues, Rows, CaseCode)
            ),
            CaseCodes),
    atomic_list_concat(CaseCodes, '\n', CasesCode),
    Mask is BucketCount - 1,
    format(atom(Code),
'    if (!val_is_unbound(*cells[0])) {
        unsigned int bucket = 0;
        if (cells[0]->tag == VAL_ATOM) {
            bucket = wam_hash_string(cells[0]->data.atom) & ~w;
        } else if (cells[0]->tag == VAL_INT) {
            bucket = ((unsigned int)cells[0]->data.integer * 2654435761u) & ~w;
        } else {
            return false;
        }
        switch (bucket) {
~w
        default:
            return false;
        }
    }',
           [Mask, Mask, CasesCode]).

lowered_fact_bucket_count(FirstValueCount, BucketCount) :-
    Needed is max(8, FirstValueCount * 2),
    lowered_fact_bucket_count_(1, Needed, BucketCount).

lowered_fact_bucket_count_(Current, Needed, Current) :-
    Current >= Needed,
    !.
lowered_fact_bucket_count_(Current, Needed, BucketCount) :-
    Next is Current * 2,
    lowered_fact_bucket_count_(Next, Needed, BucketCount).

lowered_fact_bucket_for_values(FirstValues, BucketCount, Bucket) :-
    findall(B, (member(Value, FirstValues), lowered_fact_first_arg_bucket(Value, BucketCount, B)), Buckets0),
    sort(Buckets0, Buckets),
    member(Bucket, Buckets).

first_arg_in_bucket(BucketCount, Bucket, Value) :-
    lowered_fact_first_arg_bucket(Value, BucketCount, Bucket).

lowered_fact_bucket_case(_BucketCount, Bucket, BucketValues, Rows, Code) :-
    findall(GroupCode,
            (   member(First, BucketValues),
                include(row_first_arg_equals(First), Rows, GroupRows),
                lowered_fact_first_arg_group(First, GroupRows, GroupCode)
            ),
            GroupCodes),
    atomic_list_concat(GroupCodes, '\n', GroupsCode),
    format(atom(Code),
'        case ~w:
~w
            return false;',
           [Bucket, GroupsCode]).

lowered_fact_first_arg_bucket(Value, BucketCount, Bucket) :-
    lowered_fact_first_arg_hash(Value, Hash),
    Mask is BucketCount - 1,
    Bucket is Hash /\ Mask.

lowered_fact_first_arg_hash(Value, Hash) :-
    atom(Value),
    !,
    atom_codes(Value, Codes),
    foldl(wam_c_lowered_hash_code, Codes, 5381, Hash).
lowered_fact_first_arg_hash(Value, Hash) :-
    integer(Value),
    Hash is (Value * 2654435761) /\ 0xffffffff.

wam_c_lowered_hash_code(Code, Acc, Hash) :-
    Hash is (((Acc << 5) + Acc) xor Code) /\ 0xffffffff.

row_first_arg_equals(First, [First|_]).

lowered_fact_first_arg_group(First, Rows, Code) :-
    c_value_literal(First, FirstLit),
    maplist(wam_c_lowered_fact_indexed_row, Rows, RowCodes),
    atomic_list_concat(RowCodes, '\n', RowCode),
    format(atom(Code),
'        if (val_equal(*cells[0], ~w)) {
~w
            return false;
        }',
           [FirstLit, RowCode]).

wam_c_lowered_fact_indexed_row(Args, Code) :-
    wam_c_lowered_fact_row_body(Args, BodyCode),
    format(atom(Code),
'            {
~w
            }',
           [BodyCode]).

wam_c_lowered_fact_row(Arity, Args, Code) :-
    Last is Arity - 1,
    findall(DerefLine,
            (   between(0, Last, I),
                format(atom(DerefLine),
                       '        WamValue *cell_~w = wam_deref_ptr(state, &state->A[~w]);',
                       [I, I])
            ),
            DerefLines),
    atomic_list_concat(DerefLines, '\n', DerefCode),
    wam_c_lowered_fact_row_body_with_cell_prefix('cell_', Args, BodyCode),
    format(atom(Code),
'    {
~w
~w
    }',
           [DerefCode, BodyCode]).

wam_c_lowered_fact_row_body(Args, Code) :-
    wam_c_lowered_fact_row_body_with_cell_prefix('cells', Args, Code).

wam_c_lowered_fact_row_body_with_cell_prefix(CellPrefix, Args, Code) :-
    findall(MatchLine,
            (   nth0(I, Args, Arg),
                c_value_literal(Arg, ValLit),
                lowered_fact_cell_expr(CellPrefix, I, CellExpr),
                format(atom(MatchLine),
                       '        if (!val_is_unbound(*~w) && !val_equal(*~w, ~w)) match = false;',
                       [CellExpr, CellExpr, ValLit])
            ),
            MatchLines),
    findall(BindLine,
            (   nth0(I, Args, Arg),
                c_value_literal(Arg, ValLit),
                lowered_fact_cell_expr(CellPrefix, I, CellExpr),
                format(atom(BindLine),
                       '            if (val_is_unbound(*~w)) { trail_binding(state, ~w); *~w = ~w; }',
                       [CellExpr, CellExpr, CellExpr, ValLit])
            ),
            BindLines),
    atomic_list_concat(MatchLines, '\n', MatchCode),
    atomic_list_concat(BindLines, '\n', BindCode),
    format(atom(Code),
'        bool match = true;
~w
        if (match) {
~w
            return true;
        }',
           [MatchCode, BindCode]).

lowered_fact_cell_expr('cell_', I, CellExpr) :-
    !,
    format(atom(CellExpr), 'cell_~w', [I]).
lowered_fact_cell_expr(cells, I, CellExpr) :-
    format(atom(CellExpr), 'cells[~w]', [I]).

% ============================================================================
% PHASE 2: WAM instructions -> C Struct Literals
% ============================================================================

%% wam_instruction_to_c_literal(+WamInstr, -CCode)
wam_instruction_to_c_literal(get_constant(C, Ai), Code) :-
    c_value_literal(C, Val), c_reg_index(Ai, IsY_Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_GET_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY_Ai]).
wam_instruction_to_c_literal(get_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(get_value(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_constant(C, Ai), Code) :-
    c_value_literal(C, Val), c_reg_index(Ai, IsY_Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_PUT_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY_Ai]).
wam_instruction_to_c_literal(put_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_value(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(get_structure(F, Ai), Code) :-
    c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [F, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_structure(F, Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [F, XIdx, IsY_Xn]).
wam_instruction_to_c_literal(get_list(Ai), Code) :-
    c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_list(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(set_variable(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_SET_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(set_value(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_SET_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(set_constant(C), Code) :-
    c_value_literal(C, Val),
    format(atom(Code), '{ .tag = INSTR_SET_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_instruction_to_c_literal(unify_variable(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_UNIFY_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(unify_value(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_UNIFY_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(unify_constant(C), Code) :-
    c_value_literal(C, Val),
    format(atom(Code), '{ .tag = INSTR_UNIFY_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_instruction_to_c_literal(call(P, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [P, N]).
wam_instruction_to_c_literal(execute(P), Code) :-
    format(atom(Code), '{ .tag = INSTR_EXECUTE, .as.pred = { .pred = "~w" } }', [P]).
wam_instruction_to_c_literal(builtin_call(Op, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_BUILTIN_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [Op, N]).
wam_instruction_to_c_literal(call_foreign(P, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_CALL_FOREIGN, .as.pred = { .pred = "~w", .arity = ~w } }', [P, N]).
wam_instruction_to_c_literal(begin_aggregate(Kind, TemplateReg, ResultReg), Code) :-
    c_reg_index(TemplateReg, TemplateIsY, TemplateIdx),
    c_reg_index(ResultReg, ResultIsY, ResultIdx),
    format(atom(Code), '{ .tag = INSTR_BEGIN_AGGREGATE, .as.aggregate = { .kind = "~w", .template_reg = ~w, .template_is_y = ~w, .result_reg = ~w, .result_is_y = ~w, .witness_count = 0, .witness_regs = {0}, .witness_is_y = {0} } }',
           [Kind, TemplateIdx, TemplateIsY, ResultIdx, ResultIsY]).
wam_instruction_to_c_literal(begin_aggregate(Kind, TemplateReg, ResultReg, WitnessRegs), Code) :-
    clean_aggregate_witness(WitnessRegs, WitnessString),
    aggregate_witness_fields(WitnessString, WitnessCount, WitnessRegInit, WitnessIsYInit),
    c_reg_index(TemplateReg, TemplateIsY, TemplateIdx),
    c_reg_index(ResultReg, ResultIsY, ResultIdx),
    format(atom(Code), '{ .tag = INSTR_BEGIN_AGGREGATE, .as.aggregate = { .kind = "~w", .template_reg = ~w, .template_is_y = ~w, .result_reg = ~w, .result_is_y = ~w, .witness_count = ~w, .witness_regs = ~w, .witness_is_y = ~w } }',
           [Kind, TemplateIdx, TemplateIsY, ResultIdx, ResultIsY,
            WitnessCount, WitnessRegInit, WitnessIsYInit]).
wam_instruction_to_c_literal(end_aggregate(TemplateReg), Code) :-
    c_reg_index(TemplateReg, TemplateIsY, TemplateIdx),
    format(atom(Code), '{ .tag = INSTR_END_AGGREGATE, .as.aggregate = { .kind = "collect", .template_reg = ~w, .template_is_y = ~w, .result_reg = 0, .result_is_y = 0, .witness_count = 0, .witness_regs = {0}, .witness_is_y = {0} } }',
           [TemplateIdx, TemplateIsY]).
wam_instruction_to_c_literal(get_level(Reg), Code) :-
    c_reg_index(Reg, IsY, Idx),
    format(atom(Code), '{ .tag = INSTR_GET_LEVEL, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_instruction_to_c_literal(cut(Reg), Code) :-
    c_reg_index(Reg, IsY, Idx),
    format(atom(Code), '{ .tag = INSTR_CUT, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_instruction_to_c_literal(cut_ite, '{ .tag = INSTR_CUT_ITE }').
wam_instruction_to_c_literal(jump(_Label), _) :-
    throw(error(context_error(missing_label_map, "jump/1 requires LabelMap for target_pc resolution. Use wam_instruction_to_c_literal/3 instead."), _)).
wam_instruction_to_c_literal(try_me_else(_Label), _) :-
    throw(error(context_error(missing_label_map, "try_me_else/1 requires LabelMap for target_pc resolution. Use wam_instruction_to_c_literal/3 instead."), _)).
wam_instruction_to_c_literal(retry_me_else(_Label), _) :-
    throw(error(context_error(missing_label_map, "retry_me_else/1 requires LabelMap for target_pc resolution. Use wam_instruction_to_c_literal/3 instead."), _)).


wam_instruction_to_c_literal(trust_me, '{ .tag = INSTR_TRUST_ME }').
wam_instruction_to_c_literal(proceed, '{ .tag = INSTR_PROCEED }').
wam_instruction_to_c_literal(allocate, '{ .tag = INSTR_ALLOCATE }').
wam_instruction_to_c_literal(deallocate, '{ .tag = INSTR_DEALLOCATE }').
wam_instruction_to_c_literal(Instr, _) :-
    throw(error(wam_c_target_error(unsupported_instruction(Instr)), _)).

wam_instruction_to_c_literal(try_me_else(Label), LabelMap, Code) :-
    ( member(Label-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Code), '{ .tag = INSTR_TRY_ME_ELSE, .as.choice = { .target_pc = ~w } }', [TargetPC]).
wam_instruction_to_c_literal(retry_me_else(Label), LabelMap, Code) :-
    ( member(Label-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Code), '{ .tag = INSTR_RETRY_ME_ELSE, .as.choice = { .target_pc = ~w } }', [TargetPC]).
wam_instruction_to_c_literal(jump(Label), LabelMap, Code) :-
    ( member(Label-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Code), '{ .tag = INSTR_JUMP, .as.jump = { .target_pc = ~w } }', [TargetPC]).
wam_instruction_to_c_literal(Instr, _, Code) :- wam_instruction_to_c_literal(Instr, Code).


c_value_literal(Str, Lit) :-
    string(Str),
    (   number_string(Number, Str)
    ->  c_value_literal(Number, Lit)
    ;   atom_string(Atom, Str),
        c_value_literal(Atom, Lit)
    ).
c_value_literal(Atom, Lit) :- atom(Atom), format(atom(Lit), 'val_atom("~w")', [Atom]).
c_value_literal(Int, Lit) :- integer(Int), !, format(atom(Lit), 'val_int(~w)', [Int]).
c_value_literal(Float, Lit) :- float(Float), format(atom(Lit), 'val_float(~16g)', [Float]).

c_reg_index(RegStr, IsY, Idx) :-
    string(RegStr),
    atom_string(RegAtom, RegStr),
    c_reg_index(RegAtom, IsY, Idx).
c_reg_index(RegAtom, IsY, Idx) :-
    atom_chars(RegAtom, Chars),
    (   Chars = [Prefix|NumChars],
        (Prefix == 'a'; Prefix == 'A')
    ->  IsY = 0,
        catch(number_chars(RegNo, NumChars), _, fail),
        Idx is RegNo - 1
    ;   Chars = [Prefix|NumChars],
        (Prefix == 'x'; Prefix == 'X')
    ->  IsY = 2,
        catch(number_chars(RegNo, NumChars), _, fail),
        Idx is RegNo - 1
    ;   Chars = [Prefix|NumChars],
        (Prefix == 'y'; Prefix == 'Y')
    ->  IsY = 1,
        catch(number_chars(RegNo, NumChars), _, fail),
        Idx is RegNo - 1
    ;   append(['_', 'X', 'T'], NumChars, Chars)
    ->  IsY = 2,
        catch(number_chars(TempNo, NumChars), _, fail),
        Idx is 128 + TempNo
    ;   throw(error(wam_c_target_error(unknown_register(RegAtom)), _))
    ).

% ============================================================================
% PHASE 2b: wam_predicate -> C Array
% ============================================================================

% wam_line_to_c_instr/2, /3, /4
% Note: wam_line_to_c_instr has 2-arity, 3-arity, and 4-arity clauses.
% The 4-arity clauses are used for branch instructions (like try_me_else) that require the predicate's Arity.
% Non-branch instructions safely fall back to the 3-arity or 2-arity catch-alls during pass 2.
wam_line_to_c_instr(["get_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    c_value_literal(CC, Val), c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY]).
wam_line_to_c_instr(["get_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_GET_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["get_value", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_GET_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["put_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    c_value_literal(CC, Val), c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY]).
wam_line_to_c_instr(["put_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_PUT_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["put_value", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_PUT_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["get_structure", F, Ai], Instr) :-
    clean_comma(F, CF), clean_comma(Ai, CAi),
    c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [CF, Idx, IsY]).
wam_line_to_c_instr(["put_structure", F, Xn], Instr) :-
    clean_comma(F, CF), clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [CF, Idx, IsY]).
wam_line_to_c_instr(["get_list", Ai], Instr) :-
    clean_comma(Ai, CAi),
    c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["put_list", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["set_variable", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_SET_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["set_value", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_SET_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["set_constant", C], Instr) :-
    clean_comma(C, CC),
    c_value_literal(CC, Val),
    format(atom(Instr), '{ .tag = INSTR_SET_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_line_to_c_instr(["unify_variable", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["unify_value", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["unify_constant", C], Instr) :-
    clean_comma(C, CC),
    c_value_literal(CC, Val),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_line_to_c_instr(["call", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(atom(Instr), '{ .tag = INSTR_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [CP, CN]).
wam_line_to_c_instr(["execute", P], Instr) :-
    clean_comma(P, CP),
    format(atom(Instr), '{ .tag = INSTR_EXECUTE, .as.pred = { .pred = "~w" } }', [CP]).
wam_line_to_c_instr(["builtin_call", Op, N], Instr) :-
    clean_comma(Op, COp0), clean_comma(N, CN), c_escape_atom(COp0, COp),
    format(atom(Instr), '{ .tag = INSTR_BUILTIN_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [COp, CN]).
wam_line_to_c_instr(["call_foreign", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(atom(Instr), '{ .tag = INSTR_CALL_FOREIGN, .as.pred = { .pred = "~w", .arity = ~w } }', [CP, CN]).
wam_line_to_c_instr(["begin_aggregate", Kind, TemplateReg, ResultReg], Instr) :-
    clean_comma(Kind, CKind0), clean_comma(TemplateReg, CTemplateReg),
    clean_comma(ResultReg, CResultReg), c_escape_atom(CKind0, CKind),
    c_reg_index(CTemplateReg, TemplateIsY, TemplateIdx),
    c_reg_index(CResultReg, ResultIsY, ResultIdx),
    format(atom(Instr), '{ .tag = INSTR_BEGIN_AGGREGATE, .as.aggregate = { .kind = "~w", .template_reg = ~w, .template_is_y = ~w, .result_reg = ~w, .result_is_y = ~w, .witness_count = 0, .witness_regs = {0}, .witness_is_y = {0} } }',
           [CKind, TemplateIdx, TemplateIsY, ResultIdx, ResultIsY]).
wam_line_to_c_instr(["begin_aggregate", Kind, TemplateReg, ResultReg, WitnessRegs], Instr) :-
    clean_comma(Kind, CKind0), clean_comma(TemplateReg, CTemplateReg),
    clean_comma(ResultReg, CResultReg), c_escape_atom(CKind0, CKind),
    clean_aggregate_witness(WitnessRegs, WitnessString),
    aggregate_witness_fields(WitnessString, WitnessCount, WitnessRegInit, WitnessIsYInit),
    c_reg_index(CTemplateReg, TemplateIsY, TemplateIdx),
    c_reg_index(CResultReg, ResultIsY, ResultIdx),
    format(atom(Instr), '{ .tag = INSTR_BEGIN_AGGREGATE, .as.aggregate = { .kind = "~w", .template_reg = ~w, .template_is_y = ~w, .result_reg = ~w, .result_is_y = ~w, .witness_count = ~w, .witness_regs = ~w, .witness_is_y = ~w } }',
           [CKind, TemplateIdx, TemplateIsY, ResultIdx, ResultIsY,
            WitnessCount, WitnessRegInit, WitnessIsYInit]).
wam_line_to_c_instr(["end_aggregate", TemplateReg], Instr) :-
    clean_comma(TemplateReg, CTemplateReg),
    c_reg_index(CTemplateReg, TemplateIsY, TemplateIdx),
    format(atom(Instr), '{ .tag = INSTR_END_AGGREGATE, .as.aggregate = { .kind = "collect", .template_reg = ~w, .template_is_y = ~w, .result_reg = 0, .result_is_y = 0, .witness_count = 0, .witness_regs = {0}, .witness_is_y = {0} } }',
           [TemplateIdx, TemplateIsY]).
wam_line_to_c_instr(["get_level", Reg], Instr) :-
    clean_comma(Reg, CReg),
    c_reg_index(CReg, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_LEVEL, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["cut", Reg], Instr) :-
    clean_comma(Reg, CReg),
    c_reg_index(CReg, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_CUT, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["jump", L], LabelMap, _Arity, OffsetVar, Instr) :-
    clean_comma(L, CL),
    ( member(CL-TargetPC0, LabelMap) -> c_pc_expr(OffsetVar, TargetPC0, TargetPC) ; TargetPC = -1 ),
    format(atom(Instr), '{ .tag = INSTR_JUMP, .as.jump = { .target_pc = ~w } }', [TargetPC]).
wam_line_to_c_instr(["try_me_else", L], LabelMap, Arity, OffsetVar, Instr) :-
    clean_comma(L, CL),
    ( member(CL-TargetPC0, LabelMap) -> c_pc_expr(OffsetVar, TargetPC0, TargetPC) ; TargetPC = -1 ),
    format(atom(Instr), '{ .tag = INSTR_TRY_ME_ELSE, .as.choice = { .target_pc = ~w, .arity = ~w } }', [TargetPC, Arity]).
wam_line_to_c_instr(["retry_me_else", L], LabelMap, Arity, OffsetVar, Instr) :-
    clean_comma(L, CL),
    ( member(CL-TargetPC0, LabelMap) -> c_pc_expr(OffsetVar, TargetPC0, TargetPC) ; TargetPC = -1 ),
    format(atom(Instr), '{ .tag = INSTR_RETRY_ME_ELSE, .as.choice = { .target_pc = ~w, .arity = ~w } }', [TargetPC, Arity]).
wam_line_to_c_instr(["trust_me"], _, '{ .tag = INSTR_TRUST_ME }').
wam_line_to_c_instr(["cut_ite"], _, '{ .tag = INSTR_CUT_ITE }').
wam_line_to_c_instr(["proceed"], _, '{ .tag = INSTR_PROCEED }').
wam_line_to_c_instr(["allocate"], _, '{ .tag = INSTR_ALLOCATE }').
wam_line_to_c_instr(["deallocate"], _, '{ .tag = INSTR_DEALLOCATE }').
wam_line_to_c_instr(Parts, _, _) :-
    throw(error(wam_c_target_error(unsupported_instruction_tokens(Parts)), _)).

clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).

clean_aggregate_witness(Witness0, Witness) :-
    clean_comma(Witness0, Clean0),
    (   string(Clean0)
    ->  S0 = Clean0
    ;   atom(Clean0)
    ->  atom_string(Clean0, S0)
    ;   term_string(Clean0, S0)
    ),
    (   sub_string(S0, 0, 1, _, "'"),
        sub_string(S0, _, 1, 0, "'")
    ->  sub_string(S0, 1, _, 1, Witness)
    ;   Witness = S0
    ).

aggregate_witness_fields("", 0, "{0}", "{0}") :- !.
aggregate_witness_fields(WitnessString, Count, RegInit, IsYInit) :-
    split_string(WitnessString, ";", "", WitnessRegs0),
    exclude(=(""), WitnessRegs0, WitnessRegs),
    length(WitnessRegs, Count),
    (   Count =< 8
    ->  true
    ;   throw(error(wam_c_target_error(too_many_bagof_setof_witnesses(Count)), _))
    ),
    maplist(aggregate_witness_reg_indices, WitnessRegs, RegIndices, IsYIndices),
    atomic_list_concat(RegIndices, ', ', RegBody),
    atomic_list_concat(IsYIndices, ', ', IsYBody),
    format(atom(RegInit), '{~w}', [RegBody]),
    format(atom(IsYInit), '{~w}', [IsYBody]).

aggregate_witness_reg_indices(WitnessReg, RegIdx, IsY) :-
    c_reg_index(WitnessReg, IsY, RegIdx).

%% c_escape_atom(+Atom, -Escaped)
%  Escape backslashes and double quotes so the value is safe inside a C
%  string literal. Functors carry them: =\=/2 must be emitted as "=\\=/2",
%  otherwise the C compiler reads \= as an unknown escape and drops the
%  backslash, so the runtime functor ("==/2") no longer matches its handler
%  and arithmetic disequality silently fails.
c_escape_atom(Atom, Escaped) :-
    atom_string(Atom, S0),
    split_string(S0, "\\", "", BSParts),
    atomic_list_concat(BSParts, '\\\\', S1),
    split_string(S1, "\"", "", QParts),
    atomic_list_concat(QParts, '\\"', Escaped).

wam_lines_to_c_pass1([], _, []).
wam_lines_to_c_pass1([Line|Rest], PC, LabelMap) :-
    split_string(Line, " \t", " \t", Parts),  % comma intentionally excluded: entries are now space-separated
    delete(Parts, "", CleanParts),
    (   CleanParts == [] -> wam_lines_to_c_pass1(Rest, PC, LabelMap)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            LabelMap = [LabelName-PC|RestMap],
            wam_lines_to_c_pass1(Rest, PC, RestMap)
        ;   NPC is PC + 1,
            wam_lines_to_c_pass1(Rest, NPC, LabelMap)
        )
    ).

wam_lines_to_c_pass2([], PC, _, _, _, PC, []).
wam_lines_to_c_pass2([Line|Rest], PC, LabelMap, Arity, OffsetVar, CodeSize, Instrs) :-
    split_string(Line, " \t", " \t", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == [] -> wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, OffsetVar, CodeSize, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            (   sub_string(LabelName, 0, 2, _, "L_")
            ->  wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, OffsetVar, CodeSize, Instrs)
            ;   c_pc_expr(OffsetVar, PC, PCExpr),
                format(atom(PredReg), '    wam_register_predicate_hash(state, "~w", ~w);', [LabelName, PCExpr]),
                Instrs = [PredReg|RestInstrs],
                wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, OffsetVar, CodeSize, RestInstrs)
            )
        ;   wam_generate_c_instruction(PC, CleanParts, LabelMap, Arity, OffsetVar, CodeLines),
            NPC is PC + 1,
            append(CodeLines, RestInstrs, Instrs),
            wam_lines_to_c_pass2(Rest, NPC, LabelMap, Arity, OffsetVar, CodeSize, RestInstrs)
        )
    ).

wam_generate_c_instruction(PC, Parts, LabelMap, Arity, OffsetVar, CodeLines) :-
    c_pc_expr(OffsetVar, PC, PCExpr),
    (   (   Parts = ["switch_on_constant" | Entries],
            SwitchReg = 0,
            Fallthrough = false
        ;   Parts = ["switch_on_constant_a2" | Entries],
            SwitchReg = 1,
            Fallthrough = false
        ;   Parts = ["switch_on_constant_fallthrough" | Entries],
            SwitchReg = 0,
            Fallthrough = true
        ;   Parts = ["switch_on_constant_a2_fallthrough" | Entries],
            SwitchReg = 1,
            Fallthrough = true
        )
    ->  length(Entries, HashSize),
        format(atom(L0), '    state->code[~w] = (Instruction){ .tag = INSTR_SWITCH_ON_CONSTANT, .as.switch_index = { .reg = ~w, .hash_size = ~w, .no_match_fallthrough = ~w } };', [PCExpr, SwitchReg, HashSize, Fallthrough]),
        format(atom(L1), '    state->code[~w].as.switch_index.hash_table = malloc(sizeof(HashEntry) * ~w);', [PCExpr, HashSize]),
        generate_hash_table_entries(PC, PCExpr, "as.switch_index.hash_table", Entries, 0, LabelMap, OffsetVar, HashLines),
        append([L0, L1], HashLines, CodeLines)
    ;   Parts = ["switch_on_structure" | Entries]
    ->  length(Entries, HashSize),
        format(atom(L0), '    state->code[~w] = (Instruction){ .tag = INSTR_SWITCH_ON_STRUCTURE, .as.switch_index = { .hash_size = ~w } };', [PCExpr, HashSize]),
        format(atom(L1), '    state->code[~w].as.switch_index.hash_table = malloc(sizeof(HashEntry) * ~w);', [PCExpr, HashSize]),
        generate_hash_table_entries(PC, PCExpr, "as.switch_index.hash_table", Entries, 0, LabelMap, OffsetVar, HashLines),
        append([L0, L1], HashLines, CodeLines)
    ;   Parts = ["switch_on_term", CLenStr | Rest1]
    ->  number_string(CLen, CLenStr),
        length(CEntries, CLen),
        append(CEntries, [SLenStr | Rest2], Rest1),
        number_string(SLen, SLenStr),
        length(SEntries, SLen),
        append(SEntries, [ListLabelStr], Rest2),
        (   ListLabelStr == "none"
        ->  ListPC = -1
        ;   ListLabelStr == "default"
        ->  ListPC0 is PC + 1,
            c_pc_expr(OffsetVar, ListPC0, ListPC)
        ;   member(ListLabelStr-ListPC0, LabelMap)
        ->  c_pc_expr(OffsetVar, ListPC0, ListPC)
        ;   ListPC = -1
        ),
        format(atom(L0), '    state->code[~w] = (Instruction){ .tag = INSTR_SWITCH_ON_TERM, .as.switch_index = { .hash_size = ~w, .s_hash_size = ~w, .list_target_pc = ~w } };', [PCExpr, CLen, SLen, ListPC]),
        format(atom(L1), '    state->code[~w].as.switch_index.hash_table = malloc(sizeof(HashEntry) * ~w);', [PCExpr, CLen]),
        format(atom(L2), '    state->code[~w].as.switch_index.s_hash_table = malloc(sizeof(HashEntry) * ~w);', [PCExpr, SLen]),
        generate_hash_table_entries(PC, PCExpr, "as.switch_index.hash_table", CEntries, 0, LabelMap, OffsetVar, CHashLines),
        generate_hash_table_entries(PC, PCExpr, "as.switch_index.s_hash_table", SEntries, 0, LabelMap, OffsetVar, SHashLines),
        append([L0, L1, L2 | CHashLines], SHashLines, CodeLines)
    ;   Parts = [SwitchOp|_], string_concat("switch_on_", _, SwitchOp)
    ->  % Unhandled first/second-argument indexing hint (e.g.
        % switch_on_term_a2): degrade to a NoOp rather than throwing.
        % Indexing is only an optimisation, so falling through to the
        % try_me_else clause chain is correct, and emitting a real
        % instruction keeps every later label PC aligned (a dropped one
        % would shift them — the NoOp lesson from the Rust backend).
        format(atom(L0), '    state->code[~w] = (Instruction){ .tag = INSTR_NOOP };', [PCExpr]),
        CodeLines = [L0]
    ;   (   wam_line_to_c_instr(Parts, LabelMap, Arity, OffsetVar, CInstr)
        ->  true
        ;   wam_line_to_c_instr(Parts, CInstr_NoMap)
        ->  CInstr = CInstr_NoMap
        ;   wam_line_to_c_instr(Parts, LabelMap, CInstr_NoArity)
        ->  CInstr = CInstr_NoArity
        ),
        format(atom(L0), '    state->code[~w] = (Instruction)~w;', [PCExpr, CInstr]),
        CodeLines = [L0]
    ).

generate_hash_table_entries(_, _, _, [], _, _, _, []).
generate_hash_table_entries(PC, PCExpr, TableName, [Entry|Rest], Idx, LabelMap, OffsetVar, [Line|RestLines]) :-
    split_string(Entry, ":", "", Parts),
    (   Parts = [KeyStr, LabelStr]
    ->  (   LabelStr == "default"
        ->  TargetPC0 is PC + 1,
            c_pc_expr(OffsetVar, TargetPC0, TargetPC)
        ;   member(LabelStr-TargetPC0, LabelMap)
        ->  c_pc_expr(OffsetVar, TargetPC0, TargetPC)
        ;   TargetPC = -1
        ),
        (   number_string(KeyNum, KeyStr), integer(KeyNum)
        ->  c_value_literal(KeyNum, ValLit)
        ;   atom_string(KeyAtom, KeyStr),
            c_value_literal(KeyAtom, ValLit)
        ),
        format(atom(Line), '    state->code[~w].~w[~w] = (HashEntry){ ~w, ~w };', [PCExpr, TableName, Idx, ValLit, TargetPC]),
        NextIdx is Idx + 1,
        generate_hash_table_entries(PC, PCExpr, TableName, Rest, NextIdx, LabelMap, OffsetVar, RestLines)
    ;   throw(error(wam_c_target_error(invalid_switch_entry(Entry)), _))
    ).

c_pc_expr(OffsetVar, PC, Expr) :-
    (   PC < 0
    ->  Expr = '-1'
    ;   format(atom(Expr), '~w + ~w', [OffsetVar, PC])
    ).

compile_wam_predicate_to_c(PredIndicator, WamCode, _Options, CCode) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    atom_string(Pred, PredStr),
    atom_string(WamCode, WamStr),
    % Note: WamCode is a string generated by wam_target:compile_predicate_to_wam/3
    % (e.g. "get_constant a, A1\ncall foo/2, 2\n"), NOT a list of terms.
    % We parse it line-by-line into structural C literals.
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_c_pass1(Lines, 0, LabelMap),
    wam_lines_to_c_pass2(Lines, 0, LabelMap, Arity, base_pc, CodeSize, InstrParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    
    format(atom(CCode), 
'/* WAM-compiled predicate: ~w/~w */
void setup_~w_~w(WamState* state) {
    int base_pc = state->code_size;
    int new_code_size = base_pc + ~w;
    state->code_size = new_code_size;
    state->code = realloc(state->code, sizeof(Instruction) * state->code_size);
~w
}', [PredStr, Arity, PredStr, Arity, CodeSize, InstrLiterals]).

% ============================================================================
% PHASE 3: step_wam/3 -> C switch statement
% ============================================================================

compile_step_wam_to_c(_Options, CCode) :-
    CCode =
'    bool step_wam(WamState* state, Instruction* instr) {
        switch (instr->tag) {
            case INSTR_GET_CONSTANT: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->as.constant.reg, instr->as.constant.is_y_reg));
                if (val_is_unbound(*cell)) {
                    trail_binding(state, cell);
                    *cell = instr->as.constant.val;
                    state->P++;
                    return true;
                } else if (val_equal(*cell, instr->as.constant.val)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_GET_VARIABLE: {
                // Per WAM spec: copy A[Ai] to X[Xn] without trailing.
                // Trailing is only for mutations of already-bound cells.
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
                *cell_xn = *cell_ai;
                state->P++;
                return true;
            }
            case INSTR_GET_VALUE: {
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
                if (!wam_unify(state, cell_xn, cell_ai)) return false;
                state->P++;
                return true;
            }
            case INSTR_PUT_CONSTANT: {
                WamValue *cell = resolve_reg(state, instr->as.constant.reg, instr->as.constant.is_y_reg);
                *cell = instr->as.constant.val;
                state->P++;
                return true;
            }
            case INSTR_PUT_VARIABLE: {
                WamValue ref = wam_make_ref(state);
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
                *cell_xn = ref;
                *cell_ai = ref;
                state->P++;
                return true;
            }
            case INSTR_PUT_VALUE: {
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
                *cell_ai = *cell_xn;
                state->P++;
                return true;
            }
            case INSTR_ALLOCATE: {
                int new_e_idx = state->E + 1;
                if (new_e_idx >= state->E_cap) {
                    state->E_cap = state->E_cap ? state->E_cap * 2 : WAM_INITIAL_CAP;
                    state->E_array = realloc(state->E_array, sizeof(EnvFrame) * state->E_cap);
                }
                state->E_array[new_e_idx].cp = state->CP;
                state->E_array[new_e_idx].saved_e = state->E;
                state->E = new_e_idx;
                state->P++;
                return true;
            }
            case INSTR_DEALLOCATE: {
                if (state->E >= 0) {
                    state->CP = state->E_array[state->E].cp;
                    state->E = state->E_array[state->E].saved_e;
                }
                state->P++;
                return true;
            }
            case INSTR_NOOP: {
                /* No-op: advance past an instruction the C backend does not
                   translate (chiefly first-argument indexing hints such as
                   switch_on_term_a2). Indexing is only an optimisation, so
                   falling through to the try_me_else clause chain is correct;
                   emitting a real instruction keeps every later label PC
                   aligned (a dropped instruction would shift them). */
                state->P++;
                return true;
            }
            case INSTR_PROCEED: {
                int continuation = state->CP;
                if (continuation == WAM_AGGREGATE_META_COLLECT) {
                    return wam_collect_meta_aggregate_success(state);
                }
                if (continuation == WAM_META_CONJ_RETURN) {
                    return wam_continue_conjunction(state);
                }
                if (continuation == WAM_META_ITE_THEN) {
                    return wam_continue_if_then_else(state);
                }
                if (continuation != WAM_HALT && state->call_base_top > 0) {
                    int barrier_index = --state->call_base_top;
                    int target_b = state->call_bases[barrier_index];
                    if (!state->call_base_preserve_choice[barrier_index]) {
                        wam_prune_choice_points(state, target_b);
                    }
                }
                state->P = continuation;
                return true;
            }
            case INSTR_CALL: {
                if (strcmp(instr->as.pred.pred, "findall/3") == 0) {
                    return wam_dispatch_aggregate_meta(state, "collect", state->P + 1);
                }
                if (strcmp(instr->as.pred.pred, "bagof/3") == 0 ||
                    strcmp(instr->as.pred.pred, "setof/3") == 0) {
                    const char *kind =
                        strcmp(instr->as.pred.pred, "bagof/3") == 0 ? "bagof" : "setof";
                    return wam_dispatch_aggregate_meta(state, kind, state->P + 1);
                }
                if (state->call_base_top >= WAM_CALL_STACK_SIZE) return false;
                state->call_bases[state->call_base_top] = state->B;
                state->call_base_preserve_choice[state->call_base_top] =
                    state->aggregate_top > 0;
                state->call_base_top++;
                state->CP = state->P + 1;
                int target = resolve_predicate_hash(state, instr->as.pred.pred);
                if (target >= 0) { state->P = target; return true; }
                state->call_base_top--;
                return false;
            }
            case INSTR_EXECUTE: {
                if (strcmp(instr->as.pred.pred, "findall/3") == 0) {
                    return wam_dispatch_aggregate_meta(state, "collect", state->CP);
                }
                if (strcmp(instr->as.pred.pred, "bagof/3") == 0 ||
                    strcmp(instr->as.pred.pred, "setof/3") == 0) {
                    const char *kind =
                        strcmp(instr->as.pred.pred, "bagof/3") == 0 ? "bagof" : "setof";
                    return wam_dispatch_aggregate_meta(state, kind, state->CP);
                }
                int target = resolve_predicate_hash(state, instr->as.pred.pred);
                if (target >= 0) { state->P = target; return true; }
                return false;
            }
            case INSTR_BUILTIN_CALL: {
                if (wam_execute_builtin(state, instr->as.pred.pred, instr->as.pred.arity)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_CALL_FOREIGN: {
                if (wam_execute_foreign_predicate(state, instr->as.pred.pred, instr->as.pred.arity)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_BEGIN_AGGREGATE: {
                return wam_begin_aggregate(state, instr);
            }
            case INSTR_END_AGGREGATE: {
                return wam_end_aggregate(state, instr);
            }
            case INSTR_TRY_ME_ELSE: {
                int target = instr->as.choice.target_pc;
                int arity = instr->as.choice.arity ? instr->as.choice.arity : 32;
                push_choice_point(state, target, arity);
                state->P++;
                return true;
            }
            case INSTR_RETRY_ME_ELSE: {
                int target = instr->as.choice.target_pc;
                if (state->B > 0) {
                    ChoicePoint *cp = &state->B_array[state->B - 1];
                    cp->next_pc = target;
                }
                state->P++;
                return true;
            }
            case INSTR_TRUST_ME: {
                pop_choice_point(state);
                state->P++;
                return true;
            }
            case INSTR_GET_LEVEL: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                *cell = val_int(state->B);
                state->P++;
                return true;
            }
            case INSTR_CUT: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg));
                if (cell->tag != VAL_INT) return false;
                int target_b = cell->data.integer;
                if (target_b < 0 || target_b > state->B) return false;
                wam_prune_choice_points(state, target_b);
                state->P++;
                return true;
            }
            case INSTR_CUT_ITE: {
                if (state->B <= 0) return false;
                pop_choice_point(state);
                state->P++;
                return true;
            }
            case INSTR_JUMP: {
                int target = instr->as.jump.target_pc;
                if (target < 0) return false;
                state->P = target;
                return true;
            }
            case INSTR_SWITCH_ON_CONSTANT: {
                WamValue *cell = wam_deref_ptr(state, &state->A[instr->as.switch_index.reg]);
                if (val_is_unbound(*cell)) {
                    state->P++;
                    return true; // Unbound variable falls through to the sequential try_me_else chain
                }
                if (cell->tag != VAL_ATOM && cell->tag != VAL_INT) {
                    if (instr->as.switch_index.no_match_fallthrough) {
                        state->P++;
                        return true;
                    }
                    return false; // Type mismatch, fail
                }
                for (int i = 0; i < instr->as.switch_index.hash_size; i++) {
                    if (val_equal(*cell, instr->as.switch_index.hash_table[i].key)) {
                        state->P = instr->as.switch_index.hash_table[i].target_pc;
                        return true;
                    }
                }
                if (instr->as.switch_index.no_match_fallthrough) {
                    state->P++;
                    return true;
                }
                return false; // Not found in index, fail
            }
            case INSTR_SWITCH_ON_STRUCTURE: {
                WamValue *cell = wam_deref_ptr(state, &state->A[0]);
                if (val_is_unbound(*cell)) {
                    state->P++;
                    return true; // Unbound variable falls through to the sequential try_me_else chain
                }
                if (cell->tag != VAL_STR) {
                    return false; // Type mismatch, fail
                }
                WamValue *f = &state->H_array[cell->data.ref_addr];
                for (int i = 0; i < instr->as.switch_index.hash_size; i++) {
                    if (val_equal(*f, instr->as.switch_index.hash_table[i].key)) {
                        state->P = instr->as.switch_index.hash_table[i].target_pc;
                        return true;
                    }
                }
                return false; // Not found in index, fail
            }
            case INSTR_SWITCH_ON_TERM: {
                WamValue *cell = wam_deref_ptr(state, &state->A[0]);
                if (val_is_unbound(*cell)) {
                    state->P++;
                    return true; // Unbound variable falls through to the sequential try_me_else chain
                }
                if (cell->tag == VAL_ATOM || cell->tag == VAL_INT) {
                    for (int i = 0; i < instr->as.switch_index.hash_size; i++) {
                        if (val_equal(*cell, instr->as.switch_index.hash_table[i].key)) {
                            state->P = instr->as.switch_index.hash_table[i].target_pc;
                            return true;
                        }
                    }
                } else if (cell->tag == VAL_STR) {
                    WamValue *f = &state->H_array[cell->data.ref_addr];
                    /* A "[|]/2"/"./2" structure is a list cell (nested cons
                       cells are built with put_structure), so route it to the
                       list clause exactly like a VAL_LIST — otherwise the
                       recursion tail (a cons STR) misses the list clause. */
                    if (f->tag == VAL_ATOM &&
                        (strcmp(f->data.atom, "[|]/2") == 0 || strcmp(f->data.atom, "./2") == 0)) {
                        if (instr->as.switch_index.list_target_pc >= 0) {
                            state->P = instr->as.switch_index.list_target_pc;
                        } else {
                            state->P++;
                        }
                        return true;
                    }
                    for (int i = 0; i < instr->as.switch_index.s_hash_size; i++) {
                        if (val_equal(*f, instr->as.switch_index.s_hash_table[i].key)) {
                            state->P = instr->as.switch_index.s_hash_table[i].target_pc;
                            return true;
                        }
                    }
                } else if (cell->tag == VAL_LIST) {
                    if (instr->as.switch_index.list_target_pc >= 0) {
                        state->P = instr->as.switch_index.list_target_pc;
                    } else {
                        state->P++;
                    }
                    return true;
                }
                return false; // Not found in either index — fail and backtrack
            }
            case INSTR_GET_STRUCTURE: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->as.functor.reg, instr->as.functor.is_y_reg));
                if (cell->tag == VAL_UNBOUND) {
                    trail_binding(state, cell);
                    WamValue s; s.tag = VAL_STR; s.data.ref_addr = state->H;
                    *cell = s;
                    
                    // Note: instr->as.functor.pred includes the arity suffix (e.g. "foo/2"), which is stored as the functor atom
                    const char *slash = strchr(instr->as.functor.pred, ''/'');
                    assert(slash != NULL && "Functor missing arity suffix");
                    int arity = strtol(slash + 1, NULL, 10);
                    
                    // Invariant contract: We proactively pre-reserve capacity for the functor + all arity arguments.
                    // Subsequent UNIFY_* instructions in write mode will push values sequentially via state->H++.
                    // While UNIFY_* instructions have their own single-slot capacity guards, this pre-allocation 
                    // ensures contiguous allocation and avoids multiple reallocs during the structure building sequence.
                    int required = state->H + 1 + arity;
                    if (required >= state->H_cap) {
                        if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                        while (required >= state->H_cap) state->H_cap *= 2;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->H_array[state->H] = val_atom(instr->as.functor.pred);
                    state->H++;
                    state->mode = MODE_WRITE;
                } else if (cell->tag == VAL_STR) {
                    WamValue *f = &state->H_array[cell->data.ref_addr];
                    if (f->tag == VAL_ATOM && strcmp(f->data.atom, instr->as.functor.pred) == 0) {
                        state->S = cell->data.ref_addr + 1;
                        state->mode = MODE_READ;
                    } else { return false; }
                } else { return false; }
                state->P++;
                return true;
            }
            case INSTR_PUT_STRUCTURE: {
                WamValue s; s.tag = VAL_STR; s.data.ref_addr = state->H;
                WamValue *cell = resolve_reg(state, instr->as.functor.reg, instr->as.functor.is_y_reg);
                /* If this register dereferences to an unbound placeholder heap
                   cell (the embedded argument of an enclosing structure that a
                   prior set_variable wrote, e.g. nested arithmetic
                   +(+(A,B),C)), bind that cell to the new structure as well —
                   otherwise the enclosing structure keeps pointing at the
                   still-unbound placeholder and eval/unify never see this
                   subterm.

                   A-REGISTER EXCEPTION (M139/M140 bind-through class): A
                   registers (is_y_reg == 0; X == 2, Y == 1) are argument
                   STAGING — their old occupant is an unrelated variable
                   (often a clause-head argument), and writing the new
                   structure into its heap cell creates a cyclic term
                   (X = f(X)) — and did so UNTRAILED, a backtracking
                   corruption hazard on top of the wrong-fail. set_variable
                   placeholders only ever live in X/Y registers, so the
                   bind-through is conditioned on the register class — the
                   same fix the Rust, Go, Scala, Kotlin, Haskell and LLVM
                   targets carry. */
                if (instr->as.functor.is_y_reg != 0) {
                    WamValue *placeholder = wam_deref_ptr(state, cell);
                    if (placeholder != cell && placeholder->tag == VAL_UNBOUND) {
                        *placeholder = s;
                    }
                }
                *cell = s;

                const char *slash = strchr(instr->as.functor.pred, ''/'');
                assert(slash != NULL && "Functor missing arity suffix");
                int arity = strtol(slash + 1, NULL, 10);
                
                // Invariant contract: Proactively pre-reserve capacity for functor + arguments.
                // UNIFY_* instructions will sequentially append to H.
                int required = state->H + 1 + arity;
                if (required >= state->H_cap) {
                    if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                    while (required >= state->H_cap) state->H_cap *= 2;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->H_array[state->H] = val_atom(instr->as.functor.pred);
                state->H++;
                state->mode = MODE_WRITE;
                state->P++;
                return true;
            }
            case INSTR_GET_LIST: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg));
                if (cell->tag == VAL_UNBOUND) {
                    trail_binding(state, cell);
                    WamValue l; l.tag = VAL_LIST; l.data.ref_addr = state->H;
                    *cell = l;
                    
                    // Invariant contract: Proactively pre-reserve capacity for [head|tail].
                    // UNIFY_* instructions will sequentially append to H.
                    int required = state->H + 2;
                    if (required >= state->H_cap) {
                        if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                        while (required >= state->H_cap) state->H_cap *= 2;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->mode = MODE_WRITE;
                } else if (cell->tag == VAL_LIST) {
                    state->S = cell->data.ref_addr;
                    state->mode = MODE_READ;
                } else if (cell->tag == VAL_STR) {
                    /* Cons-cell aliasing: the compiler emits the outer list
                       with put_list (VAL_LIST) but nested cons cells with
                       put_structure "[|]/2" (VAL_STR). Treat a "[|]/2"/"./2"
                       structure as a list cell: its head/tail are the two args
                       after the functor cell. */
                    WamValue *f = &state->H_array[cell->data.ref_addr];
                    if (f->tag == VAL_ATOM &&
                        (strcmp(f->data.atom, "[|]/2") == 0 ||
                         strcmp(f->data.atom, "./2") == 0)) {
                        state->S = cell->data.ref_addr + 1;
                        state->mode = MODE_READ;
                    } else { return false; }
                } else { return false; }
                state->P++;
                return true;
            }
            case INSTR_PUT_LIST: {
                WamValue l; l.tag = VAL_LIST; l.data.ref_addr = state->H;
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                *cell = l;
                
                // Invariant contract: Proactively pre-reserve capacity for [head|tail].
                // UNIFY_* instructions will sequentially append to H.
                int required = state->H + 2;
                if (required >= state->H_cap) {
                    if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                    while (required >= state->H_cap) state->H_cap *= 2;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->mode = MODE_WRITE;
                state->P++;
                return true;
            }
            case INSTR_SET_VARIABLE: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                WamValue ref = wam_make_ref(state);
                *cell = ref;
                state->P++;
                return true;
            }
            case INSTR_SET_VALUE: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                if (state->H >= state->H_cap) {
                    state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->H_array[state->H] = *cell;
                state->H++;
                state->P++;
                return true;
            }
            case INSTR_SET_CONSTANT: {
                if (state->H >= state->H_cap) {
                    state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->H_array[state->H] = instr->as.constant.val;
                state->H++;
                state->P++;
                return true;
            }
            case INSTR_UNIFY_VARIABLE: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                if (state->mode == MODE_READ) {
                    *cell = state->H_array[state->S];
                    state->S++;
                } else {
                    // Note: wam_make_ref allocates an unbound cell in H_array and increments H,
                    // satisfying the 1-slot heap pre-reservation invariant.
                    WamValue ref = wam_make_ref(state);
                    *cell = ref;
                }
                state->P++;
                return true;
            }
            case INSTR_UNIFY_VALUE: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                if (state->mode == MODE_READ) {
                    if (!wam_unify(state, cell, &state->H_array[state->S])) return false;
                    state->S++;
                } else {
                    if (state->H >= state->H_cap) {
                        state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->H_array[state->H] = *cell;
                    state->H++;
                }
                state->P++;
                return true;
            }
            case INSTR_UNIFY_CONSTANT: {
                if (state->mode == MODE_READ) {
                    WamValue *cell = wam_deref_ptr(state, &state->H_array[state->S]);
                    if (cell->tag == VAL_UNBOUND) {
                        trail_binding(state, cell);
                        *cell = instr->as.constant.val;
                    } else if (!val_equal(*cell, instr->as.constant.val)) {
                        return false;
                    }
                    state->S++;
                } else {
                    if (state->H >= state->H_cap) {
                        state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->H_array[state->H] = instr->as.constant.val;
                    state->H++;
                }
                state->P++;
                return true;
            }
            default: return false;
        }
    }

    int wam_run(WamState* state) {
        // Outer backtracking loop
        while (state->P >= 0 && state->P < state->code_size) {
            Instruction* instr = &state->code[state->P];
            if (!step_wam(state, instr)) {
                bool recovered = false;
                while (state->B > 0 && !recovered) {
                    ChoicePoint* cp = &state->B_array[state->B - 1];
                    int next_pc = cp->next_pc;
                    restore_choice_point(state, cp); // Restores H, E, CP, A, unwinds TR
                    if (next_pc == WAM_AGGREGATE_NEXT_GROUP) {
                        pop_choice_point(state);
                        recovered = wam_bind_next_aggregate_group(state);
                    } else if (next_pc == WAM_AGGREGATE_META_DONE) {
                        recovered = wam_finalize_meta_aggregate(state);
                    } else if (next_pc == WAM_META_DISJ_RIGHT) {
                        recovered = wam_resume_disjunction(state);
                    } else if (next_pc == WAM_META_ITE_ELSE) {
                        recovered = wam_resume_if_then_else(state);
                    } else if (next_pc == WAM_FOREIGN_STREAM_NEXT) {
                        recovered = wam_resume_foreign_stream(state);
                    } else {
                        state->P = next_pc; // Explicitly jump to alternative
                        recovered = true;
                    }
                }
                if (!recovered) {
                    return WAM_HALT; // Failure, no choice points left
                }
            }
        }
        return (state->P == WAM_HALT) ? 0 : WAM_ERR_OOB; // 0 on success (HALT), else OOB error
    }'.

compile_wam_helpers_to_c(_Options, CCode) :-
    CCode =
'#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "wam_runtime.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <unistd.h>

static void wam_bidirectional_distance_cache_clear(WamState *state);
static bool wam_dispatch_aggregate_meta(WamState *state, const char *kind,
                                        int return_pc);
static bool wam_invoke_goal_as_call(WamState *state, WamValue goal,
                                    int return_pc);
static bool wam_collect_meta_aggregate_success(WamState *state);
static bool wam_finalize_meta_aggregate(WamState *state);
static bool wam_continue_conjunction(WamState *state);
static bool wam_resume_disjunction(WamState *state);
static bool wam_continue_if_then_else(WamState *state);
static bool wam_resume_if_then_else(WamState *state);

static bool wam_ensure_heap_slots(WamState *state, int additional) {
    if (additional <= 0) return true;
    if (state->H > INT_MAX - additional) return false;
    int required = state->H + additional;
    if (required <= state->H_cap) return true;
    if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
    while (required > state->H_cap) {
        if (state->H_cap > INT_MAX / 2) {
            state->H_cap = required;
            break;
        }
        state->H_cap *= 2;
    }
    WamValue *heap = realloc(state->H_array, sizeof(WamValue) * (size_t)state->H_cap);
    if (!heap) return false;
    state->H_array = heap;
    return true;
}

static void wam_stored_term_free(WamStoredTerm *term) {
    free(term->cells);
    memset(term, 0, sizeof(WamStoredTerm));
}

static bool wam_stored_term_reserve(WamStoredTerm *term, int additional) {
    if (additional <= 0) return true;
    if (term->cell_count > INT_MAX - additional) return false;
    int required = term->cell_count + additional;
    if (required <= term->cell_cap) return true;
    int new_cap = term->cell_cap ? term->cell_cap : WAM_INITIAL_CAP;
    while (required > new_cap) {
        if (new_cap > INT_MAX / 2) {
            new_cap = required;
            break;
        }
        new_cap *= 2;
    }
    WamValue *cells = realloc(term->cells, sizeof(WamValue) * (size_t)new_cap);
    if (!cells) return false;
    term->cells = cells;
    term->cell_cap = new_cap;
    return true;
}

static bool wam_parse_functor_arity(const char *functor, int *arity_out) {
    const char *slash = strrchr(functor, 47);
    if (!slash) return false;
    char *end = NULL;
    long arity = strtol(slash + 1, &end, 10);
    if (!end || *end != 0 || arity < 0 || arity > 255) return false;
    *arity_out = (int)arity;
    return true;
}

static bool wam_ref_addr(WamState *state, WamValue value, int *addr_out) {
    if (value.tag != VAL_REF) return false;
    int addr = value.data.ref_addr;
    for (int steps = 0; steps < state->H_cap; steps++) {
        if (addr < 0 || addr >= state->H) return false;
        WamValue cell = state->H_array[addr];
        if (cell.tag == VAL_UNBOUND) {
            *addr_out = addr;
            return true;
        }
        if (cell.tag != VAL_REF) return false;
        if (cell.data.ref_addr == addr) {
            *addr_out = addr;
            return true;
        }
        addr = cell.data.ref_addr;
    }
    return false;
}

static bool wam_ref_array_contains(WamState *state, WamValue *values,
                                   int count, WamValue value) {
    int addr = -1;
    if (!wam_ref_addr(state, value, &addr)) return false;
    for (int i = 0; i < count; i++) {
        int other = -1;
        if (wam_ref_addr(state, values[i], &other) && other == addr) {
            return true;
        }
    }
    return false;
}

static bool wam_ref_array_append_unique_limit(WamState *state, WamValue *values,
                                              int *count, int max_count,
                                              WamValue value) {
    if (wam_ref_array_contains(state, values, *count, value)) return true;
    if (*count >= max_count) return false;
    values[*count] = value;
    (*count)++;
    return true;
}

static bool wam_ref_array_append_unique(WamState *state, WamValue *values,
                                        int *count, WamValue value) {
    return wam_ref_array_append_unique_limit(state, values, count,
                                             WAM_AGGREGATE_MAX_WITNESSES,
                                             value);
}

static bool wam_collect_term_vars(WamState *state, WamValue value,
                                  WamValue *vars, int *count,
                                  int max_count) {
    if (value.tag == VAL_REF) {
        int addr = -1;
        if (wam_ref_addr(state, value, &addr)) {
            WamValue ref;
            ref.tag = VAL_REF;
            ref.data.ref_addr = addr;
            return wam_ref_array_append_unique_limit(state, vars, count,
                                                     max_count, ref);
        }
    }
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag == VAL_LIST) {
        int base = cell->data.ref_addr;
        if (base < 0 || base + 1 >= state->H) return false;
        return wam_collect_term_vars(state, state->H_array[base], vars,
                                     count, max_count) &&
               wam_collect_term_vars(state, state->H_array[base + 1], vars,
                                     count, max_count);
    }
    if (cell->tag == VAL_STR) {
        int base = cell->data.ref_addr;
        if (base < 0 || base >= state->H) return false;
        WamValue *functor = &state->H_array[base];
        if (functor->tag != VAL_ATOM) return false;
        int arity = 0;
        if (!wam_parse_functor_arity(functor->data.atom, &arity)) return false;
        if (base + arity >= state->H) return false;
        for (int i = 0; i < arity; i++) {
            if (!wam_collect_term_vars(state, state->H_array[base + 1 + i],
                                       vars, count, max_count)) return false;
        }
    }
    return true;
}

static bool wam_collect_goal_witnesses(WamState *state, WamValue goal,
                                       WamValue *exclude, int *exclude_count,
                                       int exclude_max,
                                       WamValue *witnesses,
                                       int *witness_count) {
    if (goal.tag == VAL_REF) {
        int addr = -1;
        if (wam_ref_addr(state, goal, &addr)) {
            WamValue ref;
            ref.tag = VAL_REF;
            ref.data.ref_addr = addr;
            if (!wam_ref_array_contains(state, exclude, *exclude_count, ref)) {
                if (!wam_ref_array_append_unique(state, witnesses,
                                                 witness_count, ref)) return false;
                return wam_ref_array_append_unique_limit(state, exclude,
                                                         exclude_count,
                                                         exclude_max, ref);
            }
            return true;
        }
    }
    WamValue *cell = wam_deref_ptr(state, &goal);
    if (cell->tag == VAL_LIST) {
        int base = cell->data.ref_addr;
        if (base < 0 || base + 1 >= state->H) return false;
        return wam_collect_goal_witnesses(state, state->H_array[base],
                                          exclude, exclude_count, exclude_max,
                                          witnesses, witness_count) &&
               wam_collect_goal_witnesses(state, state->H_array[base + 1],
                                          exclude, exclude_count, exclude_max,
                                          witnesses, witness_count);
    }
    if (cell->tag != VAL_STR) return true;
    int base = cell->data.ref_addr;
    if (base < 0 || base >= state->H) return false;
    WamValue *functor = &state->H_array[base];
    if (functor->tag != VAL_ATOM) return false;
    int arity = 0;
    if (!wam_parse_functor_arity(functor->data.atom, &arity)) return false;
    if (base + arity >= state->H) return false;
    if (strcmp(functor->data.atom, "^/2") == 0 && arity == 2) {
        if (!wam_collect_term_vars(state, state->H_array[base + 1],
                                   exclude, exclude_count,
                                   exclude_max)) return false;
        return wam_collect_goal_witnesses(state, state->H_array[base + 2],
                                          exclude, exclude_count, exclude_max,
                                          witnesses, witness_count);
    }
    for (int i = 0; i < arity; i++) {
        if (!wam_collect_goal_witnesses(state, state->H_array[base + 1 + i],
                                        exclude, exclude_count, exclude_max,
                                        witnesses, witness_count)) return false;
    }
    return true;
}

static WamValue *wam_aggregate_result_cell(WamState *state,
                                           WamAggregateFrame *frame) {
    if (frame->is_meta) return &frame->meta_result;
    return resolve_reg(state, frame->result_reg, frame->result_is_y);
}

static WamValue *wam_aggregate_witness_cell(WamState *state,
                                            WamAggregateFrame *frame,
                                            int index) {
    if (index < 0 || index >= frame->witness_count) return NULL;
    if (frame->is_meta) return &frame->meta_witnesses[index];
    return resolve_reg(state, frame->witness_regs[index],
                       frame->witness_is_y[index]);
}

static bool wam_goal_structure(WamState *state, WamValue goal,
                               const char **functor_out, int *base_out) {
    WamValue current = goal;
    for (int strip = 0; strip < 8; strip++) {
        WamValue *cell = wam_deref_ptr(state, &current);
        if (cell->tag != VAL_STR) return false;
        int base = cell->data.ref_addr;
        if (base < 0 || base >= state->H) return false;
        WamValue *functor = &state->H_array[base];
        if (functor->tag != VAL_ATOM) return false;
        int arity = 0;
        if (!wam_parse_functor_arity(functor->data.atom, &arity)) return false;
        if (base + arity >= state->H) return false;
        if (strcmp(functor->data.atom, ":/2") == 0 && arity == 2) {
            current = state->H_array[base + 2];
            continue;
        }
        *functor_out = functor->data.atom;
        *base_out = base;
        return true;
    }
    return false;
}

static bool wam_complete_goal_success(WamState *state, int return_pc) {
    if (return_pc == WAM_AGGREGATE_META_COLLECT) {
        return wam_collect_meta_aggregate_success(state);
    }
    if (return_pc == WAM_META_CONJ_RETURN) {
        return wam_continue_conjunction(state);
    }
    if (return_pc == WAM_META_ITE_THEN) {
        return wam_continue_if_then_else(state);
    }
    state->P = return_pc;
    return true;
}

static bool wam_dispatch_if_then_else(WamState *state,
                                      WamValue if_goal,
                                      WamValue then_goal,
                                      WamValue else_goal,
                                      int return_pc) {
    if (state->ite_top >= WAM_META_GOAL_STACK_SIZE) return false;
    int base_b = state->B;
    WamIteFrame *frame = &state->ite_frames[state->ite_top++];
    frame->then_goal = then_goal;
    frame->else_goal = else_goal;
    frame->return_pc = return_pc;
    frame->base_b = base_b;
    push_choice_point(state, WAM_META_ITE_ELSE, 32);
    return wam_invoke_goal_as_call(state, if_goal, WAM_META_ITE_THEN);
}

static bool wam_functor_with_arity(WamState *state,
                                   const char *functor,
                                   int arity,
                                   char *out,
                                   size_t out_size) {
    const char *slash = strrchr(functor, 47);
    if (!slash) return false;
    size_t name_len = (size_t)(slash - functor);
    if (name_len == 0 || name_len >= out_size) return false;
    int written = snprintf(out, out_size, "%.*s/%d",
                           (int)name_len, functor, arity);
    if (written <= 0 || written >= (int)out_size) return false;
    (void)state;
    return true;
}

static bool wam_dispatch_call_n(WamState *state,
                                int base,
                                int arity,
                                int return_pc) {
    if (arity < 1) return false;
    WamValue callable = state->H_array[base + 1];
    if (arity == 1) {
        return wam_invoke_goal_as_call(state, callable, return_pc);
    }

    int extra_count = arity - 1;
    WamValue args[WAM_MAX_REGS];
    int goal_arity = 0;
    char functor_buf[256];

    WamValue *callable_cell = wam_deref_ptr(state, &callable);
    if (callable_cell->tag == VAL_ATOM) {
        goal_arity = extra_count;
        int written = snprintf(functor_buf, sizeof(functor_buf), "%s/%d",
                               callable_cell->data.atom, goal_arity);
        if (written <= 0 || written >= (int)sizeof(functor_buf)) return false;
    } else {
        const char *callable_functor = NULL;
        int callable_base = -1;
        if (!wam_goal_structure(state, callable, &callable_functor,
                                &callable_base)) return false;
        int callable_arity = 0;
        if (!wam_parse_functor_arity(callable_functor,
                                     &callable_arity)) return false;
        if (callable_arity > WAM_MAX_REGS - extra_count) return false;
        goal_arity = callable_arity + extra_count;
        if (!wam_functor_with_arity(state, callable_functor, goal_arity,
                                    functor_buf, sizeof(functor_buf))) {
            return false;
        }
        for (int i = 0; i < callable_arity; i++) {
            args[i] = state->H_array[callable_base + 1 + i];
        }
    }

    if (goal_arity > WAM_MAX_REGS) return false;
    int existing_args = goal_arity - extra_count;
    for (int i = 0; i < extra_count; i++) {
        args[existing_args + i] = state->H_array[base + 2 + i];
    }

    if (!wam_ensure_heap_slots(state, goal_arity + 1)) return false;
    int goal_base = state->H;
    state->H_array[state->H++] = val_atom(wam_intern_atom(state, functor_buf));
    for (int i = 0; i < goal_arity; i++) {
        state->H_array[state->H++] = args[i];
    }
    WamValue expanded_goal;
    expanded_goal.tag = VAL_STR;
    expanded_goal.data.ref_addr = goal_base;
    return wam_invoke_goal_as_call(state, expanded_goal, return_pc);
}

static bool wam_invoke_goal_as_call(WamState *state, WamValue goal,
                                    int return_pc) {
    WamValue *cell = wam_deref_ptr(state, &goal);
    if (cell->tag == VAL_ATOM) {
        if (strcmp(cell->data.atom, "true") == 0) {
            return wam_complete_goal_success(state, return_pc);
        }
        if (strcmp(cell->data.atom, "fail") == 0 ||
            strcmp(cell->data.atom, "false") == 0) {
            return false;
        }
        char key[256];
        int written = snprintf(key, sizeof(key), "%s/0", cell->data.atom);
        if (written <= 0 || written >= (int)sizeof(key)) return false;
        int target = resolve_predicate_hash(state, key);
        if (target < 0) return false;
        state->CP = return_pc;
        state->P = target;
        return true;
    }
    const char *functor = NULL;
    int base = -1;
    if (!wam_goal_structure(state, goal, &functor, &base)) return false;
    int arity = 0;
    if (!wam_parse_functor_arity(functor, &arity)) return false;
    if (arity > WAM_MAX_REGS) return false;
    if (strncmp(functor, "call/", 5) == 0 && arity >= 1) {
        return wam_dispatch_call_n(state, base, arity, return_pc);
    }
    if (strcmp(functor, ",/2") == 0 && arity == 2) {
        if (state->conj_top >= WAM_META_GOAL_STACK_SIZE) return false;
        WamConjFrame *frame = &state->conj_frames[state->conj_top++];
        frame->second_goal = state->H_array[base + 2];
        frame->return_pc = return_pc;
        return wam_invoke_goal_as_call(state, state->H_array[base + 1],
                                       WAM_META_CONJ_RETURN);
    }
    if (strcmp(functor, ";/2") == 0 && arity == 2) {
        const char *left_functor = NULL;
        int left_base = -1;
        if (wam_goal_structure(state, state->H_array[base + 1],
                               &left_functor, &left_base) &&
            strcmp(left_functor, "->/2") == 0) {
            return wam_dispatch_if_then_else(state,
                                             state->H_array[left_base + 1],
                                             state->H_array[left_base + 2],
                                             state->H_array[base + 2],
                                             return_pc);
        }
        if (state->disj_top >= WAM_META_GOAL_STACK_SIZE) return false;
        WamDisjFrame *frame = &state->disj_frames[state->disj_top++];
        frame->right_goal = state->H_array[base + 2];
        frame->return_pc = return_pc;
        push_choice_point(state, WAM_META_DISJ_RIGHT, 32);
        return wam_invoke_goal_as_call(state, state->H_array[base + 1],
                                       return_pc);
    }
    if (strcmp(functor, "->/2") == 0 && arity == 2) {
        return wam_dispatch_if_then_else(state,
                                         state->H_array[base + 1],
                                         state->H_array[base + 2],
                                         val_atom("fail"),
                                         return_pc);
    }
    if (strcmp(functor, "^/2") == 0 && arity == 2) {
        return wam_invoke_goal_as_call(state, state->H_array[base + 2],
                                       return_pc);
    }
    for (int i = 0; i < arity; i++) {
        state->A[i] = state->H_array[base + 1 + i];
    }
    if (strcmp(functor, "findall/3") == 0) {
        return wam_dispatch_aggregate_meta(state, "collect", return_pc);
    }
    if (strcmp(functor, "bagof/3") == 0) {
        return wam_dispatch_aggregate_meta(state, "bagof", return_pc);
    }
    if (strcmp(functor, "setof/3") == 0) {
        return wam_dispatch_aggregate_meta(state, "setof", return_pc);
    }
    int target = resolve_predicate_hash(state, functor);
    if (target >= 0) {
        state->CP = return_pc;
        state->P = target;
        return true;
    }
    if (wam_execute_builtin(state, functor, arity)) {
        return wam_complete_goal_success(state, return_pc);
    }
    return false;
}

static bool wam_continue_conjunction(WamState *state) {
    if (state->conj_top <= 0) return false;
    WamConjFrame *frame = &state->conj_frames[state->conj_top - 1];
    return wam_invoke_goal_as_call(state, frame->second_goal,
                                   frame->return_pc);
}

static bool wam_resume_disjunction(WamState *state) {
    if (state->disj_top <= 0 || state->B <= 0) return false;
    WamDisjFrame *frame = &state->disj_frames[state->disj_top - 1];
    WamValue right_goal = frame->right_goal;
    int return_pc = frame->return_pc;
    state->disj_top--;
    pop_choice_point(state);
    return wam_invoke_goal_as_call(state, right_goal, return_pc);
}

static bool wam_continue_if_then_else(WamState *state) {
    if (state->ite_top <= 0) return false;
    WamIteFrame *frame = &state->ite_frames[state->ite_top - 1];
    WamValue then_goal = frame->then_goal;
    int return_pc = frame->return_pc;
    int base_b = frame->base_b;
    state->ite_top--;
    wam_prune_choice_points(state, base_b);
    return wam_invoke_goal_as_call(state, then_goal, return_pc);
}

static bool wam_resume_if_then_else(WamState *state) {
    if (state->ite_top <= 0 || state->B <= 0) return false;
    WamIteFrame *frame = &state->ite_frames[state->ite_top - 1];
    WamValue else_goal = frame->else_goal;
    int return_pc = frame->return_pc;
    state->ite_top--;
    pop_choice_point(state);
    return wam_invoke_goal_as_call(state, else_goal, return_pc);
}

static bool wam_copy_term_to_stored(WamState *state,
                                    WamStoredTerm *term,
                                    WamValue value,
                                    WamValue *out) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag == VAL_ATOM || cell->tag == VAL_INT || cell->tag == VAL_FLOAT) {
        *out = *cell;
        return true;
    }
    if (cell->tag == VAL_UNBOUND || cell->tag == VAL_REF) {
        *out = val_unbound("_");
        return true;
    }
    if (cell->tag == VAL_LIST) {
        if (!wam_stored_term_reserve(term, 2)) return false;
        int base = term->cell_count;
        term->cell_count += 2;
        out->tag = VAL_LIST;
        out->data.ref_addr = base;
        WamValue head;
        WamValue tail;
        if (!wam_copy_term_to_stored(state, term,
                                     state->H_array[cell->data.ref_addr],
                                     &head)) return false;
        if (!wam_copy_term_to_stored(state, term,
                                     state->H_array[cell->data.ref_addr + 1],
                                     &tail)) return false;
        term->cells[base] = head;
        term->cells[base + 1] = tail;
        return true;
    }
    if (cell->tag == VAL_STR) {
        WamValue *functor = &state->H_array[cell->data.ref_addr];
        if (functor->tag != VAL_ATOM) return false;
        int arity = 0;
        if (!wam_parse_functor_arity(functor->data.atom, &arity)) return false;
        if (!wam_stored_term_reserve(term, 1 + arity)) return false;
        int base = term->cell_count;
        term->cell_count += 1 + arity;
        out->tag = VAL_STR;
        out->data.ref_addr = base;
        term->cells[base] = *functor;
        for (int i = 0; i < arity; i++) {
            WamValue arg;
            if (!wam_copy_term_to_stored(state, term,
                                         state->H_array[cell->data.ref_addr + 1 + i],
                                         &arg)) return false;
            term->cells[base + 1 + i] = arg;
        }
        return true;
    }
    return false;
}

static bool wam_materialize_stored_term(WamState *state,
                                        WamStoredTerm *term,
                                        WamValue value,
                                        WamValue *out) {
    if (value.tag == VAL_ATOM) {
        *out = val_atom(wam_intern_atom(state, value.data.atom));
        return true;
    }
    if (value.tag == VAL_INT || value.tag == VAL_FLOAT || value.tag == VAL_UNBOUND) {
        *out = value;
        return true;
    }
    if (value.tag == VAL_LIST) {
        if (value.data.ref_addr < 0 || value.data.ref_addr + 1 >= term->cell_count) return false;
        if (!wam_ensure_heap_slots(state, 2)) return false;
        int base = state->H;
        state->H += 2;
        out->tag = VAL_LIST;
        out->data.ref_addr = base;
        WamValue head;
        WamValue tail;
        if (!wam_materialize_stored_term(state, term,
                                         term->cells[value.data.ref_addr],
                                         &head)) return false;
        if (!wam_materialize_stored_term(state, term,
                                         term->cells[value.data.ref_addr + 1],
                                         &tail)) return false;
        state->H_array[base] = head;
        state->H_array[base + 1] = tail;
        return true;
    }
    if (value.tag == VAL_STR) {
        if (value.data.ref_addr < 0 || value.data.ref_addr >= term->cell_count) return false;
        WamValue *functor = &term->cells[value.data.ref_addr];
        if (functor->tag != VAL_ATOM) return false;
        int arity = 0;
        if (!wam_parse_functor_arity(functor->data.atom, &arity)) return false;
        if (value.data.ref_addr + arity >= term->cell_count) return false;
        if (!wam_ensure_heap_slots(state, 1 + arity)) return false;
        int base = state->H;
        state->H += 1 + arity;
        out->tag = VAL_STR;
        out->data.ref_addr = base;
        state->H_array[base] = val_atom(wam_intern_atom(state, functor->data.atom));
        for (int i = 0; i < arity; i++) {
            WamValue arg;
            if (!wam_materialize_stored_term(state, term,
                                             term->cells[value.data.ref_addr + 1 + i],
                                             &arg)) return false;
            state->H_array[base + 1 + i] = arg;
        }
        return true;
    }
    return false;
}

static int wam_stored_term_tag_rank(WamValueTag tag) {
    switch (tag) {
        case VAL_UNBOUND: return 0;
        case VAL_INT: return 1;
        case VAL_FLOAT: return 2;
        case VAL_ATOM: return 3;
        case VAL_LIST: return 4;
        case VAL_STR: return 5;
        case VAL_REF: return 6;
        default: return 7;
    }
}

static int wam_compare_ints(int left, int right) {
    return (left > right) - (left < right);
}

static int wam_compare_doubles(double left, double right) {
    return (left > right) - (left < right);
}

static int wam_compare_stored_value(const WamStoredTerm *left_term,
                                    WamValue left,
                                    const WamStoredTerm *right_term,
                                    WamValue right) {
    int left_rank = wam_stored_term_tag_rank(left.tag);
    int right_rank = wam_stored_term_tag_rank(right.tag);
    if (left_rank != right_rank) return wam_compare_ints(left_rank, right_rank);

    switch (left.tag) {
        case VAL_UNBOUND:
            return 0;
        case VAL_INT:
            return wam_compare_ints(left.data.integer, right.data.integer);
        case VAL_FLOAT:
            return wam_compare_doubles(left.data.floating, right.data.floating);
        case VAL_ATOM:
            return strcmp(left.data.atom, right.data.atom);
        case VAL_LIST: {
            int left_base = left.data.ref_addr;
            int right_base = right.data.ref_addr;
            if (left_base < 0 || left_base + 1 >= left_term->cell_count ||
                right_base < 0 || right_base + 1 >= right_term->cell_count) {
                return wam_compare_ints(left_base, right_base);
            }
            int head_cmp =
                wam_compare_stored_value(left_term, left_term->cells[left_base],
                                         right_term, right_term->cells[right_base]);
            if (head_cmp != 0) return head_cmp;
            return wam_compare_stored_value(left_term, left_term->cells[left_base + 1],
                                            right_term, right_term->cells[right_base + 1]);
        }
        case VAL_STR: {
            int left_base = left.data.ref_addr;
            int right_base = right.data.ref_addr;
            if (left_base < 0 || left_base >= left_term->cell_count ||
                right_base < 0 || right_base >= right_term->cell_count) {
                return wam_compare_ints(left_base, right_base);
            }
            WamValue left_functor = left_term->cells[left_base];
            WamValue right_functor = right_term->cells[right_base];
            int functor_cmp =
                wam_compare_stored_value(left_term, left_functor,
                                         right_term, right_functor);
            if (functor_cmp != 0) return functor_cmp;
            if (left_functor.tag != VAL_ATOM || right_functor.tag != VAL_ATOM) {
                return 0;
            }
            int left_arity = 0;
            int right_arity = 0;
            if (!wam_parse_functor_arity(left_functor.data.atom, &left_arity) ||
                !wam_parse_functor_arity(right_functor.data.atom, &right_arity)) {
                return 0;
            }
            int arity_cmp = wam_compare_ints(left_arity, right_arity);
            if (arity_cmp != 0) return arity_cmp;
            for (int i = 0; i < left_arity; i++) {
                int arg_cmp =
                    wam_compare_stored_value(left_term, left_term->cells[left_base + 1 + i],
                                             right_term, right_term->cells[right_base + 1 + i]);
                if (arg_cmp != 0) return arg_cmp;
            }
            return 0;
        }
        case VAL_REF:
            return wam_compare_ints(left.data.ref_addr, right.data.ref_addr);
        default:
            return 0;
    }
}

static int wam_compare_stored_terms_for_qsort(const void *left,
                                              const void *right) {
    const WamStoredTerm *left_term = (const WamStoredTerm *)left;
    const WamStoredTerm *right_term = (const WamStoredTerm *)right;
    return wam_compare_stored_value(left_term, left_term->root,
                                    right_term, right_term->root);
}

static void wam_aggregate_sort_dedup_items(WamAggregateFrame *frame) {
    if (frame->item_count < 2) return;
    qsort(frame->items, (size_t)frame->item_count, sizeof(WamStoredTerm),
          wam_compare_stored_terms_for_qsort);

    int out = 1;
    for (int i = 1; i < frame->item_count; i++) {
        WamStoredTerm *previous = &frame->items[out - 1];
        WamStoredTerm *current = &frame->items[i];
        if (wam_compare_stored_value(previous, previous->root,
                                     current, current->root) == 0) {
            wam_stored_term_free(current);
        } else {
            if (out != i) {
                frame->items[out] = *current;
                memset(current, 0, sizeof(WamStoredTerm));
            }
            out++;
        }
    }
    frame->item_count = out;
}

static void wam_sort_aggregate_item_indices(WamAggregateFrame *frame,
                                            int *indices,
                                            int index_count) {
    for (int i = 1; i < index_count; i++) {
        int current = indices[i];
        int j = i - 1;
        while (j >= 0 &&
               wam_compare_stored_value(&frame->items[indices[j]],
                                         frame->items[indices[j]].root,
                                         &frame->items[current],
                                         frame->items[current].root) > 0) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = current;
    }
}

static int wam_dedup_sorted_aggregate_item_indices(WamAggregateFrame *frame,
                                                   int *indices,
                                                   int index_count) {
    if (index_count < 2) return index_count;
    int out = 1;
    for (int i = 1; i < index_count; i++) {
        WamStoredTerm *previous = &frame->items[indices[out - 1]];
        WamStoredTerm *current = &frame->items[indices[i]];
        if (wam_compare_stored_value(previous, previous->root,
                                     current, current->root) != 0) {
            indices[out++] = indices[i];
        }
    }
    return out;
}

static bool wam_build_aggregate_list_from_indices(WamState *state,
                                                  WamAggregateFrame *frame,
                                                  int *indices,
                                                  int index_count,
                                                  WamValue *out) {
    WamValue tail = val_atom("[]");
    for (int i = index_count - 1; i >= 0; i--) {
        int item_index = indices[i];
        WamValue head;
        if (!wam_materialize_stored_term(state, &frame->items[item_index],
                                         frame->items[item_index].root,
                                         &head)) return false;
        if (!wam_ensure_heap_slots(state, 2)) return false;
        int base = state->H;
        state->H_array[state->H++] = head;
        state->H_array[state->H++] = tail;
        tail.tag = VAL_LIST;
        tail.data.ref_addr = base;
    }
    *out = tail;
    return true;
}

static bool wam_copy_current_witness_tuple(WamState *state,
                                           WamAggregateFrame *frame,
                                           WamStoredTerm *term) {
    memset(term, 0, sizeof(WamStoredTerm));
    if (frame->witness_count <= 0) {
        term->root = val_atom("[]");
        return true;
    }
    if (frame->witness_count == 1) {
        WamValue *cell = wam_aggregate_witness_cell(state, frame, 0);
        if (!cell) return false;
        return wam_copy_term_to_stored(state, term, *cell, &term->root);
    }

    WamValue tail = val_atom("[]");
    for (int i = frame->witness_count - 1; i >= 0; i--) {
        WamValue *cell = wam_aggregate_witness_cell(state, frame, i);
        if (!cell) return false;
        WamValue head;
        if (!wam_copy_term_to_stored(state, term, *cell, &head)) return false;
        if (!wam_stored_term_reserve(term, 2)) return false;
        int base = term->cell_count;
        term->cell_count += 2;
        term->cells[base] = head;
        term->cells[base + 1] = tail;
        tail.tag = VAL_LIST;
        tail.data.ref_addr = base;
    }
    term->root = tail;
    return true;
}

static bool wam_witness_regs_are_bound(WamState *state,
                                       WamAggregateFrame *frame) {
    for (int i = 0; i < frame->witness_count; i++) {
        WamValue *witness = wam_aggregate_witness_cell(state, frame, i);
        if (!witness) return false;
        WamValue *cell = wam_deref_ptr(state, witness);
        if (val_is_unbound(*cell)) return false;
    }
    return true;
}

static int wam_select_aggregate_witness_group(WamState *state,
                                              WamAggregateFrame *frame) {
    if (frame->witness_count <= 0) return 0;
    if (frame->item_count <= 0) return -1;
    if (!wam_witness_regs_are_bound(state, frame)) {
        return 0;
    }

    WamStoredTerm selected;
    if (!wam_copy_current_witness_tuple(state, frame, &selected)) return -1;
    int selected_index = -1;
    for (int i = 0; i < frame->item_count; i++) {
        if (wam_compare_stored_value(&selected, selected.root,
                                     &frame->witnesses[i],
                                     frame->witnesses[i].root) == 0) {
            selected_index = i;
            break;
        }
    }
    wam_stored_term_free(&selected);
    return selected_index;
}

static bool wam_build_witness_group_indices(WamAggregateFrame *frame,
                                            int selected_index,
                                            int **indices_out,
                                            int *index_count_out) {
    if (selected_index < 0 || selected_index >= frame->item_count) return false;
    int *indices = malloc(sizeof(int) * (size_t)frame->item_count);
    if (!indices) return false;
    int count = 0;
    WamStoredTerm *selected = &frame->witnesses[selected_index];
    for (int i = 0; i < frame->item_count; i++) {
        if (wam_compare_stored_value(selected, selected->root,
                                     &frame->witnesses[i],
                                     frame->witnesses[i].root) == 0) {
            indices[count++] = i;
        }
    }
    *indices_out = indices;
    *index_count_out = count;
    return true;
}

static bool wam_build_witness_group_reps(WamAggregateFrame *frame,
                                         int **reps_out,
                                         int *group_count_out) {
    if (frame->item_count <= 0) {
        *reps_out = NULL;
        *group_count_out = 0;
        return true;
    }
    int *reps = malloc(sizeof(int) * (size_t)frame->item_count);
    if (!reps) return false;
    int group_count = 0;
    for (int i = 0; i < frame->item_count; i++) {
        bool seen = false;
        for (int g = 0; g < group_count; g++) {
            int rep = reps[g];
            if (wam_compare_stored_value(&frame->witnesses[rep],
                                         frame->witnesses[rep].root,
                                         &frame->witnesses[i],
                                         frame->witnesses[i].root) == 0) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            reps[group_count++] = i;
        }
    }
    *reps_out = reps;
    *group_count_out = group_count;
    return true;
}

static bool wam_materialize_witness_component(WamState *state,
                                              WamStoredTerm *witness,
                                              int witness_count,
                                              int component_index,
                                              WamValue *out) {
    if (component_index < 0 || component_index >= witness_count) return false;
    if (witness_count == 1) {
        return wam_materialize_stored_term(state, witness, witness->root, out);
    }

    WamValue cursor = witness->root;
    for (int i = 0; i <= component_index; i++) {
        if (cursor.tag != VAL_LIST) return false;
        int base = cursor.data.ref_addr;
        if (base < 0 || base + 1 >= witness->cell_count) return false;
        if (i == component_index) {
            return wam_materialize_stored_term(state, witness,
                                               witness->cells[base], out);
        }
        cursor = witness->cells[base + 1];
    }
    return false;
}

static bool wam_unify_witness_registers_from_group(WamState *state,
                                                   WamAggregateFrame *frame,
                                                   int selected_index) {
    if (frame->witness_count <= 0) return true;
    WamStoredTerm *witness = &frame->witnesses[selected_index];
    for (int i = 0; i < frame->witness_count; i++) {
        WamValue value;
        if (!wam_materialize_witness_component(state, witness,
                                               frame->witness_count,
                                               i, &value)) return false;
        WamValue *target = wam_aggregate_witness_cell(state, frame, i);
        if (!target) return false;
        if (!wam_unify(state, target, &value)) return false;
    }
    return true;
}

static bool wam_bind_aggregate_group(WamState *state,
                                     WamAggregateGroupIterator *iter,
                                     int group_index) {
    if (group_index < 0 || group_index >= iter->group_count) return false;
    int selected_index = iter->group_reps[group_index];
    WamAggregateFrame frame_view;
    memset(&frame_view, 0, sizeof(WamAggregateFrame));
    frame_view.kind = iter->kind;
    frame_view.is_meta = iter->is_meta;
    frame_view.result_reg = iter->result_reg;
    frame_view.result_is_y = iter->result_is_y;
    frame_view.meta_result = iter->meta_result;
    frame_view.items = iter->items;
    frame_view.witnesses = iter->witnesses;
    frame_view.item_count = iter->item_count;
    frame_view.item_cap = iter->item_cap;
    frame_view.witness_count = iter->witness_count;
    for (int i = 0; i < iter->witness_count; i++) {
        frame_view.witness_regs[i] = iter->witness_regs[i];
        frame_view.witness_is_y[i] = iter->witness_is_y[i];
        frame_view.meta_witnesses[i] = iter->meta_witnesses[i];
    }

    int *indices = NULL;
    int index_count = 0;
    if (!wam_build_witness_group_indices(&frame_view, selected_index,
                                         &indices, &index_count)) return false;
    if (strcmp(iter->kind, "setof") == 0) {
        wam_sort_aggregate_item_indices(&frame_view, indices, index_count);
        index_count =
            wam_dedup_sorted_aggregate_item_indices(&frame_view, indices,
                                                    index_count);
    }

    WamValue result_list;
    bool witness_ok =
        wam_unify_witness_registers_from_group(state, &frame_view,
                                               selected_index);
    bool list_ok = witness_ok &&
        wam_build_aggregate_list_from_indices(state, &frame_view, indices,
                                              index_count, &result_list);
    free(indices);
    if (!list_ok) return false;

    WamValue *result_cell = wam_aggregate_result_cell(state, &frame_view);
    if (!wam_unify(state, result_cell, &result_list)) return false;
    if (frame_view.is_meta) {
        WamValue *meta_result = wam_deref_ptr(state, &frame_view.meta_result);
        state->A[0] = *meta_result;
    }
    state->P = iter->return_pc;
    return true;
}

static bool wam_bind_next_aggregate_group(WamState *state) {
    if (state->aggregate_group_top <= 0) return false;
    WamAggregateGroupIterator *iter =
        &state->aggregate_group_iters[state->aggregate_group_top - 1];
    if (iter->next_group >= iter->group_count) {
        state->aggregate_group_top--;
        wam_aggregate_group_iterator_free(iter);
        return false;
    }

    int group_index = iter->next_group++;
    bool has_more = iter->next_group < iter->group_count;
    int return_pc = iter->return_pc;
    if (has_more) {
        push_choice_point(state, WAM_AGGREGATE_NEXT_GROUP, 32);
    }

    bool ok = wam_bind_aggregate_group(state, iter, group_index);
    if (!has_more) {
        state->aggregate_group_top--;
        wam_aggregate_group_iterator_free(iter);
    }
    if (ok) {
        state->P = return_pc;
    }
    return ok;
}

static bool wam_push_aggregate_group_iterator(WamState *state,
                                              WamAggregateFrame *frame,
                                              int *group_reps,
                                              int group_count,
                                              int return_pc) {
    if (state->aggregate_group_top >= WAM_AGGREGATE_STACK_SIZE) return false;
    WamAggregateGroupIterator *iter =
        &state->aggregate_group_iters[state->aggregate_group_top];
    wam_aggregate_group_iterator_free(iter);
    iter->kind = frame->kind;
    iter->is_meta = frame->is_meta;
    iter->return_pc = return_pc;
    iter->result_reg = frame->result_reg;
    iter->result_is_y = frame->result_is_y;
    iter->meta_result = frame->meta_result;
    iter->items = frame->items;
    iter->witnesses = frame->witnesses;
    iter->item_count = frame->item_count;
    iter->item_cap = frame->item_cap;
    iter->witness_count = frame->witness_count;
    for (int i = 0; i < frame->witness_count; i++) {
        iter->witness_regs[i] = frame->witness_regs[i];
        iter->witness_is_y[i] = frame->witness_is_y[i];
        iter->meta_witnesses[i] = frame->meta_witnesses[i];
    }
    iter->group_reps = group_reps;
    iter->group_count = group_count;
    iter->next_group = 0;
    state->aggregate_group_top++;

    frame->items = NULL;
    frame->witnesses = NULL;
    frame->item_count = 0;
    frame->item_cap = 0;
    return true;
}

static void wam_aggregate_frame_free(WamAggregateFrame *frame) {
    for (int i = 0; i < frame->item_count; i++) {
        wam_stored_term_free(&frame->items[i]);
        if (frame->witnesses) {
            wam_stored_term_free(&frame->witnesses[i]);
        }
    }
    free(frame->items);
    free(frame->witnesses);
    memset(frame, 0, sizeof(WamAggregateFrame));
}

static void wam_aggregate_clear_all(WamState *state) {
    while (state->aggregate_top > 0) {
        state->aggregate_top--;
        wam_aggregate_frame_free(&state->aggregate_frames[state->aggregate_top]);
    }
    wam_trim_aggregate_group_iters(state, 0);
}

static bool wam_aggregate_append_item(WamState *state,
                                      WamAggregateFrame *frame,
                                      WamValue value) {
    if (frame->item_count >= frame->item_cap) {
        int new_cap = frame->item_cap ? frame->item_cap * 2 : WAM_INITIAL_CAP;
        WamStoredTerm *items =
            realloc(frame->items, sizeof(WamStoredTerm) * (size_t)new_cap);
        if (!items) return false;
        frame->items = items;
        WamStoredTerm *witnesses = frame->witnesses;
        if (frame->witness_count > 0) {
            witnesses =
                realloc(frame->witnesses, sizeof(WamStoredTerm) * (size_t)new_cap);
            if (!witnesses) return false;
            frame->witnesses = witnesses;
        }
        memset(items + frame->item_cap, 0,
               sizeof(WamStoredTerm) * (size_t)(new_cap - frame->item_cap));
        if (frame->witness_count > 0) {
            memset(witnesses + frame->item_cap, 0,
                   sizeof(WamStoredTerm) * (size_t)(new_cap - frame->item_cap));
        }
        frame->item_cap = new_cap;
    }
    WamStoredTerm *term = &frame->items[frame->item_count];
    memset(term, 0, sizeof(WamStoredTerm));
    if (!wam_copy_term_to_stored(state, term, value, &term->root)) {
        wam_stored_term_free(term);
        return false;
    }
    if (frame->witness_count > 0) {
        WamStoredTerm *witness = &frame->witnesses[frame->item_count];
        if (!wam_copy_current_witness_tuple(state, frame, witness)) {
            wam_stored_term_free(term);
            wam_stored_term_free(witness);
            return false;
        }
    }
    frame->item_count++;
    return true;
}

static bool wam_build_aggregate_list(WamState *state,
                                     WamAggregateFrame *frame,
                                     WamValue *out) {
    WamValue tail = val_atom("[]");
    for (int i = frame->item_count - 1; i >= 0; i--) {
        WamValue head;
        if (!wam_materialize_stored_term(state, &frame->items[i],
                                         frame->items[i].root, &head)) return false;
        if (!wam_ensure_heap_slots(state, 2)) return false;
        int base = state->H;
        state->H_array[state->H++] = head;
        state->H_array[state->H++] = tail;
        tail.tag = VAL_LIST;
        tail.data.ref_addr = base;
    }
    *out = tail;
    return true;
}

static int wam_find_matching_end_aggregate(WamState *state, int begin_pc) {
    int depth = 0;
    for (int pc = begin_pc + 1; pc < state->code_size; pc++) {
        if (state->code[pc].tag == INSTR_BEGIN_AGGREGATE) {
            depth++;
        } else if (state->code[pc].tag == INSTR_END_AGGREGATE) {
            if (depth == 0) return pc;
            depth--;
        }
    }
    return -1;
}

static bool wam_finalize_aggregate_frame(WamState *state,
                                         WamAggregateFrame *frame) {
    bool is_collect = strcmp(frame->kind, "collect") == 0;
    bool is_bagof = strcmp(frame->kind, "bagof") == 0;
    bool is_setof = strcmp(frame->kind, "setof") == 0;
    if (!is_collect && !is_bagof && !is_setof) return false;
    int next_pc = frame->is_meta ? frame->return_pc : frame->end_pc + 1;
    int frame_index = state->aggregate_top - 1;
    if (frame->sentinel_b > 0 && frame->sentinel_b <= state->B) {
        ChoicePoint *sentinel = &state->B_array[frame->sentinel_b - 1];
        restore_choice_point(state, sentinel);
    }
    wam_prune_choice_points(state, frame->base_b);

    if ((is_bagof || is_setof) && frame->item_count == 0) {
        wam_aggregate_frame_free(frame);
        state->aggregate_top = frame_index;
        state->P = next_pc;
        return false;
    }

    WamValue result_list;
    if (frame->witness_count > 0) {
        if (!wam_witness_regs_are_bound(state, frame)) {
            int *group_reps = NULL;
            int group_count = 0;
            if (!wam_build_witness_group_reps(frame, &group_reps,
                                              &group_count)) return false;
            if (group_count <= 0) {
                free(group_reps);
                wam_aggregate_frame_free(frame);
                state->aggregate_top = frame_index;
                state->P = next_pc;
                return false;
            }
            if (!wam_push_aggregate_group_iterator(state, frame, group_reps,
                                                   group_count, next_pc)) {
                free(group_reps);
                return false;
            }
            memset(frame, 0, sizeof(WamAggregateFrame));
            state->aggregate_top = frame_index;
            return wam_bind_next_aggregate_group(state);
        }
        int selected_index = wam_select_aggregate_witness_group(state, frame);
        if (selected_index < 0) {
            wam_aggregate_frame_free(frame);
            state->aggregate_top = frame_index;
            state->P = next_pc;
            return false;
        }
        int *indices = NULL;
        int index_count = 0;
        if (!wam_build_witness_group_indices(frame, selected_index,
                                             &indices, &index_count)) return false;
        if (is_setof) {
            wam_sort_aggregate_item_indices(frame, indices, index_count);
            index_count =
                wam_dedup_sorted_aggregate_item_indices(frame, indices, index_count);
        }
        bool witness_ok =
            wam_unify_witness_registers_from_group(state, frame, selected_index);
        bool list_ok = witness_ok &&
            wam_build_aggregate_list_from_indices(state, frame, indices,
                                                  index_count, &result_list);
        free(indices);
        if (!list_ok) return false;
    } else {
        if (is_setof) {
            wam_aggregate_sort_dedup_items(frame);
        }
        if (!wam_build_aggregate_list(state, frame, &result_list)) return false;
    }
    WamValue *result_cell = wam_aggregate_result_cell(state, frame);
    bool ok = wam_unify(state, result_cell, &result_list);
    if (ok && frame->is_meta) {
        WamValue *meta_result = wam_deref_ptr(state, &frame->meta_result);
        state->A[0] = *meta_result;
    }
    wam_aggregate_frame_free(frame);
    state->aggregate_top = frame_index;
    if (!ok) return false;
    state->P = next_pc;
    return true;
}

static bool wam_collect_meta_aggregate_success(WamState *state) {
    if (state->aggregate_top <= 0) return false;
    WamAggregateFrame *frame =
        &state->aggregate_frames[state->aggregate_top - 1];
    if (!frame->is_meta) return false;
    (void)wam_aggregate_append_item(state, frame, frame->meta_template);
    return false;
}

static bool wam_finalize_meta_aggregate(WamState *state) {
    if (state->aggregate_top <= 0) return false;
    WamAggregateFrame *frame =
        &state->aggregate_frames[state->aggregate_top - 1];
    if (!frame->is_meta) return false;
    return wam_finalize_aggregate_frame(state, frame);
}

static bool wam_dispatch_aggregate_meta(WamState *state, const char *kind,
                                        int return_pc) {
    if (strcmp(kind, "collect") != 0 &&
        strcmp(kind, "bagof") != 0 &&
        strcmp(kind, "setof") != 0) return false;
    if (state->aggregate_top >= WAM_AGGREGATE_STACK_SIZE) return false;

    WamAggregateFrame *frame = &state->aggregate_frames[state->aggregate_top++];
    memset(frame, 0, sizeof(WamAggregateFrame));
    frame->kind = kind;
    frame->is_meta = true;
    frame->begin_pc = state->P;
    frame->end_pc = return_pc - 1;
    frame->return_pc = return_pc;
    frame->base_b = state->B;
    frame->sentinel_b = state->B + 1;
    frame->meta_template = state->A[0];
    frame->meta_result = state->A[2];

    if (strcmp(kind, "bagof") == 0 || strcmp(kind, "setof") == 0) {
        WamValue exclude[WAM_AGGREGATE_META_MAX_VARS];
        int exclude_count = 0;
        if (!wam_collect_term_vars(state, state->A[0], exclude,
                                   &exclude_count,
                                   WAM_AGGREGATE_META_MAX_VARS)) {
            state->aggregate_top--;
            memset(frame, 0, sizeof(WamAggregateFrame));
            return false;
        }
        if (!wam_collect_goal_witnesses(state, state->A[1], exclude,
                                        &exclude_count,
                                        WAM_AGGREGATE_META_MAX_VARS,
                                        frame->meta_witnesses,
                                        &frame->witness_count)) {
            state->aggregate_top--;
            memset(frame, 0, sizeof(WamAggregateFrame));
            return false;
        }
    }

    push_choice_point(state, WAM_AGGREGATE_META_DONE, 32);
    return wam_invoke_goal_as_call(state, state->A[1],
                                   WAM_AGGREGATE_META_COLLECT);
}

static bool wam_begin_aggregate(WamState *state, Instruction *instr) {
    if (strcmp(instr->as.aggregate.kind, "collect") != 0 &&
        strcmp(instr->as.aggregate.kind, "bagof") != 0 &&
        strcmp(instr->as.aggregate.kind, "setof") != 0) return false;
    if (instr->as.aggregate.witness_count < 0 ||
        instr->as.aggregate.witness_count > WAM_AGGREGATE_MAX_WITNESSES) return false;
    if (state->aggregate_top > 0) {
        WamAggregateFrame *top = &state->aggregate_frames[state->aggregate_top - 1];
        if (top->begin_pc == state->P && top->sentinel_b == state->B) {
            return wam_finalize_aggregate_frame(state, top);
        }
    }
    if (state->aggregate_top >= WAM_AGGREGATE_STACK_SIZE) return false;
    int end_pc = wam_find_matching_end_aggregate(state, state->P);
    if (end_pc < 0) return false;

    WamAggregateFrame *frame = &state->aggregate_frames[state->aggregate_top++];
    memset(frame, 0, sizeof(WamAggregateFrame));
    frame->kind = instr->as.aggregate.kind;
    frame->begin_pc = state->P;
    frame->end_pc = end_pc;
    frame->base_b = state->B;
    frame->sentinel_b = state->B + 1;
    frame->template_reg = instr->as.aggregate.template_reg;
    frame->template_is_y = instr->as.aggregate.template_is_y;
    frame->result_reg = instr->as.aggregate.result_reg;
    frame->result_is_y = instr->as.aggregate.result_is_y;
    frame->witness_count = instr->as.aggregate.witness_count;
    for (int i = 0; i < frame->witness_count; i++) {
        frame->witness_regs[i] = instr->as.aggregate.witness_regs[i];
        frame->witness_is_y[i] = instr->as.aggregate.witness_is_y[i];
    }
    push_choice_point(state, state->P, 32);
    state->P++;
    return true;
}

static bool wam_end_aggregate(WamState *state, Instruction *instr) {
    if (state->aggregate_top <= 0) return false;
    WamAggregateFrame *frame = &state->aggregate_frames[state->aggregate_top - 1];
    if (strcmp(frame->kind, "collect") != 0 &&
        strcmp(frame->kind, "bagof") != 0 &&
        strcmp(frame->kind, "setof") != 0) return false;
    WamValue *template_cell =
        resolve_reg(state, instr->as.aggregate.template_reg,
                    instr->as.aggregate.template_is_y);
    if (!wam_aggregate_append_item(state, frame, *template_cell)) return false;
    if (state->B > frame->sentinel_b) {
        return false;
    }
    return wam_finalize_aggregate_frame(state, frame);
}

void wam_state_init(WamState *state) {
    memset(state, 0, sizeof(WamState));
    state->H_cap = WAM_INITIAL_CAP;
    state->TR_cap = WAM_INITIAL_CAP;
    state->B_cap = WAM_INITIAL_CAP;
    state->E_cap = WAM_INITIAL_CAP;
    state->E = -1;
    state->H_array = malloc(sizeof(WamValue) * state->H_cap);
    state->TR_array = malloc(sizeof(TrailEntry) * state->TR_cap);
    state->B_array = malloc(sizeof(ChoicePoint) * state->B_cap);
    state->E_array = malloc(sizeof(EnvFrame) * state->E_cap);
}

void wam_free_state(WamState *state) {
    wam_aggregate_clear_all(state);
    for (int i = 0; i < state->B; i++) {
        free(state->B_array[i].foreign_results);
        state->B_array[i].foreign_results = NULL;
    }
    for (int i = 0; i < state->atom_table_size; i++) {
        AtomEntry *e = state->atom_table[i];
        while (e) {
            AtomEntry *next = e->next;
            free(e->str);
            free(e);
            e = next;
        }
    }
    free(state->atom_table);
    for (int i = 0; i < state->code_size; i++) {
        if (state->code[i].tag == INSTR_SWITCH_ON_CONSTANT || state->code[i].tag == INSTR_SWITCH_ON_STRUCTURE || state->code[i].tag == INSTR_SWITCH_ON_TERM) {
            free(state->code[i].as.switch_index.hash_table);
        }
        if (state->code[i].tag == INSTR_SWITCH_ON_TERM) {
            free(state->code[i].as.switch_index.s_hash_table);
        }
    }
    free(state->code);
    free(state->H_array);
    free(state->TR_array);
    free(state->B_array);
    free(state->E_array);
    wam_bidirectional_distance_cache_clear(state);
    free(state->category_edges);
    free(state->category_edges_by_child);
    free(state->category_ids);
    free(state->category_id_by_atom);
    free(state->category_id_by_value);
    free(state->weighted_edges);
    free(state->direct_distance_edges);
    free(state->relation_edges);
    free(state->kernel_edge_bindings);
    memset(state, 0, sizeof(WamState));
}

int wam_run_predicate(WamState *state, const char *pred,
                      WamValue *args, int arity) {
    int entry = resolve_predicate_hash(state, pred);
    if (entry < 0) return WAM_ERR_OOB;
    int base_b = state->B;
    int base_call_base_top = state->call_base_top;
    if (state->call_base_top >= WAM_CALL_STACK_SIZE) return WAM_ERR_OOB;
    state->call_bases[state->call_base_top] = base_b;
    state->call_base_preserve_choice[state->call_base_top] = false;
    state->call_base_top++;
    for (int i = 0; i < arity; i++) {
        state->A[i] = val_is_unbound(args[i]) ? wam_make_ref(state) : args[i];
    }
    state->CP = WAM_HALT;
    state->P = entry;
    int rc = wam_run(state);
    wam_prune_choice_points(state, base_b);
    for (int i = 0; i < arity; i++) {
        WamValue *cell = wam_deref_ptr(state, &state->A[i]);
        state->A[i] = *cell;
    }
    state->call_base_top = base_call_base_top;
    return rc;
}

static bool wam_eval_arith(WamState *state, WamValue value, int *out) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag == VAL_INT) {
        *out = cell->data.integer;
        return true;
    }
    if (cell->tag != VAL_STR) return false;

    int addr = cell->data.ref_addr;
    WamValue *functor = &state->H_array[addr];
    if (functor->tag != VAL_ATOM) return false;

    int lhs = 0;
    int rhs = 0;
    if (!wam_eval_arith(state, state->H_array[addr + 1], &lhs)) return false;
    if (!wam_eval_arith(state, state->H_array[addr + 2], &rhs)) return false;

    if (strcmp(functor->data.atom, "+/2") == 0) {
        *out = lhs + rhs;
        return true;
    }
    if (strcmp(functor->data.atom, "-/2") == 0) {
        *out = lhs - rhs;
        return true;
    }
    if (strcmp(functor->data.atom, "*/2") == 0) {
        *out = lhs * rhs;
        return true;
    }
    if (strcmp(functor->data.atom, "//2") == 0 || strcmp(functor->data.atom, "div/2") == 0) {
        if (rhs == 0) return false;
        *out = lhs / rhs;
        return true;
    }
    if (strcmp(functor->data.atom, "///2") == 0) {
        /* // is integer (floored) division. */
        if (rhs == 0) return false;
        int q = lhs / rhs;
        if ((lhs % rhs != 0) && ((lhs < 0) != (rhs < 0))) q -= 1;
        *out = q;
        return true;
    }
    if (strcmp(functor->data.atom, "mod/2") == 0) {
        /* mod follows the sign of the divisor (floored modulo). */
        if (rhs == 0) return false;
        int m = lhs % rhs;
        if (m != 0 && ((m < 0) != (rhs < 0))) m += rhs;
        *out = m;
        return true;
    }
    return false;
}

static bool wam_term_functor(WamState *state, WamValue term,
                             WamValue *name_out, int *arity_out) {
    WamValue *cell = wam_deref_ptr(state, &term);
    if (cell->tag == VAL_ATOM) {
        *name_out = *cell;
        *arity_out = 0;
        return true;
    }
    if (cell->tag == VAL_INT) {
        *name_out = *cell;
        *arity_out = 0;
        return true;
    }
    if (cell->tag == VAL_FLOAT) {
        *name_out = *cell;
        *arity_out = 0;
        return true;
    }
    if (cell->tag == VAL_LIST) {
        *name_out = val_atom(".");
        *arity_out = 2;
        return true;
    }
    if (cell->tag != VAL_STR) return false;

    WamValue *functor = &state->H_array[cell->data.ref_addr];
    if (functor->tag != VAL_ATOM) return false;
    const char *slash = strrchr(functor->data.atom, ''/'');
    if (!slash) return false;
    char *end = NULL;
    long arity = strtol(slash + 1, &end, 10);
    if (!end || *end != 0 || arity < 0 || arity > 255) return false;

    size_t len = (size_t)(slash - functor->data.atom);
    char name_buf[256];
    if (len >= sizeof(name_buf)) return false;
    memcpy(name_buf, functor->data.atom, len);
    name_buf[len] = 0;
    *name_out = val_atom(wam_intern_atom(state, name_buf));
    *arity_out = (int)arity;
    return true;
}

static bool wam_make_structure_from_functor(WamState *state,
                                            const char *name,
                                            int arity,
                                            WamValue *out) {
    char functor_buf[320];
    int written = snprintf(functor_buf, sizeof(functor_buf), "%s/%d", name, arity);
    if (written < 0 || (size_t)written >= sizeof(functor_buf)) return false;

    int required = state->H + 1 + arity;
    if (required >= state->H_cap) {
        if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
        while (required >= state->H_cap) state->H_cap *= 2;
        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
    }

    out->tag = VAL_STR;
    out->data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(wam_intern_atom(state, functor_buf));
    for (int i = 0; i < arity; i++) {
        state->H_array[state->H++] = val_unbound("_");
    }
    return true;
}

static bool wam_term_arg(WamState *state, int index, WamValue term, WamValue **out) {
    if (index <= 0) return false;
    WamValue *cell = wam_deref_ptr(state, &term);
    if (cell->tag == VAL_LIST) {
        if (index > 2) return false;
        *out = &state->H_array[cell->data.ref_addr + index - 1];
        return true;
    }
    if (cell->tag != VAL_STR) return false;

    WamValue *functor = &state->H_array[cell->data.ref_addr];
    if (functor->tag != VAL_ATOM) return false;
    const char *slash = strrchr(functor->data.atom, ''/'');
    if (!slash) return false;
    char *end = NULL;
    long arity = strtol(slash + 1, &end, 10);
    if (!end || *end != 0 || arity < 0 || arity > 255) return false;
    if (index > arity) return false;

    *out = &state->H_array[cell->data.ref_addr + index];
    return true;
}

static bool wam_unify_atom_from_cstr(WamState *state, WamValue *target, const char *value) {
    WamValue atom = val_atom(wam_intern_atom(state, value));
    return wam_unify(state, target, &atom);
}

static bool wam_execute_atom_concat(WamState *state) {
    WamValue *left = wam_deref_ptr(state, &state->A[0]);
    WamValue *right = wam_deref_ptr(state, &state->A[1]);
    WamValue *whole = wam_deref_ptr(state, &state->A[2]);
    bool left_bound = left->tag == VAL_ATOM;
    bool right_bound = right->tag == VAL_ATOM;
    bool whole_bound = whole->tag == VAL_ATOM;

    if (left_bound && right_bound) {
        size_t left_len = strlen(left->data.atom);
        size_t right_len = strlen(right->data.atom);
        char *joined = malloc(left_len + right_len + 1);
        if (!joined) return false;
        memcpy(joined, left->data.atom, left_len);
        memcpy(joined + left_len, right->data.atom, right_len + 1);
        bool ok = wam_unify_atom_from_cstr(state, &state->A[2], joined);
        free(joined);
        return ok;
    }

    if (right_bound && whole_bound && val_is_unbound(*left)) {
        size_t right_len = strlen(right->data.atom);
        size_t whole_len = strlen(whole->data.atom);
        if (right_len > whole_len) return false;
        size_t prefix_len = whole_len - right_len;
        if (strcmp(whole->data.atom + prefix_len, right->data.atom) != 0) return false;
        char *prefix = malloc(prefix_len + 1);
        if (!prefix) return false;
        memcpy(prefix, whole->data.atom, prefix_len);
        prefix[prefix_len] = 0;
        bool ok = wam_unify_atom_from_cstr(state, &state->A[0], prefix);
        free(prefix);
        return ok;
    }

    if (left_bound && whole_bound && val_is_unbound(*right)) {
        size_t left_len = strlen(left->data.atom);
        size_t whole_len = strlen(whole->data.atom);
        if (left_len > whole_len) return false;
        if (strncmp(whole->data.atom, left->data.atom, left_len) != 0) return false;
        return wam_unify_atom_from_cstr(state, &state->A[1], whole->data.atom + left_len);
    }

    return false;
}

bool wam_execute_builtin(WamState *state, const char *op, int arity) {
    if (strcmp(op, "true/0") == 0 && arity == 0) return true;
    if ((strcmp(op, "fail/0") == 0 || strcmp(op, "false/0") == 0) && arity == 0) return false;
    if (strcmp(op, "!/0") == 0 && arity == 0) {
        if (state->call_base_top <= 0) return false;
        int target_b = state->call_bases[state->call_base_top - 1];
        if (target_b < 0 || target_b > state->B) return false;
        wam_prune_choice_points(state, target_b);
        return true;
    }

    if (strcmp(op, "=/2") == 0 && arity == 2) {
        return wam_unify(state, &state->A[0], &state->A[1]);
    }

    /* Strict (in)equality: previously missing from the table entirely,
       so the shared compiler emitted builtin_call ==/2 and it failed
       closed (even X = a, X == a was false) -- the class-7 gap found
       by the bind-through probe sweep. */
    if ((strcmp(op, "==/2") == 0 || strcmp(op, "\\\\==/2") == 0) && arity == 2) {
        bool eq = wam_term_strict_equal(state, &state->A[0], &state->A[1]);
        return (strcmp(op, "==/2") == 0) ? eq : !eq;
    }

    if (arity == 1) {
        WamValue *a1 = wam_deref_ptr(state, &state->A[0]);
        if (strcmp(op, "atom/1") == 0) return a1->tag == VAL_ATOM;
        if (strcmp(op, "integer/1") == 0) return a1->tag == VAL_INT;
        if (strcmp(op, "number/1") == 0) return a1->tag == VAL_INT || a1->tag == VAL_FLOAT;
        if (strcmp(op, "float/1") == 0) return a1->tag == VAL_FLOAT;
        if (strcmp(op, "var/1") == 0) return val_is_unbound(*a1);
        if (strcmp(op, "nonvar/1") == 0) return !val_is_unbound(*a1);
        if (strcmp(op, "compound/1") == 0) return a1->tag == VAL_STR || a1->tag == VAL_LIST;
        if (strcmp(op, "is_list/1") == 0) return a1->tag == VAL_LIST;
    }

    if (strcmp(op, "is/2") == 0 && arity == 2) {
        int result = 0;
        if (!wam_eval_arith(state, state->A[1], &result)) return false;
        WamValue value = val_int(result);
        return wam_unify(state, &state->A[0], &value);
    }

    if (strcmp(op, "functor/3") == 0 && arity == 3) {
        WamValue *term = wam_deref_ptr(state, &state->A[0]);
        if (!val_is_unbound(*term)) {
            WamValue name = val_unbound("Name");
            int term_arity = 0;
            if (!wam_term_functor(state, state->A[0], &name, &term_arity)) return false;
            WamValue arity_value = val_int(term_arity);
            return wam_unify(state, &state->A[1], &name) &&
                   wam_unify(state, &state->A[2], &arity_value);
        }

        WamValue *name = wam_deref_ptr(state, &state->A[1]);
        WamValue *arity_cell = wam_deref_ptr(state, &state->A[2]);
        if (arity_cell->tag != VAL_INT || arity_cell->data.integer < 0) return false;
        if (arity_cell->data.integer == 0) {
            if (name->tag != VAL_ATOM && name->tag != VAL_INT && name->tag != VAL_FLOAT) return false;
            return wam_unify(state, &state->A[0], name);
        }
        if (name->tag != VAL_ATOM) return false;
        WamValue structure;
        if (!wam_make_structure_from_functor(state, name->data.atom, arity_cell->data.integer, &structure)) return false;
        return wam_unify(state, &state->A[0], &structure);
    }

    if (strcmp(op, "arg/3") == 0 && arity == 3) {
        WamValue *index = wam_deref_ptr(state, &state->A[0]);
        if (index->tag != VAL_INT) return false;
        WamValue *arg = NULL;
        if (!wam_term_arg(state, index->data.integer, state->A[1], &arg)) return false;
        return wam_unify(state, &state->A[2], arg);
    }

    if (strcmp(op, "atom_concat/3") == 0 && arity == 3) {
        return wam_execute_atom_concat(state);
    }

    if (arity == 2) {
        int lhs = 0;
        int rhs = 0;
        if (!wam_eval_arith(state, state->A[0], &lhs)) return false;
        if (!wam_eval_arith(state, state->A[1], &rhs)) return false;

        if (strcmp(op, ">/2") == 0) return lhs > rhs;
        if (strcmp(op, "</2") == 0) return lhs < rhs;
        if (strcmp(op, ">=/2") == 0) return lhs >= rhs;
        if (strcmp(op, "=</2") == 0) return lhs <= rhs;
        if (strcmp(op, "=:=/2") == 0) return lhs == rhs;
        if (strcmp(op, "=\\\\=/2") == 0) return lhs != rhs;
    }

    return false;
}

bool wam_execute_foreign_predicate(WamState *state, const char *pred, int arity) {
    WamForeignHandler handler = resolve_foreign_predicate(state, pred, arity);
    if (!handler) return false;
    return handler(state, pred, arity);
}

static bool wam_ensure_category_edge_capacity(WamState *state, int additional) {
    if (additional <= 0) return true;
    if (state->category_edge_count > INT_MAX - additional) return false;
    int needed = state->category_edge_count + additional;
    if (state->category_edge_cap >= needed) return true;
    int new_cap = state->category_edge_cap ? state->category_edge_cap : WAM_INITIAL_CAP;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) {
            new_cap = needed;
            break;
        }
        new_cap *= 2;
    }
    CategoryEdge *edges =
        realloc(state->category_edges, sizeof(CategoryEdge) * (size_t)new_cap);
    if (!edges) return false;
    state->category_edges = edges;
    state->category_edge_cap = new_cap;
    return true;
}

void wam_register_category_parent(WamState *state, const char *child, const char *parent) {
    if (!wam_ensure_category_edge_capacity(state, 1)) return;
    state->category_edges[state->category_edge_count].child = wam_intern_atom(state, child);
    state->category_edges[state->category_edge_count].parent = wam_intern_atom(state, parent);
    state->category_edge_count++;
    state->category_child_index_dirty = true;
    wam_bidirectional_distance_cache_clear(state);
}

void wam_register_transitive_edge(WamState *state, const char *child, const char *parent) {
    wam_register_category_parent(state, child, parent);
}

void wam_register_relation_edge(WamState *state, const char *relation,
                                const char *from, const char *to) {
    if (!relation || !from || !to) return;
    if (state->relation_edge_count >= state->relation_edge_cap) {
        int new_cap = state->relation_edge_cap ? state->relation_edge_cap * 2 : WAM_INITIAL_CAP;
        RelationEdge *edges = realloc(state->relation_edges,
                                      sizeof(RelationEdge) * (size_t)new_cap);
        if (!edges) return;
        state->relation_edges = edges;
        state->relation_edge_cap = new_cap;
    }
    state->relation_edges[state->relation_edge_count].relation =
        wam_intern_atom(state, relation);
    state->relation_edges[state->relation_edge_count].child =
        wam_intern_atom(state, from);
    state->relation_edges[state->relation_edge_count].parent =
        wam_intern_atom(state, to);
    state->relation_edge_count++;
}

void wam_bind_kernel_edge_relation(WamState *state, const char *pred, int arity,
                                   const char *relation) {
    if (!pred || !relation || arity < 0) return;
    for (int i = 0; i < state->kernel_edge_binding_count; i++) {
        KernelEdgeBinding *binding = &state->kernel_edge_bindings[i];
        if (binding->arity == arity && strcmp(binding->pred, pred) == 0) {
            binding->relation = wam_intern_atom(state, relation);
            return;
        }
    }
    if (state->kernel_edge_binding_count >= state->kernel_edge_binding_cap) {
        int new_cap = state->kernel_edge_binding_cap
                          ? state->kernel_edge_binding_cap * 2
                          : WAM_INITIAL_CAP;
        KernelEdgeBinding *bindings =
            realloc(state->kernel_edge_bindings,
                    sizeof(KernelEdgeBinding) * (size_t)new_cap);
        if (!bindings) return;
        state->kernel_edge_bindings = bindings;
        state->kernel_edge_binding_cap = new_cap;
    }
    state->kernel_edge_bindings[state->kernel_edge_binding_count].pred =
        wam_intern_atom(state, pred);
    state->kernel_edge_bindings[state->kernel_edge_binding_count].arity = arity;
    state->kernel_edge_bindings[state->kernel_edge_binding_count].relation =
        wam_intern_atom(state, relation);
    state->kernel_edge_binding_count++;
}

const char *wam_lookup_kernel_edge_relation(WamState *state, const char *pred, int arity) {
    if (!pred) return NULL;
    for (int i = 0; i < state->kernel_edge_binding_count; i++) {
        KernelEdgeBinding *binding = &state->kernel_edge_bindings[i];
        if (binding->arity == arity && strcmp(binding->pred, pred) == 0) {
            return binding->relation;
        }
    }
    return NULL;
}

void wam_register_weighted_edge(WamState *state, const char *source, const char *target, double weight) {
    if (state->weighted_edge_count >= state->weighted_edge_cap) {
        state->weighted_edge_cap = state->weighted_edge_cap ? state->weighted_edge_cap * 2 : WAM_INITIAL_CAP;
        state->weighted_edges = realloc(state->weighted_edges, sizeof(WeightedEdge) * state->weighted_edge_cap);
    }
    state->weighted_edges[state->weighted_edge_count].source = wam_intern_atom(state, source);
    state->weighted_edges[state->weighted_edge_count].target = wam_intern_atom(state, target);
    state->weighted_edges[state->weighted_edge_count].weight = weight;
    state->weighted_edge_count++;
}

void wam_register_direct_distance_edge(WamState *state, const char *source, const char *target, double distance) {
    if (state->direct_distance_edge_count >= state->direct_distance_edge_cap) {
        state->direct_distance_edge_cap = state->direct_distance_edge_cap ? state->direct_distance_edge_cap * 2 : WAM_INITIAL_CAP;
        state->direct_distance_edges = realloc(state->direct_distance_edges, sizeof(WeightedEdge) * state->direct_distance_edge_cap);
    }
    state->direct_distance_edges[state->direct_distance_edge_count].source = wam_intern_atom(state, source);
    state->direct_distance_edges[state->direct_distance_edge_count].target = wam_intern_atom(state, target);
    state->direct_distance_edges[state->direct_distance_edge_count].weight = distance;
    state->direct_distance_edge_count++;
}

void wam_fact_source_init(WamFactSource *source) {
    memset(source, 0, sizeof(WamFactSource));
}

void wam_fact_source_close(WamFactSource *source) {
    free(source->edges);
    free(source->edges_by_child);
    memset(source, 0, sizeof(WamFactSource));
}

static void wam_fact_source_add_edge(WamState *state,
                                     WamFactSource *source,
                                     const char *child,
                                     const char *parent) {
    if (source->edge_count == 0) {
        source->owner_state = state;
    } else if (source->owner_state != state) {
        source->owner_state = NULL;
    }
    if (source->edge_count >= source->edge_cap) {
        source->edge_cap = source->edge_cap ? source->edge_cap * 2 : WAM_INITIAL_CAP;
        source->edges = realloc(source->edges, sizeof(CategoryEdge) * source->edge_cap);
    }
    source->edges[source->edge_count].child = wam_intern_atom(state, child);
    source->edges[source->edge_count].parent = wam_intern_atom(state, parent);
    source->edge_count++;
    source->child_index_dirty = true;
}

bool wam_fact_source_load_tsv(WamState *state, WamFactSource *source, const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return false;

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        char *start = line;
        while (*start == 32 || *start == 9) start++;
        if (*start == 0 || *start == 10 || *start == 35) continue;

        char *sep = strchr(start, 9);
        if (!sep) sep = strchr(start, 32);
        if (!sep) {
            fclose(file);
            return false;
        }

        *sep = 0;
        char *parent = sep + 1;
        while (*parent == 32 || *parent == 9) parent++;

        char *end = parent + strlen(parent);
        while (end > parent && (end[-1] == 10 || end[-1] == 13 ||
                                end[-1] == 32 || end[-1] == 9)) {
            *--end = 0;
        }

        if (*start == 0 || *parent == 0) {
            fclose(file);
            return false;
        }
        wam_fact_source_add_edge(state, source, start, parent);
    }

    fclose(file);
    return true;
}

bool wam_fact_source_load_lmdb(WamState *state, WamFactSource *source,
                               const char *env_path, const char *db_name) {
#ifndef WAM_C_ENABLE_LMDB
    (void)state;
    (void)source;
    (void)env_path;
    (void)db_name;
    return false;
#else
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_cursor *cursor = NULL;
    MDB_dbi dbi = 0;
    int rc = 0;
    bool ok = false;

    rc = mdb_env_create(&env);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_env_set_maxdbs(env, 16);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_env_open(env, env_path, MDB_RDONLY, 0664);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_dbi_open(txn, (db_name && db_name[0]) ? db_name : NULL, 0, &dbi);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_cursor_open(txn, dbi, &cursor);
    if (rc != MDB_SUCCESS) goto done;

    MDB_val key;
    MDB_val data;
    rc = mdb_cursor_get(cursor, &key, &data, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        char *child = malloc(key.mv_size + 1);
        char *parent = malloc(data.mv_size + 1);
        if (!child || !parent) {
            free(child);
            free(parent);
            goto done;
        }
        memcpy(child, key.mv_data, key.mv_size);
        child[key.mv_size] = 0;
        memcpy(parent, data.mv_data, data.mv_size);
        parent[data.mv_size] = 0;
        wam_fact_source_add_edge(state, source, child, parent);
        free(child);
        free(parent);
        rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT);
    }
    ok = (rc == MDB_NOTFOUND);

done:
    if (cursor) mdb_cursor_close(cursor);
    if (txn) mdb_txn_abort(txn);
    if (env) {
        if (dbi) mdb_dbi_close(env, dbi);
        mdb_env_close(env);
    }
    return ok;
#endif
}

static int wam_compare_category_edge_child_parent(const void *left, const void *right) {
    const CategoryEdge *a = (const CategoryEdge *)left;
    const CategoryEdge *b = (const CategoryEdge *)right;
    int child_cmp = strcmp(a->child, b->child);
    if (child_cmp != 0) return child_cmp;
    return strcmp(a->parent, b->parent);
}

static int wam_category_edge_lower_bound(CategoryEdge *edges, int count, const char *child) {
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (strcmp(edges[mid].child, child) < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static bool wam_fact_source_ensure_child_index(WamFactSource *source) {
    if (!source->child_index_dirty &&
        source->edges_by_child &&
        source->child_index_count == source->edge_count) {
        return true;
    }
    free(source->edges_by_child);
    source->edges_by_child = NULL;
    source->child_index_count = 0;
    if (source->edge_count == 0) {
        source->child_index_dirty = false;
        return true;
    }
    source->edges_by_child = malloc(sizeof(CategoryEdge) * (size_t)source->edge_count);
    if (!source->edges_by_child) return false;
    memcpy(source->edges_by_child, source->edges, sizeof(CategoryEdge) * (size_t)source->edge_count);
    qsort(source->edges_by_child, (size_t)source->edge_count, sizeof(CategoryEdge),
          wam_compare_category_edge_child_parent);
    source->child_index_count = source->edge_count;
    source->child_index_dirty = false;
    return true;
}

bool wam_fact_source_child_range(WamFactSource *source, const char *child,
                                 CategoryEdge **edges_out, int *count_out) {
    *edges_out = NULL;
    *count_out = 0;
    if (!wam_fact_source_ensure_child_index(source)) return false;
    int start = wam_category_edge_lower_bound(source->edges_by_child,
                                              source->child_index_count,
                                              child);
    if (start >= source->child_index_count ||
        strcmp(source->edges_by_child[start].child, child) != 0) {
        return true;
    }
    int end = start;
    while (end < source->child_index_count &&
           strcmp(source->edges_by_child[end].child, child) == 0) {
        end++;
    }
    *edges_out = source->edges_by_child + start;
    *count_out = end - start;
    return true;
}

int wam_fact_source_lookup_arg1(WamFactSource *source, const char *arg1,
                                CategoryEdge *out_edges, int max_edges) {
    CategoryEdge *edges = NULL;
    int count = 0;
    if (!wam_fact_source_child_range(source, arg1, &edges, &count)) return -1;
    for (int i = 0; i < count && i < max_edges; i++) {
        out_edges[i] = edges[i];
    }
    return count;
}

bool wam_register_category_parent_fact_source(WamState *state, WamFactSource *source) {
    if (source->edge_count <= 0) return true;
    if (!wam_ensure_category_edge_capacity(state, source->edge_count)) {
        return false;
    }

    CategoryEdge *target = state->category_edges + state->category_edge_count;
    if (source->owner_state == state) {
        memcpy(target, source->edges, sizeof(CategoryEdge) * (size_t)source->edge_count);
    } else {
        for (int i = 0; i < source->edge_count; i++) {
            target[i].child = wam_intern_atom(state, source->edges[i].child);
            target[i].parent = wam_intern_atom(state, source->edges[i].parent);
        }
    }
    state->category_edge_count += source->edge_count;
    state->category_child_index_dirty = true;
    wam_bidirectional_distance_cache_clear(state);
    return true;
}

static bool wam_state_ensure_category_child_index(WamState *state) {
    if (!state->category_child_index_dirty &&
        state->category_edges_by_child &&
        state->category_child_index_count == state->category_edge_count) {
        return true;
    }
    free(state->category_edges_by_child);
    state->category_edges_by_child = NULL;
    state->category_child_index_count = 0;
    if (state->category_edge_count == 0) {
        state->category_child_index_dirty = false;
        return true;
    }
    state->category_edges_by_child = malloc(sizeof(CategoryEdge) * (size_t)state->category_edge_count);
    if (!state->category_edges_by_child) return false;
    memcpy(state->category_edges_by_child, state->category_edges,
           sizeof(CategoryEdge) * (size_t)state->category_edge_count);
    qsort(state->category_edges_by_child, (size_t)state->category_edge_count,
          sizeof(CategoryEdge), wam_compare_category_edge_child_parent);
    state->category_child_index_count = state->category_edge_count;
    state->category_child_index_dirty = false;
    return true;
}

static bool wam_state_category_child_range(WamState *state, const char *child,
                                           CategoryEdge **edges_out,
                                           int *count_out) {
    *edges_out = NULL;
    *count_out = 0;
    if (!wam_state_ensure_category_child_index(state)) return false;
    int start = wam_category_edge_lower_bound(state->category_edges_by_child,
                                              state->category_child_index_count,
                                              child);
    if (start >= state->category_child_index_count ||
        strcmp(state->category_edges_by_child[start].child, child) != 0) {
        return true;
    }
    int end = start;
    while (end < state->category_child_index_count &&
           strcmp(state->category_edges_by_child[end].child, child) == 0) {
        end++;
    }
    *edges_out = state->category_edges_by_child + start;
    *count_out = end - start;
    return true;
}

static uint64_t wam_category_id_hash_atom(const char *atom) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)atom;
    while (*p) {
        h ^= (uint64_t)(*p++);
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t wam_category_id_hash_value(int id) {
    uint32_t x = (uint32_t)id;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return (uint64_t)x;
}

static int wam_category_id_index_capacity(int desired_count) {
    int needed = desired_count * 2;
    int capacity = WAM_INITIAL_CAP;
    while (capacity < needed && capacity < (INT_MAX / 2)) {
        capacity *= 2;
    }
    return capacity;
}

static int wam_category_id_find_atom_linear(WamState *state, const char *atom) {
    for (int i = 0; i < state->category_id_count; i++) {
        if (strcmp(state->category_ids[i].atom, atom) == 0) return i;
    }
    return -1;
}

static int wam_category_id_find_value_linear(WamState *state, int id) {
    for (int i = 0; i < state->category_id_count; i++) {
        if (state->category_ids[i].id == id) return i;
    }
    return -1;
}

static bool wam_category_id_atom_index_find(WamState *state,
                                            const char *atom,
                                            int *index_out) {
    if (!state->category_id_by_atom || state->category_id_by_atom_cap <= 0) {
        return false;
    }
    int mask = state->category_id_by_atom_cap - 1;
    int slot = (int)(wam_category_id_hash_atom(atom) & (uint64_t)mask);
    while (state->category_id_by_atom[slot].atom) {
        if (strcmp(state->category_id_by_atom[slot].atom, atom) == 0) {
            *index_out = state->category_id_by_atom[slot].index;
            return true;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

static bool wam_category_id_value_index_find(WamState *state,
                                             int id,
                                             int *index_out) {
    if (!state->category_id_by_value || state->category_id_by_value_cap <= 0) {
        return false;
    }
    int mask = state->category_id_by_value_cap - 1;
    int slot = (int)(wam_category_id_hash_value(id) & (uint64_t)mask);
    while (state->category_id_by_value[slot].occupied) {
        if (state->category_id_by_value[slot].id == id) {
            *index_out = state->category_id_by_value[slot].index;
            return true;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

static void wam_category_id_insert_atom_index(WamCategoryIdAtomIndexEntry *table,
                                              int capacity,
                                              const char *atom,
                                              int index) {
    int mask = capacity - 1;
    int slot = (int)(wam_category_id_hash_atom(atom) & (uint64_t)mask);
    while (table[slot].atom) {
        if (strcmp(table[slot].atom, atom) == 0) {
            table[slot].index = index;
            return;
        }
        slot = (slot + 1) & mask;
    }
    table[slot].atom = atom;
    table[slot].index = index;
}

static void wam_category_id_insert_value_index(WamCategoryIdValueIndexEntry *table,
                                               int capacity,
                                               int id,
                                               int index,
                                               bool replace_existing) {
    int mask = capacity - 1;
    int slot = (int)(wam_category_id_hash_value(id) & (uint64_t)mask);
    while (table[slot].occupied) {
        if (table[slot].id == id) {
            if (replace_existing) table[slot].index = index;
            return;
        }
        slot = (slot + 1) & mask;
    }
    table[slot].occupied = true;
    table[slot].id = id;
    table[slot].index = index;
}

static bool wam_rebuild_category_id_indexes_for_count(WamState *state,
                                                      int desired_count) {
    if (desired_count == 0 && state->category_id_count == 0) {
        free(state->category_id_by_atom);
        free(state->category_id_by_value);
        state->category_id_by_atom = NULL;
        state->category_id_by_value = NULL;
        state->category_id_by_atom_cap = 0;
        state->category_id_by_value_cap = 0;
        return true;
    }

    int capacity = wam_category_id_index_capacity(desired_count);
    WamCategoryIdAtomIndexEntry *atom_table =
        calloc((size_t)capacity, sizeof(WamCategoryIdAtomIndexEntry));
    WamCategoryIdValueIndexEntry *value_table =
        calloc((size_t)capacity, sizeof(WamCategoryIdValueIndexEntry));
    if (!atom_table || !value_table) {
        free(atom_table);
        free(value_table);
        return false;
    }

    for (int i = 0; i < state->category_id_count; i++) {
        wam_category_id_insert_atom_index(atom_table, capacity,
                                          state->category_ids[i].atom, i);
        wam_category_id_insert_value_index(value_table, capacity,
                                           state->category_ids[i].id, i, false);
    }

    free(state->category_id_by_atom);
    free(state->category_id_by_value);
    state->category_id_by_atom = atom_table;
    state->category_id_by_value = value_table;
    state->category_id_by_atom_cap = capacity;
    state->category_id_by_value_cap = capacity;
    return true;
}

static bool wam_rebuild_category_id_indexes(WamState *state) {
    return wam_rebuild_category_id_indexes_for_count(state,
                                                    state->category_id_count);
}

static bool wam_ensure_category_id_index_capacity(WamState *state,
                                                  int desired_count) {
    int required_capacity = wam_category_id_index_capacity(desired_count);
    if (state->category_id_by_atom &&
        state->category_id_by_value &&
        state->category_id_by_atom_cap >= required_capacity &&
        state->category_id_by_value_cap >= required_capacity) {
        return true;
    }
    return wam_rebuild_category_id_indexes_for_count(state, desired_count);
}

void wam_register_category_id(WamState *state, const char *atom, int id) {
    const char *interned = wam_intern_atom(state, atom);
    int existing_index = -1;
    if (!wam_category_id_atom_index_find(state, interned, &existing_index) &&
        (!state->category_id_by_atom || state->category_id_by_atom_cap <= 0)) {
        existing_index = wam_category_id_find_atom_linear(state, interned);
    }
    if (existing_index >= 0) {
        int old_id = state->category_ids[existing_index].id;
        state->category_ids[existing_index].id = id;
        if (old_id != id ||
            !state->category_id_by_atom ||
            !state->category_id_by_value) {
            (void)wam_rebuild_category_id_indexes(state);
        }
        wam_bidirectional_distance_cache_clear(state);
        return;
    }
    if (state->category_id_count >= state->category_id_cap) {
        state->category_id_cap = state->category_id_cap ? state->category_id_cap * 2 : WAM_INITIAL_CAP;
        WamCategoryIdEntry *entries =
            realloc(state->category_ids,
                    sizeof(WamCategoryIdEntry) * (size_t)state->category_id_cap);
        if (!entries) return;
        state->category_ids = entries;
    }
    if (!wam_ensure_category_id_index_capacity(state, state->category_id_count + 1)) {
        return;
    }
    int index = state->category_id_count;
    state->category_ids[index].atom = interned;
    state->category_ids[index].id = id;
    state->category_id_count++;
    wam_category_id_insert_atom_index(state->category_id_by_atom,
                                      state->category_id_by_atom_cap,
                                      interned, index);
    wam_category_id_insert_value_index(state->category_id_by_value,
                                       state->category_id_by_value_cap,
                                       id, index, false);
    wam_bidirectional_distance_cache_clear(state);
}

void wam_attach_bidirectional_child_csr(WamState *state, WamReverseCsrArtifact *artifact) {
    state->bidirectional_child_csr = artifact;
    wam_bidirectional_distance_cache_clear(state);
}

static int32_t wam_read_i32_le(const unsigned char *p) {
    uint32_t v = ((uint32_t)p[0]) |
                 ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) |
                 ((uint32_t)p[3] << 24);
    return (int32_t)v;
}

static uint32_t wam_read_u32_le(const unsigned char *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t wam_read_u64_le(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) {
        v = (v << 8) | (uint64_t)p[i];
    }
    return v;
}

static bool wam_pread_exact(int fd, void *buffer, size_t bytes, off_t offset) {
    unsigned char *out = (unsigned char *)buffer;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, out + done, bytes - done, offset + (off_t)done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        done += (size_t)n;
    }
    return true;
}

static bool wam_direct_read_exact(WamReverseCsrArtifact *artifact,
                                  void *buffer,
                                  size_t bytes,
                                  off_t offset) {
    if (!artifact->direct_io) {
        return wam_pread_exact(artifact->values_fd, buffer, bytes, offset);
    }
    if (artifact->direct_io_alignment == 0 || artifact->values_size < 0) return false;

    size_t alignment = artifact->direct_io_alignment;
    uint64_t start = (uint64_t)offset;
    uint64_t end = start + (uint64_t)bytes;
    if (end < start || end > (uint64_t)artifact->values_size) return false;
    uint64_t aligned_start = start - (start % (uint64_t)alignment);
    uint64_t aligned_end = ((end + (uint64_t)alignment - 1ULL) / (uint64_t)alignment) *
                           (uint64_t)alignment;
    if (aligned_end < aligned_start) return false;
    if (aligned_end > (uint64_t)artifact->values_size) return false;
    size_t read_bytes = (size_t)(aligned_end - aligned_start);
    if (read_bytes == 0 || ((uint64_t)read_bytes != aligned_end - aligned_start)) return false;

    void *scratch = NULL;
    if (posix_memalign(&scratch, alignment, read_bytes) != 0) return false;
    bool ok = wam_pread_exact(artifact->values_fd, scratch, read_bytes, (off_t)aligned_start);
    if (ok) {
        memcpy(buffer, ((unsigned char *)scratch) + (start - aligned_start), bytes);
    }
    free(scratch);
    return ok;
}

static void wam_drop_fd_range(int fd, off_t offset, off_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (fd >= 0 && bytes > 0) {
        (void)posix_fadvise(fd, offset, bytes, POSIX_FADV_DONTNEED);
    }
#else
    (void)fd;
    (void)offset;
    (void)bytes;
#endif
}

void wam_reverse_csr_init(WamReverseCsrArtifact *artifact) {
    memset(artifact, 0, sizeof(WamReverseCsrArtifact));
    artifact->index_fd = -1;
    artifact->values_fd = -1;
}

void wam_reverse_csr_close(WamReverseCsrArtifact *artifact) {
    if (artifact->index_fd >= 0) close(artifact->index_fd);
    if (artifact->values_fd >= 0) close(artifact->values_fd);
#ifdef WAM_C_ENABLE_LMDB
    if (artifact->offset_env) {
        if (artifact->offset_dbi_open) {
            mdb_dbi_close(artifact->offset_env, artifact->offset_dbi);
        }
        mdb_env_close(artifact->offset_env);
    }
#endif
    free(artifact->rows);
    wam_reverse_csr_init(artifact);
}

bool wam_reverse_csr_load(WamReverseCsrArtifact *artifact,
                          const char *index_path,
                          const char *values_path) {
    const size_t record_bytes = 16;
    unsigned char *raw = NULL;
    bool drop_after_read = artifact->drop_after_read;
    bool direct_io = artifact->direct_io;
    size_t direct_io_alignment = artifact->direct_io_alignment;
    bool ok = false;
    off_t index_size = 0;

    wam_reverse_csr_close(artifact);
    artifact->drop_after_read = drop_after_read;
    artifact->direct_io = direct_io;
    artifact->direct_io_alignment = direct_io_alignment;
    artifact->index_fd = open(index_path, O_RDONLY);
    if (artifact->index_fd < 0) goto done;
    int values_flags = O_RDONLY;
#ifdef O_DIRECT
    if (artifact->direct_io) values_flags |= O_DIRECT;
#else
    if (artifact->direct_io) goto done;
#endif
    artifact->values_fd = open(values_path, values_flags);
    if (artifact->values_fd < 0) goto done;
    artifact->values_size = lseek(artifact->values_fd, 0, SEEK_END);
    if (artifact->values_size < 0) goto done;

    index_size = lseek(artifact->index_fd, 0, SEEK_END);
    if (index_size < 0 || (index_size % (off_t)record_bytes) != 0) goto done;
    if (lseek(artifact->index_fd, 0, SEEK_SET) < 0) goto done;
    if ((index_size / (off_t)record_bytes) > INT_MAX) goto done;
    artifact->row_count = (int)(index_size / (off_t)record_bytes);
    if ((off_t)artifact->row_count * (off_t)record_bytes != index_size) goto done;

    if (artifact->row_count > 0) {
        artifact->rows = calloc((size_t)artifact->row_count, sizeof(WamReverseCsrRow));
        raw = malloc((size_t)index_size);
        if (!artifact->rows || !raw) goto done;
        if (!wam_pread_exact(artifact->index_fd, raw, (size_t)index_size, 0)) goto done;
        if (artifact->drop_after_read) {
            wam_drop_fd_range(artifact->index_fd, 0, index_size);
        }
        for (int i = 0; i < artifact->row_count; i++) {
            const unsigned char *record = raw + ((size_t)i * record_bytes);
            int parent = (int)wam_read_i32_le(record);
            if (i > 0 && parent <= artifact->rows[i - 1].parent) goto done;
            artifact->rows[i].parent = parent;
            artifact->rows[i].offset_edges = wam_read_u64_le(record + 4);
            artifact->rows[i].child_count = wam_read_u32_le(record + 12);
        }
    }

    ok = true;

done:
    free(raw);
    if (!ok) wam_reverse_csr_close(artifact);
    return ok;
}

bool wam_reverse_csr_load_pread_drop(WamReverseCsrArtifact *artifact,
                                     const char *index_path,
                                     const char *values_path) {
    bool ok = false;
    artifact->drop_after_read = true;
    ok = wam_reverse_csr_load(artifact, index_path, values_path);
    if (ok) artifact->drop_after_read = true;
    return ok;
}

bool wam_reverse_csr_load_direct_io(WamReverseCsrArtifact *artifact,
                                    const char *index_path,
                                    const char *values_path,
                                    int block_size_edges) {
    if (block_size_edges <= 0) return false;
    artifact->direct_io = true;
    artifact->direct_io_alignment = 4096;
    bool ok = wam_reverse_csr_load(artifact, index_path, values_path);
    if (ok) {
        artifact->direct_io = true;
    }
    return ok;
}

bool wam_reverse_csr_load_lmdb_offset(WamReverseCsrArtifact *artifact,
                                      const char *values_path,
                                      const char *offset_env_path,
                                      const char *db_name) {
#ifndef WAM_C_ENABLE_LMDB
    (void)artifact;
    (void)values_path;
    (void)offset_env_path;
    (void)db_name;
    return false;
#else
    MDB_txn *txn = NULL;
    int rc = 0;
    bool drop_after_read = artifact->drop_after_read;
    bool direct_io = artifact->direct_io;
    size_t direct_io_alignment = artifact->direct_io_alignment;
    bool ok = false;

    wam_reverse_csr_close(artifact);
    artifact->drop_after_read = drop_after_read;
    artifact->direct_io = direct_io;
    artifact->direct_io_alignment = direct_io_alignment;
    int values_flags = O_RDONLY;
#ifdef O_DIRECT
    if (artifact->direct_io) values_flags |= O_DIRECT;
#else
    if (artifact->direct_io) goto done;
#endif
    artifact->values_fd = open(values_path, values_flags);
    if (artifact->values_fd < 0) goto done;
    artifact->values_size = lseek(artifact->values_fd, 0, SEEK_END);
    if (artifact->values_size < 0) goto done;

    rc = mdb_env_create(&artifact->offset_env);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_env_set_maxdbs(artifact->offset_env, 2);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_env_open(artifact->offset_env, offset_env_path, MDB_RDONLY, 0664);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_txn_begin(artifact->offset_env, NULL, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_dbi_open(txn, (db_name && db_name[0]) ? db_name : "offsets", 0, &artifact->offset_dbi);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_txn_commit(txn);
    txn = NULL;
    if (rc != MDB_SUCCESS) goto done;

    artifact->offset_dbi_open = true;
    artifact->use_lmdb_offset = true;
    ok = true;

done:
    if (txn) mdb_txn_abort(txn);
    if (!ok) wam_reverse_csr_close(artifact);
    return ok;
#endif
}

bool wam_reverse_csr_load_lmdb_offset_pread_drop(WamReverseCsrArtifact *artifact,
                                                const char *values_path,
                                                const char *offset_env_path,
                                                const char *db_name) {
    bool ok = false;
    artifact->drop_after_read = true;
    ok = wam_reverse_csr_load_lmdb_offset(artifact, values_path, offset_env_path, db_name);
    if (ok) artifact->drop_after_read = true;
    return ok;
}

bool wam_reverse_csr_load_lmdb_offset_direct_io(WamReverseCsrArtifact *artifact,
                                                const char *values_path,
                                                const char *offset_env_path,
                                                const char *db_name,
                                                int block_size_edges) {
    if (block_size_edges <= 0) return false;
    artifact->direct_io = true;
    artifact->direct_io_alignment = 4096;
    bool ok = wam_reverse_csr_load_lmdb_offset(artifact, values_path, offset_env_path, db_name);
    if (ok) {
        artifact->direct_io = true;
    }
    return ok;
}

static WamReverseCsrRow *wam_reverse_csr_find_row(WamReverseCsrArtifact *artifact,
                                                  int parent) {
    int lo = 0;
    int hi = artifact->row_count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int mid_parent = artifact->rows[mid].parent;
        if (mid_parent < parent) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo >= artifact->row_count || artifact->rows[lo].parent != parent) return NULL;
    return &artifact->rows[lo];
}

#ifdef WAM_C_ENABLE_LMDB
static int wam_reverse_csr_find_lmdb_offset(WamReverseCsrArtifact *artifact,
                                            int parent,
                                            uint64_t *offset_edges,
                                            uint32_t *child_count) {
    unsigned char key_bytes[4];
    MDB_txn *txn = NULL;
    MDB_val key;
    MDB_val data;
    int rc = 0;

    key_bytes[0] = (unsigned char)((uint32_t)parent & 0xffU);
    key_bytes[1] = (unsigned char)(((uint32_t)parent >> 8) & 0xffU);
    key_bytes[2] = (unsigned char)(((uint32_t)parent >> 16) & 0xffU);
    key_bytes[3] = (unsigned char)(((uint32_t)parent >> 24) & 0xffU);
    key.mv_size = sizeof(key_bytes);
    key.mv_data = key_bytes;

    rc = mdb_txn_begin(artifact->offset_env, NULL, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) return -1;
    rc = mdb_get(txn, artifact->offset_dbi, &key, &data);
    if (rc == MDB_NOTFOUND) {
        mdb_txn_abort(txn);
        return 0;
    }
    if (rc != MDB_SUCCESS || data.mv_size != 12) {
        mdb_txn_abort(txn);
        return -1;
    }
    *offset_edges = wam_read_u64_le((const unsigned char *)data.mv_data);
    *child_count = wam_read_u32_le(((const unsigned char *)data.mv_data) + 8);
    mdb_txn_abort(txn);
    return 1;
}
#endif

int wam_reverse_csr_lookup_children(WamReverseCsrArtifact *artifact,
                                    int parent,
                                    int *out_children,
                                    int max_children) {
    uint64_t offset_edges = 0;
    uint32_t child_count = 0;

#ifdef WAM_C_ENABLE_LMDB
    if (artifact->use_lmdb_offset) {
        int status = wam_reverse_csr_find_lmdb_offset(artifact, parent, &offset_edges, &child_count);
        if (status <= 0) return status;
    } else
#endif
    {
        WamReverseCsrRow *row = wam_reverse_csr_find_row(artifact, parent);
        if (!row) return 0;
        offset_edges = row->offset_edges;
        child_count = row->child_count;
    }

    if (max_children < 0) max_children = 0;
    if (child_count > (uint32_t)INT_MAX) return -1;
    int total = (int)child_count;
    int to_read = total < max_children ? total : max_children;
    for (int i = 0; i < to_read; i++) {
        unsigned char encoded[4];
        uint64_t edge_offset = offset_edges + (uint64_t)i;
        if (edge_offset > (UINT64_MAX / 4ULL)) return -1;
        if (!wam_direct_read_exact(artifact, encoded, sizeof(encoded), (off_t)(edge_offset * 4ULL))) {
            return -1;
        }
        out_children[i] = (int)wam_read_i32_le(encoded);
    }
    if (artifact->drop_after_read && to_read > 0) {
        uint64_t first_byte = offset_edges * 4ULL;
        uint64_t byte_count = (uint64_t)to_read * 4ULL;
        if (first_byte <= (uint64_t)LLONG_MAX && byte_count <= (uint64_t)LLONG_MAX) {
            wam_drop_fd_range(artifact->values_fd, (off_t)first_byte, (off_t)byte_count);
        }
    }
    return total;
}

void wam_int_results_init(WamIntResults *results) {
    memset(results, 0, sizeof(WamIntResults));
}

void wam_int_results_close(WamIntResults *results) {
    free(results->values);
    memset(results, 0, sizeof(WamIntResults));
}

bool wam_int_results_push(WamIntResults *results, int value) {
    if (results->count >= results->cap) {
        results->cap = results->cap ? results->cap * 2 : WAM_INITIAL_CAP;
        results->values = realloc(results->values, sizeof(int) * results->cap);
        if (!results->values) {
            results->count = 0;
            results->cap = 0;
            return false;
        }
    }
    results->values[results->count++] = value;
    return true;
}

void wam_bidirectional_ancestor_results_init(WamBidirectionalAncestorResults *results) {
    memset(results, 0, sizeof(WamBidirectionalAncestorResults));
}

void wam_bidirectional_ancestor_results_close(WamBidirectionalAncestorResults *results) {
    free(results->values);
    memset(results, 0, sizeof(WamBidirectionalAncestorResults));
}

bool wam_bidirectional_ancestor_results_push(WamBidirectionalAncestorResults *results,
                                             int total_hops,
                                             int parent_hops,
                                             int child_hops) {
    if (results->count >= results->cap) {
        results->cap = results->cap ? results->cap * 2 : WAM_INITIAL_CAP;
        results->values = realloc(results->values,
                                  sizeof(WamBidirectionalAncestorResult) * results->cap);
        if (!results->values) {
            results->count = 0;
            results->cap = 0;
            return false;
        }
    }
    results->values[results->count].total_hops = total_hops;
    results->values[results->count].parent_hops = parent_hops;
    results->values[results->count].child_hops = child_hops;
    results->count++;
    return true;
}

void wam_register_category_ancestor_kernel(WamState *state, const char *pred, int max_depth) {
    state->category_max_depth = max_depth > 0 ? max_depth : 10;
    wam_register_foreign_predicate(state, pred, 4, wam_category_ancestor_handler);
}

void wam_register_bidirectional_ancestor_kernel(WamState *state, const char *pred,
                                                int max_depth,
                                                double parent_step_cost,
                                                double child_step_cost,
                                                double cost_budget) {
    state->category_max_depth = max_depth > 0 ? max_depth : 10;
    state->bidirectional_parent_step_cost =
        parent_step_cost > 0.0 ? parent_step_cost : 1.0;
    state->bidirectional_child_step_cost =
        child_step_cost > 0.0 ? child_step_cost : 3.0;
    state->bidirectional_cost_budget =
        cost_budget > 0.0 ? cost_budget : 10.0;
    wam_register_foreign_predicate(state, pred, 5, wam_bidirectional_ancestor_handler);
}

void wam_register_transitive_closure_kernel(WamState *state, const char *pred) {
    wam_register_foreign_predicate(state, pred, 2, wam_transitive_closure_handler);
}

void wam_register_transitive_distance_kernel(WamState *state, const char *pred) {
    wam_register_foreign_predicate(state, pred, 3, wam_transitive_distance_handler);
}

void wam_register_transitive_parent_distance_kernel(WamState *state, const char *pred) {
    wam_register_foreign_predicate(state, pred, 4, wam_transitive_parent_distance_handler);
}

void wam_register_transitive_step_parent_distance_kernel(WamState *state, const char *pred,
                                                         const char *edge_relation) {
    wam_register_foreign_predicate(state, pred, 5, wam_transitive_step_parent_distance_handler);
    wam_bind_kernel_edge_relation(state, pred, 5, edge_relation);
}

void wam_register_weighted_shortest_path_kernel(WamState *state, const char *pred) {
    wam_register_foreign_predicate(state, pred, 3, wam_weighted_shortest_path_handler);
}

void wam_register_astar_shortest_path_kernel(WamState *state, const char *pred) {
    wam_register_foreign_predicate(state, pred, 4, wam_astar_shortest_path_handler);
}

static bool wam_value_as_atom(WamState *state, WamValue value, const char **out) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag != VAL_ATOM) return false;
    *out = cell->data.atom;
    return true;
}

static bool wam_list_contains_atom(WamState *state, WamValue value, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, &value);
    while (cell->tag == VAL_LIST) {
        WamValue *head = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr]);
        if (head->tag == VAL_ATOM && strcmp(head->data.atom, atom) == 0) return true;
        cell = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr + 1]);
    }
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool wam_list_atoms_to_array(WamState *state,
                                    WamValue value,
                                    const char **out,
                                    int *out_len,
                                    int max_len) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int count = 0;
    while (cell->tag == VAL_LIST) {
        if (count >= max_len) return false;
        WamValue *head = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr]);
        if (head->tag != VAL_ATOM) return false;
        out[count++] = head->data.atom;
        cell = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr + 1]);
    }
    if (cell->tag == VAL_ATOM && strcmp(cell->data.atom, "[]") == 0) {
        *out_len = count;
        return true;
    }
    return false;
}

static bool wam_visited_array_contains(const char **visited, int visited_len, const char *atom) {
    for (int i = 0; i < visited_len; i++) {
        if (strcmp(visited[i], atom) == 0) return true;
    }
    return false;
}

bool wam_category_atom_to_id(WamState *state, const char *atom, int *id_out) {
    int index = -1;
    if (wam_category_id_atom_index_find(state, atom, &index) ||
        ((!state->category_id_by_atom || state->category_id_by_atom_cap <= 0) &&
         (index = wam_category_id_find_atom_linear(state, atom)) >= 0)) {
        *id_out = state->category_ids[index].id;
        return true;
    }
    return false;
}

bool wam_category_id_to_atom(WamState *state, int id, const char **atom_out) {
    int index = -1;
    if (wam_category_id_value_index_find(state, id, &index) ||
        ((!state->category_id_by_value || state->category_id_by_value_cap <= 0) &&
         (index = wam_category_id_find_value_linear(state, id)) >= 0)) {
        *atom_out = state->category_ids[index].atom;
        return true;
    }
    return false;
}

typedef struct {
    const char **keys;
    int *distances;
    int capacity;
    int count;
} WamBidirectionalDistanceMap;

typedef struct WamBidirectionalDistanceCacheEntry {
    const char *root;
    WamBidirectionalDistanceMap distances;
    struct WamBidirectionalDistanceCacheEntry *next;
} WamBidirectionalDistanceCacheEntry;

typedef struct {
    const char **items;
    int count;
    int cap;
} WamBidirectionalAtomQueue;

static uint64_t wam_bidirectional_atom_hash(const char *atom) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)atom;
    while (*p) {
        h ^= (uint64_t)(*p++);
        h *= 1099511628211ULL;
    }
    return h;
}

static void wam_bidirectional_distance_map_init(WamBidirectionalDistanceMap *map) {
    memset(map, 0, sizeof(WamBidirectionalDistanceMap));
}

static void wam_bidirectional_distance_map_close(WamBidirectionalDistanceMap *map) {
    free(map->keys);
    free(map->distances);
    memset(map, 0, sizeof(WamBidirectionalDistanceMap));
}

static int wam_bidirectional_distance_map_slot(const char **keys,
                                               int capacity,
                                               const char *key,
                                               bool *present) {
    uint64_t h = wam_bidirectional_atom_hash(key);
    int mask = capacity - 1;
    int slot = (int)(h & (uint64_t)mask);
    while (keys[slot]) {
        if (strcmp(keys[slot], key) == 0) {
            *present = true;
            return slot;
        }
        slot = (slot + 1) & mask;
    }
    *present = false;
    return slot;
}

static bool wam_bidirectional_distance_map_reserve(WamBidirectionalDistanceMap *map,
                                                   int min_capacity) {
    if (min_capacity < WAM_INITIAL_CAP) min_capacity = WAM_INITIAL_CAP;
    if (map->capacity >= min_capacity) return true;

    int new_capacity = map->capacity ? map->capacity : WAM_INITIAL_CAP;
    while (new_capacity < min_capacity) {
        if (new_capacity > INT_MAX / 2) return false;
        new_capacity *= 2;
    }

    const char **new_keys = calloc((size_t)new_capacity, sizeof(const char *));
    int *new_distances = malloc(sizeof(int) * (size_t)new_capacity);
    if (!new_keys || !new_distances) {
        free(new_keys);
        free(new_distances);
        return false;
    }

    for (int i = 0; i < map->capacity; i++) {
        if (!map->keys[i]) continue;
        bool present = false;
        int slot = wam_bidirectional_distance_map_slot(
            new_keys, new_capacity, map->keys[i], &present);
        new_keys[slot] = map->keys[i];
        new_distances[slot] = map->distances[i];
    }

    free(map->keys);
    free(map->distances);
    map->keys = new_keys;
    map->distances = new_distances;
    map->capacity = new_capacity;
    return true;
}

static bool wam_bidirectional_distance_map_get(WamBidirectionalDistanceMap *map,
                                               const char *key,
                                               int *distance_out) {
    if (map->capacity == 0) return false;
    bool present = false;
    int slot = wam_bidirectional_distance_map_slot(
        map->keys, map->capacity, key, &present);
    if (!present) return false;
    *distance_out = map->distances[slot];
    return true;
}

static bool wam_bidirectional_distance_map_put_if_absent(
        WamBidirectionalDistanceMap *map,
        const char *key,
        int distance,
        bool *inserted) {
    if (map->capacity == 0 || map->count >= map->capacity / 2) {
        int next_capacity = WAM_INITIAL_CAP;
        if (map->capacity) {
            if (map->capacity > INT_MAX / 2) return false;
            next_capacity = map->capacity * 2;
        }
        if (!wam_bidirectional_distance_map_reserve(map, next_capacity)) return false;
    }

    bool present = false;
    int slot = wam_bidirectional_distance_map_slot(
        map->keys, map->capacity, key, &present);
    if (present) {
        *inserted = false;
        return true;
    }
    map->keys[slot] = key;
    map->distances[slot] = distance;
    map->count++;
    *inserted = true;
    return true;
}

static void wam_bidirectional_atom_queue_init(WamBidirectionalAtomQueue *queue) {
    memset(queue, 0, sizeof(WamBidirectionalAtomQueue));
}

static void wam_bidirectional_atom_queue_close(WamBidirectionalAtomQueue *queue) {
    free(queue->items);
    memset(queue, 0, sizeof(WamBidirectionalAtomQueue));
}

static bool wam_bidirectional_atom_queue_push(WamBidirectionalAtomQueue *queue,
                                              const char *atom) {
    if (queue->count >= queue->cap) {
        int next_cap = WAM_INITIAL_CAP;
        if (queue->cap) {
            if (queue->cap > INT_MAX / 2) return false;
            next_cap = queue->cap * 2;
        }
        const char **items = realloc(queue->items,
                                     sizeof(const char *) * (size_t)next_cap);
        if (!items) return false;
        queue->items = items;
        queue->cap = next_cap;
    }
    queue->items[queue->count++] = atom;
    return true;
}

static bool wam_bidirectional_build_min_distances(WamState *state,
                                                  const char *root,
                                                  WamBidirectionalDistanceMap *map) {
    WamBidirectionalAtomQueue queue;
    wam_bidirectional_atom_queue_init(&queue);

    bool inserted = false;
    if (!wam_bidirectional_distance_map_put_if_absent(map, root, 0, &inserted) ||
        !wam_bidirectional_atom_queue_push(&queue, root)) {
        wam_bidirectional_atom_queue_close(&queue);
        return false;
    }

    for (int head = 0; head < queue.count; head++) {
        const char *node = queue.items[head];
        int distance = 0;
        if (!wam_bidirectional_distance_map_get(map, node, &distance)) continue;
        int next_distance = distance + 1;

        if (state->bidirectional_child_csr) {
            int parent_id = 0;
            if (!wam_category_atom_to_id(state, node, &parent_id)) continue;
            int child_count = wam_reverse_csr_lookup_children(
                state->bidirectional_child_csr, parent_id, NULL, 0);
            if (child_count < 0) {
                wam_bidirectional_atom_queue_close(&queue);
                return false;
            }
            if (child_count == 0) continue;
            int *child_ids = malloc(sizeof(int) * (size_t)child_count);
            if (!child_ids) {
                wam_bidirectional_atom_queue_close(&queue);
                return false;
            }
            int read_count = wam_reverse_csr_lookup_children(
                state->bidirectional_child_csr, parent_id, child_ids, child_count);
            if (read_count < 0) {
                free(child_ids);
                wam_bidirectional_atom_queue_close(&queue);
                return false;
            }
            int limit = read_count < child_count ? read_count : child_count;
            for (int i = 0; i < limit; i++) {
                const char *child = NULL;
                if (!wam_category_id_to_atom(state, child_ids[i], &child)) continue;
                bool child_inserted = false;
                if (!wam_bidirectional_distance_map_put_if_absent(
                        map, child, next_distance, &child_inserted)) {
                    free(child_ids);
                    wam_bidirectional_atom_queue_close(&queue);
                    return false;
                }
                if (child_inserted &&
                    !wam_bidirectional_atom_queue_push(&queue, child)) {
                    free(child_ids);
                    wam_bidirectional_atom_queue_close(&queue);
                    return false;
                }
            }
            free(child_ids);
            continue;
        }

        for (int i = 0; i < state->category_edge_count; i++) {
            CategoryEdge *edge = &state->category_edges[i];
            if (strcmp(edge->parent, node) != 0) continue;
            bool child_inserted = false;
            if (!wam_bidirectional_distance_map_put_if_absent(
                    map, edge->child, next_distance, &child_inserted)) {
                wam_bidirectional_atom_queue_close(&queue);
                return false;
            }
            if (child_inserted &&
                !wam_bidirectional_atom_queue_push(&queue, edge->child)) {
                wam_bidirectional_atom_queue_close(&queue);
                return false;
            }
        }
    }

    wam_bidirectional_atom_queue_close(&queue);
    return true;
}

static void wam_bidirectional_distance_cache_clear(WamState *state) {
    WamBidirectionalDistanceCacheEntry *entry =
        (WamBidirectionalDistanceCacheEntry *)state->bidirectional_min_distance_cache;
    while (entry) {
        WamBidirectionalDistanceCacheEntry *next = entry->next;
        wam_bidirectional_distance_map_close(&entry->distances);
        free(entry);
        entry = next;
    }
    state->bidirectional_min_distance_cache = NULL;
}

static WamBidirectionalDistanceMap *wam_bidirectional_get_min_distances(
        WamState *state,
        const char *root) {
    WamBidirectionalDistanceCacheEntry *entry =
        (WamBidirectionalDistanceCacheEntry *)state->bidirectional_min_distance_cache;
    while (entry) {
        if (strcmp(entry->root, root) == 0) {
            return &entry->distances;
        }
        entry = entry->next;
    }

    WamBidirectionalDistanceCacheEntry *new_entry =
        calloc(1, sizeof(WamBidirectionalDistanceCacheEntry));
    if (!new_entry) return NULL;
    new_entry->root = wam_intern_atom(state, root);
    wam_bidirectional_distance_map_init(&new_entry->distances);
    if (!wam_bidirectional_build_min_distances(
            state, new_entry->root, &new_entry->distances)) {
        wam_bidirectional_distance_map_close(&new_entry->distances);
        free(new_entry);
        return NULL;
    }
    new_entry->next =
        (WamBidirectionalDistanceCacheEntry *)state->bidirectional_min_distance_cache;
    state->bidirectional_min_distance_cache = new_entry;
    return &new_entry->distances;
}

static bool wam_bidirectional_can_reach_root_within_budget(
        WamBidirectionalDistanceMap *min_distances,
        const char *node,
        double next_cost,
        double parent_cost,
        double budget) {
    int min_parent_hops = 0;
    if (!wam_bidirectional_distance_map_get(
            min_distances, node, &min_parent_hops)) {
        return false;
    }
    return next_cost + ((double)min_parent_hops * parent_cost) <= budget;
}

static bool wam_category_ancestor_dfs(WamState *state,
                                      const char *cat,
                                      const char *root,
                                      int depth,
                                      int max_depth,
                                      const char **visited,
                                      int visited_len,
                                      WamIntResults *results) {
    bool found = false;
    CategoryEdge *edges = NULL;
    int edge_count = 0;
    if (!wam_state_category_child_range(state, cat, &edges, &edge_count)) return false;
    for (int i = 0; i < edge_count; i++) {
        CategoryEdge *edge = &edges[i];
        if (wam_visited_array_contains(visited, visited_len, edge->parent)) continue;
        if (strcmp(edge->parent, root) == 0) {
            if (!wam_int_results_push(results, depth + 1)) return false;
            found = true;
        }
    }

    if (visited_len >= max_depth || visited_len >= 64) return found;

    for (int i = 0; i < edge_count; i++) {
        CategoryEdge *edge = &edges[i];
        if (wam_visited_array_contains(visited, visited_len, edge->parent)) continue;
        visited[visited_len] = edge->parent;
        if (wam_category_ancestor_dfs(state, edge->parent, root, depth + 1,
                                      max_depth, visited, visited_len + 1,
                                      results)) {
            found = true;
        }
    }
    return found;
}

static bool wam_category_ancestor_inputs(WamState *state,
                                         const char **cat_out,
                                         const char **root_out,
                                         const char **visited,
                                         int *visited_len_out) {
    const char *cat = NULL;
    const char *root = NULL;
    if (!wam_value_as_atom(state, state->A[0], &cat)) return false;
    if (!wam_value_as_atom(state, state->A[1], &root)) return false;
    int visited_len = 0;
    if (!wam_list_atoms_to_array(state, state->A[3], visited, &visited_len, 64)) return false;
    if (wam_list_contains_atom(state, state->A[3], root)) return false;
    if (visited_len == 0) {
        visited[visited_len++] = cat;
    }
    *cat_out = cat;
    *root_out = root;
    *visited_len_out = visited_len;
    return true;
}

bool wam_collect_category_ancestor_hops(WamState *state, WamIntResults *results) {
    const char *cat = NULL;
    const char *root = NULL;
    const char *visited[64];
    int visited_len = 0;
    if (!wam_category_ancestor_inputs(state, &cat, &root, visited, &visited_len)) return false;

    int max_depth = state->category_max_depth > 0 ? state->category_max_depth : 10;
    return wam_category_ancestor_dfs(state, cat, root, 0, max_depth,
                                     visited, visited_len, results);
}

bool wam_category_min_parent_hops(WamState *state,
                                  const char *cat,
                                  const char *root,
                                  int *hops_out) {
    if (!cat || !root || !hops_out) return false;
    if (strcmp(cat, root) == 0) {
        *hops_out = 0;
        return true;
    }
    WamBidirectionalDistanceMap *min_distances =
        wam_bidirectional_get_min_distances(state, root);
    if (!min_distances) return false;
    return wam_bidirectional_distance_map_get(min_distances, cat, hops_out);
}

bool wam_category_child_may_reach_root_within_budget(WamState *state,
                                                     const char *cat,
                                                     const char *root,
                                                     int max_child_expansions,
                                                     int child_depth,
                                                     double parent_cost,
                                                     double child_cost,
                                                     double budget,
                                                     int *candidate_count_out) {
    if (candidate_count_out) *candidate_count_out = 0;
    if (!state || !cat || !root) return true;
    if (max_child_expansions <= 0 || child_depth <= 0) return false;
    if (child_cost > budget) return false;
    if (child_depth != 1) return true;
    if (!state->bidirectional_child_csr) return true;
    WamBidirectionalDistanceMap *min_distances =
        wam_bidirectional_get_min_distances(state, root);
    if (!min_distances) return true;

    int parent_id = 0;
    if (!wam_category_atom_to_id(state, cat, &parent_id)) return false;
    int child_count = wam_reverse_csr_lookup_children(
        state->bidirectional_child_csr, parent_id, NULL, 0);
    if (child_count < 0) return true;
    if (child_count == 0) return false;

    int child_limit = child_count < 256 ? child_count : 256;
    if (child_limit <= 0) return false;
    int *child_ids = malloc(sizeof(int) * (size_t)child_limit);
    if (!child_ids) return true;
    int read_count = wam_reverse_csr_lookup_children(
        state->bidirectional_child_csr, parent_id, child_ids, child_limit);
    if (read_count < 0) {
        free(child_ids);
        return true;
    }

    int candidate_count = 0;
    int limit = read_count < child_limit ? read_count : child_limit;
    for (int i = 0; i < limit; i++) {
        const char *child = NULL;
        int min_parent_hops = 0;
        if (!wam_category_id_to_atom(state, child_ids[i], &child)) continue;
        if (!wam_bidirectional_distance_map_get(
                min_distances, child, &min_parent_hops)) {
            continue;
        }
        double total_cost = child_cost + ((double)min_parent_hops * parent_cost);
        if (total_cost <= budget) {
            candidate_count++;
        }
    }
    free(child_ids);
    if (candidate_count_out) *candidate_count_out = candidate_count;
    return candidate_count > 0;
}

bool wam_category_ancestor_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 4) return false;

    WamIntResults results;
    wam_int_results_init(&results);
    if (!wam_collect_category_ancestor_hops(state, &results) || results.count == 0) {
        wam_int_results_close(&results);
        return false;
    }

    WamValue result = val_int(results.values[0]);
    bool ok = wam_unify(state, &state->A[2], &result);
    wam_int_results_close(&results);
    return ok;
}

static bool wam_bidirectional_ancestor_dfs(WamState *state,
                                           const char *node,
                                           const char *root,
                                           int depth,
                                           int max_depth,
                                           const char **visited,
                                           int visited_len,
                                           int parent_hops,
                                           int child_hops,
                                           double cost,
                                           WamBidirectionalDistanceMap *min_distances,
                                           WamBidirectionalAncestorResults *results);

static bool wam_bidirectional_ancestor_step_child(
        WamState *state,
        const char *child,
        const char *root,
        int depth,
        int max_depth,
        const char **visited,
        int visited_len,
        int parent_hops,
        int child_hops,
        double cost,
        double parent_cost,
        double child_cost,
        double budget,
        WamBidirectionalDistanceMap *min_distances,
        WamBidirectionalAncestorResults *results,
        bool *found) {
    if (wam_visited_array_contains(visited, visited_len, child)) return true;
    double next_cost = cost + child_cost;
    if (next_cost > budget) return true;
    if (!wam_bidirectional_can_reach_root_within_budget(
            min_distances, child, next_cost, parent_cost, budget)) {
        return true;
    }
    int next_child_hops = child_hops + 1;
    if (strcmp(child, root) == 0) {
        if (!wam_bidirectional_ancestor_results_push(
                results, parent_hops + next_child_hops,
                parent_hops, next_child_hops)) {
            return false;
        }
        *found = true;
        return true;
    }
    if (depth + 1 >= max_depth || visited_len >= 64) return true;
    visited[visited_len] = child;
    if (wam_bidirectional_ancestor_dfs(state, child, root,
                                       depth + 1, max_depth,
                                       visited, visited_len + 1,
                                       parent_hops, next_child_hops,
                                       next_cost, min_distances, results)) {
        *found = true;
    }
    return true;
}

static bool wam_bidirectional_ancestor_expand_children(
        WamState *state,
        const char *node,
        const char *root,
        int depth,
        int max_depth,
        const char **visited,
        int visited_len,
        int parent_hops,
        int child_hops,
        double cost,
        double parent_cost,
        double child_cost,
        double budget,
        WamBidirectionalDistanceMap *min_distances,
        WamBidirectionalAncestorResults *results,
        bool *found) {
    if (state->bidirectional_child_csr) {
        int parent_id = 0;
        if (!wam_category_atom_to_id(state, node, &parent_id)) return true;
        int child_ids[256];
        int child_count = wam_reverse_csr_lookup_children(
            state->bidirectional_child_csr, parent_id, child_ids, 256);
        if (child_count < 0) return false;
        int limit = child_count < 256 ? child_count : 256;
        for (int i = 0; i < limit; i++) {
            const char *child = NULL;
            if (!wam_category_id_to_atom(state, child_ids[i], &child)) continue;
            if (!wam_bidirectional_ancestor_step_child(
                    state, child, root, depth, max_depth, visited, visited_len,
                    parent_hops, child_hops, cost, parent_cost, child_cost,
                    budget, min_distances, results, found)) {
                return false;
            }
        }
        return true;
    }

    for (int i = 0; i < state->category_edge_count; i++) {
        CategoryEdge *edge = &state->category_edges[i];
        if (strcmp(edge->parent, node) != 0) continue;
        if (!wam_bidirectional_ancestor_step_child(
                state, edge->child, root, depth, max_depth, visited, visited_len,
                parent_hops, child_hops, cost, parent_cost, child_cost,
                budget, min_distances, results, found)) {
            return false;
        }
    }
    return true;
}

static bool wam_bidirectional_ancestor_dfs(WamState *state,
                                           const char *node,
                                           const char *root,
                                           int depth,
                                           int max_depth,
                                           const char **visited,
                                           int visited_len,
                                           int parent_hops,
                                           int child_hops,
                                           double cost,
                                           WamBidirectionalDistanceMap *min_distances,
                                           WamBidirectionalAncestorResults *results) {
    bool found = false;
    double parent_cost = state->bidirectional_parent_step_cost > 0.0
        ? state->bidirectional_parent_step_cost
        : 1.0;
    double child_cost = state->bidirectional_child_step_cost > 0.0
        ? state->bidirectional_child_step_cost
        : 3.0;
    double budget = state->bidirectional_cost_budget > 0.0
        ? state->bidirectional_cost_budget
        : 10.0;

    CategoryEdge *edges = NULL;
    int edge_count = 0;
    if (!wam_state_category_child_range(state, node, &edges, &edge_count)) return false;
    for (int i = 0; i < edge_count; i++) {
        CategoryEdge *edge = &edges[i];
        if (wam_visited_array_contains(visited, visited_len, edge->parent)) continue;
        double next_cost = cost + parent_cost;
        if (next_cost > budget) continue;
        if (!wam_bidirectional_can_reach_root_within_budget(
                min_distances, edge->parent, next_cost, parent_cost, budget)) {
            continue;
        }
        int next_parent_hops = parent_hops + 1;
        if (strcmp(edge->parent, root) == 0) {
            if (!wam_bidirectional_ancestor_results_push(
                    results, next_parent_hops + child_hops,
                    next_parent_hops, child_hops)) {
                return false;
            }
            found = true;
            continue;
        }
        if (depth + 1 >= max_depth || visited_len >= 64) continue;
        visited[visited_len] = edge->parent;
        if (wam_bidirectional_ancestor_dfs(state, edge->parent, root,
                                           depth + 1, max_depth,
                                           visited, visited_len + 1,
                                           next_parent_hops, child_hops,
                                           next_cost, min_distances, results)) {
            found = true;
        }
    }

    if (!wam_bidirectional_ancestor_expand_children(
            state, node, root, depth, max_depth, visited, visited_len,
            parent_hops, child_hops, cost, parent_cost, child_cost, budget,
            min_distances, results, &found)) {
        return false;
    }

    return found;
}

bool wam_collect_bidirectional_ancestor_hops(WamState *state,
                                             WamBidirectionalAncestorResults *results) {
    const char *cat = NULL;
    const char *root = NULL;
    const char *visited[64];
    int visited_len = 0;
    if (!wam_value_as_atom(state, state->A[0], &cat)) return false;
    if (!wam_value_as_atom(state, state->A[1], &root)) return false;
    if (strcmp(cat, root) == 0) {
        return wam_bidirectional_ancestor_results_push(results, 0, 0, 0);
    }

    WamBidirectionalDistanceMap *min_distances =
        wam_bidirectional_get_min_distances(state, root);
    if (!min_distances) {
        return false;
    }

    visited[visited_len++] = cat;
    int max_depth = state->category_max_depth > 0 ? state->category_max_depth : 10;
    bool ok = wam_bidirectional_ancestor_dfs(state, cat, root, 0, max_depth,
                                             visited, visited_len, 0, 0, 0.0,
                                             min_distances, results);
    return ok;
}

static bool wam_unify_bidirectional_ancestor_result(
        WamState *state,
        WamBidirectionalAncestorResult *result) {
    WamValue total_value = val_int(result->total_hops);
    WamValue parent_value = val_int(result->parent_hops);
    WamValue child_value = val_int(result->child_hops);
    return wam_unify(state, &state->A[2], &total_value) &&
           wam_unify(state, &state->A[3], &parent_value) &&
           wam_unify(state, &state->A[4], &child_value);
}

bool wam_bidirectional_ancestor_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 5) return false;

    WamBidirectionalAncestorResults results;
    wam_bidirectional_ancestor_results_init(&results);
    if (!wam_collect_bidirectional_ancestor_hops(state, &results) || results.count == 0) {
        wam_bidirectional_ancestor_results_close(&results);
        return false;
    }

    bool ok = false;
    for (int i = 0; i < results.count; i++) {
        if (wam_unify_bidirectional_ancestor_result(state, &results.values[i])) {
            ok = true;
            break;
        }
    }
    wam_bidirectional_ancestor_results_close(&results);
    return ok;
}

static bool wam_collect_transitive_closure(WamState *state,
                                           const char *start,
                                           WamValue **results_out,
                                           int *result_count_out) {
    int capacity = state->category_edge_count;
    if (capacity <= 0 || capacity == INT_MAX) return false;

    const char **visited = malloc(sizeof(const char *) * (size_t)capacity);
    const char **queue = malloc(sizeof(const char *) * (size_t)(capacity + 1));
    WamValue *results = malloc(sizeof(WamValue) * (size_t)capacity);
    if (!visited || !queue || !results) {
        free(visited);
        free(queue);
        free(results);
        return false;
    }

    int visited_len = 0;
    int head = 0;
    int tail = 0;
    int result_count = 0;
    queue[tail++] = start;

    while (head < tail) {
        const char *node = queue[head++];
        for (int i = 0; i < state->category_edge_count; i++) {
            CategoryEdge *edge = &state->category_edges[i];
            if (strcmp(edge->child, node) != 0) continue;
            if (wam_visited_array_contains(visited, visited_len, edge->parent)) continue;

            /* Strict R+: nodes enter visited/results only after traversing
             * an edge. Incremental insertion also suppresses duplicate
             * facts and multiple paths before they reach the stream. */
            visited[visited_len++] = edge->parent;
            queue[tail++] = edge->parent;
            results[result_count++] = val_atom(edge->parent);
        }
    }

    free(visited);
    free(queue);
    if (result_count == 0) {
        free(results);
        return false;
    }
    *results_out = results;
    *result_count_out = result_count;
    return true;
}

static bool wam_bind_foreign_atom_stream(WamState *state,
                                         int result_reg,
                                         WamValue *results,
                                         int result_count,
                                         int resume_pc) {
    WamValue first = results[0];
    if (result_count > 1) {
        push_choice_point(state, WAM_FOREIGN_STREAM_NEXT, result_reg + 1);
        ChoicePoint *cp = &state->B_array[state->B - 1];
        cp->foreign_results = results;
        cp->foreign_result_count = result_count;
        cp->foreign_result_index = 1;
        cp->foreign_result_reg = result_reg;
        cp->foreign_resume_pc = resume_pc;
    } else {
        free(results);
    }

    if (wam_unify(state, &state->A[result_reg], &first)) return true;
    if (result_count > 1) pop_choice_point(state);
    return false;
}

static bool wam_try_bind_foreign_triple(WamState *state,
                                        WamValue target_v,
                                        WamValue parent_v,
                                        WamValue dist_v) {
    /* A tuple is one logical candidate.  In particular, A2/A3/A4 can
     * alias the same WAM variable, so independently successful column
     * binds are not enough.  Roll the complete candidate back on the
     * first conflict before trying another tuple (or reporting failure). */
    int trail_mark = state->TR;
    if (wam_unify(state, &state->A[1], &target_v) &&
        wam_unify(state, &state->A[2], &parent_v) &&
        wam_unify(state, &state->A[3], &dist_v)) {
        return true;
    }
    unwind_trail(state, trail_mark);
    return false;
}

static bool wam_try_bind_foreign_quad(WamState *state,
                                      WamValue target_v,
                                      WamValue step_v,
                                      WamValue parent_v,
                                      WamValue dist_v) {
    /* Quad candidate for TSPD5: A[1..4] = Target, Step, Parent, Distance.
     * Alias-safe: roll back the whole candidate on the first conflict. */
    int trail_mark = state->TR;
    if (wam_unify(state, &state->A[1], &target_v) &&
        wam_unify(state, &state->A[2], &step_v) &&
        wam_unify(state, &state->A[3], &parent_v) &&
        wam_unify(state, &state->A[4], &dist_v)) {
        return true;
    }
    unwind_trail(state, trail_mark);
    return false;
}

static bool wam_resume_foreign_stream(WamState *state) {
    if (state->B <= 0) return false;
    ChoicePoint *cp = &state->B_array[state->B - 1];
    if (cp->next_pc != WAM_FOREIGN_STREAM_NEXT ||
        !cp->foreign_results ||
        cp->foreign_result_index < 0 ||
        cp->foreign_result_index >= cp->foreign_result_count) {
        if (cp->next_pc == WAM_FOREIGN_STREAM_NEXT) pop_choice_point(state);
        return false;
    }

    int result_reg = cp->foreign_result_reg;
    int resume_pc = cp->foreign_resume_pc;
    /* Paired stream (result_reg == 255): interleaved [atom, int, ...]. */
    if (result_reg == 255) {
        int index = cp->foreign_result_index;
        if (index + 1 >= cp->foreign_result_count) {
            pop_choice_point(state);
            return false;
        }
        WamValue atom_v = cp->foreign_results[index];
        WamValue dist_v = cp->foreign_results[index + 1];
        cp->foreign_result_index = index + 2;
        if (cp->foreign_result_index >= cp->foreign_result_count) {
            pop_choice_point(state);
        }
        if (!wam_unify(state, &state->A[1], &atom_v)) return false;
        if (!wam_unify(state, &state->A[2], &dist_v)) return false;
        state->P = resume_pc;
        return true;
    }
    /* Triple stream (result_reg == 254): interleaved [atom, atom, int, ...].
     * A retry may encounter candidates incompatible only because output
     * registers alias.  Scan transactionally until one complete tuple
     * unifies; an incompatible tuple must neither leak bindings nor end the
     * stream while later compatible tuples remain. */
    if (result_reg == 254) {
        while (cp->foreign_result_index + 2 < cp->foreign_result_count) {
            int index = cp->foreign_result_index;
            WamValue target_v = cp->foreign_results[index];
            WamValue parent_v = cp->foreign_results[index + 1];
            WamValue dist_v = cp->foreign_results[index + 2];
            cp->foreign_result_index = index + 3;
            if (!wam_try_bind_foreign_triple(state, target_v, parent_v,
                                             dist_v)) {
                continue;
            }
            if (cp->foreign_result_index >= cp->foreign_result_count) {
                pop_choice_point(state);
            }
            state->P = resume_pc;
            return true;
        }
        pop_choice_point(state);
        return false;
    }
    /* Quad stream (result_reg == 253): interleaved
     * [atom, atom, atom, int, ...] = Target, Step, Parent, Distance.
     * Alias-safe transactional scan, same protocol as the triple stream. */
    if (result_reg == 253) {
        while (cp->foreign_result_index + 3 < cp->foreign_result_count) {
            int index = cp->foreign_result_index;
            WamValue target_v = cp->foreign_results[index];
            WamValue step_v = cp->foreign_results[index + 1];
            WamValue parent_v = cp->foreign_results[index + 2];
            WamValue dist_v = cp->foreign_results[index + 3];
            cp->foreign_result_index = index + 4;
            if (!wam_try_bind_foreign_quad(state, target_v, step_v, parent_v,
                                           dist_v)) {
                continue;
            }
            if (cp->foreign_result_index >= cp->foreign_result_count) {
                pop_choice_point(state);
            }
            state->P = resume_pc;
            return true;
        }
        pop_choice_point(state);
        return false;
    }

    int index = cp->foreign_result_index++;
    WamValue result = cp->foreign_results[index];
    if (cp->foreign_result_index >= cp->foreign_result_count) {
        pop_choice_point(state);
    }
    if (!wam_unify(state, &state->A[result_reg], &result)) return false;
    state->P = resume_pc;
    return true;
}

static bool wam_bind_foreign_pair_stream(WamState *state,
                                         WamValue *results,
                                         int value_count,
                                         int resume_pc) {
    /* results: interleaved [atom, int, ...] with value_count = 2 * pairs. */
    if (value_count < 2) {
        free(results);
        return false;
    }
    WamValue first_atom = results[0];
    WamValue first_dist = results[1];
    if (value_count > 2) {
        push_choice_point(state, WAM_FOREIGN_STREAM_NEXT, 255);
        ChoicePoint *cp = &state->B_array[state->B - 1];
        cp->foreign_results = results;
        cp->foreign_result_count = value_count;
        cp->foreign_result_index = 2;
        cp->foreign_result_reg = 255;
        cp->foreign_resume_pc = resume_pc;
    } else {
        free(results);
    }
    if (!wam_unify(state, &state->A[1], &first_atom)) {
        if (value_count > 2) pop_choice_point(state);
        return false;
    }
    if (!wam_unify(state, &state->A[2], &first_dist)) {
        if (value_count > 2) pop_choice_point(state);
        return false;
    }
    return true;
}

static bool wam_bind_foreign_triple_stream(WamState *state,
                                           WamValue *results,
                                           int value_count,
                                           int resume_pc) {
    /* results: interleaved [atom, atom, int, ...] with value_count = 3 * triples. */
    if (value_count < 3) {
        free(results);
        return false;
    }
    for (int index = 0; index + 2 < value_count; index += 3) {
        WamValue target_v = results[index];
        WamValue parent_v = results[index + 1];
        WamValue dist_v = results[index + 2];
        int trail_mark = state->TR;
        if (!wam_try_bind_foreign_triple(state, target_v, parent_v, dist_v)) {
            continue;
        }

        if (index + 3 < value_count) {
            /* The choice-point snapshot must describe the state before this
             * first successful tuple.  Rewind the trial bind, take the
             * ordinary foreign-stream snapshot, then commit the same tuple. */
            unwind_trail(state, trail_mark);
            push_choice_point(state, WAM_FOREIGN_STREAM_NEXT, 4);
            ChoicePoint *cp = &state->B_array[state->B - 1];
            cp->foreign_results = results;
            cp->foreign_result_count = value_count;
            cp->foreign_result_index = index + 3;
            cp->foreign_result_reg = 254;
            cp->foreign_resume_pc = resume_pc;
            if (!wam_try_bind_foreign_triple(state, target_v, parent_v,
                                             dist_v)) {
                pop_choice_point(state);
                return false;
            }
        } else {
            free(results);
        }
        return true;
    }

    free(results);
    return false;
}

static bool wam_bind_foreign_quad_stream(WamState *state,
                                         WamValue *results,
                                         int value_count,
                                         int resume_pc) {
    /* results: interleaved [atom, atom, atom, int, ...]
     * with value_count = 4 * quads. */
    if (value_count < 4) {
        free(results);
        return false;
    }
    for (int index = 0; index + 3 < value_count; index += 4) {
        WamValue target_v = results[index];
        WamValue step_v = results[index + 1];
        WamValue parent_v = results[index + 2];
        WamValue dist_v = results[index + 3];
        int trail_mark = state->TR;
        if (!wam_try_bind_foreign_quad(state, target_v, step_v, parent_v,
                                       dist_v)) {
            continue;
        }

        if (index + 4 < value_count) {
            unwind_trail(state, trail_mark);
            push_choice_point(state, WAM_FOREIGN_STREAM_NEXT, 5);
            ChoicePoint *cp = &state->B_array[state->B - 1];
            cp->foreign_results = results;
            cp->foreign_result_count = value_count;
            cp->foreign_result_index = index + 4;
            cp->foreign_result_reg = 253;
            cp->foreign_resume_pc = resume_pc;
            if (!wam_try_bind_foreign_quad(state, target_v, step_v, parent_v,
                                           dist_v)) {
                pop_choice_point(state);
                return false;
            }
        } else {
            free(results);
        }
        return true;
    }

    free(results);
    return false;
}

bool wam_transitive_closure_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 2) return false;

    const char *start = NULL;
    if (!wam_value_as_atom(state, state->A[0], &start)) return false;

    WamValue *target_cell = wam_deref_ptr(state, &state->A[1]);
    const char *target = NULL;
    if (target_cell->tag == VAL_ATOM) {
        target = target_cell->data.atom;
    } else if (!val_is_unbound(*target_cell)) {
        return false;
    }

    WamValue *results = NULL;
    int result_count = 0;
    if (!wam_collect_transitive_closure(state, start, &results, &result_count)) {
        return false;
    }

    if (target) {
        bool found = false;
        for (int i = 0; i < result_count; i++) {
            if (strcmp(results[i].data.atom, target) == 0) {
                found = true;
                break;
            }
        }
        free(results);
        return found;
    }

    /* The ordinary C choice-point stack owns remaining results. This keeps
     * tc(+Source,-Target) stream-valued without a parallel retry protocol. */
    return wam_bind_foreign_atom_stream(state, 1, results, result_count,
                                        state->P + 1);
}

static bool wam_collect_transitive_distance(WamState *state,
                                            const char *start,
                                            const char *target_filter,
                                            int *distance_filter,
                                            WamValue **results_out,
                                            int *value_count_out) {
    /* dist+ (docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md): BFS;
     * visited not seeded with start. Results interleaved [atom, int].
     * Inline kept (matches surrounding wam_*_handler / collect style). */
    int capacity = state->category_edge_count;
    if (capacity <= 0 || capacity == INT_MAX) return false;

    const char **visited = malloc(sizeof(const char *) * (size_t)capacity);
    const char **queue_nodes = malloc(sizeof(const char *) * (size_t)(capacity + 1));
    int *queue_dists = malloc(sizeof(int) * (size_t)(capacity + 1));
    WamValue *results = malloc(sizeof(WamValue) * (size_t)capacity * 2);
    if (!visited || !queue_nodes || !queue_dists || !results) {
        free(visited);
        free(queue_nodes);
        free(queue_dists);
        free(results);
        return false;
    }

    int visited_len = 0;
    int head = 0;
    int tail = 0;
    int value_count = 0;
    queue_nodes[tail] = start;
    queue_dists[tail++] = 0;

    while (head < tail) {
        const char *node = queue_nodes[head];
        int distance = queue_dists[head++];
        for (int i = 0; i < state->category_edge_count; i++) {
            CategoryEdge *edge = &state->category_edges[i];
            if (strcmp(edge->child, node) != 0) continue;
            if (wam_visited_array_contains(visited, visited_len, edge->parent)) continue;
            int next_distance = distance + 1;
            visited[visited_len++] = edge->parent;
            queue_nodes[tail] = edge->parent;
            queue_dists[tail++] = next_distance;
            if (target_filter && strcmp(edge->parent, target_filter) != 0) continue;
            if (distance_filter && next_distance != *distance_filter) continue;
            results[value_count++] = val_atom(edge->parent);
            results[value_count++] = val_int(next_distance);
        }
    }

    free(visited);
    free(queue_nodes);
    free(queue_dists);
    if (value_count == 0) {
        free(results);
        return false;
    }
    *results_out = results;
    *value_count_out = value_count;
    return true;
}

bool wam_transitive_distance_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 3) return false;

    const char *start = NULL;
    if (!wam_value_as_atom(state, state->A[0], &start)) return false;

    WamValue *target_cell = wam_deref_ptr(state, &state->A[1]);
    const char *target = NULL;
    if (target_cell->tag == VAL_ATOM) {
        target = target_cell->data.atom;
    } else if (!val_is_unbound(*target_cell)) {
        return false;
    }

    WamValue *distance_cell = wam_deref_ptr(state, &state->A[2]);
    int distance_filter_value = 0;
    int *distance_filter = NULL;
    if (distance_cell->tag == VAL_INT) {
        distance_filter_value = distance_cell->data.integer;
        if (distance_filter_value <= 0) return false;
        distance_filter = &distance_filter_value;
    } else if (!val_is_unbound(*distance_cell)) {
        return false;
    }

    WamValue *results = NULL;
    int value_count = 0;
    if (!wam_collect_transitive_distance(state, start, target, distance_filter,
                                         &results, &value_count)) {
        return false;
    }

    /* Bound Target (+ optional Distance): succeed once. Bind Distance
     * when unbound; both-bound already filtered to an exact match. */
    if (target) {
        if (value_count < 2) {
            free(results);
            return false;
        }
        WamValue dist_v = results[1];
        free(results);
        if (distance_filter) return true;
        return wam_unify(state, &state->A[2], &dist_v);
    }

    return wam_bind_foreign_pair_stream(state, results, value_count,
                                        state->P + 1);
}

static bool wam_collect_transitive_parent_distance(WamState *state,
                                                   const char *start,
                                                   const char *target_filter,
                                                   const char *parent_filter,
                                                   int *distance_filter,
                                                   WamValue **results_out,
                                                   int *value_count_out) {
    /* Shortest-positive parents
     * (docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md): BFS with
     * parent sets; dist not seeded with start. Results interleaved
     * [atom, atom, int]. Inline kept (matches surrounding collect style). */
    int capacity = state->category_edge_count;
    if (capacity <= 0 || capacity == INT_MAX) return false;

    const char **queue_nodes = malloc(sizeof(const char *) * (size_t)(capacity + 1));
    int *queue_dists = malloc(sizeof(int) * (size_t)(capacity + 1));
    const char **dist_nodes = malloc(sizeof(const char *) * (size_t)(capacity + 1));
    int *dist_vals = malloc(sizeof(int) * (size_t)(capacity + 1));
    /* Parent pairs: (target_idx_in_dist_nodes, parent_atom) — up to E. */
    const char **parent_targets = malloc(sizeof(const char *) * (size_t)capacity);
    const char **parent_preds = malloc(sizeof(const char *) * (size_t)capacity);
    WamValue *results = malloc(sizeof(WamValue) * (size_t)capacity * 3);
    if (!queue_nodes || !queue_dists || !dist_nodes || !dist_vals ||
        !parent_targets || !parent_preds || !results) {
        free(queue_nodes);
        free(queue_dists);
        free(dist_nodes);
        free(dist_vals);
        free(parent_targets);
        free(parent_preds);
        free(results);
        return false;
    }

    int dist_len = 0;
    int parent_len = 0;
    int head = 0;
    int tail = 0;
    int value_count = 0;
    queue_nodes[tail] = start;
    queue_dists[tail++] = 0;

    while (head < tail) {
        const char *node = queue_nodes[head];
        int distance = queue_dists[head++];
        int next_distance = distance + 1;
        for (int i = 0; i < state->category_edge_count; i++) {
            CategoryEdge *edge = &state->category_edges[i];
            if (strcmp(edge->child, node) != 0) continue;
            const char *next = edge->parent;
            int existing = -1;
            for (int j = 0; j < dist_len; j++) {
                if (strcmp(dist_nodes[j], next) == 0) {
                    existing = j;
                    break;
                }
            }
            if (existing < 0) {
                if (dist_len >= capacity + 1 || tail >= capacity + 1 ||
                    parent_len >= capacity) {
                    free(queue_nodes); free(queue_dists); free(dist_nodes);
                    free(dist_vals); free(parent_targets); free(parent_preds);
                    free(results);
                    return false;
                }
                dist_nodes[dist_len] = next;
                dist_vals[dist_len] = next_distance;
                dist_len++;
                parent_targets[parent_len] = next;
                parent_preds[parent_len] = node;
                parent_len++;
                queue_nodes[tail] = next;
                queue_dists[tail++] = next_distance;
            } else if (dist_vals[existing] == next_distance) {
                bool have = false;
                for (int j = 0; j < parent_len; j++) {
                    if (strcmp(parent_targets[j], next) == 0 &&
                        strcmp(parent_preds[j], node) == 0) {
                        have = true;
                        break;
                    }
                }
                if (!have) {
                    if (parent_len >= capacity) {
                        free(queue_nodes); free(queue_dists); free(dist_nodes);
                        free(dist_vals); free(parent_targets); free(parent_preds);
                        free(results);
                        return false;
                    }
                    parent_targets[parent_len] = next;
                    parent_preds[parent_len] = node;
                    parent_len++;
                }
            }
        }
    }

    for (int i = 0; i < parent_len; i++) {
        const char *t = parent_targets[i];
        const char *p = parent_preds[i];
        int d = 0;
        for (int j = 0; j < dist_len; j++) {
            if (strcmp(dist_nodes[j], t) == 0) {
                d = dist_vals[j];
                break;
            }
        }
        if (target_filter && strcmp(t, target_filter) != 0) continue;
        if (parent_filter && strcmp(p, parent_filter) != 0) continue;
        if (distance_filter && d != *distance_filter) continue;
        results[value_count++] = val_atom(t);
        results[value_count++] = val_atom(p);
        results[value_count++] = val_int(d);
    }

    free(queue_nodes);
    free(queue_dists);
    free(dist_nodes);
    free(dist_vals);
    free(parent_targets);
    free(parent_preds);
    if (value_count == 0) {
        free(results);
        return false;
    }
    *results_out = results;
    *value_count_out = value_count;
    return true;
}

bool wam_transitive_parent_distance_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 4) return false;

    const char *start = NULL;
    if (!wam_value_as_atom(state, state->A[0], &start)) return false;

    WamValue *target_cell = wam_deref_ptr(state, &state->A[1]);
    const char *target = NULL;
    if (target_cell->tag == VAL_ATOM) {
        target = target_cell->data.atom;
    } else if (!val_is_unbound(*target_cell)) {
        return false;
    }

    WamValue *parent_cell = wam_deref_ptr(state, &state->A[2]);
    const char *parent = NULL;
    if (parent_cell->tag == VAL_ATOM) {
        parent = parent_cell->data.atom;
    } else if (!val_is_unbound(*parent_cell)) {
        return false;
    }

    WamValue *distance_cell = wam_deref_ptr(state, &state->A[3]);
    int distance_filter_value = 0;
    int *distance_filter = NULL;
    if (distance_cell->tag == VAL_INT) {
        distance_filter_value = distance_cell->data.integer;
        if (distance_filter_value <= 0) return false;
        distance_filter = &distance_filter_value;
    } else if (!val_is_unbound(*distance_cell)) {
        return false;
    }

    WamValue *results = NULL;
    int value_count = 0;
    if (!wam_collect_transitive_parent_distance(state, start, target, parent,
                                                distance_filter,
                                                &results, &value_count)) {
        return false;
    }

    /* Fully bound Target+Parent (+ optional Distance): succeed once.
     * Bind Distance when unbound; both-bound already filtered. */
    if (target && parent) {
        if (value_count < 3) {
            free(results);
            return false;
        }
        WamValue dist_v = results[2];
        free(results);
        if (distance_filter) return true;
        return wam_unify(state, &state->A[3], &dist_v);
    }

    return wam_bind_foreign_triple_stream(state, results, value_count,
                                          state->P + 1);
}

static int wam_count_relation_edges(WamState *state, const char *relation) {
    int n = 0;
    for (int i = 0; i < state->relation_edge_count; i++) {
        if (strcmp(state->relation_edges[i].relation, relation) == 0) n++;
    }
    return n;
}

static bool wam_tspd5_ensure_pair_cap(const char ***pair_targets,
                                      const char ***pair_steps,
                                      const char ***pair_parents,
                                      int *pair_cap,
                                      int needed) {
    if (needed <= *pair_cap) return true;
    int new_cap = *pair_cap ? *pair_cap : 8;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) {
            new_cap = needed;
            break;
        }
        new_cap *= 2;
    }
    const char **targets = realloc(*pair_targets,
                                   sizeof(const char *) * (size_t)new_cap);
    if (!targets) return false;
    *pair_targets = targets;
    const char **steps = realloc(*pair_steps,
                                 sizeof(const char *) * (size_t)new_cap);
    if (!steps) return false;
    *pair_steps = steps;
    const char **parents = realloc(*pair_parents,
                                   sizeof(const char *) * (size_t)new_cap);
    if (!parents) return false;
    *pair_parents = parents;
    *pair_cap = new_cap;
    return true;
}

static bool wam_tspd5_ensure_result_cap(WamValue **results, int *result_cap,
                                        int needed) {
    if (needed <= *result_cap) return true;
    int new_cap = *result_cap ? *result_cap : 16;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) {
            new_cap = needed;
            break;
        }
        new_cap *= 2;
    }
    WamValue *grown = realloc(*results, sizeof(WamValue) * (size_t)new_cap);
    if (!grown) return false;
    *results = grown;
    *result_cap = new_cap;
    return true;
}

static bool wam_collect_transitive_step_parent_distance(
    WamState *state,
    const char *relation,
    const char *start,
    const char *target_filter,
    const char *step_filter,
    const char *parent_filter,
    int *distance_filter,
    WamValue **results_out,
    int *value_count_out) {
    /* Shortest-positive correlated step/parent
     * (docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md):
     * FIFO BFS seed (Source, 0); dist NOT seeded with Source; store
     * correlated (Step, Parent) pairs per Target. */
    int edge_n = wam_count_relation_edges(state, relation);
    if (edge_n <= 0 || edge_n == INT_MAX) return false;

    int node_cap = edge_n + 1;
    const char **queue_nodes = malloc(sizeof(const char *) * (size_t)node_cap);
    int *queue_dists = malloc(sizeof(int) * (size_t)node_cap);
    const char **dist_nodes = malloc(sizeof(const char *) * (size_t)node_cap);
    int *dist_vals = malloc(sizeof(int) * (size_t)node_cap);
    const char **pair_targets = NULL;
    const char **pair_steps = NULL;
    const char **pair_parents = NULL;
    int pair_cap = 0;
    int pair_len = 0;
    int result_cap = edge_n * 4;
    if (result_cap < 4) result_cap = 4;
    WamValue *results = malloc(sizeof(WamValue) * (size_t)result_cap);
    if (!queue_nodes || !queue_dists || !dist_nodes || !dist_vals || !results) {
        free(queue_nodes);
        free(queue_dists);
        free(dist_nodes);
        free(dist_vals);
        free(results);
        return false;
    }
    if (!wam_tspd5_ensure_pair_cap(&pair_targets, &pair_steps, &pair_parents,
                                   &pair_cap, edge_n > 0 ? edge_n : 8)) {
        free(queue_nodes);
        free(queue_dists);
        free(dist_nodes);
        free(dist_vals);
        free(results);
        return false;
    }

    int dist_len = 0;
    int head = 0;
    int tail = 0;
    queue_nodes[tail] = start;
    queue_dists[tail++] = 0;

    while (head < tail) {
        const char *node = queue_nodes[head];
        int distance = queue_dists[head++];
        int next_distance = distance + 1;
        for (int i = 0; i < state->relation_edge_count; i++) {
            RelationEdge *edge = &state->relation_edges[i];
            if (strcmp(edge->relation, relation) != 0) continue;
            if (strcmp(edge->child, node) != 0) continue;
            const char *next = edge->parent;

            int existing = -1;
            for (int j = 0; j < dist_len; j++) {
                if (strcmp(dist_nodes[j], next) == 0) {
                    existing = j;
                    break;
                }
            }
            if (existing >= 0 && dist_vals[existing] < next_distance) {
                continue; /* Longer path — ignore. */
            }

            if (existing < 0) {
                if (dist_len >= node_cap || tail >= node_cap) {
                    free(queue_nodes); free(queue_dists); free(dist_nodes);
                    free(dist_vals); free(pair_targets);
                    free(pair_steps); free(pair_parents);
                    free(results);
                    return false;
                }
                dist_nodes[dist_len] = next;
                dist_vals[dist_len] = next_distance;
                dist_len++;
                queue_nodes[tail] = next;
                queue_dists[tail++] = next_distance;
            } else if (dist_vals[existing] != next_distance) {
                continue;
            }

            /* Emit candidate (Step, Parent) pairs for U→V and union. */
            if (distance == 0) {
                /* U is Source: cand = (V, Source). */
                bool have = false;
                for (int p = 0; p < pair_len; p++) {
                    if (strcmp(pair_targets[p], next) == 0 &&
                        strcmp(pair_steps[p], next) == 0 &&
                        strcmp(pair_parents[p], start) == 0) {
                        have = true;
                        break;
                    }
                }
                if (!have) {
                    if (!wam_tspd5_ensure_pair_cap(&pair_targets, &pair_steps,
                                                   &pair_parents, &pair_cap,
                                                   pair_len + 1)) {
                        free(queue_nodes); free(queue_dists); free(dist_nodes);
                        free(dist_vals); free(pair_targets);
                        free(pair_steps); free(pair_parents);
                        free(results);
                        return false;
                    }
                    pair_targets[pair_len] = next;
                    pair_steps[pair_len] = next;
                    pair_parents[pair_len] = start;
                    pair_len++;
                }
            } else {
                /* Distinct Steps from pairs[U]; cand = (Step, U).
                 * Snapshot pair_len so newly appended V pairs are not
                 * re-scanned as if they belonged to U. */
                int pairs_before = pair_len;
                for (int p = 0; p < pairs_before; p++) {
                    if (strcmp(pair_targets[p], node) != 0) continue;
                    const char *step = pair_steps[p];
                    bool step_seen = false;
                    for (int q = 0; q < p; q++) {
                        if (strcmp(pair_targets[q], node) == 0 &&
                            strcmp(pair_steps[q], step) == 0) {
                            step_seen = true;
                            break;
                        }
                    }
                    if (step_seen) continue;
                    bool have = false;
                    for (int r = 0; r < pair_len; r++) {
                        if (strcmp(pair_targets[r], next) == 0 &&
                            strcmp(pair_steps[r], step) == 0 &&
                            strcmp(pair_parents[r], node) == 0) {
                            have = true;
                            break;
                        }
                    }
                    if (have) continue;
                    if (!wam_tspd5_ensure_pair_cap(&pair_targets, &pair_steps,
                                                   &pair_parents, &pair_cap,
                                                   pair_len + 1)) {
                        free(queue_nodes); free(queue_dists); free(dist_nodes);
                        free(dist_vals); free(pair_targets);
                        free(pair_steps); free(pair_parents);
                        free(results);
                        return false;
                    }
                    pair_targets[pair_len] = next;
                    pair_steps[pair_len] = step;
                    pair_parents[pair_len] = node;
                    pair_len++;
                }
            }
        }
    }

    int value_count = 0;
    for (int i = 0; i < pair_len; i++) {
        const char *t = pair_targets[i];
        const char *s = pair_steps[i];
        const char *p = pair_parents[i];
        int d = 0;
        for (int j = 0; j < dist_len; j++) {
            if (strcmp(dist_nodes[j], t) == 0) {
                d = dist_vals[j];
                break;
            }
        }
        if (target_filter && strcmp(t, target_filter) != 0) continue;
        if (step_filter && strcmp(s, step_filter) != 0) continue;
        if (parent_filter && strcmp(p, parent_filter) != 0) continue;
        if (distance_filter && d != *distance_filter) continue;
        if (!wam_tspd5_ensure_result_cap(&results, &result_cap, value_count + 4)) {
            free(queue_nodes); free(queue_dists); free(dist_nodes);
            free(dist_vals); free(pair_targets);
            free(pair_steps); free(pair_parents);
            free(results);
            return false;
        }
        results[value_count++] = val_atom(t);
        results[value_count++] = val_atom(s);
        results[value_count++] = val_atom(p);
        results[value_count++] = val_int(d);
    }

    free(queue_nodes);
    free(queue_dists);
    free(dist_nodes);
    free(dist_vals);
    free(pair_targets);
    free(pair_steps);
    free(pair_parents);
    if (value_count == 0) {
        free(results);
        return false;
    }
    *results_out = results;
    *value_count_out = value_count;
    return true;
}

bool wam_transitive_step_parent_distance_handler(WamState *state, const char *pred, int arity) {
    if (arity != 5) return false;

    const char *relation = wam_lookup_kernel_edge_relation(state, pred, arity);
    if (!relation) return false;

    const char *start = NULL;
    if (!wam_value_as_atom(state, state->A[0], &start)) return false;

    WamValue *target_cell = wam_deref_ptr(state, &state->A[1]);
    const char *target = NULL;
    if (target_cell->tag == VAL_ATOM) {
        target = target_cell->data.atom;
    } else if (!val_is_unbound(*target_cell)) {
        return false;
    }

    WamValue *step_cell = wam_deref_ptr(state, &state->A[2]);
    const char *step = NULL;
    if (step_cell->tag == VAL_ATOM) {
        step = step_cell->data.atom;
    } else if (!val_is_unbound(*step_cell)) {
        return false;
    }

    WamValue *parent_cell = wam_deref_ptr(state, &state->A[3]);
    const char *parent = NULL;
    if (parent_cell->tag == VAL_ATOM) {
        parent = parent_cell->data.atom;
    } else if (!val_is_unbound(*parent_cell)) {
        return false;
    }

    WamValue *distance_cell = wam_deref_ptr(state, &state->A[4]);
    int distance_filter_value = 0;
    int *distance_filter = NULL;
    if (distance_cell->tag == VAL_INT) {
        distance_filter_value = distance_cell->data.integer;
        if (distance_filter_value <= 0) return false;
        distance_filter = &distance_filter_value;
    } else if (!val_is_unbound(*distance_cell)) {
        return false;
    }

    WamValue *results = NULL;
    int value_count = 0;
    if (!wam_collect_transitive_step_parent_distance(state, relation, start,
                                                     target, step, parent,
                                                     distance_filter,
                                                     &results, &value_count)) {
        return false;
    }

    /* Fully bound Target+Step+Parent (+ optional Distance): succeed once.
     * Bind Distance when unbound; both-bound already filtered. */
    if (target && step && parent) {
        if (value_count < 4) {
            free(results);
            return false;
        }
        WamValue dist_v = results[3];
        free(results);
        if (distance_filter) return true;
        return wam_unify(state, &state->A[4], &dist_v);
    }

    return wam_bind_foreign_quad_stream(state, results, value_count,
                                        state->P + 1);
}

static bool wam_weighted_shortest_path_dijkstra(WamState *state,
                                                const char *start,
                                                const char *target,
                                                const char **target_out,
                                                double *weight_out) {
    const char *nodes[256];
    double distances[256];
    bool done[256];
    int node_count = 0;
    const double inf = 1.0e100;

    nodes[node_count] = start;
    distances[node_count] = 0;
    done[node_count] = false;
    node_count++;

    while (true) {
        int best_idx = -1;
        double best_distance = inf;
        for (int i = 0; i < node_count; i++) {
            if (!done[i] && distances[i] < best_distance) {
                best_idx = i;
                best_distance = distances[i];
            }
        }
        if (best_idx < 0) break;

        done[best_idx] = true;
        const char *node = nodes[best_idx];
        if (target && strcmp(node, target) == 0 && best_distance > 0) {
            *target_out = node;
            *weight_out = best_distance;
            return true;
        }

        for (int i = 0; i < state->weighted_edge_count; i++) {
            WeightedEdge *edge = &state->weighted_edges[i];
            if (edge->weight < 0) continue;
            if (strcmp(edge->source, node) != 0) continue;
            int next_idx = -1;
            for (int j = 0; j < node_count; j++) {
                if (strcmp(nodes[j], edge->target) == 0) {
                    next_idx = j;
                    break;
                }
            }
            if (next_idx < 0) {
                if (node_count >= 256) return false;
                next_idx = node_count;
                nodes[next_idx] = edge->target;
                distances[next_idx] = inf;
                done[next_idx] = false;
                node_count++;
            }
            if (best_distance <= inf - edge->weight &&
                best_distance + edge->weight < distances[next_idx]) {
                distances[next_idx] = best_distance + edge->weight;
            }
        }
    }

    if (!target) {
        int best_idx = -1;
        double best_distance = inf;
        for (int i = 0; i < node_count; i++) {
            if (strcmp(nodes[i], start) != 0 && distances[i] < best_distance) {
                best_idx = i;
                best_distance = distances[i];
            }
        }
        if (best_idx >= 0) {
            *target_out = nodes[best_idx];
            *weight_out = best_distance;
            return true;
        }
    }
    return false;
}

bool wam_weighted_shortest_path_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 3) return false;

    const char *start = NULL;
    if (!wam_value_as_atom(state, state->A[0], &start)) return false;

    WamValue *target_cell = wam_deref_ptr(state, &state->A[1]);
    const char *target = NULL;
    if (target_cell->tag == VAL_ATOM) {
        target = target_cell->data.atom;
    } else if (!val_is_unbound(*target_cell)) {
        return false;
    }

    const char *result_target = NULL;
    double result_weight = 0.0;
    if (!wam_weighted_shortest_path_dijkstra(state, start, target,
                                             &result_target, &result_weight)) {
        return false;
    }

    WamValue target_value = val_atom(result_target);
    WamValue weight_value = val_number_from_double(result_weight);
    return wam_unify(state, &state->A[1], &target_value) &&
           wam_unify(state, &state->A[2], &weight_value);
}

static double wam_astar_heuristic(WamState *state, const char *node, const char *target) {
    for (int i = 0; i < state->direct_distance_edge_count; i++) {
        WeightedEdge *edge = &state->direct_distance_edges[i];
        if (strcmp(edge->source, node) == 0 && strcmp(edge->target, target) == 0) {
            return edge->weight < 0 ? 0 : edge->weight;
        }
    }
    return 0;
}

static bool wam_astar_shortest_path_search(WamState *state,
                                           const char *start,
                                           const char *target,
                                           int dimensionality,
                                           double *weight_out) {
    const char *nodes[256];
    double distances[256];
    bool done[256];
    int node_count = 0;
    const double inf = 1.0e100;
    (void)dimensionality;

    nodes[node_count] = start;
    distances[node_count] = 0;
    done[node_count] = false;
    node_count++;

    while (true) {
        int best_idx = -1;
        double best_score = inf;
        double best_distance = inf;
        for (int i = 0; i < node_count; i++) {
            if (done[i]) continue;
            double heuristic = wam_astar_heuristic(state, nodes[i], target);
            double score = distances[i] <= inf - heuristic ? distances[i] + heuristic : inf;
            if (score < best_score || (score == best_score && distances[i] < best_distance)) {
                best_idx = i;
                best_score = score;
                best_distance = distances[i];
            }
        }
        if (best_idx < 0) break;

        done[best_idx] = true;
        const char *node = nodes[best_idx];
        if (strcmp(node, target) == 0 && best_distance > 0) {
            *weight_out = best_distance;
            return true;
        }

        for (int i = 0; i < state->weighted_edge_count; i++) {
            WeightedEdge *edge = &state->weighted_edges[i];
            if (edge->weight < 0) continue;
            if (strcmp(edge->source, node) != 0) continue;
            int next_idx = -1;
            for (int j = 0; j < node_count; j++) {
                if (strcmp(nodes[j], edge->target) == 0) {
                    next_idx = j;
                    break;
                }
            }
            if (next_idx < 0) {
                if (node_count >= 256) return false;
                next_idx = node_count;
                nodes[next_idx] = edge->target;
                distances[next_idx] = inf;
                done[next_idx] = false;
                node_count++;
            }
            if (best_distance <= inf - edge->weight &&
                best_distance + edge->weight < distances[next_idx]) {
                distances[next_idx] = best_distance + edge->weight;
            }
        }
    }
    return false;
}

bool wam_astar_shortest_path_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 4) return false;

    const char *start = NULL;
    const char *target = NULL;
    if (!wam_value_as_atom(state, state->A[0], &start)) return false;
    if (!wam_value_as_atom(state, state->A[1], &target)) return false;

    WamValue *dim_cell = wam_deref_ptr(state, &state->A[2]);
    if (dim_cell->tag != VAL_INT) return false;

    double result_weight = 0.0;
    if (!wam_astar_shortest_path_search(state, start, target,
                                        dim_cell->data.integer,
                                        &result_weight)) {
        return false;
    }

    WamValue weight_value = val_number_from_double(result_weight);
    return wam_unify(state, &state->A[3], &weight_value);
}
'.

compile_wam_runtime_to_c(Options, CCode) :-
    compile_step_wam_to_c(Options, StepCode),
    compile_wam_helpers_to_c(Options, HelpersCode),
    format(atom(CCode), "~w\n\n~w", [HelpersCode, StepCode]).
