:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_llvm_target.pl - WAM-to-LLVM IR Transpilation Target
%
% Transpiles WAM runtime predicates (wam_runtime.pl) to LLVM IR code.
% Phase 2: step_wam/3 → LLVM switch dispatch
% Phase 3: helper predicates → LLVM functions
% Phase 4: WAM instructions → LLVM struct literals
% Phase 5: Hybrid module assembly
%
% Key LLVM-specific design choices:
%   - Value = { i32 tag, i64 payload } tagged union (not enum/interface)
%   - Instruction dispatch via LLVM switch on integer tag
%   - Registers = [64 x %Value] fixed array (not HashMap/map)
%   - Run loop uses musttail for constant-stack execution
%   - Arena-style memory (malloc + backtrack rewind)
%
% See: docs/design/WAM_LLVM_TRANSPILATION_IMPLEMENTATION_PLAN.md

:- module(wam_llvm_target, [
    compile_step_wam_to_llvm/2,          % +Options, -LLVMCode
    compile_wam_helpers_to_llvm/2,       % +Options, -LLVMCode
    compile_wam_runtime_to_llvm/2,       % +Options, -LLVMCode (step + helpers combined)
    compile_wam_predicate_to_llvm/4,     % +Pred/Arity, +WamCode, +Options, -LLVMCode
    wam_instruction_to_llvm_literal/2,   % +WamInstr, -LLVMLiteral (errors on label-ref instrs)
    wam_instruction_to_llvm_literal/3,   % +WamInstr, +LabelMap, -LLVMLiteral
    wam_line_to_llvm_literal/2,          % +Parts, -LLVMLit
    write_wam_llvm_project/3,            % +Predicates, +Options, +OutputFile
    wam_llvm_last_compile_counts/2,      % ?InstrCount, ?LabelCount (side channel)
    write_wam_llvm_wasm_project/3,       % +Predicates, +Options, +OutputFile (WASM variant)
    build_wam_wasm_module/3,             % +LLFile, +OutputName, -Commands
    builtin_op_to_id/2,                  % +OpName, -IntId
    % Foreign dispatch (M3)
    wam_llvm_foreign_kind_id/2,          % +Kind, -Id
    % Indexed fact table emission (M4)
    llvm_emit_atom_fact2_table/3,        % +TableName, +Pairs, -LLVMGlobal
    llvm_emit_weighted_edge_table/3,     % +TableName, +Triples, -LLVMGlobal
    % Foreign lowering pipeline (M5.6)
    llvm_foreign_kernel_spec/3,          % ?Pred/Arity, ?KernelKind, ?Config (dynamic)
    clear_llvm_foreign_kernel_specs/0,   % retractall helper (for test isolation)
    foreign_kernel/3,                    % +Pred/Arity, +Kind, +Config (user directive)
    % Shared helpers (used by wam_llvm_lowered_emitter)
    sanitize_functor_for_llvm/2,         % +Name, -SaneName (bijective hex escape)
    register_functor_string/1,           % +NameStr (assert into functor_string_global table)
    split_functor_arity/3,               % +Str, -Name, -Arity (handles `/` in name)
    llvm_emit_atom_prefix_guard/5,       % +GlobalBase, +ValueIR, +Prefix, +ResultIR, -GlobalIR-CallIR
    llvm_emit_atom_field_eq_guard/7,     % +GlobalBase, +ValueIR, +FieldIndex, +Expected, +SepCode, +ResultIR, -GlobalIR-CallIR
    llvm_emit_regex_field_match_guard/7, % +GlobalBase, +ValueIR, +FieldIndex, +Regex, +SepCode, +ResultIR, -GlobalIR-CallIR
    llvm_emit_atom_field_slice/5,        % +ValueIR, +FieldIndex, +SepCode, +SliceBase, -CallIR
    llvm_emit_atom_field_count/4,        % +ValueIR, +SepCode, +CountBase, -CallIR
    llvm_emit_atom_field_length/5,       % +ValueIR, +FieldIndex, +SepCode, +LengthBase, -CallIR
    llvm_emit_atom_field_subslice/7,     % +ValueIR, +FieldIndex, +SepCode, +Start, +Len, +SliceBase, -CallIR
    llvm_emit_atom_field_index/7,        % +GlobalBase, +ValueIR, +FieldIndex, +Needle, +SepCode, +IndexBase, -GlobalIR-CallIR
    llvm_emit_atom_field_i64_cmp_guard/7,% +ValueIR, +FieldIndex, +OpCode, +Expected, +SepCode, +ResultIR, -CallIR
    llvm_emit_atom_field_strnum_cmp_guard/7,% +ValueIR, +FieldIndex, +OpCode, +Expected, +SepCode, +ResultIR, -CallIR
    llvm_emit_atom_field_str_cmp_guard/8,% +GlobalBase, +ValueIR, +FieldIndex, +OpCode, +Expected, +SepCode, +ResultIR, -GlobalIR-CallIR
    llvm_emit_atom_field_i64/5,          % +ValueIR, +FieldIndex, +SepCode, +ParseBase, -CallIR
    llvm_emit_atom_field_i64_or_default/7,% +ValueIR, +FieldIndex, +SepCode, +DefaultValue, +ParseBase, +ResultIR, -CallIR
    llvm_emit_c_string_global/5,         % +GlobalName, +Value, -GlobalIR, -StringLen, -BytesLen
    llvm_emit_printf_i64/5,              % +FmtGlobal, +FmtVar, +PrintVar, +ValueIR, -Parts
    llvm_emit_printf_slice/6,            % +FmtGlobal, +FmtVar, +PrintVar, +LenIR, +PtrIR, -Parts
    llvm_emit_printf_string/5,           % +FmtGlobal, +FmtVar, +PrintVar, +PtrIR, -Parts
    llvm_emit_printf_string/6,           % +FmtGlobal, +FmtLen, +FmtVar, +PrintVar, +PtrIR, -Parts
    llvm_emit_printf0/5,                 % +FmtGlobal, +FmtLen, +FmtVar, +PrintVar, -Parts
    llvm_emit_stream_driver_ir/3,        % +InputPath, +DriverBlocks, -DriverIR
    llvm_emit_binary_stream_driver_ir/4, % +InputPath, +RecordSize, +DriverBlocks, -DriverIR
    llvm_emit_varlen_stream_driver_ir/5, % +InputPath, +BufSize, +ReadIR, +DriverBlocks, -DriverIR
    llvm_emit_ascii_case_slice_print/5,  % +Mode, +PtrIR, +LenIR, +PrintBase, -CallIR
    % Phase 5 (JIT): runtime-loadable WAM objects (.wamo)
    write_wam_object/3,                  % +Predicates, +Options, +OutFile
    wam_object_encode/3,                 % +Predicates, +Options, -Text (in-memory .wamo bytes)
    wam_object_support_ir/1              % -IR (loader + call primitive; append to a host module)
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module(wam_llvm_runtime_libs, [wam_llvm_runtime_ir/2]).
:- use_module('../bindings/llvm_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module(wam_llvm_lowered_emitter, [
    wam_llvm_lowerable/3,
    wam_llvm_lowerable_with_closure/4,
    lower_predicate_to_llvm/4,
    llvm_lowered_func_name/2,
    call_execute_targets/2,
    emit_native_wrapper/2,
    emit_hybrid_dispatcher/5
]).

:- discontiguous wam_llvm_case/2.
:- discontiguous wam_line_to_llvm_literal/2.
:- discontiguous wam_instruction_to_llvm_literal/2.

% ============================================================================
% Foreign Lowering Spec Table (M5.6)
% ============================================================================
%
% The canonical record for "this predicate should be lowered to a native
% foreign kernel instead of being compiled to WAM". Populated via three
% user-facing paths that all funnel into this single table:
%
%   (a) :- foreign_kernel(Pred/Arity, Kind, Config) directive (M5.6b)
%   (b) foreign_predicates([...]) entry in Options (M5.6a, this commit)
%   (c) foreign_lowering(true) + automatic pattern matching (M5.6c)
%
% Path (b) is a thin wrapper around (a) — the options-list helper just
% asserts the same facts the directive would. Path (c) invokes detector
% predicates that inspect clause shape and assert specs when a known
% recursive kernel pattern matches.
%
% The compile pipeline checks this table *before* trying native LLVM
% lowering or WAM fallback. When a spec exists, the predicate body is
% replaced with a single `call_foreign Kind, Arity` instruction, and the
% runtime template's weak @wam_<kind>_kernel_impl default is spliced
% out and replaced with a concrete body that reads registers, scans the
% fact table, and writes the result.

:- dynamic llvm_foreign_kernel_spec/3.
:- dynamic wam_llvm_last_compile_counts/2.   % M108: per-compile (Instr, Label) counts.

clear_llvm_foreign_kernel_specs :-
    retractall(llvm_foreign_kernel_spec(_, _, _)).

%% apply_foreign_predicates_option(+Options) is det.
%
%  Reads a `foreign_predicates([...])` entry from the options list and
%  asserts a llvm_foreign_kernel_spec/3 fact for each entry. Accepted
%  entry shapes:
%
%    Pred/Arity - Kind - Config
%    foreign_kernel(Pred/Arity, Kind, Config)
%
%  Any other shape is silently ignored so future extensions don't break
%  old option lists.
apply_foreign_predicates_option(Options) :-
    ( option(foreign_predicates(Entries), Options)
    -> forall(member(Entry, Entries), assert_foreign_entry(Entry))
    ; true
    ).

assert_foreign_entry(PredArity - Kind - Config) :- !,
    assertz(llvm_foreign_kernel_spec(PredArity, Kind, Config)).
assert_foreign_entry(foreign_kernel(PredArity, Kind, Config)) :- !,
    assertz(llvm_foreign_kernel_spec(PredArity, Kind, Config)).
assert_foreign_entry(_) :- true.  % ignore unrecognized shapes

%% foreign_kernel(+PredArity, +Kind, +Config) is det.
%
%  M5.6b: user-facing directive-style entry point. A user source file
%  can import this predicate and use it as a directive to declare that
%  a predicate should be lowered to a native foreign kernel:
%
%    :- use_module(library(wam_llvm_target),
%         [foreign_kernel/3, write_wam_llvm_project/3]).
%    :- foreign_kernel(my_distance/3, transitive_distance3,
%         [edge_pred(edge/2)]).
%
%  The directive simply asserts a llvm_foreign_kernel_spec/3 fact —
%  the rest of the pipeline treats it identically to specs that come
%  in via the options-list path or the auto-detector path.
foreign_kernel(PredArity, Kind, Config) :-
    assertz(llvm_foreign_kernel_spec(PredArity, Kind, Config)).

% ============================================================================
% M5.6c — Automatic Foreign Kernel Detection
% ============================================================================
%
% When `foreign_lowering(true)` is set in Options, the compile pipeline
% walks each user predicate's clauses and runs them against a registry
% of detectors. A detector is a predicate that returns a
% `recursive_kernel(Kind, Pred/Arity, Config)` term when the clauses
% match a known recursive-kernel shape. The matched spec is then
% asserted into the same llvm_foreign_kernel_spec/3 table that paths
% (a) and (b) populate, so there's only one downstream code path.
%
% This mirrors the Rust target's rust_recursive_kernel_detector/2
% registry at rust_target.pl:3742-3751 and the matcher predicates at
% rust_target.pl:3968-3988. Currently only transitive_distance3 is
% implemented; weighted_shortest_path3 and astar_shortest_path4 can
% be added by porting the same matcher structure once their
% dispatcher stubs are wired (the M5.5 weak-default pattern only
% covers td3 so far).

%% llvm_recursive_kernel_detector(?Kind, ?DetectorPredicate).
%
%  Registry of kernel kinds and the predicates that can detect them.
%  DetectorPredicate(+Pred, +Arity, +Clauses, -RecKernel) should bind
%  RecKernel to a `recursive_kernel(Kind, Pred/Arity, Config)` term
%  on success.
llvm_recursive_kernel_detector(countdown_sum2,
    llvm_recursive_kernel_countdown_sum).
llvm_recursive_kernel_detector(category_ancestor,
    llvm_recursive_kernel_category_ancestor).
llvm_recursive_kernel_detector(list_suffix2,
    llvm_recursive_kernel_list_suffix).
llvm_recursive_kernel_detector(transitive_closure2,
    llvm_recursive_kernel_transitive_closure).
llvm_recursive_kernel_detector(transitive_distance3,
    llvm_recursive_kernel_transitive_distance).
llvm_recursive_kernel_detector(weighted_shortest_path3,
    llvm_recursive_kernel_weighted_shortest_path).
llvm_recursive_kernel_detector(astar_shortest_path4,
    llvm_recursive_kernel_astar_shortest_path).

%% llvm_recursive_kernel_transitive_distance(+Pred, +Arity, +Clauses, -RecKernel).
%
%  Detects the transitive_distance3 shape:
%    pred(Start, Target, 1)     :- edge(Start, Target).
%    pred(Start, Target, Depth) :-
%        edge(Start, Mid),
%        pred(Mid, Target, PrevDepth),
%        Depth is PrevDepth + 1.
%
%  On success binds RecKernel to
%    recursive_kernel(transitive_distance3, Pred/Arity,
%                     [edge_pred(EdgePred/2)]).
llvm_recursive_kernel_transitive_distance(Pred, Arity, Clauses,
        recursive_kernel(transitive_distance3, Pred/Arity,
            [edge_pred(EdgePred/2)])) :-
    llvm_foreign_lowerable_transitive_distance(Pred, Arity, Clauses, EdgePred).

%% llvm_recursive_kernel_countdown_sum(+Pred, +Arity, +Clauses, -RecKernel).
%
%  Detects the countdown_sum2 clause shape:
%    pred(0, 0).
%    pred(N, Sum) :- N > 0, N1 is N - 1, pred(N1, PrevSum), Sum is PrevSum + N.
llvm_recursive_kernel_countdown_sum(Pred, Arity, Clauses,
        recursive_kernel(countdown_sum2, Pred/Arity, [])) :-
    llvm_foreign_lowerable_countdown_sum(Pred, Arity, Clauses).

%% llvm_foreign_lowerable_countdown_sum(+Pred, +Arity, +Clauses).
%
%  Ported from rust_target.pl:3902. Matches the countdown-sum recurrence.
llvm_foreign_lowerable_countdown_sum(Pred, 2, Clauses) :-
    member(BaseHead-true, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, 0, 0],
    RecHead =.. [Pred, N, Sum],
    RecBody = (GtGoal, (StepGoal, (RecGoal, SumGoal))),
    GtGoal =.. [>, N, 0],
    StepGoal =.. [is, PrevN, StepExpr],
    ( StepExpr =.. [-, N, 1] ; StepExpr =.. [+, N, -1] ),
    RecGoal =.. [Pred, PrevN, PrevSum],
    SumGoal =.. [is, Sum, SumExpr],
    SumExpr =.. [+, PrevSum, N].

%% llvm_recursive_kernel_category_ancestor(+Pred, +Arity, +Clauses, -RecKernel).
%
%  Detects the category_ancestor clause shape (arity 4):
%    pred(Cat, Target, 1, Visited) :-
%        edge(Cat, Target), \+ member(Target, Visited).
%    pred(Cat, Target, Hops, Visited) :-
%        edge(Cat, Mid), \+ member(Mid, Visited),
%        pred(Mid, Target, H1, [Mid|Visited]),
%        Hops is H1 + 1.
%
%  Requires user:max_depth/1 to be defined.
llvm_recursive_kernel_category_ancestor(Pred, Arity, Clauses,
        recursive_kernel(category_ancestor, Pred/Arity,
            [edge_pred(EdgePred/2), max_depth(MaxDepth)])) :-
    llvm_foreign_lowerable_category_ancestor(Pred, Arity, Clauses, EdgePred),
    ( user:max_depth(MaxDepth) -> true ; MaxDepth = 10 ).

%% llvm_foreign_lowerable_category_ancestor(+Pred, +Arity, +Clauses, -EdgePred).
%
%  Ported from rust_target.pl:3889. Matches predicates with arity 4
%  that have two clause bodies (neither is `true`), both containing
%  negation-as-failure (\+ member/2), and the recursive body containing
%  arithmetic (Hops is H1 + 1).
llvm_foreign_lowerable_category_ancestor(Pred, 4, Clauses, EdgePred) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead \== RecHead,
    BaseBody \== true,
    RecBody \== true,
    BaseHead =.. [Pred, _, _, _, _],
    RecHead =.. [Pred, _, _, _, _],
    % Base body must contain an edge call and a negation check.
    term_string(BaseBody, BaseStr),
    sub_string(BaseStr, _, _, _, "\\+"),
    % Recursive body must contain edge call, negation, recursive call, and arithmetic.
    term_string(RecBody, RecStr),
    sub_string(RecStr, _, _, _, "\\+"),
    sub_string(RecStr, _, _, _, " is "),
    % Extract edge predicate from base body.
    BaseBody = (EdgeGoal, _),
    EdgeGoal =.. [EdgePred, _, _].

%% llvm_recursive_kernel_list_suffix(+Pred, +Arity, +Clauses, -RecKernel).
%
%  Detects the list_suffix2 clause shape:
%    pred(X, X).
%    pred([_|Tail], Suffix) :- pred(Tail, Suffix).
llvm_recursive_kernel_list_suffix(Pred, Arity, Clauses,
        recursive_kernel(list_suffix2, Pred/Arity, [])) :-
    llvm_foreign_lowerable_list_suffix(Pred, Arity, Clauses).

%% llvm_foreign_lowerable_list_suffix(+Pred, +Arity, +Clauses).
%
%  Ported from rust_target.pl:3917. Matches the list-suffix pattern.
llvm_foreign_lowerable_list_suffix(Pred, 2, Clauses) :-
    member(BaseHead-true, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, BaseList, BaseList],
    var(BaseList),
    RecHead =.. [Pred, InputList, Suffix],
    InputList = [_|Tail],
    RecBody =.. [Pred, Tail, Suffix].

%% llvm_recursive_kernel_transitive_closure(+Pred, +Arity, +Clauses, -RecKernel).
%
%  Detects the transitive_closure2 clause shape:
%    pred(Start, Target) :- edge(Start, Target).
%    pred(Start, Target) :- edge(Start, Mid), pred(Mid, Target).
llvm_recursive_kernel_transitive_closure(Pred, Arity, Clauses,
        recursive_kernel(transitive_closure2, Pred/Arity,
            [edge_pred(EdgePred/2)])) :-
    llvm_foreign_lowerable_transitive_closure(Pred, Arity, Clauses, EdgePred).

%% llvm_foreign_lowerable_transitive_closure(+Pred, +Arity, +Clauses, -EdgePred).
%
%  Ported from rust_target.pl:3933. Simple two-clause transitive
%  closure pattern.
llvm_foreign_lowerable_transitive_closure(Pred, 2, Clauses, EdgePred) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, BaseStart, BaseTarget],
    BaseBody =.. [EdgePred, BaseStart, BaseTarget],
    RecHead =.. [Pred, RecStart, RecTarget],
    RecBody = (EdgeGoal, RecGoal),
    EdgeGoal =.. [EdgePred, RecStart, RecMid],
    RecGoal =.. [Pred, RecMid, RecTarget].

%% llvm_foreign_lowerable_transitive_distance(+Pred, +Arity, +Clauses, -EdgePred).
%
%  Clause-shape matcher ported from rust_target.pl:3968. The only
%  difference from the Rust version is that we do not gather the
%  fact pairs here — the M5.6a build_td3_concrete_impl/6 helper
%  reads them from the user module at codegen time, so the matcher
%  just needs to confirm the EdgePred name and arity.
llvm_foreign_lowerable_transitive_distance(Pred, 3, Clauses, EdgePred) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, BaseStart, BaseTarget, 1],
    RecHead =.. [Pred, RecStart, RecTarget, RecDepth],
    BaseBody =.. [EdgePred, BaseStart, BaseTarget],
    RecBody = (EdgeGoal, (RecGoal, IsGoal)),
    EdgeGoal =.. [EdgePred, RecStart, RecMid],
    RecGoal =.. [Pred, RecMid, RecTarget, PrevDepth],
    IsGoal =.. [is, RecDepth, Expr],
    Expr =.. [+, PrevDepth, 1].

%% llvm_recursive_kernel_weighted_shortest_path(+Pred, +Arity, +Clauses, -RecKernel).
%
%  Detects the weighted_shortest_path3 clause shape:
%    pred(X, Y, W) :- weight_pred(X, Y, W).
%    pred(X, Y, Cost) :-
%        weight_pred(X, Z, W),
%        pred(Z, Y, RestCost),
%        Cost is W + RestCost.
%
%  On success binds RecKernel to
%    recursive_kernel(weighted_shortest_path3, Pred/Arity,
%                     [weight_pred(WeightPred/3)]).
llvm_recursive_kernel_weighted_shortest_path(Pred, Arity, Clauses,
        recursive_kernel(weighted_shortest_path3, Pred/Arity,
            [weight_pred(WeightPred/3)])) :-
    llvm_foreign_lowerable_weighted_shortest_path(Pred, Arity, Clauses, WeightPred).

%% llvm_foreign_lowerable_weighted_shortest_path(+Pred, +Arity, +Clauses, -WeightPred).
%
%  Ported from rust_target.pl:4045. Accepts the simple-form recursive
%  body; the Rust target also accepts a visited-list form, which
%  M5.9 does not yet mirror because Dijkstra doesn't need cycle
%  checks (it tracks visited internally via best[]).
llvm_foreign_lowerable_weighted_shortest_path(Pred, 3, Clauses, WeightPred) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead \== RecHead,
    BaseHead =.. [Pred, BaseStart, BaseTarget, BaseWeight],
    BaseBody =.. [WeightPred, BaseStart, BaseTarget, BaseWeight],
    RecHead =.. [Pred, RecStart, RecTarget, RecCost],
    RecBody = (WeightGoal, (RecGoal, IsGoal)),
    WeightGoal =.. [WeightPred, RecStart, RecMid, W],
    RecGoal =.. [Pred, RecMid, RecTarget, RestCost],
    IsGoal =.. [is, RecCost, PlusExpr],
    PlusExpr =.. [+, W, RestCost].

%% llvm_recursive_kernel_astar_shortest_path(+Pred, +Arity, +Clauses, -RecKernel).
%
%  Detects the astar_shortest_path4 clause shape (arity 4):
%    pred(X, Y, W, _Vis) :- weight_pred(X, Y, W).
%    pred(X, Y, Cost, Vis) :-
%        weight_pred(X, Z, W),
%        pred(Z, Y, RC, [Z|Vis]),
%        Cost is W + RC.
%
%  Extracts weight_pred. If user:direct_semantic_dist/3 is defined,
%  includes it as direct_dist_pred for runtime heuristic support.
llvm_recursive_kernel_astar_shortest_path(Pred, Arity, Clauses,
        recursive_kernel(astar_shortest_path4, Pred/Arity, Config)) :-
    llvm_foreign_lowerable_astar_shortest_path(Pred, Arity, Clauses, WeightPred),
    ( predicate_property(user:direct_semantic_dist(_,_,_), defined)
    -> Config = [weight_pred(WeightPred/3), direct_dist_pred(direct_semantic_dist/3)]
    ;  Config = [weight_pred(WeightPred/3)]
    ).

%% llvm_foreign_lowerable_astar_shortest_path(+Pred, +Arity, +Clauses, -WeightPred).
%
%  Matches arity-4 predicates with weighted shortest path + visited list.
%  Base clause: pred(X, Y, W, _) :- weight(X, Y, W).
%  Recursive clause: pred(X, Y, Cost, Vis) :- weight(X, Z, W), pred(Z, Y, RC, ...), Cost is W + RC.
llvm_foreign_lowerable_astar_shortest_path(Pred, 4, Clauses, WeightPred) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead \== RecHead,
    BaseHead =.. [Pred, _, _, _, _],
    RecHead =.. [Pred, _, _, _, _],
    % Base body is a single weight_pred call (ignoring visited arg).
    BaseBody =.. [WeightPred, _, _, _],
    WeightPred \== Pred,
    % Recursive body contains weight call, recursive call, and arithmetic.
    RecBody = (WeightGoal, RestBody),
    WeightGoal =.. [WeightPred, _, _, _],
    % Rest may have \+ member(...) interleaved; just check for the recursive call and is/2.
    term_string(RestBody, RestStr),
    atom_string(Pred, PredStr),
    sub_string(RestStr, _, _, _, PredStr),
    sub_string(RestStr, _, _, _, " is ").

%% llvm_auto_detect_foreign_kernels(+Predicates) is det.
%
%  For each predicate indicator in the list, read its clauses from
%  the user module and run them against every registered detector.
%  When a detector matches, assert a llvm_foreign_kernel_spec/3 fact
%  for the result. Idempotent — if a spec already exists for the
%  predicate (e.g., from the options-list path), the auto-detect
%  skips it to avoid duplicate registration.
llvm_auto_detect_foreign_kernels([]).
llvm_auto_detect_foreign_kernels([PredIndicator|Rest]) :-
    ( PredIndicator = Module:Pred/Arity -> true
    ; PredIndicator = Pred/Arity, Module = user
    ),
    ( lookup_foreign_kernel_spec(Pred, Arity, _, _)
    -> true   % already registered via (a) or (b); don't override
    ;  llvm_collect_clause_pairs(Module, Pred, Arity, Clauses),
       ( llvm_try_detectors(Pred, Arity, Clauses, Kind, Config)
       -> assertz(llvm_foreign_kernel_spec(Pred/Arity, Kind, Config))
       ;  true
       )
    ),
    llvm_auto_detect_foreign_kernels(Rest).

%% llvm_collect_clause_pairs(+Module, +Pred, +Arity, -Pairs) is det.
%
%  Gather (Head - Body) pairs for every clause of Module:Pred/Arity.
%  Catch + default to empty list so predicates with no clauses (or
%  predicates that are opaque to clause/2) don't blow up the compile.
llvm_collect_clause_pairs(Module, Pred, Arity, Pairs) :-
    catch(
        findall(Head-Body,
            ( functor(Head, Pred, Arity),
              clause(Module:Head, Body)
            ),
            Pairs),
        _, Pairs = []).

%% llvm_try_detectors(+Pred, +Arity, +Clauses, -Kind, -Config) is semidet.
%
%  Iterate the detector registry and return the first match.
llvm_try_detectors(Pred, Arity, Clauses, Kind, Config) :-
    llvm_recursive_kernel_detector(Kind, DetectorName),
    Goal =.. [DetectorName, Pred, Arity, Clauses, Result],
    call(Goal),
    Result = recursive_kernel(Kind, _, Config),
    !.

%% lookup_foreign_kernel_spec(+Pred, +Arity, -Kind, -Config) is semidet.
%
%  Succeeds iff a spec is registered for the given predicate. Strips
%  any Module: qualifier so callers can pass `user:foo/3` or `foo/3`
%  interchangeably.
lookup_foreign_kernel_spec(Pred, Arity, Kind, Config) :-
    ( llvm_foreign_kernel_spec(Pred/Arity, Kind, Config)
    ; llvm_foreign_kernel_spec(_:Pred/Arity, Kind, Config)
    ), !.

%% substitute_foreign_kernel_impls(+StateFuncsRaw, -StateFuncs, -Globals).
%
%  Post-processes the rendered state template output to splice concrete
%  kernel impl bodies in place of the weak defaults, and collects any
%  module-level globals (fact tables, etc.) the impls depend on.
%
%  When no foreign specs are registered this is a pass-through:
%    StateFuncs = StateFuncsRaw, Globals = ''.
%
%  When one or more td3 specs are registered, M5.8 emits a single
%  concrete @wam_td3_kernel_impl that `switch`es on %instance with
%  one case per registered predicate. Each case GEPs its own
%  module-local %AtomFactPair table and calls the @wam_td3_run
%  helper. This lets multiple td3 predicates with *different*
%  edge_preds coexist in one module — each gets its own instance_id
%  and its own edge table.
substitute_foreign_kernel_impls(StateFuncsRaw, StateFuncs, Globals) :-
    % td3 pass
    findall(PredArity-Config,
        llvm_foreign_kernel_spec(PredArity, transitive_distance3, Config),
        Td3Entries),
    ( Td3Entries == []
    -> StateFuncs0 = StateFuncsRaw, Td3Tables = ''
    ;  build_td3_instance_switch(Td3Entries, Td3ImplBody, Td3Tables),
       replace_td3_weak_default(StateFuncsRaw, Td3ImplBody, StateFuncs0)
    ),
    % ca pass — category_ancestor (depth-bounded BFS).
    findall(CaPredArity-CaConfig,
        llvm_foreign_kernel_spec(CaPredArity, category_ancestor, CaConfig),
        CaEntries),
    ( CaEntries == []
    -> StateFuncsCa = StateFuncs0, CaTables = ''
    ;  build_ca_instance_switch(CaEntries, CaImplBody, CaTables),
       replace_ca_weak_default(StateFuncs0, CaImplBody, StateFuncsCa)
    ),
    % cds2 pass — countdown_sum2 (deterministic arithmetic).
    findall(CdsPredArity-CdsConfig,
        llvm_foreign_kernel_spec(CdsPredArity, countdown_sum2, CdsConfig),
        Cds2Entries),
    ( Cds2Entries == []
    -> StateFuncsCds = StateFuncsCa, Cds2Tables = ''
    ;  build_cds2_instance_switch(Cds2Entries, Cds2ImplBody, Cds2Tables),
       replace_cds2_weak_default(StateFuncsCa, Cds2ImplBody, StateFuncsCds)
    ),
    % tc2 pass — transitive_closure2 (boolean reachability).
    findall(TcPredArity-TcConfig,
        llvm_foreign_kernel_spec(TcPredArity, transitive_closure2, TcConfig),
        Tc2Entries),
    ( Tc2Entries == []
    -> StateFuncsTc = StateFuncsCds, Tc2Tables = ''
    ;  build_tc2_instance_switch(Tc2Entries, Tc2ImplBody, Tc2Tables),
       replace_tc2_weak_default(StateFuncsCds, Tc2ImplBody, StateFuncsTc)
    ),
    % ls2 pass — list_suffix2 (enumerate suffixes via backtracking).
    findall(LsPredArity-LsConfig,
        llvm_foreign_kernel_spec(LsPredArity, list_suffix2, LsConfig),
        Ls2Entries),
    ( Ls2Entries == []
    -> StateFuncsLs = StateFuncsTc, Ls2Tables = ''
    ;  build_ls2_instance_switch(Ls2Entries, Ls2ImplBody, Ls2Tables),
       replace_ls2_weak_default(StateFuncsTc, Ls2ImplBody, StateFuncsLs)
    ),
    % wsp3 pass — mirror of the td3 pass but for weighted_shortest_path3.
    findall(WspPredArity-WspConfig,
        llvm_foreign_kernel_spec(WspPredArity, weighted_shortest_path3, WspConfig),
        Wsp3Entries),
    ( Wsp3Entries == []
    -> StateFuncs1 = StateFuncsLs, Wsp3Tables = ''
    ;  build_wsp3_instance_switch(Wsp3Entries, Wsp3ImplBody, Wsp3Tables),
       replace_wsp3_weak_default(StateFuncsLs, Wsp3ImplBody, StateFuncs1)
    ),
    % astar4 pass
    findall(AsPredArity-AsConfig,
        llvm_foreign_kernel_spec(AsPredArity, astar_shortest_path4, AsConfig),
        Astar4Entries),
    ( Astar4Entries == []
    -> StateFuncs = StateFuncs1, Astar4Tables = ''
    ;  build_astar4_instance_switch(Astar4Entries, Astar4ImplBody, Astar4Tables),
       replace_astar4_weak_default(StateFuncs1, Astar4ImplBody, StateFuncs)
    ),
    % Concatenate the per-kind global tables into one Globals blob.
    ( CaTables == '', Cds2Tables == '', Td3Tables == '', Tc2Tables == '', Ls2Tables == '', Wsp3Tables == '', Astar4Tables == ''
    -> Globals = ''
    ;  atomic_list_concat([
           '; === foreign kernel support globals ===\n',
           CaTables, '\n', Cds2Tables, '\n', Tc2Tables, '\n', Ls2Tables, '\n', Td3Tables, '\n', Wsp3Tables, '\n', Astar4Tables, '\n'
       ], Globals)
    ).

%% build_td3_instance_switch(+Entries, -ImplBody, -TablesIR).
%
%  Entries is a list of PredArity-Config pairs (the order matches the
%  instance_id assignment — first entry is instance 0). Produces:
%
%    - ImplBody: the full `define i1 @wam_td3_kernel_impl(%WamState*,
%      i32 %instance)` body with one switch case per entry, each
%      loading its own edge table pointer/len/max and calling
%      @wam_td3_run.
%    - TablesIR: the concatenated %AtomFactPair global constant
%      definitions for each entry's edge table. These are dropped
%      into the module's native_predicates section so they are at
%      module scope before any use.
build_td3_instance_switch(Entries, ImplBody, TablesIR) :-
    build_td3_instance_parts(Entries, 0, SwitchCases, CaseBodies, Tables),
    atomic_list_concat(SwitchCases, '\n', SwitchCasesStr),
    atomic_list_concat(CaseBodies, '\n\n', CaseBodiesStr),
    atomic_list_concat(Tables, '\n\n', TablesIR),
    format(atom(ImplBody),
'define i1 @wam_td3_kernel_impl(%WamState* %vm, i32 %instance) {
entry:
  switch i32 %instance, label %bail [
~w
  ]

~w

bail:
  ret i1 false
}',
        [SwitchCasesStr, CaseBodiesStr]).

%% build_td3_instance_parts(+Entries, +StartIndex, -SwitchCases, -Bodies, -Tables).
%
%  Walks Entries assigning sequential instance IDs starting at
%  StartIndex. For each entry produces three pieces:
%
%    - SwitchCase: `    i32 N, label %inst_N`
%    - Body:       `inst_N:\n  %tblN = getelementptr ...\n
%                   %rN = call i1 @wam_td3_run(...)\n  ret i1 %rN`
%    - Table:      the private-constant %AtomFactPair definition
%                  for this instance's edge table.
build_td3_instance_parts([], _, [], [], []).
build_td3_instance_parts([PredArity-Config | Rest], Index,
        [SwitchCase|RestCases], [Body|RestBodies], [TableIR|RestTables]) :-
    PredArity = Pred/_,
    sanitize_atom_for_llvm(Pred, SanePred),
    format(atom(TableName), 'td3_inst_~w_~w_edges', [SanePred, Index]),
    build_td3_instance_table(Config, TableName, TableIR, GepLen, EffLen, MaxAtomId),
    format(atom(SwitchCase), '    i32 ~w, label %inst_~w', [Index, Index]),
    format(atom(Body),
'inst_~w:
  %tbl_~w = getelementptr [~w x %AtomFactPair], [~w x %AtomFactPair]* @~w, i64 0, i64 0
  %r_~w = call i1 @wam_td3_run(%WamState* %vm, %AtomFactPair* %tbl_~w, i64 ~w, i64 ~w)
  ret i1 %r_~w',
        [Index, Index, GepLen, GepLen, TableName, Index, Index, EffLen, MaxAtomId, Index]),
    NextIndex is Index + 1,
    build_td3_instance_parts(Rest, NextIndex, RestCases, RestBodies, RestTables).

%% build_td3_instance_table(+Config, +TableName, -TableIR, -GepLen, -EffLen, -MaxAtomId).
%
%  Reads the edge predicate facts from Config, emits a %AtomFactPair
%  private constant under TableName, and returns the sizing info the
%  caller needs for the GEP and the @wam_td3_run call. GepLen is the
%  declared LLVM array size (always ≥ 1 so the type matches the
%  initializer); EffLen is the logical number of edges (may be 0).
build_td3_instance_table(Config, TableName, TableIR, GepLen, EffLen, MaxAtomId) :-
    ( member(edge_pred(EdgePred), Config) -> true
    ; EdgePred = edge/2  % sensible default for smoke tests
    ),
    EdgePred = EPName/EPArity,
    ( EPArity =:= 2 -> true
    ; throw(foreign_lowering_edge_pred_arity(EPName/EPArity))
    ),
    catch(
        findall(fact(From, To),
            ( Goal =.. [EPName, From, To],
              user:Goal
            ),
            Pairs),
        _, Pairs = []),
    compute_max_atom_id(Pairs, MaxAtomId),
    llvm_emit_atom_fact2_table(TableName, Pairs, TableIR),
    length(Pairs, Len),
    ( Len == 0 -> EffLen = 0, GepLen = 1 ; EffLen = Len, GepLen = Len ).

%% build_cds2_instance_switch(+Entries, -ImplBody, -TablesIR).
%
%  countdown_sum2 is pure arithmetic — no per-instance tables. Every
%  case calls the same @wam_cds2_run helper. TablesIR is always ''.
build_cds2_instance_switch(Entries, ImplBody, '') :-
    build_cds2_instance_parts(Entries, 0, SwitchCases, CaseBodies),
    atomic_list_concat(SwitchCases, '\n', SwitchCasesStr),
    atomic_list_concat(CaseBodies, '\n\n', CaseBodiesStr),
    format(atom(ImplBody),
'define i1 @wam_cds2_kernel_impl(%WamState* %vm, i32 %instance) {
entry:
  switch i32 %instance, label %cds_bail [
~w
  ]

~w

cds_bail:
  ret i1 false
}',
        [SwitchCasesStr, CaseBodiesStr]).

build_cds2_instance_parts([], _, [], []).
build_cds2_instance_parts([_PredArity-_Config | Rest], Index,
        [SwitchCase|RestCases], [Body|RestBodies]) :-
    format(atom(SwitchCase), '    i32 ~w, label %cds_inst_~w', [Index, Index]),
    format(atom(Body),
'cds_inst_~w:
  %cds_r_~w = call i1 @wam_cds2_run(%WamState* %vm)
  ret i1 %cds_r_~w',
        [Index, Index, Index]),
    NextIndex is Index + 1,
    build_cds2_instance_parts(Rest, NextIndex, RestCases, RestBodies).

%% replace_cds2_weak_default(+StateFuncsRaw, +NewBody, -StateFuncs).
replace_cds2_weak_default(StateFuncsRaw, NewBody, StateFuncs) :-
    Old = 'define weak i1 @wam_cds2_kernel_impl(%WamState* %vm, i32 %instance) {\n  ret i1 false\n}',
    ( string_replace(StateFuncsRaw, Old, NewBody, StateFuncs0)
    -> StateFuncs = StateFuncs0
    ;  format(user_error,
        'WARNING: could not find cds2 weak-default in state template; leaving unchanged~n', []),
       StateFuncs = StateFuncsRaw
    ).

%% build_ca_instance_switch(+Entries, -ImplBody, -TablesIR).
%
%  category_ancestor: depth-bounded BFS. Each instance has its own
%  edge table and max_depth. Calls @wam_ca_run with per-instance params.
build_ca_instance_switch(Entries, ImplBody, TablesIR) :-
    build_ca_instance_parts(Entries, 0, SwitchCases, CaseBodies, Tables),
    atomic_list_concat(SwitchCases, '\n', SwitchCasesStr),
    atomic_list_concat(CaseBodies, '\n\n', CaseBodiesStr),
    atomic_list_concat(Tables, '\n\n', TablesIR),
    format(atom(ImplBody),
'define i1 @wam_ca_kernel_impl(%WamState* %vm, i32 %instance) {
entry:
  switch i32 %instance, label %ca_bail [
~w
  ]

~w

ca_bail:
  ret i1 false
}',
        [SwitchCasesStr, CaseBodiesStr]).

build_ca_instance_parts([], _, [], [], []).
build_ca_instance_parts([PredArity-Config | Rest], Index,
        [SwitchCase|RestCases], [Body|RestBodies], [TableIR|RestTables]) :-
    PredArity = Pred/_,
    sanitize_atom_for_llvm(Pred, SanePred),
    format(atom(TableName), 'ca_inst_~w_~w_edges', [SanePred, Index]),
    build_td3_instance_table(Config, TableName, TableIR, GepLen, EffLen, MaxAtomId),
    % Extract max_depth from config (default 10).
    ( memberchk(max_depth(MaxDepth), Config) -> true ; MaxDepth = 10 ),
    format(atom(SwitchCase), '    i32 ~w, label %ca_inst_~w', [Index, Index]),
    format(atom(Body),
'ca_inst_~w:
  %ca_tbl_~w = getelementptr [~w x %AtomFactPair], [~w x %AtomFactPair]* @~w, i64 0, i64 0
  %ca_r_~w = call i1 @wam_ca_run(%WamState* %vm, %AtomFactPair* %ca_tbl_~w, i64 ~w, i64 ~w, i32 ~w)
  ret i1 %ca_r_~w',
        [Index, Index, GepLen, GepLen, TableName, Index, Index, EffLen, MaxAtomId, MaxDepth, Index]),
    NextIndex is Index + 1,
    build_ca_instance_parts(Rest, NextIndex, RestCases, RestBodies, RestTables).

%% replace_ca_weak_default(+StateFuncsRaw, +NewBody, -StateFuncs).
replace_ca_weak_default(StateFuncsRaw, NewBody, StateFuncs) :-
    Old = 'define weak i1 @wam_ca_kernel_impl(%WamState* %vm, i32 %instance) {\n  ret i1 false\n}',
    ( string_replace(StateFuncsRaw, Old, NewBody, StateFuncs0)
    -> StateFuncs = StateFuncs0
    ;  format(user_error,
        'WARNING: could not find ca weak-default in state template; leaving unchanged~n', []),
       StateFuncs = StateFuncsRaw
    ).

%% build_ls2_instance_switch(+Entries, -ImplBody, -TablesIR).
%
%  list_suffix2 is stateless (no edge tables) — every instance calls
%  the same @wam_ls2_run. The switch exists for multi-instance pattern
%  consistency. TablesIR is always empty.
build_ls2_instance_switch(Entries, ImplBody, '') :-
    build_ls2_instance_parts(Entries, 0, SwitchCases, CaseBodies),
    atomic_list_concat(SwitchCases, '\n', SwitchCasesStr),
    atomic_list_concat(CaseBodies, '\n\n', CaseBodiesStr),
    format(atom(ImplBody),
'define i1 @wam_ls2_kernel_impl(%WamState* %vm, i32 %instance) {
entry:
  switch i32 %instance, label %ls_bail [
~w
  ]

~w

ls_bail:
  ret i1 false
}',
        [SwitchCasesStr, CaseBodiesStr]).

build_ls2_instance_parts([], _, [], []).
build_ls2_instance_parts([_PredArity-_Config | Rest], Index,
        [SwitchCase|RestCases], [Body|RestBodies]) :-
    format(atom(SwitchCase), '    i32 ~w, label %ls_inst_~w', [Index, Index]),
    format(atom(Body),
'ls_inst_~w:
  %ls_r_~w = call i1 @wam_ls2_run(%WamState* %vm)
  ret i1 %ls_r_~w',
        [Index, Index, Index]),
    NextIndex is Index + 1,
    build_ls2_instance_parts(Rest, NextIndex, RestCases, RestBodies).

%% replace_ls2_weak_default(+StateFuncsRaw, +NewBody, -StateFuncs).
replace_ls2_weak_default(StateFuncsRaw, NewBody, StateFuncs) :-
    Old = 'define weak i1 @wam_ls2_kernel_impl(%WamState* %vm, i32 %instance) {\n  ret i1 false\n}',
    ( string_replace(StateFuncsRaw, Old, NewBody, StateFuncs0)
    -> StateFuncs = StateFuncs0
    ;  format(user_error,
        'WARNING: could not find ls2 weak-default in state template; leaving unchanged~n', []),
       StateFuncs = StateFuncsRaw
    ).

%% build_tc2_instance_switch(+Entries, -ImplBody, -TablesIR).
%
%  Emits @wam_tc2_kernel_impl with a switch dispatching each instance
%  to its own %AtomFactPair edge table and a call to @wam_tc2_run.
%  Reuses build_td3_instance_table for the table emission since tc2
%  and td3 both use edge_pred/2 → %AtomFactPair.
build_tc2_instance_switch(Entries, ImplBody, TablesIR) :-
    build_tc2_instance_parts(Entries, 0, SwitchCases, CaseBodies, Tables),
    atomic_list_concat(SwitchCases, '\n', SwitchCasesStr),
    atomic_list_concat(CaseBodies, '\n\n', CaseBodiesStr),
    atomic_list_concat(Tables, '\n\n', TablesIR),
    format(atom(ImplBody),
'define i1 @wam_tc2_kernel_impl(%WamState* %vm, i32 %instance) {
entry:
  switch i32 %instance, label %tc_bail [
~w
  ]

~w

tc_bail:
  ret i1 false
}',
        [SwitchCasesStr, CaseBodiesStr]).

build_tc2_instance_parts([], _, [], [], []).
build_tc2_instance_parts([PredArity-Config | Rest], Index,
        [SwitchCase|RestCases], [Body|RestBodies], [TableIR|RestTables]) :-
    PredArity = Pred/_,
    sanitize_atom_for_llvm(Pred, SanePred),
    format(atom(TableName), 'tc2_inst_~w_~w_edges', [SanePred, Index]),
    build_td3_instance_table(Config, TableName, TableIR, GepLen, EffLen, MaxAtomId),
    format(atom(SwitchCase), '    i32 ~w, label %tc_inst_~w', [Index, Index]),
    format(atom(Body),
'tc_inst_~w:
  %tc_tbl_~w = getelementptr [~w x %AtomFactPair], [~w x %AtomFactPair]* @~w, i64 0, i64 0
  %tc_r_~w = call i1 @wam_tc2_run(%WamState* %vm, %AtomFactPair* %tc_tbl_~w, i64 ~w, i64 ~w)
  ret i1 %tc_r_~w',
        [Index, Index, GepLen, GepLen, TableName, Index, Index, EffLen, MaxAtomId, Index]),
    NextIndex is Index + 1,
    build_tc2_instance_parts(Rest, NextIndex, RestCases, RestBodies, RestTables).

%% replace_tc2_weak_default(+StateFuncsRaw, +NewBody, -StateFuncs).
replace_tc2_weak_default(StateFuncsRaw, NewBody, StateFuncs) :-
    Old = 'define weak i1 @wam_tc2_kernel_impl(%WamState* %vm, i32 %instance) {\n  ret i1 false\n}',
    ( string_replace(StateFuncsRaw, Old, NewBody, StateFuncs0)
    -> StateFuncs = StateFuncs0
    ;  format(user_error,
        'WARNING: could not find tc2 weak-default in state template; leaving unchanged~n', []),
       StateFuncs = StateFuncsRaw
    ).

%% build_wsp3_instance_switch(+Entries, -ImplBody, -TablesIR).
%
%  M5.9 counterpart of build_td3_instance_switch. Emits one
%  `define i1 @wam_wsp3_kernel_impl(...)` with a switch dispatching
%  each instance to its own %WeightedFact edge table and a call to
%  @wam_wsp3_run.
build_wsp3_instance_switch(Entries, ImplBody, TablesIR) :-
    build_wsp3_instance_parts(Entries, 0, SwitchCases, CaseBodies, Tables),
    atomic_list_concat(SwitchCases, '\n', SwitchCasesStr),
    atomic_list_concat(CaseBodies, '\n\n', CaseBodiesStr),
    atomic_list_concat(Tables, '\n\n', TablesIR),
    format(atom(ImplBody),
'define i1 @wam_wsp3_kernel_impl(%WamState* %vm, i32 %instance) {
entry:
  switch i32 %instance, label %wsp_bail [
~w
  ]

~w

wsp_bail:
  ret i1 false
}',
        [SwitchCasesStr, CaseBodiesStr]).

build_wsp3_instance_parts([], _, [], [], []).
build_wsp3_instance_parts([PredArity-Config | Rest], Index,
        [SwitchCase|RestCases], [Body|RestBodies], [TableIR|RestTables]) :-
    PredArity = Pred/_,
    sanitize_atom_for_llvm(Pred, SanePred),
    format(atom(TableName), 'wsp3_inst_~w_~w_edges', [SanePred, Index]),
    build_wsp3_instance_table(Config, TableName, TableIR, GepLen, EffLen, MaxAtomId),
    format(atom(SwitchCase), '    i32 ~w, label %wsp_inst_~w', [Index, Index]),
    format(atom(Body),
'wsp_inst_~w:
  %wsp_tbl_~w = getelementptr [~w x %WeightedFact], [~w x %WeightedFact]* @~w, i64 0, i64 0
  %wsp_r_~w = call i1 @wam_wsp3_run(%WamState* %vm, %WeightedFact* %wsp_tbl_~w, i64 ~w, i64 ~w)
  ret i1 %wsp_r_~w',
        [Index, Index, GepLen, GepLen, TableName, Index, Index, EffLen, MaxAtomId, Index]),
    NextIndex is Index + 1,
    build_wsp3_instance_parts(Rest, NextIndex, RestCases, RestBodies, RestTables).

%% build_wsp3_instance_table(+Config, +TableName, -TableIR, -GepLen, -EffLen, -MaxAtomId).
%
%  Reads the weight predicate clauses from Config, emits a
%  %WeightedFact private constant under TableName, and returns the
%  sizing info the caller needs for the GEP and the @wam_wsp3_run
%  call. The weight predicate has arity 3: weight_pred(From, To, Weight).
build_wsp3_instance_table(Config, TableName, TableIR, GepLen, EffLen, MaxAtomId) :-
    ( member(weight_pred(WeightPred), Config) -> true
    ; WeightPred = weight/3  % default for smoke tests
    ),
    WeightPred = WPName/WPArity,
    ( WPArity =:= 3 -> true
    ; throw(foreign_lowering_weight_pred_arity(WPName/WPArity))
    ),
    catch(
        findall(edge(From, To, Weight),
            ( Goal =.. [WPName, From, To, Weight],
              user:Goal
            ),
            Triples),
        _, Triples = []),
    compute_max_atom_id_weighted(Triples, MaxAtomId),
    llvm_emit_weighted_edge_table(TableName, Triples, TableIR),
    length(Triples, Len),
    ( Len == 0 -> EffLen = 0, GepLen = 1 ; EffLen = Len, GepLen = Len ).

%% compute_max_atom_id_weighted(+Triples, -MaxAtomId).
%  Like compute_max_atom_id/2 but for edge(From, To, Weight) terms.
%  Weight is ignored for the max-id bound.
compute_max_atom_id_weighted([], 0).
compute_max_atom_id_weighted(Triples, MaxAtomId) :-
    findall(Id,
        ( member(edge(From, To, _), Triples),
          ( intern_atom(From, Id)
          ; intern_atom(To, Id)
          )
        ),
        Ids),
    ( Ids == []
    -> MaxAtomId = 0
    ;  max_list(Ids, MaxAtomId)
    ).

%% replace_wsp3_weak_default(+StateFuncsRaw, +NewBody, -StateFuncs).
replace_wsp3_weak_default(StateFuncsRaw, NewBody, StateFuncs) :-
    Old = 'define weak i1 @wam_wsp3_kernel_impl(%WamState* %vm, i32 %instance) {\n  ret i1 false\n}',
    ( string_replace(StateFuncsRaw, Old, NewBody, StateFuncs0)
    -> StateFuncs = StateFuncs0
    ;  format(user_error,
        'WARNING: M5.9 could not find wsp3 weak-default in state template; leaving unchanged~n', []),
       StateFuncs = StateFuncsRaw
    ).

%% build_astar4_instance_switch(+Entries, -ImplBody, -TablesIR).
%
%  M5.10 counterpart for astar_shortest_path4. Each per-instance case
%  emits a %WeightedFact edge table (same as wsp3) PLUS a zeroed
%  heuristic double[] array. The zero heuristic makes A* degenerate
%  to Dijkstra, validating the full A* pipeline without requiring a
%  target-dependent heuristic mechanism.
build_astar4_instance_switch(Entries, ImplBody, TablesIR) :-
    build_astar4_instance_parts(Entries, 0, SwitchCases, CaseBodies, Tables),
    atomic_list_concat(SwitchCases, '\n', SwitchCasesStr),
    atomic_list_concat(CaseBodies, '\n\n', CaseBodiesStr),
    atomic_list_concat(Tables, '\n\n', TablesIR),
    format(atom(ImplBody),
'define i1 @wam_astar4_kernel_impl(%WamState* %vm, i32 %instance) {
entry:
  switch i32 %instance, label %as_bail [
~w
  ]

~w

as_bail:
  ret i1 false
}',
        [SwitchCasesStr, CaseBodiesStr]).

build_astar4_instance_parts([], _, [], [], []).
build_astar4_instance_parts([PredArity-Config | Rest], Index,
        [SwitchCase|RestCases], [Body|RestBodies], [AllTablesIR|RestTables]) :-
    PredArity = Pred/_,
    sanitize_atom_for_llvm(Pred, SanePred),
    format(atom(EdgeTableName), 'astar4_inst_~w_~w_edges', [SanePred, Index]),
    format(atom(HeuristicName), 'astar4_inst_~w_~w_heuristic', [SanePred, Index]),
    build_wsp3_instance_table(Config, EdgeTableName, EdgeTableIR, GepLen, EffLen, MaxAtomId),
    % Check if direct_dist_pred is configured for runtime heuristic.
    ( member(direct_dist_pred(DDPred), Config)
    -> % Runtime heuristic: emit a %WeightedFact table for direct distances
       % and build the heuristic dynamically at query time.
       format(atom(DDTableName), 'astar4_inst_~w_~w_direct', [SanePred, Index]),
       build_wsp3_instance_table(
           [weight_pred(DDPred)], DDTableName, DDTableIR, DDGepLen, DDEffLen, DDMaxAtomId),
       MaxAtomIdAll is max(MaxAtomId, DDMaxAtomId),
       format(atom(AllTablesIR), '~w\n~w', [EdgeTableIR, DDTableIR]),
       format(atom(SwitchCase), '    i32 ~w, label %as_inst_~w', [Index, Index]),
       format(atom(Body),
'as_inst_~w:
  %as_tbl_~w = getelementptr [~w x %WeightedFact], [~w x %WeightedFact]* @~w, i64 0, i64 0
  %as_dtbl_~w = getelementptr [~w x %WeightedFact], [~w x %WeightedFact]* @~w, i64 0, i64 0
  %as_target_~w = call i64 @wam_get_reg_payload(%WamState* %vm, i32 1)
  %as_h_~w = call double* @wam_build_heuristic_from_table(
      %WeightedFact* %as_dtbl_~w, i64 ~w,
      i64 %as_target_~w, i64 ~w)
  %as_r_~w = call i1 @wam_astar4_run(%WamState* %vm, %WeightedFact* %as_tbl_~w, i64 ~w, double* %as_h_~w, i64 ~w)
  %as_h_raw_~w = bitcast double* %as_h_~w to i8*
  call void @free(i8* %as_h_raw_~w)
  ret i1 %as_r_~w',
           [Index,
            Index, GepLen, GepLen, EdgeTableName,
            Index, DDGepLen, DDGepLen, DDTableName,
            Index,
            Index, Index, DDEffLen,
            Index, MaxAtomIdAll,
            Index, Index, EffLen, Index, MaxAtomIdAll,
            Index, Index, Index, Index])
    ;  % Static heuristic (compile-time, possibly zero).
       build_astar4_heuristic_array(Config, MaxAtomId, HeuristicName,
           HeuristicIR, HeuristicArrSize),
       format(atom(AllTablesIR), '~w\n~w', [EdgeTableIR, HeuristicIR]),
       format(atom(SwitchCase), '    i32 ~w, label %as_inst_~w', [Index, Index]),
       format(atom(Body),
'as_inst_~w:
  %as_tbl_~w = getelementptr [~w x %WeightedFact], [~w x %WeightedFact]* @~w, i64 0, i64 0
  %as_h_~w = getelementptr [~w x double], [~w x double]* @~w, i64 0, i64 0
  %as_r_~w = call i1 @wam_astar4_run(%WamState* %vm, %WeightedFact* %as_tbl_~w, i64 ~w, double* %as_h_~w, i64 ~w)
  ret i1 %as_r_~w',
           [Index,
            Index, GepLen, GepLen, EdgeTableName,
            Index, HeuristicArrSize, HeuristicArrSize, HeuristicName,
            Index, Index, EffLen, Index, MaxAtomId,
            Index])
    ),
    NextIndex is Index + 1,
    build_astar4_instance_parts(Rest, NextIndex, RestCases, RestBodies, RestTables).

%% build_astar4_heuristic_array(+Config, +MaxAtomId, +Name, -IR, -Size).
%
%  When the config contains `heuristic_pred(HPred/3)` and
%  `heuristic_target(TargetAtom)`, reads facts of the form
%  `HPred(Node, TargetAtom, HValue)` from the user module, interns
%  each Node to get its atom ID, and emits a `[Size x double]` global
%  constant where entry `i` = h(atom_id_i). Entries for IDs that
%  don't appear in the heuristic facts default to 0.0 (admissible
%  since h=0 never overestimates).
%
%  When the config does NOT contain heuristic_pred/heuristic_target,
%  emits a zeroinitializer (all zeros — A* degenerates to Dijkstra).
build_astar4_heuristic_array(Config, MaxAtomId, Name, IR, Size) :-
    Size0 is MaxAtomId + 1,
    ( Size0 =< 0 -> Size = 1 ; Size = Size0 ),
    (   member(heuristic_pred(HPred), Config),
        member(heuristic_target(Target), Config)
    ->  % Read heuristic facts for the fixed target.
        HPred = HPName/HPArity,
        ( HPArity =:= 3 -> true
        ; throw(heuristic_pred_arity(HPName/HPArity))
        ),
        catch(
            findall(AtomId-HVal,
                ( Goal =.. [HPName, Node, Target, HVal],
                  user:Goal,
                  atom(Node),
                  number(HVal),
                  intern_atom(Node, AtomId)
                ),
                HEntries),
            _, HEntries = []),
        % Build the array: Size entries, default 0.0, overridden by HEntries.
        numlist(0, MaxAtomId, Indices),
        maplist(heuristic_entry_value(HEntries), Indices, Values),
        maplist(format_double_entry, Values, ValueStrs),
        atomic_list_concat(ValueStrs, ', ', ValuesStr),
        format(atom(IR),
            '@~w = private constant [~w x double] [~w]',
            [Name, Size, ValuesStr])
    ;   % No heuristic config — zero array.
        format(atom(IR),
            '@~w = private constant [~w x double] zeroinitializer',
            [Name, Size])
    ).

heuristic_entry_value(HEntries, AtomId, Value) :-
    ( memberchk(AtomId-V, HEntries) -> Value = V ; Value = 0.0 ).

format_double_entry(V, Str) :-
    ( integer(V)
    -> format(atom(Str), 'double ~w.0', [V])
    ;  format(atom(Str), 'double ~w', [V])
    ).

%% replace_astar4_weak_default(+StateFuncsRaw, +NewBody, -StateFuncs).
replace_astar4_weak_default(StateFuncsRaw, NewBody, StateFuncs) :-
    Old = 'define weak i1 @wam_astar4_kernel_impl(%WamState* %vm, i32 %instance) {\n  ret i1 false\n}',
    ( string_replace(StateFuncsRaw, Old, NewBody, StateFuncs0)
    -> StateFuncs = StateFuncs0
    ;  format(user_error,
        'WARNING: M5.10 could not find astar4 weak-default in state template; leaving unchanged~n', []),
       StateFuncs = StateFuncsRaw
    ).

%% sanitize_atom_for_llvm(+Atom, -Sanitized).
%  Replace any characters that would be awkward in an LLVM global
%  identifier with underscores. Most atoms used as predicate names
%  are alphanumeric + underscore already, so this is a belt-and-braces
%  measure for edge cases.
sanitize_atom_for_llvm(Atom, Sanitized) :-
    atom_codes(Atom, Codes),
    maplist(sanitize_code, Codes, SaneCodes),
    atom_codes(Sanitized, SaneCodes).

sanitize_code(C, C) :-
    ( (C >= 0'a, C =< 0'z) ; (C >= 0'A, C =< 0'Z)
    ; (C >= 0'0, C =< 0'9) ; C =:= 0'_ ), !.
sanitize_code(_, 0'_).

%% compute_max_atom_id(+Pairs, -MaxAtomId).
%
%  Interns every atom in a list of fact(From, To) pairs and returns
%  the highest resulting ID. Empty list → 0.
compute_max_atom_id([], 0).
compute_max_atom_id(Pairs, MaxAtomId) :-
    findall(Id,
        ( member(fact(From, To), Pairs),
          ( intern_atom(From, Id)
          ; intern_atom(To, Id)
          )
        ),
        Ids),
    ( Ids == []
    -> MaxAtomId = 0
    ;  max_list(Ids, MaxAtomId)
    ).

%% replace_td3_weak_default(+StateFuncsRaw, +NewBody, -StateFuncs).
%
%  Replaces the M5.5 weak default of @wam_td3_kernel_impl with NewBody
%  in the rendered state template. If the exact weak-default fragment
%  isn't found (template drift, etc.) the original text is returned
%  unchanged and an error is logged — the M5.5 delegation test would
%  have caught any drift in the weak-default fragment itself.
%
%  M5.8: the weak default now takes an i32 %instance parameter so
%  that pure-WAM modules and foreign-lowered modules share the same
%  dispatcher signature. The string matched here must stay in sync
%  with the state.ll.mustache definition.
replace_td3_weak_default(StateFuncsRaw, NewBody, StateFuncs) :-
    Old = 'define weak i1 @wam_td3_kernel_impl(%WamState* %vm, i32 %instance) {\n  ret i1 false\n}',
    ( string_replace(StateFuncsRaw, Old, NewBody, StateFuncs0)
    -> StateFuncs = StateFuncs0
    ;  format(user_error,
        'WARNING: M5.6 could not find td3 weak-default in state template; leaving unchanged~n', []),
       StateFuncs = StateFuncsRaw
    ).

%% string_replace(+Haystack, +Needle, +Replacement, -Result) is semidet.
%
%  Simple substring replacement. Fails if Needle is not found.
string_replace(Haystack, Needle, Replacement, Result) :-
    ( atom(Haystack) -> atom_string(Haystack, HayStr) ; HayStr = Haystack ),
    ( atom(Needle)   -> atom_string(Needle, NeedleStr) ; NeedleStr = Needle ),
    ( atom(Replacement) -> atom_string(Replacement, ReplStr) ; ReplStr = Replacement ),
    sub_string(HayStr, Before, _, After, NeedleStr), !,
    sub_string(HayStr, 0, Before, _, Prefix),
    string_length(HayStr, HayLen),
    SuffixStart is HayLen - After,
    sub_string(HayStr, SuffixStart, After, 0, Suffix),
    string_concat(Prefix, ReplStr, Tmp),
    string_concat(Tmp, Suffix, Result).

% ============================================================================
% PHASE 5: Hybrid Module Assembly
% ============================================================================

%% write_wam_llvm_project(+Predicates, +Options, +OutputFile) is det.
%
%  Generates a complete LLVM IR module for the given predicates.
%  Deterministic by contract: the body's compilation pipeline leaves
%  internal choice points, and a caller that backtracks into them
%  (e.g. a test negating a goal that contains this call) re-runs the
%  whole compiler search -- one such \+ burned 45 CPU-minutes before
%  this fence existed.
write_wam_llvm_project(Predicates, Options, OutputFile) :-
    once(write_wam_llvm_project_(Predicates, Options, OutputFile)).

write_wam_llvm_project_(Predicates, Options, OutputFile) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    option(target_triple(Triple), Options, 'x86_64-pc-linux-gnu'),
    option(target_datalayout(DataLayout), Options,
        'e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128'),
    option(generated_date(Date), Options, 'reproducible'),

    % M5.6: populate the foreign kernel spec table.
    % Path (a) directives have already run by load time. Path (b) is
    % the options-list form; path (c) walks clause shapes when
    % foreign_lowering(true) is set.
    apply_foreign_predicates_option(Options),
    ( option(foreign_lowering(true), Options)
    -> llvm_auto_detect_foreign_kernels(Predicates)
    ;  true
    ),

    % Read and render type definitions template
    read_template_file('templates/targets/llvm_wam/types.ll.mustache', TypesTemplate),
    render_template(TypesTemplate, [module_name=ModuleName, date=Date], TypesDef),

    % Read and render value functions template
    read_template_file('templates/targets/llvm_wam/value.ll.mustache', ValueTemplate),
    render_template(ValueTemplate, [], ValueFuncs),

    % Read and render state functions template. When foreign lowering
    % is active, splice concrete kernel impl bodies in place of the
    % weak defaults in the rendered state template.
    read_template_file('templates/targets/llvm_wam/state.ll.mustache', StateTemplate),
    render_template(StateTemplate, [], StateFuncsRaw),
    substitute_foreign_kernel_impls(StateFuncsRaw, StateFuncs, ForeignGlobals),

    % Generate runtime (step + helpers)
    compile_step_wam_to_llvm(Options, StepFunc),
    compile_wam_helpers_to_llvm(Options, HelpersCode),
    read_template_file('templates/targets/llvm_wam/runtime.ll.mustache', RuntimeTemplate),
    render_template(RuntimeTemplate, [
        step_function=StepFunc,
        helper_functions=HelpersCode
    ], RuntimeFuncs),

    % Compile predicates (native or WAM fallback). Any predicate with a
    % llvm_foreign_kernel_spec/3 entry is compiled to a body of
    % WAM instructions ending in `call_foreign Kind, Arity`.
    retractall(functor_string_global(_, _)),
    % M105: also reset the atom-table dynamic predicates per module.
    % Atom ids are module-scoped (each module emits its own
    % @wam_atom_strings global) so persistent state across compiles
    % only serves to accumulate. The 600+ test suite hit Prolog''s
    % 4GB global-stack cap around test 650 because every compile
    % re-interns 95 ASCII chars + [] + a handful of named atoms,
    % and those atom_table_entry asserts piled up monotonically.
    retractall(atom_table_entry(_, _)),
    retractall(atom_table_next_id(_)),
    assertz(atom_table_next_id(1)),
    % Pre-register functor globals referenced directly by runtime
    % helpers (e.g. =../2 needs the cons-cell functor "." and the
    % empty-list atom "[]"). Route through register_functor_string so
    % the key type (string) matches the bytecode-side registration —
    % otherwise a program that BOTH uses findall AND explicitly
    % constructs cons cells emits two `@.fn__5B_7C_5D` globals and
    % llc rejects the module ("redefinition of global").
    register_functor_string("."),
    register_functor_string("[]"),
    % M9: cons-cell functor for the findall / aggregate_all(bag/set)
    % result list. @wam_apply_aggregation's collect_case allocates
    % `[|]`-tagged Compounds on the arena to build the result; the
    % functor string global it references must exist or llvm-as rejects
    % the module. Eager registration handles programs that consume the
    % list via findall but never explicitly construct a cons cell.
    register_functor_string("[|]"),
    % M51: pre-register `-` so pairs_keys_values/3 reverse mode can
    % construct K-V pair compounds without depending on the source
    % program to mention `-/2` itself.
    register_functor_string("-"),
    % M58: pre-register the type names char_type/2 dispatches on so
    % their @.fn_<name> globals exist for strcmp comparisons.
    register_functor_string("alpha"),
    register_functor_string("alnum"),
    register_functor_string("digit"),
    register_functor_string("upper"),
    register_functor_string("lower"),
    register_functor_string("space"),
    register_functor_string("ascii"),
    register_functor_string("white"),
    register_functor_string("punct"),
    register_functor_string("csym"),
    register_functor_string("csymf"),
    register_functor_string("newline"),
    register_functor_string("end_of_line"),
    % M60: must_be/2 type names. Some overlap with M58 char_type
    % (alpha etc.) but distinct names need separate registration.
    register_functor_string("atom"),
    register_functor_string("integer"),
    register_functor_string("float"),
    register_functor_string("number"),
    register_functor_string("compound"),
    register_functor_string("var"),
    register_functor_string("nonvar"),
    register_functor_string("atomic"),
    register_functor_string("callable"),
    register_functor_string("ground"),
    register_functor_string("list"),
    register_functor_string("boolean"),
    % M83: arithmetic atom constants. ``X is pi'' and ``X is e''
    % resolve to Float(pi) and Float(e) in eval_arith_value via
    % strcmp against these registered names.
    register_functor_string("pi"),
    register_functor_string("e"),
    % M97: set_random(seed(N)) -- builtin_set_random does strcmp on
    % the compound''s functor string against "seed". Pre-register so
    % @.fn_seed is in the module even for programs that never
    % explicitly construct a `seed/1' compound elsewhere.
    register_functor_string("seed"),
    % M98: stamp_date_time/3 builds a `date(...)' compound and uses
    % a `local' atom for the TZName slot. Pre-register both so their
    % @.fn_... globals are in every module.
    register_functor_string("date"),
    register_functor_string("local"),
    % M105: numbervars/3 binds free vars to ''$VAR''(N) compounds.
    % Pre-register so @.fn__24VAR exists in every module.
    register_functor_string("$VAR"),
    % M9: intern "[]" so the empty-list atom id is known at runtime.
    % @wam_apply_aggregation's collect_case reads this id from a
    % module-level global to terminate the cons-cell chain.
    intern_atom('[]', EmptyListAtomId),
    format(atom(EmptyListGlobal),
'; M9: empty-list atom id for the findall / aggregate_all(bag/set)
; collect path. Read by @wam_apply_aggregation when building the
; result cons-cell chain (the tail of the final cell is an Atom
; %Value with payload = this id).
@wam_empty_list_atom_id = private constant i64 ~w',
        [EmptyListAtomId]),
    % M26: eagerly intern single-character atoms for the printable
    % ASCII range (32..126). char_code/2 and atom_chars/2 use these
    % via the @wam_char_to_atom_id lookup table emitted alongside
    % the regular atom-string globals. Reserving these slots now
    % means the per-module table is dense and the runtime doesn''t
    % need to fall back to runtime atom interning for any printable
    % character.
    forall(between(32, 126, CharCode),
        ( char_code(CharAtom, CharCode),
          intern_atom(CharAtom, _)
        )),
    compile_predicates_for_llvm(Predicates, Options, NativeCode, WamCode),

    % Emit functor string globals collected during WAM compilation.
    emit_functor_string_globals(FunctorGlobals),

    % M12: emit per-atom string globals + lookup table so format/2,
    % write/1 of atoms, and other runtime code can resolve interned
    % atom ids back to printable strings.
    emit_atom_string_globals(AtomStringGlobals),

    % Prepend foreign kernel globals + functor strings to the native
    % predicates section so they are at module scope before any use.
    atomic_list_concat([ForeignGlobals, '\n', FunctorGlobals, '\n',
        EmptyListGlobal, '\n', AtomStringGlobals, '\n\n', NativeCode],
        FinalNativeCode0),
    % Phase 5 (JIT): append the .wamo loader + object-call primitive when
    % requested. It references only runtime helpers already in this module
    % plus libc open/read/close, so it can ride any host module.
    (   option(emit_wamo_loader(true), Options)
    ->  wam_object_support_ir(WamoLoaderIR),
        atomic_list_concat([FinalNativeCode0, '\n\n', WamoLoaderIR],
            FinalNativeCode)
    ;   FinalNativeCode = FinalNativeCode0
    ),

    % Generate external declarations (native vs WASM).
    generate_external_declarations(Triple, ExternalDecls),

    % M116: stream the module IR straight to the output file rather
    % than calling render_template/3 to build one multi-MB Prolog
    % atom and then writing it in one shot. The single big atom was
    % the dominant per-compile allocation that drove the suite into
    % SWI''s 4 GB global-stack cap around test 675.
    read_template_file('templates/targets/llvm_wam/module.ll.mustache', ModuleTemplate),
    Substitutions = [
        module_name=ModuleName,
        date=Date,
        target_datalayout=DataLayout,
        target_triple=Triple,
        type_definitions=TypesDef,
        external_declarations=ExternalDecls,
        value_functions=ValueFuncs,
        state_functions=StateFuncs,
        runtime_functions=RuntimeFuncs,
        native_predicates=FinalNativeCode,
        wam_predicates=WamCode,
        interop_bridge="; none"
    ],
    % Write output file. The generated module embeds UTF-8 characters from
    % the runtime templates (arrows, dashes in comments), so the stream must
    % be UTF-8 regardless of the process locale (which is POSIX/ASCII in many
    % CI containers) -- otherwise format/3 raises an encoding error.
    setup_call_cleanup(
        open(OutputFile, write, Stream, [encoding(utf8)]),
        stream_render_mustache(Stream, ModuleTemplate, Substitutions),
        close(Stream)
    ),
    format('WAM LLVM module created at: ~w~n', [OutputFile]).

% ============================================================================
% PHASE 7: WASM Variant
% ============================================================================

%% write_wam_llvm_wasm_project(+Predicates, +Options, +OutputFile)
%  Generates a WASM-compatible LLVM IR module.
%  Key differences from native: wasm32 triple, bump allocator (no malloc),
%  iterative run_loop (no musttail), i32 pointers.
write_wam_llvm_wasm_project(Predicates, Options, OutputFile) :-
    option(module_name(ModuleName), Options, 'wam_wasm'),
    option(generated_date(Date), Options, 'reproducible'),

    % WASM type definitions
    read_template_file('templates/targets/llvm_wam_wasm/types.ll.mustache', TypesTemplate),
    render_template(TypesTemplate, [module_name=ModuleName, date=Date], TypesDef),

    % Bump allocator (replaces malloc/free)
    read_template_file('templates/targets/llvm_wam_wasm/allocator.ll.mustache', AllocTemplate),
    render_template(AllocTemplate, [], AllocFuncs),

    % Value functions (shared with native — %Value is the same)
    read_template_file('templates/targets/llvm_wam/value.ll.mustache', ValueTemplate),
    render_template(ValueTemplate, [], ValueFuncs),

    % State functions (shared — uses %WamState pointers)
    read_template_file('templates/targets/llvm_wam/state.ll.mustache', StateTemplate),
    render_template(StateTemplate, [], StateFuncs),

    % Runtime with iterative loop (no musttail)
    compile_step_wam_to_llvm(Options, StepFunc),
    compile_wam_helpers_to_llvm(Options, HelpersCode),
    read_template_file('templates/targets/llvm_wam_wasm/runtime.ll.mustache', RuntimeTemplate),
    render_template(RuntimeTemplate, [
        step_function=StepFunc,
        helper_functions=HelpersCode
    ], RuntimeFuncs),

    % Compile predicates
    compile_predicates_for_llvm(Predicates, Options, NativeCode, WamCode),

    % Generate WASM export declarations for WAM-compiled predicates
    generate_wasm_exports(Predicates, WasmExports),

    % Assemble
    read_template_file('templates/targets/llvm_wam_wasm/module.ll.mustache', ModuleTemplate),
    render_template(ModuleTemplate, [
        module_name=ModuleName,
        date=Date,
        type_definitions=TypesDef,
        allocator_functions=AllocFuncs,
        value_functions=ValueFuncs,
        state_functions=StateFuncs,
        runtime_functions=RuntimeFuncs,
        native_predicates=NativeCode,
        wam_predicates=WamCode,
        wasm_exports=WasmExports
    ], FullModule),

    setup_call_cleanup(
        open(OutputFile, write, Stream, [encoding(utf8)]),
        format(Stream, "~w", [FullModule]),
        close(Stream)
    ),
    format('WAM WASM module created at: ~w~n', [OutputFile]).

%% generate_external_declarations(+Triple, -Decls)
%  For native targets: declare malloc, free, snprintf, strcmp, printf.
%  For wasm32: define malloc/free as bump allocator (self-contained),
%  omit snprintf/strcmp/printf (not available in standalone WASM).
%% stream_render_mustache(+Stream, +Template, +Substitutions) is det.
%
%  M116: streaming counterpart to render_template/3 for the
%  module.ll.mustache top-level assembly. Walks the template scanning
%  for {{key}} placeholders, writes each literal segment directly to
%  Stream, looks up Key in Substitutions, writes the corresponding
%  value. No full-IR atom is ever materialised in the Prolog atom
%  table, which is what was driving the per-test SWI-stack pressure.
%
%  Only supports the flat {{key}} form -- the module.ll.mustache
%  template doesn''t use sections or match blocks, so the simpler
%  implementation is sufficient. Falls back to writing the rest of
%  the template literally when no more placeholders are found.
stream_render_mustache(Stream, Template, Subs) :-
    atom_string(Template, S),
    stream_render_loop(Stream, S, Subs).

stream_render_loop(Stream, S, Subs) :-
    ( sub_string(S, Before, 2, _, "{{")
    -> sub_string(S, 0, Before, _, Prefix),
       write(Stream, Prefix),
       AfterOpen is Before + 2,
       sub_string(S, AfterOpen, _, 0, Rest1),
       ( sub_string(Rest1, KeyEnd, 2, _, "}}")
       -> sub_string(Rest1, 0, KeyEnd, _, KeyStr),
          atom_string(Key, KeyStr),
          ( member(Key=Value, Subs)
          -> write(Stream, Value)
          ;  format(Stream, '{{~w}}', [Key])
          ),
          AfterClose is KeyEnd + 2,
          sub_string(Rest1, AfterClose, _, 0, Rest2),
          stream_render_loop(Stream, Rest2, Subs)
       ;  % Unclosed {{ -- write the rest verbatim and stop.
          write(Stream, '{{'),
          write(Stream, Rest1)
       )
    ;  write(Stream, S)
    ).

generate_external_declarations(Triple, Decls) :-
    ( sub_atom(Triple, 0, _, _, wasm32)
    -> Decls = '
; WASM bump allocator providing malloc/free symbols.
; Heap starts at offset 65536 (64KB reserved for WASM stack).
; free() is a no-op — memory is reclaimed by @wam_cleanup via arena rewind.
@wam_malloc_ptr = internal global i64 65536

define i8* @malloc(i64 %size) {
entry:
  %ptr_val = load i64, i64* @wam_malloc_ptr
  %aligned = add i64 %size, 7
  %masked = and i64 %aligned, -8
  %new_ptr = add i64 %ptr_val, %masked
  store i64 %new_ptr, i64* @wam_malloc_ptr
  %result = inttoptr i64 %ptr_val to i8*
  ret i8* %result
}

define void @free(i8* %ptr) {
  ret void
}

; Bump-allocator @realloc for WASM: allocate fresh, memcpy old data,
; return new ptr. Old block is leaked (the bump allocator has no
; freelist). Used by M5 growable trail/stack/CP allocators.
;
; Note: this doesn\'t know the old block\'s size, so it conservatively
; copies up to %new_size bytes. The bump allocator only ever hands
; out blocks aligned to 8 bytes, and grows always double, so this
; over-copies by at most a factor of 2 — within the WASM page budget
; that callers already accept.
define i8* @realloc(i8* %old, i64 %new_size) {
entry:
  %is_null = icmp eq i8* %old, null
  br i1 %is_null, label %fresh_alloc, label %do_copy

fresh_alloc:
  %new0 = call i8* @malloc(i64 %new_size)
  ret i8* %new0

do_copy:
  %new1 = call i8* @malloc(i64 %new_size)
  ; We don\'t know the old block\'s actual size; the WAM grow path
  ; always doubles, so copying %new_size / 2 is safe (the old block
  ; was at least that large). Use shr by 1 to avoid sdiv.
  %half = lshr i64 %new_size, 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %new1, i8* %old, i64 %half, i1 false)
  ret i8* %new1
}

; Stub printf for WASM (no-op, returns 0).
define i32 @printf(i8* %fmt, ...) {
  ret i32 0
}

; Stub snprintf for WASM (no-op, returns 0).
define i32 @snprintf(i8* %buf, i64 %size, i8* %fmt, ...) {
  ret i32 0
}

; Stub strcmp for WASM (returns 0 = equal, conservative).
define i32 @strcmp(i8* %a, i8* %b) {
  ret i32 0
}

; Stub putchar for WASM (no-op, returns 0).
define i32 @putchar(i32 %c) {
  ret i32 0
}
'
    ;  Decls = 'declare i8* @malloc(i64)
declare i8* @realloc(i8*, i64)
declare void @free(i8*)
declare i32 @snprintf(i8*, i64, i8*, ...)
declare i32 @strcmp(i8*, i8*)
declare i32 @printf(i8*, ...)
declare i32 @putchar(i32)
declare i64 @time(i64*)
declare i32 @clock_gettime(i32, i8*)
declare i32 @stat(i8*, i8*)
declare i32 @unlink(i8*)
declare i32 @mkdir(i8*, i32)
declare i32 @rename(i8*, i8*)
declare i32 @rmdir(i8*)
declare i8* @getenv(i8*)
declare i32 @setenv(i8*, i8*, i32)
declare i32 @unsetenv(i8*)
declare i32 @chmod(i8*, i32)
declare i32 @access(i8*, i32)
declare i8* @opendir(i8*)
declare i8* @readdir(i8*)
declare i32 @closedir(i8*)
declare i64 @strlen(i8*)
declare i64 @strnlen(i8*, i64)
declare i32 @system(i8*)
declare i8* @getcwd(i8*, i64)
declare i32 @chdir(i8*)
declare i32 @getpid()
declare i32 @getuid()
declare i32 @geteuid()
declare i32 @getgid()
declare i32 @getegid()
declare i32 @getppid()
declare i32 @getpgrp()
declare i8* @realpath(i8*, i8*)
declare i32 @kill(i32, i32)
declare i32 @truncate(i8*, i64)
declare i32 @chown(i8*, i32, i32)
declare i64 @readlink(i8*, i8*, i64)
declare i32 @symlink(i8*, i8*)
declare i32 @link(i8*, i8*)
declare i32 @mkstemp(i8*)
declare i32 @mkfifo(i8*, i32)
declare i32 @close(i32)
declare i32 @umask(i32)
declare i32 @nice(i32)
declare i32 @getpriority(i32, i32)
declare i32 @setpriority(i32, i32, i32)
declare i32 @getrlimit(i32, i8*)
declare i32 @setrlimit(i32, i8*)
declare i8* @getlogin()
declare i32 @uname(i8*)
declare i32 @open(i8*, i32, i32)
declare i64 @read(i32, i8*, i64)
declare i64 @write(i32, i8*, i64)
declare i32* @__errno_location()
declare i8* @strerror(i32)
declare i32 @getrusage(i32, i8*)
declare i8* @popen(i8*, i8*)
declare i32 @pclose(i8*)
declare i32 @fgetc(i8*)
declare i32 @ferror(i8*)
declare i64 @fread(i8*, i64, i64, i8*)
declare i64 @fwrite(i8*, i64, i64, i8*)
declare i32 @memcmp(i8*, i8*, i64)
declare i8* @strstr(i8*, i8*)
declare i32 @usleep(i32)
declare i32 @gethostname(i8*, i64)
declare double @drand48()
declare i64 @lrand48()
declare void @srand48(i64)
declare i8* @localtime_r(i64*, i8*)
declare i64 @strftime(i8*, i64, i8*, i8*)
declare i64 @mktime(i8*)
declare double @asin(double)
declare double @acos(double)
declare double @atan(double)
declare double @atan2(double, double)
declare double @pow(double, double)
declare double @sqrt(double)
declare double @sin(double)
declare double @cos(double)
declare double @exp(double)
declare double @log(double)
declare i32 @regcomp(i8*, i8*, i32)
declare i32 @regexec(i8*, i8*, i64, i8*, i32)
declare double @strtod(i8*, i8**)

; ---- awk output redirection: print ... > "file" / >> "file" ----
; A redirected print runs between @plawk_redirect_begin(name, append) and
; @plawk_redirect_end(): begin flushes stdio, saves the real stdout once, and
; points fd 1 at the target -- opened ONCE per name (the first > truncates, >>
; appends) and kept open, as awk does; end flushes and restores fd 1. So every
; print emitter works unchanged. /dev/stderr and /dev/stdout are the real
; streams, never reopened (an O_TRUNC reopen could truncate a file stderr is
; itself redirected to). Native only: the WASM branch has no such symbols.
declare i32 @dup(i32)
declare i32 @dup2(i32, i32)
declare i32 @fflush(i8*)
@plawk_redir_saved = internal global i32 -1
@plawk_redir_names = internal global [64 x i8*] zeroinitializer
@plawk_redir_fds = internal global [64 x i32] zeroinitializer
@plawk_redir_count = internal global i64 0
@.plawk_redir_stderr = private constant [12 x i8] c"/dev/stderr\\00"
@.plawk_redir_stdout = private constant [12 x i8] c"/dev/stdout\\00"
@.plawk_redir_error = private constant [55 x i8] c"plawk: fatal: cannot open file for output redirection\\0A\\00"

define void @plawk_redirect_begin(i8* %name, i32 %append) {
entry:
  %rb.fl = call i32 @fflush(i8* null)
  %rb.saved0 = load i32, i32* @plawk_redir_saved
  %rb.need = icmp slt i32 %rb.saved0, 0
  br i1 %rb.need, label %rb.save, label %rb.special

rb.save:
  %rb.s = call i32 @dup(i32 1)
  store i32 %rb.s, i32* @plawk_redir_saved
  br label %rb.special

rb.special:
  %rb.errp = getelementptr [12 x i8], [12 x i8]* @.plawk_redir_stderr, i64 0, i64 0
  %rb.iserr = call i32 @strcmp(i8* %name, i8* %rb.errp)
  %rb.e0 = icmp eq i32 %rb.iserr, 0
  br i1 %rb.e0, label %rb.use_err, label %rb.check_out

rb.use_err:
  %rb.d1 = call i32 @dup2(i32 2, i32 1)
  ret void

rb.check_out:
  %rb.outp = getelementptr [12 x i8], [12 x i8]* @.plawk_redir_stdout, i64 0, i64 0
  %rb.isout = call i32 @strcmp(i8* %name, i8* %rb.outp)
  %rb.o0 = icmp eq i32 %rb.isout, 0
  br i1 %rb.o0, label %rb.use_out, label %rb.head

rb.use_out:
  %rb.sv = load i32, i32* @plawk_redir_saved
  %rb.d2 = call i32 @dup2(i32 %rb.sv, i32 1)
  ret void

rb.head:
  %rb.i = phi i64 [ 0, %rb.check_out ], [ %rb.i1, %rb.next ]
  %rb.cnt = load i64, i64* @plawk_redir_count
  %rb.done = icmp uge i64 %rb.i, %rb.cnt
  br i1 %rb.done, label %rb.open, label %rb.cmp

rb.cmp:
  %rb.np = getelementptr [64 x i8*], [64 x i8*]* @plawk_redir_names, i64 0, i64 %rb.i
  %rb.n = load i8*, i8** %rb.np
  %rb.c = call i32 @strcmp(i8* %name, i8* %rb.n)
  %rb.same = icmp eq i32 %rb.c, 0
  br i1 %rb.same, label %rb.found, label %rb.next

rb.next:
  %rb.i1 = add i64 %rb.i, 1
  br label %rb.head

rb.found:
  %rb.fp = getelementptr [64 x i32], [64 x i32]* @plawk_redir_fds, i64 0, i64 %rb.i
  %rb.fd = load i32, i32* %rb.fp
  %rb.d3 = call i32 @dup2(i32 %rb.fd, i32 1)
  ret void

rb.open:
  ; O_WRONLY|O_CREAT plus O_APPEND (>>) or O_TRUNC (>), mode 0666 (Linux values)
  %rb.app = icmp ne i32 %append, 0
  %rb.flags = select i1 %rb.app, i32 1089, i32 577
  %rb.nfd = call i32 @open(i8* %name, i32 %rb.flags, i32 438)
  %rb.bad = icmp slt i32 %rb.nfd, 0
  br i1 %rb.bad, label %rb.fail, label %rb.remember

rb.fail:
  %rb.msg = getelementptr [55 x i8], [55 x i8]* @.plawk_redir_error, i64 0, i64 0
  %rb.w = call i64 @write(i32 2, i8* %rb.msg, i64 54)
  call void @exit(i32 2)
  unreachable

rb.remember:
  %rb.room = icmp ult i64 %rb.cnt, 64
  br i1 %rb.room, label %rb.store, label %rb.use_new

rb.store:
  %rb.snp = getelementptr [64 x i8*], [64 x i8*]* @plawk_redir_names, i64 0, i64 %rb.cnt
  store i8* %name, i8** %rb.snp
  %rb.sfp = getelementptr [64 x i32], [64 x i32]* @plawk_redir_fds, i64 0, i64 %rb.cnt
  store i32 %rb.nfd, i32* %rb.sfp
  %rb.cnt1 = add i64 %rb.cnt, 1
  store i64 %rb.cnt1, i64* @plawk_redir_count
  br label %rb.use_new

rb.use_new:
  %rb.d4 = call i32 @dup2(i32 %rb.nfd, i32 1)
  ret void
}

define void @plawk_redirect_end() {
entry:
  %re.fl = call i32 @fflush(i8* null)
  %re.sv = load i32, i32* @plawk_redir_saved
  %re.d = call i32 @dup2(i32 %re.sv, i32 1)
  ret void
}'
    ).

%% generate_wasm_exports(+Predicates, -ExportCode)
%  Generate WASM export wrappers that expose predicates as i32-returning
%  functions. Each wrapper delegates to the predicate's own `@<pred>()`
%  function (emitted by compile_wam_predicate_to_llvm for WAM-fallback
%  predicates, or by the native LLVM lowering for native predicates),
%  which has the correct instruction-array + label-table bound.
%  Arena state is rewound via @wam_cleanup after each call so successive
%  bench calls don't leak.
generate_wasm_exports([], "").
generate_wasm_exports(Predicates, ExportCode) :-
    number_predicates(Predicates, 0, NumberedPreds),
    findall(ExportFunc, (
        member(Idx-PredIndicator, NumberedPreds),
        (   PredIndicator = _:Pred/Arity -> true
        ;   PredIndicator = Pred/Arity
        ),
        atom_string(Pred, PredStr),
        build_llvm_undef_args(Arity, ArgsList),
        format(atom(ExportFunc),
'; WASM export: ~w/~w
; Visibility: dso_local ensures symbol is retained by wasm-ld.
define dso_local i32 @~w_wasm() #~w {
entry:
  %result = call i1 @~w(~w)
  call void @wam_cleanup()
  %ret = zext i1 %result to i32
  ret i32 %ret
}', [PredStr, Arity, PredStr, Idx, PredStr, ArgsList])
    ), ExportFuncs),
    atomic_list_concat(ExportFuncs, '\n\n', ExportFuncsStr),
    findall(AttrGroup, (
        member(Idx-PredIndicator, NumberedPreds),
        (   PredIndicator = _:Pred/Arity -> true
        ;   PredIndicator = Pred/Arity
        ),
        atom_string(Pred, PredStr),
        format(atom(AttrGroup),
'attributes #~w = { "wasm-export-name"="~w_wasm" }',
            [Idx, PredStr])
    ), AttrGroups),
    atomic_list_concat(AttrGroups, '\n', AttrGroupsStr),
    format(atom(ExportCode),
'~w

; WASM export attributes (one group per exported wrapper)
~w', [ExportFuncsStr, AttrGroupsStr]).

%% number_predicates(+Preds, +Start, -Numbered)
%  Pair each predicate with a zero-based index used as the LLVM
%  attribute-group id for its WASM export wrapper.
number_predicates([], _, []).
number_predicates([P|Rest], N, [N-P|RestNum]) :-
    N1 is N + 1,
    number_predicates(Rest, N1, RestNum).

%% build_llvm_undef_args(+Arity, -ArgsStr)
%  Produce a comma-separated list of `%Value undef` (Arity copies).
%  Used by bench-style wrappers that just need to invoke the predicate
%  without binding its arguments.
build_llvm_undef_args(0, '') :- !.
build_llvm_undef_args(Arity, ArgsStr) :-
    Arity > 0,
    length(Undefs, Arity),
    maplist(=('%Value undef'), Undefs),
    atomic_list_concat(Undefs, ', ', ArgsStr).

%% build_wam_wasm_module(+LLFile, +OutputName, -Commands)
%  Generate shell commands to build a WASM module from the generated .ll file.
build_wam_wasm_module(LLFile, OutputName, Commands) :-
    format(atom(Commands),
'#!/bin/bash
# Build WAM WASM module from LLVM IR
# Requires: LLVM toolchain with WASM backend:
#   - llc (LLVM static compiler)
#   - wasm-ld (WebAssembly linker, part of lld)
# Install: apt install llvm lld  (or brew install llvm)

set -e

# Toolchain check
for tool in llc wasm-ld; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: $tool not found. Install LLVM + lld." >&2
        exit 1
    fi
done

echo "Compiling ~w to WASM..."

# Step 1: Compile to WASM object (explicit triple for reproducibility)
llc --mtriple=wasm32-unknown-unknown -filetype=obj ~w -o ~w.o

# Step 2: Link to WASM module (library mode, all symbols exported)
wasm-ld --no-entry --export-all --allow-undefined ~w.o -o ~w.wasm

# Step 3: Verify (optional)
if command -v wasm-objdump >/dev/null 2>&1; then
    echo "Exports:"
    wasm-objdump -x ~w.wasm | grep "export" | head -10
fi

echo "Created: ~w.wasm"
echo "Size: $(wc -c < ~w.wasm) bytes"
', [LLFile, LLFile, OutputName, OutputName, OutputName, OutputName, OutputName, OutputName]).

%% read_template_file(+Path, -Content)
% Captured at load time so template paths resolve against the
% repository the module was loaded from, not the process cwd.
:- dynamic wam_llvm_module_dir/1.
:- prolog_load_context(directory, Dir),
   retractall(wam_llvm_module_dir(_)),
   assertz(wam_llvm_module_dir(Dir)).

read_template_file(Path, Content) :-
    (   exists_file(Path)
    ->  read_file_to_string(Path, Content, [])
    ;   wam_llvm_module_dir(ModuleDir),
        format(atom(RootPath), '~w/../../../~w', [ModuleDir, Path]),
        exists_file(RootPath)
    ->  read_file_to_string(RootPath, Content, [])
    ;   % A missing runtime template used to become an IR comment,
        % producing a module full of undefined types that failed later
        % at clang with baffling errors. Fail here instead.
        throw(error(existence_error(source_sink, Path),
            context(wam_llvm_codegen,
                'runtime template not found relative to the working directory or the UnifyWeaver source tree')))
    ).

%% compile_predicates_for_llvm(+Predicates, +Options, -NativeCode, -WamCode)
%
%  Two-pass project-level compilation for WAM-fallback predicates:
%
%    Pass 1 (pass1_classify_predicates/4): classify each predicate as
%      native / foreign-kernel / wam-fallback / failed. For wam-fallback
%      and foreign-kernel preds, parse WAM to raw instruction parts +
%      local labels and accumulate each predicate's cumulative start PC
%      in a merged instruction array.
%
%    Label merge (build_merged_labels/2): shift every local label by
%      its predicate's StartPC and union into a project-wide
%      LabelName-AbsolutePC list. The entry label (e.g. 'sum_ints/3')
%      lives at StartPC so cross-predicate `call sum_ints/3` resolves.
%
%    Pass 2 (pass2_emit_merged/5): re-encode every wam/foreign
%      predicate's instructions against the merged LabelMap, emit ONE
%      @module_code array + ONE @module_labels array, and per-predicate
%      entry functions that share those globals but set their own start
%      PC before calling @run_loop.
%
%  Prior to this, each wam-compiled predicate had its own @<pred>_code
%  and @<pred>_labels arrays. An `execute sum_ints_args/5` inside
%  sum_ints/3 looked up the label in sum_ints/3's local map, didn't
%  find it, defaulted to index 0, and at runtime jumped to sum_ints/3's
%  own first instruction — silent self-recursion. Same class of bug
%  WAT had before PR #1476.
compile_predicates_for_llvm(Predicates, Options, NativeCode, WamCode) :-
    expand_wam_llvm_project_predicates(Predicates, Options, ExpandedPredicates),
    % M4: closure analysis — compute the set of predicates in this
    % batch whose lowered kernels can be safely inter-linked via
    % `call`/`execute`. A predicate joins the closure iff all of its
    % clause-1 call/execute targets are also in the closure (computed
    % to fixpoint). Passed through Options so pass1's per-pred
    % lowerability check can consult it.
    compute_lowered_closure(ExpandedPredicates, Options, ClosureSet),
    % M10: enable inline lowering of bagof/setof so they go through
    % the same aggregate dispatch path as findall/aggregate_all rather
    % than the runtime fallback (which is not wired up for LLVM).
    % The WAM compiler treats bagof/setof as primitive calls when
    % this flag is false; with it true they expand to begin_aggregate
    % bagof/setof, ..., end_aggregate and the LLVM target dispatches
    % to wam_apply_aggregation case 5 (bag/bagof) or 6 (set/setof).
    ( memberchk(inline_bagof_setof(_), Options)
    -> OptionsWithBS = Options
    ;  OptionsWithBS = [inline_bagof_setof(true) | Options]
    ),
    % M17: emit if-then-else with get_level Y_n / cut Y_n bytecode
    % rather than the naive cut_ite. The LLVM target''s cut_ite was
    % a "decrement cp_count by 1" stub, which silently cut the wrong
    % choice point when the condition pushed inner CPs (e.g. a call
    % to a multi-clause predicate). With get_level/cut Y_n we snapshot
    % the cp_count into a permanent Y register before the try_me_else
    % and restore it at the cut site, so soft cut works correctly
    % regardless of inner CP pushes.
    ( memberchk(ite_use_y_level(_), OptionsWithBS)
    -> OptionsWithLevel = OptionsWithBS
    ;  OptionsWithLevel = [ite_use_y_level(true) | OptionsWithBS]
    ),
    OptionsWithClosure = [lowered_closure(ClosureSet)|OptionsWithLevel],
    pass1_classify_predicates(ExpandedPredicates, OptionsWithClosure, 0, Classified),
    build_merged_labels(Classified, NamePCPairs, LabelMap),
    pass2_emit_merged(Classified, LabelMap, NamePCPairs, OptionsWithClosure,
                      NativeParts, WamParts),
    atomic_list_concat(NativeParts, '\n\n', NativeCode),
    atomic_list_concat(WamParts, '\n\n', WamCode).

%% expand_wam_llvm_project_predicates(+Roots, +Options, -Expanded) is det.
%
%  Compute a conservative same-module helper closure before project-level
%  classification. WAM fallback emits cross-predicate calls as numeric label
%  references inside the merged code array, so local private helpers must be in
%  the same project batch as their roots. Builtins and predicates without local
%  clauses remain on the existing builtin/external paths.
expand_wam_llvm_project_predicates(Roots, Options, Expanded) :-
    % M141: normalize roots to Module:Pred/Arity BEFORE seeding the
    % seen-set. Discovered call targets are always emitted qualified
    % (user:node/1), so an unqualified root (node/1) never matched
    % the memberchk dedup and its predicate was compiled into the
    % merged module TWICE -- llc then rejected the duplicate
    % @<pred>_switch_N globals ("redefinition of global").
    maplist([PI, M:P/A]>>normalize_project_predicate_indicator(PI, M, P, A),
            Roots, NormRoots0),
    list_to_set(NormRoots0, NormRoots),
    expand_wam_llvm_project_predicates_(NormRoots, Options, NormRoots, Expanded0),
    list_to_set(Expanded0, Expanded).

expand_wam_llvm_project_predicates_([], _, Seen, Seen).
expand_wam_llvm_project_predicates_([PI|Queue], Options, Seen0, Expanded) :-
    (   wam_llvm_project_predicate_targets(PI, Options, Targets0)
    ->  exclude({Seen0}/[T]>>memberchk(T, Seen0), Targets0, NewTargets),
        append(Seen0, NewTargets, Seen1),
        append(Queue, NewTargets, Queue1)
    ;   Seen1 = Seen0,
        Queue1 = Queue
    ),
    expand_wam_llvm_project_predicates_(Queue1, Options, Seen1, Expanded).

wam_llvm_project_predicate_targets(PI, Options, Targets) :-
    normalize_project_predicate_indicator(PI, Module, _Pred, _Arity),
    catch(wam_target:compile_predicate_to_wam(PI, Options, WamRaw), _, fail),
    wam_dependency_instrs(WamRaw, Instrs),
    call_execute_targets(Instrs, TargetPAs),
    findall(Module:TargetPred/TargetArity,
        ( member(TargetPred/TargetArity, TargetPAs),
          TargetPred \== call,
          \+ wam_target:is_builtin_pred(TargetPred, TargetArity),
          % Skip runtime-handled control/db predicates (retract/1, catch/3,
          % throw/1). catch/3 in particular is flagged both built_in AND
          % interpreted in SWI (it has a Prolog definition using $catch), so
          % without this it would be pulled in and fail to compile.
          \+ wam_special_call_pred(TargetPred, TargetArity),
          current_predicate(Module:TargetPred/TargetArity),
          functor(TargetHead, TargetPred, TargetArity),
          predicate_property(Module:TargetHead, interpreted)
        ),
        Targets0),
    sort(Targets0, Targets).

normalize_project_predicate_indicator(Module:Pred/Arity, Module, Pred, Arity) :- !.
normalize_project_predicate_indicator(Pred/Arity, user, Pred, Arity).

wam_dependency_instrs(WamRaw, Instrs) :-
    ( atom(WamRaw) -> atom_string(WamRaw, WamStr)
    ; string(WamRaw) -> WamStr = WamRaw
    ),
    split_string(WamStr, "\n", "", Lines),
    findall(Instr,
        ( member(Line, Lines),
          split_string(Line, " \t,", " \t,", Parts0),
          delete(Parts0, "", Parts),
          wam_dependency_instr_from_parts(Parts, Instr)
        ),
        Instrs).

wam_dependency_instr_from_parts(["call", Target, ArityText | _], call(Target, Arity)) :-
    number_string(Arity, ArityText), !.
wam_dependency_instr_from_parts(["execute", Target | _], execute(Target)) :- !.

%% compute_lowered_closure(+Predicates, +Options, -ClosureSet) is det.
%
%  M4 fixpoint analysis: returns the set of `Pred/Arity` indicators in
%  this compile batch whose lowered kernels can call each other safely.
%
%  Algorithm:
%    1. Collect every predicate that's gated for lowering and whose
%       bytecode compiles cleanly. Call this the candidate set.
%    2. Start with an empty closure. Iterate: find every candidate
%       whose clause-1 body passes wam_llvm_lowerable_with_closure/4
%       against the *current* closure. Add it. Repeat until no new
%       members are discovered.
%
%  The fixpoint always terminates because the closure is monotonic
%  and bounded by the candidate set. Predicates with no call/execute
%  in their clause-1 body join on the first iteration.
%
%  When emit_mode is `interpreter` (the default), the closure is
%  empty — no per-batch analysis needed.
compute_lowered_closure(Predicates, Options, ClosureSet) :-
    option(emit_mode(Mode), Options, interpreter),
    ( Mode == interpreter
    -> ClosureSet = []
    ;  collect_lowerable_candidates(Predicates, Options, Candidates),
       closure_fixpoint(Candidates, [], ClosureSet)
    ).

%% collect_lowerable_candidates(+Predicates, +Options, -Candidates).
%
%  Candidates is a list of `Pred/Arity-WamCode` pairs for predicates
%  that (a) are gated for lowering via emit_mode_allows_lowering and
%  (b) have a compileable WAM bytecode. Predicates that fail either
%  check are excluded — they will fall through to the WAM-fallback
%  branch of pass1 regardless of the closure.
collect_lowerable_candidates([], _, []).
collect_lowerable_candidates([PI|Rest], Options, Candidates) :-
    ( PI = Module:Pred/Arity -> true
    ; PI = Pred/Arity, Module = user
    ),
    ( emit_mode_allows_lowering(Module:Pred/Arity, Options),
      catch(
          wam_target:compile_predicate_to_wam(Module:Pred/Arity,
              Options, Wam),
          _, fail)
    -> Candidates = [Pred/Arity-Wam | RestCands],
       collect_lowerable_candidates(Rest, Options, RestCands)
    ;  collect_lowerable_candidates(Rest, Options, Candidates)
    ).

closure_fixpoint(Candidates, CurrentSet, FinalSet) :-
    findall(Pred/Arity,
        ( member(Pred/Arity-Wam, Candidates),
          \+ memberchk(Pred/Arity, CurrentSet),
          catch(
              wam_llvm_lowerable_with_closure(Pred/Arity, Wam,
                  CurrentSet, _Shape),
              _, fail)
        ),
        NewMembers),
    ( NewMembers == []
    -> FinalSet = CurrentSet
    ;  append(NewMembers, CurrentSet, ExpandedSet),
       closure_fixpoint(Candidates, ExpandedSet, FinalSet)
    ).

%% pass1_classify_predicates(+Preds, +Options, +StartPC, -Classified)
%
%  Classified is a list of records (in input order):
%    native(PredIndicator, Arity, PredCode)
%       — native-lowered (legacy llvm_target.pl or single-clause
%         wam_llvm_lowered_emitter); emit PredCode as-is. PredCode
%         already includes the `@<pred>` public-entry wrapper.
%    wam(PredIndicator, Arity, RawInstrs, LocalLabels, StartPC, NumInstrs)
%       — wam-fallback or foreign-kernel; goes into merged code.
%         Public entry `@<pred>` is emitted by emit_one_entry_func.
%    hybrid(PredIndicator, Arity, LoweredCode, RawInstrs, LocalLabels,
%           StartPC, NumInstrs)
%       — multi-clause predicate whose clause 1 is lowerable (M3).
%         LoweredCode is the `@lowered_<pred>_<arity>` clause-1 fast
%         path. The full bytecode (all clauses) goes into merged
%         @module_code at StartPC. Public entry `@<pred>` is emitted
%         by emit_hybrid_dispatcher and tries the fast path first,
%         then falls back to @run_loop at StartPC on failure.
%    failed(PredIndicator, Arity)
%       — both native + wam failed; emit a comment.
%
%  StartPC threads through wam/foreign/hybrid records only; native
%  and failed do not consume code slots.
pass1_classify_predicates([], _, _, []).
pass1_classify_predicates([PredIndicator|Rest], Options, StartPC,
                          [Record | RestRecs]) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    (   lookup_foreign_kernel_spec(Pred, Arity, Kind, _Config)
    ->  format(user_error, '  ~w/~w: foreign kernel (~w)~n', [Pred, Arity, Kind]),
        foreign_kernel_wam_text(Pred/Arity, Kind, Arity, Options, WamRaw),
        parse_wam_to_pass1(Pred/Arity, WamRaw, Arity, StartPC, Record, NumInstrs),
        NextPC is StartPC + NumInstrs
    ;   catch(
            llvm_target:compile_predicate_to_llvm(Module:Pred/Arity, Options, PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        Record = native(PredIndicator, Arity, PredCode),
        NextPC = StartPC
    ;   % WAM-lowered LLVM emission (wam_llvm_lowered_emitter).
        %
        % Gate: opt-in via emit_mode(Mode), where Mode is one of:
        %   functions          — try to lower every predicate
        %   mixed([P/A, ...])  — try to lower only the listed predicates
        %   interpreter        — never lower (default; matches prior behaviour)
        %
        % Shape (returned by wam_llvm_lowerable/3):
        %   single_clause   — full body is the lowered fn; no bytecode
        %                     needed. Classify as Native; PredCode
        %                     bundles the lowered fn + @<pred> wrapper.
        %   multi_clause_c1 — clause 1 lowered as a fast path; full
        %                     bytecode also emitted into @module_code
        %                     so the dispatcher's slow path can run
        %                     all clauses. Classify as Hybrid.
        emit_mode_allows_lowering(Module:Pred/Arity, Options),
        catch(
            wam_target:compile_predicate_to_wam(Module:Pred/Arity,
                Options, LowerWamRaw),
            _, fail),
        option(lowered_closure(Closure), Options, []),
        % Use the closure-aware lowerability check: gates call/execute
        % targets against the closure set computed in compile_predicates_for_llvm/4.
        catch(
            wam_llvm_lowerable_with_closure(Pred/Arity, LowerWamRaw,
                Closure, LowerShape),
            _, fail)
    ->  catch(
            lower_predicate_to_llvm(Pred/Arity, LowerWamRaw,
                Options, LoweredCode),
            LowerErr,
            ( format(user_error,
                '  ~w/~w: lowered emit failed (~w), falling back~n',
                [Pred, Arity, LowerErr]),
              fail
            )),
        ( LowerShape == single_clause
        -> format(user_error,
            '  ~w/~w: lowered LLVM emission (single-clause)~n',
            [Pred, Arity]),
           % Pack the lowered fn + thin @<pred> wrapper into one PredCode
           % blob so the existing Native record path emits everything.
           emit_native_wrapper(Pred/Arity, NativeWrapper),
           atomic_list_concat([LoweredCode, '\n\n', NativeWrapper],
               BundledCode),
           Record = native(PredIndicator, Arity, BundledCode),
           NextPC = StartPC
        ;  ( LowerShape == multi_clause_c1 ; LowerShape == clause_chain
           ; LowerShape == multi_clause_n )
        -> ( LowerShape == clause_chain
           ->  format(user_error,
                 '  ~w/~w: lowered LLVM emission (hybrid T5 first-arg dispatch + bytecode)~n',
                 [Pred, Arity])
           ;   LowerShape == multi_clause_n
           ->  format(user_error,
                 '  ~w/~w: lowered LLVM emission (hybrid T4 all-clauses inline + bytecode)~n',
                 [Pred, Arity])
           ;   format(user_error,
                 '  ~w/~w: lowered LLVM emission (hybrid clause-1 + bytecode)~n',
                 [Pred, Arity])
           ),
           % Parse the WAM bytecode (all clauses) into the merge so the
           % dispatcher's slow path can run it. The dispatcher entry
           % is emitted in pass2 via emit_hybrid_dispatcher.
           % clause_chain lowers ALL clauses as the fast path; the bytecode is
           % still emitted so the unbound-first-arg case (which the lowered fn
           % declines by returning false) is enumerated by @run_loop.
           %
           % Hybrid records use the bare Pred/Arity form (matching the
           % wam-record convention) so partition / resolve / dispatch
           % helpers can pattern-match the indicator uniformly.
           parse_wam_to_pass1(Pred/Arity, LowerWamRaw, Arity, StartPC,
               wam(Pred/Arity, _, RawInstrs, LocalLabels, StartPC, NumInstrs),
               NumInstrs),
           Record = hybrid(Pred/Arity, Arity, LoweredCode,
               RawInstrs, LocalLabels, StartPC, NumInstrs),
           NextPC is StartPC + NumInstrs
        )
    ;   option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        catch(
            wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamRaw),
            _, fail)
    ->  format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
        parse_wam_to_pass1(Pred/Arity, WamRaw, Arity, StartPC, Record, NumInstrs),
        NextPC is StartPC + NumInstrs
    ;   format(user_error, '  ~w/~w: compilation failed~n', [Pred, Arity]),
        Record = failed(PredIndicator, Arity),
        NextPC = StartPC
    ),
    pass1_classify_predicates(Rest, Options, NextPC, RestRecs).

%% emit_mode_allows_lowering(+PredIndicator, +Options) is semidet.
%
%  Gate for the wam_llvm_lowered_emitter branch in
%  pass1_classify_predicates/4. Matches the F# target's
%  `wam_fsharp_resolve_emit_mode/2` semantics:
%
%    emit_mode(functions)          → succeeds for every predicate.
%    emit_mode(mixed([P/A, ...]))  → succeeds iff PredIndicator is
%                                    in the list (Module:Pred/Arity
%                                    or bare Pred/Arity).
%    emit_mode(interpreter)        → always fails.
%    no emit_mode option           → defaults to interpreter (fails).
emit_mode_allows_lowering(Module:Pred/Arity, Options) :-
    option(emit_mode(Mode), Options, interpreter),
    Mode \== interpreter,
    ( Mode == functions
    -> true
    ; Mode = mixed(List)
    -> ( memberchk(Pred/Arity, List)
       ; memberchk(Module:Pred/Arity, List)
       )
    ).

%% parse_wam_to_pass1(+Pred/Arity, +WamCode, +Arity, +StartPC,
%%                    -wam(...), -NumInstrs)
%  Shared between wam-fallback and foreign-kernel classifications.
parse_wam_to_pass1(Pred/Arity, WamCode, ArityOut, StartPC,
                   wam(Pred/Arity, ArityOut, RawInstrs, LocalLabels,
                       StartPC, NumInstrs),
                   NumInstrs) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_pass1(Lines, 0, RawInstrs, LocalLabels),
    length(RawInstrs, NumInstrs).

%% foreign_kernel_wam_text(+Pred/Arity, +Kind, +Arity, +Options, -WamCode)
%  Produce the WAM text for a foreign kernel predicate.
foreign_kernel_wam_text(Pred/Arity, Kind, _Arity, _Options, WamCode) :-
    wam_llvm_foreign_kind_id(Kind, _KindId),
    allocate_foreign_instance_id(Pred/Arity, Kind, InstanceId),
    format(atom(WamCode), '~w/~w:\ncall_foreign ~w, ~w\nproceed',
           [Pred, Arity, Kind, InstanceId]).

%% build_merged_labels(+Classified, -NamePCPairs, -LabelMap)
%  Shift each wam/foreign predicate's local labels by its StartPC and
%  accumulate into NamePCPairs (LabelName-AbsolutePC, in predicate
%  order). LabelMap is the derived LabelName-Index view used by the
%  literal-resolution passes (index = physical slot in @module_labels).
%  At runtime, @wam_label_pc reads @module_labels[Index] to get the PC.
build_merged_labels(Classified, NamePCPairs, LabelMap) :-
    build_merged_labels_entries(Classified, NamePCPairs),
    build_label_index_map(NamePCPairs, LabelMap).

build_merged_labels_entries([], []).
build_merged_labels_entries([wam(_, _, _, LocalLabels, StartPC, _)|Rest],
                            Entries) :- !,
    shift_labels(LocalLabels, StartPC, Shifted),
    build_merged_labels_entries(Rest, RestEntries),
    append(Shifted, RestEntries, Entries).
build_merged_labels_entries([hybrid(_, _, _, _, LocalLabels, StartPC, _)|Rest],
                            Entries) :- !,
    shift_labels(LocalLabels, StartPC, Shifted),
    build_merged_labels_entries(Rest, RestEntries),
    append(Shifted, RestEntries, Entries).
build_merged_labels_entries([_|Rest], Entries) :-
    build_merged_labels_entries(Rest, Entries).

shift_labels([], _, []).
shift_labels([Name-LocalPC|Rest], Shift, [Name-GlobalPC|RestShifted]) :-
    GlobalPC is LocalPC + Shift,
    shift_labels(Rest, Shift, RestShifted).

%% pass2_emit_merged(+Classified, +GlobalLabels, +Options,
%%                   -NativeParts, -WamParts)
%  Emit:
%    - native predicates verbatim (→ NativeParts)
%    - one @module_code array + one @module_labels array covering all
%      wam/foreign predicates
%    - per-predicate entry functions that share those globals
%    - resolved switch tables (one per switch instruction)
%  All of these go into WamParts.
pass2_emit_merged(Classified, LabelMap, NamePCPairs, Options,
                  NativeParts, WamParts) :-
    partition_classified(Classified,
        NativeRecords, WamRecords, HybridRecords, FailedRecords),
    % A predicate that failed every compilation route used to become
    % an IR comment inside a "successful" module -- the caller thought
    % it compiled. Fail loudly instead (wam_allow_failed(true) keeps
    % the old comment-and-continue behaviour).
    (   FailedRecords == []
    ->  true
    ;   option(wam_allow_failed(true), Options)
    ->  true
    ;   findall(PI, member(failed(PI, _A), FailedRecords), FailedPIs),
        throw(error(existence_error(procedures, FailedPIs),
            context(wam_llvm_codegen,
                'these predicates could not be compiled (no clauses or no supported route); check the name/arity -- a DCG rule adds 2 to the visible arity -- or pass wam_allow_failed(true) to emit the module with them omitted')))
    ),
    % Native records' PredCode already includes both the lowered
    % function AND the @<pred> wrapper. Hybrid records contribute
    % their LoweredCode (just the @lowered_*_* fn); the matching
    % @<pred> dispatcher is emitted in the WAM entry-funcs section
    % because it needs InstrCount/LabelCount/StartPC.
    maplist([native(_,_,Code), Code]>>true, NativeRecords, NativeCodes),
    maplist([hybrid(_,_,LC,_,_,_,_), LC]>>true,
            HybridRecords, HybridLowered),
    append(NativeCodes, HybridLowered, NativeParts),
    % WAM-like records (wam + hybrid) all participate in the merged
    % @module_code. Their entry funcs differ — emit_one_entry_func
    % vs emit_hybrid_dispatcher — dispatched on record type by
    % emit_one_record_entry_func/5 inside emit_all_entry_funcs.
    append(WamRecords, HybridRecords, WamLikeRecords),
    (   WamLikeRecords == []
        % Even with no bytecode predicates (everything fully lowered),
        % the interpreter runtime's do_call/do_execute cases reference
        % @wam_dispatch_meta_call unconditionally, so its definition
        % (with empty dispatch tables) must still be emitted.
    ->  emit_meta_call_dispatch([], MergedCode),
        EntryFuncs = '', SwitchDefs = ''
    ;   emit_merged_wam_section(WamLikeRecords, LabelMap, NamePCPairs, Options,
                                MergedCode, EntryFuncs, SwitchDefs)
    ),
    maplist([failed(PI,A), C]>>format(atom(C),
               '; ~w/~w: compilation failed', [PI, A]),
            FailedRecords, FailedComments),
    WamParts0 = [SwitchDefs, MergedCode, EntryFuncs | FailedComments],
    exclude(==(''), WamParts0, WamParts).

partition_classified([], [], [], [], []).
partition_classified([native(PI,A,C)|Rest], [native(PI,A,C)|RN], RW, RH, RF) :- !,
    partition_classified(Rest, RN, RW, RH, RF).
partition_classified([wam(P,A,I,L,S,N)|Rest], RN, [wam(P,A,I,L,S,N)|RW], RH, RF) :- !,
    partition_classified(Rest, RN, RW, RH, RF).
partition_classified([hybrid(P,A,LC,I,L,S,N)|Rest], RN, RW,
                     [hybrid(P,A,LC,I,L,S,N)|RH], RF) :- !,
    partition_classified(Rest, RN, RW, RH, RF).
partition_classified([failed(PI,A)|Rest], RN, RW, RH, [failed(PI,A)|RF]) :-
    partition_classified(Rest, RN, RW, RH, RF).

%% emit_merged_wam_section(+WamRecords, +GlobalLabels, +Options,
%%                         -CodeGlobal, -EntryFuncs, -SwitchDefs)
%
%  Re-encodes every wam/foreign predicate's raw instruction parts
%  against GlobalLabels, collects all literals into one big list, and
%  produces:
%    - CodeGlobal: @module_code = [N x %Instruction] [ ... ] + @module_labels
%    - EntryFuncs: one `define i1 @<pred>(...)` per predicate, each
%      setting start PC to its own StartPC before calling @run_loop
%    - SwitchDefs: all switch-table globals referenced by the merged
%      code (names are predicate-qualified already, collisions avoided)
emit_merged_wam_section(WamRecords, LabelMap, NamePCPairs, Options,
                        CodeGlobal, EntryFuncs, SwitchDefs) :-
    resolve_all_wam_records(WamRecords, LabelMap, AllLiterals,
                            AllSwitchDefs),
    length(AllLiterals, InstrCount),
    length(NamePCPairs, LabelCount),
    retractall(wam_llvm_last_compile_counts(_, _)),
    assertz(wam_llvm_last_compile_counts(InstrCount, LabelCount)),
    (   InstrCount =:= 0
        % Even with no bytecode (every predicate fully lowered), the
        % interpreter runtime's do_call/do_execute cases reference
        % @wam_dispatch_meta_call unconditionally, so its definition
        % (with empty dispatch tables) must still be emitted.
    ->  emit_meta_call_dispatch([], MetaCallDispatch),
        format(string(CodeGlobal),
"; === Merged WAM code: none (all predicates fully lowered) ===
~w", [MetaCallDispatch]),
        EntryFuncs = '', SwitchDefs = ''
    ;   maplist([Lit, E]>>format(atom(E), '  ~w', [Lit]),
                AllLiterals, Entries),
        atomic_list_concat(Entries, ',\n', EntriesStr),
        ( NamePCPairs == []
        -> LabelsStr = "  i32 0",
           LabelArraySize = 1
        ;  maplist([_-PC, E]>>format(atom(E), '  i32 ~w', [PC]),
                   NamePCPairs, LabelRows),
           atomic_list_concat(LabelRows, ',\n', LabelsStr),
           LabelArraySize = LabelCount
        ),
        emit_meta_call_dispatch(LabelMap, MetaCallDispatch),
        format(string(CodeGlobal),
"; === Merged WAM code and labels (project-level) ===
@module_code = private constant [~w x %Instruction] [
~w
]

@module_labels = private constant [~w x i32] [
~w
]

~w",
            [InstrCount, EntriesStr, LabelArraySize, LabelsStr, MetaCallDispatch]),
        emit_all_entry_funcs(WamRecords, Options,
                             InstrCount, LabelCount, LabelArraySize,
                             EntryFuncs),
        ( AllSwitchDefs == []
        -> SwitchDefs = ''
        ;  atomic_list_concat(AllSwitchDefs, '\n', SwitchDefs)
        )
    ).

emit_meta_call_dispatch(LabelMap, IR) :-
    findall(entry(AtomId, FunctorRow, Arity, LabelIdx),
        ( member(Label-LabelIdx, LabelMap),
          catch(split_functor_arity(Label, NameStr, Arity), _, fail),
          Arity >= 0,
          atom_string(NameAtom, NameStr),
          intern_atom(NameAtom, AtomId),
          register_functor_string(NameStr),
          string_length(NameStr, NameLen),
          NameLenPlus1 is NameLen + 1,
          sanitize_functor_for_llvm(NameStr, SaneName),
          format(atom(FunctorRow),
              '  i8* bitcast ([~w x i8]* @.fn_~w to i8*)',
              [NameLenPlus1, SaneName])
        ),
        Entries0),
    sort(Entries0, Entries),
    length(Entries, Count),
    (   Count =:= 0
    ->  ArraySize = 1,
        AtomRows = "  i64 0",
        FunctorRows = "  i8* null",
        ArityRows = "  i32 0",
        LabelRows = "  i32 0"
    ;   ArraySize = Count,
        findall(A, (member(entry(AtomId, _, _, _), Entries), format(atom(A), '  i64 ~w', [AtomId])), AtomList),
        findall(F, member(entry(_, F, _, _), Entries), FunctorList),
        findall(R, (member(entry(_, _, Arity, _), Entries), format(atom(R), '  i32 ~w', [Arity])), ArityList),
        findall(L, (member(entry(_, _, _, LabelIdx), Entries), format(atom(L), '  i32 ~w', [LabelIdx])), LabelList),
        atomic_list_concat(AtomList, ',\n', AtomRows),
        atomic_list_concat(FunctorList, ',\n', FunctorRows),
        atomic_list_concat(ArityList, ',\n', ArityRows),
        atomic_list_concat(LabelList, ',\n', LabelRows)
    ),
    format(string(IR),
"; === Meta-call dispatch table: atom/functor + arity -> label index ===
; Host-global table, built at compile time from the host''s own predicates.
; A loaded .wamo object carries its OWN table on the VM (fields 25/26); the
; two find helpers below consult that table when present so an object can
; meta-call its own predicates (eval bootstrap milestone 2).
@wam_meta_call_atom_ids = private constant [~w x i64] [
~w
]
@wam_meta_call_functors = private constant [~w x i8*] [
~w
]
@wam_meta_call_arities = private constant [~w x i32] [
~w
]
@wam_meta_call_label_indexes = private constant [~w x i32] [
~w
]

; Resolve an atom goal (atom_id + target_arity) to a predicate label index,
; or -1 if none matches. Uses the VM''s object meta-call table when installed
; (loaded .wamo), else the host-global arrays.
define i32 @wam_meta_find_atom(%WamState* %vm, i64 %atom_id, i32 %target_arity) {
entry:
  %om_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 25
  %om = load %WamMetaRow*, %WamMetaRow** %om_ptr
  %om_null = icmp eq %WamMetaRow* %om, null
  br i1 %om_null, label %global, label %obj
obj:
  %omc_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 26
  %omc = load i32, i32* %omc_ptr
  br label %oloop
oloop:
  %oi = phi i32 [ 0, %obj ], [ %oi1, %onext ]
  %odone = icmp sge i32 %oi, %omc
  br i1 %odone, label %notfound, label %ocheck
ocheck:
  %orow = getelementptr %WamMetaRow, %WamMetaRow* %om, i32 %oi
  %oa_ptr = getelementptr %WamMetaRow, %WamMetaRow* %orow, i32 0, i32 0
  %oa = load i64, i64* %oa_ptr
  %oa_match = icmp eq i64 %oa, %atom_id
  %oar_ptr = getelementptr %WamMetaRow, %WamMetaRow* %orow, i32 0, i32 2
  %oar = load i32, i32* %oar_ptr
  %oar_match = icmp eq i32 %oar, %target_arity
  %o_match = and i1 %oa_match, %oar_match
  br i1 %o_match, label %ofound, label %onext
onext:
  %oi1 = add i32 %oi, 1
  br label %oloop
ofound:
  %ol_ptr = getelementptr %WamMetaRow, %WamMetaRow* %orow, i32 0, i32 3
  %ol = load i32, i32* %ol_ptr
  ret i32 %ol
global:
  br label %gloop
gloop:
  %gi = phi i32 [ 0, %global ], [ %gi1, %gnext ]
  %gdone = icmp sge i32 %gi, ~w
  br i1 %gdone, label %notfound, label %gcheck
gcheck:
  %gap = getelementptr [~w x i64], [~w x i64]* @wam_meta_call_atom_ids, i32 0, i32 %gi
  %ga = load i64, i64* %gap
  %ga_match = icmp eq i64 %ga, %atom_id
  %grp = getelementptr [~w x i32], [~w x i32]* @wam_meta_call_arities, i32 0, i32 %gi
  %gr = load i32, i32* %grp
  %gr_match = icmp eq i32 %gr, %target_arity
  %g_match = and i1 %ga_match, %gr_match
  br i1 %g_match, label %gfound, label %gnext
gnext:
  %gi1 = add i32 %gi, 1
  br label %gloop
gfound:
  %glp = getelementptr [~w x i32], [~w x i32]* @wam_meta_call_label_indexes, i32 0, i32 %gi
  %gl = load i32, i32* %glp
  ret i32 %gl
notfound:
  ret i32 -1
}

; Resolve a compound goal (functor pointer + target_arity) to a label index,
; or -1. Functor matching is by pointer identity; a loaded object''s compound
; goals carry its own functor copies, which is exactly what its object table
; stores (see the loader), so identity holds within one object.
define i32 @wam_meta_find_compound(%WamState* %vm, i8* %functor, i32 %target_arity) {
entry:
  %om_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 25
  %om = load %WamMetaRow*, %WamMetaRow** %om_ptr
  %om_null = icmp eq %WamMetaRow* %om, null
  br i1 %om_null, label %global, label %obj
obj:
  %omc_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 26
  %omc = load i32, i32* %omc_ptr
  br label %oloop
oloop:
  %oi = phi i32 [ 0, %obj ], [ %oi1, %onext ]
  %odone = icmp sge i32 %oi, %omc
  br i1 %odone, label %notfound, label %ocheck
ocheck:
  %orow = getelementptr %WamMetaRow, %WamMetaRow* %om, i32 %oi
  %of_ptr = getelementptr %WamMetaRow, %WamMetaRow* %orow, i32 0, i32 1
  %of = load i8*, i8** %of_ptr
  %of_match = icmp eq i8* %of, %functor
  %oar_ptr = getelementptr %WamMetaRow, %WamMetaRow* %orow, i32 0, i32 2
  %oar = load i32, i32* %oar_ptr
  %oar_match = icmp eq i32 %oar, %target_arity
  %o_match = and i1 %of_match, %oar_match
  br i1 %o_match, label %ofound, label %onext
onext:
  %oi1 = add i32 %oi, 1
  br label %oloop
ofound:
  %ol_ptr = getelementptr %WamMetaRow, %WamMetaRow* %orow, i32 0, i32 3
  %ol = load i32, i32* %ol_ptr
  ret i32 %ol
global:
  br label %gloop
gloop:
  %gi = phi i32 [ 0, %global ], [ %gi1, %gnext ]
  %gdone = icmp sge i32 %gi, ~w
  br i1 %gdone, label %notfound, label %gcheck
gcheck:
  %gfp = getelementptr [~w x i8*], [~w x i8*]* @wam_meta_call_functors, i32 0, i32 %gi
  %gf = load i8*, i8** %gfp
  %gf_match = icmp eq i8* %gf, %functor
  %grp = getelementptr [~w x i32], [~w x i32]* @wam_meta_call_arities, i32 0, i32 %gi
  %gr = load i32, i32* %grp
  %gr_match = icmp eq i32 %gr, %target_arity
  %g_match = and i1 %gf_match, %gr_match
  br i1 %g_match, label %gfound, label %gnext
gnext:
  %gi1 = add i32 %gi, 1
  br label %gloop
gfound:
  %glp = getelementptr [~w x i32], [~w x i32]* @wam_meta_call_label_indexes, i32 0, i32 %gi
  %gl = load i32, i32* %glp
  ret i32 %gl
notfound:
  ret i32 -1
}

define i1 @wam_dispatch_meta_call(%WamState* %vm, i32 %total_arity, i32 %after_pc) {
entry:
  %arity_ok = icmp sge i32 %total_arity, 1
  br i1 %arity_ok, label %read_goal, label %fail

read_goal:
  %goal = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %tag = extractvalue %Value %goal, 0
  %is_atom = icmp eq i32 %tag, 0
  br i1 %is_atom, label %atom_goal, label %maybe_compound

maybe_compound:
  %is_compound = icmp eq i32 %tag, 3
  br i1 %is_compound, label %compound_goal, label %fail

atom_goal:
  %atom_id = extractvalue %Value %goal, 1
  %target_arity = sub i32 %total_arity, 1
  %label_idx = call i32 @wam_meta_find_atom(%WamState* %vm, i64 %atom_id, i32 %target_arity)
  %atom_found = icmp sge i32 %label_idx, 0
  br i1 %atom_found, label %atom_resolve, label %atom_consult

atom_resolve:
  %target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %label_idx)
  %valid = icmp sge i32 %target_pc, 0
  br i1 %valid, label %prepare_atom_args, label %fail

prepare_atom_args:
  %has_atom_args = icmp sgt i32 %target_arity, 0
  br i1 %has_atom_args, label %shift_atom_args, label %go

shift_atom_args:
  %arg_i = phi i32 [ 0, %prepare_atom_args ], [ %arg_next, %shift_atom_args ]
  %src_i = add i32 %arg_i, 1
  %arg_val = call %Value @wam_get_reg(%WamState* %vm, i32 %src_i)
  call void @wam_set_reg(%WamState* %vm, i32 %arg_i, %Value %arg_val)
  %arg_next = add i32 %arg_i, 1
  %arg_done = icmp sge i32 %arg_next, %target_arity
  br i1 %arg_done, label %go, label %shift_atom_args

compound_goal:
  %compound_payload = extractvalue %Value %goal, 1
  %compound_ptr = inttoptr i64 %compound_payload to %Compound*
  %compound_fn_slot = getelementptr %Compound, %Compound* %compound_ptr, i32 0, i32 0
  %compound_functor = load i8*, i8** %compound_fn_slot
  %compound_arity_slot = getelementptr %Compound, %Compound* %compound_ptr, i32 0, i32 1
  %compound_base_arity = load i32, i32* %compound_arity_slot
  %compound_args_slot = getelementptr %Compound, %Compound* %compound_ptr, i32 0, i32 2
  %compound_args = load %Value*, %Value** %compound_args_slot
  %compound_extra_arity = sub i32 %total_arity, 1
  %compound_target_arity = add i32 %compound_base_arity, %compound_extra_arity
  %clabel_idx = call i32 @wam_meta_find_compound(%WamState* %vm, i8* %compound_functor, i32 %compound_target_arity)
  %comp_found = icmp sge i32 %clabel_idx, 0
  br i1 %comp_found, label %compound_resolve, label %compound_consult

compound_resolve:
  %ctarget_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %clabel_idx)
  %cvalid = icmp sge i32 %ctarget_pc, 0
  br i1 %cvalid, label %prepare_compound_extra, label %fail

prepare_compound_extra:
  %has_compound_extra = icmp sgt i32 %compound_extra_arity, 0
  br i1 %has_compound_extra, label %copy_compound_extra_init, label %prepare_closure_args

copy_compound_extra_init:
  %extra_last = sub i32 %compound_extra_arity, 1
  br label %copy_compound_extra

copy_compound_extra:
  %extra_i = phi i32 [ %extra_last, %copy_compound_extra_init ], [ %extra_prev, %copy_compound_extra ]
  %extra_src = add i32 %extra_i, 1
  %extra_dst = add i32 %compound_base_arity, %extra_i
  %extra_val = call %Value @wam_get_reg(%WamState* %vm, i32 %extra_src)
  call void @wam_set_reg(%WamState* %vm, i32 %extra_dst, %Value %extra_val)
  %extra_is_first = icmp eq i32 %extra_i, 0
  %extra_prev = sub i32 %extra_i, 1
  br i1 %extra_is_first, label %prepare_closure_args, label %copy_compound_extra

prepare_closure_args:
  %has_closure_args = icmp sgt i32 %compound_base_arity, 0
  br i1 %has_closure_args, label %copy_closure_args, label %compound_go

copy_closure_args:
  %base_i = phi i32 [ 0, %prepare_closure_args ], [ %base_next, %copy_closure_args ]
  %base_arg_ptr = getelementptr %Value, %Value* %compound_args, i32 %base_i
  %base_arg = load %Value, %Value* %base_arg_ptr
  call void @wam_set_reg(%WamState* %vm, i32 %base_i, %Value %base_arg)
  %base_next = add i32 %base_i, 1
  %base_done = icmp sge i32 %base_next, %compound_base_arity
  br i1 %base_done, label %compound_go, label %copy_closure_args

compound_go:
  call void @wam_set_cp(%WamState* %vm, i32 %after_pc)
  call void @wam_set_pc(%WamState* %vm, i32 %ctarget_pc)
  ret i1 true

go:
  call void @wam_set_cp(%WamState* %vm, i32 %after_pc)
  call void @wam_set_pc(%WamState* %vm, i32 %target_pc)
  ret i1 true

atom_consult:
  ; Meta table missed an atom goal -- consult the dynamic clause store
  ; for name/target_arity (see PLAWK_DYNAMIC_DB.md). target_arity > 0
  ; here means a call/N closure the iterator cannot read from reg0, so
  ; it simply finds no match and fails cleanly.
  %ac_fn = call i8* @wam_atom_to_string(i64 %atom_id)
  %ac_fn_null = icmp eq i8* %ac_fn, null
  br i1 %ac_fn_null, label %fail, label %ac_go
ac_go:
  %ac_ok = call i1 @wam_dyn_consult(%WamState* %vm, i8* %ac_fn, i32 %target_arity, i32 %after_pc)
  ret i1 %ac_ok

compound_consult:
  ; Meta table missed a compound goal -- consult the dynamic clause store.
  ; Only when the goal in reg0 is the complete goal (no call/N closure
  ; extra args); with extra args the iterator cannot see them, so fail.
  %cc_no_extra = icmp eq i32 %compound_extra_arity, 0
  br i1 %cc_no_extra, label %cc_go, label %fail
cc_go:
  %cc_ok = call i1 @wam_dyn_consult(%WamState* %vm, i8* %compound_functor, i32 %compound_base_arity, i32 %after_pc)
  ret i1 %cc_ok

fail:
  ret i1 false
}",
        [ArraySize, AtomRows, ArraySize, FunctorRows, ArraySize, ArityRows,
         ArraySize, LabelRows,
         Count, ArraySize, ArraySize, ArraySize, ArraySize, ArraySize, ArraySize,
         Count, ArraySize, ArraySize, ArraySize, ArraySize, ArraySize, ArraySize]).
%% resolve_all_wam_records(+Records, +GlobalLabels, -AllLiterals,
%%                         -AllSwitchDefs)
%  Resolves each record's raw instructions against GlobalLabels and
%  concatenates the literal lists in predicate order. Switch-table
%  names are predicate-qualified so flattening doesn't collide.
resolve_all_wam_records([], _, [], []).
resolve_all_wam_records([wam(Pred/_Arity, _, RawInstrs, _, _, _)|Rest],
                        GlobalLabels, AllLiterals, AllSwitchDefs) :- !,
    atom_string(Pred, PredStr),
    maplist(resolve_llvm_literal(GlobalLabels), RawInstrs, Literals0),
    resolve_switch_tables(Literals0, PredStr, 0, ResolvedLits, SwitchDefs),
    resolve_all_wam_records(Rest, GlobalLabels, RestLits, RestDefs),
    append(ResolvedLits, RestLits, AllLiterals),
    append(SwitchDefs, RestDefs, AllSwitchDefs).
resolve_all_wam_records([hybrid(Pred/_Arity, _, _, RawInstrs, _, _, _)|Rest],
                        GlobalLabels, AllLiterals, AllSwitchDefs) :- !,
    % Hybrid records (M3) participate in the merged @module_code on
    % equal footing with pure wam records — same resolution pass over
    % their RawInstrs, same per-predicate switch-table namespace.
    % The dispatcher entry (which uses the merged PC) is emitted by
    % emit_one_record_entry_func/5, not here.
    atom_string(Pred, PredStr),
    maplist(resolve_llvm_literal(GlobalLabels), RawInstrs, Literals0),
    resolve_switch_tables(Literals0, PredStr, 0, ResolvedLits, SwitchDefs),
    resolve_all_wam_records(Rest, GlobalLabels, RestLits, RestDefs),
    append(ResolvedLits, RestLits, AllLiterals),
    append(SwitchDefs, RestDefs, AllSwitchDefs).

%% emit_all_entry_funcs(+WamRecords, +Options,
%%                      +InstrCount, +LabelCount, +LabelArraySize,
%%                      -EntryFuncs)
%  One `define i1 @<pred>(...)` per wam/foreign predicate. All share
%  @module_code and @module_labels, setting PC = StartPC before
%  invoking the run loop.
emit_all_entry_funcs(WamLikeRecords, _Options, InstrCount, LabelCount,
                     LabelArraySize, EntryFuncs) :-
    maplist(emit_one_record_entry_func(InstrCount, LabelCount, LabelArraySize),
            WamLikeRecords, Funcs),
    atomic_list_concat(Funcs, '\n\n', EntryFuncs).

%% emit_one_record_entry_func(+IC, +LC, +LAS, +Record, -Func)
%
%  Dispatcher: wam records use the standard entry-func shape;
%  hybrid records (M3) get the fast-path-plus-slow-path dispatcher
%  from the lowered emitter.
emit_one_record_entry_func(InstrCount, LabelCount, LabelArraySize,
                           wam(Pred/Arity, _, _, _, StartPC, _),
                           Func) :- !,
    emit_one_entry_func(InstrCount, LabelCount, LabelArraySize,
                        wam(Pred/Arity, _, _, _, StartPC, _),
                        Func).
emit_one_record_entry_func(InstrCount, _LabelCount, LabelArraySize,
                           hybrid(Pred/Arity, _, _, _, _, StartPC, _),
                           Func) :- !,
    emit_hybrid_dispatcher(Pred/Arity, StartPC,
                           InstrCount, LabelArraySize, Func).

emit_one_entry_func(InstrCount, LabelCount, LabelArraySize,
                    wam(Pred/Arity, _, _, _, StartPC, _),
                    Func) :-
    atom_string(Pred, PredStr),
    build_llvm_arg_setup(Arity, ArgSetup),
    build_llvm_param_list(Arity, ParamList),
    % Per-predicate start-PC global. External drivers that build their
    % own VM (bypassing the entry function) need to know where the
    % predicate starts in @module_code. The name matches the pattern
    % used by test tooling for regex extraction.
    format(atom(Func),
'; WAM-compiled predicate: ~w/~w (merged module code, start PC ~w)
@~w_start_pc = private constant i32 ~w

define i1 @~w(~w) {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
    i32 ~w,
    i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
    i32 ~w)
  call void @wam_set_pc(%WamState* %vm, i32 ~w)
~w
  %result = call i1 @run_loop(%WamState* %vm)
  ; Free the state before returning so repeated bench-loop invocations
  ; (N thousand per workload) do not leak ~~85 KB per call.
  call void @wam_state_free(%WamState* %vm)
  ret i1 %result
}',
        [PredStr, Arity, StartPC,
         PredStr, StartPC,
         PredStr, ParamList,
         InstrCount, InstrCount,
         InstrCount,
         LabelArraySize, LabelArraySize,
         LabelCount,
         StartPC,
         ArgSetup]).

%% compile_foreign_kernel_predicate(+PredArity, +Kind, +Arity, +Options, -PredCode).
%
%  Generate an LLVM predicate body consisting of a single
%  `call_foreign Kind, InstanceId` instruction followed by `proceed`.
%  Takes the WAM-fallback compile path (not native) so the result
%  plugs into the same %Instruction array / step dispatch machinery
%  as any WAM-compiled predicate.
%
%  M5.8: op2 of call_foreign is now the instance_id (not arity). The
%  instance_id is the predicate's zero-based position among the
%  registered specs for its kind, matching the switch case layout
%  emitted by build_td3_instance_switch/3.
compile_foreign_kernel_predicate(Pred/Arity, Kind, _Arity, Options, PredCode) :-
    wam_llvm_foreign_kind_id(Kind, _KindId),  % fail fast if kind unknown
    allocate_foreign_instance_id(Pred/Arity, Kind, InstanceId),
    format(atom(WamCode), 'call_foreign ~w, ~w\nproceed', [Kind, InstanceId]),
    compile_wam_predicate_to_llvm(Pred/Arity, WamCode, Options, PredCode).

%% allocate_foreign_instance_id(+PredArity, +Kind, -InstanceId).
%
%  Looks up PredArity's position among the registered specs for Kind.
%  The position is stable within a compile because llvm_foreign_kernel_spec/3
%  is a dynamic fact table and findall/3 preserves assertion order.
%  If PredArity isn't in the table this fails — it should have been
%  added via one of the M5.6 entry paths before compile.
allocate_foreign_instance_id(PredArity, Kind, InstanceId) :-
    findall(P,
        llvm_foreign_kernel_spec(P, Kind, _),
        AllPreds),
    nth0(InstanceId, AllPreds, PredArity), !.

% ============================================================================
% PHASE 2: step_wam/3 → LLVM switch dispatch
% ============================================================================

%% compile_step_wam_to_llvm(+Options, -LLVMCode)
%  Generates the step() function body as an LLVM switch on instruction tag.
compile_step_wam_to_llvm(_Options, LLVMCode) :-
    findall(Case, compile_llvm_step_case(Case), Cases),
    atomic_list_concat(Cases, '\n', CasesCode),
    % M7: !prof branch_weights on the @step switch. These are relative
    % weights estimated from typical Prolog workload instruction
    % frequencies (head unification + body construction + control
    % dominate; backtrack-only opcodes are rare). LLVM uses them to
    % order the switch lowering — at x86 -O2 the cases are dense
    % enough that a jump table is generated regardless, but the
    % weights still influence branch prediction on the per-case
    % inner branches and can help on architectures (ARM, RISC-V)
    % where the switch lowers to a balanced binary search.
    %
    % The metadata MUST be a reference (`!prof !N`) — inline `!{...}`
    % is not valid for !prof in LLVM IR. We define the node as
    % `!wam_step_branch_weights` at module scope right after @step's
    % closing brace; the named metadata avoids collision with any
    % numeric `!N` ids the module might pick up from other sources
    % (LTO bitcode, debug info, etc.).
    %
    % First weight is for the `default` label (taken only on a
    % malformed bytecode tag, so 1). The 33 case weights match the
    % opcode order in the switch above (tag 0..32).
    format(atom(LLVMCode),
'; M7 @step branch weights (!prof !99 below) — relative frequencies
; estimated from typical Prolog workloads. The slot order in !99
; matches the switch case order: slot 0 = default (malformed tag),
; slot k+1 = case for tag k (k = 0..32). Per-opcode weights:
;   default: 1   — never expected, only on a bytecode bug
;   get_constant      50,  get_variable      80,  get_value        100
;   get_structure     20,  get_list          20,  unify_variable    30
;   unify_value       30,  unify_constant    10,  put_constant      50
;   put_variable      80,  put_value        100,  put_structure     20
;   put_list          10,  set_variable      20,  set_value         30
;   set_constant      30,  allocate          50,  deallocate        50
;   do_call           80,  do_execute        50,  proceed          100
;   builtin_call      60,  try_me_else       30,  retry_me_else      5
;   trust_me          10,  switch_on_constant 10, switch_on_structure 5
;   switch_on_const_a2 5,  begin_aggregate    5,  end_aggregate      5
;   call_foreign      10,  cut_ite            5,  jump               5
; At -O2 on x86 the dense 0..32 range typically lowers to a jump
; table regardless, but the weights still inform LLVM\'s per-case
; inner-branch ordering and help on ARM/RISC-V backends that use a
; balanced-binary-search lowering.
define i1 @step(%WamState* %vm, %Instruction* %instr) {
entry:
  %tag_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 0
  %tag = load i32, i32* %tag_ptr
  %op1_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 1
  %op1 = load i64, i64* %op1_ptr
  %op2_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 2
  %op2 = load i64, i64* %op2_ptr
  switch i32 %tag, label %default [
    i32 0, label %get_constant
    i32 1, label %get_variable
    i32 2, label %get_value
    i32 3, label %get_structure
    i32 4, label %get_list
    i32 5, label %unify_variable
    i32 6, label %unify_value
    i32 7, label %unify_constant
    i32 8, label %put_constant
    i32 9, label %put_variable
    i32 10, label %put_value
    i32 11, label %put_structure
    i32 12, label %put_list
    i32 13, label %set_variable
    i32 14, label %set_value
    i32 15, label %set_constant
    i32 16, label %allocate
    i32 17, label %deallocate
    i32 18, label %do_call
    i32 19, label %do_execute
    i32 20, label %proceed
    i32 21, label %builtin_call
    i32 22, label %try_me_else
    i32 23, label %retry_me_else
    i32 24, label %trust_me
    i32 25, label %switch_on_constant
    i32 26, label %switch_on_structure
    i32 27, label %switch_on_constant_a2
    i32 28, label %begin_aggregate
    i32 29, label %end_aggregate
    i32 30, label %call_foreign
    i32 31, label %cut_ite
    i32 32, label %jump
    i32 33, label %get_level
    i32 34, label %cut
  ], !prof !99

~w

default:
  ret i1 false
}

; M7 step-dispatch branch weights — relative frequencies estimated
; from typical Prolog workloads. Operand layout (must match the
; switch case order above):
;   slot 0 = default (malformed tag)
;   slot k+1 = case for tag k, k = 0..32
; Comments inside the operand list aren\'t allowed (the LLVM IR
; parser rejects `;` mid-operand), so the per-opcode rationale lives
; in the comment block above the @step definition.
!99 = !{!\"branch_weights\", i32 1, i32 50, i32 80, i32 100, i32 20, i32 20, i32 30, i32 30, i32 10, i32 50, i32 80, i32 100, i32 20, i32 10, i32 20, i32 30, i32 30, i32 50, i32 50, i32 80, i32 50, i32 100, i32 60, i32 30, i32 5, i32 10, i32 10, i32 5, i32 5, i32 5, i32 5, i32 10, i32 5, i32 5, i32 5, i32 5}', [CasesCode]).

compile_llvm_step_case(CaseCode) :-
    wam_llvm_case(Label, BodyCode),
    format(atom(CaseCode), '~w:\n~w', [Label, BodyCode]).

% --- Head Unification Instructions ---

wam_llvm_case('get_constant',
'  ; op1 = constant value (packed), op2 = (tag << 16) | reg_idx.
  ; Deref the register so a put_variable-aliased Ref reads as its
  ; underlying heap-stored value (Unbound for fresh vars). Bind via
  ; wam_bind_reg so the shared heap cell — and any other Y/A registers
  ; aliased to it — updates atomically.
  %gc.op2_32 = trunc i64 %op2 to i32
  %gc.reg_idx = and i32 %gc.op2_32, 65535
  %gc.tag = lshr i32 %gc.op2_32, 16
  %gc.current = call %Value @wam_get_reg_deref(%WamState* %vm, i32 %gc.reg_idx)
  %gc.is_unb = call i1 @value_is_unbound(%Value %gc.current)
  br i1 %gc.is_unb, label %gc.bind, label %gc.check_eq

gc.bind:
  call void @wam_trail_binding(%WamState* %vm, i32 %gc.reg_idx)
  %gc.const_val = insertvalue %Value undef, i32 %gc.tag, 0
  %gc.const_v2 = insertvalue %Value %gc.const_val, i64 %op1, 1
  call void @wam_bind_reg(%WamState* %vm, i32 %gc.reg_idx, %Value %gc.const_v2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gc.check_eq:
  ; Bound: check equality with proper tag.
  %gc.expected = insertvalue %Value undef, i32 %gc.tag, 0
  %gc.expected2 = insertvalue %Value %gc.expected, i64 %op1, 1
  %gc.eq = call i1 @value_equals(%Value %gc.current, %Value %gc.expected2)
  br i1 %gc.eq, label %gc.match, label %gc.fail

gc.match:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gc.fail:
  ret i1 false').

wam_llvm_case('get_variable',
'  ; op1 = Xn index, op2 = Ai index
  %gv.ai = trunc i64 %op2 to i32
  %gv.xn = trunc i64 %op1 to i32
  %gv.val = call %Value @wam_get_reg(%WamState* %vm, i32 %gv.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %gv.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %gv.xn, %Value %gv.val)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('get_value',
'  ; op1 = Xn index, op2 = Ai index -- full unification of two head
  ; positions. Campaign finding no. 9: variables must be handled via
  ; @wam_deref_keep_var, NOT the full deref. The full deref collapses a
  ; Ref-to-unbound-cell into the shared Unbound sentinel (tag 6, no cell
  ; address), so the old bind paths stored a detached Unbound instead of
  ; linking the two variables -- a var-var get_value was a silent no-op
  ; and every later binding through one side was invisible through the
  ; other (the difference-list serializer of the self-host fixpoint lost
  ; its open tail at exactly such a knot: wz_funcs([], A, A)).
  ; keep_var stops at the last Ref whose cell is unbound, so tag 5 here
  ; means variable-with-cell and tag 6 means naked register variable.
  %gval.ai = trunc i64 %op2 to i32
  %gval.xn = trunc i64 %op1 to i32
  %gval.rawa = call %Value @wam_get_reg(%WamState* %vm, i32 %gval.ai)
  %gval.rawx = call %Value @wam_get_reg(%WamState* %vm, i32 %gval.xn)
  %gval.ka = call %Value @wam_deref_keep_var(%WamState* %vm, %Value %gval.rawa)
  %gval.kx = call %Value @wam_deref_keep_var(%WamState* %vm, %Value %gval.rawx)
  %gval.ka_tag = extractvalue %Value %gval.ka, 0
  %gval.kx_tag = extractvalue %Value %gval.kx, 0
  %gval.ka_isref = icmp eq i32 %gval.ka_tag, 5
  %gval.ka_isunb = icmp eq i32 %gval.ka_tag, 6
  %gval.a_var = or i1 %gval.ka_isref, %gval.ka_isunb
  %gval.kx_isref = icmp eq i32 %gval.kx_tag, 5
  %gval.kx_isunb = icmp eq i32 %gval.kx_tag, 6
  %gval.x_var = or i1 %gval.kx_isref, %gval.kx_isunb
  br i1 %gval.a_var, label %gval.a_is_var, label %gval.a_bound

gval.a_is_var:
  br i1 %gval.x_var, label %gval.both_var, label %gval.bind_a

gval.both_var:
  ; same cell already? binding a cell to a Ref to itself would create a
  ; self-referential cell (the M139 hazard) -- the sides already alias.
  %gval.ka_pay = extractvalue %Value %gval.ka, 1
  %gval.kx_pay = extractvalue %Value %gval.kx, 1
  %gval.both_ref = and i1 %gval.ka_isref, %gval.kx_isref
  %gval.same_pay = icmp eq i64 %gval.ka_pay, %gval.kx_pay
  %gval.same_cell = and i1 %gval.both_ref, %gval.same_pay
  br i1 %gval.same_cell, label %gval.match, label %gval.link

gval.link:
  ; link the two variables, preferring an existing heap cell as target
  br i1 %gval.kx_isref, label %gval.link_a_to_x, label %gval.link_x

gval.link_a_to_x:
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.ai, %Value %gval.kx)
  br label %gval.match

gval.link_x:
  br i1 %gval.ka_isref, label %gval.link_x_to_a, label %gval.link_fresh

gval.link_x_to_a:
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.xn, %Value %gval.ka)
  br label %gval.match

gval.link_fresh:
  ; both sides are naked register variables (no heap cell): create a
  ; shared cell and point both registers at it
  %gval.unb0 = insertvalue %Value undef, i32 6, 0
  %gval.unb = insertvalue %Value %gval.unb0, i64 0, 1
  %gval.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %gval.unb)
  %gval.ref = call %Value @value_ref(i32 %gval.addr)
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.ai, %Value %gval.ref)
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.xn, %Value %gval.ref)
  br label %gval.match

gval.bind_a:
  ; A side variable, X side bound: bind through the A chain or register.
  ; wam_bind_reg trails internally (heap or register as appropriate).
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.ai, %Value %gval.kx)
  br label %gval.match

gval.a_bound:
  br i1 %gval.x_var, label %gval.bind_x, label %gval.check_eq

gval.bind_x:
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.xn, %Value %gval.ka)
  br label %gval.match

gval.check_eq:
  ; M138: get_value is a unify instruction, not an equality test.
  ; Both sides are bound here, so @wam_unify_value runs its both_bound
  ; path: identical payload compare for atomics, deep unification for
  ; compounds (binding any unbound args, with trailing). The previous
  ; @value_equals wrongly failed p(X, X) heads called with unifiable
  ; compounds like p(f(A), f(B)).
  %gval.eq = call i1 @wam_unify_value(%WamState* %vm, %Value %gval.ka, %Value %gval.kx)
  br i1 %gval.eq, label %gval.match, label %gval.fail

gval.match:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.fail:
  ret i1 false').

% --- Structure/List Head Unification ---

wam_llvm_case('get_structure',
'  ; get_structure: op1 = ptrtoint of functor string, op2 = (arity << 16) | reg_idx
  %gs.op2_32 = trunc i64 %op2 to i32
  %gs.arity = lshr i32 %gs.op2_32, 16
  %gs.ai = and i32 %gs.op2_32, 65535
  %gs.val = call %Value @wam_get_reg_deref(%WamState* %vm, i32 %gs.ai)
  %gs.tag = extractvalue %Value %gs.val, 0
  %gs.is_cp = icmp eq i32 %gs.tag, 3
  br i1 %gs.is_cp, label %gs.read, label %gs.check_unb

gs.check_unb:
  %gs.unb = call i1 @value_is_unbound(%Value %gs.val)
  br i1 %gs.unb, label %gs.write, label %gs.fail

gs.write:
  ; Write mode: allocate a Compound and bind Ai to it.
  %gs.fn_ptr = inttoptr i64 %op1 to i8*
  call void @wam_arena_ensure()
  %gs.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %gs.cp_mem = call i8* @wam_arena_alloc(i64 %gs.cp_size)
  %gs.wcp = bitcast i8* %gs.cp_mem to %Compound*
  %gs.wfn_slot = getelementptr %Compound, %Compound* %gs.wcp, i32 0, i32 0
  store i8* %gs.fn_ptr, i8** %gs.wfn_slot
  %gs.war_slot = getelementptr %Compound, %Compound* %gs.wcp, i32 0, i32 1
  store i32 %gs.arity, i32* %gs.war_slot
  %gs.arity64 = zext i32 %gs.arity to i64
  %gs.args_bytes = shl i64 %gs.arity64, 4
  %gs.args_mem = call i8* @wam_arena_alloc(i64 %gs.args_bytes)
  %gs.wargs = bitcast i8* %gs.args_mem to %Value*
  %gs.wargs_slot = getelementptr %Compound, %Compound* %gs.wcp, i32 0, i32 2
  store %Value* %gs.wargs, %Value** %gs.wargs_slot
  %gs.wcp_i64 = ptrtoint %Compound* %gs.wcp to i64
  %gs.wval0 = insertvalue %Value undef, i32 3, 0
  %gs.wval = insertvalue %Value %gs.wval0, i64 %gs.wcp_i64, 1
  %gs.old = call %Value @wam_get_reg(%WamState* %vm, i32 %gs.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %gs.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %gs.ai, %Value %gs.wval)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %gs.old, %Value %gs.wval)
  call void @wam_push_write_ctx(%WamState* %vm, i32 %gs.arity)
  call void @wam_write_ctx_set_args(%WamState* %vm, %Value* %gs.wargs)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gs.read:
  ; Read mode: the register holds a Compound. It only matches if its
  ; functor name AND arity equal the expected op1/arity -- otherwise this
  ; get_structure must FAIL. Historically this block accepted ANY compound
  ; (tag == 3) without comparing the functor, so `get_structure f/N` on a
  ; compound g/M succeeded and read g''s args as if they were f''s. Multi-
  ; clause first-argument dispatch then only worked by accident, relying on
  ; the wrong clauses'' BODIES failing; a body that did not cleanly fail
  ; (e.g. length/2 on a compound) ran the wrong clause and returned garbage.
  %gs.cp_bits = extractvalue %Value %gs.val, 1
  %gs.cp_ptr = inttoptr i64 %gs.cp_bits to %Compound*
  ; arity must match
  %gs.car_slot = getelementptr %Compound, %Compound* %gs.cp_ptr, i32 0, i32 1
  %gs.car = load i32, i32* %gs.car_slot
  %gs.arity_ok = icmp eq i32 %gs.car, %gs.arity
  br i1 %gs.arity_ok, label %gs.check_fn, label %gs.fail

gs.check_fn:
  ; functor name must match (op1 is the expected functor string pointer;
  ; wam_functor_eq is pointer-fast with a strcmp fallback for dynamically
  ; built compounds whose functor came from the atom table).
  %gs.cfn_slot = getelementptr %Compound, %Compound* %gs.cp_ptr, i32 0, i32 0
  %gs.cfn = load i8*, i8** %gs.cfn_slot
  %gs.exp_fn = inttoptr i64 %op1 to i8*
  %gs.fn_ok = call i1 @wam_functor_eq(i8* %gs.cfn, i8* %gs.exp_fn)
  br i1 %gs.fn_ok, label %gs.read_ok, label %gs.fail

gs.read_ok:
  ; extract args from the Compound and push UnifyCtx
  %gs.args_slot = getelementptr %Compound, %Compound* %gs.cp_ptr, i32 0, i32 2
  %gs.args = load %Value*, %Value** %gs.args_slot
  call void @wam_push_unify_ctx(%WamState* %vm, %Value* %gs.args, i32 %gs.arity)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gs.fail:
  ret i1 false').

wam_llvm_case('get_list',
'  ; get_list: op1 = Ai register index
  ; M10: handle the Compound representation produced by the new
  ; put_list. Tag dispatch:
  ;   tag=3 Compound -> READ mode: extract args, push UnifyCtx(2)
  ;   tag=6 Unbound  -> WRITE mode: allocate [|]/2 Compound, bind Ai,
  ;                     push WriteCtx(2)
  ;   anything else (incl. Ref to non-list) -> fail
  ; The prior impl checked tag>=5 to go to write mode, which meant
  ; calls with a constructed list (tag=3) went to a stub `gl.read`
  ; that returned false -- head unification against a list literal
  ; never matched.
  %gl.ai = trunc i64 %op1 to i32
  %gl.val = call %Value @wam_get_reg_deref(%WamState* %vm, i32 %gl.ai)
  %gl.tag = extractvalue %Value %gl.val, 0
  %gl.is_cp = icmp eq i32 %gl.tag, 3
  br i1 %gl.is_cp, label %gl.read, label %gl.check_unb

gl.check_unb:
  %gl.unb_p = call i1 @value_is_unbound(%Value %gl.val)
  br i1 %gl.unb_p, label %gl.write, label %gl.fail

gl.read:
  ; Compound: it only matches if it actually IS a cons cell -- functor
  ; "[|]" (byte compare via @wam_functor_is_cons, so reader-built lists
  ; match too) AND arity 2. Historically this block accepted ANY compound
  ; (the get_structure sibling of the missing-functor-check bug): a [H|T]
  ; clause head wrongly matched e.g. foo(A,B), reading its args as if they
  ; were list head/tail.
  %gl.cp_bits = extractvalue %Value %gl.val, 1
  %gl.cp_ptr = inttoptr i64 %gl.cp_bits to %Compound*
  %gl.rfn_slot = getelementptr %Compound, %Compound* %gl.cp_ptr, i32 0, i32 0
  %gl.rfn = load i8*, i8** %gl.rfn_slot
  %gl.is_cons = call i1 @wam_functor_is_cons(i8* %gl.rfn)
  br i1 %gl.is_cons, label %gl.read_ar, label %gl.fail

gl.read_ar:
  %gl.rar_slot = getelementptr %Compound, %Compound* %gl.cp_ptr, i32 0, i32 1
  %gl.rar = load i32, i32* %gl.rar_slot
  %gl.ar_ok = icmp eq i32 %gl.rar, 2
  br i1 %gl.ar_ok, label %gl.read_ok, label %gl.fail

gl.read_ok:
  %gl.args_slot = getelementptr %Compound, %Compound* %gl.cp_ptr, i32 0, i32 2
  %gl.args = load %Value*, %Value** %gl.args_slot
  call void @wam_push_unify_ctx(%WamState* %vm, %Value* %gl.args, i32 2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gl.write:
  ; Allocate [|]/2 Compound on arena, bind Ai, push WriteCtx.
  call void @wam_arena_ensure()
  %gl.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %gl.cp_mem = call i8* @wam_arena_alloc(i64 %gl.cp_size)
  %gl.wcp = bitcast i8* %gl.cp_mem to %Compound*
  %gl.wfn_slot = getelementptr %Compound, %Compound* %gl.wcp, i32 0, i32 0
  %gl.wfn_ptr = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  store i8* %gl.wfn_ptr, i8** %gl.wfn_slot
  %gl.war_slot = getelementptr %Compound, %Compound* %gl.wcp, i32 0, i32 1
  store i32 2, i32* %gl.war_slot
  %gl.wargs_mem = call i8* @wam_arena_alloc(i64 32)
  %gl.wargs = bitcast i8* %gl.wargs_mem to %Value*
  %gl.wargs_slot = getelementptr %Compound, %Compound* %gl.wcp, i32 0, i32 2
  store %Value* %gl.wargs, %Value** %gl.wargs_slot
  %gl.wunb = call %Value @value_unbound(i8* null)
  %gl.w0_ptr = getelementptr %Value, %Value* %gl.wargs, i32 0
  store %Value %gl.wunb, %Value* %gl.w0_ptr
  %gl.w1_ptr = getelementptr %Value, %Value* %gl.wargs, i32 1
  store %Value %gl.wunb, %Value* %gl.w1_ptr
  %gl.wcp_i64 = ptrtoint %Compound* %gl.wcp to i64
  %gl.wval0 = insertvalue %Value undef, i32 3, 0
  %gl.wval = insertvalue %Value %gl.wval0, i64 %gl.wcp_i64, 1
  %gl.old = call %Value @wam_get_reg(%WamState* %vm, i32 %gl.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %gl.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %gl.ai, %Value %gl.wval)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %gl.old, %Value %gl.wval)
  call void @wam_push_write_ctx(%WamState* %vm, i32 2)
  call void @wam_write_ctx_set_args(%WamState* %vm, %Value* %gl.wargs)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gl.fail:
  ret i1 false').

wam_llvm_case('unify_variable',
'  ; unify_variable: op1 = Xn register index
  ; Read mode (UnifyCtx on stack): pop next arg, store in Xn
  ; Write mode (WriteCtx on stack): create unbound var on heap, store in Xn
  %uv.xn = trunc i64 %op1 to i32
  %uv.stype = call i32 @wam_peek_stack_type(%WamState* %vm)
  %uv.is_read = icmp eq i32 %uv.stype, 1
  br i1 %uv.is_read, label %uv.read, label %uv.write

uv.read:
  ; Read mode: get next arg from UnifyCtx
  %uv.arg = call %Value @wam_unify_ctx_next(%WamState* %vm)
  call void @wam_trail_binding(%WamState* %vm, i32 %uv.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %uv.xn, %Value %uv.arg)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uv.write:
  ; Write mode: create unbound var, push on heap, store in reg
  %uv.unb = call %Value @value_unbound(i8* null)
  %uv.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %uv.unb)
  %uv.ref = call %Value @value_ref(i32 %uv.addr)
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %uv.ref)
  call void @wam_trail_binding(%WamState* %vm, i32 %uv.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %uv.xn, %Value %uv.ref)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').


wam_llvm_case('unify_value',
'  ; unify_value: op1 = Xn register index
  ; Read mode: unify Xn with next arg from UnifyCtx
  ; Write mode: push Xn value onto heap
  %uvl.xn = trunc i64 %op1 to i32
  %uvl.stype = call i32 @wam_peek_stack_type(%WamState* %vm)
  %uvl.is_read = icmp eq i32 %uvl.stype, 1
  br i1 %uvl.is_read, label %uvl.read, label %uvl.write

uvl.read:
  ; Read mode: get expected arg, unify with register
  %uvl.expected = call %Value @wam_unify_ctx_next(%WamState* %vm)
  %uvl.actual = call %Value @wam_get_reg(%WamState* %vm, i32 %uvl.xn)
  %uvl.exp_unb = call i1 @value_is_unbound(%Value %uvl.expected)
  %uvl.act_unb = call i1 @value_is_unbound(%Value %uvl.actual)
  ; If either side is directly unbound, keep the original bind-the-
  ; unbound-one logic below. M138: when BOTH are bound, this is a
  ; unification (the old @value_equals wrongly failed bound-bound
  ; compounds with unifiable args, and compared Ref-wrapped values
  ; by address since neither side is dereffed here).
  %uvl.any_unb = or i1 %uvl.exp_unb, %uvl.act_unb
  br i1 %uvl.any_unb, label %uvl.read_ok, label %uvl.both_bound

uvl.both_bound:
  %uvl.eq = call i1 @wam_unify_value(%WamState* %vm, %Value %uvl.expected, %Value %uvl.actual)
  br i1 %uvl.eq, label %uvl.read_done, label %uvl.fail

uvl.read_ok:
  ; If actual is unbound, bind it to expected
  br i1 %uvl.act_unb, label %uvl.bind, label %uvl.read_done

uvl.bind:
  call void @wam_trail_binding(%WamState* %vm, i32 %uvl.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %uvl.xn, %Value %uvl.expected)
  br label %uvl.read_done

uvl.read_done:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uvl.write:
  ; Write mode: push register value onto heap
  %uvl.val = call %Value @wam_get_reg(%WamState* %vm, i32 %uvl.xn)
  %uvl.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %uvl.val)
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %uvl.val)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uvl.fail:
  ret i1 false').

wam_llvm_case('unify_constant',
'  ; unify_constant: op1 = constant value (packed), op2 = tag << 16.
  ; The tag (0 = Atom, 1 = Integer) was historically missing: the
  ; comparison value was always built Atom-tagged, so an integer
  ; constant in a list/structure head (p([44|T])) compared Atom(44)
  ; against the heap Integer(44) and never matched. Read mode now
  ; also goes through the general unifier, so an unbound sub-arg is
  ; BOUND to the constant (with trailing) instead of silently
  ; "matching" while staying unbound.
  %uc.stype = call i32 @wam_peek_stack_type(%WamState* %vm)
  %uc.is_read = icmp eq i32 %uc.stype, 1
  %uc.op2_32 = trunc i64 %op2 to i32
  %uc.tag = lshr i32 %uc.op2_32, 16
  %uc.val = insertvalue %Value undef, i32 %uc.tag, 0
  %uc.val2 = insertvalue %Value %uc.val, i64 %op1, 1
  br i1 %uc.is_read, label %uc.read, label %uc.write

uc.read:
  %uc.expected = call %Value @wam_unify_ctx_next(%WamState* %vm)
  %uc.ok = call i1 @wam_unify_value(%WamState* %vm, %Value %uc.expected, %Value %uc.val2)
  br i1 %uc.ok, label %uc.read_ok, label %uc.fail

uc.read_ok:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uc.write:
  ; Write mode: push constant onto heap
  %uc.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %uc.val2)
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %uc.val2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uc.fail:
  ret i1 false').


% --- Body Construction Instructions ---

wam_llvm_case('put_constant',
'  ; op1 = constant value (packed), op2 = (tag << 16) | reg_idx.
  ; Lower 16 bits: register index. Upper bits: %Value tag.
  ; 0 = Atom, 1 = Integer, 2 = Float, etc. See value.ll.mustache.
  %pc.op2_32 = trunc i64 %op2 to i32
  %pc.reg_idx = and i32 %pc.op2_32, 65535
  %pc.tag_i32 = lshr i32 %pc.op2_32, 16
  %pc.val = insertvalue %Value undef, i32 %pc.tag_i32, 0
  %pc.val2 = insertvalue %Value %pc.val, i64 %op1, 1
  ; M140: bind-through only for non-argument registers. Writing an
  ; A register is pure staging for the next goal -- the old
  ; bind-through bound any live unbound variable whose Ref remained
  ; there (``var(X), atom(foo)'' bound X to foo). X/Y registers can
  ; hold set_variable placeholder Refs from top-down structure
  ; chaining, where binding through IS the linking mechanism.
  call void @wam_trail_binding(%WamState* %vm, i32 %pc.reg_idx)
  %pc.is_areg = icmp ult i32 %pc.reg_idx, 16
  br i1 %pc.is_areg, label %pc.store, label %pc.chain

pc.chain:
  %pc.old = call %Value @wam_get_reg(%WamState* %vm, i32 %pc.reg_idx)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pc.old, %Value %pc.val2)
  br label %pc.store

pc.store:
  call void @wam_set_reg(%WamState* %vm, i32 %pc.reg_idx, %Value %pc.val2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').


wam_llvm_case('put_variable',
'  ; op1 = Xn index, op2 = Ai index
  ; Allocate a fresh heap cell holding Unbound, then place Ref{addr}
  ; into BOTH Xn and Ai so they alias through the heap. Subsequent
  ; binding builtins (functor/3, arg/3, is/2, =/2, get_value) write
  ; through the Ref via wam_bind_reg, and both registers see the bound
  ; value via deref. Without this aliasing, binding Ai leaves Xn
  ; stuck on a stale Unbound sentinel — the bug that caused
  ; sum_ints / fib / term_depth to compute wrong results.
  ; M139: no bind-through of Ai''s previous occupant. Two put_variable
  ; in a row over the same Ai (e.g. staging args for consecutive
  ; builtin calls) left the first variable''s unbound Ref in Ai; the
  ; old @wam_bind_through_if_unbound_ref(old_Ai, fresh_ref) then bound
  ; that LIVE variable to the new fresh one, silently aliasing two
  ; unrelated variables. Overwriting an argument register must never
  ; bind what it previously held.
  %pv.xn = trunc i64 %op1 to i32
  %pv.ai = trunc i64 %op2 to i32
  %pv.unb = call %Value @value_unbound(i8* null)
  %pv.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %pv.unb)
  %pv.ref = call %Value @value_ref(i32 %pv.addr)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.xn)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.xn, %Value %pv.ref)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.ai, %Value %pv.ref)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').


wam_llvm_case('put_value',
'  ; op1 = Xn index, op2 = Ai index
  ; M139: pure register copy, per standard WAM. This previously also
  ; called @wam_bind_through_if_unbound_ref(old_Ai, val), which BOUND
  ; whatever unbound variable happened to occupy Ai before the copy --
  ; aliasing two unrelated variables. E.g. in
  ;   var(X), R is 1, X = x
  ; the put_value Y2, A1 staging R for is/2 found A1 still holding
  ; X''s Ref and bound X to R, so X = x then saw 1. Overwriting an
  ; argument register must never bind its previous occupant.
  %pvl.xn = trunc i64 %op1 to i32
  %pvl.ai = trunc i64 %op2 to i32
  %pvl.val = call %Value @wam_get_reg(%WamState* %vm, i32 %pvl.xn)
  call void @wam_trail_binding(%WamState* %vm, i32 %pvl.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pvl.ai, %Value %pvl.val)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').


wam_llvm_case('put_structure',
'  ; put_structure: op1 = ptrtoint of functor string, op2 = (arity << 16) | reg_idx
  ; Allocate %Compound struct + args array from the arena (not malloc).
  ; Compounds are transient and reclaimed on @wam_cleanup().
  %ps.fn_ptr = inttoptr i64 %op1 to i8*
  %ps.op2_32 = trunc i64 %op2 to i32
  %ps.arity = lshr i32 %ps.op2_32, 16
  %ps.ai = and i32 %ps.op2_32, 65535
  call void @wam_arena_ensure()
  ; Allocate %Compound struct from arena.
  %ps.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %ps.cp_mem = call i8* @wam_arena_alloc(i64 %ps.cp_size)
  %ps.cp = bitcast i8* %ps.cp_mem to %Compound*
  %ps.fn_slot = getelementptr %Compound, %Compound* %ps.cp, i32 0, i32 0
  store i8* %ps.fn_ptr, i8** %ps.fn_slot
  %ps.ar_slot = getelementptr %Compound, %Compound* %ps.cp, i32 0, i32 1
  store i32 %ps.arity, i32* %ps.ar_slot
  ; Allocate args array from arena.
  %ps.arity64 = zext i32 %ps.arity to i64
  %ps.args_bytes = shl i64 %ps.arity64, 4
  %ps.args_mem = call i8* @wam_arena_alloc(i64 %ps.args_bytes)
  %ps.args = bitcast i8* %ps.args_mem to %Value*
  %ps.args_slot = getelementptr %Compound, %Compound* %ps.cp, i32 0, i32 2
  store %Value* %ps.args, %Value** %ps.args_slot
  %ps.cp_i64 = ptrtoint %Compound* %ps.cp to i64
  %ps.val0 = insertvalue %Value undef, i32 3, 0
  %ps.val = insertvalue %Value %ps.val0, i64 %ps.cp_i64, 1
  ; M140: bind-through only for non-argument registers (see
  ; put_constant). A-reg writes are staging -- the old unconditional
  ; bind-through bound a live variable left in Ai to the freshly
  ; built compound (``var(X), compound(f(a))'' bound X to f(a)).
  ; X/Y-reg writes are top-down structure chaining: set_variable Xn
  ; left a Ref to the parent''s arg slot in Xn, and binding through
  ; it links this new cell into the parent (e.g. the [33,11,22]
  ; build in test_msort_head).
  call void @wam_trail_binding(%WamState* %vm, i32 %ps.ai)
  %ps.is_areg = icmp ult i32 %ps.ai, 16
  br i1 %ps.is_areg, label %ps.store, label %ps.chain

ps.chain:
  %ps.old = call %Value @wam_get_reg(%WamState* %vm, i32 %ps.ai)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %ps.old, %Value %ps.val)
  br label %ps.store

ps.store:
  call void @wam_set_reg(%WamState* %vm, i32 %ps.ai, %Value %ps.val)
  call void @wam_push_write_ctx(%WamState* %vm, i32 %ps.arity)
  call void @wam_write_ctx_set_args(%WamState* %vm, %Value* %ps.args)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('put_list',
'  ; put_list: op1 = Ai register index
  ; M10: build a [|]/2 Compound on the arena (matches put_structure''s
  ; representation). Prior impl used a heap-marker (Atom sentinel +
  ; 2 unbound cells) which deref''d to an Atom -- the resulting list
  ; was incompatible with put_structure [|]/2 (Compound) and with
  ; findall''s collect_case output, so any cross-built unification
  ; silently failed (Compound vs Atom tag mismatch).
  %pl.ai = trunc i64 %op1 to i32
  call void @wam_arena_ensure()
  ; Allocate %Compound + 2-arg args array on arena.
  %pl.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %pl.cp_mem = call i8* @wam_arena_alloc(i64 %pl.cp_size)
  %pl.cp = bitcast i8* %pl.cp_mem to %Compound*
  %pl.fn_slot = getelementptr %Compound, %Compound* %pl.cp, i32 0, i32 0
  %pl.fn_ptr = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  store i8* %pl.fn_ptr, i8** %pl.fn_slot
  %pl.ar_slot = getelementptr %Compound, %Compound* %pl.cp, i32 0, i32 1
  store i32 2, i32* %pl.ar_slot
  %pl.args_mem = call i8* @wam_arena_alloc(i64 32)
  %pl.args = bitcast i8* %pl.args_mem to %Value*
  %pl.args_slot = getelementptr %Compound, %Compound* %pl.cp, i32 0, i32 2
  store %Value* %pl.args, %Value** %pl.args_slot
  ; Init both args to Unbound so unify mode runs against valid cells.
  %pl.unb = call %Value @value_unbound(i8* null)
  %pl.a0_ptr = getelementptr %Value, %Value* %pl.args, i32 0
  store %Value %pl.unb, %Value* %pl.a0_ptr
  %pl.a1_ptr = getelementptr %Value, %Value* %pl.args, i32 1
  store %Value %pl.unb, %Value* %pl.a1_ptr
  ; Wrap as %Value{tag=3 Compound, payload=ptr}.
  %pl.cp_i64 = ptrtoint %Compound* %pl.cp to i64
  %pl.val0 = insertvalue %Value undef, i32 3, 0
  %pl.val = insertvalue %Value %pl.val0, i64 %pl.cp_i64, 1
  ; M140: bind-through only for non-argument registers (see
  ; put_structure -- same staging-vs-chaining split).
  call void @wam_trail_binding(%WamState* %vm, i32 %pl.ai)
  %pl.is_areg = icmp ult i32 %pl.ai, 16
  br i1 %pl.is_areg, label %pl.store, label %pl.chain

pl.chain:
  %pl.old = call %Value @wam_get_reg(%WamState* %vm, i32 %pl.ai)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pl.old, %Value %pl.val)
  br label %pl.store

pl.store:
  call void @wam_set_reg(%WamState* %vm, i32 %pl.ai, %Value %pl.val)
  ; WriteCtx writes into args[0..1] via set_constant/set_variable/set_value.
  call void @wam_push_write_ctx(%WamState* %vm, i32 2)
  call void @wam_write_ctx_set_args(%WamState* %vm, %Value* %pl.args)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('set_variable',
'  ; set_variable: op1 = Xn register index
  ; Allocate a fresh heap cell holding Unbound, then place Ref{addr}
  ; into BOTH the compound args (via WriteCtx) and Xn so they alias.
  %sv.xn = trunc i64 %op1 to i32
  %sv.unb = call %Value @value_unbound(i8* null)
  %sv.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %sv.unb)
  %sv.ref = call %Value @value_ref(i32 %sv.addr)
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %sv.ref)
  call void @wam_trail_binding(%WamState* %vm, i32 %sv.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %sv.xn, %Value %sv.ref)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').


wam_llvm_case('set_value',
'  ; set_value: op1 = Xn register index
  ; Write Xn value into the compound args array via WriteCtx.
  ; M10: if Xn is a direct Unbound (no backing heap cell -- usually
  ; because get_variable copied a direct-Unbound input arg verbatim),
  ; promote it to a fresh Ref-into-heap and update Xn so subsequent
  ; uses see the same Ref. Without this, the compound arg holds a
  ; direct Unbound that no unification path can bind back through.
  %sve.xn = trunc i64 %op1 to i32
  %sve.val = call %Value @wam_get_reg(%WamState* %vm, i32 %sve.xn)
  %sve.tag = extractvalue %Value %sve.val, 0
  %sve.is_unb = icmp eq i32 %sve.tag, 6
  br i1 %sve.is_unb, label %sve.promote, label %sve.write

sve.promote:
  %sve.unb = call %Value @value_unbound(i8* null)
  %sve.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %sve.unb)
  %sve.ref = call %Value @value_ref(i32 %sve.addr)
  call void @wam_trail_binding(%WamState* %vm, i32 %sve.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %sve.xn, %Value %sve.ref)
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %sve.ref)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

sve.write:
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %sve.val)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('set_constant',
'  ; set_constant: op1 = constant payload, op2 = tag (0=atom, 1=integer)
  ; Write constant into the compound args array via WriteCtx.
  %sc.tag = trunc i64 %op2 to i32
  %sc.val = insertvalue %Value undef, i32 %sc.tag, 0
  %sc.val2 = insertvalue %Value %sc.val, i64 %op1, 1
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %sc.val2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% --- Control Instructions ---

wam_llvm_case('allocate',
'  ; Push environment frame: save CP on stack and snapshot Y-regs.
  ; Y-regs (regs[16..31]) share physical slots with X-regs in this
  ; target; save/restore at allocate/deallocate boundaries gives each
  ; clause its own isolated Y-space — an inner predicate writing its
  ; Y-regs no longer clobbers the outer caller. Canonical WAM stores
  ; Y-regs in env frames directly; we snapshot instead to keep every
  ; other instruction body unchanged.
  ;
  ; M5: ensure the stack has room before reading its pointer (grow
  ; may realloc the buffer to a new address).
  call void @wam_stack_ensure_capacity(%WamState* %vm)
  %alloc.ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3
  %alloc.ss = load i32, i32* %alloc.ss_ptr
  %alloc.stack_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 2
  %alloc.stack = load %StackEntry*, %StackEntry** %alloc.stack_ptr
  %alloc.entry = getelementptr %StackEntry, %StackEntry* %alloc.stack, i32 %alloc.ss
  ; type = 0 (EnvFrame)
  %alloc.type_ptr = getelementptr %StackEntry, %StackEntry* %alloc.entry, i32 0, i32 0
  store i32 0, i32* %alloc.type_ptr
  ; aux = current CP
  %alloc.cp = call i32 @wam_get_cp(%WamState* %vm)
  %alloc.cp_ext = zext i32 %alloc.cp to i64
  %alloc.aux_ptr = getelementptr %StackEntry, %StackEntry* %alloc.entry, i32 0, i32 1
  store i64 %alloc.cp_ext, i64* %alloc.aux_ptr

  ; Save current cut_barrier into EnvFrame and update it to current cpn
  %alloc.cb_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 23
  %alloc.old_cb = load i32, i32* %alloc.cb_ptr
  %alloc.scb_ptr = getelementptr %StackEntry, %StackEntry* %alloc.entry, i32 0, i32 4
  store i32 %alloc.old_cb, i32* %alloc.scb_ptr

  %alloc.le_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 24
  %alloc.le_cpn = load i32, i32* %alloc.le_ptr
  store i32 %alloc.le_cpn, i32* %alloc.cb_ptr

  ; M10: snapshot regs[48..95] (the Y-reg window) into the env frame.
  ; Y1..Y48 map to 48..95 under the disjoint X/Y ABI -- see
  ; bindings/llvm_wam_bindings.pl reg_name_to_index. Pre-M10 the
  ; snapshot covered 16..63 (X and Y mixed); now only Y space needs
  ; protection (X space dies at calls per canonical WAM). The register
  ; file is [128 x %Value] and y_save holds 48 permanents.
  %alloc.y_src = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 48
  %alloc.y_dst = getelementptr %StackEntry, %StackEntry* %alloc.entry, i32 0, i32 3, i32 0
  %alloc.y_src_i8 = bitcast %Value* %alloc.y_src to i8*
  %alloc.y_dst_i8 = bitcast %Value* %alloc.y_dst to i8*
  ; 48 Values * 16 bytes = 768 bytes.
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %alloc.y_dst_i8, i8* %alloc.y_src_i8, i64 768, i1 false)
  ; Increment stack size
  %alloc.new_ss = add i32 %alloc.ss, 1
  store i32 %alloc.new_ss, i32* %alloc.ss_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('deallocate',
'  ; Pop environment frame: scan backward for EnvFrame (type == 0), restore CP
  %dealloc.ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3
  %dealloc.ss = load i32, i32* %dealloc.ss_ptr
  %dealloc.has_frames = icmp sgt i32 %dealloc.ss, 0
  br i1 %dealloc.has_frames, label %dealloc.scan, label %dealloc.done

dealloc.scan:
  ; Scan backward from top of stack looking for EnvFrame
  %dealloc.stack_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 2
  %dealloc.stack = load %StackEntry*, %StackEntry** %dealloc.stack_ptr
  br label %dealloc.loop

dealloc.loop:
  %dealloc.idx = phi i32 [%dealloc.ss, %dealloc.scan], [%dealloc.prev_idx, %dealloc.skip]
  %dealloc.prev_idx = sub i32 %dealloc.idx, 1
  %dealloc.exhausted = icmp slt i32 %dealloc.prev_idx, 0
  br i1 %dealloc.exhausted, label %dealloc.done, label %dealloc.check

dealloc.check:
  %dealloc.entry = getelementptr %StackEntry, %StackEntry* %dealloc.stack, i32 %dealloc.prev_idx
  %dealloc.type_ptr = getelementptr %StackEntry, %StackEntry* %dealloc.entry, i32 0, i32 0
  %dealloc.type = load i32, i32* %dealloc.type_ptr
  %dealloc.is_env = icmp eq i32 %dealloc.type, 0
  br i1 %dealloc.is_env, label %dealloc.restore, label %dealloc.skip

dealloc.skip:
  br label %dealloc.loop

dealloc.restore:
  ; Restore CP from saved value in the EnvFrame
  %dealloc.aux_ptr = getelementptr %StackEntry, %StackEntry* %dealloc.entry, i32 0, i32 1
  %dealloc.saved_cp = load i64, i64* %dealloc.aux_ptr
  %dealloc.cp = trunc i64 %dealloc.saved_cp to i32
  call void @wam_set_cp(%WamState* %vm, i32 %dealloc.cp)

  ; Restore cut_barrier
  %dealloc.scb_ptr = getelementptr %StackEntry, %StackEntry* %dealloc.entry, i32 0, i32 4
  %dealloc.saved_cb = load i32, i32* %dealloc.scb_ptr
  %dealloc.cb_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 23
  store i32 %dealloc.saved_cb, i32* %dealloc.cb_ptr

  ; M10: restore regs[48..95] (Y-reg window) from the env frame
  ; snapshot. See allocate for the save side (48 Values = 768 bytes).
  %dealloc.y_src = getelementptr %StackEntry, %StackEntry* %dealloc.entry, i32 0, i32 3, i32 0
  %dealloc.y_dst = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 48
  %dealloc.y_src_i8 = bitcast %Value* %dealloc.y_src to i8*
  %dealloc.y_dst_i8 = bitcast %Value* %dealloc.y_dst to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dealloc.y_dst_i8, i8* %dealloc.y_src_i8, i64 768, i1 false)
  ; Pop stack down to this frame (exclusive)
  store i32 %dealloc.prev_idx, i32* %dealloc.ss_ptr
  br label %dealloc.done

dealloc.done:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('do_call',
'  ; op1 = label index, -1 meta-call, -3 retract/1, -5 catch/3, -6 throw/1
  %call.label = trunc i64 %op1 to i32
  %call.pc = call i32 @wam_get_pc(%WamState* %vm)
  %call.next = add i32 %call.pc, 1
  %call.is_retract = icmp eq i32 %call.label, -3
  br i1 %call.is_retract, label %call.retract, label %call.check_catch

call.retract:
  ; retract/1: the pattern is in reg 0; consult + remove + backtrack.
  %call.retract_ok = call i1 @wam_dyn_retract_consult(%WamState* %vm, i32 %call.next)
  br i1 %call.retract_ok, label %call.meta_done, label %call.fail

call.check_catch:
  %call.is_catch = icmp eq i32 %call.label, -5
  br i1 %call.is_catch, label %call.catch, label %call.check_throw

call.catch:
  ; catch/3: A1=Goal, A2=Catcher, A3=Recovery; push a frame and run Goal.
  %call.catch_ok = call i1 @wam_catch_setup(%WamState* %vm, i32 %call.next)
  br i1 %call.catch_ok, label %call.meta_done, label %call.fail

call.check_throw:
  %call.is_throw = icmp eq i32 %call.label, -6
  br i1 %call.is_throw, label %call.throw, label %call.check_meta

call.throw:
  ; throw/1: unwind to the nearest matching catch frame and run Recovery.
  %call.throw_ok = call i1 @wam_throw(%WamState* %vm)
  br i1 %call.throw_ok, label %call.meta_done, label %call.fail

call.check_meta:
  %call.is_meta = icmp slt i32 %call.label, 0
  br i1 %call.is_meta, label %call.meta, label %call.static

call.meta:
  %call.meta_arity = trunc i64 %op2 to i32
  %call.meta_ok = call i1 @wam_dispatch_meta_call(%WamState* %vm, i32 %call.meta_arity, i32 %call.next)
  br i1 %call.meta_ok, label %call.meta_done, label %call.fail

call.meta_done:
  ret i1 true

call.static:
  %call.target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %call.label)
  %call.valid = icmp sge i32 %call.target_pc, 0
  br i1 %call.valid, label %call.go, label %call.fail

call.go:
  ; Save continuation
  call void @wam_set_cp(%WamState* %vm, i32 %call.next)
  ; Save cpn for cut_barrier
  %call.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %call.cpn = load i32, i32* %call.cpn_ptr
  %call.le_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 24
  store i32 %call.cpn, i32* %call.le_ptr
  call void @wam_set_pc(%WamState* %vm, i32 %call.target_pc)
  ret i1 true

call.fail:
  ret i1 false').

wam_llvm_case('do_execute',
'  ; op1 = label index, -1 meta-call, -3 retract/1, -5 catch/3, -6 throw/1
  %exec.label = trunc i64 %op1 to i32
  %exec.is_retract = icmp eq i32 %exec.label, -3
  br i1 %exec.is_retract, label %exec.retract, label %exec.check_catch

exec.retract:
  ; retract/1 in last-call position: continuation is the current CP.
  %exec.rpc = call i32 @wam_get_cp(%WamState* %vm)
  %exec.retract_ok = call i1 @wam_dyn_retract_consult(%WamState* %vm, i32 %exec.rpc)
  br i1 %exec.retract_ok, label %exec.meta_done, label %exec.fail

exec.check_catch:
  %exec.is_catch = icmp eq i32 %exec.label, -5
  br i1 %exec.is_catch, label %exec.catch, label %exec.check_throw

exec.catch:
  %exec.crpc = call i32 @wam_get_cp(%WamState* %vm)
  %exec.catch_ok = call i1 @wam_catch_setup(%WamState* %vm, i32 %exec.crpc)
  br i1 %exec.catch_ok, label %exec.meta_done, label %exec.fail

exec.check_throw:
  %exec.is_throw = icmp eq i32 %exec.label, -6
  br i1 %exec.is_throw, label %exec.throw, label %exec.check_meta

exec.throw:
  %exec.throw_ok = call i1 @wam_throw(%WamState* %vm)
  br i1 %exec.throw_ok, label %exec.meta_done, label %exec.fail

exec.check_meta:
  %exec.is_meta = icmp slt i32 %exec.label, 0
  br i1 %exec.is_meta, label %exec.meta, label %exec.static

exec.meta:
  %exec.meta_arity = trunc i64 %op2 to i32
  %exec.after = call i32 @wam_get_cp(%WamState* %vm)
  %exec.meta_ok = call i1 @wam_dispatch_meta_call(%WamState* %vm, i32 %exec.meta_arity, i32 %exec.after)
  br i1 %exec.meta_ok, label %exec.meta_done, label %exec.fail

exec.meta_done:
  ret i1 true

exec.static:
  %exec.target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %exec.label)
  %exec.valid = icmp sge i32 %exec.target_pc, 0
  br i1 %exec.valid, label %exec.go, label %exec.fail

exec.go:
  ; Save cpn for cut_barrier
  %exec.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %exec.cpn = load i32, i32* %exec.cpn_ptr
  %exec.le_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 24
  store i32 %exec.cpn, i32* %exec.le_ptr
  call void @wam_set_pc(%WamState* %vm, i32 %exec.target_pc)
  ret i1 true

exec.fail:
  ret i1 false').

wam_llvm_case('proceed',
'  ; Return to continuation or halt
  %proc.cp = call i32 @wam_get_cp(%WamState* %vm)
  %proc.is_halt = icmp eq i32 %proc.cp, 0
  br i1 %proc.is_halt, label %proc.halt, label %proc.return

proc.halt:
  call void @wam_set_halted(%WamState* %vm, i1 true)
  ret i1 true

proc.return:
  call void @wam_set_pc(%WamState* %vm, i32 %proc.cp)
  call void @wam_set_cp(%WamState* %vm, i32 0)
  ret i1 true').

wam_llvm_case('builtin_call',
'  ; op1 = builtin op id, op2 = arity
  %bi.op = trunc i64 %op1 to i32
  %bi.arity = trunc i64 %op2 to i32
  %bi.result = call i1 @execute_builtin(%WamState* %vm, i32 %bi.op, i32 %bi.arity)
  br i1 %bi.result, label %bi.ok, label %bi.fail

bi.ok:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

bi.fail:
  ret i1 false').

% --- Choice Point Instructions ---

wam_llvm_case('try_me_else',
'  ; op1 = label index for alternative
  %tme.label = trunc i64 %op1 to i32
  %tme.next_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %tme.label)
  ; M5: ensure CP array has room before reading its pointer.
  call void @wam_cp_ensure_capacity(%WamState* %vm)
  ; Push choice point
  %tme.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %tme.cpn = load i32, i32* %tme.cpn_ptr
  %tme.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %tme.cps = load %ChoicePoint*, %ChoicePoint** %tme.cps_ptr
  %tme.cp_slot = getelementptr %ChoicePoint, %ChoicePoint* %tme.cps, i32 %tme.cpn
  ; Set next_pc
  %tme.npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 0
  store i32 %tme.next_pc, i32* %tme.npc_ptr
  ; Save the FULL register file (128 x %Value = 2048 bytes) so backtrack
  ; can restore Y17..Y48 (regs 64..95) clobbered by a failed clause body
  %tme.dst_regs = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 1, i32 0
  %tme.src_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %tme.dst_raw = bitcast %Value* %tme.dst_regs to i8*
  %tme.src_raw = bitcast %Value* %tme.src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tme.dst_raw, i8* %tme.src_raw, i64 2048, i1 false)
  ; Save trail mark
  %tme.ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %tme.ts = load i32, i32* %tme.ts_ptr
  %tme.tm_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 2
  store i32 %tme.ts, i32* %tme.tm_ptr
  ; Save cp
  %tme.saved_cp = call i32 @wam_get_cp(%WamState* %vm)
  %tme.scp_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 3
  store i32 %tme.saved_cp, i32* %tme.scp_ptr
  ; Initialize agg fields: agg_type = -1 (not an aggregate frame)
  %tme.at_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 4
  store i32 -1, i32* %tme.at_ptr
  %tme.avr_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 5
  store i32 0, i32* %tme.avr_ptr
  %tme.arr_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 6
  store i32 0, i32* %tme.arr_ptr
  %tme.arpc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 7
  store i32 0, i32* %tme.arpc_ptr
  ; Save heap_top — restored on backtrack so put_variable heap pushes
  ; from the failing clause do not leak into the next alternative.
  %tme.hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %tme.hs = load i32, i32* %tme.hs_ptr
  %tme.sht_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 11
  store i32 %tme.hs, i32* %tme.sht_ptr
  ; Save stack_size so failed alternatives discard environment frames
  ; allocated after this choice point. Without this, a called predicate
  ; that fails after allocate leaves a stale EnvFrame on the stack.
  %tme.ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3
  %tme.ss = load i32, i32* %tme.ss_ptr
  %tme.sss_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 13
  store i32 %tme.ss, i32* %tme.sss_ptr
  ; M10: snapshot the CURRENT cut_barrier into saved_b so trust_me /
  ; retry_me_else can restore it on this CP''s removal. Pre-M10 we
  ; stored cpn here AND overwrote the global cb with cpn, which
  ; corrupted any outer ITE/clause cut barrier that an inner
  ; try_me_else encountered. With this change cb stays at the
  ; clause-entry value (set by allocate), so `!/0` cuts to the
  ; correct level even when the ITE condition triggers an inner
  ; multi-clause dispatch.
  %tme.cb_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 23
  %tme.cur_cb = load i32, i32* %tme.cb_ptr
  %tme.saved_b_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 12
  store i32 %tme.cur_cb, i32* %tme.saved_b_ptr

  %tme.new_cpn = add i32 %tme.cpn, 1
  store i32 %tme.new_cpn, i32* %tme.cpn_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('retry_me_else',
'  ; op1 = label index for next alternative
  ; If no CP exists (e.g., entered via switch_on_constant), just advance PC.
  %rme.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %rme.cpn = load i32, i32* %rme.cpn_ptr
  %rme.has_cp = icmp sgt i32 %rme.cpn, 0
  br i1 %rme.has_cp, label %rme.update_cp, label %rme.no_cp

rme.update_cp:
  %rme.label = trunc i64 %op1 to i32
  %rme.next_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %rme.label)
  %rme.top_idx = sub i32 %rme.cpn, 1
  %rme.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %rme.cps = load %ChoicePoint*, %ChoicePoint** %rme.cps_ptr
  %rme.top = getelementptr %ChoicePoint, %ChoicePoint* %rme.cps, i32 %rme.top_idx
  %rme.npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %rme.top, i32 0, i32 0
  store i32 %rme.next_pc, i32* %rme.npc_ptr
  ; Restore cut_barrier from choice point
  %rme.saved_b_ptr = getelementptr %ChoicePoint, %ChoicePoint* %rme.top, i32 0, i32 12
  %rme.saved_b = load i32, i32* %rme.saved_b_ptr
  %rme.cb_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 23
  store i32 %rme.saved_b, i32* %rme.cb_ptr
  br label %rme.done

rme.no_cp:
  br label %rme.done

rme.done:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('trust_me',
'  ; Pop top choice point if one exists, otherwise just advance PC.
  %tm.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %tm.cpn = load i32, i32* %tm.cpn_ptr
  %tm.has_cp = icmp sgt i32 %tm.cpn, 0
  br i1 %tm.has_cp, label %tm.pop, label %tm.done

tm.pop:
  %tm.top_idx = sub i32 %tm.cpn, 1
  %tm.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %tm.cps = load %ChoicePoint*, %ChoicePoint** %tm.cps_ptr
  %tm.top = getelementptr %ChoicePoint, %ChoicePoint* %tm.cps, i32 %tm.top_idx
  ; Restore cut_barrier from choice point
  %tm.saved_b_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tm.top, i32 0, i32 12
  %tm.saved_b = load i32, i32* %tm.saved_b_ptr
  %tm.cb_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 23
  store i32 %tm.saved_b, i32* %tm.cb_ptr

  %tm.new_cpn = sub i32 %tm.cpn, 1
  store i32 %tm.new_cpn, i32* %tm.cpn_ptr
  br label %tm.done

tm.done:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('switch_on_constant',
'  ; op1 = ptrtoint of %SwitchEntry* table
  ; op2 = entry count
  %soc.table = inttoptr i64 %op1 to %SwitchEntry*
  %soc.count = trunc i64 %op2 to i32
  %soc.result = call i32 @wam_switch_on_constant(%WamState* %vm, %SwitchEntry* %soc.table, i32 %soc.count)
  ; 0 = no match (backtrack), 1 = matched (PC updated), 2 = unbound (PC advanced)
  %soc.ok = icmp ne i32 %soc.result, 0
  ret i1 %soc.ok').

% switch_on_structure: nop fallthrough — safe because the try_me_else/retry_me_else
% chain still produces correct results. Proper implementation is a follow-up milestone.
wam_llvm_case('switch_on_structure',
'  ; Nop fallthrough: just advance PC and continue to the try_me_else chain.
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% switch_on_constant_a2: like switch_on_constant but indexes on A2.
wam_llvm_case('switch_on_constant_a2',
'  ; op1 = ptrtoint of %SwitchEntry* table
  ; op2 = entry count
  %soca2.table = inttoptr i64 %op1 to %SwitchEntry*
  %soca2.count = trunc i64 %op2 to i32
  %soca2.result = call i32 @wam_switch_on_constant_a2(%WamState* %vm, %SwitchEntry* %soca2.table, i32 %soca2.count)
  %soca2.ok = icmp ne i32 %soca2.result, 0
  ret i1 %soca2.ok').

% begin_aggregate: push an aggregate-frame choice point and reset accumulator.
% op1 = (agg_type)
% op2 = (value_reg_idx << 16) | result_reg_idx
wam_llvm_case('begin_aggregate',
'  %ba.agg_type = trunc i64 %op1 to i32
  %ba.op2_trunc = trunc i64 %op2 to i32
  %ba.val_reg = lshr i32 %ba.op2_trunc, 16
  %ba.res_reg = and i32 %ba.op2_trunc, 65535

  ; Compute the continuation PC = (matching end_aggregate PC) + 1 by scanning
  ; forward from this instruction for the matching tag-29, tracking nesting so
  ; inner aggregates are skipped. Doing this at begin time -- rather than
  ; relying on end_aggregate to set agg_return_pc -- keeps the return PC
  ; correct even when the goal yields ZERO solutions and end_aggregate never
  ; runs. Otherwise finalize jumps to the placeholder PC 0 and the predicate
  ; re-executes forever (empty findall / setof / bagof).
  %ba.code_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 15
  %ba.code = load %Instruction*, %Instruction** %ba.code_ptr
  %ba.clen_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 16
  %ba.clen = load i32, i32* %ba.clen_ptr
  %ba.begin_pc = call i32 @wam_get_pc(%WamState* %vm)
  %ba.scan_start = add i32 %ba.begin_pc, 1
  br label %ba.scan

ba.scan:
  %ba.si = phi i32 [ %ba.scan_start, %begin_aggregate ], [ %ba.si_next, %ba.scan_cont ]
  %ba.depth = phi i32 [ 0, %begin_aggregate ], [ %ba.depth_next, %ba.scan_cont ]
  %ba.oob = icmp sge i32 %ba.si, %ba.clen
  br i1 %ba.oob, label %ba.scan_fallback, label %ba.scan_check

ba.scan_check:
  %ba.instr = getelementptr %Instruction, %Instruction* %ba.code, i32 %ba.si
  %ba.tag_ptr = getelementptr %Instruction, %Instruction* %ba.instr, i32 0, i32 0
  %ba.itag = load i32, i32* %ba.tag_ptr
  %ba.is_begin = icmp eq i32 %ba.itag, 28
  br i1 %ba.is_begin, label %ba.nest, label %ba.maybe_end

ba.nest:
  %ba.depth_inc = add i32 %ba.depth, 1
  br label %ba.scan_cont

ba.maybe_end:
  %ba.is_end = icmp eq i32 %ba.itag, 29
  br i1 %ba.is_end, label %ba.at_end, label %ba.scan_cont

ba.at_end:
  %ba.depth_zero = icmp eq i32 %ba.depth, 0
  br i1 %ba.depth_zero, label %ba.found, label %ba.unnest

ba.unnest:
  %ba.depth_dec = sub i32 %ba.depth, 1
  br label %ba.scan_cont

ba.scan_cont:
  %ba.depth_next = phi i32 [ %ba.depth_inc, %ba.nest ], [ %ba.depth, %ba.maybe_end ], [ %ba.depth_dec, %ba.unnest ]
  %ba.si_next = add i32 %ba.si, 1
  br label %ba.scan

ba.found:
  %ba.cont_pc = add i32 %ba.si, 1
  br label %ba.setup

ba.scan_fallback:
  ; No matching end_aggregate found (should not happen for compiler output);
  ; fall back to begin_pc+1 so we at least make forward progress.
  br label %ba.setup

ba.setup:
  %ba.arpc_val = phi i32 [ %ba.cont_pc, %ba.found ], [ %ba.scan_start, %ba.scan_fallback ]

  ; M5: ensure CP array has room before reading its pointer.
  call void @wam_cp_ensure_capacity(%WamState* %vm)
  ; Push a choice point
  %ba.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %ba.cpn = load i32, i32* %ba.cpn_ptr
  %ba.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %ba.cps = load %ChoicePoint*, %ChoicePoint** %ba.cps_ptr
  %ba.cp_slot = getelementptr %ChoicePoint, %ChoicePoint* %ba.cps, i32 %ba.cpn

  ; next_pc: unused for aggregate frames (finalize uses agg_return_pc instead)
  %ba.npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 0
  store i32 0, i32* %ba.npc_ptr

  ; Save registers
  %ba.dst_regs = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 1, i32 0
  %ba.src_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %ba.dst_raw = bitcast %Value* %ba.dst_regs to i8*
  %ba.src_raw = bitcast %Value* %ba.src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %ba.dst_raw, i8* %ba.src_raw, i64 2048, i1 false)

  ; Save trail mark
  %ba.ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %ba.ts = load i32, i32* %ba.ts_ptr
  %ba.tm_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 2
  store i32 %ba.ts, i32* %ba.tm_ptr

  ; Save cp
  %ba.saved_cp = call i32 @wam_get_cp(%WamState* %vm)
  %ba.scp_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 3
  store i32 %ba.saved_cp, i32* %ba.scp_ptr

  ; Set agg fields
  %ba.at_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 4
  store i32 %ba.agg_type, i32* %ba.at_ptr
  %ba.avr_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 5
  store i32 %ba.val_reg, i32* %ba.avr_ptr
  %ba.arr_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 6
  store i32 %ba.res_reg, i32* %ba.arr_ptr
  %ba.arpc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 7
  store i32 %ba.arpc_val, i32* %ba.arpc_ptr  ; continuation PC after end_aggregate
                                             ; (correct even for zero solutions)

  ; Save heap_top so unwind via wam_finalize_aggregate can rewind.
  %ba.hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %ba.hs = load i32, i32* %ba.hs_ptr
  %ba.sht_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 11
  store i32 %ba.hs, i32* %ba.sht_ptr

  ; Save stack_size (campaign finding no. 14): an inner goal that
  ; ALLOCATES an environment and then fails leaves its frame orphaned --
  ; ordinary CPs save/restore stack_size (try_me_else / backtrack), but
  ; the aggregate frame did not, so the caller''s later deallocate popped
  ; the orphan instead of its own frame. Masked when every inner clause
  ; is allocate-free (simple fact bodies); fatal for any rule-bodied
  ; inner goal that fails -- the empty-findall case.
  %ba.sss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3
  %ba.sss = load i32, i32* %ba.sss_ptr
  %ba.csss_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 13
  store i32 %ba.sss, i32* %ba.csss_ptr

  ; Increment choice point count
  %ba.new_cpn = add i32 %ba.cpn, 1
  store i32 %ba.new_cpn, i32* %ba.cpn_ptr

  ; Reset accumulator count
  %ba.accnt_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 21
  store i32 0, i32* %ba.accnt_ptr

  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% end_aggregate: push value to accumulator, update nearest agg frame's
% return PC, then fail to trigger backtrack (which calls finalize).
% op1 = value_reg_idx
wam_llvm_case('end_aggregate',
'  %ea.val_reg = trunc i64 %op1 to i32
  ; M11: freeze (deref + deep-copy compounds) before push. Per-
  ; iteration values are typically Refs into the volatile heap
  ; region; backtrack between iterations rewinds the heap, so a
  ; raw Ref ends up dangling and every accumulated entry collapses
  ; to the same recycled address. wam_freeze_value follows the
  ; Ref, captures atomic values directly, and for Compounds
  ; recursively copies the args onto the arena so the result is
  ; self-contained -- enabling findall(p(X,Y), ..., L) of compound
  ; templates and setof/sort of compound elements.
  %ea.val_raw = call %Value @wam_get_reg(%WamState* %vm, i32 %ea.val_reg)
  %ea.val = call %Value @wam_freeze_value(%WamState* %vm, %Value %ea.val_raw)

  ; Push value to accumulator
  call void @wam_agg_push(%WamState* %vm, %Value %ea.val)

  ; Update the nearest aggregate frame''s return_pc = current PC + 1
  %ea.pc = call i32 @wam_get_pc(%WamState* %vm)
  %ea.ret_pc = add i32 %ea.pc, 1
  call void @wam_update_agg_return_pc(%WamState* %vm, i32 %ea.ret_pc)

  ; Fail to force backtrack — backtrack will check the agg frame and
  ; either re-run inner goals (if there are prior CPs) or finalize.
  ret i1 false').

% call_foreign: dispatch to a native foreign kernel.
% op1 = foreign kind ID (see wam_llvm_foreign_kind_id/2)
% op2 = instance_id (M5.8 — selects which of possibly several
%   foreign-lowered predicates of the same kind should run). Arity is
%   implicit per kernel kind (td3=3, wsp3=3, astar=4) so it doesn't
%   need to live in the instruction.
% On success, advance PC then return true so the run loop continues
% to the next instruction. On failure, return false without advancing
% so the run loop backtracks from the current PC.
wam_llvm_case('call_foreign',
'  %cf.kind = trunc i64 %op1 to i32
  %cf.instance = trunc i64 %op2 to i32
  %cf.result = call i1 @wam_execute_foreign_predicate(%WamState* %vm, i32 %cf.kind, i32 %cf.instance)
  br i1 %cf.result, label %cf.success, label %cf.fail

cf.success:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

cf.fail:
  ret i1 false').

% --- If-then-else control flow ---

wam_llvm_case('cut_ite',
'  ; Soft cut: pop only the most recent choice point (the ITE guard CP),
  ; preserving any outer CPs. Unlike !/0 which zeros cp_count, cut_ite
  ; decrements by 1 iff cp_count > 0.
  %ci.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %ci.cpn = load i32, i32* %ci.cpn_ptr
  %ci.has_cp = icmp sgt i32 %ci.cpn, 0
  br i1 %ci.has_cp, label %ci.pop, label %ci.advance

ci.pop:
  %ci.new_cpn = sub i32 %ci.cpn, 1
  store i32 %ci.new_cpn, i32* %ci.cpn_ptr
  br label %ci.advance

ci.advance:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('jump',
'  ; Unconditional jump: set PC to op1 (label-resolved absolute PC in
  ; @module_labels). Used after the then-branch of if-then-else to
  ; skip over the else-branch.
  ; op1 itself is a label index — dereference via wam_label_pc.
  %j.label = trunc i64 %op1 to i32
  %j.target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %j.label)
  %j.valid = icmp sge i32 %j.target_pc, 0
  br i1 %j.valid, label %j.go, label %j.fail

j.go:
  call void @wam_set_pc(%WamState* %vm, i32 %j.target_pc)
  ret i1 true

j.fail:
  ret i1 false').

wam_llvm_case('get_level',
'  ; M17: snapshot the current cp_count into a Y register so cut Y_n
  ; can restore it later. Used for proper if-then-else soft cut --
  ; pre-M17 the LLVM target''s cut_ite just decremented cp_count by 1,
  ; which silently cut the wrong CP when the condition pushed inner
  ; multi-clause CPs. op1 = Y register index.
  %gl.yn = trunc i64 %op1 to i32
  %gl.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %gl.cpn = load i32, i32* %gl.cpn_ptr
  %gl.cpn64 = zext i32 %gl.cpn to i64
  %gl.v = call %Value @value_integer(i64 %gl.cpn64)
  call void @wam_trail_binding(%WamState* %vm, i32 %gl.yn)
  call void @wam_set_reg(%WamState* %vm, i32 %gl.yn, %Value %gl.v)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('cut',
'  ; M17: restore cp_count from the snapshot stored in Y_n by an
  ; earlier get_level. Cuts the if-then-else CP and any inner CPs
  ; the condition pushed. op1 = Y register index.
  %cy.yn = trunc i64 %op1 to i32
  %cy.v = call %Value @wam_get_reg(%WamState* %vm, i32 %cy.yn)
  %cy.pay = extractvalue %Value %cy.v, 1
  %cy.target = trunc i64 %cy.pay to i32
  %cy.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %cy.cur = load i32, i32* %cy.cpn_ptr
  %cy.do_cut = icmp sgt i32 %cy.cur, %cy.target
  br i1 %cy.do_cut, label %cy.go, label %cy.skip
cy.go:
  store i32 %cy.target, i32* %cy.cpn_ptr
  br label %cy.skip
cy.skip:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% ============================================================================
% PHASE 3: Helper predicates → LLVM functions
% ============================================================================

%% compile_wam_helpers_to_llvm(+Options, -LLVMCode)
%  Generates LLVM IR for WAM runtime helpers.
compile_wam_helpers_to_llvm(Options, LLVMCode) :-
    compile_backtrack_to_llvm(BacktrackCode),
    compile_unwind_trail_to_llvm(UnwindCode),
    compile_execute_builtin_to_llvm(Options, BuiltinCode),
    compile_eval_arith_to_llvm(ArithCode),
    compile_copy_term_to_llvm(CopyTermCode),
    compile_term_cmp_to_llvm(TermCmpCode),
    compile_ssp_emit_segment_to_llvm(SspCode),
    atomic_list_concat([
        BacktrackCode, '\n\n',
        UnwindCode, '\n\n',
        BuiltinCode, '\n\n',
        ArithCode, '\n\n',
        CopyTermCode, '\n\n',
        TermCmpCode, '\n\n',
        SspCode
    ], LLVMCode).

compile_backtrack_to_llvm(Code) :-
    Code = 'define i1 @backtrack(%WamState* %vm) {
entry:
  ; An uncaught throw ABORTS the machine (campaign finding no. 10): refuse
  ; to resume into live choice points. Without this check, the uncaught
  ; throw behaved like an ordinary failure -- backtrack cleared halted and
  ; re-executed over the half-unwound state (catastrophic-backtracking
  ; hangs and corrupted terms). The flag is cleared per top-level query by
  ; @wam_catch_reset, so re-satisfaction of completed queries still works.
  %unc = load i1, i1* @wam_uncaught
  br i1 %unc, label %fail, label %check_cps

check_cps:
  %cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %cpn = load i32, i32* %cpn_ptr
  %has_cp = icmp sgt i32 %cpn, 0
  br i1 %has_cp, label %check_agg, label %fail

check_agg:
  ; If the top CP is an aggregate frame, delegate to finalize_aggregate.
  %ca_top_idx = sub i32 %cpn, 1
  %ca_cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %ca_cps = load %ChoicePoint*, %ChoicePoint** %ca_cps_ptr
  %ca_top = getelementptr %ChoicePoint, %ChoicePoint* %ca_cps, i32 %ca_top_idx
  %ca_at_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ca_top, i32 0, i32 4
  %ca_at = load i32, i32* %ca_at_ptr
  %is_agg = icmp sge i32 %ca_at, 0
  br i1 %is_agg, label %do_finalize, label %check_foreign

do_finalize:
  %fin_ok = call i1 @wam_finalize_aggregate(%WamState* %vm)
  ret i1 %fin_ok

check_foreign:
  ; If the top CP is a foreign-result iterator (agg_type == -2),
  ; advance the cursor and yield the next result.
  %is_foreign = icmp eq i32 %ca_at, -2
  br i1 %is_foreign, label %do_foreign_yield, label %check_dyn

check_dyn:
  ; If the top CP is a dynamic-clause iterator (agg_type == -3), advance
  ; the store scan and yield the next matching asserted clause.
  %is_dyn = icmp eq i32 %ca_at, -3
  br i1 %is_dyn, label %do_dyn_yield, label %check_retract

check_retract:
  ; agg_type == -4: a retract/1 iterator -- advance the scan, remove and
  ; yield the next matching clause.
  %is_ret = icmp eq i32 %ca_at, -4
  br i1 %is_ret, label %do_retract_yield, label %check_barrier

check_barrier:
  ; agg_type == -8: a rule-body solve barrier. Backtracking that reaches it
  ; means the nested body goal has no (more) solutions -- stop here so the
  ; nested run_loop returns false rather than unwinding into the caller.
  %is_barrier = icmp eq i32 %ca_at, -8
  br i1 %is_barrier, label %fail, label %restore

do_retract_yield:
  %ry_ok = call i1 @wam_dyn_retract_iter_next(%WamState* %vm)
  ret i1 %ry_ok

do_dyn_yield:
  %dy_ok = call i1 @wam_dyn_iter_next(%WamState* %vm)
  ret i1 %dy_ok

do_foreign_yield:
  %fy_ok = call i1 @wam_foreign_iter_next(%WamState* %vm)
  ret i1 %fy_ok

restore:
  %top_idx = sub i32 %cpn, 1
  %cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %cps = load %ChoicePoint*, %ChoicePoint** %cps_ptr
  %top = getelementptr %ChoicePoint, %ChoicePoint* %cps, i32 %top_idx

  ; Get trail mark and unwind
  %tm_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 2
  %tm = load i32, i32* %tm_ptr
  call void @unwind_trail(%WamState* %vm, i32 %tm)

  ; Campaign finding no. 12: do NOT rewind heap_top here. The heap is
  ; MONOTONIC within a top-level call, exactly like the arena. The old
  ; rewind (restore heap_top from the CP) deallocated every cell the
  ; failed attempt created -- but compounds live in the ARENA, and their
  ; argument slots are raw pointers that the trail cannot cover: a
  ; post-CP first-binding of a pre-CP compound''s unbound slot survived
  ; the trail unwind while the heap cell it referenced was deallocated.
  ; Re-satisfaction (backtracking into a completed call''s chain CP,
  ; which is correct Prolog and how SWI recovers from divergent
  ; alternatives) then re-read the compound in READ mode and followed
  ; the dangling Ref above the rewound top -- the self-hosted
  ; compiler''s deferral walkers crashed exactly there (heap oob read
  ; at heap_top, trace-proven). Bindings ARE still undone (the trail
  ; handles that above); only the CELLS persist, unreferenced, until
  ; the per-call reclamation -- the same lifetime the arena already
  ; has. The CP still SAVES heap_top (field 11) for diagnostics, but
  ; nothing restores it.
  %saved_ht_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 11
  %saved_ht = load i32, i32* %saved_ht_ptr
  ; Restore stack_size so failed clauses cannot leave EnvFrames that
  ; confuse the caller or the next alternative.
  %saved_ss_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 13
  %saved_ss = load i32, i32* %saved_ss_ptr
  %ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3
  store i32 %saved_ss, i32* %ss_ptr

  ; Restore registers
  %dst_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %src_regs = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 1, i32 0
  %dst_raw = bitcast %Value* %dst_regs to i8*
  %src_raw = bitcast %Value* %src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst_raw, i8* %src_raw, i64 2048, i1 false)

  ; Restore PC from choice point next_pc
  %npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 0
  %npc = load i32, i32* %npc_ptr
  call void @wam_set_pc(%WamState* %vm, i32 %npc)

  ; Restore CP
  %scp_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 3
  %scp = load i32, i32* %scp_ptr
  call void @wam_set_cp(%WamState* %vm, i32 %scp)

  ; Clear halted
  call void @wam_set_halted(%WamState* %vm, i1 false)

  ret i1 true

fail:
  ret i1 false
}'.

compile_unwind_trail_to_llvm(Code) :-
    Code = 'define void @unwind_trail(%WamState* %vm, i32 %saved_mark) {
entry:
  %ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %ts = load i32, i32* %ts_ptr
  %need_unwind = icmp sgt i32 %ts, %saved_mark
  br i1 %need_unwind, label %loop, label %done

loop:
  %cur_ts = load i32, i32* %ts_ptr
  %still_more = icmp sgt i32 %cur_ts, %saved_mark
  br i1 %still_more, label %unwind_one, label %done

unwind_one:
  %new_ts = sub i32 %cur_ts, 1
  store i32 %new_ts, i32* %ts_ptr
  ; Load trail entry
  %trail_arr_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 8
  %trail_arr = load %TrailEntry*, %TrailEntry** %trail_arr_ptr
  %te = getelementptr %TrailEntry, %TrailEntry* %trail_arr, i32 %new_ts
  %reg_ptr = getelementptr %TrailEntry, %TrailEntry* %te, i32 0, i32 0
  %enc_idx = load i32, i32* %reg_ptr
  %old_val_ptr = getelementptr %TrailEntry, %TrailEntry* %te, i32 0, i32 1
  %old_val = load %Value, %Value* %old_val_ptr
  ; Dispatch on the high-bit sentinel set by wam_trail_heap_binding
  ; (0x40000000): if set, the entry encodes a heap address; otherwise
  ; it is a plain register index. Without this dispatch, heap-trail
  ; entries get treated as register indices, corrupting memory beyond
  ; the regs array.
  %heap_mask = and i32 %enc_idx, 1073741824
  %is_heap = icmp ne i32 %heap_mask, 0
  br i1 %is_heap, label %unwind_heap, label %unwind_reg

unwind_heap:
  %heap_addr = and i32 %enc_idx, 1073741823
  call void @wam_heap_set(%WamState* %vm, i32 %heap_addr, %Value %old_val)
  br label %loop

unwind_reg:
  call void @wam_set_reg(%WamState* %vm, i32 %enc_idx, %Value %old_val)
  br label %loop

done:
  ret void
}'.

compile_execute_builtin_to_llvm(Options, Code) :-
    % M113: resolve target OS for struct dirent / struct tm offsets.
    % Default mode is auto (derive from target_triple); callers can
    % override with target_os(linux|darwin|freebsd|...).
    target_os_resolved(Options, OS),
    target_dirent_d_name_offset(OS, DirentNameOff),
    % The builtin dispatch runtime lives in concern libraries under
    % templates/targets/llvm_wam/runtime/, assembled by wam_llvm_runtime_libs --
    % it used to be a 16.8k-line quoted atom here, where an apostrophe in an LLVM
    % comment closed the atom (docs/design/PLAN_TEMPLATE_REFACTOR.md). Its one
    % hole is the M113/M114 dirent-d_name lookup IR: for static OS modes a single
    % getelementptr with a literal offset; for target_os(adapt) a call to the
    % runtime probe helper plus a getelementptr using the returned offset.
    target_dirent_name_ptr_ir(OS, DirentNameOff, NamePtrIR),
    wam_llvm_runtime_ir([dirent_name_ptr=NamePtrIR], ExecuteIR),
    % Append the M114 runtime probe helper. Always emitted -- it is
    % dead code when no caller is in adapt mode, which llc / clang
    % link-time DCE strips. Keeping it unconditional means the IR
    % template stays the same shape regardless of mode.
    target_dirent_probe_helper_ir(HelperIR),
    atomic_list_concat([ExecuteIR, HelperIR], '\n\n', Code).

compile_eval_arith_to_llvm(Code) :-
    Code = '; Evaluate arithmetic expression
; Takes a Value, returns the integer payload.
; For compound ops (tag=3), extracts functor and recursively evaluates args.
; For register refs (tag=6 unbound with name starting A/X), dereferences.
define i64 @eval_arith(%WamState* %vm, %Value %expr_raw) {
entry:
  ; Follow Ref chains so put_variable-aliased operands resolve to their
  ; heap-stored value before we dispatch on tag. Without this, a Ref
  ; payload (the heap address) gets returned via the `fail` path and
  ; arithmetic computes garbage for any operand that came through a
  ; put_variable + bind site.
  %expr = call %Value @wam_deref_value(%WamState* %vm, %Value %expr_raw)
  %tag = call i32 @value_tag(%Value %expr)
  switch i32 %tag, label %fail [
    i32 1, label %return_int
    i32 2, label %return_float_as_int
    i32 3, label %compound_arith
  ]

return_int:
  %val = call i64 @value_payload(%Value %expr)
  ret i64 %val

return_float_as_int:
  %fbits = call i64 @value_payload(%Value %expr)
  %fval = bitcast i64 %fbits to double
  %ival = fptosi double %fval to i64
  ret i64 %ival

compound_arith:
  ; Compound: payload is pointer to %Compound (functor, arity, args)
  %cp_bits = call i64 @value_payload(%Value %expr)
  %cp_ptr = inttoptr i64 %cp_bits to %Compound*
  ; Load arity
  %arity_ptr = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 1
  %arity = load i32, i32* %arity_ptr
  ; Load args array pointer
  %args_ptr = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 2
  %args = load %Value*, %Value** %args_ptr
  ; Load functor pointer for comparison
  %fn_ptr_ptr = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 0
  %fn_ptr = load i8*, i8** %fn_ptr_ptr
  ; Binary: evaluate both args
  %is_binary = icmp eq i32 %arity, 2
  br i1 %is_binary, label %eval_binary, label %check_unary

eval_binary:
  %arg0_ptr = getelementptr %Value, %Value* %args, i32 0
  %arg0 = load %Value, %Value* %arg0_ptr
  %arg1_ptr = getelementptr %Value, %Value* %args, i32 1
  %arg1 = load %Value, %Value* %arg1_ptr
  %a = call i64 @eval_arith(%WamState* %vm, %Value %arg0)
  %b = call i64 @eval_arith(%WamState* %vm, %Value %arg1)
  ; Dispatch on functor: check first char for +, -, *, /
  %fn_first = load i8, i8* %fn_ptr
  switch i8 %fn_first, label %check_named_binary [
    i8 43, label %do_add     ; \'+\'
    i8 45, label %do_sub     ; \'-\'
    i8 42, label %do_mul     ; \'*\'
    i8 47, label %do_div     ; \'/\'
  ]

do_add:
  %add_r = add i64 %a, %b
  ret i64 %add_r

do_sub:
  %sub_r = sub i64 %a, %b
  ret i64 %sub_r

do_mul:
  ; M10: disambiguate `**` (power) from `*` (multiply) by inspecting
  ; the second byte of the functor name. `*` has fn[1] = \\0;
  ; `**` has fn[1] = \'*\'. Anything else falls through to plain `*`.
  %mul_fn_second_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %mul_fn_second = load i8, i8* %mul_fn_second_ptr
  %mul_is_pow = icmp eq i8 %mul_fn_second, 42  ; \'*\'
  br i1 %mul_is_pow, label %do_pow, label %do_mul_plain

do_mul_plain:
  %mul_r = mul i64 %a, %b
  ret i64 %mul_r

do_pow:
  ; Integer power loop: base ** exp computed as repeated multiply.
  ; Negative exponent returns 0 (eval_arith is integer-only; proper
  ; float pow is M11 deferred work that retags eval_arith to carry
  ; either i64 or double).
  %p_neg = icmp slt i64 %b, 0
  br i1 %p_neg, label %pow_neg, label %pow_loop_setup

pow_neg:
  ret i64 0

pow_loop_setup:
  br label %pow_loop

pow_loop:
  %p_acc = phi i64 [ 1, %pow_loop_setup ], [ %p_new_acc, %pow_step ]
  %p_n = phi i64 [ %b, %pow_loop_setup ], [ %p_dec, %pow_step ]
  %p_done = icmp sle i64 %p_n, 0
  br i1 %p_done, label %pow_done, label %pow_step

pow_step:
  %p_new_acc = mul i64 %p_acc, %a
  %p_dec = sub i64 %p_n, 1
  br label %pow_loop

pow_done:
  ret i64 %p_acc

do_div:
  %div_zero = icmp eq i64 %b, 0
  br i1 %div_zero, label %fail, label %do_div_ok

do_div_ok:
  %div_r = sdiv i64 %a, %b
  ret i64 %div_r

check_named_binary:
  ; Check for named binary ops: mod, max, min (match on first char m).
  %nb_first = load i8, i8* %fn_ptr
  %nb_is_m = icmp eq i8 %nb_first, 109  ; \'m\'
  br i1 %nb_is_m, label %nb_check_second, label %fail

nb_check_second:
  %nb_second_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %nb_second = load i8, i8* %nb_second_ptr
  switch i8 %nb_second, label %fail [
    i8 111, label %do_mod   ; \'o\' → mod
    i8 97, label %do_max    ; \'a\' → max
    i8 105, label %do_min   ; \'i\' → min
  ]

do_mod:
  %mod_zero = icmp eq i64 %b, 0
  br i1 %mod_zero, label %fail, label %do_mod_ok
do_mod_ok:
  %mod_r = srem i64 %a, %b
  ret i64 %mod_r

do_max:
  %max_cmp = icmp sgt i64 %a, %b
  %max_r = select i1 %max_cmp, i64 %a, i64 %b
  ret i64 %max_r

do_min:
  %min_cmp = icmp slt i64 %a, %b
  %min_r = select i1 %min_cmp, i64 %a, i64 %b
  ret i64 %min_r

check_unary:
  %is_unary = icmp eq i32 %arity, 1
  br i1 %is_unary, label %eval_unary, label %fail

eval_unary:
  %u_arg_ptr = getelementptr %Value, %Value* %args, i32 0
  %u_arg = load %Value, %Value* %u_arg_ptr
  %u_val = call i64 @eval_arith(%WamState* %vm, %Value %u_arg)
  %u_fn_first = load i8, i8* %fn_ptr
  switch i8 %u_fn_first, label %fail [
    i8 45, label %do_neg     ; \'-\' → negation
    i8 97, label %do_abs     ; \'a\' → abs
  ]

do_neg:
  %neg_r = sub i64 0, %u_val
  ret i64 %neg_r

do_abs:
  %abs_neg = icmp slt i64 %u_val, 0
  %abs_pos = sub i64 0, %u_val
  %abs_r = select i1 %abs_neg, i64 %abs_pos, i64 %u_val
  ret i64 %abs_r

fail:
  %f.fmt = getelementptr [26 x i8], [26 x i8]* @.fmt_arith_fail, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %f.fmt)
  ret i64 0
}

; M13: Float-aware tagged arithmetic evaluator. Returns a %Value
; (Integer tag=1 or Float tag=2). Recurses into itself for compound
; args so int/float propagates through nested expressions. Promotion
; rule: any Float operand floats the whole binary op; pure Integer
; chains stay Integer. Division (/) follows Prolog convention --
; Integer / Integer that divides exactly stays Integer; otherwise
; the result is Float. The legacy integer-only @eval_arith is kept
; for paths that still need an i64 directly (currently none in the
; merged tree).
@wam_arith_err = internal global i1 0

; Capstone finding (part 2): the arith dispatch below routes on the
; FIRST BYTE of the functor name (plus second/third-byte splits), so
; an UNKNOWN functor sharing a prefix with a real op silently aliased
; it -- f(2) evaluated as floor(2), g(5,6) as gcd(5,6). SWI raises a
; type error for these; parity requires they FAIL. This gate strcmps
; the full functor name against the exact set of implemented op names
; per arity BEFORE dispatch; unknown names go to ev_zero, which sets
; @wam_arith_err so is/2 and the arithmetic comparisons fail cleanly.
; Names are emitted as raw byte arrays to sidestep string escaping.
@.aop_b0 = private constant [2 x i8] [i8 43, i8 0]
@.aop_b1 = private constant [2 x i8] [i8 45, i8 0]
@.aop_b2 = private constant [2 x i8] [i8 42, i8 0]
@.aop_b3 = private constant [3 x i8] [i8 42, i8 42, i8 0]
@.aop_b4 = private constant [2 x i8] [i8 47, i8 0]
@.aop_b5 = private constant [3 x i8] [i8 47, i8 47, i8 0]
@.aop_b6 = private constant [3 x i8] [i8 47, i8 92, i8 0]
@.aop_b7 = private constant [3 x i8] [i8 60, i8 60, i8 0]
@.aop_b8 = private constant [3 x i8] [i8 62, i8 62, i8 0]
@.aop_b9 = private constant [3 x i8] [i8 92, i8 47, i8 0]
@.aop_b10 = private constant [4 x i8] [i8 109, i8 111, i8 100, i8 0]
@.aop_b11 = private constant [4 x i8] [i8 109, i8 97, i8 120, i8 0]
@.aop_b12 = private constant [4 x i8] [i8 109, i8 105, i8 110, i8 0]
@.aop_b13 = private constant [6 x i8] [i8 97, i8 116, i8 97, i8 110, i8 50, i8 0]
@.aop_b14 = private constant [4 x i8] [i8 103, i8 99, i8 100, i8 0]
@.aop_b15 = private constant [4 x i8] [i8 108, i8 111, i8 103, i8 0]
@.aop_b16 = private constant [4 x i8] [i8 120, i8 111, i8 114, i8 0]
@.aop_u0 = private constant [2 x i8] [i8 45, i8 0]
@.aop_u1 = private constant [2 x i8] [i8 92, i8 0]
@.aop_u2 = private constant [4 x i8] [i8 97, i8 98, i8 115, i8 0]
@.aop_u3 = private constant [5 x i8] [i8 97, i8 115, i8 105, i8 110, i8 0]
@.aop_u4 = private constant [5 x i8] [i8 97, i8 99, i8 111, i8 115, i8 0]
@.aop_u5 = private constant [5 x i8] [i8 97, i8 116, i8 97, i8 110, i8 0]
@.aop_u6 = private constant [6 x i8] [i8 114, i8 111, i8 117, i8 110, i8 100, i8 0]
@.aop_u7 = private constant [6 x i8] [i8 102, i8 108, i8 111, i8 111, i8 114, i8 0]
@.aop_u8 = private constant [8 x i8] [i8 99, i8 101, i8 105, i8 108, i8 105, i8 110, i8 103, i8 0]
@.aop_u9 = private constant [4 x i8] [i8 99, i8 111, i8 115, i8 0]
@.aop_u10 = private constant [5 x i8] [i8 115, i8 113, i8 114, i8 116, i8 0]
@.aop_u11 = private constant [5 x i8] [i8 115, i8 105, i8 103, i8 110, i8 0]
@.aop_u12 = private constant [4 x i8] [i8 115, i8 105, i8 110, i8 0]
@.aop_u13 = private constant [9 x i8] [i8 116, i8 114, i8 117, i8 110, i8 99, i8 97, i8 116, i8 101, i8 0]
@.aop_u14 = private constant [4 x i8] [i8 116, i8 97, i8 110, i8 0]
@.aop_u15 = private constant [4 x i8] [i8 108, i8 111, i8 103, i8 0]
@.aop_u16 = private constant [4 x i8] [i8 101, i8 120, i8 112, i8 0]

define i1 @wam_arith_name_ok(i8* %fn, i32 %arity) {
entry:
  %is2 = icmp eq i32 %arity, 2
  br i1 %is2, label %b0, label %chk1
chk1:
  %is1 = icmp eq i32 %arity, 1
  br i1 %is1, label %u0, label %no
b0:
  %b0.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.aop_b0, i32 0, i32 0))
  %b0.z = icmp eq i32 %b0.c, 0
  br i1 %b0.z, label %yes, label %b1
b1:
  %b1.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.aop_b1, i32 0, i32 0))
  %b1.z = icmp eq i32 %b1.c, 0
  br i1 %b1.z, label %yes, label %b2
b2:
  %b2.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.aop_b2, i32 0, i32 0))
  %b2.z = icmp eq i32 %b2.c, 0
  br i1 %b2.z, label %yes, label %b3
b3:
  %b3.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.aop_b3, i32 0, i32 0))
  %b3.z = icmp eq i32 %b3.c, 0
  br i1 %b3.z, label %yes, label %b4
b4:
  %b4.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.aop_b4, i32 0, i32 0))
  %b4.z = icmp eq i32 %b4.c, 0
  br i1 %b4.z, label %yes, label %b5
b5:
  %b5.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.aop_b5, i32 0, i32 0))
  %b5.z = icmp eq i32 %b5.c, 0
  br i1 %b5.z, label %yes, label %b6
b6:
  %b6.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.aop_b6, i32 0, i32 0))
  %b6.z = icmp eq i32 %b6.c, 0
  br i1 %b6.z, label %yes, label %b7
b7:
  %b7.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.aop_b7, i32 0, i32 0))
  %b7.z = icmp eq i32 %b7.c, 0
  br i1 %b7.z, label %yes, label %b8
b8:
  %b8.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.aop_b8, i32 0, i32 0))
  %b8.z = icmp eq i32 %b8.c, 0
  br i1 %b8.z, label %yes, label %b9
b9:
  %b9.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.aop_b9, i32 0, i32 0))
  %b9.z = icmp eq i32 %b9.c, 0
  br i1 %b9.z, label %yes, label %b10
b10:
  %b10.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_b10, i32 0, i32 0))
  %b10.z = icmp eq i32 %b10.c, 0
  br i1 %b10.z, label %yes, label %b11
b11:
  %b11.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_b11, i32 0, i32 0))
  %b11.z = icmp eq i32 %b11.c, 0
  br i1 %b11.z, label %yes, label %b12
b12:
  %b12.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_b12, i32 0, i32 0))
  %b12.z = icmp eq i32 %b12.c, 0
  br i1 %b12.z, label %yes, label %b13
b13:
  %b13.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.aop_b13, i32 0, i32 0))
  %b13.z = icmp eq i32 %b13.c, 0
  br i1 %b13.z, label %yes, label %b14
b14:
  %b14.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_b14, i32 0, i32 0))
  %b14.z = icmp eq i32 %b14.c, 0
  br i1 %b14.z, label %yes, label %b15
b15:
  %b15.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_b15, i32 0, i32 0))
  %b15.z = icmp eq i32 %b15.c, 0
  br i1 %b15.z, label %yes, label %b16
b16:
  %b16.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_b16, i32 0, i32 0))
  %b16.z = icmp eq i32 %b16.c, 0
  br i1 %b16.z, label %yes, label %no
u0:
  %u0.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.aop_u0, i32 0, i32 0))
  %u0.z = icmp eq i32 %u0.c, 0
  br i1 %u0.z, label %yes, label %u1
u1:
  %u1.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.aop_u1, i32 0, i32 0))
  %u1.z = icmp eq i32 %u1.c, 0
  br i1 %u1.z, label %yes, label %u2
u2:
  %u2.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_u2, i32 0, i32 0))
  %u2.z = icmp eq i32 %u2.c, 0
  br i1 %u2.z, label %yes, label %u3
u3:
  %u3.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.aop_u3, i32 0, i32 0))
  %u3.z = icmp eq i32 %u3.c, 0
  br i1 %u3.z, label %yes, label %u4
u4:
  %u4.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.aop_u4, i32 0, i32 0))
  %u4.z = icmp eq i32 %u4.c, 0
  br i1 %u4.z, label %yes, label %u5
u5:
  %u5.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.aop_u5, i32 0, i32 0))
  %u5.z = icmp eq i32 %u5.c, 0
  br i1 %u5.z, label %yes, label %u6
u6:
  %u6.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.aop_u6, i32 0, i32 0))
  %u6.z = icmp eq i32 %u6.c, 0
  br i1 %u6.z, label %yes, label %u7
u7:
  %u7.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.aop_u7, i32 0, i32 0))
  %u7.z = icmp eq i32 %u7.c, 0
  br i1 %u7.z, label %yes, label %u8
u8:
  %u8.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.aop_u8, i32 0, i32 0))
  %u8.z = icmp eq i32 %u8.c, 0
  br i1 %u8.z, label %yes, label %u9
u9:
  %u9.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_u9, i32 0, i32 0))
  %u9.z = icmp eq i32 %u9.c, 0
  br i1 %u9.z, label %yes, label %u10
u10:
  %u10.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.aop_u10, i32 0, i32 0))
  %u10.z = icmp eq i32 %u10.c, 0
  br i1 %u10.z, label %yes, label %u11
u11:
  %u11.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.aop_u11, i32 0, i32 0))
  %u11.z = icmp eq i32 %u11.c, 0
  br i1 %u11.z, label %yes, label %u12
u12:
  %u12.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_u12, i32 0, i32 0))
  %u12.z = icmp eq i32 %u12.c, 0
  br i1 %u12.z, label %yes, label %u13
u13:
  %u13.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.aop_u13, i32 0, i32 0))
  %u13.z = icmp eq i32 %u13.c, 0
  br i1 %u13.z, label %yes, label %u14
u14:
  %u14.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_u14, i32 0, i32 0))
  %u14.z = icmp eq i32 %u14.c, 0
  br i1 %u14.z, label %yes, label %u15
u15:
  %u15.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_u15, i32 0, i32 0))
  %u15.z = icmp eq i32 %u15.c, 0
  br i1 %u15.z, label %yes, label %u16
u16:
  %u16.c = call i32 @strcmp(i8* %fn, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.aop_u16, i32 0, i32 0))
  %u16.z = icmp eq i32 %u16.c, 0
  br i1 %u16.z, label %yes, label %no
yes:
  ret i1 1
no:
  ret i1 0
}

define %Value @eval_arith_value(%WamState* %vm, %Value %expr_raw) {
entry:
  %expr = call %Value @wam_deref_value(%WamState* %vm, %Value %expr_raw)
  %tag = extractvalue %Value %expr, 0
  switch i32 %tag, label %ev_zero [
    i32 0, label %ev_atom_const
    i32 1, label %ev_atomic_in
    i32 2, label %ev_atomic_in
    i32 3, label %ev_compound
  ]

ev_atomic_in:
  ret %Value %expr

ev_atom_const:
  ; M83: arithmetic atom constants. Currently pi and e; failable
  ; otherwise. Resolve the atom''s name to a C string, then strcmp
  ; against the registered functor strings.
  %atc.aid = extractvalue %Value %expr, 1
  %atc.str = call i8* @wam_atom_to_string(i64 %atc.aid)
  %atc.null = icmp eq i8* %atc.str, null
  br i1 %atc.null, label %ev_zero, label %atc.try_pi
atc.try_pi:
  %atc.fn_pi = getelementptr [3 x i8], [3 x i8]* @.fn_pi, i32 0, i32 0
  %atc.cmp_pi = call i32 @strcmp(i8* %atc.str, i8* %atc.fn_pi)
  %atc.is_pi = icmp eq i32 %atc.cmp_pi, 0
  br i1 %atc.is_pi, label %atc.ret_pi, label %atc.try_e
atc.ret_pi:
  ; pi as IEEE 754 double = 0x400921FB54442D18 (3.141592653589793).
  %atc.pi_v = call %Value @value_float(double 0x400921FB54442D18)
  ret %Value %atc.pi_v
atc.try_e:
  %atc.fn_e = getelementptr [2 x i8], [2 x i8]* @.fn_e, i32 0, i32 0
  %atc.cmp_e = call i32 @strcmp(i8* %atc.str, i8* %atc.fn_e)
  %atc.is_e = icmp eq i32 %atc.cmp_e, 0
  br i1 %atc.is_e, label %atc.ret_e, label %ev_zero
atc.ret_e:
  ; e as IEEE 754 double = 0x4005BF0A8B145769 (2.718281828459045).
  %atc.e_v = call %Value @value_float(double 0x4005BF0A8B145769)
  ret %Value %atc.e_v

ev_compound:
  %cp_bits = extractvalue %Value %expr, 1
  %cp_ptr = inttoptr i64 %cp_bits to %Compound*
  %ar_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 1
  %arity = load i32, i32* %ar_slot
  %fn_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 0
  %fn_ptr = load i8*, i8** %fn_slot
  %fn0 = load i8, i8* %fn_ptr
  %args_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 2
  %args = load %Value*, %Value** %args_slot
  %aop_ok = call i1 @wam_arith_name_ok(i8* %fn_ptr, i32 %arity)
  br i1 %aop_ok, label %ev_known_op, label %ev_zero

ev_known_op:
  %is_binary = icmp eq i32 %arity, 2
  br i1 %is_binary, label %ev_eval_binary, label %ev_check_unary

ev_eval_binary:
  %a_ptr = getelementptr %Value, %Value* %args, i32 0
  %a_raw = load %Value, %Value* %a_ptr
  %a = call %Value @eval_arith_value(%WamState* %vm, %Value %a_raw)
  %b_ptr = getelementptr %Value, %Value* %args, i32 1
  %b_raw = load %Value, %Value* %b_ptr
  %b = call %Value @eval_arith_value(%WamState* %vm, %Value %b_raw)
  %a_tag = extractvalue %Value %a, 0
  %b_tag = extractvalue %Value %b, 0
  %a_is_float = icmp eq i32 %a_tag, 2
  %b_is_float = icmp eq i32 %b_tag, 2
  %either_float = or i1 %a_is_float, %b_is_float
  switch i8 %fn0, label %ev_check_named_binary [
    i8 43, label %ev_add
    i8 45, label %ev_sub
    i8 42, label %ev_mul_or_pow
    i8 47, label %ev_div
    i8 60, label %ev_shl_check    ; ''<'' -> << (M84)
    i8 62, label %ev_shr_check    ; ''>'' -> >> (M84)
    i8 92, label %ev_bor_check    ; ''\\'' -> \\/ (M85)
  ]

ev_add:
  br i1 %either_float, label %ev_add_f, label %ev_add_i
ev_add_i:
  %add_a_i = extractvalue %Value %a, 1
  %add_b_i = extractvalue %Value %b, 1
  %add_r_i = add i64 %add_a_i, %add_b_i
  %add_v_i = call %Value @value_integer(i64 %add_r_i)
  ret %Value %add_v_i
ev_add_f:
  %add_a_d = call double @value_to_double(%Value %a)
  %add_b_d = call double @value_to_double(%Value %b)
  %add_r_d = fadd double %add_a_d, %add_b_d
  %add_v_f = call %Value @value_float(double %add_r_d)
  ret %Value %add_v_f

ev_sub:
  br i1 %either_float, label %ev_sub_f, label %ev_sub_i
ev_sub_i:
  %sub_a_i = extractvalue %Value %a, 1
  %sub_b_i = extractvalue %Value %b, 1
  %sub_r_i = sub i64 %sub_a_i, %sub_b_i
  %sub_v_i = call %Value @value_integer(i64 %sub_r_i)
  ret %Value %sub_v_i
ev_sub_f:
  %sub_a_d = call double @value_to_double(%Value %a)
  %sub_b_d = call double @value_to_double(%Value %b)
  %sub_r_d = fsub double %sub_a_d, %sub_b_d
  %sub_v_f = call %Value @value_float(double %sub_r_d)
  ret %Value %sub_v_f

ev_mul_or_pow:
  ; Disambiguate `*` vs `**` by inspecting fn[1].
  %mul_fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %mul_fn1 = load i8, i8* %mul_fn1_ptr
  %mul_is_pow = icmp eq i8 %mul_fn1, 42
  br i1 %mul_is_pow, label %ev_pow_branch, label %ev_mul_branch

ev_mul_branch:
  br i1 %either_float, label %ev_mul_f, label %ev_mul_i
ev_mul_i:
  %mul_a_i = extractvalue %Value %a, 1
  %mul_b_i = extractvalue %Value %b, 1
  %mul_r_i = mul i64 %mul_a_i, %mul_b_i
  %mul_v_i = call %Value @value_integer(i64 %mul_r_i)
  ret %Value %mul_v_i
ev_mul_f:
  %mul_a_d = call double @value_to_double(%Value %a)
  %mul_b_d = call double @value_to_double(%Value %b)
  %mul_r_d = fmul double %mul_a_d, %mul_b_d
  %mul_v_f = call %Value @value_float(double %mul_r_d)
  ret %Value %mul_v_f

ev_pow_branch:
  ; ** rules:
  ;   int base, int exp >= 0       -> int loop
  ;   any float operand, or exp<0  -> float (via the negative-exp
  ;                                    reciprocal path or fpow loop)
  br i1 %either_float, label %ev_pow_float, label %ev_pow_int_check
ev_pow_int_check:
  %pow_b_i = extractvalue %Value %b, 1
  %pow_neg = icmp slt i64 %pow_b_i, 0
  br i1 %pow_neg, label %ev_pow_float, label %ev_pow_int
ev_pow_int:
  %pow_a_i = extractvalue %Value %a, 1
  br label %ev_pow_iloop
ev_pow_iloop:
  %pow_acc_i = phi i64 [ 1, %ev_pow_int ], [ %pow_new_i, %ev_pow_istep ]
  %pow_n_i = phi i64 [ %pow_b_i, %ev_pow_int ], [ %pow_dec_i, %ev_pow_istep ]
  %pow_done_i = icmp sle i64 %pow_n_i, 0
  br i1 %pow_done_i, label %ev_pow_idone, label %ev_pow_istep
ev_pow_istep:
  %pow_new_i = mul i64 %pow_acc_i, %pow_a_i
  %pow_dec_i = sub i64 %pow_n_i, 1
  br label %ev_pow_iloop
ev_pow_idone:
  %pow_v_i = call %Value @value_integer(i64 %pow_acc_i)
  ret %Value %pow_v_i

ev_pow_float:
  ; base ** exp via fpow loop. For int exp we loop |exp| times; if
  ; exp is float we punt to a simple loop using its truncated integer
  ; absolute value (good enough for the common int-exp-with-float-
  ; base case; full IEEE pow is a separate libm dependency).
  %pow_base_d = call double @value_to_double(%Value %a)
  %pow_exp_d = call double @value_to_double(%Value %b)
  %pow_exp_neg = fcmp olt double %pow_exp_d, 0.0
  ; Take the absolute exponent as an integer iteration count.
  %pow_abs_exp_d = call double @llvm.fabs.f64(double %pow_exp_d)
  %pow_abs_exp = fptosi double %pow_abs_exp_d to i64
  br label %ev_pow_floop
ev_pow_floop:
  %pow_acc_d = phi double [ 1.0, %ev_pow_float ], [ %pow_new_d, %ev_pow_fstep ]
  %pow_n_d = phi i64 [ %pow_abs_exp, %ev_pow_float ], [ %pow_dec_d, %ev_pow_fstep ]
  %pow_done_d = icmp sle i64 %pow_n_d, 0
  br i1 %pow_done_d, label %ev_pow_fdone, label %ev_pow_fstep
ev_pow_fstep:
  %pow_new_d = fmul double %pow_acc_d, %pow_base_d
  %pow_dec_d = sub i64 %pow_n_d, 1
  br label %ev_pow_floop
ev_pow_fdone:
  br i1 %pow_exp_neg, label %ev_pow_neg_finish, label %ev_pow_pos_finish
ev_pow_neg_finish:
  %pow_acc_zero = fcmp oeq double %pow_acc_d, 0.0
  br i1 %pow_acc_zero, label %ev_pow_zero, label %ev_pow_recip
ev_pow_zero:
  %pow_zv = call %Value @value_float(double 0.0)
  ret %Value %pow_zv
ev_pow_recip:
  %pow_recip_d = fdiv double 1.0, %pow_acc_d
  %pow_recip_v = call %Value @value_float(double %pow_recip_d)
  ret %Value %pow_recip_v
ev_pow_pos_finish:
  %pow_pos_v = call %Value @value_float(double %pow_acc_d)
  ret %Value %pow_pos_v

ev_div:
  ; M85: second-byte check disambiguates ``/'' (division) from
  ; ``/\\'' (bitwise AND, fn_ptr[1] = ''\\'' = 92) and ``//''
  ; (truncating integer division, fn_ptr[1] = ''/'' = 47 -- it used to
  ; silently alias float-capable /, so 7 // 2 evaluated to 3.5 where
  ; SWI yields 3).
  %div.fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %div.fn1 = load i8, i8* %div.fn1_ptr
  %div.is_band = icmp eq i8 %div.fn1, 92
  br i1 %div.is_band, label %ev_band, label %ev_div_chk_int
ev_div_chk_int:
  %div.is_intdiv = icmp eq i8 %div.fn1, 47
  br i1 %div.is_intdiv, label %ev_intdiv, label %ev_div_normal
ev_intdiv:
  ; // -- truncating integer division (LLVM sdiv truncates toward
  ; zero, matching SWI). Integer operands only: a float operand is an
  ; evaluation error, like SWI''s type check. Division by zero fails
  ; through the arith-error channel.
  br i1 %either_float, label %ev_zero, label %ev_intdiv_chk0
ev_intdiv_chk0:
  %idv.a_i = extractvalue %Value %a, 1
  %idv.b_i = extractvalue %Value %b, 1
  %idv.b_zero = icmp eq i64 %idv.b_i, 0
  br i1 %idv.b_zero, label %ev_zero, label %ev_intdiv_go
ev_intdiv_go:
  %idv.r = sdiv i64 %idv.a_i, %idv.b_i
  %idv.v = call %Value @value_integer(i64 %idv.r)
  ret %Value %idv.v
ev_band:
  ; M85: /\\ -- integer bitwise AND.
  %band.a_i = extractvalue %Value %a, 1
  %band.b_i = extractvalue %Value %b, 1
  %band.r = and i64 %band.a_i, %band.b_i
  %band.v = call %Value @value_integer(i64 %band.r)
  ret %Value %band.v
ev_div_normal:
  ; Prolog ``/``: int/int that divides exactly stays int; otherwise
  ; the result is float.
  br i1 %either_float, label %ev_div_f, label %ev_div_i_check
ev_div_i_check:
  %div_a_i = extractvalue %Value %a, 1
  %div_b_i = extractvalue %Value %b, 1
  %div_zero_i = icmp eq i64 %div_b_i, 0
  br i1 %div_zero_i, label %ev_zero, label %ev_div_i_test
ev_div_i_test:
  %div_rem = srem i64 %div_a_i, %div_b_i
  %div_exact = icmp eq i64 %div_rem, 0
  br i1 %div_exact, label %ev_div_i_exact, label %ev_div_f
ev_div_i_exact:
  %div_r_i = sdiv i64 %div_a_i, %div_b_i
  %div_v_i = call %Value @value_integer(i64 %div_r_i)
  ret %Value %div_v_i
ev_div_f:
  %div_a_d = call double @value_to_double(%Value %a)
  %div_b_d = call double @value_to_double(%Value %b)
  %div_b_zero = fcmp oeq double %div_b_d, 0.0
  br i1 %div_b_zero, label %ev_zero, label %ev_div_f_go
ev_div_f_go:
  %div_r_d = fdiv double %div_a_d, %div_b_d
  %div_v_f = call %Value @value_float(double %div_r_d)
  ret %Value %div_v_f

ev_shl_check:
  ; M84: << is the only ``<''-prefix binary arith op. Verify the
  ; second byte is also ``<'' (60) so a stray ``<=''/``<+''/etc.
  ; doesn''t fall through to ev_shl.
  %shl.fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %shl.fn1 = load i8, i8* %shl.fn1_ptr
  %shl.is_lt = icmp eq i8 %shl.fn1, 60
  br i1 %shl.is_lt, label %ev_shl, label %ev_zero
ev_shl:
  ; Integer-only logical left shift on i64 payloads.
  %shl.a_i = extractvalue %Value %a, 1
  %shl.b_i = extractvalue %Value %b, 1
  %shl.r = shl i64 %shl.a_i, %shl.b_i
  %shl.v = call %Value @value_integer(i64 %shl.r)
  ret %Value %shl.v

ev_shr_check:
  ; M84: >> via second-byte verify, mirroring ev_shl_check.
  %shr.fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %shr.fn1 = load i8, i8* %shr.fn1_ptr
  %shr.is_gt = icmp eq i8 %shr.fn1, 62
  br i1 %shr.is_gt, label %ev_shr, label %ev_zero
ev_shr:
  ; Integer arithmetic right shift (sign-preserving). Prolog
  ; integers are signed, so ashr matches the host semantics.
  %shr.a_i = extractvalue %Value %a, 1
  %shr.b_i = extractvalue %Value %b, 1
  %shr.r = ashr i64 %shr.a_i, %shr.b_i
  %shr.v = call %Value @value_integer(i64 %shr.r)
  ret %Value %shr.v

ev_bor_check:
  ; M85: ``\\'' (92) is the first byte of binary ``\\/'' (bitwise OR).
  ; Verify the second byte is ``/'' (47); fail otherwise (the only
  ; other ``\\''-prefix binary in standard Prolog arithmetic would
  ; be ``\\='' or ``\\=='' which are unification/equality, not
  ; arith).
  %bor.fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %bor.fn1 = load i8, i8* %bor.fn1_ptr
  %bor.is_slash = icmp eq i8 %bor.fn1, 47
  br i1 %bor.is_slash, label %ev_bor, label %ev_zero
ev_bor:
  %bor.a_i = extractvalue %Value %a, 1
  %bor.b_i = extractvalue %Value %b, 1
  %bor.r = or i64 %bor.a_i, %bor.b_i
  %bor.v = call %Value @value_integer(i64 %bor.r)
  ret %Value %bor.v

ev_check_named_binary:
  ; mod / max / min -- functor name starts with ``m``.
  ; atan2 -- functor name starts with ``a`` (M81).
  ; gcd -- functor name starts with ``g`` (M82).
  ; log/2 -- functor name starts with ``l`` (M82).
  ; xor -- functor name starts with ``x`` (M83, integer bitwise).
  switch i8 %fn0, label %ev_zero [
    i8 109, label %ev_nb_second   ; ''m'' -> mod | max | min
    i8 97,  label %ev_nb_a_second ; ''a'' -> atan2
    i8 103, label %ev_gcd         ; ''g'' -> gcd
    i8 108, label %ev_log2        ; ''l'' -> log/2
    i8 120, label %ev_xor         ; ''x'' -> xor
  ]
ev_nb_a_second:
  ; M81: only atan2 starts with ``a'' in the binary path. Verify the
  ; second byte is ``t'' so a stray ``a''-prefixed binary name
  ; doesn''t pretend to be atan2.
  %nba.fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %nba.fn1 = load i8, i8* %nba.fn1_ptr
  %nba.is_t = icmp eq i8 %nba.fn1, 116
  br i1 %nba.is_t, label %ev_atan2, label %ev_zero
ev_atan2:
  %a2.a_d = call double @value_to_double(%Value %a)
  %a2.b_d = call double @value_to_double(%Value %b)
  %a2.r = call double @atan2(double %a2.a_d, double %a2.b_d)
  %a2.v = call %Value @value_float(double %a2.r)
  ret %Value %a2.v

ev_gcd:
  ; M82: gcd(A, B) -- integer-only. Iterative Euclid on i64 with
  ; abs() of both args so negative inputs still produce a positive
  ; gcd (matching SWI semantics). gcd(0, 0) is defined as 0.
  %gcd.a_i = extractvalue %Value %a, 1
  %gcd.b_i = extractvalue %Value %b, 1
  %gcd.a_neg = icmp slt i64 %gcd.a_i, 0
  %gcd.a_op = sub i64 0, %gcd.a_i
  %gcd.a_abs = select i1 %gcd.a_neg, i64 %gcd.a_op, i64 %gcd.a_i
  %gcd.b_neg = icmp slt i64 %gcd.b_i, 0
  %gcd.b_op = sub i64 0, %gcd.b_i
  %gcd.b_abs = select i1 %gcd.b_neg, i64 %gcd.b_op, i64 %gcd.b_i
  br label %gcd.loop
gcd.loop:
  %gcd.x = phi i64 [ %gcd.a_abs, %ev_gcd ], [ %gcd.y, %gcd.step ]
  %gcd.y = phi i64 [ %gcd.b_abs, %ev_gcd ], [ %gcd.r, %gcd.step ]
  %gcd.y_zero = icmp eq i64 %gcd.y, 0
  br i1 %gcd.y_zero, label %gcd.done, label %gcd.step
gcd.step:
  %gcd.r = srem i64 %gcd.x, %gcd.y
  br label %gcd.loop
gcd.done:
  %gcd.v = call %Value @value_integer(i64 %gcd.x)
  ret %Value %gcd.v

ev_log2:
  ; M82: log(Base, X) -- change-of-base via log(X)/log(Base).
  ; Always Float; matches the existing log/1 (natural log) when
  ; Base == e.
  %lg2.base_d = call double @value_to_double(%Value %a)
  %lg2.x_d = call double @value_to_double(%Value %b)
  %lg2.lnb = call double @llvm.log.f64(double %lg2.base_d)
  %lg2.lnx = call double @llvm.log.f64(double %lg2.x_d)
  %lg2.r = fdiv double %lg2.lnx, %lg2.lnb
  %lg2.v = call %Value @value_float(double %lg2.r)
  ret %Value %lg2.v

ev_xor:
  ; M83: xor(A, B) -- integer-only bitwise XOR. Reads payloads as
  ; i64 directly; passing a Float silently XORs its bit pattern,
  ; consistent with the gcd treatment.
  %xor.a_i = extractvalue %Value %a, 1
  %xor.b_i = extractvalue %Value %b, 1
  %xor.r = xor i64 %xor.a_i, %xor.b_i
  %xor.v = call %Value @value_integer(i64 %xor.r)
  ret %Value %xor.v

ev_nb_second:
  %nb_fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %nb_fn1 = load i8, i8* %nb_fn1_ptr
  switch i8 %nb_fn1, label %ev_zero [
    i8 111, label %ev_mod
    i8 97,  label %ev_max
    i8 105, label %ev_min
  ]
ev_mod:
  ; Integer-only: truncate any float operands.
  %mod_a_i = extractvalue %Value %a, 1
  %mod_b_i = extractvalue %Value %b, 1
  %mod_b_zero = icmp eq i64 %mod_b_i, 0
  br i1 %mod_b_zero, label %ev_zero, label %ev_mod_go
ev_mod_go:
  %mod_r_i = srem i64 %mod_a_i, %mod_b_i
  %mod_v = call %Value @value_integer(i64 %mod_r_i)
  ret %Value %mod_v
ev_max:
  br i1 %either_float, label %ev_max_f, label %ev_max_i
ev_max_i:
  %max_a_i = extractvalue %Value %a, 1
  %max_b_i = extractvalue %Value %b, 1
  %max_gt_i = icmp sgt i64 %max_a_i, %max_b_i
  %max_r_i = select i1 %max_gt_i, i64 %max_a_i, i64 %max_b_i
  %max_v_i = call %Value @value_integer(i64 %max_r_i)
  ret %Value %max_v_i
ev_max_f:
  %max_a_d = call double @value_to_double(%Value %a)
  %max_b_d = call double @value_to_double(%Value %b)
  %max_gt_d = fcmp ogt double %max_a_d, %max_b_d
  %max_r_d = select i1 %max_gt_d, double %max_a_d, double %max_b_d
  %max_v_f = call %Value @value_float(double %max_r_d)
  ret %Value %max_v_f
ev_min:
  br i1 %either_float, label %ev_min_f, label %ev_min_i
ev_min_i:
  %min_a_i = extractvalue %Value %a, 1
  %min_b_i = extractvalue %Value %b, 1
  %min_lt_i = icmp slt i64 %min_a_i, %min_b_i
  %min_r_i = select i1 %min_lt_i, i64 %min_a_i, i64 %min_b_i
  %min_v_i = call %Value @value_integer(i64 %min_r_i)
  ret %Value %min_v_i
ev_min_f:
  %min_a_d = call double @value_to_double(%Value %a)
  %min_b_d = call double @value_to_double(%Value %b)
  %min_lt_d = fcmp olt double %min_a_d, %min_b_d
  %min_r_d = select i1 %min_lt_d, double %min_a_d, double %min_b_d
  %min_v_f = call %Value @value_float(double %min_r_d)
  ret %Value %min_v_f

ev_check_unary:
  %is_unary = icmp eq i32 %arity, 1
  br i1 %is_unary, label %ev_eval_unary, label %ev_zero
ev_eval_unary:
  %u_ptr = getelementptr %Value, %Value* %args, i32 0
  %u_raw = load %Value, %Value* %u_ptr
  %u = call %Value @eval_arith_value(%WamState* %vm, %Value %u_raw)
  %u_tag = extractvalue %Value %u, 0
  %u_is_float = icmp eq i32 %u_tag, 2
  ; M18/M20: dispatch on the first functor byte. Most unary math ops
  ; are uniquely identified by their first letter; the c / s / t
  ; prefixes have collisions handled by second-byte dispatch in the
  ; *_disambig labels.
  switch i8 %fn0, label %ev_zero [
    i8 45,  label %ev_neg       ; ''-''
    i8 97,  label %ev_a_disambig ; ''a'' -> abs | asin | acos | atan
    i8 114, label %ev_round     ; ''r''
    i8 102, label %ev_floor     ; ''f''
    i8 99,  label %ev_ceil_cos  ; ''c'' -> ceiling | cos
    i8 115, label %ev_sqrt_sign ; ''s'' -> sqrt | sign | sin
    i8 116, label %ev_trunc_tan ; ''t'' -> truncate | tan
    i8 108, label %ev_log       ; ''l''
    i8 101, label %ev_exp       ; ''e''
    i8 92,  label %ev_bnot      ; ''\\'' -> unary bitwise NOT (M85)
  ]
ev_neg:
  br i1 %u_is_float, label %ev_neg_f, label %ev_neg_i
ev_neg_i:
  %neg_u_i = extractvalue %Value %u, 1
  %neg_r_i = sub i64 0, %neg_u_i
  %neg_v_i = call %Value @value_integer(i64 %neg_r_i)
  ret %Value %neg_v_i
ev_neg_f:
  %neg_u_d = call double @value_to_double(%Value %u)
  %neg_r_d = fsub double 0.0, %neg_u_d
  %neg_v_f = call %Value @value_float(double %neg_r_d)
  ret %Value %neg_v_f
ev_abs:
  br i1 %u_is_float, label %ev_abs_f, label %ev_abs_i
ev_abs_i:
  %abs_u_i = extractvalue %Value %u, 1
  %abs_neg_i = icmp slt i64 %abs_u_i, 0
  %abs_op_i = sub i64 0, %abs_u_i
  %abs_r_i = select i1 %abs_neg_i, i64 %abs_op_i, i64 %abs_u_i
  %abs_v_i = call %Value @value_integer(i64 %abs_r_i)
  ret %Value %abs_v_i
ev_abs_f:
  %abs_u_d = call double @value_to_double(%Value %u)
  %abs_r_d = call double @llvm.fabs.f64(double %abs_u_d)
  %abs_v_f = call %Value @value_float(double %abs_r_d)
  ret %Value %abs_v_f

; M80: second-byte dispatch for ''a''-prefix unary functions.
; ``abs'' alone keeps the original ev_abs path; asin / acos / atan
; route to libm and always return Float.
ev_a_disambig:
  %ad.fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %ad.fn1 = load i8, i8* %ad.fn1_ptr
  switch i8 %ad.fn1, label %ev_zero [
    i8 98,  label %ev_abs   ; ''b'' -> abs
    i8 115, label %ev_asin  ; ''s'' -> asin
    i8 99,  label %ev_acos  ; ''c'' -> acos
    i8 116, label %ev_atan  ; ''t'' -> atan
  ]

ev_asin:
  %asin_d = call double @value_to_double(%Value %u)
  %asin_r = call double @asin(double %asin_d)
  %asin_v = call %Value @value_float(double %asin_r)
  ret %Value %asin_v

ev_acos:
  %acos_d = call double @value_to_double(%Value %u)
  %acos_r = call double @acos(double %acos_d)
  %acos_v = call %Value @value_float(double %acos_r)
  ret %Value %acos_v

ev_atan:
  %atan_d = call double @value_to_double(%Value %u)
  %atan_r = call double @atan(double %atan_d)
  %atan_v = call %Value @value_float(double %atan_r)
  ret %Value %atan_v

ev_bnot:
  ; M85: unary ``\\'' -- integer bitwise NOT. xor with all-ones is
  ; the LLVM idiom (no dedicated ``not'' instruction).
  %bnot.u_i = extractvalue %Value %u, 1
  %bnot.r = xor i64 %bnot.u_i, -1
  %bnot.v = call %Value @value_integer(i64 %bnot.r)
  ret %Value %bnot.v

; M18 truncate/round/floor/ceiling: take a numeric Value, convert to
; double, apply the LLVM intrinsic, return Integer (Prolog''s
; truncate/round/floor/ceiling all yield Integer results).
ev_trunc:
  br i1 %u_is_float, label %ev_trunc_f, label %ev_trunc_i
ev_trunc_i:
  ret %Value %u
ev_trunc_f:
  %trunc_d = call double @value_to_double(%Value %u)
  %trunc_r_d = call double @llvm.trunc.f64(double %trunc_d)
  %trunc_r_i = fptosi double %trunc_r_d to i64
  %trunc_v = call %Value @value_integer(i64 %trunc_r_i)
  ret %Value %trunc_v

ev_round:
  br i1 %u_is_float, label %ev_round_f, label %ev_round_i
ev_round_i:
  ret %Value %u
ev_round_f:
  %round_d = call double @value_to_double(%Value %u)
  %round_r_d = call double @llvm.round.f64(double %round_d)
  %round_r_i = fptosi double %round_r_d to i64
  %round_v = call %Value @value_integer(i64 %round_r_i)
  ret %Value %round_v

ev_floor:
  br i1 %u_is_float, label %ev_floor_f, label %ev_floor_i
ev_floor_i:
  ret %Value %u
ev_floor_f:
  %floor_d = call double @value_to_double(%Value %u)
  %floor_r_d = call double @llvm.floor.f64(double %floor_d)
  %floor_r_i = fptosi double %floor_r_d to i64
  %floor_v = call %Value @value_integer(i64 %floor_r_i)
  ret %Value %floor_v

ev_trunc_tan:
  ; ''t'' -> truncate (second byte ''r'') | tan (second byte ''a'').
  %tt_fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %tt_fn1 = load i8, i8* %tt_fn1_ptr
  switch i8 %tt_fn1, label %ev_zero [
    i8 114, label %ev_trunc  ; ''r''
    i8 97,  label %ev_tan    ; ''a''
  ]

ev_tan:
  ; tan(x) = sin(x) / cos(x); always Float.
  %tan_d = call double @value_to_double(%Value %u)
  %tan_s = call double @llvm.sin.f64(double %tan_d)
  %tan_c = call double @llvm.cos.f64(double %tan_d)
  %tan_r = fdiv double %tan_s, %tan_c
  %tan_v = call %Value @value_float(double %tan_r)
  ret %Value %tan_v

ev_log:
  ; log(x) -> natural log, always Float.
  %log_d = call double @value_to_double(%Value %u)
  %log_r = call double @llvm.log.f64(double %log_d)
  %log_v = call %Value @value_float(double %log_r)
  ret %Value %log_v

ev_exp:
  ; exp(x) -> e^x, always Float.
  %exp_d = call double @value_to_double(%Value %u)
  %exp_r = call double @llvm.exp.f64(double %exp_d)
  %exp_v = call %Value @value_float(double %exp_r)
  ret %Value %exp_v

ev_ceil_cos:
  ; Disambiguate ceiling vs cos on the second byte: ''e'' -> ceiling,
  ; ''o'' -> cos. Anything else falls back to ev_zero.
  %cc_fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %cc_fn1 = load i8, i8* %cc_fn1_ptr
  switch i8 %cc_fn1, label %ev_zero [
    i8 101, label %ev_ceil  ; ''e''
    i8 111, label %ev_cos   ; ''o''
  ]

ev_cos:
  %cos_d = call double @value_to_double(%Value %u)
  %cos_r = call double @llvm.cos.f64(double %cos_d)
  %cos_v = call %Value @value_float(double %cos_r)
  ret %Value %cos_v

ev_ceil:
  br i1 %u_is_float, label %ev_ceil_f, label %ev_ceil_i
ev_ceil_i:
  ret %Value %u
ev_ceil_f:
  %ceil_d = call double @value_to_double(%Value %u)
  %ceil_r_d = call double @llvm.ceil.f64(double %ceil_d)
  %ceil_r_i = fptosi double %ceil_r_d to i64
  %ceil_v = call %Value @value_integer(i64 %ceil_r_i)
  ret %Value %ceil_v

ev_sqrt_sign:
  ; Disambiguate sqrt | sign | sin. Second byte routes to sqrt
  ; (``q``) directly; for ``i`` we peek at the third byte to split
  ; sign (``g``) from sin (``n``).
  %ss_fn1_ptr = getelementptr i8, i8* %fn_ptr, i32 1
  %ss_fn1 = load i8, i8* %ss_fn1_ptr
  switch i8 %ss_fn1, label %ev_zero [
    i8 113, label %ev_sqrt        ; ''q''
    i8 105, label %ev_sign_or_sin ; ''i''
  ]

ev_sign_or_sin:
  %ssi_fn2_ptr = getelementptr i8, i8* %fn_ptr, i32 2
  %ssi_fn2 = load i8, i8* %ssi_fn2_ptr
  switch i8 %ssi_fn2, label %ev_zero [
    i8 103, label %ev_sign  ; ''g'' (sign)
    i8 110, label %ev_sin   ; ''n'' (sin)
  ]

ev_sin:
  %sin_d = call double @value_to_double(%Value %u)
  %sin_r = call double @llvm.sin.f64(double %sin_d)
  %sin_v = call %Value @value_float(double %sin_r)
  ret %Value %sin_v

ev_sqrt:
  ; sqrt always returns Float (Prolog ``sqrt(4)`` -> 2.0, not 2).
  %sqrt_d = call double @value_to_double(%Value %u)
  %sqrt_r_d = call double @llvm.sqrt.f64(double %sqrt_d)
  %sqrt_v = call %Value @value_float(double %sqrt_r_d)
  ret %Value %sqrt_v

ev_sign:
  ; sign returns Integer for Integer input, Float for Float input
  ; (mirroring Prolog''s usual type-preserving sign).
  br i1 %u_is_float, label %ev_sign_f, label %ev_sign_i
ev_sign_i:
  %sign_i = extractvalue %Value %u, 1
  %sign_is_pos = icmp sgt i64 %sign_i, 0
  %sign_is_neg = icmp slt i64 %sign_i, 0
  %sign_pos_v = select i1 %sign_is_pos, i64 1, i64 0
  %sign_r_i = select i1 %sign_is_neg, i64 -1, i64 %sign_pos_v
  %sign_v_i = call %Value @value_integer(i64 %sign_r_i)
  ret %Value %sign_v_i
ev_sign_f:
  %sign_d = call double @value_to_double(%Value %u)
  %sign_is_pos_d = fcmp ogt double %sign_d, 0.0
  %sign_is_neg_d = fcmp olt double %sign_d, 0.0
  %sign_pos_d = select i1 %sign_is_pos_d, double 1.0, double 0.0
  %sign_r_d = select i1 %sign_is_neg_d, double -1.0, double %sign_pos_d
  %sign_v_f = call %Value @value_float(double %sign_r_d)
  ret %Value %sign_v_f

ev_zero:
  ; Unknown / fail -- box Integer 0 so callers do not crash, and SET
  ; THE ARITHMETIC ERROR FLAG so is/2 and the comparisons FAIL like
  ; SWI raises a type/evaluation error (capstone finding: without
  ; this, Y is 48 + F with F a non-arithmetic compound silently
  ; evaluated to 48, so the self-compiled compiler dispatch clause
  ; guarded by that failure wrongly succeeded -- the numbervarred
  ; marker pattern in its own source compiled as a plain variable and
  ; the self-compile diverged from the production compiler by one
  ; instruction per marker clause). The flag is cleared by each
  ; consuming builtin before evaluation.
  store i1 1, i1* @wam_arith_err
  %z_v = call %Value @value_integer(i64 0)
  ret %Value %z_v
}'.

compile_copy_term_to_llvm(Code) :-
    Code = '; Recursively copy a %Value with a variable map: fresh %Compound cells
; (arena), fresh heap cells for variables, and sharing preserved -- each
; distinct source variable maps to exactly ONE fresh cell. Atomic values
; (Atom / Integer / Float / Bool) copy by value. Cons-cell lists fall out
; automatically since they are just compounds with functor "." / arity 2.
define %Value @wam_copy_term_value(%WamState* %vm, %Value %v) {
entry:
  call void @wam_arena_ensure()
  %map_mem = call i8* @wam_arena_alloc(i64 2048)
  %map = bitcast i8* %map_mem to i64*
  %fresh_mem = call i8* @wam_arena_alloc(i64 4096)
  %freshes = bitcast i8* %fresh_mem to %Value*
  %countp = alloca i32
  store i32 0, i32* %countp
  %r = call %Value @wam_copy_term_rec(%WamState* %vm, %Value %v, i64* %map, %Value* %freshes, i32* %countp)
  ret %Value %r
}

; The recursive worker behind copy_term/2. Derefs via @wam_deref_keep_var
; (follows Ref chains but keeps the final Ref, so an unbound variable has
; an ADDRESS to key the map on). tag 5 Ref (an unbound var): look the heap
; index up in the map; a hit reuses the fresh cell already made for that
; source variable (sharing preserved: p(X,X) copies with ONE fresh var), a
; miss pushes a fresh heap cell and records the pair (map capacity 256;
; beyond it, extra vars copy fresh-but-unshared, never aliased). The OLD
; implementation returned Refs unchanged (they fell into the atomic
; default), so the "copy" aliased the source heap cells and numbervars /
; unification on the copy bound the ORIGINAL term through them.
define %Value @wam_copy_term_rec(%WamState* %vm, %Value %v, i64* %map, %Value* %freshes, i32* %countp) {
entry:
  %kv = call %Value @wam_deref_keep_var(%WamState* %vm, %Value %v)
  %tag = call i32 @value_tag(%Value %kv)
  switch i32 %tag, label %atomic [
    i32 3, label %ct_compound
    i32 5, label %ct_var
    i32 6, label %ct_unbound
  ]

atomic:
  ret %Value %kv

ct_unbound:
  ; a bare Unbound value with no heap identity: cannot share, copy fresh.
  %fresh_unb = call %Value @value_unbound(i8* null)
  ret %Value %fresh_unb

ct_var:
  %addr = call i64 @value_payload(%Value %kv)
  %count = load i32, i32* %countp
  br label %scan
scan:
  %i = phi i32 [ 0, %ct_var ], [ %inext, %scan_cont ]
  %seen_all = icmp sge i32 %i, %count
  br i1 %seen_all, label %miss, label %probe
probe:
  %mp = getelementptr i64, i64* %map, i32 %i
  %ma = load i64, i64* %mp
  %hit = icmp eq i64 %ma, %addr
  br i1 %hit, label %use, label %scan_cont
scan_cont:
  %inext = add i32 %i, 1
  br label %scan
use:
  %fp = getelementptr %Value, %Value* %freshes, i32 %i
  %fv0 = load %Value, %Value* %fp
  ret %Value %fv0
miss:
  %unb = call %Value @value_unbound(i8* null)
  %ha = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %fv = call %Value @value_ref(i32 %ha)
  %room = icmp slt i32 %count, 256
  br i1 %room, label %record, label %no_record
record:
  %mp2 = getelementptr i64, i64* %map, i32 %count
  store i64 %addr, i64* %mp2
  %fp2 = getelementptr %Value, %Value* %freshes, i32 %count
  store %Value %fv, %Value* %fp2
  %count1 = add i32 %count, 1
  store i32 %count1, i32* %countp
  br label %no_record
no_record:
  ret %Value %fv

ct_compound:
  %cp_bits = call i64 @value_payload(%Value %kv)
  %cp_ptr = inttoptr i64 %cp_bits to %Compound*
  %fn_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 0
  %fn_ptr = load i8*, i8** %fn_slot
  %ar_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 1
  %arity = load i32, i32* %ar_slot
  %args_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 2
  %src_args = load %Value*, %Value** %args_slot

  %cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %new_mem = call i8* @wam_arena_alloc(i64 %cp_size)
  %new_cp = bitcast i8* %new_mem to %Compound*
  %new_fn = getelementptr %Compound, %Compound* %new_cp, i32 0, i32 0
  store i8* %fn_ptr, i8** %new_fn
  %new_ar = getelementptr %Compound, %Compound* %new_cp, i32 0, i32 1
  store i32 %arity, i32* %new_ar

  %ar64 = zext i32 %arity to i64
  %args_bytes = shl i64 %ar64, 4
  %new_args_mem = call i8* @wam_arena_alloc(i64 %args_bytes)
  %new_args = bitcast i8* %new_args_mem to %Value*
  %new_args_slot = getelementptr %Compound, %Compound* %new_cp, i32 0, i32 2
  store %Value* %new_args, %Value** %new_args_slot

  %has_args = icmp sgt i32 %arity, 0
  br i1 %has_args, label %ct_loop_entry, label %ct_done

ct_loop_entry:
  br label %ct_loop

ct_loop:
  %i2 = phi i32 [ 0, %ct_loop_entry ], [ %i_next, %ct_loop_step ]
  %src_ptr = getelementptr %Value, %Value* %src_args, i32 %i2
  %src_val = load %Value, %Value* %src_ptr
  %dst_val = call %Value @wam_copy_term_rec(%WamState* %vm, %Value %src_val, i64* %map, %Value* %freshes, i32* %countp)
  %dst_ptr = getelementptr %Value, %Value* %new_args, i32 %i2
  store %Value %dst_val, %Value* %dst_ptr
  %i_next = add i32 %i2, 1
  %more = icmp slt i32 %i_next, %arity
  br i1 %more, label %ct_loop_step, label %ct_done

ct_loop_step:
  br label %ct_loop

ct_done:
  %new_i64 = ptrtoint %Compound* %new_cp to i64
  %new_val0 = insertvalue %Value undef, i32 3, 0
  %new_val = insertvalue %Value %new_val0, i64 %new_i64, 1
  ret %Value %new_val
}

; M103: recursive depth-first left-to-right walk over a term,
; prepending each Unbound variable onto an accumulator list. The
; outer driver passes the empty [] atom-id list as initial acc;
; walking args right-to-left + prepending keeps the leftmost var
; at the front of the result. Does NOT deduplicate -- repeated
; occurrences of the same variable appear once per occurrence.
; SWI semantics technically dedupes -- known limitation for M103.
define %Value @wam_collect_vars(%WamState* %vm, %Value %term, %Value %acc) {
entry:
  %tv.t = call %Value @wam_deref_value(%WamState* %vm, %Value %term)
  %tv.tag = call i32 @value_tag(%Value %tv.t)
  switch i32 %tv.tag, label %tv.atomic [
    i32 6, label %tv.unbound
    i32 3, label %tv.compound
  ]

tv.atomic:
  ret %Value %acc

tv.unbound:
  call void @wam_arena_ensure()
  %tv.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %tv.cp_mem = call i8* @wam_arena_alloc(i64 %tv.cp_size)
  %tv.cp = bitcast i8* %tv.cp_mem to %Compound*
  %tv.fn_slot = getelementptr %Compound, %Compound* %tv.cp, i32 0, i32 0
  store i8* getelementptr ([4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0), i8** %tv.fn_slot
  %tv.ar_slot = getelementptr %Compound, %Compound* %tv.cp, i32 0, i32 1
  store i32 2, i32* %tv.ar_slot
  %tv.args_mem = call i8* @wam_arena_alloc(i64 32)
  %tv.args = bitcast i8* %tv.args_mem to %Value*
  %tv.args_slot = getelementptr %Compound, %Compound* %tv.cp, i32 0, i32 2
  store %Value* %tv.args, %Value** %tv.args_slot
  %tv.head_ptr = getelementptr %Value, %Value* %tv.args, i32 0
  ; Store the ORIGINAL %term (a Ref into the heap cell), not the
  ; deref''d Unbound payload. Var identity matters: callers do
  ; term_variables(foo(X, _), [V|_]) followed by V == X, which
  ; needs V and X to share the same Ref address. Storing the
  ; deref''d Unbound would give a tag-6 sentinel that compares
  ; unequal to the original Ref.
  store %Value %term, %Value* %tv.head_ptr
  %tv.tail_ptr = getelementptr %Value, %Value* %tv.args, i32 1
  store %Value %acc, %Value* %tv.tail_ptr
  %tv.cp_i64 = ptrtoint %Compound* %tv.cp to i64
  %tv.v0 = insertvalue %Value undef, i32 3, 0
  %tv.v = insertvalue %Value %tv.v0, i64 %tv.cp_i64, 1
  ret %Value %tv.v

tv.compound:
  %tv.cp_bits = call i64 @value_payload(%Value %tv.t)
  %tv.src_cp = inttoptr i64 %tv.cp_bits to %Compound*
  %tv.src_ar_slot = getelementptr %Compound, %Compound* %tv.src_cp, i32 0, i32 1
  %tv.arity = load i32, i32* %tv.src_ar_slot
  %tv.src_args_slot = getelementptr %Compound, %Compound* %tv.src_cp, i32 0, i32 2
  %tv.src_args = load %Value*, %Value** %tv.src_args_slot
  %tv.i0 = sub i32 %tv.arity, 1
  %tv.has_args = icmp sge i32 %tv.i0, 0
  br i1 %tv.has_args, label %tv.loop, label %tv.ret_acc

tv.loop:
  %tv.i = phi i32 [ %tv.i0, %tv.compound ], [ %tv.i_next, %tv.step ]
  %tv.acc_cur = phi %Value [ %acc, %tv.compound ], [ %tv.acc_next, %tv.step ]
  %tv.arg_ptr = getelementptr %Value, %Value* %tv.src_args, i32 %tv.i
  %tv.arg = load %Value, %Value* %tv.arg_ptr
  %tv.acc_next = call %Value @wam_collect_vars(%WamState* %vm, %Value %tv.arg, %Value %tv.acc_cur)
  %tv.done = icmp eq i32 %tv.i, 0
  br i1 %tv.done, label %tv.ret_loop, label %tv.step

tv.step:
  %tv.i_next = sub i32 %tv.i, 1
  br label %tv.loop

tv.ret_loop:
  ret %Value %tv.acc_next

tv.ret_acc:
  ret %Value %acc
}

; M117: recursive ground check. Returns true iff %term contains no
; unbound variables. Atomic (atom/int/float) -> true; unbound -> false;
; compound -> recurse over each arg, false if any arg is non-ground.
define i1 @wam_is_ground(%WamState* %vm, %Value %term) {
entry:
  %ig.t = call %Value @wam_deref_value(%WamState* %vm, %Value %term)
  %ig.tag = call i32 @value_tag(%Value %ig.t)
  switch i32 %ig.tag, label %ig.atomic [
    i32 6, label %ig.unbound
    i32 3, label %ig.compound
  ]

ig.atomic:
  ret i1 true

ig.unbound:
  ret i1 false

ig.compound:
  %ig.cp_bits = call i64 @value_payload(%Value %ig.t)
  %ig.src_cp = inttoptr i64 %ig.cp_bits to %Compound*
  %ig.ar_slot = getelementptr %Compound, %Compound* %ig.src_cp, i32 0, i32 1
  %ig.arity = load i32, i32* %ig.ar_slot
  %ig.args_slot = getelementptr %Compound, %Compound* %ig.src_cp, i32 0, i32 2
  %ig.src_args = load %Value*, %Value** %ig.args_slot
  %ig.has = icmp sgt i32 %ig.arity, 0
  br i1 %ig.has, label %ig.loop, label %ig.true

ig.loop:
  %ig.i = phi i32 [ 0, %ig.compound ], [ %ig.i_next, %ig.step ]
  %ig.arg_ptr = getelementptr %Value, %Value* %ig.src_args, i32 %ig.i
  %ig.arg = load %Value, %Value* %ig.arg_ptr
  %ig.sub = call i1 @wam_is_ground(%WamState* %vm, %Value %ig.arg)
  br i1 %ig.sub, label %ig.step, label %ig.false

ig.step:
  %ig.i_next = add i32 %ig.i, 1
  %ig.more = icmp slt i32 %ig.i_next, %ig.arity
  br i1 %ig.more, label %ig.loop, label %ig.true

ig.true:
  ret i1 true

ig.false:
  ret i1 false
}

; M137: proper deref-at-every-level strict equality for ==/2 and the
; structurally-inequal sibling (op id 21).
; The pre-existing @value_equals (in value.ll.mustache) compares tags
; and payloads shallowly and only derefs the top-level operands via the
; caller. Recursive arg comparison loads the heap-stored %Value
; verbatim, which means if an arg slot holds a Ref (the case for many
; put_structure-built compounds), tag-eq becomes Ref==Ref and the
; payload comparison is heap-address-vs-heap-address -- almost always
; false even when the deref''d values are identical. Symptom uncovered
; in M136: ``[a, b] == [a, b]'' returned false for two literal lists.
;
; This helper derefs on entry AND inside the recursive arg loop so
; structural equality holds regardless of how nested args are stored.
;
; M138: deref via @wam_deref_keep_var, NOT @wam_deref_value. The full
; deref collapses a Ref-to-unbound-cell into the shared Unbound
; sentinel { tag 6, payload 0 }, which erases variable identity --
; X == Y would be true for two DISTINCT fresh vars (and sort/2 dedup
; would merge f(X) with f(Y)). Stopping at the last Ref keeps the
; variable represented as Ref{cell}, so the tag-5 shallow payload
; compare becomes cell-address identity: X == X true, X == Y false.

; Chase Ref chains but stop at the final Ref when the referenced cell
; is unbound -- the Ref IS the variable''s identity. Bound cells chase
; through as usual. Hop-limited like @wam_deref_value.
define %Value @wam_deref_keep_var(%WamState* %vm, %Value %v) alwaysinline nounwind {
entry:
  br label %dkv.loop
dkv.loop:
  %dkv.cur = phi %Value [ %v, %entry ], [ %dkv.cell, %dkv.follow ]
  %dkv.i = phi i32 [ 0, %entry ], [ %dkv.i_next, %dkv.follow ]
  %dkv.tag = extractvalue %Value %dkv.cur, 0
  %dkv.is_ref = icmp eq i32 %dkv.tag, 5
  %dkv.hop_ok = icmp slt i32 %dkv.i, 64
  %dkv.cont = and i1 %dkv.is_ref, %dkv.hop_ok
  br i1 %dkv.cont, label %dkv.peek, label %dkv.done
dkv.peek:
  %dkv.addr_i64 = extractvalue %Value %dkv.cur, 1
  %dkv.addr = trunc i64 %dkv.addr_i64 to i32
  %dkv.cell = call %Value @wam_heap_get(%WamState* %vm, i32 %dkv.addr)
  %dkv.cell_unb = call i1 @value_is_unbound(%Value %dkv.cell)
  br i1 %dkv.cell_unb, label %dkv.done, label %dkv.follow
dkv.follow:
  %dkv.i_next = add i32 %dkv.i, 1
  br label %dkv.loop
dkv.done:
  ret %Value %dkv.cur
}

define i1 @wam_strict_eq(%WamState* %vm, %Value %a, %Value %b) {
entry:
  %seq.da = call %Value @wam_deref_keep_var(%WamState* %vm, %Value %a)
  %seq.db = call %Value @wam_deref_keep_var(%WamState* %vm, %Value %b)
  %seq.tag_a = extractvalue %Value %seq.da, 0
  %seq.tag_b = extractvalue %Value %seq.db, 0
  %seq.tags_eq = icmp eq i32 %seq.tag_a, %seq.tag_b
  br i1 %seq.tags_eq, label %seq.same_tag, label %seq.not_equal

seq.same_tag:
  %seq.is_compound = icmp eq i32 %seq.tag_a, 3
  br i1 %seq.is_compound, label %seq.compound_case, label %seq.shallow_payload

seq.shallow_payload:
  %seq.pay_a = extractvalue %Value %seq.da, 1
  %seq.pay_b = extractvalue %Value %seq.db, 1
  %seq.pay_eq = icmp eq i64 %seq.pay_a, %seq.pay_b
  ret i1 %seq.pay_eq

seq.compound_case:
  %seq.pa = extractvalue %Value %seq.da, 1
  %seq.pb = extractvalue %Value %seq.db, 1
  %seq.same_ptr = icmp eq i64 %seq.pa, %seq.pb
  br i1 %seq.same_ptr, label %seq.equal, label %seq.deep_check

seq.deep_check:
  %seq.ap = inttoptr i64 %seq.pa to %Compound*
  %seq.bp = inttoptr i64 %seq.pb to %Compound*
  %seq.afn_slot = getelementptr %Compound, %Compound* %seq.ap, i32 0, i32 0
  %seq.bfn_slot = getelementptr %Compound, %Compound* %seq.bp, i32 0, i32 0
  %seq.afn = load i8*, i8** %seq.afn_slot
  %seq.bfn = load i8*, i8** %seq.bfn_slot
  %seq.fn_eq = call i1 @wam_functor_eq(i8* %seq.afn, i8* %seq.bfn)
  br i1 %seq.fn_eq, label %seq.check_arity, label %seq.not_equal

seq.check_arity:
  %seq.aar_slot = getelementptr %Compound, %Compound* %seq.ap, i32 0, i32 1
  %seq.bar_slot = getelementptr %Compound, %Compound* %seq.bp, i32 0, i32 1
  %seq.aar = load i32, i32* %seq.aar_slot
  %seq.bar = load i32, i32* %seq.bar_slot
  %seq.ar_eq = icmp eq i32 %seq.aar, %seq.bar
  br i1 %seq.ar_eq, label %seq.args_init, label %seq.not_equal

seq.args_init:
  %seq.aargs_slot = getelementptr %Compound, %Compound* %seq.ap, i32 0, i32 2
  %seq.bargs_slot = getelementptr %Compound, %Compound* %seq.bp, i32 0, i32 2
  %seq.aargs = load %Value*, %Value** %seq.aargs_slot
  %seq.bargs = load %Value*, %Value** %seq.bargs_slot
  %seq.has_args = icmp sgt i32 %seq.aar, 0
  br i1 %seq.has_args, label %seq.loop, label %seq.equal

seq.loop:
  %seq.idx = phi i32 [ 0, %seq.args_init ], [ %seq.next_idx, %seq.step ]
  %seq.ae_ptr = getelementptr %Value, %Value* %seq.aargs, i32 %seq.idx
  %seq.ae = load %Value, %Value* %seq.ae_ptr
  %seq.be_ptr = getelementptr %Value, %Value* %seq.bargs, i32 %seq.idx
  %seq.be = load %Value, %Value* %seq.be_ptr
  %seq.arg_eq = call i1 @wam_strict_eq(%WamState* %vm, %Value %seq.ae, %Value %seq.be)
  br i1 %seq.arg_eq, label %seq.step, label %seq.not_equal

seq.step:
  %seq.next_idx = add i32 %seq.idx, 1
  %seq.done = icmp sge i32 %seq.next_idx, %seq.aar
  br i1 %seq.done, label %seq.equal, label %seq.loop

seq.equal:
  ret i1 true

seq.not_equal:
  ret i1 false
}

; M105: recursive depth-first left-to-right walk that binds each
; Unbound variable encountered to a fresh ''$VAR''(N) compound on
; the arena. Counter n is threaded through and returned -- caller
; gets the next-available N to unify with End.
define i64 @wam_numbervars_walk(%WamState* %vm, %Value %term, i64 %n) {
entry:
  %nw.t = call %Value @wam_deref_value(%WamState* %vm, %Value %term)
  %nw.tag = call i32 @value_tag(%Value %nw.t)
  switch i32 %nw.tag, label %nw.atomic [
    i32 6, label %nw.unbound
    i32 3, label %nw.compound
  ]

nw.atomic:
  ret i64 %n

nw.unbound:
  ; Build the ''$VAR''(n) compound on the arena. Then bind the
  ; variable''s heap cell to it via wam_trail_heap_binding + heap_set
  ; so backtrack restores. Use the ORIGINAL term (a Ref) to locate
  ; the heap cell; deref''d nw.t is just an Unbound sentinel.
  call void @wam_arena_ensure()
  %nw.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %nw.cp_mem = call i8* @wam_arena_alloc(i64 %nw.cp_size)
  %nw.cp = bitcast i8* %nw.cp_mem to %Compound*
  %nw.fn_slot = getelementptr %Compound, %Compound* %nw.cp, i32 0, i32 0
  store i8* getelementptr ([5 x i8], [5 x i8]* @.fn__24VAR, i32 0, i32 0), i8** %nw.fn_slot
  %nw.ar_slot = getelementptr %Compound, %Compound* %nw.cp, i32 0, i32 1
  store i32 1, i32* %nw.ar_slot
  %nw.args_mem = call i8* @wam_arena_alloc(i64 16)
  %nw.args = bitcast i8* %nw.args_mem to %Value*
  %nw.args_slot = getelementptr %Compound, %Compound* %nw.cp, i32 0, i32 2
  store %Value* %nw.args, %Value** %nw.args_slot
  %nw.n_v = call %Value @value_integer(i64 %n)
  %nw.arg0_ptr = getelementptr %Value, %Value* %nw.args, i32 0
  store %Value %nw.n_v, %Value* %nw.arg0_ptr
  %nw.cp_i64 = ptrtoint %Compound* %nw.cp to i64
  %nw.varN_v0 = insertvalue %Value undef, i32 3, 0
  %nw.varN_v = insertvalue %Value %nw.varN_v0, i64 %nw.cp_i64, 1
  ; Bind the var. The original %term is the Ref into the heap cell.
  %nw.t_tag = extractvalue %Value %term, 0
  %nw.t_isref = icmp eq i32 %nw.t_tag, 5
  br i1 %nw.t_isref, label %nw.bind, label %nw.skip
nw.bind:
  %nw.addr_i64 = extractvalue %Value %term, 1
  %nw.addr = trunc i64 %nw.addr_i64 to i32
  %nw.old = call %Value @wam_heap_get(%WamState* %vm, i32 %nw.addr)
  call void @wam_trail_heap_binding(%WamState* %vm, i32 %nw.addr, %Value %nw.old)
  call void @wam_heap_set(%WamState* %vm, i32 %nw.addr, %Value %nw.varN_v)
  br label %nw.skip
nw.skip:
  %nw.n1 = add i64 %n, 1
  ret i64 %nw.n1

nw.compound:
  %nw.cp_bits = call i64 @value_payload(%Value %nw.t)
  %nw.src_cp = inttoptr i64 %nw.cp_bits to %Compound*
  %nw.src_ar_slot = getelementptr %Compound, %Compound* %nw.src_cp, i32 0, i32 1
  %nw.arity = load i32, i32* %nw.src_ar_slot
  %nw.src_args_slot = getelementptr %Compound, %Compound* %nw.src_cp, i32 0, i32 2
  %nw.src_args = load %Value*, %Value** %nw.src_args_slot
  %nw.has_args = icmp sgt i32 %nw.arity, 0
  br i1 %nw.has_args, label %nw.loop, label %nw.ret_n

nw.loop:
  %nw.i = phi i32 [ 0, %nw.compound ], [ %nw.i_next, %nw.step ]
  %nw.n_cur = phi i64 [ %n, %nw.compound ], [ %nw.n_next, %nw.step ]
  %nw.arg_ptr = getelementptr %Value, %Value* %nw.src_args, i32 %nw.i
  %nw.arg = load %Value, %Value* %nw.arg_ptr
  %nw.n_next = call i64 @wam_numbervars_walk(%WamState* %vm, %Value %nw.arg, i64 %nw.n_cur)
  %nw.i_next = add i32 %nw.i, 1
  %nw.done = icmp sge i32 %nw.i_next, %nw.arity
  br i1 %nw.done, label %nw.ret_loop, label %nw.step

nw.step:
  br label %nw.loop

nw.ret_loop:
  ret i64 %nw.n_next

nw.ret_n:
  ret i64 %n
}

; M11: deref-aware deep-copy. Resolves Refs to the heap-stored value,
; then copies Compounds (recursing on args) so the returned %Value
; is self-contained -- it survives a heap rewind. Used by
; end_aggregate to capture per-iteration findall / setof results
; whose Compound args otherwise point into the per-iteration heap
; region that backtrack reclaims, silently aliasing every accumulated
; entry to the LAST iteration''s values.
;
; Atoms / Integers / Floats pass through. Unbounds after deref stay
; unbound (best effort -- typical aggregate templates have all vars
; bound by the time end_aggregate runs; an unbound template arg
; indicates a logic bug the caller will see via the spurious
; unbound in the result list).
define %Value @wam_freeze_value(%WamState* %vm, %Value %v) {
entry:
  %d = call %Value @wam_deref_value(%WamState* %vm, %Value %v)
  %tag = extractvalue %Value %d, 0
  %is_cmp = icmp eq i32 %tag, 3
  br i1 %is_cmp, label %fz_compound, label %fz_atomic

fz_atomic:
  ; Atom (0), Integer (1), Float (2), Ref-to-unbound (5 derefs to 6),
  ; Unbound (6), Bool (7) -- all self-contained values; copy nothing.
  ret %Value %d

fz_compound:
  %cp_bits = extractvalue %Value %d, 1
  %cp_ptr = inttoptr i64 %cp_bits to %Compound*
  %fn_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 0
  %fn_ptr = load i8*, i8** %fn_slot
  %ar_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 1
  %arity = load i32, i32* %ar_slot
  %args_slot = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 2
  %src_args = load %Value*, %Value** %args_slot

  call void @wam_arena_ensure()
  %cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %new_mem = call i8* @wam_arena_alloc(i64 %cp_size)
  %new_cp = bitcast i8* %new_mem to %Compound*
  %new_fn_slot = getelementptr %Compound, %Compound* %new_cp, i32 0, i32 0
  store i8* %fn_ptr, i8** %new_fn_slot
  %new_ar_slot = getelementptr %Compound, %Compound* %new_cp, i32 0, i32 1
  store i32 %arity, i32* %new_ar_slot
  %ar64 = zext i32 %arity to i64
  %args_bytes = shl i64 %ar64, 4
  %new_args_mem = call i8* @wam_arena_alloc(i64 %args_bytes)
  %new_args = bitcast i8* %new_args_mem to %Value*
  %new_args_slot = getelementptr %Compound, %Compound* %new_cp, i32 0, i32 2
  store %Value* %new_args, %Value** %new_args_slot

  %has_args = icmp sgt i32 %arity, 0
  br i1 %has_args, label %fz_loop, label %fz_done

fz_loop:
  %i = phi i32 [ 0, %fz_compound ], [ %i_next, %fz_step ]
  %src_ptr = getelementptr %Value, %Value* %src_args, i32 %i
  %src_val = load %Value, %Value* %src_ptr
  %dst_val = call %Value @wam_freeze_value(%WamState* %vm, %Value %src_val)
  %dst_ptr = getelementptr %Value, %Value* %new_args, i32 %i
  store %Value %dst_val, %Value* %dst_ptr
  br label %fz_step

fz_step:
  %i_next = add i32 %i, 1
  %more = icmp slt i32 %i_next, %arity
  br i1 %more, label %fz_loop, label %fz_done

fz_done:
  %new_i64 = ptrtoint %Compound* %new_cp to i64
  %new_val0 = insertvalue %Value undef, i32 3, 0
  %new_val = insertvalue %Value %new_val0, i64 %new_i64, 1
  ret %Value %new_val
}'.

%% compile_term_cmp_to_llvm(-IR)
%
%  M64: emit @wam_term_cmp(vm, v1, v2) -> i32 (-1, 0, +1) per standard
%  term order. Supports Atom / Integer / Float / Compound with proper
%  recursion through compound args. Used by builtin_compare (and
%  future keysort / sort migrations).
compile_term_cmp_to_llvm(Code) :-
    Code = '; M64: standard-term-order compare. Returns -1, 0, or +1.
; Category order (ISO): Number < Atom < Compound. Within each
; category, atoms by strcmp, numbers by value (mixed int/float
; widened to double), compounds by arity then functor then args
; left-to-right (recursive).
define i32 @wam_term_cmp(%WamState* %vm, %Value %v1, %Value %v2) {
entry:
  %d1 = call %Value @wam_deref_value(%WamState* %vm, %Value %v1)
  %d2 = call %Value @wam_deref_value(%WamState* %vm, %Value %v2)
  %t1 = extractvalue %Value %d1, 0
  %t2 = extractvalue %Value %d2, 0
  ; Map tag -> category: Atom=1, Number=0, Compound=2.
  %t1_atom = icmp eq i32 %t1, 0
  %t1_cmp = icmp eq i32 %t1, 3
  %t1_cat_mid = select i1 %t1_cmp, i32 2, i32 0
  %cat1 = select i1 %t1_atom, i32 1, i32 %t1_cat_mid
  %t2_atom = icmp eq i32 %t2, 0
  %t2_cmp = icmp eq i32 %t2, 3
  %t2_cat_mid = select i1 %t2_cmp, i32 2, i32 0
  %cat2 = select i1 %t2_atom, i32 1, i32 %t2_cat_mid
  %cat_eq = icmp eq i32 %cat1, %cat2
  br i1 %cat_eq, label %same_cat, label %diff_cat
diff_cat:
  %dc_lt = icmp slt i32 %cat1, %cat2
  %dc_r = select i1 %dc_lt, i32 -1, i32 1
  ret i32 %dc_r
same_cat:
  %sc_is_atom = icmp eq i32 %cat1, 1
  br i1 %sc_is_atom, label %cmp_atoms, label %sc_num_or_cmp
sc_num_or_cmp:
  %sc_is_cmp = icmp eq i32 %cat1, 2
  br i1 %sc_is_cmp, label %cmp_compounds, label %cmp_nums
cmp_atoms:
  %a_id1 = extractvalue %Value %d1, 1
  %a_id2 = extractvalue %Value %d2, 1
  %a_str1 = call i8* @wam_atom_to_string(i64 %a_id1)
  %a_str2 = call i8* @wam_atom_to_string(i64 %a_id2)
  %a_r = call i32 @strcmp(i8* %a_str1, i8* %a_str2)
  %a_lt = icmp slt i32 %a_r, 0
  %a_gt = icmp sgt i32 %a_r, 0
  %a_mid = select i1 %a_gt, i32 1, i32 0
  %a_result = select i1 %a_lt, i32 -1, i32 %a_mid
  ret i32 %a_result
cmp_nums:
  %n_t1_int = icmp eq i32 %t1, 1
  %n_t2_int = icmp eq i32 %t2, 1
  %n_both_int = and i1 %n_t1_int, %n_t2_int
  br i1 %n_both_int, label %cmp_int_int, label %cmp_num_float
cmp_int_int:
  %ii_v1 = extractvalue %Value %d1, 1
  %ii_v2 = extractvalue %Value %d2, 1
  %ii_lt = icmp slt i64 %ii_v1, %ii_v2
  %ii_gt = icmp sgt i64 %ii_v1, %ii_v2
  %ii_mid = select i1 %ii_gt, i32 1, i32 0
  %ii_result = select i1 %ii_lt, i32 -1, i32 %ii_mid
  ret i32 %ii_result
cmp_num_float:
  %nf_b1 = extractvalue %Value %d1, 1
  %nf_b2 = extractvalue %Value %d2, 1
  %nf_t1_flt = icmp eq i32 %t1, 2
  %nf_t2_flt = icmp eq i32 %t2, 2
  %nf_d1_flt = bitcast i64 %nf_b1 to double
  %nf_d1_int = sitofp i64 %nf_b1 to double
  %nf_d1 = select i1 %nf_t1_flt, double %nf_d1_flt, double %nf_d1_int
  %nf_d2_flt = bitcast i64 %nf_b2 to double
  %nf_d2_int = sitofp i64 %nf_b2 to double
  %nf_d2 = select i1 %nf_t2_flt, double %nf_d2_flt, double %nf_d2_int
  %nf_lt = fcmp olt double %nf_d1, %nf_d2
  %nf_gt = fcmp ogt double %nf_d1, %nf_d2
  %nf_mid = select i1 %nf_gt, i32 1, i32 0
  %nf_result = select i1 %nf_lt, i32 -1, i32 %nf_mid
  ret i32 %nf_result
cmp_compounds:
  %c_bits1 = extractvalue %Value %d1, 1
  %c_bits2 = extractvalue %Value %d2, 1
  %c_cp1 = inttoptr i64 %c_bits1 to %Compound*
  %c_cp2 = inttoptr i64 %c_bits2 to %Compound*
  %c_ar1_slot = getelementptr %Compound, %Compound* %c_cp1, i32 0, i32 1
  %c_ar1 = load i32, i32* %c_ar1_slot
  %c_ar2_slot = getelementptr %Compound, %Compound* %c_cp2, i32 0, i32 1
  %c_ar2 = load i32, i32* %c_ar2_slot
  %c_ar_eq = icmp eq i32 %c_ar1, %c_ar2
  br i1 %c_ar_eq, label %cmp_funcs, label %c_ar_diff
c_ar_diff:
  %c_ar_lt = icmp slt i32 %c_ar1, %c_ar2
  %c_ar_r = select i1 %c_ar_lt, i32 -1, i32 1
  ret i32 %c_ar_r
cmp_funcs:
  %c_fn1_slot = getelementptr %Compound, %Compound* %c_cp1, i32 0, i32 0
  %c_fn1 = load i8*, i8** %c_fn1_slot
  %c_fn2_slot = getelementptr %Compound, %Compound* %c_cp2, i32 0, i32 0
  %c_fn2 = load i8*, i8** %c_fn2_slot
  %c_fn_r = call i32 @strcmp(i8* %c_fn1, i8* %c_fn2)
  %c_fn_zero = icmp eq i32 %c_fn_r, 0
  br i1 %c_fn_zero, label %cmp_args_init, label %c_fn_diff
c_fn_diff:
  %c_fn_lt = icmp slt i32 %c_fn_r, 0
  %c_fn_result = select i1 %c_fn_lt, i32 -1, i32 1
  ret i32 %c_fn_result
cmp_args_init:
  %c_args1_slot = getelementptr %Compound, %Compound* %c_cp1, i32 0, i32 2
  %c_args1 = load %Value*, %Value** %c_args1_slot
  %c_args2_slot = getelementptr %Compound, %Compound* %c_cp2, i32 0, i32 2
  %c_args2 = load %Value*, %Value** %c_args2_slot
  br label %cmp_args_loop
cmp_args_loop:
  %ca_i = phi i32 [ 0, %cmp_args_init ], [ %ca_i_next, %cmp_args_continue ]
  %ca_done = icmp sge i32 %ca_i, %c_ar1
  br i1 %ca_done, label %cmp_args_equal, label %cmp_args_step
cmp_args_step:
  %ca_a1_ptr = getelementptr %Value, %Value* %c_args1, i32 %ca_i
  %ca_a1 = load %Value, %Value* %ca_a1_ptr
  %ca_a2_ptr = getelementptr %Value, %Value* %c_args2, i32 %ca_i
  %ca_a2 = load %Value, %Value* %ca_a2_ptr
  %ca_r = call i32 @wam_term_cmp(%WamState* %vm, %Value %ca_a1, %Value %ca_a2)
  %ca_nonzero = icmp ne i32 %ca_r, 0
  br i1 %ca_nonzero, label %cmp_args_ret, label %cmp_args_continue
cmp_args_ret:
  ret i32 %ca_r
cmp_args_continue:
  %ca_i_next = add i32 %ca_i, 1
  br label %cmp_args_loop
cmp_args_equal:
  ret i32 0
}'.

%% compile_ssp_emit_segment_to_llvm(-IR)
%
%  M68: emit @ssp_emit_segment(vm, src, seg_start, seg_end, pad, arr, idx)
%  -- strips pad chars from both ends of src[seg_start..seg_end), interns
%  the resulting substring as an Atom, and stores the Value at arr[idx].
%  Used by builtin_split_string to emit each segment.
compile_ssp_emit_segment_to_llvm(Code) :-
    Code = '; M68: split_string segment emitter -- strip pad chars from both ends,
; intern the resulting substring, and store the Atom Value into arr[idx].
define void @ssp_emit_segment(%WamState* %vm, i8* %src, i64 %seg_start, i64 %seg_end, i8* %pad, %Value* %arr, i32 %arr_idx) {
entry:
  br label %trim_left
trim_left:
  %tl_s = phi i64 [ %seg_start, %entry ], [ %tl_s1, %tl_skip ]
  %tl_done = icmp uge i64 %tl_s, %seg_end
  br i1 %tl_done, label %trim_right_init, label %tl_check
tl_check:
  %tl_bp = getelementptr i8, i8* %src, i64 %tl_s
  %tl_b = load i8, i8* %tl_bp
  ; Is %tl_b in %pad?
  br label %tl_pad_loop
tl_pad_loop:
  %tl_pp = phi i8* [ %pad, %tl_check ], [ %tl_pp_n, %tl_pad_step ]
  %tl_pc = load i8, i8* %tl_pp
  %tl_pc_zero = icmp eq i8 %tl_pc, 0
  br i1 %tl_pc_zero, label %trim_right_init, label %tl_pad_check
tl_pad_check:
  %tl_eq = icmp eq i8 %tl_b, %tl_pc
  br i1 %tl_eq, label %tl_skip, label %tl_pad_step
tl_pad_step:
  %tl_pp_n = getelementptr i8, i8* %tl_pp, i32 1
  br label %tl_pad_loop
tl_skip:
  %tl_s1 = add i64 %tl_s, 1
  br label %trim_left
trim_right_init:
  %tri_start = phi i64 [ %tl_s, %trim_left ], [ %tl_s, %tl_pad_loop ]
  br label %trim_right
trim_right:
  %tr_e = phi i64 [ %seg_end, %trim_right_init ], [ %tr_e1, %tr_skip ]
  %tr_some = icmp ugt i64 %tr_e, %tri_start
  br i1 %tr_some, label %tr_check, label %do_intern
tr_check:
  %tr_e1 = sub i64 %tr_e, 1
  %tr_bp = getelementptr i8, i8* %src, i64 %tr_e1
  %tr_b = load i8, i8* %tr_bp
  br label %tr_pad_loop
tr_pad_loop:
  %tr_pp = phi i8* [ %pad, %tr_check ], [ %tr_pp_n, %tr_pad_step ]
  %tr_pc = load i8, i8* %tr_pp
  %tr_pc_zero = icmp eq i8 %tr_pc, 0
  br i1 %tr_pc_zero, label %do_intern, label %tr_pad_check
tr_pad_check:
  %tr_eq = icmp eq i8 %tr_b, %tr_pc
  br i1 %tr_eq, label %tr_skip, label %tr_pad_step
tr_pad_step:
  %tr_pp_n = getelementptr i8, i8* %tr_pp, i32 1
  br label %tr_pad_loop
tr_skip:
  br label %trim_right
do_intern:
  %final_end = phi i64 [ %tr_e, %trim_right ], [ %tr_e, %tr_pad_loop ]
  %final_len = sub i64 %final_end, %tri_start
  %final_ptr = getelementptr i8, i8* %src, i64 %tri_start
  %final_id = call i64 @wam_intern_atom(i8* %final_ptr, i64 %final_len)
  %atom_v0 = insertvalue %Value undef, i32 0, 0
  %atom_v = insertvalue %Value %atom_v0, i64 %final_id, 1
  %slot = getelementptr %Value, %Value* %arr, i32 %arr_idx
  store %Value %atom_v, %Value* %slot
  ret void
}'.

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3 into complete runtime
% ============================================================================

%% compile_wam_runtime_to_llvm(+Options, -LLVMCode)
%  Generates the combined step function + all helper functions.
compile_wam_runtime_to_llvm(Options, LLVMCode) :-
    compile_step_wam_to_llvm(Options, StepCode),
    compile_wam_helpers_to_llvm(Options, HelpersCode),
    atomic_list_concat([StepCode, '\n\n', HelpersCode], LLVMCode).

% ============================================================================
% PHASE 4: WAM instructions → LLVM struct literals
% ============================================================================

%% wam_instruction_to_llvm_literal(+WamInstr, -LLVMLiteral)
%  Converts a WAM instruction term to an LLVM %Instruction struct literal.

wam_instruction_to_llvm_literal(get_constant(C, Ai), Lit) :-
    llvm_pack_value(C, PackedVal),
    reg_name_to_index(Ai, RegIdx),
    llvm_constant_tag(C, Tag),
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), '{ i32 0, i64 ~w, i64 ~w }', [PackedVal, Op2]).
wam_instruction_to_llvm_literal(get_variable(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 1, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(get_value(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 2, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(get_structure(F, Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    atom_string(F, FStr),
    split_functor_arity(FStr, NameStr, Arity),
    string_length(NameStr, NameLen),
    NameLenPlus1 is NameLen + 1,
    Op2 is (Arity << 16) \/ AiIdx,
    sanitize_functor_for_llvm(NameStr, SaneName),
    format(atom(Lit),
        '{ i32 3, i64 ptrtoint ([~w x i8]* @.fn_~w to i64), i64 ~w } ; get_structure ~w',
        [NameLenPlus1, SaneName, Op2, F]),
    register_functor_string(NameStr).
wam_instruction_to_llvm_literal(get_list(Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 4, i64 ~w, i64 0 }', [AiIdx]).
wam_instruction_to_llvm_literal(unify_variable(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 5, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(unify_value(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 6, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(unify_constant(C), Lit) :-
    llvm_pack_value(C, PackedVal),
    llvm_constant_tag(C, Tag),
    Op2 is Tag << 16,
    format(atom(Lit), '{ i32 7, i64 ~w, i64 ~w }', [PackedVal, Op2]).

wam_instruction_to_llvm_literal(put_constant(C, Ai), Lit) :-
    llvm_pack_value(C, PackedVal),
    reg_name_to_index(Ai, RegIdx),
    format(atom(Lit), '{ i32 8, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
wam_instruction_to_llvm_literal(put_variable(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 9, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(put_value(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 10, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(put_structure(F, Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 11, i64 0, i64 ~w } ; put_structure ~w', [AiIdx, F]).
wam_instruction_to_llvm_literal(put_list(Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 12, i64 ~w, i64 0 }', [AiIdx]).
wam_instruction_to_llvm_literal(set_variable(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 13, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(set_value(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 14, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(set_constant(C), Lit) :-
    llvm_pack_value(C, PackedVal),
    format(atom(Lit), '{ i32 15, i64 ~w, i64 0 }', [PackedVal]).

wam_instruction_to_llvm_literal(allocate, '{ i32 16, i64 0, i64 0 }').
wam_instruction_to_llvm_literal(deallocate, '{ i32 17, i64 0, i64 0 }').
wam_instruction_to_llvm_literal(proceed, '{ i32 20, i64 0, i64 0 }').
wam_instruction_to_llvm_literal(builtin_call(Op, N), Lit) :-
    builtin_op_to_id(Op, OpId),
    format(atom(Lit), '{ i32 21, i64 ~w, i64 ~w } ; builtin_call ~w', [OpId, N, Op]).
wam_instruction_to_llvm_literal(trust_me, '{ i32 24, i64 0, i64 0 }').

% Label-referencing instructions: the /2 form cannot resolve labels.
% Callers must use wam_instruction_to_llvm_literal/3 with a LabelMap,
% or the text-parser path (wam_line_to_llvm_literal_resolved/3).
wam_instruction_to_llvm_literal(call(P, _N), _) :-
    throw(error(label_resolution_required(call, P),
          'Use wam_instruction_to_llvm_literal/3 with a LabelMap for call/execute/try_me_else/retry_me_else')).
wam_instruction_to_llvm_literal(execute(P), _) :-
    throw(error(label_resolution_required(execute, P),
          'Use wam_instruction_to_llvm_literal/3 with a LabelMap')).
wam_instruction_to_llvm_literal(try_me_else(L), _) :-
    throw(error(label_resolution_required(try_me_else, L),
          'Use wam_instruction_to_llvm_literal/3 with a LabelMap')).
wam_instruction_to_llvm_literal(retry_me_else(L), _) :-
    throw(error(label_resolution_required(retry_me_else, L),
          'Use wam_instruction_to_llvm_literal/3 with a LabelMap')).

wam_meta_call_label(Label) :-
    wam_meta_call_total_arity(Label, _).

%% wam_special_call_label(+Label, -Op1Sentinel, -Arity) is semidet.
%  A handful of control/db predicates have no compiled label; they lower to a
%  call/execute whose op1 is a negative sentinel the runtime routes to a
%  dedicated handler (rather than being treated as an unknown static label,
%  which would warn / default to index 0). See PLAWK_DYNAMIC_DB.md.
%    retract/1 -> -3 (nondet retract iterator)
%    catch/3   -> -5 (push a catch frame, meta-call Goal)
%    throw/1   -> -6 (unwind to the nearest matching catch frame)
wam_special_call_label(Label, Sentinel, Arity) :-
    ( atom(Label) -> atom_string(Label, S) ; S = Label ),
    wam_special_call_sentinel_(S, Sentinel, Arity).

wam_special_call_sentinel_("retract/1", -3, 1).
wam_special_call_sentinel_("catch/3",   -5, 3).
wam_special_call_sentinel_("throw/1",   -6, 1).

%% wam_special_call_pred(+Pred, +Arity) is semidet.
%  Pred/Arity names a runtime-handled special call (retract/catch/throw) that
%  must not be pulled into a project as a compilable dependency.
wam_special_call_pred(Pred, Arity) :-
    format(atom(PA), '~w/~w', [Pred, Arity]),
    wam_special_call_label(PA, _, _).

wam_meta_call_total_arity(Label, Arity) :-
    (   atom(Label)
    ->  atom_string(Label, LabelString)
    ;   string(Label)
    ->  LabelString = Label
    ),
    sub_string(LabelString, 0, 5, _, "call/"),
    sub_string(LabelString, 5, _, 0, ArityString),
    number_string(Arity, ArityString),
    Arity >= 1.

%% wam_instruction_to_llvm_literal(+WamInstr, +LabelMap, -LLVMLiteral)
%  Label-aware variant. LabelMap is a list of LabelName-Index pairs.
%  NOTE: trailing `; comment` was removed because LLVM treats `;` as a
%  line comment to end-of-line, which eats the comma separator in array
%  constants and produces a parser error.
wam_instruction_to_llvm_literal(call(P, N), LabelMap, Lit) :- !,
    (   wam_special_call_label(P, LabelIdx, _)
    ->  true
    ;   wam_meta_call_label(P)
    ->  LabelIdx = -1
    ;   lookup_label_index(P, LabelMap, LabelIdx)
    ),
    format(atom(Lit), '{ i32 18, i64 ~w, i64 ~w }', [LabelIdx, N]).
wam_instruction_to_llvm_literal(execute(P), LabelMap, Lit) :- !,
    (   wam_special_call_label(P, LabelIdx, Arity)
    ->  true
    ;   wam_meta_call_label(P)
    ->  LabelIdx = -1,
        wam_meta_call_total_arity(P, Arity)
    ;   lookup_label_index(P, LabelMap, LabelIdx),
        Arity = 0
    ),
    format(atom(Lit), '{ i32 19, i64 ~w, i64 ~w }', [LabelIdx, Arity]).
wam_instruction_to_llvm_literal(try_me_else(Label), LabelMap, Lit) :- !,
    lookup_label_index(Label, LabelMap, LabelIdx),
    format(atom(Lit), '{ i32 22, i64 ~w, i64 0 }', [LabelIdx]).
wam_instruction_to_llvm_literal(retry_me_else(Label), LabelMap, Lit) :- !,
    lookup_label_index(Label, LabelMap, LabelIdx),
    format(atom(Lit), '{ i32 23, i64 ~w, i64 0 }', [LabelIdx]).
% Non-label instructions delegate to the /2 form
wam_instruction_to_llvm_literal(Instr, _LabelMap, Lit) :-
    wam_instruction_to_llvm_literal(Instr, Lit).

% Switch instructions are deferred: they reference side-table globals that
% the predicate compiler emits in a second pass. See compile_wam_predicate_to_llvm.
wam_instruction_to_llvm_literal(switch_on_constant(Entries), switch_deferred(constant, Entries)).
wam_instruction_to_llvm_literal(switch_on_structure(_Entries),
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough
wam_instruction_to_llvm_literal(switch_on_constant_a2(_Entries),
    '%Instruction { i32 27, i64 0, i64 0 }').  % nop fallthrough

% Label pseudo-instruction
wam_instruction_to_llvm_literal(label(L), Lit) :-
    format(atom(Lit), '; label: ~w', [L]).

% Fallback
wam_instruction_to_llvm_literal(Instr, Lit) :-
    format(atom(Lit), '; TODO: ~w', [Instr]).

% --- Atom table (string interning) ---
% Assigns unique sequential integer IDs to atoms. Two atoms with the
% same name always get the same ID; different names always get different IDs.
% This avoids hash collisions that would cause silent correctness bugs.

:- dynamic functor_string_global/2. % functor_string_global(NameStr, LenPlus1)

%% emit_functor_string_globals(-IR)
%  Emits LLVM global constants for functor names collected during WAM
%  compilation (by put_structure instructions). Each functor ``name'' becomes
%  @.fn_name = private constant [N x i8] c"name\00".
%
%  M85: bytes that interact with the trailing ``\00'' null terminator
%  in LLVM IR string syntax (``\'' and ``"'') are escaped to ``\NN''
%  per-byte hex. Other bytes are left raw, since string_codes /
%  maplist / append over every character of every functor string adds
%  up to a lot of intermediate-list allocation when the same emit
%  runs hundreds of times in one swipl session (M88 OOM tripped over
%  this at ~580 modules in).
emit_functor_string_globals(IR) :-
    findall(GlobalDef,
        ( functor_string_global(NameStr, Len),
          sanitize_functor_for_llvm(NameStr, SaneName),
          llvm_ir_string_content(NameStr, SafeContent),
          format(atom(GlobalDef), '@.fn_~w = private constant [~w x i8] c"~w\\00"',
              [SaneName, Len, SafeContent])
        ),
        Defs),
    ( Defs == []
    -> IR = ''
    ;  atomic_list_concat(Defs, '\n', IR)
    ).

%% llvm_ir_string_content(+Str, -Safe)
%  Wrap Str so it can be dropped between c"..." quotes in LLVM IR.
%  Fast path: if the input is pure printable ASCII with no ``\'' or
%  ``"'', it''s already safe -- return as-is. Slow path: escape
%  problem bytes (and any non-printable ones) to ``\NN'' hex.
llvm_ir_string_content(Str, Safe) :-
    ( name_needs_llvm_escape(Str)
    -> string_codes(Str, Codes),
       phrase(escape_ir_bytes(Codes), SafeCodes),
       string_codes(Safe, SafeCodes)
    ;  Safe = Str
    ).

name_needs_llvm_escape(Str) :-
    string_codes(Str, Codes),
    member(C, Codes),
    ( C < 32 ; C > 126 ; C =:= 0'\\ ; C =:= 0'" ),
    !.

escape_ir_bytes([]) --> [].
escape_ir_bytes([C|Cs]) -->
    ( { C >= 32, C =< 126, C =\= 0'\\, C =\= 0'" }
    -> [C]
    ;  { Hi is (C >> 4) /\ 0xF,
         Lo is C /\ 0xF,
         hex_digit(Hi, H),
         hex_digit(Lo, L) },
       [0'\\, H, L]
    ),
    escape_ir_bytes(Cs).

%% emit_atom_string_globals(-IR)
%
%  M12: emit per-atom string globals and a runtime lookup table so the
%  format/2 builtin (and any other runtime code that needs to print an
%  atom by id) can resolve an interned atom id back to a printable
%  C string.
%
%  Layout:
%    @.atom_str_<id> = private constant [N x i8] c"<name>\00"   (per atom)
%    @wam_atom_strings = private constant [Max+1 x i8*] [ ... ]  (lookup)
%    @wam_atom_string_count = private constant i32 Max+1        (bounds)
%
%  Atom ids start at 1 (id 0 is reserved); slot 0 of the lookup array
%  is the null pointer so a guarded lookup of "atom 0" returns null.
emit_atom_string_globals(IR) :-
    findall(Id-Name, atom_table_entry(Name, Id), Pairs0),
    ( Pairs0 == []
    -> % Emit a minimal stub so format/2 and any other runtime code that
       % references @wam_atom_to_string still links. One null slot is
       % enough -- atom-id lookups will all return null and the printer
       % silently skips them.
       IR = '@wam_atom_strings = private constant [1 x i8*] [i8* null]
@wam_atom_string_count = private constant i32 1
@wam_char_to_atom_id = private constant [128 x i64] zeroinitializer
@wam_atom_dyn_ptr   = global i8** null
@wam_atom_dyn_count = global i32 0
@wam_atom_dyn_cap   = global i32 0
@wam_transient_line_buf = global i8* null
@wam_transient_line_cap = global i64 0
@wam_rs_byte = global i8 10
@.wam_rs_default = private constant [2 x i8] c"\\0A\\00"
@wam_rs_ptr = global i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.wam_rs_default, i64 0, i64 0)
@wam_rs_len = global i64 1
@wam_rs_mode = global i8 0
@wam_rs_regex_cache = internal global i8* null
@.wam_rt_empty = private constant [1 x i8] c"\\00"
@wam_rt_ptr = global i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.wam_rt_empty, i64 0, i64 0)
@wam_rt_len = global i64 0
@wam_rt_buf = internal global i8* null
@wam_rt_cap = internal global i64 0
@wam_rt_suppress = internal global i1 false
@.wam_rs_regex_error = private constant [38 x i8] c"plawk: invalid RS regular expression\\0A\\00"
@.wam_subsep_default = private constant [2 x i8] c"\\1C\\00"
@wam_subsep_ptr = global i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.wam_subsep_default, i64 0, i64 0)
@wam_subsep_len = global i64 1

define i8* @wam_atom_to_string(i64 %id) {
  ret i8* null
}

define i64 @wam_char_to_atom(i64 %ch) {
  ret i64 0
}

define i1 @wam_str_eq_n(i8* %str, i64 %len, i8* %cand) {
  ret i1 false
}

define i64 @wam_intern_atom(i8* %str, i64 %len) {
  ret i64 0
}

; Tier-2 payload marshaling: copy Len bytes into the shared transient
; buffer (grown geometrically, NUL-terminated) and return the transient
; atom id (2^62). Constant memory: one buffer for the process, valid
; until the next transient write/read. At most one live payload at a
; time -- callers pass at most one blob argument per foreign call.
define i64 @wam_transient_atom_from_bytes(i8* %src, i64 %len) {
entry:
  %need = add i64 %len, 1
  %cap = load i64, i64* @wam_transient_line_cap
  %fits = icmp ule i64 %need, %cap
  br i1 %fits, label %copy, label %grow
grow:
  %small = icmp ult i64 %need, 4096
  %newcap = select i1 %small, i64 4096, i64 %need
  %old = load i8*, i8** @wam_transient_line_buf
  %newbuf = call i8* @realloc(i8* %old, i64 %newcap)
  %newnull = icmp eq i8* %newbuf, null
  br i1 %newnull, label %fail, label %store_new
store_new:
  store i8* %newbuf, i8** @wam_transient_line_buf
  store i64 %newcap, i64* @wam_transient_line_cap
  br label %copy
copy:
  %buf = load i8*, i8** @wam_transient_line_buf
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* %src, i64 %len, i1 false)
  %endp = getelementptr i8, i8* %buf, i64 %len
  store i8 0, i8* %endp
  ret i64 4611686018427387904
fail:
  ret i64 -1
}
'
    ;  sort(Pairs0, Pairs),   % sort by id
       last(Pairs, MaxId-_),
       ArrSize is MaxId + 1,
       % Per-atom string globals.
       findall(StrGlobal,
           ( member(Id-Name, Pairs),
             escape_llvm_string(Name, Escaped, EscapedByteLen),
             ArrLen is EscapedByteLen + 1,
             format(atom(StrGlobal),
                 '@.atom_str_~w = private constant [~w x i8] c"~w\\00"',
                 [Id, ArrLen, Escaped])
           ),
           StrGlobals),
       % Lookup table: slot 0 = null, slot Id = ptr to @.atom_str_<Id>.
       findall(SlotEntry,
           ( numlist(0, MaxId, Idxs),
             member(Idx, Idxs),
             ( Idx == 0
             -> SlotEntry = 'i8* null'
             ;  member(Idx-IdxName, Pairs)
             -> escape_llvm_string(IdxName, _, IdxByteLen),
                IdxArrLen is IdxByteLen + 1,
                format(atom(SlotEntry),
                    'i8* getelementptr ([~w x i8], [~w x i8]* @.atom_str_~w, i32 0, i32 0)',
                    [IdxArrLen, IdxArrLen, Idx])
             ;  SlotEntry = 'i8* null'
             )
           ),
           Slots),
       atomic_list_concat(Slots, ',\n  ', SlotsStr),
       % M26: char -> atom_id reverse lookup for the printable ASCII
       % range. Indices 0..31 and 127 are 0 (no atom); 32..126 hold
       % the interned id from the eager pass at write_wam_llvm_project
       % entry. char_code/2 reverse mode and atom_chars/2 forward use
       % this to map a byte to its single-char atom id without a
       % runtime string compare.
       findall(CharSlot,
           ( numlist(0, 127, ChIdxs),
             member(Ch, ChIdxs),
             ( Ch >= 32, Ch =< 126,
               char_code(ChAtom, Ch),
               atom_table_entry(ChAtom, ChId)
             -> format(atom(CharSlot), 'i64 ~w', [ChId])
             ;  CharSlot = 'i64 0'
             )
           ),
           CharSlots),
       atomic_list_concat(CharSlots, ', ', CharSlotsStr),
       format(atom(LookupTable),
'@wam_atom_strings = private constant [~w x i8*] [
  ~w
]
@wam_atom_string_count = private constant i32 ~w
@wam_char_to_atom_id = private constant [128 x i64] [~w]

; M29: runtime-interned atom table. Atoms with id >= the static
; count above live here, indexed by (id - static_count). The table
; is malloc''d on first intern and grown by realloc; entries are
; malloc''d C strings owned by the table.
@wam_atom_dyn_ptr   = global i8** null
@wam_atom_dyn_count = global i32 0
@wam_atom_dyn_cap   = global i32 0

; FNV-1a hash index over the atom tables: maps string -> atom id so
; wam_intern_atom is O(1) amortized instead of a linear scan over
; every atom (previously O(table size) per intern, which made streams
; that intern many unique lines through read_line quadratic). Slots
; hold atom_id + 1; 0 marks an empty slot. Built lazily on the first
; intern, seeded with every existing static and dynamic atom, and
; grown at 50 percent load by rehashing all ids.
@wam_atom_hash_slots = global i64* null
@wam_atom_hash_cap   = global i64 0
@wam_atom_hash_count = global i64 0

; Shared buffer behind the transient line atom (id 2^62). Grown
; geometrically, never freed; contents are valid until the next
; transient read.
@wam_transient_line_buf = global i8* null
@wam_transient_line_cap = global i64 0
@wam_rs_byte = global i8 10
@.wam_rs_default = private constant [2 x i8] c"\\0A\\00"
@wam_rs_ptr = global i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.wam_rs_default, i64 0, i64 0)
@wam_rs_len = global i64 1
@wam_rs_mode = global i8 0
@wam_rs_regex_cache = internal global i8* null
@.wam_rt_empty = private constant [1 x i8] c"\\00"
@wam_rt_ptr = global i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.wam_rt_empty, i64 0, i64 0)
@wam_rt_len = global i64 0
@wam_rt_buf = internal global i8* null
@wam_rt_cap = internal global i64 0
@wam_rt_suppress = internal global i1 false
@.wam_rs_regex_error = private constant [38 x i8] c"plawk: invalid RS regular expression\\0A\\00"
@.wam_subsep_default = private constant [2 x i8] c"\\1C\\00"
@wam_subsep_ptr = global i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.wam_subsep_default, i64 0, i64 0)
@wam_subsep_len = global i64 1

define i8* @wam_atom_to_string(i64 %id) {
entry:
  ; Transient line atom: id 2^62 resolves to the shared line buffer
  ; owned by @wam_stream_read_line_transient_value. The buffer mutates
  ; on every read, so this id must never enter the intern hash or the
  ; dynamic table; it exists so per-record consumers (field helpers,
  ; regex, printing) work on the current record without interning it.
  %is_transient = icmp eq i64 %id, 4611686018427387904
  br i1 %is_transient, label %transient, label %check_static
transient:
  %tbuf = load i8*, i8** @wam_transient_line_buf
  ret i8* %tbuf
check_static:
  %cnt = load i32, i32* @wam_atom_string_count
  %cnt64 = zext i32 %cnt to i64
  %in_static = icmp ult i64 %id, %cnt64
  br i1 %in_static, label %lookup_static, label %check_dyn
lookup_static:
  %slot_ptr = getelementptr [~w x i8*], [~w x i8*]* @wam_atom_strings, i32 0, i64 %id
  %s = load i8*, i8** %slot_ptr
  ret i8* %s
check_dyn:
  ; M29: id >= static_count, try dynamic table.
  %dyn_idx_64 = sub i64 %id, %cnt64
  %dyn_cnt = load i32, i32* @wam_atom_dyn_count
  %dyn_cnt_64 = zext i32 %dyn_cnt to i64
  %in_dyn = icmp ult i64 %dyn_idx_64, %dyn_cnt_64
  br i1 %in_dyn, label %lookup_dyn, label %nullret
lookup_dyn:
  %dyn_ptr = load i8**, i8*** @wam_atom_dyn_ptr
  %dyn_slot_ptr = getelementptr i8*, i8** %dyn_ptr, i64 %dyn_idx_64
  %dyn_s = load i8*, i8** %dyn_slot_ptr
  ret i8* %dyn_s
nullret:
  ret i8* null
}

; M29: byte -> single-char atom id lookup. Returns 0 (sentinel "no
; such atom") for non-printable bytes or out-of-range indices.
define i64 @wam_char_to_atom(i64 %ch) {
entry:
  %in_rng = icmp ult i64 %ch, 128
  br i1 %in_rng, label %get, label %zret
get:
  %ch_slot = getelementptr [128 x i64], [128 x i64]* @wam_char_to_atom_id, i32 0, i64 %ch
  %id = load i64, i64* %ch_slot
  ret i64 %id
zret:
  ret i64 0
}


; Tier-2 payload marshaling: copy Len bytes into the shared transient
; buffer (grown geometrically, NUL-terminated) and return the transient
; atom id (2^62). Constant memory: one buffer for the process, valid
; until the next transient write/read. At most one live payload at a
; time -- callers pass at most one blob argument per foreign call.
define i64 @wam_transient_atom_from_bytes(i8* %src, i64 %len) {
entry:
  %need = add i64 %len, 1
  %cap = load i64, i64* @wam_transient_line_cap
  %fits = icmp ule i64 %need, %cap
  br i1 %fits, label %copy, label %grow
grow:
  %small = icmp ult i64 %need, 4096
  %newcap = select i1 %small, i64 4096, i64 %need
  %old = load i8*, i8** @wam_transient_line_buf
  %newbuf = call i8* @realloc(i8* %old, i64 %newcap)
  %newnull = icmp eq i8* %newbuf, null
  br i1 %newnull, label %fail, label %store_new
store_new:
  store i8* %newbuf, i8** @wam_transient_line_buf
  store i64 %newcap, i64* @wam_transient_line_cap
  br label %copy
copy:
  %buf = load i8*, i8** @wam_transient_line_buf
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* %src, i64 %len, i1 false)
  %endp = getelementptr i8, i8* %buf, i64 %len
  store i8 0, i8* %endp
  ret i64 4611686018427387904
fail:
  ret i64 -1
}

; M29: case-sensitive string equality of (str, len) against a
; null-terminated C string. Returns true iff str[0..len) matches
; cand[0..len) AND cand[len] == 0.
define i1 @wam_str_eq_n(i8* %str, i64 %len, i8* %cand) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %step ]
  %done = icmp uge i64 %i, %len
  br i1 %done, label %check_term, label %step
step:
  %sp = getelementptr i8, i8* %str, i64 %i
  %cp = getelementptr i8, i8* %cand, i64 %i
  %sc = load i8, i8* %sp
  %cc = load i8, i8* %cp
  %eq = icmp eq i8 %sc, %cc
  %i1 = add i64 %i, 1
  br i1 %eq, label %loop, label %neq
neq:
  ret i1 false
check_term:
  %tp = getelementptr i8, i8* %cand, i64 %len
  %tc = load i8, i8* %tp
  %term = icmp eq i8 %tc, 0
  ret i1 %term
}

; FNV-1a over the raw bytes. The 64-bit offset basis
; 14695981039346656037 is written in two-s complement signed form
; because LLVM textual IR rejects unsigned constants above 2^63 - 1.
define i64 @wam_atom_hash_value(i8* %str, i64 %len) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %step ]
  %h = phi i64 [ -3750763034362895579, %entry ], [ %h1, %step ]
  %done = icmp uge i64 %i, %len
  br i1 %done, label %ret, label %step
step:
  %bp = getelementptr i8, i8* %str, i64 %i
  %b = load i8, i8* %bp
  %b64 = zext i8 %b to i64
  %hx = xor i64 %h, %b64
  %h1 = mul i64 %hx, 1099511628211
  %i1 = add i64 %i, 1
  br label %loop
ret:
  ret i64 %h
}

; Probe for (str, len): returns the index of the matching occupied
; slot, or of the empty slot where the string belongs. Requires
; cap > 0 with load factor below 1, which the grow-at-50-percent rule
; guarantees, so the probe always terminates.
define i64 @wam_atom_hash_find_slot(i8* %str, i64 %len) {
entry:
  %cap = load i64, i64* @wam_atom_hash_cap
  %slots = load i64*, i64** @wam_atom_hash_slots
  %h = call i64 @wam_atom_hash_value(i8* %str, i64 %len)
  %start = urem i64 %h, %cap
  br label %probe
probe:
  %idx = phi i64 [ %start, %entry ], [ %next, %miss ]
  %slot_p = getelementptr i64, i64* %slots, i64 %idx
  %v = load i64, i64* %slot_p
  %empty = icmp eq i64 %v, 0
  br i1 %empty, label %found, label %check
check:
  %cand_id = sub i64 %v, 1
  %cand_s = call i8* @wam_atom_to_string(i64 %cand_id)
  %cand_null = icmp eq i8* %cand_s, null
  br i1 %cand_null, label %miss, label %cmp
cmp:
  %eq = call i1 @wam_str_eq_n(i8* %str, i64 %len, i8* %cand_s)
  br i1 %eq, label %found, label %miss
miss:
  %idx1 = add i64 %idx, 1
  %wrap = icmp uge i64 %idx1, %cap
  %next = select i1 %wrap, i64 0, i64 %idx1
  br label %probe
found:
  ret i64 %idx
}

; Insert an existing atom id into the hash by its string. When the
; same string appears under two ids (duplicate static entries), the
; first id inserted wins, mirroring the old first-match linear scan.
define i1 @wam_atom_hash_insert_id(i64 %id) {
entry:
  %s = call i8* @wam_atom_to_string(i64 %id)
  %s_null = icmp eq i8* %s, null
  br i1 %s_null, label %skip, label %measure
measure:
  br label %len_loop
len_loop:
  %li = phi i64 [ 0, %measure ], [ %li1, %len_step ]
  %lp = getelementptr i8, i8* %s, i64 %li
  %lb = load i8, i8* %lp
  %lz = icmp eq i8 %lb, 0
  br i1 %lz, label %have_len, label %len_step
len_step:
  %li1 = add i64 %li, 1
  br label %len_loop
have_len:
  %slot_idx = call i64 @wam_atom_hash_find_slot(i8* %s, i64 %li)
  %slots = load i64*, i64** @wam_atom_hash_slots
  %slot_p = getelementptr i64, i64* %slots, i64 %slot_idx
  %v = load i64, i64* %slot_p
  %empty = icmp eq i64 %v, 0
  br i1 %empty, label %fill, label %skip
fill:
  %idp1 = add i64 %id, 1
  store i64 %idp1, i64* %slot_p
  %cnt = load i64, i64* @wam_atom_hash_count
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, i64* @wam_atom_hash_count
  ret i1 true
skip:
  ret i1 false
}

; Allocate a zeroed slot array of new_cap entries and reinsert every
; atom id currently in the static + dynamic tables. Used for both
; lazy init and growth. Fails only if malloc fails, in which case the
; previous table (if any) is left untouched and still valid.
define i1 @wam_atom_hash_rebuild(i64 %new_cap) {
entry:
  %bytes = shl i64 %new_cap, 3
  %mem = call i8* @malloc(i64 %bytes)
  %mem_null = icmp eq i8* %mem, null
  br i1 %mem_null, label %fail, label %zero_init
zero_init:
  %new_slots = bitcast i8* %mem to i64*
  br label %zloop
zloop:
  %zi = phi i64 [ 0, %zero_init ], [ %zi1, %zstep ]
  %zdone = icmp uge i64 %zi, %new_cap
  br i1 %zdone, label %swap, label %zstep
zstep:
  %zp = getelementptr i64, i64* %new_slots, i64 %zi
  store i64 0, i64* %zp
  %zi1 = add i64 %zi, 1
  br label %zloop
swap:
  %old = load i64*, i64** @wam_atom_hash_slots
  store i64* %new_slots, i64** @wam_atom_hash_slots
  store i64 %new_cap, i64* @wam_atom_hash_cap
  store i64 0, i64* @wam_atom_hash_count
  %old_null = icmp eq i64* %old, null
  br i1 %old_null, label %reinsert_init, label %free_old
free_old:
  %old_i8 = bitcast i64* %old to i8*
  call void @free(i8* %old_i8)
  br label %reinsert_init
reinsert_init:
  ; Total ids = static count + dynamic count. Id 0 and any null static
  ; slots resolve to null strings and are skipped by insert.
  %st_cnt = load i32, i32* @wam_atom_string_count
  %st_64 = zext i32 %st_cnt to i64
  %dy_cnt = load i32, i32* @wam_atom_dyn_count
  %dy_64 = zext i32 %dy_cnt to i64
  %total = add i64 %st_64, %dy_64
  br label %riloop
riloop:
  %ri = phi i64 [ 0, %reinsert_init ], [ %ri1, %ristep ]
  %ridone = icmp uge i64 %ri, %total
  br i1 %ridone, label %ok, label %ristep
ristep:
  %ins_ignored = call i1 @wam_atom_hash_insert_id(i64 %ri)
  %ri1 = add i64 %ri, 1
  br label %riloop
ok:
  ret i1 true
fail:
  ret i1 false
}

; Intern an atom by (str, len) through the hash index: hit returns the
; existing id in O(1) amortized; miss copies the string, appends it to
; the dynamic table (growing via realloc as needed), and records the
; new id in the hash.
define i64 @wam_intern_atom(i8* %str, i64 %len) {
entry:
  %cap0 = load i64, i64* @wam_atom_hash_cap
  %uninit = icmp eq i64 %cap0, 0
  br i1 %uninit, label %init_hash, label %lookup

init_hash:
  ; First intern: size the table to keep the static atoms below
  ; 50 percent load, minimum 4096 slots.
  %st_cnt0 = load i32, i32* @wam_atom_string_count
  %st0_64 = zext i32 %st_cnt0 to i64
  %seed_need = shl i64 %st0_64, 2
  %small = icmp ult i64 %seed_need, 4096
  %init_cap = select i1 %small, i64 4096, i64 %seed_need
  %init_ok = call i1 @wam_atom_hash_rebuild(i64 %init_cap)
  br i1 %init_ok, label %lookup, label %intern_fail

lookup:
  %slot_idx = call i64 @wam_atom_hash_find_slot(i8* %str, i64 %len)
  %slots = load i64*, i64** @wam_atom_hash_slots
  %slot_p = getelementptr i64, i64* %slots, i64 %slot_idx
  %v = load i64, i64* %slot_p
  %hit = icmp ne i64 %v, 0
  br i1 %hit, label %ret_hit, label %do_intern

ret_hit:
  %hit_id = sub i64 %v, 1
  ret i64 %hit_id

do_intern:
  %st_cnt = load i32, i32* @wam_atom_string_count
  %st_cnt_64 = zext i32 %st_cnt to i64
  %dyn_cnt = load i32, i32* @wam_atom_dyn_count
  %dyn_cnt_64 = zext i32 %dyn_cnt to i64
  %dyn_ptr_0 = load i8**, i8*** @wam_atom_dyn_ptr
  %dyn_cap = load i32, i32* @wam_atom_dyn_cap
  %need_grow = icmp uge i32 %dyn_cnt, %dyn_cap
  br i1 %need_grow, label %do_grow, label %do_alloc_str

do_grow:
  ; new_cap = max(8, dyn_cap * 2)
  %dyn_cap_64 = zext i32 %dyn_cap to i64
  %doubled = shl i64 %dyn_cap_64, 1
  %is_small = icmp ult i64 %doubled, 8
  %new_cap_64 = select i1 %is_small, i64 8, i64 %doubled
  %new_bytes = shl i64 %new_cap_64, 3   ; i8* is 8 bytes
  %old_ptr_i8 = bitcast i8** %dyn_ptr_0 to i8*
  %new_ptr_i8 = call i8* @realloc(i8* %old_ptr_i8, i64 %new_bytes)
  %new_ptr_null = icmp eq i8* %new_ptr_i8, null
  br i1 %new_ptr_null, label %intern_fail, label %do_grow_ok
intern_fail:
  ret i64 0
do_grow_ok:
  %new_ptr = bitcast i8* %new_ptr_i8 to i8**
  store i8** %new_ptr, i8*** @wam_atom_dyn_ptr
  %new_cap_i32 = trunc i64 %new_cap_64 to i32
  store i32 %new_cap_i32, i32* @wam_atom_dyn_cap
  br label %do_alloc_str

do_alloc_str:
  ; Allocate len+1 bytes for the null-terminated copy.
  %str_bytes = add i64 %len, 1
  %str_buf = call i8* @malloc(i64 %str_bytes)
  %str_buf_null = icmp eq i8* %str_buf, null
  br i1 %str_buf_null, label %intern_fail, label %do_copy
do_copy:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %str_buf, i8* %str, i64 %len, i1 false)
  %term_ptr = getelementptr i8, i8* %str_buf, i64 %len
  store i8 0, i8* %term_ptr
  ; Append str_buf to dynamic table at index dyn_cnt.
  %cur_ptr = load i8**, i8*** @wam_atom_dyn_ptr
  %append_slot = getelementptr i8*, i8** %cur_ptr, i32 %dyn_cnt
  store i8* %str_buf, i8** %append_slot
  %new_cnt = add i32 %dyn_cnt, 1
  store i32 %new_cnt, i32* @wam_atom_dyn_count
  %new_id = add i64 %st_cnt_64, %dyn_cnt_64
  ; Record the new id in the hash, then grow past 50 percent load.
  %idp1 = add i64 %new_id, 1
  store i64 %idp1, i64* %slot_p
  %hcnt = load i64, i64* @wam_atom_hash_count
  %hcnt1 = add i64 %hcnt, 1
  store i64 %hcnt1, i64* @wam_atom_hash_count
  %hcap = load i64, i64* @wam_atom_hash_cap
  %half = lshr i64 %hcap, 1
  %needs_hash_grow = icmp ugt i64 %hcnt1, %half
  br i1 %needs_hash_grow, label %grow_hash, label %ret_new
grow_hash:
  ; On rebuild failure the old table stays valid, only fuller; the
  ; intern itself already succeeded.
  %gcap = shl i64 %hcap, 1
  %grow_ignored = call i1 @wam_atom_hash_rebuild(i64 %gcap)
  br label %ret_new
ret_new:
  ret i64 %new_id
}',
           [ArrSize, SlotsStr, ArrSize, CharSlotsStr, ArrSize, ArrSize]),
       atomic_list_concat(StrGlobals, '\n', StrGlobalsStr),
       atomic_list_concat([StrGlobalsStr, LookupTable], '\n\n', IR)
    ).

%% escape_llvm_string(+Name, -Escaped, -ByteLen)
%
%  Encode an arbitrary atom name as the body of an LLVM `c"..."`
%  string literal. Printable ASCII (0x20..0x7E except `"` and `\\`)
%  passes through; everything else (including spaces, since those
%  print fine, but tildes / newlines / quotes / backslashes that the
%  WAM compiler''s format-string interning produces) emits as `\HH`.
%  ByteLen is the number of underlying bytes in the unescaped name,
%  which is also the length the runtime needs for the array size.
escape_llvm_string(Name, Escaped, ByteLen) :-
    ( atom(Name) -> atom_codes(Name, Codes)
    ; string(Name) -> string_codes(Name, Codes)
    ; Codes = Name
    ),
    length(Codes, ByteLen),
    escape_llvm_codes(Codes, EscCodes),
    string_codes(Escaped, EscCodes).

%% llvm_emit_atom_prefix_guard(+GlobalBase, +ValueIR, +Prefix, +ResultIR, -GlobalIR-CallIR)
%
%  Emit a reusable native prefix guard for deterministic
%  sub_atom(Value, 0, Len, _, Prefix)-style lowering. ValueIR and
%  ResultIR are LLVM SSA expressions such as `%line` and `%is_match`.
%  GlobalIR is top-level module IR; CallIR belongs inside a function body and
%  calls the general @wam_atom_prefix_value runtime helper.
llvm_emit_atom_prefix_guard(GlobalBase, ValueIR, Prefix, ResultIR, GlobalIR-CallIR) :-
    sanitize_functor_for_llvm(GlobalBase, SafeBase),
    escape_llvm_string(Prefix, EscapedPrefix, PrefixLen),
    ArrLen is PrefixLen + 1,
    format(atom(GlobalIR),
        '@.~w = private constant [~w x i8] c"~w\\00"',
        [SafeBase, ArrLen, EscapedPrefix]),
    format(atom(CallIR),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0~n  ~w = call i1 @wam_atom_prefix_value(%Value ~w, i8* %~w_ptr, i64 ~w)',
        [SafeBase, ArrLen, ArrLen, SafeBase,
         ResultIR, ValueIR, SafeBase, PrefixLen]).

%% llvm_emit_atom_field_eq_guard(+GlobalBase, +ValueIR, +FieldIndex, +Expected, +SepCode, +ResultIR, -GlobalIR-CallIR)
%
%  Emit a reusable native field-equality guard for deterministic
%  item_field(Index, Item, Expected)-style lowering over atom-backed text
%  records. FieldIndex is one-based, SepCode is the single-byte field
%  separator, and Expected is emitted as an LLVM constant without allocating
%  substrings in the hot loop.
llvm_emit_atom_field_eq_guard(GlobalBase, ValueIR, FieldIndex, Expected, SepCode, ResultIR, GlobalIR-CallIR) :-
    sanitize_functor_for_llvm(GlobalBase, SafeBase),
    escape_llvm_string(Expected, EscapedExpected, ExpectedLen),
    ArrLen is ExpectedLen + 1,
    format(atom(GlobalIR),
        '@.~w = private constant [~w x i8] c"~w\\00"',
        [SafeBase, ArrLen, EscapedExpected]),
    format(atom(CallIR),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0~n  ~w = call i1 @wam_atom_field_eq_value(%Value ~w, i64 ~w, i8* %~w_ptr, i64 ~w, i8 ~w)',
        [SafeBase, ArrLen, ArrLen, SafeBase,
         ResultIR, ValueIR, FieldIndex, SafeBase, ExpectedLen, SepCode]).

%% llvm_emit_regex_field_match_guard(+GlobalBase, +ValueIR, +FieldIndex, +Regex, +SepCode, +ResultIR, -GlobalIR-CallIR)
%
%  Emit a native POSIX ERE guard over an atom-backed text record.
%  FieldIndex 0 matches the whole record; positive indexes match the
%  projected field slice. The pattern is emitted as a string global
%  plus a per-site cache slot holding the lazily compiled regex_t;
%  matching happens in the @wam_regex_field_match runtime helper. A
%  pattern that fails to compile never matches.
llvm_emit_regex_field_match_guard(GlobalBase, ValueIR, FieldIndex, Regex, SepCode, ResultIR, GlobalIR-CallIR) :-
    integer(FieldIndex),
    FieldIndex >= 0,
    sanitize_functor_for_llvm(GlobalBase, SafeBase),
    escape_llvm_string(Regex, EscapedRegex, RegexLen),
    ArrLen is RegexLen + 1,
    format(atom(GlobalIR),
'@.~w = private constant [~w x i8] c"~w\\00"
@~w_regex_cache = internal global i8* null',
        [SafeBase, ArrLen, EscapedRegex, SafeBase]),
    format(atom(CallIR),
'  %~w_pat_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0
  ~w = call i1 @wam_regex_field_match(%Value ~w, i64 ~w, i8 ~w, i8* %~w_pat_ptr, i8** @~w_regex_cache)',
        [SafeBase, ArrLen, ArrLen, SafeBase,
         ResultIR, ValueIR, FieldIndex, SepCode, SafeBase, SafeBase]).

%% llvm_emit_atom_field_slice(+ValueIR, +FieldIndex, +SepCode, +SliceBase, -CallIR)
%
%  Emit a native field projection over an atom-backed text record. The runtime
%  returns a %WamSlice without allocating a field substring; CallIR defines
%  %SliceBase, %SliceBase_ptr, and %SliceBase_len.
llvm_emit_atom_field_slice(ValueIR, FieldIndex, SepCode, SliceBase, CallIR) :-
    format(atom(CallIR),
        '  %~w = call %WamSlice @wam_atom_field_slice_value(%Value ~w, i64 ~w, i8 ~w)~n  %~w_ptr = extractvalue %WamSlice %~w, 0~n  %~w_len64 = extractvalue %WamSlice %~w, 1~n  %~w_len = trunc i64 %~w_len64 to i32',
        [SliceBase, ValueIR, FieldIndex, SepCode,
         SliceBase, SliceBase,
         SliceBase, SliceBase,
         SliceBase, SliceBase]).

%% llvm_emit_atom_field_count(+ValueIR, +SepCode, +CountBase, -CallIR)
%
%  Emit a native field-count operation over an atom-backed text record. The
%  runtime returns the split_string/4-style field count for the single-byte
%  separator without allocating field substrings.
llvm_emit_atom_field_count(ValueIR, SepCode, CountBase, CallIR) :-
    format(atom(CallIR),
        '  %~w = call i64 @wam_atom_field_count_value(%Value ~w, i8 ~w)',
        [CountBase, ValueIR, SepCode]).

%% llvm_emit_atom_field_length(+ValueIR, +FieldIndex, +SepCode, +LengthBase, -CallIR)
%
%  Emit a native byte-length operation over an atom-backed text record. Field 0
%  measures the whole record; positive fields measure the projected field slice
%  without allocating a field substring.
llvm_emit_atom_field_length(ValueIR, FieldIndex, SepCode, LengthBase, CallIR) :-
    format(atom(CallIR),
        '  %~w = call i64 @wam_atom_field_length_value(%Value ~w, i64 ~w, i8 ~w)',
        [LengthBase, ValueIR, FieldIndex, SepCode]).

%% llvm_emit_atom_field_subslice(+ValueIR, +FieldIndex, +SepCode, +Start, +Len, +SliceBase, -CallIR)
%
%  Emit a native AWK-style substr/3 projection over an atom-backed text record.
%  Start is 1-based and Len is a byte count. Field 0 slices the whole record;
%  positive fields slice the projected field without allocating substrings.
llvm_emit_atom_field_subslice(ValueIR, FieldIndex, SepCode, Start, Len, SliceBase, CallIR) :-
    format(atom(CallIR),
        '  %~w = call %WamSlice @wam_atom_field_subslice_value(%Value ~w, i64 ~w, i8 ~w, i64 ~w, i64 ~w)~n  %~w_ptr = extractvalue %WamSlice %~w, 0~n  %~w_len64 = extractvalue %WamSlice %~w, 1~n  %~w_len = trunc i64 %~w_len64 to i32',
        [SliceBase, ValueIR, FieldIndex, SepCode, Start, Len,
         SliceBase, SliceBase,
         SliceBase, SliceBase,
         SliceBase, SliceBase]).

%% llvm_emit_atom_field_index(+GlobalBase, +ValueIR, +FieldIndex, +Needle, +SepCode, +IndexBase, -GlobalIR-CallIR)
%
%  Emit a native AWK-style index/2 over an atom-backed text record. The runtime
%  returns a one-based byte position or zero when the needle is absent. Field 0
%  searches the whole record; positive fields search the projected field without
%  allocating substrings.
llvm_emit_atom_field_index(GlobalBase, ValueIR, FieldIndex, Needle, SepCode, IndexBase, GlobalIR-CallIR) :-
    sanitize_functor_for_llvm(GlobalBase, SafeGlobalBase),
    escape_llvm_string(Needle, EscapedNeedle, NeedleLen),
    ArrLen is NeedleLen + 1,
    format(atom(GlobalIR),
        '@.~w = private constant [~w x i8] c"~w\\00"',
        [SafeGlobalBase, ArrLen, EscapedNeedle]),
    format(atom(CallIR),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0~n  %~w = call i64 @wam_atom_field_index_value(%Value ~w, i64 ~w, i8 ~w, i8* %~w_ptr, i64 ~w)',
        [SafeGlobalBase, ArrLen, ArrLen, SafeGlobalBase,
         IndexBase, ValueIR, FieldIndex, SepCode, SafeGlobalBase, NeedleLen]).

%% llvm_emit_atom_field_i64_cmp_guard(+ValueIR, +FieldIndex, +OpCode, +Expected, +SepCode, +ResultIR, -CallIR)
%
%  Emit a native numeric field comparison over an atom-backed text record.
%  OpCode is numeric for the generated hot path: 0 ==, 1 !=, 2 <, 3 <=,
%  4 >, and 5 >=. The runtime parses the selected field slice as a strict
%  signed decimal i64 and returns false for missing or non-numeric fields.
llvm_emit_atom_field_i64_cmp_guard(ValueIR, FieldIndex, OpCode, Expected, SepCode, ResultIR, CallIR) :-
    integer(FieldIndex),
    integer(OpCode),
    between(0, 5, OpCode),
    integer(Expected),
    format(atom(CallIR),
        '  ~w = call i1 @wam_atom_field_i64_cmp_value(%Value ~w, i64 ~w, i8 ~w, i64 ~w, i32 ~w)',
        [ResultIR, ValueIR, FieldIndex, SepCode, Expected, OpCode]).

%% llvm_emit_atom_field_strnum_cmp_guard(+ValueIR, +FieldIndex, +OpCode, +Expected, +SepCode, +ResultIR, -CallIR)
%
%  `$N OP int` with awk strnum semantics (@wam_atom_field_strnum_cmp_int): numeric
%  when the field looks numeric, else a string comparison against the constant's
%  decimal text. Same operands and op codes as llvm_emit_atom_field_i64_cmp_guard/7,
%  which keeps its strict-integer meaning for any other caller.
llvm_emit_atom_field_strnum_cmp_guard(ValueIR, FieldIndex, OpCode, Expected, SepCode, ResultIR, CallIR) :-
    integer(FieldIndex),
    integer(OpCode),
    between(0, 5, OpCode),
    integer(Expected),
    format(atom(CallIR),
        '  ~w = call i1 @wam_atom_field_strnum_cmp_int(%Value ~w, i64 ~w, i8 ~w, i64 ~w, i32 ~w)',
        [ResultIR, ValueIR, FieldIndex, SepCode, Expected, OpCode]).

%% llvm_emit_atom_field_str_cmp_guard(+GlobalBase, +ValueIR, +FieldIndex, +OpCode, +Expected, +SepCode, +ResultIR, -GlobalIR-CallIR)
%
%  Emit a native LEXICAL field-vs-string-literal comparison over an atom-backed
%  text record (`$N < "str"` and friends). The literal is a private c-string
%  constant; the runtime memcmp-compares the field slice against it (string
%  semantics, never numeric). OpCode is 2 <, 3 <=, 4 >, 5 >= (== / != use the
%  field_eq path).
llvm_emit_atom_field_str_cmp_guard(GlobalBase, ValueIR, FieldIndex, OpCode, Expected, SepCode, ResultIR, GlobalIR-CallIR) :-
    integer(FieldIndex),
    integer(OpCode),
    between(0, 5, OpCode),
    sanitize_functor_for_llvm(GlobalBase, SafeBase),
    escape_llvm_string(Expected, EscapedExpected, ExpectedLen),
    ArrLen is ExpectedLen + 1,
    format(atom(GlobalIR),
        '@.~w = private constant [~w x i8] c"~w\\00"',
        [SafeBase, ArrLen, EscapedExpected]),
    format(atom(CallIR),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0~n  ~w = call i1 @wam_atom_field_str_cmp_value(%Value ~w, i64 ~w, i8* %~w_ptr, i64 ~w, i8 ~w, i32 ~w)',
        [SafeBase, ArrLen, ArrLen, SafeBase,
         ResultIR, ValueIR, FieldIndex, SafeBase, ExpectedLen, SepCode, OpCode]).

%% llvm_emit_atom_field_i64(+ValueIR, +FieldIndex, +SepCode, +ParseBase, -CallIR)
%
%  Emit a native signed-decimal i64 parse over an atom-backed text record.
%  Field 0 parses the whole record; positive fields parse the projected field
%  slice. CallIR defines %ParseBase, %ParseBase_value, and %ParseBase_ok.
llvm_emit_atom_field_i64(ValueIR, FieldIndex, SepCode, ParseBase, CallIR) :-
    integer(FieldIndex),
    format(atom(CallIR),
        '  %~w = call %WamI64Parse @wam_atom_field_i64_value(%Value ~w, i64 ~w, i8 ~w)~n  %~w_value = extractvalue %WamI64Parse %~w, 0~n  %~w_ok = extractvalue %WamI64Parse %~w, 1',
        [ParseBase, ValueIR, FieldIndex, SepCode,
         ParseBase, ParseBase,
         ParseBase, ParseBase]).

%% llvm_emit_atom_field_i64_or_default(+ValueIR, +FieldIndex, +SepCode, +DefaultValue, +ParseBase, +ResultIR, -CallIR)
%
%  Emit a native signed-decimal i64 parse and coerce failed parses to
%  DefaultValue. This is useful for AWK-style numeric contexts while keeping the
%  parse value/success pair reusable for stricter consumers.
llvm_emit_atom_field_i64_or_default(ValueIR, FieldIndex, SepCode, DefaultValue,
        ParseBase, ResultIR, CallIR) :-
    integer(FieldIndex),
    integer(DefaultValue),
    llvm_emit_atom_field_i64(ValueIR, FieldIndex, SepCode, ParseBase, ParseIR),
    format(atom(SelectIR),
        '  ~w = select i1 %~w_ok, i64 %~w_value, i64 ~w',
        [ResultIR, ParseBase, ParseBase, DefaultValue]),
    format(atom(CallIR), '~w~n~w', [ParseIR, SelectIR]).

%% llvm_emit_c_string_global(+GlobalName, +Value, -GlobalIR, -StringLen, -BytesLen)
%
%  Emit a null-terminated `i8` array global for a C string literal. GlobalName is
%  the LLVM global stem after `@.`; callers that synthesize arbitrary names
%  should sanitize before calling. StringLen excludes the null terminator;
%  BytesLen includes it and is suitable for getelementptr array types.
llvm_emit_c_string_global(GlobalName, Value, GlobalIR, StringLen, BytesLen) :-
    escape_llvm_string(Value, Escaped, StringLen),
    BytesLen is StringLen + 1,
    format(atom(GlobalIR),
        '@.~w = private constant [~w x i8] c"~w\\00"',
        [GlobalName, BytesLen, Escaped]).

%% llvm_emit_printf_i64(+FmtGlobal, +FmtVar, +PrintVar, +ValueIR, -Parts)
%
%  Emit a `%ld`-style `printf` call against a shared format-string global.
llvm_emit_printf_i64(FmtGlobal, FmtVar, PrintVar, ValueIR, [FmtPtr, PrintCall]) :-
    format(atom(FmtPtr),
        '  %~w = getelementptr [4 x i8], [4 x i8]* @.~w, i32 0, i32 0',
        [FmtVar, FmtGlobal]),
    format(atom(PrintCall),
        '  %~w = call i32 (i8*, ...) @printf(i8* %~w, i64 ~w)',
        [PrintVar, FmtVar, ValueIR]).

%% llvm_emit_printf_slice(+FmtGlobal, +FmtVar, +PrintVar, +LenIR, +PtrIR, -Parts)
%
%  Emit a `%.*s`-style `printf` call for an allocation-free byte slice.
llvm_emit_printf_slice(FmtGlobal, FmtVar, PrintVar, LenIR, PtrIR, [FmtPtr, PrintCall]) :-
    format(atom(FmtPtr),
        '  %~w = getelementptr [5 x i8], [5 x i8]* @.~w, i32 0, i32 0',
        [FmtVar, FmtGlobal]),
    format(atom(PrintCall),
        '  %~w = call i32 (i8*, ...) @printf(i8* %~w, i32 ~w, i8* ~w)',
        [PrintVar, FmtVar, LenIR, PtrIR]).

%% llvm_emit_printf_string(+FmtGlobal, +FmtVar, +PrintVar, +PtrIR, -Parts)
%
%  Emit a `%s`-style `printf` call for a null-terminated C string pointer.
llvm_emit_printf_string(FmtGlobal, FmtVar, PrintVar, PtrIR, [FmtPtr, PrintCall]) :-
    llvm_emit_printf_string(FmtGlobal, 3, FmtVar, PrintVar, PtrIR,
        [FmtPtr, PrintCall]).

%% llvm_emit_printf_string(+FmtGlobal, +FmtLen, +FmtVar, +PrintVar, +PtrIR, -Parts)
%
%  Emit a string-pointer `printf` call against a shared C string format global.
%  FmtLen is the global array length including the null terminator.
llvm_emit_printf_string(FmtGlobal, FmtLen, FmtVar, PrintVar, PtrIR, [FmtPtr, PrintCall]) :-
    format(atom(FmtPtr),
        '  %~w = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [FmtVar, FmtLen, FmtLen, FmtGlobal]),
    format(atom(PrintCall),
        '  %~w = call i32 (i8*, ...) @printf(i8* %~w, i8* ~w)',
        [PrintVar, FmtVar, PtrIR]).

%% llvm_emit_printf0(+FmtGlobal, +FmtLen, +FmtVar, +PrintVar, -Parts)
%
%  Emit a no-argument `printf` call against a shared format-string global, such
%  as newline wrappers.
llvm_emit_printf0(FmtGlobal, FmtLen, FmtVar, PrintVar, [FmtPtr, PrintCall]) :-
    format(atom(FmtPtr),
        '  %~w = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [FmtVar, FmtLen, FmtLen, FmtGlobal]),
    format(atom(PrintCall),
        '  %~w = call i32 (i8*, ...) @printf(i8* %~w)',
        [PrintVar, FmtVar]).

%% llvm_emit_stream_driver_ir(+InputPath, +DriverBlocks, -DriverIR)
%
%  Emit the reusable native file-stream driver skeleton. Surface-specific
%  lowerers provide runtime globals, optional entry setup, loop-carried phis,
%  per-record lowered IR, continuation phis, optional terminal-break close IR,
%  and the close-success block.
%
%  InputPath is either a concrete path (compiled into the binary as a
%  string global) or the atom `stdin_or_argv`, which emits a
%  main(argc, argv) that opens argv[1] at runtime, treats "-" as stdin,
%  and defaults to stdin (fd 0) when no argument is given -- the awk
%  pipeline-filter convention.
%
%  Records are read with @wam_stream_read_line_transient_value: %line
%  is the reserved transient atom whose string lives in a shared buffer
%  valid only until the next read, so per-record lowerings read it
%  freely but anything persisted past the record must intern (field
%  slices used as assoc keys and $0 foreign-call arguments do).
llvm_emit_stream_driver_ir(
    InputPath,
    driver_blocks(RuntimeGlobals, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, '', LoopPhiIR, LoweredLabel, RecordIR,
            ContinueIR, CloseOkLabel, CloseOkIR),
        DriverIR).

llvm_emit_stream_driver_ir(
    InputPath,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
            ContinueIR, '', CloseOkLabel, CloseOkIR),
        DriverIR).

llvm_emit_stream_driver_ir(
    stdin_or_argv,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    !,
    format(atom(DriverIR),
'@.wam_stream_eof = private constant [12 x i8] c"end_of_file\\00"
@.wam_stream_stdin_dash = private constant [2 x i8] c"-\\00"
~w

define i32 @main(i32 %argc, i8** %argv) {
entry:
~w
  %have_arg = icmp sgt i32 %argc, 1
  br i1 %have_arg, label %check_argv_path, label %use_stdin

check_argv_path:
  %argv1_ptr = getelementptr i8*, i8** %argv, i64 1
  %argv1 = load i8*, i8** %argv1_ptr
  %dash_ptr = getelementptr [2 x i8], [2 x i8]* @.wam_stream_stdin_dash, i32 0, i32 0
  %dash_cmp = call i32 @strcmp(i8* %argv1, i8* %dash_ptr)
  %is_dash = icmp eq i32 %dash_cmp, 0
  br i1 %is_dash, label %use_stdin, label %open_argv_file

open_argv_file:
  %argv1_len = call i64 @strlen(i8* %argv1)
  %path_id = call i64 @wam_intern_atom(i8* %argv1, i64 %argv1_len)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %file_handle = call %Value @wam_stream_open_value(%Value %path)
  br label %have_handle

use_stdin:
  %stdin_handle = call %Value @wam_stream_open_fd_value(i64 0)
  br label %have_handle

have_handle:
  %handle = phi %Value [ %file_handle, %open_argv_file ], [ %stdin_handle, %use_stdin ]
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
  %line = call %Value @wam_stream_read_line_transient_value(%Value %handle)
  %line_tag = extractvalue %Value %line, 0
  %line_payload = extractvalue %Value %line, 1
  %line_is_int = icmp eq i32 %line_tag, 1
  %line_bad_payload = icmp slt i64 %line_payload, 0
  %line_bad = and i1 %line_is_int, %line_bad_payload
  br i1 %line_bad, label %fail_read, label %check_line_atom

check_line_atom:
  %line_is_atom = icmp eq i32 %line_tag, 0
  br i1 %line_is_atom, label %check_eof, label %fail_line_tag

check_eof:
  ; EOF by IDENTITY, not text. wam_stream_read_line_transient_value returns a
  ; successful data record as the reserved transient id 2^62 (text in the shared
  ; transient buffer) and real end-of-input as the interned end_of_file atom (a
  ; distinct id). The old check stringified the id and compared the text against "end_of_file",
  ; so any DATA line whose text was literally "end_of_file" (a transient, id 2^62)
  ; matched and truncated the input. Comparing the id is exact: the only non-transient
  ; value the reader returns here is the end_of_file atom = EOF. %line_s is still bound
  ; (the record body prints $0 from it); only the EOF DECISION changes to the id test.
  %line_s = call i8* @wam_atom_to_string(i64 %line_payload)
  %is_eof = icmp ne i64 %line_payload, 4611686018427387904
  br i1 %is_eof, label %close_stream, label %~w

~w:
~w

continue_loop:
~w
  br label %loop

~w
close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %~w, label %fail_close

~w

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_line_tag:
  %close_ignore_line_tag = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 12

fail_close:
  ret i32 16
}
',
        [RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel,
         LoweredLabel, RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR]).

llvm_emit_stream_driver_ir(
    InputPath,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    llvm_emit_c_string_global(wam_stream_input_path, InputPath, PathGlobalIR, PathLen, BytesLen),
    format(atom(DriverIR),
'~w
@.wam_stream_eof = private constant [12 x i8] c"end_of_file\\00"
~w

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.wam_stream_input_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
~w
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
  %line = call %Value @wam_stream_read_line_transient_value(%Value %handle)
  %line_tag = extractvalue %Value %line, 0
  %line_payload = extractvalue %Value %line, 1
  %line_is_int = icmp eq i32 %line_tag, 1
  %line_bad_payload = icmp slt i64 %line_payload, 0
  %line_bad = and i1 %line_is_int, %line_bad_payload
  br i1 %line_bad, label %fail_read, label %check_line_atom

check_line_atom:
  %line_is_atom = icmp eq i32 %line_tag, 0
  br i1 %line_is_atom, label %check_eof, label %fail_line_tag

check_eof:
  ; EOF by IDENTITY, not text. wam_stream_read_line_transient_value returns a
  ; successful data record as the reserved transient id 2^62 (text in the shared
  ; transient buffer) and real end-of-input as the interned end_of_file atom (a
  ; distinct id). The old check stringified the id and compared the text against "end_of_file",
  ; so any DATA line whose text was literally "end_of_file" (a transient, id 2^62)
  ; matched and truncated the input. Comparing the id is exact: the only non-transient
  ; value the reader returns here is the end_of_file atom = EOF. %line_s is still bound
  ; (the record body prints $0 from it); only the EOF DECISION changes to the id test.
  %line_s = call i8* @wam_atom_to_string(i64 %line_payload)
  %is_eof = icmp ne i64 %line_payload, 4611686018427387904
  br i1 %is_eof, label %close_stream, label %~w

~w:
~w

continue_loop:
~w
  br label %loop

~w
close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %~w, label %fail_close

~w

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_line_tag:
  %close_ignore_line_tag = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 12

fail_close:
  ret i32 16
}
',
        [PathGlobalIR, RuntimeGlobals,
         BytesLen, BytesLen, PathLen, EntrySetupIR, LoopPhiIR, LoweredLabel,
         LoweredLabel, RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR]).

%% llvm_emit_binary_stream_driver_ir(+InputPath, +RecordSize, +DriverBlocks, -DriverIR)
%
%  Binary sibling of llvm_emit_stream_driver_ir: the loop reads
%  fixed-size RecordSize-byte records into a reusable %rec buffer with
%  @wam_stream_read_record instead of interning text lines. There is no
%  %line/%line_s; lowered blocks read typed fields directly from %rec.
%  Block labels (%check_handle_value, %continue_loop, close labels)
%  match the text skeleton so loop-carried phis work unchanged. A
%  trailing partial record fails with the read error exit code.
%  InputPath is a concrete path or the stdin_or_argv sentinel.
llvm_emit_binary_stream_driver_ir(
    InputPath,
    RecordSize,
    driver_blocks(RuntimeGlobals, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    !,
    llvm_emit_binary_stream_driver_ir(InputPath, RecordSize,
        driver_blocks(RuntimeGlobals, '', LoopPhiIR, LoweredLabel, RecordIR,
            ContinueIR, '', CloseOkLabel, CloseOkIR),
        DriverIR).

llvm_emit_binary_stream_driver_ir(
    InputPath,
    RecordSize,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    !,
    llvm_emit_binary_stream_driver_ir(InputPath, RecordSize,
        driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
            ContinueIR, '', CloseOkLabel, CloseOkIR),
        DriverIR).

llvm_emit_binary_stream_driver_ir(
    stdin_or_argv,
    RecordSize,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    !,
    format(atom(DriverIR),
'@.wam_stream_stdin_dash = private constant [2 x i8] c"-\00"
~w

define i32 @main(i32 %argc, i8** %argv) {
entry:
~w
  %rec = call i8* @malloc(i64 ~w)
  %rec_null = icmp eq i8* %rec, null
  br i1 %rec_null, label %fail_open, label %pick_input

pick_input:
  %have_arg = icmp sgt i32 %argc, 1
  br i1 %have_arg, label %check_argv_path, label %use_stdin

check_argv_path:
  %argv1_ptr = getelementptr i8*, i8** %argv, i64 1
  %argv1 = load i8*, i8** %argv1_ptr
  %dash_ptr = getelementptr [2 x i8], [2 x i8]* @.wam_stream_stdin_dash, i32 0, i32 0
  %dash_cmp = call i32 @strcmp(i8* %argv1, i8* %dash_ptr)
  %is_dash = icmp eq i32 %dash_cmp, 0
  br i1 %is_dash, label %use_stdin, label %open_argv_file

open_argv_file:
  %argv1_len = call i64 @strlen(i8* %argv1)
  %path_id = call i64 @wam_intern_atom(i8* %argv1, i64 %argv1_len)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %file_handle = call %Value @wam_stream_open_value(%Value %path)
  br label %have_handle

use_stdin:
  %stdin_handle = call %Value @wam_stream_open_fd_value(i64 0)
  br label %have_handle

have_handle:
  %handle = phi %Value [ %file_handle, %open_argv_file ], [ %stdin_handle, %use_stdin ]
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
  %rec_status = call i64 @wam_stream_read_record(%Value %handle, i64 ~w, i8* %rec)
  %rec_eof = icmp eq i64 %rec_status, 0
  br i1 %rec_eof, label %close_stream, label %check_record

check_record:
  %rec_ok = icmp eq i64 %rec_status, 1
  br i1 %rec_ok, label %~w, label %fail_read

~w:
~w

continue_loop:
~w
  br label %loop

~w
close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %~w, label %fail_close

~w

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_close:
  ret i32 16
}
',
        [RuntimeGlobals, EntrySetupIR, RecordSize, LoopPhiIR, RecordSize,
         LoweredLabel, LoweredLabel, RecordIR, ContinueIR, BreakCloseIR,
         CloseOkLabel, CloseOkIR]).

llvm_emit_binary_stream_driver_ir(
    InputPath,
    RecordSize,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    llvm_emit_c_string_global(wam_stream_input_path, InputPath, PathGlobalIR, PathLen, BytesLen),
    format(atom(DriverIR),
'~w
~w

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.wam_stream_input_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
~w
  %rec = call i8* @malloc(i64 ~w)
  %rec_null = icmp eq i8* %rec, null
  br i1 %rec_null, label %fail_open, label %open_stream

open_stream:
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
  %rec_status = call i64 @wam_stream_read_record(%Value %handle, i64 ~w, i8* %rec)
  %rec_eof = icmp eq i64 %rec_status, 0
  br i1 %rec_eof, label %close_stream, label %check_record

check_record:
  %rec_ok = icmp eq i64 %rec_status, 1
  br i1 %rec_ok, label %~w, label %fail_read

~w:
~w

continue_loop:
~w
  br label %loop

~w
close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %~w, label %fail_close

~w

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_close:
  ret i32 16
}
',
        [PathGlobalIR, RuntimeGlobals,
         BytesLen, BytesLen, PathLen, EntrySetupIR, RecordSize, LoopPhiIR,
         RecordSize, LoweredLabel, LoweredLabel, RecordIR, ContinueIR,
         BreakCloseIR, CloseOkLabel, CloseOkIR]).

%% llvm_emit_varlen_stream_driver_ir(+InputPath, +BufSize, +ReadIR,
%%     +DriverBlocks, -DriverIR)
%
%  Variable-length sibling of llvm_emit_binary_stream_driver_ir: instead
%  of one fixed-size @wam_stream_read_record call, the caller supplies
%  ReadIR -- a field-by-field read sequence that materializes one record
%  into the BufSize-byte %rec buffer and ends by branching to the
%  lowered label (or %close_stream on clean EOF at a record boundary /
%  %fail_read on a short read). Block labels match the fixed skeletons
%  so loop phis and lowered blocks work unchanged.
llvm_emit_varlen_stream_driver_ir(
    stdin_or_argv,
    BufSize,
    ReadIR,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel,
        RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    !,
    format(atom(DriverIR),
'@.wam_stream_stdin_dash = private constant [2 x i8] c"-\00"
~w

define i32 @main(i32 %argc, i8** %argv) {
entry:
~w
  %rec = call i8* @malloc(i64 ~w)
  %rec_null = icmp eq i8* %rec, null
  br i1 %rec_null, label %fail_open, label %pick_input

pick_input:
  %have_arg = icmp sgt i32 %argc, 1
  br i1 %have_arg, label %check_argv_path, label %use_stdin

check_argv_path:
  %argv1_ptr = getelementptr i8*, i8** %argv, i64 1
  %argv1 = load i8*, i8** %argv1_ptr
  %dash_ptr = getelementptr [2 x i8], [2 x i8]* @.wam_stream_stdin_dash, i32 0, i32 0
  %dash_cmp = call i32 @strcmp(i8* %argv1, i8* %dash_ptr)
  %is_dash = icmp eq i32 %dash_cmp, 0
  br i1 %is_dash, label %use_stdin, label %open_argv_file

open_argv_file:
  %argv1_len = call i64 @strlen(i8* %argv1)
  %path_id = call i64 @wam_intern_atom(i8* %argv1, i64 %argv1_len)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %file_handle = call %Value @wam_stream_open_value(%Value %path)
  br label %have_handle

use_stdin:
  %stdin_handle = call %Value @wam_stream_open_fd_value(i64 0)
  br label %have_handle

have_handle:
  %handle = phi %Value [ %file_handle, %open_argv_file ], [ %stdin_handle, %use_stdin ]
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
~w

~w:
~w

continue_loop:
~w
  br label %loop

~w
close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %~w, label %fail_close

~w

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_close:
  ret i32 16
}
',
        [RuntimeGlobals, EntrySetupIR, BufSize, LoopPhiIR, ReadIR,
         LoweredLabel, RecordIR, ContinueIR, BreakCloseIR,
         CloseOkLabel, CloseOkIR]).

llvm_emit_varlen_stream_driver_ir(
    InputPath,
    BufSize,
    ReadIR,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel,
        RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    llvm_emit_c_string_global(wam_stream_input_path, InputPath, PathGlobalIR, PathLen, BytesLen),
    format(atom(DriverIR),
'~w
~w

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.wam_stream_input_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
~w
  %rec = call i8* @malloc(i64 ~w)
  %rec_null = icmp eq i8* %rec, null
  br i1 %rec_null, label %fail_open, label %open_stream

open_stream:
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
~w

~w:
~w

continue_loop:
~w
  br label %loop

~w
close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %~w, label %fail_close

~w

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_close:
  ret i32 16
}
',
        [PathGlobalIR, RuntimeGlobals,
         BytesLen, BytesLen, PathLen, EntrySetupIR, BufSize, LoopPhiIR,
         ReadIR, LoweredLabel, RecordIR, ContinueIR,
         BreakCloseIR, CloseOkLabel, CloseOkIR]).

%% llvm_emit_ascii_case_slice_print(+Mode, +PtrIR, +LenIR, +PrintBase, -CallIR)
%
%  Emit an allocation-free ASCII case-mapped slice printer. This is intentionally
%  a print helper, not a value materializer; callers that need a transformed atom
%  should use a separate interning/materialization helper.
llvm_emit_ascii_case_slice_print(lower, PtrIR, LenIR, PrintBase, CallIR) :-
    nonvar(PrintBase),
    format(atom(CallIR),
        '  call void @wam_print_ascii_lower_slice(i8* ~w, i64 ~w)',
        [PtrIR, LenIR]).
llvm_emit_ascii_case_slice_print(upper, PtrIR, LenIR, PrintBase, CallIR) :-
    nonvar(PrintBase),
    format(atom(CallIR),
        '  call void @wam_print_ascii_upper_slice(i8* ~w, i64 ~w)',
        [PtrIR, LenIR]).

escape_llvm_codes([], []).
escape_llvm_codes([C|Cs], Out) :-
    ( C >= 0x20, C =< 0x7E, C \== 0x22, C \== 0x5C
    -> Out = [C|Rest],
       escape_llvm_codes(Cs, Rest)
    ;  Hi is (C >> 4) /\ 0xF,
       Lo is C /\ 0xF,
       hex_digit(Hi, HiC),
       hex_digit(Lo, LoC),
       Out = [0'\\, HiC, LoC | Rest],
       escape_llvm_codes(Cs, Rest)
    ).

%% sanitize_functor_for_llvm(+Name, -Sanitized)
%  Produce a bijective LLVM-identifier encoding of Name. Alphanumeric
%  bytes pass through unchanged; everything else (including underscore)
%  is hex-escaped as `_HH`. Bijectivity matters: two distinct functor
%  strings must never collapse to the same LLVM global name, or we get
%  `redefinition of global` when llc sees the bench module (which
%  references e.g. both `+` and `*` as functors).
sanitize_functor_for_llvm(Name, Sanitized) :-
    ( atom(Name) -> atom_string(Name, Str)
    ; string(Name) -> Str = Name
    ; Str = Name
    ),
    string_codes(Str, Codes),
    sanitize_codes(Codes, SanCodes),
    string_codes(Sanitized, SanCodes).

%% split_functor_arity(+Str, -NameStr, -Arity) is det.
%
%  Parses a Prolog functor/arity string ("foo/2") into name and arity,
%  splitting on the LAST `/` rather than the first. This handles
%  predicate / functor names that themselves contain `/`:
%
%    "foo/2"   → Name = "foo",  Arity = 2
%    "//2"     → Name = "/",    Arity = 2   (integer division)
%    "/3"      → Name = "",     Arity = 3   (degenerate but accepted)
%    "a/b/4"   → Name = "a/b",  Arity = 4   (no native equivalent but
%                                            correct under this rule)
%
%  Used by put_structure / get_structure / call / execute literal
%  emitters in both the bytecode pipeline and the lowered emitter.
%  Throws if there's no `/` in the input (callers expect a Name/Arity
%  shape; absence of `/` is a bytecode bug).
split_functor_arity(Str0, Name, Arity) :-
    ( atom(Str0) -> atom_string(Str0, S)
    ; string(Str0) -> S = Str0
    ; S = Str0
    ),
    split_string(S, "/", "", Parts),
    length(Parts, NParts),
    ( NParts < 2
    -> throw(error(split_functor_arity_no_slash(S), _))
    ;  true
    ),
    last(Parts, ArityStr),
    number_string(Arity, ArityStr),
    NameCount is NParts - 1,
    length(NameParts, NameCount),
    append(NameParts, [_], Parts),
    atomic_list_concat(NameParts, "/", Name).

%% register_functor_string(+NameStr) is det.
%
%  Idempotent assert into the functor_string_global/2 table. Callers
%  that emit `@.fn_<sane>` references must call this so the matching
%  `@.fn_<sane> = private constant [N x i8] c"<name>\00"` global is
%  emitted by emit_functor_string_globals/1 at module assembly time.
%
%  Accepts a string OR an atom — converted to string internally so the
%  table key is consistent.
register_functor_string(Name) :-
    ( atom(Name) -> atom_string(Name, NameStr)
    ; string(Name) -> NameStr = Name
    ; NameStr = Name
    ),
    string_length(NameStr, Len),
    LenPlus1 is Len + 1,
    ( functor_string_global(NameStr, _) -> true
    ; assertz(functor_string_global(NameStr, LenPlus1))
    ).

sanitize_codes([], []).
sanitize_codes([C|Cs], [C|Out]) :-
    ( C >= 0'a, C =< 0'z
    ; C >= 0'A, C =< 0'Z
    ; C >= 0'0, C =< 0'9
    ), !,
    sanitize_codes(Cs, Out).
sanitize_codes([C|Cs], [0'_,H,L|Out]) :-
    Hi is (C >> 4) /\ 0xF,
    Lo is C /\ 0xF,
    hex_digit(Hi, H),
    hex_digit(Lo, L),
    sanitize_codes(Cs, Out).

hex_digit(N, C) :- N < 10, !, C is N + 0'0.
hex_digit(N, C) :- C is N - 10 + 0'A.
:- dynamic atom_table_entry/2.   % atom_table_entry(AtomName, Id)
:- dynamic atom_table_next_id/1. % atom_table_next_id(NextId)
atom_table_next_id(1).           % Start from 1; 0 reserved for empty

%% intern_atom(+AtomName, -Id)
%  Returns the unique integer ID for AtomName, allocating a new one if needed.
intern_atom(AtomName, Id) :-
    (   atom_table_entry(AtomName, Id)
    ->  true
    ;   retract(atom_table_next_id(Id)),
        NextId is Id + 1,
        assertz(atom_table_next_id(NextId)),
        assertz(atom_table_entry(AtomName, Id))
    ).

% ----------------------------------------------------------------------
% M113: target-platform abstraction for libc struct layouts.
% Most libc *functions* are POSIX-portable, but the *struct field
% offsets* we read at IR-emit time (struct dirent for M107, struct tm
% for M91/M98/M99, struct stat for M73-M76) are libc-specific. To keep
% IR generation cross-target, callers pick a strategy via
%   target_os(auto)    -- (default) derive OS from target_triple option
%   target_os(linux)   -- explicit; linux | darwin | freebsd | netbsd
%                          | openbsd | wasi | ... (atom)
% adapt-at-runtime (a third mode that emits a probe to discover offsets
% at program startup) is a follow-up milestone -- see todo at end of
% this block.

%% target_os_from_triple(+Triple, -OS) is det.
%  Maps an LLVM target triple ("x86_64-pc-linux-gnu", "x86_64-apple-darwin",
%  "x86_64-unknown-freebsd13.0", "wasm32-unknown-wasi", ...) to a short
%  atom we can pattern-match against in layout predicates.
target_os_from_triple(Triple, OS) :-
    atom_string(Triple, S),
    (   sub_string(S, _, _, _, "linux")   -> OS = linux
    ;   sub_string(S, _, _, _, "darwin")  -> OS = darwin
    ;   sub_string(S, _, _, _, "freebsd") -> OS = freebsd
    ;   sub_string(S, _, _, _, "openbsd") -> OS = openbsd
    ;   sub_string(S, _, _, _, "netbsd")  -> OS = netbsd
    ;   sub_string(S, _, _, _, "wasi")    -> OS = wasi
    ;   sub_string(S, _, _, _, "windows") -> OS = windows
    ;   sub_string(S, _, _, _, "mingw")   -> OS = windows
    ;   OS = linux   % default to linux on unknown triples (warns elsewhere)
    ).

%% resolve_target_os(+Options, -OS) is det.
%  Consults the Options list for target_os(Mode):
%   - target_os(auto)  (default if absent) -> derive from target_triple
%   - target_os(<atom>) (linux/darwin/etc.) -> use that OS directly
target_os_resolved(Options, OS) :-
    ( option(target_os(Mode), Options) -> true ; Mode = auto ),
    ( Mode == auto
    -> ( option(target_triple(T), Options)
       -> target_os_from_triple(T, OS)
       ;  OS = linux
       )
    ;  OS = Mode
    ).

%% target_dirent_d_name_offset(+OS, -Offset) is det.
%  Byte offset of d_name within struct dirent on the named OS.
%  Linux glibc: ino(8) + off(8) + reclen(2) + type(1) = 19.
%  Darwin (BSD libc): ino(8) + seekoff(8) + reclen(2) + namlen(2) + type(1) = 21.
%  FreeBSD/NetBSD/OpenBSD vary -- 11/12 historically, modern is 21.
%  WASI uses dirent compatible with Linux (the wasi-libc preview).
target_dirent_d_name_offset(linux,   19).
target_dirent_d_name_offset(darwin,  21).
target_dirent_d_name_offset(freebsd, 21).
target_dirent_d_name_offset(netbsd,  21).
target_dirent_d_name_offset(openbsd, 21).
target_dirent_d_name_offset(wasi,    19).
target_dirent_d_name_offset(adapt,   0).   % unused -- adapt uses helper.
target_dirent_d_name_offset(_, 19).        % fall-back

%% target_dirent_name_ptr_ir(+OS, +StaticOff, -IR) is det.
%  Produce the IR snippet that defines %dirf.name_ptr from
%  %dirf.entry. For static OS modes this is one getelementptr with a
%  literal byte offset. For target_os(adapt) it calls the runtime
%  probe helper @wam_dirent_d_name_offset (always emitted in the
%  module) and uses the i64 it returns.
target_dirent_name_ptr_ir(adapt, _Off, IR) :- !,
    atomic_list_concat([
        '  %dirf.dno = call i64 @wam_dirent_d_name_offset()',
        '  %dirf.name_ptr = getelementptr i8, i8* %dirf.entry, i64 %dirf.dno'
    ], '\n', IR).
target_dirent_name_ptr_ir(_OS, Off, IR) :-
    format(atom(IR), '  %dirf.name_ptr = getelementptr i8, i8* %dirf.entry, i64 ~w', [Off]).

%% target_dirent_probe_helper_ir(-IR) is det.
%  M114: emit the runtime probe helper that discovers struct dirent''s
%  d_name byte offset for the current process. Strategy: opendir("."),
%  readdir once (always returns the "." entry first), scan the buffer
%  for the first ''.'' byte (256 bytes max -- d_name is bounded by
%  NAME_MAX on every supported platform), closedir. Cache the result
%  in a private i64 global so subsequent calls reuse the probe. On
%  failure (opendir/readdir error), fall back to 19 (Linux default)
%  so callers continue with sane semantics even in chroots without /.
target_dirent_probe_helper_ir(IR) :-
    IR = '; === M114 runtime dirent::d_name offset probe ===
@.uw_dot_dir = private constant [2 x i8] c".\\00"
@uw_dirent_d_name_offset_cache = private global i64 -1

define i64 @wam_dirent_d_name_offset() {
entry:
  %cached = load i64, i64* @uw_dirent_d_name_offset_cache
  %unset = icmp slt i64 %cached, 0
  br i1 %unset, label %probe, label %done
done:
  ret i64 %cached
probe:
  %dot_ptr = getelementptr [2 x i8], [2 x i8]* @.uw_dot_dir, i32 0, i32 0
  %dir = call i8* @opendir(i8* %dot_ptr)
  %dir_null = icmp eq i8* %dir, null
  br i1 %dir_null, label %fallback, label %read
read:
  %entry_ptr = call i8* @readdir(i8* %dir)
  %entry_null = icmp eq i8* %entry_ptr, null
  br i1 %entry_null, label %fallback_close, label %scan_init
scan_init:
  br label %scan_loop
scan_loop:
  %i = phi i64 [ 0, %scan_init ], [ %i_next, %scan_step ]
  %too_far = icmp uge i64 %i, 256
  br i1 %too_far, label %fallback_close, label %scan_check
scan_check:
  %byte_ptr = getelementptr i8, i8* %entry_ptr, i64 %i
  %b = load i8, i8* %byte_ptr
  %is_dot = icmp eq i8 %b, 46
  br i1 %is_dot, label %scan_found, label %scan_step
scan_step:
  %i_next = add i64 %i, 1
  br label %scan_loop
scan_found:
  %close_ret = call i32 @closedir(i8* %dir)
  store i64 %i, i64* @uw_dirent_d_name_offset_cache
  ret i64 %i
fallback_close:
  %close_ret2 = call i32 @closedir(i8* %dir)
  br label %fallback
fallback:
  store i64 19, i64* @uw_dirent_d_name_offset_cache
  ret i64 19
}'.

%% target_struct_tm_size(+OS, -SizeBytes) is det.
%  Glibc struct tm is 56 bytes; Darwin/BSD also 56 with same field
%  order (the GNU tm_gmtoff/tm_zone extension is also present on
%  Darwin and modern BSDs).
target_struct_tm_size(_, 56).

%% target_struct_tm_gmtoff_offset(+OS, -Offset) is det.
target_struct_tm_gmtoff_offset(linux,   40).
target_struct_tm_gmtoff_offset(darwin,  40).
target_struct_tm_gmtoff_offset(freebsd, 40).
target_struct_tm_gmtoff_offset(_, 40).

% TODO: target_os(adapt) -- emit a small startup probe that opens
% "." via opendir, calls readdir once (which always returns "." as
% the first entry), and computes the d_name offset from where the
% '.' byte lands in the buffer. Result stored in a runtime global so
% builtin_directory_files reads the resolved offset instead of a
% compile-time constant. Defer to a future milestone.

% --- Value packing helpers ---

% %Value tag for a head constant: integers (raw or integer/1-wrapped)
% are tag 1, everything else interns as an atom (tag 0). get_constant
% and unify_constant share this so integer constants in heads compare
% against heap Integers, not same-payload Atoms.
llvm_constant_tag(C, 1) :- integer(C), !.
llvm_constant_tag(integer(_), 1) :- !.
llvm_constant_tag(_, 0).

llvm_pack_value(atom(A0), Packed) :- !,
    llvm_normalize_atom_token(A0, A),
    intern_atom(A, Packed).
llvm_pack_value(integer(I), I) :- !.
llvm_pack_value(N, N) :- integer(N), !.
llvm_pack_value(N, Packed) :- float(N), !, Packed is truncate(N).
llvm_pack_value(A0, Packed) :- atom(A0), !,
    llvm_normalize_atom_token(A0, A),
    intern_atom(A, Packed).
llvm_pack_value(_, 0).
llvm_normalize_atom_token(A0, A) :-
    atom_string(A0, S0),
    strip_quoted_atom(S0, S),
    atom_string(A, S).

% --- Builtin op name → integer ID mapping ---

builtin_op_to_id('is/2', 0).
builtin_op_to_id('>/2', 1).
builtin_op_to_id('</2', 2).
builtin_op_to_id('>=/2', 3).
builtin_op_to_id('=</2', 4).
builtin_op_to_id('=:=/2', 5).
builtin_op_to_id('=\\=/2', 6).
builtin_op_to_id('==/2', 7).
builtin_op_to_id('true/0', 8).
builtin_op_to_id('fail/0', 9).
builtin_op_to_id('!/0', 10).
builtin_op_to_id('write/1', 11).
builtin_op_to_id('nl/0', 12).
builtin_op_to_id('atom/1', 13).
builtin_op_to_id('integer/1', 14).
builtin_op_to_id('float/1', 15).
builtin_op_to_id('number/1', 16).
builtin_op_to_id('compound/1', 17).
builtin_op_to_id('var/1', 18).
builtin_op_to_id('nonvar/1', 19).
builtin_op_to_id('is_list/1', 20).
builtin_op_to_id('\\==/2', 21).
builtin_op_to_id('succ/2', 22).
builtin_op_to_id('plus/3', 23).
builtin_op_to_id('=/2', 24).
builtin_op_to_id('\\=/2', 25).
builtin_op_to_id('functor/3', 26).
builtin_op_to_id('arg/3', 27).
builtin_op_to_id('=../2', 28).
builtin_op_to_id('copy_term/2', 29).
builtin_op_to_id('msort/2', 30).
builtin_op_to_id('length/2', 31).
builtin_op_to_id('format/2', 32).
builtin_op_to_id('format/1', 33).
builtin_op_to_id('atom_length/2', 35).
builtin_op_to_id('atom_codes/2', 36).
builtin_op_to_id('char_code/2', 37).
builtin_op_to_id('atom_chars/2', 38).
builtin_op_to_id('between/3', 39).
builtin_op_to_id('atom_concat/3', 40).
builtin_op_to_id('sub_atom/5', 41).
builtin_op_to_id('atom_number/2', 42).
builtin_op_to_id('number_codes/2', 43).
builtin_op_to_id('number_chars/2', 44).
builtin_op_to_id('upcase_atom/2', 45).
builtin_op_to_id('downcase_atom/2', 46).
builtin_op_to_id('string_concat/3', 47).  % alias of atom_concat/3
builtin_op_to_id('string_length/2', 48).  % alias of atom_length/2
builtin_op_to_id('atom_string/2', 49).    % atom <-> string (runtime treats both as atoms)
builtin_op_to_id('string_to_atom/2', 50). % string_to_atom(?S, ?A) -- same as atom_string but swapped
builtin_op_to_id('nth0/3', 51).
builtin_op_to_id('nth1/3', 52).
builtin_op_to_id('last/2', 53).
builtin_op_to_id('reverse/2', 54).
builtin_op_to_id('append/3', 55).
builtin_op_to_id('memberchk/2', 56).
builtin_op_to_id('delete/3', 57).
builtin_op_to_id('numlist/3', 58).
builtin_op_to_id('sum_list/2', 59).
builtin_op_to_id('sumlist/2', 60).   % alias of sum_list/2
builtin_op_to_id('max_list/2', 61).
builtin_op_to_id('min_list/2', 62).
builtin_op_to_id('subtract/3', 63).
builtin_op_to_id('intersection/3', 64).
builtin_op_to_id('union/3', 65).
builtin_op_to_id('list_to_set/2', 66).
builtin_op_to_id('string_chars/2', 67).  % alias of atom_chars/2
builtin_op_to_id('string_codes/2', 68).  % alias of atom_codes/2
builtin_op_to_id('string_code/3', 69).   % string_code(+Idx, +Str, -Code) 1-based
builtin_op_to_id('pairs_keys/2', 70).    % extract keys from K-V pair list
builtin_op_to_id('pairs_values/2', 71).  % extract values
builtin_op_to_id('pairs_keys_values/3', 72). % forward only: split pairs -> keys + values
builtin_op_to_id('atomic_list_concat/2', 73). % concat list of atoms (atoms only mode)
builtin_op_to_id('atomic_list_concat/3', 74). % concat with separator atom (atoms-only forward)
builtin_op_to_id('char_type/2', 75).          % char classification (check mode)
builtin_op_to_id('compare/3', 76).            % three-way term order (num/atom)
builtin_op_to_id('must_be/2', 77).            % type guard (fail-instead-of-throw)
builtin_op_to_id('display/1', 78).            % alias of write/1 (operator-free print)
builtin_op_to_id('writeln/1', 79).            % write/1 + nl/0 combined
builtin_op_to_id('keysort/2', 80).            % stable sort of K-V pairs by Key
builtin_op_to_id('sort/2', 81).               % sort + dedup under standard term order
builtin_op_to_id('tab/1', 82).                % I/O: print N spaces.
builtin_op_to_id('put_char/1', 83).           % I/O: print first byte of single-char atom.
builtin_op_to_id('put_code/1', 84).           % I/O: print byte value as char.
builtin_op_to_id('split_string/4', 85).       % split atom on any char from SepChars; strip pad chars from segments.
builtin_op_to_id('@</2', 86).                 % standard order: less than.
builtin_op_to_id('@=</2', 87).                % standard order: less or equal.
builtin_op_to_id('@>/2', 88).                 % standard order: greater than.
builtin_op_to_id('@>=/2', 89).                % standard order: greater or equal.
builtin_op_to_id('get_time/1', 90).           % wall-clock time as Float seconds since epoch.
builtin_op_to_id('exists_file/1', 91).        % path exists AND is a regular file.
builtin_op_to_id('exists_directory/1', 92).   % path exists AND is a directory.
builtin_op_to_id('delete_file/1', 93).        % libc unlink(path).
builtin_op_to_id('make_directory/1', 94).     % libc mkdir(path, 0o755).
builtin_op_to_id('rename_file/2', 95).        % libc rename(old, new).
builtin_op_to_id('delete_directory/1', 96).   % libc rmdir(path) (empty dirs only).
builtin_op_to_id('size_file/2', 97).          % stat -> st_size as Integer.
builtin_op_to_id('time_file/2', 98).          % stat -> st_mtime as Float seconds.
builtin_op_to_id('getenv/2', 99).             % libc getenv(name) -> atom value or fail.
builtin_op_to_id('setenv/2', 100).            % libc setenv(name, value, 1).
builtin_op_to_id('shell/1', 101).             % libc system(cmd) succeeds iff exit 0.
builtin_op_to_id('shell/2', 102).             % libc system(cmd), Status = WEXITSTATUS.
builtin_op_to_id('working_directory/2', 103). % getcwd -> Old; chdir(New) if New atom.
builtin_op_to_id('getpid/1', 104).            % libc getpid() as Integer.
builtin_op_to_id('sleep/1', 105).             % libc usleep(seconds * 1e6).
builtin_op_to_id('gethostname/1', 106).       % libc gethostname() as atom.
builtin_op_to_id('cpu_time/1', 107).          % clock_gettime(CLOCK_PROCESS_CPUTIME_ID).
builtin_op_to_id('random/1', 108).            % libc drand48() in [0, 1) as Float.
builtin_op_to_id('random_between/3', 109).    % L + lrand48() % (H-L+1).
builtin_op_to_id('format_time/3', 110).       % strftime(Fmt, localtime(Stamp)).
builtin_op_to_id('halt/0', 111).              % libc exit(0).
builtin_op_to_id('halt/1', 112).              % libc exit(Code).
builtin_op_to_id('unsetenv/1', 113).          % libc unsetenv(Name).
builtin_op_to_id('getuid/1', 114).            % libc getuid() as Integer.
builtin_op_to_id('geteuid/1', 115).           % libc geteuid() as Integer.
builtin_op_to_id('getgid/1', 116).            % libc getgid() as Integer.
builtin_op_to_id('getegid/1', 117).           % libc getegid() as Integer.
builtin_op_to_id('getppid/1', 118).           % libc getppid() as Integer.
builtin_op_to_id('set_random/1', 119).        % srand48 via seed(N) compound.
builtin_op_to_id('stamp_date_time/3', 120).   % localtime_r + build 9-arity date/9.
builtin_op_to_id('date_time_stamp/2', 121).   % mktime via date/9 compound.
builtin_op_to_id('chmod/2', 122).             % libc chmod(Path, Mode).
builtin_op_to_id('term_variables/2', 123).    % depth-first var collection.
builtin_op_to_id('numbervars/3', 124).        % bind free vars to $VAR(N).
builtin_op_to_id('access/2', 125).            % libc access(path, mode_bits).
builtin_op_to_id('directory_files/2', 126).   % opendir/readdir loop -> list of atoms.
builtin_op_to_id('getpgrp/1', 127).           % libc getpgrp() as Integer.
builtin_op_to_id('realpath/2', 128).          % libc realpath(rel) -> Abs atom.
builtin_op_to_id('kill/2', 129).              % libc kill(pid, sig).
builtin_op_to_id('truncate/2', 130).          % libc truncate(path, length).
builtin_op_to_id('chown/3', 131).             % libc chown(path, uid, gid).
builtin_op_to_id('ground/1', 132).            % succeeds iff term has no unbound vars.
builtin_op_to_id('file_base_name/2', 133).    % basename(1): part after last ''/''.
builtin_op_to_id('file_directory_name/2', 134). % dirname(1): prefix before last ''/'' (or ''.'').
builtin_op_to_id('file_name_extension/3', 135). % split/join basename at last ''.'' (basename-scoped).
builtin_op_to_id('read_link/2', 136).         % libc readlink -- symlink target as atom.
builtin_op_to_id('symlink/2', 137).           % libc symlink(target, linkpath).
builtin_op_to_id('link/2', 138).              % libc link(old, new) -- hard link.
builtin_op_to_id('is_absolute_file_name/1', 139). % true iff first char is ''/''.
builtin_op_to_id('same_file/2', 140).         % stat both, compare st_dev + st_ino.
builtin_op_to_id('tmp_file/2', 141).          % mkstemp-based race-free temp file.
builtin_op_to_id('mkfifo/2', 142).            % libc mkfifo(path, mode).
builtin_op_to_id('umask/2', 143).             % libc umask(?Old, +New).
builtin_op_to_id('monotonic_time/1', 144).    % clock_gettime(CLOCK_MONOTONIC) as Float.
builtin_op_to_id('nice/1', 145).              % libc nice(inc) -- adjust priority.
builtin_op_to_id('getpriority/1', 146).       % libc getpriority(PRIO_PROCESS, 0).
builtin_op_to_id('setpriority/1', 147).       % libc setpriority(PRIO_PROCESS, 0, prio).
builtin_op_to_id('getrlimit/2', 148).         % libc getrlimit(Res, &rlim) -- soft limit.
builtin_op_to_id('setrlimit/2', 149).         % libc setrlimit(Res, &rlim) -- soft limit only.
builtin_op_to_id('getlogin/1', 150).          % libc getlogin() as atom.
builtin_op_to_id('uname_sysname/1', 151).     % libc uname() sysname field as atom.
builtin_op_to_id('uname_machine/1', 152).     % libc uname() machine field as atom.
builtin_op_to_id('copy_file/2', 153).         % open + read/write loop + close.
builtin_op_to_id('read_file_to_atom/2', 154). % stat + open + read loop -> intern as atom.
builtin_op_to_id('write_atom_to_file/2', 155). % open(O_TRUNC) + write loop + close.
builtin_op_to_id('append_atom_to_file/2', 156). % open(O_APPEND) + write loop + close.
builtin_op_to_id('errno/1', 157).             % thread-local errno via __errno_location.
builtin_op_to_id('strerror/2', 158).          % libc strerror(errno) as atom.
builtin_op_to_id('process_max_rss/1', 159).   % getrusage ru_maxrss (KB on Linux).
builtin_op_to_id('process_user_time/1', 160). % getrusage ru_utime as Float seconds.
builtin_op_to_id('process_system_time/1', 161). % getrusage ru_stime as Float seconds.
builtin_op_to_id('path_join/3', 162).         % join paths with '/' separator (absolute Rel wins).
builtin_op_to_id('system_to_atom/2', 163).    % popen(cmd, "r") + fread + pclose -> atom.
builtin_op_to_id('atom_to_system/2', 164).    % popen(cmd, "w") + fwrite + pclose, succeed iff status 0.
builtin_op_to_id('atom_split/3', 165).        % split atom on single-char sep -> list of substring atoms.
builtin_op_to_id('stream_open/2', 166).       % open + table-backed buffered line-reader handle.
builtin_op_to_id('read_line/2', 167).         % buffered line read; EOF unifies end_of_file.
builtin_op_to_id('stream_close/1', 168).      % close + free line-reader handle.
builtin_op_to_id('atom_starts_with/2', 169).  % prefix check via memcmp.
builtin_op_to_id('atom_ends_with/2', 170).    % suffix check via memcmp.
builtin_op_to_id('atom_contains/2', 171).     % substring check via strstr.
builtin_op_to_id('code_type/2', 172).        % ASCII class check; shares char_type's classifier.
builtin_op_to_id('term_to_atom/2', 173).     % render a term to its text (write semantics) and intern as an atom.
builtin_op_to_id('read_term_from_atom/2', 174). % parse an atom's text into a term (atomic terms only, for now).
builtin_op_to_id('assertz/1', 175).          % dynamic clause store: append a ground fact.
builtin_op_to_id('asserta/1', 176).          % dynamic clause store: prepend a ground fact.
builtin_op_to_id('retractall/1', 177).       % dynamic clause store: tombstone matching clauses.
% Catch-all for builtin names with no dedicated dispatch entry. Must
% be a value that no real builtin uses AND that the switch in
% @execute_builtin has no case for, so dispatch falls through to the
% %unknown label (ret i1 false). 99 was previously used here, but
% 99 is also the dispatch case for getenv/2 -- so `\+ G' where G is
% any non-builtin or a non-dispatched builtin would route to
% builtin_getenv, read whatever junk reg 0 held, and return false
% instead of running the metacall semantics. 200 is well above the
% currently-defined range and not in the switch.
builtin_op_to_id(_, 200).  % Unknown

% ============================================================================
% WAM line parser → LLVM struct literals (from WAM assembly text)
% ============================================================================

%% compile_wam_predicate_to_llvm(+Pred/Arity, +WamCode, +Options, -LLVMCode)
%  Takes WAM instruction output and produces LLVM IR with instruction
%  array and label table as global constants.
compile_wam_predicate_to_llvm(Pred/Arity, WamCode, _Options, LLVMCode) :-
    atom_string(Pred, PredStr),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_llvm(Lines, 0, LLVMLiterals, LabelEntries),
    % Second pass: resolve switch_deferred terms to real instruction literals
    % while emitting per-switch %SwitchEntry table globals.
    resolve_switch_tables(LLVMLiterals, PredStr, 0, ResolvedLiterals, SwitchTableDefs),
    length(ResolvedLiterals, InstrCount),
    length(LabelEntries, LabelCount),
    % Build instruction array entries
    maplist([Lit, Entry]>>(format(atom(Entry), '  ~w', [Lit])), ResolvedLiterals, Entries),
    atomic_list_concat(Entries, ',\n', EntriesStr),
    % Build label array entries. When the predicate has zero labels we
    % still emit a 1-element placeholder (LLVM rejects [0 x i32] with a
    % 1-element initializer, and vice versa). The logical count passed
    % to wam_state_new stays zero, but the declared array type has to
    % match the initializer length so the two are tracked separately.
    maplist([_-Idx, Entry]>>(format(atom(Entry), '  i32 ~w', [Idx])), LabelEntries, LabelRows),
    (   LabelRows == []
    ->  LabelsStr = "  i32 0",
        LabelArraySize = 1
    ;   atomic_list_concat(LabelRows, ',\n', LabelsStr),
        LabelArraySize = LabelCount
    ),
    % Build arg setup
    build_llvm_arg_setup(Arity, ArgSetup),
    build_llvm_param_list(Arity, ParamList),
    % Join switch table definitions (may be empty)
    ( SwitchTableDefs == []
    -> SwitchTablesStr = ""
    ;  atomic_list_concat(SwitchTableDefs, '\n', SwitchTablesStr0),
       format(atom(SwitchTablesStr), '~w\n', [SwitchTablesStr0])
    ),
    format(atom(LLVMCode),
'; WAM-compiled predicate: ~w/~w
~w@~w_code = private constant [~w x %Instruction] [
~w
]

@~w_labels = private constant [~w x i32] [
~w
]

define i1 @~w(~w) {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @~w_code, i32 0, i32 0),
    i32 ~w,
    i32* getelementptr ([~w x i32], [~w x i32]* @~w_labels, i32 0, i32 0),
    i32 ~w)
~w
  %result = call i1 @run_loop(%WamState* %vm)
  ret i1 %result
}
', [PredStr, Arity,
    SwitchTablesStr,
    PredStr, InstrCount, EntriesStr,
    PredStr, LabelArraySize, LabelsStr,
    PredStr, ParamList,
    InstrCount, InstrCount, PredStr, InstrCount,
    LabelArraySize, LabelArraySize, PredStr, LabelCount,
    ArgSetup]).

%% resolve_switch_tables(+LiteralsIn, +PredStr, +NextIdx, -LiteralsOut, -TableDefs)
%  Walks the literal list, replacing switch_deferred/2 terms with real
%  %Instruction literals referencing freshly-allocated switch table globals.
resolve_switch_tables([], _, _, [], []).
resolve_switch_tables([switch_deferred(constant, Entries) | Rest], PredStr, Idx,
        [InstrLit | RestOut], [TableDef | RestDefs]) :- !,
    length(Entries, Count),
    format(atom(TableName), '~w_switch_~w', [PredStr, Idx]),
    render_switch_entries(Entries, EntryLines),
    atomic_list_concat(EntryLines, ',\n', EntriesStr),
    format(atom(TableDef),
'@~w = private constant [~w x %SwitchEntry] [
~w
]',         [TableName, Count, EntriesStr]),
    format(atom(InstrLit),
'%Instruction { i32 25, i64 ptrtoint ([~w x %SwitchEntry]* @~w to i64), i64 ~w }',
        [Count, TableName, Count]),
    Idx1 is Idx + 1,
    resolve_switch_tables(Rest, PredStr, Idx1, RestOut, RestDefs).
resolve_switch_tables([switch_deferred(constant_a2, Entries) | Rest], PredStr, Idx,
        [InstrLit | RestOut], [TableDef | RestDefs]) :- !,
    length(Entries, Count),
    format(atom(TableName), '~w_switch_a2_~w', [PredStr, Idx]),
    render_switch_entries(Entries, EntryLines),
    atomic_list_concat(EntryLines, ',\n', EntriesStr),
    format(atom(TableDef),
'@~w = private constant [~w x %SwitchEntry] [
~w
]',         [TableName, Count, EntriesStr]),
    format(atom(InstrLit),
'%Instruction { i32 27, i64 ptrtoint ([~w x %SwitchEntry]* @~w to i64), i64 ~w }',
        [Count, TableName, Count]),
    Idx1 is Idx + 1,
    resolve_switch_tables(Rest, PredStr, Idx1, RestOut, RestDefs).
resolve_switch_tables([Lit | Rest], PredStr, Idx, [Lit | RestOut], RestDefs) :-
    resolve_switch_tables(Rest, PredStr, Idx, RestOut, RestDefs).

render_switch_entries([], []).
render_switch_entries([entry(Tag, Pay, LabelIdx) | Rest], [Line | RestLines]) :-
    format(atom(Line),
        '  %SwitchEntry { i32 ~w, i64 ~w, i32 ~w }',
        [Tag, Pay, LabelIdx]),
    render_switch_entries(Rest, RestLines).

%% llvm_emit_atom_fact2_table(+TableName, +Pairs, -LLVMGlobal)
%
%  Emit a private global constant array of %AtomFactPair entries for
%  a list of (FromAtom, ToAtom) facts. Both atoms are interned via
%  intern_atom/2 so the IDs stay consistent with switch_on_constant
%  tables and kernel lookups.
%
%  Example:
%      llvm_emit_atom_fact2_table('category_parent_table',
%          [fact('Physics', 'Science'), fact('Chemistry', 'Science')],
%          Code)
%
%  Produces an LLVM global definition:
%      @category_parent_table = private constant [2 x %AtomFactPair] [
%        %AtomFactPair { i64 1, i64 2 },
%        %AtomFactPair { i64 3, i64 2 }
%      ]
%
llvm_emit_atom_fact2_table(TableName, Pairs, Code) :-
    maplist(render_atom_fact_pair, Pairs, Lines),
    length(Pairs, Count),
    (   Count == 0
    ->  format(atom(Code),
            '@~w = private constant [1 x %AtomFactPair] [%AtomFactPair { i64 0, i64 0 }]',
            [TableName])
    ;   atomic_list_concat(Lines, ',\n', EntriesStr),
        format(atom(Code),
'@~w = private constant [~w x %AtomFactPair] [
~w
]', [TableName, Count, EntriesStr])
    ).

render_atom_fact_pair(fact(From, To), Line) :-
    intern_atom(From, FromId),
    intern_atom(To, ToId),
    format(atom(Line),
        '  %AtomFactPair { i64 ~w, i64 ~w }',
        [FromId, ToId]).
render_atom_fact_pair(From-To, Line) :-
    render_atom_fact_pair(fact(From, To), Line).

%% llvm_emit_weighted_edge_table(+TableName, +Triples, -LLVMGlobal)
%
%  Emit a private global constant array of %WeightedFact entries for
%  a list of (From, To, Weight) weighted edges. Atoms interned, weight
%  emitted as a double (LLVM requires decimal form for fp constants).
%
%  Example:
%      llvm_emit_weighted_edge_table('cat_weighted',
%          [edge('ml', 'ai', 0.12), edge('ai', 'cs', 0.18)], Code)
%
llvm_emit_weighted_edge_table(TableName, Triples, Code) :-
    maplist(render_weighted_fact, Triples, Lines),
    length(Triples, Count),
    (   Count == 0
    ->  format(atom(Code),
            '@~w = private constant [1 x %WeightedFact] [%WeightedFact { i64 0, i64 0, double 0.0 }]',
            [TableName])
    ;   atomic_list_concat(Lines, ',\n', EntriesStr),
        format(atom(Code),
'@~w = private constant [~w x %WeightedFact] [
~w
]', [TableName, Count, EntriesStr])
    ).

render_weighted_fact(edge(From, To, Weight), Line) :-
    intern_atom(From, FromId),
    intern_atom(To, ToId),
    format_weight_literal(Weight, WeightStr),
    format(atom(Line),
        '  %WeightedFact { i64 ~w, i64 ~w, double ~w }',
        [FromId, ToId, WeightStr]).
render_weighted_fact(From-To-Weight, Line) :-
    render_weighted_fact(edge(From, To, Weight), Line).

%% format_weight_literal(+Weight, -Str)
%  LLVM's double literal parser requires either decimal form (3.14) or
%  hex form. An integer printed as "1" is rejected where a double is
%  expected; we must emit "1.0". Also, "0" must become "0.0".
format_weight_literal(W, Str) :-
    number(W),
    ( integer(W)
    -> format(string(Str), '~w.0', [W])
    ;  format(string(Str), '~w', [W])
    ).

%% wam_lines_to_llvm(+Lines, +PC, -LLVMLits, -LabelEntries)
%  Two-pass approach: first collect all labels and raw instruction parts,
%  then generate LLVM literals with resolved label indices.
wam_lines_to_llvm(Lines, StartPC, LLVMLits, LabelEntries) :-
    % Pass 1: collect labels and raw instruction parts
    wam_lines_pass1(Lines, StartPC, RawInstrs, LabelEntries),
    % Build label name → index mapping (position in label array)
    build_label_index_map(LabelEntries, LabelMap),
    % Pass 2: generate LLVM literals with label resolution
    maplist(resolve_llvm_literal(LabelMap), RawInstrs, LLVMLits).

%% wam_lines_pass1(+Lines, +PC, -RawInstrs, -Labels)
%  First pass: separate labels from instructions, track PC.
wam_lines_pass1([], _, [], []).
wam_lines_pass1([Line|Rest], PC, RawInstrs, Labels) :-
    % M12: quote-aware split. A `put_constant 'hello world~n', A1`
    % bytecode line has whitespace inside the single-quoted atom; the
    % old space/tab split shattered it into garbage parts that the
    % per-instruction parser couldn't recognise and fell to the
    % `; TODO: ...` stub, which broke @module_code array length
    % accounting at llc time. Now `'...'` groups are preserved as a
    % single Part (with the quotes still attached -- downstream
    % clean_comma + llvm_pack_value_str strips them).
    quote_aware_split(Line, Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_pass1(Rest, PC, RawInstrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            Labels = [LabelName-PC | RestLabels],
            wam_lines_pass1(Rest, PC, RawInstrs, RestLabels)
        ;   RawInstrs = [CleanParts | RestInstrs],
            NPC is PC + 1,
            wam_lines_pass1(Rest, NPC, RestInstrs, Labels)
        )
    ).

%% quote_aware_split(+Line, -Parts)
%
%  Split Line on whitespace while preserving single-quoted runs as a
%  single Part. Quote escapes inside a quoted run follow the standard
%  Prolog convention: a doubled `''` is a single literal apostrophe.
%  Quotes themselves stay in the returned Part (so callers can tell a
%  quoted atom from an unquoted identifier); the downstream
%  clean_comma + llvm_pack_value_str path handles stripping.
quote_aware_split(Line, Parts) :-
    ( atom(Line) -> atom_codes(Line, Codes)
    ; string(Line) -> string_codes(Line, Codes)
    ; Codes = Line
    ),
    qa_split(Codes, [], Parts).

% qa_split(+Codes, +CurAcc, -PartsOut)
qa_split([], [], []) :- !.
qa_split([], Acc, [Part]) :-
    reverse(Acc, AccF),
    string_codes(Part, AccF).
qa_split([C|Cs], Acc, Out) :-
    ( ( C == 0' ; C == 0'\t )
    -> ( Acc == []
       -> qa_split(Cs, [], Out)
       ;  reverse(Acc, AccF),
          string_codes(Part, AccF),
          Out = [Part | Rest],
          qa_split(Cs, [], Rest)
       )
    ; C == 0''
    -> % Enter a single-quoted run. Read codes verbatim (handling
       % doubled `''` as an escape) until the closing quote.
       qa_quoted([C|Cs], Acc, AccQ, Cs2),
       qa_split(Cs2, AccQ, Out)
    ;  qa_split(Cs, [C|Acc], Out)
    ).

% qa_quoted(+Codes, +AccIn, -AccOut, -RestCodes)
%  Codes starts with the opening `'`. Reads through to the matching
%  closing `'`, preserving the quotes (and any escaped doubles) in AccOut.
qa_quoted([Q|Cs], AccIn, AccOut, RestOut) :-
    Q = 0'',
    qa_quoted_body(Cs, [Q|AccIn], AccOut, RestOut).

qa_quoted_body([92, 39 | Cs], Acc, AccOut, RestOut) :- !,
    qa_quoted_body(Cs, [39, 92 | Acc], AccOut, RestOut).
qa_quoted_body([92, 92 | Cs], Acc, AccOut, RestOut) :- !,
    qa_quoted_body(Cs, [92, 92 | Acc], AccOut, RestOut).
qa_quoted_body([], Acc, Acc, []).
qa_quoted_body([0'', 0''|Cs], Acc, AccOut, RestOut) :- !,
    % Escaped apostrophe: keep both in the accumulator.
    qa_quoted_body(Cs, [0'', 0''|Acc], AccOut, RestOut).
qa_quoted_body([0''|Cs], Acc, [0''|Acc], Cs) :- !.
qa_quoted_body([C|Cs], Acc, AccOut, RestOut) :-
    qa_quoted_body(Cs, [C|Acc], AccOut, RestOut).

%% build_label_index_map(+LabelEntries, -LabelMap)
%  Creates an assoc mapping label names to their index in the label array.
build_label_index_map(LabelEntries, LabelMap) :-
    length(LabelEntries, _),
    foldl(add_label_entry, LabelEntries, 0-[], _-LabelMap).

add_label_entry(Name-_PC, Idx-Map, NextIdx-[Name-Idx|Map]) :-
    NextIdx is Idx + 1.

%% resolve_llvm_literal(+LabelMap, +Parts, -LLVMLit)
%  Second pass: generate LLVM literal with resolved label indices.
resolve_llvm_literal(LabelMap, Parts, LLVMLit) :-
    wam_line_to_llvm_literal_resolved(Parts, LabelMap, LLVMLit).

%% wam_line_to_llvm_literal_resolved(+Parts, +LabelMap, -LLVMLit)
%  Converts parsed WAM instruction text to LLVM %Instruction struct literal,
%  with label names resolved to indices via LabelMap.

%% lookup_label_index(+LabelName, +LabelMap, -Index)
%  Find label index in map. Behaviour on unknown labels depends on context:
%  - Default: warn on stderr, return 0 (for external predicate references)
%  - With wam_strict_labels(true) in Options: throw an error
lookup_label_index(LabelName, LabelMap, Index) :-
    lookup_label_index(LabelName, LabelMap, [], Index).
lookup_label_index(LabelName, LabelMap, Options, Index) :-
    (   member(LabelName-Index, LabelMap)
    ->  true
    ;   (   option(wam_strict_labels(false), Options)
        ->  format(user_error,
                'Warning: unknown label "~w" in WAM LLVM codegen, defaulting to index 0~n',
                [LabelName]),
            Index = 0
        ;   throw(error(existence_error(procedure, LabelName),
                context(wam_llvm_codegen,
                    'called predicate was not compiled into this module; add it to the predicate list passed to write_wam_llvm_project/3, or pass wam_strict_labels(false) to allow the legacy index-0 fallback')))
        )
    ).

% Instructions that need label resolution:
% NOTE: trailing `; comment` removed — LLVM line comments eat the comma
% separator in array constants. The label name is preserved for humans by
% having `llvm-dis` show labels resolved from the label array, not inline.
wam_line_to_llvm_literal_resolved(["call", P, N], LabelMap, Lit) :- !,
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Arity, CN) -> true ; Arity = 0 ),
    (   wam_special_call_label(CP, LabelIdx, _)
    ->  true
    ;   wam_meta_call_label(CP)
    ->  LabelIdx = -1
    ;   lookup_label_index(CP, LabelMap, LabelIdx)
    ),
    format(atom(Lit), '%Instruction { i32 18, i64 ~w, i64 ~w }', [LabelIdx, Arity]).
wam_line_to_llvm_literal_resolved(["execute", P], LabelMap, Lit) :- !,
    clean_comma(P, CP),
    (   wam_special_call_label(CP, LabelIdx, Arity)
    ->  true
    ;   wam_meta_call_label(CP)
    ->  LabelIdx = -1,
        wam_meta_call_total_arity(CP, Arity)
    ;   lookup_label_index(CP, LabelMap, LabelIdx),
        Arity = 0
    ),
    format(atom(Lit), '%Instruction { i32 19, i64 ~w, i64 ~w }', [LabelIdx, Arity]).
wam_line_to_llvm_literal_resolved(["try_me_else", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    lookup_label_index(CL, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 22, i64 ~w, i64 0 }', [LabelIdx]).
wam_line_to_llvm_literal_resolved(["retry_me_else", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    lookup_label_index(CL, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 23, i64 ~w, i64 0 }', [LabelIdx]).
wam_line_to_llvm_literal_resolved(["jump", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    lookup_label_index(CL, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 32, i64 ~w, i64 0 }', [LabelIdx]).
wam_line_to_llvm_literal_resolved(["switch_on_constant_fallthrough" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
wam_line_to_llvm_literal_resolved(["switch_on_constant_a2_fallthrough" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 27, i64 0, i64 0 }'.
% switch_on_constant: defer until compile_wam_predicate_to_llvm can
% allocate a switch table global. Returns a switch_deferred(_) term.
wam_line_to_llvm_literal_resolved(["switch_on_constant" | EntryParts], LabelMap,
        switch_deferred(constant, Entries)) :- !,
    parse_switch_entries(EntryParts, LabelMap, Entries).
wam_line_to_llvm_literal_resolved(["switch_on_structure" | _], _, Lit) :- !,
    % nop fallthrough — the try_me_else chain still runs.
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
wam_line_to_llvm_literal_resolved(["switch_on_constant_a2" | EntryParts], LabelMap,
        switch_deferred(constant_a2, Entries)) :- !,
    parse_switch_entries(EntryParts, LabelMap, Entries).
% switch_on_term / switch_on_term_a2: type-based dispatch that falls
% back to the try_me_else / retry_me_else chain. The LLVM target does
% not implement these yet, so emit a nop and let the linear chain run.
wam_line_to_llvm_literal_resolved(["switch_on_term" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
wam_line_to_llvm_literal_resolved(["switch_on_term_a2" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
% try / retry / trust (no `_me_else` suffix): emitted by the WAM
% compiler''s indexed-dispatch group pass to chain together clauses
% that share an indexing key. The LLVM target''s switch_on_constant
% / switch_on_term are nop fallthroughs, so the indexed body never
% gets entered and these instructions are unreachable in practice;
% emit nop so they fall through to the next instruction (which
% continues the try_me_else / retry_me_else / trust_me chain).
wam_line_to_llvm_literal_resolved(["try" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
wam_line_to_llvm_literal_resolved(["retry" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
wam_line_to_llvm_literal_resolved(["trust" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
% All other instructions: delegate to existing parser (no labels needed)
wam_line_to_llvm_literal_resolved(Parts, _LabelMap, Lit) :-
    wam_line_to_llvm_literal(Parts, Lit).

%% parse_switch_entries(+Parts, +LabelMap, -Entries)
%  Each part is "key:label" possibly with trailing comma. Produces a list
%  of entry(KeyTag, KeyPayload, LabelIdx) terms. LabelMap has string keys
%  (from wam_lines_pass1 using sub_string), so we pass label strings
%  directly without converting to atoms.
parse_switch_entries([], _, []).
parse_switch_entries([Part | Rest], LabelMap, [Entry | RestEntries]) :-
    clean_comma(Part, Clean),
    % Split at the LAST colon and unquote writeq-style keys (see
    % switch_entry_split/3 -- a first-colon split sheared quoted keys
    % like '=:=' in half, and un-unquoted keys interned their quote
    % characters into the atom name; either way the table entry never
    % matched the runtime atom and dispatch silently failed).
    switch_entry_split(Clean, KeyStr0, LabelStr),
    switch_entry_unquote(KeyStr0, KeyStr),
    % Pack the key: integer keys use tag=1, atom keys use tag=0 with interned id.
    ( number_string(N, KeyStr)
    -> KeyTag = 1, KeyPayload = N
    ;  KeyTag = 0,
       atom_string(KeyAtom, KeyStr),
       intern_atom(KeyAtom, KeyPayload)
    ),
    % "default" is a pseudo-label meaning "fall through to next instruction".
    % Encode as sentinel -1 which the runtime helper maps to "advance PC".
    ( LabelStr == "default"
    -> LabelIdx = -1
    ;  % Pass the string directly — LabelMap has string keys.
       ( lookup_label_index(LabelStr, LabelMap, LabelIdx)
       -> true
       ;  LabelIdx = 0
       )
    ),
    Entry = entry(KeyTag, KeyPayload, LabelIdx),
    parse_switch_entries(Rest, LabelMap, RestEntries).

%% wam_line_to_llvm_literal(+Parts, -LLVMLit)
%  Converts parsed WAM instruction text to LLVM %Instruction struct literal.
%  For non-label-referencing instructions only. Label-referencing instructions
%  are handled by wam_line_to_llvm_literal_resolved/3 above.

wam_line_to_llvm_literal(["get_constant", C, Ai], Lit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    llvm_pack_value_str(CC, PackedVal),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, RegIdx),
    % Determine tag: integers get tag=1, atoms get tag=0.
    ( number_string(_, CC) -> Tag = 1 ; Tag = 0 ),
    % Pack tag into op2 high bits: op2 = (tag << 16) | reg_idx.
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), '%Instruction { i32 0, i64 ~w, i64 ~w }', [PackedVal, Op2]).
wam_line_to_llvm_literal(["get_structure", FN, Ai], Lit) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    atom_string(CFN, CFNStr),
    split_functor_arity(CFNStr, NameStr, Arity),
    string_length(NameStr, NameLen),
    NameLenPlus1 is NameLen + 1,
    Op2 is (Arity << 16) \/ AiIdx,
    sanitize_functor_for_llvm(NameStr, SaneName),
    format(atom(Lit),
        '%Instruction { i32 3, i64 ptrtoint ([~w x i8]* @.fn_~w to i64), i64 ~w }',
        [NameLenPlus1, SaneName, Op2]),
    register_functor_string(NameStr).
wam_line_to_llvm_literal(["get_variable", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 1, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["get_value", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 2, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["get_structure", FN, Ai], Lit) :-
    clean_comma(FN, _CFN), clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 3, i64 0, i64 ~w }', [AiIdx]).
wam_line_to_llvm_literal(["get_list", Ai], Lit) :-
    clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 4, i64 ~w, i64 0 }', [AiIdx]).
wam_line_to_llvm_literal(["unify_variable", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 5, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["unify_value", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 6, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["unify_constant", C], Lit) :-
    clean_comma(C, CC),
    llvm_pack_value_str(CC, PackedVal),
    ( number_string(_, CC) -> Tag = 1 ; Tag = 0 ),
    Op2 is Tag << 16,
    format(atom(Lit), '%Instruction { i32 7, i64 ~w, i64 ~w }', [PackedVal, Op2]).

wam_line_to_llvm_literal(["put_constant", C, Ai], Lit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, RegIdx),
    set_constant_literal_parts(CC, PayloadStr, Tag),
    % Encode op2 = (tag << 16) | reg_idx. Float literals (tag=2) and
    % integers (tag=1) and atoms (tag=0) all routed through the
    % shared set_constant_literal_parts helper so float constants
    % survive the WAM-text round-trip as `bitcast (double <c> to i64)`
    % rather than a bare `0.5` that llc rejects.
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), '%Instruction { i32 8, i64 ~w, i64 ~w }',
        [PayloadStr, Op2]).
wam_line_to_llvm_literal(["put_variable", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 9, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["put_value", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 10, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["put_structure", FN, Ai], Lit) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    % Parse functor/arity from "name/N" format. Use split_functor_arity/3
    % which splits on the LAST `/` so functor names that contain `/`
    % (e.g. the integer-division operator `//`) parse correctly. The
    % naive split_string(_, "/", "", [Name, Arity]) fails on `//2`
    % because consecutive separators produce an empty middle part.
    atom_string(CFN, CFNStr),
    split_functor_arity(CFNStr, NameStr, Arity),
    string_length(NameStr, NameLen),
    NameLenPlus1 is NameLen + 1,
    % Encode: op1 = ptrtoint of functor string global, op2 = (arity << 16) | reg_idx.
    Op2 is (Arity << 16) \/ AiIdx,
    % Sanitize functor name for LLVM identifier (replace special chars).
    sanitize_functor_for_llvm(NameStr, SaneName),
    format(atom(Lit),
        '%Instruction { i32 11, i64 ptrtoint ([~w x i8]* @.fn_~w to i64), i64 ~w }',
        [NameLenPlus1, SaneName, Op2]),
    % Record functor string for emission as global. Route through
    % register_functor_string so the table key normalises to a
    % string regardless of whether NameStr arrives as an atom or
    % string -- otherwise an eager-registered "[|]" (string key)
    % and a lazy-registered '[|]' (atom key from split_functor_arity)
    % both get emitted, causing `redefinition of global` in llc.
    register_functor_string(NameStr).
wam_line_to_llvm_literal(["put_list", Ai], Lit) :-
    clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 12, i64 ~w, i64 0 }', [AiIdx]).
wam_line_to_llvm_literal(["set_variable", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 13, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["set_value", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 14, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["set_constant", C], Lit) :-
    clean_comma(C, CC),
    set_constant_literal_parts(CC, PayloadStr, Tag),
    format(atom(Lit), '%Instruction { i32 15, i64 ~w, i64 ~w }',
        [PayloadStr, Tag]).

%% set_constant_literal_parts(+Str, -PayloadStr, -Tag)
%
%  Resolve a textual constant from a `set_constant` / `put_constant`
%  bytecode line to its (Tag, Payload) representation. Integer
%  literals box as Integer (tag=1); float literals box as Float
%  (tag=2) with the IEEE-754 bit pattern in the payload via
%  bitcast double <c> to i64. Atoms intern through the runtime
%  atom table and box as Atom (tag=0).
set_constant_literal_parts(CC, PayloadStr, Tag) :-
    (   number_string(N, CC)
    ->  ( integer(N)
        ->  Tag = 1,
            format(atom(PayloadStr), '~w', [N])
        ;   % Float: emit as a bitcast of the double constant to i64
            %   `i64 bitcast (double 0.5 to i64)`
            % so llc sees a valid i64 initializer at compile time.
            Tag = 2,
            format(atom(PayloadStr),
                'bitcast (double ~w to i64)', [N])
        )
    ;   Tag = 0,
        strip_quoted_atom(CC, Unquoted),
        atom_string(A, Unquoted),
        intern_atom(A, AtomId),
        format(atom(PayloadStr), '~w', [AtomId])
    ).

wam_line_to_llvm_literal(["allocate"], '%Instruction { i32 16, i64 0, i64 0 }').
wam_line_to_llvm_literal(["deallocate"], '%Instruction { i32 17, i64 0, i64 0 }').

% begin_aggregate type, ValueReg, ResultReg [, WitnessRegs]
% The WitnessRegs 4th arg is emitted by the WAM compiler for bagof/setof
% (M10) to support ISO grouping. The LLVM runtime does not use witness
% grouping yet -- ignore the witness string and fall through to the
% same dispatch as the 3-arg form.
wam_line_to_llvm_literal(["begin_aggregate", TypeStr, ValRegStr, ResRegStr], Lit) :- !,
    begin_aggregate_lit(TypeStr, ValRegStr, ResRegStr, Lit).
wam_line_to_llvm_literal(["begin_aggregate", TypeStr, ValRegStr, ResRegStr, _Witness], Lit) :- !,
    begin_aggregate_lit(TypeStr, ValRegStr, ResRegStr, Lit).

begin_aggregate_lit(TypeStr, ValRegStr, ResRegStr, Lit) :-
    clean_comma(TypeStr, CT), clean_comma(ValRegStr, CV), clean_comma(ResRegStr, CR),
    atom_string(CTAtom, CT),
    agg_type_id(CTAtom, TypeId),
    atom_string(CVAtom, CV), reg_name_to_index(CVAtom, ValIdx),
    atom_string(CRAtom, CR), reg_name_to_index(CRAtom, ResIdx),
    % Pack op2 = (val_idx << 16) | res_idx
    Op2 is (ValIdx << 16) \/ ResIdx,
    format(atom(Lit), '%Instruction { i32 28, i64 ~w, i64 ~w }', [TypeId, Op2]).

% end_aggregate ValueReg
wam_line_to_llvm_literal(["end_aggregate", ValRegStr], Lit) :- !,
    clean_comma(ValRegStr, CV),
    atom_string(CVAtom, CV),
    reg_name_to_index(CVAtom, ValIdx),
    format(atom(Lit), '%Instruction { i32 29, i64 ~w, i64 0 }', [ValIdx]).

% agg_type_id(+Name, -Id): pack aggregation operator name to integer id.
agg_type_id(sum, 0).
agg_type_id(count, 1).
agg_type_id(min, 2).
agg_type_id(max, 3).
agg_type_id(collect, 4).
agg_type_id(bag, 5).
% M10: set / setof / bagof -> sort + dedup of the accumulator. The
% bagof case keeps duplicates; only set/setof dedup. Use the same id
% (6) for both -- they share the sort+build path, dedup is a runtime
% switch driven by which agg_type the bytecode sets.
agg_type_id(set, 6).
agg_type_id(setof, 6).
agg_type_id(bagof, 5).  % bagof keeps duplicates, same as bag
agg_type_id(_, 4).  % fallback: collect

% call_foreign KindName, InstanceId
% Dispatches to a registered native foreign kernel. The kind name is
% resolved via wam_llvm_foreign_kind_id/2 to a stable integer ID that
% matches the first-level switch in @wam_execute_foreign_predicate.
% InstanceId is a compile-time-unique discriminator within a kind —
% the second-level switch inside the per-kind impl (e.g.
% @wam_td3_kernel_impl) uses it to pick the right per-predicate
% edge table. Arity is implicit per kind and is not in the
% instruction (M5.8 change — op2 used to carry arity).
wam_line_to_llvm_literal(["call_foreign", KindStr, InstanceStr], Lit) :- !,
    clean_comma(KindStr, CK), clean_comma(InstanceStr, CI),
    atom_string(KAtom, CK),
    ( wam_llvm_foreign_kind_id(KAtom, KindId)
    -> true
    ;  KindId = 999  % sentinel for unknown — dispatch returns false
    ),
    ( number_string(Instance, CI) -> true ; Instance = 0 ),
    format(atom(Lit), '%Instruction { i32 30, i64 ~w, i64 ~w }', [KindId, Instance]).

%% wam_llvm_foreign_kind_id(+Kind, -Id)
%  Map a foreign kernel kind name (atom) to its integer dispatch ID.
%  The IDs must stay in sync with the switch cases in
%  @wam_execute_foreign_predicate in state.ll.mustache.
%  M3 establishes the IDs; M5 will fill in the actual kernel bodies.
wam_llvm_foreign_kind_id(category_ancestor,        0).
wam_llvm_foreign_kind_id(countdown_sum2,           1).
wam_llvm_foreign_kind_id(list_suffix2,             2).
wam_llvm_foreign_kind_id(transitive_closure2,      3).
wam_llvm_foreign_kind_id(transitive_distance3,     4).
wam_llvm_foreign_kind_id(weighted_shortest_path3,  5).
wam_llvm_foreign_kind_id(astar_shortest_path4,     6).
% call, execute, try_me_else, retry_me_else are handled by
% wam_line_to_llvm_literal_resolved/3 (label resolution required).
wam_line_to_llvm_literal(["proceed"], '%Instruction { i32 20, i64 0, i64 0 }').
wam_line_to_llvm_literal(["builtin_call", Op, N], Lit) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    ( atom(COp) -> COpAtom = COp ; atom_string(COpAtom, COp) ),
    builtin_op_to_id(COpAtom, OpId),
    format(atom(Lit), '%Instruction { i32 21, i64 ~w, i64 ~w }', [OpId, Num]).
wam_line_to_llvm_literal(["trust_me"], '%Instruction { i32 24, i64 0, i64 0 }').

% Switch instructions: handled in the label-resolved variant (need LabelMap).
% The /2 form only sees them without labels, so produce nop fallthrough.
% The real path goes through wam_line_to_llvm_literal_resolved/3 below.
wam_line_to_llvm_literal(["switch_on_constant"|_],
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough (no LabelMap here)
wam_line_to_llvm_literal(["switch_on_structure"|_],
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough
wam_line_to_llvm_literal(["switch_on_constant_a2"|_],
    '%Instruction { i32 27, i64 0, i64 0 }').  % nop fallthrough
wam_line_to_llvm_literal(["switch_on_term"|_],
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough
wam_line_to_llvm_literal(["switch_on_term_a2"|_],
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough
wam_line_to_llvm_literal(["try"|_],
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough
wam_line_to_llvm_literal(["retry"|_],
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough
wam_line_to_llvm_literal(["trust"|_],
    '%Instruction { i32 26, i64 0, i64 0 }').  % nop fallthrough

wam_line_to_llvm_literal(["cut_ite"],
    '%Instruction { i32 31, i64 0, i64 0 }').

% M17: get_level Y_n / cut Y_n -- proper soft cut for if-then-else.
wam_line_to_llvm_literal(["get_level", Yn], Lit) :-
    clean_comma(Yn, CYn),
    atom_string(CYnAtom, CYn),
    reg_name_to_index(CYnAtom, YnIdx),
    format(atom(Lit), '%Instruction { i32 33, i64 ~w, i64 0 }', [YnIdx]).
wam_line_to_llvm_literal(["cut", Yn], Lit) :-
    clean_comma(Yn, CYn),
    atom_string(CYnAtom, CYn),
    reg_name_to_index(CYnAtom, YnIdx),
    format(atom(Lit), '%Instruction { i32 34, i64 ~w, i64 0 }', [YnIdx]).

% jump without a label map errors — label-referencing. Text parser that
% produces label-free literals is only used for the non-resolved path
% (unit-test fixtures). Throw so the bug is loud.
wam_line_to_llvm_literal(["jump", _], _) :-
    throw(error(label_resolution_required(jump, "provide a LabelMap"),
                _)).

wam_line_to_llvm_literal(["switch_on_constant_fallthrough" | _],
    '%Instruction { i32 26, i64 0, i64 0 }') :- !.
wam_line_to_llvm_literal(["switch_on_constant_a2_fallthrough" | _],
    '%Instruction { i32 27, i64 0, i64 0 }') :- !.

wam_line_to_llvm_literal(Parts, Lit) :-
    atomic_list_concat(Parts, " ", Line),
    format(atom(Lit), '; TODO: ~w', [Line]).

% --- Utility predicates ---

clean_comma(S, Clean) :-
    (   sub_string(S, Before, 1, 0, ",")
    ->  sub_string(S, 0, Before, 1, Clean)
    ;   Clean = S
    ).

llvm_pack_value_str(Str, Packed) :-
    (   number_string(N, Str)
    ->  Packed = N
    ;   strip_quoted_atom(Str, Unquoted),
        atom_string(A, Unquoted),
        llvm_pack_value(atom(A), Packed)
    ).

%% strip_quoted_atom(+S, -Inner)
%
%  If S is a single-quoted atom representation, return the inner string
%  with WAM quote escapes un-escaped. Otherwise return S unchanged.
strip_quoted_atom(Str, Inner) :-
    string_length(Str, L),
    ( L >= 2,
      sub_string(Str, 0, 1, _, "'"),
      sub_string(Str, _, 1, 0, "'")
    -> InnerLen is L - 2,
       sub_string(Str, 1, InnerLen, 1, Body),
       % WAM quoting uses backslash escapes; accept doubled apostrophes
       % as a legacy tokenizer convention too.
       string_codes(Body, BodyCodes),
       unescape_wam_quoted_codes(BodyCodes, UnescCodes),
       string_codes(Inner, UnescCodes)
    ;  Inner = Str
    ).
unescape_wam_quoted_codes([], []).
unescape_wam_quoted_codes([92, 92 | Cs], [92 | Rest]) :- !,
    unescape_wam_quoted_codes(Cs, Rest).
unescape_wam_quoted_codes([92, 39 | Cs], [39 | Rest]) :- !,
    unescape_wam_quoted_codes(Cs, Rest).
unescape_wam_quoted_codes([39, 39 | Cs], [39 | Rest]) :- !,
    unescape_wam_quoted_codes(Cs, Rest).
unescape_wam_quoted_codes([C | Cs], [C | Rest]) :-
    unescape_wam_quoted_codes(Cs, Rest).

% NOTE: we don't take a %WamState param — the function creates its own vm
% via wam_state_new() in the entry block. Taking %vm as a param conflicted
% with the local %vm created inside the body.
build_llvm_param_list(0, "") :- !.
build_llvm_param_list(Arity, ParamList) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(format(atom(S), "%Value %a~w", [I])), Indices, Parts),
    atomic_list_concat(Parts, ', ', ParamList).

build_llvm_arg_setup(0, "") :- !.
build_llvm_arg_setup(Arity, Setup) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(
        RegIdx is I - 1,
        format(atom(S),
            '  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %a~w)',
            [RegIdx, I])
    ), Indices, Parts),
    atomic_list_concat(Parts, '\n', Setup).

% ============================================================================
% PHASE 5 (JIT): Runtime-loadable WAM objects (.wamo)
% ============================================================================
%
% A .wamo object is a self-contained, position-independent WAM program that
% a host binary can load at RUNTIME and execute on the shared WAM runtime.
% The host is any module produced by write_wam_llvm_project/3 with the
% wam_object_support_ir/1 loader appended (option emit_wamo_loader(true)).
% This is UnifyWeaver's JIT slice-1: grammars compiled ahead of time and
% swapped in without recompiling the host.
%
% Format -- a whitespace-separated token stream. Strings are length-prefixed
% ("<bytelen> <raw bytes>") so the native loader needs no escaping:
%
%     WAMO\n
%     <version=1>
%     <default-entry-label-index>
%     <entry-count E>     then E records of (name-string, label-index) --
%                         the named entries a multi-entry object exposes;
%                         placed early so @wam_object_load skips it cheaply
%     <atom-count N>      then N length-prefixed strings
%     <functor-count M>   then M length-prefixed strings
%     <label-count L>     then L ints (PC per label; index = position)
%     <code-count C>      then C * 4 ints (tag op1 op2 reloc)
%
% Relocation classes (reloc field, per instruction):
%     0 none     - op1/op2 verbatim
%     1 atom     - op1 is an index into the atom table; the loader re-interns
%                  atoms[op1] via @wam_intern_atom (which copies the string)
%                  and writes the runtime atom id back into op1
%     2 functor  - op1 is an index into the functor table; the loader makes
%                  ONE malloc'd NUL-terminated copy per functor and writes its
%                  pointer into op1. Pointer identity within the object keeps
%                  get/put_structure unification correct (functor comparison
%                  is by pointer); arithmetic is content-based so the copy is
%                  also fine for `is`/comparison operators.
%     3 float    - op1 is an index into the C-string table (shared with
%                  functor names); the loader strtod's the stored decimal
%                  text and writes its i64 bit pattern into op1 -- the same
%                  double the AOT `bitcast (double c to i64)` yields. Used by
%                  put_constant / set_constant with a tag=2 float literal.
%
% call/execute/try_me_else/retry_me_else operands are indexes into the
% object's OWN labels array -- @wam_label_pc reads the VM's installed labels
% array, so they are already self-relative. builtin_call ids are host-stable
% per UnifyWeaver version. switch-on-constant tables and call/N meta-calls
% are still outside the loadable subset and rejected at write time.

%% write_wam_object(+Predicates, +Options, +OutFile) is det.
%
%  Compile Predicates (plus their same-module helper closure) to a .wamo
%  object and write it to OutFile. Options:
%    wamo_entry(Pred/Arity)    - default entry (bare dyncall); defaults to the
%                                first of wamo_entries, else the first predicate
%    wamo_entries([P/A, ...])  - the named entries this object exposes (for
%                                multi-entry objects resolved by name at the
%                                call site); the first is also the default entry
%  Throws error(wamo_unsupported(_), _) if any instruction is outside the
%  loadable subset, error(wamo_entry_not_found(_), _) if the entry label
%  is missing, or error(wamo_uncompilable(_), _) on a WAM compile failure.
write_wam_object(Predicates, Options, OutFile) :-
    wam_object_encode(Predicates, Options, Codes),
    setup_call_cleanup(
        open(OutFile, write, S, [encoding(octet)]),
        format(S, "~s", [Codes]),
        close(S)).

%% wam_object_encode(+Predicates, +Options, -Codes) is det.
%
%  Produce the .wamo byte stream (a list of 0..255 codes) in memory.
wam_object_encode(Predicates, Options0, Codes) :-
    % Lower bagof/setof through the aggregate path (begin_aggregate/
    % end_aggregate) rather than as calls to a setof/3 library predicate --
    % the same default write_wam_llvm_project uses. Without it, closure
    % expansion tries to compile the (clause-less) setof/3 and the object
    % fails to build; with it, and now that the aggregate opcodes are in the
    % loadable subset, setof/bagof load and run like findall.
    ( memberchk(inline_bagof_setof(_), Options0)
    -> Options = Options0
    ;  Options = [inline_bagof_setof(true) | Options0]
    ),
    expand_wam_llvm_project_predicates(Predicates, Options, Expanded),
    wamo_classify(Expanded, Options, 0, Records),
    build_merged_labels(Records, NamePCPairs, LabelMap),
    wamo_entry_index(Predicates, Options, LabelMap, EntryIndex),
    wamo_entries_list(Predicates, Options, LabelMap, Entries),
    wamo_all_instr_parts(Records, AllParts),
    foldl(wamo_encode_step(LabelMap), AllParts,
          ws([], 0, tab([], 0), tab([], 0), []),
          ws(RevEnc, _, ATab0, FTab, RevSw)),
    reverse(RevEnc, EncInstrs),
    reverse(RevSw, SwTabs),
    % A meta-call table is only needed if the object actually contains a
    % call/N meta-call (op1 = -1); interning predicate names for objects that
    % never meta-call would bloat them for no benefit (pay-for-what-you-use).
    (   wamo_has_meta_call(EncInstrs)
    ->  wamo_meta_rows(LabelMap, ATab0, ATab, FTab, MetaRows)
    ;   ATab = ATab0, MetaRows = []
    ),
    wamo_table_list(ATab, Atoms),
    wamo_table_list(FTab, Functors),
    findall(PC, member(_-PC, NamePCPairs), PCs),
    wamo_serialize(EntryIndex, Entries, Atoms, Functors, PCs, EncInstrs,
                   MetaRows, SwTabs, Codes).

%% wamo_has_meta_call(+EncInstrs) is semidet.
%  True if any encoded call/execute dispatches a goal through the object's
%  meta-call table: a plain meta-call (op1 = -1), or catch/3 (op1 = -5) whose
%  Goal and Recovery are meta-dispatched, or throw/1 (op1 = -6) whose recovery
%  runs via the same path. Without the table @wam_dispatch_meta_call has
%  nothing to resolve against and the goal fails. See PLAWK_DYNAMIC_DB.md.
wamo_has_meta_call(EncInstrs) :-
    member(enc(T, Op1, _, _), EncInstrs),
    ( T =:= 18 ; T =:= 19 ),
    ( Op1 =:= -1 ; Op1 =:= -5 ; Op1 =:= -6 ), !.

%% wamo_meta_rows(+LabelMap, +ATab0, -ATab1, +FTab, -Rows)
%  Build the object's meta-call dispatch rows: one per predicate head label
%  (Name/Arity), mirroring the host's @wam_dispatch_meta_call table but with
%  indices into the object's OWN atom/functor tables so the loader can
%  resolve them after relocation. Each row is
%      row(AtomIdx, FunIdx, Arity, LabelIdx)
%  where AtomIdx interns the predicate name (extending ATab), FunIdx is the
%  functor-table index of that name (or -1 if the object builds no such
%  compound), and LabelIdx is the object-relative label of the predicate.
wamo_meta_rows(LabelMap, ATab0, ATab1, tab(FPairs, _), Rows) :-
    findall(mpred(NameStr, Arity, LabelIdx),
        ( member(Label-LabelIdx, LabelMap),
          catch(split_functor_arity(Label, Name0, Arity), _, fail),
          Arity >= 0,
          % split_functor_arity returns an atom; the atom/functor interning
          % tables key on strings, so coerce or dedup fails and the functor
          % lookup (needed for compound meta-calls) silently misses.
          wamo_to_string(Name0, NameStr)
        ),
        Preds0),
    sort(Preds0, Preds),
    foldl(wamo_meta_row_step(FPairs), Preds, ATab0-[], ATab1-RevRows),
    reverse(RevRows, Rows).

wamo_meta_row_step(FPairs, mpred(NameStr, Arity, LabelIdx),
                   A0-R0, A1-[row(AtomIdx, FunIdx, Arity, LabelIdx)|R0]) :-
    tab_intern(NameStr, A0, AtomIdx, A1),
    ( wamo_tab_lookup(FPairs, NameStr, FI) -> FunIdx = FI ; FunIdx = -1 ).

%% wamo_classify(+Preds, +Options, +StartPC, -Records)
%
%  Like pass1_classify_predicates/4 but WAM-only: every predicate is
%  compiled to bytecode (no native lowering, no foreign kernels) so the
%  object carries real instructions. StartPC threads through all records.
wamo_classify([], _, _, []).
wamo_classify([PI|Rest], Options, StartPC, [Rec|Recs]) :-
    ( PI = M:P/A -> true ; PI = P/A, M = user ),
    (   catch(wam_target:compile_predicate_to_wam(M:P/A, Options, WamRaw), _, fail)
    ->  parse_wam_to_pass1(P/A, WamRaw, A, StartPC, Rec, NumInstrs),
        NextPC is StartPC + NumInstrs
    ;   throw(error(wamo_uncompilable(M:P/A),
            context(write_wam_object,
                'predicate has no WAM-compilable clauses')))
    ),
    wamo_classify(Rest, Options, NextPC, Recs).

%% wamo_default_entry_pi(+Predicates, +Options, -Pred/Arity)
%  The default entry (used by a bare dyncall): wamo_entry(P/A) if given,
%  else the first of wamo_entries([...]), else the first predicate.
wamo_default_entry_pi(_Predicates, Options, P/A) :-
    option(wamo_entry(P/A), Options), !.
wamo_default_entry_pi(_Predicates, Options, P/A) :-
    option(wamo_entries([P/A | _]), Options), !.
wamo_default_entry_pi(Predicates, _Options, P/A) :-
    Predicates = [First|_],
    ( First = _:P/A -> true ; First = P/A ).

%% wamo_entry_index(+Predicates, +Options, +LabelMap, -Index)
%  The default entry's label index.
wamo_entry_index(Predicates, Options, LabelMap, Index) :-
    wamo_default_entry_pi(Predicates, Options, P/A),
    wamo_lookup_entry_index(P/A, LabelMap, Index).

%% wamo_entries_list(+Predicates, +Options, +LabelMap, -Entries)
%  The named-entry table: Name-Index pairs for every exposed entry. From
%  wamo_entries([...]) if given, else just the default entry. Names are
%  the "Pred/Arity" label form so the loader can resolve dyncall@name.
wamo_entries_list(Predicates, Options, LabelMap, Entries) :-
    (   option(wamo_entries(List), Options)
    ->  EntryPIs = List
    ;   wamo_default_entry_pi(Predicates, Options, DP),
        EntryPIs = [DP]
    ),
    findall(Name-Index,
        ( member(EP/EA, EntryPIs),
          format(string(Name), "~w/~w", [EP, EA]),
          wamo_lookup_entry_index(EP/EA, LabelMap, Index)
        ),
        Entries).

wamo_lookup_entry_index(P/A, LabelMap, Index) :-
    format(string(Name), "~w/~w", [P, A]),
    (   memberchk(Name-Index, LabelMap)
    ->  true
    ;   throw(error(wamo_entry_not_found(Name),
            context(write_wam_object,
                'entry predicate has no label in the merged object')))
    ).

%% wamo_all_instr_parts(+Records, -AllParts)
%  Concatenate every record's parsed instruction Parts in emission order.
wamo_all_instr_parts([], []).
wamo_all_instr_parts([Rec|Rest], All) :-
    ( Rec = wam(_, _, RawInstrs, _, _, _) -> true ; RawInstrs = [] ),
    wamo_all_instr_parts(Rest, RestAll),
    append(RawInstrs, RestAll, All).

% --- object-local string interning tables (string -> dense index) ---------

tab_intern(Str, tab(P0, N0), Idx, tab(P1, N1)) :-
    (   wamo_tab_lookup(P0, Str, I)
    ->  Idx = I, P1 = P0, N1 = N0
    ;   Idx = N0, N1 is N0 + 1, P1 = [N0-Str | P0]
    ).

wamo_tab_lookup([I-S | _], Str, I) :- S == Str, !.
wamo_tab_lookup([_ | T], Str, I) :- wamo_tab_lookup(T, Str, I).

wamo_table_list(tab(Pairs, _), List) :-
    keysort(Pairs, Sorted),
    findall(S, member(_-S, Sorted), List).

% --- per-instruction encoding ---------------------------------------------

% The fold threads the instruction PC (one WAM line = one row = one PC
% slot, the invariant the label map depends on) and a switch-table
% accumulator. switch_on_constant / _a2 encode as REAL dispatch rows
% (tags 25 / 27 -- the same step cases the AOT dispatcher uses) with
% op1/op2 left 0; their key->label tables ride a TRAILING SECTION (like
% the meta-call table) keyed by this PC, so the code/PC/label layout is
% untouched and the loader patches op1 = table pointer, op2 = count.
% The *_fallthrough variants and switch_on_term / _structure stay nops,
% exactly matching the AOT dispatcher.
wamo_encode_step(LabelMap, Parts, ws(E0, PC, A0, F0, S0),
                 ws([Enc|E0], PC1, A1, F1, S1)) :-
    PC1 is PC + 1,
    (   Parts = [Op | EntryParts],
        wamo_switch_tag(Op, Tag)
    ->  wamo_switch_entries(EntryParts, LabelMap, A0, A1, SwEntries),
        F1 = F0,
        Enc = enc(Tag, 0, 0, none),
        S1 = [sw(PC, SwEntries) | S0]
    ;   wamo_enc(Parts, LabelMap, A0, A1, F0, F1, Enc),
        S1 = S0
    ).

wamo_switch_tag("switch_on_constant", 25).
wamo_switch_tag("switch_on_constant_a2", 27).

%% wamo_switch_entries(+Parts, +LabelMap, +A0, -A1, -Entries)
%  Parse "key:label" switch entries into se(Kind, Key, LabelIdx) rows.
%  Kind 1 = integer key (Key is the value); Kind 0 = atom key (Key is an
%  ATOM-TABLE index -- the loader resolves it through the same relocated
%  atomIds array as any reloc-1 operand, so the runtime payload compares
%  equal to a relocated get_constant/A-register atom id). "default" and
%  unresolvable labels encode as -1, the fall-through sentinel
%  @wam_switch_on_constant already maps to "advance PC".
wamo_switch_entries([], _, A, A, []).
wamo_switch_entries([Part | Rest], LabelMap, A0, A,
                    [se(Kind, Key, LabelIdx) | Es]) :-
    clean_comma(Part, Clean),
    switch_entry_split(Clean, KeyStr0, LabelStr),
    switch_entry_unquote(KeyStr0, KeyStr),
    (   number_string(N, KeyStr), integer(N)
    ->  Kind = 1, Key = N, A1 = A0
    ;   Kind = 0, tab_intern(KeyStr, A0, Key, A1)
    ),
    (   LabelStr == "default"
    ->  LabelIdx = -1
    ;   lookup_label_index(LabelStr, LabelMap, LabelIdx)
    ->  true
    ;   LabelIdx = -1
    ),
    wamo_switch_entries(Rest, LabelMap, A1, A, Es).

%% switch_entry_split(+Part, -KeyStr, -LabelStr)
%  Split "key:label" at the LAST colon: labels never contain colons, but
%  quoted atom keys can ('=:=' -- a first-colon split would shear the
%  key in half and treat its remainder as the label).
switch_entry_split(Clean, KeyStr, LabelStr) :-
    string_length(Clean, Len),
    (   between(1, Len, Back),
        Pos is Len - Back,
        sub_string(Clean, Pos, 1, _, ":")
    ->  sub_string(Clean, 0, Pos, _, KeyStr),
        P1 is Pos + 1,
        sub_string(Clean, P1, _, 0, LabelStr)
    ;   KeyStr = Clean, LabelStr = ""
    ).

%% switch_entry_unquote(+KeyStr0, -KeyStr)
%  The tier-2 compiler writes atom keys writeq-style: atoms needing
%  quotes arrive as '...' with backslash escapes doubled ('=\\=').
%  Strip the quotes and undo the escapes so the interned key matches
%  the runtime atom exactly -- a still-quoted key would intern the
%  quote characters into the name and the switch entry would NEVER
%  match at dispatch time (a silent no-match means the call FAILS,
%  since a strict switch backtracks on miss).
switch_entry_unquote(KeyStr0, KeyStr) :-
    (   string_concat("'", R1, KeyStr0),
        string_concat(Body, "'", R1)
    ->  string_codes(Body, Cs),
        switch_unescape_codes(Cs, OutCs),
        string_codes(KeyStr, OutCs)
    ;   KeyStr = KeyStr0
    ).

switch_unescape_codes([], []).
switch_unescape_codes([0'\\, C | R], [C | T]) :- !,
    switch_unescape_codes(R, T).
switch_unescape_codes([C | R], [C | T]) :-
    switch_unescape_codes(R, T).

%% wamo_enc(+Parts, +LabelMap, +A0, -A1, +F0, -F1, -enc(Tag,Op1,Op2,Reloc))
%  Encode one parsed WAM instruction into the object's triple form,
%  threading the atom (A) and functor (F) interning tables.

% constants. put_constant / set_constant support float literals (tag 2),
% matching AOT set_constant_literal_parts; get_constant / unify_constant
% stay integer/atom only (AOT itself never emits a float there).
wamo_enc(["get_constant", C, Ai], _, A0, A1, F0, F1, enc(0, Op1, Op2, R)) :- !,
    clean_comma(C, CC), clean_comma(Ai, CAi),
    atom_string(CAiA, CAi), reg_name_to_index(CAiA, Reg),
    wamo_const(no_float, CC, A0, A1, F0, F1, Op1, Tag, R),
    Op2 is (Tag << 16) \/ Reg.
wamo_enc(["unify_constant", C], _, A0, A1, F0, F1, enc(7, Op1, Op2, R)) :- !,
    clean_comma(C, CC),
    wamo_const(no_float, CC, A0, A1, F0, F1, Op1, Tag, R),
    Op2 is Tag << 16.
wamo_enc(["put_constant", C, Ai], _, A0, A1, F0, F1, enc(8, Op1, Op2, R)) :- !,
    clean_comma(C, CC), clean_comma(Ai, CAi),
    atom_string(CAiA, CAi), reg_name_to_index(CAiA, Reg),
    wamo_const(float, CC, A0, A1, F0, F1, Op1, Tag, R),
    Op2 is (Tag << 16) \/ Reg.
wamo_enc(["set_constant", C], _, A0, A1, F0, F1, enc(15, Op1, Tag, R)) :- !,
    clean_comma(C, CC),
    wamo_const(float, CC, A0, A1, F0, F1, Op1, Tag, R).

% structures
wamo_enc(["get_structure", FN, Ai], _, A, A, F0, F1, enc(3, Idx, Op2, functor)) :- !,
    wamo_structure(FN, Ai, F0, F1, Idx, Op2).
wamo_enc(["put_structure", FN, Ai], _, A, A, F0, F1, enc(11, Idx, Op2, functor)) :- !,
    wamo_structure(FN, Ai, F0, F1, Idx, Op2).

% two-register
wamo_enc(["get_variable", Xn, Ai], _, A, A, F, F, enc(1, X, Y, none)) :- !, wamo_reg2(Xn, Ai, X, Y).
wamo_enc(["get_value",    Xn, Ai], _, A, A, F, F, enc(2, X, Y, none)) :- !, wamo_reg2(Xn, Ai, X, Y).
wamo_enc(["put_variable", Xn, Ai], _, A, A, F, F, enc(9, X, Y, none)) :- !, wamo_reg2(Xn, Ai, X, Y).
wamo_enc(["put_value",    Xn, Ai], _, A, A, F, F, enc(10, X, Y, none)) :- !, wamo_reg2(Xn, Ai, X, Y).

% one-register
wamo_enc(["get_list", Ai], _, A, A, F, F, enc(4, R, 0, none)) :- !, wamo_reg1(Ai, R).
wamo_enc(["put_list", Ai], _, A, A, F, F, enc(12, R, 0, none)) :- !, wamo_reg1(Ai, R).
wamo_enc(["unify_variable", Xn], _, A, A, F, F, enc(5, R, 0, none)) :- !, wamo_reg1(Xn, R).
wamo_enc(["unify_value", Xn], _, A, A, F, F, enc(6, R, 0, none)) :- !, wamo_reg1(Xn, R).
wamo_enc(["set_variable", Xn], _, A, A, F, F, enc(13, R, 0, none)) :- !, wamo_reg1(Xn, R).
wamo_enc(["set_value", Xn], _, A, A, F, F, enc(14, R, 0, none)) :- !, wamo_reg1(Xn, R).
wamo_enc(["get_level", Yn], _, A, A, F, F, enc(33, R, 0, none)) :- !, wamo_reg1(Yn, R).
wamo_enc(["cut", Yn], _, A, A, F, F, enc(34, R, 0, none)) :- !, wamo_reg1(Yn, R).

% nullary
wamo_enc(["allocate"],   _, A, A, F, F, enc(16, 0, 0, none)) :- !.
wamo_enc(["deallocate"], _, A, A, F, F, enc(17, 0, 0, none)) :- !.
wamo_enc(["proceed"],    _, A, A, F, F, enc(20, 0, 0, none)) :- !.
wamo_enc(["trust_me"],   _, A, A, F, F, enc(24, 0, 0, none)) :- !.
wamo_enc(["cut_ite"],    _, A, A, F, F, enc(31, 0, 0, none)) :- !.

% builtins (ids host-stable per UnifyWeaver version)
wamo_enc(["builtin_call", Op, N], _, A, A, F, F, enc(21, Id, Num, none)) :- !,
    clean_comma(Op, COp), clean_comma(N, CN),
    ( number_string(Num, CN) -> true ; Num = 0 ),
    ( atom(COp) -> OpA = COp ; atom_string(OpA, COp) ),
    builtin_op_to_id(OpA, Id).

% control-flow with label operands (self-relative indexes into the object)
% call/execute with op1 = -1 is a call/N meta-call (goal built at runtime);
% op2 carries the total arity and @wam_dispatch_meta_call resolves the goal
% through the loaded object's own meta-call table (see PLAWK_EVAL_BOOTSTRAP.md
% milestone 2). A static target resolves its label index as usual.
wamo_enc(["call", P, N], LabelMap, A, A, F, F, enc(18, Idx, Arity, none)) :- !,
    clean_comma(P, CP), clean_comma(N, CN),
    ( number_string(Arity, CN) -> true ; Arity = 0 ),
    (   wam_special_call_label(CP, Idx, _)
    ->  true
    ;   wam_meta_call_label(CP)
    ->  Idx = -1
    ;   wamo_label(CP, LabelMap, Idx)
    ).
wamo_enc(["execute", P], LabelMap, A, A, F, F, enc(19, Idx, Op2, none)) :- !,
    clean_comma(P, CP),
    (   wam_special_call_label(CP, Idx, Op2)
    ->  true
    ;   wam_meta_call_label(CP)
    ->  Idx = -1, wam_meta_call_total_arity(CP, Op2)
    ;   wamo_label(CP, LabelMap, Idx), Op2 = 0
    ).
wamo_enc(["try_me_else", L], LabelMap, A, A, F, F, enc(22, Idx, 0, none)) :- !,
    clean_comma(L, CL), lookup_label_index(CL, LabelMap, Idx).
wamo_enc(["retry_me_else", L], LabelMap, A, A, F, F, enc(23, Idx, 0, none)) :- !,
    clean_comma(L, CL), lookup_label_index(CL, LabelMap, Idx).
% jump L: unconditional branch (emitted by if-then-else to the continuation
% after the then/else arm). Self-relative label index, like try_me_else.
wamo_enc(["jump", L], LabelMap, A, A, F, F, enc(32, Idx, 0, none)) :- !,
    clean_comma(L, CL), lookup_label_index(CL, LabelMap, Idx).

% Aggregate control: findall / bagof / setof / aggregate_all. The tier-2
% compiler brackets the collected goal with begin_aggregate/end_aggregate
% (tags 28/29). Both operate purely on the VM's own choice-point stack,
% registers, trail and aggregate accumulator -- no relocation, no
% host-compile-time state -- and the shared @run_loop already dispatches
% these tags, so a loaded object runs them exactly as the host does (no
% loader change needed). op1 = agg-type id, op2 = (val_reg << 16) | res_reg,
% mirroring begin_aggregate_lit/4. The 4-argument begin form carries a
% setof/bagof witness that the encoding ignores, exactly as the AOT path does.
wamo_enc(["begin_aggregate", TypeStr, ValRegStr, ResRegStr | _], _, A, A, F, F,
         enc(28, TypeId, Op2, none)) :- !,
    clean_comma(TypeStr, CT), clean_comma(ValRegStr, CV), clean_comma(ResRegStr, CR),
    atom_string(CTAtom, CT), agg_type_id(CTAtom, TypeId),
    atom_string(CVAtom, CV), reg_name_to_index(CVAtom, ValIdx),
    atom_string(CRAtom, CR), reg_name_to_index(CRAtom, ResIdx),
    Op2 is (ValIdx << 16) \/ ResIdx.
wamo_enc(["end_aggregate", ValRegStr], _, A, A, F, F, enc(29, ValIdx, 0, none)) :- !,
    clean_comma(ValRegStr, CV), atom_string(CVAtom, CV), reg_name_to_index(CVAtom, ValIdx).

% Indexing dispatch: nop fallthrough (tag 26), matching the interpreter's
% runtime semantics. Every indexing instruction the tier-2 compiler emits
% sits INLINE at the head of the predicate, immediately before the
% try_me_else / retry_me_else clause chain (see the layout in
% PLAWK_EVAL_BOOTSTRAP.md); the switch entries only point deeper into that
% same chain. So dropping the switch and falling through to try_me_else
% runs every clause in order -- correct, just unindexed. This includes
% switch_on_constant / _a2 (first/second-argument constant indexing): a
% loaded object runs them unindexed, exactly as it already does for
% switch_on_term. Lifting these is the first subset-expansion step toward
% the eval bootstrap (item 5) -- the WAM compiler's own predicates lean
% heavily on atom-keyed clause indexing.
% The remaining switch_on_* variants stay nops -- switch_on_constant and
% switch_on_constant_a2 are intercepted upstream (wamo_encode_step) and
% encode as REAL indexed dispatch with a trailing-section table. What
% reaches here: the *_fallthrough forms and switch_on_term / _structure,
% which the AOT dispatcher also runs as nop fallthroughs, so loaded
% semantics stay exactly host semantics.
wamo_enc([Op | _], _, A, A, F, F, enc(26, 0, 0, none)) :-
    string_concat("switch_on_", _, Op), !.
wamo_enc(["try" | _],   _, A, A, F, F, enc(26, 0, 0, none)) :- !.
wamo_enc(["retry" | _], _, A, A, F, F, enc(26, 0, 0, none)) :- !.
wamo_enc(["trust" | _], _, A, A, F, F, enc(26, 0, 0, none)) :- !.

% everything else is outside the loadable WAM object subset
wamo_enc(Parts, _, _, _, _, _, _) :-
    throw(error(wamo_unsupported(instruction(Parts)),
        context(write_wam_object,
            'instruction is outside the loadable WAM object subset'))).

%% wamo_const(+AllowFloat, +Str, +A0, -A1, +F0, -F1, -Op1, -Tag, -Reloc)
%  Encode a constant operand:
%    integer      -> Tag=1, Op1=value, reloc=none.
%    atom         -> Tag=0, reloc=atom, Op1=atom-table index (re-interned
%                    by the loader).
%    float        -> Tag=2, reloc=float, Op1=index into the C-string table
%                    (shared with functor names); the loader strtod's the
%                    stored decimal text and stores its i64 bit pattern.
%                    Only when AllowFloat==float (put_constant/set_constant,
%                    matching AOT); elsewhere a float is rejected.
%  The decimal text is stored verbatim so strtod reproduces exactly the
%  double the AOT `bitcast (double <c> to i64)` would have.
wamo_const(AllowFloat, CC, A0, A1, F0, F1, Op1, Tag, R) :-
    (   number_string(N, CC)
    ->  (   integer(N)
        ->  Op1 = N, Tag = 1, R = none, A1 = A0, F1 = F0
        ;   AllowFloat == float
        ->  tab_intern(CC, F0, Idx, F1),
            Op1 = Idx, Tag = 2, R = float, A1 = A0
        ;   throw(error(wamo_unsupported(float_constant(CC)),
                context(write_wam_object,
                    'float constants are only supported in put_constant / set_constant')))
        )
    ;   strip_quoted_atom(CC, Inner),
        wamo_to_string(Inner, InnerStr),
        tab_intern(InnerStr, A0, Idx, A1),
        Op1 = Idx, Tag = 0, R = atom, F1 = F0
    ).

%% wamo_structure(+FN, +Ai, +F0, -F1, -FunctorIdx, -Op2)
wamo_structure(FN, Ai, F0, F1, Idx, Op2) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    atom_string(CAiA, CAi), reg_name_to_index(CAiA, Reg),
    atom_string(CFN, CFNStr),
    split_functor_arity(CFNStr, NameStr0, Arity),
    wamo_to_string(NameStr0, NameStr),
    tab_intern(NameStr, F0, Idx, F1),
    Op2 is (Arity << 16) \/ Reg.

wamo_to_string(X, S) :- ( string(X) -> S = X ; atom_string(X, S) ).

wamo_reg1(S, R) :-
    clean_comma(S, CS), atom_string(CSA, CS), reg_name_to_index(CSA, R).
wamo_reg2(Xn, Ai, X, Y) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXnA, CXn), atom_string(CAiA, CAi),
    reg_name_to_index(CXnA, X), reg_name_to_index(CAiA, Y).

%% wamo_label(+P, +LabelMap, -Idx)
%  Resolve a static call/execute target to its object label index. call/N
%  meta-calls are handled by the caller (op1 = -1) and never reach here; the
%  guard stays as a defensive assertion should that ever change.
wamo_label(CP, _, _) :-
    wam_meta_call_label(CP), !,
    throw(error(wamo_unsupported(meta_call(CP)),
        context(write_wam_object,
            'meta-call reached wamo_label; should have been encoded as op1=-1'))).
wamo_label(CP, LabelMap, Idx) :-
    lookup_label_index(CP, LabelMap, Idx).

% --- serialization to the .wamo byte stream --------------------------------

wamo_reloc_id(none, 0).
wamo_reloc_id(atom, 1).
wamo_reloc_id(functor, 2).
wamo_reloc_id(float, 3).

wamo_serialize(EntryIndex, Entries, Atoms, Functors, PCs, EncInstrs, MetaRows,
               SwTabs, Codes) :-
    length(Entries, NE), length(Atoms, NA), length(Functors, NF),
    length(PCs, NL), length(EncInstrs, NC), length(MetaRows, NM),
    length(SwTabs, NS),
    % header: magic, version, default-entry index, then the named-entry
    % table (early, so @wam_object_load can skip it without a full parse).
    % Version 2 adds a trailing meta-call table section (milestone 2 of the
    % eval bootstrap) so loaded objects can call/N their own predicates.
    format(codes(Head), "WAMO\n2\n~w\n~w\n", [EntryIndex, NE]),
    maplist(wamo_entry_codes, Entries, EntryChunks),
    format(codes(AHdr), "~w\n", [NA]),
    maplist(wamo_string_codes, Atoms, AtomChunks),
    format(codes(FHdr), "~w\n", [NF]),
    maplist(wamo_string_codes, Functors, FunChunks),
    format(codes(LHdr), "~w\n", [NL]),
    maplist([PC, Cs]>>format(codes(Cs), "~w\n", [PC]), PCs, PCChunks),
    format(codes(CHdr), "~w\n", [NC]),
    maplist(wamo_instr_codes, EncInstrs, InstrChunks),
    % meta-call table: <count>\n then <atomIdx> <funIdx> <arity> <labelIdx>\n
    format(codes(MHdr), "~w\n", [NM]),
    maplist(wamo_meta_row_codes, MetaRows, MetaChunks),
    % switch-table section (loaded clause indexing): <count>\n then per
    % table "<pc> <n>\n" and n entry rows "<kind> <key> <labelIdx>\n".
    % Trails the meta table, and is emitted ONLY when non-empty: the
    % loader's wamo_next_int reads 0 at EOF, so switch-free objects stay
    % byte-identical to the pre-section format -- which the bootstrap
    % compiler's serializer (and its byte-exact goldens) still emit.
    (   NS =:= 0
    ->  SwSection = []
    ;   format(codes(SHdr), "~w\n", [NS]),
        maplist(wamo_switch_codes, SwTabs, SwChunks),
        SwSection = [SHdr | SwChunks]
    ),
    append([[Head], EntryChunks, [AHdr], AtomChunks, [FHdr], FunChunks,
            [LHdr], PCChunks, [CHdr], InstrChunks, [MHdr], MetaChunks,
            SwSection], Lists),
    append(Lists, Codes).

%% wamo_switch_codes(+sw(PC, Entries), -Codes)
wamo_switch_codes(sw(PC, Entries), Codes) :-
    length(Entries, N),
    format(codes(Hdr), "~w ~w\n", [PC, N]),
    maplist([se(Kind, Key, LabelIdx), Cs]>>
                format(codes(Cs), "~w ~w ~w\n", [Kind, Key, LabelIdx]),
            Entries, EntryChunks),
    append([Hdr | EntryChunks], Codes).

%% wamo_meta_row_codes(+row(AtomIdx,FunIdx,Arity,LabelIdx), -Codes)
wamo_meta_row_codes(row(AtomIdx, FunIdx, Arity, LabelIdx), Cs) :-
    format(codes(Cs), "~w ~w ~w ~w\n", [AtomIdx, FunIdx, Arity, LabelIdx]).

%% wamo_entry_codes(+Name-Index, -Codes)
%  One named-entry record: length-prefixed name string then its label index.
wamo_entry_codes(Name-Index, Codes) :-
    wamo_string_codes(Name, NameCodes),
    format(codes(IdxCodes), "~w\n", [Index]),
    append(NameCodes, IdxCodes, Codes).

wamo_instr_codes(enc(T, O1, O2, R), Cs) :-
    wamo_reloc_id(R, RI),
    format(codes(Cs), "~w ~w ~w ~w\n", [T, O1, O2, RI]).

%% wamo_string_codes(+Text, -Codes)
%  Emit a length-prefixed byte string: "<bytelen> <utf8 bytes>\n".
wamo_string_codes(Text, Codes) :-
    ( atom(Text) -> atom_codes(Text, Cps)
    ; string(Text) -> string_codes(Text, Cps)
    ; Cps = Text
    ),
    wamo_utf8_bytes(Cps, Bytes),
    length(Bytes, Len),
    format(codes(Hdr), "~w ", [Len]),
    append(Hdr, Bytes, C0),
    append(C0, [0'\n], Codes).

%% wamo_utf8_bytes(+CodePoints, -Bytes)
%  Encode Unicode code points to UTF-8 byte values (all 0..255) so the
%  object stream is byte-exact regardless of the source stream encoding.
wamo_utf8_bytes([], []).
wamo_utf8_bytes([C|Cs], Bytes) :-
    (   C < 0x80
    ->  B = [C]
    ;   C < 0x800
    ->  B0 is 0xC0 \/ (C >> 6),
        B1 is 0x80 \/ (C /\ 0x3F),
        B = [B0, B1]
    ;   C < 0x10000
    ->  B0 is 0xE0 \/ (C >> 12),
        B1 is 0x80 \/ ((C >> 6) /\ 0x3F),
        B2 is 0x80 \/ (C /\ 0x3F),
        B = [B0, B1, B2]
    ;   B0 is 0xF0 \/ (C >> 18),
        B1 is 0x80 \/ ((C >> 12) /\ 0x3F),
        B2 is 0x80 \/ ((C >> 6) /\ 0x3F),
        B3 is 0x80 \/ (C /\ 0x3F),
        B = [B0, B1, B2, B3]
    ),
    wamo_utf8_bytes(Cs, Rest),
    append(B, Rest, Bytes).

% --- native loader + call primitive ---------------------------------------

%% wam_object_support_ir(-IR) is det.
%
%  The runtime support a host module needs to load and run .wamo objects:
%    @wamo_next_int      - parse the next integer from a byte buffer
%    @wam_object_load    - read a .wamo file, relocate, build a %WamState
%    @wam_object_call_i64 - call the object's entry with N %Value args and
%                           read back an Integer result ({value, ok})
%    @wam_object_call_record - call the entry and deserialize a returned
%                           Compound's args into typed i64/f64/string slots
%                           (the "output is a term" primitive; string fields
%                           yield a (ptr,len) pair via out_slots + out_lens)
%    @wam_object_call_assoc - call the entry and materialize a returned list
%                           of [Key-Value] integer pairs into an i64 assoc
%                           table (@wam_assoc_i64_inc per pair)
%    @wam_object_call_posarray - call the entry and materialize a returned
%                           FLAT list [V1..Vn] into an i64 table by POSITION
%                           (element i -> key i, 1-indexed, @wam_assoc_i64_set)
%    @wam_object_call_posarray_str - like call_posarray but the elements are
%                           ATOMS; their registry ids are stored per position
%                           (reads resolve back to text)
%    @wam_object_entry_index / @wam_object_entry_index_bytes
%                        - resolve a named entry to its label index (or -1)
%                          for multi-entry objects; @wam_label_pc turns that
%                          into a PC to call against the loaded VM
%
%  Emitted into a host module when write_wam_llvm_project/3 is called with
%  emit_wamo_loader(true). References only runtime helpers already present
%  in the module (@wam_intern_atom, @wam_state_new, @run_loop, ...) plus
%  libc @open/@read/@close/@malloc/@realloc/@free (already declared).
wam_object_support_ir(
'; --- .wamo loader (Phase 5 JIT slice 1) ---

; Parse the next base-10 integer (optional leading -) from %buf, skipping
; leading whitespace. Returns { value, cursor-past-digits }.
define { i64, i64 } @wamo_next_int(i8* %buf, i64 %size, i64 %cur0) {
entry:
  br label %skip
skip:
  %c = phi i64 [ %cur0, %entry ], [ %c1, %skip_step ]
  %at_end = icmp uge i64 %c, %size
  br i1 %at_end, label %done_zero, label %look
look:
  %p = getelementptr i8, i8* %buf, i64 %c
  %ch = load i8, i8* %p
  %is_sp = icmp eq i8 %ch, 32
  %is_nl = icmp eq i8 %ch, 10
  %is_cr = icmp eq i8 %ch, 13
  %is_tab = icmp eq i8 %ch, 9
  %w1 = or i1 %is_sp, %is_nl
  %w2 = or i1 %is_cr, %is_tab
  %ws = or i1 %w1, %w2
  br i1 %ws, label %skip_step, label %num_init
skip_step:
  %c1 = add i64 %c, 1
  br label %skip
num_init:
  %np = getelementptr i8, i8* %buf, i64 %c
  %nch = load i8, i8* %np
  %isneg = icmp eq i8 %nch, 45
  %cplus1 = add i64 %c, 1
  %start = select i1 %isneg, i64 %cplus1, i64 %c
  br label %dloop
dloop:
  %dc = phi i64 [ %start, %num_init ], [ %dc1, %dstep ]
  %acc = phi i64 [ 0, %num_init ], [ %acc1, %dstep ]
  %dend = icmp uge i64 %dc, %size
  br i1 %dend, label %finish, label %dcheck
dcheck:
  %dp = getelementptr i8, i8* %buf, i64 %dc
  %dch = load i8, i8* %dp
  %ge0 = icmp uge i8 %dch, 48
  %le9 = icmp ule i8 %dch, 57
  %isd = and i1 %ge0, %le9
  br i1 %isd, label %dstep, label %finish
dstep:
  %digit = sub i8 %dch, 48
  %digit64 = zext i8 %digit to i64
  %acc10 = mul i64 %acc, 10
  %acc1 = add i64 %acc10, %digit64
  %dc1 = add i64 %dc, 1
  br label %dloop
finish:
  %neg_val = sub i64 0, %acc
  %val = select i1 %isneg, i64 %neg_val, i64 %acc
  %r0 = insertvalue { i64, i64 } undef, i64 %val, 0
  %r1 = insertvalue { i64, i64 } %r0, i64 %dc, 1
  ret { i64, i64 } %r1
done_zero:
  %z0 = insertvalue { i64, i64 } undef, i64 0, 0
  %z1 = insertvalue { i64, i64 } %z0, i64 %c, 1
  ret { i64, i64 } %z1
}

; Load a .wamo object. Returns { vm, entry_pc }; vm is null on any failure
; (open, bad magic). The code + label arrays are malloc''d and owned by the
; returned VM for its lifetime; functor string copies live as long as the
; code array references them.
define { %WamState*, i32 } @wam_object_load(i8* %path) {
entry:
  %fd = call i32 @open(i8* %path, i32 0, i32 0)
  %fd_bad = icmp slt i32 %fd, 0
  br i1 %fd_bad, label %fail, label %read_init
read_init:
  %buf0 = call i8* @malloc(i64 65536)
  br label %read_loop
read_loop:
  %buf = phi i8* [ %buf0, %read_init ], [ %bufc, %after_read ]
  %cap = phi i64 [ 65536, %read_init ], [ %capc, %after_read ]
  %total = phi i64 [ 0, %read_init ], [ %total3, %after_read ]
  %free = sub i64 %cap, %total
  %low = icmp ult i64 %free, 32768
  br i1 %low, label %do_grow, label %do_read
do_grow:
  %cap.d = mul i64 %cap, 2
  %buf.d = call i8* @realloc(i8* %buf, i64 %cap.d)
  br label %do_read
do_read:
  %bufc = phi i8* [ %buf.d, %do_grow ], [ %buf, %read_loop ]
  %capc = phi i64 [ %cap.d, %do_grow ], [ %cap, %read_loop ]
  %freec = sub i64 %capc, %total
  %dst = getelementptr i8, i8* %bufc, i64 %total
  %n = call i64 @read(i32 %fd, i8* %dst, i64 %freec)
  %eof = icmp sle i64 %n, 0
  br i1 %eof, label %read_done, label %after_read
after_read:
  %total3 = add i64 %total, %n
  br label %read_loop
read_done:
  %close_ignore = call i32 @close(i32 %fd)
  %r = call { %WamState*, i32 } @wam_object_load_bytes(i8* %bufc, i64 %total)
  call void @free(i8* %bufc)
  ret { %WamState*, i32 } %r
fail:
  %fnull = insertvalue { %WamState*, i32 } undef, %WamState* null, 0
  %fret = insertvalue { %WamState*, i32 } %fnull, i32 0, 1
  ret { %WamState*, i32 } %fret
}

; Parse a .wamo object from an in-memory buffer. Does NOT free %bufc --
; the caller owns it (the file loader above frees it after; an
; eval-from-value caller keeps its own bytes alive). Re-interns atoms,
; copies functor strings, patches operands, builds the VM. Returns
; { vm, entry_pc }; null vm on a short or bad-magic buffer.
define { %WamState*, i32 } @wam_object_load_bytes(i8* %bufc, i64 %total) {
entry:
  ; magic check: need >= 5 bytes and buf[0..3] == "WAMO"
  %too_small = icmp ult i64 %total, 5
  br i1 %too_small, label %fail, label %magic
magic:
  %m0p = getelementptr i8, i8* %bufc, i64 0
  %m0 = load i8, i8* %m0p
  %m1p = getelementptr i8, i8* %bufc, i64 1
  %m1 = load i8, i8* %m1p
  %m2p = getelementptr i8, i8* %bufc, i64 2
  %m2 = load i8, i8* %m2p
  %m3p = getelementptr i8, i8* %bufc, i64 3
  %m3 = load i8, i8* %m3p
  %ok0 = icmp eq i8 %m0, 87
  %ok1 = icmp eq i8 %m1, 65
  %ok2 = icmp eq i8 %m2, 77
  %ok3 = icmp eq i8 %m3, 79
  %okA = and i1 %ok0, %ok1
  %okB = and i1 %ok2, %ok3
  %magic_ok = and i1 %okA, %okB
  br i1 %magic_ok, label %parse, label %fail
parse:
  ; version
  %pv = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 4)
  %cur_v = extractvalue { i64, i64 } %pv, 1
  ; entry index
  %pe = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %cur_v)
  %entry_idx64 = extractvalue { i64, i64 } %pe, 0
  %cur_e = extractvalue { i64, i64 } %pe, 1
  ; named-entry table: (name-string, label-index) x E. Formerly skipped;
  ; now MATERIALIZED into %WamEntryRow rows installed on the VM (fields
  ; 28/29), so a call site can resolve an entry name against an
  ; already-loaded VM (@wam_object_vm_entry_pc) -- the piece that lets
  ; dyncall_at@name work on runtime sources and compile() handles, where
  ; no file is available to re-scan.
  %pne = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %cur_e)
  %nentries = extractvalue { i64, i64 } %pne, 0
  %cur_ne = extractvalue { i64, i64 } %pne, 1
  %ent_row_sz = ptrtoint %WamEntryRow* getelementptr (%WamEntryRow, %WamEntryRow* null, i32 1) to i64
  %ent_bytes = mul i64 %nentries, %ent_row_sz
  %ent_mem = call i8* @malloc(i64 %ent_bytes)
  %entRows = bitcast i8* %ent_mem to %WamEntryRow*
  br label %ebuild_loop
ebuild_loop:
  %ei = phi i64 [ 0, %parse ], [ %ei1, %ebuild_step ]
  %ecur = phi i64 [ %cur_ne, %parse ], [ %ecur2, %ebuild_step ]
  %edone = icmp uge i64 %ei, %nentries
  br i1 %edone, label %after_entries, label %ebuild_step
ebuild_step:
  %elen_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %ecur)
  %elen = extractvalue { i64, i64 } %elen_r, 0
  %ecur_d = extractvalue { i64, i64 } %elen_r, 1
  %estr_off = add i64 %ecur_d, 1
  %estr = getelementptr i8, i8* %bufc, i64 %estr_off
  %ecopy_sz = add i64 %elen, 1
  %ecopy = call i8* @malloc(i64 %ecopy_sz)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %ecopy, i8* %estr, i64 %elen, i1 false)
  %enul = getelementptr i8, i8* %ecopy, i64 %elen
  store i8 0, i8* %enul
  %eafter_str = add i64 %estr_off, %elen
  %eidx_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %eafter_str)
  %eidx = extractvalue { i64, i64 } %eidx_r, 0
  %ecur2 = extractvalue { i64, i64 } %eidx_r, 1
  %erow = getelementptr %WamEntryRow, %WamEntryRow* %entRows, i64 %ei
  %erow_name = getelementptr %WamEntryRow, %WamEntryRow* %erow, i32 0, i32 0
  store i8* %ecopy, i8** %erow_name
  %erow_len = getelementptr %WamEntryRow, %WamEntryRow* %erow, i32 0, i32 1
  store i64 %elen, i64* %erow_len
  %erow_lbl = getelementptr %WamEntryRow, %WamEntryRow* %erow, i32 0, i32 2
  %eidx32 = trunc i64 %eidx to i32
  store i32 %eidx32, i32* %erow_lbl
  %ei1 = add i64 %ei, 1
  br label %ebuild_loop
after_entries:
  ; atom count
  %pa = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %ecur)
  %natoms = extractvalue { i64, i64 } %pa, 0
  %cur_a0 = extractvalue { i64, i64 } %pa, 1
  %atom_bytes = mul i64 %natoms, 8
  %atom_mem = call i8* @malloc(i64 %atom_bytes)
  %atomIds = bitcast i8* %atom_mem to i64*
  br label %aloop
aloop:
  %ai = phi i64 [ 0, %after_entries ], [ %ai1, %astep ]
  %acur = phi i64 [ %cur_a0, %after_entries ], [ %acur2, %astep ]
  %adone = icmp uge i64 %ai, %natoms
  br i1 %adone, label %fdr_init, label %astep
astep:
  %alen_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %acur)
  %alen = extractvalue { i64, i64 } %alen_r, 0
  %acur_d = extractvalue { i64, i64 } %alen_r, 1
  %astr_off = add i64 %acur_d, 1
  %astr = getelementptr i8, i8* %bufc, i64 %astr_off
  %aid = call i64 @wam_intern_atom(i8* %astr, i64 %alen)
  %aslot = getelementptr i64, i64* %atomIds, i64 %ai
  store i64 %aid, i64* %aslot
  %acur2 = add i64 %astr_off, %alen
  %ai1 = add i64 %ai, 1
  br label %aloop
fdr_init:
  ; functor count
  %pf = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %acur)
  %nfun = extractvalue { i64, i64 } %pf, 0
  %cur_f0 = extractvalue { i64, i64 } %pf, 1
  %fun_bytes = mul i64 %nfun, 8
  %fun_mem = call i8* @malloc(i64 %fun_bytes)
  %funPtrs = bitcast i8* %fun_mem to i8**
  br label %floop
floop:
  %fi = phi i64 [ 0, %fdr_init ], [ %fi1, %fstep ]
  %fcur = phi i64 [ %cur_f0, %fdr_init ], [ %fcur2, %fstep ]
  %fdone = icmp uge i64 %fi, %nfun
  br i1 %fdone, label %lbl_init, label %fstep
fstep:
  %flen_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %fcur)
  %flen = extractvalue { i64, i64 } %flen_r, 0
  %fcur_d = extractvalue { i64, i64 } %flen_r, 1
  %fstr_off = add i64 %fcur_d, 1
  %fstr = getelementptr i8, i8* %bufc, i64 %fstr_off
  %fcopy_sz = add i64 %flen, 1
  %fcopy = call i8* @malloc(i64 %fcopy_sz)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %fcopy, i8* %fstr, i64 %flen, i1 false)
  %fnul = getelementptr i8, i8* %fcopy, i64 %flen
  store i8 0, i8* %fnul
  %fpslot = getelementptr i8*, i8** %funPtrs, i64 %fi
  store i8* %fcopy, i8** %fpslot
  %fcur2 = add i64 %fstr_off, %flen
  %fi1 = add i64 %fi, 1
  br label %floop
lbl_init:
  ; label count
  %pl = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %fcur)
  %nlbl = extractvalue { i64, i64 } %pl, 0
  %cur_l0 = extractvalue { i64, i64 } %pl, 1
  %lbl_bytes = mul i64 %nlbl, 4
  %lbl_mem = call i8* @malloc(i64 %lbl_bytes)
  %labels = bitcast i8* %lbl_mem to i32*
  br label %lloop
lloop:
  %li = phi i64 [ 0, %lbl_init ], [ %li1, %lstep ]
  %lcur = phi i64 [ %cur_l0, %lbl_init ], [ %lcur2, %lstep ]
  %ldone = icmp uge i64 %li, %nlbl
  br i1 %ldone, label %code_init, label %lstep
lstep:
  %lpc_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %lcur)
  %lpc = extractvalue { i64, i64 } %lpc_r, 0
  %lcur2 = extractvalue { i64, i64 } %lpc_r, 1
  %lpc32 = trunc i64 %lpc to i32
  %lslot = getelementptr i32, i32* %labels, i64 %li
  store i32 %lpc32, i32* %lslot
  %li1 = add i64 %li, 1
  br label %lloop
code_init:
  ; code count
  %pc_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %lcur)
  %ncode = extractvalue { i64, i64 } %pc_r, 0
  %cur_c0 = extractvalue { i64, i64 } %pc_r, 1
  %instr_sz = ptrtoint %Instruction* getelementptr (%Instruction, %Instruction* null, i32 1) to i64
  %code_bytes = mul i64 %ncode, %instr_sz
  %code_mem = call i8* @malloc(i64 %code_bytes)
  %code = bitcast i8* %code_mem to %Instruction*
  br label %cloop
cloop:
  %ci = phi i64 [ 0, %code_init ], [ %ci1, %cstore ]
  %ccur = phi i64 [ %cur_c0, %code_init ], [ %ccur4, %cstore ]
  %cdone = icmp uge i64 %ci, %ncode
  br i1 %cdone, label %meta_init, label %cstep
cstep:
  %tag_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %ccur)
  %tag64 = extractvalue { i64, i64 } %tag_r, 0
  %ccur1 = extractvalue { i64, i64 } %tag_r, 1
  %op1_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %ccur1)
  %op1 = extractvalue { i64, i64 } %op1_r, 0
  %ccur2 = extractvalue { i64, i64 } %op1_r, 1
  %op2_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %ccur2)
  %op2 = extractvalue { i64, i64 } %op2_r, 0
  %ccur3 = extractvalue { i64, i64 } %op2_r, 1
  %rel_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %ccur3)
  %reloc = extractvalue { i64, i64 } %rel_r, 0
  %ccur4 = extractvalue { i64, i64 } %rel_r, 1
  %is_atom = icmp eq i64 %reloc, 1
  br i1 %is_atom, label %patch_atom, label %chk_functor
patch_atom:
  %ai_slot = getelementptr i64, i64* %atomIds, i64 %op1
  %aid_p = load i64, i64* %ai_slot
  br label %cstore
chk_functor:
  %is_fun = icmp eq i64 %reloc, 2
  br i1 %is_fun, label %patch_fun, label %chk_float
patch_fun:
  %fp_slot = getelementptr i8*, i8** %funPtrs, i64 %op1
  %fp = load i8*, i8** %fp_slot
  %fpi = ptrtoint i8* %fp to i64
  br label %cstore
chk_float:
  %is_float = icmp eq i64 %reloc, 3
  br i1 %is_float, label %patch_float, label %cstore
patch_float:
  ; float constant: op1 indexes the C-string table (shared with functor
  ; names); parse the decimal text with strtod and store the i64 bit
  ; pattern -- the same double the AOT `bitcast (double c to i64)` yields.
  %flp_slot = getelementptr i8*, i8** %funPtrs, i64 %op1
  %flp = load i8*, i8** %flp_slot
  %fld = call double @strtod(i8* %flp, i8** null)
  %flbits = bitcast double %fld to i64
  br label %cstore
cstore:
  %op1_final = phi i64 [ %aid_p, %patch_atom ], [ %fpi, %patch_fun ], [ %flbits, %patch_float ], [ %op1, %chk_float ]
  %tag32 = trunc i64 %tag64 to i32
  %ip = getelementptr %Instruction, %Instruction* %code, i64 %ci
  %ip_tag = getelementptr %Instruction, %Instruction* %ip, i32 0, i32 0
  store i32 %tag32, i32* %ip_tag
  %ip_op1 = getelementptr %Instruction, %Instruction* %ip, i32 0, i32 1
  store i64 %op1_final, i64* %ip_op1
  %ip_op2 = getelementptr %Instruction, %Instruction* %ip, i32 0, i32 2
  store i64 %op2, i64* %ip_op2
  %ci1 = add i64 %ci, 1
  br label %cloop
meta_init:
  ; --- meta-call table (format v2) ---------------------------------------
  ; A trailing table of the object''s meta-callable predicates so a loaded
  ; object can call/N its own goals (@wam_dispatch_meta_call consults the VM
  ; field when present). Rows: <atomIdx> <funIdx> <arity> <labelIdx>, where
  ; atomIdx / funIdx index the object''s own (relocated) atom / functor
  ; tables so we resolve them here, before the scratch arrays are freed.
  ; Count is 0 for objects the writer saw contain no meta-call.
  %pm = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %ccur)
  %nmeta = extractvalue { i64, i64 } %pm, 0
  %cur_m0 = extractvalue { i64, i64 } %pm, 1
  %meta_row_sz = ptrtoint %WamMetaRow* getelementptr (%WamMetaRow, %WamMetaRow* null, i32 1) to i64
  %meta_bytes = mul i64 %nmeta, %meta_row_sz
  %meta_mem = call i8* @malloc(i64 %meta_bytes)
  %metaRows = bitcast i8* %meta_mem to %WamMetaRow*
  br label %mloop
mloop:
  %mi = phi i64 [ 0, %meta_init ], [ %mi1, %mstore ]
  %mcur = phi i64 [ %cur_m0, %meta_init ], [ %mcur4, %mstore ]
  %mdone = icmp uge i64 %mi, %nmeta
  br i1 %mdone, label %sw_init, label %mstep
mstep:
  %matom_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %mcur)
  %matomIdx = extractvalue { i64, i64 } %matom_r, 0
  %mcur1 = extractvalue { i64, i64 } %matom_r, 1
  %mfun_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %mcur1)
  %mfunIdx = extractvalue { i64, i64 } %mfun_r, 0
  %mcur2 = extractvalue { i64, i64 } %mfun_r, 1
  %mar_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %mcur2)
  %marity = extractvalue { i64, i64 } %mar_r, 0
  %mcur3 = extractvalue { i64, i64 } %mar_r, 1
  %mlbl_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %mcur3)
  %mlabel = extractvalue { i64, i64 } %mlbl_r, 0
  %mcur4 = extractvalue { i64, i64 } %mlbl_r, 1
  ; atom_id = atomIds[atomIdx] (already re-interned)
  %maslot = getelementptr i64, i64* %atomIds, i64 %matomIdx
  %matom_id = load i64, i64* %maslot
  ; functor_ptr = funIdx < 0 ? null : funPtrs[funIdx] (object''s own copy)
  %mfun_neg = icmp slt i64 %mfunIdx, 0
  br i1 %mfun_neg, label %mstore, label %mfun_load
mfun_load:
  %mfslot = getelementptr i8*, i8** %funPtrs, i64 %mfunIdx
  %mfp = load i8*, i8** %mfslot
  br label %mstore
mstore:
  %mfp_final = phi i8* [ null, %mstep ], [ %mfp, %mfun_load ]
  %mrow = getelementptr %WamMetaRow, %WamMetaRow* %metaRows, i64 %mi
  %mrow_atom = getelementptr %WamMetaRow, %WamMetaRow* %mrow, i32 0, i32 0
  store i64 %matom_id, i64* %mrow_atom
  %mrow_fun = getelementptr %WamMetaRow, %WamMetaRow* %mrow, i32 0, i32 1
  store i8* %mfp_final, i8** %mrow_fun
  %mrow_ar = getelementptr %WamMetaRow, %WamMetaRow* %mrow, i32 0, i32 2
  %marity32 = trunc i64 %marity to i32
  store i32 %marity32, i32* %mrow_ar
  %mrow_lbl = getelementptr %WamMetaRow, %WamMetaRow* %mrow, i32 0, i32 3
  %mlabel32 = trunc i64 %mlabel to i32
  store i32 %mlabel32, i32* %mrow_lbl
  %mi1 = add i64 %mi, 1
  br label %mloop
sw_init:
  ; --- switch-table section (loaded clause indexing) ----------------------
  ; One count NS, then per table a "pc n" pair and n entry rows of
  ; "kind key labelIdx" (all newline-separated ints).
  ; kind 0 = atom key (key is an atom-table index, resolved through the same
  ; relocated atomIds array as any reloc-1 operand, so it compares equal to
  ; relocated register payloads); kind 1 = integer key. Each table
  ; materializes as a malloc-ed [n x %SwitchEntry] whose pointer patches
  ; op1 (and op2 = n) of the tag-25/27 instruction at <pc> -- the exact
  ; shape the AOT dispatcher feeds @wam_switch_on_constant(_a2), so the
  ; SHARED step cases run loaded switches with no new dispatch code, and
  ; label indices resolve through the loaded VM own label table via
  ; @wam_label_pc. Objects without the section (the bootstrap compiler
  ; emits none; pre-section v2 objects end here) read NS = 0 at EOF.
  ; Tables follow the code array lifetime (not freed by wam_state_free,
  ; same policy as the code/labels/meta allocations).
  %ps = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %mcur)
  %nsw = extractvalue { i64, i64 } %ps, 0
  %cur_s0 = extractvalue { i64, i64 } %ps, 1
  br label %swloop
swloop:
  %si = phi i64 [ 0, %sw_init ], [ %si1, %swpatch ]
  %scur = phi i64 [ %cur_s0, %sw_init ], [ %secur, %swpatch ]
  %sdone = icmp uge i64 %si, %nsw
  br i1 %sdone, label %build_vm, label %swtab
swtab:
  %spc_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %scur)
  %spc = extractvalue { i64, i64 } %spc_r, 0
  %scur1 = extractvalue { i64, i64 } %spc_r, 1
  %sn_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %scur1)
  %sn = extractvalue { i64, i64 } %sn_r, 0
  %scur2 = extractvalue { i64, i64 } %sn_r, 1
  %se_sz = ptrtoint %SwitchEntry* getelementptr (%SwitchEntry, %SwitchEntry* null, i32 1) to i64
  %stab_bytes = mul i64 %sn, %se_sz
  %stab_mem = call i8* @malloc(i64 %stab_bytes)
  %stab = bitcast i8* %stab_mem to %SwitchEntry*
  br label %seloop
seloop:
  %sei = phi i64 [ 0, %swtab ], [ %sei1, %sestore ]
  %secur = phi i64 [ %scur2, %swtab ], [ %secur3, %sestore ]
  %sedone = icmp uge i64 %sei, %sn
  br i1 %sedone, label %swpatch, label %sestep
sestep:
  %sk_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %secur)
  %skind = extractvalue { i64, i64 } %sk_r, 0
  %secur1 = extractvalue { i64, i64 } %sk_r, 1
  %skey_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %secur1)
  %skey = extractvalue { i64, i64 } %skey_r, 0
  %secur2 = extractvalue { i64, i64 } %skey_r, 1
  %slbl_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %secur2)
  %slbl = extractvalue { i64, i64 } %slbl_r, 0
  %secur3 = extractvalue { i64, i64 } %slbl_r, 1
  %sk_is_atom = icmp eq i64 %skind, 0
  br i1 %sk_is_atom, label %se_atom, label %sestore
se_atom:
  %sa_slot = getelementptr i64, i64* %atomIds, i64 %skey
  %sa_id = load i64, i64* %sa_slot
  br label %sestore
sestore:
  %skey_final = phi i64 [ %sa_id, %se_atom ], [ %skey, %sestep ]
  %stag_final = phi i32 [ 0, %se_atom ], [ 1, %sestep ]
  %sep = getelementptr %SwitchEntry, %SwitchEntry* %stab, i64 %sei
  %sep_tag = getelementptr %SwitchEntry, %SwitchEntry* %sep, i32 0, i32 0
  store i32 %stag_final, i32* %sep_tag
  %sep_pay = getelementptr %SwitchEntry, %SwitchEntry* %sep, i32 0, i32 1
  store i64 %skey_final, i64* %sep_pay
  %sep_lbl = getelementptr %SwitchEntry, %SwitchEntry* %sep, i32 0, i32 2
  %slbl32 = trunc i64 %slbl to i32
  store i32 %slbl32, i32* %sep_lbl
  %sei1 = add i64 %sei, 1
  br label %seloop
swpatch:
  %swip = getelementptr %Instruction, %Instruction* %code, i64 %spc
  %swip_op1 = getelementptr %Instruction, %Instruction* %swip, i32 0, i32 1
  %stab_i = ptrtoint %SwitchEntry* %stab to i64
  store i64 %stab_i, i64* %swip_op1
  %swip_op2 = getelementptr %Instruction, %Instruction* %swip, i32 0, i32 2
  store i64 %sn, i64* %swip_op2
  %si1 = add i64 %si, 1
  br label %swloop
build_vm:
  ; entry pc = labels[entry_idx]
  %eslot = getelementptr i32, i32* %labels, i64 %entry_idx64
  %entry_pc = load i32, i32* %eslot
  ; scratch pointer arrays no longer needed: atom strings were copied by
  ; @wam_intern_atom, functor string copies are owned by %code, the meta
  ; rows already captured the ids/pointers they need, and the source buffer
  ; belongs to the caller.
  call void @free(i8* %atom_mem)
  call void @free(i8* %fun_mem)
  %ncode32 = trunc i64 %ncode to i32
  %nlbl32 = trunc i64 %nlbl to i32
  %vm = call %WamState* @wam_state_new(%Instruction* %code, i32 %ncode32, i32* %labels, i32 %nlbl32)
  ; install the object''s meta-call table (fields 25/26)
  %vm_meta_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 25
  store %WamMetaRow* %metaRows, %WamMetaRow** %vm_meta_ptr
  %vm_metac_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 26
  %nmeta32 = trunc i64 %nmeta to i32
  store i32 %nmeta32, i32* %vm_metac_ptr
  ; install the object''s named-entry table (fields 28/29)
  %vm_ent_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 28
  store %WamEntryRow* %entRows, %WamEntryRow** %vm_ent_ptr
  %vm_entc_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 29
  %nent32 = trunc i64 %nentries to i32
  store i32 %nent32, i32* %vm_entc_ptr
  %ret0 = insertvalue { %WamState*, i32 } undef, %WamState* %vm, 0
  %ret1 = insertvalue { %WamState*, i32 } %ret0, i32 %entry_pc, 1
  ret { %WamState*, i32 } %ret1
fail:
  %fnull = insertvalue { %WamState*, i32 } undef, %WamState* null, 0
  %fret = insertvalue { %WamState*, i32 } %fnull, i32 0, 1
  ret { %WamState*, i32 } %fret
}

; Call a loaded object''s entry predicate with %nargs %Value arguments (in
; registers A0..A_{nargs-1}) and read back an Integer from register
; %out_reg. Returns { value, ok }; ok=false on failure or non-integer.
; Saves/restores the VM heap top and rewinds the arena so repeated calls
; run in constant memory -- same discipline as the plawk foreign bridge.
define { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read_out, label %rewind_fail
read_out:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  %out_tag = extractvalue %Value %out, 0
  %out_is_int = icmp eq i32 %out_tag, 1
  br i1 %out_is_int, label %good, label %rewind_fail
good:
  %payload = extractvalue %Value %out, 1
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  %g0 = insertvalue { i64, i1 } undef, i64 %payload, 0
  %g1 = insertvalue { i64, i1 } %g0, i1 true, 1
  ret { i64, i1 } %g1
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}

; Double-returning variant of @wam_object_call_i64: the entry output cell
; may bind to an Integer or a Float, and @value_to_double promotes either
; to double. Returns { value, ok }; ok=false on failure or a non-numeric
; binding. Same heap-save/arena-rewind discipline for constant memory.
define { double, i1 } @wam_object_call_f64(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read_out, label %rewind_fail
read_out:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  %out_is_num = call i1 @value_is_number(%Value %out)
  br i1 %out_is_num, label %good, label %rewind_fail
good:
  %fval = call double @value_to_double(%Value %out)
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  %g0 = insertvalue { double, i1 } undef, double %fval, 0
  %g1 = insertvalue { double, i1 } %g0, i1 true, 1
  ret { double, i1 } %g1
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}

; Byte-string variant: the entry output cell must bind to an Atom whose
; interned string is the payload (atoms are byte strings here; NUL-free by
; the blob convention). Returns { ptr, len, ok } -- the pointer is into the
; persistent atom table, so it survives the arena rewind. ok=false on
; failure or a non-atom output.
define { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read_out, label %rewind_fail
read_out:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  %out_tag = extractvalue %Value %out, 0
  %out_is_atom = icmp eq i32 %out_tag, 0
  br i1 %out_is_atom, label %good, label %rewind_fail
good:
  %aid = extractvalue %Value %out, 1
  %sptr = call i8* @wam_atom_to_string(i64 %aid)
  %sptr_null = icmp eq i8* %sptr, null
  br i1 %sptr_null, label %rewind_fail, label %have_str
have_str:
  %slen = call i64 @strlen(i8* %sptr)
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  %g0 = insertvalue { i8*, i64, i1 } undef, i8* %sptr, 0
  %g1 = insertvalue { i8*, i64, i1 } %g0, i64 %slen, 1
  %g2 = insertvalue { i8*, i64, i1 } %g1, i1 true, 2
  ret { i8*, i64, i1 } %g2
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  %f0 = insertvalue { i8*, i64, i1 } undef, i8* null, 0
  %f1 = insertvalue { i8*, i64, i1 } %f0, i64 0, 1
  %f2 = insertvalue { i8*, i64, i1 } %f1, i1 false, 2
  ret { i8*, i64, i1 } %f2
}

; Structured-record variant (JIT roadmap #4): the entry output cell must bind
; to a Compound term whose arity equals %nfields; each arg is deserialized
; into out_slots[i] per typecodes[i] BEFORE the arena is rewound (the compound
; and its arg cells live in the arena -- only atoms survive a rewind). This is
; the "output is a term, not a scalar" primitive: a grammar returns e.g.
; rec(42, 3.14) and the caller lays the fields into a record buffer.
;   typecodes[i] = 0  -> i64    : arg i must be Integer; out_slots[i] = its i64
;   typecodes[i] = 1  -> f64    : arg i must be a number; out_slots[i] = the
;                                 double bits (value_to_double promotes Integer)
;   typecodes[i] = 2  -> string : arg i must be an Atom; out_slots[i] = the
;                                 atom string pointer (as i64), out_lens[i] =
;                                 its strlen. The pointer is into the persistent
;                                 atom table, so it survives the arena rewind
;                                 (same as @wam_object_call_bytes). NUL-free by
;                                 the blob convention.
; out_slots is an i64[nfields]; f64 fields are stored as bitcast double bits,
; string fields as a ptrtoint. out_lens is an i64[nfields]; only string fields
; write it (numeric fields set 0). So the caller reads slot i as i64, bitcasts
; to double, or pairs (slot,len) into a byte slice per its own shape.
; Returns i1 ok; false on call failure, a non-Compound output, an arity
; mismatch, or an arg whose type does not satisfy its typecode. Same
; heap-save / arena-rewind discipline as the scalar variants, so a per-record
; call stays constant-memory.
define i1 @wam_object_call_record(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg, i32 %nfields, i8* %typecodes, i64* %out_slots, i64* %out_lens) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read_out, label %rewind_fail
read_out:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  %out_tag = call i32 @value_tag(%Value %out)
  %is_compound = icmp eq i32 %out_tag, 3
  br i1 %is_compound, label %have_compound, label %rewind_fail
have_compound:
  %cp_bits = call i64 @value_payload(%Value %out)
  %cp = inttoptr i64 %cp_bits to %Compound*
  %ar_slot = getelementptr %Compound, %Compound* %cp, i32 0, i32 1
  %arity = load i32, i32* %ar_slot
  %arity_ok = icmp eq i32 %arity, %nfields
  br i1 %arity_ok, label %fields_init, label %rewind_fail
fields_init:
  %cargs_slot = getelementptr %Compound, %Compound* %cp, i32 0, i32 2
  %cargs = load %Value*, %Value** %cargs_slot
  br label %floop
floop:
  %fi = phi i32 [ 0, %fields_init ], [ %fi1, %fstep_ok ]
  %fdone = icmp sge i32 %fi, %nfields
  br i1 %fdone, label %ok_rewind, label %fstep
fstep:
  %fi64 = sext i32 %fi to i64
  %farg_ptr = getelementptr %Value, %Value* %cargs, i64 %fi64
  %farg_raw = load %Value, %Value* %farg_ptr
  %farg = call %Value @wam_deref_value(%WamState* %vm, %Value %farg_raw)
  %ftag = call i32 @value_tag(%Value %farg)
  %tc_ptr = getelementptr i8, i8* %typecodes, i64 %fi64
  %tc = load i8, i8* %tc_ptr
  %len_slot = getelementptr i64, i64* %out_lens, i64 %fi64
  %is_str = icmp eq i8 %tc, 2
  br i1 %is_str, label %field_str, label %field_num
field_num:
  %is_f64 = icmp eq i8 %tc, 1
  br i1 %is_f64, label %field_f64, label %field_i64
field_i64:
  %is_int = icmp eq i32 %ftag, 1
  br i1 %is_int, label %store_i64, label %rewind_fail
store_i64:
  %ival = call i64 @value_payload(%Value %farg)
  %slot_i = getelementptr i64, i64* %out_slots, i64 %fi64
  store i64 %ival, i64* %slot_i
  store i64 0, i64* %len_slot
  br label %fstep_ok
field_f64:
  %is_num = call i1 @value_is_number(%Value %farg)
  br i1 %is_num, label %store_f64, label %rewind_fail
store_f64:
  %dval = call double @value_to_double(%Value %farg)
  %dbits = bitcast double %dval to i64
  %slot_f = getelementptr i64, i64* %out_slots, i64 %fi64
  store i64 %dbits, i64* %slot_f
  store i64 0, i64* %len_slot
  br label %fstep_ok
field_str:
  %is_atom = icmp eq i32 %ftag, 0
  br i1 %is_atom, label %store_str, label %rewind_fail
store_str:
  %said = call i64 @value_payload(%Value %farg)
  %sptr = call i8* @wam_atom_to_string(i64 %said)
  %sptr_null = icmp eq i8* %sptr, null
  br i1 %sptr_null, label %rewind_fail, label %have_sptr
have_sptr:
  %slen = call i64 @strlen(i8* %sptr)
  %sbits = ptrtoint i8* %sptr to i64
  %slot_s = getelementptr i64, i64* %out_slots, i64 %fi64
  store i64 %sbits, i64* %slot_s
  store i64 %slen, i64* %len_slot
  br label %fstep_ok
fstep_ok:
  %fi1 = add i32 %fi, 1
  br label %floop
ok_rewind:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 true
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  ret i1 false
}

; Associative-array variant (JIT roadmap #4): the entry output cell must bind
; to a proper list of 2-arg pairs -- e.g. [K1-V1, K2-V2, ...] -- and each
; pair''s (Integer-or-Atom key, Integer value) is inserted into the caller''s
; i64 assoc table via @wam_assoc_i64_inc (add semantics: a fresh key ends at
; its value; a repeated key accumulates, matching a tally). An Atom key''s id
; is a global-registry id -- the same keyspace text-mode field slices and
; blob keys intern into -- so for-in reports and string-literal lookups
; resolve it like any other key (one key kind per table; integer keys
; collide with atom ids, the documented text-mode ambiguity). The walk reads
; the cons cells (functor "[|]", arity 2) and pairs (any arity-2 compound;
; arg0 = key, arg1 = value) BEFORE the arena rewind, since they live in the
; arena. Returns i1 ok; false on call failure, a non-list output, a non-pair
; element, a non-Integer/non-Atom key, or a non-Integer value.
define i1 @wam_object_call_assoc(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg, %WamAssocI64Table* %table) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %walk_init, label %rewind_fail
walk_init:
  %consfn = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  %head0 = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  br label %walk
walk:
  %cur = phi %Value [ %head0, %walk_init ], [ %tail_d, %next ]
  %ctag = call i32 @value_tag(%Value %cur)
  %is_comp = icmp eq i32 %ctag, 3
  br i1 %is_comp, label %check_cons, label %walk_done
check_cons:
  %cbits = call i64 @value_payload(%Value %cur)
  %ccp = inttoptr i64 %cbits to %Compound*
  %cfn_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 0
  %cfn = load i8*, i8** %cfn_slot
  %car_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 1
  %car = load i32, i32* %car_slot
  %car_ok = icmp eq i32 %car, 2
  br i1 %car_ok, label %cmp_cons, label %walk_done
cmp_cons:
  %fncmp = call i32 @strcmp(i8* %cfn, i8* %consfn)
  %is_cons = icmp eq i32 %fncmp, 0
  br i1 %is_cons, label %have_cell, label %walk_done
have_cell:
  %cargs_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 2
  %cargs = load %Value*, %Value** %cargs_slot
  %h_ptr = getelementptr %Value, %Value* %cargs, i64 0
  %h_raw = load %Value, %Value* %h_ptr
  %t_ptr = getelementptr %Value, %Value* %cargs, i64 1
  %t_raw = load %Value, %Value* %t_ptr
  %pair = call %Value @wam_deref_value(%WamState* %vm, %Value %h_raw)
  %ptag = call i32 @value_tag(%Value %pair)
  %p_is_comp = icmp eq i32 %ptag, 3
  br i1 %p_is_comp, label %pair_arity, label %rewind_fail
pair_arity:
  %pbits = call i64 @value_payload(%Value %pair)
  %pcp = inttoptr i64 %pbits to %Compound*
  %par_slot = getelementptr %Compound, %Compound* %pcp, i32 0, i32 1
  %par = load i32, i32* %par_slot
  %par_ok = icmp eq i32 %par, 2
  br i1 %par_ok, label %pair_args, label %rewind_fail
pair_args:
  %pargs_slot = getelementptr %Compound, %Compound* %pcp, i32 0, i32 2
  %pargs = load %Value*, %Value** %pargs_slot
  %k_ptr = getelementptr %Value, %Value* %pargs, i64 0
  %k_raw = load %Value, %Value* %k_ptr
  %v_ptr = getelementptr %Value, %Value* %pargs, i64 1
  %v_raw = load %Value, %Value* %v_ptr
  %kv = call %Value @wam_deref_value(%WamState* %vm, %Value %k_raw)
  %vv = call %Value @wam_deref_value(%WamState* %vm, %Value %v_raw)
  %ktag = call i32 @value_tag(%Value %kv)
  %vtag = call i32 @value_tag(%Value %vv)
  ; Keys: Integer (payload as-is) or Atom (JIT roadmap item 4 string
  ; fields -- the atom id IS a global-registry id, the same keyspace the
  ; text-mode assoc pipeline interns field slices and blob keys into, so
  ; for-in reports and "literal" lookups resolve it like any other key).
  ; Values stay Integer: the table is an i64 accumulator. Mixing integer
  ; and atom keys in ONE table is the documented text-mode ambiguity
  ; (integer keys collide with atom ids) -- a grammar should return one
  ; key kind per table.
  %k_is_int = icmp eq i32 %ktag, 1
  %k_is_atom = icmp eq i32 %ktag, 0
  %k_ok = or i1 %k_is_int, %k_is_atom
  %v_is_int = icmp eq i32 %vtag, 1
  %kv_ok = and i1 %k_ok, %v_is_int
  br i1 %kv_ok, label %insert, label %rewind_fail
insert:
  %kval = call i64 @value_payload(%Value %kv)
  %vval = call i64 @value_payload(%Value %vv)
  %ignored = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 %kval, i64 %vval)
  br label %next
next:
  %tail_d = call %Value @wam_deref_value(%WamState* %vm, %Value %t_raw)
  br label %walk
walk_done:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 true
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  ret i1 false
}

; The SECOND table kind: string values. Same [K1-V1, K2-V2, ...] walk as
; @wam_object_call_assoc, but each value must be an ATOM and its id (a
; global-registry id, the keyspace text-mode slices intern into) is stored
; via @wam_assoc_i64_set -- REPLACE semantics, since accumulating ids is
; meaningless; a repeated key keeps the latest label. Keys stay Integer or
; Atom as in the i64 kind. The table layout is the shared i64 table; only
; the declared value kind (surface: `as assoc(str)`) tells the reader to
; resolve values back to text. Returns i1 ok; false on call failure, a
; non-list output, a non-pair element, a bad key, or a non-Atom value.
define i1 @wam_object_call_assoc_str(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg, %WamAssocI64Table* %table) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %walk_init, label %rewind_fail
walk_init:
  %consfn = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  %head0 = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  br label %walk
walk:
  %cur = phi %Value [ %head0, %walk_init ], [ %tail_d, %next ]
  %ctag = call i32 @value_tag(%Value %cur)
  %is_comp = icmp eq i32 %ctag, 3
  br i1 %is_comp, label %check_cons, label %walk_done
check_cons:
  %cbits = call i64 @value_payload(%Value %cur)
  %ccp = inttoptr i64 %cbits to %Compound*
  %cfn_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 0
  %cfn = load i8*, i8** %cfn_slot
  %car_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 1
  %car = load i32, i32* %car_slot
  %car_ok = icmp eq i32 %car, 2
  br i1 %car_ok, label %cmp_cons, label %walk_done
cmp_cons:
  %fncmp = call i32 @strcmp(i8* %cfn, i8* %consfn)
  %is_cons = icmp eq i32 %fncmp, 0
  br i1 %is_cons, label %have_cell, label %walk_done
have_cell:
  %cargs_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 2
  %cargs = load %Value*, %Value** %cargs_slot
  %h_ptr = getelementptr %Value, %Value* %cargs, i64 0
  %h_raw = load %Value, %Value* %h_ptr
  %t_ptr = getelementptr %Value, %Value* %cargs, i64 1
  %t_raw = load %Value, %Value* %t_ptr
  %pair = call %Value @wam_deref_value(%WamState* %vm, %Value %h_raw)
  %ptag = call i32 @value_tag(%Value %pair)
  %p_is_comp = icmp eq i32 %ptag, 3
  br i1 %p_is_comp, label %pair_arity, label %rewind_fail
pair_arity:
  %pbits = call i64 @value_payload(%Value %pair)
  %pcp = inttoptr i64 %pbits to %Compound*
  %par_slot = getelementptr %Compound, %Compound* %pcp, i32 0, i32 1
  %par = load i32, i32* %par_slot
  %par_ok = icmp eq i32 %par, 2
  br i1 %par_ok, label %pair_args, label %rewind_fail
pair_args:
  %pargs_slot = getelementptr %Compound, %Compound* %pcp, i32 0, i32 2
  %pargs = load %Value*, %Value** %pargs_slot
  %k_ptr = getelementptr %Value, %Value* %pargs, i64 0
  %k_raw = load %Value, %Value* %k_ptr
  %v_ptr = getelementptr %Value, %Value* %pargs, i64 1
  %v_raw = load %Value, %Value* %v_ptr
  %kv = call %Value @wam_deref_value(%WamState* %vm, %Value %k_raw)
  %vv = call %Value @wam_deref_value(%WamState* %vm, %Value %v_raw)
  %ktag = call i32 @value_tag(%Value %kv)
  %vtag = call i32 @value_tag(%Value %vv)
  %k_is_int = icmp eq i32 %ktag, 1
  %k_is_atom = icmp eq i32 %ktag, 0
  %k_ok = or i1 %k_is_int, %k_is_atom
  %v_is_atom = icmp eq i32 %vtag, 0
  %kv_ok = and i1 %k_ok, %v_is_atom
  br i1 %kv_ok, label %insert, label %rewind_fail
insert:
  %kval = call i64 @value_payload(%Value %kv)
  %vval = call i64 @value_payload(%Value %vv)
  %ignored = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %table, i64 %kval, i64 %vval)
  br label %next
next:
  %tail_d = call %Value @wam_deref_value(%WamState* %vm, %Value %t_raw)
  br label %walk
walk_done:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 true
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  ret i1 false
}

; The THIRD table kind: a POSITIONAL array. The entry returns a FLAT
; list [V1, V2, ..., Vn] (not key-value pairs); element i is stored at
; key i (1-indexed, the awk `split` convention) via @wam_assoc_i64_set --
; REPLACE semantics, so a per-record bind reflects the most recent
; record''s list. Values must be Integer; the shared i64 table holds them
; (JIT roadmap item 4''s last target container). Returns i1 ok; false on
; call failure, a non-list output, or a non-Integer element.
define i1 @wam_object_call_posarray(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg, %WamAssocI64Table* %table) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %walk_init, label %rewind_fail
walk_init:
  %consfn = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  %head0 = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  br label %walk
walk:
  %cur = phi %Value [ %head0, %walk_init ], [ %tail_d, %next ]
  %pos = phi i64 [ 1, %walk_init ], [ %pos1, %next ]
  %ctag = call i32 @value_tag(%Value %cur)
  %is_comp = icmp eq i32 %ctag, 3
  br i1 %is_comp, label %check_cons, label %walk_done
check_cons:
  %cbits = call i64 @value_payload(%Value %cur)
  %ccp = inttoptr i64 %cbits to %Compound*
  %cfn_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 0
  %cfn = load i8*, i8** %cfn_slot
  %car_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 1
  %car = load i32, i32* %car_slot
  %car_ok = icmp eq i32 %car, 2
  br i1 %car_ok, label %cmp_cons, label %walk_done
cmp_cons:
  %fncmp = call i32 @strcmp(i8* %cfn, i8* %consfn)
  %is_cons = icmp eq i32 %fncmp, 0
  br i1 %is_cons, label %have_cell, label %walk_done
have_cell:
  %cargs_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 2
  %cargs = load %Value*, %Value** %cargs_slot
  %h_ptr = getelementptr %Value, %Value* %cargs, i64 0
  %h_raw = load %Value, %Value* %h_ptr
  %t_ptr = getelementptr %Value, %Value* %cargs, i64 1
  %t_raw = load %Value, %Value* %t_ptr
  %elem = call %Value @wam_deref_value(%WamState* %vm, %Value %h_raw)
  %etag = call i32 @value_tag(%Value %elem)
  %e_is_int = icmp eq i32 %etag, 1
  br i1 %e_is_int, label %insert, label %rewind_fail
insert:
  %eval = call i64 @value_payload(%Value %elem)
  %ignored = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %table, i64 %pos, i64 %eval)
  br label %next
next:
  %pos1 = add i64 %pos, 1
  %tail_d = call %Value @wam_deref_value(%WamState* %vm, %Value %t_raw)
  br label %walk
walk_done:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 true
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  ret i1 false
}

; Positional array, STRING values: same flat-list walk as
; @wam_object_call_posarray, but each element must be an ATOM and its
; registry id is stored at its position via @wam_assoc_i64_set -- the
; str-valued positional table (reads resolve the id back to text through
; @wam_atom_to_string, exactly as the assoc(str) kind does). Returns i1
; ok; false on call failure, a non-list output, or a non-Atom element.
define i1 @wam_object_call_posarray_str(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg, %WamAssocI64Table* %table) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %walk_init, label %rewind_fail
walk_init:
  %consfn = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  %head0 = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  br label %walk
walk:
  %cur = phi %Value [ %head0, %walk_init ], [ %tail_d, %next ]
  %pos = phi i64 [ 1, %walk_init ], [ %pos1, %next ]
  %ctag = call i32 @value_tag(%Value %cur)
  %is_comp = icmp eq i32 %ctag, 3
  br i1 %is_comp, label %check_cons, label %walk_done
check_cons:
  %cbits = call i64 @value_payload(%Value %cur)
  %ccp = inttoptr i64 %cbits to %Compound*
  %cfn_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 0
  %cfn = load i8*, i8** %cfn_slot
  %car_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 1
  %car = load i32, i32* %car_slot
  %car_ok = icmp eq i32 %car, 2
  br i1 %car_ok, label %cmp_cons, label %walk_done
cmp_cons:
  %fncmp = call i32 @strcmp(i8* %cfn, i8* %consfn)
  %is_cons = icmp eq i32 %fncmp, 0
  br i1 %is_cons, label %have_cell, label %walk_done
have_cell:
  %cargs_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 2
  %cargs = load %Value*, %Value** %cargs_slot
  %h_ptr = getelementptr %Value, %Value* %cargs, i64 0
  %h_raw = load %Value, %Value* %h_ptr
  %t_ptr = getelementptr %Value, %Value* %cargs, i64 1
  %t_raw = load %Value, %Value* %t_ptr
  %elem = call %Value @wam_deref_value(%WamState* %vm, %Value %h_raw)
  %etag = call i32 @value_tag(%Value %elem)
  %e_is_atom = icmp eq i32 %etag, 0
  br i1 %e_is_atom, label %insert, label %rewind_fail
insert:
  %eid = call i64 @value_payload(%Value %elem)
  %ignored = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %table, i64 %pos, i64 %eid)
  br label %next
next:
  %pos1 = add i64 %pos, 1
  %tail_d = call %Value @wam_deref_value(%WamState* %vm, %Value %t_raw)
  br label %walk
walk_done:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 true
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  ret i1 false
}

; Read a whole file into a fresh malloc''d buffer, writing the byte count to
; *out_total. Returns the buffer (caller frees) or null on open failure. Same
; grow-and-read loop as @wam_object_load, factored so entry-name lookup can
; read the object without a second copy of the loop.
define i8* @wamo_read_file(i8* %path, i64* %out_total) {
entry:
  %fd = call i32 @open(i8* %path, i32 0, i32 0)
  %fd_bad = icmp slt i32 %fd, 0
  br i1 %fd_bad, label %fail, label %read_init
read_init:
  %buf0 = call i8* @malloc(i64 65536)
  br label %read_loop
read_loop:
  %buf = phi i8* [ %buf0, %read_init ], [ %bufc, %after_read ]
  %cap = phi i64 [ 65536, %read_init ], [ %capc, %after_read ]
  %total = phi i64 [ 0, %read_init ], [ %total3, %after_read ]
  %free = sub i64 %cap, %total
  %low = icmp ult i64 %free, 32768
  br i1 %low, label %do_grow, label %do_read
do_grow:
  %cap.d = mul i64 %cap, 2
  %buf.d = call i8* @realloc(i8* %buf, i64 %cap.d)
  br label %do_read
do_read:
  %bufc = phi i8* [ %buf.d, %do_grow ], [ %buf, %read_loop ]
  %capc = phi i64 [ %cap.d, %do_grow ], [ %cap, %read_loop ]
  %freec = sub i64 %capc, %total
  %dst = getelementptr i8, i8* %bufc, i64 %total
  %n = call i64 @read(i32 %fd, i8* %dst, i64 %freec)
  %eof = icmp sle i64 %n, 0
  br i1 %eof, label %read_done, label %after_read
after_read:
  %total3 = add i64 %total, %n
  br label %read_loop
read_done:
  %close_ignore = call i32 @close(i32 %fd)
  store i64 %total, i64* %out_total
  ret i8* %bufc
fail:
  store i64 0, i64* %out_total
  ret i8* null
}

; Scan the named-entry table (in a buffer already read into memory) for the
; entry named `name` (namelen bytes) and return its label index, or -1 if the
; name is absent or the buffer is too small. Reads only the early table --
; version, default-entry index, entry count, then E x (name-string, index) --
; and stops at the first match, so it never parses the code section.
define i32 @wam_object_entry_index_bytes(i8* %bufc, i64 %total, i8* %name, i64 %namelen) {
entry:
  %too_small = icmp ult i64 %total, 5
  br i1 %too_small, label %notfound, label %parse
parse:
  %pv = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 4)
  %cur_v = extractvalue { i64, i64 } %pv, 1
  %pe = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %cur_v)
  %cur_e = extractvalue { i64, i64 } %pe, 1
  %pne = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %cur_e)
  %nentries = extractvalue { i64, i64 } %pne, 0
  %cur_ne = extractvalue { i64, i64 } %pne, 1
  br label %loop
loop:
  %i = phi i64 [ 0, %parse ], [ %i1, %next ]
  %cur = phi i64 [ %cur_ne, %parse ], [ %cur2, %next ]
  %done = icmp uge i64 %i, %nentries
  br i1 %done, label %notfound, label %step
step:
  %len_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %cur)
  %len = extractvalue { i64, i64 } %len_r, 0
  %cur_d = extractvalue { i64, i64 } %len_r, 1
  %str_off = add i64 %cur_d, 1
  %str = getelementptr i8, i8* %bufc, i64 %str_off
  %after_str = add i64 %str_off, %len
  %idx_r = call { i64, i64 } @wamo_next_int(i8* %bufc, i64 %total, i64 %after_str)
  %idx = extractvalue { i64, i64 } %idx_r, 0
  %cur2 = extractvalue { i64, i64 } %idx_r, 1
  %i1 = add i64 %i, 1
  %lenmatch = icmp eq i64 %len, %namelen
  br i1 %lenmatch, label %cmp, label %next
cmp:
  %cmpres = call i32 @memcmp(i8* %str, i8* %name, i64 %len)
  %eq = icmp eq i32 %cmpres, 0
  br i1 %eq, label %hit, label %next
next:
  br label %loop
hit:
  %idx32 = trunc i64 %idx to i32
  ret i32 %idx32
notfound:
  ret i32 -1
}

; Path convenience: read the file, scan its entry table for `name`, free.
; Returns the label index or -1. For the fixed-DYNLOAD call sites that name
; an entry, resolve the label index once at startup, then @wam_label_pc it
; against the loaded VM to get a PC to call.
define i32 @wam_object_entry_index(i8* %path, i8* %name, i64 %namelen) {
entry:
  %totp = alloca i64
  %bufc = call i8* @wamo_read_file(i8* %path, i64* %totp)
  %isnull = icmp eq i8* %bufc, null
  br i1 %isnull, label %fail, label %ok
ok:
  %total = load i64, i64* %totp
  %idx = call i32 @wam_object_entry_index_bytes(i8* %bufc, i64 %total, i8* %name, i64 %namelen)
  call void @free(i8* %bufc)
  ret i32 %idx
fail:
  ret i32 -1
}

; Resolve an entry name against an ALREADY-LOADED VM: scan the object''s
; materialized entry table (fields 28/29, installed by the loader) and
; convert the matching row''s label index to a PC via @wam_label_pc. This
; is the runtime-source counterpart of the two file scanners above -- it
; needs no buffer and no file, so it works uniformly for path loads,
; in-memory loads, and eval-compiled handles (dyncall_at@name). Returns
; the PC, or -1 when the VM is null, carries no entry table, or has no
; entry of that name. Cost is a short scan + memcmp per call -- cheap
; enough per record that call sites need no PC cache (and caching by VM
; pointer would go stale across an mtime-mode reload at the same
; address).
define i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %name, i64 %namelen) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %miss, label %load_tab
load_tab:
  %ent_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 28
  %rows = load %WamEntryRow*, %WamEntryRow** %ent_ptr
  %entc_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 29
  %nrows = load i32, i32* %entc_ptr
  %rows_null = icmp eq %WamEntryRow* %rows, null
  br i1 %rows_null, label %miss, label %scan
scan:
  %i = phi i32 [ 0, %load_tab ], [ %i1, %next ]
  %done = icmp sge i32 %i, %nrows
  br i1 %done, label %miss, label %check_len
check_len:
  %row = getelementptr %WamEntryRow, %WamEntryRow* %rows, i32 %i
  %len_p = getelementptr %WamEntryRow, %WamEntryRow* %row, i32 0, i32 1
  %rlen = load i64, i64* %len_p
  %len_eq = icmp eq i64 %rlen, %namelen
  br i1 %len_eq, label %check_name, label %next
check_name:
  %name_p = getelementptr %WamEntryRow, %WamEntryRow* %row, i32 0, i32 0
  %rname = load i8*, i8** %name_p
  %cmp = call i32 @memcmp(i8* %rname, i8* %name, i64 %namelen)
  %eq = icmp eq i32 %cmp, 0
  br i1 %eq, label %hit, label %next
next:
  %i1 = add i32 %i, 1
  br label %scan
hit:
  %lbl_p = getelementptr %WamEntryRow, %WamEntryRow* %row, i32 0, i32 2
  %lbl = load i32, i32* %lbl_p
  %pc = call i32 @wam_label_pc(%WamState* %vm, i32 %lbl)
  ret i32 %pc
miss:
  ret i32 -1
}

; --- eval/compile pipeline (JIT roadmap #5) ---

; Run a loaded compiler object on a source string, then load the .wamo bytes
; it emits into a fresh VM -- the "compile(Src)" primitive. The compiler entry
; is compile(Src, Wamo): the source text (interned as an atom) goes in A1 and
; the emitted byte-string atom binds A2 (out_reg = 1). The emitted bytes live
; in the persistent atom table (they survive the compiler VM''s arena rewind),
; and @wam_object_load_bytes copies what it needs, so there is no buffer to
; manage here. Returns { vm, entry_pc } for the freshly compiled object, or
; { null, 0 } if the compiler is null, fails, or yields a non-atom output.
define { %WamState*, i32 } @wam_object_eval(%WamState* %cvm, i32 %cpc, i8* %src, i64 %srclen) {
entry:
  %cvm_null = icmp eq %WamState* %cvm, null
  br i1 %cvm_null, label %fail, label %run
run:
  %srcid = call i64 @wam_intern_atom(i8* %src, i64 %srclen)
  %sv0 = insertvalue %Value undef, i32 0, 0
  %srcval = insertvalue %Value %sv0, i64 %srcid, 1
  %args = alloca %Value, i32 1
  store %Value %srcval, %Value* %args
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %cvm, i32 %cpc, i32 1, %Value* %args, i32 1)
  %ptr = extractvalue { i8*, i64, i1 } %r, 0
  %len = extractvalue { i8*, i64, i1 } %r, 1
  %ok = extractvalue { i8*, i64, i1 } %r, 2
  br i1 %ok, label %load, label %fail
load:
  %obj = call { %WamState*, i32 } @wam_object_load_bytes(i8* %ptr, i64 %len)
  ret { %WamState*, i32 } %obj
fail:
  %f0 = insertvalue { %WamState*, i32 } undef, %WamState* null, 0
  %f1 = insertvalue { %WamState*, i32 } %f0, i32 0, 1
  ret { %WamState*, i32 } %f1
}

; Lazy-load a compiler object from a file path and cache it, so repeated
; eval/compile calls reuse one loaded compiler VM (roadmap #5 DYNCACHE). A
; small fixed cache keyed by path string; a cache miss loads via
; @wam_object_load and records the result (paths are strdup''d). Beyond the
; cache capacity, later paths load uncached rather than evict.
@wam_compiler_cache_paths = internal global [16 x i8*] zeroinitializer
@wam_compiler_cache_vms = internal global [16 x %WamState*] zeroinitializer
@wam_compiler_cache_pcs = internal global [16 x i32] zeroinitializer
@wam_compiler_cache_count = internal global i32 0

define { %WamState*, i32 } @wam_object_load_cached(i8* %path) {
entry:
  %count = load i32, i32* @wam_compiler_cache_count
  br label %scan
scan:
  %i = phi i32 [ 0, %entry ], [ %inext, %scan_cont ]
  %done = icmp sge i32 %i, %count
  br i1 %done, label %miss, label %probe
probe:
  %pp = getelementptr [16 x i8*], [16 x i8*]* @wam_compiler_cache_paths, i32 0, i32 %i
  %cpath = load i8*, i8** %pp
  %cmp = call i32 @strcmp(i8* %cpath, i8* %path)
  %eq = icmp eq i32 %cmp, 0
  br i1 %eq, label %hit, label %scan_cont
scan_cont:
  %inext = add i32 %i, 1
  br label %scan
hit:
  %hvmp = getelementptr [16 x %WamState*], [16 x %WamState*]* @wam_compiler_cache_vms, i32 0, i32 %i
  %hvm = load %WamState*, %WamState** %hvmp
  %hpcp = getelementptr [16 x i32], [16 x i32]* @wam_compiler_cache_pcs, i32 0, i32 %i
  %hpc = load i32, i32* %hpcp
  %h0 = insertvalue { %WamState*, i32 } undef, %WamState* %hvm, 0
  %h1 = insertvalue { %WamState*, i32 } %h0, i32 %hpc, 1
  ret { %WamState*, i32 } %h1
miss:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %ovm = extractvalue { %WamState*, i32 } %obj, 0
  %opc = extractvalue { %WamState*, i32 } %obj, 1
  %space = icmp slt i32 %count, 16
  br i1 %space, label %record, label %ret_obj
record:
  %dup = call i8* @wam_dyn_dup_functor(i8* %path)
  %spp = getelementptr [16 x i8*], [16 x i8*]* @wam_compiler_cache_paths, i32 0, i32 %count
  store i8* %dup, i8** %spp
  %svmp = getelementptr [16 x %WamState*], [16 x %WamState*]* @wam_compiler_cache_vms, i32 0, i32 %count
  store %WamState* %ovm, %WamState** %svmp
  %spcp = getelementptr [16 x i32], [16 x i32]* @wam_compiler_cache_pcs, i32 0, i32 %count
  store i32 %opc, i32* %spcp
  %ncount = add i32 %count, 1
  store i32 %ncount, i32* @wam_compiler_cache_count
  br label %ret_obj
ret_obj:
  ret { %WamState*, i32 } %obj
}').
