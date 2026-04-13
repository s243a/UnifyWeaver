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
%   - Registers = [32 x %Value] fixed array (not HashMap/map)
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
    foreign_kernel/3                     % +Pred/Arity, +Kind, +Config (user directive)
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module('../bindings/llvm_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

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
llvm_recursive_kernel_detector(transitive_closure2,
    llvm_recursive_kernel_transitive_closure).
llvm_recursive_kernel_detector(transitive_distance3,
    llvm_recursive_kernel_transitive_distance).
llvm_recursive_kernel_detector(weighted_shortest_path3,
    llvm_recursive_kernel_weighted_shortest_path).

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
    % cds2 pass — countdown_sum2 (deterministic arithmetic).
    findall(CdsPredArity-CdsConfig,
        llvm_foreign_kernel_spec(CdsPredArity, countdown_sum2, CdsConfig),
        Cds2Entries),
    ( Cds2Entries == []
    -> StateFuncsCds = StateFuncs0, Cds2Tables = ''
    ;  build_cds2_instance_switch(Cds2Entries, Cds2ImplBody, Cds2Tables),
       replace_cds2_weak_default(StateFuncs0, Cds2ImplBody, StateFuncsCds)
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
    % wsp3 pass — mirror of the td3 pass but for weighted_shortest_path3.
    findall(WspPredArity-WspConfig,
        llvm_foreign_kernel_spec(WspPredArity, weighted_shortest_path3, WspConfig),
        Wsp3Entries),
    ( Wsp3Entries == []
    -> StateFuncs1 = StateFuncsTc, Wsp3Tables = ''
    ;  build_wsp3_instance_switch(Wsp3Entries, Wsp3ImplBody, Wsp3Tables),
       replace_wsp3_weak_default(StateFuncsTc, Wsp3ImplBody, StateFuncs1)
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
    ( Cds2Tables == '', Td3Tables == '', Tc2Tables == '', Wsp3Tables == '', Astar4Tables == ''
    -> Globals = ''
    ;  atomic_list_concat([
           '; === foreign kernel support globals ===\n',
           Cds2Tables, '\n', Tc2Tables, '\n', Td3Tables, '\n', Wsp3Tables, '\n', Astar4Tables, '\n'
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

%% write_wam_llvm_project(+Predicates, +Options, +OutputFile)
%  Generates a complete LLVM IR module for the given predicates.
write_wam_llvm_project(Predicates, Options, OutputFile) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    option(target_triple(Triple), Options, 'x86_64-pc-linux-gnu'),
    option(target_datalayout(DataLayout), Options,
        'e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

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
    compile_predicates_for_llvm(Predicates, Options, NativeCode, WamCode),

    % Prepend foreign kernel globals (fact tables, etc.) to the native
    % predicates section so they are at module scope before any use.
    ( ForeignGlobals == ''
    -> FinalNativeCode = NativeCode
    ;  atomic_list_concat([ForeignGlobals, '\n\n', NativeCode], FinalNativeCode)
    ),

    % Assemble full module
    read_template_file('templates/targets/llvm_wam/module.ll.mustache', ModuleTemplate),
    render_template(ModuleTemplate, [
        module_name=ModuleName,
        date=Date,
        target_datalayout=DataLayout,
        target_triple=Triple,
        type_definitions=TypesDef,
        value_functions=ValueFuncs,
        state_functions=StateFuncs,
        runtime_functions=RuntimeFuncs,
        native_predicates=FinalNativeCode,
        wam_predicates=WamCode,
        interop_bridge=""
    ], FullModule),

    % Write output file
    setup_call_cleanup(
        open(OutputFile, write, Stream),
        format(Stream, "~w", [FullModule]),
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
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

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
        open(OutputFile, write, Stream),
        format(Stream, "~w", [FullModule]),
        close(Stream)
    ),
    format('WAM WASM module created at: ~w~n', [OutputFile]).

%% generate_wasm_exports(+Predicates, -ExportCode)
%  Generate WASM export wrappers that expose predicates as i32-returning functions.
generate_wasm_exports([], "").
generate_wasm_exports(Predicates, ExportCode) :-
    findall(ExportFunc, (
        member(PredIndicator, Predicates),
        (   PredIndicator = _:Pred/Arity -> true
        ;   PredIndicator = Pred/Arity
        ),
        atom_string(Pred, PredStr),
        format(atom(ExportFunc),
'; WASM export: ~w/~w
; Visibility: dso_local ensures symbol is retained by wasm-ld.
; The __export_name attribute maps to the WASM export name.
define dso_local i32 @~w_wasm() #0 {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* getelementptr ([0 x %Instruction], [0 x %Instruction]* null, i32 0, i32 0),
    i32 0, i32* null, i32 0)
  %result = call i1 @run_loop(%WamState* %vm)
  %ret = zext i1 %result to i32
  ret i32 %ret
}', [PredStr, Arity, PredStr])
    ), ExportFuncs),
    atomic_list_concat(ExportFuncs, '\n\n', ExportFuncsStr),
    % Add WASM export attribute group
    format(atom(ExportCode),
'~w

; WASM export attributes
attributes #0 = { "wasm-export-name"="query" }', [ExportFuncsStr]).

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
read_template_file(Path, Content) :-
    (   exists_file(Path)
    ->  read_file_to_string(Path, Content, [])
    ;   format(atom(Content), "; Template not found: ~w", [Path])
    ).

%% compile_predicates_for_llvm(+Predicates, +Options, -NativeCode, -WamCode)
%  Collects all predicate code into lists, then joins once (O(n) concatenation).
compile_predicates_for_llvm(Predicates, Options, NativeCode, WamCode) :-
    compile_predicates_collect(Predicates, Options, NativeParts, WamParts),
    atomic_list_concat(NativeParts, '\n\n', NativeCode),
    atomic_list_concat(WamParts, '\n\n', WamCode).

compile_predicates_collect([], _, [], []).
compile_predicates_collect([PredIndicator|Rest], Options, NativeParts, WamParts) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    compile_predicates_collect(Rest, Options, RestNative, RestWam),
    (   % M5.6: foreign kernel lowering — check spec table first.
        lookup_foreign_kernel_spec(Pred, Arity, Kind, _Config)
    ->  format(user_error, '  ~w/~w: foreign kernel (~w)~n', [Pred, Arity, Kind]),
        compile_foreign_kernel_predicate(Pred/Arity, Kind, Arity, Options, PredCode),
        NativeParts = RestNative,
        WamParts = [PredCode | RestWam]
    ;   % Try native LLVM lowering
        catch(
            llvm_target:compile_predicate_to_llvm(Module:Pred/Arity, Options, PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        NativeParts = [PredCode | RestNative],
        WamParts = RestWam
    ;   % Fall back to WAM compilation
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamRaw),
        compile_wam_predicate_to_llvm(Pred/Arity, WamRaw, Options, PredCode)
    ->  format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
        NativeParts = RestNative,
        WamParts = [PredCode | RestWam]
    ;   % Neither worked
        format(user_error, '  ~w/~w: compilation failed~n', [Pred, Arity]),
        NativeParts = RestNative,
        format(atom(FailComment), '; ~w/~w: compilation failed', [Pred, Arity]),
        WamParts = [FailComment | RestWam]
    ).

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
    format(atom(LLVMCode),
'define i1 @step(%WamState* %vm, %Instruction* %instr) {
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
  ]

~w

default:
  ret i1 false
}', [CasesCode]).

compile_llvm_step_case(CaseCode) :-
    wam_llvm_case(Label, BodyCode),
    format(atom(CaseCode), '~w:\n~w', [Label, BodyCode]).

% --- Head Unification Instructions ---

wam_llvm_case('get_constant',
'  ; op1 = constant value (packed), op2 = register index
  %gc.reg_idx = trunc i64 %op2 to i32
  %gc.current = call %Value @wam_get_reg(%WamState* %vm, i32 %gc.reg_idx)
  %gc.is_unb = call i1 @value_is_unbound(%Value %gc.current)
  br i1 %gc.is_unb, label %gc.bind, label %gc.check_eq

gc.bind:
  ; Unbound: bind to constant
  call void @wam_trail_binding(%WamState* %vm, i32 %gc.reg_idx)
  %gc.const_val = insertvalue %Value undef, i32 0, 0           ; tag from op1 high bits
  %gc.const_v2 = insertvalue %Value %gc.const_val, i64 %op1, 1
  call void @wam_set_reg(%WamState* %vm, i32 %gc.reg_idx, %Value %gc.const_v2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gc.check_eq:
  ; Bound: check equality
  %gc.expected = insertvalue %Value undef, i32 0, 0
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
'  ; op1 = Xn index, op2 = Ai index
  %gval.ai = trunc i64 %op2 to i32
  %gval.xn = trunc i64 %op1 to i32
  %gval.va = call %Value @wam_get_reg(%WamState* %vm, i32 %gval.ai)
  %gval.vx = call %Value @wam_get_reg(%WamState* %vm, i32 %gval.xn)
  ; Check if either is unbound
  %gval.a_unb = call i1 @value_is_unbound(%Value %gval.va)
  br i1 %gval.a_unb, label %gval.bind_a, label %gval.check_x

gval.bind_a:
  call void @wam_trail_binding(%WamState* %vm, i32 %gval.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %gval.ai, %Value %gval.vx)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.check_x:
  %gval.x_unb = call i1 @value_is_unbound(%Value %gval.vx)
  br i1 %gval.x_unb, label %gval.bind_x, label %gval.check_eq

gval.bind_x:
  call void @wam_trail_binding(%WamState* %vm, i32 %gval.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %gval.xn, %Value %gval.va)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.check_eq:
  %gval.eq = call i1 @value_equals(%Value %gval.va, %Value %gval.vx)
  br i1 %gval.eq, label %gval.match, label %gval.fail

gval.match:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.fail:
  ret i1 false').

% --- Structure/List Head Unification ---

wam_llvm_case('get_structure',
'  ; get_structure: op2 = Ai register index
  ; Write mode (unbound): push functor marker on heap, bind Ai to Ref, push WriteCtx
  ; Read mode (bound): push args onto stack as UnifyCtx
  %gs.ai = trunc i64 %op2 to i32
  %gs.val = call %Value @wam_get_reg(%WamState* %vm, i32 %gs.ai)
  %gs.unb = call i1 @value_is_unbound(%Value %gs.val)
  br i1 %gs.unb, label %gs.write, label %gs.read

gs.write:
  ; Write mode: heap marker + Ref + WriteCtx
  %gs.marker = call %Value @value_atom(i8* null)
  %gs.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %gs.marker)
  %gs.ref = call %Value @value_ref(i32 %gs.addr)
  call void @wam_trail_binding(%WamState* %vm, i32 %gs.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %gs.ai, %Value %gs.ref)
  ; Push WriteCtx with arity (encoded in op1 low bits, default 2)
  %gs.arity = trunc i64 %op1 to i32
  %gs.arity_zero = icmp eq i32 %gs.arity, 0
  %gs.arity_safe = select i1 %gs.arity_zero, i32 2, i32 %gs.arity
  call void @wam_push_write_ctx(%WamState* %vm, i32 %gs.arity_safe)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gs.read:
  ; Read mode: succeed and advance (args decomposition via step-level context)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('get_list',
'  ; get_list: op1 = Ai register index
  ; Like get_structure but for lists (./2, arity=2)
  %gl.ai = trunc i64 %op1 to i32
  %gl.val = call %Value @wam_get_reg(%WamState* %vm, i32 %gl.ai)
  %gl.unb = call i1 @value_is_unbound(%Value %gl.val)
  br i1 %gl.unb, label %gl.write, label %gl.read

gl.write:
  %gl.marker = call %Value @value_atom(i8* null)
  %gl.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %gl.marker)
  %gl.ref = call %Value @value_ref(i32 %gl.addr)
  call void @wam_trail_binding(%WamState* %vm, i32 %gl.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %gl.ai, %Value %gl.ref)
  call void @wam_push_write_ctx(%WamState* %vm, i32 2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gl.read:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

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
  %uv.var = call %Value @value_unbound(i8* null)
  %uv.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %uv.var)
  %uv.dec = call i32 @wam_write_ctx_dec(%WamState* %vm)
  call void @wam_trail_binding(%WamState* %vm, i32 %uv.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %uv.xn, %Value %uv.var)
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
  ; Read mode: get expected arg, compare with register
  %uvl.expected = call %Value @wam_unify_ctx_next(%WamState* %vm)
  %uvl.actual = call %Value @wam_get_reg(%WamState* %vm, i32 %uvl.xn)
  ; Succeed if equal, or if either is unbound (bind the unbound one)
  %uvl.eq = call i1 @value_equals(%Value %uvl.expected, %Value %uvl.actual)
  %uvl.exp_unb = call i1 @value_is_unbound(%Value %uvl.expected)
  %uvl.act_unb = call i1 @value_is_unbound(%Value %uvl.actual)
  %uvl.ok1 = or i1 %uvl.eq, %uvl.exp_unb
  %uvl.ok = or i1 %uvl.ok1, %uvl.act_unb
  br i1 %uvl.ok, label %uvl.read_ok, label %uvl.fail

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
  %uvl.dec = call i32 @wam_write_ctx_dec(%WamState* %vm)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uvl.fail:
  ret i1 false').

wam_llvm_case('unify_constant',
'  ; unify_constant: op1 = constant value (packed)
  ; Read mode: check next arg equals constant
  ; Write mode: push constant onto heap
  %uc.stype = call i32 @wam_peek_stack_type(%WamState* %vm)
  %uc.is_read = icmp eq i32 %uc.stype, 1
  %uc.val = insertvalue %Value undef, i32 0, 0
  %uc.val2 = insertvalue %Value %uc.val, i64 %op1, 1
  br i1 %uc.is_read, label %uc.read, label %uc.write

uc.read:
  ; Read mode: get expected, check equality
  %uc.expected = call %Value @wam_unify_ctx_next(%WamState* %vm)
  %uc.eq = call i1 @value_equals(%Value %uc.expected, %Value %uc.val2)
  %uc.exp_unb = call i1 @value_is_unbound(%Value %uc.expected)
  %uc.ok = or i1 %uc.eq, %uc.exp_unb
  br i1 %uc.ok, label %uc.read_ok, label %uc.fail

uc.read_ok:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uc.write:
  ; Write mode: push onto heap
  %uc.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %uc.val2)
  %uc.dec = call i32 @wam_write_ctx_dec(%WamState* %vm)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

uc.fail:
  ret i1 false').

% --- Body Construction Instructions ---

wam_llvm_case('put_constant',
'  ; op1 = constant value (packed), op2 = register index
  %pc.reg_idx = trunc i64 %op2 to i32
  %pc.val = insertvalue %Value undef, i32 0, 0
  %pc.val2 = insertvalue %Value %pc.val, i64 %op1, 1
  call void @wam_trail_binding(%WamState* %vm, i32 %pc.reg_idx)
  call void @wam_set_reg(%WamState* %vm, i32 %pc.reg_idx, %Value %pc.val2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('put_variable',
'  ; op1 = Xn index, op2 = Ai index
  %pv.xn = trunc i64 %op1 to i32
  %pv.ai = trunc i64 %op2 to i32
  ; Create unbound variable
  %pv.pc = call i32 @wam_get_pc(%WamState* %vm)
  %pv.pc_ext = zext i32 %pv.pc to i64
  %pv.var = call %Value @value_unbound(i8* null)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.xn)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.xn, %Value %pv.var)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.ai, %Value %pv.var)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('put_value',
'  ; op1 = Xn index, op2 = Ai index
  %pvl.xn = trunc i64 %op1 to i32
  %pvl.ai = trunc i64 %op2 to i32
  %pvl.val = call %Value @wam_get_reg(%WamState* %vm, i32 %pvl.xn)
  call void @wam_trail_binding(%WamState* %vm, i32 %pvl.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pvl.ai, %Value %pvl.val)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('put_structure',
'  ; put_structure: op2 = Ai register index
  ; Push structure marker on heap, bind Ai to Ref, push WriteCtx
  %ps.ai = trunc i64 %op2 to i32
  %ps.marker = call %Value @value_atom(i8* null)
  %ps.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %ps.marker)
  %ps.ref = call %Value @value_ref(i32 %ps.addr)
  call void @wam_set_reg(%WamState* %vm, i32 %ps.ai, %Value %ps.ref)
  %ps.arity = trunc i64 %op1 to i32
  %ps.arity_zero = icmp eq i32 %ps.arity, 0
  %ps.arity_safe = select i1 %ps.arity_zero, i32 2, i32 %ps.arity
  call void @wam_push_write_ctx(%WamState* %vm, i32 %ps.arity_safe)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('put_list',
'  ; put_list: op1 = Ai register index
  ; Push list marker on heap, bind Ai to Ref, push WriteCtx(2)
  %pl.ai = trunc i64 %op1 to i32
  %pl.marker = call %Value @value_atom(i8* null)
  %pl.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %pl.marker)
  %pl.ref = call %Value @value_ref(i32 %pl.addr)
  call void @wam_set_reg(%WamState* %vm, i32 %pl.ai, %Value %pl.ref)
  call void @wam_push_write_ctx(%WamState* %vm, i32 2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('set_variable',
'  ; set_variable: op1 = Xn register index
  ; Create unbound var, push on heap, store in Xn, decrement WriteCtx
  %sv.xn = trunc i64 %op1 to i32
  %sv.var = call %Value @value_unbound(i8* null)
  %sv.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %sv.var)
  %sv.dec = call i32 @wam_write_ctx_dec(%WamState* %vm)
  call void @wam_set_reg(%WamState* %vm, i32 %sv.xn, %Value %sv.var)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('set_value',
'  ; set_value: op1 = Xn register index
  ; Push Xn value onto heap, decrement WriteCtx
  %sve.xn = trunc i64 %op1 to i32
  %sve.val = call %Value @wam_get_reg(%WamState* %vm, i32 %sve.xn)
  %sve.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %sve.val)
  %sve.dec = call i32 @wam_write_ctx_dec(%WamState* %vm)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('set_constant',
'  ; set_constant: op1 = constant value (packed)
  ; Push constant onto heap, decrement WriteCtx
  %sc.val = insertvalue %Value undef, i32 0, 0
  %sc.val2 = insertvalue %Value %sc.val, i64 %op1, 1
  %sc.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %sc.val2)
  %sc.dec = call i32 @wam_write_ctx_dec(%WamState* %vm)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% --- Control Instructions ---

wam_llvm_case('allocate',
'  ; Push environment frame: save CP on stack
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
  ; Pop stack down to this frame (exclusive)
  store i32 %dealloc.prev_idx, i32* %dealloc.ss_ptr
  br label %dealloc.done

dealloc.done:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('do_call',
'  ; op1 = label index, op2 = arity
  %call.label = trunc i64 %op1 to i32
  %call.target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %call.label)
  %call.valid = icmp sge i32 %call.target_pc, 0
  br i1 %call.valid, label %call.go, label %call.fail

call.go:
  ; Save continuation
  %call.pc = call i32 @wam_get_pc(%WamState* %vm)
  %call.next = add i32 %call.pc, 1
  call void @wam_set_cp(%WamState* %vm, i32 %call.next)
  call void @wam_set_pc(%WamState* %vm, i32 %call.target_pc)
  ret i1 true

call.fail:
  ret i1 false').

wam_llvm_case('do_execute',
'  ; op1 = label index
  %exec.label = trunc i64 %op1 to i32
  %exec.target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %exec.label)
  %exec.valid = icmp sge i32 %exec.target_pc, 0
  br i1 %exec.valid, label %exec.go, label %exec.fail

exec.go:
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
  ; Push choice point
  %tme.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %tme.cpn = load i32, i32* %tme.cpn_ptr
  %tme.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %tme.cps = load %ChoicePoint*, %ChoicePoint** %tme.cps_ptr
  %tme.cp_slot = getelementptr %ChoicePoint, %ChoicePoint* %tme.cps, i32 %tme.cpn
  ; Set next_pc
  %tme.npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 0
  store i32 %tme.next_pc, i32* %tme.npc_ptr
  ; Save registers (copy 32 x %Value)
  %tme.dst_regs = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 1, i32 0
  %tme.src_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %tme.dst_raw = bitcast %Value* %tme.dst_regs to i8*
  %tme.src_raw = bitcast %Value* %tme.src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tme.dst_raw, i8* %tme.src_raw, i64 512, i1 false)
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
  ; Increment choice point count
  %tme.new_cpn = add i32 %tme.cpn, 1
  store i32 %tme.new_cpn, i32* %tme.cpn_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('retry_me_else',
'  ; op1 = label index for next alternative
  %rme.label = trunc i64 %op1 to i32
  %rme.next_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %rme.label)
  ; Update top choice point next_pc
  %rme.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %rme.cpn = load i32, i32* %rme.cpn_ptr
  %rme.top_idx = sub i32 %rme.cpn, 1
  %rme.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %rme.cps = load %ChoicePoint*, %ChoicePoint** %rme.cps_ptr
  %rme.top = getelementptr %ChoicePoint, %ChoicePoint* %rme.cps, i32 %rme.top_idx
  %rme.npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %rme.top, i32 0, i32 0
  store i32 %rme.next_pc, i32* %rme.npc_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('trust_me',
'  ; Pop top choice point
  %tm.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %tm.cpn = load i32, i32* %tm.cpn_ptr
  %tm.new_cpn = sub i32 %tm.cpn, 1
  store i32 %tm.new_cpn, i32* %tm.cpn_ptr
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

% switch_on_constant_a2: nop fallthrough for now.
wam_llvm_case('switch_on_constant_a2',
'  ; Nop fallthrough: just advance PC and continue.
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% begin_aggregate: push an aggregate-frame choice point and reset accumulator.
% op1 = (agg_type)
% op2 = (value_reg_idx << 16) | result_reg_idx
wam_llvm_case('begin_aggregate',
'  %ba.agg_type = trunc i64 %op1 to i32
  %ba.op2_trunc = trunc i64 %op2 to i32
  %ba.val_reg = lshr i32 %ba.op2_trunc, 16
  %ba.res_reg = and i32 %ba.op2_trunc, 65535

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
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %ba.dst_raw, i8* %ba.src_raw, i64 512, i1 false)

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
  store i32 0, i32* %ba.arpc_ptr  ; placeholder, end_aggregate updates this

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
  %ea.val = call %Value @wam_get_reg(%WamState* %vm, i32 %ea.val_reg)

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

% ============================================================================
% PHASE 3: Helper predicates → LLVM functions
% ============================================================================

%% compile_wam_helpers_to_llvm(+Options, -LLVMCode)
%  Generates LLVM IR for WAM runtime helpers.
compile_wam_helpers_to_llvm(_Options, LLVMCode) :-
    compile_backtrack_to_llvm(BacktrackCode),
    compile_unwind_trail_to_llvm(UnwindCode),
    compile_execute_builtin_to_llvm(BuiltinCode),
    compile_eval_arith_to_llvm(ArithCode),
    atomic_list_concat([
        BacktrackCode, '\n\n',
        UnwindCode, '\n\n',
        BuiltinCode, '\n\n',
        ArithCode
    ], LLVMCode).

compile_backtrack_to_llvm(Code) :-
    Code = 'define i1 @backtrack(%WamState* %vm) {
entry:
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
  br i1 %is_foreign, label %do_foreign_yield, label %restore

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

  ; Restore registers
  %dst_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %src_regs = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 1, i32 0
  %dst_raw = bitcast %Value* %dst_regs to i8*
  %src_raw = bitcast %Value* %src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst_raw, i8* %src_raw, i64 512, i1 false)

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
  ; Restore old value to register
  %reg_ptr = getelementptr %TrailEntry, %TrailEntry* %te, i32 0, i32 0
  %reg_idx = load i32, i32* %reg_ptr
  %old_val_ptr = getelementptr %TrailEntry, %TrailEntry* %te, i32 0, i32 1
  %old_val = load %Value, %Value* %old_val_ptr
  call void @wam_set_reg(%WamState* %vm, i32 %reg_idx, %Value %old_val)
  br label %loop

done:
  ret void
}'.

compile_execute_builtin_to_llvm(Code) :-
    Code = '; Execute builtin operations
; Dispatches on integer op codes:
;   0 = is/2, 1 = >/2, 2 = </2, 3 = >=/2, 4 = =</2
;   5 = =:=/2, 6 = =\\=/2, 7 = ==/2, 8 = true/0, 9 = fail/0
;   10 = !/0, 11 = write/1, 12 = nl/0
;   13 = atom/1, 14 = integer/1, 15 = float/1, 16 = number/1
;   17 = compound/1, 18 = var/1, 19 = nonvar/1, 20 = is_list/1
define i1 @execute_builtin(%WamState* %vm, i32 %op, i32 %arity) {
entry:
  switch i32 %op, label %unknown [
    i32 0, label %builtin_is
    i32 1, label %builtin_gt
    i32 2, label %builtin_lt
    i32 3, label %builtin_ge
    i32 4, label %builtin_le
    i32 5, label %builtin_arith_eq
    i32 6, label %builtin_arith_ne
    i32 7, label %builtin_eq
    i32 8, label %builtin_true
    i32 9, label %builtin_fail
    i32 10, label %builtin_cut
    i32 14, label %builtin_integer_check
    i32 18, label %builtin_var
    i32 19, label %builtin_nonvar
  ]

builtin_is:
  ; A1 is result, A2 is expression — evaluate A2 via eval_arith and unify with A1
  %is.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %is.result = call i64 @eval_arith(%WamState* %vm, %Value %is.a2)
  %is.result_val = call %Value @value_integer(i64 %is.result)
  %is.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %is.a1_unb = call i1 @value_is_unbound(%Value %is.a1)
  br i1 %is.a1_unb, label %is.do_bind, label %is.check_eq

is.do_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %is.result_val)
  ret i1 true

is.check_eq:
  %is.eq = call i1 @value_equals(%Value %is.a1, %Value %is.result_val)
  ret i1 %is.eq

builtin_gt:
  %gt.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %gt.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %gt.v1 = call i64 @value_payload(%Value %gt.a1)
  %gt.v2 = call i64 @value_payload(%Value %gt.a2)
  %gt.r = icmp sgt i64 %gt.v1, %gt.v2
  ret i1 %gt.r

builtin_lt:
  %lt.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %lt.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %lt.v1 = call i64 @value_payload(%Value %lt.a1)
  %lt.v2 = call i64 @value_payload(%Value %lt.a2)
  %lt.r = icmp slt i64 %lt.v1, %lt.v2
  ret i1 %lt.r

builtin_ge:
  %ge.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ge.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %ge.v1 = call i64 @value_payload(%Value %ge.a1)
  %ge.v2 = call i64 @value_payload(%Value %ge.a2)
  %ge.r = icmp sge i64 %ge.v1, %ge.v2
  ret i1 %ge.r

builtin_le:
  %le.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %le.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %le.v1 = call i64 @value_payload(%Value %le.a1)
  %le.v2 = call i64 @value_payload(%Value %le.a2)
  %le.r = icmp sle i64 %le.v1, %le.v2
  ret i1 %le.r

builtin_arith_eq:
  %aeq.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %aeq.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %aeq.v1 = call i64 @value_payload(%Value %aeq.a1)
  %aeq.v2 = call i64 @value_payload(%Value %aeq.a2)
  %aeq.r = icmp eq i64 %aeq.v1, %aeq.v2
  ret i1 %aeq.r

builtin_arith_ne:
  %ane.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ane.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %ane.v1 = call i64 @value_payload(%Value %ane.a1)
  %ane.v2 = call i64 @value_payload(%Value %ane.a2)
  %ane.r = icmp ne i64 %ane.v1, %ane.v2
  ret i1 %ane.r

builtin_eq:
  %eq.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %eq.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %eq.r = call i1 @value_equals(%Value %eq.a1, %Value %eq.a2)
  ret i1 %eq.r

builtin_true:
  ret i1 true

builtin_fail:
  ret i1 false

builtin_cut:
  ; Clear all choice points
  %cut.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  store i32 0, i32* %cut.cpn_ptr
  ret i1 true

builtin_integer_check:
  %ic.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ic.tag = call i32 @value_tag(%Value %ic.a1)
  %ic.r = icmp eq i32 %ic.tag, 1
  ret i1 %ic.r

builtin_var:
  %var.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %var.r = call i1 @value_is_unbound(%Value %var.a1)
  ret i1 %var.r

builtin_nonvar:
  %nv.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %nv.r = call i1 @value_is_unbound(%Value %nv.a1)
  %nv.nr = xor i1 %nv.r, true
  ret i1 %nv.nr

unknown:
  ret i1 false
}'.

compile_eval_arith_to_llvm(Code) :-
    Code = '; Evaluate arithmetic expression
; Takes a Value, returns the integer payload.
; For compound ops (tag=3), extracts functor and recursively evaluates args.
; For register refs (tag=6 unbound with name starting A/X), dereferences.
define i64 @eval_arith(%WamState* %vm, %Value %expr) {
entry:
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
  switch i8 %fn_first, label %fail [
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
  %mul_r = mul i64 %a, %b
  ret i64 %mul_r

do_div:
  %div_zero = icmp eq i64 %b, 0
  br i1 %div_zero, label %fail, label %do_div_ok

do_div_ok:
  %div_r = sdiv i64 %a, %b
  ret i64 %div_r

check_unary:
  %is_unary = icmp eq i32 %arity, 1
  br i1 %is_unary, label %eval_unary, label %fail

eval_unary:
  %u_arg_ptr = getelementptr %Value, %Value* %args, i32 0
  %u_arg = load %Value, %Value* %u_arg_ptr
  %u_val = call i64 @eval_arith(%WamState* %vm, %Value %u_arg)
  %u_fn_first = load i8, i8* %fn_ptr
  %u_is_neg = icmp eq i8 %u_fn_first, 45  ; \'-\'
  br i1 %u_is_neg, label %do_neg, label %fail

do_neg:
  %neg_r = sub i64 0, %u_val
  ret i64 %neg_r

fail:
  ret i64 0
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
    format(atom(Lit), '{ i32 0, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
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
    format(atom(Lit), '{ i32 3, i64 0, i64 ~w } ; get_structure ~w', [AiIdx, F]).
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
    format(atom(Lit), '{ i32 7, i64 ~w, i64 0 }', [PackedVal]).

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

%% wam_instruction_to_llvm_literal(+WamInstr, +LabelMap, -LLVMLiteral)
%  Label-aware variant. LabelMap is a list of LabelName-Index pairs.
%  NOTE: trailing `; comment` was removed because LLVM treats `;` as a
%  line comment to end-of-line, which eats the comma separator in array
%  constants and produces a parser error.
wam_instruction_to_llvm_literal(call(P, N), LabelMap, Lit) :- !,
    lookup_label_index(P, LabelMap, LabelIdx),
    format(atom(Lit), '{ i32 18, i64 ~w, i64 ~w }', [LabelIdx, N]).
wam_instruction_to_llvm_literal(execute(P), LabelMap, Lit) :- !,
    lookup_label_index(P, LabelMap, LabelIdx),
    format(atom(Lit), '{ i32 19, i64 ~w, i64 0 }', [LabelIdx]).
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

% --- Value packing helpers ---

llvm_pack_value(atom(A), Packed) :- !,
    intern_atom(A, Packed).
llvm_pack_value(integer(I), I) :- !.
llvm_pack_value(N, N) :- integer(N), !.
llvm_pack_value(N, Packed) :- float(N), !, Packed is truncate(N).
llvm_pack_value(A, Packed) :- atom(A), !, intern_atom(A, Packed).
llvm_pack_value(_, 0).

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
builtin_op_to_id(_, 99).  % Unknown

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
    split_string(Line, " \t", " \t", Parts),
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
    ;   (   option(wam_strict_labels(true), Options)
        ->  throw(error(unknown_label(LabelName),
                'Label not found in LabelMap — enable wam_strict_labels(false) to allow fallback'))
        ;   format(user_error,
                'Warning: unknown label "~w" in WAM LLVM codegen, defaulting to index 0~n',
                [LabelName]),
            Index = 0
        )
    ).

% Instructions that need label resolution:
% NOTE: trailing `; comment` removed — LLVM line comments eat the comma
% separator in array constants. The label name is preserved for humans by
% having `llvm-dis` show labels resolved from the label array, not inline.
wam_line_to_llvm_literal_resolved(["call", P, N], LabelMap, Lit) :- !,
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Arity, CN) -> true ; Arity = 0 ),
    lookup_label_index(CP, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 18, i64 ~w, i64 ~w }', [LabelIdx, Arity]).
wam_line_to_llvm_literal_resolved(["execute", P], LabelMap, Lit) :- !,
    clean_comma(P, CP),
    lookup_label_index(CP, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 19, i64 ~w, i64 0 }', [LabelIdx]).
wam_line_to_llvm_literal_resolved(["try_me_else", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    lookup_label_index(CL, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 22, i64 ~w, i64 0 }', [LabelIdx]).
wam_line_to_llvm_literal_resolved(["retry_me_else", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    lookup_label_index(CL, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 23, i64 ~w, i64 0 }', [LabelIdx]).
% switch_on_constant: defer until compile_wam_predicate_to_llvm can
% allocate a switch table global. Returns a switch_deferred(_) term.
wam_line_to_llvm_literal_resolved(["switch_on_constant" | EntryParts], LabelMap,
        switch_deferred(constant, Entries)) :- !,
    parse_switch_entries(EntryParts, LabelMap, Entries).
wam_line_to_llvm_literal_resolved(["switch_on_structure" | _], _, Lit) :- !,
    % nop fallthrough — the try_me_else chain still runs.
    Lit = '%Instruction { i32 26, i64 0, i64 0 }'.
wam_line_to_llvm_literal_resolved(["switch_on_constant_a2" | _], _, Lit) :- !,
    Lit = '%Instruction { i32 27, i64 0, i64 0 }'.
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
    ( sub_string(Clean, Before, 1, After, ":")
    -> sub_string(Clean, 0, Before, _, KeyStr),
       sub_string(Clean, _, After, 0, LabelStr)
    ;  KeyStr = Clean, LabelStr = ""
    ),
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
    format(atom(Lit), '%Instruction { i32 0, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
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
    format(atom(Lit), '%Instruction { i32 7, i64 ~w, i64 0 }', [PackedVal]).

wam_line_to_llvm_literal(["put_constant", C, Ai], Lit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    llvm_pack_value_str(CC, PackedVal),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, RegIdx),
    format(atom(Lit), '%Instruction { i32 8, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
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
    clean_comma(FN, _CFN), clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 11, i64 0, i64 ~w }', [AiIdx]).
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
    llvm_pack_value_str(CC, PackedVal),
    format(atom(Lit), '%Instruction { i32 15, i64 ~w, i64 0 }', [PackedVal]).

wam_line_to_llvm_literal(["allocate"], '%Instruction { i32 16, i64 0, i64 0 }').
wam_line_to_llvm_literal(["deallocate"], '%Instruction { i32 17, i64 0, i64 0 }').

% begin_aggregate type, ValueReg, ResultReg
wam_line_to_llvm_literal(["begin_aggregate", TypeStr, ValRegStr, ResRegStr], Lit) :- !,
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
    atom_string(COp, COpAtom),
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
    ;   atom_string(A, Str),
        llvm_pack_value(atom(A), Packed)
    ).

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
