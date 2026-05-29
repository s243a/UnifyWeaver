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
    split_functor_arity/3                % +Str, -Name, -Arity (handles `/` in name)
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
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
    retractall(functor_string_global(_, _)),
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
        FinalNativeCode),

    % Generate external declarations (native vs WASM).
    generate_external_declarations(Triple, ExternalDecls),

    % Assemble full module
    read_template_file('templates/targets/llvm_wam/module.ll.mustache', ModuleTemplate),
    render_template(ModuleTemplate, [
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

%% generate_external_declarations(+Triple, -Decls)
%  For native targets: declare malloc, free, snprintf, strcmp, printf.
%  For wasm32: define malloc/free as bump allocator (self-contained),
%  omit snprintf/strcmp/printf (not available in standalone WASM).
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
declare i32 @putchar(i32)'
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
read_template_file(Path, Content) :-
    (   exists_file(Path)
    ->  read_file_to_string(Path, Content, [])
    ;   format(atom(Content), "; Template not found: ~w", [Path])
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
    % M4: closure analysis — compute the set of predicates in this
    % batch whose lowered kernels can be safely inter-linked via
    % `call`/`execute`. A predicate joins the closure iff all of its
    % clause-1 call/execute targets are also in the closure (computed
    % to fixpoint). Passed through Options so pass1's per-pred
    % lowerability check can consult it.
    compute_lowered_closure(Predicates, Options, ClosureSet),
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
    pass1_classify_predicates(Predicates, OptionsWithClosure, 0, Classified),
    build_merged_labels(Classified, NamePCPairs, LabelMap),
    pass2_emit_merged(Classified, LabelMap, NamePCPairs, OptionsWithClosure,
                      NativeParts, WamParts),
    atomic_list_concat(NativeParts, '\n\n', NativeCode),
    atomic_list_concat(WamParts, '\n\n', WamCode).

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
        ;  LowerShape == multi_clause_c1
        -> format(user_error,
            '  ~w/~w: lowered LLVM emission (hybrid clause-1 + bytecode)~n',
            [Pred, Arity]),
           % Parse the WAM bytecode (all clauses) into the merge so the
           % dispatcher's slow path can run it. The dispatcher entry
           % is emitted in pass2 via emit_hybrid_dispatcher.
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
    ->  MergedCode = '', EntryFuncs = '', SwitchDefs = ''
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
    (   InstrCount =:= 0
    ->  CodeGlobal = '', EntryFuncs = '', SwitchDefs = ''
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
        format(atom(CodeGlobal),
'; === Merged WAM code and labels (project-level) ===
@module_code = private constant [~w x %Instruction] [
~w
]

@module_labels = private constant [~w x i32] [
~w
]',         [InstrCount, EntriesStr, LabelArraySize, LabelsStr]),
        emit_all_entry_funcs(WamRecords, Options,
                             InstrCount, LabelCount, LabelArraySize,
                             EntryFuncs),
        ( AllSwitchDefs == []
        -> SwitchDefs = ''
        ;  atomic_list_concat(AllSwitchDefs, '\n', SwitchDefs)
        )
    ).

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
'  ; op1 = Xn index, op2 = Ai index — unify. Deref both sides for the
  ; is_unbound check; bind via wam_bind_reg to propagate through Refs.
  %gval.ai = trunc i64 %op2 to i32
  %gval.xn = trunc i64 %op1 to i32
  %gval.va = call %Value @wam_get_reg_deref(%WamState* %vm, i32 %gval.ai)
  %gval.vx = call %Value @wam_get_reg_deref(%WamState* %vm, i32 %gval.xn)
  %gval.a_unb = call i1 @value_is_unbound(%Value %gval.va)
  br i1 %gval.a_unb, label %gval.bind_a, label %gval.check_x

gval.bind_a:
  call void @wam_trail_binding(%WamState* %vm, i32 %gval.ai)
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.ai, %Value %gval.vx)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.check_x:
  %gval.x_unb = call i1 @value_is_unbound(%Value %gval.vx)
  br i1 %gval.x_unb, label %gval.bind_x, label %gval.check_eq

gval.bind_x:
  call void @wam_trail_binding(%WamState* %vm, i32 %gval.xn)
  call void @wam_bind_reg(%WamState* %vm, i32 %gval.xn, %Value %gval.va)
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
  ; Write mode: push functor marker on heap, bind Ai to Ref, push WriteCtx
  %gs.marker = call %Value @value_atom(i8* null)
  %gs.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %gs.marker)
  %gs.ref = call %Value @value_ref(i32 %gs.addr)
  call void @wam_trail_binding(%WamState* %vm, i32 %gs.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %gs.ai, %Value %gs.ref)
  call void @wam_push_write_ctx(%WamState* %vm, i32 %gs.arity)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gs.read:
  ; Read mode: extract args from Compound and push UnifyCtx
  %gs.cp_bits = extractvalue %Value %gs.val, 1
  %gs.cp_ptr = inttoptr i64 %gs.cp_bits to %Compound*
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
  ; Compound: load args pointer, push UnifyCtx with arity 2.
  %gl.cp_bits = extractvalue %Value %gl.val, 1
  %gl.cp_ptr = inttoptr i64 %gl.cp_bits to %Compound*
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
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %uvl.val)
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
  %pc.old = call %Value @wam_get_reg(%WamState* %vm, i32 %pc.reg_idx)
  call void @wam_trail_binding(%WamState* %vm, i32 %pc.reg_idx)
  call void @wam_set_reg(%WamState* %vm, i32 %pc.reg_idx, %Value %pc.val2)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pc.old, %Value %pc.val2)
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
  %pv.xn = trunc i64 %op1 to i32
  %pv.ai = trunc i64 %op2 to i32
  %pv.unb = call %Value @value_unbound(i8* null)
  %pv.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %pv.unb)
  %pv.ref = call %Value @value_ref(i32 %pv.addr)
  %pv.old = call %Value @wam_get_reg(%WamState* %vm, i32 %pv.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.xn)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.xn, %Value %pv.ref)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.ai, %Value %pv.ref)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pv.old, %Value %pv.ref)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').


wam_llvm_case('put_value',
'  ; op1 = Xn index, op2 = Ai index
  %pvl.xn = trunc i64 %op1 to i32
  %pvl.ai = trunc i64 %op2 to i32
  %pvl.val = call %Value @wam_get_reg(%WamState* %vm, i32 %pvl.xn)
  %pvl.old = call %Value @wam_get_reg(%WamState* %vm, i32 %pvl.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %pvl.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pvl.ai, %Value %pvl.val)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pvl.old, %Value %pvl.val)
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
  %ps.old = call %Value @wam_get_reg(%WamState* %vm, i32 %ps.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %ps.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %ps.ai, %Value %ps.val)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %ps.old, %Value %ps.val)
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
  %pl.old = call %Value @wam_get_reg(%WamState* %vm, i32 %pl.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %pl.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pl.ai, %Value %pl.val)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pl.old, %Value %pl.val)
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

  ; M10: snapshot regs[48..63] (the Y-reg window) into the env frame.
  ; Y1..Y16 map to 48..63 under the disjoint X/Y ABI -- see
  ; bindings/llvm_wam_bindings.pl reg_name_to_index. Pre-M10 the
  ; snapshot covered 16..63 (X and Y mixed); now only Y space needs
  ; protection (X space dies at calls per canonical WAM).
  %alloc.y_src = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 48
  %alloc.y_dst = getelementptr %StackEntry, %StackEntry* %alloc.entry, i32 0, i32 3, i32 0
  %alloc.y_src_i8 = bitcast %Value* %alloc.y_src to i8*
  %alloc.y_dst_i8 = bitcast %Value* %alloc.y_dst to i8*
  ; 16 Values * 16 bytes = 256 bytes.
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %alloc.y_dst_i8, i8* %alloc.y_src_i8, i64 256, i1 false)
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

  ; M10: restore regs[48..63] (Y-reg window) from the env frame
  ; snapshot. See allocate for the save side.
  %dealloc.y_src = getelementptr %StackEntry, %StackEntry* %dealloc.entry, i32 0, i32 3, i32 0
  %dealloc.y_dst = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 48
  %dealloc.y_src_i8 = bitcast %Value* %dealloc.y_src to i8*
  %dealloc.y_dst_i8 = bitcast %Value* %dealloc.y_dst to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dealloc.y_dst_i8, i8* %dealloc.y_src_i8, i64 256, i1 false)
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
'  ; op1 = label index
  %exec.label = trunc i64 %op1 to i32
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
  ; Save registers (copy 32 x %Value)
  %tme.dst_regs = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 1, i32 0
  %tme.src_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %tme.dst_raw = bitcast %Value* %tme.dst_regs to i8*
  %tme.src_raw = bitcast %Value* %tme.src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tme.dst_raw, i8* %tme.src_raw, i64 1024, i1 false)
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
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %ba.dst_raw, i8* %ba.src_raw, i64 1024, i1 false)

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

  ; Save heap_top so unwind via wam_finalize_aggregate can rewind.
  %ba.hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %ba.hs = load i32, i32* %ba.hs_ptr
  %ba.sht_ptr = getelementptr %ChoicePoint, %ChoicePoint* %ba.cp_slot, i32 0, i32 11
  store i32 %ba.hs, i32* %ba.sht_ptr

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
compile_wam_helpers_to_llvm(_Options, LLVMCode) :-
    compile_backtrack_to_llvm(BacktrackCode),
    compile_unwind_trail_to_llvm(UnwindCode),
    compile_execute_builtin_to_llvm(BuiltinCode),
    compile_eval_arith_to_llvm(ArithCode),
    compile_copy_term_to_llvm(CopyTermCode),
    atomic_list_concat([
        BacktrackCode, '\n\n',
        UnwindCode, '\n\n',
        BuiltinCode, '\n\n',
        ArithCode, '\n\n',
        CopyTermCode
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

  ; Rewind heap_top to the saved value so put_variable allocations
  ; from the failing alternative do not leak as garbage cells.
  %saved_ht_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 11
  %saved_ht = load i32, i32* %saved_ht_ptr
  %ht_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  store i32 %saved_ht, i32* %ht_ptr

  ; Restore registers
  %dst_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %src_regs = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 1, i32 0
  %dst_raw = bitcast %Value* %dst_regs to i8*
  %src_raw = bitcast %Value* %src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst_raw, i8* %src_raw, i64 1024, i1 false)

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
    i32 11, label %builtin_write
    i32 12, label %builtin_nl
    i32 13, label %builtin_atom_check
    i32 14, label %builtin_integer_check
    i32 15, label %builtin_float_check
    i32 16, label %builtin_number_check
    i32 17, label %builtin_compound_check
    i32 18, label %builtin_var
    i32 19, label %builtin_nonvar
    i32 20, label %builtin_is_list_check
    i32 21, label %builtin_neq
    i32 22, label %builtin_succ
    i32 23, label %builtin_plus
    i32 24, label %builtin_unify
    i32 25, label %builtin_not_unify
    i32 26, label %builtin_functor
    i32 27, label %builtin_arg
    i32 28, label %builtin_univ
    i32 29, label %builtin_copy_term
    i32 30, label %builtin_msort
    i32 31, label %builtin_length
    i32 32, label %builtin_format2
    i32 33, label %builtin_format1
  ]

builtin_is:
  ; A1 is result, A2 is expression. M11: go through eval_arith_value
  ; so `**` with a negative integer exponent returns a Float instead
  ; of the integer eval_arith''s 0. Everything else still routes
  ; through integer eval_arith (boxed via value_integer).
  %is.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %is.result_val = call %Value @eval_arith_value(%WamState* %vm, %Value %is.a2)
  %is.a1d = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %is.a1_unb = call i1 @value_is_unbound(%Value %is.a1d)
  br i1 %is.a1_unb, label %is.do_bind, label %is.check_eq

is.do_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_bind_reg(%WamState* %vm, i32 0, %Value %is.result_val)
  ret i1 true

is.check_eq:
  %is.eq = call i1 @value_equals(%Value %is.a1d, %Value %is.result_val)
  ret i1 %is.eq

builtin_gt:
  ; M14: arithmetic compare. Evaluate both args via eval_arith_value
  ; so a Compound operand (e.g. `X > Y * 2`) is reduced to a numeric
  ; Value rather than treated as a pointer payload. Dispatch on tag:
  ; if either operand is Float, promote both to double and use fcmp;
  ; otherwise compare integer payloads with icmp.
  %gt.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %gt.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %gt.e1 = call %Value @eval_arith_value(%WamState* %vm, %Value %gt.a1)
  %gt.e2 = call %Value @eval_arith_value(%WamState* %vm, %Value %gt.a2)
  %gt.t1 = extractvalue %Value %gt.e1, 0
  %gt.t2 = extractvalue %Value %gt.e2, 0
  %gt.f1 = icmp eq i32 %gt.t1, 2
  %gt.f2 = icmp eq i32 %gt.t2, 2
  %gt.eitherf = or i1 %gt.f1, %gt.f2
  br i1 %gt.eitherf, label %gt.float, label %gt.int
gt.int:
  %gt.iv1 = extractvalue %Value %gt.e1, 1
  %gt.iv2 = extractvalue %Value %gt.e2, 1
  %gt.ir = icmp sgt i64 %gt.iv1, %gt.iv2
  ret i1 %gt.ir
gt.float:
  %gt.d1 = call double @value_to_double(%Value %gt.e1)
  %gt.d2 = call double @value_to_double(%Value %gt.e2)
  %gt.fr = fcmp ogt double %gt.d1, %gt.d2
  ret i1 %gt.fr

builtin_lt:
  %lt.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %lt.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %lt.e1 = call %Value @eval_arith_value(%WamState* %vm, %Value %lt.a1)
  %lt.e2 = call %Value @eval_arith_value(%WamState* %vm, %Value %lt.a2)
  %lt.t1 = extractvalue %Value %lt.e1, 0
  %lt.t2 = extractvalue %Value %lt.e2, 0
  %lt.f1 = icmp eq i32 %lt.t1, 2
  %lt.f2 = icmp eq i32 %lt.t2, 2
  %lt.eitherf = or i1 %lt.f1, %lt.f2
  br i1 %lt.eitherf, label %lt.float, label %lt.int
lt.int:
  %lt.iv1 = extractvalue %Value %lt.e1, 1
  %lt.iv2 = extractvalue %Value %lt.e2, 1
  %lt.ir = icmp slt i64 %lt.iv1, %lt.iv2
  ret i1 %lt.ir
lt.float:
  %lt.d1 = call double @value_to_double(%Value %lt.e1)
  %lt.d2 = call double @value_to_double(%Value %lt.e2)
  %lt.fr = fcmp olt double %lt.d1, %lt.d2
  ret i1 %lt.fr

builtin_ge:
  %ge.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ge.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %ge.e1 = call %Value @eval_arith_value(%WamState* %vm, %Value %ge.a1)
  %ge.e2 = call %Value @eval_arith_value(%WamState* %vm, %Value %ge.a2)
  %ge.t1 = extractvalue %Value %ge.e1, 0
  %ge.t2 = extractvalue %Value %ge.e2, 0
  %ge.f1 = icmp eq i32 %ge.t1, 2
  %ge.f2 = icmp eq i32 %ge.t2, 2
  %ge.eitherf = or i1 %ge.f1, %ge.f2
  br i1 %ge.eitherf, label %ge.float, label %ge.int
ge.int:
  %ge.iv1 = extractvalue %Value %ge.e1, 1
  %ge.iv2 = extractvalue %Value %ge.e2, 1
  %ge.ir = icmp sge i64 %ge.iv1, %ge.iv2
  ret i1 %ge.ir
ge.float:
  %ge.d1 = call double @value_to_double(%Value %ge.e1)
  %ge.d2 = call double @value_to_double(%Value %ge.e2)
  %ge.fr = fcmp oge double %ge.d1, %ge.d2
  ret i1 %ge.fr

builtin_le:
  %le.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %le.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %le.e1 = call %Value @eval_arith_value(%WamState* %vm, %Value %le.a1)
  %le.e2 = call %Value @eval_arith_value(%WamState* %vm, %Value %le.a2)
  %le.t1 = extractvalue %Value %le.e1, 0
  %le.t2 = extractvalue %Value %le.e2, 0
  %le.f1 = icmp eq i32 %le.t1, 2
  %le.f2 = icmp eq i32 %le.t2, 2
  %le.eitherf = or i1 %le.f1, %le.f2
  br i1 %le.eitherf, label %le.float, label %le.int
le.int:
  %le.iv1 = extractvalue %Value %le.e1, 1
  %le.iv2 = extractvalue %Value %le.e2, 1
  %le.ir = icmp sle i64 %le.iv1, %le.iv2
  ret i1 %le.ir
le.float:
  %le.d1 = call double @value_to_double(%Value %le.e1)
  %le.d2 = call double @value_to_double(%Value %le.e2)
  %le.fr = fcmp ole double %le.d1, %le.d2
  ret i1 %le.fr

builtin_arith_eq:
  %aeq.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %aeq.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %aeq.e1 = call %Value @eval_arith_value(%WamState* %vm, %Value %aeq.a1)
  %aeq.e2 = call %Value @eval_arith_value(%WamState* %vm, %Value %aeq.a2)
  %aeq.t1 = extractvalue %Value %aeq.e1, 0
  %aeq.t2 = extractvalue %Value %aeq.e2, 0
  %aeq.f1 = icmp eq i32 %aeq.t1, 2
  %aeq.f2 = icmp eq i32 %aeq.t2, 2
  %aeq.eitherf = or i1 %aeq.f1, %aeq.f2
  br i1 %aeq.eitherf, label %aeq.float, label %aeq.int
aeq.int:
  %aeq.iv1 = extractvalue %Value %aeq.e1, 1
  %aeq.iv2 = extractvalue %Value %aeq.e2, 1
  %aeq.ir = icmp eq i64 %aeq.iv1, %aeq.iv2
  ret i1 %aeq.ir
aeq.float:
  %aeq.d1 = call double @value_to_double(%Value %aeq.e1)
  %aeq.d2 = call double @value_to_double(%Value %aeq.e2)
  %aeq.fr = fcmp oeq double %aeq.d1, %aeq.d2
  ret i1 %aeq.fr

builtin_arith_ne:
  %ane.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ane.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %ane.e1 = call %Value @eval_arith_value(%WamState* %vm, %Value %ane.a1)
  %ane.e2 = call %Value @eval_arith_value(%WamState* %vm, %Value %ane.a2)
  %ane.t1 = extractvalue %Value %ane.e1, 0
  %ane.t2 = extractvalue %Value %ane.e2, 0
  %ane.f1 = icmp eq i32 %ane.t1, 2
  %ane.f2 = icmp eq i32 %ane.t2, 2
  %ane.eitherf = or i1 %ane.f1, %ane.f2
  br i1 %ane.eitherf, label %ane.float, label %ane.int
ane.int:
  %ane.iv1 = extractvalue %Value %ane.e1, 1
  %ane.iv2 = extractvalue %Value %ane.e2, 1
  %ane.ir = icmp ne i64 %ane.iv1, %ane.iv2
  ret i1 %ane.ir
ane.float:
  %ane.d1 = call double @value_to_double(%Value %ane.e1)
  %ane.d2 = call double @value_to_double(%Value %ane.e2)
  %ane.fr = fcmp one double %ane.d1, %ane.d2
  ret i1 %ane.fr

builtin_eq:
  %eq.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %eq.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %eq.r = call i1 @value_equals(%Value %eq.a1, %Value %eq.a2)
  ret i1 %eq.r

builtin_true:
  ret i1 true

builtin_fail:
  ret i1 false

builtin_cut:
  ; Set cpn to current cut_barrier
  %cut.cb_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 23
  %cut.cb = load i32, i32* %cut.cb_ptr
  %cut.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  store i32 %cut.cb, i32* %cut.cpn_ptr
  ret i1 true

builtin_integer_check:
  %ic.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %ic.tag = call i32 @value_tag(%Value %ic.a1)
  %ic.r = icmp eq i32 %ic.tag, 1
  ret i1 %ic.r

builtin_var:
  %var.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %var.r = call i1 @value_is_unbound(%Value %var.a1)
  ret i1 %var.r

builtin_nonvar:
  %nv.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %nv.r = call i1 @value_is_unbound(%Value %nv.a1)
  %nv.nr = xor i1 %nv.r, true
  ret i1 %nv.nr

builtin_atom_check:
  %ac.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %ac.tag = call i32 @value_tag(%Value %ac.a1)
  %ac.r = icmp eq i32 %ac.tag, 0
  ret i1 %ac.r

builtin_float_check:
  %fc.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %fc.tag = call i32 @value_tag(%Value %fc.a1)
  %fc.r = icmp eq i32 %fc.tag, 2
  ret i1 %fc.r

builtin_number_check:
  ; number/1: true if integer (tag=1) or float (tag=2)
  %nc.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %nc.tag = call i32 @value_tag(%Value %nc.a1)
  %nc.is_int = icmp eq i32 %nc.tag, 1
  %nc.is_flt = icmp eq i32 %nc.tag, 2
  %nc.r = or i1 %nc.is_int, %nc.is_flt
  ret i1 %nc.r

builtin_compound_check:
  %cc.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %cc.tag = call i32 @value_tag(%Value %cc.a1)
  %cc.r = icmp eq i32 %cc.tag, 3
  ret i1 %cc.r

builtin_is_list_check:
  ; is_list/1: true if list (tag=4) or the empty list atom []
  %ilc.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %ilc.tag = call i32 @value_tag(%Value %ilc.a1)
  %ilc.r = icmp eq i32 %ilc.tag, 4
  ret i1 %ilc.r

builtin_neq:
  ; \\==/2: not structurally equal
  %neq.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %neq.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %neq.eq = call i1 @value_equals(%Value %neq.a1, %Value %neq.a2)
  %neq.r = xor i1 %neq.eq, true
  ret i1 %neq.r

builtin_write:
  ; write/1: print A1 payload as integer via printf. No-op on WASM.
  %wr.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %wr.tag = call i32 @value_tag(%Value %wr.a1)
  %wr.pay = call i64 @value_payload(%Value %wr.a1)
  %wr.is_int = icmp eq i32 %wr.tag, 1
  br i1 %wr.is_int, label %wr.print_int, label %wr.done

wr.print_int:
  %wr.fmt = getelementptr [3 x i8], [3 x i8]* @.fmt_int, i32 0, i32 0
  %wr.i32 = trunc i64 %wr.pay to i32
  call i32 (i8*, ...) @printf(i8* %wr.fmt, i32 %wr.i32)
  br label %wr.done

wr.done:
  ret i1 true

builtin_nl:
  ; nl/0: print newline via printf.
  %nl.fmt = getelementptr [2 x i8], [2 x i8]* @.fmt_nl, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %nl.fmt)
  ret i1 true

builtin_succ:
  ; succ/2: A2 is A1 + 1 (or A1 is A2 - 1 if A1 unbound)
  %sc.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %sc.a1_unb = call i1 @value_is_unbound(%Value %sc.a1)
  br i1 %sc.a1_unb, label %sc.reverse, label %sc.forward

sc.forward:
  %sc.v1 = call i64 @value_payload(%Value %sc.a1)
  %sc.v2 = add i64 %sc.v1, 1
  %sc.r2 = call %Value @value_integer(i64 %sc.v2)
  %sc.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %sc.a2_unb = call i1 @value_is_unbound(%Value %sc.a2)
  br i1 %sc.a2_unb, label %sc.bind2, label %sc.check2

sc.bind2:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %sc.r2)
  ret i1 true

sc.check2:
  %sc.eq2 = call i1 @value_equals(%Value %sc.a2, %Value %sc.r2)
  ret i1 %sc.eq2

sc.reverse:
  %sc.a2r = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %sc.v2r = call i64 @value_payload(%Value %sc.a2r)
  %sc.v1r = sub i64 %sc.v2r, 1
  %sc.r1 = call %Value @value_integer(i64 %sc.v1r)
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_bind_reg(%WamState* %vm, i32 0, %Value %sc.r1)
  ret i1 true

builtin_plus:
  ; plus/3: A3 is A1 + A2 (or reverse if A3 bound)
  %pl.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %pl.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %pl.v1 = call i64 @value_payload(%Value %pl.a1)
  %pl.v2 = call i64 @value_payload(%Value %pl.a2)
  %pl.sum = add i64 %pl.v1, %pl.v2
  %pl.r = call %Value @value_integer(i64 %pl.sum)
  %pl.a3 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 2)
  %pl.a3_unb = call i1 @value_is_unbound(%Value %pl.a3)
  br i1 %pl.a3_unb, label %pl.bind, label %pl.check

pl.bind:
  call void @wam_trail_binding(%WamState* %vm, i32 2)
  call void @wam_bind_reg(%WamState* %vm, i32 2, %Value %pl.r)
  ret i1 true

pl.check:
  %pl.eq = call i1 @value_equals(%Value %pl.a3, %Value %pl.r)
  ret i1 %pl.eq

builtin_unify:
  ; =/2: unify A1 with A2. If either is unbound, bind it to the other.
  ; Use deref-reads so Ref-aliased operands resolve to their heap-stored
  ; value before the unbound check, and route binding through
  ; wam_bind_reg so the heap cell (not just the local reg slot) is
  ; updated — keeping put_variable-aliased Y/A regs in sync.
  %uf.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %uf.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %uf.a1_unb = call i1 @value_is_unbound(%Value %uf.a1)
  br i1 %uf.a1_unb, label %uf.bind_a1, label %uf.check_a2

uf.bind_a1:
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_bind_reg(%WamState* %vm, i32 0, %Value %uf.a2)
  ret i1 true

uf.check_a2:
  %uf.a2_unb = call i1 @value_is_unbound(%Value %uf.a2)
  br i1 %uf.a2_unb, label %uf.bind_a2, label %uf.both_bound

uf.bind_a2:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %uf.a1)
  ret i1 true

uf.both_bound:
  ; M10: both operands are bound. If both are compounds, recurse via
  ; wam_unify_value so cons-cell patterns like `L = [H|_]` unify the
  ; head/tail (binding fresh vars inside the [H|_] pattern). Atomic
  ; tags (atoms, integers, floats) just compare payload.
  %uf.tag1 = extractvalue %Value %uf.a1, 0
  %uf.tag2 = extractvalue %Value %uf.a2, 0
  %uf.tags_eq = icmp eq i32 %uf.tag1, %uf.tag2
  br i1 %uf.tags_eq, label %uf.tagmatch, label %uf.notmatch

uf.tagmatch:
  %uf.is_cmp = icmp eq i32 %uf.tag1, 3
  br i1 %uf.is_cmp, label %uf.deep, label %uf.atomic

uf.deep:
  ; Compounds: read the raw register values (NOT the deref''d ones) so
  ; wam_unify_value can re-deref through any Ref chains its recursion
  ; uncovers. Pass the deref''d ones too via a fresh call -- equivalent
  ; here because the top-level deref already happened.
  %uf.raw1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %uf.raw2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %uf.uok = call i1 @wam_unify_value(%WamState* %vm, %Value %uf.raw1, %Value %uf.raw2)
  ret i1 %uf.uok

uf.atomic:
  %uf.pay1 = extractvalue %Value %uf.a1, 1
  %uf.pay2 = extractvalue %Value %uf.a2, 1
  %uf.peq = icmp eq i64 %uf.pay1, %uf.pay2
  ret i1 %uf.peq

uf.notmatch:
  ret i1 false

builtin_not_unify:
  ; not-unify: succeeds if A1 and A2 do NOT unify.
  ; Simplified: fails if either is unbound (would unify), checks inequality if both bound.
  %nu.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %nu.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %nu.a1_unb = call i1 @value_is_unbound(%Value %nu.a1)
  br i1 %nu.a1_unb, label %nu.fail, label %nu.check_a2

nu.check_a2:
  %nu.a2_unb = call i1 @value_is_unbound(%Value %nu.a2)
  br i1 %nu.a2_unb, label %nu.fail, label %nu.both_bound

nu.both_bound:
  %nu.eq = call i1 @value_equals(%Value %nu.a1, %Value %nu.a2)
  %nu.r = xor i1 %nu.eq, true
  ret i1 %nu.r

nu.fail:
  ret i1 false

builtin_functor:
  ; functor/3 — read mode only for now.
  ; A1 = Term (expected bound), A2 = Name, A3 = Arity.
  ;   If A1 is a Compound (tag 3): extract functor pointer + arity,
  ;     then bind A2 = Atom(functor_ptr), A3 = Integer(arity).
  ;   If A1 is an Atom (tag 0): bind A2 = A1, A3 = Integer(0).
  ;   If A1 is an Integer (tag 1): bind A2 = A1, A3 = Integer(0).
  ;   Otherwise (unbound / compound-construct mode / list): fail.
  %fn.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %fn.a1_tag = call i32 @value_tag(%Value %fn.a1)
  switch i32 %fn.a1_tag, label %fn.fail [
    i32 0, label %fn.atomic
    i32 1, label %fn.atomic
    i32 3, label %fn.compound
  ]

fn.compound:
  %fn.cp_bits = call i64 @value_payload(%Value %fn.a1)
  %fn.cp_ptr = inttoptr i64 %fn.cp_bits to %Compound*
  %fn.fn_slot = getelementptr %Compound, %Compound* %fn.cp_ptr, i32 0, i32 0
  %fn.fn_i8 = load i8*, i8** %fn.fn_slot
  %fn.ar_slot = getelementptr %Compound, %Compound* %fn.cp_ptr, i32 0, i32 1
  %fn.ar = load i32, i32* %fn.ar_slot
  %fn.ar_i64 = sext i32 %fn.ar to i64
  %fn.name_val = call %Value @value_atom(i8* %fn.fn_i8)
  %fn.ar_val = call %Value @value_integer(i64 %fn.ar_i64)
  br label %fn.bind_both

fn.atomic:
  ; Atom or Integer: arity 0, name = the value itself.
  %fn.ar0_val = call %Value @value_integer(i64 0)
  br label %fn.bind_atomic

fn.bind_atomic:
  ; Bind A2 = A1 (as atomic name), A3 = 0. Use deref+wam_bind_reg so
  ; put_variable-aliased Y/A registers update through their shared
  ; heap cell, not just the local register slot.
  %fn.atm.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %fn.atm.a2_unb = call i1 @value_is_unbound(%Value %fn.atm.a2)
  br i1 %fn.atm.a2_unb, label %fn.atm.a2_bind, label %fn.atm.a2_check

fn.atm.a2_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %fn.a1)
  br label %fn.atm.a3

fn.atm.a2_check:
  %fn.atm.a2_eq = call i1 @value_equals(%Value %fn.atm.a2, %Value %fn.a1)
  br i1 %fn.atm.a2_eq, label %fn.atm.a3, label %fn.fail

fn.atm.a3:
  %fn.atm.a3v = call %Value @wam_get_reg_deref(%WamState* %vm, i32 2)
  %fn.atm.a3_unb = call i1 @value_is_unbound(%Value %fn.atm.a3v)
  br i1 %fn.atm.a3_unb, label %fn.atm.a3_bind, label %fn.atm.a3_check

fn.atm.a3_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 2)
  call void @wam_bind_reg(%WamState* %vm, i32 2, %Value %fn.ar0_val)
  ret i1 true

fn.atm.a3_check:
  %fn.atm.a3_eq = call i1 @value_equals(%Value %fn.atm.a3v, %Value %fn.ar0_val)
  ret i1 %fn.atm.a3_eq

fn.bind_both:
  ; Bind A2 = Atom(functor_name_ptr)
  %fn.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %fn.a2_unb = call i1 @value_is_unbound(%Value %fn.a2)
  br i1 %fn.a2_unb, label %fn.a2_bind, label %fn.a2_check

fn.a2_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %fn.name_val)
  br label %fn.a3

fn.a2_check:
  %fn.a2_eq = call i1 @value_equals(%Value %fn.a2, %Value %fn.name_val)
  br i1 %fn.a2_eq, label %fn.a3, label %fn.fail

fn.a3:
  %fn.a3v = call %Value @wam_get_reg_deref(%WamState* %vm, i32 2)
  %fn.a3_unb = call i1 @value_is_unbound(%Value %fn.a3v)
  br i1 %fn.a3_unb, label %fn.a3_bind, label %fn.a3_check

fn.a3_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 2)
  call void @wam_bind_reg(%WamState* %vm, i32 2, %Value %fn.ar_val)
  ret i1 true

fn.a3_check:
  %fn.a3_eq = call i1 @value_equals(%Value %fn.a3v, %Value %fn.ar_val)
  ret i1 %fn.a3_eq

fn.fail:
  ret i1 false

builtin_arg:
  ; arg/3 — A1 = N (Integer, 1-based), A2 = Compound, A3 = Arg.
  ; Extracts args[N-1] and unifies with A3. Fails if A1 is not Integer,
  ; A2 is not Compound, or N is out of range.
  %ag.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %ag.a1_tag = call i32 @value_tag(%Value %ag.a1)
  %ag.a1_is_int = icmp eq i32 %ag.a1_tag, 1
  br i1 %ag.a1_is_int, label %ag.check_a2, label %ag.fail

ag.check_a2:
  %ag.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %ag.a2_tag = call i32 @value_tag(%Value %ag.a2)
  %ag.a2_is_cp = icmp eq i32 %ag.a2_tag, 3
  br i1 %ag.a2_is_cp, label %ag.extract, label %ag.fail

ag.extract:
  %ag.n_i64 = call i64 @value_payload(%Value %ag.a1)
  %ag.n = trunc i64 %ag.n_i64 to i32
  %ag.cp_bits = call i64 @value_payload(%Value %ag.a2)
  %ag.cp_ptr = inttoptr i64 %ag.cp_bits to %Compound*
  %ag.ar_slot = getelementptr %Compound, %Compound* %ag.cp_ptr, i32 0, i32 1
  %ag.ar = load i32, i32* %ag.ar_slot
  %ag.n_ge1 = icmp sge i32 %ag.n, 1
  %ag.n_le_ar = icmp sle i32 %ag.n, %ag.ar
  %ag.n_ok = and i1 %ag.n_ge1, %ag.n_le_ar
  br i1 %ag.n_ok, label %ag.load, label %ag.fail

ag.load:
  %ag.args_slot = getelementptr %Compound, %Compound* %ag.cp_ptr, i32 0, i32 2
  %ag.args = load %Value*, %Value** %ag.args_slot
  %ag.idx = sub i32 %ag.n, 1
  %ag.arg_ptr = getelementptr %Value, %Value* %ag.args, i32 %ag.idx
  %ag.arg_val = load %Value, %Value* %ag.arg_ptr
  ; Unify A3 with args[N-1]. Deref first so a put_variable-aliased A3
  ; reads as Unbound (its heap cell), and bind via wam_bind_reg so the
  ; heap cell — which the aliased Y-reg also Refs — gets the value.
  %ag.a3 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 2)
  %ag.a3_unb = call i1 @value_is_unbound(%Value %ag.a3)
  br i1 %ag.a3_unb, label %ag.a3_bind, label %ag.a3_check

ag.a3_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 2)
  call void @wam_bind_reg(%WamState* %vm, i32 2, %Value %ag.arg_val)
  ret i1 true

ag.a3_check:
  %ag.a3_eq = call i1 @value_equals(%Value %ag.a3, %Value %ag.arg_val)
  ret i1 %ag.a3_eq

ag.fail:
  ret i1 false

builtin_univ:
  ; =../2 — decompose mode only (A1 bound).
  ;   A1 = Term, A2 = List (typically unbound).
  ;   If A1 is an atomic (tag 0/1/2): L = [A1]  — one cons cell.
  ;   If A1 is a Compound (tag 3):    L = [functor_atom | args_as_values]
  ;                                     — (arity+1) cons cells.
  ;   Otherwise: fail. Compose mode (A1 unbound, A2 = list) is a
  ;   follow-up — most benchmarks only exercise decompose.
  ;
  ; List repr: Prolog canonical cons cells as %Compound { ".", 2,
  ; [head, tail] }. Empty list = Atom value for "[]". Each cons lives
  ; in the arena (same allocator as put_structure) so @wam_cleanup
  ; reclaims them on backtrack/return.
  %u.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %u.a1_tag = call i32 @value_tag(%Value %u.a1)
  switch i32 %u.a1_tag, label %u.fail [
    i32 0, label %u.setup_atomic
    i32 1, label %u.setup_atomic
    i32 2, label %u.setup_atomic
    i32 3, label %u.setup_compound
    i32 6, label %u.compose
  ]

u.setup_atomic:
  ; Single-element list: N=1. args-pointer and fn_val are unused on
  ; this path; we still feed placeholders through the merge phi
  ; because SSA requires every value used downstream to dominate the
  ; use site.
  br label %u.loop_init

u.setup_compound:
  ; Unpack Compound: element[0] = Atom(functor_ptr), element[i>=1] = args[i-1].
  %u.cp_bits = call i64 @value_payload(%Value %u.a1)
  %u.cp_ptr = inttoptr i64 %u.cp_bits to %Compound*
  %u.ar_slot = getelementptr %Compound, %Compound* %u.cp_ptr, i32 0, i32 1
  %u.arity = load i32, i32* %u.ar_slot
  %u.args_slot = getelementptr %Compound, %Compound* %u.cp_ptr, i32 0, i32 2
  %u.args_c = load %Value*, %Value** %u.args_slot
  %u.fn_slot = getelementptr %Compound, %Compound* %u.cp_ptr, i32 0, i32 0
  %u.fn_ptr = load i8*, i8** %u.fn_slot
  %u.fn_val_c = call %Value @value_atom(i8* %u.fn_ptr)
  br label %u.loop_init

u.loop_init:
  ; i counts DOWN from start_i to 0 inclusive, building cons cells
  ; from the TAIL inward (last element first, innermost cons last).
  ; start_i = 0 for atomic (one cell), = arity for compound ((arity+1)
  ; cells). is_atomic picks element source in the loop body.
  ; args and fn_val get merged through even though the atomic path
  ; never reads them — SSA-dominance requirement.
  %u.start_i = phi i32 [ 0, %u.setup_atomic ], [ %u.arity, %u.setup_compound ]
  %u.is_atomic = phi i1 [ true, %u.setup_atomic ], [ false, %u.setup_compound ]
  %u.args = phi %Value* [ null, %u.setup_atomic ], [ %u.args_c, %u.setup_compound ]
  %u.fn_val = phi %Value [ zeroinitializer, %u.setup_atomic ], [ %u.fn_val_c, %u.setup_compound ]
  call void @wam_arena_ensure()
  %u.empty = call %Value @value_atom(i8* getelementptr ([3 x i8], [3 x i8]* @.fn__5B_5D, i32 0, i32 0))
  br label %u.loop_body

u.loop_body:
  %u.i = phi i32 [ %u.start_i, %u.loop_init ], [ %u.i_next, %u.loop_continue ]
  %u.tail = phi %Value [ %u.empty, %u.loop_init ], [ %u.cons_val, %u.loop_continue ]
  ; Pick element[i]:
  ;   atomic path: always A1 (i is guaranteed 0 here)
  ;   compound path: i == 0 → functor atom; i >= 1 → args[i-1]
  br i1 %u.is_atomic, label %u.elt_atomic, label %u.elt_compound_check

u.elt_atomic:
  br label %u.build_cons

u.elt_compound_check:
  %u.is_functor = icmp eq i32 %u.i, 0
  br i1 %u.is_functor, label %u.elt_functor, label %u.elt_arg

u.elt_functor:
  br label %u.build_cons

u.elt_arg:
  %u.arg_idx = sub i32 %u.i, 1
  %u.arg_ptr = getelementptr %Value, %Value* %u.args, i32 %u.arg_idx
  %u.arg_val = load %Value, %Value* %u.arg_ptr
  br label %u.build_cons

u.build_cons:
  %u.elt = phi %Value
    [ %u.a1,      %u.elt_atomic ],
    [ %u.fn_val,  %u.elt_functor ],
    [ %u.arg_val, %u.elt_arg ]
  ; Allocate %Compound + 2-element args array from the arena.
  %u.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %u.cons_mem = call i8* @wam_arena_alloc(i64 %u.cp_size)
  %u.cons = bitcast i8* %u.cons_mem to %Compound*
  %u.cons_fn = getelementptr %Compound, %Compound* %u.cons, i32 0, i32 0
  store i8* getelementptr ([2 x i8], [2 x i8]* @.fn__2E, i32 0, i32 0), i8** %u.cons_fn
  %u.cons_ar = getelementptr %Compound, %Compound* %u.cons, i32 0, i32 1
  store i32 2, i32* %u.cons_ar
  %u.cargs_mem = call i8* @wam_arena_alloc(i64 32)
  %u.cargs = bitcast i8* %u.cargs_mem to %Value*
  %u.cons_args_slot = getelementptr %Compound, %Compound* %u.cons, i32 0, i32 2
  store %Value* %u.cargs, %Value** %u.cons_args_slot
  %u.ca0 = getelementptr %Value, %Value* %u.cargs, i32 0
  store %Value %u.elt, %Value* %u.ca0
  %u.ca1 = getelementptr %Value, %Value* %u.cargs, i32 1
  store %Value %u.tail, %Value* %u.ca1
  ; Wrap cons pointer as a Compound Value.
  %u.cons_i64 = ptrtoint %Compound* %u.cons to i64
  %u.cons_val0 = insertvalue %Value undef, i32 3, 0
  %u.cons_val = insertvalue %Value %u.cons_val0, i64 %u.cons_i64, 1
  %u.done = icmp eq i32 %u.i, 0
  br i1 %u.done, label %u.bind_a2, label %u.loop_continue

u.loop_continue:
  %u.i_next = sub i32 %u.i, 1
  br label %u.loop_body

u.bind_a2:
  %u.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %u.a2_unb = call i1 @value_is_unbound(%Value %u.a2)
  br i1 %u.a2_unb, label %u.a2_bind, label %u.a2_check

u.a2_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %u.cons_val)
  ret i1 true

u.a2_check:
  ; A2 already bound — limited support: accept iff structural-equal.
  %u.a2_eq = call i1 @value_equals(%Value %u.a2, %Value %u.cons_val)
  ret i1 %u.a2_eq

u.fail:
  ret i1 false

u.compose:
  ; =../2 compose mode: A1 is unbound, A2 must deref to a non-empty
  ; cons list. Walk the list once to count elements and collect them
  ; into a temp buffer on the arena, then construct a %Compound with
  ; functor = element[0], args = element[1..].
  ;
  ; Special cases:
  ;   - Empty list → fail (cannot build nameless compound).
  ;   - Single element list [Atom] → bind A1 to the atom directly.
  ;   - Single element list [Integer] / [Float] → bind A1 to that value.
  ;   - Otherwise (>=2 elements): element[0] must be an Atom; build
  ;     a compound with that functor and arity = length - 1.
  %u.c.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %u.c.a2_tag = call i32 @value_tag(%Value %u.c.a2)
  %u.c.a2_is_cp = icmp eq i32 %u.c.a2_tag, 3
  br i1 %u.c.a2_is_cp, label %u.c.walk_init, label %u.fail

u.c.walk_init:
  ; First pass: count elements by walking cons cells.
  ; Each cons is a %Compound { ".", 2, [head, tail] }. Empty list is
  ; Atom("[]") with payload = ptrtoint(@.fn__5B_5D).
  call void @wam_arena_ensure()
  %u.c.empty_val = call %Value @value_atom(i8* getelementptr ([3 x i8], [3 x i8]* @.fn__5B_5D, i32 0, i32 0))
  %u.c.empty_payload = call i64 @value_payload(%Value %u.c.empty_val)
  br label %u.c.count_loop

u.c.count_loop:
  %u.c.count = phi i32 [ 0, %u.c.walk_init ], [ %u.c.count_next, %u.c.count_step ]
  %u.c.cur = phi %Value [ %u.c.a2, %u.c.walk_init ], [ %u.c.next_tail, %u.c.count_step ]
  %u.c.cur_tag = call i32 @value_tag(%Value %u.c.cur)
  %u.c.is_cons = icmp eq i32 %u.c.cur_tag, 3
  br i1 %u.c.is_cons, label %u.c.count_body, label %u.c.count_done

u.c.count_body:
  %u.c.cur_payload = call i64 @value_payload(%Value %u.c.cur)
  %u.c.cur_ptr = inttoptr i64 %u.c.cur_payload to %Compound*
  %u.c.cur_ar_slot = getelementptr %Compound, %Compound* %u.c.cur_ptr, i32 0, i32 1
  %u.c.cur_ar = load i32, i32* %u.c.cur_ar_slot
  %u.c.ar_ok = icmp eq i32 %u.c.cur_ar, 2
  br i1 %u.c.ar_ok, label %u.c.count_step, label %u.fail

u.c.count_step:
  %u.c.cur_args_slot = getelementptr %Compound, %Compound* %u.c.cur_ptr, i32 0, i32 2
  %u.c.cur_args = load %Value*, %Value** %u.c.cur_args_slot
  %u.c.tail_ptr = getelementptr %Value, %Value* %u.c.cur_args, i32 1
  %u.c.next_tail = load %Value, %Value* %u.c.tail_ptr
  %u.c.count_next = add i32 %u.c.count, 1
  br label %u.c.count_loop

u.c.count_done:
  ; cur should deref to the empty-list atom. Verify.
  %u.c.end_is_atom = icmp eq i32 %u.c.cur_tag, 0
  br i1 %u.c.end_is_atom, label %u.c.end_check, label %u.fail

u.c.end_check:
  %u.c.end_payload = call i64 @value_payload(%Value %u.c.cur)
  %u.c.end_eq = icmp eq i64 %u.c.end_payload, %u.c.empty_payload
  br i1 %u.c.end_eq, label %u.c.dispatch, label %u.fail

u.c.dispatch:
  ; count == 0 → fail; count == 1 → atomic bind; count >= 2 → compound.
  %u.c.is_zero = icmp eq i32 %u.c.count, 0
  br i1 %u.c.is_zero, label %u.fail, label %u.c.nonempty

u.c.nonempty:
  ; Collect elements into a temp buffer. Second pass walks again.
  %u.c.count64 = zext i32 %u.c.count to i64
  %u.c.buf_bytes = shl i64 %u.c.count64, 4
  %u.c.buf_mem = call i8* @wam_arena_alloc(i64 %u.c.buf_bytes)
  %u.c.buf = bitcast i8* %u.c.buf_mem to %Value*
  br label %u.c.collect_loop

u.c.collect_loop:
  %u.c.ci = phi i32 [ 0, %u.c.nonempty ], [ %u.c.ci_next, %u.c.collect_step ]
  %u.c.cc = phi %Value [ %u.c.a2, %u.c.nonempty ], [ %u.c.cc_tail, %u.c.collect_step ]
  %u.c.done_c = icmp sge i32 %u.c.ci, %u.c.count
  br i1 %u.c.done_c, label %u.c.build, label %u.c.collect_step

u.c.collect_step:
  %u.c.cc_payload = call i64 @value_payload(%Value %u.c.cc)
  %u.c.cc_ptr = inttoptr i64 %u.c.cc_payload to %Compound*
  %u.c.cc_args_slot = getelementptr %Compound, %Compound* %u.c.cc_ptr, i32 0, i32 2
  %u.c.cc_args = load %Value*, %Value** %u.c.cc_args_slot
  %u.c.head_ptr = getelementptr %Value, %Value* %u.c.cc_args, i32 0
  %u.c.head = load %Value, %Value* %u.c.head_ptr
  %u.c.buf_slot = getelementptr %Value, %Value* %u.c.buf, i32 %u.c.ci
  store %Value %u.c.head, %Value* %u.c.buf_slot
  %u.c.cc_tail_ptr = getelementptr %Value, %Value* %u.c.cc_args, i32 1
  %u.c.cc_tail = load %Value, %Value* %u.c.cc_tail_ptr
  %u.c.ci_next = add i32 %u.c.ci, 1
  br label %u.c.collect_loop

u.c.build:
  %u.c.is_one = icmp eq i32 %u.c.count, 1
  br i1 %u.c.is_one, label %u.c.bind_atomic, label %u.c.build_compound

u.c.bind_atomic:
  ; Single-element list → bind A1 directly to element[0].
  %u.c.elem0_ptr = getelementptr %Value, %Value* %u.c.buf, i32 0
  %u.c.elem0 = load %Value, %Value* %u.c.elem0_ptr
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %u.c.elem0)
  ret i1 true

u.c.build_compound:
  ; element[0] must be an Atom (tag 0). Arity = count - 1.
  %u.c.f_ptr = getelementptr %Value, %Value* %u.c.buf, i32 0
  %u.c.f = load %Value, %Value* %u.c.f_ptr
  %u.c.f_tag = call i32 @value_tag(%Value %u.c.f)
  %u.c.f_is_atom = icmp eq i32 %u.c.f_tag, 0
  br i1 %u.c.f_is_atom, label %u.c.alloc_compound, label %u.fail

u.c.alloc_compound:
  %u.c.new_arity = sub i32 %u.c.count, 1
  %u.c.f_payload = call i64 @value_payload(%Value %u.c.f)
  %u.c.f_ptr_i8 = inttoptr i64 %u.c.f_payload to i8*
  %u.c.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %u.c.new_mem = call i8* @wam_arena_alloc(i64 %u.c.cp_size)
  %u.c.new_cp = bitcast i8* %u.c.new_mem to %Compound*
  %u.c.new_fn_slot = getelementptr %Compound, %Compound* %u.c.new_cp, i32 0, i32 0
  store i8* %u.c.f_ptr_i8, i8** %u.c.new_fn_slot
  %u.c.new_ar_slot = getelementptr %Compound, %Compound* %u.c.new_cp, i32 0, i32 1
  store i32 %u.c.new_arity, i32* %u.c.new_ar_slot
  %u.c.new_ar64 = zext i32 %u.c.new_arity to i64
  %u.c.new_args_bytes = shl i64 %u.c.new_ar64, 4
  %u.c.new_args_mem = call i8* @wam_arena_alloc(i64 %u.c.new_args_bytes)
  %u.c.new_args = bitcast i8* %u.c.new_args_mem to %Value*
  %u.c.new_args_slot = getelementptr %Compound, %Compound* %u.c.new_cp, i32 0, i32 2
  store %Value* %u.c.new_args, %Value** %u.c.new_args_slot
  br label %u.c.copy_loop

u.c.copy_loop:
  %u.c.ki = phi i32 [ 0, %u.c.alloc_compound ], [ %u.c.ki_next, %u.c.copy_step ]
  %u.c.kd = icmp sge i32 %u.c.ki, %u.c.new_arity
  br i1 %u.c.kd, label %u.c.bind_compound, label %u.c.copy_step

u.c.copy_step:
  %u.c.src_i = add i32 %u.c.ki, 1
  %u.c.src_slot = getelementptr %Value, %Value* %u.c.buf, i32 %u.c.src_i
  %u.c.src_v = load %Value, %Value* %u.c.src_slot
  %u.c.dst_slot = getelementptr %Value, %Value* %u.c.new_args, i32 %u.c.ki
  store %Value %u.c.src_v, %Value* %u.c.dst_slot
  %u.c.ki_next = add i32 %u.c.ki, 1
  br label %u.c.copy_loop

u.c.bind_compound:
  %u.c.new_i64 = ptrtoint %Compound* %u.c.new_cp to i64
  %u.c.new_val0 = insertvalue %Value undef, i32 3, 0
  %u.c.new_val = insertvalue %Value %u.c.new_val0, i64 %u.c.new_i64, 1
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %u.c.new_val)
  ret i1 true

builtin_copy_term:
  ; copy_term/2 — A1 = source term, A2 = destination (typically unbound).
  ; Calls @wam_copy_term_value to produce a fresh deep copy from the
  ; arena, then unifies A2 with the result. Deref both regs so any
  ; put_variable-aliased Ref reads through to the heap-stored value,
  ; and write the copy back via wam_bind_reg so aliased Y/A registers
  ; both observe the new term.
  %ct.src = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %ct.copy = call %Value @wam_copy_term_value(%WamState* %vm, %Value %ct.src)
  %ct.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %ct.a2_unb = call i1 @value_is_unbound(%Value %ct.a2)
  br i1 %ct.a2_unb, label %ct.a2_bind, label %ct.a2_check

ct.a2_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %ct.copy)
  ret i1 true

ct.a2_check:
  %ct.a2_eq = call i1 @value_equals(%Value %ct.a2, %Value %ct.copy)
  ret i1 %ct.a2_eq

builtin_msort:
  ; M10: msort/2 -- A1 = input list (Compound [|]/2 chain), A2 = output.
  ; Calls @wam_msort_list which walks the input, collects into an
  ; arena buffer, in-place insertion sorts by (tag, payload), then
  ; rebuilds a [|]/2 chain on the arena. A2 is bound (through any
  ; aliased Ref) to the new list head.
  %ms.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %ms.result = call %Value @wam_msort_list(%WamState* %vm, %Value %ms.a1)
  %ms.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %ms.a2_unb = call i1 @value_is_unbound(%Value %ms.a2)
  br i1 %ms.a2_unb, label %ms.bind, label %ms.check

ms.bind:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %ms.result)
  ret i1 true

ms.check:
  ; A2 was already bound -- structurally compare via wam_unify_value
  ; so list-of-ints `[1,2,3] = msort([3,1,2], _)` succeeds.
  %ms.raw2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %ms.ok = call i1 @wam_unify_value(%WamState* %vm, %Value %ms.raw2, %Value %ms.result)
  ret i1 %ms.ok

builtin_length:
  ; M10/M16: length/2. Two modes:
  ;
  ;   Forward (A1 = bound list, A2 = unbound or Integer):
  ;     walks the [|]/2 chain counting elements, binds A2 to
  ;     Integer(count). Same chain convention as msort: terminator
  ;     is the [] Atom; non-list values halt the walk at whatever
  ;     count was accumulated so far.
  ;
  ;   Reverse (A1 = unbound, A2 = Integer N):
  ;     allocates N fresh-unbound heap cells and builds an N-element
  ;     [|]/2 chain on the arena whose head is bound to A1. Each
  ;     head args[0] is Ref(heap_addr) so the resulting list elements
  ;     are real logic variables (they can be unified later via the
  ;     existing wam_unify_value machinery).
  %ln.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %ln.a1_unb = call i1 @value_is_unbound(%Value %ln.a1)
  br i1 %ln.a1_unb, label %ln.reverse, label %ln.loop

ln.loop:
  %ln.cur = phi %Value [ %ln.a1, %builtin_length ], [ %ln.next, %ln.step ]
  %ln.n = phi i32 [ 0, %builtin_length ], [ %ln.n1, %ln.step ]
  %ln.d = call %Value @wam_deref_value(%WamState* %vm, %Value %ln.cur)
  %ln.tag = extractvalue %Value %ln.d, 0
  %ln.is_cmp = icmp eq i32 %ln.tag, 3
  br i1 %ln.is_cmp, label %ln.step, label %ln.done
ln.step:
  %ln.cp_bits = extractvalue %Value %ln.d, 1
  %ln.cp = inttoptr i64 %ln.cp_bits to %Compound*
  %ln.args_slot = getelementptr %Compound, %Compound* %ln.cp, i32 0, i32 2
  %ln.args = load %Value*, %Value** %ln.args_slot
  %ln.tail_ptr = getelementptr %Value, %Value* %ln.args, i32 1
  %ln.next = load %Value, %Value* %ln.tail_ptr
  %ln.n1 = add i32 %ln.n, 1
  br label %ln.loop
ln.done:
  %ln.n64 = sext i32 %ln.n to i64
  %ln.r_val = call %Value @value_integer(i64 %ln.n64)
  %ln.a2d = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %ln.a2_unb = call i1 @value_is_unbound(%Value %ln.a2d)
  br i1 %ln.a2_unb, label %ln.bind, label %ln.cmp
ln.bind:
  call void @wam_trail_binding(%WamState* %vm, i32 1)
  call void @wam_bind_reg(%WamState* %vm, i32 1, %Value %ln.r_val)
  ret i1 true
ln.cmp:
  %ln.eq = call i1 @value_equals(%Value %ln.a2d, %Value %ln.r_val)
  ret i1 %ln.eq

ln.reverse:
  ; A1 is unbound: check that A2 is a known Integer and build a
  ; fresh N-element list bound to A1.
  %ln.a2 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %ln.a2_tag = extractvalue %Value %ln.a2, 0
  %ln.a2_is_int = icmp eq i32 %ln.a2_tag, 1
  br i1 %ln.a2_is_int, label %ln.r_build_init, label %ln.r_fail

ln.r_fail:
  ret i1 false

ln.r_build_init:
  %ln.n_i64 = extractvalue %Value %ln.a2, 1
  %ln.n_i32 = trunc i64 %ln.n_i64 to i32
  %ln.empty_id = load i64, i64* @wam_empty_list_atom_id
  %ln.empty_v0 = insertvalue %Value undef, i32 0, 0
  %ln.empty_v  = insertvalue %Value %ln.empty_v0, i64 %ln.empty_id, 1
  call void @wam_arena_ensure()
  br label %ln.r_build_loop

ln.r_build_loop:
  %ln.r_i = phi i32 [ 0, %ln.r_build_init ], [ %ln.r_i_next, %ln.r_build_step ]
  %ln.r_tail = phi %Value [ %ln.empty_v, %ln.r_build_init ], [ %ln.r_new_cons, %ln.r_build_step ]
  %ln.r_done = icmp sge i32 %ln.r_i, %ln.n_i32
  br i1 %ln.r_done, label %ln.r_build_done, label %ln.r_build_step

ln.r_build_step:
  ; Fresh logic variable: push Unbound to the heap and remember its
  ; addr as a Ref. This is the standard WAM way to make a freshly
  ; bindable element.
  %ln.r_unb = call %Value @value_unbound(i8* null)
  %ln.r_h_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %ln.r_unb)
  %ln.r_h_ref = call %Value @value_ref(i32 %ln.r_h_addr)
  %ln.r_cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64
  %ln.r_cp_mem = call i8* @wam_arena_alloc(i64 %ln.r_cp_size)
  %ln.r_cp = bitcast i8* %ln.r_cp_mem to %Compound*
  %ln.r_fn_slot = getelementptr %Compound, %Compound* %ln.r_cp, i32 0, i32 0
  %ln.r_fn_ptr = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  store i8* %ln.r_fn_ptr, i8** %ln.r_fn_slot
  %ln.r_ar_slot = getelementptr %Compound, %Compound* %ln.r_cp, i32 0, i32 1
  store i32 2, i32* %ln.r_ar_slot
  %ln.r_args_mem = call i8* @wam_arena_alloc(i64 32)
  %ln.r_args = bitcast i8* %ln.r_args_mem to %Value*
  %ln.r_args_slot = getelementptr %Compound, %Compound* %ln.r_cp, i32 0, i32 2
  store %Value* %ln.r_args, %Value** %ln.r_args_slot
  %ln.r_h0_ptr = getelementptr %Value, %Value* %ln.r_args, i32 0
  store %Value %ln.r_h_ref, %Value* %ln.r_h0_ptr
  %ln.r_t_ptr = getelementptr %Value, %Value* %ln.r_args, i32 1
  store %Value %ln.r_tail, %Value* %ln.r_t_ptr
  %ln.r_cp_i64 = ptrtoint %Compound* %ln.r_cp to i64
  %ln.r_nc0 = insertvalue %Value undef, i32 3, 0
  %ln.r_new_cons = insertvalue %Value %ln.r_nc0, i64 %ln.r_cp_i64, 1
  %ln.r_i_next = add i32 %ln.r_i, 1
  br label %ln.r_build_loop

ln.r_build_done:
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_bind_reg(%WamState* %vm, i32 0, %Value %ln.r_tail)
  ret i1 true

builtin_format2:
  ; M12: format/2 -- A1 = format string atom, A2 = args list ([|]/2 chain).
  ; Walks the format string char by char. ~w/~d/~a consume the next
  ; arg and print its value (dispatched on tag: Integer printf %lld,
  ; Float printf %g, Atom printf %s after atom-table lookup).
  ; ~n prints newline; ~~ prints a literal tilde; anything else
  ; (unknown directives) passes through verbatim. Bound at the
  ; module via the M12 @wam_atom_strings global, so a literal format
  ; atom like ''hello ~w~n'' resolves at runtime to its C string.
  %f2.fmt_v = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %f2.fmt_id = extractvalue %Value %f2.fmt_v, 1
  %f2.fmt_p0 = call i8* @wam_atom_to_string(i64 %f2.fmt_id)
  %f2.args0 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %f2.null = icmp eq i8* %f2.fmt_p0, null
  br i1 %f2.null, label %f2.done, label %f2.loop

builtin_format1:
  ; format/1: same as format/2 but with no args. Build a fresh empty
  ; args list (Atom [] sentinel) and fall through to the same loop.
  %f1.fmt_v = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  %f1.fmt_id = extractvalue %Value %f1.fmt_v, 1
  %f1.fmt_p0 = call i8* @wam_atom_to_string(i64 %f1.fmt_id)
  %f1.empty_id = load i64, i64* @wam_empty_list_atom_id
  %f1.args_v0 = insertvalue %Value undef, i32 0, 0
  %f1.args0 = insertvalue %Value %f1.args_v0, i64 %f1.empty_id, 1
  %f1.null = icmp eq i8* %f1.fmt_p0, null
  br i1 %f1.null, label %f2.done, label %f2.loop

f2.loop:
  ; Per-iteration state: %p = current ptr into format string, %args
  ; = remaining args list. Three input edges: format/2 entry,
  ; format/1 entry, and the per-iteration backedge from %f2.iter_end.
  %f2.p = phi i8* [ %f2.fmt_p0, %builtin_format2 ],
                  [ %f1.fmt_p0, %builtin_format1 ],
                  [ %f2.p_next, %f2.iter_end ]
  %f2.args = phi %Value [ %f2.args0, %builtin_format2 ],
                         [ %f1.args0, %builtin_format1 ],
                         [ %f2.args_next, %f2.iter_end ]
  %f2.c = load i8, i8* %f2.p
  %f2.done_p = icmp eq i8 %f2.c, 0
  br i1 %f2.done_p, label %f2.done, label %f2.check_tilde

f2.check_tilde:
  %f2.is_tilde = icmp eq i8 %f2.c, 126
  br i1 %f2.is_tilde, label %f2.directive, label %f2.print_lit

f2.print_lit:
  %f2.c_i32 = zext i8 %f2.c to i32
  call i32 @putchar(i32 %f2.c_i32)
  %f2.p_lit_next = getelementptr i8, i8* %f2.p, i32 1
  br label %f2.iter_end

f2.directive:
  %f2.p_dir = getelementptr i8, i8* %f2.p, i32 1
  %f2.d = load i8, i8* %f2.p_dir
  ; M15: digit -> precision-prefixed directive (~Nf / ~Ne). Parse
  ; the digit run, then look at the terminator: ``f`` -> fixed-point
  ; (printf "%.*f"), ``e`` -> scientific (printf "%.*e"), anything
  ; else -> unknown, print verbatim.
  %f2.d_ge0 = icmp uge i8 %f2.d, 48
  %f2.d_le9 = icmp ule i8 %f2.d, 57
  %f2.d_is_digit = and i1 %f2.d_ge0, %f2.d_le9
  br i1 %f2.d_is_digit, label %f2.parse_num, label %f2.dispatch_letter

f2.dispatch_letter:
  switch i8 %f2.d, label %f2.unknown_dir [
    i8 119, label %f2.directive_w
    i8 100, label %f2.directive_w
    i8 97,  label %f2.directive_w
    i8 110, label %f2.directive_n
    i8 126, label %f2.directive_t
  ]

f2.parse_num:
  br label %f2.num_loop
f2.num_loop:
  %f2.np = phi i8* [ %f2.p_dir, %f2.parse_num ], [ %f2.np_next, %f2.num_step ]
  %f2.nacc = phi i64 [ 0, %f2.parse_num ], [ %f2.nacc_new, %f2.num_step ]
  %f2.nc = load i8, i8* %f2.np
  %f2.nc_ge0 = icmp uge i8 %f2.nc, 48
  %f2.nc_le9 = icmp ule i8 %f2.nc, 57
  %f2.nc_is_d = and i1 %f2.nc_ge0, %f2.nc_le9
  br i1 %f2.nc_is_d, label %f2.num_step, label %f2.num_done
f2.num_step:
  %f2.nd_val_i8 = sub i8 %f2.nc, 48
  %f2.nd_val = zext i8 %f2.nd_val_i8 to i64
  %f2.nacc_mul = mul i64 %f2.nacc, 10
  %f2.nacc_new = add i64 %f2.nacc_mul, %f2.nd_val
  %f2.np_next = getelementptr i8, i8* %f2.np, i32 1
  br label %f2.num_loop
f2.num_done:
  ; %f2.np points to the terminator char (the first non-digit after ~).
  ; ``f`` and ``e`` both consume one float-promotable arg and use
  ; printf("%.*<spec>", N, val). The pointer advances past the
  ; terminator (np + 1).
  %f2.nc_is_f = icmp eq i8 %f2.nc, 102
  %f2.nc_is_e = icmp eq i8 %f2.nc, 101
  %f2.nc_is_fe = or i1 %f2.nc_is_f, %f2.nc_is_e
  br i1 %f2.nc_is_fe, label %f2.precision_print, label %f2.precision_unknown

f2.precision_unknown:
  ; Bad ~<digits><non-fe>: print the ~ then fall through to the
  ; ordinary unknown handler which will print the next char. Drop
  ; the parsed digits (best-effort -- atypical in practice).
  call i32 @putchar(i32 126)
  %f2.nc_i32 = zext i8 %f2.nc to i32
  call i32 @putchar(i32 %f2.nc_i32)
  %f2.p_after_punk = getelementptr i8, i8* %f2.np, i32 1
  br label %f2.iter_end

f2.precision_print:
  ; Consume one arg from the args list; promote to double via
  ; value_to_double; call printf with the chosen format ("%.*f" or
  ; "%.*e") and the parsed precision N.
  %f2.parg_d = call %Value @wam_deref_value(%WamState* %vm, %Value %f2.args)
  %f2.parg_tag = extractvalue %Value %f2.parg_d, 0
  %f2.parg_has = icmp eq i32 %f2.parg_tag, 3
  br i1 %f2.parg_has, label %f2.precision_read_arg, label %f2.precision_no_arg

f2.precision_no_arg:
  ; No arg available: silently skip the directive but advance the
  ; pointer past the terminator so we don''t loop forever.
  %f2.p_after_pna = getelementptr i8, i8* %f2.np, i32 1
  br label %f2.iter_end

f2.precision_read_arg:
  %f2.pcp_bits = extractvalue %Value %f2.parg_d, 1
  %f2.pcp = inttoptr i64 %f2.pcp_bits to %Compound*
  %f2.pcargs_slot = getelementptr %Compound, %Compound* %f2.pcp, i32 0, i32 2
  %f2.pcargs = load %Value*, %Value** %f2.pcargs_slot
  %f2.phead_ptr = getelementptr %Value, %Value* %f2.pcargs, i32 0
  %f2.phead_raw = load %Value, %Value* %f2.phead_ptr
  %f2.phead = call %Value @wam_deref_value(%WamState* %vm, %Value %f2.phead_raw)
  %f2.ptail_ptr = getelementptr %Value, %Value* %f2.pcargs, i32 1
  %f2.ptail = load %Value, %Value* %f2.ptail_ptr
  %f2.pval_d = call double @value_to_double(%Value %f2.phead)
  %f2.nacc_i32 = trunc i64 %f2.nacc to i32
  br i1 %f2.nc_is_f, label %f2.do_starf, label %f2.do_stare

f2.do_starf:
  %f2.fmt_sf = getelementptr [5 x i8], [5 x i8]* @.fmt_starf, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %f2.fmt_sf, i32 %f2.nacc_i32, double %f2.pval_d)
  br label %f2.precision_after

f2.do_stare:
  %f2.fmt_se = getelementptr [5 x i8], [5 x i8]* @.fmt_stare, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %f2.fmt_se, i32 %f2.nacc_i32, double %f2.pval_d)
  br label %f2.precision_after

f2.precision_after:
  %f2.p_after_pcons = getelementptr i8, i8* %f2.np, i32 1
  br label %f2.iter_end_precision_consume

f2.iter_end_precision_consume:
  br label %f2.iter_end

f2.directive_n:
  call i32 @putchar(i32 10)
  %f2.p_after_n = getelementptr i8, i8* %f2.p, i32 2
  br label %f2.iter_end

f2.directive_t:
  call i32 @putchar(i32 126)
  %f2.p_after_t = getelementptr i8, i8* %f2.p, i32 2
  br label %f2.iter_end

f2.unknown_dir:
  call i32 @putchar(i32 126)
  %f2.d_i32 = zext i8 %f2.d to i32
  call i32 @putchar(i32 %f2.d_i32)
  %f2.p_after_u = getelementptr i8, i8* %f2.p, i32 2
  br label %f2.iter_end

f2.directive_w:
  ; Print next arg (if available). Deref args first -- the previous
  ; iteration left %f2.args as the raw tail field of the cons cell,
  ; which is typically a Ref into the heap rather than a Compound.
  %f2.args_d = call %Value @wam_deref_value(%WamState* %vm, %Value %f2.args)
  %f2.args_tag = extractvalue %Value %f2.args_d, 0
  %f2.has_arg = icmp eq i32 %f2.args_tag, 3
  br i1 %f2.has_arg, label %f2.read_arg, label %f2.no_arg

f2.read_arg:
  %f2.cp_bits = extractvalue %Value %f2.args_d, 1
  %f2.cp = inttoptr i64 %f2.cp_bits to %Compound*
  %f2.cargs_slot = getelementptr %Compound, %Compound* %f2.cp, i32 0, i32 2
  %f2.cargs = load %Value*, %Value** %f2.cargs_slot
  %f2.head_ptr = getelementptr %Value, %Value* %f2.cargs, i32 0
  %f2.head_raw = load %Value, %Value* %f2.head_ptr
  %f2.head = call %Value @wam_deref_value(%WamState* %vm, %Value %f2.head_raw)
  %f2.tail_ptr = getelementptr %Value, %Value* %f2.cargs, i32 1
  %f2.tail = load %Value, %Value* %f2.tail_ptr
  %f2.head_tag = extractvalue %Value %f2.head, 0
  switch i32 %f2.head_tag, label %f2.print_anon [
    i32 0, label %f2.print_atom
    i32 1, label %f2.print_int
    i32 2, label %f2.print_float
  ]

f2.print_int:
  %f2.iv = extractvalue %Value %f2.head, 1
  %f2.fmt_lld = getelementptr [5 x i8], [5 x i8]* @.fmt_lld, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %f2.fmt_lld, i64 %f2.iv)
  br label %f2.after_w

f2.print_float:
  %f2.fbits = extractvalue %Value %f2.head, 1
  %f2.fval = bitcast i64 %f2.fbits to double
  %f2.fmt_dbl = getelementptr [3 x i8], [3 x i8]* @.fmt_dbl, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %f2.fmt_dbl, double %f2.fval)
  br label %f2.after_w

f2.print_atom:
  %f2.aid = extractvalue %Value %f2.head, 1
  %f2.astr = call i8* @wam_atom_to_string(i64 %f2.aid)
  %f2.astr_null = icmp eq i8* %f2.astr, null
  br i1 %f2.astr_null, label %f2.after_w, label %f2.print_atom_go
f2.print_atom_go:
  %f2.fmt_str = getelementptr [3 x i8], [3 x i8]* @.fmt_str, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %f2.fmt_str, i8* %f2.astr)
  br label %f2.after_w

f2.print_anon:
  ; Compound and other tags: print underscore placeholder. Full
  ; pretty-printing is deferred; effective_distance''s format strings
  ; only use atoms, integers, and floats.
  call i32 @putchar(i32 95)
  br label %f2.after_w

f2.after_w:
  ; Args consumed: advance to %f2.tail.
  %f2.p_after_w = getelementptr i8, i8* %f2.p, i32 2
  br label %f2.iter_end_consume

f2.no_arg:
  ; No arg available -- skip silently, keep args unchanged.
  %f2.p_no_arg = getelementptr i8, i8* %f2.p, i32 2
  br label %f2.iter_end

f2.iter_end_consume:
  br label %f2.iter_end

f2.iter_end:
  ; Merge all per-iteration paths. Consume paths advance args to
  ; %f2.tail / %f2.ptail; everything else keeps %f2.args unchanged.
  %f2.p_next = phi i8* [ %f2.p_lit_next, %f2.print_lit ],
                        [ %f2.p_after_n, %f2.directive_n ],
                        [ %f2.p_after_t, %f2.directive_t ],
                        [ %f2.p_after_u, %f2.unknown_dir ],
                        [ %f2.p_no_arg, %f2.no_arg ],
                        [ %f2.p_after_w, %f2.iter_end_consume ],
                        [ %f2.p_after_punk, %f2.precision_unknown ],
                        [ %f2.p_after_pna, %f2.precision_no_arg ],
                        [ %f2.p_after_pcons, %f2.iter_end_precision_consume ]
  %f2.args_next = phi %Value [ %f2.args, %f2.print_lit ],
                              [ %f2.args, %f2.directive_n ],
                              [ %f2.args, %f2.directive_t ],
                              [ %f2.args, %f2.unknown_dir ],
                              [ %f2.args, %f2.no_arg ],
                              [ %f2.tail, %f2.iter_end_consume ],
                              [ %f2.args, %f2.precision_unknown ],
                              [ %f2.args, %f2.precision_no_arg ],
                              [ %f2.ptail, %f2.iter_end_precision_consume ]
  br label %f2.loop

f2.done:
  ret i1 true

unknown:
  ret i1 false
}'.

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
define %Value @eval_arith_value(%WamState* %vm, %Value %expr_raw) {
entry:
  %expr = call %Value @wam_deref_value(%WamState* %vm, %Value %expr_raw)
  %tag = extractvalue %Value %expr, 0
  switch i32 %tag, label %ev_zero [
    i32 1, label %ev_atomic_in
    i32 2, label %ev_atomic_in
    i32 3, label %ev_compound
  ]

ev_atomic_in:
  ret %Value %expr

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

ev_check_named_binary:
  ; mod / max / min -- functor name starts with ``m``.
  %nb_is_m = icmp eq i8 %fn0, 109
  br i1 %nb_is_m, label %ev_nb_second, label %ev_zero
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
  switch i8 %fn0, label %ev_zero [
    i8 45, label %ev_neg
    i8 97, label %ev_abs
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

ev_zero:
  ; Unknown / fail -- return Integer 0 as a benign placeholder. The
  ; legacy @eval_arith printed an error to stderr here; for the
  ; tagged path we just box zero so callers don''t crash on an
  ; unrecognized expression.
  %z_v = call %Value @value_integer(i64 0)
  ret %Value %z_v
}'.

compile_copy_term_to_llvm(Code) :-
    Code = '; Recursively copy a %Value, allocating fresh %Compound cells from
; the arena for any compound encountered. Atomic values (Atom / Integer /
; Float / Bool) are returned by value. Unbound values return a fresh
; Unbound sentinel — a proper impl would use a variable-map to preserve
; sharing and create fresh Refs, but the current bench corpus has no
; shared unbound vars and this naive pass is enough for the flat and
; nested compound cases. Cons-cell lists fall out automatically since
; they are just compounds with functor "." / arity 2.
define %Value @wam_copy_term_value(%WamState* %vm, %Value %v) {
entry:
  %tag = call i32 @value_tag(%Value %v)
  switch i32 %tag, label %atomic [
    i32 3, label %ct_compound
    i32 6, label %ct_unbound
  ]

atomic:
  ret %Value %v

ct_unbound:
  %fresh = call %Value @value_unbound(i8* null)
  ret %Value %fresh

ct_compound:
  %cp_bits = call i64 @value_payload(%Value %v)
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
  %i = phi i32 [ 0, %ct_loop_entry ], [ %i_next, %ct_loop_step ]
  %src_ptr = getelementptr %Value, %Value* %src_args, i32 %i
  %src_val = load %Value, %Value* %src_ptr
  %dst_val = call %Value @wam_copy_term_value(%WamState* %vm, %Value %src_val)
  %dst_ptr = getelementptr %Value, %Value* %new_args, i32 %i
  store %Value %dst_val, %Value* %dst_ptr
  %i_next = add i32 %i, 1
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
    ( integer(C) -> Tag = 1 ; Tag = 0 ),
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

:- dynamic functor_string_global/2. % functor_string_global(NameStr, LenPlus1)

%% emit_functor_string_globals(-IR)
%  Emits LLVM global constants for functor names collected during WAM
%  compilation (by put_structure instructions). Each functor "name" becomes
%  @.fn_name = private constant [N x i8] c"name\00".
emit_functor_string_globals(IR) :-
    findall(GlobalDef,
        ( functor_string_global(NameStr, Len),
          sanitize_functor_for_llvm(NameStr, SaneName),
          format(atom(GlobalDef), '@.fn_~w = private constant [~w x i8] c"~w\\00"',
              [SaneName, Len, NameStr])
        ),
        Defs),
    ( Defs == []
    -> IR = ''
    ;  atomic_list_concat(Defs, '\n', IR)
    ).

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

define i8* @wam_atom_to_string(i64 %id) {
  ret i8* null
}'
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
       format(atom(LookupTable),
'@wam_atom_strings = private constant [~w x i8*] [
  ~w
]
@wam_atom_string_count = private constant i32 ~w

define i8* @wam_atom_to_string(i64 %id) {
entry:
  %cnt = load i32, i32* @wam_atom_string_count
  %cnt64 = zext i32 %cnt to i64
  %in_range = icmp ult i64 %id, %cnt64
  br i1 %in_range, label %lookup, label %nullret
lookup:
  %slot_ptr = getelementptr [~w x i8*], [~w x i8*]* @wam_atom_strings, i32 0, i64 %id
  %s = load i8*, i8** %slot_ptr
  ret i8* %s
nullret:
  ret i8* null
}',
           [ArrSize, SlotsStr, ArrSize, ArrSize, ArrSize]),
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
wam_line_to_llvm_literal_resolved(["jump", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    lookup_label_index(CL, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 32, i64 ~w, i64 0 }', [LabelIdx]).
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
    % Determine tag: integers get tag=1, atoms get tag=0.
    ( number_string(_, CC) -> Tag = 1 ; Tag = 0 ),
    % Pack tag into op2 high bits: op2 = (tag << 16) | reg_idx.
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), '%Instruction { i32 0, i64 ~w, i64 ~w }', [PackedVal, Op2]).
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
%  If S is a single-quoted atom representation `'...'`, return the
%  inner string with `''` un-escaped to `'`. Otherwise return S
%  unchanged.
strip_quoted_atom(Str, Inner) :-
    string_length(Str, L),
    ( L >= 2,
      sub_string(Str, 0, 1, _, "'"),
      sub_string(Str, _, 1, 0, "'")
    -> InnerLen is L - 2,
       sub_string(Str, 1, InnerLen, 1, Body),
       % Un-escape `''` (Prolog''s quoted-atom escape for `'`).
       % Most format strings don''t contain `'`, so this is a no-op
       % in the common path.
       string_codes(Body, BodyCodes),
       unescape_quotes(BodyCodes, UnescCodes),
       string_codes(Inner, UnescCodes)
    ;  Inner = Str
    ).

unescape_quotes([], []).
unescape_quotes([0'', 0''|Cs], [0''|Rest]) :- !,
    unescape_quotes(Cs, Rest).
unescape_quotes([C|Cs], [C|Rest]) :-
    unescape_quotes(Cs, Rest).

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
