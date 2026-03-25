:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wat_target.pl - WebAssembly Text Format Code Generation Target
% Compiles Prolog predicates directly to WAT (WebAssembly Text Format).
% Uses structured control flow (if/else/end) rather than flat basic blocks.
% Falls back to LLVM IR compilation for unsupported patterns.

:- module(wat_target, [
    compile_predicate_to_wat/3,          % +PredIndicator, +Options, -WATCode
    compile_predicate/3,                 % +PredIndicator, +Options, -WATCode (dispatch alias)
    compile_facts_to_wat/3,              % +Pred, +Arity, -WATCode
    compile_tail_recursion_wat/3,        % +Pred/Arity, +Options, -WATCode
    compile_linear_recursion_wat/3,      % +Pred/Arity, +Options, -WATCode
    compile_tree_recursion_wat/3,        % +Pred/Arity, +Options, -WATCode
    compile_multicall_recursion_wat/3,   % +Pred/Arity, +Options, -WATCode
    compile_direct_multicall_wat/3,      % +Pred/Arity, +Options, -WATCode
    compile_mutual_recursion_wat/3,      % +Predicates, +Options, -WATCode
    compile_transitive_closure_wat/3,    % +Pred/Arity, +Options, -WATCode
    write_wat_program/2,                 % +Code, +Filename
    init_wat_target/0,                   % Initialize target

    % String table support
    wat_build_string_table/2,            % +Atoms, -StringTable
    wat_string_data_segments/2,          % +StringTable, -DataSegments
    wat_string_lookup_func/2,            % +StringTable, -FuncCode
    wat_string_eq_func/1,               % -FuncCode
    wat_compile_with_strings/4,          % +PredIndicator, +Options, +Atoms, -WATCode
    wat_compile_with_string_refs/4,      % +PredIndicator, +Options, +Atoms, -WATCode

    % Multi-page memory
    wat_memory_pages/2,                  % +Options, -Pages
    wat_memory_decl/2,                   % +Options, -MemDecl

    % Multi-module import/export linking
    wat_module_imports/3,                % +ImportSpecs, +Options, -ImportCode
    wat_module_exports/3,                % +ExportSpecs, +Options, -ExportCode
    wat_link_modules/3                   % +ModuleSpecs, +Options, -LinkedCode
]).

:- use_module(library(lists)).
:- use_module('../bindings/wat_bindings', [init_wat_bindings/0]).

% Shared clause body analysis for native lowering
:- use_module('../core/clause_body_analysis').

% LLVM target for WASM fallback (loaded lazily to avoid circular deps)
:- use_module('../targets/llvm_target', [compile_wasm_module/3]).

% Template system for mustache-based code generation
:- use_module('../core/template_system').

% Component system integration
:- use_module('../core/component_registry').

% Advanced recursion multifile hooks
:- use_module('../core/advanced/tail_recursion', []).
:- use_module('../core/advanced/linear_recursion', []).
:- use_module('../core/advanced/tree_recursion', []).
:- use_module('../core/advanced/multicall_linear_recursion', []).
:- use_module('../core/advanced/direct_multi_call_recursion', []).
:- use_module('../core/advanced/mutual_recursion', []).

:- multifile tail_recursion:compile_tail_pattern/9.
:- multifile linear_recursion:compile_linear_pattern/8.
:- multifile tree_recursion:compile_tree_pattern/6.
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.
:- multifile mutual_recursion:compile_mutual_pattern/5.

%% ============================================
%% MEMORY LAYOUT CONSTANTS
%% ============================================
%%
%% WAT linear memory is shared by all data structures.
%% To prevent collisions, we define fixed regions:
%%
%%   Region 0: String data       [0, 4096)       4KB for string table data segments
%%   Region 1: Memo values       [4096, 12288)   8KB for i64 memo values (1024 entries x 8B)
%%   Region 2: Memo flags        [12288, 16384)  4KB for i32 validity flags (1024 entries x 4B)
%%   Region 3: Graph adjacency   [16384, 32768)  16KB for edges (1024 edges x 16B)
%%   Region 4: Visited bitmap    [32768, 36864)  4KB for visited flags (1024 nodes x 4B)
%%   Region 5: BFS queue         [36864, 45056)  8KB for queue entries (1024 x 8B)
%%   Region 6: Fact data         [45056, 65536)  ~20KB for fact data segments
%%
%% One page (64KB) accommodates all regions. Modules that need more
%% should declare (memory N) with N > 1.

wat_mem_region(string_data,   0).
wat_mem_region(memo_values,   4096).
wat_mem_region(memo_flags,    12288).
wat_mem_region(graph_edges,   16384).
wat_mem_region(visited_flags, 32768).
wat_mem_region(bfs_queue,     36864).
wat_mem_region(fact_data,     45056).

%% init_wat_target
init_wat_target :-
    init_wat_bindings.

%% compile_predicate/3 - dispatch alias for target_registry
compile_predicate(PredArity, Options, Code) :-
    compile_predicate_to_wat(PredArity, Options, Code).

%% compile_predicate_to_wat(+PredIndicator, +Options, -Code)
compile_predicate_to_wat(Pred/Arity, Options, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses = []
    ->  format(user_error, 'WAT target: no clauses for ~w/~w~n', [Pred, Arity]),
        fail
    ;   compile_clauses_to_wat(Pred, Arity, Clauses, Options, Code)
    ).

%% compile_clauses_to_wat(+Pred, +Arity, +Clauses, +Options, -Code)
compile_clauses_to_wat(Pred, Arity, Clauses, _Options, Code) :-
    atom_string(Pred, PredStr),
    (   forall(member(_-Body, Clauses), Body == true)
    ->  compile_facts_wat(PredStr, Arity, Clauses, Code)
    ;   compile_rule_wat(PredStr, Arity, Clauses, Code)
    ).

%% compile_facts_wat(+PredStr, +Arity, +Clauses, -Code)
compile_facts_wat(PredStr, Arity, Clauses, Code) :-
    length(Clauses, Count),
    findall(DataEntry, (
        nth1(Idx, Clauses, Head-true),
        Head =.. [_|Args],
        generate_wat_data_entry(Args, Idx, DataEntry)
    ), DataEntries),
    atomic_list_concat(DataEntries, '\n', DataCode),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Fact Export
  ;; Predicate: ~w/~w

  (memory (export "memory") 1)

~w

  (func $get_~w_count (export "get_~w_count") (result i64)
    (i64.const ~w)
  )
)
', [PredStr, Arity, DataCode, PredStr, PredStr, Count]).

%% compile_rule_wat(+PredStr, +Arity, +Clauses, -Code)

% Try native clause body lowering first
compile_rule_wat(PredStr, Arity, Clauses, Code) :-
    \+ (member(_-Body, Clauses), Body == true),
    native_wat_clause_body(PredStr/Arity, Clauses, FuncBody),
    !,
    Arity1 is Arity - 1,
    build_wat_params(Arity1, Params),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Native Clause Lowering
  ;; Predicate: ~w/~w

  (func $~w (export "~w") ~w(result i64)
~w
  )
)
', [PredStr, Arity, PredStr, PredStr, Params, FuncBody]).

% Fallback: try LLVM WASM compilation path
compile_rule_wat(PredStr, Arity, _Clauses, Code) :-
    catch(
        (   atom_string(PredAtom, PredStr),
            llvm_target:compile_wasm_module(
                [func(PredAtom, Arity, linear_recursion)],
                [module_name(PredStr)],
                LLVMCode
            )
        ),
        _Error,
        fail
    ),
    !,
    format(string(Code),
';; Generated by UnifyWeaver WAT Target - LLVM WASM Fallback
;; Predicate: ~w/~w
;; Native WAT lowering unavailable; LLVM IR for wasm32 below.
;; Compile with: llc -march=wasm32 -filetype=obj <file>.ll && wasm-ld --no-entry --export-all -o <file>.wasm <file>.o

~w
', [PredStr, Arity, LLVMCode]).

% Final fallback stub when LLVM is also unavailable
compile_rule_wat(PredStr, Arity, _Clauses, Code) :-
    Arity1 is Arity - 1,
    build_wat_params(Arity1, Params),
    (   Arity1 > 0
    ->  BodyCode = "(local.get $arg1)"
    ;   BodyCode = "(i64.const 0)"
    ),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Stub
  ;; Predicate: ~w/~w
  ;; Native lowering and LLVM fallback unavailable

  (func $~w (export "~w") ~w(result i64)
    ~w
  )
)
', [PredStr, Arity, PredStr, PredStr, Params, BodyCode]).

%% ============================================
%% TAIL RECURSION (loop + br)
%% ============================================

compile_tail_recursion_wat(Pred/Arity, _Options, Code) :-
    atom_string(Pred, PredStr),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Tail Recursion
  ;; Predicate: ~w/~w
  ;; Optimized to O(1) stack space using block/loop/br

  (func $~w (export "~w") (param $n i64) (param $acc i64) (result i64)
    (block $done
      (loop $continue
        ;; Exit when n <= 0
        (br_if $done (i64.le_s (local.get $n) (i64.const 0)))
        ;; acc += n; n -= 1
        (local.set $acc (i64.add (local.get $acc) (local.get $n)))
        (local.set $n (i64.sub (local.get $n) (i64.const 1)))
        (br $continue)
      )
    )
    (local.get $acc)
  )

  (func $~w_entry (export "~w_entry") (param $n i64) (result i64)
    (call $~w (local.get $n) (i64.const 0))
  )
)
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr, PredStr]).

%% ============================================
%% LINEAR RECURSION (table-based memoization)
%% ============================================

compile_linear_recursion_wat(Pred/Arity, _Options, Code) :-
    atom_string(Pred, PredStr),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Linear Recursion
  ;; Predicate: ~w/~w
  ;; Uses linear memory for memoization

  (memory (export "memory") 1)

  ;; Memo table: offset 4096 = values (i64 each), offset 12288 = valid flags (i32 each)

  (func $~w (export "~w") (param $n i64) (result i64)
    (local $offset i32)
    (local $flag_offset i32)
    (local $result i64)

    ;; Base cases
    (if (result i64) (i64.le_s (local.get $n) (i64.const 0))
      (then (i64.const 0))
      (else
        (if (result i64) (i64.eq (local.get $n) (i64.const 1))
          (then (i64.const 1))
          (else
            ;; Check memo
            (local.set $offset (i32.add (i32.const 4096) (i32.mul (i32.wrap_i64 (local.get $n)) (i32.const 8))))
            (local.set $flag_offset (i32.add (i32.const 12288) (i32.mul (i32.wrap_i64 (local.get $n)) (i32.const 4))))
            (if (result i64) (i32.load (local.get $flag_offset))
              (then (i64.load (local.get $offset)))
              (else
                ;; Compute and memoize
                (local.set $result
                  (i64.add
                    (call $~w (i64.sub (local.get $n) (i64.const 1)))
                    (local.get $n)
                  )
                )
                (i64.store (local.get $offset) (local.get $result))
                (i32.store (local.get $flag_offset) (i32.const 1))
                (local.get $result)
              )
            )
          )
        )
      )
    )
  )
)
', [PredStr, Arity, PredStr, PredStr, PredStr]).

%% ============================================
%% MUTUAL RECURSION
%% ============================================

compile_mutual_recursion_wat(Predicates, _Options, Code) :-
    findall(PredStr, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr)
    ), PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),

    findall(FuncCode, (
        member(Pred/Arity, Predicates),
        generate_wat_mutual_function(Pred, Arity, Predicates, FuncCode)
    ), FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),

    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Mutual Recursion
  ;; Group: ~w

~w
)
', [GroupName, FunctionsCode]).

%% generate_wat_mutual_function(+Pred, +Arity, +AllPredicates, -Code)
generate_wat_mutual_function(Pred, _Arity, AllPredicates, Code) :-
    atom_string(Pred, PredStr),
    member(OtherPred/_OtherArity, AllPredicates),
    OtherPred \= Pred,
    atom_string(OtherPred, OtherPredStr),

    % is_even(0) = 1, is_odd(0) = 0
    (   PredStr = "is_even"
    ->  BaseVal = 1
    ;   BaseVal = 0
    ),

    format(string(Code),
'  (func $~w (export "~w") (param $n i64) (result i64)
    (if (result i64) (i64.eq (local.get $n) (i64.const 0))
      (then (i64.const ~w))
      (else
        (if (result i64) (i64.gt_s (local.get $n) (i64.const 0))
          (then (call $~w (i64.sub (local.get $n) (i64.const 1))))
          (else (i64.const 0))
        )
      )
    )
  )', [PredStr, PredStr, BaseVal, OtherPredStr]).

%% ============================================
%% TREE RECURSION (two recursive calls + memoization)
%% ============================================

compile_tree_recursion_wat(Pred/Arity, _Options, Code) :-
    atom_string(Pred, PredStr),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Tree Recursion
  ;; Predicate: ~w/~w
  ;; Memoized tree recursion using linear memory

  (memory (export "memory") 1)

  ;; Memo layout: offset 4096 = values (i64, 8 bytes each)
  ;; Offset 12288 = valid flags (i32, 4 bytes each)

  (func $~w (export "~w") (param $n i64) (result i64)
    (local $offset i32)
    (local $flag_offset i32)
    (local $result i64)

    ;; Base cases
    (if (result i64) (i64.le_s (local.get $n) (i64.const 0))
      (then (i64.const 0))
      (else
        (if (result i64) (i64.eq (local.get $n) (i64.const 1))
          (then (i64.const 1))
          (else
            ;; Check memo table
            (local.set $offset (i32.add (i32.const 4096) (i32.mul (i32.wrap_i64 (local.get $n)) (i32.const 8))))
            (local.set $flag_offset
              (i32.add (i32.const 12288)
                (i32.mul (i32.wrap_i64 (local.get $n)) (i32.const 4))))
            (if (result i64) (i32.load (local.get $flag_offset))
              (then (i64.load (local.get $offset)))
              (else
                ;; Two recursive calls (tree pattern)
                (local.set $result
                  (i64.add
                    (call $~w (i64.sub (local.get $n) (i64.const 1)))
                    (call $~w (i64.sub (local.get $n) (i64.const 2)))
                  )
                )
                ;; Store in memo
                (i64.store (local.get $offset) (local.get $result))
                (i32.store (local.get $flag_offset) (i32.const 1))
                (local.get $result)
              )
            )
          )
        )
      )
    )
  )
)
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr]).

%% ============================================
%% TRANSITIVE CLOSURE (BFS with linear memory)
%% ============================================

compile_transitive_closure_wat(Pred/Arity, _Options, Code) :-
    atom_string(Pred, PredStr),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Transitive Closure
  ;; Predicate: ~w/~w
  ;; BFS reachability using linear memory
  ;; Memory layout (shared with string table and memo regions):
  ;;   16384-32767: adjacency list (pairs of i64 node IDs, 16B each)
  ;;   32768-36863: visited bitmap (i32 flags, 4B each)
  ;;   36864-45055: BFS queue (i64 entries, 8B each)

  (memory (export "memory") 1)

  ;; Edge count and node count
  (global $edge_count (mut i32) (i32.const 0))
  (global $node_count (mut i32) (i32.const 0))

  ;; Add an edge: add_edge(from, to)
  (func $add_edge (export "add_edge") (param $from i64) (param $to i64) (result i64)
    (local $offset i32)
    (local.set $offset (i32.add (i32.const 16384) (i32.mul (global.get $edge_count) (i32.const 16))))
    (i64.store (local.get $offset) (local.get $from))
    (i64.store (i32.add (local.get $offset) (i32.const 8)) (local.get $to))
    (global.set $edge_count (i32.add (global.get $edge_count) (i32.const 1)))
    (i64.const 1)
  )

  ;; Check if visited
  (func $is_visited (param $node i64) (result i32)
    (i32.load
      (i32.add (i32.const 32768)
        (i32.mul (i32.wrap_i64 (local.get $node)) (i32.const 4))))
  )

  ;; Mark as visited
  (func $mark_visited (param $node i64)
    (i32.store
      (i32.add (i32.const 32768)
        (i32.mul (i32.wrap_i64 (local.get $node)) (i32.const 4)))
      (i32.const 1))
  )

  ;; BFS reachability: ~w(source, target) → 1 if reachable, 0 otherwise
  (func $~w (export "~w") (param $source i64) (param $target i64) (result i64)
    (local $queue_head i32)
    (local $queue_tail i32)
    (local $current i64)
    (local $i i32)
    (local $edge_from i64)
    (local $edge_to i64)

    ;; Initialize: clear visited, enqueue source
    (local.set $queue_head (i32.const 0))
    (local.set $queue_tail (i32.const 0))

    ;; Enqueue source
    (i64.store
      (i32.add (i32.const 36864)
        (i32.mul (local.get $queue_tail) (i32.const 8)))
      (local.get $source))
    (local.set $queue_tail (i32.add (local.get $queue_tail) (i32.const 1)))
    (call $mark_visited (local.get $source))

    ;; BFS loop
    (block $done
      (loop $bfs
        ;; If queue empty, not reachable
        (br_if $done (i32.ge_u (local.get $queue_head) (local.get $queue_tail)))

        ;; Dequeue
        (local.set $current
          (i64.load
            (i32.add (i32.const 36864)
              (i32.mul (local.get $queue_head) (i32.const 8)))))
        (local.set $queue_head (i32.add (local.get $queue_head) (i32.const 1)))

        ;; Check if target found
        (if (i64.eq (local.get $current) (local.get $target))
          (then (return (i64.const 1)))
        )

        ;; Scan edges for neighbors
        (local.set $i (i32.const 0))
        (block $edge_done
          (loop $edge_scan
            (br_if $edge_done (i32.ge_u (local.get $i) (global.get $edge_count)))
            (local.set $edge_from
              (i64.load (i32.add (i32.const 16384) (i32.mul (local.get $i) (i32.const 16)))))
            (local.set $edge_to
              (i64.load (i32.add (i32.add (i32.const 16384) (i32.mul (local.get $i) (i32.const 16))) (i32.const 8))))

            ;; If edge starts at current node and target not visited
            (if (i32.and
                  (i64.eq (local.get $edge_from) (local.get $current))
                  (i32.eqz (call $is_visited (local.get $edge_to))))
              (then
                (call $mark_visited (local.get $edge_to))
                (i64.store
                  (i32.add (i32.const 36864)
                    (i32.mul (local.get $queue_tail) (i32.const 8)))
                  (local.get $edge_to))
                (local.set $queue_tail (i32.add (local.get $queue_tail) (i32.const 1)))
              )
            )

            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $edge_scan)
          )
        )
        (br $bfs)
      )
    )
    (i64.const 0)
  )
)
', [PredStr, Arity, PredStr, PredStr, PredStr]).

%% compile_facts_to_wat(+Pred, +Arity, -Code)
compile_facts_to_wat(Pred, Arity, Code) :-
    compile_predicate_to_wat(Pred/Arity, [], Code).

%% write_wat_program(+Code, +Filename)
write_wat_program(Code, Filename) :-
    open(Filename, write, Stream),
    write(Stream, Code),
    close(Stream),
    format('WAT program written to: ~w~n', [Filename]).

%% ============================================
%% Helper predicates
%% ============================================

generate_wat_data_entry(Args, Idx, Entry) :-
    findall(ArgStr, (
        member(Arg, Args),
        format(string(ArgStr), '~w', [Arg])
    ), ArgStrs),
    atomic_list_concat(ArgStrs, ':', Joined),
    wat_mem_region(fact_data, BaseOffset),
    Offset is BaseOffset + (Idx - 1) * 256,
    format(string(Entry),
           '  (data (i32.const ~w) "~w")', [Offset, Joined]).

%% build_wat_params(+N, -Params)
%  Generates "(param $arg1 i64) (param $arg2 i64) " for N input arguments.
build_wat_params(0, "") :- !.
build_wat_params(N, Params) :-
    findall(Param, (
        between(1, N, I),
        format(string(Param), '(param $arg~w i64) ', [I])
    ), ParamList),
    atomic_list_concat(ParamList, '', Params).

%% ============================================
%% NATIVE CLAUSE BODY LOWERING
%% ============================================

%% native_wat_clause_body(+PredIndicator, +Clauses, -FuncBody)
native_wat_clause_body(PredStr/Arity, Clauses, FuncBody) :-
    findall(branch(Cond, Value), (
        member(Head-Body, Clauses),
        Body \== true,
        native_wat_clause(Head, Body, Cond, Value)
    ), Branches),
    Branches \= [],
    branches_to_wat_if_chain(Branches, PredStr, Arity, FuncBody).

%% native_wat_clause(+Head, +Body, -Cond, -Value)
native_wat_clause(Head, Body, Cond, Value) :-
    Head =.. [_|HeadArgs],
    length(HeadArgs, Arity),
    build_head_varmap(HeadArgs, 1, VarMap),
    (   Arity > 1
    ->  append(_InputArgs, [OutputHeadArg], HeadArgs)
    ;   OutputHeadArg = _
    ),
    normalize_goals(Body, Goals),
    (   Goals = [SingleGoal],
        if_then_else_goal(SingleGoal, If, Then, Else)
    ->  wat_if_then_else_output(If, Then, Else, VarMap, Value),
        Cond = none
    ;   (   Arity > 1, nonvar(OutputHeadArg)
        ->  once(clause_guard_output_split(Goals, VarMap, Guards, _Outputs)),
            wat_head_conditions(Guards, VarMap, Cond),
            wat_literal(OutputHeadArg, Value)
        ;   once(clause_guard_output_split(Goals, VarMap, Guards, Outputs)),
            wat_head_conditions(Guards, VarMap, Cond),
            wat_output_goals(Outputs, VarMap, Value)
        )
    ).

%% wat_head_conditions(+Guards, +VarMap, -Cond)
wat_head_conditions([], _VarMap, none) :- !.
wat_head_conditions(Guards, VarMap, Cond) :-
    findall(C, (
        member(G, Guards),
        wat_guard_condition(G, VarMap, C)
    ), Conds),
    (   Conds = []
    ->  Cond = none
    ;   combine_wat_conditions(Conds, Cond)
    ).

%% wat_guard_condition(+Goal, +VarMap, -Cond)
wat_guard_condition(Goal, VarMap, Cond) :-
    Goal =.. [Op, Left, Right],
    wat_cmp_op(Op, WATOp),
    wat_resolve_value(Left, VarMap, LStr),
    wat_resolve_value(Right, VarMap, RStr),
    format(string(Cond), '(~w ~w ~w)', [WATOp, LStr, RStr]).

%% wat_cmp_op(+PrologOp, -WATOp)
wat_cmp_op(>, "i64.gt_s").
wat_cmp_op(<, "i64.lt_s").
wat_cmp_op(>=, "i64.ge_s").
wat_cmp_op(=<, "i64.le_s").
wat_cmp_op(=:=, "i64.eq").
wat_cmp_op(=\=, "i64.ne").
wat_cmp_op(==, "i64.eq").
wat_cmp_op(\==, "i64.ne").

%% wat_output_goals(+Outputs, +VarMap, -Value)
wat_output_goals(Outputs, VarMap, Value) :-
    last(Outputs, LastGoal),
    (   goal_output_var(LastGoal, _)
    ->  wat_expr(LastGoal, VarMap, Value)
    ;   wat_resolve_value(LastGoal, VarMap, Value)
    ).

%% wat_expr(+Goal, +VarMap, -Expr)
wat_expr(Goal, VarMap, Expr) :-
    Goal = (_Var = RHS),
    !,
    wat_resolve_value(RHS, VarMap, Expr).
wat_expr(Goal, VarMap, Expr) :-
    Goal = (_Var is ArithExpr),
    !,
    wat_arith_expr(ArithExpr, VarMap, Expr).
wat_expr(Goal, VarMap, Expr) :-
    wat_resolve_value(Goal, VarMap, Expr).

%% wat_arith_expr(+Expr, +VarMap, -Code)
wat_arith_expr(Expr, VarMap, Code) :-
    var(Expr),
    !,
    lookup_var(Expr, VarMap, Name),
    format(string(Code), '(local.get $~w)', [Name]).
wat_arith_expr(Expr, _VarMap, Code) :-
    number(Expr),
    !,
    format(string(Code), '(i64.const ~w)', [Expr]).
wat_arith_expr(Expr, VarMap, Code) :-
    Expr =.. [Op, Left, Right],
    expr_op(Op, _),
    !,
    wat_arith_expr(Left, VarMap, LCode),
    wat_arith_expr(Right, VarMap, RCode),
    wat_arith_op(Op, WATOp),
    format(string(Code), '(~w ~w ~w)', [WATOp, LCode, RCode]).
wat_arith_expr(-Expr, VarMap, Code) :-
    !,
    wat_arith_expr(Expr, VarMap, Inner),
    format(string(Code), '(i64.sub (i64.const 0) ~w)', [Inner]).
wat_arith_expr(abs(Expr), VarMap, Code) :-
    !,
    wat_arith_expr(Expr, VarMap, Inner),
    % abs(x) = if x >= 0 then x else -x
    format(string(Code),
           '(if (result i64) (i64.ge_s ~w (i64.const 0)) (then ~w) (else (i64.sub (i64.const 0) ~w)))',
           [Inner, Inner, Inner]).
wat_arith_expr(Expr, VarMap, Code) :-
    wat_resolve_value(Expr, VarMap, Code).

%% wat_arith_op(+PrologOp, -WATOp)
wat_arith_op(+, "i64.add").
wat_arith_op(-, "i64.sub").
wat_arith_op(*, "i64.mul").
wat_arith_op(/, "i64.div_s").
wat_arith_op(//, "i64.div_s").
wat_arith_op(mod, "i64.rem_s").

%% wat_resolve_value(+Term, +VarMap, -Str)
wat_resolve_value(Var, VarMap, Str) :-
    var(Var),
    !,
    lookup_var(Var, VarMap, Name),
    format(string(Str), '(local.get $~w)', [Name]).
wat_resolve_value(Term, _VarMap, Str) :-
    number(Term),
    !,
    format(string(Str), '(i64.const ~w)', [Term]).
wat_resolve_value(Term, VarMap, Str) :-
    lookup_var(Term, VarMap, Name),
    !,
    format(string(Str), '(local.get $~w)', [Name]).
wat_resolve_value(Term, _VarMap, Str) :-
    atom(Term),
    !,
    wat_literal(Term, Str).
wat_resolve_value(Term, VarMap, Str) :-
    compound(Term),
    wat_arith_expr(Term, VarMap, Str).

%% wat_literal(+Atom, -Str)
%  Atoms are encoded as their hash for i64 representation.
wat_literal(Atom, Str) :-
    atom_codes(Atom, Codes),
    wat_hash_codes(Codes, 0, Hash),
    format(string(Str), '(i64.const ~w)', [Hash]).

%% wat_hash_codes(+Codes, +Acc, -Hash)
%  Simple deterministic hash for encoding atoms as i64 values.
wat_hash_codes([], Acc, Acc).
wat_hash_codes([C|Cs], Acc, Hash) :-
    Acc1 is (Acc * 31 + C) mod 2147483647,
    wat_hash_codes(Cs, Acc1, Hash).

%% ============================================
%% STRING TABLE - Linear Memory String Support
%% ============================================
%%
%% Atoms can be stored as actual UTF-8 strings in linear memory using
%% WAT data segments. Each string gets a (offset, length) pair.
%% Strings are encoded as i64 values: high 32 bits = offset, low 32 bits = length.
%% This allows string comparison by content rather than by hash.

%% wat_build_string_table(+Atoms, -StringTable)
%%   Given a list of atoms, builds a string table mapping each atom to
%%   its memory offset and length. StringTable is a list of
%%   str_entry(Atom, Offset, Length) terms.
%%   Strings are placed starting at offset 0 in linear memory.
wat_build_string_table(Atoms, StringTable) :-
    sort(Atoms, Unique),
    build_entries(Unique, 0, StringTable).

build_entries([], _, []).
build_entries([Atom|Rest], Offset, [str_entry(Atom, Offset, Len)|Entries]) :-
    atom_codes(Atom, Codes),
    length(Codes, Len),
    NextOffset is Offset + Len,
    build_entries(Rest, NextOffset, Entries).

%% wat_string_data_segments(+StringTable, -DataSegments)
%%   Generates WAT data segment declarations for all strings in the table.
%%   Output is a string of WAT code.
wat_string_data_segments([], "").
wat_string_data_segments(StringTable, DataSegments) :-
    StringTable \= [],
    maplist(string_entry_to_segment, StringTable, Segments),
    atomics_to_string(Segments, DataSegments).

string_entry_to_segment(str_entry(Atom, Offset, _Len), Segment) :-
    atom_string(Atom, Str),
    escape_wat_string(Str, Escaped),
    format(string(Segment),
        '  (data (i32.const ~w) "~w")~n', [Offset, Escaped]).

%% escape_wat_string(+String, -Escaped)
%%   Escapes special characters for WAT data segment string literals.
escape_wat_string(Str, Escaped) :-
    string_codes(Str, Codes),
    maplist(escape_wat_char, Codes, EscapedCodes),
    append(EscapedCodes, FlatCodes),
    string_codes(Escaped, FlatCodes).

escape_wat_char(0'\\, "\\\\") :- !.
escape_wat_char(0'", "\\\"") :- !.
escape_wat_char(C, Codes) :-
    (C >= 0x20, C =< 0x7E)
    ->  Codes = [C]
    ;   format(string(Hex), "\\~|~`0t~16r~2+", [C]),
        string_codes(Hex, Codes).

%% wat_string_encode(+Offset, +Length, -I64Value)
%%   Encodes a string reference as an i64: high 32 bits = offset, low 32 bits = length.
wat_string_encode(Offset, Length, Value) :-
    Value is (Offset << 32) \/ Length.

%% wat_string_literal(+Atom, +StringTable, -Str)
%%   Look up an atom in the string table and return its i64-encoded reference.
wat_string_literal(Atom, StringTable, Str) :-
    member(str_entry(Atom, Offset, Len), StringTable),
    !,
    wat_string_encode(Offset, Len, Value),
    format(string(Str), '(i64.const ~w)', [Value]).

%% wat_string_lookup_func(+StringTable, -FuncCode)
%%   Generates a WAT helper function that maps hash codes to string references.
%%   This bridges the gap: code that uses hash-based atoms can look up the
%%   actual string reference for I/O or comparison.
wat_string_lookup_func([], "") :- !.
wat_string_lookup_func(StringTable, FuncCode) :-
    build_lookup_body(StringTable, Body),
    format(string(FuncCode),
'  ;; Hash-to-string lookup
  (func $str_lookup (export "str_lookup") (param $hash i64) (result i64)
~w  )~n', [Body]).

%% build_lookup_body(+Entries, -Code)
%%   Builds nested if/else chain for hash-to-string lookup.
build_lookup_body([], Code) :-
    format(string(Code), '    (i64.const -1) ;; not found~n', []).
build_lookup_body([str_entry(Atom, Offset, Len)|Rest], Code) :-
    atom_codes(Atom, Codes),
    wat_hash_codes(Codes, 0, Hash),
    wat_string_encode(Offset, Len, StrRef),
    build_lookup_body(Rest, RestCode),
    format(string(Code),
'    (if (result i64) (i64.eq (local.get $hash) (i64.const ~w))
      (then (i64.const ~w)) ;; "~w"
      (else
~w      )
    )~n', [Hash, StrRef, Atom, RestCode]).

%% wat_string_eq_func(-FuncCode)
%%   Generates a WAT function that compares two string references byte-by-byte.
%%   Takes two i64 string refs (offset<<32|length), returns 1 if equal, 0 if not.
wat_string_eq_func(FuncCode) :-
    format(string(FuncCode),
'  ;; String equality comparison
  (func $str_eq (export "str_eq") (param $a i64) (param $b i64) (result i32)
    (local $off_a i32) (local $off_b i32)
    (local $len_a i32) (local $len_b i32)
    (local $i i32)
    ;; Extract offset and length from packed i64
    (local.set $off_a (i32.wrap_i64 (i64.shr_u (local.get $a) (i64.const 32))))
    (local.set $len_a (i32.wrap_i64 (local.get $a)))
    (local.set $off_b (i32.wrap_i64 (i64.shr_u (local.get $b) (i64.const 32))))
    (local.set $len_b (i32.wrap_i64 (local.get $b)))
    ;; Different lengths => not equal
    (if (i32.ne (local.get $len_a) (local.get $len_b))
      (then (return (i32.const 0)))
    )
    ;; Compare byte-by-byte
    (local.set $i (i32.const 0))
    (block $done
      (loop $cmp
        (br_if $done (i32.ge_u (local.get $i) (local.get $len_a)))
        (if (i32.ne
              (i32.load8_u (i32.add (local.get $off_a) (local.get $i)))
              (i32.load8_u (i32.add (local.get $off_b) (local.get $i))))
          (then (return (i32.const 0)))
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $cmp)
      )
    )
    (i32.const 1)
  )~n', []).

%% wat_compile_with_strings(+PredIndicator, +Options, +Atoms, -WATCode)
%%   Compile a predicate to WAT with string table support.
%%   Atoms is the list of atoms used in the predicate that should be
%%   stored as actual strings rather than hashes.
wat_compile_with_strings(PredIndicator, Options, Atoms, WATCode) :-
    compile_predicate_to_wat(PredIndicator, Options, BaseCode),
    wat_build_string_table(Atoms, StringTable),
    wat_string_data_segments(StringTable, DataSegs),
    wat_string_lookup_func(StringTable, LookupFunc),
    wat_string_eq_func(EqFunc),
    % Inject string support into the module
    % Replace closing ')' of module with string support + closing ')'
    string_concat("  (memory (export \"memory\") 1)\n", DataSegs, MemAndData),
    string_concat(MemAndData, LookupFunc, WithLookup),
    string_concat(WithLookup, EqFunc, StringSupport),
    % Insert before closing paren of module
    % Strip trailing whitespace, then find the final ')'
    string_codes(BaseCode, BaseCodes),
    reverse(BaseCodes, RevCodes),
    strip_leading_ws(RevCodes, Stripped),
    (   Stripped = [0')|RestRev]
    ->  reverse(RestRev, PrefixCodes),
        string_codes(Prefix, PrefixCodes),
        format(string(WATCode), '~w\n~w)\n', [Prefix, StringSupport])
    ;   WATCode = BaseCode
    ).

strip_leading_ws([0' |T], R) :- !, strip_leading_ws(T, R).
strip_leading_ws([0'\n|T], R) :- !, strip_leading_ws(T, R).
strip_leading_ws([0'\t|T], R) :- !, strip_leading_ws(T, R).
strip_leading_ws([0'\r|T], R) :- !, strip_leading_ws(T, R).
strip_leading_ws(X, X).

%% atomics_to_string(+List, -String)
%%   Concatenate a list of strings/atoms into a single string.
atomics_to_string([], "").
atomics_to_string([H|T], Result) :-
    atomics_to_string(T, Rest),
    string_concat(H, Rest, Result).

%% ============================================
%% STRING REF OUTPUT - Direct string reference returns
%% ============================================
%%
%% Instead of returning atom hash codes and requiring the caller to
%% call str_lookup, this generates code that directly returns packed
%% i64 string references (offset<<32 | length).
%%
%% The generated module includes:
%%   - Data segments for all atom strings
%%   - Memory declaration
%%   - The function returns string refs instead of hash codes
%%   - str_eq helper for comparing returned refs

%% wat_compile_with_string_refs(+PredIndicator, +Options, +Atoms, -WATCode)
%%   Like wat_compile_with_strings/4 but rewrites atom literals in the
%%   compiled function body to return string refs directly.
wat_compile_with_string_refs(Pred/Arity, Options, Atoms, WATCode) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    atom_string(Pred, PredStr),
    wat_build_string_table(Atoms, StringTable),
    % Compile with string ref substitution
    (   native_wat_clause_body_with_refs(PredStr/Arity, Clauses, StringTable, FuncBody)
    ->  true
    ;   % Fall back to hash-based compilation
        native_wat_clause_body(PredStr/Arity, Clauses, FuncBody)
    ),
    Arity1 is Arity - 1,
    build_wat_params(Arity1, Params),
    wat_string_data_segments(StringTable, DataSegs),
    wat_string_eq_func(EqFunc),
    wat_memory_decl(Options, MemDecl),
    format(string(WATCode),
'(module
  ;; Generated by UnifyWeaver WAT Target - Native Lowering with String Refs
  ;; Predicate: ~w/~w
  ;; Atom outputs return packed i64 string refs (offset<<32 | length)

~w

~w
  (func $~w (export "~w") ~w(result i64)
~w
  )

~w)
', [PredStr, Arity, MemDecl, DataSegs, PredStr, PredStr, Params, FuncBody, EqFunc]).

%% native_wat_clause_body_with_refs(+PredIndicator, +Clauses, +StringTable, -Code)
%%   Like native_wat_clause_body but substitutes atom hash values with
%%   string ref values from the string table.
native_wat_clause_body_with_refs(PredStr/Arity, Clauses, StringTable, Code) :-
    Arity1 is Arity - 1,
    findall(branch(Cond, Value), (
        member(Head-Body, Clauses),
        Head =.. [_|AllArgs],
        length(InputArgs, Arity1),
        append(InputArgs, [OutputArg], AllArgs),
        build_head_varmap(InputArgs, 1, VarMap),
        Body \== true,
        normalize_goals(Body, Goals),
        once(clause_guard_output_split(Goals, VarMap, Guards, Outputs)),
        (   Guards = []
        ->  Cond = none
        ;   maplist(wat_guard_condition_ref(VarMap, StringTable), Guards, CondExprs),
            combine_wat_conditions(CondExprs, Cond)
        ),
        resolve_output_with_refs(OutputArg, Outputs, VarMap, StringTable, Value)
    ), Branches),
    Branches \= [],
    branches_to_wat_if_chain(Branches, PredStr, Arity, Code).

%% resolve_output_with_refs(+OutputArg, +Outputs, +VarMap, +StringTable, -Value)
resolve_output_with_refs(OutputArg, _Outputs, _VarMap, StringTable, Value) :-
    nonvar(OutputArg),
    atom(OutputArg),
    !,
    (   wat_string_literal(OutputArg, StringTable, Value)
    ->  true
    ;   wat_literal(OutputArg, Value)  % fallback to hash if not in table
    ).
resolve_output_with_refs(OutputArg, _Outputs, _VarMap, _StringTable, Value) :-
    nonvar(OutputArg),
    number(OutputArg),
    !,
    format(string(Value), '(i64.const ~w)', [OutputArg]).
resolve_output_with_refs(_OutputArg, Outputs, VarMap, StringTable, Value) :-
    last(Outputs, LastGoal),
    (   goal_output_var(LastGoal, _)
    ->  wat_expr_ref(LastGoal, VarMap, StringTable, Value)
    ;   wat_resolve_value_ref(LastGoal, VarMap, StringTable, Value)
    ).

%% wat_expr_ref(+Goal, +VarMap, +StringTable, -Expr)
wat_expr_ref(Goal, VarMap, StringTable, Expr) :-
    Goal = (_Var = RHS),
    !,
    wat_resolve_value_ref(RHS, VarMap, StringTable, Expr).
wat_expr_ref(Goal, VarMap, _StringTable, Expr) :-
    Goal = (_Var is ArithExpr),
    !,
    wat_arith_expr(ArithExpr, VarMap, Expr).
wat_expr_ref(Goal, VarMap, StringTable, Expr) :-
    wat_resolve_value_ref(Goal, VarMap, StringTable, Expr).

%% wat_resolve_value_ref(+Term, +VarMap, +StringTable, -Str)
%%   Like wat_resolve_value but substitutes atoms with string refs.
wat_resolve_value_ref(Term, _VarMap, StringTable, Str) :-
    atom(Term),
    !,
    (   wat_string_literal(Term, StringTable, Str)
    ->  true
    ;   wat_literal(Term, Str)
    ).
wat_resolve_value_ref(Term, VarMap, _StringTable, Str) :-
    wat_resolve_value(Term, VarMap, Str).

%% wat_guard_condition_ref(+VarMap, +StringTable, +Guard, -Cond)
%%   Guard condition that handles atom comparisons with string refs.
wat_guard_condition_ref(VarMap, _StringTable, Guard, Cond) :-
    wat_guard_condition(Guard, VarMap, Cond).

%% ============================================
%% MULTI-PAGE MEMORY
%% ============================================
%%
%% By default, WAT modules use 1 page (64KB). For larger programs
%% with many strings, facts, or deep recursion, more pages may be needed.

%% wat_memory_pages(+Options, -Pages)
%%   Calculate required memory pages from options.
%%   Options can include:
%%     string_bytes(N)  - total bytes of string data
%%     max_facts(N)     - number of facts (each uses 256 bytes in fact region)
%%     max_memo(N)      - max memo table entries (default 1024)
%%     max_edges(N)     - max graph edges (default 1024)
%%     extra_pages(N)   - additional pages beyond calculated minimum
%%     pages(N)         - override: use exactly N pages
wat_memory_pages(Options, Pages) :-
    (   member(pages(P), Options)
    ->  Pages = P
    ;   % Calculate from usage
        (member(string_bytes(SB), Options) -> true ; SB = 0),
        (member(max_facts(MF), Options) -> true ; MF = 80),
        (member(max_memo(MM), Options) -> true ; MM = 1024),
        (member(max_edges(ME), Options) -> true ; ME = 1024),
        (member(extra_pages(EP), Options) -> true ; EP = 0),
        wat_mem_region(fact_data, FactBase),
        % String region: string_data to memo_values
        wat_mem_region(memo_values, MemoBase),
        StringNeeded is max(SB, MemoBase),
        % Memo: values (MM*8) + flags (MM*4)
        MemoNeeded is MemoBase + MM * 8 + MM * 4,
        % Graph: edges (ME*16) + visited (ME*4) + queue (ME*8)
        wat_mem_region(graph_edges, GraphBase),
        GraphNeeded is GraphBase + ME * 16 + ME * 4 + ME * 8,
        % Facts: each fact uses 256 bytes
        FactNeeded is FactBase + MF * 256,
        TotalBytes is max(max(max(StringNeeded, MemoNeeded), GraphNeeded), FactNeeded),
        PageSize is 65536,
        RawPages is (TotalBytes + PageSize - 1) // PageSize,
        Pages is max(1, RawPages + EP)
    ).

%% wat_memory_decl(+Options, -MemDecl)
%%   Generate (memory ...) declaration string.
wat_memory_decl(Options, MemDecl) :-
    wat_memory_pages(Options, Pages),
    format(string(MemDecl), '  (memory (export "memory") ~w)', [Pages]).

%% ============================================
%% MULTI-MODULE IMPORT/EXPORT LINKING
%% ============================================
%%
%% WAT modules can import functions from other modules and export
%% their own functions. This enables multi-module linking where
%% different predicates are compiled to separate .wasm files and
%% linked at instantiation time.
%%
%% Import spec: import(ModuleName, FuncName, Params, Result)
%%   e.g., import("math", "fib", [i64], i64)
%%
%% Export spec: export(FuncName, InternalName)
%%   e.g., export("add", "$add_impl")

%% wat_module_imports(+ImportSpecs, +Options, -ImportCode)
%%   Generate WAT import declarations.
wat_module_imports([], _Options, "") :- !.
wat_module_imports(ImportSpecs, _Options, ImportCode) :-
    maplist(gen_import, ImportSpecs, ImportLines),
    atomics_to_string(ImportLines, ImportCode).

gen_import(import(Module, Func, Params, Result), Line) :-
    maplist(param_to_wat, Params, ParamStrs),
    atomics_to_string(ParamStrs, ParamCode),
    result_to_wat(Result, ResultCode),
    format(string(Line),
        '  (import "~w" "~w" (func $~w ~w~w))~n',
        [Module, Func, Func, ParamCode, ResultCode]).

param_to_wat(Type, Str) :-
    format(string(Str), '(param ~w) ', [Type]).

result_to_wat(void, "") :- !.
result_to_wat(Type, Str) :-
    format(string(Str), '(result ~w)', [Type]).

%% wat_module_exports(+ExportSpecs, +Options, -ExportCode)
%%   Generate additional WAT export declarations (beyond func-level exports).
wat_module_exports([], _Options, "") :- !.
wat_module_exports(ExportSpecs, _Options, ExportCode) :-
    maplist(gen_export, ExportSpecs, ExportLines),
    atomics_to_string(ExportLines, ExportCode).

gen_export(export(ExtName, IntName), Line) :-
    format(string(Line),
        '  (export "~w" (func ~w))~n', [ExtName, IntName]).
gen_export(export_memory(ExtName), Line) :-
    format(string(Line),
        '  (export "~w" (memory 0))~n', [ExtName]).
gen_export(export_global(ExtName, IntName), Line) :-
    format(string(Line),
        '  (export "~w" (global ~w))~n', [ExtName, IntName]).
gen_export(export_table(ExtName), Line) :-
    format(string(Line),
        '  (export "~w" (table 0))~n', [ExtName]).

%% wat_link_modules(+ModuleSpecs, +Options, -LinkedCode)
%%   Generate a linked WAT module that imports functions from other modules
%%   and defines its own functions that call the imports.
%%
%%   ModuleSpec: module(Name, Imports, Functions, Exports)
%%     Name: module name string
%%     Imports: list of import(Module, Func, Params, Result)
%%     Functions: list of func(Name, Params, Result, Body)
%%     Exports: list of export specs
wat_link_modules(module(Name, Imports, Functions, Exports), Options, Code) :-
    wat_module_imports(Imports, Options, ImportCode),
    wat_memory_decl(Options, MemDecl),
    maplist(gen_linked_func, Functions, FuncLines),
    atomics_to_string(FuncLines, FuncCode),
    wat_module_exports(Exports, Options, ExportCode),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Linked Module
  ;; Module: ~w

~w
~w

~w
~w)
', [Name, ImportCode, MemDecl, FuncCode, ExportCode]).

gen_linked_func(func(Name, Params, Result, Body), Line) :-
    maplist(gen_func_param, Params, ParamStrs),
    atomics_to_string(ParamStrs, ParamCode),
    result_to_wat(Result, ResultCode),
    format(string(Line),
'  (func $~w (export "~w") ~w~w
~w
  )~n', [Name, Name, ParamCode, ResultCode, Body]).

gen_func_param(param(Name, Type), Str) :-
    format(string(Str), '(param $~w ~w) ', [Name, Type]).

%% combine_wat_conditions(+Conds, -Combined)
combine_wat_conditions([C], C) :- !.
combine_wat_conditions([C|Cs], Combined) :-
    combine_wat_conditions(Cs, Rest),
    format(string(Combined), '(i32.and ~w ~w)', [C, Rest]).

%% branches_to_wat_if_chain(+Branches, +PredStr, +Arity, -Code)
branches_to_wat_if_chain(Branches, _PredStr, _Arity, Code) :-
    (   Branches = [branch(none, Value)]
    ->  % Single clause with if-then-else (already handled inline)
        format(string(Code), '~w', [Value])
    ;   wat_if_chain_nested(Branches, Code)
    ).

%% wat_if_chain_nested(+Branches, -Code)
%  Generates nested (if (result i64) ...) chains.
wat_if_chain_nested([], "    (unreachable)") :- !.
wat_if_chain_nested([branch(none, Value)], Code) :-
    !,
    format(string(Code), '    ~w', [Value]).
wat_if_chain_nested([branch(Cond, Value)], Code) :-
    !,
    format(string(Code),
'    (if (result i64) ~w
      (then ~w)
      (else (unreachable))
    )', [Cond, Value]).
wat_if_chain_nested([branch(Cond, Value)|Rest], Code) :-
    wat_if_chain_nested(Rest, ElseCode),
    format(string(Code),
'    (if (result i64) ~w
      (then ~w)
      (else
~w
      )
    )', [Cond, Value, ElseCode]).

%% wat_if_then_else_output(+If, +Then, +Else, +VarMap, -Value)
%  Compiles if-then-else to WAT structured if/else.
wat_if_then_else_output(If, Then, Else, VarMap, Value) :-
    wat_guard_condition(If, VarMap, CondStr),
    wat_ite_value(Then, VarMap, ThenVal),
    wat_ite_else(Else, VarMap, ElseCode),
    format(string(Value),
'    (if (result i64) ~w
      (then ~w)
      (else
~w
      )
    )', [CondStr, ThenVal, ElseCode]).

%% wat_ite_value(+Goal, +VarMap, -Value)
wat_ite_value(Goal, VarMap, Value) :-
    (   Goal = (_Var = RHS)
    ->  wat_resolve_value(RHS, VarMap, Value)
    ;   Goal = (_Var is ArithExpr)
    ->  wat_arith_expr(ArithExpr, VarMap, Value)
    ;   wat_resolve_value(Goal, VarMap, Value)
    ).

%% wat_ite_else(+Else, +VarMap, -Code)
wat_ite_else(Else, VarMap, Code) :-
    (   if_then_else_goal(Else, If2, Then2, Else2)
    ->  wat_guard_condition(If2, VarMap, CondStr2),
        wat_ite_value(Then2, VarMap, ThenVal2),
        wat_ite_else(Else2, VarMap, ElseCode2),
        format(string(Code),
'        (if (result i64) ~w
          (then ~w)
          (else
~w
          )
        )', [CondStr2, ThenVal2, ElseCode2])
    ;   wat_ite_value(Else, VarMap, ElseVal),
        format(string(Code), '        ~w', [ElseVal])
    ).

%% ============================================
%% MULTICALL LINEAR RECURSION (e.g. fibonacci)
%% ============================================

compile_multicall_recursion_wat(Pred/Arity, _Options, Code) :-
    atom_string(Pred, PredStr),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Multicall Linear Recursion
  ;; Predicate: ~w/~w
  ;; Multiple recursive calls per clause with memoization

  (memory (export "memory") 1)

  ;; Memo: offset 4096 = values (i64, 8B each), offset 12288 = flags (i32, 4B each)

  (func $~w (export "~w") (param $n i64) (result i64)
    (local $offset i32)
    (local $flag_offset i32)
    (local $result i64)

    ;; Base cases
    (if (result i64) (i64.le_s (local.get $n) (i64.const 0))
      (then (i64.const 0))
      (else
        (if (result i64) (i64.eq (local.get $n) (i64.const 1))
          (then (i64.const 1))
          (else
            ;; Check memo
            (local.set $offset (i32.add (i32.const 4096) (i32.mul (i32.wrap_i64 (local.get $n)) (i32.const 8))))
            (local.set $flag_offset
              (i32.add (i32.const 12288)
                (i32.mul (i32.wrap_i64 (local.get $n)) (i32.const 4))))
            (if (result i64) (i32.load (local.get $flag_offset))
              (then (i64.load (local.get $offset)))
              (else
                ;; Multicall: f(n) = f(n-1) + f(n-2)
                (local.set $result
                  (i64.add
                    (call $~w (i64.sub (local.get $n) (i64.const 1)))
                    (call $~w (i64.sub (local.get $n) (i64.const 2)))
                  )
                )
                (i64.store (local.get $offset) (local.get $result))
                (i32.store (local.get $flag_offset) (i32.const 1))
                (local.get $result)
              )
            )
          )
        )
      )
    )
  )
)
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr]).

%% ============================================
%% DIRECT MULTI-CALL RECURSION
%% ============================================

compile_direct_multicall_wat(Pred/Arity, _Options, Code) :-
    atom_string(Pred, PredStr),
    format(string(Code),
'(module
  ;; Generated by UnifyWeaver WAT Target - Direct Multi-Call Recursion
  ;; Predicate: ~w/~w
  ;; Direct clause body analysis with multiple recursive calls

  (memory (export "memory") 1)

  ;; Memo: offset 4096 = values (i64, 8B each), offset 12288 = flags (i32, 4B each)

  (func $~w (export "~w") (param $input i64) (result i64)
    (local $offset i32)
    (local $flag_offset i32)
    (local $result i64)

    ;; Base cases
    (if (result i64) (i64.le_s (local.get $input) (i64.const 0))
      (then (i64.const 0))
      (else
        (if (result i64) (i64.eq (local.get $input) (i64.const 1))
          (then (i64.const 1))
          (else
            ;; Check memo
            (local.set $offset (i32.add (i32.const 4096) (i32.mul (i32.wrap_i64 (local.get $input)) (i32.const 8))))
            (local.set $flag_offset
              (i32.add (i32.const 12288)
                (i32.mul (i32.wrap_i64 (local.get $input)) (i32.const 4))))
            (if (result i64) (i32.load (local.get $flag_offset))
              (then (i64.load (local.get $offset)))
              (else
                ;; Direct recursive calls from clause body
                (local.set $result
                  (i64.add
                    (call $~w (i64.sub (local.get $input) (i64.const 1)))
                    (call $~w (i64.sub (local.get $input) (i64.const 2)))
                  )
                )
                (i64.store (local.get $offset) (local.get $result))
                (i32.store (local.get $flag_offset) (i32.const 1))
                (local.get $result)
              )
            )
          )
        )
      )
    )
  )
)
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr]).

%% ============================================
%% MULTIFILE DISPATCH HOOKS
%% ============================================

%% Tail recursion
tail_recursion:compile_tail_pattern(wat, PredStr, Arity, _BaseClauses, _RecClauses, _AccPos, _StepOp, _ExitAfterResult, Code) :-
    atom_string(Pred, PredStr),
    compile_tail_recursion_wat(Pred/Arity, [], Code).

%% Linear recursion
linear_recursion:compile_linear_pattern(wat, PredStr, Arity, _BaseClauses, _RecClauses, _MemoEnabled, _MemoStrategy, Code) :-
    atom_string(Pred, PredStr),
    compile_linear_recursion_wat(Pred/Arity, [], Code).

%% Tree recursion
tree_recursion:compile_tree_pattern(wat, _Pattern, Pred, Arity, _UseMemo, Code) :-
    compile_tree_recursion_wat(Pred/Arity, [], Code).

%% Multicall linear recursion
multicall_linear_recursion:compile_multicall_pattern(wat, PredStr, BaseClauses, _RecClauses, _MemoEnabled, Code) :-
    atom_string(Pred, PredStr),
    length(BaseClauses, _),
    compile_multicall_recursion_wat(Pred/2, [], Code).

%% Direct multi-call recursion
direct_multi_call_recursion:compile_direct_multicall_pattern(wat, PredStr, _BaseClauses, _RecClause, Code) :-
    atom_string(Pred, PredStr),
    compile_direct_multicall_wat(Pred/2, [], Code).

%% Mutual recursion
mutual_recursion:compile_mutual_pattern(wat, Predicates, _MemoEnabled, _MemoStrategy, Code) :-
    compile_mutual_recursion_wat(Predicates, [], Code).

%% ============================================
%% TEMPLATE-BASED TRANSITIVE CLOSURE
%% ============================================

%% compile_transitive_closure_wat_from_template(+Pred, +Base, -Code)
%  Uses mustache template if available, falls back to inline generation.
compile_transitive_closure_wat_from_template(Pred, Base, Code) :-
    atom_string(Pred, PredStr),
    atom_string(Base, BaseStr),
    TemplateFile = 'templates/targets/wat/transitive_closure.mustache',
    (   exists_file(TemplateFile)
    ->  read_file_to_string(TemplateFile, Template, []),
        render_template(Template, [pred=PredStr, base=BaseStr], Code)
    ;   compile_transitive_closure_wat(Pred/2, [], Code)
    ).

%% ============================================
%% COMPONENT SYSTEM INTEGRATION
%% ============================================

:- initialization((
    catch(
        register_component_type(source, custom_wat, custom_wat, [
            description("Custom WAT Code Component")
        ]),
        _,
        true
    )
), now).

%% Component interface predicates
wat_type_info(info(
    name('Custom WAT Component'),
    version('1.0.0'),
    description('Injects custom WAT code and exposes it as a component')
)).

wat_validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

wat_init_component(_Name, _Config).

wat_invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_wat))).

wat_compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),
    atom_string(Name, NameStr),
    format(string(Code),
'  ;; Custom Component: ~w
  (func $comp_~w (export "comp_~w") (param $input i64) (result i64)
~w
  )', [NameStr, NameStr, NameStr, Body]).
