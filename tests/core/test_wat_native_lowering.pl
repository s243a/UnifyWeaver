:- module(test_wat_native_lowering, [test_wat_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/wat_target').

test_wat_native_lowering :-
    run_tests([wat_native_lowering]).

:- begin_tests(wat_native_lowering).

% Helper: compile using the public API
compile_wat(Pred/Arity, Code) :-
    wat_target:compile_predicate_to_wat(Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → nested if/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_wat(classify/2, Code),
    has(Code, "(func $classify"),
    has(Code, "i64.gt_s"),
    has(Code, "i64.lt_s"),
    has(Code, "i32.and"),
    has(Code, "i64.ge_s"),
    has(Code, "(if (result i64)"),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_wat(positive/2, Code),
    has(Code, "(func $positive"),
    has(Code, "i64.gt_s"),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_wat(double/2, Code),
    has(Code, "(func $double"),
    has(Code, "i64.mul"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_wat(identity/2, Code),
    has(Code, "(func $identity"),
    has(Code, "local.get $arg1"),
    retractall(user:identity(_, _)).

test(multi_clause_rules) :-
    assert(user:(color2(X, warm) :- X == red)),
    assert(user:(color2(X, cool) :- X == blue)),
    assert(user:(color2(X, cool) :- X == green)),
    compile_wat(color2/2, Code),
    has(Code, "(func $color2"),
    has(Code, "i64.eq"),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_wat(abs_val/2, Code),
    has(Code, "(func $abs_val"),
    has(Code, "i64.ge_s"),
    has(Code, "i64.sub"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_wat(range_classify/2, Code),
    has(Code, "(func $range_classify"),
    has(Code, "i64.lt_s"),
    has(Code, "i64.eq"),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_wat(sign/2, Code),
    has(Code, "(func $sign"),
    has(Code, "i64.gt_s"),
    has(Code, "i64.lt_s"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_wat(safe_div/3, Code),
    has(Code, "(func $safe_div"),
    has(Code, "(param $arg1 i64)"),
    has(Code, "(param $arg2 i64)"),
    has(Code, "i64.gt_s"),
    has(Code, "i64.div_s"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% WAT-specific syntax
% ============================================================================

test(wat_uses_structured_if) :-
    assert(user:(grade(X, pass) :- X >= 50)),
    assert(user:(grade(X, fail) :- X < 50)),
    compile_wat(grade/2, Code),
    has(Code, "(if (result i64)"),
    has(Code, "(then"),
    has(Code, "(else"),
    retractall(user:grade(_, _)).

test(wat_uses_module_wrapper) :-
    assert(user:(only_pos(X, yes) :- X > 0)),
    compile_wat(only_pos/2, Code),
    has(Code, "(module"),
    has(Code, "(export"),
    retractall(user:only_pos(_, _)).

% ============================================================================
% Recursion patterns
% ============================================================================

test(tail_recursion_loop_br) :-
    wat_target:compile_tail_recursion_wat(sum/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $sum"),
    has(Code, "(loop $continue"),
    has(Code, "(br $continue)"),
    has(Code, "i64.add"),
    has(Code, "$sum_entry").

test(linear_recursion_memo) :-
    wat_target:compile_linear_recursion_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "(memory"),
    has(Code, "i64.store"),
    has(Code, "i64.load"),
    has(Code, "i32.store").

test(tree_recursion_two_calls) :-
    wat_target:compile_tree_recursion_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "(memory"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 1)))"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 2)))").

test(mutual_recursion_cross_call) :-
    wat_target:compile_mutual_recursion_wat([is_even/1, is_odd/1], [], Code),
    has(Code, "(module"),
    has(Code, "(func $is_even"),
    has(Code, "(func $is_odd"),
    has(Code, "(call $is_odd"),
    has(Code, "(call $is_even").

test(transitive_closure_bfs) :-
    wat_target:compile_transitive_closure_wat(reachable/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $reachable"),
    has(Code, "$add_edge"),
    has(Code, "(loop $bfs"),
    has(Code, "$edge_scan"),
    has(Code, "$mark_visited").

test(multicall_recursion_memo) :-
    wat_target:compile_multicall_recursion_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "Multicall"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 1)))"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 2)))"),
    has(Code, "i64.store"),
    has(Code, "i32.store").

test(direct_multicall_memo) :-
    wat_target:compile_direct_multicall_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "Direct Multi-Call"),
    has(Code, "(call $fib"),
    has(Code, "i64.store").

% ============================================================================
% Template integration
% ============================================================================

test(template_transitive_closure) :-
    wat_target:compile_transitive_closure_wat_from_template(ancestor, parent, Code),
    has(Code, "(module"),
    has(Code, "ancestor"),
    has(Code, "(loop $bfs").

% ============================================================================
% Multifile dispatch hooks
% ============================================================================

test(multifile_tail_hook) :-
    tail_recursion:compile_tail_pattern(wat, "sum", 2, [], [], 2, add, false, Code),
    has(Code, "(module"),
    has(Code, "(func $sum"),
    has(Code, "(loop $continue").

test(multifile_linear_hook) :-
    linear_recursion:compile_linear_pattern(wat, "fib", 2, [], [], true, table, Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "i64.load").

test(multifile_tree_hook) :-
    tree_recursion:compile_tree_pattern(wat, fibonacci, fib, 2, true, Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 1)))").

test(multifile_multicall_hook) :-
    multicall_linear_recursion:compile_multicall_pattern(wat, "fib", [base1], [], true, Code),
    has(Code, "(module"),
    has(Code, "(func $fib").

test(multifile_direct_multicall_hook) :-
    direct_multi_call_recursion:compile_direct_multicall_pattern(wat, "fib", [], clause(fib(n,f), true), Code),
    has(Code, "(module"),
    has(Code, "(func $fib").

test(multifile_mutual_hook) :-
    mutual_recursion:compile_mutual_pattern(wat, [is_even/1, is_odd/1], true, table, Code),
    has(Code, "(module"),
    has(Code, "(func $is_even"),
    has(Code, "(func $is_odd").

% ============================================================================
% Component system
% ============================================================================

test(component_compile) :-
    wat_target:wat_compile_component(test_comp,
        [code("    (local.get $input)")],
        [],
        Code),
    has(Code, "comp_test_comp"),
    has(Code, "(export"),
    has(Code, "local.get $input").

% ============================================================================
% Target registry
% ============================================================================

test(target_registered) :-
    use_module('src/unifyweaver/core/target_registry'),
    target_registry:registered_target(wat, lowlevel, Caps),
    once(member(structured_control_flow, Caps)).

% ============================================================================
% LLVM WASM fallback
% ============================================================================

test(llvm_fallback_on_unsupported) :-
    % The LLVM fallback should activate when native lowering fails
    % and LLVM is available
    wat_target:compile_wasm_module(
        [func(test_func, 2, linear_recursion)],
        [module_name(test_func)],
        LLVMCode
    ),
    has(LLVMCode, "wasm32").

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

% ============================================================================
% String table support
% ============================================================================

test(string_table_build) :-
    wat_target:wat_build_string_table([hello, world, hello], Table),
    length(Table, 2),  % deduplicated
    once(member(str_entry(hello, _, 5), Table)),
    once(member(str_entry(world, _, 5), Table)).

test(string_table_offsets) :-
    wat_target:wat_build_string_table([abc, de], Table),
    Table = [str_entry(abc, 0, 3), str_entry(de, 3, 2)].

test(string_data_segments) :-
    wat_target:wat_build_string_table([hi], Table),
    wat_target:wat_string_data_segments(Table, Segs),
    has(Segs, "(data (i32.const 0)"),
    has(Segs, "\"hi\"").

test(string_literal_lookup) :-
    wat_target:wat_build_string_table([red, blue], Table),
    wat_target:wat_string_literal(red, Table, Str),
    has(Str, "i64.const").

test(string_eq_func) :-
    wat_target:wat_string_eq_func(Code),
    has(Code, "(func $str_eq"),
    has(Code, "i32.load8_u"),
    has(Code, "(loop $cmp").

test(string_lookup_func) :-
    wat_target:wat_build_string_table([yes, no], Table),
    wat_target:wat_string_lookup_func(Table, Code),
    has(Code, "(func $str_lookup"),
    has(Code, "\"yes\""),
    has(Code, "\"no\"").

test(compile_with_strings) :-
    assert(user:(is_ok(X, ok) :- X > 0)),
    wat_target:wat_compile_with_strings(is_ok/2, [], [ok], Code),
    has(Code, "(module"),
    has(Code, "(data"),
    has(Code, "(func $str_eq"),
    has(Code, "(func $str_lookup"),
    has(Code, "(memory"),
    retractall(user:is_ok(_, _)).

test(string_empty_table) :-
    once(wat_target:wat_string_data_segments([], Segs)),
    Segs == "".

% ============================================================================
% Memory layout constants
% ============================================================================

test(mem_regions_no_overlap) :-
    wat_target:wat_mem_region(string_data, S),
    wat_target:wat_mem_region(memo_values, MV),
    wat_target:wat_mem_region(memo_flags, MF),
    wat_target:wat_mem_region(graph_edges, GE),
    wat_target:wat_mem_region(visited_flags, VF),
    wat_target:wat_mem_region(bfs_queue, BQ),
    wat_target:wat_mem_region(fact_data, FD),
    S < MV, MV < MF, MF < GE, GE < VF, VF < BQ, BQ < FD,
    FD < 65536.  % fits in 1 page

test(memo_uses_region_offsets) :-
    wat_target:compile_linear_recursion_wat(fib/2, [], Code),
    has(Code, "i32.const 4096"),   % memo values base
    has(Code, "i32.const 12288").  % memo flags base

test(transitive_closure_uses_region_offsets) :-
    wat_target:compile_transitive_closure_wat(reachable/2, [], Code),
    has(Code, "i32.const 16384"),  % graph edges base
    has(Code, "i32.const 32768"),  % visited flags base
    has(Code, "i32.const 36864"). % bfs queue base

test(multicall_uses_region_offsets) :-
    wat_target:compile_multicall_recursion_wat(fib/2, [], Code),
    has(Code, "i32.const 4096"),
    has(Code, "i32.const 12288").

test(direct_multicall_uses_region_offsets) :-
    wat_target:compile_direct_multicall_wat(fib/2, [], Code),
    has(Code, "i32.const 4096"),
    has(Code, "i32.const 12288").

test(tree_recursion_uses_region_offsets) :-
    wat_target:compile_tree_recursion_wat(fib/2, [], Code),
    has(Code, "i32.const 4096"),
    has(Code, "i32.const 12288").

% ============================================================================
% Classify with string table (string output via hash)
% ============================================================================

test(classify_string_table) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    wat_target:wat_compile_with_strings(classify/2, [], [small, large], Code),
    has(Code, "(func $classify"),
    has(Code, "(data"),
    has(Code, "\"small\""),
    has(Code, "\"large\""),
    has(Code, "(func $str_lookup"),
    retractall(user:classify(_, _)).

% ============================================================================
% String ref output (direct string references instead of hashes)
% ============================================================================

test(string_ref_classify) :-
    assert(user:(classify2(X, small) :- X > 0, X < 10)),
    assert(user:(classify2(X, large) :- X >= 10)),
    wat_target:wat_compile_with_string_refs(classify2/2, [], [small, large], Code),
    has(Code, "(func $classify2"),
    has(Code, "(data"),
    has(Code, "\"small\""),
    has(Code, "\"large\""),
    % Should NOT have str_lookup (refs are direct, no hash-to-ref needed)
    \+ has(Code, "(func $str_lookup"),
    % Should have str_eq for comparing refs
    has(Code, "(func $str_eq"),
    retractall(user:classify2(_, _)).

test(string_ref_no_hash_in_output) :-
    assert(user:(color(X, red) :- X =:= 1)),
    assert(user:(color(X, blue) :- X =:= 2)),
    wat_target:wat_compile_with_string_refs(color/2, [], [red, blue], Code),
    % Data segments should have the actual string text
    has(Code, "\"red\""),
    has(Code, "\"blue\""),
    % Should have string ref i64 values (not hash values)
    has(Code, "(func $color"),
    has(Code, "(data"),
    retractall(user:color(_, _)).

% ============================================================================
% Multi-page memory
% ============================================================================

test(memory_default_one_page) :-
    wat_target:wat_memory_pages([], Pages),
    Pages == 1.

test(memory_explicit_pages) :-
    wat_target:wat_memory_pages([pages(4)], Pages),
    Pages == 4.

test(memory_extra_pages) :-
    wat_target:wat_memory_pages([extra_pages(2)], Pages),
    Pages == 3.  % 1 base + 2 extra

test(memory_large_facts) :-
    wat_target:wat_memory_pages([max_facts(300)], Pages),
    Pages > 1.  % 300 * 256 = 76800, exceeds 1 page

test(memory_decl_multipage) :-
    wat_target:wat_memory_decl([pages(3)], Decl),
    has(Decl, "(memory"),
    has(Decl, "3").

test(string_ref_uses_memory_decl) :-
    assert(user:(sz(X, big) :- X > 100)),
    wat_target:wat_compile_with_string_refs(sz/2, [pages(2)], [big], Code),
    has(Code, "(memory (export \"memory\") 2"),
    retractall(user:sz(_, _)).

% ============================================================================
% Multi-module import/export linking
% ============================================================================

test(import_single_func) :-
    wat_target:wat_module_imports(
        [import("math", "fib", [i64], i64)],
        [],
        Code),
    has(Code, "(import \"math\" \"fib\""),
    has(Code, "(param i64)"),
    has(Code, "(result i64)").

test(import_multi_param) :-
    wat_target:wat_module_imports(
        [import("util", "add", [i64, i64], i64)],
        [],
        Code),
    has(Code, "(import \"util\" \"add\""),
    has(Code, "(param i64)").

test(import_void_result) :-
    wat_target:wat_module_imports(
        [import("io", "print", [i32], void)],
        [],
        Code),
    has(Code, "(import \"io\" \"print\""),
    \+ has(Code, "(result").

test(export_func) :-
    wat_target:wat_module_exports(
        [export("add", "$add_impl")],
        [],
        Code),
    has(Code, "(export \"add\" (func $add_impl))").

test(export_memory) :-
    wat_target:wat_module_exports(
        [export_memory("mem")],
        [],
        Code),
    has(Code, "(export \"mem\" (memory 0))").

test(export_global) :-
    wat_target:wat_module_exports(
        [export_global("count", "$edge_count")],
        [],
        Code),
    has(Code, "(export \"count\" (global $edge_count))").

test(link_module) :-
    wat_target:wat_link_modules(
        module("app",
            [import("math", "fib", [i64], i64)],
            [func("run", [param(n, i64)], i64,
                "    (call $fib (local.get $n))")],
            []),
        [],
        Code),
    has(Code, "(module"),
    has(Code, "Module: app"),
    has(Code, "(import \"math\" \"fib\""),
    has(Code, "(func $run (export \"run\")"),
    has(Code, "(call $fib").

test(link_module_with_exports) :-
    wat_target:wat_link_modules(
        module("lib",
            [],
            [func("double", [param(x, i64)], i64,
                "    (i64.mul (local.get $x) (i64.const 2))")],
            [export_memory("memory")]),
        [pages(2)],
        Code),
    has(Code, "(memory (export \"memory\") 2"),
    has(Code, "(func $double"),
    has(Code, "(export \"memory\" (memory 0))").

test(link_empty_imports) :-
    wat_target:wat_module_imports([], [], Code),
    Code == "".

test(link_empty_exports) :-
    wat_target:wat_module_exports([], [], Code),
    Code == "".

% ============================================================================
% Indirect call tables (dynamic dispatch)
% ============================================================================

test(call_table_basic) :-
    wat_target:wat_call_table(
        [table_func(double, [i64], i64,
            "    (i64.mul (local.get $p1) (i64.const 2))"),
         table_func(triple, [i64], i64,
            "    (i64.mul (local.get $p1) (i64.const 3))")],
        [],
        Code),
    has(Code, "(type $dispatch_t (func"),
    has(Code, "(table 2 funcref)"),
    has(Code, "(elem (i32.const 0) $double $triple)"),
    has(Code, "(func $double"),
    has(Code, "(func $triple").

test(dispatch_func) :-
    wat_target:wat_dispatch_func(
        "apply",
        type_sig([i64], i64),
        [],
        Code),
    has(Code, "(func $apply (export \"apply\")"),
    has(Code, "(param $idx i32)"),
    has(Code, "call_indirect (type $dispatch_t)").

test(compile_with_dispatch) :-
    wat_target:wat_compile_with_dispatch(
        dispatch("compute", [i64], i64),
        [],
        [table_func(inc, [i64], i64,
            "    (i64.add (local.get $p1) (i64.const 1))"),
         table_func(dec, [i64], i64,
            "    (i64.sub (local.get $p1) (i64.const 1))"),
         table_func(dbl, [i64], i64,
            "    (i64.mul (local.get $p1) (i64.const 2))")],
        Code),
    has(Code, "(module"),
    has(Code, "Dynamic Dispatch"),
    has(Code, "(table 3 funcref)"),
    has(Code, "(elem (i32.const 0) $inc $dec $dbl)"),
    has(Code, "(func $compute (export \"compute\")"),
    has(Code, "call_indirect").

test(call_table_empty) :-
    wat_target:wat_call_table([], [], Code),
    Code == "".

test(dispatch_multi_param) :-
    wat_target:wat_dispatch_func(
        "binop",
        type_sig([i64, i64], i64),
        [],
        Code),
    has(Code, "(param $idx i32)"),
    has(Code, "(param $p1 i64)"),
    has(Code, "(param $p2 i64)"),
    has(Code, "(local.get $p1)"),
    has(Code, "(local.get $p2)").

% ============================================================================
% WASI I/O
% ============================================================================

test(wasi_imports) :-
    wat_target:wat_wasi_imports(Code),
    has(Code, "(import \"wasi_snapshot_preview1\" \"fd_write\""),
    has(Code, "(func $fd_write").

test(wasi_print_i64) :-
    wat_target:wat_wasi_print_i64_func(Code),
    has(Code, "(func $print_i64 (export \"print_i64\")"),
    has(Code, "(param $val i64)"),
    has(Code, "i64.rem_u"),
    has(Code, "call $fd_write").

test(wasi_print_str) :-
    wat_target:wat_wasi_print_str_func(Code),
    has(Code, "(func $print_str (export \"print_str\")"),
    has(Code, "(param $ref i64)"),
    has(Code, "i64.shr_u"),
    has(Code, "call $fd_write").

test(compile_with_wasi) :-
    assert(user:(double(X, Y) :- Y is X * 2)),
    wat_target:wat_compile_with_wasi(double/2, [], Code),
    has(Code, "(import \"wasi_snapshot_preview1\""),
    has(Code, "(func $print_i64"),
    has(Code, "(func $print_str"),
    has(Code, "(func $double"),
    retractall(user:double(_, _)).

% ============================================================================
% SIMD helpers
% ============================================================================

test(simd_dot_product) :-
    wat_target:wat_simd_dot_product_func(Code),
    has(Code, "(func $simd_dot_i64x2"),
    has(Code, "(param $a v128)"),
    has(Code, "i64x2.mul"),
    has(Code, "i64x2.extract_lane 0"),
    has(Code, "i64x2.extract_lane 1").

test(simd_sum_reduce) :-
    wat_target:wat_simd_sum_reduce_func(Code),
    has(Code, "(func $simd_sum_i64x2"),
    has(Code, "i64x2.extract_lane 0"),
    has(Code, "i64.add").

% ============================================================================
% Bulk memory helpers
% ============================================================================

test(bulk_memcpy) :-
    wat_target:wat_bulk_memcpy_func(Code),
    has(Code, "(func $memcpy (export \"memcpy\")"),
    has(Code, "memory.copy").

test(bulk_memset) :-
    wat_target:wat_bulk_memset_func(Code),
    has(Code, "(func $memset (export \"memset\")"),
    has(Code, "memory.fill").

% ============================================================================
% Bindings registration
% ============================================================================

test(simd_bindings_registered) :-
    wat_target:init_wat_target,
    binding_registry:binding(wat, 'v128_i64x2_add'/3, 'i64x2.add', _, _, _).

test(bulk_memory_bindings_registered) :-
    wat_target:init_wat_target,
    binding_registry:binding(wat, 'memory_copy'/4, 'memory.copy', _, _, _).

% ============================================================================
% Feature flags
% ============================================================================

test(feature_enable_disable) :-
    wat_target:wat_disable_feature(exceptions),
    \+ wat_target:wat_feature_enabled(exceptions),
    wat_target:wat_enable_feature(exceptions),
    wat_target:wat_feature_enabled(exceptions),
    wat_target:wat_disable_feature(exceptions).

test(feature_gc_enables_reference_types) :-
    wat_target:wat_disable_feature(gc),
    wat_target:wat_disable_feature(reference_types),
    wat_target:wat_enable_feature(gc),
    wat_target:wat_feature_enabled(gc),
    wat_target:wat_feature_enabled(reference_types),
    wat_target:wat_disable_feature(gc),
    wat_target:wat_disable_feature(reference_types).

test(feature_list) :-
    wat_target:wat_disable_feature(simd),
    wat_target:wat_enable_feature(simd),
    wat_target:wat_list_features(Fs),
    once(member(simd, Fs)),
    wat_target:wat_disable_feature(simd).

test(feature_guard_blocks_without_enable) :-
    wat_target:wat_disable_feature(exceptions),
    catch(
        wat_target:wat_exception_tag(e, [i32], _),
        error(feature_not_enabled(exceptions), _),
        true
    ).

test(feature_wabt_flags) :-
    wat_target:wat_disable_feature(exceptions),
    wat_target:wat_enable_feature(exceptions),
    wat_target:wat_feature_flags_for_tool(wabt, Flags),
    has(Flags, "--enable-exceptions"),
    wat_target:wat_disable_feature(exceptions).

test(feature_wasmtime_flags) :-
    wat_target:wat_enable_feature(gc),
    wat_target:wat_feature_flags_for_tool(wasmtime, Flags),
    has(Flags, "-W gc=y"),
    wat_target:wat_disable_feature(gc),
    wat_target:wat_disable_feature(reference_types).

% ============================================================================
% Exception handling
% ============================================================================

test(exception_tag) :-
    wat_target:wat_enable_feature(exceptions),
    wat_target:wat_exception_tag(err, [i32], Code),
    has(Code, "(tag $err"),
    has(Code, "(param i32)"),
    wat_target:wat_disable_feature(exceptions).

test(try_catch) :-
    wat_target:wat_enable_feature(exceptions),
    wat_target:wat_try_catch(
        "        (throw $err (i32.const 99))",
        err,
        "    ;; caught value on stack",
        i32,
        Code),
    has(Code, "try_table"),
    has(Code, "catch $err"),
    has(Code, "(result i32)"),
    wat_target:wat_disable_feature(exceptions).

% ============================================================================
% GC / Reference types
% ============================================================================

test(gc_struct_type) :-
    wat_target:wat_enable_feature(gc),
    wat_target:wat_gc_struct_type(point,
        [field(x, i32), field(y, i32)], Code),
    has(Code, "(type $point (struct"),
    has(Code, "(field $x i32)"),
    has(Code, "(field $y i32)"),
    wat_target:wat_disable_feature(gc),
    wat_target:wat_disable_feature(reference_types).

test(gc_struct_type_mutable) :-
    wat_target:wat_enable_feature(gc),
    wat_target:wat_gc_struct_type(cell,
        [field(val, i64, mutable)], Code),
    has(Code, "(field $val (mut i64))"),
    wat_target:wat_disable_feature(gc),
    wat_target:wat_disable_feature(reference_types).

test(gc_array_type) :-
    wat_target:wat_enable_feature(gc),
    wat_target:wat_gc_array_type(int_array, i32, Code),
    has(Code, "(type $int_array (array i32))"),
    wat_target:wat_disable_feature(gc),
    wat_target:wat_disable_feature(reference_types).

test(gc_struct_new) :-
    wat_target:wat_gc_struct_new(point,
        ["(i32.const 3)", "(i32.const 7)"], Code),
    has(Code, "(struct.new $point"),
    has(Code, "(i32.const 3)"),
    has(Code, "(i32.const 7)").

test(gc_struct_get) :-
    wat_target:wat_gc_struct_get(point, x, "(local.get $p)", Code),
    has(Code, "(struct.get $point $x"),
    has(Code, "(local.get $p)").

test(gc_struct_set) :-
    wat_target:wat_gc_struct_set(cell, val,
        "(local.get $c)", "(i64.const 42)", Code),
    has(Code, "(struct.set $cell $val"),
    has(Code, "(local.get $c)"),
    has(Code, "(i64.const 42)").

test(gc_array_new) :-
    wat_target:wat_gc_array_new(int_array,
        "(i32.const 0)", "(i32.const 10)", Code),
    has(Code, "(array.new $int_array").

test(gc_array_get) :-
    wat_target:wat_gc_array_get(int_array,
        "(local.get $arr)", "(i32.const 5)", Code),
    has(Code, "(array.get $int_array").

test(gc_compile_module) :-
    wat_target:wat_enable_feature(gc),
    wat_target:wat_gc_compile_module(
        gc_module("points",
            [struct_type(point, [field(x, i32), field(y, i32)])],
            [func("sum_xy", [param(p, '(ref $point)')], i32,
                "    (i32.add (struct.get $point $x (local.get $p)) (struct.get $point $y (local.get $p)))")]),
        [],
        Code),
    has(Code, "(module"),
    has(Code, "(type $point (struct"),
    has(Code, "(func $sum_xy"),
    has(Code, "struct.get $point $x"),
    wat_target:wat_disable_feature(gc),
    wat_target:wat_disable_feature(reference_types).

test(gc_guard_blocks_without_enable) :-
    wat_target:wat_disable_feature(gc),
    catch(
        wat_target:wat_gc_struct_type(point, [field(x, i32)], _),
        error(feature_not_enabled(gc), _),
        true
    ).

:- end_tests(wat_native_lowering).
