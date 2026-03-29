:- module(test_jvm_asm_native_lowering, [test_jvm_asm_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/jamaica_target', [
    compile_predicate_to_jamaica/3,
    compile_jamaica_class/4,
    compile_tail_recursion_jamaica/3,
    compile_linear_recursion_jamaica/3,
    compile_tree_recursion_jamaica/3,
    compile_mutual_recursion_jamaica/2,
    init_jamaica_target/0
]).
:- use_module('../../src/unifyweaver/targets/krakatau_target', [
    compile_predicate_to_krakatau/3,
    compile_krakatau_class/4,
    compile_tail_recursion_krakatau/3,
    compile_linear_recursion_krakatau/3,
    compile_tree_recursion_krakatau/3,
    compile_mutual_recursion_krakatau/2,
    init_krakatau_target/0
]).
:- use_module('../../src/unifyweaver/core/jvm_bytecode').
:- use_module('../../src/unifyweaver/core/target_registry').
:- use_module('../../src/unifyweaver/core/recursive_compiler').
:- use_module('../../src/unifyweaver/core/input_source').
:- use_module('../../src/unifyweaver/glue/jvm_glue').

test_jvm_asm_native_lowering :-
    run_tests([jvm_asm_native_lowering]).

:- begin_tests(jvm_asm_native_lowering).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Shared JVM bytecode infrastructure
% ============================================================================

test(jvm_load_const_small) :-
    jvm_bytecode:jvm_load_const(3, Instr),
    has(Instr, "iconst_3").

test(jvm_load_const_bipush) :-
    jvm_bytecode:jvm_load_const(42, Instr),
    has(Instr, "bipush 42").

test(jvm_load_const_sipush) :-
    jvm_bytecode:jvm_load_const(1000, Instr),
    has(Instr, "sipush 1000").

test(jvm_load_const_ldc) :-
    jvm_bytecode:jvm_load_const(100000, Instr),
    has(Instr, "ldc 100000").

test(jvm_load_const_string) :-
    jvm_bytecode:jvm_load_const(hello, Instr),
    has(Instr, "ldc \"hello\"").

test(jvm_type_descriptor_int) :-
    jvm_bytecode:jvm_type_descriptor(int, "I").

test(jvm_type_descriptor_string) :-
    jvm_bytecode:jvm_type_descriptor('String', "Ljava/lang/String;").

test(jvm_method_descriptor) :-
    jvm_bytecode:jvm_method_descriptor([int, int], int, Desc),
    Desc == "(II)I".

test(jvm_expr_addition) :-
    jvm_bytecode:jvm_expr_to_bytecode(3 + 4, [], symbolic, Instrs),
    length(Instrs, 3),  % const 3, const 4, iadd
    last(Instrs, Last),
    has(Last, "iadd").

test(jvm_expr_nested) :-
    jvm_bytecode:jvm_expr_to_bytecode((2 * 3) + 1, [], symbolic, Instrs),
    length(Instrs, 5).  % const 2, const 3, imul, const 1, iadd

test(jvm_guard_greater) :-
    jvm_bytecode:jvm_guard_to_bytecode(3 > 2, [], symbolic, "FAIL", Instrs),
    last(Instrs, Branch),
    has(Branch, "if_icmple FAIL").

test(jvm_guard_eq) :-
    jvm_bytecode:jvm_guard_to_bytecode(1 =:= 1, [], symbolic, "L1", Instrs),
    last(Instrs, Branch),
    has(Branch, "if_icmpne L1").

test(jvm_stack_estimate) :-
    jvm_bytecode:jvm_estimate_stack_depth(
        ["    iconst_1", "    iconst_2", "    iadd", "    ireturn"],
        Depth),
    Depth == 2.

% ============================================================================
% Jamaica target
% ============================================================================

test(jamaica_multi_clause) :-
    assert(user:(classify_j(X, small) :- X > 0, X < 10)),
    assert(user:(classify_j(X, large) :- X >= 10)),
    jamaica_target:compile_predicate_to_jamaica(classify_j/2, [], Code),
    has(Code, "public class Prolog_classify_j"),
    has(Code, "public static int classify_j(int arg1)"),
    has(Code, "iload"),
    has(Code, "ireturn"),
    retractall(user:classify_j(_, _)).

test(jamaica_arithmetic) :-
    assert(user:(double_j(X, Y) :- Y is X * 2)),
    jamaica_target:compile_predicate_to_jamaica(double_j/2, [], Code),
    has(Code, "public static int double_j(int arg1)"),
    has(Code, "imul"),
    retractall(user:double_j(_, _)).

test(jamaica_java_like_syntax) :-
    assert(user:(inc_j(X, Y) :- Y is X + 1)),
    jamaica_target:compile_predicate_to_jamaica(inc_j/2, [], Code),
    % Jamaica uses Java-like class structure
    has(Code, "public class"),
    has(Code, "%default_constructor public"),
    has(Code, "public static int"),
    retractall(user:inc_j(_, _)).

test(jamaica_symbolic_vars) :-
    assert(user:(add_j(X, Y, Z) :- Z is X + Y)),
    jamaica_target:compile_predicate_to_jamaica(add_j/3, [], Code),
    % Jamaica uses symbolic variable names
    has(Code, "iload arg"),
    retractall(user:add_j(_, _, _)).

test(jamaica_class_compilation) :-
    jamaica_target:compile_jamaica_class("MathUtils",
        [method("square", [param(x, int)], int,
            "    iload x\n    iload x\n    imul\n    ireturn")],
        [],
        Code),
    has(Code, "public class MathUtils"),
    has(Code, "public static int square(int x)"),
    has(Code, "iload x"),
    has(Code, "imul").

% ============================================================================
% Krakatau target
% ============================================================================

test(krakatau_multi_clause) :-
    assert(user:(classify_k(X, small) :- X > 0, X < 10)),
    assert(user:(classify_k(X, large) :- X >= 10)),
    krakatau_target:compile_predicate_to_krakatau(classify_k/2, [], Code),
    has(Code, ".class public Prolog_classify_k"),
    has(Code, ".super java/lang/Object"),
    has(Code, ".method public static classify_k"),
    has(Code, ".limit stack"),
    has(Code, ".limit locals"),
    has(Code, "ireturn"),
    has(Code, ".end method"),
    retractall(user:classify_k(_, _)).

test(krakatau_arithmetic) :-
    assert(user:(double_k(X, Y) :- Y is X * 2)),
    krakatau_target:compile_predicate_to_krakatau(double_k/2, [], Code),
    has(Code, ".method public static double_k"),
    has(Code, "imul"),
    retractall(user:double_k(_, _)).

test(krakatau_numeric_slots) :-
    assert(user:(add_k(X, Y, Z) :- Z is X + Y)),
    krakatau_target:compile_predicate_to_krakatau(add_k/3, [], Code),
    % Krakatau uses numeric slot references
    has(Code, "iload"),
    has(Code, "(II)I"),
    retractall(user:add_k(_, _, _)).

test(krakatau_directive_syntax) :-
    assert(user:(id_k(X, X) :- X > 0)),
    krakatau_target:compile_predicate_to_krakatau(id_k/2, [], Code),
    % Krakatau uses directive syntax
    has(Code, ".version 52 0"),
    has(Code, ".class"),
    has(Code, ".super"),
    has(Code, ".method"),
    has(Code, ".end method"),
    retractall(user:id_k(_, _)).

test(krakatau_method_descriptor) :-
    assert(user:(triple_k(X, Y) :- Y is X * 3)),
    krakatau_target:compile_predicate_to_krakatau(triple_k/2, [], Code),
    has(Code, "(I)I"),  % one int param, int return
    retractall(user:triple_k(_, _)).

test(krakatau_class_compilation) :-
    krakatau_target:compile_krakatau_class("MathLib",
        [method("abs_val", [int], int,
            "    iload_0\n    invokestatic java/lang/Math abs (I)I\n    ireturn")],
        [],
        Code),
    has(Code, ".class public MathLib"),
    has(Code, ".method public static abs_val : (I)I"),
    has(Code, "invokestatic java/lang/Math abs").

% ============================================================================
% Shared codegen: same predicate, both targets
% ============================================================================

test(same_predicate_both_targets) :-
    assert(user:(double_shared(X, Y) :- Y is X * 2)),
    jamaica_target:compile_predicate_to_jamaica(double_shared/2, [], JaCode),
    krakatau_target:compile_predicate_to_krakatau(double_shared/2, [], KrCode),
    % Both should contain the same bytecodes
    has(JaCode, "imul"),
    has(KrCode, "imul"),
    % But different wrapping syntax
    has(JaCode, "public class"),
    has(KrCode, ".class public"),
    retractall(user:double_shared(_, _)).

% ============================================================================
% Target registry
% ============================================================================

test(jamaica_registered) :-
    target_registry:registered_target(jamaica, jvm, Caps),
    once(member(assembly, Caps)),
    once(member(macros, Caps)),
    once(member(recursion, Caps)),
    once(member(bindings, Caps)),
    once(member(components, Caps)).

test(krakatau_registered) :-
    target_registry:registered_target(krakatau, jvm, Caps),
    once(member(assembly, Caps)),
    once(member(roundtrip, Caps)),
    once(member(recursion, Caps)),
    once(member(bindings, Caps)),
    once(member(components, Caps)).

test(jamaica_module_linked) :-
    target_registry:target_module(jamaica, jamaica_target).

test(krakatau_module_linked) :-
    target_registry:target_module(krakatau, krakatau_target).

% ============================================================================
% Recursion bytecode generators (shared layer)
% ============================================================================

test(jvm_tail_recursion_symbolic) :-
    jvm_bytecode:jvm_tail_recursion_bytecode("sum", 2, symbolic, Instrs),
    once(member("    goto LOOP", Instrs)),
    once(member("LOOP:", Instrs)),
    once(member("DONE:", Instrs)),
    once(member("    iadd", Instrs)),
    once(member("    ireturn", Instrs)).

test(jvm_tail_recursion_numeric) :-
    jvm_bytecode:jvm_tail_recursion_bytecode("sum", 2, numeric, Instrs),
    once(member("    goto LOOP", Instrs)),
    once(member("    iload 0", Instrs)).

test(jvm_linear_recursion) :-
    jvm_bytecode:jvm_linear_recursion_bytecode("lin_sum", 2, symbolic, Instrs),
    once(member("    invokestatic lin_sum(I)I", Instrs)),
    once(member("    iadd", Instrs)).

test(jvm_tree_recursion) :-
    jvm_bytecode:jvm_tree_recursion_bytecode("fib", 2, symbolic, Instrs),
    % Tree recursion should have TWO recursive calls
    include(sub_string_match("invokestatic fib(I)I"), Instrs, Calls),
    length(Calls, 2).

test(jvm_mutual_recursion) :-
    jvm_bytecode:jvm_mutual_recursion_bytecode(
        [is_even, is_odd], "Prolog_is_even", symbolic, Methods),
    length(Methods, 2),
    Methods = [method("is_even", _), method("is_odd", _)].

test(jvm_entry_method) :-
    jvm_bytecode:jvm_entry_method_bytecode("sum", 2, symbolic, Instrs),
    once(member("    iconst_0", Instrs)),
    once(member("    invokestatic sum(II)I", Instrs)).

% ============================================================================
% Jamaica recursion patterns
% ============================================================================

test(jamaica_tail_recursion) :-
    jamaica_target:compile_tail_recursion_jamaica(sum/2, [], Code),
    has(Code, "public class Prolog_sum"),
    has(Code, "goto LOOP"),
    has(Code, "sum_entry").

test(jamaica_linear_recursion) :-
    jamaica_target:compile_linear_recursion_jamaica(lin_sum/2, [], Code),
    has(Code, "public class Prolog_lin_sum"),
    has(Code, "invokestatic lin_sum").

test(jamaica_tree_recursion) :-
    jamaica_target:compile_tree_recursion_jamaica(fib/2, [], Code),
    has(Code, "public class Prolog_fib"),
    has(Code, "invokestatic fib").

test(jamaica_mutual_recursion) :-
    jamaica_target:compile_mutual_recursion_jamaica([is_even, is_odd], [], Code),
    has(Code, "public class Prolog_is_even"),
    has(Code, "public static int is_even"),
    has(Code, "public static int is_odd").

% ============================================================================
% Krakatau recursion patterns
% ============================================================================

test(krakatau_tail_recursion) :-
    krakatau_target:compile_tail_recursion_krakatau(sum/2, [], Code),
    has(Code, ".class public Prolog_sum"),
    has(Code, "goto LOOP"),
    has(Code, ".end method"),
    has(Code, "sum_entry").

test(krakatau_linear_recursion) :-
    krakatau_target:compile_linear_recursion_krakatau(lin_sum/2, [], Code),
    has(Code, ".class public Prolog_lin_sum"),
    has(Code, "invokestatic lin_sum").

test(krakatau_tree_recursion) :-
    krakatau_target:compile_tree_recursion_krakatau(fib/2, [], Code),
    has(Code, ".class public Prolog_fib"),
    has(Code, "invokestatic fib").

test(krakatau_mutual_recursion) :-
    krakatau_target:compile_mutual_recursion_krakatau([is_even, is_odd], [], Code),
    has(Code, ".class public Prolog_is_even"),
    has(Code, ".method public static is_even"),
    has(Code, ".method public static is_odd").

% ============================================================================
% Bindings
% ============================================================================

test(jamaica_arithmetic_binding) :-
    jamaica_target:init_jamaica_target,
    once(binding_registry:binding(jamaica, '+'/3, 'iadd', _, _, _)).

test(krakatau_arithmetic_binding) :-
    krakatau_target:init_krakatau_target,
    once(binding_registry:binding(krakatau, '+'/3, 'iadd', _, _, _)).

test(jamaica_math_binding) :-
    once(binding_registry:binding(jamaica, 'abs'/2, 'invokestatic java/lang/Math abs (I)I', _, _, _)).

test(krakatau_comparison_binding) :-
    once(binding_registry:binding(krakatau, '>'/2, 'if_icmpgt', _, _, _)).

test(dual_binding_both_registered) :-
    once(binding_registry:binding(jamaica, 'max'/3, _, _, _, _)),
    once(binding_registry:binding(krakatau, 'max'/3, _, _, _, _)).

% ============================================================================
% Cross-target: same recursion pattern, both syntaxes
% ============================================================================

test(tail_recursion_same_bytecodes) :-
    jamaica_target:compile_tail_recursion_jamaica(sum/2, [], JaCode),
    krakatau_target:compile_tail_recursion_krakatau(sum/2, [], KrCode),
    % Both contain the same core bytecodes
    has(JaCode, "goto LOOP"),
    has(KrCode, "goto LOOP"),
    has(JaCode, "iadd"),
    has(KrCode, "iadd"),
    % Different syntax wrapping
    has(JaCode, "public class"),
    has(KrCode, ".class public").

% ============================================================================
% Expanded native lowering tests — Jamaica
% ============================================================================

test(jamaica_three_clause_classify) :-
    assert(user:(grade_j(X, low) :- X < 50)),
    assert(user:(grade_j(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade_j(X, high) :- X >= 80)),
    jamaica_target:compile_predicate_to_jamaica(grade_j/2, [], Code),
    has(Code, "if_icmp"),
    has(Code, "ireturn"),
    retractall(user:grade_j(_, _)).

test(jamaica_arithmetic_guard) :-
    assert(user:(even_check_j(X, yes) :- 0 =:= X mod 2)),
    assert(user:(even_check_j(X, no) :- 0 =\= X mod 2)),
    jamaica_target:compile_predicate_to_jamaica(even_check_j/2, [], Code),
    has(Code, "irem"),
    has(Code, "if_icmp"),
    retractall(user:even_check_j(_, _)).

test(jamaica_subtraction_output) :-
    assert(user:(negate_j(X, Y) :- Y is 0 - X)),
    jamaica_target:compile_predicate_to_jamaica(negate_j/2, [], Code),
    has(Code, "isub"),
    has(Code, "iconst_0"),
    retractall(user:negate_j(_, _)).

test(jamaica_complex_arithmetic) :-
    assert(user:(formula_j(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    jamaica_target:compile_predicate_to_jamaica(formula_j/2, [], Code),
    has(Code, "imul"),
    has(Code, "iadd"),
    retractall(user:formula_j(_, _)).

test(jamaica_division_guard) :-
    assert(user:(divisible_j(X, Y, yes) :- 0 =:= X mod Y)),
    assert(user:(divisible_j(X, Y, no) :- 0 =\= X mod Y)),
    jamaica_target:compile_predicate_to_jamaica(divisible_j/3, [], Code),
    has(Code, "irem"),
    has(Code, "int arg1, int arg2"),
    retractall(user:divisible_j(_, _, _)).

test(jamaica_single_guard_return) :-
    assert(user:(positive_j(X, 1) :- X > 0)),
    assert(user:(positive_j(X, 0) :- X =< 0)),
    jamaica_target:compile_predicate_to_jamaica(positive_j/2, [], Code),
    has(Code, "if_icmp"),
    has(Code, "ireturn"),
    retractall(user:positive_j(_, _)).

% ============================================================================
% Expanded native lowering tests — Krakatau
% ============================================================================

test(krakatau_three_clause_classify) :-
    assert(user:(grade_k(X, low) :- X < 50)),
    assert(user:(grade_k(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade_k(X, high) :- X >= 80)),
    krakatau_target:compile_predicate_to_krakatau(grade_k/2, [], Code),
    has(Code, "if_icmp"),
    has(Code, "ireturn"),
    has(Code, ".method"),
    retractall(user:grade_k(_, _)).

test(krakatau_arithmetic_guard) :-
    assert(user:(even_check_k(X, yes) :- 0 =:= X mod 2)),
    assert(user:(even_check_k(X, no) :- 0 =\= X mod 2)),
    krakatau_target:compile_predicate_to_krakatau(even_check_k/2, [], Code),
    has(Code, "irem"),
    has(Code, "if_icmp"),
    retractall(user:even_check_k(_, _)).

test(krakatau_numeric_slots_arity3) :-
    assert(user:(add3_k(X, Y, Z) :- Z is X + Y)),
    krakatau_target:compile_predicate_to_krakatau(add3_k/3, [], Code),
    % Krakatau uses numeric slots: arg1=slot 0, arg2=slot 1
    has(Code, "iload_0"),
    has(Code, "iload_1"),
    has(Code, "iadd"),
    has(Code, "(II)I"),
    retractall(user:add3_k(_, _, _)).

test(krakatau_complex_arithmetic) :-
    assert(user:(formula_k(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    krakatau_target:compile_predicate_to_krakatau(formula_k/2, [], Code),
    has(Code, "imul"),
    has(Code, "iadd"),
    has(Code, ".limit stack"),
    retractall(user:formula_k(_, _)).

test(krakatau_division_guard) :-
    assert(user:(divisible_k(X, Y, yes) :- 0 =:= X mod Y)),
    assert(user:(divisible_k(X, Y, no) :- 0 =\= X mod Y)),
    krakatau_target:compile_predicate_to_krakatau(divisible_k/3, [], Code),
    has(Code, "irem"),
    has(Code, "(II)I"),
    retractall(user:divisible_k(_, _, _)).

test(krakatau_single_guard_return) :-
    assert(user:(positive_k(X, 1) :- X > 0)),
    assert(user:(positive_k(X, 0) :- X =< 0)),
    krakatau_target:compile_predicate_to_krakatau(positive_k/2, [], Code),
    has(Code, "if_icmp"),
    has(Code, "ireturn"),
    has(Code, ".end method"),
    retractall(user:positive_k(_, _)).

% ============================================================================
% Cross-target expanded: same complex predicate, both outputs
% ============================================================================

test(cross_target_three_clause) :-
    assert(user:(sign_shared(X, negative) :- X < 0)),
    assert(user:(sign_shared(X, zero) :- X =:= 0)),
    assert(user:(sign_shared(X, positive) :- X > 0)),
    jamaica_target:compile_predicate_to_jamaica(sign_shared/2, [], JaCode),
    krakatau_target:compile_predicate_to_krakatau(sign_shared/2, [], KrCode),
    % Both should have multiple branch comparisons
    has(JaCode, "if_icmp"),
    has(KrCode, "if_icmp"),
    % Different wrapping
    has(JaCode, "public static int sign_shared"),
    has(KrCode, ".method public static sign_shared"),
    retractall(user:sign_shared(_, _)).

test(cross_target_mod_guard) :-
    assert(user:(mod_test(X, even) :- 0 =:= X mod 2)),
    assert(user:(mod_test(X, odd) :- 0 =\= X mod 2)),
    jamaica_target:compile_predicate_to_jamaica(mod_test/2, [], JaCode),
    krakatau_target:compile_predicate_to_krakatau(mod_test/2, [], KrCode),
    % Both contain irem for modulo
    has(JaCode, "irem"),
    has(KrCode, "irem"),
    retractall(user:mod_test(_, _)).

% ============================================================================
% String operations (shared bytecode layer)
% ============================================================================

test(jvm_load_string) :-
    jvm_bytecode:jvm_load_string(hello, Instr),
    has(Instr, "ldc \"hello\"").

test(jvm_string_equals) :-
    jvm_bytecode:jvm_string_equals_bytecode(hello, world, [], Instrs),
    last(Instrs, Last),
    has(Last, "equals").

test(jvm_string_concat) :-
    jvm_bytecode:jvm_string_concat_bytecode(foo, bar, [], Instrs),
    last(Instrs, Last),
    has(Last, "concat").

test(jvm_tostring) :-
    jvm_bytecode:jvm_tostring_bytecode(42, [], Instrs),
    last(Instrs, Last),
    has(Last, "valueOf").

test(jvm_println) :-
    jvm_bytecode:jvm_println_bytecode(hello, [], Instrs),
    once(member(I, Instrs)),
    has(I, "System").

% ============================================================================
% String bindings
% ============================================================================

test(string_equals_binding) :-
    jamaica_target:init_jamaica_target,
    once(binding_registry:binding(jamaica, 'string_equals'/3, _, _, _, _)).

test(string_concat_binding) :-
    once(binding_registry:binding(krakatau, 'string_concat'/3, _, _, _, _)).

test(string_length_binding) :-
    once(binding_registry:binding(jamaica, 'string_length'/2, _, _, _, _)).

test(string_toupper_binding) :-
    once(binding_registry:binding(krakatau, 'string_toupper'/2, _, _, _, _)).

% ============================================================================
% Input source seed statements
% ============================================================================

test(jamaica_seed_statement) :-
    input_source:seed_statement(jamaica, _, alice, bob, Stmt),
    has(Stmt, "ldc \"alice\""),
    has(Stmt, "ldc \"bob\""),
    has(Stmt, "invokestatic addFact").

test(krakatau_seed_statement) :-
    input_source:seed_statement(krakatau, _, alice, bob, Stmt),
    has(Stmt, "ldc \"alice\""),
    has(Stmt, "ldc \"bob\""),
    has(Stmt, "invokestatic").

test(awk_seed_statement) :-
    input_source:seed_statement(awk, _, alice, bob, Stmt),
    has(Stmt, "add_fact"),
    has(Stmt, "\"alice\"").

test(vbnet_seed_statement) :-
    input_source:seed_statement(vbnet, _, alice, bob, Stmt),
    has(Stmt, "AddFact"),
    has(Stmt, "\"alice\"").

% ============================================================================
% Expanded bindings (array, conversion, stack)
% ============================================================================

test(array_newarray_binding) :-
    once(binding_registry:binding(jamaica, 'newarray_int'/2, 'newarray int', _, _, _)).

test(array_iaload_binding) :-
    once(binding_registry:binding(krakatau, 'iaload'/3, 'iaload', _, _, _)).

test(array_iastore_binding) :-
    once(binding_registry:binding(jamaica, 'iastore'/4, 'iastore', _, _, _)).

test(array_sort_binding) :-
    once(binding_registry:binding(krakatau, 'array_sort'/2, _, _, _, _)).

test(conversion_i2l_binding) :-
    once(binding_registry:binding(jamaica, 'i2l'/2, 'i2l', _, _, _)).

test(stack_dup_binding) :-
    once(binding_registry:binding(krakatau, 'dup'/1, 'dup', _, _, _)).

% ============================================================================
% Array operations (shared bytecode layer)
% ============================================================================

test(jvm_newarray_int) :-
    jvm_bytecode:jvm_newarray_bytecode(int, 10, Instrs),
    last(Instrs, Last),
    has(Last, "newarray int").

test(jvm_array_load) :-
    jvm_bytecode:jvm_array_load_bytecode(arr, 2, [arr-arr], Instrs),
    once(member(I, Instrs)),
    has(I, "aload"),
    last(Instrs, Last),
    has(Last, "iaload").

test(jvm_array_store) :-
    jvm_bytecode:jvm_array_store_bytecode(arr, 0, 42, [arr-arr], Instrs),
    last(Instrs, Last),
    has(Last, "iastore").

test(jvm_arraylength) :-
    jvm_bytecode:jvm_arraylength_bytecode(arr, [arr-arr], Instrs),
    last(Instrs, Last),
    has(Last, "arraylength").

% ============================================================================
% Component system hooks
% ============================================================================

test(jamaica_type_info) :-
    jamaica_target:jamaica_type_info(info(name(N), _, _)),
    has(N, "Jamaica").

test(jamaica_validate_config) :-
    jamaica_target:jamaica_validate_config([code("test body")]).

test(jamaica_compile_component) :-
    jamaica_target:jamaica_compile_component(test_comp,
        [code("    iload arg1\n    ireturn")], [], Code),
    has(Code, "comp_test_comp"),
    has(Code, "public static int").

test(krakatau_type_info) :-
    krakatau_target:krakatau_type_info(info(name(N), _, _)),
    has(N, "Krakatau").

test(krakatau_compile_component) :-
    krakatau_target:krakatau_compile_component(test_comp,
        [code("    iload_0\n    ireturn")], [], Code),
    has(Code, "comp_test_comp"),
    has(Code, ".method public static").

% ============================================================================
% Import management
% ============================================================================

test(jamaica_import_management) :-
    jamaica_target:clear_jamaica_imports,
    jamaica_target:collect_jamaica_import('java.util.HashMap'),
    jamaica_target:collect_jamaica_import('java.util.ArrayList'),
    jamaica_target:collect_jamaica_import('java.util.HashMap'),  % duplicate
    jamaica_target:get_jamaica_imports(Imports),
    length(Imports, 2),  % deduplicated
    jamaica_target:clear_jamaica_imports.

test(jamaica_format_imports) :-
    jamaica_target:format_jamaica_imports(
        ['java.util.HashMap', 'java.util.ArrayList'], Code),
    has(Code, "import java.util.HashMap;"),
    has(Code, "import java.util.ArrayList;").

% ============================================================================
% Pipeline mode
% ============================================================================

test(jamaica_pipeline) :-
    jamaica_target:compile_jamaica_pipeline(
        [step(transform, "    iload input\n    iconst_2\n    imul\n    ireturn"),
         step(filter, "    iload input\n    bipush 10\n    if_icmplt SKIP\n    iload input\n    ireturn\nSKIP:\n    iconst_0\n    ireturn")],
        [],
        Code),
    has(Code, "public class Pipeline"),
    has(Code, "step_transform"),
    has(Code, "step_filter"),
    has(Code, "public static void main").

test(jamaica_pipeline_custom_class) :-
    jamaica_target:compile_jamaica_pipeline(
        [step(inc, "    iload input\n    iconst_1\n    iadd\n    ireturn")],
        [class_name("MyPipe")],
        Code),
    has(Code, "public class MyPipe").

test(krakatau_pipeline) :-
    krakatau_target:compile_krakatau_pipeline(
        [step(double, "    iload_0\n    iconst_2\n    imul\n    ireturn")],
        [],
        Code),
    has(Code, ".class public Pipeline"),
    has(Code, "step_double"),
    has(Code, ".method public static main").

% ============================================================================
% Build file generation
% ============================================================================

test(jamaica_jar_manifest) :-
    jamaica_target:generate_jar_manifest([main_class("AncestorQuery")], Manifest),
    has(Manifest, "Main-Class: AncestorQuery"),
    has(Manifest, "Manifest-Version: 1.0"),
    has(Manifest, "UnifyWeaver").

test(jamaica_build_script) :-
    jamaica_target:generate_jamaica_build_script(
        [source_file("Test.ja"), main_class("Test"), output_jar("test.jar")],
        Script),
    has(Script, "jamaica.jar"),
    has(Script, "Test.ja"),
    has(Script, "jar cfm test.jar").

test(krakatau_build_script) :-
    krakatau_target:generate_krakatau_build_script(
        [source_file("Test.j"), main_class("Test"), output_jar("test.jar")],
        Script),
    has(Script, "krak2"),
    has(Script, "Test.j"),
    has(Script, "jar cfm test.jar").

% ============================================================================
% Glue support
% ============================================================================

test(jamaica_is_jvm_target) :-
    jvm_glue:jvm_target(jamaica).

test(krakatau_is_jvm_target) :-
    jvm_glue:jvm_target(krakatau).

test(jamaica_can_bridge_java) :-
    jvm_glue:can_use_direct(jamaica, java).

test(krakatau_can_bridge_kotlin) :-
    jvm_glue:can_use_direct(krakatau, kotlin).

% ============================================================================
% Transitive closure templates
% ============================================================================

test(jamaica_tc_template_exists) :-
    exists_file('templates/targets/jamaica/transitive_closure.mustache').

test(krakatau_tc_template_exists) :-
    exists_file('templates/targets/krakatau/transitive_closure.mustache').

test(vbnet_tc_template_exists) :-
    exists_file('templates/targets/vbnet/transitive_closure.mustache').

test(awk_tc_template_exists) :-
    exists_file('templates/targets/awk/transitive_closure.mustache').

test(jamaica_tc_template_content) :-
    read_file_to_string('templates/targets/jamaica/transitive_closure.mustache', Content, []),
    has(Content, "{{pred_cap}}Query"),
    has(Content, "addFact"),
    has(Content, "BFS"),
    has(Content, "invokevirtual").

test(krakatau_tc_template_content) :-
    read_file_to_string('templates/targets/krakatau/transitive_closure.mustache', Content, []),
    has(Content, ".class public"),
    has(Content, ".method"),
    has(Content, "add_fact"),
    has(Content, "BFS_LOOP").

test(vbnet_tc_template_content) :-
    read_file_to_string('templates/targets/vbnet/transitive_closure.mustache', Content, []),
    has(Content, "Public Class"),
    has(Content, "AddFact"),
    has(Content, "Queue"),
    has(Content, "HashSet").

test(awk_tc_template_content) :-
    read_file_to_string('templates/targets/awk/transitive_closure.mustache', Content, []),
    has(Content, "BEGIN"),
    has(Content, "queue"),
    has(Content, "visited"),
    has(Content, "adj").

test(jamaica_tc_dispatch) :-
    % Verify the dispatch clause compiles the template
    recursive_compiler:compile_transitive_closure(
        jamaica, ancestor, 2, parent, [], Code),
    has(Code, "AncestorQuery"),
    has(Code, "addFact").

test(krakatau_tc_dispatch) :-
    recursive_compiler:compile_transitive_closure(
        krakatau, ancestor, 2, parent, [], Code
    ),
    has(Code, ".class public AncestorQuery"),
    has(Code, "addFact").

test(vbnet_tc_dispatch) :-
    catch(
        recursive_compiler:compile_transitive_closure(
            vbnet, ancestor, 2, parent, [], Code),
        _,
        fail
    ),
    has(Code, "Public Class AncestorQuery"),
    has(Code, "AddFact").

test(awk_tc_dispatch) :-
    catch(
        recursive_compiler:compile_transitive_closure(
            awk, ancestor, 2, parent, [], Code),
        _,
        fail
    ),
    has(Code, "queue"),
    has(Code, "visited").

% Helper for inclusion check
sub_string_match(Substr, Str) :-
    sub_string(Str, _, _, _, Substr).

:- end_tests(jvm_asm_native_lowering).
