:- module(test_jvm_asm_native_lowering, [test_jvm_asm_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/jamaica_target').
:- use_module('../../src/unifyweaver/targets/krakatau_target').
:- use_module('../../src/unifyweaver/core/jvm_bytecode').
:- use_module('../../src/unifyweaver/core/target_registry').

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
    once(member(macros, Caps)).

test(krakatau_registered) :-
    target_registry:registered_target(krakatau, jvm, Caps),
    once(member(assembly, Caps)),
    once(member(roundtrip, Caps)).

test(jamaica_module_linked) :-
    target_registry:target_module(jamaica, jamaica_target).

test(krakatau_module_linked) :-
    target_registry:target_module(krakatau, krakatau_target).

:- end_tests(jvm_asm_native_lowering).
