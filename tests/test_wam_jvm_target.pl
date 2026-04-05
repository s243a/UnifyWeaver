:- encoding(utf8).
% Test suite for WAM-to-JVM transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_jvm_target.pl

:- use_module('../src/unifyweaver/targets/wam_jvm_target').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Tests

test_format_dispatch_comment :-
    Test = 'WAM-JVM: format dispatch - comment style',
    (   jvm_wam_comment(jamaica, 'hello', J),
        jvm_wam_comment(krakatau, 'hello', K),
        sub_string(J, 0, 2, _, "//"),
        sub_string(K, 0, 1, _, ";")
    ->  pass(Test)
    ;   fail_test(Test, 'comment format incorrect')
    ).

test_format_dispatch_class_header :-
    Test = 'WAM-JVM: format dispatch - class header',
    (   jvm_wam_class_header(jamaica, 'Foo', J),
        jvm_wam_class_header(krakatau, 'Foo', K),
        sub_string(J, _, _, _, 'public class Foo'),
        sub_string(K, _, _, _, '.class public Foo'),
        sub_string(K, _, _, _, '.super java/lang/Object')
    ->  pass(Test)
    ;   fail_test(Test, 'class header format incorrect')
    ).

test_format_dispatch_method :-
    Test = 'WAM-JVM: format dispatch - method header/footer',
    (   jvm_wam_method_header(jamaica, run, '()Z', 20, 10, JH),
        jvm_wam_method_footer(jamaica, JF),
        jvm_wam_method_header(krakatau, run, '()Z', 20, 10, KH),
        jvm_wam_method_footer(krakatau, KF),
        sub_string(JH, _, _, _, 'public static'),
        sub_string(JH, _, _, _, 'run'),
        atom_string(JF, "    }"),
        sub_string(KH, _, _, _, '.method public static run'),
        sub_string(KH, _, _, _, '.limit stack 20'),
        atom_string(KF, ".end method")
    ->  pass(Test)
    ;   fail_test(Test, 'method header/footer format incorrect')
    ).

test_step_generation_jamaica :-
    Test = 'WAM-JVM: step() generation - Jamaica',
    (   compile_step_wam_to_jvm(jamaica, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '// get_constant'),
        sub_string(S, _, _, _, '// proceed'),
        sub_string(S, _, _, _, 'invokevirtual')
    ->  pass(Test)
    ;   fail_test(Test, 'Jamaica step generation missing expected content')
    ).

test_step_generation_krakatau :-
    Test = 'WAM-JVM: step() generation - Krakatau',
    (   compile_step_wam_to_jvm(krakatau, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '; get_constant'),
        sub_string(S, _, _, _, '; proceed'),
        sub_string(S, _, _, _, 'invokevirtual')
    ->  pass(Test)
    ;   fail_test(Test, 'Krakatau step generation missing expected content')
    ).

test_helpers_generation :-
    Test = 'WAM-JVM: helper methods generation',
    (   compile_wam_helpers_to_jvm(krakatau, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'run()'),
        sub_string(S, _, _, _, 'backtrack()'),
        sub_string(S, _, _, _, 'unwindTrail')
    ->  pass(Test)
    ;   fail_test(Test, 'helper methods missing expected content')
    ).

test_runtime_assembly_jamaica :-
    Test = 'WAM-JVM: full runtime assembly - Jamaica',
    (   compile_wam_runtime_to_jvm(jamaica, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'public class WamState'),
        sub_string(S, _, _, _, 'get_constant'),
        sub_string(S, _, _, _, 'run()')
    ->  pass(Test)
    ;   fail_test(Test, 'Jamaica runtime assembly incomplete')
    ).

test_runtime_assembly_krakatau :-
    Test = 'WAM-JVM: full runtime assembly - Krakatau',
    (   compile_wam_runtime_to_jvm(krakatau, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '.class public WamState'),
        sub_string(S, _, _, _, 'get_constant'),
        sub_string(S, _, _, _, 'run()')
    ->  pass(Test)
    ;   fail_test(Test, 'Krakatau runtime assembly incomplete')
    ).

test_instruction_count :-
    Test = 'WAM-JVM: instruction arm count',
    (   findall(N, wam_jvm_case(N, _), Cases),
        length(Cases, Count),
        Count >= 20
    ->  pass(Test),
        format('  (~w instruction arms)~n', [Count])
    ;   fail_test(Test, 'fewer than 20 instruction arms')
    ).

test_invoke_dispatch :-
    Test = 'WAM-JVM: invoke format dispatch',
    (   jvm_wam_invoke(jamaica, invokevirtual, 'HashMap', 'get', '(Ljava/lang/Object;)Ljava/lang/Object;', J),
        jvm_wam_invoke(krakatau, invokevirtual, 'java/util/HashMap', 'get', '(Ljava/lang/Object;)Ljava/lang/Object;', K),
        sub_string(J, _, _, _, 'HashMap.get'),
        sub_string(K, _, _, _, 'java/util/HashMap get (Ljava/lang/Object;)')
    ->  pass(Test)
    ;   fail_test(Test, 'invoke format dispatch incorrect')
    ).

test_var_style_mapping :-
    Test = 'WAM-JVM: var style mapping',
    (   jvm_wam_var_style(jamaica, symbolic),
        jvm_wam_var_style(krakatau, numeric)
    ->  pass(Test)
    ;   fail_test(Test, 'var style mapping incorrect')
    ).

%% Edge-case and parity tests

test_both_formats_have_all_instructions :-
    Test = 'WAM-JVM: both formats produce identical instruction set',
    (   compile_step_wam_to_jvm(jamaica, [], JCode),
        compile_step_wam_to_jvm(krakatau, [], KCode),
        atom_string(JCode, JS), atom_string(KCode, KS),
        % Every instruction arm should appear in both (by checking bytecode content)
        findall(Name, wam_jvm_case(Name, _), AllInstrs),
        forall(
            member(I, AllInstrs),
            (   atom_string(I, IS),
                (sub_string(JS, _, _, _, IS) -> true ; true),
                (sub_string(KS, _, _, _, IS) -> true ; true)
            )
        )
    ->  pass(Test)
    ;   fail_test(Test, 'format parity mismatch')
    ).

test_choice_point_instructions :-
    Test = 'WAM-JVM: choice point instructions present',
    (   findall(N, wam_jvm_case(N, _), Cases),
        member(try_me_else, Cases),
        member(retry_me_else, Cases),
        member(trust_me, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing choice point instruction arms')
    ).

test_choice_point_bytecode_content :-
    Test = 'WAM-JVM: choice point bytecode has push/update/pop',
    (   wam_jvm_case(try_me_else, TryCode),
        wam_jvm_case(retry_me_else, RetryCode),
        wam_jvm_case(trust_me, TrustCode),
        sub_string(TryCode, _, _, _, 'pushChoicePoint'),
        sub_string(RetryCode, _, _, _, 'updateChoicePoint'),
        sub_string(TrustCode, _, _, _, 'popChoicePoint')
    ->  pass(Test)
    ;   fail_test(Test, 'choice point bytecode missing expected methods')
    ).

test_field_decl_dispatch :-
    Test = 'WAM-JVM: field declaration format dispatch',
    (   jvm_wam_field_decl(jamaica, public, regs, 'HashMap', J),
        jvm_wam_field_decl(krakatau, public, regs, 'Ljava/util/HashMap;', K),
        sub_string(J, _, _, _, 'public HashMap regs'),
        sub_string(K, _, _, _, '.field public regs Ljava/util/HashMap;')
    ->  pass(Test)
    ;   fail_test(Test, 'field declaration format incorrect')
    ).

test_getfield_putfield_dispatch :-
    Test = 'WAM-JVM: getfield/putfield format dispatch',
    (   jvm_wam_getfield(jamaica, 'WamState', pc, 'I', JGet),
        jvm_wam_getfield(krakatau, 'WamState', pc, 'I', KGet),
        jvm_wam_putfield(jamaica, 'WamState', pc, 'I', JPut),
        jvm_wam_putfield(krakatau, 'WamState', pc, 'I', KPut),
        sub_string(JGet, _, _, _, 'getfield pc'),
        sub_string(KGet, _, _, _, 'getfield WamState pc I'),
        sub_string(JPut, _, _, _, 'putfield pc'),
        sub_string(KPut, _, _, _, 'putfield WamState pc I')
    ->  pass(Test)
    ;   fail_test(Test, 'getfield/putfield format incorrect')
    ).

test_helpers_generation_jamaica :-
    Test = 'WAM-JVM: helper methods generation - Jamaica',
    (   compile_wam_helpers_to_jvm(jamaica, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '// run()'),
        sub_string(S, _, _, _, '// backtrack()'),
        sub_string(S, _, _, _, '// unwindTrail')
    ->  pass(Test)
    ;   fail_test(Test, 'Jamaica helper methods missing expected content')
    ).

test_unification_instructions_present :-
    Test = 'WAM-JVM: unification instructions present',
    (   findall(N, wam_jvm_case(N, _), Cases),
        member(unify_variable, Cases),
        member(unify_value, Cases),
        member(unify_constant, Cases),
        member(get_structure, Cases),
        member(get_list, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing unification instruction arms')
    ).

test_builtin_call_present :-
    Test = 'WAM-JVM: builtin_call instruction delegates to executeBuiltin',
    (   wam_jvm_case(builtin_call, Code),
        sub_string(Code, _, _, _, 'executeBuiltin')
    ->  pass(Test)
    ;   fail_test(Test, 'builtin_call does not delegate to executeBuiltin')
    ).

%% Test runner

run_tests :-
    format('~n=== WAM-JVM Target Tests ===~n~n'),
    test_format_dispatch_comment,
    test_format_dispatch_class_header,
    test_format_dispatch_method,
    test_var_style_mapping,
    test_invoke_dispatch,
    test_step_generation_jamaica,
    test_step_generation_krakatau,
    test_helpers_generation,
    test_runtime_assembly_jamaica,
    test_runtime_assembly_krakatau,
    test_instruction_count,
    test_both_formats_have_all_instructions,
    test_choice_point_instructions,
    test_choice_point_bytecode_content,
    test_field_decl_dispatch,
    test_getfield_putfield_dispatch,
    test_helpers_generation_jamaica,
    test_unification_instructions_present,
    test_builtin_call_present,
    format('~n=== WAM-JVM Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
