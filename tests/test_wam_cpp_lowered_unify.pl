:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_unify).

test(lowered_emit_unify_ops) :-
    with_output_to(string(UV), wam_cpp_lowered_emitter:emit_one(unify_variable("X1"), "")),
    assertion(sub_string(UV, _, _, _, "Instruction::UnifyVariable(\"X1\")")),
    with_output_to(string(UVal), wam_cpp_lowered_emitter:emit_one(unify_value("X1"), "")),
    assertion(sub_string(UVal, _, _, _, "Instruction::UnifyValue(\"X1\")")),
    with_output_to(string(UC), wam_cpp_lowered_emitter:emit_one(unify_constant("foo"), "")),
    assertion(sub_string(UC, _, _, _, "Instruction::UnifyConstant(Value::Atom(\"foo\"))")).

:- end_tests(wam_cpp_lowered_unify).
