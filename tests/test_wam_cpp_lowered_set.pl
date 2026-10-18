:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_set).

test(lowered_emit_set_ops) :-
    with_output_to(string(SV), wam_cpp_lowered_emitter:emit_one(set_variable("X2"), "")),
    assertion(sub_string(SV, _, _, _, "Instruction::SetVariable(\"X2\")")),
    with_output_to(string(SVal), wam_cpp_lowered_emitter:emit_one(set_value("X3"), "")),
    assertion(sub_string(SVal, _, _, _, "Instruction::SetValue(\"X3\")")),
    with_output_to(string(SC), wam_cpp_lowered_emitter:emit_one(set_constant("a"), "")),
    assertion(sub_string(SC, _, _, _, "Instruction::SetConstant(Value::Atom(\"a\"))")).

:- end_tests(wam_cpp_lowered_set).
