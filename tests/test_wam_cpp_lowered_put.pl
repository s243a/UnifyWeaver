:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_put).

test(lowered_emit_put_ops) :-
    with_output_to(string(SVar), wam_cpp_lowered_emitter:emit_one(put_variable("X1", "A1"), "")),
    assertion(sub_string(SVar, _, _, _, "vm->put_variable_reg(\"X1\", \"A1\");")),
    with_output_to(string(SVal), wam_cpp_lowered_emitter:emit_one(put_value("X1", "A1"), "")),
    assertion(sub_string(SVal, _, _, _, "vm->set_cell(\"A1\", vm->get_cell(\"X1\"));")).

:- end_tests(wam_cpp_lowered_put).
