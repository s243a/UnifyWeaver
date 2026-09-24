:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_put_constant).

test(lowered_emit_put_constant) :-
    with_output_to(string(P), wam_cpp_lowered_emitter:emit_one(put_constant("a", "A1"), "")),
    assertion(sub_string(P, _, _, _, "vm->assign_reg(\"A1\", Value::Atom(\"a\"));")).

:- end_tests(wam_cpp_lowered_put_constant).
