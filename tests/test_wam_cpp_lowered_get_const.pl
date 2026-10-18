:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_get_const).

test(lowered_emit_get_const_ops) :-
    with_output_to(string(S1), wam_cpp_lowered_emitter:emit_one(get_constant("foo", "A1"), "")),
    assertion(sub_string(S1, _, _, _, "vm->match_reg_atom(\"A1\", \"foo\")")),
    with_output_to(string(S2), wam_cpp_lowered_emitter:emit_one(get_integer(42, "A1"), "")),
    assertion(sub_string(S2, _, _, _, "Value::Integer(42)")),
    with_output_to(string(S3), wam_cpp_lowered_emitter:emit_one(get_nil("A1"), "")),
    assertion(sub_string(S3, _, _, _, "vm->match_reg_atom(\"A1\", \"[]\")")).

:- end_tests(wam_cpp_lowered_get_const).
