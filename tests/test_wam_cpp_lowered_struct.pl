:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_struct).

test(lowered_emit_struct_ops) :-
    with_output_to(string(SGS), wam_cpp_lowered_emitter:emit_one(get_structure("foo/2", "A1"), "")),
    assertion(sub_string(SGS, _, _, _, "Instruction::GetStructure(\"foo/2\", \"A1\")")),
    with_output_to(string(SGL), wam_cpp_lowered_emitter:emit_one(get_list("A1"), "")),
    assertion(sub_string(SGL, _, _, _, "Instruction::GetList(\"A1\")")),
    with_output_to(string(SPS), wam_cpp_lowered_emitter:emit_one(put_structure("foo/2", "A1"), "")),
    assertion(sub_string(SPS, _, _, _, "Instruction::PutStructure(\"foo/2\", \"A1\")")),
    with_output_to(string(SPL), wam_cpp_lowered_emitter:emit_one(put_list("A1"), "")),
    assertion(sub_string(SPL, _, _, _, "Instruction::PutList(\"A1\")")).

:- end_tests(wam_cpp_lowered_struct).
