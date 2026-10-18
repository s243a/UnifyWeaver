:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_emit_ctrl).

test(lowered_emit_frame_opcodes) :-
    with_output_to(string(A), wam_cpp_lowered_emitter:emit_one(allocate, "")),
    assertion(sub_string(A, _, _, _, "Instruction::Allocate()")),
    with_output_to(string(D), wam_cpp_lowered_emitter:emit_one(deallocate, "")),
    assertion(sub_string(D, _, _, _, "Instruction::Deallocate()")).

:- end_tests(wam_cpp_lowered_emit_ctrl).
