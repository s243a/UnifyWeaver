:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_emit).

test(lowered_emit_terminal_opcodes) :-
    with_output_to(string(P), wam_cpp_lowered_emitter:emit_one(proceed, "")),
    assertion(sub_string(P, _, _, _, "return true;")),
    with_output_to(string(F), wam_cpp_lowered_emitter:emit_one(fail, "")),
    assertion(sub_string(F, _, _, _, "return false;")).

:- end_tests(wam_cpp_lowered_emit).
