:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_get).

test(lowered_emit_get_ops) :-
    with_output_to(string(SVar), wam_cpp_lowered_emitter:emit_one(get_variable("X1", "A1"), "")),
    assertion(sub_string(SVar, _, _, _, "vm->set_cell(\"X1\", vm->get_cell(\"A1\"));")),
    with_output_to(string(SVal), wam_cpp_lowered_emitter:emit_one(get_value("X1", "A1"), "")),
    assertion(sub_string(SVal, _, _, _, "if (!vm->unify(va, vx)) return false;")).

:- end_tests(wam_cpp_lowered_get).
