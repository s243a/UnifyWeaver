:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_control).

test(noop_ops) :-
    forall(member(I, [try_me_else("L"), trust_me, cut_ite, cut("L"), get_level("A1"), jump("L")]),
           ( with_output_to(string(S), wam_cpp_lowered_emitter:emit_one(I, "")),
             assertion(S == "") )).

test(call_execute) :-
    with_output_to(string(SC), wam_cpp_lowered_emitter:emit_one(call("foo/2", "2"), "")),
    assertion(sub_string(SC, _, _, _, "std::size_t saved_cp = vm->cp;")),
    with_output_to(string(SE), wam_cpp_lowered_emitter:emit_one(execute("foo/2"), "")),
    assertion(sub_string(SE, _, _, _, "return vm->run();")).

test(builtin_foreign) :-
    with_output_to(string(SB), wam_cpp_lowered_emitter:emit_one(builtin_call("is/2", "2"), "")),
    assertion(sub_string(SB, _, _, _, "Instruction::BuiltinCall(\"is/2\", 2)")),
    with_output_to(string(SF), wam_cpp_lowered_emitter:emit_one(call_foreign("foo/2", "2"), "")),
    assertion(sub_string(SF, _, _, _, "Instruction::CallForeign(\"foo/2\", 2)")).

test(aggregate) :-
    with_output_to(string(SBA), wam_cpp_lowered_emitter:emit_one(begin_aggregate("sum", "A1", "A2"), "")),
    assertion(sub_string(SBA, _, _, _, "Instruction::BeginAggregate(\"sum\", \"A1\", \"A2\")")),
    with_output_to(string(SEA), wam_cpp_lowered_emitter:emit_one(end_aggregate("A1"), "")),
    assertion(sub_string(SEA, _, _, _, "Instruction::EndAggregate(\"A1\")")).

:- end_tests(wam_cpp_lowered_control).
