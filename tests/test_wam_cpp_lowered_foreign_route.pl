:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_foreign_route).

test(call_foreign_routed) :-
    with_output_to(string(S), wam_cpp_lowered_emitter:emit_one(call("foo/2", "2"), "", ["foo/2"])),
    assertion(sub_string(S, _, _, _, "via foreign kernel")),
    assertion(sub_string(S, _, _, _, "Instruction::CallForeign(\"foo/2\", 2)")).

test(execute_foreign_routed) :-
    with_output_to(string(S), wam_cpp_lowered_emitter:emit_one(execute("foo/2"), "", ["foo/2"])),
    assertion(sub_string(S, _, _, _, "via foreign kernel")),
    assertion(sub_string(S, _, _, _, "Instruction::CallForeign(\"foo/2\", 2)")).

test(call_not_routed_without_foreign) :-
    with_output_to(string(S), wam_cpp_lowered_emitter:emit_one(call("foo/2", "2"), "", [])),
    assertion(\+ sub_string(S, _, _, _, "via foreign kernel")).

:- end_tests(wam_cpp_lowered_foreign_route).
