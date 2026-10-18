:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- begin_tests(wam_cpp_lowered_func_name).

test(lowered_func_name_mangling) :-
    wam_cpp_lowered_emitter:cpp_lowered_func_name(foo/2, N1),
    assertion(N1 == lowered_foo_2),
    wam_cpp_lowered_emitter:cpp_lowered_func_name(foo_bar/1, N2),
    assertion(N2 == lowered_foo_bar_1),
    wam_cpp_lowered_emitter:cpp_lowered_func_name(append/3, N3),
    assertion(N3 == lowered_append_3),
    wam_cpp_lowered_emitter:cpp_lowered_func_name((=)/2, N4),
    assertion(N4 == lowered___2),
    wam_cpp_lowered_emitter:cpp_lowered_func_name(($)/1, N5),
    assertion(N5 == lowered___1).

:- end_tests(wam_cpp_lowered_func_name).
