:- consult(agent_loop_module).
:- initialization(run_test).

run_test :-
    write('=== Compilation Tests ==='), nl,
    test_method(python, on_token),
    test_method(rust, on_token),
    test_method(python, is_retryable_status),
    test_method(rust, is_retryable_status),
    test_method(python, compute_delay),
    test_method(rust, compute_delay),
    test_method(python, cache_get_guard),
    test_method(rust, cache_get_guard),
    test_method(python, make_key),
    test_method(rust, make_key),
    test_method(python, format_summary),
    test_method(rust, format_summary),
    test_method(python, extract_json_dispatch),
    test_method(rust, extract_json_dispatch),
    test_method(python, cache_put_guard),
    test_method(rust, cache_put_guard),
    test_method(python, is_over_budget),
    test_method(rust, is_over_budget),
    test_method(python, budget_remaining),
    test_method(rust, budget_remaining),
    test_method(python, cache_clear),
    test_method(rust, cache_clear),
    test_method(python, cache_len),
    test_method(rust, cache_len),
    nl, write('=== resolve_type Tests ==='), nl,
    resolve_type(python, optional(float), T1), format('  py optional(float) = ~w~n', [T1]),
    resolve_type(rust, optional(float), T2), format('  rs optional(float) = ~w~n', [T2]),
    resolve_type(python, set_of_string, T3), format('  py set_of_string = ~w~n', [T3]),
    resolve_type(rust, set_of_string, T4), format('  rs set_of_string = ~w~n', [T4]),
    nl, write('=== Count ==='), nl,
    aggregate_all(count, shared_logic(_, _, _), N),
    format('  Total shared_logic facts: ~w~n', [N]),
    findall(M, shared_logic(_, M, _), AllMs),
    include([M]>>(compile_logic(python, M, _), compile_logic(rust, M, _)), AllMs, OkMs),
    length(OkMs, NC),
    format('  Compiling for both targets: ~w~n', [NC]),
    write('All done.'), nl.

test_method(Target, Method) :-
    (compile_logic(Target, Method, Code) ->
        format('  ~w ~w: ~w~n', [Target, Method, Code])
    ;
        format('  FAIL: ~w ~w~n', [Target, Method])
    ).
