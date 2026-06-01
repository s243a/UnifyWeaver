:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_runtime_parser_capability').

:- begin_tests(wam_runtime_parser_capability).

test(r_defaults_to_native_parser) :-
    wam_target_runtime_parser(wam_r, [], native(parse_term)).

test(r_alias_defaults_to_native_parser) :-
    wam_target_runtime_parser(r, [], native(parse_term)).

test(off_disables_parser_even_for_r) :-
    wam_target_runtime_parser(wam_r, [runtime_parser(off)], none).

test(r_can_require_native_parser) :-
    wam_target_runtime_parser(wam_r, [runtime_parser(native)],
                              native(parse_term)).

test(r_can_opt_into_compiled_parser) :-
    wam_target_runtime_parser(wam_r, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)).

test(cpp_defaults_to_native_parser) :-
    wam_target_runtime_parser(wam_cpp, [], native(parse_term)).

test(cpp_alias_defaults_to_native_parser) :-
    wam_target_runtime_parser(cpp, [], native(parse_term)).

test(cpp_off_disables_parser) :-
    wam_target_runtime_parser(wam_cpp, [runtime_parser(off)], none).

test(cpp_can_require_native_parser) :-
    wam_target_runtime_parser(wam_cpp, [runtime_parser(native)],
                              native(parse_term)).

test(cpp_can_opt_into_compiled_parser) :-
    wam_target_runtime_parser(wam_cpp, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)).

test(unknown_target_defaults_to_none) :-
    wam_target_runtime_parser(wam_lua, [], none).

test(python_defaults_to_none) :-
    wam_target_runtime_parser(wam_python, [], none).

test(python_can_opt_into_compiled_parser) :-
    wam_target_runtime_parser(wam_python, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)).

test(elixir_defaults_to_none) :-
    wam_target_runtime_parser(wam_elixir, [], none).

test(unknown_target_native_request_errors,
     [error(domain_error(runtime_parser_mode(wam_lua), native))]) :-
    wam_target_runtime_parser(wam_lua, [runtime_parser(native)], _).

test(python_native_request_errors,
     [error(domain_error(runtime_parser_mode(wam_python), native))]) :-
    wam_target_runtime_parser(wam_python, [runtime_parser(native)], _).

test(elixir_native_request_errors,
     [error(domain_error(runtime_parser_mode(wam_elixir), native))]) :-
    wam_target_runtime_parser(wam_elixir, [runtime_parser(native)], _).

test(unknown_target_compiled_request_errors,
     [error(domain_error(runtime_parser_mode(wam_lua), compiled))]) :-
    wam_target_runtime_parser(wam_lua, [runtime_parser(compiled)], _).

test(elixir_compiled_request_errors,
     [error(domain_error(runtime_parser_mode(wam_elixir), compiled))]) :-
    wam_target_runtime_parser(wam_elixir, [runtime_parser(compiled)], _).

test(invalid_request_errors,
     [error(domain_error(runtime_parser_request, bogus))]) :-
    wam_target_runtime_parser(wam_r, [runtime_parser(bogus)], _).

test(parser_dependent_builtin_catalogue) :-
    findall(PI, parser_dependent_builtin(PI), PIs),
    assertion(PIs == [read/1,
                      read/2,
                      read_term/1,
                      read_term_from_atom/2,
                      read_term_from_atom/3,
                      term_to_atom/2]).

test(parser_dependent_goal_read_default_stream) :-
    once(parser_dependent_goal(read(_), read/1)).

test(parser_dependent_goal_read) :-
    parser_dependent_goal(read(_, _), read/2).

test(parser_dependent_goal_module_qualified) :-
    once(parser_dependent_goal(user:read_term_from_atom('f(a)', _),
                               read_term_from_atom/2)).

test(parser_dependent_goal_term_to_atom_forward_fails, [fail]) :-
    parser_dependent_goal(term_to_atom(f(a), _), _).

test(parser_dependent_goal_term_to_atom_reverse) :-
    parser_dependent_goal(term_to_atom(_, 'f(a)'), term_to_atom/2).

test(parser_dependent_goal_term_to_atom_both_bound) :-
    parser_dependent_goal(term_to_atom(f(a), 'f(a)'), term_to_atom/2).

test(parser_dependent_body_goal_conjunction) :-
    parser_dependent_body_goal((true, read_term_from_atom('f(a)', _)),
                               read_term_from_atom/2).

test(parser_dependent_body_goal_disjunction) :-
    parser_dependent_body_goal((fail ; read(_, _)), read/2).

test(parser_dependent_body_goal_if_then) :-
    parser_dependent_body_goal((true -> term_to_atom(_, 'f(a)')),
                               term_to_atom/2).

test(parser_dependent_body_goal_wrappers) :-
    parser_dependent_body_goal(once(call(read_term_from_atom('f(a)', _))),
                               read_term_from_atom/2).

test(parser_dependent_body_goal_negation) :-
    parser_dependent_body_goal(\+(read_term_from_atom('f(a)', _)),
                               read_term_from_atom/2).

test(parser_dependent_body_goal_forward_term_to_atom_ignored, [fail]) :-
    parser_dependent_body_goal((true, term_to_atom(f(a), _)), _).

test(parser_dependent_body_goal_module_qualified_body) :-
    once(parser_dependent_body_goal(user:(true, read(_, _)), read/2)).

% --- F# WAM target (compiled mode opt-in only) ----------------------------
%
% F# has no native parser today, but `runtime_parser(compiled)` runs
% the portable parser end-to-end via WAM.  Keep the default at `none`
% so existing generated projects do not silently bundle the parser library.

test(fsharp_defaults_to_none) :-
    wam_target_runtime_parser(wam_fsharp, [], none).

test(fsharp_alias_defaults_to_none) :-
    wam_target_runtime_parser(fsharp, [], none).

test(fsharp_can_opt_into_compiled_parser) :-
    wam_target_runtime_parser(wam_fsharp, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)).

test(fsharp_cannot_require_native_parser_yet,
     [throws(error(domain_error(runtime_parser_mode(wam_fsharp), native), _))]) :-
    wam_target_runtime_parser(wam_fsharp, [runtime_parser(native)], _).

test(fsharp_off_explicit_none) :-
    wam_target_runtime_parser(wam_fsharp, [runtime_parser(off)], none).


% --- Haskell WAM target (compiled mode opt-in only) -----------------------

test(haskell_defaults_to_none) :-
    wam_target_runtime_parser(wam_haskell, [], none).

test(haskell_alias_defaults_to_none) :-
    wam_target_runtime_parser(haskell, [], none).

test(haskell_alias_can_opt_into_compiled_parser) :-
    wam_target_runtime_parser(haskell, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)).

test(haskell_can_opt_into_compiled_parser) :-
    wam_target_runtime_parser(wam_haskell, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)).

test(haskell_cannot_require_native_parser_yet,
     [throws(error(domain_error(runtime_parser_mode(wam_haskell), native), _))]) :-
    wam_target_runtime_parser(wam_haskell, [runtime_parser(native)], _).

test(haskell_off_explicit_none) :-
    wam_target_runtime_parser(wam_haskell, [runtime_parser(off)], none).

:- end_tests(wam_runtime_parser_capability).
