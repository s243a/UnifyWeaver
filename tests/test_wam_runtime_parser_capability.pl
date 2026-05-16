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

test(unknown_target_defaults_to_none) :-
    wam_target_runtime_parser(wam_lua, [], none).

test(unknown_target_native_request_errors,
     [error(domain_error(runtime_parser_mode(wam_lua), native))]) :-
    wam_target_runtime_parser(wam_lua, [runtime_parser(native)], _).

test(unknown_target_compiled_request_errors,
     [error(domain_error(runtime_parser_mode(wam_lua), compiled))]) :-
    wam_target_runtime_parser(wam_lua, [runtime_parser(compiled)], _).

test(invalid_request_errors,
     [error(domain_error(runtime_parser_request, bogus))]) :-
    wam_target_runtime_parser(wam_r, [runtime_parser(bogus)], _).

test(parser_dependent_builtin_catalogue) :-
    findall(PI, parser_dependent_builtin(PI), PIs),
    assertion(PIs == [read/2,
                      read_term_from_atom/2,
                      read_term_from_atom/3,
                      term_to_atom/2]).

:- end_tests(wam_runtime_parser_capability).
