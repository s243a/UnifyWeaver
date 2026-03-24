:- module(custom_wat, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

type_info(info(
    name('Custom WAT Component'),
    version('1.0.0'),
    description('Injects custom WAT code and exposes it as a component')
)).

validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

% No initialization needed for compile-time component
init_component(_Name, _Config).

% Runtime invocation not supported in Prolog (compilation only)
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_wat))).

% Compilation: wraps custom WAT code in a function export
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),
    atom_string(Name, NameStr),
    format(string(Code),
';; Custom Component: ~w
(func $comp_~w (export "comp_~w") (param $input i64) (result i64)
~w
)', [NameStr, NameStr, NameStr, Body]).

% Register self
:- initialization((
    register_component_type(source, custom_wat, custom_wat, [
        description("Custom WAT Code")
    ])
), now).
