:- module(custom_go, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

type_info(info(
    name('Custom Go Component'),
    version('1.0.0'),
    description('Injects custom Go code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_go))).

% Compilation
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),
    
    % Collect imports if specified
    (   member(imports(Imports), Config)
    ->  forall(member(I, Imports), go_target:collect_binding_import(I))
    ;   true
    ),
    
    atom_string(Name, NameStr),
    format(string(Code), 
"// Custom Component: ~w
type comp_~w struct{}

func (c *comp_~w) Invoke(input interface{}) (interface{}, error) {
~w
}
", [NameStr, NameStr, NameStr, Body]).

% Register self
:- initialization((
    register_component_type(source, custom_go, custom_go, [
        description("Custom Go Code")
    ])
), now).
