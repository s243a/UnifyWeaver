% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% ui_registry.pl - Integration of UI primitives with component registry
%
% Registers UI primitive types (layout, container, component) with
% the unified component registry for compilation and management.
%
% Usage:
%   use_module('src/unifyweaver/ui/ui_registry').
%   init_ui_registry.  % Register UI category and types
%
%   % Declare a UI component instance
%   declare_ui_component(login_form, [
%       spec(layout(stack, [spacing(16)], [
%           component(text_input, [label("Email"), bind(email)]),
%           component(button, [label("Login"), on_click(submit)])
%       ])),
%       target(vue)
%   ]).
%
%   % Compile to target code
%   compile_ui_component(login_form, [], Code).

:- module(ui_registry, [
    % Initialization
    init_ui_registry/0,

    % Component management
    declare_ui_component/2,      % +Name, +Config
    compile_ui_component/3,      % +Name, +Options, -Code

    % Type interface (required by component_registry)
    type_info/1,
    validate_config/1,
    init_component/2,
    compile_component/4,
    invoke_component/4,

    % Testing
    test_ui_registry/0
]).

:- use_module(library(lists)).

% Load dependencies conditionally
:- catch(use_module('../core/component_registry'), _, true).
:- catch(use_module('ui_primitives'), _, true).
:- catch(use_module('vue_generator'), _, true).

% ============================================================================
% INITIALIZATION
% ============================================================================

%! init_ui_registry is det
%  Initialize the UI registry - define category and register types.
init_ui_registry :-
    (   current_predicate(component_registry:define_category/3)
    ->  init_with_registry
    ;   format('Warning: component_registry not loaded, using standalone mode~n')
    ).

init_with_registry :-
    % Define the UI category
    component_registry:define_category(ui, "User interface components", [
        requires_compilation(true),
        singleton(false)
    ]),

    % Register layout type
    component_registry:register_component_type(ui, layout, ui_registry, [
        description("Layout primitives for spatial arrangement")
    ]),

    % Register container type
    component_registry:register_component_type(ui, container, ui_registry, [
        description("Container primitives for grouping and styling")
    ]),

    % Register component type
    component_registry:register_component_type(ui, component, ui_registry, [
        description("Atomic UI components")
    ]),

    % Register page type
    component_registry:register_component_type(ui, page, ui_registry, [
        description("Complete page compositions")
    ]),

    format('UI registry initialized~n').

% ============================================================================
% TYPE INTERFACE (required by component_registry)
% ============================================================================

%! type_info(-Info) is det
%  Return metadata about the UI type.
type_info(info{
    name: "UI Primitives",
    version: "1.0.0",
    targets: [vue, react, flutter, swiftui, html],
    description: "Declarative UI primitives for cross-platform code generation"
}).

%! validate_config(+Config) is semidet
%  Validate UI component configuration.
validate_config(Config) :-
    is_list(Config),
    (   member(spec(Spec), Config)
    ->  validate_spec(Spec)
    ;   true  % spec is optional for lazy definition
    ),
    (   member(target(Target), Config)
    ->  valid_target(Target)
    ;   true  % target is optional, defaults to vue
    ).

validate_spec(Spec) :-
    (   current_predicate(ui_primitives:validate_ui_node/1)
    ->  ui_primitives:validate_ui_node(Spec)
    ;   true  % Skip validation if module not loaded
    ).

valid_target(vue).
valid_target(react).
valid_target(flutter).
valid_target(swiftui).
valid_target(html).

%! init_component(+Name, +Config) is det
%  Initialize a UI component instance.
init_component(Name, Config) :-
    format('Initialized UI component: ~w~n', [Name]),
    (   member(spec(Spec), Config),
        member(eager_compile(true), Config)
    ->  get_target(Config, Target),
        compile_spec(Spec, Target, _Code),
        format('  Pre-compiled for target: ~w~n', [Target])
    ;   true
    ).

%! compile_component(+Name, +Config, +Options, -Code) is det
%  Compile a UI component to target code.
compile_component(Name, Config, Options, Code) :-
    (   member(spec(Spec), Config)
    ->  true
    ;   format('Error: No spec defined for component ~w~n', [Name]),
        fail
    ),
    get_target_from_options(Options, Config, Target),
    compile_spec(Spec, Target, Code).

%! invoke_component(+Name, +Config, +Input, -Output) is det
%  UI components don't have runtime invocation - they compile to code.
invoke_component(Name, _Config, _Input, Output) :-
    format(atom(Output), 'UI component ~w does not support runtime invocation', [Name]).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

get_target(Config, Target) :-
    (   member(target(Target), Config)
    ->  true
    ;   Target = vue  % Default to Vue
    ).

get_target_from_options(Options, Config, Target) :-
    (   member(target(Target), Options)
    ->  true
    ;   get_target(Config, Target)
    ).

%! compile_spec(+Spec, +Target, -Code) is det
%  Compile a UI spec to target code.
compile_spec(Spec, vue, Code) :-
    (   current_predicate(vue_generator:generate_vue_template/2)
    ->  vue_generator:generate_vue_template(Spec, Code)
    ;   current_predicate(ui_primitives:generate_ui/3)
    ->  ui_primitives:generate_ui(Spec, vue, Code)
    ;   Code = 'ERROR: No Vue generator available'
    ).

compile_spec(Spec, html, Code) :-
    % HTML can reuse Vue template generation (without Vue directives)
    (   current_predicate(ui_primitives:generate_ui/3)
    ->  ui_primitives:generate_ui(Spec, html, Code)
    ;   Code = 'TODO: HTML generation'
    ).

compile_spec(Spec, react, Code) :-
    (   current_predicate(ui_primitives:generate_ui/3)
    ->  ui_primitives:generate_ui(Spec, react, Code)
    ;   Code = 'TODO: React generation'
    ).

compile_spec(Spec, flutter, Code) :-
    (   current_predicate(ui_primitives:generate_ui/3)
    ->  ui_primitives:generate_ui(Spec, flutter, Code)
    ;   Code = 'TODO: Flutter generation'
    ).

compile_spec(Spec, swiftui, Code) :-
    (   current_predicate(ui_primitives:generate_ui/3)
    ->  ui_primitives:generate_ui(Spec, swiftui, Code)
    ;   Code = 'TODO: SwiftUI generation'
    ).

% ============================================================================
% CONVENIENCE API
% ============================================================================

%! declare_ui_component(+Name, +Config) is det
%  Declare a UI component (shorthand for component_registry:declare_component).
declare_ui_component(Name, Config) :-
    (   current_predicate(component_registry:declare_component/4)
    ->  % Determine type from spec
        (   member(spec(layout(_, _, _)), Config)
        ->  Type = layout
        ;   member(spec(container(_, _, _)), Config)
        ->  Type = container
        ;   member(spec(component(_, _)), Config)
        ->  Type = component
        ;   member(spec(page(_, _)), Config)
        ->  Type = page
        ;   Type = component  % Default
        ),
        component_registry:declare_component(ui, Name, Type, Config)
    ;   % Standalone mode - store locally
        assertz(local_ui_component(Name, Config)),
        format('Declared UI component (standalone): ~w~n', [Name])
    ).

:- dynamic local_ui_component/2.

%! compile_ui_component(+Name, +Options, -Code) is det
%  Compile a UI component to target code.
compile_ui_component(Name, Options, Code) :-
    (   current_predicate(component_registry:compile_component/4)
    ->  component_registry:compile_component(ui, Name, Options, Code)
    ;   % Standalone mode
        local_ui_component(Name, Config),
        compile_component(Name, Config, Options, Code)
    ).

% ============================================================================
% TESTING
% ============================================================================

test_ui_registry :-
    format('~n=== UI Registry Tests ===~n~n'),

    % Test 1: Initialize registry
    format('Test 1: Initialize registry...~n'),
    (   init_ui_registry
    ->  format('  PASS: Registry initialized~n')
    ;   format('  WARN: Registry init skipped (no component_registry)~n')
    ),

    % Test 2: Type info
    format('~nTest 2: Type info...~n'),
    type_info(Info),
    format('  Info: ~w~n', [Info]),
    format('  PASS: Type info available~n'),

    % Test 3: Validate config
    format('~nTest 3: Validate config...~n'),
    TestConfig = [
        spec(layout(stack, [spacing(16)], [
            component(text, [content("Hello")])
        ])),
        target(vue)
    ],
    (   validate_config(TestConfig)
    ->  format('  PASS: Config validation~n')
    ;   format('  FAIL: Config validation~n')
    ),

    % Test 4: Compile spec
    format('~nTest 4: Compile to Vue...~n'),
    Spec = layout(stack, [spacing(16)], [
        component(text, [content("Hello World")]),
        component(button, [label("Click Me"), on_click(handle_click)])
    ]),
    (   compile_spec(Spec, vue, Code)
    ->  format('  Generated code:~n~w~n', [Code]),
        (   sub_atom(Code, _, _, _, 'flex-direction')
        ->  format('  PASS: Vue code generated~n')
        ;   format('  PARTIAL: Code generated but may be placeholder~n')
        )
    ;   format('  FAIL: Compilation failed~n')
    ),

    % Test 5: Declare and compile component (standalone mode)
    format('~nTest 5: Declare and compile component...~n'),
    retractall(local_ui_component(_, _)),
    declare_ui_component(test_form, [
        spec(layout(stack, [spacing(12)], [
            component(text_input, [label("Name"), bind(name)]),
            component(button, [label("Submit"), variant(primary)])
        ])),
        target(vue)
    ]),
    (   compile_ui_component(test_form, [], FormCode)
    ->  format('  Generated form:~n~w~n', [FormCode]),
        format('  PASS: Component declared and compiled~n')
    ;   format('  FAIL: Could not compile component~n')
    ),

    % Cleanup
    retractall(local_ui_component(_, _)),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('UI Registry module loaded~n')
), now).
