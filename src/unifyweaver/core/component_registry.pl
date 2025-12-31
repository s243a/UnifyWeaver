% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% component_registry.pl - Unified Component Registry
%
% Provides a common framework for registering, configuring, querying,
% and validating components across multiple categories (runtime, source, binding).
%
% See: docs/proposals/COMPONENT_REGISTRY.md

:- module(component_registry, [
    % Category management
    define_category/3,              % +Category, +Description, +Options
    category/3,                     % ?Category, ?Description, ?Options
    list_categories/1,              % -Categories

    % Type registration
    register_component_type/4,      % +Category, +Type, +Module, +Options
    component_type/4,               % ?Category, ?Type, ?Module, ?Options
    list_types/2,                   % +Category, -Types

    % Instance management
    declare_component/4,            % +Category, +Name, +Type, +Config
    retract_component/2,            % +Category, +Name
    component/4,                    % ?Category, ?Name, ?Type, ?Config
    component_metadata/3,           % +Category, +Name, -Metadata
    list_components/2,              % +Category, -Names
    components_of_type/3,           % +Category, +Type, -Names

    % Initialization
    init_component/2,               % +Category, +Name
    ensure_component_ready/2,       % +Category, +Name
    init_eager_components/0,        % Initialize all eager components
    component_initialized/2,        % ?Category, ?Name

    % Invocation
    invoke_component/4,             % +Category, +Name, +Input, -Output
    compile_component/4,            % +Category, +Name, +Options, -Code

    % Validation
    validate_component/3,           % +Category, +Name, +Config

    % Events
    on_component_ready/1,           % +Name - hook for component ready events

    % Testing
    test_component_registry/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_category/3.           % stored_category(Category, Description, Options)
:- dynamic stored_type/4.               % stored_type(Category, Type, Module, Options)
:- dynamic stored_component/5.          % stored_component(Category, Name, Type, Config, Metadata)
:- dynamic component_init_state/2.      % component_init_state(Name, State) - State: initialized | pending

% ============================================================================
% CATEGORY MANAGEMENT
% ============================================================================

%% define_category(+Category, +Description, +Options)
%
%  Define a new component category.
%
%  Options:
%    - requires_compilation(bool) - whether instances compile to code
%    - singleton(bool) - only one instance per type allowed
%    - validation_hook(Pred) - custom validation predicate
%
%  @param Category    atom - category identifier (runtime, source, binding)
%  @param Description string - human-readable description
%  @param Options     list - category options
%
define_category(Category, Description, Options) :-
    atom(Category),
    (   stored_category(Category, _, _)
    ->  retract(stored_category(Category, _, _))
    ;   true
    ),
    assertz(stored_category(Category, Description, Options)),
    format('Defined category: ~w~n', [Category]).

%% category(?Category, ?Description, ?Options)
%
%  Query category definitions.
%
category(Category, Description, Options) :-
    stored_category(Category, Description, Options).

%% list_categories(-Categories)
%
%  Get list of all defined categories.
%
list_categories(Categories) :-
    findall(Cat, stored_category(Cat, _, _), Categories).

% ============================================================================
% TYPE REGISTRATION
% ============================================================================

%% register_component_type(+Category, +Type, +Module, +Options)
%
%  Register a component type within a category.
%
%  The Module must export:
%    - type_info(-Info) - metadata about the type
%    - validate_config(+Config) - configuration validation
%    - init_component(+Name, +Config) - initialization
%    - invoke_component(+Name, +Config, +Input, -Output) - for runtime
%    - compile_component(+Name, +Config, +Options, -Code) - for compiled
%
%  @param Category  atom - the category this type belongs to
%  @param Type      atom - type identifier
%  @param Module    atom - Prolog module implementing the type
%  @param Options   list - type options (description, requires, etc.)
%
register_component_type(Category, Type, Module, Options) :-
    atom(Category),
    atom(Type),
    atom(Module),
    is_list(Options),
    % Verify category exists
    (   stored_category(Category, _, _)
    ->  true
    ;   format('Warning: Category ~w not defined, registering type anyway~n', [Category])
    ),
    % Remove existing registration if any
    retractall(stored_type(Category, Type, _, _)),
    assertz(stored_type(Category, Type, Module, Options)),
    format('Registered type: ~w/~w -> ~w~n', [Category, Type, Module]).

%% component_type(?Category, ?Type, ?Module, ?Options)
%
%  Query type registrations.
%
component_type(Category, Type, Module, Options) :-
    stored_type(Category, Type, Module, Options).

%% list_types(+Category, -Types)
%
%  Get list of all types registered in a category.
%
list_types(Category, Types) :-
    findall(Type, stored_type(Category, Type, _, _), Types).

% ============================================================================
% INSTANCE MANAGEMENT
% ============================================================================

%% declare_component(+Category, +Name, +Type, +Config)
%
%  Declare a configured instance of a component.
%
%  Config options common to all components:
%    - initialization(eager|lazy|manual) - when to initialize (default: lazy)
%    - depends([Name1, Name2, ...]) - dependencies that must load first
%
%  @param Category  atom - the category
%  @param Name      atom - unique instance name
%  @param Type      atom - the component type
%  @param Config    list - configuration options
%
declare_component(Category, Name, Type, Config) :-
    atom(Category),
    atom(Name),
    atom(Type),
    is_list(Config),
    % Verify type exists
    (   stored_type(Category, Type, _, _)
    ->  true
    ;   format('Error: Type ~w not registered in category ~w~n', [Type, Category]),
        fail
    ),
    % Validate configuration
    (   validate_component(Category, Type, Config)
    ->  true
    ;   format('Warning: Configuration validation failed for ~w~n', [Name])
    ),
    % Extract metadata
    extract_metadata(Config, Metadata),
    % Remove existing instance if any
    retractall(stored_component(Category, Name, _, _, _)),
    retractall(component_init_state(Name, _)),
    assertz(stored_component(Category, Name, Type, Config, Metadata)),
    assertz(component_init_state(Name, pending)),
    format('Declared component: ~w/~w (type: ~w)~n', [Category, Name, Type]),
    % Record dependencies
    (   member(depends(Deps), Config)
    ->  true
    ;   Deps = []
    ),
    retractall(component_depends(Name, _)),
    assertz(component_depends(Name, Deps)).

:- dynamic component_depends/2.  % component_depends(Name, DependencyList)

%% retract_component(+Category, +Name)
%
%  Remove a component instance.
%
retract_component(Category, Name) :-
    retractall(stored_component(Category, Name, _, _, _)),
    retractall(component_init_state(Name, _)),
    retractall(component_depends(Name, _)).

%% component(?Category, ?Name, ?Type, ?Config)
%
%  Query component instances.
%
component(Category, Name, Type, Config) :-
    stored_component(Category, Name, Type, Config, _).

%% component_metadata(+Category, +Name, -Metadata)
%
%  Get computed metadata for a component.
%
component_metadata(Category, Name, Metadata) :-
    stored_component(Category, Name, _, _, Metadata).

%% list_components(+Category, -Names)
%
%  Get list of all component names in a category.
%
list_components(Category, Names) :-
    findall(Name, stored_component(Category, Name, _, _, _), Names).

%% components_of_type(+Category, +Type, -Names)
%
%  Get list of component names of a specific type.
%
components_of_type(Category, Type, Names) :-
    findall(Name, stored_component(Category, Name, Type, _, _), Names).

%% extract_metadata(+Config, -Metadata)
%
%  Extract metadata dict from configuration.
%
extract_metadata(Config, Metadata) :-
    (   member(initialization(Init), Config) -> true ; Init = lazy ),
    (   member(depends(Deps), Config) -> true ; Deps = [] ),
    Metadata = metadata{initialization: Init, depends: Deps}.

% ============================================================================
% INITIALIZATION
% ============================================================================

%% component_initialized(?Category, ?Name)
%
%  Check if a component is initialized.
%
component_initialized(Category, Name) :-
    stored_component(Category, Name, _, _, _),
    component_init_state(Name, initialized).

%% init_component(+Category, +Name)
%
%  Initialize a specific component (does not check dependencies).
%
init_component(Category, Name) :-
    stored_component(Category, Name, Type, Config, _),
    stored_type(Category, Type, Module, _),
    (   component_init_state(Name, initialized)
    ->  true  % Already initialized
    ;   % Call module's init_component
        (   catch(Module:init_component(Name, Config), Error,
                  (format('Error initializing ~w: ~w~n', [Name, Error]), fail))
        ->  retract(component_init_state(Name, _)),
            assertz(component_init_state(Name, initialized)),
            on_component_ready(Name)
        ;   format('Failed to initialize component: ~w~n', [Name]),
            fail
        )
    ).

%% ensure_component_ready(+Category, +Name)
%
%  Ensure a component and its dependencies are initialized.
%
ensure_component_ready(Category, Name) :-
    stored_component(Category, Name, _, Config, _),
    % First, ensure dependencies are ready
    (   member(depends(Deps), Config)
    ->  maplist(ensure_dependency_ready, Deps)
    ;   true
    ),
    % Then initialize this component
    init_component(Category, Name).

%% ensure_dependency_ready(+Name)
%
%  Ensure a dependency is ready (find its category automatically).
%
ensure_dependency_ready(Name) :-
    stored_component(Category, Name, _, _, _),
    ensure_component_ready(Category, Name).

%% init_eager_components
%
%  Initialize all components marked as eager.
%
init_eager_components :-
    forall(
        (stored_component(Category, Name, _, Config, _),
         member(initialization(eager), Config)),
        (   ensure_component_ready(Category, Name)
        ->  true
        ;   format('Warning: Failed to initialize eager component: ~w~n', [Name])
        )
    ).

%% on_component_ready(+Name)
%
%  Hook called when a component finishes initialization.
%  Can be extended by other modules.
%
on_component_ready(Name) :-
    format('Component ready: ~w~n', [Name]).

% ============================================================================
% INVOCATION
% ============================================================================

%% invoke_component(+Category, +Name, +Input, -Output)
%
%  Invoke a component. Handles lazy initialization.
%
invoke_component(Category, Name, Input, Output) :-
    % Ensure initialized (lazy init)
    ensure_component_ready(Category, Name),
    % Get type and module
    stored_component(Category, Name, Type, Config, _),
    stored_type(Category, Type, Module, _),
    % Call module's invoke_component
    Module:invoke_component(Name, Config, Input, Output).

%% compile_component(+Category, +Name, +Options, -Code)
%
%  Compile a component definition to target code.
%
compile_component(Category, Name, Options, Code) :-
    stored_component(Category, Name, Type, Config, _),
    stored_type(Category, Type, Module, _),
    Module:compile_component(Name, Config, Options, Code).

% ============================================================================
% VALIDATION
% ============================================================================

%% validate_component(+Category, +Type, +Config)
%
%  Validate component configuration.
%
validate_component(Category, Type, Config) :-
    % Get type module
    (   stored_type(Category, Type, Module, _)
    ->  true
    ;   format('Error: Type ~w not found in category ~w~n', [Type, Category]),
        fail
    ),
    % Type-specific validation
    (   catch(Module:validate_config(Config), _, true)
    ->  true
    ;   true  % Module may not have validate_config
    ),
    % Validate dependencies exist
    validate_dependencies(Config),
    % Category-specific validation
    category_validate(Category, Config).

%% validate_dependencies(+Config)
%
%  Check that declared dependencies exist.
%
validate_dependencies(Config) :-
    (   member(depends(Deps), Config)
    ->  forall(member(D, Deps),
               (   stored_component(_, D, _, _, _)
               ->  true
               ;   format('Warning: Dependency ~w not found~n', [D])
               ))
    ;   true
    ).

%% category_validate(+Category, +Config)
%
%  Category-level validation. Override for specific categories.
%
category_validate(runtime, _Config) :- !.
category_validate(source, _Config) :- !.
category_validate(binding, _Config) :- !.
category_validate(_, _Config).  % Default: no validation

% ============================================================================
% TESTING
% ============================================================================

test_component_registry :-
    format('~n=== Component Registry Tests ===~n~n'),

    % Test 1: Define category
    format('Test 1: Define category...~n'),
    define_category(test_category, "Test category", [requires_compilation(false)]),
    (   category(test_category, _, _)
    ->  format('  PASS: Category defined~n')
    ;   format('  FAIL: Category not found~n')
    ),

    % Test 2: Register type (mock module)
    format('~nTest 2: Register type...~n'),
    register_component_type(test_category, test_type, test_type_module, [
        description("Test type")
    ]),
    (   component_type(test_category, test_type, test_type_module, _)
    ->  format('  PASS: Type registered~n')
    ;   format('  FAIL: Type not found~n')
    ),

    % Test 3: List types
    format('~nTest 3: List types...~n'),
    list_types(test_category, Types),
    format('  Types: ~w~n', [Types]),
    (   member(test_type, Types)
    ->  format('  PASS: Type in list~n')
    ;   format('  FAIL: Type not in list~n')
    ),

    % Cleanup
    retractall(stored_category(test_category, _, _)),
    retractall(stored_type(test_category, _, _, _)),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    % Define the runtime category by default
    define_category(runtime, "Execution-time components", [
        requires_compilation(false),
        singleton(false)
    ])
), now).
