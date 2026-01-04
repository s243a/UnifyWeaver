% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% mindmap_components.pl - Mind Map Component Registry Integration
%
% This module integrates the mind map system with UnifyWeaver's component
% registry, defining categories and registering built-in components.
%
% Categories:
%   - mindmap_layout    - Layout algorithms (radial, force_directed, etc.)
%   - mindmap_optimizer - Layout optimizers (overlap_removal, crossing_min, etc.)
%   - mindmap_renderer  - Output renderers (svg, smmx, graph_interactive, etc.)

:- module(mindmap_components, [
    % Category definitions
    define_mindmap_categories/0,

    % Component registration
    register_mindmap_layout/3,      % register_mindmap_layout(+Name, +Module, +Options)
    register_mindmap_optimizer/3,   % register_mindmap_optimizer(+Name, +Module, +Options)
    register_mindmap_renderer/3,    % register_mindmap_renderer(+Name, +Module, +Options)

    % Query predicates
    list_mindmap_layouts/1,         % list_mindmap_layouts(-Layouts)
    list_mindmap_optimizers/1,      % list_mindmap_optimizers(-Optimizers)
    list_mindmap_renderers/1,       % list_mindmap_renderers(-Renderers)

    % Component existence checks (supports component(Name) syntax)
    layout_exists/1,                % layout_exists(+NameOrComponent)
    optimizer_exists/1,             % optimizer_exists(+NameOrComponent)
    renderer_exists/1,              % renderer_exists(+NameOrComponent)

    % Component invocation
    invoke_layout/4,                % invoke_layout(+Name, +Graph, +Options, -Positions)
    invoke_optimizer/4,             % invoke_optimizer(+Name, +Positions, +Options, -OptimizedPositions)
    invoke_renderer/4,              % invoke_renderer(+Name, +Data, +Options, -Output)

    % Initialization
    init_mindmap_components/0,

    % Testing
    test_mindmap_components/0
]).

:- use_module(library(lists)).

% Conditionally load component_registry if available
:- if(exists_source(library('../../core/component_registry'))).
:- use_module('../core/component_registry').
:- endif.

% ============================================================================
% CATEGORY DEFINITIONS
% ============================================================================

%% define_mindmap_categories
%
%  Define the mind map component categories.
%  This should be called during module initialization.
%
define_mindmap_categories :-
    % Define layout category
    (   catch(define_category(mindmap_layout,
            "Mind map layout algorithms",
            [requires_compilation(false), singleton(false)]), _, true)
    ->  true ; true),

    % Define optimizer category
    (   catch(define_category(mindmap_optimizer,
            "Layout optimization passes",
            [requires_compilation(false), singleton(false)]), _, true)
    ->  true ; true),

    % Define renderer category
    (   catch(define_category(mindmap_renderer,
            "Output format renderers",
            [requires_compilation(true), singleton(false)]), _, true)
    ->  true ; true).

% ============================================================================
% COMPONENT REGISTRATION
% ============================================================================

%% register_mindmap_layout(+Name, +Module, +Options)
%
%  Register a layout algorithm component.
%
%  The Module should export:
%    - type_info(-Info)
%    - validate_config(+Config)
%    - init_component(+Name, +Config)
%    - compute_layout(+Graph, +Options, -Positions)
%
%  Options:
%    - description(Text)  - Human-readable description
%    - parameters(List)   - Configurable parameters
%    - complexity(Big-O)  - Algorithmic complexity
%
register_mindmap_layout(Name, Module, Options) :-
    atom(Name),
    atom(Module),
    is_list(Options),
    (   catch(register_component_type(mindmap_layout, Name, Module, Options), _, fail)
    ->  true
    ;   % Fallback: store locally if component_registry not available
        assertz(local_layout(Name, Module, Options))
    ).

%% register_mindmap_optimizer(+Name, +Module, +Options)
%
%  Register a layout optimizer component.
%
%  The Module should export:
%    - type_info(-Info)
%    - validate_config(+Config)
%    - init_component(+Name, +Config)
%    - optimize(+Positions, +Graph, +Options, -OptimizedPositions)
%
register_mindmap_optimizer(Name, Module, Options) :-
    atom(Name),
    atom(Module),
    is_list(Options),
    (   catch(register_component_type(mindmap_optimizer, Name, Module, Options), _, fail)
    ->  true
    ;   assertz(local_optimizer(Name, Module, Options))
    ).

%% register_mindmap_renderer(+Name, +Module, +Options)
%
%  Register an output renderer component.
%
%  The Module should export:
%    - type_info(-Info)
%    - validate_config(+Config)
%    - init_component(+Name, +Config)
%    - compile_component(+Name, +Config, +Options, -Code)
%    - render(+Graph, +Positions, +Options, -Output)
%
register_mindmap_renderer(Name, Module, Options) :-
    atom(Name),
    atom(Module),
    is_list(Options),
    (   catch(register_component_type(mindmap_renderer, Name, Module, Options), _, fail)
    ->  true
    ;   assertz(local_renderer(Name, Module, Options))
    ).

% Local storage for when component_registry is not available
:- dynamic local_layout/3.
:- dynamic local_optimizer/3.
:- dynamic local_renderer/3.

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% layout_exists(+Name)
%
%  Check if a layout component exists by name.
%  Supports component(Name) syntax for explicit component references.
%
layout_exists(component(Name)) :-
    !,
    layout_exists(Name).
layout_exists(Name) :-
    atom(Name),
    get_layout_module(Name, _).

%% optimizer_exists(+Name)
%
%  Check if an optimizer component exists by name.
%
optimizer_exists(component(Name)) :-
    !,
    optimizer_exists(Name).
optimizer_exists(Name) :-
    atom(Name),
    get_optimizer_module(Name, _).

%% renderer_exists(+Name)
%
%  Check if a renderer component exists by name.
%
renderer_exists(component(Name)) :-
    !,
    renderer_exists(Name).
renderer_exists(Name) :-
    atom(Name),
    get_renderer_module(Name, _).

%% list_mindmap_layouts(-Layouts)
%
%  Get list of all registered layout algorithms.
%
list_mindmap_layouts(Layouts) :-
    (   catch(list_types(mindmap_layout, Layouts), _, fail)
    ->  true
    ;   findall(Name, local_layout(Name, _, _), Layouts)
    ).

%% list_mindmap_optimizers(-Optimizers)
%
%  Get list of all registered optimizers.
%
list_mindmap_optimizers(Optimizers) :-
    (   catch(list_types(mindmap_optimizer, Optimizers), _, fail)
    ->  true
    ;   findall(Name, local_optimizer(Name, _, _), Optimizers)
    ).

%% list_mindmap_renderers(-Renderers)
%
%  Get list of all registered renderers.
%
list_mindmap_renderers(Renderers) :-
    (   catch(list_types(mindmap_renderer, Renderers), _, fail)
    ->  true
    ;   findall(Name, local_renderer(Name, _, _), Renderers)
    ).

% ============================================================================
% COMPONENT INVOCATION
% ============================================================================

%% invoke_layout(+Name, +Graph, +Options, -Positions)
%
%  Invoke a layout algorithm by name.
%
%  @param Name      atom - registered layout name
%  @param Graph     term - graph(Nodes, Edges) structure
%  @param Options   list - algorithm options
%  @param Positions list - list of position(Id, X, Y) terms
%
invoke_layout(Name, Graph, Options, Positions) :-
    get_layout_module(Name, Module),
    !,
    Module:compute_layout(Graph, Options, Positions).
invoke_layout(Name, _Graph, _Options, []) :-
    format('Error: Layout algorithm ~w not found~n', [Name]).

%% invoke_optimizer(+Name, +Positions, +Options, -OptimizedPositions)
%
%  Invoke a layout optimizer by name.
%
invoke_optimizer(Name, Positions, Options, OptimizedPositions) :-
    get_optimizer_module(Name, Module),
    !,
    Module:optimize(Positions, Options, OptimizedPositions).
invoke_optimizer(Name, Positions, _Options, Positions) :-
    format('Warning: Optimizer ~w not found, returning unchanged positions~n', [Name]).

%% invoke_renderer(+Name, +Data, +Options, -Output)
%
%  Invoke a renderer by name.
%
%  @param Name    atom - registered renderer name
%  @param Data    term - render_data(Graph, Positions, Styles)
%  @param Options list - renderer options
%  @param Output  term - rendered output (format depends on renderer)
%
invoke_renderer(Name, Data, Options, Output) :-
    get_renderer_module(Name, Module),
    !,
    Module:render(Data, Options, Output).
invoke_renderer(Name, _Data, _Options, '') :-
    format('Error: Renderer ~w not found~n', [Name]).

% Helper predicates to get module for a component
get_layout_module(Name, Module) :-
    (   catch(component_type(mindmap_layout, Name, Module, _), _, fail)
    ->  true
    ;   local_layout(Name, Module, _)
    ).

get_optimizer_module(Name, Module) :-
    (   catch(component_type(mindmap_optimizer, Name, Module, _), _, fail)
    ->  true
    ;   local_optimizer(Name, Module, _)
    ).

get_renderer_module(Name, Module) :-
    (   catch(component_type(mindmap_renderer, Name, Module, _), _, fail)
    ->  true
    ;   local_renderer(Name, Module, _)
    ).

% ============================================================================
% BUILT-IN COMPONENT REGISTRATION
% ============================================================================

%% register_builtin_layouts
%
%  Register the built-in layout algorithms.
%
register_builtin_layouts :-
    % Radial layout
    register_mindmap_layout(radial, mindmap_layout_radial, [
        description("Radial layout with root at center"),
        parameters([
            center(X, Y) - "Center point (default 500, 500)",
            base_radius(R) - "Base radius for first level (default 150)",
            min_spacing(S) - "Minimum spacing between nodes (default 80)"
        ]),
        complexity("O(n)")
    ]),

    % Force-directed layout
    register_mindmap_layout(force_directed, mindmap_layout_force, [
        description("Force-directed physics simulation"),
        parameters([
            iterations(N) - "Number of simulation iterations (default 300)",
            repulsion(R) - "Repulsion force strength (default 100000)",
            attraction(A) - "Attraction force to parent (default 0.001)",
            min_distance(D) - "Minimum distance between nodes (default 120)",
            damping(D) - "Velocity damping factor (default 0.8)"
        ]),
        complexity("O(n² × iterations)")
    ]),

    % Hierarchical layout
    register_mindmap_layout(hierarchical, mindmap_layout_hierarchical, [
        description("Tree-like hierarchical layout"),
        parameters([
            level_spacing(S) - "Vertical spacing between levels (default 100)",
            sibling_spacing(S) - "Horizontal spacing between siblings (default 50)",
            direction(D) - "Layout direction: top_down, left_right (default top_down)"
        ]),
        complexity("O(n)")
    ]).

%% register_builtin_optimizers
%
%  Register the built-in layout optimizers.
%
register_builtin_optimizers :-
    % Overlap removal
    register_mindmap_optimizer(overlap_removal, mindmap_opt_overlap, [
        description("Remove overlapping nodes"),
        parameters([
            min_distance(D) - "Minimum distance between node centers (default 50)",
            iterations(N) - "Maximum iterations (default 100)"
        ]),
        complexity("O(n²)")
    ]),

    % Crossing minimization
    register_mindmap_optimizer(crossing_minimization, mindmap_opt_crossing, [
        description("Minimize edge crossings"),
        parameters([
            max_swaps(N) - "Maximum swap operations (default 1000)",
            temperature(T) - "Simulated annealing temperature (default 1.0)"
        ]),
        complexity("O(e² + n×e)")
    ]),

    % Spacing adjustment
    register_mindmap_optimizer(spacing_adjustment, mindmap_opt_spacing, [
        description("Adjust spacing for visual balance"),
        parameters([
            target_density(D) - "Target node density (default 0.3)",
            preserve_structure(B) - "Preserve relative positions (default true)"
        ]),
        complexity("O(n)")
    ]).

%% register_builtin_renderers
%
%  Register the built-in output renderers.
%
register_builtin_renderers :-
    % SVG renderer
    register_mindmap_renderer(svg, mindmap_render_svg, [
        description("Static SVG output"),
        parameters([
            width(W) - "SVG width (default 1000)",
            height(H) - "SVG height (default 800)",
            node_shape(S) - "Node shape: ellipse, rectangle, diamond",
            edge_style(S) - "Edge style: straight, bezier, orthogonal"
        ]),
        file_extension(".svg"),
        mime_type("image/svg+xml")
    ]),

    % Native format renderer (.smmx)
    register_mindmap_renderer(smmx, mindmap_render_smmx, [
        description("Native .smmx format"),
        parameters([
            include_positions(B) - "Include computed positions (default true)",
            compression(B) - "ZIP compress output (default true)"
        ]),
        file_extension(".smmx"),
        mime_type("application/zip")
    ]),

    % Native format renderer (.mm)
    register_mindmap_renderer(mm, mindmap_render_mm, [
        description("Native .mm format"),
        parameters([
            include_positions(B) - "Include computed positions (default true)"
        ]),
        file_extension(".mm"),
        mime_type("application/xml")
    ]),

    % Interactive graph renderer
    register_mindmap_renderer(graph_interactive, mindmap_render_interactive, [
        description("Interactive React component"),
        parameters([
            library(L) - "Graph library to use (default auto)",
            include_controls(B) - "Include zoom/pan controls (default true)",
            enable_editing(B) - "Allow node editing (default false)"
        ]),
        file_extension(".tsx"),
        mime_type("text/typescript")
    ]),

    % GraphViz DOT renderer
    register_mindmap_renderer(graphviz, mindmap_render_graphviz, [
        description("GraphViz DOT format"),
        parameters([
            layout_engine(E) - "GraphViz engine: dot, neato, fdp, etc.",
            rankdir(D) - "Rank direction: TB, LR, BT, RL"
        ]),
        file_extension(".dot"),
        mime_type("text/vnd.graphviz")
    ]).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_mindmap_components
%
%  Initialize the mind map component system.
%  Defines categories and registers built-in components.
%
init_mindmap_components :-
    define_mindmap_categories,
    register_builtin_layouts,
    register_builtin_optimizers,
    register_builtin_renderers,
    format('Mind map components initialized~n', []).

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_components :-
    format('~n=== Mind Map Components Tests ===~n~n'),

    % Initialize
    init_mindmap_components,

    % Test 1: List layouts
    format('Test 1: List layouts...~n'),
    list_mindmap_layouts(Layouts),
    format('  Layouts: ~w~n', [Layouts]),
    (   member(radial, Layouts)
    ->  format('  PASS: radial layout registered~n')
    ;   format('  FAIL: radial layout not found~n')
    ),

    % Test 2: List optimizers
    format('~nTest 2: List optimizers...~n'),
    list_mindmap_optimizers(Optimizers),
    format('  Optimizers: ~w~n', [Optimizers]),
    (   member(overlap_removal, Optimizers)
    ->  format('  PASS: overlap_removal registered~n')
    ;   format('  FAIL: overlap_removal not found~n')
    ),

    % Test 3: List renderers
    format('~nTest 3: List renderers...~n'),
    list_mindmap_renderers(Renderers),
    format('  Renderers: ~w~n', [Renderers]),
    (   member(svg, Renderers)
    ->  format('  PASS: svg renderer registered~n')
    ;   format('  FAIL: svg renderer not found~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    % Delay initialization to allow component_registry to load first
    init_mindmap_components
), now).
