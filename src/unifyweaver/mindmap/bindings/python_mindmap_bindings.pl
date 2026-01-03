% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% python_mindmap_bindings.pl - Python Bindings for Mind Map Operations
%
% Defines bindings that map Prolog mind map predicates to Python
% implementations for performance-critical operations.
%
% Usage:
%   ?- binding(python, force_layout/3, TargetName, Inputs, Outputs, Options).

:- module(python_mindmap_bindings, [
    init_mindmap_bindings/0,
    test_mindmap_bindings/0
]).

:- use_module('../../core/binding_registry').
:- use_module(library(lists)).

% ============================================================================
% LAYOUT ALGORITHM BINDINGS
% ============================================================================

%% init_mindmap_bindings
%
%  Initialize all mind map Python bindings.
%
init_mindmap_bindings :-
    % Layout algorithms
    declare_binding(python, force_layout/3,
        'unifyweaver.mindmap.layout.force_directed',
        [graph_dict, options_dict],
        [positions_dict],
        [import('unifyweaver.mindmap.layout'),
         pure, deterministic,
         description("Force-directed layout with spring-electric model")]),

    declare_binding(python, radial_layout/3,
        'unifyweaver.mindmap.layout.radial',
        [graph_dict, options_dict],
        [positions_dict],
        [import('unifyweaver.mindmap.layout'),
         pure, deterministic,
         description("Radial layout around central root node")]),

    declare_binding(python, hierarchical_layout/3,
        'unifyweaver.mindmap.layout.hierarchical',
        [graph_dict, options_dict],
        [positions_dict],
        [import('unifyweaver.mindmap.layout'),
         pure, deterministic,
         description("Hierarchical tree layout")]),

    declare_binding(python, grid_layout/3,
        'unifyweaver.mindmap.layout.grid',
        [graph_dict, options_dict],
        [positions_dict],
        [import('unifyweaver.mindmap.layout'),
         pure, deterministic,
         description("Grid-based layout")]),

    % Optimization passes
    declare_binding(python, remove_overlaps/3,
        'unifyweaver.mindmap.optimize.overlap_removal',
        [positions_dict, node_data_dict, options_dict],
        [positions_dict],
        [import('unifyweaver.mindmap.optimize'),
         pure, deterministic,
         description("Remove node overlaps by pushing apart")]),

    declare_binding(python, minimize_crossings/3,
        'unifyweaver.mindmap.optimize.crossing_minimization',
        [positions_dict, edges_list, options_dict],
        [positions_dict],
        [import('unifyweaver.mindmap.optimize'),
         pure, deterministic,
         description("Minimize edge crossings")]),

    declare_binding(python, adjust_spacing/3,
        'unifyweaver.mindmap.optimize.spacing',
        [positions_dict, options_dict],
        [positions_dict],
        [import('unifyweaver.mindmap.optimize'),
         pure, deterministic,
         description("Adjust node spacing for visual balance")]),

    % Rendering operations
    declare_binding(python, render_svg/3,
        'unifyweaver.mindmap.render.svg.render',
        [graph_dict, positions_dict, style_dict],
        [svg_string],
        [import('unifyweaver.mindmap.render.svg'),
         pure, deterministic,
         description("Render mind map to SVG")]),

    declare_binding(python, render_graphviz/3,
        'unifyweaver.mindmap.render.graphviz.render',
        [graph_dict, positions_dict, options_dict],
        [dot_string],
        [import('unifyweaver.mindmap.render.graphviz'),
         pure, deterministic,
         description("Render mind map to GraphViz DOT format")]),

    % Style operations
    declare_binding(python, resolve_style/3,
        'unifyweaver.mindmap.styling.resolve',
        [node_dict, theme_dict],
        [style_dict],
        [import('unifyweaver.mindmap.styling'),
         pure, deterministic,
         description("Resolve node style from theme cascade")]),

    declare_binding(python, apply_theme/2,
        'unifyweaver.mindmap.styling.apply_theme',
        [graph_dict, theme_name],
        [styled_graph_dict],
        [import('unifyweaver.mindmap.styling'),
         pure, deterministic,
         description("Apply theme to entire graph")]),

    % Import/export operations
    declare_binding(python, parse_smmx/1,
        'unifyweaver.mindmap.io.parse_smmx',
        [file_path],
        [graph_dict],
        [import('unifyweaver.mindmap.io'),
         effect(io), deterministic,
         description("Parse .smmx mind map file")]),

    declare_binding(python, parse_mm/1,
        'unifyweaver.mindmap.io.parse_mm',
        [file_path],
        [graph_dict],
        [import('unifyweaver.mindmap.io'),
         effect(io), deterministic,
         description("Parse FreeMind .mm file")]),

    declare_binding(python, export_smmx/2,
        'unifyweaver.mindmap.io.export_smmx',
        [graph_dict, file_path],
        [],
        [import('unifyweaver.mindmap.io'),
         effect(io), deterministic,
         description("Export to .smmx format")]),

    declare_binding(python, export_mm/2,
        'unifyweaver.mindmap.io.export_mm',
        [graph_dict, file_path],
        [],
        [import('unifyweaver.mindmap.io'),
         effect(io), deterministic,
         description("Export to FreeMind .mm format")]),

    % NumPy-accelerated operations
    declare_binding(python, numpy_force_layout/4,
        'unifyweaver.mindmap.layout.numpy_force',
        [nodes_array, edges_array, options_dict],
        [positions_array],
        [import('unifyweaver.mindmap.layout.numpy_force'),
         import(numpy),
         pure, deterministic,
         description("NumPy-accelerated force-directed layout"),
         acceleration(numpy)]),

    declare_binding(python, scipy_optimize_layout/4,
        'unifyweaver.mindmap.layout.scipy_optimize',
        [positions_array, constraints_list, options_dict],
        [optimized_positions_array],
        [import('unifyweaver.mindmap.layout.scipy_optimize'),
         import(scipy),
         pure, deterministic,
         description("SciPy-based layout optimization"),
         acceleration(scipy)]),

    format('Mind map Python bindings initialized~n', []).

% ============================================================================
% EFFECT DECLARATIONS
% ============================================================================

:- initialization((
    % Declare effects for mind map predicates
    declare_effect(force_layout/3, [pure, deterministic]),
    declare_effect(radial_layout/3, [pure, deterministic]),
    declare_effect(hierarchical_layout/3, [pure, deterministic]),
    declare_effect(remove_overlaps/3, [pure, deterministic]),
    declare_effect(render_svg/3, [pure, deterministic]),
    declare_effect(parse_smmx/1, [effect(io), nondeterministic]),
    declare_effect(parse_mm/1, [effect(io), nondeterministic]),
    declare_effect(export_smmx/2, [effect(io)]),
    declare_effect(export_mm/2, [effect(io)])
), now).

% ============================================================================
% BIDIRECTIONAL PREDICATES
% ============================================================================

:- initialization((
    % Some predicates support multiple modes
    declare_bidirectional(force_layout/3, [
        mode(compute, [in, in, out]),      % Compute layout
        mode(verify, [in, in, in])         % Verify layout
    ]),

    declare_bidirectional(resolve_style/3, [
        mode(resolve, [in, in, out]),      % Resolve style
        mode(check, [in, in, in])          % Check style match
    ])
), now).

% ============================================================================
% MODE-SPECIFIC BINDINGS
% ============================================================================

:- initialization((
    % Different implementations for different modes
    declare_binding_mode(python, force_layout/3, compute,
        'unifyweaver.mindmap.layout.force_directed',
        [description("Compute force-directed layout")]),

    declare_binding_mode(python, force_layout/3, verify,
        'unifyweaver.mindmap.layout.verify_layout',
        [description("Verify layout meets constraints")])
), now).

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_bindings :-
    format('~n=== Mind Map Bindings Tests ===~n~n'),

    % Initialize bindings
    init_mindmap_bindings,

    % Test 1: Check layout binding exists
    format('Test 1: Layout binding exists...~n'),
    (   binding(python, force_layout/3, Name, _, _, _)
    ->  format('  PASS: force_layout/3 bound to ~w~n', [Name])
    ;   format('  FAIL: force_layout/3 not found~n')
    ),

    % Test 2: Check pure bindings
    format('~nTest 2: Pure bindings...~n'),
    (   is_pure_binding(python, force_layout/3)
    ->  format('  PASS: force_layout/3 is pure~n')
    ;   format('  FAIL: force_layout/3 should be pure~n')
    ),

    % Test 3: Check IO effect binding
    format('~nTest 3: IO effect binding...~n'),
    (   \+ is_pure_binding(python, parse_smmx/1)
    ->  format('  PASS: parse_smmx/1 has IO effect~n')
    ;   format('  FAIL: parse_smmx/1 should have IO effect~n')
    ),

    % Test 4: List all mind map bindings
    format('~nTest 4: List bindings for python...~n'),
    bindings_for_target(python, Bindings),
    findall(P, (member(binding(_, P, _, _, _, _), Bindings),
                P = _/_, functor(P, N, _),
                (sub_atom(N, _, _, _, layout) ; sub_atom(N, _, _, _, render)
                 ; sub_atom(N, _, _, _, style) ; sub_atom(N, _, _, _, smmx)
                 ; sub_atom(N, _, _, _, mm) ; sub_atom(N, _, _, _, overlap))),
            MindmapPreds),
    length(MindmapPreds, NumMM),
    format('  Found ~w mind map related bindings~n', [NumMM]),

    % Test 5: Check imports
    format('~nTest 5: Binding imports...~n'),
    binding_imports(python, Imports),
    (   member('unifyweaver.mindmap.layout', Imports)
    ->  format('  PASS: Layout module imported~n')
    ;   format('  FAIL: Layout module not found in imports~n')
    ),

    % Test 6: Bidirectional predicate
    format('~nTest 6: Bidirectional predicates...~n'),
    (   bidirectional(force_layout/3, Modes),
        member(mode(compute, _), Modes)
    ->  format('  PASS: force_layout/3 has compute mode~n')
    ;   format('  FAIL: force_layout/3 missing compute mode~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    init_mindmap_bindings
), now).
