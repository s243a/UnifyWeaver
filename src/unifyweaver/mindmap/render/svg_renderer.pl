% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% svg_renderer.pl - SVG Renderer for Mind Maps
%
% Generates static SVG output from positioned mind map data.
% Supports various node shapes, edge styles, and theming.
%
% Usage:
%   ?- render_mindmap_svg(Nodes, Edges, Positions, Options, SVG).

:- module(mindmap_render_svg, [
    % Component interface
    type_info/1,
    validate_config/1,
    init_component/2,
    compile_component/4,
    render/3,

    % Direct API
    render_mindmap_svg/5,           % render_mindmap_svg(+Nodes, +Edges, +Positions, +Options, -SVG)
    render_node_svg/4,              % render_node_svg(+Node, +Position, +Options, -SVGElement)
    render_edge_svg/5,              % render_edge_svg(+Edge, +Pos1, +Pos2, +Options, -SVGElement)

    % Style helpers
    node_style/3,                   % node_style(+NodeType, +Theme, -StyleAttrs)
    edge_style/3,                   % edge_style(+EdgeType, +Theme, -StyleAttrs)

    % Testing
    test_svg_renderer/0
]).

:- use_module(library(lists)).

% ============================================================================
% COMPONENT INTERFACE
% ============================================================================

%% type_info(-Info)
%
%  Component type information.
%
type_info(info{
    name: svg,
    category: mindmap_renderer,
    description: "Static SVG output renderer",
    version: "1.0.0",
    file_extension: ".svg",
    mime_type: "image/svg+xml",
    parameters: [
        width - "SVG width in pixels (default 1000)",
        height - "SVG height in pixels (default 800)",
        background - "Background color (default #ffffff)",
        node_shape - "Node shape: ellipse, rectangle, diamond (default ellipse)",
        edge_style - "Edge style: straight, bezier (default straight)",
        theme - "Color theme: light, dark, colorful (default light)",
        include_labels - "Include text labels (default true)",
        font_family - "Font family (default sans-serif)",
        font_size - "Base font size (default 12)"
    ]
}).

%% validate_config(+Config)
%
%  Validate renderer configuration.
%
validate_config(Config) :-
    is_list(Config),
    (   member(width(W), Config)
    ->  integer(W), W > 0
    ;   true
    ),
    (   member(height(H), Config)
    ->  integer(H), H > 0
    ;   true
    ).

%% init_component(+Name, +Config)
%
%  Initialize the renderer.
%
init_component(_Name, _Config).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile renderer to code (returns SVG template).
%
compile_component(_Name, _Config, _Options, Code) :-
    Code = '<!-- SVG Mind Map Template -->'.

%% render(+Data, +Options, -Output)
%
%  Main render entry point.
%
%  @param Data    term - render_data(Nodes, Edges, Positions, Styles)
%  @param Options list - render options
%  @param Output  atom - SVG string
%
render(render_data(Nodes, Edges, Positions, _Styles), Options, Output) :-
    render_mindmap_svg(Nodes, Edges, Positions, Options, Output).

% ============================================================================
% SVG RENDERING
% ============================================================================

%% render_mindmap_svg(+Nodes, +Edges, +Positions, +Options, -SVG)
%
%  Render a complete mind map to SVG.
%
%  @param Nodes     list - list of node(Id, Props)
%  @param Edges     list - list of edge(From, To, Props)
%  @param Positions list - list of position(Id, X, Y)
%  @param Options   list - rendering options
%  @param SVG       atom - complete SVG document
%
render_mindmap_svg(Nodes, Edges, Positions, Options, SVG) :-
    % Get options with defaults
    option_or_default(width, Options, 1000, Width),
    option_or_default(height, Options, 800, Height),
    option_or_default(background, Options, '#ffffff', Background),
    option_or_default(theme, Options, light, Theme),
    option_or_default(margin, Options, 50, Margin),

    % Auto-calculate viewBox if positions exceed default
    calculate_viewbox(Positions, Margin, VBX, VBY, VBW, VBH),

    % Build position lookup
    build_position_lookup(Positions, PosLookup),

    % Render edges (underneath nodes)
    render_edges_svg(Edges, PosLookup, Options, Theme, EdgesContent),

    % Render nodes
    render_nodes_svg(Nodes, PosLookup, Options, Theme, NodesContent),

    % Build complete SVG document
    format(atom(SVG),
'<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="~w" height="~w"
     viewBox="~w ~w ~w ~w">
  <defs>
    <style>
      .node-label { font-family: sans-serif; pointer-events: none; }
      .node { cursor: pointer; }
      .edge { fill: none; }
    </style>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="~w"/>

  <!-- Edges -->
  <g class="edges">
~w  </g>

  <!-- Nodes -->
  <g class="nodes">
~w  </g>
</svg>',
        [Width, Height, VBX, VBY, VBW, VBH, Background, EdgesContent, NodesContent]).

%% calculate_viewbox(+Positions, +Margin, -X, -Y, -W, -H)
%
%  Calculate viewBox to fit all nodes.
%
calculate_viewbox([], _, 0, 0, 1000, 800) :- !.
calculate_viewbox(Positions, Margin, X, Y, W, H) :-
    findall(PX, member(position(_, PX, _), Positions), Xs),
    findall(PY, member(position(_, _, PY), Positions), Ys),
    min_list(Xs, MinX),
    max_list(Xs, MaxX),
    min_list(Ys, MinY),
    max_list(Ys, MaxY),
    X is MinX - Margin,
    Y is MinY - Margin,
    W is MaxX - MinX + 2 * Margin,
    H is MaxY - MinY + 2 * Margin.

%% build_position_lookup(+Positions, -Lookup)
%
%  Build position lookup map.
%
build_position_lookup(Positions, Lookup) :-
    findall(Id-pos(X, Y), member(position(Id, X, Y), Positions), Lookup).

get_position(Id, Lookup, X, Y) :-
    member(Id-pos(X, Y), Lookup),
    !.
get_position(_, _, 0, 0).

% ============================================================================
% NODE RENDERING
% ============================================================================

%% render_nodes_svg(+Nodes, +PosLookup, +Options, +Theme, -Content)
%
%  Render all nodes to SVG elements.
%
render_nodes_svg([], _, _, _, '').
render_nodes_svg([node(Id, Props) | Rest], PosLookup, Options, Theme, Content) :-
    get_position(Id, PosLookup, X, Y),
    render_node_svg(node(Id, Props), pos(X, Y), Options, Theme, NodeSVG),
    render_nodes_svg(Rest, PosLookup, Options, Theme, RestSVG),
    atom_concat(NodeSVG, RestSVG, Content).

%% render_node_svg(+Node, +Position, +Options, +Theme, -SVGElement)
%
%  Render a single node to SVG.
%
render_node_svg(node(Id, Props), pos(X, Y), Options, Theme, SVGElement) :-
    % Get node properties
    (member(label(Label), Props) -> true ; atom_string(Id, Label)),
    (member(type(NodeType), Props) -> true ; NodeType = default),

    % Get style
    node_style(NodeType, Theme, StyleAttrs),
    option_or_default(node_shape, Options, ellipse, Shape),
    option_or_default(node_radius, Options, 40, Radius),
    option_or_default(include_labels, Options, true, IncludeLabels),

    % Get colors from style
    member(fill(Fill), StyleAttrs),
    member(stroke(Stroke), StyleAttrs),
    member(stroke_width(StrokeWidth), StyleAttrs),
    (member(text_color(TextColor), StyleAttrs) -> true ; TextColor = '#ffffff'),

    % Render shape
    render_shape(Shape, X, Y, Radius, Fill, Stroke, StrokeWidth, Id, ShapeSVG),

    % Render label
    (   IncludeLabels == true
    ->  escape_xml(Label, EscapedLabel),
        format(atom(LabelSVG),
            '    <text x="~2f" y="~2f" text-anchor="middle" dominant-baseline="middle" fill="~w" class="node-label" font-size="12">~w</text>~n',
            [X, Y, TextColor, EscapedLabel])
    ;   LabelSVG = ''
    ),

    % Combine into group
    format(atom(SVGElement),
        '    <g class="node" data-id="~w">~n~w~w    </g>~n',
        [Id, ShapeSVG, LabelSVG]).

%% render_shape(+Shape, +X, +Y, +R, +Fill, +Stroke, +SW, +Id, -SVG)
%
%  Render a node shape.
%
render_shape(ellipse, X, Y, R, Fill, Stroke, SW, _Id, SVG) :-
    !,
    format(atom(SVG),
        '      <ellipse cx="~2f" cy="~2f" rx="~w" ry="~w" fill="~w" stroke="~w" stroke-width="~w"/>~n',
        [X, Y, R, R, Fill, Stroke, SW]).

render_shape(rectangle, X, Y, R, Fill, Stroke, SW, _Id, SVG) :-
    !,
    W is R * 2,
    H is R * 1.5,
    RX is X - R,
    RY is Y - H/2,
    format(atom(SVG),
        '      <rect x="~2f" y="~2f" width="~w" height="~2f" rx="5" fill="~w" stroke="~w" stroke-width="~w"/>~n',
        [RX, RY, W, H, Fill, Stroke, SW]).

render_shape(diamond, X, Y, R, Fill, Stroke, SW, _Id, SVG) :-
    !,
    % Diamond points
    Top = [X, Y-R],
    Right = [X+R, Y],
    Bottom = [X, Y+R],
    Left = [X-R, Y],
    Top = [TX, TY], Right = [RiX, RiY], Bottom = [BX, BY], Left = [LX, LY],
    format(atom(SVG),
        '      <polygon points="~2f,~2f ~2f,~2f ~2f,~2f ~2f,~2f" fill="~w" stroke="~w" stroke-width="~w"/>~n',
        [TX, TY, RiX, RiY, BX, BY, LX, LY, Fill, Stroke, SW]).

render_shape(_, X, Y, R, Fill, Stroke, SW, Id, SVG) :-
    % Default to ellipse
    render_shape(ellipse, X, Y, R, Fill, Stroke, SW, Id, SVG).

% ============================================================================
% EDGE RENDERING
% ============================================================================

%% render_edges_svg(+Edges, +PosLookup, +Options, +Theme, -Content)
%
%  Render all edges to SVG elements.
%
render_edges_svg([], _, _, _, '').
render_edges_svg([edge(From, To, Props) | Rest], PosLookup, Options, Theme, Content) :-
    get_position(From, PosLookup, X1, Y1),
    get_position(To, PosLookup, X2, Y2),
    render_edge_svg(edge(From, To, Props), pos(X1, Y1), pos(X2, Y2), Options, Theme, EdgeSVG),
    render_edges_svg(Rest, PosLookup, Options, Theme, RestSVG),
    atom_concat(EdgeSVG, RestSVG, Content).

%% render_edge_svg(+Edge, +Pos1, +Pos2, +Options, +Theme, -SVGElement)
%
%  Render a single edge to SVG.
%
render_edge_svg(edge(_From, _To, Props), pos(X1, Y1), pos(X2, Y2), Options, Theme, SVGElement) :-
    % Get edge type
    (member(type(EdgeType), Props) -> true ; EdgeType = default),

    % Get style
    edge_style(EdgeType, Theme, StyleAttrs),
    option_or_default(edge_style, Options, straight, EdgeStyle),

    member(stroke(Stroke), StyleAttrs),
    member(stroke_width(StrokeWidth), StyleAttrs),

    % Render based on style
    render_edge_path(EdgeStyle, X1, Y1, X2, Y2, Stroke, StrokeWidth, SVGElement).

%% render_edge_path(+Style, +X1, +Y1, +X2, +Y2, +Stroke, +SW, -SVG)
%
%  Render edge path based on style.
%
render_edge_path(straight, X1, Y1, X2, Y2, Stroke, SW, SVG) :-
    !,
    format(atom(SVG),
        '    <line x1="~2f" y1="~2f" x2="~2f" y2="~2f" stroke="~w" stroke-width="~w" class="edge"/>~n',
        [X1, Y1, X2, Y2, Stroke, SW]).

render_edge_path(bezier, X1, Y1, X2, Y2, Stroke, SW, SVG) :-
    !,
    % Control points for smooth curve
    MidX is (X1 + X2) / 2,
    MidY is (Y1 + Y2) / 2,
    DX is X2 - X1,
    DY is Y2 - Y1,
    % Perpendicular offset for control point
    CX is MidX - DY * 0.2,
    CY is MidY + DX * 0.2,
    format(atom(SVG),
        '    <path d="M ~2f ~2f Q ~2f ~2f ~2f ~2f" stroke="~w" stroke-width="~w" class="edge"/>~n',
        [X1, Y1, CX, CY, X2, Y2, Stroke, SW]).

render_edge_path(_, X1, Y1, X2, Y2, Stroke, SW, SVG) :-
    % Default to straight
    render_edge_path(straight, X1, Y1, X2, Y2, Stroke, SW, SVG).

% ============================================================================
% THEMES AND STYLES
% ============================================================================

%% node_style(+NodeType, +Theme, -StyleAttrs)
%
%  Get style attributes for a node type and theme.
%
node_style(root, light, [fill('#4a90d9'), stroke('#2c5a8c'), stroke_width(3), text_color('#ffffff')]) :- !.
node_style(root, dark, [fill('#5a9ce9'), stroke('#3c6a9c'), stroke_width(3), text_color('#ffffff')]) :- !.
node_style(hub, light, [fill('#6ab04c'), stroke('#4a904c'), stroke_width(2), text_color('#ffffff')]) :- !.
node_style(hub, dark, [fill('#7ac05c'), stroke('#5aa05c'), stroke_width(2), text_color('#ffffff')]) :- !.
node_style(branch, light, [fill('#f0932b'), stroke('#c07020'), stroke_width(2), text_color('#ffffff')]) :- !.
node_style(branch, dark, [fill('#ffaa4b'), stroke('#d08030'), stroke_width(2), text_color('#000000')]) :- !.
node_style(leaf, light, [fill('#eb4d4b'), stroke('#cb2d2b'), stroke_width(1), text_color('#ffffff')]) :- !.
node_style(leaf, dark, [fill('#fb5d5b'), stroke('#db3d3b'), stroke_width(1), text_color('#ffffff')]) :- !.
node_style(default, light, [fill('#4a90d9'), stroke('#2c5a8c'), stroke_width(2), text_color('#ffffff')]) :- !.
node_style(default, dark, [fill('#5a9ce9'), stroke('#3c6a9c'), stroke_width(2), text_color('#ffffff')]) :- !.
node_style(_, Theme, Style) :-
    node_style(default, Theme, Style).

%% edge_style(+EdgeType, +Theme, -StyleAttrs)
%
%  Get style attributes for an edge type and theme.
%
edge_style(default, light, [stroke('#666666'), stroke_width(2)]) :- !.
edge_style(default, dark, [stroke('#999999'), stroke_width(2)]) :- !.
edge_style(strong, light, [stroke('#333333'), stroke_width(3)]) :- !.
edge_style(strong, dark, [stroke('#cccccc'), stroke_width(3)]) :- !.
edge_style(weak, light, [stroke('#aaaaaa'), stroke_width(1)]) :- !.
edge_style(weak, dark, [stroke('#666666'), stroke_width(1)]) :- !.
edge_style(_, Theme, Style) :-
    edge_style(default, Theme, Style).

% ============================================================================
% UTILITIES
% ============================================================================

option_or_default(Key, Options, Default, Value) :-
    Term =.. [Key, Value],
    (   member(Term, Options)
    ->  true
    ;   Value = Default
    ).

%% escape_xml(+Text, -Escaped)
%
%  Escape special XML characters.
%
escape_xml(Text, Escaped) :-
    atom_string(Text, Str),
    escape_xml_chars(Str, EscStr),
    atom_string(Escaped, EscStr).

escape_xml_chars([], []).
escape_xml_chars([C | Rest], Escaped) :-
    escape_char(C, EscC),
    escape_xml_chars(Rest, RestEsc),
    append(EscC, RestEsc, Escaped).

escape_char(0'<, "&lt;") :- !.
escape_char(0'>, "&gt;") :- !.
escape_char(0'&, "&amp;") :- !.
escape_char(0'", "&quot;") :- !.
escape_char(0'', "&#39;") :- !.
escape_char(C, [C]).

% ============================================================================
% TESTING
% ============================================================================

test_svg_renderer :-
    format('~n=== SVG Renderer Tests ===~n~n'),

    % Test data
    Nodes = [
        node(root, [label("Central Topic"), type(root)]),
        node(a, [label("Branch A"), type(branch)]),
        node(b, [label("Branch B"), type(branch)]),
        node(c, [label("Leaf C"), type(leaf)])
    ],
    Edges = [
        edge(root, a, []),
        edge(root, b, []),
        edge(a, c, [])
    ],
    Positions = [
        position(root, 500, 400),
        position(a, 350, 550),
        position(b, 650, 550),
        position(c, 300, 700)
    ],

    % Test 1: Basic rendering
    format('Test 1: Basic SVG rendering...~n'),
    render_mindmap_svg(Nodes, Edges, Positions, [], SVG),
    (   sub_atom(SVG, _, _, _, '<svg')
    ->  format('  PASS: SVG document generated~n')
    ;   format('  FAIL: Invalid SVG~n')
    ),

    % Test 2: Contains nodes
    format('~nTest 2: Contains node elements...~n'),
    (   sub_atom(SVG, _, _, _, 'class="node"')
    ->  format('  PASS: Node elements present~n')
    ;   format('  FAIL: No node elements~n')
    ),

    % Test 3: Contains edges
    format('~nTest 3: Contains edge elements...~n'),
    (   sub_atom(SVG, _, _, _, 'class="edge"')
    ->  format('  PASS: Edge elements present~n')
    ;   format('  FAIL: No edge elements~n')
    ),

    % Test 4: Dark theme
    format('~nTest 4: Dark theme rendering...~n'),
    render_mindmap_svg(Nodes, Edges, Positions, [theme(dark), background('#1a1a2e')], DarkSVG),
    (   sub_atom(DarkSVG, _, _, _, '#1a1a2e')
    ->  format('  PASS: Dark background applied~n')
    ;   format('  FAIL: Theme not applied~n')
    ),

    % Test 5: Bezier edges
    format('~nTest 5: Bezier edge style...~n'),
    render_mindmap_svg(Nodes, Edges, Positions, [edge_style(bezier)], BezierSVG),
    (   sub_atom(BezierSVG, _, _, _, '<path')
    ->  format('  PASS: Bezier paths generated~n')
    ;   format('  FAIL: No bezier paths~n')
    ),

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('SVG renderer module loaded~n', [])
), now).
