% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% style_resolver.pl - Mind Map Style Resolution System
%
% Resolves styles for nodes and edges by cascading through:
% 1. Theme defaults
% 2. Node/edge type styles
% 3. Cluster/group styles
% 4. Individual node/edge overrides
%
% Follows CSS-like specificity and cascade rules.
%
% Usage:
%   ?- resolve_node_style(NodeId, Props, Theme, ResolvedStyle).

:- module(mindmap_style_resolver, [
    % Style resolution
    resolve_node_style/4,           % resolve_node_style(+Id, +Props, +Theme, -Style)
    resolve_edge_style/5,           % resolve_edge_style(+From, +To, +Props, +Theme, -Style)

    % Selector matching
    matches_selector/3,             % matches_selector(+Selector, +Id, +Props)

    % Style merging
    merge_styles/3,                 % merge_styles(+BaseStyle, +OverrideStyle, -MergedStyle)

    % Style property access
    get_style_property/3,           % get_style_property(+Style, +Property, -Value)
    get_style_property/4,           % get_style_property(+Style, +Property, +Default, -Value)

    % Computed styles
    compute_node_dimensions/3,      % compute_node_dimensions(+Label, +Style, -Dimensions)
    compute_edge_path/5,            % compute_edge_path(+Style, +X1, +Y1, +X2, +Y2, -Path)

    % CSS custom properties (variables)
    define_css_variable/2,          % define_css_variable(+Name, +Value)
    get_css_variable/2,             % get_css_variable(+Name, -Value)
    clear_css_variables/0,          % clear_css_variables
    resolve_css_value/2,            % resolve_css_value(+Value, -ResolvedValue)
    resolve_style_variables/2,      % resolve_style_variables(+Style, -ResolvedStyle)

    % Testing
    test_style_resolver/0
]).

:- use_module(library(lists)).

% ============================================================================
% STYLE RESOLUTION
% ============================================================================

%% resolve_node_style(+Id, +Props, +Theme, -Style)
%
%  Resolve the complete style for a node.
%
%  Resolution order (later overrides earlier):
%  1. Theme defaults for nodes
%  2. Type-specific styles (root, hub, branch, leaf)
%  3. Cluster styles (if node has cluster property)
%  4. Individual style overrides (from Props)
%
%  @param Id     atom - node identifier
%  @param Props  list - node properties
%  @param Theme  atom - theme name
%  @param Style  list - resolved style properties
%
resolve_node_style(Id, Props, Theme, Style) :-
    % Start with theme defaults
    theme_node_defaults(Theme, DefaultStyle),

    % Apply type-specific style
    (member(type(NodeType), Props) -> true ; NodeType = default),
    theme_node_type_style(Theme, NodeType, TypeStyle),
    merge_styles(DefaultStyle, TypeStyle, Style1),

    % Apply cluster style if present
    (   member(cluster(Cluster), Props)
    ->  theme_cluster_style(Theme, Cluster, ClusterStyle),
        merge_styles(Style1, ClusterStyle, Style2)
    ;   Style2 = Style1
    ),

    % Apply importance-based style if present
    (   member(importance(Importance), Props)
    ->  importance_style(Importance, ImportanceStyle),
        merge_styles(Style2, ImportanceStyle, Style3)
    ;   Style3 = Style2
    ),

    % Apply individual overrides from props
    extract_style_props(Props, OverrideStyle),
    merge_styles(Style3, OverrideStyle, Style4),

    % Add computed properties
    add_computed_properties(Id, Props, Style4, Style).

%% resolve_edge_style(+From, +To, +Props, +Theme, -Style)
%
%  Resolve the complete style for an edge.
%
resolve_edge_style(_From, _To, Props, Theme, Style) :-
    % Theme edge defaults
    theme_edge_defaults(Theme, DefaultStyle),

    % Edge type style
    (member(type(EdgeType), Props) -> true ; EdgeType = default),
    theme_edge_type_style(Theme, EdgeType, TypeStyle),
    merge_styles(DefaultStyle, TypeStyle, Style1),

    % Individual overrides
    extract_style_props(Props, OverrideStyle),
    merge_styles(Style1, OverrideStyle, Style).

% ============================================================================
% THEME DEFAULTS
% ============================================================================

%% theme_node_defaults(+Theme, -Style)
theme_node_defaults(light, [
    fill('#e8f4fc'),
    stroke('#4a90d9'),
    stroke_width(2),
    text_color('#333333'),
    font_family('sans-serif'),
    font_size(12),
    padding(10),
    border_radius(5),
    shape(ellipse)
]) :- !.

theme_node_defaults(dark, [
    fill('#2d3748'),
    stroke('#4a9ce9'),
    stroke_width(2),
    text_color('#e2e8f0'),
    font_family('sans-serif'),
    font_size(12),
    padding(10),
    border_radius(5),
    shape(ellipse)
]) :- !.

theme_node_defaults(colorful, [
    fill('#ffeaa7'),
    stroke('#fdcb6e'),
    stroke_width(2),
    text_color('#2d3436'),
    font_family('sans-serif'),
    font_size(12),
    padding(10),
    border_radius(8),
    shape(ellipse)
]) :- !.

theme_node_defaults(_, Style) :-
    theme_node_defaults(light, Style).

%% theme_edge_defaults(+Theme, -Style)
theme_edge_defaults(light, [
    stroke('#666666'),
    stroke_width(2),
    edge_style(straight),
    arrow(none)
]) :- !.

theme_edge_defaults(dark, [
    stroke('#718096'),
    stroke_width(2),
    edge_style(straight),
    arrow(none)
]) :- !.

theme_edge_defaults(_, Style) :-
    theme_edge_defaults(light, Style).

% ============================================================================
% TYPE-SPECIFIC STYLES
% ============================================================================

%% theme_node_type_style(+Theme, +Type, -Style)
theme_node_type_style(light, root, [
    fill('#4a90d9'),
    stroke('#2c5a8c'),
    stroke_width(3),
    text_color('#ffffff'),
    font_size(16),
    shape(ellipse)
]) :- !.

theme_node_type_style(light, hub, [
    fill('#6ab04c'),
    stroke('#4a904c'),
    stroke_width(2),
    text_color('#ffffff'),
    font_size(14),
    shape(ellipse)
]) :- !.

theme_node_type_style(light, branch, [
    fill('#f0932b'),
    stroke('#c07020'),
    stroke_width(2),
    text_color('#ffffff'),
    shape(rectangle)
]) :- !.

theme_node_type_style(light, leaf, [
    fill('#eb4d4b'),
    stroke('#cb2d2b'),
    stroke_width(1),
    text_color('#ffffff'),
    shape(ellipse)
]) :- !.

theme_node_type_style(dark, root, [
    fill('#5a9ce9'),
    stroke('#3c6a9c'),
    stroke_width(3),
    text_color('#ffffff'),
    font_size(16),
    shape(ellipse)
]) :- !.

theme_node_type_style(dark, hub, [
    fill('#7ac05c'),
    stroke('#5aa05c'),
    stroke_width(2),
    text_color('#ffffff'),
    font_size(14),
    shape(ellipse)
]) :- !.

theme_node_type_style(dark, branch, [
    fill('#ffaa4b'),
    stroke('#d08030'),
    stroke_width(2),
    text_color('#000000'),
    shape(rectangle)
]) :- !.

theme_node_type_style(dark, leaf, [
    fill('#fb5d5b'),
    stroke('#db3d3b'),
    stroke_width(1),
    text_color('#ffffff'),
    shape(ellipse)
]) :- !.

theme_node_type_style(_, default, []) :- !.
theme_node_type_style(_, _, []).

%% theme_edge_type_style(+Theme, +Type, -Style)
theme_edge_type_style(_, strong, [
    stroke_width(3),
    stroke('#333333')
]) :- !.

theme_edge_type_style(_, weak, [
    stroke_width(1),
    stroke('#aaaaaa'),
    edge_style(dashed)
]) :- !.

theme_edge_type_style(_, _, []).

%% theme_cluster_style(+Theme, +Cluster, -Style)
theme_cluster_style(_, _, []).  % Placeholder for cluster-specific styles

%% importance_style(+Importance, -Style)
importance_style(high, [
    stroke_width(3),
    font_weight(bold)
]) :- !.

importance_style(low, [
    opacity(0.7),
    font_size(10)
]) :- !.

importance_style(_, []).

% ============================================================================
% STYLE MERGING
% ============================================================================

%% merge_styles(+BaseStyle, +OverrideStyle, -MergedStyle)
%
%  Merge two style lists, with OverrideStyle taking precedence.
%
merge_styles(BaseStyle, [], BaseStyle) :- !.
merge_styles(BaseStyle, [Prop | Rest], MergedStyle) :-
    Prop =.. [Key, Value],
    % Remove existing property with same key
    OldProp =.. [Key, _],
    (   select(OldProp, BaseStyle, BaseRest)
    ->  true
    ;   BaseRest = BaseStyle
    ),
    % Add new property
    NewProp =.. [Key, Value],
    merge_styles([NewProp | BaseRest], Rest, MergedStyle).

% ============================================================================
% SELECTOR MATCHING
% ============================================================================

%% matches_selector(+Selector, +Id, +Props)
%
%  Check if a node matches a selector.
%
%  Selectors:
%  - id(NodeId)         - matches specific node
%  - type(Type)         - matches nodes of type
%  - cluster(Name)      - matches nodes in cluster
%  - has_link           - matches nodes with links
%  - all                - matches all nodes
%
matches_selector(all, _, _) :- !.
matches_selector(id(Id), Id, _) :- !.
matches_selector(type(Type), _, Props) :-
    !,
    member(type(Type), Props).
matches_selector(cluster(Name), _, Props) :-
    !,
    member(cluster(Name), Props).
matches_selector(has_link, _, Props) :-
    !,
    member(link(URL), Props),
    URL \== ''.
matches_selector(importance(Level), _, Props) :-
    !,
    member(importance(Level), Props).

% ============================================================================
% PROPERTY ACCESS
% ============================================================================

%% get_style_property(+Style, +Property, -Value)
get_style_property(Style, Property, Value) :-
    Prop =.. [Property, Value],
    member(Prop, Style),
    !.

%% get_style_property(+Style, +Property, +Default, -Value)
get_style_property(Style, Property, Default, Value) :-
    Prop =.. [Property, Value],
    (   member(Prop, Style)
    ->  true
    ;   Value = Default
    ).

%% extract_style_props(+Props, -StyleProps)
%
%  Extract style-related properties from node properties.
%
extract_style_props(Props, StyleProps) :-
    findall(Prop, (member(Prop, Props), is_style_prop(Prop)), StyleProps).

is_style_prop(fill(_)).
is_style_prop(stroke(_)).
is_style_prop(stroke_width(_)).
is_style_prop(text_color(_)).
is_style_prop(font_size(_)).
is_style_prop(font_family(_)).
is_style_prop(shape(_)).
is_style_prop(opacity(_)).
is_style_prop(padding(_)).
is_style_prop(border_radius(_)).

% ============================================================================
% COMPUTED PROPERTIES
% ============================================================================

%% add_computed_properties(+Id, +Props, +Style, -NewStyle)
%
%  Add computed properties based on node content.
%
add_computed_properties(_Id, Props, Style, NewStyle) :-
    % Compute dimensions based on label
    (   member(label(Label), Props)
    ->  compute_node_dimensions(Label, Style, Dims),
        merge_styles(Style, Dims, NewStyle)
    ;   NewStyle = Style
    ).

%% compute_node_dimensions(+Label, +Style, -Dimensions)
%
%  Compute node dimensions based on label text.
%
compute_node_dimensions(Label, Style, [width(Width), height(Height)]) :-
    atom_string(Label, LabelStr),
    string_length(LabelStr, Len),
    get_style_property(Style, font_size, 12, FontSize),
    get_style_property(Style, padding, 10, Padding),
    % Approximate character width as 0.6 * font size
    CharWidth is FontSize * 0.6,
    Width is max(60, Len * CharWidth + 2 * Padding),
    Height is FontSize + 2 * Padding.

%% compute_edge_path(+Style, +X1, +Y1, +X2, +Y2, -Path)
%
%  Compute SVG path for an edge based on style.
%
compute_edge_path(Style, X1, Y1, X2, Y2, Path) :-
    get_style_property(Style, edge_style, straight, EdgeStyle),
    compute_path_for_style(EdgeStyle, X1, Y1, X2, Y2, Path).

compute_path_for_style(straight, X1, Y1, X2, Y2, Path) :-
    format(atom(Path), 'M ~2f ~2f L ~2f ~2f', [X1, Y1, X2, Y2]).

compute_path_for_style(bezier, X1, Y1, X2, Y2, Path) :-
    MidX is (X1 + X2) / 2,
    MidY is (Y1 + Y2) / 2,
    DX is X2 - X1,
    DY is Y2 - Y1,
    CX is MidX - DY * 0.2,
    CY is MidY + DX * 0.2,
    format(atom(Path), 'M ~2f ~2f Q ~2f ~2f ~2f ~2f', [X1, Y1, CX, CY, X2, Y2]).

compute_path_for_style(orthogonal, X1, Y1, X2, Y2, Path) :-
    MidX is (X1 + X2) / 2,
    format(atom(Path), 'M ~2f ~2f L ~2f ~2f L ~2f ~2f L ~2f ~2f',
           [X1, Y1, MidX, Y1, MidX, Y2, X2, Y2]).

compute_path_for_style(_, X1, Y1, X2, Y2, Path) :-
    compute_path_for_style(straight, X1, Y1, X2, Y2, Path).

% ============================================================================
% TESTING
% ============================================================================

test_style_resolver :-
    format('~n=== Style Resolver Tests ===~n~n'),

    % Test 1: Resolve root node style
    format('Test 1: Resolve root node style...~n'),
    resolve_node_style(root, [label("Root"), type(root)], light, RootStyle),
    (   get_style_property(RootStyle, fill, '#4a90d9')
    ->  format('  PASS: Root has correct fill color~n')
    ;   format('  FAIL: Root fill color incorrect~n')
    ),

    % Test 2: Style override
    format('~nTest 2: Style override...~n'),
    resolve_node_style(custom, [label("Custom"), type(branch), fill('#ff0000')], light, CustomStyle),
    (   get_style_property(CustomStyle, fill, '#ff0000')
    ->  format('  PASS: Override applied~n')
    ;   format('  FAIL: Override not applied~n')
    ),

    % Test 3: Merge styles
    format('~nTest 3: Merge styles...~n'),
    merge_styles([fill(blue), stroke(black)], [fill(red)], Merged),
    (   get_style_property(Merged, fill, red),
        get_style_property(Merged, stroke, black)
    ->  format('  PASS: Styles merged correctly~n')
    ;   format('  FAIL: Merge incorrect~n')
    ),

    % Test 4: Selector matching
    format('~nTest 4: Selector matching...~n'),
    (   matches_selector(type(branch), test, [type(branch)]),
        \+ matches_selector(type(leaf), test, [type(branch)])
    ->  format('  PASS: Selectors match correctly~n')
    ;   format('  FAIL: Selector matching incorrect~n')
    ),

    % Test 5: Computed dimensions
    format('~nTest 5: Computed dimensions...~n'),
    compute_node_dimensions("Hello World", [font_size(12), padding(10)], Dims),
    (   get_style_property(Dims, width, W), W > 60
    ->  format('  PASS: Dimensions computed (width: ~w)~n', [W])
    ;   format('  FAIL: Dimensions incorrect~n')
    ),

    % Test 6: Edge path computation
    format('~nTest 6: Edge path computation...~n'),
    compute_edge_path([edge_style(bezier)], 0, 0, 100, 100, Path),
    (   sub_atom(Path, _, _, _, 'Q')
    ->  format('  PASS: Bezier path generated~n')
    ;   format('  FAIL: Path incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% CSS CUSTOM PROPERTIES (VARIABLES)
% ============================================================================

:- dynamic css_variable/2.  % css_variable(Name, Value)

%% define_css_variable(+Name, +Value)
%
%  Define a CSS custom property (variable).
%  Name should start with '--' by convention.
%
define_css_variable(Name, Value) :-
    retractall(css_variable(Name, _)),
    assertz(css_variable(Name, Value)).

%% get_css_variable(+Name, -Value)
%
%  Get the value of a CSS variable.
%
get_css_variable(Name, Value) :-
    css_variable(Name, Value).

%% clear_css_variables
%
%  Remove all CSS variable definitions.
%
clear_css_variables :-
    retractall(css_variable(_, _)).

%% resolve_css_value(+Value, -ResolvedValue)
%
%  Resolve a value that may contain CSS var() references.
%  Supports: var(--name), var(--name, fallback)
%
resolve_css_value(Value, ResolvedValue) :-
    atom(Value),
    atom_string(Value, ValueStr),
    (   sub_string(ValueStr, _, _, _, "var(")
    ->  resolve_var_references(ValueStr, ResolvedStr),
        atom_string(ResolvedValue, ResolvedStr)
    ;   ResolvedValue = Value
    ).
resolve_css_value(Value, Value) :-
    \+ atom(Value).

%% resolve_var_references(+String, -ResolvedString)
%
%  Resolve all var() references in a string.
%
resolve_var_references(String, ResolvedString) :-
    (   sub_string(String, Before, _, After, "var("),
        sub_string(String, _, After, 0, Rest),
        sub_string(Rest, VarLen, _, _, ")")
    ->  sub_string(Rest, 0, VarLen, _, VarContent),
        % Parse var content: either "--name" or "--name, fallback"
        (   sub_string(VarContent, CommaPos, _, _, ",")
        ->  sub_string(VarContent, 0, CommaPos, _, VarName),
            FallbackStart is CommaPos + 1,
            sub_string(VarContent, FallbackStart, _, 0, FallbackRaw),
            string_trim(FallbackRaw, Fallback)
        ;   VarName = VarContent,
            Fallback = ""
        ),
        % Trim whitespace from variable name
        string_trim(VarName, TrimmedName),
        atom_string(VarAtom, TrimmedName),
        % Look up variable
        (   css_variable(VarAtom, VarValue)
        ->  atom_string(VarValue, ValueStr)
        ;   Fallback \= ""
        ->  ValueStr = Fallback
        ;   ValueStr = ""
        ),
        % Replace var() with value
        sub_string(String, 0, Before, _, BeforeStr),
        ClosePos is VarLen + 1,
        sub_string(Rest, ClosePos, _, 0, AfterStr),
        string_concat(BeforeStr, ValueStr, Temp),
        string_concat(Temp, AfterStr, PartialResolved),
        % Recursively resolve any remaining var() references
        resolve_var_references(PartialResolved, ResolvedString)
    ;   ResolvedString = String
    ).

%% string_trim(+String, -Trimmed)
%
%  Remove leading and trailing whitespace.
%
string_trim(String, Trimmed) :-
    string_codes(String, Codes),
    trim_codes(Codes, TrimmedCodes),
    string_codes(Trimmed, TrimmedCodes).

trim_codes([], []).
trim_codes([H|T], Result) :-
    (   char_type(H, space)
    ->  trim_codes(T, Result)
    ;   reverse([H|T], Rev),
        trim_trailing(Rev, TrimmedRev),
        reverse(TrimmedRev, Result)
    ).

trim_trailing([], []).
trim_trailing([H|T], Result) :-
    (   char_type(H, space)
    ->  trim_trailing(T, Result)
    ;   Result = [H|T]
    ).

%% resolve_style_variables(+Style, -ResolvedStyle)
%
%  Resolve all CSS variable references in a style list.
%
resolve_style_variables(Style, ResolvedStyle) :-
    maplist(resolve_style_prop, Style, ResolvedStyle).

resolve_style_prop(Prop, ResolvedProp) :-
    Prop =.. [Key, Value],
    resolve_css_value(Value, ResolvedValue),
    ResolvedProp =.. [Key, ResolvedValue].

%% Define default CSS variables for themes
:- initialization((
    % Primary colors
    define_css_variable('--primary-color', '#4a90d9'),
    define_css_variable('--primary-dark', '#2c5a8c'),
    define_css_variable('--primary-light', '#e8f4fc'),

    % Accent colors
    define_css_variable('--accent-success', '#6ab04c'),
    define_css_variable('--accent-warning', '#f0932b'),
    define_css_variable('--accent-danger', '#eb4d4b'),

    % Text colors
    define_css_variable('--text-primary', '#333333'),
    define_css_variable('--text-secondary', '#666666'),
    define_css_variable('--text-light', '#ffffff'),

    % Background colors
    define_css_variable('--bg-light', '#ffffff'),
    define_css_variable('--bg-dark', '#1a1a2e'),

    % Spacing
    define_css_variable('--spacing-sm', '8'),
    define_css_variable('--spacing-md', '16'),
    define_css_variable('--spacing-lg', '24'),

    % Border radius
    define_css_variable('--radius-sm', '4'),
    define_css_variable('--radius-md', '8'),
    define_css_variable('--radius-lg', '12')
), now).

:- initialization((
    format('Style resolver module loaded~n', [])
), now).
