% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% theme_system.pl - Mind Map Theme Management System
%
% Provides theme definition, inheritance, and application for mind maps.
% Themes define visual styles for nodes, edges, and the overall canvas.
%
% Features:
% - Built-in themes (light, dark, colorful, minimal, corporate)
% - Theme inheritance/extension
% - User-defined themes
% - Theme validation
%
% Usage:
%   ?- get_theme(dark, Theme).
%   ?- define_custom_theme(my_theme, extends(dark), [...]).

:- module(mindmap_theme_system, [
    % Theme retrieval
    get_theme/2,                    % get_theme(+ThemeName, -Theme)
    theme_property/3,               % theme_property(+Theme, +Property, -Value)

    % Theme definition
    define_theme/2,                 % define_theme(+Name, +Properties)
    define_custom_theme/3,          % define_custom_theme(+Name, +Extends, +Overrides)

    % Theme management
    list_themes/1,                  % list_themes(-ThemeNames)
    theme_exists/1,                 % theme_exists(+Name)
    clear_custom_themes/0,          % clear_custom_themes

    % Theme inheritance
    resolve_theme/2,                % resolve_theme(+ThemeName, -ResolvedTheme)
    theme_extends/2,                % theme_extends(+Theme, -Parent)

    % Theme application
    apply_theme/3,                  % apply_theme(+Theme, +Element, -StyledElement)

    % Validation
    validate_theme/2,               % validate_theme(+Theme, -Errors)

    % Testing
    test_theme_system/0
]).

:- use_module(library(lists)).

% Dynamic predicates for custom themes
:- dynamic custom_theme/2.          % custom_theme(Name, Properties)
:- dynamic custom_theme_extends/2.  % custom_theme_extends(Name, Parent)

% ============================================================================
% BUILT-IN THEMES
% ============================================================================

%% builtin_theme(+Name, -Properties)
%
%  Define built-in themes with their complete property sets.
%

% Light theme - clean, professional look
builtin_theme(light, [
    % Meta information
    name(light),
    description("Light theme with blue accents"),

    % Canvas properties
    canvas([
        background('#ffffff'),
        grid_color('#f0f0f0'),
        grid_enabled(true)
    ]),

    % Default node styles
    node_defaults([
        fill('#e8f4fc'),
        stroke('#4a90d9'),
        stroke_width(2),
        text_color('#333333'),
        font_family('sans-serif'),
        font_size(12),
        padding(10),
        border_radius(5),
        shape(ellipse),
        shadow(false)
    ]),

    % Node type styles
    node_types([
        root([
            fill('#4a90d9'),
            stroke('#2c5a8c'),
            stroke_width(3),
            text_color('#ffffff'),
            font_size(16),
            font_weight(bold),
            shape(ellipse)
        ]),
        hub([
            fill('#6ab04c'),
            stroke('#4a904c'),
            stroke_width(2),
            text_color('#ffffff'),
            font_size(14),
            shape(ellipse)
        ]),
        branch([
            fill('#f0932b'),
            stroke('#c07020'),
            stroke_width(2),
            text_color('#ffffff'),
            shape(rectangle)
        ]),
        leaf([
            fill('#eb4d4b'),
            stroke('#cb2d2b'),
            stroke_width(1),
            text_color('#ffffff'),
            shape(ellipse)
        ])
    ]),

    % Default edge styles
    edge_defaults([
        stroke('#666666'),
        stroke_width(2),
        edge_style(straight),
        arrow(none),
        opacity(1.0)
    ]),

    % Edge type styles
    edge_types([
        strong([
            stroke_width(3),
            stroke('#333333')
        ]),
        weak([
            stroke_width(1),
            stroke('#aaaaaa'),
            edge_style(dashed)
        ]),
        hierarchy([
            stroke('#4a90d9'),
            edge_style(bezier)
        ])
    ]),

    % Cluster/group styles
    cluster_defaults([
        fill('#f5f5f5'),
        stroke('#cccccc'),
        stroke_width(1),
        stroke_dasharray('5,5'),
        border_radius(10),
        padding(20),
        opacity(0.5)
    ]),

    % Color palette for auto-coloring
    palette([
        '#4a90d9', '#6ab04c', '#f0932b', '#eb4d4b',
        '#9b59b6', '#1abc9c', '#e74c3c', '#3498db'
    ])
]).

% Dark theme - modern dark mode
builtin_theme(dark, [
    name(dark),
    description("Dark theme for reduced eye strain"),

    canvas([
        background('#1a1a2e'),
        grid_color('#2a2a4e'),
        grid_enabled(true)
    ]),

    node_defaults([
        fill('#2d3748'),
        stroke('#4a9ce9'),
        stroke_width(2),
        text_color('#e2e8f0'),
        font_family('sans-serif'),
        font_size(12),
        padding(10),
        border_radius(5),
        shape(ellipse),
        shadow(true),
        shadow_color('rgba(0,0,0,0.3)')
    ]),

    node_types([
        root([
            fill('#5a9ce9'),
            stroke('#3c6a9c'),
            stroke_width(3),
            text_color('#ffffff'),
            font_size(16),
            font_weight(bold),
            shape(ellipse)
        ]),
        hub([
            fill('#7ac05c'),
            stroke('#5aa05c'),
            stroke_width(2),
            text_color('#ffffff'),
            font_size(14),
            shape(ellipse)
        ]),
        branch([
            fill('#ffaa4b'),
            stroke('#d08030'),
            stroke_width(2),
            text_color('#000000'),
            shape(rectangle)
        ]),
        leaf([
            fill('#fb5d5b'),
            stroke('#db3d3b'),
            stroke_width(1),
            text_color('#ffffff'),
            shape(ellipse)
        ])
    ]),

    edge_defaults([
        stroke('#718096'),
        stroke_width(2),
        edge_style(straight),
        arrow(none),
        opacity(0.9)
    ]),

    edge_types([
        strong([
            stroke_width(3),
            stroke('#a0aec0')
        ]),
        weak([
            stroke_width(1),
            stroke('#4a5568'),
            edge_style(dashed)
        ]),
        hierarchy([
            stroke('#5a9ce9'),
            edge_style(bezier)
        ])
    ]),

    cluster_defaults([
        fill('#2d3748'),
        stroke('#4a5568'),
        stroke_width(1),
        stroke_dasharray('5,5'),
        border_radius(10),
        padding(20),
        opacity(0.6)
    ]),

    palette([
        '#5a9ce9', '#7ac05c', '#ffaa4b', '#fb5d5b',
        '#a78bfa', '#2dd4bf', '#f87171', '#60a5fa'
    ])
]).

% Colorful theme - vibrant and playful
builtin_theme(colorful, [
    name(colorful),
    description("Vibrant theme with warm colors"),

    canvas([
        background('#fffaf0'),
        grid_color('#fff0e0'),
        grid_enabled(false)
    ]),

    node_defaults([
        fill('#ffeaa7'),
        stroke('#fdcb6e'),
        stroke_width(2),
        text_color('#2d3436'),
        font_family('sans-serif'),
        font_size(12),
        padding(10),
        border_radius(8),
        shape(ellipse),
        shadow(true),
        shadow_color('rgba(253,203,110,0.3)')
    ]),

    node_types([
        root([
            fill('#e17055'),
            stroke('#d63031'),
            stroke_width(3),
            text_color('#ffffff'),
            font_size(18),
            font_weight(bold),
            shape(ellipse)
        ]),
        hub([
            fill('#00b894'),
            stroke('#00a085'),
            stroke_width(2),
            text_color('#ffffff'),
            font_size(14),
            shape(ellipse)
        ]),
        branch([
            fill('#0984e3'),
            stroke('#0769c3'),
            stroke_width(2),
            text_color('#ffffff'),
            shape(rectangle),
            border_radius(12)
        ]),
        leaf([
            fill('#a29bfe'),
            stroke('#8b7df0'),
            stroke_width(1),
            text_color('#ffffff'),
            shape(ellipse)
        ])
    ]),

    edge_defaults([
        stroke('#b2bec3'),
        stroke_width(2),
        edge_style(bezier),
        arrow(none),
        opacity(0.8)
    ]),

    edge_types([
        strong([
            stroke_width(4),
            stroke('#636e72')
        ]),
        weak([
            stroke_width(1),
            stroke('#dfe6e9'),
            edge_style(dashed)
        ]),
        hierarchy([
            stroke('#fdcb6e'),
            edge_style(bezier)
        ])
    ]),

    cluster_defaults([
        fill('#ffeaa7'),
        stroke('#fdcb6e'),
        stroke_width(2),
        border_radius(15),
        padding(25),
        opacity(0.4)
    ]),

    palette([
        '#e17055', '#00b894', '#0984e3', '#a29bfe',
        '#fdcb6e', '#fd79a8', '#00cec9', '#6c5ce7'
    ])
]).

% Minimal theme - clean and understated
builtin_theme(minimal, [
    name(minimal),
    description("Minimalist black and white theme"),

    canvas([
        background('#ffffff'),
        grid_color('#f8f8f8'),
        grid_enabled(false)
    ]),

    node_defaults([
        fill('#ffffff'),
        stroke('#333333'),
        stroke_width(1),
        text_color('#333333'),
        font_family('Georgia, serif'),
        font_size(12),
        padding(12),
        border_radius(0),
        shape(rectangle),
        shadow(false)
    ]),

    node_types([
        root([
            fill('#333333'),
            stroke('#000000'),
            stroke_width(2),
            text_color('#ffffff'),
            font_size(14),
            font_weight(bold),
            shape(rectangle)
        ]),
        hub([
            fill('#666666'),
            stroke('#333333'),
            stroke_width(1),
            text_color('#ffffff'),
            font_size(13),
            shape(rectangle)
        ]),
        branch([
            fill('#f0f0f0'),
            stroke('#666666'),
            stroke_width(1),
            text_color('#333333'),
            shape(rectangle)
        ]),
        leaf([
            fill('#ffffff'),
            stroke('#999999'),
            stroke_width(1),
            text_color('#666666'),
            shape(rectangle)
        ])
    ]),

    edge_defaults([
        stroke('#999999'),
        stroke_width(1),
        edge_style(straight),
        arrow(none),
        opacity(1.0)
    ]),

    edge_types([
        strong([
            stroke_width(2),
            stroke('#333333')
        ]),
        weak([
            stroke_width(1),
            stroke('#cccccc'),
            edge_style(dashed)
        ]),
        hierarchy([
            stroke('#666666'),
            edge_style(orthogonal)
        ])
    ]),

    cluster_defaults([
        fill('none'),
        stroke('#cccccc'),
        stroke_width(1),
        stroke_dasharray('3,3'),
        border_radius(0),
        padding(15),
        opacity(1.0)
    ]),

    palette([
        '#333333', '#666666', '#999999', '#cccccc',
        '#000000', '#444444', '#777777', '#aaaaaa'
    ])
]).

% Corporate theme - professional business look
builtin_theme(corporate, [
    name(corporate),
    description("Professional corporate theme"),

    canvas([
        background('#f5f7fa'),
        grid_color('#e8ecf1'),
        grid_enabled(true)
    ]),

    node_defaults([
        fill('#ffffff'),
        stroke('#3182ce'),
        stroke_width(1),
        text_color('#2d3748'),
        font_family('Arial, sans-serif'),
        font_size(11),
        padding(8),
        border_radius(4),
        shape(rectangle),
        shadow(true),
        shadow_color('rgba(0,0,0,0.1)')
    ]),

    node_types([
        root([
            fill('#2c5282'),
            stroke('#1a365d'),
            stroke_width(2),
            text_color('#ffffff'),
            font_size(14),
            font_weight(bold),
            shape(rectangle),
            border_radius(6)
        ]),
        hub([
            fill('#2b6cb0'),
            stroke('#2c5282'),
            stroke_width(1),
            text_color('#ffffff'),
            font_size(12),
            shape(rectangle)
        ]),
        branch([
            fill('#ebf8ff'),
            stroke('#3182ce'),
            stroke_width(1),
            text_color('#2c5282'),
            shape(rectangle)
        ]),
        leaf([
            fill('#ffffff'),
            stroke('#90cdf4'),
            stroke_width(1),
            text_color('#4a5568'),
            shape(rectangle)
        ])
    ]),

    edge_defaults([
        stroke('#a0aec0'),
        stroke_width(1),
        edge_style(orthogonal),
        arrow(none),
        opacity(1.0)
    ]),

    edge_types([
        strong([
            stroke_width(2),
            stroke('#3182ce')
        ]),
        weak([
            stroke_width(1),
            stroke('#cbd5e0'),
            edge_style(dashed)
        ]),
        hierarchy([
            stroke('#3182ce'),
            edge_style(orthogonal)
        ])
    ]),

    cluster_defaults([
        fill('#edf2f7'),
        stroke('#a0aec0'),
        stroke_width(1),
        border_radius(6),
        padding(16),
        opacity(0.8)
    ]),

    palette([
        '#2c5282', '#2b6cb0', '#3182ce', '#4299e1',
        '#276749', '#2f855a', '#c53030', '#9c4221'
    ])
]).

% ============================================================================
% THEME RETRIEVAL
% ============================================================================

%% get_theme(+ThemeName, -Theme)
%
%  Get a theme by name. Checks custom themes first, then built-in.
%
get_theme(Name, Theme) :-
    custom_theme(Name, Theme),
    !.
get_theme(Name, Theme) :-
    builtin_theme(Name, Theme),
    !.
get_theme(Name, _) :-
    format(user_error, 'Warning: Theme ~w not found, using light~n', [Name]),
    fail.
get_theme(_, Theme) :-
    builtin_theme(light, Theme).

%% theme_property(+Theme, +Property, -Value)
%
%  Get a specific property from a theme.
%
theme_property(Theme, Property, Value) :-
    Prop =.. [Property, Value],
    member(Prop, Theme),
    !.

%% list_themes(-ThemeNames)
%
%  List all available theme names.
%
list_themes(Names) :-
    findall(Name, builtin_theme(Name, _), Builtin),
    findall(Name, custom_theme(Name, _), Custom),
    append(Builtin, Custom, Names).

%% theme_exists(+Name)
%
%  Check if a theme exists.
%
theme_exists(Name) :-
    builtin_theme(Name, _),
    !.
theme_exists(Name) :-
    custom_theme(Name, _).

% ============================================================================
% THEME DEFINITION
% ============================================================================

%% define_theme(+Name, +Properties)
%
%  Define a new custom theme.
%
define_theme(Name, Properties) :-
    (   builtin_theme(Name, _)
    ->  format(user_error, 'Error: Cannot override built-in theme ~w~n', [Name]),
        fail
    ;   retractall(custom_theme(Name, _)),
        retractall(custom_theme_extends(Name, _)),
        assertz(custom_theme(Name, Properties))
    ).

%% define_custom_theme(+Name, +Extends, +Overrides)
%
%  Define a custom theme that extends another theme.
%
define_custom_theme(Name, extends(Parent), Overrides) :-
    (   theme_exists(Parent)
    ->  retractall(custom_theme(Name, _)),
        retractall(custom_theme_extends(Name, _)),
        assertz(custom_theme_extends(Name, Parent)),
        % Resolve parent and merge overrides
        resolve_theme(Parent, ParentTheme),
        merge_theme_properties(ParentTheme, Overrides, MergedTheme),
        % Update name in merged theme
        select(name(_), MergedTheme, TempTheme),
        assertz(custom_theme(Name, [name(Name) | TempTheme]))
    ;   format(user_error, 'Error: Parent theme ~w not found~n', [Parent]),
        fail
    ).

%% clear_custom_themes
%
%  Remove all custom theme definitions.
%
clear_custom_themes :-
    retractall(custom_theme(_, _)),
    retractall(custom_theme_extends(_, _)).

% ============================================================================
% THEME INHERITANCE
% ============================================================================

%% resolve_theme(+ThemeName, -ResolvedTheme)
%
%  Resolve a theme, including any inheritance chain.
%
resolve_theme(Name, ResolvedTheme) :-
    custom_theme_extends(Name, Parent),
    !,
    resolve_theme(Parent, ParentTheme),
    custom_theme(Name, ChildTheme),
    merge_theme_properties(ParentTheme, ChildTheme, ResolvedTheme).
resolve_theme(Name, Theme) :-
    get_theme(Name, Theme).

%% theme_extends(+Theme, -Parent)
%
%  Get the parent theme if this theme extends another.
%
theme_extends(Name, Parent) :-
    custom_theme_extends(Name, Parent).

%% merge_theme_properties(+Base, +Override, -Merged)
%
%  Merge theme properties with Override taking precedence.
%  Handles nested property lists for node_types, edge_types, etc.
%
merge_theme_properties(Base, [], Base) :- !.
merge_theme_properties(Base, [Prop | Rest], Merged) :-
    Prop =.. [Key, Value],
    (   nested_theme_property(Key)
    ->  % Nested property - merge recursively
        OldProp =.. [Key, OldValue],
        (   select(OldProp, Base, BaseRest)
        ->  merge_nested_styles(OldValue, Value, MergedValue)
        ;   BaseRest = Base, MergedValue = Value
        ),
        NewProp =.. [Key, MergedValue]
    ;   % Simple property - replace
        OldProp =.. [Key, _],
        (   select(OldProp, Base, BaseRest)
        ->  true
        ;   BaseRest = Base
        ),
        NewProp = Prop
    ),
    merge_theme_properties([NewProp | BaseRest], Rest, Merged).

%% nested_theme_property(+Key)
%
%  Properties that contain nested style lists.
%
nested_theme_property(node_defaults).
nested_theme_property(node_types).
nested_theme_property(edge_defaults).
nested_theme_property(edge_types).
nested_theme_property(cluster_defaults).
nested_theme_property(canvas).

%% merge_nested_styles(+Base, +Override, -Merged)
%
%  Merge nested style lists.
%
merge_nested_styles(Base, Override, Merged) :-
    is_list(Base),
    is_list(Override),
    !,
    % Check if it's a list of key-value pairs (like node_types)
    (   Override = [FirstOverride | _],
        FirstOverride =.. [_, InnerList],
        is_list(InnerList)
    ->  % It's a list of type definitions - merge by type
        merge_type_definitions(Base, Override, Merged)
    ;   % It's a flat property list - merge directly
        merge_property_list(Base, Override, Merged)
    ).
merge_nested_styles(_, Override, Override).

%% merge_type_definitions(+Base, +Override, -Merged)
%
%  Merge type-specific style definitions (like node_types).
%
merge_type_definitions(Base, [], Base) :- !.
merge_type_definitions(Base, [TypeDef | Rest], Merged) :-
    TypeDef =.. [TypeName, TypeStyles],
    OldDef =.. [TypeName, OldStyles],
    (   select(OldDef, Base, BaseRest)
    ->  merge_property_list(OldStyles, TypeStyles, MergedStyles)
    ;   BaseRest = Base, MergedStyles = TypeStyles
    ),
    NewDef =.. [TypeName, MergedStyles],
    merge_type_definitions([NewDef | BaseRest], Rest, Merged).

%% merge_property_list(+Base, +Override, -Merged)
%
%  Merge flat property lists.
%
merge_property_list(Base, [], Base) :- !.
merge_property_list(Base, [Prop | Rest], Merged) :-
    Prop =.. [Key, Value],
    OldProp =.. [Key, _],
    (   select(OldProp, Base, BaseRest)
    ->  true
    ;   BaseRest = Base
    ),
    NewProp =.. [Key, Value],
    merge_property_list([NewProp | BaseRest], Rest, Merged).

% ============================================================================
% THEME APPLICATION
% ============================================================================

%% apply_theme(+Theme, +Element, -StyledElement)
%
%  Apply theme styles to an element (node or edge).
%
apply_theme(Theme, node(Id, Props), node(Id, StyledProps)) :-
    % Get node defaults
    (   theme_property(Theme, node_defaults, Defaults)
    ->  true
    ;   Defaults = []
    ),

    % Get type-specific styles if applicable
    (   member(type(Type), Props),
        theme_property(Theme, node_types, TypeStyles),
        TypeDef =.. [Type, TypeSpecificStyles],
        member(TypeDef, TypeStyles)
    ->  merge_property_list(Defaults, TypeSpecificStyles, BaseStyles)
    ;   BaseStyles = Defaults
    ),

    % Merge with existing props (props take precedence)
    merge_property_list(BaseStyles, Props, StyledProps).

apply_theme(Theme, edge(From, To, Props), edge(From, To, StyledProps)) :-
    % Get edge defaults
    (   theme_property(Theme, edge_defaults, Defaults)
    ->  true
    ;   Defaults = []
    ),

    % Get type-specific styles if applicable
    (   member(type(Type), Props),
        theme_property(Theme, edge_types, TypeStyles),
        TypeDef =.. [Type, TypeSpecificStyles],
        member(TypeDef, TypeStyles)
    ->  merge_property_list(Defaults, TypeSpecificStyles, BaseStyles)
    ;   BaseStyles = Defaults
    ),

    % Merge with existing props
    merge_property_list(BaseStyles, Props, StyledProps).

% ============================================================================
% VALIDATION
% ============================================================================

%% validate_theme(+Theme, -Errors)
%
%  Validate a theme definition, returning any errors found.
%
validate_theme(Theme, Errors) :-
    findall(Error, validate_theme_property(Theme, Error), Errors).

validate_theme_property(Theme, missing_name) :-
    \+ theme_property(Theme, name, _).

validate_theme_property(Theme, missing_node_defaults) :-
    \+ theme_property(Theme, node_defaults, _).

validate_theme_property(Theme, invalid_color(Prop, Color)) :-
    theme_property(Theme, node_defaults, Defaults),
    color_property(Prop),
    PropTerm =.. [Prop, Color],
    member(PropTerm, Defaults),
    \+ valid_color(Color).

validate_theme_property(Theme, invalid_numeric(Prop, Value)) :-
    theme_property(Theme, node_defaults, Defaults),
    numeric_property(Prop),
    PropTerm =.. [Prop, Value],
    member(PropTerm, Defaults),
    \+ number(Value).

%% color_property(+Property)
color_property(fill).
color_property(stroke).
color_property(text_color).
color_property(background).

%% numeric_property(+Property)
numeric_property(stroke_width).
numeric_property(font_size).
numeric_property(padding).
numeric_property(border_radius).
numeric_property(opacity).

%% valid_color(+Color)
valid_color(Color) :-
    atom(Color),
    (   sub_atom(Color, 0, 1, _, '#')
    ;   sub_atom(Color, 0, 4, _, 'rgb(')
    ;   sub_atom(Color, 0, 5, _, 'rgba(')
    ;   sub_atom(Color, 0, 4, _, 'hsl(')
    ;   named_color(Color)
    ),
    !.

%% named_color(+Color)
named_color(none).
named_color(white).
named_color(black).
named_color(red).
named_color(blue).
named_color(green).
named_color(yellow).
named_color(transparent).

% ============================================================================
% TESTING
% ============================================================================

test_theme_system :-
    format('~n=== Theme System Tests ===~n~n'),

    % Test 1: Get built-in theme
    format('Test 1: Get built-in theme...~n'),
    (   get_theme(light, LightTheme),
        theme_property(LightTheme, name, light)
    ->  format('  PASS: Light theme retrieved~n')
    ;   format('  FAIL: Could not get light theme~n')
    ),

    % Test 2: List themes
    format('~nTest 2: List themes...~n'),
    list_themes(Themes),
    (   member(light, Themes),
        member(dark, Themes),
        member(colorful, Themes)
    ->  format('  PASS: Built-in themes listed: ~w~n', [Themes])
    ;   format('  FAIL: Missing expected themes~n')
    ),

    % Test 3: Define custom theme
    format('~nTest 3: Define custom theme...~n'),
    define_theme(test_theme, [
        name(test_theme),
        node_defaults([fill('#ff0000')])
    ]),
    (   custom_theme(test_theme, _)
    ->  format('  PASS: Custom theme defined~n')
    ;   format('  FAIL: Custom theme not stored~n')
    ),

    % Test 4: Theme inheritance
    format('~nTest 4: Theme inheritance...~n'),
    define_custom_theme(dark_variant, extends(dark), [
        node_defaults([
            fill('#1a1a1a')
        ])
    ]),
    (   get_theme(dark_variant, DarkVariant),
        theme_property(DarkVariant, node_defaults, DarkNodeDefaults),
        member(fill('#1a1a1a'), DarkNodeDefaults),
        member(stroke(_), DarkNodeDefaults)  % Inherited from dark
    ->  format('  PASS: Theme inheritance working~n')
    ;   format('  FAIL: Inheritance not working~n')
    ),

    % Test 5: Apply theme to node
    format('~nTest 5: Apply theme to node...~n'),
    get_theme(light, ApplyTheme),
    apply_theme(ApplyTheme, node(test, [label("Test"), type(root)]), node(test, StyledProps)),
    (   member(fill('#4a90d9'), StyledProps),  % From root type
        member(label("Test"), StyledProps)     % Original preserved
    ->  format('  PASS: Theme applied to node~n')
    ;   format('  FAIL: Theme application incorrect~n')
    ),

    % Test 6: Validate theme
    format('~nTest 6: Validate theme...~n'),
    validate_theme([node_defaults([fill('#fff')])], Errors),
    (   member(missing_name, Errors)
    ->  format('  PASS: Validation detected missing name~n')
    ;   format('  FAIL: Validation incorrect~n')
    ),

    % Cleanup
    clear_custom_themes,

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('Theme system module loaded~n', [])
), now).
