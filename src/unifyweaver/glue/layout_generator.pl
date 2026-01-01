% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Layout Generator - Declarative UI Layout and Styling System
%
% This module provides declarative layout specifications that generate
% CSS Grid, Flexbox, or absolute positioning styles.
%
% Usage:
%   % Define a layout
%   layout(my_app, grid, [
%       areas([["sidebar", "main"]]),
%       columns(["320px", "1fr"]),
%       gap("1rem")
%   ]).
%
%   % Generate CSS
%   ?- generate_layout_css(my_app, CSS).

:- module(layout_generator, [
    % Layout definition predicates
    layout/3,                       % layout(+Name, +Strategy, +Options)
    default_layout/3,               % default_layout(+Pattern, +Strategy, +Options)

    % Subplot layout (internal to component)
    subplot_layout/3,               % subplot_layout(+Name, +Strategy, +Options)
    subplot_content/3,              % subplot_content(+Name, +Position, +Content)

    % Style predicates
    style/2,                        % style(+Component, +Properties)
    style/3,                        % style(+Component, +Selector, +Properties)
    theme/2,                        % theme(+Name, +Properties)
    component_theme/2,              % component_theme(+Component, +Theme)

    % Placement predicates
    place/3,                        % place(+Layout, +Region, +Components)

    % Control predicates
    control/3,                      % control(+Name, +Type, +Options)
    control_panel/2,                % control_panel(+Name, +Controls)

    % Control generation
    generate_control_jsx/2,         % generate_control_jsx(+ControlName, -JSX)
    generate_control_panel_jsx/2,   % generate_control_panel_jsx(+PanelName, -JSX)
    generate_control_css/2,         % generate_control_css(+PanelName, -CSS)
    generate_control_state/2,       % generate_control_state(+PanelName, -StateCode)
    generate_control_handlers/2,    % generate_control_handlers(+PanelName, -HandlersCode)

    % Control wiring (connects controls to visualization components)
    wiring_spec/2,                  % wiring_spec(+Name, +Mappings)
    generate_wired_component/3,     % generate_wired_component(+Name, +Options, -Code)
    generate_control_props/2,       % generate_control_props(+PanelName, -PropsCode)
    generate_prop_types/2,          % generate_prop_types(+PanelName, -TypesCode)

    % Generation predicates
    generate_layout_css/2,          % generate_layout_css(+Name, -CSS)
    generate_layout_css/3,          % generate_layout_css(+Name, +Options, -CSS)
    generate_theme_css/2,           % generate_theme_css(+Theme, -CSS)
    generate_component_styles/2,    % generate_component_styles(+Component, -CSS)
    generate_full_styles/2,         % generate_full_styles(+Component, -CSS)

    % HTML generation
    generate_layout_html/2,         % generate_layout_html(+Name, -HTML)
    generate_layout_jsx/2,          % generate_layout_jsx(+Name, -JSX)

    % Subplot generation (target-aware)
    generate_subplot_css/2,         % generate_subplot_css(+Name, -CSS) - for web (synthesized)
    generate_subplot_jsx/2,         % generate_subplot_jsx(+Name, -JSX) - for web
    generate_subplot_matplotlib/2,  % generate_subplot_matplotlib(+Name, -Code) - for matplotlib

    % Utility predicates
    get_layout_regions/2,           % get_layout_regions(+Name, -Regions)
    has_layout/1,                   % has_layout(+Name)
    has_subplot_layout/1,           % has_subplot_layout(+Name)

    % Management
    declare_layout/3,               % declare_layout(+Name, +Strategy, +Options)
    declare_style/2,                % declare_style(+Component, +Properties)
    declare_theme/2,                % declare_theme(+Name, +Properties)
    clear_layouts/0,                % clear_layouts

    % Testing
    test_layout_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic layout/3.
:- dynamic subplot_layout/3.
:- dynamic subplot_content/3.
:- dynamic style/2.
:- dynamic style/3.
:- dynamic theme/2.
:- dynamic component_theme/2.
:- dynamic place/3.
:- dynamic wrapper/3.
:- dynamic raw_css/2.
:- dynamic control/3.
:- dynamic control_panel/2.
:- dynamic wiring_spec/2.

:- discontiguous layout/3.
:- discontiguous subplot_layout/3.
:- discontiguous subplot_content/3.
:- discontiguous style/2.
:- discontiguous style/3.
:- discontiguous theme/2.
:- discontiguous control/3.
:- discontiguous control_panel/2.

% ============================================================================
% DEFAULT LAYOUTS
% ============================================================================

%% default_layout(+Pattern, +Strategy, +Options)
%  Predefined layout patterns for common use cases.

default_layout(single, grid, [
    areas([["content"]]),
    columns(["1fr"]),
    rows(["1fr"])
]).

default_layout(sidebar_content, grid, [
    areas([["sidebar", "content"]]),
    columns(["320px", "1fr"]),
    rows(["1fr"]),
    gap("0")
]).

default_layout(content_sidebar, grid, [
    areas([["content", "sidebar"]]),
    columns(["1fr", "320px"]),
    rows(["1fr"]),
    gap("0")
]).

default_layout(header_content, grid, [
    areas([["header"], ["content"]]),
    columns(["1fr"]),
    rows(["auto", "1fr"])
]).

default_layout(header_content_footer, grid, [
    areas([["header"], ["content"], ["footer"]]),
    columns(["1fr"]),
    rows(["auto", "1fr", "auto"])
]).

default_layout(dashboard, grid, [
    areas([
        ["header", "header"],
        ["sidebar", "content"],
        ["sidebar", "footer"]
    ]),
    columns(["280px", "1fr"]),
    rows(["60px", "1fr", "40px"]),
    gap("0")
]).

default_layout(holy_grail, grid, [
    areas([
        ["header", "header", "header"],
        ["left", "content", "right"],
        ["footer", "footer", "footer"]
    ]),
    columns(["200px", "1fr", "200px"]),
    rows(["auto", "1fr", "auto"]),
    gap("1rem")
]).

% ============================================================================
% DEFAULT THEMES
% ============================================================================

theme(dark, [
    background('#1a1a2e'),
    surface('#16213e'),
    text('#e0e0e0'),
    text_secondary('#888888'),
    accent('#00d4ff'),
    accent_secondary('#7c3aed'),
    border('rgba(255,255,255,0.1)'),
    shadow('rgba(0,0,0,0.3)'),
    success('#22c55e'),
    warning('#f59e0b'),
    error('#ef4444')
]).

theme(light, [
    background('#f8fafc'),
    surface('#ffffff'),
    text('#1e293b'),
    text_secondary('#64748b'),
    accent('#7c3aed'),
    accent_secondary('#00d4ff'),
    border('#e2e8f0'),
    shadow('rgba(0,0,0,0.1)'),
    success('#16a34a'),
    warning('#d97706'),
    error('#dc2626')
]).

theme(midnight, [
    background('#0f0f1a'),
    surface('#1a1a2e'),
    text('#f0f0f0'),
    text_secondary('#a0a0a0'),
    accent('#ff6b6b'),
    accent_secondary('#4ecdc4'),
    border('rgba(255,255,255,0.05)'),
    shadow('rgba(0,0,0,0.5)'),
    success('#4ade80'),
    warning('#fbbf24'),
    error('#f87171')
]).

% ============================================================================
% CSS GENERATION - GRID LAYOUT
% ============================================================================

%% generate_layout_css(+Name, -CSS)
%  Generate CSS from a layout specification.
generate_layout_css(Name, CSS) :-
    generate_layout_css(Name, [], CSS).

%% generate_layout_css(+Name, +Options, -CSS)
generate_layout_css(Name, Options, CSS) :-
    (   layout(Name, Strategy, LayoutOpts)
    ->  generate_strategy_css(Name, Strategy, LayoutOpts, Options, CSS)
    ;   % Fall back to default single layout
        default_layout(single, Strategy, LayoutOpts),
        generate_strategy_css(Name, Strategy, LayoutOpts, Options, CSS)
    ).

%% generate_strategy_css(+Name, +Strategy, +LayoutOpts, +GenOpts, -CSS)
generate_strategy_css(Name, grid, LayoutOpts, _GenOpts, CSS) :-
    generate_grid_css(Name, LayoutOpts, CSS).

generate_strategy_css(Name, flex, LayoutOpts, _GenOpts, CSS) :-
    generate_flex_css(Name, LayoutOpts, CSS).

generate_strategy_css(Name, absolute, LayoutOpts, _GenOpts, CSS) :-
    generate_absolute_css(Name, LayoutOpts, CSS).

%% generate_grid_css(+Name, +Options, -CSS)
generate_grid_css(Name, Options, CSS) :-
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),

    % Extract grid options
    (member(areas(Areas), Options) -> true ; Areas = [["content"]]),
    (member(columns(Cols), Options) -> true ; Cols = ["1fr"]),
    (member(rows(Rows), Options) -> true ; Rows = ["auto"]),
    (member(gap(Gap), Options) -> true ; Gap = "0"),

    % Generate grid-template-areas
    generate_grid_areas_str(Areas, AreasStr),

    % Generate grid-template-columns
    atomic_list_concat(Cols, ' ', ColsStr),

    % Generate grid-template-rows
    atomic_list_concat(Rows, ' ', RowsStr),

    % Extract unique region names
    flatten(Areas, FlatAreas),
    sort(FlatAreas, Regions),

    % Generate region CSS
    generate_region_css(ClassName, Regions, RegionCSS),

    format(atom(CSS), '.~w {
    display: grid;
    grid-template-areas: ~w;
    grid-template-columns: ~w;
    grid-template-rows: ~w;
    gap: ~w;
}

~w', [ClassName, AreasStr, ColsStr, RowsStr, Gap, RegionCSS]).

%% generate_grid_areas_str(+Areas, -Str)
generate_grid_areas_str(Areas, Str) :-
    findall(RowStr, (
        member(Row, Areas),
        atomic_list_concat(Row, ' ', RowContent),
        format(atom(RowStr), '"~w"', [RowContent])
    ), RowStrs),
    atomic_list_concat(RowStrs, '\n                        ', Str).

%% generate_region_css(+ParentClass, +Regions, -CSS)
generate_region_css(ParentClass, Regions, CSS) :-
    findall(RegionRule, (
        member(Region, Regions),
        Region \= '.',  % Skip empty cells
        atom_string(Region, RegionStr),
        to_css_class(RegionStr, RegionClass),
        format(atom(RegionRule), '.~w__~w {
    grid-area: ~w;
}', [ParentClass, RegionClass, Region])
    ), Rules),
    atomic_list_concat(Rules, '\n\n', CSS).

% ============================================================================
% CSS GENERATION - FLEXBOX LAYOUT
% ============================================================================

%% generate_flex_css(+Name, +Options, -CSS)
generate_flex_css(Name, Options, CSS) :-
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),

    % Extract flex options
    (member(direction(Dir), Options) -> flex_direction_value(Dir, DirVal) ; DirVal = "row"),
    (member(wrap(Wrap), Options) -> flex_wrap_value(Wrap, WrapVal) ; WrapVal = "nowrap"),
    (member(justify(Just), Options) -> flex_justify_value(Just, JustVal) ; JustVal = "flex-start"),
    (member(align(Align), Options) -> flex_align_value(Align, AlignVal) ; AlignVal = "stretch"),
    (member(gap(Gap), Options) -> true ; Gap = "0"),

    format(atom(CSS), '.~w {
    display: flex;
    flex-direction: ~w;
    flex-wrap: ~w;
    justify-content: ~w;
    align-items: ~w;
    gap: ~w;
}', [ClassName, DirVal, WrapVal, JustVal, AlignVal, Gap]).

flex_direction_value(row, "row").
flex_direction_value(column, "column").
flex_direction_value(row_reverse, "row-reverse").
flex_direction_value(column_reverse, "column-reverse").

flex_wrap_value(wrap, "wrap").
flex_wrap_value(nowrap, "nowrap").
flex_wrap_value(wrap_reverse, "wrap-reverse").

flex_justify_value(start, "flex-start").
flex_justify_value(end, "flex-end").
flex_justify_value(center, "center").
flex_justify_value(space_between, "space-between").
flex_justify_value(space_around, "space-around").
flex_justify_value(space_evenly, "space-evenly").

flex_align_value(start, "flex-start").
flex_align_value(end, "flex-end").
flex_align_value(center, "center").
flex_align_value(stretch, "stretch").
flex_align_value(baseline, "baseline").

% ============================================================================
% CSS GENERATION - ABSOLUTE POSITIONING
% ============================================================================

%% generate_absolute_css(+Name, +Options, -CSS)
generate_absolute_css(Name, Options, CSS) :-
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),

    % Container is relative
    format(atom(ContainerCSS), '.~w {
    position: relative;
}', [ClassName]),

    % Generate region positioning
    findall(RegionCSS, (
        member(region(RegionName, Positioning), Options),
        atom_string(RegionName, RegionStr),
        to_css_class(RegionStr, RegionClass),
        generate_positioning_css(ClassName, RegionClass, Positioning, RegionCSS)
    ), RegionCSSList),

    atomic_list_concat([ContainerCSS|RegionCSSList], '\n\n', CSS).

%% generate_positioning_css(+Parent, +Region, +Positioning, -CSS)
generate_positioning_css(Parent, Region, Positioning, CSS) :-
    findall(Prop, (
        member(PropSpec, Positioning),
        positioning_to_css(PropSpec, Prop)
    ), Props),
    atomic_list_concat(Props, '\n    ', PropsStr),
    format(atom(CSS), '.~w__~w {
    position: absolute;
    ~w
}', [Parent, Region, PropsStr]).

positioning_to_css(top(V), Prop) :- format(atom(Prop), 'top: ~w;', [V]).
positioning_to_css(right(V), Prop) :- format(atom(Prop), 'right: ~w;', [V]).
positioning_to_css(bottom(V), Prop) :- format(atom(Prop), 'bottom: ~w;', [V]).
positioning_to_css(left(V), Prop) :- format(atom(Prop), 'left: ~w;', [V]).
positioning_to_css(width(V), Prop) :- format(atom(Prop), 'width: ~w;', [V]).
positioning_to_css(height(V), Prop) :- format(atom(Prop), 'height: ~w;', [V]).
positioning_to_css(transform(V), Prop) :- format(atom(Prop), 'transform: ~w;', [V]).
positioning_to_css(z_index(V), Prop) :- format(atom(Prop), 'z-index: ~w;', [V]).

% ============================================================================
% THEME CSS GENERATION
% ============================================================================

%% generate_theme_css(+ThemeName, -CSS)
generate_theme_css(ThemeName, CSS) :-
    theme(ThemeName, Props),
    findall(VarDecl, (
        member(Prop, Props),
        Prop =.. [PropName, Value],
        atom_string(PropName, PropStr),
        format(atom(VarDecl), '    --~w: ~w;', [PropStr, Value])
    ), VarDecls),
    atomic_list_concat(VarDecls, '\n', VarsStr),
    format(atom(CSS), ':root {
~w
}', [VarsStr]).

%% generate_theme_css(+ThemeName, +Selector, -CSS)
generate_theme_css(ThemeName, Selector, CSS) :-
    theme(ThemeName, Props),
    findall(VarDecl, (
        member(Prop, Props),
        Prop =.. [PropName, Value],
        atom_string(PropName, PropStr),
        format(atom(VarDecl), '    --~w: ~w;', [PropStr, Value])
    ), VarDecls),
    atomic_list_concat(VarDecls, '\n', VarsStr),
    format(atom(CSS), '~w {
~w
}', [Selector, VarsStr]).

% ============================================================================
% COMPONENT STYLE GENERATION
% ============================================================================

%% generate_component_styles(+Component, -CSS)
generate_component_styles(Component, CSS) :-
    atom_string(Component, CompStr),
    to_css_class(CompStr, ClassName),

    % Base styles
    (   style(Component, BaseProps)
    ->  generate_properties_css(BaseProps, BasePropsStr),
        format(atom(BaseCSS), '.~w {
~w
}', [ClassName, BasePropsStr])
    ;   BaseCSS = ''
    ),

    % Selector-specific styles
    findall(SelectorCSS, (
        style(Component, Selector, Props),
        generate_properties_css(Props, PropsStr),
        format(atom(SelectorCSS), '.~w~w {
~w
}', [ClassName, Selector, PropsStr])
    ), SelectorCSSList),

    % Raw CSS
    (   raw_css(Component, RawCSS)
    ->  true
    ;   RawCSS = ''
    ),

    atomic_list_concat([BaseCSS|SelectorCSSList], '\n\n', StylesCSS),
    (RawCSS = ''
    ->  CSS = StylesCSS
    ;   atomic_list_concat([StylesCSS, '\n\n', RawCSS], CSS)
    ).

%% generate_properties_css(+Props, -CSS)
generate_properties_css(Props, CSS) :-
    findall(PropStr, (
        member(Prop, Props),
        Prop =.. [PropName, Value],
        css_property_name(PropName, CSSPropName),
        format(atom(PropStr), '    ~w: ~w;', [CSSPropName, Value])
    ), PropStrs),
    atomic_list_concat(PropStrs, '\n', CSS).

%% css_property_name(+PrologName, -CSSName)
%  Convert Prolog property names to CSS property names.
css_property_name(background, 'background').
css_property_name(background_color, 'background-color').
css_property_name(color, 'color').
css_property_name(border, 'border').
css_property_name(border_radius, 'border-radius').
css_property_name(padding, 'padding').
css_property_name(margin, 'margin').
css_property_name(font_size, 'font-size').
css_property_name(font_weight, 'font-weight').
css_property_name(font_family, 'font-family').
css_property_name(box_shadow, 'box-shadow').
css_property_name(text_align, 'text-align').
css_property_name(min_height, 'min-height').
css_property_name(max_height, 'max-height').
css_property_name(min_width, 'min-width').
css_property_name(max_width, 'max-width').
css_property_name(width, 'width').
css_property_name(height, 'height').
css_property_name(overflow, 'overflow').
css_property_name(overflow_x, 'overflow-x').
css_property_name(overflow_y, 'overflow-y').
css_property_name(z_index, 'z-index').
css_property_name(opacity, 'opacity').
css_property_name(transition, 'transition').
css_property_name(transform, 'transform').
css_property_name(cursor, 'cursor').
css_property_name(Name, Name).  % Fallback: use as-is

% ============================================================================
% FULL STYLE GENERATION
% ============================================================================

%% generate_full_styles(+Component, -CSS)
%  Generate complete CSS including layout, theme, and component styles.
generate_full_styles(Component, CSS) :-
    % Layout CSS
    (   has_layout(Component)
    ->  generate_layout_css(Component, LayoutCSS)
    ;   LayoutCSS = ''
    ),

    % Theme CSS
    (   component_theme(Component, ThemeName)
    ->  atom_string(Component, CompStr),
        to_css_class(CompStr, ClassName),
        format(atom(Selector), '.~w', [ClassName]),
        generate_theme_css(ThemeName, Selector, ThemeCSS)
    ;   ThemeCSS = ''
    ),

    % Component styles
    generate_component_styles(Component, CompCSS),

    % Combine
    findall(Part, (
        member(Part, [LayoutCSS, ThemeCSS, CompCSS]),
        Part \= ''
    ), Parts),
    atomic_list_concat(Parts, '\n\n', CSS).

% ============================================================================
% HTML/JSX GENERATION
% ============================================================================

%% generate_layout_html(+Name, -HTML)
generate_layout_html(Name, HTML) :-
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),
    get_layout_regions(Name, Regions),

    findall(RegionHTML, (
        member(Region, Regions),
        Region \= '.',
        atom_string(Region, RegionStr),
        to_css_class(RegionStr, RegionClass),
        (   place(Name, Region, Components)
        ->  generate_components_html(Components, ComponentsHTML)
        ;   ComponentsHTML = ''
        ),
        format(atom(RegionHTML), '    <div class="~w__~w">
~w
    </div>', [ClassName, RegionClass, ComponentsHTML])
    ), RegionHTMLs),

    atomic_list_concat(RegionHTMLs, '\n', RegionsStr),
    format(atom(HTML), '<div class="~w">
~w
</div>', [ClassName, RegionsStr]).

%% generate_layout_jsx(+Name, -JSX)
generate_layout_jsx(Name, JSX) :-
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),
    to_pascal_case(NameStr, ComponentName),
    get_layout_regions(Name, Regions),

    findall(RegionJSX, (
        member(Region, Regions),
        Region \= '.',
        atom_string(Region, RegionStr),
        to_css_class(RegionStr, RegionClass),
        (   place(Name, Region, Components)
        ->  generate_components_jsx(Components, ComponentsJSX)
        ;   ComponentsJSX = '{/* Content */}'
        ),
        format(atom(RegionJSX), '            <div className={styles.~w__~w}>
                ~w
            </div>', [ClassName, RegionClass, ComponentsJSX])
    ), RegionJSXs),

    atomic_list_concat(RegionJSXs, '\n', RegionsStr),
    format(atom(JSX), 'import React from "react";
import styles from "./~w.module.css";

export const ~w: React.FC = () => {
    return (
        <div className={styles.~w}>
~w
        </div>
    );
};
', [ComponentName, ComponentName, ClassName, RegionsStr]).

%% generate_components_html(+Components, -HTML)
generate_components_html([], '').
generate_components_html([Comp|Rest], HTML) :-
    atom_string(Comp, CompStr),
    format(atom(CompHTML), '        <!-- ~w -->', [CompStr]),
    generate_components_html(Rest, RestHTML),
    (RestHTML = ''
    ->  HTML = CompHTML
    ;   atomic_list_concat([CompHTML, '\n', RestHTML], HTML)
    ).

%% generate_components_jsx(+Components, -JSX)
generate_components_jsx([], '{/* Content */}').
generate_components_jsx([Comp|Rest], JSX) :-
    atom_string(Comp, CompStr),
    to_pascal_case(CompStr, CompName),
    format(atom(CompJSX), '<~w />', [CompName]),
    generate_components_jsx(Rest, RestJSX),
    (RestJSX = '{/* Content */}'
    ->  JSX = CompJSX
    ;   atomic_list_concat([CompJSX, '\n                ', RestJSX], JSX)
    ).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_layout_regions(+Name, -Regions)
get_layout_regions(Name, Regions) :-
    (   layout(Name, grid, Options),
        member(areas(Areas), Options)
    ->  flatten(Areas, FlatAreas),
        sort(FlatAreas, Regions)
    ;   layout(Name, absolute, Options)
    ->  findall(R, member(region(R, _), Options), Regions)
    ;   Regions = [content]
    ).

%% has_layout(+Name)
has_layout(Name) :-
    layout(Name, _, _), !.

%% to_css_class(+String, -ClassName)
%  Convert string to valid CSS class name (kebab-case).
to_css_class(String, ClassName) :-
    atom_string(Atom, String),
    atom_codes(Atom, Codes),
    maplist(to_kebab_char, Codes, KebabCodes),
    atom_codes(ClassName, KebabCodes).

to_kebab_char(C, C) :- C >= 0'a, C =< 0'z, !.
to_kebab_char(C, C) :- C >= 0'0, C =< 0'9, !.
to_kebab_char(C, 0'-) :- C >= 0'A, C =< 0'Z, !.  % Uppercase to hyphen
to_kebab_char(0'_, 0'-) :- !.  % Underscore to hyphen
to_kebab_char(0'-, 0'-) :- !.
to_kebab_char(_, 0'-).  % Other chars to hyphen

%% to_pascal_case(+String, -PascalCase)
to_pascal_case(String, PascalCase) :-
    atom_string(Atom, String),
    atom_codes(Atom, Codes),
    to_pascal_codes(Codes, true, PascalCodes),
    atom_codes(PascalCase, PascalCodes).

to_pascal_codes([], _, []).
to_pascal_codes([C|Cs], true, [Upper|Rest]) :-
    C >= 0'a, C =< 0'z, !,
    Upper is C - 32,
    to_pascal_codes(Cs, false, Rest).
to_pascal_codes([C|Cs], true, [C|Rest]) :-
    C >= 0'A, C =< 0'Z, !,
    to_pascal_codes(Cs, false, Rest).
to_pascal_codes([C|Cs], _, Rest) :-
    (C = 0'_ ; C = 0'-), !,
    to_pascal_codes(Cs, true, Rest).
to_pascal_codes([C|Cs], _, [C|Rest]) :-
    to_pascal_codes(Cs, false, Rest).

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_layout(+Name, +Strategy, +Options)
declare_layout(Name, Strategy, Options) :-
    retractall(layout(Name, _, _)),
    assertz(layout(Name, Strategy, Options)).

%% declare_style(+Component, +Properties)
declare_style(Component, Properties) :-
    retractall(style(Component, _)),
    assertz(style(Component, Properties)).

%% declare_theme(+Name, +Properties)
declare_theme(Name, Properties) :-
    retractall(theme(Name, _)),
    assertz(theme(Name, Properties)).

%% clear_layouts/0
clear_layouts :-
    retractall(layout(_, _, _)),
    retractall(subplot_layout(_, _, _)),
    retractall(subplot_content(_, _, _)),
    retractall(style(_, _)),
    retractall(style(_, _, _)),
    retractall(component_theme(_, _)),
    retractall(place(_, _, _)),
    retractall(wrapper(_, _, _)),
    retractall(raw_css(_, _)).

% ============================================================================
% SUBPLOT LAYOUT SYSTEM
% ============================================================================
%
% Subplot layouts define internal component arrangements (multiple charts/graphs
% in a grid). These can be:
%   - Native: matplotlib handles subplots internally
%   - Synthesized: web targets create nested CSS grid with multiple components
%
% Usage:
%   subplot_layout(comparison_demo, grid, [rows(2), cols(2)]).
%   subplot_content(comparison_demo, pos(1,1), [curve(sine_wave)]).
%   subplot_content(comparison_demo, pos(1,2), [curve(cosine_wave)]).

%% has_subplot_layout(+Name)
has_subplot_layout(Name) :-
    subplot_layout(Name, _, _), !.

%% get_subplot_dimensions(+Name, -Rows, -Cols)
get_subplot_dimensions(Name, Rows, Cols) :-
    subplot_layout(Name, grid, Options),
    (member(rows(Rows), Options) -> true ; Rows = 1),
    (member(cols(Cols), Options) -> true ; Cols = 1).

%% get_subplot_positions(+Name, -Positions)
%  Get all defined subplot positions as list of pos(Row, Col)-Content pairs.
get_subplot_positions(Name, Positions) :-
    findall(pos(R, C)-Content, subplot_content(Name, pos(R, C), Content), Positions).

% ============================================================================
% SUBPLOT CSS GENERATION (for web - synthesized nested grid)
% ============================================================================

%% generate_subplot_css(+Name, -CSS)
%  Generate CSS for a subplot grid layout (web targets).
generate_subplot_css(Name, CSS) :-
    subplot_layout(Name, grid, Options),
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),

    (member(rows(Rows), Options) -> true ; Rows = 1),
    (member(cols(Cols), Options) -> true ; Cols = 1),
    (member(gap(Gap), Options) -> true ; Gap = "1rem"),

    % Generate grid-template-columns
    findall("1fr", between(1, Cols, _), ColFrs),
    atomic_list_concat(ColFrs, ' ', ColsTemplate),

    % Generate grid-template-rows
    findall("1fr", between(1, Rows, _), RowFrs),
    atomic_list_concat(RowFrs, ' ', RowsTemplate),

    format(atom(ContainerCSS), '.~w-subplot-grid {
    display: grid;
    grid-template-columns: ~w;
    grid-template-rows: ~w;
    gap: ~w;
    width: 100%;
    height: 100%;
}', [ClassName, ColsTemplate, RowsTemplate, Gap]),

    % Generate cell styles
    findall(CellCSS, (
        between(1, Rows, R),
        between(1, Cols, C),
        format(atom(CellCSS), '
.~w-subplot-grid__cell-~w-~w {
    grid-row: ~w;
    grid-column: ~w;
    min-height: 200px;
}', [ClassName, R, C, R, C])
    ), CellCSSList),

    atomic_list_concat(CellCSSList, '\n', CellsCSS),
    format(atom(CSS), '~w~w', [ContainerCSS, CellsCSS]).

%% generate_subplot_jsx(+Name, -JSX)
%  Generate JSX for a subplot grid (web targets).
generate_subplot_jsx(Name, JSX) :-
    subplot_layout(Name, grid, Options),
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),
    to_pascal_case(NameStr, ComponentName),

    (member(rows(Rows), Options) -> true ; Rows = 1),
    (member(cols(Cols), Options) -> true ; Cols = 1),

    % Generate cells
    findall(CellJSX, (
        between(1, Rows, R),
        between(1, Cols, C),
        (   subplot_content(Name, pos(R, C), Content)
        ->  generate_subplot_content_jsx(Content, ContentJSX)
        ;   ContentJSX = '{/* Empty cell */}'
        ),
        format(atom(CellJSX), '                <div className={styles["~w-subplot-grid__cell-~w-~w"]}>
                    ~w
                </div>', [ClassName, R, C, ContentJSX])
    ), CellJSXList),

    atomic_list_concat(CellJSXList, '\n', CellsJSX),

    format(atom(JSX), 'import React from "react";
import styles from "./~w.module.css";

interface ~wProps {
    className?: string;
}

export const ~w: React.FC<~wProps> = ({ className = "" }) => {
    return (
        <div className={`${styles["~w-subplot-grid"]} ${className}`}>
~w
        </div>
    );
};

export default ~w;
', [ComponentName, ComponentName, ComponentName, ComponentName, ClassName, CellsJSX, ComponentName]).

%% generate_subplot_content_jsx(+Content, -JSX)
generate_subplot_content_jsx([], '{/* Empty */}').
generate_subplot_content_jsx([curve(CurveName)|_], JSX) :-
    atom_string(CurveName, CurveStr),
    to_pascal_case(CurveStr, CurveComp),
    format(atom(JSX), '<~wChart />', [CurveComp]).
generate_subplot_content_jsx([graph(GraphName)|_], JSX) :-
    atom_string(GraphName, GraphStr),
    to_pascal_case(GraphStr, GraphComp),
    format(atom(JSX), '<~wGraph />', [GraphComp]).
generate_subplot_content_jsx([title(Title)|Rest], JSX) :-
    generate_subplot_content_jsx(Rest, RestJSX),
    format(atom(JSX), '<h3>~w</h3>~n                    ~w', [Title, RestJSX]).
generate_subplot_content_jsx([Component|_], JSX) :-
    atom(Component),
    atom_string(Component, CompStr),
    to_pascal_case(CompStr, CompName),
    format(atom(JSX), '<~w />', [CompName]).

% ============================================================================
% SUBPLOT MATPLOTLIB GENERATION (native subplots)
% ============================================================================

%% generate_subplot_matplotlib(+Name, -Code)
%  Generate matplotlib Python code for subplot layout.
generate_subplot_matplotlib(Name, Code) :-
    subplot_layout(Name, grid, Options),
    (member(rows(Rows), Options) -> true ; Rows = 1),
    (member(cols(Cols), Options) -> true ; Cols = 1),
    (member(figsize(W, H), Options) -> true ; (W = 10, H = 8)),
    (member(title(Title), Options) -> true ; Title = ""),

    % Generate subplot creation
    format(atom(CreateCode), '# Create subplot grid
fig, axes = plt.subplots(~w, ~w, figsize=(~w, ~w))
~w
', [Rows, Cols, W, H,
    (Title \= "" -> format(atom(TitleCode), 'fig.suptitle("~w")', [Title]) ; TitleCode = "")
   ]),

    % Generate axes access helper
    (   Rows =:= 1, Cols =:= 1
    ->  AxesHelper = '# Single subplot - axes is the axis directly
ax = axes
'
    ;   Rows =:= 1
    ->  AxesHelper = '# Single row - axes is 1D array
# Access with axes[col]
'
    ;   Cols =:= 1
    ->  AxesHelper = '# Single column - axes is 1D array
# Access with axes[row]
'
    ;   AxesHelper = '# Multiple rows and cols - axes is 2D array
# Access with axes[row, col] (0-indexed)
'
    ),

    % Generate content plotting
    findall(PlotCode, (
        subplot_content(Name, pos(R, C), Content),
        generate_subplot_matplotlib_content(Rows, Cols, R, C, Content, PlotCode)
    ), PlotCodes),
    atomic_list_concat(PlotCodes, '\n', PlotCodesStr),

    format(atom(Code), '~w~w~w
plt.tight_layout()
', [CreateCode, AxesHelper, PlotCodesStr]).

%% generate_subplot_matplotlib_content(+Rows, +Cols, +Row, +Col, +Content, -Code)
generate_subplot_matplotlib_content(1, 1, _, _, Content, Code) :-
    generate_subplot_content_plot('ax', Content, Code).
generate_subplot_matplotlib_content(1, Cols, _, C, Content, Code) :-
    Cols > 1,
    Idx is C - 1,
    format(atom(AxRef), 'axes[~w]', [Idx]),
    generate_subplot_content_plot(AxRef, Content, Code).
generate_subplot_matplotlib_content(Rows, 1, R, _, Content, Code) :-
    Rows > 1,
    Idx is R - 1,
    format(atom(AxRef), 'axes[~w]', [Idx]),
    generate_subplot_content_plot(AxRef, Content, Code).
generate_subplot_matplotlib_content(Rows, Cols, R, C, Content, Code) :-
    Rows > 1, Cols > 1,
    RIdx is R - 1,
    CIdx is C - 1,
    format(atom(AxRef), 'axes[~w, ~w]', [RIdx, CIdx]),
    generate_subplot_content_plot(AxRef, Content, Code).

%% generate_subplot_content_plot(+AxRef, +Content, -Code)
generate_subplot_content_plot(AxRef, Content, Code) :-
    (member(title(Title), Content) -> true ; Title = ""),
    (member(xlabel(XLabel), Content) -> true ; XLabel = ""),
    (member(ylabel(YLabel), Content) -> true ; YLabel = ""),

    % Find curves to plot
    findall(CurveName, member(curve(CurveName), Content), Curves),

    % Generate curve plotting code
    findall(CurveCode, (
        member(CurveName, Curves),
        atom_string(CurveName, CurveStr),
        format(atom(CurveCode), '~w.plot(x, ~w_y, label="~w")', [AxRef, CurveStr, CurveStr])
    ), CurveCodes),
    atomic_list_concat(CurveCodes, '\n', CurvePlots),

    % Generate configuration
    format(atom(Code), '# Subplot at ~w
~w
~w
~w
~w
~w.legend()
~w.grid(True)
', [AxRef, CurvePlots,
    (Title \= "" -> format(atom(TitleCode), '~w.set_title("~w")', [AxRef, Title]) ; TitleCode = ""),
    (XLabel \= "" -> format(atom(XLabelCode), '~w.set_xlabel("~w")', [AxRef, XLabel]) ; XLabelCode = ""),
    (YLabel \= "" -> format(atom(YLabelCode), '~w.set_ylabel("~w")', [AxRef, YLabel]) ; YLabelCode = ""),
    AxRef, AxRef]).

% ============================================================================
% CONTROL SYSTEM
% ============================================================================
%
% Controls are UI input elements (sliders, dropdowns, checkboxes, etc.)
% that can be grouped into control panels for visualization configuration.
%
% Usage:
%   control(amplitude, slider, [min(0), max(5), step(0.1), default(1), label("Amplitude")]).
%   control(curve_type, select, [options([sine, cosine]), default(sine), label("Curve")]).
%   control_panel(curve_controls, [amplitude, frequency, curve_type]).

% ============================================================================
% DEFAULT CONTROLS (examples)
% ============================================================================

control(amplitude, slider, [
    min(0), max(5), step(0.1), default(1),
    label("Amplitude")
]).

control(frequency, slider, [
    min(0.1), max(10), step(0.1), default(1),
    label("Frequency")
]).

control(phase, slider, [
    min(0), max(6.28), step(0.01), default(0),
    label("Phase")
]).

control(curve_type, select, [
    options([sine, cosine, quadratic, cubic, exponential]),
    default(sine),
    label("Curve Type")
]).

control(show_grid, checkbox, [
    default(true),
    label("Show Grid")
]).

control(show_legend, checkbox, [
    default(true),
    label("Show Legend")
]).

control(line_color, color_picker, [
    default('#00d4ff'),
    label("Line Color")
]).

control(line_width, slider, [
    min(1), max(10), step(0.5), default(2),
    label("Line Width")
]).

control(point_count, slider, [
    min(50), max(500), step(10), default(200),
    label("Point Count")
]).

control(x_min, number_input, [
    default(-10),
    label("X Min")
]).

control(x_max, number_input, [
    default(10),
    label("X Max")
]).

% ============================================================================
% DEFAULT CONTROL PANELS
% ============================================================================

control_panel(curve_controls, [amplitude, frequency, phase, curve_type]).
control_panel(display_controls, [show_grid, show_legend, line_color, line_width]).
control_panel(range_controls, [x_min, x_max, point_count]).
control_panel(full_controls, [amplitude, frequency, curve_type, show_grid, line_color]).

% ============================================================================
% CONTROL JSX GENERATION
% ============================================================================

%% generate_control_jsx(+ControlName, -JSX)
%  Generate React JSX for a single control.
generate_control_jsx(ControlName, JSX) :-
    control(ControlName, Type, Options),
    atom_string(ControlName, NameStr),
    to_camel_case(NameStr, StateVar),
    to_pascal_case(NameStr, SetterSuffix),
    (member(label(Label), Options) -> true ; Label = NameStr),
    generate_control_by_type(Type, ControlName, StateVar, SetterSuffix, Label, Options, JSX).

%% generate_control_by_type(+Type, +Name, +StateVar, +SetterSuffix, +Label, +Options, -JSX)

% Slider control
generate_control_by_type(slider, _Name, StateVar, SetterSuffix, Label, Options, JSX) :-
    (member(min(Min), Options) -> true ; Min = 0),
    (member(max(Max), Options) -> true ; Max = 100),
    (member(step(Step), Options) -> true ; Step = 1),
    format(atom(JSX),
'<div className={styles.controlGroup}>
  <label className={styles.controlLabel}>~w: {~w}</label>
  <input
    type="range"
    min={~w}
    max={~w}
    step={~w}
    value={~w}
    onChange={(e) => set~w(Number(e.target.value))}
    className={styles.slider}
  />
</div>', [Label, StateVar, Min, Max, Step, StateVar, SetterSuffix]).

% Select/dropdown control
generate_control_by_type(select, _Name, StateVar, SetterSuffix, Label, Options, JSX) :-
    (member(options(Opts), Options) -> true ; Opts = []),
    generate_select_options(Opts, OptionsJSX),
    format(atom(JSX),
'<div className={styles.controlGroup}>
  <label className={styles.controlLabel}>~w</label>
  <select
    value={~w}
    onChange={(e) => set~w(e.target.value)}
    className={styles.select}
  >
~w
  </select>
</div>', [Label, StateVar, SetterSuffix, OptionsJSX]).

% Checkbox control
generate_control_by_type(checkbox, _Name, StateVar, SetterSuffix, Label, _Options, JSX) :-
    format(atom(JSX),
'<div className={styles.controlGroup}>
  <label className={styles.checkboxLabel}>
    <input
      type="checkbox"
      checked={~w}
      onChange={(e) => set~w(e.target.checked)}
      className={styles.checkbox}
    />
    ~w
  </label>
</div>', [StateVar, SetterSuffix, Label]).

% Color picker control
generate_control_by_type(color_picker, _Name, StateVar, SetterSuffix, Label, _Options, JSX) :-
    format(atom(JSX),
'<div className={styles.controlGroup}>
  <label className={styles.controlLabel}>~w</label>
  <input
    type="color"
    value={~w}
    onChange={(e) => set~w(e.target.value)}
    className={styles.colorPicker}
  />
</div>', [Label, StateVar, SetterSuffix]).

% Number input control
generate_control_by_type(number_input, _Name, StateVar, SetterSuffix, Label, _Options, JSX) :-
    format(atom(JSX),
'<div className={styles.controlGroup}>
  <label className={styles.controlLabel}>~w</label>
  <input
    type="number"
    value={~w}
    onChange={(e) => set~w(Number(e.target.value))}
    className={styles.numberInput}
  />
</div>', [Label, StateVar, SetterSuffix]).

% Text input control
generate_control_by_type(text_input, _Name, StateVar, SetterSuffix, Label, _Options, JSX) :-
    format(atom(JSX),
'<div className={styles.controlGroup}>
  <label className={styles.controlLabel}>~w</label>
  <input
    type="text"
    value={~w}
    onChange={(e) => set~w(e.target.value)}
    className={styles.textInput}
  />
</div>', [Label, StateVar, SetterSuffix]).

% Fallback for unknown types
generate_control_by_type(_, Name, _, _, _, _, JSX) :-
    atom_string(Name, NameStr),
    format(atom(JSX), '{/* Unknown control type: ~w */}', [NameStr]).

%% generate_select_options(+Options, -JSX)
generate_select_options([], '').
generate_select_options([Opt|Rest], JSX) :-
    atom_string(Opt, OptStr),
    format(atom(OptJSX), '    <option value="~w">~w</option>', [OptStr, OptStr]),
    generate_select_options(Rest, RestJSX),
    (RestJSX = ''
    ->  JSX = OptJSX
    ;   format(atom(JSX), '~w~n~w', [OptJSX, RestJSX])
    ).

% ============================================================================
% CONTROL PANEL JSX GENERATION
% ============================================================================

%% generate_control_panel_jsx(+PanelName, -JSX)
%  Generate React JSX for a control panel containing multiple controls.
generate_control_panel_jsx(PanelName, JSX) :-
    control_panel(PanelName, ControlNames),
    atom_string(PanelName, PanelStr),
    to_pascal_case(PanelStr, PanelComponent),

    % Generate individual controls
    findall(ControlJSX, (
        member(CtrlName, ControlNames),
        generate_control_jsx(CtrlName, ControlJSX)
    ), ControlJSXList),
    atomic_list_concat(ControlJSXList, '\n      ', ControlsJSX),

    format(atom(JSX),
'<div className={styles.controlPanel}>
  <h3 className={styles.panelTitle}>~w</h3>
  <div className={styles.controls}>
      ~w
  </div>
</div>', [PanelComponent, ControlsJSX]).

% ============================================================================
% CONTROL STATE GENERATION (React useState hooks)
% ============================================================================

%% generate_control_state(+PanelName, -StateCode)
%  Generate React useState declarations for all controls in a panel.
generate_control_state(PanelName, StateCode) :-
    control_panel(PanelName, ControlNames),
    findall(StateDecl, (
        member(CtrlName, ControlNames),
        generate_single_state(CtrlName, StateDecl)
    ), StateDecls),
    atomic_list_concat(StateDecls, '\n  ', StateCode).

%% generate_single_state(+ControlName, -StateDecl)
generate_single_state(ControlName, StateDecl) :-
    control(ControlName, Type, Options),
    atom_string(ControlName, NameStr),
    to_camel_case(NameStr, StateVar),
    get_default_value(Type, Options, DefaultValue),
    format(atom(StateDecl), 'const [~w, set~w] = useState(~w);', [StateVar, StateVar, DefaultValue]).

%% get_default_value(+Type, +Options, -DefaultValue)
get_default_value(slider, Options, DefaultValue) :-
    (member(default(D), Options) -> DefaultValue = D ; DefaultValue = 0).
get_default_value(select, Options, DefaultValue) :-
    (member(default(D), Options) -> format(atom(DefaultValue), '"~w"', [D]) ; DefaultValue = '""').
get_default_value(checkbox, Options, DefaultValue) :-
    (member(default(D), Options) -> (D == true -> DefaultValue = 'true' ; DefaultValue = 'false') ; DefaultValue = 'false').
get_default_value(color_picker, Options, DefaultValue) :-
    (member(default(D), Options) -> format(atom(DefaultValue), '"~w"', [D]) ; DefaultValue = '"#000000"').
get_default_value(number_input, Options, DefaultValue) :-
    (member(default(D), Options) -> DefaultValue = D ; DefaultValue = 0).
get_default_value(text_input, Options, DefaultValue) :-
    (member(default(D), Options) -> format(atom(DefaultValue), '"~w"', [D]) ; DefaultValue = '""').
get_default_value(_, Options, DefaultValue) :-
    (member(default(D), Options) -> DefaultValue = D ; DefaultValue = 'null').

% ============================================================================
% CONTROL HANDLERS GENERATION
% ============================================================================

%% generate_control_handlers(+PanelName, -HandlersCode)
%  Generate onChange handler functions (if more complex logic is needed).
generate_control_handlers(PanelName, HandlersCode) :-
    control_panel(PanelName, ControlNames),
    findall(Handler, (
        member(CtrlName, ControlNames),
        generate_single_handler(CtrlName, Handler)
    ), Handlers),
    atomic_list_concat(Handlers, '\n  ', HandlersCode).

%% generate_single_handler(+ControlName, -Handler)
generate_single_handler(ControlName, Handler) :-
    control(ControlName, Type, _Options),
    atom_string(ControlName, NameStr),
    to_camel_case(NameStr, StateVar),
    to_pascal_case(NameStr, HandlerName),
    get_handler_type(Type, ValueExpr),
    format(atom(Handler),
'const handle~wChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    set~w(~w);
  };', [HandlerName, StateVar, ValueExpr]).

get_handler_type(slider, 'Number(e.target.value)').
get_handler_type(number_input, 'Number(e.target.value)').
get_handler_type(checkbox, 'e.target.checked').
get_handler_type(color_picker, 'e.target.value').
get_handler_type(text_input, 'e.target.value').
get_handler_type(select, 'e.target.value').
get_handler_type(_, 'e.target.value').

% ============================================================================
% CONTROL CSS GENERATION
% ============================================================================

%% generate_control_css(+PanelName, -CSS)
%  Generate CSS for control panel styling.
generate_control_css(_PanelName, CSS) :-
    CSS = '.controlPanel {
  padding: 1rem;
  background: var(--surface, #16213e);
  border-radius: 8px;
  border: 1px solid var(--border, rgba(255,255,255,0.1));
}

.panelTitle {
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary, #888);
  margin: 0 0 1rem 0;
}

.controls {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.controlGroup {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.controlLabel {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text, #e0e0e0);
}

.slider {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: var(--border, rgba(255,255,255,0.1));
  outline: none;
  -webkit-appearance: none;
  appearance: none;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent, #00d4ff);
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent, #00d4ff);
  cursor: pointer;
  border: none;
}

.select {
  padding: 0.5rem;
  background: var(--background, #1a1a2e);
  border: 1px solid var(--border, rgba(255,255,255,0.1));
  border-radius: 4px;
  color: var(--text, #e0e0e0);
  font-size: 0.875rem;
  cursor: pointer;
}

.select:focus {
  outline: none;
  border-color: var(--accent, #00d4ff);
}

.checkboxLabel {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--text, #e0e0e0);
  cursor: pointer;
}

.checkbox {
  width: 18px;
  height: 18px;
  accent-color: var(--accent, #00d4ff);
  cursor: pointer;
}

.colorPicker {
  width: 100%;
  height: 36px;
  padding: 2px;
  background: var(--background, #1a1a2e);
  border: 1px solid var(--border, rgba(255,255,255,0.1));
  border-radius: 4px;
  cursor: pointer;
}

.colorPicker::-webkit-color-swatch-wrapper {
  padding: 2px;
}

.colorPicker::-webkit-color-swatch {
  border-radius: 2px;
  border: none;
}

.numberInput,
.textInput {
  padding: 0.5rem;
  background: var(--background, #1a1a2e);
  border: 1px solid var(--border, rgba(255,255,255,0.1));
  border-radius: 4px;
  color: var(--text, #e0e0e0);
  font-size: 0.875rem;
}

.numberInput:focus,
.textInput:focus {
  outline: none;
  border-color: var(--accent, #00d4ff);
}
'.

% ============================================================================
% UTILITY - CAMEL CASE CONVERSION
% ============================================================================

%% to_camel_case(+String, -CamelCase)
%  Convert snake_case to camelCase.
to_camel_case(String, CamelCase) :-
    atom_string(Atom, String),
    atom_codes(Atom, Codes),
    to_camel_codes(Codes, false, CamelCodes),
    atom_codes(CamelCase, CamelCodes).

to_camel_codes([], _, []).
to_camel_codes([C|Cs], true, [Upper|Rest]) :-
    C >= 0'a, C =< 0'z, !,
    Upper is C - 32,
    to_camel_codes(Cs, false, Rest).
to_camel_codes([C|Cs], _, Rest) :-
    (C = 0'_ ; C = 0'-), !,
    to_camel_codes(Cs, true, Rest).
to_camel_codes([C|Cs], _, [C|Rest]) :-
    to_camel_codes(Cs, false, Rest).

% ============================================================================
% CONTROL WIRING SYSTEM
% ============================================================================

%% Default wiring specifications
%  Maps control panels to visualization component props.
%  wiring_spec(+WiringName, +Options)
%  Options:
%    panel(PanelName)         - Control panel to wire
%    component(ComponentName) - Visualization component to receive props
%    mappings(List)           - List of control_name -> prop_name mappings

wiring_spec(curve_visualization, [
    panel(curve_controls),
    component(curve_chart),
    mappings([
        amplitude -> amplitude,
        frequency -> frequency,
        phase -> phase,
        curve_type -> curveType
    ])
]).

wiring_spec(display_settings, [
    panel(display_controls),
    component(chart_display),
    mappings([
        show_grid -> showGrid,
        show_legend -> showLegend,
        line_color -> lineColor,
        line_width -> lineWidth
    ])
]).

%% generate_control_props(+PanelName, -PropsCode)
%  Generate props object to pass to visualization component.
generate_control_props(PanelName, PropsCode) :-
    control_panel(PanelName, ControlNames),
    findall(PropPair, (
        member(CtrlName, ControlNames),
        atom_string(CtrlName, NameStr),
        to_camel_case(NameStr, PropName),
        format(atom(PropPair), '~w={~w}', [PropName, PropName])
    ), PropPairs),
    atomic_list_concat(PropPairs, ' ', PropsCode).

%% generate_prop_types(+PanelName, -TypesCode)
%  Generate TypeScript interface for component props.
generate_prop_types(PanelName, TypesCode) :-
    control_panel(PanelName, ControlNames),
    findall(TypeDef, (
        member(CtrlName, ControlNames),
        control(CtrlName, Type, _Options),
        atom_string(CtrlName, NameStr),
        to_camel_case(NameStr, PropName),
        control_type_to_ts(Type, TSType),
        format(atom(TypeDef), '  ~w: ~w;', [PropName, TSType])
    ), TypeDefs),
    atomic_list_concat(TypeDefs, '\n', TypeDefsStr),
    format(atom(TypesCode),
'interface ChartProps {
~w
}', [TypeDefsStr]).

%% control_type_to_ts(+ControlType, -TSType)
control_type_to_ts(slider, 'number').
control_type_to_ts(number_input, 'number').
control_type_to_ts(checkbox, 'boolean').
control_type_to_ts(select, 'string').
control_type_to_ts(color_picker, 'string').
control_type_to_ts(text_input, 'string').
control_type_to_ts(_, 'any').

%% generate_wired_component(+Name, +Options, -Code)
%  Generate a complete React component with controls wired to state.
%  Options:
%    panel(PanelName)      - Control panel to include
%    component(Type)       - 'curve' or 'graph'
%    layout(Pattern)       - Layout pattern (sidebar_content, dashboard, etc.)
generate_wired_component(Name, Options, Code) :-
    (member(panel(PanelName), Options) -> true ; PanelName = curve_controls),
    (member(layout(LayoutPattern), Options) -> true ; LayoutPattern = sidebar_content),
    (member(component(ComponentType), Options) -> true ; ComponentType = curve),

    atom_string(Name, NameStr),
    to_pascal_case(NameStr, ComponentName),

    % Generate state declarations
    generate_control_state(PanelName, StateCode),

    % Generate control panel JSX
    generate_control_panel_jsx(PanelName, ControlPanelJSX),

    % Generate props for visualization
    generate_control_props(PanelName, PropsCode),

    % Generate prop types
    generate_prop_types(PanelName, PropTypes),

    % Generate CSS imports
    generate_control_css(PanelName, ControlCSS),

    % Generate visualization component based on type
    get_visualization_component(ComponentType, PropsCode, VizComponent),

    % Apply layout pattern
    get_layout_structure(LayoutPattern, ControlPanelJSX, VizComponent, LayoutJSX),

    format(atom(Code),
'// Generated by UnifyWeaver - Wired Component
// Component: ~w

import React, { useState } from "react";
import styles from "./~w.module.css";

~w

export const ~w: React.FC = () => {
  // Control state
  ~w

  return (
    ~w
  );
};

export default ~w;

/* CSS Styles:
~w
*/
', [Name, ComponentName, PropTypes, ComponentName, StateCode, LayoutJSX, ComponentName, ControlCSS]).

%% get_visualization_component(+Type, +Props, -Component)
get_visualization_component(curve, Props, Component) :-
    format(atom(Component), '<CurveChart ~w />', [Props]).
get_visualization_component(graph, Props, Component) :-
    format(atom(Component), '<GraphVisualization ~w />', [Props]).
get_visualization_component(_, Props, Component) :-
    format(atom(Component), '<Visualization ~w />', [Props]).

%% get_layout_structure(+Pattern, +Controls, +Viz, -JSX)
get_layout_structure(sidebar_content, Controls, Viz, JSX) :-
    format(atom(JSX),
'<div className={styles.container}>
      <aside className={styles.sidebar}>
        ~w
      </aside>
      <main className={styles.main}>
        ~w
      </main>
    </div>', [Controls, Viz]).
get_layout_structure(header_content, Controls, Viz, JSX) :-
    format(atom(JSX),
'<div className={styles.container}>
      <header className={styles.header}>
        ~w
      </header>
      <main className={styles.main}>
        ~w
      </main>
    </div>', [Controls, Viz]).
get_layout_structure(_, Controls, Viz, JSX) :-
    format(atom(JSX),
'<div className={styles.container}>
      <div className={styles.controls}>
        ~w
      </div>
      <div className={styles.visualization}>
        ~w
      </div>
    </div>', [Controls, Viz]).

% ============================================================================
% TESTS
% ============================================================================

test_layout_generator :-
    format('Testing layout_generator module...~n~n'),

    % Test grid layout
    format('Test 1: Grid layout generation~n'),
    declare_layout(test_grid, grid, [
        areas([["sidebar", "main"]]),
        columns(["320px", "1fr"]),
        gap("1rem")
    ]),
    generate_layout_css(test_grid, GridCSS),
    (sub_atom(GridCSS, _, _, _, 'display: grid')
    -> format('  PASS: Grid CSS contains display: grid~n')
    ; format('  FAIL: Grid CSS missing display: grid~n')),
    (sub_atom(GridCSS, _, _, _, 'grid-template-areas')
    -> format('  PASS: Grid CSS contains grid-template-areas~n')
    ; format('  FAIL: Grid CSS missing grid-template-areas~n')),

    % Test flex layout
    format('~nTest 2: Flex layout generation~n'),
    declare_layout(test_flex, flex, [
        direction(row),
        justify(space_between),
        gap("0.5rem")
    ]),
    generate_layout_css(test_flex, FlexCSS),
    (sub_atom(FlexCSS, _, _, _, 'display: flex')
    -> format('  PASS: Flex CSS contains display: flex~n')
    ; format('  FAIL: Flex CSS missing display: flex~n')),

    % Test theme generation
    format('~nTest 3: Theme CSS generation~n'),
    generate_theme_css(dark, ThemeCSS),
    (sub_atom(ThemeCSS, _, _, _, '--background')
    -> format('  PASS: Theme CSS contains --background variable~n')
    ; format('  FAIL: Theme CSS missing --background variable~n')),

    % Test JSX generation
    format('~nTest 4: JSX generation~n'),
    generate_layout_jsx(test_grid, JSX),
    (sub_atom(JSX, _, _, _, 'React.FC')
    -> format('  PASS: JSX contains React.FC~n')
    ; format('  FAIL: JSX missing React.FC~n')),

    % Cleanup
    clear_layouts,
    format('~nAll tests completed.~n').

:- initialization(test_layout_generator, main).
