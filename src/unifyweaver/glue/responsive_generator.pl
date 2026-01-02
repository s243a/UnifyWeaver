% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Responsive Design Generator - Declarative Breakpoints and Media Queries
%
% This module provides declarative responsive design specifications that
% generate CSS media queries for adaptive layouts.
%
% Usage:
%   % Define breakpoints
%   breakpoint(mobile, max_width(767)).
%   breakpoint(tablet, range(768, 1023)).
%   breakpoint(desktop, min_width(1024)).
%
%   % Define responsive layout variants
%   responsive_layout(my_layout, [
%       default([columns(["320px", "1fr"])]),
%       at(mobile, [columns(["1fr"]), stack(vertical)]),
%       at(tablet, [columns(["240px", "1fr"])])
%   ]).
%
%   % Generate responsive CSS
%   ?- generate_responsive_css(my_layout, CSS).

:- module(responsive_generator, [
    % Breakpoint definitions
    breakpoint/2,                   % breakpoint(+Name, +Condition)

    % Responsive layout specifications
    responsive_layout/2,            % responsive_layout(+Name, +Variants)
    responsive_style/2,             % responsive_style(+Name, +Variants)

    % Generation predicates
    generate_responsive_css/2,      % generate_responsive_css(+Name, -CSS)
    generate_media_query/2,         % generate_media_query(+Breakpoint, -MediaQuery)
    generate_breakpoint_css/4,      % generate_breakpoint_css(+Breakpoint, +ClassName, +Opts, -CSS)

    % Container queries (modern CSS)
    container/2,                    % container(+Name, +Options)
    generate_container_css/2,       % generate_container_css(+Name, -CSS)

    % Utility predicates
    breakpoint_order/2,             % breakpoint_order(+Name, -Order)
    is_mobile_first/0,              % is_mobile_first - check strategy

    % Management
    declare_breakpoint/2,           % declare_breakpoint(+Name, +Condition)
    declare_responsive_layout/2,    % declare_responsive_layout(+Name, +Variants)
    set_responsive_strategy/1,      % set_responsive_strategy(+Strategy) - mobile_first|desktop_first
    clear_responsive/0,             % clear_responsive

    % Testing
    test_responsive_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic breakpoint/2.
:- dynamic responsive_layout/2.
:- dynamic responsive_style/2.
:- dynamic container/2.
:- dynamic responsive_strategy/1.

:- discontiguous breakpoint/2.
:- discontiguous responsive_layout/2.

% Default strategy: mobile-first
responsive_strategy(mobile_first).

% ============================================================================
% DEFAULT BREAKPOINTS
% ============================================================================

% Standard breakpoints (mobile-first approach)
breakpoint(xs, max_width(575)).
breakpoint(sm, min_width(576)).
breakpoint(md, min_width(768)).
breakpoint(lg, min_width(992)).
breakpoint(xl, min_width(1200)).
breakpoint(xxl, min_width(1400)).

% Semantic breakpoints
breakpoint(mobile, max_width(767)).
breakpoint(tablet, range(768, 1023)).
breakpoint(desktop, min_width(1024)).
breakpoint(wide, min_width(1440)).

% Device-specific breakpoints
breakpoint(phone_portrait, max_width(480)).
breakpoint(phone_landscape, range(481, 767)).
breakpoint(tablet_portrait, range(768, 1024)).
breakpoint(tablet_landscape, range(1025, 1279)).

% Breakpoint ordering for cascade
breakpoint_order(xs, 1).
breakpoint_order(sm, 2).
breakpoint_order(md, 3).
breakpoint_order(lg, 4).
breakpoint_order(xl, 5).
breakpoint_order(xxl, 6).
breakpoint_order(mobile, 1).
breakpoint_order(tablet, 2).
breakpoint_order(desktop, 3).
breakpoint_order(wide, 4).

% ============================================================================
% DEFAULT RESPONSIVE LAYOUTS
% ============================================================================

% Sidebar that collapses on mobile
responsive_layout(collapsible_sidebar, [
    default([
        strategy(grid),
        areas([["sidebar", "main"]]),
        columns(["280px", "1fr"]),
        gap("1rem")
    ]),
    at(mobile, [
        areas([["main"], ["sidebar"]]),
        columns(["1fr"]),
        sidebar_position(bottom)
    ]),
    at(tablet, [
        columns(["220px", "1fr"])
    ])
]).

% Stack layout that becomes horizontal on larger screens
responsive_layout(adaptive_stack, [
    default([
        strategy(flex),
        direction(column),
        gap("1rem")
    ]),
    at(md, [
        direction(row),
        wrap(wrap)
    ]),
    at(lg, [
        direction(row),
        wrap(nowrap)
    ])
]).

% Card grid with responsive columns
responsive_layout(card_grid, [
    default([
        strategy(grid),
        columns(["1fr"]),
        gap("1rem")
    ]),
    at(sm, [
        columns(["repeat(2, 1fr)"])
    ]),
    at(md, [
        columns(["repeat(3, 1fr)"])
    ]),
    at(lg, [
        columns(["repeat(4, 1fr)"])
    ])
]).

% Dashboard layout
responsive_layout(dashboard, [
    default([
        strategy(grid),
        areas([["nav"], ["main"], ["aside"]]),
        columns(["1fr"]),
        rows(["auto", "1fr", "auto"])
    ]),
    at(md, [
        areas([["nav", "nav"], ["main", "aside"]]),
        columns(["1fr", "300px"]),
        rows(["auto", "1fr"])
    ]),
    at(lg, [
        areas([["nav", "nav", "nav"], ["sidebar", "main", "aside"]]),
        columns(["240px", "1fr", "300px"]),
        rows(["auto", "1fr"])
    ])
]).

% ============================================================================
% MEDIA QUERY GENERATION
% ============================================================================

%% generate_media_query(+Breakpoint, -MediaQuery)
%  Generate a CSS media query string for a breakpoint.
generate_media_query(Breakpoint, MediaQuery) :-
    breakpoint(Breakpoint, Condition),
    condition_to_media(Condition, MediaQuery).

%% condition_to_media(+Condition, -MediaQuery)
condition_to_media(min_width(Px), MediaQuery) :-
    format(atom(MediaQuery), '@media (min-width: ~wpx)', [Px]).

condition_to_media(max_width(Px), MediaQuery) :-
    format(atom(MediaQuery), '@media (max-width: ~wpx)', [Px]).

condition_to_media(range(MinPx, MaxPx), MediaQuery) :-
    format(atom(MediaQuery), '@media (min-width: ~wpx) and (max-width: ~wpx)', [MinPx, MaxPx]).

condition_to_media(orientation(Orientation), MediaQuery) :-
    format(atom(MediaQuery), '@media (orientation: ~w)', [Orientation]).

condition_to_media(and(Cond1, Cond2), MediaQuery) :-
    condition_to_media(Cond1, MQ1),
    condition_to_media(Cond2, MQ2),
    % Extract the condition part from each
    atom_concat('@media ', Rest1, MQ1),
    atom_concat('@media ', Rest2, MQ2),
    format(atom(MediaQuery), '@media ~w and ~w', [Rest1, Rest2]).

condition_to_media(prefers_color_scheme(Scheme), MediaQuery) :-
    format(atom(MediaQuery), '@media (prefers-color-scheme: ~w)', [Scheme]).

condition_to_media(prefers_reduced_motion, MediaQuery) :-
    MediaQuery = '@media (prefers-reduced-motion: reduce)'.

condition_to_media(print, '@media print').
condition_to_media(screen, '@media screen').

% ============================================================================
% RESPONSIVE CSS GENERATION
% ============================================================================

%% generate_responsive_css(+Name, -CSS)
%  Generate complete responsive CSS for a layout.
generate_responsive_css(Name, CSS) :-
    responsive_layout(Name, Variants),
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),
    generate_responsive_variants(ClassName, Variants, CSS).

%% generate_responsive_variants(+ClassName, +Variants, -CSS)
generate_responsive_variants(ClassName, Variants, CSS) :-
    % Extract default styles
    (member(default(DefaultOpts), Variants) -> true ; DefaultOpts = []),
    generate_layout_styles(ClassName, DefaultOpts, DefaultCSS),

    % Generate breakpoint-specific styles
    findall(BreakpointCSS, (
        member(at(Breakpoint, Opts), Variants),
        generate_breakpoint_css(Breakpoint, ClassName, Opts, BreakpointCSS)
    ), BreakpointCSSList),

    % Combine all CSS
    atomic_list_concat([DefaultCSS|BreakpointCSSList], '\n\n', CSS).

%% generate_breakpoint_css(+Breakpoint, +ClassName, +Opts, -CSS)
generate_breakpoint_css(Breakpoint, ClassName, Opts, CSS) :-
    generate_media_query(Breakpoint, MediaQuery),
    generate_layout_styles(ClassName, Opts, InnerCSS),
    format(atom(CSS), '~w {\n~w\n}', [MediaQuery, InnerCSS]).

%% generate_layout_styles(+ClassName, +Opts, -CSS)
generate_layout_styles(ClassName, Opts, CSS) :-
    (member(strategy(Strategy), Opts) -> true ; Strategy = grid),
    generate_strategy_styles(Strategy, ClassName, Opts, CSS).

%% generate_strategy_styles(+Strategy, +ClassName, +Opts, -CSS)
generate_strategy_styles(grid, ClassName, Opts, CSS) :-
    % Grid-specific styles
    (member(areas(Areas), Opts) -> generate_grid_areas(Areas, AreasCSS) ; AreasCSS = ''),
    (member(columns(Cols), Opts) -> format(atom(ColsCSS), '  grid-template-columns: ~w;', [Cols]) ; ColsCSS = ''),
    (member(rows(Rows), Opts) -> format(atom(RowsCSS), '  grid-template-rows: ~w;', [Rows]) ; RowsCSS = ''),
    (member(gap(Gap), Opts) -> format(atom(GapCSS), '  gap: ~w;', [Gap]) ; GapCSS = ''),

    findall(Line, (
        member(Line, ['  display: grid;', AreasCSS, ColsCSS, RowsCSS, GapCSS]),
        Line \= ''
    ), Lines),
    atomic_list_concat(Lines, '\n', InnerCSS),
    format(atom(CSS), '.~w {\n~w\n}', [ClassName, InnerCSS]).

generate_strategy_styles(flex, ClassName, Opts, CSS) :-
    % Flex-specific styles
    (member(direction(Dir), Opts) -> format(atom(DirCSS), '  flex-direction: ~w;', [Dir]) ; DirCSS = ''),
    (member(wrap(Wrap), Opts) -> format(atom(WrapCSS), '  flex-wrap: ~w;', [Wrap]) ; WrapCSS = ''),
    (member(justify(Just), Opts) -> format(atom(JustCSS), '  justify-content: ~w;', [Just]) ; JustCSS = ''),
    (member(align(Align), Opts) -> format(atom(AlignCSS), '  align-items: ~w;', [Align]) ; AlignCSS = ''),
    (member(gap(Gap), Opts) -> format(atom(GapCSS), '  gap: ~w;', [Gap]) ; GapCSS = ''),

    findall(Line, (
        member(Line, ['  display: flex;', DirCSS, WrapCSS, JustCSS, AlignCSS, GapCSS]),
        Line \= ''
    ), Lines),
    atomic_list_concat(Lines, '\n', InnerCSS),
    format(atom(CSS), '.~w {\n~w\n}', [ClassName, InnerCSS]).

%% generate_grid_areas(+Areas, -CSS)
generate_grid_areas(Areas, CSS) :-
    findall(RowStr, (
        member(Row, Areas),
        atomic_list_concat(Row, ' ', RowContent),
        format(atom(RowStr), '"~w"', [RowContent])
    ), RowStrs),
    atomic_list_concat(RowStrs, ' ', AreasValue),
    format(atom(CSS), '  grid-template-areas: ~w;', [AreasValue]).

% ============================================================================
% CONTAINER QUERIES
% ============================================================================

%% container(+Name, +Options)
%  Define a container query context.
container(chart_container, [
    type(inline_size),
    breakpoints([
        at(small, max_width(400)),
        at(medium, range(401, 800)),
        at(large, min_width(801))
    ])
]).

%% generate_container_css(+Name, -CSS)
generate_container_css(Name, CSS) :-
    container(Name, Options),
    atom_string(Name, NameStr),
    to_css_class(NameStr, ClassName),

    (member(type(Type), Options) -> true ; Type = inline_size),

    % Container definition
    format(atom(ContainerCSS), '.~w {\n  container-type: ~w;\n  container-name: ~w;\n}',
           [ClassName, Type, Name]),

    % Container query rules
    (member(breakpoints(Breakpoints), Options)
    ->  findall(QueryCSS, (
            member(at(BpName, Condition), Breakpoints),
            generate_container_query(Name, ClassName, BpName, Condition, QueryCSS)
        ), QueryCSSList),
        atomic_list_concat(QueryCSSList, '\n\n', QueriesCSS)
    ;   QueriesCSS = ''
    ),

    format(atom(CSS), '~w\n\n~w', [ContainerCSS, QueriesCSS]).

%% generate_container_query(+ContainerName, +ClassName, +BpName, +Condition, -CSS)
generate_container_query(ContainerName, ClassName, BpName, Condition, CSS) :-
    container_condition_to_query(ContainerName, Condition, Query),
    format(atom(CSS), '~w {\n  .~w--~w {\n    /* Styles for ~w */\n  }\n}',
           [Query, ClassName, BpName, BpName]).

%% container_condition_to_query(+Name, +Condition, -Query)
container_condition_to_query(Name, min_width(Px), Query) :-
    format(atom(Query), '@container ~w (min-width: ~wpx)', [Name, Px]).
container_condition_to_query(Name, max_width(Px), Query) :-
    format(atom(Query), '@container ~w (max-width: ~wpx)', [Name, Px]).
container_condition_to_query(Name, range(MinPx, MaxPx), Query) :-
    format(atom(Query), '@container ~w (min-width: ~wpx) and (max-width: ~wpx)', [Name, MinPx, MaxPx]).

% ============================================================================
% RESPONSIVE UTILITIES
% ============================================================================

%% is_mobile_first
%  Check if using mobile-first strategy.
is_mobile_first :-
    responsive_strategy(mobile_first).

%% set_responsive_strategy(+Strategy)
set_responsive_strategy(Strategy) :-
    member(Strategy, [mobile_first, desktop_first]),
    retractall(responsive_strategy(_)),
    assertz(responsive_strategy(Strategy)).

% ============================================================================
% VISIBILITY UTILITIES
% ============================================================================

%% generate_visibility_utilities(-CSS)
%  Generate responsive visibility utility classes.
generate_visibility_utilities(CSS) :-
    findall(UtilityCSS, (
        breakpoint(Bp, _),
        generate_visibility_for_breakpoint(Bp, UtilityCSS)
    ), UtilityCSSList),
    atomic_list_concat(UtilityCSSList, '\n\n', CSS).

generate_visibility_for_breakpoint(Bp, CSS) :-
    generate_media_query(Bp, MediaQuery),
    format(atom(CSS),
'/* Visibility utilities for ~w */
.hidden-~w {
  display: block;
}
.visible-~w {
  display: none;
}

~w {
  .hidden-~w {
    display: none;
  }
  .visible-~w {
    display: block;
  }
}', [Bp, Bp, Bp, MediaQuery, Bp, Bp]).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_breakpoint(+Name, +Condition)
declare_breakpoint(Name, Condition) :-
    retractall(breakpoint(Name, _)),
    assertz(breakpoint(Name, Condition)).

%% declare_responsive_layout(+Name, +Variants)
declare_responsive_layout(Name, Variants) :-
    retractall(responsive_layout(Name, _)),
    assertz(responsive_layout(Name, Variants)).

%% clear_responsive
clear_responsive :-
    retractall(responsive_layout(_, _)),
    retractall(responsive_style(_, _)),
    retractall(container(_, _)).

% ============================================================================
% UTILITIES
% ============================================================================

%% to_css_class(+String, -ClassName)
to_css_class(String, ClassName) :-
    atom_string(Atom, String),
    atom_codes(Atom, Codes),
    maplist(css_safe_char, Codes, SafeCodes),
    atom_codes(ClassName, SafeCodes).

css_safe_char(C, C) :- C >= 0'a, C =< 0'z, !.
css_safe_char(C, C) :- C >= 0'A, C =< 0'Z, !.
css_safe_char(C, C) :- C >= 0'0, C =< 0'9, !.
css_safe_char(0'_, 0'-) :- !.
css_safe_char(0'-, 0'-) :- !.
css_safe_char(_, 0'-).

% ============================================================================
% TESTING
% ============================================================================

test_responsive_generator :-
    format('~n========================================~n'),
    format('Responsive Generator Tests~n'),
    format('========================================~n~n'),

    % Test 1: Media query generation
    format('Test 1: Media query generation~n'),
    generate_media_query(mobile, MobileQuery),
    (sub_atom(MobileQuery, _, _, _, 'max-width')
    -> format('  PASS: Mobile query has max-width~n')
    ; format('  FAIL: Mobile query missing max-width~n')),

    generate_media_query(desktop, DesktopQuery),
    (sub_atom(DesktopQuery, _, _, _, 'min-width')
    -> format('  PASS: Desktop query has min-width~n')
    ; format('  FAIL: Desktop query missing min-width~n')),

    generate_media_query(tablet, TabletQuery),
    (sub_atom(TabletQuery, _, _, _, 'min-width'), sub_atom(TabletQuery, _, _, _, 'max-width')
    -> format('  PASS: Tablet query has range~n')
    ; format('  FAIL: Tablet query missing range~n')),

    % Test 2: Responsive layout CSS generation
    format('~nTest 2: Responsive layout CSS generation~n'),
    generate_responsive_css(collapsible_sidebar, SidebarCSS),
    (sub_atom(SidebarCSS, _, _, _, 'display: grid')
    -> format('  PASS: CSS contains display: grid~n')
    ; format('  FAIL: CSS missing display: grid~n')),
    (sub_atom(SidebarCSS, _, _, _, '@media')
    -> format('  PASS: CSS contains media queries~n')
    ; format('  FAIL: CSS missing media queries~n')),

    % Test 3: Card grid responsive
    format('~nTest 3: Card grid responsive~n'),
    generate_responsive_css(card_grid, CardCSS),
    (sub_atom(CardCSS, _, _, _, 'repeat(2, 1fr)')
    -> format('  PASS: Has 2-column breakpoint~n')
    ; format('  FAIL: Missing 2-column breakpoint~n')),
    (sub_atom(CardCSS, _, _, _, 'repeat(4, 1fr)')
    -> format('  PASS: Has 4-column breakpoint~n')
    ; format('  FAIL: Missing 4-column breakpoint~n')),

    % Test 4: Breakpoint order
    format('~nTest 4: Breakpoint ordering~n'),
    breakpoint_order(xs, Order1),
    breakpoint_order(xl, Order2),
    (Order1 < Order2
    -> format('  PASS: xs < xl in order~n')
    ; format('  FAIL: Breakpoint order incorrect~n')),

    % Test 5: Container queries
    format('~nTest 5: Container queries~n'),
    generate_container_css(chart_container, ContainerCSS),
    (sub_atom(ContainerCSS, _, _, _, 'container-type')
    -> format('  PASS: Has container-type~n')
    ; format('  FAIL: Missing container-type~n')),

    format('~nAll tests completed.~n').

:- initialization(test_responsive_generator, main).
