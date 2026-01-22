% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% ui_primitives.pl - Declarative UI primitives for cross-platform generation
%
% Provides layout, container, and component primitives that compile
% to Vue, React, Flutter, SwiftUI, and other targets.
%
% Usage:
%   use_module('src/unifyweaver/ui/ui_primitives').
%   UI = layout(stack, [spacing(16)], [
%       component(text, [content("Hello")]),
%       component(button, [label("Click")])
%   ]),
%   generate_ui(UI, vue, Code).

:- module(ui_primitives, [
    % Layout primitives
    layout/3,                    % layout(Type, Options, Children)
    validate_layout/1,           % validate_layout(LayoutSpec)

    % Container primitives
    container/3,                 % container(Type, Options, Content)
    validate_container/1,        % validate_container(ContainerSpec)

    % Component primitives
    component/2,                 % component(Type, Options)
    validate_component/1,        % validate_component(ComponentSpec)

    % Conditional rendering
    when/2,                      % when(Condition, Content)
    unless/2,                    % unless(Condition, Content)

    % Iteration
    foreach/3,                   % foreach(Items, Var, Template)

    % Pattern usage
    use_pattern/2,               % use_pattern(PatternName, Args)

    % UI tree operations
    flatten_ui/2,                % flatten_ui(UITree, FlatList)
    transform_ui/3,              % transform_ui(UITree, Transformer, Result)

    % Validation
    validate_ui/2,               % validate_ui(UISpec, Errors)

    % Code generation dispatch
    generate_ui/3,               % generate_ui(UISpec, Target, Code)

    % Testing
    test_ui_primitives/0
]).

:- use_module(library(lists)).

% ============================================================================
% LAYOUT PRIMITIVES
% ============================================================================
%
% Layouts define spatial arrangement of children.
% All layouts follow: layout(Type, Options, Children)

%! layout(+Type, +Options, +Children) is det
%  Create a layout specification.
%
%  Types:
%    stack   - Vertical or horizontal stack
%    flex    - Flexbox-style layout
%    grid    - CSS Grid-style layout
%    scroll  - Scrollable container
%    positioned - Absolute/relative positioning
%
%  Common Options:
%    direction(row|column)  - Main axis direction
%    spacing(N)             - Gap between children
%    padding(N)             - Inner padding
%    align(start|center|end|stretch) - Cross-axis alignment
%    justify(start|center|end|between|around|evenly) - Main-axis alignment
%
layout(Type, Options, Children) :-
    valid_layout_type(Type),
    is_list(Options),
    is_list(Children).

%! valid_layout_type(+Type) is semidet
valid_layout_type(stack).
valid_layout_type(flex).
valid_layout_type(grid).
valid_layout_type(scroll).
valid_layout_type(positioned).
valid_layout_type(center).      % Convenience: centers single child
valid_layout_type(wrap).        % Wrapping flex

%! validate_layout(+LayoutSpec) is semidet
%  Validate a layout specification.
validate_layout(layout(Type, Options, Children)) :-
    valid_layout_type(Type),
    validate_layout_options(Type, Options),
    maplist(validate_ui_node, Children).

validate_layout_options(stack, Options) :-
    validate_options(Options, [direction, spacing, padding, align, justify, gap]).
validate_layout_options(flex, Options) :-
    validate_options(Options, [direction, spacing, padding, align, justify, wrap, gap, grow, shrink]).
validate_layout_options(grid, Options) :-
    validate_options(Options, [columns, rows, gap, template, areas, padding]).
validate_layout_options(scroll, Options) :-
    validate_options(Options, [direction, show_scrollbar, padding]).
validate_layout_options(positioned, Options) :-
    validate_options(Options, [position, top, right, bottom, left, z_index]).
validate_layout_options(center, Options) :-
    validate_options(Options, [padding]).
validate_layout_options(wrap, Options) :-
    validate_options(Options, [spacing, align, justify]).

% ============================================================================
% CONTAINER PRIMITIVES
% ============================================================================
%
% Containers provide visual grouping and conditional rendering.
% All containers follow: container(Type, Options, Content)

%! container(+Type, +Options, +Content) is det
%  Create a container specification.
%
%  Types:
%    panel    - Styled box with background
%    card     - Elevated card with optional header/footer
%    section  - Semantic section with optional title
%    modal    - Modal dialog
%    drawer   - Slide-out drawer
%    collapse - Collapsible content
%
container(Type, Options, Content) :-
    valid_container_type(Type),
    is_list(Options),
    (is_list(Content) ; Content = layout(_, _, _) ; Content = component(_, _)).

%! valid_container_type(+Type) is semidet
valid_container_type(panel).
valid_container_type(card).
valid_container_type(section).
valid_container_type(modal).
valid_container_type(drawer).
valid_container_type(collapse).
valid_container_type(tooltip).
valid_container_type(popover).

%! validate_container(+ContainerSpec) is semidet
validate_container(container(Type, Options, Content)) :-
    valid_container_type(Type),
    validate_container_options(Type, Options),
    validate_ui_node(Content).

validate_container_options(panel, Options) :-
    validate_options(Options, [background, padding, rounded, shadow, border]).
validate_container_options(card, Options) :-
    validate_options(Options, [elevation, padding, header, footer, rounded]).
validate_container_options(section, Options) :-
    validate_options(Options, [title, collapsible, collapsed, padding]).
validate_container_options(modal, Options) :-
    validate_options(Options, [title, on_close, dismissable, size, backdrop]).
validate_container_options(drawer, Options) :-
    validate_options(Options, [position, size, on_close]).
validate_container_options(collapse, Options) :-
    validate_options(Options, [expanded, on_toggle, header]).
validate_container_options(tooltip, Options) :-
    validate_options(Options, [content, position, delay]).
validate_container_options(popover, Options) :-
    validate_options(Options, [content, trigger, position]).

% ============================================================================
% COMPONENT PRIMITIVES
% ============================================================================
%
% Components are atomic UI elements.
% All components follow: component(Type, Options)

%! component(+Type, +Options) is det
%  Create a component specification.
%
%  Text Components:
%    text     - Plain text
%    heading  - Heading (h1-h6)
%    label    - Form label
%    link     - Hyperlink
%    code     - Code snippet
%
%  Input Components:
%    text_input  - Text input field
%    textarea    - Multi-line text
%    checkbox    - Checkbox
%    radio       - Radio button group
%    select      - Dropdown select
%    switch      - Toggle switch
%    slider      - Range slider
%
%  Button Components:
%    button      - Standard button
%    icon_button - Icon-only button
%
%  Display Components:
%    icon        - Icon
%    image       - Image
%    avatar      - User avatar
%    badge       - Badge/chip
%    divider     - Visual divider
%    spacer      - Empty space
%
%  Feedback Components:
%    spinner     - Loading spinner
%    progress    - Progress indicator
%    skeleton    - Loading skeleton
%
%  Navigation Components:
%    tabs        - Tab navigation
%    breadcrumb  - Breadcrumb navigation
%    menu        - Menu list
%
component(Type, Options) :-
    valid_component_type(Type),
    is_list(Options).

%! valid_component_type(+Type) is semidet
% Text
valid_component_type(text).
valid_component_type(heading).
valid_component_type(label).
valid_component_type(link).
valid_component_type(code).
% Input
valid_component_type(text_input).
valid_component_type(textarea).
valid_component_type(checkbox).
valid_component_type(radio).
valid_component_type(select).
valid_component_type(switch).
valid_component_type(slider).
valid_component_type(number_input).
valid_component_type(date_input).
valid_component_type(file_input).
% Buttons
valid_component_type(button).
valid_component_type(icon_button).
% Display
valid_component_type(icon).
valid_component_type(image).
valid_component_type(avatar).
valid_component_type(badge).
valid_component_type(divider).
valid_component_type(spacer).
% Feedback
valid_component_type(spinner).
valid_component_type(progress).
valid_component_type(skeleton).
valid_component_type(alert).
valid_component_type(toast).
% Navigation
valid_component_type(tabs).
valid_component_type(breadcrumb).
valid_component_type(menu).
valid_component_type(pagination).

%! validate_component(+ComponentSpec) is semidet
validate_component(component(Type, Options)) :-
    valid_component_type(Type),
    validate_component_options(Type, Options).

% Text component options
validate_component_options(text, Options) :-
    validate_options(Options, [content, style, color, size, weight, align]).
validate_component_options(heading, Options) :-
    validate_options(Options, [level, content, style]).
validate_component_options(label, Options) :-
    validate_options(Options, [for, text, required]).
validate_component_options(link, Options) :-
    validate_options(Options, [href, label, target, on_click]).
validate_component_options(code, Options) :-
    validate_options(Options, [content, language, inline]).

% Input component options
validate_component_options(text_input, Options) :-
    validate_options(Options, [type, bind, placeholder, label, disabled, required, error, on_change]).
validate_component_options(textarea, Options) :-
    validate_options(Options, [bind, placeholder, rows, label, disabled, required]).
validate_component_options(checkbox, Options) :-
    validate_options(Options, [bind, label, disabled]).
validate_component_options(radio, Options) :-
    validate_options(Options, [bind, options, label, disabled]).
validate_component_options(select, Options) :-
    validate_options(Options, [bind, options, placeholder, label, disabled, multiple]).
validate_component_options(switch, Options) :-
    validate_options(Options, [bind, label, disabled]).
validate_component_options(slider, Options) :-
    validate_options(Options, [bind, min, max, step, label]).

% Button component options
validate_component_options(button, Options) :-
    validate_options(Options, [label, on_click, variant, size, disabled, loading, icon]).
validate_component_options(icon_button, Options) :-
    validate_options(Options, [icon, on_click, aria_label, variant, size, disabled]).

% Display component options
validate_component_options(icon, Options) :-
    validate_options(Options, [name, size, color]).
validate_component_options(image, Options) :-
    validate_options(Options, [src, alt, width, height, fit]).
validate_component_options(avatar, Options) :-
    validate_options(Options, [src, name, size, status]).
validate_component_options(badge, Options) :-
    validate_options(Options, [content, variant, size]).
validate_component_options(divider, Options) :-
    validate_options(Options, [orientation, margin]).
validate_component_options(spacer, Options) :-
    validate_options(Options, [size, direction]).

% Feedback component options
validate_component_options(spinner, Options) :-
    validate_options(Options, [size, color]).
validate_component_options(progress, Options) :-
    validate_options(Options, [value, max, variant, label, show_value]).
validate_component_options(skeleton, Options) :-
    validate_options(Options, [variant, width, height, animation]).
validate_component_options(alert, Options) :-
    validate_options(Options, [message, variant, dismissable, icon]).
validate_component_options(toast, Options) :-
    validate_options(Options, [message, variant, duration, position]).

% Navigation component options
validate_component_options(tabs, Options) :-
    validate_options(Options, [items, active, on_change, variant]).
validate_component_options(breadcrumb, Options) :-
    validate_options(Options, [items, separator]).
validate_component_options(menu, Options) :-
    validate_options(Options, [items, on_select]).
validate_component_options(pagination, Options) :-
    validate_options(Options, [total, page, per_page, on_change]).

% Default: accept any options
validate_component_options(_, _).

% ============================================================================
% CONDITIONAL RENDERING
% ============================================================================

%! when(+Condition, +Content) is det
%  Render content only when condition is true.
when(Condition, Content) :-
    (atom(Condition) ; is_list(Condition) ; compound(Condition)),
    validate_ui_node(Content).

%! unless(+Condition, +Content) is det
%  Render content only when condition is false.
unless(Condition, Content) :-
    when(Condition, Content).

% ============================================================================
% ITERATION
% ============================================================================

%! foreach(+Items, +Var, +Template) is det
%  Repeat template for each item in list.
foreach(Items, Var, Template) :-
    (atom(Items) ; is_list(Items)),
    atom(Var),
    validate_ui_node(Template).

% ============================================================================
% PATTERN USAGE
% ============================================================================

%! use_pattern(+PatternName, +Args) is det
%  Instantiate a named pattern with arguments.
use_pattern(PatternName, Args) :-
    atom(PatternName),
    is_list(Args).

% ============================================================================
% UI TREE VALIDATION
% ============================================================================

%! validate_ui_node(+Node) is semidet
%  Validate any UI node.
validate_ui_node(layout(Type, Options, Children)) :-
    validate_layout(layout(Type, Options, Children)), !.
validate_ui_node(container(Type, Options, Content)) :-
    validate_container(container(Type, Options, Content)), !.
validate_ui_node(component(Type, Options)) :-
    validate_component(component(Type, Options)), !.
validate_ui_node(when(Cond, Content)) :-
    when(Cond, Content), !.
validate_ui_node(unless(Cond, Content)) :-
    unless(Cond, Content), !.
validate_ui_node(foreach(Items, Var, Template)) :-
    foreach(Items, Var, Template), !.
validate_ui_node(use_pattern(Name, Args)) :-
    use_pattern(Name, Args), !.
validate_ui_node([]) :- !.
validate_ui_node([H|T]) :-
    validate_ui_node(H),
    validate_ui_node(T), !.

%! validate_ui(+UISpec, -Errors) is det
%  Validate a UI specification and collect errors.
validate_ui(UISpec, Errors) :-
    findall(Error, validate_ui_error(UISpec, Error), Errors).

validate_ui_error(UISpec, Error) :-
    \+ validate_ui_node(UISpec),
    Error = invalid_ui_spec(UISpec).

% ============================================================================
% OPTION VALIDATION HELPERS
% ============================================================================

%! validate_options(+Options, +AllowedKeys) is semidet
%  Check that all option keys are in the allowed list.
validate_options([], _).
validate_options([Opt|Rest], Allowed) :-
    Opt =.. [Key|_],
    (member(Key, Allowed) -> true ; true),  % Warning only, don't fail
    validate_options(Rest, Allowed).

%! get_option(+Key, +Options, -Value) is semidet
%  Get an option value.
get_option(Key, Options, Value) :-
    Term =.. [Key, Value],
    member(Term, Options), !.

%! get_option(+Key, +Options, -Value, +Default) is det
%  Get an option value with default.
get_option(Key, Options, Value, _Default) :-
    get_option(Key, Options, Value), !.
get_option(_Key, _Options, Default, Default).

% ============================================================================
% UI TREE OPERATIONS
% ============================================================================

%! flatten_ui(+UITree, -FlatList) is det
%  Flatten a UI tree to a list of primitives.
flatten_ui(layout(Type, Options, Children), [layout(Type, Options)|Flat]) :-
    maplist(flatten_ui, Children, ChildFlats),
    append(ChildFlats, Flat).
flatten_ui(container(Type, Options, Content), [container(Type, Options)|Flat]) :-
    flatten_ui(Content, Flat).
flatten_ui(component(Type, Options), [component(Type, Options)]).
flatten_ui(when(Cond, Content), [when(Cond)|Flat]) :-
    flatten_ui(Content, Flat).
flatten_ui(foreach(Items, Var, Template), [foreach(Items, Var)|Flat]) :-
    flatten_ui(Template, Flat).
flatten_ui([], []).
flatten_ui([H|T], Flat) :-
    flatten_ui(H, HFlat),
    flatten_ui(T, TFlat),
    append(HFlat, TFlat, Flat).

%! transform_ui(+UITree, +Transformer, -Result) is det
%  Apply a transformer predicate to each node.
transform_ui(Node, Transformer, Result) :-
    call(Transformer, Node, Transformed),
    (   Transformed = layout(Type, Options, Children)
    ->  maplist(transform_child(Transformer), Children, NewChildren),
        Result = layout(Type, Options, NewChildren)
    ;   Transformed = container(Type, Options, Content)
    ->  transform_ui(Content, Transformer, NewContent),
        Result = container(Type, Options, NewContent)
    ;   Result = Transformed
    ).

transform_child(Transformer, Child, NewChild) :-
    transform_ui(Child, Transformer, NewChild).

% ============================================================================
% CODE GENERATION DISPATCH
% ============================================================================

%! generate_ui(+UISpec, +Target, -Code) is det
%  Generate target code from UI specification.
%
%  Targets: vue, react, flutter, swiftui, html
%
generate_ui(UISpec, Target, Code) :-
    validate_ui_node(UISpec),
    generate_ui_for_target(Target, UISpec, Code).

generate_ui_for_target(vue, UISpec, Code) :-
    generate_vue_ui(UISpec, Code).
generate_ui_for_target(react, UISpec, Code) :-
    generate_react_ui(UISpec, Code).
generate_ui_for_target(html, UISpec, Code) :-
    generate_html_ui(UISpec, Code).
generate_ui_for_target(flutter, UISpec, Code) :-
    generate_flutter_ui(UISpec, Code).
generate_ui_for_target(swiftui, UISpec, Code) :-
    generate_swiftui_ui(UISpec, Code).

% Placeholder generators - to be implemented in separate modules
generate_vue_ui(_UISpec, 'TODO: Vue generation').
generate_react_ui(_UISpec, 'TODO: React generation').
generate_html_ui(_UISpec, 'TODO: HTML generation').
generate_flutter_ui(_UISpec, 'TODO: Flutter generation').
generate_swiftui_ui(_UISpec, 'TODO: SwiftUI generation').

% ============================================================================
% TESTING
% ============================================================================

test_ui_primitives :-
    format('~n=== UI Primitives Tests ===~n~n'),

    % Test 1: Layout validation
    format('Test 1: Layout validation...~n'),
    (   validate_layout(layout(stack, [spacing(16)], []))
    ->  format('  PASS: Stack layout valid~n')
    ;   format('  FAIL: Stack layout invalid~n')
    ),

    % Test 2: Container validation
    format('~nTest 2: Container validation...~n'),
    (   validate_container(container(panel, [padding(20)], []))
    ->  format('  PASS: Panel container valid~n')
    ;   format('  FAIL: Panel container invalid~n')
    ),

    % Test 3: Component validation
    format('~nTest 3: Component validation...~n'),
    (   validate_component(component(button, [label("Click"), on_click(submit)]))
    ->  format('  PASS: Button component valid~n')
    ;   format('  FAIL: Button component invalid~n')
    ),

    % Test 4: Nested UI validation
    format('~nTest 4: Nested UI validation...~n'),
    NestedUI = layout(stack, [spacing(16)], [
        component(heading, [level(1), content("Title")]),
        container(panel, [padding(20)], [
            component(text, [content("Hello")]),
            component(button, [label("Click")])
        ])
    ]),
    (   validate_ui_node(NestedUI)
    ->  format('  PASS: Nested UI valid~n')
    ;   format('  FAIL: Nested UI invalid~n')
    ),

    % Test 5: Conditional
    format('~nTest 5: Conditional validation...~n'),
    (   validate_ui_node(when(user, component(text, [content("Logged in")])))
    ->  format('  PASS: Conditional valid~n')
    ;   format('  FAIL: Conditional invalid~n')
    ),

    % Test 6: Foreach
    format('~nTest 6: Foreach validation...~n'),
    (   validate_ui_node(foreach(items, item, component(text, [content(item)])))
    ->  format('  PASS: Foreach valid~n')
    ;   format('  FAIL: Foreach invalid~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('UI Primitives module loaded~n')
), now).
