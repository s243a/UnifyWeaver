% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% layout.pl - Cross-platform layout system
%
% Provides declarative layout primitives that compile across
% React Native, Vue, Flutter, and SwiftUI targets.
%
% Usage:
%   use_module('src/unifyweaver/layout/layout').
%   row([justify(space_between), align(center), gap(md)], Children, Spec),
%   generate_layout(Spec, react_native, Code).

:- module(layout, [
    % Layout primitives
    row/3,
    column/3,
    stack/3,
    grid/3,
    wrap/3,

    % Container layouts
    center/2,
    container/3,
    scroll_view/3,
    safe_area/2,

    % Flex properties
    flex/2,
    flex_grow/2,
    flex_shrink/2,
    flex_basis/2,

    % Alignment
    justify/2,
    align/2,
    align_self/2,

    % Spacing
    gap/2,
    padding/2,
    margin/2,

    % Sizing
    width/2,
    height/2,
    min_width/2,
    min_height/2,
    max_width/2,
    max_height/2,
    aspect_ratio/2,

    % Positioning
    position/2,
    absolute/2,
    z_index/2,

    % Responsive
    responsive/2,
    breakpoint/3,

    % Code generation
    generate_layout/3,
    generate_react_native_layout/2,
    generate_vue_layout/2,
    generate_flutter_layout/2,
    generate_swiftui_layout/2,

    % Utilities
    spacing_value/2,

    % Testing
    test_layout/0
]).

:- use_module(library(lists)).

% ============================================================================
% Spacing Scale
% ============================================================================

%! spacing_value(+Name, -Value) is semidet
%  Map spacing names to pixel values.
spacing_value(none, 0).
spacing_value(xs, 4).
spacing_value(sm, 8).
spacing_value(md, 16).
spacing_value(lg, 24).
spacing_value(xl, 32).
spacing_value(xxl, 48).
spacing_value(N, N) :- number(N).

% ============================================================================
% Layout Primitives
% ============================================================================

%! row(+Options, +Children, -Spec) is det
%  Create a horizontal row layout.
row(Options, Children, Spec) :-
    Spec = layout_spec(row, Options, Children).

%! column(+Options, +Children, -Spec) is det
%  Create a vertical column layout.
column(Options, Children, Spec) :-
    Spec = layout_spec(column, Options, Children).

%! stack(+Options, +Children, -Spec) is det
%  Create a z-axis stack (overlapping children).
stack(Options, Children, Spec) :-
    Spec = layout_spec(stack, Options, Children).

%! grid(+Options, +Children, -Spec) is det
%  Create a grid layout.
grid(Options, Children, Spec) :-
    Spec = layout_spec(grid, Options, Children).

%! wrap(+Options, +Children, -Spec) is det
%  Create a wrapping row layout.
wrap(Options, Children, Spec) :-
    Spec = layout_spec(wrap, Options, Children).

% ============================================================================
% Container Layouts
% ============================================================================

%! center(+Child, -Spec) is det
%  Center a child both horizontally and vertically.
center(Child, Spec) :-
    Spec = layout_spec(center, [], [Child]).

%! container(+Options, +Children, -Spec) is det
%  Create a container with max-width and centering.
container(Options, Children, Spec) :-
    Spec = layout_spec(container, Options, Children).

%! scroll_view(+Options, +Children, -Spec) is det
%  Create a scrollable container.
scroll_view(Options, Children, Spec) :-
    get_option(direction, Options, Direction, vertical),
    Spec = layout_spec(scroll_view, [direction(Direction)|Options], Children).

%! safe_area(+Child, -Spec) is det
%  Wrap content in safe area insets.
safe_area(Child, Spec) :-
    Spec = layout_spec(safe_area, [], [Child]).

% ============================================================================
% Flex Properties
% ============================================================================

%! flex(+Value, -Term) is det
flex(Value, flex(Value)).

%! flex_grow(+Value, -Term) is det
flex_grow(Value, flex_grow(Value)).

%! flex_shrink(+Value, -Term) is det
flex_shrink(Value, flex_shrink(Value)).

%! flex_basis(+Value, -Term) is det
flex_basis(Value, flex_basis(Value)).

% ============================================================================
% Alignment Options
% ============================================================================

%! justify(+Value, -Term) is det
%  Justify content along main axis.
%  Values: start, end, center, space_between, space_around, space_evenly
justify(Value, justify(Value)).

%! align(+Value, -Term) is det
%  Align items along cross axis.
%  Values: start, end, center, stretch, baseline
align(Value, align(Value)).

%! align_self(+Value, -Term) is det
%  Override alignment for a single item.
align_self(Value, align_self(Value)).

% ============================================================================
% Spacing Options
% ============================================================================

%! gap(+Value, -Term) is det
%  Gap between children.
gap(Value, gap(Value)).

%! padding(+Value, -Term) is det
%  Padding inside container. Can be single value or [top, right, bottom, left].
padding(Value, padding(Value)).

%! margin(+Value, -Term) is det
%  Margin outside container.
margin(Value, margin(Value)).

% ============================================================================
% Sizing Options
% ============================================================================

%! width(+Value, -Term) is det
width(Value, width(Value)).

%! height(+Value, -Term) is det
height(Value, height(Value)).

%! min_width(+Value, -Term) is det
min_width(Value, min_width(Value)).

%! min_height(+Value, -Term) is det
min_height(Value, min_height(Value)).

%! max_width(+Value, -Term) is det
max_width(Value, max_width(Value)).

%! max_height(+Value, -Term) is det
max_height(Value, max_height(Value)).

%! aspect_ratio(+Value, -Term) is det
aspect_ratio(Value, aspect_ratio(Value)).

% ============================================================================
% Positioning
% ============================================================================

%! position(+Value, -Term) is det
%  Position type: relative, absolute
position(Value, position(Value)).

%! absolute(+Edges, -Term) is det
%  Absolute positioning with edges [top, right, bottom, left].
absolute(Edges, absolute(Edges)).

%! z_index(+Value, -Term) is det
z_index(Value, z_index(Value)).

% ============================================================================
% Responsive
% ============================================================================

%! responsive(+Breakpoints, -Term) is det
%  Define responsive variations.
responsive(Breakpoints, responsive(Breakpoints)).

%! breakpoint(+Name, +MinWidth, -Term) is det
%  Define a breakpoint.
breakpoint(sm, 640, breakpoint(sm, 640)).
breakpoint(md, 768, breakpoint(md, 768)).
breakpoint(lg, 1024, breakpoint(lg, 1024)).
breakpoint(xl, 1280, breakpoint(xl, 1280)).

% ============================================================================
% Helper
% ============================================================================

get_option(Key, Options, Value, _Default) :-
    member(Opt, Options),
    Opt =.. [Key, Value],
    !.
get_option(_Key, _Options, Default, Default).

% ============================================================================
% Code Generation - Main Entry Point
% ============================================================================

%! generate_layout(+Spec, +Target, -Code) is det
%  Generate target-specific layout code.
generate_layout(Spec, Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_layout(Spec, Code)
    ;   Target = vue
    ->  generate_vue_layout(Spec, Code)
    ;   Target = flutter
    ->  generate_flutter_layout(Spec, Code)
    ;   Target = swiftui
    ->  generate_swiftui_layout(Spec, Code)
    ;   Code = ""
    ).

% ============================================================================
% React Native Layout Generation
% ============================================================================

%! generate_react_native_layout(+Spec, -Code) is det
generate_react_native_layout(layout_spec(Type, Options, Children), Code) :-
    rn_layout_type(Type, _FlexDir, BaseStyle),
    rn_options_to_style(Options, StyleProps),
    rn_children_code(Children, ChildrenCode),
    atomic_list_concat([BaseStyle|StyleProps], ', ', StyleStr),
    format(atom(Code), '<View style={{ ~w }}>\n~w</View>', [StyleStr, ChildrenCode]).

rn_layout_type(row, row, 'flexDirection: \'row\'').
rn_layout_type(column, column, 'flexDirection: \'column\'').
rn_layout_type(stack, stack, 'position: \'relative\'').
rn_layout_type(wrap, wrap, 'flexDirection: \'row\', flexWrap: \'wrap\'').
rn_layout_type(center, center, 'justifyContent: \'center\', alignItems: \'center\'').
rn_layout_type(container, container, 'width: \'100%\', maxWidth: 1200, marginHorizontal: \'auto\'').
rn_layout_type(scroll_view, scroll, 'flex: 1').
rn_layout_type(safe_area, safe, 'flex: 1').
rn_layout_type(grid, grid, 'flexDirection: \'row\', flexWrap: \'wrap\'').

rn_options_to_style(Options, Props) :-
    findall(Prop, (
        member(Opt, Options),
        rn_option_to_prop(Opt, Prop)
    ), Props).

rn_option_to_prop(justify(V), Prop) :-
    rn_justify_value(V, RNVal),
    format(atom(Prop), 'justifyContent: \'~w\'', [RNVal]).
rn_option_to_prop(align(V), Prop) :-
    rn_align_value(V, RNVal),
    format(atom(Prop), 'alignItems: \'~w\'', [RNVal]).
rn_option_to_prop(gap(V), Prop) :-
    spacing_value(V, Px),
    format(atom(Prop), 'gap: ~w', [Px]).
rn_option_to_prop(padding(V), Prop) :-
    spacing_value(V, Px),
    format(atom(Prop), 'padding: ~w', [Px]).
rn_option_to_prop(margin(V), Prop) :-
    spacing_value(V, Px),
    format(atom(Prop), 'margin: ~w', [Px]).
rn_option_to_prop(flex(V), Prop) :-
    format(atom(Prop), 'flex: ~w', [V]).
rn_option_to_prop(width(V), Prop) :-
    rn_size_value(V, RNVal),
    format(atom(Prop), 'width: ~w', [RNVal]).
rn_option_to_prop(height(V), Prop) :-
    rn_size_value(V, RNVal),
    format(atom(Prop), 'height: ~w', [RNVal]).

rn_justify_value(start, 'flex-start').
rn_justify_value(end, 'flex-end').
rn_justify_value(center, center).
rn_justify_value(space_between, 'space-between').
rn_justify_value(space_around, 'space-around').
rn_justify_value(space_evenly, 'space-evenly').

rn_align_value(start, 'flex-start').
rn_align_value(end, 'flex-end').
rn_align_value(center, center).
rn_align_value(stretch, stretch).
rn_align_value(baseline, baseline).

rn_size_value(full, '\'100%\'').
rn_size_value(auto, '\'auto\'').
rn_size_value(V, V) :- number(V).
rn_size_value(V, Str) :- atom(V), format(atom(Str), '\'~w\'', [V]).

rn_children_code(Children, Code) :-
    findall(C, (
        member(Child, Children),
        format(atom(C), '  {/* ~w */}\n', [Child])
    ), Codes),
    atomic_list_concat(Codes, '', Code).

% ============================================================================
% Vue Layout Generation
% ============================================================================

%! generate_vue_layout(+Spec, -Code) is det
generate_vue_layout(layout_spec(Type, Options, Children), Code) :-
    vue_layout_classes(Type, Options, Classes),
    vue_children_code(Children, ChildrenCode),
    format(atom(Code), '<div class="~w">\n~w</div>', [Classes, ChildrenCode]).

vue_layout_classes(Type, Options, Classes) :-
    vue_base_class(Type, BaseClass),
    findall(Class, (
        member(Opt, Options),
        vue_option_class(Opt, Class)
    ), OptClasses),
    atomic_list_concat([BaseClass|OptClasses], ' ', Classes).

vue_base_class(row, 'flex flex-row').
vue_base_class(column, 'flex flex-col').
vue_base_class(stack, 'relative').
vue_base_class(wrap, 'flex flex-row flex-wrap').
vue_base_class(center, 'flex justify-center items-center').
vue_base_class(container, 'container mx-auto').
vue_base_class(scroll_view, 'overflow-auto').
vue_base_class(safe_area, 'safe-area-inset').
vue_base_class(grid, 'grid').

vue_option_class(justify(V), Class) :-
    vue_justify_class(V, Class).
vue_option_class(align(V), Class) :-
    vue_align_class(V, Class).
vue_option_class(gap(V), Class) :-
    vue_gap_class(V, Class).
vue_option_class(padding(V), Class) :-
    vue_padding_class(V, Class).
vue_option_class(margin(V), Class) :-
    vue_margin_class(V, Class).
vue_option_class(flex(1), 'flex-1').
vue_option_class(flex(auto), 'flex-auto').
vue_option_class(width(full), 'w-full').
vue_option_class(height(full), 'h-full').
vue_option_class(height(screen), 'h-screen').

vue_justify_class(start, 'justify-start').
vue_justify_class(end, 'justify-end').
vue_justify_class(center, 'justify-center').
vue_justify_class(space_between, 'justify-between').
vue_justify_class(space_around, 'justify-around').
vue_justify_class(space_evenly, 'justify-evenly').

vue_align_class(start, 'items-start').
vue_align_class(end, 'items-end').
vue_align_class(center, 'items-center').
vue_align_class(stretch, 'items-stretch').
vue_align_class(baseline, 'items-baseline').

vue_gap_class(xs, 'gap-1').
vue_gap_class(sm, 'gap-2').
vue_gap_class(md, 'gap-4').
vue_gap_class(lg, 'gap-6').
vue_gap_class(xl, 'gap-8').

vue_padding_class(xs, 'p-1').
vue_padding_class(sm, 'p-2').
vue_padding_class(md, 'p-4').
vue_padding_class(lg, 'p-6').
vue_padding_class(xl, 'p-8').

vue_margin_class(xs, 'm-1').
vue_margin_class(sm, 'm-2').
vue_margin_class(md, 'm-4').
vue_margin_class(lg, 'm-6').
vue_margin_class(xl, 'm-8').

vue_children_code(Children, Code) :-
    findall(C, (
        member(Child, Children),
        format(atom(C), '  <!-- ~w -->\n', [Child])
    ), Codes),
    atomic_list_concat(Codes, '', Code).

% ============================================================================
% Flutter Layout Generation
% ============================================================================

%! generate_flutter_layout(+Spec, -Code) is det
generate_flutter_layout(layout_spec(Type, Options, Children), Code) :-
    flutter_layout_widget(Type, Options, Children, Code).

flutter_layout_widget(row, Options, Children, Code) :-
    flutter_main_axis(Options, MainAxis),
    flutter_cross_axis(Options, CrossAxis),
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'Row(\n  mainAxisAlignment: ~w,\n  crossAxisAlignment: ~w,\n  children: [~w],\n)',
           [MainAxis, CrossAxis, ChildrenCode]).

flutter_layout_widget(column, Options, Children, Code) :-
    flutter_main_axis(Options, MainAxis),
    flutter_cross_axis(Options, CrossAxis),
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'Column(\n  mainAxisAlignment: ~w,\n  crossAxisAlignment: ~w,\n  children: [~w],\n)',
           [MainAxis, CrossAxis, ChildrenCode]).

flutter_layout_widget(stack, _Options, Children, Code) :-
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'Stack(\n  children: [~w],\n)', [ChildrenCode]).

flutter_layout_widget(wrap, Options, Children, Code) :-
    flutter_spacing(Options, Spacing),
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'Wrap(\n  spacing: ~w,\n  runSpacing: ~w,\n  children: [~w],\n)',
           [Spacing, Spacing, ChildrenCode]).

flutter_layout_widget(center, _, Children, Code) :-
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'Center(\n  child: ~w,\n)', [ChildrenCode]).

flutter_layout_widget(container, Options, Children, Code) :-
    flutter_padding(Options, Padding),
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'Container(\n  padding: ~w,\n  child: ~w,\n)', [Padding, ChildrenCode]).

flutter_layout_widget(scroll_view, Options, Children, Code) :-
    get_option(direction, Options, Dir, vertical),
    flutter_scroll_direction(Dir, ScrollDir),
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'SingleChildScrollView(\n  scrollDirection: ~w,\n  child: ~w,\n)',
           [ScrollDir, ChildrenCode]).

flutter_layout_widget(safe_area, _, Children, Code) :-
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'SafeArea(\n  child: ~w,\n)', [ChildrenCode]).

flutter_layout_widget(grid, Options, Children, Code) :-
    get_option(columns, Options, Cols, 2),
    flutter_children(Children, ChildrenCode),
    format(atom(Code), 'GridView.count(\n  crossAxisCount: ~w,\n  children: [~w],\n)',
           [Cols, ChildrenCode]).

flutter_main_axis(Options, Axis) :-
    (   member(justify(V), Options)
    ->  flutter_main_axis_value(V, Axis)
    ;   Axis = 'MainAxisAlignment.start'
    ).

flutter_main_axis_value(start, 'MainAxisAlignment.start').
flutter_main_axis_value(end, 'MainAxisAlignment.end').
flutter_main_axis_value(center, 'MainAxisAlignment.center').
flutter_main_axis_value(space_between, 'MainAxisAlignment.spaceBetween').
flutter_main_axis_value(space_around, 'MainAxisAlignment.spaceAround').
flutter_main_axis_value(space_evenly, 'MainAxisAlignment.spaceEvenly').

flutter_cross_axis(Options, Axis) :-
    (   member(align(V), Options)
    ->  flutter_cross_axis_value(V, Axis)
    ;   Axis = 'CrossAxisAlignment.start'
    ).

flutter_cross_axis_value(start, 'CrossAxisAlignment.start').
flutter_cross_axis_value(end, 'CrossAxisAlignment.end').
flutter_cross_axis_value(center, 'CrossAxisAlignment.center').
flutter_cross_axis_value(stretch, 'CrossAxisAlignment.stretch').
flutter_cross_axis_value(baseline, 'CrossAxisAlignment.baseline').

flutter_spacing(Options, Spacing) :-
    (   member(gap(V), Options)
    ->  spacing_value(V, Spacing)
    ;   Spacing = 0
    ).

flutter_padding(Options, Padding) :-
    (   member(padding(V), Options)
    ->  spacing_value(V, Px),
        format(atom(Padding), 'EdgeInsets.all(~w)', [Px])
    ;   Padding = 'EdgeInsets.zero'
    ).

flutter_scroll_direction(vertical, 'Axis.vertical').
flutter_scroll_direction(horizontal, 'Axis.horizontal').

flutter_children([], '').
flutter_children([Child], Code) :-
    format(atom(Code), '/* ~w */', [Child]).
flutter_children([_|_], '/* children */').

% ============================================================================
% SwiftUI Layout Generation
% ============================================================================

%! generate_swiftui_layout(+Spec, -Code) is det
generate_swiftui_layout(layout_spec(Type, Options, Children), Code) :-
    swift_layout_view(Type, Options, Children, Code).

swift_layout_view(row, Options, Children, Code) :-
    swift_alignment(Options, Alignment),
    swift_spacing(Options, Spacing),
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'HStack(alignment: ~w, spacing: ~w) {\n~w}',
           [Alignment, Spacing, ChildrenCode]).

swift_layout_view(column, Options, Children, Code) :-
    swift_alignment(Options, Alignment),
    swift_spacing(Options, Spacing),
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'VStack(alignment: ~w, spacing: ~w) {\n~w}',
           [Alignment, Spacing, ChildrenCode]).

swift_layout_view(stack, _Options, Children, Code) :-
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'ZStack {\n~w}', [ChildrenCode]).

swift_layout_view(wrap, _Options, Children, Code) :-
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))]) {\n~w}', [ChildrenCode]).

swift_layout_view(center, _, Children, Code) :-
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'VStack {\n  Spacer()\n  HStack {\n    Spacer()\n~w    Spacer()\n  }\n  Spacer()\n}', [ChildrenCode]).

swift_layout_view(container, Options, Children, Code) :-
    swift_padding(Options, Padding),
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'VStack {\n~w}\n~w\n.frame(maxWidth: .infinity)', [ChildrenCode, Padding]).

swift_layout_view(scroll_view, Options, Children, Code) :-
    get_option(direction, Options, Dir, vertical),
    swift_scroll_axes(Dir, Axes),
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'ScrollView(~w) {\n~w}', [Axes, ChildrenCode]).

swift_layout_view(safe_area, _, Children, Code) :-
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'VStack {\n~w}\n.safeAreaInset(edges: .all)', [ChildrenCode]).

swift_layout_view(grid, Options, Children, Code) :-
    get_option(columns, Options, Cols, 2),
    swift_children(Children, ChildrenCode),
    format(atom(Code), 'LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: ~w)) {\n~w}',
           [Cols, ChildrenCode]).

swift_alignment(Options, Alignment) :-
    (   member(align(V), Options)
    ->  swift_align_value(V, Alignment)
    ;   Alignment = '.center'
    ).

swift_align_value(start, '.leading').
swift_align_value(end, '.trailing').
swift_align_value(center, '.center').
swift_align_value(stretch, '.center').

swift_spacing(Options, Spacing) :-
    (   member(gap(V), Options)
    ->  spacing_value(V, Spacing)
    ;   Spacing = 'nil'
    ).

swift_padding(Options, Padding) :-
    (   member(padding(V), Options)
    ->  spacing_value(V, Px),
        format(atom(Padding), '.padding(~w)', [Px])
    ;   Padding = ''
    ).

swift_scroll_axes(vertical, '.vertical').
swift_scroll_axes(horizontal, '.horizontal').

swift_children(Children, Code) :-
    findall(C, (
        member(Child, Children),
        format(atom(C), '  // ~w\n', [Child])
    ), Codes),
    atomic_list_concat(Codes, '', Code).

% ============================================================================
% Testing
% ============================================================================

%! test_layout is det
%  Run inline tests.
test_layout :-
    format('Running layout tests...~n'),

    % Test 1: Row creation
    row([justify(space_between), align(center)], [child1, child2], RowSpec),
    RowSpec = layout_spec(row, _, _),
    format('  Test 1 passed: row creation~n'),

    % Test 2: Column creation
    column([gap(md), padding(lg)], [child1], ColSpec),
    ColSpec = layout_spec(column, _, _),
    format('  Test 2 passed: column creation~n'),

    % Test 3: Center creation
    center(content, CenterSpec),
    CenterSpec = layout_spec(center, [], [content]),
    format('  Test 3 passed: center creation~n'),

    % Test 4: React Native row
    generate_layout(RowSpec, react_native, RNCode),
    sub_string(RNCode, _, _, _, "flexDirection: 'row'"),
    sub_string(RNCode, _, _, _, "space-between"),
    format('  Test 4 passed: React Native row~n'),

    % Test 5: Vue row
    generate_layout(RowSpec, vue, VueCode),
    sub_string(VueCode, _, _, _, "flex flex-row"),
    sub_string(VueCode, _, _, _, "justify-between"),
    format('  Test 5 passed: Vue row~n'),

    % Test 6: Flutter row
    generate_layout(RowSpec, flutter, FlutterCode),
    sub_string(FlutterCode, _, _, _, "Row("),
    sub_string(FlutterCode, _, _, _, "MainAxisAlignment.spaceBetween"),
    format('  Test 6 passed: Flutter row~n'),

    % Test 7: SwiftUI row
    generate_layout(RowSpec, swiftui, SwiftCode),
    sub_string(SwiftCode, _, _, _, "HStack"),
    format('  Test 7 passed: SwiftUI row~n'),

    % Test 8: Column with gap
    column([gap(md)], [a, b], ColSpec2),
    generate_layout(ColSpec2, react_native, RNCol),
    sub_string(RNCol, _, _, _, "gap: 16"),
    format('  Test 8 passed: column with gap~n'),

    % Test 9: Stack
    stack([], [layer1, layer2], StackSpec),
    generate_layout(StackSpec, swiftui, SwiftStack),
    sub_string(SwiftStack, _, _, _, "ZStack"),
    format('  Test 9 passed: stack~n'),

    % Test 10: Wrap
    wrap([gap(sm)], [item1, item2, item3], WrapSpec),
    generate_layout(WrapSpec, flutter, FlutterWrap),
    sub_string(FlutterWrap, _, _, _, "Wrap("),
    format('  Test 10 passed: wrap~n'),

    format('All 10 layout tests passed!~n'),
    !.

:- initialization(test_layout, main).
