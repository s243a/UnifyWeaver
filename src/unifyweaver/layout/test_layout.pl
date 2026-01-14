% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_layout.pl - plunit tests for layout system
%
% Run with: swipl -g "run_tests" -t halt test_layout.pl

:- module(test_layout, []).

:- use_module(library(plunit)).
:- use_module('layout').

% ============================================================================
% Tests: Layout Primitives
% ============================================================================

:- begin_tests(layout_primitives).

test(row_creates_spec) :-
    layout:row([justify(center)], [child1, child2], Spec),
    Spec = layout_spec(row, [justify(center)], [child1, child2]).

test(column_creates_spec) :-
    layout:column([align(stretch)], [child1], Spec),
    Spec = layout_spec(column, [align(stretch)], [child1]).

test(stack_creates_spec) :-
    layout:stack([], [layer1, layer2], Spec),
    Spec = layout_spec(stack, [], [layer1, layer2]).

test(grid_creates_spec) :-
    layout:grid([columns(3), gap(16)], [item1, item2, item3], Spec),
    Spec = layout_spec(grid, [columns(3), gap(16)], [item1, item2, item3]).

test(wrap_creates_spec) :-
    layout:wrap([gap(8)], [tag1, tag2, tag3], Spec),
    Spec = layout_spec(wrap, [gap(8)], [tag1, tag2, tag3]).

:- end_tests(layout_primitives).

% ============================================================================
% Tests: Container Layouts
% ============================================================================

:- begin_tests(container_layouts).

test(center_creates_spec) :-
    layout:center(child, Spec),
    Spec = layout_spec(center, [], [child]).

test(container_creates_spec) :-
    layout:container([padding(16), max_width(1200)], [content], Spec),
    Spec = layout_spec(container, [padding(16), max_width(1200)], [content]).

test(scroll_view_creates_spec) :-
    layout:scroll_view([direction(horizontal)], [content], Spec),
    Spec = layout_spec(scroll_view, [direction(horizontal)|_], [content]).

test(safe_area_creates_spec) :-
    layout:safe_area(content, Spec),
    Spec = layout_spec(safe_area, [], [content]).

:- end_tests(container_layouts).

% ============================================================================
% Tests: Flex Properties
% ============================================================================

:- begin_tests(flex_properties).

test(flex_value) :-
    layout:flex(1, Spec),
    Spec = flex(1).

test(flex_grow_value) :-
    layout:flex_grow(2, Spec),
    Spec = flex_grow(2).

test(flex_shrink_value) :-
    layout:flex_shrink(0, Spec),
    Spec = flex_shrink(0).

test(flex_basis_value) :-
    layout:flex_basis(200, Spec),
    Spec = flex_basis(200).

test(flex_basis_auto) :-
    layout:flex_basis(auto, Spec),
    Spec = flex_basis(auto).

:- end_tests(flex_properties).

% ============================================================================
% Tests: Alignment
% ============================================================================

:- begin_tests(alignment).

test(justify_center) :-
    layout:justify(center, Spec),
    Spec = justify(center).

test(justify_space_between) :-
    layout:justify(space_between, Spec),
    Spec = justify(space_between).

test(align_stretch) :-
    layout:align(stretch, Spec),
    Spec = align(stretch).

test(align_self_flex_end) :-
    layout:align_self(flex_end, Spec),
    Spec = align_self(flex_end).

:- end_tests(alignment).

% ============================================================================
% Tests: Spacing
% ============================================================================

:- begin_tests(spacing).

test(gap_numeric) :-
    layout:gap(16, Spec),
    Spec = gap(16).

test(gap_scale) :-
    layout:gap(md, Spec),
    Spec = gap(md).

test(padding_numeric) :-
    layout:padding(8, Spec),
    Spec = padding(8).

test(margin_numeric) :-
    layout:margin(12, Spec),
    Spec = margin(12).

:- end_tests(spacing).

% ============================================================================
% Tests: Sizing
% ============================================================================

:- begin_tests(sizing).

test(width_numeric) :-
    layout:width(200, Spec),
    Spec = width(200).

test(width_full) :-
    layout:width(full, Spec),
    Spec = width(full).

test(width_percent) :-
    layout:width(percent(50), Spec),
    Spec = width(percent(50)).

test(height_numeric) :-
    layout:height(100, Spec),
    Spec = height(100).

test(min_width_value) :-
    layout:min_width(300, Spec),
    Spec = min_width(300).

test(max_width_value) :-
    layout:max_width(800, Spec),
    Spec = max_width(800).

test(min_height_value) :-
    layout:min_height(200, Spec),
    Spec = min_height(200).

test(max_height_value) :-
    layout:max_height(600, Spec),
    Spec = max_height(600).

test(aspect_ratio_value) :-
    layout:aspect_ratio(16/9, Spec),
    Spec = aspect_ratio(16/9).

:- end_tests(sizing).

% ============================================================================
% Tests: React Native Code Generation
% ============================================================================

:- begin_tests(react_native_generation).

test(row_generates_view) :-
    layout:row([justify(center)], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "flexDirection: 'row'").

test(row_generates_justify_content) :-
    layout:row([justify(center)], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "justifyContent: 'center'").

test(column_generates_column_direction) :-
    layout:column([], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "flexDirection: 'column'").

test(stack_generates_relative_position) :-
    layout:stack([], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "position: 'relative'").

test(gap_generates_gap_property) :-
    layout:row([gap(16)], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "gap: 16").

test(padding_generates_padding_property) :-
    layout:row([padding(8)], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "padding: 8").

test(flex_generates_flex_property) :-
    layout:row([flex(1)], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "flex: 1").

test(space_between_justify) :-
    layout:row([justify(space_between)], [], Spec),
    layout:generate_react_native_layout(Spec, Code),
    sub_string(Code, _, _, _, "justifyContent: 'space-between'").

:- end_tests(react_native_generation).

% ============================================================================
% Tests: Vue Code Generation
% ============================================================================

:- begin_tests(vue_generation).

test(row_generates_flex_row) :-
    layout:row([], [], Spec),
    layout:generate_vue_layout(Spec, Code),
    sub_string(Code, _, _, _, "flex flex-row").

test(column_generates_flex_col) :-
    layout:column([], [], Spec),
    layout:generate_vue_layout(Spec, Code),
    sub_string(Code, _, _, _, "flex flex-col").

test(justify_center_class) :-
    layout:row([justify(center)], [], Spec),
    layout:generate_vue_layout(Spec, Code),
    sub_string(Code, _, _, _, "justify-center").

test(justify_between_class) :-
    layout:row([justify(space_between)], [], Spec),
    layout:generate_vue_layout(Spec, Code),
    sub_string(Code, _, _, _, "justify-between").

test(align_center_class) :-
    layout:row([align(center)], [], Spec),
    layout:generate_vue_layout(Spec, Code),
    sub_string(Code, _, _, _, "items-center").

test(gap_class) :-
    layout:row([gap(md)], [], Spec),
    layout:generate_vue_layout(Spec, Code),
    sub_string(Code, _, _, _, "gap-4").

test(padding_class) :-
    layout:row([padding(sm)], [], Spec),
    layout:generate_vue_layout(Spec, Code),
    sub_string(Code, _, _, _, "p-2").

:- end_tests(vue_generation).

% ============================================================================
% Tests: Flutter Code Generation
% ============================================================================

:- begin_tests(flutter_generation).

test(row_generates_row_widget) :-
    layout:row([], [], Spec),
    layout:generate_flutter_layout(Spec, Code),
    sub_string(Code, _, _, _, "Row(").

test(column_generates_column_widget) :-
    layout:column([], [], Spec),
    layout:generate_flutter_layout(Spec, Code),
    sub_string(Code, _, _, _, "Column(").

test(stack_generates_stack_widget) :-
    layout:stack([], [], Spec),
    layout:generate_flutter_layout(Spec, Code),
    sub_string(Code, _, _, _, "Stack(").

test(wrap_generates_wrap_widget) :-
    layout:wrap([], [], Spec),
    layout:generate_flutter_layout(Spec, Code),
    sub_string(Code, _, _, _, "Wrap(").

test(center_generates_center_widget) :-
    layout:center(child, Spec),
    layout:generate_flutter_layout(Spec, Code),
    sub_string(Code, _, _, _, "Center(").

test(justify_center_main_axis) :-
    layout:row([justify(center)], [], Spec),
    layout:generate_flutter_layout(Spec, Code),
    sub_string(Code, _, _, _, "MainAxisAlignment.center").

test(align_center_cross_axis) :-
    layout:row([align(center)], [], Spec),
    layout:generate_flutter_layout(Spec, Code),
    sub_string(Code, _, _, _, "CrossAxisAlignment.center").

:- end_tests(flutter_generation).

% ============================================================================
% Tests: SwiftUI Code Generation
% ============================================================================

:- begin_tests(swiftui_generation).

test(row_generates_hstack) :-
    layout:row([], [], Spec),
    layout:generate_swiftui_layout(Spec, Code),
    sub_string(Code, _, _, _, "HStack").

test(column_generates_vstack) :-
    layout:column([], [], Spec),
    layout:generate_swiftui_layout(Spec, Code),
    sub_string(Code, _, _, _, "VStack").

test(stack_generates_zstack) :-
    layout:stack([], [], Spec),
    layout:generate_swiftui_layout(Spec, Code),
    sub_string(Code, _, _, _, "ZStack").

test(wrap_generates_lazyvgrid) :-
    layout:wrap([], [], Spec),
    layout:generate_swiftui_layout(Spec, Code),
    sub_string(Code, _, _, _, "LazyVGrid").

test(center_generates_spacers) :-
    layout:center(child, Spec),
    layout:generate_swiftui_layout(Spec, Code),
    sub_string(Code, _, _, _, "Spacer").

test(row_alignment_parameter) :-
    layout:row([align(center)], [], Spec),
    layout:generate_swiftui_layout(Spec, Code),
    sub_string(Code, _, _, _, "alignment:").

test(row_spacing_parameter) :-
    layout:row([gap(16)], [], Spec),
    layout:generate_swiftui_layout(Spec, Code),
    sub_string(Code, _, _, _, "spacing:").

:- end_tests(swiftui_generation).

% ============================================================================
% Tests: generate_layout dispatcher
% ============================================================================

:- begin_tests(generate_layout_dispatcher).

test(dispatch_to_react_native) :-
    layout:row([], [], Spec),
    layout:generate_layout(Spec, react_native, Code),
    sub_string(Code, _, _, _, "View").

test(dispatch_to_vue) :-
    layout:row([], [], Spec),
    layout:generate_layout(Spec, vue, Code),
    sub_string(Code, _, _, _, "flex").

test(dispatch_to_flutter) :-
    layout:row([], [], Spec),
    layout:generate_layout(Spec, flutter, Code),
    sub_string(Code, _, _, _, "Row").

test(dispatch_to_swiftui) :-
    layout:row([], [], Spec),
    layout:generate_layout(Spec, swiftui, Code),
    sub_string(Code, _, _, _, "HStack").

:- end_tests(generate_layout_dispatcher).

% ============================================================================
% Tests: Responsive Breakpoints
% ============================================================================

:- begin_tests(responsive_breakpoints).

test(breakpoint_sm) :-
    layout:breakpoint(sm, 640, Term),
    Term = breakpoint(sm, 640).

test(breakpoint_md) :-
    layout:breakpoint(md, 768, Term),
    Term = breakpoint(md, 768).

test(breakpoint_lg) :-
    layout:breakpoint(lg, 1024, Term),
    Term = breakpoint(lg, 1024).

test(breakpoint_xl) :-
    layout:breakpoint(xl, 1280, Term),
    Term = breakpoint(xl, 1280).

:- end_tests(responsive_breakpoints).

% ============================================================================
% Tests: Spacing with Named Scales
% ============================================================================

:- begin_tests(spacing_named_scales).

test(gap_with_xs) :-
    layout:gap(xs, Term),
    Term = gap(xs).

test(gap_with_sm) :-
    layout:gap(sm, Term),
    Term = gap(sm).

test(gap_with_md) :-
    layout:gap(md, Term),
    Term = gap(md).

test(gap_with_lg) :-
    layout:gap(lg, Term),
    Term = gap(lg).

test(gap_with_xl) :-
    layout:gap(xl, Term),
    Term = gap(xl).

:- end_tests(spacing_named_scales).

% ============================================================================
% Tests: Complex Layouts
% ============================================================================

:- begin_tests(complex_layouts).

test(nested_row_in_column) :-
    layout:row([justify(center)], [item], RowSpec),
    layout:column([], [RowSpec], ColSpec),
    ColSpec = layout_spec(column, [], [layout_spec(row, [justify(center)], [item])]).

test(row_with_multiple_options) :-
    layout:row([justify(space_between), align(center), gap(16), padding(8)], [a, b, c], Spec),
    Spec = layout_spec(row, [justify(space_between), align(center), gap(16), padding(8)], [a, b, c]).

test(container_with_sizing) :-
    layout:container([max_width(1200), padding(16)], [content], Spec),
    Spec = layout_spec(container, [max_width(1200), padding(16)], [content]).

test(grid_with_columns) :-
    layout:grid([columns(4), gap(8)], [i1, i2, i3, i4], Spec),
    Spec = layout_spec(grid, [columns(4), gap(8)], [i1, i2, i3, i4]).

:- end_tests(complex_layouts).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
