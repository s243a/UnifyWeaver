% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_component_library.pl - plunit tests for component_library module
%
% Run with: swipl -g "run_tests" -t halt test_component_library.pl

:- module(test_component_library, []).

:- use_module(library(plunit)).
:- use_module('component_library').

% ============================================================================
% Tests: Modal Components
% ============================================================================

:- begin_tests(modal_components).

test(modal_alert) :-
    modal(alert, [title('Alert'), message('Test message')], Spec),
    Spec = modal_spec(alert, Props),
    member(title('Alert'), Props).

test(modal_confirm) :-
    modal(confirm, [title('Confirm'), message('Are you sure?')], Spec),
    Spec = modal_spec(confirm, _).

test(modal_with_dismissable) :-
    modal(custom, [dismissable(false)], Spec),
    Spec = modal_spec(custom, Props),
    member(dismissable(false), Props).

test(alert_dialog) :-
    alert_dialog('Warning', [message('Something happened')], Spec),
    Spec = alert_dialog_spec(Props),
    member(title('Warning'), Props).

test(bottom_sheet) :-
    bottom_sheet(content, [height(300)], Spec),
    Spec = bottom_sheet_spec(Props),
    member(height(300), Props).

test(action_sheet) :-
    action_sheet([action1, action2], [title('Choose')], Spec),
    Spec = action_sheet_spec(Props),
    member(title('Choose'), Props),
    member(actions([action1, action2]), Props).

:- end_tests(modal_components).

% ============================================================================
% Tests: Feedback Components
% ============================================================================

:- begin_tests(feedback_components).

test(toast_basic) :-
    toast('Hello', [], Spec),
    Spec = toast_spec(Props),
    member(message('Hello'), Props).

test(toast_with_type) :-
    toast('Success!', [type(success)], Spec),
    Spec = toast_spec(Props),
    member(type(success), Props).

test(toast_with_position) :-
    toast('Info', [position(top)], Spec),
    Spec = toast_spec(Props),
    member(position(top), Props).

test(snackbar_basic) :-
    snackbar('Saved', [], Spec),
    Spec = snackbar_spec(Props),
    member(message('Saved'), Props).

test(snackbar_with_action) :-
    snackbar('Deleted', [action(undo), actionText('Undo')], Spec),
    Spec = snackbar_spec(Props),
    member(actionText('Undo'), Props).

test(banner_basic) :-
    banner('Update available', [type(info)], Spec),
    Spec = banner_spec(Props),
    member(type(info), Props).

:- end_tests(feedback_components).

% ============================================================================
% Tests: Content Components
% ============================================================================

:- begin_tests(content_components).

test(card_basic) :-
    card(content, [title('Card Title')], Spec),
    Spec = card_spec(Props),
    member(title('Card Title'), Props).

test(card_with_image) :-
    card(content, [image('img.jpg'), elevated(true)], Spec),
    Spec = card_spec(Props),
    member(image('img.jpg'), Props),
    member(elevated(true), Props).

test(list_item_basic) :-
    list_item('Item text', [], Spec),
    Spec = list_item_spec(Props),
    member(content('Item text'), Props).

test(list_item_with_leading) :-
    list_item('Item', [leading(icon), trailing(chevron)], Spec),
    Spec = list_item_spec(Props),
    member(leading(icon), Props).

test(avatar_basic) :-
    avatar('https://example.com/pic.jpg', [], Spec),
    Spec = avatar_spec(Props),
    member(source('https://example.com/pic.jpg'), Props).

test(avatar_with_size) :-
    avatar('pic.jpg', [size(large), shape(square)], Spec),
    Spec = avatar_spec(Props),
    member(size(large), Props),
    member(shape(square), Props).

test(badge_basic) :-
    badge('3', [], Spec),
    Spec = badge_spec(Props),
    member(content('3'), Props).

test(badge_with_color) :-
    badge('New', [color(success), variant(outlined)], Spec),
    Spec = badge_spec(Props),
    member(color(success), Props).

test(chip_basic) :-
    chip('Tag', [], Spec),
    Spec = chip_spec(Props),
    member(label('Tag'), Props).

test(chip_with_delete) :-
    chip('Removable', [onDelete(handler)], Spec),
    Spec = chip_spec(Props),
    member(onDelete(handler), Props).

test(tag_is_chip) :-
    tag('Label', [], Spec),
    Spec = chip_spec(_).

:- end_tests(content_components).

% ============================================================================
% Tests: Layout Components
% ============================================================================

:- begin_tests(layout_components).

test(divider_basic) :-
    divider([], Spec),
    Spec = divider_spec(Props),
    member(orientation(horizontal), Props).

test(divider_vertical) :-
    divider([orientation(vertical)], Spec),
    Spec = divider_spec(Props),
    member(orientation(vertical), Props).

test(spacer_basic) :-
    spacer([], Spec),
    Spec = spacer_spec(Props),
    member(size(medium), Props).

test(spacer_with_flex) :-
    spacer([flex(true)], Spec),
    Spec = spacer_spec(Props),
    member(flex(true), Props).

test(skeleton_text) :-
    skeleton(text, [width('80%')], Spec),
    Spec = skeleton_spec(Props),
    member(type(text), Props).

test(skeleton_circle) :-
    skeleton(circle, [width(48), height(48)], Spec),
    Spec = skeleton_spec(Props),
    member(type(circle), Props).

:- end_tests(layout_components).

% ============================================================================
% Tests: Progress Components
% ============================================================================

:- begin_tests(progress_components).

test(progress_bar_basic) :-
    progress_bar(50, [], Spec),
    Spec = progress_bar_spec(Props),
    member(value(50), Props).

test(progress_bar_with_max) :-
    progress_bar(75, [max(150)], Spec),
    Spec = progress_bar_spec(Props),
    member(max(150), Props).

test(progress_circle_basic) :-
    progress_circle(30, [], Spec),
    Spec = progress_circle_spec(Props),
    member(value(30), Props).

test(progress_circle_with_size) :-
    progress_circle(60, [size(64), strokeWidth(6)], Spec),
    Spec = progress_circle_spec(Props),
    member(size(64), Props).

test(spinner_basic) :-
    spinner([], Spec),
    Spec = spinner_spec(Props),
    member(size(medium), Props).

test(spinner_large) :-
    spinner([size(large), color(secondary)], Spec),
    Spec = spinner_spec(Props),
    member(size(large), Props).

:- end_tests(progress_components).

% ============================================================================
% Tests: Input Components
% ============================================================================

:- begin_tests(input_components).

test(search_bar_basic) :-
    search_bar([], Spec),
    Spec = search_bar_spec(Props),
    member(placeholder('Search...'), Props).

test(search_bar_custom) :-
    search_bar([placeholder('Find items'), showCancel(true)], Spec),
    Spec = search_bar_spec(Props),
    member(placeholder('Find items'), Props).

test(rating_basic) :-
    rating(3, [], Spec),
    Spec = rating_spec(Props),
    member(value(3), Props).

test(rating_half_stars) :-
    rating(3.5, [allowHalf(true), max(5)], Spec),
    Spec = rating_spec(Props),
    member(allowHalf(true), Props).

test(stepper_basic) :-
    stepper(1, [], Spec),
    Spec = stepper_spec(Props),
    member(value(1), Props).

test(stepper_with_range) :-
    stepper(5, [min(0), max(10), step(1)], Spec),
    Spec = stepper_spec(Props),
    member(min(0), Props),
    member(max(10), Props).

test(slider_input_basic) :-
    slider_input(50, [], Spec),
    Spec = slider_input_spec(Props),
    member(value(50), Props).

:- end_tests(input_components).

% ============================================================================
% Tests: React Native Generation
% ============================================================================

:- begin_tests(react_native_generation).

test(rn_modal) :-
    modal(alert, [title('Test'), message('Hello')], Spec),
    generate_component(Spec, react_native, Code),
    sub_string(Code, _, _, _, "Modal").

test(rn_toast) :-
    toast('Message', [type(info)], Spec),
    generate_component(Spec, react_native, Code),
    sub_string(Code, _, _, _, "Toast").

test(rn_card) :-
    card(content, [title('Title')], Spec),
    generate_component(Spec, react_native, Code),
    sub_string(Code, _, _, _, "card").

test(rn_avatar) :-
    avatar('pic.jpg', [size(medium)], Spec),
    generate_component(Spec, react_native, Code),
    sub_string(Code, _, _, _, "Image").

test(rn_spinner) :-
    spinner([size(large)], Spec),
    generate_component(Spec, react_native, Code),
    sub_string(Code, _, _, _, "ActivityIndicator").

:- end_tests(react_native_generation).

% ============================================================================
% Tests: Vue Generation
% ============================================================================

:- begin_tests(vue_generation).

test(vue_modal) :-
    modal(alert, [title('Test'), message('Hello')], Spec),
    generate_component(Spec, vue, Code),
    sub_string(Code, _, _, _, "template").

test(vue_card) :-
    card(content, [title('Title')], Spec),
    generate_component(Spec, vue, Code),
    sub_string(Code, _, _, _, "card").

test(vue_avatar) :-
    avatar('pic.jpg', [size(medium)], Spec),
    generate_component(Spec, vue, Code),
    sub_string(Code, _, _, _, "avatar").

test(vue_search) :-
    search_bar([placeholder('Search')], Spec),
    generate_component(Spec, vue, Code),
    sub_string(Code, _, _, _, "search-bar").

:- end_tests(vue_generation).

% ============================================================================
% Tests: Flutter Generation
% ============================================================================

:- begin_tests(flutter_generation).

test(flutter_modal) :-
    modal(alert, [title('Test'), message('Hello')], Spec),
    generate_component(Spec, flutter, Code),
    sub_string(Code, _, _, _, "showDialog").

test(flutter_toast) :-
    toast('Message', [], Spec),
    generate_component(Spec, flutter, Code),
    sub_string(Code, _, _, _, "SnackBar").

test(flutter_avatar) :-
    avatar('pic.jpg', [size(large)], Spec),
    generate_component(Spec, flutter, Code),
    sub_string(Code, _, _, _, "CircleAvatar").

test(flutter_progress) :-
    progress_bar(50, [max(100)], Spec),
    generate_component(Spec, flutter, Code),
    sub_string(Code, _, _, _, "LinearProgressIndicator").

:- end_tests(flutter_generation).

% ============================================================================
% Tests: SwiftUI Generation
% ============================================================================

:- begin_tests(swiftui_generation).

test(swift_modal) :-
    modal(alert, [title('Test'), message('Hello')], Spec),
    generate_component(Spec, swiftui, Code),
    sub_string(Code, _, _, _, ".alert").

test(swift_card) :-
    card(content, [title('Title')], Spec),
    generate_component(Spec, swiftui, Code),
    sub_string(Code, _, _, _, "VStack").

test(swift_avatar) :-
    avatar('pic.jpg', [size(medium)], Spec),
    generate_component(Spec, swiftui, Code),
    sub_string(Code, _, _, _, "AsyncImage").

test(swift_badge) :-
    badge('5', [color(primary)], Spec),
    generate_component(Spec, swiftui, Code),
    sub_string(Code, _, _, _, "Text").

test(swift_spinner) :-
    spinner([], Spec),
    generate_component(Spec, swiftui, Code),
    sub_string(Code, _, _, _, "ProgressView").

:- end_tests(swiftui_generation).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
