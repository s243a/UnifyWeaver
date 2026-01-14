% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_accessibility.pl - plunit tests for accessibility module
%
% Run with: swipl -g "run_tests" -t halt test_accessibility.pl

:- module(test_accessibility, []).

:- use_module(library(plunit)).
:- use_module('accessibility').

% ============================================================================
% Tests: Role Mapping
% ============================================================================

:- begin_tests(role_mapping).

test(button_role_all_targets) :-
    map_a11y_role(button, react_native, button),
    map_a11y_role(button, vue, button),
    map_a11y_role(button, flutter, button),
    map_a11y_role(button, swiftui, button).

test(heading_role_variations) :-
    map_a11y_role(heading, react_native, header),
    map_a11y_role(heading, vue, heading),
    map_a11y_role(heading, flutter, header),
    map_a11y_role(heading, swiftui, header).

test(link_role_all_targets) :-
    map_a11y_role(link, react_native, link),
    map_a11y_role(link, vue, link),
    map_a11y_role(link, flutter, link),
    map_a11y_role(link, swiftui, link).

test(text_role_variations) :-
    map_a11y_role(text, react_native, text),
    map_a11y_role(text, vue, text),
    map_a11y_role(text, swiftui, staticText).

test(slider_role_variations) :-
    map_a11y_role(slider, react_native, adjustable),
    map_a11y_role(slider, vue, slider),
    map_a11y_role(slider, flutter, slider).

test(supported_roles_list) :-
    supported_roles(Roles),
    member(button, Roles),
    member(link, Roles),
    member(heading, Roles),
    member(checkbox, Roles),
    member(textfield, Roles).

:- end_tests(role_mapping).

% ============================================================================
% Tests: React Native Code Generation
% ============================================================================

:- begin_tests(react_native_codegen).

test(rn_label_only) :-
    generate_a11y_attrs(a11y([label('Submit')]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityLabel=\"Submit\"").

test(rn_role_only) :-
    generate_a11y_attrs(a11y([role(button)]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityRole=\"button\"").

test(rn_hint_only) :-
    generate_a11y_attrs(a11y([hint('Tap to submit')]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityHint=\"Tap to submit\"").

test(rn_all_attrs) :-
    generate_a11y_attrs(a11y([
        label('Submit'),
        role(button),
        hint('Submits form')
    ]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityLabel"),
    sub_string(Code, _, _, _, "accessibilityRole"),
    sub_string(Code, _, _, _, "accessibilityHint").

test(rn_disabled_state) :-
    generate_a11y_attrs(a11y([label('X'), disabled(true)]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityState={{ disabled: true }}").

test(rn_checked_state) :-
    generate_a11y_attrs(a11y([label('X'), checked(true)]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityState={{ checked: true }}").

test(rn_hidden) :-
    generate_a11y_attrs(a11y([label('X'), hidden(true)]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityElementsHidden={true}").

test(rn_live_region) :-
    generate_a11y_attrs(a11y([label('X'), live(polite)]), react_native, Code),
    sub_string(Code, _, _, _, "accessibilityLiveRegion=\"polite\"").

:- end_tests(react_native_codegen).

% ============================================================================
% Tests: Vue Code Generation
% ============================================================================

:- begin_tests(vue_codegen).

test(vue_label) :-
    generate_a11y_attrs(a11y([label('Submit')]), vue, Code),
    sub_string(Code, _, _, _, "aria-label=\"Submit\"").

test(vue_role) :-
    generate_a11y_attrs(a11y([role(button)]), vue, Code),
    sub_string(Code, _, _, _, "role=\"button\"").

test(vue_hint) :-
    generate_a11y_attrs(a11y([hint('description')]), vue, Code),
    sub_string(Code, _, _, _, "aria-describedby").

test(vue_disabled) :-
    generate_a11y_attrs(a11y([label('X'), disabled(true)]), vue, Code),
    sub_string(Code, _, _, _, "aria-disabled=\"true\"").

test(vue_required) :-
    generate_a11y_attrs(a11y([label('X'), required(true)]), vue, Code),
    sub_string(Code, _, _, _, "aria-required=\"true\"").

test(vue_hidden) :-
    generate_a11y_attrs(a11y([label('X'), hidden(true)]), vue, Code),
    sub_string(Code, _, _, _, "aria-hidden=\"true\"").

test(vue_checked) :-
    generate_a11y_attrs(a11y([label('X'), checked(true)]), vue, Code),
    sub_string(Code, _, _, _, "aria-checked=\"true\"").

test(vue_live) :-
    generate_a11y_attrs(a11y([label('X'), live(assertive)]), vue, Code),
    sub_string(Code, _, _, _, "aria-live=\"assertive\"").

:- end_tests(vue_codegen).

% ============================================================================
% Tests: Flutter Code Generation
% ============================================================================

:- begin_tests(flutter_codegen).

test(flutter_label) :-
    generate_a11y_attrs(a11y([label('Submit')]), flutter, Code),
    sub_string(Code, _, _, _, "Semantics("),
    sub_string(Code, _, _, _, "label: 'Submit'").

test(flutter_button_role) :-
    generate_a11y_attrs(a11y([role(button)]), flutter, Code),
    sub_string(Code, _, _, _, "button: true").

test(flutter_link_role) :-
    generate_a11y_attrs(a11y([role(link)]), flutter, Code),
    sub_string(Code, _, _, _, "link: true").

test(flutter_heading_role) :-
    generate_a11y_attrs(a11y([role(heading)]), flutter, Code),
    sub_string(Code, _, _, _, "header: true").

test(flutter_hint) :-
    generate_a11y_attrs(a11y([hint('Tap here')]), flutter, Code),
    sub_string(Code, _, _, _, "hint: 'Tap here'").

test(flutter_disabled) :-
    generate_a11y_attrs(a11y([label('X'), disabled(true)]), flutter, Code),
    sub_string(Code, _, _, _, "enabled: false").

test(flutter_checked) :-
    generate_a11y_attrs(a11y([label('X'), checked(true)]), flutter, Code),
    sub_string(Code, _, _, _, "checked: true").

test(flutter_hidden) :-
    generate_a11y_attrs(a11y([label('X'), hidden(true)]), flutter, Code),
    sub_string(Code, _, _, _, "excludeSemantics: true").

:- end_tests(flutter_codegen).

% ============================================================================
% Tests: SwiftUI Code Generation
% ============================================================================

:- begin_tests(swiftui_codegen).

test(swift_label) :-
    generate_a11y_attrs(a11y([label('Submit')]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityLabel(\"Submit\")").

test(swift_hint) :-
    generate_a11y_attrs(a11y([hint('Tap to submit')]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityHint(\"Tap to submit\")").

test(swift_button_trait) :-
    generate_a11y_attrs(a11y([role(button)]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityAddTraits(.isButton)").

test(swift_link_trait) :-
    generate_a11y_attrs(a11y([role(link)]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityAddTraits(.isLink)").

test(swift_header_trait) :-
    generate_a11y_attrs(a11y([role(heading)]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityAddTraits(.isHeader)").

test(swift_hidden) :-
    generate_a11y_attrs(a11y([label('X'), hidden(true)]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityHidden(true)").

test(swift_disabled) :-
    generate_a11y_attrs(a11y([label('X'), disabled(true)]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityAddTraits(.isDisabled)").

test(swift_selected) :-
    generate_a11y_attrs(a11y([label('X'), checked(true)]), swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityAddTraits(.isSelected)").

:- end_tests(swiftui_codegen).

% ============================================================================
% Tests: Spec Transformation
% ============================================================================

:- begin_tests(spec_transformation).

test(extract_simple_a11y) :-
    Spec = button(x, [a11y([label('X'), role(button)])]),
    extract_a11y(Spec, Attrs),
    member(label('X'), Attrs),
    member(role(button), Attrs).

test(extract_nested_a11y) :-
    Spec = screen([
        header(a11y([label('Header')])),
        body([button(x, [a11y([label('Button')])])])
    ]),
    extract_a11y(Spec, Attrs),
    member(label('Header'), Attrs),
    member(label('Button'), Attrs).

test(extract_no_a11y) :-
    Spec = button(x, [text('X')]),
    extract_a11y(Spec, Attrs),
    Attrs = [].

test(has_a11y_true) :-
    has_a11y(button(x, [a11y([label('X')])])).

test(has_a11y_false, [fail]) :-
    has_a11y(button(x, [text('X')])).

test(add_a11y_with_options) :-
    add_a11y_to_spec(button(x, [text('X')]), a11y([label('X')]), Result),
    has_a11y(Result).

test(add_a11y_without_options) :-
    add_a11y_to_spec(button(x), a11y([label('X')]), Result),
    has_a11y(Result).

:- end_tests(spec_transformation).

% ============================================================================
% Tests: Validation
% ============================================================================

:- begin_tests(validation).

test(valid_a11y_no_errors) :-
    validate_a11y(a11y([label('Submit'), role(button)]), Errors),
    \+ member(error(invalid_role, _), Errors),
    \+ member(error(empty_label, _), Errors).

test(missing_label_error) :-
    validate_a11y(a11y([role(button)]), Errors),
    member(error(missing_label, no_label), Errors).

test(invalid_role_error) :-
    validate_a11y(a11y([label('X'), role(nonexistent_role)]), Errors),
    member(error(invalid_role, nonexistent_role), Errors).

test(empty_label_error) :-
    validate_a11y(a11y([label(''), role(button)]), Errors),
    member(error(empty_label, empty), Errors).

test(empty_hint_error) :-
    validate_a11y(a11y([label('X'), hint('')]), Errors),
    member(error(empty_hint, empty), Errors).

:- end_tests(validation).

% ============================================================================
% Tests: Coverage Check
% ============================================================================

:- begin_tests(coverage).

test(full_coverage) :-
    Patterns = [
        button(a, [a11y([label('A')])]),
        button(b, [a11y([label('B')])])
    ],
    check_a11y_coverage(Patterns, coverage(total(2), covered(2), percentage(P))),
    P > 99.

test(partial_coverage) :-
    Patterns = [
        button(a, [a11y([label('A')])]),
        button(b, [text('B')]),
        button(c, [a11y([label('C')])])
    ],
    check_a11y_coverage(Patterns, coverage(total(3), covered(2), percentage(P))),
    P > 66,
    P < 67.

test(no_coverage) :-
    Patterns = [
        button(a, [text('A')]),
        button(b, [text('B')])
    ],
    check_a11y_coverage(Patterns, coverage(total(2), covered(0), percentage(P))),
    P < 1.

test(empty_patterns) :-
    check_a11y_coverage([], coverage(total(0), covered(0), percentage(0))).

:- end_tests(coverage).

% ============================================================================
% Tests: A11y Term Constructors
% ============================================================================

:- begin_tests(term_constructors).

test(a11y_label_term) :-
    a11y_label('Submit', label('Submit')).

test(a11y_role_term) :-
    a11y_role(button, role(button)).

test(a11y_hint_term) :-
    a11y_hint('Click here', hint('Click here')).

test(a11y_required_term) :-
    a11y_required(true, required(true)).

test(a11y_checked_term) :-
    a11y_checked(false, checked(false)).

test(a11y_disabled_term) :-
    a11y_disabled(true, disabled(true)).

test(a11y_hidden_term) :-
    a11y_hidden(true, hidden(true)).

test(a11y_live_term) :-
    a11y_live(polite, live(polite)).

:- end_tests(term_constructors).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
