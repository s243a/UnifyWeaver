% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% accessibility.pl - Cross-platform accessibility patterns
%
% Provides a11y attributes that compile across React Native, Vue,
% Flutter, and SwiftUI targets.
%
% Usage:
%   use_module('src/unifyweaver/a11y/accessibility').
%   A11y = a11y([label('Submit'), role(button), hint('Submit form')]),
%   generate_a11y_attrs(A11y, react_native, Code).

:- module(accessibility, [
    % A11y term constructors
    a11y/1,
    a11y_label/2,
    a11y_role/2,
    a11y_hint/2,
    a11y_required/2,
    a11y_checked/2,
    a11y_disabled/2,
    a11y_hidden/2,
    a11y_live/2,

    % Role mapping
    map_a11y_role/3,
    supported_roles/1,

    % Code generation
    generate_a11y_attrs/3,
    generate_react_native_a11y/2,
    generate_vue_a11y/2,
    generate_flutter_a11y/2,
    generate_swiftui_a11y/2,

    % Spec transformation
    add_a11y_to_spec/3,
    extract_a11y/2,
    has_a11y/1,

    % Validation
    validate_a11y/2,
    check_a11y_coverage/2,

    % Testing
    test_accessibility/0
]).

:- use_module(library(lists)).

% ============================================================================
% A11y Term Constructors
% ============================================================================

%! a11y(+Attrs) is det
%  Term constructor for accessibility attributes.
%  Attrs is a list of a11y options like label/1, role/1, hint/1.
a11y(Attrs) :-
    is_list(Attrs).

%! a11y_label(+Text, -Term) is det
%  Create a label attribute.
a11y_label(Text, label(Text)).

%! a11y_role(+Role, -Term) is det
%  Create a role attribute.
a11y_role(Role, role(Role)).

%! a11y_hint(+Text, -Term) is det
%  Create a hint attribute.
a11y_hint(Text, hint(Text)).

%! a11y_required(+Bool, -Term) is det
%  Create a required attribute.
a11y_required(Bool, required(Bool)).

%! a11y_checked(+Bool, -Term) is det
%  Create a checked state attribute.
a11y_checked(Bool, checked(Bool)).

%! a11y_disabled(+Bool, -Term) is det
%  Create a disabled state attribute.
a11y_disabled(Bool, disabled(Bool)).

%! a11y_hidden(+Bool, -Term) is det
%  Create a hidden attribute (hide from screen readers).
a11y_hidden(Bool, hidden(Bool)).

%! a11y_live(+Mode, -Term) is det
%  Create a live region attribute. Mode: polite, assertive, off
a11y_live(Mode, live(Mode)).

% ============================================================================
% Role Mapping
% ============================================================================

%! supported_roles(-Roles) is det
%  List of supported accessibility roles.
supported_roles([
    button, link, heading, image, text, checkbox, radio,
    slider, switch, tab, tablist, menu, menuitem, alert,
    dialog, textfield, searchfield, progressbar, list, listitem,
    none
]).

%! map_a11y_role(+Role, +Target, -TargetRole) is det
%  Map abstract role to target-specific role.

% Button role
map_a11y_role(button, react_native, button).
map_a11y_role(button, vue, button).
map_a11y_role(button, flutter, button).
map_a11y_role(button, swiftui, button).

% Link role
map_a11y_role(link, react_native, link).
map_a11y_role(link, vue, link).
map_a11y_role(link, flutter, link).
map_a11y_role(link, swiftui, link).

% Heading role
map_a11y_role(heading, react_native, header).
map_a11y_role(heading, vue, heading).
map_a11y_role(heading, flutter, header).
map_a11y_role(heading, swiftui, header).

% Image role
map_a11y_role(image, react_native, image).
map_a11y_role(image, vue, img).
map_a11y_role(image, flutter, image).
map_a11y_role(image, swiftui, image).

% Text role
map_a11y_role(text, react_native, text).
map_a11y_role(text, vue, text).
map_a11y_role(text, flutter, text).
map_a11y_role(text, swiftui, staticText).

% Checkbox role
map_a11y_role(checkbox, react_native, checkbox).
map_a11y_role(checkbox, vue, checkbox).
map_a11y_role(checkbox, flutter, checkbox).
map_a11y_role(checkbox, swiftui, checkbox).

% Radio role
map_a11y_role(radio, react_native, radio).
map_a11y_role(radio, vue, radio).
map_a11y_role(radio, flutter, radio).
map_a11y_role(radio, swiftui, radioButton).

% Slider role
map_a11y_role(slider, react_native, adjustable).
map_a11y_role(slider, vue, slider).
map_a11y_role(slider, flutter, slider).
map_a11y_role(slider, swiftui, slider).

% Switch role
map_a11y_role(switch, react_native, switch).
map_a11y_role(switch, vue, switch).
map_a11y_role(switch, flutter, toggleButton).
map_a11y_role(switch, swiftui, switch).

% Tab roles
map_a11y_role(tab, react_native, tab).
map_a11y_role(tab, vue, tab).
map_a11y_role(tab, flutter, tab).
map_a11y_role(tab, swiftui, tab).

map_a11y_role(tablist, react_native, tablist).
map_a11y_role(tablist, vue, tablist).
map_a11y_role(tablist, flutter, tabBar).
map_a11y_role(tablist, swiftui, tabBar).

% Menu roles
map_a11y_role(menu, react_native, menu).
map_a11y_role(menu, vue, menu).
map_a11y_role(menu, flutter, menu).
map_a11y_role(menu, swiftui, menu).

map_a11y_role(menuitem, react_native, menuitem).
map_a11y_role(menuitem, vue, menuitem).
map_a11y_role(menuitem, flutter, menuItem).
map_a11y_role(menuitem, swiftui, menuItem).

% Alert role
map_a11y_role(alert, react_native, alert).
map_a11y_role(alert, vue, alert).
map_a11y_role(alert, flutter, alert).
map_a11y_role(alert, swiftui, alert).

% Dialog role
map_a11y_role(dialog, react_native, none).
map_a11y_role(dialog, vue, dialog).
map_a11y_role(dialog, flutter, dialog).
map_a11y_role(dialog, swiftui, dialog).

% Text field role
map_a11y_role(textfield, react_native, none).
map_a11y_role(textfield, vue, textbox).
map_a11y_role(textfield, flutter, textField).
map_a11y_role(textfield, swiftui, textField).

map_a11y_role(searchfield, react_native, search).
map_a11y_role(searchfield, vue, searchbox).
map_a11y_role(searchfield, flutter, textField).
map_a11y_role(searchfield, swiftui, searchField).

% Progress bar role
map_a11y_role(progressbar, react_native, progressbar).
map_a11y_role(progressbar, vue, progressbar).
map_a11y_role(progressbar, flutter, progressIndicator).
map_a11y_role(progressbar, swiftui, progressIndicator).

% List roles
map_a11y_role(list, react_native, list).
map_a11y_role(list, vue, list).
map_a11y_role(list, flutter, list).
map_a11y_role(list, swiftui, list).

map_a11y_role(listitem, react_native, none).
map_a11y_role(listitem, vue, listitem).
map_a11y_role(listitem, flutter, listItem).
map_a11y_role(listitem, swiftui, listItem).

% None role (hide semantics)
map_a11y_role(none, react_native, none).
map_a11y_role(none, vue, presentation).
map_a11y_role(none, flutter, none).
map_a11y_role(none, swiftui, none).

% ============================================================================
% Code Generation - Main Entry Point
% ============================================================================

%! generate_a11y_attrs(+A11ySpec, +Target, -Code) is det
%  Generate target-specific accessibility code from a11y spec.
generate_a11y_attrs(a11y(Attrs), Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_a11y(Attrs, Code)
    ;   Target = vue
    ->  generate_vue_a11y(Attrs, Code)
    ;   Target = flutter
    ->  generate_flutter_a11y(Attrs, Code)
    ;   Target = swiftui
    ->  generate_swiftui_a11y(Attrs, Code)
    ;   Code = ""
    ).

% ============================================================================
% React Native Code Generation
% ============================================================================

%! generate_react_native_a11y(+Attrs, -Code) is det
%  Generate React Native accessibility props.
generate_react_native_a11y(Attrs, Code) :-
    findall(Prop, rn_attr_to_prop(Attrs, Prop), Props),
    Props \= [],
    atomic_list_concat(Props, '\n  ', Code).
generate_react_native_a11y(_, "").

rn_attr_to_prop(Attrs, Prop) :-
    member(label(Text), Attrs),
    format(atom(Prop), 'accessibilityLabel="~w"', [Text]).
rn_attr_to_prop(Attrs, Prop) :-
    member(role(Role), Attrs),
    map_a11y_role(Role, react_native, RNRole),
    format(atom(Prop), 'accessibilityRole="~w"', [RNRole]).
rn_attr_to_prop(Attrs, Prop) :-
    member(hint(Text), Attrs),
    format(atom(Prop), 'accessibilityHint="~w"', [Text]).
rn_attr_to_prop(Attrs, Prop) :-
    member(disabled(true), Attrs),
    Prop = 'accessibilityState={{ disabled: true }}'.
rn_attr_to_prop(Attrs, Prop) :-
    member(checked(Bool), Attrs),
    format(atom(Prop), 'accessibilityState={{ checked: ~w }}', [Bool]).
rn_attr_to_prop(Attrs, Prop) :-
    member(hidden(true), Attrs),
    Prop = 'accessibilityElementsHidden={true}'.
rn_attr_to_prop(Attrs, Prop) :-
    member(live(Mode), Attrs),
    format(atom(Prop), 'accessibilityLiveRegion="~w"', [Mode]).

% ============================================================================
% Vue Code Generation
% ============================================================================

%! generate_vue_a11y(+Attrs, -Code) is det
%  Generate Vue ARIA attributes.
generate_vue_a11y(Attrs, Code) :-
    findall(Attr, vue_attr_to_aria(Attrs, Attr), AriaAttrs),
    AriaAttrs \= [],
    atomic_list_concat(AriaAttrs, '\n  ', Code).
generate_vue_a11y(_, "").

vue_attr_to_aria(Attrs, Attr) :-
    member(label(Text), Attrs),
    format(atom(Attr), 'aria-label="~w"', [Text]).
vue_attr_to_aria(Attrs, Attr) :-
    member(role(Role), Attrs),
    map_a11y_role(Role, vue, VueRole),
    format(atom(Attr), 'role="~w"', [VueRole]).
vue_attr_to_aria(Attrs, Attr) :-
    member(hint(Text), Attrs),
    format(atom(Attr), 'aria-describedby="~w"', [Text]).
vue_attr_to_aria(Attrs, Attr) :-
    member(disabled(true), Attrs),
    Attr = 'aria-disabled="true"'.
vue_attr_to_aria(Attrs, Attr) :-
    member(checked(Bool), Attrs),
    format(atom(Attr), 'aria-checked="~w"', [Bool]).
vue_attr_to_aria(Attrs, Attr) :-
    member(hidden(true), Attrs),
    Attr = 'aria-hidden="true"'.
vue_attr_to_aria(Attrs, Attr) :-
    member(required(true), Attrs),
    Attr = 'aria-required="true"'.
vue_attr_to_aria(Attrs, Attr) :-
    member(live(Mode), Attrs),
    format(atom(Attr), 'aria-live="~w"', [Mode]).

% ============================================================================
% Flutter Code Generation
% ============================================================================

%! generate_flutter_a11y(+Attrs, -Code) is det
%  Generate Flutter Semantics widget properties.
generate_flutter_a11y(Attrs, Code) :-
    findall(Prop, flutter_attr_to_prop(Attrs, Prop), Props),
    Props \= [],
    atomic_list_concat(Props, ',\n  ', PropsStr),
    format(atom(Code), 'Semantics(\n  ~w,\n  child: ', [PropsStr]).
generate_flutter_a11y(_, "").

flutter_attr_to_prop(Attrs, Prop) :-
    member(label(Text), Attrs),
    format(atom(Prop), 'label: \'~w\'', [Text]).
flutter_attr_to_prop(Attrs, Prop) :-
    member(role(button), Attrs),
    Prop = 'button: true'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(role(link), Attrs),
    Prop = 'link: true'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(role(heading), Attrs),
    Prop = 'header: true'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(role(image), Attrs),
    Prop = 'image: true'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(role(textfield), Attrs),
    Prop = 'textField: true'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(role(slider), Attrs),
    Prop = 'slider: true'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(role(checkbox), Attrs),
    Prop = 'inMutuallyExclusiveGroup: false'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(hint(Text), Attrs),
    format(atom(Prop), 'hint: \'~w\'', [Text]).
flutter_attr_to_prop(Attrs, Prop) :-
    member(disabled(true), Attrs),
    Prop = 'enabled: false'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(checked(Bool), Attrs),
    format(atom(Prop), 'checked: ~w', [Bool]).
flutter_attr_to_prop(Attrs, Prop) :-
    member(hidden(true), Attrs),
    Prop = 'excludeSemantics: true'.
flutter_attr_to_prop(Attrs, Prop) :-
    member(live(Mode), Attrs),
    live_region_mode(Mode, FlutterMode),
    format(atom(Prop), 'liveRegion: ~w', [FlutterMode]).

live_region_mode(polite, 'LiveRegion.polite').
live_region_mode(assertive, 'LiveRegion.assertive').
live_region_mode(off, 'null').

% ============================================================================
% SwiftUI Code Generation
% ============================================================================

%! generate_swiftui_a11y(+Attrs, -Code) is det
%  Generate SwiftUI accessibility modifiers.
generate_swiftui_a11y(Attrs, Code) :-
    findall(Mod, swiftui_attr_to_mod(Attrs, Mod), Mods),
    Mods \= [],
    atomic_list_concat(Mods, '\n  ', Code).
generate_swiftui_a11y(_, "").

swiftui_attr_to_mod(Attrs, Mod) :-
    member(label(Text), Attrs),
    format(atom(Mod), '.accessibilityLabel("~w")', [Text]).
swiftui_attr_to_mod(Attrs, Mod) :-
    member(hint(Text), Attrs),
    format(atom(Mod), '.accessibilityHint("~w")', [Text]).
swiftui_attr_to_mod(Attrs, Mod) :-
    member(role(Role), Attrs),
    map_a11y_role(Role, swiftui, SwiftRole),
    swiftui_add_trait(SwiftRole, Mod).
swiftui_attr_to_mod(Attrs, Mod) :-
    member(disabled(true), Attrs),
    Mod = '.accessibilityAddTraits(.isDisabled)'.
swiftui_attr_to_mod(Attrs, Mod) :-
    member(hidden(true), Attrs),
    Mod = '.accessibilityHidden(true)'.
swiftui_attr_to_mod(Attrs, Mod) :-
    member(checked(true), Attrs),
    Mod = '.accessibilityAddTraits(.isSelected)'.

swiftui_add_trait(button, '.accessibilityAddTraits(.isButton)').
swiftui_add_trait(link, '.accessibilityAddTraits(.isLink)').
swiftui_add_trait(header, '.accessibilityAddTraits(.isHeader)').
swiftui_add_trait(image, '.accessibilityAddTraits(.isImage)').
swiftui_add_trait(staticText, '.accessibilityAddTraits(.isStaticText)').
swiftui_add_trait(searchField, '.accessibilityAddTraits(.isSearchField)').
swiftui_add_trait(tab, '.accessibilityAddTraits(.isTab)').
swiftui_add_trait(_, '').

% ============================================================================
% Spec Transformation
% ============================================================================

%! add_a11y_to_spec(+Spec, +A11y, -SpecWithA11y) is det
%  Add a11y attributes to a spec.
add_a11y_to_spec(Spec, A11y, SpecWithA11y) :-
    Spec =.. [Functor|Args],
    (   append(Args0, [Options], Args),
        is_list(Options)
    ->  append(Options, [A11y], NewOptions),
        append(Args0, [NewOptions], NewArgs)
    ;   append(Args, [[A11y]], NewArgs)
    ),
    SpecWithA11y =.. [Functor|NewArgs].

%! extract_a11y(+Spec, -A11yAttrs) is det
%  Extract a11y attributes from a spec.
extract_a11y(Spec, A11yAttrs) :-
    extract_a11y_recursive(Spec, [], A11yAttrs).

extract_a11y_recursive(a11y(Attrs), Acc, Result) :-
    !,
    append(Acc, Attrs, Result).
extract_a11y_recursive(Spec, Acc, Result) :-
    is_list(Spec),
    !,
    foldl(extract_a11y_from_list, Spec, Acc, Result).
extract_a11y_recursive(Spec, Acc, Result) :-
    compound(Spec),
    !,
    Spec =.. [_|Args],
    foldl(extract_a11y_from_list, Args, Acc, Result).
extract_a11y_recursive(_, Acc, Acc).

extract_a11y_from_list(Item, Acc, Result) :-
    extract_a11y_recursive(Item, Acc, Result).

%! has_a11y(+Spec) is semidet
%  Check if a spec contains a11y attributes.
has_a11y(Spec) :-
    extract_a11y(Spec, Attrs),
    Attrs \= [].

% ============================================================================
% Validation
% ============================================================================

%! validate_a11y(+A11ySpec, -Errors) is det
%  Validate a11y specification.
validate_a11y(a11y(Attrs), Errors) :-
    findall(Error, validate_a11y_attr(Attrs, Error), Errors).

validate_a11y_attr(Attrs, error(missing_label, no_label)) :-
    \+ member(label(_), Attrs).

validate_a11y_attr(Attrs, error(invalid_role, Role)) :-
    member(role(Role), Attrs),
    supported_roles(Supported),
    \+ member(Role, Supported).

validate_a11y_attr(Attrs, error(empty_label, empty)) :-
    member(label(''), Attrs).

validate_a11y_attr(Attrs, error(empty_hint, empty)) :-
    member(hint(''), Attrs).

%! check_a11y_coverage(+Patterns, -Report) is det
%  Check a11y coverage for a list of patterns.
check_a11y_coverage(Patterns, Report) :-
    length(Patterns, Total),
    include(has_a11y, Patterns, WithA11y),
    length(WithA11y, Covered),
    (   Total > 0
    ->  Percentage is (Covered * 100) / Total
    ;   Percentage = 0
    ),
    Report = coverage(total(Total), covered(Covered), percentage(Percentage)).

% ============================================================================
% Testing
% ============================================================================

%! test_accessibility is det
%  Run inline tests.
test_accessibility :-
    format('Running accessibility tests...~n'),

    % Test 1: Role mapping
    map_a11y_role(button, react_native, button),
    map_a11y_role(heading, react_native, header),
    format('  Test 1 passed: role mapping~n'),

    % Test 2: React Native code generation
    generate_a11y_attrs(a11y([label('Submit'), role(button)]), react_native, RNCode),
    sub_string(RNCode, _, _, _, "accessibilityLabel"),
    sub_string(RNCode, _, _, _, "accessibilityRole"),
    format('  Test 2 passed: React Native code generation~n'),

    % Test 3: Vue code generation
    generate_a11y_attrs(a11y([label('Submit'), role(button)]), vue, VueCode),
    sub_string(VueCode, _, _, _, "aria-label"),
    sub_string(VueCode, _, _, _, "role="),
    format('  Test 3 passed: Vue code generation~n'),

    % Test 4: Flutter code generation
    generate_a11y_attrs(a11y([label('Submit'), role(button)]), flutter, FlutterCode),
    sub_string(FlutterCode, _, _, _, "Semantics"),
    sub_string(FlutterCode, _, _, _, "button: true"),
    format('  Test 4 passed: Flutter code generation~n'),

    % Test 5: SwiftUI code generation
    generate_a11y_attrs(a11y([label('Submit'), role(button)]), swiftui, SwiftCode),
    sub_string(SwiftCode, _, _, _, ".accessibilityLabel"),
    sub_string(SwiftCode, _, _, _, ".accessibilityAddTraits"),
    format('  Test 5 passed: SwiftUI code generation~n'),

    % Test 6: Extract a11y
    Spec = button(submit, [text('OK'), a11y([label('Submit'), role(button)])]),
    extract_a11y(Spec, Attrs),
    member(label('Submit'), Attrs),
    member(role(button), Attrs),
    format('  Test 6 passed: extract a11y~n'),

    % Test 7: Has a11y
    has_a11y(button(x, [a11y([label('X')])])),
    \+ has_a11y(button(x, [])),
    format('  Test 7 passed: has_a11y~n'),

    % Test 8: Validation
    validate_a11y(a11y([label('X'), role(button)]), Errors1),
    Errors1 = [],
    validate_a11y(a11y([role(invalid_role)]), Errors2),
    member(error(invalid_role, invalid_role), Errors2),
    format('  Test 8 passed: validation~n'),

    % Test 9: Coverage check
    Patterns = [
        button(a, [a11y([label('A')])]),
        button(b, []),
        button(c, [a11y([label('C')])])
    ],
    check_a11y_coverage(Patterns, coverage(total(3), covered(2), percentage(P))),
    P > 60,
    format('  Test 9 passed: coverage check~n'),

    % Test 10: Add a11y to spec
    add_a11y_to_spec(button(x, [text('X')]), a11y([label('X')]), WithA11y),
    has_a11y(WithA11y),
    format('  Test 10 passed: add a11y to spec~n'),

    format('All 10 accessibility tests passed!~n').

:- initialization(test_accessibility, main).
