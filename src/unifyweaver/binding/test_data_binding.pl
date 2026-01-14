% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_data_binding.pl - plunit tests for data binding system
%
% Run with: swipl -g "run_tests" -t halt test_data_binding.pl

:- module(test_data_binding, []).

:- use_module(library(plunit)).
:- use_module('data_binding').

% ============================================================================
% Tests: State Term Constructors
% ============================================================================

:- begin_tests(state_terms).

test(state_creates_spec) :-
    data_binding:state(counter, 0, [type(number)]).

test(state_with_string_initial) :-
    data_binding:state(name, '', [type(string)]).

test(state_with_list_initial) :-
    data_binding:state(items, [], [type(list)]).

test(state_with_boolean_initial) :-
    data_binding:state(enabled, true, [type(boolean)]).

:- end_tests(state_terms).

% ============================================================================
% Tests: Binding Term Constructors
% ============================================================================

:- begin_tests(binding_terms).

test(bind_creates_spec) :-
    data_binding:bind(label, counter).

test(two_way_creates_spec) :-
    data_binding:two_way(input, name).

test(on_creates_spec) :-
    data_binding:on(button, press, handleClick).

:- end_tests(binding_terms).

% ============================================================================
% Tests: Action Term Constructors
% ============================================================================

:- begin_tests(action_terms).

test(action_creates_spec) :-
    data_binding:action(increment, counter, 'n => n + 1').

test(store_action_creates_spec) :-
    data_binding:store_action(app, login, user, [set(loggedIn, true)]).

:- end_tests(action_terms).

% ============================================================================
% Tests: React Native State Generation
% ============================================================================

:- begin_tests(react_native_state).

test(generates_usestate) :-
    data_binding:generate_state(state(counter, 0, []), react_native, Code),
    sub_atom(Code, _, _, _, 'useState').

test(generates_setter_name) :-
    data_binding:generate_state(state(counter, 0, []), react_native, Code),
    sub_atom(Code, _, _, _, 'setCounter').

test(generates_initial_value) :-
    data_binding:generate_state(state(counter, 42, []), react_native, Code),
    sub_atom(Code, _, _, _, '42').

test(generates_string_initial) :-
    data_binding:generate_state(state(name, '', []), react_native, Code),
    sub_atom(Code, _, _, _, '""').

test(generates_list_initial) :-
    data_binding:generate_state(state(items, [], []), react_native, Code),
    sub_atom(Code, _, _, _, '[]').

:- end_tests(react_native_state).

% ============================================================================
% Tests: React Native Computed Generation
% ============================================================================

:- begin_tests(react_native_computed).

test(generates_usememo) :-
    data_binding:generate_computed(computed(doubled, [count], 'count * 2'), react_native, Code),
    sub_atom(Code, _, _, _, 'useMemo').

test(includes_dependencies) :-
    data_binding:generate_computed(computed(doubled, [count], 'count * 2'), react_native, Code),
    sub_atom(Code, _, _, _, '[count]').

test(includes_expression) :-
    data_binding:generate_computed(computed(sum, [a, b], 'a + b'), react_native, Code),
    sub_atom(Code, _, _, _, 'a + b').

test(multiple_deps) :-
    data_binding:generate_computed(computed(fullName, [first, last], 'first + last'), react_native, Code),
    sub_atom(Code, _, _, _, 'first, last').

:- end_tests(react_native_computed).

% ============================================================================
% Tests: React Native Effect Generation
% ============================================================================

:- begin_tests(react_native_effect).

test(generates_useeffect) :-
    data_binding:generate_effect(effect(log, [user], 'console.log(user)'), react_native, Code),
    sub_atom(Code, _, _, _, 'useEffect').

test(includes_body) :-
    data_binding:generate_effect(effect(log, [user], 'console.log(user)'), react_native, Code),
    sub_atom(Code, _, _, _, 'console.log').

test(includes_deps_array) :-
    data_binding:generate_effect(effect(log, [user], 'console.log(user)'), react_native, Code),
    sub_atom(Code, _, _, _, '[user]').

:- end_tests(react_native_effect).

% ============================================================================
% Tests: React Native Binding Generation
% ============================================================================

:- begin_tests(react_native_binding).

test(one_way_binding) :-
    data_binding:generate_binding(bind(text, counter), react_native, Code),
    sub_atom(Code, _, _, _, 'text={counter}').

test(two_way_binding) :-
    data_binding:generate_binding(two_way(value, name), react_native, Code),
    sub_atom(Code, _, _, _, 'onChangeText').

test(two_way_includes_setter) :-
    data_binding:generate_binding(two_way(value, name), react_native, Code),
    sub_atom(Code, _, _, _, 'setName').

test(event_binding_press) :-
    data_binding:generate_binding(on(button, press, handleClick), react_native, Code),
    sub_atom(Code, _, _, _, 'onPress').

:- end_tests(react_native_binding).

% ============================================================================
% Tests: Vue State Generation
% ============================================================================

:- begin_tests(vue_state).

test(generates_ref) :-
    data_binding:generate_state(state(counter, 0, []), vue, Code),
    sub_atom(Code, _, _, _, 'ref').

test(generates_reactive_when_specified) :-
    data_binding:generate_state(state(user, [], [reactive(true)]), vue, Code),
    sub_atom(Code, _, _, _, 'reactive').

test(generates_initial_value) :-
    data_binding:generate_state(state(counter, 5, []), vue, Code),
    sub_atom(Code, _, _, _, '5').

:- end_tests(vue_state).

% ============================================================================
% Tests: Vue Computed Generation
% ============================================================================

:- begin_tests(vue_computed).

test(generates_computed) :-
    data_binding:generate_computed(computed(doubled, [count], 'count.value * 2'), vue, Code),
    sub_atom(Code, _, _, _, 'computed').

test(arrow_function) :-
    data_binding:generate_computed(computed(doubled, [count], 'count.value * 2'), vue, Code),
    sub_atom(Code, _, _, _, '() =>').

:- end_tests(vue_computed).

% ============================================================================
% Tests: Vue Effect Generation
% ============================================================================

:- begin_tests(vue_effect).

test(generates_watch) :-
    data_binding:generate_effect(effect(log, [user], 'console.log(user)'), vue, Code),
    sub_atom(Code, _, _, _, 'watch').

test(single_dep_watch) :-
    data_binding:generate_effect(effect(log, [user], 'console.log(user)'), vue, Code),
    sub_atom(Code, _, _, _, 'watch(user').

test(multiple_deps_array) :-
    data_binding:generate_effect(effect(log, [a, b], 'console.log(a, b)'), vue, Code),
    sub_atom(Code, _, _, _, 'watch([').

:- end_tests(vue_effect).

% ============================================================================
% Tests: Vue Binding Generation
% ============================================================================

:- begin_tests(vue_binding).

test(one_way_binding) :-
    data_binding:generate_binding(bind(text, counter), vue, Code),
    sub_atom(Code, _, _, _, ':text').

test(two_way_binding) :-
    data_binding:generate_binding(two_way(value, name), vue, Code),
    sub_atom(Code, _, _, _, 'v-model').

test(event_binding) :-
    data_binding:generate_binding(on(button, click, handleClick), vue, Code),
    sub_atom(Code, _, _, _, '@click').

:- end_tests(vue_binding).

% ============================================================================
% Tests: Flutter State Generation
% ============================================================================

:- begin_tests(flutter_state).

test(generates_typed_state) :-
    data_binding:generate_state(state(counter, 0, [type(number)]), flutter, Code),
    sub_atom(Code, _, _, _, 'int').

test(generates_private_var) :-
    data_binding:generate_state(state(counter, 0, [type(number)]), flutter, Code),
    sub_atom(Code, _, _, _, '_counter').

test(string_type) :-
    data_binding:generate_state(state(name, '', [type(string)]), flutter, Code),
    sub_atom(Code, _, _, _, 'String').

test(bool_type) :-
    data_binding:generate_state(state(enabled, true, [type(boolean)]), flutter, Code),
    sub_atom(Code, _, _, _, 'bool').

:- end_tests(flutter_state).

% ============================================================================
% Tests: Flutter Computed Generation
% ============================================================================

:- begin_tests(flutter_computed).

test(generates_getter) :-
    data_binding:generate_computed(computed(doubled, [count], '_count * 2'), flutter, Code),
    sub_atom(Code, _, _, _, 'get doubled').

:- end_tests(flutter_computed).

% ============================================================================
% Tests: Flutter Action Generation
% ============================================================================

:- begin_tests(flutter_action).

test(generates_setstate) :-
    data_binding:generate_action(action(increment, counter, '_counter + 1'), flutter, Code),
    sub_atom(Code, _, _, _, 'setState').

test(generates_function) :-
    data_binding:generate_action(action(increment, counter, '_counter + 1'), flutter, Code),
    sub_atom(Code, _, _, _, 'void increment').

:- end_tests(flutter_action).

% ============================================================================
% Tests: SwiftUI State Generation
% ============================================================================

:- begin_tests(swiftui_state).

test(generates_state_property) :-
    data_binding:generate_state(state(counter, 0, [type(number)]), swiftui, Code),
    sub_atom(Code, _, _, _, '@State').

test(generates_private) :-
    data_binding:generate_state(state(counter, 0, [type(number)]), swiftui, Code),
    sub_atom(Code, _, _, _, 'private').

test(swift_int_type) :-
    data_binding:generate_state(state(counter, 0, [type(number)]), swiftui, Code),
    sub_atom(Code, _, _, _, 'Int').

test(swift_string_type) :-
    data_binding:generate_state(state(name, '', [type(string)]), swiftui, Code),
    sub_atom(Code, _, _, _, 'String').

test(binding_mode) :-
    data_binding:generate_state(state(value, 0, [type(number), binding(true)]), swiftui, Code),
    sub_atom(Code, _, _, _, '@Binding').

:- end_tests(swiftui_state).

% ============================================================================
% Tests: SwiftUI Computed Generation
% ============================================================================

:- begin_tests(swiftui_computed).

test(generates_computed_property) :-
    data_binding:generate_computed(computed(doubled, [count], 'count * 2'), swiftui, Code),
    sub_atom(Code, _, _, _, 'var doubled').

:- end_tests(swiftui_computed).

% ============================================================================
% Tests: SwiftUI Effect Generation
% ============================================================================

:- begin_tests(swiftui_effect).

test(generates_onchange) :-
    data_binding:generate_effect(effect(log, [user], 'print(user)'), swiftui, Code),
    sub_atom(Code, _, _, _, '.onChange').

:- end_tests(swiftui_effect).

% ============================================================================
% Tests: SwiftUI Binding Generation
% ============================================================================

:- begin_tests(swiftui_binding).

test(two_way_dollar_sign) :-
    data_binding:generate_binding(two_way('TextField', name), swiftui, Code),
    sub_atom(Code, _, _, _, '$name').

:- end_tests(swiftui_binding).

% ============================================================================
% Tests: Validation
% ============================================================================

:- begin_tests(validation).

test(valid_state_no_errors) :-
    data_binding:validate_state(state(counter, 0, []), Errors),
    Errors = [].

test(valid_binding_no_errors) :-
    data_binding:validate_binding(bind(text, counter), Errors),
    Errors = [].

test(valid_two_way_no_errors) :-
    data_binding:validate_binding(two_way(input, name), Errors),
    Errors = [].

test(valid_event_no_errors) :-
    data_binding:validate_binding(on(button, press, handler), Errors),
    Errors = [].

:- end_tests(validation).

% ============================================================================
% Tests: Store Generation
% ============================================================================

:- begin_tests(store_generation).

test(react_native_zustand_store) :-
    data_binding:generate_store(store(app, [
        slice(user, [field(name, string, '')], [])
    ], []), react_native, Code),
    sub_atom(Code, _, _, _, 'create').

test(vue_pinia_store) :-
    data_binding:generate_store(store(app, [
        slice(user, [field(name, string, '')], [])
    ], []), vue, Code),
    sub_atom(Code, _, _, _, 'defineStore').

test(flutter_changenotifier) :-
    data_binding:generate_store(store(app, [
        slice(user, [field(name, string, '')], [])
    ], []), flutter, Code),
    sub_atom(Code, _, _, _, 'ChangeNotifier').

test(swiftui_observableobject) :-
    data_binding:generate_store(store(app, [
        slice(user, [field(name, string, '')], [])
    ], []), swiftui, Code),
    sub_atom(Code, _, _, _, 'ObservableObject').

test(swiftui_published) :-
    data_binding:generate_store(store(app, [
        slice(user, [field(name, string, '')], [])
    ], []), swiftui, Code),
    sub_atom(Code, _, _, _, '@Published').

:- end_tests(store_generation).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
