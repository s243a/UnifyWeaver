% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_ui_patterns_extended.pl - plunit tests for Extended UI Patterns
%
% Tests form, list, modal, and auth flow patterns.
%
% Run with: swipl -g "run_tests" -t halt test_ui_patterns_extended.pl

:- module(test_ui_patterns_extended, []).

:- use_module(library(plunit)).
:- use_module('ui_patterns_extended').
:- use_module('ui_patterns').

% ============================================================================
% Tests: Form Patterns
% ============================================================================

:- begin_tests(form_patterns).

test(form_pattern_creates_form) :-
    form_pattern(test_form1, [
        field(email, email, [required], []),
        field(password, password, [required], [])
    ], Pattern),
    Pattern = form(test_form1, _, _).

test(form_field_creates_field_spec) :-
    form_field(username, text, [required, min_length(3)], Spec),
    Spec = field(username, text, [required, min_length(3)], []).

test(form_field_accepts_all_types) :-
    Types = [text, email, password, number, phone, date, select, checkbox, radio, textarea, file],
    forall(member(T, Types),
           form_field(test, T, [], _)).

test(validation_rule_required) :-
    validation_rule(required, [], Rule),
    Rule = required.

test(validation_rule_min_length) :-
    validation_rule(min_length, [5], Rule),
    Rule = min_length(5).

test(validation_rule_max_length) :-
    validation_rule(max_length, [100], Rule),
    Rule = max_length(100).

test(validation_rule_pattern) :-
    validation_rule(pattern, ['^[a-z]+$'], Rule),
    Rule = pattern('^[a-z]+$').

test(validation_rule_email) :-
    validation_rule(email, [], email).

test(validation_rule_matches) :-
    validation_rule(matches, [password], Rule),
    Rule = matches(password).

:- end_tests(form_patterns).

% ============================================================================
% Tests: List Patterns
% ============================================================================

:- begin_tests(list_patterns).

test(infinite_list_creates_pattern) :-
    infinite_list(items_list, '/api/items', Pattern),
    Pattern = list(infinite, items_list, '/api/items', _).

test(selectable_list_single_mode) :-
    selectable_list(select_list, [mode(single)], Pattern),
    Pattern = list(selectable, select_list, single, _).

test(selectable_list_multi_mode) :-
    selectable_list(multi_list, [mode(multi)], Pattern),
    Pattern = list(selectable, multi_list, multi, _).

test(selectable_list_default_mode) :-
    selectable_list(default_list, [], Pattern),
    Pattern = list(selectable, default_list, single, _).

test(grouped_list_creates_pattern) :-
    grouped_list(grouped, category, Pattern),
    Pattern = list(grouped, grouped, category, _).

test(list_pattern_basic) :-
    list_pattern(basic, [name(basic_list), item_component('ListItem')], Pattern),
    Pattern = list(basic, basic_list, 'ListItem', _).

:- end_tests(list_patterns).

% ============================================================================
% Tests: Modal Patterns
% ============================================================================

:- begin_tests(modal_patterns).

test(alert_modal_creates_pattern) :-
    alert_modal(my_alert, [title('Alert'), message('Something happened')], Pattern),
    Pattern = modal(alert, my_alert, _).

test(confirm_modal_creates_pattern) :-
    confirm_modal(my_confirm, [title('Confirm'), message('Are you sure?')], Pattern),
    Pattern = modal(confirm, my_confirm, _).

test(confirm_modal_has_default_buttons) :-
    confirm_modal(confirm_default, [title('T'), message('M')], modal(confirm, _, Config)),
    member(confirm_text('Confirm'), Config),
    member(cancel_text('Cancel'), Config).

test(confirm_modal_custom_buttons) :-
    confirm_modal(confirm_custom, [
        title('Delete'),
        message('Delete item?'),
        confirm_text('Delete'),
        cancel_text('Keep')
    ], modal(confirm, _, Config)),
    member(confirm_text('Delete'), Config),
    member(cancel_text('Keep'), Config).

test(bottom_sheet_creates_pattern) :-
    bottom_sheet(my_sheet, 'SheetContent', Pattern),
    Pattern = modal(bottom_sheet, my_sheet, _).

test(action_sheet_creates_pattern) :-
    action_sheet(actions, [action(edit, 'Edit'), action(delete, 'Delete')], Pattern),
    Pattern = modal(action_sheet, actions, _).

test(modal_pattern_fullscreen) :-
    modal_pattern(fullscreen, [name(full), content('FullContent')], Pattern),
    Pattern = modal(fullscreen, full, _).

:- end_tests(modal_patterns).

% ============================================================================
% Tests: Auth Flow Patterns
% ============================================================================

:- begin_tests(auth_flow_patterns).

test(login_flow_creates_pattern) :-
    login_flow([endpoint('/api/login')], Pattern),
    Pattern = auth(login, Config),
    member(endpoint('/api/login'), Config).

test(login_flow_has_default_fields) :-
    login_flow([endpoint('/api/login')], auth(login, Config)),
    member(fields(Fields), Config),
    length(Fields, 2).

test(register_flow_creates_pattern) :-
    register_flow([endpoint('/api/register')], Pattern),
    Pattern = auth(register, Config),
    member(endpoint('/api/register'), Config).

test(register_flow_has_confirm_password) :-
    register_flow([endpoint('/api/register')], auth(register, Config)),
    member(fields(Fields), Config),
    member(field(confirm_password, password, _, _), Fields).

test(forgot_password_flow_creates_pattern) :-
    forgot_password_flow([endpoint('/api/forgot-password')], Pattern),
    Pattern = auth(forgot_password, _).

test(oauth_flow_creates_pattern) :-
    oauth_flow(google, [client_id('test-client-id')], Pattern),
    Pattern = auth(oauth, Config),
    member(provider(google), Config),
    member(client_id('test-client-id'), Config).

test(auth_flow_mfa) :-
    auth_flow(mfa, [endpoint('/api/mfa/verify')], Pattern),
    Pattern = auth(mfa, Config),
    member(code_length(6), Config).

test(auth_flow_reset_password) :-
    auth_flow(reset_password, [endpoint('/api/reset-password')], Pattern),
    Pattern = auth(reset_password, Config),
    member(fields(Fields), Config),
    member(field(password, password, _, _), Fields),
    member(field(confirm_password, password, _, _), Fields).

:- end_tests(auth_flow_patterns).

% ============================================================================
% Tests: Form Compilation
% ============================================================================

:- begin_tests(form_compilation).

test(form_compiles_to_react_native) :-
    ui_patterns_extended:compile_form_pattern(login, [
        field(email, email, [required], []),
        field(password, password, [required], [])
    ], react_native, [], Code),
    sub_string(Code, _, _, _, "useForm"),
    sub_string(Code, _, _, _, "Controller").

test(form_compiles_to_vue) :-
    ui_patterns_extended:compile_form_pattern(login, [
        field(email, email, [required], []),
        field(password, password, [required], [])
    ], vue, [], Code),
    sub_string(Code, _, _, _, "vee-validate"),
    sub_string(Code, _, _, _, "<form").

test(form_compiles_to_flutter) :-
    ui_patterns_extended:compile_form_pattern(login, [
        field(email, email, [required], []),
        field(password, password, [required], [])
    ], flutter, [], Code),
    sub_string(Code, _, _, _, "TextFormField"),
    sub_string(Code, _, _, _, "GlobalKey<FormState>").

test(form_compiles_to_swiftui) :-
    ui_patterns_extended:compile_form_pattern(login, [
        field(email, email, [required], []),
        field(password, password, [required], [])
    ], swiftui, [], Code),
    sub_string(Code, _, _, _, "TextField"),
    sub_string(Code, _, _, _, "@State").

test(form_includes_email_field) :-
    ui_patterns_extended:compile_form_pattern(test, [
        field(email, email, [required], [])
    ], react_native, [], Code),
    sub_string(Code, _, _, _, "email-address").

test(form_includes_password_field) :-
    ui_patterns_extended:compile_form_pattern(test, [
        field(password, password, [required], [])
    ], react_native, [], Code),
    sub_string(Code, _, _, _, "secureTextEntry").

:- end_tests(form_compilation).

% ============================================================================
% Tests: List Compilation
% ============================================================================

:- begin_tests(list_compilation).

test(infinite_list_compiles_to_react_native) :-
    ui_patterns_extended:compile_list_pattern(infinite, items, '/api/items', react_native, [], Code),
    sub_string(Code, _, _, _, "FlatList"),
    sub_string(Code, _, _, _, "useInfiniteQuery").

test(infinite_list_compiles_to_vue) :-
    ui_patterns_extended:compile_list_pattern(infinite, items, '/api/items', vue, [], Code),
    sub_string(Code, _, _, _, "IntersectionObserver"),
    sub_string(Code, _, _, _, "useInfiniteQuery").

test(infinite_list_compiles_to_flutter) :-
    ui_patterns_extended:compile_list_pattern(infinite, items, '/api/items', flutter, [], Code),
    sub_string(Code, _, _, _, "PagingController"),
    sub_string(Code, _, _, _, "RefreshIndicator").

test(infinite_list_compiles_to_swiftui) :-
    ui_patterns_extended:compile_list_pattern(infinite, items, '/api/items', swiftui, [], Code),
    sub_string(Code, _, _, _, "List"),
    sub_string(Code, _, _, _, "refreshable").

test(selectable_list_compiles_to_react_native) :-
    ui_patterns_extended:compile_list_pattern(selectable, select, single, react_native, [], Code),
    sub_string(Code, _, _, _, "toggleSelection").

test(selectable_list_compiles_to_vue) :-
    ui_patterns_extended:compile_list_pattern(selectable, select, single, vue, [], Code),
    sub_string(Code, _, _, _, "toggleSelection").

:- end_tests(list_compilation).

% ============================================================================
% Tests: Modal Compilation
% ============================================================================

:- begin_tests(modal_compilation).

test(alert_modal_compiles_to_react_native) :-
    ui_patterns_extended:compile_modal_pattern(alert, test_alert, [title('Test'), message('Message')], react_native, [], Code),
    sub_string(Code, _, _, _, "Alert.alert").

test(confirm_modal_compiles_to_react_native) :-
    ui_patterns_extended:compile_modal_pattern(confirm, test_confirm, [
        title('Confirm'),
        message('Sure?'),
        confirm_text('Yes'),
        cancel_text('No')
    ], react_native, [], Code),
    sub_string(Code, _, _, _, "Alert.alert"),
    sub_string(Code, _, _, _, "onConfirm").

test(bottom_sheet_compiles_to_react_native) :-
    ui_patterns_extended:compile_modal_pattern(bottom_sheet, test_sheet, [content('Content'), height(auto)], react_native, [], Code),
    sub_string(Code, _, _, _, "BottomSheet"),
    sub_string(Code, _, _, _, "snapPoints").

test(modal_compiles_to_vue) :-
    ui_patterns_extended:compile_modal_pattern(alert, test_modal, [title('Test'), message('Msg')], vue, [], Code),
    sub_string(Code, _, _, _, "Teleport"),
    sub_string(Code, _, _, _, "isOpen").

:- end_tests(modal_compilation).

% ============================================================================
% Tests: Auth Compilation
% ============================================================================

:- begin_tests(auth_compilation).

test(login_compiles_to_react_native) :-
    ui_patterns_extended:compile_auth_pattern(login, [endpoint('/api/login')], react_native, [], Code),
    sub_string(Code, _, _, _, "LoginForm"),
    sub_string(Code, _, _, _, "useMutation"),
    sub_string(Code, _, _, _, "/api/login").

test(login_compiles_to_vue) :-
    ui_patterns_extended:compile_auth_pattern(login, [endpoint('/api/login')], vue, [], Code),
    sub_string(Code, _, _, _, "useMutation"),
    sub_string(Code, _, _, _, "/api/login").

test(login_has_email_field) :-
    ui_patterns_extended:compile_auth_pattern(login, [endpoint('/api/login')], react_native, [], Code),
    sub_string(Code, _, _, _, "email").

test(login_has_password_field) :-
    ui_patterns_extended:compile_auth_pattern(login, [endpoint('/api/login')], react_native, [], Code),
    sub_string(Code, _, _, _, "password").

:- end_tests(auth_compilation).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
