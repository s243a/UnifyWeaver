% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_task_manager_app.pl - plunit tests for task manager integration example
%
% Run with: swipl -g "run_tests" -t halt test_task_manager_app.pl

:- module(test_task_manager_app, []).

:- use_module(library(plunit)).
:- use_module('task_manager_app').

% Import other modules for testing
:- use_module('../../../unifyweaver/theming/theming').
:- use_module('../../../unifyweaver/i18n/i18n').
:- use_module('../../../unifyweaver/a11y/accessibility').
:- use_module('../../../unifyweaver/components/component_library').

% ============================================================================
% Setup
% ============================================================================

setup_app :-
    task_manager_app:define_task_manager_app.

% ============================================================================
% Tests: App Setup
% ============================================================================

:- begin_tests(app_setup, [setup(setup_app)]).

test(app_defines_successfully) :-
    task_manager_app:define_task_manager_app.

test(app_spec_exists) :-
    task_manager_app:get_app_spec(Spec),
    Spec = app(task_manager, _).

test(app_spec_has_navigation) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(navigation(tab, _, _), Config).

test(app_spec_has_theme) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(theme(task_manager), Config).

test(app_spec_has_locales) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(locales([en, es]), Config).

:- end_tests(app_setup).

% ============================================================================
% Tests: Theme Integration
% ============================================================================

:- begin_tests(theme_integration, [setup(setup_app)]).

test(theme_exists) :-
    theming:get_theme(task_manager, Theme),
    is_list(Theme).

test(theme_has_colors) :-
    theming:get_theme(task_manager, Theme),
    member(colors(Colors), Theme),
    member(primary-_, Colors).

test(theme_has_typography) :-
    theming:get_theme(task_manager, Theme),
    member(typography(Typo), Theme),
    member(fontFamily-'Inter', Typo).

test(theme_has_spacing) :-
    theming:get_theme(task_manager, Theme),
    member(spacing(Spaces), Theme),
    member(md-16, Spaces).

test(dark_variant_exists) :-
    theming:get_variant(task_manager, dark, _).

test(dark_variant_overrides_background) :-
    theming:get_variant(task_manager, dark, Theme),
    member(colors(Colors), Theme),
    member(background-'#111827', Colors).

test(theme_generates_rn_code) :-
    theming:generate_theme_code(task_manager, react_native, Code),
    sub_string(Code, _, _, _, "export const theme").

test(theme_generates_vue_code) :-
    theming:generate_theme_code(task_manager, vue, Code),
    sub_string(Code, _, _, _, ":root").

test(theme_generates_flutter_code) :-
    theming:generate_theme_code(task_manager, flutter, Code),
    sub_string(Code, _, _, _, "class AppTheme").

test(theme_generates_swift_code) :-
    theming:generate_theme_code(task_manager, swiftui, Code),
    sub_string(Code, _, _, _, "struct Theme").

:- end_tests(theme_integration).

% ============================================================================
% Tests: i18n Integration
% ============================================================================

:- begin_tests(i18n_integration, [setup(setup_app)]).

test(en_translations_exist) :-
    i18n:resolve_translation('nav.tasks', en, [], Text),
    Text = "Tasks".

test(es_translations_exist) :-
    i18n:resolve_translation('nav.tasks', es, [], Text),
    Text = "Tareas".

test(task_title_en) :-
    i18n:resolve_translation('tasks.title', en, [], Text),
    Text = "My Tasks".

test(task_title_es) :-
    i18n:resolve_translation('tasks.title', es, [], Text),
    Text = "Mis Tareas".

test(pluralization_zero) :-
    i18n:resolve_plural('tasks.count', 0, en, Text),
    Text = "No tasks".

test(pluralization_one) :-
    i18n:resolve_plural('tasks.count', 1, en, Text),
    Text = "1 task".

test(pluralization_many) :-
    i18n:resolve_plural('tasks.count', 5, en, Text),
    Text = "5 tasks".

test(settings_translations) :-
    i18n:resolve_translation('settings.theme', en, [], Text),
    Text = "Theme".

:- end_tests(i18n_integration).

% ============================================================================
% Tests: A11y Integration
% ============================================================================

:- begin_tests(a11y_integration, [setup(setup_app)]).

test(a11y_generates_rn_code) :-
    A11y = a11y([label('Add Task'), role(button), hint('Opens form')]),
    accessibility:generate_a11y_attrs(A11y, react_native, Code),
    sub_string(Code, _, _, _, "accessibilityLabel").

test(a11y_generates_vue_code) :-
    A11y = a11y([label('Add Task'), role(button)]),
    accessibility:generate_a11y_attrs(A11y, vue, Code),
    sub_string(Code, _, _, _, "aria-label").

test(a11y_generates_flutter_code) :-
    A11y = a11y([label('Add Task'), role(button)]),
    accessibility:generate_a11y_attrs(A11y, flutter, Code),
    sub_string(Code, _, _, _, "Semantics").

test(a11y_generates_swift_code) :-
    A11y = a11y([label('Add Task'), role(button)]),
    accessibility:generate_a11y_attrs(A11y, swiftui, Code),
    sub_string(Code, _, _, _, ".accessibilityLabel").

:- end_tests(a11y_integration).

% ============================================================================
% Tests: Component Integration
% ============================================================================

:- begin_tests(component_integration, [setup(setup_app)]).

test(toast_component) :-
    component_library:toast('Task saved', [type(success)], Spec),
    Spec = toast_spec(_).

test(toast_generates_rn) :-
    component_library:toast('Saved', [type(success)], Spec),
    component_library:generate_component(Spec, react_native, Code),
    sub_string(Code, _, _, _, "Toast").

test(card_component) :-
    component_library:card(content, [title('Task')], Spec),
    Spec = card_spec(_).

test(badge_component) :-
    component_library:badge('3', [color(primary)], Spec),
    Spec = badge_spec(_).

test(spinner_component) :-
    component_library:spinner([size(large)], Spec),
    component_library:generate_component(Spec, flutter, Code),
    sub_string(Code, _, _, _, "CircularProgressIndicator").

:- end_tests(component_integration).

% ============================================================================
% Tests: Code Generation
% ============================================================================

:- begin_tests(code_generation, [setup(setup_app)]).

test(generate_rn_app) :-
    task_manager_app:generate_app(react_native, Code),
    sub_string(Code, _, _, _, "Tab.Navigator").

test(generate_vue_app) :-
    task_manager_app:generate_app(vue, Code),
    sub_string(Code, _, _, _, "createRouter").

test(generate_flutter_app) :-
    task_manager_app:generate_app(flutter, Code),
    sub_string(Code, _, _, _, "BottomNavigationBar").

test(generate_swift_app) :-
    task_manager_app:generate_app(swiftui, Code),
    sub_string(Code, _, _, _, "TabView").

test(full_app_has_theme) :-
    task_manager_app:generate_full_app(react_native, Files),
    member(file(theme, _), Files).

test(full_app_has_navigation) :-
    task_manager_app:generate_full_app(react_native, Files),
    member(file(navigation, _), Files).

test(full_app_has_locales) :-
    task_manager_app:generate_full_app(react_native, Files),
    member(locales(LocaleFiles), Files),
    length(LocaleFiles, 2).

test(full_app_has_name) :-
    task_manager_app:generate_full_app(react_native, Files),
    member(app_name(task_manager), Files).

:- end_tests(code_generation).

% ============================================================================
% Tests: App Screens
% ============================================================================

:- begin_tests(app_screens, [setup(setup_app)]).

test(has_tasks_screen) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(navigation(tab, Screens, _), Config),
    member(screen(tasks, 'TasksScreen', _), Screens).

test(has_calendar_screen) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(navigation(tab, Screens, _), Config),
    member(screen(calendar, 'CalendarScreen', _), Screens).

test(has_settings_screen) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(navigation(tab, Screens, _), Config),
    member(screen(settings, 'SettingsScreen', _), Screens).

test(tasks_screen_has_a11y) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(navigation(tab, Screens, _), Config),
    member(screen(tasks, _, Opts), Screens),
    member(a11y(_), Opts).

test(has_stack_screens) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(stack_screens(StackScreens), Config),
    length(StackScreens, 3).

:- end_tests(app_screens).

% ============================================================================
% Tests: App Patterns
% ============================================================================

:- begin_tests(app_patterns, [setup(setup_app)]).

test(has_task_list_item_pattern) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(patterns(Patterns), Config),
    member(pattern(task_list_item, _), Patterns).

test(has_task_form_pattern) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(patterns(Patterns), Config),
    member(pattern(task_form, _), Patterns).

test(has_task_detail_pattern) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(patterns(Patterns), Config),
    member(pattern(task_detail_view, _), Patterns).

test(has_settings_list_pattern) :-
    task_manager_app:get_app_spec(app(_, Config)),
    member(patterns(Patterns), Config),
    member(pattern(settings_list, _), Patterns).

:- end_tests(app_patterns).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
