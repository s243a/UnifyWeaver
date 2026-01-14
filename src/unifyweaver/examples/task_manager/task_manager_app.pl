% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% task_manager_app.pl - Full integration example
%
% Demonstrates all UnifyWeaver features working together:
% - UI patterns (navigation, screens, forms)
% - Pattern composition (refs, slots, templates)
% - i18n/localization
% - Accessibility (a11y)
% - Theming
% - Component library
% - Multi-target code generation
%
% Run with: swipl -g "demo_task_manager" -t halt task_manager_app.pl

:- module(task_manager_app, [
    % App definition
    define_task_manager_app/0,
    get_app_spec/1,

    % Theme
    get_app_theme/1,
    get_dark_theme/1,

    % Translations
    setup_translations/0,

    % Code generation
    generate_app/2,
    generate_full_app/2,

    % Demo
    demo_task_manager/0
]).

% Load all UnifyWeaver modules
:- use_module('../../../unifyweaver/patterns/ui_patterns').
:- use_module('../../../unifyweaver/patterns/pattern_composition').
:- use_module('../../../unifyweaver/i18n/i18n').
:- use_module('../../../unifyweaver/a11y/accessibility').
:- use_module('../../../unifyweaver/theming/theming').
:- use_module('../../../unifyweaver/components/component_library').
:- use_module(library(lists)).

% ============================================================================
% Theme Definition
% ============================================================================

%! get_app_theme(-Theme) is det
%  Get the app theme definition.
get_app_theme(Theme) :-
    Theme = [
        colors([
            primary-'#5B4CDB',
            primaryLight-'#7B6FE8',
            primaryDark-'#4338CA',
            secondary-'#10B981',
            background-'#F9FAFB',
            surface-'#FFFFFF',
            text-'#111827',
            textSecondary-'#6B7280',
            error-'#EF4444',
            warning-'#F59E0B',
            success-'#10B981',
            border-'#E5E7EB'
        ]),
        typography([
            fontFamily-'Inter',
            sizeXs-12,
            sizeSm-14,
            sizeMd-16,
            sizeLg-20,
            sizeXl-24,
            sizeXxl-32,
            weightNormal-400,
            weightMedium-500,
            weightBold-700
        ]),
        spacing([
            xs-4,
            sm-8,
            md-16,
            lg-24,
            xl-32,
            xxl-48
        ]),
        borders([
            radiusSm-4,
            radiusMd-8,
            radiusLg-12,
            radiusXl-16,
            radiusFull-9999
        ]),
        shadows([
            sm-'0 1px 2px rgba(0,0,0,0.05)',
            md-'0 4px 6px rgba(0,0,0,0.1)',
            lg-'0 10px 15px rgba(0,0,0,0.1)'
        ])
    ].

%! get_dark_theme(-Theme) is det
%  Get dark mode theme overrides.
get_dark_theme(Theme) :-
    Theme = [
        colors([
            background-'#111827',
            surface-'#1F2937',
            text-'#F9FAFB',
            textSecondary-'#9CA3AF',
            border-'#374151'
        ])
    ].

setup_theme :-
    theming:clear_themes,
    get_app_theme(Theme),
    theming:define_theme(task_manager, Theme),
    get_dark_theme(DarkOverrides),
    theming:define_variant(task_manager, dark, DarkOverrides).

% ============================================================================
% i18n Translations
% ============================================================================

%! setup_translations is det
%  Define all translations for the app.
setup_translations :-
    i18n:clear_translations,

    % English translations
    i18n:define_translations(en, [
        % Navigation
        'nav.tasks' - "Tasks",
        'nav.calendar' - "Calendar",
        'nav.settings' - "Settings",

        % Task list screen
        'tasks.title' - "My Tasks",
        'tasks.empty' - "No tasks yet",
        'tasks.add' - "Add Task",
        'tasks.completed' - "Completed",
        'tasks.pending' - "Pending",
        'tasks.count.zero' - "No tasks",
        'tasks.count.one' - "1 task",
        'tasks.count.other' - "{{count}} tasks",

        % Task form
        'task.title' - "Task Title",
        'task.title.placeholder' - "Enter task title",
        'task.description' - "Description",
        'task.description.placeholder' - "Add details...",
        'task.dueDate' - "Due Date",
        'task.priority' - "Priority",
        'task.priority.low' - "Low",
        'task.priority.medium' - "Medium",
        'task.priority.high' - "High",
        'task.save' - "Save Task",
        'task.cancel' - "Cancel",
        'task.delete' - "Delete Task",
        'task.edit' - "Edit Task",

        % Calendar
        'calendar.title' - "Calendar",
        'calendar.today' - "Today",
        'calendar.noTasks' - "No tasks for this day",

        % Settings
        'settings.title' - "Settings",
        'settings.theme' - "Theme",
        'settings.theme.light' - "Light",
        'settings.theme.dark' - "Dark",
        'settings.theme.system' - "System",
        'settings.language' - "Language",
        'settings.notifications' - "Notifications",
        'settings.notifications.enabled' - "Enable notifications",
        'settings.about' - "About",
        'settings.version' - "Version",

        % Common
        'common.save' - "Save",
        'common.cancel' - "Cancel",
        'common.delete' - "Delete",
        'common.edit' - "Edit",
        'common.confirm' - "Confirm",
        'common.loading' - "Loading..."
    ]),

    % Spanish translations
    i18n:define_translations(es, [
        'nav.tasks' - "Tareas",
        'nav.calendar' - "Calendario",
        'nav.settings' - "Ajustes",

        'tasks.title' - "Mis Tareas",
        'tasks.empty' - "Sin tareas",
        'tasks.add' - "Agregar Tarea",
        'tasks.completed' - "Completadas",
        'tasks.pending' - "Pendientes",
        'tasks.count.zero' - "Sin tareas",
        'tasks.count.one' - "1 tarea",
        'tasks.count.other' - "{{count}} tareas",

        'task.title' - "Titulo",
        'task.save' - "Guardar",
        'task.cancel' - "Cancelar",
        'task.delete' - "Eliminar",

        'settings.title' - "Ajustes",
        'settings.theme' - "Tema",
        'settings.language' - "Idioma",

        'common.save' - "Guardar",
        'common.cancel' - "Cancelar",
        'common.delete' - "Eliminar"
    ]).

% ============================================================================
% Pattern Templates
% ============================================================================

setup_templates :-
    % CRUD screen template
    pattern_composition:define_template(crud_list_screen, [entity, icon],
        screen(list, '~wListScreen', [
            title(t('~w.title')),
            icon('~w'),
            content([
                slot(header),
                slot(list),
                slot(fab)
            ])
        ])
    ),

    % Detail screen template
    pattern_composition:define_template(detail_screen, [entity],
        screen(detail, '~wDetailScreen', [
            content([
                slot(header),
                slot(content),
                slot(actions)
            ])
        ])
    ),

    % Form template
    pattern_composition:define_template(entity_form, [entity],
        form_pattern('~wForm', [
            slot(fields),
            slot(actions)
        ], [])
    ).

% ============================================================================
% App Specification
% ============================================================================

%! define_task_manager_app is det
%  Define the complete task manager app.
define_task_manager_app :-
    setup_theme,
    setup_translations,
    setup_templates.

%! get_app_spec(-Spec) is det
%  Get the complete app specification.
get_app_spec(Spec) :-
    Spec = app(task_manager, [
        theme(task_manager),
        locales([en, es]),
        defaultLocale(en),

        navigation(tab, [
            % Tasks tab
            screen(tasks, 'TasksScreen', [
                title(t('nav.tasks')),
                icon('check-square'),
                a11y([
                    label(t('nav.tasks')),
                    role(tab)
                ]),
                content([
                    % Header with task count
                    header([
                        text(t('tasks.title')),
                        badge(taskCount, [color(primary)])
                    ]),

                    % Task list
                    list(tasks, [
                        ref(task_list_item),
                        emptyState(t('tasks.empty'))
                    ]),

                    % Add task FAB
                    fab(add, [
                        icon('plus'),
                        onPress(navigateToAddTask),
                        a11y([
                            label(t('tasks.add')),
                            role(button),
                            hint('Opens task creation form')
                        ])
                    ])
                ])
            ]),

            % Calendar tab
            screen(calendar, 'CalendarScreen', [
                title(t('nav.calendar')),
                icon('calendar'),
                a11y([
                    label(t('nav.calendar')),
                    role(tab)
                ]),
                content([
                    calendar_view([
                        onDateSelect(showTasksForDate),
                        markedDates(taskDates)
                    ]),
                    slot(dayTasks)
                ])
            ]),

            % Settings tab
            screen(settings, 'SettingsScreen', [
                title(t('nav.settings')),
                icon('settings'),
                a11y([
                    label(t('nav.settings')),
                    role(tab)
                ]),
                content([
                    ref(settings_list)
                ])
            ])
        ], []),

        % Stack screens (modals/detail views)
        stack_screens([
            screen(taskDetail, 'TaskDetailScreen', [
                content([
                    ref(task_detail_view)
                ])
            ]),

            screen(addTask, 'AddTaskScreen', [
                title(t('tasks.add')),
                content([
                    ref(task_form)
                ])
            ]),

            screen(editTask, 'EditTaskScreen', [
                title(t('task.edit')),
                content([
                    ref(task_form)
                ])
            ])
        ]),

        % Pattern definitions
        patterns([
            % Task list item pattern
            pattern(task_list_item,
                list_item(task, [
                    leading(checkbox(completed, [
                        a11y([
                            label('Mark task complete'),
                            role(checkbox)
                        ])
                    ])),
                    content([
                        text(title, [style(taskTitle)]),
                        text(dueDate, [style(taskDueDate)])
                    ]),
                    trailing(priority_badge(priority)),
                    onPress(navigateToDetail),
                    a11y([
                        label(title),
                        hint('Tap to view details')
                    ])
                ])
            ),

            % Task form pattern
            pattern(task_form,
                form_pattern(taskForm, [
                    field(title, text, [required], [
                        label(t('task.title')),
                        placeholder(t('task.title.placeholder')),
                        a11y([
                            label(t('task.title')),
                            required(true)
                        ])
                    ]),
                    field(description, textarea, [], [
                        label(t('task.description')),
                        placeholder(t('task.description.placeholder')),
                        a11y([label(t('task.description'))])
                    ]),
                    field(dueDate, date, [], [
                        label(t('task.dueDate')),
                        a11y([label(t('task.dueDate'))])
                    ]),
                    field(priority, select, [], [
                        label(t('task.priority')),
                        options([
                            option(low, t('task.priority.low')),
                            option(medium, t('task.priority.medium')),
                            option(high, t('task.priority.high'))
                        ]),
                        a11y([label(t('task.priority'))])
                    ])
                ], [
                    onSubmit(saveTask),
                    validation(onChange)
                ])
            ),

            % Task detail view pattern
            pattern(task_detail_view,
                detail_view([
                    header([
                        text(title, [style(detailTitle)]),
                        priority_badge(priority)
                    ]),
                    section(details, [
                        row(t('task.dueDate'), dueDate),
                        row(t('task.priority'), priority)
                    ]),
                    section(description, [
                        text(description)
                    ]),
                    actions([
                        button(edit, [
                            text(t('task.edit')),
                            onPress(navigateToEdit),
                            a11y([label(t('task.edit')), role(button)])
                        ]),
                        button(delete, [
                            text(t('task.delete')),
                            variant(danger),
                            onPress(confirmDelete),
                            a11y([label(t('task.delete')), role(button)])
                        ])
                    ])
                ])
            ),

            % Settings list pattern
            pattern(settings_list,
                list(settings, [
                    section(t('settings.theme'), [
                        setting_item(theme, select, [
                            options([light, dark, system]),
                            value(currentTheme),
                            onChange(setTheme),
                            a11y([label(t('settings.theme'))])
                        ])
                    ]),
                    section(t('settings.language'), [
                        setting_item(language, select, [
                            options([en, es]),
                            value(currentLocale),
                            onChange(setLocale),
                            a11y([label(t('settings.language'))])
                        ])
                    ]),
                    section(t('settings.notifications'), [
                        setting_item(notifications, switch, [
                            value(notificationsEnabled),
                            onChange(toggleNotifications),
                            a11y([
                                label(t('settings.notifications.enabled')),
                                role(switch)
                            ])
                        ])
                    ])
                ])
            )
        ]),

        % Components used
        components([
            component(priority_badge, [
                props([priority]),
                render(badge(priority, [
                    color(priorityColor(priority)),
                    variant(filled)
                ]))
            ]),

            component(calendar_view, [
                props([onDateSelect, markedDates]),
                render(calendar([
                    onDayPress(onDateSelect),
                    markedDates(markedDates)
                ]))
            ])
        ])
    ]).

% ============================================================================
% Code Generation
% ============================================================================

%! generate_app(+Target, -Code) is det
%  Generate app code for a target.
generate_app(Target, Code) :-
    get_app_spec(Spec),
    generate_app_code(Spec, Target, Code).

generate_app_code(app(Name, Config), Target, Code) :-
    % Generate theme
    member(theme(ThemeName), Config),
    theming:generate_theme_code(ThemeName, Target, ThemeCode),

    % Generate navigation
    member(navigation(Type, Screens, _), Config),
    generate_navigation_code(navigation(Type, Screens, []), Target, NavCode),

    % Combine
    format(atom(Code), '// ~w App - Generated for ~w\n\n// Theme\n~w\n\n// Navigation\n~w\n',
           [Name, Target, ThemeCode, NavCode]).

generate_navigation_code(navigation(tab, Screens, _), react_native, Code) :-
    findall(TabCode, (
        member(screen(Id, Component, _), Screens),
        format(atom(TabCode), '      <Tab.Screen name="~w" component={~w} />', [Id, Component])
    ), TabCodes),
    atomic_list_concat(TabCodes, '\n', TabsStr),
    format(atom(Code), 'const AppNavigator = () => (\n  <NavigationContainer>\n    <Tab.Navigator>\n~w\n    </Tab.Navigator>\n  </NavigationContainer>\n);', [TabsStr]).

generate_navigation_code(navigation(tab, Screens, _), vue, Code) :-
    findall(RouteCode, (
        member(screen(Id, Component, _), Screens),
        format(atom(RouteCode), '    { path: \'/~w\', component: ~w }', [Id, Component])
    ), RouteCodes),
    atomic_list_concat(RouteCodes, ',\n', RoutesStr),
    format(atom(Code), 'const routes = [\n~w\n];\n\nconst router = createRouter({\n  history: createWebHistory(),\n  routes\n});', [RoutesStr]).

generate_navigation_code(navigation(tab, Screens, _), flutter, Code) :-
    findall(TabCode, (
        member(screen(Id, Component, Opts), Screens),
        member(icon(Icon), Opts),
        format(atom(TabCode), '        BottomNavigationBarItem(\n          icon: Icon(Icons.~w),\n          label: \'~w\',\n        )', [Icon, Id])
    ), TabCodes),
    atomic_list_concat(TabCodes, ',\n', TabsStr),
    format(atom(Code), 'class AppNavigator extends StatefulWidget {\n  @override\n  State<AppNavigator> createState() => _AppNavigatorState();\n}\n\nclass _AppNavigatorState extends State<AppNavigator> {\n  int _selectedIndex = 0;\n\n  @override\n  Widget build(BuildContext context) {\n    return Scaffold(\n      body: _screens[_selectedIndex],\n      bottomNavigationBar: BottomNavigationBar(\n        currentIndex: _selectedIndex,\n        onTap: (index) => setState(() => _selectedIndex = index),\n        items: [\n~w\n        ],\n      ),\n    );\n  }\n}', [TabsStr]).

generate_navigation_code(navigation(tab, Screens, _), swiftui, Code) :-
    findall(TabCode, (
        member(screen(Id, _, Opts), Screens),
        member(icon(Icon), Opts),
        format(atom(TabCode), '            ~wView()\n                .tabItem {\n                    Image(systemName: \"~w\")\n                    Text(\"~w\")\n                }', [Id, Icon, Id])
    ), TabCodes),
    atomic_list_concat(TabCodes, '\n', TabsStr),
    format(atom(Code), 'struct AppNavigator: View {\n    var body: some View {\n        TabView {\n~w\n        }\n    }\n}', [TabsStr]).

%! generate_full_app(+Target, -Files) is det
%  Generate all files for a complete app.
generate_full_app(Target, Files) :-
    get_app_spec(Spec),
    Spec = app(Name, Config),

    % Theme file
    member(theme(ThemeName), Config),
    theming:generate_theme_code(ThemeName, Target, ThemeCode),

    % i18n files
    member(locales(Locales), Config),
    findall(locale_file(Locale, JSON), (
        member(Locale, Locales),
        i18n:generate_translation_json(Locale, JSON)
    ), LocaleFiles),

    % Navigation
    member(navigation(Type, Screens, NavOpts), Config),
    generate_navigation_code(navigation(Type, Screens, NavOpts), Target, NavCode),

    Files = [
        file(theme, ThemeCode),
        file(navigation, NavCode),
        locales(LocaleFiles),
        app_name(Name)
    ].

% ============================================================================
% Demo
% ============================================================================

%! demo_task_manager is det
%  Run a demo showing all features.
demo_task_manager :-
    format('~n=== Task Manager App Demo ===~n~n'),

    % Setup
    format('1. Setting up app...~n'),
    define_task_manager_app,
    format('   - Theme defined~n'),
    format('   - Translations loaded (en, es)~n'),
    format('   - Templates registered~n~n'),

    % Show theme
    format('2. Theme Generation:~n'),
    theming:generate_theme_code(task_manager, react_native, RNTheme),
    format('   React Native theme:~n'),
    sub_atom(RNTheme, 0, 200, _, RNThemePreview),
    format('   ~w...~n~n', [RNThemePreview]),

    % Show i18n
    format('3. i18n Resolution:~n'),
    i18n:resolve_translation('tasks.title', en, [], EnTitle),
    i18n:resolve_translation('tasks.title', es, [], EsTitle),
    format('   tasks.title (en): ~w~n', [EnTitle]),
    format('   tasks.title (es): ~w~n~n', [EsTitle]),

    % Show pluralization
    format('4. Pluralization:~n'),
    i18n:resolve_plural('tasks.count', 0, en, Zero),
    i18n:resolve_plural('tasks.count', 1, en, One),
    i18n:resolve_plural('tasks.count', 5, en, Five),
    format('   0 tasks: ~w~n', [Zero]),
    format('   1 task: ~w~n', [One]),
    format('   5 tasks: ~w~n~n', [Five]),

    % Show a11y
    format('5. Accessibility Code:~n'),
    A11y = a11y([label('Add Task'), role(button), hint('Opens task form')]),
    accessibility:generate_a11y_attrs(A11y, react_native, RNa11y),
    format('   React Native:~n   ~w~n~n', [RNa11y]),

    % Show component
    format('6. Component Generation:~n'),
    component_library:toast('Task saved!', [type(success)], ToastSpec),
    component_library:generate_component(ToastSpec, flutter, FlutterToast),
    format('   Flutter toast:~n   ~w~n~n', [FlutterToast]),

    % Generate app
    format('7. Full App Generation:~n'),
    generate_app(react_native, AppCode),
    format('   Generated ~w characters of React Native code~n~n', [AppCode]),

    format('=== Demo Complete ===~n'),
    !.

% ============================================================================
% Inline Tests
% ============================================================================

test_task_manager :-
    format('Running task_manager tests...~n'),

    % Test 1: App setup
    define_task_manager_app,
    format('  Test 1 passed: app setup~n'),

    % Test 2: Theme exists
    theming:get_theme(task_manager, Theme),
    member(colors(_), Theme),
    format('  Test 2 passed: theme exists~n'),

    % Test 3: Translations exist
    i18n:resolve_translation('nav.tasks', en, [], TasksText),
    TasksText = "Tasks",
    format('  Test 3 passed: translations exist~n'),

    % Test 4: Dark variant
    theming:get_variant(task_manager, dark, DarkTheme),
    member(colors(DarkColors), DarkTheme),
    member(background-'#111827', DarkColors),
    format('  Test 4 passed: dark variant~n'),

    % Test 5: App spec
    get_app_spec(Spec),
    Spec = app(task_manager, _),
    format('  Test 5 passed: app spec~n'),

    % Test 6: Generate RN
    generate_app(react_native, RNCode),
    sub_string(RNCode, _, _, _, "Tab.Navigator"),
    format('  Test 6 passed: React Native generation~n'),

    % Test 7: Generate Vue
    generate_app(vue, VueCode),
    sub_string(VueCode, _, _, _, "createRouter"),
    format('  Test 7 passed: Vue generation~n'),

    % Test 8: Generate Flutter
    generate_app(flutter, FlutterCode),
    sub_string(FlutterCode, _, _, _, "BottomNavigationBar"),
    format('  Test 8 passed: Flutter generation~n'),

    % Test 9: Generate SwiftUI
    generate_app(swiftui, SwiftCode),
    sub_string(SwiftCode, _, _, _, "TabView"),
    format('  Test 9 passed: SwiftUI generation~n'),

    % Test 10: Full app generation
    generate_full_app(react_native, Files),
    member(file(theme, _), Files),
    member(file(navigation, _), Files),
    format('  Test 10 passed: full app generation~n'),

    format('All 10 task_manager tests passed!~n'),
    !.

:- initialization(test_task_manager, main).
