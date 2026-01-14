% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% app_generator.pl - Complete Application Generator
%
% Coordinates all UnifyWeaver modules to generate complete, runnable
% full-stack applications from declarative app specifications.
%
% Integrates:
%   - UI Patterns (navigation, screens)
%   - Theming (colors, typography)
%   - i18n (translations, locales)
%   - Layout (flexbox primitives)
%   - Data Binding (state, stores, computed)
%   - Accessibility (a11y attributes)
%   - Data Integration (backend data sources)
%
% Usage:
%   ?- generate_complete_project(app(myapp, [...]), [frontend-react_native, backend-fastapi], '/output', Result).

:- module(app_generator, [
    % Main generation entry point
    generate_complete_project/4,        % +AppSpec, +Targets, +OutputDir, -Result

    % Frontend generation
    generate_frontend_project/4,        % +AppSpec, +Target, +OutputDir, -Files

    % Entry point generation
    generate_entry_point/3,             % +AppSpec, +Target, -Code
    generate_react_native_entry/2,      % +AppSpec, -Code
    generate_vue_entry/2,               % +AppSpec, -Code
    generate_flutter_entry/2,           % +AppSpec, -Code
    generate_swiftui_entry/2,           % +AppSpec, -Code

    % Theme generation
    generate_theme_files/3,             % +AppSpec, +Target, -Files

    % i18n generation
    generate_locale_files/3,            % +AppSpec, +Target, -Files

    % Screen generation
    generate_screen_files/3,            % +AppSpec, +Target, -Files

    % Navigation generation
    generate_navigation_file/3,         % +AppSpec, +Target, -Code

    % API client generation
    generate_api_client/3,              % +AppSpec, +Target, -Code

    % App spec accessors
    app_name/2,                         % +AppSpec, -Name
    app_theme/2,                        % +AppSpec, -Theme
    app_locales/2,                      % +AppSpec, -Locales
    app_navigation/2,                   % +AppSpec, -Navigation
    app_screens/2,                      % +AppSpec, -Screens
    app_data/2,                         % +AppSpec, -Data

    % Testing
    test_app_generator/0
]).

:- use_module(library(lists)).

% Try to load dependent modules
:- catch(use_module('./project_generator'), _, true).
:- catch(use_module('./data_integration'), _, true).
:- catch(use_module('../theming/theming'), _, true).
:- catch(use_module('../i18n/i18n'), _, true).
:- catch(use_module('../layout/layout', [
    row/3, column/3, stack/3, grid/3, spacer/1,
    generate_layout/3, generate_layout_code/4
]), _, true).
:- catch(use_module('../binding/data_binding'), _, true).
:- catch(use_module('../a11y/accessibility'), _, true).
:- catch(use_module('../patterns/ui_patterns'), _, true).

% ============================================================================
% APP SPEC ACCESSORS
% ============================================================================

app_name(app(Name, _), Name).

app_theme(app(_, Config), Theme) :-
    (member(theme(Theme), Config) -> true ; Theme = []).

app_locales(app(_, Config), Locales) :-
    (member(locales(Locales), Config) -> true ; Locales = [en]).

app_navigation(app(_, Config), Nav) :-
    (member(navigation(Type, Screens, Opts), Config)
    -> Nav = navigation(Type, Screens, Opts)
    ; Nav = none).

app_screens(app(_, Config), Screens) :-
    (member(screens(Screens), Config) -> true ; Screens = []).

app_data(app(_, Config), Data) :-
    (member(data(Data), Config) -> true ; Data = []).

app_translations(app(_, Config), Trans) :-
    (member(translations(Trans), Config) -> true ; Trans = []).

% ============================================================================
% MAIN GENERATION ENTRY POINT
% ============================================================================

%! generate_complete_project(+AppSpec, +Targets, +OutputDir, -Result)
%
%  Generate a complete full-stack project.
%
%  Targets: [frontend-Target, backend-Target]
%
generate_complete_project(AppSpec, Targets, OutputDir, Result) :-
    app_name(AppSpec, Name),
    format('Generating complete project: ~w~n', [Name]),

    % Extract targets
    (member(frontend-FrontendTarget, Targets) -> true ; FrontendTarget = react_native),
    (member(backend-BackendTarget, Targets) -> true ; BackendTarget = fastapi),

    % Generate frontend
    atom_concat(OutputDir, '/frontend', FrontendDir),
    generate_frontend_project(AppSpec, FrontendTarget, FrontendDir, FrontendFiles),

    % Generate backend
    atom_concat(OutputDir, '/backend', BackendDir),
    generate_backend_with_data(AppSpec, BackendTarget, BackendDir, BackendFiles),

    Result = complete_project(Name, FrontendFiles, BackendFiles),
    format('Project generation complete!~n', []).

% ============================================================================
% FRONTEND PROJECT GENERATION
% ============================================================================

%! generate_frontend_project(+AppSpec, +Target, +OutputDir, -Files)
%
%  Generate complete frontend project with all files.
%
generate_frontend_project(AppSpec, Target, OutputDir, Files) :-
    app_name(AppSpec, Name),
    format('  Generating ~w frontend...~n', [Target]),

    % Create directory structure
    project_generator:create_directory_structure(Target, OutputDir, _Dirs),

    % Generate all files
    generate_entry_point(AppSpec, Target, EntryCode),
    generate_theme_files(AppSpec, Target, ThemeFiles),
    generate_locale_files(AppSpec, Target, LocaleFiles),
    generate_navigation_file(AppSpec, Target, NavCode),
    generate_screen_files(AppSpec, Target, ScreenFiles),
    generate_api_client(AppSpec, Target, ApiCode),

    % Write entry point
    entry_point_path(Target, EntryPath),
    atom_concat(OutputDir, EntryPath, _FullEntryPath),

    % Write config files
    project_generator:generate_package_json(Name, Target, ConfigContent),
    config_file_path(Target, ConfigPath),
    atom_concat(OutputDir, ConfigPath, _FullConfigPath),

    % Generate target-specific files
    (Target = vue ->
        generate_vue_specific_files(AppSpec, VueFiles)
    ;
        VueFiles = []
    ),

    % Collect all files
    append([
        [file(EntryPath, EntryCode), file(ConfigPath, ConfigContent)],
        [file('/src/navigation/index.ts', NavCode)],
        [file('/src/api/client.ts', ApiCode)],
        ThemeFiles,
        LocaleFiles,
        ScreenFiles,
        VueFiles
    ], AllFiles),

    % Write all files
    write_all_files(AllFiles, OutputDir),

    Files = frontend_files(Target, AllFiles).

% ============================================================================
% ENTRY POINT GENERATION
% ============================================================================

%! generate_entry_point(+AppSpec, +Target, -Code)
generate_entry_point(AppSpec, react_native, Code) :-
    generate_react_native_entry(AppSpec, Code).
generate_entry_point(AppSpec, vue, Code) :-
    generate_vue_entry(AppSpec, Code).
generate_entry_point(AppSpec, flutter, Code) :-
    generate_flutter_entry(AppSpec, Code).
generate_entry_point(AppSpec, swiftui, Code) :-
    generate_swiftui_entry(AppSpec, Code).

%! generate_react_native_entry(+AppSpec, -Code)
generate_react_native_entry(AppSpec, Code) :-
    app_name(AppSpec, Name),
    app_locales(AppSpec, Locales),
    (Locales \= [] -> I18nImport = "import { I18nProvider } from './i18n';\n" ; I18nImport = ""),
    (Locales \= [] -> I18nWrap = "<I18nProvider>\n          " ; I18nWrap = ""),
    (Locales \= [] -> I18nClose = "\n          </I18nProvider>" ; I18nClose = ""),
    format(atom(Code),
"import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './theme';
~wimport { AppNavigator } from './navigation';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 2,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        ~w<NavigationContainer>
            <AppNavigator />
          </NavigationContainer>~w
      </ThemeProvider>
    </QueryClientProvider>
  );
}

// Generated by UnifyWeaver for ~w
", [I18nImport, I18nWrap, I18nClose, Name]).

%! generate_vue_entry(+AppSpec, -Code)
generate_vue_entry(AppSpec, Code) :-
    app_name(AppSpec, Name),
    format(atom(Code),
"import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { VueQueryPlugin } from '@tanstack/vue-query';
import { router } from './navigation';
import App from './App.vue';
import './styles/theme.css';

const app = createApp(App);

app.use(createPinia());
app.use(router);
app.use(VueQueryPlugin);

app.mount('#app');

// Generated by UnifyWeaver for ~w
", [Name]).

%! generate_flutter_entry(+AppSpec, -Code)
generate_flutter_entry(AppSpec, Code) :-
    app_name(AppSpec, Name),
    capitalize_first(Name, ClassName),
    format(atom(Code),
"import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'theme/app_theme.dart';
import 'router/app_router.dart';

void main() {
  runApp(
    const ProviderScope(
      child: ~wApp(),
    ),
  );
}

class ~wApp extends ConsumerWidget {
  const ~wApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = ref.watch(themeProvider);
    final router = ref.watch(routerProvider);

    return MaterialApp.router(
      title: '~w',
      theme: theme,
      routerConfig: router,
      debugShowCheckedModeBanner: false,
    );
  }
}

// Generated by UnifyWeaver for ~w
", [ClassName, ClassName, ClassName, Name, Name]).

%! generate_swiftui_entry(+AppSpec, -Code)
generate_swiftui_entry(AppSpec, Code) :-
    app_name(AppSpec, Name),
    capitalize_first(Name, ClassName),
    format(atom(Code),
"import SwiftUI

@main
struct ~wApp: App {
    @StateObject private var appStore = AppStore()
    @StateObject private var themeManager = ThemeManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appStore)
                .environmentObject(themeManager)
        }
    }
}

// Generated by UnifyWeaver for ~w
", [ClassName, Name]).

% ============================================================================
% THEME FILE GENERATION
% ============================================================================

%! generate_theme_files(+AppSpec, +Target, -Files)
generate_theme_files(AppSpec, Target, Files) :-
    app_theme(AppSpec, ThemeSpec),
    (   ThemeSpec \= []
    ->  generate_theme_code_for_target(ThemeSpec, Target, Code),
        theme_file_path(Target, Path),
        Files = [file(Path, Code)]
    ;   generate_default_theme(Target, Code),
        theme_file_path(Target, Path),
        Files = [file(Path, Code)]
    ).

generate_theme_code_for_target(ThemeSpec, react_native, Code) :-
    (member(colors(Colors), ThemeSpec), member(primary-Primary, Colors) -> true ; Primary = '#6366F1'),
    (member(colors(Colors2), ThemeSpec), member(secondary-Secondary, Colors2) -> true ; Secondary = '#10B981'),
    (member(typography(Typo), ThemeSpec), member(fontFamily-FontFamily, Typo) -> true ; FontFamily = 'System'),
    format(atom(Code),
"import { createContext, useContext, ReactNode } from 'react';

export const theme = {
  colors: {
    primary: '~w',
    secondary: '~w',
    background: '#FFFFFF',
    text: '#111827',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
  },
  typography: {
    fontFamily: '~w',
  },
};

const ThemeContext = createContext(theme);

export const ThemeProvider = ({ children }: { children: ReactNode }) => (
  <ThemeContext.Provider value={theme}>{children}</ThemeContext.Provider>
);

export const useTheme = () => useContext(ThemeContext);
", [Primary, Secondary, FontFamily]).

generate_theme_code_for_target(ThemeSpec, vue, Code) :-
    (member(colors(Colors), ThemeSpec), member(primary-Primary, Colors) -> true ; Primary = '#6366F1'),
    format(atom(Code),
":root {
  --color-primary: ~w;
  --color-secondary: #10B981;
  --color-background: #FFFFFF;
  --color-text: #111827;

  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
}
", [Primary]).

generate_theme_code_for_target(ThemeSpec, flutter, Code) :-
    (member(colors(Colors), ThemeSpec), member(primary-Primary, Colors) -> true ; Primary = '#6366F1'),
    atom_concat('#', PrimaryHex, Primary),
    format(atom(Code),
"import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final themeProvider = Provider<ThemeData>((ref) {
  return ThemeData(
    primarySwatch: Colors.indigo,
    colorScheme: ColorScheme.fromSeed(
      seedColor: Color(0x~w),
    ),
    useMaterial3: true,
  );
});
", [PrimaryHex]).

generate_theme_code_for_target(_ThemeSpec, swiftui, Code) :-
    Code = "import SwiftUI

struct Theme {
    static let primary = Color(hex: \"#6366F1\")
    static let secondary = Color(hex: \"#10B981\")
    static let background = Color.white
    static let text = Color(hex: \"#111827\")

    struct Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 16
        static let lg: CGFloat = 24
        static let xl: CGFloat = 32
    }
}

class ThemeManager: ObservableObject {
    @Published var isDarkMode = false
}
".

generate_default_theme(Target, Code) :-
    generate_theme_code_for_target([], Target, Code).

% ============================================================================
% LOCALE FILE GENERATION
% ============================================================================

%! generate_locale_files(+AppSpec, +Target, -Files)
generate_locale_files(AppSpec, Target, Files) :-
    app_locales(AppSpec, Locales),
    app_translations(AppSpec, Trans),
    findall(file(Path, Content), (
        member(Locale, Locales),
        generate_locale_content(Locale, Trans, Target, Content),
        locale_file_path(Locale, Target, Path)
    ), Files).

generate_locale_content(Locale, Translations, react_native, Content) :-
    findall(Entry, (
        member(Key-LocaleMap, Translations),
        member(Locale-Value, LocaleMap),
        format(atom(Entry), '  "~w": "~w"', [Key, Value])
    ), Entries),
    (   Entries \= []
    ->  atomic_list_concat(Entries, ',\n', EntriesStr),
        format(atom(Content), '{\n~w\n}', [EntriesStr])
    ;   Content = '{}'
    ).

generate_locale_content(Locale, Translations, vue, Content) :-
    generate_locale_content(Locale, Translations, react_native, Content).

generate_locale_content(Locale, Translations, flutter, Content) :-
    findall(Entry, (
        member(Key-LocaleMap, Translations),
        member(Locale-Value, LocaleMap),
        format(atom(Entry), '  "~w": "~w"', [Key, Value])
    ), Entries),
    (   Entries \= []
    ->  atomic_list_concat(Entries, ',\n', EntriesStr),
        format(atom(Content), '{\n  "@@locale": "~w",\n~w\n}', [Locale, EntriesStr])
    ;   format(atom(Content), '{\n  "@@locale": "~w"\n}', [Locale])
    ).

generate_locale_content(Locale, Translations, swiftui, Content) :-
    findall(Entry, (
        member(Key-LocaleMap, Translations),
        member(Locale-Value, LocaleMap),
        format(atom(Entry), '"~w" = "~w";', [Key, Value])
    ), Entries),
    (   Entries \= []
    ->  atomic_list_concat(Entries, '\n', Content)
    ;   Content = '/* No translations */'
    ).

% ============================================================================
% NAVIGATION FILE GENERATION
% ============================================================================

%! generate_navigation_file(+AppSpec, +Target, -Code)
generate_navigation_file(AppSpec, Target, Code) :-
    app_navigation(AppSpec, Nav),
    (   Nav = navigation(Type, Screens, _Opts)
    ->  generate_nav_code(Type, Screens, Target, Code)
    ;   generate_default_nav(Target, Code)
    ).

generate_nav_code(tab, Screens, react_native, Code) :-
    findall(ScreenCode, (
        member(screen(Name, Component, _), Screens),
        format(atom(ScreenCode), "      <Tab.Screen name=\"~w\" component={~w} />", [Name, Component])
    ), ScreenCodes),
    atomic_list_concat(ScreenCodes, '\n', ScreensStr),
    findall(ImportCode, (
        member(screen(_, Component, _), Screens),
        format(atom(ImportCode), "import { ~w } from '../screens/~w';", [Component, Component])
    ), ImportCodes),
    atomic_list_concat(ImportCodes, '\n', ImportsStr),
    format(atom(Code),
"import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
~w

const Tab = createBottomTabNavigator();

export const AppNavigator = () => (
  <Tab.Navigator>
~w
  </Tab.Navigator>
);
", [ImportsStr, ScreensStr]).

generate_nav_code(stack, Screens, react_native, Code) :-
    findall(ScreenCode, (
        member(screen(Name, Component, _), Screens),
        format(atom(ScreenCode), "      <Stack.Screen name=\"~w\" component={~w} />", [Name, Component])
    ), ScreenCodes),
    atomic_list_concat(ScreenCodes, '\n', ScreensStr),
    format(atom(Code),
"import { createNativeStackNavigator } from '@react-navigation/native-stack';

const Stack = createNativeStackNavigator();

export const AppNavigator = () => (
  <Stack.Navigator>
~w
  </Stack.Navigator>
);
", [ScreensStr]).

generate_nav_code(tab, Screens, swiftui, Code) :-
    findall(TabItem, (
        member(screen(Name, Component, _), Screens),
        capitalize_first(Name, CapName),
        format(atom(TabItem), "            ~w()
                .tabItem {
                    Label(\"~w\", systemImage: \"house\")
                }", [Component, CapName])
    ), TabItems),
    atomic_list_concat(TabItems, '\n', TabsStr),
    format(atom(Code),
"import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
~w
        }
    }
}
", [TabsStr]).

generate_nav_code(tab, Screens, flutter, Code) :-
    findall(RouteCode, (
        member(screen(Name, _, _), Screens),
        atom_string(Name, NameStr),
        format(atom(RouteCode), "      GoRoute(path: '/~w', builder: (_, __) => const ~wScreen()),", [NameStr, Name])
    ), RouteCodes),
    atomic_list_concat(RouteCodes, '\n', RoutesStr),
    format(atom(Code),
"import 'package:go_router/go_router.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    routes: [
~w
    ],
  );
});
", [RoutesStr]).

generate_nav_code(router, Screens, vue, Code) :-
    findall(RouteCode, (
        member(screen(Name, Component, _), Screens),
        format(atom(RouteCode), "  { path: '/~w', component: () => import('../views/~w.vue') },", [Name, Component])
    ), RouteCodes),
    atomic_list_concat(RouteCodes, '\n', RoutesStr),
    format(atom(Code),
"import { createRouter, createWebHistory } from 'vue-router';

const routes = [
~w
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
", [RoutesStr]).

generate_nav_code(_, _, Target, Code) :-
    generate_default_nav(Target, Code).

generate_default_nav(react_native, Code) :-
    Code = "import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { HomeScreen } from '../screens/HomeScreen';

const Stack = createNativeStackNavigator();

export const AppNavigator = () => (
  <Stack.Navigator>
    <Stack.Screen name=\"Home\" component={HomeScreen} />
  </Stack.Navigator>
);
".

generate_default_nav(vue, Code) :-
    Code = "import { createRouter, createWebHistory } from 'vue-router';

const routes = [
  { path: '/', component: () => import('../views/HomeView.vue') },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
".

generate_default_nav(flutter, Code) :-
    Code = "import 'package:go_router/go_router.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../screens/home_screen.dart';

final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    routes: [
      GoRoute(path: '/', builder: (_, __) => const HomeScreen()),
    ],
  );
});
".

generate_default_nav(swiftui, Code) :-
    Code = "import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            HomeView()
        }
    }
}
".

% ============================================================================
% SCREEN FILE GENERATION
% ============================================================================

%! generate_screen_files(+AppSpec, +Target, -Files)
%
%  Generate screen files from both explicit screens config and navigation.
%
generate_screen_files(AppSpec, Target, Files) :-
    % Get screens from screens config
    app_screens(AppSpec, ExplicitScreens),
    % Get screens from navigation
    app_navigation(AppSpec, Nav),
    (Nav = navigation(_, NavScreens, _) ->
        findall(Name, member(screen(Name, _, _), NavScreens), NavScreenNames)
    ;
        NavScreenNames = []
    ),
    % Combine screen names (navigation screens + explicit screens)
    (ExplicitScreens \= [] ->
        findall(Name, member(screen(Name, _), ExplicitScreens), ExplicitNames)
    ;
        ExplicitNames = []
    ),
    append(NavScreenNames, ExplicitNames, AllScreenNames),
    sort(AllScreenNames, UniqueScreenNames),  % Remove duplicates
    (UniqueScreenNames \= [] ->
        findall(file(Path, Code), (
            member(Name, UniqueScreenNames),
            % Use explicit config if available, otherwise empty
            (member(screen(Name, Config), ExplicitScreens) -> true ; Config = []),
            generate_screen_code(Name, Config, Target, Code),
            screen_file_path(Name, Target, Path)
        ), Files)
    ;
        % Generate default home screen if no screens defined
        generate_default_screen(Target, Code),
        screen_file_path(home, Target, Path),
        Files = [file(Path, Code)]
    ).

generate_screen_code(Name, Config, react_native, Code) :-
    capitalize_first(Name, ClassName),
    (member(layout(_, _, _), Config) -> _HasLayout = true ; _HasLayout = false),
    format(atom(Code),
"import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../theme';

export const ~wScreen = () => {
  const theme = useTheme();

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Text style={[styles.title, { color: theme.colors.text }]}>~w</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
});
", [ClassName, ClassName]).

generate_screen_code(Name, _Config, vue, Code) :-
    capitalize_first(Name, ClassName),
    format(atom(Code),
"<script setup lang=\"ts\">
// ~w View
</script>

<template>
  <div class=\"~w-view\">
    <h1>~w</h1>
  </div>
</template>

<style scoped>
.~w-view {
  padding: var(--spacing-md);
}
</style>
", [ClassName, Name, ClassName, Name]).

generate_screen_code(Name, _Config, flutter, Code) :-
    capitalize_first(Name, ClassName),
    format(atom(Code),
"import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class ~wScreen extends ConsumerWidget {
  const ~wScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(title: const Text('~w')),
      body: const Center(
        child: Text('~w Screen'),
      ),
    );
  }
}
", [ClassName, ClassName, ClassName, ClassName]).

generate_screen_code(Name, _Config, swiftui, Code) :-
    capitalize_first(Name, ClassName),
    format(atom(Code),
"import SwiftUI

struct ~wView: View {
    var body: some View {
        VStack {
            Text(\"~w\")
                .font(.title)
        }
        .padding()
    }
}
", [ClassName, ClassName]).

generate_default_screen(Target, Code) :-
    generate_screen_code(home, [], Target, Code).

% ============================================================================
% API CLIENT GENERATION
% ============================================================================

%! generate_api_client(+AppSpec, +Target, -Code)
generate_api_client(AppSpec, Target, Code) :-
    app_data(AppSpec, DataBindings),
    generate_api_client_code(DataBindings, Target, Code).

generate_api_client_code(_, react_native, Code) :-
    Code = "import axios from 'axios';

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);
".

generate_api_client_code(_, vue, Code) :-
    Code = "import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});
".

generate_api_client_code(_, flutter, Code) :-
    Code = "import 'package:dio/dio.dart';

class ApiClient {
  static final Dio _dio = Dio(BaseOptions(
    baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://localhost:8000'),
    connectTimeout: const Duration(seconds: 10),
  ));

  static Dio get instance => _dio;
}
".

generate_api_client_code(_, swiftui, Code) :-
    Code = "import Foundation

class APIClient {
    static let shared = APIClient()
    private let baseURL = ProcessInfo.processInfo.environment[\"API_URL\"] ?? \"http://localhost:8000\"

    func fetch<T: Decodable>(_ endpoint: String) async throws -> T {
        guard let url = URL(string: baseURL + endpoint) else {
            throw URLError(.badURL)
        }
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(T.self, from: data)
    }
}
".

% ============================================================================
% BACKEND GENERATION WITH DATA INTEGRATION
% ============================================================================

generate_backend_with_data(AppSpec, Target, OutputDir, Files) :-
    app_name(AppSpec, Name),
    app_data(AppSpec, DataBindings),
    format('  Generating ~w backend...~n', [Target]),

    project_generator:create_directory_structure(Target, OutputDir, _),

    % Generate backend files based on target
    generate_backend_files(Name, DataBindings, Target, OutputDir, Files).

generate_backend_files(Name, DataBindings, fastapi, OutputDir, Files) :-
    % Generate main.py
    generate_fastapi_main(Name, DataBindings, MainCode),
    atom_concat(OutputDir, '/main.py', MainPath),
    project_generator:write_project_file(MainPath, MainCode, []),

    % Generate requirements.txt
    project_generator:generate_requirements_txt(fastapi, ReqContent),
    atom_concat(OutputDir, '/requirements.txt', ReqPath),
    project_generator:write_project_file(ReqPath, ReqContent, []),

    Files = backend_files(fastapi, [MainPath, ReqPath]).

generate_backend_files(Name, DataBindings, flask, OutputDir, Files) :-
    % Generate app.py
    generate_flask_main(Name, DataBindings, AppCode),
    atom_concat(OutputDir, '/app.py', AppPath),
    project_generator:write_project_file(AppPath, AppCode, []),

    % Generate requirements.txt
    project_generator:generate_requirements_txt(flask, ReqContent),
    atom_concat(OutputDir, '/requirements.txt', ReqPath),
    project_generator:write_project_file(ReqPath, ReqContent, []),

    Files = backend_files(flask, [AppPath, ReqPath]).

generate_fastapi_main(Name, DataBindings, Code) :-
    generate_fastapi_routes(DataBindings, RoutesCode),
    format(atom(Code),
"from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=\"~w API\")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"*\"],
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

@app.get(\"/\")
async def root():
    return {\"message\": \"~w API\", \"status\": \"running\"}

@app.get(\"/health\")
async def health():
    return {\"status\": \"healthy\"}

~w

# Generated by UnifyWeaver
", [Name, Name, RoutesCode]).

generate_fastapi_routes([], "").
generate_fastapi_routes([binding(Name, Config)|Rest], Code) :-
    (member(endpoint(Endpoint), Config) -> true ; format(atom(Endpoint), '/api/~w', [Name])),
    format(atom(RouteCode),
"@app.get(\"~w\")
async def get_~w():
    # TODO: Implement data fetching
    return []
", [Endpoint, Name]),
    generate_fastapi_routes(Rest, RestCode),
    atom_concat(RouteCode, RestCode, Code).

generate_flask_main(Name, DataBindings, Code) :-
    generate_flask_routes(DataBindings, RoutesCode),
    format(atom(Code),
"from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({'message': '~w API', 'status': 'running'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

~w

if __name__ == '__main__':
    app.run(debug=True, port=8000)

# Generated by UnifyWeaver
", [Name, RoutesCode]).

generate_flask_routes([], "").
generate_flask_routes([binding(Name, Config)|Rest], Code) :-
    (member(endpoint(Endpoint), Config) -> true ; format(atom(Endpoint), '/api/~w', [Name])),
    format(atom(RouteCode),
"@app.route('~w')
def get_~w():
    # TODO: Implement data fetching
    return jsonify([])
", [Endpoint, Name]),
    generate_flask_routes(Rest, RestCode),
    atom_concat(RouteCode, RestCode, Code).

% ============================================================================
% PATH HELPERS
% ============================================================================

entry_point_path(react_native, '/App.tsx').
entry_point_path(vue, '/src/main.ts').
entry_point_path(flutter, '/lib/main.dart').
entry_point_path(swiftui, '/Sources/App.swift').

config_file_path(react_native, '/package.json').
config_file_path(vue, '/package.json').
config_file_path(flutter, '/pubspec.yaml').
config_file_path(swiftui, '/Package.swift').

theme_file_path(react_native, '/src/theme/index.tsx').
theme_file_path(vue, '/src/styles/theme.css').
theme_file_path(flutter, '/lib/theme/app_theme.dart').
theme_file_path(swiftui, '/Sources/Theme/Theme.swift').

locale_file_path(Locale, react_native, Path) :-
    format(atom(Path), '/src/i18n/~w.json', [Locale]).
locale_file_path(Locale, vue, Path) :-
    format(atom(Path), '/src/locales/~w.json', [Locale]).
locale_file_path(Locale, flutter, Path) :-
    format(atom(Path), '/lib/l10n/app_~w.arb', [Locale]).
locale_file_path(Locale, swiftui, Path) :-
    format(atom(Path), '/Resources/~w.lproj/Localizable.strings', [Locale]).

screen_file_path(Name, react_native, Path) :-
    capitalize_first(Name, ClassName),
    format(atom(Path), '/src/screens/~wScreen.tsx', [ClassName]).
screen_file_path(Name, vue, Path) :-
    capitalize_first(Name, ClassName),
    format(atom(Path), '/src/views/~wView.vue', [ClassName]).
screen_file_path(Name, flutter, Path) :-
    format(atom(Path), '/lib/screens/~w_screen.dart', [Name]).
screen_file_path(Name, swiftui, Path) :-
    capitalize_first(Name, ClassName),
    format(atom(Path), '/Sources/Views/~wView.swift', [ClassName]).

% ============================================================================
% VUE-SPECIFIC FILE GENERATION
% ============================================================================

%! generate_vue_index_html(+AppSpec, -Code)
%
%  Generate index.html for Vue/Vite project.
%
generate_vue_index_html(AppSpec, Code) :-
    app_name(AppSpec, Name),
    format(atom(Code),
'<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <link rel="icon" href="/favicon.ico">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>~w</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
', [Name]).

%! generate_vue_app_component(+AppSpec, -Code)
%
%  Generate App.vue root component.
%
generate_vue_app_component(_AppSpec, Code) :-
    Code = '<script setup lang="ts">
import { RouterView } from "vue-router";
</script>

<template>
  <div id="app">
    <RouterView />
  </div>
</template>

<style>
#app {
  font-family: system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
</style>
'.

%! generate_vue_vite_config(-Code)
%
%  Generate vite.config.ts for Vue project.
%
generate_vue_vite_config(Code) :-
    Code = 'import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
  },
});
'.

%! generate_vue_tsconfig(-Code)
%
%  Generate tsconfig.json for Vue project.
%
generate_vue_tsconfig(Code) :-
    Code = '{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "preserve",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
'.

%! generate_vue_tsconfig_node(-Code)
%
%  Generate tsconfig.node.json for Vue/Vite project.
%
generate_vue_tsconfig_node(Code) :-
    Code = '{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
'.

%! generate_vue_specific_files(+AppSpec, -Files)
%
%  Generate all Vue-specific config files.
%
generate_vue_specific_files(AppSpec, Files) :-
    generate_vue_index_html(AppSpec, IndexHtml),
    generate_vue_app_component(AppSpec, AppVue),
    generate_vue_vite_config(ViteConfig),
    generate_vue_tsconfig(TsConfig),
    generate_vue_tsconfig_node(TsConfigNode),
    Files = [
        file('/index.html', IndexHtml),
        file('/src/App.vue', AppVue),
        file('/vite.config.ts', ViteConfig),
        file('/tsconfig.json', TsConfig),
        file('/tsconfig.node.json', TsConfigNode)
    ].

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

capitalize_first(Atom, Capitalized) :-
    atom_chars(Atom, [First|Rest]),
    upcase_atom(First, Upper),
    atom_chars(Capitalized, [Upper|Rest]).

write_all_files([], _).
write_all_files([file(RelPath, Content)|Rest], OutputDir) :-
    atom_concat(OutputDir, RelPath, FullPath),
    % Ensure parent directory exists
    file_directory_name(FullPath, Dir),
    catch(make_directory_path(Dir), _, true),
    project_generator:write_project_file(FullPath, Content, []),
    write_all_files(Rest, OutputDir).

% ============================================================================
% TESTING
% ============================================================================

test_app_generator :-
    format('Running app generator tests...~n', []),

    % Test 1: App spec accessors
    TestApp = app(myapp, [theme([colors([primary-'#6366F1'])]), locales([en, es])]),
    app_name(TestApp, Name),
    (Name = myapp -> format('  Test 1 passed: app_name~n', []) ; format('  Test 1 FAILED~n', [])),

    % Test 2: Theme accessor
    app_theme(TestApp, Theme),
    (Theme \= [] -> format('  Test 2 passed: app_theme~n', []) ; format('  Test 2 FAILED~n', [])),

    % Test 3: Locales accessor
    app_locales(TestApp, Locales),
    (Locales = [en, es] -> format('  Test 3 passed: app_locales~n', []) ; format('  Test 3 FAILED~n', [])),

    % Test 4: React Native entry generation
    generate_entry_point(TestApp, react_native, RNCode),
    (sub_atom(RNCode, _, _, _, 'QueryClientProvider') -> format('  Test 4 passed: RN entry~n', []) ; format('  Test 4 FAILED~n', [])),

    % Test 5: Vue entry generation
    generate_entry_point(TestApp, vue, VueCode),
    (sub_atom(VueCode, _, _, _, 'createPinia') -> format('  Test 5 passed: Vue entry~n', []) ; format('  Test 5 FAILED~n', [])),

    % Test 6: Flutter entry generation
    generate_entry_point(TestApp, flutter, FlutterCode),
    (sub_atom(FlutterCode, _, _, _, 'ProviderScope') -> format('  Test 6 passed: Flutter entry~n', []) ; format('  Test 6 FAILED~n', [])),

    % Test 7: SwiftUI entry generation
    generate_entry_point(TestApp, swiftui, SwiftCode),
    (sub_atom(SwiftCode, _, _, _, '@main') -> format('  Test 7 passed: SwiftUI entry~n', []) ; format('  Test 7 FAILED~n', [])),

    % Test 8: Theme file generation
    generate_theme_files(TestApp, react_native, ThemeFiles),
    (ThemeFiles \= [] -> format('  Test 8 passed: theme files~n', []) ; format('  Test 8 FAILED~n', [])),

    % Test 9: Locale file generation
    generate_locale_files(TestApp, react_native, LocaleFiles),
    (length(LocaleFiles, 2) -> format('  Test 9 passed: locale files~n', []) ; format('  Test 9 FAILED~n', [])),

    % Test 10: Navigation generation
    NavApp = app(navapp, [navigation(tab, [screen(home, 'HomeScreen', []), screen(settings, 'SettingsScreen', [])], [])]),
    generate_navigation_file(NavApp, react_native, NavCode),
    (sub_atom(NavCode, _, _, _, 'Tab.Navigator') -> format('  Test 10 passed: navigation~n', []) ; format('  Test 10 FAILED~n', [])),

    format('All 10 app generator tests completed!~n', []).

:- initialization(test_app_generator, program).
