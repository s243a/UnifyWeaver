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
%   - Navigation Guards (auth, roles, permissions)
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
    app_guards/2,                       % +AppSpec, -Guards
    app_https/2,                        % +AppSpec, -HttpsConfig
    app_auth/2,                         % +AppSpec, -AuthConfig
    app_requires_https/1,               % +AppSpec
    app_requires_secure_auth/1,         % +AppSpec

    % Guard generation
    generate_guard_files/3,             % +AppSpec, +Target, -Files
    generate_auth_store/2,              % +Target, -Code
    screen_has_guards/2,                % +Screen, -Guards

    % Secure auth backend generation
    generate_auth_backend/3,            % +AppSpec, +Target, -Files

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

% Discontiguous predicates (clauses spread across file for readability)
:- discontiguous generate_nav_code/4.
:- discontiguous generate_default_nav/2.

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

app_guards(app(_, Config), Guards) :-
    (member(guards(Guards), Config) -> true ; Guards = []).

%! app_https(+AppSpec, -HttpsConfig)
%
%  Extract HTTPS configuration from app spec.
%  Options: https(true) or https([port(Port), cert(CertPath), key(KeyPath)])
%
app_https(app(_, Config), HttpsConfig) :-
    (   member(https(HttpsConfig), Config)
    ->  true
    ;   HttpsConfig = false
    ).

%! app_auth(+AppSpec, -AuthConfig)
%
%  Extract secure auth configuration from app spec.
%  Options: auth(secure, [...]) for backend auth generation
%
app_auth(app(_, Config), AuthConfig) :-
    (   member(auth(secure, AuthConfig), Config)
    ->  true
    ;   member(auth(secure), Config)
    ->  AuthConfig = []  % secure with defaults
    ;   AuthConfig = none
    ).

%! app_requires_https(+AppSpec)
%  True if the app requires HTTPS.
app_requires_https(AppSpec) :-
    app_https(AppSpec, HttpsConfig),
    HttpsConfig \= false.

%! app_requires_secure_auth(+AppSpec)
%  True if the app requires secure backend authentication.
app_requires_secure_auth(AppSpec) :-
    app_auth(AppSpec, AuthConfig),
    AuthConfig \= none.

%! screen_has_guards(+Screen, -Guards)
%
%  Extract guards from screen options.
%  Supports both guards([...]) and protected(true) shorthand.
%
screen_has_guards(screen(_, _, Opts), Guards) :-
    (   member(guards(Guards), Opts)
    ->  true
    ;   member(protected(true), Opts)
    ->  Guards = [auth]
    ;   Guards = []
    ).

%! get_guard_spec(+GuardName, +AppSpec, -GuardSpec)
%
%  Look up a guard definition from the app spec.
%
get_guard_spec(GuardName, AppSpec, GuardSpec) :-
    app_guards(AppSpec, Guards),
    member(guard(GuardName, GuardSpec), Guards).

%! get_guard_spec(+GuardName, +AppSpec, -GuardSpec)
%
%  Default guard specs for common guards.
%
get_guard_spec(auth, _, [check(authenticated), redirect('/login')]) :- !.
get_guard_spec(guest, _, [check(not_authenticated), redirect('/')]) :- !.
get_guard_spec(admin, _, [check(role(admin)), redirect('/unauthorized')]) :- !.

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
    generate_guard_files(AppSpec, Target, GuardFiles),

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
        GuardFiles,
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
    (   some_screen_has_guards(Screens)
    ->  generate_rn_stack_with_guards(Screens, Code)
    ;   generate_rn_stack_simple(Screens, Code)
    ).

%! generate_rn_stack_simple(+Screens, -Code)
%  Generate simple React Native stack navigator without guards.
generate_rn_stack_simple(Screens, Code) :-
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

%! generate_rn_stack_with_guards(+Screens, -Code)
%  Generate React Native stack navigator with authentication guards.
generate_rn_stack_with_guards(Screens, Code) :-
    findall(ScreenCode, (
        member(Screen, Screens),
        Screen = screen(Name, Component, _),
        screen_has_guards(Screen, Guards),
        generate_rn_screen_with_guards(Name, Component, Guards, ScreenCode)
    ), ScreenCodes),
    atomic_list_concat(ScreenCodes, '\n', ScreensStr),
    format(atom(Code),
"import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useAuthStore } from '../stores/authStore';
import { LoginScreen } from '../screens/LoginScreen';
import { UnauthorizedScreen } from '../screens/UnauthorizedScreen';

const Stack = createNativeStackNavigator();

// Guard wrapper component for protected screens
const withGuards = (Component: React.ComponentType<any>, guards: string[]) => {
  return function GuardedScreen(props: any) {
    const authStore = useAuthStore();

    for (const guard of guards) {
      switch (guard) {
        case 'auth':
          if (!authStore.isAuthenticated()) {
            return <LoginScreen {...props} />;
          }
          break;
        case 'guest':
          if (authStore.isAuthenticated()) {
            // Redirect to home - in practice, use navigation
            return null;
          }
          break;
        case 'admin':
          if (!authStore.hasRole('admin')) {
            return <UnauthorizedScreen {...props} />;
          }
          break;
        default:
          if (!authStore.hasPermission(guard)) {
            return <UnauthorizedScreen {...props} />;
          }
      }
    }
    return <Component {...props} />;
  };
};

export const AppNavigator = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen name=\"Login\" component={LoginScreen} options={{ headerShown: false }} />
      <Stack.Screen name=\"Unauthorized\" component={UnauthorizedScreen} />
~w
    </Stack.Navigator>
  );
};
", [ScreensStr]).

%! generate_rn_screen_with_guards(+Name, +Component, +Guards, -ScreenCode)
generate_rn_screen_with_guards(Name, Component, [], ScreenCode) :-
    format(atom(ScreenCode), "      <Stack.Screen name=\"~w\" component={~w} />", [Name, Component]).
generate_rn_screen_with_guards(Name, Component, Guards, ScreenCode) :-
    Guards \= [],
    format_guard_list(Guards, GuardsStr),
    format(atom(ScreenCode), "      <Stack.Screen name=\"~w\" component={withGuards(~w, [~w])} />", [Name, Component, GuardsStr]).

generate_nav_code(tab, Screens, swiftui, Code) :-
    (   some_screen_has_guards(Screens)
    ->  generate_swiftui_nav_with_guards(Screens, Code)
    ;   generate_swiftui_nav_simple(Screens, Code)
    ).

%! generate_swiftui_nav_simple(+Screens, -Code)
%  Generate simple SwiftUI TabView without guards.
generate_swiftui_nav_simple(Screens, Code) :-
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

%! generate_swiftui_nav_with_guards(+Screens, -Code)
%  Generate SwiftUI navigation with authentication guards.
generate_swiftui_nav_with_guards(Screens, Code) :-
    findall(TabItem, (
        member(Screen, Screens),
        Screen = screen(Name, Component, _),
        capitalize_first(Name, CapName),
        screen_has_guards(Screen, Guards),
        generate_swiftui_tab_with_guards(Component, CapName, Guards, TabItem)
    ), TabItems),
    atomic_list_concat(TabItems, '\n', TabsStr),
    format(atom(Code),
"import SwiftUI

struct ContentView: View {
    @StateObject private var authStore = AuthStore()

    var body: some View {
        Group {
            if authStore.isAuthenticated {
                TabView {
~w
                }
            } else {
                LoginView()
            }
        }
        .environmentObject(authStore)
        .task {
            await authStore.checkAuth()
        }
    }
}

// Guard wrapper view for protected content
struct GuardedView<Content: View>: View {
    @EnvironmentObject var authStore: AuthStore
    let guards: [String]
    let content: () -> Content

    var body: some View {
        Group {
            if checkGuards() {
                content()
            } else {
                UnauthorizedView()
            }
        }
    }

    private func checkGuards() -> Bool {
        for guard in guards {
            switch guard {
            case \"auth\":
                if !authStore.isAuthenticated { return false }
            case \"guest\":
                if authStore.isAuthenticated { return false }
            case \"admin\":
                if !authStore.hasRole(\"admin\") { return false }
            default:
                if !authStore.hasPermission(guard) { return false }
            }
        }
        return true
    }
}
", [TabsStr]).

%! generate_swiftui_tab_with_guards(+Component, +CapName, +Guards, -TabItem)
generate_swiftui_tab_with_guards(Component, CapName, [], TabItem) :-
    format(atom(TabItem), "            ~w()
                .tabItem {
                    Label(\"~w\", systemImage: \"house\")
                }", [Component, CapName]).
generate_swiftui_tab_with_guards(Component, CapName, Guards, TabItem) :-
    Guards \= [],
    format_swiftui_guard_list(Guards, GuardsStr),
    format(atom(TabItem), "            GuardedView(guards: [~w]) {
                ~w()
            }
            .tabItem {
                Label(\"~w\", systemImage: \"house\")
            }", [GuardsStr, Component, CapName]).

%! format_swiftui_guard_list(+Guards, -Str)
format_swiftui_guard_list(Guards, Str) :-
    findall(Quoted, (member(G, Guards), format(atom(Quoted), "\"~w\"", [G])), QuotedList),
    atomic_list_concat(QuotedList, ', ', Str).

generate_nav_code(tab, Screens, flutter, Code) :-
    (   some_screen_has_guards(Screens)
    ->  generate_flutter_router_with_guards(Screens, Code)
    ;   generate_flutter_router_simple(Screens, Code)
    ).

%! generate_flutter_router_simple(+Screens, -Code)
%  Generate simple Flutter GoRouter without guards.
generate_flutter_router_simple(Screens, Code) :-
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

%! generate_flutter_router_with_guards(+Screens, -Code)
%  Generate Flutter GoRouter with navigation guards.
generate_flutter_router_with_guards(Screens, Code) :-
    findall(RouteCode, (
        member(Screen, Screens),
        Screen = screen(Name, _, _),
        atom_string(Name, NameStr),
        screen_has_guards(Screen, Guards),
        generate_flutter_route_with_guards(NameStr, Name, Guards, RouteCode)
    ), RouteCodes),
    atomic_list_concat(RouteCodes, '\n', RoutesStr),
    format(atom(Code),
"import 'package:go_router/go_router.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/auth_provider.dart';
import '../screens/login_screen.dart';
import '../screens/unauthorized_screen.dart';

final routerProvider = Provider<GoRouter>((ref) {
  final authState = ref.watch(authProvider);

  return GoRouter(
    redirect: (context, state) {
      final isAuthenticated = authState.isAuthenticated;
      final isLoggingIn = state.matchedLocation == '/login';

      // If not authenticated and not on login page, redirect to login
      if (!isAuthenticated && !isLoggingIn) {
        // Check if route requires auth
        final requiresAuth = _routeRequiresAuth(state.matchedLocation);
        if (requiresAuth) {
          return '/login?redirect=\\${state.matchedLocation}';
        }
      }

      // If authenticated and on login page, redirect away
      if (isAuthenticated && isLoggingIn) {
        return '/';
      }

      return null;
    },
    routes: [
      GoRoute(
        path: '/login',
        builder: (_, __) => const LoginScreen(),
      ),
      GoRoute(
        path: '/unauthorized',
        builder: (_, __) => const UnauthorizedScreen(),
      ),
~w
    ],
  );
});

// Map of routes that require authentication
bool _routeRequiresAuth(String path) {
  const protectedRoutes = <String>[
    // Add protected routes here
  ];
  return protectedRoutes.any((route) => path.startsWith(route));
}
", [RoutesStr]).

%! generate_flutter_route_with_guards(+NameStr, +Name, +Guards, -RouteCode)
generate_flutter_route_with_guards(NameStr, Name, [], RouteCode) :-
    format(atom(RouteCode), "      GoRoute(path: '/~w', builder: (_, __) => const ~wScreen()),", [NameStr, Name]).
generate_flutter_route_with_guards(NameStr, Name, Guards, RouteCode) :-
    Guards \= [],
    format_flutter_guard_redirect(Guards, GuardCode),
    format(atom(RouteCode), "      GoRoute(
        path: '/~w',
        builder: (_, __) => const ~wScreen(),
        redirect: (context, state) {
          final authState = context.read(authProvider);
~w
          return null;
        },
      ),", [NameStr, Name, GuardCode]).

%! format_flutter_guard_redirect(+Guards, -Code)
format_flutter_guard_redirect(Guards, Code) :-
    findall(GuardCheck, (
        member(Guard, Guards),
        format_single_flutter_guard(Guard, GuardCheck)
    ), GuardChecks),
    atomic_list_concat(GuardChecks, '\n', Code).

format_single_flutter_guard(auth, Code) :-
    Code = "          if (!authState.isAuthenticated) return '/login';".
format_single_flutter_guard(guest, Code) :-
    Code = "          if (authState.isAuthenticated) return '/';".
format_single_flutter_guard(admin, Code) :-
    Code = "          if (!authState.hasRole('admin')) return '/unauthorized';".
format_single_flutter_guard(Guard, Code) :-
    Guard \= auth, Guard \= guest, Guard \= admin,
    format(atom(Code), "          if (!authState.hasPermission('~w')) return '/unauthorized';", [Guard]).

generate_nav_code(router, Screens, vue, Code) :-
    % Get first screen for default redirect
    (Screens = [screen(FirstName, _, _)|_] -> true ; FirstName = home),
    % Check if any screens have guards
    (   some_screen_has_guards(Screens)
    ->  generate_vue_router_with_guards(Screens, FirstName, Code)
    ;   generate_vue_router_simple(Screens, FirstName, Code)
    ).

%! some_screen_has_guards(+Screens)
%  True if any screen in the list has guards.
some_screen_has_guards(Screens) :-
    member(Screen, Screens),
    screen_has_guards(Screen, Guards),
    Guards \= [],
    !.

%! generate_vue_router_simple(+Screens, +FirstName, -Code)
%  Generate simple Vue router without guards.
generate_vue_router_simple(Screens, FirstName, Code) :-
    findall(RouteCode, (
        member(screen(Name, _, _), Screens),
        capitalize_first(Name, CapName),
        format(atom(RouteCode), "  { path: '/~w', component: () => import('../views/~wView.vue') },", [Name, CapName])
    ), RouteCodes),
    atomic_list_concat(RouteCodes, '\n', RoutesStr),
    format(atom(Code),
"import { createRouter, createWebHistory } from 'vue-router';

const routes = [
  { path: '/', redirect: '/~w' },
~w
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
", [FirstName, RoutesStr]).

%! generate_vue_router_with_guards(+Screens, +FirstName, -Code)
%  Generate Vue router with navigation guards.
generate_vue_router_with_guards(Screens, FirstName, Code) :-
    findall(RouteCode, (
        member(Screen, Screens),
        Screen = screen(Name, _, _),
        capitalize_first(Name, CapName),
        screen_has_guards(Screen, Guards),
        generate_vue_route_with_meta(Name, CapName, Guards, RouteCode)
    ), RouteCodes),
    atomic_list_concat(RouteCodes, '\n', RoutesStr),
    format(atom(Code),
"import { createRouter, createWebHistory } from 'vue-router';
import type { RouteLocationNormalized } from 'vue-router';
import { useAuthStore } from '../stores/auth';

const routes = [
  { path: '/', redirect: '/~w' },
  { path: '/login', name: 'login', component: () => import('../views/LoginView.vue') },
  { path: '/unauthorized', name: 'unauthorized', component: () => import('../views/UnauthorizedView.vue') },
~w
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});

// Navigation guards
router.beforeEach((to: RouteLocationNormalized, _from: RouteLocationNormalized) => {
  const authStore = useAuthStore();
  const guards = to.meta.guards as string[] | undefined;

  if (!guards || guards.length === 0) {
    return true;
  }

  for (const guard of guards) {
    switch (guard) {
      case 'auth':
        if (!authStore.isAuthenticated) {
          return { name: 'login', query: { redirect: to.fullPath } };
        }
        break;
      case 'guest':
        if (authStore.isAuthenticated) {
          return { path: '/' };
        }
        break;
      case 'admin':
        if (!authStore.hasRole('admin')) {
          return { name: 'unauthorized' };
        }
        break;
      default:
        // Custom guard - check if user has the required role/permission
        if (!authStore.hasPermission(guard)) {
          return { name: 'unauthorized' };
        }
    }
  }
  return true;
});

export default router;
", [FirstName, RoutesStr]).

%! generate_vue_route_with_meta(+Name, +CapName, +Guards, -RouteCode)
%  Generate a Vue route with meta.guards if guards are present.
generate_vue_route_with_meta(Name, CapName, [], RouteCode) :-
    format(atom(RouteCode), "  { path: '/~w', name: '~w', component: () => import('../views/~wView.vue') },", [Name, Name, CapName]).
generate_vue_route_with_meta(Name, CapName, Guards, RouteCode) :-
    Guards \= [],
    format_guard_list(Guards, GuardsStr),
    format(atom(RouteCode), "  { path: '/~w', name: '~w', component: () => import('../views/~wView.vue'), meta: { guards: [~w] } },", [Name, Name, CapName, GuardsStr]).

%! format_guard_list(+Guards, -Str)
%  Format a list of guard names as a JavaScript array literal.
format_guard_list(Guards, Str) :-
    findall(Quoted, (member(G, Guards), format(atom(Quoted), "'~w'", [G])), QuotedList),
    atomic_list_concat(QuotedList, ', ', Str).

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

%! generate_vue_vite_config(+AppSpec, -Code)
%
%  Generate vite.config.ts for Vue project.
%  Includes HTTPS configuration if https(true) is specified.
%
generate_vue_vite_config(AppSpec, Code) :-
    (   app_requires_https(AppSpec)
    ->  generate_vue_vite_config_https(Code)
    ;   generate_vue_vite_config_http(Code)
    ).

generate_vue_vite_config_http(Code) :-
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

generate_vue_vite_config_https(Code) :-
    Code = 'import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";
import fs from "node:fs";
import path from "node:path";

// Generate self-signed certificate for development
// Run: npm run generate-cert (or use mkcert for trusted local certs)
const httpsConfig = (() => {
  const certPath = path.resolve(__dirname, "certs/localhost.pem");
  const keyPath = path.resolve(__dirname, "certs/localhost-key.pem");

  if (fs.existsSync(certPath) && fs.existsSync(keyPath)) {
    return {
      key: fs.readFileSync(keyPath),
      cert: fs.readFileSync(certPath),
    };
  }

  // Fallback: use Vite\'s built-in self-signed cert generation
  console.warn("No certs found in ./certs/. Using Vite\'s auto-generated self-signed cert.");
  console.warn("For trusted local HTTPS, run: npx mkcert localhost");
  return true;
})();

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    https: httpsConfig,
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
    generate_vue_vite_config(AppSpec, ViteConfig),
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
% GUARD FILES GENERATION
% ============================================================================

%! generate_guard_files(+AppSpec, +Target, -Files)
%
%  Generate authentication/authorization store files for navigation guards.
%  Only generates files if the app uses guards.
%  Generates: auth store, login screen, and unauthorized screen.
%
generate_guard_files(AppSpec, Target, Files) :-
    app_screens(AppSpec, Screens),
    app_navigation(AppSpec, Nav),
    (Nav = navigation(_, NavScreens, _) -> AllScreens = NavScreens ; AllScreens = Screens),
    (   some_screen_has_guards(AllScreens)
    ->  generate_auth_store(Target, AuthStoreCode),
        guard_store_path(Target, StorePath),
        generate_login_screen(Target, LoginCode),
        login_screen_path(Target, LoginPath),
        generate_unauthorized_screen(Target, UnauthorizedCode),
        unauthorized_screen_path(Target, UnauthorizedPath),
        Files = [
            file(StorePath, AuthStoreCode),
            file(LoginPath, LoginCode),
            file(UnauthorizedPath, UnauthorizedCode)
        ]
    ;   Files = []
    ).

%! login_screen_path(+Target, -Path)
login_screen_path(vue, '/src/views/LoginView.vue').
login_screen_path(react_native, '/src/screens/LoginScreen.tsx').
login_screen_path(flutter, '/lib/screens/login_screen.dart').
login_screen_path(swiftui, '/Sources/Views/LoginView.swift').

%! unauthorized_screen_path(+Target, -Path)
unauthorized_screen_path(vue, '/src/views/UnauthorizedView.vue').
unauthorized_screen_path(react_native, '/src/screens/UnauthorizedScreen.tsx').
unauthorized_screen_path(flutter, '/lib/screens/unauthorized_screen.dart').
unauthorized_screen_path(swiftui, '/Sources/Views/UnauthorizedView.swift').

%! generate_login_screen(+Target, -Code)
generate_login_screen(vue, Code) :-
    Code = "<script setup lang=\"ts\">
import { ref } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useAuthStore } from '../stores/auth';

const router = useRouter();
const route = useRoute();
const authStore = useAuthStore();

const email = ref('');
const password = ref('');
const error = ref('');
const loading = ref(false);

const handleLogin = async () => {
  error.value = '';
  loading.value = true;
  try {
    await authStore.login(email.value, password.value);
    const redirect = route.query.redirect as string || '/';
    router.push(redirect);
  } catch (e) {
    error.value = 'Invalid email or password';
  } finally {
    loading.value = false;
  }
};
</script>

<template>
  <div class=\"login-container\">
    <div class=\"login-card\">
      <h1>Login</h1>
      <form @submit.prevent=\"handleLogin\">
        <div class=\"form-group\">
          <label for=\"email\">Email</label>
          <input
            id=\"email\"
            v-model=\"email\"
            type=\"email\"
            placeholder=\"Enter your email\"
            required
          />
        </div>
        <div class=\"form-group\">
          <label for=\"password\">Password</label>
          <input
            id=\"password\"
            v-model=\"password\"
            type=\"password\"
            placeholder=\"Enter your password\"
            required
          />
        </div>
        <p v-if=\"error\" class=\"error\">{{ error }}</p>
        <button type=\"submit\" :disabled=\"loading\">
          {{ loading ? 'Logging in...' : 'Login' }}
        </button>
      </form>
    </div>
  </div>
</template>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: var(--color-background);
}

.login-card {
  background: var(--color-surface);
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

h1 {
  text-align: center;
  color: var(--color-text-primary);
  margin-bottom: 1.5rem;
}

.form-group {
  margin-bottom: 1rem;
}

label {
  display: block;
  margin-bottom: 0.5rem;
  color: var(--color-text-secondary);
}

input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 1rem;
}

button {
  width: 100%;
  padding: 0.75rem;
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  margin-top: 1rem;
}

button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.error {
  color: var(--color-error);
  margin-top: 0.5rem;
  text-align: center;
}
</style>
".

generate_login_screen(react_native, Code) :-
    Code = "import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { useAuthStore } from '../stores/authStore';
import { useTheme } from '../theme';

export const LoginScreen = () => {
  const navigation = useNavigation();
  const route = useRoute();
  const authStore = useAuthStore();
  const theme = useTheme();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setError('');
    setLoading(true);
    try {
      await authStore.login(email, password);
      const redirect = (route.params as any)?.redirect || 'Home';
      navigation.navigate(redirect as never);
    } catch (e) {
      setError('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={[styles.card, { backgroundColor: theme.colors.surface }]}>
        <Text style={[styles.title, { color: theme.colors.text }]}>Login</Text>

        <TextInput
          style={[styles.input, { borderColor: theme.colors.border }]}
          placeholder=\"Email\"
          value={email}
          onChangeText={setEmail}
          keyboardType=\"email-address\"
          autoCapitalize=\"none\"
        />

        <TextInput
          style={[styles.input, { borderColor: theme.colors.border }]}
          placeholder=\"Password\"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        {error ? <Text style={styles.error}>{error}</Text> : null}

        <TouchableOpacity
          style={[styles.button, { backgroundColor: theme.colors.primary }]}
          onPress={handleLogin}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            {loading ? 'Logging in...' : 'Login'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  card: {
    padding: 24,
    borderRadius: 8,
    width: '100%',
    maxWidth: 400,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 24,
  },
  input: {
    borderWidth: 1,
    borderRadius: 4,
    padding: 12,
    marginBottom: 16,
    fontSize: 16,
  },
  button: {
    padding: 12,
    borderRadius: 4,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  error: {
    color: 'red',
    textAlign: 'center',
    marginBottom: 8,
  },
});
".

generate_login_screen(flutter, Code) :-
    Code = "import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../providers/auth_provider.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  String? _error;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _handleLogin() async {
    setState(() => _error = null);
    try {
      await ref.read(authProvider.notifier).login(
        _emailController.text,
        _passwordController.text,
      );
      if (mounted) {
        final redirect = GoRouterState.of(context).uri.queryParameters['redirect'] ?? '/';
        context.go(redirect);
      }
    } catch (e) {
      setState(() => _error = 'Invalid email or password');
    }
  }

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);

    return Scaffold(
      body: Center(
        child: Container(
          padding: const EdgeInsets.all(24),
          constraints: const BoxConstraints(maxWidth: 400),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                'Login',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 24),
              TextField(
                controller: _emailController,
                decoration: const InputDecoration(
                  labelText: 'Email',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.emailAddress,
              ),
              const SizedBox(height: 16),
              TextField(
                controller: _passwordController,
                decoration: const InputDecoration(
                  labelText: 'Password',
                  border: OutlineInputBorder(),
                ),
                obscureText: true,
              ),
              if (_error != null) ...[
                const SizedBox(height: 8),
                Text(_error!, style: const TextStyle(color: Colors.red)),
              ],
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: authState.loading ? null : _handleLogin,
                  child: Text(authState.loading ? 'Logging in...' : 'Login'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
".

generate_login_screen(swiftui, Code) :-
    Code = "import SwiftUI

struct LoginView: View {
    @EnvironmentObject var authStore: AuthStore
    @State private var email = \"\"
    @State private var password = \"\"
    @State private var error: String?

    var body: some View {
        VStack(spacing: 24) {
            Text(\"Login\")
                .font(.largeTitle)
                .fontWeight(.bold)

            VStack(spacing: 16) {
                TextField(\"Email\", text: $email)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .autocapitalization(.none)
                    .keyboardType(.emailAddress)

                SecureField(\"Password\", text: $password)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
            .padding(.horizontal)

            if let error = error {
                Text(error)
                    .foregroundColor(.red)
            }

            Button(action: handleLogin) {
                if authStore.loading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Text(\"Login\")
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
            .padding(.horizontal)
            .disabled(authStore.loading)
        }
        .padding()
    }

    private func handleLogin() {
        error = nil
        Task {
            do {
                try await authStore.login(email: email, password: password)
            } catch {
                self.error = \"Invalid email or password\"
            }
        }
    }
}
".

%! generate_unauthorized_screen(+Target, -Code)
generate_unauthorized_screen(vue, Code) :-
    Code = "<script setup lang=\"ts\">
import { useRouter } from 'vue-router';

const router = useRouter();

const goBack = () => {
  router.back();
};

const goHome = () => {
  router.push('/');
};
</script>

<template>
  <div class=\"unauthorized-container\">
    <div class=\"unauthorized-card\">
      <h1>Access Denied</h1>
      <p>You don't have permission to access this page.</p>
      <div class=\"actions\">
        <button @click=\"goBack\">Go Back</button>
        <button @click=\"goHome\" class=\"secondary\">Go Home</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.unauthorized-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: var(--color-background);
}

.unauthorized-card {
  background: var(--color-surface);
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  max-width: 400px;
}

h1 {
  color: var(--color-error);
  margin-bottom: 1rem;
}

p {
  color: var(--color-text-secondary);
  margin-bottom: 1.5rem;
}

.actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

button {
  padding: 0.75rem 1.5rem;
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button.secondary {
  background: var(--color-secondary);
}
</style>
".

generate_unauthorized_screen(react_native, Code) :-
    Code = "import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme';

export const UnauthorizedScreen = () => {
  const navigation = useNavigation();
  const theme = useTheme();

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={[styles.card, { backgroundColor: theme.colors.surface }]}>
        <Text style={[styles.title, { color: theme.colors.error }]}>Access Denied</Text>
        <Text style={[styles.message, { color: theme.colors.textSecondary }]}>
          You don't have permission to access this page.
        </Text>
        <View style={styles.actions}>
          <TouchableOpacity
            style={[styles.button, { backgroundColor: theme.colors.primary }]}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.buttonText}>Go Back</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, { backgroundColor: theme.colors.secondary }]}
            onPress={() => navigation.navigate('Home' as never)}
          >
            <Text style={styles.buttonText}>Go Home</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  card: {
    padding: 24,
    borderRadius: 8,
    alignItems: 'center',
    maxWidth: 400,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  message: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 4,
  },
  buttonText: {
    color: 'white',
    fontWeight: '600',
  },
});
".

generate_unauthorized_screen(flutter, Code) :-
    Code = "import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class UnauthorizedScreen extends StatelessWidget {
  const UnauthorizedScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Container(
          padding: const EdgeInsets.all(24),
          constraints: const BoxConstraints(maxWidth: 400),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.lock, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                'Access Denied',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: Colors.red,
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'You don\\'t have permission to access this page.',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: () => context.pop(),
                    child: const Text('Go Back'),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton(
                    onPressed: () => context.go('/'),
                    child: const Text('Go Home'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
".

generate_unauthorized_screen(swiftui, Code) :-
    Code = "import SwiftUI

struct UnauthorizedView: View {
    @Environment(\\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: \"lock.fill\")
                .font(.system(size: 64))
                .foregroundColor(.red)

            Text(\"Access Denied\")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.red)

            Text(\"You don't have permission to access this page.\")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)

            HStack(spacing: 12) {
                Button(\"Go Back\") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
    }
}
".

%! guard_store_path(+Target, -Path)
%  Path for the auth store file.
guard_store_path(vue, '/src/stores/auth.ts').
guard_store_path(react_native, '/src/stores/authStore.ts').
guard_store_path(flutter, '/lib/providers/auth_provider.dart').
guard_store_path(swiftui, '/Sources/Stores/AuthStore.swift').

%! generate_auth_store(+Target, -Code)
%
%  Generate the authentication store for each target.
%
generate_auth_store(vue, Code) :-
    Code = "import { defineStore } from 'pinia';

interface User {
  id: string;
  email: string;
  roles: string[];
  permissions: string[];
}

interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    user: null,
    token: null,
    loading: false,
  }),

  getters: {
    isAuthenticated: (state) => !!state.token && !!state.user,

    currentUser: (state) => state.user,

    hasRole: (state) => (role: string) => {
      return state.user?.roles.includes(role) ?? false;
    },

    hasPermission: (state) => (permission: string) => {
      return state.user?.permissions.includes(permission) ?? false;
    },
  },

  actions: {
    async login(email: string, password: string) {
      this.loading = true;
      try {
        // Replace with actual API call
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password }),
        });
        const data = await response.json();
        this.token = data.token;
        this.user = data.user;
        localStorage.setItem('auth_token', data.token);
      } finally {
        this.loading = false;
      }
    },

    logout() {
      this.token = null;
      this.user = null;
      localStorage.removeItem('auth_token');
    },

    async checkAuth() {
      const token = localStorage.getItem('auth_token');
      if (!token) return;

      this.token = token;
      try {
        const response = await fetch('/api/auth/me', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (response.ok) {
          this.user = await response.json();
        } else {
          this.logout();
        }
      } catch {
        this.logout();
      }
    },
  },
});
".

generate_auth_store(react_native, Code) :-
    Code = "import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface User {
  id: string;
  email: string;
  roles: string[];
  permissions: string[];
}

interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
  isAuthenticated: () => boolean;
  hasRole: (role: string) => boolean;
  hasPermission: (permission: string) => boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      loading: false,

      isAuthenticated: () => !!get().token && !!get().user,

      hasRole: (role: string) => get().user?.roles.includes(role) ?? false,

      hasPermission: (permission: string) =>
        get().user?.permissions.includes(permission) ?? false,

      login: async (email: string, password: string) => {
        set({ loading: true });
        try {
          // Replace with actual API call
          const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
          });
          const data = await response.json();
          set({ token: data.token, user: data.user });
        } finally {
          set({ loading: false });
        }
      },

      logout: () => {
        set({ token: null, user: null });
      },

      checkAuth: async () => {
        const { token } = get();
        if (!token) return;

        try {
          const response = await fetch('/api/auth/me', {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (response.ok) {
            const user = await response.json();
            set({ user });
          } else {
            get().logout();
          }
        } catch {
          get().logout();
        }
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);
".

generate_auth_store(flutter, Code) :-
    Code = "import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class User {
  final String id;
  final String email;
  final List<String> roles;
  final List<String> permissions;

  User({
    required this.id,
    required this.email,
    required this.roles,
    required this.permissions,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      email: json['email'],
      roles: List<String>.from(json['roles'] ?? []),
      permissions: List<String>.from(json['permissions'] ?? []),
    );
  }
}

class AuthState {
  final User? user;
  final String? token;
  final bool loading;

  AuthState({this.user, this.token, this.loading = false});

  bool get isAuthenticated => token != null && user != null;

  bool hasRole(String role) => user?.roles.contains(role) ?? false;

  bool hasPermission(String permission) =>
      user?.permissions.contains(permission) ?? false;

  AuthState copyWith({User? user, String? token, bool? loading}) {
    return AuthState(
      user: user ?? this.user,
      token: token ?? this.token,
      loading: loading ?? this.loading,
    );
  }
}

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier() : super(AuthState());

  Future<void> login(String email, String password) async {
    state = state.copyWith(loading: true);
    try {
      // Replace with actual API call
      final response = await http.post(
        Uri.parse('/api/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'email': email, 'password': password}),
      );
      final data = jsonDecode(response.body);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('auth_token', data['token']);
      state = AuthState(
        token: data['token'],
        user: User.fromJson(data['user']),
      );
    } finally {
      state = state.copyWith(loading: false);
    }
  }

  Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('auth_token');
    state = AuthState();
  }

  Future<void> checkAuth() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('auth_token');
    if (token == null) return;

    try {
      final response = await http.get(
        Uri.parse('/api/auth/me'),
        headers: {'Authorization': 'Bearer $token'},
      );
      if (response.statusCode == 200) {
        state = AuthState(
          token: token,
          user: User.fromJson(jsonDecode(response.body)),
        );
      } else {
        await logout();
      }
    } catch (_) {
      await logout();
    }
  }
}

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  return AuthNotifier();
});
".

generate_auth_store(swiftui, Code) :-
    Code = "import Foundation
import Combine

struct User: Codable {
    let id: String
    let email: String
    let roles: [String]
    let permissions: [String]
}

@MainActor
class AuthStore: ObservableObject {
    @Published var user: User?
    @Published var token: String?
    @Published var loading = false

    private let tokenKey = \"auth_token\"

    var isAuthenticated: Bool {
        token != nil && user != nil
    }

    func hasRole(_ role: String) -> Bool {
        user?.roles.contains(role) ?? false
    }

    func hasPermission(_ permission: String) -> Bool {
        user?.permissions.contains(permission) ?? false
    }

    func login(email: String, password: String) async throws {
        loading = true
        defer { loading = false }

        // Replace with actual API call
        guard let url = URL(string: \"/api/auth/login\") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = \"POST\"
        request.setValue(\"application/json\", forHTTPHeaderField: \"Content-Type\")
        request.httpBody = try JSONEncoder().encode([\"email\": email, \"password\": password])

        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(LoginResponse.self, from: data)

        token = response.token
        user = response.user
        UserDefaults.standard.set(response.token, forKey: tokenKey)
    }

    func logout() {
        token = nil
        user = nil
        UserDefaults.standard.removeObject(forKey: tokenKey)
    }

    func checkAuth() async {
        guard let storedToken = UserDefaults.standard.string(forKey: tokenKey) else { return }
        token = storedToken

        guard let url = URL(string: \"/api/auth/me\") else { return }
        var request = URLRequest(url: url)
        request.setValue(\"Bearer \\(storedToken)\", forHTTPHeaderField: \"Authorization\")

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                user = try JSONDecoder().decode(User.self, from: data)
            } else {
                logout()
            }
        } catch {
            logout()
        }
    }
}

struct LoginResponse: Codable {
    let token: String
    let user: User
}
".

% ============================================================================
% SECURE AUTH BACKEND GENERATION
% ============================================================================

%! generate_auth_backend(+AppSpec, +Target, -Files)
%
%  Generate secure authentication backend with:
%  - Password hashing (bcrypt)
%  - JWT token generation/verification
%  - User model with roles/permissions
%  - SQLAlchemy database integration
%
generate_auth_backend(AppSpec, fastapi, Files) :-
    (   app_requires_secure_auth(AppSpec)
    ->  generate_fastapi_auth_main(MainCode),
        generate_fastapi_auth_models(ModelsCode),
        generate_fastapi_auth_schemas(SchemasCode),
        generate_fastapi_auth_utils(UtilsCode),
        generate_fastapi_auth_routes(RoutesCode),
        generate_fastapi_auth_deps(DepsCode),
        generate_fastapi_requirements(ReqsCode),
        Files = [
            file('/main.py', MainCode),
            file('/models.py', ModelsCode),
            file('/schemas.py', SchemasCode),
            file('/auth_utils.py', UtilsCode),
            file('/routes/auth.py', RoutesCode),
            file('/dependencies.py', DepsCode),
            file('/requirements.txt', ReqsCode)
        ]
    ;   Files = []
    ).

generate_auth_backend(AppSpec, flask, Files) :-
    (   app_requires_secure_auth(AppSpec)
    ->  generate_flask_auth_app(AppCode),
        generate_flask_auth_models(ModelsCode),
        generate_flask_requirements(ReqsCode),
        Files = [
            file('/app.py', AppCode),
            file('/models.py', ModelsCode),
            file('/requirements.txt', ReqsCode)
        ]
    ;   Files = []
    ).

generate_auth_backend(_, _, []).

%! generate_fastapi_auth_main(-Code)
generate_fastapi_auth_main(Code) :-
    Code = "\"\"\"
FastAPI Application with Secure Authentication
Generated by UnifyWeaver
\"\"\"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.auth import router as auth_router
import uvicorn

app = FastAPI(
    title=\"Secure API\",
    description=\"API with JWT authentication\",
    version=\"1.0.0\"
)

# CORS - configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"https://localhost:5173\", \"http://localhost:5173\"],
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

# Include auth routes
app.include_router(auth_router, prefix=\"/api/auth\", tags=[\"Authentication\"])

@app.get(\"/health\")
async def health_check():
    return {\"status\": \"healthy\"}

if __name__ == \"__main__\":
    uvicorn.run(
        \"main:app\",
        host=\"0.0.0.0\",
        port=8000,
        ssl_keyfile=\"certs/localhost-key.pem\",
        ssl_certfile=\"certs/localhost.pem\",
        reload=True
    )
".

%! generate_fastapi_auth_models(-Code)
generate_fastapi_auth_models(Code) :-
    Code = "\"\"\"
SQLAlchemy User Model
\"\"\"
from sqlalchemy import Column, String, Boolean, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

DATABASE_URL = \"sqlite:///./app.db\"  # Use PostgreSQL in production

engine = create_engine(DATABASE_URL, connect_args={\"check_same_thread\": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = \"users\"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    roles = Column(JSON, default=[\"user\"])
    permissions = Column(JSON, default=[])
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def has_role(self, role: str) -> bool:
        return role in (self.roles or [])

    def has_permission(self, permission: str) -> bool:
        return permission in (self.permissions or [])


# Create tables
Base.metadata.create_all(bind=engine)
".

%! generate_fastapi_auth_schemas(-Code)
generate_fastapi_auth_schemas(Code) :-
    Code = "\"\"\"
Pydantic Schemas for Authentication
\"\"\"
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    roles: List[str]
    permissions: List[str]

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = \"bearer\"
    user: UserResponse


class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user: Optional[UserResponse] = None
    message: Optional[str] = None
".

%! generate_fastapi_auth_utils(-Code)
generate_fastapi_auth_utils(Code) :-
    Code = "\"\"\"
Authentication Utilities - Password Hashing & JWT
\"\"\"
from datetime import datetime, timedelta
from typing import Optional
import os
from passlib.context import CryptContext
from jose import JWTError, jwt

# Password hashing
pwd_context = CryptContext(schemes=[\"bcrypt\"], deprecated=\"auto\")

# JWT settings - USE ENVIRONMENT VARIABLES IN PRODUCTION
SECRET_KEY = os.getenv(\"JWT_SECRET_KEY\", \"your-secret-key-change-in-production\")
ALGORITHM = \"HS256\"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


def verify_password(plain_password: str, hashed_password: str) -> bool:
    \"\"\"Verify a password against its hash.\"\"\"
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    \"\"\"Hash a password using bcrypt.\"\"\"
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    \"\"\"Create a JWT access token.\"\"\"
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({\"exp\": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    \"\"\"Decode and verify a JWT token.\"\"\"
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
".

%! generate_fastapi_auth_routes(-Code)
generate_fastapi_auth_routes(Code) :-
    Code = "\"\"\"
Authentication Routes
\"\"\"
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from dependencies import get_db, get_current_user
from models import User
from schemas import UserCreate, UserLogin, TokenResponse, AuthResponse, UserResponse
from auth_utils import verify_password, get_password_hash, create_access_token

router = APIRouter()


@router.post(\"/register\", response_model=AuthResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    \"\"\"Register a new user.\"\"\"
    # Check if user exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=\"Email already registered\"
        )

    # Create user with hashed password
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        roles=[\"user\"],
        permissions=[\"read\"]
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate token
    token = create_access_token({\"sub\": user.id, \"email\": user.email})

    return AuthResponse(
        success=True,
        token=token,
        user=UserResponse.model_validate(user),
        message=\"Registration successful\"
    )


@router.post(\"/login\", response_model=AuthResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    \"\"\"Authenticate user and return JWT token.\"\"\"
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=\"Invalid email or password\",
            headers={\"WWW-Authenticate\": \"Bearer\"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=\"Account is disabled\"
        )

    token = create_access_token({\"sub\": user.id, \"email\": user.email})

    return AuthResponse(
        success=True,
        token=token,
        user=UserResponse.model_validate(user)
    )


@router.get(\"/me\", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    \"\"\"Get current authenticated user info.\"\"\"
    return UserResponse.model_validate(current_user)


@router.post(\"/logout\")
async def logout():
    \"\"\"Logout - client should discard token.\"\"\"
    return {\"success\": True, \"message\": \"Logged out successfully\"}
".

%! generate_fastapi_auth_deps(-Code)
generate_fastapi_auth_deps(Code) :-
    Code = "\"\"\"
FastAPI Dependencies
\"\"\"
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from models import SessionLocal, User
from auth_utils import decode_token

security = HTTPBearer()


def get_db():
    \"\"\"Database session dependency.\"\"\"
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    \"\"\"Get current authenticated user from JWT token.\"\"\"
    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=\"Invalid or expired token\",
            headers={\"WWW-Authenticate\": \"Bearer\"}
        )

    user_id = payload.get(\"sub\")
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=\"User not found\"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=\"Account is disabled\"
        )

    return user


def require_role(role: str):
    \"\"\"Dependency to require a specific role.\"\"\"
    async def role_checker(current_user: User = Depends(get_current_user)):
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f\"Role '{role}' required\"
            )
        return current_user
    return role_checker


def require_permission(permission: str):
    \"\"\"Dependency to require a specific permission.\"\"\"
    async def permission_checker(current_user: User = Depends(get_current_user)):
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f\"Permission '{permission}' required\"
            )
        return current_user
    return permission_checker
".

%! generate_fastapi_requirements(-Code)
generate_fastapi_requirements(Code) :-
    Code = "fastapi>=0.100.0
uvicorn[standard]>=0.22.0
sqlalchemy>=2.0.0
pydantic[email]>=2.0.0
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0
python-multipart>=0.0.6
".

%! generate_flask_auth_app(-Code)
generate_flask_auth_app(Code) :-
    Code = "\"\"\"
Flask Application with Secure Authentication
Generated by UnifyWeaver
\"\"\"
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from functools import wraps
import jwt
import os
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, origins=['https://localhost:5173', 'http://localhost:5173'], supports_credentials=True)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

from models import User


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'success': False, 'message': 'Token required'}), 401
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(payload['sub'])
            if not current_user or not current_user.is_active:
                return jsonify({'success': False, 'message': 'Invalid user'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'message': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'message': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated


@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'message': 'Email already registered'}), 400

    user = User(
        email=data['email'],
        password_hash=bcrypt.generate_password_hash(data['password']).decode('utf-8'),
        roles=['user'],
        permissions=['read']
    )
    db.session.add(user)
    db.session.commit()

    token = jwt.encode(
        {'sub': user.id, 'email': user.email, 'exp': datetime.utcnow() + timedelta(days=1)},
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )

    return jsonify({
        'success': True,
        'token': token,
        'user': user.to_dict(),
        'message': 'Registration successful'
    })


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()

    if not user or not bcrypt.check_password_hash(user.password_hash, data['password']):
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    if not user.is_active:
        return jsonify({'success': False, 'message': 'Account disabled'}), 403

    token = jwt.encode(
        {'sub': user.id, 'email': user.email, 'exp': datetime.utcnow() + timedelta(days=1)},
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )

    return jsonify({
        'success': True,
        'token': token,
        'user': user.to_dict()
    })


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_me(current_user):
    return jsonify(current_user.to_dict())


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(
        host='0.0.0.0',
        port=8000,
        ssl_context=('certs/localhost.pem', 'certs/localhost-key.pem'),
        debug=True
    )
".

%! generate_flask_auth_models(-Code)
generate_flask_auth_models(Code) :-
    Code = "\"\"\"
Flask-SQLAlchemy User Model
\"\"\"
from app import db
from datetime import datetime
import uuid


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    roles = db.Column(db.JSON, default=['user'])
    permissions = db.Column(db.JSON, default=[])
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def has_role(self, role):
        return role in (self.roles or [])

    def has_permission(self, permission):
        return permission in (self.permissions or [])

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'roles': self.roles or [],
            'permissions': self.permissions or []
        }
".

%! generate_flask_requirements(-Code)
generate_flask_requirements(Code) :-
    Code = "flask>=2.3.0
flask-cors>=4.0.0
flask-sqlalchemy>=3.0.0
flask-bcrypt>=1.0.1
PyJWT>=2.8.0
".

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
