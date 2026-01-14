% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% project_generator.pl - Generate Complete Project Files
%
% Generates actual project directories and files from pattern specifications.
% Supports all frontend targets and Python backends.
%
% Features:
%   - Create directory structures for each target
%   - Generate config files (package.json, requirements.txt, etc.)
%   - Write component/screen files
%   - Generate navigation, stores, and API clients
%   - Full-stack project generation (frontend + backend)
%
% Usage:
%   ?- generate_project(AppSpec, react_native, '/output/myapp', Result).
%   ?- generate_full_stack_project(AppSpec, vue, fastapi, '/output/myapp', Result).

:- module(project_generator, [
    % Project generation
    generate_project/4,                    % +AppSpec, +Target, +OutputDir, -Result
    generate_full_stack_project/5,         % +AppSpec, +FrontendTarget, +BackendTarget, +OutputDir, -Result

    % Directory structure
    create_directory_structure/3,          % +Target, +OutputDir, -Directories
    ensure_directory/1,                    % +Path

    % File writing
    write_project_file/3,                  % +Path, +Content, +Options
    write_project_files/2,                 % +FileSpecs, +OutputDir

    % Config generation
    generate_package_json/3,               % +AppName, +Target, -JSON
    generate_requirements_txt/2,           % +Backend, -Content
    generate_pubspec_yaml/2,               % +AppName, -YAML
    generate_tsconfig/1,                   % -JSON

    % Component generation
    generate_component_file/4,             % +Name, +Target, +PatternSpec, -Content
    generate_screen_file/4,                % +Name, +Target, +PatternSpec, -Content
    generate_store_file/4,                 % +Name, +Target, +PatternSpec, -Content
    generate_api_client_file/3,            % +Target, +Endpoints, -Content

    % Testing
    test_project_generator/0
]).

:- use_module(library(lists)).

% Try to load pattern modules
:- catch(use_module('../patterns/ui_patterns'), _, true).
:- catch(use_module('./pattern_glue'), _, true).
:- catch(use_module('./fastapi_generator'), _, true).
:- catch(use_module('./flask_generator'), _, true).

% ============================================================================
% PROJECT GENERATION
% ============================================================================

%% generate_project(+AppSpec, +Target, +OutputDir, -Result)
%
%  Generate a complete frontend project.
%
%  AppSpec: app(Name, Patterns, Options)
%  Target: react_native | vue | flutter | swiftui
%
generate_project(app(Name, Patterns, Options), Target, OutputDir, Result) :-
    atom_string(Name, NameStr),
    format('Generating ~w project: ~w~n', [Target, NameStr]),

    % Create directory structure
    create_directory_structure(Target, OutputDir, Dirs),

    % Generate and write files
    generate_project_files(Name, Patterns, Target, Options, OutputDir, Files),

    Result = project_result(Name, Target, OutputDir, Dirs, Files).

%% generate_full_stack_project(+AppSpec, +FrontendTarget, +BackendTarget, +OutputDir, -Result)
%
%  Generate a complete full-stack project with frontend and backend.
%
generate_full_stack_project(app(Name, Patterns, Options), FrontendTarget, BackendTarget, OutputDir, Result) :-
    atom_string(Name, NameStr),
    format('Generating full-stack project: ~w~n', [NameStr]),
    format('  Frontend: ~w~n', [FrontendTarget]),
    format('  Backend: ~w~n', [BackendTarget]),

    % Generate frontend
    atom_concat(OutputDir, '/frontend', FrontendDir),
    generate_project(app(Name, Patterns, Options), FrontendTarget, FrontendDir, FrontendResult),

    % Generate backend
    atom_concat(OutputDir, '/backend', BackendDir),
    generate_backend_project(Name, Patterns, BackendTarget, Options, BackendDir, BackendResult),

    Result = full_stack_result(FrontendResult, BackendResult).

%% generate_backend_project(+Name, +Patterns, +Target, +Options, +OutputDir, -Result)
%
%  Generate backend project files.
%
generate_backend_project(Name, Patterns, fastapi, _Options, OutputDir, Result) :-
    ensure_directory(OutputDir),
    atom_concat(OutputDir, '/app', AppDir),
    ensure_directory(AppDir),

    % Generate requirements.txt
    generate_requirements_txt(fastapi, RequirementsContent),
    atom_concat(OutputDir, '/requirements.txt', RequirementsPath),
    write_project_file(RequirementsPath, RequirementsContent, []),

    % Generate main.py
    fastapi_generator:generate_fastapi_app(Patterns, [], MainPyContent),
    atom_concat(OutputDir, '/main.py', MainPyPath),
    write_project_file(MainPyPath, MainPyContent, []),

    % Generate __init__.py
    atom_concat(AppDir, '/__init__.py', InitPath),
    write_project_file(InitPath, "# App package", []),

    Result = backend_result(fastapi, OutputDir, [RequirementsPath, MainPyPath, InitPath]).

generate_backend_project(Name, Patterns, flask, _Options, OutputDir, Result) :-
    ensure_directory(OutputDir),

    % Generate requirements.txt
    generate_requirements_txt(flask, RequirementsContent),
    atom_concat(OutputDir, '/requirements.txt', RequirementsPath),
    write_project_file(RequirementsPath, RequirementsContent, []),

    % Generate app.py
    flask_generator:generate_flask_app(Patterns, [], AppPyContent),
    atom_concat(OutputDir, '/app.py', AppPyPath),
    write_project_file(AppPyPath, AppPyContent, []),

    Result = backend_result(flask, OutputDir, [RequirementsPath, AppPyPath]).

generate_backend_project(Name, Patterns, Target, Options, OutputDir, Result) :-
    Target \= fastapi,
    Target \= flask,
    % Fallback for Express/Go - generate single file
    ensure_directory(OutputDir),
    pattern_glue:generate_backend_code(Patterns, Target, Options, Code),
    (   Target = express
    ->  FileName = '/routes.js'
    ;   FileName = '/handlers.go'
    ),
    atom_concat(OutputDir, FileName, FilePath),
    write_project_file(FilePath, Code, []),
    Result = backend_result(Target, OutputDir, [FilePath]).

% ============================================================================
% DIRECTORY STRUCTURE
% ============================================================================

%% create_directory_structure(+Target, +OutputDir, -Directories)
%
%  Create the directory structure for a target.
%
create_directory_structure(react_native, OutputDir, Dirs) :-
    Subdirs = ['src', 'src/components', 'src/screens', 'src/hooks',
               'src/stores', 'src/api', 'src/navigation', 'src/types'],
    create_subdirectories(OutputDir, Subdirs, Dirs).

create_directory_structure(vue, OutputDir, Dirs) :-
    Subdirs = ['src', 'src/components', 'src/views', 'src/stores',
               'src/composables', 'src/router', 'src/api', 'src/types',
               'src/navigation', 'src/styles', 'src/locales', 'public'],
    create_subdirectories(OutputDir, Subdirs, Dirs).

create_directory_structure(flutter, OutputDir, Dirs) :-
    Subdirs = ['lib', 'lib/screens', 'lib/widgets', 'lib/providers',
               'lib/models', 'lib/services', 'test'],
    create_subdirectories(OutputDir, Subdirs, Dirs).

create_directory_structure(swiftui, OutputDir, Dirs) :-
    Subdirs = ['Sources', 'Sources/Views', 'Sources/Models',
               'Sources/ViewModels', 'Sources/Services', 'Tests'],
    create_subdirectories(OutputDir, Subdirs, Dirs).

create_directory_structure(fastapi, OutputDir, Dirs) :-
    Subdirs = ['app', 'app/routes', 'app/models', 'app/schemas',
               'app/services', 'tests'],
    create_subdirectories(OutputDir, Subdirs, Dirs).

create_directory_structure(flask, OutputDir, Dirs) :-
    Subdirs = ['app', 'app/routes', 'app/models', 'tests'],
    create_subdirectories(OutputDir, Subdirs, Dirs).

create_subdirectories(BaseDir, Subdirs, AllDirs) :-
    ensure_directory(BaseDir),
    findall(FullPath, (
        member(Sub, Subdirs),
        atomic_list_concat([BaseDir, '/', Sub], FullPath),
        ensure_directory(FullPath)
    ), SubPaths),
    AllDirs = [BaseDir | SubPaths].

%% ensure_directory(+Path)
%
%  Create directory if it doesn't exist.
%
ensure_directory(Path) :-
    (   atom(Path) -> PathStr = Path ; atom_string(PathStr, Path) ),
    catch(
        (   exists_directory(PathStr)
        ->  true
        ;   make_directory_path(PathStr)
        ),
        _,
        true  % Ignore errors (directory may already exist)
    ).

% ============================================================================
% FILE WRITING
% ============================================================================

%% write_project_file(+Path, +Content, +Options)
%
%  Write content to a file.
%
write_project_file(Path, Content, _Options) :-
    (   atom(Path) -> PathAtom = Path ; atom_string(PathAtom, Path) ),
    (   atom(Content) -> ContentStr = Content
    ;   string(Content) -> ContentStr = Content
    ;   term_string(Content, ContentStr)
    ),
    catch(
        setup_call_cleanup(
            open(PathAtom, write, Stream, [encoding(utf8)]),
            write(Stream, ContentStr),
            close(Stream)
        ),
        Error,
        format('Error writing file ~w: ~w~n', [Path, Error])
    ).

%% write_project_files(+FileSpecs, +OutputDir)
%
%  Write multiple files from specifications.
%
write_project_files([], _).
write_project_files([file(RelPath, Content)|Rest], OutputDir) :-
    atomic_list_concat([OutputDir, '/', RelPath], FullPath),
    ensure_parent_directory(FullPath),
    write_project_file(FullPath, Content, []),
    write_project_files(Rest, OutputDir).

%% ensure_parent_directory(+FilePath)
%
%  Ensure the parent directory of a file exists.
%
ensure_parent_directory(FilePath) :-
    (   atom(FilePath) -> PathAtom = FilePath ; atom_string(PathAtom, FilePath) ),
    file_directory_name(PathAtom, DirPath),
    ensure_directory(DirPath).

% ============================================================================
% CONFIG FILE GENERATION
% ============================================================================

%% generate_package_json(+AppName, +Target, -JSON)
%
%  Generate package.json content for React Native or Vue.
%
generate_package_json(AppName, react_native, JSON) :-
    atom_string(AppName, NameStr),
    format(string(JSON),
'{
  "name": "~w",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "start": "expo start",
    "android": "expo start --android",
    "ios": "expo start --ios",
    "web": "expo start --web"
  },
  "dependencies": {
    "expo": "~~49.0.0",
    "react": "18.2.0",
    "react-native": "0.72.0",
    "@react-navigation/native": "^6.1.0",
    "@react-navigation/bottom-tabs": "^6.5.0",
    "@react-navigation/native-stack": "^6.9.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "@react-native-async-storage/async-storage": "^1.19.0",
    "react-hook-form": "^7.48.0",
    "zod": "^3.22.0",
    "@hookform/resolvers": "^3.3.0"
  },
  "devDependencies": {
    "@types/react": "~~18.2.0",
    "typescript": "^5.0.0"
  }
}', [NameStr]).

generate_package_json(AppName, vue, JSON) :-
    atom_string(AppName, NameStr),
    format(string(JSON),
'{
  "name": "~w",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.3.0",
    "vue-router": "^4.2.0",
    "pinia": "^2.1.0",
    "@tanstack/vue-query": "^5.0.0",
    "vee-validate": "^4.11.0",
    "zod": "^3.22.0",
    "@vee-validate/zod": "^4.11.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^4.4.0",
    "typescript": "^5.0.0",
    "vite": "^4.5.0",
    "vue-tsc": "^1.8.0"
  }
}', [NameStr]).

%% generate_requirements_txt(+Backend, -Content)
%
%  Generate Python requirements.txt content.
%
generate_requirements_txt(fastapi, Content) :-
    Content = "fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
sqlalchemy>=2.0.0
alembic>=1.12.0
httpx>=0.25.0
pytest>=7.4.0
pytest-asyncio>=0.21.0".

generate_requirements_txt(flask, Content) :-
    Content = "flask>=3.0.0
flask-cors>=4.0.0
flask-sqlalchemy>=3.1.0
flask-migrate>=4.0.0
python-dotenv>=1.0.0
pyjwt>=2.8.0
bcrypt>=4.1.0
pytest>=7.4.0
pytest-flask>=1.3.0".

%% generate_pubspec_yaml(+AppName, -YAML)
%
%  Generate Flutter pubspec.yaml content.
%
generate_pubspec_yaml(AppName, YAML) :-
    atom_string(AppName, NameStr),
    format(string(YAML),
'name: ~w
description: A Flutter application generated by UnifyWeaver
version: 1.0.0+1

environment:
  sdk: ">=3.0.0 <4.0.0"

dependencies:
  flutter:
    sdk: flutter
  go_router: ^12.0.0
  flutter_riverpod: ^2.4.0
  riverpod_annotation: ^2.3.0
  dio: ^5.4.0
  shared_preferences: ^2.2.0
  freezed_annotation: ^2.4.0
  json_annotation: ^4.8.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  build_runner: ^2.4.0
  riverpod_generator: ^2.3.0
  freezed: ^2.4.0
  json_serializable: ^6.7.0
  flutter_lints: ^3.0.0

flutter:
  uses-material-design: true
', [NameStr]).

%% generate_tsconfig(-JSON)
%
%  Generate TypeScript configuration.
%
generate_tsconfig(JSON) :-
    JSON = '{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "jsx": "react-jsx",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"]
}'.

% ============================================================================
% COMPONENT/FILE GENERATION
% ============================================================================

%% generate_project_files(+Name, +Patterns, +Target, +Options, +OutputDir, -Files)
%
%  Generate all project files for a target.
%
generate_project_files(Name, _Patterns, react_native, _Options, OutputDir, Files) :-
    % Generate package.json
    generate_package_json(Name, react_native, PackageJson),
    atom_concat(OutputDir, '/package.json', PackageJsonPath),
    write_project_file(PackageJsonPath, PackageJson, []),

    % Generate tsconfig.json
    generate_tsconfig(TsConfig),
    atom_concat(OutputDir, '/tsconfig.json', TsConfigPath),
    write_project_file(TsConfigPath, TsConfig, []),

    % Generate App.tsx
    generate_react_native_app(Name, AppContent),
    atom_concat(OutputDir, '/App.tsx', AppPath),
    write_project_file(AppPath, AppContent, []),

    Files = [PackageJsonPath, TsConfigPath, AppPath].

generate_project_files(Name, _Patterns, vue, _Options, OutputDir, Files) :-
    % Generate package.json
    generate_package_json(Name, vue, PackageJson),
    atom_concat(OutputDir, '/package.json', PackageJsonPath),
    write_project_file(PackageJsonPath, PackageJson, []),

    % Generate tsconfig.json
    generate_tsconfig(TsConfig),
    atom_concat(OutputDir, '/tsconfig.json', TsConfigPath),
    write_project_file(TsConfigPath, TsConfig, []),

    % Generate main.ts
    generate_vue_main(Name, MainContent),
    atom_concat(OutputDir, '/src/main.ts', MainPath),
    write_project_file(MainPath, MainContent, []),

    Files = [PackageJsonPath, TsConfigPath, MainPath].

generate_project_files(Name, _Patterns, flutter, _Options, OutputDir, Files) :-
    % Generate pubspec.yaml
    generate_pubspec_yaml(Name, PubspecYaml),
    atom_concat(OutputDir, '/pubspec.yaml', PubspecPath),
    write_project_file(PubspecPath, PubspecYaml, []),

    % Generate main.dart
    generate_flutter_main(Name, MainContent),
    atom_concat(OutputDir, '/lib/main.dart', MainPath),
    write_project_file(MainPath, MainContent, []),

    Files = [PubspecPath, MainPath].

generate_project_files(Name, _Patterns, swiftui, _Options, OutputDir, Files) :-
    % Generate Package.swift
    generate_swift_package(Name, PackageSwift),
    atom_concat(OutputDir, '/Package.swift', PackagePath),
    write_project_file(PackagePath, PackageSwift, []),

    % Generate App.swift
    generate_swiftui_app(Name, AppContent),
    atom_concat(OutputDir, '/Sources/App.swift', AppPath),
    write_project_file(AppPath, AppContent, []),

    Files = [PackagePath, AppPath].

%% generate_react_native_app(+Name, -Content)
generate_react_native_app(Name, Content) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Content),
"import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <NavigationContainer>
        {/* Add your navigation here */}
        <></>
      </NavigationContainer>
    </QueryClientProvider>
  );
}
", []).

%% generate_vue_main(+Name, -Content)
generate_vue_main(Name, Content) :-
    atom_string(Name, NameStr),
    format(string(Content),
"import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { VueQueryPlugin } from '@tanstack/vue-query';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';

const pinia = createPinia();

const router = createRouter({
  history: createWebHistory(),
  routes: [
    // Add your routes here
  ]
});

const app = createApp(App);
app.use(pinia);
app.use(router);
app.use(VueQueryPlugin);
app.mount('#app');
", []).

%% generate_flutter_main(+Name, -Content)
generate_flutter_main(Name, Content) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Content),
"import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

void main() {
  runApp(
    const ProviderScope(
      child: ~wApp(),
    ),
  );
}

class ~wApp extends StatelessWidget {
  const ~wApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: '~w',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      routerConfig: _router,
    );
  }
}

final _router = GoRouter(
  routes: [
    // Add your routes here
  ],
);
", [CapName, CapName, CapName, NameStr]).

%% generate_swift_package(+Name, -Content)
generate_swift_package(Name, Content) :-
    atom_string(Name, NameStr),
    format(string(Content),
"// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: \"~w\",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: \"~w\", targets: [\"~w\"])
    ],
    targets: [
        .target(name: \"~w\", path: \"Sources\"),
        .testTarget(name: \"~wTests\", dependencies: [\"~w\"], path: \"Tests\")
    ]
)
", [NameStr, NameStr, NameStr, NameStr, NameStr, NameStr]).

%% generate_swiftui_app(+Name, -Content)
generate_swiftui_app(Name, Content) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Content),
"import SwiftUI

@main
struct ~wApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        NavigationStack {
            Text(\"Welcome to ~w\")
                .navigationTitle(\"~w\")
        }
    }
}

#Preview {
    ContentView()
}
", [CapName, NameStr, NameStr]).

%% generate_component_file(+Name, +Target, +PatternSpec, -Content)
generate_component_file(Name, react_native, _, Content) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Content),
"import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface ~wProps {
  // Add props here
}

export const ~w: React.FC<~wProps> = (props) => {
  return (
    <View style={styles.container}>
      <Text>~w Component</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
", [CapName, CapName, CapName, CapName]).

%% generate_screen_file(+Name, +Target, +PatternSpec, -Content)
generate_screen_file(Name, react_native, _, Content) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Content),
"import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export const ~wScreen: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>~w</Text>
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
", [CapName, CapName]).

%% generate_store_file(+Name, +Target, +PatternSpec, -Content)
generate_store_file(Name, react_native, _, Content) :-
    atom_string(Name, NameStr),
    format(string(Content),
"import { create } from 'zustand';

interface ~wState {
  // Add state fields here
}

interface ~wActions {
  // Add actions here
}

export const use~wStore = create<~wState & ~wActions>((set) => ({
  // Initial state

  // Actions
}));
", [NameStr, NameStr, NameStr, NameStr, NameStr]).

%% generate_api_client_file(+Target, +Endpoints, -Content)
generate_api_client_file(react_native, _Endpoints, Content) :-
    Content = "import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60000,
      retry: 3,
    },
  },
});

const BASE_URL = process.env.API_URL || 'http://localhost:8000';

export const api = {
  get: async <T>(endpoint: string, params?: Record<string, string>): Promise<T> => {
    const url = new URL(`${BASE_URL}${endpoint}`);
    if (params) {
      Object.entries(params).forEach(([key, value]) =>
        url.searchParams.append(key, value)
      );
    }
    const response = await fetch(url.toString());
    if (!response.ok) throw new Error('Request failed');
    return response.json();
  },

  post: async <T>(endpoint: string, body: unknown): Promise<T> => {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error('Request failed');
    return response.json();
  },
};
".

% ============================================================================
% UTILITIES
% ============================================================================

capitalize_first(Str, Cap) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HC]),
    string_chars(Cap, [HC|T]).

% ============================================================================
% TESTING
% ============================================================================

test_project_generator :-
    format('~n=== Project Generator Tests ===~n~n'),

    % Test 1: Package.json generation
    format('Test 1: Package.json generation...~n'),
    generate_package_json(testapp, react_native, PkgJson),
    (   sub_string(PkgJson, _, _, _, "testapp"),
        sub_string(PkgJson, _, _, _, "zustand")
    ->  format('  PASS: Generated React Native package.json~n')
    ;   format('  FAIL: Package.json generation failed~n')
    ),

    % Test 2: Requirements.txt generation
    format('~nTest 2: Requirements.txt generation...~n'),
    generate_requirements_txt(fastapi, Reqs),
    (   sub_string(Reqs, _, _, _, "fastapi"),
        sub_string(Reqs, _, _, _, "uvicorn")
    ->  format('  PASS: Generated FastAPI requirements.txt~n')
    ;   format('  FAIL: Requirements.txt generation failed~n')
    ),

    % Test 3: Pubspec.yaml generation
    format('~nTest 3: Pubspec.yaml generation...~n'),
    generate_pubspec_yaml(testapp, Pubspec),
    (   sub_string(Pubspec, _, _, _, "flutter"),
        sub_string(Pubspec, _, _, _, "riverpod")
    ->  format('  PASS: Generated Flutter pubspec.yaml~n')
    ;   format('  FAIL: Pubspec.yaml generation failed~n')
    ),

    % Test 4: React Native App generation
    format('~nTest 4: React Native App generation...~n'),
    generate_react_native_app(testapp, RNApp),
    (   sub_string(RNApp, _, _, _, "QueryClientProvider"),
        sub_string(RNApp, _, _, _, "NavigationContainer")
    ->  format('  PASS: Generated React Native App~n')
    ;   format('  FAIL: React Native App generation failed~n')
    ),

    % Test 5: Flutter main generation
    format('~nTest 5: Flutter main generation...~n'),
    generate_flutter_main(testapp, FlutterMain),
    (   sub_string(FlutterMain, _, _, _, "ProviderScope"),
        sub_string(FlutterMain, _, _, _, "GoRouter")
    ->  format('  PASS: Generated Flutter main.dart~n')
    ;   format('  FAIL: Flutter main generation failed~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Project generator module loaded~n', [])
), now).
