%! project_scaffold.pl - Generate complete runnable project scaffolds
%
%  Generates complete project structures for each UI target:
%  - Vue.js: package.json, vite.config.ts, src/, components/
%  - React: package.json, vite.config.ts, src/, components/
%  - Flutter: pubspec.yaml, lib/, widgets/
%  - TUI/Shell: Executable shell script with runner
%
%  Example usage:
%      ?- generate_project(vue, my_app, Spec, '/output/dir').
%      ?- generate_project(react, my_app, Spec, '/output/dir').
%      ?- generate_project(flutter, my_app, Spec, '/output/dir').
%      ?- generate_project(tui, my_app, Spec, '/output/dir').
%
%  @author UnifyWeaver
%  @version 1.0.0

:- module(project_scaffold, [
    generate_project/4,
    generate_project/5,           % With options (e.g., mode(dialog) for TUI)
    generate_project_files/4,
    generate_project_files/5,     % With options
    project_file/4,
    test_project_scaffold/0
]).

:- use_module(library(lists)).

% Load the UI generators
:- use_module(vue_generator, [generate_vue_template/2]).
:- use_module(react_generator, [generate_react_template/2]).
:- use_module(flutter_generator, [generate_flutter_widget/2]).
:- use_module(tui_generator, [generate_tui_script/3, generate_tui_script/4, generate_dialog_script/3]).

% ============================================================================
% MAIN ENTRY POINTS
% ============================================================================

%! generate_project(+Target, +AppName, +UISpec, +OutputDir) is det
%  Generate a complete project for the given target.
%  Target is one of: vue, react, flutter, tui
generate_project(Target, AppName, UISpec, OutputDir) :-
    generate_project_files(Target, AppName, UISpec, Files),
    write_project_files(OutputDir, Files).

%! generate_project(+Target, +AppName, +UISpec, +OutputDir, +Options) is det
%  Generate a project with options.
%  Options for TUI:
%    - mode(echo) : Plain ANSI echo-based TUI (default)
%    - mode(dialog) : Interactive dialog-based TUI
generate_project(Target, AppName, UISpec, OutputDir, Options) :-
    generate_project_files(Target, AppName, UISpec, Options, Files),
    write_project_files(OutputDir, Files).

%! generate_project_files(+Target, +AppName, +UISpec, -Files) is det
%  Generate all project files as a list of path-content pairs.
generate_project_files(vue, AppName, UISpec, Files) :-
    generate_vue_project(AppName, UISpec, Files).
generate_project_files(react, AppName, UISpec, Files) :-
    generate_react_project(AppName, UISpec, Files).
generate_project_files(flutter, AppName, UISpec, Files) :-
    generate_flutter_project(AppName, UISpec, Files).
generate_project_files(tui, AppName, UISpec, Files) :-
    generate_tui_project(AppName, UISpec, Files).

%! generate_project_files(+Target, +AppName, +UISpec, +Options, -Files) is det
%  Generate all project files with options.
generate_project_files(vue, AppName, UISpec, _Options, Files) :-
    generate_vue_project(AppName, UISpec, Files).
generate_project_files(react, AppName, UISpec, _Options, Files) :-
    generate_react_project(AppName, UISpec, Files).
generate_project_files(flutter, AppName, UISpec, _Options, Files) :-
    generate_flutter_project(AppName, UISpec, Files).
generate_project_files(tui, AppName, UISpec, Options, Files) :-
    generate_tui_project(AppName, UISpec, Options, Files).

%! project_file(+Target, +FileType, +AppName, -Content) is det
%  Get a specific project file content.
project_file(vue, package_json, AppName, Content) :-
    vue_package_json(AppName, Content).
project_file(vue, vite_config, _AppName, Content) :-
    vue_vite_config(Content).
project_file(react, package_json, AppName, Content) :-
    react_package_json(AppName, Content).
project_file(react, vite_config, _AppName, Content) :-
    react_vite_config(Content).
project_file(flutter, pubspec, AppName, Content) :-
    flutter_pubspec(AppName, Content).

% ============================================================================
% VUE.JS PROJECT GENERATION
% ============================================================================

generate_vue_project(AppName, UISpec, Files) :-
    % Generate component template
    generate_vue_template(UISpec, Template),
    wrap_vue_sfc(Template, AppComponent),

    % Package files
    vue_package_json(AppName, PackageJson),
    vue_vite_config(ViteConfig),
    vue_tsconfig(TsConfig),
    vue_index_html(AppName, IndexHtml),
    vue_main_ts(MainTs),
    vue_env_d_ts(EnvDTs),
    vue_gitignore(GitIgnore),
    vue_readme(AppName, Readme),

    Files = [
        'package.json'-PackageJson,
        'vite.config.ts'-ViteConfig,
        'tsconfig.json'-TsConfig,
        'index.html'-IndexHtml,
        'src/main.ts'-MainTs,
        'src/App.vue'-AppComponent,
        'src/vite-env.d.ts'-EnvDTs,
        '.gitignore'-GitIgnore,
        'README.md'-Readme
    ].

vue_package_json(AppName, Content) :-
    format(atom(Content), '{
  "name": "~w",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.4.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "vue-tsc": "^1.8.0"
  }
}
', [AppName]).

vue_vite_config(Content) :-
    Content = 'import { defineConfig } from \'vite\'
import vue from \'@vitejs/plugin-vue\'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    open: true
  }
})
'.

vue_tsconfig(Content) :-
    Content = '{
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
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
'.

vue_index_html(AppName, Content) :-
    format(atom(Content), '<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>~w</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
', [AppName]).

vue_main_ts(Content) :-
    Content = 'import { createApp } from \'vue\'
import App from \'./App.vue\'

createApp(App).mount(\'#app\')
'.

vue_env_d_ts(Content) :-
    Content = '/// <reference types="vite/client" />
'.

vue_gitignore(Content) :-
    Content = '# Dependencies
node_modules/

# Build output
dist/

# Local env files
.env.local
.env.*.local

# Editor directories
.vscode/*
!.vscode/extensions.json
.idea/

# Logs
*.log
npm-debug.log*

# OS files
.DS_Store
Thumbs.db
'.

vue_readme(AppName, Content) :-
    format(atom(Content), '# ~w

Generated by UnifyWeaver Project Scaffold Generator.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
~w/
├── src/
│   ├── App.vue        # Main application component
│   └── main.ts        # Application entry point
├── index.html         # HTML template
├── package.json       # Dependencies and scripts
├── vite.config.ts     # Vite configuration
└── tsconfig.json      # TypeScript configuration
```
', [AppName, AppName]).

% ============================================================================
% REACT PROJECT GENERATION
% ============================================================================

generate_react_project(AppName, UISpec, Files) :-
    % Generate component JSX
    generate_react_template(UISpec, Template),
    wrap_react_component(Template, AppComponent),

    % Package files
    react_package_json(AppName, PackageJson),
    react_vite_config(ViteConfig),
    react_tsconfig(TsConfig),
    react_tsconfig_node(TsConfigNode),
    react_index_html(AppName, IndexHtml),
    react_main_tsx(MainTsx),
    react_index_css(IndexCss),
    react_env_d_ts(EnvDTs),
    react_gitignore(GitIgnore),
    react_readme(AppName, Readme),

    Files = [
        'package.json'-PackageJson,
        'vite.config.ts'-ViteConfig,
        'tsconfig.json'-TsConfig,
        'tsconfig.node.json'-TsConfigNode,
        'index.html'-IndexHtml,
        'src/main.tsx'-MainTsx,
        'src/App.tsx'-AppComponent,
        'src/index.css'-IndexCss,
        'src/vite-env.d.ts'-EnvDTs,
        '.gitignore'-GitIgnore,
        'README.md'-Readme
    ].

react_package_json(AppName, Content) :-
    format(atom(Content), '{
  "name": "~w",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
', [AppName]).

react_vite_config(Content) :-
    Content = 'import { defineConfig } from \'vite\'
import react from \'@vitejs/plugin-react\'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true
  }
})
'.

react_tsconfig(Content) :-
    Content = '{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
'.

react_tsconfig_node(Content) :-
    Content = '{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
'.

react_index_html(AppName, Content) :-
    format(atom(Content), '<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>~w</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
', [AppName]).

react_main_tsx(Content) :-
    Content = 'import React from \'react\'
import ReactDOM from \'react-dom/client\'
import App from \'./App\'
import \'./index.css\'

ReactDOM.createRoot(document.getElementById(\'root\')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
'.

react_index_css(Content) :-
    Content = ':root {
  font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;

  color: #ffffffde;
  background-color: #1a1a2e;

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 2rem;
}

#root {
  width: 100%;
  max-width: 1200px;
}
'.

react_env_d_ts(Content) :-
    Content = '/// <reference types="vite/client" />
'.

react_gitignore(Content) :-
    vue_gitignore(Content).  % Same as Vue

react_readme(AppName, Content) :-
    format(atom(Content), '# ~w

Generated by UnifyWeaver Project Scaffold Generator.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
~w/
├── src/
│   ├── App.tsx        # Main application component
│   ├── main.tsx       # Application entry point
│   └── index.css      # Global styles
├── index.html         # HTML template
├── package.json       # Dependencies and scripts
├── vite.config.ts     # Vite configuration
└── tsconfig.json      # TypeScript configuration
```
', [AppName, AppName]).

% ============================================================================
% FLUTTER PROJECT GENERATION
% ============================================================================

generate_flutter_project(AppName, UISpec, Files) :-
    % Generate widget code
    generate_flutter_widget(UISpec, WidgetBody),
    wrap_flutter_widget(WidgetBody, AppWidget),

    % Package files
    flutter_pubspec(AppName, Pubspec),
    flutter_analysis_options(AnalysisOptions),
    flutter_main_dart(AppName, MainDart),
    flutter_gitignore(GitIgnore),
    flutter_readme(AppName, Readme),

    Files = [
        'pubspec.yaml'-Pubspec,
        'analysis_options.yaml'-AnalysisOptions,
        'lib/main.dart'-MainDart,
        'lib/app.dart'-AppWidget,
        '.gitignore'-GitIgnore,
        'README.md'-Readme
    ].

flutter_pubspec(AppName, Content) :-
    atom_string(AppName, AppNameStr),
    string_lower(AppNameStr, LowerName),
    format(atom(Content), 'name: ~w
description: Generated by UnifyWeaver Project Scaffold Generator.
publish_to: \'none\'
version: 0.1.0

environment:
  sdk: \'>=3.0.0 <4.0.0\'

dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.6

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0

flutter:
  uses-material-design: true
', [LowerName]).

flutter_analysis_options(Content) :-
    Content = 'include: package:flutter_lints/flutter.yaml

linter:
  rules:
    - prefer_const_constructors
    - prefer_const_literals_to_create_immutables
    - avoid_print
'.

flutter_main_dart(AppName, Content) :-
    format(atom(Content), 'import \'package:flutter/material.dart\';
import \'app.dart\';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: \'~w\',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.dark(
          primary: const Color(0xFFE94560),
          secondary: const Color(0xFF0F3460),
          surface: const Color(0xFF16213E),
          background: const Color(0xFF1A1A2E),
        ),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFF1A1A2E),
      ),
      home: const Scaffold(
        body: SafeArea(
          child: SingleChildScrollView(
            padding: EdgeInsets.all(16),
            child: App(),
          ),
        ),
      ),
    );
  }
}
', [AppName]).

flutter_gitignore(Content) :-
    Content = '# Miscellaneous
*.class
*.log
*.pyc
*.swp
.DS_Store
.atom/
.buildlog/
.history
.svn/
migrate_working_dir/

# IntelliJ related
*.iml
*.ipr
*.iws
.idea/

# Flutter/Dart/Pub related
**/doc/api/
**/ios/Flutter/.last_build_id
.dart_tool/
.flutter-plugins
.flutter-plugins-dependencies
.packages
.pub-cache/
.pub/
/build/

# Symbolication related
app.*.symbols

# Obfuscation related
app.*.map.json

# Android Studio will place build artifacts here
/android/app/debug
/android/app/profile
/android/app/release
'.

flutter_readme(AppName, Content) :-
    format(atom(Content), '# ~w

Generated by UnifyWeaver Project Scaffold Generator.

## Development

```bash
# Get dependencies
flutter pub get

# Run in debug mode
flutter run

# Build for release
flutter build apk     # Android
flutter build ios     # iOS
flutter build web     # Web
flutter build macos   # macOS
flutter build windows # Windows
flutter build linux   # Linux
```

## Project Structure

```
~w/
├── lib/
│   ├── main.dart      # Application entry point
│   └── app.dart       # Main application widget
├── pubspec.yaml       # Dependencies and configuration
├── analysis_options.yaml # Linter rules
└── README.md
```
', [AppName, AppName]).

% ============================================================================
% TUI/SHELL PROJECT GENERATION
% ============================================================================

%! generate_tui_project(+AppName, +UISpec, -Files) is det
%  Generate TUI project with default (echo) mode.
generate_tui_project(AppName, UISpec, Files) :-
    generate_tui_project(AppName, UISpec, [], Files).

%! generate_tui_project(+AppName, +UISpec, +Options, -Files) is det
%  Generate TUI project with options.
%  Options:
%    - mode(echo) : Plain ANSI echo-based TUI (default)
%    - mode(dialog) : Interactive dialog-based TUI
generate_tui_project(AppName, UISpec, Options, Files) :-
    % Determine mode
    (member(mode(Mode), Options) -> true ; Mode = echo),

    % Generate shell script based on mode
    (Mode == dialog ->
        generate_dialog_script(UISpec, AppName, ShellScript),
        tui_dialog_runner_script(AppName, Runner),
        tui_dialog_readme(AppName, Readme)
    ;
        generate_tui_script(UISpec, AppName, ShellScript),
        tui_runner_script(AppName, Runner),
        tui_readme(AppName, Readme)
    ),

    % Common files
    tui_makefile(AppName, Makefile),

    atom_concat(AppName, '.sh', ScriptName),

    Files = [
        ScriptName-ShellScript,
        'run.sh'-Runner,
        'Makefile'-Makefile,
        'README.md'-Readme
    ].

tui_runner_script(AppName, Content) :-
    atom_concat(AppName, '.sh', ScriptName),
    format(atom(Content), '#!/bin/bash
# Runner script for ~w

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for required commands
command -v tput >/dev/null 2>&1 || { echo "tput required but not installed."; exit 1; }

# Run the TUI application
exec bash "$SCRIPT_DIR/~w"
', [AppName, ScriptName]).

tui_makefile(AppName, Content) :-
    atom_concat(AppName, '.sh', ScriptName),
    format(atom(Content), '.PHONY: run clean install

# Run the TUI application
run:
\t@bash ~w

# Make scripts executable
install:
\t@chmod +x ~w
\t@chmod +x run.sh
\t@echo "Scripts are now executable"

# Clean generated files (if any)
clean:
\t@echo "Nothing to clean"
', [ScriptName, ScriptName]).

tui_readme(AppName, Content) :-
    atom_concat(AppName, '.sh', ScriptName),
    format(atom(Content), '# ~w

Generated by UnifyWeaver Project Scaffold Generator.

A terminal user interface application with ANSI styling.

## Requirements

- Bash 4.0+
- A terminal with UTF-8 and ANSI color support

## Running

```bash
# Make executable (first time only)
make install

# Run the application
make run

# Or directly:
./~w
```

## Features

- ANSI 256-color styling
- Unicode box-drawing characters
- Interactive input collection
- Theme-consistent colors

## Project Structure

```
~w/
├── ~w   # Main TUI application
├── run.sh          # Runner script
├── Makefile        # Build/run commands
└── README.md
```
', [AppName, ScriptName, AppName, ScriptName]).

%! tui_dialog_runner_script(+AppName, -Content) is det
%  Generate runner script for dialog-based TUI.
tui_dialog_runner_script(AppName, Content) :-
    atom_concat(AppName, '.sh', ScriptName),
    format(atom(Content), '#!/bin/bash
# Runner script for ~w (dialog-based)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for required commands
command -v dialog >/dev/null 2>&1 || { echo "dialog required but not installed."; echo "Install with: apt install dialog (Termux) or brew install dialog (macOS)"; exit 1; }

# Run the TUI application
exec bash "$SCRIPT_DIR/~w"
', [AppName, ScriptName]).

%! tui_dialog_readme(+AppName, -Content) is det
%  Generate README for dialog-based TUI.
tui_dialog_readme(AppName, Content) :-
    atom_concat(AppName, '.sh', ScriptName),
    format(atom(Content), '# ~w

Generated by UnifyWeaver Project Scaffold Generator.

An interactive terminal user interface application using dialog.

## Requirements

- Bash 4.0+
- dialog (ncurses-based dialog utility)
- A terminal with UTF-8 support

### Installing dialog

```bash
# Termux (Android)
apt install dialog

# Debian/Ubuntu
sudo apt install dialog

# macOS
brew install dialog

# Fedora/RHEL
sudo dnf install dialog
```

## Running

```bash
# Make executable (first time only)
make install

# Run the application
make run

# Or directly:
./~w
```

## Features

- Interactive menus and dialogs
- File selection dialogs
- Input boxes for user data
- Message boxes for notifications
- Keyboard navigation (arrow keys, Enter, Escape)

## Project Structure

```
~w/
├── ~w   # Main TUI application (dialog-based)
├── run.sh          # Runner script
├── Makefile        # Build/run commands
└── README.md
```

## Dialog Controls

- **Arrow keys**: Navigate menu items
- **Enter**: Select item / Confirm
- **Tab**: Switch between buttons (OK/Cancel)
- **Escape**: Cancel / Go back
', [AppName, ScriptName, AppName, ScriptName]).

% ============================================================================
% COMPONENT WRAPPERS
% ============================================================================

%! wrap_vue_sfc(+Template, -SFC) is det
%  Wrap a Vue template in a complete Single File Component.
wrap_vue_sfc(Template, SFC) :-
    format(atom(SFC), '<script setup lang="ts">
import { ref, reactive } from \'vue\'

// State
const loading = ref(false)
const error = ref(\'\')

// Form data (add your bindings here)
const formData = reactive({})

// Event handlers
const handleSubmit = () => {
  console.log(\'Submit clicked\')
}

const handleClick = () => {
  console.log(\'Click handler\')
}
</script>

<template>
  <div class="app-container">
~w
  </div>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
  background: #1a1a2e;
  color: #fff;
  padding: 2rem;
  font-family: system-ui, -apple-system, sans-serif;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  color: #888;
}

.form-group input,
.form-group textarea,
.form-group select {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #333;
  border-radius: 5px;
  background: #16213e;
  color: #fff;
}

button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-weight: bold;
}

button.primary {
  background: #e94560;
  color: #fff;
}

button.secondary {
  background: #333;
  color: #fff;
}
</style>
', [Template]).

%! wrap_react_component(+JSX, -Component) is det
%  Wrap React JSX in a complete functional component.
wrap_react_component(JSX, Component) :-
    format(atom(Component), 'import React, { useState } from \'react\'

// Helper function for formatting file sizes
const formatSize = (bytes: number): string => {
  if (!bytes || bytes === 0) return \'0 B\'
  const k = 1024
  const sizes = [\'B\', \'KB\', \'MB\', \'GB\']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + \' \' + sizes[i]
}

function App() {
  // State
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(\'\')

  // Browse state (example - wire up to your data source)
  const [browse, setBrowse] = useState({
    path: \'.\',
    parent: null as string | null,
    entries: [] as Array<{name: string, type: string, size: number}>,
    selected: null as string | null
  })

  const [workingDir, setWorkingDir] = useState(\'.\')

  // Event handlers
  const navigateUp = () => {
    console.log(\'Navigate up\')
  }

  const setWorkingDirTo = (path: string) => {
    setWorkingDir(path)
  }

  const viewFile = () => {
    console.log(\'View file:\', browse.selected)
  }

  const downloadFile = () => {
    console.log(\'Download file:\', browse.selected)
  }

  const searchHere = () => {
    console.log(\'Search in:\', browse.path)
  }

  return (
    <div className="app-container">
~w
    </div>
  )
}

export default App
', [JSX]).

%! wrap_flutter_widget(+WidgetBody, -Widget) is det
%  Wrap Flutter widget body in a complete StatefulWidget with state and handlers.
wrap_flutter_widget(WidgetBody, Widget) :-
    format(atom(Widget), 'import \'package:flutter/material.dart\';
import \'dart:math\';

// Helper function for formatting file sizes
String formatSize(dynamic bytes) {
  if (bytes == null || bytes == 0) return \'0 B\';
  const sizes = [\'B\', \'KB\', \'MB\', \'GB\'];
  final i = (bytes > 0) ? (log(bytes) / log(1024)).floor() : 0;
  return \'\\${(bytes / pow(1024, i)).toStringAsFixed(1)} \\${sizes[i]}\';
}

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  // Loading and error state
  bool _loading = false;
  String _error = \'\';

  // Browse state
  final Map<String, dynamic> _browse = {
    \'path\': \'.\',
    \'entries\': <Map<String, dynamic>>[
      {\'name\': \'example.txt\', \'type\': \'file\', \'size\': 1024},
      {\'name\': \'folder\', \'type\': \'directory\', \'size\': 0},
    ],
    \'selected\': null,
    \'parent\': false,
  };

  // Working directory
  String _workingDir = \'.\';

  // Controllers
  final TextEditingController _pathController = TextEditingController();

  @override
  void dispose() {
    _pathController.dispose();
    super.dispose();
  }

  // Navigation handlers
  void navigateUp() {
    setState(() {
      final parts = (_browse[\'path\'] as String).split(\'/\');
      if (parts.length > 1) {
        parts.removeLast();
        _browse[\'path\'] = parts.join(\'/\');
        if (_browse[\'path\'].isEmpty) _browse[\'path\'] = \'.\';
      }
      _browse[\'parent\'] = _browse[\'path\'] != \'.\';
    });
    debugPrint(\'Navigate up to: \\${_browse[\"path\"]}\');
  }

  void handleEntryClick(Map<String, dynamic> entry) {
    setState(() {
      if (entry[\'type\'] == \'directory\') {
        final currentPath = _browse[\'path\'] as String;
        _browse[\'path\'] = currentPath == \'.\'
            ? entry[\'name\']
            : \'\\$currentPath/\\${entry[\"name\"]}\';
        _browse[\'parent\'] = true;
        _browse[\'selected\'] = null;
      } else {
        _browse[\'selected\'] = entry[\'name\'];
      }
    });
  }

  void setWorkingDir() {
    setState(() {
      _workingDir = _browse[\'path\'] as String;
    });
    debugPrint(\'Working dir set to: \\$_workingDir\');
  }

  void viewFile() {
    final selected = _browse[\'selected\'];
    if (selected != null) {
      debugPrint(\'View file: \\$selected\');
    }
  }

  void downloadFile() {
    final selected = _browse[\'selected\'];
    if (selected != null) {
      debugPrint(\'Download file: \\$selected\');
    }
  }

  void searchHere() {
    debugPrint(\'Search in: \\${_browse[\"path\"]}\');
  }

  @override
  Widget build(BuildContext context) {
    return ~w
  }
}
', [WidgetBody]).

% ============================================================================
% FILE WRITING UTILITIES
% ============================================================================

%! write_project_files(+OutputDir, +Files) is det
%  Write all project files to the output directory.
write_project_files(OutputDir, Files) :-
    forall(
        member(RelPath-Content, Files),
        write_project_file(OutputDir, RelPath, Content)
    ).

write_project_file(OutputDir, RelPath, Content) :-
    % Build full path
    atomic_list_concat([OutputDir, '/', RelPath], FullPath),

    % Ensure parent directory exists
    file_directory_name(FullPath, ParentDir),
    (exists_directory(ParentDir) -> true ; make_directory_path(ParentDir)),

    % Write file
    open(FullPath, write, Stream),
    write(Stream, Content),
    close(Stream),

    format('  Created: ~w~n', [RelPath]).

% Helper to create nested directories
make_directory_path(Dir) :-
    (exists_directory(Dir) -> true
    ;   file_directory_name(Dir, Parent),
        make_directory_path(Parent),
        make_directory(Dir)
    ).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

%! string_lower(+String, -Lower) is det
%  Convert string to lowercase.
string_lower(String, Lower) :-
    string_codes(String, Codes),
    maplist(to_lower_code, Codes, LowerCodes),
    string_codes(Lower, LowerCodes).

to_lower_code(C, L) :-
    (C >= 65, C =< 90 -> L is C + 32 ; L = C).

% ============================================================================
% TESTS
% ============================================================================

test_project_scaffold :-
    format('~n=== Project Scaffold Generator Tests ===~n~n', []),

    % Test spec
    TestSpec = layout(stack, [spacing(16)], [
        component(heading, [level(1), content('Test App')]),
        component(text, [content('Welcome to the test application.')]),
        component(button, [label('Click Me'), on_click(handleClick)])
    ]),

    % Test 1: Vue project files
    format('Test 1: Vue project file generation...~n', []),
    generate_project_files(vue, test_app, TestSpec, VueFiles),
    length(VueFiles, VueCount),
    format('  Generated ~w files~n', [VueCount]),
    forall(member(Path-_, VueFiles), format('    - ~w~n', [Path])),

    % Test 2: React project files
    format('~nTest 2: React project file generation...~n', []),
    generate_project_files(react, test_app, TestSpec, ReactFiles),
    length(ReactFiles, ReactCount),
    format('  Generated ~w files~n', [ReactCount]),
    forall(member(Path-_, ReactFiles), format('    - ~w~n', [Path])),

    % Test 3: Flutter project files
    format('~nTest 3: Flutter project file generation...~n', []),
    generate_project_files(flutter, test_app, TestSpec, FlutterFiles),
    length(FlutterFiles, FlutterCount),
    format('  Generated ~w files~n', [FlutterCount]),
    forall(member(Path-_, FlutterFiles), format('    - ~w~n', [Path])),

    % Test 4: TUI project files
    format('~nTest 4: TUI project file generation...~n', []),
    generate_project_files(tui, test_app, TestSpec, TuiFiles),
    length(TuiFiles, TuiCount),
    format('  Generated ~w files~n', [TuiCount]),
    forall(member(Path-_, TuiFiles), format('    - ~w~n', [Path])),

    % Test 5: Package.json content
    format('~nTest 5: Vue package.json content...~n', []),
    project_file(vue, package_json, my_app, VuePackage),
    sub_atom(VuePackage, _, _, _, '"name": "my_app"'),
    sub_atom(VuePackage, _, _, _, '"vue"'),
    format('  Contains correct name and vue dependency~n', []),

    % Test 6: React package.json content
    format('~nTest 6: React package.json content...~n', []),
    project_file(react, package_json, my_app, ReactPackage),
    sub_atom(ReactPackage, _, _, _, '"react"'),
    sub_atom(ReactPackage, _, _, _, '"react-dom"'),
    format('  Contains react and react-dom dependencies~n', []),

    % Test 7: Flutter pubspec content
    format('~nTest 7: Flutter pubspec.yaml content...~n', []),
    project_file(flutter, pubspec, my_app, FlutterPubspec),
    sub_atom(FlutterPubspec, _, _, _, 'flutter:'),
    sub_atom(FlutterPubspec, _, _, _, 'sdk: flutter'),
    format('  Contains flutter SDK dependency~n', []),

    % Test 8: Generated component contains content
    format('~nTest 8: Generated Vue component contains UI...~n', []),
    member('src/App.vue'-VueApp, VueFiles),
    sub_atom(VueApp, _, _, _, 'Test App'),
    sub_atom(VueApp, _, _, _, 'Click Me'),
    format('  Vue component contains heading and button~n', []),

    format('~n=== Tests Complete ===~n', []).

:- format('Project Scaffold Generator module loaded~n', []).
