# Skill: GUI Generation

Sub-master skill for generating frontend applications and compiling to TypeScript.

## When to Use

- User asks "how do I generate a frontend app?"
- User wants Vue, React Native, Flutter, or SwiftUI output
- User asks about TypeScript compilation
- User needs to scaffold a complete project

## Overview

UnifyWeaver generates complete frontend applications from declarative Prolog specifications:

1. **App Specification** - Define screens, navigation, and features in Prolog
2. **Target Selection** - Choose Vue, React Native, Flutter, or SwiftUI
3. **Code Generation** - Generate TypeScript/Dart/Swift code
4. **Project Scaffolding** - Create build files, dependencies, configs

## Individual Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `skill_app_generation.md` | Generate complete apps | Scaffolding new projects |
| `skill_typescript_target.md` | TypeScript compilation | Custom TS code generation |

## Supported Targets

| Target | Framework | Output | Use Case |
|--------|-----------|--------|----------|
| `frontend-vue` | Vue 3 + Vite | Web app | Modern web applications |
| `frontend-react_native` | React Native + Expo | Mobile | iOS/Android apps |
| `frontend-flutter` | Flutter + Dart | Cross-platform | Mobile + Web + Desktop |
| `frontend-swiftui` | SwiftUI | iOS/macOS | Native Apple apps |

## App Specification

### Basic Structure

```prolog
app(app_name, [
    navigation(Type, Screens, Options),
    theme(ThemeName, ThemeOptions),
    auth(AuthOptions),
    layout(LayoutType, LayoutOptions)
]).
```

### Navigation Types

| Type | Description | Best For |
|------|-------------|----------|
| `tabs` | Bottom tab bar | Mobile apps |
| `drawer` | Side drawer menu | Complex navigation |
| `stack` | Stack-based navigation | Wizards, flows |

```prolog
% Tab navigation
navigation(tabs, [
    screen(home, 'HomeView', [icon(home)]),
    screen(search, 'SearchView', [icon(search)]),
    screen(profile, 'ProfileView', [icon(person)])
], [])

% Drawer navigation
navigation(drawer, [
    screen(dashboard, 'DashboardView', []),
    screen(settings, 'SettingsView', []),
    screen(help, 'HelpView', [])
], [header('My App')])
```

### Screen Options

```prolog
screen(screen_id, 'ComponentName', [
    icon(IconName),           % Navigation icon
    guard(GuardName),         % Auth guard
    params([id, name]),       % Route parameters
    lazy(true)                % Lazy loading
])
```

## Generation Pipeline

### Step 1: Define App Specification

```prolog
:- use_module('src/unifyweaver/glue/app_generator').

my_app(App) :-
    App = app(taskmanager, [
        navigation(tabs, [
            screen(tasks, 'TaskListView', [icon(list)]),
            screen(add, 'AddTaskView', [icon(add)]),
            screen(settings, 'SettingsView', [icon(settings)])
        ], []),
        theme(default, [
            primary_color('#3498db'),
            dark_mode(auto)
        ])
    ]).
```

### Step 2: Generate Project

```prolog
?- my_app(App),
   generate_complete_project(App, [frontend-vue], '/tmp/taskmanager', Result).
```

### Step 3: Build and Run

```bash
cd /tmp/taskmanager
npm install
npm run dev
```

## TypeScript Integration

All frontend targets generate TypeScript code. The TypeScript target provides:

- Type-safe component generation
- Interface definitions for props/state
- Proper imports and module structure

See `skill_typescript_target.md` for TypeScript-specific details.

## Backend Integration

Generate full-stack applications with backend:

```prolog
?- generate_complete_project(
       app(myapp, [...]),
       [frontend-vue, backend-fastapi],  % Vue + FastAPI
       '/tmp/fullstack',
       Result
   ).
```

Supported backends:
- `backend-fastapi` - Python FastAPI
- `backend-flask` - Python Flask
- `backend-express` - Node.js Express

## Common Workflows

### Generate Vue Web App

```prolog
?- generate_complete_project(
       app(webapp, [
           navigation(tabs, [
               screen(home, 'HomeView', []),
               screen(about, 'AboutView', [])
           ], [])
       ]),
       [frontend-vue],
       'output/webapp',
       Result
   ).
```

### Generate Mobile App

```prolog
?- generate_complete_project(
       app(mobileapp, [
           navigation(tabs, [...], []),
           theme(default, [platform(mobile)])
       ]),
       [frontend-react_native],
       'output/mobileapp',
       Result
   ).
```

### Generate with Auth

```prolog
?- generate_complete_project(
       app(secureapp, [
           auth([backend(jwt), guards([authenticated])]),
           navigation(tabs, [
               screen(public, 'PublicView', []),
               screen(private, 'PrivateView', [guard(authenticated)])
           ], [])
       ]),
       [frontend-vue, backend-fastapi],
       'output/secureapp',
       Result
   ).
```

## Related

**Parent Skill:**
- `skill_gui_tools.md` - GUI master skill

**Individual Skills:**
- `skill_app_generation.md` - Detailed app generation
- `skill_typescript_target.md` - TypeScript compilation

**Sibling Sub-Masters:**
- `skill_gui_design.md` - UI design
- `skill_gui_runtime.md` - Client-side runtime

**Code:**
- `src/unifyweaver/glue/app_generator.pl` - Main generator
- `src/unifyweaver/targets/vue_target.pl` - Vue target
- `src/unifyweaver/targets/react_native_target.pl` - React Native target
- `src/unifyweaver/targets/flutter_target.pl` - Flutter target
- `src/unifyweaver/targets/swiftui_target.pl` - SwiftUI target
- `src/unifyweaver/targets/typescript_target.pl` - TypeScript target
