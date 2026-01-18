# Skill: App Generation

Generate complete frontend applications from declarative Prolog specifications.

## When to Use

- User wants to generate a Vue, React Native, Flutter, or SwiftUI app
- User asks "can I generate an app from Prolog?"
- User needs to scaffold a frontend project
- User wants declarative UI specification

## Quick Start

```prolog
?- use_module('src/unifyweaver/glue/app_generator').
?- generate_complete_project(
       app(myapp, [
           navigation(tabs, [
               screen(home, 'HomeScreen', []),
               screen(settings, 'SettingsScreen', [])
           ], [])
       ]),
       [frontend-vue],
       '/tmp/myapp',
       Result
   ).
```

## Commands

### Generate Vue 3 App
```prolog
?- generate_complete_project(
       app(myapp, [
           navigation(tabs, [
               screen(home, 'HomeView', []),
               screen(profile, 'ProfileView', [])
           ], [])
       ]),
       [frontend-vue],
       'OUTPUT_DIR',
       Result
   ).
```

### Generate React Native App
```prolog
?- generate_complete_project(
       app(myapp, [...]),
       [frontend-react_native],
       'OUTPUT_DIR',
       Result
   ).
```

### Generate Flutter App
```prolog
?- generate_complete_project(
       app(myapp, [...]),
       [frontend-flutter],
       'OUTPUT_DIR',
       Result
   ).
```

### Generate with Backend
```prolog
?- generate_complete_project(
       app(myapp, [...]),
       [frontend-vue, backend-fastapi],
       'OUTPUT_DIR',
       Result
   ).
```

## Frontend Targets

| Target | Framework | Output |
|--------|-----------|--------|
| `frontend-vue` | Vue 3 + Vite + TypeScript | Web app |
| `frontend-react_native` | React Native + Expo | Mobile app |
| `frontend-flutter` | Flutter + Dart | Cross-platform |
| `frontend-swiftui` | SwiftUI | iOS/macOS |

## App Specification

```prolog
app(app_name, [
    navigation(Type, Screens, Options),
    theme(ThemeName, ThemeOptions),
    auth(AuthOptions)  % See skill_webapp_security.md
]).

% Navigation types: tabs, drawer, stack
% Screen: screen(id, 'ComponentName', ScreenOptions)
```

## Features

| Feature | Prolog Option | Description |
|---------|---------------|-------------|
| Theming | `theme(default, [...])` | Colors, dark mode |
| Auth | `auth([...])` | Login, guards |
| Data binding | `data_binding([...])` | Reactive state |
| Layouts | `layout(grid, [...])` | CSS Grid/Flexbox |

## Related

**Skills:**
- `skill_webapp_security.md` - Add authentication and guards
- `skill_unifyweaver_compile.md` - General compilation

**Documentation:**
- `pr_shell_sandbox.md` - Security features for generated apps

**Education (in `education/` subfolder):**
- `other-books/book-gui-generation/01_introduction.md` - Overview
- `other-books/book-gui-generation/02_app_generation.md` - Full project scaffolding
- `other-books/book-gui-generation/03_component_library.md` - Pre-built components
- `other-books/book-gui-generation/04_layout_system.md` - Grid and Flexbox
- `other-books/book-gui-generation/05_data_binding.md` - State and reactivity
- `other-books/book-gui-generation/06_accessibility.md` - ARIA and keyboard nav
- `other-books/book-gui-generation/07_responsive_design.md` - Breakpoints
- `other-books/book-gui-generation/08_theming.md` - Colors and dark mode

**Code:**
- `src/unifyweaver/glue/app_generator.pl` - Main generator
