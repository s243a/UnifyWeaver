# Skill: GUI Tools

Top master skill for frontend application development - app generation, UI design, client-side runtime, and accessibility.

## When to Use

- User asks "how do I build a frontend app?"
- User wants to generate Vue, React Native, Flutter, or SwiftUI apps
- User asks about UI components, layouts, or responsive design
- User needs client-side WebAssembly or Python in browser
- User asks about accessibility or frontend security

## Skill Hierarchy

```
skill_gui_tools.md (this file)
│
├── App Generation & Targets
│   └── skill_gui_generation.md (sub-master)
│       ├── skill_app_generation.md - Vue, React Native, Flutter, SwiftUI
│       └── skill_typescript_target.md - TypeScript compilation, bindings
│
├── UI Design
│   └── skill_gui_design.md (sub-master)
│       ├── skill_component_library.md - Pre-built components
│       ├── skill_layout_system.md - Grid, Flexbox
│       ├── skill_responsive_design.md - Breakpoints, adaptive
│       └── skill_theming.md - Colors, dark mode
│
├── Client-Side Runtime
│   └── skill_gui_runtime.md (sub-master)
│       ├── skill_data_binding.md - Reactive state
│       ├── skill_webassembly.md - LLVM to WASM, Rust/C in browser
│       └── skill_browser_python.md - Pyodide, Numba
│
├── Accessibility
│   └── skill_accessibility.md - ARIA, keyboard navigation
│
└── Frontend Security
    └── skill_frontend_security.md - Navigation guards, CORS
```

## Sub-Skills Overview

### App Generation & Targets

Generate complete frontend applications from declarative Prolog specifications.

**Sub-master:** `skill_gui_generation.md`

Supported targets:
- Vue 3 + Vite + TypeScript
- React Native + Expo
- Flutter + Dart
- SwiftUI (iOS/macOS)

### UI Design

Create visually consistent, responsive user interfaces.

**Sub-master:** `skill_gui_design.md`

Covers:
- Component libraries (buttons, forms, cards)
- Layout systems (Grid, Flexbox)
- Responsive breakpoints
- Theming and dark mode

### Client-Side Runtime

Run compiled code in the browser for performance-critical applications.

**Sub-master:** `skill_gui_runtime.md`

Technologies:
- WebAssembly (from Rust, C, LLVM)
- Pyodide (Python in browser)
- Reactive data binding

### Accessibility

Make applications usable by everyone.

**Skill:** `skill_accessibility.md`

Covers:
- ARIA attributes
- Keyboard navigation
- Screen reader support

### Frontend Security

Protect client-side applications.

**Skill:** `skill_frontend_security.md`

Covers:
- Navigation guards
- CORS configuration
- Client-side validation

## Quick Start

### Generate a Vue App

```prolog
?- use_module('src/unifyweaver/glue/app_generator').
?- generate_complete_project(
       app(myapp, [
           navigation(tabs, [
               screen(home, 'HomeView', []),
               screen(settings, 'SettingsView', [])
           ], []),
           theme(default, [dark_mode(true)])
       ]),
       [frontend-vue],
       '/tmp/myapp',
       Result
   ).
```

### Add Responsive Layout

```prolog
app(myapp, [
    navigation(...),
    layout(responsive, [
        breakpoint(mobile, 0, 768),
        breakpoint(tablet, 769, 1024),
        breakpoint(desktop, 1025, infinity)
    ])
]).
```

### Include WebAssembly Module

```prolog
app(myapp, [
    navigation(...),
    wasm_modules([
        wasm(image_processor, 'image_processor.wasm', [])
    ])
]).
```

## Related

**Sub-Master Skills:**
- `skill_gui_generation.md` - App generation & TypeScript
- `skill_gui_design.md` - UI design
- `skill_gui_runtime.md` - Client-side runtime

**Individual Skills:**
- `skill_accessibility.md` - Accessibility
- `skill_frontend_security.md` - Frontend security

**Backend Skills (separate tree):**
- `skill_backend_tools.md` - Server-side development

**Documentation:**
- `education/other-books/book-gui-generation/` - GUI generation tutorial

**Code:**
- `src/unifyweaver/glue/app_generator.pl` - Main generator
- `src/unifyweaver/targets/vue_target.pl` - Vue target
- `src/unifyweaver/targets/typescript_target.pl` - TypeScript target
