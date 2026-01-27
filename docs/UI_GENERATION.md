# Multi-Target UI Generation

UnifyWeaver includes a declarative UI specification system that generates interfaces for multiple targets from a single specification.

## Overview

Write your UI once using Prolog-based primitives, then generate:
- **Vue.js** - Single Page Applications
- **React** - Component-based web apps
- **Flutter** - Cross-platform mobile/desktop apps
- **TUI (Terminal UI)** - ANSI-styled terminal interfaces
- **Dialog TUI** - Interactive ncurses-based terminal apps

## Architecture

```
UI Specification (Prolog)
         │
         ▼
   ┌─────────────┐
   │ UI Primitives│  (layout, container, component)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │ project_    │
   │ scaffold.pl │
   └─────────────┘
         │
    ┌────┼────┬────────┬──────────┐
    ▼    ▼    ▼        ▼          ▼
  Vue  React Flutter  TUI    Dialog TUI
```

## UI Primitives

### Layouts
- `layout(stack, Options, Children)` - Vertical stack
- `layout(flex, Options, Children)` - Horizontal flex
- `layout(grid, Options, Children)` - Grid layout
- `layout(center, Options, Children)` - Centered content

### Containers
- `container(panel, Options, Content)` - Bordered panel
- `container(card, Options, Content)` - Card with shadow
- `container(scroll, Options, Content)` - Scrollable area
- `container(modal, Options, Content)` - Modal dialog

### Components
- `component(text, Options)` - Text display
- `component(button, Options)` - Clickable button
- `component(text_input, Options)` - Text input field
- `component(select, Options)` - Dropdown select
- `component(checkbox, Options)` - Checkbox
- `component(badge, Options)` - Status badge

### Conditionals
- `when(Condition, Children)` - Conditional rendering
- `foreach(Collection, ItemVar, Children)` - List iteration

## Usage

### Generate a Project

```prolog
:- use_module('src/unifyweaver/ui/project_scaffold').
:- use_module('src/unifyweaver/ui/http_cli_ui').

% Get a UI specification
http_cli_ui:browse_panel_spec(Spec),

% Generate Vue project
project_scaffold:generate_project(vue, my_app, Spec, 'output/vue-app').

% Generate Flutter project
project_scaffold:generate_project(flutter, my_app, Spec, 'output/flutter-app').

% Generate TUI (echo-based)
project_scaffold:generate_project(tui, my_app, Spec, 'output/tui-app').

% Generate Dialog TUI (interactive)
project_scaffold:generate_project(tui, my_app, Spec, 'output/dialog-app', [mode(dialog)]).
```

### Example UI Specification

```prolog
browse_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            % Navigation bar
            layout(flex, [align(center), gap(10)], [
                component(button, [
                    label("⬆️ Up"),
                    on_click(navigate_up)
                ]),
                component(code, [content(var('browse.path'))])
            ]),

            % File list
            foreach(var('browse.entries'), entry, [
                container(panel, [class(file_entry)], [
                    layout(flex, [justify(between)], [
                        component(text, [content(var('entry.name'))]),
                        component(text, [content(format_size(var('entry.size')))])
                    ])
                ])
            ]),

            % Selected file actions
            when(var('browse.selected'), [
                layout(flex, [gap(10)], [
                    component(button, [label("View"), on_click(view_file)]),
                    component(button, [label("Download"), on_click(download_file)])
                ])
            ])
        ])
    ])
).
```

## TUI Modes

### Echo Mode (default)
Generates static ANSI-styled output using `echo -e` commands. Good for displaying information but limited interactivity.

```bash
# Generate echo-based TUI
generate_project(tui, my_app, Spec, 'output/').
```

### Dialog Mode
Generates interactive terminal UI using the `dialog` utility (ncurses-based). Supports:
- Menu navigation with keyboard
- File browser with directory traversal
- Input dialogs
- Message boxes
- Quick-select keys (0-9, a-z)

```bash
# Generate dialog-based TUI
generate_project(tui, my_app, Spec, 'output/', [mode(dialog)]).
```

#### Dialog Features
- **File Browser** - Navigate directories, select files
- **Quick Select** - Press 0-9 for first 10 items, a-z for items 11-36
- **Action Menu** - View, download, search operations
- **Keyboard Navigation** - Arrow keys, Enter, Escape

#### Requirements
```bash
# Install dialog
apt install dialog      # Debian/Ubuntu/Termux
brew install dialog     # macOS
dnf install dialog      # Fedora
```

## Generated Project Structure

### Vue/React
```
my_app/
├── src/
│   ├── App.vue (or App.tsx)
│   └── main.ts
├── package.json
├── vite.config.ts
└── tsconfig.json
```

### Flutter
```
my_app/
├── lib/
│   ├── main.dart
│   └── app.dart
├── pubspec.yaml
└── analysis_options.yaml
```

### TUI
```
my_app/
├── my_app.sh      # Main script
├── run.sh         # Runner with dependency check
├── Makefile       # Build/run commands
└── README.md
```

## Examples

See `examples/` directory:
- `examples/tui-cli/` - Echo-based TUI example
- `examples/tui-dialog-cli/` - Dialog-based interactive TUI

## Files

| File | Description |
|------|-------------|
| `src/unifyweaver/ui/ui_primitives.pl` | Core UI primitive definitions |
| `src/unifyweaver/ui/project_scaffold.pl` | Project generation orchestrator |
| `src/unifyweaver/ui/vue_generator.pl` | Vue.js code generator |
| `src/unifyweaver/ui/react_generator.pl` | React code generator |
| `src/unifyweaver/ui/flutter_generator.pl` | Flutter code generator |
| `src/unifyweaver/ui/tui_generator.pl` | TUI/Dialog code generator |
| `src/unifyweaver/ui/http_cli_ui.pl` | HTTP CLI interface specification |
