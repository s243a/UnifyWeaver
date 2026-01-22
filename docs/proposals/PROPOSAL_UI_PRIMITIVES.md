# Proposal: Declarative UI Primitives System

## Executive Summary

This proposal introduces a unified UI primitives system for UnifyWeaver that enables declarative specification of user interfaces with cross-platform code generation. Instead of embedding HTML/CSS templates directly in generators, we define UI through composable primitives that compile to Vue, React, Flutter, SwiftUI, and native web technologies.

## Philosophy

### 1. Declarative Over Imperative

UIs should be described by *what* they represent, not *how* to render them:

```prolog
% GOOD: Declarative - describes the UI structure
ui(login_form, [
    layout(stack, [spacing(16)], [
        component(text_input, [label("Email"), bind(email)]),
        component(text_input, [label("Password"), type(password), bind(password)]),
        component(button, [label("Login"), on_click(submit)])
    ])
]).

% BAD: Imperative - HTML template string
login_form_html('<div><input type="email">...</div>').
```

### 2. Composition Over Inheritance

Complex UIs are built by composing simple primitives, not by extending base classes:

```prolog
% Define reusable patterns
pattern(form_field, [Label, Type, Binding],
    layout(stack, [spacing(4)], [
        component(label, [text(Label)]),
        component(text_input, [type(Type), bind(Binding)])
    ])).

% Compose patterns
ui(signup_form, [
    layout(stack, [spacing(16)], [
        use_pattern(form_field, ["Name", text, name]),
        use_pattern(form_field, ["Email", email, email]),
        use_pattern(form_field, ["Password", password, password])
    ])
]).
```

### 3. Target Agnosticism

The same UI specification compiles to multiple platforms:

```prolog
generate_ui(login_form, vue, VueCode).
generate_ui(login_form, react, ReactCode).
generate_ui(login_form, flutter, FlutterCode).
generate_ui(login_form, swiftui, SwiftUICode).
```

### 4. Theme Integration

Styling comes from themes, not inline definitions:

```prolog
% Theme defines visual tokens
define_theme(app_theme, [
    colors([primary('#e94560'), background('#1a1a2e')]),
    spacing([xs(4), sm(8), md(16), lg(24)]),
    typography([body(14), heading(24)])
]).

% Components reference tokens, not values
component(button, [variant(primary)])  % Uses theme.colors.primary
```

### 5. Accessibility First

Accessibility is built into the primitive layer:

```prolog
component(button, [
    label("Submit"),
    aria_label("Submit login form"),
    role(button),
    focusable(true)
]).
```

---

## Existing Infrastructure

### Component Registry (`core/component_registry.pl`)

Provides the foundation for component management:
- Category-based organization (runtime, source, binding, **ui**)
- Type registration with modules
- Instance configuration and validation
- Dependency management
- Compilation to target code

### Theming System (`theming/theming.pl`)

Existing theme infrastructure (already in main):
- Color definitions with variants (light/dark)
- Typography scale (font families, sizes, weights)
- Spacing scale (padding, margin, gap)
- Borders and shadows
- Cross-platform code generation

### Component Library (`components/component_library.pl`)

Existing component patterns (already in main):
- **Modals**: modal, alert_dialog, bottom_sheet, action_sheet
- **Feedback**: toast, snackbar, banner
- **Content**: card, list_item, avatar, badge, chip, tag
- **Layout**: divider, spacer, skeleton
- **Progress**: progress_bar, progress_circle, spinner
- **Input**: search_bar, rating, stepper, slider_input

### Server Components (`components/server_components.pl`)

Backend component patterns:
- Command validator with risk assessment
- File browser with sandbox enforcement
- Feedback store
- WebSocket shell

---

## Specification

### Layer 1: Layout Primitives

Layouts define spatial organization of children:

```prolog
% Stack: vertical or horizontal arrangement
layout(stack, Options, Children).
% Options: direction(row|column), spacing(N), align(start|center|end), justify(...)

% Grid: 2D arrangement
layout(grid, Options, Children).
% Options: columns(N), rows(N), gap(N), template([...])

% Flex: flexible box model
layout(flex, Options, Children).
% Options: direction(row|column), wrap(bool), gap(N), justify(...), align(...)

% Positioned: absolute/relative positioning
layout(positioned, Options, Children).
% Options: position(absolute|relative|fixed), top(N), left(N), ...

% Scroll: scrollable container
layout(scroll, Options, Children).
% Options: direction(vertical|horizontal|both), showScrollbar(bool)
```

### Layer 2: Container Primitives

Containers provide visual grouping and styling:

```prolog
% Panel: styled container with background
container(panel, Options, Content).
% Options: background(Color), padding(N), rounded(N), shadow(N)

% Card: elevated content container
container(card, Options, Content).
% Options: elevation(N), padding(N), header(Component), footer(Component)

% Section: semantic grouping
container(section, Options, Content).
% Options: title(Text), collapsible(bool), collapsed(bool)

% Conditional: show/hide based on condition
container(when, Condition, Content).
container(unless, Condition, Content).

% Loop: repeat content for each item
container(foreach, [items(List), as(Var)], Template).
```

### Layer 3: Component Primitives

Atomic UI elements:

```prolog
% Text display
component(text, [content(Text), style(Style)]).
component(heading, [level(1-6), content(Text)]).
component(label, [for(Id), text(Text)]).

% Input elements
component(text_input, [type(text|email|password|...), bind(Var), placeholder(Text)]).
component(textarea, [bind(Var), rows(N), placeholder(Text)]).
component(checkbox, [bind(Var), label(Text)]).
component(radio, [bind(Var), options(List)]).
component(select, [bind(Var), options(List)]).
component(switch, [bind(Var), label(Text)]).

% Buttons
component(button, [label(Text), on_click(Action), variant(primary|secondary|danger)]).
component(icon_button, [icon(Name), on_click(Action), aria_label(Text)]).
component(link, [href(URL), label(Text)]).

% Media
component(image, [src(URL), alt(Text), width(N), height(N)]).
component(icon, [name(Name), size(N), color(Color)]).
component(avatar, [src(URL), name(Text), size(sm|md|lg)]).

% Feedback
component(spinner, [size(N)]).
component(progress, [value(N), max(N), variant(bar|circle)]).
component(badge, [content(Text), variant(info|success|warning|error)]).

% Navigation
component(tabs, [items(List), active(Var), on_change(Action)]).
component(breadcrumb, [items(List)]).
component(menu, [items(List), on_select(Action)]).
```

### Layer 4: Composite Patterns

Higher-level patterns built from primitives:

```prolog
% Form pattern
pattern(form, Options, Fields,
    layout(stack, [spacing(16)], [
        foreach(Fields, field, use_pattern(form_field, field)),
        layout(flex, [justify(end), gap(8)], [
            component(button, [label("Cancel"), variant(secondary), on_click(cancel)]),
            component(button, [label("Submit"), variant(primary), on_click(submit)])
        ])
    ])).

% Data table pattern
pattern(data_table, [columns(Cols), data(Data), on_row_click(Action)],
    container(panel, [padding(0)], [
        layout(grid, [columns(length(Cols))], [
            foreach(Cols, col, component(text, [content(col.label), style(header)])),
            foreach(Data, row,
                foreach(Cols, col, component(text, [content(row[col.key])])))
        ])
    ])).

% Login form pattern
pattern(login_form, [on_submit(Action), on_forgot(ForgotAction)],
    container(card, [padding(24), max_width(400)], [
        layout(stack, [spacing(16)], [
            component(heading, [level(2), content("Login")]),
            component(text_input, [label("Email"), type(email), bind(email)]),
            component(text_input, [label("Password"), type(password), bind(password)]),
            component(button, [label("Login"), variant(primary), on_click(Action)]),
            component(link, [label("Forgot password?"), on_click(ForgotAction)])
        ])
    ])).
```

### Layer 5: Page Composition

Complete page structures:

```prolog
% Page with header, content, footer
page(app_page, [
    region(header, [
        layout(flex, [justify(between), align(center), padding(16)], [
            component(heading, [level(1), content("App Title")]),
            use_pattern(user_menu, [user(current_user)])
        ])
    ]),
    region(sidebar, [
        use_pattern(navigation, [items(nav_items), active(current_route)])
    ]),
    region(main, [
        outlet(content)  % Dynamic content goes here
    ]),
    region(footer, [
        component(text, [content("Copyright 2026")])
    ])
]).
```

---

## Implementation Plan

### Phase 1: Core Primitives (Week 1-2)

1. **Create `ui_primitives.pl` module**
   - Layout primitives: stack, flex, grid
   - Container primitives: panel, card, section, when, foreach
   - Basic components: text, button, text_input

2. **Integrate with component_registry.pl**
   - Register 'ui' category
   - Define primitive types
   - Wire up validation

3. **Vue.js code generator**
   - Template generation
   - Script setup generation
   - Scoped styles from theme

### Phase 2: Component Library (Week 3-4)

1. **Extend existing component_library.pl**
   - Theming system already available
   - Component library already available

2. **Extend component primitives**
   - All input types
   - Navigation components
   - Feedback components

3. **Pattern system**
   - Pattern definition and registration
   - Pattern instantiation with parameters
   - Nested pattern composition

### Phase 3: HTTP CLI Integration (Week 5)

1. **Refactor http_server_generator.pl**
   - Replace HTML template with UI primitives
   - Define HTTP CLI interface as page composition

2. **Generate from spec**
   ```prolog
   http_cli_ui(spec, [
       use_pattern(login_form, [...]),
       use_pattern(tabbed_interface, [
           tab(browse, use_pattern(file_browser, [...])),
           tab(grep, use_pattern(search_form, [...])),
           tab(shell, use_pattern(terminal, [...]))
       ])
   ]).
   ```

### Phase 4: Multi-Target (Week 6+)

1. **React generator**
2. **Flutter generator**
3. **CLI/TUI generator** (for terminal interfaces)

---

## File Structure

```
src/unifyweaver/
├── ui/
│   ├── ui_primitives.pl      # Core layout/container/component primitives
│   ├── ui_patterns.pl        # Reusable composite patterns
│   ├── ui_page.pl            # Page composition
│   └── ui_validation.pl      # UI spec validation
├── generators/
│   ├── vue_generator.pl      # Vue.js code generation
│   ├── react_generator.pl    # React code generation
│   └── flutter_generator.pl  # Flutter code generation
├── theming/
│   └── theming.pl            # (from feat/theming-components)
└── components/
    ├── component_library.pl  # (from feat/theming-components)
    └── server_components.pl  # (existing)
```

---

## Example: HTTP CLI Interface

Current approach (embedded HTML string):
```prolog
generate_html_interface(Code) :-
    Code = '<!DOCTYPE html><html>... 800 lines of HTML/CSS/JS ...'.
```

Proposed approach (declarative primitives):
```prolog
http_cli_interface(ui_spec([
    page(main, [
        region(header, [
            layout(flex, [justify(between), align(center)], [
                component(heading, [level(1), content("UnifyWeaver CLI")]),
                container(when, user, [
                    use_pattern(user_badge, [user(user)])
                ])
            ])
        ]),
        region(main, [
            container(when, [auth_required, not(user)], [
                use_pattern(login_form, [on_submit(do_login)])
            ]),
            container(when, [not(auth_required), or, user], [
                use_pattern(working_dir_bar, [path(working_dir)]),
                component(tabs, [
                    items([browse, grep, find, cat, custom, feedback, shell]),
                    active(current_tab)
                ]),
                container(panel, [], [
                    outlet(tab_content)
                ])
            ])
        ])
    ])
])).
```

---

## Benefits

1. **Maintainability**: UI structure is visible and modifiable in Prolog
2. **Reusability**: Patterns compose across different interfaces
3. **Consistency**: Theming ensures visual coherence
4. **Testability**: UI specs can be validated before generation
5. **Multi-target**: Same spec compiles to multiple platforms
6. **Documentation**: Specs serve as living documentation

---

## Implementation Status

### ✅ Phase 1: Core Primitives (Completed)

**Branch**: `feat/ui-primitives` (merged to main)

- Created `ui_primitives.pl` with layout, container, and component primitives
- Integrated with component_registry.pl (ui category)
- Created `vue_generator.pl` for Vue.js template generation
- All tests passing

**Files created**:
- `src/unifyweaver/ui/ui_primitives.pl`
- `src/unifyweaver/ui/vue_generator.pl`
- `src/unifyweaver/ui/ui_registry.pl`

### ✅ Phase 2: Pattern System (Completed)

**Branch**: `feat/ui-patterns`

- Created `ui_patterns.pl` with pattern definition, expansion, and 16 built-in patterns
- Patterns include: form_field, login_form, data_table, modal_dialog, search_bar, etc.
- Pattern instantiation with parameter substitution
- Nested pattern composition via `use_pattern/2`

**Files created**:
- `src/unifyweaver/ui/ui_patterns.pl`

### ✅ Phase 3: HTTP CLI Integration (Completed)

**Branch**: `feat/ui-patterns`

- Created `http_cli_ui.pl` - declarative specification for the HTTP CLI interface
- Created `html_interface_generator.pl` - generates HTML/Vue from UI specs
- Integrated with `http_server_generator.pl` - uses declarative UI when `serve_html(true)`
- Regenerated HTTP CLI server with fully declarative UI (40KB, theme-driven CSS)

**Files created/modified**:
- `src/unifyweaver/ui/http_cli_ui.pl` - HTTP CLI interface specification
- `src/unifyweaver/ui/html_interface_generator.pl` - HTML/CSS/Vue generation
- `src/unifyweaver/glue/http_server_generator.pl` - integration
- `examples/http-cli-server/generated/server.ts` - regenerated server

### ✅ Phase 4: Multi-Target - React (Completed)

**Branch**: `feat/ui-multi-target`

- Created `react_generator.pl` module (986 lines)
- Full JSX template generation from UI specs
- React-specific syntax:
  - JSX conditionals: `{condition && (...)}`
  - Array mapping: `items.map((item, index) => ...)`
  - State setters: `setEmail`, `setPassword`
  - Style objects: `style={{ key: value }}`
  - Event handlers: `onClick`, `onChange`
- All 8 tests passing
- Successfully generates all HTTP CLI panels

**Files created**:
- `src/unifyweaver/ui/react_generator.pl`

### 🔲 Phase 4: Multi-Target - Remaining (Future)

- Flutter generator
- CLI/TUI generator

---

## Related Documents

- `docs/proposals/COMPONENT_REGISTRY.md` - Component registry design
- `docs/CROSS_TARGET_COMPILATION.md` - Multi-target compilation
- `src/unifyweaver/theming/theming.pl` - Theming system (in main)
- `src/unifyweaver/components/component_library.pl` - Component library (in main)
- `feat/layout-system` branch - Layout system exploration
