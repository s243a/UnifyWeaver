# Skill: GUI Design

Sub-master skill for creating visually consistent, responsive user interfaces across platforms.

## When to Use

- User asks "how do I create UI layouts?"
- User wants pre-built components (buttons, cards, modals)
- User asks about responsive breakpoints or media queries
- User needs theming with colors, typography, dark mode
- User wants consistent styling across Vue, React Native, Flutter, SwiftUI

## Overview

UnifyWeaver's design system provides declarative UI specifications that compile to platform-specific code:

```prolog
% Define a themed card component
card(content, [title('Welcome'), elevated(true)], Spec),
generate_component(Spec, vue, Code).
```

## Individual Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `skill_component_library.md` | Pre-built UI components | Modals, cards, toasts, avatars |
| `skill_layout_system.md` | Grid, Flexbox layouts | Row, column, stack arrangements |
| `skill_responsive_design.md` | Breakpoints, media queries | Adaptive layouts |
| `skill_theming.md` | Colors, typography, dark mode | Consistent styling |

## Component Library

Pre-built components that generate code for all targets:

### Modal/Dialog Components

| Component | Purpose | Example |
|-----------|---------|---------|
| `modal/3` | Generic modal | `modal(alert, [title('Warning')], Spec)` |
| `alert_dialog/3` | Alert with confirm | `alert_dialog('Delete?', [message(...)], Spec)` |
| `bottom_sheet/3` | Bottom slide-up | `bottom_sheet(content, [height(auto)], Spec)` |
| `action_sheet/3` | Action options | `action_sheet([save, delete], [...], Spec)` |

### Feedback Components

| Component | Purpose | Example |
|-----------|---------|---------|
| `toast/3` | Brief notification | `toast('Saved!', [type(success)], Spec)` |
| `snackbar/3` | Snackbar with action | `snackbar('Undo?', [action(undo)], Spec)` |
| `banner/3` | Persistent banner | `banner('Update available', [...], Spec)` |

### Content Components

| Component | Purpose | Example |
|-----------|---------|---------|
| `card/3` | Content card | `card(content, [title('Title')], Spec)` |
| `list_item/3` | List row | `list_item('Item', [leading(icon)], Spec)` |
| `avatar/3` | User avatar | `avatar('url', [size(large)], Spec)` |
| `badge/3` | Badge/indicator | `badge('5', [color(error)], Spec)` |
| `chip/3` | Chip/tag | `chip('Label', [variant(filled)], Spec)` |

### Progress Components

| Component | Purpose | Example |
|-----------|---------|---------|
| `progress_bar/3` | Linear progress | `progress_bar(75, [max(100)], Spec)` |
| `progress_circle/3` | Circular progress | `progress_circle(50, [size(48)], Spec)` |
| `spinner/2` | Loading spinner | `spinner([size(medium)], Spec)` |

### Input Components

| Component | Purpose | Example |
|-----------|---------|---------|
| `search_bar/2` | Search input | `search_bar([placeholder('Search...')], Spec)` |
| `rating/3` | Star rating | `rating(4, [max(5)], Spec)` |
| `stepper/3` | Number stepper | `stepper(1, [min(0), max(10)], Spec)` |
| `slider_input/3` | Slider | `slider_input(50, [min(0), max(100)], Spec)` |

See `skill_component_library.md` for full details.

## Layout System

Declarative layout primitives:

### Layout Primitives

```prolog
% Horizontal row
row([justify(space_between), align(center), gap(md)], [child1, child2], Spec).

% Vertical column
column([gap(lg), padding(md)], [header, body, footer], Spec).

% Z-axis stack (overlapping)
stack([], [background, foreground], Spec).

% CSS Grid
grid([columns(3), gap(sm)], Items, Spec).

% Wrapping row
wrap([gap(xs)], Tags, Spec).
```

### Spacing Scale

| Name | Pixels | Usage |
|------|--------|-------|
| `none` | 0 | No spacing |
| `xs` | 4 | Tight spacing |
| `sm` | 8 | Small spacing |
| `md` | 16 | Medium (default) |
| `lg` | 24 | Large spacing |
| `xl` | 32 | Extra large |
| `xxl` | 48 | Maximum spacing |

### Alignment Options

**Justify (main axis):** `start`, `end`, `center`, `space_between`, `space_around`, `space_evenly`

**Align (cross axis):** `start`, `end`, `center`, `stretch`, `baseline`

See `skill_layout_system.md` for full details.

## Responsive Design

Breakpoints and media queries:

### Standard Breakpoints

| Name | Width | Description |
|------|-------|-------------|
| `xs` | ≤575px | Extra small |
| `sm` | ≥576px | Small |
| `md` | ≥768px | Medium |
| `lg` | ≥992px | Large |
| `xl` | ≥1200px | Extra large |
| `xxl` | ≥1400px | XXL |

### Semantic Breakpoints

| Name | Range | Description |
|------|-------|-------------|
| `mobile` | ≤767px | Mobile devices |
| `tablet` | 768-1023px | Tablet devices |
| `desktop` | ≥1024px | Desktop screens |
| `wide` | ≥1440px | Wide screens |

### Responsive Layouts

```prolog
responsive_layout(card_grid, [
    default([
        strategy(grid),
        columns(["1fr"]),
        gap("1rem")
    ]),
    at(sm, [columns(["repeat(2, 1fr)"])]),
    at(md, [columns(["repeat(3, 1fr)"])]),
    at(lg, [columns(["repeat(4, 1fr)"])])
]).
```

### Container Queries

Modern CSS container-based responsiveness:

```prolog
container(chart_container, [
    type(inline_size),
    breakpoints([
        at(small, max_width(400)),
        at(medium, range(401, 800)),
        at(large, min_width(801))
    ])
]).
```

See `skill_responsive_design.md` for full details.

## Theming

Consistent styling with design tokens:

### Theme Definition

```prolog
define_theme(my_app, [
    colors([
        primary-'#007AFF',
        secondary-'#5856D6',
        background-'#FFFFFF',
        text-'#000000'
    ]),
    typography([
        family-'Inter',
        sizeBase-16,
        sizeLarge-20,
        weightBold-700
    ]),
    spacing([xs-4, sm-8, md-16, lg-24, xl-32])
]).
```

### Theme Variants (Dark Mode)

```prolog
define_variant(my_app, dark, [
    colors([
        background-'#1C1C1E',
        text-'#FFFFFF'
    ])
]).

% Get merged dark theme
get_variant(my_app, dark, DarkTheme).
```

### Generated Output by Target

**React Native:**
```typescript
export const theme = {
  colors: { primary: '#007AFF', ... },
  typography: { family: 'Inter', ... },
  spacing: { xs: 4, sm: 8, ... },
};
```

**Vue (CSS Variables):**
```css
:root {
  --color-primary: #007AFF;
  --font-family: Inter;
  --spacing-md: 16px;
}
```

**Flutter:**
```dart
class AppTheme {
  static const Color primary = Color(0xFF007AFF);
  static const double fontSizeBase = 16;
}
```

**SwiftUI:**
```swift
struct Theme {
  struct Colors {
    static let primary = Color(hex: "#007AFF")
  }
}
```

See `skill_theming.md` for full details.

## Code Generation Pipeline

### Generate Component

```prolog
% 1. Create component spec
card(content, [title('Welcome'), elevated(true)], Spec).

% 2. Generate for target
generate_component(Spec, Target, Code).
% Targets: react_native, vue, flutter, swiftui
```

### Generate Layout

```prolog
% 1. Create layout spec
row([justify(space_between), gap(md)], [logo, nav, user], Spec).

% 2. Generate for target
generate_layout(Spec, Target, Code).
```

### Generate Theme

```prolog
% 1. Define theme
define_theme(my_app, [...]).

% 2. Generate for target
generate_theme_code(my_app, Target, Code).
```

## Common Workflows

### Create Themed Card Grid

```prolog
% Define theme
define_theme(cards, [
    colors([cardBg-'#FFFFFF', shadow-'rgba(0,0,0,0.1)']),
    spacing([cardGap-16, cardPadding-20])
]).

% Create responsive grid
responsive_layout(card_grid, [
    default([strategy(grid), columns(["1fr"]), gap("var(--spacing-cardGap)")]),
    at(md, [columns(["repeat(2, 1fr)"])]),
    at(lg, [columns(["repeat(3, 1fr)"])])
]).

% Create card component
card(slot, [elevated(true), title('Card Title')], CardSpec).

% Generate
generate_theme_code(cards, vue, ThemeCSS),
generate_responsive_css(card_grid, GridCSS),
generate_component(CardSpec, vue, CardCode).
```

### Mobile-First Dashboard

```prolog
responsive_layout(dashboard, [
    default([
        strategy(grid),
        areas([["nav"], ["main"], ["aside"]]),
        columns(["1fr"])
    ]),
    at(md, [
        areas([["nav", "nav"], ["main", "aside"]]),
        columns(["1fr", "300px"])
    ]),
    at(lg, [
        areas([["nav", "nav", "nav"], ["sidebar", "main", "aside"]]),
        columns(["240px", "1fr", "300px"])
    ])
]).
```

## Related

**Parent Skill:**
- `skill_gui_tools.md` - GUI master skill

**Individual Skills:**
- `skill_component_library.md` - Pre-built components
- `skill_layout_system.md` - Grid, Flexbox layouts
- `skill_responsive_design.md` - Breakpoints, media queries
- `skill_theming.md` - Colors, typography, dark mode

**Sibling Sub-Masters:**
- `skill_gui_generation.md` - App generation
- `skill_gui_runtime.md` - Client-side runtime

**Code:**
- `src/unifyweaver/components/component_library.pl` - Component definitions
- `src/unifyweaver/layout/layout.pl` - Layout system
- `src/unifyweaver/theming/theming.pl` - Theme system
- `src/unifyweaver/glue/responsive_generator.pl` - Responsive CSS
