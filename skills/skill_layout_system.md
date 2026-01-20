# Skill: Layout System

Declarative layout primitives (Flexbox, Grid) that compile across Vue, React Native, Flutter, and SwiftUI.

## When to Use

- User asks "how do I create rows/columns/grids?"
- User wants consistent layouts across platforms
- User needs Flexbox or CSS Grid patterns
- User asks about spacing, alignment, or sizing

## Quick Start

```prolog
:- use_module('src/unifyweaver/layout/layout').

% Create a horizontal row with spacing
row([justify(space_between), align(center), gap(md)], [logo, nav, user], Spec),
generate_layout(Spec, vue, Code).
```

## Layout Primitives

### row/3

Horizontal layout (Flexbox row):

```prolog
row(Options, Children, Spec).
```

```prolog
row([justify(space_between), gap(sm)], [child1, child2, child3], Spec).
```

### column/3

Vertical layout (Flexbox column):

```prolog
column(Options, Children, Spec).
```

```prolog
column([gap(lg), padding(md)], [header, body, footer], Spec).
```

### stack/3

Z-axis stack (overlapping children):

```prolog
stack(Options, Children, Spec).
```

```prolog
stack([], [background_image, gradient_overlay, content], Spec).
```

### grid/3

CSS Grid layout:

```prolog
grid(Options, Children, Spec).
```

```prolog
grid([columns(3), gap(md)], Items, Spec).
```

### wrap/3

Wrapping row (flex-wrap):

```prolog
wrap(Options, Children, Spec).
```

```prolog
wrap([gap(xs)], TagChips, Spec).
```

## Container Layouts

### center/2

Center content both axes:

```prolog
center(Child, Spec).
```

### container/3

Max-width container with centering:

```prolog
container(Options, Children, Spec).
```

### scroll_view/3

Scrollable container:

```prolog
scroll_view(Options, Children, Spec).
```

| Option | Values | Default |
|--------|--------|---------|
| `direction(D)` | `vertical`, `horizontal` | `vertical` |

### safe_area/2

Respect device safe area insets:

```prolog
safe_area(Child, Spec).
```

## Spacing Scale

Named spacing values that map to pixels:

| Name | Pixels | Usage |
|------|--------|-------|
| `none` | 0 | No spacing |
| `xs` | 4 | Tight (icons, chips) |
| `sm` | 8 | Small (between related items) |
| `md` | 16 | Medium (default, sections) |
| `lg` | 24 | Large (major sections) |
| `xl` | 32 | Extra large |
| `xxl` | 48 | Maximum spacing |

Numbers are also accepted: `gap(12)` → 12px

## Layout Options

### Flex Properties

```prolog
flex(1, Term).         % flex: 1
flex_grow(1, Term).    % flex-grow: 1
flex_shrink(0, Term).  % flex-shrink: 0
flex_basis(auto, Term). % flex-basis: auto
```

### Alignment (justify)

Main axis alignment:

| Value | CSS | Description |
|-------|-----|-------------|
| `start` | `flex-start` | Pack at start |
| `end` | `flex-end` | Pack at end |
| `center` | `center` | Center items |
| `space_between` | `space-between` | Even spacing, no edges |
| `space_around` | `space-around` | Even spacing, half edges |
| `space_evenly` | `space-evenly` | Equal spacing everywhere |

```prolog
row([justify(space_between)], [left, right], Spec).
```

### Alignment (align)

Cross axis alignment:

| Value | CSS | Description |
|-------|-----|-------------|
| `start` | `flex-start` | Align to start |
| `end` | `flex-end` | Align to end |
| `center` | `center` | Center items |
| `stretch` | `stretch` | Stretch to fill |
| `baseline` | `baseline` | Align text baselines |

```prolog
row([align(center)], [icon, text], Spec).
```

### Spacing Options

```prolog
gap(md, Term).      % Gap between children
padding(lg, Term).  % Inner padding
margin(sm, Term).   % Outer margin
```

### Sizing Options

```prolog
width(Value, Term).      % full, auto, or number
height(Value, Term).
min_width(Value, Term).
min_height(Value, Term).
max_width(Value, Term).
max_height(Value, Term).
aspect_ratio(1.5, Term). % width/height ratio
```

| Size Value | Meaning |
|------------|---------|
| `full` | 100% |
| `auto` | Content-based |
| `screen` | Viewport size |
| Number | Pixels |

### Positioning

```prolog
position(relative, Term).
position(absolute, Term).
absolute([top(0), right(0)], Term).  % Edge positions
z_index(10, Term).
```

## Code Generation

### Generate for Target

```prolog
generate_layout(Spec, Target, Code).
% Targets: react_native, vue, flutter, swiftui
```

### Example Outputs

**React Native:**
```jsx
<View style={{
  flexDirection: 'row',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: 16
}}>
  {/* children */}
</View>
```

**Vue (Tailwind classes):**
```html
<div class="flex flex-row justify-between items-center gap-4">
  <!-- children -->
</div>
```

**Flutter:**
```dart
Row(
  mainAxisAlignment: MainAxisAlignment.spaceBetween,
  crossAxisAlignment: CrossAxisAlignment.center,
  children: [/* children */],
)
```

**SwiftUI:**
```swift
HStack(alignment: .center, spacing: 16) {
  // children
}
```

## Target-Specific Generators

```prolog
generate_react_native_layout(Spec, Code).
generate_vue_layout(Spec, Code).
generate_flutter_layout(Spec, Code).
generate_swiftui_layout(Spec, Code).
```

## Common Patterns

### Header with Logo and Navigation

```prolog
row([justify(space_between), align(center), padding(md)], [
    logo,
    row([gap(lg)], nav_items, NavSpec),
    user_menu
], HeaderSpec).
```

### Card with Image and Content

```prolog
column([gap(none)], [
    image,
    column([padding(md), gap(sm)], [
        title,
        description,
        row([justify(end)], [action_button], ActionsSpec)
    ], ContentSpec)
], CardSpec).
```

### Centered Page Content

```prolog
center(
    container([max_width(800), padding(lg)], [
        column([gap(xl)], page_sections, ContentSpec)
    ], ContainerSpec),
    PageSpec
).
```

### Responsive Grid (with wrap)

```prolog
wrap([gap(md)], [
    card1, card2, card3, card4
], GridSpec).
```

### Mobile Navigation

```prolog
safe_area(
    column([flex(1)], [
        scroll_view([flex(1)], content, ScrollSpec),
        row([justify(space_around), padding(sm)], tab_items, TabsSpec)
    ], MainSpec),
    AppSpec
).
```

## Related

**Parent Skill:**
- `skill_gui_design.md` - GUI design sub-master

**Sibling Skills:**
- `skill_component_library.md` - Pre-built components
- `skill_responsive_design.md` - Breakpoints
- `skill_theming.md` - Colors, typography

**Code:**
- `src/unifyweaver/layout/layout.pl` - Layout system
