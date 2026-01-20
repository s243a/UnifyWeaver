# Skill: Responsive Design

Breakpoints, media queries, and adaptive layouts for multi-device support.

## When to Use

- User asks "how do I make layouts responsive?"
- User wants mobile-first or desktop-first strategies
- User needs media queries or container queries
- User asks about breakpoints for different devices

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/responsive_generator').

% Define a responsive card grid
responsive_layout(my_grid, [
    default([strategy(grid), columns(["1fr"])]),
    at(sm, [columns(["repeat(2, 1fr)"])]),
    at(lg, [columns(["repeat(4, 1fr)"])])
]).

% Generate CSS
generate_responsive_css(my_grid, CSS).
```

## Standard Breakpoints

### Size-Based Breakpoints

| Name | Width | Media Query |
|------|-------|-------------|
| `xs` | ≤575px | `@media (max-width: 575px)` |
| `sm` | ≥576px | `@media (min-width: 576px)` |
| `md` | ≥768px | `@media (min-width: 768px)` |
| `lg` | ≥992px | `@media (min-width: 992px)` |
| `xl` | ≥1200px | `@media (min-width: 1200px)` |
| `xxl` | ≥1400px | `@media (min-width: 1400px)` |

### Semantic Breakpoints

| Name | Range | Description |
|------|-------|-------------|
| `mobile` | ≤767px | Phones |
| `tablet` | 768-1023px | Tablets |
| `desktop` | ≥1024px | Desktop screens |
| `wide` | ≥1440px | Wide monitors |

### Device-Specific Breakpoints

| Name | Range | Description |
|------|-------|-------------|
| `phone_portrait` | ≤480px | Phone portrait |
| `phone_landscape` | 481-767px | Phone landscape |
| `tablet_portrait` | 768-1024px | Tablet portrait |
| `tablet_landscape` | 1025-1279px | Tablet landscape |

## Breakpoint Conditions

### min_width

Mobile-first approach (default):

```prolog
breakpoint(sm, min_width(576)).
% @media (min-width: 576px)
```

### max_width

Desktop-first approach:

```prolog
breakpoint(mobile, max_width(767)).
% @media (max-width: 767px)
```

### range

Specific range:

```prolog
breakpoint(tablet, range(768, 1023)).
% @media (min-width: 768px) and (max-width: 1023px)
```

### orientation

```prolog
breakpoint(landscape, orientation(landscape)).
% @media (orientation: landscape)
```

### prefers_color_scheme

```prolog
breakpoint(dark_mode, prefers_color_scheme(dark)).
% @media (prefers-color-scheme: dark)
```

### prefers_reduced_motion

```prolog
breakpoint(reduced_motion, prefers_reduced_motion).
% @media (prefers-reduced-motion: reduce)
```

## Responsive Layouts

### Define Layout with Variants

```prolog
responsive_layout(Name, Variants).
```

**Variants:**
- `default(Options)` - Base styles (no media query)
- `at(Breakpoint, Options)` - Styles at breakpoint

### Layout Options

**Grid Strategy:**
```prolog
strategy(grid),
columns(["240px", "1fr"]),     % grid-template-columns
rows(["auto", "1fr", "auto"]), % grid-template-rows
areas([["header"], ["main"]]), % grid-template-areas
gap("1rem")
```

**Flex Strategy:**
```prolog
strategy(flex),
direction(row),     % flex-direction
wrap(wrap),         % flex-wrap
justify(center),    % justify-content
align(stretch),     % align-items
gap("1rem")
```

### Example: Collapsible Sidebar

```prolog
responsive_layout(sidebar_layout, [
    default([
        strategy(grid),
        areas([["sidebar", "main"]]),
        columns(["280px", "1fr"]),
        gap("1rem")
    ]),
    at(mobile, [
        areas([["main"], ["sidebar"]]),
        columns(["1fr"]),
        sidebar_position(bottom)
    ]),
    at(tablet, [
        columns(["220px", "1fr"])
    ])
]).
```

### Example: Card Grid

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

### Example: Dashboard

```prolog
responsive_layout(dashboard, [
    default([
        strategy(grid),
        areas([["nav"], ["main"], ["aside"]]),
        columns(["1fr"]),
        rows(["auto", "1fr", "auto"])
    ]),
    at(md, [
        areas([["nav", "nav"], ["main", "aside"]]),
        columns(["1fr", "300px"]),
        rows(["auto", "1fr"])
    ]),
    at(lg, [
        areas([["nav", "nav", "nav"], ["sidebar", "main", "aside"]]),
        columns(["240px", "1fr", "300px"])
    ])
]).
```

## Container Queries

Modern CSS container-based responsiveness:

### Define Container

```prolog
container(Name, Options).
```

| Option | Description |
|--------|-------------|
| `type(T)` | `inline_size`, `size`, `normal` |
| `breakpoints(L)` | List of `at(Name, Condition)` |

### Example

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

**Generated CSS:**
```css
.chart-container {
  container-type: inline-size;
  container-name: chart_container;
}

@container chart_container (min-width: 801px) {
  .chart-container--large {
    /* Styles for large */
  }
}
```

## Code Generation

### Generate Media Query

```prolog
generate_media_query(Breakpoint, MediaQuery).
```

```prolog
?- generate_media_query(mobile, Q).
Q = '@media (max-width: 767px)'
```

### Generate Responsive CSS

```prolog
generate_responsive_css(LayoutName, CSS).
```

### Generate Container CSS

```prolog
generate_container_css(ContainerName, CSS).
```

## Strategy Selection

### Mobile-First (Default)

Start with mobile styles, add complexity for larger screens:

```prolog
set_responsive_strategy(mobile_first).

responsive_layout(content, [
    default([columns(["1fr"])]),      % Mobile: single column
    at(md, [columns(["1fr", "1fr"])]) % Tablet+: two columns
]).
```

### Desktop-First

Start with desktop styles, simplify for smaller screens:

```prolog
set_responsive_strategy(desktop_first).

responsive_layout(content, [
    default([columns(["1fr", "1fr", "1fr"])]), % Desktop: three columns
    at(mobile, [columns(["1fr"])])              % Mobile: single column
]).
```

### Check Current Strategy

```prolog
is_mobile_first.  % Succeeds if mobile-first
```

## Custom Breakpoints

### Declare New Breakpoint

```prolog
declare_breakpoint(ultrawide, min_width(2560)).
declare_breakpoint(foldable, range(280, 600)).
```

### Declare Custom Layout

```prolog
declare_responsive_layout(my_layout, [
    default([...]),
    at(ultrawide, [...])
]).
```

## Visibility Utilities

Generate show/hide classes per breakpoint:

```prolog
generate_visibility_utilities(CSS).
```

**Generated:**
```css
.hidden-mobile { display: block; }
.visible-mobile { display: none; }

@media (max-width: 767px) {
  .hidden-mobile { display: none; }
  .visible-mobile { display: block; }
}
```

## Common Patterns

### Two-Column at Tablet

```prolog
responsive_layout(two_col, [
    default([strategy(flex), direction(column), gap("1rem")]),
    at(md, [direction(row)])
]).
```

### Sticky Header

```prolog
responsive_layout(sticky_header, [
    default([
        strategy(grid),
        rows(["60px", "1fr"]),
        position(sticky, 0)
    ])
]).
```

### Full-Width Mobile, Constrained Desktop

```prolog
responsive_layout(constrained, [
    default([width("100%"), padding("1rem")]),
    at(lg, [width("1200px"), margin("0 auto")])
]).
```

## Related

**Parent Skill:**
- `skill_gui_design.md` - GUI design sub-master

**Sibling Skills:**
- `skill_component_library.md` - Pre-built components
- `skill_layout_system.md` - Layout primitives
- `skill_theming.md` - Colors, typography

**Code:**
- `src/unifyweaver/glue/responsive_generator.pl` - Responsive CSS generation
