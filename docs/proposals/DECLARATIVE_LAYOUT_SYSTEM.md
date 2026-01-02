# Declarative Layout and Styling System

**Status:** Implemented (All 9 phases complete)
**Author:** Claude Code
**Date:** 2026-01-01
**Tests:** 182 passing

## Overview

This proposal defines a declarative system for specifying UI layouts, styles, and component composition in Prolog. The system generates CSS and HTML structure from high-level specifications, enabling full customization while providing sensible defaults.

## Philosophy

### 1. Defaults Over Configuration

Every component should work without explicit layout or styling configuration. Users only specify what they want to customize.

```prolog
% This works with defaults
generate_curve_component(trig_demo, Code).

% This customizes layout
layout(trig_demo, grid, [...]),
generate_curve_component(trig_demo, Code).
```

### 2. Declarative Over Imperative

Layouts are declared as data structures, not procedural code. The generator interprets the structure.

```prolog
% Declarative: what, not how
layout(demo, grid, [
    areas([["sidebar", "main"]]),
    columns(["320px", "1fr"])
]).

% Not imperative
% set_grid_template_columns("320px 1fr"),
% add_grid_area("sidebar", 1, 1),
% ...
```

### 3. Composition Over Inheritance

Components compose via slots and regions, not class hierarchies.

```prolog
% Compose components into regions
place(demo, sidebar, [control_panel, curve_list]).
place(demo, main, [chart]).
```

### 4. Escape Hatches

When declarative specs aren't enough, raw CSS/HTML can be injected.

```prolog
% Raw CSS escape hatch
raw_css(my_component, "
    .custom-animation { animation: pulse 2s infinite; }
").
```

## Specification

### Layout Strategies

#### Grid Layout (Default)

```prolog
%% layout(+ComponentName, +Strategy, +Options)
layout(Name, grid, Options).

% Options:
%   areas(GridAreas)      - 2D list of area names
%   columns(ColSizes)     - List of column sizes
%   rows(RowSizes)        - List of row sizes
%   gap(Size)             - Grid gap
%   gap(RowGap, ColGap)   - Separate row/column gaps

% Example: Sidebar + Main layout
layout(curve_demo, grid, [
    areas([
        ["header",  "header"],
        ["sidebar", "main"],
        ["footer",  "footer"]
    ]),
    columns(["320px", "1fr"]),
    rows(["auto", "1fr", "auto"]),
    gap("1rem")
]).
```

#### Flexbox Layout

```prolog
layout(Name, flex, Options).

% Options:
%   direction(row|column)
%   wrap(wrap|nowrap)
%   justify(start|center|end|space_between|space_around)
%   align(start|center|end|stretch)
%   gap(Size)

% Example: Horizontal toolbar
layout(toolbar, flex, [
    direction(row),
    justify(space_between),
    align(center),
    gap("0.5rem")
]).
```

#### Absolute Positioning

```prolog
layout(Name, absolute, Options).

% Options:
%   region(Name, Positioning)
%
% Positioning:
%   top(V), right(V), bottom(V), left(V)
%   width(V), height(V)

% Example: Overlay layout
layout(overlay_demo, absolute, [
    region(modal, [top("50%"), left("50%"),
                   width("400px"), height("auto"),
                   transform("translate(-50%, -50%)")]),
    region(backdrop, [top(0), left(0), right(0), bottom(0)])
]).
```

### Component Placement

```prolog
%% place(+LayoutName, +Region, +Components)
place(curve_demo, header, [title_bar]).
place(curve_demo, sidebar, [control_panel, curve_list]).
place(curve_demo, main, [chart]).
place(curve_demo, footer, [status_bar]).
```

### Style System

#### Component Styles

```prolog
%% style(+Component, +Properties)
style(chart_container, [
    background('#1a1a2e'),
    border_radius('12px'),
    padding('1.5rem'),
    box_shadow('0 4px 20px rgba(0,0,0,0.3)')
]).

%% style(+Component, +Selector, +Properties)
style(chart_container, '.legend', [
    font_size('0.85rem'),
    color('#888')
]).

style(chart_container, ':hover', [
    box_shadow('0 6px 24px rgba(0,0,0,0.4)')
]).
```

#### Theme System

```prolog
%% theme(+ThemeName, +Properties)
theme(dark, [
    background('#1a1a2e'),
    surface('#16213e'),
    text('#e0e0e0'),
    text_secondary('#888'),
    accent('#00d4ff'),
    accent_secondary('#7c3aed'),
    border('rgba(255,255,255,0.1)'),
    shadow('rgba(0,0,0,0.3)')
]).

theme(light, [
    background('#f8fafc'),
    surface('#ffffff'),
    text('#1a1a2e'),
    text_secondary('#64748b'),
    accent('#7c3aed'),
    accent_secondary('#00d4ff'),
    border('#e2e8f0'),
    shadow('rgba(0,0,0,0.1)')
]).

%% Apply theme to component
component_theme(curve_demo, dark).
```

#### CSS Variables Generation

Themes generate CSS custom properties:

```css
/* Generated from theme(dark, [...]) */
:root {
    --background: #1a1a2e;
    --surface: #16213e;
    --text: #e0e0e0;
    --accent: #00d4ff;
    /* ... */
}
```

### HTML Structure Customization

#### Wrapper Templates

```prolog
%% wrapper(+ComponentType, +Variant, +Template)
% ~class~ = generated class name
% ~content~ = component content
% ~id~ = component id

wrapper(chart, default, '<div class="~class~">~content~</div>').

wrapper(chart, with_toolbar, '
<div class="~class~">
    <div class="toolbar">~toolbar~</div>
    <div class="chart-content">~content~</div>
</div>
').

wrapper(chart, card, '
<div class="~class~ card">
    <div class="card-header">~header~</div>
    <div class="card-body">~content~</div>
</div>
').
```

#### Slot System

```prolog
%% slots(+Component, +SlotDefs)
slots(curve_demo, [
    slot(header, [optional, default(none)]),
    slot(controls, [required, default(auto_controls)]),
    slot(chart, [required]),
    slot(footer, [optional, default(none)])
]).

%% slot_content(+Component, +SlotName, +Content)
slot_content(curve_demo, header, title("Curve Plotting Demo")).
slot_content(curve_demo, controls, [
    slider(amplitude, [min(0), max(5)]),
    slider(frequency, [min(0.1), max(10)])
]).
```

### Control Generation

```prolog
%% control(+Name, +Type, +Options)
control(amplitude, slider, [
    min(0), max(5), step(0.1), default(1),
    label("Amplitude")
]).

control(curve_type, select, [
    options([sine, cosine, quadratic, exponential]),
    default(sine),
    label("Curve Type")
]).

control(show_grid, checkbox, [
    default(true),
    label("Show Grid")
]).

control(color, color_picker, [
    default('#00d4ff'),
    label("Line Color")
]).

%% control_panel(+Name, +Controls)
control_panel(curve_controls, [amplitude, frequency, curve_type, show_grid]).
```

### Default Layouts

```prolog
%% default_layout(+Pattern, +Strategy, +Options)

% Single component, full width
default_layout(single, grid, [
    areas([["content"]]),
    columns(["1fr"]),
    rows(["1fr"])
]).

% Sidebar + Content
default_layout(sidebar_content, grid, [
    areas([["sidebar", "content"]]),
    columns(["320px", "1fr"]),
    rows(["1fr"]),
    gap("0")
]).

% Header + Content + Footer
default_layout(header_content_footer, grid, [
    areas([["header"], ["content"], ["footer"]]),
    columns(["1fr"]),
    rows(["auto", "1fr", "auto"])
]).

% Full dashboard
default_layout(dashboard, grid, [
    areas([
        ["header", "header"],
        ["sidebar", "content"],
        ["sidebar", "footer"]
    ]),
    columns(["280px", "1fr"]),
    rows(["60px", "1fr", "40px"])
]).
```

### Subplot Layouts (Internal Component Layout)

Subplot layouts define internal component arrangements - multiple charts/graphs in a grid within a single component. This is distinct from outer layouts which position containers.

```prolog
%% subplot_layout(+Name, +Strategy, +Options)
subplot_layout(comparison_demo, grid, [
    rows(2),
    cols(2),
    gap("1rem"),
    figsize(10, 8)  % matplotlib-specific
]).

%% subplot_content(+Name, +Position, +Content)
subplot_content(comparison_demo, pos(1,1), [curve(sine), title("Sine")]).
subplot_content(comparison_demo, pos(1,2), [curve(cosine), title("Cosine")]).
subplot_content(comparison_demo, pos(2,1), [curve(quadratic), title("Quadratic")]).
subplot_content(comparison_demo, pos(2,2), [curve(exponential), title("Exponential")]).
```

#### Target-Aware Generation

The subplot system generates different output based on target capabilities:

**Web targets (Chart.js/Cytoscape)** - Synthesized nested CSS grid:
```css
.comparison-demo-subplot-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 1rem;
}
```

**Matplotlib** - Native subplot support:
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, sine_y, label="sine")
axes[0, 0].set_title("Sine")
# ...
plt.tight_layout()
```

This allows the same declarative specification to work across targets, using native features when available and synthesizing when not.

## Implementation Plan

### Phase 1: Core Layout Generator - COMPLETE

1. [x] Create `layout_generator.pl` module
2. [x] Implement `layout/3` predicate parsing
3. [x] Generate CSS Grid from specs
4. [x] Generate CSS Flexbox from specs
5. [x] Handle absolute positioning

### Phase 2: Style System - COMPLETE

1. [x] Implement `style/2` and `style/3` predicates
2. [x] Implement `theme/2` system (dark, light, midnight themes)
3. [x] Generate CSS variables from themes
4. [x] Implement style merging (defaults + overrides)

### Phase 3: HTML Structure - COMPLETE

1. [x] Implement `wrapper/3` template system
2. [x] Implement `place/3` for component placement
3. [x] Generate HTML and JSX output

### Phase 4: Subplot Layout System - COMPLETE

1. [x] Implement `subplot_layout/3` and `subplot_content/3`
2. [x] Generate synthesized CSS grid for web targets
3. [x] Generate native matplotlib subplots
4. [x] Target-aware generation dispatch

### Phase 5: Integration - COMPLETE

1. [x] Update `graph_generator.pl` to use layout system
2. [x] Update `curve_plot_generator.pl` to use layout system
3. [x] Matplotlib uses native subplot system (not CSS layouts)
4. [x] Add integration tests (75 tests passing)

### Phase 6: Control Generation - COMPLETE

1. [x] Implement `control/3` predicates
2. [x] Implement `control_panel/2`
3. [x] Generate React control components (slider, select, checkbox, color_picker, number_input, text_input)
4. [x] Wire controls to component state via `generate_wired_component/3`
5. [x] Add TypeScript interface generation via `generate_prop_types/2`
6. [x] Add 24 new tests (99 tests passing total)

### Phase 7: Responsive Design System - COMPLETE

1. [x] Create `responsive_generator.pl` module
2. [x] Implement breakpoint definitions (xs, sm, md, lg, xl, xxl, mobile, tablet, desktop, wide)
3. [x] Generate CSS media queries from breakpoint specs
4. [x] Implement responsive layout variants with `at(breakpoint, options)` syntax
5. [x] Add container query support for modern CSS
6. [x] Default responsive layouts (collapsible_sidebar, adaptive_stack, card_grid, dashboard)
7. [x] Add 10 responsive tests (109 tests passing)

### Phase 8: Accessibility Features - COMPLETE

1. [x] Create `accessibility_generator.pl` module
2. [x] Implement ARIA specifications for visualization components
3. [x] Generate keyboard navigation handlers (arrow keys, Enter, Escape, Tab)
4. [x] Implement focus trap generation for modals and dialogs
5. [x] Add live region announcements for screen readers
6. [x] Generate skip links for navigation
7. [x] Accessibility CSS (screen reader only, focus styles, reduced motion)
8. [x] Add 15 accessibility tests (164 tests passing total)

### Phase 9: Animation System - COMPLETE

1. [x] Create `animation_generator.pl` module
2. [x] Implement keyframe animation definitions (fade, scale, slide, rotate, pulse)
3. [x] Add transition effect specifications with hover/focus states
4. [x] Implement easing function library (linear, ease, cubic-bezier variants)
5. [x] Generate @keyframes CSS and animation classes
6. [x] Add chart-specific animations (draw_line, bar_grow, pie_slice)
7. [x] React hook and component generation for animations
8. [x] Add 18 animation tests (182 tests passing total)

## Generated Output Examples

### Input

```prolog
layout(my_demo, grid, [
    areas([["sidebar", "main"]]),
    columns(["320px", "1fr"]),
    gap("1rem")
]).

style(my_demo, [
    background('#1a1a2e'),
    min_height('100vh')
]).

theme(my_demo, dark).

place(my_demo, sidebar, [control_panel]).
place(my_demo, main, [chart]).
```

### Generated CSS

```css
.my-demo {
    display: grid;
    grid-template-areas: "sidebar main";
    grid-template-columns: 320px 1fr;
    gap: 1rem;
    background: #1a1a2e;
    min-height: 100vh;
}

.my-demo__sidebar {
    grid-area: sidebar;
}

.my-demo__main {
    grid-area: main;
}

/* Theme variables */
.my-demo {
    --background: #1a1a2e;
    --text: #e0e0e0;
    --accent: #00d4ff;
    /* ... */
}
```

### Generated React Component

```tsx
export const MyDemo: React.FC = () => {
    return (
        <div className={styles.myDemo}>
            <div className={styles.myDemo__sidebar}>
                <ControlPanel />
            </div>
            <div className={styles.myDemo__main}>
                <Chart />
            </div>
        </div>
    );
};
```

## Testing Strategy

1. **Unit tests**: Each layout strategy generates correct CSS
2. **Integration tests**: Full component generation with layouts
3. **Snapshot tests**: Generated CSS/HTML matches expected output
4. **Visual regression**: Rendered output looks correct (manual/Playwright)

## Implemented Features

### Responsive Design (Phase 7) - COMPLETE

Declarative breakpoint specifications for adaptive layouts via `responsive_generator.pl`:

```prolog
% Use predefined breakpoints
breakpoint(mobile, max_width(767)).
breakpoint(tablet, range(768, 1023)).
breakpoint(desktop, min_width(1024)).

% Define responsive layouts with breakpoint variants
responsive_layout(my_layout, [
    default([strategy(grid), columns(["320px", "1fr"])]),
    at(mobile, [columns(["1fr"])]),
    at(tablet, [columns(["280px", "1fr"])])
]).

% Generate responsive CSS
?- generate_responsive_css(my_layout, CSS).
% Generates @media queries for each breakpoint

% Container queries for component-level responsiveness
container(chart_container, [type(inline_size), name(chart)]).
?- generate_container_css(chart_container, CSS).
```

### Accessibility Features (Phase 8) - COMPLETE

ARIA attributes, keyboard navigation, and focus management via `accessibility_generator.pl`:

```prolog
% Define ARIA specifications
aria_spec(line_chart, [
    role(img),
    label("Interactive line chart"),
    describedby(chart_description)
]).

% Define keyboard navigation
keyboard_nav(data_table, [
    key('ArrowUp', 'moveFocus("up")'),
    key('ArrowDown', 'moveFocus("down")'),
    key('Enter', 'activateCell()'),
    key('Escape', 'exitEditMode()')
]).

% Focus trap for modals
focus_trap(modal_dialog, [
    container('.modal'),
    initial_focus('.modal-close'),
    escape_deactivates(false)
]).

% Generate accessibility code
?- generate_aria_props(line_chart, Props).
?- generate_keyboard_handler(data_table, Handler).
?- generate_focus_trap_jsx(modal_dialog, JSX).
?- generate_accessibility_css(line_chart, CSS).
```

### Animation System (Phase 9) - COMPLETE

Declarative transitions and animations via `animation_generator.pl`:

```prolog
% Keyframe animations with easing
animation(fade_in, [
    duration(300),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0)]),
        frame(100, [opacity(1)])
    ])
]).

% Transitions with hover/focus states
transition(hover_lift, [
    properties([transform, box_shadow]),
    duration(200),
    easing(ease_out),
    on_hover([
        transform('translateY(-2px)'),
        box_shadow('0 4px 12px rgba(0,0,0,0.15)')
    ])
]).

% Chart-specific animations
animation(draw_line, [
    duration(1500),
    easing(ease_out),
    keyframes([
        frame(0, [stroke_dashoffset(1000)]),
        frame(100, [stroke_dashoffset(0)])
    ])
]).

% Generate CSS and React components
?- generate_animation_css(fade_in, CSS).
?- generate_transition_css(hover_lift, TransCSS).
?- generate_animation_hook(fade_in, Hook).
```

## Future Work

The following enhancements are planned for future development:

### Interactive Visualizations
Event handling, tooltips, and interactive controls:
```prolog
interaction(chart, [
    on_hover(show_tooltip),
    on_click(select_point),
    on_drag(pan_view),
    on_scroll(zoom)
]).
```

### Live Preview
Development server with hot-reload for visualization prototyping.

## References

- CSS Grid: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Grid_Layout
- CSS Flexbox: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Flexible_Box_Layout
- BEM naming: https://getbem.com/
