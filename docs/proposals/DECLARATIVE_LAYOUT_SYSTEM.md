# Declarative Layout and Styling System

**Status:** Proposal
**Author:** Claude Code
**Date:** 2025-12-31

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

## Implementation Plan

### Phase 1: Core Layout Generator

1. Create `layout_generator.pl` module
2. Implement `layout/3` predicate parsing
3. Generate CSS Grid from specs
4. Generate CSS Flexbox from specs
5. Handle absolute positioning

### Phase 2: Style System

1. Implement `style/2` and `style/3` predicates
2. Implement `theme/2` system
3. Generate CSS variables from themes
4. Implement style merging (defaults + overrides)

### Phase 3: HTML Structure

1. Implement `wrapper/3` template system
2. Implement `slots/2` and `slot_content/3`
3. Implement `place/3` for component placement

### Phase 4: Control Generation

1. Implement `control/3` predicates
2. Implement `control_panel/2`
3. Generate React control components
4. Wire controls to component state

### Phase 5: Integration

1. Update `graph_generator.pl` to use layout system
2. Update `curve_plot_generator.pl` to use layout system
3. Update `matplotlib_generator.pl` (Python has different patterns)
4. Add integration tests

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

## Open Questions

1. **Responsive design**: How to specify breakpoints declaratively?
2. **Animation**: Should transitions/animations be declarative?
3. **Accessibility**: How to ensure generated HTML is accessible?
4. **CSS-in-JS vs CSS Modules**: Which output format for React?

## References

- CSS Grid: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Grid_Layout
- CSS Flexbox: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Flexible_Box_Layout
- BEM naming: https://getbem.com/
