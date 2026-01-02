# Declarative Layout and Styling System

**Status:** Implemented (13 phases complete)
**Author:** Claude Code
**Date:** 2026-01-02
**Tests:** 285 (280 passing)

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

### Phase 10: Interactive Visualizations - COMPLETE

1. [x] Create `interaction_generator.pl` module
2. [x] Implement event handling specifications (hover, click, drag, scroll)
3. [x] Add tooltip generation with positioning and styling
4. [x] Implement zoom controls with min/max scale and reset
5. [x] Add pan handler with inertia support
6. [x] Implement drag handlers (free, rotate, node_move modes)
7. [x] Add selection handlers (single, multi, brush selection)
8. [x] Generate interaction state management hooks
9. [x] Add 19 interaction tests (201 tests passing total)

### Phase 11: Export Capabilities - COMPLETE

1. [x] Create `export_generator.pl` module
2. [x] Implement SVG export with inline styles
3. [x] Add PNG export with canvas rendering and scaling
4. [x] Add PDF export using jsPDF library
5. [x] Support JSON and CSV data export
6. [x] Generate export menu UI components
7. [x] Generate export CSS styles
8. [x] Add 22 export tests (223 tests passing total)

### Phase 12: Live Preview System - COMPLETE

1. [x] Create `live_preview_generator.pl` module
2. [x] Implement dev server configuration with Vite
3. [x] Add WebSocket-based hot-reload support
4. [x] Generate preview application with split layout
5. [x] Implement React hooks for hot reload and state sync
6. [x] Generate error boundary wrapper component
7. [x] Add code editor with syntax highlighting
8. [x] Generate preview CSS with theme support
9. [x] Add 23 live preview tests (246 tests passing total)

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

### Interactive Visualizations (Phase 10) - COMPLETE

Event handling, tooltips, and interactive controls via `interaction_generator.pl`:

```prolog
% Define chart interactions
interaction(scatter_plot, [
    on_hover(show_tooltip),
    on_click(select_point),
    on_brush(select_range),
    on_scroll(zoom),
    on_drag(pan)
]).

% Tooltip specifications
tooltip_spec(scatter_plot, [
    position(cursor),
    offset(12, 12),
    delay(150),
    content([
        field(x, "X"),
        field(y, "Y"),
        field(label, "Label")
    ])
]).

% Zoom and pan specifications
zoom_spec(scatter_plot, [
    enabled(true),
    min_scale(0.5),
    max_scale(20),
    controls([zoom_in, zoom_out, zoom_fit, reset])
]).

% Generate React components and hooks
?- generate_event_handlers(scatter_plot, Handlers).
?- generate_tooltip_jsx(scatter_plot, TooltipJSX).
?- generate_zoom_controls(scatter_plot, ZoomControls).
?- generate_pan_handler(scatter_plot, PanHandler).
?- generate_selection_handler(scatter_plot, SelectionHandler).
```

### Export Capabilities (Phase 11) - COMPLETE

Export visualizations to various formats via `export_generator.pl`:

```prolog
% Configure export for a chart type
export_config(scatter_plot, [
    formats([svg, png, pdf, json]),
    filename_template("scatter-{timestamp}"),
    default_size(800, 800),
    scale(2)
]).

% Generate export components
?- generate_export_component(scatter_plot, Component).
?- generate_export_hook(scatter_plot, Hook).
?- generate_export_menu(scatter_plot, Menu).
?- generate_export_css(CSS).

% Individual format exports
?- generate_svg_export(scatter_plot, SVGCode).
?- generate_png_export(scatter_plot, PNGCode).
?- generate_pdf_export(scatter_plot, PDFCode).
```

### Live Preview System (Phase 12) - COMPLETE

Development server with hot-reload for visualization prototyping via `live_preview_generator.pl`:

```prolog
% Configure dev server
dev_server_config(visualization_preview, [
    port(3001),
    hot_reload(true),
    watch_paths(['src/unifyweaver/glue/**/*.pl', 'src/**/*.tsx'])
]).

% Configure preview layout
preview_config(chart_preview, [
    layout(split),
    editor_position(left),
    show_console(true),
    show_props_panel(true),
    theme(dark)
]).

% Generate dev server and preview components
?- generate_dev_server(visualization_preview, ServerCode).
?- generate_vite_config(visualization_preview, ViteConfig).
?- generate_preview_app(chart_preview, PreviewApp).
?- generate_hot_reload_hook(HotReloadHook).
?- generate_preview_css(CSS).
```

## Future Work

All major visualization system features are now complete. Future enhancements are categorized below:

### Phase 13: Data Binding System - COMPLETE

Declarative bindings between Prolog facts and visualization props via `data_binding_generator.pl`:

```prolog
% Define a data source
data_source(sales_data, [
    predicate(sales_record/4),
    fields([date, product, quantity, amount]),
    primary_key(date),
    refresh_interval(5000)
]).

% Bind to a chart component
binding(sales_chart, sales_data, [
    x_axis(date),
    y_axis(amount),
    series(product)
]).

% Two-way binding for editable components
two_way_binding(data_table, sales_data, [
    columns([date, product, amount]),
    editable([amount]),
    on_edit(update_record),
    debounce(300)
]).

% Computed/derived sources
computed_source(sales_summary, [
    base_source(sales_data),
    computation(aggregate),
    group_by([product]),
    aggregations([sum(amount, total), count(_, record_count)])
]).

% Generate React hooks and components
?- generate_binding_hook(sales_chart, Hook).
?- generate_data_provider(sales_data, Provider).
?- generate_websocket_sync(sales_data, SyncCode).
?- generate_mutation_handler(sales_data, Handler).
?- generate_computed_hook(sales_summary, ComputedHook).
```

### Phase 14: Theme System (Complete)
Centralized theme definitions reusable across visualizations:
- Theme definitions with color palettes (Tailwind-style c50-c950), typography, spacing
- Theme inheritance and composition via `extends(parent)`
- Runtime theme switching with localStorage persistence
- CSS custom properties generation

**Implementation:** `src/unifyweaver/glue/theme_generator.pl`

**Example Usage:**
```prolog
% Define a custom theme extending light
theme(corporate, [
    extends(light),
    colors([primary('#1e40af'), secondary('#475569')])
]).

% Generate theme CSS
?- generate_theme_css(corporate, CSS).

% Generate React theme provider
?- generate_theme_provider([light, dark, corporate], Provider).

% Generate useTheme hook
?- generate_theme_hook(Hook).
```

### Phase 15: Animation Presets (Complete)
Library of reusable animation patterns:
- Entry/exit animations (fade_in, slide_in_*, bounce_in, scale_in, flip_in_*)
- Attention animations (pulse, bounce, shake, wiggle, heartbeat, jello)
- Chart-specific animations (chart_draw, bar_grow, pie_reveal, data_point_pop)
- Transition presets (smooth, snappy, elastic, spring)
- Animation composition and sequencing

**Implementation:** `src/unifyweaver/glue/animation_presets.pl`

**Example Usage:**
```prolog
% Generate CSS for a preset
?- generate_preset_css(fade_in, CSS).

% Generate all presets CSS
?- generate_all_presets_css(AllCSS).

% Generate useAnimation React hook
?- generate_preset_hook(Hook).

% Compose multiple presets
?- compose_presets([fade_in, scale_in], [duration(500)], Combined).

% Create animation sequence with staggered timing
?- sequence_presets([fade_in, slide_in_up, scale_in], [stagger(100)], Sequence).
```

### Phase 16: Template Library (Complete)
Pre-built visualization templates:
- Dashboard templates (analytics, sales, realtime monitor)
- Report templates with print optimization (monthly, comparison)
- Data explorer interfaces (data explorer, chart explorer)
- Presentation slides with animations

**Implementation:** `src/unifyweaver/glue/template_library.pl`

**Example Usage:**
```prolog
% Generate complete template bundle (JSX, CSS, Types, Hook)
?- generate_template(analytics_dashboard, Code).

% Generate template JSX component
?- generate_template_jsx(sales_dashboard, JSX).

% Generate template CSS with responsive breakpoints
?- generate_template_css(analytics_dashboard, CSS).

% Generate data management hook
?- generate_template_hook(realtime_monitor, Hook).

% Check template features
?- template_has_feature(analytics_dashboard, export_pdf).

% Generate print-optimized styles
?- generate_print_styles(PrintCSS).
```

### Performance Enhancements (Complete)
- **Lazy loading** - On-demand loading with pagination, infinite scroll, chunked loading
- **Virtual scrolling** - Efficient rendering for lists, tables, and grids
- **WebWorker support** - Background computation for data processing and chart calculations

**Implementations:**
- `src/unifyweaver/glue/lazy_loading_generator.pl` - Lazy loading patterns
- `src/unifyweaver/glue/virtual_scroll_generator.pl` - Virtual scrolling components
- `src/unifyweaver/glue/webworker_generator.pl` - WebWorker generation

**Example Usage:**
```prolog
% Generate lazy data hook with caching
?- generate_lazy_hook(default, Hook).

% Generate pagination controls
?- generate_pagination_hook(default, PagHook).

% Generate infinite scroll with intersection observer
?- generate_infinite_scroll(infinite_scroll, InfHook).

% Generate virtual scroll for large lists
?- generate_virtual_scroll_hook(default, VirtualHook).

% Generate virtualized table with sorting
?- generate_virtual_table(large_table, Table).

% Generate virtualized grid for cards
?- generate_virtual_grid(card_grid, Grid).

% Generate data processor WebWorker
?- generate_worker(data_processor, Worker).

% Generate worker pool for parallel processing
?- generate_worker_pool(default, Pool).

% Generate chart calculation worker (interpolation, downsampling)
?- generate_chart_worker(ChartWorker).
```

### Infrastructure (Planned)
- **CI/CD integration** - Automated testing for visualization generators
- **Storybook integration** - Component documentation and visual testing
- **E2E testing** - Playwright/Cypress tests for generated components

### Additional Chart Types (Planned)
- Radar/spider charts
- Funnel charts
- Gauge/meter charts
- Sankey diagrams
- Chord diagrams

## References

- CSS Grid: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Grid_Layout
- CSS Flexbox: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Flexible_Box_Layout
- BEM naming: https://getbem.com/
