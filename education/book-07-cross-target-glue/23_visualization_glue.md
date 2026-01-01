<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 23: Declarative Visualization Glue

**Generating React Components and Python Plots from Prolog Specifications**

This chapter covers six declarative glue modules that generate visualization code:

- **graph_generator.pl** - Cytoscape.js graph visualization with React/TypeScript
- **curve_plot_generator.pl** - Chart.js curve plotting with React/TypeScript
- **matplotlib_generator.pl** - Python matplotlib code generation
- **heatmap_generator.pl** - Heatmap visualization with React and seaborn
- **treemap_generator.pl** - Hierarchical treemap visualization with React and Plotly
- **plot3d_generator.pl** - 3D surface, scatter, and line plots with Plotly.js

## Overview

The visualization glue modules follow the UnifyWeaver pattern of declarative specifications that generate target-language code. Instead of manually writing React components or Python plotting scripts, you define your data and configuration in Prolog, then generate the complete implementation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prolog Specifications                        │
│  graph_node/2, graph_edge/3, curve/2, matplotlib_curve/2, etc. │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ React/TS +    │ │ React/TS +    │ │ Python +      │
    │ Cytoscape.js  │ │ Chart.js      │ │ Matplotlib    │
    └───────────────┘ └───────────────┘ └───────────────┘
```

## Graph Generator

The graph generator creates interactive network visualizations using Cytoscape.js within React components.

### Defining Nodes and Edges

```prolog
:- use_module('src/unifyweaver/glue/graph_generator').

% Define nodes with properties
graph_node(abraham, [label("Abraham"), type(person), generation(1)]).
graph_node(isaac, [label("Isaac"), type(person), generation(2)]).
graph_node(jacob, [label("Jacob"), type(person), generation(3)]).

% Define edges with relationships
graph_edge(abraham, isaac, [relation(parent), via(sarah)]).
graph_edge(isaac, jacob, [relation(parent), via(rebekah)]).
```

### Graph Specifications

```prolog
% Define a complete graph specification
graph_spec(family_tree, [
    title("Family Tree"),
    layout(cose),              % Force-directed layout
    theme(dark),
    nodes([abraham, sarah, isaac, rebekah, jacob, esau])
]).
```

### Code Generation

```prolog
% Generate React/TypeScript component
?- generate_graph_component(family_tree, ComponentCode).

% Generate Cytoscape.js configuration
?- generate_cytoscape_config(family_tree, Config).

% Generate CSS module
?- generate_graph_styles(family_tree, CssCode).

% Generate graph data as JavaScript
?- generate_graph_data(family_tree, DataCode).
```

### Generated Component Structure

The generator produces a complete React component with:

- TypeScript interfaces for type safety
- Cytoscape.js initialization and configuration
- CSS modules for styling
- Responsive container layout
- Interactive features (pan, zoom, select)

```typescript
// Generated: FamilyTreeGraph.tsx
import React, { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import styles from './FamilyTreeGraph.module.css';

interface GraphProps {
  onNodeClick?: (nodeId: string) => void;
}

export const FamilyTreeGraph: React.FC<GraphProps> = ({ onNodeClick }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  // ... component implementation
};
```

### Available Layouts

| Layout | Description |
|--------|-------------|
| `cose` | Force-directed layout (default) |
| `breadthfirst` | Hierarchical tree layout |
| `circle` | Circular arrangement |
| `grid` | Regular grid layout |
| `dagre` | Directed acyclic graph (requires plugin) |

## Curve Plot Generator

The curve plot generator creates interactive mathematical curve visualizations using Chart.js.

### Defining Curves

Curves can be defined in three ways: mathematical expressions, explicit data points (MATLAB-style), or predefined types.

#### Method 1: Mathematical Expressions (Recommended)

Use `expr()` to define curves with arbitrary mathematical expressions:

```prolog
:- use_module('src/unifyweaver/glue/curve_plot_generator').

% Gaussian curve
curve(gaussian, [
    expr(exp(-(x^2) / 2)),
    color('#10b981'),
    label("Gaussian")
]).

% Damped oscillation
curve(damped_sine, [
    expr(exp(-abs(x) / 3) * sin(x * 2)),
    color('#6366f1'),
    label("Damped sine")
]).

% Rational function (Cauchy distribution)
curve(rational, [
    expr(1 / (1 + x^2)),
    color('#14b8a6'),
    label("1/(1+x²)")
]).

% Higher-order polynomial
curve(polynomial, [
    expr(x^4 - 2*x^2 + 0.5),
    color('#ec4899'),
    label("x⁴ - 2x² + 0.5")
]).
```

#### Method 2: Explicit Data Points (MATLAB-Style)

For measured or sampled data:

```prolog
curve(sampled_data, [
    data([-2, -1, 0, 1, 2, 3, 4],    % X values
         [4, 1, 0, 1, 4, 9, 16]),     % Y values
    color('#f97316'),
    label("Measured Data")
]).

curve(experimental, [
    data([0, 0.5, 1.0, 1.5, 2.0],
         [0.0, 0.48, 0.84, 1.0, 0.91]),
    color('#8b5cf6'),
    label("Experimental")
]).
```

#### Method 3: Predefined Types

For common curve types with parameters:

```prolog
% Trigonometric curves
curve(sine_wave, [
    type(sine),
    amplitude(1),
    frequency(1),
    phase(0),
    color('#00d4ff'),
    label("sin(x)")
]).

curve(cosine_wave, [
    type(cosine),
    amplitude(1),
    frequency(1),
    phase(0),
    color('#ff6b6b'),
    label("cos(x)")
]).

% Polynomial curves
curve(parabola, [
    type(quadratic),
    a(1), b(0), c(0),
    color('#22c55e'),
    label("y = x²")
]).
```

### Curve Evaluation

The module includes runtime curve evaluation:

```prolog
% Evaluate a curve at a specific x value
?- evaluate_curve(sine_wave, 0, Y).
Y = 0.0.

?- evaluate_curve(sine_wave, 1.5708, Y).  % π/2
Y = 1.0.

?- evaluate_curve(parabola, 3, Y).
Y = 9.0.
```

### Plot Specifications

```prolog
plot_spec(trig_demo, [
    title("Trigonometric Functions"),
    curves([sine_wave, cosine_wave]),
    x_range(-6.28, 6.28),    % -2π to 2π
    y_range(-1.5, 1.5),
    theme(dark),
    points(200)              % Number of sample points
]).
```

### Code Generation

```prolog
% Generate React/TypeScript component
?- generate_curve_component(trig_demo, ComponentCode).

% Generate Chart.js configuration
?- generate_chartjs_config(trig_demo, Config).

% Generate CSS module
?- generate_curve_styles(trig_demo, CssCode).
```

### Supported Curve Types

| Type | Formula | Parameters |
|------|---------|------------|
| `linear` | y = mx + b | m, b |
| `quadratic` | y = ax² + bx + c | a, b, c |
| `cubic` | y = ax³ + bx² + cx + d | a, b, c, d |
| `sine` | y = A·sin(ωx + φ) | amplitude, frequency, phase |
| `cosine` | y = A·cos(ωx + φ) | amplitude, frequency, phase |
| `exponential` | y = base·e^(scale·x) | base, scale |
| `logarithm` | y = log_base(x) | base |
| `absolute` | y = \|x\| | (none) |

## Matplotlib Generator

The matplotlib generator produces Python scripts for publication-quality plots.

### Defining Curves

```prolog
:- use_module('src/unifyweaver/glue/matplotlib_generator').

matplotlib_curve(sine, [
    type(sine),
    amplitude(1),
    frequency(1),
    phase(0),
    color('blue'),
    linestyle('-'),
    label("sin(x)")
]).

matplotlib_curve(gaussian, [
    type(gaussian),
    mu(0),
    sigma(1),
    color('navy'),
    linestyle('-'),
    label("Gaussian")
]).
```

### Plot Specifications

```prolog
matplotlib_plot(trig_functions, [
    title("Trigonometric Functions"),
    curves([sine, cosine]),
    x_range(-6.28318, 6.28318),
    y_range(-1.5, 1.5),
    style(seaborn),
    figsize(10, 6),
    grid(true),
    legend(true),
    xlabel("x"),
    ylabel("y"),
    output(show)              % or: png("output.png"), pdf("output.pdf")
]).
```

### Code Generation

```prolog
% Generate Python code
?- generate_matplotlib_code(trig_functions, PythonCode).

% Generate complete runnable script
?- generate_matplotlib_script(trig_functions, Script).
```

### Generated Python Code

```python
#!/usr/bin/env python3
# Generated by UnifyWeaver matplotlib_generator

import numpy as np
import matplotlib.pyplot as plt

def plot_trig_functions():
    """Trigonometric Functions"""
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(-6.28318, 6.28318, 200)

    # sine curve
    y_sine = np.sin(x)
    ax.plot(x, y_sine, color='blue', linestyle='-', label='sin(x)')

    # cosine curve
    y_cosine = np.cos(x)
    ax.plot(x, y_cosine, color='orange', linestyle='-', label='cos(x)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trigonometric Functions')
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    ax.legend()

    plt.show()

if __name__ == '__main__':
    plot_trig_functions()
```

### NumPy Expression Generation

The generator converts curve types to NumPy expressions:

```prolog
% Get NumPy expression for a curve type
?- matplotlib_curve(sine, Props),
   generate_numpy_expression(sine, Props, Expr).
Expr = 'np.sin(x)'.

?- matplotlib_curve(gaussian, Props),
   generate_numpy_expression(gaussian, Props, Expr).
Expr = 'np.exp(-((x - 0) ** 2) / (2 * 1 ** 2))'.
```

### Output Formats

| Format | Example | Description |
|--------|---------|-------------|
| `show` | `output(show)` | Display interactive window |
| `png(File)` | `output(png("plot.png"))` | Save as PNG image |
| `pdf(File)` | `output(pdf("plot.pdf"))` | Save as PDF document |
| `svg(File)` | `output(svg("plot.svg"))` | Save as SVG vector graphic |

## Integration with Other Glue Modules

The visualization modules integrate with other UnifyWeaver glue modules:

### With Express Generator

```prolog
% Define an API endpoint that returns graph data
api_endpoint('/api/graph/:name', get, [
    handler(graph_data_handler),
    response_type(json)
]).

% Handler generates graph data from Prolog
graph_data_handler(Request, Response) :-
    member(name=Name, Request.params),
    generate_graph_data(Name, Data),
    Response = json(Data).
```

### With React Generator

```prolog
% Include graph component in a React page
react_page(dashboard, [
    components([
        graph_component(family_tree, [interactive(true)]),
        curve_component(trig_demo, [responsive(true)])
    ])
]).
```

## Layout System

The visualization modules integrate with a declarative layout system for composing UIs.

### Outer Layouts (Container Positioning)

```prolog
:- use_module('src/unifyweaver/glue/layout_generator').

% Define a sidebar + content layout
layout(my_dashboard, grid, [
    areas([["sidebar", "main"]]),
    columns(["320px", "1fr"]),
    gap("1rem")
]).

% Place components in regions
place(my_dashboard, sidebar, [controls]).
place(my_dashboard, main, [chart]).

% Generate with layout
?- generate_graph_with_layout(family_tree, sidebar_content, Code).
?- generate_curve_with_layout(trig_demo, dashboard, Code).
```

### Subplot Layouts (Internal Component Grids)

For composite components with multiple charts/graphs:

```prolog
% Define a 2x2 subplot grid
subplot_layout(comparison_demo, grid, [rows(2), cols(2)]).

% Place content in cells
subplot_content(comparison_demo, pos(1,1), [curve(sine), title("Sine")]).
subplot_content(comparison_demo, pos(1,2), [curve(cosine), title("Cosine")]).
subplot_content(comparison_demo, pos(2,1), [curve(quadratic), title("Quadratic")]).
subplot_content(comparison_demo, pos(2,2), [curve(exponential), title("Exp")]).

% Generate - target-aware!
?- generate_subplot_css(comparison_demo, CSS).      % Web: nested CSS grid
?- generate_subplot_matplotlib(comparison_demo, Code). % Python: native subplots
```

The layout system is target-aware:
- **Web targets**: Synthesizes nested CSS grids with multiple chart instances
- **Matplotlib**: Uses native `plt.subplots()` for efficient multi-plot figures

## Control System

The control system provides declarative UI controls that integrate with visualizations.

### Defining Controls

```prolog
:- use_module('src/unifyweaver/glue/layout_generator').

% Slider control for numeric values
control(amplitude, slider, [
    min(0), max(5), step(0.1),
    default(1),
    label("Amplitude")
]).

% Dropdown select
control(curve_type, select, [
    options([sine, cosine, quadratic, cubic, exponential]),
    default(sine),
    label("Curve Type")
]).

% Checkbox for boolean values
control(show_grid, checkbox, [
    default(true),
    label("Show Grid")
]).

% Color picker
control(line_color, color_picker, [
    default('#00d4ff'),
    label("Line Color")
]).
```

### Control Panels

Group related controls into panels:

```prolog
% Control panel for curve parameters
control_panel(curve_controls, [amplitude, frequency, phase, curve_type]).

% Control panel for display settings
control_panel(display_controls, [show_grid, show_legend, line_color, line_width]).
```

### Generating Control JSX

```prolog
% Generate individual control
?- generate_control_jsx(amplitude, JSX).
% Produces slider input with label and onChange handler

% Generate entire control panel
?- generate_control_panel_jsx(curve_controls, PanelJSX).
% Produces panel with all controls grouped

% Generate React useState declarations
?- generate_control_state(curve_controls, StateCode).
% const [amplitude, setAmplitude] = useState(1);
% const [frequency, setFrequency] = useState(1);
% ...
```

### Control Types

| Type | Description | Generated HTML |
|------|-------------|----------------|
| `slider` | Numeric range input | `<input type="range">` |
| `select` | Dropdown selection | `<select><option>...</option></select>` |
| `checkbox` | Boolean toggle | `<input type="checkbox">` |
| `color_picker` | Color selection | `<input type="color">` |
| `number_input` | Numeric text input | `<input type="number">` |
| `text_input` | Text input | `<input type="text">` |

### Wired Components

Generate complete components with controls wired to visualization:

```prolog
% Generate a wired component with sidebar layout
?- generate_wired_component(my_demo, [
       panel(curve_controls),
       component(curve),
       layout(sidebar_content)
   ], Code).
```

This produces:
- React component with useState hooks for each control
- Control panel in sidebar region
- Visualization receiving props from control state
- TypeScript interface for props

### TypeScript Interface Generation

```prolog
?- generate_prop_types(curve_controls, TypesCode).
% interface ChartProps {
%   amplitude: number;
%   frequency: number;
%   phase: number;
%   curveType: string;
% }
```

## Heatmap Generator

The heatmap generator creates grid-based visualizations for correlation matrices, activity data, and more.

### Defining Heatmap Data

```prolog
:- use_module('src/unifyweaver/glue/heatmap_generator').

% Define heatmap configuration
heatmap_spec(correlation_demo, [
    title("Correlation Matrix"),
    x_labels(["Variable A", "Variable B", "Variable C"]),
    y_labels(["Variable A", "Variable B", "Variable C"]),
    color_scale(diverging),
    show_values(true)
]).

% Define cell values (x, y, value)
heatmap_cell(correlation_demo, 0, 0, 1.0).
heatmap_cell(correlation_demo, 0, 1, 0.8).
heatmap_cell(correlation_demo, 1, 0, 0.8).
heatmap_cell(correlation_demo, 1, 1, 1.0).

% Or use rows for efficiency
heatmap_row(activity_demo, 0, [0.1, 0.2, 0.15, 0.18]).
heatmap_row(activity_demo, 1, [0.5, 0.6, 0.55, 0.58]).
```

### Color Scales

| Scale | Description |
|-------|-------------|
| `sequential` | White to blue gradient |
| `diverging` | Blue (-1) to white (0) to red (+1) |
| `viridis` | Perceptually uniform color map |
| `heat` | Black to red to yellow to white |

### Code Generation

```prolog
% Generate React component
?- generate_heatmap_component(correlation_demo, Code).

% Generate Python/seaborn code
?- generate_heatmap_matplotlib(correlation_demo, PyCode).
```

## Treemap Generator

The treemap generator creates hierarchical visualizations using nested rectangles.

### Defining Treemap Data

```prolog
:- use_module('src/unifyweaver/glue/treemap_generator').

% Define treemap configuration
treemap_spec(filesystem_demo, [
    title("Project File Sizes"),
    root(project_root),
    color_by(category),
    show_labels(true)
]).

% Define nodes: id, parent, label, value
treemap_node(project_root, null, "Project", 0).
treemap_node(src, project_root, "src", 0).
treemap_node(main_ts, src, "main.ts", 150).
treemap_node(utils_ts, src, "utils.ts", 80).
```

### Color Modes

| Mode | Description |
|------|-------------|
| `depth` | Color by tree depth level |
| `value` | Color by node value |
| `category` | Color by hash of node ID |

### Code Generation

```prolog
% Generate React component
?- generate_treemap_component(filesystem_demo, Code).

% Generate Python/Plotly code
?- generate_treemap_plotly(filesystem_demo, PyCode).
```

## 3D Plot Generator

The 3D plot generator creates interactive 3D visualizations including surfaces, scatter plots, and lines.

### Defining 3D Surfaces

Surfaces can be defined in three ways: mathematical expressions, explicit data points (MATLAB-style), or predefined functions.

#### Method 1: Mathematical Expressions (Recommended)

Use `expr()` to define surfaces with arbitrary Prolog mathematical expressions that are translated to JavaScript (React/Plotly) or Python (NumPy/matplotlib):

```prolog
:- use_module('src/unifyweaver/glue/plot3d_generator').

% Define a surface using a mathematical expression
surface3d(wave_surface, [
    title("3D Wave Surface"),
    expr(sin(x) * cos(y)),      % z = sin(x) * cos(y)
    x_range(-pi, pi),
    y_range(-pi, pi),
    resolution(50),
    colorscale(viridis)
]).

% More complex expressions
surface3d(gaussian_surface, [
    title("Gaussian"),
    expr(exp(-(x^2 + y^2) / 2)),
    x_range(-3, 3),
    y_range(-3, 3)
]).

% Complex ripple pattern
surface3d(ripple_surface, [
    title("Ripple"),
    expr(sin(sqrt(x^2 + y^2) * 3) / (sqrt(x^2 + y^2) + 0.1)),
    x_range(-5, 5),
    y_range(-5, 5)
]).
```

#### Supported Math Functions

The expression translator supports:

| Category | Functions |
|----------|-----------|
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Exponential | `exp`, `log`, `log10`, `log2` |
| Power/Roots | `^`, `**`, `sqrt`, `cbrt` |
| Rounding | `floor`, `ceil`, `round`, `abs` |
| Comparison | `min`, `max`, `sign` |
| Constants | `pi`, `e` |
| Variables | `x`, `y`, `z`, `t`, `r` |

#### Method 2: Explicit Data Points (MATLAB-Style)

For measured data or pre-computed surfaces, use `data()` with explicit x, y vectors and z matrix:

```prolog
surface3d(measured_data, [
    title("Measured Data Surface"),
    data([0, 1, 2, 3],                    % X vector
         [0, 1, 2, 3],                    % Y vector
         [[0, 1, 4, 9],                   % Z matrix (row-major)
          [1, 2, 5, 10],
          [4, 5, 8, 13],
          [9, 10, 13, 18]]),
    colorscale(viridis)
]).
```

#### Method 3: Predefined Functions (Legacy)

For backward compatibility, predefined functions are still supported:

| Function | Formula |
|----------|---------|
| `sin_cos` | z = sin(x) * cos(y) |
| `paraboloid` | z = x² + y² |
| `saddle` | z = x² - y² |
| `ripple` | z = sin(r*3) / (r+0.1) |
| `gaussian` | z = e^(-(x²+y²)/2) |

### Defining 3D Scatter Plots

```prolog
scatter3d_spec(cluster_demo, [
    title("3D Cluster Visualization"),
    marker_size(8),
    colorscale(portland)
]).

scatter3d_point(cluster_demo, 1.0, 1.0, 1.0, [label("A1"), cluster(1)]).
scatter3d_point(cluster_demo, -1.0, -1.0, 1.0, [label("B1"), cluster(2)]).
```

### Defining 3D Lines

```prolog
line3d_spec(helix, [
    title("3D Helix"),
    line_width(3),
    color('#00d4ff')
]).

% Points are ordered by index
line3d_point(helix, 0, 1.0, 0.0, 0.0).
line3d_point(helix, 1, 0.809, 0.588, 0.2).
line3d_point(helix, 2, 0.309, 0.951, 0.4).
```

### Code Generation

```prolog
% Generate React/Plotly.js component
?- generate_plot3d_component(wave_surface, Code).

% Generate Python/matplotlib code
?- generate_plot3d_matplotlib(wave_surface, PyCode).

% Generate Python/Plotly code
?- generate_plot3d_plotly(wave_surface, PyCode).
```

## Testing

The integration tests verify all visualization glue modules:

```bash
# Run visualization glue tests
swipl -g "run_tests" -t halt tests/integration/glue/test_visualization_glue.pl

# Expected output:
# Results: 128/128 tests passed
# All tests passed!
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| graph_generator | 16 | Node/edge queries, component generation, CSS, config |
| curve_plot_generator | 17 | Curve queries, evaluation, component generation |
| matplotlib_generator | 16 | Curve queries, code generation, NumPy expressions |
| layout_generator | 8 | Default layouts, themes, CSS/JSX generation |
| layout_integration | 8 | Graph/curve with layout patterns |
| subplot_layout | 10 | Subplot CSS, JSX, matplotlib generation |
| control_system | 14 | Control definitions, JSX generation, state, CSS |
| wiring_system | 10 | Wiring specs, props, types, wired components |
| heatmap_generator | 9 | Specs, cells, dimensions, React/matplotlib |
| treemap_generator | 9 | Specs, nodes, hierarchy, React/Plotly |
| plot3d_generator | 11 | Surfaces, scatter, lines, React/matplotlib |

## Best Practices

### 1. Use Meaningful Names

```prolog
% Good: descriptive names
graph_node(user_alice, [label("Alice"), role(admin)]).
curve(revenue_growth, [type(exponential), label("Revenue")]).

% Avoid: generic names
graph_node(n1, [label("Node 1")]).
curve(c1, [type(linear)]).
```

### 2. Configure Themes Consistently

```prolog
% Define a shared theme
visualization_theme(corporate, [
    primary_color('#1e40af'),
    secondary_color('#3b82f6'),
    background('#f8fafc'),
    font('Inter, sans-serif')
]).

% Apply to visualizations
graph_spec(org_chart, [theme(corporate), ...]).
plot_spec(sales_chart, [theme(corporate), ...]).
```

### 3. Separate Data from Presentation

```prolog
% Data definition (reusable)
graph_node(dept_engineering, [label("Engineering"), headcount(42)]).
graph_node(dept_sales, [label("Sales"), headcount(28)]).

% Presentation specification (view-specific)
graph_spec(org_overview, [nodes([dept_engineering, dept_sales]), layout(breadthfirst)]).
graph_spec(dept_detail, [nodes([dept_engineering]), layout(cose), show_members(true)]).
```

## Summary

The visualization glue modules provide:

- **Declarative definitions** - Define graphs and plots in Prolog
- **Multi-target generation** - React/TypeScript for web, Python for data science
- **Layout system** - Declarative CSS Grid/Flexbox layouts with subplot support
- **Target-aware subplots** - Native matplotlib subplots or synthesized CSS grids
- **Control system** - Declarative UI controls (sliders, selects, checkboxes, etc.)
- **Wired components** - Controls automatically connected to visualization state
- **Runtime evaluation** - Evaluate curves programmatically
- **Consistent patterns** - Same workflow as other UnifyWeaver glue modules
- **Full test coverage** - 128 integration tests

## What's Next?

- Explore the generated components in your React application
- Use matplotlib scripts for data analysis workflows
- Combine with RPyC bridges for remote visualization
- Extend with custom curve types or graph layouts
