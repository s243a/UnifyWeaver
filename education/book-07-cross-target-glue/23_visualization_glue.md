<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 23: Declarative Visualization Glue

**Generating React Components and Python Plots from Prolog Specifications**

This chapter covers three declarative glue modules that generate visualization code:

- **graph_generator.pl** - Cytoscape.js graph visualization with React/TypeScript
- **curve_plot_generator.pl** - Chart.js curve plotting with React/TypeScript
- **matplotlib_generator.pl** - Python matplotlib code generation

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

```prolog
:- use_module('src/unifyweaver/glue/curve_plot_generator').

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

## Testing

The integration tests verify all visualization glue modules:

```bash
# Run visualization glue tests
swipl -g "run_tests" -t halt tests/integration/glue/test_visualization_glue.pl

# Expected output:
# Results: 99/99 tests passed
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
- **Full test coverage** - 99 integration tests

## What's Next?

- Explore the generated components in your React application
- Use matplotlib scripts for data analysis workflows
- Combine with RPyC bridges for remote visualization
- Extend with custom curve types or graph layouts
