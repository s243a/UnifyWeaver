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

## Testing

The integration tests verify all visualization glue modules:

```bash
# Run visualization glue tests
swipl -g "run_tests" -t halt tests/integration/glue/test_visualization_glue.pl

# Expected output:
# Results: 49/49 tests passed
# All tests passed!
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| graph_generator | 16 | Node/edge queries, component generation, CSS, config |
| curve_plot_generator | 17 | Curve queries, evaluation, component generation |
| matplotlib_generator | 16 | Curve queries, code generation, NumPy expressions |

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
- **Runtime evaluation** - Evaluate curves programmatically
- **Consistent patterns** - Same workflow as other UnifyWeaver glue modules
- **Full test coverage** - 49 integration tests

## What's Next?

- Explore the generated components in your React application
- Use matplotlib scripts for data analysis workflows
- Combine with RPyC bridges for remote visualization
- Extend with custom curve types or graph layouts
