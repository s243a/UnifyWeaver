# Design Philosophy: Declarative Mind Map Layout System

## Core Principle: Separation of Concerns

The declarative mind map system separates **what** from **how**:

1. **Specification** - What the mind map contains and how it should behave
2. **Layout Algorithm** - How nodes are positioned (force-directed, radial, etc.)
3. **Rendering Target** - How the result is output (SVG, Canvas, interactive graphs, etc.)
4. **Optimization** - How the layout is refined (crossing minimization, overlap removal)

Each concern is independently composable and replaceable.

## Design Principles

### 1. Declarative Over Imperative

Describe the desired outcome, not the steps to achieve it:

```prolog
% Good: Declarative specification
mindmap_node(root, [label("Central Topic"), style(hub)]).
mindmap_node(child1, [label("Branch A"), parent(root)]).
mindmap_constraint(no_overlap, [min_distance(20)]).
mindmap_layout(force_directed, [iterations(300)]).

% Bad: Imperative positioning
set_node_position(root, 400, 300).
set_node_position(child1, 600, 200).
```

### 2. Composable Components

Systems should be built from small, reusable pieces:

```prolog
% Layout algorithms are components
:- register_component_type(layout, force_directed, force_directed_layout, []).
:- register_component_type(layout, radial, radial_layout, []).
:- register_component_type(layout, hierarchical, hierarchical_layout, []).

% Optimizers are components
:- register_component_type(optimizer, overlap_removal, overlap_optimizer, []).
:- register_component_type(optimizer, crossing_minimization, crossing_optimizer, []).

% Renderers are components
:- register_component_type(renderer, svg, svg_renderer, []).
:- register_component_type(renderer, canvas, canvas_renderer, []).
:- register_component_type(renderer, graph_interactive, graph_interactive_renderer, []).
```

### 3. Target Independence

The same specification should compile to multiple targets:

```prolog
% Single specification
mindmap_spec(my_map, [
    nodes([root, a, b, c]),
    layout(force_directed),
    theme(dark)
]).

% Multiple targets
?- compile_mindmap(my_map, svg, SVGCode).
?- compile_mindmap(my_map, graph_interactive, InteractiveCode).
?- compile_mindmap(my_map, smmx, SMMXCode).
?- compile_mindmap(my_map, graphviz, DotCode).
```

### 4. Binding-Based Extension

New targets are added through bindings, not code changes:

```prolog
% Bind layout operations to target implementations
declare_binding(python, compute_forces/3, 'layout.compute_forces',
    [nodes, edges, params], [positions],
    [import('mindmap.layout')]).

declare_binding(javascript, compute_forces/3, 'computeForces',
    [nodes, edges, params], [positions],
    [import('layout-engine')]).
```

### 5. Constraint Satisfaction

Layout is framed as constraint satisfaction, not procedural positioning:

```prolog
% Hard constraints (must satisfy)
mindmap_constraint(hierarchy, [child_further_from_center]).
mindmap_constraint(no_overlap, [min_distance(NodeRadius * 2)]).

% Soft constraints (optimize for)
mindmap_preference(minimize_crossings, [weight(0.8)]).
mindmap_preference(angular_balance, [weight(0.5)]).
mindmap_preference(radial_alignment, [weight(0.3)]).
```

### 6. Progressive Enhancement

Start simple, add complexity as needed:

```prolog
% Level 1: Basic specification (uses all defaults)
mindmap(my_map, [nodes([a, b, c]), edges([a-b, b-c])]).

% Level 2: With layout preference
mindmap(my_map, [nodes([...]), layout(radial)]).

% Level 3: With customization
mindmap(my_map, [nodes([...]), layout(radial),
    options([iterations(500), min_distance(50)])]).

% Level 4: With optimization pipeline
mindmap(my_map, [nodes([...]),
    pipeline([radial_init, force_refine, crossing_minimize])]).
```

### 7. Semantic Preservation

Meaning travels through the pipeline:

```prolog
% Semantic annotations
mindmap_node(topic, [
    label("Machine Learning"),
    semantic_type(concept),
    importance(high),
    cluster(ai_topics)
]).

% Semantics inform layout
layout_hint(cluster(ai_topics), [group_nearby]).
layout_hint(importance(high), [larger_node, central_position]).
```

## Architectural Layers

```
+------------------------------------------------------------------+
|                    User Specification (DSL)                       |
|  mindmap_node/2, mindmap_edge/3, mindmap_spec/2, constraints     |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    Intermediate Representation                    |
|  Normalized graph structure with attributes and constraints       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    Layout Engine (Pluggable)                      |
|  Force-directed, Radial, Hierarchical, Custom algorithms          |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    Optimization Pipeline                          |
|  Overlap removal, Crossing minimization, Spacing adjustment       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    Target Binding Layer                           |
|  Maps operations to target-specific implementations               |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    Code Generator                                 |
|  SVG, Canvas, interactive graphs, native formats (.smmx, .mm), etc. |
+------------------------------------------------------------------+
```

## Integration with Existing Systems

### Component Registry Integration

Layout algorithms, optimizers, and renderers register as components:

```prolog
:- define_category(mindmap_layout, "Layout algorithms", []).
:- define_category(mindmap_optimizer, "Layout optimizers", []).
:- define_category(mindmap_renderer, "Output renderers", []).
```

### Binding Registry Integration

Target-specific operations use the binding system:

```prolog
% Abstract operation
compute_node_positions(Graph, Layout, Positions).

% Bound to targets
declare_binding(python, compute_node_positions/3, ...).
declare_binding(javascript, compute_node_positions/3, ...).
```

### Target Registry Integration

Mind map targets register with capabilities:

```prolog
register_target(svg, graphics, [static, vector, pan, zoom]).
register_target(canvas, graphics, [dynamic, raster, pan, zoom, interaction]).
register_target(graph_interactive, graph_viz, [dynamic, interaction, layout_algorithms]).
register_target(smmx, mindmap, [native, editable, collaboration]).
register_target(mm, mindmap, [native, editable, folding]).
```

## Future GUI Considerations

The architecture builds on UnifyWeaver's existing GUI infrastructure:

**Existing modules to leverage:**
- `react_generator.pl` (1,004 lines) - React component generation
- `interaction_generator.pl` (1,111 lines) - Tooltips, pan, zoom, drag
- `data_binding_generator.pl` (989 lines) - Real-time data sync
- `animation_generator.pl` + `animation_presets.pl` (28,879 lines) - Animations
- `graph_generator.pl` (803 lines) - Interactive graph visualization

### Canvas Drawing
Uses existing `react_generator.pl` patterns:
- `ui_component/2` for component definition
- `generate_react_component/2` for code generation

### Pan and Zoom
Uses existing `interaction_generator.pl` patterns:
- `zoom_spec/2` for zoom configuration
- `drag_spec/2` with `mode(pan)` for panning
- `generate_zoom_controls/2`, `generate_pan_handler/2`

### Hyperlink Following
Uses existing interaction patterns:
- `tooltip_spec/2` for hover previews
- `selection_spec/2` with `on_select(follow_link)`

```prolog
% Using existing interaction_generator.pl patterns
zoom_spec(mindmap_view, [
    enabled(true),
    min_scale(0.1),
    max_scale(5.0),
    controls([zoom_in, zoom_out, zoom_fit, reset])
]).

drag_spec(mindmap_view, [
    enabled(true),
    mode(pan),
    inertia(true)
]).

tooltip_spec(mindmap_nodes, [
    position(node_center),
    delay(200),
    content([field(label), field(link)])
]).

selection_spec(mindmap_nodes, [
    mode(single),
    method(click),
    on_select(follow_link)
]).
```

## Guiding Questions

When extending the system, ask:

1. **Is this declarative?** Can it be specified without describing execution?
2. **Is this composable?** Can it be combined with other components?
3. **Is this target-independent?** Does it make sense across multiple outputs?
4. **Does it use bindings?** Is the implementation pluggable?
5. **Does it preserve semantics?** Does meaning flow through the pipeline?
