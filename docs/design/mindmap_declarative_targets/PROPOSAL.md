# Proposal: Declarative Mind Map Layout System

## Executive Summary

This proposal introduces a declarative system for generating mind map layouts across multiple targets. The system preserves existing Python tools (`export_mindmap.py`, `render_mindmap.py`, `generate_mindmap.py`) while adding a composable Prolog DSL that compiles to various output formats, starting with layout optimization.

## Motivation

### Current State

UnifyWeaver has powerful mind mapping tools:
- **Export**: Convert between mind map formats (.smmx, .mm, .vue, .opml, .graphml)
- **Render**: Generate SVG, PNG with customizable styling
- **Generate**: Procedural generation with force-directed layout and optimization

However, these tools are:
- **Imperative**: Python scripts that execute steps sequentially
- **Monolithic**: Each script handles the full pipeline
- **Single-target**: Each output requires separate code paths

### Proposed State

A declarative system where:
- **Specification is separate from execution**: Define *what* you want, not *how* to get it
- **Components are composable**: Mix and match layouts, optimizers, and renderers
- **Targets are pluggable**: Same spec compiles to SVG, interactive graphs, native formats, etc.
- **Custom functions extend the system**: Users add algorithms without modifying core code

## Key Features

### 1. Declarative Mind Map DSL

Following the patterns from `layout_generator.pl`:

```prolog
% Define structure (similar to graph_node/2 pattern)
mindmap_node(root, [label("Project Plan"), type(root)]).
mindmap_node(phase1, [label("Research"), parent(root)]).
mindmap_node(phase2, [label("Development"), parent(root)]).

% Define layout (following declare_layout/3 pattern)
declare_mindmap_layout(project_map, force_directed, [
    iterations(300),
    min_distance(50),
    theme(dark)
]).

% Generate to any target
?- generate_mindmap_svg(project_map, SVGCode).
?- generate_mindmap_jsx(project_map, ReactComponent).
```

### 2. Layout Optimization Pipeline

Build multi-stage optimization as composable pipeline:

```prolog
mindmap_pipeline(production, [
    stage(radial, [level_spacing(120)]),      % Initial layout
    stage(force_directed, [iterations(200)]), % Physics refinement
    optimize(overlap_removal, []),             % Remove overlaps
    optimize(crossing_minimization, [])        % Clean up crossings
]).
```

### 3. Custom Function System

Leverages UnifyWeaver's existing **custom component system** (`custom_<target>.pl`
modules for 25+ targets):

```prolog
% Use existing custom_python for a layout algorithm
declare_component(source, embedding_layout, custom_python, [
    code("
        from sklearn.manifold import TSNE
        positions = TSNE(n_components=2).fit_transform(embeddings)
        return {n: (x,y) for n,(x,y) in zip(nodes, positions)}
    "),
    imports(["sklearn.manifold"])
]).

% Use existing custom_go for fast overlap removal
declare_component(source, fast_overlap, custom_go, [
    code("
        tree := rtree.New()
        for _, n := range nodes { tree.Insert(n.Bounds(), n) }
        return resolveOverlaps(tree, nodes)
    "),
    imports(["github.com/dhconnelly/rtreego"])
]).

% Reference custom components in mind map spec
declare_mindmap_layout(custom_map, component(embedding_layout), [
    pipeline([
        stage(component(embedding_layout), []),
        optimize(component(fast_overlap), [])
    ])
]).
```

**Existing custom component modules:**
- `targets/go_runtime/custom_go.pl`
- `targets/python_runtime/custom_python.pl`
- `targets/typescript_runtime/custom_typescript.pl`
- ... and 20+ more targets

### 4. Component Registry Integration

All algorithms register as queryable components:

```prolog
% Built-in layouts
:- register_component_type(mindmap_layout, radial, radial_layout_module, [...]).
:- register_component_type(mindmap_layout, force_directed, force_module, [...]).

% User custom layouts also registered
?- list_components(mindmap_layout, All).
% All = [radial, force_directed, hierarchical, spiral, embedding_layout, ...]
```

### 5. Binding-Based Target Support

New targets added via bindings, not code changes:

```prolog
% Python target for NumPy-accelerated layout
declare_binding(python, force_layout/3, 'mindmap.layout.force_directed',
    [graph, options], [positions], [import('unifyweaver.mindmap')]).

% JavaScript target for browser execution
declare_binding(javascript, force_layout/3, 'computeForceLayout',
    [graph, options], [positions], [import('@unifyweaver/layout')]).
```

### 6. Constraint-Based Layout

Frame layout as constraint satisfaction:

```prolog
% Hard constraints (must satisfy)
mindmap_constraint(no_overlap, [min_distance(20)]).
mindmap_constraint(hierarchy, [child_further_from_root]).

% Soft preferences (optimize for)
mindmap_preference(minimize_crossings, [weight(0.8)]).
mindmap_preference(angular_balance, [weight(0.5)]).
```

## Relation to Existing Tools

| Existing Tool | Preserved? | Integration |
|---------------|------------|-------------|
| `export_mindmap.py` | Yes | Can use DSL renderers internally |
| `render_mindmap.py` | Yes | SVG renderer ports its logic |
| `generate_mindmap.py` | Yes | Algorithms become DSL components |

The existing tools remain as standalone scripts. The DSL provides an alternative declarative interface that can eventually wrap or replace them.

## Output Targets

### Phase 1 (Layout Optimization Focus)
- **Internal positions**: Compute optimized (x, y) coordinates
- **SVG**: Static vector output
- **Native formats**: .smmx, .mm with computed positions

### Phase 2 (Interactive)
- **Graph visualization libraries**: React components with interaction
- **Force simulation**: Animated layout with D3-style physics

### Phase 3 (Export Formats)
- **GraphViz (DOT)**: For external rendering
- **Various mind map formats**: .mm, .vue, .opml, .graphml
- **Text-based diagrams**: Mermaid, PlantUML

### Future (GUI Integration)
- **HTML Canvas**: With pan/zoom/hyperlinks
- **WebGL**: For large graphs
- **Native GUI**: Platform-specific rendering

## API Alignment with layout_generator.pl

The mind map DSL follows established patterns from the existing layout system:

| Existing API | Mind Map Equivalent |
|--------------|---------------------|
| `declare_layout/3` | `declare_mindmap_layout/3` |
| `layout/3` | `mindmap_layout/3` |
| `generate_layout_css/2` | `generate_mindmap_svg/2` |
| `generate_layout_jsx/2` | `generate_mindmap_jsx/2` |
| `style/2` | `mindmap_style/2` |
| `theme/2` | `mindmap_theme/2` |
| `has_layout/1` | `has_mindmap_layout/1` |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Specification                        │
│  mindmap_node/2, declare_mindmap_layout/3, mindmap_style/2  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Intermediate Representation                  │
│  Normalized graph with attributes, constraints, preferences  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Component Registry + Custom Functions           │
│  Built-in + user-defined layouts, optimizers, renderers     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layout Pipeline                           │
│  Sequential execution of layout stages and optimizers       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Binding Layer (Target Resolution)               │
│  Maps operations to Python/JavaScript/Prolog implementations │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Code Generator                            │
│  SVG, interactive graphs, native formats, DOT, etc.         │
└─────────────────────────────────────────────────────────────┘
```

## Future: GUI Capabilities

The architecture builds on UnifyWeaver's existing GUI infrastructure:

**Existing generators to leverage:**
- `react_generator.pl` - React component generation
- `interaction_generator.pl` - Tooltips, pan, zoom, drag handlers
- `data_binding_generator.pl` - Real-time data sync
- `animation_generator.pl` + `animation_presets.pl` - Animation library
- `graph_generator.pl` - Interactive graph visualization

### Canvas Drawing (using existing patterns)
```prolog
% Following react_generator.pl patterns
ui_component(mindmap_canvas, [
    type(visualization),
    title("Mind Map"),
    width(1920), height(1080),
    theme(dark),
    render_quality(high)
]).

?- generate_react_component(mindmap_canvas, ReactCode).
```

### Pan and Zoom (using interaction_generator.pl)
```prolog
% Following existing zoom_spec/2 and pan patterns
zoom_spec(mindmap_view, [
    enabled(true),
    min_scale(0.1),
    max_scale(5.0),
    step(0.1),
    controls([zoom_in, zoom_out, zoom_fit, reset])
]).

drag_spec(mindmap_view, [
    enabled(true),
    mode(pan),
    inertia(true)
]).

?- generate_zoom_controls(mindmap_view, ZoomJSX).
?- generate_pan_handler(mindmap_view, PanHandler).
```

### Hyperlink Navigation (using interaction patterns)
```prolog
mindmap_node(docs, [
    label("Documentation"),
    link("https://docs.example.com"),
    link_style(underline)
]).

% Following existing tooltip_spec/2 pattern
tooltip_spec(mindmap_nodes, [
    position(node_center),
    offset(0, -10),
    delay(200),
    content([field(label), field(link)])
]).

% Following existing selection patterns
selection_spec(mindmap_nodes, [
    mode(single),
    method(click),
    on_select(follow_link)
]).
```

## Benefits

### For Users
- **Simpler specification**: Describe the result, not the process
- **Consistent interface**: Same DSL for all targets
- **Customizable**: Add custom algorithms without forking

### For Developers
- **Modular codebase**: Each component is isolated
- **Testable**: Pure functions, clear interfaces
- **Extensible**: New targets without touching existing code

### For the Project
- **Unified approach**: Aligns with existing declarative systems
- **Reuses infrastructure**: Component registry, bindings, targets
- **Future-proof**: GUI features build on same foundation

## Implementation Phases

1. **Core DSL**: Node/edge/spec predicates, basic validation
2. **Layout Optimization**: Port force-directed, overlap removal, crossing minimization
3. **First Targets**: SVG renderer, native format export
4. **Custom Functions**: Extension mechanism for all component types
5. **Additional Targets**: Interactive graphs, DOT, text diagrams
6. **Interactivity**: Event model, viewport, navigation

## Related Documents

- [PHILOSOPHY.md](./PHILOSOPHY.md) - Design principles and architectural decisions
- [SPECIFICATION.md](./SPECIFICATION.md) - Complete DSL reference
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) - Detailed implementation roadmap

## Success Criteria

1. **Functional parity**: DSL can produce output matching existing Python tools
2. **Multi-target**: Same spec generates valid SVG, native formats, and interactive output
3. **Custom functions work**: Users can add layouts/optimizers without core changes
4. **Performance acceptable**: Large mind maps (500+ nodes) optimize in reasonable time
5. **Integration clean**: Uses existing component/binding/target registries

## Conclusion

This proposal extends UnifyWeaver's declarative approach to mind map generation. By separating specification from execution and making all components pluggable, we create a flexible system that:

- Preserves existing tool functionality
- Enables multi-target output from single specifications
- Allows user customization through the custom function system
- Prepares for future GUI integration with canvas, pan/zoom, and hyperlinks

The phased implementation minimizes risk while delivering incremental value, starting with the most requested feature: layout optimization.
