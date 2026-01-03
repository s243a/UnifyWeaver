# Implementation Plan: Declarative Mind Map Layout System

## Overview

This document outlines the phased implementation of the declarative mind map layout system. The plan prioritizes layout optimization as the first target, building foundation infrastructure, and progressively adding targets and features.

## Phase 1: Core DSL Foundation

### 1.1 Create Base Module Structure

**File:** `src/unifyweaver/mindmap/mindmap_dsl.pl`

```prolog
:- module(mindmap_dsl, [...]).

% Dynamic predicates for node/edge storage
:- dynamic mindmap_node/2.
:- dynamic mindmap_edge/3.
:- dynamic mindmap_spec/2.
:- dynamic mindmap_constraint/2.
:- dynamic mindmap_preference/2.
:- dynamic mindmap_layout/2.
:- dynamic mindmap_pipeline/2.
:- dynamic mindmap_style/2.
:- dynamic mindmap_theme/2.
```

**Tasks:**
- [x] Define core predicates for node and edge declaration
- [x] Implement specification predicates with validation
- [x] Add constraint and preference storage
- [x] Create management predicates (declare_*, clear_*)

### 1.2 Integrate with Component Registry

**File:** `src/unifyweaver/mindmap/mindmap_components.pl`

**Tasks:**
- [x] Define `mindmap_layout` category
- [x] Define `mindmap_optimizer` category
- [x] Define `mindmap_renderer` category
- [x] Register default implementations

### 1.3 Create Intermediate Representation

**File:** `src/unifyweaver/mindmap/mindmap_ir.pl`

**Tasks:**
- [x] Define normalized graph structure
- [x] Implement spec-to-IR transformation
- [x] Add attribute extraction and normalization
- [x] Handle implicit edges from `parent()` properties

## Phase 2: Layout Optimization Target (Priority)

### 2.1 Port Existing Python Algorithms

**Reference:** `scripts/generate_mindmap.py` (force-directed, radial)

**File:** `src/unifyweaver/mindmap/layout/force_directed.pl`

```prolog
%% compute_force_layout(+Nodes, +Edges, +Options, -Positions)
% Pure Prolog implementation with binding to Python for computation
```

**Tasks:**
- [x] Extract algorithm parameters from existing Python
- [x] Create Prolog interface predicates
- [ ] Add Python bindings for heavy computation
- [x] Implement fallback pure-Prolog version for small graphs

### 2.2 Port Optimization Algorithms

**Reference:** `scripts/generate_mindmap.py` (overlap removal, crossing minimization)

**Files:**
- `src/unifyweaver/mindmap/optimize/overlap_removal.pl`
- `src/unifyweaver/mindmap/optimize/crossing_minimization.pl`

**Tasks:**
- [x] Create overlap detection predicates
- [x] Implement push-apart algorithm
- [ ] Create edge crossing detection
- [ ] Implement angular adjustment optimizer
- [ ] Add Python bindings for O(n²) operations

### 2.3 Create Layout Pipeline

**File:** `src/unifyweaver/mindmap/layout_pipeline.pl`

```prolog
%% execute_pipeline(+Graph, +Pipeline, -Result)
% Execute a sequence of layout and optimization stages
```

**Tasks:**
- [x] Implement stage sequencing
- [x] Add intermediate result passing
- [x] Create convergence detection
- [x] Add progress reporting hooks

## Phase 3: First Output Targets

### 3.1 SVG Renderer

**File:** `src/unifyweaver/mindmap/render/svg_renderer.pl`

**Tasks:**
- [x] Port rendering logic from `render_mindmap.py`
- [x] Create SVG element generation predicates
- [x] Implement node shape rendering (ellipse, rectangle, diamond)
- [x] Implement edge rendering (straight, bezier)
- [x] Add style application
- [x] Generate complete SVG document

### 3.2 Native Format Export (.smmx, .mm)

**File:** `src/unifyweaver/mindmap/render/native_format_renderer.pl`

**Tasks:**
- [ ] Port XML generation from `export_mindmap.py`
- [ ] Create format-specific structure generation
- [ ] Map positions to native format coordinates
- [ ] Preserve styling information

### 3.3 Interactive Graph Component

**File:** `src/unifyweaver/mindmap/render/graph_interactive_renderer.pl`

**Tasks:**
- [ ] Extend existing `graph_generator.pl` patterns
- [ ] Add mind map-specific node types
- [ ] Create layout preset export
- [ ] Generate React/TypeScript component

## Phase 4: Custom Component Integration

### 4.1 Leverage Existing Custom Components

UnifyWeaver already has `custom_<target>.pl` modules for 25+ targets. The mind map
system should leverage these existing components.

**Existing modules to use:**
- `targets/go_runtime/custom_go.pl` - Custom Go code injection
- `targets/python_runtime/custom_python.pl` - Custom Python code
- `targets/typescript_runtime/custom_typescript.pl` - Custom TypeScript
- ... and 20+ more

**Tasks:**
- [ ] Document how to use existing custom components for layouts
- [ ] Create example custom layout using `custom_python`
- [ ] Create example custom optimizer using `custom_go`
- [ ] Add `component(Name)` reference syntax to mind map specs

### 4.2 Mind Map-Specific Custom Components

**Files:**
- `src/unifyweaver/mindmap/custom_mindmap_layout.pl`
- `src/unifyweaver/mindmap/custom_mindmap_optimizer.pl`

```prolog
% Follow existing pattern from custom_go.pl / custom_python.pl
:- module(custom_mindmap_layout, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).
```

**Tasks:**
- [ ] Create mind map layout component type (delegates to target custom components)
- [ ] Create mind map optimizer component type
- [ ] Register with component registry on initialization
- [ ] Add validation for mind map-specific options

### 4.3 Binding Integration

**File:** `src/unifyweaver/bindings/python_mindmap_bindings.pl`

```prolog
declare_binding(python, force_layout/3, 'mindmap.layout.force_directed',
    [graph_dict, options_dict], [positions_dict],
    [import('unifyweaver.mindmap.layout')]).
```

**Tasks:**
- [ ] Define bindings for built-in layout algorithms
- [ ] Define bindings for optimization passes
- [ ] Create Python implementation module
- [ ] Add NumPy/SciPy acceleration

## Phase 5: Styling System

### 5.1 Style Resolution

**File:** `src/unifyweaver/mindmap/styling/style_resolver.pl`

**Tasks:**
- [ ] Implement selector matching
- [ ] Create property cascading (theme → type → node)
- [ ] Add computed style calculation
- [ ] Support CSS custom properties

### 5.2 Theme System

**File:** `src/unifyweaver/mindmap/styling/theme_system.pl`

**Tasks:**
- [ ] Define built-in themes
- [ ] Create theme application predicates
- [ ] Add theme inheritance/extension
- [ ] Support user-defined themes

## Phase 6: Additional Targets

### 6.1 GraphViz Export

**File:** `src/unifyweaver/mindmap/render/graphviz_renderer.pl`

**Tasks:**
- [ ] Generate DOT format output
- [ ] Map styles to GraphViz attributes
- [ ] Support multiple GraphViz layouts
- [ ] Add subgraph/cluster support

### 6.2 D3.js Component

**File:** `src/unifyweaver/mindmap/render/d3_renderer.pl`

**Tasks:**
- [ ] Generate D3.js force simulation code
- [ ] Add interactive features (drag, zoom, pan)
- [ ] Create animation support
- [ ] Generate React wrapper component

### 6.3 Additional Native Formats (.mm, .vue)

**File:** `src/unifyweaver/mindmap/render/mm_renderer.pl`

**Tasks:**
- [ ] Generate .mm XML format
- [ ] Map styling to native properties
- [ ] Support icons and links

## Phase 7: Interactive Features (Future GUI)

This phase leverages UnifyWeaver's existing GUI infrastructure:

**Existing modules to build on:**
- `react_generator.pl` - React component generation
- `interaction_generator.pl` - Tooltips, pan, zoom, drag handlers
- `data_binding_generator.pl` - Real-time data sync
- `animation_generator.pl` - Animation system
- `graph_generator.pl` - Interactive graph visualization

**Existing examples:**
- `examples/storybook-react/` - React component examples
- `Interactions.tsx` - Tooltip, PanZoomCanvas patterns
- `Performance.tsx` - VirtualList for large datasets

### 7.1 Event Model (extend interaction_generator.pl)

**File:** `src/unifyweaver/mindmap/interaction/mindmap_interaction.pl`

```prolog
% Using existing interaction_generator.pl patterns
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

drag_spec(mindmap_nodes, [
    enabled(true),
    mode(node_move),
    update_layout(true)
]).
```

**Tasks:**
- [ ] Extend `interaction_generator.pl` with mind map-specific modes
- [ ] Add `generate_mindmap_interaction/2` predicate
- [ ] Integrate with existing event handler patterns
- [ ] Add gesture support using existing patterns

### 7.2 Viewport Management (extend zoom/pan patterns)

**File:** `src/unifyweaver/mindmap/interaction/mindmap_viewport.pl`

```prolog
% Using existing zoom_spec/2 and drag_spec/2 patterns
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
    inertia(true),
    bounds(auto)
]).
```

**Tasks:**
- [ ] Use existing `generate_zoom_controls/2`
- [ ] Use existing `generate_pan_handler/2`
- [ ] Add fit-to-content calculation for mind maps
- [ ] Integrate with layout system for bounds

### 7.3 Hyperlink Navigation (extend selection patterns)

**File:** `src/unifyweaver/mindmap/interaction/mindmap_navigation.pl`

**Tasks:**
- [ ] Extend `selection_spec/2` with `on_select(follow_link)`
- [ ] Use existing tooltip system for link previews
- [ ] Add smooth scroll animation using `animation_generator.pl`
- [ ] Support both external URLs and internal node links

## File Structure

```
src/unifyweaver/mindmap/
├── mindmap_dsl.pl              # Core DSL module
├── mindmap_components.pl       # Component registry integration
├── mindmap_ir.pl               # Intermediate representation
│
├── layout/
│   ├── layout_interface.pl     # Layout algorithm interface
│   ├── radial.pl               # Radial layout
│   ├── force_directed.pl       # Force-directed layout
│   ├── hierarchical.pl         # Tree layout
│   └── layout_pipeline.pl      # Pipeline execution
│
├── optimize/
│   ├── optimizer_interface.pl  # Optimizer interface
│   ├── overlap_removal.pl      # Overlap removal
│   ├── crossing_minimization.pl# Edge crossing minimization
│   └── spacing.pl              # Spacing adjustment
│
├── render/
│   ├── renderer_interface.pl   # Renderer interface
│   ├── svg_renderer.pl         # SVG output
│   ├── smmx_renderer.pl        # .smmx format
│   ├── graph_interactive_renderer.pl  # Interactive React
│   ├── graphviz_renderer.pl    # DOT format
│   ├── d3_renderer.pl          # D3.js
│   └── mm_renderer.pl          # .mm format
│
├── styling/
│   ├── style_resolver.pl       # Style resolution
│   └── theme_system.pl         # Theme management
│
├── interaction/
│   ├── event_model.pl          # Event handling
│   ├── viewport.pl             # Pan/zoom
│   └── navigation.pl           # Link navigation
│
└── bindings/
    ├── python_layout.py        # Python layout implementations
    └── typescript/             # TypeScript implementations
        ├── package.json
        └── src/
            └── layout.ts
```

## Milestones

### Milestone 1: Minimal Viable DSL
- Core DSL predicates working
- Can define nodes and edges
- Basic spec validation

### Milestone 2: Layout Optimization Working
- Force-directed layout functional
- Overlap removal working
- Can process existing mind map data

### Milestone 3: First Output Target
- SVG rendering complete
- Styles applied correctly
- Matches quality of existing `render_mindmap.py`

### Milestone 4: Multi-Target Support
- 3+ output targets working
- Binding integration complete
- Pipeline execution reliable

### Milestone 5: Interactive Foundation
- Event model defined
- Basic pan/zoom working
- Link navigation functional

## Testing Strategy

### Unit Tests

```prolog
% test/mindmap/mindmap_dsl_test.pl
:- begin_tests(mindmap_dsl).

test(node_declaration) :-
    clear_mindmap,
    declare_mindmap_node(test, [label("Test")]),
    mindmap_node(test, Props),
    member(label("Test"), Props).

test(implicit_edge_creation) :-
    clear_mindmap,
    declare_mindmap_node(parent, [label("Parent")]),
    declare_mindmap_node(child, [label("Child"), parent(parent)]),
    mindmap_edge(parent, child, _).

:- end_tests(mindmap_dsl).
```

### Integration Tests

- Process existing mind map files (.smmx, .mm) through pipeline
- Compare output with existing Python tools
- Validate against multiple targets

### Visual Regression Tests

- Generate reference images
- Compare new outputs pixel-by-pixel
- Flag layout differences for review

## Dependencies

### Existing Modules to Integrate

- `component_registry.pl` - Component management
- `binding_registry.pl` - Target bindings
- `target_registry.pl` - Target capabilities
- `layout_generator.pl` - Layout patterns (for web targets)
- `graph_generator.pl` - Graph visualization patterns

### External Dependencies

**Python:**
- NumPy (array operations)
- SciPy (optimization)
- svgwrite (SVG generation)

**JavaScript/TypeScript:**
- Graph visualization libraries (for interactive output)
- D3.js (data visualization)
- React (component framework)

## Risk Mitigation

### Performance Risk

**Risk:** Layout optimization too slow for large mind maps

**Mitigation:**
- Use Python/NumPy for heavy computation via bindings
- Implement spatial indexing (R-trees) for collision detection
- Add progressive refinement with early termination

### Compatibility Risk

**Risk:** Output doesn't match existing tool quality

**Mitigation:**
- Port existing algorithms faithfully before optimizing
- Create comparison test suite
- Allow fallback to existing Python tools

### Complexity Risk

**Risk:** DSL becomes too complex to use

**Mitigation:**
- Progressive enhancement approach (simple → complex)
- Sensible defaults for all options
- Comprehensive examples and documentation
