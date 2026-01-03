# Mind Map DSL Specification

## Overview

This specification defines a declarative domain-specific language (DSL) for mind map layout and visualization. The DSL is implemented in Prolog and generates code for multiple targets.

The API follows patterns established in `layout_generator.pl` and `graph_generator.pl` for consistency with existing UnifyWeaver declarative systems.

## Module Structure

```prolog
:- module(mindmap_dsl, [
    % Node and edge definition
    mindmap_node/2,           % mindmap_node(+Id, +Properties)
    mindmap_edge/3,           % mindmap_edge(+From, +To, +Properties)

    % Specification
    mindmap_spec/2,           % mindmap_spec(+Name, +Config)

    % Constraints and preferences
    mindmap_constraint/2,     % mindmap_constraint(+Type, +Options)
    mindmap_preference/2,     % mindmap_preference(+Type, +Options)

    % Layout configuration
    mindmap_layout/2,         % mindmap_layout(+Algorithm, +Options)
    mindmap_pipeline/2,       % mindmap_pipeline(+Name, +Stages)

    % Styling
    mindmap_style/2,          % mindmap_style(+Selector, +Properties)
    mindmap_theme/2,          % mindmap_theme(+Name, +Properties)

    % Code generation
    compile_mindmap/3,        % compile_mindmap(+Name, +Target, -Code)
    compile_mindmap/4,        % compile_mindmap(+Name, +Target, +Options, -Code)

    % Management
    declare_mindmap_node/2,
    declare_mindmap_edge/3,
    clear_mindmap/0,
    clear_mindmap/1
]).
```

## Core Predicates

### Node Definition

```prolog
%% mindmap_node(+Id, +Properties)
%
% Define a mind map node.
%
% Properties:
%   label(Text)           - Display text
%   parent(ParentId)      - Parent node (creates implicit edge)
%   type(Type)            - Node type (root, branch, leaf, hub)
%   style(StyleName)      - Reference to style definition
%   color(Color)          - Node color (hex or name)
%   shape(Shape)          - Node shape (ellipse, rectangle, diamond, half_round)
%   size(Width, Height)   - Node dimensions
%   icon(IconRef)         - Icon reference
%   link(URL)             - External hyperlink
%   link_node(NodeId)     - Internal link to another node
%   data(Key, Value)      - Custom metadata
%   cluster(ClusterId)    - Cluster membership for grouping
%   importance(Level)     - Importance level (low, medium, high)
%   position(X, Y)        - Initial/fixed position (optional)

% Examples:
mindmap_node(root, [
    label("Central Topic"),
    type(root),
    style(hub_style),
    importance(high)
]).

mindmap_node(branch1, [
    label("First Branch"),
    parent(root),
    type(branch),
    color('#7c3aed')
]).

mindmap_node(leaf1, [
    label("Detail Item"),
    parent(branch1),
    type(leaf),
    link("https://example.com/detail")
]).
```

### Edge Definition

```prolog
%% mindmap_edge(+From, +To, +Properties)
%
% Define an explicit edge between nodes.
% Note: parent(X) in node properties creates implicit edges.
%
% Properties:
%   label(Text)           - Edge label
%   style(StyleName)      - Reference to edge style
%   weight(N)             - Edge weight for layout algorithms
%   type(Type)            - Edge type (parent, sibling, cross_reference)
%   curve(CurveType)      - bezier, straight, orthogonal
%   color(Color)          - Edge color
%   width(N)              - Line width
%   arrow(ArrowSpec)      - Arrow configuration (none, end, both)

% Examples:
mindmap_edge(branch1, branch2, [
    type(cross_reference),
    label("related to"),
    style(dashed)
]).

mindmap_edge(root, overview, [
    type(parent),
    weight(2),
    curve(bezier)
]).
```

### Mind Map Specification

```prolog
%% mindmap_spec(+Name, +Config)
%
% Define a complete mind map specification.
%
% Config options:
%   title(Text)           - Map title
%   description(Text)     - Map description
%   nodes(NodeList)       - Explicit node list (optional, defaults to all)
%   root(NodeId)          - Root node identifier
%   layout(Algorithm)     - Layout algorithm
%   theme(ThemeName)      - Theme to apply
%   constraints(List)     - Constraint specifications
%   preferences(List)     - Optimization preferences
%   pipeline(Stages)      - Processing pipeline

% Example:
mindmap_spec(my_project_map, [
    title("Project Overview"),
    description("Main concepts and relationships"),
    root(central),
    layout(force_directed),
    theme(dark),
    constraints([
        no_overlap(min_distance(30)),
        hierarchy(child_further)
    ]),
    preferences([
        minimize_crossings(weight(0.8)),
        angular_balance(weight(0.5))
    ])
]).
```

## Layout System

### Layout Algorithms

```prolog
%% mindmap_layout(+Algorithm, +Options)
%
% Configure layout algorithm.
%
% Algorithms:
%   radial          - Radial/circular layout from root
%   force_directed  - Physics-based force simulation
%   hierarchical    - Tree-like hierarchical layout
%   cose            - Compound Spring Embedder (force-based)
%   grid            - Grid-based layout
%   concentric      - Concentric circles by attribute
%   custom(Module)  - Custom algorithm implementation

% Algorithm-specific options:

% radial options:
%   level_spacing(N)      - Distance between levels
%   angular_width(Deg)    - Angular span in degrees
%   start_angle(Deg)      - Starting angle
%   clockwise(Bool)       - Direction of layout

% force_directed options:
%   iterations(N)         - Number of simulation iterations
%   repulsion_strength(N) - Node repulsion force
%   attraction_strength(N)- Edge attraction force
%   damping(N)            - Velocity damping factor
%   min_distance(N)       - Minimum node distance

% hierarchical options:
%   direction(Dir)        - top_down, bottom_up, left_right, right_left
%   level_sep(N)          - Separation between levels
%   node_sep(N)           - Separation between siblings

% Examples:
mindmap_layout(radial, [
    level_spacing(150),
    angular_width(360),
    start_angle(0)
]).

mindmap_layout(force_directed, [
    iterations(300),
    repulsion_strength(100000),
    min_distance(120)
]).
```

### Layout Pipeline

```prolog
%% mindmap_pipeline(+Name, +Stages)
%
% Define a multi-stage layout pipeline.
%
% Each stage is a term: stage(Algorithm, Options)
% Or an optimizer: optimize(Type, Options)

% Example:
mindmap_pipeline(production_pipeline, [
    % Stage 1: Initial radial layout
    stage(radial, [level_spacing(100)]),

    % Stage 2: Force refinement
    stage(force_directed, [
        iterations(200),
        preserve_hierarchy(true)
    ]),

    % Stage 3: Overlap removal
    optimize(overlap_removal, [
        min_distance(30),
        method(push_apart)
    ]),

    % Stage 4: Crossing minimization
    optimize(crossing_minimization, [
        passes(50),
        method(angular_adjustment)
    ]),

    % Stage 5: Final spacing
    optimize(spacing, [
        uniform(false),
        density_aware(true)
    ])
]).
```

## Constraints and Preferences

### Hard Constraints

```prolog
%% mindmap_constraint(+Type, +Options)
%
% Define hard constraints that must be satisfied.
%
% Types:
%   no_overlap            - Nodes must not overlap
%   hierarchy             - Parent-child positioning rules
%   boundary              - Stay within bounds
%   fixed_position        - Some nodes have fixed positions
%   cluster_proximity     - Cluster members stay together
%   level_alignment       - Nodes at same depth align

% Examples:
mindmap_constraint(no_overlap, [
    min_distance(20),
    include_labels(true)
]).

mindmap_constraint(hierarchy, [
    rule(child_further_from_root),
    min_parent_child_distance(50)
]).

mindmap_constraint(boundary, [
    width(2000),
    height(1500),
    margin(50)
]).
```

### Soft Preferences

```prolog
%% mindmap_preference(+Type, +Options)
%
% Define soft preferences for optimization.
%
% Types:
%   minimize_crossings    - Reduce edge crossings
%   angular_balance       - Even angular distribution
%   radial_alignment      - Maintain radial structure
%   compact               - Minimize total area
%   symmetry              - Prefer symmetric layouts
%   cluster_cohesion      - Keep clusters together

% Examples:
mindmap_preference(minimize_crossings, [
    weight(0.8),
    method(hierarchical_priority)
]).

mindmap_preference(angular_balance, [
    weight(0.5),
    per_level(true)
]).

mindmap_preference(compact, [
    weight(0.3),
    aspect_ratio(16/9)
]).
```

## Styling System

### Style Definitions

```prolog
%% mindmap_style(+Selector, +Properties)
%
% Define styles for nodes and edges.
%
% Selectors:
%   node                  - All nodes
%   node(Type)            - Nodes of specific type
%   node(id(Id))          - Specific node
%   edge                  - All edges
%   edge(Type)            - Edges of specific type
%   label                 - All labels
%   .class_name           - Class-based selection

% Node properties:
%   fill(Color)           - Background color
%   stroke(Color)         - Border color
%   stroke_width(N)       - Border width
%   shape(Shape)          - Node shape
%   width(N), height(N)   - Dimensions
%   font_size(N)          - Label font size
%   font_family(Name)     - Label font
%   font_weight(Weight)   - Label font weight
%   opacity(N)            - Node opacity
%   shadow(ShadowSpec)    - Drop shadow

% Edge properties:
%   stroke(Color)         - Line color
%   stroke_width(N)       - Line width
%   stroke_dasharray(Pat) - Dash pattern
%   curve(Type)           - Curve type
%   arrow_end(Spec)       - End arrow
%   arrow_start(Spec)     - Start arrow

% Examples:
mindmap_style(node, [
    fill('#2a2a4a'),
    stroke('#7c3aed'),
    stroke_width(2),
    font_size(14),
    font_family('Inter, sans-serif')
]).

mindmap_style(node(root), [
    fill('#7c3aed'),
    font_size(18),
    font_weight(bold),
    shadow(offset(2, 2), blur(4), color('rgba(0,0,0,0.3)'))
]).

mindmap_style(node(leaf), [
    shape(ellipse),
    opacity(0.9)
]).

mindmap_style(edge, [
    stroke('#00d4ff'),
    stroke_width(2),
    curve(bezier)
]).

mindmap_style(edge(cross_reference), [
    stroke('#888888'),
    stroke_dasharray('5,5'),
    arrow_end(triangle)
]).
```

### Themes

```prolog
%% mindmap_theme(+Name, +Properties)
%
% Define complete themes.
%
% Properties define CSS custom properties / theme variables.

mindmap_theme(dark, [
    background('#1a1a2e'),
    surface('#2a2a4a'),
    text('#e0e0e0'),
    text_secondary('#888888'),
    accent('#7c3aed'),
    accent_secondary('#00d4ff'),
    node_fill('#2a2a4a'),
    node_stroke('#7c3aed'),
    edge_color('#00d4ff'),
    shadow_color('rgba(0,0,0,0.3)')
]).

mindmap_theme(light, [
    background('#f8fafc'),
    surface('#ffffff'),
    text('#1e293b'),
    text_secondary('#64748b'),
    accent('#7c3aed'),
    accent_secondary('#00d4ff'),
    node_fill('#ffffff'),
    node_stroke('#7c3aed'),
    edge_color('#7c3aed'),
    shadow_color('rgba(0,0,0,0.1)')
]).

mindmap_theme(nature, [
    background('#1a2f1a'),
    surface('#2d4a2d'),
    text('#d4e6d4'),
    accent('#4ade80'),
    node_fill('#2d4a2d'),
    node_stroke('#4ade80'),
    edge_color('#86efac')
]).
```

## Target Code Generation

### Supported Targets

```prolog
% Target capabilities
mindmap_target(svg, [
    static(true),
    vector(true),
    interactive(false),
    capabilities([pan, zoom, export_png, export_pdf])
]).

mindmap_target(canvas, [
    static(false),
    vector(false),
    interactive(true),
    capabilities([pan, zoom, click, hover, animation])
]).

mindmap_target(graph_interactive, [
    static(false),
    vector(true),
    interactive(true),
    capabilities([pan, zoom, click, hover, drag, layout_algorithms])
]).

mindmap_target(smmx, [
    format(smmx),
    native(true),
    editable(true),
    capabilities([manual_edit, collaboration])
]).

mindmap_target(mm, [
    format(mm),
    native(true),
    editable(true),
    capabilities([manual_edit, folding])
]).

mindmap_target(graphviz, [
    format(dot),
    static(true),
    capabilities([multiple_layouts, export_formats])
]).

mindmap_target(d3, [
    static(false),
    interactive(true),
    capabilities([pan, zoom, force_simulation, transitions])
]).
```

### Code Generation

```prolog
%% compile_mindmap(+Name, +Target, -Code)
%% compile_mindmap(+Name, +Target, +Options, -Code)
%
% Generate target-specific code from mind map specification.
%
% Options:
%   embed_data(Bool)      - Embed node/edge data in output
%   minify(Bool)          - Minify output code
%   include_styles(Bool)  - Include style definitions
%   include_interactions(Bool) - Include interaction handlers
%   export_format(Format) - For targets with multiple outputs

% Example usage:
?- mindmap_spec(my_map, Config),
   compile_mindmap(my_map, svg, SVGCode).

?- compile_mindmap(my_map, graph_interactive, [
       embed_data(true),
       include_interactions(true)
   ], ReactComponent).

?- compile_mindmap(my_map, smmx, SMMXContent).
```

## Component Registration

### Layout Algorithm Components

```prolog
% Register layout algorithms as components
:- define_category(mindmap_layout, "Mind map layout algorithms", [
    required_interface([
        compute_positions/3   % compute_positions(+Graph, +Options, -Positions)
    ])
]).

:- register_component_type(mindmap_layout, radial, radial_layout_module, [
    description("Radial layout from central root"),
    options([level_spacing, angular_width, start_angle, clockwise])
]).

:- register_component_type(mindmap_layout, force_directed, force_layout_module, [
    description("Force-directed physics simulation"),
    options([iterations, repulsion_strength, attraction_strength, damping])
]).
```

### Optimizer Components

```prolog
% Register optimizers as components
:- define_category(mindmap_optimizer, "Layout optimization passes", [
    required_interface([
        optimize_layout/3     % optimize_layout(+Positions, +Options, -NewPositions)
    ])
]).

:- register_component_type(mindmap_optimizer, overlap_removal, overlap_optimizer_module, [
    description("Remove node overlaps"),
    options([min_distance, method, iterations])
]).

:- register_component_type(mindmap_optimizer, crossing_minimization, crossing_optimizer_module, [
    description("Minimize edge crossings"),
    options([passes, method, preserve_hierarchy])
]).
```

### Renderer Components

```prolog
% Register renderers as components
:- define_category(mindmap_renderer, "Output renderers", [
    required_interface([
        render/3              % render(+Graph, +Positions, -Output)
    ])
]).

:- register_component_type(mindmap_renderer, svg, svg_renderer_module, [
    description("Static SVG output"),
    capabilities([vector, export])
]).

:- register_component_type(mindmap_renderer, canvas, canvas_renderer_module, [
    description("HTML5 Canvas output"),
    capabilities([raster, animation, interaction])
]).

:- register_component_type(mindmap_renderer, graph_interactive, graph_interactive_renderer, [
    description("Interactive graph React component"),
    capabilities([vector, interaction, builtin_layouts])
]).
```

## Custom Functions

The DSL integrates with UnifyWeaver's existing **custom component system**, which
provides `custom_<target>.pl` modules for 25+ target languages. Each module follows
the component interface pattern:

**Existing Custom Component Modules:**
- `targets/go_runtime/custom_go.pl`
- `targets/python_runtime/custom_python.pl`
- `targets/bash_runtime/custom_bash.pl`
- `targets/typescript_runtime/custom_typescript.pl`
- ... and 20+ more targets

### Custom Component Interface

Each custom component module implements:

```prolog
:- module(custom_<target>, [
    type_info/1,           % Metadata about the component type
    validate_config/1,     % Validate configuration options
    init_component/2,      % Initialize component (if needed)
    invoke_component/4,    % Runtime invocation (optional)
    compile_component/4    % Compile to target code
]).
```

### Using Custom Components

Declare custom components with inline target code:

```prolog
% Custom Python layout algorithm
declare_component(source, embedding_layout, custom_python, [
    code("
        import numpy as np
        from sklearn.manifold import TSNE

        positions = TSNE(n_components=2).fit_transform(embeddings)
        return {node: (x, y) for node, (x, y) in zip(nodes, positions)}
    "),
    imports(["numpy", "sklearn.manifold"])
]).

% Custom Go optimizer
declare_component(source, fast_overlap_removal, custom_go, [
    code("
        // Use spatial indexing for O(n log n) overlap detection
        tree := rtree.New()
        for _, node := range nodes {
            tree.Insert(node.Bounds(), node)
        }
        return resolveOverlaps(tree, nodes)
    "),
    imports(["github.com/dhconnelly/rtreego"])
]).

% Custom TypeScript renderer
declare_component(source, canvas_renderer, custom_typescript, [
    code("
        const ctx = canvas.getContext('2d');
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fill();
        });
        return canvas.toDataURL();
    "),
    imports([])
]).
```

### Mind Map Custom Components

For mind map-specific custom functions, we extend the pattern:

```prolog
%% Custom layout using existing custom_python
declare_component(source, spiral_layout, custom_python, [
    code("
        import math
        positions = {}
        for i, node in enumerate(nodes):
            angle = i * 0.5
            radius = 50 + i * 10
            positions[node] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle)
            )
        return positions
    "),
    imports(["math"])
]).

% Use in mind map spec
mindmap_spec(my_map, [
    layout(component(spiral_layout)),  % Reference component by name
    ...
]).
```

### Custom Mind Map Component Types

New component types for mind map operations (following the existing pattern):

```prolog
% File: src/unifyweaver/mindmap/custom_mindmap_layout.pl
:- module(custom_mindmap_layout, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

type_info(info(
    name('Custom Mind Map Layout'),
    version('1.0.0'),
    description('Custom layout algorithm for mind maps')
)).

validate_config(Config) :-
    member(algorithm(Alg), Config), atom(Alg).

compile_component(Name, Config, Options, Code) :-
    member(algorithm(Alg), Config),
    member(target(Target), Options),
    % Delegate to target-specific custom component
    compile_for_target(Target, Name, Alg, Config, Code).

:- initialization((
    register_component_type(mindmap_layout, custom, custom_mindmap_layout, [
        description("Custom Mind Map Layout Algorithm")
    ])
), now).
```

### Integration Pattern

The mind map DSL leverages existing custom components:

1. **Reuse existing `custom_<target>` modules** for target-specific code
2. **Register mind map component types** following the same interface
3. **Reference components by name** in mind map specs

```prolog
% Register a custom layout that uses custom_python internally
declare_component(source, my_layout, custom_python, [
    code("..."),
    imports([...])
]).

% Use in mind map (references the component)
mindmap_spec(example, [
    layout(component(my_layout)),
    pipeline([
        stage(component(my_layout), []),
        optimize(overlap_removal, [])
    ])
]).
```

## Binding Integration

### Target Bindings

```prolog
% Bind layout computation to Python (for NumPy/SciPy)
declare_binding(python, compute_force_layout/4, 'mindmap_layout.force_directed',
    [nodes, edges, options], [positions],
    [import('unifyweaver.mindmap.layout'),
     pure, deterministic]).

% Bind to JavaScript (for browser execution)
declare_binding(javascript, compute_force_layout/4, 'computeForceLayout',
    [nodes, edges, options], [positions],
    [import('@unifyweaver/mindmap-layout'),
     async]).

% Bind SVG generation
declare_binding(python, render_svg/3, 'mindmap_render.to_svg',
    [graph, positions], [svg_string],
    [import('unifyweaver.mindmap.render')]).
```

## Example: Complete Mind Map

```prolog
% Nodes
mindmap_node(ai, [label("Artificial Intelligence"), type(root), importance(high)]).
mindmap_node(ml, [label("Machine Learning"), parent(ai), cluster(learning)]).
mindmap_node(dl, [label("Deep Learning"), parent(ml), cluster(learning)]).
mindmap_node(rl, [label("Reinforcement Learning"), parent(ml), cluster(learning)]).
mindmap_node(nlp, [label("Natural Language Processing"), parent(ai), cluster(language)]).
mindmap_node(cv, [label("Computer Vision"), parent(ai), cluster(perception)]).
mindmap_node(transformers, [label("Transformers"), parent(dl), link_node(nlp)]).
mindmap_node(cnn, [label("CNN"), parent(dl), link_node(cv)]).

% Cross-references
mindmap_edge(transformers, nlp, [type(cross_reference), label("revolutionized")]).
mindmap_edge(cnn, cv, [type(cross_reference), label("core of")]).

% Specification
mindmap_spec(ai_overview, [
    title("AI Overview"),
    root(ai),
    layout(force_directed),
    theme(dark),
    pipeline([
        stage(radial, [level_spacing(120)]),
        stage(force_directed, [iterations(200)]),
        optimize(overlap_removal, []),
        optimize(crossing_minimization, [passes(30)])
    ])
]).

% Generate outputs
?- compile_mindmap(ai_overview, svg, SVG).
?- compile_mindmap(ai_overview, graph_interactive, ReactCode).
```
