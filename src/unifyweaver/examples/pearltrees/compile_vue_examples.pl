%% compile_vue_examples.pl - Generate Vue components from Prolog specifications
%%
%% Demonstrates UnifyWeaver's code generation from Prolog to Vue 3 SFC format.
%% Shows how visual/GUI components are generated, including which operations
%% require glue for non-web targets.
%%
%% Run with: swipl -g "compile_all_vue_examples" compile_vue_examples.pl

:- module(compile_vue_examples, [
    compile_all_vue_examples/0,
    compile_hierarchy_component/1,
    compile_mindmap_component/2,
    compile_viewport_component/1,
    show_target_capabilities/0,
    test_vue_codegen/0
]).

:- use_module('../../mindmap/render/d3_renderer').
:- use_module('../../mindmap/interaction/mindmap_viewport').
:- use_module('../../targets/vue_target').
:- use_module(hierarchy).
:- use_module(semantic_hierarchy).

% ============================================================================
% EXAMPLE DATA
% ============================================================================

%% Sample hierarchy for demonstration
example_nodes([
    node(root, [label("My Collections"), type(root)]),
    node(science, [label("Science"), type(hub)]),
    node(arts, [label("Arts & Culture"), type(hub)]),
    node(physics, [label("Physics"), type(branch)]),
    node(chemistry, [label("Chemistry"), type(branch)]),
    node(music, [label("Music"), type(leaf), link("https://example.com/music")]),
    node(quantum, [label("Quantum Mechanics"), type(leaf)])
]).

example_edges([
    edge(root, science, []),
    edge(root, arts, []),
    edge(science, physics, []),
    edge(science, chemistry, []),
    edge(arts, music, []),
    edge(physics, quantum, [])
]).

% ============================================================================
% MAIN COMPILATION
% ============================================================================

%% compile_all_vue_examples
%
%  Generate all Vue example components and write them to the vue/ directory.
%
compile_all_vue_examples :-
    format('~n=== Compiling Vue Examples ===~n~n'),

    % 1. Hierarchy visualization component
    format('1. Generating hierarchy component...~n'),
    compile_hierarchy_component(HierarchyCode),
    write_vue_file('vue/GeneratedHierarchy.vue', HierarchyCode),
    format('   Written: vue/GeneratedHierarchy.vue~n'),

    % 2. Mindmap with custom data
    format('~n2. Generating mindmap component...~n'),
    example_nodes(Nodes),
    example_edges(Edges),
    compile_mindmap_component([nodes(Nodes), edges(Edges)], MindmapCode),
    write_vue_file('vue/GeneratedMindMap.vue', MindmapCode),
    format('   Written: vue/GeneratedMindMap.vue~n'),

    % 3. Viewport component
    format('~n3. Generating viewport component...~n'),
    compile_viewport_component(ViewportCode),
    write_vue_file('vue/GeneratedViewport.vue', ViewportCode),
    format('   Written: vue/GeneratedViewport.vue~n'),

    format('~n=== Compilation Complete ===~n').

% ============================================================================
% COMPONENT GENERATORS
% ============================================================================

%% compile_hierarchy_component(-VueCode)
%
%  Generate a Vue component for pearl trees hierarchy visualization.
%
compile_hierarchy_component(VueCode) :-
    example_nodes(Nodes),
    example_edges(Edges),
    Options = [
        component_name('PearltreesHierarchyGenerated'),
        width(1000),
        height(700),
        theme(light)
    ],
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], Options, VueCode).

%% compile_mindmap_component(+DataSpec, -VueCode)
%
%  Generate a Vue mindmap component from a data specification.
%
compile_mindmap_component(DataSpec, VueCode) :-
    member(nodes(Nodes), DataSpec),
    member(edges(Edges), DataSpec),
    (   member(options(Options), DataSpec)
    ->  true
    ;   Options = [component_name('GeneratedMindMap'), width(800), height(600)]
    ),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], Options, VueCode).

%% compile_viewport_component(-VueCode)
%
%  Generate a reusable viewport component from Prolog specifications.
%
compile_viewport_component(VueCode) :-
    % Get viewport specs from Prolog definitions
    mindmap_viewport:mindmap_zoom_spec(default, ZoomOpts),
    mindmap_viewport:mindmap_pan_spec(default, PanOpts),
    mindmap_viewport:mindmap_viewport_spec(default, ViewportOpts),

    % Extract values
    member(min_scale(MinScale), ZoomOpts),
    member(max_scale(MaxScale), ZoomOpts),
    member(animation_duration(AnimDuration), ZoomOpts),
    member(width(Width), ViewportOpts),
    member(height(Height), ViewportOpts),
    member(show_grid(ShowGrid), ViewportOpts),
    member(grid_size(GridSize), ViewportOpts),

    % Generate Vue component with these specs
    format(string(VueCode),
"<template>
  <div class=\"generated-viewport\">
    <MindMapViewport
      :width=\"~w\"
      :height=\"~w\"
      :zoom-spec=\"zoomSpec\"
      :pan-spec=\"panSpec\"
      :show-grid=\"~w\"
      :grid-size=\"~w\"
      show-controls
      show-zoom-indicator
      @stateChange=\"handleStateChange\"
    >
      <slot />
    </MindMapViewport>
  </div>
</template>

<script setup lang=\"ts\">
/**
 * GeneratedViewport.vue
 *
 * Auto-generated from Prolog viewport specifications:
 * - mindmap_zoom_spec(default, [...])
 * - mindmap_pan_spec(default, [...])
 * - mindmap_viewport_spec(default, [...])
 *
 * Zoom limits: ~w - ~w
 * Animation duration: ~wms
 */

import { ref } from 'vue';
import MindMapViewport from './MindMapViewport.vue';

interface ViewportState {
  scale: number;
  x: number;
  y: number;
}

// Prolog-derived specifications
const zoomSpec = {
  min: ~w,
  max: ~w,
  step: 0.1,
  wheelEnabled: true,
  pinchEnabled: true,
  animationDuration: ~w
};

const panSpec = {
  enabled: true,
  inertia: true,
  inertiaDecay: 0.95,
  constrainToBounds: false
};

const emit = defineEmits<{
  (e: 'stateChange', state: ViewportState): void;
}>();

const handleStateChange = (state: ViewportState) => {
  emit('stateChange', state);
};
</script>

<style scoped>
.generated-viewport {
  width: 100%;
  height: 100%;
}
</style>
", [Width, Height, ShowGrid, GridSize, MinScale, MaxScale, AnimDuration,
    MinScale, MaxScale, AnimDuration]).

% ============================================================================
% TARGET CAPABILITIES DOCUMENTATION
% ============================================================================

%% show_target_capabilities
%
%  Display what Vue target can do directly vs what requires glue.
%
show_target_capabilities :-
    format('~n=== Vue Target Capabilities ===~n~n'),

    vue_target:vue_capabilities(Caps),

    format('DIRECT SUPPORT (no backend required):~n'),
    forall(member(supports(Feature), Caps),
        format('  + ~w~n', [Feature])),

    format('~nREQUIRES GLUE (backend service needed):~n'),
    forall(member(glue_required(Feature), Caps),
        format('  * ~w~n', [Feature])),

    format('~nGLUE PREDICATES:~n'),
    forall(vue_target:requires_glue(Pred/Arity, Reason),
        format('  ~w/~w -> ~w~n', [Pred, Arity, Reason])),

    format('~n=== End Capabilities ===~n').

% ============================================================================
% HELPERS
% ============================================================================

write_vue_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream).

% ============================================================================
% TESTING
% ============================================================================

test_vue_codegen :-
    format('~n=== Vue Code Generation Tests ===~n~n'),

    % Test 1: Hierarchy component generation
    format('Test 1: Hierarchy component generation...~n'),
    compile_hierarchy_component(Code1),
    (   sub_string(Code1, _, _, _, "<template>"),
        sub_string(Code1, _, _, _, "PearltreesHierarchyGenerated")
    ->  format('  PASS: Hierarchy component generated~n')
    ;   format('  FAIL: Hierarchy component incorrect~n')
    ),

    % Test 2: Mindmap with custom data
    format('~nTest 2: Custom mindmap component...~n'),
    example_nodes(Nodes),
    example_edges(Edges),
    compile_mindmap_component([nodes(Nodes), edges(Edges)], Code2),
    (   sub_string(Code2, _, _, _, "const nodes"),
        sub_string(Code2, _, _, _, "const links")
    ->  format('  PASS: Mindmap component has data~n')
    ;   format('  FAIL: Mindmap component missing data~n')
    ),

    % Test 3: Viewport component
    format('~nTest 3: Viewport component from specs...~n'),
    compile_viewport_component(Code3),
    (   sub_string(Code3, _, _, _, "zoomSpec"),
        sub_string(Code3, _, _, _, "panSpec")
    ->  format('  PASS: Viewport has specs~n')
    ;   format('  FAIL: Viewport missing specs~n')
    ),

    % Test 4: Target capabilities
    format('~nTest 4: Target capabilities...~n'),
    vue_target:vue_capabilities(Caps),
    (   member(supports(mindmap_visualization), Caps),
        member(glue_required(database_queries), Caps)
    ->  format('  PASS: Capabilities defined correctly~n')
    ;   format('  FAIL: Capabilities incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Vue examples compilation module loaded~n', [])
), now).
