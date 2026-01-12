%% test_vue_codegen.pl - plunit tests for Vue code generation
%%
%% Tests Vue 3 SFC code generation from D3 renderer, Cytoscape renderer,
%% and Vue target module.
%%
%% Run with: swipl -g "run_tests" -t halt test_vue_codegen.pl

:- module(test_vue_codegen, []).

:- use_module(library(plunit)).

%% Load Vue-related modules
:- use_module('../../mindmap/render/d3_renderer').
:- use_module('../../mindmap/render/graph_interactive_renderer').
:- use_module('../../targets/vue_target').

%% ============================================================================
%% Test Data
%% ============================================================================

test_nodes([
    node(root, [label("Main Topic"), type(root)]),
    node(child1, [label("Child 1"), type(hub)]),
    node(child2, [label("Child 2"), type(branch)]),
    node(leaf1, [label("Leaf Node"), type(leaf), link("https://example.com")])
]).

test_edges([
    edge(root, child1, []),
    edge(root, child2, []),
    edge(child1, leaf1, [type(strong)])
]).

%% ============================================================================
%% Tests: D3 Renderer Vue Output
%% ============================================================================

:- begin_tests(d3_vue_generation).

test(d3_vue_generates_template) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "<template>").

test(d3_vue_generates_script_setup) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "<script setup lang=\"ts\">").

test(d3_vue_generates_style_scoped) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "<style scoped>").

test(d3_vue_imports_vue_composition) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "import { ref, onMounted").

test(d3_vue_imports_d3) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "import * as d3 from 'd3'").

test(d3_vue_has_zoom_controls) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "zoomIn"),
    sub_string(Code, _, _, _, "zoomOut"),
    sub_string(Code, _, _, _, "fitToContent").

test(d3_vue_embeds_node_data) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "Main Topic"),
    sub_string(Code, _, _, _, "Child 1").

test(d3_vue_respects_dimensions, [nondet]) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [width(1200), height(800)], Code),
    sub_string(Code, _, _, _, "1200"),
    sub_string(Code, _, _, _, "800").

:- end_tests(d3_vue_generation).

%% ============================================================================
%% Tests: Cytoscape Renderer Vue Output
%% ============================================================================

:- begin_tests(cytoscape_vue_generation).

test(cytoscape_vue_generates_sfc) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "<template>"),
    sub_string(Code, _, _, _, "<script setup lang=\"ts\">"),
    sub_string(Code, _, _, _, "<style scoped>").

test(cytoscape_vue_imports_cytoscape) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "import cytoscape from 'cytoscape'").

test(cytoscape_vue_imports_dagre) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "import dagre from 'cytoscape-dagre'").

test(cytoscape_vue_has_layout_select) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "<select v-model=\"currentLayout\""),
    sub_string(Code, _, _, _, "Force"),
    sub_string(Code, _, _, _, "Hierarchical").

test(cytoscape_vue_has_zoom_controls) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "@click=\"fit\""),
    sub_string(Code, _, _, _, "@click=\"zoomIn\""),
    sub_string(Code, _, _, _, "@click=\"zoomOut\"").

test(cytoscape_vue_has_event_emits) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "emit('nodeClick'"),
    sub_string(Code, _, _, _, "emit('nodeDoubleClick'").

test(cytoscape_vue_exposes_methods) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "defineExpose({"),
    sub_string(Code, _, _, _, "getCy: () => cy").

test(cytoscape_vue_respects_theme, [nondet]) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [theme(dark)], Code),
    sub_string(Code, _, _, _, "theme: 'dark'").

:- end_tests(cytoscape_vue_generation).

%% ============================================================================
%% Tests: Vue Target Module
%% ============================================================================

:- begin_tests(vue_target_module).

test(vue_target_capabilities_defined) :-
    vue_target:vue_capabilities(Caps),
    is_list(Caps),
    member(supports(mindmap_visualization), Caps).

test(vue_target_glue_required_for_database) :-
    vue_target:requires_glue(pearl_trees/5, Reason),
    Reason == database_access.

test(vue_target_glue_required_for_computation) :-
    vue_target:requires_glue(compute_embedding/2, Reason),
    Reason == computation.

test(vue_target_no_glue_for_unknown) :-
    \+ vue_target:requires_glue(some_random_pred/3, _).

test(vue_target_compile_direct_component) :-
    vue_target:compile_predicate_to_vue(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "<template>"),
    sub_string(Code, _, _, _, "<script setup").

test(vue_target_compile_glue_component) :-
    vue_target:compile_predicate_to_vue(pearltrees:pearl_trees/5, [], Code),
    sub_string(Code, _, _, _, "Requires glue"),
    sub_string(Code, _, _, _, "fetch(").

test(vue_target_glue_component_has_loading_state) :-
    vue_target:compile_predicate_to_vue(pearltrees:pearl_trees/5, [], Code),
    sub_string(Code, _, _, _, "v-if=\"loading\""),
    sub_string(Code, _, _, _, "v-else-if=\"error\"").

test(vue_target_glue_component_has_api_call) :-
    vue_target:compile_predicate_to_vue(pearltrees:pearl_trees/5, [], Code),
    sub_string(Code, _, _, _, "/api/prolog/").

:- end_tests(vue_target_module).

%% ============================================================================
%% Tests: Vue SFC Structure
%% ============================================================================

:- begin_tests(vue_sfc_structure).

test(vue_sfc_has_all_three_sections) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    % Count occurrences of each section
    sub_string(Code, TemplateStart, _, _, "<template>"),
    sub_string(Code, ScriptStart, _, _, "<script"),
    sub_string(Code, StyleStart, _, _, "<style"),
    % Verify order: template < script < style
    TemplateStart < ScriptStart,
    ScriptStart < StyleStart.

test(vue_sfc_template_is_valid_html, [nondet]) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    % Check template has matching open/close div
    sub_string(Code, _, _, _, "<div class=\"mindmap-container\">"),
    sub_string(Code, _, _, _, "</template>").

test(vue_sfc_script_has_typescript) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "lang=\"ts\"").

test(vue_sfc_style_is_scoped) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "scoped").

:- end_tests(vue_sfc_structure).

%% ============================================================================
%% Tests: TypeScript Types in Vue Output
%% ============================================================================

:- begin_tests(vue_typescript_types).

test(vue_has_node_interface) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "interface Node").

test(vue_has_link_interface) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "interface Link").

test(vue_has_props_interface) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "interface Props").

test(vue_has_typed_refs) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "ref<SVGSVGElement").

test(vue_has_defineprops_typed) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "defineProps<Props>").

test(vue_has_defineemits_typed) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "defineEmits<").

:- end_tests(vue_typescript_types).

%% ============================================================================
%% Tests: Vue Composition API Usage
%% ============================================================================

:- begin_tests(vue_composition_api).

test(vue_uses_ref) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "const svgRef = ref<").

test(vue_uses_computed) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "computed(() =>").

test(vue_uses_onmounted) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "onMounted(").

test(vue_uses_onunmounted) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "onUnmounted(").

test(vue_uses_watch) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "watch(").

test(vue_uses_defineexpose) :-
    mindmap_graph_interactive:generate_mindmap_vue_component(test_map, [], Code),
    sub_string(Code, _, _, _, "defineExpose(").

:- end_tests(vue_composition_api).

%% ============================================================================
%% Tests: Theme Support
%% ============================================================================

:- begin_tests(vue_theme_support).

test(vue_has_light_theme_colors) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "#ffffff"),
    sub_string(Code, _, _, _, "#4a90d9").

test(vue_has_dark_theme_colors) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "#1a1a2e"),
    sub_string(Code, _, _, _, "#5a9ce9").

test(vue_theme_is_reactive) :-
    test_nodes(Nodes),
    test_edges(Edges),
    mindmap_render_d3:render_d3_vue(Nodes, Edges, [], [], Code),
    sub_string(Code, _, _, _, "watch(() => props.theme").

:- end_tests(vue_theme_support).

%% ============================================================================
%% Run tests when loaded directly
%% ============================================================================

:- initialization(run_tests, main).
