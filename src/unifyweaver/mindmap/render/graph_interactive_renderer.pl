% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% graph_interactive_renderer.pl - Mind Map Interactive Graph Component
%
% Extends graph_generator.pl with mind map-specific node types,
% layout presets, and React/TypeScript component generation.
%
% Usage:
%   ?- generate_mindmap_graph_component(my_map, Code).

:- module(mindmap_graph_interactive, [
    % Mind map node types
    mindmap_node_type/2,
    declare_mindmap_node_type/2,

    % Layout presets
    mindmap_layout_preset/2,
    declare_layout_preset/2,

    % Component generation
    generate_mindmap_graph_component/2,
    generate_mindmap_graph_component/3,
    generate_mindmap_cytoscape_styles/2,
    generate_mindmap_react_component/3,
    generate_mindmap_vue_component/3,

    % Integration with core mind map DSL
    mindmap_to_graph_nodes/2,
    mindmap_to_graph_edges/2,

    % Testing
    test_graph_interactive/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic mindmap_node_type/2.
:- dynamic mindmap_layout_preset/2.

% ============================================================================
% DEFAULT MIND MAP NODE TYPES
% ============================================================================

% Root node styling
mindmap_node_type(root, [
    shape(ellipse),
    background_color('#4a90d9'),
    border_color('#2c5a8c'),
    border_width(3),
    color('#ffffff'),
    font_size(16),
    font_weight(bold),
    width(100),
    height(50),
    text_valign(center),
    text_halign(center)
]).

% Hub node styling (main topics)
mindmap_node_type(hub, [
    shape(ellipse),
    background_color('#6ab04c'),
    border_color('#4a904c'),
    border_width(2),
    color('#ffffff'),
    font_size(14),
    width(90),
    height(45),
    text_valign(center),
    text_halign(center)
]).

% Branch node styling (subtopics)
mindmap_node_type(branch, [
    shape(roundrectangle),
    background_color('#f0932b'),
    border_color('#c07020'),
    border_width(2),
    color('#ffffff'),
    font_size(12),
    width(80),
    height(40),
    text_valign(center),
    text_halign(center)
]).

% Leaf node styling
mindmap_node_type(leaf, [
    shape(ellipse),
    background_color('#eb4d4b'),
    border_color('#cb2d2b'),
    border_width(1),
    color('#ffffff'),
    font_size(11),
    width(70),
    height(35),
    text_valign(center),
    text_halign(center)
]).

% Default node styling
mindmap_node_type(default, [
    shape(ellipse),
    background_color('#e8f4fc'),
    border_color('#4a90d9'),
    border_width(2),
    color('#333333'),
    font_size(12),
    width(80),
    height(40),
    text_valign(center),
    text_halign(center)
]).

% Link node (has URL)
mindmap_node_type(link, [
    shape(ellipse),
    background_color('#9b59b6'),
    border_color('#7b429c'),
    border_width(2),
    color('#ffffff'),
    font_size(12),
    width(80),
    height(40),
    text_valign(center),
    text_halign(center),
    cursor(pointer)
]).

% ============================================================================
% DEFAULT LAYOUT PRESETS
% ============================================================================

mindmap_layout_preset(radial, [
    name(concentric),
    concentric(depth_function),
    levelWidth(constant(2)),
    minNodeSpacing(50),
    animate(true),
    animationDuration(500)
]).

mindmap_layout_preset(hierarchical, [
    name(dagre),
    rankDir('TB'),
    nodeSep(50),
    rankSep(80),
    animate(true),
    animationDuration(500)
]).

mindmap_layout_preset(force, [
    name(cose),
    nodeRepulsion(8000),
    idealEdgeLength(100),
    edgeElasticity(100),
    gravity(0.25),
    numIter(1000),
    animate(true),
    animationDuration(500)
]).

mindmap_layout_preset(circle, [
    name(circle),
    radius(200),
    startAngle(0),
    sweep(6.283),
    clockwise(true),
    animate(true),
    animationDuration(500)
]).

mindmap_layout_preset(grid, [
    name(grid),
    columns(4),
    padding(30),
    animate(true),
    animationDuration(500)
]).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%% generate_mindmap_graph_component(+Name, -Code)
%
%  Generate a React/TypeScript component for mind map visualization.
%
generate_mindmap_graph_component(Name, Code) :-
    generate_mindmap_graph_component(Name, [], Code).

%% generate_mindmap_graph_component(+Name, +Options, -Code)
generate_mindmap_graph_component(Name, Options, Code) :-
    option_value(Options, layout, force, LayoutName),
    option_value(Options, theme, light, Theme),
    option_value(Options, width, '100%', Width),
    option_value(Options, height, '600px', Height),

    % Get layout preset
    (   mindmap_layout_preset(LayoutName, LayoutConfig)
    ->  true
    ;   mindmap_layout_preset(force, LayoutConfig)
    ),

    % Generate component name
    atom_string(Name, NameStr),

    % Generate styles
    generate_mindmap_cytoscape_styles(Theme, StylesCode),

    % Generate layout config
    generate_layout_config(LayoutConfig, LayoutCode),

    format(string(Code),
"// Mind Map Interactive Component: ~w
// Generated by UnifyWeaver

import React, { useRef, useEffect, useCallback } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import coseBilkent from 'cytoscape-cose-bilkent';

// Register layout extensions
cytoscape.use(dagre);
cytoscape.use(coseBilkent);

interface MindMapNode {
  id: string;
  label: string;
  type?: 'root' | 'hub' | 'branch' | 'leaf' | 'link';
  url?: string;
  parent?: string;
}

interface MindMapEdge {
  source: string;
  target: string;
  type?: string;
}

interface ~wProps {
  nodes: MindMapNode[];
  edges: MindMapEdge[];
  onNodeClick?: (node: MindMapNode) => void;
  onNodeDoubleClick?: (node: MindMapNode) => void;
  width?: string;
  height?: string;
}

const styles: cytoscape.Stylesheet[] = ~w;

const layoutConfig = ~w;

export const ~w: React.FC<~wProps> = ({
  nodes,
  edges,
  onNodeClick,
  onNodeDoubleClick,
  width = '~w',
  height = '~w'
}) => {
  const cyRef = useRef<cytoscape.Core | null>(null);

  const elements = [
    ...nodes.map(node => ({
      data: {
        id: node.id,
        label: node.label,
        type: node.type || 'default',
        url: node.url
      }
    })),
    ...edges.map(edge => ({
      data: {
        id: `${edge.source}-${edge.target}`,
        source: edge.source,
        target: edge.target,
        type: edge.type || 'default'
      }
    }))
  ];

  const handleNodeClick = useCallback((event: cytoscape.EventObject) => {
    const node = event.target.data();
    if (onNodeClick) {
      onNodeClick(node);
    } else if (node.url) {
      window.open(node.url, '_blank');
    }
  }, [onNodeClick]);

  const handleNodeDoubleClick = useCallback((event: cytoscape.EventObject) => {
    const node = event.target.data();
    if (onNodeDoubleClick) {
      onNodeDoubleClick(node);
    }
  }, [onNodeDoubleClick]);

  useEffect(() => {
    if (cyRef.current) {
      cyRef.current.on('tap', 'node', handleNodeClick);
      cyRef.current.on('dbltap', 'node', handleNodeDoubleClick);

      return () => {
        cyRef.current?.off('tap', 'node', handleNodeClick);
        cyRef.current?.off('dbltap', 'node', handleNodeDoubleClick);
      };
    }
  }, [handleNodeClick, handleNodeDoubleClick]);

  return (
    <div style={{ width, height }}>
      <CytoscapeComponent
        cy={(cy) => { cyRef.current = cy; }}
        elements={elements}
        stylesheet={styles}
        layout={layoutConfig}
        style={{ width: '100%', height: '100%' }}
        minZoom={0.1}
        maxZoom={5}
        wheelSensitivity={0.2}
      />
    </div>
  );
};

export default ~w;
", [NameStr, NameStr, StylesCode, LayoutCode, NameStr, NameStr, Width, Height, NameStr]).

%% generate_mindmap_cytoscape_styles(+Theme, -StylesCode)
generate_mindmap_cytoscape_styles(light, StylesCode) :-
    StylesCode = "[
  // Node base styles
  {
    selector: 'node',
    style: {
      'label': 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-family': 'sans-serif',
      'background-color': '#e8f4fc',
      'border-color': '#4a90d9',
      'border-width': 2,
      'color': '#333333',
      'font-size': 12
    }
  },
  // Root node
  {
    selector: 'node[type=\"root\"]',
    style: {
      'background-color': '#4a90d9',
      'border-color': '#2c5a8c',
      'border-width': 3,
      'color': '#ffffff',
      'font-size': 16,
      'font-weight': 'bold',
      'width': 100,
      'height': 50
    }
  },
  // Hub node
  {
    selector: 'node[type=\"hub\"]',
    style: {
      'background-color': '#6ab04c',
      'border-color': '#4a904c',
      'color': '#ffffff',
      'font-size': 14,
      'width': 90,
      'height': 45
    }
  },
  // Branch node
  {
    selector: 'node[type=\"branch\"]',
    style: {
      'shape': 'roundrectangle',
      'background-color': '#f0932b',
      'border-color': '#c07020',
      'color': '#ffffff'
    }
  },
  // Leaf node
  {
    selector: 'node[type=\"leaf\"]',
    style: {
      'background-color': '#eb4d4b',
      'border-color': '#cb2d2b',
      'border-width': 1,
      'color': '#ffffff',
      'font-size': 11,
      'width': 70,
      'height': 35
    }
  },
  // Link node
  {
    selector: 'node[url]',
    style: {
      'border-style': 'dashed',
      'cursor': 'pointer'
    }
  },
  // Selected node
  {
    selector: 'node:selected',
    style: {
      'border-color': '#ff6b6b',
      'border-width': 4
    }
  },
  // Edge styles
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#666666',
      'target-arrow-color': '#666666',
      'target-arrow-shape': 'none',
      'curve-style': 'bezier'
    }
  },
  // Strong edge
  {
    selector: 'edge[type=\"strong\"]',
    style: {
      'width': 3,
      'line-color': '#333333'
    }
  },
  // Weak edge
  {
    selector: 'edge[type=\"weak\"]',
    style: {
      'width': 1,
      'line-color': '#aaaaaa',
      'line-style': 'dashed'
    }
  }
]".

generate_mindmap_cytoscape_styles(dark, StylesCode) :-
    StylesCode = "[
  {
    selector: 'node',
    style: {
      'label': 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-family': 'sans-serif',
      'background-color': '#2d3748',
      'border-color': '#4a9ce9',
      'border-width': 2,
      'color': '#e2e8f0',
      'font-size': 12
    }
  },
  {
    selector: 'node[type=\"root\"]',
    style: {
      'background-color': '#5a9ce9',
      'border-color': '#3c6a9c',
      'border-width': 3,
      'color': '#ffffff',
      'font-size': 16,
      'font-weight': 'bold',
      'width': 100,
      'height': 50
    }
  },
  {
    selector: 'node[type=\"hub\"]',
    style: {
      'background-color': '#7ac05c',
      'border-color': '#5aa05c',
      'color': '#ffffff',
      'font-size': 14
    }
  },
  {
    selector: 'node[type=\"branch\"]',
    style: {
      'shape': 'roundrectangle',
      'background-color': '#ffaa4b',
      'border-color': '#d08030',
      'color': '#000000'
    }
  },
  {
    selector: 'node[type=\"leaf\"]',
    style: {
      'background-color': '#fb5d5b',
      'border-color': '#db3d3b',
      'color': '#ffffff'
    }
  },
  {
    selector: 'node:selected',
    style: {
      'border-color': '#ff6b6b',
      'border-width': 4
    }
  },
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#718096',
      'curve-style': 'bezier'
    }
  }
]".

generate_mindmap_cytoscape_styles(_, StylesCode) :-
    generate_mindmap_cytoscape_styles(light, StylesCode).

%% generate_layout_config(+LayoutConfig, -LayoutCode)
generate_layout_config(LayoutConfig, LayoutCode) :-
    findall(PropStr,
            (member(Prop, LayoutConfig),
             format_layout_prop(Prop, PropStr)),
            PropStrs),
    atomic_list_concat(PropStrs, ',\n    ', PropsConcat),
    format(string(LayoutCode), "{\n    ~w\n  }", [PropsConcat]).

format_layout_prop(name(Name), PropStr) :-
    format(string(PropStr), "name: '~w'", [Name]).
format_layout_prop(rankDir(Dir), PropStr) :-
    format(string(PropStr), "rankDir: '~w'", [Dir]).
format_layout_prop(concentric(Func), PropStr) :-
    format(string(PropStr), "concentric: (node: any) => node.data('depth') || 0", []).
format_layout_prop(levelWidth(constant(N)), PropStr) :-
    format(string(PropStr), "levelWidth: () => ~w", [N]).
format_layout_prop(animate(Bool), PropStr) :-
    format(string(PropStr), "animate: ~w", [Bool]).
format_layout_prop(animationDuration(N), PropStr) :-
    format(string(PropStr), "animationDuration: ~w", [N]).
format_layout_prop(Prop, PropStr) :-
    Prop =.. [Key, Value],
    (   number(Value)
    ->  format(string(PropStr), "~w: ~w", [Key, Value])
    ;   format(string(PropStr), "~w: '~w'", [Key, Value])
    ).

%% generate_mindmap_react_component(+Nodes, +Edges, -Code)
generate_mindmap_react_component(Nodes, Edges, Code) :-
    generate_nodes_json(Nodes, NodesJson),
    generate_edges_json(Edges, EdgesJson),
    format(string(Code),
"const nodes = ~w;
const edges = ~w;
", [NodesJson, EdgesJson]).

%% generate_mindmap_vue_component(+Name, +Options, -VueCode)
%
%  Generate a Vue 3 SFC component using Cytoscape.js for mind map visualization.
%
generate_mindmap_vue_component(Name, Options, VueCode) :-
    option_value(Options, layout, force, LayoutName),
    option_value(Options, theme, light, Theme),
    option_value(Options, width, '100%', Width),
    option_value(Options, height, '600px', Height),

    % Get layout preset
    (   mindmap_layout_preset(LayoutName, LayoutConfig)
    ->  true
    ;   mindmap_layout_preset(force, LayoutConfig)
    ),

    atom_string(Name, NameStr),
    generate_mindmap_cytoscape_styles(Theme, StylesCode),
    generate_layout_config(LayoutConfig, LayoutCode),

    format(string(VueCode),
"<template>
  <div class=\"mindmap-cytoscape-container\" :style=\"containerStyle\">
    <div ref=\"cyContainer\" class=\"cy-container\" />
    <div class=\"controls\">
      <button @click=\"fit\" title=\"Fit to content\">[ ]</button>
      <button @click=\"zoomIn\" title=\"Zoom In\">+</button>
      <button @click=\"zoomOut\" title=\"Zoom Out\">-</button>
      <button @click=\"resetZoom\" title=\"Reset\">1:1</button>
      <select v-model=\"currentLayout\" @change=\"applyLayout\">
        <option value=\"force\">Force</option>
        <option value=\"hierarchical\">Hierarchical</option>
        <option value=\"radial\">Radial</option>
        <option value=\"circle\">Circle</option>
        <option value=\"grid\">Grid</option>
      </select>
    </div>
  </div>
</template>

<script setup lang=\"ts\">
/**
 * ~w - Mind Map Cytoscape Component
 * Generated by UnifyWeaver
 *
 * Uses Cytoscape.js for interactive graph visualization with
 * multiple layout algorithms and theme support.
 */

import { ref, onMounted, onUnmounted, computed, watch, type PropType } from 'vue';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import coseBilkent from 'cytoscape-cose-bilkent';

// Register layout extensions
cytoscape.use(dagre);
cytoscape.use(coseBilkent);

// Types
interface MindMapNode {
  id: string;
  label: string;
  type?: 'root' | 'hub' | 'branch' | 'leaf' | 'link' | 'default';
  url?: string;
  depth?: number;
}

interface MindMapEdge {
  source: string;
  target: string;
  type?: string;
}

interface Props {
  nodes: MindMapNode[];
  edges: MindMapEdge[];
  width?: string;
  height?: string;
  theme?: 'light' | 'dark';
  initialLayout?: string;
}

const props = withDefaults(defineProps<Props>(), {
  nodes: () => [],
  edges: () => [],
  width: '~w',
  height: '~w',
  theme: '~w',
  initialLayout: '~w'
});

const emit = defineEmits<{
  (e: 'nodeClick', node: MindMapNode): void;
  (e: 'nodeDoubleClick', node: MindMapNode): void;
  (e: 'edgeClick', edge: MindMapEdge): void;
}>();

// Refs
const cyContainer = ref<HTMLDivElement | null>(null);
const currentLayout = ref(props.initialLayout);
let cy: cytoscape.Core | null = null;

// Computed
const containerStyle = computed(() => ({
  width: props.width,
  height: props.height
}));

// Cytoscape styles
const cytoscapeStyles: cytoscape.Stylesheet[] = ~w;

// Layout configurations
const layoutConfigs: Record<string, any> = {
  force: ~w,
  hierarchical: {
    name: 'dagre',
    rankDir: 'TB',
    nodeSep: 50,
    rankSep: 80,
    animate: true,
    animationDuration: 500
  },
  radial: {
    name: 'concentric',
    concentric: (node: any) => node.data('depth') || 0,
    levelWidth: () => 2,
    minNodeSpacing: 50,
    animate: true,
    animationDuration: 500
  },
  circle: {
    name: 'circle',
    radius: 200,
    startAngle: 0,
    sweep: 6.283,
    clockwise: true,
    animate: true,
    animationDuration: 500
  },
  grid: {
    name: 'grid',
    columns: 4,
    padding: 30,
    animate: true,
    animationDuration: 500
  }
};

// Methods
const initCytoscape = () => {
  if (!cyContainer.value) return;

  const elements = [
    ...props.nodes.map(node => ({
      data: {
        id: node.id,
        label: node.label,
        type: node.type || 'default',
        url: node.url,
        depth: node.depth || 0
      }
    })),
    ...props.edges.map(edge => ({
      data: {
        id: `${edge.source}-${edge.target}`,
        source: edge.source,
        target: edge.target,
        type: edge.type || 'default'
      }
    }))
  ];

  cy = cytoscape({
    container: cyContainer.value,
    elements,
    style: cytoscapeStyles,
    layout: layoutConfigs[currentLayout.value] || layoutConfigs.force,
    minZoom: 0.1,
    maxZoom: 5,
    wheelSensitivity: 0.2
  });

  // Event handlers
  cy.on('tap', 'node', (event) => {
    const nodeData = event.target.data();
    emit('nodeClick', nodeData);
    if (nodeData.url) {
      window.open(nodeData.url, '_blank');
    }
  });

  cy.on('dbltap', 'node', (event) => {
    emit('nodeDoubleClick', event.target.data());
  });

  cy.on('tap', 'edge', (event) => {
    emit('edgeClick', event.target.data());
  });
};

const fit = () => {
  cy?.fit(undefined, 50);
};

const zoomIn = () => {
  if (!cy) return;
  cy.zoom(cy.zoom() * 1.3);
};

const zoomOut = () => {
  if (!cy) return;
  cy.zoom(cy.zoom() / 1.3);
};

const resetZoom = () => {
  cy?.reset();
};

const applyLayout = () => {
  if (!cy) return;
  const layout = cy.layout(layoutConfigs[currentLayout.value] || layoutConfigs.force);
  layout.run();
};

// Lifecycle
onMounted(() => {
  initCytoscape();
});

onUnmounted(() => {
  cy?.destroy();
  cy = null;
});

// Watch for data changes
watch(() => [props.nodes, props.edges], () => {
  if (cy) {
    cy.destroy();
    initCytoscape();
  }
}, { deep: true });

watch(() => props.theme, () => {
  // Theme change would require regenerating styles
  if (cy) {
    cy.destroy();
    initCytoscape();
  }
});

// Expose methods
defineExpose({
  fit,
  zoomIn,
  zoomOut,
  resetZoom,
  applyLayout,
  getCy: () => cy
});
</script>

<style scoped>
.mindmap-cytoscape-container {
  position: relative;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}

.cy-container {
  width: 100%;
  height: 100%;
}

.controls {
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  gap: 8px;
  z-index: 10;
}

.controls button {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.controls button:hover {
  background: #f7fafc;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.controls select {
  padding: 8px 12px;
  border: none;
  border-radius: 6px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  cursor: pointer;
  font-size: 14px;
}
</style>
", [NameStr, Width, Height, Theme, LayoutName, StylesCode, LayoutCode]).

generate_nodes_json([], "[]").
generate_nodes_json(Nodes, Json) :-
    maplist(node_to_json, Nodes, NodeJsons),
    atomic_list_concat(NodeJsons, ', ', NodesConcat),
    format(string(Json), "[~w]", [NodesConcat]).

node_to_json(node(Id, Props), Json) :-
    atom_string(Id, IdStr),
    (   member(label(Label), Props)
    ->  true
    ;   Label = IdStr
    ),
    (   member(type(Type), Props)
    ->  atom_string(Type, TypeStr)
    ;   TypeStr = "default"
    ),
    (   member(link(URL), Props)
    ->  format(string(UrlPart), ", url: '~w'", [URL])
    ;   UrlPart = ""
    ),
    format(string(Json),
        "{ id: '~w', label: '~w', type: '~w'~w }",
        [IdStr, Label, TypeStr, UrlPart]).

generate_edges_json([], "[]").
generate_edges_json(Edges, Json) :-
    maplist(edge_to_json, Edges, EdgeJsons),
    atomic_list_concat(EdgeJsons, ', ', EdgesConcat),
    format(string(Json), "[~w]", [EdgesConcat]).

edge_to_json(edge(From, To, _Props), Json) :-
    atom_string(From, FromStr),
    atom_string(To, ToStr),
    format(string(Json), "{ source: '~w', target: '~w' }", [FromStr, ToStr]).

%% mindmap_to_graph_nodes(+MindmapNodes, -GraphNodes)
mindmap_to_graph_nodes([], []).
mindmap_to_graph_nodes([node(Id, Props) | Rest], [GraphNode | RestGraph]) :-
    mindmap_node_to_graph_node(Id, Props, GraphNode),
    mindmap_to_graph_nodes(Rest, RestGraph).

mindmap_node_to_graph_node(Id, Props, graph_node(Id, GraphProps)) :-
    findall(Prop,
            (member(Prop, Props),
             is_graph_prop(Prop)),
            GraphProps).

is_graph_prop(label(_)).
is_graph_prop(type(_)).
is_graph_prop(color(_)).
is_graph_prop(link(_)).

%% mindmap_to_graph_edges(+MindmapEdges, -GraphEdges)
mindmap_to_graph_edges([], []).
mindmap_to_graph_edges([edge(From, To, Props) | Rest], [graph_edge(From, To, Props) | RestGraph]) :-
    mindmap_to_graph_edges(Rest, RestGraph).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_mindmap_node_type(+Type, +Style)
declare_mindmap_node_type(Type, Style) :-
    retractall(mindmap_node_type(Type, _)),
    assertz(mindmap_node_type(Type, Style)).

%% declare_layout_preset(+Name, +Config)
declare_layout_preset(Name, Config) :-
    retractall(mindmap_layout_preset(Name, _)),
    assertz(mindmap_layout_preset(Name, Config)).

% ============================================================================
% UTILITIES
% ============================================================================

option_value(Options, Key, Default, Value) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

% ============================================================================
% TESTING
% ============================================================================

test_graph_interactive :-
    format('~n=== Graph Interactive Renderer Tests ===~n~n'),

    % Test 1: Node types exist
    format('Test 1: Node types exist...~n'),
    (   mindmap_node_type(root, _),
        mindmap_node_type(hub, _),
        mindmap_node_type(branch, _)
    ->  format('  PASS: Node types defined~n')
    ;   format('  FAIL: Missing node types~n')
    ),

    % Test 2: Layout presets exist
    format('~nTest 2: Layout presets exist...~n'),
    (   mindmap_layout_preset(radial, _),
        mindmap_layout_preset(hierarchical, _),
        mindmap_layout_preset(force, _)
    ->  format('  PASS: Layout presets defined~n')
    ;   format('  FAIL: Missing layout presets~n')
    ),

    % Test 3: Generate component
    format('~nTest 3: Generate component...~n'),
    generate_mindmap_graph_component(test_map, Code),
    (   sub_string(Code, _, _, _, "CytoscapeComponent")
    ->  format('  PASS: Component generated~n')
    ;   format('  FAIL: Component generation failed~n')
    ),

    % Test 4: Generate styles
    format('~nTest 4: Generate styles...~n'),
    generate_mindmap_cytoscape_styles(light, LightStyles),
    generate_mindmap_cytoscape_styles(dark, DarkStyles),
    (   sub_string(LightStyles, _, _, _, "#4a90d9"),
        sub_string(DarkStyles, _, _, _, "#5a9ce9")
    ->  format('  PASS: Theme styles generated~n')
    ;   format('  FAIL: Style generation failed~n')
    ),

    % Test 5: Generate Vue component
    format('~nTest 5: Generate Vue component...~n'),
    generate_mindmap_vue_component(test_map, [], VueCode),
    (   sub_string(VueCode, _, _, _, "<template>"),
        sub_string(VueCode, _, _, _, "<script setup lang=\"ts\">"),
        sub_string(VueCode, _, _, _, "cytoscape")
    ->  format('  PASS: Vue component generated~n')
    ;   format('  FAIL: Vue component generation failed~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Mind map graph interactive renderer loaded~n', [])
), now).
