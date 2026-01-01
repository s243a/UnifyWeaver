% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Graph Generator - Declarative Graph Visualization with Cytoscape.js
%
% This module provides declarative graph definitions that generate
% TypeScript/React components using Cytoscape.js for visualization.
%
% Usage:
%   % Define graph nodes
%   graph_node(abraham, [label("Abraham"), type(person), color('#7c3aed')]).
%   graph_node(isaac, [label("Isaac"), type(person)]).
%
%   % Define edges
%   graph_edge(abraham, isaac, [relation(parent), label("father of")]).
%
%   % Generate React component
%   ?- generate_graph_component(family_tree, Code).

:- module(graph_generator, [
    % Graph definition predicates
    graph_node/2,                       % graph_node(+Id, +Properties)
    graph_edge/3,                       % graph_edge(+From, +To, +Properties)
    graph_spec/2,                       % graph_spec(+Name, +Config)

    % Graph management
    declare_graph_node/2,               % declare_graph_node(+Id, +Properties)
    declare_graph_edge/3,               % declare_graph_edge(+From, +To, +Properties)
    declare_graph_spec/2,               % declare_graph_spec(+Name, +Config)
    clear_graph/0,                      % clear_graph
    clear_graph/1,                      % clear_graph(+Name)

    % Query predicates
    all_nodes/1,                        % all_nodes(-Nodes)
    all_edges/1,                        % all_edges(-Edges)
    nodes_for_graph/2,                  % nodes_for_graph(+Name, -Nodes)
    edges_for_graph/2,                  % edges_for_graph(+Name, -Edges)

    % Code generation
    generate_graph_component/2,         % generate_graph_component(+Name, -Code)
    generate_graph_component/3,         % generate_graph_component(+Name, +Options, -Code)
    generate_graph_styles/2,            % generate_graph_styles(+Name, -CssCode)
    generate_cytoscape_config/2,        % generate_cytoscape_config(+Name, -ConfigCode)
    generate_graph_data/2,              % generate_graph_data(+Name, -DataCode)

    % Testing
    test_graph_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic graph_node/2.
:- dynamic graph_edge/3.
:- dynamic graph_spec/2.

:- discontiguous graph_node/2.
:- discontiguous graph_edge/3.
:- discontiguous graph_spec/2.

% ============================================================================
% DEFAULT GRAPH DEFINITIONS - Family Tree Example
% ============================================================================

% Family tree nodes
graph_node(abraham, [label("Abraham"), type(person), generation(1)]).
graph_node(sarah, [label("Sarah"), type(person), generation(1)]).
graph_node(hagar, [label("Hagar"), type(person), generation(1)]).
graph_node(ishmael, [label("Ishmael"), type(person), generation(2)]).
graph_node(isaac, [label("Isaac"), type(person), generation(2)]).
graph_node(rebekah, [label("Rebekah"), type(person), generation(2)]).
graph_node(esau, [label("Esau"), type(person), generation(3)]).
graph_node(jacob, [label("Jacob"), type(person), generation(3)]).

% Family tree edges (parent relationships)
graph_edge(abraham, ishmael, [relation(parent), via(hagar)]).
graph_edge(hagar, ishmael, [relation(parent)]).
graph_edge(abraham, isaac, [relation(parent), via(sarah)]).
graph_edge(sarah, isaac, [relation(parent)]).
graph_edge(isaac, esau, [relation(parent)]).
graph_edge(isaac, jacob, [relation(parent)]).
graph_edge(rebekah, esau, [relation(parent)]).
graph_edge(rebekah, jacob, [relation(parent)]).

% Graph specification
graph_spec(family_tree, [
    title("Family Tree"),
    description("Biblical family relationships"),
    layout(cose),
    theme(dark),
    nodes([abraham, sarah, hagar, ishmael, isaac, rebekah, esau, jacob]),
    node_colors([
        person('#7c3aed')
    ]),
    edge_colors([
        parent('#00d4ff')
    ])
]).

% Simple directed graph example
graph_node(a, [label("A"), type(vertex)]).
graph_node(b, [label("B"), type(vertex)]).
graph_node(c, [label("C"), type(vertex)]).
graph_node(d, [label("D"), type(vertex)]).

graph_edge(a, b, [weight(1)]).
graph_edge(b, c, [weight(2)]).
graph_edge(c, d, [weight(1)]).
graph_edge(a, d, [weight(3)]).

graph_spec(simple_graph, [
    title("Simple Directed Graph"),
    layout(circle),
    theme(light),
    nodes([a, b, c, d])
]).

% ============================================================================
% GRAPH MANAGEMENT
% ============================================================================

%% declare_graph_node(+Id, +Properties)
%  Add a new graph node.
declare_graph_node(Id, Properties) :-
    (   graph_node(Id, _)
    ->  retract(graph_node(Id, _))
    ;   true
    ),
    assertz(graph_node(Id, Properties)).

%% declare_graph_edge(+From, +To, +Properties)
%  Add a new graph edge.
declare_graph_edge(From, To, Properties) :-
    (   graph_edge(From, To, _)
    ->  retract(graph_edge(From, To, _))
    ;   true
    ),
    assertz(graph_edge(From, To, Properties)).

%% declare_graph_spec(+Name, +Config)
%  Declare a graph specification.
declare_graph_spec(Name, Config) :-
    (   graph_spec(Name, _)
    ->  retract(graph_spec(Name, _))
    ;   true
    ),
    assertz(graph_spec(Name, Config)).

%% clear_graph
%  Clear all graph definitions.
clear_graph :-
    retractall(graph_node(_, _)),
    retractall(graph_edge(_, _, _)),
    retractall(graph_spec(_, _)).

%% clear_graph(+Name)
%  Clear a specific graph specification (keeps nodes/edges).
clear_graph(Name) :-
    retractall(graph_spec(Name, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% all_nodes(-Nodes)
%  Get all defined nodes.
all_nodes(Nodes) :-
    findall(Id, graph_node(Id, _), Nodes).

%% all_edges(-Edges)
%  Get all defined edges as from-to pairs.
all_edges(Edges) :-
    findall(From-To, graph_edge(From, To, _), Edges).

%% nodes_for_graph(+Name, -Nodes)
%  Get nodes for a specific graph.
nodes_for_graph(Name, Nodes) :-
    graph_spec(Name, Config),
    (   member(nodes(Nodes), Config)
    ->  true
    ;   all_nodes(Nodes)
    ).

%% edges_for_graph(+Name, -Edges)
%  Get edges for a specific graph (only between nodes in the graph).
edges_for_graph(Name, Edges) :-
    nodes_for_graph(Name, Nodes),
    findall(edge(From, To, Props), (
        graph_edge(From, To, Props),
        member(From, Nodes),
        member(To, Nodes)
    ), Edges).

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

%% generate_graph_component(+Name, -Code)
%  Generate a React component for the graph.
generate_graph_component(Name, Code) :-
    generate_graph_component(Name, [], Code).

%% generate_graph_component(+Name, +Options, -Code)
%  Generate a React component with options.
generate_graph_component(Name, _Options, Code) :-
    graph_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Graph"),
    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Generate data
    generate_graph_data(Name, DataCode),

    % Generate Cytoscape config
    generate_cytoscape_config(Name, CytoscapeConfig),

    format(atom(Code),
'// Generated by UnifyWeaver - Graph Visualization Component
// Graph: ~w

import React, { useEffect, useRef, useState } from "react";
import cytoscape, { Core } from "cytoscape";
import styles from "./~w.module.css";

interface ~wProps {
  onNodeClick?: (nodeId: string) => void;
  onEdgeClick?: (edgeId: string) => void;
  className?: string;
}

~w

export const ~w: React.FC<~wProps> = ({
  onNodeClick,
  onEdgeClick,
  className = ""
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements: graphData,
      ~w
    });

    cyRef.current = cy;
    setNodeCount(cy.nodes().length);
    setEdgeCount(cy.edges().length);

    // Event handlers
    cy.on("tap", "node", (evt) => {
      const nodeId = evt.target.id();
      setSelectedNode(nodeId);
      onNodeClick?.(nodeId);
    });

    cy.on("tap", "edge", (evt) => {
      const edgeId = evt.target.id();
      onEdgeClick?.(edgeId);
    });

    return () => {
      cy.destroy();
    };
  }, [onNodeClick, onEdgeClick]);

  const handleLayout = (layoutName: string) => {
    cyRef.current?.layout({ name: layoutName, animate: true }).run();
  };

  const handleFit = () => {
    cyRef.current?.fit();
  };

  const handleCenter = () => {
    if (selectedNode && cyRef.current) {
      cyRef.current.center(cyRef.current.$(`#${selectedNode}`));
    }
  };

  return (
    <div className={`${styles.container} ${className}`}>
      <div className={styles.header}>
        <h2 className={styles.title}>~w</h2>
        <div className={styles.stats}>
          <span>{nodeCount} nodes</span>
          <span>{edgeCount} edges</span>
        </div>
      </div>

      <div className={styles.toolbar}>
        <button onClick={() => handleLayout("cose")} className={styles.button}>
          Force Layout
        </button>
        <button onClick={() => handleLayout("circle")} className={styles.button}>
          Circle
        </button>
        <button onClick={() => handleLayout("grid")} className={styles.button}>
          Grid
        </button>
        <button onClick={handleFit} className={styles.button}>
          Fit
        </button>
        {selectedNode && (
          <button onClick={handleCenter} className={styles.button}>
            Center on {selectedNode}
          </button>
        )}
      </div>

      <div ref={containerRef} className={styles.graph} />

      {selectedNode && (
        <div className={styles.infoPanel}>
          <h3>Selected: {selectedNode}</h3>
        </div>
      )}
    </div>
  );
};

export default ~w;
', [Title, ComponentName, ComponentName, DataCode, ComponentName, ComponentName, CytoscapeConfig, Title, ComponentName]).

%% generate_graph_data(+Name, -DataCode)
%  Generate the graph data as TypeScript.
generate_graph_data(Name, DataCode) :-
    nodes_for_graph(Name, NodeIds),
    edges_for_graph(Name, Edges),

    % Generate node data
    findall(NodeCode, (
        member(NodeId, NodeIds),
        graph_node(NodeId, Props),
        (member(label(Label), Props) -> true ; atom_string(NodeId, Label)),
        atom_string(NodeId, IdStr),
        format(atom(NodeCode), '    { data: { id: "~w", label: "~w" } }', [IdStr, Label])
    ), NodeCodes),
    atomic_list_concat(NodeCodes, ',\n', NodesSection),

    % Generate edge data
    findall(EdgeCode, (
        member(edge(From, To, _Props), Edges),
        atom_string(From, FromStr),
        atom_string(To, ToStr),
        format(atom(EdgeId), '~w_~w', [FromStr, ToStr]),
        format(atom(EdgeCode), '    { data: { id: "~w", source: "~w", target: "~w" } }', [EdgeId, FromStr, ToStr])
    ), EdgeCodes),
    atomic_list_concat(EdgeCodes, ',\n', EdgesSection),

    format(atom(DataCode),
'const graphData = [
  // Nodes
~w,
  // Edges
~w
];', [NodesSection, EdgesSection]).

%% generate_cytoscape_config(+Name, -ConfigCode)
%  Generate Cytoscape.js configuration.
generate_cytoscape_config(Name, ConfigCode) :-
    graph_spec(Name, Config),
    (member(layout(Layout), Config) -> true ; Layout = cose),
    (member(theme(Theme), Config) -> true ; Theme = dark),

    theme_colors(Theme, NodeColor, EdgeColor, TextColor),

    format(atom(ConfigCode),
'style: [
        {
          selector: "node",
          style: {
            "background-color": "~w",
            "label": "data(label)",
            "color": "~w",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "12px",
            "width": "50px",
            "height": "50px",
            "border-width": "2px",
            "border-color": "~w"
          }
        },
        {
          selector: "edge",
          style: {
            "width": 2,
            "line-color": "~w",
            "target-arrow-color": "~w",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier"
          }
        },
        {
          selector: "node:selected",
          style: {
            "border-width": "4px",
            "border-color": "#ffd700"
          }
        }
      ],
      layout: { name: "~w", animate: true }', [NodeColor, TextColor, NodeColor, EdgeColor, EdgeColor, Layout]).

%% theme_colors(+Theme, -NodeColor, -EdgeColor, -TextColor)
theme_colors(dark, '#7c3aed', '#00d4ff', '#ffffff').
theme_colors(light, '#6366f1', '#0ea5e9', '#1f2937').
theme_colors(nature, '#22c55e', '#84cc16', '#ffffff').
theme_colors(warm, '#f97316', '#eab308', '#ffffff').
theme_colors(_, '#7c3aed', '#00d4ff', '#ffffff').  % Default to dark

% ============================================================================
% CODE GENERATION - CSS STYLES
% ============================================================================

%% generate_graph_styles(+Name, -CssCode)
%  Generate CSS module for the graph component.
generate_graph_styles(Name, CssCode) :-
    graph_spec(Name, Config),
    (member(theme(Theme), Config) -> true ; Theme = dark),
    theme_css(Theme, BgColor, BorderColor),

    format(atom(CssCode),
'.container {
  display: flex;
  flex-direction: column;
  height: 100%%;
  background: ~w;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid ~w;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid ~w;
}

.title {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
}

.stats {
  display: flex;
  gap: 1rem;
  font-size: 0.875rem;
  opacity: 0.7;
}

.toolbar {
  display: flex;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid ~w;
  flex-wrap: wrap;
}

.button {
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, #7c3aed, #6366f1);
  border: none;
  border-radius: 6px;
  color: white;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.1s, opacity 0.2s;
}

.button:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}

.button:active {
  transform: translateY(0);
}

.graph {
  flex: 1;
  min-height: 400px;
}

.infoPanel {
  padding: 1rem;
  border-top: 1px solid ~w;
  background: rgba(0, 0, 0, 0.2);
}

.infoPanel h3 {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 500;
}
', [BgColor, BorderColor, BorderColor, BorderColor, BorderColor]).

%% theme_css(+Theme, -BgColor, -BorderColor)
theme_css(dark, 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', 'rgba(255, 255, 255, 0.1)').
theme_css(light, '#ffffff', 'rgba(0, 0, 0, 0.1)').
theme_css(_, 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', 'rgba(255, 255, 255, 0.1)').

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% pascal_case(+SnakeCase, -PascalCase)
%  Convert snake_case to PascalCase.
pascal_case(Input, Output) :-
    atom_string(Input, InputStr),
    split_string(InputStr, "_", "", Parts),
    maplist(capitalize_first, Parts, CapParts),
    atomics_to_string(CapParts, OutputStr),
    atom_string(Output, OutputStr).

capitalize_first(Str, Cap) :-
    string_chars(Str, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HUA, [HU]),
    atom_string(HUA, HUS),
    string_chars(Rest, T),
    string_concat(HUS, Rest, Cap).

% ============================================================================
% TESTING
% ============================================================================

test_graph_generator :-
    format('~n=== Graph Generator Tests ===~n~n'),

    % Test node queries
    format('Test 1: Node queries~n'),
    all_nodes(Nodes),
    length(Nodes, NodeCount),
    (NodeCount > 5
    ->  format('  [PASS] Has ~w nodes~n', [NodeCount])
    ;   format('  [FAIL] Expected >5 nodes, got ~w~n', [NodeCount])
    ),

    % Test edge queries
    format('Test 2: Edge queries~n'),
    all_edges(Edges),
    length(Edges, EdgeCount),
    (EdgeCount > 5
    ->  format('  [PASS] Has ~w edges~n', [EdgeCount])
    ;   format('  [FAIL] Expected >5 edges, got ~w~n', [EdgeCount])
    ),

    % Test graph spec query
    format('Test 3: Graph spec query~n'),
    (graph_spec(family_tree, _)
    ->  format('  [PASS] family_tree spec exists~n')
    ;   format('  [FAIL] family_tree spec not found~n')
    ),

    % Test component generation
    format('Test 4: Component generation~n'),
    generate_graph_component(family_tree, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 3000
    ->  format('  [PASS] Generated ~w chars~n', [CodeLen])
    ;   format('  [FAIL] Code too short: ~w~n', [CodeLen])
    ),

    % Test data generation
    format('Test 5: Data generation~n'),
    generate_graph_data(family_tree, DataCode),
    (sub_atom(DataCode, _, _, _, 'graphData')
    ->  format('  [PASS] Contains graphData~n')
    ;   format('  [FAIL] Missing graphData~n')
    ),

    % Test CSS generation
    format('Test 6: CSS generation~n'),
    generate_graph_styles(family_tree, CssCode),
    (sub_atom(CssCode, _, _, _, '.container')
    ->  format('  [PASS] Contains .container class~n')
    ;   format('  [FAIL] Missing .container class~n')
    ),

    format('~n=== Tests Complete ===~n').
