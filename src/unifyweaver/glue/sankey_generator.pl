% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Sankey Diagram Generator - Declarative Flow Visualization
%
% This module provides declarative Sankey diagram definitions that generate
% TypeScript/React components for flow and allocation visualization.
%
% Usage:
%   % Define Sankey nodes
%   sankey_node(my_sankey, source_a, [label("Source A"), column(0)]).
%   sankey_node(my_sankey, target_b, [label("Target B"), column(1)]).
%
%   % Define flows between nodes
%   sankey_flow(my_sankey, source_a, target_b, 100).
%
%   % Generate React component
%   ?- generate_sankey_component(my_sankey, Code).

:- module(sankey_generator, [
    % Sankey diagram definition predicates
    sankey_spec/2,                      % sankey_spec(+Name, +Config)
    sankey_node/3,                      % sankey_node(+Name, +NodeId, +Config)
    sankey_flow/4,                      % sankey_flow(+Name, +Source, +Target, +Value)

    % Sankey management
    declare_sankey_spec/2,
    declare_sankey_node/3,
    declare_sankey_flow/4,
    clear_sankey/0,
    clear_sankey/1,

    % Query predicates
    all_sankeys/1,
    sankey_nodes/2,                     % sankey_nodes(+Name, -Nodes)
    sankey_flows/2,                     % sankey_flows(+Name, -Flows)
    node_inflow/3,                      % node_inflow(+Name, +NodeId, -Total)
    node_outflow/3,                     % node_outflow(+Name, +NodeId, -Total)

    % Code generation
    generate_sankey_component/2,
    generate_sankey_component/3,
    generate_sankey_styles/2,
    generate_sankey_data/2,

    % Python generation
    generate_sankey_matplotlib/2,
    generate_sankey_plotly/2,

    % Testing
    test_sankey_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic sankey_spec/2.
:- dynamic sankey_node/3.
:- dynamic sankey_flow/4.

:- discontiguous sankey_spec/2.
:- discontiguous sankey_node/3.
:- discontiguous sankey_flow/4.

% ============================================================================
% DEFAULT SANKEY DEFINITIONS
% ============================================================================

% Energy flow example
sankey_spec(energy_flow, [
    title("Energy Flow"),
    width(800),
    height(400),
    node_width(20),
    node_padding(10),
    link_opacity(0.5),
    theme(dark)
]).

% Source nodes (column 0)
sankey_node(energy_flow, coal, [label("Coal"), column(0), color('#6b7280')]).
sankey_node(energy_flow, natural_gas, [label("Natural Gas"), column(0), color('#3b82f6')]).
sankey_node(energy_flow, nuclear, [label("Nuclear"), column(0), color('#a855f7')]).
sankey_node(energy_flow, renewables, [label("Renewables"), column(0), color('#22c55e')]).

% Intermediate nodes (column 1)
sankey_node(energy_flow, electricity, [label("Electricity"), column(1), color('#f59e0b')]).
sankey_node(energy_flow, direct_heat, [label("Direct Heat"), column(1), color('#ef4444')]).

% End use nodes (column 2)
sankey_node(energy_flow, residential, [label("Residential"), column(2), color('#8b5cf6')]).
sankey_node(energy_flow, commercial, [label("Commercial"), column(2), color('#06b6d4')]).
sankey_node(energy_flow, industrial, [label("Industrial"), column(2), color('#84cc16')]).
sankey_node(energy_flow, transport, [label("Transport"), column(2), color('#f97316')]).

% Flows (source -> target, value)
sankey_flow(energy_flow, coal, electricity, 200).
sankey_flow(energy_flow, coal, direct_heat, 50).
sankey_flow(energy_flow, natural_gas, electricity, 150).
sankey_flow(energy_flow, natural_gas, direct_heat, 100).
sankey_flow(energy_flow, nuclear, electricity, 180).
sankey_flow(energy_flow, renewables, electricity, 120).

sankey_flow(energy_flow, electricity, residential, 200).
sankey_flow(energy_flow, electricity, commercial, 180).
sankey_flow(energy_flow, electricity, industrial, 220).
sankey_flow(energy_flow, electricity, transport, 50).
sankey_flow(energy_flow, direct_heat, residential, 80).
sankey_flow(energy_flow, direct_heat, industrial, 70).

% Website traffic flow example
sankey_spec(traffic_flow, [
    title("Website Traffic Flow"),
    width(700),
    height(350),
    node_width(15),
    node_padding(8),
    theme(dark)
]).

sankey_node(traffic_flow, organic, [label("Organic Search"), column(0), color('#22c55e')]).
sankey_node(traffic_flow, paid, [label("Paid Ads"), column(0), color('#f59e0b')]).
sankey_node(traffic_flow, social, [label("Social Media"), column(0), color('#3b82f6')]).
sankey_node(traffic_flow, direct, [label("Direct"), column(0), color('#8b5cf6')]).

sankey_node(traffic_flow, landing, [label("Landing Page"), column(1), color('#64748b')]).
sankey_node(traffic_flow, product, [label("Product Page"), column(1), color('#64748b')]).

sankey_node(traffic_flow, signup, [label("Sign Up"), column(2), color('#22c55e')]).
sankey_node(traffic_flow, bounce, [label("Bounce"), column(2), color('#ef4444')]).

sankey_flow(traffic_flow, organic, landing, 5000).
sankey_flow(traffic_flow, organic, product, 3000).
sankey_flow(traffic_flow, paid, landing, 4000).
sankey_flow(traffic_flow, paid, product, 2000).
sankey_flow(traffic_flow, social, landing, 2000).
sankey_flow(traffic_flow, direct, product, 1500).

sankey_flow(traffic_flow, landing, signup, 3000).
sankey_flow(traffic_flow, landing, bounce, 8000).
sankey_flow(traffic_flow, product, signup, 2500).
sankey_flow(traffic_flow, product, bounce, 4000).

% ============================================================================
% SANKEY MANAGEMENT
% ============================================================================

declare_sankey_spec(Name, Config) :-
    retractall(sankey_spec(Name, _)),
    assertz(sankey_spec(Name, Config)).

declare_sankey_node(Name, NodeId, Config) :-
    assertz(sankey_node(Name, NodeId, Config)).

declare_sankey_flow(Name, Source, Target, Value) :-
    assertz(sankey_flow(Name, Source, Target, Value)).

clear_sankey :-
    retractall(sankey_spec(_, _)),
    retractall(sankey_node(_, _, _)),
    retractall(sankey_flow(_, _, _, _)).

clear_sankey(Name) :-
    retractall(sankey_spec(Name, _)),
    retractall(sankey_node(Name, _, _)),
    retractall(sankey_flow(Name, _, _, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

all_sankeys(Names) :-
    findall(Name, sankey_spec(Name, _), Names).

sankey_nodes(Name, Nodes) :-
    findall(NodeId, sankey_node(Name, NodeId, _), Nodes).

sankey_flows(Name, Flows) :-
    findall(flow(Source, Target, Value), sankey_flow(Name, Source, Target, Value), Flows).

% Calculate total inflow to a node
node_inflow(Name, NodeId, Total) :-
    findall(Value, sankey_flow(Name, _, NodeId, Value), Values),
    sum_list(Values, Total).

% Calculate total outflow from a node
node_outflow(Name, NodeId, Total) :-
    findall(Value, sankey_flow(Name, NodeId, _, Value), Values),
    sum_list(Values, Total).

sum_list([], 0).
sum_list([H|T], Sum) :-
    sum_list(T, Rest),
    Sum is H + Rest.

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

generate_sankey_component(Name, Code) :-
    generate_sankey_component(Name, [], Code).

generate_sankey_component(Name, _Options, Code) :-
    sankey_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Sankey Diagram"),
    (member(width(Width), Config) -> true ; Width = 800),
    (member(height(Height), Config) -> true ; Height = 400),
    (member(node_width(NodeWidth), Config) -> true ; NodeWidth = 20),
    (member(node_padding(NodePadding), Config) -> true ; NodePadding = 10),
    (member(link_opacity(LinkOpacity), Config) -> true ; LinkOpacity = 0.5),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Generate nodes and links data
    generate_nodes_data(Name, NodesDataJS),
    generate_links_data(Name, LinksDataJS),

    format(atom(Code),
'// Generated by UnifyWeaver - Sankey Diagram Component
// Chart: ~w

import React, { useMemo } from "react";
import styles from "./~w.module.css";

interface SankeyProps {
  onNodeClick?: (nodeId: string) => void;
  onLinkClick?: (source: string, target: string, value: number) => void;
}

~w

~w

interface NodePosition {
  id: string;
  label: string;
  color: string;
  column: number;
  x: number;
  y: number;
  height: number;
}

export const ~w: React.FC<SankeyProps> = ({ onNodeClick, onLinkClick }) => {
  const width = ~w;
  const height = ~w;
  const nodeWidth = ~w;
  const nodePadding = ~w;
  const linkOpacity = ~w;
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Calculate node positions
  const nodePositions = useMemo(() => {
    // Group nodes by column
    const columns: Map<number, typeof nodes> = new Map();
    nodes.forEach((node) => {
      const col = columns.get(node.column) || [];
      col.push(node);
      columns.set(node.column, col);
    });

    const numColumns = Math.max(...nodes.map((n) => n.column)) + 1;
    const columnWidth = (innerWidth - nodeWidth) / (numColumns - 1 || 1);

    // Calculate node heights based on total flow
    const nodeFlows: Map<string, number> = new Map();
    nodes.forEach((node) => {
      const inflow = links
        .filter((l) => l.target === node.id)
        .reduce((sum, l) => sum + l.value, 0);
      const outflow = links
        .filter((l) => l.source === node.id)
        .reduce((sum, l) => sum + l.value, 0);
      nodeFlows.set(node.id, Math.max(inflow, outflow, 1));
    });

    const maxFlow = Math.max(...nodeFlows.values());

    const positions: NodePosition[] = [];

    columns.forEach((colNodes, colIndex) => {
      const x = colIndex * columnWidth;
      const totalHeight = colNodes.reduce(
        (sum, n) => sum + (nodeFlows.get(n.id) || 1) / maxFlow * (innerHeight * 0.8),
        0
      );
      const totalPadding = (colNodes.length - 1) * nodePadding;
      const startY = (innerHeight - totalHeight - totalPadding) / 2;

      let currentY = startY;
      colNodes.forEach((node) => {
        const nodeHeight = (nodeFlows.get(node.id) || 1) / maxFlow * (innerHeight * 0.8);
        positions.push({
          id: node.id,
          label: node.label,
          color: node.color,
          column: node.column,
          x,
          y: currentY,
          height: nodeHeight,
        });
        currentY += nodeHeight + nodePadding;
      });
    });

    return positions;
  }, [innerWidth, innerHeight, nodePadding]);

  // Generate link paths
  const linkPaths = useMemo(() => {
    const nodeMap = new Map(nodePositions.map((n) => [n.id, n]));

    // Track vertical offsets for stacking links
    const sourceOffsets: Map<string, number> = new Map();
    const targetOffsets: Map<string, number> = new Map();

    return links.map((link) => {
      const source = nodeMap.get(link.source)!;
      const target = nodeMap.get(link.target)!;

      if (!source || !target) return null;

      // Calculate link thickness proportional to value
      const maxValue = Math.max(...links.map((l) => l.value));
      const thickness = (link.value / maxValue) * (innerHeight * 0.15);

      // Get current offsets
      const sourceOffset = sourceOffsets.get(link.source) || 0;
      const targetOffset = targetOffsets.get(link.target) || 0;

      // Calculate path
      const x0 = source.x + nodeWidth;
      const y0 = source.y + sourceOffset + thickness / 2;
      const x1 = target.x;
      const y1 = target.y + targetOffset + thickness / 2;
      const curvature = 0.5;
      const xi = (x0 + x1) * curvature;

      const path = `M${x0},${y0} C${xi},${y0} ${xi},${y1} ${x1},${y1}`;

      // Update offsets
      sourceOffsets.set(link.source, sourceOffset + thickness);
      targetOffsets.set(link.target, targetOffset + thickness);

      return {
        ...link,
        path,
        thickness,
        color: source.color,
      };
    }).filter(Boolean);
  }, [nodePositions, innerHeight]);

  return (
    <div className={styles.sankeyContainer}>
      <h3 className={styles.title}>~w</h3>
      <svg width={width} height={height} className={styles.sankeySvg}>
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Links */}
          {linkPaths.map((link, i) => link && (
            <path
              key={i}
              d={link.path}
              fill="none"
              stroke={link.color}
              strokeWidth={link.thickness}
              strokeOpacity={linkOpacity}
              className={styles.link}
              onClick={() => onLinkClick?.(link.source, link.target, link.value)}
            >
              <title>{`${link.source} → ${link.target}: ${link.value}`}</title>
            </path>
          ))}

          {/* Nodes */}
          {nodePositions.map((node) => (
            <g key={node.id} className={styles.node}>
              <rect
                x={node.x}
                y={node.y}
                width={nodeWidth}
                height={node.height}
                fill={node.color}
                onClick={() => onNodeClick?.(node.id)}
              />
              <text
                x={node.column === 0 ? node.x - 5 : node.x + nodeWidth + 5}
                y={node.y + node.height / 2}
                textAnchor={node.column === 0 ? "end" : "start"}
                dominantBaseline="middle"
                className={styles.nodeLabel}
              >
                {node.label}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
};

export default ~w;
', [Name, ComponentName, NodesDataJS, LinksDataJS, ComponentName,
    Width, Height, NodeWidth, NodePadding, LinkOpacity, Title, ComponentName]).

% Generate nodes data array
generate_nodes_data(Name, JS) :-
    findall(NodeJS, (
        sankey_node(Name, NodeId, Config),
        (member(label(Label), Config) -> true ; atom_string(NodeId, Label)),
        (member(column(Column), Config) -> true ; Column = 0),
        (member(color(Color), Config) -> true ; Color = '#6b7280'),
        format(atom(NodeJS), '  { id: "~w", label: "~w", column: ~w, color: "~w" }',
               [NodeId, Label, Column, Color])
    ), NodeJSList),
    atomic_list_concat(NodeJSList, ',\n', NodesStr),
    format(atom(JS), 'const nodes = [\n~w\n];', [NodesStr]).

% Generate links data array
generate_links_data(Name, JS) :-
    findall(LinkJS, (
        sankey_flow(Name, Source, Target, Value),
        format(atom(LinkJS), '  { source: "~w", target: "~w", value: ~w }',
               [Source, Target, Value])
    ), LinkJSList),
    atomic_list_concat(LinkJSList, ',\n', LinksStr),
    format(atom(JS), 'const links = [\n~w\n];', [LinksStr]).

% ============================================================================
% CSS GENERATION
% ============================================================================

generate_sankey_styles(_Name, CSS) :-
    CSS = '.sankeyContainer {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem;
  background: var(--surface, #16213e);
  border-radius: 12px;
}

.title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text, #e0e0e0);
  margin: 0 0 1rem 0;
}

.sankeySvg {
  overflow: visible;
}

.node rect {
  cursor: pointer;
  transition: opacity 0.2s ease;
}

.node rect:hover {
  opacity: 0.8;
}

.nodeLabel {
  font-size: 0.75rem;
  fill: var(--text, #e0e0e0);
  pointer-events: none;
}

.link {
  cursor: pointer;
  transition: stroke-opacity 0.2s ease;
}

.link:hover {
  stroke-opacity: 0.8 !important;
}
'.

% ============================================================================
% DATA GENERATION
% ============================================================================

generate_sankey_data(Name, DataCode) :-
    generate_nodes_data(Name, NodesJS),
    generate_links_data(Name, LinksJS),
    format(atom(DataCode), '~w~n~n~w', [NodesJS, LinksJS]).

% ============================================================================
% MATPLOTLIB GENERATION
% ============================================================================

generate_sankey_matplotlib(Name, PythonCode) :-
    sankey_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Sankey Diagram"),

    % Get nodes and create index mapping
    sankey_nodes(Name, NodeIds),
    findall(Label, (
        member(NodeId, NodeIds),
        sankey_node(Name, NodeId, NodeConfig),
        (member(label(Label), NodeConfig) -> true ; atom_string(NodeId, Label))
    ), Labels),
    format_python_list(Labels, LabelsPy),

    % Generate source, target, value arrays
    findall(SrcIdx-TgtIdx-Value, (
        sankey_flow(Name, Source, Target, Value),
        nth0(SrcIdx, NodeIds, Source),
        nth0(TgtIdx, NodeIds, Target)
    ), FlowData),
    findall(S, member(S-_-_, FlowData), Sources),
    findall(T, member(_-T-_, FlowData), Targets),
    findall(V, member(_-_-V, FlowData), Values),
    format_python_number_list(Sources, SourcesPy),
    format_python_number_list(Targets, TargetsPy),
    format_python_number_list(Values, ValuesPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Sankey Diagram (matplotlib)
# Chart: ~w

import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

def plot_~w():
    """~w - Note: matplotlib Sankey is limited, consider using Plotly"""
    labels = ~w
    sources = ~w
    targets = ~w
    values = ~w

    # matplotlib Sankey is basic - showing as text representation
    print("Sankey Diagram: ~w")
    print("=" * 50)
    for i, (s, t, v) in enumerate(zip(sources, targets, values)):
        print(f"{labels[s]} --({v})--> {labels[t]}")

    print("\\nNote: For interactive Sankey, use generate_sankey_plotly/2")

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, LabelsPy, SourcesPy, TargetsPy, ValuesPy, Title, Name]).

% ============================================================================
% PLOTLY GENERATION
% ============================================================================

generate_sankey_plotly(Name, PythonCode) :-
    sankey_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Sankey Diagram"),

    % Get nodes and create index mapping
    sankey_nodes(Name, NodeIds),
    findall(Label, (
        member(NodeId, NodeIds),
        sankey_node(Name, NodeId, NodeConfig),
        (member(label(Label), NodeConfig) -> true ; atom_string(NodeId, Label))
    ), Labels),
    findall(Color, (
        member(NodeId, NodeIds),
        sankey_node(Name, NodeId, NodeConfig),
        (member(color(Color), NodeConfig) -> true ; Color = '#6b7280')
    ), Colors),
    format_python_list(Labels, LabelsPy),
    format_python_list(Colors, ColorsPy),

    % Generate source, target, value arrays
    findall(SrcIdx-TgtIdx-Value, (
        sankey_flow(Name, Source, Target, Value),
        nth0(SrcIdx, NodeIds, Source),
        nth0(TgtIdx, NodeIds, Target)
    ), FlowData),
    findall(S, member(S-_-_, FlowData), Sources),
    findall(T, member(_-T-_, FlowData), Targets),
    findall(V, member(_-_-V, FlowData), Values),
    format_python_number_list(Sources, SourcesPy),
    format_python_number_list(Targets, TargetsPy),
    format_python_number_list(Values, ValuesPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Sankey Diagram (Plotly)
# Chart: ~w

import plotly.graph_objects as go

def plot_~w():
    """~w"""
    labels = ~w
    colors = ~w

    sources = ~w
    targets = ~w
    values = ~w

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text="~w",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        font=dict(size=12, color="#e0e0e0")
    )

    fig.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, LabelsPy, ColorsPy, SourcesPy, TargetsPy, ValuesPy, Title, Name]).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

format_python_list([], '[]').
format_python_list(List, PyList) :-
    List \= [],
    findall(Q, (member(I, List), format(atom(Q), '"~w"', [I])), Qs),
    atomic_list_concat(Qs, ', ', S),
    format(atom(PyList), '[~w]', [S]).

format_python_number_list(Numbers, PyList) :-
    atomic_list_concat_numbers(Numbers, ', ', S),
    format(atom(PyList), '[~w]', [S]).

atomic_list_concat_numbers([], _, '').
atomic_list_concat_numbers([X], _, S) :- format(atom(S), '~w', [X]).
atomic_list_concat_numbers([X|Xs], Sep, S) :-
    Xs \= [],
    atomic_list_concat_numbers(Xs, Sep, Rest),
    format(atom(S), '~w~w~w', [X, Sep, Rest]).

pascal_case(String, PascalCase) :-
    atom_string(Atom, String),
    atom_codes(Atom, Codes),
    pascal_codes(Codes, true, PascalCodes),
    atom_codes(PascalCase, PascalCodes).

pascal_codes([], _, []).
pascal_codes([C|Cs], true, [Upper|Rest]) :-
    C >= 0'a, C =< 0'z, !,
    Upper is C - 32,
    pascal_codes(Cs, false, Rest).
pascal_codes([C|Cs], true, [C|Rest]) :-
    pascal_codes(Cs, false, Rest).
pascal_codes([C|Cs], _, Rest) :-
    (C = 0'_ ; C = 0'-), !,
    pascal_codes(Cs, true, Rest).
pascal_codes([C|Cs], _, [C|Rest]) :-
    pascal_codes(Cs, false, Rest).

% ============================================================================
% TESTS
% ============================================================================

test_sankey_generator :-
    format('Testing sankey_generator module...~n~n'),

    format('Test 1: Sankey spec query~n'),
    (sankey_spec(energy_flow, _)
    ->  format('  PASS: energy_flow spec exists~n')
    ;   format('  FAIL: energy_flow spec not found~n')
    ),

    format('~nTest 2: Nodes query~n'),
    sankey_nodes(energy_flow, Nodes),
    length(Nodes, NumNodes),
    (NumNodes =:= 10
    ->  format('  PASS: 10 nodes found~n')
    ;   format('  FAIL: Expected 10 nodes, got ~w~n', [NumNodes])
    ),

    format('~nTest 3: Flows query~n'),
    sankey_flows(energy_flow, Flows),
    length(Flows, NumFlows),
    (NumFlows =:= 12
    ->  format('  PASS: 12 flows found~n')
    ;   format('  FAIL: Expected 12 flows, got ~w~n', [NumFlows])
    ),

    format('~nTest 4: Node inflow~n'),
    node_inflow(energy_flow, electricity, Inflow),
    (Inflow =:= 650
    ->  format('  PASS: Electricity inflow is 650~n')
    ;   format('  FAIL: Expected 650, got ~w~n', [Inflow])
    ),

    format('~nTest 5: Node outflow~n'),
    node_outflow(energy_flow, electricity, Outflow),
    (Outflow =:= 650
    ->  format('  PASS: Electricity outflow is 650~n')
    ;   format('  FAIL: Expected 650, got ~w~n', [Outflow])
    ),

    format('~nTest 6: Component generation~n'),
    generate_sankey_component(energy_flow, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 3000
    ->  format('  PASS: Generated ~w chars~n', [CodeLen])
    ;   format('  FAIL: Code too short: ~w~n', [CodeLen])
    ),

    format('~nTest 7: Plotly generation~n'),
    generate_sankey_plotly(energy_flow, PyCode),
    (sub_atom(PyCode, _, _, _, 'go.Sankey')
    ->  format('  PASS: Contains Plotly Sankey~n')
    ;   format('  FAIL: Missing Plotly Sankey~n')
    ),

    format('~nAll tests completed.~n').
