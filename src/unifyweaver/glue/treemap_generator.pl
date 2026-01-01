% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Treemap Generator - Declarative Hierarchical Visualization
%
% This module provides declarative treemap definitions that generate
% TypeScript/React components for hierarchical data visualization.
%
% Usage:
%   % Define treemap nodes
%   treemap_node(root, null, "Root", 0).
%   treemap_node(child1, root, "Child 1", 100).
%   treemap_node(child2, root, "Child 2", 200).
%
%   % Define treemap configuration
%   treemap_spec(my_treemap, [
%       title("File Sizes"),
%       root(root),
%       color_by(depth),
%       show_labels(true)
%   ]).
%
%   % Generate React component
%   ?- generate_treemap_component(my_treemap, Code).

:- module(treemap_generator, [
    % Treemap definition predicates
    treemap_node/4,                     % treemap_node(+Id, +Parent, +Label, +Value)
    treemap_spec/2,                     % treemap_spec(+Name, +Config)

    % Treemap management
    declare_treemap_node/4,             % declare_treemap_node(+Id, +Parent, +Label, +Value)
    declare_treemap_spec/2,             % declare_treemap_spec(+Name, +Config)
    clear_treemap/0,                    % clear_treemap
    clear_treemap/1,                    % clear_treemap(+Name)

    % Query predicates
    all_treemaps/1,                     % all_treemaps(-Names)
    treemap_children/3,                 % treemap_children(+Name, +Parent, -Children)
    treemap_total_value/2,              % treemap_total_value(+Name, -Total)
    treemap_depth/3,                    % treemap_depth(+Name, +NodeId, -Depth)

    % Code generation
    generate_treemap_component/2,       % generate_treemap_component(+Name, -Code)
    generate_treemap_component/3,       % generate_treemap_component(+Name, +Options, -Code)
    generate_treemap_styles/2,          % generate_treemap_styles(+Name, -CssCode)
    generate_treemap_data/2,            % generate_treemap_data(+Name, -DataCode)

    % Python/plotly generation
    generate_treemap_plotly/2,          % generate_treemap_plotly(+Name, -PythonCode)

    % Layout-integrated generation
    generate_treemap_with_layout/3,     % generate_treemap_with_layout(+Name, +LayoutPattern, -Code)

    % Testing
    test_treemap_generator/0
]).

:- use_module(library(lists)).
:- use_module(layout_generator).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic treemap_node/4.
:- dynamic treemap_spec/2.

:- discontiguous treemap_node/4.
:- discontiguous treemap_spec/2.

% ============================================================================
% DEFAULT TREEMAP DEFINITIONS - File System Example
% ============================================================================

% File system treemap
treemap_spec(filesystem_demo, [
    title("Project File Sizes"),
    root(project_root),
    color_by(category),
    show_labels(true),
    theme(dark)
]).

% Root node (value 0 for containers)
treemap_node(project_root, null, "Project", 0).

% Source directory
treemap_node(src, project_root, "src", 0).
treemap_node(src_main, src, "main.ts", 150).
treemap_node(src_utils, src, "utils.ts", 80).
treemap_node(src_types, src, "types.ts", 45).

% Components directory
treemap_node(components, project_root, "components", 0).
treemap_node(comp_header, components, "Header.tsx", 120).
treemap_node(comp_footer, components, "Footer.tsx", 90).
treemap_node(comp_sidebar, components, "Sidebar.tsx", 200).

% Tests directory
treemap_node(tests, project_root, "tests", 0).
treemap_node(test_main, tests, "main.test.ts", 100).
treemap_node(test_utils, tests, "utils.test.ts", 60).

% Budget treemap example
treemap_spec(budget_demo, [
    title("Monthly Budget"),
    root(budget_root),
    color_by(value),
    show_labels(true),
    value_format(currency),
    theme(dark)
]).

treemap_node(budget_root, null, "Budget", 0).
treemap_node(housing, budget_root, "Housing", 0).
treemap_node(rent, housing, "Rent", 1500).
treemap_node(utilities, housing, "Utilities", 200).
treemap_node(insurance, housing, "Insurance", 100).
treemap_node(food, budget_root, "Food", 0).
treemap_node(groceries, food, "Groceries", 400).
treemap_node(dining, food, "Dining Out", 200).
treemap_node(transport, budget_root, "Transport", 0).
treemap_node(gas, transport, "Gas", 150).
treemap_node(maintenance, transport, "Maintenance", 50).

% ============================================================================
% TREEMAP MANAGEMENT
% ============================================================================

%% declare_treemap_node(+Id, +Parent, +Label, +Value)
%  Dynamically add a treemap node.
declare_treemap_node(Id, Parent, Label, Value) :-
    assertz(treemap_node(Id, Parent, Label, Value)).

%% declare_treemap_spec(+Name, +Config)
%  Dynamically add a treemap specification.
declare_treemap_spec(Name, Config) :-
    retractall(treemap_spec(Name, _)),
    assertz(treemap_spec(Name, Config)).

%% clear_treemap
%  Clear all treemap definitions.
clear_treemap :-
    retractall(treemap_node(_, _, _, _)),
    retractall(treemap_spec(_, _)).

%% clear_treemap(+Name)
%  Clear nodes associated with a specific treemap.
clear_treemap(Name) :-
    treemap_spec(Name, Config),
    member(root(Root), Config),
    clear_subtree(Root),
    retractall(treemap_spec(Name, _)).

clear_subtree(NodeId) :-
    findall(Child, treemap_node(Child, NodeId, _, _), Children),
    forall(member(C, Children), clear_subtree(C)),
    retractall(treemap_node(NodeId, _, _, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% all_treemaps(-Names)
%  Get all defined treemap names.
all_treemaps(Names) :-
    findall(Name, treemap_spec(Name, _), Names).

%% treemap_children(+Name, +Parent, -Children)
%  Get direct children of a node.
treemap_children(_Name, Parent, Children) :-
    findall(child(Id, Label, Value),
            treemap_node(Id, Parent, Label, Value),
            Children).

%% treemap_total_value(+Name, -Total)
%  Get total value of all leaf nodes in the treemap.
treemap_total_value(Name, Total) :-
    treemap_spec(Name, Config),
    member(root(Root), Config),
    subtree_value(Root, Total).

subtree_value(NodeId, Value) :-
    treemap_node(NodeId, _, _, NodeValue),
    findall(ChildValue, (
        treemap_node(ChildId, NodeId, _, _),
        subtree_value(ChildId, ChildValue)
    ), ChildValues),
    (ChildValues = []
    ->  Value = NodeValue  % Leaf node
    ;   sum_list(ChildValues, Value)  % Container node
    ).

%% treemap_depth(+Name, +NodeId, -Depth)
%  Get depth of a node in the tree.
treemap_depth(_Name, NodeId, Depth) :-
    node_depth(NodeId, 0, Depth).

node_depth(NodeId, Acc, Depth) :-
    treemap_node(NodeId, Parent, _, _),
    (Parent = null
    ->  Depth = Acc
    ;   NextAcc is Acc + 1,
        node_depth(Parent, NextAcc, Depth)
    ).

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

%% generate_treemap_component(+Name, -Code)
%  Generate a React component for the treemap.
generate_treemap_component(Name, Code) :-
    generate_treemap_component(Name, [], Code).

%% generate_treemap_component(+Name, +Options, -Code)
%  Generate a React component with options.
generate_treemap_component(Name, _Options, Code) :-
    treemap_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Treemap"),
    (member(show_labels(ShowLabels), Config) -> true ; ShowLabels = true),
    (member(color_by(ColorBy), Config) -> true ; ColorBy = depth),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Generate data
    generate_treemap_data(Name, DataCode),

    % Show labels as string
    (ShowLabels == true -> ShowLabelsStr = 'true' ; ShowLabelsStr = 'false'),

    % Color function
    get_color_by_function(ColorBy, ColorFunction),

    format(atom(Code),
'// Generated by UnifyWeaver - Treemap Component
// Treemap: ~w

import React, { useMemo, useCallback } from "react";
import styles from "./~w.module.css";

interface TreemapNode {
  id: string;
  label: string;
  value: number;
  children?: TreemapNode[];
}

interface TreemapProps {
  width?: number;
  height?: number;
  onNodeClick?: (node: TreemapNode) => void;
}

~w

export const ~w: React.FC<TreemapProps> = ({
  width = 800,
  height = 600,
  onNodeClick
}) => {
  const showLabels = ~w;

  // Calculate treemap layout using squarified algorithm
  const calculateLayout = useCallback((
    node: TreemapNode,
    x: number,
    y: number,
    w: number,
    h: number,
    depth: number
  ): LayoutNode[] => {
    const result: LayoutNode[] = [];

    if (!node.children || node.children.length === 0) {
      result.push({ ...node, x, y, w, h, depth });
      return result;
    }

    const total = node.children.reduce((sum, c) => sum + getNodeValue(c), 0);
    if (total === 0) return result;

    let currentX = x;
    let currentY = y;
    const isHorizontal = w >= h;

    node.children.forEach((child) => {
      const ratio = getNodeValue(child) / total;
      let childW: number, childH: number;

      if (isHorizontal) {
        childW = w * ratio;
        childH = h;
      } else {
        childW = w;
        childH = h * ratio;
      }

      result.push(...calculateLayout(child, currentX, currentY, childW, childH, depth + 1));

      if (isHorizontal) {
        currentX += childW;
      } else {
        currentY += childH;
      }
    });

    return result;
  }, []);

  const getNodeValue = (node: TreemapNode): number => {
    if (!node.children || node.children.length === 0) {
      return node.value;
    }
    return node.children.reduce((sum, c) => sum + getNodeValue(c), 0);
  };

  ~w

  interface LayoutNode extends TreemapNode {
    x: number;
    y: number;
    w: number;
    h: number;
    depth: number;
  }

  const layout = useMemo(() => {
    return calculateLayout(data, 0, 0, width, height, 0);
  }, [data, width, height, calculateLayout]);

  return (
    <div className={styles.treemapContainer}>
      <h3 className={styles.title}>~w</h3>
      <svg width={width} height={height} className={styles.treemap}>
        {layout.map((node) => (
          <g key={node.id} onClick={() => onNodeClick?.(node)}>
            <rect
              x={node.x}
              y={node.y}
              width={Math.max(0, node.w - 2)}
              height={Math.max(0, node.h - 2)}
              fill={getColor(node)}
              stroke="var(--border, rgba(255,255,255,0.2))"
              strokeWidth={1}
              className={styles.node}
            />
            {showLabels && node.w > 40 && node.h > 20 && (
              <text
                x={node.x + node.w / 2}
                y={node.y + node.h / 2}
                textAnchor="middle"
                dominantBaseline="middle"
                className={styles.label}
                style={{ fontSize: Math.min(14, node.w / 8) }}
              >
                {node.label}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
};

export default ~w;
', [Name, ComponentName, DataCode, ComponentName, ShowLabelsStr, ColorFunction, Title, ComponentName]).

%% get_color_by_function(+ColorBy, -Code)
get_color_by_function(depth, Code) :-
    Code = 'const getColor = useCallback((node: LayoutNode): string => {
    const colors = ["#7c3aed", "#3b82f6", "#22c55e", "#eab308", "#ef4444"];
    return colors[node.depth % colors.length];
  }, []);'.

get_color_by_function(value, Code) :-
    Code = 'const getColor = useCallback((node: LayoutNode): string => {
    const maxValue = Math.max(...layout.map(n => n.value));
    const ratio = node.value / maxValue;
    const r = Math.round(124 + 131 * ratio);
    const g = Math.round(58 + 72 * (1 - ratio));
    const b = Math.round(237 - 100 * ratio);
    return `rgb(${r}, ${g}, ${b})`;
  }, [layout]);'.

get_color_by_function(category, Code) :-
    Code = 'const getColor = useCallback((node: LayoutNode): string => {
    // Color by parent category
    const hash = node.id.split("").reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 70%, 50%)`;
  }, []);'.

get_color_by_function(_, Code) :-
    get_color_by_function(depth, Code).

% ============================================================================
% TREEMAP DATA GENERATION
% ============================================================================

%% generate_treemap_data(+Name, -DataCode)
%  Generate TypeScript data structure for the treemap.
generate_treemap_data(Name, DataCode) :-
    treemap_spec(Name, Config),
    member(root(Root), Config),
    generate_node_js(Root, NodeJS),
    format(atom(DataCode), 'const data: TreemapNode = ~w;', [NodeJS]).

generate_node_js(NodeId, JS) :-
    treemap_node(NodeId, _, Label, Value),
    findall(ChildJS, (
        treemap_node(ChildId, NodeId, _, _),
        generate_node_js(ChildId, ChildJS)
    ), ChildrenJS),
    (ChildrenJS = []
    ->  format(atom(JS), '{ id: "~w", label: "~w", value: ~w }', [NodeId, Label, Value])
    ;   atomic_list_concat(ChildrenJS, ', ', ChildrenStr),
        format(atom(JS), '{ id: "~w", label: "~w", value: ~w, children: [~w] }',
               [NodeId, Label, Value, ChildrenStr])
    ).

% ============================================================================
% CSS GENERATION
% ============================================================================

%% generate_treemap_styles(+Name, -CssCode)
%  Generate CSS for treemap styling.
generate_treemap_styles(_Name, CssCode) :-
    CssCode = '.treemapContainer {
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

.treemap {
  border: 1px solid var(--border, rgba(255,255,255,0.1));
  border-radius: 8px;
  overflow: hidden;
}

.node {
  cursor: pointer;
  transition: opacity 0.15s ease;
}

.node:hover {
  opacity: 0.8;
}

.label {
  fill: white;
  font-weight: 500;
  pointer-events: none;
  text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}
'.

% ============================================================================
% PLOTLY GENERATION
% ============================================================================

%% generate_treemap_plotly(+Name, -PythonCode)
%  Generate Python/Plotly code for the treemap.
generate_treemap_plotly(Name, PythonCode) :-
    treemap_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Treemap"),
    member(root(Root), Config),

    % Collect all nodes for plotly format
    collect_plotly_data(Root, Ids, Labels, Parents, Values),
    format_python_list(Ids, IdsStr),
    format_python_list(Labels, LabelsStr),
    format_python_list(Parents, ParentsStr),
    format_python_values(Values, ValuesStr),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Treemap
# Treemap: ~w

import plotly.express as px
import pandas as pd

def plot_~w():
    """~w"""
    # Data
    ids = ~w
    labels = ~w
    parents = ~w
    values = ~w

    # Create dataframe
    df = pd.DataFrame({
        "id": ids,
        "label": labels,
        "parent": parents,
        "value": values
    })

    # Create treemap
    fig = px.treemap(
        df,
        ids="id",
        names="label",
        parents="parent",
        values="value",
        title="~w"
    )

    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)
    )

    fig.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, IdsStr, LabelsStr, ParentsStr, ValuesStr, Title, Name]).

collect_plotly_data(Root, Ids, Labels, Parents, Values) :-
    collect_nodes(Root, "", Nodes),
    findall(Id, member(node(Id, _, _, _), Nodes), Ids),
    findall(Label, member(node(_, Label, _, _), Nodes), Labels),
    findall(Parent, member(node(_, _, Parent, _), Nodes), Parents),
    findall(Value, member(node(_, _, _, Value), Nodes), Values).

collect_nodes(NodeId, ParentStr, [node(IdStr, Label, ParentStr, Value)|RestNodes]) :-
    treemap_node(NodeId, _, Label, Value),
    atom_string(NodeId, IdStr),
    findall(ChildNodes, (
        treemap_node(ChildId, NodeId, _, _),
        collect_nodes(ChildId, IdStr, ChildNodes)
    ), ChildNodeLists),
    append(ChildNodeLists, RestNodes).

%% format_python_list(+List, -PythonList)
format_python_list([], '[]').
format_python_list(List, PythonList) :-
    List \= [],
    findall(QuotedItem, (
        member(Item, List),
        format(atom(QuotedItem), '"~w"', [Item])
    ), QuotedItems),
    atomic_list_concat(QuotedItems, ', ', ItemsStr),
    format(atom(PythonList), '[~w]', [ItemsStr]).

%% format_python_values(+List, -PythonList)
format_python_values([], '[]').
format_python_values(List, PythonList) :-
    List \= [],
    findall(Item, member(Item, List), Items),
    atomic_list_concat(Items, ', ', ItemsStr),
    format(atom(PythonList), '[~w]', [ItemsStr]).

% ============================================================================
% LAYOUT INTEGRATION
% ============================================================================

%% generate_treemap_with_layout(+Name, +LayoutPattern, -Code)
%  Generate treemap component with layout wrapper.
generate_treemap_with_layout(Name, LayoutPattern, Code) :-
    generate_treemap_component(Name, ComponentCode),
    generate_treemap_styles(Name, TreemapCSS),
    (has_layout(LayoutPattern)
    ->  generate_layout_css(LayoutPattern, LayoutCSS),
        format(atom(Code), '~w~n~n/* Layout CSS */~n~w~n~n/* Treemap CSS */~n~w', [ComponentCode, LayoutCSS, TreemapCSS])
    ;   format(atom(Code), '~w~n~n/* CSS */~n~w', [ComponentCode, TreemapCSS])
    ).

% ============================================================================
% UTILITY - PASCAL CASE
% ============================================================================

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

test_treemap_generator :-
    format('Testing treemap_generator module...~n~n'),

    % Test treemap spec query
    format('Test 1: Treemap spec query~n'),
    (treemap_spec(filesystem_demo, _)
    ->  format('  PASS: filesystem_demo spec exists~n')
    ;   format('  FAIL: filesystem_demo spec not found~n')
    ),

    % Test children query
    format('~nTest 2: Children query~n'),
    (treemap_children(filesystem_demo, project_root, Children), length(Children, Len), Len >= 3
    ->  format('  PASS: Found ~w children~n', [Len])
    ;   format('  FAIL: Children query failed~n')
    ),

    % Test total value
    format('~nTest 3: Total value~n'),
    (treemap_total_value(filesystem_demo, Total), Total > 0
    ->  format('  PASS: Total value = ~w~n', [Total])
    ;   format('  FAIL: Total value calculation failed~n')
    ),

    % Test component generation
    format('~nTest 4: Component generation~n'),
    generate_treemap_component(filesystem_demo, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 2000
    ->  format('  PASS: Generated ~w chars~n', [CodeLen])
    ;   format('  FAIL: Code too short: ~w~n', [CodeLen])
    ),

    % Test plotly generation
    format('~nTest 5: Plotly generation~n'),
    generate_treemap_plotly(budget_demo, PyCode),
    (sub_atom(PyCode, _, _, _, 'plotly')
    ->  format('  PASS: Contains plotly import~n')
    ;   format('  FAIL: Missing plotly import~n')
    ),

    format('~nAll tests completed.~n').

:- initialization(test_treemap_generator, main).
