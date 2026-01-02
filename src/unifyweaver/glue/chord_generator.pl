% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Chord Diagram Generator - Declarative Relationship Visualization
%
% This module provides declarative chord diagram definitions that generate
% TypeScript/React components for relationship and flow visualization.
%
% Usage:
%   % Define chord entities
%   chord_entity(my_chord, entity_a, [label("Entity A"), color('#3b82f6')]).
%
%   % Define relationships (connections with magnitude)
%   chord_connection(my_chord, entity_a, entity_b, 50).
%
%   % Generate React component
%   ?- generate_chord_component(my_chord, Code).

:- module(chord_generator, [
    % Chord diagram definition predicates
    chord_spec/2,                       % chord_spec(+Name, +Config)
    chord_entity/3,                     % chord_entity(+Name, +EntityId, +Config)
    chord_connection/4,                 % chord_connection(+Name, +Source, +Target, +Value)

    % Chord management
    declare_chord_spec/2,
    declare_chord_entity/3,
    declare_chord_connection/4,
    clear_chord/0,
    clear_chord/1,

    % Query predicates
    all_chords/1,
    chord_entities/2,                   % chord_entities(+Name, -Entities)
    chord_connections/2,                % chord_connections(+Name, -Connections)
    entity_total/3,                     % entity_total(+Name, +EntityId, -Total)
    connection_matrix/2,                % connection_matrix(+Name, -Matrix)

    % Code generation
    generate_chord_component/2,
    generate_chord_component/3,
    generate_chord_styles/2,
    generate_chord_data/2,

    % Python generation
    generate_chord_matplotlib/2,
    generate_chord_plotly/2,

    % Testing
    test_chord_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic chord_spec/2.
:- dynamic chord_entity/3.
:- dynamic chord_connection/4.

:- discontiguous chord_spec/2.
:- discontiguous chord_entity/3.
:- discontiguous chord_connection/4.

% ============================================================================
% DEFAULT CHORD DEFINITIONS
% ============================================================================

% Trade flow example
chord_spec(trade_flow, [
    title("International Trade"),
    size(500),
    inner_radius_ratio(0.9),
    pad_angle(0.02),
    show_labels(true),
    theme(dark)
]).

chord_entity(trade_flow, usa, [label("USA"), color('#3b82f6')]).
chord_entity(trade_flow, china, [label("China"), color('#ef4444')]).
chord_entity(trade_flow, eu, [label("EU"), color('#22c55e')]).
chord_entity(trade_flow, japan, [label("Japan"), color('#f59e0b')]).
chord_entity(trade_flow, uk, [label("UK"), color('#8b5cf6')]).

% Trade flows (bidirectional values)
chord_connection(trade_flow, usa, china, 500).
chord_connection(trade_flow, usa, eu, 400).
chord_connection(trade_flow, usa, japan, 200).
chord_connection(trade_flow, usa, uk, 150).
chord_connection(trade_flow, china, eu, 350).
chord_connection(trade_flow, china, japan, 250).
chord_connection(trade_flow, china, uk, 100).
chord_connection(trade_flow, eu, japan, 150).
chord_connection(trade_flow, eu, uk, 300).
chord_connection(trade_flow, japan, uk, 80).

% Department communication example
chord_spec(dept_comms, [
    title("Department Communications"),
    size(450),
    inner_radius_ratio(0.85),
    show_labels(true),
    theme(dark)
]).

chord_entity(dept_comms, engineering, [label("Engineering"), color('#3b82f6')]).
chord_entity(dept_comms, design, [label("Design"), color('#ec4899')]).
chord_entity(dept_comms, product, [label("Product"), color('#22c55e')]).
chord_entity(dept_comms, marketing, [label("Marketing"), color('#f59e0b')]).
chord_entity(dept_comms, sales, [label("Sales"), color('#8b5cf6')]).

chord_connection(dept_comms, engineering, design, 80).
chord_connection(dept_comms, engineering, product, 120).
chord_connection(dept_comms, engineering, marketing, 30).
chord_connection(dept_comms, design, product, 90).
chord_connection(dept_comms, design, marketing, 60).
chord_connection(dept_comms, product, marketing, 100).
chord_connection(dept_comms, product, sales, 85).
chord_connection(dept_comms, marketing, sales, 150).

% ============================================================================
% CHORD MANAGEMENT
% ============================================================================

declare_chord_spec(Name, Config) :-
    retractall(chord_spec(Name, _)),
    assertz(chord_spec(Name, Config)).

declare_chord_entity(Name, EntityId, Config) :-
    assertz(chord_entity(Name, EntityId, Config)).

declare_chord_connection(Name, Source, Target, Value) :-
    assertz(chord_connection(Name, Source, Target, Value)).

clear_chord :-
    retractall(chord_spec(_, _)),
    retractall(chord_entity(_, _, _)),
    retractall(chord_connection(_, _, _, _)).

clear_chord(Name) :-
    retractall(chord_spec(Name, _)),
    retractall(chord_entity(Name, _, _)),
    retractall(chord_connection(Name, _, _, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

all_chords(Names) :-
    findall(Name, chord_spec(Name, _), Names).

chord_entities(Name, Entities) :-
    findall(EntityId, chord_entity(Name, EntityId, _), Entities).

chord_connections(Name, Connections) :-
    findall(conn(Source, Target, Value), chord_connection(Name, Source, Target, Value), Connections).

% Calculate total connections for an entity (both directions)
entity_total(Name, EntityId, Total) :-
    findall(V, chord_connection(Name, EntityId, _, V), OutValues),
    findall(V, chord_connection(Name, _, EntityId, V), InValues),
    append(OutValues, InValues, AllValues),
    sum_list(AllValues, Total).

sum_list([], 0).
sum_list([H|T], Sum) :-
    sum_list(T, Rest),
    Sum is H + Rest.

% Generate connection matrix
connection_matrix(Name, Matrix) :-
    chord_entities(Name, Entities),
    findall(Row, (
        member(Source, Entities),
        findall(Value, (
            member(Target, Entities),
            (chord_connection(Name, Source, Target, V) -> Value = V ; Value = 0)
        ), Row)
    ), Matrix).

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

generate_chord_component(Name, Code) :-
    generate_chord_component(Name, [], Code).

generate_chord_component(Name, _Options, Code) :-
    chord_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Chord Diagram"),
    (member(size(Size), Config) -> true ; Size = 500),
    (member(inner_radius_ratio(InnerRatio), Config) -> true ; InnerRatio = 0.9),
    (member(pad_angle(PadAngle), Config) -> true ; PadAngle = 0.02),
    (member(show_labels(ShowLabels), Config) -> true ; ShowLabels = true),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Generate entities and matrix data
    generate_entities_data(Name, EntitiesDataJS),
    generate_matrix_data(Name, MatrixDataJS),

    (ShowLabels == true -> ShowLabelsStr = 'true' ; ShowLabelsStr = 'false'),

    format(atom(Code),
'// Generated by UnifyWeaver - Chord Diagram Component
// Chart: ~w

import React, { useMemo } from "react";
import styles from "./~w.module.css";

interface ChordDiagramProps {
  onEntityClick?: (entityId: string) => void;
  onChordClick?: (source: string, target: string, value: number) => void;
}

~w

~w

export const ~w: React.FC<ChordDiagramProps> = ({ onEntityClick, onChordClick }) => {
  const size = ~w;
  const innerRadiusRatio = ~w;
  const padAngle = ~w;
  const showLabels = ~w;

  const center = size / 2;
  const outerRadius = size / 2 - 40;
  const innerRadius = outerRadius * innerRadiusRatio;

  // Calculate chord layout
  const layout = useMemo(() => {
    const n = entities.length;
    const totals = entities.map((_, i) =>
      matrix[i].reduce((sum, v) => sum + v, 0) +
      matrix.reduce((sum, row) => sum + row[i], 0)
    );
    const grandTotal = totals.reduce((sum, t) => sum + t, 0);

    // Calculate arc angles for each entity
    const arcs: { id: string; label: string; color: string; startAngle: number; endAngle: number }[] = [];
    let currentAngle = 0;

    entities.forEach((entity, i) => {
      const angleSpan = (totals[i] / grandTotal) * (2 * Math.PI - n * padAngle);
      arcs.push({
        id: entity.id,
        label: entity.label,
        color: entity.color,
        startAngle: currentAngle,
        endAngle: currentAngle + angleSpan,
      });
      currentAngle += angleSpan + padAngle;
    });

    // Calculate chord paths
    const chords: {
      source: { id: string; startAngle: number; endAngle: number };
      target: { id: string; startAngle: number; endAngle: number };
      sourceColor: string;
      value: number;
    }[] = [];

    // Track angle offsets within each arc
    const arcOffsets = new Array(n).fill(0);

    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const value = matrix[i][j] + (i !== j ? matrix[j][i] : 0);
        if (value === 0) continue;

        const sourceAngleSpan = (value / totals[i]) * (arcs[i].endAngle - arcs[i].startAngle);
        const targetAngleSpan = (value / totals[j]) * (arcs[j].endAngle - arcs[j].startAngle);

        chords.push({
          source: {
            id: entities[i].id,
            startAngle: arcs[i].startAngle + arcOffsets[i],
            endAngle: arcs[i].startAngle + arcOffsets[i] + sourceAngleSpan,
          },
          target: {
            id: entities[j].id,
            startAngle: arcs[j].startAngle + arcOffsets[j],
            endAngle: arcs[j].startAngle + arcOffsets[j] + targetAngleSpan,
          },
          sourceColor: entities[i].color,
          value,
        });

        arcOffsets[i] += sourceAngleSpan;
        if (i !== j) arcOffsets[j] += targetAngleSpan;
      }
    }

    return { arcs, chords };
  }, []);

  // Helper to generate arc path
  const arcPath = (startAngle: number, endAngle: number, r: number) => {
    const x1 = center + r * Math.cos(startAngle - Math.PI / 2);
    const y1 = center + r * Math.sin(startAngle - Math.PI / 2);
    const x2 = center + r * Math.cos(endAngle - Math.PI / 2);
    const y2 = center + r * Math.sin(endAngle - Math.PI / 2);
    const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
    return `M${x1},${y1} A${r},${r} 0 ${largeArc} 1 ${x2},${y2}`;
  };

  // Helper to generate chord path
  const chordPath = (
    s: { startAngle: number; endAngle: number },
    t: { startAngle: number; endAngle: number }
  ) => {
    const r = innerRadius;
    const sx1 = center + r * Math.cos(s.startAngle - Math.PI / 2);
    const sy1 = center + r * Math.sin(s.startAngle - Math.PI / 2);
    const sx2 = center + r * Math.cos(s.endAngle - Math.PI / 2);
    const sy2 = center + r * Math.sin(s.endAngle - Math.PI / 2);
    const tx1 = center + r * Math.cos(t.startAngle - Math.PI / 2);
    const ty1 = center + r * Math.sin(t.startAngle - Math.PI / 2);
    const tx2 = center + r * Math.cos(t.endAngle - Math.PI / 2);
    const ty2 = center + r * Math.sin(t.endAngle - Math.PI / 2);

    return `
      M${sx1},${sy1}
      A${r},${r} 0 0 1 ${sx2},${sy2}
      Q${center},${center} ${tx1},${ty1}
      A${r},${r} 0 0 1 ${tx2},${ty2}
      Q${center},${center} ${sx1},${sy1}
      Z
    `;
  };

  return (
    <div className={styles.chordContainer}>
      <h3 className={styles.title}>~w</h3>
      <svg width={size} height={size} className={styles.chordSvg}>
        {/* Chord ribbons */}
        {layout.chords.map((chord, i) => (
          <path
            key={i}
            d={chordPath(chord.source, chord.target)}
            fill={chord.sourceColor}
            fillOpacity={0.6}
            className={styles.chord}
            onClick={() => onChordClick?.(chord.source.id, chord.target.id, chord.value)}
          >
            <title>{`${chord.source.id} ↔ ${chord.target.id}: ${chord.value}`}</title>
          </path>
        ))}

        {/* Entity arcs */}
        {layout.arcs.map((arc) => (
          <g key={arc.id} className={styles.arc}>
            <path
              d={`${arcPath(arc.startAngle, arc.endAngle, outerRadius)}
                  L${center + innerRadius * Math.cos(arc.endAngle - Math.PI / 2)},${center + innerRadius * Math.sin(arc.endAngle - Math.PI / 2)}
                  ${arcPath(arc.endAngle, arc.startAngle, innerRadius).replace("M", "A").split("A")[1] ? "" : ""}
                  A${innerRadius},${innerRadius} 0 ${arc.endAngle - arc.startAngle > Math.PI ? 1 : 0} 0 ${center + innerRadius * Math.cos(arc.startAngle - Math.PI / 2)},${center + innerRadius * Math.sin(arc.startAngle - Math.PI / 2)}
                  Z`}
              fill={arc.color}
              onClick={() => onEntityClick?.(arc.id)}
            />
            {showLabels && (
              <text
                x={center + (outerRadius + 15) * Math.cos((arc.startAngle + arc.endAngle) / 2 - Math.PI / 2)}
                y={center + (outerRadius + 15) * Math.sin((arc.startAngle + arc.endAngle) / 2 - Math.PI / 2)}
                textAnchor="middle"
                dominantBaseline="middle"
                className={styles.label}
                transform={`rotate(${((arc.startAngle + arc.endAngle) / 2 * 180 / Math.PI) - 90}, ${center + (outerRadius + 15) * Math.cos((arc.startAngle + arc.endAngle) / 2 - Math.PI / 2)}, ${center + (outerRadius + 15) * Math.sin((arc.startAngle + arc.endAngle) / 2 - Math.PI / 2)})`}
              >
                {arc.label}
              </text>
            )}
          </g>
        ))}
      </svg>

      {/* Legend */}
      <div className={styles.legend}>
        {entities.map((entity) => (
          <div key={entity.id} className={styles.legendItem}>
            <span className={styles.legendColor} style={{ backgroundColor: entity.color }} />
            <span className={styles.legendLabel}>{entity.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ~w;
', [Name, ComponentName, EntitiesDataJS, MatrixDataJS, ComponentName,
    Size, InnerRatio, PadAngle, ShowLabelsStr, Title, ComponentName]).

% Generate entities data array
generate_entities_data(Name, JS) :-
    findall(EntityJS, (
        chord_entity(Name, EntityId, Config),
        (member(label(Label), Config) -> true ; atom_string(EntityId, Label)),
        (member(color(Color), Config) -> true ; Color = '#6b7280'),
        format(atom(EntityJS), '  { id: "~w", label: "~w", color: "~w" }',
               [EntityId, Label, Color])
    ), EntityJSList),
    atomic_list_concat(EntityJSList, ',\n', EntitiesStr),
    format(atom(JS), 'const entities = [\n~w\n];', [EntitiesStr]).

% Generate connection matrix
generate_matrix_data(Name, JS) :-
    connection_matrix(Name, Matrix),
    findall(RowJS, (
        member(Row, Matrix),
        format_js_number_array(Row, RowJS)
    ), RowJSList),
    atomic_list_concat(RowJSList, ',\n  ', MatrixStr),
    format(atom(JS), 'const matrix = [\n  ~w\n];', [MatrixStr]).

format_js_number_array(Numbers, JS) :-
    atomic_list_concat_numbers(Numbers, ', ', NumStr),
    format(atom(JS), '[~w]', [NumStr]).

atomic_list_concat_numbers([], _, '').
atomic_list_concat_numbers([X], _, S) :- format(atom(S), '~w', [X]).
atomic_list_concat_numbers([X|Xs], Sep, S) :-
    Xs \= [],
    atomic_list_concat_numbers(Xs, Sep, Rest),
    format(atom(S), '~w~w~w', [X, Sep, Rest]).

% ============================================================================
% CSS GENERATION
% ============================================================================

generate_chord_styles(_Name, CSS) :-
    CSS = '.chordContainer {
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

.chordSvg {
  overflow: visible;
}

.chord {
  cursor: pointer;
  transition: fill-opacity 0.2s ease;
}

.chord:hover {
  fill-opacity: 0.85;
}

.arc path {
  cursor: pointer;
  transition: opacity 0.2s ease;
}

.arc path:hover {
  opacity: 0.8;
}

.label {
  font-size: 0.75rem;
  fill: var(--text, #e0e0e0);
  pointer-events: none;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
  justify-content: center;
}

.legendItem {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.legendColor {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.legendLabel {
  font-size: 0.875rem;
  color: var(--text-secondary, #aaa);
}
'.

% ============================================================================
% DATA GENERATION
% ============================================================================

generate_chord_data(Name, DataCode) :-
    generate_entities_data(Name, EntitiesJS),
    generate_matrix_data(Name, MatrixJS),
    format(atom(DataCode), '~w~n~n~w', [EntitiesJS, MatrixJS]).

% ============================================================================
% MATPLOTLIB GENERATION
% ============================================================================

generate_chord_matplotlib(Name, PythonCode) :-
    chord_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Chord Diagram"),

    chord_entities(Name, Entities),
    findall(Label, (
        member(EntityId, Entities),
        chord_entity(Name, EntityId, EntityConfig),
        (member(label(Label), EntityConfig) -> true ; atom_string(EntityId, Label))
    ), Labels),
    format_python_list(Labels, LabelsPy),

    connection_matrix(Name, Matrix),
    generate_python_matrix(Matrix, MatrixPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Chord Diagram
# Chart: ~w

import numpy as np
import matplotlib.pyplot as plt

def plot_~w():
    """~w - Note: matplotlib has limited chord support, consider mpl_chord_diagram or Plotly"""
    labels = ~w
    matrix = np.array(~w)

    n = len(labels)
    totals = matrix.sum(axis=1) + matrix.sum(axis=0)

    # Create a simple circular visualization
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.8

    # Draw bars for each entity (showing total connections)
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    bars = ax.bar(angles, totals, width=width, bottom=0, color=colors, alpha=0.7)

    # Add labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)

    ax.set_title("~w\\n(Simplified - use Plotly for full chord diagram)", pad=20)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, LabelsPy, MatrixPy, Title, Name]).

% ============================================================================
% PLOTLY GENERATION (USING HOLOVIEWS/CHORD OR PLOTLY)
% ============================================================================

generate_chord_plotly(Name, PythonCode) :-
    chord_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Chord Diagram"),

    chord_entities(Name, Entities),
    findall(Label, (
        member(EntityId, Entities),
        chord_entity(Name, EntityId, EntityConfig),
        (member(label(Label), EntityConfig) -> true ; atom_string(EntityId, Label))
    ), Labels),
    findall(Color, (
        member(EntityId, Entities),
        chord_entity(Name, EntityId, EntityConfig),
        (member(color(Color), EntityConfig) -> true ; Color = '#6b7280')
    ), Colors),
    format_python_list(Labels, LabelsPy),
    format_python_list(Colors, ColorsPy),

    % Generate source, target, value arrays from connections
    findall(SrcIdx-TgtIdx-Value, (
        chord_connection(Name, Source, Target, Value),
        nth0(SrcIdx, Entities, Source),
        nth0(TgtIdx, Entities, Target)
    ), ConnectionData),
    findall(S, member(S-_-_, ConnectionData), Sources),
    findall(T, member(_-T-_, ConnectionData), Targets),
    findall(V, member(_-_-V, ConnectionData), Values),
    format_python_number_list(Sources, SourcesPy),
    format_python_number_list(Targets, TargetsPy),
    format_python_number_list(Values, ValuesPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Chord Diagram (using Plotly circular network)
# Chart: ~w
# Note: For true chord diagrams, consider holoviews or d3.js

import plotly.graph_objects as go
import numpy as np

def plot_~w():
    """~w"""
    labels = ~w
    colors = ~w

    sources = ~w
    targets = ~w
    values = ~w

    n = len(labels)

    # Calculate positions in a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_nodes = np.cos(angles)
    y_nodes = np.sin(angles)

    # Create edges
    edge_x = []
    edge_y = []
    for s, t, v in zip(sources, targets, values):
        edge_x.extend([x_nodes[s], x_nodes[t], None])
        edge_y.extend([y_nodes[s], y_nodes[t], None])

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(100, 100, 100, 0.5)"),
        hoverinfo="none"
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_nodes.tolist(), y=y_nodes.tolist(),
        mode="markers+text",
        marker=dict(size=30, color=colors),
        text=labels,
        textposition="top center",
        hoverinfo="text"
    ))

    fig.update_layout(
        title="~w",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600,
        height=600
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

generate_python_matrix(Matrix, PyMatrix) :-
    findall(RowPy, (
        member(Row, Matrix),
        format_python_number_list(Row, RowPy)
    ), Rows),
    atomic_list_concat(Rows, ', ', RowsStr),
    format(atom(PyMatrix), '[~w]', [RowsStr]).

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

test_chord_generator :-
    format('Testing chord_generator module...~n~n'),

    format('Test 1: Chord spec query~n'),
    (chord_spec(trade_flow, _)
    ->  format('  PASS: trade_flow spec exists~n')
    ;   format('  FAIL: trade_flow spec not found~n')
    ),

    format('~nTest 2: Entities query~n'),
    chord_entities(trade_flow, Entities),
    length(Entities, NumEntities),
    (NumEntities =:= 5
    ->  format('  PASS: 5 entities found~n')
    ;   format('  FAIL: Expected 5 entities, got ~w~n', [NumEntities])
    ),

    format('~nTest 3: Connections query~n'),
    chord_connections(trade_flow, Connections),
    length(Connections, NumConnections),
    (NumConnections =:= 10
    ->  format('  PASS: 10 connections found~n')
    ;   format('  FAIL: Expected 10 connections, got ~w~n', [NumConnections])
    ),

    format('~nTest 4: Entity total~n'),
    entity_total(trade_flow, usa, Total),
    (Total =:= 1250
    ->  format('  PASS: USA total is 1250~n')
    ;   format('  FAIL: Expected 1250, got ~w~n', [Total])
    ),

    format('~nTest 5: Connection matrix~n'),
    connection_matrix(trade_flow, Matrix),
    length(Matrix, MatrixRows),
    (MatrixRows =:= 5
    ->  format('  PASS: Matrix has 5 rows~n')
    ;   format('  FAIL: Expected 5 rows, got ~w~n', [MatrixRows])
    ),

    format('~nTest 6: Component generation~n'),
    generate_chord_component(trade_flow, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 4000
    ->  format('  PASS: Generated ~w chars~n', [CodeLen])
    ;   format('  FAIL: Code too short: ~w~n', [CodeLen])
    ),

    format('~nTest 7: Plotly generation~n'),
    generate_chord_plotly(trade_flow, PyCode),
    (sub_atom(PyCode, _, _, _, 'go.Scatter')
    ->  format('  PASS: Contains Plotly Scatter~n')
    ;   format('  FAIL: Missing Plotly Scatter~n')
    ),

    format('~nAll tests completed.~n').
