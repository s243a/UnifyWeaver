% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Heatmap Generator - Declarative Heatmap Visualization
%
% This module provides declarative heatmap definitions that generate
% TypeScript/React components for data visualization.
%
% Usage:
%   % Define heatmap data
%   heatmap_cell(my_heatmap, 0, 0, 0.8).
%   heatmap_cell(my_heatmap, 0, 1, 0.5).
%
%   % Define heatmap configuration
%   heatmap_spec(my_heatmap, [
%       title("Correlation Matrix"),
%       x_labels(["A", "B", "C"]),
%       y_labels(["X", "Y", "Z"]),
%       color_scale(viridis)
%   ]).
%
%   % Generate React component
%   ?- generate_heatmap_component(my_heatmap, Code).

:- module(heatmap_generator, [
    % Heatmap definition predicates
    heatmap_cell/4,                     % heatmap_cell(+Name, +X, +Y, +Value)
    heatmap_spec/2,                     % heatmap_spec(+Name, +Config)
    heatmap_row/3,                      % heatmap_row(+Name, +RowIndex, +Values)

    % Heatmap management
    declare_heatmap_cell/4,             % declare_heatmap_cell(+Name, +X, +Y, +Value)
    declare_heatmap_spec/2,             % declare_heatmap_spec(+Name, +Config)
    declare_heatmap_row/3,              % declare_heatmap_row(+Name, +RowIndex, +Values)
    clear_heatmap/0,                    % clear_heatmap
    clear_heatmap/1,                    % clear_heatmap(+Name)

    % Query predicates
    all_heatmaps/1,                     % all_heatmaps(-Names)
    heatmap_dimensions/3,               % heatmap_dimensions(+Name, -Rows, -Cols)
    heatmap_value_range/3,              % heatmap_value_range(+Name, -Min, -Max)
    get_heatmap_cell/4,                 % get_heatmap_cell(+Name, +X, +Y, -Value)

    % Code generation
    generate_heatmap_component/2,       % generate_heatmap_component(+Name, -Code)
    generate_heatmap_component/3,       % generate_heatmap_component(+Name, +Options, -Code)
    generate_heatmap_styles/2,          % generate_heatmap_styles(+Name, -CssCode)
    generate_heatmap_data/2,            % generate_heatmap_data(+Name, -DataCode)

    % Python/matplotlib generation
    generate_heatmap_matplotlib/2,      % generate_heatmap_matplotlib(+Name, -PythonCode)

    % Layout-integrated generation
    generate_heatmap_with_layout/3,     % generate_heatmap_with_layout(+Name, +LayoutPattern, -Code)

    % Testing
    test_heatmap_generator/0
]).

:- use_module(library(lists)).
:- use_module(layout_generator).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic heatmap_cell/4.
:- dynamic heatmap_spec/2.
:- dynamic heatmap_row/3.

:- discontiguous heatmap_cell/4.
:- discontiguous heatmap_spec/2.
:- discontiguous heatmap_row/3.

% ============================================================================
% DEFAULT HEATMAP DEFINITIONS - Correlation Matrix Example
% ============================================================================

% Correlation matrix example (3x3)
heatmap_spec(correlation_demo, [
    title("Correlation Matrix"),
    x_labels(["Variable A", "Variable B", "Variable C"]),
    y_labels(["Variable A", "Variable B", "Variable C"]),
    color_scale(diverging),
    show_values(true),
    theme(dark)
]).

% Correlation values (symmetric matrix)
heatmap_cell(correlation_demo, 0, 0, 1.0).
heatmap_cell(correlation_demo, 0, 1, 0.8).
heatmap_cell(correlation_demo, 0, 2, -0.3).
heatmap_cell(correlation_demo, 1, 0, 0.8).
heatmap_cell(correlation_demo, 1, 1, 1.0).
heatmap_cell(correlation_demo, 1, 2, 0.5).
heatmap_cell(correlation_demo, 2, 0, -0.3).
heatmap_cell(correlation_demo, 2, 1, 0.5).
heatmap_cell(correlation_demo, 2, 2, 1.0).

% Activity heatmap example (7x24 - days x hours)
heatmap_spec(activity_demo, [
    title("Weekly Activity"),
    x_labels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    y_labels(["00:00", "06:00", "12:00", "18:00"]),
    color_scale(sequential),
    show_values(false),
    cell_size(40),
    theme(dark)
]).

% Sample activity data (using rows for efficiency)
heatmap_row(activity_demo, 0, [0.1, 0.2, 0.15, 0.18, 0.12, 0.05, 0.03]).
heatmap_row(activity_demo, 1, [0.5, 0.6, 0.55, 0.58, 0.52, 0.3, 0.2]).
heatmap_row(activity_demo, 2, [0.8, 0.9, 0.85, 0.88, 0.82, 0.6, 0.4]).
heatmap_row(activity_demo, 3, [0.4, 0.5, 0.45, 0.48, 0.42, 0.7, 0.5]).

% ============================================================================
% HEATMAP MANAGEMENT
% ============================================================================

%% declare_heatmap_cell(+Name, +X, +Y, +Value)
%  Dynamically add a heatmap cell.
declare_heatmap_cell(Name, X, Y, Value) :-
    assertz(heatmap_cell(Name, X, Y, Value)).

%% declare_heatmap_spec(+Name, +Config)
%  Dynamically add a heatmap specification.
declare_heatmap_spec(Name, Config) :-
    retractall(heatmap_spec(Name, _)),
    assertz(heatmap_spec(Name, Config)).

%% declare_heatmap_row(+Name, +RowIndex, +Values)
%  Dynamically add a row of heatmap data.
declare_heatmap_row(Name, RowIndex, Values) :-
    assertz(heatmap_row(Name, RowIndex, Values)).

%% clear_heatmap
%  Clear all heatmap definitions.
clear_heatmap :-
    retractall(heatmap_cell(_, _, _, _)),
    retractall(heatmap_spec(_, _)),
    retractall(heatmap_row(_, _, _)).

%% clear_heatmap(+Name)
%  Clear a specific heatmap.
clear_heatmap(Name) :-
    retractall(heatmap_cell(Name, _, _, _)),
    retractall(heatmap_spec(Name, _)),
    retractall(heatmap_row(Name, _, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% all_heatmaps(-Names)
%  Get all defined heatmap names.
all_heatmaps(Names) :-
    findall(Name, heatmap_spec(Name, _), Names).

%% heatmap_dimensions(+Name, -Rows, -Cols)
%  Get the dimensions of a heatmap.
heatmap_dimensions(Name, Rows, Cols) :-
    heatmap_spec(Name, Config),
    (member(x_labels(XLabels), Config) -> length(XLabels, Cols) ; Cols = 0),
    (member(y_labels(YLabels), Config) -> length(YLabels, Rows) ; Rows = 0).

%% heatmap_value_range(+Name, -Min, -Max)
%  Get the min and max values in a heatmap.
heatmap_value_range(Name, Min, Max) :-
    findall(V, heatmap_cell(Name, _, _, V), CellValues),
    findall(V, (heatmap_row(Name, _, Row), member(V, Row)), RowValues),
    append(CellValues, RowValues, AllValues),
    (AllValues = [] -> (Min = 0, Max = 1) ; (min_list(AllValues, Min), max_list(AllValues, Max))).

%% get_heatmap_cell(+Name, +X, +Y, -Value)
%  Get a specific cell value (from cells or rows).
get_heatmap_cell(Name, X, Y, Value) :-
    heatmap_cell(Name, X, Y, Value), !.
get_heatmap_cell(Name, X, Y, Value) :-
    heatmap_row(Name, Y, Row),
    nth0(X, Row, Value), !.
get_heatmap_cell(_, _, _, 0).

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

%% generate_heatmap_component(+Name, -Code)
%  Generate a React component for the heatmap.
generate_heatmap_component(Name, Code) :-
    generate_heatmap_component(Name, [], Code).

%% generate_heatmap_component(+Name, +Options, -Code)
%  Generate a React component with options.
generate_heatmap_component(Name, _Options, Code) :-
    heatmap_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Heatmap"),
    (member(show_values(ShowValues), Config) -> true ; ShowValues = false),
    (member(cell_size(CellSize), Config) -> true ; CellSize = 50),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Generate data array
    generate_heatmap_data(Name, DataCode),

    % Get labels
    (member(x_labels(XLabels), Config) -> true ; XLabels = []),
    (member(y_labels(YLabels), Config) -> true ; YLabels = []),
    format_labels(XLabels, XLabelsJS),
    format_labels(YLabels, YLabelsJS),

    % Color scale
    (member(color_scale(ColorScale), Config) -> true ; ColorScale = sequential),
    get_color_scale_function(ColorScale, ColorScaleCode),

    % Show values as string
    (ShowValues == true -> ShowValuesStr = 'true' ; ShowValuesStr = 'false'),

    format(atom(Code),
'// Generated by UnifyWeaver - Heatmap Component
// Heatmap: ~w

import React, { useMemo } from "react";
import styles from "./~w.module.css";

interface HeatmapProps {
  onCellClick?: (x: number, y: number, value: number) => void;
}

~w

const xLabels = ~w;
const yLabels = ~w;

export const ~w: React.FC<HeatmapProps> = ({ onCellClick }) => {
  const cellSize = ~w;
  const showValues = ~w;

  const getColor = useMemo(() => {
    ~w
  }, []);

  return (
    <div className={styles.heatmapContainer}>
      <h3 className={styles.title}>~w</h3>
      <div className={styles.heatmap}>
        {/* Y-axis labels */}
        <div className={styles.yLabels}>
          {yLabels.map((label, i) => (
            <div key={i} className={styles.yLabel} style={{ height: cellSize }}>
              {label}
            </div>
          ))}
        </div>

        {/* Heatmap grid */}
        <div className={styles.grid}>
          {data.map((row, y) => (
            <div key={y} className={styles.row}>
              {row.map((value, x) => (
                <div
                  key={x}
                  className={styles.cell}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: getColor(value),
                  }}
                  onClick={() => onCellClick?.(x, y, value)}
                  title={`(${x}, ${y}): ${value.toFixed(2)}`}
                >
                  {showValues && <span className={styles.cellValue}>{value.toFixed(1)}</span>}
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* X-axis labels */}
        <div className={styles.xLabels}>
          <div className={styles.xLabelSpacer} />
          {xLabels.map((label, i) => (
            <div key={i} className={styles.xLabel} style={{ width: cellSize }}>
              {label}
            </div>
          ))}
        </div>
      </div>

      {/* Color scale legend */}
      <div className={styles.legend}>
        <div className={styles.legendGradient} />
        <div className={styles.legendLabels}>
          <span>Low</span>
          <span>High</span>
        </div>
      </div>
    </div>
  );
};

export default ~w;
', [Name, ComponentName, DataCode, XLabelsJS, YLabelsJS, ComponentName, CellSize,
    ShowValuesStr, ColorScaleCode, Title, ComponentName]).

%% format_labels(+Labels, -JSCode)
format_labels([], '[]').
format_labels(Labels, JSCode) :-
    Labels \= [],
    findall(QuotedLabel, (
        member(Label, Labels),
        format(atom(QuotedLabel), '"~w"', [Label])
    ), QuotedLabels),
    atomic_list_concat(QuotedLabels, ', ', LabelsStr),
    format(atom(JSCode), '[~w]', [LabelsStr]).

%% get_color_scale_function(+Scale, -Code)
get_color_scale_function(sequential, Code) :-
    Code = '(value: number): string => {
      // Sequential: white to blue
      const intensity = Math.max(0, Math.min(1, value));
      const r = Math.round(255 * (1 - intensity));
      const g = Math.round(255 * (1 - intensity * 0.5));
      const b = 255;
      return `rgb(${r}, ${g}, ${b})`;
    }'.

get_color_scale_function(diverging, Code) :-
    Code = '(value: number): string => {
      // Diverging: blue (-1) to white (0) to red (+1)
      const clamped = Math.max(-1, Math.min(1, value));
      if (clamped < 0) {
        const intensity = -clamped;
        return `rgb(${Math.round(255 * (1 - intensity))}, ${Math.round(255 * (1 - intensity))}, 255)`;
      } else {
        const intensity = clamped;
        return `rgb(255, ${Math.round(255 * (1 - intensity))}, ${Math.round(255 * (1 - intensity))})`;
      }
    }'.

get_color_scale_function(viridis, Code) :-
    Code = '(value: number): string => {
      // Viridis-like color scale
      const t = Math.max(0, Math.min(1, value));
      const r = Math.round(68 + 187 * t);
      const g = Math.round(1 + 180 * Math.sin(Math.PI * t));
      const b = Math.round(84 + 171 * (1 - t));
      return `rgb(${r}, ${g}, ${b})`;
    }'.

get_color_scale_function(heat, Code) :-
    Code = '(value: number): string => {
      // Heat: black -> red -> yellow -> white
      const t = Math.max(0, Math.min(1, value));
      const r = Math.min(255, Math.round(510 * t));
      const g = Math.max(0, Math.min(255, Math.round(510 * (t - 0.5))));
      const b = Math.max(0, Math.min(255, Math.round(510 * (t - 0.75))));
      return `rgb(${r}, ${g}, ${b})`;
    }'.

get_color_scale_function(_, Code) :-
    get_color_scale_function(sequential, Code).

% ============================================================================
% HEATMAP DATA GENERATION
% ============================================================================

%% generate_heatmap_data(+Name, -DataCode)
%  Generate JavaScript data array for the heatmap.
generate_heatmap_data(Name, DataCode) :-
    heatmap_dimensions(Name, Rows, Cols),
    generate_data_rows(Name, 0, Rows, Cols, RowsCode),
    format(atom(DataCode), 'const data: number[][] = [~n~w];', [RowsCode]).

generate_data_rows(_, Row, Rows, _, '') :- Row >= Rows, !.
generate_data_rows(Name, Row, Rows, Cols, Code) :-
    Row < Rows,
    generate_data_row(Name, Row, 0, Cols, RowCode),
    NextRow is Row + 1,
    generate_data_rows(Name, NextRow, Rows, Cols, RestCode),
    (RestCode = ''
    ->  format(atom(Code), '  [~w]', [RowCode])
    ;   format(atom(Code), '  [~w],~n~w', [RowCode, RestCode])
    ).

generate_data_row(_, _, Col, Cols, '') :- Col >= Cols, !.
generate_data_row(Name, Row, Col, Cols, Code) :-
    Col < Cols,
    get_heatmap_cell(Name, Col, Row, Value),
    NextCol is Col + 1,
    generate_data_row(Name, Row, NextCol, Cols, RestCode),
    (RestCode = ''
    ->  format(atom(Code), '~w', [Value])
    ;   format(atom(Code), '~w, ~w', [Value, RestCode])
    ).

% ============================================================================
% CSS GENERATION
% ============================================================================

%% generate_heatmap_styles(+Name, -CssCode)
%  Generate CSS for heatmap styling.
generate_heatmap_styles(_Name, CssCode) :-
    CssCode = '.heatmapContainer {
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

.heatmap {
  display: flex;
  flex-direction: column;
}

.yLabels {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  padding-right: 0.5rem;
}

.yLabel {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  font-size: 0.75rem;
  color: var(--text-secondary, #888);
}

.grid {
  display: flex;
  flex-direction: column;
  gap: 1px;
  background: var(--border, rgba(255,255,255,0.1));
  padding: 1px;
}

.row {
  display: flex;
  gap: 1px;
}

.cell {
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.cell:hover {
  transform: scale(1.05);
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  z-index: 1;
}

.cellValue {
  font-size: 0.625rem;
  font-weight: 600;
  color: rgba(0,0,0,0.7);
  text-shadow: 0 0 2px rgba(255,255,255,0.5);
}

.xLabels {
  display: flex;
  padding-top: 0.5rem;
}

.xLabelSpacer {
  width: 60px;
}

.xLabel {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  font-size: 0.75rem;
  color: var(--text-secondary, #888);
}

.legend {
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.legendGradient {
  width: 200px;
  height: 12px;
  background: linear-gradient(to right, #ffffff, #0066ff);
  border-radius: 2px;
}

.legendLabels {
  display: flex;
  justify-content: space-between;
  width: 200px;
  font-size: 0.625rem;
  color: var(--text-secondary, #888);
}
'.

% ============================================================================
% MATPLOTLIB GENERATION
% ============================================================================

%% generate_heatmap_matplotlib(+Name, -PythonCode)
%  Generate Python/matplotlib code for the heatmap.
generate_heatmap_matplotlib(Name, PythonCode) :-
    heatmap_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Heatmap"),
    (member(x_labels(XLabels), Config) -> true ; XLabels = []),
    (member(y_labels(YLabels), Config) -> true ; YLabels = []),
    (member(color_scale(ColorScale), Config) -> true ; ColorScale = sequential),

    format_python_list(XLabels, XLabelsPy),
    format_python_list(YLabels, YLabelsPy),
    get_matplotlib_cmap(ColorScale, Cmap),

    % Generate data matrix
    generate_matplotlib_data(Name, DataPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Heatmap
# Heatmap: ~w

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_~w():
    """~w"""
    # Data
    data = np.array(~w)

    # Labels
    x_labels = ~w
    y_labels = ~w

    # Create figure
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="~w",
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=ax
    )

    ax.set_title("~w")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, DataPy, XLabelsPy, YLabelsPy, Cmap, Title, Name]).

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

%% get_matplotlib_cmap(+Scale, -Cmap)
get_matplotlib_cmap(sequential, 'Blues').
get_matplotlib_cmap(diverging, 'RdBu_r').
get_matplotlib_cmap(viridis, 'viridis').
get_matplotlib_cmap(heat, 'hot').
get_matplotlib_cmap(_, 'Blues').

%% generate_matplotlib_data(+Name, -DataPy)
generate_matplotlib_data(Name, DataPy) :-
    heatmap_dimensions(Name, Rows, Cols),
    generate_py_rows(Name, 0, Rows, Cols, RowsStr),
    format(atom(DataPy), '[~w]', [RowsStr]).

generate_py_rows(_, Row, Rows, _, '') :- Row >= Rows, !.
generate_py_rows(Name, Row, Rows, Cols, Code) :-
    Row < Rows,
    generate_py_row(Name, Row, 0, Cols, RowCode),
    NextRow is Row + 1,
    generate_py_rows(Name, NextRow, Rows, Cols, RestCode),
    (RestCode = ''
    ->  format(atom(Code), '[~w]', [RowCode])
    ;   format(atom(Code), '[~w], ~w', [RowCode, RestCode])
    ).

generate_py_row(_, _, Col, Cols, '') :- Col >= Cols, !.
generate_py_row(Name, Row, Col, Cols, Code) :-
    Col < Cols,
    get_heatmap_cell(Name, Col, Row, Value),
    NextCol is Col + 1,
    generate_py_row(Name, Row, NextCol, Cols, RestCode),
    (RestCode = ''
    ->  format(atom(Code), '~w', [Value])
    ;   format(atom(Code), '~w, ~w', [Value, RestCode])
    ).

% ============================================================================
% LAYOUT INTEGRATION
% ============================================================================

%% generate_heatmap_with_layout(+Name, +LayoutPattern, -Code)
%  Generate heatmap component with layout wrapper.
generate_heatmap_with_layout(Name, LayoutPattern, Code) :-
    generate_heatmap_component(Name, ComponentCode),
    generate_heatmap_styles(Name, HeatmapCSS),
    (has_layout(LayoutPattern)
    ->  generate_layout_css(LayoutPattern, LayoutCSS),
        format(atom(Code), '~w~n~n/* Layout CSS */~n~w~n~n/* Heatmap CSS */~n~w', [ComponentCode, LayoutCSS, HeatmapCSS])
    ;   format(atom(Code), '~w~n~n/* CSS */~n~w', [ComponentCode, HeatmapCSS])
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

test_heatmap_generator :-
    format('Testing heatmap_generator module...~n~n'),

    % Test heatmap spec query
    format('Test 1: Heatmap spec query~n'),
    (heatmap_spec(correlation_demo, _)
    ->  format('  PASS: correlation_demo spec exists~n')
    ;   format('  FAIL: correlation_demo spec not found~n')
    ),

    % Test cell query
    format('~nTest 2: Cell query~n'),
    (get_heatmap_cell(correlation_demo, 0, 0, V), V =:= 1.0
    ->  format('  PASS: Cell (0,0) = 1.0~n')
    ;   format('  FAIL: Cell query failed~n')
    ),

    % Test dimensions
    format('~nTest 3: Dimensions~n'),
    (heatmap_dimensions(correlation_demo, Rows, Cols), Rows =:= 3, Cols =:= 3
    ->  format('  PASS: Dimensions 3x3~n')
    ;   format('  FAIL: Dimensions incorrect~n')
    ),

    % Test component generation
    format('~nTest 4: Component generation~n'),
    generate_heatmap_component(correlation_demo, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 1000
    ->  format('  PASS: Generated ~w chars~n', [CodeLen])
    ;   format('  FAIL: Code too short: ~w~n', [CodeLen])
    ),

    % Test matplotlib generation
    format('~nTest 5: Matplotlib generation~n'),
    generate_heatmap_matplotlib(correlation_demo, PyCode),
    (sub_atom(PyCode, _, _, _, 'seaborn')
    ->  format('  PASS: Contains seaborn import~n')
    ;   format('  FAIL: Missing seaborn import~n')
    ),

    format('~nAll tests completed.~n').

:- initialization(test_heatmap_generator, main).
