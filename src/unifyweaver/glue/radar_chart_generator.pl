% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Radar Chart Generator - Declarative Radar/Spider Chart Visualization
%
% This module provides declarative radar chart definitions that generate
% TypeScript/React components for multi-axis data comparison.
%
% Usage:
%   % Define radar chart axes
%   radar_axis(my_radar, speed, [label("Speed"), max(100)]).
%   radar_axis(my_radar, power, [label("Power"), max(100)]).
%
%   % Define data series
%   radar_series(my_radar, player1, [speed(80), power(65), defense(70)]).
%
%   % Generate React component
%   ?- generate_radar_component(my_radar, Code).

:- module(radar_chart_generator, [
    % Radar chart definition predicates
    radar_spec/2,                       % radar_spec(+Name, +Config)
    radar_axis/3,                       % radar_axis(+Name, +AxisId, +Config)
    radar_series/3,                     % radar_series(+Name, +SeriesId, +Values)

    % Radar management
    declare_radar_spec/2,
    declare_radar_axis/3,
    declare_radar_series/3,
    clear_radar/0,
    clear_radar/1,

    % Query predicates
    all_radars/1,
    radar_axes/2,                       % radar_axes(+Name, -Axes)
    radar_series_list/2,                % radar_series_list(+Name, -Series)
    get_axis_config/3,                  % get_axis_config(+Name, +AxisId, -Config)

    % Code generation
    generate_radar_component/2,
    generate_radar_component/3,
    generate_radar_styles/2,
    generate_radar_data/2,

    % Python/matplotlib generation
    generate_radar_matplotlib/2,

    % Testing
    test_radar_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic radar_spec/2.
:- dynamic radar_axis/3.
:- dynamic radar_series/3.

:- discontiguous radar_spec/2.
:- discontiguous radar_axis/3.
:- discontiguous radar_series/3.

% ============================================================================
% DEFAULT RADAR CHART DEFINITIONS
% ============================================================================

% Player stats comparison example
radar_spec(player_stats, [
    title("Player Statistics"),
    size(400),
    show_legend(true),
    fill_opacity(0.3),
    theme(dark)
]).

radar_axis(player_stats, speed, [label("Speed"), max(100), min(0)]).
radar_axis(player_stats, power, [label("Power"), max(100), min(0)]).
radar_axis(player_stats, defense, [label("Defense"), max(100), min(0)]).
radar_axis(player_stats, stamina, [label("Stamina"), max(100), min(0)]).
radar_axis(player_stats, technique, [label("Technique"), max(100), min(0)]).

radar_series(player_stats, player_a, [
    values([speed(85), power(70), defense(60), stamina(90), technique(75)]),
    color('#3b82f6'),
    label("Player A")
]).

radar_series(player_stats, player_b, [
    values([speed(70), power(85), defense(80), stamina(65), technique(60)]),
    color('#10b981'),
    label("Player B")
]).

% Product comparison example
radar_spec(product_comparison, [
    title("Product Comparison"),
    size(350),
    show_legend(true),
    fill_opacity(0.25),
    theme(dark)
]).

radar_axis(product_comparison, price, [label("Price"), max(5), min(0)]).
radar_axis(product_comparison, quality, [label("Quality"), max(5), min(0)]).
radar_axis(product_comparison, features, [label("Features"), max(5), min(0)]).
radar_axis(product_comparison, support, [label("Support"), max(5), min(0)]).
radar_axis(product_comparison, ease_of_use, [label("Ease of Use"), max(5), min(0)]).

radar_series(product_comparison, product_x, [
    values([price(4), quality(5), features(3), support(4), ease_of_use(5)]),
    color('#6366f1'),
    label("Product X")
]).

% ============================================================================
% RADAR MANAGEMENT
% ============================================================================

declare_radar_spec(Name, Config) :-
    retractall(radar_spec(Name, _)),
    assertz(radar_spec(Name, Config)).

declare_radar_axis(Name, AxisId, Config) :-
    assertz(radar_axis(Name, AxisId, Config)).

declare_radar_series(Name, SeriesId, Values) :-
    assertz(radar_series(Name, SeriesId, Values)).

clear_radar :-
    retractall(radar_spec(_, _)),
    retractall(radar_axis(_, _, _)),
    retractall(radar_series(_, _, _)).

clear_radar(Name) :-
    retractall(radar_spec(Name, _)),
    retractall(radar_axis(Name, _, _)),
    retractall(radar_series(Name, _, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

all_radars(Names) :-
    findall(Name, radar_spec(Name, _), Names).

radar_axes(Name, Axes) :-
    findall(AxisId, radar_axis(Name, AxisId, _), Axes).

radar_series_list(Name, Series) :-
    findall(SeriesId, radar_series(Name, SeriesId, _), Series).

get_axis_config(Name, AxisId, Config) :-
    radar_axis(Name, AxisId, Config).

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

generate_radar_component(Name, Code) :-
    generate_radar_component(Name, [], Code).

generate_radar_component(Name, _Options, Code) :-
    radar_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Radar Chart"),
    (member(size(Size), Config) -> true ; Size = 400),
    (member(fill_opacity(Opacity), Config) -> true ; Opacity = 0.3),
    (member(show_legend(ShowLegend), Config) -> true ; ShowLegend = true),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Get axes and series
    radar_axes(Name, Axes),
    length(Axes, NumAxes),
    generate_axes_labels(Name, Axes, AxesLabelsJS),
    generate_series_data(Name, SeriesDataJS),

    (ShowLegend == true -> ShowLegendStr = 'true' ; ShowLegendStr = 'false'),

    format(atom(Code),
'// Generated by UnifyWeaver - Radar Chart Component
// Chart: ~w

import React, { useMemo } from "react";
import styles from "./~w.module.css";

interface RadarChartProps {
  onPointClick?: (seriesId: string, axisId: string, value: number) => void;
}

const axes = ~w;
const numAxes = ~w;

~w

export const ~w: React.FC<RadarChartProps> = ({ onPointClick }) => {
  const size = ~w;
  const center = size / 2;
  const radius = size * 0.35;
  const fillOpacity = ~w;
  const showLegend = ~w;

  // Calculate point positions for each axis
  const getPoint = (axisIndex: number, value: number, max: number = 100) => {
    const angle = (Math.PI * 2 * axisIndex) / numAxes - Math.PI / 2;
    const normalizedValue = Math.min(1, Math.max(0, value / max));
    const r = radius * normalizedValue;
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle),
    };
  };

  // Generate polygon points for a series
  const getPolygonPoints = (values: number[]) => {
    return values
      .map((v, i) => {
        const point = getPoint(i, v);
        return `${point.x},${point.y}`;
      })
      .join(" ");
  };

  // Generate axis line endpoints
  const axisLines = useMemo(() => {
    return axes.map((_, i) => {
      const angle = (Math.PI * 2 * i) / numAxes - Math.PI / 2;
      return {
        x2: center + radius * Math.cos(angle),
        y2: center + radius * Math.sin(angle),
      };
    });
  }, []);

  // Generate grid circles
  const gridCircles = [0.2, 0.4, 0.6, 0.8, 1.0];

  return (
    <div className={styles.radarContainer}>
      <h3 className={styles.title}>~w</h3>
      <svg width={size} height={size} className={styles.radarSvg}>
        {/* Grid circles */}
        {gridCircles.map((scale, i) => (
          <circle
            key={i}
            cx={center}
            cy={center}
            r={radius * scale}
            className={styles.gridCircle}
          />
        ))}

        {/* Axis lines */}
        {axisLines.map((line, i) => (
          <line
            key={i}
            x1={center}
            y1={center}
            x2={line.x2}
            y2={line.y2}
            className={styles.axisLine}
          />
        ))}

        {/* Axis labels */}
        {axes.map((axis, i) => {
          const angle = (Math.PI * 2 * i) / numAxes - Math.PI / 2;
          const labelRadius = radius + 25;
          const x = center + labelRadius * Math.cos(angle);
          const y = center + labelRadius * Math.sin(angle);
          return (
            <text
              key={i}
              x={x}
              y={y}
              className={styles.axisLabel}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {axis.label}
            </text>
          );
        })}

        {/* Data polygons */}
        {seriesData.map((series) => (
          <polygon
            key={series.id}
            points={getPolygonPoints(series.values)}
            fill={series.color}
            fillOpacity={fillOpacity}
            stroke={series.color}
            strokeWidth={2}
            className={styles.dataPolygon}
          />
        ))}

        {/* Data points */}
        {seriesData.map((series) =>
          series.values.map((value, i) => {
            const point = getPoint(i, value);
            return (
              <circle
                key={`${series.id}-${i}`}
                cx={point.x}
                cy={point.y}
                r={4}
                fill={series.color}
                className={styles.dataPoint}
                onClick={() => onPointClick?.(series.id, axes[i].id, value)}
              />
            );
          })
        )}
      </svg>

      {/* Legend */}
      {showLegend && (
        <div className={styles.legend}>
          {seriesData.map((series) => (
            <div key={series.id} className={styles.legendItem}>
              <span
                className={styles.legendColor}
                style={{ backgroundColor: series.color }}
              />
              <span className={styles.legendLabel}>{series.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ~w;
', [Name, ComponentName, AxesLabelsJS, NumAxes, SeriesDataJS, ComponentName,
    Size, Opacity, ShowLegendStr, Title, ComponentName]).

% Generate axes labels array
generate_axes_labels(Name, Axes, JS) :-
    findall(AxisJS, (
        member(AxisId, Axes),
        radar_axis(Name, AxisId, Config),
        (member(label(Label), Config) -> true ; atom_string(AxisId, Label)),
        (member(max(Max), Config) -> true ; Max = 100),
        format(atom(AxisJS), '{ id: "~w", label: "~w", max: ~w }', [AxisId, Label, Max])
    ), AxisJSList),
    atomic_list_concat(AxisJSList, ', ', AxesStr),
    format(atom(JS), '[~w]', [AxesStr]).

% Generate series data
generate_series_data(Name, JS) :-
    findall(SeriesJS, (
        radar_series(Name, SeriesId, Config),
        member(values(ValuesList), Config),
        (member(color(Color), Config) -> true ; Color = '#3b82f6'),
        (member(label(Label), Config) -> true ; atom_string(SeriesId, Label)),
        extract_values(ValuesList, Values),
        format_js_array(Values, ValuesJS),
        format(atom(SeriesJS), '  { id: "~w", label: "~w", color: "~w", values: ~w }',
               [SeriesId, Label, Color, ValuesJS])
    ), SeriesJSList),
    atomic_list_concat(SeriesJSList, ',\n', SeriesStr),
    format(atom(JS), 'const seriesData = [\n~w\n];', [SeriesStr]).

extract_values([], []).
extract_values([Term|Rest], [Value|Values]) :-
    Term =.. [_, Value],
    extract_values(Rest, Values).

format_js_array(Values, JS) :-
    atomic_list_concat_numbers(Values, ', ', ValuesStr),
    format(atom(JS), '[~w]', [ValuesStr]).

atomic_list_concat_numbers([], _, '').
atomic_list_concat_numbers([X], _, S) :- format(atom(S), '~w', [X]).
atomic_list_concat_numbers([X|Xs], Sep, S) :-
    Xs \= [],
    atomic_list_concat_numbers(Xs, Sep, Rest),
    format(atom(S), '~w~w~w', [X, Sep, Rest]).

% ============================================================================
% CSS GENERATION
% ============================================================================

generate_radar_styles(_Name, CSS) :-
    CSS = '.radarContainer {
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

.radarSvg {
  overflow: visible;
}

.gridCircle {
  fill: none;
  stroke: var(--border, rgba(255, 255, 255, 0.1));
  stroke-width: 1;
}

.axisLine {
  stroke: var(--border, rgba(255, 255, 255, 0.2));
  stroke-width: 1;
}

.axisLabel {
  font-size: 0.75rem;
  fill: var(--text-secondary, #888);
}

.dataPolygon {
  transition: opacity 0.2s ease;
}

.dataPolygon:hover {
  opacity: 0.8;
}

.dataPoint {
  cursor: pointer;
  transition: r 0.15s ease;
}

.dataPoint:hover {
  r: 6;
}

.legend {
  display: flex;
  gap: 1.5rem;
  margin-top: 1rem;
  flex-wrap: wrap;
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

generate_radar_data(Name, DataCode) :-
    radar_axes(Name, Axes),
    radar_series_list(Name, Series),
    generate_axes_labels(Name, Axes, AxesJS),
    generate_series_data(Name, SeriesJS),
    format(atom(DataCode), '~w~n~n~w', [AxesJS, SeriesJS]).

% ============================================================================
% MATPLOTLIB GENERATION
% ============================================================================

generate_radar_matplotlib(Name, PythonCode) :-
    radar_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Radar Chart"),
    radar_axes(Name, Axes),
    length(Axes, NumAxes),

    % Generate axis labels
    findall(Label, (
        member(AxisId, Axes),
        radar_axis(Name, AxisId, AxisConfig),
        (member(label(Label), AxisConfig) -> true ; atom_string(AxisId, Label))
    ), Labels),
    format_python_list(Labels, LabelsPy),

    % Generate series data
    generate_matplotlib_series(Name, Axes, SeriesPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Radar Chart
# Chart: ~w

import numpy as np
import matplotlib.pyplot as plt
from math import pi

def plot_~w():
    """~w"""
    # Axis labels
    categories = ~w
    num_vars = ~w

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot data
~w

    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    ax.set_title("~w", size=14, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, LabelsPy, NumAxes, SeriesPy, Title, Name]).

generate_matplotlib_series(Name, Axes, SeriesPy) :-
    findall(SeriesCode, (
        radar_series(Name, SeriesId, Config),
        member(values(ValuesList), Config),
        (member(color(Color), Config) -> true ; Color = 'blue'),
        (member(label(Label), Config) -> true ; atom_string(SeriesId, Label)),
        extract_ordered_values(ValuesList, Axes, Values),
        append(Values, [FirstVal], ValuesLoop),
        Values = [FirstVal|_],
        format_python_number_list(ValuesLoop, ValuesPy),
        format(atom(SeriesCode),
'    values_~w = ~w
    ax.plot(angles, values_~w, "o-", linewidth=2, label="~w", color="~w")
    ax.fill(angles, values_~w, alpha=0.25, color="~w")',
               [SeriesId, ValuesPy, SeriesId, Label, Color, SeriesId, Color])
    ), SeriesCodes),
    atomic_list_concat(SeriesCodes, '\n\n', SeriesPy).

extract_ordered_values(_, [], []).
extract_ordered_values(ValuesList, [Axis|Axes], [Value|Values]) :-
    (member(Term, ValuesList), Term =.. [Axis, Value] -> true ; Value = 0),
    extract_ordered_values(ValuesList, Axes, Values).

format_python_list([], '[]').
format_python_list(List, PyList) :-
    List \= [],
    findall(Q, (member(I, List), format(atom(Q), '"~w"', [I])), Qs),
    atomic_list_concat(Qs, ', ', S),
    format(atom(PyList), '[~w]', [S]).

format_python_number_list(Numbers, PyList) :-
    atomic_list_concat_numbers(Numbers, ', ', S),
    format(atom(PyList), '[~w]', [S]).

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

test_radar_generator :-
    format('Testing radar_chart_generator module...~n~n'),

    format('Test 1: Radar spec query~n'),
    (radar_spec(player_stats, _)
    ->  format('  PASS: player_stats spec exists~n')
    ;   format('  FAIL: player_stats spec not found~n')
    ),

    format('~nTest 2: Axes query~n'),
    radar_axes(player_stats, Axes),
    length(Axes, NumAxes),
    (NumAxes =:= 5
    ->  format('  PASS: 5 axes found~n')
    ;   format('  FAIL: Expected 5 axes, got ~w~n', [NumAxes])
    ),

    format('~nTest 3: Series query~n'),
    radar_series_list(player_stats, Series),
    length(Series, NumSeries),
    (NumSeries =:= 2
    ->  format('  PASS: 2 series found~n')
    ;   format('  FAIL: Expected 2 series, got ~w~n', [NumSeries])
    ),

    format('~nTest 4: Component generation~n'),
    generate_radar_component(player_stats, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 1000
    ->  format('  PASS: Generated ~w chars~n', [CodeLen])
    ;   format('  FAIL: Code too short: ~w~n', [CodeLen])
    ),

    format('~nTest 5: Matplotlib generation~n'),
    generate_radar_matplotlib(player_stats, PyCode),
    (sub_atom(PyCode, _, _, _, 'polar=True')
    ->  format('  PASS: Contains polar plot~n')
    ;   format('  FAIL: Missing polar plot~n')
    ),

    format('~nAll tests completed.~n').
