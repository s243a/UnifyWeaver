% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Funnel Chart Generator - Declarative Funnel/Pipeline Visualization
%
% This module provides declarative funnel chart definitions that generate
% TypeScript/React components for conversion and pipeline visualization.
%
% Usage:
%   % Define funnel stages
%   funnel_stage(my_funnel, visitors, [label("Visitors"), value(10000)]).
%   funnel_stage(my_funnel, signups, [label("Sign-ups"), value(3000)]).
%
%   % Generate React component
%   ?- generate_funnel_component(my_funnel, Code).

:- module(funnel_chart_generator, [
    % Funnel chart definition predicates
    funnel_spec/2,                      % funnel_spec(+Name, +Config)
    funnel_stage/3,                     % funnel_stage(+Name, +StageId, +Config)

    % Funnel management
    declare_funnel_spec/2,
    declare_funnel_stage/3,
    clear_funnel/0,
    clear_funnel/1,

    % Query predicates
    all_funnels/1,
    funnel_stages/2,                    % funnel_stages(+Name, -Stages)
    get_stage_config/3,                 % get_stage_config(+Name, +StageId, -Config)
    funnel_conversion_rate/3,           % funnel_conversion_rate(+Name, +StageId, -Rate)

    % Code generation
    generate_funnel_component/2,
    generate_funnel_component/3,
    generate_funnel_styles/2,
    generate_funnel_data/2,

    % Python generation
    generate_funnel_matplotlib/2,
    generate_funnel_plotly/2,

    % Testing
    test_funnel_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic funnel_spec/2.
:- dynamic funnel_stage/3.

:- discontiguous funnel_spec/2.
:- discontiguous funnel_stage/3.

% ============================================================================
% DEFAULT FUNNEL DEFINITIONS
% ============================================================================

% Sales funnel example
funnel_spec(sales_funnel, [
    title("Sales Pipeline"),
    width(400),
    height(300),
    show_percentages(true),
    show_values(true),
    color_scheme(blues),
    theme(dark)
]).

funnel_stage(sales_funnel, leads, [label("Leads"), value(5000), order(1)]).
funnel_stage(sales_funnel, qualified, [label("Qualified"), value(2500), order(2)]).
funnel_stage(sales_funnel, proposals, [label("Proposals"), value(1200), order(3)]).
funnel_stage(sales_funnel, negotiations, [label("Negotiations"), value(600), order(4)]).
funnel_stage(sales_funnel, closed, [label("Closed Won"), value(300), order(5)]).

% User conversion funnel
funnel_spec(user_conversion, [
    title("User Conversion"),
    width(350),
    height(280),
    show_percentages(true),
    show_values(true),
    color_scheme(greens),
    theme(dark)
]).

funnel_stage(user_conversion, visitors, [label("Visitors"), value(100000), order(1)]).
funnel_stage(user_conversion, signups, [label("Sign-ups"), value(15000), order(2)]).
funnel_stage(user_conversion, activated, [label("Activated"), value(8000), order(3)]).
funnel_stage(user_conversion, subscribers, [label("Subscribers"), value(2000), order(4)]).

% ============================================================================
% FUNNEL MANAGEMENT
% ============================================================================

declare_funnel_spec(Name, Config) :-
    retractall(funnel_spec(Name, _)),
    assertz(funnel_spec(Name, Config)).

declare_funnel_stage(Name, StageId, Config) :-
    assertz(funnel_stage(Name, StageId, Config)).

clear_funnel :-
    retractall(funnel_spec(_, _)),
    retractall(funnel_stage(_, _, _)).

clear_funnel(Name) :-
    retractall(funnel_spec(Name, _)),
    retractall(funnel_stage(Name, _, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

all_funnels(Names) :-
    findall(Name, funnel_spec(Name, _), Names).

% Get stages ordered by their order property
funnel_stages(Name, Stages) :-
    findall(Order-StageId, (
        funnel_stage(Name, StageId, Config),
        (member(order(Order), Config) -> true ; Order = 0)
    ), Pairs),
    keysort(Pairs, SortedPairs),
    findall(S, member(_-S, SortedPairs), Stages).

get_stage_config(Name, StageId, Config) :-
    funnel_stage(Name, StageId, Config).

% Calculate conversion rate from first stage
funnel_conversion_rate(Name, StageId, Rate) :-
    funnel_stages(Name, [FirstStage|_]),
    funnel_stage(Name, FirstStage, FirstConfig),
    member(value(FirstValue), FirstConfig),
    funnel_stage(Name, StageId, Config),
    member(value(Value), Config),
    FirstValue > 0,
    Rate is (Value / FirstValue) * 100.

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

generate_funnel_component(Name, Code) :-
    generate_funnel_component(Name, [], Code).

generate_funnel_component(Name, _Options, Code) :-
    funnel_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Funnel Chart"),
    (member(width(Width), Config) -> true ; Width = 400),
    (member(height(Height), Config) -> true ; Height = 300),
    (member(show_percentages(ShowPct), Config) -> true ; ShowPct = true),
    (member(show_values(ShowVals), Config) -> true ; ShowVals = true),
    (member(color_scheme(ColorScheme), Config) -> true ; ColorScheme = blues),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Get stages data
    generate_stages_data(Name, StagesDataJS),
    get_color_palette(ColorScheme, ColorPaletteJS),

    (ShowPct == true -> ShowPctStr = 'true' ; ShowPctStr = 'false'),
    (ShowVals == true -> ShowValsStr = 'true' ; ShowValsStr = 'false'),

    format(atom(Code),
'// Generated by UnifyWeaver - Funnel Chart Component
// Chart: ~w

import React, { useMemo } from "react";
import styles from "./~w.module.css";

interface FunnelChartProps {
  onStageClick?: (stageId: string, value: number, percentage: number) => void;
}

~w

const colorPalette = ~w;

export const ~w: React.FC<FunnelChartProps> = ({ onStageClick }) => {
  const width = ~w;
  const height = ~w;
  const showPercentages = ~w;
  const showValues = ~w;

  const maxValue = useMemo(() => Math.max(...stages.map(s => s.value)), []);

  const stageHeight = useMemo(() => height / stages.length, [height]);

  // Calculate trapezoid points for each stage
  const getStagePoints = (index: number, value: number) => {
    const widthRatio = value / maxValue;
    const nextRatio = index < stages.length - 1
      ? stages[index + 1].value / maxValue
      : widthRatio * 0.6;

    const topWidth = width * widthRatio;
    const bottomWidth = width * nextRatio;
    const y = index * stageHeight;

    const topLeft = (width - topWidth) / 2;
    const bottomLeft = (width - bottomWidth) / 2;

    return `${topLeft},${y} ${topLeft + topWidth},${y} ${bottomLeft + bottomWidth},${y + stageHeight} ${bottomLeft},${y + stageHeight}`;
  };

  const formatValue = (value: number) => {
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toString();
  };

  return (
    <div className={styles.funnelContainer}>
      <h3 className={styles.title}>~w</h3>
      <div className={styles.chartWrapper}>
        <svg width={width} height={height} className={styles.funnelSvg}>
          {stages.map((stage, index) => {
            const percentage = ((stage.value / maxValue) * 100).toFixed(1);
            return (
              <g key={stage.id}>
                <polygon
                  points={getStagePoints(index, stage.value)}
                  fill={colorPalette[index % colorPalette.length]}
                  className={styles.stage}
                  onClick={() => onStageClick?.(stage.id, stage.value, parseFloat(percentage))}
                />
                <text
                  x={width / 2}
                  y={index * stageHeight + stageHeight / 2}
                  className={styles.stageLabel}
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {stage.label}
                  {showValues && ` (${formatValue(stage.value)})`}
                  {showPercentages && ` - ${percentage}%`}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Side labels with conversion rates */}
        <div className={styles.conversionRates}>
          {stages.slice(1).map((stage, index) => {
            const prevValue = stages[index].value;
            const dropRate = ((1 - stage.value / prevValue) * 100).toFixed(1);
            return (
              <div
                key={stage.id}
                className={styles.conversionLabel}
                style={{ top: `${(index + 0.5) * stageHeight}px` }}
              >
                <span className={styles.dropArrow}>↓</span>
                <span className={styles.dropRate}>{dropRate}% drop</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ~w;
', [Name, ComponentName, StagesDataJS, ColorPaletteJS, ComponentName,
    Width, Height, ShowPctStr, ShowValsStr, Title, ComponentName]).

% Generate stages data array
generate_stages_data(Name, JS) :-
    funnel_stages(Name, Stages),
    findall(StageJS, (
        member(StageId, Stages),
        funnel_stage(Name, StageId, Config),
        member(value(Value), Config),
        (member(label(Label), Config) -> true ; atom_string(StageId, Label)),
        format(atom(StageJS), '  { id: "~w", label: "~w", value: ~w }', [StageId, Label, Value])
    ), StageJSList),
    atomic_list_concat(StageJSList, ',\n', StagesStr),
    format(atom(JS), 'const stages = [\n~w\n];', [StagesStr]).

% Color palettes
get_color_palette(blues, '["#1e40af", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]').
get_color_palette(greens, '["#166534", "#22c55e", "#4ade80", "#86efac", "#bbf7d0"]').
get_color_palette(purples, '["#5b21b6", "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"]').
get_color_palette(oranges, '["#c2410c", "#ea580c", "#f97316", "#fb923c", "#fdba74"]').
get_color_palette(_, Palette) :- get_color_palette(blues, Palette).

% ============================================================================
% CSS GENERATION
% ============================================================================

generate_funnel_styles(_Name, CSS) :-
    CSS = '.funnelContainer {
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

.chartWrapper {
  position: relative;
  display: flex;
}

.funnelSvg {
  overflow: visible;
}

.stage {
  cursor: pointer;
  transition: opacity 0.2s ease, filter 0.2s ease;
}

.stage:hover {
  opacity: 0.85;
  filter: brightness(1.1);
}

.stageLabel {
  font-size: 0.8rem;
  font-weight: 500;
  fill: white;
  pointer-events: none;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

.conversionRates {
  position: absolute;
  right: -80px;
  top: 0;
  height: 100%;
}

.conversionLabel {
  position: absolute;
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.7rem;
  color: var(--text-secondary, #888);
  transform: translateY(-50%);
}

.dropArrow {
  color: #ef4444;
}

.dropRate {
  white-space: nowrap;
}
'.

% ============================================================================
% DATA GENERATION
% ============================================================================

generate_funnel_data(Name, DataCode) :-
    generate_stages_data(Name, DataCode).

% ============================================================================
% MATPLOTLIB GENERATION
% ============================================================================

generate_funnel_matplotlib(Name, PythonCode) :-
    funnel_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Funnel Chart"),

    funnel_stages(Name, Stages),
    findall(Label, (
        member(StageId, Stages),
        funnel_stage(Name, StageId, StageConfig),
        (member(label(Label), StageConfig) -> true ; atom_string(StageId, Label))
    ), Labels),
    findall(Value, (
        member(StageId, Stages),
        funnel_stage(Name, StageId, StageConfig),
        member(value(Value), StageConfig)
    ), Values),

    format_python_list(Labels, LabelsPy),
    format_python_number_list(Values, ValuesPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Funnel Chart
# Chart: ~w

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_~w():
    """~w"""
    labels = ~w
    values = ~w

    fig, ax = plt.subplots(figsize=(10, 8))

    max_val = max(values)
    n_stages = len(values)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_stages))

    for i, (label, value) in enumerate(zip(labels, values)):
        # Calculate widths
        width_ratio = value / max_val
        next_ratio = values[i + 1] / max_val if i < n_stages - 1 else width_ratio * 0.6

        y = n_stages - i - 1
        height = 0.8

        # Create trapezoid
        top_width = width_ratio
        bottom_width = next_ratio
        left_top = (1 - top_width) / 2
        left_bottom = (1 - bottom_width) / 2

        vertices = [
            (left_top, y + height),
            (left_top + top_width, y + height),
            (left_bottom + bottom_width, y),
            (left_bottom, y)
        ]

        polygon = patches.Polygon(vertices, closed=True, facecolor=colors[i], edgecolor="white")
        ax.add_patch(polygon)

        # Add label
        ax.text(0.5, y + height / 2, f"{label}\\n{value:,}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_stages)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("~w", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, LabelsPy, ValuesPy, Title, Name]).

% ============================================================================
% PLOTLY GENERATION
% ============================================================================

generate_funnel_plotly(Name, PythonCode) :-
    funnel_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Funnel Chart"),

    funnel_stages(Name, Stages),
    findall(Label, (
        member(StageId, Stages),
        funnel_stage(Name, StageId, StageConfig),
        (member(label(Label), StageConfig) -> true ; atom_string(StageId, Label))
    ), Labels),
    findall(Value, (
        member(StageId, Stages),
        funnel_stage(Name, StageId, StageConfig),
        member(value(Value), StageConfig)
    ), Values),

    format_python_list(Labels, LabelsPy),
    format_python_number_list(Values, ValuesPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Funnel Chart (Plotly)
# Chart: ~w

import plotly.express as px
import pandas as pd

def plot_~w():
    """~w"""
    data = dict(
        Stage=~w,
        Value=~w
    )

    fig = px.funnel(
        data,
        x="Value",
        y="Stage",
        title="~w"
    )

    fig.update_layout(
        template="plotly_dark",
        font=dict(size=12)
    )

    fig.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, LabelsPy, ValuesPy, Title, Name]).

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

test_funnel_generator :-
    format('Testing funnel_chart_generator module...~n~n'),

    format('Test 1: Funnel spec query~n'),
    (funnel_spec(sales_funnel, _)
    ->  format('  PASS: sales_funnel spec exists~n')
    ;   format('  FAIL: sales_funnel spec not found~n')
    ),

    format('~nTest 2: Stages query~n'),
    funnel_stages(sales_funnel, Stages),
    length(Stages, NumStages),
    (NumStages =:= 5
    ->  format('  PASS: 5 stages found~n')
    ;   format('  FAIL: Expected 5 stages, got ~w~n', [NumStages])
    ),

    format('~nTest 3: Stage ordering~n'),
    funnel_stages(sales_funnel, [First|_]),
    (First == leads
    ->  format('  PASS: First stage is leads~n')
    ;   format('  FAIL: First stage should be leads, got ~w~n', [First])
    ),

    format('~nTest 4: Conversion rate~n'),
    funnel_conversion_rate(sales_funnel, closed, Rate),
    (Rate =:= 6.0
    ->  format('  PASS: Conversion rate 6.0%~n')
    ;   format('  FAIL: Expected 6.0%, got ~w~n', [Rate])
    ),

    format('~nTest 5: Component generation~n'),
    generate_funnel_component(sales_funnel, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 1000
    ->  format('  PASS: Generated ~w chars~n', [CodeLen])
    ;   format('  FAIL: Code too short: ~w~n', [CodeLen])
    ),

    format('~nTest 6: Plotly generation~n'),
    generate_funnel_plotly(sales_funnel, PyCode),
    (sub_atom(PyCode, _, _, _, 'px.funnel')
    ->  format('  PASS: Contains Plotly funnel~n')
    ;   format('  FAIL: Missing Plotly funnel~n')
    ),

    format('~nAll tests completed.~n').
