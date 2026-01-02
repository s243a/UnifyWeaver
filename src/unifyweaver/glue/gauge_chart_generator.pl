% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Gauge Chart Generator - Declarative Gauge/Meter Visualization
%
% This module provides declarative gauge chart definitions that generate
% TypeScript/React components for single-value indicator visualization.
%
% Usage:
%   % Define gauge configuration
%   gauge_spec(my_gauge, [
%       title("CPU Usage"),
%       min(0), max(100),
%       value(75),
%       thresholds([ok(60), warning(80), danger(100)])
%   ]).
%
%   % Generate React component
%   ?- generate_gauge_component(my_gauge, Code).

:- module(gauge_chart_generator, [
    % Gauge chart definition predicates
    gauge_spec/2,                       % gauge_spec(+Name, +Config)

    % Gauge management
    declare_gauge_spec/2,
    clear_gauge/0,
    clear_gauge/1,

    % Query predicates
    all_gauges/1,
    get_gauge_value/2,                  % get_gauge_value(+Name, -Value)
    get_gauge_range/3,                  % get_gauge_range(+Name, -Min, -Max)
    get_gauge_status/2,                 % get_gauge_status(+Name, -Status)

    % Code generation
    generate_gauge_component/2,
    generate_gauge_component/3,
    generate_gauge_styles/2,
    generate_gauge_hook/2,              % generate_gauge_hook(+Name, -HookCode)

    % Multi-gauge dashboard
    generate_gauge_dashboard/2,         % generate_gauge_dashboard(+GaugeNames, -Code)

    % Python generation
    generate_gauge_matplotlib/2,
    generate_gauge_plotly/2,

    % Testing
    test_gauge_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic gauge_spec/2.

:- discontiguous gauge_spec/2.

% ============================================================================
% DEFAULT GAUGE DEFINITIONS
% ============================================================================

% CPU usage gauge
gauge_spec(cpu_usage, [
    title("CPU Usage"),
    min(0),
    max(100),
    value(72),
    unit("%"),
    thresholds([
        threshold(ok, 0, 60, '#22c55e'),
        threshold(warning, 60, 80, '#f59e0b'),
        threshold(danger, 80, 100, '#ef4444')
    ]),
    size(200),
    arc_width(20),
    show_value(true),
    show_label(true),
    theme(dark)
]).

% Memory usage gauge
gauge_spec(memory_usage, [
    title("Memory"),
    min(0),
    max(100),
    value(45),
    unit("%"),
    thresholds([
        threshold(ok, 0, 70, '#22c55e'),
        threshold(warning, 70, 90, '#f59e0b'),
        threshold(danger, 90, 100, '#ef4444')
    ]),
    size(180),
    arc_width(18),
    show_value(true),
    theme(dark)
]).

% Temperature gauge
gauge_spec(temperature, [
    title("Temperature"),
    min(0),
    max(120),
    value(68),
    unit("°C"),
    thresholds([
        threshold(cold, 0, 40, '#3b82f6'),
        threshold(normal, 40, 70, '#22c55e'),
        threshold(warm, 70, 90, '#f59e0b'),
        threshold(hot, 90, 120, '#ef4444')
    ]),
    size(200),
    arc_width(24),
    show_value(true),
    theme(dark)
]).

% Speed gauge (speedometer style)
gauge_spec(speedometer, [
    title("Speed"),
    min(0),
    max(200),
    value(85),
    unit("km/h"),
    thresholds([
        threshold(slow, 0, 60, '#22c55e'),
        threshold(normal, 60, 120, '#3b82f6'),
        threshold(fast, 120, 160, '#f59e0b'),
        threshold(danger, 160, 200, '#ef4444')
    ]),
    size(250),
    arc_width(20),
    start_angle(225),
    end_angle(-45),
    show_ticks(true),
    tick_count(10),
    theme(dark)
]).

% ============================================================================
% GAUGE MANAGEMENT
% ============================================================================

declare_gauge_spec(Name, Config) :-
    retractall(gauge_spec(Name, _)),
    assertz(gauge_spec(Name, Config)).

clear_gauge :-
    retractall(gauge_spec(_, _)).

clear_gauge(Name) :-
    retractall(gauge_spec(Name, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

all_gauges(Names) :-
    findall(Name, gauge_spec(Name, _), Names).

get_gauge_value(Name, Value) :-
    gauge_spec(Name, Config),
    member(value(Value), Config).

get_gauge_range(Name, Min, Max) :-
    gauge_spec(Name, Config),
    (member(min(Min), Config) -> true ; Min = 0),
    (member(max(Max), Config) -> true ; Max = 100).

% Determine status based on thresholds
get_gauge_status(Name, Status) :-
    gauge_spec(Name, Config),
    member(value(Value), Config),
    member(thresholds(Thresholds), Config),
    find_status(Value, Thresholds, Status).

find_status(Value, [threshold(Status, Min, Max, _)|_], Status) :-
    Value >= Min, Value < Max, !.
find_status(Value, [threshold(Status, Min, Max, _)], Status) :-
    Value >= Min, Value =< Max, !.
find_status(Value, [_|Rest], Status) :-
    find_status(Value, Rest, Status).
find_status(_, [], unknown).

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

generate_gauge_component(Name, Code) :-
    generate_gauge_component(Name, [], Code).

generate_gauge_component(Name, _Options, Code) :-
    gauge_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Gauge"),
    (member(min(Min), Config) -> true ; Min = 0),
    (member(max(Max), Config) -> true ; Max = 100),
    (member(value(Value), Config) -> true ; Value = 0),
    (member(unit(Unit), Config) -> true ; Unit = ""),
    (member(size(Size), Config) -> true ; Size = 200),
    (member(arc_width(ArcWidth), Config) -> true ; ArcWidth = 20),
    (member(start_angle(StartAngle), Config) -> true ; StartAngle = 180),
    (member(end_angle(EndAngle), Config) -> true ; EndAngle = 0),
    (member(show_value(ShowValue), Config) -> true ; ShowValue = true),
    (member(show_ticks(ShowTicks), Config) -> true ; ShowTicks = false),
    (member(tick_count(TickCount), Config) -> true ; TickCount = 5),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Generate thresholds
    (member(thresholds(Thresholds), Config)
    ->  generate_thresholds_js(Thresholds, ThresholdsJS)
    ;   ThresholdsJS = '[]'
    ),

    (ShowValue == true -> ShowValueStr = 'true' ; ShowValueStr = 'false'),
    (ShowTicks == true -> ShowTicksStr = 'true' ; ShowTicksStr = 'false'),

    format(atom(Code),
'// Generated by UnifyWeaver - Gauge Chart Component
// Chart: ~w

import React, { useMemo } from "react";
import styles from "./~w.module.css";

interface GaugeChartProps {
  value?: number;
  onChange?: (value: number) => void;
}

const thresholds = ~w;

export const ~w: React.FC<GaugeChartProps> = ({
  value: propValue,
  onChange,
}) => {
  const value = propValue ?? ~w;
  const min = ~w;
  const max = ~w;
  const size = ~w;
  const arcWidth = ~w;
  const startAngle = ~w;
  const endAngle = ~w;
  const showValue = ~w;
  const showTicks = ~w;
  const tickCount = ~w;
  const unit = "~w";

  const center = size / 2;
  const radius = (size - arcWidth) / 2 - 10;

  // Calculate the arc path
  const polarToCartesian = (cx: number, cy: number, r: number, angle: number) => {
    const rad = ((angle - 90) * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    };
  };

  const describeArc = (cx: number, cy: number, r: number, startA: number, endA: number) => {
    const start = polarToCartesian(cx, cy, r, endA);
    const end = polarToCartesian(cx, cy, r, startA);
    const largeArcFlag = endA - startA <= 180 ? "0" : "1";
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
  };

  // Calculate value angle
  const range = max - min;
  const angleRange = startAngle - endAngle;
  const valueAngle = useMemo(() => {
    const normalized = Math.max(0, Math.min(1, (value - min) / range));
    return startAngle - normalized * angleRange;
  }, [value, min, range, startAngle, angleRange]);

  // Get current color based on thresholds
  const currentColor = useMemo(() => {
    for (const t of thresholds) {
      if (value >= t.min && value <= t.max) {
        return t.color;
      }
    }
    return "#666";
  }, [value]);

  // Generate tick marks
  const ticks = useMemo(() => {
    if (!showTicks) return [];
    return Array.from({ length: tickCount + 1 }, (_, i) => {
      const tickValue = min + (range / tickCount) * i;
      const tickAngle = startAngle - (i / tickCount) * angleRange;
      const innerPoint = polarToCartesian(center, center, radius - 8, tickAngle);
      const outerPoint = polarToCartesian(center, center, radius + 2, tickAngle);
      const labelPoint = polarToCartesian(center, center, radius - 20, tickAngle);
      return { value: tickValue, innerPoint, outerPoint, labelPoint };
    });
  }, [showTicks, tickCount, min, range, startAngle, angleRange, center, radius]);

  // Needle position
  const needlePoint = polarToCartesian(center, center, radius - 15, valueAngle);

  return (
    <div className={styles.gaugeContainer}>
      <svg width={size} height={size * 0.7} className={styles.gaugeSvg}>
        {/* Background arc */}
        <path
          d={describeArc(center, center, radius, startAngle, endAngle)}
          fill="none"
          stroke="var(--gauge-bg, rgba(255,255,255,0.1))"
          strokeWidth={arcWidth}
          strokeLinecap="round"
        />

        {/* Threshold arcs */}
        {thresholds.map((t, i) => {
          const tStartNorm = (t.min - min) / range;
          const tEndNorm = (t.max - min) / range;
          const tStart = startAngle - tStartNorm * angleRange;
          const tEnd = startAngle - tEndNorm * angleRange;
          return (
            <path
              key={i}
              d={describeArc(center, center, radius, tStart, tEnd)}
              fill="none"
              stroke={t.color}
              strokeWidth={arcWidth}
              strokeLinecap="butt"
              opacity={0.3}
            />
          );
        })}

        {/* Value arc */}
        <path
          d={describeArc(center, center, radius, startAngle, valueAngle)}
          fill="none"
          stroke={currentColor}
          strokeWidth={arcWidth}
          strokeLinecap="round"
        />

        {/* Tick marks */}
        {ticks.map((tick, i) => (
          <g key={i}>
            <line
              x1={tick.innerPoint.x}
              y1={tick.innerPoint.y}
              x2={tick.outerPoint.x}
              y2={tick.outerPoint.y}
              stroke="var(--text-secondary, #666)"
              strokeWidth={2}
            />
            <text
              x={tick.labelPoint.x}
              y={tick.labelPoint.y}
              className={styles.tickLabel}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {tick.value}
            </text>
          </g>
        ))}

        {/* Needle */}
        <line
          x1={center}
          y1={center}
          x2={needlePoint.x}
          y2={needlePoint.y}
          stroke={currentColor}
          strokeWidth={3}
          strokeLinecap="round"
        />
        <circle cx={center} cy={center} r={8} fill={currentColor} />
        <circle cx={center} cy={center} r={4} fill="var(--surface, #1a1a2e)" />

        {/* Value display */}
        {showValue && (
          <text
            x={center}
            y={center + 35}
            className={styles.valueText}
            textAnchor="middle"
          >
            {value.toFixed(0)}{unit}
          </text>
        )}
      </svg>
      <div className={styles.title}>~w</div>
    </div>
  );
};

export default ~w;
', [Name, ComponentName, ThresholdsJS, ComponentName, Value, Min, Max, Size,
    ArcWidth, StartAngle, EndAngle, ShowValueStr, ShowTicksStr, TickCount,
    Unit, Title, ComponentName]).

% Generate thresholds JS array
generate_thresholds_js(Thresholds, JS) :-
    findall(ThresholdJS, (
        member(threshold(Status, Min, Max, Color), Thresholds),
        format(atom(ThresholdJS), '{ status: "~w", min: ~w, max: ~w, color: "~w" }',
               [Status, Min, Max, Color])
    ), ThresholdJSList),
    atomic_list_concat(ThresholdJSList, ', ', ThresholdsStr),
    format(atom(JS), '[~w]', [ThresholdsStr]).

% ============================================================================
% CSS GENERATION
% ============================================================================

generate_gauge_styles(_Name, CSS) :-
    CSS = '.gaugeContainer {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  background: var(--surface, #16213e);
  border-radius: 12px;
}

.gaugeSvg {
  overflow: visible;
}

.valueText {
  font-size: 1.5rem;
  font-weight: 700;
  fill: var(--text, #e0e0e0);
}

.tickLabel {
  font-size: 0.625rem;
  fill: var(--text-secondary, #888);
}

.title {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary, #aaa);
  margin-top: 0.5rem;
}
'.

% ============================================================================
% GAUGE HOOK GENERATION
% ============================================================================

generate_gauge_hook(Name, HookCode) :-
    gauge_spec(Name, Config),
    (member(min(Min), Config) -> true ; Min = 0),
    (member(max(Max), Config) -> true ; Max = 100),
    (member(value(DefaultValue), Config) -> true ; DefaultValue = 0),

    atom_string(Name, NameStr),
    pascal_case(NameStr, HookName),

    format(atom(HookCode),
'// Generated by UnifyWeaver - Gauge Hook
// Hook: use~w

import { useState, useCallback } from "react";

export const use~w = (initialValue: number = ~w) => {
  const [value, setValue] = useState(initialValue);
  const min = ~w;
  const max = ~w;

  const setGaugeValue = useCallback((newValue: number) => {
    setValue(Math.max(min, Math.min(max, newValue)));
  }, []);

  const increment = useCallback((amount: number = 1) => {
    setValue((v) => Math.min(max, v + amount));
  }, []);

  const decrement = useCallback((amount: number = 1) => {
    setValue((v) => Math.max(min, v - amount));
  }, []);

  const percentage = ((value - min) / (max - min)) * 100;

  return {
    value,
    setValue: setGaugeValue,
    increment,
    decrement,
    percentage,
    min,
    max,
  };
};
', [HookName, HookName, DefaultValue, Min, Max]).

% ============================================================================
% MULTI-GAUGE DASHBOARD
% ============================================================================

generate_gauge_dashboard(GaugeNames, Code) :-
    findall(ImportCode, (
        member(Name, GaugeNames),
        atom_string(Name, NameStr),
        pascal_case(NameStr, ComponentName),
        format(atom(ImportCode), 'import { ~w } from "./~w";', [ComponentName, ComponentName])
    ), Imports),
    atomic_list_concat(Imports, '\n', ImportsCode),

    findall(GaugeJSX, (
        member(Name, GaugeNames),
        atom_string(Name, NameStr),
        pascal_case(NameStr, ComponentName),
        format(atom(GaugeJSX), '        <~w />', [ComponentName])
    ), GaugeJSXList),
    atomic_list_concat(GaugeJSXList, '\n', GaugesCode),

    format(atom(Code),
'// Generated by UnifyWeaver - Gauge Dashboard

import React from "react";
~w

export const GaugeDashboard: React.FC = () => {
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
      gap: "1rem",
      padding: "1rem",
    }}>
~w
    </div>
  );
};

export default GaugeDashboard;
', [ImportsCode, GaugesCode]).

% ============================================================================
% MATPLOTLIB GENERATION
% ============================================================================

generate_gauge_matplotlib(Name, PythonCode) :-
    gauge_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Gauge"),
    (member(min(Min), Config) -> true ; Min = 0),
    (member(max(Max), Config) -> true ; Max = 100),
    (member(value(Value), Config) -> true ; Value = 0),
    (member(unit(Unit), Config) -> true ; Unit = ""),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Gauge Chart
# Chart: ~w

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Circle

def plot_~w():
    """~w"""
    value = ~w
    min_val = ~w
    max_val = ~w
    unit = "~w"

    fig, ax = plt.subplots(figsize=(8, 6))

    # Gauge parameters
    center = (0.5, 0.35)
    radius = 0.35
    width = 0.08

    # Background arc
    theta1, theta2 = 180, 0
    bg_wedge = Wedge(center, radius, theta1, theta2, width=width, facecolor="#333", edgecolor="none")
    ax.add_patch(bg_wedge)

    # Value arc
    normalized = (value - min_val) / (max_val - min_val)
    value_angle = 180 - normalized * 180
    value_wedge = Wedge(center, radius, 180, value_angle, width=width,
                        facecolor="#22c55e" if normalized < 0.6 else "#f59e0b" if normalized < 0.8 else "#ef4444",
                        edgecolor="none")
    ax.add_patch(value_wedge)

    # Center circle
    center_circle = Circle(center, 0.05, facecolor="#1a1a2e", edgecolor="white", linewidth=2)
    ax.add_patch(center_circle)

    # Needle
    angle_rad = np.deg2rad(value_angle + 90)
    needle_length = radius - 0.1
    end_x = center[0] + needle_length * np.cos(angle_rad)
    end_y = center[1] + needle_length * np.sin(angle_rad)
    ax.plot([center[0], end_x], [center[1], end_y], color="white", linewidth=3)

    # Value text
    ax.text(center[0], center[1] - 0.15, f"{value}{unit}", ha="center", va="center",
            fontsize=24, fontweight="bold", color="white")

    # Title
    ax.text(center[0], 0.05, "~w", ha="center", va="center",
            fontsize=14, color="#888")

    # Min/Max labels
    ax.text(center[0] - radius - 0.05, center[1], str(min_val), ha="right", va="center",
            fontsize=10, color="#888")
    ax.text(center[0] + radius + 0.05, center[1], str(max_val), ha="left", va="center",
            fontsize=10, color="#888")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, Value, Min, Max, Unit, Title, Name]).

% ============================================================================
% PLOTLY GENERATION
% ============================================================================

generate_gauge_plotly(Name, PythonCode) :-
    gauge_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Gauge"),
    (member(min(Min), Config) -> true ; Min = 0),
    (member(max(Max), Config) -> true ; Max = 100),
    (member(value(Value), Config) -> true ; Value = 0),
    (member(unit(Unit), Config) -> true ; Unit = ""),

    % Generate steps from thresholds
    (member(thresholds(Thresholds), Config)
    ->  generate_plotly_steps(Thresholds, StepsPy)
    ;   StepsPy = '[]'
    ),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - Gauge Chart (Plotly)
# Chart: ~w

import plotly.graph_objects as go

def plot_~w():
    """~w"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=~w,
        title={"text": "~w"},
        number={"suffix": "~w"},
        gauge={
            "axis": {"range": [~w, ~w]},
            "bar": {"color": "#3b82f6"},
            "steps": ~w,
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": ~w
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        font={"color": "#e0e0e0"}
    )

    fig.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, Value, Title, Unit, Min, Max, StepsPy, Value, Name]).

generate_plotly_steps(Thresholds, StepsPy) :-
    findall(StepPy, (
        member(threshold(_, Min, Max, Color), Thresholds),
        format(atom(StepPy), '{"range": [~w, ~w], "color": "~w"}', [Min, Max, Color])
    ), Steps),
    atomic_list_concat(Steps, ', ', StepsStr),
    format(atom(StepsPy), '[~w]', [StepsStr]).

% ============================================================================
% UTILITY PREDICATES
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

test_gauge_generator :-
    format('Testing gauge_chart_generator module...~n~n'),

    format('Test 1: Gauge spec query~n'),
    (gauge_spec(cpu_usage, _)
    ->  format('  PASS: cpu_usage spec exists~n')
    ;   format('  FAIL: cpu_usage spec not found~n')
    ),

    format('~nTest 2: Get gauge value~n'),
    get_gauge_value(cpu_usage, Value),
    (Value =:= 72
    ->  format('  PASS: Value is 72~n')
    ;   format('  FAIL: Expected 72, got ~w~n', [Value])
    ),

    format('~nTest 3: Get gauge range~n'),
    get_gauge_range(cpu_usage, Min, Max),
    (Min =:= 0, Max =:= 100
    ->  format('  PASS: Range 0-100~n')
    ;   format('  FAIL: Expected 0-100, got ~w-~w~n', [Min, Max])
    ),

    format('~nTest 4: Get gauge status~n'),
    get_gauge_status(cpu_usage, Status),
    (Status == warning
    ->  format('  PASS: Status is warning~n')
    ;   format('  FAIL: Expected warning, got ~w~n', [Status])
    ),

    format('~nTest 5: Component generation~n'),
    generate_gauge_component(cpu_usage, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 2000
    ->  format('  PASS: Generated ~w chars~n', [CodeLen])
    ;   format('  FAIL: Code too short: ~w~n', [CodeLen])
    ),

    format('~nTest 6: Hook generation~n'),
    generate_gauge_hook(cpu_usage, HookCode),
    (sub_atom(HookCode, _, _, _, 'useCpuUsage')
    ->  format('  PASS: Hook contains useCpuUsage~n')
    ;   format('  FAIL: Missing hook name~n')
    ),

    format('~nTest 7: Plotly generation~n'),
    generate_gauge_plotly(cpu_usage, PyCode),
    (sub_atom(PyCode, _, _, _, 'go.Indicator')
    ->  format('  PASS: Contains Plotly Indicator~n')
    ;   format('  FAIL: Missing Plotly Indicator~n')
    ),

    format('~nAll tests completed.~n').
