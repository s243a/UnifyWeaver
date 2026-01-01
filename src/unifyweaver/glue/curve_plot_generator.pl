% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Curve Plot Generator - Declarative Mathematical Curve Visualization
%
% This module provides declarative curve definitions that generate
% TypeScript/React components using Chart.js for visualization.
%
% Usage:
%   % Define curves
%   curve(sine_wave, [type(sine), amplitude(1.5), frequency(2.0)]).
%   curve(parabola, [type(quadratic), a(1), b(0), c(0)]).
%
%   % Define a plot with multiple curves
%   plot_spec(math_demo, [
%       title("Math Functions"),
%       curves([sine_wave, parabola]),
%       x_range(-5, 5),
%       theme(dark)
%   ]).
%
%   % Generate React component
%   ?- generate_curve_component(math_demo, Code).

:- module(curve_plot_generator, [
    % Curve definition predicates
    curve/2,                            % curve(+Name, +Properties)
    plot_spec/2,                        % plot_spec(+Name, +Config)

    % Curve management
    declare_curve/2,                    % declare_curve(+Name, +Properties)
    declare_plot_spec/2,                % declare_plot_spec(+Name, +Config)
    clear_curves/0,                     % clear_curves

    % Query predicates
    all_curves/1,                       % all_curves(-Curves)
    curves_for_plot/2,                  % curves_for_plot(+Name, -Curves)
    evaluate_curve/3,                   % evaluate_curve(+CurveName, +X, -Y)

    % Code generation
    generate_curve_component/2,         % generate_curve_component(+Name, -Code)
    generate_curve_component/3,         % generate_curve_component(+Name, +Options, -Code)
    generate_curve_styles/2,            % generate_curve_styles(+Name, -CssCode)
    generate_curve_data/3,              % generate_curve_data(+CurveName, +Range, -DataCode)
    generate_chartjs_config/2,          % generate_chartjs_config(+Name, -ConfigCode)

    % Layout-integrated generation
    generate_curve_with_layout/3,       % generate_curve_with_layout(+Name, +LayoutPattern, -Code)
    generate_curve_full_styles/2,       % generate_curve_full_styles(+Name, -CssCode)

    % Testing
    test_curve_plot_generator/0
]).

:- use_module(library(lists)).
:- use_module(layout_generator).
:- use_module(math_expr).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic curve/2.
:- dynamic plot_spec/2.

:- discontiguous curve/2.
:- discontiguous plot_spec/2.
:- discontiguous evaluate_curve/3.

% ============================================================================
% DEFAULT CURVE DEFINITIONS
% ============================================================================

% Linear curves: y = mx + b
curve(identity, [type(linear), m(1), b(0), color('#00d4ff'), label("y = x")]).
curve(line_positive, [type(linear), m(2), b(1), color('#7c3aed'), label("y = 2x + 1")]).
curve(line_negative, [type(linear), m(-1), b(3), color('#f97316'), label("y = -x + 3")]).

% Quadratic curves: y = ax² + bx + c
curve(parabola, [type(quadratic), a(1), b(0), c(0), color('#22c55e'), label("y = x²")]).
curve(parabola_shifted, [type(quadratic), a(1), b(-2), c(1), color('#eab308'), label("y = x² - 2x + 1")]).
curve(parabola_inverted, [type(quadratic), a(-0.5), b(0), c(4), color('#ec4899'), label("y = -0.5x² + 4")]).

% Cubic curves: y = ax³ + bx² + cx + d
curve(cubic, [type(cubic), a(1), b(0), c(0), d(0), color('#14b8a6'), label("y = x³")]).
curve(cubic_wave, [type(cubic), a(0.1), b(0), c(-1), d(0), color('#8b5cf6'), label("y = 0.1x³ - x")]).

% Trigonometric curves
curve(sine_wave, [type(sine), amplitude(1), frequency(1), phase(0), color('#00d4ff'), label("sin(x)")]).
curve(cosine_wave, [type(cosine), amplitude(1), frequency(1), phase(0), color('#7c3aed'), label("cos(x)")]).
curve(sine_double, [type(sine), amplitude(1.5), frequency(2), phase(0), color('#f97316'), label("1.5·sin(2x)")]).
curve(cosine_shifted, [type(cosine), amplitude(1), frequency(1), phase(1.57), color('#22c55e'), label("cos(x + π/2)")]).

% Exponential curves: y = scale * e^(base*x)
curve(exp_growth, [type(exponential), base(1), scale(1), color('#ef4444'), label("e^x")]).
curve(exp_decay, [type(exponential), base(-1), scale(1), color('#3b82f6'), label("e^(-x)")]).

% Absolute value: y = a|x - h| + k
curve(abs_basic, [type(absolute), a(1), h(0), k(0), color('#a855f7'), label("|x|")]).
curve(abs_shifted, [type(absolute), a(2), h(1), k(-1), color('#06b6d4'), label("2|x - 1| - 1")]).

%% Expression-based curves (declarative mathematical expressions)
curve(gaussian, [expr(exp(-(x^2) / 2)), color('#10b981'), label("Gaussian")]).
curve(sinc, [expr(sin(x) / x), color('#f59e0b'), label("sinc(x)")]).  % Note: undefined at x=0
curve(damped_sine, [expr(exp(-abs(x) / 3) * sin(x * 2)), color('#6366f1'), label("Damped sine")]).
curve(polynomial, [expr(x^4 - 2*x^2 + 0.5), color('#ec4899'), label("x⁴ - 2x² + 0.5")]).
curve(rational, [expr(1 / (1 + x^2)), color('#14b8a6'), label("1/(1+x²)")]).  % Cauchy distribution

%% MATLAB-style data point curves
curve(sampled_data, [
    data([-2, -1, 0, 1, 2, 3, 4], [4, 1, 0, 1, 4, 9, 16]),
    color('#f97316'),
    label("Measured Data")
]).
curve(experimental_points, [
    data([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
         [0.0, 0.48, 0.84, 1.0, 0.91, 0.6, 0.14]),
    color('#8b5cf6'),
    label("Experimental")
]).

% ============================================================================
% DEFAULT PLOT SPECIFICATIONS
% ============================================================================

plot_spec(trig_demo, [
    title("Trigonometric Functions"),
    description("Sine and cosine waves"),
    curves([sine_wave, cosine_wave]),
    x_range(-6.28, 6.28),
    y_range(-2, 2),
    theme(dark),
    grid(true),
    legend(true)
]).

plot_spec(polynomial_demo, [
    title("Polynomial Functions"),
    description("Linear, quadratic, and cubic"),
    curves([identity, parabola, cubic]),
    x_range(-3, 3),
    y_range(-5, 10),
    theme(dark),
    grid(true)
]).

plot_spec(exponential_demo, [
    title("Exponential Functions"),
    description("Growth and decay"),
    curves([exp_growth, exp_decay]),
    x_range(-3, 3),
    y_range(0, 10),
    theme(dark)
]).

plot_spec(all_functions, [
    title("Mathematical Functions"),
    description("Overview of common functions"),
    curves([sine_wave, parabola, exp_growth, abs_basic]),
    x_range(-5, 5),
    y_range(-5, 10),
    theme(dark),
    interactive(true)
]).

% ============================================================================
% CURVE MANAGEMENT
% ============================================================================

%% declare_curve(+Name, +Properties)
%  Add or update a curve definition.
declare_curve(Name, Properties) :-
    (   curve(Name, _)
    ->  retract(curve(Name, _))
    ;   true
    ),
    assertz(curve(Name, Properties)).

%% declare_plot_spec(+Name, +Config)
%  Declare a plot specification.
declare_plot_spec(Name, Config) :-
    (   plot_spec(Name, _)
    ->  retract(plot_spec(Name, _))
    ;   true
    ),
    assertz(plot_spec(Name, Config)).

%% clear_curves
%  Clear all curve and plot definitions.
clear_curves :-
    retractall(curve(_, _)),
    retractall(plot_spec(_, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% all_curves(-Curves)
%  Get all defined curve names.
all_curves(Curves) :-
    findall(Name, curve(Name, _), Curves).

%% curves_for_plot(+Name, -Curves)
%  Get curves for a specific plot.
curves_for_plot(Name, Curves) :-
    plot_spec(Name, Config),
    member(curves(Curves), Config).

%% evaluate_curve(+CurveName, +X, -Y)
%  Evaluate a curve at point X.
evaluate_curve(CurveName, X, Y) :-
    curve(CurveName, Props),
    member(type(Type), Props),
    evaluate_curve_type(Type, Props, X, Y).

evaluate_curve_type(linear, Props, X, Y) :-
    (member(m(M), Props) -> true ; M = 1),
    (member(b(B), Props) -> true ; B = 0),
    Y is M * X + B.

evaluate_curve_type(quadratic, Props, X, Y) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(b(B), Props) -> true ; B = 0),
    (member(c(C), Props) -> true ; C = 0),
    Y is A * X * X + B * X + C.

evaluate_curve_type(cubic, Props, X, Y) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(b(B), Props) -> true ; B = 0),
    (member(c(C), Props) -> true ; C = 0),
    (member(d(D), Props) -> true ; D = 0),
    Y is A * X * X * X + B * X * X + C * X + D.

evaluate_curve_type(sine, Props, X, Y) :-
    (member(amplitude(Amp), Props) -> true ; Amp = 1),
    (member(frequency(Freq), Props) -> true ; Freq = 1),
    (member(phase(Phase), Props) -> true ; Phase = 0),
    Y is Amp * sin(Freq * X + Phase).

evaluate_curve_type(cosine, Props, X, Y) :-
    (member(amplitude(Amp), Props) -> true ; Amp = 1),
    (member(frequency(Freq), Props) -> true ; Freq = 1),
    (member(phase(Phase), Props) -> true ; Phase = 0),
    Y is Amp * cos(Freq * X + Phase).

evaluate_curve_type(exponential, Props, X, Y) :-
    (member(base(Base), Props) -> true ; Base = 1),
    (member(scale(Scale), Props) -> true ; Scale = 1),
    Y is Scale * exp(Base * X).

evaluate_curve_type(absolute, Props, X, Y) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(h(H), Props) -> true ; H = 0),
    (member(k(K), Props) -> true ; K = 0),
    Y is A * abs(X - H) + K.

%% Custom expression type - evaluate arbitrary Prolog math expressions
%% Example: curve(my_curve, [type(expr), formula(sin(x)^2 + cos(x)^2)])
evaluate_curve_type(expr, Props, X, Y) :-
    member(formula(Formula), Props),
    evaluate_expr(Formula, X, Y).

%% Also support expr() directly in properties for convenience
%% Example: curve(my_curve, [expr(x^3 - x)])
evaluate_curve(CurveName, X, Y) :-
    curve(CurveName, Props),
    member(expr(Formula), Props),
    !,  % Cut - use expr if present
    evaluate_expr(Formula, X, Y).

%% Data-based curves - MATLAB style: interpolate or return nearest point
%% Example: curve(my_curve, [data([x1, x2, ...], [y1, y2, ...])])
evaluate_curve(CurveName, X, Y) :-
    curve(CurveName, Props),
    member(data(Xs, Ys), Props),
    !,  % Cut - use data if present
    interpolate_data(Xs, Ys, X, Y).

%% Linear interpolation for data curves
interpolate_data(Xs, Ys, X, Y) :-
    find_interval(Xs, Ys, X, X1, Y1, X2, Y2),
    (X2 =:= X1
    ->  Y = Y1  % Avoid division by zero
    ;   T is (X - X1) / (X2 - X1),
        Y is Y1 + T * (Y2 - Y1)
    ).

%% Find the interval containing X
find_interval([X1, X2|_], [Y1, Y2|_], X, X1, Y1, X2, Y2) :-
    X >= X1, X =< X2, !.
find_interval([_|Xs], [_|Ys], X, X1, Y1, X2, Y2) :-
    find_interval(Xs, Ys, X, X1, Y1, X2, Y2).
find_interval([X1], [Y1], _, X1, Y1, X1, Y1) :- !.  % Single point or beyond range
find_interval([X1, X2], [Y1, Y2], _, X1, Y1, X2, Y2) :- !.  % Last interval

%% evaluate_expr(+Formula, +X, -Y)
%% Evaluate a Prolog expression with x bound to value X
evaluate_expr(Formula, X, Y) :-
    substitute_x(Formula, X, SubstFormula),
    Y is SubstFormula.

%% substitute_x(+Expr, +Val, -Result)
%% Replace x with a concrete value in an expression
substitute_x(x, Val, Val) :- !.
substitute_x(y, _, 0) :- !.  % For 2D curves, y defaults to 0
substitute_x(t, _, 0) :- !.  % t defaults to 0
substitute_x(r, Val, Val) :- !.  % r treated as x for curves
substitute_x(pi, _, Result) :- !, Result is pi.
substitute_x(e, _, Result) :- !, Result is e.
substitute_x(N, _, N) :- number(N), !.
substitute_x(A + B, Val, RA + RB) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(A - B, Val, RA - RB) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(A * B, Val, RA * RB) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(A / B, Val, RA / RB) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(A ^ B, Val, RA ^ RB) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(A ** B, Val, RA ** RB) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(-A, Val, -RA) :- !, substitute_x(A, Val, RA).
substitute_x(sin(A), Val, sin(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(cos(A), Val, cos(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(tan(A), Val, tan(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(exp(A), Val, exp(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(log(A), Val, log(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(sqrt(A), Val, sqrt(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(abs(A), Val, abs(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(floor(A), Val, floor(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(ceiling(A), Val, ceiling(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(sinh(A), Val, sinh(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(cosh(A), Val, cosh(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(tanh(A), Val, tanh(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(asin(A), Val, asin(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(acos(A), Val, acos(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(atan(A), Val, atan(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(atan(A, B), Val, atan(RA, RB)) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(sign(A), Val, sign(RA)) :- !, substitute_x(A, Val, RA).
substitute_x(min(A, B), Val, min(RA, RB)) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).
substitute_x(max(A, B), Val, max(RA, RB)) :- !, substitute_x(A, Val, RA), substitute_x(B, Val, RB).

% ============================================================================
% CODE GENERATION - REACT COMPONENT
% ============================================================================

%% generate_curve_component(+Name, -Code)
%  Generate a React component for the plot.
generate_curve_component(Name, Code) :-
    generate_curve_component(Name, [], Code).

%% generate_curve_component(+Name, +Options, -Code)
%  Generate a React component with options.
generate_curve_component(Name, _Options, Code) :-
    plot_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Curve Plot"),
    (member(x_range(XMin, XMax), Config) -> true ; (XMin = -10, XMax = 10)),
    (member(y_range(YMin, YMax), Config) -> true ; (YMin = -10, YMax = 10)),
    (member(interactive(Interactive), Config) -> true ; Interactive = false),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Generate datasets
    curves_for_plot(Name, CurveNames),
    generate_all_datasets(CurveNames, XMin, XMax, DatasetsCode),

    % Generate chart config
    generate_chartjs_config(Name, ChartConfig),

    format(atom(Code),
'// Generated by UnifyWeaver - Curve Plot Component
// Plot: ~w

import React, { useEffect, useRef, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import styles from "./~w.module.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ~wProps {
  className?: string;
  numPoints?: number;
}

// Generate curve data
function generateCurveData(
  evalFn: (x: number) => number,
  xMin: number,
  xMax: number,
  numPoints: number
): { x: number; y: number }[] {
  const step = (xMax - xMin) / numPoints;
  const data: { x: number; y: number }[] = [];
  for (let i = 0; i <= numPoints; i++) {
    const x = xMin + i * step;
    const y = evalFn(x);
    if (isFinite(y)) {
      data.push({ x, y });
    }
  }
  return data;
}

// Curve evaluation functions
~w

export const ~w: React.FC<~wProps> = ({
  className = "",
  numPoints = 200
}) => {
  const [xRange, setXRange] = useState<[number, number]>([~w, ~w]);
  const [yRange, setYRange] = useState<[number, number]>([~w, ~w]);

  const datasets = [
~w
  ];

  const options = {
~w
  };

  const data = {
    datasets: datasets.map(ds => ({
      ...ds,
      data: generateCurveData(ds.evalFn, xRange[0], xRange[1], numPoints)
    }))
  };

  return (
    <div className={`${styles.container} ${className}`}>
      <div className={styles.header}>
        <h2 className={styles.title}>~w</h2>
      </div>

      ~w

      <div className={styles.chartContainer}>
        <Line options={options} data={data} />
      </div>
    </div>
  );
};

export default ~w;
', [Title, ComponentName, ComponentName, DatasetsCode, ComponentName, ComponentName,
    XMin, XMax, YMin, YMax, DatasetsCode, ChartConfig, Title,
    (Interactive == true -> generate_interactive_controls(XMin, XMax, YMin, YMax) ; ""),
    ComponentName]).

%% generate_all_datasets(+CurveNames, +XMin, +XMax, -DatasetsCode)
generate_all_datasets(CurveNames, _XMin, _XMax, DatasetsCode) :-
    findall(DatasetCode, (
        member(CurveName, CurveNames),
        curve(CurveName, Props),
        generate_dataset(CurveName, Props, DatasetCode)
    ), DatasetCodes),
    atomic_list_concat(DatasetCodes, ',\n', DatasetsCode).

%% generate_dataset(+CurveName, +Props, -DatasetCode)

%% Expression-based curves - priority 1
generate_dataset(CurveName, Props, DatasetCode) :-
    member(expr(Formula), Props),
    !,  % Cut for expr mode
    (member(color(Color), Props) -> true ; Color = '#00d4ff'),
    (member(label(Label), Props) -> true ; atom_string(CurveName, Label)),
    % Use math_expr module to translate expression to JavaScript
    expr_to_js(Formula, JSFormula),
    format(atom(DatasetCode),
'    {
      label: "~w",
      borderColor: "~w",
      backgroundColor: "~w",
      fill: false,
      tension: 0.1,
      pointRadius: 0,
      evalFn: (x: number) => ~w
    }', [Label, Color, Color, JSFormula]).

%% Data-based curves (MATLAB-style) - priority 2
generate_dataset(CurveName, Props, DatasetCode) :-
    member(data(Xs, Ys), Props),
    !,  % Cut for data mode
    (member(color(Color), Props) -> true ; Color = '#00d4ff'),
    (member(label(Label), Props) -> true ; atom_string(CurveName, Label)),
    % Generate JS arrays for the data points
    format_js_array(Xs, XsJS),
    format_js_array(Ys, YsJS),
    format(atom(DatasetCode),
'    {
      label: "~w",
      borderColor: "~w",
      backgroundColor: "~w",
      fill: false,
      tension: 0.1,
      pointRadius: 2,
      // Explicit data points
      rawData: { xs: ~w, ys: ~w },
      evalFn: (() => {
        const xs = ~w;
        const ys = ~w;
        return (x: number) => {
          // Linear interpolation
          for (let i = 0; i < xs.length - 1; i++) {
            if (x >= xs[i] && x <= xs[i+1]) {
              const t = (x - xs[i]) / (xs[i+1] - xs[i]);
              return ys[i] + t * (ys[i+1] - ys[i]);
            }
          }
          // Extrapolate at boundaries
          if (x < xs[0]) return ys[0];
          return ys[ys.length - 1];
        };
      })()
    }', [Label, Color, Color, XsJS, YsJS, XsJS, YsJS]).

%% Type-based curves (legacy) - priority 3
generate_dataset(CurveName, Props, DatasetCode) :-
    member(type(Type), Props),
    (member(color(Color), Props) -> true ; Color = '#00d4ff'),
    (member(label(Label), Props) -> true ; atom_string(CurveName, Label)),
    generate_eval_function(Type, Props, EvalFn),
    format(atom(DatasetCode),
'    {
      label: "~w",
      borderColor: "~w",
      backgroundColor: "~w",
      fill: false,
      tension: 0.1,
      pointRadius: 0,
      evalFn: ~w
    }', [Label, Color, Color, EvalFn]).

%% format_js_array for curve_plot_generator
format_js_array(List, JS) :-
    findall(Str, (member(N, List), number_string(N, Str)), Strs),
    atomic_list_concat(Strs, ', ', Inner),
    format(atom(JS), '[~w]', [Inner]).

%% generate_eval_function(+Type, +Props, -EvalFn)
generate_eval_function(linear, Props, EvalFn) :-
    (member(m(M), Props) -> true ; M = 1),
    (member(b(B), Props) -> true ; B = 0),
    format(atom(EvalFn), '(x: number) => ~w * x + ~w', [M, B]).

generate_eval_function(quadratic, Props, EvalFn) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(b(B), Props) -> true ; B = 0),
    (member(c(C), Props) -> true ; C = 0),
    format(atom(EvalFn), '(x: number) => ~w * x * x + ~w * x + ~w', [A, B, C]).

generate_eval_function(cubic, Props, EvalFn) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(b(B), Props) -> true ; B = 0),
    (member(c(C), Props) -> true ; C = 0),
    (member(d(D), Props) -> true ; D = 0),
    format(atom(EvalFn), '(x: number) => ~w * Math.pow(x, 3) + ~w * x * x + ~w * x + ~w', [A, B, C, D]).

generate_eval_function(sine, Props, EvalFn) :-
    (member(amplitude(Amp), Props) -> true ; Amp = 1),
    (member(frequency(Freq), Props) -> true ; Freq = 1),
    (member(phase(Phase), Props) -> true ; Phase = 0),
    format(atom(EvalFn), '(x: number) => ~w * Math.sin(~w * x + ~w)', [Amp, Freq, Phase]).

generate_eval_function(cosine, Props, EvalFn) :-
    (member(amplitude(Amp), Props) -> true ; Amp = 1),
    (member(frequency(Freq), Props) -> true ; Freq = 1),
    (member(phase(Phase), Props) -> true ; Phase = 0),
    format(atom(EvalFn), '(x: number) => ~w * Math.cos(~w * x + ~w)', [Amp, Freq, Phase]).

generate_eval_function(exponential, Props, EvalFn) :-
    (member(base(Base), Props) -> true ; Base = 1),
    (member(scale(Scale), Props) -> true ; Scale = 1),
    format(atom(EvalFn), '(x: number) => ~w * Math.exp(~w * x)', [Scale, Base]).

generate_eval_function(absolute, Props, EvalFn) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(h(H), Props) -> true ; H = 0),
    (member(k(K), Props) -> true ; K = 0),
    format(atom(EvalFn), '(x: number) => ~w * Math.abs(x - ~w) + ~w', [A, H, K]).

%% generate_interactive_controls(+XMin, +XMax, +YMin, +YMax, -ControlsCode)
generate_interactive_controls(_XMin, _XMax, _YMin, _YMax) :-
    format(atom(ControlsCode),
'<div className={styles.controls}>
        <div className={styles.rangeControl}>
          <label>X Range:</label>
          <input
            type="number"
            value={xRange[0]}
            onChange={(e) => setXRange([Number(e.target.value), xRange[1]])}
          />
          <span>to</span>
          <input
            type="number"
            value={xRange[1]}
            onChange={(e) => setXRange([xRange[0], Number(e.target.value)])}
          />
        </div>
        <div className={styles.rangeControl}>
          <label>Y Range:</label>
          <input
            type="number"
            value={yRange[0]}
            onChange={(e) => setYRange([Number(e.target.value), yRange[1]])}
          />
          <span>to</span>
          <input
            type="number"
            value={yRange[1]}
            onChange={(e) => setYRange([yRange[0], Number(e.target.value)])}
          />
        </div>
      </div>', []),
    ControlsCode = ControlsCode.
generate_interactive_controls(_, _, _, _) :- !.

%% generate_chartjs_config(+Name, -ConfigCode)
generate_chartjs_config(Name, ConfigCode) :-
    plot_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = ""),
    (member(grid(ShowGrid), Config) -> true ; ShowGrid = true),
    (member(legend(ShowLegend), Config) -> true ; ShowLegend = true),

    format(atom(ConfigCode),
'    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: ~w,
        position: "top" as const,
      },
      title: {
        display: ~w,
        text: "~w",
      },
    },
    scales: {
      x: {
        type: "linear" as const,
        position: "center" as const,
        grid: {
          display: ~w,
          color: "rgba(255, 255, 255, 0.1)",
        },
        ticks: {
          color: "rgba(255, 255, 255, 0.7)",
        },
        min: xRange[0],
        max: xRange[1],
      },
      y: {
        type: "linear" as const,
        position: "center" as const,
        grid: {
          display: ~w,
          color: "rgba(255, 255, 255, 0.1)",
        },
        ticks: {
          color: "rgba(255, 255, 255, 0.7)",
        },
        min: yRange[0],
        max: yRange[1],
      },
    }', [ShowLegend, (Title \= "" -> true ; false), Title, ShowGrid, ShowGrid]).

% ============================================================================
% CODE GENERATION - CSS STYLES
% ============================================================================

%% generate_curve_styles(+Name, -CssCode)
%  Generate CSS module for the curve component.
generate_curve_styles(Name, CssCode) :-
    plot_spec(Name, Config),
    (member(theme(Theme), Config) -> true ; Theme = dark),
    theme_css(Theme, BgColor, BorderColor, TextColor),

    format(atom(CssCode),
'.container {
  display: flex;
  flex-direction: column;
  height: 100%%;
  background: ~w;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid ~w;
  color: ~w;
}

.header {
  padding: 1rem 1.5rem;
  border-bottom: 1px solid ~w;
}

.title {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
}

.controls {
  display: flex;
  gap: 1.5rem;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid ~w;
  flex-wrap: wrap;
}

.rangeControl {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.rangeControl label {
  font-weight: 500;
  font-size: 0.875rem;
}

.rangeControl input {
  width: 80px;
  padding: 0.375rem 0.5rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: inherit;
  font-size: 0.875rem;
}

.rangeControl span {
  opacity: 0.6;
}

.chartContainer {
  flex: 1;
  min-height: 400px;
  padding: 1rem;
}
', [BgColor, BorderColor, TextColor, BorderColor, BorderColor]).

%% theme_css(+Theme, -BgColor, -BorderColor, -TextColor)
theme_css(dark, 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', 'rgba(255, 255, 255, 0.1)', '#e0e0e0').
theme_css(light, '#ffffff', 'rgba(0, 0, 0, 0.1)', '#1f2937').
theme_css(_, 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', 'rgba(255, 255, 255, 0.1)', '#e0e0e0').

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% pascal_case(+SnakeCase, -PascalCase)
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
% CURVE DATA GENERATION (for testing)
% ============================================================================

%% generate_curve_data(+CurveName, +Range, -DataCode)
%  Generate sample data points for a curve.
generate_curve_data(CurveName, range(XMin, XMax, NumPoints), DataCode) :-
    Step is (XMax - XMin) / NumPoints,
    findall(point(X, Y), (
        between(0, NumPoints, I),
        X is XMin + I * Step,
        evaluate_curve(CurveName, X, Y)
    ), Points),
    format(atom(DataCode), '~w', [Points]).

% ============================================================================
% LAYOUT-INTEGRATED GENERATION
% ============================================================================

%% generate_curve_with_layout(+Name, +LayoutPattern, -Code)
%  Generate a curve plot component using a layout pattern from layout_generator.
%  LayoutPattern is one of: single, sidebar_content, content_sidebar, dashboard
generate_curve_with_layout(Name, LayoutPattern, Code) :-
    plot_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Curve Plot"),
    (member(x_range(XMin, XMax), Config) -> true ; (XMin = -10, XMax = 10)),
    (member(y_range(YMin, YMax), Config) -> true ; (YMin = -10, YMax = 10)),
    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Set up layout for this plot
    setup_curve_layout(Name, LayoutPattern),

    % Generate datasets
    curves_for_plot(Name, CurveNames),
    generate_all_datasets(CurveNames, XMin, XMax, DatasetsCode),

    % Generate chart config
    generate_chartjs_config(Name, ChartConfig),

    % Generate JSX using layout system
    layout_generator:generate_layout_jsx(Name, LayoutJSX),

    % Combine into full component
    format(atom(Code),
'// Generated by UnifyWeaver - Curve Plot Component with Layout
// Plot: ~w
// Layout: ~w

import React, { useEffect, useRef, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import styles from "./~w.module.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ~wProps {
  className?: string;
  numPoints?: number;
}

// Generate curve data
function generateCurveData(
  evalFn: (x: number) => number,
  xMin: number,
  xMax: number,
  numPoints: number
): { x: number; y: number }[] {
  const step = (xMax - xMin) / numPoints;
  const data: { x: number; y: number }[] = [];
  for (let i = 0; i <= numPoints; i++) {
    const x = xMin + i * step;
    const y = evalFn(x);
    if (isFinite(y)) {
      data.push({ x, y });
    }
  }
  return data;
}

// Dataset definitions
const curveDatasets = [
~w
];

export const ~w: React.FC<~wProps> = ({
  className = "",
  numPoints = 200
}) => {
  const [xRange, setXRange] = useState<[number, number]>([~w, ~w]);
  const [yRange, setYRange] = useState<[number, number]>([~w, ~w]);

  const datasets = curveDatasets.map(ds => ({
    ...ds,
    data: generateCurveData(ds.evalFn, xRange[0], xRange[1], numPoints)
  }));

  const options = {
~w
  };

  const data = { datasets };

  return (
    <div className={`${styles.container} ${className}`}>
      ~w
    </div>
  );
};

export default ~w;
', [Title, LayoutPattern, ComponentName, ComponentName, DatasetsCode,
    ComponentName, ComponentName, XMin, XMax, YMin, YMax, ChartConfig, LayoutJSX, ComponentName]).

%% setup_curve_layout(+Name, +LayoutPattern)
%  Configure layout_generator for a curve plot component.
setup_curve_layout(Name, sidebar_content) :-
    layout_generator:declare_layout(Name, grid, [
        areas([["sidebar", "chart"]]),
        columns(["320px", "1fr"]),
        rows(["1fr"]),
        gap("0")
    ]),
    layout_generator:assertz(place(Name, sidebar, [controls])),
    layout_generator:assertz(place(Name, chart, [chart_canvas])).

setup_curve_layout(Name, content_sidebar) :-
    layout_generator:declare_layout(Name, grid, [
        areas([["chart", "sidebar"]]),
        columns(["1fr", "320px"]),
        rows(["1fr"]),
        gap("0")
    ]),
    layout_generator:assertz(place(Name, chart, [chart_canvas])),
    layout_generator:assertz(place(Name, sidebar, [controls])).

setup_curve_layout(Name, dashboard) :-
    layout_generator:declare_layout(Name, grid, [
        areas([
            ["header", "header"],
            ["sidebar", "chart"],
            ["footer", "footer"]
        ]),
        columns(["280px", "1fr"]),
        rows(["60px", "1fr", "40px"]),
        gap("0")
    ]),
    layout_generator:assertz(place(Name, header, [title_bar])),
    layout_generator:assertz(place(Name, sidebar, [controls, curve_list])),
    layout_generator:assertz(place(Name, chart, [chart_canvas])),
    layout_generator:assertz(place(Name, footer, [status_bar])).

setup_curve_layout(Name, single) :-
    layout_generator:declare_layout(Name, grid, [
        areas([["chart"]]),
        columns(["1fr"]),
        rows(["1fr"])
    ]),
    layout_generator:assertz(place(Name, chart, [chart_canvas])).

setup_curve_layout(Name, _) :-
    % Default to single layout
    setup_curve_layout(Name, single).

%% generate_curve_full_styles(+Name, -CssCode)
%  Generate CSS using both layout_generator and curve-specific styles.
generate_curve_full_styles(Name, CssCode) :-
    plot_spec(Name, Config),
    (member(theme(ThemeName), Config) -> true ; ThemeName = dark),

    % Get layout CSS from layout_generator
    (layout_generator:has_layout(Name)
    ->  layout_generator:generate_layout_css(Name, LayoutCSS)
    ;   LayoutCSS = ''
    ),

    % Get theme CSS
    layout_generator:generate_theme_css(ThemeName, ThemeCSS),

    % Curve-specific styles
    format(atom(CurveCSS),
'.chart-canvas {
  width: 100%;
  height: 100%;
  min-height: 400px;
}

.controls {
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.control-group label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary, #888);
}

.control-group input[type="range"] {
  width: 100%;
  accent-color: var(--accent, #00d4ff);
}

.control-group input[type="number"] {
  padding: 0.5rem;
  background: var(--surface, #16213e);
  border: 1px solid var(--border, rgba(255,255,255,0.1));
  border-radius: 4px;
  color: var(--text, #e0e0e0);
  font-size: 0.875rem;
}

.curve-list {
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.curve-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  background: var(--surface, #16213e);
  border-radius: 4px;
}

.curve-color {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.curve-label {
  font-size: 0.875rem;
}

.title-bar {
  display: flex;
  align-items: center;
  padding: 0 1rem;
  background: var(--surface, #16213e);
  border-bottom: 1px solid var(--border, rgba(255,255,255,0.1));
}

.title-bar h1 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
}

.status-bar {
  display: flex;
  align-items: center;
  padding: 0 1rem;
  font-size: 0.75rem;
  color: var(--text-secondary, #888);
  background: var(--surface, #16213e);
  border-top: 1px solid var(--border, rgba(255,255,255,0.1));
}
', []),

    % Combine all CSS
    format(atom(CssCode), '~w~n~n~w~n~n~w', [LayoutCSS, ThemeCSS, CurveCSS]).

% ============================================================================
% TESTING
% ============================================================================

test_curve_plot_generator :-
    format('~n=== Curve Plot Generator Tests ===~n~n'),

    % Test curve queries
    format('Test 1: Curve queries~n'),
    all_curves(Curves),
    length(Curves, CurveCount),
    (CurveCount > 10
    ->  format('  [PASS] Has ~w curves~n', [CurveCount])
    ;   format('  [FAIL] Expected >10 curves, got ~w~n', [CurveCount])
    ),

    % Test curve evaluation
    format('Test 2: Curve evaluation~n'),
    (   evaluate_curve(sine_wave, 0, Y1),
        abs(Y1) < 0.001
    ->  format('  [PASS] sin(0) ≈ 0~n')
    ;   format('  [FAIL] sin(0) evaluation failed~n')
    ),
    (   evaluate_curve(parabola, 2, Y2),
        abs(Y2 - 4) < 0.001
    ->  format('  [PASS] x² at x=2 = 4~n')
    ;   format('  [FAIL] parabola evaluation failed~n')
    ),

    % Test plot spec query
    format('Test 3: Plot spec query~n'),
    (plot_spec(trig_demo, _)
    ->  format('  [PASS] trig_demo spec exists~n')
    ;   format('  [FAIL] trig_demo spec not found~n')
    ),

    % Test component generation
    format('Test 4: Component generation~n'),
    generate_curve_component(trig_demo, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 3000
    ->  format('  [PASS] Generated ~w chars~n', [CodeLen])
    ;   format('  [FAIL] Code too short: ~w~n', [CodeLen])
    ),

    % Test Chart.js config
    format('Test 5: Chart.js config~n'),
    generate_chartjs_config(trig_demo, ChartConfig),
    (sub_atom(ChartConfig, _, _, _, 'scales')
    ->  format('  [PASS] Contains scales config~n')
    ;   format('  [FAIL] Missing scales config~n')
    ),

    % Test CSS generation
    format('Test 6: CSS generation~n'),
    generate_curve_styles(trig_demo, CssCode),
    (sub_atom(CssCode, _, _, _, '.chartContainer')
    ->  format('  [PASS] Contains .chartContainer class~n')
    ;   format('  [FAIL] Missing .chartContainer class~n')
    ),

    format('~n=== Tests Complete ===~n').
