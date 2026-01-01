% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Matplotlib Generator - Declarative Python Plotting
%
% This module provides declarative plot definitions that generate
% Python code using matplotlib for visualization.
%
% Usage:
%   % Define a matplotlib plot
%   matplotlib_plot(wave_analysis, [
%       title("Wave Analysis"),
%       curves([sine_wave, cosine_wave]),
%       x_range(-6.28, 6.28),
%       style(seaborn),
%       output(png, "waves.png")
%   ]).
%
%   % Generate Python code
%   ?- generate_matplotlib_code(wave_analysis, PythonCode).

:- module(matplotlib_generator, [
    % Plot definition predicates
    matplotlib_plot/2,                  % matplotlib_plot(+Name, +Config)
    matplotlib_curve/2,                 % matplotlib_curve(+Name, +Properties)

    % Plot management
    declare_matplotlib_plot/2,          % declare_matplotlib_plot(+Name, +Config)
    declare_matplotlib_curve/2,         % declare_matplotlib_curve(+Name, +Properties)
    clear_matplotlib/0,                 % clear_matplotlib

    % Query predicates
    all_matplotlib_plots/1,             % all_matplotlib_plots(-Plots)
    all_matplotlib_curves/1,            % all_matplotlib_curves(-Curves)

    % Code generation
    generate_matplotlib_code/2,         % generate_matplotlib_code(+Name, -Code)
    generate_matplotlib_code/3,         % generate_matplotlib_code(+Name, +Options, -Code)
    generate_matplotlib_script/2,       % generate_matplotlib_script(+Name, -ScriptCode)
    generate_numpy_arrays/3,            % generate_numpy_arrays(+CurveName, +Range, -Code)

    % Testing
    test_matplotlib_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic matplotlib_plot/2.
:- dynamic matplotlib_curve/2.

:- discontiguous matplotlib_plot/2.
:- discontiguous matplotlib_curve/2.

% ============================================================================
% DEFAULT CURVE DEFINITIONS (shared with curve_plot_generator)
% ============================================================================

% Trigonometric curves
matplotlib_curve(sine, [type(sine), amplitude(1), frequency(1), phase(0), color('blue'), linestyle('-'), label("sin(x)")]).
matplotlib_curve(cosine, [type(cosine), amplitude(1), frequency(1), phase(0), color('orange'), linestyle('-'), label("cos(x)")]).
matplotlib_curve(tangent, [type(tangent), amplitude(1), frequency(1), phase(0), color('green'), linestyle('-'), label("tan(x)")]).

% Polynomial curves
matplotlib_curve(linear, [type(linear), m(1), b(0), color('red'), linestyle('-'), label("y = x")]).
matplotlib_curve(quadratic, [type(quadratic), a(1), b(0), c(0), color('purple'), linestyle('-'), label("y = x²")]).
matplotlib_curve(cubic, [type(cubic), a(1), b(0), c(0), d(0), color('brown'), linestyle('-'), label("y = x³")]).

% Exponential and logarithmic
matplotlib_curve(exponential, [type(exponential), base(1), scale(1), color('cyan'), linestyle('-'), label("e^x")]).
matplotlib_curve(logarithm, [type(logarithm), base(e), color('magenta'), linestyle('-'), label("ln(x)")]).

% Special functions
matplotlib_curve(gaussian, [type(gaussian), mu(0), sigma(1), color('navy'), linestyle('-'), label("Gaussian")]).
matplotlib_curve(sigmoid, [type(sigmoid), color('darkgreen'), linestyle('-'), label("Sigmoid")]).

% ============================================================================
% DEFAULT PLOT SPECIFICATIONS
% ============================================================================

matplotlib_plot(trig_functions, [
    title("Trigonometric Functions"),
    curves([sine, cosine]),
    x_range(-6.28318, 6.28318),
    y_range(-1.5, 1.5),
    style(seaborn),
    figsize(10, 6),
    grid(true),
    legend(true),
    xlabel("x"),
    ylabel("y"),
    output(show)
]).

matplotlib_plot(polynomial_comparison, [
    title("Polynomial Functions"),
    curves([linear, quadratic, cubic]),
    x_range(-3, 3),
    y_range(-10, 10),
    style(ggplot),
    figsize(10, 6),
    grid(true),
    legend(true),
    output(show)
]).

matplotlib_plot(exponential_analysis, [
    title("Exponential and Logarithmic"),
    curves([exponential, logarithm]),
    x_range(0.1, 5),
    y_range(-2, 10),
    style(seaborn),
    figsize(10, 6),
    grid(true),
    output(png, "exp_log.png")
]).

matplotlib_plot(bell_curve, [
    title("Gaussian Distribution"),
    curves([gaussian]),
    x_range(-4, 4),
    y_range(0, 0.5),
    style(classic),
    figsize(8, 6),
    grid(true),
    fill(true),
    alpha(0.3),
    output(show)
]).

matplotlib_plot(all_functions, [
    title("Mathematical Functions Overview"),
    curves([sine, quadratic, exponential, sigmoid]),
    x_range(-5, 5),
    y_range(-5, 10),
    style(dark_background),
    figsize(12, 8),
    grid(true),
    legend(true),
    output(show)
]).

% ============================================================================
% PLOT MANAGEMENT
% ============================================================================

%% declare_matplotlib_plot(+Name, +Config)
%  Add or update a matplotlib plot definition.
declare_matplotlib_plot(Name, Config) :-
    (   matplotlib_plot(Name, _)
    ->  retract(matplotlib_plot(Name, _))
    ;   true
    ),
    assertz(matplotlib_plot(Name, Config)).

%% declare_matplotlib_curve(+Name, +Properties)
%  Add or update a matplotlib curve definition.
declare_matplotlib_curve(Name, Properties) :-
    (   matplotlib_curve(Name, _)
    ->  retract(matplotlib_curve(Name, _))
    ;   true
    ),
    assertz(matplotlib_curve(Name, Properties)).

%% clear_matplotlib
%  Clear all matplotlib definitions.
clear_matplotlib :-
    retractall(matplotlib_plot(_, _)),
    retractall(matplotlib_curve(_, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% all_matplotlib_plots(-Plots)
%  Get all defined matplotlib plot names.
all_matplotlib_plots(Plots) :-
    findall(Name, matplotlib_plot(Name, _), Plots).

%% all_matplotlib_curves(-Curves)
%  Get all defined matplotlib curve names.
all_matplotlib_curves(Curves) :-
    findall(Name, matplotlib_curve(Name, _), Curves).

% ============================================================================
% CODE GENERATION - PYTHON/MATPLOTLIB
% ============================================================================

%% generate_matplotlib_code(+Name, -Code)
%  Generate Python code for a matplotlib plot.
generate_matplotlib_code(Name, Code) :-
    generate_matplotlib_code(Name, [], Code).

%% generate_matplotlib_code(+Name, +Options, -Code)
%  Generate Python code with options.
generate_matplotlib_code(Name, _Options, Code) :-
    matplotlib_plot(Name, Config),
    (member(title(Title), Config) -> true ; Title = "Plot"),
    (member(x_range(XMin, XMax), Config) -> true ; (XMin = -10, XMax = 10)),
    (member(y_range(YMin, YMax), Config) -> true ; (YMin = -10, YMax = 10)),
    (member(style(Style), Config) -> true ; Style = seaborn),
    (member(figsize(FigW, FigH), Config) -> true ; (FigW = 10, FigH = 6)),
    (member(grid(ShowGrid), Config) -> true ; ShowGrid = true),
    (member(legend(ShowLegend), Config) -> true ; ShowLegend = true),
    (member(xlabel(XLabel), Config) -> true ; XLabel = "x"),
    (member(ylabel(YLabel), Config) -> true ; YLabel = "y"),
    (member(fill(Fill), Config) -> true ; Fill = false),
    (member(alpha(Alpha), Config) -> true ; Alpha = 1.0),
    (member(output(Output), Config) -> true ; Output = show),
    (member(curves(CurveNames), Config) -> true ; CurveNames = []),

    % Generate curve plotting code
    generate_curve_plots(CurveNames, Fill, Alpha, CurvePlotCode),

    % Generate output code
    generate_output_code(Output, OutputCode),

    format(atom(Code),
'#!/usr/bin/env python3
"""
Generated by UnifyWeaver - Matplotlib Plot
Plot: ~w
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use("~w")

# Create figure
fig, ax = plt.subplots(figsize=(~w, ~w))

# Generate x values
x = np.linspace(~w, ~w, 1000)

# Plot curves
~w

# Configure axes
ax.set_xlim(~w, ~w)
ax.set_ylim(~w, ~w)
ax.set_xlabel("~w")
ax.set_ylabel("~w")
ax.set_title("~w")
ax.grid(~w)
~w

# Output
~w
', [Name, Style, FigW, FigH, XMin, XMax, CurvePlotCode,
    XMin, XMax, YMin, YMax, XLabel, YLabel, Title,
    (ShowGrid == true -> 'True' ; 'False'),
    (ShowLegend == true -> 'ax.legend()' ; ''),
    OutputCode]).

%% generate_curve_plots(+CurveNames, +Fill, +Alpha, -Code)
generate_curve_plots(CurveNames, Fill, Alpha, Code) :-
    findall(PlotCode, (
        member(CurveName, CurveNames),
        matplotlib_curve(CurveName, Props),
        generate_single_curve_plot(CurveName, Props, Fill, Alpha, PlotCode)
    ), PlotCodes),
    atomic_list_concat(PlotCodes, '\n', Code).

%% generate_single_curve_plot(+CurveName, +Props, +Fill, +Alpha, -Code)
generate_single_curve_plot(CurveName, Props, Fill, Alpha, Code) :-
    member(type(Type), Props),
    (member(color(Color), Props) -> true ; Color = 'blue'),
    (member(linestyle(LineStyle), Props) -> true ; LineStyle = '-'),
    (member(label(Label), Props) -> true ; atom_string(CurveName, Label)),

    generate_numpy_expression(Type, Props, NumpyExpr),
    atom_string(CurveName, CurveNameStr),

    (Fill == true
    ->  format(atom(Code),
'y_~w = ~w
ax.plot(x, y_~w, color="~w", linestyle="~w", label="~w")
ax.fill_between(x, y_~w, alpha=~w, color="~w")',
            [CurveNameStr, NumpyExpr, CurveNameStr, Color, LineStyle, Label, CurveNameStr, Alpha, Color])
    ;   format(atom(Code),
'y_~w = ~w
ax.plot(x, y_~w, color="~w", linestyle="~w", label="~w")',
            [CurveNameStr, NumpyExpr, CurveNameStr, Color, LineStyle, Label])
    ).

%% generate_numpy_expression(+Type, +Props, -Expr)
generate_numpy_expression(linear, Props, Expr) :-
    (member(m(M), Props) -> true ; M = 1),
    (member(b(B), Props) -> true ; B = 0),
    format(atom(Expr), '~w * x + ~w', [M, B]).

generate_numpy_expression(quadratic, Props, Expr) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(b(B), Props) -> true ; B = 0),
    (member(c(C), Props) -> true ; C = 0),
    format(atom(Expr), '~w * x**2 + ~w * x + ~w', [A, B, C]).

generate_numpy_expression(cubic, Props, Expr) :-
    (member(a(A), Props) -> true ; A = 1),
    (member(b(B), Props) -> true ; B = 0),
    (member(c(C), Props) -> true ; C = 0),
    (member(d(D), Props) -> true ; D = 0),
    format(atom(Expr), '~w * x**3 + ~w * x**2 + ~w * x + ~w', [A, B, C, D]).

generate_numpy_expression(sine, Props, Expr) :-
    (member(amplitude(Amp), Props) -> true ; Amp = 1),
    (member(frequency(Freq), Props) -> true ; Freq = 1),
    (member(phase(Phase), Props) -> true ; Phase = 0),
    format(atom(Expr), '~w * np.sin(~w * x + ~w)', [Amp, Freq, Phase]).

generate_numpy_expression(cosine, Props, Expr) :-
    (member(amplitude(Amp), Props) -> true ; Amp = 1),
    (member(frequency(Freq), Props) -> true ; Freq = 1),
    (member(phase(Phase), Props) -> true ; Phase = 0),
    format(atom(Expr), '~w * np.cos(~w * x + ~w)', [Amp, Freq, Phase]).

generate_numpy_expression(tangent, Props, Expr) :-
    (member(amplitude(Amp), Props) -> true ; Amp = 1),
    (member(frequency(Freq), Props) -> true ; Freq = 1),
    (member(phase(Phase), Props) -> true ; Phase = 0),
    format(atom(Expr), '~w * np.tan(~w * x + ~w)', [Amp, Freq, Phase]).

generate_numpy_expression(exponential, Props, Expr) :-
    (member(base(Base), Props) -> true ; Base = 1),
    (member(scale(Scale), Props) -> true ; Scale = 1),
    format(atom(Expr), '~w * np.exp(~w * x)', [Scale, Base]).

generate_numpy_expression(logarithm, Props, Expr) :-
    (member(base(Base), Props) -> true ; Base = e),
    (Base == e
    ->  Expr = 'np.log(x)'
    ;   Base == 10
    ->  Expr = 'np.log10(x)'
    ;   format(atom(Expr), 'np.log(x) / np.log(~w)', [Base])
    ).

generate_numpy_expression(gaussian, Props, Expr) :-
    (member(mu(Mu), Props) -> true ; Mu = 0),
    (member(sigma(Sigma), Props) -> true ; Sigma = 1),
    format(atom(Expr), '(1/(~w * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-~w)/~w)**2)', [Sigma, Mu, Sigma]).

generate_numpy_expression(sigmoid, _Props, Expr) :-
    Expr = '1 / (1 + np.exp(-x))'.

%% generate_output_code(+Output, -Code)
generate_output_code(show, 'plt.show()').
generate_output_code(png(Filename), Code) :-
    format(atom(Code), 'plt.savefig("~w", dpi=150, bbox_inches="tight")~nplt.close()', [Filename]).
generate_output_code(pdf(Filename), Code) :-
    format(atom(Code), 'plt.savefig("~w", format="pdf", bbox_inches="tight")~nplt.close()', [Filename]).
generate_output_code(svg(Filename), Code) :-
    format(atom(Code), 'plt.savefig("~w", format="svg", bbox_inches="tight")~nplt.close()', [Filename]).

% ============================================================================
% SCRIPT GENERATION
% ============================================================================

%% generate_matplotlib_script(+Name, -ScriptCode)
%  Generate a complete Python script for the plot.
generate_matplotlib_script(Name, ScriptCode) :-
    generate_matplotlib_code(Name, Code),
    format(atom(ScriptCode),
'~w

if __name__ == "__main__":
    print("Plot generated successfully!")
', [Code]).

%% generate_numpy_arrays(+CurveName, +Range, -Code)
%  Generate NumPy array code for a curve.
generate_numpy_arrays(CurveName, range(XMin, XMax, NumPoints), Code) :-
    matplotlib_curve(CurveName, Props),
    member(type(Type), Props),
    generate_numpy_expression(Type, Props, Expr),
    atom_string(CurveName, CurveNameStr),
    format(atom(Code),
'x_~w = np.linspace(~w, ~w, ~w)
y_~w = ~w', [CurveNameStr, XMin, XMax, NumPoints, CurveNameStr, Expr]).

% ============================================================================
% TESTING
% ============================================================================

test_matplotlib_generator :-
    format('~n=== Matplotlib Generator Tests ===~n~n'),

    % Test curve queries
    format('Test 1: Curve queries~n'),
    all_matplotlib_curves(Curves),
    length(Curves, CurveCount),
    (CurveCount >= 10
    ->  format('  [PASS] Has ~w curves~n', [CurveCount])
    ;   format('  [FAIL] Expected >=10 curves, got ~w~n', [CurveCount])
    ),

    % Test plot queries
    format('Test 2: Plot queries~n'),
    all_matplotlib_plots(Plots),
    length(Plots, PlotCount),
    (PlotCount >= 4
    ->  format('  [PASS] Has ~w plots~n', [PlotCount])
    ;   format('  [FAIL] Expected >=4 plots, got ~w~n', [PlotCount])
    ),

    % Test code generation
    format('Test 3: Code generation~n'),
    generate_matplotlib_code(trig_functions, Code),
    atom_length(Code, CodeLen),
    (CodeLen > 500
    ->  format('  [PASS] Generated ~w chars~n', [CodeLen])
    ;   format('  [FAIL] Code too short: ~w~n', [CodeLen])
    ),

    % Test numpy expression generation
    format('Test 4: NumPy expression generation~n'),
    matplotlib_curve(sine, SineProps),
    generate_numpy_expression(sine, SineProps, SineExpr),
    (sub_atom(SineExpr, _, _, _, 'np.sin')
    ->  format('  [PASS] Sine uses np.sin~n')
    ;   format('  [FAIL] Sine expression incorrect~n')
    ),

    % Test output code
    format('Test 5: Output code generation~n'),
    generate_output_code(png("test.png"), PngCode),
    (sub_atom(PngCode, _, _, _, 'savefig')
    ->  format('  [PASS] PNG output uses savefig~n')
    ;   format('  [FAIL] PNG output incorrect~n')
    ),

    % Test imports
    format('Test 6: Imports in generated code~n'),
    generate_matplotlib_code(polynomial_comparison, PolyCode),
    (sub_atom(PolyCode, _, _, _, 'import numpy')
    ->  format('  [PASS] Contains numpy import~n')
    ;   format('  [FAIL] Missing numpy import~n')
    ),
    (sub_atom(PolyCode, _, _, _, 'import matplotlib')
    ->  format('  [PASS] Contains matplotlib import~n')
    ;   format('  [FAIL] Missing matplotlib import~n')
    ),

    format('~n=== Tests Complete ===~n').
