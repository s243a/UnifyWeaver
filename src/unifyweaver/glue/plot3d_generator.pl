% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% 3D Plot Generator - Declarative 3D Visualization
%
% This module provides declarative 3D plot definitions that generate
% TypeScript/React components using Plotly.js and Python using matplotlib/plotly.
%
% Usage:
%   % Define a 3D surface
%   surface3d(my_surface, [
%       function(sin_cos),      % z = sin(x) * cos(y)
%       x_range(-3.14, 3.14),
%       y_range(-3.14, 3.14),
%       colorscale(viridis)
%   ]).
%
%   % Define 3D scatter points
%   scatter3d_point(my_scatter, 1.0, 2.0, 3.0, [label("Point A")]).
%
%   % Generate React component
%   ?- generate_plot3d_component(my_surface, Code).

:- module(plot3d_generator, [
    % 3D definition predicates
    surface3d/2,                        % surface3d(+Name, +Config)
    scatter3d_point/5,                  % scatter3d_point(+Name, +X, +Y, +Z, +Props)
    scatter3d_spec/2,                   % scatter3d_spec(+Name, +Config)
    line3d_point/5,                     % line3d_point(+Name, +Index, +X, +Y, +Z)
    line3d_spec/2,                      % line3d_spec(+Name, +Config)

    % Management
    declare_surface3d/2,                % declare_surface3d(+Name, +Config)
    declare_scatter3d_point/5,          % declare_scatter3d_point(+Name, +X, +Y, +Z, +Props)
    clear_plot3d/0,                     % clear_plot3d
    clear_plot3d/1,                     % clear_plot3d(+Name)

    % Query predicates
    all_plot3d/1,                       % all_plot3d(-Names)
    plot3d_type/2,                      % plot3d_type(+Name, -Type)

    % Code generation - React/Plotly.js
    generate_plot3d_component/2,        % generate_plot3d_component(+Name, -Code)
    generate_plot3d_component/3,        % generate_plot3d_component(+Name, +Options, -Code)
    generate_plot3d_styles/2,           % generate_plot3d_styles(+Name, -CssCode)

    % Python generation
    generate_plot3d_matplotlib/2,       % generate_plot3d_matplotlib(+Name, -PythonCode)
    generate_plot3d_plotly/2,           % generate_plot3d_plotly(+Name, -PythonCode)

    % Layout-integrated generation
    generate_plot3d_with_layout/3,      % generate_plot3d_with_layout(+Name, +LayoutPattern, -Code)

    % Testing
    test_plot3d_generator/0
]).

:- use_module(library(lists)).
:- use_module(layout_generator).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic surface3d/2.
:- dynamic scatter3d_point/5.
:- dynamic scatter3d_spec/2.
:- dynamic line3d_point/5.
:- dynamic line3d_spec/2.

:- discontiguous surface3d/2.
:- discontiguous scatter3d_point/5.
:- discontiguous scatter3d_spec/2.
:- discontiguous line3d_point/5.
:- discontiguous line3d_spec/2.

% ============================================================================
% DEFAULT 3D PLOT DEFINITIONS
% ============================================================================

% 3D Surface: sin(x) * cos(y)
surface3d(wave_surface, [
    title("3D Wave Surface"),
    function(sin_cos),
    x_range(-3.14159, 3.14159),
    y_range(-3.14159, 3.14159),
    resolution(50),
    colorscale(viridis),
    theme(dark)
]).

% 3D Surface: Paraboloid z = x^2 + y^2
surface3d(paraboloid, [
    title("Paraboloid"),
    function(paraboloid),
    x_range(-2, 2),
    y_range(-2, 2),
    resolution(40),
    colorscale(plasma),
    theme(dark)
]).

% 3D Surface: Saddle z = x^2 - y^2
surface3d(saddle, [
    title("Saddle Surface"),
    function(saddle),
    x_range(-2, 2),
    y_range(-2, 2),
    resolution(40),
    colorscale(cool),
    theme(dark)
]).

% 3D Scatter example
scatter3d_spec(cluster_demo, [
    title("3D Cluster Visualization"),
    marker_size(8),
    colorscale(portland),
    theme(dark)
]).

scatter3d_point(cluster_demo, 1.0, 1.0, 1.0, [label("A1"), cluster(1)]).
scatter3d_point(cluster_demo, 1.2, 0.9, 1.1, [label("A2"), cluster(1)]).
scatter3d_point(cluster_demo, 0.8, 1.1, 0.9, [label("A3"), cluster(1)]).
scatter3d_point(cluster_demo, -1.0, -1.0, 1.0, [label("B1"), cluster(2)]).
scatter3d_point(cluster_demo, -0.9, -1.1, 1.2, [label("B2"), cluster(2)]).
scatter3d_point(cluster_demo, -1.1, -0.9, 0.8, [label("B3"), cluster(2)]).
scatter3d_point(cluster_demo, 0.0, 0.0, -1.5, [label("C1"), cluster(3)]).
scatter3d_point(cluster_demo, 0.1, -0.1, -1.4, [label("C2"), cluster(3)]).

% 3D Line/trajectory example
line3d_spec(helix, [
    title("3D Helix"),
    line_width(3),
    color('#00d4ff'),
    theme(dark)
]).

% Helix points (parametric: x=cos(t), y=sin(t), z=t/5)
line3d_point(helix, 0, 1.0, 0.0, 0.0).
line3d_point(helix, 1, 0.809, 0.588, 0.2).
line3d_point(helix, 2, 0.309, 0.951, 0.4).
line3d_point(helix, 3, -0.309, 0.951, 0.6).
line3d_point(helix, 4, -0.809, 0.588, 0.8).
line3d_point(helix, 5, -1.0, 0.0, 1.0).
line3d_point(helix, 6, -0.809, -0.588, 1.2).
line3d_point(helix, 7, -0.309, -0.951, 1.4).
line3d_point(helix, 8, 0.309, -0.951, 1.6).
line3d_point(helix, 9, 0.809, -0.588, 1.8).
line3d_point(helix, 10, 1.0, 0.0, 2.0).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_surface3d(+Name, +Config)
declare_surface3d(Name, Config) :-
    retractall(surface3d(Name, _)),
    assertz(surface3d(Name, Config)).

%% declare_scatter3d_point(+Name, +X, +Y, +Z, +Props)
declare_scatter3d_point(Name, X, Y, Z, Props) :-
    assertz(scatter3d_point(Name, X, Y, Z, Props)).

%% clear_plot3d
clear_plot3d :-
    retractall(surface3d(_, _)),
    retractall(scatter3d_point(_, _, _, _, _)),
    retractall(scatter3d_spec(_, _)),
    retractall(line3d_point(_, _, _, _, _)),
    retractall(line3d_spec(_, _)).

%% clear_plot3d(+Name)
clear_plot3d(Name) :-
    retractall(surface3d(Name, _)),
    retractall(scatter3d_point(Name, _, _, _, _)),
    retractall(scatter3d_spec(Name, _)),
    retractall(line3d_point(Name, _, _, _, _)),
    retractall(line3d_spec(Name, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% all_plot3d(-Names)
all_plot3d(Names) :-
    findall(Name, surface3d(Name, _), SurfaceNames),
    findall(Name, scatter3d_spec(Name, _), ScatterNames),
    findall(Name, line3d_spec(Name, _), LineNames),
    append([SurfaceNames, ScatterNames, LineNames], AllNames),
    list_to_set(AllNames, Names).

%% plot3d_type(+Name, -Type)
plot3d_type(Name, surface) :- surface3d(Name, _), !.
plot3d_type(Name, scatter) :- scatter3d_spec(Name, _), !.
plot3d_type(Name, line) :- line3d_spec(Name, _), !.

% ============================================================================
% CODE GENERATION - REACT/PLOTLY.JS
% ============================================================================

%% generate_plot3d_component(+Name, -Code)
generate_plot3d_component(Name, Code) :-
    generate_plot3d_component(Name, [], Code).

%% generate_plot3d_component(+Name, +Options, -Code)
generate_plot3d_component(Name, _Options, Code) :-
    plot3d_type(Name, Type),
    generate_plot3d_by_type(Type, Name, Code).

%% generate_plot3d_by_type(+Type, +Name, -Code)
generate_plot3d_by_type(surface, Name, Code) :-
    surface3d(Name, Config),
    (member(title(Title), Config) -> true ; Title = "3D Surface"),
    (member(function(Func), Config) -> true ; Func = sin_cos),
    (member(x_range(XMin, XMax), Config) -> true ; (XMin = -3.14, XMax = 3.14)),
    (member(y_range(YMin, YMax), Config) -> true ; (YMin = -3.14, YMax = 3.14)),
    (member(resolution(Res), Config) -> true ; Res = 50),
    (member(colorscale(ColorScale), Config) -> true ; ColorScale = viridis),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),
    get_surface_function_js(Func, FuncJS),
    atom_string(ColorScale, ColorScaleStr),

    format(atom(Code),
'// Generated by UnifyWeaver - 3D Surface Plot
// Plot: ~w

import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import styles from "./~w.module.css";

interface SurfaceProps {
  width?: number;
  height?: number;
}

export const ~w: React.FC<SurfaceProps> = ({ width = 800, height = 600 }) => {
  const data = useMemo(() => {
    const xRange = { min: ~w, max: ~w };
    const yRange = { min: ~w, max: ~w };
    const resolution = ~w;

    const x: number[] = [];
    const y: number[] = [];
    const z: number[][] = [];

    for (let i = 0; i < resolution; i++) {
      const xVal = xRange.min + (xRange.max - xRange.min) * i / (resolution - 1);
      x.push(xVal);
    }

    for (let j = 0; j < resolution; j++) {
      const yVal = yRange.min + (yRange.max - yRange.min) * j / (resolution - 1);
      y.push(yVal);
      const row: number[] = [];
      for (let i = 0; i < resolution; i++) {
        const xVal = x[i];
        ~w
        row.push(zVal);
      }
      z.push(row);
    }

    return [
      {
        type: "surface" as const,
        x,
        y,
        z,
        colorscale: "~w",
        showscale: true,
      },
    ];
  }, []);

  const layout = useMemo(
    () => ({
      title: "~w",
      autosize: false,
      width,
      height,
      scene: {
        xaxis: { title: "X" },
        yaxis: { title: "Y" },
        zaxis: { title: "Z" },
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.2 },
        },
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#e0e0e0" },
    }),
    [width, height]
  );

  return (
    <div className={styles.plotContainer}>
      <Plot data={data} layout={layout} config={{ responsive: true }} />
    </div>
  );
};

export default ~w;
', [Name, ComponentName, ComponentName, XMin, XMax, YMin, YMax, Res, FuncJS, ColorScaleStr, Title, ComponentName]).

generate_plot3d_by_type(scatter, Name, Code) :-
    scatter3d_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "3D Scatter"),
    (member(marker_size(MarkerSize), Config) -> true ; MarkerSize = 8),
    (member(colorscale(ColorScale), Config) -> true ; ColorScale = portland),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),
    atom_string(ColorScale, ColorScaleStr),

    % Collect points
    findall(point(X, Y, Z, Props), scatter3d_point(Name, X, Y, Z, Props), Points),
    generate_scatter_data_js(Points, DataJS),

    format(atom(Code),
'// Generated by UnifyWeaver - 3D Scatter Plot
// Plot: ~w

import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import styles from "./~w.module.css";

interface ScatterProps {
  width?: number;
  height?: number;
  onPointClick?: (index: number) => void;
}

export const ~w: React.FC<ScatterProps> = ({ width = 800, height = 600, onPointClick }) => {
  const { x, y, z, labels, clusters } = useMemo(() => {
    ~w
    return { x, y, z, labels, clusters };
  }, []);

  const data = useMemo(
    () => [
      {
        type: "scatter3d" as const,
        mode: "markers" as const,
        x,
        y,
        z,
        text: labels,
        marker: {
          size: ~w,
          color: clusters,
          colorscale: "~w",
          opacity: 0.8,
        },
        hoverinfo: "text",
      },
    ],
    [x, y, z, labels, clusters]
  );

  const layout = useMemo(
    () => ({
      title: "~w",
      autosize: false,
      width,
      height,
      scene: {
        xaxis: { title: "X" },
        yaxis: { title: "Y" },
        zaxis: { title: "Z" },
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#e0e0e0" },
    }),
    [width, height]
  );

  return (
    <div className={styles.plotContainer}>
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true }}
        onClick={(event) => onPointClick?.(event.points[0]?.pointIndex)}
      />
    </div>
  );
};

export default ~w;
', [Name, ComponentName, ComponentName, DataJS, MarkerSize, ColorScaleStr, Title, ComponentName]).

generate_plot3d_by_type(line, Name, Code) :-
    line3d_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "3D Line"),
    (member(line_width(LineWidth), Config) -> true ; LineWidth = 3),
    (member(color(Color), Config) -> true ; Color = '#00d4ff'),

    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Collect points in order
    findall(idx_point(Idx, X, Y, Z), line3d_point(Name, Idx, X, Y, Z), Points),
    sort(Points, SortedPoints),
    generate_line_data_js(SortedPoints, DataJS),

    format(atom(Code),
'// Generated by UnifyWeaver - 3D Line Plot
// Plot: ~w

import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import styles from "./~w.module.css";

interface LineProps {
  width?: number;
  height?: number;
}

export const ~w: React.FC<LineProps> = ({ width = 800, height = 600 }) => {
  const { x, y, z } = useMemo(() => {
    ~w
    return { x, y, z };
  }, []);

  const data = useMemo(
    () => [
      {
        type: "scatter3d" as const,
        mode: "lines" as const,
        x,
        y,
        z,
        line: {
          width: ~w,
          color: "~w",
        },
      },
    ],
    [x, y, z]
  );

  const layout = useMemo(
    () => ({
      title: "~w",
      autosize: false,
      width,
      height,
      scene: {
        xaxis: { title: "X" },
        yaxis: { title: "Y" },
        zaxis: { title: "Z" },
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.0 },
        },
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#e0e0e0" },
    }),
    [width, height]
  );

  return (
    <div className={styles.plotContainer}>
      <Plot data={data} layout={layout} config={{ responsive: true }} />
    </div>
  );
};

export default ~w;
', [Name, ComponentName, ComponentName, DataJS, LineWidth, Color, Title, ComponentName]).

%% get_surface_function_js(+Func, -JS)
get_surface_function_js(sin_cos, 'const zVal = Math.sin(xVal) * Math.cos(yVal);').
get_surface_function_js(paraboloid, 'const zVal = xVal * xVal + yVal * yVal;').
get_surface_function_js(saddle, 'const zVal = xVal * xVal - yVal * yVal;').
get_surface_function_js(ripple, 'const r = Math.sqrt(xVal * xVal + yVal * yVal); const zVal = Math.sin(r * 3) / (r + 0.1);').
get_surface_function_js(gaussian, 'const zVal = Math.exp(-(xVal * xVal + yVal * yVal) / 2);').
get_surface_function_js(_, JS) :- get_surface_function_js(sin_cos, JS).

%% generate_scatter_data_js(+Points, -JS)
generate_scatter_data_js(Points, JS) :-
    findall(X, member(point(X, _, _, _), Points), Xs),
    findall(Y, member(point(_, Y, _, _), Points), Ys),
    findall(Z, member(point(_, _, Z, _), Points), Zs),
    findall(Label, (member(point(_, _, _, Props), Points),
                    (member(label(Label), Props) -> true ; Label = "")), Labels),
    findall(Cluster, (member(point(_, _, _, Props), Points),
                      (member(cluster(Cluster), Props) -> true ; Cluster = 1)), Clusters),
    format_js_array(Xs, XsJS),
    format_js_array(Ys, YsJS),
    format_js_array(Zs, ZsJS),
    format_js_string_array(Labels, LabelsJS),
    format_js_array(Clusters, ClustersJS),
    format(atom(JS),
'const x = ~w;
    const y = ~w;
    const z = ~w;
    const labels = ~w;
    const clusters = ~w;', [XsJS, YsJS, ZsJS, LabelsJS, ClustersJS]).

%% generate_line_data_js(+Points, -JS)
generate_line_data_js(Points, JS) :-
    findall(X, member(idx_point(_, X, _, _), Points), Xs),
    findall(Y, member(idx_point(_, _, Y, _), Points), Ys),
    findall(Z, member(idx_point(_, _, _, Z), Points), Zs),
    format_js_array(Xs, XsJS),
    format_js_array(Ys, YsJS),
    format_js_array(Zs, ZsJS),
    format(atom(JS),
'const x = ~w;
    const y = ~w;
    const z = ~w;', [XsJS, YsJS, ZsJS]).

%% format_js_array(+List, -JS)
format_js_array([], '[]').
format_js_array(List, JS) :-
    List \= [],
    atomic_list_concat(List, ', ', ItemsStr),
    format(atom(JS), '[~w]', [ItemsStr]).

%% format_js_string_array(+List, -JS)
format_js_string_array([], '[]').
format_js_string_array(List, JS) :-
    List \= [],
    findall(Quoted, (member(Item, List), format(atom(Quoted), '"~w"', [Item])), QuotedList),
    atomic_list_concat(QuotedList, ', ', ItemsStr),
    format(atom(JS), '[~w]', [ItemsStr]).

% ============================================================================
% CSS GENERATION
% ============================================================================

%% generate_plot3d_styles(+Name, -CssCode)
generate_plot3d_styles(_Name, CssCode) :-
    CssCode = '.plotContainer {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem;
  background: var(--surface, #16213e);
  border-radius: 12px;
}

.plotContainer :global(.js-plotly-plot) {
  border-radius: 8px;
  overflow: hidden;
}
'.

% ============================================================================
% PYTHON/MATPLOTLIB GENERATION
% ============================================================================

%% generate_plot3d_matplotlib(+Name, -PythonCode)
generate_plot3d_matplotlib(Name, PythonCode) :-
    plot3d_type(Name, Type),
    generate_matplotlib_by_type(Type, Name, PythonCode).

generate_matplotlib_by_type(surface, Name, PythonCode) :-
    surface3d(Name, Config),
    (member(title(Title), Config) -> true ; Title = "3D Surface"),
    (member(function(Func), Config) -> true ; Func = sin_cos),
    (member(x_range(XMin, XMax), Config) -> true ; (XMin = -3.14, XMax = 3.14)),
    (member(y_range(YMin, YMax), Config) -> true ; (YMin = -3.14, YMax = 3.14)),
    (member(resolution(Res), Config) -> true ; Res = 50),
    (member(colorscale(ColorScale), Config) -> true ; ColorScale = viridis),

    get_surface_function_py(Func, FuncPy),
    atom_string(ColorScale, CmapStr),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - 3D Surface Plot
# Plot: ~w

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_~w():
    """~w"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Generate mesh
    x = np.linspace(~w, ~w, ~w)
    y = np.linspace(~w, ~w, ~w)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values
    ~w

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap="~w", edgecolor="none", alpha=0.9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("~w")

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, XMin, XMax, Res, YMin, YMax, Res, FuncPy, CmapStr, Title, Name]).

generate_matplotlib_by_type(scatter, Name, PythonCode) :-
    scatter3d_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "3D Scatter"),
    (member(marker_size(MarkerSize), Config) -> true ; MarkerSize = 50),

    findall(point(X, Y, Z, Props), scatter3d_point(Name, X, Y, Z, Props), Points),
    generate_scatter_data_py(Points, DataPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - 3D Scatter Plot
# Plot: ~w

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_~w():
    """~w"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Data
    ~w

    # Plot scatter
    scatter = ax.scatter(x, y, z, c=clusters, s=~w, cmap="viridis", alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("~w")

    fig.colorbar(scatter, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, DataPy, MarkerSize, Title, Name]).

generate_matplotlib_by_type(line, Name, PythonCode) :-
    line3d_spec(Name, Config),
    (member(title(Title), Config) -> true ; Title = "3D Line"),
    (member(line_width(LineWidth), Config) -> true ; LineWidth = 2),
    (member(color(Color), Config) -> true ; Color = 'blue'),

    findall(idx_point(Idx, X, Y, Z), line3d_point(Name, Idx, X, Y, Z), Points),
    sort(Points, SortedPoints),
    generate_line_data_py(SortedPoints, DataPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - 3D Line Plot
# Plot: ~w

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_~w():
    """~w"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Data
    ~w

    # Plot line
    ax.plot(x, y, z, linewidth=~w, color="~w")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("~w")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, DataPy, LineWidth, Color, Title, Name]).

%% get_surface_function_py(+Func, -Py)
get_surface_function_py(sin_cos, 'Z = np.sin(X) * np.cos(Y)').
get_surface_function_py(paraboloid, 'Z = X**2 + Y**2').
get_surface_function_py(saddle, 'Z = X**2 - Y**2').
get_surface_function_py(ripple, 'R = np.sqrt(X**2 + Y**2); Z = np.sin(R * 3) / (R + 0.1)').
get_surface_function_py(gaussian, 'Z = np.exp(-(X**2 + Y**2) / 2)').
get_surface_function_py(_, Py) :- get_surface_function_py(sin_cos, Py).

%% generate_scatter_data_py(+Points, -Py)
generate_scatter_data_py(Points, Py) :-
    findall(X, member(point(X, _, _, _), Points), Xs),
    findall(Y, member(point(_, Y, _, _), Points), Ys),
    findall(Z, member(point(_, _, Z, _), Points), Zs),
    findall(C, (member(point(_, _, _, Props), Points),
                (member(cluster(C), Props) -> true ; C = 1)), Clusters),
    format_py_array(Xs, XsPy),
    format_py_array(Ys, YsPy),
    format_py_array(Zs, ZsPy),
    format_py_array(Clusters, ClustersPy),
    format(atom(Py),
'x = np.array(~w)
    y = np.array(~w)
    z = np.array(~w)
    clusters = np.array(~w)', [XsPy, YsPy, ZsPy, ClustersPy]).

%% generate_line_data_py(+Points, -Py)
generate_line_data_py(Points, Py) :-
    findall(X, member(idx_point(_, X, _, _), Points), Xs),
    findall(Y, member(idx_point(_, _, Y, _), Points), Ys),
    findall(Z, member(idx_point(_, _, _, Z), Points), Zs),
    format_py_array(Xs, XsPy),
    format_py_array(Ys, YsPy),
    format_py_array(Zs, ZsPy),
    format(atom(Py),
'x = np.array(~w)
    y = np.array(~w)
    z = np.array(~w)', [XsPy, YsPy, ZsPy]).

%% format_py_array(+List, -Py)
format_py_array([], '[]').
format_py_array(List, Py) :-
    List \= [],
    atomic_list_concat(List, ', ', ItemsStr),
    format(atom(Py), '[~w]', [ItemsStr]).

% ============================================================================
% PLOTLY PYTHON GENERATION
% ============================================================================

%% generate_plot3d_plotly(+Name, -PythonCode)
generate_plot3d_plotly(Name, PythonCode) :-
    plot3d_type(Name, surface),
    surface3d(Name, Config),
    (member(title(Title), Config) -> true ; Title = "3D Surface"),
    (member(function(Func), Config) -> true ; Func = sin_cos),
    (member(x_range(XMin, XMax), Config) -> true ; (XMin = -3.14, XMax = 3.14)),
    (member(y_range(YMin, YMax), Config) -> true ; (YMin = -3.14, YMax = 3.14)),
    (member(resolution(Res), Config) -> true ; Res = 50),
    (member(colorscale(ColorScale), Config) -> true ; ColorScale = viridis),

    get_surface_function_py(Func, FuncPy),

    format(atom(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver - 3D Surface (Plotly)
# Plot: ~w

import numpy as np
import plotly.graph_objects as go

def plot_~w():
    """~w"""
    # Generate mesh
    x = np.linspace(~w, ~w, ~w)
    y = np.linspace(~w, ~w, ~w)
    X, Y = np.meshgrid(x, y)

    # Calculate Z
    ~w

    # Create figure
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="~w")])

    fig.update_layout(
        title="~w",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()

if __name__ == "__main__":
    plot_~w()
', [Name, Name, Title, XMin, XMax, Res, YMin, YMax, Res, FuncPy, ColorScale, Title, Name]).

% ============================================================================
% LAYOUT INTEGRATION
% ============================================================================

%% generate_plot3d_with_layout(+Name, +LayoutPattern, -Code)
generate_plot3d_with_layout(Name, LayoutPattern, Code) :-
    generate_plot3d_component(Name, ComponentCode),
    generate_plot3d_styles(Name, PlotCSS),
    (has_layout(LayoutPattern)
    ->  generate_layout_css(LayoutPattern, LayoutCSS),
        format(atom(Code), '~w~n~n/* Layout CSS */~n~w~n~n/* Plot CSS */~n~w', [ComponentCode, LayoutCSS, PlotCSS])
    ;   format(atom(Code), '~w~n~n/* CSS */~n~w', [ComponentCode, PlotCSS])
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

test_plot3d_generator :-
    format('Testing plot3d_generator module...~n~n'),

    % Test surface spec query
    format('Test 1: Surface spec query~n'),
    (surface3d(wave_surface, _)
    ->  format('  PASS: wave_surface spec exists~n')
    ;   format('  FAIL: wave_surface spec not found~n')
    ),

    % Test plot type
    format('~nTest 2: Plot type detection~n'),
    (plot3d_type(wave_surface, surface)
    ->  format('  PASS: wave_surface is surface type~n')
    ;   format('  FAIL: type detection failed~n')
    ),

    % Test surface component generation
    format('~nTest 3: Surface component generation~n'),
    generate_plot3d_component(wave_surface, SurfaceCode),
    atom_length(SurfaceCode, SurfaceLen),
    (SurfaceLen > 1500
    ->  format('  PASS: Generated ~w chars~n', [SurfaceLen])
    ;   format('  FAIL: Code too short: ~w~n', [SurfaceLen])
    ),

    % Test scatter component
    format('~nTest 4: Scatter component generation~n'),
    generate_plot3d_component(cluster_demo, ScatterCode),
    (sub_atom(ScatterCode, _, _, _, 'scatter3d')
    ->  format('  PASS: Contains scatter3d~n')
    ;   format('  FAIL: Missing scatter3d~n')
    ),

    % Test line component
    format('~nTest 5: Line component generation~n'),
    generate_plot3d_component(helix, LineCode),
    (sub_atom(LineCode, _, _, _, 'lines')
    ->  format('  PASS: Contains lines mode~n')
    ;   format('  FAIL: Missing lines mode~n')
    ),

    % Test matplotlib generation
    format('~nTest 6: Matplotlib generation~n'),
    generate_plot3d_matplotlib(wave_surface, PyCode),
    (sub_atom(PyCode, _, _, _, 'plot_surface')
    ->  format('  PASS: Contains plot_surface~n')
    ;   format('  FAIL: Missing plot_surface~n')
    ),

    format('~nAll tests completed.~n').

:- initialization(test_plot3d_generator, main).
