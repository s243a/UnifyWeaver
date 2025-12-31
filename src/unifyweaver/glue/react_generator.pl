% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% React Generator - Declarative React Component Generation
%
% This module provides declarative UI component definitions and generates
% React/TypeScript code from Prolog specifications.
%
% Usage:
%   % Define a UI component
%   ui_component(numpy_calculator, [
%       type(form),
%       title("NumPy Calculator"),
%       inputs([input(numbers, array(number), "Enter numbers")]),
%       operations([
%           operation(mean, '/numpy/mean', "Calculate Mean"),
%           operation(std, '/numpy/std', "Calculate Std Dev")
%       ])
%   ]).
%
%   % Generate React component
%   ?- generate_react_component(numpy_calculator, Code).

:- module(react_generator, [
    % Component declaration
    ui_component/2,                     % ui_component(+Name, +Config)
    ui_theme/2,                         % ui_theme(+Name, +Config)

    % Component management
    declare_ui_component/2,             % declare_ui_component(+Name, +Config)
    clear_ui_components/0,              % clear_ui_components

    % Code generation
    generate_react_component/2,         % generate_react_component(+Name, -Code)
    generate_react_component/3,         % generate_react_component(+Name, +Options, -Code)
    generate_react_app/2,               % generate_react_app(+Name, -Code)
    generate_react_app/3,               % generate_react_app(+Name, +Options, -Code)
    generate_component_styles/2,        % generate_component_styles(+Name, -CSS)
    generate_api_hooks/2,               % generate_api_hooks(+Name, -Code)

    % Utilities
    all_ui_components/1,                % all_ui_components(-Components)

    % Testing
    test_react_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic ui_component/2.
:- dynamic ui_theme/2.

:- discontiguous ui_component/2.

% ============================================================================
% DEFAULT THEME
% ============================================================================

ui_theme(default, [
    primary_color('#3b82f6'),
    secondary_color('#64748b'),
    success_color('#22c55e'),
    error_color('#ef4444'),
    background('#ffffff'),
    surface('#f8fafc'),
    text_primary('#1e293b'),
    text_secondary('#64748b'),
    border_radius('8px'),
    font_family('system-ui, -apple-system, sans-serif')
]).

ui_theme(dark, [
    primary_color('#60a5fa'),
    secondary_color('#94a3b8'),
    success_color('#4ade80'),
    error_color('#f87171'),
    background('#0f172a'),
    surface('#1e293b'),
    text_primary('#f1f5f9'),
    text_secondary('#94a3b8'),
    border_radius('8px'),
    font_family('system-ui, -apple-system, sans-serif')
]).

% ============================================================================
% DEFAULT UI COMPONENTS
% ============================================================================

% NumPy Calculator Component
ui_component(numpy_calculator, [
    type(form),
    title("NumPy Calculator"),
    description("Perform NumPy operations on arrays of numbers"),
    inputs([
        input(data, array(number), "Numbers", [
            placeholder("Enter comma-separated numbers"),
            required(true)
        ])
    ]),
    operations([
        operation(mean, '/api/numpy/mean', "Mean", [icon(calculator)]),
        operation(std, '/api/numpy/std', "Std Dev", [icon(chart)]),
        operation(sum, '/api/numpy/sum', "Sum", [icon(plus)]),
        operation(min, '/api/numpy/min', "Min", [icon(arrow_down)]),
        operation(max, '/api/numpy/max', "Max", [icon(arrow_up)])
    ]),
    result_display(number, [precision(6)])
]).

% Math Functions Component
ui_component(math_calculator, [
    type(form),
    title("Math Functions"),
    description("Common mathematical operations"),
    inputs([
        input(value, number, "Value", [
            placeholder("Enter a number"),
            required(true)
        ])
    ]),
    operations([
        operation(sqrt, '/api/math/sqrt', "Square Root", []),
        operation(log, '/api/math/log', "Natural Log", []),
        operation(exp, '/api/math/exp', "Exponential", []),
        operation(sin, '/api/math/sin', "Sine", []),
        operation(cos, '/api/math/cos', "Cosine", [])
    ]),
    result_display(number, [precision(8)])
]).

% Statistics Component
ui_component(statistics_calculator, [
    type(form),
    title("Statistics Calculator"),
    description("Statistical analysis functions"),
    inputs([
        input(data, array(number), "Data Points", [
            placeholder("Enter comma-separated numbers"),
            required(true)
        ])
    ]),
    operations([
        operation(mean, '/api/statistics/mean', "Mean", []),
        operation(median, '/api/statistics/median', "Median", []),
        operation(stdev, '/api/statistics/stdev', "Std Dev", [])
    ]),
    result_display(number, [precision(6)])
]).

% Constants Display Component
ui_component(math_constants, [
    type(display),
    title("Mathematical Constants"),
    description("Common mathematical constants from Python"),
    constants([
        constant(pi, '/api/math/pi', "Pi (π)"),
        constant(e, '/api/math/e', "Euler's Number (e)")
    ])
]).

% ============================================================================
% COMPONENT MANAGEMENT
% ============================================================================

%% declare_ui_component(+Name, +Config)
%  Dynamically declare a UI component.
declare_ui_component(Name, Config) :-
    (   ui_component(Name, _)
    ->  retract(ui_component(Name, _))
    ;   true
    ),
    assertz(ui_component(Name, Config)).

%% clear_ui_components
%  Clear all dynamic UI components.
clear_ui_components :-
    retractall(ui_component(_, _)).

%% all_ui_components(-Components)
%  Get all defined UI components.
all_ui_components(Components) :-
    findall(Name-Config, ui_component(Name, Config), Components).

% ============================================================================
% REACT COMPONENT GENERATION
% ============================================================================

%% generate_react_component(+Name, -Code)
%  Generate React component with default options.
generate_react_component(Name, Code) :-
    generate_react_component(Name, [], Code).

%% generate_react_component(+Name, +Options, -Code)
%  Generate React component with options.
generate_react_component(Name, Options, Code) :-
    ui_component(Name, Config),
    member(type(Type), Config),
    (   Type == form
    ->  generate_form_component(Name, Config, Options, Code)
    ;   Type == display
    ->  generate_display_component(Name, Config, Options, Code)
    ;   generate_generic_component(Name, Config, Options, Code)
    ).

%% generate_form_component(+Name, +Config, +Options, -Code)
%  Generate a form-based React component.
generate_form_component(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    % Get component metadata
    (member(title(Title), Config) -> true ; Title = NameStr),
    (member(description(Desc), Config) -> true ; Desc = ""),

    % Generate input fields
    member(inputs(Inputs), Config),
    generate_input_state(Inputs, StateCode),
    generate_input_fields(Inputs, FieldsCode),

    % Generate operation buttons
    member(operations(Operations), Config),
    generate_operation_handlers(Operations, Inputs, HandlersCode),
    generate_operation_buttons(Operations, ButtonsCode),

    % Get result display config
    (member(result_display(_, ResultOpts), Config)
    ->  (member(precision(Prec), ResultOpts) -> true ; Prec = 4)
    ;   Prec = 4
    ),

    format(atom(Code), '/**
 * ~w Component
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import React, { useState } from ''react'';
import styles from ''./~w.module.css'';

interface ~wProps {
  className?: string;
}

export const ~w: React.FC<~wProps> = ({ className }) => {
  // Input state
~w
  // Result state
  const [result, setResult] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Operation handlers
~w

  // Parse array input
  const parseArrayInput = (value: string): number[] => {
    return value
      .split('','')
      .map(s => s.trim())
      .filter(s => s !== '''')
      .map(s => parseFloat(s))
      .filter(n => !isNaN(n));
  };

  return (
    <div className={`${styles.container} ${className || ''''}`}>
      <div className={styles.header}>
        <h2 className={styles.title}>~w</h2>
        <p className={styles.description}>~w</p>
      </div>

      <div className={styles.form}>
~w
      </div>

      <div className={styles.operations}>
~w
      </div>

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}

      {result !== null && (
        <div className={styles.result}>
          <span className={styles.resultLabel}>Result:</span>
          <span className={styles.resultValue}>
            {result.toFixed(~w)}
          </span>
        </div>
      )}

      {loading && (
        <div className={styles.loading}>
          Computing...
        </div>
      )}
    </div>
  );
};

export default ~w;
', [ComponentName, NameStr, ComponentName, ComponentName, ComponentName,
    StateCode, HandlersCode, Title, Desc, FieldsCode, ButtonsCode, Prec, ComponentName]).

%% generate_display_component(+Name, +Config, +Options, -Code)
%  Generate a display-only React component.
generate_display_component(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    (member(title(Title), Config) -> true ; Title = NameStr),
    (member(description(Desc), Config) -> true ; Desc = ""),

    % Generate constants fetching
    member(constants(Constants), Config),
    generate_constants_fetchers(Constants, FetchersCode),
    generate_constants_display(Constants, DisplayCode),

    format(atom(Code), '/**
 * ~w Component
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import React, { useState, useEffect } from ''react'';
import styles from ''./~w.module.css'';

interface ConstantValue {
  name: string;
  label: string;
  value: number | null;
  loading: boolean;
}

interface ~wProps {
  className?: string;
}

export const ~w: React.FC<~wProps> = ({ className }) => {
~w

  return (
    <div className={`${styles.container} ${className || ''''}`}>
      <div className={styles.header}>
        <h2 className={styles.title}>~w</h2>
        <p className={styles.description}>~w</p>
      </div>

      <div className={styles.constants}>
~w
      </div>
    </div>
  );
};

export default ~w;
', [ComponentName, NameStr, ComponentName, ComponentName, ComponentName, FetchersCode,
    Title, Desc, DisplayCode, ComponentName]).

%% generate_generic_component(+Name, +Config, +Options, -Code)
%  Generate a generic React component.
generate_generic_component(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    pascal_case(NameStr, ComponentName),

    (member(title(Title), Config) -> true ; Title = NameStr),

    format(atom(Code), '/**
 * ~w Component
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import React from ''react'';
import styles from ''./~w.module.css'';

interface ~wProps {
  className?: string;
}

export const ~w: React.FC<~wProps> = ({ className }) => {
  return (
    <div className={`${styles.container} ${className || ''''}`}>
      <h2>~w</h2>
    </div>
  );
};

export default ~w;
', [ComponentName, NameStr, ComponentName, ComponentName, Title, ComponentName]).

% ============================================================================
% INPUT GENERATION HELPERS
% ============================================================================

%% generate_input_state(+Inputs, -Code)
%  Generate useState hooks for inputs.
generate_input_state(Inputs, Code) :-
    findall(StateHook, (
        member(input(Name, Type, _, _), Inputs),
        generate_single_state_hook(Name, Type, StateHook)
    ), StateHooks),
    atomic_list_concat(StateHooks, '\n', Code).

generate_single_state_hook(Name, array(_), Hook) :-
    atom_string(Name, NameStr),
    format(atom(Hook), '  const [~w, set~w] = useState<string>('''');',
           [NameStr, NameStr]).
generate_single_state_hook(Name, number, Hook) :-
    atom_string(Name, NameStr),
    capitalize_first_str(NameStr, CapName),
    format(atom(Hook), '  const [~w, set~w] = useState<string>('''');',
           [NameStr, CapName]).
generate_single_state_hook(Name, string, Hook) :-
    atom_string(Name, NameStr),
    capitalize_first_str(NameStr, CapName),
    format(atom(Hook), '  const [~w, set~w] = useState<string>('''');',
           [NameStr, CapName]).
generate_single_state_hook(Name, _, Hook) :-
    atom_string(Name, NameStr),
    capitalize_first_str(NameStr, CapName),
    format(atom(Hook), '  const [~w, set~w] = useState<string>('''');',
           [NameStr, CapName]).

%% generate_input_fields(+Inputs, -Code)
%  Generate JSX for input fields.
generate_input_fields(Inputs, Code) :-
    findall(FieldJSX, (
        member(input(Name, Type, Label, Opts), Inputs),
        generate_single_input_field(Name, Type, Label, Opts, FieldJSX)
    ), FieldsJSX),
    atomic_list_concat(FieldsJSX, '\n', Code).

generate_single_input_field(Name, Type, Label, Opts, JSX) :-
    atom_string(Name, NameStr),
    capitalize_first_str(NameStr, CapName),
    (member(placeholder(Placeholder), Opts) -> true ; Placeholder = ""),
    (member(required(true), Opts) -> Required = "required" ; Required = ""),
    (Type = array(_) -> InputType = "text" ; InputType = "text"),

    format(atom(JSX), '        <div className={styles.inputGroup}>
          <label className={styles.label} htmlFor="~w">~w</label>
          <input
            id="~w"
            type="~w"
            className={styles.input}
            value={~w}
            onChange={(e) => set~w(e.target.value)}
            placeholder="~w"
            ~w
          />
        </div>',
           [NameStr, Label, NameStr, InputType, NameStr, CapName, Placeholder, Required]).

% ============================================================================
% OPERATION GENERATION HELPERS
% ============================================================================

%% generate_operation_handlers(+Operations, +Inputs, -Code)
%  Generate handler functions for operations.
generate_operation_handlers(Operations, Inputs, Code) :-
    % Determine input preparation based on first input type
    (   Inputs = [input(InputName, array(_), _, _)|_]
    ->  atom_string(InputName, InputNameStr),
        format(atom(InputPrep), 'const inputData = parseArrayInput(~w);', [InputNameStr])
    ;   Inputs = [input(InputName, number, _, _)|_]
    ->  atom_string(InputName, InputNameStr),
        format(atom(InputPrep), 'const inputData = parseFloat(~w);', [InputNameStr])
    ;   InputPrep = 'const inputData = {};'
    ),

    findall(Handler, (
        member(operation(OpName, Endpoint, _, _), Operations),
        generate_single_operation_handler(OpName, Endpoint, InputPrep, Handler)
    ), Handlers),
    atomic_list_concat(Handlers, '\n\n', Code).

generate_single_operation_handler(OpName, Endpoint, InputPrep, Handler) :-
    atom_string(OpName, OpNameStr),
    capitalize_first_str(OpNameStr, CapOpName),
    atom_string(Endpoint, EndpointStr),

    format(atom(Handler), '  const handle~w = async () => {
    setLoading(true);
    setError(null);
    try {
      ~w
      const response = await fetch(''~w'', {
        method: ''POST'',
        headers: { ''Content-Type'': ''application/json'' },
        body: JSON.stringify({ data: inputData }),
      });
      const json = await response.json();
      if (json.success) {
        setResult(json.result);
      } else {
        setError(json.error || ''Operation failed'');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : ''Network error'');
    } finally {
      setLoading(false);
    }
  };', [CapOpName, InputPrep, EndpointStr]).

%% generate_operation_buttons(+Operations, -Code)
%  Generate JSX for operation buttons.
generate_operation_buttons(Operations, Code) :-
    findall(ButtonJSX, (
        member(operation(OpName, _, Label, _), Operations),
        generate_single_operation_button(OpName, Label, ButtonJSX)
    ), ButtonsJSX),
    atomic_list_concat(ButtonsJSX, '\n', Code).

generate_single_operation_button(OpName, Label, JSX) :-
    atom_string(OpName, OpNameStr),
    capitalize_first_str(OpNameStr, CapOpName),
    format(atom(JSX), '        <button
          className={styles.button}
          onClick={handle~w}
          disabled={loading}
        >
          ~w
        </button>', [CapOpName, Label]).

% ============================================================================
% CONSTANTS GENERATION HELPERS
% ============================================================================

%% generate_constants_fetchers(+Constants, -Code)
%  Generate useEffect hooks to fetch constants.
generate_constants_fetchers(Constants, Code) :-
    findall(Name-Label, member(constant(Name, _, Label), Constants), Pairs),
    generate_constants_state(Pairs, StateCode),
    generate_constants_effects(Constants, EffectsCode),

    format(atom(Code), '  const [constants, setConstants] = useState<ConstantValue[]>([
~w
  ]);

~w', [StateCode, EffectsCode]).

generate_constants_state(Pairs, Code) :-
    findall(Entry, (
        member(Name-Label, Pairs),
        atom_string(Name, NameStr),
        format(atom(Entry), '    { name: ''~w'', label: ''~w'', value: null, loading: true }',
               [NameStr, Label])
    ), Entries),
    atomic_list_concat(Entries, ',\n', Code).

generate_constants_effects(Constants, Code) :-
    findall(Effect, (
        member(constant(Name, Endpoint, _), Constants),
        generate_single_constant_effect(Name, Endpoint, Effect)
    ), Effects),
    atomic_list_concat(Effects, '\n\n', Code).

generate_single_constant_effect(Name, Endpoint, Effect) :-
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    format(atom(Effect), '  useEffect(() => {
    fetch(''~w'')
      .then(res => res.json())
      .then(json => {
        if (json.success) {
          setConstants(prev => prev.map(c =>
            c.name === ''~w'' ? { ...c, value: json.result, loading: false } : c
          ));
        }
      })
      .catch(() => {
        setConstants(prev => prev.map(c =>
          c.name === ''~w'' ? { ...c, loading: false } : c
        ));
      });
  }, []);', [EndpointStr, NameStr, NameStr]).

%% generate_constants_display(+Constants, -Code)
%  Generate JSX for displaying constants.
generate_constants_display(_Constants, Code) :-
    Code = '        {constants.map((c) => (
          <div key={c.name} className={styles.constant}>
            <span className={styles.constantLabel}>{c.label}</span>
            <span className={styles.constantValue}>
              {c.loading ? ''Loading...'' : c.value?.toFixed(10)}
            </span>
          </div>
        ))}'.

% ============================================================================
% STYLES GENERATION
% ============================================================================

%% generate_component_styles(+Name, -CSS)
%  Generate CSS module for a component.
generate_component_styles(Name, CSS) :-
    ui_component(Name, Config),
    (member(theme(ThemeName), Config) -> true ; ThemeName = default),
    ui_theme(ThemeName, Theme),

    (member(primary_color(Primary), Theme) -> true ; Primary = '#3b82f6'),
    (member(background(Bg), Theme) -> true ; Bg = '#ffffff'),
    (member(surface(Surface), Theme) -> true ; Surface = '#f8fafc'),
    (member(text_primary(TextPrim), Theme) -> true ; TextPrim = '#1e293b'),
    (member(text_secondary(TextSec), Theme) -> true ; TextSec = '#64748b'),
    (member(error_color(ErrorColor), Theme) -> true ; ErrorColor = '#ef4444'),
    (member(success_color(_SuccessColor), Theme) -> true ; true),  % Reserved for future use
    (member(border_radius(Radius), Theme) -> true ; Radius = '8px'),

    format(atom(CSS), '/**
 * Component Styles
 * Generated by UnifyWeaver
 */

.container {
  max-width: 480px;
  margin: 0 auto;
  padding: 24px;
  background: ~w;
  border-radius: ~w;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.header {
  margin-bottom: 24px;
}

.title {
  font-size: 1.5rem;
  font-weight: 600;
  color: ~w;
  margin: 0 0 8px 0;
}

.description {
  font-size: 0.875rem;
  color: ~w;
  margin: 0;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 20px;
}

.inputGroup {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.label {
  font-size: 0.875rem;
  font-weight: 500;
  color: ~w;
}

.input {
  padding: 10px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  font-size: 1rem;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.input:focus {
  outline: none;
  border-color: ~w;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.operations {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 20px;
}

.button {
  padding: 10px 16px;
  background: ~w;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.15s;
}

.button:hover:not(:disabled) {
  filter: brightness(0.9);
}

.button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.result {
  padding: 16px;
  background: ~w;
  border-radius: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.resultLabel {
  font-size: 0.875rem;
  color: ~w;
}

.resultValue {
  font-size: 1.25rem;
  font-weight: 600;
  color: ~w;
  font-family: ''SF Mono'', Monaco, monospace;
}

.error {
  padding: 12px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 6px;
  color: ~w;
  font-size: 0.875rem;
  margin-bottom: 16px;
}

.loading {
  text-align: center;
  padding: 12px;
  color: ~w;
  font-size: 0.875rem;
}

.constants {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.constant {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: ~w;
  border-radius: 6px;
}

.constantLabel {
  font-size: 0.875rem;
  color: ~w;
}

.constantValue {
  font-family: ''SF Mono'', Monaco, monospace;
  font-size: 0.875rem;
  color: ~w;
}
', [Bg, Radius, TextPrim, TextSec, TextPrim, Primary, Primary, Surface, TextSec, TextPrim,
    ErrorColor, TextSec, Surface, TextSec, TextPrim]).

% ============================================================================
% API HOOKS GENERATION
% ============================================================================

%% generate_api_hooks(+Name, -Code)
%  Generate custom React hooks for API calls.
generate_api_hooks(Name, Code) :-
    atom_string(Name, NameStr),
    pascal_case(NameStr, HooksName),

    format(atom(Code), '/**
 * API Hooks: ~w
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import { useState, useCallback } from ''react'';

interface ApiResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface UseApiCallResult<T> extends ApiResult<T> {
  execute: (...args: unknown[]) => Promise<void>;
  reset: () => void;
}

export function useApiCall<T>(
  endpoint: string,
  method: ''GET'' | ''POST'' = ''POST''
): UseApiCallResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async (...args: unknown[]) => {
    setLoading(true);
    setError(null);
    try {
      const options: RequestInit = {
        method,
        headers: { ''Content-Type'': ''application/json'' },
      };
      if (method === ''POST'' && args.length > 0) {
        options.body = JSON.stringify(args[0]);
      }
      const response = await fetch(endpoint, options);
      const json = await response.json();
      if (json.success) {
        setData(json.result);
      } else {
        setError(json.error || ''Operation failed'');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : ''Network error'');
    } finally {
      setLoading(false);
    }
  }, [endpoint, method]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, loading, error, execute, reset };
}

// Pre-configured hooks for common operations
export const use~wApi = () => ({
  mean: useApiCall<number>(''/api/numpy/mean''),
  std: useApiCall<number>(''/api/numpy/std''),
  sum: useApiCall<number>(''/api/numpy/sum''),
  min: useApiCall<number>(''/api/numpy/min''),
  max: useApiCall<number>(''/api/numpy/max''),
});
', [HooksName, HooksName]).

% ============================================================================
% REACT APP GENERATION
% ============================================================================

%% generate_react_app(+Name, -Code)
%  Generate a complete React app.
generate_react_app(Name, Code) :-
    generate_react_app(Name, [], Code).

%% generate_react_app(+Name, +Options, -Code)
%  Generate a complete React app with options.
generate_react_app(Name, Options, Code) :-
    atom_string(Name, NameStr),
    pascal_case(NameStr, AppName),

    % Get components to include
    (   member(components(ComponentNames), Options)
    ->  true
    ;   findall(N, ui_component(N, _), ComponentNames)
    ),

    % Generate imports
    generate_component_imports(ComponentNames, ImportsCode),

    % Generate component usage
    generate_component_usage(ComponentNames, UsageCode),

    format(atom(Code), '/**
 * ~w Application
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import React from ''react'';
~w
import ''./App.css'';

const App: React.FC = () => {
  return (
    <div className="app">
      <header className="app-header">
        <h1>~w</h1>
        <p>Generated by UnifyWeaver</p>
      </header>

      <main className="app-main">
~w
      </main>

      <footer className="app-footer">
        <p>Powered by UnifyWeaver + RPyC</p>
      </footer>
    </div>
  );
};

export default App;
', [AppName, ImportsCode, AppName, UsageCode]).

%% generate_component_imports(+Names, -Code)
%  Generate import statements for components.
generate_component_imports(Names, Code) :-
    findall(Import, (
        member(Name, Names),
        atom_string(Name, NameStr),
        pascal_case(NameStr, ComponentName),
        format(atom(Import), 'import { ~w } from ''./components/~w'';',
               [ComponentName, ComponentName])
    ), Imports),
    atomic_list_concat(Imports, '\n', Code).

%% generate_component_usage(+Names, -Code)
%  Generate JSX for using components.
generate_component_usage(Names, Code) :-
    findall(Usage, (
        member(Name, Names),
        atom_string(Name, NameStr),
        pascal_case(NameStr, ComponentName),
        format(atom(Usage), '        <~w />', [ComponentName])
    ), Usages),
    atomic_list_concat(Usages, '\n', Code).

% ============================================================================
% UTILITY HELPERS
% ============================================================================

%% pascal_case(+Input, -Output)
%  Convert snake_case to PascalCase.
pascal_case(Input, Output) :-
    atom_string(InputAtom, Input),
    atomic_list_concat(Parts, '_', InputAtom),
    maplist(capitalize_first, Parts, CapParts),
    atomic_list_concat(CapParts, '', Output).

%% capitalize_first(+Atom, -Capitalized)
capitalize_first(Atom, Capitalized) :-
    atom_string(Atom, Str),
    capitalize_first_str(Str, CapStr),
    atom_string(Capitalized, CapStr).

%% capitalize_first_str(+Str, -Capitalized)
capitalize_first_str("", "").
capitalize_first_str(Str, CapStr) :-
    Str \= "",
    string_codes(Str, [First|Rest]),
    to_upper(First, Upper),
    string_codes(CapStr, [Upper|Rest]).

% ============================================================================
% TESTING
% ============================================================================

test_react_generator :-
    format('~n=== React Generator Tests ===~n~n'),

    % Test component queries
    format('Component Queries:~n'),
    all_ui_components(AllComps),
    length(AllComps, CompCount),
    format('  Total components: ~w~n', [CompCount]),

    % Test code generation
    format('~nCode Generation:~n'),
    (   generate_react_component(numpy_calculator, CompCode),
        atom_length(CompCode, CompLen),
        format('  NumPy Calculator: ~d chars~n', [CompLen])
    ;   format('  NumPy Calculator: FAILED~n')
    ),

    (   generate_react_component(math_constants, ConstCode),
        atom_length(ConstCode, ConstLen),
        format('  Math Constants: ~d chars~n', [ConstLen])
    ;   format('  Math Constants: FAILED~n')
    ),

    (   generate_component_styles(numpy_calculator, CSSCode),
        atom_length(CSSCode, CSSLen),
        format('  Component CSS: ~d chars~n', [CSSLen])
    ;   format('  Component CSS: FAILED~n')
    ),

    (   generate_api_hooks(python, HooksCode),
        atom_length(HooksCode, HooksLen),
        format('  API Hooks: ~d chars~n', [HooksLen])
    ;   format('  API Hooks: FAILED~n')
    ),

    (   generate_react_app(demo_app, AppCode),
        atom_length(AppCode, AppLen),
        format('  React App: ~d chars~n', [AppLen])
    ;   format('  React App: FAILED~n')
    ),

    format('~n=== Tests Complete ===~n').
