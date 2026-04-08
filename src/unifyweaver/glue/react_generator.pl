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
    ;   Type == file_browser
    ->  generate_file_browser_component(Name, Config, Options, Code)
    ;   Type == http_cli
    ->  generate_http_cli_component(Name, Config, Options, Code)
    ;   Type == custom
    ->  generate_custom_component(Name, Config, Options, Code)
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

%% generate_custom_component(+Name, +Config, +Options, -Code)
%  Generate a custom React component from raw code or file.
generate_custom_component(_Name, Config, _Options, Code) :-
    (   member(code(RawCode), Config)
    ->  Code = RawCode
    ;   member(file(Path), Config)
    ->  read_file_to_string(Path, FileString, []),
        Code = FileString
    ;   Code = '// No code or file specified for custom component\n'
    ).

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

% ============================================================================
% FILE BROWSER COMPONENT GENERATION
% ============================================================================

%% generate_mock_fs_js(+FSDirs, -Code)
%  Generate JavaScript mock filesystem object from Prolog directory specs.
%  Each fs_dir(Path, Entries) becomes a key-value pair in a JS Record.
generate_mock_fs_js([], Code) :-
    Code = 'const mockFS: Record<string, Array<{name: string, type: string, size: number}>> = {}'.
generate_mock_fs_js(FSDirs, Code) :-
    FSDirs \= [],
    maplist(generate_mock_fs_dir_js, FSDirs, DirCodes),
    atomic_list_concat(DirCodes, '\n', DirsJoined),
    format(atom(Code),
'// Mock file system for demo
const mockFS: Record<string, Array<{name: string, type: string, size: number}>> = {
~w
}', [DirsJoined]).

%% generate_mock_fs_dir_js(+FSDir, -Code)
generate_mock_fs_dir_js(fs_dir(Path, Entries), Code) :-
    maplist(generate_mock_fs_entry_js, Entries, EntryCodes),
    atomic_list_concat(EntryCodes, '\n', EntriesJoined),
    format(atom(Code), '  ''~w'': [\n~w\n  ],', [Path, EntriesJoined]).

%% generate_mock_fs_entry_js(+Entry, -Code)
generate_mock_fs_entry_js(entry(Name, Type, Size), Code) :-
    format(atom(Code), '    { name: ''~w'', type: ''~w'', size: ~d },', [Name, Type, Size]).

%% compute_fb_parent(+Path, +RootPath, -JSExpr)
%  Compute the JavaScript expression for the initial parent path.
%  Returns 'null' if Path equals RootPath, otherwise a quoted string.
compute_fb_parent(Path, RootPath, 'null') :-
    Path == RootPath, !.
compute_fb_parent(Path, _RootPath, Expr) :-
    atom_string(Path, PathStr),
    split_string(PathStr, "/", "", Parts),
    append(ParentParts, [_], Parts),
    atomic_list_concat(ParentParts, '/', Parent),
    format(atom(Expr), '''~w''', [Parent]).

%% generate_file_browser_component(+Name, +Config, +Options, -Code)
%  Generate a file browser React component from Prolog specifications.
%  Config keys used:
%    initial_path/1  - Starting directory path
%    root_path/1     - Top-level boundary for navigation
%    mock_fs/1       - List of fs_dir(Path, [entry(Name, Type, Size)])
%    features/1      - Feature flags (search, download, view_contents, etc.)
generate_file_browser_component(_Name, Config, _Options, Code) :-
    % Extract configuration with defaults
    (member(initial_path(InitPath), Config) -> true ; InitPath = '/home/user'),
    (member(root_path(RootPath), Config)   -> true ; RootPath = '/'),
    (member(mock_fs(FSDirs), Config)       -> true ; FSDirs = []),

    % Generate the mock filesystem JavaScript object
    generate_mock_fs_js(FSDirs, MockFSCode),

    % Compute the initial parent path expression
    compute_fb_parent(InitPath, RootPath, ParentExpr),

    % Build the complete component from a template with generated values
    format(atom(Code),
'import { useState } from ''react''
import ''./index.css''

// Helper function for formatting file sizes
const formatSize = (bytes: number): string => {
  if (!bytes || bytes === 0) return ''0 B''
  const k = 1024
  const sizes = [''B'', ''KB'', ''MB'', ''GB'']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + '' '' + sizes[i]
}

~w

function App() {
  // State
  const [loading, setLoading] = useState(false)
  const [fileContent, setFileContent] = useState<string | null>(null)
  const [notification, setNotification] = useState<string | null>(null)

  // Search state
  const [isSearching, setIsSearching] = useState(false)
  const [searchQuery, setSearchQuery] = useState('''')

  // Browse state
  const [browse, setBrowse] = useState<{
    path: string,
    parent: string | null,
    entries: Array<{name: string, type: string, size: number}>,
    selected: string | null
  }>({
    path: ''~w'',
    parent: ~w,
    entries: mockFS[''~w''] || [],
    selected: null
  })

  const [workingDir, setWorkingDir] = useState(''.'')

  // Get parent path
  const getParent = (path: string): string | null => {
    if (path === ''/'' || path === ''~w'') return null
    const parts = path.split(''/'')
    parts.pop()
    return parts.join(''/'') || ''/''
  }

  // Event handlers
  const navigateUp = () => {
    if (browse.parent) {
      const newPath = browse.parent
      setBrowse({
        path: newPath,
        parent: getParent(newPath),
        entries: mockFS[newPath] || [],
        selected: null
      })
      setFileContent(null)
      setNotification(null)
      setIsSearching(false)
    }
  }

  const handleEntryClick = (entry: {name: string, type: string, size: number}) => {
    if (entry.type === ''directory'') {
      const newPath = `${browse.path}/${entry.name}`
      setBrowse({
        path: newPath,
        parent: browse.path,
        entries: mockFS[newPath] || [],
        selected: null
      })
      setFileContent(null)
      setNotification(null)
      setIsSearching(false)
    } else {
      setBrowse(prev => ({ ...prev, selected: entry.name }))
      setFileContent(null)
      setNotification(null)
      setIsSearching(false)
    }
  }

  const setWorkingDirTo = (path: string) => {
    setWorkingDir(path)
    setNotification(`Working directory set to: ${path}`)
  }

  const viewFile = () => {
    if (browse.selected) {
      setLoading(true)
      // NOTE: demo simulation
      setTimeout(() => {
        setFileContent(`// Contents of ${browse.selected}\\n\\nexport default function Example() {\\n  return <div>Hello World</div>\\n}`)
        setLoading(false)
      }, 300)
    }
  }

  const downloadFile = () => {
    if (browse.selected) {
      setNotification(`Downloading: ${browse.path}/${browse.selected}`)
    }
  }

  const handleSearchSubmit = () => {
    if (searchQuery) {
      setNotification(`Searching for "${searchQuery}" in ${browse.path}...`)
      setIsSearching(false)
      setSearchQuery('''')
    }
  }

  return (
    <div className="app-container">
      <div className="panel">
        <div className="flex-col">
          {/* Navigation bar */}
          <div className="flex-row">
            {browse.parent && (
              <button onClick={navigateUp} className="btn btn-secondary">⬆️ Up</button>
            )}
            <span style={{ fontSize: 18 }}>📁 </span>
            <code className="path-code">{browse.path}</code>
            <button
              onClick={() => setWorkingDirTo(browse.path)}
              disabled={workingDir === browse.path}
              className={`btn ${workingDir === browse.path ? ''btn-panel'' : ''btn-primary''}`}
            >📌 Set as Working Dir</button>
          </div>

          {/* Entry count */}
          <span className="text-muted">{browse.entries.length} items</span>

          {/* File list */}
          <div className="file-list">
            {browse.entries.map((entry, index) => (
              <div
                key={index}
                onClick={() => handleEntryClick(entry)}
                className={`file-item ${browse.selected === entry.name ? ''file-item-selected'' : ''file-item-normal''} ${entry.type === ''directory'' ? ''file-item-dir'' : ''file-item-file''}`}
              >
                <div className="file-item-content">
                  <div className="file-item-left">
                    <span>{entry.type === ''directory'' ? ''📁'' : ''📄''}</span>
                    <span>{entry.name}</span>
                  </div>
                  <span className="text-muted">{formatSize(entry.size)}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Empty state */}
          {browse.entries.length === 0 && !loading && (
            <span className="text-muted text-center">Empty directory</span>
          )}

          {/* Selected file actions */}
          {browse.selected && (
            <div className="selected-panel">
              <div className="selected-panel-col">
                <span className="text-muted">Selected file:</span>
                <code className="path-code">{browse.selected}</code>
                <div className="flex-row">
                  <button onClick={viewFile} disabled={loading} className="btn btn-primary">
                    {loading ? "Loading..." : "View Contents"}
                  </button>
                  <button onClick={downloadFile} className="btn btn-primary">📥 Download</button>
                  <button onClick={() => setIsSearching(true)} className="btn btn-panel">Search Here</button>
                </div>

                {isSearching && (
                  <div className="flex-row" style={{ marginTop: ''10px'' }}>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder="Enter search pattern..."
                      className="search-input"
                      autoFocus
                      onKeyDown={e => e.key === ''Enter'' && handleSearchSubmit()}
                    />
                    <button onClick={handleSearchSubmit} className="btn btn-secondary">Go</button>
                    <button onClick={() => setIsSearching(false)} className="btn btn-text text-muted">Cancel</button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Notification */}
          {notification && (
            <div className="notification">
              <span className="text-success">{notification}</span>
              <button onClick={() => setNotification(null)} className="btn-text text-muted">✕</button>
            </div>
          )}

          {/* File content viewer */}
          {fileContent && (
            <div className="content-viewer">
              <div className="content-header">
                <span className="text-success">File contents:</span>
                <button onClick={() => setFileContent(null)} className="btn-text btn-text-error">✕ Close</button>
              </div>
              <pre className="content-pre">{fileContent}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
', [MockFSCode, InitPath, ParentExpr, InitPath, RootPath]).

% ============================================================================
% HTTP CLI COMPONENT GENERATION
% ============================================================================

%% generate_hc_tabs_js(+Tabs, -Code)
%  Generate JavaScript TABS array from Prolog tab specs.
generate_hc_tabs_js([], '[]').
generate_hc_tabs_js(Tabs, Code) :-
    Tabs \= [],
    maplist(generate_hc_tab_js, Tabs, TabCodes),
    atomic_list_concat(TabCodes, ',\n', TabsJoined),
    format(atom(Code), '[\n~w\n]', [TabsJoined]).

%% generate_hc_tab_js(+Tab, -Code)
generate_hc_tab_js(tab(Id, Label, Icon, Opts), Code) :-
    (member(highlight, Opts) -> HL = ', highlight: true' ; HL = ''),
    format(atom(Code), '  { id: ''~w'', label: ''~w'', icon: ''~w''~w }',
           [Id, Label, Icon, HL]).

%% generate_hc_auth_state_hooks(+AuthFields, -Code)
%  Generate useState hooks for authentication fields.
generate_hc_auth_state_hooks(Fields, Code) :-
    maplist(generate_hc_auth_state_hook, Fields, Hooks),
    atomic_list_concat(Hooks, '\n', Code).

%% generate_hc_auth_state_hook(+Field, -Code)
generate_hc_auth_state_hook(field(Name, _Label, Default), Hook) :-
    atom_string(Name, NameStr),
    capitalize_first_str(NameStr, CapName),
    format(atom(Hook), '  const [login~w, setLogin~w] = useState(''~w'')',
           [CapName, CapName, Default]).

%% generate_hc_auth_form_fields(+AuthFields, -Code)
%  Generate login form field JSX from auth field specs.
%  The last field gets extra bottom margin for visual separation.
generate_hc_auth_form_fields(Fields, Code) :-
    length(Fields, Len),
    generate_hc_auth_form_fields_(Fields, 1, Len, FieldCodes),
    atomic_list_concat(FieldCodes, '\n', Code).

generate_hc_auth_form_fields_([], _, _, []).
generate_hc_auth_form_fields_([Field|Rest], Idx, Len, [Code|Codes]) :-
    (Idx =:= Len -> Margin = 20 ; Margin = 15),
    generate_hc_auth_form_field(Field, Margin, Code),
    Idx1 is Idx + 1,
    generate_hc_auth_form_fields_(Rest, Idx1, Len, Codes).

%% generate_hc_auth_form_field(+Field, +Margin, -Code)
generate_hc_auth_form_field(field(Name, Label, _Default), Margin, Code) :-
    atom_string(Name, NameStr),
    capitalize_first_str(NameStr, CapName),
    (Name == password -> InputType = password ; InputType = Name),
    format(atom(Code),
'          <div style={{ marginBottom: ~d }}>
            <label className="text-muted" style={{ display: ''block'', marginBottom: 5 }}>~w</label>
            <input
              type="~w"
              value={login~w}
              onChange={e => setLogin~w(e.target.value)}
              onKeyDown={e => e.key === ''Enter'' && handleLogin()}
              className="input-field"
            />
          </div>', [Margin, Label, InputType, CapName, CapName]).

%% generate_hc_auth_defaults_hint(+AuthFields, -Hint)
%  Generate the default credentials hint text (e.g., "shell@local / shell").
generate_hc_auth_defaults_hint(Fields, Hint) :-
    maplist(get_auth_field_default, Fields, Defaults),
    atomic_list_concat(Defaults, ' / ', Hint).

get_auth_field_default(field(_, _, Default), Default).

%% generate_hc_root_options_jsx(+Roots, -Code)
%  Generate <option> elements for browse root selector.
generate_hc_root_options_jsx(Roots, Code) :-
    maplist(generate_hc_root_option_jsx, Roots, OptCodes),
    atomic_list_concat(OptCodes, '\n', Code).

%% generate_hc_root_option_jsx(+Root, -Code)
generate_hc_root_option_jsx(root(Value, Label), Code) :-
    format(atom(Code), '          <option value="~w">~w</option>', [Value, Label]).

%% generate_http_cli_component(+Name, +Config, +Options, -Code)
%  Generate an HTTP CLI React component from Prolog specifications.
%  Config keys used:
%    api_base/1      - Server base URL
%    title/1         - Application title
%    auth/2          - Authentication config with field specs
%    tabs/1          - Tab definitions [tab(Id, Label, Icon, Opts)]
%    browse_roots/1  - Filesystem root options [root(Value, Label)]
%    default_root/1  - Initial browse root
generate_http_cli_component(_Name, Config, _Options, Code) :-
    % Extract configuration with defaults
    (member(api_base(ApiBase), Config)     -> true ; ApiBase = 'http://localhost:3001'),
    (member(title(Title), Config)          -> true ; Title = 'CLI App'),
    (member(tabs(TabsConfig), Config)      -> true ; TabsConfig = []),
    (member(auth(_, AuthFields), Config)   -> true ; AuthFields = []),
    (member(browse_roots(RootsConfig), Config) -> true ; RootsConfig = []),

    % Generate data-driven code fragments
    generate_hc_tabs_js(TabsConfig, TabsJSCode),
    generate_hc_auth_state_hooks(AuthFields, AuthStateCode),
    generate_hc_auth_form_fields(AuthFields, AuthFormFieldsCode),
    generate_hc_auth_defaults_hint(AuthFields, DefaultsHint),
    generate_hc_root_options_jsx(RootsConfig, RootOptionsCode),

    % Generate per-tab code (state, handlers, panels)
    findall(Id, member(tab(Id, _, _, _), TabsConfig), TabIds),
    maplist(generate_hc_tab_state_code(Config), TabIds, TabStateCodes),
    atomic_list_concat(TabStateCodes, '\n', AllTabStateCode),
    generate_hc_auth_handlers_code(AuthHandlersCode),
    maplist(generate_hc_tab_handlers_code(Config), TabIds, TabHandlerCodes),
    atomic_list_concat(TabHandlerCodes, '\n', AllTabHandlersCode),
    maplist(generate_hc_tab_panel_code(Config), TabIds, TabPanelCodes),
    atomic_list_concat(TabPanelCodes, '\n\n', AllTabPanelsCode),

    % Build component from template sections
    generate_hc_section1(ApiBase, TabsJSCode, Section1),
    generate_hc_section2(AuthStateCode, AllTabStateCode, Section2),
    atomic_list_concat([AuthHandlersCode, '\n', AllTabHandlersCode, '\n'], Section3),
    generate_hc_section4(Title, AuthFormFieldsCode, DefaultsHint, Section4),
    generate_hc_section5(Title, RootOptionsCode, AllTabPanelsCode, Section5),

    atomic_list_concat([Section1, Section2, Section3, Section4, Section5], Code).

%% generate_hc_section1(+ApiBase, +TabsJSCode, -Code)
%  Section 1: Imports, configuration, types, helpers, API client, tabs.
generate_hc_section1(ApiBase, TabsJSCode, Code) :-
    format(atom(Code),
'import { useState, useRef } from ''react''
import ''./index.css''

// Configuration - update to match your server
const API_BASE = ''~w''

// Types
interface FileEntry {
  name: string
  type: ''file'' | ''directory''
  size: number
}

interface User {
  email: string
  roles: string[]
}

// Helper function for formatting file sizes
const formatSize = (bytes: number): string => {
  if (!bytes || bytes === 0) return ''0 B''
  const k = 1024
  const sizes = [''B'', ''KB'', ''MB'', ''GB'']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + '' '' + sizes[i]
}

// Strip ANSI escape codes for clean terminal display
const stripAnsi = (str: string): string => {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\\x1b\\[[0-9;]*[a-zA-Z]|\\x1b\\][^\\x07]*\\x07|\\x1b\\[\\?[0-9;]*[a-zA-Z]/g, '''')
    .replace(/\\r\\n/g, ''\\n'')
    .replace(/\\r/g, '''')
}

// API helper with auth
const api = {
  token: '''',

  async fetch(endpoint: string, options: RequestInit = {}) {
    const headers: Record<string, string> = {
      ''Content-Type'': ''application/json'',
      ...(options.headers as Record<string, string> || {})
    }
    if (this.token) {
      headers[''Authorization''] = `Bearer ${this.token}`
    }

    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || res.statusText)
    }

    return res.json()
  },

  async login(email: string, password: string) {
    const res = await this.fetch(''/auth/login'', {
      method: ''POST'',
      body: JSON.stringify({ email, password })
    })
    // Server returns {success: true, data: {token, user}}
    if (res.success && res.data) {
      this.token = res.data.token
      return res.data
    }
    throw new Error(res.error || ''Login failed'')
  },

  logout() {
    this.token = ''''
  }
}

// Tab definitions
const TABS = ~w

', [ApiBase, TabsJSCode]).

%% generate_hc_section2(+AuthStateCode, +TabStateCode, -Code)
%  Section 2: App function header, auth state, core UI state,
%  and per-tab state (generated from tab specs).
generate_hc_section2(AuthStateCode, TabStateCode, Code) :-
    format(atom(Code),
'function App() {
  // Auth state
  const [user, setUser] = useState<User | null>(null)
~w
  const [loginError, setLoginError] = useState('''')

  // UI state
  const [activeTab, setActiveTab] = useState(''browse'')
  const [loading, setLoading] = useState(false)
  const [workingDir, setWorkingDir] = useState(''.'')

~w
', [AuthStateCode, TabStateCode]).

% ---- Per-tab state generators ----

%% generate_hc_tab_state_code(+Config, +TabId, -Code)
%  Generate state declarations for a specific tab type.
generate_hc_tab_state_code(Config, browse, Code) :- !,
    (member(default_root(DefRoot), Config) -> true ; DefRoot = sandbox),
    format(atom(Code),
'  // Browse state
  const [browsePath, setBrowsePath] = useState(''.'')
  const [browseEntries, setBrowseEntries] = useState<FileEntry[]>([])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [browseRoot, setBrowseRoot] = useState(''~w'')
', [DefRoot]).

generate_hc_tab_state_code(_, upload, Code) :- !,
    Code =
'  // Upload state
  const [uploadFiles, setUploadFiles] = useState<File[]>([])
  const [uploadResult, setUploadResult] = useState('''')
'.

generate_hc_tab_state_code(_, cat, Code) :- !,
    Code =
'  // Cat state
  const [catPath, setCatPath] = useState('''')
  const [catContent, setCatContent] = useState('''')
'.

generate_hc_tab_state_code(_, shell, Code) :- !,
    Code =
'  // Shell state
  const [shellOutput, setShellOutput] = useState('''')
  const [shellInput, setShellInput] = useState('''')
  const [shellConnected, setShellConnected] = useState(false)
  const shellWs = useRef<WebSocket | null>(null)
'.

generate_hc_tab_state_code(_, _, '').

% ---- Auth handlers (always present) ----

%% generate_hc_auth_handlers_code(-Code)
%  Generate login/logout event handlers.
generate_hc_auth_handlers_code(Code) :-
    Code =
'  // Login handler
  const handleLogin = async () => {
    setLoading(true)
    setLoginError('''')
    try {
      const res = await api.login(loginEmail, loginPassword)
      setUser({ email: res.user.email, roles: res.user.roles })
      loadBrowse(''.'')
    } catch (err: any) {
      setLoginError(err.message || ''Login failed'')
    } finally {
      setLoading(false)
    }
  }

  // Logout handler
  const handleLogout = () => {
    api.logout()
    setUser(null)
    setShellConnected(false)
    if (shellWs.current) {
      shellWs.current.close()
    }
  }
'.

% ---- Per-tab handler generators ----

%% generate_hc_tab_handlers_code(+Config, +TabId, -Code)
%  Generate event handler functions for a specific tab type.
generate_hc_tab_handlers_code(_, browse, Code) :- !,
    Code =
'  // Browse handlers
  const loadBrowse = async (path: string, root?: string) => {
    setLoading(true)
    const useRoot = root || browseRoot
    try {
      const res = await api.fetch(''/browse'', {
        method: ''POST'',
        body: JSON.stringify({ path, root: useRoot })
      })
      // Server returns {success: true, data: {path, entries}}
      const data = res.data || res
      setBrowsePath(data.path || path)
      setBrowseEntries(data.entries || [])
      setSelectedFile(null)
    } catch (err: any) {
      console.error(''Browse failed:'', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRootChange = (newRoot: string) => {
    setBrowseRoot(newRoot)
    setBrowsePath(''.'')
    loadBrowse(''.'', newRoot)
  }

  const handleEntryClick = (entry: FileEntry) => {
    if (entry.type === ''directory'') {
      const newPath = browsePath === ''.'' ? entry.name : `${browsePath}/${entry.name}`
      loadBrowse(newPath)
    } else {
      setSelectedFile(entry.name)
    }
  }

  const navigateUp = () => {
    if (browsePath !== ''.'' && browsePath !== ''/'') {
      const parts = browsePath.split(''/'')
      parts.pop()
      loadBrowse(parts.join(''/'') || ''.'')
    }
  }

  const viewFile = async () => {
    if (!selectedFile) return
    const path = browsePath === ''.'' ? selectedFile : `${browsePath}/${selectedFile}`
    setCatPath(path)
    setActiveTab(''cat'')
    handleCat(path)
  }

  const downloadFile = async () => {
    if (!selectedFile) return
    const path = browsePath === ''.'' ? selectedFile : `${browsePath}/${selectedFile}`
    try {
      const res = await fetch(`${API_BASE}/download?path=${encodeURIComponent(path)}&root=${browseRoot}`, {
        headers: { ''Authorization'': `Bearer ${api.token}` }
      })
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement(''a'')
      a.href = url
      a.download = selectedFile
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error(''Download failed:'', err)
    }
  }
'.

generate_hc_tab_handlers_code(_, upload, Code) :- !,
    Code =
'  // Upload handlers
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setUploadFiles(Array.from(e.target.files))
    }
  }

  // File System Access API - better file picker on Android
  const openFilePicker = async () => {
    // @ts-ignore - File System Access API
    if (!window.showOpenFilePicker) {
      setUploadResult(''File System Access API not available - use standard file input'')
      return
    }
    try {
      // @ts-ignore
      const handles = await window.showOpenFilePicker({ multiple: true })
      const files: File[] = []
      for (const handle of handles) {
        const file = await handle.getFile()
        files.push(file)
      }
      if (files.length === 0) return

      // Upload immediately
      setLoading(true)
      setUploadResult(`Uploading ${files.length} file(s)...`)

      const formData = new FormData()
      formData.append(''destination'', workingDir)
      formData.append(''root'', browseRoot)
      files.forEach(file => formData.append(''files'', file))

      const res = await fetch(`${API_BASE}/upload`, {
        method: ''POST'',
        headers: { ''Authorization'': `Bearer ${api.token}` },
        body: formData
      })
      const json = await res.json()
      if (json.success) {
        setUploadResult(`Uploaded ${json.data.count} file(s) to ${json.data.destination}`)
      } else {
        setUploadResult(`Error: ${json.error || ''Upload failed''}`)
      }
    } catch (err: any) {
      if (err.name !== ''AbortError'') {
        setUploadResult(`Error: ${err.message}`)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async () => {
    if (uploadFiles.length === 0) return
    setLoading(true)
    setUploadResult('''')

    const formData = new FormData()
    uploadFiles.forEach(file => formData.append(''files'', file))
    formData.append(''destination'', workingDir)
    formData.append(''root'', browseRoot)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: ''POST'',
        headers: { ''Authorization'': `Bearer ${api.token}` },
        body: formData
      })
      const json = await res.json()
      // Server returns {success: true, data: {count, destination, uploaded}}
      if (json.success) {
        setUploadResult(`Uploaded ${json.data.count} file(s) to ${json.data.destination}`)
      } else {
        setUploadResult(`Error: ${json.error || ''Upload failed''}`)
      }
      setUploadFiles([])
    } catch (err: any) {
      setUploadResult(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }
'.

generate_hc_tab_handlers_code(_, cat, Code) :- !,
    Code =
'  // Cat handler
  const handleCat = async (path?: string) => {
    const targetPath = path || catPath
    if (!targetPath) return
    setLoading(true)
    setCatContent('''')
    try {
      const res = await api.fetch(''/cat'', {
        method: ''POST'',
        body: JSON.stringify({ path: targetPath, root: browseRoot })
      })
      // Server returns {success: true, data: {content}}
      const data = res.data || res
      setCatContent(data.content || '''')
    } catch (err: any) {
      setCatContent(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }
'.

generate_hc_tab_handlers_code(_, shell, Code) :- !,
    Code =
'  // Shell handlers
  const connectShell = () => {
    const wsUrl = API_BASE.replace(''https:'', ''wss:'').replace(''http:'', ''ws:'')
    shellWs.current = new WebSocket(`${wsUrl}/shell?token=${api.token}`)

    shellWs.current.onopen = () => {
      setShellConnected(true)
      setShellOutput(prev => prev + ''--- Connected ---\\n'')
    }

    shellWs.current.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === ''output'' && msg.data) {
          setShellOutput(prev => prev + msg.data)
        }
      } catch {
        // If not JSON, append as-is
        setShellOutput(prev => prev + e.data)
      }
    }

    shellWs.current.onclose = () => {
      setShellConnected(false)
      setShellOutput(prev => prev + ''\\n--- Disconnected ---\\n'')
    }

    shellWs.current.onerror = () => {
      setShellOutput(prev => prev + ''\\n--- Connection error ---\\n'')
    }
  }

  const disconnectShell = () => {
    if (shellWs.current) {
      shellWs.current.close()
    }
  }

  const sendShellCommand = () => {
    if (shellWs.current && shellInput) {
      shellWs.current.send(shellInput + ''\\n'')
      setShellInput('''')
    }
  }
'.

generate_hc_tab_handlers_code(_, _, '').

%% generate_hc_section4(+Title, +AuthFormFieldsCode, +DefaultsHint, -Code)
%  Section 4: Login form rendering (shown when not authenticated).
generate_hc_section4(Title, AuthFormFieldsCode, DefaultsHint, Code) :-
    format(atom(Code),
'  // Render login form
  if (!user) {
    return (
      <div className="app-container">
        <h1 className="text-center" style={{ marginBottom: 30 }}>🔍 ~w</h1>
        <div className="panel login-container">
          <h2 style={{ marginBottom: 20 }}>Login Required</h2>
~w
          <button
            onClick={handleLogin}
            disabled={loading}
            className="btn btn-primary"
            style={{ width: ''100%'' }}
          >
            {loading ? ''Logging in...'' : ''Login''}
          </button>
          {loginError && (
            <p className="text-error text-center" style={{ marginTop: 15 }}>{loginError}</p>
          )}
          <p className="text-muted text-center" style={{ marginTop: 20 }}>
            Default: ~w
          </p>
        </div>
      </div>
    )
  }

', [Title, AuthFormFieldsCode, DefaultsHint]).

% ---- Per-tab panel JSX generators ----

%% generate_hc_tab_panel_code(+Config, +TabId, -Code)
%  Generate the JSX panel for a specific tab type.
generate_hc_tab_panel_code(_, browse, Code) :- !,
    Code =
'      {/* Browse panel */}
      {activeTab === ''browse'' && (
        <div className="panel">
          <div className="flex-row" style={{ marginBottom: 15 }}>
            {browsePath !== ''.'' && (
              <button onClick={navigateUp} className="btn btn-small btn-secondary">⬆️ Up</button>
            )}
            <span>📁</span>
            <code className="path-code">{browsePath}</code>
            <button
              onClick={() => setWorkingDir(browsePath)}
              disabled={workingDir === browsePath}
              className={`btn btn-small ${workingDir === browsePath ? ''btn-outline'' : ''btn-success''}`}
            >
              📌 Set as Working Dir
            </button>
          </div>

          <div className="file-list">
            {browseEntries.map((entry, i) => (
              <div
                key={i}
                onClick={() => handleEntryClick(entry)}
                className={`file-item ${selectedFile === entry.name ? ''file-item-selected'' : ''file-item-normal''} ${entry.type === ''directory'' ? ''file-item-dir'' : ''file-item-file''}`}
              >
                <div className="flex-between">
                  <span>{entry.type === ''directory'' ? ''📁'' : ''📄''} {entry.name}</span>
                  <span className="text-muted">{formatSize(entry.size)}</span>
                </div>
              </div>
            ))}
            {browseEntries.length === 0 && !loading && (
              <p className="text-muted text-center">Empty directory</p>
            )}
          </div>

          {selectedFile && (
            <div className="selected-panel">
              <p className="text-muted" style={{ marginBottom: 10 }}>Selected: <code className="path-code">{selectedFile}</code></p>
              <div className="flex-row">
                <button onClick={viewFile} className="btn btn-primary">View Contents</button>
                <button onClick={downloadFile} className="btn btn-primary">📥 Download</button>
              </div>
            </div>
          )}
        </div>
      )}'.

generate_hc_tab_panel_code(_, upload, Code) :- !,
    Code =
'      {/* Upload panel */}
      {activeTab === ''upload'' && (
        <div className="panel">
          <div className="header-panel" style={{ marginBottom: 15, padding: 10 }}>
            <p className="text-muted" style={{ margin: 0 }}>Destination: <code className="path-code">{workingDir}</code> ({browseRoot})</p>
          </div>

          {/* File System Access API - better for Android */}
          <div className="drop-zone" onClick={openFilePicker}>
            <p style={{ fontSize: 18, margin: ''0 0 5px 0'' }}>📂 Open File Picker</p>
            <p className="text-muted" style={{ margin: 0 }}>Recommended for Android - picks and uploads immediately</p>
          </div>

          {/* Fallback standard file input */}
          <div className="drop-zone-fallback">
            <p className="text-muted" style={{ fontSize: 14, margin: ''0 0 10px 0'' }}>Or use standard file input:</p>
            <input
              type="file"
              multiple
              onChange={handleFileSelect}
              style={{ padding: 10 }}
            />
          </div>

          {uploadFiles.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <p className="text-muted" style={{ marginBottom: 10 }}>Selected files:</p>
              {uploadFiles.map((file, i) => (
                <div key={i} className="flex-between" style={{ padding: ''8px 12px'', background: ''var(--color-bg-item)'', marginBottom: 4, borderRadius: 5 }}>
                  <span>{file.name}</span>
                  <span className="text-muted">{formatSize(file.size)}</span>
                </div>
              ))}
            </div>
          )}

          {uploadFiles.length > 0 && (
            <button
              onClick={handleUpload}
              disabled={loading}
              className="btn btn-primary"
              style={{ width: ''100%'' }}
            >
              {loading ? ''Uploading...'' : ''📤 Upload Files''}
            </button>
          )}

          {uploadResult && (
            <div className={`result-box ${uploadResult.startsWith(''Error'') ? ''result-error'' : ''result-success''}`}>
              {uploadResult}
            </div>
          )}
        </div>
      )}'.

generate_hc_tab_panel_code(_, cat, Code) :- !,
    Code =
'      {/* Cat panel */}
      {activeTab === ''cat'' && (
        <div className="panel">
          <div className="flex-row" style={{ marginBottom: 15 }}>
            <input
              type="text"
              value={catPath}
              onChange={e => setCatPath(e.target.value)}
              onKeyDown={e => e.key === ''Enter'' && handleCat()}
              placeholder="File path..."
              className="input-field"
              style={{ flex: 1 }}
            />
            <button
              onClick={() => handleCat()}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? ''Loading...'' : ''Read File''}
            </button>
          </div>

          {catContent && (
            <>
              <div style={{ marginBottom: 10 }}>
                <button
                  onClick={() => { setActiveTab(''browse''); setCatContent(''''); }}
                  className="btn btn-small btn-secondary"
                >
                  ← Back to Browse
                </button>
              </div>
              <div className="content-viewer">
                <pre className="content-pre">{catContent}</pre>
              </div>
            </>
          )}
        </div>
      )}'.

generate_hc_tab_panel_code(_, shell, Code) :- !,
    Code =
'      {/* Shell panel */}
      {activeTab === ''shell'' && (
        <div style={{ borderRadius: 5, overflow: ''hidden'' }}>
          <div className="shell-header flex-between">
            <span style={{ color: ''var(--color-accent)'', fontWeight: ''bold'' }}>🔐 Shell</span>
            <div className="flex-row">
              <span style={{ color: shellConnected ? ''var(--color-success)'' : ''var(--color-error)'', fontSize: 12 }}>
                ● {shellConnected ? ''Connected'' : ''Disconnected''}
              </span>
              {!shellConnected ? (
                <button onClick={connectShell} className="btn btn-tiny btn-success">Connect</button>
              ) : (
                <button onClick={disconnectShell} className="btn btn-tiny btn-error">Disconnect</button>
              )}
              <button onClick={() => setShellOutput('''')} className="btn btn-tiny btn-outline">Clear</button>
            </div>
          </div>

          <div className="shell-body">
            {shellOutput ? stripAnsi(shellOutput) : ''Click "Connect" to start a shell session.''}
          </div>

          <div className="shell-footer flex-row">
            <span style={{ color: ''var(--color-success)'' }}>$</span>
            <input
              type="text"
              value={shellInput}
              onChange={e => setShellInput(e.target.value)}
              onKeyDown={e => e.key === ''Enter'' && sendShellCommand()}
              placeholder="Enter command..."
              disabled={!shellConnected}
              className="input-shell"
            />
            <button
              onClick={sendShellCommand}
              disabled={!shellConnected}
              className="btn btn-tiny btn-primary"
            >
              Send
            </button>
          </div>
        </div>
      )}'.

generate_hc_tab_panel_code(_, _, '').

%% generate_hc_section5(+Title, +RootOptionsCode, +TabPanelsCode, -Code)
%  Section 5: Main app JSX shell with tab bar and generated panels.
generate_hc_section5(Title, RootOptionsCode, TabPanelsCode, Code) :-
    format(atom(Code),
'  // Render main app
  return (
    <div className="app-container">
      <h1 style={{ marginBottom: 20 }}>🔍 ~w</h1>

      {/* User header */}
      <div className="header-panel flex-between">
        <div className="flex-row">
          <span>{user.email}</span>
          {user.roles.map(role => (
            <span key={role} className="role-badge">{role}</span>
          ))}
        </div>
        <button onClick={handleLogout} className="btn btn-small btn-secondary">
          Logout
        </button>
      </div>

      {/* Working directory bar */}
      <div className="header-panel flex-row">
        <select
          value={browseRoot}
          onChange={e => handleRootChange(e.target.value)}
          className="select-field"
        >
~w
        </select>
        <span className="text-muted">Working Dir:</span>
        <code className="path-code path-code-success">{workingDir}</code>
      </div>

      {/* Tabs */}
      <div className="flex-row" style={{ marginBottom: 20 }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`btn ${activeTab === tab.id ? ''btn-primary'' : (tab.highlight ? ''btn-accent'' : ''btn-panel'')}`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

~w

      {loading && (
        <div className="loading-toast">
          Loading...
        </div>
      )}
    </div>
  )
}

export default App
', [Title, RootOptionsCode, TabPanelsCode]).

