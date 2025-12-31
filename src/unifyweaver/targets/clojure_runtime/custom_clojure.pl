% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_clojure.pl - Custom Clojure Component Type
%
% Allows injecting raw Clojure code as a component.
% The code is wrapped in a defn with invoke function.
%
% Example:
%   declare_component(source, my_transform, custom_clojure, [
%       code("(clojure.string/upper-case input)"),
%       requires(["clojure.string", "clojure.set"])
%   ]).

:- module(custom_clojure, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Clojure Component'),
    version('1.0.0'),
    description('Injects custom Clojure code and exposes it as a component')
)).

%% validate_config(+Config)
validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

%% init_component(+Name, +Config)
init_component(_Name, _Config).

%% invoke_component(+Name, +Config, +Input, -Output)
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_clojure))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(requires(Requires), Config)
    ->  maplist(format_clojure_require, Requires, RequireLines),
        atomic_list_concat(RequireLines, '\n', RequiresCode)
    ;   RequiresCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
";; Custom Component: ~w
~w

(defn comp-~w-invoke
  \"Custom component: ~w\"
  [input]
  ~w)
", [NameStr, RequiresCode, NameStr, NameStr, Body]).

%% format_clojure_require(+Namespace, -RequireLine)
format_clojure_require(Namespace, RequireLine) :-
    format(string(RequireLine), "(require '[~w])", [Namespace]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_clojure, custom_clojure, [
        description("Custom Clojure Code")
    ])
), now).
