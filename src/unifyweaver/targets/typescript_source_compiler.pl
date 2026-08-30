:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% typescript_source_compiler.pl - TypeScript/Node data-source consumer (G-P9)
%
% Thin wrapper that lets the TypeScript pattern target (and, by inheritance,
% annotated_js / vanilla_js) consume registered JSON and CSV data sources,
% emitting a self-contained Node script (no npm deps: fs + JSON.parse).
%
% Modelled directly on powershell_compiler:compile_to_pure_powershell/3 — it
% requests the `_typescript` suffixed templates and dispatches by source type
% to csv_source:compile_source/4 / json_source:compile_source/4. This is a
% parallel, independent path: it does NOT touch the native clause / guard /
% recursion machinery of typescript_target.pl.
%
% Unlike the PowerShell-pure entry point (which reads source_type + config from
% its Options), the TypeScript path is reached from typescript_target via
% dynamic_source_compiler:is_dynamic_source/1, so the source type and config
% are looked up from the registered dynamic_source_def/3.

:- module(typescript_source_compiler, [
    compile_to_typescript_source/3   % +Pred/Arity, +Options, -TsCode
]).

:- use_module(library(lists)).
:- use_module('../core/dynamic_source_compiler').
:- use_module('../sources/csv_source', []).
:- use_module('../sources/json_source', []).

%% compile_to_typescript_source(+Pred/Arity, +Options, -TsCode)
%  Compile a registered CSV/JSON dynamic source to a self-contained Node
%  script using the `_typescript` templates. Fails for unsupported source
%  types so the caller can fall through to its normal paths.
compile_to_typescript_source(Pred/Arity, Options, TsCode) :-
    dynamic_source_compiler:dynamic_source_def(Pred/Arity, Type, Config),
    ts_source_dispatch(Type, Pred/Arity, Config, Options, TsCode).

%% ts_source_dispatch(+Type, +Pred/Arity, +Config, +Options, -TsCode)
ts_source_dispatch(csv, PredArity, Config, Options, TsCode) :- !,
    ts_source_options(Config, Options, SrcOptions),
    csv_source:compile_source(PredArity, SrcOptions, [], TsCode).
ts_source_dispatch(json, PredArity, Config, Options, TsCode) :- !,
    ensure_json_defaults(Config, Config1),
    ts_source_options(Config1, Options, SrcOptions),
    json_source:compile_source(PredArity, SrcOptions, [], TsCode).

%% ts_source_options(+Config, +Options, -SrcOptions)
%  Merge runtime Options ahead of the stored Config (runtime wins) and append
%  the template_suffix that selects the `_typescript` template variants.
ts_source_options(Config, Options, SrcOptions) :-
    append(Options, Config, Merged),
    append(Merged, [template_suffix('_typescript')], SrcOptions).

%% ensure_json_defaults(+Config0, -Config)
%  json_source:compile_source/4 validates that a JSON source carries a
%  jq_filter and an input mode (json_file / json_stdin). The pure-Node JSON
%  template ignores the jq filter entirely (it reads the array with
%  JSON.parse), so supply harmless defaults when the declaration omits them —
%  this keeps the Node path usable without leaking jq semantics into the
%  user's source declaration.
ensure_json_defaults(Config0, Config) :-
    (   member(jq_filter(_), Config0)
    ->  Config1 = Config0
    ;   Config1 = [jq_filter('.[]')|Config0]
    ),
    (   ( member(json_file(_), Config1) ; member(json_stdin(true), Config1) )
    ->  Config = Config1
    ;   Config = [json_stdin(true)|Config1]
    ).
