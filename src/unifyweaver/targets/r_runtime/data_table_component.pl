% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% data_table_component.pl - data.table component for R target
% Injects library(data.table) and modifies R generation options

:- module(r_data_table_component, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

type_info(info(
    name('R data.table Component'),
    version('1.0.0'),
    description('Enables high-performance streaming and data manipulation in R using data.table')
)).

validate_config(Config) :-
    is_list(Config).

% Initialize component
init_component(Name, Config) :-
    % Here we could set some dynamic flags for the r_target
    % to generate data.table specific syntax
    true.

% Runtime invocation not supported
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(r_data_table))).

% Compilation - Inject library and configuration
compile_component(_Name, _Config, _Options, Code) :-
    format(string(Code), 
"# data.table component loaded
suppressPackageStartupMessages(library(data.table))

# Override default read_jsonl to use data.table for performance
read_jsonl <- function(file_path) {
    con <- file(file_path, \"r\")
    lines <- readLines(con)
    close(con)
    return(rbindlist(lapply(lines, jsonlite::fromJSON)))
}
", []).

% Register self
:- initialization((
    register_component_type(source, r_data_table, r_data_table_component, [
        description("R data.table performance component")
    ])
), now).
