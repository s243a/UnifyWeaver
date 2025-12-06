/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Pipe Glue - Generate glue code for pipe-based inter-target communication
 *
 * This module generates reader/writer code for TSV and JSON pipe protocols,
 * enabling targets to communicate via Unix pipes.
 */

:- module(pipe_glue, [
    % Writer generation (producer side)
    generate_pipe_writer/4,     % generate_pipe_writer(Target, Fields, Options, Code)

    % Reader generation (consumer side)
    generate_pipe_reader/4,     % generate_pipe_reader(Target, Fields, Options, Code)

    % Format helpers
    generate_tsv_writer/3,      % generate_tsv_writer(Target, Fields, Code)
    generate_tsv_reader/3,      % generate_tsv_reader(Target, Fields, Code)
    generate_json_writer/3,     % generate_json_writer(Target, Fields, Code)
    generate_json_reader/3,     % generate_json_reader(Target, Fields, Code)

    % Orchestrator generation
    generate_pipeline_script/3  % generate_pipeline_script(Steps, Options, Script)
]).

:- use_module(library(lists)).

%% ============================================
%% Writer Generation (Producer Side)
%% ============================================

%% generate_pipe_writer(+Target, +Fields, +Options, -Code)
%  Generate code that writes records to stdout in the specified format.
%
%  Target: awk, python, bash, go, rust, etc.
%  Fields: List of field names
%  Options: [format(tsv|json), ...]
%  Code: Generated code string
%
generate_pipe_writer(Target, Fields, Options, Code) :-
    (   member(format(json), Options)
    ->  generate_json_writer(Target, Fields, Code)
    ;   generate_tsv_writer(Target, Fields, Code)
    ).

%% ============================================
%% TSV Writers
%% ============================================

%% generate_tsv_writer(+Target, +Fields, -Code)
%  Generate TSV output code for a target.
%
generate_tsv_writer(awk, Fields, Code) :-
    fields_to_awk_print(Fields, PrintExpr),
    format(atom(Code), 'print ~w', [PrintExpr]).

generate_tsv_writer(python, Fields, Code) :-
    fields_to_python_list(Fields, FieldList),
    format(atom(Code),
        'print("\\t".join(str(x) for x in [~w]))',
        [FieldList]).

generate_tsv_writer(bash, Fields, Code) :-
    fields_to_bash_echo(Fields, EchoExpr),
    format(atom(Code), 'echo -e "~w"', [EchoExpr]).

generate_tsv_writer(go, Fields, Code) :-
    fields_to_go_printf(Fields, FormatStr, Args),
    format(atom(Code),
        'fmt.Printf("~w\\n", ~w)',
        [FormatStr, Args]).

generate_tsv_writer(rust, Fields, Code) :-
    fields_to_rust_println(Fields, FormatStr, Args),
    format(atom(Code),
        'println!("~w", ~w);',
        [FormatStr, Args]).

%% ============================================
%% TSV Readers
%% ============================================

%% generate_tsv_reader(+Target, +Fields, -Code)
%  Generate TSV input parsing code for a target.
%
generate_tsv_reader(awk, Fields, Code) :-
    % AWK automatically splits on FS, fields are $1, $2, etc.
    fields_to_awk_assign(Fields, Assignments),
    format(atom(Code), '~w', [Assignments]).

generate_tsv_reader(python, Fields, Code) :-
    fields_to_python_unpack(Fields, UnpackCode),
    format(atom(Code),
'import sys
for line in sys.stdin:
    fields = line.rstrip("\\n").split("\\t")
    ~w
    # Process record here',
        [UnpackCode]).

generate_tsv_reader(bash, Fields, Code) :-
    fields_to_bash_read(Fields, ReadVars),
    format(atom(Code),
'while IFS=$\'\\t\' read -r ~w; do
    # Process record here
done',
        [ReadVars]).

generate_tsv_reader(go, Fields, Code) :-
    fields_to_go_scan(Fields, ScanCode),
    format(atom(Code),
'scanner := bufio.NewScanner(os.Stdin)
for scanner.Scan() {
    fields := strings.Split(scanner.Text(), "\\t")
    ~w
    // Process record here
}',
        [ScanCode]).

generate_tsv_reader(rust, Fields, Code) :-
    fields_to_rust_parse(Fields, ParseCode),
    format(atom(Code),
'use std::io::{self, BufRead};
let stdin = io::stdin();
for line in stdin.lock().lines() {
    let line = line.unwrap();
    let fields: Vec<&str> = line.split(\'\\t\').collect();
    ~w
    // Process record here
}',
        [ParseCode]).

%% ============================================
%% JSON Writers
%% ============================================

%% generate_json_writer(+Target, +Fields, -Code)
%  Generate JSON Lines output code for a target.
%
generate_json_writer(awk, Fields, Code) :-
    fields_to_awk_json(Fields, JsonExpr),
    format(atom(Code), 'print "~w"', [JsonExpr]).

generate_json_writer(python, Fields, Code) :-
    fields_to_python_dict(Fields, DictExpr),
    format(atom(Code),
'import json
print(json.dumps(~w))',
        [DictExpr]).

generate_json_writer(go, Fields, Code) :-
    fields_to_go_json(Fields, StructCode, MarshalCode),
    format(atom(Code),
'~w
jsonBytes, _ := json.Marshal(~w)
fmt.Println(string(jsonBytes))',
        [StructCode, MarshalCode]).

generate_json_writer(rust, Fields, Code) :-
    fields_to_rust_json(Fields, JsonCode),
    format(atom(Code),
'let json = serde_json::json!(~w);
println!("{}", json);',
        [JsonCode]).

%% ============================================
%% JSON Readers
%% ============================================

%% generate_json_reader(+Target, +Fields, -Code)
%  Generate JSON Lines input parsing code for a target.
%
generate_json_reader(python, Fields, Code) :-
    fields_to_python_json_extract(Fields, ExtractCode),
    format(atom(Code),
'import sys, json
for line in sys.stdin:
    record = json.loads(line)
    ~w
    # Process record here',
        [ExtractCode]).

generate_json_reader(go, Fields, Code) :-
    fields_to_go_json_decode(Fields, DecodeCode),
    format(atom(Code),
'scanner := bufio.NewScanner(os.Stdin)
for scanner.Scan() {
    var record map[string]interface{}
    json.Unmarshal([]byte(scanner.Text()), &record)
    ~w
    // Process record here
}',
        [DecodeCode]).

generate_json_reader(rust, Fields, Code) :-
    fields_to_rust_json_parse(Fields, ParseCode),
    format(atom(Code),
'use std::io::{self, BufRead};
use serde_json::Value;
let stdin = io::stdin();
for line in stdin.lock().lines() {
    let record: Value = serde_json::from_str(&line.unwrap()).unwrap();
    ~w
    // Process record here
}',
        [ParseCode]).

%% ============================================
%% Reader/Writer with Options
%% ============================================

%% generate_pipe_reader(+Target, +Fields, +Options, -Code)
%  Generate code that reads records from stdin.
%
generate_pipe_reader(Target, Fields, Options, Code) :-
    (   member(format(json), Options)
    ->  generate_json_reader(Target, Fields, Code)
    ;   generate_tsv_reader(Target, Fields, Code)
    ).

%% ============================================
%% Pipeline Orchestrator
%% ============================================

%% generate_pipeline_script(+Steps, +Options, -Script)
%  Generate a bash script that orchestrates a pipeline of targets.
%
%  Steps: List of step(Pred/Arity, Target, ScriptPath)
%  Options: [input(File), output(File), ...]
%  Script: Generated bash script
%
generate_pipeline_script(Steps, Options, Script) :-
    steps_to_commands(Steps, Commands),
    (   member(input(InputFile), Options)
    ->  format(atom(InputCmd), 'cat "~w"', [InputFile])
    ;   InputCmd = 'cat -'  % Read from stdin
    ),
    (   member(output(OutputFile), Options)
    ->  format(atom(OutputRedir), ' > "~w"', [OutputFile])
    ;   OutputRedir = ''
    ),
    join_commands(Commands, PipelineCmd),
    format(atom(Script),
'#!/bin/bash
# Generated UnifyWeaver Pipeline
set -euo pipefail

~w \\
    | ~w~w
',
        [InputCmd, PipelineCmd, OutputRedir]).

%% steps_to_commands(+Steps, -Commands)
%  Convert pipeline steps to shell commands.
%
steps_to_commands([], []).
steps_to_commands([step(_, Target, ScriptPath)|Rest], [Cmd|RestCmds]) :-
    target_to_command(Target, ScriptPath, Cmd),
    steps_to_commands(Rest, RestCmds).

%% target_to_command(+Target, +ScriptPath, -Command)
%  Generate the shell command to run a target's script.
%
target_to_command(awk, ScriptPath, Cmd) :-
    format(atom(Cmd), 'awk -f "~w"', [ScriptPath]).
target_to_command(python, ScriptPath, Cmd) :-
    format(atom(Cmd), 'python3 "~w"', [ScriptPath]).
target_to_command(bash, ScriptPath, Cmd) :-
    format(atom(Cmd), 'bash "~w"', [ScriptPath]).
target_to_command(go, ScriptPath, Cmd) :-
    % Assumes pre-compiled binary
    format(atom(Cmd), '"~w"', [ScriptPath]).
target_to_command(rust, ScriptPath, Cmd) :-
    % Assumes pre-compiled binary
    format(atom(Cmd), '"~w"', [ScriptPath]).

%% join_commands(+Commands, -Joined)
%  Join commands with pipe and line continuation.
%
join_commands([Cmd], Cmd) :- !.
join_commands([Cmd|Rest], Joined) :-
    join_commands(Rest, RestJoined),
    format(atom(Joined), '~w \\~n    | ~w', [Cmd, RestJoined]).

%% ============================================
%% Field Conversion Helpers
%% ============================================

%% AWK helpers
fields_to_awk_print(Fields, Expr) :-
    length(Fields, N),
    numlist(1, N, Indices),
    maplist(awk_field_ref, Indices, Refs),
    atomic_list_concat(Refs, ' "\\t" ', Expr).

awk_field_ref(N, Ref) :-
    format(atom(Ref), '$~w', [N]).

fields_to_awk_assign(Fields, Assignments) :-
    length(Fields, N),
    numlist(1, N, Indices),
    maplist(awk_field_assign, Fields, Indices, AssignList),
    atomic_list_concat(AssignList, '\n    ', Assignments).

awk_field_assign(Field, N, Assign) :-
    format(atom(Assign), '~w = $~w', [Field, N]).

fields_to_awk_json(Fields, JsonExpr) :-
    maplist(awk_json_field, Fields, JsonFields),
    atomic_list_concat(JsonFields, ', ', JsonBody),
    format(atom(JsonExpr), '{~w}', [JsonBody]).

awk_json_field(Field, JsonField) :-
    format(atom(JsonField), '\\"~w\\": \\"" ~w "\\""', [Field, Field]).

%% Python helpers
fields_to_python_list(Fields, FieldList) :-
    atomic_list_concat(Fields, ', ', FieldList).

fields_to_python_unpack(Fields, Code) :-
    length(Fields, N),
    numlist(0, N, Indices0),
    Indices0 = [_|Indices],  % Skip 0, use 0..N-1
    maplist(python_field_assign, Fields, Indices, Assigns),
    atomic_list_concat(Assigns, '\n    ', Code).

python_field_assign(Field, Index, Assign) :-
    format(atom(Assign), '~w = fields[~w]', [Field, Index]).

fields_to_python_dict(Fields, DictExpr) :-
    maplist(python_dict_entry, Fields, Entries),
    atomic_list_concat(Entries, ', ', Body),
    format(atom(DictExpr), '{~w}', [Body]).

python_dict_entry(Field, Entry) :-
    format(atom(Entry), '"~w": ~w', [Field, Field]).

fields_to_python_json_extract(Fields, Code) :-
    maplist(python_json_extract, Fields, Extracts),
    atomic_list_concat(Extracts, '\n    ', Code).

python_json_extract(Field, Extract) :-
    format(atom(Extract), '~w = record.get("~w")', [Field, Field]).

%% Bash helpers
fields_to_bash_echo(Fields, EchoExpr) :-
    maplist(bash_var_ref, Fields, Refs),
    atomic_list_concat(Refs, '\\t', EchoExpr).

bash_var_ref(Field, Ref) :-
    format(atom(Ref), '$~w', [Field]).

fields_to_bash_read(Fields, ReadVars) :-
    atomic_list_concat(Fields, ' ', ReadVars).

%% Go helpers
fields_to_go_printf(Fields, FormatStr, Args) :-
    length(Fields, N),
    findall('%v', between(1, N, _), Formats),
    atomic_list_concat(Formats, '\\t', FormatStr),
    atomic_list_concat(Fields, ', ', Args).

fields_to_go_scan(Fields, Code) :-
    length(Fields, N),
    numlist(0, N, Indices0),
    Indices0 = [_|Indices],
    maplist(go_field_assign, Fields, Indices, Assigns),
    atomic_list_concat(Assigns, '\n    ', Code).

go_field_assign(Field, Index, Assign) :-
    format(atom(Assign), '~w := fields[~w]', [Field, Index]).

fields_to_go_json(Fields, StructCode, MarshalCode) :-
    % Simplified: use map
    StructCode = '',
    maplist(go_map_entry, Fields, Entries),
    atomic_list_concat(Entries, ', ', Body),
    format(atom(MarshalCode), 'map[string]interface{}{~w}', [Body]).

go_map_entry(Field, Entry) :-
    format(atom(Entry), '"~w": ~w', [Field, Field]).

fields_to_go_json_decode(Fields, Code) :-
    maplist(go_json_extract, Fields, Extracts),
    atomic_list_concat(Extracts, '\n    ', Code).

go_json_extract(Field, Extract) :-
    format(atom(Extract), '~w := record["~w"]', [Field, Field]).

%% Rust helpers
fields_to_rust_println(Fields, FormatStr, Args) :-
    length(Fields, N),
    findall('{}', between(1, N, _), Formats),
    atomic_list_concat(Formats, '\\t', FormatStr),
    atomic_list_concat(Fields, ', ', Args).

fields_to_rust_parse(Fields, Code) :-
    length(Fields, N),
    numlist(0, N, Indices0),
    Indices0 = [_|Indices],
    maplist(rust_field_assign, Fields, Indices, Assigns),
    atomic_list_concat(Assigns, '\n    ', Code).

rust_field_assign(Field, Index, Assign) :-
    format(atom(Assign), 'let ~w = fields[~w];', [Field, Index]).

fields_to_rust_json(Fields, JsonCode) :-
    maplist(rust_json_entry, Fields, Entries),
    atomic_list_concat(Entries, ', ', Body),
    format(atom(JsonCode), '{~w}', [Body]).

rust_json_entry(Field, Entry) :-
    format(atom(Entry), '"~w": ~w', [Field, Field]).

fields_to_rust_json_parse(Fields, Code) :-
    maplist(rust_json_extract, Fields, Extracts),
    atomic_list_concat(Extracts, '\n    ', Code).

rust_json_extract(Field, Extract) :-
    format(atom(Extract), 'let ~w = &record["~w"];', [Field, Field]).
