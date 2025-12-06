/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Shell Glue - Generate complete glue code for shell-based targets
 *
 * This module generates full scripts with input/output glue for AWK, Python,
 * and Bash targets to communicate via Unix pipes.
 */

:- module(shell_glue, [
    % Complete script generation
    generate_awk_script/4,      % generate_awk_script(Logic, Fields, Options, Script)
    generate_python_script/4,   % generate_python_script(Logic, Fields, Options, Script)
    generate_bash_script/4,     % generate_bash_script(Logic, Fields, Options, Script)

    % Pipeline generation
    generate_pipeline/3,        % generate_pipeline(Steps, Options, Script)

    % Format detection
    input_format/2,             % input_format(Options, Format)
    output_format/2             % output_format(Options, Format)
]).

:- use_module(library(lists)).

%% ============================================
%% Format Detection
%% ============================================

%% input_format(+Options, -Format)
%  Determine input format from options.
%
input_format(Options, Format) :-
    (   member(input_format(Format), Options)
    ->  true
    ;   member(format(Format), Options)
    ->  true
    ;   Format = tsv  % Default
    ).

%% output_format(+Options, -Format)
%  Determine output format from options.
%
output_format(Options, Format) :-
    (   member(output_format(Format), Options)
    ->  true
    ;   member(format(Format), Options)
    ->  true
    ;   Format = tsv  % Default
    ).

%% ============================================
%% AWK Script Generation
%% ============================================

%% generate_awk_script(+Logic, +Fields, +Options, -Script)
%  Generate a complete AWK script with input parsing and output formatting.
%
%  Logic: The core AWK logic (string)
%  Fields: List of input field names (atoms)
%  Options: [format(tsv|json), header(true|false), ...]
%
generate_awk_script(Logic, Fields, Options, Script) :-
    input_format(Options, InFormat),
    output_format(Options, OutFormat),
    (member(header(true), Options) -> SkipHeader = true ; SkipHeader = false),

    % Build BEGIN block
    awk_begin_block(InFormat, BeginBlock),

    % Build field assignments
    awk_field_assignments(Fields, InFormat, FieldAssigns),

    % Build output
    awk_output_code(OutFormat, Fields, OutputCode),

    % Combine
    (   SkipHeader == true
    ->  format(atom(Script),
'#!/usr/bin/awk -f
~w

NR == 1 { next }

{
~w

~w

~w
}
', [BeginBlock, FieldAssigns, Logic, OutputCode])
    ;   format(atom(Script),
'#!/usr/bin/awk -f
~w

{
~w

~w

~w
}
', [BeginBlock, FieldAssigns, Logic, OutputCode])
    ).

%% awk_begin_block(+Format, -Block)
awk_begin_block(tsv, Block) :-
    Block = 'BEGIN {\n    FS = "\\t"\n    OFS = "\\t"\n}'.
awk_begin_block(csv, Block) :-
    Block = 'BEGIN {\n    FS = ","\n    OFS = ","\n}'.
awk_begin_block(json, Block) :-
    Block = 'BEGIN {\n    # JSON input mode\n}'.

%% awk_field_assignments(+Fields, +Format, -Code)
awk_field_assignments(Fields, tsv, Code) :-
    awk_tsv_assignments(Fields, 1, Assigns),
    atomic_list_concat(Assigns, '\n    ', Code).
awk_field_assignments(Fields, csv, Code) :-
    awk_tsv_assignments(Fields, 1, Assigns),  % Same as TSV for basic CSV
    atomic_list_concat(Assigns, '\n    ', Code).
awk_field_assignments(Fields, json, Code) :-
    awk_json_assignments(Fields, Assigns),
    atomic_list_concat(Assigns, '\n    ', Code).

awk_tsv_assignments([], _, []).
awk_tsv_assignments([Field|Rest], N, [Assign|RestAssigns]) :-
    format(atom(Assign), '~w = $~w', [Field, N]),
    N1 is N + 1,
    awk_tsv_assignments(Rest, N1, RestAssigns).

awk_json_assignments([], []).
awk_json_assignments([Field|Rest], [Assign|RestAssigns]) :-
    format(atom(Assign), '~w = json_get($0, "~w")', [Field, Field]),
    awk_json_assignments(Rest, RestAssigns).

%% awk_output_code(+Format, +Fields, -Code)
awk_output_code(tsv, Fields, Code) :-
    maplist(atom_string, Fields, FieldStrs),
    atomic_list_concat(FieldStrs, ', ', FieldList),
    format(atom(Code), 'print ~w', [FieldList]).
awk_output_code(csv, Fields, Code) :-
    awk_output_code(tsv, Fields, Code).  % Same for basic output
awk_output_code(json, Fields, Code) :-
    awk_json_output(Fields, Code).

awk_json_output(Fields, Code) :-
    maplist(awk_json_field_pair, Fields, Pairs),
    atomic_list_concat(Pairs, ', ', PairStr),
    format(atom(Code), 'print "{~w}"', [PairStr]).

awk_json_field_pair(Field, Pair) :-
    format(atom(Pair), '\\"~w\\": \\"" ~w "\\""', [Field, Field]).

%% ============================================
%% Python Script Generation
%% ============================================

%% generate_python_script(+Logic, +Fields, +Options, -Script)
%  Generate a complete Python script with input parsing and output formatting.
%
generate_python_script(Logic, Fields, Options, Script) :-
    input_format(Options, InFormat),
    output_format(Options, OutFormat),
    (member(header(true), Options) -> SkipHeader = "True" ; SkipHeader = "False"),

    % Build imports
    python_imports(InFormat, OutFormat, Imports),

    % Build reader
    python_reader(InFormat, Fields, SkipHeader, ReaderCode),

    % Build writer
    python_writer(OutFormat, Fields, WriterCode),

    format(atom(Script),
'#!/usr/bin/env python3
~w

def process_record(record):
    \"\"\"Process a single record. Modify this function.\"\"\"
~w
    return record

def main():
~w

    for record in read_input():
        result = process_record(record)
        if result is not None:
~w

if __name__ == "__main__":
    main()
', [Imports, Logic, ReaderCode, WriterCode]).

%% python_imports(+InFormat, +OutFormat, -Imports)
python_imports(InFormat, OutFormat, Imports) :-
    findall(Import, (
        python_format_import(InFormat, Import)
    ;   python_format_import(OutFormat, Import)
    ), ImportList0),
    sort(ImportList0, ImportList),
    (   ImportList == []
    ->  Imports = 'import sys'
    ;   atomic_list_concat(['import sys'|ImportList], '\n', Imports)
    ).

python_format_import(json, 'import json').
python_format_import(csv, 'import csv').

%% python_reader(+Format, +Fields, +SkipHeader, -Code)
python_reader(tsv, Fields, SkipHeader, Code) :-
    python_field_dict(Fields, FieldDict),
    format(atom(Code),
'    skip_header = ~w

    def read_input():
        first = True
        for line in sys.stdin:
            if first and skip_header:
                first = False
                continue
            first = False
            fields = line.rstrip("\\n").split("\\t")
            yield ~w', [SkipHeader, FieldDict]).

python_reader(csv, Fields, SkipHeader, Code) :-
    python_field_list(Fields, FieldList),
    format(atom(Code),
'    skip_header = ~w
    reader = csv.reader(sys.stdin)

    def read_input():
        first = True
        for row in reader:
            if first and skip_header:
                first = False
                continue
            first = False
            ~w = row
            yield {~w}', [SkipHeader, FieldList, FieldList]).

python_reader(json, _Fields, _SkipHeader, Code) :-
    format(atom(Code),
'    def read_input():
        for line in sys.stdin:
            record = json.loads(line)
            yield record', []).

python_field_dict(Fields, Dict) :-
    length(Fields, N),
    N1 is N - 1,
    numlist(0, N1, Indices),
    maplist(python_dict_pair, Fields, Indices, Pairs),
    atomic_list_concat(Pairs, ', ', PairStr),
    format(atom(Dict), '{~w}', [PairStr]).

python_dict_pair(Field, Index, Pair) :-
    format(atom(Pair), '"~w": fields[~w]', [Field, Index]).

python_field_entry(Field, Entry) :-
    format(atom(Entry), '"~w"', [Field]).

python_field_list(Fields, List) :-
    atomic_list_concat(Fields, ', ', List).

%% python_writer(+Format, +Fields, -Code)
python_writer(tsv, Fields, Code) :-
    maplist(python_result_field, Fields, ResultFields),
    atomic_list_concat(ResultFields, ', ', FieldStr),
    format(atom(Code),
'            print("\\t".join(str(x) for x in [~w]))', [FieldStr]).

python_writer(csv, Fields, Code) :-
    python_writer(tsv, Fields, Code).  % Simplified

python_writer(json, _Fields, Code) :-
    Code = '            print(json.dumps(result))'.

python_result_field(Field, Result) :-
    format(atom(Result), 'result["~w"]', [Field]).

%% ============================================
%% Bash Script Generation
%% ============================================

%% generate_bash_script(+Logic, +Fields, +Options, -Script)
%  Generate a complete Bash script with input parsing and output formatting.
%
generate_bash_script(Logic, Fields, Options, Script) :-
    input_format(Options, InFormat),
    output_format(Options, OutFormat),
    (member(header(true), Options) -> SkipHeader = true ; SkipHeader = false),

    bash_reader(InFormat, Fields, SkipHeader, ReaderCode),
    bash_writer(OutFormat, Fields, WriterCode),

    format(atom(Script),
'#!/bin/bash
set -euo pipefail

~w
~w

~w
done
', [ReaderCode, Logic, WriterCode]).

%% bash_reader(+Format, +Fields, +SkipHeader, -Code)
bash_reader(tsv, Fields, SkipHeader, Code) :-
    atomic_list_concat(Fields, ' ', FieldVars),
    (   SkipHeader == true
    ->  format(atom(Code),
'# Skip header
read -r _header

while IFS=$\'\\t\' read -r ~w; do', [FieldVars])
    ;   format(atom(Code),
'while IFS=$\'\\t\' read -r ~w; do', [FieldVars])
    ).

bash_reader(csv, Fields, SkipHeader, Code) :-
    bash_reader(tsv, Fields, SkipHeader, Code).  % Simplified

bash_reader(json, Fields, _SkipHeader, Code) :-
    maplist(bash_jq_extract, Fields, Extracts),
    atomic_list_concat(Extracts, '\n    ', ExtractCode),
    format(atom(Code),
'while read -r line; do
    ~w', [ExtractCode]).

bash_jq_extract(Field, Extract) :-
    format(atom(Extract), '~w=$(echo "$line" | jq -r \'.~w\')', [Field, Field]).

%% bash_writer(+Format, +Fields, -Code)
bash_writer(tsv, Fields, Code) :-
    maplist(bash_var_ref, Fields, Refs),
    atomic_list_concat(Refs, '\\t', RefStr),
    format(atom(Code), '    echo -e "~w"', [RefStr]).

bash_writer(csv, Fields, Code) :-
    bash_writer(tsv, Fields, Code).

bash_writer(json, Fields, Code) :-
    maplist(bash_json_field, Fields, JsonFields),
    atomic_list_concat(JsonFields, ', ', JsonStr),
    format(atom(Code), '    echo "\\"{~w}\\""', [JsonStr]).

bash_var_ref(Field, Ref) :-
    format(atom(Ref), '$~w', [Field]).

bash_json_field(Field, JsonField) :-
    format(atom(JsonField), '\\"~w\\": \\"$~w\\"', [Field, Field]).

%% ============================================
%% Pipeline Generation
%% ============================================

%% generate_pipeline(+Steps, +Options, -Script)
%  Generate a complete pipeline orchestrator script.
%
%  Steps: List of step(Name, Target, ScriptPath, StepOptions)
%  Options: [input(File), output(File), ...]
%
generate_pipeline(Steps, Options, Script) :-
    % Input handling
    (   member(input(InputFile), Options)
    ->  format(atom(InputCmd), 'cat "~w"', [InputFile])
    ;   InputCmd = 'cat'
    ),

    % Output handling
    (   member(output(OutputFile), Options)
    ->  format(atom(OutputRedir), ' > "~w"', [OutputFile])
    ;   OutputRedir = ''
    ),

    % Build pipeline commands
    steps_to_pipeline(Steps, PipelineCmd),

    % Error handling option
    (   member(stop_on_error(false), Options)
    ->  SetFlags = 'set -uo pipefail'
    ;   SetFlags = 'set -euo pipefail'
    ),

    format(atom(Script),
'#!/bin/bash
# Generated UnifyWeaver Pipeline
~w

~w \\
    | ~w~w
', [SetFlags, InputCmd, PipelineCmd, OutputRedir]).

%% steps_to_pipeline(+Steps, -PipelineCmd)
steps_to_pipeline([Step], Cmd) :-
    !,
    step_to_command(Step, Cmd).
steps_to_pipeline([Step|Rest], Cmd) :-
    step_to_command(Step, StepCmd),
    steps_to_pipeline(Rest, RestCmd),
    format(atom(Cmd), '~w \\~n    | ~w', [StepCmd, RestCmd]).

%% step_to_command(+Step, -Command)
step_to_command(step(_Name, awk, ScriptPath, _Opts), Cmd) :-
    format(atom(Cmd), 'awk -f "~w"', [ScriptPath]).
step_to_command(step(_Name, python, ScriptPath, _Opts), Cmd) :-
    format(atom(Cmd), 'python3 "~w"', [ScriptPath]).
step_to_command(step(_Name, bash, ScriptPath, _Opts), Cmd) :-
    format(atom(Cmd), 'bash "~w"', [ScriptPath]).
step_to_command(step(_Name, go, BinaryPath, _Opts), Cmd) :-
    format(atom(Cmd), '"~w"', [BinaryPath]).
step_to_command(step(_Name, rust, BinaryPath, _Opts), Cmd) :-
    format(atom(Cmd), '"~w"', [BinaryPath]).
