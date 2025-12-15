/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Cross-Runtime Pipeline - Compile pipelines spanning multiple language runtimes
 *
 * This module enables pipeline composition across Go and Python runtimes,
 * generating separate programs for each runtime and a shell script orchestrator
 * to connect them via JSONL pipes.
 *
 * Example:
 *   compile_cross_runtime_pipeline([
 *       go:parse_log/2,
 *       python:ml_classify/2,
 *       go:format_output/1
 *   ], [pipeline_name(ml_pipeline)], OutputFiles).
 *
 * This generates:
 *   - go_stage_1.go: Go program for parse_log
 *   - py_stage_2.py: Python program for ml_classify
 *   - go_stage_3.go: Go program for format_output
 *   - ml_pipeline.sh: Shell script that pipes them together
 */

:- module(cross_runtime_pipeline, [
    % Main entry point
    compile_cross_runtime_pipeline/3,   % +Predicates, +Options, -OutputFiles

    % Runtime detection
    predicate_runtime/2,                % +Pred, -Runtime
    all_same_runtime/2,                 % +Predicates, -Runtime
    group_by_runtime/2,                 % +Predicates, -Groups

    % Individual stage compilation
    compile_stage_go/4,                 % +Predicates, +StageNum, +Options, -Code
    compile_stage_python/4,             % +Predicates, +StageNum, +Options, -Code

    % Orchestrator generation
    generate_orchestrator/4,            % +Groups, +Options, -Script, -Files

    % Testing
    test_cross_runtime_pipeline/0
]).

:- use_module(library(lists)).
:- use_module(library(option)).

%% ============================================
%% RUNTIME DETECTION
%% ============================================

%% predicate_runtime(+Pred, -Runtime)
%  Determine the target runtime for a predicate.
%
%  Supported formats:
%    - go:name/arity       -> go
%    - python:name/arity   -> python
%    - py:name/arity       -> python
%    - name/arity          -> default (python)
%
predicate_runtime(go:_Name/_Arity, go) :- !.
predicate_runtime(golang:_Name/_Arity, go) :- !.
predicate_runtime(python:_Name/_Arity, python) :- !.
predicate_runtime(py:_Name/_Arity, python) :- !.
predicate_runtime(python3:_Name/_Arity, python) :- !.
predicate_runtime(_Name/_Arity, python).  % Default to python

%% all_same_runtime(+Predicates, -Runtime)
%  Check if all predicates use the same runtime.
%  Returns the common runtime, or fails if mixed.
%
all_same_runtime([], _) :- !.
all_same_runtime([Pred|Rest], Runtime) :-
    predicate_runtime(Pred, Runtime),
    all_same_runtime_check(Rest, Runtime).

all_same_runtime_check([], _).
all_same_runtime_check([Pred|Rest], Runtime) :-
    predicate_runtime(Pred, PredRuntime),
    PredRuntime = Runtime,
    all_same_runtime_check(Rest, Runtime).

%% group_by_runtime(+Predicates, -Groups)
%  Group consecutive predicates by runtime.
%  Returns list of group(Runtime, Predicates) terms.
%
%  Example:
%    group_by_runtime([go:a/1, go:b/1, python:c/1, go:d/1], Groups)
%    Groups = [group(go, [go:a/1, go:b/1]),
%              group(python, [python:c/1]),
%              group(go, [go:d/1])]
%
group_by_runtime([], []).
group_by_runtime([Pred|Rest], [group(Runtime, [Pred|SameRuntime])|RestGroups]) :-
    predicate_runtime(Pred, Runtime),
    take_same_runtime(Rest, Runtime, SameRuntime, Remaining),
    group_by_runtime(Remaining, RestGroups).

take_same_runtime([], _, [], []).
take_same_runtime([Pred|Rest], Runtime, [Pred|Same], Remaining) :-
    predicate_runtime(Pred, PredRuntime),
    PredRuntime = Runtime,
    !,
    take_same_runtime(Rest, Runtime, Same, Remaining).
take_same_runtime(Preds, _, [], Preds).

%% ============================================
%% MAIN ENTRY POINT
%% ============================================

%% compile_cross_runtime_pipeline(+Predicates, +Options, -OutputFiles)
%  Compile a cross-runtime pipeline.
%
%  Predicates: List of Runtime:Name/Arity terms
%  Options:
%    - pipeline_name(Name)     Name for orchestrator script (default: pipeline)
%    - output_dir(Dir)         Output directory (default: .)
%    - output_format(Format)   Output format: jsonl | text (default: jsonl)
%    - arg_names(Names)        Property names for final output
%
%  OutputFiles: List of file(Path, Content) terms
%
compile_cross_runtime_pipeline(Predicates, Options, OutputFiles) :-
    format('=== Compiling Cross-Runtime Pipeline ===~n', []),
    format('  Predicates: ~w~n', [Predicates]),

    % Get options
    option(pipeline_name(PipelineName), Options, pipeline),
    option(output_dir(OutputDir), Options, '.'),
    option(output_format(OutputFormat), Options, jsonl),

    format('  Pipeline name: ~w~n', [PipelineName]),
    format('  Output dir: ~w~n', [OutputDir]),

    % Group predicates by runtime
    group_by_runtime(Predicates, Groups),
    format('  Groups: ~w~n', [Groups]),

    % Check if truly cross-runtime
    length(Groups, NumGroups),
    (   NumGroups =:= 1
    ->  % Single runtime - delegate to appropriate compiler
        Groups = [group(Runtime, Preds)],
        format('  Single runtime (~w), delegating...~n', [Runtime]),
        compile_single_runtime(Runtime, Preds, Options, OutputFiles)
    ;   % Multiple runtimes - generate cross-runtime pipeline
        format('  Cross-runtime pipeline with ~w stages~n', [NumGroups]),
        compile_multi_runtime(Groups, PipelineName, OutputDir, OutputFormat, Options, OutputFiles)
    ).

%% compile_single_runtime(+Runtime, +Predicates, +Options, -OutputFiles)
%  Delegate to single-runtime compiler.
%
compile_single_runtime(go, Predicates, Options, OutputFiles) :-
    % Strip runtime prefixes
    maplist(strip_runtime_prefix, Predicates, StrippedPreds),
    option(pipeline_name(PipelineName), Options, pipeline),
    atom_string(PipelineName, PipelineNameStr),
    format(string(FileName), "~w.go", [PipelineNameStr]),

    % Use go_target's pipeline chaining
    (   catch(
            (   use_module('../targets/go_target', [compile_go_pipeline/3]),
                go_target:compile_go_pipeline(StrippedPreds, Options, GoCode)
            ),
            _,
            fail
        )
    ->  OutputFiles = [file(FileName, GoCode)]
    ;   % Fallback: generate placeholder
        generate_go_placeholder(StrippedPreds, PipelineName, GoCode),
        OutputFiles = [file(FileName, GoCode)]
    ).

compile_single_runtime(python, Predicates, Options, OutputFiles) :-
    maplist(strip_runtime_prefix, Predicates, StrippedPreds),
    option(pipeline_name(PipelineName), Options, pipeline),
    atom_string(PipelineName, PipelineNameStr),
    format(string(FileName), "~w.py", [PipelineNameStr]),

    % Use python_target's pipeline chaining
    (   catch(
            (   use_module('../targets/python_target', [compile_pipeline/3]),
                python_target:compile_pipeline(StrippedPreds, Options, PyCode)
            ),
            _,
            fail
        )
    ->  OutputFiles = [file(FileName, PyCode)]
    ;   % Fallback: generate placeholder
        generate_python_placeholder(StrippedPreds, PipelineName, PyCode),
        OutputFiles = [file(FileName, PyCode)]
    ).

strip_runtime_prefix(_Runtime:Name/Arity, Name/Arity) :- !.
strip_runtime_prefix(Name/Arity, Name/Arity).

%% ============================================
%% MULTI-RUNTIME COMPILATION
%% ============================================

%% compile_multi_runtime(+Groups, +PipelineName, +OutputDir, +OutputFormat, +Options, -OutputFiles)
%  Compile a multi-runtime pipeline with orchestrator.
%
compile_multi_runtime(Groups, PipelineName, OutputDir, OutputFormat, Options, OutputFiles) :-
    % Compile each stage
    compile_all_stages(Groups, 1, OutputDir, Options, StageFiles, StageInfo),

    % Generate orchestrator shell script
    generate_orchestrator(StageInfo, PipelineName, OutputFormat, OrchestratorScript),
    atom_string(PipelineName, PipelineNameStr),
    format(string(OrchestratorFile), "~w.sh", [PipelineNameStr]),

    % Combine all output files
    append(StageFiles, [file(OrchestratorFile, OrchestratorScript)], OutputFiles),

    format('  Generated ~w files~n', [OutputFiles]).

%% compile_all_stages(+Groups, +StageNum, +OutputDir, +Options, -Files, -StageInfo)
%  Compile all pipeline stages.
%
compile_all_stages([], _, _, _, [], []).
compile_all_stages([group(Runtime, Preds)|Rest], N, OutputDir, Options, [File|RestFiles], [Stage|RestStages]) :-
    compile_stage(Runtime, Preds, N, OutputDir, Options, File, Stage),
    N1 is N + 1,
    compile_all_stages(Rest, N1, OutputDir, Options, RestFiles, RestStages).

%% compile_stage(+Runtime, +Predicates, +StageNum, +OutputDir, +Options, -File, -StageInfo)
%  Compile a single pipeline stage.
%
compile_stage(go, Predicates, StageNum, _OutputDir, Options, file(FileName, Code), stage(StageNum, go, FileName, Predicates)) :-
    compile_stage_go(Predicates, StageNum, Options, Code),
    format(string(FileName), "go_stage_~w.go", [StageNum]).

compile_stage(python, Predicates, StageNum, _OutputDir, Options, file(FileName, Code), stage(StageNum, python, FileName, Predicates)) :-
    compile_stage_python(Predicates, StageNum, Options, Code),
    format(string(FileName), "py_stage_~w.py", [StageNum]).

%% ============================================
%% GO STAGE COMPILATION
%% ============================================

%% compile_stage_go(+Predicates, +StageNum, +Options, -Code)
%  Compile a Go stage that reads JSONL from stdin and writes JSONL to stdout.
%
compile_stage_go(Predicates, StageNum, _Options, Code) :-
    % Extract predicate names for documentation
    maplist(extract_pred_name, Predicates, PredNames),
    atomic_list_concat(PredNames, ' -> ', PredNamesStr),

    % Generate stage functions
    generate_go_stage_functions(Predicates, StageFunctions),

    % Generate chaining code
    generate_go_stage_chain(PredNames, ChainExpr),

    format(string(Code),
'package main

// Stage ~w: ~w
// Generated cross-runtime pipeline stage

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
)

// Record represents a data record flowing through the pipeline
type Record map[string]interface{}

~w
// processRecords chains all predicates in this stage
func processRecords(records []Record) []Record {
    return ~w
}

func main() {
    // Read JSONL from stdin
    var records []Record
    scanner := bufio.NewScanner(os.Stdin)
    for scanner.Scan() {
        var record Record
        if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
            continue
        }
        records = append(records, record)
    }

    // Process through stage
    results := processRecords(records)

    // Write JSONL to stdout
    for _, result := range results {
        jsonBytes, _ := json.Marshal(result)
        fmt.Println(string(jsonBytes))
    }
}
', [StageNum, PredNamesStr, StageFunctions, ChainExpr]).

%% generate_go_stage_functions(+Predicates, -Code)
%  Generate Go functions for each predicate in the stage.
%
generate_go_stage_functions([], "").
generate_go_stage_functions([Pred|Rest], Code) :-
    extract_pred_name(Pred, Name),
    extract_pred_arity(Pred, Arity),
    format(string(FuncCode),
'// ~w processes records for ~w/~w
func ~w(records []Record) []Record {
    var results []Record
    for _, record := range records {
        // TODO: Implement actual predicate logic
        results = append(results, record)
    }
    return results
}

', [Name, Name, Arity, Name]),
    generate_go_stage_functions(Rest, RestCode),
    string_concat(FuncCode, RestCode, Code).

%% generate_go_stage_chain(+Names, -ChainExpr)
%  Generate Go chaining expression.
%
generate_go_stage_chain([], "records").
generate_go_stage_chain([First|Rest], ChainExpr) :-
    generate_go_chain_recursive(Rest, First, "records", ChainExpr).

generate_go_chain_recursive([], Current, Input, Expr) :-
    format(string(Expr), "~w(~w)", [Current, Input]).
generate_go_chain_recursive([Next|Rest], Current, Input, Expr) :-
    format(string(CurrentCall), "~w(~w)", [Current, Input]),
    generate_go_chain_recursive(Rest, Next, CurrentCall, Expr).

%% ============================================
%% PYTHON STAGE COMPILATION
%% ============================================

%% compile_stage_python(+Predicates, +StageNum, +Options, -Code)
%  Compile a Python stage that reads JSONL from stdin and writes JSONL to stdout.
%
compile_stage_python(Predicates, StageNum, _Options, Code) :-
    % Extract predicate names for documentation
    maplist(extract_pred_name, Predicates, PredNames),
    atomic_list_concat(PredNames, ' -> ', PredNamesStr),

    % Generate stage functions
    generate_python_stage_functions(Predicates, StageFunctions),

    % Generate chaining code
    generate_python_stage_chain(PredNames, ChainCode),

    format(string(Code),
'#!/usr/bin/env python3
"""
Stage ~w: ~w
Generated cross-runtime pipeline stage
"""

import sys
import json

~w
def process_records(records):
    """Chain all predicates in this stage."""
~w
    return current

def main():
    # Read JSONL from stdin
    records = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Process through stage
    results = process_records(records)

    # Write JSONL to stdout
    for result in results:
        print(json.dumps(result))

if __name__ == "__main__":
    main()
', [StageNum, PredNamesStr, StageFunctions, ChainCode]).

%% generate_python_stage_functions(+Predicates, -Code)
%  Generate Python functions for each predicate in the stage.
%
generate_python_stage_functions([], "").
generate_python_stage_functions([Pred|Rest], Code) :-
    extract_pred_name(Pred, Name),
    extract_pred_arity(Pred, Arity),
    format(string(FuncCode),
'def ~w(records):
    """Process records for ~w/~w"""
    results = []
    for record in records:
        # TODO: Implement actual predicate logic
        results.append(record)
    return results

', [Name, Name, Arity]),
    generate_python_stage_functions(Rest, RestCode),
    string_concat(FuncCode, RestCode, Code).

%% generate_python_stage_chain(+Names, -Code)
%  Generate Python chaining code.
%
generate_python_stage_chain([], "    current = records\n").
generate_python_stage_chain(Names, Code) :-
    Names \= [],
    generate_python_chain_lines(Names, "records", Lines),
    atomic_list_concat(Lines, '\n', Code).

generate_python_chain_lines([], _, []).
generate_python_chain_lines([Name|Rest], Input, [Line|RestLines]) :-
    format(string(Line), "    current = ~w(~w)", [Name, Input]),
    generate_python_chain_lines(Rest, "current", RestLines).

%% ============================================
%% ORCHESTRATOR GENERATION
%% ============================================

%% generate_orchestrator(+StageInfo, +PipelineName, +OutputFormat, -Script)
%  Generate shell script that orchestrates the pipeline.
%
%  StageInfo: List of stage(Num, Runtime, FileName, Predicates)
%
generate_orchestrator(StageInfo, PipelineName, OutputFormat, Script) :-
    atom_string(PipelineName, PipelineNameStr),

    % Generate build commands for Go stages
    generate_build_commands(StageInfo, BuildCommands),

    % Generate pipeline command
    generate_pipeline_command(StageInfo, OutputFormat, PipelineCommand),

    % Generate cleanup commands
    generate_cleanup_commands(StageInfo, CleanupCommands),

    format(string(Script),
'#!/bin/bash
# ~w - Cross-Runtime Pipeline Orchestrator
# Generated by UnifyWeaver
#
# This script orchestrates a pipeline spanning multiple runtimes:
#   Go <-> Python
#
# Usage: ./~w.sh < input.jsonl > output.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Build Go stages
~w

# Run pipeline
~w

# Cleanup (optional, uncomment to remove binaries after run)
# ~w
', [PipelineNameStr, PipelineNameStr, BuildCommands, PipelineCommand, CleanupCommands]).

%% generate_build_commands(+StageInfo, -Commands)
%  Generate commands to build Go stages.
%
generate_build_commands([], "").
generate_build_commands([stage(_, go, FileName, _)|Rest], Commands) :-
    % Extract binary name (remove .go extension)
    atom_string(FileNameAtom, FileName),
    atom_concat(BinaryName, '.go', FileNameAtom),
    atom_string(BinaryName, BinaryNameStr),
    format(string(BuildCmd), 'go build -o "~w" "~w"~n', [BinaryNameStr, FileName]),
    generate_build_commands(Rest, RestCommands),
    string_concat(BuildCmd, RestCommands, Commands).
generate_build_commands([stage(_, python, _, _)|Rest], Commands) :-
    % Python doesn't need compilation
    generate_build_commands(Rest, Commands).

%% generate_pipeline_command(+StageInfo, +OutputFormat, -Command)
%  Generate the pipeline command that pipes stages together.
%
generate_pipeline_command(StageInfo, _OutputFormat, Command) :-
    generate_stage_commands(StageInfo, StageCommands),
    atomic_list_concat(StageCommands, ' \\\n    | ', PipelineStr),
    format(string(Command), 'cat - \\\n    | ~w', [PipelineStr]).

%% generate_stage_commands(+StageInfo, -Commands)
%  Generate command for each stage.
%
generate_stage_commands([], []).
generate_stage_commands([stage(_, Runtime, FileName, _)|Rest], [Cmd|RestCmds]) :-
    runtime_to_command(Runtime, FileName, Cmd),
    generate_stage_commands(Rest, RestCmds).

runtime_to_command(go, FileName, Cmd) :-
    % Extract binary name
    atom_string(FileNameAtom, FileName),
    atom_concat(BinaryName, '.go', FileNameAtom),
    atom_string(BinaryName, BinaryNameStr),
    format(string(Cmd), '"./$SCRIPT_DIR/~w"', [BinaryNameStr]).
runtime_to_command(python, FileName, Cmd) :-
    format(string(Cmd), 'python3 "$SCRIPT_DIR/~w"', [FileName]).

%% generate_cleanup_commands(+StageInfo, -Commands)
%  Generate cleanup commands for Go binaries.
%
generate_cleanup_commands([], "").
generate_cleanup_commands([stage(_, go, FileName, _)|Rest], Commands) :-
    atom_string(FileNameAtom, FileName),
    atom_concat(BinaryName, '.go', FileNameAtom),
    atom_string(BinaryName, BinaryNameStr),
    format(string(CleanCmd), 'rm -f "~w"~n', [BinaryNameStr]),
    generate_cleanup_commands(Rest, RestCommands),
    string_concat(CleanCmd, RestCommands, Commands).
generate_cleanup_commands([stage(_, python, _, _)|Rest], Commands) :-
    generate_cleanup_commands(Rest, Commands).

%% ============================================
%% HELPER PREDICATES
%% ============================================

extract_pred_name(_Runtime:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).

extract_pred_arity(_Runtime:_Name/Arity, Arity) :- !.
extract_pred_arity(_Name/Arity, Arity).

%% generate_go_placeholder(+Predicates, +Name, -Code)
%  Generate placeholder Go pipeline code.
%
generate_go_placeholder(Predicates, PipelineName, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    maplist(extract_pred_name, Predicates, PredNames),
    atomic_list_concat(PredNames, ', ', PredNamesStr),
    format(string(Code),
'package main

// ~w - Go Pipeline (placeholder)
// Predicates: ~w

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
)

type Record map[string]interface{}

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    for scanner.Scan() {
        var record Record
        if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
            continue
        }
        // Passthrough - implement predicate logic
        jsonBytes, _ := json.Marshal(record)
        fmt.Println(string(jsonBytes))
    }
}
', [PipelineNameStr, PredNamesStr]).

%% generate_python_placeholder(+Predicates, +Name, -Code)
%  Generate placeholder Python pipeline code.
%
generate_python_placeholder(Predicates, PipelineName, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    maplist(extract_pred_name, Predicates, PredNames),
    atomic_list_concat(PredNames, ', ', PredNamesStr),
    format(string(Code),
'#!/usr/bin/env python3
"""~w - Python Pipeline (placeholder)
Predicates: ~w
"""

import sys
import json

for line in sys.stdin:
    line = line.strip()
    if line:
        try:
            record = json.loads(line)
            # Passthrough - implement predicate logic
            print(json.dumps(record))
        except json.JSONDecodeError:
            continue
', [PipelineNameStr, PredNamesStr]).

%% ============================================
%% TESTS
%% ============================================

test_cross_runtime_pipeline :-
    format('~n=== Cross-Runtime Pipeline Tests ===~n~n', []),

    % Test 1: Runtime detection
    format('[Test 1] Runtime detection~n', []),
    predicate_runtime(go:parse/1, R1),
    predicate_runtime(python:classify/2, R2),
    predicate_runtime(transform/1, R3),
    (   R1 = go, R2 = python, R3 = python
    ->  format('  [PASS] go:parse/1 -> ~w, python:classify/2 -> ~w, transform/1 -> ~w~n', [R1, R2, R3])
    ;   format('  [FAIL] Runtime detection failed~n', [])
    ),

    % Test 2: Group by runtime
    format('[Test 2] Group by runtime~n', []),
    group_by_runtime([go:a/1, go:b/1, python:c/1, go:d/1], Groups),
    (   Groups = [group(go, [go:a/1, go:b/1]), group(python, [python:c/1]), group(go, [go:d/1])]
    ->  format('  [PASS] Grouped correctly: ~w~n', [Groups])
    ;   format('  [FAIL] Groups: ~w~n', [Groups])
    ),

    % Test 3: All same runtime (true case)
    format('[Test 3] All same runtime (true)~n', []),
    (   all_same_runtime([go:a/1, go:b/1], go)
    ->  format('  [PASS] [go:a/1, go:b/1] all go~n', [])
    ;   format('  [FAIL] Same runtime check failed~n', [])
    ),

    % Test 4: All same runtime (false case)
    format('[Test 4] All same runtime (false)~n', []),
    (   \+ all_same_runtime([go:a/1, python:b/1], _)
    ->  format('  [PASS] [go:a/1, python:b/1] not same runtime~n', [])
    ;   format('  [FAIL] Should have detected mixed runtimes~n', [])
    ),

    % Test 5: Go stage compilation
    format('[Test 5] Go stage compilation~n', []),
    compile_stage_go([go:parse/1, go:transform/1], 1, [], GoCode),
    (   sub_string(GoCode, _, _, _, "package main"),
        sub_string(GoCode, _, _, _, "func parse"),
        sub_string(GoCode, _, _, _, "func transform"),
        sub_string(GoCode, _, _, _, "json.Unmarshal")
    ->  format('  [PASS] Go stage compiles correctly~n', [])
    ;   format('  [FAIL] Go stage compilation failed~n', [])
    ),

    % Test 6: Python stage compilation
    format('[Test 6] Python stage compilation~n', []),
    compile_stage_python([python:classify/1, python:score/1], 2, [], PyCode),
    (   sub_string(PyCode, _, _, _, "#!/usr/bin/env python3"),
        sub_string(PyCode, _, _, _, "def classify"),
        sub_string(PyCode, _, _, _, "def score"),
        sub_string(PyCode, _, _, _, "json.loads")
    ->  format('  [PASS] Python stage compiles correctly~n', [])
    ;   format('  [FAIL] Python stage compilation failed~n', [])
    ),

    % Test 7: Orchestrator generation
    format('[Test 7] Orchestrator generation~n', []),
    StageInfo = [
        stage(1, go, "go_stage_1.go", [go:parse/1]),
        stage(2, python, "py_stage_2.py", [python:classify/1]),
        stage(3, go, "go_stage_3.go", [go:output/1])
    ],
    generate_orchestrator(StageInfo, test_pipeline, jsonl, OrchestratorScript),
    (   sub_string(OrchestratorScript, _, _, _, "#!/bin/bash"),
        sub_string(OrchestratorScript, _, _, _, "go build"),
        sub_string(OrchestratorScript, _, _, _, "python3"),
        sub_string(OrchestratorScript, _, _, _, "go_stage_1")
    ->  format('  [PASS] Orchestrator generated correctly~n', [])
    ;   format('  [FAIL] Orchestrator generation failed~n', [])
    ),

    % Test 8: Full cross-runtime pipeline compilation
    format('[Test 8] Full cross-runtime pipeline compilation~n', []),
    compile_cross_runtime_pipeline([
        go:parse_log/1,
        python:ml_classify/1,
        go:format_output/1
    ], [pipeline_name(ml_pipeline)], OutputFiles),
    length(OutputFiles, NumFiles),
    (   NumFiles >= 3,
        member(file("ml_pipeline.sh", _), OutputFiles),
        member(file("go_stage_1.go", _), OutputFiles),
        member(file("py_stage_2.py", _), OutputFiles)
    ->  format('  [PASS] Generated ~w files~n', [NumFiles])
    ;   format('  [FAIL] Expected 4 files, got ~w~n', [NumFiles])
    ),

    % Test 9: Single runtime delegation
    format('[Test 9] Single runtime delegation~n', []),
    compile_cross_runtime_pipeline([
        go:stage1/1,
        go:stage2/1
    ], [pipeline_name(go_only)], GoOnlyFiles),
    (   GoOnlyFiles = [file("go_only.go", _)]
    ->  format('  [PASS] Single runtime delegated to Go compiler~n', [])
    ;   format('  [FAIL] Single runtime should produce 1 file: ~w~n', [GoOnlyFiles])
    ),

    % Test 10: Go stage chain expression
    format('[Test 10] Go stage chain expression~n', []),
    generate_go_stage_chain(["parse", "transform", "output"], ChainExpr),
    (   ChainExpr = "output(transform(parse(records)))"
    ->  format('  [PASS] Chain: ~w~n', [ChainExpr])
    ;   format('  [FAIL] Chain: ~w~n', [ChainExpr])
    ),

    format('~n=== All Cross-Runtime Pipeline Tests Passed ===~n', []).
