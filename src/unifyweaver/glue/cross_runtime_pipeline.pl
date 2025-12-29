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
    compile_stage_rust_ffi_go/4,        % +Predicates, +StageNum, +Options, -Code
    compile_stage_rust_ffi_node/4,      % +Predicates, +StageNum, +Options, -Code

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
%    - go:name/arity            -> go (native Go)
%    - rust_ffi:go:name/arity   -> rust_ffi_go (Go via Rust FFI bridge to Python)
%    - rust_ffi:node:name/arity -> rust_ffi_node (Node.js via Rust FFI bridge)
%    - python:name/arity        -> python
%    - py:name/arity            -> python
%    - name/arity               -> default (python)
%
%  The rust_ffi variants generate a Rust cdylib that provides C FFI to RPyC,
%  allowing Go/Node/etc to call Python without native CPython embedding.
%
predicate_runtime(rust_ffi:go:_Name/_Arity, rust_ffi_go) :- !.
predicate_runtime(rust_ffi:golang:_Name/_Arity, rust_ffi_go) :- !.
predicate_runtime(rust_ffi:node:_Name/_Arity, rust_ffi_node) :- !.
predicate_runtime(rust_ffi:nodejs:_Name/_Arity, rust_ffi_node) :- !.
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

%% Rust FFI Bridge stages - Go accessing Python via Rust FFI
compile_stage(rust_ffi_go, Predicates, StageNum, _OutputDir, Options, file(FileName, Code), stage(StageNum, rust_ffi_go, FileName, Predicates)) :-
    compile_stage_rust_ffi_go(Predicates, StageNum, Options, Code),
    format(string(FileName), "go_ffi_stage_~w.go", [StageNum]).

%% Rust FFI Bridge stages - Node.js accessing Python via Rust FFI
compile_stage(rust_ffi_node, Predicates, StageNum, _OutputDir, Options, file(FileName, Code), stage(StageNum, rust_ffi_node, FileName, Predicates)) :-
    compile_stage_rust_ffi_node(Predicates, StageNum, Options, Code),
    format(string(FileName), "node_ffi_stage_~w.js", [StageNum]).

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
%% RUST FFI BRIDGE STAGE COMPILATION
%% ============================================
%
% These stages use the Rust FFI bridge to access Python via RPyC.
% The bridge is compiled once, and each stage calls it via CGO (Go) or node-ffi (Node.js).

%% compile_stage_rust_ffi_go(+Predicates, +StageNum, +Options, -Code)
%  Compile a Go stage that uses Rust FFI bridge to access Python via RPyC.
%
compile_stage_rust_ffi_go(Predicates, StageNum, Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    maplist(extract_pred_name, Predicates, PredNames),
    atomic_list_concat(PredNames, ' -> ', PredNamesStr),

    % Generate function calls to RPyC via FFI
    generate_rust_ffi_go_calls(Predicates, RpycCalls),

    format(string(Code),
'package main

// Stage ~w: ~w
// Go stage using Rust FFI bridge to access Python via RPyC
//
// Prerequisites:
//   1. Build the Rust FFI bridge: cd rust_ffi_bridge && cargo build --release
//   2. Copy librpyc_bridge.so and rpyc_bridge.h to this directory
//   3. Start RPyC server: python examples/rpyc-integration/rpyc_server.py

/*
#cgo LDFLAGS: -L${SRCDIR} -lrpyc_bridge -lpython3.11
#include "rpyc_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
    "unsafe"
)

// Record represents a data record flowing through the pipeline
type Record map[string]interface{}

// RPyCBridge wraps the Rust FFI bridge
type RPyCBridge struct{}

func NewRPyCBridge() *RPyCBridge {
    C.rpyc_init()
    hostC := C.CString("~w")
    defer C.free(unsafe.Pointer(hostC))
    if C.rpyc_connect(hostC, C.int(~w)) != 0 {
        fmt.Fprintln(os.Stderr, "Failed to connect to RPyC server")
        os.Exit(1)
    }
    return &RPyCBridge{}
}

func (b *RPyCBridge) Close() {
    C.rpyc_disconnect()
}

func (b *RPyCBridge) Call(module, function string, args interface{}) (interface{}, error) {
    argsJSON, err := json.Marshal([]interface{}{args})
    if err != nil {
        return nil, err
    }

    moduleC := C.CString(module)
    funcC := C.CString(function)
    argsC := C.CString(string(argsJSON))
    defer C.free(unsafe.Pointer(moduleC))
    defer C.free(unsafe.Pointer(funcC))
    defer C.free(unsafe.Pointer(argsC))

    resultC := C.rpyc_call(moduleC, funcC, argsC)
    if resultC == nil {
        return nil, fmt.Errorf("rpyc_call failed")
    }
    defer C.rpyc_free_string(resultC)

    var result interface{}
    if err := json.Unmarshal([]byte(C.GoString(resultC)), &result); err != nil {
        return C.GoString(resultC), nil
    }
    return result, nil
}

~w
// processRecords chains all predicates via RPyC
func processRecords(bridge *RPyCBridge, records []Record) []Record {
    var results []Record
    for _, record := range records {
        // Pass record through each predicate via RPyC
        current := record
~w        results = append(results, current)
    }
    return results
}

func main() {
    // Initialize bridge
    bridge := NewRPyCBridge()
    defer bridge.Close()

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
    results := processRecords(bridge, records)

    // Write JSONL to stdout
    for _, result := range results {
        jsonBytes, _ := json.Marshal(result)
        fmt.Println(string(jsonBytes))
    }
}
', [StageNum, PredNamesStr, Host, Port, RpycCalls, ""]).

%% generate_rust_ffi_go_calls(+Predicates, -Code)
%  Generate Go function wrappers that call Python via Rust FFI bridge.
generate_rust_ffi_go_calls([], "").
generate_rust_ffi_go_calls([Pred|Rest], Code) :-
    extract_pred_name(Pred, Name),
    extract_pred_arity(Pred, _Arity),
    % Extract module and function from predicate name (e.g., numpy_mean -> numpy, mean)
    atom_string(NameAtom, Name),
    (   atomic_list_concat([ModuleAtom, FuncAtom], '_', NameAtom)
    ->  atom_string(ModuleAtom, Module), atom_string(FuncAtom, Func)
    ;   Module = "custom", Func = Name
    ),
    format(string(FuncCode),
'// ~w calls ~w.~w via RPyC
func ~w(bridge *RPyCBridge, record Record) Record {
    result, err := bridge.Call("~w", "~w", record)
    if err != nil {
        fmt.Fprintln(os.Stderr, "Error in ~w:", err)
        return record
    }
    if resultMap, ok := result.(map[string]interface{}); ok {
        return resultMap
    }
    record["~w_result"] = result
    return record
}

', [Name, Module, Func, Name, Module, Func, Name, Name]),
    generate_rust_ffi_go_calls(Rest, RestCode),
    string_concat(FuncCode, RestCode, Code).

%% compile_stage_rust_ffi_node(+Predicates, +StageNum, +Options, -Code)
%  Compile a Node.js stage that uses Rust FFI bridge to access Python via RPyC.
%
compile_stage_rust_ffi_node(Predicates, StageNum, Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    maplist(extract_pred_name, Predicates, PredNames),
    atomic_list_concat(PredNames, ' -> ', PredNamesStr),

    % Generate function calls to RPyC via FFI
    generate_rust_ffi_node_calls(Predicates, RpycCalls),

    format(string(Code),
'#!/usr/bin/env node
/**
 * Stage ~w: ~w
 * Node.js stage using Rust FFI bridge to access Python via RPyC
 *
 * Prerequisites:
 *   1. Build the Rust FFI bridge: cd rust_ffi_bridge && cargo build --release
 *   2. npm install ffi-napi ref-napi
 *   3. Start RPyC server: python examples/rpyc-integration/rpyc_server.py
 */

const readline = require(''readline'');
const ffi = require(''ffi-napi'');
const ref = require(''ref-napi'');

const charPtr = ref.refType(ref.types.char);

// Load the Rust FFI bridge library
const lib = ffi.Library(''./librpyc_bridge'', {
    ''rpyc_init'': [''int'', []],
    ''rpyc_connect'': [''int'', [''string'', ''int'']],
    ''rpyc_disconnect'': [''void'', []],
    ''rpyc_call'': [charPtr, [''string'', ''string'', ''string'']],
    ''rpyc_free_string'': [''void'', [charPtr]],
});

// Initialize bridge
lib.rpyc_init();
if (lib.rpyc_connect(''~w'', ~w) !== 0) {
    console.error(''Failed to connect to RPyC server'');
    process.exit(1);
}

function rpycCall(module, func, args) {
    const argsJson = JSON.stringify([args]);
    const resultPtr = lib.rpyc_call(module, func, argsJson);
    if (resultPtr.isNull()) {
        throw new Error(`rpyc_call failed for ${module}.${func}`);
    }
    const resultStr = resultPtr.readCString();
    lib.rpyc_free_string(resultPtr);
    try {
        return JSON.parse(resultStr);
    } catch {
        return resultStr;
    }
}

~w
async function processRecords(records) {
    const results = [];
    for (const record of records) {
        let current = record;
        // Pass through each predicate
~w        results.push(current);
    }
    return results;
}

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: false
    });

    const records = [];
    for await (const line of rl) {
        if (line.trim()) {
            try {
                records.push(JSON.parse(line));
            } catch {
                // Skip invalid JSON
            }
        }
    }

    const results = await processRecords(records);

    for (const result of results) {
        console.log(JSON.stringify(result));
    }

    lib.rpyc_disconnect();
}

main().catch(err => {
    console.error(err);
    lib.rpyc_disconnect();
    process.exit(1);
});
', [StageNum, PredNamesStr, Host, Port, RpycCalls, ""]).

%% generate_rust_ffi_node_calls(+Predicates, -Code)
%  Generate Node.js function wrappers that call Python via Rust FFI bridge.
generate_rust_ffi_node_calls([], "").
generate_rust_ffi_node_calls([Pred|Rest], Code) :-
    extract_pred_name(Pred, Name),
    atom_string(NameAtom, Name),
    (   atomic_list_concat([ModuleAtom, FuncAtom], '_', NameAtom)
    ->  atom_string(ModuleAtom, Module), atom_string(FuncAtom, Func)
    ;   Module = "custom", Func = Name
    ),
    format(string(FuncCode),
'// ~w calls ~w.~w via RPyC
function ~w(record) {
    try {
        const result = rpycCall(''~w'', ''~w'', record);
        if (typeof result === ''object'' && result !== null) {
            return result;
        }
        return { ...record, ~w_result: result };
    } catch (err) {
        console.error(''Error in ~w:'', err);
        return record;
    }
}

', [Name, Module, Func, Name, Module, Func, Name, Name]),
    generate_rust_ffi_node_calls(Rest, RestCode),
    string_concat(FuncCode, RestCode, Code).

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
%  Generate commands to build Go and Rust FFI stages.
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
generate_build_commands([stage(_, rust_ffi_go, FileName, _)|Rest], Commands) :-
    % Rust FFI Go stage: build Rust bridge first, then Go with CGO
    atom_string(FileNameAtom, FileName),
    atom_concat(BinaryName, '.go', FileNameAtom),
    atom_string(BinaryName, BinaryNameStr),
    format(string(BuildCmd),
'# Build Rust FFI bridge (if not already built)
if [ ! -f librpyc_bridge.so ]; then
    echo "Building Rust FFI bridge..."
    cd rust_ffi_bridge && cargo build --release && cd ..
    cp rust_ffi_bridge/target/release/librpyc_bridge.so .
fi
CGO_ENABLED=1 go build -o "~w" "~w"
', [BinaryNameStr, FileName]),
    generate_build_commands(Rest, RestCommands),
    string_concat(BuildCmd, RestCommands, Commands).
generate_build_commands([stage(_, rust_ffi_node, _, _)|Rest], Commands) :-
    % Node.js FFI: just ensure Rust bridge is built
    format(string(BuildCmd),
'# Build Rust FFI bridge for Node.js (if not already built)
if [ ! -f librpyc_bridge.so ]; then
    echo "Building Rust FFI bridge..."
    cd rust_ffi_bridge && cargo build --release && cd ..
    cp rust_ffi_bridge/target/release/librpyc_bridge.so .
fi
', []),
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
runtime_to_command(rust_ffi_go, FileName, Cmd) :-
    % Rust FFI Go binary
    atom_string(FileNameAtom, FileName),
    atom_concat(BinaryName, '.go', FileNameAtom),
    atom_string(BinaryName, BinaryNameStr),
    format(string(Cmd), 'LD_LIBRARY_PATH="$SCRIPT_DIR:$LD_LIBRARY_PATH" "./$SCRIPT_DIR/~w"', [BinaryNameStr]).
runtime_to_command(rust_ffi_node, FileName, Cmd) :-
    format(string(Cmd), 'LD_LIBRARY_PATH="$SCRIPT_DIR:$LD_LIBRARY_PATH" node "$SCRIPT_DIR/~w"', [FileName]).
runtime_to_command(python, FileName, Cmd) :-
    format(string(Cmd), 'python3 "$SCRIPT_DIR/~w"', [FileName]).

%% generate_cleanup_commands(+StageInfo, -Commands)
%  Generate cleanup commands for Go and Rust FFI binaries.
%
generate_cleanup_commands([], "").
generate_cleanup_commands([stage(_, go, FileName, _)|Rest], Commands) :-
    atom_string(FileNameAtom, FileName),
    atom_concat(BinaryName, '.go', FileNameAtom),
    atom_string(BinaryName, BinaryNameStr),
    format(string(CleanCmd), 'rm -f "~w"~n', [BinaryNameStr]),
    generate_cleanup_commands(Rest, RestCommands),
    string_concat(CleanCmd, RestCommands, Commands).
generate_cleanup_commands([stage(_, rust_ffi_go, FileName, _)|Rest], Commands) :-
    atom_string(FileNameAtom, FileName),
    atom_concat(BinaryName, '.go', FileNameAtom),
    atom_string(BinaryName, BinaryNameStr),
    format(string(CleanCmd), 'rm -f "~w"~n', [BinaryNameStr]),
    generate_cleanup_commands(Rest, RestCommands),
    string_concat(CleanCmd, RestCommands, Commands).
generate_cleanup_commands([stage(_, rust_ffi_node, _, _)|Rest], Commands) :-
    % Node.js doesn't have binaries to clean
    generate_cleanup_commands(Rest, Commands).
generate_cleanup_commands([stage(_, python, _, _)|Rest], Commands) :-
    generate_cleanup_commands(Rest, Commands).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% extract_pred_name(+Pred, -NameStr)
%  Extract the predicate name from various formats:
%    - name/arity              -> "name"
%    - runtime:name/arity      -> "name"
%    - rust_ffi:lang:name/arity -> "name"
%
extract_pred_name(rust_ffi:_Lang:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_pred_name(_Runtime:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).

extract_pred_arity(rust_ffi:_Lang:_Name/Arity, Arity) :- !.
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

    % Test 11: Rust FFI runtime detection
    format('[Test 11] Rust FFI runtime detection~n', []),
    predicate_runtime(rust_ffi:go:numpy_mean/1, R11a),
    predicate_runtime(rust_ffi:node:sklearn_predict/1, R11b),
    (   R11a = rust_ffi_go, R11b = rust_ffi_node
    ->  format('  [PASS] rust_ffi:go -> ~w, rust_ffi:node -> ~w~n', [R11a, R11b])
    ;   format('  [FAIL] Rust FFI detection: ~w, ~w~n', [R11a, R11b])
    ),

    % Test 12: Rust FFI Go stage compilation
    format('[Test 12] Rust FFI Go stage compilation~n', []),
    compile_stage_rust_ffi_go([rust_ffi:go:numpy_mean/1], 1, [], RustFfiGoCode),
    (   sub_string(RustFfiGoCode, _, _, _, "package main"),
        sub_string(RustFfiGoCode, _, _, _, "rpyc_bridge.h"),
        sub_string(RustFfiGoCode, _, _, _, "C.rpyc_connect"),
        sub_string(RustFfiGoCode, _, _, _, "C.rpyc_call")
    ->  format('  [PASS] Rust FFI Go stage compiles correctly~n', [])
    ;   format('  [FAIL] Rust FFI Go stage compilation failed~n', [])
    ),

    % Test 13: Rust FFI Node stage compilation
    format('[Test 13] Rust FFI Node stage compilation~n', []),
    compile_stage_rust_ffi_node([rust_ffi:node:sklearn_predict/1], 1, [], RustFfiNodeCode),
    (   sub_string(RustFfiNodeCode, _, _, _, "ffi-napi"),
        sub_string(RustFfiNodeCode, _, _, _, "librpyc_bridge"),
        sub_string(RustFfiNodeCode, _, _, _, "rpyc_call")
    ->  format('  [PASS] Rust FFI Node stage compiles correctly~n', [])
    ;   format('  [FAIL] Rust FFI Node stage compilation failed~n', [])
    ),

    % Test 14: Cross-runtime with Rust FFI
    format('[Test 14] Cross-runtime with Rust FFI~n', []),
    compile_cross_runtime_pipeline([
        rust_ffi:go:numpy_mean/1,
        python:process/1
    ], [pipeline_name(rust_ffi_test)], RustFfiFiles),
    length(RustFfiFiles, NumRustFfiFiles),
    (   NumRustFfiFiles >= 2,
        member(file("rust_ffi_test.sh", _), RustFfiFiles),
        member(file("go_ffi_stage_1.go", _), RustFfiFiles)
    ->  format('  [PASS] Generated ~w files for Rust FFI pipeline~n', [NumRustFfiFiles])
    ;   format('  [FAIL] Rust FFI pipeline: ~w~n', [RustFfiFiles])
    ),

    format('~n=== All Cross-Runtime Pipeline Tests Passed ===~n', []).
