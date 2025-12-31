/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Native Glue - Binary orchestration for Go and Rust targets
 *
 * This module generates:
 * - Pipe-compatible main() wrappers for Go and Rust
 * - Binary compilation management
 * - Cross-compilation support
 * - Pipeline orchestration for native binaries
 */

:- module(native_glue, [
    % Binary management
    register_binary/4,              % register_binary(+Pred/Arity, +Target, +Path, +Options)
    compiled_binary/3,              % compiled_binary(?Pred/Arity, ?Target, ?Path)
    compile_if_needed/4,            % compile_if_needed(+Pred/Arity, +Target, +SourcePath, -BinaryPath)

    % Toolchain detection
    detect_go/1,                    % detect_go(-Version)
    detect_rust/1,                  % detect_rust(-Version)
    detect_cargo/1,                 % detect_cargo(-Version)

    % Go code generation
    generate_go_pipe_main/3,        % generate_go_pipe_main(+Logic, +Options, -Code)
    generate_go_wrapper/4,          % generate_go_wrapper(+FuncName, +Schema, +Options, -Code)
    generate_go_build_script/3,     % generate_go_build_script(+SourcePath, +Options, -Script)

    % Rust code generation
    generate_rust_pipe_main/3,      % generate_rust_pipe_main(+Logic, +Options, -Code)
    generate_rust_wrapper/4,        % generate_rust_wrapper(+FuncName, +Schema, +Options, -Code)
    generate_rust_build_script/3,   % generate_rust_build_script(+SourcePath, +Options, -Script)

    % Cross-compilation
    cross_compile_targets/1,        % cross_compile_targets(-Targets)
    generate_cross_compile/4,       % generate_cross_compile(+Target, +Source, +Platforms, -Script)

    % Pipeline orchestration
    generate_native_pipeline/3      % generate_native_pipeline(+Steps, +Options, -Script)
]).

:- use_module(library(lists)).

:- discontiguous generate_cross_compile/4.

%% ============================================
%% Binary Management
%% ============================================

:- dynamic compiled_binary_db/3.

%% register_binary(+Pred/Arity, +Target, +Path, +Options)
%  Register a compiled binary for a predicate.
%
register_binary(Pred/Arity, Target, Path, Options) :-
    retractall(compiled_binary_db(Pred/Arity, Target, _)),
    assertz(compiled_binary_db(Pred/Arity, Target, binary_info(Path, Options))).

%% compiled_binary(?Pred/Arity, ?Target, ?Path)
%  Query registered binaries.
%
compiled_binary(Pred/Arity, Target, Path) :-
    compiled_binary_db(Pred/Arity, Target, binary_info(Path, _)).

%% compile_if_needed(+Pred/Arity, +Target, +SourcePath, -BinaryPath)
%  Compile source to binary if not already compiled or source is newer.
%
compile_if_needed(Pred/Arity, Target, SourcePath, BinaryPath) :-
    compiled_binary(Pred/Arity, Target, BinaryPath),
    exists_file(BinaryPath),
    exists_file(SourcePath),
    time_file(BinaryPath, BinTime),
    time_file(SourcePath, SrcTime),
    BinTime >= SrcTime,
    !.

compile_if_needed(Pred/Arity, Target, SourcePath, BinaryPath) :-
    % Determine binary path
    file_base_name(SourcePath, BaseName),
    file_name_extension(Base, _, BaseName),
    (   Target == go
    ->  BinaryPath = Base
    ;   Target == rust
    ->  atom_concat('target/release/', Base, BinaryPath)
    ),
    % Compile
    compile_native(Target, SourcePath, BinaryPath),
    register_binary(Pred/Arity, Target, BinaryPath, []).

compile_native(go, SourcePath, BinaryPath) :-
    format(atom(Cmd), 'go build -o "~w" "~w"', [BinaryPath, SourcePath]),
    shell(Cmd, 0).

compile_native(rust, SourcePath, _BinaryPath) :-
    file_directory_name(SourcePath, Dir),
    format(atom(Cmd), 'cd "~w" && cargo build --release', [Dir]),
    shell(Cmd, 0).

%% ============================================
%% Toolchain Detection
%% ============================================

%% detect_go(-Version)
%  Detect Go installation and version.
%
detect_go(Version) :-
    catch(
        (process_create(path(go), [version], [stdout(pipe(S))]),
         read_line_to_string(S, VersionStr),
         close(S),
         % Parse "go version go1.21.0 linux/amd64"
         sub_string(VersionStr, Before, _, _, "go1."),
         Before > 0,
         sub_string(VersionStr, Before, _, 0, Rest),
         sub_string(Rest, 0, End, _, Version),
         sub_string(Rest, End, 1, _, " ")),
        _, fail).
detect_go(none).

%% detect_rust(-Version)
%  Detect Rust installation and version.
%
detect_rust(Version) :-
    catch(
        (process_create(path(rustc), ['--version'], [stdout(pipe(S))]),
         read_line_to_string(S, VersionStr),
         close(S),
         % Parse "rustc 1.75.0 (..."
         sub_string(VersionStr, 6, Len, _, Version),
         sub_string(VersionStr, _, 1, After, " "),
         After > 0,
         Len > 0),
        _, fail).
detect_rust(none).

%% detect_cargo(-Version)
%  Detect Cargo installation.
%
detect_cargo(Version) :-
    catch(
        (process_create(path(cargo), ['--version'], [stdout(pipe(S))]),
         read_line_to_string(S, VersionStr),
         close(S),
         % Parse "cargo 1.75.0 (..."
         sub_string(VersionStr, 6, _, _, Rest),
         sub_string(Rest, 0, End, _, Version),
         sub_string(Rest, End, 1, _, " ")),
        _, fail).
detect_cargo(none).

%% ============================================
%% Go Code Generation
%% ============================================

%% generate_go_pipe_main(+Logic, +Options, -Code)
%  Generate a Go main() that reads TSV from stdin, processes, writes to stdout.
%
%  Logic: Go code that processes a record ([]string) and returns []string
%  Options:
%    - format(tsv|json) : Input/output format
%    - fields(List) : Field names for JSON
%    - parallel(N) : Number of worker goroutines
%
generate_go_pipe_main(Logic, Options, Code) :-
    option_or_default(format(Format), Options, tsv),
    option_or_default(parallel(Parallel), Options, 1),

    (Format == json ->
        generate_go_json_main(Logic, Options, Code)
    ; Parallel > 1 ->
        generate_go_parallel_main(Logic, Parallel, Code)
    ;
        generate_go_tsv_main(Logic, Code)
    ).

generate_go_tsv_main(Logic, Code) :-
    format(atom(Code), '
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
)

func process(fields []string) []string {
~w
}

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    // Increase buffer size for large lines
    buf := make([]byte, 0, 1024*1024)
    scanner.Buffer(buf, 10*1024*1024)

    for scanner.Scan() {
        line := scanner.Text()
        fields := strings.Split(line, "\\t")
        result := process(fields)
        if result != nil {
            fmt.Println(strings.Join(result, "\\t"))
        }
    }

    if err := scanner.Err(); err != nil {
        fmt.Fprintln(os.Stderr, "Error reading input:", err)
        os.Exit(1)
    }
}
', [Logic]).

generate_go_json_main(Logic, Options, Code) :-
    option_or_default(fields(Fields), Options, []),
    fields_to_go_struct(Fields, StructDef),

    format(atom(Code), '
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
)

~w

func process(record Record) *Record {
~w
}

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    buf := make([]byte, 0, 1024*1024)
    scanner.Buffer(buf, 10*1024*1024)

    for scanner.Scan() {
        var record Record
        if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
            fmt.Fprintln(os.Stderr, "JSON parse error:", err)
            continue
        }

        result := process(record)
        if result != nil {
            output, _ := json.Marshal(result)
            fmt.Println(string(output))
        }
    }
}
', [StructDef, Logic]).

generate_go_parallel_main(Logic, Workers, Code) :-
    format(atom(Code), '
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
    "sync"
)

func process(fields []string) []string {
~w
}

func main() {
    // Input channel
    lines := make(chan string, 1000)
    results := make(chan string, 1000)

    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < ~w; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for line := range lines {
                fields := strings.Split(line, "\\t")
                result := process(fields)
                if result != nil {
                    results <- strings.Join(result, "\\t")
                }
            }
        }()
    }

    // Output goroutine
    var outputWg sync.WaitGroup
    outputWg.Add(1)
    go func() {
        defer outputWg.Done()
        for result := range results {
            fmt.Println(result)
        }
    }()

    // Read input
    scanner := bufio.NewScanner(os.Stdin)
    buf := make([]byte, 0, 1024*1024)
    scanner.Buffer(buf, 10*1024*1024)

    for scanner.Scan() {
        lines <- scanner.Text()
    }
    close(lines)

    wg.Wait()
    close(results)
    outputWg.Wait()
}
', [Logic, Workers]).

%% fields_to_go_struct(+Fields, -StructDef)
fields_to_go_struct([], StructDef) :-
    StructDef = 'type Record map[string]interface{}'.
fields_to_go_struct(Fields, StructDef) :-
    Fields \= [],
    maplist(field_to_go_field, Fields, GoFields),
    atomic_list_concat(GoFields, '\n    ', FieldsStr),
    format(atom(StructDef), 'type Record struct {\n    ~w\n}', [FieldsStr]).

field_to_go_field(Field, GoField) :-
    atom_string(Field, FieldStr),
    string_chars(FieldStr, [First|Rest]),
    char_type(First, alpha),
    upcase_atom(First, Upper),
    atom_chars(UpperFirst, [Upper]),
    atom_chars(RestAtom, Rest),
    atom_concat(UpperFirst, RestAtom, GoName),
    format(atom(GoField), '~w string `json:"~w"`', [GoName, Field]).

%% generate_go_wrapper(+FuncName, +Schema, +Options, -Code)
%  Generate a complete Go file wrapping a function for pipe I/O.
%
generate_go_wrapper(FuncName, Schema, Options, Code) :-
    schema_to_go_logic(FuncName, Schema, Logic),
    generate_go_pipe_main(Logic, Options, Code).

schema_to_go_logic(FuncName, Schema, Logic) :-
    Schema = schema(InputFields, OutputFields),
    length(InputFields, InLen),
    length(OutputFields, OutLen),
    format(atom(Logic), '
    // Input: ~w fields, Output: ~w fields
    if len(fields) < ~w {
        return nil
    }
    result := ~w(fields)
    return result[:~w]
', [InLen, OutLen, InLen, FuncName, OutLen]).

%% generate_go_build_script(+SourcePath, +Options, -Script)
%  Generate a build script for Go source.
%
generate_go_build_script(SourcePath, Options, Script) :-
    file_base_name(SourcePath, BaseName),
    file_name_extension(Base, _, BaseName),
    option_or_default(output_dir(OutDir), Options, '.'),
    option_or_default(optimize(Opt), Options, true),

    (Opt == true ->
        LdFlags = '-ldflags="-s -w"'
    ;
        LdFlags = ''
    ),

    format(atom(Script), '#!/bin/bash
set -euo pipefail

SOURCE="~w"
OUTPUT="~w/~w"

echo "Building Go binary..."
go build ~w -o "$OUTPUT" "$SOURCE"

echo "Built: $OUTPUT"
ls -lh "$OUTPUT"
', [SourcePath, OutDir, Base, LdFlags]).

%% ============================================
%% Rust Code Generation
%% ============================================

%% generate_rust_pipe_main(+Logic, +Options, -Code)
%  Generate a Rust main() that reads TSV from stdin, processes, writes to stdout.
%
generate_rust_pipe_main(Logic, Options, Code) :-
    option_or_default(format(Format), Options, tsv),

    (Format == json ->
        generate_rust_json_main(Logic, Options, Code)
    ;
        generate_rust_tsv_main(Logic, Code)
    ).

generate_rust_tsv_main(Logic, Code) :-
    format(atom(Code), '
use std::io::{self, BufRead, Write};

fn process(fields: &[&str]) -> Option<Vec<String>> {
~w
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        match line {
            Ok(text) => {
                let fields: Vec<&str> = text.split(\'\\t\').collect();
                if let Some(result) = process(&fields) {
                    writeln!(stdout, "{}", result.join("\\t")).unwrap();
                }
            }
            Err(e) => {
                eprintln!("Error reading line: {}", e);
            }
        }
    }
}
', [Logic]).

generate_rust_json_main(Logic, Options, Code) :-
    option_or_default(fields(Fields), Options, []),
    fields_to_rust_struct(Fields, StructDef),

    format(atom(Code), '
use std::io::{self, BufRead, Write};
use serde::{Deserialize, Serialize};
use serde_json;

~w

fn process(record: Record) -> Option<Record> {
~w
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        match line {
            Ok(text) => {
                match serde_json::from_str::<Record>(&text) {
                    Ok(record) => {
                        if let Some(result) = process(record) {
                            let output = serde_json::to_string(&result).unwrap();
                            writeln!(stdout, "{}", output).unwrap();
                        }
                    }
                    Err(e) => {
                        eprintln!("JSON parse error: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading line: {}", e);
            }
        }
    }
}
', [StructDef, Logic]).

%% fields_to_rust_struct(+Fields, -StructDef)
fields_to_rust_struct([], StructDef) :-
    StructDef = 'type Record = serde_json::Value;'.
fields_to_rust_struct(Fields, StructDef) :-
    Fields \= [],
    maplist(field_to_rust_field, Fields, RustFields),
    atomic_list_concat(RustFields, '\n    ', FieldsStr),
    format(atom(StructDef), '#[derive(Debug, Deserialize, Serialize)]
struct Record {
    ~w
}', [FieldsStr]).

field_to_rust_field(Field, RustField) :-
    format(atom(RustField), '~w: Option<String>,', [Field]).

%% generate_rust_wrapper(+FuncName, +Schema, +Options, -Code)
%  Generate a complete Rust file wrapping a function for pipe I/O.
%
generate_rust_wrapper(FuncName, Schema, Options, Code) :-
    schema_to_rust_logic(FuncName, Schema, Logic),
    generate_rust_pipe_main(Logic, Options, Code).

schema_to_rust_logic(FuncName, Schema, Logic) :-
    Schema = schema(InputFields, OutputFields),
    length(InputFields, InLen),
    length(OutputFields, OutLen),
    format(atom(Logic), '
    // Input: ~w fields, Output: ~w fields
    if fields.len() < ~w {
        return None;
    }
    let result = ~w(fields);
    Some(result[..~w].to_vec())
', [InLen, OutLen, InLen, FuncName, OutLen]).

%% generate_rust_build_script(+SourcePath, +Options, -Script)
%  Generate a build script for Rust source.
%
generate_rust_build_script(SourcePath, Options, Script) :-
    file_directory_name(SourcePath, Dir),
    option_or_default(profile(Profile), Options, release),

    (Profile == release ->
        ProfileFlag = '--release'
    ;
        ProfileFlag = ''
    ),

    format(atom(Script), '#!/bin/bash
set -euo pipefail

cd "~w"

echo "Building Rust binary..."
cargo build ~w

echo "Built successfully"
ls -lh target/~w/
', [Dir, ProfileFlag, Profile]).

%% generate_rust_cargo_toml(+Name, +Dependencies, -Toml)
%  Generate Cargo.toml for a Rust project.
%
generate_rust_cargo_toml(Name, Dependencies, Toml) :-
    maplist(dep_to_toml, Dependencies, DepLines),
    atomic_list_concat(DepLines, '\n', DepsStr),
    format(atom(Toml), '[package]
name = "~w"
version = "0.1.0"
edition = "2021"

[dependencies]
~w

[profile.release]
lto = true
codegen-units = 1
', [Name, DepsStr]).

dep_to_toml(Dep, Line) :-
    (Dep = Name-Version ->
        format(atom(Line), '~w = "~w"', [Name, Version])
    ;
        format(atom(Line), '~w = "*"', [Dep])
    ).

%% ============================================
%% Cross-Compilation
%% ============================================

%% cross_compile_targets(-Targets)
%  List supported cross-compilation targets.
%
cross_compile_targets([
    target(linux, amd64, 'linux', 'amd64'),
    target(linux, arm64, 'linux', 'arm64'),
    target(darwin, amd64, 'darwin', 'amd64'),
    target(darwin, arm64, 'darwin', 'arm64'),
    target(windows, amd64, 'windows', 'amd64')
]).

%% generate_cross_compile(+Target, +Source, +Platforms, -Script)
%  Generate cross-compilation script.
%
%  Target: go | rust
%  Platforms: List of os-arch pairs
%
generate_cross_compile(go, Source, Platforms, Script) :-
    file_base_name(Source, BaseName),
    file_name_extension(Base, _, BaseName),
    maplist(go_cross_cmd(Base, Source), Platforms, Commands),
    atomic_list_concat(Commands, '\n', CmdStr),
    format(atom(Script), '#!/bin/bash
set -euo pipefail

SOURCE="~w"
NAME="~w"

~w

echo "Cross-compilation complete!"
ls -lh ${NAME}_*
', [Source, Base, CmdStr]).

go_cross_cmd(_Base, _Source, OS-Arch, Cmd) :-
    (OS == windows -> Ext = '.exe' ; Ext = ''),
    format(atom(Cmd), 'echo "Building for ~w/~w..."
GOOS=~w GOARCH=~w go build -o "${NAME}_~w_~w~w" "$SOURCE"',
           [OS, Arch, OS, Arch, OS, Arch, Ext]).

generate_cross_compile(rust, Source, Platforms, Script) :-
    file_directory_name(Source, Dir),
    maplist(rust_cross_target, Platforms, Targets),
    atomic_list_concat(Targets, ' ', TargetList),
    format(atom(Script), '#!/bin/bash
set -euo pipefail

cd "~w"

# Add targets
for target in ~w; do
    rustup target add $target 2>/dev/null || true
done

# Build for each target
for target in ~w; do
    echo "Building for $target..."
    cargo build --release --target $target
done

echo "Cross-compilation complete!"
ls -lh target/*/release/
', [Dir, TargetList, TargetList]).

rust_cross_target(linux-amd64, 'x86_64-unknown-linux-gnu').
rust_cross_target(linux-arm64, 'aarch64-unknown-linux-gnu').
rust_cross_target(darwin-amd64, 'x86_64-apple-darwin').
rust_cross_target(darwin-arm64, 'aarch64-apple-darwin').
rust_cross_target(windows-amd64, 'x86_64-pc-windows-gnu').

%% ============================================
%% Pipeline Orchestration
%% ============================================

%% generate_native_pipeline(+Steps, +Options, -Script)
%  Generate a shell script that orchestrates native binaries in a pipeline.
%
%  Steps: List of step(Name, Target, BinaryPath, StepOptions)
%         Target = go | rust | awk | python | bash
%
generate_native_pipeline(Steps, Options, Script) :-
    option_or_default(input(InputFile), Options, '-'),
    option_or_default(output(OutputFile), Options, '-'),
    option_or_default(parallel(Parallel), Options, false),

    (InputFile == '-' ->
        InputCmd = 'cat'
    ;
        format(atom(InputCmd), 'cat "~w"', [InputFile])
    ),

    (OutputFile == '-' ->
        OutputRedir = ''
    ;
        format(atom(OutputRedir), ' > "~w"', [OutputFile])
    ),

    (Parallel == true ->
        generate_parallel_pipeline(Steps, PipelineCmd)
    ;
        steps_to_pipeline_cmd(Steps, PipelineCmd)
    ),

    format(atom(Script), '#!/bin/bash
# Generated Native Pipeline
set -euo pipefail

~w \\
    | ~w~w
', [InputCmd, PipelineCmd, OutputRedir]).

steps_to_pipeline_cmd([Step], Cmd) :-
    !,
    step_to_cmd(Step, Cmd).
steps_to_pipeline_cmd([Step|Rest], Cmd) :-
    step_to_cmd(Step, StepCmd),
    steps_to_pipeline_cmd(Rest, RestCmd),
    format(atom(Cmd), '~w \\
    | ~w', [StepCmd, RestCmd]).

step_to_cmd(step(_Name, go, BinaryPath, _Opts), Cmd) :-
    format(atom(Cmd), '"~w"', [BinaryPath]).
step_to_cmd(step(_Name, rust, BinaryPath, _Opts), Cmd) :-
    format(atom(Cmd), '"~w"', [BinaryPath]).
step_to_cmd(step(_Name, awk, ScriptPath, _Opts), Cmd) :-
    format(atom(Cmd), 'awk -f "~w"', [ScriptPath]).
step_to_cmd(step(_Name, python, ScriptPath, _Opts), Cmd) :-
    format(atom(Cmd), 'python3 "~w"', [ScriptPath]).
step_to_cmd(step(_Name, bash, ScriptPath, _Opts), Cmd) :-
    format(atom(Cmd), 'bash "~w"', [ScriptPath]).

generate_parallel_pipeline(Steps, Cmd) :-
    % Use GNU parallel if available
    maplist(step_to_parallel_cmd, Steps, ParallelCmds),
    atomic_list_concat(ParallelCmds, ' | ', Cmd).

step_to_parallel_cmd(step(_Name, Target, Path, Opts), Cmd) :-
    option_or_default(workers(N), Opts, 4),
    step_to_cmd(step(_, Target, Path, []), BaseCmd),
    format(atom(Cmd), 'parallel --pipe -k -j~w ~w', [N, BaseCmd]).

%% ============================================
%% Utility Predicates
%% ============================================

option_or_default(Option, Options, _Default) :-
    member(Option, Options),
    !.
option_or_default(Option, _Options, Default) :-
    Option =.. [_, Default].

exists_file(Path) :-
    catch(
        (open(Path, read, S), close(S)),
        _, fail).

file_base_name(Path, Base) :-
    atom_string(Path, PathStr),
    split_string(PathStr, "/", "", Parts),
    last(Parts, BaseStr),
    atom_string(Base, BaseStr).

file_name_extension(Base, Ext, FullName) :-
    atom_string(FullName, FullStr),
    (   sub_string(FullStr, Before, 1, After, ".")
    ->  sub_string(FullStr, 0, Before, _, BaseStr),
        sub_string(FullStr, _, After, 0, ExtStr),
        atom_string(Base, BaseStr),
        atom_string(Ext, ExtStr)
    ;   Base = FullName,
        Ext = ''
    ).

file_directory_name(Path, Dir) :-
    atom_string(Path, PathStr),
    split_string(PathStr, "/", "", Parts),
    length(Parts, Len),
    Len1 is Len - 1,
    length(DirParts, Len1),
    append(DirParts, [_], Parts),
    atomic_list_concat(DirParts, '/', Dir).
