:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% powershell_compiler.pl - Compile Prolog predicates to PowerShell scripts
% Strategy: Generate bash code via existing compilers, wrap in PowerShell compatibility layer
% This allows reuse of all bash templates while providing PowerShell target support
%
% Future: Phase 2 will add native PowerShell code generation with object pipeline support

:- module(powershell_compiler, [
    compile_to_powershell/3,      % compile_to_powershell(+Predicate, +Options, -PowerShellCode)
    compile_to_powershell/2,      % compile_to_powershell(+Predicate, -PowerShellCode)
    powershell_wrapper/3,         % powershell_wrapper(+BashCode, +Options, -PowerShellCode)
    test_powershell_compiler/0    % Run tests
]).

:- use_module(library(lists)).
:- use_module('stream_compiler').
:- use_module('recursive_compiler').
% Load source modules without importing predicates (we call them with module qualification)
:- use_module('../sources/csv_source', []).
:- use_module('../sources/json_source', []).
:- use_module('../sources/http_source', []).

%% compile_to_powershell(+Predicate, -PowerShellCode)
%  Simplified interface with default options
compile_to_powershell(Predicate, PowerShellCode) :-
    compile_to_powershell(Predicate, [], PowerShellCode).

%% compile_to_powershell(+Predicate, +Options, -PowerShellCode)
%  Main compilation entry point
%
%  Predicate: Pred/Arity indicator (e.g., grandparent/2)
%  Options:
%    - compiler(stream|recursive) - which bash compiler to use (default: stream)
%    - wrapper_style(inline|tempfile) - how to invoke bash (default: inline)
%    - compat_check(true|false) - add compatibility layer check (default: true)
%    - output_file(Path) - write to file instead of returning code
%    - script_name(Name) - name for generated script (default: derived from predicate)
%    - powershell_mode(baas|pure|auto) - pure PowerShell or bash-as-a-service (default: auto)
%    ... other options passed to bash compiler
%
compile_to_powershell(Predicate, Options, PowerShellCode) :-
    Predicate = PredAtom/Arity,
    format('[PowerShell Compiler] Compiling ~w/~w to PowerShell~n', [PredAtom, Arity]),

    % Determine PowerShell compilation mode
    option(powershell_mode(Mode), Options, auto),
    format('[PowerShell Compiler] PowerShell mode: ~w~n', [Mode]),

    % Check if pure mode and predicate supports it
    (   (Mode = pure ; Mode = auto),
        supports_pure_powershell(Predicate, Options)
    ->  % Use pure PowerShell template
        format('[PowerShell Compiler] Using pure PowerShell templates~n', []),
        compile_to_pure_powershell(Predicate, Options, PowerShellCode)
    ;   % Fall back to bash-as-a-service
        format('[PowerShell Compiler] Using bash-as-a-service (BaaS) mode~n', []),
        compile_to_baas_powershell(Predicate, Options, PowerShellCode)
    ),

    % Optionally write to file
    (   option(output_file(OutputFile), Options)
    ->  write_powershell_file(OutputFile, PowerShellCode),
        format('[PowerShell Compiler] Written to: ~w~n', [OutputFile])
    ;   true
    ).

%% supports_pure_powershell(+Predicate, +Options)
%  Check if predicate can be compiled to pure PowerShell
supports_pure_powershell(_Predicate, Options) :-
    % Check if it's a dynamic source with pure PowerShell support
    (   member(source_type(csv), Options) -> true
    ;   member(source_type(json), Options) -> true
    ;   member(source_type(http), Options) -> true
    ;   fail  % Other source types don't support pure mode yet
    ).

%% compile_to_pure_powershell(+Predicate, +Options, -PowerShellCode)
%  Compile using pure PowerShell templates (no bash dependency)
compile_to_pure_powershell(Predicate, Options, PowerShellCode) :-
    % Add template_suffix to request _powershell_pure templates
    append(Options, [template_suffix('_powershell_pure')], PureOptions),

    % Dispatch to appropriate source compiler based on source_type
    (   member(source_type(csv), PureOptions)
    ->  csv_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   member(source_type(json), PureOptions)
    ->  json_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   member(source_type(http), PureOptions)
    ->  http_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   format('[PowerShell Compiler] Error: No pure PowerShell support for this source type~n', []),
        fail
    ).

%% compile_to_baas_powershell(+Predicate, +Options, -PowerShellCode)
%  Compile using bash-as-a-service approach (original implementation)
compile_to_baas_powershell(Predicate, Options, PowerShellCode) :-
    % Determine which bash compiler to use
    option(compiler(Compiler), Options, stream),
    format('[PowerShell Compiler] Using ~w compiler~n', [Compiler]),

    % Generate bash code using selected compiler
    compile_to_bash(Compiler, Predicate, Options, BashCode),

    % Wrap bash code in PowerShell compatibility layer invocation
    powershell_wrapper(BashCode, Options, PowerShellCode).

%% compile_to_bash(+Compiler, +Predicate, +Options, -BashCode)
%  Internal: dispatch to appropriate bash compiler
compile_to_bash(stream, Predicate, Options, BashCode) :-
    stream_compiler:compile_predicate(Predicate, Options, BashCode).

compile_to_bash(recursive, Predicate, Options, BashCode) :-
    recursive_compiler:compile_predicate(Predicate, Options, BashCode).

%% powershell_wrapper(+BashCode, +Options, -PowerShellCode)
%  Wraps bash code in PowerShell compatibility layer invocation
%
%  Options:
%    - wrapper_style(inline|tempfile) - default: inline
%    - compat_check(true|false) - default: true
%    - script_name(Name) - for comments
%    - predicate(Pred/Arity) - for comments
%
powershell_wrapper(BashCode, Options, PowerShellCode) :-
    option(wrapper_style(Style), Options, inline),
    option(compat_check(CompatCheck), Options, true),
    option(script_name(ScriptName), Options, 'generated_script'),

    format('[PowerShell Compiler] Wrapper style: ~w~n', [Style]),

    % Generate appropriate wrapper
    generate_powershell_wrapper(Style, BashCode, ScriptName, CompatCheck, PowerShellCode).

%% generate_powershell_wrapper(+Style, +BashCode, +ScriptName, +CompatCheck, -PowerShellCode)
%  Generate PowerShell wrapper based on style

% Inline style: Bash code as heredoc, executed via uw-bash
generate_powershell_wrapper(inline, BashCode, ScriptName, CompatCheck, PowerShellCode) :-
    % Get timestamp
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Build compatibility check code
    (   CompatCheck = true
    ->  CompatCheckCode = '# Ensure compatibility layer is available\nif (-not (Get-Command uw-bash -ErrorAction SilentlyContinue)) {\n    Write-Error "UnifyWeaver PowerShell compatibility layer not loaded"\n    Write-Host "Please run: . .\\scripts\\init_unify_compat.ps1" -ForegroundColor Yellow\n    exit 1\n}\n\n'
    ;   CompatCheckCode = ''
    ),

    % Construct PowerShell script
    atomic_list_concat([
        '# Generated PowerShell Script\n',
        '# Script: ', ScriptName, '.ps1\n',
        '# Generated by UnifyWeaver PowerShell Compiler\n',
        '# Generated: ', DateStr, '\n',
        '#\n',
        '# This script wraps a bash implementation via the PowerShell compatibility layer.\n',
        '# The bash code is executed using uw-bash from UnifyWeaver compatibility layer.\n',
        '\n',
        CompatCheckCode,
        '# Embedded bash implementation\n',
        '$bashScript = @''\n',
        BashCode, '\n',
        '''@\n',
        '\n',
        '# Execute via compatibility layer\n',
        'if ($Input) {\n',
        '    $Input | uw-bash -c $bashScript\n',
        '} else {\n',
        '    uw-bash -c $bashScript\n',
        '}\n'
    ], PowerShellCode).

% Tempfile style: Write bash to temp file, execute, cleanup
generate_powershell_wrapper(tempfile, BashCode, ScriptName, CompatCheck, PowerShellCode) :-
    % Get timestamp
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Build compatibility check code
    (   CompatCheck = true
    ->  CompatCheckCode = '# Ensure compatibility layer is available\nif (-not (Get-Command uw-bash -ErrorAction SilentlyContinue)) {\n    Write-Error "UnifyWeaver PowerShell compatibility layer not loaded"\n    Write-Host "Please run: . .\\scripts\\init_unify_compat.ps1" -ForegroundColor Yellow\n    exit 1\n}\n\n'
    ;   CompatCheckCode = ''
    ),

    % Construct PowerShell script with tempfile approach
    atomic_list_concat([
        '# Generated PowerShell Script\n',
        '# Script: ', ScriptName, '.ps1\n',
        '# Generated by UnifyWeaver PowerShell Compiler\n',
        '# Generated: ', DateStr, '\n',
        '#\n',
        '# This script creates a temporary bash file and executes it.\n',
        '\n',
        CompatCheckCode,
        '# Create temporary bash script\n',
        '$tempFile = [System.IO.Path]::GetTempFileName() + ".sh"\n',
        '\n',
        '# Embedded bash implementation\n',
        '$bashScript = @''\n',
        BashCode, '\n',
        '''@\n',
        '\n',
        '# Write to temp file and execute\n',
        'try {\n',
        '    Set-Content -Path $tempFile -Value $bashScript -Encoding UTF8\n',
        '    \n',
        '    if ($Input) {\n',
        '        $Input | uw-bash $tempFile\n',
        '    } else {\n',
        '        uw-bash $tempFile\n',
        '    }\n',
        '} finally {\n',
        '    # Cleanup temp file\n',
        '    if (Test-Path $tempFile) {\n',
        '        Remove-Item $tempFile -ErrorAction SilentlyContinue\n',
        '    }\n',
        '}\n'
    ], PowerShellCode).

%% write_powershell_file(+FilePath, +PowerShellCode)
%  Write PowerShell code to file
write_powershell_file(FilePath, PowerShellCode) :-
    open(FilePath, write, Stream, [encoding(utf8)]),
    write(Stream, PowerShellCode),
    close(Stream).

%% option(+Key, +Options, -Value)
%% option(+Key, +Options, -Value, +Default)
%  Helper to extract option from list
option(Key, Options, Default) :-
    (   member(Key, Options)
    ->  true
    ;   Key =.. [Functor, Default],
        \+ member(Functor=_, Options)
    ).

%% ============================================================================
%% TESTS
%% ============================================================================

test_powershell_compiler :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  PowerShell Compiler Tests            ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Test 1: Basic inline wrapper
    format('~n[Test 1] Inline wrapper generation~n', []),
    test_inline_wrapper,

    % Test 2: Tempfile wrapper
    format('~n[Test 2] Tempfile wrapper generation~n', []),
    test_tempfile_wrapper,

    % Test 3: Compilation with options
    format('~n[Test 3] Compilation with various options~n', []),
    test_compilation_options,

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All PowerShell Compiler Tests Passed ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).

test_inline_wrapper :-
    BashCode = '#!/bin/bash\necho "Hello from bash"',
    powershell_wrapper(BashCode, [wrapper_style(inline), script_name(test_inline)], PSCode),

    % Verify it contains key components
    sub_string(PSCode, _, _, _, '$bashScript = @'''),
    sub_string(PSCode, _, _, _, 'uw-bash -c $bashScript'),
    sub_string(PSCode, _, _, _, 'if (-not (Get-Command uw-bash'),

    format('[✓] Inline wrapper contains expected components~n', []).

test_tempfile_wrapper :-
    BashCode = '#!/bin/bash\necho "Hello from bash"',
    powershell_wrapper(BashCode, [wrapper_style(tempfile), script_name(test_tempfile)], PSCode),

    % Verify it contains key components
    sub_string(PSCode, _, _, _, '$tempFile = [System.IO.Path]::GetTempFileName'),
    sub_string(PSCode, _, _, _, 'Set-Content -Path $tempFile'),
    sub_string(PSCode, _, _, _, 'Remove-Item $tempFile'),

    format('[✓] Tempfile wrapper contains expected components~n', []).

test_compilation_options :-
    % Test without compat check
    BashCode = '#!/bin/bash\necho "Test"',
    powershell_wrapper(BashCode, [wrapper_style(inline), compat_check(false)], PSCode1),
    \+ sub_string(PSCode1, _, _, _, 'if (-not (Get-Command uw-bash'),
    format('[✓] compat_check(false) omits compatibility check~n', []),

    % Test with compat check (default)
    powershell_wrapper(BashCode, [wrapper_style(inline)], PSCode2),
    sub_string(PSCode2, _, _, _, 'if (-not (Get-Command uw-bash'),
    format('[✓] Default includes compatibility check~n', []).
