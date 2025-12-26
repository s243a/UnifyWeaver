% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% powershell_bindings.pl - PowerShell-specific bindings
%
% This module defines bindings for PowerShell target language features.
% Based on:
%   - Book 12, Chapter 3: Cmdlet Generation
%   - Book 12, Chapter 5: Windows Automation
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(powershell_bindings, [
    init_powershell_bindings/0,
    ps_binding/5,               % Convenience: ps_binding(Pred, TargetName, Inputs, Outputs, Options)
    ps_binding_using/2,         % ps_binding_using(Pred, Namespace)
    test_powershell_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_powershell_bindings
%
%  Initialize all PowerShell bindings. Call this before using the compiler.
%
init_powershell_bindings :-
    register_cmdlet_bindings,
    register_automation_bindings,
    register_dotnet_bindings,
    register_csharp_hosting_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

%% ps_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query PowerShell bindings with reduced arity (Target=powershell implied).
%
ps_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(powershell, Pred, TargetName, Inputs, Outputs, Options).

%% ps_binding_using(?Pred, ?Namespace)
%  Get the using namespace required for a PowerShell binding.
ps_binding_using(Pred, Namespace) :-
    ps_binding(Pred, _, _, _, Options),
    member(using(Namespace), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- ps_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined PowerShell bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- ps_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(powershell, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CHAPTER 3: CMDLET GENERATION BINDINGS
% ============================================================================

register_cmdlet_bindings :-
    % -------------------------------------------
    % Core Output Cmdlets
    % -------------------------------------------

    % Write-Output - primary output mechanism
    declare_binding(powershell, write_output/1, 'Write-Output',
        [any], [],
        [effect(io), pattern(object_pipeline), deterministic]),

    % Write-Host - display text (not pipelined)
    declare_binding(powershell, write_host/1, 'Write-Host',
        [string], [],
        [effect(io), pattern(stdout_return), deterministic]),

    % Write-Verbose - diagnostic output with -Verbose
    declare_binding(powershell, write_verbose/1, 'Write-Verbose',
        [string], [],
        [effect(io), deterministic]),

    % Write-Debug - debug output with -Debug
    declare_binding(powershell, write_debug/1, 'Write-Debug',
        [string], [],
        [effect(io), deterministic]),

    % Write-Warning - warning messages
    declare_binding(powershell, write_warning/1, 'Write-Warning',
        [string], [],
        [effect(io), deterministic]),

    % Write-Error - non-terminating errors
    declare_binding(powershell, write_error/1, 'Write-Error',
        [string], [],
        [effect(io), effect(throws), deterministic]),

    % -------------------------------------------
    % Object Creation
    % -------------------------------------------

    % PSCustomObject - create structured output
    declare_binding(powershell, ps_object/2, '[PSCustomObject]@{}',
        [hashtable], [object],
        [pure, deterministic, pattern(cmdlet_output)]),

    % Select-Object - project specific properties
    declare_binding(powershell, select_object/2, 'Select-Object',
        [object, list(property)], [object],
        [pure, deterministic, pattern(pipe_transform)]),

    % -------------------------------------------
    % Pipeline Operations
    % -------------------------------------------

    % ForEach-Object - iterate over pipeline
    declare_binding(powershell, foreach_object/2, 'ForEach-Object',
        [object, scriptblock], [object],
        [nondeterministic, pattern(pipe_transform)]),

    % Where-Object - filter pipeline
    declare_binding(powershell, where_object/2, 'Where-Object',
        [object, scriptblock], [object],
        [pure, nondeterministic, pattern(pipe_transform)]),

    % Sort-Object - sort pipeline output
    declare_binding(powershell, sort_object/2, 'Sort-Object',
        [object, property], [object],
        [pure, nondeterministic, pattern(pipe_transform)]),

    % Group-Object - group pipeline items
    declare_binding(powershell, group_object/2, 'Group-Object',
        [object, property], [group],
        [pure, nondeterministic, pattern(pipe_transform)]),

    % Measure-Object - compute statistics
    declare_binding(powershell, measure_object/2, 'Measure-Object',
        [object, list(property)], [measurement],
        [pure, deterministic, pattern(pipe_transform)]),

    % -------------------------------------------
    % Type Conversions
    % -------------------------------------------

    % String casting
    declare_binding(powershell, to_string/2, '[string]',
        [any], [string],
        [pure, deterministic, total]),

    % Integer casting
    declare_binding(powershell, to_int/2, '[int]',
        [any], [int],
        [pure, deterministic, partial, effect(throws)]),

    % Array casting
    declare_binding(powershell, to_array/2, '@()',
        [any], [array],
        [pure, deterministic, total]),

    % Hashtable creation
    declare_binding(powershell, to_hashtable/2, '@{}',
        [list(pair)], [hashtable],
        [pure, deterministic, total]).

% ============================================================================
% CHAPTER 5: WINDOWS AUTOMATION BINDINGS
% ============================================================================

register_automation_bindings :-
    % -------------------------------------------
    % File System Operations
    % -------------------------------------------

    % Get-ChildItem - list files/directories
    declare_binding(powershell, get_child_item/2, 'Get-ChildItem',
        [path], [list(file_info)],
        [effect(io), nondeterministic, import('Microsoft.PowerShell.Management')]),

    % Test-Path - check if path exists
    declare_binding(powershell, test_path/1, 'Test-Path',
        [path], [],
        [effect(io), deterministic, pattern(cmdlet_output),
         exit_status(true=exists, false=not_exists)]),

    % Get-Content - read file content
    declare_binding(powershell, get_content/2, 'Get-Content',
        [path], [list(string)],
        [effect(io), nondeterministic, partial]),

    % Set-Content - write file content
    declare_binding(powershell, set_content/2, 'Set-Content',
        [path, content], [],
        [effect(io), effect(state), deterministic]),

    % Remove-Item - delete file/directory
    declare_binding(powershell, remove_item/1, 'Remove-Item',
        [path], [],
        [effect(io), effect(state), deterministic]),

    % New-Item - create file/directory
    declare_binding(powershell, new_item/2, 'New-Item',
        [path, item_type], [file_info],
        [effect(io), effect(state), deterministic]),

    % -------------------------------------------
    % Windows Services
    % -------------------------------------------

    % Get-Service - get service information
    declare_binding(powershell, get_service/1, 'Get-Service',
        [], [list(service)],
        [effect(io), nondeterministic, import('Microsoft.PowerShell.Management')]),

    % Get-Service with name
    declare_binding(powershell, get_service/2, 'Get-Service -Name',
        [service_name], [service],
        [effect(io), deterministic, partial]),

    % Start-Service
    declare_binding(powershell, start_service/1, 'Start-Service',
        [service_name], [],
        [effect(io), effect(state), deterministic]),

    % Stop-Service
    declare_binding(powershell, stop_service/1, 'Stop-Service',
        [service_name], [],
        [effect(io), effect(state), deterministic]),

    % Restart-Service
    declare_binding(powershell, restart_service/1, 'Restart-Service',
        [service_name], [],
        [effect(io), effect(state), deterministic]),

    % -------------------------------------------
    % Registry Operations
    % -------------------------------------------

    % Get-ItemProperty - read registry value
    declare_binding(powershell, get_item_property/3, 'Get-ItemProperty',
        [registry_path, property_name], [value],
        [effect(io), deterministic, partial]),

    % Set-ItemProperty - write registry value
    declare_binding(powershell, set_item_property/3, 'Set-ItemProperty',
        [registry_path, property_name, value], [],
        [effect(io), effect(state), deterministic]),

    % New-Item (registry)
    declare_binding(powershell, new_registry_key/1, 'New-Item -Path',
        [registry_path], [registry_key],
        [effect(io), effect(state), deterministic]),

    % -------------------------------------------
    % Event Log Operations
    % -------------------------------------------

    % Get-WinEvent - query event logs
    declare_binding(powershell, get_win_event/2, 'Get-WinEvent -FilterHashtable',
        [hashtable], [list(event)],
        [effect(io), nondeterministic, import('Microsoft.PowerShell.Diagnostics')]),

    % Write-EventLog - write to event log
    declare_binding(powershell, write_event_log/4, 'Write-EventLog',
        [log_name, source, event_id, message], [],
        [effect(io), effect(state), deterministic]),

    % -------------------------------------------
    % WMI/CIM Operations
    % -------------------------------------------

    % Get-CimInstance - query WMI/CIM
    declare_binding(powershell, get_cim_instance/2, 'Get-CimInstance',
        [class_name], [list(cim_object)],
        [effect(io), nondeterministic, import('CimCmdlets')]),

    % Invoke-CimMethod - call WMI method
    declare_binding(powershell, invoke_cim_method/4, 'Invoke-CimMethod',
        [class_name, method_name, arguments], [result],
        [effect(io), effect(state), deterministic]),

    % -------------------------------------------
    % Process Operations
    % -------------------------------------------

    % Get-Process
    declare_binding(powershell, get_process/1, 'Get-Process',
        [], [list(process)],
        [effect(io), nondeterministic]),

    % Get-Process with name
    declare_binding(powershell, get_process/2, 'Get-Process -Name',
        [process_name], [list(process)],
        [effect(io), nondeterministic, partial]),

    % Stop-Process
    declare_binding(powershell, stop_process/1, 'Stop-Process',
        [process_id], [],
        [effect(io), effect(state), deterministic]),

    % Start-Process
    declare_binding(powershell, start_process/2, 'Start-Process',
        [path, list(argument)], [process],
        [effect(io), effect(state), deterministic]).

% ============================================================================
% .NET INTEGRATION BINDINGS (Chapter 4 related)
% ============================================================================

register_dotnet_bindings :-
    % -------------------------------------------
    % System.IO
    % -------------------------------------------

    % File.Exists
    declare_binding(powershell, file_exists/1, '[System.IO.File]::Exists',
        [type(string)], [],
        [effect(io), deterministic, total, pattern(exit_code_bool)]),

    % File.ReadAllText
    declare_binding(powershell, file_read_all_text/2, '[System.IO.File]::ReadAllText',
        [type(string)], [type(string)],
        [effect(io), deterministic, partial, effect(throws)]),

    % File.WriteAllText
    declare_binding(powershell, file_write_all_text/2, '[System.IO.File]::WriteAllText',
        [type(string), type(string)], [],
        [effect(io), effect(state), deterministic]),

    % Path.GetFullPath
    declare_binding(powershell, path_get_full/2, '[System.IO.Path]::GetFullPath',
        [type(string)], [type(string)],
        [pure, deterministic, total]),

    % Path.Combine
    declare_binding(powershell, path_combine/3, '[System.IO.Path]::Combine',
        [type(string), type(string)], [type(string)],
        [pure, deterministic, total]),

    % -------------------------------------------
    % System.Xml
    % -------------------------------------------

    % XmlReader.Create
    declare_binding(powershell, xml_reader_create/2, '[System.Xml.XmlReader]::Create',
        [type(string)], [type('System.Xml.XmlReader')],
        [effect(io), deterministic, partial, effect(throws)]),

    % XmlDocument.Load
    declare_binding(powershell, xml_document_load/2, '[System.Xml.XmlDocument]::new().Load',
        [type('System.Xml.XmlReader')], [type('System.Xml.XmlDocument')],
        [effect(io), deterministic]),

    % -------------------------------------------
    % System.Math
    % -------------------------------------------

    % Math.Sqrt
    declare_binding(powershell, sqrt/2, '[Math]::Sqrt',
        [type(double)], [type(double)],
        [pure, deterministic, total]),

    % Math.Round
    declare_binding(powershell, round/2, '[Math]::Round',
        [type(double)], [type(double)],
        [pure, deterministic, total]),

    % Math.Abs
    declare_binding(powershell, abs/2, '[Math]::Abs',
        [type(double)], [type(double)],
        [pure, deterministic, total]),

    % -------------------------------------------
    % String Operations
    % -------------------------------------------

    % String.Split
    declare_binding(powershell, string_split/3, '.Split',
        [type(string), type(char)], [type('string[]')],
        [pure, deterministic, total]),

    % String.Trim
    declare_binding(powershell, string_trim/2, '.Trim()',
        [type(string)], [type(string)],
        [pure, deterministic, total]),

    % String.Replace
    declare_binding(powershell, string_replace/4, '.Replace',
        [type(string), type(string), type(string)], [type(string)],
        [pure, deterministic, total]).

% ============================================================================
% C# HOSTING BINDINGS (Chapter 6)
% ============================================================================
%
% These bindings support in-process C# ↔ PowerShell communication.
% They enable hosting C# code from PowerShell and vice versa.

register_csharp_hosting_bindings :-
    % -------------------------------------------
    % Add-Type (Inline C# Compilation)
    % -------------------------------------------

    % Add-Type with TypeDefinition - compile inline C# code
    declare_binding(powershell, add_type/1, 'Add-Type -TypeDefinition',
        [type(string)], [],
        [effect(state), deterministic,
         imports(['Microsoft.CSharp'])]),

    % Add-Type with AssemblyName - load existing assembly
    declare_binding(powershell, load_assembly/1, 'Add-Type -AssemblyName',
        [type(string)], [],
        [effect(state), deterministic]),

    % Add-Type with Path - load DLL from path
    declare_binding(powershell, load_dll/1, 'Add-Type -Path',
        [type(string)], [],
        [effect(state), effect(io), deterministic]),

    % -------------------------------------------
    % .NET Object Creation
    % -------------------------------------------

    % New-Object - create .NET instance
    declare_binding(powershell, new_object/2, 'New-Object',
        [type(string)], [type(object)],
        [effect(state), deterministic]),

    % New-Object with ArgumentList
    declare_binding(powershell, new_object/3, 'New-Object -TypeName',
        [type(string), type(array)], [type(object)],
        [effect(state), deterministic]),

    % -------------------------------------------
    % Runspace Management
    % -------------------------------------------

    % Create a new runspace
    declare_binding(powershell, create_runspace/1, '[System.Management.Automation.Runspaces.RunspaceFactory]::CreateRunspace()',
        [], [type('System.Management.Automation.Runspaces.Runspace')],
        [effect(state), deterministic,
         imports(['System.Management.Automation'])]),

    % Open a runspace
    declare_binding(powershell, open_runspace/1, '.Open()',
        [type('Runspace')], [],
        [effect(state), deterministic]),

    % Create PowerShell instance
    declare_binding(powershell, create_powershell/1, '[System.Management.Automation.PowerShell]::Create()',
        [], [type('System.Management.Automation.PowerShell')],
        [effect(state), deterministic,
         imports(['System.Management.Automation'])]),

    % -------------------------------------------
    % Script Execution
    % -------------------------------------------

    % Add script to PowerShell instance
    declare_binding(powershell, add_script/2, '.AddScript',
        [type('PowerShell'), type(string)], [type('PowerShell')],
        [effect(state), deterministic]),

    % Add command to PowerShell instance
    declare_binding(powershell, add_command/2, '.AddCommand',
        [type('PowerShell'), type(string)], [type('PowerShell')],
        [effect(state), deterministic]),

    % Add parameter to PowerShell instance
    declare_binding(powershell, add_parameter/3, '.AddParameter',
        [type('PowerShell'), type(string), type(any)], [type('PowerShell')],
        [effect(state), deterministic]),

    % Invoke PowerShell - execute and get results
    declare_binding(powershell, invoke_powershell/2, '.Invoke()',
        [type('PowerShell')], [type('Collection<PSObject>')],
        [effect(io), deterministic]),

    % -------------------------------------------
    % Type Conversion
    % -------------------------------------------

    % Cast to specific .NET type
    declare_binding(powershell, cast_type/3, '-as',
        [type(any), type(type)], [type(any)],
        [pure, deterministic]),

    % Check type
    declare_binding(powershell, is_type/2, '-is',
        [type(any), type(type)], [],
        [pure, deterministic, pattern(exit_code_bool)]),

    % -------------------------------------------
    % Assembly Inspection
    % -------------------------------------------

    % Get types from assembly
    declare_binding(powershell, get_assembly_types/2, '.GetTypes()',
        [type('Assembly')], [type('Type[]')],
        [pure, deterministic]),

    % Get assembly from type
    declare_binding(powershell, get_type_assembly/2, '.Assembly',
        [type('Type')], [type('Assembly')],
        [pure, deterministic]).

% ============================================================================
% TESTING
% ============================================================================

test_powershell_bindings :-
    format('~n=== Testing PowerShell Bindings ===~n~n'),

    % Initialize bindings
    format('[Setup] Initializing PowerShell bindings...~n'),
    init_powershell_bindings,

    % Test 1: Check cmdlet bindings exist
    format('[Test 1] Cmdlet bindings~n'),
    (   ps_binding(write_output/1, _, _, _, _)
    ->  format('  [PASS] write_output/1 binding exists~n')
    ;   format('  [FAIL] write_output/1 binding missing~n')
    ),
    (   ps_binding(foreach_object/2, _, _, _, Opts1),
        member(pattern(pipe_transform), Opts1)
    ->  format('  [PASS] foreach_object/2 has pipe_transform pattern~n')
    ;   format('  [FAIL] foreach_object/2 pattern missing~n')
    ),

    % Test 2: Check automation bindings
    format('[Test 2] Automation bindings~n'),
    (   ps_binding(get_service/1, _, _, _, _)
    ->  format('  [PASS] get_service/1 binding exists~n')
    ;   format('  [FAIL] get_service/1 binding missing~n')
    ),
    (   ps_binding(get_cim_instance/2, _, _, _, Opts2),
        member(import(_), Opts2)
    ->  format('  [PASS] get_cim_instance/2 has import declaration~n')
    ;   format('  [FAIL] get_cim_instance/2 import missing~n')
    ),

    % Test 3: Check .NET bindings
    format('[Test 3] .NET bindings~n'),
    (   ps_binding(sqrt/2, Name, _, _, Opts3),
        member(pure, Opts3)
    ->  format('  [PASS] sqrt/2 is pure: ~w~n', [Name])
    ;   format('  [FAIL] sqrt/2 binding issue~n')
    ),

    % Test 4: Count total bindings
    format('[Test 4] Total binding count~n'),
    bindings_for_target(powershell, AllBindings),
    length(AllBindings, Total),
    format('  Total PowerShell bindings: ~w~n', [Total]),
    (   Total >= 40
    ->  format('  [PASS] Sufficient bindings registered~n')
    ;   format('  [WARN] Expected more bindings (got ~w)~n', [Total])
    ),

    % Test 5: Check imports
    format('[Test 5] Import declarations~n'),
    binding_imports(powershell, Imports),
    format('  Required imports: ~w~n', [Imports]),

    format('~n=== PowerShell Bindings Tests Complete ===~n').
