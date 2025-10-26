:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% tool_detection.pl - Cross-platform tool availability detection
%
% This module provides predicates for detecting whether external tools
% (executables, PowerShell cmdlets, etc.) are available on the system.
%
% Philosophy: Multi-level configuration
% - Global defaults
% - Firewall policy level
% - Compilation option level (highest priority)

:- module(tool_detection, [
    % Detection predicates
    detect_tool_availability/2,    % detect_tool_availability(+Tool, -Status)
    check_executable_exists/1,     % check_executable_exists(+Command)
    check_powershell_available/0,  % Check if PowerShell is available
    check_powershell_cmdlet/1,     % check_powershell_cmdlet(+Cmdlet)

    % Tool registry
    tool_executable/2,             % tool_executable(+ToolName, -ExecutableName)
    tool_cmdlet/2,                 % tool_cmdlet(+ToolName, -CmdletName)
    tool_alternative/2,            % tool_alternative(+Tool, -Alternative)

    % Batch checking
    check_all_tools/2,             % check_all_tools(+ToolList, -Result)
    determine_available_tools/2,   % determine_available_tools(+ToolList, -AvailableTools)

    % Utility predicates
    suggest_tool_installation/2    % suggest_tool_installation(+Tool, -Suggestion)
]).

:- use_module(library(process)).
:- use_module(library(lists)).

%% ============================================
%% TOOL REGISTRY
%% ============================================

%% tool_executable(?ToolName, ?ExecutableName)
%  Registry of executable tools and their command names
tool_executable(bash, 'bash').
tool_executable(sh, 'sh').
tool_executable(awk, 'awk').
tool_executable(gawk, 'gawk').
tool_executable(sed, 'sed').
tool_executable(grep, 'grep').
tool_executable(jq, 'jq').
tool_executable(curl, 'curl').
tool_executable(wget, 'wget').
tool_executable(python3, 'python3').
tool_executable(python, 'python').
tool_executable(sqlite3, 'sqlite3').
tool_executable(powershell, 'pwsh').      % PowerShell Core (cross-platform)
tool_executable(powershell_legacy, 'powershell.exe').  % Windows PowerShell

%% tool_cmdlet(?ToolName, ?CmdletName)
%  Registry of PowerShell cmdlets
%  Note: These are only available when PowerShell is the execution context
tool_cmdlet(import_csv, 'Import-Csv').
tool_cmdlet(export_csv, 'Export-Csv').
tool_cmdlet(convertfrom_json, 'ConvertFrom-Json').
tool_cmdlet(convertto_json, 'ConvertTo-Json').
tool_cmdlet(invoke_restmethod, 'Invoke-RestMethod').
tool_cmdlet(invoke_webrequest, 'Invoke-WebRequest').
tool_cmdlet(get_content, 'Get-Content').
tool_cmdlet(set_content, 'Set-Content').
tool_cmdlet(select_object, 'Select-Object').
tool_cmdlet(where_object, 'Where-Object').
tool_cmdlet(foreach_object, 'ForEach-Object').

%% tool_alternative(?Tool, ?Alternative)
%  What tools can substitute for each other
tool_alternative(awk, gawk).
tool_alternative(awk, pure_bash).
tool_alternative(jq, powershell_cmdlets).   % ConvertFrom-Json
tool_alternative(curl, wget).
tool_alternative(curl, powershell_cmdlets). % Invoke-RestMethod
tool_alternative(python, python3).
tool_alternative(powershell_legacy, powershell).  % pwsh can replace powershell.exe

%% ============================================
%% CORE DETECTION PREDICATES
%% ============================================

%% detect_tool_availability(+Tool, -Status)
%  Detect if a tool is available on the system.
%
%  Tool: atom representing the tool (e.g., bash, jq, import_csv)
%  Status: available | unavailable(Reason)
%
%  Examples:
%    detect_tool_availability(bash, Status).
%    detect_tool_availability(import_csv, Status).

detect_tool_availability(Tool, Status) :-
    % Check if it's an executable
    (   tool_executable(Tool, Command)
    ->  (   check_executable_exists(Command)
        ->  Status = available
        ;   Status = unavailable(executable_not_found)
        )
    % Check if it's a PowerShell cmdlet
    ;   tool_cmdlet(Tool, Cmdlet)
    ->  (   check_powershell_cmdlet(Cmdlet)
        ->  Status = available
        ;   Status = unavailable(powershell_not_available)
        )
    % Unknown tool
    ;   Status = unavailable(unknown_tool)
    ).

%% check_executable_exists(+Command)
%  Cross-platform check if an executable exists in PATH.
%  Works on Linux, macOS, and Windows.
%
%  Command: string or atom representing the executable name

check_executable_exists(Command) :-
    atom_string(CommandAtom, Command),

    % Try 'which' on Unix-like systems (Linux, macOS, WSL)
    (   catch(
            process_create(path(which), [CommandAtom],
                          [stdout(null), stderr(null), process(PID)]),
            _,
            fail
        ),
        process_wait(PID, exit(0))
    ->  true

    % Try 'where' on Windows (cmd.exe)
    ;   catch(
            process_create(path(where), [CommandAtom],
                          [stdout(null), stderr(null), process(PID)]),
            _,
            fail
        ),
        process_wait(PID, exit(0))

    % Try 'command -v' as fallback (POSIX shells)
    ;   catch(
            process_create(path(sh), ['-c', 'command -v ' + CommandAtom],
                          [stdout(null), stderr(null), process(PID)]),
            _,
            fail
        ),
        process_wait(PID, exit(0))
    ).

%% check_powershell_available
%  Check if PowerShell is available on the system.
%  Checks for both pwsh (cross-platform) and powershell.exe (Windows).

check_powershell_available :-
    % Try PowerShell Core (cross-platform: Linux, macOS, Windows)
    (   check_executable_exists('pwsh')
    ->  true
    % Try Windows PowerShell (Windows only)
    ;   check_executable_exists('powershell.exe')
    ).

%% check_powershell_cmdlet(+Cmdlet)
%  Check if a PowerShell cmdlet is available.
%  This requires PowerShell to be installed and accessible.
%
%  Cmdlet: atom or string representing cmdlet name (e.g., 'Import-Csv')

check_powershell_cmdlet(Cmdlet) :-
    atom_string(CmdletAtom, Cmdlet),

    % First check if PowerShell is available
    check_powershell_available,

    % Try PowerShell Core first
    (   check_cmdlet_with_powershell('pwsh', CmdletAtom)
    ->  true
    % Fall back to Windows PowerShell
    ;   check_cmdlet_with_powershell('powershell.exe', CmdletAtom)
    ).

%% check_cmdlet_with_powershell(+PowerShellExe, +Cmdlet)
%  Internal helper to check cmdlet with specific PowerShell executable

check_cmdlet_with_powershell(PowerShellExe, Cmdlet) :-
    format(atom(CheckCommand), 'Get-Command ~w -ErrorAction SilentlyContinue', [Cmdlet]),
    catch(
        process_create(path(PowerShellExe), ['-Command', CheckCommand],
                      [stdout(null), stderr(null), process(PID)]),
        _,
        fail
    ),
    process_wait(PID, exit(0)).

%% ============================================
%% BATCH CHECKING
%% ============================================

%% check_all_tools(+ToolList, -Result)
%  Check availability of all tools in list.
%
%  ToolList: list of tool names
%  Result: all_available | missing(MissingTools)
%
%  Example:
%    check_all_tools([bash, awk, jq], Result).

check_all_tools(ToolList, Result) :-
    findall(Tool, (
        member(Tool, ToolList),
        detect_tool_availability(Tool, unavailable(_))
    ), MissingTools),

    (   MissingTools = []
    ->  Result = all_available
    ;   Result = missing(MissingTools)
    ).

%% determine_available_tools(+ToolList, -AvailableTools)
%  From a list of tools, return only those that are available.
%
%  Example:
%    determine_available_tools([bash, jq, nonexistent], Available).
%    Available = [bash, jq].

determine_available_tools(ToolList, AvailableTools) :-
    findall(Tool, (
        member(Tool, ToolList),
        detect_tool_availability(Tool, available)
    ), AvailableTools).

%% ============================================
%% INSTALLATION SUGGESTIONS
%% ============================================

%% suggest_tool_installation(+Tool, -Suggestion)
%  Provide installation suggestions for missing tools.
%  Returns platform-specific installation commands.

suggest_tool_installation(Tool, Suggestion) :-
    tool_executable(Tool, Executable),
    format(atom(Suggestion),
           'Install ~w:~n  Ubuntu/Debian: sudo apt-get install ~w~n  macOS: brew install ~w~n  Windows: Use WSL or install via package manager',
           [Tool, Executable, Executable]).

suggest_tool_installation(Tool, Suggestion) :-
    tool_cmdlet(Tool, Cmdlet),
    format(atom(Suggestion),
           'PowerShell cmdlet ~w requires PowerShell to be installed:~n  Linux: https://docs.microsoft.com/powershell/scripting/install/installing-powershell-core-on-linux~n  macOS: brew install --cask powershell~n  Windows: PowerShell is pre-installed',
           [Cmdlet]).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% find_best_available_tool(+PreferredTools, -SelectedTool)
%  Given a list of preferred tools (in priority order),
%  return the first one that is available.
%
%  Example:
%    find_best_available_tool([awk, gawk, pure_bash], Tool).

find_best_available_tool([Tool|Rest], SelectedTool) :-
    (   detect_tool_availability(Tool, available)
    ->  SelectedTool = Tool
    ;   find_best_available_tool(Rest, SelectedTool)
    ).

find_best_available_tool([], none) :-
    % No tools available
    !.
