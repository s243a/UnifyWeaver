# Tool Availability Detection Design

**Created:** 2025-10-26
**Status:** Design Phase
**Branch:** `firewall-tool-detection`

---

## Overview

Design a multi-level system for detecting tool availability and controlling how UnifyWeaver responds when required tools are missing.

---

## Core Philosophy: Multi-Level Configuration

Following UnifyWeaver's configuration philosophy, tool availability policies can be specified at three levels:

1. **Global/Default Level** - System-wide preferences
2. **Firewall Policy Level** - Policy-specific rules
3. **Compilation Options Level** - Per-predicate overrides

**Priority:** Compilation Options > Firewall Policy > Global Default

---

## Configuration Predicates

### 1. Global/Default Level

Location: User configuration file or environment variables

```prolog
% Global preference for handling unavailable tools
tool_availability_policy(warn).     % warn | forbid | ignore | list_required

% Specific tool requirements (optional)
require_tool(bash).
require_tool(jq).
prefer_available_tools(true).       % Prefer alternatives if tool unavailable
```

**Behaviors:**
- `warn` - Compile anyway, print warnings about missing tools
- `forbid` - Fail compilation if required tool is unavailable
- `ignore` - Don't check tool availability (assume all available)
- `list_required` - Compile, but output list of required tools at end

### 2. Firewall Policy Level

Location: `src/unifyweaver/core/firewall_v2.pl` policies

```prolog
% Pure PowerShell policy example
load_firewall_policy(pure_powershell) :-
    set_firewall_mode(permissive),

    % Deny bash service for PowerShell
    assertz(denied_service(powershell, executable(bash))),

    % If bash unavailable, prefer pure PowerShell
    assertz(tool_availability_policy(prefer_alternatives)),
    assertz(prefer_tool_alternative(bash, powershell_cmdlets)),

    format('[Firewall] PowerShell will use pure mode (no bash)~n', []).

% Strict security policy example
load_firewall_policy(strict_security) :-
    set_firewall_mode(strict),

    % Forbid compilation if required tools unavailable
    assertz(tool_availability_policy(forbid)),

    % Only allow explicitly approved tools
    assertz(allowed_tool(bash)),
    assertz(allowed_tool(awk)),
    % jq, curl, python not allowed
    assertz(denied_tool(jq)),
    assertz(denied_tool(curl)),
    assertz(denied_tool(python3)).

% Permissive developer policy
load_firewall_policy(permissive_dev) :-
    set_firewall_mode(permissive),

    % List what's needed but don't fail
    assertz(tool_availability_policy(list_required)),

    % Prefer available tools over unavailable ones
    assertz(prefer_available_tools(true)).
```

### 3. Compilation Options Level

Location: Per-predicate compilation call

```prolog
% Override at compilation time
compile_to_powershell(my_pred/2, [
    source_type(json),
    json_file('data.json'),

    % Tool availability options (override firewall/global)
    tool_availability_policy(forbid),      % Fail if jq missing
    require_tool(jq),                      % Explicitly require jq

    % Or: allow compilation but warn
    tool_availability_policy(warn)
], Code).

% Another example: prefer alternatives
compile_to_bash(csv_pred/3, [
    source_type(csv),
    csv_file('data.csv'),

    % If AWK unavailable, try pure bash solution
    prefer_available_tools(true),
    tool_alternatives([
        preferred(awk),
        fallback(pure_bash),
        fallback(python)
    ])
], Code).
```

---

## Tool Detection System

### Tool Registry

```prolog
% src/unifyweaver/core/tool_detection.pl

%% Executable tools
tool_executable(bash, 'bash').
tool_executable(awk, 'awk').
tool_executable(sed, 'sed').
tool_executable(jq, 'jq').
tool_executable(curl, 'curl').
tool_executable(python3, 'python3').
tool_executable(sqlite3, 'sqlite3').

%% PowerShell cmdlets (always available in PowerShell)
tool_cmdlet(import_csv, 'Import-Csv').
tool_cmdlet(convertfrom_json, 'ConvertFrom-Json').
tool_cmdlet(invoke_restmethod, 'Invoke-RestMethod').

%% Tool alternatives (what can replace what)
tool_alternative(awk, pure_bash).
tool_alternative(jq, powershell_cmdlets).
tool_alternative(curl, powershell_cmdlets).
```

### Detection Predicates

```prolog
%% detect_tool_availability(+Tool, -Status)
%  Check if a tool is available on the system
%  Status: available | unavailable(Reason)
detect_tool_availability(Tool, Status) :-
    tool_executable(Tool, Command),
    (   check_executable_exists(Command)
    ->  Status = available
    ;   Status = unavailable(executable_not_found)
    ).

detect_tool_availability(Tool, available) :-
    tool_cmdlet(Tool, _).  % Cmdlets assumed available

%% check_executable_exists(+Command)
%  Cross-platform check if executable exists
check_executable_exists(Command) :-
    % Try 'which' on Unix-like systems
    (   catch(process_create(path(which), [Command], [stdout(null), stderr(null), process(PID)]), _, fail),
        process_wait(PID, exit(0))
    ->  true
    % Try 'where' on Windows
    ;   catch(process_create(path(where), [Command], [stdout(null), stderr(null), process(PID)]), _, fail),
        process_wait(PID, exit(0))
    ).

%% check_all_tools_for_compilation(+ToolList, +Policy, -Result)
%  Check all required tools and apply policy
%  Result: ok | unavailable(MissingTools, Action)
check_all_tools_for_compilation(ToolList, Policy, Result) :-
    findall(Tool, (
        member(Tool, ToolList),
        detect_tool_availability(Tool, unavailable(_))
    ), MissingTools),

    (   MissingTools = []
    ->  Result = ok
    ;   apply_availability_policy(Policy, MissingTools, Result)
    ).

%% apply_availability_policy(+Policy, +MissingTools, -Result)
apply_availability_policy(forbid, MissingTools, Result) :-
    Result = fail(missing_required_tools(MissingTools)).

apply_availability_policy(warn, MissingTools, Result) :-
    format('[Warning] Missing tools: ~w~n', [MissingTools]),
    Result = ok_with_warnings(MissingTools).

apply_availability_policy(list_required, MissingTools, Result) :-
    Result = ok_list_required(MissingTools).

apply_availability_policy(ignore, _MissingTools, ok).
```

---

## Integration with Compilation Pipeline

### Compilation Flow

```prolog
compile_to_bash(Predicate, Options, BashCode) :-
    % 1. Determine required tools
    determine_required_tools(Predicate, Options, RequiredTools),

    % 2. Get tool availability policy (check all three levels)
    resolve_tool_availability_policy(Options, Policy),

    % 3. Check tool availability
    check_all_tools_for_compilation(RequiredTools, Policy, CheckResult),

    % 4. Handle result
    (   CheckResult = ok
    ->  % All tools available, proceed
        generate_bash_code(Predicate, Options, BashCode)

    ;   CheckResult = ok_with_warnings(MissingTools)
    ->  % Tools missing but policy allows, generate with warnings
        generate_bash_code(Predicate, Options, BashCode),
        report_missing_tools(MissingTools, warning)

    ;   CheckResult = ok_list_required(MissingTools)
    ->  % Generate code and list requirements
        generate_bash_code(Predicate, Options, BashCode),
        append_tool_requirements(BashCode, MissingTools, BashCodeWithRequirements)

    ;   CheckResult = fail(missing_required_tools(MissingTools))
    ->  % Policy forbids compilation
        format('[Error] Cannot compile: missing required tools: ~w~n', [MissingTools]),
        fail
    ).

%% resolve_tool_availability_policy(+Options, -Policy)
%  Resolve policy from three levels (compilation > firewall > global)
resolve_tool_availability_policy(Options, Policy) :-
    % Level 3: Compilation options (highest priority)
    (   member(tool_availability_policy(P), Options)
    ->  Policy = P
    % Level 2: Firewall policy
    ;   tool_availability_policy(P)
    ->  Policy = P
    % Level 1: Global default
    ;   Policy = warn  % Default behavior
    ).
```

---

## Example Scenarios

### Scenario 1: Developer Machine (All Tools Available)

```prolog
% Compile JSON source
compile_to_bash(api_data/3, [
    source_type(json),
    json_file('api.json'),
    jq_filter('.results[]')
], Code).

% Result: Compiles successfully using jq
```

### Scenario 2: Restricted Windows (No WSL/Bash)

```prolog
% Global policy: prefer available tools
:- assertz(prefer_available_tools(true)).

% Compile CSV source
compile_to_powershell(user_data/3, [
    source_type(csv),
    csv_file('users.csv'),
    powershell_mode(auto)
], Code).

% Result:
% - Detects bash unavailable
% - Firewall derives powershell_mode(pure)
% - Generates pure PowerShell using Import-Csv
```

### Scenario 3: CI/CD Pipeline (Strict Validation)

```prolog
% Load strict policy
load_firewall_policy(strict_ci_validation).

% Policy includes:
assertz(tool_availability_policy(forbid)).
assertz(require_tool(bash)).
assertz(require_tool(awk)).

% Attempt to compile
compile_to_bash(data/2, [source_type(csv), csv_file('data.csv')], Code).

% Result:
% - Checks for bash and awk
% - If missing: fails with error message
% - If present: compiles successfully
```

### Scenario 4: List Requirements Mode

```prolog
% User wants to see what tools are needed
compile_to_bash(pipeline/3, [
    source_type(json),
    json_file('data.json'),
    tool_availability_policy(list_required)
], Code).

% Generated bash script includes:
# #!/bin/bash
# # Required tools:
# #   - jq (JSON processor)
# #   - curl (HTTP client)
# #
# # Install on Ubuntu/Debian:
# #   sudo apt-get install jq curl
# #
# # Install on macOS:
# #   brew install jq curl
```

---

## Firewall Integration

### New Firewall Predicates

```prolog
% Dynamic predicates in firewall_v2.pl
:- dynamic tool_availability_policy/1.
:- dynamic require_tool/1.
:- dynamic prefer_available_tools/1.
:- dynamic allowed_tool/1.
:- dynamic denied_tool/1.
:- dynamic tool_alternative/2.
```

### Firewall Implications

```prolog
% If bash unavailable → deny bash service for PowerShell
firewall_implies_default(tool_unavailable(bash),
                        denied_service(powershell, executable(bash))).

% If prefer_available_tools → prefer pure PowerShell when bash unavailable
firewall_implies_default(prefer_available_tools(true),
                        prefer_alternative_when_unavailable).
```

---

## Implementation Plan

### Phase 1: Core Tool Detection
1. Create `src/unifyweaver/core/tool_detection.pl`
2. Implement `detect_tool_availability/2`
3. Implement cross-platform executable checking
4. Add tool registry (executables, cmdlets)

### Phase 2: Policy System
1. Add dynamic predicates to `firewall_v2.pl`
2. Implement multi-level policy resolution
3. Add policy application logic
4. Create policy templates

### Phase 3: Compilation Integration
1. Integrate detection into compilation pipeline
2. Add tool requirement determination
3. Implement warning/error reporting
4. Add "list required tools" feature

### Phase 4: Testing
1. Test on Windows (no bash)
2. Test on Linux (full tools)
3. Test with various policies
4. Test tool alternatives

---

## Benefits

1. **Flexible**: Users choose how to handle missing tools
2. **Informative**: Clear messages about what's needed
3. **Portable**: Code adapts to available tools
4. **Secure**: Can enforce tool restrictions
5. **Developer-Friendly**: Helpful error messages and suggestions

---

**Next Steps:**
1. Get user feedback on design
2. Implement Phase 1 (core detection)
3. Test on multiple platforms
4. Integrate with firewall

---

**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
**License:** MIT OR Apache-2.0
