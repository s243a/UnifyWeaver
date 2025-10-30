<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->

# UnifyWeaver Firewall Guide

The UnifyWeaver firewall system provides security and preference guidance for compilation decisions. It controls access to external tools, network resources, file operations, and more.

## Table of Contents

1. [Overview](#overview)
2. [Philosophy](#philosophy)
3. [Higher-Order Firewall Implications](#higher-order-firewall-implications)
4. [Multi-Level Configuration](#multi-level-configuration)
5. [Network Access Control](#network-access-control)
6. [Service Control](#service-control)
7. [Policy Templates](#policy-templates)
8. [Tool Detection](#tool-detection)
9. [Examples](#examples)

---

## Overview

The firewall system operates on fundamental security policies rather than derived predicates. It provides three modes of operation:

- **Strict Mode** - Deny by default, require explicit allow
- **Permissive Mode** - Allow by default, deny specific operations
- **Disabled Mode** - No firewall checks

### Key Features

- **Network Access Control** - URL/domain whitelisting and blacklisting
- **Service Restrictions** - Control which external tools can be used
- **Python Module Whitelisting** - Restrict Python imports
- **File Access Patterns** - Control read/write permissions
- **Multi-Level Configuration** - Global, policy, and compilation-specific settings
- **Tool Availability Detection** - Cross-platform tool checking

---

## Philosophy

### Fundamental vs. Derived Rules

The firewall operates on **fundamental policies**, not derived predicates:

**Fundamental Rules** (stored in firewall):
```prolog
% Target language policies
allowed_target_language(bash).
allowed_target_language(powershell).

% Service/tool policies
allowed_service(bash, executable(awk)).
denied_service(powershell, executable(bash)).  % Pure PowerShell

% Network policies
network_access_policy(whitelist).
allowed_domain('example.com').
```

**Derived Preferences** (computed from fundamental rules):
```prolog
% Derived rule: PowerShell mode based on fundamental policies
powershell_mode_preference(pure) :-
    % Pure mode required if bash service is denied for PowerShell
    denied_service(powershell, executable(bash)),
    allowed_target_language(powershell).

powershell_mode_preference(baas) :-
    % BaaS allowed if bash service is allowed for PowerShell
    allowed_service(powershell, executable(bash)),
    allowed_target_language(powershell).
```

### Hybrid Response System

The firewall provides three-level responses:

```prolog
firewall_check(Action, Result) :-
    (   allowed(Action)          -> Result = allow
    ;   preferred_fallback(Action) -> Result = warn(Reason)
    ;   denied(Action)            -> Result = deny(Reason)
    ;   unknown_action(Action)   -> Result = mode_dependent
    ).
```

**Results:**
- `allow` - Operation permitted
- `warn(Reason)` - Permitted with warning
- `deny(Reason)` - Operation blocked
- `mode_dependent` - Depends on firewall mode (strict/permissive)

---

## Higher-Order Firewall Implications

The `firewall_implies` system demonstrates Prolog's strength in logical inference by deriving complex policies from simple fundamental conditions. This powerful feature allows security policies to be composed declaratively with full user control.

### Architecture

The system provides three levels of implication rules:

1. **Default Implications** (`firewall_implies_default/2`) - Built-in sensible defaults
2. **User-Defined Implications** (`firewall_implies/2`) - Custom rules that extend or override defaults
3. **Disabled Implications** (`firewall_implies_disabled/2`) - Explicit blocking of unwanted implications

### How It Works

When a condition is detected (e.g., `no_bash_available`), the firewall can automatically derive appropriate policies:

```prolog
% Default implication (built-in)
firewall_implies_default(no_bash_available,
                        denied(service(powershell, executable(bash)))).

% Derive all policies from a condition
?- derive_policy(no_bash_available, Policies).
Policies = [denied(service(powershell, executable(bash)))].
```

### Default Implications

UnifyWeaver includes 10 default implications covering common scenarios:

1. **No Bash Available** → Deny bash service for PowerShell
2. **Bash Target Denied** → Deny bash service for all targets
3. **Network Denied** → Deny all network services
4. **Executable Denied** → Deny specific tool as service
5. **Strict Security** → Prefer built-ins over executables
6. **Restricted Environment** → Deny external services
7. **Language Denied** → Deny target execution
8. **Pure Mode Preferred** → Prefer native implementations
9. **Offline Mode** → Deny network access
10. **Portable Required** → Prefer cross-platform tools

### User Control

Users have full control over implications:

**Override Defaults:**
```prolog
% Add custom implication (coexists with default)
:- assertz(firewall:firewall_implies(no_bash_available,
                                     allowed(service(powershell, executable(wsl))))).
```

**Disable Defaults:**
```prolog
% Disable specific default implication
:- assertz(firewall:firewall_implies_disabled(no_bash_available,
                                              denied(service(powershell, executable(bash))))).
```

**Replace Completely:**
```prolog
% Disable default and add custom rule
:- assertz(firewall:firewall_implies_disabled(no_bash_available, _)).
:- assertz(firewall:firewall_implies(no_bash_available,
                                     allowed(service(powershell, executable(wsl))))).
```

### Deriving Policies

The `derive_policy/2` predicate collects all applicable policies from a condition:

```prolog
% Get all policies derived from offline mode
?- derive_policy(mode(offline), Policies).
Policies = [
    network_access(denied),
    denied(service(_, http(_)))
].

% Check if condition implies expected policy
?- check_derived_policy(security_policy(strict),
                       [prefer(service(powershell, cmdlet(_)),
                               service(powershell, executable(_)))],
                       Result).
Result = true.
```

### Complex Scenarios

Implications can be combined to handle complex environments:

```prolog
% Corporate banking policy
:- assertz(firewall:firewall_implies(corporate_policy(banking),
                                     denied(service(_, network_access(external))))).

% Restricted offline environment
derive_policy(environment(restricted), RestrictedPolicies),
derive_policy(mode(offline), OfflinePolicies),
% Combine both sets of policies for complete restrictions
```

### Why This Matters

This feature showcases Prolog's unique strengths:

- **Declarative Reasoning** - Policies derived through logical inference, not imperative code
- **Composability** - Simple rules combine to express complex security policies
- **User Control** - Defaults can be overridden without modifying core code
- **Testability** - `check_derived_policy/3` enables policy verification
- **Elegance** - What would require complex inheritance hierarchies in OOP is expressed concisely in Prolog

### Example: Pure PowerShell Environment

When bash is unavailable (common in restricted Windows environments):

```prolog
% System detects condition
Condition = no_bash_available,

% Firewall automatically derives policy
derive_policy(Condition, Policies),
% Policies = [denied(service(powershell, executable(bash)))]

% Compiler queries firewall
PowerShellMode = (
    firewall_implies(no_bash_available, denied(service(powershell, executable(bash))))
    -> pure  % Must use pure PowerShell
    ;  auto  % Can choose
).
```

### Testing

The firewall includes comprehensive tests for the implication system:

```bash
$ swipl -q -l examples/test_firewall_implies.pl -g main -t halt
```

Tests verify:
- ✅ Default implications work correctly
- ✅ User-defined implications override defaults
- ✅ Disabling defaults works as expected
- ✅ `derive_policy/2` aggregates policies correctly
- ✅ Complex multi-condition scenarios handled properly

---

## Multi-Level Configuration

Following UnifyWeaver's configuration philosophy, firewall policies can be specified at three levels:

### 1. Global/Default Level

System-wide preferences (lowest priority):

```prolog
% Set firewall mode
:- assertz(firewall_mode(permissive)).

% Global tool requirements
:- assertz(require_tool(bash)).
:- assertz(require_tool(jq)).

% Tool availability policy
:- assertz(tool_availability_policy(warn)).  % warn | forbid | ignore
```

### 2. Firewall Policy Level

Policy-specific rules (medium priority):

```prolog
% Load a policy template
load_firewall_policy(strict_security) :-
    set_firewall_mode(strict),
    assertz(tool_availability_policy(forbid)),
    assertz(allowed_tool(bash)),
    assertz(allowed_tool(awk)),
    assertz(denied_tool(curl)),
    assertz(denied_tool(python3)).

% Load the policy
?- load_firewall_policy(strict_security).
```

### 3. Compilation Options Level

Per-predicate overrides (highest priority):

```prolog
% Override at compilation time
compile_to_bash(my_pred/2, [
    source_type(json),
    json_file('data.json'),

    % Tool availability options (override firewall/global)
    tool_availability_policy(forbid),      % Fail if jq missing
    require_tool(jq)                       % Explicitly require jq
], Code).
```

**Priority:** Compilation Options > Firewall Policy > Global Default

---

## Network Access Control

Control which URLs and domains can be accessed during compilation.

### Global Network Policies

```prolog
% Deny all network access
assertz(network_access_policy(deny_all)).

% Allow all network access
assertz(network_access_policy(allow_all)).

% Whitelist mode (deny unless explicitly allowed)
assertz(network_access_policy(whitelist)).

% Blacklist mode (allow unless explicitly denied)
assertz(network_access_policy(blacklist)).
```

### Domain Control

```prolog
% Allow specific domains
assertz(allowed_domain('example.com')).
assertz(allowed_domain('trusted.org')).
assertz(allowed_domain('*.internal.company.com')).  % Wildcard

% Deny specific domains
assertz(denied_domain('malicious.com')).
assertz(denied_domain('untrusted.net')).
```

### URL Pattern Matching

```prolog
% Allow URL patterns
assertz(allowed_url_pattern('/api/')).
assertz(allowed_url_pattern('/public/')).

% Deny URL patterns
assertz(denied_url_pattern('admin')).
assertz(denied_url_pattern('/private/')).
```

### Domain Pattern Matching

Supports three matching modes:

**1. Exact Match:**
```prolog
domain_matches_pattern('example.com', 'example.com').  % true
```

**2. Subdomain Match:**
```prolog
domain_matches_pattern('api.example.com', 'example.com').  % true
domain_matches_pattern('deep.api.example.com', 'example.com').  % true
```

**3. Wildcard Match:**
```prolog
domain_matches_pattern('api.example.com', '*.example.com').  % true
domain_matches_pattern('v2.api.example.com', '*.example.com').  % true
```

### Checking URL Access

```prolog
% Check if URL is allowed
check_url_access('https://api.example.com/data', Result).

% Possible results:
% - allow
% - deny(network_access_denied)
% - deny(url_pattern_denied(Pattern))
% - deny(domain_not_in_whitelist)
% - deny(domain_denied(Domain))
```

### Access Control Flow

```
1. Global network policy (deny_all, allow_all)
   ↓
2. Denied URL patterns
   ↓
3. Allowed URL patterns (whitelist mode)
   ↓
4. Domain-based rules
   ↓
5. Firewall mode (strict/permissive)
```

---

## Service Control

Control which external tools and services can be used.

### Service Types

```prolog
% Executable tools
service(TargetLang, executable(Tool))

% PowerShell cmdlets
service(powershell, cmdlet(Cmdlet))

% Network access
service(_, network_access(Type))

% File operations
service(_, file_operation(read, Pattern))
service(_, file_operation(write, Pattern))
```

### Allowing Services

```prolog
% Allow bash to use AWK
assertz(allowed_service(bash, executable(awk))).

% Allow PowerShell cmdlets
assertz(allowed_service(powershell, cmdlet(import_csv))).
assertz(allowed_service(powershell, cmdlet(convertfrom_json))).

% Allow network access
assertz(allowed_service(_, network_access(external))).
```

### Denying Services

```prolog
% Pure PowerShell mode - deny bash
assertz(denied_service(powershell, executable(bash))).

% Deny all network access
assertz(denied_service(_, network_access(_))).

% Deny specific executables
assertz(denied_service(_, executable(curl))).
assertz(denied_service(_, executable(wget))).
```

### Python Module Control

```prolog
% Allow specific Python modules
assertz(allowed_python_module(sys)).
assertz(allowed_python_module(json)).
assertz(allowed_python_module(sqlite3)).
assertz(allowed_python_module(csv)).

% Deny dangerous modules
assertz(denied_python_module(os)).
assertz(denied_python_module(subprocess)).
assertz(denied_python_module(shutil)).
```

### File Access Control

```prolog
% Allow reading from specific directories
assertz(allowed_file_read('data/*')).
assertz(allowed_file_read('config/*')).
assertz(allowed_file_read('/tmp/*')).

% Allow writing to specific directories
assertz(allowed_file_write('output/*')).
assertz(allowed_file_write('/tmp/*')).
assertz(allowed_file_write('cache/*')).

% Deny sensitive paths
assertz(denied_file_read('/etc/*')).
assertz(denied_file_write('/etc/*')).
assertz(denied_file_write('/bin/*')).
```

---

## Policy Templates

Pre-configured firewall policies for common scenarios.

### No Network Policy

Blocks all network access:

```prolog
load_firewall_policy(no_network) :-
    set_firewall_mode(permissive),
    assertz(network_access_policy(deny_all)),
    assertz(denied_service(_, network_access(_))),
    assertz(denied_service(_, executable(curl))),
    assertz(denied_service(_, executable(wget))),
    format('[Firewall] Network access blocked~n', []).
```

Usage:
```prolog
?- load_firewall_policy(no_network).
?- check_url_access('https://example.com', Result).
Result = deny(network_access_denied).
```

### Whitelist Domains Policy

Allow only specific domains:

```prolog
load_firewall_policy(whitelist_domains(Domains)) :-
    set_firewall_mode(permissive),
    assertz(network_access_policy(whitelist)),
    forall(member(Domain, Domains),
           assertz(allowed_domain(Domain))),
    format('[Firewall] Only domains ~w are allowed~n', [Domains]).
```

Usage:
```prolog
?- load_firewall_policy(whitelist_domains(['example.com', 'trusted.org'])).
?- check_url_access('https://api.example.com/data', Result).
Result = allow.
?- check_url_access('https://untrusted.com/data', Result).
Result = deny(domain_not_in_whitelist).
```

### Pure PowerShell Policy

Deny bash for PowerShell, forcing pure mode:

```prolog
load_firewall_policy(pure_powershell) :-
    set_firewall_mode(permissive),
    assertz(denied_service(powershell, executable(bash))),
    assertz(allowed_service(powershell, cmdlet(_))),  % Allow all cmdlets
    format('[Firewall] Pure PowerShell mode enforced~n', []).
```

### Strict Security Policy

Deny by default, allow only explicitly permitted operations:

```prolog
load_firewall_policy(strict_security) :-
    set_firewall_mode(strict),

    % Only allow specific tools
    assertz(allowed_tool(bash)),
    assertz(allowed_tool(awk)),

    % Network whitelist
    assertz(network_access_policy(whitelist)),
    assertz(allowed_domain('api.internal.company.com')),

    % Python restrictions
    assertz(allowed_python_module(sys)),
    assertz(allowed_python_module(json)),

    % File access restrictions
    assertz(allowed_file_read('data/*')),
    assertz(allowed_file_write('output/*')),

    format('[Firewall] Strict security policy loaded~n', []).
```

### Permissive Development Policy

Allow by default, warn on potential issues:

```prolog
load_firewall_policy(permissive_dev) :-
    set_firewall_mode(permissive),
    assertz(tool_availability_policy(list_required)),
    assertz(prefer_available_tools(true)),
    format('[Firewall] Permissive development mode~n', []).
```

---

## Tool Detection

Cross-platform detection of tool availability.

### Detecting Tools

```prolog
% Check if a tool is available
detect_tool_availability(bash, Status).
% Status = available | unavailable(Reason)

% Check specific executables
check_executable_exists('bash').    % true/false
check_executable_exists('jq').      % true/false

% Check PowerShell availability
check_powershell_available.         % true/false

% Check PowerShell cmdlets
check_powershell_cmdlet('Import-Csv').  % true/false
```

### Tool Registry

```prolog
% Executable tools
tool_executable(bash, 'bash').
tool_executable(awk, 'awk').
tool_executable(jq, 'jq').
tool_executable(curl, 'curl').
tool_executable(python3, 'python3').
tool_executable(powershell, 'pwsh').

% PowerShell cmdlets
tool_cmdlet(import_csv, 'Import-Csv').
tool_cmdlet(convertfrom_json, 'ConvertFrom-Json').
tool_cmdlet(invoke_restmethod, 'Invoke-RestMethod').

% Tool alternatives
tool_alternative(awk, gawk).
tool_alternative(awk, pure_bash).
tool_alternative(jq, powershell_cmdlets).
tool_alternative(curl, wget).
tool_alternative(curl, powershell_cmdlets).
```

### Batch Checking

```prolog
% Check all required tools
check_all_tools([bash, awk, jq], Result).
% Result = all_available | missing(MissingTools)

% Get available tools from list
determine_available_tools([bash, jq, nonexistent], Available).
% Available = [bash, jq]
```

### Tool Availability Policies

```prolog
% Policy options:
%   warn - Compile anyway, print warnings
%   forbid - Fail compilation if tool missing
%   ignore - Don't check tool availability
%   list_required - Compile, output required tools

% Set global policy
assertz(tool_availability_policy(warn)).

% Require specific tool
assertz(require_tool(jq)).

% Prefer available tools
assertz(prefer_available_tools(true)).
```

---

## Examples

### Example 1: Secure API Integration

```prolog
% Load security policy
load_firewall_policy(strict_security).

% Configure network access
assertz(network_access_policy(whitelist)).
assertz(allowed_domain('api.github.com')).
assertz(allowed_domain('api.internal.company.com')).

% Define HTTP source
:- source(http, github_data, [
    url('https://api.github.com/users/octocat/repos'),
    headers(['User-Agent: UnifyWeaver/0.0.2'])
]).

% Attempt to access allowed domain
?- check_url_access('https://api.github.com/users/octocat/repos', Result).
Result = allow.

% Attempt to access denied domain
?- check_url_access('https://untrusted.com/data', Result).
Result = deny(domain_not_in_whitelist).
```

### Example 2: Pure PowerShell Environment

```prolog
% Windows environment without WSL/bash
load_firewall_policy(pure_powershell).

% Compile CSV source
compile_to_powershell(users/3, [
    source_type(csv),
    csv_file('users.csv'),
    powershell_mode(auto)  % Will choose 'pure' due to firewall
], Code).

% Result: Pure PowerShell using Import-Csv
```

### Example 3: Data Processing Pipeline with Restrictions

```prolog
% Configure firewall
assertz(firewall_mode(permissive)).
assertz(network_access_policy(whitelist)).
assertz(allowed_domain('*.github.com')).
assertz(allowed_domain('*.typicode.com')).

% Python module restrictions
assertz(allowed_python_module(sys)).
assertz(allowed_python_module(json)).
assertz(allowed_python_module(sqlite3)).
assertz(denied_python_module(os)).
assertz(denied_python_module(subprocess)).

% File access restrictions
assertz(allowed_file_read('data/*')).
assertz(allowed_file_read('config/*')).
assertz(allowed_file_write('output/*')).
assertz(allowed_file_write('/tmp/*')).

% Define ETL pipeline
:- source(http, api_data, [
    url('https://api.github.com/users/octocat/repos')
]).

:- source(json, parse_data, [
    jq_filter('.[] | {name, stars: .stargazers_count}'),
    json_stdin(true)
]).

:- source(python, store_data, [
    python_inline('
import sqlite3
import sys
conn = sqlite3.connect("output/repos.db")
conn.execute("CREATE TABLE IF NOT EXISTS repos (name, stars)")
for line in sys.stdin:
    name, stars = line.strip().split("\t")
    conn.execute("INSERT INTO repos VALUES (?, ?)", (name, stars))
conn.commit()
'),
    timeout(30)
]).

% Run pipeline
etl_pipeline :-
    api_data | parse_data | store_data.
```

### Example 4: Development vs. Production Policies

**Development:**
```prolog
load_firewall_policy(permissive_dev).

% Warnings only, doesn't block
compile_to_bash(data/2, [
    source_type(csv),
    csv_file('data.csv')
], Code).
% [Warning] Missing tools: [jq]
% Compilation succeeds
```

**Production:**
```prolog
load_firewall_policy(strict_security).

% Fails if tools missing
compile_to_bash(data/2, [
    source_type(csv),
    csv_file('data.csv')
], Code).
% [Error] Cannot compile: missing required tools: [jq]
% Compilation fails
```

---

## Testing

Test firewall functionality:

```bash
# Run firewall tests
swipl -g main -t halt examples/test_firewall.pl

# Run network access control tests
swipl -g main -t halt examples/test_network_firewall.pl

# Run tool detection tests
swipl -g main -t halt examples/test_tool_detection.pl
```

---

## Best Practices

1. **Start Permissive** - Use permissive mode during development
2. **Tighten for Production** - Switch to strict mode for deployment
3. **Use Policy Templates** - Leverage pre-configured policies
4. **Layer Configuration** - Set defaults globally, override per-compilation
5. **Test Security Policies** - Verify firewall blocks unwanted operations
6. **Document Policies** - Comment why specific rules exist
7. **Review Warnings** - Pay attention to firewall warnings

---

## Related Documentation

- [EXTENDED_README.md](EXTENDED_README.md) - Comprehensive documentation
- [CONTROL_PLANE.md](CONTROL_PLANE.md) - Firewall architecture details
- [TOOL_AVAILABILITY_DESIGN.md](../planning/TOOL_AVAILABILITY_DESIGN.md) - Tool detection design

---

**Last Updated:** 2025-10-26
**Version:** 0.0.2
