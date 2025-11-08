# Tool Availability vs. Firewall: Design Philosophy

**Created:** 2025-10-26
**Status:** Design Specification
**Priority:** CRITICAL - Core architectural decision

---

## Core Distinction

These are **separate but complementary** systems that work together:

### Firewall = Policy Decision
**"What SHOULD be used (security/preference)"**

- Enforces security policies
- Encodes organizational rules
- Expresses user preferences
- **Examples:**
  - "PowerShell MUST NOT use bash" (security policy)
  - "No network access to external sites" (security policy)
  - "Prefer pure PowerShell over BaaS" (preference)

### Tool Availability = Practical Decision
**"What CAN be used (reality check)"**

- Detects what tools exist on the system
- Determines what to do when tools are missing
- Provides fallback strategies
- **Examples:**
  - "bash is not installed on this Windows machine" (fact)
  - "jq is available at /usr/bin/jq" (fact)
  - "If tool missing, warn but continue" (response strategy)

---

## Key Principle

**Firewall policies are INDEPENDENT of tool availability.**

A firewall rule like `denied_service(powershell, executable(bash))` means:
- "Bash MUST NOT be used for PowerShell compilation"
- This is true **regardless** of whether bash is installed
- It's a **policy decision**, not a **capability decision**

Tool availability detection can **inform** firewall decisions, but they remain separate concerns.

---

## Tool Availability Policies

These control **how the system responds** when required tools are missing:

### 1. `tool_availability_policy(warn)` - DEFAULT
**Behavior:** Compile anyway, print warnings

```prolog
assertz(tool_availability_policy(warn)).
compile_to_bash(my_pred/2, [source_type(json)], Code).

% Output:
% [Warning] Missing tools: [jq]
% [Warning] Generated script may fail at runtime

% Result: Compilation succeeds
% Runtime: May fail when script runs
```

**Use Case:** Development mode - want to see the generated code even if tools missing

### 2. `tool_availability_policy(forbid)` - STRICT
**Behavior:** Fail compilation if tools missing

```prolog
assertz(tool_availability_policy(forbid)).
assertz(require_tool(jq)).
compile_to_bash(my_pred/2, [source_type(json)], Code).

% Output:
% [Error] Cannot compile: missing required tools: [jq]
% [Error] Install with: sudo apt-get install jq

% Result: Compilation fails
```

**Use Case:** CI/CD pipeline - ensure all dependencies present before deployment

### 3. `tool_availability_policy(list_required)` - DOCUMENTATION
**Behavior:** Compile and document required tools in output

```prolog
assertz(tool_availability_policy(list_required)).
compile_to_bash(my_pred/2, [source_type(json)], Code).

% Generated bash includes:
#!/bin/bash
# Required tools:
#   - jq (JSON processor) - sudo apt-get install jq
#   - curl (HTTP client) - sudo apt-get install curl
#
# This script will fail if these tools are not installed.

% Result: Compilation succeeds with documentation
```

**Use Case:** Sharing scripts - recipient knows what to install

### 4. `tool_availability_policy(ignore)` - PERMISSIVE
**Behavior:** Don't check tool availability

```prolog
assertz(tool_availability_policy(ignore)).
compile_to_bash(my_pred/2, [source_type(json)], Code).

% No tool checking performed
% Result: Compilation succeeds (may fail at runtime)
```

**Use Case:** Trust environment is correct, skip checks for speed

---

## How They Work Together

### Decision Flow

```
1. Check Firewall Policy
   ↓
   Is this tool/service ALLOWED by firewall?
   ↓
   ├─ No  → DENY (firewall blocks it)
   ├─ Yes → Continue to step 2
   └─ No explicit rule → Continue to step 2

2. Check Tool Availability
   ↓
   Is this tool AVAILABLE on the system?
   ↓
   ├─ Yes → Use it
   └─ No  → Apply tool_availability_policy:
            ├─ warn → Warn, continue
            ├─ forbid → Fail compilation
            ├─ list_required → Document, continue
            └─ ignore → Continue without check

3. Select Implementation
   ↓
   Based on firewall + availability, choose:
   - Original approach (if allowed and available)
   - Alternative approach (if prefer_available_tools)
   - Fail (if forbidden or required but missing)
```

### Example Scenario Matrix

| Firewall Rule | Bash Available? | Tool Policy | Result |
|---------------|-----------------|-------------|---------|
| `denied(bash)` | Yes | (any) | Use pure PowerShell (firewall blocks bash) |
| `denied(bash)` | No | (any) | Use pure PowerShell (firewall blocks bash) |
| No rule | Yes | warn | Use BaaS (bash available and allowed) |
| No rule | No | warn | Use pure PowerShell + warn |
| No rule | No | forbid | Fail compilation |
| No rule | No | list_required | Use pure PowerShell + document |
| No rule | No | ignore | Use pure PowerShell (adapted) |

---

## Implementation Architecture

### Layer 1: Firewall Check
```prolog
% Check if service is allowed by firewall
check_firewall_service(powershell, executable(bash), Result) :-
    (   denied_service(powershell, executable(bash))
    ->  Result = deny(firewall_policy)
    ;   allowed_service(powershell, executable(bash))
    ->  Result = allow
    ;   firewall_mode(strict)
    ->  Result = deny(not_explicitly_allowed)
    ;   Result = allow  % Permissive mode
    ).
```

### Layer 2: Tool Detection
```prolog
% Detect if tool is available
detect_tool_availability(bash, Status) :-
    (   check_executable_exists('bash')
    ->  Status = available
    ;   Status = unavailable(not_found)
    ).
```

### Layer 3: Policy Application
```prolog
% Combine firewall + availability + policy
select_powershell_mode(SourceType, Options, SelectedMode) :-
    % Step 1: Check firewall
    check_firewall_service(powershell, executable(bash), FirewallResult),

    % Step 2: Check availability
    detect_tool_availability(bash, AvailStatus),

    % Step 3: Get tool policy
    (   member(tool_availability_policy(Policy), Options)
    ->  true
    ;   tool_availability_policy(Policy)
    ->  true
    ;   Policy = warn  % Default
    ),

    % Step 4: Decide mode
    (   FirewallResult = deny(_)
    ->  % Firewall blocks bash
        SelectedMode = pure,
        format('[Firewall] Bash denied for PowerShell - using pure mode~n', [])

    ;   AvailStatus = unavailable(_),
        Policy = forbid
    ->  % Bash required but missing
        format('[Error] Bash required but not available~n', []),
        fail

    ;   AvailStatus = unavailable(_),
        supports_pure_powershell(SourceType)
    ->  % Bash missing, but we can use pure
        SelectedMode = pure,
        apply_unavailable_policy(Policy, bash)

    ;   AvailStatus = available
    ->  % Bash available and allowed
        SelectedMode = baas

    ;   % Fallback
        SelectedMode = pure
    ).

% Apply tool availability policy for unavailable tools
apply_unavailable_policy(warn, Tool) :-
    format('[Warning] ~w not available - using alternative~n', [Tool]).

apply_unavailable_policy(list_required, Tool) :-
    assertz(required_tool_missing(Tool)).

apply_unavailable_policy(ignore, _Tool).
```

---

## Common Patterns

### Pattern 1: Security-First (Firewall Dominates)
```prolog
% Firewall rule: No bash for PowerShell (security policy)
assertz(denied_service(powershell, executable(bash))).

% Tool policy: Just warn (we'll adapt)
assertz(tool_availability_policy(warn)).

% Result: Always pure PowerShell (firewall decides)
```

### Pattern 2: Pragmatic Adaptation (Availability Influences Choice)
```prolog
% No firewall restrictions
% Tool policy: Prefer available tools
assertz(tool_availability_policy(warn)).
assertz(prefer_available_tools(true)).

% Result:
% - If bash available → use BaaS
% - If bash missing → use pure PowerShell (with warning)
```

### Pattern 3: Strict Validation (Fail Fast)
```prolog
% No firewall restrictions
% Tool policy: All tools must be present
assertz(tool_availability_policy(forbid)).
assertz(require_tool(bash)).
assertz(require_tool(awk)).

% Result:
% - If tools present → compile successfully
% - If tools missing → fail with clear error
```

### Pattern 4: Documentation Mode (List Everything)
```prolog
% No firewall restrictions
% Tool policy: Document requirements
assertz(tool_availability_policy(list_required)).

% Result: Compiles successfully
% Generated script header lists all required tools
```

---

## Firewall Can Inform Itself from Tool Detection

The firewall can automatically create rules based on detected tool availability:

```prolog
% Detect bash availability and update firewall accordingly
update_firewall_from_environment :-
    detect_tool_availability(bash, Status),
    (   Status = unavailable(_)
    ->  % Bash not available - add firewall rule
        assertz(denied_service(powershell, executable(bash))),
        format('[Firewall] Bash unavailable - denying bash for PowerShell~n', [])
    ;   % Bash available - no firewall rule needed
        true
    ).
```

**BUT:** This is an **optional** convenience. The firewall rule is still a separate policy decision, just informed by reality.

---

## Multi-Level Configuration

All three levels can specify both firewall rules AND tool policies:

### Level 1: Global Defaults
```prolog
% Global firewall mode
:- assertz(firewall_mode(permissive)).

% Global tool policy
:- assertz(tool_availability_policy(warn)).
:- assertz(prefer_available_tools(true)).
```

### Level 2: Firewall Policy
```prolog
load_firewall_policy(strict_security) :-
    % Firewall rules
    set_firewall_mode(strict),
    assertz(denied_service(_, network_access(_))),

    % Tool policy
    assertz(tool_availability_policy(forbid)),
    assertz(require_tool(bash)),
    assertz(require_tool(awk)).
```

### Level 3: Compilation Options
```prolog
compile_to_powershell(my_pred/2, [
    source_type(csv),

    % Firewall override
    allow_service(powershell, executable(bash)),

    % Tool policy override
    tool_availability_policy(list_required),
    require_tool(bash)
], Code).
```

**Priority:** Compilation Options > Firewall Policy > Global Defaults

---

## Critical Design Decisions

### Decision 1: Firewall Blocks Take Precedence
**If firewall denies something, tool availability doesn't matter.**

```prolog
% Even if bash is available...
detect_tool_availability(bash, available).

% ...firewall can still block it
assertz(denied_service(powershell, executable(bash))).

% Result: Use pure PowerShell (firewall decision)
```

**Rationale:** Security policies must override convenience.

### Decision 2: Tool Policies Are Response Strategies
**Tool availability policies control RESPONSE, not PERMISSION.**

```prolog
% Tool policy controls what happens when tool is missing
tool_availability_policy(warn).   % Response: warn
tool_availability_policy(forbid). % Response: fail
tool_availability_policy(list_required). % Response: document

% NOT:
% tool_availability_policy(deny). % Wrong! Use firewall for this
```

**Rationale:** Clear separation of concerns - firewall = policy, tool policy = response.

### Decision 3: Prefer Available Tools Is a Strategy
**`prefer_available_tools(true)` means "use what's available when possible"**

```prolog
assertz(prefer_available_tools(true)).

% Effect:
% - If bash available → prefer BaaS (faster)
% - If bash missing → use pure PowerShell (adapt)
% - If both available → choose based on other criteria
```

**Rationale:** Pragmatic adaptation to environment.

### Decision 4: Auto Mode Respects All Layers
**`powershell_mode(auto)` considers firewall + availability + preferences**

```prolog
compile_to_powershell(pred/2, [
    powershell_mode(auto),
    source_type(csv)
], Code).

% Auto mode decision tree:
% 1. Check firewall - bash denied? → pure
% 2. Check availability - bash missing? → pure (with policy response)
% 3. Check preferences - prefer pure? → pure
% 4. Default for CSV → pure (performance similar)
```

---

## Testing Strategy

### Test 1: Firewall Blocks Despite Availability
```prolog
% Setup: bash is available
mock_tool_available(bash).

% Firewall blocks it
assertz(denied_service(powershell, executable(bash))).

% Compile
compile_to_powershell(users/3, [powershell_mode(auto), source_type(csv)], Code).

% Assert: Uses pure PowerShell
assert_contains(Code, 'Import-Csv').
\+ assert_contains(Code, 'bash').
```

### Test 2: Tool Missing, Warn Policy
```prolog
% Setup: bash not available
mock_tool_unavailable(bash).

% Tool policy: warn
assertz(tool_availability_policy(warn)).

% Compile
compile_to_powershell(users/3, [powershell_mode(auto), source_type(csv)], Code).

% Assert: Uses pure PowerShell with warning
assert_warning_issued('bash not available').
assert_contains(Code, 'Import-Csv').
```

### Test 3: Tool Missing, Forbid Policy
```prolog
% Setup: bash not available
mock_tool_unavailable(bash).

% Tool policy: forbid
assertz(tool_availability_policy(forbid)).
assertz(require_tool(bash)).

% Compile
\+ compile_to_powershell(users/3, [powershell_mode(baas), source_type(csv)], Code).

% Assert: Compilation fails
assert_error_issued('required tool missing: bash').
```

### Test 4: Tool Missing, List Policy
```prolog
% Setup: bash not available
mock_tool_unavailable(bash).

% Tool policy: list
assertz(tool_availability_policy(list_required)).

% Compile
compile_to_powershell(users/3, [powershell_mode(auto), source_type(csv)], Code).

% Assert: Pure PowerShell with documentation
assert_contains(Code, '# Required tools:').
assert_contains(Code, '#   - bash').
assert_contains(Code, 'Import-Csv').
```

---

## Implementation Checklist

- [ ] Document tool_availability_policy/1 predicate options
- [ ] Implement detect_tool_availability/2 (DONE in tool_detection.pl)
- [ ] Implement check_firewall_service/3
- [ ] Implement select_powershell_mode/3 with layered decision logic
- [ ] Implement apply_unavailable_policy/2
- [ ] Add tool policy support to compilation options
- [ ] Update firewall policies to include tool policies
- [ ] Add tests for all four tool policies
- [ ] Add tests for firewall + availability interaction
- [ ] Document in FIREWALL_GUIDE.md (DONE)
- [ ] Update EXTENDED_README.md with examples

---

## Summary

**Two separate but complementary systems:**

1. **Firewall** = "What SHOULD happen" (policy)
   - Security rules
   - Organizational policies
   - User preferences

2. **Tool Availability** = "What CAN happen" (reality)
   - Tool detection
   - Response strategies (warn/forbid/list/ignore)
   - Fallback mechanisms

**They work together:**
- Firewall checks permission
- Tool detection checks capability
- Tool policy controls response to missing tools
- System selects best implementation given all constraints

**Key Principle:** Firewall policies are independent of tool availability. A firewall rule means "don't use X" regardless of whether X exists.

---

**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
**License:** MIT OR Apache-2.0
