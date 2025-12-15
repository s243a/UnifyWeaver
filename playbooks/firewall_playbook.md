# Playbook: Security Policy Enforcement (Firewall)

## Audience
This playbook is a high-level guide for coding agents. It demonstrates UnifyWeaver's firewall module for enforcing security policies on backend execution, service access, and resource usage.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "firewall" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use firewall"


## Workflow Overview
Use the firewall module for security enforcement:
1. Define security policies with rule_firewall/2 (execution backends, services, network/file access)
2. Set firewall mode (disabled, permissive, strict)
3. Validate operations before compilation using validate_* predicates

## Agent Inputs
Reference the following artifacts:
1. **Core Module** – `src/unifyweaver/core/firewall.pl` contains all security policy predicates
2. **Module Documentation** – See module header for API details

## Key Features

- Backend execution whitelisting
- Service access control
- Network and file access restrictions
- Python import restrictions
- Multiple enforcement modes (disabled/permissive/strict)

## Execution Guidance

Consult the module for predicate usage:

```prolog
:- use_module('src/unifyweaver/core/firewall').

% Define security policy
:- rule_firewall(my_pred/1, [
    execution([bash, python]),
    services([]),
    network(disabled),
    files(['/tmp/data.txt'])
]).

% Set enforcement mode
?- set_firewall_mode(strict).

% Validate operations
?- validate_service(my_pred/1, http_service).
?- validate_network_access(my_pred/1, 'api.example.com').
```

## Expected Outcome
- Security policies enforced before compilation
- Backend and service access validated
- Network and file operations controlled
- Violations logged or rejected based on mode

## Citations
[1] src/unifyweaver/core/firewall.pl

