# Playbook: Security Policy Enforcement (Firewall)

## Audience
This playbook demonstrates UnifyWeaver's firewall module for enforcing security policies on backend and service usage.

## Overview
The `firewall` module enforces security boundaries before compilation:
- Backend execution whitelists
- Service access control
- Network/file access restrictions
- Python import restrictions
- Cache directory validation

## When to Use

✅ **Use firewall when:**
- Need security policy enforcement
- Restricting backend/service access
- Validating file and network operations
- Building secure compilation pipelines
- Need audit trails for security

## Example Usage

### Define Firewall Rules

```prolog
:- use_module('src/unifyweaver/core/firewall').

% Restrict which backends can execute this predicate
:- rule_firewall(my_pred/1, [
    execution([bash, python]),  % Only bash and python allowed
    services([]),                % No external services
    network(disabled)            % No network access
]).
```

### Set Firewall Mode

```prolog
% Set firewall to strict mode
?- set_firewall_mode(strict).

% Check current mode
?- current_firewall_mode(Mode).
Mode = strict.
```

### Validate Operations

```prolog
% Validate service access
?- validate_service(my_pred/1, http_service).
false.  % Denied - not in whitelist

% Validate file access
?- validate_file_access(my_pred/1, '/tmp/data.txt', read).
true.  % Allowed

% Validate network access
?- validate_network_access(my_pred/1, 'api.example.com').
false.  % Denied - network disabled
```

## Firewall Modes

- **disabled**: No enforcement (default for development)
- **permissive**: Log violations but allow
- **strict**: Enforce all policies, reject violations

## Policy Terms

- `execution([Backends])` - Allowed backends
- `services([Services])` - Allowed external services
- `network(enabled|disabled)` - Network access
- `files([Paths])` - File access whitelist
- `python_imports([Modules])` - Python import whitelist

## See Also

- `playbooks/deployment_glue_playbook.md` - Deployment security
- `playbooks/network_glue_playbook.md` - Network services

## Summary

**Key Concepts:**
- ✅ Security policy enforcement
- ✅ Backend/service whitelisting
- ✅ Network/file access control
- ✅ Multiple firewall modes
- ✅ Pre-compilation validation
