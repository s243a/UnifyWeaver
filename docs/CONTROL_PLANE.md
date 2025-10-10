# Control Plane: Firewall and Preference System

## Overview

The Control Plane is a new architectural layer in UnifyWeaver designed to provide declarative control over how the compiler evaluates and generates code for Prolog predicates. It introduces a strict separation between **security policies (Firewall)** and **implementation choices (Preferences)**.

This system allows fine-grained control over which backends and services can be used, and in what order, ensuring both security and flexibility.

## 1. Philosophy

### Separation of Policy and Choice

-   **Firewall (Security Policy):** This is a mandatory, security-focused layer that defines what backends and services are *allowed* or *denied*. It acts as a safety net, enforcing hard boundaries. Firewall rules should be changed infrequently and with careful review.
-   **Preferences (Implementation Choice):** This is a flexible, layered system that defines the *desired* implementation strategy from among the options permitted by the firewall. Developers can change preferences freely to optimize for different scenarios (e.g., speed vs. memory, local debugging vs. production).

### Layered Configuration

Settings are applied in a three-tiered hierarchy, with more specific layers overriding more general ones. The final choice at each step must always be validated against the firewall.

1.  **Runtime Options:** Options passed directly to a compilation predicate (highest priority).
2.  **Rule-Specific Declarations:** Per-predicate settings that override global defaults.
3.  **Global Defaults:** Project-wide settings for preferences and firewall rules (lowest priority).

### Pluggable Backends

The system distinguishes between:
-   **Execution Backends:** Primary environments where generated code runs (e.g., `bash`, `python`).
-   **Service Backends:** External tools or APIs that can be called *from* an execution backend (e.g., `sql`, `vector_db`, `llm`).

### Default to Safety

If no explicit firewall policy is defined for a predicate, the system defaults to a safe, symbolic computation mode, ensuring that no unintended external services are invoked.

## 2. The Firewall System

The firewall enforces hard security and policy boundaries. It is defined using `rule_firewall/2` for predicate-specific rules and `firewall_default/1` for global defaults.

### API Reference

#### `firewall:rule_firewall(+PredicateIndicator, +PolicyTerms:list)`

Declares predicate-specific firewall policies.

**Policy Terms:**
-   `execution([backend1, backend2, ...])`: Whitelist of allowed primary execution backends (e.g., `bash`, `python`).
-   `services([service1, service2, ...])`: Whitelist of allowed services that can be called from an execution backend (e.g., `sql`, `vector_db`, `llm`).
-   `denied([backend1, service1, ...])`: Blacklist of backends or services that are explicitly forbidden. This always takes precedence.
-   `max_cost(low | medium | high)`: An abstract cost limit to prevent expensive operations.

#### `firewall:firewall_default(+PolicyTerms:list)`

Declares global default firewall policies. If no `rule_firewall` is found for a predicate, `firewall_default` is used. If `firewall_default` is also not defined, the system defaults to implicit allow.

### Integration

`firewall:validate_against_firewall(+Target, +Options, +Firewall)` is called by the main compiler dispatcher to enforce policies.

## 3. The Preference System

The preference system guides the compiler on which implementation to choose from the options permitted by the firewall. It is defined using `rule_preferences/2` for predicate-specific settings and `preferences_default/1` for global defaults.

### API Reference

#### `preferences:rule_preferences(+PredicateIndicator, +PreferenceTerms:list)`

Declares predicate-specific preference settings.

**Preference Terms:**
-   `prefer([backend1, service1, ...])`: The desired order of implementation choice.
-   `fallback_order([backend2, service2, ...])`: The order to try if preferred implementations fail.
-   `optimization(speed | memory | balance)`: A hint for the compiler to choose the most appropriate template or strategy.
-   `service_mode(embedded | remote)`: For services like databases, whether to prefer a local file/process or a remote server.

#### `preferences:preferences_default(+PreferenceTerms:list)`

Declares global default preference settings for the entire project.

### Integration

`preferences:get_final_options(+Rule, +RuntimeOptions, -FinalOptions)` is called by the main compiler dispatcher to merge options from all layers.

## 4. Integration with Compilation Pipeline

The main compilation dispatcher (`recursive_compiler.pl`) orchestrates the control plane logic. The conceptual flow is:

1.  **Get Preferences:** `preferences:get_final_options/3` retrieves and merges preferences.
2.  **Get Firewall:** `firewall:rule_firewall/2` (or `firewall_default/1`) retrieves the policy.
3.  **Validate Request:** `firewall:validate_against_firewall/3` validates the request against the policy. If it fails, compilation halts.
4.  **Compile:** If validation passes, the compilation proceeds with the existing logic, using the merged `FinalOptions`.

## 5. Complete Working Examples

```prolog
% 1. Define firewall rules (in config/firewall.pl or asserted at runtime)
:- firewall:rule_firewall(sensitive_pred/2, [denied([llm])]).
:- firewall:rule_firewall(db_access_pred/2, [execution([bash]), services([sql])]).

% 2. Define preferences (in config/preferences.pl or asserted at runtime)
:- preferences:rule_preferences(db_access_pred/2, [prefer([bash]), service_mode(embedded)]).
:- preferences:preferences_default([prefer([bash]), optimization(balance)]).

% 3. Compile with runtime override (in your Prolog session)
% This should succeed, using bash and embedded SQL
?- compile_recursive(db_access_pred/2, [], BashCode).

% This should fail due to firewall policy
?- compile_recursive(sensitive_pred/2, [use_services([llm])], BashCode).
% Output: Firewall Error: Service 'llm' is explicitly denied.

% This should succeed, overriding preferences to use Python (if allowed by firewall)
?- compile_recursive(db_access_pred/2, [prefer([python])], BashCode).
```

## 6. Configuration Files

Configuration files for project-wide firewall rules and preferences are located in `src/unifyweaver/config/`:

-   `firewall.pl`: For global firewall policies.
-   `preferences.pl`: For global preference settings.

## 7. Testing

The control plane's functionality is verified by `tests/core/test_policy.pl`. This test suite covers firewall validation, preference merging, and ensures correct behavior for allowed, denied, and overridden scenarios.

## 8. Future Enhancements

-   Integration with other execution backends (e.g., Python, C#).
-   More sophisticated policy terms (e.g., `max_cost`, `data_hygiene`).
-   Dynamic selection of backends based on preferences and runtime context.
