# Firewall Implementation TODO

**Created:** 2025-10-26
**Status:** Planning Phase
**Related:** POST_RELEASE_TODO.md #13, #15

---

## Overview

This document tracks firewall-related work items, combining post-release fixes with pure PowerShell integration. The firewall system provides security and preference guidance for compilation decisions.

## Integration with Pure PowerShell

### New Requirement: Firewall-Driven Mode Selection

With pure PowerShell implementation complete, the firewall should help select between pure PowerShell and BaaS modes based on **fundamental security policies** about target languages and services.

**Firewall Architecture - Fundamental Rules:**

The firewall operates on fundamental policies, not derived predicates:

1. **Target Language Policies** - What languages can be used
   - `allowed_target_language(bash)`
   - `allowed_target_language(powershell)`
   - `denied_target_language(python)` (example)

2. **Service/Tool Policies** - What services each language can use
   - `allowed_service(bash, executable(awk))`
   - `allowed_service(powershell, cmdlet(import_csv))`
   - `denied_service(powershell, executable(bash))` (pure PowerShell requirement)

3. **Derived Preferences** - Higher-level predicates computed from fundamental rules
   - `powershell_mode(Mode)` is derived from language and service policies
   - NOT stored directly in firewall

**Example Derivation:**
```prolog
% Derived rule: PowerShell mode based on fundamental policies
powershell_mode_preference(pure) :-
    % Pure mode required if bash service is denied for PowerShell
    denied_service(powershell, executable(bash)),
    % And target language PowerShell is allowed
    allowed_target_language(powershell).

powershell_mode_preference(baas) :-
    % BaaS allowed if bash service is allowed for PowerShell
    allowed_service(powershell, executable(bash)),
    allowed_target_language(powershell).

powershell_mode_preference(auto) :-
    % Auto mode when no explicit service restrictions
    \+ denied_service(powershell, executable(bash)),
    allowed_target_language(powershell).
```

**Example Use Cases:**
1. **Restricted Environment:**
   - Firewall: `denied_service(powershell, executable(bash))`
   - Derived: `powershell_mode(pure)` for CSV/JSON/HTTP

2. **Enterprise Security:**
   - Firewall: `preferred_service(powershell, cmdlet(_))` over `executable(_)`
   - Derived: Suggest pure PowerShell but allow BaaS with warning

3. **Performance Priority:**
   - Firewall: `allowed_service(powershell, executable(bash))`
   - Derived: `powershell_mode(baas)` allowed for AWK sources

---

## Todo Items

### 1. Firewall Philosophy Decision (POST_RELEASE_TODO #13)

**Status:** 📋 DESIGN DECISION NEEDED
**Priority:** HIGH (affects all other firewall work)
**Estimated Effort:** 2-3 hours design + 4-6 hours implementation

**Issue:**
Current firewall has inconsistent behavior - sometimes fails silently, tests expect exceptions. Need to clarify blocking vs guidance philosophy.

**Recommended Approach: Hybrid (3-Level)**

```prolog
% Three-level firewall responses
firewall_check(Action, Result) :-
    (   allowed(Action)          -> Result = allow
    ;   preferred_fallback(Action) -> Result = warn(Reason)
    ;   denied(Action)            -> Result = deny(Reason)
    ;   unknown_action(Action)   -> Result = mode_dependent
    ).
```

**Implementation Checklist:**
- [ ] Add `firewall_mode(strict|permissive|disabled)` configuration
- [ ] Define `allowed/1`, `preferred_fallback/1`, `denied/1` predicates
- [ ] Implement warning system (print warnings without failing)
- [ ] Update `check_firewall/1` to return structured results
- [ ] Create policy templates:
  - [ ] `firewall_strict.pl` - Enterprise security (deny by default)
  - [ ] `firewall_permissive.pl` - Development (allow by default)
  - [ ] `firewall_pure_powershell.pl` - Pure PowerShell preference
- [ ] Update tests to match new behavior
- [ ] Document in `docs/FIREWALL_GUIDE.md`

**Example Policy (Pure PowerShell Preference):**
```prolog
% firewall_pure_powershell.pl - Fundamental rules only
firewall_mode(permissive).

% Target language policies
allowed_target_language(bash).
allowed_target_language(powershell).

% Service policies for PowerShell
denied_service(powershell, executable(bash)).  % Pure PowerShell only
allowed_service(powershell, cmdlet(import_csv)).
allowed_service(powershell, cmdlet(convertfrom_json)).
allowed_service(powershell, cmdlet(invoke_restmethod)).

% Service policies for bash
allowed_service(bash, executable(awk)).
allowed_service(bash, executable(sed)).
allowed_service(bash, executable(grep)).

% Network policy (example)
denied_service(_, network_access(external)).
```

**Derived Rules (in compiler, not firewall):**
```prolog
% Compiler derives PowerShell mode from fundamental firewall rules
select_powershell_mode(TargetLanguage, SourceType, Mode) :-
    TargetLanguage = powershell,
    (   % If bash service denied for PowerShell → pure mode required
        denied_service(powershell, executable(bash)),
        supports_pure_powershell(SourceType)
    ->  Mode = pure
    ;   % If bash service allowed → can use BaaS
        allowed_service(powershell, executable(bash))
    ->  Mode = baas
    ;   % Default: auto (prefer pure if available)
        Mode = auto
    ).
```

---

### 2. Integrate Firewall with PowerShell Mode Selection

**Status:** 🆕 NEW (from pure PowerShell implementation)
**Priority:** MEDIUM
**Estimated Effort:** 3-4 hours
**Depends On:** Item #1 (philosophy decision)

**Goal:**
Firewall should influence `powershell_mode(auto)` selection based on security policies.

**Current Behavior:**
```prolog
% Auto mode always prefers pure for CSV/JSON/HTTP
compile_to_powershell(Pred, [powershell_mode(auto), source_type(csv)], Code) :-
    % Always uses pure PowerShell for CSV
    ...
```

**Desired Behavior:**
```prolog
% Auto mode consults firewall
compile_to_powershell(Pred, [powershell_mode(auto), source_type(csv)], Code) :-
    check_firewall(powershell_mode(pure), FirewallResult),
    select_mode_with_firewall(auto, csv, FirewallResult, SelectedMode),
    ...
```

**Implementation Checklist:**
- [ ] Add `select_mode_with_firewall/4` predicate
- [ ] Query firewall for `powershell_mode(pure)` preference
- [ ] Handle firewall responses:
  - `allow` → use pure PowerShell
  - `warn(Reason)` → use pure but warn user
  - `deny(Reason)` → fall back to BaaS
  - `mode_dependent` → use default auto behavior
- [ ] Add firewall query to `compile_to_powershell/3`
- [ ] Test with different firewall policies
- [ ] Document firewall-driven mode selection

**Example:**
```prolog
select_mode_with_firewall(auto, SourceType, FirewallResult, SelectedMode) :-
    supports_pure_powershell(SourceType),
    check_firewall(powershell_mode(pure), FirewallResult),
    (   FirewallResult = allow
    ->  SelectedMode = pure
    ;   FirewallResult = warn(Reason)
    ->  format('[Firewall Warning] ~w~n', [Reason]),
        SelectedMode = pure
    ;   FirewallResult = deny(Reason)
    ->  format('[Firewall] ~w - using BaaS~n', [Reason]),
        SelectedMode = baas
    ;   % Default auto behavior
        SelectedMode = pure
    ).
```

---

### 3. Firewall Detection of Bash Availability

**Status:** 🆕 NEW
**Priority:** MEDIUM
**Estimated Effort:** 2-3 hours

**Goal:**
Firewall should detect if bash is available and adjust recommendations.

**Use Case:**
- Restricted Windows environment: No WSL, no Cygwin, no Git Bash
- Firewall detects this → forces `powershell_mode(pure)` for all sources
- If user tries to use AWK source → error with helpful message

**Implementation Checklist:**
- [ ] Add `detect_bash_availability/0` predicate
- [ ] Check for WSL (`wsl.exe --version`)
- [ ] Check for Cygwin (`C:\cygwin64\bin\bash.exe`)
- [ ] Check for Git Bash (`C:\Program Files\Git\bin\bash.exe`)
- [ ] Add firewall rule: `no_bash_available → denied(powershell_mode(baas))`
- [ ] Test in restricted environment (VM without bash)
- [ ] Provide helpful error messages:
  ```
  [Firewall Error] Bash not available - cannot use BaaS mode
  Suggestion: Use pure PowerShell for CSV/JSON/HTTP sources, or install WSL
  ```

---

### 4. Firewall Policy for Network Access

**Status:** 📋 IDENTIFIED (POST_RELEASE_TODO mentions this)
**Priority:** LOW (future enhancement)
**Estimated Effort:** 2-3 hours

**Goal:**
Control HTTP source access based on security policy.

**Example Policy:**
```prolog
% Allow internal network only
denied(source_type(http), url(URL)) :-
    \+ sub_atom(URL, _, _, _, 'internal.company.com').

% Block specific domains
denied(source_type(http), url(URL)) :-
    member(BlockedDomain, ['badsite.com', 'malware.net']),
    sub_atom(URL, _, _, _, BlockedDomain).
```

**Implementation:**
- [ ] Parse URL in HTTP source validation
- [ ] Query firewall with `check_firewall(http_access(URL), Result)`
- [ ] Deny compilation if firewall blocks URL
- [ ] Support URL whitelist/blacklist patterns

---

### 5. Higher-Order Firewall Policies with Overridable Defaults (POST_RELEASE_TODO #15)

**Status:** 📋 DESIGN PROPOSAL
**Priority:** MEDIUM (important for flexibility)
**Estimated Effort:** 6-8 hours

**Goal:**
Derive firewall policies from fundamental rules using Prolog inference, with ability to override default implications.

**Architecture: Two-Level Implication System**

1. **Default Implications** - Built-in derivations (overridable)
2. **User Implications** - User-defined derivations (override defaults)

**Implementation Pattern - Overridable Defaults:**

```prolog
% Default implication: no bash available → require pure PowerShell
firewall_implies(Condition, Consequence) :-
    firewall_implies_default(Condition, Consequence),
    \+ firewall_implies_disabled(Condition, Consequence).

% Built-in default implications
firewall_implies_default(no_bash_available,
                        denied_service(powershell, executable(bash))).

firewall_implies_default(denied_target_language(bash),
                        denied_service(powershell, executable(bash))).

firewall_implies_default(network_access(denied),
                        denied_service(_, network_access(_))).

firewall_implies_default(denied_executable(Tool),
                        denied_service(_, executable(Tool))).

% Users can disable default implications by asserting:
% :- assert(firewall_implies_disabled(no_bash_available,
%                                     denied_service(powershell, executable(bash)))).
```

**User-Defined Implications (override defaults):**

```prolog
% User adds custom implication
firewall_implies(corporate_policy(strict_security),
                denied_service(_, network_access(external))) :-
    \+ firewall_implies_disabled(corporate_policy(strict_security),
                                 denied_service(_, network_access(external))).

% Or user can completely replace by disabling default and adding their own
:- assert(firewall_implies_disabled(no_bash_available,
                                    denied_service(powershell, executable(bash)))).

% Custom rule: allow BaaS even without bash (use WSL fallback)
firewall_implies(no_bash_available,
                allowed_service(powershell, executable(wsl))) :-
    wsl_available.
```

**Benefits:**

1. **Default Safety** - Sensible defaults for common scenarios
2. **User Override** - Can disable defaults that don't apply
3. **Composable** - Stack implications to derive complex policies
4. **Prolog Showcase** - Demonstrates logical inference elegantly

**Example: Deriving PowerShell Mode**

```prolog
% Fundamental firewall fact
denied_service(powershell, executable(bash)).

% Default implication (if not disabled)
firewall_implies(denied_service(TargetLang, Service),
                require_alternative(TargetLang, Service)) :-
    firewall_implies_default(denied_service(TargetLang, Service),
                            require_alternative(TargetLang, Service)),
    \+ firewall_implies_disabled(denied_service(TargetLang, Service),
                                 require_alternative(TargetLang, Service)).

% Compiler queries derived policy
select_powershell_mode(powershell, csv, Mode) :-
    (   firewall_implies(denied_service(powershell, executable(bash)), _)
    ->  Mode = pure  % Must use pure PowerShell
    ;   Mode = auto  % Can choose
    ).
```

**This is a showcase feature** - demonstrates Prolog's logical inference in a way that's difficult in imperative languages, while remaining practical and user-controllable.

**Implement in Phase 2** (elevated priority due to practical benefits)

---

### 6. Fix Firewall Singleton Warnings

**Status:** ⚠️ WARNINGS (POST_RELEASE_TODO #13 mentions these)
**Priority:** LOW (cosmetic)
**Estimated Effort:** 15 minutes

**Warnings:**
```
firewall.pl:198 - Singleton: [P,Ps]
firewall.pl:223 - Singleton: [M,Ms]
firewall.pl:268 - Singleton: [D,Ds]
```

**Fix:**
Prefix with underscore or use in predicate logic.

---

## Implementation Priority

### Phase 1 (v0.0.3 - Post-Release)
1. ✅ Firewall Philosophy Decision (#1) - DESIGN COMPLETE
2. Integrate with PowerShell Mode Selection (#2)
3. Detect Bash Availability (#3)

### Phase 2 (v0.0.4)
4. Network Access Control (#4)
5. Fix Singleton Warnings (#6)

### Phase 3 (v0.1.0)
6. Higher-Order Firewall Policies (#5) - Showcase feature

---

## Testing Strategy

### Test Scenarios

1. **Pure PowerShell Forced:**
   ```prolog
   firewall_policy(pure_powershell_only).
   compile_to_powershell(user/2, [source_type(csv)], Code).
   % Should generate pure PowerShell
   ```

2. **BaaS Blocked:**
   ```prolog
   firewall_policy(no_bash).
   compile_to_powershell(log/3, [source_type(awk)], Code).
   % Should fail with helpful error
   ```

3. **Network Blocked:**
   ```prolog
   firewall_policy(no_network).
   compile_to_powershell(api/2, [source_type(http), url('...')], Code).
   % Should fail if URL not whitelisted
   ```

4. **Permissive Mode:**
   ```prolog
   firewall_mode(permissive).
   compile_to_powershell(anything/2, [...], Code).
   % Should allow with warnings
   ```

---

## Documentation Updates Needed

1. **`docs/FIREWALL_GUIDE.md`** - Update with:
   - Hybrid philosophy (allow/warn/deny)
   - PowerShell mode integration
   - Policy templates
   - Examples

2. **`docs/POWERSHELL_PURE_IMPLEMENTATION.md`** - Add:
   - Firewall-driven mode selection
   - How firewall affects pure vs BaaS choice

3. **`README.md`** - Mention:
   - Firewall as a unique Prolog advantage
   - Security and preference guidance

---

## Open Questions

1. **Default Firewall Mode:**
   - Should default be `permissive` (developer-friendly) or `strict` (security-first)?
   - **Recommendation:** `permissive` for v0.0.3, allow users to opt into `strict`

2. **Firewall Configuration:**
   - Environment variable (`UNIFYWEAVER_FIREWALL_MODE=strict`)?
   - Config file (`.unifyweaver/firewall.pl`)?
   - Both?
   - **Recommendation:** Both, env var overrides file

3. **Warning Display:**
   - Print to stderr?
   - Collect in list and return?
   - Both (print during compilation, also return)?
   - **Recommendation:** Print during, also track in compilation metadata

4. **Integration with Pure PowerShell:**
   - Should `powershell_mode(pure)` be the default when firewall detects no bash?
   - **Recommendation:** Yes, but with clear message explaining why

---

**Next Steps:**
1. Commit import conflict fix
2. Decide on firewall philosophy (recommend hybrid approach documented above)
3. Implement firewall integration with PowerShell mode selection
4. Test in restricted environment (no bash)

**Created:** 2025-10-26
**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
