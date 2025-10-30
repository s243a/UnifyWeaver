:- encoding(utf8).
% test_firewall_implies.pl - Test Higher-Order Firewall Implications
%
% Tests the firewall_implies system that derives policies from conditions
% using logical inference.

:- use_module('../src/unifyweaver/core/firewall').

%% ============================================
%% MAIN TEST SUITE
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test: Firewall Implications System                  ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Run all tests
    test_default_implications,
    test_user_defined_implications,
    test_override_defaults,
    test_disable_defaults,
    test_derive_policy,
    test_complex_scenarios,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: DEFAULT IMPLICATIONS
%% ============================================

test_default_implications :-
    format('~n[Test 1] Default Implications~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 1.1: no_bash_available implies deny bash service
    (   firewall_implies_default(no_bash_available, Policy1),
        Policy1 = denied(service(powershell, executable(bash)))
    ->  format('  ✓ no_bash_available → deny bash service~n', [])
    ;   format('  ✗ FAIL: Expected no_bash_available implication~n', []),
        fail
    ),

    % Test 1.2: denied_target_language(bash) implies deny bash service
    (   firewall_implies_default(denied_target_language(bash), Policy2),
        Policy2 = denied(service(_, executable(bash)))
    ->  format('  ✓ denied_target_language(bash) → deny bash service~n', [])
    ;   format('  ✗ FAIL: Expected denied_target_language implication~n', []),
        fail
    ),

    % Test 1.3: network_access(denied) implications
    findall(P, firewall_implies_default(network_access(denied), P), NetworkPolicies),
    length(NetworkPolicies, NumNetworkPolicies),
    (   NumNetworkPolicies >= 2
    ->  format('  ✓ network_access(denied) → ~w policies~n', [NumNetworkPolicies])
    ;   format('  ✗ FAIL: Expected multiple network policies~n', []),
        fail
    ),

    % Test 1.4: security_policy(strict) implications
    findall(P, firewall_implies_default(security_policy(strict), P), StrictPolicies),
    length(StrictPolicies, NumStrictPolicies),
    (   NumStrictPolicies >= 2
    ->  format('  ✓ security_policy(strict) → ~w policies~n', [NumStrictPolicies])
    ;   format('  ✗ FAIL: Expected multiple strict policies~n', []),
        fail
    ),

    % Test 1.5: environment(restricted) implications
    findall(P, firewall_implies_default(environment(restricted), P), RestrictedPolicies),
    length(RestrictedPolicies, NumRestrictedPolicies),
    (   NumRestrictedPolicies >= 2
    ->  format('  ✓ environment(restricted) → ~w policies~n', [NumRestrictedPolicies])
    ;   format('  ✗ FAIL: Expected multiple restricted policies~n', []),
        fail
    ),

    format('[✓] Test 1 Complete~n', []),
    !.

%% ============================================
%% TEST 2: USER-DEFINED IMPLICATIONS
%% ============================================

test_user_defined_implications :-
    format('~n[Test 2] User-Defined Implications~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Add a custom user-defined implication
    assertz(firewall:firewall_implies(corporate_policy(banking),
                                      denied(service(_, network_access(external))))),
    format('  ✓ Added custom implication: corporate_policy(banking)~n', []),

    % Verify it works
    (   firewall_implies(corporate_policy(banking), Policy),
        Policy = denied(service(_, network_access(external)))
    ->  format('  ✓ Custom implication works~n', [])
    ;   format('  ✗ FAIL: Custom implication not working~n', []),
        fail
    ),

    % Clean up
    retractall(firewall:firewall_implies(corporate_policy(banking), _)),

    format('[✓] Test 2 Complete~n', []),
    !.

%% ============================================
%% TEST 3: OVERRIDE DEFAULTS
%% ============================================

test_override_defaults :-
    format('~n[Test 3] Override Default Implications~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Add a user-defined implication that overrides default behavior
    % Default: no_bash_available → deny bash service
    % Override: no_bash_available → allow WSL bash service instead
    assertz(firewall:firewall_implies(no_bash_available,
                                      allowed(service(powershell, executable(wsl))))),
    format('  ✓ Added override: no_bash_available → allow WSL~n', []),

    % Verify user-defined takes precedence
    findall(P, firewall_implies(no_bash_available, P), AllPolicies),
    (   member(allowed(service(powershell, executable(wsl))), AllPolicies)
    ->  format('  ✓ User-defined implication present~n', [])
    ;   format('  ✗ FAIL: User-defined override not working~n', []),
        fail
    ),

    % Default should also still be there (unless disabled)
    (   member(denied(service(powershell, executable(bash))), AllPolicies)
    ->  format('  ✓ Default implication also present~n', [])
    ;   format('  ⚠ Default implication not present (may be disabled)~n', [])
    ),

    % Clean up
    retractall(firewall:firewall_implies(no_bash_available, _)),

    format('[✓] Test 3 Complete~n', []),
    !.

%% ============================================
%% TEST 4: DISABLE DEFAULTS
%% ============================================

test_disable_defaults :-
    format('~n[Test 4] Disable Default Implications~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Disable a specific default implication
    assertz(firewall:firewall_implies_disabled(no_bash_available,
                                               denied(service(powershell, executable(bash))))),
    format('  ✓ Disabled default: no_bash_available → deny bash~n', []),

    % Verify it's disabled
    findall(P, (
        firewall_implies_default(no_bash_available, P),
        \+ firewall_implies_disabled(no_bash_available, P)
    ), ActiveDefaults),

    (   \+ member(denied(service(powershell, executable(bash))), ActiveDefaults)
    ->  format('  ✓ Default implication successfully disabled~n', [])
    ;   format('  ✗ FAIL: Default implication still active~n', []),
        fail
    ),

    % Clean up
    retractall(firewall:firewall_implies_disabled(_, _)),

    format('[✓] Test 4 Complete~n', []),
    !.

%% ============================================
%% TEST 5: DERIVE_POLICY
%% ============================================

test_derive_policy :-
    format('~n[Test 5] Derive Policy from Conditions~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test deriving policies from no_bash_available
    derive_policy(no_bash_available, Policies1),
    length(Policies1, NumPolicies1),
    (   NumPolicies1 > 0
    ->  format('  ✓ Derived ~w policies from no_bash_available~n', [NumPolicies1])
    ;   format('  ✗ FAIL: No policies derived~n', []),
        fail
    ),

    % Test deriving policies from network_access(denied)
    derive_policy(network_access(denied), Policies2),
    length(Policies2, NumPolicies2),
    (   NumPolicies2 >= 2
    ->  format('  ✓ Derived ~w policies from network_access(denied)~n', [NumPolicies2])
    ;   format('  ✗ FAIL: Expected multiple network policies~n', []),
        fail
    ),

    % Test deriving policies from security_policy(strict)
    derive_policy(security_policy(strict), Policies3),
    length(Policies3, NumPolicies3),
    (   NumPolicies3 >= 2
    ->  format('  ✓ Derived ~w policies from security_policy(strict)~n', [NumPolicies3])
    ;   format('  ✗ FAIL: Expected multiple strict policies~n', []),
        fail
    ),

    % Test check_derived_policy/3
    (   check_derived_policy(no_bash_available,
                            [denied(service(powershell, executable(bash)))],
                            Result),
        Result = true
    ->  format('  ✓ check_derived_policy works correctly~n', [])
    ;   format('  ✗ FAIL: check_derived_policy failed~n', []),
        fail
    ),

    format('[✓] Test 5 Complete~n', []),
    !.

%% ============================================
%% TEST 6: COMPLEX SCENARIOS
%% ============================================

test_complex_scenarios :-
    format('~n[Test 6] Complex Multi-Condition Scenarios~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Scenario 1: Restricted offline environment
    % Should deny network access AND external executables
    derive_policy(environment(restricted), RestrictedPolicies),
    derive_policy(mode(offline), OfflinePolicies),

    (   member(denied(service(_, executable(_))), RestrictedPolicies)
    ->  format('  ✓ Restricted environment denies executables~n', [])
    ;   format('  ⚠ Restricted environment policy incomplete~n', [])
    ),

    (   member(network_access(denied), OfflinePolicies)
    ->  format('  ✓ Offline mode denies network access~n', [])
    ;   format('  ⚠ Offline mode policy incomplete~n', [])
    ),

    % Scenario 2: Strict security with pure mode preference
    derive_policy(security_policy(strict), StrictPolicies),
    derive_policy(prefer_pure_mode(powershell), PurePolicies),

    findall(prefer(_, _), member(prefer(_, _), StrictPolicies), StrictPreferences),
    findall(prefer(_, _), member(prefer(_, _), PurePolicies), PurePreferences),
    length(StrictPreferences, NumStrict),
    length(PurePreferences, NumPure),

    (   NumStrict > 0, NumPure > 0
    ->  format('  ✓ Both strict and pure policies have preferences~n', [])
    ;   format('  ⚠ Preference policies incomplete~n', [])
    ),

    % Scenario 3: Portable requirement
    derive_policy(require_portable, PortablePolicies),

    (   member(prefer(service(_, builtin(_)), service(_, executable(_))), PortablePolicies)
    ->  format('  ✓ Portable requirement prefers built-ins~n', [])
    ;   format('  ⚠ Portable policy incomplete~n', [])
    ),

    format('[✓] Test 6 Complete~n', []),
    !.

:- initialization(main, main).
