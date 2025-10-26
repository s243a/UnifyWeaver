:- encoding(utf8).
% test_firewall_powershell.pl - Test firewall integration with PowerShell compiler

:- use_module('../src/unifyweaver/core/powershell_compiler').
:- use_module('../src/unifyweaver/core/firewall_v2').

%% ============================================
%% TEST CASES
%% ============================================

test_all :-
    format('~n╔════════════════════════════════════════════════════╗~n', []),
    format('║  Firewall + PowerShell Integration Tests          ║~n', []),
    format('╚════════════════════════════════════════════════════╝~n~n', []),

    test_permissive_mode,
    test_pure_powershell_policy,
    test_firewall_auto_mode,
    test_user_override,

    format('~n╔════════════════════════════════════════════════════╗~n', []),
    format('║  All Firewall Tests Passed ✓                      ║~n', []),
    format('╚════════════════════════════════════════════════════╝~n', []).

%% Test 1: Permissive mode (default)
test_permissive_mode :-
    format('~n[Test 1] Permissive mode - both pure and BaaS allowed~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    % Load permissive policy
    load_firewall_policy(permissive),

    % CSV should work in auto mode (prefer pure)
    compile_to_powershell(test_csv/2, [
        source_type(csv),
        csv_file('test.csv'),
        powershell_mode(auto)
    ], Code1),
    sub_string(Code1, _, _, _, 'Import-Csv'),
    format('[✓] Auto mode with permissive policy chose pure PowerShell~n', []),

    % BaaS should also be allowed (just check mode resolution, not full compilation)
    resolve_powershell_mode(baas, csv, ResolvedBaas),
    ResolvedBaas = baas,
    format('[✓] BaaS mode explicitly allowed (user override respected)~n', []),

    !.

%% Test 2: Pure PowerShell policy
test_pure_powershell_policy :-
    format('~n[Test 2] Pure PowerShell policy - bash denied~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    % Clean previous policy
    retractall(firewall_v2:denied_service(_, _)),

    % Load pure PowerShell policy
    load_firewall_policy(pure_powershell),

    % Auto mode should now choose pure
    compile_to_powershell(test_csv/2, [
        source_type(csv),
        csv_file('test.csv'),
        powershell_mode(auto)
    ], Code),
    sub_string(Code, _, _, _, 'Import-Csv'),
    \+ sub_string(Code, _, _, _, 'uw-bash'),
    format('[✓] Auto mode forced pure (bash denied by firewall)~n', []),

    % Verify firewall denies bash service
    check_service(powershell, executable(bash), Result),
    Result = deny(_),
    format('[✓] Firewall correctly denies bash service for PowerShell~n', []),

    !.

%% Test 3: Firewall auto mode derivation
test_firewall_auto_mode :-
    format('~n[Test 3] Firewall derives mode in auto~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    % Clean and reload permissive
    retractall(firewall_v2:denied_service(_, _)),
    load_firewall_policy(permissive),

    % Firewall should derive auto_with_preference(pure) for CSV
    derive_powershell_mode(csv, DerivedMode),
    format('[✓] Firewall derived mode: ~w for CSV~n', [DerivedMode]),

    % For AWK (no pure support), should derive baas
    derive_powershell_mode(awk, AwkMode),
    AwkMode = baas,
    format('[✓] Firewall derived BaaS for AWK (no pure support)~n', []),

    !.

%% Test 4: User override of firewall
test_user_override :-
    format('~n[Test 4] User explicitly overrides firewall~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    % Load pure PowerShell policy (denies bash)
    retractall(firewall_v2:denied_service(_, _)),
    load_firewall_policy(pure_powershell),

    % User explicitly requests BaaS (should respect user choice)
    resolve_powershell_mode(baas, csv, BaasResolved),
    BaasResolved = baas,
    format('[✓] User explicit baas mode respected (overrides firewall preference)~n', []),

    % User explicitly requests pure (should work)
    resolve_powershell_mode(pure, csv, PureResolved),
    PureResolved = pure,
    format('[✓] User explicit pure mode respected~n', []),

    !.

%% ============================================
%% MAIN ENTRY POINT
%% ============================================

main :-
    test_all,
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

:- initialization(main, main).
