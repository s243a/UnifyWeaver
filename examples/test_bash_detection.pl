:- encoding(utf8).
% test_bash_detection.pl - Test bash availability auto-detection

:- use_module('../src/unifyweaver/core/firewall_v2').
:- use_module('../src/unifyweaver/core/tool_detection').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Bash Availability Auto-Detection Tests               ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_bash_detection_no_auto_config,
    test_bash_detection_with_auto_config,
    test_auto_detect_environment_policy,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Bash Detection Tests Passed ✓                     ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Bash detection tests failed~n', []),
    halt(1).

%% Test 1: Bash detection without auto-configuration
test_bash_detection_no_auto_config :-
    format('~n[Test 1] Bash detection without auto-configuration~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(denied_service(_, _)),
    retractall(firewall_mode(_)),
    retractall(auto_configure_firewall_for_missing_tools(_)),

    % Don't auto-configure
    % (default behavior - auto_configure_firewall_for_missing_tools is undefined)

    % Detect bash
    detect_and_configure_bash_availability,

    % Check if firewall rule was added
    (   denied_service(powershell, executable(bash))
    ->  format('  ✗ Firewall rule added (should not auto-configure)~n', []),
        fail
    ;   format('  ✓ No firewall rule added (correct - no auto-config)~n', [])
    ),

    % Clean up
    retractall(denied_service(_, _)),
    retractall(firewall_mode(_)),

    format('[✓] Detection without auto-config works~n', []),
    !.

%% Test 2: Bash detection with auto-configuration enabled
test_bash_detection_with_auto_config :-
    format('~n[Test 2] Bash detection with auto-configuration~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(denied_service(_, _)),
    retractall(firewall_mode(_)),
    retractall(auto_configure_firewall_for_missing_tools(_)),

    % Enable auto-configuration
    assertz(auto_configure_firewall_for_missing_tools(true)),

    % Detect bash availability
    detect_tool_availability(bash, BashStatus),
    format('  Bash status: ~w~n', [BashStatus]),

    % Detect and configure
    detect_and_configure_bash_availability,

    % Check result based on actual bash availability
    (   BashStatus = available
    ->  % Bash is available - should NOT add deny rule
        (   denied_service(powershell, executable(bash))
        ->  format('  ✗ Firewall rule added when bash is available~n', []),
            fail
        ;   format('  ✓ No firewall rule (bash is available)~n', [])
        )
    ;   % Bash is unavailable - SHOULD add deny rule (with auto-config)
        (   denied_service(powershell, executable(bash))
        ->  format('  ✓ Firewall rule added (bash unavailable + auto-config)~n', [])
        ;   format('  ✗ No firewall rule (should add when bash unavailable)~n', []),
            fail
        )
    ),

    % Clean up
    retractall(denied_service(_, _)),
    retractall(firewall_mode(_)),
    retractall(auto_configure_firewall_for_missing_tools(_)),

    format('[✓] Detection with auto-config works~n', []),
    !.

%% Test 3: auto_detect_environment policy
test_auto_detect_environment_policy :-
    format('~n[Test 3] auto_detect_environment policy~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(denied_service(_, _)),
    retractall(firewall_mode(_)),
    retractall(auto_configure_firewall_for_missing_tools(_)),

    % Load auto-detect policy
    load_firewall_policy(auto_detect_environment),

    % Check that detection was run
    format('  ✓ auto_detect_environment policy loaded~n', []),

    % Check firewall mode
    (   firewall_mode(permissive)
    ->  format('  ✓ Firewall mode: permissive~n', [])
    ;   format('  ✗ Firewall mode not set to permissive~n', []),
        fail
    ),

    % Clean up
    retractall(denied_service(_, _)),
    retractall(firewall_mode(_)),
    retractall(auto_configure_firewall_for_missing_tools(_)),

    format('[✓] auto_detect_environment policy works~n', []),
    !.

:- initialization(main, main).
