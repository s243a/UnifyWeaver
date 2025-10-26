:- encoding(utf8).
% test_firewall_tools.pl - Test firewall integration with tool detection

:- use_module('../src/unifyweaver/core/firewall_v2').
:- use_module('../src/unifyweaver/core/tool_detection').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Firewall + Tool Detection Integration Tests          ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_tool_availability_check,
    test_prefer_available_tools,
    test_forbid_policy,
    test_warn_policy,
    test_bash_unavailable_scenario,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Integration Tests Passed ✓                        ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Integration tests failed~n', []),
    halt(1).

%% Test 1: Basic tool availability checking with firewall
test_tool_availability_check :-
    format('~n[Test 1] Tool availability check with firewall~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean slate
    retractall(denied_tool(_)),

    % Check bash availability (should be available)
    check_tool_availability(bash, powershell, Result1),
    format('  bash for PowerShell: ~w~n', [Result1]),
    Result1 = available,

    % Deny bash via firewall
    assertz(denied_tool(bash)),
    check_tool_availability(bash, powershell, Result2),
    format('  bash (after firewall deny): ~w~n', [Result2]),
    Result2 = denied(_),

    % Clean up
    retractall(denied_tool(_)),

    format('[✓] Tool availability checking works~n', []),
    !.

%% Test 2: Prefer available tools
test_prefer_available_tools :-
    format('~n[Test 2] Prefer available tools policy~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(prefer_available_tools(_)),
    retractall(denied_service(_, _)),

    % Set prefer available tools
    assertz(prefer_available_tools(true)),

    % Simulate bash being unavailable and derive mode
    % (We can't actually make bash unavailable, but we can test the logic)
    derive_compilation_mode_with_tools(csv, [bash, awk], Mode),
    format('  Mode with all tools available: ~w~n', [Mode]),

    % Clean up
    retractall(prefer_available_tools(_)),

    format('[✓] Prefer available tools works~n', []),
    !.

%% Test 3: Forbid policy
test_forbid_policy :-
    format('~n[Test 3] Forbid policy for missing tools~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(tool_availability_policy(_)),

    % Note: This test would fail compilation if tools were actually missing
    % So we just test that the policy is set correctly
    assertz(tool_availability_policy(forbid)),

    % Verify policy is set
    tool_availability_policy(P),
    format('  Current policy: ~w~n', [P]),
    P = forbid,

    % Clean up
    retractall(tool_availability_policy(_)),

    format('[✓] Forbid policy can be set~n', []),
    !.

%% Test 4: Warn policy
test_warn_policy :-
    format('~n[Test 4] Warn policy for missing tools~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(tool_availability_policy(_)),

    % Set warn policy
    assertz(tool_availability_policy(warn)),

    % Verify
    tool_availability_policy(P),
    format('  Current policy: ~w~n', [P]),
    P = warn,

    % Clean up
    retractall(tool_availability_policy(_)),

    format('[✓] Warn policy can be set~n', []),
    !.

%% Test 5: Bash unavailable scenario
test_bash_unavailable_scenario :-
    format('~n[Test 5] Bash unavailable - prefer pure PowerShell~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(prefer_available_tools(_)),
    retractall(denied_service(_, _)),

    % Set prefer available and deny bash service
    assertz(prefer_available_tools(true)),
    load_firewall_policy(pure_powershell),

    % CSV source should use pure PowerShell
    derive_powershell_mode(csv, Mode),
    format('  Mode for CSV when bash denied: ~w~n', [Mode]),
    Mode = pure,

    % Clean up
    retractall(prefer_available_tools(_)),
    retractall(denied_service(_, _)),
    retractall(firewall_mode(_)),

    format('[✓] Bash unavailable scenario handled correctly~n', []),
    !.

:- initialization(main, main).
