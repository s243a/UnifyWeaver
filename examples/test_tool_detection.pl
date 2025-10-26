:- encoding(utf8).
% test_tool_detection.pl - Test tool detection module

:- use_module('../src/unifyweaver/core/tool_detection').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Tool Detection Tests                                  ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_executable_detection,
    test_powershell_detection,
    test_cmdlet_detection,
    test_batch_checking,
    test_alternatives,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tool Detection Tests Completed                    ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% Test executable detection
test_executable_detection :-
    format('[Test] Executable Detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test common tools
    TestTools = [bash, awk, jq, curl, python3, nonexistent_tool],
    forall(member(Tool, TestTools), (
        detect_tool_availability(Tool, Status),
        format('  ~w: ~w~n', [Tool, Status])
    )),
    nl.

%% Test PowerShell detection
test_powershell_detection :-
    format('[Test] PowerShell Detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Check if PowerShell is available
    (   check_powershell_available
    ->  format('  ✓ PowerShell is available~n', []),
        % Determine which variant
        (   check_executable_exists('pwsh')
        ->  format('    - PowerShell Core (pwsh) found~n', [])
        ;   true
        ),
        (   check_executable_exists('powershell.exe')
        ->  format('    - Windows PowerShell found~n', [])
        ;   true
        )
    ;   format('  ✗ PowerShell is not available~n', [])
    ),
    nl.

%% Test cmdlet detection
test_cmdlet_detection :-
    format('[Test] PowerShell Cmdlet Detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    TestCmdlets = [import_csv, convertfrom_json, invoke_restmethod],
    forall(member(Cmdlet, TestCmdlets), (
        detect_tool_availability(Cmdlet, Status),
        format('  ~w: ~w~n', [Cmdlet, Status])
    )),
    nl.

%% Test batch checking
test_batch_checking :-
    format('[Test] Batch Tool Checking~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    ToolList = [bash, awk, sed, grep],
    check_all_tools(ToolList, Result),
    format('  check_all_tools(~w): ~w~n', [ToolList, Result]),

    determine_available_tools(ToolList, Available),
    format('  Available from list: ~w~n', [Available]),
    nl.

%% Test tool alternatives
test_alternatives :-
    format('[Test] Tool Alternatives~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    format('  Alternatives for awk: ', []),
    findall(Alt, tool_alternative(awk, Alt), AwkAlts),
    format('~w~n', [AwkAlts]),

    format('  Alternatives for jq: ', []),
    findall(Alt, tool_alternative(jq, Alt), JqAlts),
    format('~w~n', [JqAlts]),

    format('  Alternatives for curl: ', []),
    findall(Alt, tool_alternative(curl, Alt), CurlAlts),
    format('~w~n', [CurlAlts]),
    nl.

:- initialization(main, main).
