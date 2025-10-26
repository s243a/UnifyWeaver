:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% powershell_compilation_demo.pl - Demonstrate PowerShell compilation
% Shows how to compile Prolog predicates to PowerShell scripts

:- initialization(main, main).

:- use_module(unifyweaver(core/powershell_compiler)).
:- use_module(unifyweaver(core/platform_compat)).

%% ============================================================================
%% Example Predicates to Compile
%% ============================================================================

%% Simple facts - colors
color(red).
color(green).
color(blue).
color(yellow).

%% Simple rule - primary colors
primary_color(C) :- color(C), member(C, [red, blue, yellow]).

%% Family relationships (for more complex example)
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).
parent(pat, jim).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

%% ============================================================================
%% Compilation Demos
%% ============================================================================

demo_simple_facts :-
    safe_format('~n╔════════════════════════════════════════╗~n', []),
    safe_format('║  Demo 1: Compile Simple Facts         ║~n', []),
    safe_format('╚════════════════════════════════════════╝~n~n', []),

    safe_format('📝 Compiling color/1 facts to PowerShell...~n~n', []),

    compile_to_powershell(
        color/1,
        [compiler(stream), wrapper_style(inline), script_name(colors)],
        PowerShellCode
    ),

    safe_format('✅ Generated PowerShell Code (~w characters):~n~n', [string_length(PowerShellCode, Len), Len]),
    writeln('--- BEGIN PowerShell Script ---'),
    writeln(PowerShellCode),
    writeln('--- END PowerShell Script ---').

demo_simple_rule :-
    safe_format('~n╔════════════════════════════════════════╗~n', []),
    safe_format('║  Demo 2: Compile Simple Rule          ║~n', []),
    safe_format('╚════════════════════════════════════════╝~n~n', []),

    safe_format('📝 Compiling primary_color/1 rule to PowerShell...~n~n', []),

    compile_to_powershell(
        primary_color/1,
        [compiler(stream), wrapper_style(inline), script_name(primary_colors)],
        PowerShellCode
    ),

    safe_format('✅ Generated PowerShell Code:~n~n', []),
    writeln('--- BEGIN PowerShell Script ---'),
    writeln(PowerShellCode),
    writeln('--- END PowerShell Script ---').

demo_join_rule :-
    safe_format('~n╔════════════════════════════════════════╗~n', []),
    safe_format('║  Demo 3: Compile Join (grandparent)   ║~n', []),
    safe_format('╚════════════════════════════════════════╝~n~n', []),

    safe_format('📝 Compiling grandparent/2 join rule to PowerShell...~n~n', []),

    compile_to_powershell(
        grandparent/2,
        [compiler(stream), wrapper_style(inline), script_name(grandparents)],
        PowerShellCode
    ),

    safe_format('✅ Generated PowerShell Code:~n~n', []),
    writeln('--- BEGIN PowerShell Script ---'),
    writeln(PowerShellCode),
    writeln('--- END PowerShell Script ---').

demo_tempfile_style :-
    safe_format('~n╔════════════════════════════════════════╗~n', []),
    safe_format('║  Demo 4: Tempfile Wrapper Style       ║~n', []),
    safe_format('╚════════════════════════════════════════╝~n~n', []),

    safe_format('📝 Compiling color/1 with tempfile wrapper...~n~n', []),

    compile_to_powershell(
        color/1,
        [compiler(stream), wrapper_style(tempfile), script_name(colors_tempfile)],
        PowerShellCode
    ),

    safe_format('✅ Generated PowerShell Code (tempfile style):~n~n', []),
    writeln('--- BEGIN PowerShell Script ---'),
    writeln(PowerShellCode),
    writeln('--- END PowerShell Script ---').

demo_file_output :-
    safe_format('~n╔════════════════════════════════════════╗~n', []),
    safe_format('║  Demo 5: Write to File                ║~n', []),
    safe_format('╚════════════════════════════════════════╝~n~n', []),

    % Create output directory if needed
    (   exists_directory('output')
    ->  true
    ;   make_directory('output')
    ),

    safe_format('📝 Compiling grandparent/2 to output/grandparent.ps1...~n', []),

    compile_to_powershell(
        grandparent/2,
        [
            compiler(stream),
            wrapper_style(inline),
            script_name(grandparent),
            output_file('output/grandparent.ps1')
        ],
        _PowerShellCode
    ),

    safe_format('✅ Written to: output/grandparent.ps1~n', []),
    safe_format('   Run with: .\\output\\grandparent.ps1~n', []).

demo_no_compat_check :-
    safe_format('~n╔════════════════════════════════════════╗~n', []),
    safe_format('║  Demo 6: Disable Compat Check         ║~n', []),
    safe_format('╚════════════════════════════════════════╝~n~n', []),

    safe_format('📝 Compiling color/1 without compat layer check...~n~n', []),

    compile_to_powershell(
        color/1,
        [compiler(stream), wrapper_style(inline), compat_check(false), script_name(colors_no_check)],
        PowerShellCode
    ),

    safe_format('✅ Generated PowerShell Code (no compat check):~n~n', []),
    writeln('--- BEGIN PowerShell Script ---'),
    writeln(PowerShellCode),
    writeln('--- END PowerShell Script ---').

%% ============================================================================
%% Main Demo Runner
%% ============================================================================

main :-
    safe_format('~n╔════════════════════════════════════════════════════════╗~n', []),
    safe_format('║  UnifyWeaver PowerShell Compilation Demo              ║~n', []),
    safe_format('╚════════════════════════════════════════════════════════╝~n', []),

    safe_format('~nThis demo shows how to compile Prolog predicates to PowerShell.~n', []),
    safe_format('The generated scripts use the PowerShell compatibility layer~n', []),
    safe_format('to execute bash implementations.~n', []),

    % Run all demos
    catch(demo_simple_facts, E1, format('[ERROR] Demo 1 failed: ~w~n', [E1])),
    catch(demo_simple_rule, E2, format('[ERROR] Demo 2 failed: ~w~n', [E2])),
    catch(demo_join_rule, E3, format('[ERROR] Demo 3 failed: ~w~n', [E3])),
    catch(demo_tempfile_style, E4, format('[ERROR] Demo 4 failed: ~w~n', [E4])),
    catch(demo_file_output, E5, format('[ERROR] Demo 5 failed: ~w~n', [E5])),
    catch(demo_no_compat_check, E6, format('[ERROR] Demo 6 failed: ~w~n', [E6])),

    safe_format('~n╔════════════════════════════════════════════════════════╗~n', []),
    safe_format('║  Demo Complete                                         ║~n', []),
    safe_format('╚════════════════════════════════════════════════════════╝~n', []),

    safe_format('~n📚 For more information, see docs/POWERSHELL_TARGET.md~n~n', []).
