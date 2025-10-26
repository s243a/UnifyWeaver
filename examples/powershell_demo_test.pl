:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% powershell_demo_test.pl - Demo and test PowerShell compilation
% This demonstrates compiling Prolog predicates to PowerShell scripts

:- initialization(main, main).

:- use_module('src/unifyweaver/core/powershell_compiler').
:- use_module('src/unifyweaver/core/stream_compiler').
:- use_module(library(filesex)).

% Example data: family relationships
:- dynamic parent/2.
parent(alice, bob).
parent(alice, barbara).
parent(bob, charlie).
parent(bob, cathy).
parent(charlie, diana).

% Example rule using stream compiler
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

main :-
    writeln(''),
    writeln('╔════════════════════════════════════════════════════════════╗'),
    writeln('║  PowerShell Compiler Demo and Test                        ║'),
    writeln('╚════════════════════════════════════════════════════════════╝'),
    writeln(''),
    
    % Create output directory
    (   exists_directory('output') -> true
    ;   make_directory('output')
    ),
    
    writeln('=== Compiling parent/2 (facts) ==='),
    writeln(''),
    
    % Test 1: Compile parent/2 with inline wrapper (default)
    writeln('Test 1: parent/2 with inline wrapper'),
    compile_to_powershell(parent/2, 
        [wrapper_style(inline), script_name(parent)], 
        ParentPS),
    
    % Write to file
    open('output/parent.ps1', write, PS1, [encoding(utf8)]),
    write(PS1, ParentPS),
    close(PS1),
    writeln('  ✓ Generated: output/parent.ps1'),
    
    % Show a preview
    writeln(''),
    writeln('Preview (first 15 lines):'),
    writeln('---'),
    atom_string(ParentPS, ParentStr),
    split_string(ParentStr, "\n", "", Lines),
    length(Preview, 15),
    append(Preview, _, Lines),
    maplist(writeln, Preview),
    writeln('  ... (truncated)'),
    writeln(''),
    
    % Test 2: Compile grandparent/2 with tempfile wrapper
    writeln('=== Compiling grandparent/2 (rule with pipeline) ==='),
    writeln(''),
    writeln('Test 2: grandparent/2 with tempfile wrapper'),
    compile_to_powershell(grandparent/2,
        [wrapper_style(tempfile), compiler(stream), script_name(grandparent)],
        GrandparentPS),
    
    open('output/grandparent.ps1', write, PS2, [encoding(utf8)]),
    write(PS2, GrandparentPS),
    close(PS2),
    writeln('  ✓ Generated: output/grandparent.ps1'),
    
    % Show preview
    writeln(''),
    writeln('Preview (first 15 lines):'),
    writeln('---'),
    atom_string(GrandparentPS, GrandparentStr),
    split_string(GrandparentStr, "\n", "", GPLines),
    length(GPPreview, 15),
    append(GPPreview, _, GPLines),
    maplist(writeln, GPPreview),
    writeln('  ... (truncated)'),
    writeln(''),
    
    % Test 3: Show both wrapper styles for comparison
    writeln('=== Comparing Wrapper Styles ==='),
    writeln(''),
    
    % Generate parent with tempfile style
    compile_to_powershell(parent/2,
        [wrapper_style(tempfile), script_name(parent_tempfile)],
        ParentTempPS),
    open('output/parent_tempfile.ps1', write, PS3, [encoding(utf8)]),
    write(PS3, ParentTempPS),
    close(PS3),
    writeln('  ✓ Generated: output/parent_tempfile.ps1 (tempfile style)'),
    
    % Generate grandparent with inline style
    compile_to_powershell(grandparent/2,
        [wrapper_style(inline), compiler(stream), script_name(grandparent_inline)],
        GrandparentInlinePS),
    open('output/grandparent_inline.ps1', write, PS4, [encoding(utf8)]),
    write(PS4, GrandparentInlinePS),
    close(PS4),
    writeln('  ✓ Generated: output/grandparent_inline.ps1 (inline style)'),
    
    writeln(''),
    writeln('=== Summary ==='),
    writeln(''),
    writeln('Generated 4 PowerShell scripts in output/:'),
    writeln('  1. parent.ps1            - facts, inline wrapper'),
    writeln('  2. parent_tempfile.ps1   - facts, tempfile wrapper'),
    writeln('  3. grandparent.ps1       - rule, tempfile wrapper'),
    writeln('  4. grandparent_inline.ps1 - rule, inline wrapper'),
    writeln(''),
    writeln('Each script:'),
    writeln('  • Contains embedded bash code from existing compilers'),
    writeln('  • Uses uw-bash compatibility layer for execution'),
    writeln('  • Includes compatibility checks'),
    writeln('  • Supports both inline and tempfile execution styles'),
    writeln(''),
    writeln('To test (after loading uw-bash compatibility layer):'),
    writeln('  pwsh output/parent.ps1'),
    writeln('  pwsh output/grandparent.ps1'),
    writeln(''),
    
    halt(0).
