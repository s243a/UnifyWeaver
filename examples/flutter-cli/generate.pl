#!/usr/bin/env swipl

% Generate Flutter CLI example from browse_panel_spec

:- use_module('../../src/unifyweaver/ui/project_scaffold', [generate_project_files/4]).
:- use_module('../../src/unifyweaver/ui/http_cli_ui', [browse_panel_spec/1]).

main :-
    % Get the browse panel spec
    browse_panel_spec(UISpec),

    % Generate the Flutter project files
    generate_project_files(flutter, http_cli, UISpec, Files),

    % Write files to current directory
    OutputDir = '.',
    forall(
        member(RelPath-Content, Files),
        (   atomic_list_concat([OutputDir, '/', RelPath], FullPath),
            format('Writing: ~w~n', [FullPath]),
            open(FullPath, write, Stream),
            write(Stream, Content),
            close(Stream)
        )
    ),

    format('~nFlutter project generated successfully!~n'),
    halt(0).

:- main.
