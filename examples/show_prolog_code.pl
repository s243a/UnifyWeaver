:- encoding(utf8).
:- initialization(main, main).

:- use_module('../src/unifyweaver/sources').
:- use_module('../src/unifyweaver/core/dynamic_source_compiler').

% Test with minimal fields
:- source(xml, test_prolog, [
    file('../context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId']),
    field_compiler(prolog)
]).

main :-
    format('Attempting to compile...~n~n', []),
    (   compile_dynamic_source(test_prolog/1, [], Code)
    ->  format('Generated bash code:~n', []),
        format('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━~n', []),
        format('~w~n', [Code]),
        format('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━~n~n', []),

        % Extract the prolog script path from the bash code
        (   sub_atom(Code, Before, _, After, '.pl')
        ->  sub_atom(Code, PathStart, PathLen, _, _),
            PathStart < Before,
            PathEnd is Before + 3,
            TotalLen is PathEnd - PathStart,
            sub_atom(Code, PathStart, TotalLen, _, MaybePath),
            atom_concat('/tmp/', Rest, MaybePath),
            atom_concat(ScriptName, '.pl', Rest),
            format('Found script reference: ~w~n~n', [MaybePath]),

            % Read and show the generated Prolog code
            (   exists_file(MaybePath)
            ->  format('Generated Prolog script (~w):~n', [MaybePath]),
                format('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━~n', []),
                read_file_to_string(MaybePath, PrologCode, []),
                format('~w~n', [PrologCode]),
                format('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━~n', [])
            ;   format('Prolog script file not found: ~w~n', [MaybePath])
            )
        ;   format('Could not find .pl reference in bash code~n', [])
        )
    ;   format('Compilation failed~n', []),
        halt(1)
    ),
    halt(0).
