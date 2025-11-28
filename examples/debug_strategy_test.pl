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
    format('Step 1: Checking if source is registered...~n', []),
    (   current_predicate(source_info/2)
    ->  format('  ✓ source_info/2 exists~n', [])
    ;   format('  ✗ source_info/2 not found~n', [])
    ),

    format('~nStep 2: Attempting compilation...~n', []),
    (   compile_dynamic_source(test_prolog/1, [], Code)
    ->  format('  ✓ Compilation succeeded~n', []),
        format('~nGenerated code (~w bytes):~n', [Code]),
        atom_length(Code, Len),
        format('Length: ~w~n', [Len]),
        (   Len > 200
        ->  sub_atom(Code, 0, 200, _, Start),
            format('First 200 chars: ~w...~n', [Start])
        ;   format('~w~n', [Code])
        )
    ;   format('  ✗ Compilation failed~n', []),
        halt(1)
    ),
    halt(0).
