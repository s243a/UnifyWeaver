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
    format('Attempting to compile prolog strategy...~n', []),
    catch(
        (
            compile_dynamic_source(test_prolog/1, [], Code),
            format('SUCCESS! Generated code:~n~n~w~n', [Code])
        ),
        Error,
        (
            format('ERROR: ~w~n', [Error]),
            halt(1)
        )
    ),
    halt(0).
