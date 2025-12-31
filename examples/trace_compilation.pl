:- encoding(utf8).
:- initialization(main, main).

main :-
    format('Loading modules...~n', []),
    use_module('../src/unifyweaver/sources'),
    use_module('../src/unifyweaver/core/dynamic_source_compiler'),
    format('Modules loaded.~n~n', []),

    format('Defining source...~n', []),
    catch(
        source(xml, test_prolog, [
            file('../context/PT/pearltrees_export.rdf'),
            tag('pt:Tree'),
            fields([id: 'pt:treeId']),
            field_compiler(prolog)
        ]),
        E1,
        (format('ERROR in source definition: ~w~n', [E1]), halt(1))
    ),
    format('Source defined successfully.~n~n', []),

    format('Attempting compilation...~n', []),
    catch(
        compile_dynamic_source(test_prolog/1, [], Code),
        E2,
        (format('ERROR in compilation: ~w~n', [E2]), halt(1))
    ),

    (   var(Code)
    ->  format('ERROR: Code is unbound (compilation returned false)~n', []),
        halt(1)
    ;   format('SUCCESS! Generated ~w bytes of code~n', [_]),
        atom_length(Code, Len),
        format('Code length: ~w~n', [Len])
    ),
    halt(0).
