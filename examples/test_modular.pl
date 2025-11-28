:- encoding(utf8).
:- initialization(main, main).

main :-
    format('Loading modules...~n', []),
    use_module('../src/unifyweaver/sources'),
    use_module('../src/unifyweaver/core/dynamic_source_compiler'),
    format('Modules loaded.~n~n', []),

    format('Defining source with modular strategy...~n', []),
    source(xml, test_modular, [
        file('../context/PT/pearltrees_export.rdf'),
        tag('pt:Tree'),
        fields([id: 'pt:treeId']),
        field_compiler(modular)
    ]),
    format('Source defined successfully.~n~n', []),

    format('Attempting compilation...~n', []),
    compile_dynamic_source(test_modular/1, [], Code),
    format('SUCCESS! Generated ~w bytes of code~n', [_]),
    atom_length(Code, Len),
    format('Code length: ~w~n', [Len]),
    halt(0).
