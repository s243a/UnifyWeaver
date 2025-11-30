:- module(test_go_xml_integration, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/targets/go_target).
:- use_module(library(process)).

run_tests :- 
    run_tests([go_xml_integration]).

:- begin_tests(go_xml_integration).

test(go_xml_streaming) :-
    (   path_to_go(GoPath)
    ->  true
    ;   format('Skipping go_xml_streaming: go not found~n', []),
        true
    ),
    
    (   nonvar(GoPath)
    ->
        XmlFile = 'test_go.xml',
        GoFile = 'test_xml.go',
        
        setup_call_cleanup(
            open(XmlFile, write, Stream),
            write(Stream, '<root><item id="1">A</item><item id="2">B</item></root>'),
            close(Stream)
        ),
        
        % Define predicate to map XML fields
        % item(Id, Text) :- json_get('@id', Id), json_get('text', Text).
        % We simulate this by creating a dummy clause structure without asserting it,
        % or asserting it. go_target expects user:clause.
        
        assertz((user:test_xml_go(Id, Text) :- json_get('@id', Id), json_get('text', Text))),
        
        Options = [
            xml_input(true),
            xml_file(XmlFile),
            tags(['item']),
            field_delimiter(colon)
        ],
        
        compile_predicate_to_go(test_xml_go/2, Options, GoCode),
        
        setup_call_cleanup(
            open(GoFile, write, GoStream),
            write(GoStream, GoCode),
            close(GoStream)
        ),
        
        % Run Go program
        setup_call_cleanup(
            process_create(GoPath, ['run', GoFile], [stdout(pipe(Out)), process(PID)]),
            (
                read_string(Out, _, Output),
                process_wait(PID, ExitStatus),
                assertion(ExitStatus == exit(0))
            ),
            close(Out)
        ),
        
        split_string(Output, "\n", "", Lines),
        assertion(member("1:A", Lines)),
        assertion(member("2:B", Lines)),
        
        % Cleanup
        retractall(user:test_xml_go(_, _)),
        delete_file(XmlFile),
        delete_file(GoFile)
    ;   true
    ).

path_to_go(Path) :-
    catch(process_create(path(go), ['version'], [stdout(null)]), _, fail),
    Path = path(go).

:- end_tests(go_xml_integration).
