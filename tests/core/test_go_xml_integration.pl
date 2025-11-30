:- module(test_go_xml_integration, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/targets/go_target).
:- use_module(library(process)).

run_tests :-
    run_tests([go_xml_integration]).

:- begin_tests(go_xml_integration).

test(go_xml_file) :-
    run_go_xml_test(file, []).

test(go_xml_stdin) :-
    run_go_xml_test(stdin, []).

test(go_xml_fragments) :-
    run_go_xml_test(stdin_fragments, []).

run_go_xml_test(Mode, ExtraOptions) :-
    (   path_to_go(GoPath)
    ->  true
    ;   format('Skipping go_xml test: go not found~n', []),
        true
    ),
    
    (   nonvar(GoPath)
    ->
        GoFile = 'test_xml.go',
        (   Mode == stdin_fragments
        ->  XmlContent = '<item id="1">A</item><item id="2">B</item>'
        ;   XmlContent = '<root><item id="1">A</item><item id="2">B</item></root>'
        ),
        
        % Define dummy predicate
        retractall(user:test_xml_go(_, _)),
        assertz((user:test_xml_go(Id, Text) :- json_get('@id', Id), json_get('text', Text))),
        
        (   Mode == file
        ->  XmlFile = 'test_go.xml',
            setup_call_cleanup(
                open(XmlFile, write, Stream),
                write(Stream, XmlContent),
                close(Stream)
            ),
            BaseOptions = [xml_file(XmlFile)]
        ;   BaseOptions = [xml_file(stdin)]
        ),
        
        append(BaseOptions, [
            xml_input(true),
            tags(['item']),
            field_delimiter(colon)
        | ExtraOptions], Options),
        
        compile_predicate_to_go(test_xml_go/2, Options, GoCode),
        
        setup_call_cleanup(
            open(GoFile, write, GoStream),
            write(GoStream, GoCode),
            close(GoStream)
        ),
        
        % Run Go program
                (   Mode == file
                ->  setup_call_cleanup(
                        process_create(GoPath, ['run', GoFile], [stdout(pipe(Out)), process(PID)]),
                        (
                            read_string(Out, _, Output),
                            process_wait(PID, ExitStatus),
                            assertion(ExitStatus == exit(0))
                        ),
                        close(Out)
                    )
                ;   (Mode == stdin ; Mode == stdin_fragments)
                ->  setup_call_cleanup(
                        process_create(GoPath, ['run', GoFile], [stdin(pipe(In)), stdout(pipe(Out)), process(PID)]),
                        (
                            write(In, XmlContent),
                            close(In),
                            read_string(Out, _, Output),
                            process_wait(PID, ExitStatus),
                            assertion(ExitStatus == exit(0))
                        ),
                        close(Out)
                    )
                ),
        
        % Verify output
        split_string(Output, "\n", "", Lines),
        assertion(member("1:A", Lines)),
        assertion(member("2:B", Lines)),
        
        % Cleanup
        retractall(user:test_xml_go(_, _)),
        (Mode == file -> delete_file(XmlFile) ; true),
        delete_file(GoFile)
    ;   true
    ).

path_to_go(Path) :-
    catch(process_create(path(go), ['version'], [stdout(null)]), _, fail),
    Path = path(go).

:- end_tests(go_xml_integration).