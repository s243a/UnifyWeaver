:- module(test_python_xml_integration, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/targets/python_target).
:- use_module(library(process)).

run_tests :-
    run_tests([python_xml_integration]).

:- begin_tests(python_xml_integration).

test(python_xml_source) :-
    % Create test XML file
    XmlFile = 'test_integration.xml',
    setup_call_cleanup(
        open(XmlFile, write, Stream),
        write(Stream, '<root><item id="1">A</item><item id="2">B</item></root>'),
        close(Stream)
    ),
    
    % Define options
    Options = [
        input_source(xml(XmlFile, ['item'])),
        record_format(jsonl),
        mode(procedural)
    ],
    
    % Compile predicate (identity - just pass through)
    % We use a dummy predicate name since we aren't using Prolog clauses for logic here, 
    % but we need to pass a name. 
    % Wait, compile_predicate_to_python requires clauses to exist!
    % We need to assert a dummy clause in user module or similar.
    
    assertz(user:test_identity(X) :- true),
    
    compile_predicate_to_python(user:test_identity/1, Options, PythonCode),
    
    % Write Python script
    ScriptFile = 'test_xml_int.py',
    setup_call_cleanup(
        open(ScriptFile, write, PyStream),
        write(PyStream, PythonCode),
        close(PyStream)
    ),
    
    % Execute Python script
    % Note: It should read from file, not stdin
    setup_call_cleanup(
        process_create(path(python3), [ScriptFile], [stdout(pipe(Out)), process(PID)]),
        (
            read_string(Out, _, Output),
            process_wait(PID, ExitStatus),
            assertion(ExitStatus == exit(0))
        ),
        close(Out)
    ),
    
    format('Output: ~s~n', [Output]),

    % Verify output
    sub_string(Output, _, _, _, '{"@id": "1", "text": "A"}'),
    sub_string(Output, _, _, _, '{"@id": "2", "text": "B"}'),
    
    % Clean up
    retract(user:test_identity(_)),
    delete_file(XmlFile),
    delete_file(ScriptFile).

:- end_tests(python_xml_integration).
