:- module(test_xml_source, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/sources/xml_source).
:- use_module(library(process)).

run_tests :-
    run_tests([xml_source]).

:- begin_tests(xml_source).

test(basic_extraction) :-
    % Compile the source
    xml_source:compile_source(test_xml/1, [
        xml_file('tests/test_data/sample.rdf'),
        tags(['pt:Tree'])
    ], [], BashCode),

    % Write the generated script to a file
    open('test_xml.sh', write, Stream, [newline(posix)]),
    write(Stream, BashCode),
    close(Stream),

    % Execute the script and capture the output
    process_create(path(bash), ['test_xml.sh'], [stdout(pipe(Out))]),
    read_stream_to_codes(Out, Codes),
    close(Out),

    % The output should be null-delimited. Count the null characters.
    include(=(0), Codes, Nulls),
    length(Nulls, 2),

    % Clean up
    delete_file('test_xml.sh').

test(xmlstarlet_extraction) :-
    % Compile the source with xmlstarlet engine
    xml_source:compile_source(test_xml/1, [
        xml_file('tests/test_data/sample.rdf'),
        tags(['pt:Tree']),
        engine(xmlstarlet)
    ], [], BashCode),

    % Write the generated script to a file
    open('test_xml_xmlstarlet.sh', write, Stream, [newline(posix)]),
    write(Stream, BashCode),
    close(Stream),

    % Execute the script and capture the output
    process_create(path(bash), ['test_xml_xmlstarlet.sh'], [stdout(pipe(Out))]),
    read_stream_to_codes(Out, Codes),
    close(Out),

    % The output should be null-delimited. Count the null characters.
    include(=(0), Codes, Nulls),
    length(Nulls, 2),

    % Clean up
    delete_file('test_xml_xmlstarlet.sh').

:- end_tests(xml_source).
