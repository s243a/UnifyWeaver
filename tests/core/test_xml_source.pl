:- module(test_xml_source, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/sources/xml_source).
:- use_module(library(process)).
:- use_module(library(apply)). % for exclude/3

:- set_test_options([output(always)]).

run_tests :-
    run_tests([xml_source]).

:- begin_tests(xml_source).

test(basic_extraction) :-
    % Compile the source
    once(xml_source:compile_source(test_xml/1, [
        xml_file('tests/test_data/sample.rdf'),
        tags(['pt:Tree'])
    ], [], BashCode)),

    % Write the generated script to a file
    setup_call_cleanup(
        open('test_xml.sh', write, Stream, [newline(posix)]),
        write(Stream, BashCode),
        close(Stream)
    ),

    % Execute the script and capture the output
    setup_call_cleanup(
        process_create(path(bash), ['test_xml.sh'], [stdout(pipe(Out)), process(PID)]),
        (
            read_stream_to_codes(Out, Codes),
            process_wait(PID, ExitStatus),
            assertion(ExitStatus == exit(0))
        ),
        close(Out)
    ),

    % The output should be null-delimited. Count the null characters.
    include(=(0), Codes, Nulls),
    length(Nulls, 2),

    % Decode records and show them for visibility
    codes_to_records(Codes, Records),
    length(Records, 2),
    report_records(iterparse, Records),

    % Basic content assertions
    forall(member(R, Records), sub_string(R, _, _, _, "<pt:Tree")),
    records_contains(Records, "Test Tree 1"),
    records_contains(Records, "Test Tree 2"),

    % Clean up
    delete_file('test_xml.sh').

test(xmlstarlet_extraction) :-
    % Compile the source with xmlstarlet engine
    once(xml_source:compile_source(test_xml/1, [
        xml_file('tests/test_data/sample.rdf'),
        tags(['pt:Tree']),
        engine(xmlstarlet)
    ], [], BashCode)),

    % Write the generated script to a file
    setup_call_cleanup(
        open('test_xml_xmlstarlet.sh', write, Stream, [newline(posix)]),
        write(Stream, BashCode),
        close(Stream)
    ),

    % Execute the script and capture the output
    setup_call_cleanup(
        process_create(path(bash), ['test_xml_xmlstarlet.sh'], [stdout(pipe(Out)), process(PID)]),
        (
            read_stream_to_codes(Out, Codes),
            process_wait(PID, ExitStatus),
            assertion(ExitStatus == exit(0))
        ),
        close(Out)
    ),

    % The output should be null-delimited. Count the null characters.
    include(=(0), Codes, Nulls),
    length(Nulls, 2),

    % Decode records and show them for visibility
    codes_to_records(Codes, Records),
    length(Records, 2),
    report_records(xmlstarlet, Records),

    % Basic content assertions
    forall(member(R, Records), sub_string(R, _, _, _, "<pt:Tree")),
    records_contains(Records, "Test Tree 1"),
    records_contains(Records, "Test Tree 2"),

    % Clean up
    delete_file('test_xml_xmlstarlet.sh').

test(xmllint_extraction) :-
    % Compile the source with xmllint engine
    once(xml_source:compile_source(test_xml/1, [
        xml_file('tests/test_data/sample.rdf'),
        tags(['pt:Tree']),
        engine(xmllint)
    ], [], BashCode)),

    % Write the generated script to a file
    setup_call_cleanup(
        open('test_xml_xmllint.sh', write, Stream, [newline(posix)]),
        write(Stream, BashCode),
        close(Stream)
    ),

    % Execute the script and capture the output
    setup_call_cleanup(
        process_create(path(bash), ['test_xml_xmllint.sh'], [stdout(pipe(Out)), process(PID)]),
        (
            read_stream_to_codes(Out, Codes),
            process_wait(PID, ExitStatus),
            assertion(ExitStatus == exit(0))
        ),
        close(Out)
    ),

    % The output should be null-delimited. Count the null characters.
    include(=(0), Codes, Nulls),
    length(Nulls, 2),

    % Decode records and show them for visibility
    codes_to_records(Codes, Records),
    length(Records, 2),
    report_records(xmllint, Records),

    % Basic content assertions
    forall(member(R, Records), sub_string(R, _, _, _, "<pt:Tree")),
    records_contains(Records, "Test Tree 1"),
    records_contains(Records, "Test Tree 2"),

    % Clean up
    delete_file('test_xml_xmllint.sh').

test(xmllint_extraction_without_namespace_fix) :-
    % Compile the source with xmllint engine but disable namespace repair
    once(xml_source:compile_source(test_xml_ns/1, [
        xml_file('tests/test_data/sample.rdf'),
        tags(['pt:Tree']),
        engine(xmllint),
        namespace_fix(false)
    ], [], BashCode)),

    setup_call_cleanup(
        open('test_xml_xmllint_no_fix.sh', write, Stream, [newline(posix)]),
        write(Stream, BashCode),
        close(Stream)
    ),

    setup_call_cleanup(
        process_create(path(bash), ['test_xml_xmllint_no_fix.sh'], [stdout(pipe(Out)), process(PID)]),
        (
            read_stream_to_codes(Out, Codes),
            process_wait(PID, ExitStatus),
            assertion(ExitStatus == exit(0))
        ),
        close(Out)
    ),

    include(=(0), Codes, Nulls),
    length(Nulls, 2),

    codes_to_records(Codes, Records),
    length(Records, 2),
    report_records(xmllint_no_fix, Records),

    forall(member(R, Records), sub_string(R, _, _, _, "<pt:Tree")),
    records_contains(Records, "Test Tree 1"),
    records_contains(Records, "Test Tree 2"),
    forall(member(R, Records), \+ sub_string(R, _, _, _, "xmlns:")),

    delete_file('test_xml_xmllint_no_fix.sh').

:- end_tests(xml_source).

codes_to_records(Codes, Records) :-
    string_codes(String, Codes),
    split_string(String, "\u0000", "\u0000", RawRecords),
    exclude(=( ""), RawRecords, Records).

report_records(Label, Records) :-
    forall(nth1(Index, Records, Record),
           print_message(informational, xml_record(Label, Index, Record))).

records_contains(Records, Needle) :-
    once((
        member(Record, Records),
        sub_string(Record, _, _, _, Needle)
    )).

:- multifile prolog:message//1.

prolog:message(xml_record(Label, Index, Record)) -->
    [ '~w record ~d:'-[Label, Index], nl,
      '~s'-[Record], nl
    ].
