:- module(xml_query, [
    extract_elements/4,
    extract_elements/5,
    find_elements/5,
    count_elements/3,
    count_elements/4
]).

:- use_module('../sources/xml_source').
:- use_module(library(process)).
:- use_module(library(readutil)).

%% extract_elements(+File, +Tag, +Engine, -Elements)
%  Extract all elements matching Tag from File using Engine.
%  Elements are returned as a list of XML strings.
%
%  Example:
%    extract_elements('data.rdf', 'pt:Tree', awk_pipeline, Trees).
extract_elements(File, Tag, Engine, Elements) :-
    extract_elements(File, Tag, Engine, [], Elements).

%% extract_elements(+File, +Tag, +Engine, +Options, -Elements)
%  Options may include:
%    - case_insensitive(true|false) : pass to awk (IGNORECASE)
extract_elements(File, Tag, Engine, Options, Elements) :-
    % Compile the xml_source
    append([file(File), tag(Tag), engine(Engine)], Options, Config),
    xml_source:compile_source(extract/1, Config, [], BashCode),

    % Save to temp file and execute
    tmp_file('xml_extract', TmpScript),
    tmp_file('xml_output', TmpOutput),
    setup_call_cleanup(
        open(TmpScript, write, S1),
        write(S1, BashCode),
        close(S1)
    ),

    % Execute and capture to file
    format(atom(Command), 'bash ~w > ~w 2>&1', [TmpScript, TmpOutput]),
    shell(Command),

    % Read output
    setup_call_cleanup(
        open(TmpOutput, read, S2, [type(binary)]),
        read_stream_to_codes(S2, Codes),
        close(S2)
    ),

    % Clean up temp files
    delete_file(TmpScript),
    delete_file(TmpOutput),

    % Split on null delimiter
    split_null_delimited(Codes, Elements).

%% find_elements(+File, +Tag, +Engine, +Filter, -Elements)
%  Extract elements and filter them.
%  Filter can be:
%    - contains(Field, Text) - check if Field contains Text (case-insensitive)
%    - matches(Pattern) - check if element matches regex Pattern
%
%  Example:
%    find_elements('data.rdf', 'pt:Tree', awk_pipeline,
%                  contains(title, 'physics'), Trees).
find_elements(File, Tag, Engine, Filter, Filtered) :-
    extract_elements(File, Tag, Engine, Elements),
    include(satisfies_filter(Filter), Elements, Filtered).

%% count_elements(+File, +Tag, -Count)
%  Count elements matching Tag in File using awk_pipeline.
count_elements(File, Tag, Count) :-
    count_elements(File, Tag, awk_pipeline, Count).

%% count_elements(+File, +Tag, +Engine, -Count)
%  Count elements matching Tag in File using specified Engine.
count_elements(File, Tag, Engine, Count) :-
    extract_elements(File, Tag, Engine, Elements),
    length(Elements, Count).

%% ============================================
%% HELPER PREDICATES
%% ============================================

split_null_delimited(Codes, Elements) :-
    split_null_delimited(Codes, [], [], Elements).

split_null_delimited([], Current, Acc, Elements) :-
    (   Current = []
    ->  reverse(Acc, Elements)
    ;   reverse(Current, RevCurrent),
        atom_codes(Element, RevCurrent),
        reverse([Element|Acc], Elements)
    ).
split_null_delimited([0|Rest], Current, Acc, Elements) :-
    !,
    (   Current = []
    ->  split_null_delimited(Rest, [], Acc, Elements)
    ;   reverse(Current, RevCurrent),
        atom_codes(Element, RevCurrent),
        split_null_delimited(Rest, [], [Element|Acc], Elements)
    ).
split_null_delimited([C|Rest], Current, Acc, Elements) :-
    split_null_delimited(Rest, [C|Current], Acc, Elements).

satisfies_filter(contains(Field, Text), Element) :-
    atom_codes(Element, Codes),
    atom_codes(Text, TextCodes),
    downcase_codes(TextCodes, LowerText),
    downcase_codes(Codes, LowerCodes),
    extract_field_value(Field, LowerCodes, FieldValue),
    sub_list(LowerText, FieldValue).

satisfies_filter(matches(Pattern), Element) :-
    atom_codes(Element, Codes),
    atom_codes(Pattern, PatternCodes),
    downcase_codes(Codes, LowerCodes),
    downcase_codes(PatternCodes, LowerPattern),
    sub_list(LowerPattern, LowerCodes).

extract_field_value(title, Codes, Value) :-
    % Find <dcterms:title>...</dcterms:title>
    append(_, [60,100,99,116,101,114,109,115,58,116,105,116,108,101,62|Rest], Codes),
    append(Value, [60,47,100,99,116,101,114,109,115,58,116,105,116,108,101,62|_], Rest),
    !.
extract_field_value(_, Codes, Codes).

downcase_codes([], []).
downcase_codes([C|Rest], [LC|LRest]) :-
    (   C >= 65, C =< 90
    ->  LC is C + 32
    ;   LC = C
    ),
    downcase_codes(Rest, LRest).

sub_list([], _) :- !.
sub_list(Sub, List) :-
    append(Sub, _, List), !.
sub_list(Sub, [_|Rest]) :-
    sub_list(Sub, Rest).
