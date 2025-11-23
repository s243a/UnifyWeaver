:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% xml_field_compiler.pl - Field extraction compiler for XML sources
% Compiles field extraction queries to AWK/Python code (declarative approach)

:- module(xml_field_compiler, [
    compile_field_extraction/6,  % Main entry point
    validate_field_spec/1,        % Validate field specification
    supports_engine/2             % Check if engine supports field extraction
]).

:- use_module(library(lists)).
:- use_module('../core/template_system').

%% ============================================
%% FIELD EXTRACTION COMPILER
%% ============================================

%% compile_field_extraction(+Pred/Arity, +File, +Tag, +FieldSpec, +Options, -BashCode)
%  Compile field extraction to executable code.
%
%  FieldSpec is a list of Field:Selector pairs:
%    fields([
%        id: 'pt:treeId',
%        title: 'dcterms:title'
%    ])
%
%  Options may include:
%    - engine(Engine): awk_pipeline, iterparse, xmllint
%    - output(Format): dict, list, compound(Functor)
compile_field_extraction(Pred/Arity, File, Tag, FieldSpec, Options, BashCode) :-
    format('  Compiling field extraction: ~w/~w~n', [Pred, Arity]),

    % Validate field specification
    validate_field_spec(FieldSpec),

    % Select engine (default: awk_pipeline for performance)
    option(engine(Engine), Options, awk_pipeline),

    % Validate engine supports field extraction
    (   supports_engine(Engine, field_extraction)
    ->  true
    ;   format('Error: Engine ~w does not support field extraction~n', [Engine]),
        format('  Supported engines: awk_pipeline, iterparse~n', []),
        fail
    ),

    % Select output format (default: dict)
    option(output(OutputFormat), Options, dict),

    % Compile based on engine
    (   Engine = awk_pipeline
    ->  compile_awk_field_extraction(Pred, File, Tag, FieldSpec, OutputFormat, BashCode)
    ;   Engine = iterparse
    ->  compile_iterparse_field_extraction(Pred, File, Tag, FieldSpec, OutputFormat, BashCode)
    ;   format('Error: Field extraction not yet implemented for engine ~w~n', [Engine]),
        fail
    ).

%% ============================================
%% VALIDATION
%% ============================================

%% validate_field_spec(+FieldSpec)
%  Validate field specification syntax.
validate_field_spec(FieldSpec) :-
    is_list(FieldSpec),
    !,
    maplist(validate_field_entry, FieldSpec).
validate_field_spec(Other) :-
    format('Error: fields() must be a list, got: ~w~n', [Other]),
    fail.

validate_field_entry(FieldName:Selector) :-
    !,
    atom(FieldName),
    validate_selector(Selector).
validate_field_entry(Other) :-
    format('Error: field entry must be FieldName:Selector, got: ~w~n', [Other]),
    fail.

validate_selector(Atom) :-
    atom(Atom),
    !.  % Simple tag name like 'pt:treeId'
validate_selector(xpath(Path)) :-
    !,
    atom(Path).  % XPath expression
validate_selector(xpath(Path, Transform)) :-
    !,
    atom(Path),
    atom(Transform).  % XPath with transformation
validate_selector(Other) :-
    format('Error: invalid selector: ~w~n', [Other]),
    fail.

%% supports_engine(+Engine, +Feature)
%  Check if an engine supports a feature.
supports_engine(awk_pipeline, field_extraction).
supports_engine(iterparse, field_extraction).
supports_engine(iterparse, xpath).

%% ============================================
%% AWK FIELD EXTRACTION COMPILER
%% ============================================

%% compile_awk_field_extraction(+Pred, +File, +Tag, +FieldSpec, +OutputFormat, -BashCode)
%  Generate AWK script for field extraction.
compile_awk_field_extraction(Pred, File, Tag, FieldSpec, OutputFormat, BashCode) :-
    % Generate AWK script
    generate_awk_field_script(Tag, FieldSpec, OutputFormat, AwkScript),

    % Save AWK script to temp file
    tmp_file_stream(text, TmpAwkFile, AwkStream),
    format(AwkStream, '~w', [AwkScript]),
    close(AwkStream),

    % Generate bash wrapper
    atom_string(Pred, PredStr),
    atom_string(Tag, TagStr),
    render_named_template(xml_awk_field_extraction,
        [
            pred=PredStr,
            file=File,
            tag=TagStr,
            awk_script=TmpAwkFile
        ],
        [source_order([file, generated])],
        BashCode).

%% generate_awk_field_script(+Tag, +FieldSpec, +OutputFormat, -AwkScript)
%  Generate AWK script that extracts fields from XML elements.
generate_awk_field_script(Tag, FieldSpec, OutputFormat, AwkScript) :-
    % Extract field names and selectors
    maplist(extract_field_info, FieldSpec, FieldNames, Selectors),

    % Generate AWK extraction code for each field
    maplist(generate_awk_extractor, FieldNames, Selectors, Extractors),
    atomic_list_concat(Extractors, '\n', ExtractorCode),

    % Generate output formatting code
    generate_awk_output(FieldNames, OutputFormat, OutputCode),

    % Combine into complete AWK script
    format(atom(AwkScript),
'BEGIN {
    RS = "\\0"  # Null-delimited input
    ORS = "\\0" # Null-delimited output
}

/<~w[> ]/ {
    # Extract fields
~w

    # Output formatted result
~w
}
', [Tag, ExtractorCode, OutputCode]).

extract_field_info(FieldName:Selector, FieldName, Selector).

%% generate_awk_extractor(+FieldName, +Selector, -ExtractorCode)
%  Generate AWK code to extract a single field.
generate_awk_extractor(FieldName, TagName, Code) :-
    atom(TagName),
    !,
    % Tag extraction - handles both CDATA and regular content
    % Pattern: <tag>content</tag> or <tag><![CDATA[content]]></tag>
    format(atom(Code), '    if (match($0, /<~w><!\\[CDATA\\[([^\\]]+)\\]\\]><\\/~w>/, arr)) { ~w = arr[1] } else { match($0, /<~w>([^<]+)<\\/~w>/, arr); ~w = arr[1] }',
           [TagName, TagName, FieldName, TagName, TagName, FieldName]).
generate_awk_extractor(FieldName, xpath(Path), Code) :-
    % XPath not supported in AWK yet
    format(atom(Code), '    # XPath not yet supported: ~w = ~w', [FieldName, Path]).

%% generate_awk_output(+FieldNames, +OutputFormat, -OutputCode)
%  Generate AWK code to format output.
generate_awk_output(FieldNames, dict, Code) :-
    !,
    % Generate dict-like output: _{field1:val1, field2:val2}
    maplist(dict_field_format, FieldNames, FieldFormats),
    atomic_list_concat(FieldFormats, ', ', FormatStr),
    atomic_list_concat(FieldNames, ', ', VarsStr),
    format(atom(Code), '    printf "_{~w}\\n", ~w', [FormatStr, VarsStr]).
generate_awk_output(FieldNames, list, Code) :-
    !,
    % Generate list output: [val1, val2, ...]
    length(FieldNames, N),
    length(Formats, N),
    maplist(=('%s'), Formats),
    atomic_list_concat(Formats, ', ', FormatStr),
    atomic_list_concat(FieldNames, ', ', VarsStr),
    format(atom(Code), '    printf "[~w]\\n", ~w', [FormatStr, VarsStr]).
generate_awk_output(FieldNames, compound(Functor), Code) :-
    !,
    % Generate compound output: functor(val1, val2, ...)
    length(FieldNames, N),
    length(Formats, N),
    maplist(=('%s'), Formats),
    atomic_list_concat(Formats, ', ', FormatStr),
    atomic_list_concat(FieldNames, ', ', VarsStr),
    format(atom(Code), '    printf "~w(~w)\\n", ~w', [Functor, FormatStr, VarsStr]).

dict_field_format(FieldName, Format) :-
    format(atom(Format), '~w:%s', [FieldName]).

%% ============================================
%% ITERPARSE FIELD EXTRACTION (Future)
%% ============================================

compile_iterparse_field_extraction(Pred, File, Tag, FieldSpec, OutputFormat, BashCode) :-
    format('Error: iterparse field extraction not yet implemented~n', []),
    format('  Use engine(awk_pipeline) for now~n', []),
    fail.

%% ============================================
%% UTILITIES
%% ============================================

option(Option, Options, Default) :-
    (   member(Option, Options)
    ->  true
    ;   Option =.. [Functor, Default],
        true
    ).

tmp_file_stream(Type, Path, Stream) :-
    tmp_file('xml_field', Path),
    open(Path, write, Stream).

tmp_file(Prefix, Path) :-
    % Use TMPDIR if available, otherwise current directory
    (   getenv('TMPDIR', TmpDir)
    ->  true
    ;   TmpDir = '.'
    ),
    get_time(Time),
    format(atom(Path), '~w/~w_~w.awk', [TmpDir, Prefix, Time]).
