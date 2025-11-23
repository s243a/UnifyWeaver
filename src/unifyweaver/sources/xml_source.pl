:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% xml_source.pl - XML source plugin for dynamic sources
% Compiles predicates that read data from XML files using a streaming parser

:- module(xml_source, [compile_source/4, generate_lxml_python_code/3]).

:- use_module(library(lists)).
:- use_module(library(readutil)).
:- use_module(library(process)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').
:- use_module('../core/perl_service').
:- use_module('../core/firewall_v2').
:- use_module('../core/tool_detection').
:- use_module('xml_field_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(xml, xml_source),
    now
).

%% ============================================ 
%% PLUGIN INTERFACE
%% ============================================ 

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('XML Source'),
    version('1.0.0'),
    description('Extract XML elements using a streaming parser'),
    supported_arities([1])
)).

%% validate_config(+Config)
%  Validate configuration for XML source
validate_config(Config) :-
    % Must have xml_file or file (support both interfaces)
    (   (member(xml_file(File), Config) ; member(file(File), Config)),
        (
            exists_file(File)
        ->  true
        ;   format('Warning: XML file ~w does not exist~n', [File])
        )
    ->  true
    ;   format('Error: XML source requires xml_file(File) or file(File)~n', []),
        fail
    ),
    % Must have tags or tag (support both interfaces)
    (   member(tags(Tags), Config),
        is_list(Tags),
        Tags \= []
    ->  true
    ;   member(tag(Tag), Config),
        atom(Tag)
    ->  true
    ;   format('Error: XML source requires tags(List) or tag(Atom)~n', []),
        fail
    ),
    % Validate engine if specified
    (   member(engine(Engine), Config)
    ->  (
            memberchk(Engine, [iterparse, xmllint, xmlstarlet, awk_pipeline])
        ->  true
        ;   format('Error: invalid engine ~w. Must be one of [iterparse, xmllint, xmlstarlet, awk_pipeline]~n', [Engine]),
            fail
        )
    ;   true
    ),
    % Optional namespace repair flag (bool)
    (   member(namespace_fix(Value), Config)
    ->  (   memberchk(Value, [true, false])
        ->  true
        ;   format('Error: namespace_fix(~w) must be true or false~n', [Value]),
            fail
        )
    ;   true
    ),
    % Optional splitter selection for xmllint
    (   member(xmllint_splitter(Value), Config)
    ->  (   memberchk(Value, [python, perl])
        ->  true
        ;   format('Error: xmllint_splitter(~w) must be python or perl~n', [Value]),
            fail
        )
    ;   true
    ),
    % Optional case-insensitive tag matching (bool)
    (   member(case_insensitive(Value), Config)
    ->  (   memberchk(Value, [true, false])
        ->  true
        ;   format('Error: case_insensitive(~w) must be true or false~n', [Value]),
            fail
        )
    ;   true
    ).

%% ============================================ 
%% ENGINE DETECTION
%% ============================================ 

%% detect_available_engine(-Engine)
%  Auto-detect the best available XML parsing engine.
detect_available_engine(Engine) :-
    (
        check_lxml_available
    ->  Engine = iterparse
    ;   check_xmllint_available
    ->  Engine = xmllint,
        format('Warning: lxml not found, using xmllint streaming fallback~n', [])
    ;   check_xmlstarlet_available
    ->  Engine = xmlstarlet,
        format('Warning: lxml/xmllint not found, using xmlstarlet (limited functionality)~n', [])
    ;   format('Error: No XML parsing engine available~n', []),
        format('  Install lxml: pip3 install lxml~n', []),
        format('  Or install xmllint: apt-get install libxml2-utils~n', []),
        format('  Or install xmlstarlet: apt-get install xmlstarlet~n', []),
        fail
    ).
    
%% check_lxml_available
%  Check if lxml is available.
check_lxml_available :-
    python_lxml_candidates(Candidates),
    member(Exec-Args, Candidates),
    python_lxml_check(Exec, Args),
    !.
check_lxml_available :-
    format('lxml check failed using all configured interpreters.~n', []),
    fail.

python_lxml_candidates([
    path(python3)-['-c', 'import lxml'],
    path(py)-['-3', '-c', 'import lxml'],
    '/usr/bin/python3'-['-c', 'import lxml'],
    '/bin/python3'-['-c', 'import lxml'],
    path(wsl)-['python3', '-c', 'import lxml'],
    'C:\\Windows\\System32\\wsl.exe'-['python3', '-c', 'import lxml'],
    '/mnt/c/Windows/System32/wsl.exe'-['python3', '-c', 'import lxml'],
    '/c/Windows/System32/wsl.exe'-['python3', '-c', 'import lxml']
]).

python_lxml_check(Exec, Args) :-
    catch(
        setup_call_cleanup(
            process_create(Exec, Args, [stdout(pipe(Out)), stderr(pipe(Err)), process(Process)]),
            (
                read_stream_to_codes(Out, OutCodes),
                read_stream_to_codes(Err, ErrCodes),
                process_wait(Process, ExitStatus),
                (
                    ExitStatus = exit(0)
                ->  true
                ;   format('lxml check via ~q failed (status ~w). stdout: ~s, stderr: ~s~n', [Exec, ExitStatus, OutCodes, ErrCodes]),
                    fail
                )
            ),
            (
                close(Out),
                close(Err)
            )
        ),
        Error,
        (
            (   Error = error(existence_error(_, _), _)
            ->  format('lxml check via ~q failed (executable not found)~n', [Exec]),
                fail
            ;   format('lxml check via ~q raised exception: ~q~n', [Exec, Error]),
                fail
            )
        )
    ).

%% check_xmllint_available
%  Check if xmllint is available.
check_xmllint_available :-
    xmllint_candidates(Candidates),
    member(Exec-Args, Candidates),
    xmllint_check(Exec, Args),
    !.
check_xmllint_available :-
    format('xmllint check failed using all configured candidates.~n', []),
    fail.

xmllint_candidates([
    path(xmllint)-['--version'],
    '/usr/bin/xmllint'-['--version'],
    '/bin/xmllint'-['--version'],
    path(wsl)-['xmllint', '--version'],
    'C:\\Windows\\System32\\wsl.exe'-['xmllint', '--version'],
    '/mnt/c/Windows/System32/wsl.exe'-['xmllint', '--version'],
    '/c/Windows/System32/wsl.exe'-['xmllint', '--version']
]).

xmllint_check(Exec, Args) :-
    catch(
        setup_call_cleanup(
            process_create(Exec, Args, [stdout(pipe(Out)), stderr(pipe(Err)), process(Process)]),
            (
                read_stream_to_codes(Out, OutCodes),
                read_stream_to_codes(Err, ErrCodes),
                process_wait(Process, ExitStatus),
                (
                    ExitStatus = exit(0)
                ->  true
                ;   format('xmllint check via ~q failed (status ~w). stdout: ~s, stderr: ~s~n', [Exec, ExitStatus, OutCodes, ErrCodes]),
                    fail
                )
            ),
            (
                close(Out),
                close(Err)
            )
        ),
        Error,
        (
            (   Error = error(existence_error(_, _), _)
            ->  fail
            ;   format('xmllint check via ~q raised exception: ~q~n', [Exec, Error]),
                fail
            )
        )
    ).

%% check_xmlstarlet_available
%  Check if xmlstarlet is available.
check_xmlstarlet_available :-
    xmlstarlet_candidates(Candidates),
    member(Exec-Args, Candidates),
    xmlstarlet_check(Exec, Args),
    !.
check_xmlstarlet_available :-
    format('xmlstarlet check failed using all configured candidates.~n', []),
    fail.

xmlstarlet_candidates([
    path(xmlstarlet)-['--version'],
    '/usr/bin/xmlstarlet'-['--version'],
    '/bin/xmlstarlet'-['--version'],
    path(wsl)-['xmlstarlet', '--version'],
    'C:\\Windows\\System32\\wsl.exe'-['xmlstarlet', '--version'],
    '/mnt/c/Windows/System32/wsl.exe'-['xmlstarlet', '--version'],
    '/c/Windows/System32/wsl.exe'-['xmlstarlet', '--version']
]).

xmlstarlet_check(Exec, Args) :-
    catch(
        setup_call_cleanup(
            process_create(Exec, Args, [stdout(pipe(Out)), stderr(pipe(Err)), process(Process)]),
            (
                read_stream_to_codes(Out, OutCodes),
                read_stream_to_codes(Err, ErrCodes),
                process_wait(Process, ExitStatus),
                (
                    ExitStatus = exit(0)
                ->  true
                ;   format('xmlstarlet check via ~q failed (status ~w). stdout: ~s, stderr: ~s~n', [Exec, ExitStatus, OutCodes, ErrCodes]),
                    fail
                )
            ),
            (
                close(Out),
                close(Err)
            )
        ),
        Error,
        (
            (   Error = error(existence_error(_, _), _)
            ->  fail
            ;   format('xmlstarlet check via ~q raised exception: ~q~n', [Exec, Error]),
                fail
            )
        )
    ).

%% ============================================ 
%% COMPILATION
%% ============================================ 

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile XML source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling XML source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract file (support both interfaces)
    (   member(xml_file(File), AllOptions)
    ->  true
    ;   member(file(File), AllOptions)
    ),

    % Extract tags (support both interfaces)
    (   member(tags(TagList), AllOptions)
    ->  Tags = TagList
    ;   member(tag(SingleTag), AllOptions)
    ->  Tags = [SingleTag]
    ),

    % Check for field extraction
    (   member(fields(FieldSpec), AllOptions)
    ->  % Field extraction requested
        (   Tags = [SingleTag]
        ->  true
        ;   format('Error: Field extraction requires exactly one tag, got: ~w~n', [Tags]),
            fail
        ),
        % Choose implementation based on configuration
        field_extraction_strategy(AllOptions, Strategy),
        (   Strategy = modular
        ->  format('  Using field extraction compiler (modular)~n', []),
            xml_field_compiler:compile_field_extraction(
                Pred/Arity,
                File,
                SingleTag,
                FieldSpec,
                AllOptions,
                BashCode
            )
        ;   Strategy = inline
        ->  format('  Using field extraction (inline)~n', []),
            compile_field_extraction_inline(
                Pred/Arity,
                File,
                SingleTag,
                FieldSpec,
                AllOptions,
                BashCode
            )
        )
    ;   % No fields() - use standard element extraction
        compile_element_extraction(Pred/Arity, File, Tags, AllOptions, BashCode)
    ).

%% field_extraction_strategy(+Options, -Strategy)
%  Determine which field extraction strategy to use.
%  Strategies: modular (delegate to xml_field_compiler) or inline (self-contained)
field_extraction_strategy(Options, Strategy) :-
    (   member(field_compiler(StrategySpec), Options)
    ->  (   memberchk(StrategySpec, [modular, inline])
        ->  Strategy = StrategySpec
        ;   format('Warning: Invalid field_compiler option ~w, using default (modular)~n', [StrategySpec]),
            Strategy = modular
        )
    ;   % Default: use modular approach (Option B)
        Strategy = modular
    ).

%% ============================================
%% INLINE FIELD EXTRACTION (Option A)
%% ============================================

%% compile_field_extraction_inline(+Pred/Arity, +File, +Tag, +FieldSpec, +Options, -BashCode)
%  Inline implementation of field extraction (self-contained in xml_source.pl)
compile_field_extraction_inline(Pred/Arity, File, Tag, FieldSpec, Options, BashCode) :-
    format('  Compiling field extraction (inline): ~w/~w~n', [Pred, Arity]),

    % Validate field specification
    validate_field_spec_inline(FieldSpec),

    % Select output format
    (   member(output(OutputFormat), Options)
    ->  true
    ;   OutputFormat = dict
    ),

    % Generate AWK field extraction script
    generate_awk_field_script_inline(Tag, FieldSpec, OutputFormat, AwkScript),

    % Save AWK script to temp file
    tmp_file_inline('xml_field', TmpAwkFile),
    setup_call_cleanup(
        open(TmpAwkFile, write, AwkStream),
        format(AwkStream, '~w', [AwkScript]),
        close(AwkStream)
    ),

    % Generate bash wrapper using template
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

%% validate_field_spec_inline(+FieldSpec)
validate_field_spec_inline(FieldSpec) :-
    is_list(FieldSpec),
    !,
    maplist(validate_field_entry_inline, FieldSpec).
validate_field_spec_inline(Other) :-
    format('Error: fields() must be a list, got: ~w~n', [Other]),
    fail.

validate_field_entry_inline(FieldName:Selector) :-
    !,
    atom(FieldName),
    atom(Selector).
validate_field_entry_inline(Other) :-
    format('Error: field entry must be FieldName:Selector, got: ~w~n', [Other]),
    fail.

%% generate_awk_field_script_inline(+Tag, +FieldSpec, +OutputFormat, -AwkScript)
generate_awk_field_script_inline(Tag, FieldSpec, OutputFormat, AwkScript) :-
    % Extract field names and selectors
    findall(Name-Selector, member(Name:Selector, FieldSpec), Pairs),
    findall(Name, member(Name:_, FieldSpec), FieldNames),

    % Generate AWK extraction code for each field
    maplist(generate_awk_extractor_inline, Pairs, Extractors),
    atomic_list_concat(Extractors, '\n', ExtractorCode),

    % Generate output formatting code
    generate_awk_output_inline(FieldNames, OutputFormat, OutputCode),

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

%% generate_awk_extractor_inline(+FieldName-TagName, -ExtractorCode)
generate_awk_extractor_inline(FieldName-TagName, Code) :-
    % Handles both CDATA and regular content
    format(atom(Code), '    if (match($0, /<~w><!\\[CDATA\\[([^\\]]+)\\]\\]><\\/~w>/, arr)) { ~w = arr[1] } else { match($0, /<~w>([^<]+)<\\/~w>/, arr); ~w = arr[1] }',
           [TagName, TagName, FieldName, TagName, TagName, FieldName]).

%% generate_awk_output_inline(+FieldNames, +OutputFormat, -OutputCode)
generate_awk_output_inline(FieldNames, dict, Code) :-
    !,
    maplist(dict_field_format_inline, FieldNames, FieldFormats),
    atomic_list_concat(FieldFormats, ', ', FormatStr),
    atomic_list_concat(FieldNames, ', ', VarsStr),
    format(atom(Code), '    printf "_{~w}\\n", ~w', [FormatStr, VarsStr]).
generate_awk_output_inline(FieldNames, list, Code) :-
    !,
    length(FieldNames, N),
    length(Formats, N),
    maplist(=('%s'), Formats),
    atomic_list_concat(Formats, ', ', FormatStr),
    atomic_list_concat(FieldNames, ', ', VarsStr),
    format(atom(Code), '    printf "[~w]\\n", ~w', [FormatStr, VarsStr]).
generate_awk_output_inline(FieldNames, compound(Functor), Code) :-
    !,
    length(FieldNames, N),
    length(Formats, N),
    maplist(=('%s'), Formats),
    atomic_list_concat(Formats, ', ', FormatStr),
    atomic_list_concat(FieldNames, ', ', VarsStr),
    format(atom(Code), '    printf "~w(~w)\\n", ~w', [Functor, FormatStr, VarsStr]).

dict_field_format_inline(FieldName, Format) :-
    format(atom(Format), '~w:%s', [FieldName]).

%% tmp_file_inline(+Prefix, -Path)
tmp_file_inline(Prefix, Path) :-
    (   getenv('TMPDIR', TmpDir)
    ->  true
    ;   TmpDir = '.'
    ),
    get_time(Time),
    format(atom(Path), '~w/~w_~w.awk', [TmpDir, Prefix, Time]).

%% ============================================
%% STANDARD ELEMENT EXTRACTION
%% ============================================

%% compile_element_extraction(+Pred/Arity, +File, +Tags, +Options, -BashCode)
%  Compile standard element extraction (without field extraction)
compile_element_extraction(Pred/Arity, File, Tags, AllOptions, BashCode) :-

    namespace_fix_option(AllOptions, NamespaceFix),
    xmllint_splitter_option(AllOptions, Splitter),

    % Detect or use specified engine
    (
        member(engine(Engine), AllOptions)
    ->  true
    ;   detect_available_engine(Engine)
    ),
    (   Engine == xmllint
    ->  check_xmllint_available,
        (Splitter == perl -> check_perl_available ; true)
    ;   true
    ),

    % Generate code based on engine
    (
        Engine == iterparse
    ->  generate_lxml_python_code(File, Tags, PythonCode),
        atom_string(Pred, PredStr),
        render_named_template(xml_iterparse_source,
            [pred=PredStr, python_code=PythonCode],
            [source_order([file, generated])],
            BashCode)
    ;   Engine == xmllint
    ->  generate_xmllint_bash(Pred, File, Tags, NamespaceFix, Splitter, BashCode)
    ;   Engine == xmlstarlet
    ->  generate_xmlstarlet_bash(Pred, File, Tags, BashCode)
    ;   Engine == awk_pipeline
    ->  generate_awk_pipeline_bash(Pred, File, Tags, AllOptions, BashCode)
    ).

%% namespace_fix_option(+Options, -FixBool)
namespace_fix_option(Options, Fix) :-
    (
        member(namespace_fix(Value), Options)
    ->  Fix = Value
    ;   Fix = true
    ).

%% xmllint_splitter_option(+Options, -Splitter)
%  Determine which splitter implementation to use for xmllint, respecting
%  explicit options, preference ordering, firewall policies, and tool availability.
xmllint_splitter_option(Options, Splitter) :-
    (   member(xmllint_splitter(Value), Options)
    ->  ensure_splitter_allowed(Value),
        Splitter = Value
    ;   splitter_candidate_order(Options, Candidates),
        select_available_splitter(Candidates, Splitter)
    ).

ensure_splitter_allowed(Splitter) :-
    splitter_status(Splitter, available),
    !.
ensure_splitter_allowed(Splitter) :-
    splitter_status(Splitter, Status),
    report_splitter_error(Splitter, Status),
    fail.

splitter_candidate_order(Options, Order) :-
    preferred_splitter_values(Options, Preferred),
    default_splitters(Default),
    append(Preferred, Default, Combined),
    dedupe_splitter_order(Combined, Order).

default_splitters([python, perl]).

preferred_splitter_values(Options, Values) :-
    findall(Value,
        (   member(prefer(List), Options),
            member(Entry, List),
            splitter_preference_entry(Entry, Value)
        ),
        RawValues),
    dedupe_splitter_order(RawValues, Values).

splitter_preference_entry(xmllint_splitter(Value), Value) :-
    splitter_tool(Value, _).

splitter_preference_entry(Value, Value) :-
    splitter_tool(Value, _).

dedupe_splitter_order(List, Deduped) :-
    dedupe_splitter_order(List, [], Deduped).

dedupe_splitter_order([], _Seen, []).
dedupe_splitter_order([H|T], Seen, [H|Rest]) :-
    splitter_tool(H, _),
    \+ memberchk(H, Seen),
    dedupe_splitter_order(T, [H|Seen], Rest).
dedupe_splitter_order([H|T], Seen, Rest) :-
    (   splitter_tool(H, _)
    ->  memberchk(H, Seen)
    ;   true
    ),
    dedupe_splitter_order(T, Seen, Rest).

select_available_splitter([Candidate|_], Splitter) :-
    splitter_status(Candidate, available),
    !,
    Splitter = Candidate.
select_available_splitter([Candidate|Rest], Splitter) :-
    splitter_status(Candidate, Status),
    (   Status = denied(_) ; Status = unavailable(_)
    ->  true
    ;   true
    ),
    select_available_splitter(Rest, Splitter).
select_available_splitter([], _) :-
    format(user_error, 'Error: No xmllint splitter is available or permitted (tried python and perl).~n', []),
    fail.

splitter_status(Splitter, Status) :-
    splitter_tool(Splitter, Tool),
    (   catch(firewall_v2:check_tool_availability(Tool, bash, Result), _, fail)
    ->  Status = Result
    ;   tool_detection:detect_tool_availability(Tool, DetectStatus),
        (   DetectStatus = available
        ->  Status = available
        ;   Status = DetectStatus
        )
    ).

splitter_tool(python, python3).
splitter_tool(perl, perl).

report_splitter_error(Splitter, denied(Reason)) :-
    format(user_error,
           'Error: xmllint splitter "~w" denied by firewall (~w).~n',
           [Splitter, Reason]).
report_splitter_error(Splitter, unavailable(Reason)) :-
    format(user_error,
           'Error: xmllint splitter "~w" unavailable (~w).~n',
           [Splitter, Reason]).
report_splitter_error(Splitter, Status) :-
    format(user_error,
           'Error: xmllint splitter "~w" cannot be selected (status: ~w).~n',
           [Splitter, Status]).

%% ============================================ 
%% PYTHON CODE GENERATION
%% ============================================ 

%% generate_lxml_python_code(+File, +Tags, -PythonCode)
%  Generate Python code for lxml iterparse
generate_lxml_python_code(File, Tags, PythonCode) :-
    tags_to_python_set(Tags, TagsSet),
    atomic_list_concat([
        "import sys\n",
        "from lxml import etree\n\n",
        "file = \"", File, "\"\n",
        "tags = {", TagsSet, "}\n",
        "null = b'\\0'\n\n",
        "# Parse with namespace handling\n",
        "context = etree.iterparse(file, events=('start', 'end'), recover=True)\n",
        "event, root = next(context)\n",
        "nsmap = root.nsmap or {}\n\n",
        "def expand(tag):\n",
        "    if ':' in tag:\n",
        "        pfx, local = tag.split(':', 1)\n",
        "        uri = nsmap.get(pfx)\n",
        "        if uri:\n",
        "            return '{' + uri + '}' + local\n",
        "        else:\n",
        "            return tag\n",
        "    return tag\n\n",
        "want = {expand(t) for t in tags}\n\n",
        "# Stream with memory release\n",
        "for event, elem in context:\n",
        "    if event == 'end' and elem.tag in want:\n",
        "        sys.stdout.buffer.write(etree.tostring(elem))\n",
        "        sys.stdout.buffer.write(null)\n",
        "        # Release memory immediately\n",
        "        elem.clear()\n",
        "        while elem.getprevious() is not None:\n",
        "            del elem.getparent()[0]\n",
        "        root.clear()\n"
    ], PythonCode).

%% tags_to_python_set(+Tags, -Set)
%  Convert a list of Prolog atoms to a Python set string.
tags_to_python_set(Tags, Set) :-
    maplist(quote_atom, Tags, QuotedTags),
    atomic_list_concat(QuotedTags, ', ', Set).

%% tags_to_python_list(+Tags, -List)
%  Convert a list of Prolog atoms to a Python list string.
tags_to_python_list(Tags, List) :-
    maplist(quote_atom, Tags, QuotedTags),
    atomic_list_concat(QuotedTags, ', ', Inner),
    format(atom(List), '[~w]', [Inner]).

%% quote_atom(+Atom, -Quoted)
%  Quote a Prolog atom for Python.
quote_atom(Atom, Quoted) :-
    format(atom(Quoted), '\'~w\'', [Atom]).

%% generate_xmlstarlet_bash(+Pred, +File, +Tags, -BashCode)
%  Generate bash code for xmlstarlet engine
generate_xmlstarlet_bash(Pred, File, Tags, BashCode) :-
    tags_to_xpath(Tags, XPath),
    atom_string(Pred, PredStr),
    render_named_template(xml_xmlstarlet_source,
        [pred=PredStr, file=File, xpath=XPath],
        [source_order([file, generated])],
        BashCode).

%% generate_xmllint_bash(+Pred, +File, +Tags, +NamespaceFix, +Splitter, -BashCode)
%  Generate bash code for xmllint engine. The splitter can be `python` (default)
%  or `perl` to use the inline Perl service.
generate_xmllint_bash(Pred, File, Tags, NamespaceFix, Splitter, BashCode) :-
    tags_to_xmllint_xpath(Tags, XPath),
    atom_string(Pred, PredStr),
    (   Splitter == perl
    ->  namespace_map_shell(NamespaceMapShell),
        boolean_sh_literal(NamespaceFix, RepairLiteralSh),
        read_file_to_string('scripts/xml_splitter.pl', PerlScript, []),
        generate_inline_perl_call(PerlScript, [RepairLiteralSh, NamespaceMapShell | Tags], stdin, PerlCall0),
        indent_block(PerlCall0, 8, PerlCall),
        render_named_template(xml_xmllint_perl_source,
            [pred=PredStr,
             file=File,
             xpath=XPath,
             perl_call=PerlCall],
            [source_order([file, generated])],
            BashCode)
    ;   tags_to_python_list(Tags, TagsList),
        namespace_map_python(NamespaceMapPy),
        boolean_py_literal(NamespaceFix, RepairLiteralPy),
        render_named_template(xml_xmllint_source,
            [pred=PredStr,
             file=File,
             xpath=XPath,
             tags_py_list=TagsList,
             namespace_map_py=NamespaceMapPy,
             repair_flag_py=RepairLiteralPy],
            [source_order([file, generated])],
            BashCode)
    ).

%% namespace_map_shell(-Map)
%  Produce a semicolon-separated string for known namespaces.
namespace_map_shell(Map) :-
    findall(Entry, known_namespace_entry_shell(Entry), Entries),
    atomic_list_concat(Entries, ';', Map).

%% namespace_map_python(-Dict)
%  Produce a Python dictionary literal for known namespaces.
namespace_map_python(Dict) :-
    findall(Entry, known_namespace_entry(Entry), Entries),
    atomic_list_concat(Entries, ', ', Inner),
    format(atom(Dict), '{~w}', [Inner]).

known_namespace_entry_shell(Entry) :-
    known_namespace(Prefix, Uri),
    format(atom(Entry), '~w=~w', [Prefix, Uri]).

known_namespace_entry(Entry) :-
    known_namespace(Prefix, Uri),
    atom_string(Prefix, PrefixStr),
    atom_string(Uri, UriStr),
    format(atom(Entry), '\'~s\': \'~s\'', [PrefixStr, UriStr]).

%% boolean_sh_literal(+Bool, -Literal)
boolean_sh_literal(true, 'true').
boolean_sh_literal(false, 'false').

%% boolean_py_literal(+Bool, -Literal)
boolean_py_literal(true, 'True').
boolean_py_literal(false, 'False').

%% tags_to_xpath(+Tags, -XPath)
%  Convert a list of tags to an XPath expression.
tags_to_xpath(Tags, XPath) :-
    maplist(tag_to_xpath, Tags, XPathList),
    atomic_list_concat(XPathList, ' | ', XPath).

tag_to_xpath(Tag, XPath) :-
    atom_string(Tag, TagStr),
    string_concat('//', TagStr, XPath).

%% tags_to_xmllint_xpath(+Tags, -XPath)
%  Convert tags to an XPath expression suitable for xmllint.
tags_to_xmllint_xpath(Tags, XPath) :-
    maplist(tag_to_xmllint_expr, Tags, XPathList),
    atomic_list_concat(XPathList, ' | ', XPath).

tag_to_xmllint_expr(Tag, XPath) :-
    (
        sub_atom(Tag, _, _, _, ':')
    ->  atomic_list_concat([Prefix, Local], ':', Tag),
        (
            known_namespace(Prefix, Uri)
        ->  format(atom(XPath), '//*[local-name()="~w" and namespace-uri()="~w"]', [Local, Uri])
        ;   format(atom(XPath), '//*[name()="~w"]', [Tag])
        )
    ;
        format(atom(XPath), '//~w', [Tag])
    ).

%% indent_block(+Code, +Spaces, -Indented)
%  Indent each line of Code by the given number of spaces, preserving heredoc
%  terminators (last line stays unindented). Trailing blank lines are retained.
indent_block(Code, Spaces, Indented) :-
    atom_string(Code, String),
    split_string(String, "\n", "", RawLines),
    strip_trailing_empty(RawLines, Lines, HadEmpty),
    (   Lines = []
    ->  IndentedLines = []
    ;   append(BodyLines, [LastLine], Lines),
        indent_body(BodyLines, Spaces, BodyIndented),
        append(BodyIndented, [LastLine], IndentedLines)
    ),
    (   HadEmpty == true
    ->  append(IndentedLines, [''], LinesWithEmpty)
    ;   LinesWithEmpty = IndentedLines
    ),
    atomic_list_concat(LinesWithEmpty, '\n', Indented).

indent_body(Lines, Spaces, IndentedLines) :-
    length(SpacesList, Spaces),
    maplist(=(' '), SpacesList),
    atomic_list_concat(SpacesList, Indent),
    maplist(indent_line(Indent), Lines, IndentedLines).

strip_trailing_empty(Lines0, Lines, true) :-
    append(Lines, [""], Lines0),
    !.
strip_trailing_empty(Lines, Lines, false).

indent_line(_, "", "") :- !.
indent_line(Indent, Line, IndentedLine) :-
    atomic_list_concat([Indent, Line], IndentedLine).


%% generate_awk_pipeline_bash(+Pred, +File, +Tags, +AllOptions, -BashCode)
%  Generate bash code for awk_pipeline engine (uses awk + Python streaming)
generate_awk_pipeline_bash(Pred, File, Tags, AllOptions, BashCode) :-
    % Convert tags list to single tag pattern for awk
    % For now, use first tag (TODO: support multiple tags)
    (   Tags = [FirstTag|_]
    ->  TagPattern = FirstTag
    ;   Tags = []
    ->  TagPattern = ''
    ),

    % Check for parallel mode
    (   member(parallel(true), AllOptions)
    ->  (   member(workers(Workers), AllOptions)
        ->  true
        ;   Workers = 4  % Default workers
        ),
        atom_string(Pred, PredStr),
        render_named_template(xml_awk_pipeline_parallel_source,
            [pred=PredStr, file=File, tag=TagPattern, workers=Workers],
            [source_order([file, generated])],
            BashCode)
    ;   atom_string(Pred, PredStr),
        (   member(case_insensitive(true), AllOptions)
        ->  Icase = 1
        ;   Icase = 0
        ),
        render_named_template(xml_awk_pipeline_source,
            [pred=PredStr, file=File, tag=TagPattern, icase=Icase],
            [source_order([file, generated])],
            BashCode)
    ).

%% known_namespace(+Prefix, -URI)
known_namespace(pt, 'http://www.pearltrees.com/xmlns/pearl-trees#').
known_namespace(rdf, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#').
known_namespace(dcterms, 'http://purl.org/dc/terms/').

%% ============================================ 
%% BASH TEMPLATES
%% ============================================ 

:- multifile template_system:template/2.

% Template for lxml iterparse engine
template_system:template(xml_iterparse_source, '#!/bin/bash
# {{pred}} - XML source (lxml iterparse)

{{pred}}() {
    python3 /dev/fd/3 3<<\'EOF\'
{{python_code}}
EOF
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

template_system:template(xml_xmllint_source, Template) :-
    atomic_list_concat([
        '#!/bin/bash\n',
        '# {{pred}} - XML source (xmllint)\n',
        '\n',
        '{{pred}}() {\n',
        '    local resolved=""\n',
        '    local -a cmd=()\n',
        '\n',
        '    if resolved=$(command -v xmllint 2>/dev/null); then\n',
        '        cmd=("$resolved")\n',
        '    elif [[ -x /usr/bin/xmllint ]]; then\n',
        '        cmd=("/usr/bin/xmllint")\n',
        '    elif resolved=$(command -v wsl 2>/dev/null); then\n',
        '        cmd=("$resolved" "xmllint")\n',
        '    elif [[ -x /mnt/c/Windows/System32/wsl.exe ]]; then\n',
        '        cmd=("/mnt/c/Windows/System32/wsl.exe" "xmllint")\n',
        '    elif [[ -x /c/Windows/System32/wsl.exe ]]; then\n',
        '        cmd=("/c/Windows/System32/wsl.exe" "xmllint")\n',
        '    else\n',
        '        echo "xmllint not found; install libxml2-utils or adjust PATH." >&2\n',
        '        return 127\n',
        '    fi\n',
        '\n',
        '    python3 - <<\'PY\' "${cmd[@]}" -- \'{{file}}\' \'{{xpath}}\'\n',
        'import re\n',
        'import subprocess\n',
        'import sys\n',
        '\n',
        'TAGS = {{tags_py_list}}\n',
        'NAMESPACE_MAP = {{namespace_map_py}}\n',
        'REPAIR = {{repair_flag_py}}\n',
        '\n',
        'def run_xmllint(cmd, xml_file, xpath):\n',
        '    proc = subprocess.run(cmd + ["--xpath", xpath, xml_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n',
        '    if proc.returncode == 0:\n',
        '        return proc.stdout\n',
        '    if proc.returncode == 10 and "XPath set is empty" in proc.stderr:\n',
        '        return ""\n',
        '    sys.stderr.write(proc.stderr)\n',
        '    raise SystemExit(proc.returncode)\n',
        '\n',
        'def extract_records(data, tags):\n',
        '    records = []\n',
        '    pos = 0\n',
        '    length = len(data)\n',
        '    while True:\n',
        '        next_start = None\n',
        '        next_tag = None\n',
        '        for tag in tags:\n',
        '            marker = f"<{tag}"\n',
        '            idx = data.find(marker, pos)\n',
        '            if idx != -1 and (next_start is None or idx < next_start):\n',
        '                next_start = idx\n',
        '                next_tag = tag\n',
        '        if next_start is None:\n',
        '            break\n',
        '        end_token = f"</{next_tag}>"\n',
        '        end_idx = data.find(end_token, next_start)\n',
        '        if end_idx == -1:\n',
        '            break\n',
        '        end_idx += len(end_token)\n',
        '        tail = end_idx\n',
        '        while tail < length and data[tail] in "\\r\\n\\t ":\n',
        '            tail += 1\n',
        '        records.append(data[next_start:tail])\n',
        '        pos = tail\n',
        '    return records\n',
        '\n',
        'def ensure_namespaces(record):\n',
        '    first_gt = record.find(">")\n',
        '    if first_gt == -1:\n',
        '        return record\n',
        '    header = record[:first_gt]\n',
        '    needed = []\n',
        '    prefixes = set(re.findall(r"[<\\s]([A-Za-z_][\\w.-]*):", record))\n',
        '    for prefix in sorted(prefixes):\n',
        '        if prefix == "xml":\n',
        '            continue\n',
        '        uri = NAMESPACE_MAP.get(prefix)\n',
        '        if not uri:\n',
        '            continue\n',
        '        if f"xmlns:{prefix}=" not in header:\n',
        '            needed.append(f"xmlns:{prefix}={chr(34)}{uri}{chr(34)}")\n',
        '    if not needed:\n',
        '        return record\n',
        '    insert_pos = record.find(" ")\n',
        '    if insert_pos == -1 or insert_pos > first_gt:\n',
        '        insert_pos = first_gt\n',
        '    insertion = " " + " ".join(needed)\n',
        '    return record[:insert_pos] + insertion + record[insert_pos:]\n',
        '\n',
        'def main():\n',
        '    argv = sys.argv[1:]\n',
        '    if "--" not in argv:\n',
        '        raise SystemExit("internal error: missing separator")\n',
        '    sep = argv.index("--")\n',
        '    cmd = argv[:sep]\n',
        '    xml_file = argv[sep + 1]\n',
        '    xpath = argv[sep + 2]\n',
        '    data = run_xmllint(cmd, xml_file, xpath)\n',
        '    if not data:\n',
        '        return\n',
        '    for record in extract_records(data, TAGS):\n',
        '        fixed = ensure_namespaces(record) if REPAIR else record\n',
        '        sys.stdout.buffer.write(fixed.encode("utf-8"))\n',
        '        sys.stdout.buffer.write(b"\\0")\n',
        '\n',
        'if __name__ == "__main__":\n',
        '    main()\n',
        'PY\n',
        '}\n',
        '\n',
        '{{pred}}_stream() {\n',
        '    {{pred}}\n',
        '}\n',
        '\n',
        'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then\n',
        '    {{pred}} "$@"\n',
        'fi\n'
    ], Template).
template_system:template(xml_xmllint_perl_source, Template) :-
    atomic_list_concat([
        '#!/bin/bash\n',
        '# {{pred}} - XML source (xmllint with Perl splitter)\n',
        '\n',
        '{{pred}}() {\n',
        '    local resolved=""\n',
        '    local -a cmd=()\n',
        '\n',
        '    if resolved=$(command -v xmllint 2>/dev/null); then\n',
        '        cmd=("$resolved")\n',
        '    elif [[ -x /usr/bin/xmllint ]]; then\n',
        '        cmd=("/usr/bin/xmllint")\n',
        '    elif resolved=$(command -v wsl 2>/dev/null); then\n',
        '        cmd=("$resolved" "xmllint")\n',
        '    elif [[ -x /mnt/c/Windows/System32/wsl.exe ]]; then\n',
        '        cmd=("/mnt/c/Windows/System32/wsl.exe" "xmllint")\n',
        '    elif [[ -x /c/Windows/System32/wsl.exe ]]; then\n',
        '        cmd=("/c/Windows/System32/wsl.exe" "xmllint")\n',
        '    else\n',
        '        echo "xmllint not found; install libxml2-utils or adjust PATH." >&2\n',
        '        return 127\n',
        '    fi\n',
        '\n',
        '    local tmp_err\n',
        '    local tmp_out\n',
        '    tmp_err=$(mktemp)\n',
        '    tmp_out=$(mktemp)\n',
        '\n',
        '    if ! "${cmd[@]}" --xpath \'{{xpath}}\' \'{{file}}\' >"$tmp_out" 2>"$tmp_err"; then\n',
        '        local status=$?\n',
        '        if [[ $status -eq 10 ]] && grep -q "XPath set is empty" "$tmp_err"; then\n',
        '            rm -f "$tmp_err" "$tmp_out"\n',
        '            return 0\n',
        '        fi\n',
        '        cat "$tmp_err" >&2\n',
        '        rm -f "$tmp_err" "$tmp_out"\n',
        '        return $status\n',
        '    fi\n',
        '\n',
        '    rm -f "$tmp_err"\n',
        '    if [[ ! -s "$tmp_out" ]]; then\n',
        '        rm -f "$tmp_out"\n',
        '        return 0\n',
        '    fi\n',
        '\n',
        '    if ! {\n',
        '{{perl_call}}\n',
        '    } < "$tmp_out"; then\n',
        '        local status=$?\n',
        '        rm -f "$tmp_out"\n',
        '        return $status\n',
        '    fi\n',
        '\n',
        '    rm -f "$tmp_out"\n',
        '    return 0\n',
        '}\n',
        '\n',
        '{{pred}}_stream() {\n',
        '    {{pred}}\n',
        '}\n',
        '\n',
        'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then\n',
        '    {{pred}} "$@"\n',
        'fi\n'
    ], Template).

% Template for xmlstarlet engine
template_system:template(xml_xmlstarlet_source, Template) :-
    atomic_list_concat([
        '#!/bin/bash\n',
        '# {{pred}} - XML source (xmlstarlet)\n',
        '\n',
        '{{pred}}() {\n',
        '    local resolved=""\n',
        '    local -a cmd=()\n',
        '\n',
        '    if resolved=$(command -v xmlstarlet 2>/dev/null); then\n',
        '        cmd=($resolved)\n',
        '    elif [[ -x /usr/bin/xmlstarlet ]]; then\n',
        '        cmd=("/usr/bin/xmlstarlet")\n',
        '    elif resolved=$(command -v wsl 2>/dev/null); then\n',
        '        cmd=($resolved "xmlstarlet")\n',
        '    elif [[ -x /mnt/c/Windows/System32/wsl.exe ]]; then\n',
        '        cmd=("/mnt/c/Windows/System32/wsl.exe" "xmlstarlet")\n',
        '    elif [[ -x /c/Windows/System32/wsl.exe ]]; then\n',
        '        cmd=("/c/Windows/System32/wsl.exe" "xmlstarlet")\n',
        '    else\n',
        '        echo "xmlstarlet not found; install xmlstarlet or switch to the iterparse engine." >&2\n',
        '        return 127\n',
        '    fi\n',
        '\n',
        '    local sentinel="__XMLSTARLET_RECORD__"\n',
        '\n',
        '    "${cmd[@]}" sel -N pt="http://www.pearltrees.com/xmlns/pearl-trees#" -t -m "{{xpath}}" -c "." -o "${sentinel}" -n "{{file}}" |\\\n',
        '        awk -v RS="${sentinel}\\n" -v ORS="" ''length($0) {printf "%s%c", $0, 0}''\n',
        '}\n',
        '\n',
        '{{pred}}_stream() {\n',
        '    {{pred}}\n',
        '}\n',
        '\n',
        'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then\n',
        '    {{pred}} "$@"\n',
        'fi\n'
    ], Template).

% Template for awk_pipeline engine (sequential)
template_system:template(xml_awk_pipeline_source, '#!/bin/bash
# {{pred}} - XML source (awk pipeline)

{{pred}}() {
    awk -f scripts/utils/select_xml_elements.awk -v tag="{{tag}}" -v icase={{icase}} "{{file}}"
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

% Template for awk_pipeline engine (parallel)
template_system:template(xml_awk_pipeline_parallel_source, '#!/bin/bash
# {{pred}} - XML source (awk pipeline parallel)

{{pred}}() {
    bash scripts/extract_pearltrees_parallel.sh "{{file}}" "$@" --workers={{workers}}
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').
