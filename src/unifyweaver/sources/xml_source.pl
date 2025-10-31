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
    % Must have xml_file
    (   member(xml_file(File), Config),
        (
            exists_file(File)
        ->  true
        ;   format('Warning: XML file ~w does not exist~n', [File])
        )
    ->  true
    ;   format('Error: XML source requires xml_file(File)~n', []),
        fail
    ),
    % Must have tags
    (   member(tags(Tags), Config),
        is_list(Tags),
        Tags \= []
    ->  true
    ;   format('Error: XML source requires non-empty tags(List)~n', []),
        fail
    ),
    % Validate engine if specified
    (   member(engine(Engine), Config)
    ->  (
            (Engine == iterparse ; Engine == xmlstarlet)
        ->  true
        ;   format('Error: invalid engine ~w. Must be one of [iterparse, xmlstarlet]~n', [Engine]),
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
    ;   check_xmlstarlet_available
    ->  Engine = xmlstarlet,
        format('Warning: lxml not found, using xmlstarlet (limited functionality)~n', [])
    ;   format('Error: No XML parsing engine available~n', []),
        format('  Install lxml: pip3 install lxml~n', []),
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
        path(py)-['-3', '-c', 'import lxml']
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
                    ;   format('lxml check via ~q failed (status ~w). stdout: ~s, stderr: ~s~n',
                               [Exec, ExitStatus, OutCodes, ErrCodes]),
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
                ;   format('lxml check via ~q raised exception: ~q~n', [Exec, Error]),
                    fail
                )
            )
        ).
    
    %% check_xmlstarlet_available
    %  Check if xmlstarlet is available.
    check_xmlstarlet_available :-
        catch(
            process_create(path(xmlstarlet),
                          ['--version'],
                          [stdout(null), stderr(null)]),
            _, 
            fail
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

    % Extract required parameters
    member(xml_file(File), AllOptions),
    member(tags(Tags), AllOptions),

    % Detect or use specified engine
    (   member(engine(Engine), AllOptions)
    ->  true
    ;   detect_available_engine(Engine)
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
    ;   Engine == xmlstarlet
    ->  generate_xmlstarlet_bash(Pred, File, Tags, BashCode)
    ).

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
        "null = b'\0'\n\n",
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

%% tags_to_xpath(+Tags, -XPath)
%  Convert a list of tags to an XPath expression.
tags_to_xpath(Tags, XPath) :-
    maplist(tag_to_xpath, Tags, XPathList),
    atomic_list_concat(XPathList, ' | ', XPath).

tag_to_xpath(Tag, XPath) :-
    atom_string(Tag, TagStr),
    string_concat("//", TagStr, XPath).

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

% Template for xmllint engine
template_system:template(xml_xmlstarlet_source, '#!/bin/bash
# {{pred}} - XML source (xmlstarlet)

{{pred}}() {
    xmlstarlet sel -t -c "{{xpath}}" "{{file}}" | awk \'{printf "%s\0", $0}\'
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').
