:- module(playbook_compiler, [
    compile_playbook/2, % +PlaybookIn, -BashScriptOut
    compile_playbook/3  % +PlaybookIn, +Options, -BashScriptOut
]).

% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)

/** <module> A compiler for turning Economic Agent Playbooks into executable scripts.

This compiler performs the following steps:
1. Parses a playbook markdown file to extract strategies, tools, and decision logic.
2. Profiles the associated tools to gather real-world performance metrics (e.g., latency).
3. Injects these metrics into the playbook's economic model.
4. Generates a final, executable bash script that contains the decision tree and all
   necessary tool implementations.

This compiler integrates with UnifyWeaver's existing infrastructure:
- template_system.pl for variable substitution and template rendering
- Follows the pattern established by other source compilers (csv_source, python_source, etc.)
*/

:- use_module(library(shell)).
:- use_module(library(readutil)).
:- use_module('../core/template_system', [
    render_named_template/3,
    render_named_template/4,
    render_template/3
]).

% =============================================================================
% Main Compilation Predicate
% =============================================================================

%% compile_playbook(+PlaybookIn, -BashScriptOut)
%  The main entry point for the compiler with default options.
compile_playbook(PlaybookIn, BashScriptOut) :-
    compile_playbook(PlaybookIn, [], BashScriptOut).

%% compile_playbook(+PlaybookIn, +Options, -BashScriptOut)
%  The main entry point for the compiler with configurable options.
%
%  Options:
%    - profile_tools(true/false) - Whether to profile tools (default: false)
%    - output_format(bash/markdown/both) - Output format (default: bash)
%    - tool_dir(Dir) - Directory containing tool scripts
%    - include_analysis(true/false) - Include cost analysis in output (default: true)
%
compile_playbook(PlaybookIn, Options, BashScriptOut) :-
    format('~`=t Compiling Playbook: ~w ~72|~n', [PlaybookIn]),

    % Step 1: Parse the playbook to extract its structure and logic.
    parse_playbook(PlaybookIn, Options, Playbook),

    % Step 2: Profile the tools defined in the playbook to get real metrics (optional).
    (   memberchk(profile_tools(true), Options)
    ->  profile_tools(Playbook, Options, ProfiledPlaybook)
    ;   ProfiledPlaybook = Playbook
    ),

    % Step 3: Generate the final output (bash script or markdown or both).
    generate_output(ProfiledPlaybook, Options, BashScriptOut),

    format('~`=t Compilation Complete ~72|~n', []).


% =============================================================================
% Step 1: Playbook Parsing
% =============================================================================

%% parse_playbook(+PlaybookIn, +Options, -Playbook)
%  Reads the playbook file and extracts a structured representation.
%
%  The playbook structure contains:
%    - metadata: name, version, description from frontmatter
%    - content: the full markdown content (for natural language target)
%    - tools: parsed tool specifications
%    - strategies: extracted strategy descriptions
%    - prolog_blocks: code blocks marked as ```prolog or ```pseudocode_prolog
%    - template_vars: list of {{VAR}} placeholders found in the content
%
parse_playbook(PlaybookIn, Options, Playbook) :-
    format('~`│t [Parsing] Reading playbook from ~w...~n', [PlaybookIn]),

    % Read the markdown file
    read_file_to_string(PlaybookIn, Content, []),

    % Extract frontmatter (YAML between --- markers)
    extract_frontmatter(Content, Metadata),

    % Extract code blocks (for Prolog logic and tool definitions)
    extract_code_blocks(Content, CodeBlocks),

    % Find template variables ({{VAR | default: value}} or {{VAR}})
    find_template_variables(Content, TemplateVars),

    % Build structured playbook
    Playbook = playbook(
        PlaybookIn,
        [
            metadata(Metadata),
            content(Content),
            code_blocks(CodeBlocks),
            template_vars(TemplateVars)
        ]
    ),

    format('~`│t   Found ~w code blocks, ~w template variables~n',
           [CodeBlocks, TemplateVars]).

%% extract_frontmatter(+Content, -Metadata)
%  Extract YAML frontmatter if present
extract_frontmatter(Content, Metadata) :-
    (   sub_string(Content, 0, _, _, "---\n"),
        sub_string(Content, Start, _, After, "---\n"),
        Start > 0,
        sub_string(Content, End, _, _, "\n---\n"),
        End > Start
    ->  % Extract the YAML section
        Start2 is Start + 4,
        Length is End - Start2,
        sub_string(Content, Start2, Length, _, YAMLStr),
        parse_yaml_simple(YAMLStr, Metadata)
    ;   % No frontmatter
        Metadata = []
    ).

%% parse_yaml_simple(+YAMLString, -KeyValueList)
%  Simple YAML parser for key: value pairs
parse_yaml_simple(YAML, Pairs) :-
    split_string(YAML, "\n", " \t", Lines),
    maplist(parse_yaml_line, Lines, PairLists),
    flatten(PairLists, Pairs).

parse_yaml_line(Line, [Key=Value]) :-
    sub_string(Line, Before, _, After, ":"),
    Before > 0,
    !,
    sub_string(Line, 0, Before, _, KeyStr),
    AfterColon is Before + 1,
    sub_string(Line, AfterColon, After, 0, ValueStr),
    string_trim(KeyStr, Key),
    string_trim(ValueStr, Value).
parse_yaml_line(_, []).  % Skip lines without colons

string_trim(Str, Trimmed) :-
    split_string(Str, "", " \t\n\r", [Trimmed|_]),
    !.
string_trim(_, "").

%% extract_code_blocks(+Content, -CodeBlocks)
%  Extract all code blocks with their language tags
extract_code_blocks(Content, CodeBlocks) :-
    findall(
        block(Lang, Code),
        (   sub_string(Content, Start, _, _, "```"),
            Start1 is Start + 3,
            sub_string(Content, Start1, _, After, "\n"),
            LineEnd is Start1 + (string_length(Content) - Start1 - After - 1),
            sub_string(Content, Start1, _, After, LangLine),
            sub_string(LangLine, 0, _, _, "\n"),
            split_string(LangLine, "\n", "", [Lang|_]),
            % Find the closing ```
            EndMarker is Start1 + string_length(Lang) + 1,
            sub_string(Content, EndMarker, _, _, RestContent),
            sub_string(RestContent, CodeStart, _, CodeAfter, "\n```"),
            sub_string(RestContent, 0, CodeStart, _, Code)
        ),
        CodeBlocks
    ).

%% find_template_variables(+Content, -Variables)
%  Find all {{VAR}} or {{VAR | default: value}} patterns
find_template_variables(Content, Variables) :-
    findall(
        Var,
        (   sub_string(Content, _, _, _, "{{"),
            sub_string(Content, Start, _, _, "{{"),
            Start1 is Start + 2,
            sub_string(Content, Start1, _, After, "}}"),
            Length is string_length(Content) - Start1 - After - 2,
            sub_string(Content, Start1, Length, _, VarContent),
            % Extract variable name (before | or entire content)
            (   sub_string(VarContent, Before, _, _, "|")
            ->  sub_string(VarContent, 0, Before, _, VarName)
            ;   VarName = VarContent
            ),
            string_trim(VarName, Var)
        ),
        VarList
    ),
    list_to_set(VarList, Variables).


% =============================================================================
% Step 2: Tool Profiling
% =============================================================================

%% profile_tools(+Playbook, +Options, -ProfiledPlaybook)
%  Identifies the tools in the playbook and measures their performance.
%
%  This step finds tool scripts, executes them with sample data,
%  measures execution time, and injects the results as metrics.
%
profile_tools(Playbook, Options, ProfiledPlaybook) :-
    format('~`│t [Profiling] Measuring tool performance...~n', []),

    Playbook = playbook(Path, Data),

    % Get tool directory from options or default to proposals/llm_workflows/tools
    (   memberchk(tool_dir(ToolDir), Options)
    ->  true
    ;   ToolDir = 'proposals/llm_workflows/tools'
    ),

    % Find all tool scripts
    (   exists_directory(ToolDir)
    ->  directory_files(ToolDir, Files),
        include(is_shell_script, Files, ToolScripts),
        format('~`│t   Found ~w tool scripts in ~w~n', [length(ToolScripts), ToolDir]),

        % Profile each tool
        maplist(profile_single_tool(ToolDir), ToolScripts, Metrics),

        % Add metrics to playbook data
        ProfiledPlaybook = playbook(Path, [metrics(Metrics)|Data])
    ;   % Tool directory doesn't exist, skip profiling
        format('~`│t   Tool directory ~w not found, skipping profiling~n', [ToolDir]),
        ProfiledPlaybook = Playbook
    ).

%% is_shell_script(+Filename)
%  Check if filename ends with .sh
is_shell_script(File) :-
    atom_string(File, FileStr),
    sub_string(FileStr, _, 3, 0, ".sh").

%% profile_single_tool(+ToolDir, +ToolScript, -Metric)
%  Profile a single tool by measuring its execution time
%  TODO: Actually execute tools with sample inputs
profile_single_tool(ToolDir, ToolScript, metric(ToolName, latency(0), cost(0))) :-
    atom_string(ToolScript, ToolStr),
    sub_string(ToolStr, 0, _, 3, ToolNameStr),  % Remove .sh extension
    atom_string(ToolName, ToolNameStr),
    % For now, return placeholder metrics
    % In the future, this would:
    % 1. Read tool's header to find sample inputs
    % 2. Execute the tool with those inputs
    % 3. Measure execution time
    % 4. Extract cost from tool's header comments
    format('~`│t     Profiled ~w~n', [ToolName]).


% =============================================================================
% Step 3: Output Generation
% =============================================================================

%% generate_output(+ProfiledPlaybook, +Options, -Output)
%  Generates the final output in the requested format.
%
%  Supports multiple output formats:
%    - bash: Executable bash script with embedded playbook
%    - markdown: Resolved markdown with template variables substituted
%
generate_output(Playbook, Options, Output) :-
    (   memberchk(output_format(markdown), Options)
    ->  generate_markdown_output(Playbook, Options, Output)
    ;   % Default: bash
        generate_bash_output(Playbook, Options, Output)
    ).

%% generate_bash_output(+Playbook, +Options, -BashScript)
%  Generates an executable bash script with embedded playbook
generate_bash_output(playbook(PlaybookPath, PlaybookData), _Options, BashScript) :-
    format('~`│t [Generating] Creating bash script...~n', []),

    % Extract data
    memberchk(content(Content), PlaybookData),
    (   memberchk(metadata(Metadata), PlaybookData),
        memberchk(name=PlaybookName, Metadata)
    ->  true
    ;   PlaybookName = PlaybookPath
    ),

    % Render header with embedded playbook content
    render_named_template('playbook/playbook_header',
        [playbook_name=PlaybookName, playbook_content=Content],
        [], Header),

    % TODO: Render tool functions
    ToolsBlock = '# Tool functions would go here\n\n',

    % TODO: Render decision tree
    render_named_template('playbook/strategy_decision_tree',
        [available_strategies='quick_triage,balanced_deep_dive',
         strategy_case_block='"quick_triage") echo "Triage...";;\n'],
        [], DecisionTree),

    % Assemble final script
    atomic_list_concat([Header, ToolsBlock, DecisionTree], BashScript).

%% generate_markdown_output(+Playbook, +Options, -Markdown)
%  Generates a markdown playbook with template variables resolved
generate_markdown_output(playbook(_Path, PlaybookData), _Options, ResolvedMarkdown) :-
    format('~`│t [Generating] Creating markdown playbook...~n', []),

    memberchk(content(Content), PlaybookData),

    % For now, just return content as-is
    % TODO: Substitute template variables using metrics
    ResolvedMarkdown = Content.
