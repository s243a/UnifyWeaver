:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% csv_source.pl - CSV/TSV source plugin for dynamic sources
% Compiles predicates that read data from CSV/TSV files

% Export nothing - all access goes through plugin registry
:- module(csv_source, []).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(csv, csv_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('CSV/TSV Source'),
    version('1.0.0'),
    description('Read data from CSV and TSV files with header auto-detection'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% validate_config(+Config)
%  Validate configuration for CSV source
validate_config(Config) :-
    % Must have csv_file
    (   member(csv_file(File), Config),
        (   exists_file(File)
        ->  true
        ;   format('Warning: CSV file ~w does not exist~n', [File])
        )
    ->  true
    ;   format('Error: CSV source requires csv_file(File)~n', []),
        fail
    ),
    
    % Validate delimiter if specified
    (   member(delimiter(Delim), Config)
    ->  (   atom_length(Delim, 1)
        ->  true
        ;   format('Error: delimiter must be single character, got ~w~n', [Delim]),
            fail
        )
    ;   true
    ),
    
    % Validate skip_lines if specified
    (   member(skip_lines(N), Config)
    ->  (   integer(N), N >= 0
        ->  true
        ;   format('Error: skip_lines must be non-negative integer, got ~w~n', [N]),
            fail
        )
    ;   true
    ),

    % Validate project_columns if specified (additive column-subset projection)
    (   member(project_columns(Proj), Config)
    ->  (   is_list(Proj), Proj \== []
        ->  true
        ;   format('Error: project_columns must be a non-empty list of column names, got ~w~n', [Proj]),
            fail
        )
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile CSV source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling CSV source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract required parameters
    member(csv_file(CsvFile), AllOptions),
    
    % Extract optional parameters with defaults
    (   member(delimiter(Delimiter), AllOptions)
    ->  true
    ;   Delimiter = ','  % Default delimiter
    ),
    (   member(skip_lines(SkipLines), AllOptions)
    ->  true
    ;   SkipLines = 0
    ),
    (   member(quote_char(QuoteChar), AllOptions)
    ->  true
    ;   QuoteChar = '"'
    ),
    (   member(quote_handling(QuoteHandling), AllOptions)
    ->  true
    ;   QuoteHandling = strip
    ),

    % Determine if we have header auto-detection
    (   member(has_header(true), AllOptions)
    ->  HeaderMode = auto,
        detect_csv_headers(CsvFile, Delimiter, DetectedColumns),
        (   member(project_columns(ProjNames), AllOptions)
        % Column-subset projection: arity is the projected count, which is
        % expected to differ from the full header width. Use the projected
        % names as the documented column list; no arity-mismatch warning.
        ->  Columns = ProjNames
        ;   length(DetectedColumns, DetectedArity),
            DetectedArity =:= Arity
        ->  Columns = DetectedColumns
        ;   format('Warning: Detected ~w columns but arity is ~w~n', [DetectedColumns, Arity]),
            generate_default_columns(Arity, Columns)
        )
    ;   member(columns(ManualColumns), AllOptions)
    ->  HeaderMode = manual,
        Columns = ManualColumns,
        length(Columns, ManualArity),
        (   ManualArity =:= Arity
        ->  true
        ;   format('Error: columns list length (~w) does not match arity (~w)~n', [ManualArity, Arity]),
            fail
        )
    ;   HeaderMode = positional,
        generate_default_columns(Arity, Columns)
    ),

    % Generate bash code using template
    atom_string(Pred, PredStr),
    generate_csv_bash(PredStr, Arity, CsvFile, Delimiter, SkipLines,
                     QuoteChar, QuoteHandling, HeaderMode, Columns, AllOptions, BashCode).

%% ============================================
%% HEADER DETECTION
%% ============================================

%% detect_csv_headers(+File, +Delimiter, -Headers)
%  Auto-detect column headers from first line of CSV file
detect_csv_headers(File, Delimiter, Headers) :-
    catch(
        (   open(File, read, Stream),
            read_line_to_string(Stream, FirstLine),
            close(Stream),
            split_string(FirstLine, Delimiter, ' "', HeaderStrings),
            maplist(string_to_atom, HeaderStrings, Headers)
        ),
        Error,
        (   format('Error reading CSV headers from ~w: ~w~n', [File, Error]),
            Headers = []
        )
    ).

%% generate_default_columns(+Arity, -Columns)
%  Generate default column names: col1, col2, etc.
generate_default_columns(Arity, Columns) :-
    numlist(1, Arity, Numbers),
    maplist(default_column_name, Numbers, Columns).

default_column_name(N, ColName) :-
    format(atom(ColName), 'col~w', [N]).

%% ============================================
%% BASH CODE GENERATION
%% ============================================

%% generate_csv_bash(+PredStr, +Arity, +File, +Delimiter, +SkipLines,
%%                   +QuoteChar, +QuoteHandling, +HeaderMode, +Columns, +Options, -Code)
%  Generate bash code for CSV source (or pure PowerShell if template_suffix specified)
generate_csv_bash(PredStr, Arity, File, Delimiter, SkipLines,
                  QuoteChar, QuoteHandling, _HeaderMode, Columns, Options, Code) :-

    % TotalSkip is just SkipLines - sources.pl already adds 1 for headers
    % when has_header(true) is set (see augment_csv_options/2)
    TotalSkip = SkipLines,

    % Generate column output format
    generate_output_format(Arity, OutputFormat),

    % Generate quote handling code
    generate_quote_handling_code(QuoteChar, QuoteHandling, QuoteCode),

    % Escape delimiter for awk
    escape_delimiter(Delimiter, EscapedDelimiter),

    % Escape delimiter for a JS double-quoted string literal (TypeScript/Node
    % templates split on this via String.prototype.split with a *string*
    % argument, so no RegExp escaping is needed - only string-literal escaping).
    js_escape_delimiter(Delimiter, JsDelimiter),

    % Create column list for comments
    atomic_list_concat(Columns, ', ', ColumnList),

    % Column-subset projection (G-P9 residual, additive). Resolve any
    % project_columns([...]) names to 0-based header indices and build the
    % TypeScript-only template vars. Empty ProjIndices => pass-through defaults
    % that reproduce the pre-projection template bytes exactly, so bash / PS /
    % non-projecting TS output are unchanged.
    resolve_projection(Options, File, Delimiter, Arity, ProjIndices),
    projection_ts_vars(ProjIndices, Arity, ProjVars),

    % Determine template name with optional suffix
    (   member(template_suffix(Suffix), Options)
    ->  atom_concat(csv_source_unary, Suffix, UnaryTemplate),
        atom_concat(csv_source_binary_plus, Suffix, BinaryTemplate)
    ;   UnaryTemplate = csv_source_unary,
        BinaryTemplate = csv_source_binary_plus
    ),

    % Render template based on arity
    (   Arity =:= 1 ->
        append([pred=PredStr, file=File, delimiter=EscapedDelimiter,
                js_delimiter=JsDelimiter,
                skip_lines=TotalSkip, quote_code=QuoteCode,
                columns=ColumnList],
               ProjVars, UnaryVars),
        render_named_template(UnaryTemplate,
            UnaryVars,
            [source_order([file, generated])],
            Code)
    ;   append([pred=PredStr, file=File, delimiter=EscapedDelimiter,
                js_delimiter=JsDelimiter,
                skip_lines=TotalSkip, quote_code=QuoteCode,
                output_format=OutputFormat, columns=ColumnList, arity=Arity],
               ProjVars, BinaryVars),
        render_named_template(BinaryTemplate,
            BinaryVars,
            [source_order([file, generated])],
            Code)
    ).

%% generate_output_format(+Arity, -Format)
%  Generate awk output format for given arity
generate_output_format(Arity, Format) :-
    numlist(1, Arity, Numbers),
    maplist(field_reference, Numbers, Fields),
    atomic_list_concat(Fields, '":"', Format).

field_reference(N, Field) :-
    format(atom(Field), '$~w', [N]).

%% generate_quote_handling_code(+QuoteChar, +Handling, -Code)
%  Generate awk code for quote handling
generate_quote_handling_code(QuoteChar, strip, Code) :-
    format(atom(Code), 'gsub(/\\~w/, "", $0)', [QuoteChar]).
generate_quote_handling_code(_QuoteChar, preserve, '') :-
    % No quote handling - preserve as-is
    true.
generate_quote_handling_code(QuoteChar, escape, Code) :-
    format(atom(Code), 'gsub(/~w/, "\\~w", $0)', [QuoteChar, QuoteChar]).

%% escape_delimiter(+Delimiter, -Escaped)
%  Escape delimiter for awk field separator
escape_delimiter('\t', '\\t') :- !.  % Tab
escape_delimiter('|', '\\|') :- !.   % Pipe
escape_delimiter('\\', '\\\\') :- !. % Backslash
escape_delimiter(D, D).              % Others pass through

%% js_escape_delimiter(+Delimiter, -JsLiteral)
%  Escape the raw delimiter for embedding inside a JS double-quoted string
%  literal used with String.prototype.split(<string>). Because split receives
%  a plain string (not a RegExp), metacharacters such as `|` need NO escaping;
%  only characters that are special *inside a double-quoted JS string* do.
%    tab      -> \t   (so the emitted source reads "\t")
%    CR / LF  -> \r / \n
%    backslash-> \\   (emitted "\\")
%    dquote   -> \"   (emitted "\"")
%    others   -> pass through (comma, pipe, semicolon, colon, space, ...)
js_escape_delimiter('\t', '\\t') :- !.  % Tab  -> "\t"
js_escape_delimiter('\r', '\\r') :- !.  % CR   -> "\r"
js_escape_delimiter('\n', '\\n') :- !.  % LF   -> "\n"
js_escape_delimiter('\\', '\\\\') :- !. % \\    -> "\\"
js_escape_delimiter('"', '\\"') :- !.   % "    -> "\""
js_escape_delimiter(D, D).              % pipe, semicolon, comma, ... pass through

%% ============================================
%% COLUMN-SUBSET PROJECTION (G-P9 residual)
%% ============================================
%% Additive `project_columns([Name, ...])` option for the `_typescript` CSV
%% templates. Distinct from the arity-defining `columns([...])` option: it
%% selects and REORDERS a subset of the file''s columns by matching the file''s
%% detected HEADER row (so the file must have a header). Names are resolved to
%% 0-based indices at compile time and threaded into the TS templates only; the
%% bash and pure-PowerShell templates never reference the projection vars, so
%% their output is unchanged (documented follow-up from GP9POLISH_INTEGRATION_PATCH).

%% resolve_projection(+Options, +CsvFile, +Delimiter, +Arity, -ProjIndices)
%  ProjIndices is the list of 0-based header indices for a project_columns
%  option (in the requested order), or [] when no projection is requested.
%  The number of projected columns must equal Arity. Fails with a clear
%  message on a missing header, an unknown column name, or an arity mismatch.
resolve_projection(Options, CsvFile, Delimiter, Arity, ProjIndices) :-
    (   member(project_columns(Names), Options)
    ->  (   is_list(Names)
        ->  true
        ;   format('Error: project_columns must be a list of column names, got ~w~n', [Names]),
            fail
        ),
        detect_csv_headers(CsvFile, Delimiter, Header),
        (   Header == []
        ->  format('Error: project_columns requires a readable header row in ~w~n', [CsvFile]),
            fail
        ;   true
        ),
        length(Names, NProj),
        (   NProj =:= Arity
        ->  true
        ;   format('Error: project_columns length (~w) does not match arity (~w)~n', [NProj, Arity]),
            fail
        ),
        maplist(resolve_column_index(Header, CsvFile), Names, ProjIndices)
    ;   ProjIndices = []
    ).

%% resolve_column_index(+Header, +CsvFile, +Name, -Index)
%  0-based index of the first header entry equal to Name.
resolve_column_index(Header, CsvFile, Name, Index) :-
    (   nth0(Index, Header, Name)
    ->  true
    ;   format('Error: project_columns name ~w not found in header ~w of ~w~n',
               [Name, Header, CsvFile]),
        fail
    ).

%% projection_ts_vars(+ProjIndices, +Arity, -Vars)
%  Build the TypeScript-only template vars. With no projection the vars carry
%  pass-through defaults that reproduce the pre-projection template bytes
%  exactly; with projection they select/reorder the chosen fields.
projection_ts_vars([], _Arity,
                   [min_fields='arity',
                    output_expr='fields.slice(0, arity).join(":")',
                    u_min='1',
                    u_field='fields[0]']) :- !.
projection_ts_vars(Indices, _Arity, Vars) :-
    max_list(Indices, MaxIdx),
    MinFields is MaxIdx + 1,
    maplist(js_field_ref, Indices, FieldRefs),
    atomic_list_concat(FieldRefs, ', ', FieldsJoined),
    format(atom(OutputExpr), '[~w].join(":")', [FieldsJoined]),
    Indices = [FirstIdx|_],
    format(atom(UField), 'fields[~w]', [FirstIdx]),
    Vars = [min_fields=MinFields,
            output_expr=OutputExpr,
            u_min=MinFields,
            u_field=UField].

js_field_ref(Idx, Ref) :- format(atom(Ref), 'fields[~w]', [Idx]).

%% ============================================
%% HARDCODED TEMPLATES (fallback)
%% ============================================

:- multifile template_system:template/2.

% Arity 1 template: pred(X) - return all values from first column
template_system:template(csv_source_unary, '#!/bin/bash
# {{pred}} - CSV source (arity 1)
# Columns: {{columns}}

{{pred}}() {
    awk -F"{{delimiter}}" ''
    NR > {{skip_lines}} {
        {{quote_code}}
        if (NF >= 1) print $1
    }
    '' {{file}}
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_check() {
    local value="$1"
    [[ -n $({{pred}} | grep -F "$value") ]] && echo "$value exists"
}

# Auto-execute when run directly (not when sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

% Arity 2+ template: pred(Col1, Col2, ...) - return all columns  
template_system:template(csv_source_binary_plus, '#!/bin/bash
# {{pred}} - CSV source (arity {{arity}})
# Columns: {{columns}}

{{pred}}() {
    local target_key="$1"
    
    if [[ -z "$target_key" ]]; then
        # No key provided, stream all rows
        awk -F"{{delimiter}}" ''
        NR > {{skip_lines}} {
            {{quote_code}}
            if (NF >= {{arity}}) print {{output_format}}
        }
        '' {{file}}
    else
        # Lookup mode: find rows where first column matches key
        awk -F"{{delimiter}}" -v key="$target_key" ''
        NR > {{skip_lines}} {
            {{quote_code}}
            if (NF >= {{arity}} && $1 == key) print {{output_format}}
        }
        '' {{file}}
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_all() {
    {{pred}}
}

{{pred}}_check() {
    local key="$1"
    [[ -n $({{pred}} "$key") ]] && echo "$key exists"
}

# Auto-execute when run directly (not when sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

%% ============================================
%% PURE POWERSHELL TEMPLATES
%% ============================================

% Pure PowerShell template for arity 1: pred(X)
template_system:template(csv_source_unary_powershell_pure, '# {{pred}} - CSV source (arity 1) - Pure PowerShell
# Columns: {{columns}}
# Generated by UnifyWeaver - Pure PowerShell mode (no bash dependency)

function {{pred}} {
    param([string]$Value)

    $data = Import-Csv -Path ''{{file}}''{{#delimiter_param}} -Delimiter ''{{delimiter}}''{{/delimiter_param}}

    if ($Value) {
        # Lookup mode: check if value exists in first column
        $columnName = $data[0].PSObject.Properties.Name[0]
        $result = $data | Where-Object { $_.$columnName -eq $Value }
        if ($result) {
            return $result.$columnName
        }
    } else {
        # Stream mode: return all values from first column
        $columnName = $data[0].PSObject.Properties.Name[0]
        return $data | ForEach-Object { $_.$columnName }
    }
}

function {{pred}}_stream {
    {{pred}}
}

function {{pred}}_check {
    param([string]$Value)
    $result = {{pred}} $Value
    if ($result) {
        return "$Value exists"
    }
}

# Auto-execute when run directly (not when dot-sourced)
if ($MyInvocation.InvocationName -ne ''.'') {
    {{pred}} @args
}
').

%% ============================================
%% TYPESCRIPT / NODE TEMPLATES (G-P9)
%% ============================================
%% Self-contained Node scripts (no npm deps): fs + line split. Mirror the
%% pure-PowerShell semantics (lookup by first column when a key is given, else
%% stream all rows; arity 1 emits the first field, arity 2+ joins the first
%% <arity> fields with '':''). Uses only the template variables
%% generate_csv_bash/11 already provides (pred, file, delimiter, skip_lines,
%% arity) plus js_delimiter (G-P9 polish), so csv_source''s existing bash/PS
%% behaviour is unchanged. Runs under either
%% `node --experimental-strip-types file.ts` or plain `node file.js`.
%% DELIMITERS (G-P9 polish): the `_typescript` templates split on
%% {{js_delimiter}}, the raw delimiter escaped for a JS double-quoted string
%% literal (see js_escape_delimiter/2). split() is given a *string*, not a
%% RegExp, so `|` needs no escaping; tab becomes "\t", etc. Comma, tab, pipe,
%% semicolon all work.
%% PROJECTION (G-P9 residual, project_columns): the `_typescript` templates
%% additionally consume {{min_fields}}/{{output_expr}} (binary) and
%% {{u_min}}/{{u_field}} (unary), built by projection_ts_vars/3. With NO
%% project_columns option these carry their pass-through defaults (`arity`,
%% `fields.slice(0, arity).join(":")`, `1`, `fields[0]`) so the emitted script
%% is byte-identical to the pre-projection template. With project_columns([...])
%% the option''s column NAMES are resolved to 0-based indices against the file''s
%% detected header at compile time (resolve_projection/5) and the templates
%% select/reorder exactly those fields, e.g. `[fields[2], fields[0]].join(":")`.
%% bash and _powershell_pure never reference these vars -> unchanged.

% TypeScript/Node template for arity 1: pred(X)
template_system:template(csv_source_unary_typescript, '#!/usr/bin/env node
// {{pred}} - CSV source (arity 1) - self-contained Node (no npm deps)
// Columns: {{columns}}
// Generated by UnifyWeaver - TypeScript/Node data-source consumer
const fs = require("fs");

function {{pred}}(value) {
    const raw = fs.readFileSync("{{file}}", "utf8");
    const lines = raw.split(/\\r?\\n/).slice({{skip_lines}});
    const out = [];
    for (const line of lines) {
        if (line === "") { continue; }
        const fields = line.split("{{js_delimiter}}");
        if (fields.length < {{u_min}}) { continue; }
        if (value !== undefined && value !== "") {
            if ({{u_field}} === value) { out.push({{u_field}}); }
        } else {
            out.push({{u_field}});
        }
    }
    return out;
}

function {{pred}}_stream() {
    return {{pred}}();
}

// Auto-execute when run directly
if (require.main === module) {
    const value = process.argv[2];
    for (const row of {{pred}}(value)) {
        console.log(row);
    }
}

module.exports = { {{pred}}: {{pred}} };
').

% TypeScript/Node template for arity 2+: pred(Col1, Col2, ...)
template_system:template(csv_source_binary_plus_typescript, '#!/usr/bin/env node
// {{pred}} - CSV source (arity {{arity}}) - self-contained Node (no npm deps)
// Columns: {{columns}}
// Generated by UnifyWeaver - TypeScript/Node data-source consumer
const fs = require("fs");

function {{pred}}(key) {
    const raw = fs.readFileSync("{{file}}", "utf8");
    const lines = raw.split(/\\r?\\n/).slice({{skip_lines}});
    const arity = {{arity}};
    const out = [];
    for (const line of lines) {
        if (line === "") { continue; }
        const fields = line.split("{{js_delimiter}}");
        if (fields.length < {{min_fields}}) { continue; }
        if (key !== undefined && key !== "" && fields[0] !== key) { continue; }
        out.push({{output_expr}});
    }
    return out;
}

function {{pred}}_stream() {
    return {{pred}}();
}

function {{pred}}_all() {
    return {{pred}}();
}

// Auto-execute when run directly
if (require.main === module) {
    const key = process.argv[2];
    for (const row of {{pred}}(key)) {
        console.log(row);
    }
}

module.exports = { {{pred}}: {{pred}} };
').

% Pure PowerShell template for arity 2+: pred(Col1, Col2, ...)
template_system:template(csv_source_binary_plus_powershell_pure, '# {{pred}} - CSV source (arity {{arity}}) - Pure PowerShell
# Columns: {{columns}}
# Generated by UnifyWeaver - Pure PowerShell mode (no bash dependency)

function {{pred}} {
    param([string]$Key)

    $data = Import-Csv -Path ''{{file}}''

    if ($Key) {
        # Lookup mode: find rows where first column matches key
        $keyColumn = $data[0].PSObject.Properties.Name[0]
        $matches = $data | Where-Object { $_.$keyColumn -eq $Key }

        # Output in colon-separated format
        foreach ($row in $matches) {
            $values = $row.PSObject.Properties | ForEach-Object { $_.Value }
            $values -join ":"
        }
    } else {
        # Stream all rows in colon-separated format
        foreach ($row in $data) {
            $values = $row.PSObject.Properties | ForEach-Object { $_.Value }
            $values -join ":"
        }
    }
}

function {{pred}}_stream {
    {{pred}}
}

function {{pred}}_all {
    {{pred}}
}

function {{pred}}_check {
    param([string]$Key)
    $result = {{pred}} $Key
    if ($result) {
        return "$Key exists"
    }
}

# Auto-execute when run directly (not when dot-sourced)
if ($MyInvocation.InvocationName -ne ''.'') {
    {{pred}} @args
}
').
