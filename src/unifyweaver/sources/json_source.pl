:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% json_source.pl - JSON source plugin for dynamic sources
% Compiles predicates that process JSON data using jq

% Export nothing - all access goes through plugin registry
:- module(json_source, []).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(json, json_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('JSON Source'),
    version('1.0.0'),
    description('Process JSON data using jq with flexible filtering and output formats'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% validate_config(+Config)
%  Validate configuration for JSON source
validate_config(Config) :-
    % Must have either json_file or json_stdin, and jq_filter
    (   member(jq_filter(Filter), Config),
        atom(Filter)
    ->  true
    ;   format('Error: JSON source requires jq_filter(Filter)~n', []),
        fail
    ),
    
    % Must have input source
    (   member(json_file(_), Config)
    ->  true
    ;   member(json_stdin(true), Config)
    ->  true
    ;   format('Error: JSON source requires json_file(File) or json_stdin(true)~n', []),
        fail
    ),
    
    % Validate json_file if present
    (   member(json_file(File), Config)
    ->  (   exists_file(File)
        ->  true
        ;   format('Warning: JSON file ~w does not exist~n', [File])
        )
    ;   true
    ),
    
    % Validate output_format if specified
    (   member(output_format(Format), Config)
    ->  (   member(Format, [tsv, json, raw, csv])
        ->  true
        ;   format('Error: output_format must be tsv/json/raw/csv, got ~w~n', [Format]),
            fail
        )
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile JSON source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling JSON source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract required parameters
    member(jq_filter(Filter), AllOptions),

    % Extract optional parameters with defaults
    (   member(json_file(JsonFile), AllOptions)
    ->  InputMode = file
    ;   member(json_stdin(true), AllOptions)
    ->  InputMode = stdin,
        JsonFile = ''
    ),
    (   member(output_format(OutputFormat), AllOptions)
    ->  true
    ;   OutputFormat = tsv  % Default output format
    ),
    (   member(raw_output(RawOutput), AllOptions)
    ->  true
    ;   RawOutput = true  % Default to raw output
    ),
    (   member(compact_output(CompactOutput), AllOptions)
    ->  true
    ;   CompactOutput = false  % Default to pretty output
    ),
    (   member(null_input(NullInput), AllOptions)
    ->  true
    ;   NullInput = false  % Default to normal input
    ),
    (   member(error_handling(ErrorHandling), AllOptions)
    ->  true
    ;   ErrorHandling = fail  % Default error handling
    ),

    % Generate bash code using template
    atom_string(Pred, PredStr),
    generate_json_bash(PredStr, Arity, Filter, JsonFile, InputMode,
                      OutputFormat, RawOutput, CompactOutput, NullInput,
                      ErrorHandling, AllOptions, BashCode).

%% ============================================
%% BASH CODE GENERATION
%% ============================================

%% generate_json_bash(+PredStr, +Arity, +Filter, +JsonFile, +InputMode,
%%                    +OutputFormat, +RawOutput, +CompactOutput, +NullInput,
%%                    +ErrorHandling, +Options, -BashCode)
%  Generate bash code for JSON source
generate_json_bash(PredStr, Arity, Filter, JsonFile, InputMode,
                  OutputFormat, RawOutput, CompactOutput, NullInput,
                  ErrorHandling, Options, BashCode) :-

    % Generate jq flags
    generate_jq_flags(RawOutput, CompactOutput, NullInput, JqFlags),

    % Generate error handling code
    generate_json_error_handling(ErrorHandling, ErrorCode),

    % Escape filter for bash (handle single quotes)
    escape_jq_filter(Filter, EscapedFilter),

    % Generate output processing based on format
    generate_output_processing(OutputFormat, OutputProcessing),

    % Build the column projection literal for the TypeScript/Node templates.
    % The bash/PS templates ignore {{projection}} (jq does projection via the
    % jq_filter), so their behaviour is unchanged; only the `_typescript`
    % templates consume it. See json_projection_js/2.
    json_projection_js(Options, Projection),

    % Build the schema-mode field descriptor literal for the TypeScript/Node
    % `_typescript_schema` templates (G-P9 schema mode). The bash/PS templates
    % ignore {{schema_projection}} entirely, so their behaviour is unchanged;
    % only the `_typescript_schema` templates consume it. See
    % json_schema_projection_js/2.
    json_schema_projection_js(Options, SchemaProjection),

    % Select template based on input mode and template_suffix option
    (   InputMode = file ->
        BaseTemplate = json_file_source
    ;   BaseTemplate = json_stdin_source
    ),

    % Check for template_suffix option (for PowerShell, etc.)
    (   member(template_suffix(Suffix), Options)
    ->  atom_concat(BaseTemplate, Suffix, TemplateName)
    ;   TemplateName = BaseTemplate
    ),

    % Render template
    render_named_template(TemplateName,
        [pred=PredStr, filter=EscapedFilter, json_file=JsonFile,
         jq_flags=JqFlags, error_code=ErrorCode,
         output_processing=OutputProcessing, output_format=OutputFormat,
         arity=Arity, input_mode=InputMode, projection=Projection,
         schema_projection=SchemaProjection],
        [source_order([file, generated])],
        BashCode).

%% json_projection_js(+Options, -Projection)
%  Build a JS array-literal of the requested column key paths, e.g.
%  columns([id, name, price]) -> '["id","name","price"]'. This is the same
%  `columns` list that sources.pl requires for a JSON source (count == arity)
%  and that csharp_target feeds to JsonStreamReader.ColumnSelectors. The
%  TypeScript/Node templates read each key from the object (dotted paths
%  supported at runtime), giving true selection/ordering rather than relying on
%  object insertion order. When no columns are declared the literal is `[]` and
%  the template falls back to Object.values.
json_projection_js(Options, Projection) :-
    (   member(columns(Cols), Options),
        is_list(Cols),
        Cols \= []
    ->  maplist(json_column_key, Cols, Keys),
        maplist(js_string_literal, Keys, Literals),
        atomic_list_concat(Literals, ',', Inner),
        format(atom(Projection), '[~w]', [Inner])
    ;   Projection = '[]'
    ).

%% json_column_key(+ColumnEntry, -KeyAtom)
%  Normalise a columns/1 entry to an atom key path. Accepts atoms, strings and
%  jsonpath(Path) wrappers (the leading `$.`/`$` is stripped so a plain object
%  lookup works in the pure-Node template).
json_column_key(jsonpath(Path), Key) :- !,
    json_column_key(Path, Key).
json_column_key(Entry, Key) :-
    (   atom(Entry) -> atom_string(Entry, S)
    ;   string(Entry) -> S = Entry
    ;   term_to_atom(Entry, A), atom_string(A, S)
    ),
    (   sub_string(S, 0, 2, _, "$.")
    ->  sub_string(S, 2, _, 0, S1)
    ;   sub_string(S, 0, 1, _, "$")
    ->  sub_string(S, 1, _, 0, S1)
    ;   S1 = S
    ),
    atom_string(Key, S1).

%% js_string_literal(+Atom, -Literal)
%  Wrap an atom in double quotes, escaping backslash and double-quote so the
%  emitted JS source is a valid string literal.
js_string_literal(Atom, Literal) :-
    atom_string(Atom, S0),
    % Escape backslashes first, then double quotes.
    split_string(S0, "\\", "", BParts),
    atomic_list_concat(BParts, '\\\\', S1),
    atom_string(S1, S1s),
    split_string(S1s, "\"", "", QParts),
    atomic_list_concat(QParts, '\\"', S2),
    format(atom(Literal), '"~w"', [S2]).

%% ============================================
%% SCHEMA MODE (G-P9): typed-object output
%% ============================================
%% When a JSON source is declared with schema([field(Name, Path, Type), ...]),
%% dynamic_source_compiler stores the normalized field dicts in
%% dynamic_source_metadata(Pred/Arity, Meta).schema_fields. The TypeScript/Node
%% wrapper (typescript_source_compiler) reads that metadata and threads the flat
%% root field list through as a schema_fields(Fields) option, selecting the
%% `_typescript_schema` template variant. This is a sibling of the flat
%% `columns` projection path — additive, and it leaves bash/PS/non-schema TS
%% output byte-identical (those templates never reference {{schema_projection}}).
%%
%% Per-record output shape: each JSON record is emitted as JSON.stringify of a
%% typed object with exactly the schema's field keys, in declared order, so a
%% caller gets the full typed record (mirrors csharp_target's typed records,
%% but as a JSON object line rather than the flat ':'-joined projection).

%% json_schema_projection_js(+Options, -Projection)
%  Build a JS array-literal of {"key","path","type"} descriptors, one per
%  declared schema field, in declared order — e.g.
%    [{"key":"id","path":"id","type":"string"},
%     {"key":"price","path":"price","type":"integer"}]
%  Reads the schema_fields(Fields) option (the normalized field dicts from
%  dynamic_source_metadata). When absent/empty the literal is `[]` and the
%  `_typescript_schema` template emits an empty object per record.
json_schema_projection_js(Options, Projection) :-
    (   member(schema_fields(Fields), Options),
        is_list(Fields),
        Fields \= []
    ->  maplist(json_schema_field_js, Fields, Objs),
        atomic_list_concat(Objs, ',', Inner),
        format(atom(Projection), '[~w]', [Inner])
    ;   Projection = '[]'
    ).

%% json_schema_field_js(+FieldDict, -JsObject)
%  Render one normalized schema field dict as a JS object literal. Reads the
%  same keys csharp_target reads: `name` (the object key), `path` (the source
%  key path, jsonpath `$.`/`$` prefix stripped so a plain object lookup works),
%  and `column_type` (the declared type; record fields carry column_type json,
%  so they pass through unchanged in the emitted coercion switch).
json_schema_field_js(Field, Obj) :-
    get_dict(name, Field, NameAtom),
    get_dict(column_type, Field, TypeAtom),
    get_dict(path, Field, Path),
    json_column_key(Path, PathKey),
    js_string_literal(NameAtom, KeyLit),
    js_string_literal(PathKey, PathLit),
    js_string_literal(TypeAtom, TypeLit),
    format(atom(Obj), '{"key":~w,"path":~w,"type":~w}', [KeyLit, PathLit, TypeLit]).

%% generate_jq_flags(+RawOutput, +CompactOutput, +NullInput, -Flags)
%  Generate jq command line flags
generate_jq_flags(RawOutput, CompactOutput, NullInput, Flags) :-
    FlagsList = [],
    (   RawOutput = true ->
        append(FlagsList, ['-r'], Flags1)
    ;   Flags1 = FlagsList
    ),
    (   CompactOutput = true ->
        append(Flags1, ['-c'], Flags2)
    ;   Flags2 = Flags1
    ),
    (   NullInput = true ->
        append(Flags2, ['-n'], Flags3)
    ;   Flags3 = Flags2
    ),
    atomic_list_concat(Flags3, ' ', Flags).

%% generate_json_error_handling(+Mode, -Code)
%  Generate error handling code
generate_json_error_handling(fail, 'set -e  # Exit on JSON processing errors') :- !.
generate_json_error_handling(warn, '# JSON warnings enabled') :- !.
generate_json_error_handling(continue, '# Continue on JSON errors') :- !.
generate_json_error_handling(_, '# Default JSON error handling').

%% escape_jq_filter(+Filter, -Escaped)
%  Escape jq filter for bash usage
escape_jq_filter(Filter, Escaped) :-
    % For now, simple pass-through - could add more escaping if needed
    % Main concern is single quotes in bash
    Escaped = Filter.

%% generate_output_processing(+Format, -Processing)
%  Generate output post-processing based on format
generate_output_processing(tsv, '# TSV output - use @tsv in jq filter') :- !.
generate_output_processing(csv, '# CSV output - use @csv in jq filter') :- !.
generate_output_processing(json, '# JSON output') :- !.
generate_output_processing(raw, '# Raw output') :- !.
generate_output_processing(_, '# Default output processing').

%% ============================================
%% HARDCODED TEMPLATES (fallback)
%% ============================================

:- multifile template_system:template/2.

% JSON file template - reads from file
template_system:template(json_file_source, '#!/bin/bash
# {{pred}} - JSON source from file ({{json_file}})

{{pred}}() {
    local json_file="{{json_file}}"
    local additional_filter="$1"
    
    {{error_code}}
    {{output_processing}}
    
    # Check if file exists
    if [[ ! -f "$json_file" ]]; then
        echo "JSON file not found: $json_file" >&2
        return 1
    fi
    
    # Apply jq filter, optionally with additional filter
    if [[ -n "$additional_filter" ]]; then
        # Combine filters with pipe
        jq {{jq_flags}} "{{filter}} | $additional_filter" "$json_file"
    else
        jq {{jq_flags}} "{{filter}}" "$json_file"
    fi
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "jq processing failed with exit code $exit_code" >&2
        return $exit_code
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_raw() {
    # Raw jq output without additional processing
    jq {{jq_flags}} "{{filter}}" "{{json_file}}"
}

{{pred}}_filter() {
    # Apply custom filter
    local custom_filter="$1"
    shift
    jq {{jq_flags}} "$custom_filter" "{{json_file}}" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

% JSON stdin template - reads from stdin
template_system:template(json_stdin_source, '#!/bin/bash
# {{pred}} - JSON source from stdin

{{pred}}() {
    local additional_filter="$1"
    
    {{error_code}}
    {{output_processing}}
    
    # Apply jq filter to stdin, optionally with additional filter
    if [[ -n "$additional_filter" ]]; then
        # Combine filters with pipe
        jq {{jq_flags}} "{{filter}} | $additional_filter"
    else
        jq {{jq_flags}} "{{filter}}"
    fi
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "jq processing failed with exit code $exit_code" >&2
        return $exit_code
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_raw() {
    # Raw jq output without additional processing
    jq {{jq_flags}} "{{filter}}"
}

{{pred}}_filter() {
    # Apply custom filter
    local custom_filter="$1"
    shift
    jq {{jq_flags}} "$custom_filter" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

%% ============================================
%% PURE POWERSHELL TEMPLATES
%% ============================================

% Pure PowerShell template for JSON file source
template_system:template(json_file_source_powershell_pure, '# {{pred}} - JSON file source - Pure PowerShell
# Generated by UnifyWeaver - Pure PowerShell mode (no bash dependency)

function {{pred}} {
    param([string]$Key)

    try {
        $jsonContent = Get-Content ''{{json_file}}'' -Raw | ConvertFrom-Json

        # Output each item in the JSON array
        foreach ($item in $jsonContent) {
            # Get all property values and join with colon
            $values = $item.PSObject.Properties | ForEach-Object { $_.Value }
            $values -join ":"
        }
    } catch {
        Write-Error "JSON processing failed: $_"
    }
}

function {{pred}}_stream {
    {{pred}}
}

function {{pred}}_check {
    param([string]$Key)
    $result = {{pred}} $Key
    if ($result) {
        return "$Key exists"
    }
}

function {{pred}}_filter {
    param([string]$Filter)
    # Apply custom PowerShell filter expression
    $jsonContent = Get-Content ''{{json_file}}'' -Raw | ConvertFrom-Json
    $jsonContent | Where-Object $Filter | ForEach-Object {
        $values = $_.PSObject.Properties | ForEach-Object { $_.Value }
        $values -join ":"
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
%% Self-contained Node scripts (no npm deps): fs + JSON.parse. Mirror the
%% pure-PowerShell semantics (read the JSON array, join each item''s property
%% values with '':''; arity 1 emits the first value). Uses the template
%% variables generate_json_bash/12 provides (pred, json_file, arity) plus
%% projection (G-P9 polish), so json_source''s existing bash/PS behaviour is
%% unchanged. Runs under either `node --experimental-strip-types file.ts` or
%% plain `node file.js` (the CommonJS require works in both).
%% PROJECTION (G-P9 polish): when the source declares columns([...]) (which
%% sources.pl requires for JSON, count == arity), {{projection}} is a JS array
%% of those key paths and the emitted script selects/orders exactly those keys
%% (dotted paths traversed at runtime) instead of relying on object insertion
%% order. With no columns the literal is [] and it falls back to Object.values.

% TypeScript/Node template for JSON file source
template_system:template(json_file_source_typescript, '#!/usr/bin/env node
// {{pred}} - JSON file source - self-contained Node (no npm deps)
// Generated by UnifyWeaver - TypeScript/Node data-source consumer
const fs = require("fs");

// projection: declared columns([...]) key paths, in order (empty => all values)
const {{pred}}_projection = {{projection}};

function {{pred}}_pick(obj, path) {
    let cur = obj;
    for (const part of String(path).split(".")) {
        if (cur === null || cur === undefined) { return ""; }
        cur = cur[part];
    }
    return cur === null || cur === undefined ? "" : String(cur);
}

function {{pred}}(key) {
    const raw = fs.readFileSync("{{json_file}}", "utf8");
    let data = JSON.parse(raw);
    if (!Array.isArray(data)) { data = [data]; }
    const arity = {{arity}};
    const out = [];
    for (const item of data) {
        let values;
        if ({{pred}}_projection.length > 0) {
            values = {{pred}}_projection.map((p) => {{pred}}_pick(item, p));
        } else if (item !== null && typeof item === "object") {
            values = Object.values(item).map((v) => String(v));
        } else {
            values = [String(item)];
        }
        if (arity === 1) {
            out.push(values[0]);
        } else {
            out.push(values.join(":"));
        }
    }
    return out;
}

function {{pred}}_stream() {
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

% TypeScript/Node template for JSON stdin source
template_system:template(json_stdin_source_typescript, '#!/usr/bin/env node
// {{pred}} - JSON stdin source - self-contained Node (no npm deps)
// Generated by UnifyWeaver - TypeScript/Node data-source consumer
const fs = require("fs");

// projection: declared columns([...]) key paths, in order (empty => all values)
const {{pred}}_projection = {{projection}};

function {{pred}}_pick(obj, path) {
    let cur = obj;
    for (const part of String(path).split(".")) {
        if (cur === null || cur === undefined) { return ""; }
        cur = cur[part];
    }
    return cur === null || cur === undefined ? "" : String(cur);
}

function {{pred}}FromString(raw, key) {
    let data = JSON.parse(raw);
    if (!Array.isArray(data)) { data = [data]; }
    const arity = {{arity}};
    const out = [];
    for (const item of data) {
        let values;
        if ({{pred}}_projection.length > 0) {
            values = {{pred}}_projection.map((p) => {{pred}}_pick(item, p));
        } else if (item !== null && typeof item === "object") {
            values = Object.values(item).map((v) => String(v));
        } else {
            values = [String(item)];
        }
        if (arity === 1) {
            out.push(values[0]);
        } else {
            out.push(values.join(":"));
        }
    }
    return out;
}

// Auto-execute when run directly: parse JSON from stdin
if (require.main === module) {
    const raw = fs.readFileSync(0, "utf8");
    const key = process.argv[2];
    for (const row of {{pred}}FromString(raw, key)) {
        console.log(row);
    }
}

module.exports = { {{pred}}FromString: {{pred}}FromString };
').

%% ============================================
%% TYPESCRIPT / NODE SCHEMA TEMPLATES (G-P9 schema mode)
%% ============================================
%% Selected (via template_suffix(''_typescript_schema'')) when the JSON source
%% carries a declared schema (dynamic_source_metadata schema_fields). Each JSON
%% record is parsed into a typed object with exactly the schema''s field keys in
%% declared order, coercing each field to its declared type (string/integer/
%% float/number/boolean; unknown/json passes through), then emitted as a
%% JSON.stringify line so the caller gets the full typed record. Arity is always
%% 1 for a schema source. Bash/PS and the flat-columns `_typescript` templates
%% are untouched.

% TypeScript/Node schema template for JSON file source
template_system:template(json_file_source_typescript_schema, '#!/usr/bin/env node
// {{pred}} - JSON file source (schema mode) - self-contained Node (no npm deps)
// Generated by UnifyWeaver - TypeScript/Node data-source consumer
const fs = require("fs");

// schema: declared field descriptors {key,path,type} in declared order
const {{pred}}_schema = {{schema_projection}};

function {{pred}}_pickRaw(obj, path) {
    let cur = obj;
    if (path === "") { return cur; }
    for (const part of String(path).split(".")) {
        if (cur === null || cur === undefined) { return null; }
        cur = cur[part];
    }
    return cur === undefined ? null : cur;
}

function {{pred}}_coerce(value, type) {
    if (value === null || value === undefined) { return null; }
    switch (type) {
        case "string":
            return String(value);
        case "integer":
        case "long": {
            const n = Number(value);
            return Number.isNaN(n) ? null : Math.trunc(n);
        }
        case "float":
        case "double":
        case "number": {
            const n = Number(value);
            return Number.isNaN(n) ? null : n;
        }
        case "boolean":
            if (typeof value === "boolean") { return value; }
            if (typeof value === "string") { return value.toLowerCase() === "true"; }
            return Boolean(value);
        default:
            return value;
    }
}

function {{pred}}_record(item) {
    const record = {};
    for (const f of {{pred}}_schema) {
        record[f.key] = {{pred}}_coerce({{pred}}_pickRaw(item, f.path), f.type);
    }
    return record;
}

function {{pred}}(key) {
    const raw = fs.readFileSync("{{json_file}}", "utf8");
    let data = JSON.parse(raw);
    if (!Array.isArray(data)) { data = [data]; }
    const out = [];
    for (const item of data) {
        out.push(JSON.stringify({{pred}}_record(item)));
    }
    return out;
}

function {{pred}}_stream() {
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

% TypeScript/Node schema template for JSON stdin source
template_system:template(json_stdin_source_typescript_schema, '#!/usr/bin/env node
// {{pred}} - JSON stdin source (schema mode) - self-contained Node (no npm deps)
// Generated by UnifyWeaver - TypeScript/Node data-source consumer
const fs = require("fs");

// schema: declared field descriptors {key,path,type} in declared order
const {{pred}}_schema = {{schema_projection}};

function {{pred}}_pickRaw(obj, path) {
    let cur = obj;
    if (path === "") { return cur; }
    for (const part of String(path).split(".")) {
        if (cur === null || cur === undefined) { return null; }
        cur = cur[part];
    }
    return cur === undefined ? null : cur;
}

function {{pred}}_coerce(value, type) {
    if (value === null || value === undefined) { return null; }
    switch (type) {
        case "string":
            return String(value);
        case "integer":
        case "long": {
            const n = Number(value);
            return Number.isNaN(n) ? null : Math.trunc(n);
        }
        case "float":
        case "double":
        case "number": {
            const n = Number(value);
            return Number.isNaN(n) ? null : n;
        }
        case "boolean":
            if (typeof value === "boolean") { return value; }
            if (typeof value === "string") { return value.toLowerCase() === "true"; }
            return Boolean(value);
        default:
            return value;
    }
}

function {{pred}}_record(item) {
    const record = {};
    for (const f of {{pred}}_schema) {
        record[f.key] = {{pred}}_coerce({{pred}}_pickRaw(item, f.path), f.type);
    }
    return record;
}

function {{pred}}FromString(raw, key) {
    let data = JSON.parse(raw);
    if (!Array.isArray(data)) { data = [data]; }
    const out = [];
    for (const item of data) {
        out.push(JSON.stringify({{pred}}_record(item)));
    }
    return out;
}

// Auto-execute when run directly: parse JSON from stdin
if (require.main === module) {
    const raw = fs.readFileSync(0, "utf8");
    const key = process.argv[2];
    for (const row of {{pred}}FromString(raw, key)) {
        console.log(row);
    }
}

module.exports = { {{pred}}FromString: {{pred}}FromString };
').

% Pure PowerShell template for JSON stdin source
template_system:template(json_stdin_source_powershell_pure, '# {{pred}} - JSON stdin source - Pure PowerShell
# Generated by UnifyWeaver - Pure PowerShell mode (no bash dependency)

function {{pred}} {
    param(
        [Parameter(ValueFromPipeline=$true)]
        [string[]]$InputData,
        [string]$Key
    )

    begin {
        $jsonLines = @()
    }

    process {
        if ($InputData) {
            $jsonLines += $InputData
        }
    }

    end {
        $jsonContent = $jsonLines -join "`n" | ConvertFrom-Json

        # Apply filter logic based on arity
        {{#has_key_filter}}
        if ($Key) {
            $results = $jsonContent | Where-Object { $_.{{key_field}} -eq $Key }
        } else {
            $results = $jsonContent
        }
        {{/has_key_filter}}
        {{^has_key_filter}}
        $results = $jsonContent
        {{/has_key_filter}}

        # Format output
        foreach ($item in $results) {
            {{#arity_1}}
            $item.{{field_0}}
            {{/arity_1}}
            {{#arity_2plus}}
            $values = @({{#fields}}$item.{{field}}{{^last}}, {{/last}}{{/fields}})
            $values -join ":"
            {{/arity_2plus}}
        }
    }
}

function {{pred}}_stream {
    $Input | {{pred}}
}

# Auto-execute when run directly (not when dot-sourced)
if ($MyInvocation.InvocationName -ne ''.'') {
    if ($Input) {
        $Input | {{pred}} @args
    } else {
        {{pred}} @args
    }
}
').
