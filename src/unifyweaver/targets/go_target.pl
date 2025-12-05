:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% go_target.pl - Go Target for UnifyWeaver
% Generates standalone Go programs for record/field processing
% Supports configurable delimiters, quoting, and regex matching

:- module(go_target, [
    compile_predicate_to_go/3,      % +Predicate, +Options, -GoCode
    write_go_program/2,             % +GoCode, +FilePath
    json_schema/2,                  % +SchemaName, +Fields (directive)
    get_json_schema/2,              % +SchemaName, -Fields (lookup)
    get_json_schema/2,              % +SchemaName, -Fields (lookup)
    get_field_info/4                % +SchemaName, +FieldName, -Type, -Options (lookup)
]).

:- use_module(library(lists)).
:- use_module(library(filesex)).

% Suppress singleton warnings in this experimental generator target.
:- style_check(-singleton).
:- discontiguous extract_match_constraints/2.
:- use_module(library(filesex)).

%% ============================================
%% JSON SCHEMA SUPPORT
%% ============================================

:- dynamic json_schema_def/2.

%% json_schema(+SchemaName, +Fields)
%  Define a JSON schema with typed fields
%  Used as a directive: :- json_schema(person, [field(name, string), field(age, integer)]).
%
json_schema(SchemaName, Fields) :-
    % Validate schema fields
    (   validate_schema_fields(Fields)
    ->  % Store schema definition
        retractall(json_schema_def(SchemaName, _)),
        assertz(json_schema_def(SchemaName, Fields)),
        format('Schema defined: ~w with ~w fields~n', [SchemaName, Fields])
    ;   format('ERROR: Invalid schema definition for ~w: ~w~n', [SchemaName, Fields]),
        fail
    ).

%% validate_schema_fields(+Fields)
%  Validate that all fields have correct format: field(Name, Type)
%
validate_schema_fields([]).
validate_schema_fields([field(Name, Type)|Rest]) :-
    atom(Name),
    valid_json_type(Type),
    validate_schema_fields(Rest).
validate_schema_fields([field(Name, Type, Options)|Rest]) :-
    atom(Name),
    valid_json_type(Type),
    is_list(Options),
    validate_field_options(Options),
    validate_schema_fields(Rest).
validate_schema_fields([Invalid|_]) :-
    format('ERROR: Invalid field specification: ~w~n', [Invalid]),
    fail.

%% validate_field_options(+Options)
%  Validate field options: min(N), max(N), format(F), required, optional
validate_field_options([]).
validate_field_options([Option|Rest]) :-
    validate_field_option(Option),
    validate_field_options(Rest).

validate_field_option(min(N)) :- number(N).
validate_field_option(max(N)) :- number(N).
validate_field_option(format(F)) :- atom(F).
validate_field_option(required).
validate_field_option(optional).
validate_field_option(Invalid) :-
    format('ERROR: Invalid field option: ~w~n', [Invalid]),
    fail.

%% valid_json_type(+Type)
%  Check if a type is valid for JSON schemas
%
valid_json_type(string).
valid_json_type(integer).
valid_json_type(float).
valid_json_type(boolean).
valid_json_type(any).  % Fallback to interface{}

%% get_json_schema(+SchemaName, -Fields)
%  Retrieve a schema definition by name
%
get_json_schema(SchemaName, Fields) :-
    json_schema_def(SchemaName, Fields), !.
get_json_schema(SchemaName, _) :-
    format('ERROR: Schema not found: ~w~n', [SchemaName]),
    fail.

%% get_field_info(+SchemaName, +FieldName, -Type, -Options)
%  Get the type and options of a specific field from a schema
%
get_field_info(SchemaName, FieldName, Type, Options) :-
    get_json_schema(SchemaName, Fields),
    (   member(field(FieldName, Type, Options), Fields) -> true
    ;   member(field(FieldName, Type), Fields) -> Options = []
    ), !.
get_field_info(SchemaName, FieldName, any, []) :-
    % Field not in schema - default to 'any' (interface{})
    format('WARNING: Field ~w not in schema ~w, defaulting to type ''any''~n', [FieldName, SchemaName]).

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_go(+Predicate, +Options, -GoCode)
%  Compile a Prolog predicate to Go code
%
%  @arg Predicate Predicate indicator (Name/Arity)
%  @arg Options List of options
%  @arg GoCode Generated Go code as atom
%
%  Options:
%  - record_delimiter(null|newline|Char) - Record separator (default: newline)
%  - field_delimiter(colon|tab|comma|Char) - Field separator (default: colon)
%  - quoting(csv|none) - Quoting style (default: none)
%  - escape_char(Char) - Escape character (default: backslash)
%  - include_package(true|false) - Include package main (default: true)
%  - unique(true|false) - Deduplicate results (default: true)
%  - aggregation(sum|count|max|min|avg) - Aggregation operation
%  - db_backend(bbolt) - Database backend (bbolt only for now)
%  - db_file(Path) - Database file path (default: 'data.db')
%  - db_bucket(Name) - Bucket name (default: predicate name)
%  - db_key_field(Field) - Field to use as key
%  - db_mode(read|write) - Database operation mode (default: write with json_input, read otherwise)
%
compile_predicate_to_go(PredIndicator, Options, GoCode) :-
    (   PredIndicator = _Module:Pred/Arity
    ->  true
    ;   PredIndicator = Pred/Arity
    ),
    format('=== Compiling ~w/~w to Go ===~n', [Pred, Arity]),

    % Check if this is database read mode
    (   option(db_backend(bbolt), Options),
        (option(db_mode(read), Options) ; \+ option(json_input(true), Options)),
        \+ option(json_output(true), Options)
    ->  % Compile for database read
        format('  Mode: Database read (bbolt)~n'),
        compile_database_read_mode(Pred, Arity, Options, GoCode)
    % Check if this is JSON input mode
    ;   option(json_input(true), Options)
    ->  % Compile for JSON input (may include database write)
        (   option(db_backend(bbolt), Options)
        ->  format('  Mode: JSON input (JSONL) with database storage~n')
        ;   format('  Mode: JSON input (JSONL)~n')
        ),
        % Check for parallel execution
        (   option(workers(Workers), Options), Workers > 1
        ->  format('  Parallel execution: ~w workers~n', [Workers]),
            compile_parallel_json_input_mode(Pred, Arity, Options, Workers, GoCode)
        ;   compile_json_input_mode(Pred, Arity, Options, GoCode)
        )
    % Check if this is XML input mode
    ;   option(xml_input(true), Options)
    ->  % Compile for XML input
        format('  Mode: XML input (streaming + flattening)~n'),
        compile_xml_input_mode(Pred, Arity, Options, GoCode)
    % Check if this is JSON output mode
    ;   option(json_output(true), Options)
    ->  % Compile for JSON output
        format('  Mode: JSON output~n'),
        compile_json_output_mode(Pred, Arity, Options, GoCode)
    % Check if this is an aggregation operation
    ;   option(aggregation(AggOp), Options, none),
        AggOp \= none
    ->  % Compile as aggregation
        option(field_delimiter(FieldDelim), Options, colon),
        option(include_package(IncludePackage), Options, true),
        compile_aggregation_to_go(Pred, Arity, AggOp, FieldDelim, IncludePackage, GoCode)
    ;   % Continue with normal compilation
        compile_predicate_to_go_normal(Pred, Arity, Options, GoCode)
    ).

%% compile_predicate_to_go_normal(+Pred, +Arity, +Options, -GoCode)
%  Normal (non-aggregation) compilation path
%
compile_predicate_to_go_normal(Pred, Arity, Options, GoCode) :-
    % Get options
    option(record_delimiter(RecordDelim), Options, newline),
    option(field_delimiter(FieldDelim), Options, colon),
    option(quoting(Quoting), Options, none),
    option(escape_char(EscapeChar), Options, backslash),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),

    % Create head with correct arity
    functor(Head, Pred, Arity),

    % Get all clauses for this predicate
    findall(Head-Body, user:clause(Head, Body), Clauses),

    % Determine compilation strategy
    (   Clauses = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    
    % Semantic Rule Check
    ;   Clauses = [Head-Body], Body \= true,
        extract_predicates(Body, [SinglePred]),
        is_semantic_predicate(SinglePred)
    ->  Head =.. [_|HeadArgs],
        compile_semantic_rule_go(Pred, HeadArgs, SinglePred, GoCode)

    ;   maplist(is_fact_clause, Clauses) ->
        % All bodies are 'true' - these are facts
        format('Type: facts (~w clauses)~n', [length(Clauses, _)]),
        compile_facts_to_go(Pred, Arity, Clauses, RecordDelim, FieldDelim,
                           Unique, ScriptBody),
        GenerateProgram = true
    ;   is_tail_recursive_pattern(Pred, Clauses) ->
        % Tail recursive pattern - compile to iterative loop
        format('Type: tail_recursion~n'),
        compile_tail_recursive_to_go(Pred, Arity, Clauses, ScriptBody),
        GenerateProgram = true
    ;   Clauses = [SingleHead-SingleBody], SingleBody \= true ->
        % Single rule
        format('Type: single_rule~n'),
        compile_single_rule_to_go(Pred, Arity, SingleHead, SingleBody, RecordDelim,
                                 FieldDelim, Unique, ScriptBody),
        GenerateProgram = true
    ;   % Multiple rules (OR pattern)
        format('Type: multiple_rules (~w clauses)~n', [length(Clauses, _)]),
        compile_multiple_rules_to_go(Pred, Arity, Clauses, RecordDelim,
                                    FieldDelim, Unique, ScriptBody),
        GenerateProgram = true
    ),

    % Determine if we need imports (only if GenerateProgram is true)
    (   var(GenerateProgram) -> true
    ;   (   maplist(is_fact_clause, Clauses) ->
            NeedsStdin = false
        ;   is_tail_recursive_pattern(Pred, Clauses) ->
            NeedsStdin = false
        ;   NeedsStdin = true
        ),

        % Determine if we need regexp imports
        (   member(_Head-Body, Clauses),
            extract_match_constraints(Body, MatchCs),
            MatchCs \= []
        ->  NeedsRegexp = true
        ;   NeedsRegexp = false
        ),

        % Determine if we need strings import
        (   NeedsStdin,
            (   length(Clauses, NumClauses), NumClauses > 1, \+ maplist(is_fact_clause, Clauses)
            ;   member(Head-Body, Clauses), \+ is_fact_clause(Head-Body),
                Head =.. [_|Args], length(Args, ArgCount), ArgCount > 1,
                extract_predicates(Body, Preds), Preds \= []
            )
        ->  NeedsStrings = true
        ;   NeedsStrings = false
        ),

        % Determine if we need strconv import
        (   member(_Head-Body, Clauses),
            extract_constraints(Body, Cs),
            Cs \= []
        ->  NeedsStrconv = true
        ;   NeedsStrconv = false
        ),

        % Generate complete Go program
        (   IncludePackage ->
            generate_go_program(Pred, Arity, RecordDelim, FieldDelim, Quoting,
                               EscapeChar, NeedsStdin, NeedsRegexp, NeedsStrings, NeedsStrconv, ScriptBody, GoCode)
        ;   GoCode = ScriptBody
        )
    ),
    !.

%% Helper to check if a clause is a fact (body is just 'true')
is_fact_clause(_-true).

%% ============================================
%% AGGREGATION PATTERN COMPILATION
%% ============================================

%% compile_aggregation_to_go(+Pred, +Arity, +AggOp, +FieldDelim, +IncludePackage, -GoCode)
%  Compile aggregation operations (sum, count, max, min, avg)
%
compile_aggregation_to_go(Pred, Arity, AggOp, FieldDelim, IncludePackage, GoCode) :-
    atom_string(Pred, PredStr),
    map_field_delimiter(FieldDelim, DelimChar),

    format('  Aggregation type: ~w~n', [AggOp]),

    % Generate aggregation Go code based on operation
    generate_aggregation_go(AggOp, Arity, DelimChar, ScriptBody),

    % Determine imports based on aggregation type
    %  count doesn't need strconv, others do
    (   AggOp = count ->
        Imports = '\t"bufio"\n\t"fmt"\n\t"os"'
    ;   Imports = '\t"bufio"\n\t"fmt"\n\t"os"\n\t"strconv"'
    ),

    % Wrap in package main if requested
    (   IncludePackage ->
        format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, ScriptBody])
    ;   GoCode = ScriptBody
    ).

%% generate_aggregation_go(+AggOp, +Arity, +DelimChar, -GoCode)
%  Generate Go code for specific aggregation operations
%
generate_aggregation_go(sum, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tsum := 0.0
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tsum += val
\t\t}
\t}
\t
\tfmt.Println(sum)
', []).

generate_aggregation_go(count, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tcount := 0
\t
\tfor scanner.Scan() {
\t\tcount++
\t}
\t
\tfmt.Println(count)
', []).

generate_aggregation_go(max, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tvar max float64
\tfirst := true
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tif first || val > max {
\t\t\t\tmax = val
\t\t\t\tfirst = false
\t\t\t}
\t\t}
\t}
\t
\tif !first {
\t\tfmt.Println(max)
\t}
', []).

generate_aggregation_go(min, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tvar min float64
\tfirst := true
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tif first || val < min {
\t\t\t\tmin = val
\t\t\t\tfirst = false
\t\t\t}
\t\t}
\t}
\t
\tif !first {
\t\tfmt.Println(min)
\t}
', []).

generate_aggregation_go(avg, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tsum := 0.0
\tcount := 0
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tsum += val
\t\t\tcount++
\t\t}
\t}
\t
\tif count > 0 {
\t\tfmt.Println(sum / float64(count))
\t}
', []).

%% ============================================
%% FACTS COMPILATION
%% ============================================

%% compile_facts_to_go(+Pred, +Arity, +Clauses, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile fact predicates to Go map lookup
%
compile_facts_to_go(Pred, Arity, Clauses, _RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Build map entries
    findall(Entry,
        (   member(Fact-true, Clauses),
            Fact =.. [_|Args],
            format_go_fact_entry(Args, FieldDelim, Entry)
        ),
        Entries),
    atomic_list_concat(Entries, ',\n\t\t', EntriesStr),

    % Generate Go code
    format(string(GoCode), '
\tfacts := map[string]bool{
\t\t~s,
\t}

\tfor key := range facts {
\t\tfmt.Println(key)
\t}
', [EntriesStr]).

%% format_go_fact_entry(+Args, +FieldDelim, -Entry)
%  Format a fact as a Go map entry
format_go_fact_entry(Args, FieldDelim, Entry) :-
    map_field_delimiter(FieldDelim, DelimChar),
    maplist(atom_string, Args, ArgStrs),
    atomic_list_concat(ArgStrs, DelimChar, Key),
    format(string(Entry), '"~s": true', [Key]).

%% ============================================
%% SINGLE RULE COMPILATION
%% ============================================

%% compile_single_rule_to_go(+Pred, +Arity, +Head, +Body, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile single rule to Go code
%
compile_single_rule_to_go(Pred, Arity, Head, Body, RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Build variable mapping from head arguments
    Head =.. [_|HeadArgs],
    build_var_map(HeadArgs, VarMap),
    format('  Variable map: ~w~n', [VarMap]),

    % Extract predicates, match constraints, and arithmetic constraints from body
    extract_predicates(Body, Predicates),
    extract_match_constraints(Body, MatchConstraints),
    extract_constraints(Body, Constraints),
    format('  Body predicates: ~w~n', [Predicates]),
    format('  Match constraints: ~w~n', [MatchConstraints]),
    format('  Arithmetic constraints: ~w~n', [Constraints]),

    % Handle simple case: single predicate in body (with optional constraints)
    (   Predicates = [SinglePred] ->
        (   is_semantic_predicate(SinglePred)
        ->  compile_semantic_rule_go(PredStr, HeadArgs, SinglePred, GoCode)
        ;   compile_single_predicate_rule_go(PredStr, HeadArgs, SinglePred, VarMap,
                                            FieldDelim, Unique, MatchConstraints, Constraints, GoCode)
        )
    ;   Predicates = [], MatchConstraints \= [] ->
        % No predicates, just match constraints - read from stdin and filter
        compile_match_only_rule_go(PredStr, HeadArgs, VarMap, FieldDelim,
                                   Unique, MatchConstraints, GoCode)
    ;   % Multiple predicates or unsupported pattern
        format(user_error,
               'Go target: multi-predicate or constraint-only rules not supported (yet) for ~w/~w~n',
               [Pred, Arity]),
        fail
    ).

%% compile_single_predicate_rule_go(+PredStr, +HeadArgs, +BodyPred, +VarMap, +FieldDelim, +Unique, +MatchConstraints, +Constraints, -GoCode)
%  Compile a rule with single predicate in body (e.g., child(C,P) :- parent(P,C))
%  Optional match constraints for regex filtering and arithmetic constraints
%
compile_single_predicate_rule_go(PredStr, HeadArgs, BodyPred, VarMap, FieldDelim, Unique, MatchConstraints, Constraints, GoCode) :-
    % Get the body predicate name and args
    BodyPred =.. [BodyPredName|BodyArgs],
    atom_string(BodyPredName, BodyPredStr),
    map_field_delimiter(FieldDelim, DelimChar),

    % Build capture mapping from match constraints
    % Map head argument positions to capture group positions
    % For each head arg, check if it appears in any capture group
    findall((HeadPos, CapIdx),
        (   nth1(HeadPos, HeadArgs, HeadArg),
            var(HeadArg),
            member(match(_, _, _, Groups), MatchConstraints),
            Groups \= [],
            nth1(CapIdx, Groups, GroupVar),
            HeadArg == GroupVar
        ),
        CaptureMapping),
    format('  Capture mapping (HeadPos -> CapIdx): ~w~n', [CaptureMapping]),

    % Build output format by checking three sources:
    % 1. Variables from BodyArgs -> field{pos}
    % 2. Variables from capture groups (via position mapping) -> cap{idx}
    % 3. Constants -> literal value
    findall(OutputPart,
        (   nth1(HeadPos, HeadArgs, HeadArg),
            (   var(HeadArg) ->
                (   % Check if it's from body args
                    nth1(BodyPos, BodyArgs, BodyArg),
                    HeadArg == BodyArg
                ->  format(atom(OutputPart), 'field~w', [BodyPos])
                ;   % Check if it's from capture groups (using position mapping)
                    member((HeadPos, CapIdx), CaptureMapping)
                ->  format(atom(OutputPart), 'cap~w', [CapIdx])
                ;   % Variable not found - should not happen
                    format('WARNING: Variable at position ~w not found in body or captures~n', [HeadPos]),
                    fail
                )
            ;   % Constant in head
                atom_string(HeadArg, HeadArgStr),
                format(atom(OutputPart), '"~s"', [HeadArgStr])
            )
        ),
        OutputParts),

    % Build Go expression with delimiter between parts
    build_go_concat_expr(OutputParts, DelimChar, OutputExpr),

    % Generate Go code that reads from stdin and processes records
    length(BodyArgs, NumFields),

    % For single-field records, use the entire line without splitting
    (   NumFields = 1 ->
        FieldAssignments = '\t\t\tfield1 := line',
        SplitCode = '',
        LenCheck = ''
    ;   % Multi-field records need splitting and length check
        generate_field_assignments(BodyArgs, FieldAssignments),
        format(atom(SplitCode), '\t\tparts := strings.Split(line, "~s")\n', [DelimChar]),
        format(atom(LenCheck), '\t\tif len(parts) == ~w {\n', [NumFields])
    ),

    % Generate match constraint code if present
    generate_go_match_code(MatchConstraints, HeadArgs, BodyArgs, MatchRegexDecls, MatchChecks, MatchCaptureCode),

    % Generate arithmetic constraint checks if present
    generate_go_constraint_code(Constraints, VarMap, ConstraintChecks),

    % Combine all checks (match + arithmetic)
    (   MatchChecks = '', ConstraintChecks = '' ->
        AllChecks = ''
    ;   MatchChecks = '' ->
        AllChecks = ConstraintChecks
    ;   ConstraintChecks = '' ->
        AllChecks = MatchChecks
    ;   atomic_list_concat([MatchChecks, '\n', ConstraintChecks], AllChecks)
    ),

    % Add capture extraction after checks if present
    (   MatchCaptureCode = '' ->
        AllChecksAndCaptures = AllChecks
    ;   AllChecks = '' ->
        AllChecksAndCaptures = MatchCaptureCode
    ;   atomic_list_concat([AllChecks, '\n', MatchCaptureCode], AllChecksAndCaptures)
    ),

    % Build complete Go code with optional constraints
    (   NumFields = 1 ->
        % Single field - no splitting needed
        (   AllChecksAndCaptures = '' ->
            format(string(GoCode), '
\t// Read from stdin and process ~s records
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\t~s
\t\tresult := ~s
\t\tif !seen[result] {
\t\t\tseen[result] = true
\t\t\tfmt.Println(result)
\t\t}
\t}
', [BodyPredStr, FieldAssignments, OutputExpr])
        ;   % With constraints (match and/or arithmetic)
            format(string(GoCode), '
\t// Read from stdin and process ~s records with filtering
~s
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\t~s
~s
\t\tresult := ~s
\t\tif !seen[result] {
\t\t\tseen[result] = true
\t\t\tfmt.Println(result)
\t\t}
\t}
', [BodyPredStr, MatchRegexDecls, FieldAssignments, AllChecksAndCaptures, OutputExpr])
        )
    ;   % Multi-field - needs splitting and length check
        (   AllChecksAndCaptures = '' ->
            format(string(GoCode), '
\t// Read from stdin and process ~s records
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, "~s")
\t\tif len(parts) == ~w {
\t\t\t~s
\t\t\tresult := ~s
\t\t\tif !seen[result] {
\t\t\t\tseen[result] = true
\t\t\t\tfmt.Println(result)
\t\t\t}
\t\t}
\t}
', [BodyPredStr, DelimChar, NumFields, FieldAssignments, OutputExpr])
        ;   % With constraints (match and/or arithmetic)
            format(string(GoCode), '
\t// Read from stdin and process ~s records with filtering
~s
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, "~s")
\t\tif len(parts) == ~w {
\t\t\t~s
~s
\t\t\tresult := ~s
\t\t\tif !seen[result] {
\t\t\t\tseen[result] = true
\t\t\t\tfmt.Println(result)
\t\t\t}
\t\t}
\t}
', [BodyPredStr, MatchRegexDecls, DelimChar, NumFields, FieldAssignments, AllChecksAndCaptures, OutputExpr])
        )
    ).

%% is_semantic_predicate(+Goal)
is_semantic_predicate(semantic_search(_, _, _)).
is_semantic_predicate(crawler_run(_, _)).

%% compile_semantic_rule_go(+PredStr, +HeadArgs, +Goal, -GoCode)
compile_semantic_rule_go(_PredStr, HeadArgs, Goal, GoCode) :-
    build_var_map(HeadArgs, VarMap),
    Goal =.. [GoalName | GoalArgs],
    
    Imports = '\t"fmt"\n\t"log"\n\n\t"unifyweaver/targets/go_runtime/search"\n\t"unifyweaver/targets/go_runtime/embedder"\n\t"unifyweaver/targets/go_runtime/storage"\n\t"unifyweaver/targets/go_runtime/crawler"',
    
    (   GoalName == semantic_search
    ->  GoalArgs = [Query, TopK, _Results],
        term_to_go_expr(Query, VarMap, QueryExpr),
        term_to_go_expr(TopK, VarMap, TopKExpr),
        
        format(string(Body), '
\t// Initialize runtime
\tstore, err := storage.NewStore("data.db")
\tif err != nil { log.Fatal(err) }
\tdefer store.Close()

\temb, err := embedder.NewHugotEmbedder("models/model.onnx", "all-MiniLM-L6-v2")
\tif err != nil { log.Fatal(err) }
\tdefer emb.Close()

\t// Embed query
\tqVec, err := emb.Embed(~s)
\tif err != nil { log.Fatal(err) }
\t
\t// Search
\tresults, err := search.Search(store, qVec, ~w)
\tif err != nil { log.Fatal(err) }

\tfor _, res := range results {
\t\tfmt.Printf("Result: %%s (Score: %%f)\\n", res.ID, res.Score)
\t}
', [QueryExpr, TopKExpr])
    ;   GoalName == crawler_run
    ->  GoalArgs = [Seeds, MaxDepth],
        term_to_go_expr(MaxDepth, VarMap, DepthExpr),
        
        (   is_list(Seeds)
        ->  maplist(atom_string, Seeds, SeedStrs),
            atomic_list_concat(SeedStrs, '", "', Inner),
            format(string(SeedsGo), '[]string{"~w"}', [Inner])
        ;   term_to_go_expr(Seeds, VarMap, SeedsExpr),
            SeedsGo = SeedsExpr
        ),

        format(string(Body), '
\t// Initialize runtime
\tstore, err := storage.NewStore("data.db")
\tif err != nil { log.Fatal(err) }
\tdefer store.Close()

\temb, err := embedder.NewHugotEmbedder("models/model.onnx", "all-MiniLM-L6-v2")
\tif err != nil { 
\t\tlog.Printf("Warning: Embeddings disabled: %%v\\n", err) 
\t\temb = nil
\t} else {
\t\tdefer emb.Close()
\t}

\tcraw := crawler.NewCrawler(store, emb)
\tcraw.Crawl(~w, int(~w))
', [SeedsGo, DepthExpr])
    ),
    
    format(string(GoCode), 'package main

import (
~s
)

func main() {
\t// Parse input arguments if needed (e.g. if HeadArgs are used)
\t// For now, we assume simple stdin/args or constants
~s}
', [Imports, Body]).

%% generate_field_assignments(+Args, -Code)
%  Generate field assignment statements
generate_field_assignments(Args, Code) :-
    findall(Assignment,
        (   nth1(N, Args, _),
            I is N - 1,
            format(atom(Assignment), 'field~w := parts[~w]', [N, I])
        ),
        Assignments),
    atomic_list_concat(Assignments, '\n\t\t\t', Code).

%% generate_go_match_code(+MatchConstraints, +HeadArgs, +BodyArgs, -RegexDecls, -MatchChecks, -CaptureCode)
%  Generate Go regex declarations, match checks, and capture extractions
generate_go_match_code([], _, _, "", "", "") :- !.
generate_go_match_code(MatchConstraints, HeadArgs, BodyArgs, RegexDecls, MatchChecks, CaptureCode) :-
    findall(Decl-Check-Capture,
        (   member(match(Var, Pattern, _Type, Groups), MatchConstraints),
            generate_single_match_code(Var, Pattern, Groups, HeadArgs, BodyArgs, Decl, Check, Capture)
        ),
        DeclsChecksCaps),
    findall(D, member(D-_-_, DeclsChecksCaps), Decls),
    findall(C, member(_-C-_, DeclsChecksCaps), Checks),
    findall(Cap, member(_-_-Cap, DeclsChecksCaps), Captures),
    atomic_list_concat(Decls, '\n', RegexDecls),
    atomic_list_concat(Checks, '\n', MatchChecks),
    % Filter out empty captures and join with newlines
    exclude(=(''), Captures, NonEmptyCaptures),
    (   NonEmptyCaptures = [] ->
        CaptureCode = ""
    ;   atomic_list_concat(NonEmptyCaptures, '\n', CaptureCode)
    ).

%% generate_single_match_code(+Var, +Pattern, +Groups, +HeadArgs, +BodyArgs, -Decl, -Check, -CaptureExtraction)
%  Generate regex declaration, check, and capture extraction for a single match constraint
generate_single_match_code(Var, Pattern, Groups, HeadArgs, BodyArgs, Decl, Check, CaptureExtraction) :-
    % Convert pattern to string
    (   atom(Pattern) ->
        atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),

    % Find which field variable contains Var
    % Check HeadArgs first, then BodyArgs
    (   nth1(FieldPos, HeadArgs, HeadVar),
        Var == HeadVar
    ->  format(atom(FieldVar), 'field~w', [FieldPos])
    ;   nth1(FieldPos, BodyArgs, BodyVar),
        Var == BodyVar
    ->  format(atom(FieldVar), 'field~w', [FieldPos])
    ;   FieldVar = 'line'  % If not in head or body args, match against the whole line
    ),

    % Generate unique regex variable name
    gensym(regex, RegexVar),

    % Generate regex compilation
    format(string(Decl), '\t~w := regexp.MustCompile(`~s`)', [RegexVar, PatternStr]),

    % Generate match check code
    (   Groups = [] ->
        % Boolean match - no captures
        format(string(Check), '\t\t\tif !~w.MatchString(~w) {\n\t\t\t\tcontinue\n\t\t\t}',
               [RegexVar, FieldVar]),
        CaptureExtraction = ""
    ;   % Match with capture groups
        length(Groups, NumGroups),
        NumGroups1 is NumGroups + 1,  % +1 for full match
        format(string(Check), '\t\t\tmatches := ~w.FindStringSubmatch(~w)\n\t\t\tif matches == nil || len(matches) != ~w {\n\t\t\t\tcontinue\n\t\t\t}',
               [RegexVar, FieldVar, NumGroups1]),
        % Generate capture extraction code: cap1 := matches[1], cap2 := matches[2], etc.
        findall(CapAssignment,
            (   between(1, NumGroups, CapIdx),
                format(atom(CapAssignment), '\t\t\tcap~w := matches[~w]', [CapIdx, CapIdx])
            ),
            CapAssignments),
        atomic_list_concat(CapAssignments, '\n', CaptureExtraction)
    ).

%% generate_go_constraint_code(+Constraints, +VarMap, -ConstraintChecks)
%  Generate Go code for arithmetic and comparison constraints
%  Includes type conversion from string fields to ints
generate_go_constraint_code([], _, "") :- !.
generate_go_constraint_code(Constraints, VarMap, ConstraintChecks) :-
    % First, collect which fields need int conversion
    findall(Pos,
        (   member(Constraint, Constraints),
            constraint_uses_field(Constraint, VarMap, Pos)
        ),
        FieldPoss),
    list_to_set(FieldPoss, UniqueFieldPoss),

    % Generate int conversion declarations
    findall(Decl,
        (   member(Pos, UniqueFieldPoss),
            format(atom(FieldName), 'field~w', [Pos]),
            format(atom(IntName), 'int~w', [Pos]),
            format(atom(Decl), '\t\t\t~w, err := strconv.Atoi(~w)\n\t\t\tif err != nil {\n\t\t\t\tcontinue\n\t\t\t}',
                   [IntName, FieldName])
        ),
        IntDecls),

    % Generate constraint checks
    findall(Check,
        (   member(Constraint, Constraints),
            constraint_to_go(Constraint, VarMap, GoConstraint),
            % Handle is/2 as assignment, others as conditions
            (   Constraint = is(_, _) ->
                % is/2 is an assignment
                format(atom(Check), '\t\t\t~w', [GoConstraint])
            ;   % Other constraints are conditions
                format(atom(Check), '\t\t\tif !(~w) {\n\t\t\t\tcontinue\n\t\t\t}', [GoConstraint])
            )
        ),
        Checks),

    % Combine declarations and checks
    append(IntDecls, Checks, AllParts),
    atomic_list_concat(AllParts, '\n', ConstraintChecks).

%% constraint_uses_field(+Constraint, +VarMap, -Pos)
%  Check if a constraint uses a field variable and return its position
constraint_uses_field(gt(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(gt(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(lt(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(lt(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(gte(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(gte(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(lte(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(lte(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(eq(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(eq(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(neq(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(neq(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(inequality(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(inequality(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.

%% ==============================================
%% MATCH CONSTRAINTS (REGEX WITH CAPTURES)
%% ==============================================

%% extract_match_constraints(+Body, -MatchConstraints)
%  Extract match/3 or match/4 predicates from rule body
%  match/3: match(Field, Pattern, Type)
%  match/4: match(Field, Pattern, Type, Captures)
extract_match_constraints(true, []) :- !.
extract_match_constraints((A, B), Constraints) :- !,
    extract_match_constraints(A, C1),
    extract_match_constraints(B, C2),
    append(C1, C2, Constraints).
% match/4 with capture groups
extract_match_constraints(match(Field, Pattern, Type, Captures), [match(Field, Pattern, Type, Captures)]) :- !.
% match/3 without captures
extract_match_constraints(match(Field, Pattern, Type), [match(Field, Pattern, Type, [])]) :- !.
extract_match_constraints(_, []).

%% compile_match_only_rule_go(+PredStr, +HeadArgs, +VarMap, +FieldDelim, +Unique, +MatchConstraints, -GoCode)
%  Compile rules with only match constraints (no body predicates)
%  Reads from stdin and filters based on regex patterns with capture groups
compile_match_only_rule_go(PredStr, HeadArgs, VarMap, FieldDelim, Unique, MatchConstraints, GoCode) :-
    % Get the first match constraint (for now, we support single match)
    (   MatchConstraints = [match(MatchField, Pattern, _Type, Captures)] ->
        % Generate code for regex matching with captures
        map_field_delimiter(FieldDelim, DelimChar),

        % Build capture variable assignments
        % FindStringSubmatch returns [fullMatch, group1, group2, ...]
        % So we need to map Captures list to matches[1], matches[2], etc.
        length(Captures, NumCaptures),
        findall(Assignment,
            (   between(1, NumCaptures, Idx),
                nth1(Idx, Captures, CaptureVar),
                member((Var, Pos), VarMap),
                CaptureVar == Var,
                format(atom(Assignment), '\t\t\tcap~w := matches[~w]', [Pos, Idx])
            ),
            CaptureAssignments),
        atomic_list_concat(CaptureAssignments, '\n', CaptureCode),

        % Build output expression using head args
        % Map each head arg to either original line or captured value
        findall(OutputPart,
            (   member(Arg, HeadArgs),
                (   % Check if this arg is the matched field
                    Arg == MatchField ->
                    OutputPart = 'line'
                ;   % Check if this arg is a captured variable
                    member((Var, Pos), VarMap),
                    Arg == Var,
                    member(Arg, Captures) ->
                    format(atom(OutputPart), 'cap~w', [Pos])
                ;   OutputPart = 'line'
                )
            ),
            OutputParts),
        build_go_concat_expr(OutputParts, DelimChar, OutputExpr),

        % Generate uniqueness check if needed
        (   Unique = true ->
            SeenMapCode = '\n\tseen := make(map[string]bool)',
            UniqueCode = '\n\t\t\tif !seen[result] {\n\t\t\t\tseen[result] = true\n\t\t\t\tfmt.Println(result)\n\t\t\t}'
        ;   SeenMapCode = '',
            UniqueCode = '\n\t\t\tfmt.Println(result)'
        ),

        % Generate the complete Go code
        format(atom(GoCode),
'\n\t// Read from stdin and process with regex pattern matching\n\n\tpattern := regexp.MustCompile(`~w`)\n\tscanner := bufio.NewScanner(os.Stdin)~w\n\t\n\tfor scanner.Scan() {\n\t\tline := scanner.Text()\n\t\tmatches := pattern.FindStringSubmatch(line)\n\t\tif matches != nil {\n~w\n\t\t\tresult := ~w~w\n\t\t}\n\t}\n',
            [Pattern, SeenMapCode, CaptureCode, OutputExpr, UniqueCode])
    ;   % Multiple or no match constraints
        length(HeadArgs, Arity),
        format(user_error,
               'Go target: multiple/no match constraints not supported for ~w/~w~n',
               [PredStr, Arity]),
        fail
    ).

%% build_go_concat_expr(+Parts, +Delimiter, -Expression)
%  Build a Go string concatenation expression with delimiters
%  e.g., [field1, field2] with ":" -> field1 + ":" + field2
build_go_concat_expr([], _, '""') :- !.
build_go_concat_expr([Single], _, Single) :- !.
build_go_concat_expr([First|Rest], Delim, Expr) :-
    build_go_concat_expr_rest(Rest, Delim, RestExpr),
    format(atom(Expr), '~w + "~s" + ~w', [First, Delim, RestExpr]).

build_go_concat_expr_rest([Single], _, Single) :- !.
build_go_concat_expr_rest([First|Rest], Delim, Expr) :-
    build_go_concat_expr_rest(Rest, Delim, RestExpr),
    format(atom(Expr), '~w + "~s" + ~w', [First, Delim, RestExpr]).

%% Helper functions
build_var_map(HeadArgs, VarMap) :-
    build_var_map_(HeadArgs, 1, VarMap).

build_var_map_([], _, []).
build_var_map_([Arg|Rest], Pos, [(Arg, Pos)|RestMap]) :-
    NextPos is Pos + 1,
    build_var_map_(Rest, NextPos, RestMap).

%% Skip match predicates when extracting regular predicates
extract_predicates(_:Goal, Preds) :- !, extract_predicates(Goal, Preds).
extract_predicates(match(_, _), []) :- !.
extract_predicates(match(_, _, _), []) :- !.
extract_predicates(match(_, _, _, _), []) :- !.

%% Skip constraint operators
extract_predicates(_ > _, []) :- !.
extract_predicates(_ < _, []) :- !.
extract_predicates(_ >= _, []) :- !.
extract_predicates(_ =< _, []) :- !.
extract_predicates(_ =:= _, []) :- !.
extract_predicates(_ =\= _, []) :- !.
extract_predicates(_ \= _, []) :- !.
extract_predicates(\=(_,  _), []) :- !.
extract_predicates(is(_, _), []) :- !.

extract_predicates(true, []) :- !.
extract_predicates((A, B), Predicates) :- !,
    extract_predicates(A, P1),
    extract_predicates(B, P2),
    append(P1, P2, Predicates).
extract_predicates(Goal, [Goal]) :-
    functor(Goal, Functor, _),
    Functor \= ',',
    Functor \= true,
    Functor \= match.

%% extract_match_constraints(+Body, -Constraints)
%  Extract all match/2, match/3, match/4 constraints from body
extract_match_constraints(true, []) :- !.
extract_match_constraints((A, B), Constraints) :- !,
    extract_match_constraints(A, C1),
    extract_match_constraints(B, C2),
    append(C1, C2, Constraints).
extract_match_constraints(match(Var, Pattern), [match(Var, Pattern, auto, [])]) :- !.
extract_match_constraints(match(Var, Pattern, Type), [match(Var, Pattern, Type, [])]) :- !.
extract_match_constraints(match(Var, Pattern, Type, Groups), [match(Var, Pattern, Type, Groups)]) :- !.
extract_match_constraints(_, []).

%% extract_constraints(+Body, -Constraints)
%  Extract all arithmetic and comparison constraints from body
%  Similar to extract_match_constraints but for operators like >, <, is/2, etc.
extract_constraints(true, []) :- !.
extract_constraints((A, B), Constraints) :- !,
    extract_constraints(A, C1),
    extract_constraints(B, C2),
    append(C1, C2, Constraints).
extract_constraints(Goal, []) :-
    var(Goal), !.
% Capture inequality constraints
extract_constraints(A \= B, [inequality(A, B)]) :- !.
extract_constraints(\=(A, B), [inequality(A, B)]) :- !.
% Capture arithmetic comparison constraints
extract_constraints(A > B, [gt(A, B)]) :- !.
extract_constraints(A < B, [lt(A, B)]) :- !.
extract_constraints(A >= B, [gte(A, B)]) :- !.
extract_constraints(A =< B, [lte(A, B)]) :- !.
extract_constraints(A =:= B, [eq(A, B)]) :- !.
extract_constraints(A =\= B, [neq(A, B)]) :- !.
extract_constraints(is(A, B), [is(A, B)]) :- !.
% Skip match predicates (handled separately)
extract_constraints(match(_, _), []) :- !.
extract_constraints(match(_, _, _), []) :- !.
extract_constraints(match(_, _, _, _), []) :- !.
% Skip other predicates
extract_constraints(Goal, []) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% constraint_to_go(+Constraint, +VarMap, -GoCode)
%  Convert a constraint to Go code
%  VarMap maps Prolog variables to field positions for Go
%  Numeric comparisons need strconv.Atoi for string-to-int conversion
constraint_to_go(inequality(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w != ~w', [GoA, GoB]).
constraint_to_go(gt(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w > ~w', [GoA, GoB]).
constraint_to_go(lt(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w < ~w', [GoA, GoB]).
constraint_to_go(gte(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w >= ~w', [GoA, GoB]).
constraint_to_go(lte(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w <= ~w', [GoA, GoB]).
constraint_to_go(eq(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w == ~w', [GoA, GoB]).
constraint_to_go(neq(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w != ~w', [GoA, GoB]).
constraint_to_go(is(A, B), VarMap, GoCode) :-
    % For is/2, we need to assign the result
    % e.g., Double is Age * 2 becomes: double := age * 2
    term_to_go_expr(A, VarMap, GoA),
    term_to_go_expr(B, VarMap, GoB),
    format(atom(GoCode), '~w := ~w', [GoA, GoB]).

%% term_to_go_expr_numeric(+Term, +VarMap, -GoExpr)
%  Convert a Prolog term to a Go expression for numeric contexts
%  Wraps field references with strconv.Atoi for type conversion
%
term_to_go_expr_numeric(Term, VarMap, GoExpr) :-
    var(Term), !,
    % It's a Prolog variable - convert to int
    (   member((Var, Pos), VarMap),
        Term == Var
    ->  % Need to convert string field to int
        format(atom(FieldName), 'field~w', [Pos]),
        format(atom(IntName), 'int~w', [Pos]),
        % We'll define intN variables in the constraint check code
        GoExpr = IntName
    ;   GoExpr = 'unknown'
    ).
term_to_go_expr_numeric(Term, _, Term) :-
    number(Term), !.
term_to_go_expr_numeric(Term, VarMap, GoExpr) :-
    compound(Term), !,
    Term =.. [Op, Left, Right],
    term_to_go_expr_numeric(Left, VarMap, GoLeft),
    term_to_go_expr_numeric(Right, VarMap, GoRight),
    go_operator(Op, GoOp),
    format(atom(GoExpr), '(~w ~w ~w)', [GoLeft, GoOp, GoRight]).
term_to_go_expr_numeric(Term, _, GoExpr) :-
    atom(Term), !,
    format(atom(GoExpr), '"~w"', [Term]).

%% term_to_go_expr(+Term, +VarMap, -GoExpr)
%  Convert a Prolog term to a Go expression using variable mapping
%  For string contexts (no type conversion)
%
term_to_go_expr(Term, VarMap, GoExpr) :-
    var(Term), !,
    % It's a Prolog variable - look it up in VarMap using identity check
    (   member((Var, Pos), VarMap),
        Term == Var
    ->  % Use field reference - will add type conversion in constraint_to_go if needed
        format(atom(GoExpr), 'field~w', [Pos])
    ;   % Variable not in map - use placeholder
        GoExpr = 'unknown'
    ).
term_to_go_expr(Term, _, GoExpr) :-
    atom(Term), !,
    % Atom constant - quote it for Go
    format(atom(GoExpr), '"~w"', [Term]).
term_to_go_expr(Term, _, Term) :-
    number(Term), !.
term_to_go_expr(Term, VarMap, GoExpr) :-
    compound(Term), !,
    Term =.. [Op, Left, Right],
    term_to_go_expr(Left, VarMap, GoLeft),
    term_to_go_expr(Right, VarMap, GoRight),
    % Map Prolog operators to Go operators
    go_operator(Op, GoOp),
    format(atom(GoExpr), '(~w ~w ~w)', [GoLeft, GoOp, GoRight]).
term_to_go_expr(Term, _, Term).

%% go_operator(+PrologOp, -GoOp)
%  Map Prolog operators to Go operators
go_operator(+, '+') :- !.
go_operator(-, '-') :- !.
go_operator(*, '*') :- !.
go_operator(/, '/') :- !.
go_operator(mod, '%') :- !.
go_operator(Op, Op).  % Default: use as-is

%% ============================================
%% MULTIPLE RULES COMPILATION
%% ============================================

%% compile_multiple_rules_to_go(+Pred, +Arity, +Clauses, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile multiple rules (OR pattern) to Go code
%
compile_multiple_rules_to_go(Pred, Arity, Clauses, _RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Check if all rules have compatible structure
    (   all_rules_compatible(Clauses, BodyPredFunctor, BodyArity) ->
        % All rules use the same body predicate with different match constraints
        format('  All rules use ~w/~w~n', [BodyPredFunctor, BodyArity]),

        % Collect all match constraints from all rules
        findall(Pattern,
            (   member(_Head-Body, Clauses),
                extract_match_constraints(Body, Constraints),
                member(match(_Var, Pattern, _Type, _Groups), Constraints)
            ),
            Patterns),

        % If we have patterns, combine them into OR regex
        (   Patterns \= [] ->
            % Combine patterns with | for OR
            atomic_list_concat(Patterns, '|', CombinedPattern),
            format('  Combined pattern: ~w~n', [CombinedPattern]),

            % Get a sample head and body to use for compilation
            Clauses = [Head-Body|_],
            Head =.. [_|HeadArgs],
            extract_predicates(Body, [BodyPred|_]),

            % Create a combined match constraint
            CombinedConstraint = [match(HeadArgs, CombinedPattern, auto, [])],

            % Compile as a single rule with combined match
            build_var_map(HeadArgs, VarMap),
            compile_single_predicate_rule_go(PredStr, HeadArgs, BodyPred, VarMap,
                                            FieldDelim, Unique, CombinedConstraint, GoCode)
        ;   % No match constraints - just multiple rules with same body
            format(user_error,
                   'Go target: multiple rules without match constraints not supported for ~w~n',
                   [PredStr]),
            fail
        )
    ;   % Rules not compatible for simple merging - different body predicates
        format('  Rules have different body predicates - compiling separately~n'),
        compile_different_body_rules_to_go(PredStr, Clauses, FieldDelim, Unique, GoCode)
    ).

%% all_rules_compatible(+Clauses, -BodyPredFunctor, -BodyArity)
%  Check if all rules use the same body predicate (compatible for merging)
all_rules_compatible(Clauses, BodyPredFunctor, BodyArity) :-
    maplist(get_body_predicate_info, Clauses, BodyInfos),
    BodyInfos = [BodyPredFunctor/BodyArity|Rest],
    maplist(==(BodyPredFunctor/BodyArity), Rest).

%% get_body_predicate_info(+Clause, -BodyPredInfo)
%  Extract body predicate functor/arity from a clause
get_body_predicate_info(_Head-Body, BodyPredFunctor/BodyArity) :-
    extract_predicates(Body, [BodyPred|_]),
    functor(BodyPred, BodyPredFunctor, BodyArity).

%% compile_different_body_rules_to_go(+PredStr, +Clauses, +FieldDelim, +Unique, -GoCode)
%  Compile multiple rules with different body predicates
%  Generates code that tries each rule pattern sequentially
compile_different_body_rules_to_go(PredStr, Clauses, FieldDelim, Unique, GoCode) :-
    map_field_delimiter(FieldDelim, DelimChar),

    % For each clause, generate the code to try that rule
    findall(RuleCode,
        (   member(Head-Body, Clauses),
            compile_single_rule_attempt(Head, Body, DelimChar, RuleCode)
        ),
        RuleCodes),

    % Combine all rule attempts
    atomic_list_concat(RuleCodes, '\n', AllRuleAttempts),

    % Build the complete Go code
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n',
        UniqueCheck = '\t\t\tif !seen[result] {\n\t\t\t\tseen[result] = true\n\t\t\t\tfmt.Println(result)\n\t\t\t}\n'
    ;   SeenDecl = '',
        UniqueCheck = '\t\t\tfmt.Println(result)\n'
    ),

    format(string(GoCode), '
\t// Read from stdin and try each rule pattern
\tscanner := bufio.NewScanner(os.Stdin)
~w\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, "~s")
\t\t
~s\t}
', [SeenDecl, DelimChar, AllRuleAttempts]).

%% compile_single_rule_attempt(+Head, +Body, +DelimChar, -RuleCode)
%  Generate Go code to try matching and transforming a single rule
compile_single_rule_attempt(Head, Body, DelimChar, RuleCode) :-
    % Extract head arguments
    Head =.. [_|HeadArgs],

    % Extract body predicate and constraints
    extract_predicates(Body, Predicates),
    extract_match_constraints(Body, MatchConstraints),
    extract_constraints(Body, Constraints),

    % Get body predicate info
    (   Predicates = [BodyPred] ->
        BodyPred =.. [BodyPredName|BodyArgs],
        length(BodyArgs, NumFields),
        atom_string(BodyPredName, BodyPredStr),

        % Build variable map
        build_var_map(HeadArgs, VarMap),

        % Build capture mapping
        findall((HeadPos, CapIdx),
            (   nth1(HeadPos, HeadArgs, HeadArg),
                var(HeadArg),
                member(match(_, _, _, Groups), MatchConstraints),
                Groups \= [],
                nth1(CapIdx, Groups, GroupVar),
                HeadArg == GroupVar
            ),
            CaptureMapping),

        % Build output expression
        findall(OutputPart,
            (   nth1(HeadPos, HeadArgs, HeadArg),
                (   var(HeadArg) ->
                    (   nth1(BodyPos, BodyArgs, BodyArg),
                        HeadArg == BodyArg
                    ->  format(atom(OutputPart), 'field~w', [BodyPos])
                    ;   member((HeadPos, CapIdx), CaptureMapping)
                    ->  format(atom(OutputPart), 'cap~w', [CapIdx])
                    ;   fail
                    )
                ;   atom_string(HeadArg, HeadArgStr),
                    format(atom(OutputPart), '"~s"', [HeadArgStr])
                )
            ),
            OutputParts),
        build_go_concat_expr(OutputParts, DelimChar, OutputExpr),

        % Determine which fields are actually used
        findall(UsedFieldNum,
            (   member(OutputPart, OutputParts),
                atom(OutputPart),
                atom_concat('field', NumAtom, OutputPart),
                atom_number(NumAtom, UsedFieldNum)
            ),
            UsedFields),

        % Generate field assignments only for used fields
        findall(Assignment,
            (   nth1(N, BodyArgs, _),
                member(N, UsedFields),
                I is N - 1,
                format(atom(Assignment), 'field~w := parts[~w]', [N, I])
            ),
            Assignments),
        atomic_list_concat(Assignments, '\n\t\t\t', FieldAssignments),

        % Generate match and constraint checks
        generate_go_match_code(MatchConstraints, HeadArgs, BodyArgs, MatchRegexDecls, MatchChecks, MatchCaptureCode),
        generate_go_constraint_code(Constraints, VarMap, ConstraintChecks),

        % Combine checks
        findall(CheckPart,
            (   MatchChecks \= '', member(CheckPart, [MatchChecks])
            ;   MatchCaptureCode \= '', member(CheckPart, [MatchCaptureCode])
            ;   ConstraintChecks \= '', member(CheckPart, [ConstraintChecks])
            ),
            CheckParts),
        atomic_list_concat(CheckParts, '\n', AllChecks),

        % Generate the rule attempt code
        (   AllChecks = '' ->
            format(string(RuleCode), '\t\t// Try rule: ~s/~w~n\t\tif len(parts) == ~w {~n\t\t\t~s~n\t\t\tresult := ~s~n\t\t\tif !seen[result] {~n\t\t\t\tseen[result] = true~n\t\t\t\tfmt.Println(result)~n\t\t\t}~n\t\t\tcontinue~n\t\t}~n',
                   [BodyPredStr, NumFields, NumFields, FieldAssignments, OutputExpr])
        ;   format(string(RuleCode), '\t\t// Try rule: ~s/~w~n\t\tif len(parts) == ~w {~n\t\t\t~s~n~s~n\t\t\tresult := ~s~n\t\t\tif !seen[result] {~n\t\t\t\tseen[result] = true~n\t\t\t\tfmt.Println(result)~n\t\t\t}~n\t\t\tcontinue~n\t\t}~n',
                   [BodyPredStr, NumFields, NumFields, FieldAssignments, AllChecks, OutputExpr])
        )
    ;   % No body predicate or multiple body predicates - skip
        RuleCode = '\t\t// Skipping rule without single body predicate\n'
    ).

%% ============================================
%% TAIL RECURSION COMPILATION
%% ============================================

%% is_tail_recursive_pattern(+Pred, +Clauses)
%  Check if clauses form a tail recursive pattern
%  Pattern: base case + recursive case where recursion is last goal
is_tail_recursive_pattern(Pred, Clauses) :-
    % Must have at least 2 clauses (base + recursive)
    length(Clauses, Len),
    Len >= 2,

    % Separate base and recursive clauses
    partition(is_recursive_clause_for(Pred), Clauses, RecClauses, BaseClauses),

    % Must have at least one base case and one recursive case
    RecClauses \= [],
    BaseClauses \= [],

    % Check that recursive clauses are tail recursive
    forall(member(_-Body, RecClauses),
           is_tail_recursive_body(Pred, Body)).

%% is_recursive_clause_for(+Pred, +Clause)
%  Check if clause is recursive (calls Pred in body)
is_recursive_clause_for(Pred, _Head-Body) :-
    contains_call_to(Pred, Body).

%% contains_call_to(+Pred, +Body)
%  Check if Body contains a call to Pred
contains_call_to(Pred, Body) :-
    (   Body =.. [Pred|_] -> true
    ;   Body = (_,_) ->
        (   Body = (A, B),
            (contains_call_to(Pred, A) ; contains_call_to(Pred, B))
        )
    ;   false
    ).

%% is_tail_recursive_body(+Pred, +Body)
%  Check if Body is tail recursive (Pred call is last goal)
is_tail_recursive_body(Pred, Body) :-
    get_last_goal(Body, LastGoal),
    functor(LastGoal, Pred, _).

%% get_last_goal(+Body, -LastGoal)
%  Extract the last goal from a body
get_last_goal((_, B), LastGoal) :- !, get_last_goal(B, LastGoal).
get_last_goal(Goal, Goal).

%% compile_tail_recursive_to_go(+Pred, +Arity, +Clauses, -GoCode)
%  Compile tail recursive predicate to Go iterative loop
compile_tail_recursive_to_go(Pred, Arity, Clauses, GoCode) :-
    atom_string(Pred, PredStr),

    % Separate base and recursive clauses
    partition(is_recursive_clause_for(Pred), Clauses, RecClauses, BaseClauses),

    % Extract base case value
    BaseClauses = [BaseHead-_|_],
    BaseHead =.. [_|BaseArgs],

    % Extract recursive pattern
    RecClauses = [RecHead-RecBody|_],
    RecHead =.. [_|RecArgs],

    % Determine accumulator pattern for arity 3: pred(Input, Acc, Result)
    (   Arity =:= 3 ->
        generate_ternary_tail_recursion_go(PredStr, BaseArgs, RecArgs, RecBody, GoCode)
    ;   Arity =:= 2 ->
        generate_binary_tail_recursion_go(PredStr, BaseArgs, RecArgs, RecBody, GoCode)
    ;   % Unsupported arity
        format(user_error,
               'Go target: tail recursion for arity ~w not supported for ~w~n',
               [Arity, PredStr]),
        fail
    ).

%% generate_ternary_tail_recursion_go(+PredStr, +BaseArgs, +RecArgs, +RecBody, -GoCode)
%  Generate Go code for arity-3 tail recursion: count([H|T], Acc, N) :- ...
generate_ternary_tail_recursion_go(PredStr, BaseArgs, _RecArgs, RecBody, GoCode) :-
    % Extract step operation from recursive body
    extract_step_operation(RecBody, StepOp),

    % Convert step operation to Go code
    step_op_to_go(StepOp, GoStepCode),

    % Extract base accumulator value (usually second argument)
    (   BaseArgs = [_, BaseAcc, _] ->
        format(atom(BaseAccStr), '~w', [BaseAcc])
    ;   BaseAccStr = '0'
    ),

    % Check if we need to parse the accumulator as an integer
    (   atom_number(BaseAccStr, _) ->
        AccInit = BaseAccStr
    ;   AccInit = 'acc'  % Use parameter if not a number
    ),

    % Generate complete Go function with iterative loop
    format(atom(GoCode), '
// ~w implements tail-recursive ~w with iterative loop
func ~w(n int, acc int) int {
\tcurrentAcc := acc
\tcurrentN := n
\t
\t// Iterative loop (tail recursion optimization)
\tfor currentN > 0 {
\t\t// Step operation
\t\t~s
\t\tcurrentN--
\t}
\t
\treturn currentAcc
}
', [PredStr, PredStr, PredStr, GoStepCode]).

%% generate_binary_tail_recursion_go(+PredStr, +BaseArgs, +RecArgs, +RecBody, -GoCode)
%  Generate Go code for arity-2 tail recursion
generate_binary_tail_recursion_go(PredStr, _BaseArgs, _RecArgs, _RecBody, _GoCode) :-
    format(user_error,
           'Go target: binary tail recursion not supported for ~w~n',
           [PredStr]),
    fail.

%% extract_step_operation(+Body, -StepOp)
%  Extract the accumulator step operation from recursive body
%  Looks for patterns like: Acc1 is Acc * N or Acc1 is Acc + N
%  Skips decrement operations like N1 is N - 1
extract_step_operation((Goal, RestBody), StepOp) :- !,
    (   is_accumulator_update(Goal) ->
        Goal = (_ is Expr),
        StepOp = arithmetic(Expr)
    ;   extract_step_operation(RestBody, StepOp)
    ).
extract_step_operation(Goal, arithmetic(Expr)) :-
    is_accumulator_update(Goal), !,
    Goal = (_ is Expr).
extract_step_operation(_, unknown).

%% is_accumulator_update(+Goal)
%  Check if goal is an accumulator update (not a simple decrement)
is_accumulator_update(_ is Expr) :-
    \+ is_simple_decrement(Expr).

%% is_simple_decrement(+Expr)
%  Check if expression is just a simple decrement like N-1
%  Note: Prolog parses N-1 as +(N, -1) so we check for that too
is_simple_decrement(_ - 1) :- !.
is_simple_decrement(_ - _Const) :- !.
is_simple_decrement(_ + (-1)) :- !.
is_simple_decrement(_ + Const) :- integer(Const), Const < 0, !.

%% step_op_to_go(+StepOp, -GoCode)
%  Convert Prolog step operation to Go code
step_op_to_go(arithmetic(_Acc + Const), GoCode) :-
    integer(Const), !,
    format(atom(GoCode), 'currentAcc += ~w', [Const]).
step_op_to_go(arithmetic(_Acc + _N), 'currentAcc += currentN') :- !.
step_op_to_go(arithmetic(_Acc * _N), 'currentAcc *= currentN') :- !.
step_op_to_go(arithmetic(_Acc - _N), 'currentAcc -= currentN') :- !.
step_op_to_go(unknown, 'currentAcc += 1') :- !.
step_op_to_go(_, 'currentAcc += 1').  % Fallback

%% ============================================
%% GO PROGRAM GENERATION
%% ============================================

%% generate_go_program(+Pred, +Arity, +RecordDelim, +FieldDelim, +Quoting, +EscapeChar, +NeedsStdin, +NeedsRegexp, +NeedsStrings, +NeedsStrconv, +Body, -GoCode)
%  Generate complete Go program with imports and main function
%
generate_go_program(Pred, Arity, RecordDelim, FieldDelim, Quoting, EscapeChar, NeedsStdin, NeedsRegexp, NeedsStrings, NeedsStrconv, Body, GoCode) :-
    atom_string(Pred, PredStr),

    % Generate imports based on what's needed
    (   NeedsStdin ->
        % Build import list for stdin processing
        findall(Import,
            (   member(Pkg, [bufio, fmt, os]),
                format(atom(Import), '\t"~w"', [Pkg])
            ;   NeedsRegexp,
                format(atom(Import), '\t"regexp"', [])
            ;   NeedsStrings,
                format(atom(Import), '\t"strings"', [])
            ;   NeedsStrconv,
                format(atom(Import), '\t"strconv"', [])
            ),
            ImportList),
        list_to_set(ImportList, UniqueImports),  % Remove duplicates
        atomic_list_concat(UniqueImports, '\n', Imports)
    ;   NeedsRegexp ->
        Imports = '\t"fmt"\n\t"regexp"'
    ;   % Facts only need fmt
        Imports = '\t"fmt"'
    ),

    % Generate program template
    format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, Body]).

%% ============================================
%% UTILITY FUNCTIONS
%% ============================================

%% ============================================
%% DATABASE READ MODE COMPILATION
%% ============================================

%% compile_database_read_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate to read from bbolt database and output as JSON
%
compile_database_read_mode(Pred, Arity, Options, GoCode) :-
    % Get database options
    option(db_file(DbFile), Options, 'data.db'),
    option(db_bucket(BucketName), Options, Pred),
    atom_string(BucketName, BucketStr),
    option(include_package(IncludePackage), Options, true),

    % Generate database read code
    format(string(Body), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Read all records from bucket
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\t// Deserialize JSON record
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\tfmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\\n", err)
\t\t\t\treturn nil // Continue with next record
\t\t\t}

\t\t\t// Output as JSON
\t\t\toutput, err := json.Marshal(data)
\t\t\tif err != nil {
\t\t\t\tfmt.Fprintf(os.Stderr, "Error marshaling output: %v\\n", err)
\t\t\t\treturn nil // Continue with next record
\t\t\t}

\t\t\tfmt.Println(string(output))
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}
', [DbFile, BucketStr, BucketStr]),

    % Wrap in package if requested
    (   IncludePackage = true
    ->  format(string(GoCode), 'package main

import (
\t"encoding/json"
\t"fmt"
\t"os"

\tbolt "go.etcd.io/bbolt"
)

func main() {
~s}
', [Body])
    ;   GoCode = Body
    ).

%% ============================================
%% JSON INPUT MODE COMPILATION
%% ============================================

%% compile_json_input_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate with JSON input (JSONL format)
%  Reads JSON lines from stdin, extracts fields, outputs in configured format
%
compile_json_input_mode(Pred, Arity, Options, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),

    % Get predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),

    (   Clauses = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    ;   Clauses = [SingleHead-SingleBody] ->
        % Single clause - extract JSON field mappings
        format('  Clause: ~w~n', [SingleHead-SingleBody]),
        extract_json_field_mappings(SingleBody, FieldMappings),
        format('  Field mappings: ~w~n', [FieldMappings]),

        % Check for schema option
        SingleHead =.. [_|HeadArgs],
        (   option(json_schema(SchemaName), Options)
        ->  % Typed compilation with schema
            format('  Using schema: ~w~n', [SchemaName]),
            compile_json_to_go_typed(HeadArgs, FieldMappings, SchemaName, FieldDelim, Unique, CoreBody)
        ;   % Untyped compilation (current behavior)
            compile_json_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, CoreBody)
        ),

        % Check if database backend is specified
        (   option(db_backend(bbolt), Options)
        ->  % Wrap core body with database operations
            format('  Database: bbolt~n'),
            wrap_with_database(CoreBody, FieldMappings, Pred, Options, ScriptBody),
            NeedsDatabase = true
        ;   ScriptBody = CoreBody,
            NeedsDatabase = false
        ),

        % Check if helpers are needed (for the package wrapping)
        (   member(nested(_, _), FieldMappings)
        ->  generate_nested_helper(HelperFunc),
            HelperSection = HelperFunc
        ;   HelperSection = ''
        )
    ;   format('ERROR: Multiple clauses not yet supported for JSON input mode~n'),
        fail
    ),

    % Wrap in package if requested
    (   IncludePackage ->
        % Generate imports based on what's needed
        (   NeedsDatabase = true
        ->  BaseImports = ["bufio", "encoding/json", "fmt", "os", "bolt \"go.etcd.io/bbolt\""]
        ;   BaseImports = ["bufio", "encoding/json", "fmt", "os"]
        ),
        
        (   option(json_schema(SchemaName), Options), schema_needs_strings(SchemaName)
        ->  append(BaseImports, ["strings"], AllImportsList)
        ;   AllImportsList = BaseImports
        ),
        
        maplist(format_import, AllImportsList, ImportLines),
        atomic_list_concat(ImportLines, '\n', Imports),

        (   HelperSection = '' ->
            format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, ScriptBody])
        ;   format(string(GoCode), 'package main

import (
~s
)

~s

func main() {
~s}
', [Imports, HelperSection, ScriptBody])
        )
    ;   GoCode = ScriptBody
    ).

%% compile_parallel_json_input_mode(+Pred, +Arity, +Options, +Workers, -GoCode)
%  Compile predicate with JSON input for parallel execution
compile_parallel_json_input_mode(Pred, Arity, Options, Workers, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),

    % Get predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),

    (   Clauses = [SingleHead-SingleBody] ->
        % Single clause - extract JSON field mappings
        extract_json_field_mappings(SingleBody, FieldMappings),
        
        SingleHead =.. [_|HeadArgs],
        compile_parallel_json_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, Workers, ScriptBody),

        % Check if helpers are needed
        (   member(nested(_, _), FieldMappings)
        ->  generate_nested_helper(HelperFunc),
            HelperSection = HelperFunc
        ;   HelperSection = ''
        )
    ;   format('ERROR: Multiple clauses not yet supported for parallel JSON input mode~n'),
        fail
    ),

    % Wrap in package if requested
    (   IncludePackage ->
        BaseImports = ["bufio", "encoding/json", "fmt", "os", "sync"],
        
        (   option(json_schema(SchemaName), Options), schema_needs_strings(SchemaName)
        ->  append(BaseImports, ["strings"], AllImportsList)
        ;   AllImportsList = BaseImports
        ),
        
        maplist(format_import, AllImportsList, ImportLines),
        atomic_list_concat(ImportLines, '\n', Imports),
        (   HelperSection = '' ->
            format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, ScriptBody])
        ;   format(string(GoCode), 'package main

import (
~s
)

~s

func main() {
~s}
', [Imports, HelperSection, ScriptBody])
        )
    ;   GoCode = ScriptBody
    ).

%% compile_parallel_json_to_go(+HeadArgs, +Operations, +FieldDelim, +Unique, +Workers, -GoCode)
%  Generate Go code for parallel JSON processing
compile_parallel_json_to_go(HeadArgs, Operations, FieldDelim, Unique, Workers, GoCode) :-
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate processing code (recursive)
    generate_parallel_json_processing(Operations, HeadArgs, DelimChar, Unique, 1, [], ProcessingCode),

    (   Unique = true 
    ->  UniqueVars = "var seenMutex sync.Mutex\n\tseen := make(map[string]bool)" 
    ;   UniqueVars = ""
    ),

    format(string(GoCode), '
	// Parallel execution with ~w workers
	jobs := make(chan []byte, 100)
	var wg sync.WaitGroup
	var outputMutex sync.Mutex
	~s

	// Start workers
	for i := 0; i < ~w; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for lineBytes := range jobs {
				var data map[string]interface{}
				if err := json.Unmarshal(lineBytes, &data); err != nil {
					continue
				}
				
				~s
			}
		}()
	}

	// Scanner loop
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		// Copy bytes because scanner.Bytes() is reused
		b := make([]byte, len(scanner.Bytes()))
		copy(b, scanner.Bytes())
		jobs <- b
	}
	close(jobs)
	wg.Wait()
', [Workers, UniqueVars, Workers, ProcessingCode]).

%% generate_parallel_json_processing(+Operations, +HeadArgs, +Delim, +Unique, +VIdx, +VarMap, -Code)
generate_parallel_json_processing([], HeadArgs, Delim, Unique, _, VarMap, Code) :-
    !,
    generate_json_output_from_map(HeadArgs, VarMap, Delim, OutputExpr),
    (   Unique = true ->
        format(string(Code), '
				result := ~s
				seenMutex.Lock()
				if !seen[result] {
					seen[result] = true
					outputMutex.Lock()
					fmt.Println(result)
					outputMutex.Unlock()
				}
				seenMutex.Unlock()', [OutputExpr])
    ;   format(string(Code), '
				result := ~s
				outputMutex.Lock()
				fmt.Println(result)
				outputMutex.Unlock()', [OutputExpr])
    ).

generate_parallel_json_processing([Op|Rest], HeadArgs, Delim, Unique, VIdx, VarMap, Code) :-
    % Reuse existing logic for extraction, just recurse to parallel version
    NextVIdx is VIdx + 1,
    (   Op = nested(Path, Var) ->
        format(atom(GoVar), 'v~w', [VIdx]),
        generate_nested_extraction_code(Path, 'data', GoVar, ExtractCode),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = Field-Var ->
        format(atom(GoVar), 'v~w', [VIdx]),
        atom_string(Field, FieldStr),
        generate_flat_field_extraction(FieldStr, GoVar, ExtractCode),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = extract(SourceVar, Path, Var) ->
        (   lookup_var_identity(SourceVar, VarMap, SourceGoVar) -> true
        ;   format('ERROR: Source variable not found in map: ~w~n', [SourceVar]), fail
        ),
        format(atom(GoVar), 'v~w', [VIdx]),
        format(string(ExtractCode), '
				var ~w interface{}
				if val, ok := ~w.(map[string]interface{}); ok {
					if v, ok := val["~w"]; ok {
						~w = v
					} else {
						continue
					}
				} else {
					continue
				}', [GoVar, SourceGoVar, Path, GoVar]),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = iterate(ListVar, ItemVar) ->
        (   lookup_var_identity(ListVar, VarMap, ListGoVar) -> true
        ;   format('ERROR: List variable not found in map: ~w~n', [ListVar]), fail
        ),
        format(atom(ItemGoVar), 'v~w', [VIdx]),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(ItemVar, ItemGoVar)|VarMap], LoopBody),
        format(string(Code), '
				if listVal, ok := ~w.([]interface{}); ok {
					for _, itemVal := range listVal {
						~w := itemVal
						~s
					}
				}', [ListGoVar, ItemGoVar, LoopBody])
    ).

%% wrap_with_database(+CoreBody, +FieldMappings, +Pred, +Options, -WrappedBody)
%  Wrap core extraction code with database operations
%
wrap_with_database(CoreBody, FieldMappings, Pred, Options, WrappedBody) :-
    % Get database options
    option(db_file(DbFile), Options, 'data.db'),
    option(db_bucket(BucketName), Options, Pred),
    atom_string(BucketName, BucketStr),

    % Determine key field (use first field by default)
    (   option(db_key_field(KeyField), Options)
    ->  true
    ;   FieldMappings = [KeyField-_|_]
    ->  true
    ;   FieldMappings = [nested(Path, _)|_],
        last(Path, KeyField)
    ),

    % Find which field position is the key
    (   nth1(KeyPos, FieldMappings, KeyField-_)
    ->  true
    ;   nth1(KeyPos, FieldMappings, nested(KeyPath, _)),
        last(KeyPath, KeyField)
    ->  true
    ;   KeyPos = 1  % Default to first field
    ),

    format(atom(KeyVar), 'field~w', [KeyPos]),

    % Create storage code block
    format(string(StorageCode), '\t\t// Store in database
\t\terr = db.Update(func(tx *bolt.Tx) error {
\t\t\tbucket := tx.Bucket([]byte("~s"))
\t\t\tkey := []byte(fmt.Sprintf("%v", ~s))
\t\t\t
\t\t\t// Store full JSON record
\t\t\tvalue, err := json.Marshal(data)
\t\t\tif err != nil {
\t\t\t\treturn err
\t\t\t}
\t\t\t
\t\t\treturn bucket.Put(key, value)
\t\t})
\t\t
\t\tif err != nil {
\t\t\terrorCount++
\t\t\tfmt.Fprintf(os.Stderr, "Database error: %v\\n", err)
\t\t\tcontinue
\t\t}
\t\t
\t\trecordCount++', [BucketStr, KeyVar]),

    % Remove output block and inject storage code
    split_string(CoreBody, "\n", "", Lines),
    filter_and_replace_lines(Lines, StorageCode, KeyPos, FilteredLines),
    atomics_to_string(FilteredLines, "\n", CleanedCore),

    % Generate wrapped code with storage inside loop
    format(string(WrappedBody), '\t// Open database
\tdb, err := bolt.Open("~s", 0600, nil)
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Create bucket
\terr = db.Update(func(tx *bolt.Tx) error {
\t\t_, err := tx.CreateBucketIfNotExists([]byte("~s"))
\t\treturn err
\t})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error creating bucket: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Process records
\trecordCount := 0
\terrorCount := 0

~s

\t// Summary
\tfmt.Fprintf(os.Stderr, "Stored %d records, %d errors\\n", recordCount, errorCount)
', [DbFile, BucketStr, CleanedCore]).

% Helper to filter output lines and inject storage code
filter_and_replace_lines(Lines, StorageCode, KeyPos, Result) :-
    filter_and_replace_lines(Lines, StorageCode, KeyPos, [], Result).

filter_and_replace_lines([], _StorageCode, _KeyPos, Acc, Result) :-
    reverse(Acc, Result).
filter_and_replace_lines([Line|Rest], StorageCode, KeyPos, Acc, Result) :-
    (   % Skip seen map declaration
        sub_string(Line, _, _, _, "seen := make(map[string]bool)")
    ->  filter_and_replace_lines(Rest, StorageCode, KeyPos, Acc, Result)
    ;   % Skip result formatting
        sub_string(Line, _, _, _, "result := fmt.Sprintf")
    ->  filter_and_replace_lines(Rest, StorageCode, KeyPos, Acc, Result)
    ;   % Skip seen check and inject storage code instead
        sub_string(Line, _, _, _, "if !seen[result]")
    ->  % Skip the entire if block (this line + seen[result]=true + println + closing brace)
        Rest = [_SeenTrue, _Println, _CloseBrace|RestAfterBlock],
        % Inject storage code
        filter_and_replace_lines(RestAfterBlock, StorageCode, KeyPos, [StorageCode|Acc], Result)
    ;   % Replace non-key field variables with _ (blank identifier)
        % Match lines like: "\t\tfieldN, fieldNOk := data[...]"
        sub_string(Line, _, _, _, "field"),
        replace_unused_field_var(Line, KeyPos, NewLine),
        NewLine \= Line  % Only if replacement was made
    ->  filter_and_replace_lines(Rest, StorageCode, KeyPos, [NewLine|Acc], Result)
    ;   % Keep all other lines
        filter_and_replace_lines(Rest, StorageCode, KeyPos, [Line|Acc], Result)
    ).

% Replace fieldN with _ if N != KeyPos
replace_unused_field_var(Line, KeyPos, NewLine) :-
    % Try to replace field1, field2, field3, etc. up to field9
    between(1, 9, N),
    N \= KeyPos,
    (   % Pattern 1: "fieldN," for untyped extraction
        format(atom(FieldVar), 'field~w,', [N]),
        sub_string(Line, Before, Len, After, FieldVar)
    ->  % Replace with "_,"
        sub_string(Line, 0, Before, _, Prefix),
        string_concat(Prefix, "_,", NewPrefix),
        Skip is Before + Len,
        sub_string(Line, Skip, _, 0, Suffix),
        string_concat(NewPrefix, Suffix, NewLine),
        !
    ;   % Pattern 2: "fieldN := " for typed final assignment
        format(atom(FieldAssign), 'field~w := ', [N]),
        sub_string(Line, Before2, Len2, After2, FieldAssign),
        % Replace entire line with "_ = " version
        sub_string(Line, 0, Before2, _, Prefix2),
        string_concat(Prefix2, "_ = ", NewPrefix2),
        Skip2 is Before2 + Len2,
        sub_string(Line, Skip2, _, 0, Suffix2),
        string_concat(NewPrefix2, Suffix2, NewLine),
        !
    ;   fail
    ).
replace_unused_field_var(Line, _, Line).  % No replacement needed

%% ============================================
%% JSON OUTPUT MODE
%% ============================================

%% compile_json_output_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile a predicate to generate JSON output
%  Reads delimiter-based input and generates JSON
%
compile_json_output_mode(Pred, Arity, Options, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),

    % Get field names from options or generate defaults
    (   option(json_fields(FieldNames), Options)
    ->  true
    ;   % Generate default field names from predicate
        functor(Head, Pred, Arity),
        Head =.. [_|Args],
        generate_default_field_names(Args, 1, FieldNames)
    ),

    % Generate struct definition
    atom_string(Pred, PredStr),
    upcase_atom(Pred, UpperPred),
    generate_json_struct(UpperPred, FieldNames, StructDef),

    % Generate parsing and output code
    compile_json_output_to_go(UpperPred, FieldNames, FieldDelim, OutputBody),

    % Wrap in package if requested
    (   IncludePackage ->
        format(string(GoCode), 'package main

import (
\t"bufio"
\t"encoding/json"
\t"fmt"
\t"os"
\t"strings"
\t"sync"
\t"strconv"
)

~s

func main() {
~s}
', [StructDef, OutputBody])
    ;   format(string(GoCode), '~s~n~s', [StructDef, OutputBody])
    ).

%% generate_default_field_names(+Args, +StartNum, -FieldNames)
%  Generate default field names (Field1, Field2, ...)
%
generate_default_field_names([], _, []).
generate_default_field_names([_|Args], N, [FieldName|Rest]) :-
    format(atom(FieldName), 'Field~w', [N]),
    N1 is N + 1,
    generate_default_field_names(Args, N1, Rest).

%% generate_json_struct(+StructName, +FieldNames, -StructDef)
%  Generate Go struct definition with JSON tags
%
generate_json_struct(StructName, FieldNames, StructDef) :-
    findall(FieldLine,
        (   nth1(Pos, FieldNames, FieldName),
            upcase_atom(FieldName, UpperField),
            downcase_atom(FieldName, LowerField),
            format(atom(FieldLine), '\t~w interface{} `json:"~w"`', [UpperField, LowerField])
        ),
        FieldLines),
    atomic_list_concat(FieldLines, '\n', FieldsStr),
    format(atom(StructDef), 'type ~wRecord struct {\n~w\n}', [StructName, FieldsStr]).

%% compile_json_output_to_go(+StructName, +FieldNames, +FieldDelim, -GoCode)
%  Generate Go code to read delimited input and output JSON
%
compile_json_output_to_go(StructName, FieldNames, FieldDelim, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate field count and parsing code
    length(FieldNames, NumFields),
    generate_field_parsing(FieldNames, 1, ParseCode),

    % Generate struct initialization
    generate_struct_init(StructName, FieldNames, StructInit),

    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\t
\tfor scanner.Scan() {
\t\tparts := strings.Split(scanner.Text(), "~s")
\t\tif len(parts) != ~w {
\t\t\tcontinue
\t\t}
\t\t
~s
\t\t
\t\trecord := ~s
\t\t
\t\tjsonBytes, err := json.Marshal(record)
\t\tif err != nil {
\t\t\tcontinue
\t\t}
\t\tfmt.Println(string(jsonBytes))
\t}
', [DelimChar, NumFields, ParseCode, StructInit]).

%% generate_field_parsing(+FieldNames, +StartPos, -ParseCode)
%  Generate code to parse fields from split parts
%  Tries to convert to numbers if possible, otherwise uses string
%
generate_field_parsing([], _, '').
generate_field_parsing([FieldName|Rest], Pos, ParseCode) :-
    Pos0 is Pos - 1,
    upcase_atom(FieldName, UpperField),
    downcase_atom(FieldName, LowerField),

    % Generate parsing code that tries numeric conversion
    format(atom(FieldParse), '\t\t// Parse ~w
\t\t~w := parts[~w]
\t\tvar ~wValue interface{}
\t\tif intVal, err := strconv.Atoi(~w); err == nil {
\t\t\t~wValue = intVal
\t\t} else if floatVal, err := strconv.ParseFloat(~w, 64); err == nil {
\t\t\t~wValue = floatVal
\t\t} else if boolVal, err := strconv.ParseBool(~w); err == nil {
\t\t\t~wValue = boolVal
\t\t} else {
\t\t\t~wValue = ~w
\t\t}',
        [LowerField, LowerField, Pos0, LowerField, LowerField,
         LowerField, LowerField, LowerField, LowerField, LowerField,
         LowerField, LowerField]),

    Pos1 is Pos + 1,
    generate_field_parsing(Rest, Pos1, RestParse),

    (   RestParse = ''
    ->  ParseCode = FieldParse
    ;   format(atom(ParseCode), '~s~n~s', [FieldParse, RestParse])
    ).

%% generate_struct_init(+StructName, +FieldNames, -StructInit)
%  Generate struct initialization code
%
generate_struct_init(StructName, FieldNames, StructInit) :-
    findall(FieldInit,
        (   member(FieldName, FieldNames),
            upcase_atom(FieldName, UpperField),
            downcase_atom(FieldName, LowerField),
            format(atom(FieldInit), '~w: ~wValue', [UpperField, LowerField])
        ),
        FieldInits),
    atomic_list_concat(FieldInits, ', ', FieldsStr),
    format(atom(StructInit), '~wRecord{~w}', [StructName, FieldsStr]).

%% extract_json_field_mappings(+Body, -FieldMappings)
%  Extract field-to-variable mappings from json_record([field-Var, ...]) and json_get(Path, Var)
%
extract_json_field_mappings(Body, FieldMappings) :-
    extract_json_operations(Body, Operations),
    (   Operations = [] ->
        format('WARNING: Body does not contain json_record/1 or json_get/2: ~w~n', [Body]),
        FieldMappings = []
    ;   FieldMappings = Operations
    ).

%% extract_json_operations(+Body, -Operations)
%  Extract all JSON operations from body (handles conjunction)
%
extract_json_operations(_:Goal, Ops) :- !,
    extract_json_operations(Goal, Ops).
extract_json_operations((A, B), Ops) :- !,
    extract_json_operations(A, OpsA),
    extract_json_operations(B, OpsB),
    append(OpsA, OpsB, Ops).
extract_json_operations(json_record(Fields), RecordOps) :- !,
    extract_field_list(Fields, RecordOps).
extract_json_operations(json_get(Path, Var), [nested(Path, Var)]) :- !.
extract_json_operations(json_get(Source, Path, Var), [extract(Source, Path, Var)]) :- !,
    var(Source).
extract_json_operations(json_array_member(List, Item), [iterate(List, Item)]) :- !.
extract_json_operations(_, []).


extract_field_list([], []).
extract_field_list([Field-Var|Rest], [Field-Var|Mappings]) :- !,
    extract_field_list(Rest, Mappings).
extract_field_list([Other|Rest], Mappings) :-
    format('WARNING: Unexpected field format: ~w~n', [Other]),
    extract_field_list(Rest, Mappings).

%% compile_json_to_go(+HeadArgs, +Operations, +FieldDelim, +Unique, -GoCode)
%  Generate Go code for JSON input mode (recursive for arrays)
%
compile_json_to_go(HeadArgs, Operations, FieldDelim, Unique, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate processing code (recursive)
    % Initial VarMap contains 'data' -> 'data'
    generate_json_processing(Operations, HeadArgs, DelimChar, Unique, 1, [], ProcessingCode),

    % Build the loop code
    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
\t\t
~s
\t}
', [ProcessingCode]).

%% generate_json_processing(+Operations, +HeadArgs, +Delim, +Unique, +VIdx, +VarMap, -Code)
%  Generate nested Go code for JSON operations
generate_json_processing([], HeadArgs, Delim, Unique, _, VarMap, Code) :-
    !,
    % No more operations - generate output
    generate_json_output_from_map(HeadArgs, VarMap, Delim, OutputExpr),
    (   Unique = true ->
        format(string(Code), '
\t\tresult := ~s
\t\tif !seen[result] {
\t\t\tseen[result] = true
\t\t\tfmt.Println(result)
\t\t}', [OutputExpr])
    ;   format(string(Code), '
\t\tresult := ~s
\t\tfmt.Println(result)', [OutputExpr])
    ).

generate_json_processing([Op|Rest], HeadArgs, Delim, Unique, VIdx, VarMap, Code) :-
    NextVIdx is VIdx + 1,
    (   Op = nested(Path, Var) ->
        format(atom(GoVar), 'v~w', [VIdx]),
        generate_nested_extraction_code(Path, 'data', GoVar, ExtractCode),
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = Field-Var ->
        format(atom(GoVar), 'v~w', [VIdx]),
        generate_nested_extraction_code(Field, 'data', GoVar, ExtractCode),
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = extract(SourceVar, Path, Var) ->
        (   lookup_var_identity(SourceVar, VarMap, SourceGoVar) -> true
        ;   format('ERROR: Source variable not found in map: ~w~n', [SourceVar]), fail
        ),
        format(atom(GoVar), 'v~w', [VIdx]),
        % Source must be a map
        format(string(ExtractCode), '
\t\tsourceMap~w, ok := ~w.(map[string]interface{})
\t\tif !ok { continue }', [VIdx, SourceGoVar]),
        
        format(atom(SourceMapVar), 'sourceMap~w', [VIdx]),
        generate_nested_extraction_code(Path, SourceMapVar, GoVar, InnerExtract),
        
        format(string(FullExtract), '~s\n~s', [ExtractCode, InnerExtract]),
        format(atom(FixedExtract), FullExtract, []),
        
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [FixedExtract, RestCode])

    ;   Op = iterate(ListVar, ItemVar) ->
        (   lookup_var_identity(ListVar, VarMap, ListGoVar) -> true
        ;   format('ERROR: List variable not found in map: ~w~n', [ListVar]), fail
        ),
        format(atom(ItemGoVar), 'v~w', [VIdx]),
        
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(ItemVar, ItemGoVar)|VarMap], InnerCode),
        
        format(string(Code), '
\t\tif listVal~w, ok := ~w.([]interface{}); ok {
\t\t\tfor _, itemVal~w := range listVal~w {
\t\t\t\t~w := itemVal~w
~s
\t\t\t}
\t\t}', [VIdx, ListGoVar, VIdx, VIdx, ItemGoVar, VIdx, InnerCode])
    ;   % Fallback for unknown ops
        format('WARNING: Unknown JSON operation: ~w~n', [Op]),
        generate_json_processing(Rest, HeadArgs, Delim, Unique, VIdx, VarMap, Code)
    ).

generate_nested_extraction_code(Path, Source, Target, Code) :-
    (   is_list(Path) -> PathList = Path ; PathList = [Path] ),
    maplist(atom_string, PathList, PathStrs),
    atomic_list_concat(PathStrs, '", "', PathStr),
    format(string(Code), '
\t\t~w, found := getNestedField(~w, []string{"~s"})
\t\tif !found { continue }', [Target, Source, PathStr]).

generate_json_output_from_map(HeadArgs, VarMap, DelimChar, OutputExpr) :-
    maplist(arg_to_go_var(VarMap), HeadArgs, GoVars),
    length(HeadArgs, NumArgs),
    findall('%v', between(1, NumArgs, _), FormatParts),
    atomic_list_concat(FormatParts, DelimChar, FormatStr),
    atomic_list_concat(GoVars, ', ', VarList),
    format(atom(OutputExpr), 'fmt.Sprintf("~s", ~s)', [FormatStr, VarList]).

arg_to_go_var(VarMap, Arg, GoVar) :-
    var(Arg), !,
    (   lookup_var_identity(Arg, VarMap, Name) ->
        GoVar = Name
    ;   GoVar = '"<unknown>"'
    ).
arg_to_go_var(_, Arg, GoVar) :-
    atom(Arg), !,
    format(atom(GoVar), '"~w"', [Arg]).
arg_to_go_var(_, Arg, Arg) :- number(Arg), !.

%% lookup_var_identity(+Key, +Map, -Val)
%  Lookup value in association list using identity check (==)
lookup_var_identity(Key, [(K, V)|_], Val) :- Key == K, !, Val = V.
lookup_var_identity(Key, [_|Rest], Val) :- lookup_var_identity(Key, Rest, Val).



%% compile_json_to_go_typed(+HeadArgs, +FieldMappings, +SchemaName, +FieldDelim, +Unique, -GoCode)
%  Generate Go code for JSON input mode with type safety from schema
%
compile_json_to_go_typed(HeadArgs, FieldMappings, SchemaName, FieldDelim, Unique, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate typed field extraction code
    generate_typed_field_extractions(FieldMappings, SchemaName, HeadArgs, ExtractCode),

    % Generate output expression (same as untyped)
    generate_json_output_expr(HeadArgs, DelimChar, OutputExpr),

    % Build main loop
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n\t',
        UniqueCheck = '\t\tif !seen[result] {\n\t\t\tseen[result] = true\n\t\t\tfmt.Println(result)\n\t\t}\n'
    ;   SeenDecl = '\t',
        UniqueCheck = '\t\tfmt.Println(result)\n'
    ),

    % Build the loop code
    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
~w
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
\t\t
~s
\t\t
\t\tresult := ~s
~s\t}
', [SeenDecl, ExtractCode, OutputExpr, UniqueCheck]).

%% generate_typed_field_extractions(+FieldMappings, +SchemaName, +HeadArgs, -ExtractCode)
%  Generate typed field extraction code based on schema
%
generate_typed_field_extractions(FieldMappings, SchemaName, _HeadArgs, ExtractCode) :-
    findall(ExtractLine,
        (   nth1(Pos, FieldMappings, Mapping),
            format(atom(VarName), 'field~w', [Pos]),
            % Dispatch based on mapping type
            (   Mapping = Field-_Var
            ->  % Flat field - get type and options from schema
                get_field_info(SchemaName, Field, Type, Options),
                atom_string(Field, FieldStr),
                generate_typed_flat_field_extraction(FieldStr, VarName, Type, Options, ExtractLine)
            ;   Mapping = nested(Path, _Var)
            ->  % Nested field - get type and options from last element of path
                last(Path, LastField),
                get_field_info(SchemaName, LastField, Type, Options),
                generate_typed_nested_field_extraction(Path, VarName, Type, Options, ExtractLine)
            )
        ),
        ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode).

%% generate_typed_flat_field_extraction(+FieldName, +VarName, +Type, +Options, -ExtractCode)
%  Generate type-safe extraction code for a flat field with validation
%
generate_typed_flat_field_extraction(FieldName, VarName, Type, Options, ExtractCode) :-
    % Generate validation code
    generate_validation_code(VarName, Type, Options, ValidationCode),

    % Determine if optional
    (   member(optional, Options) -> Optional = true ; Optional = false ),

    % Generate extraction based on type
    (   Type = string ->
        format(atom(ExtractCode), '\t\tvar ~w string\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wIsString := ~wRaw.(string)\n\t\tif !~wIsString {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a string\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   Type = integer ->
        format(atom(ExtractCode), '\t\tvar ~w int\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wFloat, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = int(~wFloat)\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   Type = float ->
        format(atom(ExtractCode), '\t\tvar ~w float64\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   Type = boolean ->
        format(atom(ExtractCode), '\t\tvar ~w bool\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wBoolOk := ~wRaw.(bool)\n\t\tif !~wBoolOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a boolean\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   % Fallback to untyped for 'any' type
        generate_flat_field_extraction(FieldName, VarName, ExtractCode)
    ).

%% generate_typed_nested_field_extraction(+Path, +VarName, +Type, +Options, -ExtractCode)
%  Generate type-safe extraction code for a nested field with validation
%
generate_typed_nested_field_extraction(Path, VarName, Type, Options, ExtractCode) :-
    % Convert path to Go string slice
    maplist(atom_string, Path, PathStrs),
    atomic_list_concat(PathStrs, '", "', PathStr),
    last(Path, LastField),
    atom_string(LastField, LastFieldStr),

    % Generate validation code
    generate_validation_code(VarName, Type, Options, ValidationCode),

    % Determine if optional
    (   member(optional, Options) -> Optional = true ; Optional = false ),

    % Generate extraction with type assertion based on type
    (   Type = string ->
        format(atom(ExtractCode), '\t\tvar ~w string\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wIsString := ~wRaw.(string)\n\t\tif !~wIsString {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a string\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   Type = integer ->
        format(atom(ExtractCode), '\t\tvar ~w int\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wFloat, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = int(~wFloat)\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   Type = float ->
        format(atom(ExtractCode), '\t\tvar ~w float64\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   Type = boolean ->
        format(atom(ExtractCode), '\t\tvar ~w bool\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wBoolOk := ~wRaw.(bool)\n\t\tif !~wBoolOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a boolean\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   % Fallback to untyped
        generate_nested_field_extraction(Path, VarName, ExtractCode)
    ).

%% generate_validation_code(+VarName, +Type, +Options, -Code)
%  Generate Go validation logic for a variable
generate_validation_code(VarName, Type, Options, Code) :-
    findall(Check,
        (   member(Option, Options),
            generate_check(VarName, Type, Option, Check)
        ),
        Checks),
    atomic_list_concat(Checks, '\n', Code).

generate_check(VarName, integer, min(Min), Check) :-
    format(atom(Check), '\t\tif ~w < ~w { continue }', [VarName, Min]).
generate_check(VarName, integer, max(Max), Check) :-
    format(atom(Check), '\t\tif ~w > ~w { continue }', [VarName, Max]).
generate_check(VarName, float, min(Min), Check) :-
    format(atom(Check), '\t\tif ~w < ~w { continue }', [VarName, Min]).
generate_check(VarName, float, max(Max), Check) :-
    format(atom(Check), '\t\tif ~w > ~w { continue }', [VarName, Max]).
generate_check(VarName, string, format(email), Check) :-
    format(atom(Check), '\t\tif !strings.Contains(~w, "@") { continue }', [VarName]).
generate_check(_, _, _, '').

%% generate_nested_helper(-HelperCode)
%  Generate the getNestedField helper function for traversing nested JSON
%
generate_nested_helper(HelperCode) :-
    HelperCode = 'func getNestedField(data map[string]interface{}, path []string) (interface{}, bool) {
\tcurrent := interface{}(data)
\t
\tfor _, key := range path {
\t\tcurrentMap, ok := current.(map[string]interface{})
\t\tif !ok {
\t\t\treturn nil, false
\t\t}
\t\t
\t\tvalue, exists := currentMap[key]
\t\tif !exists {
\t\t\treturn nil, false
\t\t}
\t\t
\t\tcurrent = value
\t}
\t
\treturn current, true
}'.

%% generate_json_field_extractions(+FieldMappings, +HeadArgs, -ExtractCode)
%  Generate Go code to extract and type-assert JSON fields (flat and nested)
%
generate_json_field_extractions(FieldMappings, HeadArgs, ExtractCode) :-
    % Generate extractions by pairing field mappings with positions
    findall(ExtractLine,
        (   nth1(Pos, FieldMappings, Mapping),
            format(atom(VarName), 'field~w', [Pos]),
            generate_field_extraction_dispatch(Mapping, VarName, ExtractLine)
        ),
        ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode).

%% generate_field_extraction_dispatch(+Mapping, +VarName, -ExtractCode)
%  Dispatch to appropriate extraction based on mapping type
%
generate_field_extraction_dispatch(Field-_Var, VarName, ExtractCode) :- !,
    % Flat field extraction
    atom_string(Field, FieldStr),
    generate_flat_field_extraction(FieldStr, VarName, ExtractCode).
generate_field_extraction_dispatch(nested(Path, _Var), VarName, ExtractCode) :- !,
    % Nested field extraction
    generate_nested_field_extraction(Path, VarName, ExtractCode).

%% generate_flat_field_extraction(+FieldName, +VarName, -ExtractCode)
%  Generate extraction code for a flat field
%  Extract as interface{} to support any JSON type (string, number, bool, etc.)
%
generate_flat_field_extraction(FieldName, VarName, ExtractCode) :-
    format(atom(ExtractCode), '\t\t~w, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}',
        [VarName, VarName, FieldName, VarName]).

%% generate_nested_field_extraction(+Path, +VarName, -ExtractCode)
%  Generate extraction code for a nested field using getNestedField helper
%
generate_nested_field_extraction(Path, VarName, ExtractCode) :-
    % Ensure path is a list
    (   is_list(Path)
    ->  PathList = Path
    ;   PathList = [Path]
    ),
    % Convert path atoms to Go string slice
    maplist(atom_string, PathList, PathStrs),
    atomic_list_concat(PathStrs, '", "', PathStr),
    format(atom(ExtractCode), '\t\t~w, ~wOk := getNestedField(data, []string{"~s"})\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}',
        [VarName, VarName, PathStr, VarName]).

%% generate_json_output_expr(+HeadArgs, +DelimChar, -OutputExpr)
%  Generate Go fmt.Sprintf expression for output
%  Use %v to handle any JSON type (string, number, bool, etc.)
%
generate_json_output_expr(HeadArgs, DelimChar, OutputExpr) :-
    length(HeadArgs, NumArgs),
    findall('%v', between(1, NumArgs, _), FormatParts),  % %v instead of %s
    atomic_list_concat(FormatParts, DelimChar, FormatStr),

    findall(VarName,
        (   nth1(Pos, HeadArgs, _),
            format(atom(VarName), 'field~w', [Pos])
        ),
        VarNames),
    atomic_list_concat(VarNames, ', ', VarList),

    format(atom(OutputExpr), 'fmt.Sprintf("~s", ~s)', [FormatStr, VarList]).

%% ============================================
%% UTILITY FUNCTIONS
%% ============================================

%% ============================================
%% XML INPUT MODE COMPILATION
%% ============================================

%% compile_xml_input_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate for XML input (streaming + flattening)
compile_xml_input_mode(Pred, Arity, Options, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),
    
    % Get predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    (   Clauses = [SingleHead-SingleBody] ->
        % Extract mappings (same as JSON)
        extract_json_field_mappings(SingleBody, FieldMappings),
        format('DEBUG: FieldMappings = ~w~n', [FieldMappings]),
        
        % Generate XML loop
        SingleHead =.. [_|HeadArgs],
        compile_xml_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, Options, CoreBody),
        
        % Check if database backend is specified
        (   option(db_backend(bbolt), Options)
        ->  format('  Database: bbolt~n'),
            wrap_with_database(CoreBody, FieldMappings, Pred, Options, ScriptBody),
            NeedsDatabase = true
        ;   ScriptBody = CoreBody,
            NeedsDatabase = false
        )
    ;   format('ERROR: XML mode supports single clause only~n'),
        fail
    ),
    
    % Generate XML helpers
    generate_xml_helpers(XmlHelpers),
    
    % Check if nested helper is needed
    (   member(nested(_, _), FieldMappings)
    ->  generate_nested_helper(NestedHelper),
        format(string(Helpers), "~s\n~s", [XmlHelpers, NestedHelper])
    ;   Helpers = XmlHelpers
    ),
    
    % Wrap in package
    (   IncludePackage ->
        (   NeedsDatabase = true
        ->  Imports = '\t"encoding/xml"\n\t"fmt"\n\t"os"\n\t"strings"\n\t"io"\n\n\tbolt "go.etcd.io/bbolt"'
        ;   Imports = '\t"encoding/xml"\n\t"fmt"\n\t"os"\n\t"strings"\n\t"io"'
        ),
        
        format(string(GoCode), 'package main

import (
~s
)

~s

func main() {
~s}
', [Imports, Helpers, ScriptBody])
    ;   GoCode = ScriptBody
    ).

%% compile_xml_to_go(+HeadArgs, +FieldMappings, +FieldDelim, +Unique, +Options, -GoCode)
compile_xml_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, Options, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),
    
    % Generate field extraction code (resusing JSON logic as data is map[string]interface{})
    generate_json_field_extractions(FieldMappings, HeadArgs, ExtractCode),
    generate_json_output_expr(HeadArgs, DelimChar, OutputExpr),
    
    % Get XML file and tags
    option(xml_file(XmlFile), Options, stdin),
    
    (   XmlFile == stdin
    ->  FileOpenCode = '\tf := os.Stdin'
    ;   format(string(FileOpenCode), '
\tf, err := os.Open("~w")
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening file: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer f.Close()', [XmlFile])
    ),

    (   option(tags(Tags), Options)
    ->  true
    ;   option(tag(Tag), Options)
    ->  Tags = [Tag]
    ;   Tags = []
    ),
    
    % Build tag check
    (   Tags = []
    ->  TagCheck = 'true'
    ;   maplist(tag_to_go_cond, Tags, Conds),
        atomic_list_concat(Conds, " || ", TagCheck)
    ),
    
    % Build main loop
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n\t',
        UniqueCheck = '\t\tif !seen[result] {\n\t\t\tseen[result] = true\n\t\t\tfmt.Println(result)\n\t\t}\n'
    ;   SeenDecl = '\t',
        UniqueCheck = '\t\tfmt.Println(result)\n'
    ),
    
    format(string(GoCode), '
~s

\tdecoder := xml.NewDecoder(f)
~w
\tfor {
\t\tt, err := decoder.Token()
\t\tif err == io.EOF {
\t\t\tbreak
\t\t}
\t\tif err != nil {
\t\t\tcontinue
\t\t}
\t\t
\t\tswitch se := t.(type) {
\t\tcase xml.StartElement:
\t\t\tif ~w {
\t\t\t\tvar node XmlNode
\t\t\t\tif err := decoder.DecodeElement(&node, &se); err != nil {
\t\t\t\t\tcontinue
\t\t\t\t}
\t\t\t\t
\t\t\t\tdata := FlattenXML(node)
\t\t\t\t
~s
\t\t\t\t
\t\t\t\tresult := ~s
~s
\t\t\t}
\t\t}
\t}
', [FileOpenCode, SeenDecl, TagCheck, ExtractCode, OutputExpr, UniqueCheck]).

tag_to_go_cond(Tag, Cond) :-
    format(string(Cond), 'se.Name.Local == "~w"', [Tag]).

generate_xml_helpers(Code) :-
    Code = '
type XmlNode struct {
	XMLName xml.Name
	Attrs   []xml.Attr `xml:",any,attr"`
	Content string     `xml:",chardata"`
	Nodes   []XmlNode  `xml:",any"`
}

func FlattenXML(n XmlNode) map[string]interface{} {
	m := make(map[string]interface{})
	for _, a := range n.Attrs {
		m["@"+a.Name.Local] = a.Value
	}
	trim := strings.TrimSpace(n.Content)
	if trim != "" {
		m["text"] = trim
	}
	for _, child := range n.Nodes {
        tag := child.XMLName.Local
        flatChild := FlattenXML(child)
        
        if existing, ok := m[tag]; ok {
            if list, isList := existing.([]interface{}); isList {
                m[tag] = append(list, flatChild)
            } else {
                m[tag] = []interface{}{existing, flatChild}
            }
        } else {
		    m[tag] = flatChild
        }
	}
	return m
}
'.

%% map_field_delimiter(+Delimiter, -String)
%  Map delimiter atom to string
map_field_delimiter(colon, ':') :- !.
map_field_delimiter(tab, '\t') :- !.
map_field_delimiter(comma, ',') :- !.
map_field_delimiter(pipe, '|') :- !.
map_field_delimiter(Char, Char) :- atom(Char), atom_length(Char, 1), !.

%% write_go_program(+GoCode, +FilePath)
%  Write Go code to file
%
write_go_program(GoCode, FilePath) :-
    file_directory_name(FilePath, Dir),
    (   Dir \= '.' -> make_directory_path(Dir) ; true ),
    open(FilePath, write, Stream),
    write(Stream, GoCode),
    close(Stream),
    format('Go program written to: ~w~n', [FilePath]).

%% schema_needs_strings(+SchemaName)
%  Check if schema requires the strings package
schema_needs_strings(SchemaName) :-
    get_json_schema(SchemaName, Fields),
    member(field(_, _, Options), Fields),
    member(format(_), Options), !.

format_import(Import, Line) :-
    format(atom(Line), '\t"~w"', [Import]).
format_import(Import, Line) :-
    sub_string(Import, _, _, _, '"'), % Already quoted (e.g. bolt "...")
    format(atom(Line), '\t~w', [Import]).
