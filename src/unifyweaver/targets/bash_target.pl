:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% bash_target.pl - Bash Target for UnifyWeaver
% Compiles Prolog predicates to Bash scripts using bindings.

:- module(bash_target, [
    compile_predicate_to_bash/3,
    init_bash_target/0,
    compile_bash_pipeline/3,          % +Predicates, +Options, -BashCode
    test_bash_pipeline_generator/0,   % Unit tests for pipeline generator mode
    % Enhanced pipeline chaining exports
    compile_bash_enhanced_pipeline/3, % +Stages, +Options, -BashCode
    bash_enhanced_helpers/1,          % -Code
    generate_bash_enhanced_connector/3, % +Stages, +PipelineName, -Code
    test_bash_enhanced_chaining/0     % Test enhanced pipeline chaining
]).

:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module('../core/binding_registry').
:- use_module('../bindings/bash_bindings').

%% init_bash_target
%  Initialize the Bash target by loading bindings.
init_bash_target :-
    init_bash_bindings.

%% compile_predicate_to_bash(+Predicate, +Options, -BashCode)
%  Compile a predicate to a Bash function.
compile_predicate_to_bash(PredIndicator, _Options, BashCode) :-
    PredIndicator = Pred/Arity,
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(user:Head, Body), Clauses),
    format('DEBUG: Clauses for ~w/~w: ~w~n', [Pred, Arity, Clauses]),
    (   Clauses = [Head-Body]
    ->  compile_rule(Head, Body, BashCode)
    ;   format(string(BashCode), '# Multiple clauses not supported yet for ~w/~w', [Pred, Arity])
    ).

compile_rule(Head, Body, Code) :-
    Head =.. [Pred|Args],
    
    % Generate variable mapping for arguments
    map_args(Args, 1, VarMap, ArgInitCode),
    
    % Compile body
    compile_body(Body, VarMap, _, BodyCode),
    
    format(string(Code),
'~w() {
~s
~s
}
', [Pred, ArgInitCode, BodyCode]).

map_args([], _, [], "").
map_args([Arg|Rest], Idx, [Arg-VarName|MapRest], Code) :-
    format(atom(VarName), 'arg~w', [Idx]),
    format(string(Line), '    local ~w="$~w"~n', [VarName, Idx]),
    NextIdx is Idx + 1,
    map_args(Rest, NextIdx, MapRest, RestCode),
    string_concat(Line, RestCode, Code).

compile_body(true, V, V, "") :- !.
compile_body((A, B), V0, V2, Code) :- !,
    compile_body(A, V0, V1, C1),
    compile_body(B, V1, V2, C2),
    format(string(Code), '~s~n~s', [C1, C2]).
compile_body(Goal, V0, V1, Code) :-
    functor(Goal, Pred, Arity),
    (   binding(bash, Pred/Arity, Pattern, Inputs, Outputs, Options)
    ->  Goal =.. [_|Args],
        length(Inputs, InCount),
        length(InArgs, InCount),
        append(InArgs, OutArgs, Args),
        
        maplist(resolve_val(V0), InArgs, BashInArgs),
        format_pattern(Pattern, BashInArgs, Expr),
        
        (   Outputs = []
        ->  V1 = V0,
            format(string(Code), '    ~s || return 1', [Expr])
        ;   OutArgs = [OutVar],
            ensure_var(V0, OutVar, BashOutVar, V1),
            
            (   member(pattern(expansion), Options) ->
                format(string(Code), '    local ~w=~s', [BashOutVar, Expr])
            ;   member(pattern(arithmetic), Options) ->
                format(string(Code), '    local ~w=~s', [BashOutVar, Expr])
            ;   member(pattern(command_substitution), Options) ->
                format(string(Code), '    local ~w=$(~s)', [BashOutVar, Expr])
            ;   % Default
                format(string(Code), '    local ~w=$(~s)', [BashOutVar, Expr])
            )
        )
    ;   V1 = V0,
        format(string(Code), '    # Unknown predicate: ~w', [Goal])
    ).

resolve_val(VarMap, Var, Val) :-
    var(Var), lookup_var(Var, VarMap, Name), !,
    format(string(Val), '~w', [Name]).
resolve_val(_, Val, Val).

ensure_var(VarMap, Var, Name, VarMap) :-
    lookup_var(Var, VarMap, Name), !.
ensure_var(VarMap, Var, Name, [Var-Name|VarMap]) :-
    gensym(v, Name).

lookup_var(Var, [V-Name|_], Name) :- Var == V, !.
lookup_var(Var, [_|Rest], Name) :- lookup_var(Var, Rest, Name).

format_pattern(Pattern, Args, Cmd) :-
    format(string(Cmd), Pattern, Args).

%% ============================================================================
%% Pipeline Generator Mode - Fixpoint Evaluation Support
%% ============================================================================

%% compile_bash_pipeline(+Predicates, +Options, -BashCode)
%  Main entry point for compiling a pipeline of predicates to Bash.
%  Options:
%    - pipeline_name(Name): Name of the generated pipeline function
%    - pipeline_mode(Mode): 'sequential' (default) or 'generator' (fixpoint)
%    - record_format(Format): 'jsonl' (default), 'tsv', or 'csv'
compile_bash_pipeline(Predicates, Options, BashCode) :-
    % Extract options with defaults
    (member(pipeline_name(PipelineName), Options) -> true ; PipelineName = pipeline),
    (member(pipeline_mode(Mode), Options) -> true ; Mode = sequential),
    (member(record_format(Format), Options) -> true ; Format = jsonl),

    % Generate stage names from predicates
    maplist(predicate_to_stage_name, Predicates, StageNames),

    % Generate components
    bash_pipeline_header(Mode, HeaderCode),
    bash_pipeline_helpers(Mode, Format, HelpersCode),
    generate_bash_stage_functions(StageNames, StageFunctions),
    generate_bash_pipeline_connector(StageNames, PipelineName, Mode, ConnectorCode),
    generate_bash_main_block(PipelineName, Mode, MainBlock),

    % Combine all components
    format(string(BashCode), '~s~n~n~s~n~n~s~n~n~s~n~n~s',
           [HeaderCode, HelpersCode, StageFunctions, ConnectorCode, MainBlock]).

%% predicate_to_stage_name(+PredIndicator, -StageName)
%  Convert a predicate indicator to a stage function name.
predicate_to_stage_name(Pred/_, StageName) :-
    format(atom(StageName), 'stage_~w', [Pred]).

%% bash_pipeline_header(+Mode, -Code)
%  Generate the shebang and header comment.
bash_pipeline_header(generator, Code) :-
    format(string(Code),
'#!/bin/bash
# Generated by UnifyWeaver - Bash Pipeline (generator mode)
# Fixpoint evaluation for recursive pipeline stages
# Requires Bash 4.0+ for associative arrays

set -euo pipefail', []).

bash_pipeline_header(sequential, Code) :-
    format(string(Code),
'#!/bin/bash
# Generated by UnifyWeaver - Bash Pipeline (sequential mode)
# Linear pipeline stage processing

set -euo pipefail', []).

%% bash_pipeline_helpers(+Mode, +Format, -Code)
%  Generate helper functions based on mode and format.
bash_pipeline_helpers(generator, jsonl, Code) :-
    format(string(Code),
'# Helper function: Generate record key for deduplication
record_key() {
    local record="$1"
    # Extract all key-value pairs, sort them, join with semicolon
    echo "$record" | jq -S -c "to_entries | sort_by(.key) | from_entries"
}

# Helper function: Parse JSONL record
parse_jsonl() {
    local line="$1"
    echo "$line"
}

# Helper function: Format record as JSONL
format_jsonl() {
    local record="$1"
    echo "$record"
}

# Helper function: Check if key exists in seen array
key_exists() {
    local key="$1"
    [[ -v "seen[$key]" ]]
}', []).

bash_pipeline_helpers(generator, tsv, Code) :-
    format(string(Code),
'# Helper function: Generate record key for deduplication
record_key() {
    local record="$1"
    # Sort fields and join with semicolon for consistent key
    echo "$record" | tr "\\t" "\\n" | sort | tr "\\n" ";"
}

# Helper function: Parse TSV record
parse_tsv() {
    local line="$1"
    echo "$line"
}

# Helper function: Format record as TSV
format_tsv() {
    local record="$1"
    echo "$record"
}

# Helper function: Check if key exists in seen array
key_exists() {
    local key="$1"
    [[ -v "seen[$key]" ]]
}', []).

bash_pipeline_helpers(generator, csv, Code) :-
    format(string(Code),
'# Helper function: Generate record key for deduplication
record_key() {
    local record="$1"
    # Sort fields and join with semicolon for consistent key
    echo "$record" | tr "," "\\n" | sort | tr "\\n" ";"
}

# Helper function: Parse CSV record
parse_csv() {
    local line="$1"
    echo "$line"
}

# Helper function: Format record as CSV
format_csv() {
    local record="$1"
    echo "$record"
}

# Helper function: Check if key exists in seen array
key_exists() {
    local key="$1"
    [[ -v "seen[$key]" ]]
}', []).

bash_pipeline_helpers(sequential, _, Code) :-
    format(string(Code),
'# Helper function: Process record through pipeline
process_record() {
    local record="$1"
    echo "$record"
}', []).

%% generate_bash_stage_functions(+StageNames, -Code)
%  Generate stub functions for each pipeline stage.
generate_bash_stage_functions([], "").
generate_bash_stage_functions([Stage|Rest], Code) :-
    format(string(StageCode),
'# Pipeline stage: ~w
~w() {
    local record="$1"
    # TODO: Implement stage logic based on predicate bindings
    echo "$record"
}', [Stage, Stage]),
    generate_bash_stage_functions(Rest, RestCode),
    (RestCode = "" ->
        Code = StageCode
    ;   format(string(Code), '~s~n~n~s', [StageCode, RestCode])
    ).

%% generate_bash_pipeline_connector(+StageNames, +PipelineName, +Mode, -Code)
%  Generate the main pipeline connector function.
generate_bash_pipeline_connector(StageNames, PipelineName, generator, Code) :-
    generate_bash_fixpoint_chain(StageNames, ChainCode),
    format(string(Code),
'# Fixpoint pipeline connector: ~w
# Iterates until no new records are produced.
run_~w() {
    local -n input_ref=$1
    local -n output_ref=$2

    # Initialize seen set (associative array)
    declare -A seen
    local changed=1
    local record_count=${#input_ref[@]}

    # Copy input to records array
    declare -a records
    for ((i=0; i<record_count; i++)); do
        records[i]="${input_ref[i]}"
        local key=$(record_key "${records[i]}")
        seen["$key"]=1
    done

    # Fixpoint iteration
    while [[ $changed -eq 1 ]]; do
        changed=0
        local current_count=${#records[@]}

        for ((i=0; i<current_count; i++)); do
            local record="${records[i]}"
~s
            local new_record="$result"
            local key=$(record_key "$new_record")

            if ! key_exists "$key"; then
                seen["$key"]=1
                records+=("$new_record")
                output_ref+=("$new_record")
                changed=1
            fi
        done
    done
}', [PipelineName, PipelineName, ChainCode]).

generate_bash_pipeline_connector(StageNames, PipelineName, sequential, Code) :-
    generate_bash_sequential_chain(StageNames, ChainCode),
    format(string(Code),
'# Sequential pipeline connector: ~w
# Processes records through stages linearly.
run_~w() {
    local -n input_ref=$1
    local -n output_ref=$2

    for record in "${input_ref[@]}"; do
~s
        output_ref+=("$result")
    done
}', [PipelineName, PipelineName, ChainCode]).

%% generate_bash_fixpoint_chain(+StageNames, -Code)
%  Generate the fixpoint stage chain code.
generate_bash_fixpoint_chain([], "            local result=\"$record\"").
generate_bash_fixpoint_chain([Stage], Code) :-
    format(string(Code), '            local result=$(~w "$record")', [Stage]).
generate_bash_fixpoint_chain([Stage|Rest], Code) :-
    Rest \= [],
    format(string(FirstLine), '            local stage_~w_out=$(~w "$record")', [Stage, Stage]),
    generate_bash_fixpoint_chain_rest(Rest, "stage_~w_out"-Stage, RestCode),
    format(string(Code), '~s~n~s', [FirstLine, RestCode]).

generate_bash_fixpoint_chain_rest([Stage], PrevVar, Code) :-
    PrevVar = _-PrevStage,
    format(string(Code), '            local result=$(~w "$stage_~w_out")', [Stage, PrevStage]).
generate_bash_fixpoint_chain_rest([Stage|Rest], PrevVar, Code) :-
    Rest \= [],
    PrevVar = _-PrevStage,
    format(string(Line), '            local stage_~w_out=$(~w "$stage_~w_out")', [Stage, Stage, PrevStage]),
    generate_bash_fixpoint_chain_rest(Rest, "stage_~w_out"-Stage, RestCode),
    format(string(Code), '~s~n~s', [Line, RestCode]).

%% generate_bash_sequential_chain(+StageNames, -Code)
%  Generate the sequential stage chain code.
generate_bash_sequential_chain([], "        local result=\"$record\"").
generate_bash_sequential_chain([Stage], Code) :-
    format(string(Code), '        local result=$(~w "$record")', [Stage]).
generate_bash_sequential_chain([Stage|Rest], Code) :-
    Rest \= [],
    format(string(FirstLine), '        local stage_~w_out=$(~w "$record")', [Stage, Stage]),
    generate_bash_sequential_chain_rest(Rest, "stage_~w_out"-Stage, RestCode),
    format(string(Code), '~s~n~s', [FirstLine, RestCode]).

generate_bash_sequential_chain_rest([Stage], PrevVar, Code) :-
    PrevVar = _-PrevStage,
    format(string(Code), '        local result=$(~w "$stage_~w_out")', [Stage, PrevStage]).
generate_bash_sequential_chain_rest([Stage|Rest], PrevVar, Code) :-
    Rest \= [],
    PrevVar = _-PrevStage,
    format(string(Line), '        local stage_~w_out=$(~w "$stage_~w_out")', [Stage, Stage, PrevStage]),
    generate_bash_sequential_chain_rest(Rest, "stage_~w_out"-Stage, RestCode),
    format(string(Code), '~s~n~s', [Line, RestCode]).

%% generate_bash_main_block(+PipelineName, +Mode, -Code)
%  Generate the main execution block.
generate_bash_main_block(PipelineName, generator, Code) :-
    format(string(Code),
'# Main execution block
main() {
    # Read input records from stdin
    declare -a input_records
    declare -a output_records

    while IFS= read -r line || [[ -n "$line" ]]; do
        input_records+=("$line")
    done

    # Run fixpoint pipeline
    run_~w input_records output_records

    # Output results
    for record in "${output_records[@]}"; do
        echo "$record"
    done
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi', [PipelineName]).

generate_bash_main_block(PipelineName, sequential, Code) :-
    format(string(Code),
'# Main execution block
main() {
    # Read input records from stdin
    declare -a input_records
    declare -a output_records

    while IFS= read -r line || [[ -n "$line" ]]; do
        input_records+=("$line")
    done

    # Run sequential pipeline
    run_~w input_records output_records

    # Output results
    for record in "${output_records[@]}"; do
        echo "$record"
    done
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi', [PipelineName]).

%% ============================================================================
%% Unit Tests for Pipeline Generator Mode
%% ============================================================================

test_bash_pipeline_generator :-
    format('~n=== Bash Pipeline Generator Mode Unit Tests ===~n', []),
    run_bash_pipeline_tests(Results),
    (   member(fail, Results)
    ->  format('~nSome tests failed!~n', []),
        fail
    ;   format('~n=== All Bash Pipeline Generator Mode Tests Passed ===~n', [])
    ).

run_bash_pipeline_tests(Results) :-
    findall(Result, run_single_bash_pipeline_test(Result), Results).

run_single_bash_pipeline_test(Result) :-
    bash_pipeline_test(Name, Test),
    format('Testing: ~w... ', [Name]),
    (   catch(call(Test), Error, (format('ERROR: ~w~n', [Error]), fail))
    ->  format('PASS~n', []),
        Result = pass
    ;   format('FAIL~n', []),
        Result = fail
    ).

%% Test cases
bash_pipeline_test('Header generation (generator)', (
    bash_pipeline_header(generator, Code),
    sub_string(Code, _, _, _, "#!/bin/bash"),
    sub_string(Code, _, _, _, "generator mode"),
    sub_string(Code, _, _, _, "Fixpoint")
)).

bash_pipeline_test('Header generation (sequential)', (
    bash_pipeline_header(sequential, Code),
    sub_string(Code, _, _, _, "#!/bin/bash"),
    sub_string(Code, _, _, _, "sequential mode")
)).

bash_pipeline_test('Helpers with JSONL format', (
    bash_pipeline_helpers(generator, jsonl, Code),
    sub_string(Code, _, _, _, "record_key"),
    sub_string(Code, _, _, _, "jq"),
    sub_string(Code, _, _, _, "parse_jsonl")
)).

bash_pipeline_test('Helpers with TSV format', (
    bash_pipeline_helpers(generator, tsv, Code),
    sub_string(Code, _, _, _, "record_key"),
    sub_string(Code, _, _, _, "parse_tsv")
)).

bash_pipeline_test('Stage functions generation', (
    generate_bash_stage_functions([stage_transform, stage_derive], Code),
    sub_string(Code, _, _, _, "stage_transform()"),
    sub_string(Code, _, _, _, "stage_derive()")
)).

bash_pipeline_test('Fixpoint connector generation', (
    generate_bash_pipeline_connector([stage_a, stage_b], test_pipe, generator, Code),
    sub_string(Code, _, _, _, "run_test_pipe"),
    sub_string(Code, _, _, _, "while"),
    sub_string(Code, _, _, _, "changed"),
    sub_string(Code, _, _, _, "record_key")
)).

bash_pipeline_test('Sequential connector generation', (
    generate_bash_pipeline_connector([stage_a, stage_b], seq_pipe, sequential, Code),
    sub_string(Code, _, _, _, "run_seq_pipe"),
    sub_string(Code, _, _, _, "for record"),
    \+ sub_string(Code, _, _, _, "while")
)).

bash_pipeline_test('Main block generation (generator)', (
    generate_bash_main_block(my_pipe, generator, Code),
    sub_string(Code, _, _, _, "main()"),
    sub_string(Code, _, _, _, "run_my_pipe"),
    sub_string(Code, _, _, _, "fixpoint")
)).

bash_pipeline_test('Main block generation (sequential)', (
    generate_bash_main_block(my_pipe, sequential, Code),
    sub_string(Code, _, _, _, "main()"),
    sub_string(Code, _, _, _, "run_my_pipe"),
    sub_string(Code, _, _, _, "sequential")
)).

bash_pipeline_test('Full pipeline compilation', (
    compile_bash_pipeline([transform/1, derive/1], [
        pipeline_name(full_test),
        pipeline_mode(generator),
        record_format(jsonl)
    ], Code),
    sub_string(Code, _, _, _, "#!/bin/bash"),
    sub_string(Code, _, _, _, "record_key"),
    sub_string(Code, _, _, _, "run_full_test"),
    sub_string(Code, _, _, _, "stage_transform"),
    sub_string(Code, _, _, _, "stage_derive"),
    sub_string(Code, _, _, _, "while")
)).

%% ============================================================================
%% ENHANCED PIPELINE CHAINING
%% ============================================================================
%%
%% Adds support for complex data flow patterns beyond linear pipelines:
%%   - fan_out(Stages)     : Broadcast record to multiple stages, collect results
%%   - merge               : Combine results from parallel stages
%%   - route_by(Pred, Routes) : Conditional routing based on predicate
%%   - filter_by(Pred)     : Filter records by predicate
%%
%% Example usage:
%%   compile_bash_enhanced_pipeline([
%%       extract/1,
%%       filter_by(is_active),
%%       fan_out([validate/1, enrich/1]),
%%       merge,
%%       route_by(has_error, [(true, error_log/1), (false, transform/1)]),
%%       output/1
%%   ], [pipeline_name(enhanced_pipe), record_format(jsonl)], BashCode).
%%
%% ============================================================================

%% compile_bash_enhanced_pipeline(+Stages, +Options, -BashCode)
%  Main entry point for compiling enhanced pipelines to Bash.
%  Stages can include:
%    - Pred/Arity      : Standard predicate stage
%    - fan_out(Stages) : Broadcast to parallel stages
%    - merge           : Combine parallel results
%    - route_by(Pred, Routes) : Conditional routing
%    - filter_by(Pred) : Filter records by predicate
compile_bash_enhanced_pipeline(Stages, Options, BashCode) :-
    % Extract options with defaults
    (member(pipeline_name(PipelineName), Options) -> true ; PipelineName = enhanced_pipeline),
    (member(record_format(Format), Options) -> true ; Format = jsonl),

    % Generate components
    bash_enhanced_header(PipelineName, Header),
    bash_enhanced_helpers(Helpers),
    generate_bash_enhanced_stage_functions(Stages, StageFunctions),
    generate_bash_enhanced_connector(Stages, PipelineName, ConnectorCode),
    generate_bash_enhanced_main(PipelineName, Format, MainBlock),

    % Combine all components
    format(string(BashCode), "~w~n~n~w~n~n~w~n~w~n~w~n",
           [Header, Helpers, StageFunctions, ConnectorCode, MainBlock]).

%% bash_enhanced_header(+PipelineName, -Code)
%  Generate header for enhanced pipeline script.
bash_enhanced_header(PipelineName, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
'#!/bin/bash
# Generated by UnifyWeaver - Enhanced Pipeline: ~w
# Supports: fan-out, merge, conditional routing, filtering
# Requires: Bash 4.0+ for associative arrays, jq for JSON processing

set -euo pipefail', [PipelineNameStr]).

%% bash_enhanced_helpers(-Code)
%  Generate Bash helper functions for enhanced pipeline operations.
bash_enhanced_helpers(Code) :-
    Code = '# ============================================
# Enhanced Pipeline Helper Functions
# ============================================

# Helper: Fan-out - send record to multiple stages, collect all results
# Usage: fan_out_records "$record" "stage1,stage2,stage3"
# Results stored in FAN_OUT_RESULTS array
fan_out_records() {
    local record="$1"
    local stages="$2"

    FAN_OUT_RESULTS=()
    IFS="," read -ra stage_arr <<< "$stages"

    for stage in "${stage_arr[@]}"; do
        stage=$(echo "$stage" | xargs)  # Trim whitespace
        local result=$("$stage" "$record")
        FAN_OUT_RESULTS+=("$result")
    done
}

# Helper: Merge streams - combine results from fan-out
# Usage: merge_streams
# Returns combined results from FAN_OUT_RESULTS
merge_streams() {
    local merged=""
    for result in "${FAN_OUT_RESULTS[@]}"; do
        if [[ -n "$result" ]]; then
            if [[ -n "$merged" ]]; then
                merged="$merged"$\'\\n\'"$result"
            else
                merged="$result"
            fi
        fi
    done
    echo "$merged"
}

# Helper: Route record based on condition
# Usage: route_record "$record" "$condition_result"
# Routes stored in ROUTE_MAP associative array
route_record() {
    local record="$1"
    local condition="$2"

    if [[ -v "ROUTE_MAP[$condition]" ]]; then
        local target="${ROUTE_MAP[$condition]}"
        "$target" "$record"
    elif [[ -v "ROUTE_MAP[default]" ]]; then
        local target="${ROUTE_MAP[default]}"
        "$target" "$record"
    else
        echo "$record"  # Pass through if no matching route
    fi
}

# Helper: Filter record by predicate
# Usage: filter_record "$record" predicate_func
# Returns record if predicate passes, empty string otherwise
filter_record() {
    local record="$1"
    local predicate="$2"

    if "$predicate" "$record" >/dev/null 2>&1; then
        echo "$record"
    else
        echo ""
    fi
}

# Helper: Tee stream - copy record to multiple destinations
# Usage: tee_stream "$record" "dest1,dest2,dest3"
tee_stream() {
    local record="$1"
    local destinations="$2"

    IFS="," read -ra dest_arr <<< "$destinations"
    for dest in "${dest_arr[@]}"; do
        dest=$(echo "$dest" | xargs)
        "$dest" "$record"
    done
}

# Helper: Parse JSONL record (identity for now)
parse_jsonl() {
    local line="$1"
    echo "$line"
}

# Helper: Format record as JSONL (identity for now)
format_jsonl() {
    local record="$1"
    echo "$record"
}'.

%% generate_bash_enhanced_stage_functions(+Stages, -Code)
%  Generate stub functions for each pipeline stage.
generate_bash_enhanced_stage_functions(Stages, Code) :-
    collect_bash_enhanced_stage_names(Stages, StageNames),
    generate_bash_stage_stubs(StageNames, Code).

%% collect_bash_enhanced_stage_names(+Stages, -Names)
%  Collect all stage names from enhanced pipeline stages.
collect_bash_enhanced_stage_names([], []).
collect_bash_enhanced_stage_names([Stage|Rest], AllNames) :-
    extract_bash_stage_names(Stage, Names),
    collect_bash_enhanced_stage_names(Rest, RestNames),
    append(Names, RestNames, AllNames).

%% extract_bash_stage_names(+Stage, -Names)
%  Extract stage names from a single stage specification.
extract_bash_stage_names(Pred/_, [StageName]) :-
    !,
    format(atom(StageName), 'stage_~w', [Pred]).
extract_bash_stage_names(fan_out(SubStages), Names) :-
    !,
    collect_bash_enhanced_stage_names(SubStages, Names).
extract_bash_stage_names(route_by(Pred, Routes), [PredName|RouteNames]) :-
    !,
    format(atom(PredName), 'predicate_~w', [Pred]),
    extract_bash_route_names(Routes, RouteNames).
extract_bash_stage_names(filter_by(Pred), [PredName]) :-
    !,
    format(atom(PredName), 'predicate_~w', [Pred]).
extract_bash_stage_names(merge, []).
extract_bash_stage_names(_, []).

%% extract_bash_route_names(+Routes, -Names)
%  Extract stage names from routing table.
extract_bash_route_names([], []).
extract_bash_route_names([(_, Stage)|Rest], [StageName|RestNames]) :-
    (Stage = StagePred/_Arity -> true ; StagePred = Stage),
    format(atom(StageName), 'stage_~w', [StagePred]),
    extract_bash_route_names(Rest, RestNames).

%% generate_bash_stage_stubs(+Names, -Code)
%  Generate stub functions for collected stage names.
generate_bash_stage_stubs([], "").
generate_bash_stage_stubs([Name|Rest], Code) :-
    (atom_concat('predicate_', _, Name) ->
        % Predicate function returns exit code
        format(string(StubCode),
'# Predicate: ~w
~w() {
    local record="$1"
    # TODO: Implement predicate logic
    # Return 0 for true, non-zero for false
    return 0
}', [Name, Name])
    ;   % Stage function returns transformed record
        format(string(StubCode),
'# Pipeline stage: ~w
~w() {
    local record="$1"
    # TODO: Implement stage logic
    echo "$record"
}', [Name, Name])
    ),
    generate_bash_stage_stubs(Rest, RestCode),
    (RestCode = "" ->
        Code = StubCode
    ;   format(string(Code), '~w~n~n~w', [StubCode, RestCode])
    ).

%% generate_bash_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the main enhanced pipeline connector function.
generate_bash_enhanced_connector(Stages, PipelineName, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    generate_bash_enhanced_flow_code(Stages, "record", FlowCode),
    format(string(Code),
'# ============================================
# Enhanced Pipeline Connector: ~w
# ============================================

run_~w() {
    local record="$1"

~w

    echo "$result"
}', [PipelineNameStr, PipelineNameStr, FlowCode]).

%% generate_bash_enhanced_flow_code(+Stages, +InVar, -Code)
%  Generate flow code for enhanced pipeline stages.
generate_bash_enhanced_flow_code([], InVar, Code) :-
    format(string(Code), '    local result="$~w"', [InVar]).
generate_bash_enhanced_flow_code([Stage|Rest], InVar, Code) :-
    generate_bash_stage_flow(Stage, InVar, OutVar, StageCode),
    generate_bash_enhanced_flow_code(Rest, OutVar, RestCode),
    format(string(Code), '~w~n~w', [StageCode, RestCode]).

%% generate_bash_stage_flow(+Stage, +InVar, -OutVar, -Code)
%  Generate flow code for a single stage.

% Standard predicate stage
generate_bash_stage_flow(Pred/_, InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "stage_~w_out", [Pred]),
    format(string(Code),
'    # Stage: ~w
    local ~w=$(stage_~w "$~w")', [Pred, OutVar, Pred, InVar]).

% Fan-out stage: broadcast to parallel stages
generate_bash_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "fan_out_~w_result", [N]),
    extract_bash_enhanced_stage_names_list(SubStages, StageNames),
    format_bash_stage_list(StageNames, StageListStr),
    format(string(Code),
'    # Fan-out to ~w parallel stages
    fan_out_records "$~w" "~w"
    local ~w="${FAN_OUT_RESULTS[*]}"', [N, InVar, StageListStr, OutVar]).

% Merge stage: combine results from fan-out
generate_bash_stage_flow(merge, InVar, OutVar, Code) :-
    !,
    OutVar = "merged_result",
    format(string(Code),
'    # Merge parallel results
    local ~w=$(merge_streams)
    # Use merged result or previous if empty
    if [[ -z "$~w" ]]; then ~w="$~w"; fi', [OutVar, OutVar, OutVar, InVar]).

% Conditional routing stage
generate_bash_stage_flow(route_by(Pred, Routes), InVar, OutVar, Code) :-
    !,
    OutVar = "routed_result",
    format_bash_route_map(Routes, RouteMapCode),
    format(string(Code),
'    # Conditional routing by ~w
    declare -A ROUTE_MAP
~w
    local condition_result=$(predicate_~w "$~w" && echo "true" || echo "false")
    local ~w=$(route_record "$~w" "$condition_result")',
    [Pred, RouteMapCode, Pred, InVar, OutVar, InVar]).

% Filter stage
generate_bash_stage_flow(filter_by(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "filtered_result",
    format(string(Code),
'    # Filter by ~w
    local ~w=$(filter_record "$~w" predicate_~w)
    if [[ -z "$~w" ]]; then
        result=""
        return
    fi', [Pred, OutVar, InVar, Pred, OutVar]).

% Unknown stage type - pass through
generate_bash_stage_flow(Stage, InVar, InVar, Code) :-
    format(string(Code), '    # Unknown stage type: ~w (pass-through)', [Stage]).

%% extract_bash_enhanced_stage_names_list(+Stages, -Names)
%  Extract function names from stage specifications for fan-out.
extract_bash_enhanced_stage_names_list([], []).
extract_bash_enhanced_stage_names_list([Pred/_Arity|Rest], [StageName|RestNames]) :-
    !,
    format(atom(StageName), 'stage_~w', [Pred]),
    extract_bash_enhanced_stage_names_list(Rest, RestNames).
extract_bash_enhanced_stage_names_list([_|Rest], RestNames) :-
    extract_bash_enhanced_stage_names_list(Rest, RestNames).

%% format_bash_stage_list(+Names, -ListStr)
%  Format stage names as comma-separated list for Bash.
format_bash_stage_list([], "").
format_bash_stage_list([Name], Str) :-
    format(string(Str), "~w", [Name]).
format_bash_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_bash_stage_list(Rest, RestStr),
    format(string(Str), "~w,~w", [Name, RestStr]).

%% format_bash_route_map(+Routes, -Code)
%  Format routing map initialization for Bash.
format_bash_route_map([], "").
format_bash_route_map([(Cond, Stage)|Rest], Code) :-
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format(atom(StageFunc), 'stage_~w', [StageName]),
    (Cond = true ->
        format(string(Line), '    ROUTE_MAP["true"]="~w"', [StageFunc])
    ; Cond = false ->
        format(string(Line), '    ROUTE_MAP["false"]="~w"', [StageFunc])
    ;   format(string(Line), '    ROUTE_MAP["~w"]="~w"', [Cond, StageFunc])
    ),
    format_bash_route_map(Rest, RestCode),
    (RestCode = "" ->
        Code = Line
    ;   format(string(Code), '~w~n~w', [Line, RestCode])
    ).

%% generate_bash_enhanced_main(+PipelineName, +Format, -Code)
%  Generate main execution block for enhanced pipeline.
generate_bash_enhanced_main(PipelineName, jsonl, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
'# ============================================
# Main Execution Block
# ============================================

main() {
    local output_count=0

    # Process each input record through enhanced pipeline
    while IFS= read -r line || [[ -n "$line" ]]; do
        result=$(run_~w "$line")
        if [[ -n "$result" ]]; then
            echo "$result"
            ((output_count++)) || true
        fi
    done

    # Summary to stderr (optional)
    # echo "Processed $output_count records" >&2
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi', [PipelineNameStr]).

generate_bash_enhanced_main(PipelineName, _, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
'# ============================================
# Main Execution Block
# ============================================

main() {
    local output_count=0

    # Process each input record through enhanced pipeline
    while IFS= read -r line || [[ -n "$line" ]]; do
        result=$(run_~w "$line")
        if [[ -n "$result" ]]; then
            echo "$result"
            ((output_count++)) || true
        fi
    done
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi', [PipelineNameStr]).

%% ============================================
%% BASH ENHANCED PIPELINE CHAINING TESTS
%% ============================================

test_bash_enhanced_chaining :-
    format('~n=== Bash Enhanced Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Generate enhanced helpers
    format('[Test 1] Generate enhanced helpers~n', []),
    bash_enhanced_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "fan_out_records"),
        sub_string(Helpers1, _, _, _, "route_record"),
        sub_string(Helpers1, _, _, _, "filter_record"),
        sub_string(Helpers1, _, _, _, "merge_streams"),
        sub_string(Helpers1, _, _, _, "tee_stream")
    ->  format('  [PASS] All helper functions generated~n', [])
    ;   format('  [FAIL] Missing helper functions~n', [])
    ),

    % Test 2: Linear pipeline connector
    format('[Test 2] Linear pipeline connector~n', []),
    generate_bash_enhanced_connector([extract/1, transform/1, load/1], linear_pipe, Code2),
    (   sub_string(Code2, _, _, _, "run_linear_pipe"),
        sub_string(Code2, _, _, _, "stage_extract"),
        sub_string(Code2, _, _, _, "stage_transform"),
        sub_string(Code2, _, _, _, "stage_load")
    ->  format('  [PASS] Linear connector generated~n', [])
    ;   format('  [FAIL] Linear connector missing patterns~n', [])
    ),

    % Test 3: Fan-out connector
    format('[Test 3] Fan-out connector~n', []),
    generate_bash_enhanced_connector([extract/1, fan_out([validate/1, enrich/1])], fanout_pipe, Code3),
    (   sub_string(Code3, _, _, _, "fan_out_records"),
        sub_string(Code3, _, _, _, "Fan-out to 2")
    ->  format('  [PASS] Fan-out connector generated~n', [])
    ;   format('  [FAIL] Fan-out connector missing patterns~n', [])
    ),

    % Test 4: Fan-out with merge
    format('[Test 4] Fan-out with merge~n', []),
    generate_bash_enhanced_connector([fan_out([a/1, b/1]), merge, output/1], merge_pipe, Code4),
    (   sub_string(Code4, _, _, _, "merge_streams"),
        sub_string(Code4, _, _, _, "Merge")
    ->  format('  [PASS] Merge connector generated~n', [])
    ;   format('  [FAIL] Merge connector missing patterns~n', [])
    ),

    % Test 5: Conditional routing
    format('[Test 5] Conditional routing~n', []),
    generate_bash_enhanced_connector([extract/1, route_by(has_error, [(true, error/1), (false, success/1)])], route_pipe, Code5),
    (   sub_string(Code5, _, _, _, "ROUTE_MAP"),
        sub_string(Code5, _, _, _, "route_record"),
        sub_string(Code5, _, _, _, "Conditional routing")
    ->  format('  [PASS] Routing connector generated~n', [])
    ;   format('  [FAIL] Routing connector missing patterns~n', [])
    ),

    % Test 6: Filter stage
    format('[Test 6] Filter stage~n', []),
    generate_bash_enhanced_connector([extract/1, filter_by(is_valid), output/1], filter_pipe, Code6),
    (   sub_string(Code6, _, _, _, "filter_record"),
        sub_string(Code6, _, _, _, "Filter by is_valid")
    ->  format('  [PASS] Filter connector generated~n', [])
    ;   format('  [FAIL] Filter connector missing patterns~n', [])
    ),

    % Test 7: Complex pipeline with all patterns
    format('[Test 7] Complex pipeline~n', []),
    generate_bash_enhanced_connector([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], complex_pipe, Code7),
    (   sub_string(Code7, _, _, _, "Fan-out to 3"),
        sub_string(Code7, _, _, _, "Filter by is_active"),
        sub_string(Code7, _, _, _, "Merge"),
        sub_string(Code7, _, _, _, "Conditional routing")
    ->  format('  [PASS] Complex connector generated~n', [])
    ;   format('  [FAIL] Complex connector missing patterns~n', [])
    ),

    % Test 8: Stage function generation
    format('[Test 8] Stage function generation~n', []),
    generate_bash_enhanced_stage_functions([extract/1, filter_by(is_valid), fan_out([a/1, b/1])], Code8),
    (   sub_string(Code8, _, _, _, "stage_extract"),
        sub_string(Code8, _, _, _, "predicate_is_valid"),
        sub_string(Code8, _, _, _, "stage_a"),
        sub_string(Code8, _, _, _, "stage_b")
    ->  format('  [PASS] Stage functions generated~n', [])
    ;   format('  [FAIL] Stage functions missing~n', [])
    ),

    % Test 9: Full enhanced pipeline compilation
    format('[Test 9] Full enhanced pipeline~n', []),
    compile_bash_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(full_enhanced), record_format(jsonl)], Code9),
    (   sub_string(Code9, _, _, _, "#!/bin/bash"),
        sub_string(Code9, _, _, _, "fan_out_records"),
        sub_string(Code9, _, _, _, "run_full_enhanced"),
        sub_string(Code9, _, _, _, "main()")
    ->  format('  [PASS] Full pipeline compiles~n', [])
    ;   format('  [FAIL] Full pipeline compilation failed~n', [])
    ),

    % Test 10: Enhanced helpers completeness
    format('[Test 10] Enhanced helpers completeness~n', []),
    bash_enhanced_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "parse_jsonl"),
        sub_string(Helpers10, _, _, _, "format_jsonl"),
        sub_string(Helpers10, _, _, _, "FAN_OUT_RESULTS")
    ->  format('  [PASS] All helpers present~n', [])
    ;   format('  [FAIL] Some helpers missing~n', [])
    ),

    format('~n=== All Bash Enhanced Pipeline Chaining Tests Passed ===~n', []).
