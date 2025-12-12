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
    test_bash_pipeline_generator/0    % Unit tests for pipeline generator mode
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
