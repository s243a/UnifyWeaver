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
:- use_module(common_generator).

% Suppress singleton warnings in this experimental generator target.
:- style_check(-singleton).
:- discontiguous extract_match_constraints/2.
:- discontiguous term_to_go_expr/3.
:- use_module(library(filesex)).

%% Go generator config for common_generator.pl
go_generator_config(Config) :-
    Config = [
        access_fmt-"~w.Args[\"arg~w\"]",
        atom_fmt-"\"~w\"",
        null_val-"nil",
        ops-[
            + - "+", - - "-", * - "*", / - "/",
            > - ">", < - "<", >= - ">=", =< - "<=",
            =:= - "==", =\= - "!="
        ]
    ].

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

    % Check for generator mode (fixpoint evaluation) - MUST come first
    (   option(mode(generator), Options)
    ->  format('  Mode: Generator (fixpoint)~n'),
        compile_generator_mode_go(Pred, Arity, Options, GoCode)
    % Check if this is an aggregation predicate (aggregate/3 in body)
    ;   functor(Head, Pred, Arity),
        clause(Head, Body),
        is_aggregation_predicate(Body)
    ->  format('  Mode: Aggregation (New)~n'),
        compile_aggregation_mode(Pred, Arity, Options, GoCode)
    % Check if this is a GROUP BY predicate (group_by/4 in body)
    ;   functor(Head, Pred, Arity),
        clause(Head, Body),
        is_group_by_predicate(Body)
    ->  format('  Mode: GROUP BY Aggregation~n'),
        compile_group_by_mode(Pred, Arity, Options, GoCode)
    % Check if this is database read mode
    ;   option(db_backend(bbolt), Options),
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
    % Check if this is an aggregation operation (legacy option-based)
    ;   option(aggregation(AggOp), Options, none),
        AggOp \= none
    ->  % Compile as aggregation
        option(field_delimiter(FieldDelim), Options, colon),
        option(include_package(IncludePackage), Options, true),
        compile_aggregation_to_go(Pred, Arity, AggOp, FieldDelim, IncludePackage, GoCode)
    ;   % Continue with normal compilation
        compile_predicate_to_go_normal(Pred, Arity, Options, GoCode)
    ).

%% ============================================
%% GENERATOR MODE IMPLEMENTATION (Fixpoint)
%% ============================================

%% compile_generator_mode_go(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate using fixpoint evaluation (generator mode)
%  Produces standalone Go program with Solve() iteration loop
compile_generator_mode_go(Pred, Arity, Options, GoCode) :-
    go_generator_config(Config),
    
    % Gather all clauses for this predicate (use copy_term to normalize)
    functor(Head, Pred, Arity),
    findall(HC-BC, 
        (user:clause(Head, B), copy_term((Head, B), (HC, BC))), 
        TargetClauses0),
    remove_variant_duplicates(TargetClauses0, TargetClauses),
    
    % Also gather clauses for predicates referenced in rule bodies (dependency closure)
    % This includes extracting inner goals from aggregate_all
    findall(DepPred/DepArity,
        (   member(_-Body, TargetClauses),
            Body \= true,
            body_to_list_go(Body, Goals),
            member(Goal, Goals),
            extract_goal_predicate(Goal, DepPred, DepArity),
            DepPred/DepArity \= Pred/Arity  % Don't include self-references
        ),
        DepPredList0),
    sort(DepPredList0, DepPredList),
    
    % Gather facts from dependencies
    findall(DepHead,
        (   member(DP/DA, DepPredList),
            functor(DepHead, DP, DA),
            user:clause(DepHead, true)
        ),
        DepFacts),
    
    % Combine target clauses with dependency facts
    findall(FactHead, member(FactHead-true, TargetClauses), TargetFacts),
    append(TargetFacts, DepFacts, AllFacts0),
    remove_variant_duplicates_single(AllFacts0, AllFacts),
    
    % Compile facts
    compile_go_generator_facts(AllFacts, Config, FactsCode),
    
    % Compile rules (only for target predicate, exclude facts)
    findall(RuleClause, 
        (member(RuleClause, TargetClauses), RuleClause = (_-RB), RB \= true),
        TargetRuleClauses),
    compile_go_generator_rules(TargetRuleClauses, Config, RulesCode, RuleNames),
    
    % Compile execution (fixpoint loop)
    compile_go_generator_execution(Pred, RuleNames, Options, ExecutionCode),
    
    % Assemble complete program
    go_generator_header(Pred, Options, Header),
    format(string(GoCode), "~w\n~w\n~w\n~w\n", [Header, FactsCode, RulesCode, ExecutionCode]).

%% remove_variant_duplicates(+List, -Unique)
%  Remove duplicates where terms are variants (same structure, different vars)
remove_variant_duplicates([], []).
remove_variant_duplicates([H|T], Result) :-
    (   member_variant(H, T)
    ->  remove_variant_duplicates(T, Result)
    ;   Result = [H|Rest],
        remove_variant_duplicates(T, Rest)
    ).

member_variant(X, [H|_]) :- X =@= H, !.
member_variant(X, [_|T]) :- member_variant(X, T).

%% remove_variant_duplicates_single(+List, -Unique)
%  For single terms (not pairs)
remove_variant_duplicates_single([], []).
remove_variant_duplicates_single([H|T], Result) :-
    (   member_variant_single(H, T)
    ->  remove_variant_duplicates_single(T, Result)
    ;   Result = [H|Rest],
        remove_variant_duplicates_single(T, Rest)
    ).

member_variant_single(X, [H|_]) :- X =@= H, !.
member_variant_single(X, [_|T]) :- member_variant_single(X, T).

%% extract_goal_predicate(+Goal, -Pred, -Arity)
%  Extract predicate name and arity from a goal
%  Handles aggregate_all/3,4 by extracting from inner goal
extract_goal_predicate(aggregate_all(_, InnerGoal, _), Pred, Arity) :-
    !,
    InnerGoal =.. [Pred|Args],
    length(Args, Arity).
extract_goal_predicate(aggregate_all(_, InnerGoal, _, _), Pred, Arity) :-
    !,
    InnerGoal =.. [Pred|Args],
    length(Args, Arity).
extract_goal_predicate(aggregate(_, InnerGoal, _), Pred, Arity) :-
    !,
    InnerGoal =.. [Pred|Args],
    length(Args, Arity).
extract_goal_predicate(Goal, Pred, Arity) :-
    \+ is_builtin_goal_go(Goal),
    Goal =.. [Pred|Args],
    length(Args, Arity).

%% go_generator_header(+Pred, +Options, -Header)
%  Generate Go boilerplate with Fact type
go_generator_header(Pred, Options, Header) :-
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    % Add bufio and os imports if json_input is enabled
    (   option(json_input(true), Options)
    ->  JsonImports = "\"bufio\"\n    \"os\"\n    "
    ;   JsonImports = ""
    ),
    % Add sync import if workers (parallel) is enabled
    (   option(workers(W), Options), W > 1
    ->  SyncImport = "\"sync\"\n    "
    ;   SyncImport = ""
    ),
    % Add bbolt import if db_backend is specified
    (   option(db_backend(bbolt), Options)
    ->  DbImport = "\"log\"\n    \"os\"\n    bolt \"go.etcd.io/bbolt\"\n    "
    ;   DbImport = ""
    ),
    format(string(ExtraImports), "~w~w~w", [JsonImports, SyncImport, DbImport]),
    format(string(Header),
"// Generated by UnifyWeaver Go Generator Mode
// Date: ~w
// Predicate: ~w

package main

import (
    ~w\"encoding/json\"
    \"fmt\"
    \"strconv\"
)

// Fact represents a relation tuple
type Fact struct {
    Relation string                 \x60json:\"relation\"\x60
    Args     map[string]interface{} \x60json:\"args\"\x60
}

// Key returns a canonical string for set membership
func (f Fact) Key() string {
    b, _ := json.Marshal(f)
    return string(b)
}

// toFloat64 converts interface{} to float64 for aggregation
func toFloat64(v interface{}) (float64, bool) {
    switch val := v.(type) {
    case float64:
        return val, true
    case int:
        return float64(val), true
    case int64:
        return float64(val), true
    case string:
        f, err := strconv.ParseFloat(val, 64)
        return f, err == nil
    default:
        return 0, false
    }
}

// Index provides O(1) lookup by relation and argument value
type Index struct {
    // byArg[relation][argN][value] -> []*Fact
    byArg map[string]map[string]map[interface{}][]*Fact
}

// NewIndex creates an empty index
func NewIndex() *Index {
    return &Index{byArg: make(map[string]map[string]map[interface{}][]*Fact)}
}

// Add indexes a fact by all its arguments
func (idx *Index) Add(fact *Fact) {
    rel := fact.Relation
    if _, ok := idx.byArg[rel]; !ok {
        idx.byArg[rel] = make(map[string]map[interface{}][]*Fact)
    }
    for argName, argVal := range fact.Args {
        if _, ok := idx.byArg[rel][argName]; !ok {
            idx.byArg[rel][argName] = make(map[interface{}][]*Fact)
        }
        idx.byArg[rel][argName][argVal] = append(idx.byArg[rel][argName][argVal], fact)
    }
}

// Lookup returns facts matching relation, argument name, and value
func (idx *Index) Lookup(relation, argName string, value interface{}) []*Fact {
    if relIdx, ok := idx.byArg[relation]; ok {
        if argIdx, ok := relIdx[argName]; ok {
            return argIdx[value]
        }
    }
    return nil
}

// BuildIndex creates an index from all facts in the total map
func BuildIndex(total map[string]Fact) *Index {
    idx := NewIndex()
    for _, fact := range total {
        f := fact // Create copy for stable pointer
        idx.Add(&f)
    }
    return idx
}

", [DateStr, Pred, ExtraImports]).

%% compile_go_generator_facts(+FactHeads, +Config, -FactsCode)
%  Generate GetInitialFacts() function
compile_go_generator_facts(FactHeads, _Config, Code) :-
    findall(FactCode,
        (   member(Head, FactHeads),
            Head =.. [Pred|Args],
            generate_go_generator_fact(Pred, Args, FactCode)
        ),
        FactCodes),
    (   FactCodes == []
    ->  FactsBody = ""
    ;   atomic_list_concat(FactCodes, "\n\t\t", FactsBody)
    ),
    format(string(Code),
"// GetInitialFacts returns base facts for fixpoint computation
func GetInitialFacts() []Fact {
    return []Fact{
        ~w
    }
}
", [FactsBody]).

%% generate_go_generator_fact(+Pred, +Args, -FactCode)
generate_go_generator_fact(Pred, Args, Code) :-
    findall(ArgCode,
        (   nth0(I, Args, Arg),
            format(string(ArgCode), "\"arg~w\": \"~w\"", [I, Arg])
        ),
        ArgCodes),
    atomic_list_concat(ArgCodes, ", ", ArgsStr),
    format(string(Code), 
        "{Relation: \"~w\", Args: map[string]interface{}{~w}},", 
        [Pred, ArgsStr]).

%% compile_go_generator_rules(+RuleClauses, +Config, -RulesCode, -RuleNames)
%  Generate ApplyRule_N functions for each rule clause
compile_go_generator_rules(RuleClauses, Config, RulesCode, RuleNames) :-
    findall(RuleCode-RuleName,
        (   nth1(I, RuleClauses, Head-Body),
            Body \= true,
            once(compile_go_generator_rule(I, Head, Body, Config, RuleCode, RuleName))
        ),
        Pairs),
    pairs_keys_values(Pairs, RuleCodes, RuleNames),
    atomic_list_concat(RuleCodes, "\n\n", RulesCode).

%% compile_go_generator_rule(+Index, +Head, +Body, +Config, -Code, -RuleName)
%  Compile a single rule to a Go function
compile_go_generator_rule(Index, Head, Body, Config, Code, RuleName) :-
    format(string(RuleName), "ApplyRule_~w", [Index]),
    Head =.. [HeadPred|HeadArgs],
    
    % Parse body into goals
    body_to_list_go(Body, Goals),
    
    % Check for aggregate goal first
    (   member(AggGoal, Goals),
        is_aggregate_goal_go(AggGoal)
    ->  % Aggregate rule - extract HAVING filters (builtins after aggregate)
        extract_having_filters(Goals, AggGoal, HavingFilters),
        compile_go_aggregate_rule(Index, HeadPred, HeadArgs, AggGoal, HavingFilters, Config, Code)
    ;   % Normal rule (joins, negation, etc.)
        partition(is_builtin_goal_go, Goals, Builtins, RelGoals),
        
        (   RelGoals = []
        ->  % Only builtins - not a productive rule
            format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    // Rule ~w: constraint-only (no relational goals)
    return nil
}", [RuleName, Index]),
            !
        ;   RelGoals = [FirstGoal|RestGoals],
            FirstGoal =.. [Pred1|_],
            
            % Build initial variable map from first goal
            build_variable_map([FirstGoal-fact], VarMap0),
            
            (   RestGoals == []
            ->  % Simple rule: no join
                % Generate builtin checks
                compile_go_builtins(Builtins, VarMap0, Config, BuiltinCode),
                % Generate head construction
                compile_go_head_construction(HeadPred, HeadArgs, VarMap0, Config, HeadCode),
                format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    var results []Fact
    
    // Match first goal: ~w
    if fact.Relation != \"~w\" {
        return results
    }
~w
    // Emit result
    ~w
    
    return results
}", [RuleName, FirstGoal, Pred1, BuiltinCode, HeadCode])
            ;   % Join rule: need to nest the result inside join loops
                compile_go_join_with_result(RestGoals, VarMap0, Config, Builtins, HeadPred, HeadArgs, JoinWithResultCode),
                format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    var results []Fact
    
    // Match first goal: ~w
    if fact.Relation != \"~w\" {
        return results
    }
~w
    return results
}", [RuleName, FirstGoal, Pred1, JoinWithResultCode])
            )
        )
    ).

%% is_aggregate_goal_go(+Goal)
%  Check if goal is an aggregate
is_aggregate_goal_go(aggregate_all(_, _, _)).
is_aggregate_goal_go(aggregate_all(_, _, _, _)).
is_aggregate_goal_go(aggregate(_, _, _)).  % Will be normalized

%% normalize_aggregate_goal_go(+Goal, -NormalizedGoal)
%  Normalize aggregate/3 to aggregate_all/3 for consistency
normalize_aggregate_goal_go(aggregate(Op, Body, Res), aggregate_all(Op, Body, Res)) :- !.
normalize_aggregate_goal_go(G, G).

%% extract_having_filters(+Goals, +AggGoal, -HavingFilters)
%  Extract builtin goals that appear after the aggregate (HAVING clause)
extract_having_filters(Goals, AggGoal, HavingFilters) :-
    % Find position of aggregate goal
    append(Before, [AggGoal|After], Goals),
    !,
    % Builtins after aggregate are HAVING filters
    include(is_builtin_goal_go, After, HavingFilters).
extract_having_filters(_, _, []).

%% compile_go_aggregate_rule(+Index, +HeadPred, +HeadArgs, +AggGoal, +HavingFilters, +Config, -Code)
%  Compile an aggregate rule to Go with optional HAVING filters
compile_go_aggregate_rule(Index, HeadPred, HeadArgs, AggGoal0, HavingFilters, Config, Code) :-
    format(string(RuleName), "ApplyRule_~w", [Index]),
    
    % Normalize aggregate/3 -> aggregate_all/3
    normalize_aggregate_goal_go(AggGoal0, AggGoal),
    
    % Decompose aggregate goal
    (   AggGoal = aggregate_all(OpTerm, InnerGoal, GroupVar, Result)
    ->  % Grouped aggregation (aggregate_all/4)
        compile_go_grouped_aggregate(RuleName, HeadPred, HeadArgs, OpTerm, InnerGoal, GroupVar, Result, HavingFilters, Config, Code)
    ;   AggGoal = aggregate_all(OpTerm, InnerGoal, Result)
    ->  % Ungrouped aggregation (aggregate_all/3)
        compile_go_ungrouped_aggregate(RuleName, HeadPred, HeadArgs, OpTerm, InnerGoal, Result, HavingFilters, Config, Code)
    ;   format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    // Unsupported aggregate form
    return nil
}", [RuleName])
    ).

%% compile_go_ungrouped_aggregate(+RuleName, +HeadPred, +HeadArgs, +OpTerm, +InnerGoal, +Result, +HavingFilters, +Config, -Code)
compile_go_ungrouped_aggregate(RuleName, HeadPred, HeadArgs, OpTerm, InnerGoal, Result, HavingFilters, _Config, Code) :-
    InnerGoal =.. [Pred|Args],
    
    % Determine the aggregate operation and value variable
    decompose_agg_op(OpTerm, Op, ValueVar),
    
    % Find value variable index in inner goal args
    (   var(ValueVar)
    ->  find_var_index_go(ValueVar, Args, ValueIdx)
    ;   ValueIdx = -1  % count doesn't need a value index
    ),
    
    % Generate aggregate code
    go_agg_code(Op, AggCode),
    
    % Build result args
    length(HeadArgs, NumArgs),
    (   NumArgs == 1
    ->  ArgsStr = "\"arg0\": agg"
    ;   ArgsStr = "\"arg0\": agg"  % Default for now
    ),
    
    % Generate HAVING filter code if any
    (   HavingFilters \= []
    ->  generate_having_conditions(HavingFilters, Result, HavingCond),
        format(string(HavingIfStart), "
        // HAVING clause filter
        if ~w {", [HavingCond]),
        HavingIfEnd = "
        }"
    ;   HavingIfStart = "",
        HavingIfEnd = ""
    ),
    
    format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    var results []Fact
    
    // Skip if not triggered by this relation
    if fact.Relation != \"~w\" {
        return results
    }
    
    // Collect values for aggregation
    var values []float64
    for _, f := range total {
        if f.Relation == \"~w\" {
            val, ok := toFloat64(f.Args[\"arg~w\"])
            if ok {
                values = append(values, val)
            }
        }
    }
    
    // Compute aggregate
    if len(values) > 0 {
        ~w~w
            results = append(results, Fact{Relation: \"~w\", Args: map[string]interface{}{~w}})~w
    }
    
    return results
}", [RuleName, Pred, Pred, ValueIdx, AggCode, HavingIfStart, HeadPred, ArgsStr, HavingIfEnd]).

%% compile_go_grouped_aggregate(+RuleName, +HeadPred, +HeadArgs, +OpTerm, +InnerGoal, +GroupVar, +Result, +HavingFilters, +Config, -Code)
compile_go_grouped_aggregate(RuleName, HeadPred, _HeadArgs, OpTerm, InnerGoal, GroupVar, Result, HavingFilters, _Config, Code) :-
    InnerGoal =.. [Pred|Args],
    
    % Determine the aggregate operation and value variable
    decompose_agg_op(OpTerm, Op, ValueVar),
    
    % Find group key and value indices
    find_var_index_go(GroupVar, Args, GroupIdx),
    (   var(ValueVar)
    ->  find_var_index_go(ValueVar, Args, ValueIdx)
    ;   ValueIdx = -1
    ),
    
    % Generate aggregate code
    go_agg_code(Op, AggCode),
    
    % Generate HAVING filter code if any
    (   HavingFilters \= []
    ->  generate_having_conditions(HavingFilters, Result, HavingCond),
        format(string(HavingCode), "
            // HAVING clause filter
            if !(~w) {
                continue
            }", [HavingCond])
    ;   HavingCode = ""
    ),
    
    format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    var results []Fact
    
    // Skip if not triggered by this relation
    if fact.Relation != \"~w\" {
        return results
    }
    
    // Group by key
    groups := make(map[interface{}][]float64)
    for _, f := range total {
        if f.Relation == \"~w\" {
            key := f.Args[\"arg~w\"]
            val, ok := toFloat64(f.Args[\"arg~w\"])
            if ok {
                groups[key] = append(groups[key], val)
            }
        }
    }
    
    // Compute aggregate per group
    for key, values := range groups {
        if len(values) > 0 {
            ~w~w
            results = append(results, Fact{
                Relation: \"~w\",
                Args: map[string]interface{}{\"arg0\": key, \"arg1\": agg},
            })
        }
    }
    
    return results
}", [RuleName, Pred, Pred, GroupIdx, ValueIdx, AggCode, HavingCode, HeadPred]).

%% decompose_agg_op(+OpTerm, -Op, -ValueVar)
decompose_agg_op(count, count, _).
decompose_agg_op(sum(V), sum, V).
decompose_agg_op(min(V), min, V).
decompose_agg_op(max(V), max, V).
decompose_agg_op(avg(V), avg, V).
decompose_agg_op(set(V), set, V).
decompose_agg_op(bag(V), bag, V).

%% generate_having_conditions(+HavingFilters, +ResultVar, -GoCondition)
%  Translate HAVING filter builtins to Go conditions
%  ResultVar is the Prolog variable bound to the aggregate result
generate_having_conditions([], _, "true").
generate_having_conditions([Filter|Rest], ResultVar, Condition) :-
    generate_single_having_condition(Filter, ResultVar, Cond1),
    (   Rest == []
    ->  Condition = Cond1
    ;   generate_having_conditions(Rest, ResultVar, RestCond),
        format(string(Condition), "(~w) && (~w)", [Cond1, RestCond])
    ).

%% generate_single_having_condition(+Filter, +ResultVar, -GoCond)
generate_single_having_condition(Filter, ResultVar, GoCond) :-
    Filter =.. [Op, Left, Right],
    member(Op-GoOp, [(>)-">", (<)-"<", (>=)-">=", (=<)-"<=", (=:=)-"==", (=\=)-"!="]),
    !,
    translate_having_operand(Left, ResultVar, LeftGo),
    translate_having_operand(Right, ResultVar, RightGo),
    format(string(GoCond), "~w ~w ~w", [LeftGo, GoOp, RightGo]).
generate_single_having_condition(_, _, "true").  % Fallback

%% translate_having_operand(+Term, +ResultVar, -GoExpr)
translate_having_operand(Term, ResultVar, "agg") :-
    var(Term), Term == ResultVar, !.
translate_having_operand(N, _, Str) :-
    number(N), !,
    format(string(Str), "~w", [N]).
translate_having_operand(Term, _, Str) :-
    format(string(Str), "~w", [Term]).

%% go_agg_code(+Op, -GoCode)
go_agg_code(count, "agg := float64(len(values))").
go_agg_code(sum, "agg := 0.0; for _, v := range values { agg += v }").
go_agg_code(min, "agg := values[0]; for _, v := range values { if v < agg { agg = v } }").
go_agg_code(max, "agg := values[0]; for _, v := range values { if v > agg { agg = v } }").
go_agg_code(avg, "agg := 0.0; for _, v := range values { agg += v }; agg /= float64(len(values))").
go_agg_code(set, "agg := float64(len(values))").  % Simplified for now
go_agg_code(bag, "agg := float64(len(values))").  % Simplified for now

%% find_var_index_go(+Var, +Args, -Index)
find_var_index_go(Var, Args, Index) :-
    nth0(Index, Args, Arg),
    Var == Arg,
    !.
find_var_index_go(_, _, 0).  % Default to 0 if not found

%% body_to_list_go(+Body, -Goals)
%  Convert conjunction body to list of goals
body_to_list_go((A, B), Goals) :- !,
    body_to_list_go(A, GoalsA),
    body_to_list_go(B, GoalsB),
    append(GoalsA, GoalsB, Goals).
body_to_list_go(Goal, [Goal]).

%% is_builtin_goal_go(+Goal)
%  Check if goal is a builtin (comparison, arithmetic)
is_builtin_goal_go(Goal) :-
    Goal =.. [Functor|_],
    member(Functor, [is, '>', '<', '>=', '=<', '=:=', '=\\=', '==', '\\=', not, '\\+']).

%% compile_go_join_with_result(+Goals, +VarMap, +Config, +Builtins, +HeadPred, +HeadArgs, -Code)
%  Generate nested join loops with result emission inside innermost loop
compile_go_join_with_result([], VarMap, Config, Builtins, HeadPred, HeadArgs, Code) :-
    % Base case: generate builtin checks and result emit
    compile_go_builtins(Builtins, VarMap, Config, BuiltinCode),
    compile_go_head_construction(HeadPred, HeadArgs, VarMap, Config, HeadCode),
    format(string(Code), "~w\n            ~w", [BuiltinCode, HeadCode]).

compile_go_join_with_result([Goal|Rest], VarMap, Config, Builtins, HeadPred, HeadArgs, Code) :-
    Goal =.. [Pred|Args],
    length([Goal|Rest], Len),
    format(string(VarName), "j~w", [Len]),
    
    % Find first shared variable that can use index lookup
    (   find_indexable_join(Args, VarMap, ArgIdx, Source, SrcIdx)
    ->  % Use indexed lookup
        format(string(LookupKey), "~w.Args[\"arg~w\"]", [Source, SrcIdx]),
        format(string(ArgName), "arg~w", [ArgIdx]),
        
        % Build remaining join conditions (excluding the indexed one)
        compile_go_join_condition_excluding(Args, VarMap, Config, VarName, ArgIdx, JoinCond),
        
        % Update variable map with this goal's bindings
        build_variable_map([Goal-VarName], NewBindings),
        append(VarMap, NewBindings, VarMap1),
        
        % Recurse to get inner code
        compile_go_join_with_result(Rest, VarMap1, Config, Builtins, HeadPred, HeadArgs, InnerCode),
        
        format(string(Code),
"    
    // Join with ~w (indexed on ~w)
    for _, ~wPtr := range idx.Lookup(\"~w\", \"~w\", ~w) {
        ~w := *~wPtr
        if true~w {
~w
        }
    }
", [Pred, ArgName, VarName, Pred, ArgName, LookupKey, VarName, VarName, JoinCond, InnerCode])
    ;   % Fallback to linear scan (no indexable join condition)
        compile_go_join_condition(Args, VarMap, Config, VarName, JoinCond),
        
        % Update variable map with this goal's bindings
        build_variable_map([Goal-VarName], NewBindings),
        append(VarMap, NewBindings, VarMap1),
        
        % Recurse to get inner code
        compile_go_join_with_result(Rest, VarMap1, Config, Builtins, HeadPred, HeadArgs, InnerCode),
        
        format(string(Code),
"    
    // Join with ~w (linear scan)
    for _, ~w := range total {
        if ~w.Relation == \"~w\"~w {
~w
        }
    }
", [Pred, VarName, VarName, Pred, JoinCond, InnerCode])
    ).

%% find_indexable_join(+Args, +VarMap, -ArgIdx, -Source, -SrcIdx)
%  Find the first argument that is bound in VarMap (prefer arg0)
find_indexable_join(Args, VarMap, ArgIdx, Source, SrcIdx) :-
    nth0(ArgIdx, Args, Arg),
    var(Arg),
    member(Arg0-source(Source, SrcIdx), VarMap),
    Arg == Arg0,
    !.

%% compile_go_join_condition_excluding(+Args, +VarMap, +Config, +VarName, +ExcludeIdx, -Condition)
%  Generate join conditions excluding the indexed argument
compile_go_join_condition_excluding(Args, VarMap, _Config, VarName, ExcludeIdx, Condition) :-
    findall(Cond,
        (   nth0(I, Args, Arg),
            I \= ExcludeIdx,
            var(Arg),
            member(Arg0-source(Source, SrcIdx), VarMap),
            Arg == Arg0,
            format(string(Cond), " && ~w.Args[\"arg~w\"] == ~w.Args[\"arg~w\"]", [VarName, I, Source, SrcIdx])
        ),
        Conds),
    atomic_list_concat(Conds, "", Condition).

%% compile_go_joins(+Goals, +VarMap, +Config, +Index, -JoinCode, -FinalVarMap)
%  Generate nested loops for join goals
compile_go_joins([], VarMap, _, _, "", VarMap).
compile_go_joins([Goal|Rest], VarMap, Config, Index, Code, FinalVarMap) :-
    Goal =.. [Pred|Args],
    format(string(VarName), "j~w", [Index]),
    
    % Build join condition from shared variables
    compile_go_join_condition(Args, VarMap, Config, VarName, JoinCond),
    
    % Update variable map with this goal's bindings
    build_variable_map([Goal-VarName], NewBindings),
    append(VarMap, NewBindings, VarMap1),
    
    % Recurse for remaining goals
    Next is Index + 1,
    compile_go_joins(Rest, VarMap1, Config, Next, RestCode, FinalVarMap),
    
    format(string(Code),
"    
    // Join with ~w
    for _, ~w := range total {
        if ~w.Relation == \"~w\"~w {
~w        }
    }
", [Pred, VarName, VarName, Pred, JoinCond, RestCode]).

%% compile_go_join_condition(+Args, +VarMap, +Config, +VarName, -Condition)
%  Generate join condition checking shared variables
compile_go_join_condition(Args, VarMap, _Config, VarName, Condition) :-
    findall(Cond,
        (   nth0(I, Args, Arg),
            var(Arg),
            member(Arg0-source(Source, SrcIdx), VarMap),
            Arg == Arg0,
            format(string(Cond), " && ~w.Args[\"arg~w\"] == ~w.Args[\"arg~w\"]", [VarName, I, Source, SrcIdx])
        ),
        Conds),
    atomic_list_concat(Conds, "", Condition).

%% compile_go_builtins(+Builtins, +VarMap, +Config, -Code)
%  Generate builtin constraint checks
compile_go_builtins([], _, _, "").
compile_go_builtins(Builtins, VarMap, Config, Code) :-
    findall(Check,
        (   member(B, Builtins),
            compile_go_single_builtin(B, VarMap, Config, Check)
        ),
        Checks),
    (   Checks == []
    ->  Code = ""
    ;   atomic_list_concat(Checks, "\n", ChecksStr),
        format(string(Code), "\n    // Constraint checks\n~w\n", [ChecksStr])
    ).

%% compile_go_single_builtin(+Builtin, +VarMap, +Config, -Check)
compile_go_single_builtin(Goal, VarMap, Config, Check) :-
    (   Goal = (\+ NegGoal)
    ;   Goal = not(NegGoal)
    ),
    !,
    % Negation check
    NegGoal =.. [NegPred|NegArgs],
    findall(Assign,
        (   nth0(I, NegArgs, Arg),
            translate_go_expr(Arg, VarMap, Config, Expr),
            format(string(Assign), "\"arg~w\": ~w", [I, Expr])
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", ArgsStr),
    format(string(Check),
"    negFact := Fact{Relation: \"~w\", Args: map[string]interface{}{~w}}
    if _, exists := total[negFact.Key()]; exists {
        return results
    }", [NegPred, ArgsStr]).

compile_go_single_builtin(Goal, VarMap, Config, Check) :-
    Goal =.. [Op, Left, Right],
    member(Op-GoOp, ['>' - ">", '<' - "<", '>=' - ">=", '=<' - "<=", '=:=' - "==", '=\\=' - "!="]),
    !,
    translate_go_expr(Left, VarMap, Config, LeftExpr),
    translate_go_expr(Right, VarMap, Config, RightExpr),
    format(string(Check),
"    if !(~w ~w ~w) {
        return results
    }", [LeftExpr, GoOp, RightExpr]).

compile_go_single_builtin(_, _, _, "").

%% translate_go_expr(+Expr, +VarMap, +Config, -GoExpr)
%  Translate Prolog expression to Go expression
translate_go_expr(Var, VarMap, _, Expr) :-
    var(Var),
    !,
    (   member(V-source(Source, Idx), VarMap), Var == V
    ->  format(string(Expr), "~w.Args[\"arg~w\"]", [Source, Idx])
    ;   Expr = "nil"
    ).
translate_go_expr(Num, _, _, Expr) :-
    number(Num),
    !,
    format(string(Expr), "~w", [Num]).
translate_go_expr(Atom, _, _, Expr) :-
    atom(Atom),
    !,
    format(string(Expr), "\"~w\"", [Atom]).
translate_go_expr(Expr, VarMap, Config, GoExpr) :-
    Expr =.. [Op, Left, Right],
    member(Op, [+, -, *, /]),
    !,
    translate_go_expr(Left, VarMap, Config, LeftExpr),
    translate_go_expr(Right, VarMap, Config, RightExpr),
    format(string(GoExpr), "(~w ~w ~w)", [LeftExpr, Op, RightExpr]).
translate_go_expr(_, _, _, "nil").

%% compile_go_head_construction(+HeadPred, +HeadArgs, +VarMap, +Config, -Code)
%  Generate code to construct result fact
compile_go_head_construction(HeadPred, HeadArgs, VarMap, Config, Code) :-
    findall(ArgCode,
        (   nth0(I, HeadArgs, Arg),
            translate_go_expr(Arg, VarMap, Config, Expr),
            format(string(ArgCode), "\"arg~w\": ~w", [I, Expr])
        ),
        ArgCodes),
    atomic_list_concat(ArgCodes, ", ", ArgsStr),
    format(string(Code),
"results = append(results, Fact{Relation: \"~w\", Args: map[string]interface{}{~w}})", 
        [HeadPred, ArgsStr]).

%% compile_go_generator_execution(+Pred, +RuleNames, +Options, -Code)
%  Generate Solve() fixpoint loop and main()
compile_go_generator_execution(_Pred, RuleNames, Options, Code) :-
    findall(Call,
        (   member(Name, RuleNames),
            format(string(Call), "\t\t\tnewFacts = append(newFacts, ~w(fact, total, idx)...)", [Name])
        ),
        Calls),
    atomic_list_concat(Calls, "\n", CallsStr),
    
    % Generate stdin loading code if json_input is enabled
    (   option(json_input(true), Options)
    ->  StdinCode = "
    // Load additional facts from stdin (JSONL format)
    scanner := bufio.NewScanner(os.Stdin)
    for scanner.Scan() {
        var fact Fact
        if err := json.Unmarshal(scanner.Bytes(), &fact); err == nil {
            total[fact.Key()] = fact
        }
    }
"
    ;   StdinCode = ""
    ),
    
    % Generate database code if db_backend is specified
    (   option(db_backend(bbolt), Options)
    ->  option(db_file(DbFile), Options, 'facts.db'),
        option(db_bucket(Bucket), Options, 'facts'),
        atom_string(Bucket, BucketStr),
        format(string(DbLoadCode), "
    // Load facts from database
    db, err := bolt.Open(\"~w\", 0600, nil)
    if err != nil {
        log.Printf(\"Warning: Could not open database: %v\", err)
    } else {
        defer db.Close()
        db.View(func(tx *bolt.Tx) error {
            b := tx.Bucket([]byte(\"~w\"))
            if b != nil {
                b.ForEach(func(k, v []byte) error {
                    var fact Fact
                    if err := json.Unmarshal(v, &fact); err == nil {
                        total[fact.Key()] = fact
                    }
                    return nil
                })
            }
            return nil
        })
    }
", [DbFile, BucketStr]),
        format(string(DbSaveCode), "
    // Save all facts to database
    db2, err := bolt.Open(\"~w\", 0600, nil)
    if err == nil {
        defer db2.Close()
        db2.Update(func(tx *bolt.Tx) error {
            b, _ := tx.CreateBucketIfNotExists([]byte(\"~w\"))
            for key, fact := range total {
                v, _ := json.Marshal(fact)
                b.Put([]byte(key), v)
            }
            return nil
        })
    }
", [DbFile, BucketStr])
    ;   DbLoadCode = "",
        DbSaveCode = ""
    ),
    
    % Check for parallel execution
    (   option(workers(Workers), Options), Workers > 1
    ->  compile_go_generator_parallel_solve(Workers, RuleNames, StdinCode, Code)
    ;   % Sequential execution (default)
        format(string(Code),
"// Solve runs fixpoint iteration until no new facts are derived
func Solve() map[string]Fact {
    total := make(map[string]Fact)
    
    // Initialize with base facts
    for _, fact := range GetInitialFacts() {
        total[fact.Key()] = fact
    }
~w~w
    // Build initial index
    idx := BuildIndex(total)
    
    // Fixpoint iteration
    changed := true
    for changed {
        changed = false
        var newFacts []Fact
        
        for _, fact := range total {
~w
        }
        
        // Add new facts to total and index
        for _, nf := range newFacts {
            key := nf.Key()
            if _, exists := total[key]; !exists {
                total[key] = nf
                idx.Add(&nf)
                changed = true
            }
        }
    }
~w
    return total
}

func main() {
    result := Solve()
    for _, fact := range result {
        b, _ := json.Marshal(fact)
        fmt.Println(string(b))
    }
}
", [StdinCode, DbLoadCode, CallsStr, DbSaveCode])
    ).

%% compile_go_generator_parallel_solve(+Workers, +RuleNames, +StdinCode, -Code)
%  Generate parallel Solve() using goroutines
compile_go_generator_parallel_solve(Workers, RuleNames, StdinCode, Code) :-
    % Generate rule calls for worker function
    findall(Call,
        (   member(Name, RuleNames),
            format(string(Call), "\t\t\tresults = append(results, ~w(fact, total, idx)...)", [Name])
        ),
        Calls),
    atomic_list_concat(Calls, "\n", CallsStr),
    
    format(string(Code),
"// applyRules applies all rules to a fact and sends results to channel
func applyRules(fact Fact, total map[string]Fact, idx *Index, resultChan chan<- Fact, wg *sync.WaitGroup) {
    defer wg.Done()
    var results []Fact
~w
    for _, r := range results {
        resultChan <- r
    }
}

// Solve runs parallel fixpoint iteration with ~w workers
func Solve() map[string]Fact {
    total := make(map[string]Fact)
    
    // Initialize with base facts
    for _, fact := range GetInitialFacts() {
        total[fact.Key()] = fact
    }
~w
    // Build initial index
    idx := BuildIndex(total)
    
    // Fixpoint iteration with parallel workers
    changed := true
    for changed {
        changed = false
        
        // Collect facts into slice for parallel processing
        facts := make([]Fact, 0, len(total))
        for _, f := range total {
            facts = append(facts, f)
        }
        
        // Channel to collect new facts
        resultChan := make(chan Fact, len(facts)*10)
        var wg sync.WaitGroup
        
        // Dispatch workers (each processes a subset of facts)
        numWorkers := ~w
        chunkSize := (len(facts) + numWorkers - 1) / numWorkers
        
        for i := 0; i < numWorkers; i++ {
            start := i * chunkSize
            end := start + chunkSize
            if end > len(facts) {
                end = len(facts)
            }
            if start >= len(facts) {
                break
            }
            
            // Process chunk in goroutine
            wg.Add(1)
            go func(chunk []Fact) {
                defer wg.Done()
                for _, fact := range chunk {
                    var results []Fact
~w
                    for _, r := range results {
                        resultChan <- r
                    }
                }
            }(facts[start:end])
        }
        
        // Close channel when all workers done
        go func() {
            wg.Wait()
            close(resultChan)
        }()
        
        // Collect results
        for nf := range resultChan {
            key := nf.Key()
            if _, exists := total[key]; !exists {
                total[key] = nf
                idx.Add(&nf)
                changed = true
            }
        }
    }
    
    return total
}

func main() {
    result := Solve()
    for _, fact := range result {
        b, _ := json.Marshal(fact)
        fmt.Println(string(b))
    }
}
", [CallsStr, Workers, StdinCode, Workers, CallsStr]).

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

%% is_aggregation_predicate(+Body)
%  Check if body contains an aggregation goal
is_aggregation_predicate(Body) :-
    format('Checking body for aggregation: ~w~n', [Body]),
    extract_aggregation_spec(Body, _, _, _).

%% extract_aggregation_spec(+Body, -Op, -Goal, -Result)
%  Extract aggregation specification from body
%  Supports: aggregate(Op, Goal, Result)
extract_aggregation_spec(aggregate(Op, Goal, Result), Op, Goal, Result).
extract_aggregation_spec((aggregate(Op, Goal, Result), _), Op, Goal, Result).

%% ============================================
%% AGGREGATION PATTERN COMPILATION
%% ============================================

%% compile_aggregation_to_go(+Pred, +Arity, +AggOp, +FieldDelim, +IncludePackage, -GoCode)
%  Compile aggregation operations (sum, count, max, min, avg)
%
%% compile_aggregation_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile aggregation predicate (new style)
%
compile_aggregation_mode(Pred, Arity, Options, GoCode) :- !,
    % Get predicate clauses
    functor(Head, Pred, Arity),
    clause(Head, Body),
    
    % Extract aggregation spec
    extract_aggregation_spec(Body, Op, Goal, ResultVar),
    
    % Determine if we are aggregating over a field or just counting
    (   Op = count
    ->  AggField = none,
        OpType = count
    ;   compound(Op), functor(Op, AggType, 1), arg(1, Op, AggVar)
    ->  AggField = AggVar,
        OpType = AggType
    ;   % Fallback / Error case
        OpType = Op, AggField = none
    ),
    
    % Extract JSON field mappings from the inner goal
    extract_json_field_mappings(Goal, FieldMappings),
    
    % Generate Go code
    generate_aggregation_code(OpType, AggField, ResultVar, FieldMappings, Options, ScriptBody),
    
    % Determine imports
    (   OpType = count ->
        Imports = '\t"bufio"\n\t"encoding/json"\n\t"fmt"\n\t"os"'
    ;   Imports = '\t"bufio"\n\t"encoding/json"\n\t"fmt"\n\t"os"'
    ),
    
    % Wrap in package main if requested
    option(include_package(IncludePackage), Options, true),
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

%% compile_aggregation_to_go(+Pred, +Arity, +AggOp, +FieldDelim, +IncludePackage, -GoCode)
%  Compile aggregation operations (legacy option-based)
compile_aggregation_to_go(Pred, Arity, AggOp, FieldDelim, IncludePackage, GoCode) :-
    atom_string(Pred, PredStr),
    map_field_delimiter(FieldDelim, DelimChar),
    format('  Legacy Aggregation type: ~w~n', [AggOp]),
    generate_aggregation_go(AggOp, Arity, DelimChar, ScriptBody),
    (   AggOp = count ->
        Imports = '\t"bufio"\n\t"fmt"\n\t"os"'
    ;   Imports = '\t"bufio"\n\t"fmt"\n\t"os"\n\t"strconv"'
    ),
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
term_to_go_expr(Term, _, GoExpr) :-
    number(Term), !,
    format(atom(GoExpr), '~w', [Term]).
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
%% DATABASE KEY EXPRESSION COMPILER
%% ============================================

%% compile_key_expression(+KeyExpr, +FieldMappings, +Options, -KeyCode, -Imports)
%  Compile a key expression into Go code that generates a database key
%
%  KeyExpr can be:
%    - field(FieldName)              - Extract a single field value
%    - composite([Expr1, Expr2, ...]) - Concatenate multiple expressions
%    - hash(Expr)                    - SHA-256 hash of an expression
%    - hash(Expr, Algorithm)         - Hash with specific algorithm
%    - literal(String)               - Constant string value
%    - substring(Expr, Start, Len)   - Extract substring
%    - uuid()                        - Generate UUID
%    - auto_increment()              - Sequential counter (future)
%
%  Returns:
%    - KeyCode: Go code that evaluates to the key (as string)
%    - Imports: List of required import packages
%
compile_key_expression(KeyExpr, FieldMappings, Options, KeyCode, Imports) :-
    compile_key_expr(KeyExpr, FieldMappings, Options, KeyCode, Imports).

%% compile_key_expr/5 - Main expression compiler
%
%  field(FieldName) - Extract field value
compile_key_expr(field(FieldName), FieldMappings, _Options, KeyCode, []) :-
    % Find field position
    (   nth1(Pos, FieldMappings, FieldName-_)
    ->  true
    ;   nth1(Pos, FieldMappings, nested(Path, _)),
        last(Path, FieldName)
    ->  true
    ;   format('ERROR: Field ~w not found in mappings: ~w~n', [FieldName, FieldMappings]),
        fail
    ),
    format(atom(FieldVar), 'field~w', [Pos]),
    format(string(KeyCode), 'fmt.Sprintf("%v", ~s)', [FieldVar]).

%  composite([Expr1, Expr2, ...]) - Concatenate expressions
compile_key_expr(composite(Exprs), FieldMappings, Options, KeyCode, AllImports) :-
    % Get delimiter (default ':')
    option(db_key_delimiter(Delimiter), Options, ':'),

    % Compile each sub-expression
    maplist(compile_key_expr_for_composite(FieldMappings, Options), Exprs, ExprCodes, ExprImportsList),

    % Flatten imports
    append(ExprImportsList, AllImports),

    % Build format string and args
    length(Exprs, NumExprs),
    length(FormatSpecifiers, NumExprs),
    maplist(=('%s'), FormatSpecifiers),
    atomic_list_concat(FormatSpecifiers, Delimiter, FormatString),
    atomic_list_concat(ExprCodes, ', ', ArgsString),

    format(string(KeyCode), 'fmt.Sprintf("~s", ~s)', [FormatString, ArgsString]).

%  hash(Expr) - SHA-256 hash of expression
compile_key_expr(hash(Expr), FieldMappings, Options, KeyCode, Imports) :-
    compile_key_expr(hash(Expr, sha256), FieldMappings, Options, KeyCode, Imports).

%  hash(Expr, Algorithm) - Hash with specific algorithm
compile_key_expr(hash(Expr, Algorithm), FieldMappings, Options, KeyCode, Imports) :-
    % Compile the inner expression
    compile_key_expr(Expr, FieldMappings, Options, ExprCode, ExprImports),

    % Generate hash code based on algorithm
    (   Algorithm = sha256
    ->  HashImport = 'crypto/sha256',
        format(string(KeyCode), 'func() string {
\t\tvalStr := ~s
\t\thash := sha256.Sum256([]byte(valStr))
\t\treturn hex.EncodeToString(hash[:])
\t}()', [ExprCode])
    ;   Algorithm = md5
    ->  HashImport = 'crypto/md5',
        format(string(KeyCode), 'func() string {
\t\tvalStr := ~s
\t\thash := md5.Sum([]byte(valStr))
\t\treturn hex.EncodeToString(hash[:])
\t}()', [ExprCode])
    ;   format('ERROR: Unsupported hash algorithm: ~w~n', [Algorithm]),
        fail
    ),

    append(ExprImports, [HashImport, 'encoding/hex'], Imports).

%  literal(String) - Constant string value
compile_key_expr(literal(String), _FieldMappings, _Options, KeyCode, []) :-
    format(string(KeyCode), '"~s"', [String]).

%  substring(Expr, Start, Length) - Extract substring
compile_key_expr(substring(Expr, Start, Length), FieldMappings, Options, KeyCode, Imports) :-
    compile_key_expr(Expr, FieldMappings, Options, ExprCode, Imports),
    End is Start + Length,
    format(string(KeyCode), 'func() string {
\t\tstr := ~s
\t\tif len(str) > ~w {
\t\t\treturn str[~w:~w]
\t\t}
\t\treturn str
\t}()', [ExprCode, End, Start, End]).

%  uuid() - Generate UUID
compile_key_expr(uuid(), _FieldMappings, _Options, KeyCode, ['github.com/google/uuid']) :-
    KeyCode = 'uuid.New().String()'.

%  auto_increment() - Sequential counter (requires state management)
compile_key_expr(auto_increment(), _FieldMappings, _Options, _KeyCode, _Imports) :-
    format('ERROR: auto_increment() not yet implemented~n'),
    fail.

%% Helper for composite - wraps each expression's code
compile_key_expr_for_composite(FieldMappings, Options, Expr, Code, Imports) :-
    compile_key_expr(Expr, FieldMappings, Options, Code, Imports).

%% normalize_key_strategy(+Options, -NormalizedOptions)
%  Normalize key strategy options for backward compatibility
%
%  Converts:
%    db_key_field(Field) → db_key_strategy(field(Field))
%    db_key_fields([F1,F2]) → db_key_strategy(composite([field(F1), field(F2)]))
%
normalize_key_strategy(Options, NormalizedOptions) :-
    % Check if db_key_strategy is already present
    (   option(db_key_strategy(_), Options)
    ->  % Already normalized
        NormalizedOptions = Options
    ;   option(db_key_field(Field), Options)
    ->  % Convert db_key_field(F) to db_key_strategy(field(F))
        select(db_key_field(Field), Options, TempOptions),
        NormalizedOptions = [db_key_strategy(field(Field))|TempOptions]
    ;   option(db_key_fields(Fields), Options)
    ->  % Convert db_key_fields([...]) to db_key_strategy(composite([field(F1), ...]))
        maplist(wrap_field_expr, Fields, FieldExprs),
        select(db_key_fields(Fields), Options, TempOptions),
        NormalizedOptions = [db_key_strategy(composite(FieldExprs))|TempOptions]
    ;   % No key strategy specified - will use default
        NormalizedOptions = Options
    ).

wrap_field_expr(Field, field(Field)).

%% ============================================
%% DATABASE QUERY CONSTRAINTS (Phase 8a)
%% ============================================

%% extract_db_constraints(+Body, -JsonRecord, -Constraints)
%  Extract filter constraints from predicate body
%  Separates json_record/1 from comparison constraints
%
%  Supported constraints (Phase 8a):
%    - Comparisons: >, <, >=, =<, =, \=
%    - Implicit AND (multiple constraints in body)
%
extract_db_constraints(Body, JsonRecord, Constraints) :-
    extract_constraints_impl(Body, none, JsonRecord, [], Constraints).

% Helper: recursively extract constraints from conjunction
extract_constraints_impl((A, B), JsonRecAcc, JsonRec, ConsAcc, Constraints) :- !,
    extract_constraints_impl(A, JsonRecAcc, JsonRec1, ConsAcc, Cons1),
    extract_constraints_impl(B, JsonRec1, JsonRec, Cons1, Constraints).

% json_record/1 - save for later
extract_constraints_impl(json_record(Fields), _JsonRecAcc, json_record(Fields), ConsAcc, ConsAcc) :- !.

% Comparison constraints - collect them
extract_constraints_impl(Constraint, JsonRecAcc, JsonRecAcc, ConsAcc, [Constraint|ConsAcc]) :-
    is_comparison_constraint(Constraint), !.

extract_constraints_impl(Constraint, JsonRecAcc, JsonRecAcc, ConsAcc, [Constraint|ConsAcc]) :-
    is_functional_constraint(Constraint), !.

% Skip other predicates (like json_get, etc.)
extract_constraints_impl(_, JsonRecAcc, JsonRecAcc, ConsAcc, ConsAcc).

%% is_comparison_constraint(+Term)
%  Check if term is a supported comparison constraint
%
is_comparison_constraint(_ > _).
is_comparison_constraint(_ < _).
is_comparison_constraint(_ >= _).
is_comparison_constraint(_ =< _).
is_comparison_constraint(_ = _).
is_comparison_constraint(_ \= _).
is_comparison_constraint(_ =@= _).  % Case-insensitive equality

%% is_functional_constraint(+Term)
%  Check if term is a functional constraint (contains, member, etc.)
%
is_functional_constraint(contains(_, _)).
is_functional_constraint(member(_, _)).

%% is_numeric_constraint(+Constraint)
%  Check if constraint requires numeric type conversion (>, <, >=, =<)
%  Equality and inequality (=, \=) can work with any type
%
is_numeric_constraint(_ > _).
is_numeric_constraint(_ < _).
is_numeric_constraint(_ >= _).
is_numeric_constraint(_ =< _).

%% constraints_need_strings(+Constraints)
%  Check if any constraint requires the strings package
%  True if constraints contain =@= or contains/2
%
constraints_need_strings(Constraints) :-
    member(Constraint, Constraints),
    (   Constraint = (_ =@= _)
    ;   Constraint = contains(_, _)
    ), !.

%% generate_filter_checks(+Constraints, +FieldMappings, -GoCode)
%  Generate Go if statements for constraint checking
%  Returns empty string if no constraints
%
generate_filter_checks([], _, '') :- !.
generate_filter_checks(Constraints, FieldMappings, GoCode) :-
    findall(CheckCode,
        (member(Constraint, Constraints),
         constraint_to_go_check(Constraint, FieldMappings, CheckCode)),
        Checks),
    atomic_list_concat(Checks, '\n', GoCode).

%% constraint_to_go_check(+Constraint, +FieldMappings, -GoCode)
%  Convert a Prolog constraint to Go if statement
%
constraint_to_go_check(Left > Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w > ~w\n\t\t\tif !(~s > ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left < Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w < ~w\n\t\t\tif !(~s < ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left >= Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w >= ~w\n\t\t\tif !(~s >= ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left =< Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w =< ~w\n\t\t\tif !(~s <= ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left = Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w = ~w\n\t\t\tif !(~s == ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left \= Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w \\= ~w\n\t\t\tif !(~s != ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

% Case-insensitive equality (requires strings package)
constraint_to_go_check(Left =@= Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w =@= ~w (case-insensitive)\n\t\t\tif !strings.EqualFold(fmt.Sprintf("%v", ~s), fmt.Sprintf("%v", ~s)) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

% Contains check (requires strings package)
constraint_to_go_check(contains(Haystack, Needle), FieldMappings, Code) :- !,
    field_term_to_go_expr(Haystack, FieldMappings, HaystackExpr),
    field_term_to_go_expr(Needle, FieldMappings, NeedleExpr),
    format(string(Code), '\t\t\t// Filter: contains(~w, ~w)\n\t\t\tif !strings.Contains(fmt.Sprintf("%v", ~s), fmt.Sprintf("%v", ~s)) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Haystack, Needle, HaystackExpr, NeedleExpr]).

% List membership check
constraint_to_go_check(member(Element, List), FieldMappings, Code) :- !,
    field_term_to_go_expr(Element, FieldMappings, ElementExpr),
    generate_member_check_code(ElementExpr, List, FieldMappings, Code).

%% field_term_to_go_expr(+Term, +FieldMappings, -GoExpr)
%  Convert a Prolog term to Go expression for filter constraints
%  Handles variables (map to fieldN) and literals
%  FieldMappings is a list of Name-Var pairs from json_record
%
field_term_to_go_expr(Term, FieldMappings, GoExpr) :-
    var(Term), !,
    % Find which field this variable corresponds to
    (   nth1(Pos, FieldMappings, _-Var),
        Term == Var
    ->  format(atom(GoExpr), 'field~w', [Pos])
    ;   % Variable not in mappings - shouldn't happen
        format('WARNING: Variable ~w not found in field mappings~n', [Term]),
        GoExpr = 'unknownVar'
    ).

field_term_to_go_expr(Term, _, GoExpr) :-
    string(Term), !,
    % String literal - use as-is with quotes
    format(atom(GoExpr), '"~s"', [Term]).

field_term_to_go_expr(Term, _, GoExpr) :-
    atom(Term), !,
    % Atom literal - quote it for Go string
    format(atom(GoExpr), '"~w"', [Term]).

field_term_to_go_expr(Term, _, GoExpr) :-
    number(Term), !,
    format(atom(GoExpr), '~w', [Term]).

field_term_to_go_expr(Term, _, GoExpr) :-
    % Fallback for unknown terms
    format(atom(GoExpr), '%s /* ~w */', [Term]).

%% generate_member_check_code(+ElementExpr, +List, +FieldMappings, -Code)
%  Generate Go code for list membership check
%  Handles both string and numeric list members
%
generate_member_check_code(ElementExpr, List, FieldMappings, Code) :-
    % Convert list elements to Go expressions
    findall(GoExpr,
        (member(ListItem, List),
         field_term_to_go_expr(ListItem, FieldMappings, GoExpr)),
        GoExprs),
    % Generate the list items as Go slice literals
    atomic_list_concat(GoExprs, ', ', GoListItems),
    % Determine if we're checking strings or numbers
    (   List = [FirstItem|_],
        (atom(FirstItem) ; string(FirstItem))
    ->  % String membership
        format(string(Code), '\t\t\t// Filter: member(~w, list)
\t\t\toptions := []string{~s}
\t\t\tfound := false
\t\t\tfor _, opt := range options {
\t\t\t\tif fmt.Sprintf("%v", ~s) == opt {
\t\t\t\t\tfound = true
\t\t\t\t\tbreak
\t\t\t\t}
\t\t\t}
\t\t\tif !found {
\t\t\t\treturn nil // Skip record
\t\t\t}',
            [ElementExpr, GoListItems, ElementExpr])
    ;   % Numeric membership
        format(string(Code), '\t\t\t// Filter: member(~w, list)
\t\t\tfound := false
\t\t\tfor _, opt := range []interface{}{~s} {
\t\t\t\tif fmt.Sprintf("%v", ~s) == fmt.Sprintf("%v", opt) {
\t\t\t\t\tfound = true
\t\t\t\t\tbreak
\t\t\t\t}
\t\t\t}
\t\t\tif !found {
\t\t\t\treturn nil // Skip record
\t\t\t}',
            [ElementExpr, GoListItems, ElementExpr])
    ).

%% extract_used_fields(+KeyExpr, -UsedFieldPositions)
%  Extract which field positions are referenced by the key expression
%
extract_used_fields(field(FieldName), [Pos]) :-
    % Single field reference - need to find its position in FieldMappings
    % This is a simplified version - full implementation would need FieldMappings
    % For now, extract field number from field(name) atom
    !,
    Pos = 1.  % Placeholder - will be computed properly in context

extract_used_fields(composite(Exprs), AllUsedFields) :-
    !,
    findall(UsedFields,
        (member(Expr, Exprs),
         extract_used_fields(Expr, UsedFields)),
        UsedFieldsList),
    append(UsedFieldsList, AllUsedFields).

extract_used_fields(hash(Expr), UsedFields) :-
    !,
    extract_used_fields(Expr, UsedFields).

extract_used_fields(hash(Expr, _Algorithm), UsedFields) :-
    !,
    extract_used_fields(Expr, UsedFields).

extract_used_fields(substring(Expr, _, _), UsedFields) :-
    !,
    extract_used_fields(Expr, UsedFields).

extract_used_fields(literal(_), []) :- !.
extract_used_fields(uuid(), []) :- !.
extract_used_fields(auto_increment(), []) :- !.

%% extract_used_field_positions(+KeyExpr, +FieldMappings, -UsedPositions)
%  Extract actual field positions by matching field names
%
extract_used_field_positions(KeyExpr, FieldMappings, UsedPositions) :-
    extract_field_names_from_expr(KeyExpr, FieldNames),
    findall(Pos,
        (member(FieldName, FieldNames),
         nth1(Pos, FieldMappings, FieldName-_)),
        UsedPositions).

%% extract_field_names_from_expr(+KeyExpr, -FieldNames)
%  Extract all field names referenced in the expression
%
extract_field_names_from_expr(field(FieldName), [FieldName]) :- !.
extract_field_names_from_expr(composite(Exprs), AllFields) :-
    !,
    findall(Fields,
        (member(Expr, Exprs),
         extract_field_names_from_expr(Expr, Fields)),
        FieldsList),
    append(FieldsList, AllFields).
extract_field_names_from_expr(hash(Expr), Fields) :-
    !,
    extract_field_names_from_expr(Expr, Fields).
extract_field_names_from_expr(hash(Expr, _), Fields) :-
    !,
    extract_field_names_from_expr(Expr, Fields).
extract_field_names_from_expr(substring(Expr, _, _), Fields) :-
    !,
    extract_field_names_from_expr(Expr, Fields).
extract_field_names_from_expr(_, []).  % literals, uuid, etc.

%% ============================================
%% DATABASE READ MODE COMPILATION
%% ============================================

%% ============================================
%% KEY OPTIMIZATION DETECTION (Phase 8c)
%% ============================================

%% analyze_key_optimization(+KeyStrategy, +Constraints, +FieldMappings, -OptType, -OptDetails)
%  Analyze if predicate can use optimized key lookup
%  OptType: direct_lookup | prefix_scan | full_scan
%  OptDetails: Details needed for code generation
%
analyze_key_optimization(KeyStrategy, Constraints, FieldMappings, OptType, OptDetails) :-
    (   can_use_direct_lookup(KeyStrategy, Constraints, FieldMappings, KeyValue)
    ->  OptType = direct_lookup,
        OptDetails = key_value(KeyValue),
        format('  Optimization: Direct lookup (key=~w)~n', [KeyValue])
    ;   can_use_prefix_scan(KeyStrategy, Constraints, FieldMappings, PrefixValue)
    ->  OptType = prefix_scan,
        OptDetails = prefix_value(PrefixValue),
        format('  Optimization: Prefix scan (prefix=~w)~n', [PrefixValue])
    ;   OptType = full_scan,
        OptDetails = none,
        format('  Optimization: Full scan (no key match)~n')
    ).

%% can_use_direct_lookup(+KeyStrategy, +Constraints, +FieldMappings, -KeyValue)
%  Check if we can use bucket.Get() for direct key lookup
%  True if there's an exact equality constraint on the key field
%
can_use_direct_lookup([KeyField], Constraints, FieldMappings, KeyValue) :-
    % Single key field
    member(KeyField-_Var, FieldMappings),
    member(Constraint, Constraints),
    is_exact_equality_on_field(Constraint, KeyField, FieldMappings, KeyValue),
    !.

can_use_direct_lookup(KeyFields, Constraints, FieldMappings, CompositeKey) :-
    % Composite key - all fields must have exact equality
    is_list(KeyFields),
    length(KeyFields, Len),
    Len > 1,
    maplist(has_exact_constraint_for_field(Constraints, FieldMappings), KeyFields, Values),
    build_composite_key_value(Values, CompositeKey),
    !.

%% can_use_prefix_scan(+KeyStrategy, +Constraints, +FieldMappings, -PrefixValue)
%  Check if we can use cursor.Seek() for prefix scan
%  True if first N fields of composite key have exact equality
%
can_use_prefix_scan(KeyFields, Constraints, FieldMappings, PrefixValue) :-
    is_list(KeyFields),
    length(KeyFields, TotalLen),
    TotalLen > 1,  % Must be composite key
    find_matching_prefix_fields(KeyFields, Constraints, FieldMappings, PrefixFields, PrefixValues),
    length(PrefixFields, PrefixLen),
    PrefixLen > 0,
    PrefixLen < TotalLen,  % Not all fields (that would be direct lookup)
    build_composite_key_value(PrefixValues, PrefixValue),
    !.

%% is_exact_equality_on_field(+Constraint, +FieldName, +FieldMappings, -Value)
%  Check if constraint is exact equality (=) on the field and extract value
%  Rejects case-insensitive (=@=), contains, member, etc.
%
is_exact_equality_on_field(Var = Value, FieldName, FieldMappings, Value) :-
    member(FieldName-Var, FieldMappings),
    ground(Value),
    \+ is_variable_reference(Value, FieldMappings),  % Value must be literal, not another field
    !.

%% is_variable_reference(+Term, +FieldMappings)
%  Check if Term is a variable that appears in FieldMappings
%
is_variable_reference(Var, FieldMappings) :-
    var(Var),
    member(_-Var, FieldMappings),
    !.

%% has_exact_constraint_for_field(+Constraints, +FieldMappings, +FieldName, -Value)
%  Check if there's an exact equality constraint for this field
%
has_exact_constraint_for_field(Constraints, FieldMappings, FieldName, Value) :-
    member(Constraint, Constraints),
    is_exact_equality_on_field(Constraint, FieldName, FieldMappings, Value).

%% find_matching_prefix_fields(+KeyFields, +Constraints, +FieldMappings, -PrefixFields, -PrefixValues)
%  Find the longest prefix of KeyFields that all have exact equality constraints
%
find_matching_prefix_fields([Field|Rest], Constraints, FieldMappings, [Field|RestFields], [Value|RestValues]) :-
    has_exact_constraint_for_field(Constraints, FieldMappings, Field, Value),
    !,
    find_matching_prefix_fields(Rest, Constraints, FieldMappings, RestFields, RestValues).
find_matching_prefix_fields(_, _, _, [], []).

%% build_composite_key_value(+Values, -CompositeKey)
%  Build composite key string with colon separator
%  For direct lookup and prefix scan
%
build_composite_key_value([Single], Single) :- !.
build_composite_key_value(Values, CompositeKey) :-
    maplist(value_to_key_string, Values, Strings),
    atomic_list_concat(Strings, ':', CompositeKey).

%% value_to_key_string(+Value, -String)
%  Convert a value to string for key construction
%
value_to_key_string(Value, String) :-
    (   atom(Value) -> atom_string(Value, String)
    ;   string(Value) -> String = Value
    ;   number(Value) -> format(string(String), '~w', [Value])
    ;   format(string(String), '~w', [Value])
    ).

%% ============================================
%% FIELD EXTRACTION FOR DATABASE READ
%% ============================================

%% generate_field_extractions_for_read(+FieldMappings, +Constraints, +HeadArgs, -GoCode)
%  Generate field extraction code for read mode with proper type conversions
%  - Extracts all fields from FieldMappings
%  - Adds type conversions for fields used in constraints
%  - Marks unused fields with _ = fieldN to avoid Go compiler warnings
%
generate_field_extractions_for_read(FieldMappings, Constraints, HeadArgs, GoCode) :-
    % Build set of field positions used in NUMERIC constraints (need float64 conversion)
    findall(NumericPos,
        (   nth1(NumericPos, FieldMappings, _-Var),
            member(C, Constraints),
            is_numeric_constraint(C),
            term_variables(C, CVars),
            member(CV, CVars),
            CV == Var
        ),
        NumericConstraintPositions),

    findall(HeadPos,
        (   nth1(HeadPos, FieldMappings, _-Var),
            member(HV, HeadArgs),
            HV == Var
        ),
        HeadPositions),

    findall(ExtractBlock,
        (   nth1(Pos, FieldMappings, Field-_Var),
            atom_string(Field, FieldStr),

            % Check if this position needs numeric type conversion
            (   member(Pos, NumericConstraintPositions)
            ->  NeedsNumericConversion = true
            ;   NeedsNumericConversion = false
            ),

            (   member(Pos, HeadPositions)
            ->  UsedInHead = true
            ;   UsedInHead = false
            ),

            % Generate extraction with type conversion if needed
            (   NeedsNumericConversion = true
            ->  % Need type conversion for numeric comparison
                format(string(ExtractBlock), '\t\t\t// Extract field: ~w (with type conversion)
\t\t\tfield~wRaw, field~wOk := data["~s"]
\t\t\tif !field~wOk {
\t\t\t\treturn nil // Skip if field missing
\t\t\t}
\t\t\tfield~wFloat, field~wFloatOk := field~wRaw.(float64)
\t\t\tif !field~wFloatOk {
\t\t\t\treturn nil // Skip if wrong type
\t\t\t}
\t\t\tfield~w := field~wFloat',
                    [Field, Pos, Pos, FieldStr, Pos, Pos, Pos, Pos, Pos, Pos, Pos])
            ;   UsedInHead = true
            ->  % Keep as interface{} for output
                format(string(ExtractBlock), '\t\t\t// Extract field: ~w
\t\t\tfield~w, field~wOk := data["~s"]
\t\t\tif !field~wOk {
\t\t\t\treturn nil // Skip if field missing
\t\t\t}',
                    [Field, Pos, Pos, FieldStr, Pos])
            ;   % Unused field - extract and mark as unused
                format(string(ExtractBlock), '\t\t\t// Extract field: ~w (unused)
\t\t\tfield~w, field~wOk := data["~s"]
\t\t\tif !field~wOk {
\t\t\t\treturn nil // Skip if field missing
\t\t\t}
\t\t\t_ = field~w  // Mark as intentionally unused',
                    [Field, Pos, Pos, FieldStr, Pos, Pos])
            )
        ),
        ExtractBlocks),
    atomic_list_concat(ExtractBlocks, '\n', GoCode).

%% generate_output_for_read(+HeadArgs, +FieldMappings, -GoCode)
%  Generate JSON output code with selected fields only
%  Creates a map with only the fields that appear in the predicate head
%
generate_output_for_read(HeadArgs, FieldMappings, GoCode) :-
    % Build a map of selected fields
    findall(FieldName:Pos,
        (   nth1(Idx, HeadArgs, Var),
            nth1(Pos, FieldMappings, FieldName-MappedVar),
            Var == MappedVar
        ),
        FieldSelections),

    % Generate output struct
    findall(FieldPair,
        (   member(FieldName:Pos, FieldSelections),
            atom_string(FieldName, FieldStr),
            format(string(FieldPair), '"~s": field~w', [FieldStr, Pos])
        ),
        FieldPairs),
    atomic_list_concat(FieldPairs, ', ', FieldsStr),

    format(string(GoCode), '\t\t\t// Output selected fields
\t\t\toutput, err := json.Marshal(map[string]interface{}{~s})
\t\t\tif err != nil {
\t\t\t\treturn nil
\t\t\t}
\t\t\tfmt.Println(string(output))', [FieldsStr]).

%% ============================================
%% DATABASE ACCESS CODE GENERATION (Phase 8c)
%% ============================================

%% generate_direct_lookup_code(+DbFile, +BucketStr, +KeyValue, +ProcessCode, -BodyCode)
%  Generate optimized code using bucket.Get() for direct key lookup
%
generate_direct_lookup_code(DbFile, BucketStr, KeyValue, ProcessCode, BodyCode) :-
    atom_string(KeyValue, KeyStr),
    format(string(BodyCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Direct lookup using bucket.Get() (optimized)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\t// Get record by key
\t\tkey := []byte("~s")
\t\tvalue := bucket.Get(key)
\t\tif value == nil {
\t\t\treturn nil // Key not found
\t\t}

\t\t// Deserialize JSON record
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(value, &data); err != nil {
\t\t\tfmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\\n", err)
\t\t\treturn nil
\t\t}

~s
\t\treturn nil
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}
', [DbFile, BucketStr, BucketStr, KeyStr, ProcessCode]).

%% generate_prefix_scan_code(+DbFile, +BucketStr, +PrefixValue, +ProcessCode, -BodyCode)
%  Generate optimized code using cursor.Seek() for prefix scan
%
generate_prefix_scan_code(DbFile, BucketStr, PrefixValue, ProcessCode, BodyCode) :-
    atom_string(PrefixValue, PrefixStr),
    format(string(BodyCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Prefix scan using cursor.Seek() (optimized)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\t// Seek to first key with prefix
\t\tcursor := bucket.Cursor()
\t\tprefix := []byte("~s:")

\t\tfor k, v := cursor.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = cursor.Next() {
\t\t\t// Deserialize JSON record
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\tfmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\\n", err)
\t\t\t\tcontinue // Continue with next record
\t\t\t}

~s
\t\t}
\t\treturn nil
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}
', [DbFile, BucketStr, BucketStr, PrefixStr, ProcessCode]).

%% generate_full_scan_code(+DbFile, +BucketStr, +RecordsDesc, +ProcessCode, -BodyCode)
%  Generate standard code using bucket.ForEach() for full scan
%
generate_full_scan_code(DbFile, BucketStr, RecordsDesc, ProcessCode, BodyCode) :-
    format(string(Header), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Read ~s from bucket
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

', [DbFile, RecordsDesc, BucketStr, BucketStr]),
    Footer = '
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}
',
    string_concat(Header, ProcessCode, Temp),
    string_concat(Temp, Footer, BodyCode).

%% compile_database_read_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate to read from bbolt database and output as JSON
%  Supports optional filtering based on constraints in predicate body
%  Phase 8c: Includes key optimization detection and optimized code generation
%
compile_database_read_mode(Pred, Arity, Options, GoCode) :-
    % Get database options
    option(db_file(DbFile), Options, 'data.db'),
    option(db_bucket(BucketName), Options, Pred),
    atom_string(BucketName, BucketStr),
    option(include_package(IncludePackage), Options, true),

    % Get key strategy (can be single field or list of fields)
    (   option(db_key_field(KeyField), Options)
    ->  (   is_list(KeyField)
        ->  KeyStrategy = KeyField
        ;   KeyStrategy = [KeyField]
        ),
        format('  Key strategy: ~w~n', [KeyStrategy])
    ;   KeyStrategy = none,
        format('  No key strategy specified~n')
    ),

    % Check if predicate has a body with constraints
    functor(Head, Pred, Arity),
    (   clause(Head, Body),
        Body \= true
    ->  % Has body - extract constraints and field mappings
        format('  Predicate body: ~w~n', [Body]),
        extract_db_constraints(Body, JsonRecord, Constraints),
        (   JsonRecord = json_record(FieldMappings0)
        ->  % FieldMappings0 is the list of Name-Var pairs from json_record
            FieldMappings = FieldMappings0,
            Head =.. [_|HeadArgs],
            format('  Field mappings: ~w~n', [FieldMappings]),
            format('  Constraints: ~w~n', [Constraints])
        ;   format('ERROR: No json_record/1 found in predicate body~n'),
            fail
        ),

        % Analyze key optimization opportunities (Phase 8c)
        (   KeyStrategy \= none,
            Constraints \= []
        ->  format('  Analyzing key optimization...~n'),
            analyze_key_optimization(KeyStrategy, Constraints, FieldMappings, OptType, OptDetails)
        ;   OptType = full_scan,
            OptDetails = none,
            format('  Skipping optimization (no key strategy or constraints)~n')
        ),

        % Generate field extraction code (with type conversions for constraints)
        format('  Generating field extractions...~n'),
        generate_field_extractions_for_read(FieldMappings, Constraints, HeadArgs, ExtractCode),
        format('  Generated ~w chars of extraction code~n', [ExtractCode]),
        % Generate filter checks
        format('  Generating filter checks...~n'),
        generate_filter_checks(Constraints, FieldMappings, FilterCode),
        format('  Generated ~w chars of filter code~n', [FilterCode]),
        % Generate output code (selected fields only)
        format('  Generating output code...~n'),
        generate_output_for_read(HeadArgs, FieldMappings, OutputCode),
        format('  Generated ~w chars of output code~n', [OutputCode]),
        % Combine extraction + filter + output
        format('  Combining code sections...~n'),
        (   FilterCode \= ''
        ->  format(string(ProcessCode), '~s\n~s\n~s', [ExtractCode, FilterCode, OutputCode])
        ;   format(string(ProcessCode), '~s\n~s', [ExtractCode, OutputCode])
        ),
        format('  Process code ready: ~w chars~n', [ProcessCode]),
        HasFilters = true,
        format('  HasFilters set to true~n')
    ;   % No body or body is 'true' - read all records as-is
        format('  No predicate body found - reading all records~n'),
        ProcessCode = '\t\t\t// Output as JSON
\t\t\toutput, err := json.Marshal(data)
\t\t\tif err != nil {
\t\t\t\tfmt.Fprintf(os.Stderr, "Error marshaling output: %v\\n", err)
\t\t\t\treturn nil // Continue with next record
\t\t\t}

\t\t\tfmt.Println(string(output))',
        HasFilters = false
    ),

    % Generate database read code (Phase 8c: with optimizations)
    format('  Generating database read code...~n'),
    (   HasFilters = true
    ->  RecordsDesc = 'filtered records'
    ;   RecordsDesc = 'all records'
    ),

    % Generate appropriate database access code based on optimization type
    (   var(OptType)
    ->  % No optimization analysis (no constraints or no key strategy)
        format('  Using full scan (no optimization analysis)~n'),
        generate_full_scan_code(DbFile, BucketStr, RecordsDesc, ProcessCode, BodyCode)
    ;   OptType = direct_lookup
    ->  % Direct lookup optimization
        OptDetails = key_value(KeyValue),
        format('  Generating direct lookup code~n'),
        generate_direct_lookup_code(DbFile, BucketStr, KeyValue, ProcessCode, BodyCode)
    ;   OptType = prefix_scan
    ->  % Prefix scan optimization
        OptDetails = prefix_value(PrefixValue),
        format('  Generating prefix scan code~n'),
        generate_prefix_scan_code(DbFile, BucketStr, PrefixValue, ProcessCode, BodyCode)
    ;   % Full scan (default/fallback)
        format('  Using full scan~n'),
        generate_full_scan_code(DbFile, BucketStr, RecordsDesc, ProcessCode, BodyCode)
    ),
    format('  Body generated successfully (~w chars)~n', [BodyCode]),

    % Wrap in package if requested
    (   IncludePackage = true
    ->  % Check if we need bytes package (for prefix scan optimization)
        (   nonvar(OptType), OptType = prefix_scan
        ->  BytesImport = '\t"bytes"\n'
        ;   BytesImport = ''
        ),
        % Check if we need strings package (only if Constraints is defined)
        (   (var(Constraints) ; Constraints = [])
        ->  % No constraints or empty constraints - no strings needed
            StringsImport = ''
        ;   % Check if constraints need strings package
            (   constraints_need_strings(Constraints)
            ->  StringsImport = '\t"strings"\n'
            ;   StringsImport = ''
            )
        ),
        % Build package with conditional bytes and strings imports
        format(string(GoCode), 'package main

import (
~s\t"encoding/json"
\t"fmt"
\t"os"
~s
\tbolt "go.etcd.io/bbolt"
)

func main() {
~s}
', [BytesImport, StringsImport, BodyCode])
    ;   GoCode = BodyCode
    ).

%% ============================================


%% ============================================
%% GROUP BY AGGREGATION SUPPORT (Phase 9b)
%% ============================================

%% is_group_by_predicate(+Body)
%  Check if predicate body contains group_by/3 or group_by/4
%  group_by/3: group_by(GroupField, Goal, AggOpList) for multiple aggregations
%  group_by/4: group_by(GroupField, Goal, AggOp, Result) for single aggregation
%
is_group_by_predicate(group_by(_GroupField, _Goal, _AggOp)).
is_group_by_predicate(group_by(_GroupField, _Goal, _AggOp, _Result)).
is_group_by_predicate((group_by(_GroupField, _Goal, _AggOp), _Rest)).
is_group_by_predicate((group_by(_GroupField, _Goal, _AggOp, _Result), _Rest)).
is_group_by_predicate((_First, Rest)) :-
    is_group_by_predicate(Rest).

%% extract_group_by_spec(+Body, -GroupField, -Goal, -AggOp, -Result)
%  Extract group_by operation components
%  Handles both group_by/3 (Result = null) and group_by/4 (explicit Result)
%
extract_group_by_spec(group_by(GroupField, Goal, AggOp), GroupField, Goal, AggOp, null) :- !.
extract_group_by_spec(group_by(GroupField, Goal, AggOp, Result), GroupField, Goal, AggOp, Result) :- !.
extract_group_by_spec((group_by(GroupField, Goal, AggOp), _Rest), GroupField, Goal, AggOp, null) :- !.
extract_group_by_spec((group_by(GroupField, Goal, AggOp, Result), _Rest), GroupField, Goal, AggOp, Result) :- !.
extract_group_by_spec((_First, Rest), GroupField, Goal, AggOp, Result) :-
    extract_group_by_spec(Rest, GroupField, Goal, AggOp, Result).

%% extract_having_constraints(+Body, -Constraints)
%  Extract constraints that appear after group_by (HAVING clause)
%  Returns null if no constraints, or the constraint goals
%
extract_having_constraints(group_by(_, _, _), null) :- !.
extract_having_constraints(group_by(_, _, _, _), null) :- !.
extract_having_constraints((group_by(_, _, _), Rest), Rest) :- !.
extract_having_constraints((group_by(_, _, _, _), Rest), Rest) :- !.
extract_having_constraints((_First, Rest), Constraints) :-
    extract_having_constraints(Rest, Constraints).

%% compile_group_by_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate with GROUP BY aggregation
%
compile_group_by_mode(Pred, Arity, Options, GoCode) :-
    % Get predicate definition
    functor(Head, Pred, Arity),
    clause(Head, Body),

    % Extract group_by specification
    extract_group_by_spec(Body, GroupField, Goal, AggOp, Result),
    format('  Group by field: ~w~n', [GroupField]),
    format('  Aggregation: ~w~n', [AggOp]),
    format('  Goal: ~w~n', [Goal]),

    % Extract HAVING constraints (Phase 9c-2)
    extract_having_constraints(Body, HavingConstraints),
    (   HavingConstraints \= null
    ->  format('  HAVING constraints: ~w~n', [HavingConstraints])
    ;   true
    ),

    % Extract field mappings from json_record
    (   Goal = json_record(FieldMappings)
    ->  format('  Field mappings: ~w~n', [FieldMappings])
    ;   format('ERROR: No json_record/1 found in group_by goal~n'),
        fail
    ),

    % Generate grouped aggregation code
    (   option(json_input(true), Options)
    ->  % JSONL Stream Mode
        format('  Generating JSONL stream code for group_by~n'),
        generate_group_by_code_jsonl(GroupField, FieldMappings, AggOp, Result, HavingConstraints, Options, AggBody),
        BaseImports = ["bufio", "encoding/json", "fmt", "os"]
    ;   % Database Mode (bbolt)
        format('  Generating bbolt code for group_by~n'),
        
        % Get database options
        option(db_file(DbFile), Options, 'data.db'),
        option(db_bucket(BucketAtom), Options, Pred),
        atom_string(BucketAtom, BucketStr),
        
        generate_group_by_code(GroupField, FieldMappings, AggOp, Result, HavingConstraints, DbFile, BucketStr, AggBody),
        BaseImports = ["encoding/json", "fmt", "os", "bolt \"go.etcd.io/bbolt\""]
    ),

    % Add strings import if needed (for nested grouping or string manipulation)
    (   (is_list(GroupField) ; member(nested(_, _), FieldMappings))
    ->  append(BaseImports, ["strings"], ImportsList)
    ;   ImportsList = BaseImports
    ),
    
    % Sort and format imports
    sort(ImportsList, UniqueImports),
    maplist(format_import, UniqueImports, ImportLines),
    atomic_list_concat(ImportLines, '\n', Imports),

    % Wrap in package main
    format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, AggBody]).



%% generate_group_by_code(+GroupField, +FieldMappings, +AggOp, +Result, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Generate Go code for grouped aggregation with optional HAVING clause
%
generate_group_by_code(GroupField, FieldMappings, AggOp, Result, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Check if GroupField is a list (nested grouping) - handle differently
    (   is_list(GroupField)
    ->  % Nested grouping: GroupField is a list like [_1820, _1822]
        % Pass the list directly to multi-aggregation code (it will extract field names)
        format('  Nested grouping field list detected: ~w~n', [GroupField]),
        (   is_list(AggOp)
        ->  % Already in list format
            generate_multi_aggregation_code(GroupField, FieldMappings, AggOp, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   % Single aggregation with nested grouping - convert to multi-agg format
            format('  Converting single aggregation ~w to list format~n', [AggOp]),
            (   AggOp = count
            ->  OpList = [count(Result)]
            ;   AggOp = sum(AggVar)
            ->  OpList = [sum(AggVar, Result)]
            ;   AggOp = avg(AggVar)
            ->  OpList = [avg(AggVar, Result)]
            ;   AggOp = max(AggVar)
            ->  OpList = [max(AggVar, Result)]
            ;   AggOp = min(AggVar)
            ->  OpList = [min(AggVar, Result)]
            ;   format('ERROR: Unknown aggregation operation: ~w~n', [AggOp]),
                fail
            ),
            generate_multi_aggregation_code(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode)
        )
    ;   % Single field grouping: GroupField is a variable
        % Find the field name for group field variable
        find_field_for_var(GroupField, FieldMappings, GroupFieldName),

        % Check if AggOp is a list (multiple aggregations) or single operation
        (   is_list(AggOp)
        ->  % Multiple aggregations (Phase 9c-1 + HAVING support Phase 9c-2)
            format('  Multiple aggregations detected: ~w~n', [AggOp]),
            generate_multi_aggregation_code(GroupFieldName, FieldMappings, AggOp, HavingConstraints, DbFile, BucketStr, GoCode)
        % Single aggregation (Phase 9b - backward compatible, HAVING support added)
        ;   AggOp = count
        ->  generate_group_by_count_with_having(GroupFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = sum(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_sum_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = avg(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_avg_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = max(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_max_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = min(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_min_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   format('ERROR: Unknown group_by aggregation operation: ~w~n', [AggOp]),
            fail
        )
    ).

%% generate_multi_aggregation_code(+GroupField, +FieldMappings, +OpList, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY code for multiple aggregations in single query with HAVING support
%  GroupField can be a single field name or a list of field names for nested grouping
%  OpList is a list like [count(Count), avg(Age, AvgAge), max(Age, MaxAge)]
%  HavingConstraints are post-aggregation filters (null if none)
%
generate_multi_aggregation_code(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Check if GroupField is a list (nested grouping) or single field
    (   is_list(GroupField)
    ->  format('  Nested grouping detected: ~w~n', [GroupField]),
        generate_nested_group_multi_agg(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode)
    ;   % Single field grouping - original implementation
        generate_single_field_multi_agg(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode)
    ).

%% generate_single_field_multi_agg(+GroupField, +FieldMappings, +OpList, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Original implementation for single-field grouping
%
generate_single_field_multi_agg(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Parse operations to determine what's needed
    parse_multi_agg_operations(OpList, FieldMappings, AggInfo),

    % Generate struct fields based on operations
    generate_multi_agg_struct_fields(AggInfo, StructFields),

    % Generate field extractions and accumulation code
    generate_multi_agg_accumulation(AggInfo, AccumulationCode),

    % Generate output code for all metrics
    generate_multi_agg_output(GroupField, AggInfo, OutputCode),

    % Get struct initialization values
    get_struct_init_values(AggInfo, InitValues),

    % Get output calculations (e.g., avg = sum/count)
    get_output_calculations(AggInfo, OutputCalcs),

    % Generate HAVING filter code (Phase 9c-2)
    generate_having_filter_code(HavingConstraints, AggInfo, OpList, HavingFilterCode),

    % Combine into full Go code
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s with multiple aggregations
\ttype GroupStats struct {
~s
\t}
\tstats := make(map[string]*GroupStats)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group field
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\t// Initialize stats for this group if needed
\t\t\t\t\tif _, exists := stats[groupStr]; !exists {
\t\t\t\t\t\tstats[groupStr] = &GroupStats{~s}
\t\t\t\t\t}
~s
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, s := range stats {
~s
~s
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
~s
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
',  [DbFile, GroupField, StructFields, BucketStr, BucketStr, GroupField,
    InitValues, AccumulationCode, OutputCalcs, HavingFilterCode, GroupField, OutputCode]).

%% extract_group_field_names(+GroupFieldVars, +FieldMappings, -FieldNames)
%  Extract field names from a list of variables by looking them up in FieldMappings
%  GroupFieldVars: List of variables like [_1820, _1822]
%  FieldMappings: List of pairs like [state-_1820, city-_1822, name-_1892]
%  FieldNames: Extracted atom list like [state, city]
%
extract_group_field_names([], _, []) :- !.
extract_group_field_names([Var|Rest], FieldMappings, [FieldName|RestNames]) :-
    member(FieldName-MappedVar, FieldMappings),
    Var == MappedVar,  % Use == for variable identity check
    !,
    extract_group_field_names(Rest, FieldMappings, RestNames).
extract_group_field_names([Var|_], FieldMappings, _) :-
    % If we get here, variable wasn't found in mappings
    format('ERROR: Group field variable ~w not found in mappings ~w~n', [Var, FieldMappings]),
    fail.

%% generate_nested_group_multi_agg(+GroupFields, +FieldMappings, +OpList, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY code for nested grouping (multiple group fields)
%  GroupFields is a list like [State, City]
%
generate_nested_group_multi_agg(GroupFields, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Extract field names from variables
    extract_group_field_names(GroupFields, FieldMappings, FieldNames),

    % Parse operations
    parse_multi_agg_operations(OpList, FieldMappings, AggInfo),

    % Generate struct fields
    generate_multi_agg_struct_fields(AggInfo, StructFields),

    % Generate accumulation code
    generate_multi_agg_accumulation(AggInfo, AccumulationCode),

    % Get struct initialization
    get_struct_init_values(AggInfo, InitValues),

    % Get output calculations
    get_output_calculations(AggInfo, OutputCalcs),

    % Generate HAVING filter code
    generate_having_filter_code(HavingConstraints, AggInfo, OpList, HavingFilterCode),

    % Generate composite key extraction code (includes accumulation) - use field names
    generate_composite_key_extraction(FieldNames, AccumulationCode, KeyExtractionCode),

    % Generate composite key parsing and output - use field names
    generate_composite_key_output(FieldNames, AggInfo, KeyOutputCode),

    % Create group fields description for comment
    atomic_list_concat(FieldNames, ', ', GroupFieldsStr),

    % Combine into full Go code
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by [~s] with multiple aggregations
\ttype GroupStats struct {
~s
\t}
\tstats := make(map[string]*GroupStats)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

~s
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor groupKey, s := range stats {
~s
~s
~s
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupFieldsStr, StructFields, BucketStr, BucketStr,
    KeyExtractionCode, OutputCalcs, HavingFilterCode, KeyOutputCode]).

%% generate_composite_key_extraction(+GroupFields, -Code)
%  Generate Go code to extract multiple fields and create composite key
%
generate_composite_key_extraction(GroupFields, AccumulationCode, Code) :-
    length(GroupFields, NumFields),
    (   NumFields =:= 1
    ->  % Single field - simpler case
        [Field] = GroupFields,
        format(string(Code), '\t\t\t// Extract group field
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\t// Initialize stats for this group if needed
\t\t\t\t\tif _, exists := stats[groupStr]; !exists {
\t\t\t\t\t\tstats[groupStr] = &GroupStats{}
\t\t\t\t\t}
~s
\t\t\t\t}
\t\t\t}', [Field, AccumulationCode])
    ;   % Multiple fields - use composite key
        generate_field_extractions(GroupFields, Extractions),
        % Replace groupStr with groupKey in accumulation code for nested grouping
        atomic_list_concat(Split, 'groupStr', AccumulationCode),
        atomic_list_concat(Split, 'groupKey', AccumulationCodeFixed),
        format(string(Code), '\t\t\t// Extract group fields for composite key
\t\t\tkeyParts := make([]string, 0, ~d)
~s
\t\t\t// Check all fields were extracted
\t\t\tif len(keyParts) == ~d {
\t\t\t\tgroupKey := strings.Join(keyParts, "|")
\t\t\t\t// Initialize stats for this group if needed
\t\t\t\tif _, exists := stats[groupKey]; !exists {
\t\t\t\t\tstats[groupKey] = &GroupStats{}
\t\t\t\t}
~s
\t\t\t}', [NumFields, Extractions, NumFields, AccumulationCodeFixed])
    ).

%% generate_field_extractions(+Fields, -Code)
%  Generate extraction code for each field in the composite key
%
generate_field_extractions(Fields, Code) :-
    generate_field_extractions_impl(Fields, 1, CodeLines),
    atomic_list_concat(CodeLines, '\n', Code).

generate_field_extractions_impl([], _, []).
generate_field_extractions_impl([Field|Rest], Index, [ThisCode|RestCodes]) :-
    format(string(ThisCode), '\t\t\tif val~d, ok := data["~s"]; ok {
\t\t\t\tif str~d, ok := val~d.(string); ok {
\t\t\t\t\tkeyParts = append(keyParts, str~d)
\t\t\t\t}
\t\t\t}', [Index, Field, Index, Index, Index]),
    NextIndex is Index + 1,
    generate_field_extractions_impl(Rest, NextIndex, RestCodes).

%% generate_key_parts_list(+Fields, -Code)
%  Generate the keyParts list initialization
%
generate_key_parts_list(Fields, Code) :-
    length(Fields, Len),
    format(string(Code), 'make([]string, 0, ~d)', [Len]).

%% generate_composite_key_output(+GroupFields, +AggInfo, -Code)
%  Generate code to parse composite key and output all group fields
%
generate_composite_key_output([Field], AggInfo, Code) :- !,
    % Single field - simple output
    generate_multi_agg_output_fields(AggInfo, AggOutputCode),
    format(string(Code), '\t\tresult := map[string]interface{}{
\t\t\t"~s": groupKey,
~s
\t\t}', [Field, AggOutputCode]).

generate_composite_key_output(GroupFields, AggInfo, Code) :-
    % Multiple fields - parse composite key
    length(GroupFields, NumFields),
    generate_groupkey_field_parsing(GroupFields, 0, FieldParsing),
    generate_multi_agg_output_fields(AggInfo, AggOutputCode),
    (   AggOutputCode = ''
    ->  FullOutput = FieldParsing
    ;   atomic_list_concat([FieldParsing, ',\n', AggOutputCode], FullOutput)
    ),
    format(string(Code), '\t\t// Parse composite key
\t\tparts := strings.Split(groupKey, "|")
\t\tif len(parts) < ~d {
\t\t\tcontinue  // Skip malformed keys
\t\t}
\t\tresult := map[string]interface{}{
~s
\t\t}', [NumFields, FullOutput]).

%% generate_groupkey_field_parsing(+Fields, +Index, -Code)
%  Generate parsing code for composite key fields from group_by
%
generate_groupkey_field_parsing([], _, '').
generate_groupkey_field_parsing([Field|Rest], Index, Code) :-
    format(string(ThisCode), '\t\t\t"~s": parts[~d]', [Field, Index]),
    NextIndex is Index + 1,
    generate_groupkey_field_parsing(Rest, NextIndex, RestCode),
    (   RestCode = ''
    ->  Code = ThisCode
    ;   atomic_list_concat([ThisCode, ',\n', RestCode], Code)
    ).

%% generate_multi_agg_output_fields(+AggInfo, -Code)
%  Generate output field code for aggregations (without map wrapper)
%
generate_multi_agg_output_fields(AggInfo, Code) :-
    findall(FieldCode, (
        member(agg(OpType, _, _, OutputName), AggInfo),
        operation_output_field(OpType, OutputName, FieldCode)
    ), FieldCodes),
    atomic_list_concat(FieldCodes, ',\n', Code).

%% operation_output_field(+OpType, +Name, -Code)
%  Generate single output field code
%
operation_output_field(count, count, '\t\t\t"count": s.count').
operation_output_field(sum, sum, '\t\t\t"sum": s.sum').
operation_output_field(avg, avg, '\t\t\t"avg": avg').
operation_output_field(max, max, '\t\t\t"max": s.maxValue').
operation_output_field(min, min, '\t\t\t"min": s.minValue').

%% parse_multi_agg_operations(+OpList, +FieldMappings, -AggInfo)
%  Parse list of operations into structured info
%  AggInfo is a list of operation specs like:
%    [op(count, null, CountVar), op(avg, age, AvgVar), op(max, age, MaxVar)]
%
parse_multi_agg_operations([], _FieldMappings, []).
parse_multi_agg_operations([Op|Rest], FieldMappings, [Info|RestInfo]) :-
    parse_single_agg_op(Op, FieldMappings, Info),
    parse_multi_agg_operations(Rest, FieldMappings, RestInfo).

%% parse_single_agg_op(+Op, +FieldMappings, -Info)
%  Parse a single operation specification
%
parse_single_agg_op(count, _FieldMappings, agg(count, null, null, count)) :- !.
parse_single_agg_op(count(_ResultVar), _FieldMappings, agg(count, null, null, count)) :- !.
parse_single_agg_op(sum(AggVar), FieldMappings, agg(sum, FieldName, null, sum)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(sum(AggVar, _ResultVar), FieldMappings, agg(sum, FieldName, null, sum)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(avg(AggVar), FieldMappings, agg(avg, FieldName, null, avg)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(avg(AggVar, _ResultVar), FieldMappings, agg(avg, FieldName, null, avg)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(max(AggVar), FieldMappings, agg(max, FieldName, null, max)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(max(AggVar, _ResultVar), FieldMappings, agg(max, FieldName, null, max)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(min(AggVar), FieldMappings, agg(min, FieldName, null, min)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(min(AggVar, _ResultVar), FieldMappings, agg(min, FieldName, null, min)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).

%% generate_multi_agg_struct_fields(+AggInfo, -StructFields)
%  Generate Go struct field declarations based on operations
%
generate_multi_agg_struct_fields(AggInfo, StructFields) :-
    collect_needed_fields(AggInfo, NeededFields),
    format_struct_fields(NeededFields, StructFields).

%% collect_needed_fields(+AggInfo, -NeededFields)
%  Determine which struct fields are needed
%
collect_needed_fields(AggInfo, NeededFields) :-
    findall(Field, (
        member(agg(OpType, _FieldName, _, _), AggInfo),
        needed_struct_field(OpType, Field)
    ), AllFields),
    sort(AllFields, NeededFields).

%% needed_struct_field(+OpType, -Field)
%  Map operation type to required struct fields
%
needed_struct_field(count, count).
needed_struct_field(sum, sum).
needed_struct_field(sum, count).  % sum also needs count for proper tracking
needed_struct_field(avg, sum).
needed_struct_field(avg, count).
needed_struct_field(max, maxValue).
needed_struct_field(max, maxFirst).
needed_struct_field(min, minValue).
needed_struct_field(min, minFirst).

%% format_struct_fields(+NeededFields, -StructFields)
%  Format struct fields as Go code
%
format_struct_fields(NeededFields, StructFields) :-
    findall(FieldLine, (
        member(Field, NeededFields),
        go_struct_field_line(Field, FieldLine)
    ), Lines),
    atomic_list_concat(Lines, '\n', StructFields).

%% go_struct_field_line(+Field, -Line)
%  Generate Go struct field declaration
%
go_struct_field_line(count, '\t\tcount    int').
go_struct_field_line(sum, '\t\tsum      float64').
go_struct_field_line(maxValue, '\t\tmaxValue float64').
go_struct_field_line(maxFirst, '\t\tmaxFirst bool').
go_struct_field_line(minValue, '\t\tminValue float64').
go_struct_field_line(minFirst, '\t\tminFirst bool').

%% get_struct_init_values(+AggInfo, -InitValues)
%  Generate initialization values for struct
%
get_struct_init_values(AggInfo, 'maxFirst: true, minFirst: true') :-
    member(agg(max, _, _, _), AggInfo),
    member(agg(min, _, _, _), AggInfo),
    !.
get_struct_init_values(AggInfo, 'maxFirst: true') :-
    member(agg(max, _, _, _), AggInfo),
    !.
get_struct_init_values(AggInfo, 'minFirst: true') :-
    member(agg(min, _, _, _), AggInfo),
    !.
get_struct_init_values(_, '').

%% generate_multi_agg_accumulation(+AggInfo, -AccumulationCode)
%  Generate accumulation code for all operations
%  Smart handling: if 'count' operation is present, it owns the count field
%
generate_multi_agg_accumulation(AggInfo, AccumulationCode) :-
    % Check if count operation is present
    (   member(agg(count, _, _, _), AggInfo)
    ->  HasCount = true
    ;   HasCount = false
    ),
    % Generate code for each operation
    findall(Code, (
        member(AggSpec, AggInfo),
        generate_single_accumulation(AggSpec, HasCount, Code)
    ), CodeLines),
    atomic_list_concat(CodeLines, '', AccumulationCode).

%% generate_single_accumulation(+AggSpec, +HasCount, -Code)
%  Generate accumulation code for one operation
%  HasCount indicates if a separate count operation exists
%
generate_single_accumulation(agg(count, null, _, _), _, '\t\t\t\t\t// Count operation
\t\t\t\t\tstats[groupStr].count++
') :- !.

generate_single_accumulation(agg(sum, FieldName, _, _), HasCount, Code) :- !,
    % Only increment count if no separate count operation exists
    (   HasCount = true
    ->  format(string(Code), '\t\t\t\t\t// Sum ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ;   format(string(Code), '\t\t\t\t\t// Sum ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t\tstats[groupStr].count++
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ).

generate_single_accumulation(agg(avg, FieldName, _, _), HasCount, Code) :- !,
    % Only increment count if no separate count operation exists
    (   HasCount = true
    ->  format(string(Code), '\t\t\t\t\t// Average ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ;   format(string(Code), '\t\t\t\t\t// Average ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t\tstats[groupStr].count++
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ).

generate_single_accumulation(agg(max, FieldName, _, _), _, Code) :- !,
    format(string(Code), '\t\t\t\t\t// Max ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif stats[groupStr].maxFirst || valueFloat > stats[groupStr].maxValue {
\t\t\t\t\t\t\t\tstats[groupStr].maxValue = valueFloat
\t\t\t\t\t\t\t\tstats[groupStr].maxFirst = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName]).

generate_single_accumulation(agg(min, FieldName, _, _), _, Code) :- !,
    format(string(Code), '\t\t\t\t\t// Min ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif stats[groupStr].minFirst || valueFloat < stats[groupStr].minValue {
\t\t\t\t\t\t\t\tstats[groupStr].minValue = valueFloat
\t\t\t\t\t\t\t\tstats[groupStr].minFirst = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName]).

%% get_output_calculations(+AggInfo, -OutputCalcs)
%  Generate pre-output calculations (e.g., avg = sum / count)
%
get_output_calculations(AggInfo, '\t\tavg := 0.0
\t\tif s.count > 0 {
\t\t\tavg = s.sum / float64(s.count)
\t\t}') :-
    member(agg(avg, _, _, _), AggInfo),
    !.
get_output_calculations(_, '').

%% generate_multi_agg_output(+GroupField, +AggInfo, -OutputCode)
%  Generate output field mappings for result JSON
%
generate_multi_agg_output(_GroupField, AggInfo, OutputCode) :-
    findall(Line, (
        member(AggSpec, AggInfo),
        generate_output_field(AggSpec, Line)
    ), Lines),
    atomic_list_concat(Lines, ',\n', OutputCode).

%% generate_output_field(+AggSpec, -Line)
%  Generate single output field mapping
%
generate_output_field(agg(count, _, _, _), '\t\t\t"count": s.count') :- !.
generate_output_field(agg(sum, _, _, _), '\t\t\t"sum": s.sum') :- !.
generate_output_field(agg(avg, _, _, _), '\t\t\t"avg": avg') :- !.
generate_output_field(agg(max, _, _, _), '\t\t\t"max": s.maxValue') :- !.
generate_output_field(agg(min, _, _, _), '\t\t\t"min": s.minValue') :- !.

%% ============================================
%% HAVING CLAUSE SUPPORT (Phase 9c-2)
%% ============================================

%% generate_having_filter_code(+HavingConstraints, +AggInfo, +OpList, -FilterCode)
%  Generate HAVING filter code for output loop
%  Parses constraints and generates Go filter code with continue statements
%
generate_having_filter_code(null, _, _, '') :- !.
generate_having_filter_code(Constraints, AggInfo, OpList, FilterCode) :-
    % Parse constraints into list
    parse_constraints_to_list(Constraints, ConstraintList),
    % Generate filter code for each constraint
    findall(Code, (
        member(Constraint, ConstraintList),
        generate_single_having_filter(Constraint, AggInfo, OpList, Code)
    ), CodeLines),
    atomic_list_concat(CodeLines, '', FilterCode).

%% parse_constraints_to_list(+Constraints, -ConstraintList)
%  Convert conjunction of constraints into a list
%
parse_constraints_to_list((C1, C2), List) :- !,
    parse_constraints_to_list(C1, L1),
    parse_constraints_to_list(C2, L2),
    append(L1, L2, List).
parse_constraints_to_list(C, [C]).

%% generate_single_having_filter(+Constraint, +AggInfo, +OpList, -Code)
%  Generate filter code for a single constraint
%  Supports: >, <, >=, <=, =, =\=
%
generate_single_having_filter(Constraint, AggInfo, OpList, Code) :-
    % Extract operator and operands
    parse_constraint_operator(Constraint, Var, Op, Value),
    % Map variable to Go expression
    map_variable_to_go_expr(Var, AggInfo, OpList, GoExpr),
    % Generate Go comparison code
    format(string(Code), '\t\t// HAVING filter: ~w ~w ~w
\t\tif !(~s ~s ~w) {
\t\t\tcontinue
\t\t}
', [Var, Op, Value, GoExpr, Op, Value]).

%% parse_constraint_operator(+Constraint, -Var, -Op, -Value)
%  Parse constraint to extract variable, operator, and value
%
parse_constraint_operator(Var > Value, Var, '>', Value) :- !.
parse_constraint_operator(Var < Value, Var, '<', Value) :- !.
parse_constraint_operator(Var >= Value, Var, '>=', Value) :- !.
parse_constraint_operator(Var =< Value, Var, '=<', Value) :- !.
parse_constraint_operator(Var = Value, Var, '==', Value) :- !.
parse_constraint_operator(Var =\= Value, Var, '!=', Value) :- !.

%% map_variable_to_go_expr(+Var, +AggInfo, +OpList, -GoExpr)
%  Map Prolog variable to Go expression (e.g., Count -> s.count, Avg -> avg)
%
map_variable_to_go_expr(Var, _AggInfo, OpList, GoExpr) :-
    % Find which operation produces this variable
    member(Op, OpList),
    variable_from_operation(Op, Var, OpType),
    !,
    % Map to Go expression
    operation_to_go_expr(OpType, GoExpr).

%% variable_from_operation(+Operation, ?Var, -OpType)
%  Extract result variable and operation type from operation spec
%  Uses == for variable identity checking to avoid incorrect unification
%
variable_from_operation(count(Var), RequestedVar, count) :-
    Var == RequestedVar,  % Use == to check if they're the same variable
    !.
variable_from_operation(count, _, count) :- !.
variable_from_operation(sum(_, Var), RequestedVar, sum) :-
    Var == RequestedVar, !.
variable_from_operation(avg(_, Var), RequestedVar, avg) :-
    Var == RequestedVar, !.
variable_from_operation(max(_, Var), RequestedVar, max) :-
    Var == RequestedVar, !.
variable_from_operation(min(_, Var), RequestedVar, min) :-
    Var == RequestedVar, !.

%% operation_to_go_expr(+OpType, -GoExpr)
%  Map operation type to Go expression in output loop
%
operation_to_go_expr(count, 's.count').
operation_to_go_expr(sum, 's.sum').
operation_to_go_expr(avg, 'avg').
operation_to_go_expr(max, 's.maxValue').
operation_to_go_expr(min, 's.minValue').

%% ============================================
%% SINGLE AGGREGATION WRAPPERS WITH HAVING
%% ============================================

%% Wrapper predicates with HAVING support (Phase 9c-2)
%% For single aggregations, convert to multi-agg format and use multi-agg code generator

generate_group_by_count_with_having(GroupField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single count to multi-agg format: [count(Result)]
    OpList = [count(Result)],
    % Use empty field mappings for count (doesn't need any field)
    FieldMappings = [],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_sum_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single sum to multi-agg format: [sum(AggField, Result)]
    % Create a fake variable for the field (we'll use AggField directly)
    OpList = [sum(AggField, Result)],
    % Create field mappings with the aggregation field
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_avg_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single avg to multi-agg format: [avg(AggField, Result)]
    OpList = [avg(AggField, Result)],
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_max_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single max to multi-agg format: [max(AggField, Result)]
    OpList = [max(AggField, Result)],
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_min_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single min to multi-agg format: [min(AggField, Result)]
    OpList = [min(AggField, Result)],
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

%% ============================================
%% ORIGINAL SINGLE AGGREGATION GENERATORS (Phase 9b)
%% ============================================

%% generate_group_by_count(+GroupField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY count code
%
generate_group_by_count(GroupField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and count
\tcounts := make(map[string]int)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group field
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tcounts[groupStr]++
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, count := range counts {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"count": count,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, BucketStr, BucketStr, GroupField, GroupField]).

%% generate_group_by_sum(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY sum code
%
generate_group_by_sum(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and sum ~s
\tsums := make(map[string]float64)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tsums[groupStr] += valueFloat
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, sum := range sums {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"sum": sum,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

%% generate_group_by_avg(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY average code
%
generate_group_by_avg(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and average ~s
\ttype GroupStats struct {
\t\tsum   float64
\t\tcount int
\t}
\tstats := make(map[string]*GroupStats)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif _, exists := stats[groupStr]; !exists {
\t\t\t\t\t\t\t\tstats[groupStr] = &GroupStats{}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t\tstats[groupStr].count++
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, s := range stats {
\t\tavg := 0.0
\t\tif s.count > 0 {
\t\t\tavg = s.sum / float64(s.count)
\t\t}
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"avg": avg,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

%% generate_group_by_max(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY max code
%
generate_group_by_max(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and find max ~s
\ttype GroupMax struct {
\t\tmaxValue float64
\t\tfirst    bool
\t}
\tmaxes := make(map[string]*GroupMax)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif _, exists := maxes[groupStr]; !exists {
\t\t\t\t\t\t\t\tmaxes[groupStr] = &GroupMax{first: true}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t\tif maxes[groupStr].first || valueFloat > maxes[groupStr].maxValue {
\t\t\t\t\t\t\t\tmaxes[groupStr].maxValue = valueFloat
\t\t\t\t\t\t\t\tmaxes[groupStr].first = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, m := range maxes {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"max": m.maxValue,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

%% generate_group_by_min(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY min code
%
generate_group_by_min(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and find min ~s
\ttype GroupMin struct {
\t\tminValue float64
\t\tfirst    bool
\t}
\tmins := make(map[string]*GroupMin)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif _, exists := mins[groupStr]; !exists {
\t\t\t\t\t\t\t\tmins[groupStr] = &GroupMin{first: true}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t\tif mins[groupStr].first || valueFloat < mins[groupStr].minValue {
\t\t\t\t\t\t\t\tmins[groupStr].minValue = valueFloat
\t\t\t\t\t\t\t\tmins[groupStr].first = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, m := range mins {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"min": m.minValue,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

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

        % Check for schema option OR database backend (both require typed compilation)
        SingleHead =.. [_|HeadArgs],
        (   option(json_schema(SchemaName), Options)
        ->  % Typed compilation with schema validation
            format('  Using schema: ~w~n', [SchemaName]),
            compile_json_to_go_typed(HeadArgs, FieldMappings, SchemaName, FieldDelim, Unique, CoreBody)
        ;   option(db_backend(bbolt), Options)
        ->  % Typed compilation without schema (for database writes)
            format('  Database mode: using typed compilation~n'),
            compile_json_to_go_typed_noschema(HeadArgs, FieldMappings, FieldDelim, Unique, CoreBody)
        ;   % Untyped compilation (current behavior)
            compile_json_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, CoreBody)
        ),

        % Check if database backend is specified
        (   option(db_backend(bbolt), Options)
        ->  % Wrap core body with database operations
            format('  Database: bbolt~n'),
            wrap_with_database(CoreBody, FieldMappings, Pred, Options, ScriptBody, KeyImports),
            NeedsDatabase = true
        ;   ScriptBody = CoreBody,
            NeedsDatabase = false,
            KeyImports = []
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

        % Add key expression imports if any (from main)
        (   KeyImports \= []
        ->  maplist(atom_string, KeyImports, KeyImportStrs),
            append(BaseImports, KeyImportStrs, ImportsWithKeys)
        ;   ImportsWithKeys = BaseImports
        ),
        
        % Add strings import if needed (from feature branch)
        (   option(json_schema(SchemaName), Options), schema_needs_strings(SchemaName)
        ->  append(ImportsWithKeys, ["strings"], AllImportsList)
        ;   AllImportsList = ImportsWithKeys
        ),
        
        % Format all imports
        sort(AllImportsList, UniqueImports), % Deduplicate
        maplist(format_import, UniqueImports, ImportLines),
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

%% wrap_with_database(+CoreBody, +FieldMappings, +Pred, +Options, -WrappedBody, -KeyImports)
%  Wrap core extraction code with database operations
%  Returns additional imports needed for key expressions
%
wrap_with_database(CoreBody, FieldMappings, Pred, Options, WrappedBody, KeyImports) :-
    % Get database options
    option(db_file(DbFile), Options, 'data.db'),
    option(db_bucket(BucketName), Options, Pred),
    atom_string(BucketName, BucketStr),

    % Normalize key strategy options (backward compatibility)
    normalize_key_strategy(Options, NormalizedOptions),

    % Determine key strategy (with default fallback)
    (   option(db_key_strategy(KeyStrategy), NormalizedOptions)
    ->  true
    ;   % Default: use first field
        (   FieldMappings = [FirstField-_|_]
        ->  KeyStrategy = field(FirstField)
        ;   FieldMappings = [nested(Path, _)|_],
            last(Path, FirstField)
        ->  KeyStrategy = field(FirstField)
        ;   format('ERROR: No fields found in mappings: ~w~n', [FieldMappings]),
            fail
        )
    ),

    % Compile key expression to Go code
    compile_key_expression(KeyStrategy, FieldMappings, NormalizedOptions, KeyCode, KeyImports),

    % Determine which fields are used by the key expression
    extract_used_field_positions(KeyStrategy, FieldMappings, UsedFieldPositions),

    % Generate blank assignments for unused fields (to avoid Go unused variable errors)
    findall(BlankAssignment,
        (nth1(Pos, FieldMappings, _),
         \+ memberchk(Pos, UsedFieldPositions),
         format(string(BlankAssignment), '\t\t_ = field~w  // Unused in key\n', [Pos])),
        BlankAssignments),
    atomic_list_concat(BlankAssignments, '', UnusedFieldCode),

    % Create storage code block with compiled key
    format(string(StorageCode), '~s\t\t// Store in database
\t\terr = db.Update(func(tx *bolt.Tx) error {
\t\t\tbucket := tx.Bucket([]byte("~s"))
\t\t\t
\t\t\t// Generate key using strategy
\t\t\tkeyStr := ~s
\t\t\tkey := []byte(keyStr)
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
\t\trecordCount++', [UnusedFieldCode, BucketStr, KeyCode]),

    % Remove output block and inject storage code
    split_string(CoreBody, "\n", "", Lines),
    % Pass -1 to keep all fields (key expression will determine which it needs)
    filter_and_replace_lines(Lines, StorageCode, -1, FilteredLines),
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
    % If KeyPos is -1, keep all fields (don't replace anything)
    KeyPos \= -1,
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
        % Generate flat field extraction (not nested)
        atom_string(Field, FieldStr),
        format(string(ExtractCode), '
\t\t~w, ok~w := data["~s"]
\t\tif !ok~w {
\t\t\tcontinue
\t\t}', [GoVar, VIdx, FieldStr, VIdx]),
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



%% compile_json_to_go_typed_noschema(+HeadArgs, +FieldMappings, +FieldDelim, +Unique, -GoCode)
%  Generate Go code for JSON input mode with fieldN variables but no type validation
%  Used for database writes without schema
%
compile_json_to_go_typed_noschema(HeadArgs, FieldMappings, FieldDelim, Unique, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate untyped field extraction code with fieldN variable names
    findall(ExtractLine,
        (   nth1(Pos, FieldMappings, Mapping),
            format(atom(VarName), 'field~w', [Pos]),
            % Generate extraction for flat or nested fields
            (   Mapping = Field-_Var
            ->  % Flat field - untyped extraction
                atom_string(Field, FieldStr),
                format(atom(ExtractLine), '\t\t~w, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}',
                    [VarName, VarName, FieldStr, VarName])
            ;   Mapping = nested(Path, _Var)
            ->  % Nested field - untyped extraction
                generate_nested_field_extraction(Path, VarName, ExtractLine)
            )
        ),
        ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode),

    % Generate output expression (same as typed)
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
%% generate_aggregation_code(+Op, +AggField, +ResultVar, +FieldMappings, +Options, -GoCode)
%  Generate Go code for aggregation
generate_aggregation_code(Op, AggField, ResultVar, FieldMappings, Options, GoCode) :-
    % Generate field extraction code
    (   AggField == none
    ->  ExtractCode = ''
    ;   % Find the field mapping for the aggregation variable
        (   member(Field-Var, FieldMappings), Var == AggField
        ->  true
        ;   format('ERROR: Aggregation variable ~w not found in field mappings~n', [AggField]),
            fail
        ),
        % Generate extraction logic
        format(atom(ExtractCode), '
		valRaw, ok := data["~w"]
		if !ok { continue }
		val, ok := valRaw.(float64)
		if !ok { continue }', [Field])
    ),

    % Generate specific aggregation logic
    (   Op = count -> generate_count_aggregation(ExtractCode, BodyCode)
    ;   Op = sum   -> generate_sum_aggregation(ExtractCode, BodyCode)
    ;   Op = avg   -> generate_avg_aggregation(ExtractCode, BodyCode)
    ;   Op = max   -> generate_max_aggregation(ExtractCode, BodyCode)
    ;   Op = min   -> generate_min_aggregation(ExtractCode, BodyCode)
    ;   format('ERROR: Unknown aggregation operation: ~w~n', [Op]), fail
    ),
    
    BodyCode = code(Init, Loop, Output),
    
    % Wrap in loop
    format(string(GoCode), '
	scanner := bufio.NewScanner(os.Stdin)
	~s
	
	for scanner.Scan() {
		line := scanner.Bytes()
		var data map[string]interface{}
		if err := json.Unmarshal(line, &data); err != nil {
			continue
		}
		~s
	}
	
	~s
', [Init, Loop, Output]).

%% Aggregation Generators

generate_count_aggregation(_, code(
    'count := 0',
    'count++',
    'fmt.Println(count)'
)).

generate_sum_aggregation(ExtractCode, code(
    'sum := 0.0',
    LoopCode,
    'fmt.Println(sum)'
)) :-
    format(string(LoopCode), '~s
		sum += val', [ExtractCode]).

generate_avg_aggregation(ExtractCode, code(
    'sum := 0.0\n\tcount := 0',
    LoopCode,
    'if count > 0 {\n\t\tfmt.Println(sum / float64(count))\n\t} else {\n\t\tfmt.Println(0)\n\t}'
)) :-
    format(string(LoopCode), '~s
		sum += val
		count++', [ExtractCode]).

generate_max_aggregation(ExtractCode, code(
    'var maxVal float64\n\tfirst := true',
    LoopCode,
    'if !first {\n\t\tfmt.Println(maxVal)\n\t}'
)) :-
    format(string(LoopCode), '~s
		if first || val > maxVal {
			maxVal = val
			first = false
		}', [ExtractCode]).

generate_min_aggregation(ExtractCode, code(
    'var minVal float64\n\tfirst := true',
    LoopCode,
    'if !first {\n\t\tfmt.Println(minVal)\n\t}'
)) :-
    format(string(LoopCode), '~s
		if first || val < minVal {
			minVal = val
			first = false
		}', [ExtractCode]).

%% generate_group_by_code_jsonl(+GroupField, +FieldMappings, +AggOp, +Result, +HavingConstraints, +Options, -GoCode)
%  Generate Go code for JSONL stream grouped aggregation
generate_group_by_code_jsonl(GroupField, FieldMappings, AggOp, _Result, HavingConstraints, _Options, GoCode) :-
    % 1. Determine key type (simplification: assume string for now)
    KeyType = "string",
    
    % 2. Normalize AggOp to list
    (   is_list(AggOp)
    ->  AggOpList = AggOp
    ;   AggOpList = [AggOp]
    ),

    % 3. Generate Code using Multi-Agg logic
    generate_group_by_code_jsonl_multi(GroupField, FieldMappings, AggOpList, HavingConstraints, KeyType, GoCode).

%% find_field_for_var(+Var, +FieldMappings, -FieldName)
%  Find field name for a variable in field mappings
find_field_for_var(Var, [FieldName-MappedVar|_], FieldName) :-
    Var == MappedVar, !.
find_field_for_var(Var, [_|Rest], FieldName) :-
    find_field_for_var(Var, Rest, FieldName).

%% format_import(+Import, -Line)
%  Format import string for Go

%% generate_group_by_code_jsonl_multi(+GroupField, +FieldMappings, +AggOpList, +HavingConstraints, +KeyType, -GoCode)
%  Generate Go code for multiple aggregations on JSONL stream
generate_group_by_code_jsonl_multi(GroupField, FieldMappings, AggOpList, HavingConstraints, KeyType, GoCode) :-
    % 1. Parse operations
    parse_multi_agg_operations(AggOpList, FieldMappings, AggInfo),
    
    % 2. Generate Struct Definition
    generate_multi_agg_struct_fields(AggInfo, StructFields),
    format(string(StateStructDef), 'struct {
~w
    }', [StructFields]),
    StateStruct = "AggState",
    
    % 3. Generate Update Logic
    generate_multi_agg_jsonl_updates(AggInfo, UpdateLogic),
    
    % 4. Generate Output/Having Logic
    generate_multi_agg_jsonl_output(AggInfo, HavingConstraints, OutputLogic),
    
    % 5. Key Extraction
    find_field_for_var(GroupField, FieldMappings, GroupFieldName),
    format(string(KeyExtraction), '
        keyRaw, ok := data["~w"]
        if !ok { continue }
        key := fmt.Sprintf("%v", keyRaw)', [GroupFieldName]),

    % 6. Generate Complete Code
    format(string(GoCode), '
    type AggState ~w
    
    scanner := bufio.NewScanner(os.Stdin)
    results := make(map[~w]*AggState)
    
    for scanner.Scan() {
        var data map[string]interface{}
        if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
            continue
        }
        
        ~w
        
        state, exists := results[key]
        if !exists {
            state = &AggState{}
            results[key] = state
        }
        
        ~w
    }
    
    if err := scanner.Err(); err != nil {
        fmt.Fprintln(os.Stderr, "reading standard input:", err)
    }
    
    for key, s := range results {
        ~w
    }
', [StateStructDef, KeyType, KeyExtraction, UpdateLogic, OutputLogic]).

%% generate_multi_agg_jsonl_updates(+AggInfo, -Code)
%  Generate update statements for each aggregation in struct
generate_multi_agg_jsonl_updates(AggInfo, Code) :-
    findall(UpdateStr, (
        member(agg(OpType, FieldName, _Var, _OutName), AggInfo),
        jsonl_op_update(OpType, FieldName, UpdateStr)
    ), UpdateStrs),
    atomic_list_concat(UpdateStrs, '\n        ', Code).

jsonl_op_update(count, _Field, 'state.count++').
jsonl_op_update(sum, Field, Code) :-
    format(string(Code), 'if val, ok := data["~w"].(float64); ok { state.sum += val }', [Field]).
% Extensible for avg, max, min...

%% generate_multi_agg_jsonl_output(+AggInfo, +HavingConstraints, -Code)
%  Generate output printing code, with optional HAVING filtering
generate_multi_agg_jsonl_output(AggInfo, HavingConstraints, Code) :-
    % Generate print args
    findall(Arg, (
        member(agg(OpType, _, _, _), AggInfo),
        jsonl_op_print_arg(OpType, Arg)
    ), ArgsList),
    atomic_list_concat(["key"|ArgsList], ', ', PrintArgs),
    
    % Generate format string
    findall(Fmt, (
        member(agg(OpType, _, _, _), AggInfo),
        jsonl_op_fmt(OpType, Fmt)
    ), Fmts),
    atomic_list_concat(["%v"|Fmts], ': ', FmtStr),
    
    format(string(PrintStmt), 'fmt.Printf("~w\\n", ~w)', [FmtStr, PrintArgs]),
    
    % Handle HAVING
    (   HavingConstraints \== null, HavingConstraints \== []
    ->  % Let's implement basic "SUM > Val" support via manual check for now.
        generate_having_check(HavingConstraints, AggInfo, CheckCode),
        format(string(Code), '
        if ~w {
            ~w
        }', [CheckCode, PrintStmt])
    ;   Code = PrintStmt
    ).

jsonl_op_print_arg(count, 's.count').
jsonl_op_print_arg(sum, 's.sum').

jsonl_op_fmt(count, '%d').
jsonl_op_fmt(sum, '%g').
% Add others as needed

%% generate_having_check(+Constraints, +AggInfo, -GoExpr)
%  Generate Go boolean expression for HAVING clause
generate_having_check((A > B), AggInfo, Code) :-
    resolve_having_var(A, AggInfo, ACode),
    resolve_having_var(B, AggInfo, BCode),
    format(string(Code), '~w > ~w', [ACode, BCode]).
% Add other operators as needed

resolve_having_var(Var, AggInfo, Code) :-
    var(Var),
    member(agg(OpType, _, Var, _), AggInfo),
    jsonl_op_print_arg(OpType, CodeStr),
    atom_string(Code, CodeStr).
resolve_having_var(Num, _, Num) :- number(Num).

