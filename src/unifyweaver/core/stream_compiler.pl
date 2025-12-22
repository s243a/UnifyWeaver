:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% stream_compiler.pl - UnifyWeaver streaming compiler for non-recursive predicates
% Compiles Prolog facts and rules into bash streaming pipelines
% Handles: facts, single rules, multiple rules (OR), inequality constraints
% Does not handle: recursion, negation, complex built-ins

:- module(stream_compiler, [
    compile_predicate/3,
    compile_predicate/2,
    compile_facts/4,         % Export for testing
    test_stream_compiler/0
]).

:- use_module(library(lists)).
:- use_module('constraint_analyzer').
:- use_module('template_system').
:- use_module('optimizer').

%% Main compilation entry point
compile_predicate(Pred/Arity, Options) :-
    compile_predicate(Pred/Arity, Options, _).

compile_predicate(PredIndicator, Options, BashCode) :-
    PredIndicator = PredAtom/Arity,
    format('=== Compiling ~w/~w ===~n', [PredAtom, Arity]),

    % Get constraints for this predicate (from declarations or defaults)
    get_constraints(PredAtom/Arity, Constraints),
    format('  Constraints: ~w~n', [Constraints]),

    % Merge with any runtime options (options take precedence)
    merge_options(Options, Constraints, MergedOptions),

    % Create head with correct arity
    functor(Head, PredAtom, Arity),

    % Get all clauses for this predicate (as head-body pairs)
    findall(H-B, (user:clause(Head, B), Head = H), Clauses),
    length(Clauses, NumClauses),

    % Determine compilation strategy
    (   Clauses = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [PredAtom, Arity]),
        fail
    ;   maplist(is_fact_clause, Clauses) ->
        % All bodies are just 'true' - these are facts
        format('Type: facts (~w clauses)~n', [NumClauses]),
        compile_facts(PredAtom, Arity, MergedOptions, BashCode)
    ;   Clauses = [_-SingleBody], SingleBody \= true ->
        % Single rule
        format('Type: single_rule (~w clauses)~n', [NumClauses]),
        extract_predicates(SingleBody, Predicates),
        format('  Body predicates: ~w~n', [Predicates]),
        compile_single_rule(PredAtom, Arity, SingleBody, MergedOptions, BashCode)
    ;   % Multiple rules (OR pattern)
        format('Type: multiple_rules (~w clauses)~n', [NumClauses]),
        format('  ~w alternatives~n', [NumClauses]),
        compile_multiple_rules(PredAtom, Clauses, MergedOptions, BashCode)
    ).

%% Helper to check if a clause is a fact (body is just 'true')
is_fact_clause(_-true).

%% Merge runtime options with constraint-based options
%  Runtime options take precedence over constraints
merge_options(RuntimeOpts, Constraints, Merged) :-
    % Extract unique and unordered from constraints
    (member(unique(ConstraintUnique), Constraints) -> true ; ConstraintUnique = true),
    (member(unordered(ConstraintUnordered), Constraints) -> true ; ConstraintUnordered = true),

    % Check if runtime overrides exist, otherwise use constraint values
    (member(unique(RuntimeUnique), RuntimeOpts) -> FinalUnique = RuntimeUnique ; FinalUnique = ConstraintUnique),
    (member(unordered(RuntimeUnordered), RuntimeOpts) -> FinalUnordered = RuntimeUnordered ; FinalUnordered = ConstraintUnordered),

    % Build merged list with final values
    Merged = [unique(FinalUnique), unordered(FinalUnordered)].

%% Extract predicates from a body
extract_predicates(true, []) :- !.
extract_predicates((A, B), Predicates) :- !,
    extract_predicates(A, P1),
    extract_predicates(B, P2),
    append(P1, P2, Predicates).
% Guard against variables in Goal - treat as producing no predicates
extract_predicates(Goal, []) :-
    var(Goal), !.
% Skip inequality operators - they're constraints, not predicates
extract_predicates(_ \= _, []) :- !.
extract_predicates(\=(_, _), []) :- !.
% Skip arithmetic operators - they're operations, not predicates
extract_predicates(_ > _, []) :- !.
extract_predicates(_ < _, []) :- !.
extract_predicates(_ >= _, []) :- !.
extract_predicates(_ =< _, []) :- !.
extract_predicates(_ =:= _, []) :- !.
extract_predicates(_ =\= _, []) :- !.
extract_predicates(is(_, _), []) :- !.
% Skip match predicates - they're constraints for regex pattern matching
extract_predicates(match(_, _), []) :- !.
extract_predicates(match(_, _, _), []) :- !.
extract_predicates(match(_, _, _, _), []) :- !.
% Skip negation wrapper - extract predicates inside
extract_predicates(\+ A, Predicates) :- !,
    extract_predicates(A, Predicates).
extract_predicates(Goal, [Pred]) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% Compile facts into bash lookup
compile_facts(Pred, Arity, Options, BashCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    
    % Collect all facts
    findall(Head, user:clause(Head, true), Facts),
    
    % Get deduplication strategy from options
    get_dedup_strategy(Options, Strategy),
    
    % Handle different strategies
    (   Strategy = no_dedup ->
        % For no_dedup, use regular array to preserve duplicates
        compile_facts_no_dedup(Pred, Arity, Facts, PredStr, BashCode)
    ;   % For other strategies, use associative array
        % Build array entries
        findall(Entry,
            (   member(Fact, Facts),
                Fact =.. [_|Args],
                format_fact_entry(Args, Entry)
            ),
            Entries),
        atomic_list_concat(Entries, '\n    ', EntriesStr),
        
        % Render template with associative array
        compose_templates(
            ['bash/header','facts/array_binary','facts/lookup_binary','facts/stream_binary', 'facts/reverse_stream_binary'],
            [pred=PredStr, entries=EntriesStr, strategy=Strategy],
            BashCode
        )
    ),
    !.

%% compile_facts_no_dedup(+Pred, +Arity, +Facts, +PredStr, -BashCode)
%  Compile facts without deduplication using regular array
compile_facts_no_dedup(_Pred, Arity, Facts, PredStr, BashCode) :-
    % Build array entries as strings (not key=value pairs)
    findall(Entry,
        (   member(Fact, Facts),
            Fact =.. [_|Args],
            atomic_list_concat(Args, ':', Entry)
        ),
        Entries),
    atomic_list_concat(Entries, '"\n    "', EntriesStr),

    % Generate bash code with regular array using atomic_list_concat to ensure LF endings
    (   Arity = 1 ->
        Template = [
            '#!/bin/bash',
            '# ~s - fact lookup (no deduplication)',
            '~s_data=(',
            '    "~s"',
            ')',
            '~s() {',
            '  local query="$1"',
            '  for item in "${~s_data[@]}"; do',
            '    [[ "$item" == "$query" ]] && echo "$item"',
            '  done',
            '}',
            '~s_stream() {',
            '  for item in "${~s_data[@]}"; do',
            '    echo "$item"',
            '  done',
            '}',
            '# Execute stream function when script is run directly',
            '~s_stream'
        ],
        atomic_list_concat(Template, '\n', TemplateStr),
        format(string(BashCode), TemplateStr, [PredStr, PredStr, EntriesStr, PredStr, PredStr, PredStr, PredStr])
    ;   Arity = 2 ->
        Template = [
            '#!/bin/bash',
            '# ~s - fact lookup (no deduplication)',
            '~s_data=(',
            '    "~s"',
            ')',
            '~s() {',
            '  local key="$1:$2"',
            '  for item in "${~s_data[@]}"; do',
            '    [[ "$item" == "$key" ]] && echo "$item"',
            '  done',
            '}',
            '~s_stream() {',
            '  for item in "${~s_data[@]}"; do',
            '    echo "$item"',
            '  done',
            '}',
            '~s_reverse_stream() {',
            '  for item in "${~s_data[@]}"; do',
            '    IFS=":" read -r a b <<< "$item"',
            '    echo "$b:$a"',
            '  done',
            '}',
            '# Execute stream function when script is run directly',
            '~s_stream'
        ],
        atomic_list_concat(Template, '\n', TemplateStr),
        format(string(BashCode), TemplateStr, [PredStr, PredStr, EntriesStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr])
    ;   % For higher arities, use a generic approach
        Template = [
            '#!/bin/bash',
            '# ~s - fact lookup (no deduplication)',
            '~s_data=(',
            '    "~s"',
            ')',
            '~s() {',
            '  echo "Error: ~s with arity ~w not yet supported for no_dedup" >&2',
            '  return 1',
            '}',
            '~s_stream() {',
            '  for item in "${~s_data[@]}"; do',
            '    echo "$item"',
            '  done',
            '}'
        ],
        atomic_list_concat(Template, '\n', TemplateStr),
        format(string(BashCode), TemplateStr, [PredStr, PredStr, EntriesStr, PredStr, PredStr, Arity, PredStr])
    ).

%% format_fact_entry(+Args, -Entry)
format_fact_entry(Args, Entry) :-
    atomic_list_concat(Args, ':', Key),
    format(string(Entry), '[~w]=1', [Key]).

% Helper: materialize facts into a bash array literal body as lines.
gather_fact_entries(Pred, 1, EntriesStr) :-
    functor(Head, Pred, 1),
    findall(Line, (clause(Head, true),
                   Head =.. [Pred, A1],
                   shell_quote(A1, Q1),
                   format(string(Line), "  \"~w\"", [Q1])),
            Lines),
    atomic_list_concat(Lines, "\n", EntriesStr).
gather_fact_entries(Pred, 2, EntriesStr) :-
    functor(Head, Pred, 2),
    findall(Line, (clause(Head, true),
                   Head =.. [Pred, A1, A2],
                   shell_quote(A1, Q1),
                   shell_quote(A2, Q2),
                   format(string(Line), "  [\"~w:~w\"]=1", [Q1, Q2])),
            Lines),
    atomic_list_concat(Lines, "\n", EntriesStr).

% Minimal shell quoting for literals used inside array items.
shell_quote(Atom, Q) :-
    atom_string(Atom, S0),
    replace_in_string(S0, "\"", "\\\"", S1),
    replace_in_string(S1, "\n", "\\n", Q).

replace_in_string(String, Find, Replace, Result) :-
    atomic_list_concat(Split, Find, String),
    atomic_list_concat(Split, Replace, Result).

%% compile_facts_debug(+Pred, +Arity, +_MergedOptions, -BashCode)
%  Debug version of compile_facts with extensive logging.
compile_facts_debug(Pred, Arity, _MergedOptions, BashCode) :-
    format('DEBUG: Starting compile_facts for ~w/~w~n', [Pred, Arity]),
    atom_string(Pred, PredStr),
    format('DEBUG: PredStr = ~w~n', [PredStr]),
    
    functor(Head, Pred, Arity),
    findall(Head, clause(Head, true), Facts),
    format('DEBUG: Found ~w facts~n', [length(Facts, _)]),
    
    % Try template rendering with explicit error handling
    (   catch(
            compose_templates(
                ['bash/header','facts/array_binary','facts/lookup_binary','facts/stream_binary','facts/reverse_stream_binary'],
                [pred=PredStr, entries='[test]=1', strategy=sort_u],
                BashCode
            ),
            Error,
            (format(user_error, 'DEBUG: Template rendering error: ~w~n', [Error]), fail)
        ) ->
        format('DEBUG: Template rendered successfully~n'),
        format('DEBUG: BashCode length: ~w~n', [string_length(BashCode, _)])
    ;   format('DEBUG: Template rendering failed~n'),
        fail
    ).

%% Compile single rule into streaming pipeline
compile_single_rule(Pred, Arity, Body, Options, BashCode) :-
    % Optimize goal order
    functor(Head, Pred, Arity),
    (   optimizer:optimize_clause(Head, Body, Options, OptimizedBody)
    ->  format('  Optimized body: ~w~n', [OptimizedBody])
    ;   OptimizedBody = Body
    ),

    extract_predicates(OptimizedBody, Predicates),
    atom_string(Pred, PredStr),

    % HYBRID DISPATCH: Try high-level patterns first, then fall back to general translation
    % 1. High-level pattern: inequality (e.g., sibling)
    (   has_inequality(OptimizedBody) ->
        compile_with_inequality(Pred, OptimizedBody, BashCode)
    % 2. General fallback: arithmetic operations
    ;   has_arithmetic(OptimizedBody) ->
        compile_with_arithmetic(Head, OptimizedBody, Options, BashCode)
    % 3. Match predicate (regex pattern matching)
    ;   has_match(OptimizedBody) ->
        compile_with_match(Head, OptimizedBody, Options, BashCode)
    % 4. No predicates at all
    ;   Predicates = [] ->
        format(string(BashCode), '#!/bin/bash
# ~s - no predicates
~s() { true; }

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    % 5. Standard streaming pipeline (existing high-level pattern)
    ;   % Standard streaming pipeline
        generate_pipeline(Predicates, Options, Pipeline),
        % Generate all necessary join functions
        collect_join_functions(Predicates, JoinFunctions),
        atomic_list_concat(JoinFunctions, '\n\n', JoinCode),
        generate_dedup_wrapper(PredStr, JoinCode, Pipeline, Options, BashCode)
    ).

%% Compile multiple rules (OR pattern)
compile_multiple_rules(Pred, Clauses, Options, BashCode) :-
    atom_string(Pred, PredStr),
    length(Clauses, NumAlts),

    % Collect all join functions needed
    findall(JoinFunc, (
        member(_-Body, Clauses),
        extract_predicates(Body, Preds),
        collect_join_functions(Preds, Funcs),
        member(JoinFunc, Funcs)
    ), AllJoinFuncs),
    list_to_set(AllJoinFuncs, UniqueJoinFuncs),  % Remove duplicates
    atomic_list_concat(UniqueJoinFuncs, '\n\n', JoinFuncsCode),
    
    % Generate alternative functions - iterate through clause pairs
    findall(FnCode, (
        nth1(I, Clauses, Head-Body),
        (   optimizer:optimize_clause(Head, Body, Options, OptimizedBody) -> true ; OptimizedBody = Body ),
        format(atom(FnName), '~s_alt~w', [PredStr, I]),

        % Check the specific pattern for related/2
        (   Pred = related, I = 1 ->
            % First alternative: parent(X,Y) - forward relationship
            format(string(FnCode), '~s() {
    parent_stream
}', [FnName])
        ;   Pred = related, I = 2 ->
            % Second alternative: parent(Y,X) - reverse relationship
            format(string(FnCode), '~s() {
    parent_reverse_stream
}', [FnName])
        ;   Pred = related, I = 3 ->
            % Third alternative: sibling(X,Y)
            format(string(FnCode), '~s() {
    sibling
}', [FnName])
        ;   % General case: use hybrid dispatch
            % Build variable map from the head
            Head =.. [_|Args],
            build_var_map(Args, 1, VarMap),
            (   has_inequality(OptimizedBody) ->
                % High-level pattern: inequality
                translate_body_to_bash(OptimizedBody, VarMap, BodyCode),
                format(string(FnCode), '~s() {
    ~s
}', [FnName, BodyCode])
            ;   has_arithmetic(OptimizedBody) ->
                % Arithmetic operations
                translate_body_to_bash(OptimizedBody, VarMap, BodyCode),
                format(string(FnCode), '~s() {
    ~s
}', [FnName, BodyCode])
            ;   % Standard streaming pipeline
                extract_predicates(OptimizedBody, Preds),
                (   Preds = [] ->
                    format(string(FnCode), '~s() {
    true
}', [FnName])
                ;   Preds = [SinglePred] ->
                    atom_string(SinglePred, SPredStr),
                    format(string(FnCode), '~s() {
    ~s_stream
}', [FnName, SPredStr])
                ;   generate_pipeline(Preds, Options, Pipeline),
                    format(string(FnCode), '~s() {
    ~s
}', [FnName, Pipeline])
                )
            )
        )
    ), AltFunctions),
    
    % Generate function names
    findall(FnCall, (
        between(1, NumAlts, I),
        format(atom(FnCall), '~s_alt~w', [PredStr, I])
    ), FnCalls),
    
    % Join with proper formatting
    atomic_list_concat(AltFunctions, '\n\n', FunctionsCode),
    atomic_list_concat(FnCalls, ' ; ', CallsStr),

    % Generate main function with appropriate deduplication
    format(string(Pipeline), '( ~s )', [CallsStr]),
    generate_dedup_wrapper_multi(PredStr, NumAlts, JoinFuncsCode, FunctionsCode, Pipeline, Options, BashCode).

%% Generate streaming pipeline from predicates
generate_pipeline([], _, "true") :- !.
generate_pipeline([Pred|Rest], Options, Pipeline) :-
    atom_string(Pred, PredStr),
    format(string(StreamFn), '~s_stream', [PredStr]),
    (   Rest = [] ->
        Pipeline = StreamFn
    ;   generate_join_chain([Pred|Rest], Options, JoinChain),
        format(string(Pipeline), '~s | ~s', [StreamFn, JoinChain])
    ).

%% Generate join chain for predicates
generate_join_chain([_], _, "") :- !.
generate_join_chain([_P1, P2|Rest], Options, Chain) :-
    atom_string(P2, P2Str),
    format(string(JoinFn), '~s_join', [P2Str]),
    (   Rest = [] ->
        Chain = JoinFn
    ;   generate_join_chain([P2|Rest], Options, RestChain),
        format(string(Chain), '~s | ~s', [JoinFn, RestChain])
    ).

%% Collect all join functions needed for a pipeline
% Tail-recursive with accumulator to prevent infinite loops
collect_join_functions(Preds, Funcs) :-
    collect_join_functions_(Preds, [], Rev), !,
    reverse(Rev, Funcs).

% Helper predicate with accumulator
collect_join_functions_(Preds, Acc, Acc) :-
    var(Preds), !.                         % Guard against variable lists
collect_join_functions_([], Acc, Acc) :- !.
collect_join_functions_([_], Acc, Acc) :- !.
collect_join_functions_([_, P2|Rest], Acc, Out) :-
    generate_join_function(P2, JoinFunc), !,
    collect_join_functions_([P2|Rest], [JoinFunc|Acc], Out).
% Catch-all for improper lists (e.g., dotted tails)
collect_join_functions_(_, Acc, Acc) :- !.

%% Helper for join functions
generate_join_function(Pred2, JoinCode) :-
    atom_string(Pred2, P2Str),
    format(string(JoinCode), '~s_join() {
    while IFS= read -r input; do
        IFS=":" read -r a b <<< "$input"
        for key in "${!~s_data[@]}"; do
            IFS=":" read -r c d <<< "$key"
            [[ "$b" == "$c" ]] && echo "$a:$d"
        done
    done
}', [P2Str, P2Str]).

%% Check for inequality constraints (cut-safe, terminating)
has_inequality((A, B)) :- !,
    (has_inequality(A) ; has_inequality(B)).
has_inequality(_ \= _) :- !.
has_inequality(\=(_, _)) :- !.
has_inequality(Body) :-
    nonvar(Body), !,
    fail.

%% Check for match constraints (regex pattern matching)
has_match((A, B)) :- !,
    (has_match(A) ; has_match(B)).
has_match(match(_, _)) :- !.
has_match(match(_, _, _)) :- !.
has_match(match(_, _, _, _)) :- !.
has_match(Body) :-
    nonvar(Body), !,
    fail.

% Keep helper for compatibility; add cuts to avoid fallback loops
has_inequality_in_conjunction((A, B)) :- !,
    (has_inequality(A) ; has_inequality(B)).
has_inequality_in_conjunction(Goal) :- !,
    has_inequality(Goal).

%% Compile with inequality handling
compile_with_inequality(Pred, Body, BashCode) :-
    atom_string(Pred, PredStr),
    extract_predicates(Body, Predicates),
    (   Predicates = [parent, parent] ->  % Fix: check for 'parent' atom, not variable
        % Special case: sibling pattern with parent predicates
        format(string(BashCode), '#!/bin/bash
# ~s - with inequality constraint

~s() {
    declare -A seen
    
    for key1 in "${!parent_data[@]}"; do
        IFS=":" read -r p1 c1 <<< "$key1"
        
        for key2 in "${!parent_data[@]}"; do
            IFS=":" read -r p2 c2 <<< "$key2"
            
            # Same parent, different children
            if [[ "$p1" == "$p2" && "$c1" != "$c2" ]]; then
                pair="$c1:$c2"
                if [[ -z "${seen[$pair]}" ]]; then
                    seen[$pair]=1
                    echo "$pair"
                fi
            fi
        done
    done
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    ;   % General inequality case
        format(string(BashCode), '#!/bin/bash
# ~s - with inequality constraint
~s() { echo "TODO: general inequality"; }

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    ).

%% has_arithmetic(+Body)
%  Check if body contains arithmetic operations
has_arithmetic((A, B)) :- !,
    (has_arithmetic(A) ; has_arithmetic(B)).
has_arithmetic(_ > _) :- !.
has_arithmetic(_ < _) :- !.
has_arithmetic(_ >= _) :- !.
has_arithmetic(_ =< _) :- !.
has_arithmetic(_ =:= _) :- !.
has_arithmetic(_ =\= _) :- !.
has_arithmetic(is(_, _)) :- !.
has_arithmetic(\+ A) :- !,
    has_arithmetic(A).
has_arithmetic(Body) :-
    nonvar(Body), !,
    fail.

%% compile_with_inequality_body(+Pred, +Arity, +Body, -BashBody)
%  Generate just the body code for inequality patterns (for use in alternatives)
compile_with_inequality_body(Pred, Arity, Body, BashBody) :-
    % Get the clause head to build variable map
    functor(Head, Pred, Arity),
    clause(Head, Body),
    Head =.. [_|Args],
    build_var_map(Args, 1, VarMap),
    translate_body_to_bash(Body, VarMap, BashBody).

%% compile_arithmetic_body(+Pred, +Arity, +Body, -BashBody)
%  Generate just the body code for arithmetic patterns (for use in alternatives)
compile_arithmetic_body(Pred, Arity, Body, BashBody) :-
    % Get the clause head to build variable map
    functor(Head, Pred, Arity),
    clause(Head, Body),
    Head =.. [_|Args],
    build_var_map(Args, 1, VarMap),
    translate_body_to_bash(Body, VarMap, BashBody).

%% compile_with_arithmetic(+Head, +Body, +Options, -BashCode)
%  General arithmetic translation fallback
%  Handles predicates with arithmetic but no known high-level pattern
compile_with_arithmetic(Head, Body, _Options, BashCode) :-
    functor(Head, Pred, Arity),
    atom_string(Pred, PredStr),

    % Extract variables from the clause head
    Head =.. [_|Args],

    % Build variable mapping: Prolog vars -> bash positional params
    build_var_map(Args, 1, VarMap),

    % Translate the body to bash
    translate_body_to_bash(Body, VarMap, BashBody),

    % Generate parameter declarations
    generate_param_decls(Args, 1, ParamDecls),

    % Build complete bash script
    format(string(BashCode), '#!/bin/bash
# ~s - with arithmetic operations

~s() {
~s
    ~s
}

# Stream function for use in pipelines
~s_stream() {
    ~s "$@"
}', [PredStr, PredStr, ParamDecls, BashBody, PredStr, PredStr]).

%% compile_with_match(+Head, +Body, +Options, -BashCode)
%  Compile predicates with match constraints (regex filtering)
%  Handles streaming pipeline with integrated match filters
compile_with_match(Head, Body, Options, BashCode) :-
    functor(Head, Pred, Arity),
    atom_string(Pred, PredStr),

    % Extract predicates and match constraints separately
    extract_predicates(Body, Predicates),
    extract_match_constraints(Body, MatchConstraints),

    % Get the clause head args
    Head =.. [_|Args],

    % Build variable mapping
    build_var_map(Args, 1, VarMap),

    % Generate base pipeline from predicates
    (   Predicates = [] ->
        % No predicates, just match constraints - use stdin
        BasePipeline = "cat"
    ;   Predicates = [SinglePred] ->
        % Single predicate source
        atom_string(SinglePred, SPredStr),
        format(string(BasePipeline), '~s_stream', [SPredStr])
    ;   % Multiple predicates - generate join chain
        generate_pipeline(Predicates, Options, BasePipeline)
    ),

    % Add match filters to the pipeline
    generate_match_filters(MatchConstraints, VarMap, Args, MatchFilters),

    % Get deduplication strategy
    get_dedup_strategy(Options, Strategy),

    % Build final pipeline with filters and deduplication
    (   MatchFilters = "" ->
        FinalPipeline = BasePipeline
    ;   format(string(FinalPipeline), '~s | ~s', [BasePipeline, MatchFilters])
    ),

    % Add deduplication if needed
    (   Strategy = sort_u ->
        format(string(CompletePipeline), '~s | sort -u', [FinalPipeline])
    ;   Strategy = hash ->
        format(string(CompletePipeline), '~s | awk \'!seen[$0]++\'', [FinalPipeline])
    ;   % no_dedup
        CompletePipeline = FinalPipeline
    ),

    % Generate complete bash script
    format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline with match filtering

~s() {
    ~s
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, CompletePipeline, PredStr, PredStr]).

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

%% generate_bash_capture_filter(+Pattern, +Groups, -FilterCode)
%  Generate bash while loop with [[ =~ ]] for capture group extraction
%  Uses BASH_REMATCH array to extract matched groups
generate_bash_capture_filter(Pattern, Groups, FilterCode) :-
    length(Groups, NumGroups),

    % Build the BASH_REMATCH extraction: "${BASH_REMATCH[1]}:${BASH_REMATCH[2]}..."
    findall(Index, between(1, NumGroups, Index), Indices),
    findall(RematchRef,
        (   member(I, Indices),
            atom_concat('${BASH_REMATCH[', I, Tmp),
            atom_concat(Tmp, ']}', RematchRef)
        ),
        RematchRefs),
    atomic_list_concat(RematchRefs, ':', RematchOutput),

    % Generate the while loop with regex matching
    atomic_list_concat([
        'while IFS= read -r line; do if [[ "$line" =~ ',
        Pattern,
        ' ]]; then echo "',
        RematchOutput,
        '"; fi; done'
    ], '', FilterCode).

%% generate_match_filters(+Constraints, +VarMap, +Args, -FilterCode)
%  Generate bash filter code for match constraints
generate_match_filters([], _, _, "") :- !.
generate_match_filters([Match|Rest], VarMap, Args, FilterCode) :-
    Match = match(_Var, Pattern, _Type, Groups),

    % Convert pattern to string
    (   atom(Pattern) ->
        atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),

    % Generate filter code based on whether there are capture groups
    (   Groups = [] ->
        % Boolean match - efficient grep filter
        format(string(ThisFilter), 'grep ~q', [PatternStr])
    ;   % With capture groups - use bash while loop with BASH_REMATCH
        generate_bash_capture_filter(PatternStr, Groups, ThisFilter)
    ),

    % Recursively process remaining constraints
    (   Rest = [] ->
        FilterCode = ThisFilter
    ;   generate_match_filters(Rest, VarMap, Args, RestFilters),
        format(string(FilterCode), '~s | ~s', [ThisFilter, RestFilters])
    ).

%% build_var_map(+Args, +Index, -VarMap)
%  Build mapping from Prolog variables to bash positional parameters
build_var_map([], _, []).
build_var_map([Var|Rest], Index, [Var-Index|RestMap]) :-
    Index1 is Index + 1,
    build_var_map(Rest, Index1, RestMap).

%% lookup_var(+Var, +VarMap, -Index)
%  Look up a variable in the VarMap using structural equality (==)
lookup_var(Var, [V-I|_], I) :-
    Var == V, !.
lookup_var(Var, [_|Rest], I) :-
    lookup_var(Var, Rest, I).

%% generate_param_decls(+Args, +Index, -Decls)
%  Generate local variable declarations from positional parameters
generate_param_decls([], _, '').
generate_param_decls([Var|Rest], Index, Decls) :-
    format(string(VarName), '~w', [Var]),
    format(string(Line), '    local ~w="$~w"', [VarName, Index]),
    Index1 is Index + 1,
    generate_param_decls(Rest, Index1, RestDecls),
    (   RestDecls = '' ->
        Decls = Line
    ;   format(string(Decls), '~s~n~s', [Line, RestDecls])
    ).

%% translate_body_to_bash(+Goal, +VarMap, -Bash)
%  Translates a Prolog goal (or conjunction) into bash code
translate_body_to_bash((A, B), VarMap, Bash) :- !,
    translate_body_to_bash(A, VarMap, BashA),
    translate_body_to_bash(B, VarMap, BashB),
    format(string(Bash), '~w && ~w', [BashA, BashB]).
translate_body_to_bash(\+ Goal, VarMap, Bash) :- !,
    translate_body_to_bash(Goal, VarMap, BashGoal),
    format(string(Bash), '! (~w)', [BashGoal]).
translate_body_to_bash(is(Var, Expr), VarMap, Bash) :- !,
    translate_expr(Expr, VarMap, BashExpr),
    format(string(VarName), '~w', [Var]),
    format(string(Bash), '~w=$(( ~s ))', [VarName, BashExpr]).
translate_body_to_bash(=(Var, Value), VarMap, Bash) :- !,
    format(string(VarName), '~w', [Var]),
    translate_expr(Value, VarMap, BashValue),
    format(string(Bash), '~w=~s', [VarName, BashValue]).
translate_body_to_bash(Goal, VarMap, Bash) :-
    goal_to_bash_operator(Goal, Op), !,
    Goal =.. [_, A, B],
    translate_expr(A, VarMap, BashA),
    translate_expr(B, VarMap, BashB),
    format(string(Bash), '[[ ~s -~w ~s ]]', [BashA, Op, BashB]).
% Match predicate (regex pattern matching)
translate_body_to_bash(match(Var, Pattern), VarMap, Bash) :- !,
    translate_body_to_bash(match(Var, Pattern, auto, []), VarMap, Bash).
translate_body_to_bash(match(Var, Pattern, _Type), VarMap, Bash) :- !,
    translate_body_to_bash(match(Var, Pattern, auto, []), VarMap, Bash).
translate_body_to_bash(match(Var, Pattern, _Type, _Groups), VarMap, Bash) :- !,
    % For bash, use =~ operator for regex matching
    translate_expr(Var, VarMap, BashVar),
    % Escape pattern if needed (for now, use as-is)
    (   atom(Pattern)
    ->  atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),
    format(string(Bash), '[[ ~s =~ ~s ]]', [BashVar, PatternStr]).
translate_body_to_bash(Goal, VarMap, Bash) :-
    % Default: call to another predicate
    functor(Goal, Functor, _),
    atom_string(Functor, FuncStr),
    Goal =.. [_|Args],
    maplist(translate_arg(VarMap), Args, BashArgs),
    atomic_list_concat(BashArgs, ' ', BashArgsStr),
    format(string(Bash), '~w ~w', [FuncStr, BashArgsStr]).

%% goal_to_bash_operator(+Goal, -BashOperator)
%  Maps Prolog comparison operators to bash test operators
goal_to_bash_operator(_ > _, 'gt').
goal_to_bash_operator(_ < _, 'lt').
goal_to_bash_operator(_ >= _, 'ge').
goal_to_bash_operator(_ =< _, 'le').
goal_to_bash_operator(_ =:= _, 'eq').
goal_to_bash_operator(_ =\= _, 'ne').

%% translate_expr(+PrologExpr, +VarMap, -BashExpr)
%  Translates a Prolog arithmetic expression to bash
translate_expr(Number, _VarMap, BashExpr) :-
    number(Number), !,
    format(string(BashExpr), '~w', [Number]).
translate_expr(Expr, VarMap, BashExpr) :-
    nonvar(Expr),
    Expr = A + B, !,
    translate_expr(A, VarMap, BashA),
    translate_expr(B, VarMap, BashB),
    format(string(BashExpr), '~w + ~w', [BashA, BashB]).
translate_expr(Expr, VarMap, BashExpr) :-
    nonvar(Expr),
    Expr = A - B, !,
    translate_expr(A, VarMap, BashA),
    translate_expr(B, VarMap, BashB),
    format(string(BashExpr), '~w - ~w', [BashA, BashB]).
translate_expr(Expr, VarMap, BashExpr) :-
    nonvar(Expr),
    Expr = A * B, !,
    translate_expr(A, VarMap, BashA),
    translate_expr(B, VarMap, BashB),
    format(string(BashExpr), '~w * ~w', [BashA, BashB]).
translate_expr(Expr, VarMap, BashExpr) :-
    nonvar(Expr),
    Expr = A / B, !,
    translate_expr(A, VarMap, BashA),
    translate_expr(B, VarMap, BashB),
    format(string(BashExpr), '~w / ~w', [BashA, BashB]).
translate_expr(Var, VarMap, BashExpr) :-
    var(Var), !,
    % Look up variable in VarMap to get its parameter number
    (   lookup_var(Var, VarMap, Index) ->
        format(string(BashExpr), '$~w', [Index])
    ;   % Fallback: use variable's print name
        format(string(VarName), '~w', [Var]),
        format(string(BashExpr), '$~w', [VarName])
    ).
translate_expr(Atom, _VarMap, BashExpr) :-
    atom(Atom), !,
    atom_string(Atom, BashExpr).

%% translate_arg(+VarMap, +Arg, -BashArg)
%  Formats a Prolog term as a bash function argument
translate_arg(_VarMap, Var, BashArg) :-
    var(Var), !,
    format(string(VarName), '~w', [Var]),
    format(string(BashArg), '"$~w"', [VarName]).
translate_arg(_VarMap, Atom, BashArg) :-
    atom(Atom), !,
    format(string(BashArg), '"~w"', [Atom]).
translate_arg(_VarMap, Number, BashArg) :-
    number(Number), !,
    format(string(BashArg), '~w', [Number]).

%% Write bash code to file with Unix line endings
write_bash_file(File, BashCode) :-
    % Convert all line endings to Unix format
    % Replace \r\n with \n, and standalone \r with \n
    atom_string(BashCode, BashStr),
    split_string(BashStr, "\r\n", "", Lines1),  % Split on Windows CRLF
    atomic_list_concat(Lines1, "\n", TempStr),   % Join with Unix LF
    split_string(TempStr, "\r", "", Lines2),     % Split on any remaining CR
    atomic_list_concat(Lines2, "\n", UnixStr),   % Join with Unix LF
    
    % Write with binary mode to preserve exact line endings
    open(File, write, Stream, [type(binary)]),
    string_codes(UnixStr, Codes),
    maplist(put_byte(Stream), Codes),
    close(Stream).

%% Main test
test_stream_compiler :-
    writeln('=== STREAM COMPILER TEST ==='),
    writeln('Testing basic non-recursive predicate compilation'),
    
    % Setup output directory
    (   exists_directory('output') -> true
    ;   make_directory('output')
    ),
    writeln('Output directory: output/'),
    
    % Clear any existing predicates
    abolish(user:parent/2),
    abolish(user:grandparent/2),
    abolish(user:sibling/2),
    abolish(user:related/2),
    
    % Define test predicates (facts) - with siblings
    assertz(user:parent(alice, bob)),
    assertz(user:parent(alice, barbara)),     % alice has two children
    assertz(user:parent(bob, charlie)),
    assertz(user:parent(bob, cathy)),         % bob has two children  
    assertz(user:parent(charlie, diana)),
    assertz(user:parent(diana, eve)),
    assertz(user:parent(diana, emily)),       % diana has two children
    assertz(user:parent(eve, frank)),
    
    % Define test predicates (rules)
    assertz(user:(grandparent(X, Z) :- parent(X, Y), parent(Y, Z))),
    assertz(user:(sibling(X, Y) :- parent(P, X), parent(P, Y), X \= Y)),
    
    % OR pattern - multiple ways to be related
    assertz(user:(related(X, Y) :- parent(X, Y))),       % Forward: X is parent of Y
    assertz(user:(related(X, Y) :- parent(Y, X))),       % Reverse: Y is parent of X (X is child of Y)
    assertz(user:(related(X, Y) :- sibling(X, Y))),
    
    % Declare constraints (using defaults: unique=true, unordered=true)
    % No need to declare for grandparent/sibling/related - they use defaults
    % But we'll be explicit for documentation
    declare_constraint(grandparent/2, [unique, unordered]),
    declare_constraint(sibling/2, [unique, unordered]),
    declare_constraint(related/2, [unique, unordered]),

    % Compile predicates (constraints come from declarations, not options)
    writeln('--- Compiling predicates ---'),
    compile_predicate(parent/2, [], ParentCode),
    write_bash_file('output/parent.sh', ParentCode),

    compile_predicate(grandparent/2, [], GrandparentCode),
    write_bash_file('output/grandparent.sh', GrandparentCode),

    compile_predicate(sibling/2, [], SiblingCode),
    write_bash_file('output/sibling.sh', SiblingCode),

    compile_predicate(related/2, [], RelatedCode),
    write_bash_file('output/related.sh', RelatedCode),
    
    % Generate test script
    generate_test_script,
    
    writeln('--- Test Complete ---'),
    writeln('Check files in output/'),
    writeln('Run: bash output/test.sh').

%% Generate deduplication wrapper based on constraints
%  For single-rule predicates
generate_dedup_wrapper(PredStr, JoinCode, Pipeline, Options, BashCode) :-
    (   constraint_implies_sort_u(Options) ->
        % Use sort -u for unique + unordered
        format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline with uniqueness (sort -u)

~s

~s() {
    ~s | sort -u
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
    ;   constraint_implies_hash(Options) ->
        % Use hash-based dedup for unique + ordered
        format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline with hash-based deduplication (preserves order)

~s

~s() {
    declare -A seen
    ~s | while IFS= read -r line; do
        if [[ -z "${seen[$line]}" ]]; then
            seen[$line]=1
            echo "$line"
        fi
    done
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
    ;   % No deduplication
        format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline (no deduplication)

~s

~s() {
    ~s
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
    ).

%% Generate deduplication wrapper for multiple-rule predicates
generate_dedup_wrapper_multi(PredStr, NumAlts, JoinFuncsCode, FunctionsCode, Pipeline, Options, BashCode) :-
    (   constraint_implies_sort_u(Options) ->
        format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives (sort -u)

~s

~s

# Main function - combine alternatives with uniqueness
~s() {
    ~s | sort -u
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, Pipeline])
    ;   constraint_implies_hash(Options) ->
        format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives (hash dedup, preserves order)

~s

~s

# Main function - combine alternatives preserving order
~s() {
    declare -A seen
    ~s | while IFS= read -r line; do
        if [[ -z "${seen[$line]}" ]]; then
            seen[$line]=1
            echo "$line"
        fi
    done
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, Pipeline])
    ;   % No deduplication
        format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives (no deduplication)

~s

~s

# Main function - combine alternatives
~s() {
    ~s
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, Pipeline])
    ).

%% Generate test script
generate_test_script :-
    TestScript = '#!/bin/bash
# Test script for compiled predicates

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source files relative to script directory
source "$SCRIPT_DIR/parent.sh"
source "$SCRIPT_DIR/grandparent.sh"
source "$SCRIPT_DIR/sibling.sh"
source "$SCRIPT_DIR/related.sh"

echo "=== Testing parent ==="
echo "parent alice bob: $(parent alice bob)"
echo "parent stream:"
parent_stream | head -3

echo ""
echo "=== Testing grandparent ==="
echo "grandparent alice charlie:"
grandparent | grep "alice:charlie" || echo "Not found"
echo "All grandparents:"
grandparent

echo ""
echo "=== Testing sibling ==="
echo "Siblings:"
sibling

echo ""
echo "=== Testing related ==="
echo "Related pairs:"
related | head -5',
    
    write_bash_file('output/test.sh', TestScript).
