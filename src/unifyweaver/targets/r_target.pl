% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% r_target.pl - R Target for UnifyWeaver
% Generates standalone R scripts for record/field processing.
% Supports vectorized operations, data frames, and native pipes.

:- module(r_target, [
    compile_predicate_to_r/3,       % +Predicate, +Options, -RCode
    compile_facts_to_r/3,           % +Pred, +Arity, -RCode
    init_r_target/0,                % Initialize R target with bindings
    compile_r_pipeline/3,           % +Predicates, +Options, -RCode
    test_r_pipeline/0               % Test pipeline generation
]).

:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module(common_generator).
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('../bindings/r_bindings').
:- use_module('../core/template_system').

% Track required packages/libraries
:- dynamic required_r_package/1.
:- dynamic collected_component/2.

%% init_r_target
%  Initialize the R target by loading bindings and clearing state.
init_r_target :-
    retractall(required_r_package(_)),
    retractall(collected_component(_, _)),
    init_r_bindings.

%% compile_predicate_to_r(+PredIndicator, +Options, -RCode)
%  Compile a Prolog predicate indicator into an R function.
%  Handles: no clauses, single clause, multiple clauses (if/else if chain).
compile_predicate_to_r(PredIndicator, _Options, RCode) :-
    (   PredIndicator = Module:Pred/Arity
    ->  true
    ;   PredIndicator = Pred/Arity,
        Module = user
    ),
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(Module:Head, Body), Clauses),
    (   Clauses = []
    ->  format(string(RCode), '# No clauses found for ~w/~w', [Pred, Arity])
    ;   Clauses = [SingleHead-SingleBody]
    ->  compile_rule(SingleHead, SingleBody, RCode)
    ;   compile_multi_clause_r(Pred, Arity, Clauses, RCode)
    ).

%% compile_multi_clause_r(+Pred, +Arity, +Clauses, -RCode)
%  Compile multiple clauses into an R function with if/else if chain.
compile_multi_clause_r(Pred, Arity, Clauses, RCode) :-
    % Generate standard arg names
    numlist(1, Arity, Indices),
    findall(ArgName, (
        member(I, Indices),
        format(atom(ArgName), 'arg~w', [I])
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList),

    % Compile each clause into an if/else if branch
    compile_clause_branches(Clauses, ArgNames, Branches),
    atomic_list_concat(Branches, ' else ', BranchCode),

    format(string(RCode),
'~w <- function(~w) {
    ~w else {
        stop("No matching clause for ~w")
    }
}
', [Pred, ArgList, BranchCode, Pred]).

%% compile_clause_branches(+Clauses, +ArgNames, -Branches)
compile_clause_branches([], _, []).
compile_clause_branches([Head-Body|Rest], ArgNames, [Branch|RestBranches]) :-
    compile_clause_branch(Head, Body, ArgNames, Branch),
    compile_clause_branches(Rest, ArgNames, RestBranches).

%% compile_clause_branch(+Head, +Body, +ArgNames, -Branch)
%  Compile one clause into an if(...) { ... } branch.
compile_clause_branch(Head, Body, ArgNames, Branch) :-
    Head =.. [_|HeadArgs],
    % Build condition from head argument patterns
    generate_head_condition(HeadArgs, ArgNames, Condition),
    % Build VarMap mapping head variables to arg names
    build_head_varmap(HeadArgs, ArgNames, VarMap),
    % Compile body with the VarMap
    compile_body(Body, VarMap, _, BodyCode),
    format(string(Branch), 'if (~w) {\n~s\n    }', [Condition, BodyCode]).

%% generate_head_condition(+HeadArgs, +ArgNames, -Condition)
%  Build R condition string from head argument patterns.
generate_head_condition(HeadArgs, ArgNames, Condition) :-
    maplist(generate_arg_condition, HeadArgs, ArgNames, Conditions),
    exclude(==('TRUE'), Conditions, NonTrivial),
    (   NonTrivial = []
    ->  Condition = 'TRUE'
    ;   atomic_list_concat(NonTrivial, ' && ', Condition)
    ).

%% generate_arg_condition(+HeadArg, +ArgName, -Condition)
%  Single argument condition: ground → equality check, var → TRUE.
generate_arg_condition(Arg, _ArgName, 'TRUE') :-
    var(Arg), !.
generate_arg_condition(Arg, ArgName, Condition) :-
    number(Arg), !,
    format(string(Condition), '~w == ~w', [ArgName, Arg]).
generate_arg_condition(Arg, ArgName, Condition) :-
    atom(Arg), !,
    format(string(Condition), '~w == "~w"', [ArgName, Arg]).
generate_arg_condition(Arg, ArgName, Condition) :-
    format(string(Condition), '~w == ~w', [ArgName, Arg]).

%% build_head_varmap(+HeadArgs, +ArgNames, -VarMap)
%  Map head variables to their corresponding arg names.
build_head_varmap([], [], []).
build_head_varmap([Arg|Args], [Name|Names], [Arg-Name|Rest]) :-
    var(Arg), !,
    build_head_varmap(Args, Names, Rest).
build_head_varmap([_|Args], [_|Names], Rest) :-
    build_head_varmap(Args, Names, Rest).

%% collect_declared_component(+Category, +Name)
%  Track a component that was referenced during compilation.
collect_declared_component(Category, Name) :-
    (   collected_component(Category, Name)
    ->  true
    ;   assertz(collected_component(Category, Name))
    ).

%% compile_collected_components(-Code)
%  Compile all collected components into R code.
compile_collected_components(Code) :-
    findall(CompCode, (
        collected_component(Category, Name),
        component_registry:compile_component(Category, Name, [], CompCode)
    ), CompCodes),
    (   CompCodes = []
    ->  Code = ''
    ;   atomic_list_concat(CompCodes, '\n\n', Code)
    ).

compile_rule(Head, Body, Code) :-
    Head =.. [Pred|Args],
    
    % Generate variable mapping for arguments
    map_args(Args, 1, VarMap, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList),
    
    % Compile body
    compile_body(Body, VarMap, _, BodyCode),
    
    format(string(Code),
'~w <- function(~w) {
~s
}
', [Pred, ArgList, BodyCode]).

map_args([], _, [], []).
map_args([Arg|Rest], Idx, [Arg-VarName|MapRest], [VarName|NamesRest]) :-
    format(atom(VarName), 'arg~w', [Idx]),
    NextIdx is Idx + 1,
    map_args(Rest, NextIdx, MapRest, NamesRest).

compile_body(true, V, V, "") :- !.
compile_body((A, B), V0, V2, Code) :- !,
    compile_body(A, V0, V1, C1),
    compile_body(B, V1, V2, C2),
    format(string(Code), '~s~n~s', [C1, C2]).
compile_body(_M:Goal, V0, V1, Code) :- !,
    compile_body(Goal, V0, V1, Code).

% Specialized data frame streaming operations
compile_body(filter(DF, Expr, Out), V0, V1, Code) :- !,
    ensure_var(V0, Out, ROut, V1),
    resolve_val(V0, DF, RDF),
    translate_r_expr(Expr, V0, RExpr),
    format(string(Code), '    ~w <- subset(~w, ~w)', [ROut, RDF, RExpr]).

compile_body(group_by(DF, Col, Out), V0, V1, Code) :- !,
    ensure_var(V0, Out, ROut, V1),
    resolve_val(V0, DF, RDF),
    resolve_val(V0, Col, RCol),
    format(string(Code), '    ~w <- aggregate(. ~~ ~w, data=~w, FUN=list)', [ROut, RCol, RDF]).

compile_body(sort_by(DF, Col, Out), V0, V1, Code) :- !,
    ensure_var(V0, Out, ROut, V1),
    resolve_val(V0, DF, RDF),
    resolve_val(V0, Col, RCol),
    format(string(Code), '    ~w <- ~w[order(~w[[~w]]), ]', [ROut, RDF, RDF, RCol]).

compile_body(Goal, V0, V1, Code) :-
    functor(Goal, Pred, Arity),
    (   binding(r, Pred/Arity, TargetName, Inputs, Outputs, _Options)
    ->  Goal =.. [_|Args],
        length(Inputs, InCount),
        length(InArgs, InCount),
        append(InArgs, OutArgs, Args),
        
        maplist(resolve_val(V0), InArgs, RInArgs),
        atomic_list_concat(RInArgs, ', ', RArgsStr),
        format(string(Expr), '~w(~w)', [TargetName, RArgsStr]),
        
        (   Outputs = []
        ->  V1 = V0,
            format(string(Code), '    ~w', [Expr])
        ;   OutArgs = [OutVar],
            ensure_var(V0, OutVar, ROutVar, V1),
            format(string(Code), '    ~w <- ~w', [ROutVar, Expr])
        )
    ;   V1 = V0,
        format(string(Code), '    # Unknown predicate: ~w', [Goal])
    ).

resolve_val(VarMap, Var, Val) :-
    var(Var), lookup_var(Var, VarMap, Name), !,
    format(string(Val), '~w', [Name]).
resolve_val(_, Val, StrVal) :-
    format(string(StrVal), '~w', [Val]).

ensure_var(VarMap, Var, Name, VarMap) :-
    lookup_var(Var, VarMap, Name), !.
ensure_var(VarMap, Var, Name, [Var-Name|VarMap]) :-
    gensym(v, Name).

lookup_var(Var, [V-Name|_], Name) :- Var == V, !.
lookup_var(Var, [_|Rest], Name) :- lookup_var(Var, Rest, Name).

% Expression translation for filter and arithmetic
translate_r_expr(Var, V, Res) :- var(Var), lookup_var(Var, V, Name), !, format(string(Res), '~w', [Name]).
translate_r_expr(Atom, _, Res) :- atom(Atom), !, format(string(Res), '~w', [Atom]). % Column names unquoted in subset
translate_r_expr(Num, _, Res) :- number(Num), !, format(string(Res), '~w', [Num]).
translate_r_expr(Expr, V, Res) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    translate_r_expr(Left, V, LRes),
    translate_r_expr(Right, V, RRes),
    r_op_map(Op, ROp),
    format(string(Res), '(~w ~w ~w)', [LRes, ROp, RRes]).

r_op_map(>, '>').
r_op_map(<, '<').
r_op_map(>=, '>=').
r_op_map(=<, '<=').
r_op_map(==, '==').
r_op_map(\=, '!=').
r_op_map(and, '&').
r_op_map(or, '|').

% ============================================================================
% FACT EXPORT - compile_facts_to_r/3
% ============================================================================

%% compile_facts_to_r(+Pred, +Arity, -RCode)
%  Export Prolog facts as an R script with list data and accessor functions.
%  Generates: <pred>_facts list, get_all_<pred>(), stream_<pred>(fn),
%  contains_<pred>(...) functions, and a main block.
compile_facts_to_r(Pred, Arity, RCode) :-
    atom_string(Pred, PredStr),

    % Collect all facts
    functor(Head, Pred, Arity),
    findall(Args, (user:clause(Head, true), Head =.. [_|Args]), AllFacts),

    % Format facts as R list of c() vectors
    findall(Entry, (
        member(Args, AllFacts),
        maplist(format_r_fact_arg, Args, FormattedArgs),
        atomic_list_concat(FormattedArgs, ', ', ArgsStr),
        format(string(Entry), '    c(~w)', [ArgsStr])
    ), Entries),
    atomic_list_concat(Entries, ',\n', EntriesCode),

    % Generate arg names for contains function
    numlist(1, Arity, Indices),
    findall(ArgName, (
        member(I, Indices),
        format(atom(ArgName), 'arg~w', [I])
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ContainsArgStr),

    % Generate contains comparison
    findall(Check, (
        member(I, Indices),
        format(string(Check), 'fact[~w] == arg~w', [I, I])
    ), Checks),
    atomic_list_concat(Checks, ' && ', CheckStr),

    format(string(RCode),
'#!/usr/bin/env Rscript
# Generated by UnifyWeaver R Target - Fact Export
# Predicate: ~w/~w

# Fact data
~w_facts <- list(
~w
)

# Return all facts
get_all_~w <- function() {
    return(~w_facts)
}

# Apply function to each fact
stream_~w <- function(fn) {
    lapply(~w_facts, fn)
}

# Check if a fact exists
contains_~w <- function(~w) {
    for (fact in ~w_facts) {
        if (~w) return(TRUE)
    }
    return(FALSE)
}

# Main
if (!interactive()) {
    for (fact in ~w_facts) {
        cat(paste(fact, collapse=":"), "\\n")
    }
}
', [PredStr, Arity,
    PredStr, EntriesCode,
    PredStr, PredStr,
    PredStr, PredStr,
    PredStr, ContainsArgStr, PredStr, CheckStr,
    PredStr]).

%% format_r_fact_arg(+Arg, -Formatted)
%  Format a single fact argument for R output.
format_r_fact_arg(Arg, Formatted) :-
    (   number(Arg)
    ->  format(string(Formatted), '~w', [Arg])
    ;   format(string(Formatted), '"~w"', [Arg])
    ).

% ============================================================================
% PIPELINE MODE
% ============================================================================

%% compile_r_pipeline(+Predicates, +Options, -RCode)
%  Generates an R pipeline (e.g. using native |> or dplyr %>%).
compile_r_pipeline(Predicates, Options, RCode) :-
    (member(pipeline_name(PipelineName), Options) -> true ; PipelineName = 'pipeline'),
    (member(pipeline_mode(Mode), Options) -> true ; Mode = sequential),
    
    r_pipeline_header(HeaderCode),
    generate_r_stage_functions(Predicates, StageFunctions),
    generate_r_pipeline_connector(Predicates, PipelineName, Mode, ConnectorCode),
    
    format(string(RCode), '~s~n~n~s~n~n~s',
           [HeaderCode, StageFunctions, ConnectorCode]).

r_pipeline_header(Code) :-
    format(string(Code),
'# Auto-generated R Pipeline
suppressPackageStartupMessages({
    library(jsonlite)
    # library(dplyr)  # Uncomment if using dplyr pipes
})

# Helper for reading JSONL
read_jsonl <- function(file_path) {
    con <- file(file_path, "r")
    lines <- readLines(con)
    close(con)
    df <- do.call(rbind, lapply(lines, fromJSON))
    return(as.data.frame(df))
}
', []).

generate_r_stage_functions([], "").
generate_r_stage_functions([Pred|Rest], Code) :-
    compile_predicate_to_r(Pred, [], StageCode),
    generate_r_stage_functions(Rest, RestCode),
    (RestCode = "" -> Code = StageCode ; format(string(Code), '~s~n~n~s', [StageCode, RestCode])).

generate_r_pipeline_connector(Predicates, PipelineName, sequential, Code) :-
    maplist(get_pred_name, Predicates, StageNames),
    atomic_list_concat(StageNames, ' |>\n    ', PipelineStr),
    format(string(Code),
'# Sequential pipeline connector: ~w
~w <- function(data) {
    data |>
    ~w
}
', [PipelineName, PipelineName, PipelineStr]).

generate_r_pipeline_connector(Predicates, PipelineName, generator, Code) :-
    maplist(get_pred_name, Predicates, StageNames),
    atomic_list_concat(StageNames, ' |>\n            ', PipelineStr),
    format(string(Code),
'# Fixpoint pipeline connector: ~w
# Iterates until no new rows are produced.
~w <- function(data) {
    # Initialize seen set with hash representation of rows
    seen <- new.env(hash=TRUE, parent=emptyenv())
    
    # helper to hash rows
    hash_row <- function(row) {
        digest::digest(row) # requires digest package
    }
    
    # Initial setup
    records <- data
    output_records <- data
    for (i in seq_len(nrow(records))) {
        seen[[hash_row(records[i,])]] <- TRUE
    }
    
    changed <- TRUE
    while (changed) {
        changed <- FALSE
        current_records <- records
        
        for (i in seq_len(nrow(current_records))) {
            row <- current_records[i, , drop=FALSE]
            # Run pipeline stages on the current row
            new_row <- row |>
            ~w
            
            key <- hash_row(new_row)
            if (is.null(seen[[key]])) {
                seen[[key]] <- TRUE
                records <- rbind(records, new_row)
                output_records <- rbind(output_records, new_row)
                changed <- TRUE
            }
        }
    }
    return(output_records)
}
', [PipelineName, PipelineName, PipelineStr]).

get_pred_name(Pred/_Arity, Pred).

% ============================================================================
% TREE RECURSION - R target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tree_recursion').
:- multifile tree_recursion:compile_tree_pattern/6.

%% R fibonacci-like tree recursion
tree_recursion:compile_tree_pattern(r, fibonacci, Pred, _Arity, UseMemo, RCode) :-
    atom_string(Pred, PredStr),
    (   UseMemo = true ->
        MemoDecl = '~w_memo <- new.env(hash=TRUE, parent=emptyenv())'
    ;   MemoDecl = '# Memoization disabled'
    ),
    (   UseMemo = true ->
        format(string(MemoDeclFormatted), MemoDecl, [PredStr])
    ;   MemoDeclFormatted = MemoDecl
    ),

    TemplateLines = [
        "# {{pred}}/2 - tree recursive pattern (Fibonacci-like in R)",
        "{{memo_decl}}",
        "",
        "{{pred}} <- function(n, expected=NULL) {",
        "    if (n == 0) return(0)",
        "    if (n == 1) return(1)",
        "",
        "    # Check memo",
        "    key <- as.character(n)",
        "    if (!is.null({{{pred}}_memo[[key]]})) {",
        "        result <- {{{pred}}_memo[[key]]}",
        "    } else {",
        "        # Recursive calls",
        "        result <- {{pred}}(n - 1) + {{pred}}(n - 2)",
        "        {{{pred}}_memo[[key]]} <- result",
        "    }",
        "",
        "    if (!is.null(expected)) {",
        "        return(result == expected)",
        "    }",
        "    return(result)",
        "}"
    ],
    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, memo_decl=MemoDeclFormatted], RCode).

%% R binary tree recursion
tree_recursion:compile_tree_pattern(r, binary_tree, Pred, _Arity, UseMemo, RCode) :-
    atom_string(Pred, PredStr),

    % Detect operation type from predicate name
    (   sub_atom(Pred, _, _, _, sum) ->
        BaseValue = "0",
        CombineExpr = "value + left_result + right_result"
    ;   sub_atom(Pred, _, _, _, height) ->
        BaseValue = "0",
        CombineExpr = "1 + max(left_result, right_result)"
    ;   sub_atom(Pred, _, _, _, count) ->
        BaseValue = "0",
        CombineExpr = "1 + left_result + right_result"
    ;   % Generic: sum-like default
        BaseValue = "0",
        CombineExpr = "value + left_result + right_result"
    ),

    % Memo declaration
    (   UseMemo = true ->
        format(string(MemoDecl), '~w_memo <- new.env(hash=TRUE, parent=emptyenv())', [PredStr])
    ;   MemoDecl = '# Memoization disabled'
    ),

    % Memo check code
    (   UseMemo = true ->
        format(string(MemoCheck),
'    # Check memo
    key <- digest::digest(tree)
    if (!is.null(~w_memo[[key]])) return(~w_memo[[key]])', [PredStr, PredStr])
    ;   MemoCheck = ''
    ),

    % Memo store code
    (   UseMemo = true ->
        format(string(MemoStore), '\n    ~w_memo[[key]] <<- result', [PredStr])
    ;   MemoStore = ''
    ),

    format(string(RCode),
'#!/usr/bin/env Rscript
# ~w - tree recursion (binary tree pattern, R)
# Trees: list(value=V, left=L, right=R) or NULL for empty

~w

~w <- function(tree) {
    # Base case: empty tree
    if (is.null(tree)) return(~w)

~w

    # Decompose tree
    value <- tree$value
    left <- tree$left
    right <- tree$right

    # Recursive calls
    left_result <- ~w(left)
    right_result <- ~w(right)

    # Combine results
    result <- ~w~w
    return(result)
}

# Example usage:
# tree <- list(value=5, left=list(value=3, left=NULL, right=NULL),
#              right=list(value=7, left=NULL, right=NULL))
# cat(~w(tree), "\\n")

# Run when script executed directly
if (!interactive()) {
    # Demo with a simple tree
    tree <- list(value=1,
        left=list(value=2, left=NULL, right=NULL),
        right=list(value=3, left=NULL, right=NULL))
    cat(~w(tree), "\\n")
}
', [PredStr, MemoDecl, PredStr, BaseValue, MemoCheck,
    PredStr, PredStr, CombineExpr, MemoStore,
    PredStr, PredStr]).

%% R generic tree recursion (template with commented example)
tree_recursion:compile_tree_pattern(r, generic, Pred, Arity, _UseMemo, RCode) :-
    atom_string(Pred, PredStr),
    format(string(RCode),
'#!/usr/bin/env Rscript
# ~w/~w - generic tree recursive pattern (R)
# Auto-generated template. Customize the tree structure and combine logic below.

~w_memo <- new.env(hash=TRUE, parent=emptyenv())

~w <- function(tree) {
    # Base case: empty/leaf node
    if (is.null(tree)) return(0)

    # Memo check
    key <- digest::digest(tree)
    if (!is.null(~w_memo[[key]])) return(~w_memo[[key]])

    # --- Customize below ---
    # Decompose tree into value and children
    # Example for binary tree: list(value=V, left=L, right=R)
    value <- tree$value
    children <- list(tree$left, tree$right)

    # Recurse on each child
    child_results <- sapply(children, function(child) {
        if (is.null(child)) 0 else ~w(child)
    })

    # Combine: sum, max, count, etc.
    result <- value + sum(child_results)
    # --- End customization ---

    # Store in memo
    ~w_memo[[key]] <<- result
    return(result)
}
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% TESTING
% ============================================================================

test_r_pipeline :-
    init_r_target,
    asserta((user:filter_stage(In, Out) :- filter(In, >(age, 18), Out))),
    asserta((user:sort_stage(In, Out) :- sort_by(In, age, Out))),
    asserta((user:group_stage(In, Out) :- group_by(In, role, Out))),
    
    format('~n=== R Sequential Pipeline Test ===~n'),
    compile_r_pipeline([filter_stage/2, sort_stage/2, group_stage/2], [pipeline_name(data_processor), pipeline_mode(sequential)], CodeSeq),
    format('~s~n', [CodeSeq]),
    
    format('~n=== R Fixpoint Generator Pipeline Test ===~n'),
    compile_r_pipeline([filter_stage/2, sort_stage/2, group_stage/2], [pipeline_name(data_processor_gen), pipeline_mode(generator)], CodeGen),
    format('~s~n', [CodeGen]),
    
    retractall(user:filter_stage(_, _)),
    retractall(user:sort_stage(_, _)),
    retractall(user:group_stage(_, _)).
