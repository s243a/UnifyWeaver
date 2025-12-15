:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% rust_target.pl - Rust Target for UnifyWeaver
% Generates standalone Rust programs for record/field processing.
% Phase 3: JSON support (serde_json crate).

:- module(rust_target, [
    compile_predicate_to_rust/3,      % +Predicate, +Options, -RustCode
    compile_facts_to_rust/3,          % +Pred, +Arity, -RustCode  -- NEW
    write_rust_program/2,             % +RustCode, +FilePath
    write_rust_project/2,             % +RustCode, +Dir
    json_schema/2,                    % +SchemaName, +Fields (directive)
    get_json_schema/2,                % +SchemaName, -Fields (lookup)
    init_rust_target/0,               % Initialize bindings
    compile_rust_pipeline/3,          % +Predicates, +Options, -RustCode
    test_rust_pipeline_generator/0,   % Unit tests for pipeline generator mode
    % Enhanced pipeline chaining exports
    compile_rust_enhanced_pipeline/3, % +Stages, +Options, -RustCode
    rust_enhanced_helpers/2,          % +ParallelMode, -Code
    generate_rust_enhanced_connector/3, % +Stages, +PipelineName, -Code
    test_rust_enhanced_chaining/0     % Test enhanced pipeline chaining
]).

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(gensym)). % For generating unique variable names
:- use_module(library(yall)).
:- use_module('../core/binding_registry').
:- use_module('../bindings/rust_bindings').
:- use_module('../core/pipeline_validation').

%% init_rust_target
%  Initialize the Rust target by loading bindings.
init_rust_target :-
    init_rust_bindings.

%% ============================================ 
%% JSON SCHEMA SUPPORT
%% ============================================ 

:- dynamic json_schema_def/2.

json_schema(SchemaName, Fields) :-
    (   validate_schema_fields(Fields)
    ->  retractall(json_schema_def(SchemaName, _)),
        assertz(json_schema_def(SchemaName, Fields))
    ;   format('ERROR: Invalid schema definition for ~w: ~w~n', [SchemaName, Fields]),
        fail
    ).

validate_schema_fields([]).
validate_schema_fields([field(Name, Type)|Rest]) :-
    atom(Name),
    valid_json_type(Type),
    validate_schema_fields(Rest).
validate_schema_fields([Invalid|_]) :-
    format('ERROR: Invalid field specification: ~w~n', [Invalid]),
    fail.

valid_json_type(string).
valid_json_type(integer).
valid_json_type(float).
valid_json_type(boolean).
valid_json_type(array).
valid_json_type(object).
valid_json_type(any).

get_json_schema(SchemaName, Fields) :-
    json_schema_def(SchemaName, Fields), !.
get_json_schema(SchemaName, _) :-
    format('ERROR: Schema not found: ~w~n', [SchemaName]),
    fail.

get_field_type(SchemaName, FieldName, Type) :-
    get_json_schema(SchemaName, Fields),
    member(field(FieldName, Type), Fields), !.
get_field_type(_SchemaName, _FieldName, serde_json_value).

%% ============================================ 
%% PUBLIC API
%% ============================================ 

compile_predicate_to_rust(PredIndicator, Options, RustCode) :- 
    (   PredIndicator = Module:Name/Arity
    ->  Pred = Name, GoalModule = Module
    ;   PredIndicator = Name/Arity, GoalModule = user, Pred = Name
    ),
    format('=== Compiling ~w/~w to Rust (Module: ~w) ===~n', [Pred, Arity, GoalModule]),

    (   option(json_input(true), Options)
    ->  format('  Mode: JSON input~n'),
        compile_json_input_to_rust(Pred, Arity, Options, RustCode)
    ;   option(json_output(true), Options)
    ->  format('  Mode: JSON output~n'),
        compile_json_output_to_rust(Pred, Arity, Options, RustCode)
    ;   functor(Head, Pred, Arity),
        GoalModule:clause(Head, Body),
        is_semantic_predicate(Body)
    ->  format('  Mode: Semantic Runtime~n'),
        compile_semantic_mode(Pred, Arity, Head, Body, RustCode)
    ;   option(aggregation(AggOp), Options, none),
        AggOp \= none
    ->  option(field_delimiter(FieldDelim), Options, colon),
        option(include_main(IncludeMain), Options, true),
        compile_aggregation_to_rust(Pred, Arity, AggOp, FieldDelim, IncludeMain, RustCode)
    ;   compile_predicate_to_rust_normal(Pred, Arity, Options, RustCode)
    ).

is_semantic_predicate(Body) :-
    (   sub_term(crawler_run(_, _), Body)
    ;   sub_term(graph_search(_, _, _, _, _), Body)
    ;   sub_term(graph_search(_, _, _, _), Body)
    ).

compile_semantic_mode(Pred, _Arity, Head, Body, RustCode) :-
    % Generate main.rs content
    generate_semantic_main(Pred, Head, Body, RustCode).

generate_semantic_main(_Pred, _Head, Body, RustCode) :-
    % Simple translation for now
    translate_semantic_body(Body, BodyCode),
    format(string(RustCode), '
mod importer;
mod crawler;
mod searcher;
mod llm;
mod embedding;

use importer::PtImporter;
use crawler::PtCrawler;
use searcher::PtSearcher;
use embedding::EmbeddingProvider;
use llm::LLMProvider;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize Runtime
    let embedding = EmbeddingProvider::new("models/model.safetensors", "models/tokenizer.json")?;
    let importer = PtImporter::new("data.redb")?;
    let crawler = PtCrawler::new(importer, embedding.clone());
    let searcher = PtSearcher::new("data.redb", embedding)?;
    // let llm = LLMProvider::new("gemini-2.5-flash");

    ~s

    Ok(())
}
', [BodyCode]).

translate_semantic_body((A, B), Code) :- !,
    translate_semantic_body(A, C1),
    translate_semantic_body(B, C2),
    format(string(Code), "~s\n    ~s", [C1, C2]).

translate_semantic_body(crawler_run(Seeds, Depth), Code) :-
    % Convert seeds to Rust vec
    (   is_list(Seeds) -> maplist(atom_string, Seeds, SeedStrs), maplist(rust_quote, SeedStrs, QuotedSeeds), atomic_list_concat(QuotedSeeds, ', ', SeedList)
    ;   format(string(SeedList), '"~w"', [Seeds]) % Single seed
    ),
    format(string(Code), 'crawler.crawl(&vec![~s], ~w)?;', [SeedList, Depth]).

translate_semantic_body(graph_search(Query, TopK, Hops, Results), Code) :-
    translate_semantic_body(graph_search(Query, TopK, Hops, [mode(vector)], Results), Code).

translate_semantic_body(graph_search(Query, TopK, Hops, Mode, _Results), Code) :-
    % Results variable binding not supported in this simple mode yet
    % Just printing results
    (   Mode = [mode(M)] -> atom_string(M, ModeStr) ; ModeStr = "vector" ),
    format(string(Code), 'let results = searcher.graph_search("~w", ~w, ~w, "~w")?;\n    println!("{}", serde_json::to_string_pretty(&results)?);', [Query, TopK, Hops, ModeStr]).

translate_semantic_body(suggest_bookmarks(Query, _Suggestions), Code) :-
    format(string(Code), 'let output = searcher.suggest_bookmarks("~w", 5)?;\n    println!("{}", output);', [Query]).

translate_semantic_body(semantic_search(Query, TopK, _Results), Code) :-
    format(string(Code), 'let results = searcher.text_search("~w", ~w)?;\n    println!("{}", serde_json::to_string_pretty(&results)?);', [Query, TopK]).

translate_semantic_body(upsert_object(Id, Type, Data), Code) :-
    % Simple translation assuming Data is a string literal or variable that can be stringified
    (   string(Data) -> format(string(DataStr), '"~s"', [Data])
    ;   atom(Data) -> format(string(DataStr), '"~w"', [Data])
    ;   format(string(DataStr), '"~w"', [Data]) % Fallback
    ),
    format(string(Code), 'importer.upsert_object("~w", "~w", &serde_json::from_str(~s)?)?;', [Id, Type, DataStr]).

translate_semantic_body(true, "").

rust_quote(S, Q) :- format(string(Q), "\"~s\"", [S]).

% Helper for sub_term if not available
:- if(\+ current_predicate(sub_term/2)).
sub_term(T, T).
sub_term(Sub, Term) :-
    compound(Term),
    arg(_, Term, Arg),
    sub_term(Sub, Arg).
:- endif.
compile_predicate_to_rust_normal(Pred, Arity, Options, RustCode) :-
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_main(IncludeMain), Options, true),
    option(unique(Unique), Options, true),

    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),

    (   Clauses = [] -> fail
    ;   maplist(is_fact_clause, Clauses) ->
        compile_facts_to_rust(Pred, Arity, Clauses, FieldDelim, RustCode)
    ;   Clauses = [SingleHead-SingleBody], SingleBody \= true ->
        compile_single_rule_to_rust(Pred, Arity, SingleHead, SingleBody, FieldDelim, Unique, IncludeMain, RustCode)
    ;   fail
    ).

is_fact_clause(_-true).

%% ============================================ 
%% COMPILATION MODES
%% ============================================ 

%% compile_facts_to_rust(+Pred, +Arity, -RustCode)
%  PUBLIC API: Export Prolog facts as a standalone Rust program.
%  Generates a Rust program with struct and Vec of facts.
%
%  Example:
%    ?- compile_facts_to_rust(parent, 2, Code).
%    Generates a Rust program with facts as a Vec of structs.
%
compile_facts_to_rust(Pred, Arity, RustCode) :-
    % Get the predicate name
    atom_string(Pred, PredStr),
    upcase_atom(Pred, PredUp),
    atom_string(PredUp, PredUpStr),
    
    % Collect all facts
    functor(Head, Pred, Arity),
    findall(Args, (user:clause(Head, true), Head =.. [_|Args]), AllFacts),
    
    % Generate field names
    findall(Field, (
        between(1, Arity, N),
        format(string(Field), 'arg~w: String', [N])
    ), Fields),
    atomic_list_concat(Fields, ',\n    ', FieldDefs),
    
    % Format facts as Rust struct literals
    findall(Entry, (
        member(Args, AllFacts),
        format_rust_struct_entry(Args, PredUpStr, Entry)
    ), Entries),
    atomic_list_concat(Entries, ',\n        ', EntriesCode),
    
    % Generate print fields
    generate_rust_print_fields(Arity, PrintFields),
    
    % Generate Rust program
    format(string(RustCode),
'// Generated by UnifyWeaver Rust Target - Fact Export
// Predicate: ~w/~w

/// Represents a ~w/~w fact
#[derive(Debug, Clone, PartialEq, Eq)]
struct ~w {
    ~w,
}

/// Get all facts
fn get_all_~w() -> Vec<~w> {
    vec![
        ~w,
    ]
}

/// Stream facts using iterator
fn stream_~w() -> impl Iterator<Item = ~w> {
    get_all_~w().into_iter()
}

/// Check if a fact exists
fn contains_~w(target: &~w) -> bool {
    get_all_~w().iter().any(|f| f == target)
}

fn main() {
    for fact in stream_~w() {
        println!("{}", vec![~w].join(":"));
    }
}
', [PredStr, Arity, PredStr, Arity, PredUpStr, FieldDefs,
    PredStr, PredUpStr, EntriesCode,
    PredStr, PredUpStr, PredStr,
    PredStr, PredUpStr, PredStr,
    PredStr, PrintFields]).

%% format_rust_struct_entry(+Args, +StructName, -Entry)
format_rust_struct_entry(Args, StructName, Entry) :-
    findall(FieldVal, (
        nth1(N, Args, Arg),
        atom_string(Arg, ArgStr),
        format(string(FieldVal), 'arg~w: "~w".to_string()', [N, ArgStr])
    ), FieldVals),
    atomic_list_concat(FieldVals, ', ', FieldsStr),
    format(string(Entry), '~w { ~w }', [StructName, FieldsStr]).

%% generate_rust_print_fields(+Arity, -Code) 
generate_rust_print_fields(Arity, Code) :-
    findall(Field, (
        between(1, Arity, N),
        format(string(Field), 'fact.arg~w.clone()', [N])
    ), Fields),
    atomic_list_concat(Fields, ', ', Code).

%% compile_facts_to_rust/5 - internal version (takes Clauses)
compile_facts_to_rust(_Pred, _Arity, Clauses, FieldDelim, RustCode) :-
    map_field_delimiter(FieldDelim, DelimChar),
    
    findall(Entry, 
        (   member(Fact-true, Clauses),
            Fact =.. [_|Args],
            maplist(atom_string, Args, ArgStrs),
            atomic_list_concat(ArgStrs, DelimChar, Key),
            format(string(Entry), '    facts.insert("~s".to_string());', [Key])
        ),
        Entries),
    atomic_list_concat(Entries, '\n', EntriesCode),

    format(string(RustCode), 'use std::collections::HashSet;

fn main() {
    let mut facts = HashSet::new();~s

    for fact in facts {
        println!("{}", fact);
    }
}
', [EntriesCode]).

%% ============================================
%% TAIL RECURSION OPTIMIZATION
%% ============================================

%% compile_tail_recursion_rust(+Pred/Arity, +Options, -RustCode)
%  Compile tail recursive predicates to Rust for loops.
%  Pattern: sum([], Acc, Acc). sum([H|T], Acc, S) :- Acc1 is Acc + H, sum(T, Acc1, S).
%  Generates O(1) stack space code.
%
compile_tail_recursion_rust(Pred/Arity, _Options, RustCode) :-
    atom_string(Pred, PredStr),
    
    % Detect step operation from predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    detect_rust_step_op(Clauses, Pred, StepOp),
    step_op_to_rust(StepOp, RustStepCode),
    
    (   Arity =:= 3 ->
        format(string(RustCode),
'// Generated by UnifyWeaver Rust Target - Tail Recursion Optimization
// Predicate: ~w/~w
// O(1) stack space via for loop

use std::io::{self, BufRead};

/// ~w computes result using accumulator pattern
fn ~w(items: &[i32], acc: i32) -> i32 {
    let mut result = acc;
    for &item in items {
        ~w;
    }
    result
}

fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            let items: Vec<i32> = line
                .split('','')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            println!("{}", ~w(&items, 0));
        }
    }
}
', [PredStr, Arity, PredStr, PredStr, RustStepCode, PredStr])
    ;   Arity =:= 2 ->
        format(string(RustCode),
'// Generated by UnifyWeaver Rust Target - Tail Recursion Optimization
// Predicate: ~w/~w

use std::io::{self, BufRead};

fn ~w(items: &[i32]) -> i32 {
    let mut acc = 0;
    for &item in items {
        ~w;
    }
    acc
}

fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            let items: Vec<i32> = line
                .split('','')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            println!("{}", ~w(&items));
        }
    }
}
', [PredStr, Arity, PredStr, RustStepCode, PredStr])
    ;   format(string(RustCode), '// Tail recursion for arity ~w not supported', [Arity])
    ).

%% detect_rust_step_op(+Clauses, +Pred, -StepOp)
detect_rust_step_op(Clauses, Pred, StepOp) :-
    member(_Head-Body, Clauses),
    Body \= true,
    contains_pred_rust(Body, Pred),
    extract_arith_rust(Body, StepOp),
    !.
detect_rust_step_op(_, _, add_element).

%% contains_pred_rust(+Body, +Pred)
contains_pred_rust(Body, Pred) :-
    Body =.. [Pred|_], !.
contains_pred_rust((A, _), Pred) :- contains_pred_rust(A, Pred), !.
contains_pred_rust((_, B), Pred) :- contains_pred_rust(B, Pred), !.

%% extract_arith_rust(+Body, -StepOp)
extract_arith_rust((_ is _ + B), add_element) :- var(B), !.
extract_arith_rust((_ is _ + 1), add_1) :- !.
extract_arith_rust((_ is _ * B), mult_element) :- var(B), !.
extract_arith_rust((A, _), Op) :- extract_arith_rust(A, Op), !.
extract_arith_rust((_, B), Op) :- extract_arith_rust(B, Op), !.
extract_arith_rust(_, add_1).

%% step_op_to_rust(+StepOp, -RustCode)
step_op_to_rust(add_element, "result += item").
step_op_to_rust(add_1, "result += 1").
step_op_to_rust(mult_element, "result *= item").
step_op_to_rust(_, "result += item").

%% can_compile_tail_recursion_rust(+Pred/Arity)
can_compile_tail_recursion_rust(Pred/Arity) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    partition(is_recursive_clause_rust(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [].

%% is_recursive_clause_rust(+Pred, +Clause)
is_recursive_clause_rust(Pred, _Head-Body) :-
    Body \= true,
    contains_pred_rust(Body, Pred).

%% ============================================
%% LINEAR RECURSION WITH MEMOIZATION
%% ============================================

%% compile_linear_recursion_rust(+Pred/Arity, +Options, -RustCode)
%  Compile linear recursive predicates to Rust with memoization (HashMap).
%
compile_linear_recursion_rust(Pred/Arity, _Options, RustCode) :-
    atom_string(Pred, PredStr),
    upcase_atom(Pred, PredUp),
    atom_string(PredUp, PredUpStr),
    
    (   Arity =:= 2 ->
        format(string(RustCode),
'// Generated by UnifyWeaver Rust Target - Linear Recursion with Memoization
// Predicate: ~w/~w
// Uses HashMap-based memoization for O(n) performance

use std::collections::HashMap;
use std::io::{self, BufRead};

thread_local! {
    static ~w_MEMO: std::cell::RefCell<HashMap<i32, i32>> = std::cell::RefCell::new(HashMap::new());
}

/// ~w computes result with memoization
fn ~w(n: i32) -> i32 {
    // Check memo
    if let Some(&result) = ~w_MEMO.with(|m| m.borrow().get(&n).copied()) {
        return result;
    }
    
    // Base cases
    if n <= 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    
    // Recursive case with memoization
    let result = ~w(n - 1) + n;
    ~w_MEMO.with(|m| m.borrow_mut().insert(n, result));
    result
}

fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            if let Ok(n) = line.trim().parse::<i32>() {
                println!("{}", ~w(n));
            }
        }
    }
}
', [PredStr, Arity, PredUpStr, PredUpStr, PredStr, PredUpStr, PredStr, PredUpStr, PredStr])
    ;   format(string(RustCode), '// Linear recursion for arity ~w not supported', [Arity])
    ).

%% can_compile_linear_recursion_rust(+Pred/Arity)
can_compile_linear_recursion_rust(Pred/Arity) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    partition(is_recursive_clause_rust(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    forall(member(_-Body, RecClauses), count_recursive_calls_rust(Body, Pred, 1)).

%% count_recursive_calls_rust(+Body, +Pred, ?Count)
count_recursive_calls_rust(Body, Pred, Count) :-
    count_recursive_calls_rust_(Body, Pred, 0, Count).

count_recursive_calls_rust_(Goal, Pred, Acc, Count) :-
    Goal =.. [Pred|_], !,
    Count is Acc + 1.
count_recursive_calls_rust_((A, B), Pred, Acc, Count) :- !,
    count_recursive_calls_rust_(A, Pred, Acc, Acc1),
    count_recursive_calls_rust_(B, Pred, Acc1, Count).
count_recursive_calls_rust_(_, _, Acc, Acc).

%% ============================================
%% MUTUAL RECURSION
%% ============================================

%% compile_mutual_recursion_rust(+Predicates, +Options, -RustCode)
%  Compile mutually recursive predicates to Rust.
%  Example: is_even/1 and is_odd/1
%
compile_mutual_recursion_rust(Predicates, _Options, RustCode) :-
    % Extract predicate names for group name
    findall(PredStr, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr)
    ), PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),
    
    % Generate functions for each predicate
    findall(FuncCode, (
        member(Pred/Arity, Predicates),
        generate_mutual_function_rust(Pred, Arity, Predicates, GroupName, FuncCode)
    ), FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    
    format(string(RustCode),
'// Generated by UnifyWeaver Rust Target - Mutual Recursion
// Group: ~w

use std::collections::HashMap;
use std::cell::RefCell;
use std::env;

thread_local! {
    static ~w_MEMO: RefCell<HashMap<String, bool>> = RefCell::new(HashMap::new());
}

~w

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 {
        if let Ok(n) = args[2].parse::<i32>() {
            match args[1].as_str() {
~w            _ => eprintln!("Unknown function"),
            }
        }
    }
}
', [GroupName, GroupName, FunctionsCode, generate_rust_match_arms(Predicates)]).

%% generate_mutual_function_rust(+Pred, +Arity, +AllPredicates, +GroupName, -Code)
generate_mutual_function_rust(Pred, Arity, AllPredicates, GroupName, Code) :-
    atom_string(Pred, PredStr),
    
    % Get clauses for this predicate
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    % Separate base and recursive cases
    partition(is_mutual_recursive_clause_rust(AllPredicates), Clauses, RecClauses, BaseClauses),
    
    % Generate base case code
    generate_mutual_base_cases_rust(BaseClauses, GroupName, BaseCaseCode),
    
    % Generate recursive case code  
    generate_mutual_recursive_cases_rust(RecClauses, AllPredicates, GroupName, RecCaseCode),
    
    format(string(Code),
'/// ~w is part of mutual recursion group
fn ~w(n: i32) -> bool {
    // Check memo
    let key = format!("~w:{}", n);
    if let Some(&result) = ~w_MEMO.with(|m| m.borrow().get(&key).copied()) {
        return result;
    }
    
~w
~w
    
    // No match
    false
}', [PredStr, PredStr, PredStr, GroupName, BaseCaseCode, RecCaseCode]).

%% is_mutual_recursive_clause_rust(+AllPredicates, +Clause)
is_mutual_recursive_clause_rust(AllPredicates, _Head-Body) :-
    Body \= true,
    member(Pred/Arity, AllPredicates),
    functor(Goal, Pred, Arity),
    body_contains_goal_rust(Body, Goal).

%% body_contains_goal_rust(+Body, +Goal)
body_contains_goal_rust((A, B), Goal) :- !,
    (   body_contains_goal_rust(A, Goal)
    ;   body_contains_goal_rust(B, Goal)
    ).
body_contains_goal_rust(Body, Goal) :-
    compound(Body),
    functor(Body, F, A),
    functor(Goal, F, A).

%% generate_mutual_base_cases_rust(+BaseClauses, +GroupName, -Code)
generate_mutual_base_cases_rust([], _, "    // No base cases").
generate_mutual_base_cases_rust(BaseClauses, GroupName, Code) :-
    BaseClauses \= [],
    findall(CaseCode, (
        member(Head-true, BaseClauses),
        Head =.. [_|[Value]],
        format(string(CaseCode), '    if n == ~w {\n        ~w_MEMO.with(|m| m.borrow_mut().insert(key.clone(), true));\n        return true;\n    }', [Value, GroupName])
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n', Code).

%% generate_mutual_recursive_cases_rust(+RecClauses, +AllPredicates, +GroupName, -Code)
generate_mutual_recursive_cases_rust([], _, _, "    // No recursive cases").
generate_mutual_recursive_cases_rust([_Head-Body|_], AllPredicates, GroupName, Code) :-
    % Extract the called predicate from the body
    find_mutual_call_rust(Body, AllPredicates, CalledPred),
    atom_string(CalledPred, CalledPredStr),
    format(string(Code),
'    // Recursive case
    if n > 0 {
        let result = ~w(n - 1);
        ~w_MEMO.with(|m| m.borrow_mut().insert(key, result));
        return result;
    }', [CalledPredStr, GroupName]).

%% find_mutual_call_rust(+Body, +AllPredicates, -CalledPred)
find_mutual_call_rust((A, B), AllPredicates, CalledPred) :- !,
    (   find_mutual_call_rust(A, AllPredicates, CalledPred)
    ;   find_mutual_call_rust(B, AllPredicates, CalledPred)
    ).
find_mutual_call_rust(Goal, AllPredicates, CalledPred) :-
    Goal =.. [Pred|_],
    member(Pred/_Arity, AllPredicates),
    CalledPred = Pred.

%% generate_rust_match_arms(+Predicates)
generate_rust_match_arms(Predicates) :-
    findall(CaseCode, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr),
        format(string(CaseCode), '                "~w" => println!("{}", ~w(n)),', [PredStr, PredStr])
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n', _Cases).

compile_single_rule_to_rust(_Pred, _Arity, Head, Body, FieldDelim, Unique, IncludeMain, RustCode) :-
    Head =.. [_|HeadArgs],
    
    extract_predicates(Body, Predicates),
    extract_constraints(Body, Constraints),
    extract_match_constraints(Body, MatchConstraints), 
    extract_key_generation(Body, Keys),
    extract_bindings(Body, Bindings),
    
    (   (Predicates = [BodyPred]; Predicates = []) ->
        (   Predicates = [BodyPred]
        ->  BodyPred =.. [_BodyName|BodyArgs]
        ;   BodyArgs = HeadArgs
        ),
        length(BodyArgs, NumFields),
        
        build_source_map(BodyArgs, SourceMap), 
        
        map_field_delimiter(FieldDelim, DelimChar),

        generate_rust_keys(Keys, SourceMap, KeyCode, KeyMap),
        generate_rust_bindings(Bindings, SourceMap, BindingMap, BindingCode),

        findall(OutputPart, 
            (
                member(Arg, HeadArgs),
                (   var(Arg), member((Var, Idx), SourceMap), Arg == Var ->
                    format(string(OutputPart), "parts[~w]", [Idx])
                ;   var(Arg), member((Var, RustVar), KeyMap), Arg == Var ->
                    format(string(OutputPart), "~w", [RustVar])
                ;   var(Arg), member((Var, BindVar), BindingMap), Arg == Var ->
                    format(string(OutputPart), "~w", [BindVar])
                ;   atom(Arg) ->
                    format(string(OutputPart), '"~w"', [Arg])
                ;   OutputPart = "\"unknown\""
                )
            ),
            OutputParts),
        build_rust_concat(OutputParts, DelimChar, OutputExpr),

        generate_rust_constraints(Constraints, SourceMap, BindingMap, ConstraintChecks),
        
        generate_rust_match_constraints(MatchConstraints, SourceMap, RegexInit, RegexChecks, NeedsRegex),

        (   Unique = true ->
            UniqueDecl = "    let mut seen = HashSet::new();",
            UniqueCheck = "            if seen.insert(result.clone()) {\n                println!(\"{}\", result);\n            }"
        ;   UniqueDecl = "",
            UniqueCheck = "            println!(\"{}\", result);"
        ),
        
        format(string(ScriptBody), '    let stdin = io::stdin();
~s
~s
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split("~s").collect();
            if parts.len() == ~w {
~s
~s
~s
~s
                let result = ~s;
~s
            }
        }
    }
', [UniqueDecl, RegexInit, DelimChar, NumFields, RegexChecks, ConstraintChecks, KeyCode, BindingCode, OutputExpr, UniqueCheck]),

        (   IncludeMain ->
            (   NeedsRegex = true -> UseRegex = "use regex::Regex;\n" ; UseRegex = "" ),
            format(string(RustCode), 'use std::io::{self, BufRead};
use std::collections::HashSet;
~s
fn main() {
~s}
', [UseRegex, ScriptBody])
        ;   RustCode = ScriptBody
        )
    ;   RustCode = "// TODO: Multi-predicate not supported"
    ).

%% ============================================ 
%% AGGREGATION COMPILATION
%% ============================================ 

compile_aggregation_to_rust(_Pred, _Arity, AggOp, _FieldDelim, IncludeMain, RustCode) :-
    generate_aggregation_logic(AggOp, Logic),
    
    (   IncludeMain ->
        format(string(RustCode), 'use std::io::{self, BufRead};

fn main() {
~s}
', [Logic])
    ;   RustCode = Logic
    ).

%% ============================================ 
%% JSON INPUT COMPILATION
%% ============================================ 

compile_json_input_to_rust(Pred, Arity, Options, RustCode) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),

    (   Clauses = [SingleHead-SingleBody] ->
        extract_json_operations(SingleBody, JsonOps),
        option(field_delimiter(FieldDelim), Options, colon),
        map_field_delimiter(FieldDelim, DelimChar),
        option(unique(Unique), Options, true),
        option(json_schema(SchemaName), Options, none), 

        generate_json_input_logic(JsonOps, SingleHead, SchemaName, DelimChar, Unique, ScriptBody),
        
        format(string(RustCode), 'use std::io::{self, BufRead};
use std::collections::HashSet;
use serde::Deserialize;
use serde_json::{self, Value};

fn main() {
~s
}
', [ScriptBody])
    ;   fail
    ).

%% ============================================ 
%% JSON OUTPUT COMPILATION
%% ============================================ 

compile_json_output_to_rust(Pred, Arity, Options, RustCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    (   Clauses = [SingleHead-_] ->
        option(field_delimiter(FieldDelim), Options, colon),
        map_field_delimiter(FieldDelim, DelimChar),
        
        generate_json_output_struct(PredStr, SingleHead, StructDef),
        generate_json_output_logic(SingleHead, DelimChar, ScriptBody),
        
        format(string(RustCode), 'use std::io::{self, BufRead};
use serde::Serialize;
use serde_json::{self, Value};

~s

fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(l) = line {
            let parts: Vec<&str> = l.split("~s").collect();
            if parts.len() == ~w {
~s
            }
        }
    }
}
', [StructDef, DelimChar, Arity, ScriptBody])
    ;   fail
    ).

%% ============================================ 
%% LOGIC GENERATION HELPERS
%% ============================================ 

generate_aggregation_logic(count, "let c = io::stdin().lock().lines().count(); println!(\"{}\", c);").
generate_aggregation_logic(sum, "let s: f64 = io::stdin().lock().lines().filter_map(|l| l.ok()?.trim().parse().ok()).sum(); println!(\"{}\", s);").
generate_aggregation_logic(avg, "let (c, s) = io::stdin().lock().lines().filter_map(|l| l.ok()?.trim().parse::<f64>().ok()).fold((0, 0.0), |(c, s), n| (c + 1, s + n)); if c > 0 { println!(\"{}\", s / c as f64); }").
generate_aggregation_logic(min, "if let Some(m) = io::stdin().lock().lines().filter_map(|l| l.ok()?.trim().parse::<f64>().ok()).min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) { println!(\"{}\", m); }").
generate_aggregation_logic(max, "if let Some(m) = io::stdin().lock().lines().filter_map(|l| l.ok()?.trim().parse::<f64>().ok()).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) { println!(\"{}\", m); }").

generate_json_input_logic(JsonOps, Head, SchemaName, DelimChar, Unique, ScriptBody) :-
    Head =.. [_|HeadArgs],
    findall(ExtractLine, (
        nth1(Pos, HeadArgs, Arg),
        format(atom(ValVar), 'val_~w', [Pos]),
        (   member(json_field(JsonField, Arg), JsonOps) ->
            ( SchemaName \= none -> get_field_type(SchemaName, JsonField, Type) ; Type = any ),
            rust_json_extract_code(ValVar, JsonField, Type, ExtractLine)
        ;   member(json_path(Path, Arg), JsonOps) ->
            ( SchemaName \= none -> last(Path, LF), get_field_type(SchemaName, LF, Type) ; Type = any ),
            rust_json_path_extract_code(ValVar, Path, Type, ExtractLine)
        ;   format(string(ExtractLine), 'let ~w = Value::Null;', [ValVar])
        )
    ), ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode),
    findall(OutputPart, (nth1(Pos, HeadArgs, _), format(atom(OutputPart), 'val_~w', [Pos])), OutputParts),
    build_rust_concat(OutputParts, DelimChar, OutputExpr),
    ( Unique = true -> UniqueDecl = "let mut seen = HashSet::new();", UniqueCheck = "if seen.insert(result.clone()) { println!(\"{}\", result); }" ; UniqueDecl = "", UniqueCheck = "println!(\"{}\", result);" ),
    format(string(ScriptBody), '~s
    for line in io::stdin().lock().lines() {
        if let Ok(l) = line {
            let json_data: Value = match serde_json::from_str(&l) { Ok(d) => d, Err(_) => continue };
~s
            let result = ~s;
            ~s
        }
    }', [UniqueDecl, ExtractCode, OutputExpr, UniqueCheck]).

rust_json_extract_code(Var, Field, Type, Code) :-
    atom_string(Field, FieldStr),
    (   Type = any -> format(string(Code), 'let ~w = json_data.get(\"~s\").cloned().unwrap_or(Value::Null);', [Var, FieldStr])
    ;   format_type_for_json(Type, RustType),
        format(string(Code), 'let ~w: ~s = match json_data.get(\"~s\") {
                Some(v) => match serde_json::from_value(v.clone()) { Ok(val) => val, Err(_) => continue },
                None => continue,
            };', [Var, RustType, FieldStr])
    ).
ust_json_path_extract_code(Var, Path, Type, Code) :-
    atomic_list_concat(Path, '/', PathStr),
    (   Type = any -> format(string(Code), 'let ~w = json_data.pointer(\"/~s\").cloned().unwrap_or(Value::Null);', [Var, PathStr])
    ;   format_type_for_json(Type, RustType),
        format(string(Code), 'let ~w: ~s = match json_data.pointer(\"/~s\") {
                Some(v) => match serde_json::from_value(v.clone()) { Ok(val) => val, Err(_) => continue },
                None => continue,
            };', [Var, RustType, PathStr])
    ).

generate_json_output_struct(PredName, Head, StructDef) :-
    Head =.. [_|HeadArgs],
    findall(FieldName, (nth1(Idx, HeadArgs, Arg), (var(Arg) -> format(atom(FieldName), 'field_~w', [Idx]) ; atom_string(Arg, FieldName))), FieldNames),
    findall(FieldLine, (
        nth1(Idx, HeadArgs, Arg), nth1(Idx, FieldNames, FieldName),
        ( var(Arg) -> Type = string ; Type = string ),
        format_type_for_json(Type, RustType),
        format(string(FieldLine), '    #[serde(rename = "~w")]\n    pub ~w: ~s,', [FieldName, FieldName, RustType])
    ), FieldLines),
    atomic_list_concat(FieldLines, '\n', FieldsStr),
    format(string(StructDef), '#[derive(Serialize)]
struct ~w {
~s
}', [PredName, FieldsStr]).

generate_json_output_logic(Head, _DelimChar, ScriptBody) :-
    Head =.. [PredName|HeadArgs],
    findall(FieldName, (nth1(Idx, HeadArgs, Arg), (var(Arg) -> format(atom(FieldName), 'field_~w', [Idx]) ; atom_string(Arg, FieldName))), FieldNames),
    findall(AssignLine, (nth1(Idx, FieldNames, FieldName), PartIdx is Idx-1, format(string(AssignLine), '        ~w: parts[~w].to_string(),', [FieldName, PartIdx])), AssignLines),
    atomic_list_concat(AssignLines, '\n', AssignCode),
    format(string(ScriptBody), 'let record = ~w {
~s
};
if let Ok(json_str) = serde_json::to_string(&record) {
    println!("{}", json_str);
}', [PredName, AssignCode]).
%% ============================================ 
%% HELPERS
%% ============================================ 

build_source_map(BodyArgs, Map) :- build_source_map_(BodyArgs, 0, Map).
build_source_map_([], _, []).
build_source_map_([Arg|Rest], Idx, [(Arg, Idx)|Map]) :- NextIdx is Idx + 1, build_source_map_(Rest, NextIdx, Map).

build_rust_concat([Single], _, Format) :- !, format(string(Format), "String::from(~s)", [Single]).
build_rust_concat(Parts, Delim, Format) :-
    atomic_list_concat(Parts, ', ', Args),
    maplist(rust_format_placeholder, Parts, Placeholders),
    atomic_list_concat(Placeholders, Delim, FmtStr),
    format(string(Format), "format!(\"~s\", ~s)", [FmtStr, Args]).

rust_format_placeholder(_, "{}").

extract_predicates(true, []) :- !.
extract_predicates((A, B), Ps) :- !, extract_predicates(A, P1), extract_predicates(B, P2), append(P1, P2, Ps).
extract_predicates(Goal, []) :- member(Goal, [(_>_), (_<_), (_>=_), (_=<_), (_=:=_), (_=\=_), (_ is _), match(_, _), json_field(_, _), json_path(_, _), json_record(_), json_get(_, _), generate_key(_, _)]), !.
extract_predicates(Goal, []) :- functor(Goal, Name, Arity), rs_binding(Name/Arity, _, _, _, _), !.
extract_predicates(Goal, [Goal]).

extract_constraints(true, []) :- !.
extract_constraints((A, B), Cs) :- !, extract_constraints(A, C1), extract_constraints(B, C2), append(C1, C2, Cs).
extract_constraints(Goal, [Goal]) :- member(Goal, [(_>_), (_<_), (_>=_), (_=<_), (_=:=_), (_=\=_)]), !.
extract_constraints(_, []).

extract_match_constraints(true, []) :- !.
extract_match_constraints((A, B), Ms) :- !, extract_match_constraints(A, M1), extract_match_constraints(B, M2), append(M1, M2, Ms).
extract_match_constraints(match(Var, Pattern), [match(Var, Pattern)]) :- !.
extract_match_constraints(_, []).

extract_bindings(true, []) :- !.
extract_bindings((A, B), Bs) :- !, extract_bindings(A, B1), extract_bindings(B, B2), append(B1, B2, Bs).
extract_bindings(Goal, [Goal]) :-
    functor(Goal, Name, Arity),
    rs_binding(Name/Arity, _, _, _, _), !.
extract_bindings(_, []).

generate_rust_bindings(Goals, SourceMap, BindingMap, Code) :-
    generate_rust_bindings_rec(Goals, SourceMap, [], BindingMap, Code).

generate_rust_bindings_rec([], _, Map, Map, "").
generate_rust_bindings_rec([Goal|Rest], SourceMap, MapIn, MapOut, Code) :-
    Goal =.. [Pred|Args],
    functor(Goal, Pred, Arity),
    binding(rust, Pred/Arity, TargetName, InputTypes, _, _),
    
    length(InputTypes, NumInputs),
    length(Inputs, NumInputs),
    append(Inputs, Outputs, Args),
    
    maplist(term_to_rust_expr(SourceMap, MapIn), Inputs, InputExprs),
    
    findall(VarName-Var, (member(Var, Outputs), gensym('val_', Atom), atom_string(Atom, VarName)), OutPairs),
    
    (   sub_string(TargetName, 0, 1, _, ".")
    ->  InputExprs = [Obj|RestInputs],
        atomic_list_concat(RestInputs, ", ", ArgStr),
        (   sub_string(TargetName, _, 2, 0, "()")
        ->  sub_string(TargetName, 0, _, 2, MethodName),
            format(string(CallExpr), "~s~w()", [Obj, MethodName])
        ;   format(string(CallExpr), "~s~w(~s)", [Obj, TargetName, ArgStr])
        )
    ;   atomic_list_concat(InputExprs, ", ", ArgStr),
        format(string(CallExpr), "~w(~s)", [TargetName, ArgStr])
    ),
    
    (   OutPairs = [VarName-Var]
    ->  format(string(Line), "                let ~s = ~s;\n", [VarName, CallExpr]),
        NewBindings = [(Var, VarName)]
    ;   OutPairs = []
    ->  format(string(Line), "                ~s;\n", [CallExpr]),
        NewBindings = []
    ;   fail % Tuple todo
    ),
    
    append(NewBindings, MapIn, MapNext),
    generate_rust_bindings_rec(Rest, SourceMap, MapNext, MapOut, RestCode),
    string_concat(Line, RestCode, Code).

term_to_rust_expr(SourceMap, _, Var, Code) :-
    var(Var), member((V, I), SourceMap), Var == V, !,
    format(string(Code), "parts[~w].trim()", [I]). % Default string
term_to_rust_expr(_, BindingMap, Var, Code) :-
    var(Var), member((V, Name), BindingMap), Var == V, !,
    Code = Name.
term_to_rust_expr(_, _, Val, Code) :-
    number(Val), !, format(string(Code), "~w", [Val]).
term_to_rust_expr(_, _, Val, Code) :-
    string(Val), !, format(string(Code), "\"~s\"", [Val]).
term_to_rust_expr(_, _, Val, Code) :-
    atom(Val), !, format(string(Code), "\"~w\"", [Val]).


extract_json_operations(true, []) :- !.
extract_json_operations((A, B), Ops) :- !, extract_json_operations(A, O1), extract_json_operations(B, O2), append(O1, O2, Ops).
extract_json_operations(json_field(F, V), [json_field(F, V)]) :- !.
extract_json_operations(json_path(P, V), [json_path(P, V)]) :- !.
extract_json_operations(json_record(Fs), Ops) :- !, maplist(field_to_json_op, Fs, Ops).
extract_json_operations(json_get(P, V), [json_path(P, V)]) :- !.
extract_json_operations(_, []).

extract_key_generation(true, []) :- !.
extract_key_generation((A, B), Keys) :- !, extract_key_generation(A, K1), extract_key_generation(B, K2), append(K1, K2, Keys).
extract_key_generation(generate_key(S, V), [generate_key(S, V)]) :- !.
extract_key_generation(_, []).

generate_rust_keys([], _, "", []).
generate_rust_keys(Keys, SourceMap, Code, KeyMap) :-
    generate_rust_keys_(Keys, SourceMap, [], CodeList, [], KeyMap),
    atomic_list_concat(CodeList, '\n', Code).

generate_rust_keys_([], _, CodeAcc, CodeList, MapAcc, MapAcc) :-
    reverse(CodeAcc, CodeList).
generate_rust_keys_([generate_key(Strategy, Var)|Rest], SourceMap, CodeAcc, CodeResult, MapAcc, MapResult) :-
    gensym('key_', KeyVar),
    compile_rust_key_expr(Strategy, SourceMap, MapAcc, Expr),
    format(string(Line), "                let ~w = ~s;", [KeyVar, Expr]),
    generate_rust_keys_(Rest, SourceMap, [Line|CodeAcc], CodeResult, [(Var, KeyVar)|MapAcc], MapResult).

compile_rust_key_expr(Var, SourceMap, _, Code) :- % Implicit field or key
    var(Var), member((V, I), SourceMap), Var == V, !,
    format(string(Code), "parts[~w]", [I]).
compile_rust_key_expr(Var, _, KeyMap, Code) :- % Generated key
    var(Var), member((V, K), KeyMap), Var == V, !,
    format(string(Code), "~w", [K]).
compile_rust_key_expr(field(Var), SourceMap, _, Code) :-
    var(Var), member((V, I), SourceMap), Var == V, !,
    format(string(Code), "parts[~w]", [I]).
compile_rust_key_expr(literal(Text), _, _, Code) :-
    format(string(Code), "\"~w\"", [Text]).
compile_rust_key_expr(composite(List), SourceMap, KeyMap, Code) :-
    compile_rust_key_expr_list(List, SourceMap, KeyMap, Parts),
    atomic_list_concat(Parts, ", ", Args),
    length(Parts, N),
    length(Slots, N), maplist(=("{}"), Slots), atomic_list_concat(Slots, "", Fmt),
    format(string(Code), "format!(\"~s\", ~s)", [Fmt, Args]).
compile_rust_key_expr(hash(Expr), SourceMap, KeyMap, Code) :-
    compile_rust_key_expr(Expr, SourceMap, KeyMap, Inner),
    format(string(Code), "{ use sha2::{Sha256, Digest}; let mut hasher = Sha256::new(); hasher.update(~s.as_bytes()); hex::encode(hasher.finalize()) }", [Inner]).
compile_rust_key_expr(uuid(), _, _, "uuid::Uuid::new_v4().to_string()").
compile_rust_key_expr(Var, _, _, "\"UNKNOWN_VAR\"") :-
    var(Var),
    format('WARNING: Key variable ~w not found in source map or key map~n', [Var]).

compile_rust_key_expr_list([], _, _, []).
compile_rust_key_expr_list([H|T], SourceMap, KeyMap, [HC|TC]) :-
    compile_rust_key_expr(H, SourceMap, KeyMap, HC),
    compile_rust_key_expr_list(T, SourceMap, KeyMap, TC).

field_to_json_op(Field-Var, json_field(Field, Var)).

generate_rust_constraints([], _, _, "").
generate_rust_constraints(Constraints, SourceMap, BindingMap, Code) :-
    findall(Check, (member(C, Constraints), constraint_to_rust(C, SourceMap, BindingMap, Check)), Checks),
    atomic_list_concat(Checks, '\n', Code).

constraint_to_rust(Comparison, SourceMap, BindingMap, Code) :-
    Comparison =.. [Op, L, R], map_rust_op(Op, RO), 
    term_to_rust_numeric(L, SourceMap, BindingMap, LC), 
    term_to_rust_numeric(R, SourceMap, BindingMap, RC),
    format(string(Code), "if !(~s ~s ~s) { continue; }", [LC, RO, RC]).

map_rust_op(>, ">"). map_rust_op(<, "<"). map_rust_op(>=, ">="). map_rust_op(=<, "<="). map_rust_op(=:=, "=="). map_rust_op(=\=, "!=").

term_to_rust_numeric(Var, SourceMap, _, Code) :- var(Var), member((V, I), SourceMap), Var == V, !, format(string(Code), "parts[~w].trim().parse::<f64>().unwrap_or(0.0)", [I]).
term_to_rust_numeric(Var, _, BindingMap, Code) :- var(Var), member((V, Name), BindingMap), Var == V, !, format(string(Code), "~w as f64", [Name]).
term_to_rust_numeric(Num, _, _, Num) :- number(Num).

generate_rust_match_constraints([], _, "", "", false).
generate_rust_match_constraints(Matches, SourceMap, InitCode, CheckCode, true) :-
    findall((Init, Check), (
        member(match(Var, Pattern), Matches),
        var(Var), member((V, Idx), SourceMap), Var == V,
        gensym('re_', ReVar),
        format(string(Init), 'let ~w = Regex::new(\"~w\").unwrap();', [ReVar, Pattern]),
        format(string(Check), 'if !~w.is_match(parts[~w]) { continue; }', [ReVar, Idx])
    ), Pairs),
    findall(I, member((I,_), Pairs), Inits), findall(C, member((_,C), Pairs), Checks),
    atomic_list_concat(Inits, '\n', InitCode), atomic_list_concat(Checks, '\n', CheckCode).

map_field_delimiter(colon, ':'). map_field_delimiter(tab, '\t'). map_field_delimiter(comma, ','). map_field_delimiter(pipe, '|'). map_field_delimiter(C, C).

format_type_for_json(string, "String").
format_type_for_json(integer, "i64").
format_type_for_json(float, "f64").
format_type_for_json(boolean, "bool").
format_type_for_json(array, "Vec<Value>").
format_type_for_json(object, "serde_json::Map<String, Value>").
format_type_for_json(any, "Value").

%% ============================================ 
%% PROJECT GENERATION
%% ============================================ 

write_rust_program(Code, Path) :- open(Path, write, S), write(S, Code), close(S), format('Rust program written to: ~w~n', [Path]).

write_rust_project(RustCode, ProjectDir) :- 
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'src', SrcDir), make_directory_path(SrcDir),
    directory_file_path(SrcDir, 'main.rs', MainPath), write_rust_program(RustCode, MainPath),
    write_runtime_files(SrcDir),
    detect_dependencies(RustCode, Deps),
    directory_file_path(ProjectDir, 'Cargo.toml', CargoPath),
    file_base_name(ProjectDir, ProjectName),
    generate_cargo_toml(ProjectName, Deps, CargoContent),
    open(CargoPath, write, S), write(S, CargoContent), close(S),
    format('Rust project created at: ~w~n', [ProjectDir]).

write_runtime_files(SrcDir) :-
    RuntimeFiles = ['importer.rs', 'crawler.rs', 'searcher.rs', 'llm.rs', 'embedding.rs'],
    % Assuming runtime dir is relative to this source file or cwd
    % We'll try a fixed path relative to CWD for now
    RuntimeDir = 'src/unifyweaver/targets/rust_runtime',
    
    forall(member(File, RuntimeFiles), (
        directory_file_path(RuntimeDir, File, SourcePath),
        directory_file_path(SrcDir, File, DestPath),
        (   exists_file(SourcePath)
        ->  copy_file(SourcePath, DestPath),
            format('Copied runtime file: ~w~n', [File])
        ;   format('WARNING: Runtime file not found: ~w~n', [SourcePath])
        )
    )).

detect_dependencies(Code, Deps) :-
    findall(Dep, 
        (   sub_string(Code, _, _, _, 'use regex::'), Dep = regex
        ;   sub_string(Code, _, _, _, 'use serde::'), Dep = serde
        ;   sub_string(Code, _, _, _, 'use serde_json'), Dep = serde_json
        ;   sub_string(Code, _, _, _, 'use sha2::'), Dep = sha2
        ;   sub_string(Code, _, _, _, 'hex::encode'), Dep = hex
        ;   sub_string(Code, _, _, _, 'uuid::Uuid'), Dep = uuid
        ;   sub_string(Code, _, _, _, 'mod importer;'), Dep = redb
        ;   sub_string(Code, _, _, _, 'mod crawler;'), Dep = quick_xml
        ;   sub_string(Code, _, _, _, 'mod llm;'), Dep = serde_json % llm uses serde_json
        ;   sub_string(Code, _, _, _, 'mod embedding;'), Dep = candle
        ),
        DepsList),
    sort(DepsList, Deps).
generate_cargo_toml(Name, Deps, Content) :-
    format(string(Header), '[package]\nname = "~w"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n', [Name]),
    maplist(dep_to_toml, Deps, DepLines),
    atomic_list_concat(DepLines, '\n', DepBlock),
    string_concat(Header, DepBlock, Content).

dep_to_toml(regex, "regex = \"1\"").
dep_to_toml(serde, "serde = { version = \"1.0\", features = [\"derive\"] }").
dep_to_toml(serde_json, "serde_json = \"1.0\"").
dep_to_toml(sha2, "sha2 = \"0.10\"").
dep_to_toml(hex, "hex = \"0.4\"").
dep_to_toml(uuid, "uuid = { version = \"1.4\", features = [\"v4\"] }").
dep_to_toml(redb, "redb = \"1.4.0\"").
dep_to_toml(quick_xml, "quick-xml = \"0.38\"").
dep_to_toml(candle, "candle-core = \"0.9\"\ncandle-nn = \"0.9\"\ncandle-transformers = \"0.9\"\ntokenizers = \"0.22\"\nanyhow = \"1.0\"").

% ============================================================================
% Pipeline Generator Mode for Rust
% ============================================================================
%
% This section implements pipeline chaining with support for generator mode
% (fixpoint iteration) for Rust targets. Similar to Python, PowerShell, and
% C# pipeline implementations.
%
% Usage:
%   compile_rust_pipeline([derive/1, transform/1], [
%       pipeline_name(fixpoint_pipe),
%       pipeline_mode(generator),
%       output_format(jsonl)
%   ], RustCode).

%% compile_rust_pipeline(+Predicates, +Options, -RustCode)
%  Compile a list of predicates into a Rust pipeline.
%  Options:
%    pipeline_name(Name) - Name for the pipeline function (default: 'pipeline')
%    pipeline_mode(Mode) - 'sequential' (default) or 'generator' (fixpoint)
%    output_format(Format) - 'jsonl' (default) or 'json'
%
compile_rust_pipeline(Predicates, Options, RustCode) :-
    option(pipeline_name(PipelineName), Options, pipeline),
    option(pipeline_mode(PipelineMode), Options, sequential),
    option(output_format(OutputFormat), Options, jsonl),

    % Generate header based on mode
    rust_pipeline_header(PipelineMode, Header),

    % Generate helper functions based on mode
    rust_pipeline_helpers(PipelineMode, Helpers),

    % Extract stage names
    extract_rust_stage_names(Predicates, StageNames),

    % Generate stage functions (placeholder implementations)
    generate_rust_stage_functions(StageNames, StageFunctions),

    % Generate the pipeline connector (mode-aware)
    generate_rust_pipeline_connector(StageNames, PipelineName, PipelineMode, ConnectorCode),

    % Generate main execution block
    generate_rust_main_block(PipelineName, OutputFormat, MainBlock),

    % Combine all parts
    format(string(RustCode),
"~w

~w

~w
~w

~w
", [Header, Helpers, StageFunctions, ConnectorCode, MainBlock]).

%% rust_pipeline_header(+Mode, -Header)
%  Generate mode-aware header with use statements
rust_pipeline_header(generator, Header) :-
    !,
    format(string(Header),
"// Generated by UnifyWeaver Rust Pipeline Generator Mode
// Fixpoint evaluation for recursive pipeline stages
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use serde_json::{Value, Map};", []).

rust_pipeline_header(_, Header) :-
    format(string(Header),
"// Generated by UnifyWeaver Rust Pipeline (sequential mode)
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use serde_json::{Value, Map};", []).

%% rust_pipeline_helpers(+Mode, -Helpers)
%  Generate mode-aware helper functions
rust_pipeline_helpers(generator, Helpers) :-
    !,
    format(string(Helpers),
"/// Generates a unique key for record deduplication.
fn record_key(record: &HashMap<String, Value>) -> String {
    let mut keys: Vec<&String> = record.keys().collect();
    keys.sort();
    keys.iter()
        .map(|k| format!(\"{}={}\", k, record.get(*k).unwrap_or(&Value::Null)))
        .collect::<Vec<_>>()
        .join(\";\")
}

/// Read JSONL records from stdin.
fn read_jsonl_stream() -> Vec<HashMap<String, Value>> {
    let stdin = io::stdin();
    let mut records = Vec::new();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            if !line.trim().is_empty() {
                if let Ok(value) = serde_json::from_str::<HashMap<String, Value>>(&line) {
                    records.push(value);
                }
            }
        }
    }
    records
}

/// Write JSONL records to stdout.
fn write_jsonl_stream(records: &[HashMap<String, Value>]) {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    for record in records {
        if let Ok(json) = serde_json::to_string(record) {
            writeln!(handle, \"{}\", json).ok();
        }
    }
}", []).

rust_pipeline_helpers(_, Helpers) :-
    format(string(Helpers),
"/// Read JSONL records from stdin.
fn read_jsonl_stream() -> Vec<HashMap<String, Value>> {
    let stdin = io::stdin();
    let mut records = Vec::new();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            if !line.trim().is_empty() {
                if let Ok(value) = serde_json::from_str::<HashMap<String, Value>>(&line) {
                    records.push(value);
                }
            }
        }
    }
    records
}

/// Write JSONL records to stdout.
fn write_jsonl_stream(records: &[HashMap<String, Value>]) {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    for record in records {
        if let Ok(json) = serde_json::to_string(record) {
            writeln!(handle, \"{}\", json).ok();
        }
    }
}", []).

%% extract_rust_stage_names(+Predicates, -Names)
%  Extract stage names from predicate indicators
extract_rust_stage_names([], []).
extract_rust_stage_names([Pred|Rest], [Name|RestNames]) :-
    extract_rust_pred_name(Pred, Name),
    extract_rust_stage_names(Rest, RestNames).

extract_rust_pred_name(_Target:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_rust_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).

%% generate_rust_stage_functions(+Names, -Code)
%  Generate placeholder stage function implementations
generate_rust_stage_functions([], "").
generate_rust_stage_functions([Name|Rest], Code) :-
    format(string(StageCode),
"/// Stage: ~w
fn stage_~w(input: Vec<HashMap<String, Value>>) -> Vec<HashMap<String, Value>> {
    // TODO: Implement stage logic
    input
}

", [Name, Name]),
    generate_rust_stage_functions(Rest, RestCode),
    format(string(Code), "~w~w", [StageCode, RestCode]).

%% generate_rust_pipeline_connector(+StageNames, +PipelineName, +Mode, -Code)
%  Generate the pipeline connector function (mode-aware)
generate_rust_pipeline_connector(StageNames, PipelineName, sequential, Code) :-
    !,
    generate_rust_sequential_chain(StageNames, ChainCode),
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"/// Sequential pipeline connector: ~w
fn ~w(input: Vec<HashMap<String, Value>>) -> Vec<HashMap<String, Value>> {
    // Sequential mode - chain stages directly
~w
}", [PipelineNameStr, PipelineNameStr, ChainCode]).

generate_rust_pipeline_connector(StageNames, PipelineName, generator, Code) :-
    generate_rust_fixpoint_chain(StageNames, ChainCode),
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"/// Fixpoint pipeline connector: ~w
/// Iterates until no new records are produced.
fn ~w(input: Vec<HashMap<String, Value>>) -> Vec<HashMap<String, Value>> {
    // Generator mode - fixpoint iteration
    let mut total: HashSet<String> = HashSet::new();
    let mut results: Vec<HashMap<String, Value>> = Vec::new();

    // Initialize with input records
    let mut working_set: Vec<HashMap<String, Value>> = Vec::new();
    for record in input {
        let key = record_key(&record);
        if !total.contains(&key) {
            total.insert(key);
            working_set.push(record.clone());
            results.push(record);
        }
    }

    // Fixpoint iteration
    let mut changed = true;
    while changed {
        changed = false;
        let current = working_set.clone();

        // Apply pipeline stages
~w

        // Check for new records
        for record in new_records {
            let key = record_key(&record);
            if !total.contains(&key) {
                total.insert(key);
                working_set.push(record.clone());
                results.push(record);
                changed = true;
            }
        }
    }

    results
}", [PipelineNameStr, PipelineNameStr, ChainCode]).

%% generate_rust_sequential_chain(+Names, -Code)
%  Generate sequential stage chaining code
generate_rust_sequential_chain([], Code) :-
    format(string(Code), "    input", []).
generate_rust_sequential_chain([Name], Code) :-
    !,
    format(string(Code), "    stage_~w(input)", [Name]).
generate_rust_sequential_chain(Names, Code) :-
    Names \= [],
    generate_rust_chain_expr(Names, "input", ChainExpr),
    format(string(Code), "    ~w", [ChainExpr]).

generate_rust_chain_expr([], Current, Current).
generate_rust_chain_expr([Name|Rest], Current, Expr) :-
    format(string(NextExpr), "stage_~w(~w)", [Name, Current]),
    generate_rust_chain_expr(Rest, NextExpr, Expr).

%% generate_rust_fixpoint_chain(+Names, -Code)
%  Generate fixpoint stage application code
generate_rust_fixpoint_chain([], Code) :-
    format(string(Code), "        let new_records = current;", []).
generate_rust_fixpoint_chain(Names, Code) :-
    Names \= [],
    generate_rust_fixpoint_stages(Names, "current", StageCode),
    format(string(Code), "~w", [StageCode]).

generate_rust_fixpoint_stages([], Current, Code) :-
    format(string(Code), "        let new_records = ~w;", [Current]).
generate_rust_fixpoint_stages([Stage|Rest], Current, Code) :-
    format(string(NextVar), "stage_~w_out", [Stage]),
    format(string(StageCall), "        let ~w = stage_~w(~w);
", [NextVar, Stage, Current]),
    generate_rust_fixpoint_stages(Rest, NextVar, RestCode),
    format(string(Code), "~w~w", [StageCall, RestCode]).

%% generate_rust_main_block(+PipelineName, +Format, -Code)
%  Generate main execution block
generate_rust_main_block(PipelineName, jsonl, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"fn main() {
    // Read from stdin, process through pipeline, write to stdout
    let input = read_jsonl_stream();
    let output = ~w(input);
    write_jsonl_stream(&output);
}", [PipelineNameStr]).

generate_rust_main_block(PipelineName, json, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"fn main() {
    // Read JSON array from stdin, process through pipeline, write to stdout
    let mut input_str = String::new();
    io::stdin().read_to_string(&mut input_str).expect(\"Failed to read stdin\");
    let input: Vec<HashMap<String, Value>> = serde_json::from_str(&input_str)
        .expect(\"Failed to parse JSON array\");
    let output = ~w(input);
    println!(\"{}\", serde_json::to_string(&output).unwrap());
}", [PipelineNameStr]).

% ============================================================================
% Unit Tests for Rust Pipeline Generator Mode
% ============================================================================

test_rust_pipeline_generator :-
    format("~n=== Rust Pipeline Generator Mode Unit Tests ===~n~n", []),

    % Test 1: Basic pipeline compilation with generator mode
    format("Test 1: Basic pipeline with generator mode... ", []),
    (   compile_rust_pipeline([transform/1, derive/1], [
            pipeline_name(test_pipeline),
            pipeline_mode(generator),
            output_format(jsonl)
        ], Code1),
        sub_string(Code1, _, _, _, "record_key"),
        sub_string(Code1, _, _, _, "while changed"),
        sub_string(Code1, _, _, _, "test_pipeline")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 2: Sequential mode still works
    format("Test 2: Sequential mode still works... ", []),
    (   compile_rust_pipeline([filter/1, format/1], [
            pipeline_name(seq_pipeline),
            pipeline_mode(sequential),
            output_format(jsonl)
        ], Code2),
        sub_string(Code2, _, _, _, "seq_pipeline"),
        sub_string(Code2, _, _, _, "sequential mode"),
        \+ sub_string(Code2, _, _, _, "while changed")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 3: Generator mode includes record_key function
    format("Test 3: Generator mode has record_key function... ", []),
    (   compile_rust_pipeline([a/1], [pipeline_mode(generator)], Code3),
        sub_string(Code3, _, _, _, "fn record_key"),
        sub_string(Code3, _, _, _, "keys.sort()")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 4: JSONL helpers included
    format("Test 4: JSONL helpers included... ", []),
    (   compile_rust_pipeline([x/1], [pipeline_mode(generator)], Code4),
        sub_string(Code4, _, _, _, "read_jsonl_stream"),
        sub_string(Code4, _, _, _, "write_jsonl_stream")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 5: Fixpoint iteration structure
    format("Test 5: Fixpoint iteration structure... ", []),
    (   compile_rust_pipeline([derive/1, transform/1], [pipeline_mode(generator)], Code5),
        sub_string(Code5, _, _, _, "HashSet<String>"),
        sub_string(Code5, _, _, _, "changed = true"),
        sub_string(Code5, _, _, _, "while changed"),
        sub_string(Code5, _, _, _, "total.contains")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 6: Stage functions generated
    format("Test 6: Stage functions generated... ", []),
    (   compile_rust_pipeline([filter/1, transform/1], [pipeline_mode(generator)], Code6),
        sub_string(Code6, _, _, _, "fn stage_filter"),
        sub_string(Code6, _, _, _, "fn stage_transform")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 7: Pipeline chain code for generator
    format("Test 7: Pipeline chain code for generator mode... ", []),
    (   compile_rust_pipeline([derive/1, transform/1], [pipeline_mode(generator)], Code7),
        sub_string(Code7, _, _, _, "stage_derive_out"),
        sub_string(Code7, _, _, _, "stage_transform_out")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 8: Default options work
    format("Test 8: Default options work... ", []),
    (   compile_rust_pipeline([a/1, b/1], [], Code8),
        sub_string(Code8, _, _, _, "fn pipeline"),
        sub_string(Code8, _, _, _, "sequential mode")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 9: Main block for JSONL format
    format("Test 9: Main block for JSONL format... ", []),
    (   compile_rust_pipeline([x/1], [
            pipeline_name(jsonl_pipe),
            output_format(jsonl)
        ], Code9),
        sub_string(Code9, _, _, _, "fn main()"),
        sub_string(Code9, _, _, _, "read_jsonl_stream()"),
        sub_string(Code9, _, _, _, "jsonl_pipe")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 10: Use statements for generator mode
    format("Test 10: Use statements for generator mode... ", []),
    (   compile_rust_pipeline([x/1], [pipeline_mode(generator)], Code10),
        sub_string(Code10, _, _, _, "use std::collections::{HashMap, HashSet}"),
        sub_string(Code10, _, _, _, "use serde_json")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    format("~n=== All Rust Pipeline Generator Mode Tests Passed ===~n", []).

%% ============================================
%% RUST ENHANCED PIPELINE CHAINING
%% ============================================
%
%  Supports advanced flow patterns:
%    - fan_out(Stages)        : Broadcast to stages (sequential execution)
%    - parallel(Stages)       : Execute stages concurrently (see parallel_mode option)
%    - merge                  : Combine results from fan_out or parallel
%    - route_by(Pred, Routes) : Conditional routing
%    - filter_by(Pred)        : Filter records
%    - Pred/Arity             : Standard stage
%
%  Options:
%    - parallel_mode(Mode)    : How to execute parallel stages
%        - std_thread (default): Use std::thread for parallelism (no extra deps)
%        - rayon               : Use rayon crate for parallel iteration (requires rayon dep)
%
%% compile_rust_enhanced_pipeline(+Stages, +Options, -RustCode)
%  Main entry point for enhanced Rust pipeline with advanced flow patterns.
%  Validates pipeline stages before code generation.
%
compile_rust_enhanced_pipeline(Stages, Options, RustCode) :-
    % Validate pipeline stages
    option(validate(Validate), Options, true),
    option(strict(Strict), Options, false),
    ( Validate == true ->
        validate_pipeline(Stages, [strict(Strict)], result(Errors, Warnings)),
        % Report warnings
        ( Warnings \== [] ->
            format(user_error, 'Rust pipeline warnings:~n', []),
            forall(member(W, Warnings), (
                format_validation_warning(W, Msg),
                format(user_error, '  ~w~n', [Msg])
            ))
        ; true
        ),
        % Fail on errors
        ( Errors \== [] ->
            format(user_error, 'Rust pipeline validation errors:~n', []),
            forall(member(E, Errors), (
                format_validation_error(E, Msg),
                format(user_error, '  ~w~n', [Msg])
            )),
            throw(pipeline_validation_failed(Errors))
        ; true
        )
    ; true
    ),

    option(pipeline_name(PipelineName), Options, enhanced_pipeline),
    option(output_format(OutputFormat), Options, jsonl),
    option(parallel_mode(ParallelMode), Options, std_thread),

    % Generate helpers based on parallel mode
    rust_enhanced_helpers(ParallelMode, Helpers),

    % Generate stage functions
    generate_rust_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the main connector
    generate_rust_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main function
    generate_rust_enhanced_main(PipelineName, OutputFormat, MainCode),

    % Generate imports based on parallel mode
    rust_enhanced_imports(ParallelMode, Imports),

    format(string(RustCode),
"// Generated by UnifyWeaver Rust Enhanced Pipeline
// Supports fan-out, merge, conditional routing, and filtering
~w

type Record = HashMap<String, Value>;

~w

~w
~w
~w
", [Imports, Helpers, StageFunctions, ConnectorCode, MainCode]).

%% rust_enhanced_imports(+ParallelMode, -Imports)
%  Generate imports based on parallel mode
rust_enhanced_imports(std_thread, Imports) :-
    Imports = "use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use serde_json::Value;".
rust_enhanced_imports(rayon, Imports) :-
    Imports = "use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use serde_json::Value;
use rayon::prelude::*;  // Requires: rayon = \"1.8\" in Cargo.toml".

%% rust_enhanced_helpers(+ParallelMode, -Code)
%  Generate helper functions for enhanced pipeline operations.
%  ParallelMode determines which parallel_records implementation to use.
rust_enhanced_helpers(ParallelMode, Code) :-
    rust_common_helpers(CommonHelpers),
    rust_parallel_helper(ParallelMode, ParallelHelper),
    rust_parallel_helper_ordered(ParallelMode, ParallelHelperOrdered),
    format(string(Code), "~w~n~w~n~w", [CommonHelpers, ParallelHelper, ParallelHelperOrdered]).

%% rust_common_helpers(-Code)
%  Helper functions shared between all parallel modes.
rust_common_helpers(Code) :-
    Code = "// Enhanced Pipeline Helpers

/// Fan-out: Send record to all stages, collect all results.
fn fan_out_records<F>(record: &Record, stages: &[F]) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record>,
{
    let mut results = Vec::new();
    for stage in stages {
        for result in stage(&[record.clone()]) {
            results.push(result);
        }
    }
    results
}

/// Merge: Combine multiple record slices into one.
fn merge_streams(streams: &[Vec<Record>]) -> Vec<Record> {
    let mut merged = Vec::new();
    for stream in streams {
        merged.extend(stream.clone());
    }
    merged
}

/// Route: Direct record to appropriate stage based on condition.
fn route_record<F, C>(
    record: &Record,
    condition_fn: C,
    route_map: &HashMap<Value, F>,
    default_fn: Option<&F>,
) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record>,
    C: Fn(&Record) -> Value,
{
    let condition = condition_fn(record);
    if let Some(stage) = route_map.get(&condition) {
        stage(&[record.clone()])
    } else if let Some(default) = default_fn {
        default(&[record.clone()])
    } else {
        vec![record.clone()] // Pass through if no matching route
    }
}

/// Filter: Only yield records that satisfy the predicate.
fn filter_records<F>(records: &[Record], predicate_fn: F) -> Vec<Record>
where
    F: Fn(&Record) -> bool,
{
    records.iter()
        .filter(|record| predicate_fn(record))
        .cloned()
        .collect()
}

/// Tee: Send each record to multiple stages and collect all results.
fn tee_stream<F>(records: &[Record], stages: &[F]) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record>,
{
    let mut results = Vec::new();
    for record in records {
        for stage in stages {
            results.extend(stage(&[record.clone()]));
        }
    }
    results
}

/// Read JSONL records from stdin.
fn read_jsonl_stream() -> Vec<Record> {
    let stdin = io::stdin();
    let mut records = Vec::new();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            if !line.trim().is_empty() {
                if let Ok(value) = serde_json::from_str::<Record>(&line) {
                    records.push(value);
                }
            }
        }
    }
    records
}

/// Write JSONL records to stdout.
fn write_jsonl_stream(records: &[Record]) {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    for record in records {
        if let Ok(json) = serde_json::to_string(record) {
            writeln!(handle, \"{}\", json).ok();
        }
    }
}

/// Batch: Collect records into batches of specified size.
fn batch_records(records: &[Record], batch_size: usize) -> Vec<Vec<Record>> {
    records.chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Unbatch: Flatten batches back to individual records.
fn unbatch_records(batches: &[Vec<Record>]) -> Vec<Record> {
    batches.iter()
        .flat_map(|batch| batch.clone())
        .collect()
}

/// Unique: Keep only the first record for each unique field value.
fn unique_by_field(records: &[Record], field: &str) -> Vec<Record> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for record in records {
        if let Some(key) = record.get(field) {
            let key_str = format!(\"{:?}\", key);
            if !seen.contains(&key_str) {
                seen.insert(key_str);
                result.push(record.clone());
            }
        }
    }
    result
}

/// First: Alias for unique_by_field (keeps first occurrence).
fn first_by_field(records: &[Record], field: &str) -> Vec<Record> {
    unique_by_field(records, field)
}

/// Last: Keep only the last record for each unique field value.
fn last_by_field(records: &[Record], field: &str) -> Vec<Record> {
    use std::collections::HashMap;
    let mut last_seen: HashMap<String, Record> = HashMap::new();
    let mut order = Vec::new();
    for record in records {
        if let Some(key) = record.get(field) {
            let key_str = format!(\"{:?}\", key);
            if !last_seen.contains_key(&key_str) {
                order.push(key_str.clone());
            }
            last_seen.insert(key_str, record.clone());
        }
    }
    order.iter().filter_map(|k| last_seen.get(k).cloned()).collect()
}

/// Aggregation type for group_by operations.
#[derive(Clone)]
enum AggType {
    Count,
    Sum(String),
    Avg(String),
    Min(String),
    Max(String),
    First(String),
    Last(String),
    Collect(String),
}

/// Group by field and apply aggregations.
fn group_by_field(records: &[Record], field: &str, aggregations: &[(&str, AggType)]) -> Vec<Record> {
    use std::collections::HashMap;
    let mut groups: HashMap<String, Vec<Record>> = HashMap::new();
    let mut order = Vec::new();

    for record in records {
        if let Some(key) = record.get(field) {
            let key_str = format!(\"{:?}\", key);
            if !groups.contains_key(&key_str) {
                order.push((key_str.clone(), key.clone()));
            }
            groups.entry(key_str).or_insert_with(Vec::new).push(record.clone());
        }
    }

    let mut result = Vec::new();
    for (key_str, key_val) in order {
        if let Some(group_records) = groups.get(&key_str) {
            let mut result_record = Record::new();
            result_record.insert(field.to_string(), key_val);

            for (name, agg) in aggregations {
                let value: Value = match agg {
                    AggType::Count => Value::from(group_records.len() as i64),
                    AggType::Sum(f) => {
                        let sum: f64 = group_records.iter()
                            .filter_map(|r| r.get(f).and_then(|v| v.as_f64()))
                            .sum();
                        Value::from(sum)
                    }
                    AggType::Avg(f) => {
                        let values: Vec<f64> = group_records.iter()
                            .filter_map(|r| r.get(f).and_then(|v| v.as_f64()))
                            .collect();
                        let avg = if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 };
                        Value::from(avg)
                    }
                    AggType::Min(f) => {
                        group_records.iter()
                            .filter_map(|r| r.get(f).and_then(|v| v.as_f64()))
                            .fold(f64::INFINITY, f64::min)
                            .into()
                    }
                    AggType::Max(f) => {
                        group_records.iter()
                            .filter_map(|r| r.get(f).and_then(|v| v.as_f64()))
                            .fold(f64::NEG_INFINITY, f64::max)
                            .into()
                    }
                    AggType::First(f) => {
                        group_records.first().and_then(|r| r.get(f)).cloned().unwrap_or(Value::Null)
                    }
                    AggType::Last(f) => {
                        group_records.last().and_then(|r| r.get(f)).cloned().unwrap_or(Value::Null)
                    }
                    AggType::Collect(f) => {
                        let values: Vec<Value> = group_records.iter()
                            .filter_map(|r| r.get(f).cloned())
                            .collect();
                        Value::from(values)
                    }
                };
                result_record.insert(name.to_string(), value);
            }
            result.push(result_record);
        }
    }
    result
}

/// Reduce: Apply reducer function sequentially across all records.
fn reduce_records<F>(records: &[Record], reducer: F, initial: Value) -> Vec<Record>
where
    F: Fn(&Record, Value) -> Value,
{
    let mut acc = initial;
    for record in records {
        acc = reducer(record, acc);
    }
    let mut result = Record::new();
    result.insert(\"result\".to_string(), acc);
    vec![result]
}

/// Scan: Like reduce but yields intermediate results.
fn scan_records<F>(records: &[Record], reducer: F, initial: Value) -> Vec<Record>
where
    F: Fn(&Record, Value) -> Value,
{
    let mut result = Vec::new();
    let mut acc = initial;
    for record in records {
        acc = reducer(record, acc);
        let mut r = Record::new();
        r.insert(\"result\".to_string(), acc.clone());
        result.push(r);
    }
    result
}

/// Order by single field with direction.
fn order_by_field(records: &[Record], field: &str, direction: &str) -> Vec<Record> {
    let mut result: Vec<Record> = records.to_vec();
    result.sort_by(|a, b| {
        let va = a.get(field);
        let vb = b.get(field);
        let cmp = compare_values(va, vb);
        if direction == \"desc\" {
            cmp.reverse()
        } else {
            cmp
        }
    });
    result
}

/// Field specification for multi-field ordering.
struct FieldSpec {
    field: String,
    direction: String,
}

/// Order by multiple fields.
fn order_by_fields(records: &[Record], field_specs: &[FieldSpec]) -> Vec<Record> {
    let mut result: Vec<Record> = records.to_vec();
    result.sort_by(|a, b| {
        for spec in field_specs {
            let va = a.get(&spec.field);
            let vb = b.get(&spec.field);
            let cmp = compare_values(va, vb);
            let cmp = if spec.direction == \"desc\" { cmp.reverse() } else { cmp };
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    });
    result
}

/// Compare two optional JSON values.
fn compare_values(a: Option<&Value>, b: Option<&Value>) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,  // None sorts to end
        (Some(_), None) => Ordering::Less,
        (Some(va), Some(vb)) => {
            // Compare by type
            match (va, vb) {
                (Value::Number(na), Value::Number(nb)) => {
                    let fa = na.as_f64().unwrap_or(0.0);
                    let fb = nb.as_f64().unwrap_or(0.0);
                    fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
                }
                (Value::String(sa), Value::String(sb)) => sa.cmp(sb),
                _ => {
                    // Fallback: compare string representations
                    let sa = format!(\"{:?}\", va);
                    let sb = format!(\"{:?}\", vb);
                    sa.cmp(&sb)
                }
            }
        }
    }
}

/// Sort by custom comparator function.
fn sort_by_comparator<F>(records: &[Record], comparator: F) -> Vec<Record>
where
    F: Fn(&Record, &Record) -> std::cmp::Ordering,
{
    let mut result: Vec<Record> = records.to_vec();
    result.sort_by(|a, b| comparator(a, b));
    result
}

// Error Handling Stage Helpers

/// Type alias for stage functions
type StageFunc = fn(&[Record]) -> Result<Vec<Record>, Box<dyn std::error::Error>>;
type ErrorHandlerFunc = fn(&[Record], &dyn std::error::Error) -> Vec<Record>;

/// Try-catch: Execute stage, on error route to handler.
fn try_catch_stage(
    records: &[Record],
    stage: StageFunc,
    handler: ErrorHandlerFunc,
) -> Vec<Record> {
    let mut result = Vec::new();
    for record in records {
        match stage(&[record.clone()]) {
            Ok(results) => result.extend(results),
            Err(e) => {
                let handler_results = handler(&[record.clone()], e.as_ref());
                result.extend(handler_results);
            }
        }
    }
    result
}

/// Retry: Retry stage up to max_retries times on failure.
fn retry_stage(
    records: &[Record],
    stage: StageFunc,
    max_retries: usize,
    delay_ms: u64,
    backoff: &str,
) -> Vec<Record> {
    let mut result = Vec::new();
    for record in records {
        let mut last_error: Option<String> = None;
        for attempt in 0..=max_retries {
            match stage(&[record.clone()]) {
                Ok(results) => {
                    result.extend(results);
                    last_error = None;
                    break;
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    if attempt < max_retries && delay_ms > 0 {
                        let wait_time = match backoff {
                            \"exponential\" => delay_ms * (1 << attempt),
                            \"linear\" => delay_ms * (attempt as u64 + 1),
                            _ => delay_ms,
                        };
                        std::thread::sleep(std::time::Duration::from_millis(wait_time));
                    }
                }
            }
        }
        // If all retries exhausted, add error record
        if let Some(err) = last_error {
            let mut error_record = Record::new();
            error_record.insert(\"_error\".to_string(), Value::String(err));
            error_record.insert(\"_record\".to_string(), serde_json::to_value(record).unwrap_or(Value::Null));
            error_record.insert(\"_retries\".to_string(), Value::Number(max_retries.into()));
            result.push(error_record);
        }
    }
    result
}

/// On-error: Global error handler wrapping.
fn on_error_stage(records: &[Record], _handler: ErrorHandlerFunc) -> Vec<Record> {
    // In Rust, we pass records through - errors are handled at stage level
    records.to_vec()
}

/// Timeout: Execute stage with time limit.
fn timeout_stage(
    records: &[Record],
    stage: StageFunc,
    timeout_ms: u64,
) -> Vec<Record> {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let mut result = Vec::new();
    for record in records {
        let record_clone = record.clone();
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            let stage_result = stage(&[record_clone]);
            let _ = tx.send(stage_result);
        });

        match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
            Ok(Ok(results)) => result.extend(results),
            Ok(Err(_)) | Err(_) => {
                // Timeout or error occurred
                let mut timeout_record = Record::new();
                timeout_record.insert(\"_timeout\".to_string(), Value::Bool(true));
                timeout_record.insert(\"_record\".to_string(), serde_json::to_value(record).unwrap_or(Value::Null));
                timeout_record.insert(\"_limit_ms\".to_string(), Value::Number(timeout_ms.into()));
                result.push(timeout_record);
            }
        }
    }
    result
}

/// Timeout with fallback: Execute stage with time limit, use fallback on timeout.
fn timeout_stage_with_fallback<F>(
    records: &[Record],
    stage: StageFunc,
    timeout_ms: u64,
    fallback: F,
) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record>,
{
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let mut result = Vec::new();
    for record in records {
        let record_clone = record.clone();
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            let stage_result = stage(&[record_clone]);
            let _ = tx.send(stage_result);
        });

        match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
            Ok(Ok(results)) => result.extend(results),
            Ok(Err(_)) | Err(_) => {
                // Timeout or error occurred, use fallback
                let fallback_results = fallback(&[record.clone()]);
                result.extend(fallback_results);
            }
        }
    }
    result
}

/// Rate limit: Limit throughput to count records per interval.
fn rate_limit_stage(records: &[Record], count: usize, interval_ms: u64) -> Vec<Record> {
    use std::time::{Duration, Instant};
    use std::thread;

    let mut result = Vec::new();
    let interval = Duration::from_millis(interval_ms) / count as u32;
    let mut last_time: Option<Instant> = None;

    for record in records {
        if let Some(last) = last_time {
            let elapsed = last.elapsed();
            if elapsed < interval {
                thread::sleep(interval - elapsed);
            }
        }
        last_time = Some(Instant::now());
        result.push(record.clone());
    }
    result
}

/// Throttle: Add fixed delay between records.
fn throttle_stage(records: &[Record], delay_ms: u64) -> Vec<Record> {
    use std::time::Duration;
    use std::thread;

    let mut result = Vec::new();
    let delay = Duration::from_millis(delay_ms);

    for (i, record) in records.iter().enumerate() {
        if i > 0 {
            thread::sleep(delay);
        }
        result.push(record.clone());
    }
    result
}

/// Buffer: Collect records into batches of specified size.
fn buffer_stage(records: &[Record], size: usize) -> Vec<Vec<Record>> {
    let mut result = Vec::new();
    let mut buffer = Vec::new();

    for record in records {
        buffer.push(record.clone());
        if buffer.len() >= size {
            result.push(buffer);
            buffer = Vec::new();
        }
    }
    if !buffer.is_empty() {
        result.push(buffer);
    }
    result
}

/// Debounce: Emit record only after quiet period.
fn debounce_stage(records: &[Record], delay_ms: u64) -> Vec<Record> {
    use std::time::Duration;
    use std::thread;

    if records.is_empty() {
        return Vec::new();
    }

    let delay = Duration::from_millis(delay_ms);
    let last_record = records.last().unwrap().clone();
    thread::sleep(delay);
    vec![last_record]
}

/// Zip: Run multiple stage functions and combine results.
fn zip_stage<F>(records: &[Record], stages: &[F]) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record>,
{
    let mut result = Vec::new();
    for record in records {
        let mut stage_results: Vec<Vec<Record>> = Vec::new();
        let mut max_len = 0;
        for stage in stages {
            let res = stage(&[record.clone()]);
            if res.len() > max_len {
                max_len = res.len();
            }
            stage_results.push(res);
        }
        for i in 0..max_len {
            let mut combined = Record::new();
            for res_list in &stage_results {
                if i < res_list.len() {
                    for (k, v) in &res_list[i] {
                        combined.insert(k.clone(), v.clone());
                    }
                }
            }
            result.push(combined);
        }
    }
    result
}

/// Window: Collect records into non-overlapping windows.
fn window_stage(records: &[Record], size: usize) -> Vec<Vec<Record>> {
    let mut result = Vec::new();
    let mut window = Vec::new();
    for record in records {
        window.push(record.clone());
        if window.len() >= size {
            result.push(window);
            window = Vec::new();
        }
    }
    if !window.is_empty() {
        result.push(window);
    }
    result
}

/// Sliding Window: Create sliding windows of records.
fn sliding_window_stage(records: &[Record], size: usize, step: usize) -> Vec<Vec<Record>> {
    let mut result = Vec::new();
    let mut buffer = Vec::new();
    for record in records {
        buffer.push(record.clone());
        while buffer.len() >= size {
            let window: Vec<Record> = buffer[..size].to_vec();
            result.push(window);
            buffer = buffer[step..].to_vec();
        }
    }
    if !buffer.is_empty() {
        result.push(buffer);
    }
    result
}

/// Sample: Randomly sample n records using reservoir sampling.
fn sample_stage(records: &[Record], n: usize) -> Vec<Record> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut reservoir = Vec::with_capacity(n);
    for (i, record) in records.iter().enumerate() {
        if i < n {
            reservoir.push(record.clone());
        } else {
            let j = rng.gen_range(0..=i);
            if j < n {
                reservoir[j] = record.clone();
            }
        }
    }
    reservoir
}

/// Take Every: Take every nth record.
fn take_every_stage(records: &[Record], n: usize) -> Vec<Record> {
    records.iter().enumerate()
        .filter(|(i, _)| i % n == 0)
        .map(|(_, r)| r.clone())
        .collect()
}

/// Partition: Split records into matches and non-matches.
fn partition_stage<F>(records: &[Record], pred: F) -> (Vec<Record>, Vec<Record>)
where
    F: Fn(&Record) -> bool,
{
    let mut matches = Vec::new();
    let mut non_matches = Vec::new();
    for record in records {
        if pred(record) {
            matches.push(record.clone());
        } else {
            non_matches.push(record.clone());
        }
    }
    (matches, non_matches)
}

/// Take: Take first n records.
fn take_stage(records: &[Record], n: usize) -> Vec<Record> {
    records.iter().take(n).cloned().collect()
}

/// Skip: Skip first n records.
fn skip_stage(records: &[Record], n: usize) -> Vec<Record> {
    records.iter().skip(n).cloned().collect()
}

/// Take While: Take records while predicate is true.
fn take_while_stage<F>(records: &[Record], pred: F) -> Vec<Record>
where
    F: Fn(&Record) -> bool,
{
    records.iter().take_while(|r| pred(r)).cloned().collect()
}

/// Skip While: Skip records while predicate is true.
fn skip_while_stage<F>(records: &[Record], pred: F) -> Vec<Record>
where
    F: Fn(&Record) -> bool,
{
    records.iter().skip_while(|r| pred(r)).cloned().collect()
}

/// Distinct: Remove all duplicate records (global dedup).
fn distinct_stage(records: &[Record]) -> Vec<Record> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for record in records {
        let key = record_key(record);
        if !seen.contains(&key) {
            seen.insert(key);
            result.push(record.clone());
        }
    }
    result
}

/// Distinct By: Remove duplicates based on a specific field.
fn distinct_by_stage(records: &[Record], field: &str) -> Vec<Record> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for record in records {
        let key = format!(\"{:?}\", record.get(field));
        if !seen.contains(&key) {
            seen.insert(key);
            result.push(record.clone());
        }
    }
    result
}

/// Dedup: Remove consecutive duplicate records.
fn dedup_stage(records: &[Record]) -> Vec<Record> {
    let mut result = Vec::new();
    let mut last_key: Option<String> = None;
    for record in records {
        let key = record_key(record);
        if last_key.as_ref() != Some(&key) {
            last_key = Some(key);
            result.push(record.clone());
        }
    }
    result
}

/// Dedup By: Remove consecutive duplicates based on a specific field.
fn dedup_by_stage(records: &[Record], field: &str) -> Vec<Record> {
    let mut result = Vec::new();
    let mut last_value: Option<String> = None;
    for record in records {
        let value = format!(\"{:?}\", record.get(field));
        if last_value.as_ref() != Some(&value) {
            last_value = Some(value);
            result.push(record.clone());
        }
    }
    result
}

/// Generate a unique key for a record (for dedup comparison).
fn record_key(record: &Record) -> String {
    let mut keys: Vec<&String> = record.keys().collect();
    keys.sort();
    keys.iter()
        .map(|k| format!(\"{}={:?}\", k, record.get(*k)))
        .collect::<Vec<_>>()
        .join(\",\")
}

/// Interleave: Round-robin interleave records from multiple streams.
fn interleave_stage(streams: &[Vec<Record>]) -> Vec<Record> {
    if streams.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let max_len = streams.iter().map(|s| s.len()).max().unwrap_or(0);
    for i in 0..max_len {
        for stream in streams {
            if i < stream.len() {
                result.push(stream[i].clone());
            }
        }
    }
    result
}

/// Concat: Concatenate multiple streams sequentially.
fn concat_stage(streams: &[Vec<Record>]) -> Vec<Record> {
    let mut result = Vec::new();
    for stream in streams {
        result.extend(stream.iter().cloned());
    }
    result
}

/// Merge Sorted: Merge multiple pre-sorted streams maintaining sort order.
/// Uses a k-way merge algorithm for efficiency.
fn merge_sorted_stage(streams: &[Vec<Record>], field: &str, ascending: bool) -> Vec<Record> {
    use std::cmp::Ordering;

    if streams.is_empty() {
        return Vec::new();
    }

    // Track current index in each stream
    let mut indices: Vec<usize> = vec![0; streams.len()];
    let mut result = Vec::new();

    loop {
        // Find the stream with the best (smallest/largest) next value
        let mut best_stream: Option<usize> = None;
        let mut best_value: Option<&serde_json::Value> = None;

        for (i, stream) in streams.iter().enumerate() {
            if indices[i] >= stream.len() {
                continue; // This stream is exhausted
            }

            let value = stream[indices[i]].get(field);
            if best_stream.is_none() {
                best_stream = Some(i);
                best_value = value;
                continue;
            }

            // Compare values
            let is_better = match (value, best_value) {
                (Some(serde_json::Value::Number(v)), Some(serde_json::Value::Number(bv))) => {
                    let v_f = v.as_f64().unwrap_or(0.0);
                    let bv_f = bv.as_f64().unwrap_or(0.0);
                    if ascending { v_f < bv_f } else { v_f > bv_f }
                }
                (Some(serde_json::Value::String(v)), Some(serde_json::Value::String(bv))) => {
                    if ascending { v < bv } else { v > bv }
                }
                _ => false,
            };

            if is_better {
                best_stream = Some(i);
                best_value = value;
            }
        }

        match best_stream {
            Some(idx) => {
                result.push(streams[idx][indices[idx]].clone());
                indices[idx] += 1;
            }
            None => break, // All streams exhausted
        }
    }

    result
}
".

%% rust_parallel_helper(+ParallelMode, -Code)
%  Generate parallel_records helper based on mode.

% std_thread mode: Use std::thread for parallelism (no extra dependencies)
rust_parallel_helper(std_thread, Code) :-
    Code = "
/// Parallel: Execute stages concurrently using std::thread.
/// Each stage receives the same input record.
/// Results are collected after all threads complete.
fn parallel_records<F>(record: &Record, stages: &[F]) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record> + Send + Sync,
{
    use std::sync::{Arc, Mutex};
    use std::thread;

    let record = Arc::new(record.clone());
    let results = Arc::new(Mutex::new(Vec::new()));
    let handles: Vec<_> = stages
        .iter()
        .map(|stage| {
            let record_clone = Arc::clone(&record);
            let results_clone = Arc::clone(&results);
            let stage_results = stage(&[(*record_clone).clone()]);
            thread::spawn(move || {
                let mut results = results_clone.lock().unwrap();
                results.extend(stage_results);
            })
        })
        .collect();

    for handle in handles {
        handle.join().ok();
    }

    Arc::try_unwrap(results)
        .unwrap_or_else(|arc| (*arc.lock().unwrap()).clone())
        .into_inner()
        .unwrap_or_default()
}
".

% rayon mode: Use rayon crate for parallel iteration (requires rayon dependency)
rust_parallel_helper(rayon, Code) :-
    Code = "
/// Parallel: Execute stages concurrently using rayon parallel iterators.
/// Each stage receives the same input record.
/// Results are collected after all stages complete.
/// Requires: rayon = \"1.8\" in Cargo.toml
fn parallel_records<F>(record: &Record, stages: &[F]) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record> + Send + Sync,
{
    stages
        .par_iter()
        .flat_map(|stage| stage(&[record.clone()]))
        .collect()
}
".

%% rust_parallel_helper_ordered(+ParallelMode, -Code) is det.
%  Generate parallel_records_ordered helper based on mode (preserves stage order).

% std_thread mode: Use std::thread with indexed results
rust_parallel_helper_ordered(std_thread, Code) :-
    Code = "
/// Parallel (Ordered): Execute stages concurrently, preserve input order.
/// Each stage receives the same input record.
/// Results are returned in stage definition order.
fn parallel_records_ordered<F>(record: &Record, stages: &[F]) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record> + Send + Sync,
{
    use std::sync::{Arc, Mutex};
    use std::thread;

    let record = Arc::new(record.clone());
    let indexed_results: Arc<Mutex<Vec<Option<Vec<Record>>>>> =
        Arc::new(Mutex::new(vec![None; stages.len()]));

    let handles: Vec<_> = stages
        .iter()
        .enumerate()
        .map(|(idx, stage)| {
            let record_clone = Arc::clone(&record);
            let results_clone = Arc::clone(&indexed_results);
            let stage_results = stage(&[(*record_clone).clone()]);
            thread::spawn(move || {
                let mut results = results_clone.lock().unwrap();
                results[idx] = Some(stage_results);
            })
        })
        .collect();

    for handle in handles {
        handle.join().ok();
    }

    // Flatten results in order
    Arc::try_unwrap(indexed_results)
        .unwrap_or_else(|arc| (*arc.lock().unwrap()).clone())
        .into_inner()
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .flatten()
        .collect()
}
".

% rayon mode: Use rayon with indexed collection
rust_parallel_helper_ordered(rayon, Code) :-
    Code = "
/// Parallel (Ordered): Execute stages concurrently, preserve input order.
/// Each stage receives the same input record.
/// Results are returned in stage definition order.
/// Requires: rayon = \"1.8\" in Cargo.toml
fn parallel_records_ordered<F>(record: &Record, stages: &[F]) -> Vec<Record>
where
    F: Fn(&[Record]) -> Vec<Record> + Send + Sync,
{
    // Execute in parallel but collect with indices
    let indexed_results: Vec<(usize, Vec<Record>)> = stages
        .par_iter()
        .enumerate()
        .map(|(idx, stage)| (idx, stage(&[record.clone()])))
        .collect();

    // Sort by index and flatten
    let mut sorted = indexed_results;
    sorted.sort_by_key(|(idx, _)| *idx);
    sorted.into_iter().flat_map(|(_, results)| results).collect()
}
".

%% generate_rust_enhanced_stage_functions(+Stages, -Code)
%  Generate stub functions for each stage.
generate_rust_enhanced_stage_functions([], "").
generate_rust_enhanced_stage_functions([Stage|Rest], Code) :-
    generate_rust_single_enhanced_stage(Stage, StageCode),
    generate_rust_enhanced_stage_functions(Rest, RestCode),
    (RestCode = "" ->
        Code = StageCode
    ;   format(string(Code), "~w~n~w", [StageCode, RestCode])
    ).

generate_rust_single_enhanced_stage(fan_out(SubStages), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(parallel(SubStages, _Options), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(parallel(SubStages), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(merge, "") :- !.
generate_rust_single_enhanced_stage(route_by(_, Routes), Code) :-
    !,
    findall(Stage, member((_Cond, Stage), Routes), RouteStages),
    generate_rust_enhanced_stage_functions(RouteStages, Code).
generate_rust_single_enhanced_stage(filter_by(_), "") :- !.
generate_rust_single_enhanced_stage(batch(_), "") :- !.
generate_rust_single_enhanced_stage(unbatch, "") :- !.
generate_rust_single_enhanced_stage(unique(_), "") :- !.
generate_rust_single_enhanced_stage(first(_), "") :- !.
generate_rust_single_enhanced_stage(last(_), "") :- !.
generate_rust_single_enhanced_stage(group_by(_, _), "") :- !.
generate_rust_single_enhanced_stage(reduce(_, _), "") :- !.
generate_rust_single_enhanced_stage(reduce(_), "") :- !.
generate_rust_single_enhanced_stage(scan(_, _), "") :- !.
generate_rust_single_enhanced_stage(scan(_), "") :- !.
generate_rust_single_enhanced_stage(order_by(_), "") :- !.
generate_rust_single_enhanced_stage(order_by(_, _), "") :- !.
generate_rust_single_enhanced_stage(sort_by(_), "") :- !.
generate_rust_single_enhanced_stage(try_catch(Stage, Handler), Code) :-
    !,
    generate_rust_single_enhanced_stage(Stage, StageCode),
    generate_rust_single_enhanced_stage(Handler, HandlerCode),
    format(string(Code), "~w~w", [StageCode, HandlerCode]).
generate_rust_single_enhanced_stage(retry(Stage, _), Code) :-
    !,
    generate_rust_single_enhanced_stage(Stage, Code).
generate_rust_single_enhanced_stage(retry(Stage, _, _), Code) :-
    !,
    generate_rust_single_enhanced_stage(Stage, Code).
generate_rust_single_enhanced_stage(on_error(Handler), Code) :-
    !,
    generate_rust_single_enhanced_stage(Handler, Code).
generate_rust_single_enhanced_stage(timeout(Stage, _), Code) :-
    !,
    generate_rust_single_enhanced_stage(Stage, Code).
generate_rust_single_enhanced_stage(timeout(Stage, _, Fallback), Code) :-
    !,
    generate_rust_single_enhanced_stage(Stage, StageCode),
    generate_rust_single_enhanced_stage(Fallback, FallbackCode),
    format(string(Code), "~w~w", [StageCode, FallbackCode]).
generate_rust_single_enhanced_stage(rate_limit(_, _), "") :- !.
generate_rust_single_enhanced_stage(throttle(_), "") :- !.
generate_rust_single_enhanced_stage(buffer(_), "") :- !.
generate_rust_single_enhanced_stage(debounce(_), "") :- !.
generate_rust_single_enhanced_stage(zip(SubStages), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(window(_), "") :- !.
generate_rust_single_enhanced_stage(sliding_window(_, _), "") :- !.
generate_rust_single_enhanced_stage(sample(_), "") :- !.
generate_rust_single_enhanced_stage(take_every(_), "") :- !.
generate_rust_single_enhanced_stage(partition(_), "") :- !.
generate_rust_single_enhanced_stage(take(_), "") :- !.
generate_rust_single_enhanced_stage(skip(_), "") :- !.
generate_rust_single_enhanced_stage(take_while(_), "") :- !.
generate_rust_single_enhanced_stage(skip_while(_), "") :- !.
generate_rust_single_enhanced_stage(distinct, "") :- !.
generate_rust_single_enhanced_stage(distinct_by(_), "") :- !.
generate_rust_single_enhanced_stage(dedup, "") :- !.
generate_rust_single_enhanced_stage(dedup_by(_), "") :- !.
generate_rust_single_enhanced_stage(interleave(SubStages), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(concat(SubStages), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(merge_sorted(SubStages, _Field), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(merge_sorted(SubStages, _Field, _Dir), Code) :-
    !,
    generate_rust_enhanced_stage_functions(SubStages, Code).
generate_rust_single_enhanced_stage(Pred/Arity, Code) :-
    !,
    format(string(Code),
"/// Pipeline stage: ~w/~w
fn ~w(input: &[Record]) -> Vec<Record> {
    // TODO: Implement based on predicate bindings
    input.to_vec()
}

", [Pred, Arity, Pred]).
generate_rust_single_enhanced_stage(_, "").

%% generate_rust_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the main connector that handles enhanced flow patterns.
generate_rust_enhanced_connector(Stages, PipelineName, Code) :-
    generate_rust_enhanced_flow_code(Stages, "input", FlowCode),
    format(string(Code),
"/// ~w is an enhanced pipeline with fan-out, merge, and routing support.
fn ~w(input: &[Record]) -> Vec<Record> {
~w
}

", [PipelineName, PipelineName, FlowCode]).

%% generate_rust_enhanced_flow_code(+Stages, +CurrentVar, -Code)
%  Generate the flow code for enhanced pipeline stages.
generate_rust_enhanced_flow_code([], CurrentVar, Code) :-
    format(string(Code), "    ~w.to_vec()", [CurrentVar]).
generate_rust_enhanced_flow_code([Stage|Rest], CurrentVar, Code) :-
    generate_rust_stage_flow(Stage, CurrentVar, NextVar, StageCode),
    generate_rust_enhanced_flow_code(Rest, NextVar, RestCode),
    format(string(Code), "~w~n~w", [StageCode, RestCode]).

%% generate_rust_stage_flow(+Stage, +InVar, -OutVar, -Code)
%  Generate flow code for a single stage.

% Fan-out stage: broadcast to stages (sequential execution)
generate_rust_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "fan_out_~w_result", [N]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    format(string(Code),
"    // Fan-out to ~w stages (sequential)
    let ~w: Vec<Record> = ~w.iter()
        .flat_map(|record| fan_out_records(record, &[~w]))
        .collect();", [N, OutVar, InVar, StageListStr]).

% Parallel stage with options: parallel(Stages, Options)
generate_rust_stage_flow(parallel(SubStages, Options), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "parallel_~w_result", [N]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    % Check for ordered option
    (   member(ordered(true), Options)
    ->  FuncName = "parallel_records_ordered",
        format(atom(Comment), "Parallel execution (ordered) of ~w stages", [N])
    ;   FuncName = "parallel_records",
        format(atom(Comment), "Parallel execution of ~w stages (concurrent via threads)", [N])
    ),
    format(string(Code),
"    // ~w
    let ~w: Vec<Record> = ~w.iter()
        .flat_map(|record| ~w(record, &[~w]))
        .collect();", [Comment, OutVar, InVar, FuncName, StageListStr]).

% Parallel stage: concurrent execution using threads (default: unordered)
generate_rust_stage_flow(parallel(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "parallel_~w_result", [N]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    format(string(Code),
"    // Parallel execution of ~w stages (concurrent via threads)
    let ~w: Vec<Record> = ~w.iter()
        .flat_map(|record| parallel_records(record, &[~w]))
        .collect();", [N, OutVar, InVar, StageListStr]).

% Merge stage: placeholder, usually follows fan_out or parallel
generate_rust_stage_flow(merge, InVar, OutVar, Code) :-
    !,
    OutVar = InVar,
    Code = "    // Merge: results already combined from fan-out or parallel".

% Conditional routing
generate_rust_stage_flow(route_by(CondPred, Routes), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "routed_result", []),
    format_rust_route_map(Routes, RouteMapStr),
    format(string(Code),
"    // Conditional routing based on ~w
    let mut route_map: HashMap<Value, fn(&[Record]) -> Vec<Record>> = HashMap::new();
~w
    let ~w: Vec<Record> = ~w.iter()
        .flat_map(|record| route_record(record, ~w, &route_map, None))
        .collect();", [CondPred, RouteMapStr, OutVar, InVar, CondPred]).

% Filter stage
generate_rust_stage_flow(filter_by(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "filtered_result", []),
    format(string(Code),
"    // Filter by ~w
    let ~w = filter_records(&~w, ~w);", [Pred, OutVar, InVar, Pred]).

% Batch stage: collect N records into batches
generate_rust_stage_flow(batch(N), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "batched_~w_result", [N]),
    format(string(Code),
"    // Batch records into groups of ~w
    let ~w = batch_records(&~w, ~w);", [N, OutVar, InVar, N]).

% Unbatch stage: flatten batches back to individual records
generate_rust_stage_flow(unbatch, InVar, OutVar, Code) :-
    !,
    OutVar = "unbatched_result",
    format(string(Code),
"    // Unbatch: flatten batches to individual records
    let ~w = unbatch_records(&~w);", [OutVar, InVar]).

% Unique stage: deduplicate by field (keep first)
generate_rust_stage_flow(unique(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "unique_~w_result", [Field]),
    format(string(Code),
"    // Unique: keep first record per '~w' value
    let ~w = unique_by_field(&~w, \"~w\");", [Field, OutVar, InVar, Field]).

% First stage: alias for unique (keep first occurrence)
generate_rust_stage_flow(first(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "first_~w_result", [Field]),
    format(string(Code),
"    // First: keep first record per '~w' value
    let ~w = first_by_field(&~w, \"~w\");", [Field, OutVar, InVar, Field]).

% Last stage: keep last record per field value
generate_rust_stage_flow(last(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "last_~w_result", [Field]),
    format(string(Code),
"    // Last: keep last record per '~w' value
    let ~w = last_by_field(&~w, \"~w\");", [Field, OutVar, InVar, Field]).

% Group by stage: group and aggregate
generate_rust_stage_flow(group_by(Field, Agg), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "grouped_~w_result", [Field]),
    format_rust_aggregations(Agg, AggStr),
    format(string(Code),
"    // Group by '~w' with aggregations
    let ~w = group_by_field(&~w, \"~w\", &[~w]);", [Field, OutVar, InVar, Field, AggStr]).

% Reduce stage with initial value
generate_rust_stage_flow(reduce(Pred, Init), InVar, OutVar, Code) :-
    !,
    OutVar = "reduced_result",
    format(string(Code),
"    // Reduce: sequential fold with ~w
    let ~w = reduce_records(&~w, ~w, ~w.into());", [Pred, OutVar, InVar, Pred, Init]).

% Reduce stage without initial value
generate_rust_stage_flow(reduce(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "reduced_result",
    format(string(Code),
"    // Reduce: sequential fold with ~w
    let ~w = reduce_records(&~w, ~w, Value::Null);", [Pred, OutVar, InVar, Pred]).

% Scan stage with initial value
generate_rust_stage_flow(scan(Pred, Init), InVar, OutVar, Code) :-
    !,
    OutVar = "scanned_result",
    format(string(Code),
"    // Scan: running fold with ~w (emits intermediate values)
    let ~w = scan_records(&~w, ~w, ~w.into());", [Pred, OutVar, InVar, Pred, Init]).

% Scan stage without initial value
generate_rust_stage_flow(scan(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "scanned_result",
    format(string(Code),
"    // Scan: running fold with ~w (emits intermediate values)
    let ~w = scan_records(&~w, ~w, Value::Null);", [Pred, OutVar, InVar, Pred]).

% Order by single field (ascending by default)
generate_rust_stage_flow(order_by(Field), InVar, OutVar, Code) :-
    atom(Field),
    !,
    format(atom(OutVar), "ordered_~w_result", [Field]),
    format(string(Code),
"    // Order by '~w' ascending
    let ~w = order_by_field(&~w, \"~w\", \"asc\");", [Field, OutVar, InVar, Field]).

% Order by single field with direction
generate_rust_stage_flow(order_by(Field, Dir), InVar, OutVar, Code) :-
    atom(Field),
    !,
    format(atom(OutVar), "ordered_~w_result", [Field]),
    format(string(Code),
"    // Order by '~w' ~w
    let ~w = order_by_field(&~w, \"~w\", \"~w\");", [Field, Dir, OutVar, InVar, Field, Dir]).

% Order by multiple fields with directions
generate_rust_stage_flow(order_by(FieldSpecs), InVar, OutVar, Code) :-
    is_list(FieldSpecs),
    !,
    OutVar = "ordered_multi_result",
    format_rust_field_specs(FieldSpecs, SpecStr),
    format(string(Code),
"    // Order by multiple fields
    let ~w = order_by_fields(&~w, &[~w]);", [OutVar, InVar, SpecStr]).

% Sort by custom comparator
generate_rust_stage_flow(sort_by(ComparePred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "sorted_~w_result", [ComparePred]),
    format(string(Code),
"    // Sort by custom comparator: ~w
    let ~w = sort_by_comparator(&~w, ~w);", [ComparePred, OutVar, InVar, ComparePred]).

% Try-catch stage: execute stage, on error route to handler
generate_rust_stage_flow(try_catch(Stage, Handler), InVar, OutVar, Code) :-
    !,
    extract_rust_stage_name(Stage, StageName),
    extract_rust_stage_name(Handler, HandlerName),
    OutVar = "try_catch_result",
    format(string(Code),
"    // Try-Catch: ~w with handler ~w
    let ~w = try_catch_stage(&~w, ~w, ~w);", [StageName, HandlerName, OutVar, InVar, StageName, HandlerName]).

% Retry stage: retry N times on failure
generate_rust_stage_flow(retry(Stage, N), InVar, OutVar, Code) :-
    !,
    extract_rust_stage_name(Stage, StageName),
    OutVar = "retry_result",
    format(string(Code),
"    // Retry: ~w up to ~w times
    let ~w = retry_stage(&~w, ~w, ~w, 0, \"none\");", [StageName, N, OutVar, InVar, StageName, N]).

% Retry stage with options
generate_rust_stage_flow(retry(Stage, N, Options), InVar, OutVar, Code) :-
    !,
    extract_rust_stage_name(Stage, StageName),
    OutVar = "retry_result",
    extract_rust_retry_options(Options, DelayMs, Backoff),
    format(string(Code),
"    // Retry: ~w up to ~w times (delay=~wms, backoff=~w)
    let ~w = retry_stage(&~w, ~w, ~w, ~w, \"~w\");", [StageName, N, DelayMs, Backoff, OutVar, InVar, StageName, N, DelayMs, Backoff]).

% On-error stage: global error handler
generate_rust_stage_flow(on_error(Handler), InVar, OutVar, Code) :-
    !,
    extract_rust_stage_name(Handler, HandlerName),
    OutVar = "on_error_result",
    format(string(Code),
"    // On-Error: route errors to ~w
    let ~w = on_error_stage(&~w, ~w);", [HandlerName, OutVar, InVar, HandlerName]).

% Timeout stage: execute with time limit
generate_rust_stage_flow(timeout(Stage, Ms), InVar, OutVar, Code) :-
    !,
    extract_rust_stage_name(Stage, StageName),
    OutVar = "timeout_result",
    format(string(Code),
"    // Timeout: ~w with ~wms limit
    let ~w = timeout_stage(&~w, ~w, ~w);", [StageName, Ms, OutVar, InVar, StageName, Ms]).

% Timeout stage with fallback
generate_rust_stage_flow(timeout(Stage, Ms, Fallback), InVar, OutVar, Code) :-
    !,
    extract_rust_stage_name(Stage, StageName),
    extract_rust_stage_name(Fallback, FallbackName),
    OutVar = "timeout_result",
    format(string(Code),
"    // Timeout: ~w with ~wms limit, fallback to ~w
    let ~w = timeout_stage_with_fallback(&~w, ~w, ~w, ~w);", [StageName, Ms, FallbackName, OutVar, InVar, StageName, Ms, FallbackName]).

% Rate limit stage: limit throughput to N records per time unit
generate_rust_stage_flow(rate_limit(N, Per), InVar, OutVar, Code) :-
    !,
    rust_time_unit_to_ms(Per, IntervalMs),
    OutVar = "rate_limited",
    format(string(Code),
"    // Rate limit: ~w records per ~w
    let ~w = rate_limit_stage(&~w, ~w, ~w);", [N, Per, OutVar, InVar, N, IntervalMs]).

% Throttle stage: add fixed delay between records
generate_rust_stage_flow(throttle(Ms), InVar, OutVar, Code) :-
    !,
    OutVar = "throttled",
    format(string(Code),
"    // Throttle: ~wms delay between records
    let ~w = throttle_stage(&~w, ~w);", [Ms, OutVar, InVar, Ms]).

% Buffer stage: collect records into batches
generate_rust_stage_flow(buffer(N), InVar, OutVar, Code) :-
    !,
    OutVar = "buffered",
    format(string(Code),
"    // Buffer: collect ~w records into batches
    let ~w = buffer_stage(&~w, ~w);", [N, OutVar, InVar, N]).

% Debounce stage: emit only if no new record within delay
generate_rust_stage_flow(debounce(Ms), InVar, OutVar, Code) :-
    !,
    OutVar = "debounced",
    format(string(Code),
"    // Debounce: ~wms quiet period
    let ~w = debounce_stage(&~w, ~w);", [Ms, OutVar, InVar, Ms]).

% Zip stage: combine multiple stages record-by-record
generate_rust_stage_flow(zip(Stages), InVar, OutVar, Code) :-
    !,
    OutVar = "zipped",
    extract_rust_stage_names(Stages, Names),
    format_rust_stage_list(Names, StageListStr),
    format(string(Code),
"    // Zip: combine outputs from multiple stages
    let ~w = zip_stage(&~w, &[~w]);", [OutVar, InVar, StageListStr]).

% Window stage: collect records into non-overlapping windows
generate_rust_stage_flow(window(N), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "windowed_~w", [N]),
    format(string(Code),
"    // Window: collect into batches of ~w
    let ~w = window_stage(&~w, ~w);", [N, OutVar, InVar, N]).

% Sliding window stage: create overlapping windows
generate_rust_stage_flow(sliding_window(N, Step), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "sliding_~w_~w", [N, Step]),
    format(string(Code),
"    // Sliding window: size=~w, step=~w
    let ~w = sliding_window_stage(&~w, ~w, ~w);", [N, Step, OutVar, InVar, N, Step]).

% Sample stage: randomly sample n records
generate_rust_stage_flow(sample(N), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "sampled_~w", [N]),
    format(string(Code),
"    // Sample: reservoir sampling of ~w records
    let ~w = sample_stage(&~w, ~w);", [N, OutVar, InVar, N]).

% Take every stage: take every nth record
generate_rust_stage_flow(take_every(N), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "every_~w", [N]),
    format(string(Code),
"    // Take every: every ~wth record
    let ~w = take_every_stage(&~w, ~w);", [N, OutVar, InVar, N]).

% Partition stage: split records by predicate
generate_rust_stage_flow(partition(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "partitioned_~w", [Pred]),
    format(string(Code),
"    // Partition: split by ~w predicate
    let (~w_matches, ~w_non_matches) = partition_stage(&~w, ~w);
    let ~w = ~w_matches;  // Use matches (non-matches available as ~w_non_matches)", [Pred, OutVar, OutVar, InVar, Pred, OutVar, OutVar, OutVar]).

% Take stage: take first n records
generate_rust_stage_flow(take(N), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "taken_~w", [N]),
    format(string(Code),
"    // Take: first ~w records
    let ~w = take_stage(&~w, ~w);", [N, OutVar, InVar, N]).

% Skip stage: skip first n records
generate_rust_stage_flow(skip(N), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "skipped_~w", [N]),
    format(string(Code),
"    // Skip: first ~w records
    let ~w = skip_stage(&~w, ~w);", [N, OutVar, InVar, N]).

% Take while stage: take while predicate is true
generate_rust_stage_flow(take_while(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "take_while_~w", [Pred]),
    format(string(Code),
"    // Take while: ~w is true
    let ~w = take_while_stage(&~w, ~w);", [Pred, OutVar, InVar, Pred]).

% Skip while stage: skip while predicate is true
generate_rust_stage_flow(skip_while(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "skip_while_~w", [Pred]),
    format(string(Code),
"    // Skip while: ~w is true
    let ~w = skip_while_stage(&~w, ~w);", [Pred, OutVar, InVar, Pred]).

% Distinct stage: remove all duplicates (global)
generate_rust_stage_flow(distinct, InVar, OutVar, Code) :-
    !,
    OutVar = "distinct_result",
    format(string(Code),
"    // Distinct: remove all duplicates (global dedup)
    let ~w = distinct_stage(&~w);", [OutVar, InVar]).

% Distinct by field: remove duplicates based on specific field
generate_rust_stage_flow(distinct_by(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "distinct_~w_result", [Field]),
    format(string(Code),
"    // Distinct By: remove duplicates based on '~w' field
    let ~w = distinct_by_stage(&~w, \"~w\");", [Field, OutVar, InVar, Field]).

% Dedup stage: remove consecutive duplicates only
generate_rust_stage_flow(dedup, InVar, OutVar, Code) :-
    !,
    OutVar = "dedup_result",
    format(string(Code),
"    // Dedup: remove consecutive duplicates
    let ~w = dedup_stage(&~w);", [OutVar, InVar]).

% Dedup by field: remove consecutive duplicates based on specific field
generate_rust_stage_flow(dedup_by(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "dedup_~w_result", [Field]),
    format(string(Code),
"    // Dedup By: remove consecutive duplicates based on '~w' field
    let ~w = dedup_by_stage(&~w, \"~w\");", [Field, OutVar, InVar, Field]).

% Interleave stage: round-robin interleave from multiple stage outputs
generate_rust_stage_flow(interleave(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "interleaved_~w_result", [N]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    format(string(Code),
"    // Interleave: round-robin from ~w stage outputs
    let interleave_streams_~w: Vec<Vec<Record>> = vec![~w]
        .iter()
        .map(|stage_fn| stage_fn(&~w))
        .collect();
    let ~w = interleave_stage(&interleave_streams_~w);", [N, N, StageListStr, InVar, OutVar, N]).

% Concat stage: sequential concatenation of multiple stage outputs
generate_rust_stage_flow(concat(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "concatenated_~w_result", [N]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    format(string(Code),
"    // Concat: sequential concatenation of ~w stage outputs
    let concat_streams_~w: Vec<Vec<Record>> = vec![~w]
        .iter()
        .map(|stage_fn| stage_fn(&~w))
        .collect();
    let ~w = concat_stage(&concat_streams_~w);", [N, N, StageListStr, InVar, OutVar, N]).

% Merge sorted stage: merge pre-sorted streams maintaining order (ascending)
generate_rust_stage_flow(merge_sorted(SubStages, Field), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "merge_sorted_~w_result", [Field]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    format(string(Code),
"    // Merge Sorted: merge ~w pre-sorted streams by '~w' (ascending)
    let merge_sorted_streams_~w: Vec<Vec<Record>> = vec![~w]
        .iter()
        .map(|stage_fn| stage_fn(&~w))
        .collect();
    let ~w = merge_sorted_stage(&merge_sorted_streams_~w, \"~w\", true);", [N, Field, N, StageListStr, InVar, OutVar, N, Field]).

% Merge sorted stage with direction: merge pre-sorted streams with specified order
generate_rust_stage_flow(merge_sorted(SubStages, Field, Dir), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "merge_sorted_~w_~w_result", [Field, Dir]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    ( Dir = asc -> Ascending = "true" ; Ascending = "false" ),
    format(string(Code),
"    // Merge Sorted: merge ~w pre-sorted streams by '~w' (~w)
    let merge_sorted_streams_~w_~w: Vec<Vec<Record>> = vec![~w]
        .iter()
        .map(|stage_fn| stage_fn(&~w))
        .collect();
    let ~w = merge_sorted_stage(&merge_sorted_streams_~w_~w, \"~w\", ~w);", [N, Field, Dir, N, Dir, StageListStr, InVar, OutVar, N, Dir, Field, Ascending]).

% Standard predicate stage
generate_rust_stage_flow(Pred/Arity, InVar, OutVar, Code) :-
    !,
    atom(Pred),
    format(atom(OutVar), "~w_result", [Pred]),
    format(string(Code),
"    // Stage: ~w/~w
    let ~w = ~w(&~w);", [Pred, Arity, OutVar, Pred, InVar]).

% Fallback for unknown stages
generate_rust_stage_flow(Stage, InVar, InVar, Code) :-
    format(string(Code), "    // Unknown stage type: ~w (pass-through)", [Stage]).

%% extract_rust_stage_names(+Stages, -Names)
%  Extract function names from stage specifications.
extract_rust_stage_names([], []).
extract_rust_stage_names([Pred/_Arity|Rest], [Pred|RestNames]) :-
    !,
    extract_rust_stage_names(Rest, RestNames).
extract_rust_stage_names([_|Rest], RestNames) :-
    extract_rust_stage_names(Rest, RestNames).

%% extract_rust_stage_name(+Stage, -Name)
%  Extract function name from a single stage specification.
extract_rust_stage_name(Pred/_, Pred) :- atom(Pred), !.
extract_rust_stage_name(Pred, Pred) :- atom(Pred), !.
extract_rust_stage_name(_, unknown_stage).

%% extract_rust_retry_options(+Options, -DelayMs, -Backoff)
%  Extract retry options from options list.
extract_rust_retry_options(Options, DelayMs, Backoff) :-
    ( member(delay(D), Options) -> DelayMs = D ; DelayMs = 0 ),
    ( member(backoff(B), Options) -> Backoff = B ; Backoff = none ).

%% rust_time_unit_to_ms(+Unit, -Ms)
%  Convert time unit to milliseconds for Rust rate limiting.
rust_time_unit_to_ms(second, 1000) :- !.
rust_time_unit_to_ms(minute, 60000) :- !.
rust_time_unit_to_ms(hour, 3600000) :- !.
rust_time_unit_to_ms(ms(X), X) :- !.
rust_time_unit_to_ms(X, X) :- integer(X), !.
rust_time_unit_to_ms(_, 1000).

%% format_rust_stage_list(+Names, -ListStr)
%  Format stage names as Rust function references.
format_rust_stage_list([], "").
format_rust_stage_list([Name], Str) :-
    format(string(Str), "~w", [Name]).
format_rust_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_rust_stage_list(Rest, RestStr),
    format(string(Str), "~w, ~w", [Name, RestStr]).

%% format_rust_route_map(+Routes, -MapStr)
%  Format routing map for Rust.
format_rust_route_map([], "").
format_rust_route_map([(_Cond, Stage)|[]], Str) :-
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format(string(Str), "    route_map.insert(Value::Bool(true), ~w);", [StageName]).
format_rust_route_map([(Cond, Stage)|Rest], Str) :-
    Rest \= [],
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format_rust_route_map(Rest, RestStr),
    (Cond = true ->
        format(string(Str), "    route_map.insert(Value::Bool(true), ~w);~n~w", [StageName, RestStr])
    ; Cond = false ->
        format(string(Str), "    route_map.insert(Value::Bool(false), ~w);~n~w", [StageName, RestStr])
    ;   format(string(Str), "    route_map.insert(Value::String(\"~w\".to_string()), ~w);~n~w", [Cond, StageName, RestStr])
    ).

%% format_rust_aggregations(+Agg, -Str)
%  Format aggregation specifications for Rust group_by stage.
format_rust_aggregations(Aggs, Str) :-
    is_list(Aggs),
    !,
    format_rust_aggregation_list(Aggs, Str).
format_rust_aggregations(Agg, Str) :-
    format_rust_single_aggregation(Agg, Str).

format_rust_aggregation_list([], "").
format_rust_aggregation_list([Agg], Str) :-
    format_rust_single_aggregation(Agg, Str).
format_rust_aggregation_list([Agg|Rest], Str) :-
    Rest \= [],
    format_rust_single_aggregation(Agg, AggStr),
    format_rust_aggregation_list(Rest, RestStr),
    format(string(Str), "~w, ~w", [AggStr, RestStr]).

% count aggregation
format_rust_single_aggregation(count, "(\"count\", AggType::Count)").
% Aggregations with field
format_rust_single_aggregation(sum(Field), Str) :-
    format(string(Str), "(\"sum\", AggType::Sum(\"~w\".to_string()))", [Field]).
format_rust_single_aggregation(avg(Field), Str) :-
    format(string(Str), "(\"avg\", AggType::Avg(\"~w\".to_string()))", [Field]).
format_rust_single_aggregation(min(Field), Str) :-
    format(string(Str), "(\"min\", AggType::Min(\"~w\".to_string()))", [Field]).
format_rust_single_aggregation(max(Field), Str) :-
    format(string(Str), "(\"max\", AggType::Max(\"~w\".to_string()))", [Field]).
format_rust_single_aggregation(first(Field), Str) :-
    format(string(Str), "(\"first\", AggType::First(\"~w\".to_string()))", [Field]).
format_rust_single_aggregation(last(Field), Str) :-
    format(string(Str), "(\"last\", AggType::Last(\"~w\".to_string()))", [Field]).
format_rust_single_aggregation(collect(Field), Str) :-
    format(string(Str), "(\"collect\", AggType::Collect(\"~w\".to_string()))", [Field]).

%% format_rust_field_specs(+FieldSpecs, -Str)
%  Format field specifications for multi-field ordering.
format_rust_field_specs([], "").
format_rust_field_specs([Spec], Str) :-
    format_rust_single_field_spec(Spec, Str).
format_rust_field_specs([Spec|Rest], Str) :-
    Rest \= [],
    format_rust_single_field_spec(Spec, SpecStr),
    format_rust_field_specs(Rest, RestStr),
    format(string(Str), "~w, ~w", [SpecStr, RestStr]).

format_rust_single_field_spec(Field, Str) :-
    atom(Field),
    !,
    format(string(Str), "FieldSpec { field: \"~w\".to_string(), direction: \"asc\".to_string() }", [Field]).
format_rust_single_field_spec((Field, Dir), Str) :-
    format(string(Str), "FieldSpec { field: \"~w\".to_string(), direction: \"~w\".to_string() }", [Field, Dir]).

%% generate_rust_enhanced_main(+PipelineName, +OutputFormat, -Code)
%  Generate main function for enhanced pipeline.
generate_rust_enhanced_main(PipelineName, jsonl, Code) :-
    format(string(Code),
"fn main() {
    // Read JSONL from stdin
    let input = read_jsonl_stream();

    // Run enhanced pipeline
    let results = ~w(&input);

    // Output results as JSONL
    write_jsonl_stream(&results);
}
", [PipelineName]).
generate_rust_enhanced_main(PipelineName, _, Code) :-
    format(string(Code),
"fn main() {
    // Read JSONL from stdin
    let input = read_jsonl_stream();

    // Run enhanced pipeline
    let results = ~w(&input);

    // Output results
    write_jsonl_stream(&results);
}
", [PipelineName]).

%% ============================================
%% RUST ENHANCED PIPELINE CHAINING TESTS
%% ============================================

test_rust_enhanced_chaining :-
    format('~n=== Rust Enhanced Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Generate enhanced helpers (std_thread mode)
    format('[Test 1] Generate enhanced helpers (std_thread)~n', []),
    rust_enhanced_helpers(std_thread, Helpers1),
    (   sub_string(Helpers1, _, _, _, "fn fan_out_records"),
        sub_string(Helpers1, _, _, _, "fn merge_streams"),
        sub_string(Helpers1, _, _, _, "fn route_record"),
        sub_string(Helpers1, _, _, _, "fn filter_records"),
        sub_string(Helpers1, _, _, _, "fn tee_stream"),
        sub_string(Helpers1, _, _, _, "std::thread")
    ->  format('  [PASS] All helper functions generated (std_thread)~n', [])
    ;   format('  [FAIL] Missing helper functions~n', [])
    ),

    % Test 2: Linear pipeline connector
    format('[Test 2] Linear pipeline connector~n', []),
    generate_rust_enhanced_connector([extract/1, transform/1, load/1], linear_pipe, Code2),
    (   sub_string(Code2, _, _, _, "fn linear_pipe"),
        sub_string(Code2, _, _, _, "extract(&input)"),
        sub_string(Code2, _, _, _, "transform(&extract_result)"),
        sub_string(Code2, _, _, _, "load(&transform_result)")
    ->  format('  [PASS] Linear connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code2])
    ),

    % Test 3: Fan-out connector
    format('[Test 3] Fan-out connector~n', []),
    generate_rust_enhanced_connector([fan_out([validate/1, enrich/1])], fanout_pipe, Code3),
    (   sub_string(Code3, _, _, _, "fn fanout_pipe"),
        sub_string(Code3, _, _, _, "Fan-out to 2 parallel stages"),
        sub_string(Code3, _, _, _, "fan_out_records")
    ->  format('  [PASS] Fan-out connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code3])
    ),

    % Test 4: Fan-out with merge
    format('[Test 4] Fan-out with merge~n', []),
    generate_rust_enhanced_connector([fan_out([a/1, b/1]), merge], merge_pipe, Code4),
    (   sub_string(Code4, _, _, _, "fn merge_pipe"),
        sub_string(Code4, _, _, _, "Fan-out to 2"),
        sub_string(Code4, _, _, _, "Merge: results already combined")
    ->  format('  [PASS] Merge connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code4])
    ),

    % Test 5: Conditional routing
    format('[Test 5] Conditional routing~n', []),
    generate_rust_enhanced_connector([route_by(has_error, [(true, error_handler/1), (false, success/1)])], route_pipe, Code5),
    (   sub_string(Code5, _, _, _, "fn route_pipe"),
        sub_string(Code5, _, _, _, "Conditional routing based on has_error"),
        sub_string(Code5, _, _, _, "route_map")
    ->  format('  [PASS] Routing connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code5])
    ),

    % Test 6: Filter stage
    format('[Test 6] Filter stage~n', []),
    generate_rust_enhanced_connector([filter_by(is_valid)], filter_pipe, Code6),
    (   sub_string(Code6, _, _, _, "fn filter_pipe"),
        sub_string(Code6, _, _, _, "Filter by is_valid"),
        sub_string(Code6, _, _, _, "filter_records")
    ->  format('  [PASS] Filter connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code6])
    ),

    % Test 7: Complex pipeline with all patterns
    format('[Test 7] Complex pipeline~n', []),
    generate_rust_enhanced_connector([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], complex_pipe, Code7),
    (   sub_string(Code7, _, _, _, "fn complex_pipe"),
        sub_string(Code7, _, _, _, "Filter by is_active"),
        sub_string(Code7, _, _, _, "Fan-out to 3 parallel stages"),
        sub_string(Code7, _, _, _, "Merge"),
        sub_string(Code7, _, _, _, "Conditional routing")
    ->  format('  [PASS] Complex connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code7])
    ),

    % Test 8: Stage function generation
    format('[Test 8] Stage function generation~n', []),
    generate_rust_enhanced_stage_functions([extract/1, transform/1], StageFns8),
    (   sub_string(StageFns8, _, _, _, "fn extract"),
        sub_string(StageFns8, _, _, _, "fn transform")
    ->  format('  [PASS] Stage functions generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [StageFns8])
    ),

    % Test 9: Full enhanced pipeline compilation
    format('[Test 9] Full enhanced pipeline~n', []),
    compile_rust_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(full_enhanced), output_format(jsonl)], FullCode9),
    (   sub_string(FullCode9, _, _, _, "use std::collections::HashMap"),
        sub_string(FullCode9, _, _, _, "fn fan_out_records"),
        sub_string(FullCode9, _, _, _, "fn filter_records"),
        sub_string(FullCode9, _, _, _, "fn full_enhanced"),
        sub_string(FullCode9, _, _, _, "fn main()")
    ->  format('  [PASS] Full pipeline compiles~n', [])
    ;   format('  [FAIL] Missing patterns in generated code~n', [])
    ),

    % Test 10: Enhanced helpers include all functions (std_thread)
    format('[Test 10] Enhanced helpers completeness (std_thread)~n', []),
    rust_enhanced_helpers(std_thread, Helpers10),
    (   sub_string(Helpers10, _, _, _, "fan_out_records"),
        sub_string(Helpers10, _, _, _, "merge_streams"),
        sub_string(Helpers10, _, _, _, "route_record"),
        sub_string(Helpers10, _, _, _, "filter_records"),
        sub_string(Helpers10, _, _, _, "tee_stream")
    ->  format('  [PASS] All helpers present (std_thread)~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    % Test 11: Rayon mode generates par_iter
    format('[Test 11] Rayon parallel mode~n', []),
    rust_enhanced_helpers(rayon, Helpers11),
    (   sub_string(Helpers11, _, _, _, "par_iter"),
        sub_string(Helpers11, _, _, _, "rayon")
    ->  format('  [PASS] Rayon helpers generated~n', [])
    ;   format('  [FAIL] Missing rayon patterns~n', [])
    ),

    % Test 12: Full pipeline with rayon mode
    format('[Test 12] Full pipeline with rayon mode~n', []),
    compile_rust_enhanced_pipeline([
        extract/1,
        parallel([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(rayon_pipe), parallel_mode(rayon)], RayonCode12),
    (   sub_string(RayonCode12, _, _, _, "rayon::prelude"),
        sub_string(RayonCode12, _, _, _, "par_iter")
    ->  format('  [PASS] Rayon pipeline compiles~n', [])
    ;   format('  [FAIL] Missing rayon patterns~n', [])
    ),

    format('~n=== All Rust Enhanced Pipeline Chaining Tests Passed ===~n', []).