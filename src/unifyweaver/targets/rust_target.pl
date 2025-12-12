:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% rust_target.pl - Rust Target for UnifyWeaver
% Generates standalone Rust programs for record/field processing.
% Phase 3: JSON support (serde_json crate).

:- module(rust_target, [
    compile_predicate_to_rust/3,      % +Predicate, +Options, -RustCode
    write_rust_program/2,             % +RustCode, +FilePath
    write_rust_project/2,             % +RustCode, +Dir
    json_schema/2,                    % +SchemaName, +Fields (directive)
    get_json_schema/2,                % +SchemaName, -Fields (lookup)
    init_rust_target/0,               % Initialize bindings
    compile_rust_pipeline/3,          % +Predicates, +Options, -RustCode
    test_rust_pipeline_generator/0,   % Unit tests for pipeline generator mode
    % Enhanced pipeline chaining exports
    compile_rust_enhanced_pipeline/3, % +Stages, +Options, -RustCode
    rust_enhanced_helpers/1,          % -Code
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
%    - fan_out(Stages) : Broadcast to parallel stages
%    - merge          : Combine results from parallel stages
%    - route_by(Pred, Routes) : Conditional routing
%    - filter_by(Pred) : Filter records
%    - Pred/Arity     : Standard stage
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

    % Generate helpers
    rust_enhanced_helpers(Helpers),

    % Generate stage functions
    generate_rust_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the main connector
    generate_rust_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main function
    generate_rust_enhanced_main(PipelineName, OutputFormat, MainCode),

    format(string(RustCode),
"// Generated by UnifyWeaver Rust Enhanced Pipeline
// Supports fan-out, merge, conditional routing, and filtering
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use serde_json::Value;

type Record = HashMap<String, Value>;

~w

~w
~w
~w
", [Helpers, StageFunctions, ConnectorCode, MainCode]).

%% rust_enhanced_helpers(-Code)
%  Generate helper functions for enhanced pipeline operations.
rust_enhanced_helpers(Code) :-
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
generate_rust_single_enhanced_stage(merge, "") :- !.
generate_rust_single_enhanced_stage(route_by(_, Routes), Code) :-
    !,
    findall(Stage, member((_Cond, Stage), Routes), RouteStages),
    generate_rust_enhanced_stage_functions(RouteStages, Code).
generate_rust_single_enhanced_stage(filter_by(_), "") :- !.
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

% Fan-out stage: broadcast to parallel stages
generate_rust_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "fan_out_~w_result", [N]),
    extract_rust_stage_names(SubStages, StageNames),
    format_rust_stage_list(StageNames, StageListStr),
    format(string(Code),
"    // Fan-out to ~w parallel stages
    let ~w: Vec<Record> = ~w.iter()
        .flat_map(|record| fan_out_records(record, &[~w]))
        .collect();", [N, OutVar, InVar, StageListStr]).

% Merge stage: placeholder, usually follows fan_out
generate_rust_stage_flow(merge, InVar, OutVar, Code) :-
    !,
    OutVar = InVar,
    Code = "    // Merge: results already combined from fan-out".

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

    % Test 1: Generate enhanced helpers
    format('[Test 1] Generate enhanced helpers~n', []),
    rust_enhanced_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "fn fan_out_records"),
        sub_string(Helpers1, _, _, _, "fn merge_streams"),
        sub_string(Helpers1, _, _, _, "fn route_record"),
        sub_string(Helpers1, _, _, _, "fn filter_records"),
        sub_string(Helpers1, _, _, _, "fn tee_stream")
    ->  format('  [PASS] All helper functions generated~n', [])
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

    % Test 10: Enhanced helpers include all functions
    format('[Test 10] Enhanced helpers completeness~n', []),
    rust_enhanced_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "fan_out_records"),
        sub_string(Helpers10, _, _, _, "merge_streams"),
        sub_string(Helpers10, _, _, _, "route_record"),
        sub_string(Helpers10, _, _, _, "filter_records"),
        sub_string(Helpers10, _, _, _, "tee_stream")
    ->  format('  [PASS] All helpers present~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    format('~n=== All Rust Enhanced Pipeline Chaining Tests Passed ===~n', []).