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
    init_rust_target/0                % Initialize bindings
]).

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(gensym)). % For generating unique variable names
:- use_module(library(yall)).
:- use_module('../core/binding_registry').
:- use_module('../bindings/rust_bindings').

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