:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% csharp_stream_target.pl - C# streaming target for UnifyWeaver
% Generates LINQ-based streaming code for non-recursive predicates.

:- module(csharp_stream_target, [
    compile_predicate_to_csharp/3,   % +PredIndicator, +Options, -CSharpCode
    write_csharp_streamer/2,         % +CSharpCode, +FilePath
    test_csharp_stream_target/0
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(date)).

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_csharp(+Pred/Arity, +Options, -Code)
%  Compile a non-recursive predicate to C# streaming code.
%  Currently supports single-clause streaming rules composed from binary relations.
compile_predicate_to_csharp(PredIndicator, Options, Code) :-
    PredIndicator = Pred/Arity,
    functor(Head, Pred, Arity),
    findall(Body, clause(Head, Body), Bodies),
    classify_predicate(Bodies, Type),
    compile_by_type(Type, Pred, Arity, Bodies, Options, Code).

%% write_csharp_streamer(+Code, +FilePath)
%  Persist generated C# code to disk.
write_csharp_streamer(Code, FilePath) :-
    setup_call_cleanup(
        open(FilePath, write, Stream),
        write(Stream, Code),
        close(Stream)
    ),
    format('[CSharpTarget] Generated C# streamer: ~w~n', [FilePath]).

%% ============================================
%% PREDICATE CLASSIFICATION
%% ============================================

classify_predicate([], none).
classify_predicate(Bodies, facts) :-
    Bodies \= [],
    forall(member(Body, Bodies), Body == true), !.
classify_predicate([Body], single_rule) :-
    Body \= true, !.
classify_predicate(Bodies, multiple_rules) :-
    Bodies \= [],
    Bodies \= [true],
    Bodies \= [_], !.
classify_predicate(_Bodies, unsupported).

compile_by_type(none, Pred, Arity, _Bodies, _Options, _) :-
    format('[CSharpTarget] ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
    fail.
compile_by_type(facts, Pred, Arity, _Bodies, Options, Code) :-
    compile_facts_to_csharp(Pred, Arity, Options, Code).
compile_by_type(multiple_rules, Pred, Arity, Bodies, Options, Code) :-
    compile_multiple_rules_to_csharp(Pred, Arity, Bodies, Options, Code).
compile_by_type(unsupported, Pred, Arity, _Bodies, _Options, _) :-
    format('[CSharpTarget] ERROR: Unsupported predicate form for C# target (~w/~w)~n', [Pred, Arity]),
    fail.
compile_by_type(single_rule, Pred, Arity, [Body], Options, Code) :-
    compile_single_rule_to_csharp(Pred, Arity, Body, Options, Code).

%% ============================================
%% SINGLE RULE COMPILATION
%% ============================================

compile_single_rule_to_csharp(Pred, Arity, Body, Options, Code) :-
    collect_predicate_terms(Body, Terms),
    Terms \= [],
    maplist(term_signature, Terms, Signatures),
    list_to_set(Signatures, UniqueSigs),
    ensure_binary_relations(UniqueSigs, Pred/Arity),
    maplist(gather_fact_entries, UniqueSigs, FactEntries),
    maplist(data_section_for_signature, FactEntries, DataSections),
    maplist(stream_helper_for_signature, UniqueSigs, StreamHelpers),
    predicate_name_pascal(Pred, ModuleName),
    predicate_name_pascal(Pred, TargetName),
    maplist(predicate_name_pascal_signature, Signatures, PredicateOrder),
    generate_linq_pipeline(PredicateOrder, PipelineCode),
    maybe_distinct(PipelineCode, Options, PipelineWithDedup),
    fact_result_type(Arity, ResultType),
    fact_print_expression(Arity, PrintExpr),
    compose_csharp_program(ModuleName, DataSections, StreamHelpers,
        TargetName, ResultType, PipelineWithDedup, PrintExpr, Code).

%% Compile fact-only predicates into simple stream emitters
compile_facts_to_csharp(Pred, Arity, Options, Code) :-
    gather_fact_entries(Pred/Arity, FactInfo),
    FactInfo = fact_info(_, Arity, Entries),
    Entries \= [],
    fact_result_type(Arity, ResultType),
    fact_print_expression(Arity, PrintExpr),
    predicate_name_pascal(Pred, ModuleName),
    predicate_name_pascal(Pred, TargetName),
    fact_data_section(FactInfo, DataSection),
    fact_stream_helper(FactInfo, ResultType, HelperName, HelperSection),
    format(atom(PipelineBase), '~w()', [HelperName]),
    maybe_distinct(PipelineBase, Options, PipelineWithDedup),
    compose_csharp_program(ModuleName, [DataSection], [HelperSection],
        TargetName, ResultType, PipelineWithDedup, PrintExpr, Code).
compile_facts_to_csharp(Pred, Arity, _Options, _) :-
    \+ member(Arity, [1, 2]),
    format('[CSharpTarget] ERROR: Fact-only predicates of arity ~w are not supported for ~w/~w in the C# target.~n',
           [Arity, Pred, Arity]),
    fail.

%% Compile multi-clause predicates (OR combinations)
compile_multiple_rules_to_csharp(Pred, Arity, Bodies, Options, Code) :-
    predicate_name_pascal(Pred, ModuleName),
    predicate_name_pascal(Pred, TargetName),
    fact_result_type(Arity, ResultType),
    fact_print_expression(Arity, PrintExpr),
    process_clause_bodies(Bodies, Pred, Arity, Options, TargetName, 1, false,
        [], [], [], DataSections0, StreamHelpers0, ClauseCalls),
    (   ClauseCalls = []
    ->  format('[CSharpTarget] ERROR: Unable to generate C# stream for ~w/~w (no valid clauses).~n', [Pred, Arity]),
        fail
    ;   true
    ),
    list_to_set(DataSections0, DataSections),
    list_to_set(StreamHelpers0, StreamHelpers),
    build_concat_pipeline(ClauseCalls, CombinedPipeline0),
    maybe_distinct(CombinedPipeline0, Options, CombinedPipeline),
    compose_csharp_program(ModuleName, DataSections, StreamHelpers,
        TargetName, ResultType, CombinedPipeline, PrintExpr, Code).

process_clause_bodies([], _Pred, _Arity, _Options, _TargetName, _Index, _FactIncluded,
        DataAcc, HelperAcc, ClauseAcc, DataAcc, HelperAcc, ClauseAcc).
process_clause_bodies([Body|Rest], Pred, Arity, Options, TargetName, Index, FactIncluded,
        DataAcc, HelperAcc, ClauseAcc, DataOut, HelperOut, ClauseOut) :-
    (   Body == true
    ->  (   FactIncluded
        ->  NextIndex is Index + 1,
            process_clause_bodies(Rest, Pred, Arity, Options, TargetName, NextIndex,
                FactIncluded, DataAcc, HelperAcc, ClauseAcc, DataOut, HelperOut, ClauseOut)
        ;   build_fact_clause_components(Pred, Arity, TargetName, Index,
                ClauseData, ClauseHelpers, ClauseCall),
            append(DataAcc, ClauseData, DataAcc1),
            append(HelperAcc, ClauseHelpers, HelperAcc1),
            append(ClauseAcc, [ClauseCall], ClauseAcc1),
            NextIndex is Index + 1,
            process_clause_bodies(Rest, Pred, Arity, Options, TargetName, NextIndex,
                true, DataAcc1, HelperAcc1, ClauseAcc1, DataOut, HelperOut, ClauseOut)
        )
    ;   build_rule_clause_components(Pred, Arity, Body, TargetName, Index,
            ClauseData, ClauseHelpers, ClauseCall),
        append(DataAcc, ClauseData, DataAcc1),
        append(HelperAcc, ClauseHelpers, HelperAcc1),
        append(ClauseAcc, [ClauseCall], ClauseAcc1),
        NextIndex is Index + 1,
        process_clause_bodies(Rest, Pred, Arity, Options, TargetName, NextIndex,
            FactIncluded, DataAcc1, HelperAcc1, ClauseAcc1, DataOut, HelperOut, ClauseOut)
    ).

build_rule_clause_components(Pred, Arity, Body, TargetName, Index,
        DataSections, Helpers, ClauseCall) :-
    collect_predicate_terms(Body, Terms),
    Terms \= [],
    maplist(term_signature, Terms, Signatures),
    list_to_set(Signatures, UniqueSigs),
    ensure_binary_relations(UniqueSigs, Pred/Arity),
    maplist(gather_fact_entries, UniqueSigs, FactEntries),
    maplist(data_section_for_signature, FactEntries, DataSections0),
    maplist(stream_helper_for_signature, UniqueSigs, StreamHelpers0),
    maplist(predicate_name_pascal_signature, Signatures, PredicateOrder),
    generate_linq_pipeline(PredicateOrder, PipelineCode),
    format(atom(ClauseName), '~wAlt~wStream', [TargetName, Index]),
    fact_result_type(Arity, ResultType),
    format(atom(ClauseMethod),
'        public static IEnumerable<~w> ~w()
        {
            return
                ~w;
        }', [ResultType, ClauseName, PipelineCode]),
    append(StreamHelpers0, [ClauseMethod], Helpers),
    DataSections = DataSections0,
    format(atom(ClauseCall), '~w()', [ClauseName]).

build_fact_clause_components(Pred, Arity, TargetName, Index,
        DataSections, Helpers, ClauseCall) :-
    gather_fact_entries(Pred/Arity, FactInfo),
    FactInfo = fact_info(_, Arity, Entries),
    Entries \= [],
    fact_result_type(Arity, ResultType),
    fact_data_section(FactInfo, DataSection),
    fact_stream_helper(FactInfo, ResultType, HelperName, HelperCode),
    format(atom(ClauseName), '~wAlt~wStream', [TargetName, Index]),
    format(atom(ClauseMethod),
'        public static IEnumerable<~w> ~w()
        {
            return
                ~w();
        }', [ResultType, ClauseName, HelperName]),
    DataSections = [DataSection],
    Helpers = [HelperCode, ClauseMethod],
    format(atom(ClauseCall), '~w()', [ClauseName]).

build_concat_pipeline([First|Rest], Pipeline) :-
    foldl(concat_pipeline_expr, Rest, First, Pipeline).

concat_pipeline_expr(ClauseCall, Acc, Combined) :-
    format(atom(Combined), '~w\n            .Concat(~w)', [Acc, ClauseCall]).



%% Ensure we only handle binary relations for now
ensure_binary_relations([], _).
ensure_binary_relations([_Name/2|Rest], Target) :-
    ensure_binary_relations(Rest, Target).
ensure_binary_relations([Name/Arity|_], Target) :-
    Arity \= 2,
    format('[CSharpTarget] ERROR: Unsupported arity (~w) for predicate ~w when compiling ~w~n',
           [Arity, Name, Target]),
    fail.

%% ============================================
%% DATA COLLECTION HELPERS
%% ============================================

collect_predicate_terms(true, []) :- !.
collect_predicate_terms((A, B), Terms) :- !,
    collect_predicate_terms(A, TermsA),
    collect_predicate_terms(B, TermsB),
    append(TermsA, TermsB, Terms).
collect_predicate_terms(Goal, []) :-
    var(Goal), !.
collect_predicate_terms(_ \= _, []) :- !.
collect_predicate_terms(\=(_, _), []) :- !.
collect_predicate_terms(Goal, [Goal]).

term_signature(Goal, Name/Arity) :-
    functor(Goal, Name, Arity).

gather_fact_entries(Name/Arity, fact_info(Name, Arity, Entries)) :-
    functor(Head, Name, Arity),
    findall(Args,
        ( clause(Head, true),
          Head =.. [_|Args]),
        Entries).

%% ============================================
%% CODE GENERATION HELPERS
%% ============================================

predicate_name_pascal(Pred, Pascal) :-
    atom_string(Pred, PredStr),
    split_string(PredStr, '_', '_', Parts),
    maplist(capitalize_first, Parts, Caps),
    atomic_list_concat(Caps, '', Pascal).

predicate_name_pascal_signature(Name/_, Pascal) :-
    predicate_name_pascal(Name, Pascal).

capitalize_first(Part, Capitalized) :-
    (   Part == ""
    ->  Capitalized = ""
    ;   sub_string(Part, 0, 1, RestLen, First),
        sub_string(Part, 1, RestLen, 0, Rest),
        string_upper(First, UpperFirst),
        string_concat(UpperFirst, Rest, Capitalized)
    ).

data_section_for_signature(fact_info(Name, 1, Entries), Section) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), '~wData', [Pascal]),
    with_output_to(atom(Section),
        ( format('        private static readonly string[] ~w = new[] {~n', [DataName]),
          (   Entries = []
          ->  true
          ;   maplist(unary_literal, Entries, LiteralTuples),
              emit_literal_block(LiteralTuples, "            ")
          ),
          format('        };', [])
        )).

data_section_for_signature(fact_info(Name, 2, Entries), Section) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), '~wData', [Pascal]),
    with_output_to(atom(Section),
        ( format('        private static readonly (string, string)[] ~w = new[] {~n', [DataName]),
          (   Entries = []
          ->  true
          ;   maplist(tuple_literal, Entries, LiteralTuples),
              emit_literal_block(LiteralTuples, "            ")
          ),
          format('        };', [])
        )).

data_section_for_signature(fact_info(Name, Arity, _), _) :-
    \+ member(Arity, [1, 2]),
    format('[CSharpTarget] ERROR: data emission for arity ~w of predicate ~w is unimplemented~n', [Arity, Name]),
    fail.

emit_literal_block([], _).
emit_literal_block([First|Rest], Indent) :-
    format('~s~w', [Indent, First]),
    forall(member(Literal, Rest),
           format(',~n~s~w', [Indent, Literal])),
    format('~n', []).

stream_helper_for_signature(Name/1, HelperCode) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), '~wData', [Pascal]),
    format(atom(HelperCode),
'        public static IEnumerable<string> ~wStream()
        {
            return ~w;
        }', [Pascal, DataName]).

stream_helper_for_signature(Name/2, HelperCode) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), '~wData', [Pascal]),
    format(atom(HelperCode),
'        public static IEnumerable<(string, string)> ~wStream()
        {
            return ~w;
        }', [Pascal, DataName]).

stream_helper_for_signature(Name/Arity, _) :-
    \+ member(Arity, [1, 2]),
    format('[CSharpTarget] ERROR: stream helper for arity ~w of predicate ~w is unimplemented~n', [Arity, Name]),
    fail.


fact_result_type(1, 'string').
fact_result_type(2, '(string, string)').

fact_print_expression(1, 'Console.WriteLine(item);').
fact_print_expression(2, 'Console.WriteLine("${item.Item1}:{item.Item2}");').

fact_data_section(FactInfo, Section) :-
    data_section_for_signature(FactInfo, Section).

fact_stream_helper(fact_info(Name, _Arity, _Entries), ResultType, HelperName, HelperCode) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), '~wData', [Pascal]),
    format(atom(HelperName), '~wFactStream', [Pascal]),
    format(atom(HelperCode),
'        public static IEnumerable<~w> ~w()
        {
            return ~w;
        }', [ResultType, HelperName, DataName]).

tuple_literal([A, B], Literal) :-
    escape_csharp_string(A, AEsc),
    escape_csharp_string(B, BEsc),
    format(atom(Literal), '("~w", "~w")', [AEsc, BEsc]).

unary_literal([A], Literal) :-
    escape_csharp_string(A, Esc),
    format(atom(Literal), '"~w"', [Esc]).

escape_csharp_string(Atom, Escaped) :-
    atom_string(Atom, String),
    replace_substring(String, "\\", "\\\\", Step1),
    replace_substring(Step1, "\"", "\\\"", Step2),
    replace_substring(Step2, "\n", "\\n", Escaped).

replace_substring(String, Pattern, Replacement, Result) :-
    split_string(String, Pattern, Pattern, Parts),
    atomic_list_concat(Parts, Replacement, Result).

generate_linq_pipeline([First|Rest], Pipeline) :-
    format(atom(Start), '~wStream()', [First]),
    foldl(linq_join_step, Rest, Start, Pipeline).

generate_linq_pipeline([], 'Enumerable.Empty<(string, string)>()').

linq_join_step(PredName, Acc, Out) :-
    format(atom(Out),
'~w
            .Join(~wStream(),
                  left => left.Item2,
                  right => right.Item1,
                  (left, right) => (left.Item1, right.Item2))', [Acc, PredName]).

maybe_distinct(Pipeline, Options, PipelineWithDistinct) :-
    (   member(unique(UniqueFlag), Options),
        UniqueFlag == true
    ->  format(atom(PipelineWithDistinct), '~w~n            .Distinct()', [Pipeline])
    ;   PipelineWithDistinct = Pipeline
    ).

compose_csharp_program(ModuleName, DataSections, StreamHelpers,
        TargetName, ResultType, Pipeline, PrintExpr, Code) :-
    get_time(Timestamp),
    format_time(atom(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    format(atom(HeaderComment),
'// Generated by UnifyWeaver v0.0.3
// Target: C# Streaming (LINQ)
// Generated: ~w
', [DateStr]),
    (   DataSections = []
    ->  DataBlock = ''
    ;   atomic_list_concat(DataSections, '\n\n', DataConcat),
        format(atom(DataBlock), '~w\n\n', [DataConcat])
    ),
    (   StreamHelpers = []
    ->  HelperBlock = ''
    ;   atomic_list_concat(StreamHelpers, '\n\n', HelperConcat),
        format(atom(HelperBlock), '~w\n\n', [HelperConcat])
    ),
    format(atom(Program),
'~wusing System;
using System.Collections.Generic;
using System.Linq;

namespace UnifyWeaver.Generated
{
    public static class ~wModule
    {
~w~w        public static IEnumerable<~w> ~wStream()
        {
            return
                ~w;
        }

        public static void Main(string[] args)
        {
            foreach (var item in ~wStream())
            {
                ~w
            }
        }
    }
}', [HeaderComment, ModuleName, DataBlock, HelperBlock, ResultType, TargetName, Pipeline, TargetName, PrintExpr]),
    normalize_whitespace(Program, Code).

normalize_whitespace(Input, Output) :-
    split_string(Input, '\n', '\n', Lines),
    maplist(trim_trailing_spaces, Lines, Trimmed),
    atomic_list_concat(Trimmed, '\n', Output).

trim_trailing_spaces(Line, Trimmed) :-
    re_replace("\\s+$"/a, "", Line, Trimmed).


%% ============================================
%% TEST SUPPORT
%% ============================================

test_csharp_stream_target :-
    retractall(parent(_, _)),
    retractall(grandparent(_, _)),
    retractall(related(_, _)),
    retractall(color(_)),
    assertz(parent(alice, bob)),
    assertz(parent(bob, charlie)),
    assertz((grandparent(X, Z) :- parent(X, Y), parent(Y, Z))),
    compile_predicate_to_csharp(grandparent/2, [unique(true)], GrandCode),
    sub_string(GrandCode, _, _, _, '.Join(ParentStream()'),
    sub_string(GrandCode, _, _, _, '.Distinct()'),
    compile_predicate_to_csharp(parent/2, [unique(true)], ParentCode),
    sub_string(ParentCode, _, _, _, 'ParentFactStream()'),
    sub_string(ParentCode, _, _, _, 'Console.WriteLine("${item.Item1}:{item.Item2}");'),
    sub_string(ParentCode, _, _, _, '.Distinct()'),
    assertz((related(X, Y) :- parent(X, Y))),
    assertz((related(X, Y) :- parent(Y, X))),
    compile_predicate_to_csharp(related/2, [unique(true)], RelatedCode),
    sub_string(RelatedCode, _, _, _, '.Concat('),
    sub_string(RelatedCode, _, _, _, '.Distinct()'),
    assertz(color(red)),
    assertz(color(blue)),
    compile_predicate_to_csharp(color/1, [], ColorCode),
    sub_string(ColorCode, _, _, _, 'IEnumerable<string> ColorStream'),
    sub_string(ColorCode, _, _, _, 'Console.WriteLine(item);'),
    \+ sub_string(ColorCode, _, _, _, '.Distinct()'),
    retractall(color(_)),
    retractall(related(_, _)),
    retractall(parent(_, _)),
    retractall(grandparent(_, _)).
