:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% csharp_native_target.pl - C# native target for UnifyWeaver
% Generates standalone C# code using LINQ and procedural recursion.

:- module(csharp_native_target, [
    compile_predicate_to_csharp/3,   % +PredIndicator, +Options, -CSharpCode
    write_csharp_streamer/2,         % +CSharpCode, +FilePath
    test_csharp_native_target/0,
    test_linq_recursive_output/0     % Compare inline vs LINQ recursive styles
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(date)).
:- use_module(library(option)).
:- use_module('../core/dynamic_source_compiler', [is_dynamic_source/1]).

:- dynamic reported_dynamic_source/3.
:- dynamic needed_data/1.

%% ============================================ 
%% PUBLIC API
%% ============================================ 

%% compile_predicate_to_csharp(+Pred/Arity, +Options, -Code)
compile_predicate_to_csharp(PredIndicator, Options, Code) :-
    retractall(reported_dynamic_source(_, _, _)),
    retractall(needed_data(_)),
    (   PredIndicator = Module:Pred/Arity
    ->  true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    ensure_not_dynamic_source('C# native target', Pred/Arity),
    functor(Head, Pred, Arity),
    findall(HeadCopy-BodyCopy,
            ( clause(Module:Head, Body),
              copy_term((Head, Body), (HeadCopy, BodyCopy))
            ),
            Clauses),
    classify_clauses(Pred, Arity, Clauses, Type),
    compile_by_type(Type, Pred, Arity, Clauses, Options, Code).

%% write_csharp_streamer(+Code, +FilePath)
write_csharp_streamer(Code, FilePath) :-
    setup_call_cleanup(
        open(FilePath, write, Stream),
        write(Stream, Code),
        close(Stream)
    ),
    format('[CSharpNative] Generated C# code: ~w~n', [FilePath]).

%% ============================================ 
%% PREDICATE CLASSIFICATION
%% ============================================ 

classify_clauses(_Pred, _Arity, [], none).
classify_clauses(Pred, Arity, Clauses, recursive) :-
    member(_-Body, Clauses),
    contains_recursive_call(Pred, Arity, Body),
    !.
classify_clauses(_Pred, _Arity, Clauses, facts) :-
    Clauses \= [],
    forall(member(_-Body, Clauses), Body == true), !.
classify_clauses(_Pred, _Arity, [_-Body], single_rule) :-
    Body \= true, !.
classify_clauses(_Pred, _Arity, Clauses, multiple_rules) :-
    Clauses \= [],
    length(Clauses, Len),
    Len > 1,
    !.
classify_clauses(_Pred, _Arity, _Clauses, unsupported).

contains_recursive_call(Pred, Arity, Goal) :-
    Goal =.. [',', A, B], !,
    (contains_recursive_call(Pred, Arity, A) ; contains_recursive_call(Pred, Arity, B)).
contains_recursive_call(Pred, Arity, Goal) :-
    Goal =.. [';', A, B], !,
    (contains_recursive_call(Pred, Arity, A) ; contains_recursive_call(Pred, Arity, B)).
contains_recursive_call(Pred, Arity, Goal) :-
    Goal = _Mod:Inner, !,
    contains_recursive_call(Pred, Arity, Inner).
contains_recursive_call(Pred, Arity, Goal) :-
    functor(Goal, Pred, Arity).

%% strip_module/2 - Remove module qualifications from a goal
strip_module(_Mod:Goal, Stripped) :- !, strip_module(Goal, Stripped).
strip_module(Goal, Goal).

body_goals(true, []) :- !.
body_goals(Goal, Goals) :-
    strip_module(Goal, G),
    G =.. [',', A, B], !,
    body_goals(A, GA), body_goals(B, GB), append(GA, GB, Goals).
body_goals(Goal, [Stripped]) :- strip_module(Goal, Stripped).

%% ============================================ 
%% COMPILATION DISPATCH
%% ============================================ 

compile_by_type(none, Pred, Arity, _Clauses, _Options, _) :-
    format('[CSharpNative] ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
    fail.
compile_by_type(facts, Pred, Arity, Clauses, Options, Code) :-
    compile_multi_rule_procedural(Pred, Arity, Clauses, Options, Code).
compile_by_type(single_rule, Pred, Arity, Clauses, Options, Code) :-
    compile_multi_rule_procedural(Pred, Arity, Clauses, Options, Code).
compile_by_type(multiple_rules, Pred, Arity, Clauses, Options, Code) :-
    compile_multi_rule_procedural(Pred, Arity, Clauses, Options, Code).
compile_by_type(recursive, Pred, Arity, Clauses, Options, Code) :-
    compile_multi_rule_procedural(Pred, Arity, Clauses, Options, Code).
compile_by_type(unsupported, Pred, Arity, _Clauses, _Options, _) :-
    format('[CSharpNative] ERROR: Unsupported predicate form for ~w/~w~n', [Pred, Arity]),
    fail.

%% ============================================ 
%% PROCEDURAL COMPILATION
%% ============================================ 

compile_multi_rule_procedural(Pred, Arity, Clauses, Options, Code) :-
    predicate_name_pascal(Pred, ModuleName),
    predicate_name_pascal(Pred, TargetName),
    fact_result_type(Arity, ResultType),
    fact_print_expression(Arity, PrintExpr),

    % Check if this predicate is recursive
    (   member(_-Body, Clauses),
        contains_recursive_call(Pred, Arity, Body)
    ->  IsRecursive = true
    ;   IsRecursive = false
    ),

    % Generate clause methods (only non-recursive clauses for recursive predicates)
    findall(ClauseCode, (
        nth1(Idx, Clauses, ClauseItem),
        ClauseItem = ClHead-ClBody,
        (   IsRecursive == true,
            contains_recursive_call(Pred, Arity, ClBody)
        ->  fail  % Skip recursive clauses - handled specially
        ;   translate_clause_procedural(Pred, Arity, ClHead, ClBody, Idx, ClauseCode, Sigs),
            forall((member(Sg, Sigs), Sg \= Pred/Arity), assert_needed_data(Sg))
        )
    ), ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", AllClausesCode),

    % LINQ recursive style is the default; use inline_recursion(true) for standalone code
    (   option(inline_recursion(true), Options)
    ->  UseLinqRecursive = false
    ;   UseLinqRecursive = true
    ),

    % Generate main stream method
    (   IsRecursive == true, UseLinqRecursive == false
    ->  generate_recursive_stream(Pred, Arity, Clauses, TargetName, ResultType, MemoField, MainStream)
    ;   IsRecursive == true
    ->  generate_recursive_stream_linq(Pred, Arity, Clauses, TargetName, ResultType, MemoField, MainStream)
    ;   % Non-recursive: simple iteration over clauses
        findall(Call, (
            nth1(Index, Clauses, _),
            format(atom(Call), "            foreach (var item in _clause_~w()) yield return item;", [Index])
        ), Calls),
        atomic_list_concat(Calls, "\n", CallsCode),
        format(atom(MemoField), "        private static readonly Dictionary<string, List<~w>> _memo = new();", [ResultType]),
        format(atom(MainStream), '        public static IEnumerable<~w> ~wStream()
        {
            if (_memo.TryGetValue("all", out var cached)) return cached;
            var results = new List<~w>();
            foreach (var item in ~wInternal()) {
                results.Add(item);
            }
            _memo["all"] = results;
            return results;
        }

        private static IEnumerable<~w> ~wInternal()
        {
~s
        }', [ResultType, TargetName, ResultType, TargetName, ResultType, TargetName, CallsCode])
    ),

    get_needed_data(DataSections),
    get_needed_helpers(StreamHelpers),

    compose_csharp_program(ModuleName, DataSections, StreamHelpers,
        [MemoField, AllClausesCode, MainStream],
        TargetName, ResultType, PrintExpr, UseLinqRecursive, Code),
    !.  % Ensure determinism

%% generate_recursive_stream/6 - Generate semi-naive iteration for recursive predicates
generate_recursive_stream(Pred, Arity, Clauses, TargetName, ResultType, MemoField, MainStream) :-
    % Find base case clauses (non-recursive)
    findall(Index, (
        nth1(Index, Clauses, _-Body),
        \+ contains_recursive_call(Pred, Arity, Body)
    ), BaseIndices),

    % Find recursive clauses and extract join info
    findall(rec_info(BasePred, JoinPos), (
        member(_-Body, Clauses),
        contains_recursive_call(Pred, Arity, Body),
        extract_recursive_join_info(Pred, Arity, Body, BasePred, JoinPos)
    ), RecInfos),

    % Generate base case calls
    findall(BaseCall, (
        member(Idx, BaseIndices),
        format(atom(BaseCall), "            foreach (var item in _clause_~w())
            {
                if (seen.Add(item)) { delta.Add(item); yield return item; }
            }", [Idx])
    ), BaseCalls),
    atomic_list_concat(BaseCalls, "\n", BaseCode),

    % Generate recursive iteration (assuming transitive closure pattern for now)
    (   RecInfos = [rec_info(BasePred, JoinPos)|_]
    ->  predicate_name_pascal(BasePred, BasePascal),
        (   JoinPos == first  % base(X,Y), rec(Y,Z) pattern
        ->  format(atom(RecursiveCode), '            while (delta.Count > 0)
            {
                var newDelta = new List<~w>();
                foreach (var d in delta)
                {
                    foreach (var b in ~wStream())
                    {
                        if (b.Item2 == d.Item1)
                        {
                            var newItem = (b.Item1, d.Item2);
                            if (seen.Add(newItem)) { newDelta.Add(newItem); yield return newItem; }
                        }
                    }
                }
                delta = newDelta;
            }', [ResultType, BasePascal])
        ;   % base(Y,X), rec(Y,Z) or other patterns - use Item1 match
            format(atom(RecursiveCode), '            while (delta.Count > 0)
            {
                var newDelta = new List<~w>();
                foreach (var d in delta)
                {
                    foreach (var b in ~wStream())
                    {
                        if (b.Item1 == d.Item1)
                        {
                            var newItem = (b.Item2, d.Item2);
                            if (seen.Add(newItem)) { newDelta.Add(newItem); yield return newItem; }
                        }
                    }
                }
                delta = newDelta;
            }', [ResultType, BasePascal])
        )
    ;   RecursiveCode = "            // Unknown recursive pattern"
    ),

    format(atom(MemoField), "        private static readonly HashSet<~w> _seen = new();", [ResultType]),

    format(atom(MainStream), '        public static IEnumerable<~w> ~wStream()
        {
            if (_seen.Count > 0)
            {
                foreach (var item in _seen) yield return item;
                yield break;
            }

            var seen = _seen;
            var delta = new List<~w>();

            // Base case
~s

            // Semi-naive iteration
~s
        }', [ResultType, TargetName, ResultType, BaseCode, RecursiveCode]).

%% extract_recursive_join_info/5 - Extract info about recursive clause structure
extract_recursive_join_info(Pred, Arity, Body, BasePred, JoinPos) :-
    body_goals(Body, Goals),
    Goals = [First, Second|_],
    functor(First, BasePred, 2),
    BasePred \= Pred,
    functor(Second, Pred, Arity),
    First =.. [_|[Arg1, Arg2]],
    Second =.. [_|[RecArg1|_]],
    (   Arg2 == RecArg1 -> JoinPos = first   % base(X,Y), rec(Y,Z)
    ;   Arg1 == RecArg1 -> JoinPos = second  % base(Y,X), rec(Y,Z)
    ;   JoinPos = unknown
    ).

%% generate_recursive_stream_linq/7 - Generate LINQ-style TransitiveClosure code
%% Uses the LinqRecursive.TransitiveClosure extension method for cleaner code
generate_recursive_stream_linq(Pred, Arity, Clauses, TargetName, ResultType, MemoField, MainStream) :-
    % Find recursive clauses and extract join info
    findall(rec_info(BasePred, JoinPos), (
        member(_-Body, Clauses),
        contains_recursive_call(Pred, Arity, Body),
        extract_recursive_join_info(Pred, Arity, Body, BasePred, JoinPos)
    ), RecInfos),

    % Generate the TransitiveClosure call
    (   RecInfos = [rec_info(BasePred, JoinPos)|_]
    ->  (   JoinPos == first  % base(X,Y), rec(Y,Z) pattern -> join on Item2 == Item1
        ->  format(atom(ExpandExpr),
                '(d, baseRel) => baseRel
                    .Where(b => b.Item2 == d.Item1)
                    .Select(b => (b.Item1, d.Item2))', [])
        ;   % base(Y,X), rec(Y,Z) pattern -> join on Item1 == Item1
            format(atom(ExpandExpr),
                '(d, baseRel) => baseRel
                    .Where(b => b.Item1 == d.Item1)
                    .Select(b => (b.Item2, d.Item2))', [])
        )
    ;   ExpandExpr = '(d, baseRel) => Enumerable.Empty<(string, string)>()  // Unknown pattern'
    ),

    format(atom(MemoField), "        private static List<~w>? _cache;", [ResultType]),

    (   RecInfos = [rec_info(BasePred2, _)|_]
    ->  predicate_name_pascal(BasePred2, BasePascal2),
        format(atom(MainStream), '        public static IEnumerable<~w> ~wStream()
        {
            if (_cache != null) return _cache;
            _cache = ~wStream().TransitiveClosure(
                ~w
            ).ToList();
            return _cache;
        }', [ResultType, TargetName, BasePascal2, ExpandExpr])
    ;   format(atom(MainStream), '        public static IEnumerable<~w> ~wStream()
        {
            // Could not determine recursive pattern
            yield break;
        }', [ResultType, TargetName])
    ).

assert_needed_data(Sig) :- (needed_data(Sig) -> true ; assertz(needed_data(Sig))).
get_needed_data(Sections) :- 
    findall(Section, (needed_data(Sig), gather_fact_entries(Sig, Info), data_section_for_signature(Info, Section)), Sections).
get_needed_helpers(Helpers) :- 
    findall(Helper, (needed_data(Sig), stream_helper_for_signature(Sig, Helper)), Helpers).

translate_clause_procedural(Pred, Arity, Head, Body, Index, Code, Signatures) :-
    body_goals(Body, Goals),
    maplist(goal_signature, Goals, Signatures),
    fact_result_type(Arity, ResultType),
    maplist(build_literal_info_native(Pred/Arity), Goals, LiteralInfos),
    % Apply numbervars to both Head and LiteralInfos together so variable identity is preserved
    numbervars((Head, LiteralInfos), 0, _),
    (   LiteralInfos == []
    ->  TargetName = Pred, predicate_name_pascal(TargetName, Pascal),
        format(atom(PipelineCode), "~wFactStream()", [Pascal])
    ;   generate_linq_pipeline(Head, LiteralInfos, PipelineCode)
    ),
    format(atom(Code), 
'        private static IEnumerable<~w> _clause_~w()
        {
            return ~w;
        }', [ResultType, Index, PipelineCode]).

goal_signature(Goal, Name/Arity) :- functor(Goal, Name, Arity).

build_literal_info_native(TargetSig, Goal, literal_info(Name, Arity, Args, StreamCall)) :-
    functor(Goal, Name, Arity),
    Goal =.. [_|Args],
    predicate_name_pascal(Name, Pascal),
    (   Name/Arity == TargetSig
    ->  format(atom(StreamCall), "~wInternal()", [Pascal])
    ;   format(atom(StreamCall), "~wStream()", [Pascal])
    ).

%% ============================================ 
%% DATA COLLECTION & HELPERS
%% ============================================ 

ensure_not_dynamic_source(Label, PredIndicator) :-
    (is_dynamic_source(PredIndicator) -> 
        format('[~w] ERROR: ~w is dynamic source, not supported in native target.~n', [Label, PredIndicator]), fail
    ; true).

predicate_name_pascal(Pred, Pascal) :-
    atom_string(Pred, PredStr),
    split_string(PredStr, "_", "_", Parts),
    maplist(capitalize_first, Parts, Caps),
    atomic_list_concat(Caps, "", Pascal).

capitalize_first("", "") :- !.
capitalize_first(S, C) :-
    sub_string(S, 0, 1, _, F), sub_string(S, 1, _, 0, R),
    atom_string(F, FAt), upcase_atom(FAt, FUAt), atom_string(FUAt, FUStr),
    string_concat(FUStr, R, C).

data_section_for_signature(fact_info(Name, 1, Entries), Section) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), "~wData", [Pascal]),
    with_output_to(atom(Section),
        ( format('        private static readonly string[] ~w = new[] {~n', [DataName]),
          emit_lits(unary_literal, Entries, "            "),
          format('        };', [])
        )).
data_section_for_signature(fact_info(Name, 2, Entries), Section) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), "~wData", [Pascal]),
    with_output_to(atom(Section),
        ( format('        private static readonly (string, string)[] ~w = new[] {~n', [DataName]),
          emit_lits(tuple_literal, Entries, "            "),
          format('        };', [])
        )).

emit_lits(Pred, Entries, Indent) :-
    ( Entries = [] -> true
    ; maplist(Pred, Entries, Lits), emit_literal_block(Lits, Indent)
    ).

stream_helper_for_signature(Name/1, Helper) :- !,
    predicate_name_pascal(Name, P),
    format(atom(Helper), '        public static IEnumerable<string> ~wFactStream() => ~wData;

        public static IEnumerable<string> ~wStream() => ~wFactStream();', [P, P, P, P]).
stream_helper_for_signature(Name/2, Helper) :- !,
    predicate_name_pascal(Name, P),
    format(atom(Helper), '        public static IEnumerable<(string, string)> ~wFactStream() => ~wData;

        public static IEnumerable<(string, string)> ~wStream() => ~wFactStream();', [P, P, P, P]).
stream_helper_for_signature(Name/Arity, Helper) :-
    Arity > 2,
    predicate_name_pascal(Name, P),
    fact_result_type(Arity, RT),
    format(atom(Helper), '        public static IEnumerable<~w> ~wFactStream() => ~wData;

        public static IEnumerable<~w> ~wStream() => ~wFactStream();', [RT, P, P, RT, P, P]).

unary_literal([A], L) :- escape_cs(A, E), format(atom(L), '"~w"', [E]).
tuple_literal([A,B], L) :- escape_cs(A, EA), escape_cs(B, EB), format(atom(L), '("~w", "~w")', [EA, EB]).

escape_cs(A, E) :- atom_string(A, S), 
    re_replace("\\\\"/g, "\\\\\\\\", S, S1), 
    re_replace("\""/g, "\\\"", S1, S2), 
    re_replace("\n"/g, "\\n", S2, E).

emit_literal_block([], _).
emit_literal_block([F|R], I) :- format('~s~w', [I, F]), forall(member(L, R), format(',~n~s~w', [I, L])), format('~n', []).

gather_fact_entries(Name/Arity, fact_info(Name, Arity, Entries)) :-
    findall(Args, (functor(H, Name, Arity), clause(user:H, true), H =.. [_|Args]), Entries).

fact_result_type(1, "string") :- !.
fact_result_type(2, "(string, string)") :- !.
fact_result_type(Arity, T) :-
    Arity > 2,
    findall("string", between(1, Arity, _), L),
    atomic_list_concat(L, ", ", Inner),
    format(atom(T), "(~w)", [Inner]).

fact_print_expression(1, 'Console.WriteLine(item);').
fact_print_expression(2, 'Console.WriteLine($"{item.Item1}:{item.Item2}");').
fact_print_expression(_, 'Console.WriteLine(item);').

compose_csharp_program(ModuleName, DataSections, StreamHelpers, LogicMembers, TargetName, _ResultType, PrintExpr, UseLinqRecursive, Code) :-
    get_time(T), format_time(atom(Date), '%Y-%m-%d %H:%M:%S', T),
    atomic_list_concat(DataSections, "\n\n", DataBlock),
    atomic_list_concat(StreamHelpers, "\n\n", HelperBlock),
    atomic_list_concat(LogicMembers, "\n\n", LogicBlock),
    % Add extra using when LinqRecursive is enabled
    (   UseLinqRecursive == true
    ->  ExtraUsing = "using UnifyWeaver.Native;\n"
    ;   ExtraUsing = ""
    ),
    format(atom(Code),
'// Generated by UnifyWeaver v0.0.3 - Native Procedural
// Date: ~w
using System;
using System.Collections.Generic;
using System.Linq;
~w
namespace UnifyWeaver.Generated
{
    public static class ~wModule
    {
~s

~s

~s

        public static void Main(string[] args)
        {
            foreach (var item in ~wStream())
            {
                ~w
            }
        }
    }
}', [Date, ExtraUsing, ModuleName, DataBlock, HelperBlock, LogicBlock, TargetName, PrintExpr]).

%% ============================================ 
%% LINQ GENERATOR
%% ============================================ 

head_variables(Head, Vars) :- Head =.. [_|Vars].
vars_unique(L, U) :- list_to_set(L, U).
future_vars([], []).
future_vars([literal_info(_,_,Args,_)|R], V) :- vars_collect(Args, VA), future_vars(R, VR), append(VA, VR, VC), list_to_set(VC, V).
vars_collect(Args, Vars) :- findall(A, (member(A, Args), is_logic_var(A)), Vars).

%% is_logic_var(+Term)
%% True if Term is an unbound variable or a numbered variable ($VAR(N))
is_logic_var(V) :- var(V), !.
is_logic_var('$VAR'(_)).

generate_linq_pipeline(Head, [First|Rest], Pipeline) :-
    head_variables(Head, HeadArgs),
    vars_unique(HeadArgs, HeadVarSet),
    future_vars(Rest, FutureVarSet),
    build_seed_pipeline(HeadVarSet, FutureVarSet, First, SeedExpr, SeedOrder, SeedSeen),
    build_pipeline_rest(Rest, HeadVarSet, SeedSeen, SeedOrder, 1, SeedExpr, BodyExpr, FinalOrder),
    finalize_pipeline(BodyExpr, FinalOrder, HeadArgs, Pipeline).

build_seed_pipeline(HeadVarSet, FutureVarSet, literal_info(_, _Arity, Args, StreamCall), SeedExpr, VarOrder, SeenVars) :-
    append(HeadVarSet, FutureVarSet, Needed),
    findall(V, (member(V, Args), is_logic_var(V), member(V2, Needed), V == V2), VarsToKeep0),
    list_to_set(VarsToKeep0, VarsToKeep),
    maplist(arg_access(Args, "row0"), VarsToKeep, Proj),
    tuple_expr(Proj, ProjExpr),
    format(atom(SeedExpr), "~w.Select(row0 => ~w)", [StreamCall, ProjExpr]),
    VarOrder = VarsToKeep, SeenVars = VarsToKeep.

build_pipeline_rest([], _, _, VarOrder, _, Expr, Expr, VarOrder).
build_pipeline_rest([literal_info(_, _, Args, StreamCall)|Rest], HeadVarSet, SeenVars, VarOrder, Step, ExprIn, ExprOut, FinalOrder) :-
    findall(V, (member(V, Args), is_logic_var(V), member(V2, SeenVars), V == V2), Shared),
    future_vars(Rest, FutureVarSet),
    append(HeadVarSet, FutureVarSet, AllNeeded),
    maplist(arg_access_v(VarOrder, "s"), Shared, LProj), tuple_expr(LProj, LKey),
    maplist(arg_access(Args, "r"), Shared, RProj), tuple_expr(RProj, RKey),
    findall(V, (member(V, Args), is_logic_var(V), member(V2, AllNeeded), V == V2, \+ (member(V3, VarOrder), V3 == V)), NewVars0),
    list_to_set(NewVars0, NewVars),
    append(VarOrder, NewVars, NextOrder),
    maplist(arg_access_v(VarOrder, "s"), VarOrder, SProj),
    maplist(arg_access(Args, "r"), NewVars, RProj2),
    append(SProj, RProj2, AllProj), tuple_expr(AllProj, NextProj),
    format(atom(NextExpr), "~w.Join(~w, s => ~w, r => ~w, (s, r) => ~w)", [ExprIn, StreamCall, LKey, RKey, NextProj]),
    Step1 is Step + 1,
    build_pipeline_rest(Rest, HeadVarSet, SeenVars, NextOrder, Step1, NextExpr, ExprOut, FinalOrder).

finalize_pipeline(Body, Order, HeadArgs, Pipeline) :-
    maplist(arg_access_v(Order, "res"), HeadArgs, Proj),
    tuple_expr(Proj, ProjExpr),
    format(atom(Pipeline), "~w.Select(res => ~w)", [Body, ProjExpr]).

arg_access(Args, Param, Var, Code) :- nth1(I, Args, A), A == Var, !, format(atom(Code), "~w.Item~w", [Param, I]).
arg_access_v(Vars, Param, Var, Code) :- nth1(I, Vars, V), V == Var, !, format(atom(Code), "~w.Item~w", [Param, I]).
tuple_expr([S], S) :- !.
tuple_expr(L, E) :- atomic_list_concat(L, ", ", Inner), format(atom(E), "(~w)", [Inner]).

%% ============================================ 
%% TEST SUPPORT
%% ============================================ 

test_csharp_native_target :-
    format('~n=== C# Native Target Tests ===~n'),
    retractall(user:parent(_,_)),
    retractall(user:ancestor(_,_)),
    assertz(user:parent(a, b)),
    assertz(user:parent(b, c)),
    compile_predicate_to_csharp(parent/2, [], Code1),
    once(sub_string(Code1, _, _, _, "ParentStream")),
    format('  ✓ Non-recursive facts passed~n'),
    assertz((user:ancestor(X, Y) :- user:parent(X, Y))),
    assertz((user:ancestor(X, Y) :- user:parent(X, Z), user:ancestor(Z, Y))),
    % Default is now LINQ style
    compile_predicate_to_csharp(ancestor/2, [], Code2),
    once(sub_string(Code2, _, _, _, "TransitiveClosure")),
    once(sub_string(Code2, _, _, _, "using UnifyWeaver.Native;")),
    format('  ✓ Recursive LINQ style (default) passed~n'),
    % Test inline fallback with inline_recursion(true)
    compile_predicate_to_csharp(ancestor/2, [inline_recursion(true)], Code3),
    once(sub_string(Code3, _, _, _, "while (delta.Count > 0)")),
    format('  ✓ Recursive inline style (fallback) passed~n').

%% test_linq_recursive_output/0 - Print both styles for comparison
test_linq_recursive_output :-
    format('~n=== Comparing Recursive Styles ===~n~n'),
    retractall(user:parent(_,_)),
    retractall(user:ancestor(_,_)),
    assertz(user:parent(alice, bob)),
    assertz(user:parent(bob, charlie)),
    assertz(user:parent(charlie, diana)),
    assertz((user:ancestor(X, Y) :- user:parent(X, Y))),
    assertz((user:ancestor(X, Y) :- user:parent(X, Z), user:ancestor(Z, Y))),
    format('--- LINQ TransitiveClosure Style (Default) ---~n'),
    compile_predicate_to_csharp(ancestor/2, [], Code1),
    format('~w~n~n', [Code1]),
    format('--- Inline Semi-Naive Style (inline_recursion(true)) ---~n'),
    compile_predicate_to_csharp(ancestor/2, [inline_recursion(true)], Code2),
    format('~w~n', [Code2]).
