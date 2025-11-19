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
:- use_module('../core/dynamic_source_compiler', [is_dynamic_source/1]).

:- dynamic reported_dynamic_source/3.

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_csharp(+Pred/Arity, +Options, -Code)
%  Compile a non-recursive predicate to C# streaming code.
%  Currently supports single-clause streaming rules composed from binary relations.
compile_predicate_to_csharp(PredIndicator, Options, Code) :-
    retractall(reported_dynamic_source(_, _, _)),
    PredIndicator = Pred/Arity,
    ensure_not_dynamic_source('C# stream target', PredIndicator),
    functor(Head, Pred, Arity),
    findall(HeadCopy-BodyCopy,
            ( clause(Head, Body),
              copy_term((Head, Body), (HeadCopy, BodyCopy))
            ),
            Clauses),
    classify_clauses(Clauses, Type),
    compile_by_type(Type, Pred, Arity, Clauses, Options, Code).

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

classify_clauses([], none).
classify_clauses(Clauses, facts) :-
    Clauses \= [],
    forall(member(_-Body, Clauses), Body == true), !.
classify_clauses([_-Body], single_rule) :-
    Body \= true, !.
classify_clauses(Clauses, multiple_rules) :-
    Clauses \= [],
    length(Clauses, Len),
    Len > 1,
    \+ forall(member(_-Body, Clauses), Body == true), !.
classify_clauses(_Clauses, unsupported).

compile_by_type(none, Pred, Arity, _Clauses, _Options, _) :-
    format('[CSharpTarget] ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
    fail.
compile_by_type(facts, Pred, Arity, _Clauses, Options, Code) :-
    compile_facts_to_csharp(Pred, Arity, Options, Code).
compile_by_type(multiple_rules, Pred, Arity, Clauses, Options, Code) :-
    compile_multiple_rules_to_csharp(Pred, Arity, Clauses, Options, Code).
compile_by_type(unsupported, Pred, Arity, _Clauses, _Options, _) :-
    format('[CSharpTarget] ERROR: Unsupported predicate form for C# target (~w/~w)~n', [Pred, Arity]),
    format('[CSharpTarget] SUGGESTION: Check for recursion or advanced constraints; if present, use target(csharp_query).~n'),
    fail.
compile_by_type(single_rule, Pred, Arity, [Head-Body], Options, Code) :-
    compile_single_rule_to_csharp(Pred, Arity, Head, Body, Options, Code).

%% ============================================
%% SINGLE RULE COMPILATION
%% ============================================

compile_single_rule_to_csharp(Pred, Arity, Head, Body, Options, Code) :-
    collect_predicate_terms(Body, Terms),
    Terms \= [],
    maplist(term_signature, Terms, Signatures),
    list_to_set(Signatures, UniqueSigs),
    ensure_supported_relations(UniqueSigs, Pred/Arity),
    maplist(gather_fact_entries, UniqueSigs, FactEntries),
    maplist(data_section_for_signature, FactEntries, DataSections),
    maplist(stream_helper_for_signature, UniqueSigs, StreamHelpers),
    predicate_name_pascal(Pred, ModuleName),
    predicate_name_pascal(Pred, TargetName),
    maplist(build_literal_info, Terms, LiteralInfos),
    generate_linq_pipeline(Head, LiteralInfos, PipelineCode),
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
    \+ member(Arity, [1, 2, 3]),
    format('[CSharpTarget] ERROR: Fact-only predicates of arity ~w are not supported for ~w/~w in the C# target.~n',
           [Arity, Pred, Arity]),
    fail.

%% Compile multi-clause predicates (OR combinations)
compile_multiple_rules_to_csharp(Pred, Arity, Clauses, Options, Code) :-
    predicate_name_pascal(Pred, ModuleName),
    predicate_name_pascal(Pred, TargetName),
    fact_result_type(Arity, ResultType),
    fact_print_expression(Arity, PrintExpr),
    process_clause_bodies(Clauses, Pred, Arity, Options, TargetName, 1, false,
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

ensure_not_dynamic_source(TargetLabel, Pred/Arity) :-
    (   is_dynamic_source(Pred/Arity)
    ->  dynamic_source_error(TargetLabel, Pred, Arity)
    ;   true
    ).

dynamic_source_error(TargetLabel, Pred, Arity) :-
    (   reported_dynamic_source(TargetLabel, Pred, Arity)
    ->  fail
    ;   assertz(reported_dynamic_source(TargetLabel, Pred, Arity)),
        format('[~w] ERROR: ~w/~w is defined via source/3. Dynamic sources are not yet supported by the C# stream target.~n',
               [TargetLabel, Pred, Arity]),
        format('[~w] HINT: Compile this predicate via compiler_driver with a bash target or dynamic_source_compiler:compile_dynamic_source/3.~n',
               [TargetLabel]),
        fail
    ).

process_clause_bodies([], _Pred, _Arity, _Options, _TargetName, _Index, _FactIncluded,
        DataAcc, HelperAcc, ClauseAcc, DataAcc, HelperAcc, ClauseAcc).
process_clause_bodies([Head-Body|Rest], Pred, Arity, Options, TargetName, Index, FactIncluded,
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
    ;   build_rule_clause_components(Head, Body, TargetName, Index,
            ClauseData, ClauseHelpers, ClauseCall),
        append(DataAcc, ClauseData, DataAcc1),
        append(HelperAcc, ClauseHelpers, HelperAcc1),
        append(ClauseAcc, [ClauseCall], ClauseAcc1),
        NextIndex is Index + 1,
        process_clause_bodies(Rest, Pred, Arity, Options, TargetName, NextIndex,
            FactIncluded, DataAcc1, HelperAcc1, ClauseAcc1, DataOut, HelperOut, ClauseOut)
    ).

build_rule_clause_components(Head, Body, TargetName, Index,
        DataSections, Helpers, ClauseCall) :-
    collect_predicate_terms(Body, Terms),
    Terms \= [],
    maplist(term_signature, Terms, Signatures),
    list_to_set(Signatures, UniqueSigs),
    functor(Head, Pred, Arity),
    ensure_supported_relations(UniqueSigs, Pred/Arity),
    maplist(gather_fact_entries, UniqueSigs, FactEntries),
    maplist(data_section_for_signature, FactEntries, DataSections0),
    maplist(stream_helper_for_signature, UniqueSigs, StreamHelpers0),
    maplist(build_literal_info, Terms, LiteralInfos),
    generate_linq_pipeline(Head, LiteralInfos, PipelineCode),
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



ensure_supported_relations([], _).
ensure_supported_relations([Name/Arity|Rest], Target) :-
    (   member(Arity, [1, 2, 3])
    ->  ensure_supported_relations(Rest, Target)
    ;   format('[CSharpTarget] ERROR: Unsupported arity (~w) for predicate ~w when compiling ~w~n',
               [Arity, Name, Target]),
        format('[CSharpTarget] SUGGESTION: Try the C# query runtime instead:~n'),
        format('[CSharpTarget]   compile_predicate_to_csharp(~q, [target(csharp_query)], Code).~n', [Target]),
        fail
    ).

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

build_literal_info(Goal, literal_info(Name, Arity, Args, StreamCall)) :-
    Goal =.. [Name|Args],
    length(Args, Arity),
    ensure_literal_args_supported(Name/Arity, Args),
    predicate_name_pascal(Name, Pascal),
    format(atom(StreamCall), '~wStream()', [Pascal]).

ensure_literal_args_supported(Name/Arity, Args) :-
    (   forall(member(Arg, Args), var(Arg))
    ->  true
    ;   format('[CSharpTarget] ERROR: Literal ~w/~w contains non-variable arguments; this shape is not yet supported by the C# stream target.~n',
               [Name, Arity]),
        fail
    ).

%% ============================================
%% CODE GENERATION HELPERS
%% ============================================

predicate_name_pascal(Pred, Pascal) :-
    atom_string(Pred, PredStr),
    split_string(PredStr, '_', '_', Parts),
    maplist(capitalize_first, Parts, Caps),
    atomic_list_concat(Caps, '', Pascal).

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

data_section_for_signature(fact_info(Name, 3, Entries), Section) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), '~wData', [Pascal]),
    with_output_to(atom(Section),
        ( format('        private static readonly (string, string, string)[] ~w = new[] {~n', [DataName]),
          (   Entries = []
          ->  true
          ;   maplist(ternary_literal, Entries, LiteralTuples),
              emit_literal_block(LiteralTuples, "            ")
          ),
          format('        };', [])
        )).

data_section_for_signature(fact_info(Name, Arity, _), _) :-
    \+ member(Arity, [1, 2, 3]),
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

stream_helper_for_signature(Name/3, HelperCode) :-
    predicate_name_pascal(Name, Pascal),
    format(atom(DataName), '~wData', [Pascal]),
    format(atom(HelperCode),
'        public static IEnumerable<(string, string, string)> ~wStream()
        {
            return ~w;
        }', [Pascal, DataName]).

stream_helper_for_signature(Name/Arity, _) :-
    \+ member(Arity, [1, 2, 3]),
    format('[CSharpTarget] ERROR: stream helper for arity ~w of predicate ~w is unimplemented~n', [Arity, Name]),
    fail.


fact_result_type(1, 'string').
fact_result_type(2, '(string, string)').
fact_result_type(3, '(string, string, string)').

fact_print_expression(1, 'Console.WriteLine(item);').
fact_print_expression(2, 'Console.WriteLine($"{item.Item1}:{item.Item2}");').
fact_print_expression(3, 'Console.WriteLine("${item.Item1}:{item.Item2}:{item.Item3}");').

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

ternary_literal([A, B, C], Literal) :-
    escape_csharp_string(A, AEsc),
    escape_csharp_string(B, BEsc),
    escape_csharp_string(C, CEsc),
    format(atom(Literal), '("~w", "~w", "~w")', [AEsc, BEsc, CEsc]).

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

head_variables(Head, Vars) :-
    Head =.. [_|Vars].

generate_linq_pipeline(_Head, [], _) :-
    format('[CSharpTarget] ERROR: Rule body has no literals for C# stream compilation.~n'),
    fail.
generate_linq_pipeline(Head, [First|Rest], Pipeline) :-
    head_variables(Head, HeadArgs),
    vars_unique(HeadArgs, HeadVarSet),
    future_vars(Rest, FutureVarSet),
    build_seed_pipeline(HeadVarSet, FutureVarSet, First, SeedExpr, SeedOrder, SeedSeen),
    build_pipeline_rest(Rest, HeadVarSet, SeedSeen, SeedOrder, 1, SeedExpr, BodyExpr, FinalOrder),
    finalize_pipeline(BodyExpr, FinalOrder, HeadArgs, Pipeline).

build_seed_pipeline(HeadVarSet, FutureVarSet, Literal, SeedExpr, VarOrder, SeenVars) :-
    literal_info_stream_call(Literal, StreamCall),
    literal_args(Literal, Args),
    literal_vars(Literal, LiteralVars),
    vars_union(HeadVarSet, FutureVarSet, NeededVars),
    vars_filter_in_order(LiteralVars, NeededVars, VarsToKeep),
    (   VarsToKeep == []
    ->  literal_signature(Literal, Name/Arity),
        format('[CSharpTarget] ERROR: Literal ~w/~w does not bind any variables needed for the clause head; unable to seed streaming pipeline.~n',
               [Name, Arity]),
        fail
    ;   true
    ),
    seed_param(Param),
    maplist(literal_projection_expr(Args, Param), VarsToKeep, ProjectionElems),
    tuple_or_value_expr(ProjectionElems, ProjectionExpr),
    format(atom(SeedExpr),
'~w
            .Select(~w => ~w)', [StreamCall, Param, ProjectionExpr]),
    VarOrder = VarsToKeep,
    SeenVars = VarsToKeep.

build_pipeline_rest([], _HeadVarSet, _SeenVars, VarOrder, _StepIdx, Expr, Expr, VarOrder).
build_pipeline_rest([Literal|Rest], HeadVarSet, SeenVars, VarOrder, StepIdx,
        ExprIn, ExprOut, FinalOrder) :-
    literal_vars(Literal, LiteralVars),
    vars_intersection(LiteralVars, SeenVars, SharedVars),
    (   SharedVars == []
    ->  literal_signature(Literal, Name/Arity),
        format('[CSharpTarget] ERROR: Literal ~w/~w has no shared variables with earlier literals; cartesian products are not supported in the C# stream target.~n',
               [Name, Arity]),
        fail
    ;   true
    ),
    future_vars(Rest, FutureVarSet),
    vars_union(HeadVarSet, FutureVarSet, NeededVars),
    literal_info_stream_call(Literal, StreamCall),
    state_param(StepIdx, LeftParam),
    right_param(StepIdx, RightParam),
    build_join_keys(SharedVars, VarOrder, LeftParam, Literal, RightParam,
        LeftKeyExpr, RightKeyExpr),
    vars_union(SeenVars, LiteralVars, SeenVars1),
    filter_vars(VarOrder, NeededVars, RetainedVars),
    vars_new_from_literal(LiteralVars, RetainedVars, NeededVars, NewVars),
    append(RetainedVars, NewVars, NewOrder),
    maplist(var_projection_expr(VarOrder, Literal, LeftParam, RightParam),
            NewOrder, ProjectionElems),
    tuple_or_value_expr(ProjectionElems, ProjectionExpr),
    format(atom(NextExpr),
'~w
            .Join(~w,
                  ~w => ~w,
                  ~w => ~w,
                  (~w, ~w) => ~w)', [ExprIn, StreamCall, LeftParam, LeftKeyExpr, RightParam, RightKeyExpr, LeftParam, RightParam, ProjectionExpr]),
    NextStep is StepIdx + 1,
    build_pipeline_rest(Rest, HeadVarSet, SeenVars1, NewOrder, NextStep,
        NextExpr, ExprOut, FinalOrder).

finalize_pipeline(BodyExpr, FinalOrder, HeadArgs, Pipeline) :-
    final_param(FinalParam),
    ensure_head_vars_covered(HeadArgs, FinalOrder),
    maplist(final_projection_expr(FinalOrder, FinalParam), HeadArgs, Elements),
    tuple_or_value_expr(Elements, ProjectionExpr),
    format(atom(Pipeline),
'~w
            .Select(~w => ~w)', [BodyExpr, FinalParam, ProjectionExpr]).

%% Pipeline helper predicates

literal_info_stream_call(literal_info(_, _, _, StreamCall), StreamCall).

literal_args(literal_info(_, _, Args, _), Args).

literal_vars(Literal, Vars) :-
    literal_args(Literal, Args),
    vars_collect_from_args(Args, Vars).

literal_projection_expr(Args, Param, Var, Expr) :-
    literal_var_access_expr(Args, Param, Var, Expr).

literal_signature(literal_info(Name, Arity, _Args, _Stream), Name/Arity).
literal_signature(Goal, Signature) :-
    Goal =.. [Name|Args],
    length(Args, Arity),
    Signature = Name/Arity.

vars_unique(List, Unique) :-
    vars_add_all(List, [], Unique).

vars_union(List1, List2, Union) :-
    vars_add_all(List1, [], Temp),
    vars_add_all(List2, Temp, Union).

vars_add_all([], Acc, Acc).
vars_add_all([Var|Rest], Acc, Union) :-
    (   var_memberchk(Var, Acc)
    ->  Acc1 = Acc
    ;   append(Acc, [Var], Acc1)
    ),
    vars_add_all(Rest, Acc1, Union).

vars_filter_in_order([], _, []).
vars_filter_in_order([Var|Rest], Allowed, [Var|FilteredRest]) :-
    var_memberchk(Var, Allowed), !,
    vars_filter_in_order(Rest, Allowed, FilteredRest).
vars_filter_in_order([_|Rest], Allowed, FilteredRest) :-
    vars_filter_in_order(Rest, Allowed, FilteredRest).

filter_vars(List, Allowed, Filtered) :-
    vars_filter_in_order(List, Allowed, Filtered).

vars_new_from_literal(LiteralVars, Retained, Needed, NewVars) :-
    vars_new_from_literal(LiteralVars, Retained, Needed, [], NewVars).

vars_new_from_literal([], _Retained, _Needed, Acc, Acc).
vars_new_from_literal([Var|Rest], Retained, Needed, Acc, Out) :-
    (   var_memberchk(Var, Needed),
        \+ var_memberchk(Var, Retained),
        \+ var_memberchk(Var, Acc)
    ->  append(Acc, [Var], Acc1)
    ;   Acc1 = Acc
    ),
    vars_new_from_literal(Rest, Retained, Needed, Acc1, Out).

var_projection_expr(PrevOrder, Literal, LeftParam, RightParam, Var, Expr) :-
    (   var_memberchk(Var, PrevOrder)
    ->  state_var_access_expr(PrevOrder, LeftParam, Var, Expr)
    ;   literal_args(Literal, Args),
        literal_var_access_expr(Args, RightParam, Var, Expr)
    ).

state_var_access_expr(VarOrder, Param, Var, Expr) :-
    var_index(VarOrder, Var, Index),
    tuple_item(Index, Item),
    format(atom(Expr), '~w.~w', [Param, Item]).

literal_var_access_expr(Args, Param, Var, Expr) :-
    literal_arg_index(Args, Var, Index),
    tuple_item(Index, Item),
    format(atom(Expr), '~w.~w', [Param, Item]).

build_join_keys(SharedVars, VarOrder, LeftParam, Literal, RightParam,
        LeftKeyExpr, RightKeyExpr) :-
    literal_args(Literal, Args),
    maplist(state_var_access_expr(VarOrder, LeftParam), SharedVars, LeftExprs),
    maplist(literal_var_access_expr(Args, RightParam), SharedVars, RightExprs),
    tuple_or_value_expr(LeftExprs, LeftKeyExpr),
    tuple_or_value_expr(RightExprs, RightKeyExpr).

tuple_or_value_expr([Single], Single) :- !.
tuple_or_value_expr(Elements, Expr) :-
    atomic_list_concat(Elements, ', ', Inner),
    format(atom(Expr), '(~w)', [Inner]).

ensure_head_vars_covered([], _).
ensure_head_vars_covered([Var|Rest], VarOrder) :-
    (   var(Var)
    ->  (   var_memberchk(Var, VarOrder)
        ->  true
        ;   format('[CSharpTarget] ERROR: Head variable ~w is not bound by the clause body.~n', [Var]),
            fail
        )
    ;   format('[CSharpTarget] ERROR: Non-variable head arguments are not supported in the C# stream target.~n'),
        fail
    ),
    ensure_head_vars_covered(Rest, VarOrder).

final_projection_expr(VarOrder, Param, Var, Expr) :-
    state_var_access_expr(VarOrder, Param, Var, Expr).

future_vars([], []).
future_vars([Literal|Rest], Vars) :-
    literal_vars(Literal, LiteralVars),
    future_vars(Rest, FutureVars),
    vars_union(LiteralVars, FutureVars, Vars).

vars_collect_from_args(Args, Vars) :-
    vars_collect_from_args(Args, [], Vars).

vars_collect_from_args([], Acc, Acc).
vars_collect_from_args([Arg|Rest], Acc, Vars) :-
    (   var(Arg),
        \+ var_memberchk(Arg, Acc)
    ->  append(Acc, [Arg], Acc1)
    ;   Acc1 = Acc
    ),
    vars_collect_from_args(Rest, Acc1, Vars).

vars_intersection([], _, []).
vars_intersection([Var|Rest], Others, [Var|Intersection]) :-
    var_memberchk(Var, Others), !,
    vars_intersection(Rest, Others, Intersection).
vars_intersection([_|Rest], Others, Intersection) :-
    vars_intersection(Rest, Others, Intersection).

var_memberchk(Var, [Head|_]) :-
    Var == Head, !.
var_memberchk(Var, [_|Rest]) :-
    var_memberchk(Var, Rest).

var_index([Head|_], Var, 1) :-
    Var == Head, !.
var_index([_|Rest], Var, Index) :-
    var_index(Rest, Var, Index1),
    Index is Index1 + 1.
var_index([], Var, _) :-
    format('[CSharpTarget] ERROR: Unable to locate variable ~w in tuple state.~n', [Var]),
    fail.

literal_arg_index(Args, Var, Index) :-
    literal_arg_index(Args, Var, 1, Index).

literal_arg_index([Arg|_], Var, Index, Index) :-
    Var == Arg, !.
literal_arg_index([_|Rest], Var, Next, Index) :-
    Next1 is Next + 1,
    literal_arg_index(Rest, Var, Next1, Index).
literal_arg_index([], Var, _, _) :-
    format('[CSharpTarget] ERROR: Variable ~w not found in literal arguments.~n', [Var]),
    fail.

tuple_item(Index, Item) :-
    format(atom(Item), 'Item~w', [Index]).

seed_param('row0').

state_param(StepIdx, Param) :-
    format(atom(Param), 'state~w', [StepIdx]).

right_param(StepIdx, Param) :-
    format(atom(Param), 'row~w', [StepIdx]).

final_param('result').

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
    format(atom(MethodComment),
'        /// <summary>
        /// Stream results produced by ~w.
        /// </summary>
        // Usage example:
        // foreach (var result in ~wStream()) {
        //     Console.WriteLine(result);
        // }

', [TargetName, TargetName]),
    format(atom(Program),
'~wusing System;
using System.Collections.Generic;
using System.Linq;

namespace UnifyWeaver.Generated
{
    public static class ~wModule
    {
~w~w~w        public static IEnumerable<~w> ~wStream()
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
}', [HeaderComment, ModuleName, DataBlock, HelperBlock, MethodComment, ResultType, TargetName, Pipeline, TargetName, PrintExpr]),
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
    retractall(person(_, _, _)),
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
    assertz(person(alice, '25', engineer)),
    assertz(person(bob, '30', teacher)),
    compile_predicate_to_csharp(person/3, [], PersonCode),
    sub_string(PersonCode, _, _, _, '(string, string, string)[]'),
    sub_string(PersonCode, _, _, _, 'IEnumerable<(string, string, string)> PersonStream'),
    sub_string(PersonCode, _, _, _, 'Console.WriteLine("${item.Item1}:{item.Item2}:{item.Item3}");'),
    \+ sub_string(PersonCode, _, _, _, '.Distinct()'),
    assertz(rel1(a, b, a1)),
    assertz(rel2(a1, c, b1)),
    assertz(rel3(b1, b)),
    assertz((triple(X, Y, Z) :- rel1(X, Y, A), rel2(A, Z, B), rel3(B, Y))),
    compile_predicate_to_csharp(triple/3, [], TripleCode),
    sub_string(TripleCode, _, _, _, '.Join(Rel2Stream()'),
    sub_string(TripleCode, _, _, _, '.Join(Rel3Stream()'),
    sub_string(TripleCode, _, _, _, '(state2.Item4, state2.Item2)'),
    \+ sub_string(TripleCode, _, _, _, 'result.Item4'),
    sub_string(TripleCode, _, _, _, 'Select(result => (result.Item1, result.Item2, result.Item3))'),
    assertz(edge(a, b)),
    assertz((symmetric_triple(X, Y, X) :- edge(X, Y))),
    compile_predicate_to_csharp(symmetric_triple/3, [], SymmetricCode),
    sub_string(SymmetricCode, _, _, _, 'Select(result => (result.Item1, result.Item2, result.Item1))'),
    retractall(color(_)),
    retractall(related(_, _)),
    retractall(parent(_, _)),
    retractall(grandparent(_, _)),
    retractall(person(_, _, _)),
    retractall(rel1(_, _, _)),
    retractall(rel2(_, _, _)),
    retractall(rel3(_, _)),
    retractall(triple(_, _, _)),
    retractall(edge(_, _)),
    retractall(symmetric_triple(_, _, _)).
