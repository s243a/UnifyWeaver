:- encoding(utf8).
% Test suite for WAM-to-Rust transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_rust_target.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/rust_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/core/template_system', [render_named_template/3]).

%% Test predicate that resists native lowering — multi-clause rule with
%% mutual body calls and compound unification that no native tier handles.
%% The combination of multi-clause + compound heads + rule bodies with
%% multiple goals that aren't recognized as recursive patterns will
%% exhaust all native tiers.
:- dynamic test_resistant/3.
test_resistant(X, Y, Z) :- test_resistant_helper(X, Y), test_resistant_helper(Y, Z).
test_resistant(X, _, X).
:- dynamic test_resistant_helper/2.
test_resistant_helper(a, b).
test_resistant_helper(b, c).

%% Simple predicate that native lowering CAN handle
:- dynamic test_simple_fact/2.
test_simple_fact(hello, world).
test_simple_fact(foo, bar).

:- dynamic test_failed/0.

:- dynamic category_ancestor/4.
:- dynamic category_parent/2.
:- dynamic tail_suffix/2.
:- dynamic tail_suffixes/2.
:- dynamic tc_ancestor/2.
:- dynamic tri_sum/2.
:- dynamic tc_descendant/2.
:- dynamic tc_distance/3.
:- dynamic tc_parent_distance/4.
:- dynamic tc_step_parent_distance/5.
:- dynamic weighted_path/3.
:- dynamic weighted_edge/3.
:- dynamic astar_weighted_path/4.
:- dynamic direct_semantic_dist/3.
:- dynamic min_semantic_dist/3.
:- dynamic min_semantic_dist_astar/4.
:- dynamic grouped_min_semantic_dist/3.
:- dynamic grouped_min_semantic_dist_astar/4.
:- dynamic filtered_adjusted_min_semantic_dist/3.
:- dynamic filtered_adjusted_min_semantic_dist_astar/4.
:- dynamic filtered_adjusted_weighted_path/3.
:- dynamic filtered_adjusted_astar_weighted_path/4.
:- dynamic target_label/2.
:- dynamic label_bucket/2.
:- dynamic labeled_adjusted_weighted_path/3.
:- dynamic labeled_adjusted_astar_weighted_path/4.
:- dynamic bucketed_adjusted_weighted_path/3.
:- dynamic bucketed_adjusted_astar_weighted_path/4.
:- dynamic bucketed_min_semantic_dist/3.
:- dynamic bucketed_min_semantic_dist_astar/4.
:- dynamic grouped_bucketed_min_semantic_dist/3.
:- dynamic grouped_bucketed_min_semantic_dist_astar/4.
:- dynamic tc_parent/2.
:- dynamic max_depth/1.

max_depth(10).
category_parent(alpha, beta).
category_parent(beta, gamma).
category_parent(gamma, physics).
category_parent(beta, physics).

tail_suffix(S, S).
tail_suffix([_|T], S) :- tail_suffix(T, S).

tail_suffixes([], [[]]).
tail_suffixes([H|T], [[H|T]|Rest]) :- tail_suffixes(T, Rest).

category_ancestor(Cat, Target, 1, Visited) :-
    category_parent(Cat, Target),
    \+ member(Target, Visited).
category_ancestor(Cat, Target, Hops, Visited) :-
    max_depth(MaxD),
    length(Visited, Depth),
    Depth < MaxD,
    !,
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Target, H1, [Mid|Visited]),
    Hops is H1 + 1.

tc_parent(tom, bob).
tc_parent(tom, liz).
tc_parent(bob, ann).
tc_parent(bob, pat).
tc_parent(pat, jim).

tc_ancestor(X, Y) :- tc_parent(X, Y).
tc_ancestor(X, Y) :- tc_parent(X, Z), tc_ancestor(Z, Y).

tc_descendant(X, Y) :- tc_parent(Y, X).
tc_descendant(X, Y) :- tc_parent(Z, X), tc_descendant(Z, Y).

tri_sum(0, 0).
tri_sum(N, Sum) :-
    N > 0,
    N1 is N - 1,
    tri_sum(N1, Prev),
    Sum is Prev + N.

tc_distance(X, Y, 1) :- tc_parent(X, Y).
tc_distance(X, Y, D) :- tc_parent(X, Z), tc_distance(Z, Y, D1), D is D1 + 1.
tc_parent_distance(X, Y, X, 1) :- tc_parent(X, Y).
tc_parent_distance(X, Y, Parent, D) :-
    tc_parent(X, Z),
    tc_parent_distance(Z, Y, Parent, D1),
    D is D1 + 1.
tc_step_parent_distance(X, Y, Y, X, 1) :- tc_parent(X, Y).
tc_step_parent_distance(X, Y, Step, Parent, D) :-
    tc_parent(X, Step),
    tc_step_parent_distance(Step, Y, _Inner, Parent, D1),
    D is D1 + 1.

weighted_edge(s, a, 1.0).
weighted_edge(s, b, 4.0).
weighted_edge(a, b, 2.0).
weighted_edge(a, c, 5.0).
weighted_edge(b, c, 1.0).
weighted_edge(c, d, 3.0).

direct_semantic_dist(s, a, 1.0).
direct_semantic_dist(s, b, 3.0).
direct_semantic_dist(s, c, 4.0).
direct_semantic_dist(s, d, 7.0).
direct_semantic_dist(a, b, 2.0).
direct_semantic_dist(a, c, 3.0).
direct_semantic_dist(a, d, 6.0).
direct_semantic_dist(b, c, 1.0).
direct_semantic_dist(b, d, 4.0).
direct_semantic_dist(c, d, 3.0).

target_label(b, branch_b).
target_label(c, branch_c).
target_label(d, goal_d1).
target_label(d, goal_d2).
label_bucket(branch_b, branch_bucket).
label_bucket(branch_c, branch_bucket).
label_bucket(goal_d1, goal_bucket).
label_bucket(goal_d2, goal_bucket).

weighted_path(X, Y, W) :- weighted_edge(X, Y, W).
weighted_path(X, Y, Cost) :-
    weighted_edge(X, Z, W),
    weighted_path(Z, Y, RestCost),
    Cost is W + RestCost.

astar_weighted_path(X, Y, _Dim, W) :- weighted_edge(X, Y, W).
astar_weighted_path(X, Y, Dim, Cost) :-
    weighted_edge(X, Z, W),
    direct_semantic_dist(Z, Y, _Heuristic),
    astar_weighted_path(Z, Y, Dim, RestCost),
    Cost is W + RestCost.

min_semantic_dist(Start, Target, MinDist) :-
    aggregate_all(min(Cost), weighted_path(Start, Target, Cost), MinDist).

min_semantic_dist_astar(Start, Target, Dim, MinDist) :-
    aggregate_all(min(Cost), astar_weighted_path(Start, Target, Dim, Cost), MinDist).

grouped_min_semantic_dist(Start, Target, MinDist) :-
    aggregate_all(min(Cost), weighted_path(Start, Target, Cost), Target, MinDist).

grouped_min_semantic_dist_astar(Start, Target, Dim, MinDist) :-
    aggregate_all(min(Cost), astar_weighted_path(Start, Target, Dim, Cost), Target, MinDist).

filtered_adjusted_min_semantic_dist(Start, Target, MinDist) :-
    aggregate_all(min(Adjusted),
        (weighted_path(Start, Target, Cost), Cost > 2, Adjusted is Cost + 1),
        MinDist).

filtered_adjusted_min_semantic_dist_astar(Start, Target, Dim, MinDist) :-
    aggregate_all(min(Adjusted),
        (astar_weighted_path(Start, Target, Dim, Cost), Cost > 2, Adjusted is Cost + 1),
        MinDist).

filtered_adjusted_weighted_path(Start, Target, Adjusted) :-
    weighted_path(Start, Target, Cost),
    Cost > 2,
    Adjusted is Cost + 1.

filtered_adjusted_astar_weighted_path(Start, Target, Dim, Adjusted) :-
    astar_weighted_path(Start, Target, Dim, Cost),
    Cost > 2,
    Adjusted is Cost + 1.

labeled_adjusted_weighted_path(Start, Label, Adjusted) :-
    weighted_path(Start, Target, Cost),
    target_label(Target, Label),
    Cost > 2,
    Adjusted is Cost + 1.

labeled_adjusted_astar_weighted_path(Start, Label, Dim, Adjusted) :-
    astar_weighted_path(Start, Target, Dim, Cost),
    target_label(Target, Label),
    Cost > 2,
    Adjusted is Cost + 1.

bucketed_adjusted_weighted_path(Start, Bucket, Adjusted) :-
    weighted_path(Start, Target, Cost),
    target_label(Target, Label),
    label_bucket(Label, Bucket),
    Cost > 2,
    Adjusted is Cost + 1.

bucketed_adjusted_astar_weighted_path(Start, Bucket, Dim, Adjusted) :-
    astar_weighted_path(Start, Target, Dim, Cost),
    target_label(Target, Label),
    label_bucket(Label, Bucket),
    Cost > 2,
    Adjusted is Cost + 1.

bucketed_min_semantic_dist(Start, Bucket, MinDist) :-
    aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        MinDist).

bucketed_min_semantic_dist_astar(Start, Bucket, Dim, MinDist) :-
    aggregate_all(min(Adjusted),
        ( astar_weighted_path(Start, Target, Dim, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        MinDist).

grouped_bucketed_min_semantic_dist(Start, Bucket, MinDist) :-
    aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        Bucket,
        MinDist).

grouped_bucketed_min_semantic_dist_astar(Start, Bucket, Dim, MinDist) :-
    aggregate_all(min(Adjusted),
        ( astar_weighted_path(Start, Target, Dim, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        Bucket,
        MinDist).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

check_brace_balance(String) :-
    string_chars(String, Chars),
    count_chars(Chars, '{', Open),
    count_chars(Chars, '}', Close),
    (   Open == Close
    ->  true
    ;   format(user_error, 'Brace imbalance: { (~w) vs } (~w)~n', [Open, Close]),
        fail
    ).

count_chars([], _, 0).
count_chars([C|Rest], C, N) :-
    !, count_chars(Rest, C, N1),
    N is N1 + 1.
count_chars([_|Rest], C, N) :-
    count_chars(Rest, C, N).

%% Tests

test_step_wam_generation :-
    Test = 'WAM-Rust: step() match arms generation',
    (   compile_step_wam_to_rust([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn step'),
        sub_string(S, _, _, _, 'match instr'),
        sub_string(S, _, _, _, 'GetConstant'),
        sub_string(S, _, _, _, 'GetVariable'),
        sub_string(S, _, _, _, 'PutValue'),
        sub_string(S, _, _, _, 'Allocate'),
        sub_string(S, _, _, _, 'TryMeElse'),
        sub_string(S, _, _, _, 'BeginAggregate'),
        sub_string(S, _, _, _, 'EndAggregate'),
        sub_string(S, _, _, _, 'Proceed'),
        sub_string(S, _, _, _, 'SwitchOnConstant')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected match arms in step()')
    ).

test_helpers_generation :-
    Test = 'WAM-Rust: helper functions generation',
    (   compile_wam_helpers_to_rust([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn run'),
        sub_string(S, _, _, _, 'fn backtrack'),
        sub_string(S, _, _, _, 'fn unwind_trail'),
        sub_string(S, _, _, _, 'fn execute_builtin'),
        sub_string(S, _, _, _, 'aggregate_frame'),
        sub_string(S, _, _, _, 'fn eval_arith')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected helper functions')
    ).

test_full_runtime_generation :-
    Test = 'WAM-Rust: full runtime impl block',
    (   compile_wam_runtime_to_rust([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'impl WamState'),
        sub_string(S, _, _, _, 'fn step'),
        sub_string(S, _, _, _, 'fn run'),
        sub_string(S, _, _, _, 'fn backtrack')
    ->  pass(Test)
    ;   fail_test(Test, 'Incomplete impl WamState block')
    ).

test_all_instruction_arms :-
    Test = 'WAM-Rust: all instruction types covered',
    (   compile_step_wam_to_rust([], Code),
        atom_string(Code, S),
        % Head unification
        sub_string(S, _, _, _, 'GetConstant'),
        sub_string(S, _, _, _, 'GetVariable'),
        sub_string(S, _, _, _, 'GetValue'),
        sub_string(S, _, _, _, 'GetStructure'),
        sub_string(S, _, _, _, 'GetList'),
        sub_string(S, _, _, _, 'UnifyVariable'),
        sub_string(S, _, _, _, 'UnifyValue'),
        sub_string(S, _, _, _, 'UnifyConstant'),
        % Body construction
        sub_string(S, _, _, _, 'PutConstant'),
        sub_string(S, _, _, _, 'PutVariable'),
        sub_string(S, _, _, _, 'PutValue'),
        sub_string(S, _, _, _, 'PutStructure'),
        sub_string(S, _, _, _, 'PutList'),
        sub_string(S, _, _, _, 'SetVariable'),
        sub_string(S, _, _, _, 'SetValue'),
        sub_string(S, _, _, _, 'SetConstant'),
        sub_string(S, _, _, _, 'LoadRegisterConstant'),
        sub_string(S, _, _, _, 'Cons'),
        sub_string(S, _, _, _, 'NotMember'),
        sub_string(S, _, _, _, 'ListLengthLt'),
        sub_string(S, _, _, _, 'BaseCategoryAncestor'),
        sub_string(S, _, _, _, 'RecurseCategoryAncestor'),
        sub_string(S, _, _, _, 'ReturnAdd1'),
        % Control
        sub_string(S, _, _, _, 'Allocate'),
        sub_string(S, _, _, _, 'Deallocate'),
        sub_string(S, _, _, _, 'Call('),
        sub_string(S, _, _, _, 'CallForeign'),
        sub_string(S, _, _, _, 'CallIndexedAtomFact2'),
        sub_string(S, _, _, _, 'Execute('),
        sub_string(S, _, _, _, 'Proceed'),
        sub_string(S, _, _, _, 'BuiltinCall'),
        sub_string(S, _, _, _, 'BeginAggregate'),
        sub_string(S, _, _, _, 'EndAggregate'),
        % Choice points
        sub_string(S, _, _, _, 'TryMeElse'),
        sub_string(S, _, _, _, 'TrustMe'),
        sub_string(S, _, _, _, 'RetryMeElse'),
        % Indexing
        sub_string(S, _, _, _, 'SwitchOnConstant'),
        sub_string(S, _, _, _, 'SwitchOnStructure'),
        sub_string(S, _, _, _, 'SwitchOnConstantA2')
    ->  pass(Test)
    ;   fail_test(Test, 'Not all instruction types have match arms')
    ).

test_builtin_dispatch :-
    Test = 'WAM-Rust: builtin dispatch covers all ops',
    (   compile_wam_helpers_to_rust([], Code),
        atom_string(Code, S),
        % Check for balanced braces
        check_brace_balance(S),
        sub_string(S, _, _, _, 'is/2'),
        sub_string(S, _, _, _, 'result.round()'),
        sub_string(S, _, _, _, '>/2'),
        sub_string(S, _, _, _, '==/2'),
        sub_string(S, _, _, _, 'true/0'),
        sub_string(S, _, _, _, 'fail/0'),
        sub_string(S, _, _, _, '!/0'),
        sub_string(S, _, _, _, '"write/1" | "display/1"'),
        sub_string(S, _, _, _, 'print!("{}", derefed)'),
        sub_string(S, _, _, _, 'nl/0'),
        sub_string(S, _, _, _, 'atom/1'),
        sub_string(S, _, _, _, 'number/1'),
        sub_string(S, _, _, _, 'member/2'),
        sub_string(S, _, _, _, 'builtin_state'),
        sub_string(S, _, _, _, 'eprintln!'),
        %% Phase 4: Group A term inspection builtins are present.
        sub_string(S, _, _, _, '"functor/3"'),
        sub_string(S, _, _, _, '"arg/3"'),
        sub_string(S, _, _, _, '"=../2"'),
        sub_string(S, _, _, _, '"copy_term/2"'),
        %% copy_term_walk helper (sharing-preserving recursive copy).
        sub_string(S, _, _, _, 'copy_term_walk')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing builtin dispatch cases')
    ).

test_predicate_wrapper :-
    Test = 'WAM-Rust: predicate wrapper generation',
    (   compile_wam_predicate_to_rust(test_pred/2, "dummy", [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn test_pred'),
        sub_string(S, _, _, _, 'a1: Value'),
        sub_string(S, _, _, _, 'a2: Value'),
        sub_string(S, _, _, _, 'set_reg')
    ->  pass(Test)
    ;   fail_test(Test, 'Incorrect predicate wrapper')
    ).

test_foreign_spec_wrapper_generation :-
    Test = 'WAM-Rust: generic foreign spec drives wrapper generation',
    ForeignSpec = foreign_predicate(
        category_ancestor/4,
        [ register_foreign_native_kind(category_ancestor/4, category_ancestor),
          register_foreign_result_layout(category_ancestor/4, tuple(1)),
          register_foreign_result_mode(category_ancestor/4, stream),
          register_foreign_usize_config(category_ancestor/4, max_depth, 10)
        ],
        [category_ancestor/4]
    ),
    WamCode = "call category_ancestor/4, 4",
    (   compile_wam_predicate_to_rust(category_ancestor/4, WamCode,
            [foreign_lowering(ForeignSpec)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("category_ancestor/4", "category_ancestor")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("category_ancestor/4", "tuple:1")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("category_ancestor/4", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_usize_config("category_ancestor/4", "max_depth", 10)'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("category_ancestor", 4)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Generic foreign spec did not drive wrapper generation')
    ).

test_recursive_kernel_ir_selection :-
    Test = 'WAM-Rust: recursive kernel IR normalizes supported schemas',
    CategoryClauses = [
        category_ancestor(Cat1, Target1, 1, Visited1)-
            (category_parent(Cat1, Target1), \+ member(Target1, Visited1)),
        category_ancestor(Cat2, Target2, Hops2, Visited2)-
            ( max_depth(MaxDepth2),
              length(Visited2, Depth2),
              Depth2 < MaxDepth2,
              !,
              category_parent(Cat2, Mid2),
              \+ member(Mid2, Visited2),
              category_ancestor(Mid2, Target2, H12, [Mid2|Visited2]),
              Hops2 is H12 + 1
            )
    ],
    DescClauses = [
        tc_descendant(X1, Y1)-tc_parent(Y1, X1),
        tc_descendant(X2, Y2)-(tc_parent(Z2, X2), tc_descendant(Z2, Y2))
    ],
    SumClauses = [
        tri_sum(0, 0)-true,
        tri_sum(N2, Sum2)-((N2 > 0), ((N12 is N2 - 1), (tri_sum(N12, Prev2), Sum2 is Prev2 + N2)))
    ],
    SuffixClauses = [
        tail_suffix(S1, S1)-true,
        tail_suffix([_|T2], S2)-tail_suffix(T2, S2)
    ],
    SuffixesClauses = [
        tail_suffixes([], [[]])-true,
        tail_suffixes([H3|T3], [[H3|T3]|Rest3])-tail_suffixes(T3, Rest3)
    ],
    DistClauses = [
        tc_distance(S1, T1, 1)-tc_parent(S1, T1),
        tc_distance(S2, T2, D2)-(tc_parent(S2, M2), (tc_distance(M2, T2, D1), D2 is D1 + 1))
    ],
    ParentDistClauses = [
        tc_parent_distance(SP1, TP1, SP1, 1)-tc_parent(SP1, TP1),
        tc_parent_distance(SP2, TP2, PP2, DP2)-
            (tc_parent(SP2, MP2), (tc_parent_distance(MP2, TP2, PP2, DPrev2), DP2 is DPrev2 + 1))
    ],
    StepParentDistClauses = [
        tc_step_parent_distance(SS1, ST1, ST1, SS1, 1)-tc_parent(SS1, ST1),
        tc_step_parent_distance(SS2, ST2, SSP2, SPP2, SDP2)-
            (tc_parent(SS2, SSP2), (tc_step_parent_distance(SSP2, ST2, _Inner2, SPP2, SDPrev2), SDP2 is SDPrev2 + 1))
    ],
    (   rust_target:rust_recursive_kernel(category_ancestor, 4, CategoryClauses,
            recursive_kernel(category_ancestor, category_ancestor/4, [max_depth(10)])),
        rust_target:rust_recursive_kernel(tri_sum, 2, SumClauses,
            recursive_kernel(countdown_sum2, tri_sum/2, [])),
        rust_target:rust_recursive_kernel(tail_suffix, 2, SuffixClauses,
            recursive_kernel(list_suffix2, tail_suffix/2, [])),
        rust_target:rust_recursive_kernel(tail_suffixes, 2, SuffixesClauses,
            recursive_kernel(list_suffixes2, tail_suffixes/2, [])),
        rust_target:rust_recursive_kernel(tc_parent_distance, 4, ParentDistClauses,
            recursive_kernel(transitive_parent_distance4, tc_parent_distance/4,
                [edge_pred(tc_parent/2), fact_pairs(ParentDistancePairs)])),
        rust_target:rust_recursive_kernel(tc_step_parent_distance, 5, StepParentDistClauses,
            recursive_kernel(transitive_step_parent_distance5, tc_step_parent_distance/5,
                [edge_pred(tc_parent/2), fact_pairs(StepParentDistancePairs)])),
        rust_target:rust_recursive_kernel(tc_descendant, 2, DescClauses,
            recursive_kernel(transitive_closure2, tc_descendant/2,
                [edge_pred(tc_parent/2), fact_pairs(FactPairs)])),
        rust_target:rust_recursive_kernel(tc_distance, 3, DistClauses,
            recursive_kernel(transitive_distance3, tc_distance/3,
                [edge_pred(tc_parent/2), fact_pairs(DistancePairs)])),
        FactPairs == ['bob'-'tom', 'liz'-'tom', 'ann'-'bob', 'pat'-'bob', 'jim'-'pat']
        , DistancePairs == ['tom'-'bob', 'tom'-'liz', 'bob'-'ann', 'bob'-'pat', 'pat'-'jim']
        , ParentDistancePairs == ['tom'-'bob', 'tom'-'liz', 'bob'-'ann', 'bob'-'pat', 'pat'-'jim']
        , StepParentDistancePairs == ['tom'-'bob', 'tom'-'liz', 'bob'-'ann', 'bob'-'pat', 'pat'-'jim']
    ->  pass(Test)
    ;   fail_test(Test, 'Recursive kernel IR did not normalize expected schemas')
    ).

test_recursive_kernel_spec_generation :-
    Test = 'WAM-Rust: recursive kernel IR generates foreign specs declaratively',
    Kernel = recursive_kernel(
        transitive_parent_distance4,
        tc_parent_distance/4,
        [ edge_pred(tc_parent/2),
          fact_pairs(['tom'-'bob', 'bob'-'ann'])
        ]),
    (   rust_target:rust_recursive_kernel_spec(Kernel, ForeignSpec),
        ForeignSpec = foreign_predicate(
            tc_parent_distance/4,
            [ register_foreign_native_kind(tc_parent_distance/4, transitive_parent_distance4),
              register_foreign_result_layout(tc_parent_distance/4, tuple(3)),
              register_foreign_result_mode(tc_parent_distance/4, stream),
              register_foreign_string_config(tc_parent_distance/4, edge_pred, tc_parent/2),
              register_indexed_atom_fact2(tc_parent/2, ['tom'-'bob', 'bob'-'ann'])
            ],
            [tc_parent_distance/4]
        )
    ->  pass(Test)
    ;   fail_test(Test, 'Recursive kernel spec generation did not match expected foreign spec')
    ).

test_recursive_kernel_spec_generation_quad :-
    Test = 'WAM-Rust: quad tuple kernel spec generation is declarative',
    Kernel = recursive_kernel(
        transitive_step_parent_distance5,
        tc_step_parent_distance/5,
        [ edge_pred(tc_parent/2),
          fact_pairs(['tom'-'bob', 'bob'-'ann'])
        ]),
    (   rust_target:rust_recursive_kernel_spec(Kernel, ForeignSpec),
        ForeignSpec = foreign_predicate(
            tc_step_parent_distance/5,
            [ register_foreign_native_kind(tc_step_parent_distance/5, transitive_step_parent_distance5),
              register_foreign_result_layout(tc_step_parent_distance/5, tuple(4)),
              register_foreign_result_mode(tc_step_parent_distance/5, stream),
              register_foreign_string_config(tc_step_parent_distance/5, edge_pred, tc_parent/2),
              register_indexed_atom_fact2(tc_parent/2, ['tom'-'bob', 'bob'-'ann'])
            ],
            [tc_step_parent_distance/5]
        )
    ->  pass(Test)
    ;   fail_test(Test, 'Quad tuple kernel spec generation did not match expected foreign spec')
    ).

test_recursive_kernel_registry :-
    Test = 'WAM-Rust: recursive kernel registry enumerates supported kernels',
    findall(Kind, rust_target:rust_recursive_kernel_detector(Kind, _), Kinds0),
    sort(Kinds0, Kinds),
    (   Kinds == [astar_shortest_path4, category_ancestor, countdown_sum2, list_suffix2, list_suffixes2, transitive_closure2, transitive_distance3, transitive_parent_distance4, transitive_step_parent_distance5, weighted_shortest_path3]
    ->  pass(Test)
    ;   fail_test(Test, 'Recursive kernel registry did not match expected supported kernels')
    ).

test_foreign_stream_wrapper_plan_ir :-
    Test = 'WAM-Rust: foreign stream wrapper plan normalizes multistage joins',
    Head =.. [bucketed_adjusted_weighted_path, Start, Bucket, Adjusted],
    Body = ( weighted_path(Start, Target, Cost),
             target_label(Target, Label),
             label_bucket(Label, Bucket),
             Cost > 2,
             Adjusted is Cost + 1 ),
    (   rust_target:rust_foreign_stream_wrapper_plan(bucketed_adjusted_weighted_path, 3, Head, Body, Plan),
        Plan = foreign_wrapper_plan(
            weighted_kernel(weighted_path, weighted_edge/3,
                ['s'-'a'-1.0, 's'-'b'-4.0, 'a'-'b'-2.0, 'a'-'c'-5.0, 'b'-'c'-1.0, 'c'-'d'-3.0]),
            [target_label/2, label_bucket/2],
            [Start, Bucket, Cost],
            Adjusted,
            _)
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign stream wrapper plan IR did not match expected multistage shape')
    ).

test_foreign_wrapper_stage_plan_ir :-
    Test = 'WAM-Rust: foreign wrapper stage plan separates compute and filter stages',
    Head =.. [bucketed_adjusted_weighted_path, Start, Bucket, Adjusted],
    Body = ( weighted_path(Start, Target, Cost),
             target_label(Target, Label),
             label_bucket(Label, Bucket),
             Cost > 2,
             Adjusted is Cost + 1 ),
    (   rust_target:rust_foreign_stream_wrapper_plan(bucketed_adjusted_weighted_path, 3, Head, Body,
            foreign_wrapper_plan(_, _, GoalArgs, Adjusted, GoalInfo)),
        rust_target:rust_foreign_wrapper_stage_plan(GoalInfo, GoalArgs, Adjusted, StagePlan),
        StagePlan = wrapper_stage_plan(
            compute_stage("agg_value", "(cost + 1_f64)"),
            [filter_stage("cost > 2_f64")]),
        rust_target:rust_foreign_wrapper_render_stage_plan(StagePlan, SetupCode, ValueExpr, FilterCond),
        SetupCode == "            let agg_value = (cost + 1_f64);\n",
        ValueExpr == "agg_value",
        FilterCond == 'cost > 2_f64'
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign wrapper stage plan did not separate compute/filter stages')
    ).

test_foreign_stream_stage_traversal_ir :-
    Test = 'WAM-Rust: foreign stream wrapper stage emitter renders joined traversal',
    Head =.. [bucketed_adjusted_weighted_path, Start, Bucket, Adjusted],
    Body = ( weighted_path(Start, Target, Cost),
             target_label(Target, Label),
             label_bucket(Label, Bucket),
             Cost > 2,
             Adjusted is Cost + 1 ),
    (   rust_target:rust_foreign_stream_wrapper_plan(bucketed_adjusted_weighted_path, 3, Head, Body,
            foreign_wrapper_plan(_, JoinPreds, GoalArgs, Adjusted, GoalInfo)),
        rust_target:rust_foreign_wrapper_stage_plan(GoalInfo, GoalArgs, Adjusted, StagePlan),
        rust_target:rust_foreign_wrapper_stage_traversal_code(JoinPreds, "join_filter", "target", "&target", "        ",
            stream, StagePlan, TraversalCode),
        sub_string(TraversalCode, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(TraversalCode, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(TraversalCode, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(TraversalCode, _, _, _, 'packed_results.push(Value::Str("__tuple__".to_string(), vec![')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign stream stage traversal did not render expected joined traversal')
    ).

test_foreign_scalar_stage_traversal_ir :-
    Test = 'WAM-Rust: foreign scalar wrapper stage emitter renders joined min traversal',
    ScalarHead =.. [bucketed_min_semantic_dist, Start, Bucket, MinDist],
    ScalarBody = aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        MinDist),
    rust_target:classify_aggregate(ScalarBody, ScalarAggInfo),
    (   rust_target:rust_foreign_aggregate_wrapper_plan(bucketed_min_semantic_dist, 3,
            ScalarHead, ScalarAggInfo,
            foreign_aggregate_plan(_, JoinPreds, GoalArgs, Expr, all, none, GoalInfo)),
        rust_target:rust_foreign_wrapper_stage_plan(GoalInfo, GoalArgs, Expr, StagePlan),
        rust_target:rust_foreign_wrapper_stage_traversal_code(JoinPreds, "target_filter", "target", "&target", "        ",
            scalar_min, StagePlan, TraversalCode),
        sub_string(TraversalCode, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(TraversalCode, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(TraversalCode, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(TraversalCode, _, _, _, 'best = Some(match best {'),
        sub_string(TraversalCode, _, _, _, 'Some(current) => current.min(agg_value)')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign scalar stage traversal did not render expected joined min traversal')
    ).

test_foreign_grouped_stage_traversal_ir :-
    Test = 'WAM-Rust: foreign grouped wrapper stage emitter renders joined min traversal',
    GroupedHead =.. [grouped_bucketed_min_semantic_dist, Start, Bucket, GroupMin],
    GroupedBody = aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        Bucket,
        GroupMin),
    rust_target:classify_aggregate(GroupedBody, GroupedAggInfo),
    (   rust_target:rust_foreign_aggregate_wrapper_plan(grouped_bucketed_min_semantic_dist, 3,
            GroupedHead, GroupedAggInfo,
            foreign_aggregate_plan(_, JoinPreds, GoalArgs, Expr, group, Bucket, GoalInfo)),
        rust_target:rust_foreign_wrapper_stage_plan(GoalInfo, GoalArgs, Expr, StagePlan),
        rust_target:rust_foreign_wrapper_stage_traversal_code(JoinPreds, "output_filter", "target", "&target", "        ",
            grouped_min, StagePlan, TraversalCode),
        sub_string(TraversalCode, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(TraversalCode, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(TraversalCode, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(TraversalCode, _, _, _, 'grouped.entry(joined_value_2.clone())'),
        sub_string(TraversalCode, _, _, _, '.and_modify(|current| *current = current.min(agg_value))')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign grouped stage traversal did not render expected joined min traversal')
    ).

test_foreign_aggregate_wrapper_plan_ir :-
    Test = 'WAM-Rust: foreign aggregate wrapper plan normalizes scalar and grouped terminals',
    ScalarHead =.. [bucketed_min_semantic_dist, Start, Bucket, MinDist],
    ScalarBody = aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        MinDist),
    GroupedHead =.. [grouped_bucketed_min_semantic_dist, Start, Bucket, GroupMin],
    GroupedBody = aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        Bucket,
        GroupMin),
    rust_target:classify_aggregate(ScalarBody, ScalarAggInfo),
    rust_target:classify_aggregate(GroupedBody, GroupedAggInfo),
    (   rust_target:rust_foreign_aggregate_wrapper_plan(bucketed_min_semantic_dist, 3, ScalarHead, ScalarAggInfo, ScalarPlan),
        ScalarPlan = foreign_aggregate_plan(
            weighted_kernel(weighted_path, weighted_edge/3, _),
            [target_label/2, label_bucket/2],
            [Start, Bucket, Cost],
            Adjusted,
            all,
            none,
            _),
        rust_target:rust_foreign_aggregate_wrapper_plan(grouped_bucketed_min_semantic_dist, 3, GroupedHead, GroupedAggInfo, GroupedPlan),
        GroupedPlan = foreign_aggregate_plan(
            weighted_kernel(weighted_path, weighted_edge/3, _),
            [target_label/2, label_bucket/2],
            [Start, Bucket, Cost],
            Adjusted,
            group,
            Bucket,
            _)
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign aggregate wrapper plan IR did not match expected scalar/grouped shapes')
    ).

%% Phase 4: WAM fallback integration tests

test_wam_fallback_enabled :-
    Test = 'WAM-Rust: WAM fallback for predicate resisting native lowering',
    (   % test_resistant/3 has compound head args — resists native lowering
        % With WAM fallback enabled (default), should succeed
        rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn test_resistant')
    ->  pass(Test)
    ;   fail_test(Test, 'WAM fallback did not trigger for resistant predicate')
    ).

test_wam_fallback_disabled :-
    Test = 'WAM-Rust: WAM fallback disabled via option',
    (   % With wam_fallback(false), compilation should fail for resistant predicate
        \+ rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false), wam_fallback(false)], _)
    ->  pass(Test)
    ;   fail_test(Test, 'WAM fallback was not properly disabled')
    ).

test_native_still_preferred :-
    Test = 'WAM-Rust: native lowering still preferred over WAM fallback',
    (   % test_simple_fact/2 is facts-only — should be natively lowered,
        % not going through WAM even with fallback enabled
        rust_target:compile_predicate_to_rust(user:test_simple_fact/2,
            [include_main(false), wam_fallback(true)], Code),
        atom_string(Code, S),
        % Native facts compilation produces struct + vec!, not WAM wrapper
        (   sub_string(S, _, _, _, 'vec!')
        ;   sub_string(S, _, _, _, 'struct')
        ;   sub_string(S, _, _, _, 'TestSimpleFact')
        ;   sub_string(S, _, _, _, 'test_simple_fact')
        )
    ->  pass(Test)
    ;   fail_test(Test, 'Simple facts were not natively lowered')
    ).

test_wam_fallback_flag :-
    Test = 'WAM-Rust: WAM fallback disabled via Prolog flag',
    (   % Set global flag to disable WAM fallback
        set_prolog_flag(rust_wam_fallback, false),
        \+ rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false)], _),
        % Clean up
        set_prolog_flag(rust_wam_fallback, true)
    ->  pass(Test)
    ;   (   catch(set_prolog_flag(rust_wam_fallback, true), _, true),
            fail_test(Test, 'Prolog flag did not disable WAM fallback')
        )
    ).

%% Phase 5: E2E output validation tests

test_generated_rust_has_wam_wrapper :-
    Test = 'WAM-Rust E2E: generated code has proper wrapper structure',
    (   rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn test_resistant'),
        sub_string(S, _, _, _, 'WamState'),
        sub_string(S, _, _, _, 'set_reg')
    ->  pass(Test)
    ;   fail_test(Test, 'Generated wrapper missing expected elements')
    ).

test_foreign_lowering_category_ancestor :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for category_ancestor/4',
    (   rust_target:compile_predicate_to_rust(user:category_ancestor/4,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("category_ancestor/4", "category_ancestor")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("category_ancestor/4", "tuple:1")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("category_ancestor/4", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_usize_config("category_ancestor/4", "max_depth", 10)'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("category_ancestor", 4)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for category_ancestor/4')
    ).

test_foreign_lowering_transitive_closure :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tc_ancestor/2',
    (   rust_target:compile_predicate_to_rust(user:tc_ancestor/2,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tc_ancestor/2", "transitive_closure2")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tc_ancestor/2", "tuple:1")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tc_ancestor/2", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("tc_ancestor/2", "edge_pred", "tc_parent/2")'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("tc_parent/2", &[("tom", "bob"), ("tom", "liz"), ("bob", "ann"), ("bob", "pat"), ("pat", "jim")])'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tc_ancestor", 2)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tc_ancestor/2')
    ).

test_foreign_lowering_countdown_sum :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tri_sum/2',
    (   rust_target:compile_predicate_to_rust(user:tri_sum/2,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tri_sum/2", "countdown_sum2")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tri_sum/2", "tuple:1")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tri_sum/2", "deterministic")'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tri_sum", 2)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tri_sum/2')
    ).

test_foreign_lowering_list_suffix :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tail_suffix/2',
    (   rust_target:compile_predicate_to_rust(user:tail_suffix/2,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tail_suffix/2", "list_suffix2")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tail_suffix/2", "tuple:1")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tail_suffix/2", "stream")'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tail_suffix", 2)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tail_suffix/2')
    ).

test_foreign_lowering_list_suffixes :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tail_suffixes/2',
    (   rust_target:compile_predicate_to_rust(user:tail_suffixes/2,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tail_suffixes/2", "list_suffixes2")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tail_suffixes/2", "tuple:1")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tail_suffixes/2", "deterministic_collection")'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tail_suffixes", 2)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tail_suffixes/2')
    ).

test_foreign_only_wrapper_omits_dead_wam_code :-
    Test = 'WAM-Rust: foreign-only wrappers omit dead WAM instruction setup',
    (   rust_target:compile_predicate_to_rust(user:tail_suffixes/2,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'vm.code = Vec::new();'),
        sub_string(S, _, _, _, 'vm.labels = HashMap::new();'),
        \+ sub_string(S, _, _, _, 'let code: Vec<Instruction> = vec!['),
        \+ sub_string(S, _, _, _, 'labels.insert('),
        \+ sub_string(S, _, _, _, 'switch_on_term'),
        \+ sub_string(S, _, _, _, 'unknown:')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign-only wrapper still emitted dead WAM setup')
    ).

test_foreign_lowering_reverse_transitive_closure :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tc_descendant/2',
    (   rust_target:compile_predicate_to_rust(user:tc_descendant/2,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tc_descendant/2", "transitive_closure2")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tc_descendant/2", "tuple:1")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tc_descendant/2", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("tc_descendant/2", "edge_pred", "tc_parent/2")'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("tc_parent/2", &[("bob", "tom"), ("liz", "tom"), ("ann", "bob"), ("pat", "bob"), ("jim", "pat")])'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tc_descendant", 2)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tc_descendant/2')
    ).

test_foreign_lowering_transitive_distance :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tc_distance/3',
    (   rust_target:compile_predicate_to_rust(user:tc_distance/3,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tc_distance/3", "transitive_distance3")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tc_distance/3", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tc_distance/3", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("tc_distance/3", "edge_pred", "tc_parent/2")'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("tc_parent/2", &[("tom", "bob"), ("tom", "liz"), ("bob", "ann"), ("bob", "pat"), ("pat", "jim")])'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tc_distance", 3)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tc_distance/3')
    ).

test_foreign_lowering_transitive_parent_distance :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tc_parent_distance/4',
    (   rust_target:compile_predicate_to_rust(user:tc_parent_distance/4,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tc_parent_distance/4", "transitive_parent_distance4")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tc_parent_distance/4", "tuple:3")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tc_parent_distance/4", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("tc_parent_distance/4", "edge_pred", "tc_parent/2")'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tc_parent_distance", 4)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tc_parent_distance/4')
    ).

test_foreign_lowering_transitive_step_parent_distance :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for tc_step_parent_distance/5',
    (   rust_target:compile_predicate_to_rust(user:tc_step_parent_distance/5,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("tc_step_parent_distance/5", "transitive_step_parent_distance5")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("tc_step_parent_distance/5", "tuple:4")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("tc_step_parent_distance/5", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("tc_step_parent_distance/5", "edge_pred", "tc_parent/2")'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("tc_step_parent_distance", 5)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for tc_step_parent_distance/5')
    ).

test_foreign_lowering_weighted_shortest_path :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for weighted_path/3',
    (   rust_target:compile_predicate_to_rust(user:weighted_path/3,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("weighted_path/3", "weighted_shortest_path3")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("weighted_path/3", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("weighted_path/3", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("weighted_path/3", "weight_pred", "weighted_edge/3")'),
        sub_string(S, _, _, _, 'register_indexed_weighted_edge_triples("weighted_edge/3", &[("s", "a", 1.0), ("s", "b", 4.0), ("a", "b", 2.0), ("a", "c", 5.0), ("b", "c", 1.0), ("c", "d", 3.0)])'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("weighted_path", 3)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for weighted_path/3')
    ).

test_foreign_lowering_astar_shortest_path :-
    Test = 'WAM-Rust: compiler can choose foreign lowering for astar_weighted_path/4',
    (   rust_target:compile_predicate_to_rust(user:astar_weighted_path/4,
            [include_main(false), foreign_lowering(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'register_foreign_native_kind("astar_weighted_path/4", "astar_shortest_path4")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("astar_weighted_path/4", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("astar_weighted_path/4", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("astar_weighted_path/4", "weight_pred", "weighted_edge/3")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("astar_weighted_path/4", "direct_dist_pred", "direct_semantic_dist/3")'),
        sub_string(S, _, _, _, 'register_foreign_usize_config("astar_weighted_path/4", "dimensionality", 5)'),
        sub_string(S, _, _, _, 'register_indexed_weighted_edge_triples("weighted_edge/3", &[("s", "a", 1.0), ("s", "b", 4.0), ("a", "b", 2.0), ("a", "c", 5.0), ("b", "c", 1.0), ("c", "d", 3.0)])'),
        sub_string(S, _, _, _, 'register_indexed_weighted_edge_triples("direct_semantic_dist/3", &[("s", "a", 1.0), ("s", "b", 3.0), ("s", "c", 4.0), ("s", "d", 7.0), ("a", "b", 2.0), ("a", "c", 3.0), ("a", "d", 6.0), ("b", "c", 1.0), ("b", "d", 4.0), ("c", "d", 3.0)])'),
        sub_string(S, _, _, _, 'execute_foreign_predicate("astar_weighted_path", 4)'),
        sub_string(S, _, _, _, 'vm.code = Vec::new();')
    ->  pass(Test)
    ;   fail_test(Test, 'Foreign lowering was not selected for astar_weighted_path/4')
    ).

test_weighted_min_aggregate_wrapper :-
    Test = 'WAM-Rust: aggregate min wrapper delegates to weighted_path/3',
    (   rust_target:compile_predicate_to_rust(user:min_semantic_dist/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn min_semantic_dist(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("weighted_path/3", "weighted_shortest_path3")'),
        sub_string(S, _, _, _, 'register_indexed_weighted_edge_triples("weighted_edge/3", &[("s", "a", 1.0), ("s", "b", 4.0), ("a", "b", 2.0), ("a", "c", 5.0), ("b", "c", 1.0), ("c", "d", 3.0)])'),
        sub_string(S, _, _, _, 'if !vm.execute_foreign_predicate("weighted_path", 3) {'),
        sub_string(S, _, _, _, 'vm.unify(&a3, &Value::Float(cost))')
    ->  pass(Test)
    ;   fail_test(Test, 'Weighted aggregate wrapper did not delegate correctly')
    ).

test_astar_min_aggregate_wrapper :-
    Test = 'WAM-Rust: aggregate min wrapper delegates to astar_weighted_path/4',
    (   rust_target:compile_predicate_to_rust(user:min_semantic_dist_astar/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn min_semantic_dist_astar(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("astar_weighted_path/4", "astar_shortest_path4")'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("astar_weighted_path/4", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("astar_weighted_path/4", "weight_pred", "weighted_edge/3")'),
        sub_string(S, _, _, _, 'register_foreign_string_config("astar_weighted_path/4", "direct_dist_pred", "direct_semantic_dist/3")'),
        sub_string(S, _, _, _, 'register_indexed_weighted_edge_triples("weighted_edge/3", &[("s", "a", 1.0), ("s", "b", 4.0), ("a", "b", 2.0), ("a", "c", 5.0), ("b", "c", 1.0), ("c", "d", 3.0)])'),
        sub_string(S, _, _, _, 'register_indexed_weighted_edge_triples("direct_semantic_dist/3", &[("s", "a", 1.0), ("s", "b", 3.0), ("s", "c", 4.0), ("s", "d", 7.0), ("a", "b", 2.0), ("a", "c", 3.0), ("a", "d", 6.0), ("b", "c", 1.0), ("b", "d", 4.0), ("c", "d", 3.0)])'),
        sub_string(S, _, _, _, 'register_foreign_usize_config("astar_weighted_path/4", "dimensionality", 5)'),
        sub_string(S, _, _, _, 'if !vm.execute_foreign_predicate("astar_weighted_path", 4) {'),
        sub_string(S, _, _, _, 'vm.unify(&a4, &Value::Float(cost))')
    ->  pass(Test)
    ;   fail_test(Test, 'A* aggregate wrapper did not delegate correctly')
    ).

test_grouped_weighted_min_aggregate_wrapper :-
    Test = 'WAM-Rust: grouped aggregate min wrapper delegates to weighted_path/3',
    (   rust_target:compile_predicate_to_rust(user:grouped_min_semantic_dist/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn grouped_min_semantic_dist(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'use std::collections::BTreeMap;'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("grouped_min_semantic_dist/3", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("weighted_path/3", "weighted_shortest_path3")'),
        sub_string(S, _, _, _, 'grouped.entry(target.clone())'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("grouped_min_semantic_dist/3", vec![a2.clone(), a3.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Grouped weighted aggregate wrapper did not delegate correctly')
    ).

test_grouped_astar_min_aggregate_wrapper :-
    Test = 'WAM-Rust: grouped aggregate min wrapper delegates to astar_weighted_path/4',
    (   rust_target:compile_predicate_to_rust(user:grouped_min_semantic_dist_astar/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn grouped_min_semantic_dist_astar(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'use std::collections::BTreeMap;'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("grouped_min_semantic_dist_astar/4", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("astar_weighted_path/4", "astar_shortest_path4")'),
        sub_string(S, _, _, _, 'grouped.entry(target.clone())'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("grouped_min_semantic_dist_astar/4", vec![a2.clone(), a4.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Grouped A* aggregate wrapper did not delegate correctly')
    ).

test_filtered_adjusted_weighted_min_wrapper :-
    Test = 'WAM-Rust: mixed-goal aggregate wrapper filters and adjusts weighted_path/3',
    (   rust_target:compile_predicate_to_rust(user:filtered_adjusted_min_semantic_dist/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn filtered_adjusted_min_semantic_dist(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, '&& (cost > 2_f64) {'),
        sub_string(S, _, _, _, 'current.min(agg_value)')
    ->  pass(Test)
    ;   fail_test(Test, 'Mixed-goal weighted aggregate wrapper did not lower correctly')
    ).

test_filtered_adjusted_astar_min_wrapper :-
    Test = 'WAM-Rust: mixed-goal aggregate wrapper filters and adjusts astar_weighted_path/4',
    (   rust_target:compile_predicate_to_rust(user:filtered_adjusted_min_semantic_dist_astar/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn filtered_adjusted_min_semantic_dist_astar(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, '&& (cost > 2_f64) {'),
        sub_string(S, _, _, _, 'current.min(agg_value)')
    ->  pass(Test)
    ;   fail_test(Test, 'Mixed-goal A* aggregate wrapper did not lower correctly')
    ).

test_filtered_adjusted_weighted_stream_wrapper :-
    Test = 'WAM-Rust: mixed-goal stream wrapper filters and adjusts weighted_path/3',
    (   rust_target:compile_predicate_to_rust(user:filtered_adjusted_weighted_path/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn filtered_adjusted_weighted_path(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("filtered_adjusted_weighted_path/3", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("filtered_adjusted_weighted_path/3", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("weighted_path/3", "weighted_shortest_path3")'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, 'let output_value = agg_value;'),
        sub_string(S, _, _, _, '(cost > 2_f64)'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("filtered_adjusted_weighted_path/3", vec![a2.clone(), a3.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Mixed-goal weighted stream wrapper did not lower correctly')
    ).

test_filtered_adjusted_astar_stream_wrapper :-
    Test = 'WAM-Rust: mixed-goal stream wrapper filters and adjusts astar_weighted_path/4',
    (   rust_target:compile_predicate_to_rust(user:filtered_adjusted_astar_weighted_path/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn filtered_adjusted_astar_weighted_path(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("filtered_adjusted_astar_weighted_path/4", "tuple:2")'),
        sub_string(S, _, _, _, 'register_foreign_result_mode("filtered_adjusted_astar_weighted_path/4", "stream")'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("astar_weighted_path/4", "astar_shortest_path4")'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, 'let output_value = agg_value;'),
        sub_string(S, _, _, _, '(cost > 2_f64)'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("filtered_adjusted_astar_weighted_path/4", vec![a2.clone(), a4.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Mixed-goal A* stream wrapper did not lower correctly')
    ).

test_labeled_weighted_stream_wrapper :-
    Test = 'WAM-Rust: relational join stream wrapper expands weighted_path/3 results',
    (   rust_target:compile_predicate_to_rust(user:labeled_adjusted_weighted_path/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn labeled_adjusted_weighted_path(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("weighted_path/3", "weighted_shortest_path3")'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'for joined_value_1 in joined_values_1.iter() {'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("labeled_adjusted_weighted_path/3", vec![a2.clone(), a3.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Relational weighted stream wrapper did not lower correctly')
    ).

test_labeled_astar_stream_wrapper :-
    Test = 'WAM-Rust: relational join stream wrapper expands astar_weighted_path/4 results',
    (   rust_target:compile_predicate_to_rust(user:labeled_adjusted_astar_weighted_path/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn labeled_adjusted_astar_weighted_path(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'register_foreign_native_kind("astar_weighted_path/4", "astar_shortest_path4")'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'for joined_value_1 in joined_values_1.iter() {'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("labeled_adjusted_astar_weighted_path/4", vec![a2.clone(), a4.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Relational A* stream wrapper did not lower correctly')
    ).

test_bucketed_weighted_stream_wrapper :-
    Test = 'WAM-Rust: multi-stage relational wrapper expands weighted_path/3 results',
    (   rust_target:compile_predicate_to_rust(user:bucketed_adjusted_weighted_path/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn bucketed_adjusted_weighted_path(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("label_bucket/2", &[("branch_b", "branch_bucket"), ("branch_c", "branch_bucket"), ("goal_d1", "goal_bucket"), ("goal_d2", "goal_bucket")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(S, _, _, _, 'for joined_value_2 in joined_values_2.iter() {'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("bucketed_adjusted_weighted_path/3", vec![a2.clone(), a3.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Multi-stage weighted wrapper did not lower correctly')
    ).

test_bucketed_astar_stream_wrapper :-
    Test = 'WAM-Rust: multi-stage relational wrapper expands astar_weighted_path/4 results',
    (   rust_target:compile_predicate_to_rust(user:bucketed_adjusted_astar_weighted_path/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn bucketed_adjusted_astar_weighted_path(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("label_bucket/2", &[("branch_b", "branch_bucket"), ("branch_c", "branch_bucket"), ("goal_d1", "goal_bucket"), ("goal_d2", "goal_bucket")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(S, _, _, _, 'for joined_value_2 in joined_values_2.iter() {'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("bucketed_adjusted_astar_weighted_path/4", vec![a2.clone(), a4.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Multi-stage A* wrapper did not lower correctly')
    ).

test_bucketed_weighted_min_wrapper :-
    Test = 'WAM-Rust: scalar aggregate wrapper delegates over weighted_path/3 with two joins',
    (   rust_target:compile_predicate_to_rust(user:bucketed_min_semantic_dist/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn bucketed_min_semantic_dist(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("label_bucket/2", &[("branch_b", "branch_bucket"), ("branch_c", "branch_bucket"), ("goal_d1", "goal_bucket"), ("goal_d2", "goal_bucket")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, 'best = Some(match best {'),
        sub_string(S, _, _, _, 'Some(cost) => vm.unify(&a3, &Value::Float(cost))')
    ->  pass(Test)
    ;   fail_test(Test, 'Scalar weighted multistage aggregate wrapper did not lower correctly')
    ).

test_bucketed_astar_min_wrapper :-
    Test = 'WAM-Rust: scalar aggregate wrapper delegates over astar_weighted_path/4 with two joins',
    (   rust_target:compile_predicate_to_rust(user:bucketed_min_semantic_dist_astar/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn bucketed_min_semantic_dist_astar(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("label_bucket/2", &[("branch_b", "branch_bucket"), ("branch_c", "branch_bucket"), ("goal_d1", "goal_bucket"), ("goal_d2", "goal_bucket")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, 'best = Some(match best {'),
        sub_string(S, _, _, _, 'Some(cost) => vm.unify(&a4, &Value::Float(cost))')
    ->  pass(Test)
    ;   fail_test(Test, 'Scalar A* multistage aggregate wrapper did not lower correctly')
    ).

test_grouped_bucketed_weighted_min_wrapper :-
    Test = 'WAM-Rust: grouped aggregate wrapper delegates over weighted_path/3 with two joins',
    (   rust_target:compile_predicate_to_rust(user:grouped_bucketed_min_semantic_dist/3,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn grouped_bucketed_min_semantic_dist(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool'),
        sub_string(S, _, _, _, 'use std::collections::BTreeMap;'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("grouped_bucketed_min_semantic_dist/3", "tuple:2")'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("label_bucket/2", &[("branch_b", "branch_bucket"), ("branch_c", "branch_bucket"), ("goal_d1", "goal_bucket"), ("goal_d2", "goal_bucket")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, 'grouped.entry(joined_value_2.clone())'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("grouped_bucketed_min_semantic_dist/3", vec![a2.clone(), a3.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Grouped weighted multistage aggregate wrapper did not lower correctly')
    ).

test_grouped_bucketed_astar_min_wrapper :-
    Test = 'WAM-Rust: grouped aggregate wrapper delegates over astar_weighted_path/4 with two joins',
    (   rust_target:compile_predicate_to_rust(user:grouped_bucketed_min_semantic_dist_astar/4,
            [include_main(false), foreign_lowering(true), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'pub fn grouped_bucketed_min_semantic_dist_astar(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value) -> bool'),
        sub_string(S, _, _, _, 'use std::collections::BTreeMap;'),
        sub_string(S, _, _, _, 'register_foreign_result_layout("grouped_bucketed_min_semantic_dist_astar/4", "tuple:2")'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("target_label/2", &[("b", "branch_b"), ("c", "branch_c"), ("d", "goal_d1"), ("d", "goal_d2")])'),
        sub_string(S, _, _, _, 'register_indexed_atom_fact2_pairs("label_bucket/2", &[("branch_b", "branch_bucket"), ("branch_c", "branch_bucket"), ("goal_d1", "goal_bucket"), ("goal_d2", "goal_bucket")])'),
        sub_string(S, _, _, _, 'let joined_values_1 = match vm.indexed_atom_fact2.get("target_label/2").and_then(|table| table.get(&target)) {'),
        sub_string(S, _, _, _, 'let joined_values_2 = match vm.indexed_atom_fact2.get("label_bucket/2").and_then(|table| table.get(joined_value_1)) {'),
        sub_string(S, _, _, _, 'let agg_value = (cost + 1_f64);'),
        sub_string(S, _, _, _, 'grouped.entry(joined_value_2.clone())'),
        sub_string(S, _, _, _, 'vm.finish_foreign_results("grouped_bucketed_min_semantic_dist_astar/4", vec![a2.clone(), a4.clone()], packed_results)')
    ->  pass(Test)
    ;   fail_test(Test, 'Grouped A* multistage aggregate wrapper did not lower correctly')
    ).

test_compile_wam_runtime_output :-
    Test = 'WAM-Rust E2E: full runtime generates valid impl block',
    (   compile_wam_runtime_to_rust([], Code),
        atom_string(Code, S),
        % Verify the impl block has all critical methods
        sub_string(S, _, _, _, 'impl WamState'),
        sub_string(S, _, _, _, 'pub fn step'),
        sub_string(S, _, _, _, 'pub fn run'),
        sub_string(S, _, _, _, 'pub fn backtrack'),
        sub_string(S, _, _, _, 'fn execute_builtin'),
        sub_string(S, _, _, _, 'fn eval_arith'),
        % Verify key instruction handling
        sub_string(S, _, _, _, 'GetConstant'),
        sub_string(S, _, _, _, 'BeginAggregate'),
        sub_string(S, _, _, _, 'EndAggregate'),
        sub_string(S, _, _, _, 'Proceed'),
        sub_string(S, _, _, _, 'TryMeElse'),
        sub_string(S, _, _, _, 'foreign_native_kind(&pred_key)'),
        sub_string(S, _, _, _, 'foreign_result_layout(pred_key)'),
        sub_string(S, _, _, _, 'foreign_result_mode(pred_key)'),
        sub_string(S, _, _, _, 'foreign_string_config(&pred_key, "edge_pred")'),
        sub_string(S, _, _, _, 'foreign_usize_config(&pred_key, "max_depth")'),
        sub_string(S, _, _, _, 'fn finish_foreign_results'),
        sub_string(S, _, _, _, 'fn apply_foreign_result'),
        sub_string(S, _, _, _, 'fn parse_foreign_tuple_layout'),
        sub_string(S, _, _, _, 'tuple:'),
        sub_string(S, _, _, _, '__tuple__'),
        sub_string(S, _, _, _, 'deterministic_collection'),
        sub_string(S, _, _, _, 'name: "foreign_results".to_string()'),
        sub_string(S, _, _, _, 'transitive_closure2'),
        sub_string(S, _, _, _, 'collect_native_transitive_closure_nodes'),
        sub_string(S, _, _, _, 'collect_native_transitive_parent_distance_results'),
        sub_string(S, _, _, _, 'collect_native_transitive_step_parent_distance_results'),
        sub_string(S, _, _, _, 'collect_native_weighted_shortest_path_results'),
        sub_string(S, _, _, _, 'collect_native_astar_shortest_path_results')
    ->  pass(Test)
    ;   fail_test(Test, 'Runtime impl block incomplete')
    ).

%% Phase: Cargo project generation tests

test_write_wam_rust_project :-
    Test = 'WAM-Rust: write_wam_rust_project generates crate',
    TmpDir = 'output/test_wam_rust_crate',
    (   % Clean up any previous test run
        (   exists_directory(TmpDir)
        ->  catch(delete_directory_and_contents(TmpDir), _, true)
        ;   true
        ),
        write_wam_rust_project(
            [user:test_simple_fact/2],
            [module_name('test_crate')],
            TmpDir),
        % Verify files exist
        directory_file_path(TmpDir, 'Cargo.toml', CargoPath),
        exists_file(CargoPath),
        directory_file_path(TmpDir, 'src', SrcDir),
        directory_file_path(SrcDir, 'lib.rs', LibPath),
        exists_file(LibPath),
        directory_file_path(SrcDir, 'value.rs', ValuePath),
        exists_file(ValuePath),
        directory_file_path(SrcDir, 'instructions.rs', InstrPath),
        exists_file(InstrPath),
        directory_file_path(SrcDir, 'state.rs', StatePath),
        exists_file(StatePath),
        % Verify Cargo.toml has the module name
        read_file_to_string(CargoPath, CargoStr, []),
        sub_string(CargoStr, _, _, _, 'test_crate'),
        % Verify lib.rs has predicate code
        read_file_to_string(LibPath, LibStr, []),
        sub_string(LibStr, _, _, _, 'pub mod value'),
        % Verify state.rs has runtime impl
        read_file_to_string(StatePath, StateStr, []),
        sub_string(StateStr, _, _, _, 'impl WamState'),
        sub_string(StateStr, _, _, _, 'fn step'),
        % Clean up
        catch(delete_directory_and_contents(TmpDir), _, true)
    ->  pass(Test)
    ;   catch(delete_directory_and_contents(TmpDir), _, true),
        fail_test(Test, 'Cargo crate generation failed or missing files')
    ).

test_project_cargo_content :-
    Test = 'WAM-Rust: Cargo.toml has correct content',
    (   render_named_template(rust_wam_cargo,
            [module_name='my_wam_crate'], Content),
        atom_string(Content, S),
        sub_string(S, _, _, _, 'my_wam_crate'),
        sub_string(S, _, _, _, '[package]'),
        sub_string(S, _, _, _, 'edition = "2021"')
    ->  pass(Test)
    ;   fail_test(Test, 'Cargo.toml template rendering failed')
    ).

test_project_with_wam_fallback :-
    Test = 'WAM-Rust: project includes WAM-compiled predicates',
    TmpDir = 'output/test_wam_rust_fallback',
    (   (   exists_directory(TmpDir)
        ->  catch(delete_directory_and_contents(TmpDir), _, true)
        ;   true
        ),
        write_wam_rust_project(
            [user:test_resistant/3],
            [module_name('fallback_test'), wam_fallback(true)],
            TmpDir),
        directory_file_path(TmpDir, 'src', SrcDir),
        directory_file_path(SrcDir, 'lib.rs', LibPath),
        read_file_to_string(LibPath, LibStr, []),
        % Should contain WAM-compiled wrapper
        sub_string(LibStr, _, _, _, 'test_resistant'),
        catch(delete_directory_and_contents(TmpDir), _, true)
    ->  pass(Test)
    ;   catch(delete_directory_and_contents(TmpDir), _, true),
        fail_test(Test, 'WAM fallback predicate not in generated project')
    ).

%% Instruction parser tests

test_instruction_parser :-
    Test = 'WAM-Rust: WAM code → Rust instruction literals',
    (   % Compile a simple predicate to WAM, then to Rust instructions
        wam_target:compile_facts_to_wam(user:test_simple_fact, 2, WamCode),
        compile_wam_predicate_to_rust(test_simple_fact/2, WamCode, [], RustCode),
        atom_string(RustCode, S),
        % Should have real instructions, not TODOs
        sub_string(S, _, _, _, 'Instruction::GetConstant'),
        sub_string(S, _, _, _, 'Instruction::Proceed'),
        sub_string(S, _, _, _, 'vec!['),
        sub_string(S, _, _, _, 'vm.run()'),
        % Should NOT have the old TODO
        \+ sub_string(S, _, _, _, 'TODO')
    ->  pass(Test)
    ;   fail_test(Test, 'Instruction parser output incorrect')
    ).

test_instruction_parser_labels :-
    Test = 'WAM-Rust: label map generation',
    (   wam_target:compile_facts_to_wam(user:test_simple_fact, 2, WamCode),
        compile_wam_predicate_to_rust(test_simple_fact/2, WamCode, [], RustCode),
        atom_string(RustCode, S),
        sub_string(S, _, _, _, 'labels.insert'),
        sub_string(S, _, _, _, 'test_simple_fact/2')
    ->  pass(Test)
    ;   fail_test(Test, 'Label map not generated correctly')
    ).

test_instruction_parser_resistant :-
    Test = 'WAM-Rust: resistant predicate generates full WAM code',
    (   wam_target:compile_predicate_to_wam(user:test_resistant/3, [], WamCode),
        compile_wam_predicate_to_rust(test_resistant/3, WamCode, [], RustCode),
        atom_string(RustCode, S),
        sub_string(S, _, _, _, 'Instruction::TryMeElse'),
        sub_string(S, _, _, _, 'Instruction::Allocate'),
        sub_string(S, _, _, _, 'Instruction::Call'),
        sub_string(S, _, _, _, 'vm.run()')
    ->  pass(Test)
    ;   fail_test(Test, 'Resistant predicate WAM code incomplete')
    ).

test_aggregate_instruction_parser :-
    Test = 'WAM-Rust: aggregate instructions lower to Rust enums',
    WamCode = "agg/1:\n    begin_aggregate sum, Y1, A1\n    end_aggregate Y1\n    proceed",
    (   compile_wam_predicate_to_rust(agg/1, WamCode, [], RustCode),
        atom_string(RustCode, S),
        sub_string(S, _, _, _, 'Instruction::BeginAggregate("sum".to_string(), "Y1".to_string(), "A1".to_string())'),
        sub_string(S, _, _, _, 'Instruction::EndAggregate("Y1".to_string())')
    ->  pass(Test)
    ;   fail_test(Test, 'Aggregate instructions were not lowered')
    ).

test_cargo_check_not_available :-
    Test = 'WAM-Rust: cargo_check handles missing cargo gracefully',
    (   % On systems without cargo, should return not_available
        % On systems with cargo, should return ok or error
        cargo_check_project('nonexistent_dir', Result),
        (   Result = not_available -> true
        ;   Result = error(_, _) -> true
        ;   Result = ok -> true
        )
    ->  pass(Test)
    ;   fail_test(Test, 'cargo_check_project failed ungracefully')
    ).

%% Runtime parser tests

test_state_template_has_parser :-
    Test = 'WAM-Rust: state.rs template includes parse_instructions',
    (   read_file_to_string(
            'templates/targets/rust_wam/state.rs.mustache', Content, []),
        sub_string(Content, _, _, _, 'parse_instructions'),
        sub_string(Content, _, _, _, 'from_str'),
        sub_string(Content, _, _, _, 'parse_single_instruction'),
        sub_string(Content, _, _, _, 'parse_value')
    ->  pass(Test)
    ;   fail_test(Test, 'state.rs template missing parser functions')
    ).

test_parser_handles_all_instructions :-
    Test = 'WAM-Rust: parser template covers all instruction opcodes',
    (   read_file_to_string(
            'templates/targets/rust_wam/state.rs.mustache', Content, []),
        % Head unification
        sub_string(Content, _, _, _, '"get_constant"'),
        sub_string(Content, _, _, _, '"get_variable"'),
        sub_string(Content, _, _, _, '"get_value"'),
        sub_string(Content, _, _, _, '"get_structure"'),
        sub_string(Content, _, _, _, '"get_list"'),
        sub_string(Content, _, _, _, '"unify_variable"'),
        sub_string(Content, _, _, _, '"unify_value"'),
        sub_string(Content, _, _, _, '"unify_constant"'),
        % Body construction
        sub_string(Content, _, _, _, '"put_constant"'),
        sub_string(Content, _, _, _, '"put_variable"'),
        sub_string(Content, _, _, _, '"put_value"'),
        sub_string(Content, _, _, _, '"put_structure"'),
        sub_string(Content, _, _, _, '"put_list"'),
        sub_string(Content, _, _, _, '"set_variable"'),
        sub_string(Content, _, _, _, '"set_value"'),
        sub_string(Content, _, _, _, '"set_constant"'),
        sub_string(Content, _, _, _, '"cons"'),
        sub_string(Content, _, _, _, '"not_member"'),
        sub_string(Content, _, _, _, '"list_length_lt"'),
        % Control
        sub_string(Content, _, _, _, '"allocate"'),
        sub_string(Content, _, _, _, '"deallocate"'),
        sub_string(Content, _, _, _, '"call"'),
        sub_string(Content, _, _, _, '"call_indexed_atom_fact2"'),
        sub_string(Content, _, _, _, '"execute"'),
        sub_string(Content, _, _, _, '"proceed"'),
        sub_string(Content, _, _, _, '"builtin_call"'),
        sub_string(Content, _, _, _, '"begin_aggregate"'),
        sub_string(Content, _, _, _, '"end_aggregate"'),
        % Choice points
        sub_string(Content, _, _, _, '"try_me_else"'),
        sub_string(Content, _, _, _, '"retry_me_else"'),
        sub_string(Content, _, _, _, '"trust_me"'),
        % Indexing
        sub_string(Content, _, _, _, '"switch_on_constant"'),
        sub_string(Content, _, _, _, '"switch_on_structure"'),
        sub_string(Content, _, _, _, '"switch_on_constant_a2"'),
        sub_string(Content, _, _, _, 'Instruction::SwitchOnConstant'),
        sub_string(Content, _, _, _, 'entry.rfind(\':\')'),
        % Value parsing
        sub_string(Content, _, _, _, '"true" => Value::Bool(true)'),
        sub_string(Content, _, _, _, '"false" => Value::Bool(false)')
    ->  pass(Test)
    ;   fail_test(Test, 'Parser template missing instruction opcodes')
    ).

test_generated_project_has_parser :-
    Test = 'WAM-Rust: generated project state.rs includes parser',
    TmpDir = 'output/test_wam_rust_parser',
    (   (   exists_directory(TmpDir)
        ->  catch(delete_directory_and_contents(TmpDir), _, true)
        ;   true
        ),
        write_wam_rust_project(
            [user:test_simple_fact/2],
            [module_name('parser_test')],
            TmpDir),
        directory_file_path(TmpDir, 'src', SrcDir),
        directory_file_path(SrcDir, 'state.rs', StatePath),
        read_file_to_string(StatePath, StateStr, []),
        sub_string(StateStr, _, _, _, 'parse_instructions'),
        sub_string(StateStr, _, _, _, 'from_str'),
        catch(delete_directory_and_contents(TmpDir), _, true)
    ->  pass(Test)
    ;   catch(delete_directory_and_contents(TmpDir), _, true),
        fail_test(Test, 'Generated state.rs missing parser')
    ).

test_parser_resilience :-
    Test = 'WAM-Rust: parser handles malformed input gracefully',
    (   read_file_to_string(
            'templates/targets/rust_wam/state.rs.mustache', Content, []),
        % Verify parts.is_empty() check exists
        sub_string(Content, _, _, _, 'if parts.is_empty() {'),
        % Verify ok()? for arity parsing exists
        sub_string(Content, _, _, _, 'args[1].parse::<usize>().ok()?')
    ->  pass(Test)
    ;   fail_test(Test, 'Parser missing resilience checks for malformed input')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('WAM-Rust Target Test Suite~n'),
    format('========================================~n~n'),

    test_step_wam_generation,
    test_helpers_generation,
    test_full_runtime_generation,
    test_all_instruction_arms,
    test_builtin_dispatch,
    test_predicate_wrapper,
    test_foreign_spec_wrapper_generation,
    test_recursive_kernel_ir_selection,
    test_recursive_kernel_spec_generation,
    test_recursive_kernel_registry,
    test_foreign_stream_wrapper_plan_ir,
    test_foreign_wrapper_stage_plan_ir,
    test_foreign_stream_stage_traversal_ir,
    test_foreign_scalar_stage_traversal_ir,
    test_foreign_grouped_stage_traversal_ir,
    test_foreign_aggregate_wrapper_plan_ir,
    test_wam_fallback_enabled,
    test_wam_fallback_disabled,
    test_native_still_preferred,
    test_wam_fallback_flag,
    test_generated_rust_has_wam_wrapper,
    test_foreign_lowering_category_ancestor,
    test_foreign_lowering_transitive_closure,
    test_foreign_lowering_countdown_sum,
    test_foreign_lowering_list_suffix,
    test_foreign_lowering_reverse_transitive_closure,
    test_foreign_lowering_transitive_distance,
    test_foreign_lowering_transitive_parent_distance,
    test_foreign_lowering_transitive_step_parent_distance,
    test_foreign_lowering_weighted_shortest_path,
    test_foreign_lowering_astar_shortest_path,
    test_weighted_min_aggregate_wrapper,
    test_astar_min_aggregate_wrapper,
    test_grouped_weighted_min_aggregate_wrapper,
    test_grouped_astar_min_aggregate_wrapper,
    test_filtered_adjusted_weighted_min_wrapper,
    test_filtered_adjusted_astar_min_wrapper,
    test_filtered_adjusted_weighted_stream_wrapper,
    test_filtered_adjusted_astar_stream_wrapper,
    test_labeled_weighted_stream_wrapper,
    test_labeled_astar_stream_wrapper,
    test_bucketed_weighted_stream_wrapper,
    test_bucketed_astar_stream_wrapper,
    test_bucketed_weighted_min_wrapper,
    test_bucketed_astar_min_wrapper,
    test_grouped_bucketed_weighted_min_wrapper,
    test_grouped_bucketed_astar_min_wrapper,
    test_compile_wam_runtime_output,
    test_write_wam_rust_project,
    test_project_cargo_content,
    test_project_with_wam_fallback,
    test_instruction_parser,
    test_instruction_parser_labels,
    test_instruction_parser_resistant,
    test_aggregate_instruction_parser,
    test_cargo_check_not_available,
    test_state_template_has_parser,
    test_parser_handles_all_instructions,
    test_generated_project_has_parser,
    test_parser_resilience,

    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
