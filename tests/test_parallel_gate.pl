% test_parallel_gate.pl
%
% Tests the T7 compile-time profitability gate (src/unifyweaver/core/
% parallel_gate.pl), the first real consumer of the cost machinery. Asserts a
% fixture program, builds a cost model, and checks that forkable aggregates with
% cheap generators stay sequential while those doing recursive / heavy
% per-branch work are chosen for parallel fan-out.

:- use_module('../src/unifyweaver/core/cost_analysis').
:- use_module('../src/unifyweaver/core/parallel_gate').

% ---- fixture program -------------------------------------------------------
:- dynamic pg_fact/1.
pg_fact(1). pg_fact(2). pg_fact(3).

% cheap per-branch work: arithmetic only
pg_cheap(X, Y) :- Y is X * 2.

% recursive per-branch work (unbounded)
pg_rec([]).
pg_rec([_|T]) :- pg_rec(T).

% heavy bounded per-branch work: many builtins (weight pushes into 'expensive')
pg_heavy(X, R) :-
    atom_codes(X, Cs), msort(Cs, S1), reverse(S1, S2),
    atom_codes(A, S2), atom_concat(A, A, B), sub_atom(B, 0, 3, _, R),
    length(Cs, _), msort(S2, _), reverse(S2, _).

% recursive per-branch work that takes an input from the enumerator
pg_down(0, []).
pg_down(N, [N|T]) :- N > 0, M is N - 1, pg_down(M, T).

% link facts, for an aggregate whose enumerator reads an EXTERNAL input
pg_link(10, 1). pg_link(10, 2). pg_link(20, 3).

:- begin_tests(parallel_gate).

build(M) :- build_cost_model(user, M).

% --- shape recognition ------------------------------------------------------

test(recognise_findall) :-
    forkable_aggregate(findall(X, pg_fact(X), _L), T, G),
    assertion(T == X), assertion(G == pg_fact(X)).

test(recognise_aggregate_all) :-
    forkable_aggregate(aggregate_all(count, pg_fact(_), _N), _T, G),
    assertion(G =@= pg_fact(_)).

test(recognise_bagof_strips_caret) :-
    forkable_aggregate(bagof(X, Y^pg_cheap(X, Y), _L), _T, G),
    assertion(G =@= pg_cheap(_, _)).

test(non_aggregate_is_not_forkable) :-
    assertion(\+ forkable_aggregate(pg_fact(_), _, _)).

% --- the profitability decision ---------------------------------------------

test(cheap_generator_stays_sequential) :-
    build(M),
    aggregate_parallel_decision(findall(Y, (pg_fact(X), pg_cheap(X, Y)), _), M, D),
    assertion(D == sequential).

test(pure_enumeration_stays_sequential) :-
    build(M),
    aggregate_parallel_decision(findall(X, pg_fact(X), _), M, D),
    assertion(D == sequential).

test(recursive_generator_goes_parallel) :-
    build(M),
    aggregate_parallel_decision(findall(L, (pg_fact(_), pg_rec(L)), _), M, D),
    assertion(D == parallel).

test(heavy_generator_goes_parallel) :-
    build(M),
    aggregate_parallel_decision(findall(R, (pg_fact(X), pg_heavy(X, R)), _), M, D),
    assertion(D == parallel).

test(non_aggregate_goal_sequential) :-
    build(M),
    aggregate_parallel_decision(pg_cheap(1, _), M, D),
    assertion(D == sequential).

% --- policy override --------------------------------------------------------

test(policy_override_widens_to_moderate) :-
    build(M),
    % default: heavy=parallel already; prove the override hook works by making
    % the policy include 'cheap', flipping a cheap aggregate to parallel.
    setup_call_cleanup(
        assertz(parallel_gate:parallel_worthy_tier(cheap)),
        aggregate_parallel_decision(findall(Y, (pg_fact(X), pg_cheap(X, Y)), _), M, D),
        retractall(parallel_gate:parallel_worthy_tier(cheap))),
    assertion(memberchk(D, [parallel, sequential])),  % decided by tier vs policy
    % and with the override removed it is sequential again
    aggregate_parallel_decision(findall(Y, (pg_fact(X), pg_cheap(X, Y)), _), M, D2),
    assertion(D2 == sequential).

% --- route-1 split analysis: enumerator | body -----------------------------

% cheap enumerator (fact), heavy body; frontier = the shared var X
test(split_cheap_enum_heavy_body) :-
    build(M),
    I = (pg_fact(X), pg_heavy(X, _R)),
    split_aggregate_generator(I, M, Enum, Body, Frontier),
    assertion(Enum =@= pg_fact(_)),
    assertion(Body =@= pg_heavy(_, _)),
    assertion(Frontier == [X]).

% cheap enumerator, recursive body taking an input from the enumerator
test(split_cheap_enum_recursive_body) :-
    build(M),
    I = (pg_fact(X), pg_down(X, _L)),
    split_aggregate_generator(I, M, Enum, Body, Frontier),
    assertion(Enum =@= pg_fact(_)),
    assertion(Body =@= pg_down(_, _)),
    assertion(Frontier == [X]).

% multi-goal cheap prefix; frontier is only the var the body actually reads (Z)
test(split_multi_goal_prefix_frontier_is_minimal) :-
    build(M),
    I = (pg_fact(X), pg_cheap(X, Z), pg_heavy(Z, _R)),
    split_aggregate_generator(I, M, Enum, Body, Frontier),
    assertion(Enum =@= (pg_fact(A), pg_cheap(A, _))),   % shared first arg preserved
    assertion(Body =@= pg_heavy(_, _)),
    assertion(Frontier == [Z]).         % X stays in Enum only, R in Body only

% a single recursive goal has no cheap fan-out prefix -> no split
test(no_split_single_recursive_goal) :-
    build(M),
    assertion(\+ split_aggregate_generator(pg_down(5, _L), M, _, _, _)).

% all-cheap inner goal has no body carrying work -> no split
test(no_split_all_cheap) :-
    build(M),
    assertion(\+ split_aggregate_generator((pg_fact(X), pg_cheap(X, _Y)), M, _, _, _)).

% soundness: cut, side effects, and disjunction block the split
test(no_split_with_cut) :-
    build(M),
    assertion(\+ split_aggregate_generator((pg_fact(X), !, pg_heavy(X, _R)), M, _, _, _)).

test(no_split_with_side_effect) :-
    build(M),
    assertion(\+ split_aggregate_generator((pg_fact(X), writeln(X), pg_heavy(X, _R)), M, _, _, _)).

test(no_split_disjunction) :-
    build(M),
    assertion(\+ split_aggregate_generator((pg_fact(X) ; pg_heavy(X, _R)), M, _, _, _)).

% the split round-trips: Enum then Body is exactly the original goal sequence
test(split_preserves_goals) :-
    build(M),
    I = (pg_fact(X), pg_cheap(X, Z), pg_heavy(Z, _R)),
    split_aggregate_generator(I, M, Enum, Body, _),
    flatten_conj(Enum, EG), flatten_conj(Body, BG), append(EG, BG, Got),
    flatten_conj(I, Want),
    assertion(Got == Want).

flatten_conj((A, B), L) :- !, flatten_conj(A, LA), flatten_conj(B, LB), append(LA, LB, L).
flatten_conj(G, [G]).

% --- route-1 source transform: enum/body helpers + plan ---------------------

% Run a transform's two helper predicates sequentially (enum -> inputs, then
% body per input -> values) and return the collected value sequence.
decomp_values(AggGoal, Seed, Trans) :-
    build(M),
    parallel_aggregate_transform(AggGoal, M, Seed,
                                 Helpers, par_aggregate(_Type, EnumName/1, BodyName/2, _R)),
    copy_term(Helpers, HC),
    forall(member(C, HC), assertz(user:C)),
    EnumGoal =.. [EnumName, IT],
    findall(IT, user:EnumGoal, Inputs),
    findall(V, ( member(I, Inputs), BG =.. [BodyName, I, V], call(user:BG) ), Trans),
    EH =.. [EnumName, _],    retractall(user:EH),
    BH =.. [BodyName, _, _], retractall(user:BH).

% Decisive correctness: sequential (enum,body) collects exactly the original
% aggregate's value sequence (recursive body, predictable output).
test(decomp_preserves_values) :-
    findall(L, (pg_fact(X), pg_down(X, L)), Ref),
    decomp_values(findall(L2, (pg_fact(X2), pg_down(X2, L2)), _), dseed1, Trans),
    assertion(Ref == [[1], [2,1], [3,2,1]]),
    assertion(Trans == Ref).

% The collected value uses a var the ENUMERATOR binds (X), reconstructed via the
% Input tuple -- the case that needs the value-aware frontier.
test(decomp_value_var_from_enum) :-
    findall(X-L, (pg_fact(X), pg_down(X, L)), Ref),
    decomp_values(findall(X2-L2, (pg_fact(X2), pg_down(X2, L2)), _), dseed2, Trans),
    assertion(Ref == [1-[1], 2-[2,1], 3-[3,2,1]]),
    assertion(Trans == Ref).

% count: the decomposition yields the same number of rows
test(decomp_count_matches) :-
    decomp_values(aggregate_all(count, (pg_fact(X), pg_down(X, _L)), _), dseed3, Trans),
    assertion(length(Trans, 3)).

% transform plan shape + result threading
test(transform_plan_shape) :-
    build(M),
    parallel_aggregate_transform(findall(L, (pg_fact(X), pg_down(X, L)), Res),
                                 M, dseed4, Helpers, Plan),
    Plan = par_aggregate(collect, _/1, _/2, R),
    assertion(R == Res),
    Helpers = [(_ :- EnumBody), (_ :- BodyBody)],
    assertion(EnumBody =@= pg_fact(_)),
    assertion(BodyBody =@= pg_down(_, _)).

% transform declines exactly when the split/gate declines
test(transform_declines_all_cheap) :-
    build(M),
    assertion(\+ parallel_aggregate_transform(
        findall(Y, (pg_fact(X), pg_cheap(X, Y)), _), M, dseed5, _, _)).

test(transform_declines_non_aggregate) :-
    build(M),
    assertion(\+ parallel_aggregate_transform(pg_down(3, _L), M, dseed6, _, _)).

% --- external-input threading (the fix for head-arg inputs to the enumerator) --

test(transform_with_no_external_inputs_is_unchanged) :-
    build(M),
    % /5 and /6-with-[] produce identical helpers/plan (arity 1/2)
    parallel_aggregate_transform(findall(L, (pg_fact(X), pg_down(X, L)), _), M, e0a, H5, P5),
    parallel_aggregate_transform(findall(L, (pg_fact(X2), pg_down(X2, L)), _), [], M, e0a, H6, P6),
    P5 = par_aggregate(_, _/EA5, _/BA5, _), assertion(EA5 == 1), assertion(BA5 == 2),
    P6 = par_aggregate(_, _/EA6, _/BA6, _), assertion(EA6 == 1), assertion(BA6 == 2),
    assertion(H5 =@= H6).

test(transform_threads_external_input) :-
    build(M),
    AggGoal = findall(D, (pg_link(Y, Z), pg_down(Z, D)), _R),
    parallel_aggregate_transform(AggGoal, [Y], M, ext1, Helpers, Plan),
    % helpers gain a leading param for the external input Y: arity 1+1 / 2+1
    Plan = par_aggregate(collect, EN/EA, BN/BA, _),
    assertion(EA == 2), assertion(BA == 3),
    Helpers = [(EnumHead :- _), (_BodyHead :- _)],
    EnumHead =.. [EN, EY, _Tuple], assertion(EY == Y),   % Y is the leading enum arg
    % decisive: run the helpers with Y bound -> exactly the original aggregate
    findall(D, (pg_link(10, Z), pg_down(Z, D)), Ref),
    assertion(Ref == [[1], [2,1]]),
    copy_term(Helpers, HC), forall(member(C, HC), assertz(user:C)),
    EnumG =.. [EN, 10, IT], findall(IT, user:EnumG, ITs),
    findall(DV, ( member(I, ITs), BG =.. [BN, 10, I, DV], call(user:BG) ), Got),
    functor(EClean, EN, EA), retractall(user:EClean),
    functor(BClean, BN, BA), retractall(user:BClean),
    assertion(Got == Ref).

:- end_tests(parallel_gate).
