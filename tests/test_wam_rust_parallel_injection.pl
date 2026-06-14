% test_wam_rust_parallel_injection.pl
%
% T7 R1.2c (compile-time injection, core logic). Unit-tests
% wam_rust_target:rust_parallel_aggregate_wrapper/4 — given a predicate whose
% single clause body is a parallel-eligible forkable aggregate, it returns the
% two helper clauses (__par_enum/1, __par_body/2) and a native Rust wrapper that
% calls par_collect + reduces + unifies the result arg. Pure Prolog (no cargo).

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/core/cost_analysis').
:- use_module('../src/unifyweaver/core/parallel_gate').

:- dynamic rpaw_fact/1.
rpaw_fact(1). rpaw_fact(2). rpaw_fact(3).
rpaw_down(0, []).
rpaw_down(N, [N|T]) :- N > 0, M is N - 1, rpaw_down(M, T).
rpaw_cheap(X, Y) :- Y is X * 2.

% single clause = a parallel-eligible aggregate (recursive body)
rpaw_collect(L) :- findall(D, (rpaw_fact(X), rpaw_down(X, D)), L).
% count variant
rpaw_count(N)   :- aggregate_all(count, (rpaw_fact(X), rpaw_down(X, _D)), N).
% NOT parallel-eligible (cheap body) -> wrapper should decline
rpaw_cheapagg(L) :- findall(Y, (rpaw_fact(X), rpaw_cheap(X, Y)), L).
% multi-clause -> decline
rpaw_multi(1). rpaw_multi(L) :- findall(D, (rpaw_fact(X), rpaw_down(X, D)), L).

wrap(PI, Helpers, W) :-
    wam_rust_target:rust_parallel_aggregate_wrapper(PI,
        [parallel_aggregates(true), module_name(user)], Helpers, W).

:- begin_tests(wam_rust_parallel_injection).

test(collect_wrapper_shape) :-
    wrap(rpaw_collect/1, Helpers, W),
    assertion(Helpers = [(_ :- _), (_ :- _)]),         % enum + body clauses
    assertion(sub_string(W, _, _, _, "par_collect")),
    assertion(sub_string(W, _, _, _, "Value::List(__vals)")),  % collect reduce
    assertion(sub_string(W, _, _, _, "vm.unify(&a1")),         % result is A1
    assertion(sub_string(W, _, _, _, "pub fn")).

test(helpers_are_enum_and_body) :-
    wrap(rpaw_collect/1, [(EH :- EB), (BH :- BB)], _),
    functor(EH, EN, 1), functor(BH, BN, 2),
    assertion(sub_atom(EN, 0, _, _, '__par_enum')),
    assertion(sub_atom(BN, 0, _, _, '__par_body')),
    % enum body is the cheap fact; body body is the recursive call
    assertion(EB =@= rpaw_fact(_)),
    assertion(BB =@= rpaw_down(_, _)).

test(count_wrapper_uses_len_reduce) :-
    wrap(rpaw_count/1, _, W),
    assertion(sub_string(W, _, _, _, "__vals.len() as i64")).

test(declines_cheap_aggregate) :-
    assertion(\+ wrap(rpaw_cheapagg/1, _, _)).

test(declines_multi_clause) :-
    assertion(\+ wrap(rpaw_multi/1, _, _)).

test(declines_when_feature_off) :-
    assertion(\+ wam_rust_target:rust_parallel_aggregate_wrapper(
        rpaw_collect/1, [module_name(user)], _, _)).

:- end_tests(wam_rust_parallel_injection).
