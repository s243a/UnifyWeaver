% test_wam_rust_embedded_wiring.pl
%
% T7 embedded-aggregate wiring, step 1 (pure clause-lifting pass):
% wam_rust_target:rust_lift_predicate_clauses/4 lifts every embedded
% parallel-eligible aggregate in a predicate's clauses into whole-body helper
% predicates + in-place calls, WITHOUT mutating the user's module. Pure Prolog.

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/core/cost_analysis').

% ---- fixtures --------------------------------------------------------------
ew_base(1, 10). ew_base(2, 20).
ew_link(10, 1). ew_link(10, 2). ew_link(20, 3).
ew_down(0, []).
ew_down(N, [N|T]) :- N > 0, M is N - 1, ew_down(M, T).
ew_cheap(X, Y) :- Y is X * 2.

% predicate with an EMBEDDED parallel-eligible aggregate
ew_pred(X, Result) :- ew_base(X, Y), findall(D, (ew_link(Y, Z), ew_down(Z, D)), Result).
% predicate with NO embedded aggregate (plain rule)
ew_plain(X, Y) :- ew_base(X, Y).
% predicate whose embedded aggregate is cheap (not eligible)
ew_cheapagg(X, R) :- ew_base(X, Y), findall(V, (ew_link(Y, _), ew_cheap(Y, V)), R).

build(M) :- build_cost_model(user, M).
lift(PI, Rew, Helpers) :-
    wam_rust_target:rust_lift_predicate_clauses(user:PI, _Model, Rew, Helpers).
lift(PI, M, Rew, Helpers) :-
    wam_rust_target:rust_lift_predicate_clauses(user:PI, M, Rew, Helpers).

:- begin_tests(wam_rust_embedded_wiring).

test(lifts_embedded_aggregate) :-
    build(M),
    lift(ew_pred/2, M, Rewritten, Helpers),
    % one clause rewritten, one helper synthesised
    assertion(Rewritten = [(_ :- _)]),
    assertion(Helpers = [(_ :- _)]),
    Rewritten = [(_Head :- NewBody)],
    Helpers = [(HelperHead :- HelperBody)],
    % rewritten body: ew_base(X,Y), __lift_agg_...(Y, Result)
    assertion(NewBody = (ew_base(_, _), _Call)),
    % helper is the whole-body aggregate (Z link<->down, D tmpl<->down preserved)
    assertion(HelperBody =@= findall(DD, (ew_link(_, ZZ), ew_down(ZZ, DD)), _)),
    functor(HelperHead, HN, _), assertion(sub_atom(HN, 0, _, _, '__lift_agg')).

test(does_not_mutate_user_module) :-
    build(M),
    % snapshot ew_pred's clauses, lift, snapshot again -> unchanged
    findall(B0, clause(ew_pred(_, _), B0), Before),
    lift(ew_pred/2, M, _, _),
    findall(B1, clause(ew_pred(_, _), B1), After),
    assertion(Before =@= After),
    % and no __lift_agg_* predicate leaked into the DB
    assertion(\+ ( current_predicate(N/_), sub_atom(N, 0, _, _, '__lift_agg') )).

test(declines_no_embedded_aggregate) :-
    build(M),
    assertion(\+ lift(ew_plain/2, M, _, _)).

test(declines_cheap_embedded_aggregate) :-
    build(M),
    assertion(\+ lift(ew_cheapagg/2, M, _, _)).

% rewritten clause + helper, run sequentially, reproduce the original solutions
test(lifted_predicate_preserves_solutions) :-
    build(M),
    findall(X-R, ew_pred(X, R), Orig),
    lift(ew_pred/2, M, [(Head :- NewBody)], [Helper]),
    copy_term(Helper, HC), assertz(user:HC),
    findall(QX-QR,
            ( member(QX, [1, 2]),
              copy_term(Head-NewBody, ew_pred(QX, QR)-Goal),
              call(user:Goal) ),
            New),
    Helper = (HH :- _), functor(HH, HN, HA), functor(Clean, HN, HA), retractall(user:Clean),
    assertion(New == Orig).

:- end_tests(wam_rust_embedded_wiring).
