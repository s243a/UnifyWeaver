% test_wam_rust_embedded_lift.pl
%
% T7 embedded aggregates, step 1 (pure source transform):
% parallel_gate:lift_embedded_aggregate/6 lifts an aggregate embedded in a larger
% clause body into a whole-body helper predicate + an in-place call. The decisive
% test runs the rewritten body (with the helper asserted) and checks it produces
% exactly the original clause's solutions. Pure Prolog (no cargo).

:- use_module('../src/unifyweaver/core/cost_analysis').
:- use_module('../src/unifyweaver/core/parallel_gate').

% ---- fixture: a clause with an EMBEDDED parallel-eligible aggregate ----------
emb_base(1, 10). emb_base(2, 20).
emb_link(10, 1). emb_link(10, 2). emb_link(20, 3).
emb_down(0, []).
emb_down(N, [N|T]) :- N > 0, M is N - 1, emb_down(M, T).
% cheap arithmetic (for the negative case)
emb_cheap(X, Y) :- Y is X * 2.

% original: base, then an embedded findall over a recursive body
emb_p(X, Result) :- emb_base(X, Y), findall(D, (emb_link(Y, Z), emb_down(Z, D)), Result).

build(M) :- build_cost_model(user, M).

:- begin_tests(wam_rust_embedded_lift).

% Decisive: the lifted clause computes the same solutions as the original.
test(lift_preserves_solutions) :-
    build(M),
    findall(X-R, emb_p(X, R), Orig),
    assertion(Orig == [1-[[1],[2,1]], 2-[[3,2,1]]]),
    % lift the embedded aggregate
    Head = emb_p(HX, HR),
    Body = (emb_base(HX, Y), findall(D, (emb_link(Y, Z), emb_down(Z, D)), HR)),
    lift_embedded_aggregate(Head, Body, M, lt1, NewBody, Helper),
    copy_term(Helper, HC), assertz(user:HC),
    % run the rewritten body for each X, collecting results
    findall(QX-QR,
            ( member(QX, [1, 2]),
              copy_term((HX-HR)-NewBody, (QX-QR)-Goal),
              call(user:Goal) ),
            New),
    % cleanup the asserted helper
    Helper = (HelperHead :- _), functor(HelperHead, HN, HA),
    functor(Clean, HN, HA), retractall(user:Clean),
    assertion(New == Orig).

% Structural: helper is whole-body aggregate; call replaces the aggregate in place;
% input frontier is the var shared with the rest of the clause (Y), result last.
test(lift_shape_and_frontier) :-
    build(M),
    Head = emb_p(HX, HR),
    Body = (emb_base(HX, Y), findall(D, (emb_link(Y, Z), emb_down(Z, D)), HR)),
    lift_embedded_aggregate(Head, Body, M, lt2, NewBody, (HelperHead :- HelperBody)),
    % helper body is the original aggregate (sharing preserved: Z link<->down, D tmpl<->down)
    assertion(HelperBody =@= findall(DD, (emb_link(_, ZZ), emb_down(ZZ, DD)), _)),
    % helper head: __lift_agg_lt2(Y, HR) -> arity 2, last arg is the result HR
    functor(HelperHead, Name, 2),
    assertion(sub_atom(Name, 0, _, _, '__lift_agg')),
    HelperHead =.. [_, In, Res],
    assertion(In == Y), assertion(Res == HR),
    % rewritten body: emb_base(HX,Y), __lift_agg_lt2(Y, HR)  (unify OUTSIDE assertion)
    NewBody = (emb_base(HX, Y), Call),
    assertion(Call == HelperHead).

% Declines: a whole-body aggregate (single goal) is NOT an embedded case.
test(declines_whole_body) :-
    build(M),
    assertion(\+ lift_embedded_aggregate(wb(R),
        findall(D, (emb_link(10, Z), emb_down(Z, D)), R), M, lt3, _, _)).

% Declines: embedded but cheap (not parallel-eligible).
test(declines_cheap_embedded) :-
    build(M),
    assertion(\+ lift_embedded_aggregate(cp(X, R),
        (emb_base(X, Y), findall(V, (emb_link(Y, _), emb_cheap(Y, V)), R)),
        M, lt4, _, _)).

:- end_tests(wam_rust_embedded_lift).
