% test_wam_rust_fact_table.pl
%
% T9 fact-table inline, step 1 (detection + row extraction):
% wam_rust_target:rust_fact_table_classify/3 recognises a ground-unit-clause
% predicate and extracts its rows (source order), gated by t9_min_rows. Declines
% rule-bearing, non-ground, and below-threshold predicates. Pure Prolog (no cargo).

:- use_module('../src/unifyweaver/targets/wam_rust_target').

% ---- fixtures --------------------------------------------------------------
ft_color(red). ft_color(green). ft_color(blue).
ft_edge(a, b). ft_edge(a, c). ft_edge(b, d).
ft_mixed(1). ft_mixed(2). ft_mixed(X) :- X > 100.   % has a rule -> not a fact table
ft_nonground(foo). ft_nonground(_).                 % a non-ground unit clause
ft_small(only).                                     % 1 row -> below threshold

classify(PI, Opts, Info) :-
    wam_rust_target:rust_fact_table_classify(user:PI, Opts, Info).

:- begin_tests(wam_rust_fact_table).

test(detects_unary_facts_in_order) :-
    classify(ft_color/1, [t9_min_rows(2)], fact_info(A, Rows)),
    assertion(A == 1),
    assertion(Rows == [[red], [green], [blue]]).

test(detects_binary_facts_in_order) :-
    classify(ft_edge/2, [t9_min_rows(2)], fact_info(A, Rows)),
    assertion(A == 2),
    assertion(Rows == [[a, b], [a, c], [b, d]]).

test(declines_predicate_with_a_rule) :-
    assertion(\+ classify(ft_mixed/1, [t9_min_rows(1)], _)).

test(declines_non_ground_unit_clause) :-
    assertion(\+ classify(ft_nonground/1, [t9_min_rows(1)], _)).

test(declines_below_threshold) :-
    % ft_color has 3 rows; threshold 4 -> decline
    assertion(\+ classify(ft_color/1, [t9_min_rows(4)], _)),
    % ft_small has 1 row
    assertion(\+ classify(ft_small/1, [t9_min_rows(2)], _)).

test(default_threshold_is_64) :-
    % no t9_min_rows option -> default 64 -> ft_color (3 rows) declines
    assertion(\+ classify(ft_color/1, [], _)).

:- end_tests(wam_rust_fact_table).
