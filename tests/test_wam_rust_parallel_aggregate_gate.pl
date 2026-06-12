% test_wam_rust_parallel_aggregate_gate.pl
%
% Slice 2a: the cost machinery driving real Rust codegen. Compiles predicates
% containing forkable aggregates through the WAM->Rust pipeline with the T7 gate
% enabled (parallel_aggregates(true)) and asserts the generated function carries
% a "parallel-eligible" annotation exactly when the aggregate's generator is in a
% parallel-worthy cost tier (recursive / expensive) — and not for cheap
% generators, predicates without aggregates, or when the feature is off.
%
% No interpreter semantics change here: this proves the compile-time decision
% (parallel_gate.aggregate_parallel_decision via the cost model) reaches codegen.

:- use_module('../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../src/unifyweaver/targets/wam_rust_target', [compile_wam_predicate_to_rust/4]).

% ---- fixture program (asserted into user) ----------------------------------
:- dynamic tpa_fact/1.
tpa_fact(1). tpa_fact(2). tpa_fact(3).

tpa_double(X, Y) :- Y is X * 2.                         % cheap per-branch

tpa_rec([]).                                            % recursive generator
tpa_rec([_|T]) :- tpa_rec(T).

tpa_heavy(X, R) :-                                      % heavy bounded per-branch
    atom_codes(X, Cs), msort(Cs, S1), reverse(S1, S2),
    atom_codes(A, S2), atom_concat(A, A, B), sub_atom(B, 0, 1, _, R),
    msort(S2, _), reverse(S2, _), atom_length(A, _).

% predicates under test
tpa_cheap_agg(Xs)  :- findall(Y, (tpa_fact(X), tpa_double(X, Y)), Xs).
tpa_rec_agg(Ls)    :- findall(L, (tpa_fact(_), tpa_rec(L)), Ls).
tpa_heavy_agg(Rs)  :- findall(R, (tpa_fact(X), tpa_heavy(X, R)), Rs).
tpa_plain(X, Y)    :- tpa_double(X, Y).                 % no aggregate

:- begin_tests(wam_rust_parallel_aggregate_gate).

% Compile a predicate to Rust with the T7 gate enabled.
rust_of(Name/Arity, RustCode) :-
    once(( compile_predicate_to_wam(user:Name/Arity, [], WamCode),
           compile_wam_predicate_to_rust(Name/Arity, WamCode,
               [parallel_aggregates(true), module_name(user)], RustCode) )).

rust_of_default(Name/Arity, RustCode) :-
    once(( compile_predicate_to_wam(user:Name/Arity, [], WamCode),
           compile_wam_predicate_to_rust(Name/Arity, WamCode, [module_name(user)], RustCode) )).

has_anno(RustCode) :- sub_string(RustCode, _, _, _, "parallel-eligible").

test(recursive_generator_annotated) :-
    rust_of(tpa_rec_agg/1, R),
    assertion(has_anno(R)),
    assertion(sub_string(R, _, _, _, "recursive")).

test(heavy_generator_annotated_expensive) :-
    rust_of(tpa_heavy_agg/1, R),
    assertion(has_anno(R)),
    assertion(sub_string(R, _, _, _, "expensive")).

test(cheap_generator_not_annotated) :-
    rust_of(tpa_cheap_agg/1, R),
    assertion(\+ has_anno(R)).

test(no_aggregate_not_annotated) :-
    rust_of(tpa_plain/2, R),
    assertion(\+ has_anno(R)).

% Feature gate: with the option absent, output carries no annotation (so default
% codegen is unchanged) even for a parallel-worthy generator.
test(feature_off_no_annotation) :-
    rust_of_default(tpa_rec_agg/1, R),
    assertion(\+ has_anno(R)).

% The gated decision still produces a valid Rust function either way.
test(annotated_code_still_well_formed) :-
    rust_of(tpa_rec_agg/1, R),
    assertion(sub_string(R, _, _, _, "pub fn")),
    assertion(sub_string(R, _, _, _, "-> bool")).

:- end_tests(wam_rust_parallel_aggregate_gate).
