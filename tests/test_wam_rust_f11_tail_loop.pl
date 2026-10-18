:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_f11_tail_loop.pl
%
% Emitter-level probes for Stage 1 F11 (self-tail-recursion -> native loop) in
% wam_rust_lowered_emitter.pl. See
% docs/reports/wam_rust_f11_census_and_stage1.md and
% docs/proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md §4.
%
% Pins the eligibility gate (the load-bearing soundness property) and the loop
% shape, without needing a Rust toolchain:
%
%   f01  a classic pure-body tail loop ([] vs [_|_]) IS tail_loop
%   f02  its emitted function is a `loop { }` that `continue`s and has no
%        per-recursion re-dispatch
%   f03  a NON-tail self-recursive predicate is NOT tail_loop
%   f04  a predicate whose body makes a user call is NOT tail_loop (pure-body)
%   f05  overlapping (unifiable) heads are rejected by the mutual-exclusivity
%        determinism gate even when the recursion shape fits
%   f06  a distinct-first-argument-structure walker (accessor shape) IS tail_loop
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_f11_tail_loop.pl

:- module(test_wam_rust_f11_tail_loop, [test_wam_rust_f11_tail_loop/0]).

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_target', []).
:- use_module('../src/unifyweaver/targets/wam_rust_lowered_emitter').

% --- synthetic predicates under test (asserted into user) ---
:- dynamic user:'$f11test_loaded'/0.
setup_preds :-
    % F11 is DEFAULT-OFF; these probes exercise it, so enable the emitter flag.
    nb_setval(rust_f11_flag, true),
    ( user:'$f11test_loaded' -> true
    ; assertz((user:f11_len([], A, A))),
      assertz((user:f11_len([_|T], A0, A) :- A1 is A0 + 1, f11_len(T, A1, A))),
      % non-tail: recursion is not the last goal
      assertz((user:f11_bad([], 0))),
      assertz((user:f11_bad([_|T], N) :- f11_bad(T, N0), N is N0 + 1)),
      % body makes a user call (pure-body gate must reject)
      assertz((user:f11_call([], []))),
      assertz((user:f11_call([X|Xs], [Y|Ys]) :- f11_helper(X, Y), f11_call(Xs, Ys))),
      assertz((user:f11_helper(a, 1))),
      % overlapping heads (both match a cons; NOT mutually exclusive)
      assertz((user:f11_overlap([X|Xs], [X|Xs]))),
      assertz((user:f11_overlap([_|Xs], Ys) :- f11_overlap(Xs, Ys))),
      % accessor shape: distinct first-arg structures + one recursive clause
      assertz((user:f11_acc(box(V), V))),
      assertz((user:f11_acc(wrap(Inner), V) :- f11_acc(Inner, V))),
      assertz(user:'$f11test_loaded')
    ).

wam_of(P/A, Wam) :-
    wam_target:compile_predicate_to_wam(user:P/A,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam).

class_of(P/A, R) :-
    wam_of(P/A, Wam),
    ( wam_rust_lowerable(P/A, Wam, R) -> true ; R = not_lowerable ).

heads_of(P/A, Heads) :-
    functor(Proto, P, A),
    findall(H, ( clause(user:Proto, _), copy_term(Proto, H) ), Heads).

:- begin_tests(f11).

test(f01_tail_loop_detected, [setup(setup_preds)]) :-
    class_of(f11_len/3, R),
    assertion(R == tail_loop).

test(f02_loop_shape, [setup(setup_preds)]) :-
    wam_of(f11_len/3, Wam),
    lower_predicate_to_rust(f11_len/3, Wam, [], Lines),
    atomic_list_concat(Lines, '\n', Src),
    assertion(sub_atom(Src, _, _, _, 'loop {')),
    assertion(sub_atom(Src, _, _, _, 'continue;')),
    assertion(sub_atom(Src, _, _, _, 'F11 self-tail-recursion')),
    % the recursion edge is the loop, not an interpreter re-dispatch of self
    assertion(\+ sub_atom(Src, _, _, _, 'labels.get("f11_len/3")')).

test(f03_non_tail_rejected, [setup(setup_preds)]) :-
    class_of(f11_bad/2, R),
    assertion(R \== tail_loop).

test(f04_body_user_call_rejected, [setup(setup_preds)]) :-
    class_of(f11_call/2, R),
    assertion(R \== tail_loop).

test(f05_overlapping_heads_not_exclusive, [setup(setup_preds)]) :-
    heads_of(f11_overlap/2, Heads),
    assertion(\+ rust_heads_mutually_exclusive(Heads)).

test(f06_accessor_tail_loop, [setup(setup_preds)]) :-
    heads_of(f11_acc/2, Heads),
    assertion(rust_heads_mutually_exclusive(Heads)),
    class_of(f11_acc/2, R),
    assertion(R == tail_loop).

:- end_tests(f11).

test_wam_rust_f11_tail_loop :-
    run_tests(f11).
