:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_error_fixtures.pl - raise every sealed error fixture live and
% compare against the sealed form (typed-diagnostics oracle, closing
% the "no oracle; recommend own error-fixture goldens" row of
% DESIGN_prolog_elaborator.md §4).
%
% Run from this directory:
%   swipl -g run_tests -t halt test_error_fixtures.pl
%
% THE SEAL: fixture_sha256/1 below freezes the sha256 of
% ERROR_FIXTURES_pattern_stache.pl.  The suite verifies the file's
% hash before comparing anything, so a fixture edit and a test edit
% must meet in the same review — changing an error's sealed shape is a
% deliberate two-file act, never a drive-by.
%
% The fixture file is READ AS DATA (read_term/2 over a stream), never
% consulted as code.  vNext's Python error strings are deliberately
% not consulted — that coupling is forbidden by the standing rule.

:- use_module(pe_where,
              [where_semantic/2, where_full/2]).
:- use_module(pe_elaborate,
              [elaborate/3, elaborate/4]).
:- use_module(library(plunit)).
:- use_module(library(sha)).
:- use_module(library(lists)).

% Frozen at sealing time; regenerate deliberately with:
%   sha256sum ERROR_FIXTURES_pattern_stache.pl
fixture_sha256('0c84b2927726ec2d0de71f91120a7cc584973def61de2d3e0a1a16a3a0a9e745').

here(Dir) :-
    module_property(pe_elaborate, file(Here)),
    file_directory_name(Here, Dir).

fixture_path(Path) :-
    here(Dir),
    atomic_list_concat([Dir, '/ERROR_FIXTURES_pattern_stache.pl'], Path).

%% Load fixtures as data.
:- dynamic fixture/2.   % fixture(Class, SealedErrorTerm)

load_fixtures :-
    retractall(fixture(_, _)),
    fixture_path(Path),
    setup_call_cleanup(
        open(Path, read, S, [encoding(utf8)]),
        read_fixture_terms(S),
        close(S)).

read_fixture_terms(S) :-
    read_term(S, T, []),
    (   T == end_of_file
    ->  true
    ;   (   T = error_fixture(Class, Sealed)
        ->  assertz(fixture(Class, Sealed))
        ;   T = error_fixtures_version(_)
        ->  true
        ;   throw(error(error_fixtures(unexpected_term(T)), _))
        ),
        read_fixture_terms(S)
    ).

:- initialization(load_fixtures).

%% trigger(Class, Goal): the input that must raise each sealed class.
%  (This table plus the sealed terms IS the diagnostics contract:
%  input -> exact error.)
trigger(w_not_where,   where_semantic(lca_frac(simplemind), _)).
trigger(w_bad_list,    where_semantic(where(lca_frac(_C), oops), _)).
trigger(w_bad_binding, where_semantic(where(lca_frac(_C), [simplemind]), _)).
trigger(w_dup,         where_semantic(where(lca_frac(C), [C = simplemind, C = fs]), _)).
trigger(w_dead,        where_semantic(where(lca_frac(simplemind), [_C = fs]), _)).
trigger(w_pin_pos,     where_full(where(pin(lineage(pearltrees, decay(0.85)), P), [P = 'run/1']), _)).
trigger(w_illegal,     where_semantic(where(e5(margin(t(T))), [T = foo]), _)).
trigger(w_unbound,     where_semantic(where(product(hop_decay(C, gamma(0.6)), lca_frac(_D)), [C = simplemind]), _)).
trigger(e_bad_pairs,   elaborate(fs, [not_a_pair], _, _)).
trigger(e_dead,        elaborate(fs, [_B = simplemind], _)).
trigger(e_unknown,     elaborate(fs, [frobnicate(x)], _)).
trigger(e_failed,      elaborate(fs, [has_type(t1, substrate(haiku))-'surface:t1::substrate'], _, _)).
trigger(e_unsat,       elaborate(fs, [has_type(_X, substrate(frobnicate))], _)).
trigger(e_pin_pos,     elaborate(pin(lineage(pearltrees, decay(0.85)), P), [P = 'run/1'], _)).
trigger(e_pin_val,     elaborate(lca_frac(C), [C = pin(fs, 'run/1')], _)).

:- begin_tests(error_fixture_goldens).

% The seal: the fixture file's bytes are exactly the sealed bytes.
test(fixture_file_hash_is_sealed) :-
    fixture_path(Path),
    read_file_to_string(Path, S, [encoding(octet)]),
    sha_hash(S, H, [algorithm(sha256), encoding(octet)]),
    hash_atom(H, Actual),
    fixture_sha256(Expected),
    (   Actual == Expected
    ->  true
    ;   format(user_error,
               "ERROR_FIXTURES_pattern_stache.pl drifted from its seal~n  sealed ~w~n  actual ~w~nRe-seal deliberately: update fixture_sha256/1 in the same review.~n",
               [Expected, Actual]),
        fail
    ).

% Bijection: every sealed class has a trigger and every trigger a
% sealed class — no error class silently unaccounted for.
test(fixture_trigger_bijection) :-
    forall(fixture(Class, _), trigger(Class, _)),
    forall(trigger(Class, _), fixture(Class, _)).

% Raise each class live; the thrown term (variables numbered in
% first-occurrence order) must equal the sealed term exactly.
test(raised_error_equals_sealed_form, [forall(fixture(Class, Sealed))]) :-
    trigger(Class, Goal),
    catch(
        ( Goal,
          format(user_error, "~w: trigger did not raise~n", [Class]),
          fail
        ),
        error(E, _),
        ( copy_term(E, EC),
          numbervars(EC, 0, _),
          (   EC == Sealed
          ->  true
          ;   format(user_error, "~w:~n  sealed ~q~n  raised ~q~n",
                     [Class, Sealed, EC]),
              fail
          )
        )).

:- end_tests(error_fixture_goldens).
