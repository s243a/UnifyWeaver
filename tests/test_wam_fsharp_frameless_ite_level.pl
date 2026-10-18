:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_fsharp_frameless_ite_level.pl
%
% Verdict pin for the WAM_FLEET_GAPS gap-A2 hazard in its *frameless-Y-write*
% form on wam_fsharp: UNREACHABLE on this target's own emitter path.
%
% Background
% ----------
% `compile_if_then_else/7` in the shared emitter reserves a permanent Y for
% the if-then-else barrier AFTER deciding whether the clause needs an
% environment, so under `ite_use_y_level(true)` it emits `get_level Yn` in
% `allocate`-less clauses. Runtimes that route Y writes into "the topmost env
% frame" then write the CALLER's frame -- 15 differential divergences on
% wam_rust (ledger D50); reproduced and fixed on wam_python / wam_haskell in
% this round.
%
% wam_fsharp is NOT exposed to it, for two independent reasons:
%
%   1. **wam_fsharp never turns the M17 barrier on.** Every WAM compile in
%      `wam_fsharp_target.pl` passes a literal `[]`
%      (`wam_target:compile_predicate_to_wam(PI, [], WamCode)`), so
%      `compile_if_then_else/7` takes its `BarrierReg = none` branch: no Y is
%      reserved for the barrier and the if-then-else compiles to the legacy
%      `cut_ite` / `CutIte`.
%   2. **the emitter has no `get_level` clause at all.** `wam_instr_to_fsharp/2`
%      knows `cut_ite` and nothing else in that family, so the tokens fall
%      through to the catch-all, which warns on stderr at codegen time and
%      emits an `(* UNKNOWN: ... *) Proceed` stub -- noisy, never a silent
%      register write.
%
% This is a *routing* immunity, not a structural one. The F# runtime IS
% frame-based: `putReg` sends every `n >= 201` into the head of `s.WsStack`
% (`src/unifyweaver/bindings/fsharp_wam_bindings.pl`, the generated
% `WamRuntime.fs`), i.e. the topmost env frame -- the CALLER's for an
% `allocate`-less callee. If wam_fsharp ever opts into
% `ite_use_y_level(true)` it inherits the hazard and must adopt the
% choice-point model (`ChoicePoint::levels` in the Rust runtime; `cpLevels`
% in wam_haskell; `ChoicePoint.levels` in the Python runtime).
%
% No dotnet needed: every check is emission-level.
%
%   swipl -q -g run_tests -t halt tests/test_wam_fsharp_frameless_ite_level.pl

:- module(test_wam_fsharp_frameless_ite_level,
          [test_wam_fsharp_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:fsf_lt/2, user:fsf_sat/2, user:fsf_pick_a/4,
           user:fsf_pick_b/4, user:fsf_ite/2.

% The D50 reproduction shape: a multi-clause callee whose if-then-else clause
% needs no environment, called from a parent that has one with live Ys.
user:fsf_lt(A, B) :- A < B.
user:fsf_sat(_V, any).
user:fsf_sat(V, gte(G)) :- \+ user:fsf_lt(V, G).
user:fsf_pick_a(Ver, C, Tag, Out) :- user:fsf_sat(Ver, C), Out = Tag.
user:fsf_pick_b(Ver, C, Tag, Out) :- user:fsf_sat(Ver, C), Tag = Out.

% An explicit `->/2` as well: under the legacy option set `\+` is rewritten to
% `((G, !, fail) ; true)` and never reaches compile_if_then_else/7, so this is
% the predicate that shows the legacy soft cut wam_fsharp actually consumes.
user:fsf_ite(X, Y) :- ( X > 0 -> Y = pos ; Y = nonpos ).

probe_preds([user:fsf_lt/2, user:fsf_sat/2, user:fsf_pick_a/4,
             user:fsf_pick_b/4, user:fsf_ite/2]).

test_wam_fsharp_frameless_ite_level :-
    run_tests(wam_fsharp_frameless_ite_level).

:- begin_tests(wam_fsharp_frameless_ite_level).

% 1a. The shape IS a hazard under the M17 option set -- if this ever stops
% holding, the probes below are vacuous and should be revisited.
test(shape_is_a_hazard_under_m17) :-
    compile_predicate_to_wam_text(user:fsf_sat/2,
                                  [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

% 1b. ... and is NOT one under the option set wam_fsharp actually passes.
test(no_get_level_under_fsharp_option_set) :-
    probe_preds(Preds),
    forall(member(PI, Preds),
           ( compile_predicate_to_wam_text(PI, [], T),
             atom_string(T, S),
             assertion(\+ sub_string(S, _, _, _, "get_level")) )).

% 1c. Stronger: no `allocate`-less clause names a Y register at all.
test(no_frameless_clause_names_a_y_register) :-
    probe_preds(Preds),
    forall(member(PI, Preds),
           ( compile_predicate_to_wam_text(PI, [], T),
             frameless_y_regs(T, Ys),
             assertion(Ys == []) )).

% 2. The generated F# project shows the legacy soft cut and no GetLevel.
test(generated_project_uses_cut_ite) :-
    probe_preds(Preds),
    Dir = 'output/test_wam_fsharp_frameless_ite_level',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    once(write_wam_fsharp_project(Preds, [module_name(fsfprobe)], Dir)),
    atomic_list_concat([Dir, '/Predicates.fs'], PredPath),
    read_file_to_string(PredPath, Src, []),
    assertion(sub_string(Src, _, _, _, "CutIte")),
    assertion(\+ sub_string(Src, _, _, _, "GetLevel")),
    assertion(\+ sub_string(Src, _, _, _, "get_level")).

% 3. The emitter has no get_level clause, so the shape would arrive as the
% loud UNKNOWN stub rather than as a silent frame write.
test(emitter_routes_get_level_to_the_unknown_stub) :-
    once(wam_fsharp_target:wam_instr_to_fsharp(["get_level", "Y1"], Fs)),
    atom_string(Fs, S),
    assertion(sub_string(S, _, _, _, "UNKNOWN")).

:- end_tests(wam_fsharp_frameless_ite_level).

% --- helpers ---------------------------------------------------------------
%
% Split emitted WAM text into clause regions (a clause body ends at `proceed`
% or `execute`) and report the Y registers named by regions with no
% `allocate`.

frameless_y_regs(Text, YRegs) :-
    atom_string(Text, S),
    split_string(S, "\n", "", Lines),
    maplist(line_tokens, Lines, Toks),
    clause_regions(Toks, [], Regions),
    findall(Y,
            ( member(R, Regions),
              \+ region_has_allocate(R),
              member(Line, R), member(Y, Line),
              string_concat("Y", D, Y), D \== "",
              catch(number_string(_, D), _, fail) ),
            Ys0),
    sort(Ys0, YRegs).

line_tokens(Line, Toks) :-
    split_string(Line, " \t,", " \t,", Toks0),
    exclude(==(""), Toks0, Toks).

clause_regions([], Acc, Out) :- ( Acc == [] -> Out = [] ; Out = [Acc] ).
clause_regions([Toks|Rest], Acc, Out) :-
    append(Acc, [Toks], Acc1),
    (   ( Toks = ["proceed"|_] ; Toks = ["execute"|_] )
    ->  clause_regions(Rest, [], More),
        Out = [Acc1|More]
    ;   clause_regions(Rest, Acc1, Out)
    ).

region_has_allocate(Region) :-
    member(Toks, Region), memberchk("allocate", Toks), !.
