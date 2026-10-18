:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_r_frameless_ite_level.pl
%
% Verdict pin for the WAM_FLEET_GAPS gap-A2 hazard in its *frameless-Y-write*
% form on wam_r: UNREACHABLE on this target's own emitter path.
%
% Background
% ----------
% `compile_if_then_else/7` in the shared emitter reserves a permanent Y for
% the if-then-else barrier AFTER deciding whether the clause needs an
% environment, so under `ite_use_y_level(true)` it emits `get_level Yn` in
% `allocate`-less clauses. Runtimes that route Y writes into "the topmost env
% frame" then write the CALLER's frame. That is real: it produced 15
% differential divergences on wam_rust (ledger D50) and is reproduced and
% fixed on wam_python / wam_haskell in this round.
%
% wam_r is NOT exposed to it, for one reason and one reason only:
% **wam_r never turns the M17 barrier on.** Every WAM compile in
% `wam_r_target.pl` passes a literal `[]` option list
% (`compile_predicate_to_wam_text(Pred, [], ...)` at the project writer, the
% items-mode bridge and the ISO audit), so `compile_if_then_else/7` takes its
% `BarrierReg = none` branch: no Y is reserved for the barrier at all and the
% if-then-else compiles to the legacy `cut_ite`.
%
% This is a *routing* immunity, not a structural one. The R runtime IS
% frame-based -- `WamRuntime$put_reg` sends every `idx >= 201` into
% `state$stack[[length(state$stack)]]$ys`, i.e. the topmost frame
% (`templates/targets/r_wam/runtime.R.mustache`) -- so if wam_r ever opts into
% `ite_use_y_level(true)` it inherits the hazard and must adopt the
% choice-point model (see `ChoicePoint.levels` in the Rust runtime, or
% `cpLevels` in wam_haskell). The one mitigation R does have is that the
% instruction decoder has no `get_level` clause: the tokens fall through to
% `Raw(...)`, which the runtime dispatcher answers with
% `stop("unrecognized WAM instruction: ...")` -- loud, not silent.
%
% So this suite pins three things, all of which would have to change together
% for the hazard to become reachable:
%
%   1. the shared emitter, under the option set wam_r uses, emits NO
%      `get_level` and NO Y register in any `allocate`-less clause -- checked
%      on the exact program shape that breaks the M17 targets;
%   2. the generated R project for that program carries `CutIte` and no
%      `get_level`;
%   3. `wam_parts_to_r/2` still has no `get_level` clause, so the shape would
%      arrive as a loud `Raw`, never as a silent register write.
%
% No Rscript needed: every check is emission-level.
%
%   swipl -q -g run_tests -t halt tests/test_wam_r_frameless_ite_level.pl

:- module(test_wam_r_frameless_ite_level,
          [test_wam_r_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_r_target',
              [write_wam_r_project/3, wam_parts_to_r/2]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:rf_lt/2, user:rf_sat/2, user:rf_pick_a/4, user:rf_pick_b/4,
           user:rf_ite/2.

% The D50 reproduction shape: a multi-clause callee whose if-then-else clause
% needs no environment, called from a parent that has one with live Ys.
user:rf_lt(A, B) :- A < B.
user:rf_sat(_V, any).
user:rf_sat(V, gte(G)) :- \+ user:rf_lt(V, G).
user:rf_pick_a(Ver, C, Tag, Out) :- user:rf_sat(Ver, C), Out = Tag.
user:rf_pick_b(Ver, C, Tag, Out) :- user:rf_sat(Ver, C), Tag = Out.

% An explicit `->/2` as well: under the legacy option set `\+` is rewritten to
% `((G, !, fail) ; true)` (the hard-cut form) and never reaches
% compile_if_then_else/7 at all, so this is the predicate that shows the
% legacy soft cut wam_r actually consumes.
user:rf_ite(X, Y) :- ( X > 0 -> Y = pos ; Y = nonpos ).

probe_preds([user:rf_lt/2, user:rf_sat/2,
             user:rf_pick_a/4, user:rf_pick_b/4, user:rf_ite/2]).

test_wam_r_frameless_ite_level :-
    run_tests(wam_r_frameless_ite_level).

:- begin_tests(wam_r_frameless_ite_level).

% 1a. The shape IS a hazard under the M17 option set -- if this ever stops
% holding, the probe below is vacuous and should be revisited.
test(shape_is_a_hazard_under_m17) :-
    compile_predicate_to_wam_text(user:rf_sat/2,
                                  [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

% 1b. ... and is NOT one under the option set wam_r actually passes.
test(no_get_level_under_r_option_set) :-
    probe_preds(Preds),
    forall(member(PI, Preds),
           ( compile_predicate_to_wam_text(PI, [], T),
             atom_string(T, S),
             assertion(\+ sub_string(S, _, _, _, "get_level")) )).

% 1c. Stronger: no `allocate`-less clause names a Y register at all. This is
% the general statement of the hazard, not just its if-then-else instance.
test(no_frameless_clause_names_a_y_register) :-
    probe_preds(Preds),
    forall(member(PI, Preds),
           ( compile_predicate_to_wam_text(PI, [], T),
             frameless_y_regs(T, Ys),
             assertion(Ys == []) )).

% 2. The generated R project shows the legacy soft cut and no get_level.
test(generated_project_uses_cut_ite) :-
    probe_preds(Preds),
    Dir = 'output/test_wam_r_frameless_ite_level',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    once(write_wam_r_project(Preds, [module_name(rfprobe)], Dir)),
    atomic_list_concat([Dir, '/R/generated_program.R'], ProgPath),
    read_file_to_string(ProgPath, Src, []),
    assertion(sub_string(Src, _, _, _, "CutIte()")),
    assertion(\+ sub_string(Src, _, _, _, "get_level")),
    assertion(\+ sub_string(Src, _, _, _, "GetLevel")).

% 3. The decoder has no get_level clause, so the shape would arrive as Raw --
% which the runtime turns into a hard stop(), not a silent frame write.
test(decoder_routes_get_level_to_a_loud_raw) :-
    once(wam_parts_to_r(["get_level", "Y1"], Lit)),
    atom_string(Lit, S),
    assertion(sub_string(S, _, _, _, "Raw(")).

:- end_tests(wam_r_frameless_ite_level).

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
