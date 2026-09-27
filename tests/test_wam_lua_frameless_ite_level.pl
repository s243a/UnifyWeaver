:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_lua_frameless_ite_level.pl
%
% Static verdict probe for the WAM_FLEET_GAPS gap-A2 hazard in its
% *frameless-Y-write* form on wam_lua. Companion to
% tests/test_wam_{python,haskell,go}_frameless_ite_level.pl (ledger rows
% D50/D52) and to the "unreachable" probes for r and fsharp.
%
% VERDICT: EXPOSED, UNFIXED.
% ---------------------------------------------------------------------------
% All three preconditions hold, and nothing in the runtime protects against
% the consequence:
%
%   1. wam_lua ALWAYS passes `ite_use_y_level(true)`
%      (`wam_lua_target.pl` compile_lua_predicate_wam/3, both arms), so the
%      shared emitter's `compile_if_then_else/7` plants `get_level Yn` /
%      `cut Yn` in clauses that get no `allocate`.
%   2. The Lua emitter ACCEPTS that shape rather than refusing loudly:
%      `wam_parts_to_lua(["get_level", Yn], 'I.GetLevel(<reg>)')`. Y_n maps
%      to `n + 200` (`reg_to_int/2`), so `get_level Y1` becomes
%      `I.GetLevel(201)` -- the same slot a framed CALLER uses for its own
%      first permanent (`I.GetVariable(201, 3)`).
%   3. The runtime routes it into the single flat, VM-global register table:
%      `Runtime.put_reg(state, instr.yn, #state.cps)` where
%      `Runtime.put_reg(state, idx, val)` is `state.regs[idx] = val`.
%      There is NO protection anywhere: `Allocate` pushes only
%      `{ cp = state.cp, locals = {} }` (no Y snapshot, and `Deallocate`
%      restores nothing), and `Call` only sets `state.cp` -- there is no
%      analogue of wam_javascript's / wam_go's Call-time Y save.
%
% So a caller holding a permanent across a call to an `allocate`-less
% if-then-else clause has its Y1 overwritten with a choice-point depth and
% never repaired. wam_lua is worse off than wam_go was before its fix: Go's
% interpreter lane was covered by `vm.YSaves` and only the lowered lane
% broke; Lua has no such cover on any lane.
%
% Not fixed here: this round owns wam_go; lua and llvm get audited verdicts
% only. wam_lua also has no conformance arm at all (WAM_FLEET_GAPS B-H2), so
% there is no execution oracle to diff against without first building one --
% recorded in docs/WAM_LUA_STATUS.md.
%
% The fix, when someone takes it: the wam_rust `ChoicePoint::levels` /
% wam_python `ChoicePoint.levels` / wam_go `ChoicePoint.Levels` model --
% keep the barrier level on the if-then-else's own choice point and never
% write it into a register.
%
% THESE TESTS ASSERT THE DEFECT. When wam_lua is fixed, the three
% `..._is_still_exposed` tests must be inverted (or deleted) and the verdict
% updated in docs/WAM_LUA_STATUS.md and docs/WAM_FLEET_GAPS.md. Their going
% red is the signal, not a nuisance.
%
% Emission-level only -- no Lua interpreter is required to run this suite.
%
%   swipl -q -g run_tests -t halt tests/test_wam_lua_frameless_ite_level.pl

:- module(test_wam_lua_frameless_ite_level,
          [test_wam_lua_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3,
                                 delete_directory_and_contents/1]).
:- use_module('../src/unifyweaver/targets/wam_lua_target',
              [write_wam_lua_project/3, reg_to_int/2]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:lt/2, user:sat/2, user:pick_b/4.

% --- the probe program -----------------------------------------------------

user:lt(A, B) :- A < B.

% Clause 2 needs NO environment: its only permanent would be the ITE barrier
% the emitter reserves for the inlined negation.
user:sat(_V, any).
user:sat(V, gte(G)) :- \+ user:lt(V, G).

% A caller that DOES hold a permanent across the call to it.
user:pick_b(V, C, Tag, Out) :- user:sat(V, C), Tag = Out.

:- dynamic repo_root/1.
:- prolog_load_context(directory, TestsDir),
   file_directory_name(TestsDir, Root),
   asserta(repo_root(Root)).

repo_file(Rel, Abs) :-
    repo_root(Root),
    directory_file_path(Root, Rel, Abs).

lua_runtime_template(Text) :-
    repo_file('templates/targets/lua_wam/runtime.lua.mustache', Path),
    read_file_to_string(Path, Text, []).

generated_lua(Text) :-
    Dir = 'output/test_wam_lua_frameless_ite_level_gen',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_lua_project([user:lt/2, user:sat/2, user:pick_b/4], [], Dir),
    directory_file_path(Dir, 'lua', LuaDir),
    directory_file_path(LuaDir, 'generated_program.lua', Prog),
    read_file_to_string(Prog, Text, []),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

test_wam_lua_frameless_ite_level :-
    run_tests(wam_lua_frameless_ite_level).

:- begin_tests(wam_lua_frameless_ite_level).

% Precondition 1 -- the shared emitter still produces the hazard shape.
test(shared_emitter_plants_get_level_without_allocate) :-
    compile_predicate_to_wam_text(user:sat/2, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

% Precondition 2 -- wam_lua passes the flag AND its emitter accepts the
% instruction (contrast wam_r / wam_fsharp, which pass `[]` and would refuse
% loudly). Asserted end-to-end on the generated Lua: the frameless `sat/2`
% clause carries `I.GetLevel(201)` with no `I.Allocate()`, and the framed
% caller writes its own first permanent to the SAME slot 201.
test(lua_emits_get_level_into_a_frameless_clause) :-
    once(generated_lua(Src)),
    assertion(reg_to_int('Y1', 201)),
    assertion(sub_string(Src, _, _, _, "I.GetLevel(201)")),
    assertion(sub_string(Src, _, _, _, "I.Cut(201)")),
    % the barrier is planted with no Allocate between the clause head and it
    once(sub_string(Src, GLBefore, _, _, "I.GetLevel(201)")),
    once(sub_string(Src, 0, GLBefore, _, Prefix)),
    (   once(sub_string(Prefix, ABefore, _, _, "I.Allocate()"))
    ->  assertion(ABefore < GLBefore),
       % ... and the nearest preceding TrustMe/TryMeElse must come after it,
       % i.e. that Allocate belongs to an earlier predicate, not this clause.
       assertion(sub_string(Prefix, _, _, _, "I.TrustMe()"))
    ;  true
    ),
    % the framed caller parks a permanent in the very same slot
    assertion(sub_string(Src, _, _, _, "I.GetVariable(201,")).

% Precondition 3 (a) -- the runtime writes the barrier into a register.
test(runtime_get_level_is_still_exposed) :-
    lua_runtime_template(Rt),
    assertion(sub_string(Rt, _, _, _,
        "Runtime.put_reg(state, instr.yn, #state.cps)")),
    assertion(sub_string(Rt, _, _, _,
        "local lvl = Runtime.get_reg(state, instr.yn)")).

% Precondition 3 (b) -- that register lives in one flat, VM-global table.
test(runtime_registers_are_still_exposed) :-
    lua_runtime_template(Rt),
    assertion(sub_string(Rt, _, _, _,
        "function Runtime.put_reg(state, idx, val) state.regs[idx] = val end")),
    assertion(sub_string(Rt, _, _, _,
        "function Runtime.get_reg(state, idx) return state.regs[idx] end")).

% Precondition 3 (c) -- nothing saves or restores the Y range. `Allocate`
% stores only the continuation, and `Call` only sets state.cp: there is no
% Y snapshot on either, so an `allocate`-less callee's write is permanent.
test(runtime_has_no_y_save_and_is_still_exposed) :-
    lua_runtime_template(Rt),
    assertion(sub_string(Rt, _, _, _,
        "table.insert(state.stack, { cp = state.cp, locals = {} })")),
    assertion(\+ sub_string(Rt, _, _, _, "ysave")),
    assertion(\+ sub_string(Rt, _, _, _, "y_save")),
    assertion(\+ sub_string(Rt, _, _, _, "saved_ys")),
    % the Call arm: continuation only.
    once(sub_string(Rt, CB, _, _, "if op == \"Call\" then")),
    once(sub_string(Rt, CB, 220, _, CallArm)),
    assertion(sub_string(CallArm, _, _, _, "state.cp = state.pc + 1")),
    assertion(\+ sub_string(CallArm, _, _, _, "regs")).

:- end_tests(wam_lua_frameless_ite_level).
