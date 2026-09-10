:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_llvm_frameless_ite_level.pl
%
% Static verdict probe for the WAM_FLEET_GAPS gap-A2 hazard in its
% *frameless-Y-write* form on wam_llvm. Companion to
% tests/test_wam_{python,haskell,go,lua}_frameless_ite_level.pl (ledger rows
% D50/D52).
%
% VERDICT: EXPOSED on the BYTECODE lane, UNFIXED. Immune on the LOWERED lane.
% ---------------------------------------------------------------------------
%   1. wam_llvm DEFAULTS to `ite_use_y_level(true)`
%      (`wam_llvm_target.pl` write_wam_llvm_project/3: the option is added
%      unless the caller already supplied one), so the shared emitter's
%      `compile_if_then_else/7` plants `get_level Yn` / `cut Yn` in clauses
%      that get no `allocate`.
%   2. The LLVM emitter ACCEPTS that shape rather than refusing loudly:
%      `wam_line_to_llvm_literal(["get_level", 'Y1'], ...)` yields opcode 33
%      with operand 48 -- Y1..Y16 live at regs[48..63] under the disjoint
%      X/Y ABI (`bindings/llvm_wam_bindings.pl` reg_name_to_index).
%   3. The `get_level` opcode body writes that register:
%      `call void @wam_set_reg(%WamState* %vm, i32 %gl.yn, %Value %gl.v)`,
%      and `@wam_set_reg` in templates/targets/llvm_wam/state.ll.mustache is
%      a GEP straight into the ONE flat %WamState register array (field 1) --
%      no environment-frame indirection.
%
% The only Y protection wam_llvm has is the `allocate` case, which memcpys
% regs[48..63] into the env frame (and `deallocate` restores it) -- which an
% `allocate`-less if-then-else clause never executes. `do_call` saves the
% continuation and the cut barrier and nothing else: there is no analogue of
% wam_javascript's / wam_go's Call-time Y snapshot. So a caller holding a
% permanent across a call into such a clause has its Y1 overwritten with a
% choice-point count and never repaired on the success path.
%
% `get_level` does call `@wam_trail_binding` before the write, so the old
% register value is on the trail -- but only a backtrack PAST that trail mark
% restores it, and the wrong answer is delivered on the success path (the
% `sat/2` shape returns to its caller normally). Trailing narrows the window;
% it does not close it.
%
% The LOWERED lane is immune for a structural reason worth recording:
% `wam_llvm_lowered_emitter.pl` renders `get_level Yn` as an explicit no-op
% (`supported(get_level(_))`, emit_instr/4 emits a comment and a branch),
% because the soft cut there is realised by the basic-block layout rather
% than by a choice-point level. Nothing is written to a register at all.
%
% Not fixed here: this round owns wam_go; lua and llvm get audited verdicts
% only. The fix, when someone takes it, is the wam_rust `ChoicePoint::levels`
% / wam_python `ChoicePoint.levels` / wam_go `ChoicePoint.Levels` model --
% keep the barrier level on the if-then-else's own choice point and never
% write it into a register.
%
% THE `..._is_still_exposed` TESTS ASSERT THE DEFECT. When wam_llvm is fixed
% they must be inverted (or deleted) and the verdict updated in
% docs/WAM_LLVM_STATUS.md and docs/WAM_FLEET_GAPS.md.
%
% Emission-level only -- no LLVM toolchain (llc/clang) is required.
%
%   swipl -q -g run_tests -t halt tests/test_wam_llvm_frameless_ite_level.pl

:- module(test_wam_llvm_frameless_ite_level,
          [test_wam_llvm_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3,
                                 delete_directory_and_contents/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target',
              [write_wam_llvm_project/3, wam_line_to_llvm_literal/2]).
:- use_module('../src/unifyweaver/targets/wam_llvm_lowered_emitter', []).
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

llvm_state_template(Text) :-
    repo_file('templates/targets/llvm_wam/state.ll.mustache', Path),
    read_file_to_string(Path, Text, []).

% The generated module for the probe program (written with the target's own
% default options, so the ite_use_y_level default is what is under test).
generated_ll(Text) :-
    Dir = 'output/test_wam_llvm_frameless_ite_level_gen',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    directory_file_path(Dir, 'probe.ll', Out),
    write_wam_llvm_project([user:lt/2, user:sat/2, user:pick_b/4], [], Out),
    read_file_to_string(Out, Text, [encoding(octet)]),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

test_wam_llvm_frameless_ite_level :-
    run_tests(wam_llvm_frameless_ite_level).

:- begin_tests(wam_llvm_frameless_ite_level).

% Precondition 1 -- the shared emitter still produces the hazard shape.
test(shared_emitter_plants_get_level_without_allocate) :-
    compile_predicate_to_wam_text(user:sat/2, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

% Precondition 2 -- the LLVM emitter accepts get_level/cut and encodes Y1 as
% the flat register index 48 (contrast wam_r / wam_fsharp, which have no
% clause for the instruction at all and refuse loudly).
test(llvm_encodes_get_level_and_cut_for_y1) :-
    once(wam_line_to_llvm_literal(["get_level", 'Y1'], GL)),
    once(wam_line_to_llvm_literal(["cut", 'Y1'], CU)),
    assertion(GL == '%Instruction { i32 33, i64 48, i64 0 }'),
    assertion(CU == '%Instruction { i32 34, i64 48, i64 0 }').

% ... and the default project build really does plant it: opcode 33 with
% operand 48 appears in the emitted code array, i.e. the target defaulted
% ite_use_y_level(true) and nothing refused the instruction.
test(default_project_build_plants_the_barrier) :-
    once(generated_ll(Src)),
    assertion(sub_string(Src, _, _, _, "i32 33, i64 48")),
    assertion(sub_string(Src, _, _, _, "i32 34, i64 48")).

% Precondition 3 (a) -- the get_level opcode body writes a register.
test(get_level_opcode_is_still_exposed) :-
    wam_llvm_target:wam_llvm_case('get_level', Body),
    assertion(sub_string(Body, _, _, _,
        "call void @wam_set_reg(%WamState* %vm, i32 %gl.yn, %Value %gl.v)")),
    wam_llvm_target:wam_llvm_case('cut', CutBody),
    assertion(sub_string(CutBody, _, _, _,
        "call %Value @wam_get_reg(%WamState* %vm, i32 %cy.yn)")).

% Precondition 3 (b) -- that register is a slot in the single flat %WamState
% register array (field 1), with no environment-frame indirection.
test(registers_are_still_a_flat_array_and_exposed) :-
    llvm_state_template(Rt),
    assertion(sub_string(Rt, _, _, _,
        "define void @wam_set_reg(%WamState* %vm, i32 %idx, %Value %val)")),
    once(sub_string(Rt, SB, _, _,
        "define void @wam_set_reg(%WamState* %vm, i32 %idx, %Value %val)")),
    once(sub_string(Rt, SB, 260, _, SetBody)),
    assertion(sub_string(SetBody, _, _, _,
        "getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 %idx")),
    assertion(\+ sub_string(SetBody, _, _, _, "StackEntry")).

% Precondition 3 (c) -- the ONLY Y save is the allocate/deallocate memcpy of
% regs[48..63], which an allocate-less clause never runs; `do_call` saves the
% continuation and the cut barrier and no registers at all.
test(only_allocate_saves_y_and_call_is_still_exposed) :-
    wam_llvm_target:wam_llvm_case('allocate', AllocBody),
    assertion(sub_string(AllocBody, _, _, _, "i32 0, i32 1, i32 48")),
    assertion(sub_string(AllocBody, _, _, _, "llvm.memcpy")),
    wam_llvm_target:wam_llvm_case('do_call', CallBody),
    assertion(sub_string(CallBody, _, _, _, "@wam_set_cp")),
    assertion(\+ sub_string(CallBody, _, _, _, "memcpy")),
    assertion(\+ sub_string(CallBody, _, _, _, "wam_set_reg")),
    wam_llvm_target:wam_llvm_case('proceed', ProceedBody),
    assertion(\+ sub_string(ProceedBody, _, _, _, "memcpy")),
    assertion(\+ sub_string(ProceedBody, _, _, _, "wam_set_reg")).

% The lowered lane is immune, structurally: get_level is an explicit no-op
% there because the soft cut is realised by the basic-block layout.
test(lowered_lane_treats_get_level_as_a_no_op) :-
    assertion(wam_llvm_lowered_emitter:supported(get_level(_))),
    wam_llvm_lowered_emitter:emit_instr(get_level('Y1'), 7, 'pc_8', Block),
    assertion(sub_string(Block, _, _, _, "no-op")),
    assertion(\+ sub_string(Block, _, _, _, "wam_set_reg")).

:- end_tests(wam_llvm_frameless_ite_level).
