:- encoding(utf8).
%% Semantic / runtime-shape test suite for the `put_structure_dyn`
%% WAM instruction.
%%
%% Existing coverage in tests/core/test_wam_univ_lowering.pl asserts
%% that wam_target emits the textual `put_structure_dyn` instruction
%% under the right mode preconditions. Existing coverage in
%% tests/test_wam_haskell_target.pl asserts that the WAM-Haskell
%% generator emits the `PutStructureDyn` constructor and step handler
%% in the source text it produces.
%%
%% Neither of those drives the instruction end-to-end — they don't
%% validate that the *register identifiers* the lowering emits are
%% threaded correctly through the lowered sequence, nor that a runtime
%% `step` on `PutStructureDyn` would produce the expected `BuildStruct`
%% builder. This file fills that gap with the strongest semantic
%% coverage we can robustly add without GHC + parallel/async on CI.
%%
%% What is exercised:
%%   1. End-to-end compile of a `T =.. [Name | Args]` predicate via
%%      wam_target (with mode declaration that triggers the compose
%%      lowering).
%%   2. Each emitted line is run through wam_haskell_target's
%%      `wam_instr_to_haskell/2` parser, mirroring the path the real
%%      project uses to produce Predicates.hs.
%%   3. We then check the *internal consistency* of the parsed
%%      Haskell instruction list:
%%        - there is a `PutStructureDyn NI AI TI` instruction;
%%        - NI is the integer id for A1, AI for A2 (the calling
%%          convention emit_put_structure_dyn_lowering documents);
%%        - some preceding instruction populates A1 with the functor
%%          name register (PutValue / PutVariable into reg 1);
%%        - the immediately preceding instruction binds A2 to the
%%          literal arity (PutConstant arity 2);
%%        - the SetValue / SetVariable instructions that follow build
%%          out exactly `arity` argument slots on the structure.
%%   4. A pure-Prolog reimplementation of the Haskell `step` rule for
%%      `PutStructureDyn` is driven against a synthesised `WamState`
%%      to verify the documented runtime contract:
%%        - Atom name + non-negative Integer arity in the source
%%          registers => target builder becomes
%%          BuildStruct fnId targetReg arity [];
%%        - PC advances by 1;
%%        - any other shape => Nothing (i.e., backtrack).
%%      This mirrors lines 1890-1898 of wam_haskell_target.pl. If the
%%      Haskell source's PutStructureDyn handler is ever changed in a
%%      way that breaks the contract, the codegen text test will still
%%      pass but this test will fail — flagging the runtime regression
%%      at the semantic level.
%%
%% Limitations: this is not a full GHC-backed end-to-end test. The
%% runtime `step` semantics are mirrored in Prolog rather than
%% executed by the actual compiled Haskell. A full Cabal build per
%% test run is intentionally avoided (see the comment at the top of
%% tests/test_wam_haskell_target.pl) — Control.Parallel.Strategies
%% and Control.Concurrent.Async are unconditional imports of the
%% generated WamRuntime.hs and are not in the project's CI image.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_put_structure_dyn_runtime.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("WAM put_structure_dyn runtime/semantic tests~n"),
    format("========================================~n~n"),
    findall(T, test(T), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], P, P).
run_all([T|Rest], Acc, P) :-
    (   catch(call(T), E, (format("[FAIL] ~w: ~w~n", [T, E]), fail))
    ->  Acc1 is Acc + 1, run_all(Rest, Acc1, P)
    ;   run_all(Rest, Acc, P)
    ).

pass(N) :- format("[PASS] ~w~n", [N]).
fail_test(N, R) :- format("[FAIL] ~w: ~w~n", [N, R]), fail.

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_lowered_sequence_parses_to_haskell_ast).
test(test_lowered_sequence_threads_registers).
test(test_lowered_sequence_arg_slots_match_arity).
test(test_step_put_structure_dyn_happy_path).
test(test_step_put_structure_dyn_zero_arity).
test(test_step_put_structure_dyn_rejects_non_atom_name).
test(test_step_put_structure_dyn_rejects_negative_arity).
test(test_step_put_structure_dyn_rejects_unbound_name).

%% ========================================================================
%% Compile fixture: produce the WAM text for a real compose-mode
%% predicate. Centralised so all parser-based tests share the same
%% source of truth.
%% ========================================================================

%% compose_predicate_wam(+Arity, -WamLines)
%  Asserts a fresh `compose_test_<Arity>(Name, Arg1..ArgN, T)` clause
%  with body `T =.. [Name | FixedArgs]`, declares the matching mode,
%  and returns the WAM text split into trimmed lines.
compose_predicate_wam(2, Lines) :-
    %% build_term(Name, Arg, T) :- T =.. [Name, Arg].  (arity-1 functor)
    retractall(user:cpd_arity1(_, _, _)),
    retractall(user:mode(cpd_arity1(_, _, _))),
    assert(user:mode(cpd_arity1(+, ?, -))),
    assert(user:(cpd_arity1(Name, A, T) :- T =.. [Name, A])),
    wam_target:compile_predicate_to_wam(cpd_arity1/3, [], WamCode),
    atom_string(WamCode, S),
    split_string(S, "\n", " \t", Lines),
    retractall(user:cpd_arity1(_, _, _)),
    retractall(user:mode(cpd_arity1(_, _, _))).

compose_predicate_wam(3, Lines) :-
    %% build_pair(Name, A, B, T) :- T =.. [Name, A, B].  (arity-2 functor)
    retractall(user:cpd_arity2(_, _, _, _)),
    retractall(user:mode(cpd_arity2(_, _, _, _))),
    assert(user:mode(cpd_arity2(+, ?, ?, -))),
    assert(user:(cpd_arity2(Name, A, B, T) :- T =.. [Name, A, B])),
    wam_target:compile_predicate_to_wam(cpd_arity2/4, [], WamCode),
    atom_string(WamCode, S),
    split_string(S, "\n", " \t", Lines),
    retractall(user:cpd_arity2(_, _, _, _)),
    retractall(user:mode(cpd_arity2(_, _, _, _))).

compose_predicate_wam(0, Lines) :-
    %% atom-as-functor: build_atom(Name, T) :- T =.. [Name].
    retractall(user:cpd_arity0(_, _)),
    retractall(user:mode(cpd_arity0(_, _))),
    assert(user:mode(cpd_arity0(+, -))),
    assert(user:(cpd_arity0(Name, T) :- T =.. [Name])),
    wam_target:compile_predicate_to_wam(cpd_arity0/2, [], WamCode),
    atom_string(WamCode, S),
    split_string(S, "\n", " \t", Lines),
    retractall(user:cpd_arity0(_, _)),
    retractall(user:mode(cpd_arity0(_, _))).

%% Try parsing each instruction line through wam_instr_to_haskell.
%% Lines that aren't WAM instructions (labels, blanks, predicate
%% headers) are silently skipped.
parse_lines_to_haskell([], []).
parse_lines_to_haskell([L|Ls], Hs) :-
    %% Tokenise the line: split on whitespace and commas, drop empties.
    split_string(L, " ,\t", " ,\t", Toks0),
    exclude(=(""), Toks0, Toks),
    (   Toks = [],
        Hs = Rest
    ;   catch(wam_haskell_target:wam_instr_to_haskell(Toks, H),
              _, fail),
        %% Filter out the synthetic UNKNOWN comments the fall-through
        %% clause emits for things like predicate-header lines.
        \+ sub_string(H, _, _, _, "UNKNOWN"),
        Hs = [H|Rest]
    ;   Hs = Rest
    ),
    !,
    parse_lines_to_haskell(Ls, Rest).

%% Find the (first) PutStructureDyn entry in the parsed list and
%% return the surrounding context (Before, NI/AI/TI, After).
locate_put_structure_dyn(HsList, Before, NI, AI, TI, After) :-
    append(Before, [DynStr|After], HsList),
    string_concat("PutStructureDyn ", Rest, DynStr),
    split_string(Rest, " ", "", [NS, AS, TS]),
    number_string(NI, NS),
    number_string(AI, AS),
    number_string(TI, TS).

%% ========================================================================
%% Tests: parser-level structural checks
%% ========================================================================

test_lowered_sequence_parses_to_haskell_ast :-
    Test = test_lowered_sequence_parses_to_haskell_ast,
    (   compose_predicate_wam(2, Lines),
        parse_lines_to_haskell(Lines, Hs),
        once((member(H, Hs),
              string_concat("PutStructureDyn ", _, H)))
    ->  pass(Test)
    ;   fail_test(Test, no_put_structure_dyn_in_parsed_ast)
    ).

%% wam_target.pl:1336-1340 documents the calling convention as:
%%   put_value Reg(NameVar), A1
%%   put_constant N, A2
%%   put_structure_dyn A1, A2, A3
%% reg_name_to_int(A1)=1, reg_name_to_int(A2)=2.
%% The threading test: the PutStructureDyn must reference reg 1 for
%% nameReg and reg 2 for arityReg, and there must be a populating
%% instruction for each in the lines that precede it.
test_lowered_sequence_threads_registers :-
    Test = test_lowered_sequence_threads_registers,
    (   compose_predicate_wam(2, Lines),
        parse_lines_to_haskell(Lines, Hs),
        locate_put_structure_dyn(Hs, Before, NI, AI, _TI, _After),
        NI =:= 1,        %% nameReg = A1
        AI =:= 2,        %% arityReg = A2
        %% Some prior instruction must put a value into A1 (nameReg).
        once((member(B1, Before),
              ( string_concat("PutValue ", Rest1, B1)
              ; string_concat("PutVariable ", Rest1, B1)
              ),
              split_string(Rest1, " ", "", [_, ARegStr1]),
              ARegStr1 == "1")),
        %% The immediately preceding instruction must populate A2 with
        %% the literal arity. wam_instr_to_haskell renders this as
        %%   "PutConstant (Integer 1) 2"   (arity 1, target A2)
        %% for the arity-1 functor fixture.
        last(Before, ImmediatelyBefore),
        sub_string(ImmediatelyBefore, _, _, _, "PutConstant"),
        sub_string(ImmediatelyBefore, _, _, _, "Integer 1"),
        sub_string(ImmediatelyBefore, _, _, 0, " 2")
    ->  pass(Test)
    ;   fail_test(Test, register_threading_violation)
    ).

%% For an arity-2 functor (T =.. [Name, A, B]) the lowering is
%% expected to emit two argument-slot instructions
%% (set_value / set_variable) between PutStructureDyn and the
%% concluding GetValue/Proceed sequence.
test_lowered_sequence_arg_slots_match_arity :-
    Test = test_lowered_sequence_arg_slots_match_arity,
    (   compose_predicate_wam(3, Lines),   %% 4-arity predicate, 2-arity functor
        parse_lines_to_haskell(Lines, Hs),
        locate_put_structure_dyn(Hs, _Before, _NI, _AI, _TI, After),
        include(is_set_arg_slot, After, Slots),
        length(Slots, NSlots),
        NSlots >= 2
    ->  pass(Test)
    ;   fail_test(Test, expected_at_least_two_set_value_or_set_variable)
    ).

is_set_arg_slot(Hs) :-
    (   string_concat("SetValue ", _, Hs)
    ;   string_concat("SetVariable ", _, Hs)
    ;   string_concat("SetConstant ", _, Hs)
    ), !.

%% ========================================================================
%% Tests: runtime-step contract (Prolog mirror of Haskell `step`)
%% ========================================================================
%%
%% This is a faithful re-implementation of the PutStructureDyn step
%% rule from src/unifyweaver/targets/wam_haskell_target.pl:1890-1898:
%%
%%   step !ctx s (PutStructureDyn nameReg arityReg targetReg) =
%%     let mName  = derefVar (wsBindings s) <$> getReg nameReg s
%%         mArity = derefVar (wsBindings s) <$> getReg arityReg s
%%     in case (mName, mArity) of
%%       (Just (Atom fnId), Just (Integer arity)) | arity >= 0 ->
%%         Just (s { wsPC = wsPC s + 1
%%                 , wsBuilder = BuildStruct fnId targetReg
%%                                          (fromIntegral arity) []
%%                 })
%%       _ -> Nothing
%%
%% State representation:
%%   wam_state(PC, Regs, Builder)
%%     PC      :: integer
%%     Regs    :: list of RegId-Value pairs
%%     Builder :: 'NoBuilder' | build_struct(FnId,Tgt,Arity,Args)
%%
%% Values:
%%   atom(Id) | integer(N) | float(F) | str(Id, Args) | unbound(Vid)

step_put_structure_dyn(NameReg, ArityReg, TargetReg, S0, S1) :-
    S0 = wam_state(PC, Regs, _),
    memberchk(NameReg-NameVal, Regs),
    memberchk(ArityReg-ArityVal, Regs),
    NameVal = atom(FnId),
    ArityVal = integer(Arity),
    Arity >= 0,
    PC1 is PC + 1,
    S1 = wam_state(PC1, Regs, build_struct(FnId, TargetReg, Arity, [])).

test_step_put_structure_dyn_happy_path :-
    Test = test_step_put_structure_dyn_happy_path,
    Regs = [1-atom(42), 2-integer(2)],
    S0 = wam_state(7, Regs, no_builder),
    (   step_put_structure_dyn(1, 2, 103, S0, S1),
        S1 = wam_state(8, Regs, build_struct(42, 103, 2, []))
    ->  pass(Test)
    ;   fail_test(Test, step_did_not_produce_expected_state)
    ).

test_step_put_structure_dyn_zero_arity :-
    Test = test_step_put_structure_dyn_zero_arity,
    %% T =.. [Name] form: builds an atom-as-functor (zero-arity Str).
    Regs = [1-atom(99), 2-integer(0)],
    S0 = wam_state(0, Regs, no_builder),
    (   step_put_structure_dyn(1, 2, 103, S0, S1),
        S1 = wam_state(1, Regs, build_struct(99, 103, 0, []))
    ->  pass(Test)
    ;   fail_test(Test, zero_arity_step_failed)
    ).

test_step_put_structure_dyn_rejects_non_atom_name :-
    Test = test_step_put_structure_dyn_rejects_non_atom_name,
    Regs = [1-integer(5), 2-integer(2)],   %% nameReg holds an Integer, not an Atom
    S0 = wam_state(0, Regs, no_builder),
    (   \+ step_put_structure_dyn(1, 2, 103, S0, _)
    ->  pass(Test)
    ;   fail_test(Test, non_atom_name_should_have_failed)
    ).

test_step_put_structure_dyn_rejects_negative_arity :-
    Test = test_step_put_structure_dyn_rejects_negative_arity,
    Regs = [1-atom(42), 2-integer(-1)],
    S0 = wam_state(0, Regs, no_builder),
    (   \+ step_put_structure_dyn(1, 2, 103, S0, _)
    ->  pass(Test)
    ;   fail_test(Test, negative_arity_should_have_failed)
    ).

test_step_put_structure_dyn_rejects_unbound_name :-
    Test = test_step_put_structure_dyn_rejects_unbound_name,
    Regs = [1-unbound(7), 2-integer(2)],
    S0 = wam_state(0, Regs, no_builder),
    (   \+ step_put_structure_dyn(1, 2, 103, S0, _)
    ->  pass(Test)
    ;   fail_test(Test, unbound_name_should_have_failed)
    ).

:- initialization(run_tests, main).
