:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_haskell_lowered_emitter.pl - WAM-lowered Haskell emission (Phase 3+)
%
% Emits one Haskell function per Prolog predicate, mirroring the WAM
% interpreter's step-function semantics inline. The resulting function
% has the shape:
%
%     lowered_<name>_<arity> :: WamContext -> WamState -> Maybe WamState
%
% and is invoked by WamRuntime.step's Call dispatch chain via the
% wcLoweredPredicates lookup (see Phase 2).
%
% Phase 3 ships only the smallest useful whitelist — `get_constant` and
% `proceed` — which covers single-clause facts with integer/atom
% arguments. Phase 4+ expands the whitelist.
%
% See docs/design/WAM_HASKELL_LOWERED_SPECIFICATION.md §2 for the
% per-instruction → Haskell mapping contract, and
% docs/design/WAM_HASKELL_LOWERED_IMPLEMENTATION_PLAN.md §"Phase 3" for
% the ordering.

:- module(wam_haskell_lowered_emitter, [
    wam_haskell_lowerable/3,       % +PredIndicator, +WamCode, -Reason
    lower_predicate_to_haskell/4   % +PredIndicator, +WamCode, +Options, -Lowered
]).

:- use_module(library(lists)).

%% ---------------------------------------------------------------------
%% Parsing
%% ---------------------------------------------------------------------
%
% The WAM target (wam_target.pl) emits WAM as text. Each predicate's
% WamCode argument is a multi-line atom/string with one line per
% instruction. The first line is always the predicate label
% (`<name>/<arity>:`), followed by indented instruction lines. An
% instruction line splits into a mnemonic (e.g. `get_constant`) and
% argument tokens, possibly with trailing commas.

%% parse_wam_text(+WamText, -Instructions)
%  Parse a WAM text atom/string into a list of instruction terms.
%  Returns a list of compound terms whose functor is the mnemonic and
%  whose arguments are the token strings in order (commas stripped).
%  Skips blank lines and the predicate-label line.
parse_wam_text(WamText, Instructions) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_wam_lines(Lines, Instructions).

parse_wam_lines([], []).
parse_wam_lines([Line|Rest], Instructions) :-
    string_trim(Line, Trimmed),
    (   Trimmed == ""
    ->  parse_wam_lines(Rest, Instructions)
    ;   sub_string(Trimmed, _, 1, 0, ":")   % label line like "foo/1:"
    ->  parse_wam_lines(Rest, Instructions)
    ;   tokenize_instruction(Trimmed, Instr),
        Instructions = [Instr|InstructionsRest],
        parse_wam_lines(Rest, InstructionsRest)
    ).

string_trim(S, T) :-
    split_string(S, "", " \t", [T0]),
    T = T0.

tokenize_instruction(Line, Term) :-
    split_string(Line, " \t", " \t,", Tokens),
    exclude(=(""), Tokens, NonEmpty),
    NonEmpty = [MnemonicStr|Args],
    atom_string(Mnemonic, MnemonicStr),
    Term =.. [Mnemonic|Args].

%% ---------------------------------------------------------------------
%% Lowerability
%% ---------------------------------------------------------------------

%% wam_haskell_lowerable(+PredIndicator, +WamCode, -Reason)
%  True iff PredIndicator's WamCode consists entirely of instructions
%  in the Phase 3 whitelist. On failure, Reason is left unbound — the
%  caller that cares why a predicate could not be lowered should log
%  the Reason via a separate helper.
wam_haskell_lowerable(_PredIndicator, WamCode, _Reason) :-
    parse_wam_text(WamCode, Instructions),
    % Phase 3 whitelist + structural constraint: all instructions must
    % be supported, the sequence must end with proceed, and there must
    % be exactly one proceed (no branching within the predicate body).
    forall(member(I, Instructions), phase3_supported(I)),
    last(Instructions, proceed),
    include(=(proceed), Instructions, Proceeds),
    length(Proceeds, 1).

phase3_supported(get_constant(_, _)).
phase3_supported(proceed).

%% ---------------------------------------------------------------------
%% Emission
%% ---------------------------------------------------------------------

%% lower_predicate_to_haskell(+PredIndicator, +WamCode, +Options, -Lowered)
%  Lower PredIndicator to a Haskell function. Lowered is a term
%      lowered(PredName, FuncName, HaskellCode)
%  where PredName is the Prolog-style "<name>/<arity>" key used by
%  wcLoweredPredicates dispatch, FuncName is the Haskell identifier of
%  the generated function, and HaskellCode is the full source text of
%  that function (ready to splice into Lowered.hs).
lower_predicate_to_haskell(PredIndicator, WamCode, _Options,
                           lowered(PredName, FuncName, HaskellCode)) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    format(atom(FuncName), 'lowered_~w_~w', [Pred, Arity]),
    parse_wam_text(WamCode, Instructions),
    emit_function(FuncName, Instructions, HaskellCode).

%% emit_function(+FuncName, +Instructions, -HaskellCode)
%  Emit a complete Haskell function definition for the whitelisted
%  instruction sequence. Phase 3 only supports a run of GetConstants
%  followed by a single Proceed.
emit_function(FuncName, Instructions, HaskellCode) :-
    append(GetConstants, [proceed], Instructions),
    forall(member(GC, GetConstants), GC = get_constant(_, _)),
    with_output_to(string(HaskellCode), (
        format("-- | Lowered form (Phase 3: get_constant+proceed)~n"),
        format("~w :: WamContext -> WamState -> Maybe WamState~n", [FuncName]),
        format("~w !_ctx s0 =~n", [FuncName]),
        emit_chain(GetConstants, 0)
    )).

%% emit_chain(+GetConstantList, +Depth)
%  Print the body chain directly to current output. Each recursive step
%  handles one GetConstant at state variable s<Depth>, threading into
%  s<Depth+1>. When the list is empty, prints the inlined Proceed tail
%  at the current state variable.
emit_chain([], Depth) :-
    format(atom(SV), 's~w', [Depth]),
    emit_proceed_tail(SV).
emit_chain([get_constant(CStr, RegStr)|Rest], Depth) :-
    value_to_haskell(CStr, HsConst),
    reg_to_int(RegStr, RegId),
    NextDepth is Depth + 1,
    format(atom(SV),   's~w',   [Depth]),
    format(atom(SVn),  's~w',   [NextDepth]),
    format(atom(VIDn), 'vid~w', [Depth]),
    indent_for(Depth, Pad),
    format("~w  case fmap (derefVar (wsBindings ~w)) (IM.lookup ~w (wsRegs ~w)) of~n",
           [Pad, SV, RegId, SV]),
    format("~w    Just v | v == (~w) ->~n", [Pad, HsConst]),
    format("~w      let ~w = ~w in~n",      [Pad, SVn, SV]),
    emit_chain(Rest, NextDepth),
    format("~w    Just (Unbound ~w) ->~n",  [Pad, VIDn]),
    format("~w      let ~w = ~w { wsRegs = IM.insert ~w (~w) (wsRegs ~w)~n",
           [Pad, SVn, SV, RegId, HsConst, SV]),
    format("~w                  , wsBindings = IM.insert ~w (~w) (wsBindings ~w)~n",
           [Pad, VIDn, HsConst, SV]),
    format("~w                  , wsTrail = TrailEntry ~w (IM.lookup ~w (wsBindings ~w)) : wsTrail ~w~n",
           [Pad, VIDn, VIDn, SV, SV]),
    format("~w                  , wsTrailLen = wsTrailLen ~w + 1~n", [Pad, SV]),
    format("~w                  }~n", [Pad]),
    format("~w      in~n", [Pad]),
    emit_chain(Rest, NextDepth),
    format("~w    _ -> Nothing~n", [Pad]).

%% emit_proceed_tail(+StateVar)
%  Emit the inline Proceed logic as the innermost continuation of a
%  lowered predicate at the given state variable.
emit_proceed_tail(StateVar) :-
    format("      let retAddr = wsCP ~w~n", [StateVar]),
    format("      in if retAddr == 0~n"),
    format("         then Just (~w { wsPC = 0 })~n", [StateVar]),
    format("         else Just (~w { wsPC = retAddr, wsCP = 0 })~n", [StateVar]).

%% indent_for(+Depth, -Indent)
%  Produce an indentation string for a given nesting depth.
indent_for(0, '').
indent_for(N, Indent) :-
    N > 0,
    N1 is N - 1,
    indent_for(N1, Prev),
    atom_concat(Prev, '      ', Indent).

%% ---------------------------------------------------------------------
%% Value and register translation (local to the emitter; intentionally
%% duplicates the equivalent helpers in wam_haskell_target.pl so the
%% emitter module has no circular import into its caller).
%% ---------------------------------------------------------------------

value_to_haskell(Str, Hs) :-
    (   number_string(N, Str), integer(N)
    ->  format(string(Hs), 'Integer ~w', [N])
    ;   number_string(F, Str), float(F)
    ->  format(string(Hs), 'Float ~w', [F])
    ;   format(string(Hs), 'Atom "~w"', [Str])
    ).

reg_to_int(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, Bank),
    sub_atom(RegA, 1, _, 0, NumA),
    atom_number(NumA, Num),
    (   Bank == 'A' -> Int = Num
    ;   Bank == 'X' -> Int is Num + 100
    ;   Bank == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).
