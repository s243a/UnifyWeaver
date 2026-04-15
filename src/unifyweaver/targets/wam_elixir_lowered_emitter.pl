:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_elixir_lowered_emitter.pl - WAM-to-Elixir Lowered Emitter
%
% Compiles WAM instructions directly to Elixir function calls/expressions
% instead of instruction-array interpretation.
%
% Note: Currently, only a few head unification instructions (e.g., get_constant,
% get_variable, get_value) and `proceed` are fully lowered to establish the
% architecture. The rest of the instructions are explicitly stubbed out with
% `raise "TODO: ..."` to guide future progressive implementation.

:- module(wam_elixir_lowered_emitter, [
    lower_predicate_to_elixir/3,  % +PredIndicator, +WamCode, -ElixirCode
    wam_elixir_lower_instr/3      % +Instr, +PC, -Code
]).

:- use_module(library(lists)).
:- use_module('wam_elixir_utils', [reg_id/2, is_label_part/1]).

%% lower_predicate_to_elixir(+PredIndicator, +WamCode, -ElixirCode)
lower_predicate_to_elixir(Pred/Arity, WamCode, Code) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    lower_lines_to_elixir(Lines, 1, Exprs),
    atomic_list_concat(Exprs, '\n', Body),
    atom_string(Pred, PredStr),
    format(string(Code),
'defmodule WamPredLow.~w do
  @moduledoc "Lowered WAM-compiled predicate: ~w/~w"

  def run(state) do
    # Lowered execution start
~w
  end
end', [PredStr, PredStr, Arity, Body]).

lower_lines_to_elixir([], _, []).
lower_lines_to_elixir([Line|Rest], PC, OutExprs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts = [First|_], \+ is_label_part(First)
    ->  instr_from_parts(CleanParts, Instr),
        wam_elixir_lower_instr(Instr, PC, Expr),
        NPC is PC + 1,
        OutExprs = [Expr|Exprs],
        lower_lines_to_elixir(Rest, NPC, Exprs)
    ;   lower_lines_to_elixir(Rest, PC, OutExprs)
    ).

instr_from_parts(["get_constant", C, Ai], get_constant(C, Ai)).
instr_from_parts(["get_variable", Xn, Ai], get_variable(Xn, Ai)).
instr_from_parts(["get_value", Xn, Ai], get_value(Xn, Ai)).
instr_from_parts(["put_structure", F, Ai], put_structure(F, Ai)).
instr_from_parts(["get_structure", F, Ai], get_structure(F, Ai)).
instr_from_parts(["unify_variable", Xn], unify_variable(Xn)).
instr_from_parts(["unify_value", Xn], unify_value(Xn)).
instr_from_parts(["unify_constant", C], unify_constant(C)).
instr_from_parts(["try_me_else", L], try_me_else(L)).
instr_from_parts(["retry_me_else", L], retry_me_else(L)).
instr_from_parts(["trust_me"], trust_me).
instr_from_parts(["allocate"], allocate).
instr_from_parts(["deallocate"], deallocate).
instr_from_parts(["call", P, N], call(P, N)).
instr_from_parts(["execute", P], execute(P)).
instr_from_parts(["builtin_call", Op, Ar], builtin_call(Op, Ar)).
instr_from_parts(["put_constant", C, Ai], put_constant(C, Ai)).
instr_from_parts(["put_variable", Xn, Ai], put_variable(Xn, Ai)).
instr_from_parts(["put_value", Xn, Ai], put_value(Xn, Ai)).
instr_from_parts(["put_list", Ai], put_list(Ai)).
instr_from_parts(["get_list", Ai], get_list(Ai)).
instr_from_parts(["set_variable", Xn], set_variable(Xn)).
instr_from_parts(["set_value", Xn], set_value(Xn)).
instr_from_parts(["set_constant", C], set_constant(C)).
instr_from_parts(["switch_on_constant"|_], switch_on_constant).
instr_from_parts(["proceed"], proceed).
instr_from_parts(Parts, raw(Combined)) :-
    atomic_list_concat(Parts, ' ', Combined).

%% wam_elixir_lower_instr(+Instr, +PC, -Code)
wam_elixir_lower_instr(get_constant(C, AiName), _PC, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = cond do
      val == "~w" -> state
      match?({:unbound, _}, val) ->
        {:unbound, id} = val
        state |> WamRuntime.trail_binding(id) |> WamRuntime.put_reg(id, "~w")
      true -> throw(:fail)
    end', [Ai, C, C]).

wam_elixir_lower_instr(get_variable(XnName, AiName), _PC, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = state |> WamRuntime.trail_binding(~w) |> WamRuntime.put_reg(~w, val)', [Ai, Xn, Xn]).

wam_elixir_lower_instr(get_value(XnName, AiName), _PC, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val_a = WamRuntime.deref_var(state, Map.get(state.regs, ~w))
    val_x = WamRuntime.get_reg(state, ~w)
    state = case WamRuntime.unify(state, val_a, val_x) do
      {:ok, s} -> s
      :fail -> throw(:fail)
    end', [Ai, Xn]).

wam_elixir_lower_instr(put_structure(F, AiName), _PC, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    raise "TODO: put_structure ~w, ~w"', [F, Ai]).

wam_elixir_lower_instr(get_structure(F, AiName), _PC, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    raise "TODO: get_structure ~w, ~w"', [F, Ai]).

wam_elixir_lower_instr(unify_variable(XnName), _PC, Code) :-
    reg_id(XnName, Xn),
    format(string(Code), '    raise "TODO: unify_variable ~w"', [Xn]).

wam_elixir_lower_instr(unify_value(XnName), _PC, Code) :-
    reg_id(XnName, Xn),
    format(string(Code), '    raise "TODO: unify_value ~w"', [Xn]).

wam_elixir_lower_instr(unify_constant(C), _PC, Code) :-
    format(string(Code), '    raise "TODO: unify_constant ~w"', [C]).

wam_elixir_lower_instr(try_me_else(L), _PC, Code) :-
    format(string(Code), '    raise "TODO: try_me_else ~w"', [L]).

wam_elixir_lower_instr(retry_me_else(L), _PC, Code) :-
    format(string(Code), '    raise "TODO: retry_me_else ~w"', [L]).

wam_elixir_lower_instr(trust_me, _PC, Code) :-
    Code = '    raise "TODO: trust_me"'.

wam_elixir_lower_instr(allocate, _PC, Code) :-
    Code = '    raise "TODO: allocate"'.

wam_elixir_lower_instr(deallocate, _PC, Code) :-
    Code = '    raise "TODO: deallocate"'.

wam_elixir_lower_instr(call(P, N), _PC, Code) :-
    format(string(Code), '    raise "TODO: call ~w, ~w"', [P, N]).

wam_elixir_lower_instr(execute(P), _PC, Code) :-
    format(string(Code), '    raise "TODO: execute ~w"', [P]).

wam_elixir_lower_instr(builtin_call(Op, Ar), _PC, Code) :-
    format(string(Code), '    raise "TODO: builtin_call ~w, ~w"', [Op, Ar]).

wam_elixir_lower_instr(put_constant(C, AiName), _PC, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    raise "TODO: put_constant ~w, ~w"', [C, Ai]).

wam_elixir_lower_instr(put_variable(XnName, AiName), _PC, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code), '    raise "TODO: put_variable ~w, ~w"', [Xn, Ai]).

wam_elixir_lower_instr(put_value(XnName, AiName), _PC, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code), '    raise "TODO: put_value ~w, ~w"', [Xn, Ai]).

wam_elixir_lower_instr(put_list(AiName), _PC, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    raise "TODO: put_list ~w"', [Ai]).

wam_elixir_lower_instr(get_list(AiName), _PC, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    raise "TODO: get_list ~w"', [Ai]).

wam_elixir_lower_instr(set_variable(XnName), _PC, Code) :-
    reg_id(XnName, Xn),
    format(string(Code), '    raise "TODO: set_variable ~w"', [Xn]).

wam_elixir_lower_instr(set_value(XnName), _PC, Code) :-
    reg_id(XnName, Xn),
    format(string(Code), '    raise "TODO: set_value ~w"', [Xn]).

wam_elixir_lower_instr(set_constant(C), _PC, Code) :-
    format(string(Code), '    raise "TODO: set_constant ~w"', [C]).

wam_elixir_lower_instr(switch_on_constant, _PC, Code) :-
    Code = '    raise "TODO: switch_on_constant"'.

wam_elixir_lower_instr(proceed, _PC, Code) :-
    Code = '    {:ok, %{state | pc: state.cp}}'.

wam_elixir_lower_instr(raw(Combined), _PC, Code) :-
    format(string(Code), '    # raw: ~w\n    raise "TODO: ~w"', [Combined, Combined]).
