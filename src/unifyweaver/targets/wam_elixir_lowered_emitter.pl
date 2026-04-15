:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_elixir_lowered_emitter.pl - WAM-to-Elixir Lowered Emitter
%
% Compiles WAM instructions directly to Elixir function calls/expressions
% instead of instruction-array interpretation.
%
% Note: Many instructions are now fully lowered (head unification, term construction,
% control flow, choice points). Some remain as stubs (e.g., switch_on_constant)
% to guide future progressive implementation.

:- module(wam_elixir_lowered_emitter, [
    lower_predicate_to_elixir/3,  % +PredIndicator, +WamCode, -ElixirCode
    wam_elixir_lower_instr/4      % +Instr, +PC, +Labels, -Code
]).

:- use_module(library(lists)).
:- use_module('wam_elixir_utils', [reg_id/2, is_label_part/1]).

%% lower_predicate_to_elixir(+PredIndicator, +WamCode, -ElixirCode)
lower_predicate_to_elixir(Pred/Arity, WamCode, Code) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    % First pass: collect labels
    collect_labels(Lines, 1, Labels),
    % Second pass: generate code
    lower_lines_to_elixir(Lines, 1, Labels, Exprs),
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

collect_labels([], _, []).
collect_labels([Line|Rest], PC, OutLabels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts = [First|_], is_label_part(First)
    ->  sub_string(First, 0, _, 1, LabelName),
        OutLabels = [LabelName-PC|RestLabels],
        collect_labels(Rest, PC, RestLabels)
    ;   CleanParts = []
    ->  collect_labels(Rest, PC, OutLabels)
    ;   NPC is PC + 1,
        collect_labels(Rest, NPC, OutLabels)
    ).

lower_lines_to_elixir([], _, _, []).
lower_lines_to_elixir([Line|Rest], PC, Labels, OutExprs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts = [First|_], \+ is_label_part(First)
    ->  instr_from_parts(CleanParts, Instr),
        wam_elixir_lower_instr(Instr, PC, Labels, Expr),
        NPC is PC + 1,
        OutExprs = [Expr|Exprs],
        lower_lines_to_elixir(Rest, NPC, Labels, Exprs)
    ;   lower_lines_to_elixir(Rest, PC, Labels, OutExprs)
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

%% wam_elixir_lower_instr(+Instr, +PC, +Labels, -Code)
wam_elixir_lower_instr(get_constant(C, AiName), _PC, _Labels, Code) :-
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

wam_elixir_lower_instr(get_variable(XnName, AiName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = state |> WamRuntime.trail_binding(~w) |> WamRuntime.put_reg(~w, val)', [Ai, Xn, Xn]).

wam_elixir_lower_instr(get_value(XnName, AiName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val_a = WamRuntime.deref_var(state, Map.get(state.regs, ~w))
    val_x = WamRuntime.get_reg(state, ~w)
    state = case WamRuntime.unify(state, val_a, val_x) do
      {:ok, s} -> s
      :fail -> throw(:fail)
    end', [Ai, Xn]).

wam_elixir_lower_instr(put_structure(F, AiName), _PC, _Labels, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    addr = length(state.heap)
    new_heap = state.heap ++ [{:str, "~w"}]
    arity = WamRuntime.parse_functor_arity("~w")
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
    |> Map.put(:heap, new_heap)
    |> Map.put(:stack, [{:write_ctx, arity} | state.stack])', [F, F, Ai, Ai]).

wam_elixir_lower_instr(get_structure(F, AiName), _PC, _Labels, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = cond do
      match?({:unbound, _}, val) ->
        addr = length(state.heap)
        new_heap = state.heap ++ [{:str, "~w"}]
        arity = WamRuntime.parse_functor_arity("~w")
        state
        |> WamRuntime.trail_binding(~w)
        |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:stack, [{:write_ctx, arity} | state.stack])
      match?({:ref, _}, val) ->
        {:ref, addr} = val
        case WamRuntime.step_get_structure_ref(state, "~w", addr) do
          :fail -> throw(:fail)
          s -> s
        end
      true -> throw(:fail)
    end', [Ai, F, F, Ai, Ai, F]).

wam_elixir_lower_instr(unify_variable(XnName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_variable(state, ~w) do
      :fail -> throw(:fail)
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_value(XnName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_value(state, ~w) do
      :fail -> throw(:fail)
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_constant(C), _PC, _Labels, Code) :-
    format(string(Code),
'    state = case WamRuntime.step_unify_constant(state, "~w") do
      :fail -> throw(:fail)
      s -> s
    end', [C]).

wam_elixir_lower_instr(try_me_else(L), _PC, Labels, Code) :-
    wam_resolve_label(L, Labels, Target),
    format(string(Code),
'    cp = %{pc: ~w, regs: state.regs, heap: state.heap,
           cp: state.cp, trail: state.trail, stack: state.stack}
    state = %{state | choice_points: [cp | state.choice_points]}', [Target]).

wam_elixir_lower_instr(retry_me_else(L), _PC, Labels, Code) :-
    wam_resolve_label(L, Labels, Target),
    format(string(Code),
'    case state.choice_points do
      [_old | rest] ->
        cp = %{pc: ~w, regs: state.regs, heap: state.heap,
               cp: state.cp, trail: state.trail, stack: state.stack}
        state = %{state | choice_points: [cp | rest]}
      _ -> state
    end', [Target]).

wam_elixir_lower_instr(trust_me, _PC, _Labels, Code) :-
    Code = '    state = case state.choice_points do
      [_ | rest] -> %{state | choice_points: rest}
      _ -> state
    end'.

wam_elixir_lower_instr(allocate, _PC, _Labels, Code) :-
    Code = '    new_env = %{cp: state.cp, regs: %{}}
    state = %{state | stack: [new_env | state.stack]}'.

wam_elixir_lower_instr(deallocate, _PC, _Labels, Code) :-
    Code = '    state = case state.stack do
      [env | rest] -> %{state | cp: env.cp, stack: rest}
      _ -> state
    end'.

wam_elixir_lower_instr(call(P, _N), _PC, _Labels, Code) :-
    format(string(Code),
'    # call ~w
    state = %{state | cp: ~w} # PC incremented by caller in wam_lines_to_elixir
    # In lowered mode, we might need a dispatcher or direct module call.
    # For now, we set PC and return to runtime loop.
    target_pc = Map.get(state.labels, "~w")
    if target_pc == nil, do: throw(:fail)
    {:ok, %{state | pc: target_pc}}', [P, 0, P]). % CP handled differently in lowered

wam_elixir_lower_instr(execute(P), _PC, _Labels, Code) :-
    format(string(Code),
'    # execute ~w
    target_pc = Map.get(state.labels, "~w")
    if target_pc == nil, do: throw(:fail)
    {:ok, %{state | pc: target_pc}}', [P, P]).

wam_elixir_lower_instr(builtin_call(Op, Ar), _PC, _Labels, Code) :-
    format(string(Code),
'    state = case WamRuntime.execute_builtin(state, "~w", ~w) do
      :fail -> throw(:fail)
      s -> s
    end', [Op, Ar]).

wam_elixir_lower_instr(put_constant(C, AiName), _PC, _Labels, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    state = %{state | regs: Map.put(state.regs, ~w, "~w")}', [Ai, C]).

wam_elixir_lower_instr(put_variable(XnName, AiName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    fresh = {:unbound, make_ref()}
    state = state
    |> WamRuntime.trail_binding(~w)
    |> WamRuntime.put_reg(~w, fresh)
    |> WamRuntime.put_reg(~w, fresh)', [Xn, Xn, Ai]).

wam_elixir_lower_instr(put_value(XnName, AiName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val = WamRuntime.get_reg(state, ~w)
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, val))', [Xn, Ai, Ai]).

wam_elixir_lower_instr(put_list(AiName), _PC, _Labels, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    addr = length(state.heap)
    new_heap = state.heap ++ [{:str, "./2"}]
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
    |> Map.put(:heap, new_heap)
    |> Map.put(:stack, [{:write_ctx, 2} | state.stack])', [Ai, Ai]).

wam_elixir_lower_instr(get_list(AiName), _PC, _Labels, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = cond do
      match?({:unbound, _}, val) ->
        addr = length(state.heap)
        new_heap = state.heap ++ [{:str, "./2"}]
        state
        |> WamRuntime.trail_binding(~w)
        |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:stack, [{:write_ctx, 2} | state.stack])
      match?({:ref, _}, val) ->
        {:ref, addr} = val
        case WamRuntime.step_get_structure_ref(state, "./2", addr) do
          :fail -> throw(:fail)
          s -> s
        end
      is_list(val) ->
        %{state | stack: [{:unify_ctx, val} | state.stack]}
      true -> throw(:fail)
    end', [Ai, Ai, Ai]).

wam_elixir_lower_instr(set_variable(XnName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    addr = length(state.heap)
    fresh = {:unbound, {:heap_ref, addr}}
    new_heap = state.heap ++ [fresh]
    state = state
    |> WamRuntime.put_reg(~w, fresh)
    |> Map.put(:heap, new_heap)', [Xn]).

wam_elixir_lower_instr(set_value(XnName), _PC, _Labels, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    val = WamRuntime.get_reg(state, ~w)
    state = %{state | heap: state.heap ++ [val]}', [Xn]).

wam_elixir_lower_instr(set_constant(C), _PC, _Labels, Code) :-
    format(string(Code), '    state = %{state | heap: state.heap ++ ["~w"]}', [C]).

wam_elixir_lower_instr(switch_on_constant, _PC, _Labels, Code) :-
    Code = '    # TODO: switch_on_constant
    raise "TODO: switch_on_constant"'.

wam_elixir_lower_instr(proceed, _PC, _Labels, Code) :-
    Code = '    {:ok, %{state | pc: state.cp}}'.

wam_elixir_lower_instr(raw(Combined), _PC, _Labels, Code) :-
    format(string(Code), '    # raw: ~w\n    raise "TODO: ~w"', [Combined, Combined]).

wam_resolve_label(Label, Labels, Target) :-
    (   member(Label-Target, Labels)
    ->  true
    ;   Target = 0 % Fallback
    ).
