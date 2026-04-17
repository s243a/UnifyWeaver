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
    lower_predicate_to_elixir/4,  % +PredIndicator, +WamCode, +Options, -Code
    wam_elixir_lower_instr/5      % +Instr, +PC, +Labels, +FuncName, -Code
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('wam_elixir_utils', [reg_id/2, is_label_part/1, camel_case/2]).

% ============================================================================
% MAIN ENTRY POINT
% ============================================================================

%% lower_predicate_to_elixir(+PredIndicator, +WamCode, +Options, -Code)
%  Compiles a WAM predicate into a lowered Elixir module with one function
%  per clause and try/catch for backtracking.
lower_predicate_to_elixir(Pred/Arity, WamCode, Options, Code) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    collect_labels(Lines, 1, Labels),
    split_into_segments(Lines, 1, Segments),
    generate_all_segments(Segments, Labels, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FuncsBody),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    option(module_name(ModName), Options, 'WamPredLow'),
    camel_case(ModName, CamelMod),
    Segments = [FirstSegName-_|_],
    segment_func_name(FirstSegName, FirstFunc),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc "Lowered WAM-compiled predicate: ~w/~w"

  def run(%WamRuntime.WamState{} = state) do
    try do
      ~w(state)
    catch
      :fail -> 
        # IO.puts("Predicate failed")
        :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: [], labels: %{}, pc: 1}
    state = Enum.with_index(args, 1)
    |> Enum.reduce(state, fn {arg, i}, s ->
      %{s | regs: Map.put(s.regs, i, arg)}
    end)
    run(state)
  end

~w
end', [CamelMod, CamelPred, PredStr, Arity, FirstFunc, CamelPred, FuncsBody]).

% ============================================================================
% LABEL COLLECTION
% ============================================================================

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

% ============================================================================
% SEGMENTATION & CODE GENERATION
% ============================================================================

%% split_into_segments(+Lines, +PC, -Segments)
%  Partitions WAM instructions into segments starting at labels.
split_into_segments([], _, []).
split_into_segments([Line|Rest], PC, Segments) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == [] -> split_into_segments(Rest, PC, Segments)
    ;   CleanParts = [First|_], is_label_part(First)
    ->  sub_string(First, 0, _, 1, LabelName),
        extract_segment_body(Rest, PC, BodyLines, NextLines, NPC),
        Segments = [LabelName-BodyLines | RestSegments],
        split_into_segments(NextLines, NPC, RestSegments)
    ;   extract_segment_body([Line|Rest], PC, BodyLines, NextLines, NPC),
        Segments = ["clause_start"-BodyLines | RestSegments],
        split_into_segments(NextLines, NPC, RestSegments)
    ).

extract_segment_body([], PC, [], [], PC).
extract_segment_body([Line|Rest], PC, Body, Next, NPC) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts = [First|_], is_label_part(First)
    ->  Body = [], Next = [Line|Rest], NPC = PC
    ;   CleanParts == [] -> extract_segment_body(Rest, PC, Body, Next, NPC)
    ;   Body = [PC-Instr | RestBody],
        instr_from_parts(CleanParts, Instr),
        PC1 is PC + 1,
        extract_segment_body(Rest, PC1, RestBody, Next, NPC)
    ).

generate_all_segments([], _, []).
generate_all_segments([Name-Instrs | Rest], Labels, [Code | RestCodes]) :-
    segment_func_name(Name, FuncName),
    classify_segment_head(Instrs, HeadType, BodyInstrs),  % strip CP ops
    lower_instr_list(BodyInstrs, Labels, FuncName, BodyExprs), % use stripped list
    atomic_list_concat(BodyExprs, '\n', BodyCode),
    % Pre-extract any switch maps to module attributes
    extract_switch_maps(BodyInstrs, FuncName, MapAttrBody),
    wrap_segment(FuncName, HeadType, BodyCode, SegmentCode),
    atomic_list_concat([MapAttrBody, SegmentCode], Code),
    generate_all_segments(Rest, Labels, RestCodes).

extract_switch_maps(Instrs, FuncName, AttrBody) :-
    (   (member(_PC-switch_on_constant(Entries), Instrs) ; member(_PC-switch_on_constant_a2(Entries), Instrs))
    ->  maplist(switch_entry_pair, Entries, Pairs),
        atomic_list_concat(Pairs, ', ', PairsStr),
        format(string(AttrBody), '  @switch_map_~w Map.new([~w])\n', [FuncName, PairsStr])
    ;   AttrBody = ""
    ).

switch_entry_pair(Entry, Pair) :-
    % entry is "key:label"
    atom_string(E, Entry),
    (   sub_atom(E, Before, 1, After, ':'), \+ sub_atom(E, _, 1, After, ':')
    ->  sub_atom(E, 0, Before, _, Key),
        sub_atom(E, _, After, 0, Label)
    ;   Key = Entry, Label = "default"
    ),
    format(string(Pair), '{"~w", "~w"}', [Key, Label]).

segment_func_name("clause_start", "clause_main") :- !.
segment_func_name(Label, Name) :- 
    camel_case(Label, Camel),
    format(string(Name), "clause_~w", [Camel]).

%% classify_segment_head(+Instrs, -HeadType, -BodyInstrs)
%  Identifies and strips choice point instructions from segment head.
classify_segment_head(Instrs, HeadType, BodyInstrs) :-
    (   select(_PC-try_me_else(L), Instrs, BodyInstrs) -> HeadType = try_me_else(L)
    ;   select(_PC-retry_me_else(L), Instrs, BodyInstrs) -> HeadType = retry_me_else(L)
    ;   select(_PC-trust_me, Instrs, BodyInstrs) -> HeadType = trust_me
    ;   HeadType = none, BodyInstrs = Instrs
    ), !.

%% wrap_segment(+FuncName, +HeadType, +BodyCode, -Code)
%  Wraps segment body in appropriate control structure.
wrap_segment(FuncName, try_me_else(L), BodyCode, Code) :-
    segment_func_name(L, FallbackFunc),
    format(string(Code),
'  defp ~w(state) do
    cp = %{pc: &~w/1, regs: state.regs, heap: state.heap,
           cp: state.cp, trail: state.trail, stack: state.stack}
    state = %{state | choice_points: [cp | state.choice_points]}
~w
  end', [FuncName, FallbackFunc, BodyCode]).

wrap_segment(FuncName, retry_me_else(L), BodyCode, Code) :-
    segment_func_name(L, FallbackFunc),
    format(string(Code),
'  defp ~w(state) do
    case state.choice_points do
      [_old | rest] ->
        cp = %{pc: &~w/1, regs: state.regs, heap: state.heap,
               cp: state.cp, trail: state.trail, stack: state.stack}
        state = %{state | choice_points: [cp | rest]}
~w
      _ -> throw(:fail)
    end
  end', [FuncName, FallbackFunc, BodyCode]).

wrap_segment(FuncName, trust_me, BodyCode, Code) :-
    format(string(Code),
'  defp ~w(state) do
~w
  end', [FuncName, BodyCode]).

wrap_segment(FuncName, none, BodyCode, Code) :-
    format(string(Code),
'  defp ~w(state) do
~w
  end', [FuncName, BodyCode]).

%% lower_instr_list(+PCInstrs, +Labels, +FuncName, -Exprs)
lower_instr_list([], _, _, []).
lower_instr_list([PC-Instr|Rest], Labels, FuncName, [Expr|Exprs]) :-
    wam_elixir_lower_instr(Instr, PC, Labels, FuncName, Expr),
    lower_instr_list(Rest, Labels, FuncName, Exprs).

% ============================================================================
% INSTRUCTION PARSING
% ============================================================================

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
instr_from_parts(["switch_on_constant"|Entries], switch_on_constant(Entries)).
instr_from_parts(["switch_on_constant_a2"|Entries], switch_on_constant_a2(Entries)).
instr_from_parts(["proceed"], proceed).
instr_from_parts(Parts, raw(Combined)) :-
    atomic_list_concat(Parts, ' ', Combined).

% ============================================================================
% INSTRUCTION LOWERING
% ============================================================================

%% wam_elixir_lower_instr(+Instr, +PC, +Labels, +FuncName, -Code)
wam_elixir_lower_instr(get_constant(C, AiName), _PC, _Labels, _FuncName, Code) :-
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

wam_elixir_lower_instr(get_variable(XnName, AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = state |> WamRuntime.trail_binding(~w) |> WamRuntime.put_reg(~w, val)', [Ai, Xn, Xn]).

wam_elixir_lower_instr(get_value(XnName, AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val_a = WamRuntime.deref_var(state, Map.get(state.regs, ~w))
    val_x = WamRuntime.get_reg(state, ~w)
    state = case WamRuntime.unify(state, val_a, val_x) do
      {:ok, s} -> s
      :fail -> throw(:fail)
    end', [Ai, Xn]).

wam_elixir_lower_instr(put_structure(F, AiName), _PC, _Labels, _FuncName, Code) :-
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

wam_elixir_lower_instr(get_structure(F, AiName), _PC, _Labels, _FuncName, Code) :-
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

wam_elixir_lower_instr(unify_variable(XnName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_variable(state, ~w) do
      :fail -> throw(:fail)
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_value(XnName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_value(state, ~w) do
      :fail -> throw(:fail)
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_constant(C), _PC, _Labels, _FuncName, Code) :-
    format(string(Code),
'    state = case WamRuntime.step_unify_constant(state, "~w") do
      :fail -> throw(:fail)
      s -> s
    end', [C]).

wam_elixir_lower_instr(try_me_else(_L), _PC, _Labels, _FuncName, Code) :-
    Code = '    :ok # Handled by wrap_segment'.

wam_elixir_lower_instr(retry_me_else(_L), _PC, _Labels, _FuncName, Code) :-
    Code = '    :ok # Handled by wrap_segment'.

wam_elixir_lower_instr(trust_me, _PC, _Labels, _FuncName, Code) :-
    Code = '    :ok # Handled by wrap_segment'.

wam_elixir_lower_instr(allocate, _PC, _Labels, _FuncName, Code) :-
    Code = '    new_env = %{cp: state.cp, regs: %{}}
    state = %{state | stack: [new_env | state.stack]}'.

wam_elixir_lower_instr(deallocate, _PC, _Labels, _FuncName, Code) :-
    Code = '    state = case state.stack do
      [env | rest] -> %{state | cp: env.cp, stack: rest}
      _ -> state
    end'.

wam_elixir_lower_instr(call(P, _N), PC, _Labels, _FuncName, Code) :-
    NPC is PC + 1,
    format(string(Code),
'    state = case WamDispatcher.call("~w", state) do
      {:ok, s} -> %{s | pc: ~w}
      :fail -> throw(:fail)
    end', [P, NPC]).

wam_elixir_lower_instr(execute(P), _PC, _Labels, _FuncName, Code) :-
    format(string(Code),
'    WamDispatcher.call("~w", state)', [P]).

wam_elixir_lower_instr(builtin_call(Op, Ar), _PC, _Labels, _FuncName, Code) :-
    format(string(Code),
'    state = case WamRuntime.execute_builtin(state, "~w", ~w) do
      :fail -> throw(:fail)
      s -> s
    end', [Op, Ar]).

wam_elixir_lower_instr(put_constant(C, AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    state = %{state | regs: Map.put(state.regs, ~w, "~w")}', [Ai, C]).

wam_elixir_lower_instr(put_variable(XnName, AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    fresh = {:unbound, make_ref()}
    state = state
    |> WamRuntime.trail_binding(~w)
    |> WamRuntime.put_reg(~w, fresh)
    |> WamRuntime.put_reg(~w, fresh)', [Xn, Xn, Ai]).

wam_elixir_lower_instr(put_value(XnName, AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val = WamRuntime.get_reg(state, ~w)
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, val))', [Xn, Ai, Ai]).

wam_elixir_lower_instr(put_list(AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    addr = length(state.heap)
    new_heap = state.heap ++ [{:str, "./2"}]
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
    |> Map.put(:heap, new_heap)
    |> Map.put(:stack, [{:write_ctx, 2} | state.stack])', [Ai, Ai]).

wam_elixir_lower_instr(get_list(AiName), _PC, _Labels, _FuncName, Code) :-
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
        [h | t] = val
        %{state | stack: [{:unify_ctx, [h, t]} | state.stack]}
      true -> throw(:fail)
    end', [Ai, Ai, Ai]).

wam_elixir_lower_instr(set_variable(XnName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    addr = length(state.heap)
    fresh = {:unbound, {:heap_ref, addr}}
    new_heap = state.heap ++ [fresh]
    state = state
    |> WamRuntime.put_reg(~w, fresh)
    |> Map.put(:heap, new_heap)', [Xn]).

wam_elixir_lower_instr(set_value(XnName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    val = WamRuntime.get_reg(state, ~w)
    state = %{state | heap: state.heap ++ [val]}', [Xn]).

wam_elixir_lower_instr(set_constant(C), _PC, _Labels, _FuncName, Code) :-
    format(string(Code), '    state = %{state | heap: state.heap ++ ["~w"]}', [C]).

wam_elixir_lower_instr(switch_on_constant(_Entries), _PC, _Labels, FuncName, Code) :-
    format(string(Code),
'    val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    case val do
      {:unbound, _} -> :ok
      _ ->
        case Map.get(@switch_map_~w, val) do
          nil -> throw(:fail)
          "default" -> :ok
          label -> throw({:return, WamDispatcher.call(label, state)})
        end
    end', [FuncName]).

wam_elixir_lower_instr(switch_on_constant_a2(_Entries), _PC, _Labels, FuncName, Code) :-
    format(string(Code),
'    val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 2))
    case val do
      {:unbound, _} -> :ok
      _ ->
        case Map.get(@switch_map_~w, val) do
          nil -> throw(:fail)
          "default" -> :ok
          label -> throw({:return, WamDispatcher.call(label, state)})
        end
    end', [FuncName]).

wam_elixir_lower_instr(proceed, _PC, _Labels, _FuncName, Code) :-
    Code = '    {:ok, %{state | pc: state.cp}}'.

wam_elixir_lower_instr(raw(Combined), _PC, _Labels, _FuncName, Code) :-
    format(string(Code), '    # raw: ~w\n    raise "TODO: ~w"', [Combined, Combined]).

%% pred_to_module(+PredStr, -ModuleName)
%  Converts "pred_name/arity" to WamPredLow.PredName
pred_to_module(PredStr, ModName) :-
    (   sub_string(PredStr, Before, _, _, "/")
    ->  sub_string(PredStr, 0, Before, _, Name)
    ;   Name = PredStr
    ),
    camel_case(Name, CamelName),
    format(atom(ModName), 'WamPredLow.~w', [CamelName]).
