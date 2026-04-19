:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_elixir_lowered_emitter.pl - WAM-to-Elixir Lowered Emitter
%
% Compiles WAM instructions directly to Elixir function calls/expressions
% instead of instruction-array interpretation.

:- module(wam_elixir_lowered_emitter, [
    lower_predicate_to_elixir/4,  % +PredIndicator, +WamCode, +Options, -Code
    wam_elixir_lower_instr/5      % +Instr, +PC, +Labels, +FuncName, -Code
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(pairs), [group_pairs_by_key/2]).
:- use_module('wam_elixir_utils', [reg_id/2, is_label_part/1, camel_case/2, parse_arity/2]).

% ============================================================================
% MAIN ENTRY POINT
% ============================================================================

%% lower_predicate_to_elixir(+PredIndicator, +WamCode, +Options, -Code)
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
      :fail -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    # Rewrite each caller-supplied unbound to a fresh make_ref so the id
    # cannot collide with any register slot — subsequent put_variable on
    # the same A-reg would otherwise corrupt the binding chain. Remember
    # the refs in state.arg_vars so run/1 and next_solution/1 can read
    # the correct bound value even after body code has used the A-reg as
    # scratch.
    {state, arg_vars} = Enum.with_index(args, 1)
    |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
      case arg do
        {:unbound, _} ->
          v = {:unbound, make_ref()}
          {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
        other ->
          {%{s | regs: Map.put(s.regs, i, other)}, vars}
      end
    end)
    state = %{state | arg_vars: arg_vars}
    case run(state) do
      {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
      other -> other
    end
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
    classify_segment_head(Instrs, HeadType, BodyInstrs),
    lower_instr_list(BodyInstrs, Labels, FuncName, BodyExprs),
    atomic_list_concat(BodyExprs, '\n', BodyCode),
    wrap_segment(FuncName, HeadType, BodyCode, Code),
    generate_all_segments(Rest, Labels, RestCodes).

%% split_last_colon(+Entry, -Key, -Label)
%  Splits an "apple:L1" entry into Key="apple" and Label="L1".
%  For entries with multiple colons, splits at the LAST colon.
%  For entries with no colon, Label defaults to "default".
split_last_colon(Entry, Key, Label) :-
    split_string(Entry, ":", "", Parts),
    (   Parts = [_, _|_]
    ->  append(KeyParts, [LabelStr], Parts),
        atomic_list_concat(KeyParts, ':', KeyAtom),
        atom_string(KeyAtom, Key),
        Label = LabelStr
    ;   Key = Entry, Label = "default"
    ).

%% build_switch_arms(+Entries, -ArmsStr)
%  Builds inline Elixir case arms for switch_on_constant. Groups entries by
%  key first, because the WAM emitter may produce multiple entries for the
%  same first-arg constant (when multiple clauses share a first arg) — and
%  Elixir would then warn that later case-arms are unreachable.
%
%  For a key with a single non-"default" entry: dispatch directly to the
%  labeled clause (the switch's whole reason for existing — skip the
%  try_me_else chain). For any key with multiple entries OR a single
%  "default" entry: fall through to :ok, letting the surrounding
%  try_me_else / retry_me_else chain handle the choice non-deterministically.
build_switch_arms(Entries, ArmsStr) :-
    maplist([E,K-L]>>split_last_colon(E, K, L), Entries, Pairs),
    keysort(Pairs, SortedPairs),
    group_pairs_by_key(SortedPairs, Groups),
    maplist(build_switch_arm_group, Groups, Arms),
    atomic_list_concat(Arms, '\n          ', ArmsStr).

build_switch_arm_group(Key-Labels, Arm) :-
    (   Labels = [OnlyLabel],
        OnlyLabel \== "default"
    ->  segment_func_name(OnlyLabel, LocalFunc),
        format(string(Arm), '"~w" -> throw({:return, ~w(state)})', [Key, LocalFunc])
    ;   format(string(Arm), '"~w" -> :ok', [Key])
    ).

segment_func_name("clause_start", "clause_main") :- !.
segment_func_name(Label, Name) :- 
    camel_case(Label, Camel),
    format(string(Name), "clause_~w", [Camel]).

classify_segment_head(Instrs, HeadType, BodyInstrs) :-
    (   select(_PC-try_me_else(L), Instrs, BodyInstrs) -> HeadType = try_me_else(L)
    ;   select(_PC-retry_me_else(L), Instrs, BodyInstrs) -> HeadType = retry_me_else(L)
    ;   select(_PC-trust_me, Instrs, BodyInstrs) -> HeadType = trust_me
    ;   HeadType = none, BodyInstrs = Instrs
    ), !.

wrap_segment(FuncName, try_me_else(L), BodyCode, Code) :-
    segment_func_name(L, FallbackFunc),
    format(string(Code),
'  defp ~w(state) do
    cp = %{pc: &~w/1, regs: state.regs, heap: state.heap, heap_len: state.heap_len,
           cp: state.cp, trail: state.trail, trail_len: state.trail_len, stack: state.stack}
    state = %{state | choice_points: [cp | state.choice_points]}
~w
  end', [FuncName, FallbackFunc, BodyCode]).

wrap_segment(FuncName, retry_me_else(L), BodyCode, Code) :-
    segment_func_name(L, FallbackFunc),
    format(string(Code),
'  defp ~w(state) do
    case state.choice_points do
      [_old | rest] ->
        cp = %{pc: &~w/1, regs: state.regs, heap: state.heap, heap_len: state.heap_len,
               cp: state.cp, trail: state.trail, trail_len: state.trail_len, stack: state.stack}
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
'    addr = state.heap_len
    new_heap = Map.put(state.heap, addr, {:str, "~w"})
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
    |> Map.put(:heap, new_heap)
    |> Map.put(:heap_len, addr + 1)', [F, Ai, Ai]).

wam_elixir_lower_instr(get_structure(F, AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(AiName, Ai),
    parse_arity(F, Arity),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = cond do
      match?({:unbound, _}, val) ->
        addr = state.heap_len
        new_heap = Map.put(state.heap, addr, {:str, "~w"})
        state
        |> WamRuntime.trail_binding(~w)
        |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:heap_len, addr + 1)
        |> Map.put(:stack, [{:write_ctx, ~w} | state.stack])
      match?({:ref, _}, val) ->
        {:ref, addr} = val
        case WamRuntime.step_get_structure_ref(state, "~w", ~w, addr) do
          :fail -> throw(:fail)
          s -> s
        end
      true -> throw(:fail)
    end', [Ai, F, Ai, Ai, Arity, F, Arity]).

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
    Code = '    # Save caller\'s Y-regs so the callee can freely overwrite
    # slots 201-299 without corrupting the outer frame. Env popped by
    # deallocate.
    {y_regs_saved, base_regs} = WamRuntime.split_y_regs(state.regs)
    new_env = %{cp: state.cp, y_regs_saved: y_regs_saved}
    state = %{state | stack: [new_env | state.stack], regs: base_regs}'.

wam_elixir_lower_instr(deallocate, _PC, _Labels, _FuncName, Code) :-
    Code = '    state = case state.stack do
      [env | rest] ->
        {_callee_ys, base_regs} = WamRuntime.split_y_regs(state.regs)
        merged = Map.merge(base_regs, Map.get(env, :y_regs_saved, %{}))
        %{state | cp: env.cp, stack: rest, regs: merged}
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
'    addr = state.heap_len
    new_heap = Map.put(state.heap, addr, {:str, "./2"})
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
    |> Map.put(:heap, new_heap)
    |> Map.put(:heap_len, addr + 1)', [Ai, Ai]).

wam_elixir_lower_instr(get_list(AiName), _PC, _Labels, _FuncName, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = cond do
      match?({:unbound, _}, val) ->
        addr = state.heap_len
        new_heap = Map.put(state.heap, addr, {:str, "./2"})
        state
        |> WamRuntime.trail_binding(~w)
        |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:heap_len, addr + 1)
        |> Map.put(:stack, [{:write_ctx, 2} | state.stack])
      match?({:ref, _}, val) ->
        {:ref, addr} = val
        case WamRuntime.step_get_structure_ref(state, "./2", 2, addr) do
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
'    addr = state.heap_len
    fresh = {:unbound, {:heap_ref, addr}}
    new_heap = Map.put(state.heap, addr, fresh)
    state = state
    |> WamRuntime.put_reg(~w, fresh)
    |> Map.put(:heap, new_heap)
    |> Map.put(:heap_len, addr + 1)', [Xn]).

wam_elixir_lower_instr(set_value(XnName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    val = WamRuntime.get_reg(state, ~w)
    addr = state.heap_len
    state = %{state | heap: Map.put(state.heap, addr, val), heap_len: addr + 1}', [Xn]).

wam_elixir_lower_instr(set_constant(C), _PC, _Labels, _FuncName, Code) :-
    format(string(Code),
'    addr = state.heap_len
    state = %{state | heap: Map.put(state.heap, addr, "~w"), heap_len: addr + 1}', [C]).

wam_elixir_lower_instr(switch_on_constant(Entries), _PC, _Labels, _FuncName, Code) :-
    build_switch_arms(Entries, ArmsStr),
    format(string(Code),
'    val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    case val do
      {:unbound, _} -> :ok
      _ ->
        case val do
          ~w
          _ -> throw(:fail)
        end
    end', [ArmsStr]).

wam_elixir_lower_instr(switch_on_constant_a2(Entries), _PC, _Labels, _FuncName, Code) :-
    build_switch_arms(Entries, ArmsStr),
    format(string(Code),
'    val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 2))
    case val do
      {:unbound, _} -> :ok
      _ ->
        case val do
          ~w
          _ -> throw(:fail)
        end
    end', [ArmsStr]).

wam_elixir_lower_instr(proceed, _PC, _Labels, _FuncName, Code) :-
    Code = '    {:ok, %{state | pc: state.cp}}'.

wam_elixir_lower_instr(raw(Combined), _PC, _Labels, _FuncName, Code) :-
    format(string(Code), '    # raw: ~w\n    raise "TODO: ~w"', [Combined, Combined]).

%% pred_to_module(+PredStr, -ModuleName)
pred_to_module(PredStr, ModName) :-
    (   sub_string(PredStr, Before, _, _, "/")
    ->  sub_string(PredStr, 0, Before, _, Name)
    ;   Name = PredStr
    ),
    camel_case(Name, CamelName),
    format(atom(ModName), 'WamPredLow.~w', [CamelName]).
