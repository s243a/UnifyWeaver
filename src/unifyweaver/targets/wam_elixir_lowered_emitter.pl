:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_elixir_lowered_emitter.pl - WAM-to-Elixir Lowered Emitter
%
% Compiles WAM instructions directly to Elixir function calls/expressions
% instead of instruction-array interpretation.
%
% Architecture: Each clause in a multi-clause predicate becomes a separate
% Elixir function. Choice point instructions (try_me_else, retry_me_else)
% are implemented via try/catch, with catch blocks dispatching to the next
% clause with the original (unmodified) state — matching WAM backtracking
% semantics through Elixir's immutable variable scoping.
%
% Inter-predicate calls use synchronous function dispatch to the target
% predicate's lowered module.

:- module(wam_elixir_lowered_emitter, [
    lower_predicate_to_elixir/3,  % +PredIndicator, +WamCode, -ElixirCode
    wam_elixir_lower_instr/4      % +Instr, +PC, +Labels, -Code
]).

:- use_module(library(lists)).
:- use_module('wam_elixir_utils', [reg_id/2, is_label_part/1]).

% ============================================================================
% MAIN ENTRY POINT
% ============================================================================

%% lower_predicate_to_elixir(+PredIndicator, +WamCode, -ElixirCode)
%  Compiles a WAM predicate into a lowered Elixir module with one function
%  per clause and try/catch for backtracking.
lower_predicate_to_elixir(Pred/Arity, WamCode, Code) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    collect_labels(Lines, 1, Labels),
    split_into_segments(Lines, 1, Segments),
    generate_all_segments(Segments, Labels, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FuncsBody),
    atom_string(Pred, PredStr),
    Segments = [FirstSegName-_|_],
    segment_func_name(FirstSegName, FirstFunc),
    format(string(Code),
'defmodule WamPredLow.~w do
  @moduledoc "Lowered WAM-compiled predicate: ~w/~w"

  def run(state) do
    try do
      ~w(state)
    catch
      :fail -> :fail
      {:return, result} -> result
    end
  end

~w
end', [PredStr, PredStr, Arity, FirstFunc, FuncsBody]).

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
% SEGMENT SPLITTING
% ============================================================================

%% split_into_segments(+Lines, +StartPC, -Segments)
%  Splits WAM lines into segments at label boundaries.
%  Each segment is Name-InstrList where InstrList = [PC-Instr, ...]
split_into_segments(Lines, StartPC, Segments) :-
    split_segs_acc(Lines, StartPC, "entry", [], Segments).

split_segs_acc([], _, Name, Acc, [Name-Rev]) :-
    reverse(Acc, Rev).
split_segs_acc([Line|Rest], PC, Name, Acc, Segs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts = []
    ->  split_segs_acc(Rest, PC, Name, Acc, Segs)
    ;   CleanParts = [First|_], is_label_part(First)
    ->  sub_string(First, 0, _, 1, LabelName),
        (   Acc = []
        ->  % Empty segment — just rename, don't emit an empty segment
            split_segs_acc(Rest, PC, LabelName, [], Segs)
        ;   reverse(Acc, Rev),
            Segs = [Name-Rev | RestSegs],
            split_segs_acc(Rest, PC, LabelName, [], RestSegs)
        )
    ;   instr_from_parts(CleanParts, Instr),
        NPC is PC + 1,
        split_segs_acc(Rest, NPC, Name, [PC-Instr|Acc], Segs)
    ).

% ============================================================================
% SEGMENT CODE GENERATION
% ============================================================================

generate_all_segments([], _, []).
generate_all_segments([Name-Instrs|Rest], Labels, [Code|Codes]) :-
    generate_segment(Name, Instrs, Labels, Code),
    generate_all_segments(Rest, Labels, Codes).

generate_segment(Name, Instrs, Labels, Code) :-
    segment_func_name(Name, FuncName),
    classify_segment_head(Instrs, HeadType, BodyInstrs),
    lower_instr_list(BodyInstrs, Labels, BodyExprs),
    atomic_list_concat(BodyExprs, '\n', BodyCode),
    wrap_segment(FuncName, HeadType, BodyCode, Code).

%% classify_segment_head(+Instrs, -HeadType, -BodyInstrs)
%  Identifies and strips choice point instructions from segment head.
classify_segment_head(Instrs, HeadType, BodyInstrs) :-
    (   select(_PC-try_me_else(L), Instrs, BodyInstrs1) -> HeadType = try_me_else(L), BodyInstrs = BodyInstrs1
    ;   select(_PC-retry_me_else(L), Instrs, BodyInstrs1) -> HeadType = retry_me_else(L), BodyInstrs = BodyInstrs1
    ;   select(_PC-trust_me, Instrs, BodyInstrs1) -> HeadType = trust_me, BodyInstrs = BodyInstrs1
    ;   HeadType = none, BodyInstrs = Instrs
    ), !.

%% wrap_segment(+FuncName, +HeadType, +BodyCode, -Code)
%  Wraps segment body in appropriate control structure.
wrap_segment(FuncName, try_me_else(L), BodyCode, Code) :-
    segment_func_name(L, FallbackFunc),
    format(string(Code),
'  defp ~w(state) do
    try do
~w
    catch
      :fail -> ~w(state)
      {:return, result} -> throw({:return, result})
    end
  end', [FuncName, BodyCode, FallbackFunc]).

wrap_segment(FuncName, retry_me_else(L), BodyCode, Code) :-
    segment_func_name(L, FallbackFunc),
    format(string(Code),
'  defp ~w(state) do
    try do
~w
    catch
      :fail -> ~w(state)
      {:return, result} -> throw({:return, result})
    end
  end', [FuncName, BodyCode, FallbackFunc]).

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

%% lower_instr_list(+PCInstrs, +Labels, -Exprs)
lower_instr_list([], _, []).
lower_instr_list([PC-Instr|Rest], Labels, [Expr|Exprs]) :-
    wam_elixir_lower_instr(Instr, PC, Labels, Expr),
    lower_instr_list(Rest, Labels, Exprs).

% ============================================================================
% NAMING HELPERS
% ============================================================================

segment_func_name("entry", run_entry) :- !.
segment_func_name(Name, FuncName) :-
    split_string(Name, "/", "", Parts),
    atomic_list_concat(Parts, '_', Sanitized),
    atom_concat(clause_, Sanitized, FuncName).

%% pred_to_module(+PredStr, -ModuleName)
%  Converts "pred_name/arity" to WamPredLow.pred_name
pred_to_module(PredStr, ModName) :-
    (   sub_string(PredStr, Before, _, _, "/")
    ->  sub_string(PredStr, 0, Before, _, Name)
    ;   Name = PredStr
    ),
    atom_string(NameAtom, Name),
    format(atom(ModName), 'WamPredLow.~w', [NameAtom]).

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
instr_from_parts(["allocate", _], allocate).
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

%% wam_elixir_lower_instr(+Instr, +PC, +Labels, -Code)

% --- Head Unification ---

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

% --- Term Construction / Deconstruction ---

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

% --- Control Flow ---

wam_elixir_lower_instr(proceed, _PC, _Labels, Code) :-
    Code = '    {:ok, state}'.

wam_elixir_lower_instr(call(P, _N), _PC, _Labels, Code) :-
    format(string(Code),
'    state = case WamDispatcher.call("~w", state) do
      {:ok, s} -> s
      :fail -> throw(:fail)
    end', [P]).

wam_elixir_lower_instr(execute(P), _PC, _Labels, Code) :-
    format(string(Code),
'    case WamDispatcher.call("~w", state) do
      {:ok, _} = result -> result
      :fail -> throw(:fail)
    end', [P]).

wam_elixir_lower_instr(allocate, _PC, _Labels, Code) :-
    Code = '    new_env = %{cp: state.cp, regs: %{}}
    state = %{state | stack: [new_env | state.stack]}'.

wam_elixir_lower_instr(deallocate, _PC, _Labels, Code) :-
    Code = '    state = case state.stack do
      [env | rest] -> %{state | cp: env.cp, stack: rest}
      _ -> state
    end'.

wam_elixir_lower_instr(builtin_call(Op, Ar), _PC, _Labels, Code) :-
    format(string(Code),
'    state = case WamRuntime.execute_builtin(state, "~w", ~w) do
      :fail -> throw(:fail)
      s -> s
    end', [Op, Ar]).

% --- Choice Points (handled at segment level, stubs for direct calls) ---

wam_elixir_lower_instr(try_me_else(_), _PC, _Labels,
    '    # try_me_else: handled by clause dispatch').
wam_elixir_lower_instr(retry_me_else(_), _PC, _Labels,
    '    # retry_me_else: handled by clause dispatch').
wam_elixir_lower_instr(trust_me, _PC, _Labels,
    '    # trust_me: handled by clause dispatch').

% --- Remaining stubs ---

wam_elixir_lower_instr(switch_on_constant(Entries), _PC, _Labels, Code) :-
    maplist(elixir_switch_entry, Entries, Cases),
    atomic_list_concat(Cases, '\n', CasesStr),
    format(string(Code),
'    val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    case val do
      {:unbound, _} -> :ok # Variables fall through to choice points
~w
      _ -> throw(:fail) # Unmatched constants fail
    end', [CasesStr]).

wam_elixir_lower_instr(switch_on_constant_a2(Entries), _PC, _Labels, Code) :-
    maplist(elixir_switch_entry, Entries, Cases),
    atomic_list_concat(Cases, '\n', CasesStr),
    format(string(Code),
'    val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 2))
    case val do
      {:unbound, _} -> :ok # Variables fall through to choice points
~w
      _ -> throw(:fail) # Unmatched constants fail
    end', [CasesStr]).

wam_elixir_lower_instr(raw(Combined), _PC, _Labels, Code) :-
    format(string(Code), '    # raw: ~w\n    raise "TODO: ~w"', [Combined, Combined]).

elixir_switch_entry(Entry, CaseCode) :-
    split_string(Entry, ":", "", [Key, Label]),
    (   Label == "default"
    ->  format(string(CaseCode), '      "~w" -> :ok # default label: fall through to choice point', [Key])
    ;   segment_func_name(Label, FuncName),
        % Note: The {:return, result} throw is explicitly caught and re-thrown by 
        % the try/catch wrappers in wrap_segment/4 to safely bubble up.
        format(string(CaseCode), '      "~w" -> throw({:return, ~w(state)})', [Key, FuncName])
    ).
