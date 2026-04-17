:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_elixir_target.pl - WAM-to-Elixir Transpilation Target
%
% Transpiles WAM runtime to Elixir source code.
% Phase 2: WAM instructions → case arms (Elixir pattern match)
% Phase 3: Helper functions → Elixir def bodies
%
% WAM state uses Elixir structs with immutable updates:
%   %WamState{pc: 1, cp: :halt, regs: %{}, heap: [], trail: [],
%             choice_points: [], stack: [], code: [], labels: %{}}

:- module(wam_elixir_target, [
    compile_step_wam_to_elixir/2,       % +Options, -Code
    compile_wam_helpers_to_elixir/2,     % +Options, -Code
    compile_wam_runtime_to_elixir/2,     % +Options, -Code
    compile_wam_predicate_to_elixir/4,   % +Pred/Arity, +WamCode, +Options, -Code
    write_wam_elixir_project/3,          % +Predicates, +Options, +ProjectDir
    wam_elixir_case/2,                   % +InstrName, -ElixirCode
    wam_elixir_resolve_emit_mode/2       % +Options, -Mode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../bindings/elixir_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('wam_elixir_lowered_emitter', [lower_predicate_to_elixir/4]).
:- use_module('wam_elixir_utils', [reg_id/2, clean_comma/2, is_label_part/1, camel_case/2]).

:- discontiguous wam_elixir_case/2.

%% wam_elixir_resolve_emit_mode(+Options, -Mode)
wam_elixir_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M), Options)
    ->  Mode = M
    ;   Mode = interpreter
    ).

% ============================================================================
% PHASE 2: WAM Instructions → Elixir case arms
% ============================================================================

%% compile_step_wam_to_elixir(+Options, -Code)
%  Generates the step/2 function with a case expression on instruction tuples.
compile_step_wam_to_elixir(_Options, Code) :-
    findall(Arm, compile_elixir_step_arm(Arm), Arms),
    atomic_list_concat(Arms, '\n\n', ArmsCode),
    format(string(Code),
'  @doc "Execute a single WAM instruction, returning updated state or :fail"
  def step(state, instr) do
    case instr do
~w

      _ -> :fail
    end
  end', [ArmsCode]).

compile_elixir_step_arm(ArmCode) :-
    wam_elixir_case(InstrName, BodyCode),
    format(string(ArmCode), '      # ~w\n~w', [InstrName, BodyCode]).

% --- Head Unification Instructions ---

wam_elixir_case(get_constant,
'      {:get_constant, c, ai} ->
        val = Map.get(state.regs, ai)
        cond do
          val == c ->
            %{state | pc: state.pc + 1}
          match?({:unbound, _}, val) ->
            state
            |> trail_binding(ai)
            |> put_reg(ai, c)
            |> Map.put(:pc, state.pc + 1)
          true -> :fail
        end').

wam_elixir_case(get_variable,
'      {:get_variable, xn, ai} ->
        val = Map.get(state.regs, ai)
        state
        |> trail_binding(xn)
        |> put_reg(xn, val)
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(get_value,
'      {:get_value, xn, ai} ->
        val_a = deref_var(state, Map.get(state.regs, ai))
        val_x = get_reg(state, xn)
        case unify(state, val_a, val_x) do
          {:ok, new_state} -> %{new_state | pc: new_state.pc + 1}
          :fail -> :fail
        end').

wam_elixir_case(get_structure,
'      {:get_structure, fn_name, ai} ->
        val = Map.get(state.regs, ai)
        cond do
          match?({:unbound, _}, val) ->
            addr = length(state.heap)
            new_heap = state.heap ++ [{:str, fn_name}]
            arity = parse_functor_arity(fn_name)
            state
            |> trail_binding(ai)
            |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
            |> Map.put(:heap, new_heap)
            |> Map.put(:stack, [{:write_ctx, arity} | state.stack])
            |> Map.put(:pc, state.pc + 1)
          match?({:ref, _}, val) ->
            {:ref, addr} = val
            step_get_structure_ref(state, fn_name, addr)
          true -> :fail
        end').

wam_elixir_case(get_list,
'      {:get_list, ai} ->
        val = Map.get(state.regs, ai)
        cond do
          match?({:unbound, _}, val) ->
            addr = length(state.heap)
            new_heap = state.heap ++ [{:str, "./2"}]
            state
            |> trail_binding(ai)
            |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
            |> Map.put(:heap, new_heap)
            |> Map.put(:stack, [{:write_ctx, 2} | state.stack])
            |> Map.put(:pc, state.pc + 1)
          match?({:ref, _}, val) ->
            {:ref, addr} = val
            step_get_structure_ref(state, "./2", addr)
          is_list(val) ->
            %{state | stack: [{:unify_ctx, val} | state.stack], pc: state.pc + 1}
          true -> :fail
        end').

% --- Body Construction Instructions ---

wam_elixir_case(put_constant,
'      {:put_constant, c, ai} ->
        %{state | regs: Map.put(state.regs, ai, c), pc: state.pc + 1}').

wam_elixir_case(put_variable,
'      {:put_variable, xn, ai} ->
        fresh = {:unbound, make_ref()}
        state
        |> trail_binding(xn)
        |> put_reg(xn, fresh)
        |> put_reg(ai, fresh)
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(put_value,
'      {:put_value, xn, ai} ->
        val = get_reg(state, xn)
        state
        |> trail_binding(ai)
        |> Map.put(:regs, Map.put(state.regs, ai, val))
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(put_structure,
'      {:put_structure, fn_name, ai} ->
        addr = length(state.heap)
        new_heap = state.heap ++ [{:str, fn_name}]
        state
        |> trail_binding(ai)
        |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:stack, [{:write_ctx, parse_functor_arity(fn_name)} | state.stack])
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(put_list,
'      {:put_list, ai} ->
        addr = length(state.heap)
        new_heap = state.heap ++ [{:str, "./2"}]
        state
        |> trail_binding(ai)
        |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:stack, [{:write_ctx, 2} | state.stack])
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(set_variable,
'      {:set_variable, xn} ->
        addr = length(state.heap)
        fresh = {:unbound, {:heap_ref, addr}}
        new_heap = state.heap ++ [fresh]
        state
        |> put_reg(xn, fresh)
        |> Map.put(:heap, new_heap)
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(set_value,
'      {:set_value, xn} ->
        val = get_reg(state, xn)
        %{state | heap: state.heap ++ [val], pc: state.pc + 1}').

wam_elixir_case(set_constant,
'      {:set_constant, c} ->
        %{state | heap: state.heap ++ [c], pc: state.pc + 1}').

% --- Unification Instructions ---

wam_elixir_case(unify_variable,
'      {:unify_variable, xn} ->
        step_unify_variable(state, xn)').

wam_elixir_case(unify_value,
'      {:unify_value, xn} ->
        step_unify_value(state, xn)').

wam_elixir_case(unify_constant,
'      {:unify_constant, c} ->
        step_unify_constant(state, c)').

% --- Control Flow Instructions ---

wam_elixir_case(call,
'      {:call, p, _n} ->
        # Note: Interpreter mode currently only supports intra-module calls via labels.
        # Dynamic cross-module dispatch requires the lowered emitter and WamDispatcher.
        new_cp = state.pc + 1
        case Map.get(state.labels, p) do
          nil -> :fail
          target_pc -> %{state | pc: target_pc, cp: new_cp}
        end').

wam_elixir_case(execute,
'      {:execute, p} ->
        # Note: Interpreter mode currently only supports intra-module calls via labels.
        case Map.get(state.labels, p) do
          nil -> :fail
          target_pc -> %{state | pc: target_pc}
        end').

wam_elixir_case(proceed,
'      :proceed ->
        %{state | pc: state.cp}').

wam_elixir_case(allocate,
'      {:allocate, _n} ->
        new_env = %{cp: state.cp, regs: %{}}
        %{state | stack: [new_env | state.stack], pc: state.pc + 1}').

wam_elixir_case(deallocate,
'      :deallocate ->
        case state.stack do
          [env | rest] -> %{state | cp: env.cp, stack: rest, pc: state.pc + 1}
          _ -> %{state | pc: state.pc + 1}
        end').

% --- Choice Point Instructions ---

wam_elixir_case(try_me_else,
'      {:try_me_else, label} ->
        target = resolve_label(state, label)
        cp = %{pc: target, regs: state.regs, heap: state.heap,
               cp: state.cp, trail: state.trail, stack: state.stack}
        %{state | choice_points: [cp | state.choice_points], pc: state.pc + 1}').

wam_elixir_case(retry_me_else,
'      {:retry_me_else, label} ->
        target = resolve_label(state, label)
        case state.choice_points do
          [_old | rest] ->
            cp = %{pc: target, regs: state.regs, heap: state.heap,
                   cp: state.cp, trail: state.trail, stack: state.stack}
            %{state | choice_points: [cp | rest], pc: state.pc + 1}
          _ -> %{state | pc: state.pc + 1}
        end').

wam_elixir_case(trust_me,
'      :trust_me ->
        case state.choice_points do
          [_ | rest] -> %{state | choice_points: rest, pc: state.pc + 1}
          _ -> %{state | pc: state.pc + 1}
        end').

% --- Indexing Instructions ---

wam_elixir_case(switch_on_constant,
'      {:switch_on_constant, entries} ->
        step_switch_on_constant(state, entries)').

% --- Builtin Instructions ---

wam_elixir_case(builtin_call,
'      {:builtin_call, op, arity} ->
        execute_builtin(state, op, arity)').

% ============================================================================
% PHASE 3: Helper functions → Elixir def bodies
% ============================================================================

%% compile_wam_helpers_to_elixir(+Options, -Code)
compile_wam_helpers_to_elixir(_Options, Code) :-
    compile_run_loop_to_elixir(RunCode),
    compile_backtrack_to_elixir(BTCode),
    compile_unwind_trail_to_elixir(UnwindCode),
    compile_utility_helpers_to_elixir(UtilCode),
    atomic_list_concat([RunCode, '\n\n', BTCode, '\n\n', UnwindCode, '\n\n', UtilCode], Code).

compile_run_loop_to_elixir(Code) :-
    format(string(Code),
'  @doc "Main fetch-step-backtrack loop"
  def run(state) do
    cond do
      state.pc == :halt -> {:ok, state}
      true ->
        instr = fetch(state)
        case step(state, instr) do
          :fail ->
            case backtrack(state) do
              :fail -> :fail
              new_state -> run(new_state)
            end
          new_state -> run(new_state)
        end
    end
  end

  defp fetch(state) do
    Enum.at(state.code, state.pc - 1)
  end', []).

compile_backtrack_to_elixir(Code) :-
    format(string(Code),
'  @doc "Restore from most recent choice point"
  def backtrack(state) do
    case state.choice_points do
      [] -> :fail
      [cp | rest] ->
        # cp.trail contains the trail snapshot at save time.
        # mark = length(cp.trail). We unwind all entries added after that.
        state = state
        |> unwind_trail(length(cp.trail))
        |> Map.put(:pc, cp.pc)
        |> Map.put(:regs, cp.regs)
        |> Map.put(:heap, cp.heap)
        |> Map.put(:cp, cp.cp)
        |> Map.put(:stack, cp.stack)
        |> Map.put(:trail, cp.trail)
        |> Map.put(:choice_points, rest)
        
        if is_function(cp.pc) do
          cp.pc.(state)
        else
          {:ok, state}
        end
    end
  end', []).

compile_unwind_trail_to_elixir(Code) :-
    format(string(Code),
'  @doc "Undo register bindings back to trail mark"
  def unwind_trail(state, mark) do
    # mark is the length of the trail when the choice point was created.
    # Newest entries are at the head of state.trail.
    entries_to_undo = Enum.take(state.trail, length(state.trail) - mark)
    new_trail = Enum.drop(state.trail, length(state.trail) - mark)
    
    Enum.reduce(entries_to_undo, %{state | trail: new_trail}, fn {key, old_val}, s ->
      case key do
        {:heap_ref, addr} ->
           val = if old_val == :not_set, do: {:unbound, {:heap_ref, addr}}, else: old_val
           %{s | heap: List.replace_at(s.heap, addr, val)}
        _ ->
           new_regs = if old_val == :not_set, do: Map.delete(s.regs, key), else: Map.put(s.regs, key, old_val)
           %{s | regs: new_regs}
      end
    end)
  end', []).

compile_utility_helpers_to_elixir(Code) :-
    format(string(Code),
'  def trail_binding(state, {:heap_ref, addr} = key) do
    old = Enum.at(state.heap, addr, :not_set)
    %{state | trail: [{key, old} | state.trail]}
  end

  def trail_binding(state, key) do
    old = Map.get(state.regs, key, :not_set)
    %{state | trail: [{key, old} | state.trail]}
  end

  def put_reg(state, reg, val) do
    %{state | regs: Map.put(state.regs, reg, val)}
  end

  def get_reg(state, reg) do
    val = Map.get(state.regs, reg, {:unbound, reg})
    deref_var(state, val)
  end

  defp resolve_label(state, label) do
    Map.get(state.labels, label, state.pc)
  end

  def parse_functor_arity(fn_name) do
    case String.split(fn_name, "/") do
      [_, arity_str] -> String.to_integer(arity_str)
      _ -> 0
    end
  end

  def deref_var(state, {:unbound, {:heap_ref, addr}} = ref) do
    case Enum.at(state.heap, addr) do
      {:unbound, {:heap_ref, _addr}} -> ref
      val -> deref_var(state, val)
    end
  end

  def deref_var(state, {:unbound, id}) do
    case Map.get(state.regs, id) do
      nil -> {:unbound, id}
      {:unbound, ^id} -> {:unbound, id}
      val -> deref_var(state, val)
    end
  end
  def deref_var(_state, val), do: val

  def step_get_structure_ref(state, fn_name, addr) do
    entry = Enum.at(state.heap, addr)
    cond do
      entry == {:str, fn_name} ->
        arity = parse_functor_arity(fn_name)
        args = Enum.slice(state.heap, addr + 1, arity)
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        %{state | stack: [{:unify_ctx, args} | state.stack], pc: new_pc}
      true -> :fail
    end
  end

  def step_unify_variable(state, xn) do
    case state.stack do
      [{:unify_ctx, [arg | rest]} | stack_rest] ->
        new_stack = if rest == [], do: stack_rest, else: [{:unify_ctx, rest} | stack_rest]
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        state |> trail_binding(xn) |> put_reg(xn, arg)
        |> Map.put(:stack, new_stack) |> Map.put(:pc, new_pc)
      [{:write_ctx, n} | stack_rest] when n > 0 ->
        addr = length(state.heap)
        fresh = {:unbound, {:heap_ref, addr}}
        new_stack = if n == 1, do: stack_rest, else: [{:write_ctx, n - 1} | stack_rest]
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        state |> put_reg(xn, fresh)
        |> Map.put(:heap, state.heap ++ [fresh])
        |> Map.put(:stack, new_stack) |> Map.put(:pc, new_pc)
      _ -> :fail
    end
  end

  def step_unify_value(state, xn) do
    val = get_reg(state, xn)
    case state.stack do
      [{:unify_ctx, [arg | rest]} | stack_rest] ->
        case unify(state, val, arg) do
          {:ok, new_state} ->
            new_stack = if rest == [], do: stack_rest, else: [{:unify_ctx, rest} | stack_rest]
            new_pc = if is_integer(new_state.pc), do: new_state.pc + 1, else: new_state.pc
            %{new_state | stack: new_stack, pc: new_pc}
          :fail -> :fail
        end
      [{:write_ctx, n} | stack_rest] when n > 0 ->
        new_stack = if n == 1, do: stack_rest, else: [{:write_ctx, n - 1} | stack_rest]
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        %{state | heap: state.heap ++ [val], stack: new_stack, pc: new_pc}
      _ -> :fail
    end
  end

  def step_unify_constant(state, c) do
    case state.stack do
      [{:unify_ctx, [arg | rest]} | stack_rest] ->
        case unify(state, c, arg) do
          {:ok, new_state} ->
            new_stack = if rest == [], do: stack_rest, else: [{:unify_ctx, rest} | stack_rest]
            new_pc = if is_integer(new_state.pc), do: new_state.pc + 1, else: new_state.pc
            %{new_state | stack: new_stack, pc: new_pc}
          :fail -> :fail
        end
      [{:write_ctx, n} | stack_rest] when n > 0 ->
        new_stack = if n == 1, do: stack_rest, else: [{:write_ctx, n - 1} | stack_rest]
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        %{state | heap: state.heap ++ [c], stack: new_stack, pc: new_pc}
      _ -> :fail
    end
  end

  defp step_switch_on_constant(state, entries) do
    val = get_reg(state, 1)
    cond do
      match?({:unbound, _}, val) -> %{state | pc: if(is_integer(state.pc), do: state.pc + 1, else: state.pc)}
      true ->
        # entries is a list of {"key", "label"}. Convert to Map for faster lookup.
        # Ideally this conversion happens at compile time or load time.
        map = Map.new(entries)
        case Map.get(map, val) do
          "default" -> %{state | pc: if(is_integer(state.pc), do: state.pc + 1, else: state.pc)}
          nil -> :fail
          label -> %{state | pc: resolve_label(state, label)}
        end
    end
  end

  @doc "Unify two WAM values"
  def unify(state, v1, v2) do
    cond do
      v1 == v2 -> {:ok, state}
      match?({:unbound, {:heap_ref, _addr}}, v1) ->
        {:unbound, {:heap_ref, addr}} = v1
        new_heap = List.replace_at(state.heap, addr, v2)
        new_state = state |> trail_binding({:heap_ref, addr})
        {:ok, %{new_state | heap: new_heap}}
      match?({:unbound, {:heap_ref, _addr}}, v2) ->
        {:unbound, {:heap_ref, addr}} = v2
        new_heap = List.replace_at(state.heap, addr, v1)
        new_state = state |> trail_binding({:heap_ref, addr})
        {:ok, %{new_state | heap: new_heap}}
      match?({:unbound, _}, v1) ->
        {:unbound, id} = v1
        new_state = state |> trail_binding(id) |> put_reg(id, v2)
        {:ok, new_state}
      match?({:unbound, _}, v2) ->
        {:unbound, id} = v2
        new_state = state |> trail_binding(id) |> put_reg(id, v1)
        {:ok, new_state}
      true -> :fail
    end
  end

  @doc "Execute a builtin predicate"
  def execute_builtin(state, op, arity) do
    case {op, arity} do
      {"is/2", 2} ->
        expr = get_reg(state, 2)
        result = eval_arith(state, expr)
        lhs = get_reg(state, 1)
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        cond do
          match?({:unbound, _}, lhs) ->
            id = case lhs do {:unbound, i} -> i end
            state |> trail_binding(id)
            |> Map.put(:regs, Map.put(state.regs, id, result))
            |> Map.put(:pc, new_pc)
          lhs == result -> %{state | pc: new_pc}
          true -> :fail
        end
      {"</2", 2} ->
        v1 = eval_arith(state, get_reg(state, 1))
        v2 = eval_arith(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if v1 < v2, do: %{state | pc: new_pc}, else: :fail
      {"length/2", 2} ->
        list = get_reg(state, 1)
        len = if is_list(list), do: length(list), else: throw(:fail)
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        case unify(state, len, get_reg(state, 2)) do
          {:ok, s} -> %{s | pc: new_pc}
          :fail -> :fail
        end
      {neg_op, 1} when neg_op in ["\\\\+/1", "\\+/1"] ->
        # Negation: \\+ Goal
        goal = get_reg(state, 1)
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        case goal do
          {:ref, addr} ->
            case Enum.at(state.heap, addr) do
              {:str, "member/2"} ->
                 [v1, v2] = Enum.slice(state.heap, addr + 1, 2)
                 temp_state = %{state | regs: Map.merge(state.regs, %{1 => v1, 2 => v2})}
                 case execute_builtin(temp_state, "member/2", 2) do
                   :fail -> %{state | pc: new_pc}
                   _ -> :fail
                 end
               _ -> :fail # TODO: General negation
            end
          _ -> :fail
        end
      {"member/2", 2} ->
        item = get_reg(state, 1)
        list = get_reg(state, 2)
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        cond do
          is_list(list) ->
            if item in list, do: %{state | pc: new_pc}, else: :fail
          true -> :fail
        end
      _ -> :fail
    end
  end

  defp eval_arith(_state, n) when is_number(n), do: n
  defp eval_arith(_state, n) when is_binary(n) do
    case Float.parse(n) do
      {f, _} -> f
      :error ->
        case Integer.parse(n) do
          {i, _} -> i
          :error -> throw({:eval_error, n})
        end
    end
  end
  defp eval_arith(state, {:ref, addr}) do
    case Enum.at(state.heap, addr) do
      {:str, "+/2"} ->
        [v1, v2] = Enum.slice(state.heap, addr + 1, 2)
        eval_arith(state, v1) + eval_arith(state, v2)
      {:str, "-/2"} ->
        [v1, v2] = Enum.slice(state.heap, addr + 1, 2)
        eval_arith(state, v1) - eval_arith(state, v2)
      {:str, "*/2"} ->
        [v1, v2] = Enum.slice(state.heap, addr + 1, 2)
        eval_arith(state, v1) * eval_arith(state, v2)
      val -> eval_arith(state, val)
    end
  end
  defp eval_arith(state, val), do: eval_arith(state, deref_var(state, val))', []).

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3
% ============================================================================

%% compile_wam_runtime_to_elixir(+Options, -Code)
compile_wam_runtime_to_elixir(Options, Code) :-
    compile_step_wam_to_elixir(Options, StepCode),
    compile_wam_helpers_to_elixir(Options, HelpersCode),
    format(string(Code),
'defmodule WamRuntime do
  @moduledoc "WAM Virtual Machine runtime - generated by UnifyWeaver"

  defmodule WamState do
    defstruct pc: 1, cp: :halt, regs: %{}, heap: [], trail: [],
              choice_points: [], stack: [], code: [], labels: %{}
  end

~w

~w
end', [StepCode, HelpersCode]).

% ============================================================================
% PREDICATE WRAPPER
% ============================================================================

%% compile_wam_predicate_to_elixir(+Pred/Arity, +WamCode, +Options, -Code)
compile_wam_predicate_to_elixir(Pred/Arity, WamCode, _Options, Code) :-
    atom_string(Pred, PredStr),
    wam_code_to_elixir_instructions(WamCode, InstrLiterals, LabelLiterals),
    format(string(Code),
'defmodule WamPred.~w do
  @moduledoc "WAM-compiled predicate: ~w/~w"

  def code do
    [
~w
    ]
  end

  def labels do
    %{~w}
  end

  def run(args) do
    state = %WamRuntime.WamState{code: code(), labels: labels(), pc: 1}
    state = Enum.with_index(args, 1)
    |> Enum.reduce(state, fn {arg, i}, s ->
      %{s | regs: Map.put(s.regs, i, arg)}
    end)
    WamRuntime.run(state)
  end
end', [PredStr, PredStr, Arity, InstrLiterals, LabelLiterals]).

%% wam_code_to_elixir_instructions(+WamCodeStr, -InstrLiterals, -LabelLiterals)
wam_code_to_elixir_instructions(WamCode, InstrLiterals, LabelLiterals) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_elixir(Lines, 1, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    atomic_list_concat(LabelParts, ', ', LabelLiterals).

wam_lines_to_elixir([], _, [], []).
wam_lines_to_elixir([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_elixir(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   is_label_part(First)
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelInsert), '"~w" => ~w', [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_elixir(Rest, PC, Instrs, RestLabels)
        ;   wam_line_to_elixir_instr(CleanParts, ElixirInstr),
            format(string(InstrEntry), '      ~w,', [ElixirInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_elixir(Rest, NPC, RestInstrs, Labels)
        )
    ).

wam_line_to_elixir_instr(["try_me_else", L], Instr) :-
    clean_comma(L, CL),
    format(string(Instr), '{:try_me_else, "~w"}', [CL]).
wam_line_to_elixir_instr(["retry_me_else", L], Instr) :-
    clean_comma(L, CL),
    format(string(Instr), '{:retry_me_else, "~w"}', [CL]).
wam_line_to_elixir_instr(["trust_me"], ':trust_me').
wam_line_to_elixir_instr(["put_structure", F, Ai], Instr) :-
    clean_comma(F, CF), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    format(string(Instr), '{:put_structure, "~w", ~w}', [CF, AiId]).
wam_line_to_elixir_instr(["get_structure", F, Ai], Instr) :-
    clean_comma(F, CF), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    format(string(Instr), '{:get_structure, "~w", ~w}', [CF, AiId]).
wam_line_to_elixir_instr(["unify_variable", Xn], Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:unify_variable, ~w}', [XnId]).
wam_line_to_elixir_instr(["unify_value", Xn], Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:unify_value, ~w}', [XnId]).
wam_line_to_elixir_instr(["unify_constant", C], Instr) :-
    clean_comma(C, CC),
    format(string(Instr), '{:unify_constant, "~w"}', [CC]).
wam_line_to_elixir_instr(["set_variable", Xn], Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:set_variable, ~w}', [XnId]).
wam_line_to_elixir_instr(["set_value", Xn], Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:set_value, ~w}', [XnId]).
wam_line_to_elixir_instr(["set_constant", C], Instr) :-
    clean_comma(C, CC),
    format(string(Instr), '{:set_constant, "~w"}', [CC]).
wam_line_to_elixir_instr(["get_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    format(string(Instr), '{:get_constant, "~w", ~w}', [CC, AiId]).
wam_line_to_elixir_instr(["get_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi), 
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:get_variable, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["get_value", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi), 
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:get_value, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["put_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    format(string(Instr), '{:put_constant, "~w", ~w}', [CC, AiId]).
wam_line_to_elixir_instr(["put_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi), 
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:put_variable, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["put_value", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi), 
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:put_value, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["proceed"], ':proceed').
wam_line_to_elixir_instr(["call", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(string(Instr), '{:call, "~w", ~w}', [CP, CN]).
wam_line_to_elixir_instr(["execute", P], Instr) :-
    clean_comma(P, CP),
    format(string(Instr), '{:execute, "~w"}', [CP]).
wam_line_to_elixir_instr(["allocate", N], Instr) :-
    clean_comma(N, CN),
    format(string(Instr), '{:allocate, ~w}', [CN]).
wam_line_to_elixir_instr(["deallocate"], ':deallocate').
wam_line_to_elixir_instr(["builtin_call", Op, Ar], Instr) :-
    clean_comma(Op, COp), clean_comma(Ar, CAr),
    (   sub_atom(COp, 0, 1, _, '\\') % Handle \+ / 1
    ->  sub_atom(COp, 1, _, 0, Rest),
        format(string(Instr), '{:builtin_call, "\\\\~w", ~w}', [Rest, CAr])
    ;   format(string(Instr), '{:builtin_call, "~w", ~w}', [COp, CAr])
    ).
wam_line_to_elixir_instr(Parts, Instr) :-
    atomic_list_concat(Parts, ' ', Combined),
    format(string(Instr), '{:raw, "~w"}', [Combined]).

% ============================================================================
% PROJECT GENERATION
% ============================================================================

%% write_wam_elixir_project(Predicates, Options, ProjectDir)
write_wam_elixir_project(Predicates, Options, ProjectDir) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    wam_elixir_resolve_emit_mode(Options, Mode),
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'lib', LibDir),
    make_directory_path(LibDir),
    % Generate runtime module
    compile_wam_runtime_to_elixir(Options, RuntimeCode),
    directory_file_path(LibDir, 'wam_runtime.ex', RuntimePath),
    open(RuntimePath, write, RS),
    write(RS, RuntimeCode),
    close(RS),
    % Generate dispatcher for lowered mode
    (   Mode == lowered
    ->  generate_elixir_dispatcher(Predicates, Options, DispatcherCode),
        directory_file_path(LibDir, 'wam_dispatcher.ex', DispatcherPath),
        open(DispatcherPath, write, DS),
        write(DS, DispatcherCode),
        close(DS)
    ;   true
    ),
    % Generate predicate modules
    forall(
        member(Pred/Arity-WamCode, Predicates),
        (   (   Mode == lowered
            ->  lower_predicate_to_elixir(Pred/Arity, WamCode, Options, PredCode)
            ;   compile_wam_predicate_to_elixir(Pred/Arity, WamCode, Options, PredCode)
            ),
            atom_string(Pred, PredStr),
            format(atom(PredFile), '~w.ex', [PredStr]),
            directory_file_path(LibDir, PredFile, PredPath),
            open(PredPath, write, PS),
            write(PS, PredCode),
            close(PS)
        )
    ),
    % Generate mix.exs
    format(string(MixCode),
'defmodule ~w.MixProject do
  use Mix.Project

  def project do
    [
      app: :~w,
      version: "0.1.0",
      elixir: "~~> 1.14",
      deps: []
    ]
  end
end', [ModuleName, ModuleName]),
    directory_file_path(ProjectDir, 'mix.exs', MixPath),
    open(MixPath, write, MS),
    write(MS, MixCode),
    close(MS).

generate_elixir_dispatcher(Predicates, Options, Code) :-
    option(module_name(ModName), Options, 'WamPredLow'),
    camel_case(ModName, CamelMod),
    findall(Case,
        (   member(Pred/Arity-_, Predicates),
            atom_string(Pred, PredStr),
            camel_case(PredStr, CamelPred),
            format(string(Case), '  def call("~w/~w", state), do: ~w.~w.run(state)', [PredStr, Arity, CamelMod, CamelPred])
        ),
        Cases
    ),
    atomic_list_concat(Cases, '\n', CasesStr),
    format(string(Code),
'defmodule WamDispatcher do
  @moduledoc "Global dispatcher for dynamic WAM calls"

~w
  def call(pred, _state), do: throw({:undefined_predicate, pred})
end', [CasesStr]).
