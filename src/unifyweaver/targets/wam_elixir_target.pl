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
:- use_module('wam_elixir_utils', [reg_id/2, clean_comma/2, is_label_part/1, camel_case/2, parse_arity/2]).

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
'      {:get_structure, fn_name, arity, ai} ->
        val = Map.get(state.regs, ai)
        cond do
          match?({:unbound, _}, val) ->
            addr = state.heap_len
            new_heap = Map.put(state.heap, addr, {:str, fn_name})
            state
            |> trail_binding(ai)
            |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
            |> Map.put(:heap, new_heap)
            |> Map.put(:heap_len, addr + 1)
            |> Map.put(:stack, [{:write_ctx, arity} | state.stack])
            |> Map.put(:pc, state.pc + 1)
          match?({:ref, _}, val) ->
            {:ref, addr} = val
            step_get_structure_ref(state, fn_name, arity, addr)
          true -> :fail
        end').

wam_elixir_case(get_list,
'      {:get_list, ai} ->
        val = Map.get(state.regs, ai)
        cond do
          match?({:unbound, _}, val) ->
            addr = state.heap_len
            new_heap = Map.put(state.heap, addr, {:str, "./2"})
            state
            |> trail_binding(ai)
            |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
            |> Map.put(:heap, new_heap)
            |> Map.put(:heap_len, addr + 1)
            |> Map.put(:stack, [{:write_ctx, 2} | state.stack])
            |> Map.put(:pc, state.pc + 1)
          match?({:ref, _}, val) ->
            {:ref, addr} = val
            step_get_structure_ref(state, "./2", 2, addr)
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
'      {:put_structure, fn_name, arity, ai} ->
        addr = state.heap_len
        new_heap = Map.put(state.heap, addr, {:str, fn_name})
        state
        |> trail_binding(ai)
        |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:heap_len, addr + 1)
        |> Map.put(:stack, [{:write_ctx, arity} | state.stack])
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(put_list,
'      {:put_list, ai} ->
        addr = state.heap_len
        new_heap = Map.put(state.heap, addr, {:str, "./2"})
        state
        |> trail_binding(ai)
        |> Map.put(:regs, Map.put(state.regs, ai, {:ref, addr}))
        |> Map.put(:heap, new_heap)
        |> Map.put(:heap_len, addr + 1)
        |> Map.put(:stack, [{:write_ctx, 2} | state.stack])
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(set_variable,
'      {:set_variable, xn} ->
        addr = state.heap_len
        fresh = {:unbound, {:heap_ref, addr}}
        new_heap = Map.put(state.heap, addr, fresh)
        state
        |> put_reg(xn, fresh)
        |> Map.put(:heap, new_heap)
        |> Map.put(:heap_len, addr + 1)
        |> Map.put(:pc, state.pc + 1)').

wam_elixir_case(set_value,
'      {:set_value, xn} ->
        val = get_reg(state, xn)
        addr = state.heap_len
        %{state | heap: Map.put(state.heap, addr, val), heap_len: addr + 1, pc: state.pc + 1}').

wam_elixir_case(set_constant,
'      {:set_constant, c} ->
        addr = state.heap_len
        %{state | heap: Map.put(state.heap, addr, c), heap_len: addr + 1, pc: state.pc + 1}').

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
'      # Fast path: label pre-resolved at codegen to integer PC.
      {:call, target_pc, _n} when is_integer(target_pc) ->
        %{state | pc: target_pc, cp: state.pc + 1}
      # Fallback: unresolved string label (cross-module/orphan).
      {:call, p, _n} when is_binary(p) ->
        case Map.get(state.labels, p) do
          nil -> :fail
          target_pc -> %{state | pc: target_pc, cp: state.pc + 1}
        end').

wam_elixir_case(execute,
'      # Fast path: label pre-resolved at codegen to integer PC.
      {:execute, target_pc} when is_integer(target_pc) ->
        %{state | pc: target_pc}
      # Fallback: unresolved string label.
      {:execute, p} when is_binary(p) ->
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
'      # Fast path: label pre-resolved at codegen to integer PC.
      {:try_me_else, target} when is_integer(target) ->
        cp = %{pc: target, regs: state.regs, heap: state.heap, heap_len: state.heap_len,
               cp: state.cp, trail: state.trail, trail_len: state.trail_len, stack: state.stack}
        %{state | choice_points: [cp | state.choice_points], pc: state.pc + 1}
      # Fallback: unresolved string label.
      {:try_me_else, label} when is_binary(label) ->
        target = resolve_label(state, label)
        cp = %{pc: target, regs: state.regs, heap: state.heap, heap_len: state.heap_len,
               cp: state.cp, trail: state.trail, trail_len: state.trail_len, stack: state.stack}
        %{state | choice_points: [cp | state.choice_points], pc: state.pc + 1}').

wam_elixir_case(retry_me_else,
'      # Fast path: label pre-resolved at codegen to integer PC.
      {:retry_me_else, target} when is_integer(target) ->
        case state.choice_points do
          [_old | rest] ->
            cp = %{pc: target, regs: state.regs, heap: state.heap, heap_len: state.heap_len,
                   cp: state.cp, trail: state.trail, trail_len: state.trail_len, stack: state.stack}
            %{state | choice_points: [cp | rest], pc: state.pc + 1}
          _ -> %{state | pc: state.pc + 1}
        end
      # Fallback: unresolved string label.
      {:retry_me_else, label} when is_binary(label) ->
        target = resolve_label(state, label)
        case state.choice_points do
          [_old | rest] ->
            cp = %{pc: target, regs: state.regs, heap: state.heap, heap_len: state.heap_len,
                   cp: state.cp, trail: state.trail, trail_len: state.trail_len, stack: state.stack}
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
              {:ok, new_state} -> run(new_state)
            end
          new_state -> run(new_state)
        end
    end
  end

  defp fetch(state) do
    elem(state.code, state.pc - 1)
  end', []).

compile_backtrack_to_elixir(Code) :-
    format(string(Code),
'  @doc "Restore from most recent choice point"
  def backtrack(state) do
    case state.choice_points do
      [] -> :fail
      [cp | rest] ->
        # cp.trail_len is the mark — unwind all entries pushed after that.
        state = state
        |> unwind_trail(cp.trail_len)
        |> Map.put(:pc, cp.pc)
        |> Map.put(:regs, cp.regs)
        |> Map.put(:heap, cp.heap)
        |> Map.put(:heap_len, cp.heap_len)
        |> Map.put(:cp, cp.cp)
        |> Map.put(:stack, cp.stack)
        |> Map.put(:trail, cp.trail)
        |> Map.put(:trail_len, cp.trail_len)
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
    # mark is the trail_len when the choice point was created.
    # Newest entries are at the head of state.trail.
    curr_len = state.trail_len
    if curr_len <= mark do
      state
    else
      drop_n = curr_len - mark
      entries_to_undo = Enum.take(state.trail, drop_n)
      new_trail = Enum.drop(state.trail, drop_n)

      Enum.reduce(entries_to_undo, %{state | trail: new_trail, trail_len: mark}, fn {key, old_val}, s ->
        case key do
          {:heap_ref, addr} ->
             val = if old_val == :not_set, do: {:unbound, {:heap_ref, addr}}, else: old_val
             %{s | heap: Map.put(s.heap, addr, val)}
          _ ->
             new_regs = if old_val == :not_set, do: Map.delete(s.regs, key), else: Map.put(s.regs, key, old_val)
             %{s | regs: new_regs}
        end
      end)
    end
  end', []).

compile_utility_helpers_to_elixir(Code) :-
    format(string(Code),
'  def trail_binding(state, {:heap_ref, addr} = key) do
    old = Map.get(state.heap, addr, :not_set)
    %{state | trail: [{key, old} | state.trail], trail_len: state.trail_len + 1}
  end

  def trail_binding(state, key) do
    old = Map.get(state.regs, key, :not_set)
    %{state | trail: [{key, old} | state.trail], trail_len: state.trail_len + 1}
  end

  def put_reg(state, reg, val) do
    %{state | regs: Map.put(state.regs, reg, val)}
  end

  def get_reg(state, reg) do
    val = Map.get(state.regs, reg, {:unbound, reg})
    deref_var(state, val)
  end

  defp resolve_label(state, label) do
    target = Map.get(state.labels, label, state.pc)
    target
  end

  def parse_functor_arity(fn_name) do
    case String.split(fn_name, "/") do
      [_, arity_str] -> String.to_integer(arity_str)
      _ -> 0
    end
  end

  @doc "Read `len` consecutive heap cells starting at `start`."
  def heap_slice(_state, _start, 0), do: []
  def heap_slice(state, start, len) when len > 0 do
    Enum.map(0..(len - 1), fn i -> Map.get(state.heap, start + i) end)
  end

  @doc "Append `val` to the heap; returns {new_state, addr}."
  def heap_push(state, val) do
    addr = state.heap_len
    {%{state | heap: Map.put(state.heap, addr, val), heap_len: addr + 1}, addr}
  end

  def deref_var(state, {:unbound, {:heap_ref, addr}} = ref) do
    case Map.get(state.heap, addr) do
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

  def step_get_structure_ref(state, fn_name, arity, addr) do
    entry = Map.get(state.heap, addr)
    cond do
      entry == {:str, fn_name} ->
        args = heap_slice(state, addr + 1, arity)
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
        addr = state.heap_len
        fresh = {:unbound, {:heap_ref, addr}}
        new_stack = if n == 1, do: stack_rest, else: [{:write_ctx, n - 1} | stack_rest]
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        state |> put_reg(xn, fresh)
        |> Map.put(:heap, Map.put(state.heap, addr, fresh))
        |> Map.put(:heap_len, addr + 1)
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
        addr = state.heap_len
        %{state | heap: Map.put(state.heap, addr, val), heap_len: addr + 1, stack: new_stack, pc: new_pc}
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
        addr = state.heap_len
        %{state | heap: Map.put(state.heap, addr, c), heap_len: addr + 1, stack: new_stack, pc: new_pc}
      _ -> :fail
    end
  end

  defp step_switch_on_constant(state, entries) do
    val = get_reg(state, 1)
    cond do
      match?({:unbound, _}, val) -> %{state | pc: if(is_integer(state.pc), do: state.pc + 1, else: state.pc)}
      true ->
        # entries is expected to be a list of {key, label} or a map.
        map = if is_map(entries), do: entries, else: Map.new(entries)
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
        new_heap = Map.put(state.heap, addr, v2)
        new_state = state |> trail_binding({:heap_ref, addr})
        {:ok, %{new_state | heap: new_heap}}
      match?({:unbound, {:heap_ref, _addr}}, v2) ->
        {:unbound, {:heap_ref, addr}} = v2
        new_heap = Map.put(state.heap, addr, v1)
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
            state 
            |> trail_binding(id)
            |> put_reg(id, result)
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
        # Note: Ground atoms as goals (e.g. \\+ true) always fail negation check 
        # in this implementation as they are not runnable without arity.
        goal_val = deref_var(state, get_reg(state, 1))
        # Isolated state for negation check
        temp_state = %{state | choice_points: [], stack: []}
        
        # Determine if goal succeeds or fails
        res = case goal_val do
          {:ref, addr} ->
            case Map.get(state.heap, addr) do
              {:str, pred_arity} ->
                arity = parse_functor_arity(pred_arity)
                args = heap_slice(state, addr + 1, arity)
                call_state = Enum.with_index(args, 1)
                |> Enum.reduce(temp_state, fn {arg, i}, s -> %{s | regs: Map.put(s.regs, i, arg)} end)

                try do
                  case WamDispatcher.call(pred_arity, call_state) do
                    {:ok, _} -> :success
                    :fail -> :fail
                  end
                catch
                  :fail -> :fail
                  {:return, _} -> :success
                  # Let other exceptions propagate for easier debugging
                end
              _ -> :fail
            end
          _ -> :fail
        end

        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        case res do
          :fail -> %{state | pc: new_pc}
          :success -> :fail
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
    case Map.get(state.heap, addr) do
      {:str, "+/2"} ->
        [v1, v2] = heap_slice(state, addr + 1, 2)
        eval_arith(state, v1) + eval_arith(state, v2)
      {:str, "-/2"} ->
        [v1, v2] = heap_slice(state, addr + 1, 2)
        eval_arith(state, v1) - eval_arith(state, v2)
      {:str, "*/2"} ->
        [v1, v2] = heap_slice(state, addr + 1, 2)
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
    # heap is a map keyed by integer address; heap_len is the next free addr.
    # trail_len caches list length to avoid O(n) re-measure on unwind.
    # code is a tuple so fetch is O(1) via elem/2 instead of O(pc) via Enum.at.
    # Phase A perf: O(1) append/read/replace instead of list O(n) operations.
    defstruct pc: 1, cp: :halt, regs: %{}, heap: %{}, heap_len: 0,
              trail: [], trail_len: 0,
              choice_points: [], stack: [], code: {}, labels: %{}
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

  # code is a tuple so WamRuntime.fetch uses elem/2 for O(1) instruction lookup.
  def code do
    {
~w
    }
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
%  Two-pass: pass 1 collects labels → PC map; pass 2 emits instructions
%  with label references pre-resolved to integer PCs where possible.
wam_code_to_elixir_instructions(WamCode, InstrLiterals, LabelLiterals) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    collect_labels_pass(Lines, 1, LabelsList),
    wam_lines_to_elixir(Lines, 1, LabelsList, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    atomic_list_concat(LabelParts, ', ', LabelLiterals).

%% collect_labels_pass(+Lines, +StartPC, -LabelsList)
%  Returns a list of LabelName-PC pairs (strings and integers).
collect_labels_pass([], _, []).
collect_labels_pass([Line|Rest], PC, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == [] -> collect_labels_pass(Rest, PC, Labels)
    ;   CleanParts = [First|_], is_label_part(First)
    ->  sub_string(First, 0, _, 1, LabelName),
        Labels = [LabelName-PC | RestLabels],
        collect_labels_pass(Rest, PC, RestLabels)
    ;   NPC is PC + 1,
        collect_labels_pass(Rest, NPC, Labels)
    ).

wam_lines_to_elixir([], _, _, [], []).
wam_lines_to_elixir([Line|Rest], PC, LabelsList, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_elixir(Rest, PC, LabelsList, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   is_label_part(First)
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelInsert), '"~w" => ~w', [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_elixir(Rest, PC, LabelsList, Instrs, RestLabels)
        ;   wam_line_to_elixir_instr(CleanParts, LabelsList, ElixirInstr),
            format(string(InstrEntry), '      ~w,', [ElixirInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_elixir(Rest, NPC, LabelsList, RestInstrs, Labels)
        )
    ).

%% resolve_label(+LabelStr, +LabelsList, -Resolved)
%  Finds the PC for LabelStr in LabelsList. On hit, Resolved is an integer.
%  On miss (cross-module / orphan), Resolved is the original string — the
%  runtime path through state.labels handles those.
resolve_label(LabelStr, LabelsList, Resolved) :-
    (   member(LabelStr-PC, LabelsList)
    ->  Resolved = PC
    ;   Resolved = LabelStr
    ).

wam_line_to_elixir_instr(["try_me_else", L], LabelsList, Instr) :-
    clean_comma(L, CL),
    resolve_label(CL, LabelsList, R),
    (   integer(R)
    ->  format(string(Instr), '{:try_me_else, ~w}', [R])
    ;   format(string(Instr), '{:try_me_else, "~w"}', [R])
    ).
wam_line_to_elixir_instr(["retry_me_else", L], LabelsList, Instr) :-
    clean_comma(L, CL),
    resolve_label(CL, LabelsList, R),
    (   integer(R)
    ->  format(string(Instr), '{:retry_me_else, ~w}', [R])
    ;   format(string(Instr), '{:retry_me_else, "~w"}', [R])
    ).
wam_line_to_elixir_instr(["trust_me"], _, ':trust_me').
wam_line_to_elixir_instr(["put_structure", F, Ai], _, Instr) :-
    clean_comma(F, CF), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    parse_arity(CF, Arity),
    format(string(Instr), '{:put_structure, "~w", ~w, ~w}', [CF, Arity, AiId]).
wam_line_to_elixir_instr(["get_structure", F, Ai], _, Instr) :-
    clean_comma(F, CF), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    parse_arity(CF, Arity),
    format(string(Instr), '{:get_structure, "~w", ~w, ~w}', [CF, Arity, AiId]).
wam_line_to_elixir_instr(["unify_variable", Xn], _, Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:unify_variable, ~w}', [XnId]).
wam_line_to_elixir_instr(["unify_value", Xn], _, Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:unify_value, ~w}', [XnId]).
wam_line_to_elixir_instr(["unify_constant", C], _, Instr) :-
    clean_comma(C, CC),
    format(string(Instr), '{:unify_constant, "~w"}', [CC]).
wam_line_to_elixir_instr(["set_variable", Xn], _, Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:set_variable, ~w}', [XnId]).
wam_line_to_elixir_instr(["set_value", Xn], _, Instr) :-
    clean_comma(Xn, CXn), reg_id(CXn, XnId),
    format(string(Instr), '{:set_value, ~w}', [XnId]).
wam_line_to_elixir_instr(["set_constant", C], _, Instr) :-
    clean_comma(C, CC),
    format(string(Instr), '{:set_constant, "~w"}', [CC]).
wam_line_to_elixir_instr(["get_constant", C, Ai], _, Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    format(string(Instr), '{:get_constant, "~w", ~w}', [CC, AiId]).
wam_line_to_elixir_instr(["get_variable", Xn, Ai], _, Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:get_variable, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["get_value", Xn, Ai], _, Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:get_value, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["put_constant", C, Ai], _, Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi), reg_id(CAi, AiId),
    format(string(Instr), '{:put_constant, "~w", ~w}', [CC, AiId]).
wam_line_to_elixir_instr(["put_variable", Xn, Ai], _, Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:put_variable, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["put_value", Xn, Ai], _, Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_id(CXn, XnId), reg_id(CAi, AiId),
    format(string(Instr), '{:put_value, ~w, ~w}', [XnId, AiId]).
wam_line_to_elixir_instr(["proceed"], _, ':proceed').
wam_line_to_elixir_instr(["call", P, N], LabelsList, Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    resolve_label(CP, LabelsList, R),
    (   integer(R)
    ->  format(string(Instr), '{:call, ~w, ~w}', [R, CN])
    ;   format(string(Instr), '{:call, "~w", ~w}', [R, CN])
    ).
wam_line_to_elixir_instr(["execute", P], LabelsList, Instr) :-
    clean_comma(P, CP),
    resolve_label(CP, LabelsList, R),
    (   integer(R)
    ->  format(string(Instr), '{:execute, ~w}', [R])
    ;   format(string(Instr), '{:execute, "~w"}', [R])
    ).
wam_line_to_elixir_instr(["allocate", N], _, Instr) :-
    clean_comma(N, CN),
    format(string(Instr), '{:allocate, ~w}', [CN]).
wam_line_to_elixir_instr(["deallocate"], _, ':deallocate').
wam_line_to_elixir_instr(["builtin_call", Op, Ar], _, Instr) :-
    clean_comma(Op, COp), clean_comma(Ar, CAr),
    (   sub_atom(COp, 0, 1, _, '\\') % Handle \+ / 1
    ->  sub_atom(COp, 1, _, 0, Rest),
        format(string(Instr), '{:builtin_call, "\\\\~w", ~w}', [Rest, CAr])
    ;   format(string(Instr), '{:builtin_call, "~w", ~w}', [COp, CAr])
    ).
wam_line_to_elixir_instr(Parts, _, Instr) :-
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
