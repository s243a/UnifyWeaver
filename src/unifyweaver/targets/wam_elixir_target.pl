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
        # Save caller\'s Y-regs in the new env so the callee can freely
        # use Y-reg slots (ids 201-299) without clobbering the caller.
        {y_regs_saved, base_regs} = WamRuntime.split_y_regs(state.regs)
        new_env = %{cp: state.cp, y_regs_saved: y_regs_saved}
        %{state | stack: [new_env | state.stack], regs: base_regs, pc: state.pc + 1}').

wam_elixir_case(deallocate,
'      :deallocate ->
        case state.stack do
          [env | rest] ->
            # Discard current frame\'s Y-regs and restore the caller\'s.
            {_callee_ys, base_regs} = WamRuntime.split_y_regs(state.regs)
            merged = Map.merge(base_regs, Map.get(env, :y_regs_saved, %{}))
            %{state | cp: env.cp, stack: rest, regs: merged, pc: state.pc + 1}
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
    compile_aggregate_helpers_to_elixir(AggCode),
    atomic_list_concat([RunCode, '\n\n', BTCode, '\n\n', UnwindCode, '\n\n', UtilCode, '\n\n', AggCode], Code).

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
        # Aggregate frames are popped via finalise_aggregate/4 instead
        # of the ordinary unwind/restore path: they have no failure
        # target (cp.pc is nil) and need to bind the accumulator into
        # :agg_result_reg before resuming via :agg_return_cp.
        case Map.get(cp, :agg_type) do
          nil ->
            backtrack_ordinary(state, cp, rest)
          agg_type ->
            finalise_aggregate(state, cp, rest, agg_type)
        end
    end
  end

  defp backtrack_ordinary(state, cp, rest) do
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

    cond do
      is_function(cp.pc) ->
        # The retried clause may throw {:fail, thrown_state} (its own
        # guards failed, with CPs accumulated during its body) or
        # {:return, result} (it succeeded). Translate back into the
        # {:ok, state} | :fail contract so the caller does not need to
        # know about clause-local control flow.
        try do
          cp.pc.(state)
        catch
          {:fail, thrown_state} -> backtrack(thrown_state)
          {:return, result} -> result
        end

      match?({:fact_stream, _, _}, cp.pc) ->
        # Phase-B fact-stream CP: resume iteration over the remaining
        # tail of the fact list. The snapshot already restored regs /
        # trail / heap to their pre-unify state, so resume_fact_stream
        # just needs to attempt the next fact.
        {:fact_stream, remaining, arity} = cp.pc
        try do
          resume_fact_stream(state, remaining, arity)
        catch
          {:fail, thrown_state} -> backtrack(thrown_state)
          {:return, result} -> result
        end

      true ->
        {:ok, state}
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

  @doc """
  Partition a regs map into {y_regs, other_regs} by reg-id range.
  Y-registers use ids 201..299 (see wam_elixir_utils:reg_id/2).
  Used by allocate/deallocate to isolate Y-regs per env frame —
  without this, recursive calls to the same predicate clobber each
  other\'s permanents.
  """
  def split_y_regs(regs) do
    Enum.reduce(regs, {%{}, %{}}, fn {k, v}, {ys, others} ->
      if is_integer(k) and k >= 201 and k < 300 do
        {Map.put(ys, k, v), others}
      else
        {ys, Map.put(others, k, v)}
      end
    end)
  end

  @doc """
  Rematerialise output A-registers from the caller\'s saved unbound vars.
  Call after run(args) and after each next_solution/1 to get correct
  values in state.regs — during a clause the A-regs are used as scratch
  and may hold intermediate heap refs or foreign temporaries.
  """
  def materialise_args(state) do
    Enum.reduce(state.arg_vars, state, fn {i, v}, s ->
      %{s | regs: Map.put(s.regs, i, deref_var(s, v))}
    end)
  end

  @doc """
  Driver-facing next-solution helper: backtracks, then rematerialises
  the A-regs for caller-supplied unbound args. Returns {:ok, state} or
  :fail — matches the run/1 contract.
  """
  def next_solution(state) do
    case backtrack(state) do
      {:ok, s} -> {:ok, materialise_args(s)}
      other -> other
    end
  end

  @doc """
  Terminal continuation for CPS-lowered predicates. When a clause\'s
  `proceed` opcode runs, it tail-calls the continuation stored in
  state.cp. At the top level of a driver-initiated call, state.cp is
  this function — `run(args)` installs it before entering the clause
  chain. All it does is wrap the final state in the run/1 success
  contract.
  """
  def terminal_cp(state), do: {:ok, state}

  @doc """
  Phase-B fact-stream entry point. Called by an `inline_data`-layout
  predicate\'s `run/1` with the module\'s `@facts` literal. Iterates the
  list, attempts to unify each tuple with state.regs[1..arity], and on
  success pushes a fact-stream CP (shape `{:fact_stream, remaining,
  arity}`) so backtracking resumes at the next tuple. Emits the driver
  contract via state.cp — same as a CPS-lowered clause after a
  successful head unify.
  """
  def stream_facts(state, facts, arity) do
    resume_fact_stream(state, facts, arity)
  end

  @doc """
  Resume a fact-stream scan from the current list tail. Called both from
  `stream_facts/3` (initial call) and from `backtrack/1` (after popping
  a fact-stream CP). Empty list → `{:fail, state}` propagates to the
  caller; caller\'s outer catch routes to `backtrack`.
  """
  def resume_fact_stream(state, [], _arity), do: throw({:fail, state})
  def resume_fact_stream(state, [tuple | rest], arity) do
    trail_mark = state.trail_len
    case try_unify_fact_tuple(state, tuple, 1, arity) do
      {:ok, bound_state} ->
        # Push a CP snapshotting PRE-unify state. On later backtrack,
        # unwind_trail(trail_mark) rolls back any bindings we just made.
        cp = %{
          pc: {:fact_stream, rest, arity},
          regs: state.regs,
          heap: state.heap,
          heap_len: state.heap_len,
          cp: state.cp,
          trail: state.trail,
          trail_len: trail_mark,
          stack: state.stack
        }
        s_with_cp = %{bound_state | choice_points: [cp | bound_state.choice_points]}
        s_with_cp.cp.(s_with_cp)

      :fail ->
        # Partial unify may have bound regs before failing — roll back
        # to the pre-attempt trail mark so `rest` is tried on clean state.
        cleaned = unwind_trail(state, trail_mark)
        resume_fact_stream(cleaned, rest, arity)
    end
  end

  defp try_unify_fact_tuple(state, _tuple, i, arity) when i > arity, do: {:ok, state}
  defp try_unify_fact_tuple(state, tuple, i, arity) do
    element = elem(tuple, i - 1)
    case element do
      :_var ->
        # Head was a variable — any incoming value unifies trivially.
        try_unify_fact_tuple(state, tuple, i + 1, arity)

      value ->
        reg_val = deref_var(state, Map.get(state.regs, i))
        case reg_val do
          {:unbound, _} ->
            # Caller left this arg unbound. Bind it to the fact\'s value
            # via put_reg so the trail records it and the binding is
            # undoable on backtrack.
            {:unbound, id} = reg_val
            s2 = state
                 |> trail_binding(id)
                 |> put_reg(id, value)
            try_unify_fact_tuple(s2, tuple, i + 1, arity)

          bound when bound == value ->
            try_unify_fact_tuple(state, tuple, i + 1, arity)

          _ ->
            :fail
        end
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
        # length walks the list. The list may be either a native Elixir
        # list (driver-supplied, e.g. ["Classical_mechanics"]) or a
        # heap-built `./2` chain (produced by put_list when the
        # compiler constructs [Mid|Visited] from a mix of reg and heap
        # values). Mixed recursion handles both forms; the empty list
        # terminator is either Elixir `[]` or the atom string "[]".
        list = deref_var(state, get_reg(state, 1))
        len = wam_list_length(state, list)
        if len == :fail, do: throw({:fail, state})
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
        # member walks the list. Like length/2, handles both native
        # Elixir lists and heap-built `./2` chains produced by put_list.
        item = deref_var(state, get_reg(state, 1))
        list = deref_var(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if wam_list_member?(state, item, list) do
          %{state | pc: new_pc}
        else
          :fail
        end
      {"!/0", 0} ->
        # Cut: truncate choice_points back to the barrier set at
        # predicate entry (saved by `allocate` into state.cut_point).
        # Preserves caller CPs while clearing CPs pushed inside this
        # clause body.
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        %{state | choice_points: state.cut_point, pc: new_pc}
      _ -> :fail
    end
  end

  defp wam_list_length(_state, []), do: 0
  defp wam_list_length(_state, "[]"), do: 0
  defp wam_list_length(_state, list) when is_list(list), do: length(list)
  defp wam_list_length(state, {:ref, addr}) do
    case Map.get(state.heap, addr) do
      {:str, "./2"} ->
        [_h, tail] = heap_slice(state, addr + 1, 2)
        case wam_list_length(state, deref_var(state, tail)) do
          :fail -> :fail
          n -> n + 1
        end
      _ -> :fail
    end
  end
  defp wam_list_length(_state, _), do: :fail

  defp wam_list_member?(_state, _item, []), do: false
  defp wam_list_member?(_state, _item, "[]"), do: false
  defp wam_list_member?(_state, item, list) when is_list(list), do: item in list
  defp wam_list_member?(state, item, {:ref, addr}) do
    case Map.get(state.heap, addr) do
      {:str, "./2"} ->
        [h, tail] = heap_slice(state, addr + 1, 2)
        if deref_var(state, h) == item do
          true
        else
          wam_list_member?(state, item, deref_var(state, tail))
        end
      _ -> false
    end
  end
  defp wam_list_member?(_state, _item, _), do: false

  defp eval_arith(_state, n) when is_number(n), do: n
  defp eval_arith(_state, n) when is_binary(n) do
    # Try integer first and only accept a full-match parse — otherwise
    # `Integer.parse("1.5")` would swallow the `1` and drop the `.5`.
    # Fall back to float only when the integer parse leaves a remainder
    # (i.e. the input was `"1.5"` or `"3.14e2"`), not when the number is
    # genuinely integral like `"1"`. Previous order (Float first) turned
    # every integer head-constant into a float the moment `is/2`
    # touched it, breaking drivers that expected `is_integer(hops)`.
    case Integer.parse(n) do
      {i, ""} -> i
      _ ->
        case Float.parse(n) do
          {f, ""} -> f
          _ -> throw({:eval_error, n})
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

%% compile_aggregate_helpers_to_elixir(-Code)
%
%  Emits the Tier-2 aggregate-frame substrate: two runtime helpers that
%  consume aggregate-frame choice points but don\'t produce them. See
%  docs/design/WAM_TIERED_LOWERING.md for the full plan.
%
%  An "aggregate frame" is a choice point with an `:agg_type` field set
%  to one of `:findall | :bagof | :setof | :aggregate_all | :sum | :count`.
%  Aggregate wrappers (findall/3, bagof/3, etc.) push such a CP at entry
%  and consume it at exit, accumulating solutions in the CP\'s `:agg_accum`
%  list. Infrastructure PR: no wrapper pushes the marker yet; these
%  helpers are dead code until PR2 adds Tier-2 emission and a later PR
%  adds findall/bagof/aggregate_all.
%
%  `in_forkable_aggregate_frame?/1` answers "may a Tier-2 fan-out emit
%  under the current CP stack?" — true only if the nearest aggregate
%  frame is order-independent (findall/aggregate_all). bagof/setof have
%  witness-variable semantics that forbid naive parallelisation.
%
%  `merge_into_aggregate/2` takes a state and a branch-results list,
%  locates the nearest aggregate frame, and appends the results to its
%  accumulator. Returns the updated state.
compile_aggregate_helpers_to_elixir(Code) :-
    format(string(Code),
'  @doc """
  True if the nearest aggregate frame on the CP stack is forkable
  (findall / aggregate_all — both order-independent). Returns false
  for bagof/setof (witness-variable dependency) and for an empty or
  non-aggregate-bearing CP stack.

  Note on the alphabet: the WAM compiler\'s compile_aggregate_all/5
  emits `:collect` for findall/3 (via the collect-Template wrapper
  in compile_findall/5), but the Elixir target translates
  `collect → findall` at the begin_aggregate emission site
  (agg_type_atom/2 in wam_elixir_lowered_emitter.pl, per
  WAM_ELIXIR_TIER2_FINDALL.md §6.4). Consequently the only
  forkable atoms this function ever sees in emitted modules are
  `:findall` and `:aggregate_all` — `:collect` never reaches the
  runtime substrate.

  Consumed by the Tier-2 wrapper (`par_wrap_segment/4`, PR #1608)
  as a correctness gate: outside a forkable aggregate, parallel
  fan-out would strand solutions that the sequential
  enumeration path would otherwise surface.
  """
  def in_forkable_aggregate_frame?(state) do
    Enum.any?(state.choice_points, fn cp ->
      case Map.get(cp, :agg_type) do
        :findall -> true
        :aggregate_all -> true
        _ -> false
      end
    end)
  end

  @doc """
  Append branch results to the nearest aggregate frame\'s accumulator.
  Used by the Tier-2 wrapper after `Task.async_stream` fully exhausts
  all branches — each branch produces a list of solutions, and the
  wrapper hands the flattened list to this function before returning
  the updated state to the aggregate-wrapper continuation.

  If no aggregate frame is present (caller misuse — Tier-2 gate should
  have prevented this), the state is returned unchanged.
  """
  def merge_into_aggregate(state, branch_results) when is_list(branch_results) do
    {updated_cps, _merged} =
      Enum.map_reduce(state.choice_points, false, fn cp, merged ->
        cond do
          merged -> {cp, merged}
          Map.has_key?(cp, :agg_type) ->
            prior = Map.get(cp, :agg_accum, [])
            {Map.put(cp, :agg_accum, prior ++ branch_results), true}
          true -> {cp, merged}
        end
      end)
    %{state | choice_points: updated_cps}
  end

  @doc """
  Push an aggregate-frame choice point. Captures the same snapshot
  fields as a try_me_else CP (regs, heap, heap_len, trail, trail_len,
  stack, cp) so finalise_aggregate/4 can restore the pre-aggregate
  state, plus four aggregate-specific fields:

    - :agg_type        — :collect|:findall|:aggregate_all|:sum|:count|
                         :max|:min|:bag|:set (the alphabet emitted by
                         compile_aggregate_all/5 in wam_target.pl).
    - :agg_value_reg   — register holding the per-solution Template
                         value at end_aggregate time.
    - :agg_result_reg  — register that finalise binds the aggregated
                         value to.
    - :agg_accum       — accumulator, prepended to (O(1)) by
                         aggregate_collect/2; reversed at finalise.
                         merge_into_aggregate/2 also writes here for
                         the Tier-2 parallel-fan-out path.

  state.cp is captured as :cp; finalise restores it then tail-calls
  via the restored state.cp (no separate :agg_return_cp field — the
  proposal §4.1 listed one, but they would always equal :cp at push
  time, so the duplication was dropped during implementation).
  The .pc field is set to nil because aggregate frames have no
  failure target — backtrack/1 routes them to finalise instead of
  resuming a clause.
  """
  def push_aggregate_frame(state, agg_type, value_reg, result_reg) do
    cp = %{
      pc: nil,
      regs: state.regs,
      heap: state.heap,
      heap_len: state.heap_len,
      cp: state.cp,
      trail: state.trail,
      trail_len: state.trail_len,
      stack: state.stack,
      agg_type: agg_type,
      agg_value_reg: value_reg,
      agg_result_reg: result_reg,
      agg_accum: []
    }
    %{state | choice_points: [cp | state.choice_points]}
  end

  @doc """
  Per-solution collector for sequential fail-driven enumeration.
  Reads value_reg, derefs it, prepends to the nearest aggregate
  frame\'s :agg_accum (O(1)). Returns updated state.

  Atomic values (strings/numbers/atoms) survive trail unwind without
  copy because they\'re value types in BEAM. Compound values that
  reference heap cells (e.g. lists, structures) are a known
  follow-up — see WAM_ELIXIR_TIER2_FINDALL.md §6 risk #1. For Phase
  1 substrate the simple cases (`findall(X, member(X, [a,b,c]), L)`,
  `findall(X, p(X), L)` over fact predicates) are unaffected.

  If no aggregate frame is present, returns state unchanged (caller
  misuse — emitter should always pair end_aggregate with begin).

  Walk cost: O(N) in choice-point depth per call. Acceptable for
  Phase 1; if Phase 3 profiling shows this dominates, an
  agg_frame_idx field on state can collapse it to O(1).
  """
  def aggregate_collect(state, value_reg) do
    val = deref_var(state, Map.get(state.regs, value_reg))
    {updated_cps, _collected} =
      Enum.map_reduce(state.choice_points, false, fn cp, collected ->
        cond do
          collected -> {cp, collected}
          Map.has_key?(cp, :agg_type) ->
            prior = Map.get(cp, :agg_accum, [])
            {Map.put(cp, :agg_accum, [val | prior]), true}
          true -> {cp, collected}
        end
      end)
    %{state | choice_points: updated_cps}
  end

  @doc """
  Reduce :agg_accum per :agg_type, bind the result to :agg_result_reg,
  restore the pre-aggregate snapshot, and tail-call the saved
  continuation (the restored state.cp). Called by backtrack/1 when
  the popped CP carries :agg_type.

  :collect / :findall / :aggregate_all → reversed list (preserves
  enumeration order for sequential; non-deterministic for the
  parallel Tier-2 path, which is acceptable because both forkable
  aggregators are order-independent by definition).

  :sum / :count / :max / :min mirror compile_aggregate_all/5\'s
  alphabet. :bag is collect-with-duplicates; :set deduplicates.

  Empty-accumulator semantics:
    :collect/:findall/:aggregate_all/:bag/:set → []
    :sum   → 0     (Enum.sum identity)
    :count → 0     (length identity)
    :max/:min → throws {:fail, state} — no identity exists, and
                returning nil would silently propagate as a non-WAM
                value into downstream get_constant unification. Fail
                is the canonical Prolog semantics for max/min over
                an empty bag.
  """
  def finalise_aggregate(state, agg_cp, rest_cps, agg_type) do
    accum_rev = Enum.reverse(agg_cp.agg_accum)
    result =
      case agg_type do
        t when t in [:collect, :findall, :aggregate_all, :bag] -> accum_rev
        :set -> Enum.uniq(accum_rev)
        :sum -> Enum.sum(accum_rev)
        :count -> length(accum_rev)
        :max ->
          if accum_rev == [], do: throw({:fail, state}), else: Enum.max(accum_rev)
        :min ->
          if accum_rev == [], do: throw({:fail, state}), else: Enum.min(accum_rev)
      end
    restored = %{state |
      regs: Map.put(agg_cp.regs, agg_cp.agg_result_reg, result),
      heap: agg_cp.heap,
      heap_len: agg_cp.heap_len,
      trail: agg_cp.trail,
      trail_len: agg_cp.trail_len,
      stack: agg_cp.stack,
      cp: agg_cp.cp,
      choice_points: rest_cps
    }
    restored.cp.(restored)
  end', []).

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3
% ============================================================================

%% compile_wam_runtime_to_elixir(+Options, -Code)
compile_wam_runtime_to_elixir(Options, Code) :-
    compile_step_wam_to_elixir(Options, StepCode),
    compile_wam_helpers_to_elixir(Options, HelpersCode),
    compile_fact_source_runtime_to_elixir(FactSourceCode),
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
              choice_points: [], stack: [], code: {}, labels: %{},
              # arg_vars is the list of {reg_id, unbound_tuple} for each
              # caller-supplied unbound arg. Set by run(args) at entry so
              # next_solution/1 can rematerialise output regs on every
              # solution (A-regs get clobbered as scratch during clauses).
              arg_vars: [],
              # cut_point is the choice_points snapshot to restore on `!`.
              # Saved to the env frame by `allocate`; restored by
              # `deallocate`. A cut truncates state.choice_points back to
              # this snapshot — clearing CPs pushed inside the current
              # predicate body without touching the caller\'s CPs.
              cut_point: [],
              # parallel_depth counts how many Tier-2 parallel fan-outs
              # are currently active on this branch of the search. The
              # Tier-2 wrapper increments it when spawning Task.async_stream
              # children and checks > 0 to short-circuit back to
              # sequential — prevents B^D spark explosion on recursive
              # predicates. See docs/design/WAM_TIERED_LOWERING.md.
              parallel_depth: 0
  end

~w

~w
end

~w', [StepCode, HelpersCode, FactSourceCode]).

%% compile_fact_source_runtime_to_elixir(-Code)
%  Emits the Phase-D external_source runtime: FactSource behaviour,
%  FactSource.Tsv adaptor, and FactSourceRegistry. Generic enough
%  that future adaptors (SQLite, ETS, hash-table artifacts, etc.)
%  plug in by implementing the behaviour — mirrors C# query runtime\'s
%  IRetentionAwareRelationProvider + preprocessed-artifact direction.
compile_fact_source_runtime_to_elixir(Code) :-
    format(string(Code),
'defmodule WamRuntime.FactSource do
  @moduledoc """
  Logical access contract for fact-only predicates backed by external
  data. TSV today; SQLite / ETS / hash-table artifacts tomorrow.

  Drivers register a concrete source for a predicate before calling
  its `run/1`. The generated `external_source` layout looks the
  source up by predicate indicator, then dispatches through this
  module\'s facade which forwards to the struct\'s module.
  """

  @callback open(term(), pos_integer(), term()) :: struct()
  @callback stream_all(struct(), term()) :: Enumerable.t()
  @callback lookup_by_arg1(struct(), term(), term()) :: Enumerable.t()
  @callback close(struct(), term()) :: :ok

  @optional_callbacks close: 2

  # Struct-based dispatch so the caller doesn\'t need to know which
  # concrete module implements the behaviour.
  def stream_all(%module{} = handle, state), do: module.stream_all(handle, state)
  def lookup_by_arg1(%module{} = handle, key, state), do: module.lookup_by_arg1(handle, key, state)
  def close(%module{} = handle, state) do
    if function_exported?(module, :close, 2), do: module.close(handle, state), else: :ok
  end
end

defmodule WamRuntime.FactSource.Tsv do
  @moduledoc """
  Two-column TSV fact source. Loads the file eagerly on `open/3`,
  caches rows as a tuple list and a first-arg index map. Later
  adaptors may stream or memory-map; the behaviour contract does
  not require eager retention.

  Spec fields:
    path    — path to the TSV file (required)
    arity   — number of columns per row (required; only 2 supported)
    header  — :skip (default) discards the first line; :none keeps it
  """
  @behaviour WamRuntime.FactSource
  defstruct [:path, :arity, :rows, :by_arg1]

  @impl true
  def open(%{path: path, arity: arity} = spec, _pred_arity, _state) when arity == 2 do
    header_mode = Map.get(spec, :header, :skip)

    stream = File.stream!(path)

    stream =
      case header_mode do
        :skip -> Stream.drop(stream, 1)
        :none -> stream
      end

    rows =
      stream
      |> Enum.map(fn line ->
        [a, b] = line |> String.trim_trailing("\n") |> String.split("\t", parts: 2)
        {a, b}
      end)

    by_arg1 =
      rows
      |> Enum.reverse()
      |> Enum.reduce(%{}, fn {a, _b} = tuple, acc ->
        Map.update(acc, a, [tuple], &[tuple | &1])
      end)

    %__MODULE__{path: path, arity: arity, rows: rows, by_arg1: by_arg1}
  end

  @impl true
  def stream_all(%__MODULE__{rows: rows}, _state), do: rows

  @impl true
  def lookup_by_arg1(%__MODULE__{by_arg1: idx}, key, _state) do
    Map.get(idx, key, [])
  end

  @impl true
  def close(_handle, _state), do: :ok
end

defmodule WamRuntime.FactSource.Ets do
  @moduledoc """
  ETS-table fact source. Lightweight second adaptor that proves the
  FactSource behaviour is generic — zero external deps (ETS ships
  with OTP) and a storage shape meaningfully different from the
  TSV adaptor\'s list-of-tuples. The driver populates the table
  before registering the source; the adaptor just wraps the table
  reference and forwards lookups.

  Spec fields:
    table   — ETS table identifier or named atom (required)
    arity   — number of columns per tuple (required; only 2 supported)

  Keying convention: arg1 is the ETS key. Use a :bag table if a
  single arg1 can map to multiple tuples (e.g. `category_parent/2`);
  :set tables are fine for unique-key predicates.
  """
  @behaviour WamRuntime.FactSource
  defstruct [:table, :arity]

  @impl true
  def open(%{table: table, arity: arity}, _pred_arity, _state) when arity == 2 do
    %__MODULE__{table: table, arity: arity}
  end

  @impl true
  def stream_all(%__MODULE__{table: table}, _state), do: :ets.tab2list(table)

  @impl true
  def lookup_by_arg1(%__MODULE__{table: table}, key, _state) do
    :ets.lookup(table, key)
  end

  @impl true
  def close(_handle, _state), do: :ok
end

defmodule WamRuntime.FactSource.Sqlite do
  @moduledoc """
  SQLite fact source. Disk-backed artifact adaptor — the
  "preprocessed-artifact" shape `PREPROCESSED_PREDICATE_ARTIFACTS.md`
  describes for the C# side, but for Elixir. Predicates too large to
  inline can live in a .sqlite file; the runtime prepares the two
  queries once per source and steps them per call.

  **Driver must add `:exqlite` to its own mix deps** before loading
  this generated runtime. To keep the runtime dep-free for drivers
  that don\'t use SQLite, this module references `Exqlite.Sqlite3`
  indirectly via `Module.concat/1`; the reference is resolved at
  call time, so compilation does not fail if `:exqlite` is absent
  (you just get an `UndefinedFunctionError` when the adaptor is
  actually used).

  Spec fields:
    path          — path to the .sqlite file (required)
    query_all     — SQL returning all rows as 2-column result set
                    (required; e.g. "SELECT a, b FROM cp")
    query_by_arg1 — SQL with one bound parameter returning matching
                    rows (required; e.g. "SELECT a, b FROM cp WHERE a = ?1")
    arity         — number of columns per tuple (required; only 2 supported)
  """
  @behaviour WamRuntime.FactSource
  defstruct [:db, :query_all, :query_by_arg1, :arity]

  # Resolve Exqlite.Sqlite3 indirectly so the runtime still compiles
  # when the driver hasn\'t added :exqlite. Callers who never touch
  # this adaptor pay zero cost; callers who do get a clear
  # UndefinedFunctionError at open/3.
  defp sqlite3_module, do: Module.concat([Exqlite, Sqlite3])

  @impl true
  def open(%{path: path, query_all: q_all, query_by_arg1: q_by1, arity: arity},
           _pred_arity, _state) when arity == 2 do
    mod = sqlite3_module()
    {:ok, db} = apply(mod, :open, [path])
    %__MODULE__{db: db, query_all: q_all, query_by_arg1: q_by1, arity: arity}
  end

  @impl true
  def stream_all(%__MODULE__{db: db, query_all: q_all}, _state) do
    mod = sqlite3_module()
    {:ok, stmt} = apply(mod, :prepare, [db, q_all])
    try do
      collect_rows(mod, db, stmt, [])
    after
      apply(mod, :release, [db, stmt])
    end
  end

  @impl true
  def lookup_by_arg1(%__MODULE__{db: db, query_by_arg1: q_by1}, key, _state) do
    mod = sqlite3_module()
    {:ok, stmt} = apply(mod, :prepare, [db, q_by1])
    try do
      :ok = apply(mod, :bind, [db, stmt, [key]])
      collect_rows(mod, db, stmt, [])
    after
      apply(mod, :release, [db, stmt])
    end
  end

  @impl true
  def close(%__MODULE__{db: db}, _state) do
    apply(sqlite3_module(), :close, [db])
    :ok
  end

  defp collect_rows(mod, db, stmt, acc) do
    case apply(mod, :step, [db, stmt]) do
      :done -> Enum.reverse(acc)
      {:row, row} -> collect_rows(mod, db, stmt, [List.to_tuple(row) | acc])
      _other -> Enum.reverse(acc)
    end
  end
end

defmodule WamRuntime.FactSourceRegistry do
  @moduledoc """
  Predicate-indicator → source-handle map. Uses :persistent_term so
  lookups are O(1) and lock-free from any process.
  """

  def register(pred_indicator, source) when is_binary(pred_indicator) do
    :persistent_term.put({__MODULE__, pred_indicator}, source)
  end

  def lookup(pred_indicator) when is_binary(pred_indicator) do
    :persistent_term.get({__MODULE__, pred_indicator}, nil)
  end

  def lookup!(pred_indicator) do
    case lookup(pred_indicator) do
      nil ->
        raise "No FactSource registered for #{pred_indicator}. " <>
              "Call WamRuntime.FactSourceRegistry.register/2 before calling Pred.run/1."

      source ->
        source
    end
  end

  def unregister(pred_indicator) do
    :persistent_term.erase({__MODULE__, pred_indicator})
  end
end', []).

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
    # Rewrite caller unbounds to fresh refs + track them in state.arg_vars
    # (see lowered emitter run(args) for the full rationale).
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
    case WamRuntime.run(state) do
      {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
      other -> other
    end
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
  # Fallback: meta-calls (e.g. \\+ Goal) may target builtins whose
  # module the compiler didn\'t register. Try the builtin table before
  # declaring the predicate undefined.
  def call(pred, state) do
    arity = WamRuntime.parse_functor_arity(pred)
    case WamRuntime.execute_builtin(state, pred, arity) do
      :fail -> :fail
      new_state when is_map(new_state) -> {:ok, new_state}
      _ -> throw({:undefined_predicate, pred})
    end
  end
end', [CasesStr]).
