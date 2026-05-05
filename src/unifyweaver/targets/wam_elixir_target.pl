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
:- use_module('../targets/wam_elixir_utils', [camel_case/2]).
:- use_module('../core/recursive_kernel_detection',
              [detect_recursive_kernel/4, kernel_config/2]).
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
        #
        # In Tier-2 parallel-branch context (branch_mode: true), the
        # branchs PARENT agg CP returns its local accum to the parent
        # instead of finalising — the parent merges branch contributions
        # before finalising on its own state. The parent agg CP is
        # tagged with :branch_sentinel at branch entry (Phase 4d);
        # NESTED agg CPs pushed by inner findalls inside the branchs
        # clause body lack the sentinel and finalise normally. Without
        # this distinction, a nested findalls accum would leak up as
        # the branchs return value. See branch_backtrack/1 and
        # WAM_ELIXIR_TIER2_FINDALL_PHASE4.md sections 4 and 4.6.
        case Map.get(cp, :agg_type) do
          nil ->
            backtrack_ordinary(state, cp, rest)
          agg_type ->
            cond do
              Map.get(state, :branch_mode, false) and
                  Map.get(cp, :branch_sentinel, false) ->
                {:branch_exhausted, Enum.reverse(Map.get(cp, :agg_accum, []))}
              true ->
                finalise_aggregate(state, cp, rest, agg_type)
            end
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
      {"=/2", 2} ->
        # Body unification: X = Y. Reuses unify/3 — the same machinery
        # the WAM head-unification instructions call into. On success,
        # advance pc and keep the bindings; on failure, return :fail
        # so the surrounding catch can backtrack.
        v1 = get_reg(state, 1)
        v2 = get_reg(state, 2)
        case unify(state, v1, v2) do
          {:ok, new_state} ->
            new_pc = if is_integer(new_state.pc), do: new_state.pc + 1, else: new_state.pc
            %{new_state | pc: new_pc}
          :fail -> :fail
        end
      {"\\\\=/2", 2} ->
        # Body negation-of-unify (the \\=/2 op). Tests unify without
        # committing — Elixirs immutability means we just discard the
        # post-unify state on the success path. If unify succeeds the
        # two args *would* unify, so the goal fails; if unify fails,
        # the goal holds and we advance pc with the original
        # (untouched) state.
        v1 = get_reg(state, 1)
        v2 = get_reg(state, 2)
        case unify(state, v1, v2) do
          {:ok, _discarded} -> :fail
          :fail ->
            new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
            %{state | pc: new_pc}
        end
      {"fail/0", 0} ->
        # Explicit Prolog fail. The default-arm hardening below throws
        # {:unknown_builtin, ...} for un-handled ops; without an
        # explicit arm here, fail/0 would have hit the throw arm and
        # surface as a crash instead of an ordinary backtrack-driving
        # failure. (Pre-hardening, fail/0 worked accidentally because
        # the default arm returned :fail — same semantic, wrong reason.)
        :fail
      {"</2", 2} ->
        v1 = eval_arith(state, get_reg(state, 1))
        v2 = eval_arith(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if v1 < v2, do: %{state | pc: new_pc}, else: :fail
      {">/2", 2} ->
        v1 = eval_arith(state, get_reg(state, 1))
        v2 = eval_arith(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if v1 > v2, do: %{state | pc: new_pc}, else: :fail
      {"=</2", 2} ->
        v1 = eval_arith(state, get_reg(state, 1))
        v2 = eval_arith(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if v1 <= v2, do: %{state | pc: new_pc}, else: :fail
      {">=/2", 2} ->
        v1 = eval_arith(state, get_reg(state, 1))
        v2 = eval_arith(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if v1 >= v2, do: %{state | pc: new_pc}, else: :fail
      {"=:=/2", 2} ->
        v1 = eval_arith(state, get_reg(state, 1))
        v2 = eval_arith(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if v1 == v2, do: %{state | pc: new_pc}, else: :fail
      {"=\\\\=/2", 2} ->
        v1 = eval_arith(state, get_reg(state, 1))
        v2 = eval_arith(state, get_reg(state, 2))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        if v1 != v2, do: %{state | pc: new_pc}, else: :fail
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
      {"append/3", 3} ->
        # Deterministic append: walk arg1 + arg2 into a native list,
        # unify with arg3. Native-list output works against unbound
        # arg3 (the common case) and against an arg3 that is itself
        # a native list of the same shape. Doesnt work against arg3
        # being a heap-built ./2 chain — unify/3 only does shallow
        # comparison and the structural equality check fails. Filed
        # as an audit-known limitation; covers the typical usage.
        l1 = deref_var(state, get_reg(state, 1))
        l2 = deref_var(state, get_reg(state, 2))
        case wam_list_to_native(state, l1) do
          {:ok, n1} ->
            case wam_list_to_native(state, l2) do
              {:ok, n2} ->
                appended = n1 ++ n2
                case unify(state, get_reg(state, 3), appended) do
                  {:ok, new_state} ->
                    new_pc = if is_integer(new_state.pc), do: new_state.pc + 1, else: new_state.pc
                    %{new_state | pc: new_pc}
                  :fail -> :fail
                end
              :fail -> :fail
            end
          :fail -> :fail
        end
      {"write/1", 1} ->
        v = deref_var(state, get_reg(state, 1))
        IO.write(format_term_for_write(v))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        %{state | pc: new_pc}
      {"nl/0", 0} ->
        IO.puts("")
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        %{state | pc: new_pc}
      {"format/1", 1} ->
        # format(Atom) — atom is the format string, no args.
        fstr = deref_var(state, get_reg(state, 1))
        IO.write(prolog_format(state, fstr, []))
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        %{state | pc: new_pc}
      {"format/2", 2} ->
        # format(Atom, Args) — atom is the format string, Args is a list.
        fstr = deref_var(state, get_reg(state, 1))
        args = deref_var(state, get_reg(state, 2))
        case wam_list_to_native(state, args) do
          {:ok, native_args} ->
            IO.write(prolog_format(state, fstr, native_args))
            new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
            %{state | pc: new_pc}
          :fail -> :fail
        end
      {"functor/3", 3} ->
        # functor(Term, Name, Arity). Two modes:
        #
        # Decompose: Term is bound — extract Name and Arity. Atomic
        # terms have arity 0 and report themselves as the name.
        # Compound terms unpack from their heap-side {:str, "F/N"}
        # cell.
        #
        # Build: Term is unbound, Name + Arity are bound — allocate
        # a fresh compound on the heap with `arity` unbound argument
        # cells, bind Term to its ref. Atomic-result case (arity 0)
        # binds Term directly to the Name value.
        term = deref_var(state, get_reg(state, 1))
        case term do
          {:unbound, _} = unb ->
            name = deref_var(state, get_reg(state, 2))
            arity = deref_var(state, get_reg(state, 3))
            case build_functor_term(state, name, arity) do
              {:ok, state2, term_val} ->
                case unify(state2, unb, term_val) do
                  {:ok, s} ->
                    new_pc = if is_integer(s.pc), do: s.pc + 1, else: s.pc
                    %{s | pc: new_pc}
                  :fail -> :fail
                end
              :fail -> :fail
            end
          _ ->
            case term_functor_arity(state, term) do
              {:ok, fname, farity} ->
                case unify(state, get_reg(state, 2), fname) do
                  {:ok, s1} ->
                    case unify(s1, get_reg(s1, 3), farity) do
                      {:ok, s2} ->
                        new_pc = if is_integer(s2.pc), do: s2.pc + 1, else: s2.pc
                        %{s2 | pc: new_pc}
                      :fail -> :fail
                    end
                  :fail -> :fail
                end
              :fail -> :fail
            end
        end
      {"arg/3", 3} ->
        # arg(N, Term, Arg). Read mode: N must be a bound positive
        # integer, Term must be a bound compound. heap[Term-addr + N]
        # is the Nth arg slot; deref + unify with Arg.
        n = deref_var(state, get_reg(state, 1))
        term = deref_var(state, get_reg(state, 2))
        case {n, term} do
          {n, {:ref, addr}} when is_integer(n) and n >= 1 ->
            case Map.get(state.heap, addr) do
              {:str, _fn} ->
                arg_val = deref_var(state, Map.get(state.heap, addr + n))
                case unify(state, get_reg(state, 3), arg_val) do
                  {:ok, s} ->
                    new_pc = if is_integer(s.pc), do: s.pc + 1, else: s.pc
                    %{s | pc: new_pc}
                  :fail -> :fail
                end
              _ -> :fail
            end
          _ -> :fail
        end
      {"=../2", 2} ->
        # Univ. Two modes:
        #
        # Decompose: Term bound — Term =.. List builds [Functor | Args]
        # as a native list. Atomic Term decomposes to [Term].
        #
        # Compose: Term unbound, List bound — first list element is
        # the functor name, rest are args. Build a fresh compound
        # on the heap and bind Term to its ref. Singleton list
        # ([Atom]) binds Term directly to the atom.
        term = deref_var(state, get_reg(state, 1))
        case term do
          {:unbound, _} = unb ->
            list = deref_var(state, get_reg(state, 2))
            case wam_list_to_native(state, list) do
              {:ok, [name]} ->
                # Singleton: Term = Name (atomic).
                case unify(state, unb, name) do
                  {:ok, s} ->
                    new_pc = if is_integer(s.pc), do: s.pc + 1, else: s.pc
                    %{s | pc: new_pc}
                  :fail -> :fail
                end
              {:ok, [name | args]} ->
                case build_compound_term(state, name, args) do
                  {:ok, state2, term_val} ->
                    case unify(state2, unb, term_val) do
                      {:ok, s} ->
                        new_pc = if is_integer(s.pc), do: s.pc + 1, else: s.pc
                        %{s | pc: new_pc}
                      :fail -> :fail
                    end
                  :fail -> :fail
                end
              _ -> :fail
            end
          _ ->
            result =
              case term do
                {:ref, addr} ->
                  case Map.get(state.heap, addr) do
                    {:str, fn_name} ->
                      fname = parse_functor_name(fn_name)
                      farity = parse_functor_arity(fn_name)
                      args = if farity > 0 do
                        Enum.map(heap_slice(state, addr + 1, farity),
                                 &deref_var(state, &1))
                      else
                        []
                      end
                      [fname | args]
                    _ -> :fail
                  end
                v when is_number(v) or is_binary(v) or is_atom(v) -> [v]
                _ -> :fail
              end
            case result do
              :fail -> :fail
              list ->
                case unify(state, get_reg(state, 2), list) do
                  {:ok, s} ->
                    new_pc = if is_integer(s.pc), do: s.pc + 1, else: s.pc
                    %{s | pc: new_pc}
                  :fail -> :fail
                end
            end
        end
      {"copy_term/2", 2} ->
        # copy_term(Original, Copy). Walks Originals heap structure,
        # allocating a fresh copy with renamed variables. Sharing is
        # preserved: occurrences of the SAME unbound var in the input
        # map to the SAME fresh var in the output (e.g., copy_term of
        # p(X, X) gives p(Z, Z), not p(Z1, Z2)). Atomic values pass
        # through unchanged — theyre immutable and have no var
        # identity to rename.
        original = deref_var(state, get_reg(state, 1))
        {state2, copy_val, _vmap} = deep_copy_with_fresh_vars(state, original, %{})
        case unify(state2, get_reg(state2, 2), copy_val) do
          {:ok, s} ->
            new_pc = if is_integer(s.pc), do: s.pc + 1, else: s.pc
            %{s | pc: new_pc}
          :fail -> :fail
        end
      {"!/0", 0} ->
        # Cut: truncate choice_points back to the barrier set at
        # predicate entry (saved by `allocate` into state.cut_point).
        # Preserves caller CPs while clearing CPs pushed inside this
        # clause body.
        #
        # Aggregate frames between the current top and cut_point are
        # PRESERVED as additional cut barriers. Without this, cut
        # inside a findall body (or in a sub-predicate called from
        # findall after deallocate restores cut_point to the outer
        # level) removes the agg frame, causing end_aggregates throw
        # fail to propagate without finalisation. Closes proposal
        # section 6 risk 3 — cut x findall interaction.
        new_pc = if is_integer(state.pc), do: state.pc + 1, else: state.pc
        cut_target_len = length(state.cut_point)
        current_len = length(state.choice_points)
        new_cps =
          if current_len <= cut_target_len do
            state.choice_points
          else
            above_cut_count = current_len - cut_target_len
            above_cut = Enum.take(state.choice_points, above_cut_count)
            preserved_aggs = Enum.filter(above_cut, &Map.has_key?(&1, :agg_type))
            preserved_aggs ++ state.cut_point
          end
        %{state | choice_points: new_cps, pc: new_pc}
      _ ->
        # Hardening: surface unimplemented builtins instead of silently
        # returning :fail. The pre-hardening default arm masked
        # missing-builtin bugs as ordinary clause failures (see
        # benchmarks/wam_elixir_builtin_coverage.md). Throw a tagged
        # tuple so the error catch-all in run/1 reports it clearly
        # rather than collapsing to :fail. Real Prolog `fail` has its
        # own explicit `{"fail/0", 0}` arm above; new builtins should
        # be added rather than relying on the silent-:fail accident.
        throw({:unknown_builtin, op, arity})
    end
  end

  # Extract the functor-name part from a "name/arity" string.
  defp parse_functor_name(fn_name) do
    case String.split(fn_name, "/") do
      [name, _arity_str] -> name
      _ -> fn_name
    end
  end

  # Compute (Name, Arity) for a derefd term — used by functor/3.
  # Atomic values (numbers, strings/atoms) report themselves as the
  # name with arity 0. Compound terms unpack from the {:str, F/N}
  # heap cell. Unbound and unrecognised shapes return :fail (build
  # mode is not yet supported).
  defp term_functor_arity(state, {:ref, addr}) do
    case Map.get(state.heap, addr) do
      {:str, fn_name} ->
        {:ok, parse_functor_name(fn_name), parse_functor_arity(fn_name)}
      _ -> :fail
    end
  end
  defp term_functor_arity(_state, v) when is_number(v), do: {:ok, v, 0}
  defp term_functor_arity(_state, v) when is_binary(v), do: {:ok, v, 0}
  defp term_functor_arity(_state, v) when is_atom(v) and not is_nil(v), do: {:ok, v, 0}
  defp term_functor_arity(_state, _), do: :fail

  # Build a fresh compound term with `arity` unbound argument cells
  # for functor/3 in build mode. Atomic-result case (arity 0) just
  # returns the name as the term value. Returns {:ok, state, term}
  # on success or :fail when args are mistyped.
  defp build_functor_term(state, name, 0)
       when is_binary(name) or is_atom(name) or is_number(name) do
    {:ok, state, name}
  end
  defp build_functor_term(state, name, arity)
       when is_integer(arity) and arity > 0 and (is_binary(name) or is_atom(name)) do
    name_str = if is_atom(name), do: Atom.to_string(name), else: name
    fn_name = "#{name_str}/#{arity}"
    addr = state.heap_len
    heap1 = Map.put(state.heap, addr, {:str, fn_name})
    {final_heap, _} =
      Enum.reduce(1..arity, {heap1, addr + 1}, fn _, {h, a} ->
        {Map.put(h, a, {:unbound, {:heap_ref, a}}), a + 1}
      end)
    new_state = %{state | heap: final_heap, heap_len: addr + 1 + arity}
    {:ok, new_state, {:ref, addr}}
  end
  defp build_functor_term(_state, _, _), do: :fail

  # Build a fresh compound term from a functor name and a list of
  # already-evaluated argument values for =../2 in compose mode.
  defp build_compound_term(state, name, args)
       when (is_binary(name) or is_atom(name)) and is_list(args) do
    arity = length(args)
    name_str = if is_atom(name), do: Atom.to_string(name), else: name
    fn_name = "#{name_str}/#{arity}"
    addr = state.heap_len
    heap1 = Map.put(state.heap, addr, {:str, fn_name})
    {final_heap, _} =
      Enum.reduce(args, {heap1, addr + 1}, fn arg, {h, a} ->
        {Map.put(h, a, arg), a + 1}
      end)
    new_state = %{state | heap: final_heap, heap_len: addr + 1 + arity}
    {:ok, new_state, {:ref, addr}}
  end
  defp build_compound_term(_state, _, _), do: :fail

  # Walk a term, allocating a fresh copy with renamed variables for
  # copy_term/2. The var_map argument carries the in-progress mapping
  # from input-var-id to output-fresh-var so the same input var
  # consistently maps to the same fresh output var (sharing
  # preserved). Returns {state, copy, updated_var_map}.
  defp deep_copy_with_fresh_vars(state, {:unbound, id}, var_map) do
    case Map.get(var_map, id) do
      nil ->
        fresh = {:unbound, make_ref()}
        {state, fresh, Map.put(var_map, id, fresh)}
      existing ->
        {state, existing, var_map}
    end
  end
  defp deep_copy_with_fresh_vars(state, {:ref, addr}, var_map) do
    case Map.get(state.heap, addr) do
      {:str, fn_name} ->
        arity = parse_functor_arity(fn_name)
        # Recursively copy each arg, threading state and var_map.
        {args_rev, state2, vm2} =
          if arity > 0 do
            Enum.reduce(1..arity, {[], state, var_map}, fn i, {acc, s, vm} ->
              raw = Map.get(s.heap, addr + i)
              arg_val = deref_var(s, raw)
              {s_next, copy, vm_next} = deep_copy_with_fresh_vars(s, arg_val, vm)
              {[copy | acc], s_next, vm_next}
            end)
          else
            {[], state, var_map}
          end
        args = Enum.reverse(args_rev)
        new_addr = state2.heap_len
        heap1 = Map.put(state2.heap, new_addr, {:str, fn_name})
        {final_heap, _} =
          Enum.reduce(args, {heap1, new_addr + 1}, fn a_val, {h, a} ->
            {Map.put(h, a, a_val), a + 1}
          end)
        new_state = %{state2 | heap: final_heap, heap_len: new_addr + 1 + arity}
        {new_state, {:ref, new_addr}, vm2}
      _ ->
        {state, {:ref, addr}, var_map}
    end
  end
  defp deep_copy_with_fresh_vars(state, val, var_map), do: {state, val, var_map}

  defp wam_list_to_native(_state, []), do: {:ok, []}
  defp wam_list_to_native(_state, "[]"), do: {:ok, []}
  defp wam_list_to_native(_state, list) when is_list(list), do: {:ok, list}
  defp wam_list_to_native(state, {:ref, addr}) do
    # Cons cells get tagged either `./2` (early put_list emit) or
    # `[|]/2` (later set_value emit) depending on which lowering arm
    # produced them. Both shapes appear in lists built inline in a
    # clause body — the existing wam_list_length/wam_list_member?
    # check only `./2`, which is a separate audit-noted limitation.
    # We accept both to make append/3 work end-to-end on inline lists.
    case Map.get(state.heap, addr) do
      {:str, cons} when cons in ["./2", "[|]/2"] ->
        [h, tail] = heap_slice(state, addr + 1, 2)
        head_val = deref_var(state, h)
        case wam_list_to_native(state, deref_var(state, tail)) do
          {:ok, rest} -> {:ok, [head_val | rest]}
          :fail -> :fail
        end
      _ -> :fail
    end
  end
  defp wam_list_to_native(_state, _), do: :fail

  defp format_term_for_write({:str, name}), do: name
  defp format_term_for_write({:ref, _} = ref), do: inspect(ref)
  defp format_term_for_write({:unbound, _}), do: "_"
  defp format_term_for_write(v) when is_binary(v), do: v
  defp format_term_for_write(v) when is_number(v), do: to_string(v)
  defp format_term_for_write(v) when is_atom(v), do: Atom.to_string(v)
  defp format_term_for_write(v) when is_list(v) do
    "[" <> Enum.map_join(v, ",", &format_term_for_write/1) <> "]"
  end
  defp format_term_for_write(v), do: inspect(v)

  # Minimal Prolog format/2 implementation. Supports `~~w` (write next
  # arg), `~~a` (atom), `~~d` (integer), `~~s` (string), `~~n` (newline).
  # Unknown directives pass through verbatim — better to print weird
  # output than to crash a debug-line.
  defp prolog_format(state, fmt, args) when is_binary(fmt) do
    do_format(state, String.graphemes(fmt), args, [])
    |> Enum.reverse()
    |> Enum.join("")
  end
  defp prolog_format(state, fmt, args) when is_list(fmt) do
    prolog_format(state, IO.iodata_to_binary(fmt), args)
  end
  defp prolog_format(state, fmt, args) do
    prolog_format(state, format_term_for_write(fmt), args)
  end
  defp do_format(_state, [], _args, acc), do: acc
  defp do_format(state, ["~~", "n" | rest], args, acc) do
    do_format(state, rest, args, ["\n" | acc])
  end
  defp do_format(state, ["~~", d | rest], [arg | args_rest], acc)
       when d in ["w", "a", "d", "s", "p"] do
    do_format(state, rest, args_rest, [format_term_for_write(deref_var(state, arg)) | acc])
  end
  defp do_format(state, [c | rest], args, acc) do
    do_format(state, rest, args, [c | acc])
  end

  defp wam_list_length(_state, []), do: 0
  defp wam_list_length(_state, "[]"), do: 0
  defp wam_list_length(_state, list) when is_list(list), do: length(list)
  defp wam_list_length(state, {:ref, addr}) do
    # Cons cells get tagged either `./2` (early put_list emit) or
    # `[|]/2` (later put_structure emit) depending on which lowering
    # arm produced them. Both shapes appear in inline list literals
    # built in a clause body. Accept both so list-walking builtins
    # work end-to-end on the mixed-functor chains.
    case Map.get(state.heap, addr) do
      {:str, cons} when cons in ["./2", "[|]/2"] ->
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
      {:str, cons} when cons in ["./2", "[|]/2"] ->
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
  Per-value aggregator entry point — same role as `aggregate_collect/2`
  but takes the value directly instead of reading from a register.
  Used by kernel-dispatch wrappers running in fold-form, where the
  kernel emits each solution value via a callback instead of building
  a list and calling `merge_into_aggregate/2`.

  Convention matches `aggregate_collect/2`: prepend (O(1)); the final
  `Enum.reverse` in `finalise_aggregate/4` restores encounter order.
  This is *different* from `merge_into_aggregate/2`, which appends a
  pre-built list. If the same kernel run uses both helpers in the
  same call, ordering will be inconsistent — pick one path per call.

  Cost note: each call walks `state.choice_points` to find the
  aggregate frame (O(D) where D is the cp-stack depth). For folds
  that emit many values per kernel call, prefer
  `split_at_aggregate_cp/1` — extract the agg cp once, thread its
  `:agg_accum` directly through the fold, reassemble at the end. The
  dispatch wrapper does that.
  """
  def aggregate_push_one(state, value) do
    {updated_cps, _pushed} =
      Enum.map_reduce(state.choice_points, false, fn cp, pushed ->
        cond do
          pushed -> {cp, pushed}
          Map.has_key?(cp, :agg_type) ->
            prior = Map.get(cp, :agg_accum, [])
            {Map.put(cp, :agg_accum, [value | prior]), true}
          true -> {cp, pushed}
        end
      end)
    %{state | choice_points: updated_cps}
  end

  @doc """
  One-pass cp-stack walk that extracts the topmost aggregate frame.
  Returns `{above, agg_cp, below}` where `above` are the cps before
  the aggregate frame (in original order) and `below` are the cps
  after it. Returns `nil` if no aggregate frame is on the stack.

  Use to amortise the cp-stack walk across many fold steps: walk once
  to extract, fold against `agg_cp.agg_accum` directly (avoids one cp
  walk per push), then reassemble via
  `above ++ [updated_agg_cp | below]`. Without this, a fold that
  emits N values into a stack of depth D would do O(N×D) cp walks via
  `aggregate_push_one/2`; with it, the total cp work is O(D).
  """
  def split_at_aggregate_cp(state) do
    split_at_aggregate_cp_loop(state.choice_points, [])
  end

  defp split_at_aggregate_cp_loop([], _above), do: nil
  defp split_at_aggregate_cp_loop([cp | rest], above) do
    if Map.has_key?(cp, :agg_type) do
      {Enum.reverse(above), cp, rest}
    else
      split_at_aggregate_cp_loop(rest, [cp | above])
    end
  end

  @doc """
  Tier-2 runtime cost probe — companion to the static forkMinCost
  gate. The static gate (par_wrap_segment/4) decides at codegen
  time whether a predicates worst-case clause body is heavy enough
  to amortise Task.async_stream overhead. The runtime probe takes a
  measurement on the FIRST invocation of each Tier-2-eligible
  predicate, then sticks with that decision for subsequent calls.

  Decision is keyed by predicate-name string and stored in an ETS
  table created lazily on first call. Three states:

    :probe          — no measurement yet; caller must run sequential
                      and report elapsed time via tier2_probe_update/3.
    :go_sequential  — sequential beat or matched threshold; stay
                      sequential.
    :go_parallel    — sequential exceeded threshold; fan out
                      via Task.async_stream from now on.

  See benchmarks/wam_elixir_tier2_findall.md "Calibration" section
  for the derivation. Both nested_findall and arith workloads
  converge at a sequential-time crossover near 1500us on a 4-core
  box; recommended default is `runtime_cost_probe(1500)`. Setting a
  lower value (e.g. 1000) errs toward fanning out earlier; higher
  values (e.g. 2000) only fork when confidently above crossover.
  """
  def tier2_probe_decision(pred_key) do
    ensure_tier2_probe_table()
    case :ets.lookup(:tier2_cost_probe, pred_key) do
      [{_, decision}] -> decision
      [] -> :probe
    end
  end

  def tier2_probe_update(pred_key, us, threshold_us) do
    ensure_tier2_probe_table()
    decision = if us > threshold_us, do: :go_parallel, else: :go_sequential
    :ets.insert(:tier2_cost_probe, {pred_key, decision})
    :ok
  end

  defp ensure_tier2_probe_table do
    if :ets.whereis(:tier2_cost_probe) == :undefined do
      try do
        :ets.new(:tier2_cost_probe, [:set, :public, :named_table])
        :ok
      rescue
        ArgumentError -> :ok
      end
    end
    :ok
  end

  @doc """
  Update the topmost aggregate frames :cp field to a new continuation.
  Called by end_aggregate-terminated sub-segments to point the agg
  frame at the post-end_aggregate sub-segment, so finalise tail-calls
  the right code after enumeration completes.

  Without this, agg_cp.cp captures whatever state.cp was at
  begin_aggregate push time — usually terminal_cp for top-level
  predicates, or the outer callers continuation for nested findall.
  Neither is correct when the user clauses body has multiple findalls
  in sequence: finalise of the first findall would jump past the
  remaining body code (to terminal_cp or the outer caller), making
  the second findalls setup dead code.

  By updating agg_cp.cp at end_aggregate time, finalise jumps to the
  immediate post-end_aggregate sub-segment, which can run subsequent
  body code (including a second findall, deallocate, proceed, etc.).
  """
  def update_topmost_agg_cp(state, new_cp) do
    {updated_cps, _updated} =
      Enum.map_reduce(state.choice_points, false, fn cp, updated ->
        cond do
          updated -> {cp, updated}
          Map.has_key?(cp, :agg_type) -> {Map.put(cp, :cp, new_cp), true}
          true -> {cp, updated}
        end
      end)
    %{state | choice_points: updated_cps}
  end

  @doc """
  Phase 4a substrate — variant of backtrack/1 for parallel-branch
  context. When the topmost CP is an aggregate frame, returns
  {:branch_exhausted, local_accum} INSTEAD of finalising. This lets
  a Tier-2 super-wrappers Task.async_stream branch return its local
  contribution to the parent for merging via merge_into_aggregate/2,
  rather than triggering a finalise that would only see the branchs
  partial accum.

  Phase 4b will wire this into the super-wrappers branch wrapper —
  see WAM_ELIXIR_TIER2_FINDALL_PHASE4.md sections 4.2/4.3 for the
  control flow. Phase 4c activates intra_query_parallel(true) and
  validates end-to-end.

  Behaviour by topmost-CP type:
    - empty stack → {:branch_exhausted, []} — branch produced
      nothing (e.g., the clauses head failed to match).
    - :agg_type set → {:branch_exhausted, Enum.reverse(accum)} —
      branch reached its own agg frame, enumeration is exhausted,
      return local accum (reversed because aggregate_collect
      prepends for O(1)).
    - other CP type → falls through to backtrack_ordinary/3 —
      resume the next clause-body alternative. backtrack_ordinarys
      result (a state via cp.pc.(state), or {:ok, state}) flows
      through; the wrap_segment catch chain may re-enter via
      backtrack/branch_backtrack as enumeration continues.

  Note on wiring: Phase 4a only adds this helper. Phase 4b decides
  how branches route their wrap_segment catches through it (e.g.,
  via a :branch_mode state field that backtrack/1 dispatches on,
  or via super-wrapper-installed catch arms). The proposals
  question 1 (section 9 Q1) asks reviewers about return-vs-throw
  shape; for Phase 4a we use the return shape per the proposal
  default, leaving room for Phase 4b to revisit if needed.
  """
  def branch_backtrack(state) do
    case state.choice_points do
      [] ->
        {:branch_exhausted, []}
      [cp | rest] ->
        case Map.get(cp, :agg_type) do
          nil ->
            backtrack_ordinary(state, cp, rest)
          _agg_type ->
            {:branch_exhausted, Enum.reverse(Map.get(cp, :agg_accum, []))}
        end
    end
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
  copy because theyre value types in BEAM. Compound values that
  reference heap cells (e.g. structures from put_structure +
  set_value, emitted by compile_aggregate_alls compound-Template
  construction code) are deep-copied via deep_copy_value/2 below —
  the captured value becomes a self-contained {:struct, "name/arity",
  [args]} tuple that survives backtracks heap-rewind. Without
  deep-copy, all elements of accum would point to the same heap
  region (because end_aggregates throw fail rewinds the heap; the
  next iterations put_structure overwrites the same addresses).

  If no aggregate frame is present, returns state unchanged (caller
  misuse — emitter should always pair end_aggregate with begin).

  Walk cost: O(N) in choice-point depth per call (for the agg-frame
  search) plus O(M) in heap-structure size (for deep_copy). Both
  acceptable for Phase 3; if Phase 4 profiling shows either
  dominates, separate optimisations can address them.
  """
  def aggregate_collect(state, value_reg) do
    raw = Map.get(state.regs, value_reg)
    val = deep_copy_value(state, raw)
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
  Recursively walk a captured value, resolving heap references into
  self-contained Elixir tuples that survive backtracks heap-rewind.

  Cases:
    {:unbound, _}  → deref through state.regs; recurse on the bound
                     value if any, else return as-is (degenerate).
    {:ref, addr}   → read heap[addr]. If {:str, "name/arity"}, parse
                     arity and recursively deep-copy the args at
                     heap[addr+1..addr+arity]. Yields {:struct,
                     "name/arity", [arg_copies...]}.
    Anything else  → atomic value (string/number/atom/list of
                     atomics) — return as-is.

  Lists built via put_list / set_value chains arent yet handled —
  they would need heap-walking through ./2 cons cells. For Phase 3c
  the compound-Template scenarios use only structures and atomic
  args.
  """
  def deep_copy_value(state, val) do
    case val do
      {:unbound, _} = unb ->
        derefed = deref_var(state, unb)
        if derefed == unb, do: unb, else: deep_copy_value(state, derefed)
      {:ref, addr} ->
        case Map.get(state.heap, addr) do
          {:str, functor} ->
            arity = parse_functor_arity(functor)
            args =
              if arity > 0 do
                for i <- 1..arity, do: deep_copy_value(state, Map.get(state.heap, addr + i))
              else
                []
              end
            {:struct, functor, args}
          _ ->
            val
        end
      _ ->
        val
    end
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
    # Bind the result through the trail when result_reg holds an
    # unbound ref. Direct slot overwrite (the pre-#1659 behaviour)
    # writes to agg_cp.regs[result_reg_idx] but leaves the underlying
    # ref unbound — fine for the simple case where the result_reg
    # slot IS the binding target, but breaks for nested findall: the
    # inner findalls result_reg shares its unbound ref with the
    # outer callers reg via the calls get_variable, and the inner
    # predicates deallocate merge restores the outers Y-reg
    # snapshot — overwriting the inners reg-slot binding. Trail-style
    # binding (Map.put under the ref id, not the integer reg-slot)
    # propagates through deref_var across frame boundaries and
    # survives deallocate. Proposal section 6 risk 7 (nested findall)
    # fix surfaced by Phase 3c.
    bound_regs =
      case Map.get(agg_cp.regs, agg_cp.agg_result_reg) do
        {:unbound, id} ->
          # Trail the binding under the unbound refs id so any
          # aliased copy sees it via deref_var.
          agg_cp.regs
          |> Map.put(id, result)
          |> Map.put(agg_cp.agg_result_reg, result)
        _ ->
          # Slot was bound or empty — preserve legacy direct overwrite.
          Map.put(agg_cp.regs, agg_cp.agg_result_reg, result)
      end
    restored = %{state |
      regs: bound_regs,
      heap: agg_cp.heap,
      heap_len: agg_cp.heap_len,
      trail: agg_cp.trail,
      trail_len: agg_cp.trail_len,
      stack: agg_cp.stack,
      cp: agg_cp.cp,
      choice_points: rest_cps
    }
    # No env-frame pop here. The pop logic from #1661 (which
    # simulated deallocate to handle the case where end_aggregates
    # throw fail bypassed the predicates deallocate) is no longer
    # needed: with the end_aggregate sub-segment split (this PR),
    # the post-end_aggregate sub-segment ALWAYS contains the
    # predicates deallocate (or, for findalls in the middle of a
    # body, more body code that eventually reaches deallocate).
    # agg_cp.cp now points at that sub-segment thanks to
    # update_topmost_agg_cp/2. The natural deallocate handles env
    # cleanup correctly for all cases: single-level findall (k2 =
    # deallocate+proceed), nested findall (inner_k2 = deallocate
    # restoring outer_k1 as cp, then proceed jumps to outer_k1),
    # and multi-findall in one body (k2..k_n stay in the predicates
    # frame, the final k_last deallocates).
    #
    # Removing the pop also fixes the multi-findall-in-one-body
    # bug: the prior pop overwrote agg_cp.cp with env.cp during
    # restoration, defeating update_topmost_agg_cp.
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
              parallel_depth: 0,
              # branch_mode marks a snapshot as belonging to a Tier-2
              # parallel-branch task. When true, backtrack/1 routes
              # aggregate-frame CPs to branch_backtracks return-shape
              # (returning local accum to the parent for merging) rather
              # than to finalise_aggregate (which would only see the
              # branchs partial accum). Set by the super-wrapper when
              # forking; preserved across the branchs internal
              # enumeration. See WAM_ELIXIR_TIER2_FINDALL_PHASE4.md
              # section 4 for the protocol.
              branch_mode: false
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

defmodule WamRuntime.FactSource.Lmdb do
  @moduledoc """
  LMDB fact source. Memory-mapped key/value store — the response to
  the materialisation-cost bottleneck above 100k+ facts documented
  in docs/WAM_TARGET_ROADMAP.md. Mirrors what Haskell uses, with
  one important caveat: targets the **safe key/value API**, not the
  raw-pointer interface that caused crashes in the Haskell pipeline.
  Lookups go through txn_get and cursor get/get-next, never through
  raw-pointer dereferences.

  Driver responsibility:
    1. Add an LMDB binding to its mix deps. The runtime expects a
       module named `Elmdb`; the Hex `:elmdb` package exposes an
       Erlang module named `:elmdb`, so real drivers should provide a
       tiny bridge module with the function names used below. To keep
       this runtime dep-free for drivers that dont use LMDB, the module
       is referenced indirectly via Module.concat — same trick as the
       SQLite adaptor uses for :exqlite.
    2. Open the LMDB env + database handle (dbi). Populate facts.
    3. Pass env/dbi to this adaptors open/3 via the spec.

  Spec fields:
    env     — LMDB env handle (required; pre-opened by driver)
    dbi     — LMDB database handle within env (required)
    arity   — number of columns per tuple (required; only 2 supported)
    dupsort — true if a single arg1 maps to multiple arg2s (default
              false; matches the LMDB MDB_DUPSORT flag the driver
              should have set when opening the dbi)

  Keying convention: the LMDB key is arg1; the value is arg2 (or
  one of several arg2s under MDB_DUPSORT). lookup_by_arg1 uses
  txn_get for unique keys and cursor MDB_SET / MDB_NEXT_DUP for
  dupsort.
  """
  @behaviour WamRuntime.FactSource
  defstruct [:env, :dbi, :arity, :dupsort]

  defp lmdb_module, do: Module.concat([Elmdb])

  @impl true
  def open(%{env: env, dbi: dbi, arity: arity} = spec, _pred_arity, _state)
      when arity == 2 do
    %__MODULE__{
      env: env,
      dbi: dbi,
      arity: arity,
      dupsort: Map.get(spec, :dupsort, false)
    }
  end

  @impl true
  def stream_all(%__MODULE__{env: env, dbi: dbi}, _state) do
    mod = lmdb_module()
    {:ok, txn} = apply(mod, :ro_txn_begin, [env])
    try do
      {:ok, cur} = apply(mod, :ro_txn_cursor_open, [txn, dbi])
      try do
        scan_cursor_all(mod, cur, [])
      after
        apply(mod, :ro_txn_cursor_close, [cur])
      end
    after
      apply(mod, :ro_txn_commit, [txn])
    end
  end

  @impl true
  def lookup_by_arg1(%__MODULE__{env: env, dbi: dbi, dupsort: dupsort}, key, _state) do
    mod = lmdb_module()
    {:ok, txn} = apply(mod, :ro_txn_begin, [env])
    try do
      if dupsort do
        {:ok, cur} = apply(mod, :ro_txn_cursor_open, [txn, dbi])
        try do
          collect_dupsort(mod, cur, key)
        after
          apply(mod, :ro_txn_cursor_close, [cur])
        end
      else
        case apply(mod, :txn_get, [txn, dbi, key]) do
          {:ok, val} -> [{key, val}]
          :not_found -> []
          _ -> []
        end
      end
    after
      apply(mod, :ro_txn_commit, [txn])
    end
  end

  @impl true
  def close(%__MODULE__{env: _env}, _state) do
    # Env lifecycle belongs to the driver — multiple FactSources may
    # share one env. The driver calls env_close when its done.
    :ok
  end

  # Scan the entire database via cursor MDB_FIRST + MDB_NEXT.
  defp scan_cursor_all(mod, cur, acc) do
    case apply(mod, :ro_txn_cursor_get, [cur, :first, nil]) do
      {:ok, k, v} -> scan_cursor_next(mod, cur, [{k, v} | acc])
      :not_found -> Enum.reverse(acc)
      _ -> Enum.reverse(acc)
    end
  end

  defp scan_cursor_next(mod, cur, acc) do
    case apply(mod, :ro_txn_cursor_get, [cur, :next, nil]) do
      {:ok, k, v} -> scan_cursor_next(mod, cur, [{k, v} | acc])
      :not_found -> Enum.reverse(acc)
      _ -> Enum.reverse(acc)
    end
  end

  # Dupsort scan: position cursor at MDB_SET key, then walk
  # MDB_NEXT_DUP until exhausted.
  defp collect_dupsort(mod, cur, key) do
    case apply(mod, :ro_txn_cursor_get, [cur, :set, key]) do
      {:ok, ^key, v} -> collect_dupsort_next(mod, cur, key, [{key, v}])
      :not_found -> []
      _ -> []
    end
  end

  defp collect_dupsort_next(mod, cur, key, acc) do
    case apply(mod, :ro_txn_cursor_get, [cur, :next_dup, nil]) do
      {:ok, ^key, v} -> collect_dupsort_next(mod, cur, key, [{key, v} | acc])
      _ -> Enum.reverse(acc)
    end
  end
end

defmodule WamRuntime.FactSource.LmdbIntIds do
  @moduledoc """
  LMDB FactSource variant that exposes integer IDs end-to-end. Designed
  for production-scale workloads (>50k unique nodes) where atom-interning
  is not viable due to BEAMs atom-table cap (~~1M default), and where the
  in-memory int-tuple FactSource recipe from PR #1815 wouldnt fit RAM
  (~~1M categories at full Wikipedia scale).

  See `docs/proposals/wam_elixir_lmdb_int_id_factsource.md` for the full
  architecture rationale; this moduledoc is the API summary only.

  Status: function bodies are wired and covered by a MockElmdb e2e plus
  a real-`:elmdb` integration fixture that runs when the local NIF
  toolchain can compile the dependency.

  Architecture: three LMDB sub-databases inside one env.

  | Sub-DB       | Key                       | Value                     | Notes                                  |
  | ------------ | ------------------------- | ------------------------- | -------------------------------------- |
  | facts_dbi    | 8-byte BE u64 (arg1 id)   | 8-byte BE u64 (arg2 id)   | MDB_DUPSORT for arg1 -> many arg2s.    |
  | key_to_id    | binary (UTF-8 string)     | 8-byte BE u64             | Input translation (string -> id).      |
  | id_to_key    | 8-byte BE u64             | binary                    | Output translation (id -> string).     |

  ID assignment is INSERT-time (driver responsibility). Driver populates
  all three sub-databases consistently when ingesting facts.

  Driver responsibility (mirrors PR #1792s `Lmdb`):
    1. Add an LMDB binding to mix deps and provide a module named
       `Elmdb` with this runtime shape. For the Hex `:elmdb`
       package, that bridge delegates to the Erlang module `:elmdb`.
       The module is referenced indirectly via `Module.concat([Elmdb])`
       so this runtime emits without the dependency.
    2. Open the LMDB env, create the three sub-databases.
    3. Ingest facts: for each `(arg1_str, arg2_str)` row, look up or
       allocate the integer ID for each side and write to all three DBs.
    4. Pass env + the three dbi handles to this adaptors `open/3`.

  Pair this with the int-tuple kernel path: the dispatch wrappers
  `dests_fn` reads ids directly from `lookup_by_arg1_id/3` and feeds them
  to `WamRuntime.GraphKernel.CategoryAncestor.fold_hops_with_dests/6`.
  No string interning at lookup time; the kernel sees integers throughout.
  """
  @behaviour WamRuntime.FactSource
  defstruct [:env, :facts_dbi, :key_to_id_dbi, :id_to_key_dbi, :arity, :dupsort]

  defp lmdb_module, do: Module.concat([Elmdb])

  defp encode_id(id) when is_integer(id) and id >= 0, do: <<id::64-big-unsigned>>
  defp decode_id(<<id::64-big-unsigned>>), do: id

  @impl true
  def open(%{env: env, facts_dbi: facts, key_to_id_dbi: k2i,
             id_to_key_dbi: i2k, arity: arity} = spec, _pred_arity, _state)
      when arity == 2 do
    %__MODULE__{
      env: env,
      facts_dbi: facts,
      key_to_id_dbi: k2i,
      id_to_key_dbi: i2k,
      arity: arity,
      dupsort: Map.get(spec, :dupsort, true)
    }
  end

  @doc """
  Fast-path lookup. Returns `[{key_id, value_id}, ...]` where both
  components are integers — no string translation. This is the API the
  kernel-dispatch wrapper should call when the FactSource is int-id
  backed.
  """
  def lookup_by_arg1_id(%__MODULE__{env: env, facts_dbi: dbi, dupsort: dupsort},
                        key_id, _state) when is_integer(key_id) do
    mod = lmdb_module()
    bin_key = encode_id(key_id)
    {:ok, txn} = apply(mod, :ro_txn_begin, [env])
    try do
      if dupsort do
        {:ok, cur} = apply(mod, :ro_txn_cursor_open, [txn, dbi])
        try do
          collect_dupsort_ids(mod, cur, bin_key, key_id)
        after
          apply(mod, :ro_txn_cursor_close, [cur])
        end
      else
        case apply(mod, :txn_get, [txn, dbi, bin_key]) do
          {:ok, bin_val} -> [{key_id, decode_id(bin_val)}]
          :not_found -> []
          _ -> []
        end
      end
    after
      apply(mod, :ro_txn_commit, [txn])
    end
  end

  @impl true
  def lookup_by_arg1(%__MODULE__{} = handle, key, state) when is_binary(key) do
    # Backwards-compatible binary-key entry. Translates key -> id,
    # delegates to the int-id fast path, translates value ids -> binaries
    # on the way out. Two extra LMDB reads per lookup vs the fast path —
    # use lookup_by_arg1_id/3 directly when possible.
    case lookup_id(handle, key) do
      nil -> []
      key_id ->
        for {k_id, v_id} <- lookup_by_arg1_id(handle, key_id, state) do
          {lookup_key(handle, k_id), lookup_key(handle, v_id)}
        end
    end
  end

  @doc """
  Boundary translator: binary node name -> integer ID. Returns `nil` if
  the key has no entry. Caller usually invokes this once at the input
  boundary (parse user-supplied article name).
  """
  def lookup_id(%__MODULE__{env: env, key_to_id_dbi: dbi}, key) when is_binary(key) do
    mod = lmdb_module()
    {:ok, txn} = apply(mod, :ro_txn_begin, [env])
    try do
      case apply(mod, :txn_get, [txn, dbi, key]) do
        {:ok, bin_id} -> decode_id(bin_id)
        :not_found -> nil
        _ -> nil
      end
    after
      apply(mod, :ro_txn_commit, [txn])
    end
  end

  @doc """
  Boundary translator: integer ID -> binary node name. Returns `nil` if
  the ID has no entry. Caller usually invokes this once at the output
  boundary (format result IDs back to display names).
  """
  def lookup_key(%__MODULE__{env: env, id_to_key_dbi: dbi}, id) when is_integer(id) do
    mod = lmdb_module()
    bin_id = encode_id(id)
    {:ok, txn} = apply(mod, :ro_txn_begin, [env])
    try do
      case apply(mod, :txn_get, [txn, dbi, bin_id]) do
        {:ok, key} -> key
        :not_found -> nil
        _ -> nil
      end
    after
      apply(mod, :ro_txn_commit, [txn])
    end
  end

  @doc """
  Driver-side ingestion helper. Populates all three sub-databases
  consistently for a list of `{arg1_str, arg2_str}` pairs:

  - `key_to_id_dbi`: assigns sequential integer IDs (starting at
    `:start_id`, default 0) to each unique string in encounter order.
  - `id_to_key_dbi`: reverse map, written atomically alongside.
  - `facts_dbi`: writes the `(arg1_id, arg2_id)` pair as 8-byte BE
    u64 keys/values.

  Idempotent for previously-seen strings — looks up the existing ID
  before allocating a new one.

  Options:
    `:start_id` (integer, default 0) — ID to assign to the first
      unseen string. For batch ingestion across multiple calls,
      pass the previous calls returned `:next_id`.
    `:txn` (LMDB rw_txn handle, default nil) — if set, run all
      writes inside the caller-supplied transaction. If nil, this
      function opens its own rw_txn and commits at the end.

  Returns `{:ok, %{pairs_seen: integer, new_ids: integer, next_id: integer}}`
  on success. The `next_id` is `start_id + new_ids` and should be
  persisted (e.g. via the callers metadata DB or a sentinel key in
  `id_to_key_dbi`) for the next batch.

  Runtime exercise is covered by MockElmdb and by a gated real-`:elmdb`
  fixture.
  """
  def ingest_pairs(%__MODULE__{} = handle, pairs, opts \\\\ [])
      when is_list(pairs) do
    start_id = Keyword.get(opts, :start_id, 0)
    user_txn = Keyword.get(opts, :txn, nil)
    mod = lmdb_module()

    {txn, owns_txn} =
      if user_txn do
        {user_txn, false}
      else
        {:ok, t} = apply(mod, :rw_txn_begin, [handle.env])
        {t, true}
      end

    try do
      {next_id, pairs_seen, new_ids} =
        Enum.reduce(pairs, {start_id, 0, 0}, fn {a1, a2}, {nid, seen, allocated} ->
          {a1_id, nid, allocated} = intern_one(handle, mod, txn, a1, nid, allocated)
          {a2_id, nid, allocated} = intern_one(handle, mod, txn, a2, nid, allocated)
          :ok = apply(mod, :txn_put, [txn, handle.facts_dbi,
                                      encode_id(a1_id), encode_id(a2_id)])
          {nid, seen + 1, allocated}
        end)

      if owns_txn, do: :ok = apply(mod, :txn_commit, [txn])
      {:ok, %{pairs_seen: pairs_seen, new_ids: new_ids, next_id: next_id}}
    catch
      kind, reason ->
        if owns_txn, do: apply(mod, :txn_abort, [txn])
        {:error, {kind, reason}}
    end
  end

  # Look up or allocate an ID for one string.
  # Returns {id, updated_next_id, updated_allocated_count}.
  defp intern_one(handle, mod, txn, str, next_id, allocated) when is_binary(str) do
    case apply(mod, :txn_get, [txn, handle.key_to_id_dbi, str]) do
      {:ok, bin_id} ->
        {decode_id(bin_id), next_id, allocated}
      :not_found ->
        bin_id = encode_id(next_id)
        :ok = apply(mod, :txn_put, [txn, handle.key_to_id_dbi, str, bin_id])
        :ok = apply(mod, :txn_put, [txn, handle.id_to_key_dbi, bin_id, str])
        {next_id, next_id + 1, allocated + 1}
    end
  end

  @doc """
  Migration helper: convert an existing PR #1792 string-keyed `Lmdb`
  env to an int-id-keyed `LmdbIntIds` env. Cursor-walks the source
  envs facts DB, assigns sequential IDs to unique strings in
  encounter order, and populates the destination envs three
  sub-databases consistently.

  Arguments:
    `source_handle` — `%WamRuntime.FactSource.Lmdb{}` pointing at the
      existing string-keyed env (PR #1792 shape).
    `dest_handle` — freshly opened `%WamRuntime.FactSource.LmdbIntIds{}`
      with empty `facts_dbi` / `key_to_id_dbi` / `id_to_key_dbi`.

  Options:
    `:start_id` (default 0) — first ID to assign.
    `:batch_size` (default 10_000) — commit per-batch to bound the
      rw_txns dirty-page memory at large scale.

  Returns `{:ok, %{pairs_migrated: integer, ids_assigned: integer,
  next_id: integer}}`.

  Runtime exercise is covered by MockElmdb and by a gated real-`:elmdb`
  fixture.
  """
  def migrate_from_string_keyed(source_handle, %__MODULE__{} = dest_handle, opts \\\\ []) do
    start_id = Keyword.get(opts, :start_id, 0)
    batch_size = Keyword.get(opts, :batch_size, 10_000)

    # Stream the source envs (key, value) pairs (binaries from PR #1792)
    # then batch into ingest_pairs/3 calls.
    all_pairs = WamRuntime.FactSource.Lmdb.stream_all(source_handle, nil)

    {final_next_id, total_pairs, total_new_ids} =
      all_pairs
      |> Enum.chunk_every(batch_size)
      |> Enum.reduce({start_id, 0, 0}, fn batch, {nid, total_p, total_n} ->
        case ingest_pairs(dest_handle, batch, start_id: nid) do
          {:ok, %{pairs_seen: p, new_ids: n, next_id: new_nid}} ->
            {new_nid, total_p + p, total_n + n}
          {:error, reason} ->
            throw({:migrate_failed, reason})
        end
      end)

    {:ok, %{pairs_migrated: total_pairs, ids_assigned: total_new_ids,
            next_id: final_next_id}}
  end

  @impl true
  def stream_all(%__MODULE__{env: env, facts_dbi: dbi}, _state) do
    mod = lmdb_module()
    {:ok, txn} = apply(mod, :ro_txn_begin, [env])
    try do
      {:ok, cur} = apply(mod, :ro_txn_cursor_open, [txn, dbi])
      try do
        scan_int_pairs_all(mod, cur, [])
      after
        apply(mod, :ro_txn_cursor_close, [cur])
      end
    after
      apply(mod, :ro_txn_commit, [txn])
    end
  end

  @impl true
  def close(%__MODULE__{env: _env}, _state) do
    # Env lifecycle belongs to the driver — multiple FactSources may share
    # one env. The driver calls env_close when its done with all of them.
    :ok
  end

  defp collect_dupsort_ids(mod, cur, bin_key, key_id) do
    case apply(mod, :ro_txn_cursor_get, [cur, :set, bin_key]) do
      {:ok, ^bin_key, bin_v} ->
        collect_dupsort_ids_next(mod, cur, bin_key, key_id,
                                  [{key_id, decode_id(bin_v)}])
      :not_found -> []
      _ -> []
    end
  end

  defp collect_dupsort_ids_next(mod, cur, bin_key, key_id, acc) do
    case apply(mod, :ro_txn_cursor_get, [cur, :next_dup, nil]) do
      {:ok, ^bin_key, bin_v} ->
        collect_dupsort_ids_next(mod, cur, bin_key, key_id,
                                  [{key_id, decode_id(bin_v)} | acc])
      _ -> Enum.reverse(acc)
    end
  end

  defp scan_int_pairs_all(mod, cur, acc) do
    case apply(mod, :ro_txn_cursor_get, [cur, :first, nil]) do
      {:ok, k, v} -> scan_int_pairs_next(mod, cur, [{decode_id(k), decode_id(v)} | acc])
      :not_found -> Enum.reverse(acc)
      _ -> Enum.reverse(acc)
    end
  end

  defp scan_int_pairs_next(mod, cur, acc) do
    case apply(mod, :ro_txn_cursor_get, [cur, :next, nil]) do
      {:ok, k, v} -> scan_int_pairs_next(mod, cur, [{decode_id(k), decode_id(v)} | acc])
      :not_found -> Enum.reverse(acc)
      _ -> Enum.reverse(acc)
    end
  end
end

defmodule WamRuntime.GraphKernel.TransitiveClosure do
  @moduledoc """
  Native transitive-closure kernel — bypasses WAM dispatch for the
  canonical pattern:

      tc(X, Z) :- edge(X, Z).
      tc(X, Z) :- edge(X, Y), tc(Y, Z).

  Iterative BFS using a MapSet for visited tracking. Composes with
  any edge source — the callers neighbors_fn(node) callback returns
  outgoing edges as `[{from, to}, ...]` tuples (matching the
  FactSource lookup_by_arg1 contract). For LMDB-backed graphs the
  callback closes over the FactSource handle; for in-memory ETS or
  native lists it walks them directly.

  Why bypass WAM: per benchmarks/wam_elixir_tier2_findall.md and
  docs/WAM_TARGET_ROADMAP.md, kernel-based lowering is the largest
  unrealised perf lever for graph workloads (Gos category_ancestor
  FFI kernel hit 52x at scale-300). This is the first such kernel
  for Elixir. Future work: pattern-recognition pass that detects
  the tc/2 shape in source Prolog and routes calls here automatically.

  Semantics: returns nodes reachable from `start` via at least one
  edge step. The starting node itself is NOT included unless reachable
  via a cycle (matches Prolog tc/2 — same as `findall(Z, tc(X, Z), Zs)`
  with X bound, Z unbound).
  """

  @doc """
  Reachable-set as an Elixir list. Order is not specified.

  `neighbors_fn` is a 1-arity function: given a node, returns a list
  of `{from, to}` tuples for outgoing edges. The from arg is the
  query node (kept for FactSource-shape compatibility); only `to`
  is used.
  """
  def reachable_from(neighbors_fn, start) when is_function(neighbors_fn, 1) do
    seeds =
      neighbors_fn.(start)
      |> Enum.map(fn {_from, to} -> to end)
    bfs(neighbors_fn, seeds, MapSet.new())
    |> MapSet.to_list()
  end

  defp bfs(_fn, [], visited), do: visited
  defp bfs(neighbors_fn, [node | rest], visited) do
    if MapSet.member?(visited, node) do
      bfs(neighbors_fn, rest, visited)
    else
      next =
        neighbors_fn.(node)
        |> Enum.map(fn {_from, to} -> to end)
      bfs(neighbors_fn, rest ++ next, MapSet.put(visited, node))
    end
  end

  @doc """
  Convenience for callers backed by a FactSource adaptor:
  `reachable_from_source(source_module, source_handle, start, state)`.
  Wraps the lookup_by_arg1 callback into a neighbors_fn closure.
  """
  def reachable_from_source(source_module, source_handle, start, state \\\\ nil) do
    fun = fn node -> source_module.lookup_by_arg1(source_handle, node, state) end
    reachable_from(fun, start)
  end
end

defmodule WamRuntime.GraphKernel.TransitiveDistance do
  @moduledoc """
  Native transitive-distance kernel — bypasses WAM dispatch for the
  canonical pattern:

      trans_dist(X, Y, 1) :- edge(X, Y).
      trans_dist(X, Y, N) :-
          edge(X, Mid),
          trans_dist(Mid, Y, N1),
          N is N1 + 1.

  Returns all `(target, distance)` pairs reachable from `start` via
  simple paths (per-path visited set, A->B->C and A->C are DIFFERENT
  paths to C with distance 2 and 1, both yielded — same convention
  CategoryAncestor uses).

  Ported from the Rust kernel `collect_native_transitive_distance_results`
  in src/unifyweaver/targets/wam_rust_target.pl. Identical algorithm:
  recursive DFS with explicit per-path visited list, push (target,
  next_depth) on every edge to an unvisited node, recurse with the
  target appended to visited.

  Termination: each recursive step adds the target to `visited`,
  blocking re-entry. The recursion bottoms out when the current
  nodes neighbours are all in visited (i.e. we have walked every
  simple path from `start`).

  Why bypass WAM: same reasoning as TransitiveClosure / CategoryAncestor.
  Adding more kernel kinds is the second-largest perf lever after
  walker tuning (per docs/WAM_TARGET_ROADMAP.md). This kernel was
  the simplest of the five Rust+Haskell kinds Elixir was missing —
  shape mirrors CategoryAncestor (per-path enumeration) but without
  the explicit Visited-list arg or max_depth bound.
  """

  @doc """
  Returns `[{target, distance}, ...]` for every distinct simple path
  from `start`. `neighbors_fn` receives a node, returns a list of
  `{from, to}` tuples (FactSource lookup_by_arg1 contract); only `to`
  is read.
  """
  def collect_pairs(neighbors_fn, start) when is_function(neighbors_fn, 1) do
    walk(neighbors_fn, start, [start], 0, [])
    |> Enum.reverse()
  end

  defp walk(neighbors_fn, node, visited, depth, acc) do
    edges = neighbors_fn.(node)
    next_depth = depth + 1
    Enum.reduce(edges, acc, fn {_from, target}, ac ->
      if :lists.member(target, visited) do
        ac
      else
        ac1 = [{target, next_depth} | ac]
        walk(neighbors_fn, target, [target | visited], next_depth, ac1)
      end
    end)
  end

  @doc """
  Convenience for FactSource-backed graphs.
  """
  def collect_pairs_from_source(source_module, source_handle, start, state \\\\ nil) do
    fun = fn node -> source_module.lookup_by_arg1(source_handle, node, state) end
    collect_pairs(fun, start)
  end
end

defmodule WamRuntime.GraphKernel.TransitiveParentDistance do
  @moduledoc """
  Native transitive-parent-distance kernel — bypasses WAM dispatch
  for the canonical pattern:

      pd(X, Y, X, 1) :- edge(X, Y).
      pd(X, Y, P, N) :-
          edge(X, Mid),
          pd(Mid, Y, P, N1),
          N is N1 + 1.

  Returns `[{target, predecessor, distance}, ...]` triples — for every
  edge encountered during DFS from `start`, records the edge plus the
  depth at which it was traversed. The predecessor is the immediate
  source-side of the recorded edge (the node we stepped FROM), which
  matches the Prolog patterns base-case binding (`pd(X, Y, X, 1)` —
  parent equals start when one edge connects start to target).

  Ported from the Rust kernel
  `collect_native_transitive_parent_distance_results` in
  src/unifyweaver/targets/wam_rust_target.pl. Identical algorithm:
  iterative DFS via an explicit stack, push `(target, predecessor,
  next_depth)` per edge.

  WARNING: NO cycle detection. Matches the Rust reference impl which
  uses a stack-based DFS without a visited list. On cyclic inputs
  this kernel **loops indefinitely**; callers must either ensure
  acyclic input (DAGs only — e.g. typical category-parent structures)
  or wrap with a depth-bound consumer like
  `findall(P-N, (pd(X, Y, P, N), N < MaxD), Triples)`. If you need
  cycle-safe behaviour AND the predecessor, the right next step is a
  separate `transitive_parent_distance4_safe` kernel that adds a
  visited list — not in this PR.
  """

  @doc """
  Returns `[{target, predecessor, distance}, ...]` for every edge
  encountered during DFS from `start`. `neighbors_fn` follows the
  FactSource lookup_by_arg1 contract: receives a node, returns
  `[{from, to}, ...]` tuples.

  Edge-iteration order: matches Rusts `for next in next_nodes.iter().rev()
  + stack.push` — edges are visited in forward order. We reverse the
  edge list before reducing onto the stack so the first edge pops
  first. This preserves the same enumeration order across targets,
  which matters when callers depend on it for deterministic test
  output.
  """
  def collect_triples(neighbors_fn, start) when is_function(neighbors_fn, 1) do
    walk(neighbors_fn, [{start, 0}], [])
    |> Enum.reverse()
  end

  defp walk(_, [], acc), do: acc
  defp walk(neighbors_fn, [{node, depth} | rest], acc) do
    edges = neighbors_fn.(node)
    next_depth = depth + 1
    {new_acc, new_stack} =
      edges
      |> Enum.reverse()
      |> Enum.reduce({acc, rest}, fn {_from, target}, {a, s} ->
        {[{target, node, next_depth} | a], [{target, next_depth} | s]}
      end)
    walk(neighbors_fn, new_stack, new_acc)
  end

  @doc """
  Convenience for FactSource-backed graphs.
  """
  def collect_triples_from_source(source_module, source_handle, start, state \\\\ nil) do
    fun = fn node -> source_module.lookup_by_arg1(source_handle, node, state) end
    collect_triples(fun, start)
  end
end

defmodule WamRuntime.GraphKernel.TransitiveStepParentDistance do
  @moduledoc """
  Native transitive-step-parent-distance kernel — bypasses WAM dispatch
  for the canonical pattern:

      tspd(X, Y, Y, X, 1) :- edge(X, Y).
      tspd(X, Y, Mid, P, D) :-
          edge(X, Mid),
          tspd(Mid, Y, _IgnoredStep, P, D1),
          D is D1 + 1.

  Returns `[{target, step, parent, distance}, ...]` quadruples where:
    - target: the reachable node
    - step: the FIRST hop taken from `start` (the immediate child of
      `start` on the recorded path)
    - parent: the immediate predecessor of `target`
    - distance: total path length from `start` to `target`

  Note that `step` and `parent` differ in general — `step` is anchored
  to the start, `parent` to the target. They coincide only on direct
  edges (depth 1).

  Ported from the Rust kernel
  `collect_native_transitive_step_parent_distance_results` in
  src/unifyweaver/targets/wam_rust_target.pl. Identical algorithm:
  for each direct edge from `start`, emit the depth-1 quadruple, then
  delegate to `TransitiveParentDistance.collect_triples/2` for the
  deeper walk and decorate every returned `(target, parent, dist)`
  triple with the current first-hop and `dist + 1`.

  WARNING: NO cycle detection (inherited from TransitiveParentDistance).
  Cyclic inputs loop indefinitely — use only on DAGs (typical
  category-parent / ontology shapes are fine).
  """

  @doc """
  Returns `[{target, step, parent, distance}, ...]` for every edge in
  the DFS expansion from `start`, decorated with the FIRST hop taken.
  Reuses `WamRuntime.GraphKernel.TransitiveParentDistance.collect_triples/2`
  for the inner walk; the step is fixed at the first hop per outer
  iteration.
  """
  def collect_quads(neighbors_fn, start) when is_function(neighbors_fn, 1) do
    edges = neighbors_fn.(start)
    Enum.flat_map(edges, fn {_from, next} ->
      direct = [{next, next, start, 1}]
      nested =
        WamRuntime.GraphKernel.TransitiveParentDistance.collect_triples(
          neighbors_fn, next)
      bumped =
        Enum.map(nested, fn {target, parent, dist} ->
          {target, next, parent, dist + 1}
        end)
      direct ++ bumped
    end)
  end

  @doc """
  Convenience for FactSource-backed graphs.
  """
  def collect_quads_from_source(source_module, source_handle, start, state \\\\ nil) do
    fun = fn node -> source_module.lookup_by_arg1(source_handle, node, state) end
    collect_quads(fun, start)
  end
end

defmodule WamRuntime.GraphKernel.WeightedShortestPath do
  @moduledoc """
  Native Dijkstra shortest-path kernel — bypasses WAM dispatch for
  the canonical pattern:

      wsp(X, Y, W) :- weighted_edge(X, Y, W).
      wsp(X, Y, TotalW) :-
          weighted_edge(X, Mid, W1),
          wsp(Mid, Y, RestW),
          TotalW is RestW + W1.

  Returns `[{target, cost}, ...]` — the SHORTEST-path cost from
  `start` to each reachable node (excluding start itself).

  Important semantic narrowing: the canonical Prolog pattern above
  enumerates ALL paths from start to target with their summed
  weights. This kernel computes only the SHORTEST. So:

      findall(W, wsp(s, t, W), Ws)
      ;; Prolog source: yields all path weights to t.
      ;; Kernel:        yields a 1-element list — only the shortest.

      findall(T-W, wsp(s, T, W), Pairs)
      ;; Prolog source: yields one pair per (path, weight); same
      ;;                target may appear multiple times.
      ;; Kernel:        yields one pair per reachable target with
      ;;                its shortest-path cost.

      aggregate_all(min, wsp(s, t, W), MinW)
      ;; Both produce the same answer — the kernel just computes it
      ;; in O(E log V) instead of exponential path enumeration.

  Ported from the Rust kernel
  `collect_native_weighted_shortest_path_results` in
  src/unifyweaver/targets/wam_rust_target.pl (which uses BinaryHeap
  with reversed Ordering for min-heap behaviour). Elixir uses
  `:gb_sets` of `{cost, node}` tuples — Erlang term ordering puts
  numerically-smaller cost first, then ties broken by node ordering,
  giving min-first via `:gb_sets.take_smallest/1`.

  Edge predicate is 3-ary: `weighted_edges_fn(node)` receives a node
  and returns `[{from, to, weight}, ...]` triples (the weight at
  position 3, matching the FactSource lookup_by_arg1 contract for
  arity-3 predicates). Weights must be non-negative for Dijkstras
  correctness — the kernel does not validate this; caller responsibility.

  Cycle handling: Dijkstra naturally bounds termination via the dist
  map. Cycles do not loop indefinitely (a nodes dist gets updated
  only on improvement, eventually no improvement possible).
  """

  @doc """
  Returns `[{target, cost}, ...]` — shortest path cost from `start`
  to every reachable node. `weighted_edges_fn` is a 1-arity function
  receiving a node, returns `[{from, to, weight}, ...]` triples.
  """
  def collect_path_costs(weighted_edges_fn, start)
      when is_function(weighted_edges_fn, 1) do
    initial_dist = %{start => 0}
    initial_heap = :gb_sets.singleton({0, start})
    final_dist = dijkstra(weighted_edges_fn, initial_heap, initial_dist)
    final_dist
    |> Enum.flat_map(fn
      {^start, _} -> []
      {node, cost} -> [{node, cost}]
    end)
  end

  defp dijkstra(weighted_edges_fn, heap, dist) do
    case :gb_sets.is_empty(heap) do
      true -> dist
      false ->
        {{cost, node}, heap1} = :gb_sets.take_smallest(heap)
        # Skip stale entries: if dist[node] is already lower than
        # `cost`, we found a better path AFTER pushing this entry.
        best = Map.fetch!(dist, node)
        if cost > best do
          dijkstra(weighted_edges_fn, heap1, dist)
        else
          {new_heap, new_dist} =
            weighted_edges_fn.(node)
            |> Enum.reduce({heap1, dist}, fn {_from, next, weight}, {h, d} ->
              next_cost = cost + weight
              case Map.fetch(d, next) do
                {:ok, prev} when prev <= next_cost ->
                  {h, d}
                _ ->
                  {:gb_sets.add({next_cost, next}, h), Map.put(d, next, next_cost)}
              end
            end)
          dijkstra(weighted_edges_fn, new_heap, new_dist)
        end
    end
  end

  @doc """
  Convenience for FactSource-backed graphs (3-arity weighted_edge
  predicate). The FactSource adaptor must support `arity == 3` and
  return `[{from, to, weight}, ...]` from `lookup_by_arg1`.
  """
  def collect_path_costs_from_source(source_module, source_handle, start, state \\\\ nil) do
    fun = fn node -> source_module.lookup_by_arg1(source_handle, node, state) end
    collect_path_costs(fun, start)
  end
end

defmodule WamRuntime.GraphKernel.AstarShortestPath do
  @moduledoc """
  Native A* shortest-path kernel — bypasses WAM dispatch for the
  canonical pattern:

      astar(X, Y, _Dim, W) :- weighted_edge(X, Y, W).
      astar(X, Y, Dim, TotalW) :-
          weighted_edge(X, Mid, W1),
          astar(Mid, Y, Dim, RestW),
          TotalW is RestW + W1.

  Goal-directed shortest-path search with a dimensionality-aware
  heuristic. Priority: f(n) = g(n)^D + h(n)^D where D is graph
  dimensionality. h(n) = direct semantic distance from n to target
  (via the configured `direct_dist_fn`). Minkowski-style f-cost
  (NOT standard Euclidean f = g + h) — by Minkowski inequality this
  is admissible and tighter than L1 A*.

  Ported from the Rust kernel
  `collect_native_astar_shortest_path_results` in
  src/unifyweaver/targets/wam_rust_target.pl. Identical algorithm:
  min-heap keyed by f-cost, dist map for stale-entry skip, early
  termination when target pops, post-filter dist map.

  Dijkstra fallback: if `target` is `nil`, the heuristic returns 0
  for every node, f-cost reduces to g(n)^D, and the search behaves
  like Dijkstra (no early termination — explores all reachable).
  This matches Rusts `target.is_empty()` branch.

  Two edge predicates per call:
    - `weighted_edges_fn(node)` -> [{from, to, weight}, ...]
      Forward weighted edges. Same contract as
      WeightedShortestPath.collect_path_costs/2.
    - `direct_dist_fn(node)` -> [{from, to, weight}, ...]
      Heuristic edges from `node`. The kernel filters for the entry
      where `to == target`; if none, defaults to 1.0 (Rusts
      "conservative default"). When no separate `direct_dist_pred`
      is configured, callers pass `weighted_edges_fn` here too —
      using forward edges as the heuristic is admissible (any direct
      edge weight is a lower bound on shortest path through it).

  Result shape: same as WeightedShortestPath — `[{node, g_cost}, ...]`
  for every node whose dist was finalised before target popped, plus
  target itself. With early termination, result size depends on
  search depth, not graph size. Caller post-filters to target if
  only that single result is wanted (the dispatch wrapper does this
  when target is bound on entry).

  Cycle handling: same as Dijkstra (PR #1825) — naturally bounded
  via the dist map. No explicit visited list needed.
  """

  @doc """
  Returns `[{node, g_cost}, ...]` — finalised shortest-path costs
  from `start` to nodes explored by A* search. With `target` bound,
  search terminates as soon as target pops from the heap; result
  contains target plus all nodes finalised along the way. With
  `target == nil`, behaves like Dijkstra (full reachable set).

  `dim` is the Minkowski exponent for f-cost. Use 5 by default
  (matches Rusts foreign_usize_config fallback for `dimensionality`).
  """
  def collect_path_costs(weighted_edges_fn, direct_dist_fn, start, target, dim)
      when is_function(weighted_edges_fn, 1) and is_function(direct_dist_fn, 1)
       and is_number(dim) do
    h_start = heuristic(direct_dist_fn, start, target)
    initial_f = f_cost(0, h_start, dim)
    initial_dist = %{start => 0}
    initial_heap = :gb_sets.singleton({initial_f, 0, start})
    final_dist = astar_loop(weighted_edges_fn, direct_dist_fn, target, dim,
                            initial_heap, initial_dist)
    final_dist
    |> Enum.flat_map(fn
      {^start, _} -> []
      {node, cost} -> [{node, cost}]
    end)
  end

  defp astar_loop(weighted_edges_fn, direct_dist_fn, target, dim, heap, dist) do
    case :gb_sets.is_empty(heap) do
      true -> dist
      false ->
        {{_f_cost, g_cost, node}, heap1} = :gb_sets.take_smallest(heap)
        best = Map.fetch!(dist, node)
        cond do
          # Stale-entry skip: better path was found AFTER pushing this entry.
          g_cost > best ->
            astar_loop(weighted_edges_fn, direct_dist_fn, target, dim, heap1, dist)

          # Early termination: target popped from heap. With an admissible
          # heuristic this is guaranteed to be the shortest path to target.
          target != nil and node == target ->
            dist

          # Expand neighbours.
          true ->
            edges = weighted_edges_fn.(node)
            {new_heap, new_dist} =
              edges
              |> Enum.reduce({heap1, dist}, fn {_from, next, weight}, {h, d} ->
                next_g = g_cost + weight
                case Map.fetch(d, next) do
                  {:ok, prev} when prev <= next_g ->
                    {h, d}
                  _ ->
                    h_next = heuristic(direct_dist_fn, next, target)
                    f_next = f_cost(next_g, h_next, dim)
                    {:gb_sets.add({f_next, next_g, next}, h),
                     Map.put(d, next, next_g)}
                end
              end)
            astar_loop(weighted_edges_fn, direct_dist_fn, target, dim,
                       new_heap, new_dist)
        end
    end
  end

  # h(n) = direct distance from `node` to `target`. nil target means
  # Dijkstra mode (no goal). Default 1.0 if no direct edge recorded
  # (matches Rusts conservative default).
  defp heuristic(_direct_dist_fn, _node, nil), do: 0
  defp heuristic(direct_dist_fn, node, target) do
    edges = direct_dist_fn.(node)
    Enum.find_value(edges, 1.0, fn
      {_from, to, w} when to == target -> w
      _ -> false
    end)
  end

  # f(n) = g(n)^D + h(n)^D — Minkowski-style. NOT standard A*s g + h.
  defp f_cost(g, h, dim), do: :math.pow(g, dim) + :math.pow(h, dim)

  @doc """
  Convenience for FactSource-backed graphs. Both `weighted_source` and
  `direct_source` may be the same handle if no separate
  direct_dist_pred is configured (the detectors fallback shape).
  """
  def collect_path_costs_from_source(weighted_source_module, weighted_source_handle,
                                     direct_source_module, direct_source_handle,
                                     start, target, dim, state \\\\ nil) do
    weighted_fn = fn node ->
      weighted_source_module.lookup_by_arg1(weighted_source_handle, node, state)
    end
    direct_fn = fn node ->
      direct_source_module.lookup_by_arg1(direct_source_handle, node, state)
    end
    collect_path_costs(weighted_fn, direct_fn, start, target, dim)
  end
end

defmodule WamRuntime.GraphKernel.CategoryAncestor do
  @moduledoc """
  Native bounded-ancestor kernel — path-enumeration variant of TC
  with hops counter, per-path visited list, and max-depth cutoff.
  Matches the canonical shape:

      category_ancestor(Cat, Parent, 1, Visited) :-
          category_parent(Cat, Parent),
          \\+ member(Parent, Visited).
      category_ancestor(Cat, Ancestor, Hops, Visited) :-
          max_depth(MaxD), length(Visited, D), D < MaxD, !,
          category_parent(Cat, Mid),
          \\+ member(Mid, Visited),
          category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
          Hops is H1 + 1.

  Critically, the Visited list is **per-path** (grows as
  `[Mid|Visited]` on recursion). For a graph A->B->C and A->C the
  Prolog version enumerates BOTH paths to C with different hop
  counts — the kernel preserves that semantics rather than the
  BFS-dedup shortcut, so aggregations like effective_distances
  `Σ d^(-N)` produce correct results.

  Ported from the Rust kernel `collect_native_category_ancestor_hops`
  in src/unifyweaver/targets/wam_rust_target.pl. Identical algorithm,
  BEAM idioms.

  Node identifier types — the kernel is term-agnostic. `cat`, `root`,
  visited-list members, and dests-list members are compared with `==`
  and `:lists.member/2`; any term with sensible equality works. In
  practice three patterns are common:

  * **Atoms** (e.g. `:Physics`). Atom comparison is immediate-word
    compare on BEAM, plus atom-table entries carry precomputed hashes
    so `Map.get` lookups are fast. Best per-call latency at workloads
    up to ~~50k unique nodes. Above that, the global atom table cap
    (~~1M default, shared with the rest of the VM) makes atoms
    unsuitable.

  * **Integers** with `Map`-backed FactSource. Required for production
    scale (e.g. full Wikipedia category data ~~1M unique categories).
    `Map.get` on integer keys recomputes `:erlang.phash2` per call
    (~~10-15% slower than atoms at the scales we measured), but no
    table cap.

  * **Integers with tuple-as-array FactSource** (RECOMMENDED at
    scale). When integer IDs are contiguous (`0..N-1`, e.g. produced
    by an LMDB key-id mapping or a one-shot intern pass), store
    `cat -> [parent_id, ...]` as a tuple of length N+1. `dests_fn` is
    `fn n -> elem(cp_tuple, n) end` — O(1) read, no hashing. About
    2x faster than atom-Map and 2.2x faster than int-Map at 10k.
    Note: contrary to an earlier claim in this doc, no shipping
    target currently uses LMDBs internal record positions as the
    interning key — Haskell, Rust, and Scala all maintain a separate
    string -> int table at codegen or runtime. See
    docs/proposals/wam_elixir_lmdb_int_id_factsource.md for the
    LMDB-native design that would actually achieve the integration.
  """

  @doc """
  Collect hop counts for every simple path from `cat` to `root` of
  length up to `max_depth`. Returns a list of integers — same path
  may show up multiple times under different routes; aggregations
  consume this list as-is.

  `neighbors_fn` is a 1-arity function: given a node, returns a list
  of `{from, to}` tuples (matching the FactSource.lookup_by_arg1
  contract). For each call, only `to` is read.

  Performance note: the implementation pattern-matches `{_from, to}` per
  neighbor. For workloads where the FactSource can return bare
  destinations directly (no tuple wrapping), prefer
  `collect_hops_with_dests/4` — at scale-1k+ on Wikipedia category data
  it cuts ~~20-25% off the runtime by eliminating the per-neighbor
  tuple construction.
  """
  def collect_hops(neighbors_fn, cat, root, max_depth)
      when is_function(neighbors_fn, 1) and is_integer(max_depth) do
    collect_n(neighbors_fn, cat, root, [], max_depth, [], 0)
    |> Enum.reverse()
  end

  @doc """
  Tuple-input fold variant — same as `fold_hops_with_dests/6` but
  pairs with `collect_hops/4` (neighbors_fn returning `{from, to}`
  tuples instead of bare destinations). Used by the kernel-dispatch
  wrapper that sits behind `WamRuntime.FactSource.lookup_by_arg1`,
  whose contract returns tuples.
  """
  def fold_hops(neighbors_fn, cat, root, max_depth, init_acc, hop_fn)
      when is_function(neighbors_fn, 1) and is_integer(max_depth)
       and is_function(hop_fn, 2) do
    fold_n(neighbors_fn, cat, root, [], max_depth, init_acc, 0, hop_fn)
  end

  @doc """
  Same semantics as `collect_hops/4` but `dests_fn` returns a bare list
  of destination nodes (no `{from, to}` tuple wrapping). Use this when
  the underlying FactSource can produce destinations directly.
  """
  def collect_hops_with_dests(dests_fn, cat, root, max_depth)
      when is_function(dests_fn, 1) and is_integer(max_depth) do
    collect_d(dests_fn, cat, root, [], max_depth, [], 0)
    |> Enum.reverse()
  end

  @doc """
  Same tree walk as `collect_hops_with_dests/4` but folds each hop count
  through `hop_fn.(hop, acc)` instead of consing onto a list. The walker
  never materialises an intermediate list — `init_acc` threads through
  every call directly.

  `hop_fn :: (hop_count :: integer, acc) -> acc`. Called once per simple
  path from `cat` to `root` of length up to `max_depth`. Hop counts are
  delivered in DFS post-order (the same sequence `collect_hops_with_dests/4`
  would prepend, just without the cons cells).

  This is the BEAM-native analogue of Rust iterator `fold` /
  Haskell list-comprehension fusion: passing the consumer fold INTO
  the producer eliminates the intermediate sequence allocation
  entirely. Use this when the consumer is `sum`, `count`, `max`,
  power-sum aggregation, etc. Use `collect_hops_with_dests/4` only
  when the consumer needs the materialised list (multi-pass or
  random-access).

  Example — power-sum aggregation as in `effective_distance/3`:

      fold_hops_with_dests(dests_fn, cat, root, 10, 0.0, fn hop, acc ->
        acc + :math.pow(hop + 1, -n)
      end)

  No list of hops is ever built; the BEAM stays in the recursion frame
  and accumulates the float directly. Equivalent to the GHC
  deforestation that `wam_haskell_target.pl` relies on for the same
  workload.
  """
  def fold_hops_with_dests(dests_fn, cat, root, max_depth, init_acc, hop_fn)
      when is_function(dests_fn, 1) and is_integer(max_depth)
       and is_function(hop_fn, 2) do
    fold_d(dests_fn, cat, root, [], max_depth, init_acc, 0, hop_fn)
  end

  # `depth` is the number of edges already taken to reach `cat`; pushed hop
  # counts are `depth + 1` for any direct edge `cat -> root` we discover.
  # That replaces the original post-recursion split/map/++ bump pattern
  # (which spent ~~28% of runtime on length/1, Enum.split, and ++ at scale).
  # The direct-edge check is also fused into the same reduce that recurses,
  # so neighbors are scanned once per step instead of twice.
  #
  # Bound semantics: matches the Rust reference impl
  # `collect_native_category_ancestor_hops` in wam_rust_target.pl. Rust seeds
  # `visited` with `[cat]` and stops recursing when `visited.len() >= max_depth`,
  # giving max top-level hop count = max_depth. Here `visited` is seeded with
  # `[]`, so the equivalent gate is `next_depth < max_depth` (the deepest
  # direct-edge check still pushes `next_depth = max_depth`). The earlier
  # Elixir kernel started with `visited = []` AND used `length(visited) >= max_depth`,
  # which allowed one extra level of direct check vs Rust and over-counted
  # paths near the depth boundary by ~~7% on Wikipedia category data.
  #
  # Two variants below — `_n` takes neighbors_fn returning {from, to} tuples
  # (FactSource.lookup_by_arg1 contract), `_d` takes dests_fn returning bare
  # destinations. Sharing one implementation via a normalise lambda costs
  # ~~20% in measurement (closure dispatch per neighbor), so they are kept
  # separate. Keep the bodies in sync.
  # Use `:lists.member/2` directly instead of `to in visited` (which compiles
  # to `Enum.member?` outside guards). Skips the Enum protocol dispatch — at
  # scale-1k this is ~~6-7% of total runtime that goes to dispatch overhead
  # before falling through to the same lists.member call.
  # Invariant: `root` is never in `visited`. Both public entry points
  # (`collect_hops/4`, `collect_hops_with_dests/4`) seed visited = []; the
  # `to == root` branch below pushes a hop and stops without recursing, so
  # only non-root nodes ever enter visited via the `true ->` branch. The
  # `root_blocked? = :lists.member(root, visited)` check that earlier
  # versions ran on every recursion step was therefore always false. Removed
  # — saves ~~6.5% of runtime at 1k by halving the lists:member call count.
  # If a future caller needs to seed visited with root pre-included, that
  # contract should be made explicit (e.g., a separate entry point).
  #
  # `can_recurse?` is also constant for the entire dests list at one
  # recursion level (depends only on `depth`, not on the dest), so we
  # split into two walkers — `walk_*_recurse/N` may descend, `walk_*_leaf/N`
  # only counts direct hits. That hoists the per-element conditional out
  # of the inner loop.
  #
  # Hand-recursive walkers replace `Enum.reduce` to avoid per-call closure
  # allocation + dispatch (which was the largest cost block in the post-PR-#1810
  # profile: `Enum.reduce/3 lists^foldl/2-0` at ~~22% combined with the
  # closure body at ~~14%).
  # `collect_n/7` and `collect_d/7` are kept only as the public-API entries
  # that establish the initial state. The per-recursion-step trampoline they
  # used to be (compute next_depth, branch on bound, dispatch to walker) is
  # now inlined into walk_*_recurse below — saves one function dispatch per
  # recursion step (was ~~20% of runtime in `collect_d/7` alone in the
  # post-Lever-A+B profile at 1k).
  defp collect_n(neighbors_fn, cat, root, visited, max_depth, acc, depth) do
    next_depth = depth + 1
    if next_depth < max_depth do
      walk_n_recurse(neighbors_fn.(cat), neighbors_fn, root, visited, max_depth, next_depth, acc)
    else
      walk_n_leaf(neighbors_fn.(cat), root, next_depth, acc)
    end
  end

  defp walk_n_recurse([], _, _, _, _, _, acc), do: acc
  defp walk_n_recurse([{_from, to} | rest], nf, root, vis, mx, nd, acc) when to === root do
    walk_n_recurse(rest, nf, root, vis, mx, nd, [nd | acc])
  end
  defp walk_n_recurse([{_from, to} | rest], nf, root, vis, mx, nd, acc) do
    ac =
      case :lists.member(to, vis) do
        true -> acc
        false ->
          new_nd = nd + 1
          if new_nd < mx do
            walk_n_recurse(nf.(to), nf, root, [to | vis], mx, new_nd, acc)
          else
            walk_n_leaf(nf.(to), root, new_nd, acc)
          end
      end
    walk_n_recurse(rest, nf, root, vis, mx, nd, ac)
  end

  defp walk_n_leaf([], _, _, acc), do: acc
  defp walk_n_leaf([{_from, to} | rest], root, nd, acc) do
    ac = if to == root, do: [nd | acc], else: acc
    walk_n_leaf(rest, root, nd, ac)
  end

  # Tuple-input fold walkers — same shape as fold_d/fold_d_recurse/fold_d_leaf
  # but the input neighbor list is `[{from, to} | rest]` (FactSource
  # `lookup_by_arg1` contract) so the from-side is unpacked at pattern-match
  # time. Keep in sync with walk_n_recurse / walk_n_leaf and with the
  # dests-side fold_d_* below.
  defp fold_n(neighbors_fn, cat, root, visited, max_depth, acc, depth, hop_fn) do
    next_depth = depth + 1
    if next_depth < max_depth do
      fold_n_recurse(neighbors_fn.(cat), neighbors_fn, root, visited, max_depth, next_depth, acc, hop_fn)
    else
      fold_n_leaf(neighbors_fn.(cat), root, next_depth, acc, hop_fn)
    end
  end

  defp fold_n_recurse([], _, _, _, _, _, acc, _), do: acc
  defp fold_n_recurse([{_from, to} | rest], nf, root, vis, mx, nd, acc, hop_fn) when to === root do
    fold_n_recurse(rest, nf, root, vis, mx, nd, hop_fn.(nd, acc), hop_fn)
  end
  defp fold_n_recurse([{_from, to} | rest], nf, root, vis, mx, nd, acc, hop_fn) do
    ac =
      case :lists.member(to, vis) do
        true -> acc
        false ->
          new_nd = nd + 1
          if new_nd < mx do
            fold_n_recurse(nf.(to), nf, root, [to | vis], mx, new_nd, acc, hop_fn)
          else
            fold_n_leaf(nf.(to), root, new_nd, acc, hop_fn)
          end
      end
    fold_n_recurse(rest, nf, root, vis, mx, nd, ac, hop_fn)
  end

  defp fold_n_leaf([], _, _, acc, _), do: acc
  defp fold_n_leaf([{_from, to} | rest], root, nd, acc, hop_fn) do
    ac = if to == root, do: hop_fn.(nd, acc), else: acc
    fold_n_leaf(rest, root, nd, ac, hop_fn)
  end

  defp collect_d(dests_fn, cat, root, visited, max_depth, acc, depth) do
    next_depth = depth + 1
    if next_depth < max_depth do
      walk_d_recurse(dests_fn.(cat), dests_fn, root, visited, max_depth, next_depth, acc)
    else
      walk_d_leaf(dests_fn.(cat), root, next_depth, acc)
    end
  end

  defp walk_d_recurse([], _, _, _, _, _, acc), do: acc
  # Direct hit on root: a separate clause with `when to === root` guard
  # so BEAMs clause selection dispatches without paying the cond chain
  # overhead. Measured ~~5-12% improvement on chunked-parallel at 10k.
  defp walk_d_recurse([to | rest], df, root, vis, mx, nd, acc) when to === root do
    walk_d_recurse(rest, df, root, vis, mx, nd, [nd | acc])
  end
  defp walk_d_recurse([to | rest], df, root, vis, mx, nd, acc) do
    ac =
      case :lists.member(to, vis) do
        true -> acc
        false ->
          new_nd = nd + 1
          if new_nd < mx do
            walk_d_recurse(df.(to), df, root, [to | vis], mx, new_nd, acc)
          else
            walk_d_leaf(df.(to), root, new_nd, acc)
          end
      end
    walk_d_recurse(rest, df, root, vis, mx, nd, ac)
  end

  defp walk_d_leaf([], _, _, acc), do: acc
  defp walk_d_leaf([to | rest], root, nd, acc) do
    ac = if to == root, do: [nd | acc], else: acc
    walk_d_leaf(rest, root, nd, ac)
  end

  # === fold variant ===
  # Same recursion shape as collect_d / walk_d_recurse / walk_d_leaf, but
  # threads `acc` through `hop_fn.(nd, acc)` on each direct hit instead of
  # prepending to a list. Kept as a separate set of defps so the
  # collect_hops_with_dests/4 (list-building) path keeps its inlined
  # `[nd | acc]` and pays no closure dispatch — the fold path pays one
  # closure call per HIT (much rarer than per-recursion-step), which is
  # vastly cheaper than the cons-cell allocation + GC that the list path
  # would trigger across thousands of paths. Bodies tagged "keep in sync
  # with walk_d_*" — the only structural difference is the action on a
  # direct hit (cons vs hop_fn).
  defp fold_d(dests_fn, cat, root, visited, max_depth, acc, depth, hop_fn) do
    next_depth = depth + 1
    if next_depth < max_depth do
      fold_d_recurse(dests_fn.(cat), dests_fn, root, visited, max_depth, next_depth, acc, hop_fn)
    else
      fold_d_leaf(dests_fn.(cat), root, next_depth, acc, hop_fn)
    end
  end

  defp fold_d_recurse([], _, _, _, _, _, acc, _), do: acc
  defp fold_d_recurse([to | rest], df, root, vis, mx, nd, acc, hop_fn) when to === root do
    fold_d_recurse(rest, df, root, vis, mx, nd, hop_fn.(nd, acc), hop_fn)
  end
  defp fold_d_recurse([to | rest], df, root, vis, mx, nd, acc, hop_fn) do
    ac =
      case :lists.member(to, vis) do
        true -> acc
        false ->
          new_nd = nd + 1
          if new_nd < mx do
            fold_d_recurse(df.(to), df, root, [to | vis], mx, new_nd, acc, hop_fn)
          else
            fold_d_leaf(df.(to), root, new_nd, acc, hop_fn)
          end
      end
    fold_d_recurse(rest, df, root, vis, mx, nd, ac, hop_fn)
  end

  defp fold_d_leaf([], _, _, acc, _), do: acc
  defp fold_d_leaf([to | rest], root, nd, acc, hop_fn) do
    ac = if to == root, do: hop_fn.(nd, acc), else: acc
    fold_d_leaf(rest, root, nd, ac, hop_fn)
  end

  @doc """
  Convenience wrapper for FactSource-backed graphs.
  """
  def collect_hops_from_source(source_module, source_handle, cat, root,
                               max_depth, state \\\\ nil) do
    fun = fn node -> source_module.lookup_by_arg1(source_handle, node, state) end
    collect_hops(fun, cat, root, max_depth)
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
    % Atom-interning experiment: when the user passes intern_atoms(true),
    % set a process-wide flag that elixir_constant_literal/2 in the
    % lowered emitter consults. setup_call_cleanup/3 retracts the flag
    % afterward so a failed lowering doesnt leak the setting into
    % subsequent generations. Off by default (existing behaviour
    % preserved — every constant emits as a binary).
    setup_call_cleanup(
        atom_interning_setup(Options),
        do_write_wam_elixir_project(Predicates, Options, ProjectDir,
                                     Mode, ModuleName, LibDir),
        atom_interning_cleanup(Options)
    ).

atom_interning_setup(Options) :-
    (   option(intern_atoms(true), Options)
    ->  assertz(wam_elixir_lowered_emitter:intern_atoms_enabled)
    ;   true
    ).

%% detect_kernel_for_predicate(+Pred, +Arity, -Kernel)
%  Runs the shared recursive-kernel detector on the user-asserted
%  clauses for Pred/Arity. The detector is target-neutral (lives in
%  src/unifyweaver/core/recursive_kernel_detection.pl); same module
%  Rust uses. Returns the full recursive_kernel(Kind, Pred/Arity,
%  ConfigOps) term on success.
detect_kernel_for_predicate(Pred, Arity, Kernel) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    detect_recursive_kernel(Pred, Arity, Clauses, Kernel).

%% emit_kernel_dispatch_module(+ModuleName, +Pred, +Arity, +Kernel, -Code)
%  Dispatch by kernel kind to the appropriate emitter. Adding a new
%  kernel kind requires only adding a new clause here + the runtime
%  module in compile_wam_runtime_to_elixir.
emit_kernel_dispatch_module(ModuleName, Pred, _Arity,
                            recursive_kernel(transitive_closure2, _, ConfigOps),
                            Code) :-
    member(edge_pred(EdgePred/_), ConfigOps),
    compile_tc_kernel_dispatch_module(ModuleName, Pred, EdgePred, Code).
emit_kernel_dispatch_module(ModuleName, Pred, _Arity,
                            recursive_kernel(category_ancestor, _, ConfigOps),
                            Code) :-
    member(edge_pred(EdgePred/_), ConfigOps),
    member(max_depth(MaxDepth), ConfigOps),
    compile_category_ancestor_dispatch_module(ModuleName, Pred, EdgePred, MaxDepth, Code).
emit_kernel_dispatch_module(ModuleName, Pred, _Arity,
                            recursive_kernel(transitive_distance3, _, ConfigOps),
                            Code) :-
    member(edge_pred(EdgePred/_), ConfigOps),
    compile_transitive_distance_dispatch_module(ModuleName, Pred, EdgePred, Code).
emit_kernel_dispatch_module(ModuleName, Pred, _Arity,
                            recursive_kernel(transitive_parent_distance4, _, ConfigOps),
                            Code) :-
    member(edge_pred(EdgePred/_), ConfigOps),
    compile_transitive_parent_distance_dispatch_module(ModuleName, Pred, EdgePred, Code).
emit_kernel_dispatch_module(ModuleName, Pred, _Arity,
                            recursive_kernel(transitive_step_parent_distance5, _, ConfigOps),
                            Code) :-
    member(edge_pred(EdgePred/_), ConfigOps),
    compile_transitive_step_parent_distance_dispatch_module(ModuleName, Pred, EdgePred, Code).
emit_kernel_dispatch_module(ModuleName, Pred, _Arity,
                            recursive_kernel(weighted_shortest_path3, _, ConfigOps),
                            Code) :-
    member(edge_pred(EdgePred/_), ConfigOps),
    compile_weighted_shortest_path_dispatch_module(ModuleName, Pred, EdgePred, Code).
emit_kernel_dispatch_module(ModuleName, Pred, _Arity,
                            recursive_kernel(astar_shortest_path4, _, ConfigOps),
                            Code) :-
    member(edge_pred(EdgePred/_), ConfigOps),
    member(direct_dist_pred(DirectPred/_), ConfigOps),
    member(dimensionality(Dim), ConfigOps),
    compile_astar_shortest_path_dispatch_module(ModuleName, Pred, EdgePred, DirectPred, Dim, Code).

%% compile_tc_kernel_dispatch_module(+ModuleName, +Pred, +EdgePred, -Code)
%
%  Emit a per-predicate dispatch module that routes calls through
%  WamRuntime.GraphKernel.TransitiveClosure.reachable_from instead
%  of the WAM-bytecode lower chain. Replaces what lower_predicate_to_-
%  elixir would have emitted for predicates matching the canonical
%  TC pattern.
%
%  Driver responsibility: register the edge predicate as a FactSource
%  so the kernel can fetch neighbours. The dispatch module looks the
%  source up by indicator string (e.g., "edge/2") at call time.
%
%  Findall integration: when called from inside a `findall(Z, tc(X, Z),
%  L)` aggregate frame, the surrounding state.cp does aggregate_collect
%  + throw fail per result. We iterate the kernels reachable list,
%  trail-bind reg 2, call state.cp, catch the fail, unwind trail, try
%  the next. After exhaustion, throw fail to terminate the predicate.
%
%  Driver-direct call (state.cp = terminal_cp): the FIRST iteration
%  returns {:ok, state} without throwing; we halt and propagate.
compile_tc_kernel_dispatch_module(ModuleName, Pred, EdgePred, Code) :-
    atom_string(ModuleName, ModName),
    camel_case(ModName, CamelMod),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    atom_string(EdgePred, EdgePredStr),
    format(string(EdgeKey), "~w/2", [EdgePredStr]),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc """
  Kernel-dispatched ~w/2 (auto-recognised TC pattern over ~w/2).
  Replaces the WAM-bytecode lowering for this predicate; routes
  calls through WamRuntime.GraphKernel.TransitiveClosure.

  Edge source must be registered via WamRuntime.FactSourceRegistry
  under the indicator "~w" before calling. The registered handle
  must implement the WamRuntime.FactSource behaviour (any of
  FactSource.Lmdb, FactSource.Sqlite, FactSource.Ets, FactSource.Tsv,
  or a driver-supplied module).
  """

  def run(%WamRuntime.WamState{} = state) do
    start_val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    edge_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    neighbors_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(edge_handle, node, state)
    end
    reachable =
      WamRuntime.GraphKernel.TransitiveClosure.reachable_from(
        neighbors_fn, start_val)

    # Two-mode dispatch:
    #
    # (a) Findall / aggregate context — an aggregate frame is on the
    #     CP stack (gated by in_forkable_aggregate_frame?/1, the same
    #     check the Tier-2 super-wrapper uses). Dump ALL kernel
    #     results into the agg_accum in one go via merge_into_-
    #     aggregate, then throw fail to drive the standard
    #     backtrack -> finalise_aggregate flow. Avoids iterating
    #     state.cp per result, which would let the calling clauses
    #     k1 catch its own fail and finalise after one solution.
    #
    # (b) Driver-direct call — no agg frame; return the FIRST
    #     reachable node bound into result reg 2 as a single solution.
    #     Subsequent solutions are not enumerable in this mode (no
    #     choice point to drive backtracking), but the typical
    #     driver-direct call shape is `findall(Z, tc(X, Z), L)` which
    #     hits path (a).
    if WamRuntime.in_forkable_aggregate_frame?(state) do
      merged = WamRuntime.merge_into_aggregate(state, reachable)
      throw({:fail, merged})
    else
      case reachable do
        [first | _] ->
          case bind_result_reg(state, 2, first) do
            nil -> throw({:fail, state})
            bound -> {:ok, bound}
          end
        [] -> throw({:fail, state})
      end
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    {state, arg_vars} =
      Enum.with_index(args, 1)
      |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
        case arg do
          {:unbound, _} ->
            v = {:unbound, make_ref()}
            {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
          other ->
            {%{s | regs: Map.put(s.regs, i, other)}, vars}
        end
      end)
    state = %{state | arg_vars: arg_vars, cp: &WamRuntime.terminal_cp/1}
    try do
      case run(state) do
        {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
        other -> other
      end
    catch
      {:fail, _} -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  defp bind_result_reg(state, reg_idx, value) do
    case Map.get(state.regs, reg_idx) do
      {:unbound, id} ->
        state |> WamRuntime.trail_binding(id) |> WamRuntime.put_reg(id, value)
      ^value -> state
      _ -> nil
    end
  end
end
',
    [CamelMod, CamelPred,
     PredStr, EdgePredStr,
     EdgeKey,
     EdgeKey,
     CamelPred]).

%% compile_category_ancestor_dispatch_module(+ModuleName, +Pred,
%%   +EdgePred, +MaxDepth, -Code)
%
%  Dispatch module for category_ancestor pattern (4-ary; bounded
%  ancestor with hops + per-path visited list). Mirrors the TC
%  dispatchers shape but binds the hops register (A3) per result
%  rather than the destination register, since the canonical form
%  is `category_ancestor(StartCat, Root, Hops, Visited)` — Root is
%  bound on entry (an arg, not a variable to enumerate) and the
%  enumeration is over distinct path-lengths.
%
%  Aggregations like effective_distance call this inside
%  `aggregate_all(sum(W), (path_to_root(...), W is Hops^(-N)), S)` —
%  each enumerated Hops produces one W contribution. Faithful path-
%  enumeration semantics matter because two routes A->B->C and A->C
%  give DIFFERENT hop counts that both contribute to the sum.
compile_category_ancestor_dispatch_module(ModuleName, Pred, EdgePred, MaxDepth, Code) :-
    atom_string(ModuleName, ModName),
    camel_case(ModName, CamelMod),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    atom_string(EdgePred, EdgePredStr),
    format(string(EdgeKey), "~w/2", [EdgePredStr]),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc """
  Kernel-dispatched ~w/4 (auto-recognised category_ancestor pattern
  over ~w/2; max_depth=~w). Routes calls through
  WamRuntime.GraphKernel.CategoryAncestor.

  Edge source must be registered via WamRuntime.FactSourceRegistry
  under the indicator "~w" before calling.
  """

  @max_depth ~w

  def run(%WamRuntime.WamState{} = state) do
    cat_val  = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    root_val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 2))
    edge_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    neighbors_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(edge_handle, node, state)
    end

    if WamRuntime.in_forkable_aggregate_frame?(state) do
      # Fold-form path: stream each hop directly into the aggregate
      # frames :agg_accum. Skips both the intermediate hops list AND
      # the per-hit cp-stack walk that aggregate_push_one/2 would do —
      # split_at_aggregate_cp/1 walks the cp stack ONCE and exposes
      # :agg_accum, the fold prepends in O(1) per hit, and the
      # reassembled state is thrown back up to finalise_aggregate.
      # Total cp work is O(D) for the dispatch instead of O(N×D).
      #
      # Equivalent semantics to the legacy collect+merge path for
      # findall/sum/count/max aggregates whose value register IS the
      # kernel hop output. Transform-aware shapes —
      # `aggregate_all(sum(W), (Goal, W is f(Hops)), R)` — are NOT
      # yet handled: the dispatch wrapper does not see `f`, so it
      # would push raw hops where f(Hops) values are expected. Those
      # queries currently bypass kernel dispatch and run on plain
      # WAM; the follow-up emitter pass will recognise them at
      # compile time and compile the arithmetic into hop_fn.
      {above, agg_cp, below} = WamRuntime.split_at_aggregate_cp(state)
      new_accum =
        WamRuntime.GraphKernel.CategoryAncestor.fold_hops(
          neighbors_fn, cat_val, root_val, @max_depth, agg_cp.agg_accum,
          fn hop, accum -> [hop | accum] end
        )
      new_agg_cp = %{agg_cp | agg_accum: new_accum}
      new_state = %{state | choice_points: above ++ [new_agg_cp | below]}
      throw({:fail, new_state})
    else
      hops_list =
        WamRuntime.GraphKernel.CategoryAncestor.collect_hops(
          neighbors_fn, cat_val, root_val, @max_depth)
      case hops_list do
        [first | _] ->
          case bind_hops_reg(state, 3, first) do
            nil -> throw({:fail, state})
            bound -> {:ok, bound}
          end
        [] -> throw({:fail, state})
      end
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    {state, arg_vars} =
      Enum.with_index(args, 1)
      |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
        case arg do
          {:unbound, _} ->
            v = {:unbound, make_ref()}
            {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
          other ->
            {%{s | regs: Map.put(s.regs, i, other)}, vars}
        end
      end)
    state = %{state | arg_vars: arg_vars, cp: &WamRuntime.terminal_cp/1}
    try do
      case run(state) do
        {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
        other -> other
      end
    catch
      {:fail, _} -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  defp bind_hops_reg(state, reg_idx, value) do
    case Map.get(state.regs, reg_idx) do
      {:unbound, id} ->
        state |> WamRuntime.trail_binding(id) |> WamRuntime.put_reg(id, value)
      ^value -> state
      _ -> nil
    end
  end
end
',
    [CamelMod, CamelPred,
     PredStr, EdgePredStr, MaxDepth,
     EdgeKey,
     MaxDepth,
     EdgeKey,
     CamelPred]).

%% compile_transitive_distance_dispatch_module(+ModuleName, +Pred, +EdgePred, -Code)
%
%  Dispatch module for the transitive_distance3 pattern (3-ary; the
%  shape detected by detect_transitive_distance/4 in
%  recursive_kernel_detection.pl). Routes calls through
%  WamRuntime.GraphKernel.TransitiveDistance.collect_pairs.
%
%  Unlike TC (where reg 2 alone is enumerated) or category_ancestor
%  (where reg 2 is bound on entry and only reg 3 is enumerated), here
%  BOTH reg 2 (target) and reg 3 (distance) get bound per solution.
%  The kernel returns [{target, distance}, ...]; the dispatch wrapper
%  inspects the active aggregate frames :agg_value_reg at runtime to
%  decide which slice of the pair to contribute:
%    - val_reg == 2 -> push targets       (`findall(T, td(s, T, _), Ts)`)
%    - val_reg == 3 -> push distances     (`findall(D, td(s, _, D), Ds)`)
%    - otherwise    -> push pair tuples   (`findall(T-D, ...)` etc.)
%
%  Driver-direct call (no aggregate frame): binds A2 and A3 to the
%  first solutions target and distance.
compile_transitive_distance_dispatch_module(ModuleName, Pred, EdgePred, Code) :-
    atom_string(ModuleName, ModName),
    camel_case(ModName, CamelMod),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    atom_string(EdgePred, EdgePredStr),
    format(string(EdgeKey), "~w/2", [EdgePredStr]),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc """
  Kernel-dispatched ~w/3 (auto-recognised transitive_distance3
  pattern over ~w/2). Routes calls through
  WamRuntime.GraphKernel.TransitiveDistance.

  Edge source must be registered via WamRuntime.FactSourceRegistry
  under the indicator "~w" before calling.
  """

  def run(%WamRuntime.WamState{} = state) do
    start_val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    edge_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    neighbors_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(edge_handle, node, state)
    end
    pairs =
      WamRuntime.GraphKernel.TransitiveDistance.collect_pairs(
        neighbors_fn, start_val)

    if WamRuntime.in_forkable_aggregate_frame?(state) do
      # Slice the pairs based on which register the active aggregate
      # is capturing. split_at_aggregate_cp/1 walks the cp stack ONCE
      # so the per-pair contribution is O(1) (no per-push cp walk,
      # same fix-shape as PR #1814 used for category_ancestor).
      {above, agg_cp, below} = WamRuntime.split_at_aggregate_cp(state)
      contributions =
        case agg_cp.agg_value_reg do
          2 -> Enum.map(pairs, fn {t, _d} -> t end)
          3 -> Enum.map(pairs, fn {_t, d} -> d end)
          _ -> pairs
        end
      # merge_into_aggregate appends; preserve encounter order.
      new_accum = agg_cp.agg_accum ++ contributions
      new_agg_cp = %{agg_cp | agg_accum: new_accum}
      new_state = %{state | choice_points: above ++ [new_agg_cp | below]}
      throw({:fail, new_state})
    else
      case pairs do
        [{first_target, first_dist} | _] ->
          case bind_two_regs(state, first_target, first_dist) do
            nil -> throw({:fail, state})
            bound -> {:ok, bound}
          end
        [] -> throw({:fail, state})
      end
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    {state, arg_vars} =
      Enum.with_index(args, 1)
      |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
        case arg do
          {:unbound, _} ->
            v = {:unbound, make_ref()}
            {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
          other ->
            {%{s | regs: Map.put(s.regs, i, other)}, vars}
        end
      end)
    state = %{state | arg_vars: arg_vars, cp: &WamRuntime.terminal_cp/1}
    try do
      case run(state) do
        {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
        other -> other
      end
    catch
      {:fail, _} -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  # Bind both A2 (target) and A3 (distance) for the driver-direct
  # single-solution path. Either binding failure (slot already bound
  # to a different value) returns nil so the caller throws fail.
  defp bind_two_regs(state, target, distance) do
    with {:ok, s1} <- bind_one(state, 2, target),
         {:ok, s2} <- bind_one(s1, 3, distance) do
      s2
    else
      _ -> nil
    end
  end

  defp bind_one(state, reg_idx, value) do
    case Map.get(state.regs, reg_idx) do
      {:unbound, id} ->
        {:ok,
         state
         |> WamRuntime.trail_binding(id)
         |> WamRuntime.put_reg(id, value)}
      ^value -> {:ok, state}
      _ -> :error
    end
  end
end
',
    [CamelMod, CamelPred,
     PredStr, EdgePredStr,
     EdgeKey,
     EdgeKey,
     CamelPred]).

%% compile_transitive_parent_distance_dispatch_module(+ModuleName, +Pred, +EdgePred, -Code)
%
%  Dispatch module for the transitive_parent_distance4 pattern (4-ary;
%  detected by detect_transitive_parent_distance/4 in
%  recursive_kernel_detection.pl). Routes calls through
%  WamRuntime.GraphKernel.TransitiveParentDistance.collect_triples.
%
%  Three enumerated registers per solution: A2 (target), A3 (parent
%  / immediate predecessor), A4 (distance). The kernel returns
%  [{target, parent, distance}, ...]; the dispatch wrapper inspects
%  the active aggregate frames :agg_value_reg at runtime to select
%  which slice to push:
%    - val_reg == 2 -> push targets
%    - val_reg == 3 -> push parents
%    - val_reg == 4 -> push distances
%    - otherwise    -> push triple tuples (compound template)
%
%  Driver-direct call: bind_three_regs/4 binds A2, A3, A4 to the
%  first solution.
%
%  WARNING (inherited from the kernel): NO cycle detection. Caller
%  must ensure acyclic input or wrap with a depth-bounded consumer.
compile_transitive_parent_distance_dispatch_module(ModuleName, Pred, EdgePred, Code) :-
    atom_string(ModuleName, ModName),
    camel_case(ModName, CamelMod),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    atom_string(EdgePred, EdgePredStr),
    format(string(EdgeKey), "~w/2", [EdgePredStr]),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc """
  Kernel-dispatched ~w/4 (auto-recognised transitive_parent_distance4
  pattern over ~w/2). Routes calls through
  WamRuntime.GraphKernel.TransitiveParentDistance.

  Edge source must be registered via WamRuntime.FactSourceRegistry
  under the indicator "~w" before calling.

  WARNING: the underlying kernel has NO cycle detection. Use this
  only on acyclic graphs (DAGs) or wrap calls with a depth-bound
  consumer.
  """

  def run(%WamRuntime.WamState{} = state) do
    start_val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    edge_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    neighbors_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(edge_handle, node, state)
    end
    triples =
      WamRuntime.GraphKernel.TransitiveParentDistance.collect_triples(
        neighbors_fn, start_val)

    if WamRuntime.in_forkable_aggregate_frame?(state) do
      # Slice triples based on which register the active aggregate
      # is capturing. split_at_aggregate_cp/1 walks the cp stack ONCE
      # so per-pair contribution is O(1) (PR #1814 amortisation).
      {above, agg_cp, below} = WamRuntime.split_at_aggregate_cp(state)
      contributions =
        case agg_cp.agg_value_reg do
          2 -> Enum.map(triples, fn {t, _p, _d} -> t end)
          3 -> Enum.map(triples, fn {_t, p, _d} -> p end)
          4 -> Enum.map(triples, fn {_t, _p, d} -> d end)
          _ -> triples
        end
      new_accum = agg_cp.agg_accum ++ contributions
      new_agg_cp = %{agg_cp | agg_accum: new_accum}
      new_state = %{state | choice_points: above ++ [new_agg_cp | below]}
      throw({:fail, new_state})
    else
      case triples do
        [{first_target, first_parent, first_dist} | _] ->
          case bind_three_regs(state, first_target, first_parent, first_dist) do
            nil -> throw({:fail, state})
            bound -> {:ok, bound}
          end
        [] -> throw({:fail, state})
      end
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    {state, arg_vars} =
      Enum.with_index(args, 1)
      |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
        case arg do
          {:unbound, _} ->
            v = {:unbound, make_ref()}
            {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
          other ->
            {%{s | regs: Map.put(s.regs, i, other)}, vars}
        end
      end)
    state = %{state | arg_vars: arg_vars, cp: &WamRuntime.terminal_cp/1}
    try do
      case run(state) do
        {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
        other -> other
      end
    catch
      {:fail, _} -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  # Bind A2 (target), A3 (parent), A4 (distance) for driver-direct
  # single-solution mode. Any binding failure (slot already bound to a
  # different value) aborts with nil so the caller throws fail.
  defp bind_three_regs(state, target, parent, distance) do
    with {:ok, s1} <- bind_one(state, 2, target),
         {:ok, s2} <- bind_one(s1, 3, parent),
         {:ok, s3} <- bind_one(s2, 4, distance) do
      s3
    else
      _ -> nil
    end
  end

  defp bind_one(state, reg_idx, value) do
    case Map.get(state.regs, reg_idx) do
      {:unbound, id} ->
        {:ok,
         state
         |> WamRuntime.trail_binding(id)
         |> WamRuntime.put_reg(id, value)}
      ^value -> {:ok, state}
      _ -> :error
    end
  end
end
',
    [CamelMod, CamelPred,
     PredStr, EdgePredStr,
     EdgeKey,
     EdgeKey,
     CamelPred]).

%% compile_transitive_step_parent_distance_dispatch_module(+ModuleName, +Pred, +EdgePred, -Code)
%
%  Dispatch module for the transitive_step_parent_distance5 pattern
%  (5-ary; detected by detect_transitive_step_parent_distance/4 in
%  recursive_kernel_detection.pl). Routes calls through
%  WamRuntime.GraphKernel.TransitiveStepParentDistance.collect_quads.
%
%  FOUR enumerated registers per solution: A2 (target), A3 (step —
%  the FIRST hop from start), A4 (parent — immediate predecessor of
%  target), A5 (distance). The wrapper inspects the active aggregate
%  frames :agg_value_reg at runtime to slice quads:
%    - val_reg == 2 -> push targets
%    - val_reg == 3 -> push first-hop steps
%    - val_reg == 4 -> push parents
%    - val_reg == 5 -> push distances
%    - otherwise    -> push quad tuples
%
%  Driver-direct call: bind_four_regs/5 binds A2, A3, A4, A5 to the
%  first solution.
%
%  WARNING (inherited from the kernel): NO cycle detection. Use on
%  acyclic graphs only.
compile_transitive_step_parent_distance_dispatch_module(ModuleName, Pred, EdgePred, Code) :-
    atom_string(ModuleName, ModName),
    camel_case(ModName, CamelMod),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    atom_string(EdgePred, EdgePredStr),
    format(string(EdgeKey), "~w/2", [EdgePredStr]),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc """
  Kernel-dispatched ~w/5 (auto-recognised transitive_step_parent_distance5
  pattern over ~w/2). Routes calls through
  WamRuntime.GraphKernel.TransitiveStepParentDistance.

  Edge source must be registered via WamRuntime.FactSourceRegistry
  under the indicator "~w" before calling.

  WARNING: the underlying kernel has NO cycle detection. Use this
  only on acyclic graphs (DAGs) or wrap calls with a depth-bound
  consumer.
  """

  def run(%WamRuntime.WamState{} = state) do
    start_val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    edge_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    neighbors_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(edge_handle, node, state)
    end
    quads =
      WamRuntime.GraphKernel.TransitiveStepParentDistance.collect_quads(
        neighbors_fn, start_val)

    if WamRuntime.in_forkable_aggregate_frame?(state) do
      # Slice quads on which register the active aggregate captures.
      # split_at_aggregate_cp/1 walks cp stack ONCE (PR #1814 amortisation).
      {above, agg_cp, below} = WamRuntime.split_at_aggregate_cp(state)
      contributions =
        case agg_cp.agg_value_reg do
          2 -> Enum.map(quads, fn {t, _s, _p, _d} -> t end)
          3 -> Enum.map(quads, fn {_t, s, _p, _d} -> s end)
          4 -> Enum.map(quads, fn {_t, _s, p, _d} -> p end)
          5 -> Enum.map(quads, fn {_t, _s, _p, d} -> d end)
          _ -> quads
        end
      new_accum = agg_cp.agg_accum ++ contributions
      new_agg_cp = %{agg_cp | agg_accum: new_accum}
      new_state = %{state | choice_points: above ++ [new_agg_cp | below]}
      throw({:fail, new_state})
    else
      case quads do
        [{first_target, first_step, first_parent, first_dist} | _] ->
          case bind_four_regs(state, first_target, first_step, first_parent, first_dist) do
            nil -> throw({:fail, state})
            bound -> {:ok, bound}
          end
        [] -> throw({:fail, state})
      end
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    {state, arg_vars} =
      Enum.with_index(args, 1)
      |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
        case arg do
          {:unbound, _} ->
            v = {:unbound, make_ref()}
            {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
          other ->
            {%{s | regs: Map.put(s.regs, i, other)}, vars}
        end
      end)
    state = %{state | arg_vars: arg_vars, cp: &WamRuntime.terminal_cp/1}
    try do
      case run(state) do
        {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
        other -> other
      end
    catch
      {:fail, _} -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  # Bind A2 (target), A3 (step), A4 (parent), A5 (distance) for
  # driver-direct single-solution mode. Any binding failure aborts
  # with nil so the caller throws fail.
  defp bind_four_regs(state, target, step, parent, distance) do
    with {:ok, s1} <- bind_one(state, 2, target),
         {:ok, s2} <- bind_one(s1, 3, step),
         {:ok, s3} <- bind_one(s2, 4, parent),
         {:ok, s4} <- bind_one(s3, 5, distance) do
      s4
    else
      _ -> nil
    end
  end

  defp bind_one(state, reg_idx, value) do
    case Map.get(state.regs, reg_idx) do
      {:unbound, id} ->
        {:ok,
         state
         |> WamRuntime.trail_binding(id)
         |> WamRuntime.put_reg(id, value)}
      ^value -> {:ok, state}
      _ -> :error
    end
  end
end
',
    [CamelMod, CamelPred,
     PredStr, EdgePredStr,
     EdgeKey,
     EdgeKey,
     CamelPred]).

%% compile_weighted_shortest_path_dispatch_module(+ModuleName, +Pred, +EdgePred, -Code)
%
%  Dispatch module for the weighted_shortest_path3 pattern (3-ary
%  predicate over a 3-ary weighted_edge/3 source). Routes calls
%  through WamRuntime.GraphKernel.WeightedShortestPath.collect_path_costs.
%
%  TWO enumerated registers per solution: A2 (target), A3 (cost). The
%  wrapper inspects the active aggregate frames :agg_value_reg at
%  runtime to slice the (target, cost) pair:
%    - val_reg == 2 -> push targets
%    - val_reg == 3 -> push costs
%    - otherwise    -> push pair tuples
%
%  Driver-direct call: bind_two_regs/3 binds A2, A3 to the first
%  shortest-path solution.
%
%  Semantic narrowing (documented in the runtime kernels moduledoc):
%  the canonical Prolog pattern enumerates ALL paths; the kernel
%  computes only the SHORTEST. Match the Rust reference impl.
compile_weighted_shortest_path_dispatch_module(ModuleName, Pred, EdgePred, Code) :-
    atom_string(ModuleName, ModName),
    camel_case(ModName, CamelMod),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    atom_string(EdgePred, EdgePredStr),
    format(string(EdgeKey), "~w/3", [EdgePredStr]),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc """
  Kernel-dispatched ~w/3 (auto-recognised weighted_shortest_path3
  pattern over ~w/3). Routes calls through
  WamRuntime.GraphKernel.WeightedShortestPath (Dijkstra).

  Edge source must be registered via WamRuntime.FactSourceRegistry
  under the indicator "~w" before calling. The registered handle
  must support arity 3 and return `[{from, to, weight}, ...]` from
  lookup_by_arg1.

  Semantic narrowing: the canonical Prolog pattern enumerates all
  paths from start to target with their summed weights; this kernel
  computes only the SHORTEST path cost (one result per reachable
  target). Match the Rust reference impl. See the kernel moduledoc
  in WamRuntime.GraphKernel.WeightedShortestPath for the full
  contract.
  """

  def run(%WamRuntime.WamState{} = state) do
    start_val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    edge_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    weighted_edges_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(edge_handle, node, state)
    end
    pairs =
      WamRuntime.GraphKernel.WeightedShortestPath.collect_path_costs(
        weighted_edges_fn, start_val)

    if WamRuntime.in_forkable_aggregate_frame?(state) do
      # Slice (target, cost) on which register the aggregate captures.
      # split_at_aggregate_cp/1 walks cp stack ONCE (PR #1814 fix).
      {above, agg_cp, below} = WamRuntime.split_at_aggregate_cp(state)
      contributions =
        case agg_cp.agg_value_reg do
          2 -> Enum.map(pairs, fn {t, _w} -> t end)
          3 -> Enum.map(pairs, fn {_t, w} -> w end)
          _ -> pairs
        end
      new_accum = agg_cp.agg_accum ++ contributions
      new_agg_cp = %{agg_cp | agg_accum: new_accum}
      new_state = %{state | choice_points: above ++ [new_agg_cp | below]}
      throw({:fail, new_state})
    else
      case pairs do
        [{first_target, first_cost} | _] ->
          case bind_two_regs(state, first_target, first_cost) do
            nil -> throw({:fail, state})
            bound -> {:ok, bound}
          end
        [] -> throw({:fail, state})
      end
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    {state, arg_vars} =
      Enum.with_index(args, 1)
      |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
        case arg do
          {:unbound, _} ->
            v = {:unbound, make_ref()}
            {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
          other ->
            {%{s | regs: Map.put(s.regs, i, other)}, vars}
        end
      end)
    state = %{state | arg_vars: arg_vars, cp: &WamRuntime.terminal_cp/1}
    try do
      case run(state) do
        {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
        other -> other
      end
    catch
      {:fail, _} -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  defp bind_two_regs(state, target, cost) do
    with {:ok, s1} <- bind_one(state, 2, target),
         {:ok, s2} <- bind_one(s1, 3, cost) do
      s2
    else
      _ -> nil
    end
  end

  defp bind_one(state, reg_idx, value) do
    case Map.get(state.regs, reg_idx) do
      {:unbound, id} ->
        {:ok,
         state
         |> WamRuntime.trail_binding(id)
         |> WamRuntime.put_reg(id, value)}
      ^value -> {:ok, state}
      _ -> :error
    end
  end
end
',
    [CamelMod, CamelPred,
     PredStr, EdgePredStr,
     EdgeKey,
     EdgeKey,
     CamelPred]).

%% compile_astar_shortest_path_dispatch_module(+ModuleName, +Pred, +EdgePred, +DirectPred, +Dim, -Code)
%
%  Dispatch module for the astar_shortest_path4 pattern. Routes
%  calls through WamRuntime.GraphKernel.AstarShortestPath. Reads
%  TWO FactSources at runtime (weighted_edge + direct_dist edges)
%  via the registry, plus three argument registers:
%    - A1: start (must be bound to an atom)
%    - A2: target (atom for goal-directed mode; unbound for Dijkstra
%      fallback)
%    - A3: dim (number on entry overrides default; unbound uses
%      compile-time Dim from kernel config)
%    - A4: cost (typically unbound; gets bound on success)
%
%  TWO enumerated registers per solution: A2 (target, only if it
%  was unbound on entry) and A4 (cost). When target is bound on
%  entry, the kernel returns AT MOST ONE solution (the goal cost).
%
%  Driver-direct call: bind_two_regs/4 binds A2 and A4 (skipping
%  A2 if already bound) to the first solution.
compile_astar_shortest_path_dispatch_module(ModuleName, Pred, EdgePred, DirectPred, Dim, Code) :-
    atom_string(ModuleName, ModName),
    camel_case(ModName, CamelMod),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    atom_string(EdgePred, EdgePredStr),
    atom_string(DirectPred, DirectPredStr),
    format(string(EdgeKey), "~w/3", [EdgePredStr]),
    format(string(DirectKey), "~w/3", [DirectPredStr]),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc """
  Kernel-dispatched ~w/4 (auto-recognised astar_shortest_path4 pattern
  over ~w/3 with ~w/3 as the heuristic source). Routes calls through
  WamRuntime.GraphKernel.AstarShortestPath.

  Both edge sources must be registered via WamRuntime.FactSourceRegistry:
    - "~w" — forward weighted edges
    - "~w" — direct heuristic distances (may be the same indicator
      if no separate direct_dist_pred was configured at compile time)
  """

  @default_dim ~w

  def run(%WamRuntime.WamState{} = state) do
    start_val = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 1))
    target_raw = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 2))
    dim_raw = WamRuntime.deref_var(state, WamRuntime.get_reg(state, 3))

    # target: nil = unbound (Dijkstra mode); else the atom/binary.
    target_val =
      case target_raw do
        {:unbound, _} -> nil
        v -> v
      end

    # dim: numeric override on entry; else the compile-time default.
    dim_val =
      case dim_raw do
        {:unbound, _} -> @default_dim
        n when is_number(n) -> n
        _ -> @default_dim
      end

    weighted_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    direct_handle = WamRuntime.FactSourceRegistry.lookup!("~w")
    weighted_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(weighted_handle, node, state)
    end
    direct_fn = fn node ->
      WamRuntime.FactSource.lookup_by_arg1(direct_handle, node, state)
    end

    pairs =
      WamRuntime.GraphKernel.AstarShortestPath.collect_path_costs(
        weighted_fn, direct_fn, start_val, target_val, dim_val)

    # Post-filter: when target was bound on entry, surface only its
    # entry from the dist map (matches Rust dispatchs `results.retain`
    # behaviour). When target was unbound, return the full set
    # (Dijkstra mode).
    pairs =
      case target_val do
        nil -> pairs
        t -> Enum.filter(pairs, fn {n, _} -> n == t end)
      end

    if WamRuntime.in_forkable_aggregate_frame?(state) do
      {above, agg_cp, below} = WamRuntime.split_at_aggregate_cp(state)
      contributions =
        case agg_cp.agg_value_reg do
          2 -> Enum.map(pairs, fn {t, _w} -> t end)
          4 -> Enum.map(pairs, fn {_t, w} -> w end)
          _ -> pairs
        end
      new_accum = agg_cp.agg_accum ++ contributions
      new_agg_cp = %{agg_cp | agg_accum: new_accum}
      new_state = %{state | choice_points: above ++ [new_agg_cp | below]}
      throw({:fail, new_state})
    else
      case pairs do
        [{first_target, first_cost} | _] ->
          case bind_target_and_cost(state, target_val, first_target, first_cost) do
            nil -> throw({:fail, state})
            bound -> {:ok, bound}
          end
        [] -> throw({:fail, state})
      end
    end
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
    {state, arg_vars} =
      Enum.with_index(args, 1)
      |> Enum.reduce({state, []}, fn {arg, i}, {s, vars} ->
        case arg do
          {:unbound, _} ->
            v = {:unbound, make_ref()}
            {%{s | regs: Map.put(s.regs, i, v)}, [{i, v} | vars]}
          other ->
            {%{s | regs: Map.put(s.regs, i, other)}, vars}
        end
      end)
    state = %{state | arg_vars: arg_vars, cp: &WamRuntime.terminal_cp/1}
    try do
      case run(state) do
        {:ok, final} -> {:ok, WamRuntime.materialise_args(final)}
        other -> other
      end
    catch
      {:fail, _} -> :fail
      {:return, result} -> result
      error ->
        IO.puts("Predicate ~w CRASHED: #{inspect(error)}")
        :fail
    end
  end

  # If target was bound on entry, A2 is already set — skip rebinding.
  # Always bind A4 (cost).
  defp bind_target_and_cost(state, nil, first_target, first_cost) do
    # Target was unbound on entry — bind both A2 and A4.
    with {:ok, s1} <- bind_one(state, 2, first_target),
         {:ok, s2} <- bind_one(s1, 4, first_cost) do
      s2
    else
      _ -> nil
    end
  end
  defp bind_target_and_cost(state, _bound_target, _first_target, first_cost) do
    # Target was bound on entry — only bind A4.
    case bind_one(state, 4, first_cost) do
      {:ok, s} -> s
      _ -> nil
    end
  end

  defp bind_one(state, reg_idx, value) do
    case Map.get(state.regs, reg_idx) do
      {:unbound, id} ->
        {:ok,
         state
         |> WamRuntime.trail_binding(id)
         |> WamRuntime.put_reg(id, value)}
      ^value -> {:ok, state}
      _ -> :error
    end
  end
end
',
    [CamelMod, CamelPred,
     PredStr, EdgePredStr, DirectPredStr,
     EdgeKey,
     DirectKey,
     Dim,
     EdgeKey,
     DirectKey,
     CamelPred]).

atom_interning_cleanup(Options) :-
    (   option(intern_atoms(true), Options)
    ->  retractall(wam_elixir_lowered_emitter:intern_atoms_enabled)
    ;   true
    ).

do_write_wam_elixir_project(Predicates, Options, ProjectDir,
                             Mode, ModuleName, LibDir) :-
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
    % Generate predicate modules. When kernel_dispatch(true) is set,
    % run the shared recursive-kernel detector (target-neutral; same
    % module Rust uses) on each predicate. If a kernel kind is
    % detected, emit a kernel-dispatch module instead of the WAM
    % lower chain. Today supports transitive_closure2 and
    % category_ancestor; adding more kernel kinds is just an Elixir-
    % runtime port + an emit_kernel_dispatch_module clause.
    forall(
        member(Pred/Arity-WamCode, Predicates),
        (   (   option(kernel_dispatch(true), Options),
                Mode == lowered,
                detect_kernel_for_predicate(Pred, Arity, Kernel)
            ->  emit_kernel_dispatch_module(ModuleName, Pred, Arity, Kernel, PredCode)
            ;   (   Mode == lowered
                ->  lower_predicate_to_elixir(Pred/Arity, WamCode, Options, PredCode)
                ;   compile_wam_predicate_to_elixir(Pred/Arity, WamCode, Options, PredCode)
                )
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
