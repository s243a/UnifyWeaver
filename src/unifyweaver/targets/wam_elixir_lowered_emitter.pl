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
    wam_elixir_lower_instr/5,     % +Instr, +PC, +Labels, +FuncName, -Code
    % Phase A fact-shape classification (see docs/proposals/WAM_FACT_SHAPE_*.md).
    % Emitter-internal for now; may move to its own module once other
    % targets adopt the same classification.
    classify_predicate/4,         % +PredIndicator, +Segments, +Options, -Info
    clause_count/2,               % +Segments, -N
    fact_only/2,                  % +Segments, -Bool
    first_arg_groundness/3        % +Segments, +Arity, -Status
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
    % Phase A: classify the predicate and record the chosen layout as a
    % module-level comment. Phase A emits every predicate via the
    % existing `compiled` path regardless of the chosen layout — this
    % is observation-only. Phase B switches on Info.layout.
    classify_predicate(Pred/Arity, Segments, Options, Info),
    format_fact_shape_comment(Info, ShapeComment),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc "Lowered WAM-compiled predicate: ~w/~w"

~w

  # run/1 is a plain tail call into the first clause segment. No
  # try/catch here — failures propagate via throw({:fail, state}) up to
  # the outermost catch in run(args). Without this, every cross-predicate
  # dispatch (WamDispatcher.call → Pred.run) would add a try frame that
  # defeats tail-call optimisation. Carrying state in the throw lets each
  # enclosing catch backtrack on the most-current state (with CPs pushed
  # during body execution intact), not the pre-try snapshot.
  def run(%WamRuntime.WamState{} = state), do: ~w(state)

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
    # state.cp holds the current continuation — a function of
    # `(state) -> {:ok, state} | :fail`. At the top level we terminate
    # in WamRuntime.terminal_cp/1 which just returns {:ok, state}.
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

~w
end', [CamelMod, CamelPred, PredStr, Arity, ShapeComment, FirstFunc, CamelPred, FuncsBody]).

% ============================================================================
% PHASE A: FACT-SHAPE CLASSIFICATION
% ============================================================================
%
% Observation-only in this phase — the chosen `Layout` field is
% surfaced as a module-level comment, but every predicate still emits
% via the existing `compiled` code path. Phase B switches actual
% emission on Layout. See docs/proposals/WAM_FACT_SHAPE_SPEC.md for
% the wider contract.

%% classify_predicate(+PredIndicator, +Segments, +Options, -Info)
%  Info is the term `fact_shape_info(NClauses, FactOnly, FirstArg, Layout)`.
classify_predicate(Pred/Arity, Segments, Options, Info) :-
    clause_count(Segments, NClauses),
    fact_only(Segments, FactOnly),
    first_arg_groundness(Segments, Arity, FirstArg),
    pick_layout(Pred/Arity, NClauses, FactOnly, Options, Layout),
    Info = fact_shape_info(NClauses, FactOnly, FirstArg, Layout).

%% clause_count(+Segments, -N)
%  Each top-level segment corresponds to one clause (or the predicate
%  header segment, which is still per-clause since the WAM emitter
%  synthesises one segment per clause). CPS sub-segments (`_kN`) are
%  generated later and never appear in this list.
clause_count(Segments, N) :-
    length(Segments, N).

%% fact_only(+Segments, -Bool)
%  `true` iff no clause has a body-level call.
fact_only(Segments, true) :-
    forall(member(_-Instrs, Segments),
           forall(member(_-Instr, Instrs),
                  \+ is_body_call_instr(Instr))), !.
fact_only(_, false).

is_body_call_instr(call(_, _)).
is_body_call_instr(execute(_)).
is_body_call_instr(builtin_call(_, _)).

%% first_arg_groundness(+Segments, +Arity, -Status)
%  Status is one of: `none` (arity 0), `all_ground`, `all_variable`,
%  `mixed`. Determined by which head-unification instruction binds A1
%  in each clause.
first_arg_groundness(_Segments, 0, none) :- !.
first_arg_groundness(Segments, Arity, Status) :-
    Arity > 0,
    maplist(clause_arg1_type, Segments, Types),
    combine_groundness(Types, Status).

clause_arg1_type(_-Instrs, Type) :-
    (   member(_-get_constant(_, "A1"), Instrs) -> Type = ground
    ;   member(_-get_structure(_, "A1"), Instrs) -> Type = ground
    ;   member(_-get_variable(_, "A1"), Instrs) -> Type = variable
    ;   member(_-get_value(_, "A1"), Instrs)    -> Type = variable
    ;   Type = unknown
    ).

combine_groundness(Types, all_ground)   :- forall(member(T, Types), T == ground), !.
combine_groundness(Types, all_variable) :- forall(member(T, Types), T == variable), !.
combine_groundness(_, mixed).

%% pick_layout(+PredIndicator, +NClauses, +FactOnly, +Options, -Layout)
%  User override via `fact_layout/2` in Options or `user:fact_layout/2`
%  always wins. Otherwise default policy: fact-only ∧ count > threshold
%  → `inline_data`; else `compiled`. Threshold from `fact_count_threshold`
%  option, default 100.
pick_layout(PredIndicator, _NClauses, _FactOnly, Options, Layout) :-
    option(fact_layout(PredIndicator, UserLayout), Options), !,
    Layout = UserLayout.
pick_layout(PredIndicator, _NClauses, _FactOnly, _Options, Layout) :-
    catch(user:fact_layout(PredIndicator, UserLayout), _, fail), !,
    Layout = UserLayout.
pick_layout(_PredIndicator, NClauses, FactOnly, Options, Layout) :-
    option(fact_count_threshold(Threshold), Options, 100),
    (   FactOnly == true, NClauses > Threshold
    ->  Layout = inline_data([])
    ;   Layout = compiled
    ).

%% format_fact_shape_comment(+Info, -Comment)
%  Renders `Info` as an Elixir comment surfaced in the generated
%  module. Phase A uses this purely as documentation; Phase B+ reads
%  Info to pick the emission path.
format_fact_shape_comment(fact_shape_info(N, FactOnly, FirstArg, Layout), Comment) :-
    format(string(Comment),
'  # Phase-A fact-shape classification (observation-only):
  #   clauses=~w fact_only=~w first_arg=~w layout=~w',
           [N, FactOnly, FirstArg, Layout]).

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

generate_all_segments(Segments, Labels, SegCodes) :-
    % Build a list of per-segment code-lists and flatten with append/2
    % at the end. The previous recursive append(ThisSegCodes,
    % RestCodes, _) was O(N²) over segment count — noticeable on
    % predicates with thousands of fact clauses.
    maplist(generate_one_segment(Labels), Segments, SegCodesNested),
    append(SegCodesNested, SegCodes).

generate_one_segment(Labels, Name-Instrs, ThisSegCodes) :-
    segment_func_name(Name, FuncName),
    classify_segment_head(Instrs, HeadType, BodyInstrs),
    % CPS split: every non-tail `call P, N` terminates a sub-segment;
    % subsequent instrs go into a fresh continuation function. This lets
    % backtrack re-enter a retry point without losing the outer caller's
    % post-call code (which Elixir would otherwise have on a collapsed
    % tail-call stack).
    split_body_at_calls(BodyInstrs, SubSegs),
    emit_sub_segments(SubSegs, FuncName, HeadType, Labels, ThisSegCodes).

%% split_body_at_calls(+Instrs, -SubSegs)
%  Splits a flat instr list into sub-segment lists, cutting after every
%  `call P, N` opcode. The call is the last element of its sub-segment;
%  the next sub-segment starts with the first instr after it. A body with
%  no `call`s yields exactly one sub-segment.
split_body_at_calls(Instrs, SubSegs) :-
    split_body_at_calls_(Instrs, [], SubSegs).

split_body_at_calls_([], AccRev, [Seg]) :-
    reverse(AccRev, Seg).
split_body_at_calls_([PC-call(P, N) | Rest], AccRev, [Seg | RestSegs]) :-
    !,
    reverse([PC-call(P, N) | AccRev], Seg),
    split_body_at_calls_(Rest, [], RestSegs).
split_body_at_calls_([Instr | Rest], AccRev, Segs) :-
    split_body_at_calls_(Rest, [Instr | AccRev], Segs).

%% emit_sub_segments(+SubSegs, +BaseFunc, +HeadType, +Labels, -Codes)
%  Emits one `defp` per sub-segment. The first uses wrap_segment (with
%  CP push for try_me_else / retry_me_else). Subsequent sub-segments are
%  plain `defp BaseFunc_kN(state) do ... end` continuations. Every
%  sub-segment except the last ends with a tail call to the next
%  continuation; the last ends with its natural tail (execute/proceed).
emit_sub_segments([OnlySeg], BaseFunc, HeadType, Labels, [Code]) :-
    lower_instr_list(OnlySeg, Labels, BaseFunc, Exprs),
    atomic_list_concat(Exprs, '\n', BodyCode),
    wrap_segment(BaseFunc, HeadType, BodyCode, Code).

emit_sub_segments([FirstSeg | MoreSegs], BaseFunc, HeadType, Labels, [FirstCode | MoreCodes]) :-
    MoreSegs = [_|_],
    format(string(NextFunc), '~w_k1', [BaseFunc]),
    lower_seg_with_continuation(FirstSeg, NextFunc, Labels, BaseFunc, FirstBody),
    wrap_segment(BaseFunc, HeadType, FirstBody, FirstCode),
    emit_cont_segments(MoreSegs, BaseFunc, 1, Labels, MoreCodes).

emit_cont_segments([FinalSeg], BaseFunc, Idx, Labels, [Code]) :-
    format(string(FuncName), '~w_k~w', [BaseFunc, Idx]),
    lower_instr_list(FinalSeg, Labels, FuncName, Exprs),
    atomic_list_concat(Exprs, '\n', BodyCode),
    format(string(Code),
'  defp ~w(state) do
    try do
~w
    catch
      {:fail, s} ->
        case WamRuntime.backtrack(s) do
          :fail -> throw({:fail, %{s | choice_points: []}})
          other -> other
        end
    end
  end', [FuncName, BodyCode]).
emit_cont_segments([Seg | More], BaseFunc, Idx, Labels, [Code | RestCodes]) :-
    More = [_|_],
    format(string(FuncName), '~w_k~w', [BaseFunc, Idx]),
    NextIdx is Idx + 1,
    format(string(NextFunc), '~w_k~w', [BaseFunc, NextIdx]),
    lower_seg_with_continuation(Seg, NextFunc, Labels, FuncName, Body),
    format(string(Code),
'  defp ~w(state) do
    try do
~w
    catch
      {:fail, s} ->
        case WamRuntime.backtrack(s) do
          :fail -> throw({:fail, %{s | choice_points: []}})
          other -> other
        end
    end
  end', [FuncName, Body]),
    emit_cont_segments(More, BaseFunc, NextIdx, Labels, RestCodes).

%% lower_seg_with_continuation(+Instrs, +NextFunc, +Labels, +FuncName, -Body)
%  Lowers a sub-segment that ends with `call P, N`. The body is the
%  ordinary lowering of the pre-call instrs, followed by a tail-call that
%  stores NextFunc in state.cp so the called predicate\'s `proceed`
%  routes control back to NextFunc rather than collapsing the stack.
lower_seg_with_continuation(Instrs, NextFunc, Labels, FuncName, Body) :-
    append(InitInstrs, [_PC-call(P, _N)], Instrs),
    lower_instr_list(InitInstrs, Labels, FuncName, InitExprs),
    format(string(TailCallCode),
'    state = %{state | cp: &~w/1}
    WamDispatcher.call("~w", state)', [NextFunc, P]),
    append(InitExprs, [TailCallCode], AllExprs),
    atomic_list_concat(AllExprs, '\n', Body).

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
        % Drop the outer try_me_else CP (pushed just before this switch):
        % we are deterministically dispatching to LocalFunc, so the outer
        % CP — which points at that same clause — would cause a duplicate
        % solution on backtracking. The inline-dispatched clause pushes
        % its own retry CP, which remains correct.
        format(string(Arm),
               '"~w" -> throw({:return, ~w(%{state | choice_points: tl(state.choice_points)})})',
               [Key, LocalFunc])
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
    try do
~w
    catch
      {:fail, s} ->
        case WamRuntime.backtrack(s) do
          :fail -> throw({:fail, %{s | choice_points: []}})
          other -> other
        end
    end
  end', [FuncName, FallbackFunc, BodyCode]).

wrap_segment(FuncName, retry_me_else(L), BodyCode, Code) :-
    segment_func_name(L, FallbackFunc),
    format(string(Code),
'  defp ~w(state) do
    cp = %{pc: &~w/1, regs: state.regs, heap: state.heap, heap_len: state.heap_len,
           cp: state.cp, trail: state.trail, trail_len: state.trail_len, stack: state.stack}
    state = %{state | choice_points: [cp | state.choice_points]}
    try do
~w
    catch
      {:fail, s} ->
        case WamRuntime.backtrack(s) do
          :fail -> throw({:fail, %{s | choice_points: []}})
          other -> other
        end
    end
  end', [FuncName, FallbackFunc, BodyCode]).

wrap_segment(FuncName, trust_me, BodyCode, Code) :-
    format(string(Code),
'  defp ~w(state) do
    try do
~w
    catch
      {:fail, s} ->
        case WamRuntime.backtrack(s) do
          :fail -> throw({:fail, %{s | choice_points: []}})
          other -> other
        end
    end
  end', [FuncName, BodyCode]).

wrap_segment(FuncName, none, BodyCode, Code) :-
    format(string(Code),
'  defp ~w(state) do
    try do
~w
    catch
      {:fail, s} ->
        case WamRuntime.backtrack(s) do
          :fail -> throw({:fail, %{s | choice_points: []}})
          other -> other
        end
    end
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
      true -> throw({:fail, state})
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
      :fail -> throw({:fail, state})
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
          :fail -> throw({:fail, state})
          s -> s
        end
      true -> throw({:fail, state})
    end', [Ai, F, Ai, Ai, Arity, F, Arity]).

wam_elixir_lower_instr(unify_variable(XnName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_variable(state, ~w) do
      :fail -> throw({:fail, state})
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_value(XnName), _PC, _Labels, _FuncName, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_value(state, ~w) do
      :fail -> throw({:fail, state})
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_constant(C), _PC, _Labels, _FuncName, Code) :-
    format(string(Code),
'    state = case WamRuntime.step_unify_constant(state, "~w") do
      :fail -> throw({:fail, state})
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
      :fail -> throw({:fail, state})
    end', [P, NPC]).

wam_elixir_lower_instr(execute(P), _PC, _Labels, _FuncName, Code) :-
    format(string(Code),
'    WamDispatcher.call("~w", state)', [P]).

wam_elixir_lower_instr(builtin_call(Op, Ar), _PC, _Labels, _FuncName, Code) :-
    format(string(Code),
'    state = case WamRuntime.execute_builtin(state, "~w", ~w) do
      :fail -> throw({:fail, state})
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
          :fail -> throw({:fail, state})
          s -> s
        end
      is_list(val) ->
        [h | t] = val
        %{state | stack: [{:unify_ctx, [h, t]} | state.stack]}
      true -> throw({:fail, state})
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
          _ -> throw({:fail, state})
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
          _ -> throw({:fail, state})
        end
    end', [ArmsStr]).

wam_elixir_lower_instr(proceed, _PC, _Labels, _FuncName, Code) :-
    % CPS: proceed = tail-call the continuation stored in state.cp. The
    % caller\'s post-call code (or the driver\'s terminal_cp) lives in
    % state.cp — this tail call invokes it. BEAM TCO collapses the stack
    % so deep recursion doesn\'t grow it.
    Code = '    state.cp.(state)'.

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
