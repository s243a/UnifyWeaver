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
    wam_elixir_lower_instr/6,     % +Instr, +PC, +Labels, +FuncName, +Suffix, -Code
    % Phase A fact-shape classification (see docs/proposals/WAM_FACT_SHAPE_*.md).
    % Emitter-internal for now; may move to its own module once other
    % targets adopt the same classification.
    classify_predicate/4,         % +PredIndicator, +Segments, +Options, -Info
    clause_count/2,               % +Segments, -N
    fact_only/2,                  % +Segments, -Bool
    first_arg_groundness/3,       % +Segments, +Arity, -Status
    % Phase B fact extraction. Exported so tests can exercise it.
    extract_facts/3,              % +Segments, +Arity, -ElixirListLiteral
    % Phase C first-argument indexing.
    extract_arg1_index/3,         % +Segments, +Arity, -IndexResult
    % Tier-2 purity gate (see docs/design/WAM_TIERED_LOWERING.md).
    % Consumed by `par_wrap_segment/4` to decide whether a predicate
    % is eligible for host-native parallel emission.
    tier2_purity_eligible/3,      % +Pred, +Arity, -Cert
    % Tier-2 super-wrapper emitter. Takes clause segments + options,
    % emits either the cond-based Task.async_stream fan-out or an
    % empty string (gates rejected). Not wired into the main emission
    % path yet — live hook-up is a follow-on PR.
    par_wrap_segment/4            % +Pred/Arity, +Segments, +Options, -Code
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(pairs), [group_pairs_by_key/2]).
:- use_module('wam_elixir_utils', [reg_id/2, is_label_part/1, camel_case/2, parse_arity/2]).
:- use_module('../core/purity_certificate', [analyze_predicate_purity/2]).
:- use_module('../core/predicate_preprocessing',
              [declared_preprocess_metadata/4]).

% ============================================================================
% MAIN ENTRY POINT
% ============================================================================

%% lower_predicate_to_elixir(+PredIndicator, +WamCode, +Options, -Code)
%
%  Dispatches on the Phase-A classification layout:
%    - `compiled`    — emit one `defp` per clause (unchanged path)
%    - `inline_data` — emit `@facts [{...}, ...]` and a single
%      `run/1` that delegates to `WamRuntime.stream_facts/3`
%    - `external_source` — Phase D, currently falls back to `compiled`
%
%  Falling back to `compiled` is always safe and preserves correctness,
%  so an extraction failure (e.g., a compound head arg we can\'t yet
%  represent inline) silently falls back instead of erroring.
lower_predicate_to_elixir(Pred/Arity, WamCode, Options, Code) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    collect_labels(Lines, 1, Labels),
    split_into_segments(Lines, 1, Segments),
    atom_string(Pred, PredStr),
    camel_case(PredStr, CamelPred),
    option(module_name(ModName), Options, 'WamPredLow'),
    camel_case(ModName, CamelMod),
    classify_predicate(Pred/Arity, Segments, Options, Info),
    format_fact_shape_comment(Info, ShapeComment),
    Info = fact_shape_info(_, _, _, Layout),
    (   Layout = external_source(_)
    ->  render_external_source_module(CamelMod, CamelPred, PredStr, Arity,
                                      ShapeComment, Layout, Code)
    ;   Layout = external_source(_, _)
    ->  render_external_source_module(CamelMod, CamelPred, PredStr, Arity,
                                      ShapeComment, Layout, Code)
    ;   Layout = inline_data(_),
        catch(extract_facts(Segments, Arity, FactsLiteral), _, fail),
        choose_index(Segments, Arity, Options, IndexResult)
    ->  render_inline_data_module(CamelMod, CamelPred, PredStr, Arity,
                                  ShapeComment, FactsLiteral, IndexResult, Code)
    ;   render_compiled_module(CamelMod, CamelPred, Pred/Arity, PredStr,
                               ShapeComment, Segments, Labels, Options, Code)
    ).

%% choose_index(+Segments, +Arity, +Options, -IndexResult)
%  IndexResult is `indexed(MapLiteral)` when Phase-C first-arg indexing
%  applies (arity >= 1 ∧ every clause has a ground arg1 ∧ policy allows
%  it), or `no_index` otherwise. The policy is the `fact_index_policy`
%  option: `none` disables; `auto` (default) and `first_arg` enable.
choose_index(_Segments, 0, _Options, no_index) :- !.
choose_index(Segments, Arity, Options, IndexResult) :-
    Arity >= 1,
    option(fact_index_policy(Policy), Options, auto),
    (   Policy == none
    ->  IndexResult = no_index
    ;   catch(extract_arg1_index(Segments, Arity, IndexResult), _, IndexResult = no_index)
    ).

render_compiled_module(CamelMod, CamelPred, PredIndicator, PredStr, ShapeComment,
                       Segments, Labels, Options, Code) :-
    PredIndicator = Pred/Arity,
    % Tier-2 decision point: if par_wrap_segment/4 emits a super-wrapper,
    % the per-clause bodies need to live under `_impl`-suffixed names so
    % the canonical `clause_main` slot can be taken by the super-wrapper.
    % When the gate rejects, Tier2Wrapper = "" and Suffix = "" — every
    % emitted byte is byte-for-byte identical to the pre-wiring output.
    par_wrap_segment(Pred/Arity, Segments, Options, Tier2Wrapper),
    (   Tier2Wrapper == ""
    ->  Suffix = "",
        Tier2Extras = "",
        Tier2ModAttr = ""
    ;   Suffix = "_impl",
        % The super-wrapper (Tier2Wrapper) already calls the
        % `*_impl` entry directly from its gate-miss cond arms, so
        % no separate sequential alias is emitted here.
        Tier2Extras = Tier2Wrapper,
        Tier2ModAttr = '  @tier2_eligible true\n'
    ),
    generate_all_segments(Segments, Labels, Suffix, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FuncsBody),
    % `def run(state)` always delegates to `clause_main` — which is the
    % super-wrapper under Tier 2, or the first clause's body pre-Tier-2.
    % Both end up at the same surface name.
    Segments = [FirstSegName-_|_],
    segment_func_name(FirstSegName, "", FirstFunc),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc "Lowered WAM-compiled predicate: ~w/~w"
~w
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

~w
end', [CamelMod, CamelPred, PredStr, Arity, ShapeComment, Tier2ModAttr,
       FirstFunc, CamelPred, FuncsBody, Tier2Extras]).

render_inline_data_module(CamelMod, CamelPred, PredStr, Arity, ShapeComment,
                          FactsLiteral, IndexResult, Code) :-
    inline_data_index_block(IndexResult, IndexBlock),
    inline_data_run1_body(IndexResult, Arity, RunBody),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc "Lowered WAM-compiled predicate: ~w/~w (inline_data)"

~w

  # Phase-B inline_data layout: facts live as a module attribute, not as
  # per-fact defp functions. run/1 delegates to WamRuntime.stream_facts,
  # which iterates @facts, attempts unification on each tuple, and
  # pushes a fact-stream CP on success so backtracking resumes at the
  # next tuple. :_var in a tuple slot means the head arg was a variable
  # — it unifies trivially with any incoming value.
  @facts ~w
~w

  def run(%WamRuntime.WamState{} = state) do
~w  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
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
end', [CamelMod, CamelPred, PredStr, Arity, ShapeComment,
       FactsLiteral, IndexBlock, RunBody, CamelPred]).

%% inline_data_index_block(+IndexResult, -Block)
%  Emits the `@facts_by_arg1` module attribute when indexing is on;
%  otherwise empty string.
inline_data_index_block(no_index, "").
inline_data_index_block(indexed(IndexLit), Block) :-
    format(string(Block), '  @facts_by_arg1 ~w', [IndexLit]).

%% inline_data_run1_body(+IndexResult, +Arity, -Body)
%  `run(%WamState{} = state)` body. Indexed: deref arg1 and pick either
%  the matching bucket or the full list. Non-indexed: stream the full
%  list unconditionally.
inline_data_run1_body(no_index, Arity, Body) :-
    format(string(Body),
           '    WamRuntime.stream_facts(state, @facts, ~w)\n', [Arity]).
inline_data_run1_body(indexed(_), Arity, Body) :-
    format(string(Body),
'    arg1 = WamRuntime.deref_var(state, Map.get(state.regs, 1))
    facts = case arg1 do
      {:unbound, _} -> @facts
      key -> Map.get(@facts_by_arg1, key, [])
    end
    WamRuntime.stream_facts(state, facts, ~w)
', [Arity]).

%% render_external_source_module(+CamelMod, +CamelPred, +PredStr, +Arity,
%%                               +ShapeComment, +Layout, -Code)
%  Phase-D external_source layout: no inline facts. `run/1` looks the
%  source up in the FactSourceRegistry (drivers must register before
%  calling), then dispatches to stream_all/lookup_by_arg1 through the
%  FactSource behaviour facade. The SourceSpec the user passed in
%  fact_layout/2 remains opaque to the runtime adaptor path, but the
%  generated module now also exposes any shared preprocess declaration
%  metadata that was attached to the layout so manifest- / provider-like
%  tooling can inspect the intended access contract.
render_external_source_module(CamelMod, CamelPred, PredStr, Arity, ShapeComment,
                              Layout, Code) :-
    format(string(PredIndicator), '~w/~w', [PredStr, Arity]),
    external_source_layout_parts(Layout, SourceSpec, Metadata),
    external_source_metadata_block(SourceSpec, Metadata, MetadataBlock),
    format(string(Code),
'defmodule ~w.~w do
  @moduledoc "Lowered WAM-compiled predicate: ~w/~w (external_source)"

~w

~w

  # Phase-D external_source layout: facts live outside the compiled
  # module. A driver registers a concrete WamRuntime.FactSource (e.g.
  # Tsv) for this predicate before calling run/1; the registry lookup
  # returns the handle which dispatches via the FactSource behaviour.
  @pred_indicator "~w"

  def run(%WamRuntime.WamState{} = state) do
    source = WamRuntime.FactSourceRegistry.lookup!(@pred_indicator)
    arg1 = WamRuntime.deref_var(state, Map.get(state.regs, 1))
    facts = case arg1 do
      {:unbound, _} -> WamRuntime.FactSource.stream_all(source, state)
      key -> WamRuntime.FactSource.lookup_by_arg1(source, key, state)
    end
    WamRuntime.stream_facts(state, facts, ~w)
  end

  def run(args) when is_list(args) do
    state = %WamRuntime.WamState{code: {}, labels: %{}, pc: 1}
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
end', [CamelMod, CamelPred, PredStr, Arity, ShapeComment, MetadataBlock,
       PredIndicator, Arity, CamelPred]).

external_source_layout_parts(external_source(SourceSpec), SourceSpec, none).
external_source_layout_parts(external_source(SourceSpec, Metadata), SourceSpec, Metadata).

external_source_metadata_block(SourceSpec, Metadata, Block) :-
    elixir_term_string_literal(SourceSpec, SourceSpecLit),
    external_source_preprocess_map(Metadata, PreprocessMap),
    format(string(Block),
'  @external_source_spec ~w
  @external_source_metadata %{source_spec: @external_source_spec, preprocess: ~w}
  def external_source_metadata, do: @external_source_metadata',
           [SourceSpecLit, PreprocessMap]).

external_source_preprocess_map(none, 'nil').
external_source_preprocess_map(preprocess_metadata(Source, Mode, Kind, Format,
                                                   AccessContracts, Options),
                               MapLiteral) :-
    elixir_term_string_literal(Source, SourceLit),
    elixir_term_string_literal(Mode, ModeLit),
    elixir_term_string_literal(Kind, KindLit),
    elixir_term_string_literal(Format, FormatLit),
    elixir_string_list_literal(AccessContracts, AccessLit),
    elixir_string_list_literal(Options, OptionsLit),
    format(string(MapLiteral),
           '%{source: ~w, mode: ~w, kind: ~w, format: ~w, access_contracts: ~w, options: ~w}',
           [SourceLit, ModeLit, KindLit, FormatLit, AccessLit, OptionsLit]).

elixir_string_list_literal(Terms, Literal) :-
    maplist(elixir_term_string_literal, Terms, Literals),
    sort(Literals, SortedLiterals),
    atomic_list_concat(SortedLiterals, ', ', Body),
    format(atom(Literal), '[~w]', [Body]).

elixir_term_string_literal(Term, Literal) :-
    term_string(Term, TermString),
    escape_elixir_string(TermString, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).

escape_elixir_string(In, Out) :-
    split_string(In, "\\", "", Parts1),
    atomic_list_concat(Parts1, "\\\\", Tmp1),
    split_string(Tmp1, "\"", "", Parts2),
    atomic_list_concat(Parts2, "\\\"", Out).

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
%  always wins. Otherwise dispatches to the Phase-E pluggable policy
%  `layout_policy/5` — the default `auto` policy implements the
%  pre-Phase-E rule (fact-only ∧ count > threshold → `inline_data`),
%  but callers can select a different built-in or register their own
%  via the multifile `user:wam_elixir_layout_policy/5` hook.
pick_layout(PredIndicator, _NClauses, _FactOnly, Options, Layout) :-
    option(fact_layout(PredIndicator, UserLayout0), Options), !,
    augment_layout_metadata(PredIndicator, UserLayout0, Layout).
pick_layout(PredIndicator, _NClauses, _FactOnly, _Options, Layout) :-
    catch(user:fact_layout(PredIndicator, UserLayout0), _, fail), !,
    augment_layout_metadata(PredIndicator, UserLayout0, Layout).
pick_layout(PredIndicator, NClauses, FactOnly, Options, Layout) :-
    option(fact_layout_policy(PolicyName), Options, auto),
    layout_policy(PolicyName, PredIndicator, NClauses, FactOnly, Options, Layout0),
    augment_layout_metadata(PredIndicator, Layout0, Layout).

augment_layout_metadata(_PredIndicator, external_source(SourceSpec, Metadata),
                        external_source(SourceSpec, Metadata)) :-
    !.
augment_layout_metadata(PredIndicator, external_source(SourceSpec),
                        external_source(SourceSpec, MetadataTerm)) :-
    declared_preprocess_metadata(PredIndicator, Mode,
                                 preprocess_info(Kind, Options),
                                 Metadata),
    !,
    MetadataTerm = preprocess_metadata(shared_preprocess, Mode, Kind,
                                       Metadata.format,
                                       Metadata.access_contracts,
                                       Options).
augment_layout_metadata(_PredIndicator, Layout, Layout).

%% layout_policy(+Policy, +PredIndicator, +NClauses, +FactOnly, +Options, -Layout)
%  Phase-E pluggable layout selector. Three built-in policies:
%
%    auto            — Fact-only ∧ count > threshold → `inline_data`,
%                      else `compiled`. Threshold from
%                      `fact_count_threshold` option (default 100).
%                      Mirrors the pre-Phase-E default.
%    compiled_only   — Always `compiled`. Useful for bisecting
%                      regressions or for hosts where the inline
%                      literal cost is too high.
%    inline_eager    — Fact-only always → `inline_data([])`,
%                      ignoring the count threshold. Rule-bearing
%                      predicates still fall to `compiled`.
%
%  Users can add their own by asserting clauses on the multifile
%  `user:wam_elixir_layout_policy/5`; those take precedence over the
%  built-ins for the same policy name.
:- multifile user:wam_elixir_layout_policy/5.

layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout) :-
    catch(user:wam_elixir_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Layout0),
          _, fail),
    !,
    (   nonvar(Layout0)
    ->  Layout = Layout0
    ;   builtin_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout)
    ).
layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout) :-
    builtin_layout_policy(Policy, PredIndicator, NClauses, FactOnly, Options, Layout).

builtin_layout_policy(auto, _PredIndicator, NClauses, FactOnly, Options, Layout) :-
    option(fact_count_threshold(Threshold), Options, 100),
    (   FactOnly == true, NClauses > Threshold
    ->  Layout = inline_data([])
    ;   Layout = compiled
    ).
builtin_layout_policy(compiled_only, _PredIndicator, _NClauses, _FactOnly, _Options, compiled).
builtin_layout_policy(inline_eager, _PredIndicator, _NClauses, FactOnly, _Options, Layout) :-
    (   FactOnly == true
    ->  Layout = inline_data([])
    ;   Layout = compiled
    ).
builtin_layout_policy(cost_aware, _Pred/Arity, NClauses, FactOnly, Options, Layout) :-
    % Static cost estimate: NClauses × max(1, Arity). Proxies the
    % generated-module size since each clause contributes a tuple
    % of Arity slots. Default threshold 200 keeps the classic
    % arity-2 /count≥100 boundary (100 × 2 = 200) — same shape as
    % the `auto` policy for 2-column fact predicates, but smarter
    % when arity varies (arity-1 stays `compiled` longer; wide
    % arities flip to `inline_data` sooner).
    option(fact_cost_threshold(Threshold), Options, 200),
    Mult is max(1, Arity),
    CostScore is NClauses * Mult,
    (   FactOnly == true, CostScore > Threshold
    ->  Layout = inline_data([])
    ;   Layout = compiled
    ).

%% format_fact_shape_comment(+Info, -Comment)
%  Renders `Info` as an Elixir comment surfaced in the generated
%  module. Phase A uses this purely as documentation; Phase B+ reads
%  Info to pick the emission path.
format_fact_shape_comment(fact_shape_info(N, FactOnly, FirstArg, Layout), Comment) :-
    format(string(Comment),
'  # Fact-shape classification:
  #   clauses=~w fact_only=~w first_arg=~w layout=~w',
           [N, FactOnly, FirstArg, Layout]).

% ============================================================================
% TIER-2 PURITY GATE
% ============================================================================
%
% `par_wrap_segment/3` (PR2) checks a predicate\'s purity certificate
% before emitting the host-native parallel wrapper. Encapsulated here
% so the threshold (0.85) lives alongside the other emitter-facing
% classification helpers — and so a failing certificate fails the
% whole gate predicate (no partial-match leakage of Cert into the
% sequential fallback path). The 0.85 threshold matches Haskell\'s
% Tier-2 implementation (Phase P4).
%
% This is the Tier-2 infrastructure hand-off: the gate is wired but
% nothing calls it yet. PR2 adds `par_wrap_segment/3` in the
% SEGMENTATION section, which depends on this predicate.

%% tier2_purity_eligible(+Pred, +Arity, -Cert)
%  Succeeds iff `Pred/Arity` has a purity certificate with verdict
%  `pure` and confidence ≥ 0.85. Binds `Cert` for downstream logging
%  / debug. Fails silently for any other verdict — Tier-2 emission
%  defers to the Tier-3 sequential fallback when this fails.
tier2_purity_eligible(Pred, Arity, Cert) :-
    analyze_predicate_purity(Pred/Arity, Cert),
    Cert = purity_cert(pure, _Proof, Confidence, _Reasons),
    Confidence >= 0.85.

% ============================================================================
% PHASE B: INLINE_DATA FACT EXTRACTION
% ============================================================================
%
% Walks each clause\'s instruction list to build an Elixir tuple of the
% head args. Constants (from `get_constant`) become quoted strings.
% Variables (from `get_variable`) become `:_var`, meaning the head was
% a variable at that slot and any incoming value unifies trivially.
% Compound args (`get_structure` / `get_list`) are not representable
% inline yet — extraction fails and the caller falls back to `compiled`.

%% extract_facts(+Segments, +Arity, -ElixirLiteral)
%  ElixirLiteral is a string like "[\n    {\"a\", \"b\"},\n    ...\n  ]"
%  suitable for dropping into a module attribute.
extract_facts(Segments, Arity, ElixirLiteral) :-
    maplist(extract_clause_tuple(Arity), Segments, TupleLiterals),
    (   TupleLiterals = []
    ->  ElixirLiteral = '[]'
    ;   atomic_list_concat(TupleLiterals, ',\n    ', Joined),
        format(string(ElixirLiteral), '[\n    ~w\n  ]', [Joined])
    ).

extract_clause_tuple(Arity, _Name-Instrs, TupleLiteral) :-
    numlist(1, Arity, Slots),
    maplist(extract_arg_value(Instrs), Slots, Values),
    atomic_list_concat(Values, ', ', Inner),
    format(string(TupleLiteral), '{~w}', [Inner]).

extract_arg_value(Instrs, Slot, ElixirVal) :-
    format(string(AStr), 'A~w', [Slot]),
    (   member(_-get_constant(C, AStr), Instrs)
    ->  elixir_string_literal(C, ElixirVal)
    ;   member(_-get_variable(_, AStr), Instrs)
    ->  ElixirVal = ':_var'
    ;   member(_-get_value(_, AStr), Instrs)
    ->  ElixirVal = ':_var'
    ;   % get_structure / get_list / anything else → unrepresentable;
        % fail the whole extraction so lower_predicate_to_elixir/4
        % falls back to the compiled path.
        fail
    ).

%% elixir_string_literal(+RawString, -Literal)
%  Wraps the raw WAM operand in double quotes, escaping backslashes and
%  embedded double-quotes so the result is a valid Elixir string.
elixir_string_literal(C, Literal) :-
    (   atom(C) -> atom_string(C, CStr) ; CStr = C ),
    string_chars(CStr, Chars),
    maplist(escape_elixir_char, Chars, NestedChars),
    append(NestedChars, EscapedChars),
    string_chars(Escaped, EscapedChars),
    format(string(Literal), '"~w"', [Escaped]).

escape_elixir_char('\\', ['\\', '\\']).
escape_elixir_char('"', ['\\', '"']).
escape_elixir_char(C, [C]).

% ============================================================================
% PHASE C: FIRST-ARGUMENT INDEXING
% ============================================================================
%
% When every clause has a ground first argument, an `@facts_by_arg1` map
% accompanies `@facts`. `run/1` derefs regs[1] and picks the matching
% bucket if ground, else falls back to the full list. This turns seeded
% queries from O(N) linear scan into O(1) bucket lookup + O(M) scan of
% just that bucket.

%% extract_arg1_index(+Segments, +Arity, -IndexResult)
%  IndexResult is `indexed(MapLiteral)` when a first-arg index applies;
%  `no_index` if any clause has a variable (or otherwise non-ground)
%  arg1 that we cannot use as a map key.
extract_arg1_index(Segments, Arity, IndexResult) :-
    Arity >= 1,
    maplist(extract_clause_arg1(Arity), Segments, KeyTuplePairs),
    (   all_pairs_have_ground_key(KeyTuplePairs)
    ->  group_and_render_index(KeyTuplePairs, MapLiteral),
        IndexResult = indexed(MapLiteral)
    ;   IndexResult = no_index
    ).

extract_clause_arg1(Arity, _Name-Instrs, Arg1Key-TupleLiteral) :-
    numlist(1, Arity, Slots),
    maplist(extract_arg_value(Instrs), Slots, Values),
    atomic_list_concat(Values, ', ', Inner),
    format(string(TupleLiteral), '{~w}', [Inner]),
    Values = [Arg1Literal | _],
    (   Arg1Literal == ':_var'
    ->  Arg1Key = var
    ;   Arg1Key = Arg1Literal
    ).

all_pairs_have_ground_key([]).
all_pairs_have_ground_key([Key-_ | Rest]) :-
    Key \== var,
    all_pairs_have_ground_key(Rest).

group_and_render_index(Pairs, MapLiteral) :-
    sort(0, @=<, Pairs, SortedPairs),
    group_pairs_by_key(SortedPairs, Grouped),
    maplist(render_index_entry, Grouped, Entries),
    atomic_list_concat(Entries, ',\n    ', Joined),
    format(string(MapLiteral), '%{\n    ~w\n  }', [Joined]).

render_index_entry(Key-Tuples, Entry) :-
    atomic_list_concat(Tuples, ', ', TupleList),
    format(string(Entry), '~w => [~w]', [Key, TupleList]).

% ============================================================================
% QUOTE-AWARE LINE TOKENIZATION
% ============================================================================
%
% The WAM text uses ` `, `,`, `\t` as token separators. Atoms containing
% those characters (e.g. `'Washington,_D.C.'`) are wrapped in single
% quotes by `wam_target:quote_wam_constant/2`. This tokenizer recognises
% the quotes, strips them, and honours `\'` / `\\` escapes so quoted
% tokens reparse to the original atom text.
%
% Unquoted tokens pass through unchanged — keeping hot paths (identifier
% atoms like `A1` or labels like `L_parent_2_2`) as cheap as the previous
% split_string-based code.

%% tokenize_wam_line(+Line, -Tokens)
%  `Line` is any string or atom; `Tokens` is a list of strings, in order,
%  with separators removed and quoted regions unquoted/unescaped.
tokenize_wam_line(Line, Tokens) :-
    (   string(Line) -> LineStr = Line ; atom_string(Line, LineStr) ),
    string_chars(LineStr, Chars),
    tokenize_chars(Chars, Tokens).

tokenize_chars([], []).
tokenize_chars([C | Rest], Tokens) :-
    (   wam_separator_char(C)
    ->  tokenize_chars(Rest, Tokens)
    ;   C == '\''
    ->  read_quoted_chars(Rest, TokenChars, Remainder),
        string_chars(Token, TokenChars),
        Tokens = [Token | RestTokens],
        tokenize_chars(Remainder, RestTokens)
    ;   read_unquoted_chars([C | Rest], TokenChars, Remainder),
        string_chars(Token, TokenChars),
        Tokens = [Token | RestTokens],
        tokenize_chars(Remainder, RestTokens)
    ).

wam_separator_char(' ').
wam_separator_char('\t').
wam_separator_char(',').

read_quoted_chars([], [], []).             % unterminated quote — yield what we have
read_quoted_chars(['\'' | Rest], [], Rest).
read_quoted_chars(['\\', Escaped | Rest], [Real | More], Remainder) :-
    !,
    unescape_wam_char(Escaped, Real),
    read_quoted_chars(Rest, More, Remainder).
read_quoted_chars([C | Rest], [C | More], Remainder) :-
    read_quoted_chars(Rest, More, Remainder).

unescape_wam_char('\'', '\'').
unescape_wam_char('\\', '\\').
unescape_wam_char(C, C).

read_unquoted_chars([], [], []).
read_unquoted_chars([C | Rest], [], [C | Rest]) :- wam_separator_char(C), !.
read_unquoted_chars([C | Rest], [C | More], Remainder) :-
    read_unquoted_chars(Rest, More, Remainder).

% ============================================================================
% LABEL COLLECTION
% ============================================================================

collect_labels([], _, []).
collect_labels([Line|Rest], PC, OutLabels) :-
    tokenize_wam_line(Line, CleanParts),
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
    tokenize_wam_line(Line, CleanParts),
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
    tokenize_wam_line(Line, CleanParts),
    (   CleanParts = [First|_], is_label_part(First)
    ->  Body = [], Next = [Line|Rest], NPC = PC
    ;   CleanParts == [] -> extract_segment_body(Rest, PC, Body, Next, NPC)
    ;   Body = [PC-Instr | RestBody],
        instr_from_parts(CleanParts, Instr),
        PC1 is PC + 1,
        extract_segment_body(Rest, PC1, RestBody, Next, NPC)
    ).

generate_all_segments(Segments, Labels, Suffix, SegCodes) :-
    % Build a list of per-segment code-lists and flatten with append/2
    % at the end. The previous recursive append(ThisSegCodes,
    % RestCodes, _) was O(N²) over segment count — noticeable on
    % predicates with thousands of fact clauses.
    %
    % Suffix is threaded through so Tier-2-eligible predicates can
    % emit `clause_X_impl` names while the surface `clause_X` slot is
    % taken by the super-wrapper. For the default Suffix="" path every
    % emitted byte is byte-for-byte identical to the pre-wiring output.
    maplist(generate_one_segment(Labels, Suffix), Segments, SegCodesNested),
    append(SegCodesNested, SegCodes).

generate_one_segment(Labels, Suffix, Name-Instrs, ThisSegCodes) :-
    segment_func_name(Name, Suffix, FuncName),
    classify_segment_head(Instrs, HeadType, BodyInstrs),
    % CPS split: every non-tail `call P, N` terminates a sub-segment;
    % subsequent instrs go into a fresh continuation function. This lets
    % backtrack re-enter a retry point without losing the outer caller's
    % post-call code (which Elixir would otherwise have on a collapsed
    % tail-call stack).
    split_body_at_calls(BodyInstrs, SubSegs),
    emit_sub_segments(SubSegs, FuncName, HeadType, Labels, Suffix, ThisSegCodes).

%% split_body_at_calls(+Instrs, -SubSegs)
%  Splits a flat instr list into sub-segment lists, cutting after every
%  `call P, N` opcode AND after every `end_aggregate ValReg` opcode.
%  The terminator is the last element of its sub-segment; the next
%  sub-segment starts with the first instr after it. A body with
%  no terminators yields exactly one sub-segment.
%
%  Splitting at end_aggregate is what makes multiple findalls in one
%  clause body compose correctly: the post-end_aggregate code lives
%  in its own sub-segment, and end_aggregates lowering uses
%  WamRuntime.update_topmost_agg_cp/2 to point the agg frame at it
%  before throwing fail. Without this split, the post-end_aggregate
%  code ends up in the same sub-segment as end_aggregates throw fail
%  (dead code), and finalise tail-calls a stale agg_cp.cp.
split_body_at_calls(Instrs, SubSegs) :-
    split_body_at_calls_(Instrs, [], SubSegs).

split_body_at_calls_([], AccRev, [Seg]) :-
    reverse(AccRev, Seg).
split_body_at_calls_([PC-call(P, N) | Rest], AccRev, [Seg | RestSegs]) :-
    !,
    reverse([PC-call(P, N) | AccRev], Seg),
    split_body_at_calls_(Rest, [], RestSegs).
split_body_at_calls_([PC-end_aggregate(V) | Rest], AccRev, [Seg | RestSegs]) :-
    !,
    reverse([PC-end_aggregate(V) | AccRev], Seg),
    split_body_at_calls_(Rest, [], RestSegs).
split_body_at_calls_([Instr | Rest], AccRev, Segs) :-
    split_body_at_calls_(Rest, [Instr | AccRev], Segs).

%% emit_sub_segments(+SubSegs, +BaseFunc, +HeadType, +Labels, +Suffix, -Codes)
%  Emits one `defp` per sub-segment. The first uses wrap_segment (with
%  CP push for try_me_else / retry_me_else). Subsequent sub-segments are
%  plain `defp BaseFunc_kN(state) do ... end` continuations. Every
%  sub-segment except the last ends with a tail call to the next
%  continuation; the last ends with its natural tail (execute/proceed).
%
%  Suffix propagates to wrap_segment (for FallbackFunc resolution) and
%  to lower_instr_list (so switch_on_constant arms pick up the right
%  target names). BaseFunc is already suffixed by the caller, so the
%  `_kN` continuation format produces e.g. `clause_main_impl_k1`.
emit_sub_segments([OnlySeg], BaseFunc, HeadType, Labels, Suffix, [Code]) :-
    lower_instr_list(OnlySeg, Labels, BaseFunc, Suffix, Exprs),
    atomic_list_concat(Exprs, '\n', BodyCode),
    wrap_segment(BaseFunc, HeadType, BodyCode, Suffix, Code).

emit_sub_segments([FirstSeg | MoreSegs], BaseFunc, HeadType, Labels, Suffix, [FirstCode | MoreCodes]) :-
    MoreSegs = [_|_],
    format(string(NextFunc), '~w_k1', [BaseFunc]),
    lower_seg_with_continuation(FirstSeg, NextFunc, Labels, BaseFunc, Suffix, FirstBody),
    wrap_segment(BaseFunc, HeadType, FirstBody, Suffix, FirstCode),
    emit_cont_segments(MoreSegs, BaseFunc, 1, Labels, Suffix, MoreCodes).

emit_cont_segments([FinalSeg], BaseFunc, Idx, Labels, Suffix, [Code]) :-
    format(string(FuncName), '~w_k~w', [BaseFunc, Idx]),
    lower_instr_list(FinalSeg, Labels, FuncName, Suffix, Exprs),
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
emit_cont_segments([Seg | More], BaseFunc, Idx, Labels, Suffix, [Code | RestCodes]) :-
    More = [_|_],
    format(string(FuncName), '~w_k~w', [BaseFunc, Idx]),
    NextIdx is Idx + 1,
    format(string(NextFunc), '~w_k~w', [BaseFunc, NextIdx]),
    lower_seg_with_continuation(Seg, NextFunc, Labels, FuncName, Suffix, Body),
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
    emit_cont_segments(More, BaseFunc, NextIdx, Labels, Suffix, RestCodes).

%% lower_seg_with_continuation(+Instrs, +NextFunc, +Labels, +FuncName, +Suffix, -Body)
%  Lowers a sub-segment that ends with either `call P, N` or
%  `end_aggregate ValReg` — the two segment terminators recognised by
%  split_body_at_calls/2.
%
%  For `call`: emits pre-call instrs, sets state.cp = &NextFunc/1, then
%  tail-calls WamDispatcher.call. The called predicates `proceed`
%  routes control back to NextFunc rather than collapsing the stack.
%
%  For `end_aggregate`: emits pre-end_aggregate instrs, then updates
%  the topmost agg frames cp via update_topmost_agg_cp/2 to point at
%  NextFunc, then aggregate_collect, then throws fail. The throw
%  drives backtrack-to-finalise; finalise tail-calls the now-updated
%  agg_cp.cp = NextFunc, which is the post-end_aggregate sub-segment.
%  This lets multiple findalls in one body compose correctly.
lower_seg_with_continuation(Instrs, NextFunc, Labels, FuncName, Suffix, Body) :-
    append(InitInstrs, [_PC-Last], Instrs),
    (   Last = call(P, _N)
    ->  lower_call_terminator(InitInstrs, P, NextFunc, Labels, FuncName, Suffix, Body)
    ;   Last = end_aggregate(ValReg)
    ->  lower_end_aggregate_terminator(InitInstrs, ValReg, NextFunc, Labels, FuncName, Suffix, Body)
    ;   throw(error(unexpected_seg_terminator(Last), lower_seg_with_continuation/6))
    ).

lower_call_terminator(InitInstrs, P, NextFunc, Labels, FuncName, Suffix, Body) :-
    lower_instr_list(InitInstrs, Labels, FuncName, Suffix, InitExprs),
    format(string(TailCallCode),
'    state = %{state | cp: &~w/1}
    WamDispatcher.call("~w", state)', [NextFunc, P]),
    append(InitExprs, [TailCallCode], AllExprs),
    atomic_list_concat(AllExprs, '\n', Body).

lower_end_aggregate_terminator(InitInstrs, ValReg, NextFunc, Labels, FuncName, Suffix, Body) :-
    reg_id(ValReg, ValRegId),
    lower_instr_list(InitInstrs, Labels, FuncName, Suffix, InitExprs),
    format(string(EndAggCode),
'    state = WamRuntime.update_topmost_agg_cp(state, &~w/1)
    state = WamRuntime.aggregate_collect(state, ~w)
    throw({:fail, state})', [NextFunc, ValRegId]),
    append(InitExprs, [EndAggCode], AllExprs),
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
build_switch_arms(Entries, Suffix, ArmsStr) :-
    maplist([E,K-L]>>split_last_colon(E, K, L), Entries, Pairs),
    keysort(Pairs, SortedPairs),
    group_pairs_by_key(SortedPairs, Groups),
    maplist(build_switch_arm_group(Suffix), Groups, Arms),
    atomic_list_concat(Arms, '\n          ', ArmsStr).

build_switch_arm_group(Suffix, Key-Labels, Arm) :-
    (   Labels = [OnlyLabel],
        OnlyLabel \== "default"
    ->  segment_func_name(OnlyLabel, Suffix, LocalFunc),
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

%% segment_func_name(+Label, +Suffix, -Name)
%  Suffix-aware variant used when the Tier-2 super-wrapper has taken
%  the canonical `clause_main` name and the per-clause bodies need to
%  live under an `_impl`-suffixed name (see
%  docs/proposals/WAM_ELIXIR_TIER2_WIRING.md). For Suffix="" the result
%  is identical to segment_func_name/2 — byte-for-byte regression
%  check is the existing test suite.
segment_func_name(Label, Suffix, Name) :-
    segment_func_name(Label, BaseName),
    (   Suffix == "" -> Name = BaseName
    ;   format(string(Name), "~w~w", [BaseName, Suffix])
    ).

classify_segment_head(Instrs, HeadType, BodyInstrs) :-
    (   select(_PC-try_me_else(L), Instrs, BodyInstrs) -> HeadType = try_me_else(L)
    ;   select(_PC-retry_me_else(L), Instrs, BodyInstrs) -> HeadType = retry_me_else(L)
    ;   select(_PC-trust_me, Instrs, BodyInstrs) -> HeadType = trust_me
    ;   HeadType = none, BodyInstrs = Instrs
    ), !.

wrap_segment(FuncName, try_me_else(L), BodyCode, Suffix, Code) :-
    segment_func_name(L, Suffix, FallbackFunc),
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

wrap_segment(FuncName, retry_me_else(L), BodyCode, Suffix, Code) :-
    segment_func_name(L, Suffix, FallbackFunc),
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

% Suffix is unused here because trust_me / none clauses have no
% try_me_else fallback label to resolve through segment_func_name/3
% — only the try_me_else / retry_me_else arms above need it. Suffix
% still flows into BodyCode via lower_instr_list/5, so any
% switch_on_constant inside the body picks the right `_impl` target.
wrap_segment(FuncName, trust_me, BodyCode, _Suffix, Code) :-
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

wrap_segment(FuncName, none, BodyCode, _Suffix, Code) :-
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

lower_instr_list([], _, _, _, []).
lower_instr_list([PC-Instr|Rest], Labels, FuncName, Suffix, [Expr|Exprs]) :-
    wam_elixir_lower_instr(Instr, PC, Labels, FuncName, Suffix, Expr),
    lower_instr_list(Rest, Labels, FuncName, Suffix, Exprs).

% ============================================================================
% TIER-2 SUPER-WRAPPER EMITTER
% ============================================================================
%
% `par_wrap_segment/4` is the emitter-side analogue of Haskell\'s
% `parWrapSegment` (purity_certificate Phase P4). It replaces the
% sequential clause-chain entry point with a `cond`-based super-wrapper
% that:
%
%   - checks `in_forkable_aggregate_frame?/1` — outside a forkable
%     aggregate (findall / aggregate_all), falls back to sequential;
%   - checks `parallel_depth > 0` — nested forks suppressed, fall back;
%   - otherwise pins the cut barrier, increments parallel_depth, fans
%     out via `Task.async_stream` with a per-branch try/catch so CPS
%     throws convert to return values (see
%     `examples/debug_tier2_async_stream_throw_catch.exs` — naked throws
%     crash the parent process on any BEAM node, not just termux), and
%     hands full-exhaustion branch results to `merge_into_aggregate/2`.
%
% Three static gates at Prolog level:
%   1. `intra_query_parallel(false)` option absent — runtime kill-switch.
%   2. `tier2_purity_eligible/3` succeeds (purity ≥ 0.85, verdict pure).
%   3. Segment count ≥ 3 — matches Haskell\'s `forkMinBranches`.
%
% Wired into `render_compiled_module/8` via the `Suffix` parameter
% threaded through the segment-emission pipeline. On gate-pass the
% super-wrapper takes the canonical `clause_main` slot and per-clause
% bodies are emitted as `clause_X_impl`; on gate-reject Suffix=""
% and output is byte-for-byte identical to pre-wiring.

%% par_wrap_segment(+Pred/Arity, +Segments, +Options, -Code)
%  On gate-pass: Code is the emitted Elixir super-wrapper. On
%  gate-reject: Code is the empty string `""`.
%
%  The super-wrapper\'s function name matches the FIRST segment\'s
%  no-suffix name — so `run/1`\'s existing tail call into the first
%  segment reaches the super-wrapper, and the per-clause bodies
%  (emitted under `_impl`-suffixed names by the generate_all_segments
%  flow) are reachable via the super-wrapper\'s cond arms.
par_wrap_segment(Pred/Arity, Segments, Options, Code) :-
    \+ option(intra_query_parallel(false), Options),
    tier2_purity_eligible(Pred, Arity, _Cert),
    length(Segments, N), N >= 3,
    !,
    Segments = [FirstName-_|_],
    segment_func_name(FirstName, "", EntryFunc),
    segment_func_name(FirstName, "_impl", EntryImplFunc),
    maplist([Name-_Instrs, ImplFunc]>>segment_func_name(Name, "_impl", ImplFunc),
            Segments, BranchImplFuncs),
    emit_par_tier2_wrapper(EntryFunc, EntryImplFunc, BranchImplFuncs, Code).
par_wrap_segment(_Pred, _Segments, _Options, "").

%% emit_par_tier2_wrapper(+EntryFunc, +EntryImplFunc, +BranchImplFuncs, -Code)
%  Formats the `cond`-based super-wrapper per the template in
%  `docs/design/WAM_TIERED_LOWERING.md`. EntryFunc is the no-suffix
%  surface name (what `run/1` delegates to). EntryImplFunc is that
%  name + "_impl" — the sequential fallback target. BranchImplFuncs
%  is the list of per-clause `_impl` function names that the
%  parallel fan-out dispatches across.
%
%  Both gate-miss `cond` arms (`not in_forkable_aggregate_frame?` and
%  `parallel_depth > 0`) intentionally call the same EntryImplFunc:
%  outside a forkable aggregate or under a nested fork, the only safe
%  option is to fall back to sequential evaluation via the renamed
%  clause chain. Two arms rather than a combined predicate so each
%  fall-through reason stays self-documenting in the emitted Elixir.
emit_par_tier2_wrapper(EntryFunc, EntryImplFunc, BranchImplFuncs, Code) :-
    maplist([F, Ref]>>format(string(Ref), '&~w/1', [F]),
            BranchImplFuncs, BranchRefs),
    atomic_list_concat(BranchRefs, ', ', BranchListInner),
    format(string(Code),
'  defp ~w(state) do
    cond do
      not WamRuntime.in_forkable_aggregate_frame?(state) ->
        ~w(state)

      Map.get(state, :parallel_depth, 0) > 0 ->
        ~w(state)

      true ->
        branch_state = %{state |
          cut_point: state.choice_points,
          parallel_depth: Map.get(state, :parallel_depth, 0) + 1
        }

        branches = [~w]

        branch_results =
          branches
          |> Task.async_stream(fn branch ->
               try do
                 branch.(branch_state)
               catch
                 {:fail, _state} -> []
                 {:return, result} when is_list(result) -> result
                 {:return, result} -> [result]
               end
             end,
             on_timeout: :kill_task,
             ordered: false,
             max_concurrency: System.schedulers_online())
          |> Enum.flat_map(fn
            {:ok, solutions} when is_list(solutions) -> solutions
            _ -> []
          end)

        WamRuntime.merge_into_aggregate(state, branch_results)
    end
  end',
    [EntryFunc, EntryImplFunc, EntryImplFunc, BranchListInner]).

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
instr_from_parts(["begin_aggregate", Type, ValReg, ResReg], begin_aggregate(Type, ValReg, ResReg)).
instr_from_parts(["end_aggregate", ValReg], end_aggregate(ValReg)).
instr_from_parts(Parts, raw(Combined)) :-
    atomic_list_concat(Parts, ' ', Combined).

% ============================================================================
% INSTRUCTION LOWERING
% ============================================================================

wam_elixir_lower_instr(get_constant(C, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
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

wam_elixir_lower_instr(get_variable(XnName, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val = Map.get(state.regs, ~w)
    state = state |> WamRuntime.trail_binding(~w) |> WamRuntime.put_reg(~w, val)', [Ai, Xn, Xn]).

wam_elixir_lower_instr(get_value(XnName, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val_a = WamRuntime.deref_var(state, Map.get(state.regs, ~w))
    val_x = WamRuntime.get_reg(state, ~w)
    state = case WamRuntime.unify(state, val_a, val_x) do
      {:ok, s} -> s
      :fail -> throw({:fail, state})
    end', [Ai, Xn]).

wam_elixir_lower_instr(put_structure(F, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    addr = state.heap_len
    new_heap = Map.put(state.heap, addr, {:str, "~w"})
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
    |> Map.put(:heap, new_heap)
    |> Map.put(:heap_len, addr + 1)', [F, Ai, Ai]).

wam_elixir_lower_instr(get_structure(F, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
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

wam_elixir_lower_instr(unify_variable(XnName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_variable(state, ~w) do
      :fail -> throw({:fail, state})
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_value(XnName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    state = case WamRuntime.step_unify_value(state, ~w) do
      :fail -> throw({:fail, state})
      s -> s
    end', [Xn]).

wam_elixir_lower_instr(unify_constant(C), _PC, _Labels, _FuncName, _Suffix, Code) :-
    format(string(Code),
'    state = case WamRuntime.step_unify_constant(state, "~w") do
      :fail -> throw({:fail, state})
      s -> s
    end', [C]).

wam_elixir_lower_instr(try_me_else(_L), _PC, _Labels, _FuncName, _Suffix, Code) :-
    Code = '    :ok # Handled by wrap_segment'.

wam_elixir_lower_instr(retry_me_else(_L), _PC, _Labels, _FuncName, _Suffix, Code) :-
    Code = '    :ok # Handled by wrap_segment'.

wam_elixir_lower_instr(trust_me, _PC, _Labels, _FuncName, _Suffix, Code) :-
    Code = '    :ok # Handled by wrap_segment'.

wam_elixir_lower_instr(allocate, _PC, _Labels, _FuncName, _Suffix, Code) :-
    Code = '    # Save caller\'s Y-regs so the callee can freely overwrite
    # slots 201-299 without corrupting the outer frame. Also snapshot
    # the current choice_points as the new cut barrier so `!` inside
    # this predicate truncates only CPs pushed during its own body,
    # not the caller\'s. Env popped by deallocate restores both.
    {y_regs_saved, base_regs} = WamRuntime.split_y_regs(state.regs)
    new_env = %{cp: state.cp, y_regs_saved: y_regs_saved, cut_point: state.cut_point}
    state = %{state | stack: [new_env | state.stack], regs: base_regs,
                      cut_point: state.choice_points}'.

wam_elixir_lower_instr(deallocate, _PC, _Labels, _FuncName, _Suffix, Code) :-
    Code = '    state = case state.stack do
      [env | rest] ->
        {_callee_ys, base_regs} = WamRuntime.split_y_regs(state.regs)
        merged = Map.merge(base_regs, Map.get(env, :y_regs_saved, %{}))
        %{state | cp: env.cp, stack: rest, regs: merged,
                  cut_point: Map.get(env, :cut_point, state.cut_point)}
      _ -> state
    end'.

wam_elixir_lower_instr(call(P, _N), PC, _Labels, _FuncName, _Suffix, Code) :-
    NPC is PC + 1,
    format(string(Code),
'    state = case WamDispatcher.call("~w", state) do
      {:ok, s} -> %{s | pc: ~w}
      :fail -> throw({:fail, state})
    end', [P, NPC]).

wam_elixir_lower_instr(execute(P), _PC, _Labels, _FuncName, _Suffix, Code) :-
    format(string(Code),
'    WamDispatcher.call("~w", state)', [P]).

wam_elixir_lower_instr(builtin_call(Op, Ar), _PC, _Labels, _FuncName, _Suffix, Code) :-
    format(string(Code),
'    state = case WamRuntime.execute_builtin(state, "~w", ~w) do
      :fail -> throw({:fail, state})
      s -> s
    end', [Op, Ar]).

wam_elixir_lower_instr(put_constant(C, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(AiName, Ai),
    format(string(Code), '    state = %{state | regs: Map.put(state.regs, ~w, "~w")}', [Ai, C]).

wam_elixir_lower_instr(put_variable(XnName, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    fresh = {:unbound, make_ref()}
    state = state
    |> WamRuntime.trail_binding(~w)
    |> WamRuntime.put_reg(~w, fresh)
    |> WamRuntime.put_reg(~w, fresh)', [Xn, Xn, Ai]).

wam_elixir_lower_instr(put_value(XnName, AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn), reg_id(AiName, Ai),
    format(string(Code),
'    val = WamRuntime.get_reg(state, ~w)
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, val))', [Xn, Ai, Ai]).

wam_elixir_lower_instr(put_list(AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(AiName, Ai),
    format(string(Code),
'    addr = state.heap_len
    new_heap = Map.put(state.heap, addr, {:str, "./2"})
    state = state
    |> WamRuntime.trail_binding(~w)
    |> Map.put(:regs, Map.put(state.regs, ~w, {:ref, addr}))
    |> Map.put(:heap, new_heap)
    |> Map.put(:heap_len, addr + 1)', [Ai, Ai]).

wam_elixir_lower_instr(get_list(AiName), _PC, _Labels, _FuncName, _Suffix, Code) :-
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

wam_elixir_lower_instr(set_variable(XnName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    addr = state.heap_len
    fresh = {:unbound, {:heap_ref, addr}}
    new_heap = Map.put(state.heap, addr, fresh)
    state = state
    |> WamRuntime.put_reg(~w, fresh)
    |> Map.put(:heap, new_heap)
    |> Map.put(:heap_len, addr + 1)', [Xn]).

wam_elixir_lower_instr(set_value(XnName), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(XnName, Xn),
    format(string(Code),
'    val = WamRuntime.get_reg(state, ~w)
    addr = state.heap_len
    state = %{state | heap: Map.put(state.heap, addr, val), heap_len: addr + 1}', [Xn]).

wam_elixir_lower_instr(set_constant(C), _PC, _Labels, _FuncName, _Suffix, Code) :-
    format(string(Code),
'    addr = state.heap_len
    state = %{state | heap: Map.put(state.heap, addr, "~w"), heap_len: addr + 1}', [C]).

wam_elixir_lower_instr(switch_on_constant(Entries), _PC, _Labels, _FuncName, Suffix, Code) :-
    build_switch_arms(Entries, Suffix, ArmsStr),
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

wam_elixir_lower_instr(switch_on_constant_a2(Entries), _PC, _Labels, _FuncName, Suffix, Code) :-
    build_switch_arms(Entries, Suffix, ArmsStr),
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

wam_elixir_lower_instr(proceed, _PC, _Labels, _FuncName, _Suffix, Code) :-
    % CPS: proceed = tail-call the continuation stored in state.cp. The
    % caller\'s post-call code (or the driver\'s terminal_cp) lives in
    % state.cp — this tail call invokes it. BEAM TCO collapses the stack
    % so deep recursion doesn\'t grow it.
    Code = '    state.cp.(state)'.

% begin_aggregate / end_aggregate lowering — Phase 2 of the findall
% implementation per docs/proposals/WAM_ELIXIR_TIER2_FINDALL.md §4.2/§4.3.
% Substrate helpers live in WamRuntime (PR #1627).

wam_elixir_lower_instr(begin_aggregate(AggTypeStr, ValueReg, ResultReg),
                       _PC, _Labels, _FuncName, _Suffix, Code) :-
    agg_type_atom(AggTypeStr, AggType),
    reg_id(ValueReg, ValReg),
    reg_id(ResultReg, ResReg),
    format(string(Code),
'    state = WamRuntime.push_aggregate_frame(state, :~w, ~w, ~w)',
        [AggType, ValReg, ResReg]).

% Control flow note: the throw({:fail, state}) here is caught by the
% enclosing wrap_segment\'s try/catch, which calls WamRuntime.backtrack
% on the thrown state. backtrack/1 sees the aggregate CP at the top of
% choice_points (pushed by the matching begin_aggregate), checks
% Map.get(cp, :agg_type), and routes to finalise_aggregate/4 — which
% binds the aggregated result and tail-calls the saved continuation.
% The control flow is non-obvious from the emitted Elixir alone: throw →
% segment catch → backtrack → finalise. See WAM_ELIXIR_TIER2_FINDALL.md
% §3.3 (LLVM precedent) and §4.4 (backtrack extension).
wam_elixir_lower_instr(end_aggregate(ValueReg), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(ValueReg, ValReg),
    format(string(Code),
'    state = WamRuntime.aggregate_collect(state, ~w)
    throw({:fail, state})', [ValReg]).

%% agg_type_atom(+Str, -Atom)
%  Translates the WAM-text aggregator name to the Elixir runtime atom.
%
%  The WAM compiler emits `:collect` for findall/3 (via the
%  `collect-Template` wrapper in compile_findall/5). The Tier-2 gate
%  `in_forkable_aggregate_frame?/1` (PR #1586) only recognises
%  `:findall` and `:aggregate_all` as forkable, so we translate
%  `collect → findall` at the emission site rather than broadening the
%  substrate. Decision per WAM_ELIXIR_TIER2_FINDALL.md §6.4 and §9 Q5.
%
%  All other aggregator atoms (sum/count/max/min/bag/set/aggregate_all)
%  pass through unchanged — finalise_aggregate/4 (PR #1627) handles the
%  full alphabet.
agg_type_atom("collect", findall) :- !.
agg_type_atom(Str, Atom) :- atom_string(Atom, Str).

wam_elixir_lower_instr(raw(Combined), _PC, _Labels, _FuncName, _Suffix, Code) :-
    format(string(Code), '    # raw: ~w\n    raise "TODO: ~w"', [Combined, Combined]).

%% pred_to_module(+PredStr, -ModuleName)
pred_to_module(PredStr, ModName) :-
    (   sub_string(PredStr, Before, _, _, "/")
    ->  sub_string(PredStr, 0, Before, _, Name)
    ;   Name = PredStr
    ),
    camel_case(Name, CamelName),
    format(atom(ModName), 'WamPredLow.~w', [CamelName]).
