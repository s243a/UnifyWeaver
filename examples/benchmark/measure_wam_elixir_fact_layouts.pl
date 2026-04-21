:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% measure_wam_elixir_fact_layouts.pl
%
% Emits a single predicate under each of the fact-shape layouts
% (compiled / inline_data / inline_data with arg1 index /
% external_source) and reports the cheap-to-measure signals:
%
%   - Prolog-side codegen time (walltime ms).
%   - Emitted Elixir source size (bytes).
%
% This is the MVP of the Phase-E "measurement-driven tuning" the plan
% describes. It measures what we can cheaply here and in CI; the
% richer side (host-compile cost, query latency, RAM footprint)
% belongs in a desktop-environment follow-up.
%
% Usage:
%
%   swipl -q -s measure_wam_elixir_fact_layouts.pl -- \
%         <facts-path> <pred>/<arity>
%
% Example:
%
%   swipl -q -s measure_wam_elixir_fact_layouts.pl -- \
%         data/benchmark/dev/facts.pl category_parent/2

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_elixir_lowered_emitter',
              [lower_predicate_to_elixir/4, classify_predicate/4]).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPathAtom, PredIndAtom]
    ->  true
    ;   format(user_error,
               'Usage: swipl -q -s measure_wam_elixir_fact_layouts.pl -- <facts-path> <pred>/<arity>~n', []),
        halt(1)
    ),
    atom_string(FactsPathAtom, FactsPath),
    parse_pred_indicator(PredIndAtom, Pred, Arity),
    load_files(FactsPath, [silent(true)]),
    % Introspection pass: run the classification once to surface
    % clause count and groundness before measuring.
    wam_target:compile_predicate_to_wam(user:Pred/Arity, [], IntroWam),
    atom_string(IntroWam, IntroWamStr),
    split_string(IntroWamStr, "\n", "", IntroLines),
    wam_elixir_lowered_emitter:split_into_segments(IntroLines, 1, IntroSegs),
    classify_predicate(Pred/Arity, IntroSegs, [],
                       fact_shape_info(NClauses, FactOnly, FirstArg, _)),
    format(user_error,
           'classification: pred=~w/~w clauses=~w fact_only=~w first_arg=~w~n',
           [Pred, Arity, NClauses, FactOnly, FirstArg]),
    % Measurement header. Tab-separated so the output is easy to
    % pipe into awk, spreadsheet, or the Python harness.
    format('scale\tpred\tlayout\tcodegen_ms\tmodule_bytes\tresolved_as~n'),
    scale_label(FactsPath, ScaleLabel),
    % Explicit layouts first — the "ground truth" per-layout cost.
    % Then the two auto-policies that report which explicit layout
    % they resolve to for this predicate at this scale. Matches the
    % C# hybrid-WAM matrix harness pattern: explicit modes +
    % `auto` side-by-side so the planner's choice is observable.
    Layouts = [
        compiled,
        inline_data_no_index,
        inline_data_indexed,
        external_source,
        auto,
        cost_aware
    ],
    forall(member(Layout, Layouts),
           measure_one(ScaleLabel, Pred, Arity, Layout)),
    halt(0).

parse_pred_indicator(Indicator, Pred, Arity) :-
    atom_string(Indicator, Str),
    split_string(Str, "/", "", [PredStr, ArityStr]),
    atom_string(Pred, PredStr),
    number_string(Arity, ArityStr).

%% scale_label(+FactsPath, -Label)
%  Extracts a human-readable scale tag from the facts path.
%  `data/benchmark/dev/facts.pl` → "dev"; any other shape → "?".
scale_label(FactsPath, Label) :-
    (   file_directory_name(FactsPath, Dir),
        file_base_name(Dir, BaseAtom),
        atom_string(BaseAtom, Label)
    ->  true
    ;   Label = "?"
    ).

%% measure_one(+Scale, +Pred, +Arity, +Layout)
%  Emits one measurement row to stdout. Row shape:
%    scale  pred  layout  codegen_ms  module_bytes  resolved_as
%  For explicit layouts (`compiled`, `inline_data_*`, `external_source`)
%  the `resolved_as` field just echoes the layout label. For the
%  auto-policies (`auto`, `cost_aware`) it reports which concrete
%  layout the classifier actually chose — matches the C# harness's
%  `auto_planner` column.
measure_one(Scale, Pred, Arity, Layout) :-
    options_for_layout(Layout, Opts),
    statistics(walltime, [T0|_]),
    catch(wam_target:compile_predicate_to_wam(user:Pred/Arity, [], WamCode),
          _, (WamCode = "", fail)),
    lower_predicate_to_elixir(Pred/Arity, WamCode,
                              [module_name('MeasureModule') | Opts],
                              Code),
    statistics(walltime, [T1|_]),
    Elapsed is T1 - T0,
    atom_string(Code, CodeStr),
    string_length(CodeStr, Bytes),
    layout_label(Layout, LayoutLabel),
    resolved_layout(Layout, WamCode, Pred, Arity, Opts, ResolvedLabel),
    format('~w\t~w/~w\t~w\t~w\t~w\t~w~n',
           [Scale, Pred, Arity, LayoutLabel, Elapsed, Bytes, ResolvedLabel]).

%% resolved_layout(+Layout, +WamCode, +Pred, +Arity, +Opts, -ResolvedLabel)
%  Echoes the concrete layout an auto-policy picks; for explicit
%  layouts returns the same label.
resolved_layout(auto, WamCode, Pred, Arity, Opts, ResolvedLabel) :- !,
    classify_into_label(WamCode, Pred, Arity, Opts, ResolvedLabel).
resolved_layout(cost_aware, WamCode, Pred, Arity, Opts, ResolvedLabel) :- !,
    classify_into_label(WamCode, Pred, Arity, Opts, ResolvedLabel).
resolved_layout(Layout, _WamCode, _Pred, _Arity, _Opts, Label) :-
    layout_label(Layout, Label).

classify_into_label(WamCode, Pred, Arity, Opts, Label) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    classify_predicate(Pred/Arity, Segments, Opts,
                       fact_shape_info(_, _, _, Layout)),
    concrete_layout_label(Layout, Label).

concrete_layout_label(compiled,           "compiled").
concrete_layout_label(inline_data(_),     "inline_data").
concrete_layout_label(external_source(_), "external_source").

%% options_for_layout(+Layout, -Options)
%  Maps the measurement layout enum onto the emitter's option
%  surface. `external_source` stays opt-in (the default classifier
%  doesn't pick it), so we force it via `fact_layout/2`.
options_for_layout(compiled, [fact_layout_policy(compiled_only),
                              fact_index_policy(none)]).
options_for_layout(inline_data_no_index,
                   [fact_layout_policy(inline_eager),
                    fact_index_policy(none)]).
options_for_layout(inline_data_indexed,
                   [fact_layout_policy(inline_eager),
                    fact_index_policy(first_arg)]).
options_for_layout(external_source, Options) :-
    % Use a dummy tag; the emitter doesn't consult the spec at
    % codegen time (drivers register the real source at runtime).
    Options = [fact_layout(_/_, external_source(measurement_tag))].
options_for_layout(auto, []).  % default policy — no options needed
options_for_layout(cost_aware, [fact_layout_policy(cost_aware)]).

layout_label(compiled,              "compiled").
layout_label(inline_data_no_index,  "inline_data").
layout_label(inline_data_indexed,   "inline_data_indexed").
layout_label(external_source,       "external_source").
layout_label(auto,                  "auto").
layout_label(cost_aware,            "cost_aware").

:- initialization(main, main).
