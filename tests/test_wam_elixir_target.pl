:- encoding(utf8).
% Test suite for WAM-to-Elixir transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_target.pl

:- use_module('../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_lowered_emitter',
              [lower_predicate_to_elixir/4, classify_predicate/4,
               extract_facts/3, extract_arg1_index/3]).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Core generation tests

test_step_generation :-
    Test = 'WAM-Elixir: step/2 case expression generation',
    (   compile_step_wam_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def step(state, instr)'),
        sub_string(S, _, _, _, 'case instr do'),
        sub_string(S, _, _, _, ':get_constant')
    ->  pass(Test)
    ;   fail_test(Test, 'step generation missing expected content')
    ).

test_helpers_generation :-
    Test = 'WAM-Elixir: helper functions generation',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def run(state)'),
        sub_string(S, _, _, _, 'def backtrack(state)'),
        sub_string(S, _, _, _, 'def unwind_trail(state, mark)')
    ->  pass(Test)
    ;   fail_test(Test, 'helper generation missing expected content')
    ).

test_runtime_assembly :-
    Test = 'WAM-Elixir: full runtime assembly',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'defmodule WamRuntime'),
        sub_string(S, _, _, _, 'defmodule WamState'),
        sub_string(S, _, _, _, 'defstruct'),
        sub_string(S, _, _, _, 'def step(state, instr)'),
        sub_string(S, _, _, _, 'def run(state)')
    ->  pass(Test)
    ;   fail_test(Test, 'runtime assembly missing expected content')
    ).

test_instruction_count :-
    Test = 'WAM-Elixir: instruction arm count',
    (   findall(N, wam_elixir_case(N, _), Cases),
        length(Cases, Count),
        Count >= 26
    ->  pass(Test),
        format('  (~w instruction arms)~n', [Count])
    ;   fail_test(Test, 'fewer than 26 instruction arms')
    ).

%% Instruction category tests

test_head_unification_instructions :-
    Test = 'WAM-Elixir: head unification instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(get_constant, Cases),
        member(get_variable, Cases),
        member(get_value, Cases),
        member(get_structure, Cases),
        member(get_list, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing head unification instruction arms')
    ).

test_body_construction_instructions :-
    Test = 'WAM-Elixir: body construction instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(put_constant, Cases),
        member(put_variable, Cases),
        member(put_value, Cases),
        member(put_structure, Cases),
        member(put_list, Cases),
        member(set_variable, Cases),
        member(set_value, Cases),
        member(set_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing body construction instruction arms')
    ).

test_unification_instructions :-
    Test = 'WAM-Elixir: unification instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(unify_variable, Cases),
        member(unify_value, Cases),
        member(unify_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing unification instruction arms')
    ).

test_control_flow_instructions :-
    Test = 'WAM-Elixir: control flow instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(call, Cases),
        member(execute, Cases),
        member(proceed, Cases),
        member(allocate, Cases),
        member(deallocate, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing control flow instruction arms')
    ).

test_choice_point_instructions :-
    Test = 'WAM-Elixir: choice point instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(try_me_else, Cases),
        member(retry_me_else, Cases),
        member(trust_me, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing choice point instruction arms')
    ).

test_choice_point_bytecode :-
    Test = 'WAM-Elixir: choice point code saves/restores state',
    (   wam_elixir_case(try_me_else, TryCode),
        wam_elixir_case(retry_me_else, RetryCode),
        wam_elixir_case(trust_me, TrustCode),
        sub_string(TryCode, _, _, _, 'choice_points'),
        sub_string(RetryCode, _, _, _, 'choice_points'),
        sub_string(TrustCode, _, _, _, 'choice_points')
    ->  pass(Test)
    ;   fail_test(Test, 'choice point code missing state save/restore')
    ).

test_builtin_call_delegates :-
    Test = 'WAM-Elixir: builtin_call delegates to execute_builtin',
    (   wam_elixir_case(builtin_call, Code),
        sub_string(Code, _, _, _, 'execute_builtin')
    ->  pass(Test)
    ;   fail_test(Test, 'builtin_call does not delegate to execute_builtin')
    ).

%% Code style tests

test_elixir_idioms :-
    Test = 'WAM-Elixir: uses Elixir idioms (pipe, pattern match)',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '|>'),
        sub_string(S, _, _, _, 'Map.put'),
        sub_string(S, _, _, _, 'match?')
    ->  pass(Test)
    ;   fail_test(Test, 'missing Elixir idioms')
    ).

test_immutable_state_updates :-
    Test = 'WAM-Elixir: uses immutable state updates (%{state | ...})',
    (   wam_elixir_case(put_constant, Code),
        sub_string(Code, _, _, _, '%{state |')
    ->  pass(Test)
    ;   fail_test(Test, 'not using immutable struct update syntax')
    ).

test_functional_run_loop :-
    Test = 'WAM-Elixir: run loop is recursive (not imperative)',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'run(new_state)')
    ->  pass(Test)
    ;   fail_test(Test, 'run loop not recursive')
    ).

%% Phase A fact-shape classification tests

:- dynamic
    phase_a_test:small_fact/2,
    phase_a_test:big_fact/2,
    phase_a_test:rule/2,
    phase_a_test:variable_head/1.

phase_a_fixture_setup :-
    % Small fact-only predicate (4 clauses).
    retractall(phase_a_test:small_fact(_, _)),
    forall(member(X-Y, [a-1, b-2, c-3, d-4]),
           assertz(phase_a_test:small_fact(X, Y))),
    % Big fact-only predicate (150 clauses — above default threshold of 100).
    retractall(phase_a_test:big_fact(_, _)),
    forall(between(1, 150, I), (
        atom_number(K, I),
        assertz(phase_a_test:big_fact(K, I))
    )),
    % Rule-bearing predicate (non-fact-only).
    retractall(phase_a_test:rule(_, _)),
    assertz((phase_a_test:rule(X, Y) :- phase_a_test:small_fact(X, Y))),
    % Variable-head fact (for first_arg_groundness=all_variable).
    retractall(phase_a_test:variable_head(_)),
    assertz((phase_a_test:variable_head(_X))).

compile_and_segments(PredArity, Segments) :-
    wam_target:compile_predicate_to_wam(phase_a_test:PredArity, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments).

test_classify_small_fact_only :-
    Test = 'Phase A: small fact-only predicate → compiled (below threshold)',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs, [],
                       fact_shape_info(NCl, FactOnly, G, Layout)),
    (   NCl == 4, FactOnly == true, G == all_ground, Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, classified(NCl, FactOnly, G, Layout))
    ).

test_classify_big_fact_only :-
    Test = 'Phase A: big fact-only predicate → inline_data',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    classify_predicate(big_fact/2, Segs, [],
                       fact_shape_info(NCl, FactOnly, _G, Layout)),
    (   NCl == 150, FactOnly == true, Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, classified(NCl, FactOnly, Layout))
    ).

test_classify_rule :-
    Test = 'Phase A: rule-bearing predicate → compiled (not fact-only)',
    phase_a_fixture_setup,
    compile_and_segments(rule/2, Segs),
    classify_predicate(rule/2, Segs, [],
                       fact_shape_info(_NCl, FactOnly, _G, Layout)),
    (   FactOnly == false, Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, classified(FactOnly, Layout))
    ).

test_classify_variable_head :-
    Test = 'Phase A: all-variable first arg → first_arg=all_variable',
    phase_a_fixture_setup,
    compile_and_segments(variable_head/1, Segs),
    classify_predicate(variable_head/1, Segs, [],
                       fact_shape_info(_, _, G, _)),
    (   G == all_variable
    ->  pass(Test)
    ;   fail_test(Test, groundness(G))
    ).

test_classify_user_override_layout :-
    Test = 'Phase A: user fact_layout/2 override wins over default',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    Opts = [fact_layout(small_fact/2, external_source(tsv("/tmp/x.tsv")))],
    classify_predicate(small_fact/2, Segs, Opts,
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = external_source(_)
    ->  pass(Test)
    ;   fail_test(Test, layout(Layout))
    ).

test_classify_user_override_threshold :-
    Test = 'Phase A: fact_count_threshold(1) promotes small fact set to inline_data',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs, [fact_count_threshold(1)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, layout(Layout))
    ).

test_shape_comment_in_generated_module :-
    Test = 'Phase A: generated module carries fact-shape comment',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode, [module_name('TestMod')], Code),
    (   sub_string(Code, _, _, _, 'Fact-shape classification'),
        sub_string(Code, _, _, _, 'clauses=150'),
        sub_string(Code, _, _, _, 'fact_only=true'),
        sub_string(Code, _, _, _, 'layout=inline_data')
    ->  pass(Test)
    ;   fail_test(Test, 'comment missing or wrong content')
    ).

test_phase_a_preserves_compiled_output :-
    Test = 'Phase A: small predicate with default threshold uses compiled defps',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    lower_predicate_to_elixir(small_fact/2, WamCode, [module_name('TestMod')], Code),
    (   sub_string(Code, _, _, _, 'defp clause_'),
        sub_string(Code, _, _, _, 'def run(%WamRuntime.WamState{}')
    ->  pass(Test)
    ;   fail_test(Test, 'expected compiled-shape output not present')
    ).

%% Phase B inline_data emission tests

test_extract_facts_simple :-
    Test = 'Phase B: extract_facts yields tuple list for fact-only predicate',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    extract_facts(Segments, 2, Literal),
    (   sub_string(Literal, _, _, _, '{"a", "1"}'),
        sub_string(Literal, _, _, _, '{"d", "4"}')
    ->  pass(Test)
    ;   fail_test(Test, Literal)
    ).

test_phase_b_emits_inline_data_when_chosen :-
    Test = 'Phase B: inline_data layout emits @facts and no defp clauses',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode, [module_name('TestMod')], Code),
    (   sub_string(Code, _, _, _, '@facts ['),
        sub_string(Code, _, _, _, 'WamRuntime.stream_facts(state, @facts, 2)'),
        \+ sub_string(Code, _, _, _, 'defp clause_')
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline_data shape missing or leaked defp')
    ).

test_phase_b_variable_head_becomes_sentinel :-
    Test = 'Phase B: variable head arg emits :_var sentinel',
    phase_a_fixture_setup,
    % Force variable_head/1 to inline_data via low threshold.
    wam_target:compile_predicate_to_wam(phase_a_test:variable_head/1, [], WamCode),
    lower_predicate_to_elixir(variable_head/1, WamCode,
                              [module_name('TestMod'), fact_count_threshold(0)], Code),
    (   sub_string(Code, _, _, _, ':_var'),
        sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, 'defp clause_')
    ->  pass(Test)
    ;   fail_test(Test, 'variable_head did not lower to :_var sentinel')
    ).

test_phase_b_fallback_on_unextractable :-
    Test = 'Phase B: unextractable fact falls back to compiled (safe default)',
    phase_a_fixture_setup,
    % Inject a compound-head fact; extraction should fail and fall back.
    retractall(phase_a_test:compound_head(_)),
    assertz((phase_a_test:compound_head(foo(1, 2)))),
    wam_target:compile_predicate_to_wam(phase_a_test:compound_head/1, [], WamCode),
    % Force inline_data via override, but extraction should fail and
    % lower_predicate_to_elixir/4 falls through to compiled.
    lower_predicate_to_elixir(compound_head/1, WamCode,
                              [module_name('TestMod'),
                               fact_layout(compound_head/1, inline_data([]))], Code),
    (   sub_string(Code, _, _, _, 'defp clause_'),
        \+ sub_string(Code, _, _, _, '@facts [')
    ->  pass(Test)
    ;   fail_test(Test, 'expected fallback to compiled did not occur')
    ).

%% WAM constant quoting tests

:- dynamic phase_a_test:comma_atom/2.

test_quote_wam_constant_plain :-
    Test = 'Quoting: identifier-like atom passes through unquoted',
    wam_target:quote_wam_constant('foo_bar_123', Str),
    (   Str == "foo_bar_123"
    ->  pass(Test)
    ;   fail_test(Test, Str)
    ).

test_quote_wam_constant_comma :-
    Test = 'Quoting: atom containing comma is single-quoted',
    wam_target:quote_wam_constant('Has,comma', Str),
    (   Str == "'Has,comma'"
    ->  pass(Test)
    ;   fail_test(Test, Str)
    ).

test_quote_wam_constant_escape :-
    Test = 'Quoting: atom containing single-quote is escaped',
    wam_target:quote_wam_constant('it\'s', Str),
    (   Str == "'it\\'s'"
    ->  pass(Test)
    ;   fail_test(Test, Str)
    ).

test_tokenize_unquoted :-
    Test = 'Tokenizer: plain line splits on spaces and commas',
    wam_elixir_lowered_emitter:tokenize_wam_line("    get_constant foo, A1", Tokens),
    (   Tokens == ["get_constant", "foo", "A1"]
    ->  pass(Test)
    ;   fail_test(Test, Tokens)
    ).

test_tokenize_quoted_atom_with_comma :-
    Test = 'Tokenizer: quoted atom containing comma stays one token',
    wam_elixir_lowered_emitter:tokenize_wam_line("    get_constant 'Has,comma', A1", Tokens),
    (   Tokens == ["get_constant", "Has,comma", "A1"]
    ->  pass(Test)
    ;   fail_test(Test, Tokens)
    ).

test_tokenize_quoted_atom_with_escape :-
    Test = 'Tokenizer: quoted atom with \\\' escape unquotes to literal',
    wam_elixir_lowered_emitter:tokenize_wam_line("    get_constant 'it\\'s', A1", Tokens),
    (   Tokens == ["get_constant", "it's", "A1"]
    ->  pass(Test)
    ;   fail_test(Test, Tokens)
    ).

test_round_trip_comma_atom :-
    Test = 'Round trip: atom with comma survives WAM-text -> tokenizer',
    retractall(phase_a_test:comma_atom(_, _)),
    assertz((phase_a_test:comma_atom('Has,comma', value))),
    wam_target:compile_predicate_to_wam(phase_a_test:comma_atom/2, [], WamCode),
    lower_predicate_to_elixir(comma_atom/2, WamCode,
                              [module_name('TestMod'), fact_count_threshold(0)], Code),
    % Should reach inline_data and emit the atom as an Elixir string.
    (   sub_string(Code, _, _, _, '{"Has,comma", "value"}'),
        \+ sub_string(Code, _, _, _, 'raw:')
    ->  pass(Test)
    ;   fail_test(Test, 'comma-atom round-trip failed')
    ).

%% Phase C first-argument indexing tests

test_extract_arg1_index_ground :-
    Test = 'Phase C: all-ground first arg → indexed map literal',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:small_fact/2, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    extract_arg1_index(Segments, 2, IndexResult),
    (   IndexResult = indexed(Lit),
        sub_string(Lit, _, _, _, '"a" => [{"a", "1"}]'),
        sub_string(Lit, _, _, _, '"d" => [{"d", "4"}]')
    ->  pass(Test)
    ;   fail_test(Test, IndexResult)
    ).

test_extract_arg1_index_variable :-
    Test = 'Phase C: variable first arg → no_index',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:variable_head/1, [], WamCode),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_elixir_lowered_emitter:split_into_segments(Lines, 1, Segments),
    extract_arg1_index(Segments, 1, IndexResult),
    (   IndexResult == no_index
    ->  pass(Test)
    ;   fail_test(Test, IndexResult)
    ).

test_phase_c_indexed_module_emission :-
    Test = 'Phase C: indexed predicate emits @facts_by_arg1 and dispatching run/1',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode, [module_name('TestMod')], Code),
    (   sub_string(Code, _, _, _, '@facts_by_arg1 %{'),
        sub_string(Code, _, _, _, 'case arg1 do'),
        sub_string(Code, _, _, _, 'Map.get(@facts_by_arg1, key, [])')
    ->  pass(Test)
    ;   fail_test(Test, 'indexed shape missing')
    ).

test_phase_c_index_policy_none :-
    Test = 'Phase C: fact_index_policy(none) suppresses @facts_by_arg1',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    lower_predicate_to_elixir(big_fact/2, WamCode,
                              [module_name('TestMod'), fact_index_policy(none)], Code),
    (   sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, '@facts_by_arg1')
    ->  pass(Test)
    ;   fail_test(Test, 'index was not suppressed by policy(none)')
    ).

test_phase_c_variable_head_no_index :-
    Test = 'Phase C: variable head uses flat-only inline_data (no index block)',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:variable_head/1, [], WamCode),
    lower_predicate_to_elixir(variable_head/1, WamCode,
                              [module_name('TestMod'), fact_count_threshold(0)], Code),
    (   sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, '@facts_by_arg1')
    ->  pass(Test)
    ;   fail_test(Test, 'variable-head emitted unexpected index')
    ).

%% Phase D external_source tests

test_phase_d_emits_external_source_shape :-
    Test = 'Phase D: fact_layout external_source emits FactSource-facade run/1',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    Opts = [module_name('TestMod'),
            fact_layout(big_fact/2, external_source(tsv_marker))],
    lower_predicate_to_elixir(big_fact/2, WamCode, Opts, Code),
    (   sub_string(Code, _, _, _, '@pred_indicator "big_fact/2"'),
        sub_string(Code, _, _, _, 'WamRuntime.FactSourceRegistry.lookup!'),
        sub_string(Code, _, _, _, 'WamRuntime.FactSource.stream_all'),
        sub_string(Code, _, _, _, 'WamRuntime.FactSource.lookup_by_arg1'),
        \+ sub_string(Code, _, _, _, '@facts ['),
        \+ sub_string(Code, _, _, _, 'defp clause_')
    ->  pass(Test)
    ;   fail_test(Test, 'external_source shape missing or bled through to other layouts')
    ).

test_phase_d_runtime_emits_fact_source :-
    Test = 'Phase D: runtime assembly emits FactSource behaviour + Tsv adaptor + Registry',
    compile_wam_runtime_to_elixir([], RuntimeCode),
    (   sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource do'),
        sub_string(RuntimeCode, _, _, _, '@callback open'),
        sub_string(RuntimeCode, _, _, _, '@callback stream_all'),
        sub_string(RuntimeCode, _, _, _, '@callback lookup_by_arg1'),
        sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSource.Tsv do'),
        sub_string(RuntimeCode, _, _, _, '@behaviour WamRuntime.FactSource'),
        sub_string(RuntimeCode, _, _, _, 'defmodule WamRuntime.FactSourceRegistry do'),
        sub_string(RuntimeCode, _, _, _, ':persistent_term')
    ->  pass(Test)
    ;   fail_test(Test, 'FactSource runtime pieces missing')
    ).

test_phase_d_external_beats_inline_override :-
    Test = 'Phase D: external_source user override preempts default inline_data',
    phase_a_fixture_setup,
    wam_target:compile_predicate_to_wam(phase_a_test:big_fact/2, [], WamCode),
    % big_fact defaults to inline_data (150 clauses > threshold 100).
    % Explicit external_source must win.
    Opts = [module_name('TestMod'),
            fact_layout(big_fact/2, external_source(tsv_marker))],
    lower_predicate_to_elixir(big_fact/2, WamCode, Opts, Code),
    (   sub_string(Code, _, _, _, '(external_source)'),
        \+ sub_string(Code, _, _, _, '(inline_data)')
    ->  pass(Test)
    ;   fail_test(Test, 'external_source did not win over default inline_data')
    ).

%% Phase E pluggable layout policy tests

test_phase_e_auto_matches_pre_phase_e :-
    Test = 'Phase E: auto policy is equivalent to pre-Phase-E default',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    classify_predicate(big_fact/2, Segs, [fact_layout_policy(auto)],
                       fact_shape_info(_, _, _, Layout1)),
    classify_predicate(big_fact/2, Segs, [],
                       fact_shape_info(_, _, _, Layout2)),
    (   Layout1 == Layout2, Layout1 = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, mismatch(Layout1, Layout2))
    ).

test_phase_e_compiled_only_forces_compiled :-
    Test = 'Phase E: compiled_only policy forces big fact set to compiled',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    classify_predicate(big_fact/2, Segs,
                       [fact_layout_policy(compiled_only)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_phase_e_inline_eager_ignores_threshold :-
    Test = 'Phase E: inline_eager picks inline_data even below threshold',
    phase_a_fixture_setup,
    compile_and_segments(small_fact/2, Segs),
    classify_predicate(small_fact/2, Segs,
                       [fact_layout_policy(inline_eager)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = inline_data(_)
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_phase_e_inline_eager_respects_fact_only :-
    Test = 'Phase E: inline_eager still falls to compiled for rule-bearing',
    phase_a_fixture_setup,
    compile_and_segments(rule/2, Segs),
    classify_predicate(rule/2, Segs,
                       [fact_layout_policy(inline_eager)],
                       fact_shape_info(_, _, _, Layout)),
    (   Layout == compiled
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

test_phase_e_user_override_preempts_policy :-
    Test = 'Phase E: user fact_layout/2 preempts any policy',
    phase_a_fixture_setup,
    compile_and_segments(big_fact/2, Segs),
    Opts = [fact_layout_policy(compiled_only),
            fact_layout(big_fact/2, external_source(tsv_marker))],
    classify_predicate(big_fact/2, Segs, Opts,
                       fact_shape_info(_, _, _, Layout)),
    (   Layout = external_source(_)
    ->  pass(Test)
    ;   fail_test(Test, Layout)
    ).

%% Test runner

run_tests :-
    format('~n=== WAM-Elixir Target Tests ===~n~n'),
    test_step_generation,
    test_helpers_generation,
    test_runtime_assembly,
    test_instruction_count,
    test_head_unification_instructions,
    test_body_construction_instructions,
    test_unification_instructions,
    test_control_flow_instructions,
    test_choice_point_instructions,
    test_choice_point_bytecode,
    test_builtin_call_delegates,
    test_elixir_idioms,
    test_immutable_state_updates,
    test_functional_run_loop,
    test_classify_small_fact_only,
    test_classify_big_fact_only,
    test_classify_rule,
    test_classify_variable_head,
    test_classify_user_override_layout,
    test_classify_user_override_threshold,
    test_shape_comment_in_generated_module,
    test_phase_a_preserves_compiled_output,
    test_extract_facts_simple,
    test_phase_b_emits_inline_data_when_chosen,
    test_phase_b_variable_head_becomes_sentinel,
    test_phase_b_fallback_on_unextractable,
    test_quote_wam_constant_plain,
    test_quote_wam_constant_comma,
    test_quote_wam_constant_escape,
    test_tokenize_unquoted,
    test_tokenize_quoted_atom_with_comma,
    test_tokenize_quoted_atom_with_escape,
    test_round_trip_comma_atom,
    test_extract_arg1_index_ground,
    test_extract_arg1_index_variable,
    test_phase_c_indexed_module_emission,
    test_phase_c_index_policy_none,
    test_phase_c_variable_head_no_index,
    test_phase_d_emits_external_source_shape,
    test_phase_d_runtime_emits_fact_source,
    test_phase_d_external_beats_inline_override,
    test_phase_e_auto_matches_pre_phase_e,
    test_phase_e_compiled_only_forces_compiled,
    test_phase_e_inline_eager_ignores_threshold,
    test_phase_e_inline_eager_respects_fact_only,
    test_phase_e_user_override_preempts_policy,
    format('~n=== WAM-Elixir Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).
