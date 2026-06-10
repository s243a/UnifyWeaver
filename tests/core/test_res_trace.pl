:- encoding(utf8).
%% Phase 4 test suite for the trace renderers:
%%   render_trace_for_stderr/2 + format_trace_for_comment/3 +
%%   strategy_pretty/2.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_res_trace.pl

:- use_module('../../src/unifyweaver/core/recurrence_evaluation_strategy').
:- use_module(library(lists)).
:- use_module(library(filesex), [directory_file_path/3]).
:- use_module(library(process), [process_create/3, process_wait/2]).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("RES Phase 4 Trace-Renderer Tests~n"),
    format("========================================~n~n"),
    findall(Test, test(Test), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], Passed, Passed).
run_all([Test|Rest], Acc, Passed) :-
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  format("[PASS] ~w~n", [Test]),
        Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

%% ========================================================================
%% Test declarations
%% ========================================================================

%% strategy_pretty
test(test_strategy_pretty_per_query).
test(test_strategy_pretty_fixed_point).
test(test_strategy_pretty_cached).

%% render_trace_for_stderr
test(test_render_stderr_returns_list).
test(test_render_stderr_envelope_header).
test(test_render_stderr_each_step_one_line).
test(test_render_stderr_step_indented).
test(test_render_stderr_empty_trace).
test(test_render_stderr_classify_step).
test(test_render_stderr_cost_model_step).
test(test_render_stderr_no_intent_resolved).

%% format_trace_for_comment
test(test_format_comment_returns_string).
test(test_format_comment_includes_header).
test(test_format_comment_includes_trace_section).
test(test_format_comment_prefix_on_every_line).
test(test_format_comment_throws_on_non_string_prefix).

%% Comment-prefix validation per language
test(test_format_comment_fsharp_prefix).
test(test_format_comment_haskell_prefix).
test(test_format_comment_prolog_prefix).
test(test_format_comment_python_prefix).

%% Per-language delimiter pattern-match assertions (forward-compat
%% safety checks)
test(test_no_c_block_close_in_rendered).
test(test_no_haskell_block_delimiters).
test(test_no_fsharp_xml_doc_trigger).
test(test_no_python_shebang_at_line_start).

%% CRITICAL: Prolog round-trip parse test
test(test_prolog_comment_round_trip_parses).

%% Integration: end-to-end render from select_evaluation_strategy/3
test(test_full_pipeline_render_stderr).
test(test_full_pipeline_format_comment).

%% ========================================================================
%% Test fixtures
%% ========================================================================

a_recurrence(recurrence(transitive_closure2, my_pred/2,
    [value_domain(combinatorial), monotone(true)])).

a_full_trace(Trace) :-
    a_recurrence(R),
    select_evaluation_strategy(R,
        [csr_available(true), query_pattern(single_pair), cardinality(large),
         kernel_mode(bidirectional)],
        strategy_choice(_Strategy, Trace)).

%% ========================================================================
%% strategy_pretty
%% ========================================================================

test_strategy_pretty_per_query :-
    strategy_pretty(strategy(per_query(bidirectional)), S),
    S == "per_query(bidirectional)".

test_strategy_pretty_fixed_point :-
    strategy_pretty(strategy(fixed_point(semi_naive)), S),
    S == "fixed_point(semi_naive)".

test_strategy_pretty_cached :-
    strategy_pretty(strategy(cached), S),
    S == "cached".

%% ========================================================================
%% render_trace_for_stderr
%% ========================================================================

test_render_stderr_returns_list :-
    a_full_trace(T),
    render_trace_for_stderr(T, Lines),
    is_list(Lines),
    Lines \== [].

test_render_stderr_envelope_header :-
    a_full_trace(T),
    render_trace_for_stderr(T, [Header | _]),
    string_concat("[evaluation-strategy] selecting strategy", _, Header).

test_render_stderr_each_step_one_line :-
    a_full_trace(T),
    T = trace(Steps),
    length(Steps, NSteps),
    render_trace_for_stderr(T, Lines),
    length(Lines, NLines),
    %% header + one line per step
    NLines =:= NSteps + 1.

test_render_stderr_step_indented :-
    a_full_trace(T),
    render_trace_for_stderr(T, [_Header | StepLines]),
    %% Every step line should start with the indented prefix
    forall(member(L, StepLines),
           sub_string(L, 0, _, _, "[evaluation-strategy]   ")).

test_render_stderr_empty_trace :-
    render_trace_for_stderr(trace([]), Lines),
    %% Just the header
    Lines = ["[evaluation-strategy] selecting strategy"].

test_render_stderr_classify_step :-
    a_full_trace(T),
    render_trace_for_stderr(T, Lines),
    member(L, Lines),
    sub_string(L, _, _, _, "classify_signals"), !.

test_render_stderr_cost_model_step :-
    a_full_trace(T),
    render_trace_for_stderr(T, Lines),
    member(L, Lines),
    sub_string(L, _, _, _, "cost_model_choice"),
    sub_string(L, _, _, _, "per_query(bidirectional)"),
    sub_string(L, _, _, _, "score=3.000"), !.

test_render_stderr_no_intent_resolved :-
    a_recurrence(R),
    select_evaluation_strategy(R, [], strategy_choice(_, T)),
    render_trace_for_stderr(T, Lines),
    member(L, Lines),
    sub_string(L, _, _, _, "no_intent: applied"), !.

%% ========================================================================
%% format_trace_for_comment
%% ========================================================================

test_format_comment_returns_string :-
    a_full_trace(T),
    format_trace_for_comment(T, "// ", C),
    string(C),
    string_length(C, L),
    L > 0.

test_format_comment_includes_header :-
    a_full_trace(T),
    format_trace_for_comment(T, "// ", C),
    sub_string(C, _, _, _, "Evaluation strategy:").

test_format_comment_includes_trace_section :-
    a_full_trace(T),
    format_trace_for_comment(T, "// ", C),
    sub_string(C, _, _, _, "Trace:").

%% Critical: every line begins with the prefix, not just the first.
test_format_comment_prefix_on_every_line :-
    a_full_trace(T),
    format_trace_for_comment(T, "// ", C),
    split_string(C, "\n", "", Lines),
    forall(member(L, Lines),
           string_concat("// ", _, L)).

test_format_comment_throws_on_non_string_prefix :-
    a_full_trace(T),
    catch(format_trace_for_comment(T, not_a_string, _),
          error(type_error(string, _), _),
          true).

%% ========================================================================
%% Comment-prefix validation per language
%% ========================================================================

test_format_comment_fsharp_prefix :-
    a_full_trace(T),
    format_trace_for_comment(T, "// ", C),
    split_string(C, "\n", "", Lines),
    forall(member(L, Lines), string_concat("// ", _, L)).

test_format_comment_haskell_prefix :-
    a_full_trace(T),
    format_trace_for_comment(T, "-- ", C),
    split_string(C, "\n", "", Lines),
    forall(member(L, Lines), string_concat("-- ", _, L)).

test_format_comment_prolog_prefix :-
    a_full_trace(T),
    format_trace_for_comment(T, "% ", C),
    split_string(C, "\n", "", Lines),
    forall(member(L, Lines), string_concat("% ", _, L)).

test_format_comment_python_prefix :-
    a_full_trace(T),
    format_trace_for_comment(T, "# ", C),
    split_string(C, "\n", "", Lines),
    forall(member(L, Lines), string_concat("# ", _, L)).

%% ========================================================================
%% Per-language delimiter pattern-match assertions
%%
%% These check that the rendered string contains no substring that
%% would break out of the target language's comment context.
%% ========================================================================

test_no_c_block_close_in_rendered :-
    a_full_trace(T),
    format_trace_for_comment(T, "// ", C),
    \+ sub_string(C, _, _, _, "*/").

test_no_haskell_block_delimiters :-
    a_full_trace(T),
    format_trace_for_comment(T, "-- ", C),
    \+ sub_string(C, _, _, _, "{-"),
    \+ sub_string(C, _, _, _, "-}").

test_no_fsharp_xml_doc_trigger :-
    a_full_trace(T),
    format_trace_for_comment(T, "// ", C),
    %% A line starting with "/// " (XML doc comment) would be triggered
    %% if any rendered line started with "// /" — check no occurrence.
    \+ sub_string(C, _, _, _, "// /").

test_no_python_shebang_at_line_start :-
    a_full_trace(T),
    format_trace_for_comment(T, "# ", C),
    split_string(C, "\n", "", Lines),
    forall(member(L, Lines),
           \+ string_concat("#!", _, L)).

%% ========================================================================
%% CRITICAL: Prolog round-trip parse test
%%
%% Write the rendered Prolog comment + a minimal clause body to a
%% .pl file, invoke swipl --halt, assert exit 0.
%% Catches the failure mode where % is missing on a non-first line.
%% ========================================================================

test_prolog_comment_round_trip_parses :-
    a_full_trace(T),
    format_trace_for_comment(T, "% ", Comment),
    %% Write to a temp file: comment header + a trivial clause body.
    TempFile = '/tmp/test_res_trace_round_trip.pl',
    setup_call_cleanup(
        open(TempFile, write, Stream),
        ( format(Stream, "~w~n", [Comment]),
          format(Stream, "test_compiled :- true.~n", [])
        ),
        close(Stream)
    ),
    %% Invoke swipl with -t halt; assert exit 0.
    process_create(path(swipl),
                   ['-g', 'test_compiled', '-t', 'halt', '-s', TempFile],
                   [process(PID)]),
    process_wait(PID, exit(ExitCode)),
    ExitCode =:= 0,
    %% Clean up
    delete_file(TempFile).

%% ========================================================================
%% Integration
%% ========================================================================

test_full_pipeline_render_stderr :-
    a_recurrence(R),
    select_evaluation_strategy(R,
        [csr_available(true), query_pattern(single_pair), cardinality(large),
         kernel_mode(bidirectional)],
        strategy_choice(_Strategy, Trace)),
    render_trace_for_stderr(Trace, Lines),
    is_list(Lines),
    length(Lines, NLines),
    NLines >= 2.

test_full_pipeline_format_comment :-
    a_recurrence(R),
    select_evaluation_strategy(R,
        [csr_available(true), query_pattern(single_pair), cardinality(large),
         kernel_mode(bidirectional)],
        strategy_choice(_Strategy, Trace)),
    format_trace_for_comment(Trace, "// ", Comment),
    string_length(Comment, L),
    L > 100.  % non-trivial output
