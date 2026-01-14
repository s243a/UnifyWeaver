% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_doc_generator.pl - plunit tests for doc_generator module
%
% Run with: swipl -g "run_tests" -t halt test_doc_generator.pl

:- module(test_doc_generator, []).

:- use_module(library(plunit)).
:- use_module('doc_generator').

% ============================================================================
% Tests: Pattern Description
% ============================================================================

:- begin_tests(pattern_description).

test(navigation_tab_desc) :-
    once(pattern_description(navigation(tab), Desc)),
    sub_string(Desc, _, _, _, "Tab").

test(navigation_stack_desc) :-
    once(pattern_description(navigation(stack), Desc)),
    sub_string(Desc, _, _, _, "Stack").

test(state_local_desc) :-
    once(pattern_description(state(local), Desc)),
    sub_string(Desc, _, _, _, "local").

test(form_pattern_desc) :-
    once(pattern_description(form_pattern, Desc)),
    sub_string(Desc, _, _, _, "Form").

test(default_desc) :-
    once(pattern_description(unknown_pattern, Desc)),
    sub_string(Desc, _, _, _, "Pattern").

:- end_tests(pattern_description).

% ============================================================================
% Tests: Pattern Options
% ============================================================================

:- begin_tests(pattern_options).

test(nav_tab_options) :-
    once(pattern_options(navigation(tab), Opts)),
    member(option(icon, string, _), Opts),
    member(option(title, string, _), Opts).

test(nav_stack_options) :-
    once(pattern_options(navigation(stack), Opts)),
    member(option(headerShown, boolean, _), Opts).

test(form_field_options) :-
    once(pattern_options(form_field, Opts)),
    member(option(placeholder, string, _), Opts),
    member(option(required, boolean, _), Opts).

test(a11y_options) :-
    once(pattern_options(a11y(_), Opts)),
    member(option(label, string, _), Opts),
    member(option(role, atom, _), Opts).

test(empty_options_for_unknown) :-
    once(pattern_options(unknown_pattern, Opts)),
    Opts = [].

:- end_tests(pattern_options).

% ============================================================================
% Tests: Pattern Example
% ============================================================================

:- begin_tests(pattern_example).

test(nav_tab_example) :-
    once(pattern_example(navigation(tab), Ex)),
    sub_string(Ex, _, _, _, "navigation(tab").

test(state_global_example) :-
    once(pattern_example(state(global), Ex)),
    sub_string(Ex, _, _, _, "state(global").

test(form_pattern_example) :-
    once(pattern_example(form_pattern, Ex)),
    sub_string(Ex, _, _, _, "form_pattern").

test(a11y_example) :-
    once(pattern_example(a11y(_), Ex)),
    sub_string(Ex, _, _, _, "a11y").

:- end_tests(pattern_example).

% ============================================================================
% Tests: Target Support
% ============================================================================

:- begin_tests(target_support).

test(supported_targets_list) :-
    supported_targets(Targets),
    member(react_native, Targets),
    member(vue, Targets),
    member(flutter, Targets),
    member(swiftui, Targets).

test(nav_tab_react_native) :-
    once(target_support_status(navigation(tab), react_native, Status)),
    Status = supported("React Navigation").

test(nav_tab_vue) :-
    once(target_support_status(navigation(tab), vue, Status)),
    Status = supported("Vue Router").

test(nav_drawer_swiftui_partial) :-
    once(target_support_status(navigation(drawer), swiftui, Status)),
    Status = partial(_).

test(pattern_target_support_length) :-
    once(pattern_target_support(navigation(tab), Support)),
    length(Support, 4).

test(all_targets_have_status) :-
    once(pattern_target_support(form_pattern, Support)),
    forall(
        member(target_info(_, Status), Support),
        (Status = supported(_) ; Status = partial(_))
    ).

:- end_tests(target_support).

% ============================================================================
% Tests: Document Pattern
% ============================================================================

:- begin_tests(document_pattern).

test(doc_structure) :-
    once(document_pattern(navigation(tab), [], Doc)),
    Doc = doc(name(_), description(_), example(_), options(_), targets(_), opts([])).

test(doc_name) :-
    once(document_pattern(navigation(tab), [], doc(name(Name), _, _, _, _, _))),
    Name = 'navigation(tab)'.

test(doc_with_options) :-
    once(document_pattern(state(local), [include_examples(true)], Doc)),
    Doc = doc(_, _, _, _, _, opts([include_examples(true)])).

test(doc_targets_populated) :-
    once(document_pattern(navigation(tab), [], doc(_, _, _, _, targets(T), _))),
    length(T, 4).

:- end_tests(document_pattern).

% ============================================================================
% Tests: Markdown Formatting
% ============================================================================

:- begin_tests(markdown_formatting).

test(md_has_title) :-
    once(document_pattern(navigation(tab), [], Doc)),
    once(format_markdown(Doc, MD)),
    sub_string(MD, _, _, _, "# navigation(tab)").

test(md_has_usage_section) :-
    once(document_pattern(navigation(tab), [], Doc)),
    once(format_markdown(Doc, MD)),
    sub_string(MD, _, _, _, "## Usage").

test(md_has_options_table) :-
    once(document_pattern(navigation(tab), [], Doc)),
    once(format_markdown(Doc, MD)),
    sub_string(MD, _, _, _, "## Options").

test(md_has_target_support) :-
    once(document_pattern(navigation(tab), [], Doc)),
    once(format_markdown(Doc, MD)),
    sub_string(MD, _, _, _, "## Target Support").

test(md_has_code_block) :-
    once(document_pattern(navigation(tab), [], Doc)),
    once(format_markdown(Doc, MD)),
    sub_string(MD, _, _, _, "```prolog").

:- end_tests(markdown_formatting).

% ============================================================================
% Tests: Format Section
% ============================================================================

:- begin_tests(format_section).

test(section_has_header) :-
    once(format_section('Test Section', 'Content here', S)),
    sub_string(S, _, _, _, "## Test Section").

test(section_has_content) :-
    once(format_section('Header', 'Body content', S)),
    sub_string(S, _, _, _, "Body content").

:- end_tests(format_section).

% ============================================================================
% Tests: Format Status
% ============================================================================

:- begin_tests(format_status).

test(supported_status) :-
    format_status(supported("Test"), Status, Notes),
    Status = "✓",
    Notes = "Test".

test(partial_status) :-
    format_status(partial("Partial impl"), Status, Notes),
    Status = "◐",
    Notes = "Partial impl".

test(unsupported_status) :-
    format_status(unsupported, Status, Notes),
    Status = "✗",
    Notes = "Not supported".

:- end_tests(format_status).

% ============================================================================
% Tests: Index Generation
% ============================================================================

:- begin_tests(index_generation).

test(index_entries) :-
    once(generate_index([navigation(tab), state(local)], Index)),
    length(Index, 2).

test(index_entry_structure) :-
    once(generate_index([navigation(tab)], Index)),
    Index = [entry(_, _)].

test(empty_index) :-
    once(generate_index([], Index)),
    Index = [].

:- end_tests(index_generation).

% ============================================================================
% Tests: TOC Generation
% ============================================================================

:- begin_tests(toc_generation).

test(toc_has_header) :-
    once(document_pattern(navigation(tab), [], Doc)),
    once(generate_toc([Doc], TOC)),
    sub_string(TOC, _, _, _, "Table of Contents").

test(toc_has_link) :-
    once(document_pattern(navigation(tab), [], Doc)),
    once(generate_toc([Doc], TOC)),
    sub_string(TOC, _, _, _, "navigation(tab)").

:- end_tests(toc_generation).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
