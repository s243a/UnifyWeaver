% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% doc_generator.pl - Auto-generate API docs from pattern definitions
%
% Generates markdown documentation for patterns, including usage examples,
% option tables, and target support information.
%
% Usage:
%   use_module('src/unifyweaver/docs/doc_generator').
%   document_pattern(navigation(tab), [], Doc),
%   format_markdown(Doc, MD).

:- module(doc_generator, [
    % Documentation generation
    generate_pattern_docs/3,
    generate_module_docs/3,
    generate_api_reference/2,

    % Pattern documentation
    document_pattern/3,
    pattern_description/2,
    pattern_options/2,
    pattern_example/2,

    % Target support
    supported_targets/1,
    pattern_target_support/2,
    target_support_status/3,

    % Formatting
    format_markdown/2,
    format_section/3,
    format_status/3,

    % Index generation
    generate_index/2,
    generate_toc/2,

    % Testing
    test_doc_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% Pattern Descriptions
% ============================================================================

%! pattern_description(+PatternType, -Description) is det
%  Get human-readable description for a pattern type.

pattern_description(navigation(tab), "Tab-based navigation with bottom tab bar.").
pattern_description(navigation(stack), "Stack-based navigation with push/pop screens.").
pattern_description(navigation(drawer), "Drawer/sidebar navigation pattern.").

pattern_description(state(local), "Component-level local state management.").
pattern_description(state(global), "Global state management with shared store.").
pattern_description(state(async), "Asynchronous state with loading/error handling.").

pattern_description(form_pattern, "Form container with validation and submission.").
pattern_description(form_field, "Form input field with validation rules.").

pattern_description(screen(_), "Screen component for navigation target.").
pattern_description(button(_), "Clickable button component.").
pattern_description(text(_), "Text display component.").
pattern_description(list(_), "List container for items.").
pattern_description(card(_), "Card container component.").

pattern_description(a11y(_), "Accessibility attributes for screen readers.").

pattern_description(_, "Pattern component.").

% ============================================================================
% Pattern Options
% ============================================================================

%! pattern_options(+PatternType, -Options) is det
%  Get available options for a pattern type.

pattern_options(navigation(tab), [
    option(icon, string, "Tab bar icon name"),
    option(title, string, "Tab title text"),
    option(badge, number, "Badge count number"),
    option(testID, string, "Testing identifier")
]).

pattern_options(navigation(stack), [
    option(headerShown, boolean, "Show navigation header"),
    option(animation, string, "Screen transition animation"),
    option(gestureEnabled, boolean, "Enable swipe gestures")
]).

pattern_options(navigation(drawer), [
    option(position, string, "Drawer position (left/right)"),
    option(drawerWidth, number, "Drawer width in pixels"),
    option(gestureEnabled, boolean, "Enable swipe gestures")
]).

pattern_options(state(local), [
    option(initial, any, "Initial state value"),
    option(persist, boolean, "Persist state across sessions")
]).

pattern_options(state(global), [
    option(namespace, string, "State namespace/slice"),
    option(initial, any, "Initial state value"),
    option(middleware, list, "State middleware list")
]).

pattern_options(form_pattern, [
    option(onSubmit, function, "Form submit handler"),
    option(validation, string, "Validation mode (onChange/onBlur/onSubmit)"),
    option(resetOnSubmit, boolean, "Reset form after submission")
]).

pattern_options(form_field, [
    option(placeholder, string, "Input placeholder text"),
    option(required, boolean, "Field is required"),
    option(minLength, number, "Minimum input length"),
    option(maxLength, number, "Maximum input length"),
    option(pattern, string, "Regex validation pattern")
]).

pattern_options(screen(_), [
    option(title, string, "Screen title"),
    option(icon, string, "Screen icon"),
    option(testID, string, "Testing identifier")
]).

pattern_options(button(_), [
    option(text, string, "Button label text"),
    option(disabled, boolean, "Disable button"),
    option(variant, string, "Button style variant")
]).

pattern_options(a11y(_), [
    option(label, string, "Screen reader label"),
    option(role, atom, "Accessibility role"),
    option(hint, string, "Additional hint text"),
    option(hidden, boolean, "Hide from screen readers")
]).

pattern_options(_, []).

% ============================================================================
% Target Support
% ============================================================================

%! supported_targets(-Targets) is det
%  List of supported code generation targets.
supported_targets([react_native, vue, flutter, swiftui]).

%! target_support_status(+PatternType, +Target, -Status) is det
%  Get support status for a pattern on a target.

% Navigation patterns
target_support_status(navigation(tab), react_native, supported("React Navigation")).
target_support_status(navigation(tab), vue, supported("Vue Router")).
target_support_status(navigation(tab), flutter, supported("GoRouter")).
target_support_status(navigation(tab), swiftui, supported("TabView")).

target_support_status(navigation(stack), react_native, supported("React Navigation")).
target_support_status(navigation(stack), vue, supported("Vue Router")).
target_support_status(navigation(stack), flutter, supported("GoRouter")).
target_support_status(navigation(stack), swiftui, supported("NavigationStack")).

target_support_status(navigation(drawer), react_native, supported("React Navigation")).
target_support_status(navigation(drawer), vue, partial("Manual implementation")).
target_support_status(navigation(drawer), flutter, supported("Drawer widget")).
target_support_status(navigation(drawer), swiftui, partial("NavigationSplitView")).

% State patterns
target_support_status(state(local), react_native, supported("useState")).
target_support_status(state(local), vue, supported("ref/reactive")).
target_support_status(state(local), flutter, supported("StatefulWidget")).
target_support_status(state(local), swiftui, supported("@State")).

target_support_status(state(global), react_native, supported("Redux/Zustand")).
target_support_status(state(global), vue, supported("Pinia")).
target_support_status(state(global), flutter, supported("Provider/Riverpod")).
target_support_status(state(global), swiftui, supported("@EnvironmentObject")).

% Form patterns
target_support_status(form_pattern, react_native, supported("React Hook Form")).
target_support_status(form_pattern, vue, supported("Vee-Validate")).
target_support_status(form_pattern, flutter, supported("Form widget")).
target_support_status(form_pattern, swiftui, supported("Form view")).

% A11y support
target_support_status(a11y(_), react_native, supported("Accessibility API")).
target_support_status(a11y(_), vue, supported("ARIA attributes")).
target_support_status(a11y(_), flutter, supported("Semantics widget")).
target_support_status(a11y(_), swiftui, supported("Accessibility modifiers")).

% Default: supported
target_support_status(_, Target, supported("Native")) :-
    supported_targets(Targets),
    member(Target, Targets).

%! pattern_target_support(+PatternType, -SupportTable) is det
%  Get full target support table for a pattern.
pattern_target_support(PatternType, SupportTable) :-
    supported_targets(Targets),
    findall(
        target_info(Target, Status),
        (   member(Target, Targets),
            once(target_support_status(PatternType, Target, Status))
        ),
        SupportTable
    ).

% ============================================================================
% Pattern Examples
% ============================================================================

%! pattern_example(+PatternType, -Example) is det
%  Get example usage for a pattern type.

pattern_example(navigation(tab), Example) :-
    Example = 'navigation(tab, [
    screen(home, \'HomeScreen\', [icon(\'home\')]),
    screen(profile, \'ProfileScreen\', [icon(\'user\')])
], [])'.

pattern_example(navigation(stack), Example) :-
    Example = 'navigation(stack, [
    screen(main, \'MainScreen\', []),
    screen(detail, \'DetailScreen\', [])
], [])'.

pattern_example(navigation(drawer), Example) :-
    Example = 'navigation(drawer, [
    screen(home, \'HomeScreen\', []),
    screen(settings, \'SettingsScreen\', [])
], [position(left)])'.

pattern_example(state(local), Example) :-
    Example = 'state(local, counter, [initial(0)])'.

pattern_example(state(global), Example) :-
    Example = 'state(global, user, [namespace(auth), initial(null)])'.

pattern_example(form_pattern, Example) :-
    Example = 'form_pattern(login, [
    field(email, email, [required]),
    field(password, password, [required, minLength(8)])
], [onSubmit(handleLogin)])'.

pattern_example(a11y(_), Example) :-
    Example = 'button(submit, [
    text(\'Submit\'),
    a11y([label(\'Submit form\'), role(button), hint(\'Submits the form\')])
])'.

pattern_example(screen(_), Example) :-
    Example = 'screen(home, \'HomeScreen\', [title(\'Home\'), icon(\'home\')])'.

pattern_example(button(_), Example) :-
    Example = 'button(submit, [text(\'Submit\'), variant(primary)])'.

pattern_example(_, Example) :-
    Example = '% No example available'.

% ============================================================================
% Document Generation
% ============================================================================

%! document_pattern(+PatternType, +Options, -Doc) is det
%  Generate documentation structure for a pattern.
document_pattern(PatternType, Options, Doc) :-
    pattern_description(PatternType, Description),
    pattern_options(PatternType, PatternOpts),
    pattern_example(PatternType, Example),
    pattern_target_support(PatternType, TargetSupport),

    % Get pattern name string
    pattern_name_str(PatternType, NameStr),

    Doc = doc(
        name(NameStr),
        description(Description),
        example(Example),
        options(PatternOpts),
        targets(TargetSupport),
        opts(Options)
    ).

pattern_name_str(PatternType, NameStr) :-
    (   PatternType =.. [Name, SubType],
        atom(SubType)
    ->  format(atom(NameStr), '~w(~w)', [Name, SubType])
    ;   PatternType =.. [Name|_]
    ->  atom_string(Name, NameStr)
    ;   atom_string(PatternType, NameStr)
    ).

% ============================================================================
% Markdown Formatting
% ============================================================================

%! format_markdown(+Doc, -Markdown) is det
%  Format documentation structure as markdown.
format_markdown(doc(name(Name), description(Desc), example(Ex), options(Opts), targets(Targets), _), MD) :-
    % Title
    format(atom(TitleSection), '# ~w~n~n~w~n', [Name, Desc]),

    % Usage section
    format(atom(UsageSection), '~n## Usage~n~n```prolog~n~w~n```~n', [Ex]),

    % Options table
    format_options_table(Opts, OptsTable),

    % Target support table
    format_target_table(Targets, TargetTable),

    atomic_list_concat([TitleSection, UsageSection, OptsTable, TargetTable], MD).

%! format_options_table(+Options, -Table) is det
%  Format options as markdown table.
format_options_table([], "") :- !.
format_options_table(Options, Table) :-
    Header = '\n## Options\n\n| Option | Type | Description |\n|--------|------|-------------|\n',
    findall(Row, (
        member(option(Name, Type, Desc), Options),
        format(atom(Row), '| `~w` | ~w | ~w |\n', [Name, Type, Desc])
    ), Rows),
    atomic_list_concat(Rows, RowsStr),
    atom_concat(Header, RowsStr, Table).

%! format_target_table(+Targets, -Table) is det
%  Format target support as markdown table.
format_target_table(Targets, Table) :-
    Header = '\n## Target Support\n\n| Target | Status | Notes |\n|--------|--------|-------|\n',
    findall(Row, (
        member(target_info(Target, Status), Targets),
        format_status(Status, StatusStr, Notes),
        format(atom(Row), '| ~w | ~w | ~w |\n', [Target, StatusStr, Notes])
    ), Rows),
    atomic_list_concat(Rows, RowsStr),
    atom_concat(Header, RowsStr, Table).

format_status(supported(Notes), "✓", Notes).
format_status(partial(Notes), "◐", Notes).
format_status(unsupported, "✗", "Not supported").

%! format_section(+Title, +Content, -Section) is det
%  Format a markdown section.
format_section(Title, Content, Section) :-
    format(atom(Section), '~n## ~w~n~n~w~n', [Title, Content]).

% ============================================================================
% Index and TOC Generation
% ============================================================================

%! generate_index(+Patterns, -Index) is det
%  Generate pattern index.
generate_index(Patterns, Index) :-
    findall(
        entry(Name, Desc),
        (   member(P, Patterns),
            once(pattern_name_str(P, Name)),
            once(pattern_description(P, Desc))
        ),
        Index
    ).

%! generate_toc(+Docs, -TOC) is det
%  Generate table of contents from docs.
generate_toc(Docs, TOC) :-
    findall(Item, (
        member(doc(name(Name), _, _, _, _, _), Docs),
        atom_string(Name, NameStr),
        string_lower(NameStr, Lower),
        format(atom(Item), '- [~w](#~w)~n', [Name, Lower])
    ), Items),
    atomic_list_concat(['## Table of Contents\n\n'|Items], TOC).

% ============================================================================
% File Generation
% ============================================================================

%! generate_pattern_docs(+OutputDir, +Format, +Options) is det
%  Generate documentation files for all patterns.
generate_pattern_docs(OutputDir, Format, Options) :-
    Patterns = [
        navigation(tab),
        navigation(stack),
        navigation(drawer),
        state(local),
        state(global),
        form_pattern,
        a11y(attrs)
    ],
    findall(Doc, (
        member(P, Patterns),
        document_pattern(P, Options, Doc)
    ), Docs),
    (   Format = markdown
    ->  generate_toc(Docs, TOC),
        findall(MD, (
            member(Doc, Docs),
            format_markdown(Doc, MD)
        ), MDs),
        atomic_list_concat([TOC, '\n---\n'|MDs], FullDoc),
        atom_concat(OutputDir, '/patterns.md', FilePath),
        format('Would write to: ~w~n', [FilePath]),
        format('Content length: ~w chars~n', [FullDoc])
    ;   format('Unsupported format: ~w~n', [Format])
    ).

%! generate_module_docs(+Module, +OutputFile, +Options) is det
%  Generate documentation for a specific module.
generate_module_docs(Module, OutputFile, Options) :-
    format('Generating docs for module ~w to ~w~n', [Module, OutputFile]),
    format('Options: ~w~n', [Options]).

%! generate_api_reference(+OutputDir, +Options) is det
%  Generate full API reference.
generate_api_reference(OutputDir, Options) :-
    format('Generating API reference to ~w~n', [OutputDir]),
    format('Options: ~w~n', [Options]).

% ============================================================================
% Testing
% ============================================================================

%! test_doc_generator is det
%  Run inline tests.
test_doc_generator :-
    format('Running doc_generator tests...~n'),

    % Test 1: Pattern description
    once(pattern_description(navigation(tab), Desc1)),
    once(sub_string(Desc1, _, _, _, "Tab")),
    format('  Test 1 passed: pattern description~n'),

    % Test 2: Pattern options
    once(pattern_options(navigation(tab), Opts)),
    once(member(option(icon, string, _), Opts)),
    format('  Test 2 passed: pattern options~n'),

    % Test 3: Pattern example
    once(pattern_example(navigation(tab), Ex)),
    once(sub_string(Ex, _, _, _, "navigation(tab")),
    format('  Test 3 passed: pattern example~n'),

    % Test 4: Target support status
    once(target_support_status(navigation(tab), react_native, supported(_))),
    format('  Test 4 passed: target support status~n'),

    % Test 5: Pattern target support
    once(pattern_target_support(navigation(tab), Support)),
    length(Support, 4),
    format('  Test 5 passed: pattern target support~n'),

    % Test 6: Document pattern
    once(document_pattern(navigation(tab), [], Doc)),
    Doc = doc(name(_), description(_), example(_), options(_), targets(_), opts([])),
    format('  Test 6 passed: document pattern~n'),

    % Test 7: Format markdown
    once(format_markdown(Doc, MD)),
    once(sub_string(MD, _, _, _, "# navigation(tab)")),
    once(sub_string(MD, _, _, _, "## Usage")),
    once(sub_string(MD, _, _, _, "## Options")),
    format('  Test 7 passed: format markdown~n'),

    % Test 8: Generate index
    once(generate_index([navigation(tab), state(local)], Index)),
    length(Index, 2),
    format('  Test 8 passed: generate index~n'),

    % Test 9: Format status
    once(format_status(supported("Test"), S1, N1)),
    S1 = "✓",
    N1 = "Test",
    once(format_status(partial("Test"), S2, _)),
    S2 = "◐",
    format('  Test 9 passed: format status~n'),

    % Test 10: Format section
    once(format_section('Test', 'Content', Section)),
    once(sub_string(Section, _, _, _, "## Test")),
    once(sub_string(Section, _, _, _, "Content")),
    format('  Test 10 passed: format section~n'),

    format('All 10 doc_generator tests passed!~n'),
    !.

:- initialization(test_doc_generator, main).
