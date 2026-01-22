% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% http_cli_ui.pl - Declarative UI specification for HTTP CLI interface
%
% Defines the HTTP CLI web interface using UI primitives.
% This specification is compiled to HTML/Vue by the html_interface_generator.
%
% Usage:
%   use_module('http_cli_ui').
%   http_cli_interface(Spec),
%   generate_html_interface(Spec, HTML).

:- module(http_cli_ui, [
    % UI Specification
    http_cli_interface/1,        % http_cli_interface(-UISpec)
    http_cli_theme/1,            % http_cli_theme(-Theme)
    http_cli_tabs/1,             % http_cli_tabs(-Tabs)

    % Component specs
    login_form_spec/1,
    user_header_spec/1,
    working_dir_bar_spec/1,
    browse_panel_spec/1,
    grep_panel_spec/1,
    find_panel_spec/1,
    cat_panel_spec/1,
    exec_panel_spec/1,
    feedback_panel_spec/1,
    shell_panel_spec/1,

    % Testing
    test_http_cli_ui/0
]).

:- use_module(library(lists)).
:- catch(use_module('ui_primitives'), _, true).
:- catch(use_module('ui_patterns'), _, true).

% ============================================================================
% THEME DEFINITION
% ============================================================================

%! http_cli_theme(-Theme) is det
%  Define the HTTP CLI interface theme.
http_cli_theme(theme(http_cli, [
    colors([
        primary('#e94560'),
        secondary('#0f3460'),
        background('#1a1a2e'),
        surface('#16213e'),
        text('#eee'),
        muted('#94a3b8'),
        success('#4ade80'),
        warning('#fbbf24'),
        error('#ff6b6b'),
        shell('#a855f7')
    ]),
    typography([
        font_family('-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, monospace'),
        code_family('monospace')
    ]),
    spacing([
        xs(5),
        sm(10),
        md(15),
        lg(20),
        xl(30)
    ]),
    borders([
        radius(5),
        radius_lg(10)
    ])
])).

% ============================================================================
% TAB DEFINITIONS
% ============================================================================

%! http_cli_tabs(-Tabs) is det
%  Define the available tabs in the interface.
http_cli_tabs([
    tab(browse, "Browse", [icon('📁'), roles([user, admin, shell])]),
    tab(grep, "Grep", [icon('🔍'), roles([user, admin, shell])]),
    tab(find, "Find", [icon('📂'), roles([user, admin, shell])]),
    tab(cat, "Cat", [icon('📄'), roles([user, admin, shell])]),
    tab(exec, "Custom", [icon('⚡'), roles([admin, shell])]),
    tab(feedback, "Feedback", [icon('💬'), roles([user, admin, shell])]),
    tab(shell, "Shell", [icon('🔐'), roles([shell]), highlight(true)])
]).

% ============================================================================
% MAIN INTERFACE SPECIFICATION
% ============================================================================

%! http_cli_interface(-UISpec) is det
%  Complete HTTP CLI interface specification.
http_cli_interface(
    page(http_cli, [
        title("UnifyWeaver CLI Search"),
        theme(http_cli)
    ], [
        region(main, [class(container)], [
            % Header
            component(heading, [level(1), content("🔍 UnifyWeaver CLI Search")]),

            % Login form (conditional)
            when(and(auth_required, not(user)), [
                use_spec(login_form_spec)
            ]),

            % Main app (conditional)
            when(or(not(auth_required), user), [
                % User header
                when(user, [use_spec(user_header_spec)]),

                % Working directory bar
                use_spec(working_dir_bar_spec),

                % Tab navigation
                use_spec(tab_navigation_spec),

                % Tab panels
                container(outlet, [id(tab_content)], [
                    panel_switch(active_tab, [
                        case(browse, [use_spec(browse_panel_spec)]),
                        case(grep, [use_spec(grep_panel_spec)]),
                        case(find, [use_spec(find_panel_spec)]),
                        case(cat, [use_spec(cat_panel_spec)]),
                        case(exec, [use_spec(exec_panel_spec)]),
                        case(feedback, [use_spec(feedback_panel_spec)]),
                        case(shell, [use_spec(shell_panel_spec)])
                    ])
                ])
            ])
        ])
    ])
).

% ============================================================================
% COMPONENT SPECIFICATIONS
% ============================================================================

%! login_form_spec(-Spec) is det
login_form_spec(
    container(card, [class(login_container), padding(30), max_width(400)], [
        layout(stack, [spacing(15)], [
            component(heading, [level(2), content("Login Required"), align(center)]),
            component(text_input, [
                label("Email"),
                type(email),
                bind(login_email),
                placeholder("e.g., shell@local"),
                on_keyup_enter(do_login)
            ]),
            component(text_input, [
                label("Password"),
                type(password),
                bind(login_password),
                placeholder("Password"),
                on_keyup_enter(do_login)
            ]),
            component(button, [
                label(loading, "Logging in...", "Login"),
                on_click(do_login),
                disabled(loading),
                full_width(true)
            ]),
            when(login_error, [
                component(text, [content(login_error), style(error), align(center)])
            ]),
            component(text, [
                content("Default users: shell@local/shell, admin@local/admin, user@local/user"),
                style(muted),
                size(12),
                align(center)
            ])
        ])
    ])
).

%! user_header_spec(-Spec) is det
user_header_spec(
    container(panel, [class(user_header), padding(10, 15)], [
        layout(flex, [justify(between), align(center), wrap(true), gap(10)], [
            layout(flex, [class(user_info), align(center), gap(10)], [
                component(text, [content(var('user.email')), class(user_email)]),
                layout(flex, [class(user_roles), gap(5)], [
                    foreach(var('user.roles'), role, [
                        component(badge, [content(var(role)), class([role_badge, var(role)])])
                    ])
                ])
            ]),
            component(button, [
                label("Logout"),
                on_click(do_logout),
                variant(secondary)
            ])
        ])
    ])
).

%! working_dir_bar_spec(-Spec) is det
working_dir_bar_spec(
    container(panel, [class(working_dir_bar), padding(10, 15)], [
        layout(flex, [align(center), gap(10), wrap(true)], [
            component(text, [content("Working directory:"), style(muted)]),
            component(code, [content(working_dir)]),
            when(not_eq(working_dir, "."), [
                component(button, [
                    label("Reset to root"),
                    on_click(reset_working_dir),
                    variant(ghost),
                    size(small)
                ])
            ])
        ])
    ])
).

%! tab_navigation_spec(-Spec) is det
tab_navigation_spec(Spec) :-
    http_cli_tabs(Tabs),
    findall(
        component(tab_button, [
            id(Id),
            label(Label),
            active(eq(tab, Id)),
            on_click(set_tab(Id)),
            icon(Icon),
            visible(has_role(Roles)),
            highlight(Highlight)
        ]),
        (   member(tab(Id, Label, Opts), Tabs),
            (member(icon(Icon), Opts) -> true ; Icon = ''),
            (member(roles(Roles), Opts) -> true ; Roles = []),
            (member(highlight(Highlight), Opts) -> true ; Highlight = false)
        ),
        TabButtons
    ),
    Spec = container(panel, [class(tabs)], [
        layout(flex, [gap(5), wrap(true)], TabButtons)
    ]).

%! browse_panel_spec(-Spec) is det
browse_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            % Navigation bar
            layout(flex, [align(center), gap(10), wrap(true)], [
                when(var('browse.parent'), [
                    component(button, [
                        label("⬆️ Up"),
                        on_click(navigate_up),
                        variant(secondary)
                    ])
                ]),
                component(text, [content("📁 "), style(icon)]),
                component(code, [content(var('browse.path'))]),
                component(button, [
                    label("📌 Set as Working Dir"),
                    on_click(set_working_dir(var('browse.path'))),
                    disabled(eq(var(working_dir), var('browse.path'))),
                    variant(success),
                    size(small)
                ])
            ]),

            % Entry count
            when(var('browse.entries'), [
                component(text, [content(var('browse.entries.length'), " items"), class(count)])
            ]),

            % File list
            container(scroll, [max_height(400)], [
                foreach(var('browse.entries'), entry, [
                    container(panel, [
                        class(file_entry),
                        on_click(handle_entry_click(var(entry))),
                        border_left(var('entry.type'), directory, primary, info)
                    ], [
                        layout(flex, [justify(between), align(center)], [
                            layout(flex, [gap(8)], [
                                component(icon, [name(var('entry.type'), directory, folder, file)]),
                                component(text, [content(var('entry.name'))])
                            ]),
                            component(text, [content(format_size(var('entry.size'))), style(muted), size(12)])
                        ])
                    ])
                ])
            ]),

            % Empty state
            when(and(empty(var('browse.entries')), not(var(loading))), [
                component(text, [content("Empty directory"), style(muted), align(center)])
            ]),

            % Selected file actions
            when(var('browse.selected'), [
                container(panel, [class(selected_file)], [
                    layout(stack, [spacing(10)], [
                        component(text, [content("Selected file:"), style(muted), size(12)]),
                        component(code, [content(var('browse.selected'))]),
                        layout(flex, [gap(10)], [
                            component(button, [label("View Contents"), on_click(view_file)]),
                            component(button, [label("Search Here"), on_click(search_here), variant(secondary)])
                        ])
                    ])
                ])
            ])
        ])
    ])
).

%! grep_panel_spec(-Spec) is det
grep_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            component(text_input, [
                label("Search Pattern (regex)"),
                bind('grep.pattern'),
                placeholder("e.g., function.*export"),
                on_keyup_enter(do_grep)
            ]),
            component(text_input, [
                label("Path (relative to sandbox)"),
                bind('grep.path'),
                placeholder("e.g., src/ or .")
            ]),
            component(text_input, [
                label("Options (space-separated)"),
                bind('grep.options'),
                placeholder("e.g., -i --include=*.ts")
            ]),
            component(button, [
                label(var(loading), "Searching...", "Search"),
                on_click(do_grep),
                disabled(var(loading))
            ])
        ])
    ])
).

%! find_panel_spec(-Spec) is det
find_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            component(text_input, [
                label("File Pattern"),
                bind('find.pattern'),
                placeholder("e.g., *.ts or index.*"),
                on_keyup_enter(do_find)
            ]),
            component(text_input, [
                label("Search Path"),
                bind('find.path'),
                placeholder("e.g., src/ or .")
            ]),
            component(text_input, [
                label("Options (space-separated)"),
                bind('find.options'),
                placeholder("e.g., -type f -maxdepth 3")
            ]),
            component(button, [
                label(var(loading), "Finding...", "Find Files"),
                on_click(do_find),
                disabled(var(loading))
            ])
        ])
    ])
).

%! cat_panel_spec(-Spec) is det
cat_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            component(text_input, [
                label("File Path"),
                bind('cat.path'),
                placeholder("e.g., src/index.ts"),
                on_keyup_enter(do_cat)
            ]),
            component(button, [
                label(var(loading), "Reading...", "Read File"),
                on_click(do_cat),
                disabled(var(loading))
            ])
        ])
    ])
).

%! exec_panel_spec(-Spec) is det
exec_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            component(text_input, [
                label("Command (as you'd type in shell)"),
                bind('exec.commandLine'),
                placeholder("e.g., ls -la src/ or wc -l *.ts"),
                on_keyup_enter(do_exec)
            ]),
            component(button, [
                label(var(loading), "Running...", "Execute"),
                on_click(do_exec),
                disabled(var(loading))
            ]),
            component(text, [
                content("Allowed: cd, pwd, grep, find, cat, head, tail, ls, wc"),
                class(count)
            ])
        ])
    ])
).

%! feedback_panel_spec(-Spec) is det
feedback_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            component(select, [
                label("Feedback Type"),
                bind('feedback.type'),
                options([
                    option(info, "Info"),
                    option(success, "Success"),
                    option(suggestion, "Suggestion"),
                    option(warning, "Warning"),
                    option(error, "Error")
                ])
            ]),
            component(textarea, [
                label("Message"),
                bind('feedback.message'),
                rows(4),
                placeholder("Enter your feedback, notes, or observations..."),
                on_keyup_ctrl_enter(do_feedback)
            ]),
            component(button, [
                label(var(loading), "Submitting...", "Submit Feedback"),
                on_click(do_feedback),
                disabled(var(loading))
            ])
        ])
    ])
).

%! shell_panel_spec(-Spec) is det
shell_panel_spec(
    container(panel, [class(panel)], [
        layout(stack, [spacing(15)], [
            % Shell status
            layout(flex, [justify(between), align(center)], [
                component(text, [content("PTY Shell"), class(shell_title)]),
                layout(flex, [gap(10)], [
                    component(badge, [
                        content(var('shell.connected'), "Connected", "Disconnected"),
                        variant(var('shell.connected'), success, error)
                    ]),
                    component(button, [
                        label("Clear"),
                        on_click(clear_shell),
                        variant(ghost),
                        size(small)
                    ])
                ])
            ]),

            % Terminal output
            container(panel, [class(terminal), id(shell_output)], [
                component(pre, [content(var('shell.output')), class(terminal_text)])
            ]),

            % Input line
            layout(flex, [gap(10)], [
                component(text, [content("$"), class(prompt)]),
                component(text_input, [
                    bind('shell.input'),
                    placeholder("Enter command..."),
                    on_keyup_enter(send_shell_command),
                    id(shell_input),
                    class(shell_input)
                ])
            ])
        ])
    ])
).

% ============================================================================
% TESTING
% ============================================================================

test_http_cli_ui :-
    format('~n=== HTTP CLI UI Tests ===~n~n'),

    % Test 1: Theme
    format('Test 1: Theme definition...~n'),
    http_cli_theme(Theme),
    (   Theme = theme(http_cli, _)
    ->  format('  PASS: Theme defined~n')
    ;   format('  FAIL: Theme not defined~n')
    ),

    % Test 2: Tabs
    format('~nTest 2: Tab definitions...~n'),
    http_cli_tabs(Tabs),
    length(Tabs, TabCount),
    format('  Found ~w tabs~n', [TabCount]),
    (   TabCount >= 6
    ->  format('  PASS: Tabs defined~n')
    ;   format('  FAIL: Expected at least 6 tabs~n')
    ),

    % Test 3: Interface spec
    format('~nTest 3: Interface specification...~n'),
    http_cli_interface(Interface),
    (   Interface = page(http_cli, _, _)
    ->  format('  PASS: Interface spec created~n')
    ;   format('  FAIL: Interface spec invalid~n')
    ),

    % Test 4: Component specs
    format('~nTest 4: Component specifications...~n'),
    login_form_spec(LoginSpec),
    browse_panel_spec(BrowseSpec),
    grep_panel_spec(GrepSpec),
    (   LoginSpec = container(card, _, _),
        BrowseSpec = container(panel, _, _),
        GrepSpec = container(panel, _, _)
    ->  format('  PASS: Component specs valid~n')
    ;   format('  FAIL: Component specs invalid~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('HTTP CLI UI module loaded~n')
), now).
