% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% ui_patterns.pl - Reusable UI pattern definitions
%
% Provides a pattern system for building complex UIs from composable,
% parameterized templates. Patterns are defined once and instantiated
% with different arguments throughout the application.
%
% Usage:
%   use_module('src/unifyweaver/ui/ui_patterns').
%
%   % Define a custom pattern
%   define_pattern(my_card, [title, content],
%       container(card, [padding(16)], [
%           component(heading, [level(3), content(title)]),
%           component(text, [content(content)])
%       ])).
%
%   % Use the pattern
%   UI = use_pattern(my_card, [title("Hello"), content("World")]),
%   expand_patterns(UI, Expanded).

:- module(ui_patterns, [
    % Pattern definition
    define_pattern/3,            % define_pattern(+Name, +Params, +Template)
    get_pattern/2,               % get_pattern(+Name, -Pattern)
    list_patterns/1,             % list_patterns(-Names)
    clear_patterns/0,            % clear_patterns

    % Pattern expansion
    expand_pattern/3,            % expand_pattern(+Name, +Args, -UISpec)
    expand_patterns/2,           % expand_patterns(+UITree, -Expanded)

    % Built-in patterns
    register_builtin_patterns/0,

    % Testing
    test_ui_patterns/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_pattern/3.     % stored_pattern(Name, Params, Template)

% ============================================================================
% PATTERN DEFINITION
% ============================================================================

%! define_pattern(+Name, +Params, +Template) is det
%  Define a reusable UI pattern.
%
%  @param Name     atom - unique pattern identifier
%  @param Params   list of atoms - parameter names used in template
%  @param Template UI specification with parameter references
%
%  Example:
%    define_pattern(labeled_input, [label, binding],
%        layout(stack, [spacing(4)], [
%            component(label, [text(label)]),
%            component(text_input, [bind(binding)])
%        ])).
%
define_pattern(Name, Params, Template) :-
    atom(Name),
    is_list(Params),
    retractall(stored_pattern(Name, _, _)),
    assertz(stored_pattern(Name, Params, Template)).

%! get_pattern(+Name, -Pattern) is semidet
%  Retrieve a pattern definition.
%
%  @param Name    atom - pattern identifier
%  @param Pattern pattern(Params, Template)
%
get_pattern(Name, pattern(Params, Template)) :-
    stored_pattern(Name, Params, Template).

%! list_patterns(-Names) is det
%  Get list of all defined pattern names.
list_patterns(Names) :-
    findall(Name, stored_pattern(Name, _, _), Names).

%! clear_patterns is det
%  Remove all pattern definitions.
clear_patterns :-
    retractall(stored_pattern(_, _, _)).

% ============================================================================
% PATTERN EXPANSION
% ============================================================================

%! expand_pattern(+Name, +Args, -UISpec) is det
%  Expand a pattern with given arguments to produce a UI specification.
%
%  @param Name   atom - pattern identifier
%  @param Args   list of Key(Value) terms
%  @param UISpec expanded UI specification
%
expand_pattern(Name, Args, UISpec) :-
    stored_pattern(Name, Params, Template),
    build_substitution(Params, Args, Substitution),
    substitute_in_template(Template, Substitution, UISpec).

%! build_substitution(+Params, +Args, -Substitution) is det
%  Build substitution map from params and args.
build_substitution(Params, Args, Substitution) :-
    maplist(param_to_sub(Args), Params, Substitution).

param_to_sub(Args, Param, Param-Value) :-
    Term =.. [Param, Value],
    (   member(Term, Args)
    ->  true
    ;   Value = ''  % Default to empty if not provided
    ).

%! substitute_in_template(+Template, +Substitution, -Result) is det
%  Recursively substitute parameter references in a template.
substitute_in_template(Var, Substitution, Value) :-
    atom(Var),
    member(Var-Value, Substitution), !.

substitute_in_template(Atom, _Substitution, Atom) :-
    atom(Atom), !.

substitute_in_template(Number, _Substitution, Number) :-
    number(Number), !.

substitute_in_template(String, _Substitution, String) :-
    string(String), !.

substitute_in_template([], _Substitution, []) :- !.

substitute_in_template([H|T], Substitution, [H2|T2]) :- !,
    substitute_in_template(H, Substitution, H2),
    substitute_in_template(T, Substitution, T2).

% Special handling for UI primitives - don't substitute the Type argument
substitute_in_template(component(Type, Opts), Substitution, component(Type, NewOpts)) :- !,
    substitute_in_template(Opts, Substitution, NewOpts).

substitute_in_template(container(Type, Opts, Content), Substitution, container(Type, NewOpts, NewContent)) :- !,
    substitute_in_template(Opts, Substitution, NewOpts),
    substitute_in_template(Content, Substitution, NewContent).

substitute_in_template(layout(Type, Opts, Children), Substitution, layout(Type, NewOpts, NewChildren)) :- !,
    substitute_in_template(Opts, Substitution, NewOpts),
    substitute_in_template(Children, Substitution, NewChildren).

substitute_in_template(Term, Substitution, Result) :-
    compound(Term),
    Term =.. [Functor|Args],
    maplist(substitute_in_template_arg(Substitution), Args, NewArgs),
    Result =.. [Functor|NewArgs].

substitute_in_template_arg(Substitution, Arg, NewArg) :-
    substitute_in_template(Arg, Substitution, NewArg).

%! expand_patterns(+UITree, -Expanded) is det
%  Recursively expand all use_pattern/2 nodes in a UI tree.
expand_patterns(use_pattern(Name, Args), Expanded) :- !,
    expand_pattern(Name, Args, Template),
    expand_patterns(Template, Expanded).

expand_patterns(layout(Type, Options, Children), layout(Type, Options, ExpandedChildren)) :- !,
    maplist(expand_patterns, Children, ExpandedChildren).

expand_patterns(container(Type, Options, Content), container(Type, Options, ExpandedContent)) :- !,
    expand_patterns(Content, ExpandedContent).

expand_patterns(when(Cond, Content), when(Cond, ExpandedContent)) :- !,
    expand_patterns(Content, ExpandedContent).

expand_patterns(unless(Cond, Content), unless(Cond, ExpandedContent)) :- !,
    expand_patterns(Content, ExpandedContent).

expand_patterns(foreach(Items, Var, Template), foreach(Items, Var, ExpandedTemplate)) :- !,
    expand_patterns(Template, ExpandedTemplate).

expand_patterns([], []) :- !.

expand_patterns([H|T], [H2|T2]) :- !,
    expand_patterns(H, H2),
    expand_patterns(T, T2).

expand_patterns(Other, Other).  % Atoms, components, etc. pass through

% ============================================================================
% BUILT-IN PATTERNS
% ============================================================================

%! register_builtin_patterns is det
%  Register the standard library of UI patterns.
register_builtin_patterns :-
    % Form field: labeled input with optional error
    define_pattern(form_field, [label, type, binding, placeholder, error],
        layout(stack, [spacing(4)], [
            component(label, [text(label)]),
            component(text_input, [type(type), bind(binding), placeholder(placeholder)]),
            when(error, component(text, [content(error), style(error)]))
        ])),

    % Simple labeled input (no error handling)
    define_pattern(labeled_input, [label, binding],
        layout(stack, [spacing(4)], [
            component(label, [text(label)]),
            component(text_input, [bind(binding)])
        ])),

    % Form with fields and submit button
    define_pattern(form, [fields, submit_label, on_submit],
        layout(stack, [spacing(16)], [
            foreach(fields, field, use_pattern(form_field, field)),
            component(button, [label(submit_label), variant(primary), on_click(on_submit)])
        ])),

    % Button group (horizontal row of buttons)
    define_pattern(button_group, [buttons, spacing],
        layout(flex, [direction(row), gap(spacing), justify(end)], [
            foreach(buttons, btn, component(button, btn))
        ])),

    % Card with header
    define_pattern(header_card, [title, content, actions],
        container(card, [padding(0)], [
            layout(stack, [spacing(0)], [
                container(panel, [padding(16), background(surface)], [
                    layout(flex, [justify(between), align(center)], [
                        component(heading, [level(3), content(title)]),
                        when(actions, use_pattern(button_group, [buttons(actions), spacing(8)]))
                    ])
                ]),
                container(panel, [padding(16)], [
                    content
                ])
            ])
        ])),

    % Login form
    define_pattern(login_form, [on_submit, on_forgot],
        container(card, [padding(24), max_width(400)], [
            layout(stack, [spacing(16)], [
                component(heading, [level(2), content("Login")]),
                use_pattern(form_field, [
                    label("Email"), type(email), binding(email),
                    placeholder("Enter your email"), error('')
                ]),
                use_pattern(form_field, [
                    label("Password"), type(password), binding(password),
                    placeholder("Enter your password"), error('')
                ]),
                layout(flex, [justify(between), align(center)], [
                    component(checkbox, [label("Remember me"), bind(remember)]),
                    component(link, [label("Forgot password?"), on_click(on_forgot)])
                ]),
                component(button, [label("Sign In"), variant(primary), on_click(on_submit)])
            ])
        ])),

    % Data table
    define_pattern(data_table, [columns, data, on_row_click],
        container(panel, [padding(0), rounded(8), border(1)], [
            layout(stack, [spacing(0)], [
                % Header row
                container(panel, [padding(12), background(surface)], [
                    layout(grid, [columns(columns), gap(16)], [
                        foreach(columns, col,
                            component(text, [content(col), style(header)]))
                    ])
                ]),
                % Data rows
                foreach(data, row,
                    container(panel, [padding(12), on_click(on_row_click)], [
                        layout(grid, [columns(columns), gap(16)], [
                            foreach(columns, col,
                                component(text, [content(row)]))
                        ])
                    ]))
            ])
        ])),

    % Navigation menu
    define_pattern(nav_menu, [items, active, on_select],
        layout(stack, [spacing(4)], [
            foreach(items, item,
                component(button, [
                    label(item),
                    variant(ghost),
                    on_click(on_select),
                    active(active)
                ]))
        ])),

    % Tabs with content
    define_pattern(tabbed_interface, [tabs, active_tab, on_tab_change],
        layout(stack, [spacing(0)], [
            container(panel, [padding(0), border_bottom(1)], [
                component(tabs, [items(tabs), active(active_tab), on_change(on_tab_change)])
            ]),
            container(panel, [padding(16)], [
                outlet(tab_content)
            ])
        ])),

    % Modal dialog
    define_pattern(modal_dialog, [title, content, actions],
        container(modal, [max_width(500)], [
            layout(stack, [spacing(16)], [
                component(heading, [level(3), content(title)]),
                content,
                layout(flex, [justify(end), gap(8)], [
                    foreach(actions, action, component(button, action))
                ])
            ])
        ])),

    % Alert/notification
    define_pattern(alert, [variant, message, dismissible],
        container(panel, [padding(12), rounded(8), background(variant)], [
            layout(flex, [justify(between), align(center)], [
                component(text, [content(message)]),
                when(dismissible, component(icon_button, [icon(close), on_click(dismiss)]))
            ])
        ])),

    % Search bar with button
    define_pattern(search_bar, [placeholder, on_search],
        layout(flex, [gap(8)], [
            component(text_input, [
                type(search),
                placeholder(placeholder),
                bind(search_query)
            ]),
            component(button, [label("Search"), on_click(on_search)])
        ])),

    % User avatar with name
    define_pattern(user_badge, [user],
        layout(flex, [gap(8), align(center)], [
            component(avatar, [name(user), size(32)]),
            component(text, [content(user)])
        ])),

    % Page header with title and actions
    define_pattern(page_header, [title, subtitle, actions],
        container(panel, [padding(16), border_bottom(1)], [
            layout(flex, [justify(between), align(center)], [
                layout(stack, [spacing(4)], [
                    component(heading, [level(1), content(title)]),
                    when(subtitle, component(text, [content(subtitle), style(muted)]))
                ]),
                when(actions, use_pattern(button_group, [buttons(actions), spacing(8)]))
            ])
        ])),

    % Empty state placeholder
    define_pattern(empty_state, [icon, title, message, action],
        layout(center, [], [
            layout(stack, [spacing(16), align(center)], [
                component(icon, [name(icon), size(64), color(muted)]),
                component(heading, [level(3), content(title)]),
                component(text, [content(message), style(muted)]),
                when(action, component(button, action))
            ])
        ])),

    % Loading state
    define_pattern(loading_state, [message],
        layout(center, [], [
            layout(stack, [spacing(12), align(center)], [
                component(spinner, [size(40)]),
                when(message, component(text, [content(message), style(muted)]))
            ])
        ])),

    format('Registered ~w built-in patterns~n', [16]).

% ============================================================================
% TESTING
% ============================================================================

test_ui_patterns :-
    format('~n=== UI Patterns Tests ===~n~n'),

    % Setup: clear and register patterns
    clear_patterns,
    register_builtin_patterns,

    % Test 1: Pattern definition
    format('Test 1: Custom pattern definition...~n'),
    define_pattern(test_pattern, [title, body],
        container(card, [], [
            component(heading, [content(title)]),
            component(text, [content(body)])
        ])),
    (   get_pattern(test_pattern, _)
    ->  format('  PASS: Pattern defined~n')
    ;   format('  FAIL: Pattern not found~n')
    ),

    % Test 2: Pattern expansion
    format('~nTest 2: Pattern expansion...~n'),
    expand_pattern(test_pattern, [title("Hello"), body("World")], Expanded),
    format('  Expanded: ~w~n', [Expanded]),
    (   Expanded = container(card, [], [
            component(heading, [content("Hello")]),
            component(text, [content("World")])
        ])
    ->  format('  PASS: Substitution correct~n')
    ;   format('  FAIL: Substitution incorrect~n')
    ),

    % Test 3: List patterns
    format('~nTest 3: List patterns...~n'),
    list_patterns(Patterns),
    length(Patterns, Count),
    format('  Found ~w patterns~n', [Count]),
    (   Count >= 16
    ->  format('  PASS: Built-in patterns registered~n')
    ;   format('  FAIL: Expected at least 16 patterns~n')
    ),

    % Test 4: Nested pattern expansion
    format('~nTest 4: Nested pattern expansion (use_pattern in template)...~n'),
    UI = use_pattern(login_form, [on_submit(do_login), on_forgot(forgot_password)]),
    expand_patterns(UI, ExpandedLogin),
    (   ExpandedLogin = container(card, _, _)
    ->  format('  PASS: Login form expanded~n'),
        format('  Result: ~w~n', [ExpandedLogin])
    ;   format('  FAIL: Login form not expanded correctly~n')
    ),

    % Test 5: Expand patterns in layout
    format('~nTest 5: Expand patterns in layout children...~n'),
    LayoutUI = layout(stack, [spacing(16)], [
        component(heading, [content("Welcome")]),
        use_pattern(form_field, [label("Name"), type(text), binding(name), placeholder(""), error('')])
    ]),
    expand_patterns(LayoutUI, ExpandedLayout),
    (   ExpandedLayout = layout(stack, _, [_, layout(stack, _, _)])
    ->  format('  PASS: Pattern in layout expanded~n')
    ;   format('  FAIL: Pattern in layout not expanded~n'),
        format('  Got: ~w~n', [ExpandedLayout])
    ),

    % Test 6: form_field pattern
    format('~nTest 6: Form field pattern...~n'),
    expand_pattern(form_field, [
        label("Email"),
        type(email),
        binding(user_email),
        placeholder("Enter email"),
        error('')
    ], FormField),
    format('  Form field: ~w~n', [FormField]),
    (   FormField = layout(stack, _, [component(label, _), component(text_input, _)|_])
    ->  format('  PASS: Form field structure correct~n')
    ;   format('  FAIL: Form field structure incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('UI Patterns module loaded~n'),
    register_builtin_patterns
), now).
